use std::env;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use axum::{
    routing::{get, post},
    Router, Json, Extension,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
mod cluster;
mod gguf;
mod inference;
mod model;
mod q8;
mod speculative;
mod tokenizer;



const CONTEXT_LIMIT_ENV: &str = "NANOCAMELID_CONTEXT_LIMIT";
const PREFILL_BATCH_ENV: &str = "NANOCAMELID_PREFILL_BATCH";
const RAYON_THREADS_ENV: &str = "NANOCAMELID_RAYON_THREADS";
const WORKER_CORES_ENV: &str = "NANOCAMELID_WORKER_CORES";
const DEFAULT_RAYON_THREADS: usize = 4;
const DEFAULT_Q4_PREFILL_BATCH: usize = 16;

struct ModelContext {
    config: model::LlamaModelConfig,
    weights: model::LlamaWeights,
    tokenizer: tokenizer::Tokenizer,
    runtime_options: inference::LlamaRuntimeOptions,
    session: ChatSession,
}

struct ChatSession {
    cache: inference::LlamaKvCache,
    ws: inference::LlamaWorkspace,
    batch_ws: inference::LlamaBatchWorkspace,
    context_tokens: Vec<u32>,
    pos: usize,
}

impl ChatSession {
    fn new(config: &model::LlamaModelConfig) -> Self {
        Self {
            cache: inference::LlamaKvCache::new(
                config.block_count,
                config.context_length,
                config.kv_width,
            ),
            ws: inference::LlamaWorkspace::new(config),
            batch_ws: inference::LlamaBatchWorkspace::new(config, prefill_batch_size()),
            context_tokens: Vec::new(),
            pos: 0,
        }
    }
}

struct AppState {
    context: Mutex<Option<ModelContext>>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessageJson>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessageJson {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: CompletionUsage,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChoice {
    index: u32,
    message: ChatMessageJson,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct CompletionUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
}

fn uuid_v4_dummy() -> String {
    let rand_val = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", rand_val)
}

fn setup_thread_pool() {
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let worker_core_indices =
        worker_core_indices_from_env().or_else(isolated_cpu_indices_from_sysfs);
    let worker_core_ids = worker_core_indices
        .as_ref()
        .map(|indices| {
            indices
                .iter()
                .filter_map(|&idx| core_ids.get(idx).copied())
                .collect::<Vec<_>>()
        })
        .filter(|ids| !ids.is_empty())
        .unwrap_or_else(|| core_ids.clone());
    let default_threads = worker_core_ids.len().clamp(1, DEFAULT_RAYON_THREADS);
    let thread_count = env::var(RAYON_THREADS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default_threads);

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .start_handler(move |thread_idx| {
            if let Some(core_id) = worker_core_ids.get(thread_idx % worker_core_ids.len().max(1)) {
                core_affinity::set_for_current(*core_id);
            }
        })
        .build_global();
}

fn worker_core_indices_from_env() -> Option<Vec<usize>> {
    env::var(WORKER_CORES_ENV)
        .ok()
        .and_then(|value| parse_cpu_list(&value))
}

fn isolated_cpu_indices_from_sysfs() -> Option<Vec<usize>> {
    std::fs::read_to_string("/sys/devices/system/cpu/isolated")
        .ok()
        .and_then(|value| parse_cpu_list(&value))
}

fn parse_cpu_list(value: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value.trim().split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = part.split_once('-') {
            let start = start.trim().parse::<usize>().ok()?;
            let end = end.trim().parse::<usize>().ok()?;
            if start > end {
                return None;
            }
            cpus.extend(start..=end);
        } else {
            cpus.push(part.trim().parse::<usize>().ok()?);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    (!cpus.is_empty()).then_some(cpus)
}

fn prefill_batch_size() -> usize {
    env::var(PREFILL_BATCH_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_Q4_PREFILL_BATCH)
}

fn rope_scaling_from_gguf(gguf: &gguf::GgufFile) -> inference::RopeScaling {
    let prefix = gguf
        .metadata_string("general.architecture")
        .and_then(model::metadata_prefix_for_arch)
        .unwrap_or("llama");
    inference::RopeScaling {
        factor: gguf.metadata_f32(&format!("{prefix}.rope.scaling.factor")),
        original_context_length: gguf
            .metadata_u32(&format!("{prefix}.rope.scaling.original_context_length"))
            .map(|value| value as f32),
        low_freq_factor: gguf.metadata_f32(&format!("{prefix}.rope.scaling.low_freq_factor")),
        high_freq_factor: gguf.metadata_f32(&format!("{prefix}.rope.scaling.high_freq_factor")),
    }
}

fn apply_context_limit(config: &mut model::LlamaModelConfig) -> Result<(), String> {
    if let Ok(raw_limit) = env::var(CONTEXT_LIMIT_ENV) {
        if let Ok(limit) = raw_limit.parse::<usize>() {
            config.context_length = config.context_length.min(limit);
        }
    }
    Ok(())
}

fn runtime_options_from_gguf(
    gguf: &gguf::GgufFile,
    selector: q8::Q8DotKernelSelector,
) -> inference::LlamaRuntimeOptions {
    inference::LlamaRuntimeOptions {
        q8_selector: selector,
        compute_logits: true,
        rope_scaling: rope_scaling_from_gguf(gguf),
    }
}

fn validate_generation_budget(
    prompt_token_count: usize,
    requested_generation_tokens: usize,
    context_length: usize,
) -> Result<(), String> {
    if prompt_token_count > context_length {
        return Err(format!(
            "prompt requires {prompt_token_count} tokens but model context length is {context_length}"
        ));
    }

    let remaining_tokens = context_length - prompt_token_count;
    if requested_generation_tokens > remaining_tokens {
        return Err(format!(
            "prompt uses {prompt_token_count} of {context_length} context tokens; requested {requested_generation_tokens} generation tokens but only {remaining_tokens} remain"
        ));
    }

    Ok(())
}

fn shared_token_prefix_len(lhs: &[u32], rhs: &[u32]) -> usize {
    lhs.iter()
        .zip(rhs)
        .take_while(|(left, right)| left == right)
        .count()
}

struct PrefillTokenState<'a> {
    cache: &'a mut inference::LlamaKvCache,
    ws: &'a mut inference::LlamaWorkspace,
    batch_ws: Option<&'a mut inference::LlamaBatchWorkspace>,
    context_tokens: &'a mut Vec<u32>,
    pos: &'a mut usize,
}

fn prefill_tokens(
    tokens: &[u32],
    config: &model::LlamaModelConfig,
    weights: &model::LlamaWeights,
    mut state: PrefillTokenState<'_>,
    runtime_options: inference::LlamaRuntimeOptions,
) {
    let batch_size = state
        .batch_ws
        .as_ref()
        .map(|ws| ws.max_batch)
        .unwrap_or_else(prefill_batch_size);
    if batch_size <= 1 {
        for &token in tokens {
            inference::prefill_pass(
                token as usize,
                *state.pos,
                config,
                weights,
                state.cache,
                state.ws,
                runtime_options,
            );
            state.context_tokens.push(token);
            *state.pos += 1;
        }
        return;
    }

    for chunk in tokens.chunks(batch_size) {
        if chunk.len() == 1 {
            let token = chunk[0];
            inference::prefill_pass(
                token as usize,
                *state.pos,
                config,
                weights,
                state.cache,
                state.ws,
                runtime_options,
            );
        } else if let Some(batch_ws) = state.batch_ws.as_deref_mut() {
            inference::prefill_pass_batch(
                chunk,
                *state.pos,
                config,
                weights,
                state.cache,
                batch_ws,
                runtime_options,
            );
        } else {
            for &token in chunk {
                inference::prefill_pass(
                    token as usize,
                    *state.pos,
                    config,
                    weights,
                    state.cache,
                    state.ws,
                    runtime_options,
                );
                state.context_tokens.push(token);
                *state.pos += 1;
            }
            continue;
        }
        state.context_tokens.extend_from_slice(chunk);
        *state.pos += chunk.len();
    }
}

fn is_generation_stop_token(special: &tokenizer::SpecialTokens, token_id: u32) -> bool {
    special.eog.contains(&token_id)
}

async fn health(Extension(state): Extension<Arc<AppState>>) -> Json<serde_json::Value> {
    let ctx_guard = state.context.lock().await;
    let loaded_now = ctx_guard.is_some();
    Json(serde_json::json!({
        "ok": true,
        "engine": "nanocamelid-pi",
        "loaded_now": loaded_now,
        "generation_ready": loaded_now,
    }))
}

async fn capabilities() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "engine": "nanocamelid-pi",
        "gguf_metadata": true,
        "tensor_loading": true,
        "tokenization": true,
        "inference": true,
        "streaming": false,
    }))
}

async fn chat_completions(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let mut ctx_guard = state.context.lock().await;
    let ctx = match ctx_guard.as_mut() {
        Some(c) => c,
        None => {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "No model loaded on this server.".to_owned(),
                        error_type: "invalid_request_error",
                        code: "no_model_loaded",
                    }
                })
            ));
        }
    };

    let chat_messages: Vec<tokenizer::ChatMessage> = req.messages
        .iter()
        .map(|msg| tokenizer::ChatMessage {
            role: &msg.role,
            content: &msg.content,
        })
        .collect();

    let rendered = ctx.tokenizer.render_chat_prompt(&chat_messages);

    let prompt_tokens = match ctx.tokenizer.encode(
        &rendered.text,
        rendered.add_special,
        rendered.parse_special,
    ) {
        Ok(t) => t,
        Err(e) => {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Tokenization failed: {e}"),
                        error_type: "invalid_request_error",
                        code: "tokenization_failed",
                    }
                })
            ));
        }
    };

    if prompt_tokens.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "Tokenized prompt is empty.".to_owned(),
                    error_type: "invalid_request_error",
                    code: "empty_prompt",
                }
            })
        ));
    }

    let temp = req.temperature.unwrap_or(0.0);
    let max_tokens = req.max_tokens.unwrap_or(256);

    if let Err(err) = validate_generation_budget(prompt_tokens.len(), max_tokens, ctx.config.context_length) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: err,
                    error_type: "invalid_request_error",
                    code: "context_length_exceeded",
                }
            })
        ));
    }

    let shared_prefix = shared_token_prefix_len(&ctx.session.context_tokens, &prompt_tokens);
    if shared_prefix < ctx.session.context_tokens.len() {
        ctx.session.context_tokens.truncate(shared_prefix);
        ctx.session.pos = shared_prefix;
    }
    let new_prompt_tokens = &prompt_tokens[shared_prefix..];

    if let Some((&last_token, prefix_tokens)) = new_prompt_tokens.split_last() {
        prefill_tokens(
            prefix_tokens,
            &ctx.config,
            &ctx.weights,
            PrefillTokenState {
                cache: &mut ctx.session.cache,
                ws: &mut ctx.session.ws,
                batch_ws: Some(&mut ctx.session.batch_ws),
                context_tokens: &mut ctx.session.context_tokens,
                pos: &mut ctx.session.pos,
            },
            ctx.runtime_options,
        );
        inference::forward_pass(
            last_token as usize,
            ctx.session.pos,
            &ctx.config,
            &ctx.weights,
            &mut ctx.session.cache,
            &mut ctx.session.ws,
            ctx.runtime_options,
        );
        ctx.session.context_tokens.push(last_token);
        ctx.session.pos += 1;
    }

    let mut generated_tokens = Vec::new();

    loop {
        let next_token = inference::sample_logits(&ctx.session.ws.logits, temp);

        if is_generation_stop_token(&ctx.tokenizer.special, next_token as u32)
            || ctx.session.pos >= ctx.config.context_length
            || generated_tokens.len() >= max_tokens
        {
            break;
        }

        generated_tokens.push(next_token as u32);

        if generated_tokens.len() >= max_tokens {
            break;
        }

        inference::forward_pass(
            next_token,
            ctx.session.pos,
            &ctx.config,
            &ctx.weights,
            &mut ctx.session.cache,
            &mut ctx.session.ws,
            ctx.runtime_options,
        );

        ctx.session.pos += 1;
    }

    let assistant_text = match ctx.tokenizer.decode(&generated_tokens, true) {
        Ok(t) => t,
        Err(e) => {
            return Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Decoding generated tokens failed: {e}"),
                        error_type: "internal_error",
                        code: "decoding_failed",
                    }
                })
            ));
        }
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid_v4_dummy()),
        object: "chat.completion",
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: "nanocamelid-local".to_owned(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessageJson {
                role: "assistant".to_owned(),
                content: assistant_text,
            },
            finish_reason: "stop",
        }],
        usage: CompletionUsage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: generated_tokens.len(),
            total_tokens: prompt_tokens.len() + generated_tokens.len(),
        },
    };

    Ok(Json(response))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_thread_pool();

    let args: Vec<String> = env::args().collect();
    
    let mut addr_str = "127.0.0.1:8181".to_owned();
    let mut model_path_str: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--addr" => {
                if i + 1 < args.len() {
                    addr_str = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--model" => {
                if i + 1 < args.len() {
                    model_path_str = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    let model_context = if let Some(ref path_str) = model_path_str {
        let model_path = Path::new(path_str);
        println!("🚀 Local Server: Loading GGUF model: {:?}", model_path);
        
        let gguf = gguf::read_file(model_path).map_err(|e| format!("Failed to read GGUF: {e}"))?;
        let mut config = model::LlamaModelConfig::from_gguf(&gguf).map_err(|e| format!("Failed to parse config: {e}"))?;
        apply_context_limit(&mut config)?;
        
        let tokenizer = tokenizer::Tokenizer::from_gguf(&gguf).map_err(|e| format!("Failed to load tokenizer: {e}"))?;
        let weights = model::LlamaWeights::load(model_path, &config, &gguf).map_err(|e| format!("Failed to load weights: {e}"))?;
        
        let selector = q8::Q8DotKernelSelector::from_env();
        let runtime_options = runtime_options_from_gguf(&gguf, selector);
        
        let session = ChatSession::new(&config);
        
        println!("✅ Local Server: GGUF model successfully loaded! Selected dot-product kernel: {}", selector.selected.name());
        Some(ModelContext {
            config,
            weights,
            tokenizer,
            runtime_options,
            session,
        })
    } else {
        println!("⚠️  Local Server: No GGUF model specified. Starting in headless mode.");
        None
    };

    let state = Arc::new(AppState {
        context: Mutex::new(model_context),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/api/capabilities", get(capabilities))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .layer(Extension(state));

    let addr: SocketAddr = addr_str.parse().map_err(|e| format!("Failed to parse address '{addr_str}': {e}"))?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("🚀 Local Server: Listening on HTTP Address: http://{}", addr);
    
    axum::serve(listener, app).await?;

    Ok(())
}

#!/usr/bin/env bash
# Automate, monitor, and orchestrate Llama 3 70B multi-node distributed AI clustering.
set -euo pipefail

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_NAME="Meta-Llama-3-70B-Instruct.Q4_0.gguf"
MAC1_MODEL_DIR="/Volumes/SSK Drive/Camelid/models"
MAC2_MODEL_DIR="/Users/timtoole/openclaw-camelid-worker/models"
PI_MODEL_DIR="/mnt/nanocamelid/models"

MAC1_IP="192.168.86.45"
MAC2_IP="169.254.156.89"
PI1_IP="192.168.86.27"
PI2_IP="192.168.86.48"
PI3_IP="192.168.86.39"

MAC2_SSH_KEY="/Users/timtoole/.ssh/openclaw_second_mac_ed25519"
PI_SSH_KEY="/Users/timtoole/.gemini/antigravity/scratch/pi5_tooleman_ed25519"

## Layer Splits (80 layers total)
# Mac 1: 0..32 (32 layers) -> connects to Mac 2 (169.254.156.89:5005)
# Mac 2: 32..64 (32 layers) -> connects to Pi 2 (192.168.86.48:5005)
# Pi 2: 64..80 (16 layers) -> final worker binds on port 5005

# ---------------------------------------------------------
# Usage
# ---------------------------------------------------------
usage() {
  cat <<USAGE
Usage: scripts/pi/llama-70b-cluster.sh [command]

Commands:
  plan              Print the entire 3-node pipeline plan and commands
  status            Monitor the local netcat download progress & speed
  sync-mac          Transfer the 70B GGUF model to the second Mac Mini once downloaded
  start-pi2         SSH and start the final worker on Pi 2 (layers 64..80)
  start-mac2        SSH and start the middle worker on Mac Mini 2 (layers 32..64)
  start-master      Start the coordinator/master on local Mac Mini 1 (layers 0..32)
USAGE
}

# ---------------------------------------------------------
# Commands Implementation
# ---------------------------------------------------------
case "${1:-}" in
  plan)
    echo "========================================================================="
    echo "            NanoCamelid Llama 3 70B 3-Node Distributed Plan"
    echo "========================================================================="
    echo "Model: $MODEL_NAME"
    echo "Context Limit: NANOCAMELID_CLUSTER_CONTEXT_LIMIT=512"
    echo "-------------------------------------------------------------------------"
    echo "1. Final Worker (Pi 2 - $PI2_IP)"
    echo "   Layers: 64..80 (16 layers)"
    echo "   Command:"
    echo "     ssh -i $PI_SSH_KEY tooleman@$PI2_IP \"NANOCAMELID_CLUSTER_CONTEXT_LIMIT=512 /mnt/nanocamelid/target/release/cluster_tcp_smoke worker $PI_MODEL_DIR/$MODEL_NAME 0.0.0.0:5005 64\""
    echo ""
    echo "2. Middle Worker (Mac Mini 2 - $MAC2_IP)"
    echo "   Layers: 32..64 (32 layers) -> connects to Pi 2 ($PI2_IP:5005)"
    echo "   Command:"
    echo "     ssh -i $MAC2_SSH_KEY -o StrictHostKeyChecking=no -o IdentitiesOnly=yes timtoole@$MAC2_IP \"NANOCAMELID_CLUSTER_CONTEXT_LIMIT=512 /Users/timtoole/openclaw-camelid-worker/bin/cluster_tcp_smoke middle-worker $MAC2_MODEL_DIR/$MODEL_NAME 0.0.0.0:5005 $PI2_IP:5005 32 64\""
    echo ""
    echo "3. Coordinator/Master (Mac Mini 1 - Local - $MAC1_IP)"
    echo "   Layers: 0..32 (32 layers) -> connects to Mac Mini 2 ($MAC2_IP:5005)"
    echo "   Command:"
    echo "     NANOCAMELID_CLUSTER_CONTEXT_LIMIT=512 cargo run --release --bin cluster_tcp_smoke -- master-chat \\"
    echo "       \"$MAC1_MODEL_DIR/$MODEL_NAME\" $MAC2_IP:5005 \"Explain quantum computing in one sentence.\" 32 32"
    echo "========================================================================="
    ;;

  status)
    local_file="$MAC1_MODEL_DIR/$MODEL_NAME"
    if [[ ! -f "$local_file" ]]; then
      echo "No local 70B model file found yet at $local_file"
      exit 1
    fi
    size_bytes=$(stat -f %z "$local_file")
    size_gb=$(echo "scale=2; $size_bytes / 1073741824" | bc)
    progress=$(echo "scale=2; ($size_bytes / 39970245088) * 100" | bc)
    echo "Model transfer status:"
    echo "  Location: $local_file"
    echo "  Bytes downloaded: $size_bytes / 39,970,245,088 bytes"
    echo "  GB downloaded: $size_gb / 37.22 GB"
    echo "  Progress: $progress%"
    ;;

  sync-mac)
    local_file="$MAC1_MODEL_DIR/$MODEL_NAME"
    if [[ ! -f "$local_file" ]]; then
      echo "Error: Local 70B model file not found at $local_file"
      exit 1
    fi
    size_bytes=$(stat -f %z "$local_file")
    if (( size_bytes < 39000000000 )); then
      echo "Error: Model file is not fully downloaded yet ($size_bytes bytes)."
      echo "Please wait for 'status' progress to reach 100%."
      exit 1
    fi
    echo "Starting high-speed Thunderbolt transfer to Mac Mini 2 ($MAC2_IP)..."
    rsync -ah --progress "$local_file" -e "ssh -i $MAC2_SSH_KEY -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" timtoole@$MAC2_IP:"$MAC2_MODEL_DIR/$MODEL_NAME"
    echo "Transfer complete!"
    ;;

  start-pi2)
    echo "Starting Pi 2 worker..."
    ssh -i "$PI_SSH_KEY" tooleman@"$PI2_IP" \
      "NANOCAMELID_CLUSTER_CONTEXT_LIMIT=512 /mnt/nanocamelid/target/release/cluster_tcp_smoke worker $PI_MODEL_DIR/$MODEL_NAME 0.0.0.0:5005 64"
    ;;

  start-mac2)
    echo "Starting Mac Mini 2 middle worker..."
    # Ensure Mac 2 has the binary in place or build it. We'll copy from local.
    echo "Deploying latest binary to Mac Mini 2..."
    scp -i "$MAC2_SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes \
      /Volumes/SSK\ Drive/OpenClaw/cargo-targets/global/release/cluster_tcp_smoke \
      timtoole@$MAC2_IP:/Users/timtoole/openclaw-camelid-worker/bin/cluster_tcp_smoke
    
    ssh -i "$MAC2_SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes timtoole@$MAC2_IP \
      "NANOCAMELID_CLUSTER_CONTEXT_LIMIT=512 /Users/timtoole/openclaw-camelid-worker/bin/cluster_tcp_smoke middle-worker $MAC2_MODEL_DIR/$MODEL_NAME 0.0.0.0:5005 $PI2_IP:5005 32 64"
    ;;

  start-master)
    echo "Starting Coordinator/Master on local Mac Mini 1..."
    NANOCAMELID_CLUSTER_CONTEXT_LIMIT=512 cargo run --release --bin cluster_tcp_smoke -- master-chat \
      "$MAC1_MODEL_DIR/$MODEL_NAME" "$MAC2_IP:5005" "Explain quantum computing in one sentence." 32 32
    ;;

  *)
    usage
    ;;
esac

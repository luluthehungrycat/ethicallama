#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
PROMPT="${2:-The quick brown fox jumps over the lazy dog}"
THREADS="${3:-$(nproc)}"

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model-path> [prompt] [threads]"
    exit 1
fi

echo "Benchmarking: $MODEL"
echo "Threads: $THREADS"
echo "Prompt: ${PROMPT:0:50}..."
echo "---"

# Warmup run
ethllama run "$MODEL" --prompt "$PROMPT" --threads "$THREADS" > /dev/null 2>&1

# Timed run (3 iterations)
for i in 1 2 3; do
    echo "Run $i:"
    /usr/bin/time -f "  Elapsed: %e s, CPU: %P, Memory: %M KB" \
        ethllama run "$MODEL" --prompt "$PROMPT" --threads "$THREADS" 2>&1 | tail -1
done

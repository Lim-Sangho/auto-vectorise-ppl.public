#!/bin/bash

SEED=${1:-0}
CUDA=${2:-0}
TIMEOUT=1h
NUM_CHAINS=1

trap "echo 'kill all processes [python main.py ... --cuda ${CUDA} ...' && pkill -f 'python main.py.*--cuda ${CUDA}.*' 2>/dev/null" EXIT

timeout ${TIMEOUT} python main.py --benchmark hmm-ord1  --method seq      --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark hmm-ord1  --method plate    --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark hmm-ord1  --method vmarkov  --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark hmm-ord1  --method discHMM  --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark hmm-ord1  --method manual   --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark hmm-ord1  --method ours     --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA}

timeout ${TIMEOUT} python main.py --benchmark hmm-ord2   --method seq     --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA} --num_batch 50 
timeout ${TIMEOUT} python main.py --benchmark hmm-ord2   --method plate   --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA} --num_batch 50 
timeout ${TIMEOUT} python main.py --benchmark hmm-ord2   --method vmarkov --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA} --num_batch 50 
timeout ${TIMEOUT} python main.py --benchmark hmm-ord2   --method manual  --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA} --num_batch 50 
timeout ${TIMEOUT} python main.py --benchmark hmm-ord2   --method ours    --num_chains ${NUM_CHAINS} --step_size 1e-2 --seed ${SEED} --cuda ${CUDA} --num_batch 50 

timeout ${TIMEOUT} python main.py --benchmark nhmm-train  --method seq    --num_chains ${NUM_CHAINS} --step_size 1e-3 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark nhmm-train  --method plate  --num_chains ${NUM_CHAINS} --step_size 1e-3 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark nhmm-train  --method manual --num_chains ${NUM_CHAINS} --step_size 1e-3 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark nhmm-train  --method ours   --num_chains ${NUM_CHAINS} --step_size 1e-3 --seed ${SEED} --cuda ${CUDA}

timeout ${TIMEOUT} python main.py --benchmark nhmm-stock  --method seq    --num_chains ${NUM_CHAINS} --step_size 1e-3 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark nhmm-stock  --method plate  --num_chains ${NUM_CHAINS} --step_size 1e-3 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark nhmm-stock  --method manual --num_chains ${NUM_CHAINS} --step_size 1e-3 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark nhmm-stock  --method ours   --num_chains ${NUM_CHAINS} --step_size 1e-3 --seed ${SEED} --cuda ${CUDA}

timeout ${TIMEOUT} python main.py --benchmark tcm  --method seq           --num_chains ${NUM_CHAINS} --step_size 1e-4 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark tcm  --method ours          --num_chains ${NUM_CHAINS} --step_size 1e-4 --seed ${SEED} --cuda ${CUDA}

timeout ${TIMEOUT} python main.py --benchmark arm  --method seq           --num_chains ${NUM_CHAINS} --step_size 1e-4 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark arm  --method vmarkov       --num_chains ${NUM_CHAINS} --step_size 1e-4 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark arm  --method manual        --num_chains ${NUM_CHAINS} --step_size 1e-4 --seed ${SEED} --cuda ${CUDA}
timeout ${TIMEOUT} python main.py --benchmark arm  --method ours          --num_chains ${NUM_CHAINS} --step_size 1e-4 --seed ${SEED} --cuda ${CUDA}



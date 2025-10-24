nohup bash -c '{ python main.py --benchmark hmm-ord1 --method all --num_seed 2 --num_batch 10 --lr 1e-2 --cuda 0;
                 python main.py --benchmark arm --method all --num_seed 2 --lr 1e-2 --cuda 0; }' > 0.out &

nohup bash -c '{ python main.py --benchmark hmm-neural --method all --num_seed 2 --num_batch 10 --lr 1e-1 --cuda 1;
                 python main.py --benchmark tcm --method all --num_seed 2 --lr 1e-2 --cuda 1; }' > 1.out &

nohup bash -c '{ python main.py --benchmark hmm-ord2 --method all --num_seed 2 --num_batch 10 --lr 1e-4 --cuda 2;
                 python main.py --benchmark nhmm-stock --method all --num_seed 2 --lr 1e-2 --cuda 2; }' > 2.out &

nohup bash -c '{ python main.py --benchmark dmm --method all --num_seed 2 --num_batch 10 --lr 1e-4 --cuda 3;
                 python main.py --benchmark nhmm-train --method all --num_seed 2 --lr 1e-1 --cuda 3; }' > 3.out &
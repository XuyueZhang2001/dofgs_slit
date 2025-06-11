OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_synthetic_cozy2room.yaml --scene 'synthetic_cozy2room'  --iteration 25000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_synthetic_factory.yaml --scene 'synthetic_factory'  --iteration 25000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_synthetic_pool.yaml --scene 'synthetic_pool'  --iteration 40000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_synthetic_tanabata.yaml --scene 'synthetic_tanabata'  --iteration 40000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_synthetic_wine.yaml --scene 'synthetic_wine'  --iteration 40000 --skip_train


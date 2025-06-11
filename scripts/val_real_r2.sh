
# resolution=2
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_real_cake_res=2.yaml --scene 'real_cake' --iteration 15000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_real_caps_res=2.yaml --scene 'real_caps' --iteration 15000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_real_cisco_res=2.yaml --scene 'real_cisco' --iteration 15000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_real_cupcake_res=2.yaml --scene 'real_cupcake' --iteration 15000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_real_cups_res=2.yaml --scene 'real_cups' --iteration 40000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_real_daisy_res=2.yaml --scene 'real_daisy' --iteration 40000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_real_sausage_res=2.yaml --scene 'real_sausage' --iteration 25000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_real_seal_res=2.yaml --scene 'real_seal' --iteration 25000 --skip_train
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py --config ./config_files/config_real_tools_res=2.yaml --scene 'real_tools' --iteration 25000 --skip_train



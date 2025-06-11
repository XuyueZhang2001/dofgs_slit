
for scene in 'synthetic_cozy2room' 'synthetic_factory' 'synthetic_pool' 'synthetic_tanabata' 'synthetic_wine'; do
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=7 python train.py --config ./config_files/config_${scene}.yaml 
done
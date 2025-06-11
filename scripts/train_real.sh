for scene in 'real_cake' 'real_caps' 'real_cisco' 'real_cupcake' 'real_cups' 'real_daisy' 'real_sausage' 'real_seal' 'real_tools' 'real_coral'; do
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 python train.py --config ./config_files/config_${scene}.yaml 
done


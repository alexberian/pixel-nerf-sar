# /workspace/berian/public/miniconda3/envs/edm/bin/python train/train.py -n srn_car_exp -c conf/exp/srn.conf -D /workspace/data/srncars/cars --gpu_id='7' --nviews='1 2 3' --combine_type='cross_attention'

/workspace/berian/public/miniconda3/envs/edm/bin/python  \
    train/train.py -n srn_car_exp -c conf/exp/srn.conf \
    -D /workspace/data/srncars/cars --gpu_id='7' \
    --nviews='1 2 3' \
    --combine_type='learned_cross_attention' \
    --only_train_view_combiner \

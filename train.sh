# /workspace/berian/public/miniconda3/envs/edm/bin/python train/train.py -n srn_car_exp -c conf/exp/srn.conf -D /workspace/data/srncars/cars --gpu_id='7' --nviews='1 2 3' --combine_type='cross_attention'
if test -f /workspace/berian/public/miniconda3/envs/genvs/bin/python; then
	PYTHONSTR="/workspace/berian/public/miniconda3/envs/genvs/bin/python"
	DATASTR="/workspace/data/srncars/cars"
fi
if test -f /home/berian/miniconda3/envs/seed/bin/python; then
	PYTHONSTR="/home/berian/miniconda3/envs/seed/bin/python"
	DATASTR="/home/berian/Documents/shapenet/cars"
fi

$PYTHONSTR  \
    train/train.py -n srn_car_exp -c conf/exp/srn.conf \
    -D $DATASTR --gpu_id='0' \
    --nviews='1 2 3' \
    --combine_type='relative_pose_self_attention' \
    --lr='0.00001' \
    --resume \
    # --only_train_view_combiner \
    # --combine_type='average' \


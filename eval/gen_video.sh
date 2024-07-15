python gen_video.py -n srn_car --gpu_id='7' --split test -P '64 128' \
 -D /workspace/data/srncars/cars -S 5 --combine_type='relative_pose_self_attention' \
  --ray_batch_size=16384 \
  -c ../conf/exp/srn.conf \


python eval.py --gpu_id='6' -n srn_car -c ../conf/exp/srn.conf -D /workspace/data/srncars/cars -F srn \
--source='64 128' --eval_view_list ../viewlist/srn_eval_views.txt \
--combine_type='relative_pose_self_attention' \
--ray_batch_size=16384

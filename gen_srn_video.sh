# needed to run `pip install --force-reinstall -v "pyhocon==0.3.55"` to fix a parsing error

# set dataset directory
# dataset_dir="/mnt/c/Users/Berian/Documents/Arizona/research/Mahalanobis/genvs/shapenet/cars"
dataset_dir="/home/berian/Documents/shapenet/cars"

# for video generation
python eval/gen_video.py -n srn_car --gpu_id 0 --split test -P '64 104' -D $dataset_dir -S 20

# # for my script
# python eval/gen_srn_predictions.py -n srn_car --gpu_id 0 --split test -P '64 104' -D $dataset_dir -S 1

# for my second script
# python eval/gen_images.py -n srn_car --gpu_id 0 --split test -P '64 104' -D $dataset_dir -S 0
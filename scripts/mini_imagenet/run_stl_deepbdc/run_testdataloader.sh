gpuid=1

DATA_ROOT=/path/mini_imagenet
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill/last_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_pretrain/last_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born1/last_model.tar
#  MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born2/ema_smooth_ssl/ema_140.tar
#  MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born2/ema_distill_nossl/ema_last_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born2/140.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain/best_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_pretrain/smoothsoft/last_model.tar
#MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_pretrain/margin/last_model.tar
MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born2/ema_distill_nossl_margin/ema_last_model.tar
#MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born2/ema_distill_nossl_margin/ema_170.tar
#  MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born1/ema_distill_nossl_margin/ema_150.tar

cd ../../../

echo "============= meta-test 1-shot ============="
python testselfloader.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --n_shot 1 --model_path $MODEL_PATH --test_task_nums 1 --penalty_C 0.1 --reduce_dim 128 --test_n_episode 2000

echo "============= meta-test 5-shot ============="
python testselfloader.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --n_shot 5 --model_path $MODEL_PATH --test_task_nums 1 --penalty_C 2 --reduce_dim 128 --test_n_episode 2000

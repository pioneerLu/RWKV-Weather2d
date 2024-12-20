export CUDA_VISIBLE_DEVICES=1

python train.py --load_model "" \
    --wandb "" --proj_dir out/dummy \
    --data_file /home/rwkv/RWKV-TS/WeatherBench/era5_data/ERA5_merged.nc \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 96 --epoch_steps 100 --epoch_count 1 --epoch_begin 0 --epoch_save 10 \
    --micro_bsz 4 --accumulate_grad_batches 1 --n_layer 2 --n_embd 512 --pre_ffn 0 \
    --lr_init 2e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --freeze_rwkv 0 --exp_name 'baseline'\
    --enable_progress_bar True

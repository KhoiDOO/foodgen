accelerate launch \
    --mixed_precision=fp16 \
    --num_processes=1 \
    --num_machines=1 \
    train_vqganvae.py \
    --config config/vqganvae.yaml \
    trainer_config.folder="./data/src"
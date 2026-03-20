Stable:
    1. 24 batch size, 8 workers, uses ELA, only phase 2 stem uses ELA: Stable, long GPU utilization waveform
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python CNN/train.py \
        --authentic-dir data/synthetic/generated/authentic \
        --forged-dir data/synthetic/generated/forged \
        --target-size 800 600 \
        --batch-size 24 \
        --grad-accum-steps 4 \
        --num-workers 8 \
        --use-amp \
        --amp-dtype float16 \
        --train-augmentation light \
        --prefetch-factor 4 \
        --use-ela \
        --no-compile \
        --loss-type focal \
        --focal-gamma 2.0
    2. Step up, batch size 48, accum 5: Very spiky waveform!

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python CNN/train.py \
        --authentic-dir data/synthetic/generated/authentic \
        --forged-dir data/synthetic/generated/forged \
        --target-size 800 600 \
        --batch-size 48 \
        --grad-accum-steps 5 \
        --num-workers 8 \
        --use-amp \
        --amp-dtype float16 \
        --train-augmentation light \
        --prefetch-factor 4 \
        --use-ela \
        --no-compile \
        --loss-type focal \
        --focal-gamma 2.0
        
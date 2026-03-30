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
    2. Layout-aware run, RGB + ELA + X/Y coordinates: best when class2/class4 depend on field position:
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python CNN/train.py \
        --authentic-dir data/synthetic/generated/authentic \
        --forged-dir data/synthetic/generated/forged \
        --target-size 800 600 \
        --batch-size 32 \
        --grad-accum-steps 12 \
        --num-workers 8 \
        --use-amp \
        --amp-dtype float16 \
        --train-augmentation light \
        --prefetch-factor 5 \
        --use-ela \
        --use-coord-channels \
        --phase1-stem-trainable \
        --loss-type focal \
        --focal-gamma 2.5 \
        --no-compile

Adversarial evaluation:
    1. False-positive stress test: generate degraded authentic cards. These should still predict authentic.
        python data/adversarial_generator.py \
        --input data/synthetic/generated/authentic \
        --output-dir data/synthetic/generated/adversarial_authentic \
        --count 16

    2. False-negative stress test: generate degraded forged cards. These should still predict as forgeries.
        python data/adversarial_generator.py \
        --input data/synthetic/generated/forged \
        --output-dir data/synthetic/generated/adversarial_forged \
        --count 16

    3. Run single-image inference with the trained 6-channel model and write Grad-CAM artifacts:
        python CNN/inference.py \
        --image data/synthetic/generated/forged/class1_0001__src__NCC-2367-2168.jpg \
        --checkpoint results/best_model_phase2.pth \
        --output-dir results/inference

    4. Batch inference over adversarial authentic samples:
        mkdir -p results/inference_adversarial_authentic
        for image in data/synthetic/generated/adversarial_authentic/*; do \
            python CNN/inference.py \
            --image "$image" \
            --checkpoint results/best_model_phase2.pth \
            --output-dir results/inference_adversarial_authentic; \
        done

    5. Batch inference over adversarial forged samples:
        mkdir -p results/inference_adversarial_forged
        for image in data/synthetic/generated/adversarial_forged/*; do \
            python CNN/inference.py \
            --image "$image" \
            --checkpoint results/best_model_phase2.pth \
            --output-dir results/inference_adversarial_forged; \
        done

    6. Inspect one generated adversarial image with Grad-CAM:
        python CNN/inference.py \
        --image data/synthetic/generated/adversarial_authentic/NCC-1602-6301__adv_01__glare_jpeg.png \
        --checkpoint results/best_model_phase2.pth \
        --output-dir results/inference_adversarial_authentic

    7. Step up, batch size 48, accum 5: Very spiky waveform!

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

Adversarial Training (30% Rule / curriculum):
    This mode integrates adversarial attacks directly into the training pipeline, but
    only applies them to a fraction of the training samples. By default, 70% of images
    stay clean and 30% receive an adversarial recipe on-the-fly.
    
    Why this is the right balance:
    - Clean samples preserve the baseline geometry, typography, and localized ELA signal.
    - Adversarial samples regularize the model against glare, shadow, blur, noise, and JPEG.
    - The ELA channel is not wiped out by hammering every image with destructive compression.
    - The model learns both what a clean card looks like and how to stay robust under attack.
    
    Defaults in train.py:
    - --adversarial-prob 0.3
    - --adversarial-jpeg-min-quality 40
    - --adversarial-jpeg-max-quality 60
    
    Fresh adversarial run (recommended):
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
        --train-augmentation adversarial \
        --adversarial-prob 0.3 \
        --adversarial-jpeg-min-quality 40 \
        --adversarial-jpeg-max-quality 60 \
        --use-ela \
        --use-coord-channels \
        --phase1-stem-trainable \
        --prefetch-factor 4 \
        --no-compile \
        --loss-type focal \
        --focal-gamma 2.0
    
    If the model is still too brittle, first increase clean-data share instead of making
    attacks stronger:
    - try --adversarial-prob 0.2
    - only then consider lowering JPEG quality or increasing other attack severity

Batch Adversarial Robustness Evaluation:
    After adversarial training (Phase 3), evaluate the model's real-world robustness.
    
    1. Generate adversarial authentic samples (false-positive stress test):
        python data/adversarial_generator.py \
        --input data/synthetic/generated/authentic \
        --output-dir data/synthetic/generated/adversarial_authentic \
        --count 16

    2. Generate adversarial forged samples (false-negative stress test):
        python data/adversarial_generator.py \
        --input data/synthetic/generated/forged \
        --output-dir data/synthetic/generated/adversarial_forged \
        --count 16

    3. Batch evaluate model performance on both sets, generate classification report + confusion matrix:
        python CNN/batch_evaluate.py \
        --input-dir data/synthetic/generated/adversarial_authentic \
        --input-dir data/synthetic/generated/adversarial_forged \
        --checkpoint results/best_model_phase2.pth \
        --output-dir results/adversarial_eval_phase3
        
        Output structure:
        - results.csv                       per-image predictions, attack type, confidence, correct/wrong flag
        - classification_report.txt         sklearn report (precision, recall, F1 per class)
        - confusion_matrix.png              ConfusionMatrixDisplay visualization
        - mispredictions/                   Grad-CAM overlays for failures only (saves disk space)
        
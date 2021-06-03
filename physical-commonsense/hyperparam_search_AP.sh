for lr in "0.00001" "0.000001" "0.0000001"; do
    for wm in "0.1" "0.3"; do
        for dropout in "0.0" "0.05"; do
            for act in "relu" "gelu"; do
                for task in "situated-AP"; do
                    python -m pc.finetune_clip --dev \
                                               --task ${task} \
                                               --epochs 5 \
                                               --lr ${lr} \
                                               --warmup-ratio ${wm} \
                                               --dropout ${dropout} \
                                               --activation ${act}
                done
            done
        done
    done
done
                    

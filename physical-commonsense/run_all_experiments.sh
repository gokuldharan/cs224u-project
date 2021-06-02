python -m pc.finetune_clip --text-only --task "situated-OP" --epochs 15
python -m pc.finetune_clip --text-only --task "situated-OA" --epochs 15
python -m pc.finetune_clip --text-only --task "situated-AP" --epochs 6
python -m pc.finetune_clip --task "situated-OP" --epochs 15
python -m pc.finetune_clip --task "situated-OA" --epochs 15
python -m pc.finetune_clip --task "situated-AP" --epochs 6
python -m pc.finetune_clip --gan-imgs --task "situated-OP" --epochs 15
python -m pc.finetune_clip --gan-imgs --task "situated-OA" --epochs 15
python -m pc.finetune_clip --gan-imgs --task "situated-AP" --epochs 6

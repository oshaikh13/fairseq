TOTAL_NUM_UPDATES=2036  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=122      # 6 percent of the number of updates
LR=1e-06                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=16        # Batch size.
# LM_PATH=checkpoint_best.pt
LM_PATH=best_lm_checkpoint.pt

python train.py CoLA-bin/ \
    --restore-file $LM_PATH \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 3584 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --arch transformer_lm_wmt \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --lr $LR \
    --dropout 0.15 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" \
    --clip-norm 0.0 \
    --warmup-updates $WARMUP_UPDATES \
    --max-epoch 9999 \
    --find-unused-parameters \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --fp16 \
    --best-checkpoint-metric matthew --maximize-best-checkpoint-metric;

#     --update-freq 16 \
#  --lr-scheduler inverse_sqrt \

# python train.py CoLA-bin/ \
#     --max-sentences $MAX_SENTENCES \
#     --max-tokens 3584 \
#     --task sentence_prediction \
#     --reset-optimizer --reset-dataloader --reset-meters \
#     --arch transformer_lm_wmt \
#     --criterion sentence_prediction \
#     --keep-last-epochs 3 \
#     --num-classes $NUM_CLASSES \
#     --lr $LR --lr-scheduler inverse_sqrt \
#     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" \
#     --clip-norm 0.0 \
#     --warmup-updates $WARMUP_UPDATES \
#     --max-epoch 9999 \
#     --find-unused-parameters \
#     --best-checkpoint-metric matthew --maximize-best-checkpoint-metric;
    

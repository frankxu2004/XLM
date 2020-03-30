OUTPATH=data/processed/XLM_en/30k

CUDA_VISIBLE_DEVICES=8 python train.py   \
--exp_name xlm_en   \
--dump_path ./dumped   \
--data_path $OUTPATH   \
--lgs 'en'   \
--clm_steps ''   \
--mlm_steps 'en'   \
--emb_dim 1024   \
--n_layers 12   \
--n_heads 16   \
--dropout 0.1   \
--attention_dropout 0.1   \
--gelu_activation true   \
--batch_size 8   \
--bptt 256   \
--optimizer adam_inverse_sqrt,lr=0.00010,warmup_updates=30000,beta1=0.9,beta2=0.999,weight_decay=0.01,eps=0.000001   \
--epoch_size 300000   \
--max_epoch 100000   \
--validation_metrics _valid_mlm_ppl   \
--stopping_criterion _valid_mlm_ppl,25   \
--fp16 true   \
--word_mask_keep_rand '0.8,0.1,0.1'   \
--word_pred '0.15'

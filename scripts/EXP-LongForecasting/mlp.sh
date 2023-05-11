if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=MLP
rtPath=~/Work/TSPred/
for data_file in ETTh1 ETTh2 ETTm1 ETTm2
do
for pred_len in 96 192 336 729
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path $rtPath/dataset/ETT-small \
  --data_path $data_file.csv \
  --model_id $data_file'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_file \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 \
  >logs/LongForecasting/$model_name'_I_'$data_file'_'$seq_len'_'$pred_len.log 
done
done
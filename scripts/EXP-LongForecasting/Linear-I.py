import os

if not os.path.isdir("./logs"):
    os.mkdir("./logs")

if not os.path.isdir("./logs/LongForecasting"):
    os.mkdir("./logs/LongForecasting")

seq_len = 336
model_name = "NLinear"
pred_lens = [96, 192, 336, 729]

for pred_len in pred_lens:
    python_command = "python -u run_longExp.py " + \
        "--is_training 1 " + \
        "--root_path ./dataset/ " + \
        "--data_path electricity.csv " + \
        "--model_id Electricity_" + str(seq_len) + "_" + str(pred_len) + " " + \
        "--model " + model_name + " " + \
        "--data custom " + \
        "--features M " + \
        "--seq_len " + str(seq_len) + " " + \
        "--pred_len " + str(pred_len) + " " + \
        "--enc_in 321 " + \
        "--des 'Exp' " + \
        "--itr 1 --batch_size 16 --learning_rate 0.005 --individual >logs/LongForecasting/" + \
        model_name + "_I_electricity_" + str(seq_len) + "_" + str(pred_len) + ".log"

    os.system(python_command)

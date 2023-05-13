import pandas as pd




df= pd.read_csv(r'C:\Programming\Python\LTSF-Linear\dataset\all_six_datasets\bitcoin\from220330_1200to230330_1312_1h.csv')

# covert before LTSF
# 1. time을 LTSF 타입으로 바꿔야 함

# Convert the Unix timestamp (in milliseconds) to datetime format
df['date'] = pd.to_datetime(df['Kline open time'], unit='ms')
# Format the datetime to the desired string format
df['date'] = df['date'].dt.strftime('%Y-%m-%d %I:%M:%S %p')
df['Kline open time'] = df['date']
# 판다스 csv 저장



# 2. KClose time 열 삭제
# 3. Quote asset volume 열 삭제
# 4. Taker buy quote asset volume 열 삭제
# 5. Ignore 열 삭제
# 6. Date 열 삭제


df.to_csv(r'C:\Programming\Python\LTSF-Linear\dataset\all_six_datasets\bitcoin\from220330_1200to230330_1312_1h_2.csv', index=False)

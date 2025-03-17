import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_path = "./data/taxi_fare_data.csv"
df = pd.read_csv(data_path)

# 결측치 제거
df = df.dropna()

# 이상치 제거: 음수 승객 수 제거
df = df[df["passenger_count"] > 0]

# 시각화 - 승객 수 vs. 운임 요금 (regplot)
plt.figure(figsize=(15,10))
sns.regplot(x=df["passenger_count"], y=df["fare_amount"], scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})

# 그래프 설정
plt.xlabel("Passenger Count")
plt.ylabel("Fare Amount")
plt.title("Taxi Fare vs. Passenger Count (Regression Plot)")
plt.grid(True)

# 그래프 출력
plt.show()
plt.savefig("q7.png") # 탑승 인원대비 운임요금, regplot




# 거리차이 계산
df["거리차이"] = np.sqrt((df["dropoff_latitude"] - df["pickup_latitude"])**2 + (df["dropoff_longitude"] - df["pickup_longitude"])**2)

# 거리차이 vs. 운임 요금 (regplot)
plt.figure(figsize=(10, 6))
sns.regplot(x=df["거리차이"], y=df["fare_amount"], scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})

# 그래프 설정
plt.xlabel("Distance Difference")
plt.ylabel("Fare Amount")
plt.title("Taxi Fare vs. Distance Difference (Regression Plot)")
plt.grid(True)

# 그래프 출력
plt.show()
plt.savefig("q7_regplot.png") # 탑승 거리 대비 운임요금, regplot
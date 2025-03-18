import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (단위: km)

    # 위도와 경도를 라디안(radian)으로 변환
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # 하버사인 공식 적용
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 최종 거리 계산
    distance = R * c
    return distance  # 단위: km


data_path = "./data/train.csv"
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
plt.savefig("q7_0318_PassengerVSFareamount.png") # 탑승 인원대비 운임요금, regplot





# 거리차이 계산
df["Distance"] = df.apply(lambda row: haversine(row["dropoff_latitude"], row["dropoff_longitude"],
                                                 row["pickup_latitude"], row["pickup_longitude"]), axis=1)

print(df["Distance"])

# 거리차이 vs. 운임 요금 (regplot)
plt.figure(figsize=(10, 6))
sns.regplot(x=df["Distance"], y=df["fare_amount"], scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})

# 그래프 설정
plt.xlabel("Distance Difference")
plt.ylabel("Fare Amount")
plt.title("Taxi Fare vs. Distance Difference (Regression Plot)")
plt.grid(True)

# 그래프 출력
plt.show()
plt.savefig("q7_0318_FareVSDistance.png") # 탑승 거리 대비 운임요금, regplot
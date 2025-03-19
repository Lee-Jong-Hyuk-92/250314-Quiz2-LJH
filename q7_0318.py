import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 하버사인 거리 계산 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (단위: km)
    phi1, phi2 = map(math.radians, [lat1, lat2])
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance  # 단위: km

# 데이터 로드
data_path = "./data/train.csv"
df = pd.read_csv(data_path)

# 결측치 제거
df = df.dropna()

# 이상치 제거: 음수 승객 수 제거
df = df[df["passenger_count"] > 0]

# 거리차이 계산 (apply 사용)
df["Distance"] = df.apply(lambda row: haversine(row["dropoff_latitude"], row["dropoff_longitude"],
                                                 row["pickup_latitude"], row["pickup_longitude"]), axis=1)

# 사이킷런 선형 회귀 분석**
X = df[["Distance"]]  # 독립 변수
y = df["fare_amount"]  # 종속 변수

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 회귀 계수 및 절편 출력
print(f"기울기 (Slope): {model.coef_[0]}")
print(f"절편 (Intercept): {model.intercept_}")

# 예측값 생성
df["Predicted_Fare"] = model.predict(X)

X_ols = sm.add_constant(X)  # 절편 추가
ols_model = sm.OLS(y, X_ols).fit()

# OLS 결과 출력
print(ols_model.summary())

# 거리 대비 운임 요금 선형 회귀 그래프 출력**
plt.figure(figsize=(10, 6))

# 실제 데이터 점
sns.scatterplot(x=df["Distance"], y=df["fare_amount"], alpha=0.5, label="Actual Data")

# 회귀선 (사이킷런)
plt.plot(df["Distance"], df["Predicted_Fare"], color="red", label="Linear Regression (Sklearn)")

# 그래프 설정
plt.xlabel("Distance (km)")
plt.ylabel("Fare Amount ($)")
plt.title("Taxi Fare vs. Distance (Linear Regression)")
plt.legend()
plt.grid(True)

# 그래프 출력
plt.savefig("q7_0318_FareVSDistance_Regression.png")  # 결과 저장
plt.show()


print(df[df["Distance"]>50])
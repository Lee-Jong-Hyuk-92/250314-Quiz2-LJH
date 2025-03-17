import pandas as pd
import matplotlib.pyplot as plt

# 한글 요일 매핑
WEEK_KOR = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}


# 데이터 로드
data_path ="data/electronic.csv"
df = pd.read_csv(data_path)

# 날짜 변환
df["DateTime"] = pd.to_datetime(df["DateTime"])

# 요일 추가
df["요일"] = df["DateTime"].dt.dayofweek.map(WEEK_KOR)

# 요일별 평균 전력 소비량 계산
s_mean = df.groupby("요일")["Consumption"].mean()

# 전력 소비량 시계열 그래프 출력
plt.figure(figsize=(12, 6))
plt.plot(df["DateTime"], df["Consumption"], label="Power Consumption")

plt.xlabel("Date Time")
plt.ylabel("Consumption")
plt.title("delta Power Consumption per hour")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# 그래프 출력
plt.show()
plt.savefig("q9_wattage.png")








# DateTime 컬럼을 datetime 형식으로 변환
df['DateTime'] = pd.to_datetime(df['DateTime'])

# 시간만 추출하여 새로운 컬럼 생성
df['Hour'] = df['DateTime'].dt.hour

# 시간별 평균 전력 소비량 계산
hourly_avg = df.groupby('Hour')['Consumption'].mean()

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linestyle='-')
plt.xlabel("Hour of Day")
plt.ylabel("Average Electricity Consumption")
plt.title("Hourly Average Electricity Consumption")
plt.grid(True)
plt.xticks(range(24))  # 0~23시간 설정

# 그래프 출력
plt.show()
plt.savefig("q9_hour_consumption.png")
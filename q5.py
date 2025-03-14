import numpy as np
import pandas as pd

df = pd.read_csv("./data/taxi_fare_data.csv", quoting=3)

pickup_datetime = df['pickup_datetime'] 

year_date = []
time = []

for data in pickup_datetime :
    try:
        date_time_split = data.split(" ")  # 공백을 기준으로 날짜와 시간 분리
        year_date.append(date_time_split[0])  # YYYY-MM-DD 형식의 날짜
        time.append(date_time_split[1])  # HH:MM:SS 형식의 시간
    except:
        year_date.append(None)
        time.append(None)

# 연월일 변수에서 각각의 '연도', '월', '일'을 추출하여 years, months, days 변수에 넣어줍니다.
years = []
months = []
days = []

for data in year_date:
    try:
        y, m, d = data.split("-")  # YYYY-MM-DD를 "-" 기준으로 분리
        years.append(int(y))
        months.append(int(m))
        days.append(int(d))
    except:
        years.append(None)
        months.append(None)
        days.append(None)

#시간만 따로 int의 형태로 추출합니다.
hours = []

for t in time:
    try:
        h, _, _ = t.split(":")  # HH:MM:SS를 ":" 기준으로 분리
        hours.append(int(h))
    except:
        hours.append(None)

#각 변수의 상위 10개씩만 출력해서 추출이 제대로 되었는지 확인해봅시다.
print(years[:10])
print(months[:10])
print(days[:10])
print(hours[:10])
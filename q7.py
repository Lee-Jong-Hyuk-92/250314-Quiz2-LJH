import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "./data/taxi_fare_data.csv" # 택시 요금 데이터 경로 저장

def load_csv(path):
    data_frame = pd.read_csv(path)
    return data_frame   # data_frame에 택시요금 데이터 저장

def del_missing(df):
    del_un_df = df.drop(['Unnamed: 0'], axis='columns') # 어떤(df) 프레임에서 'Unnamed: 0'열 지우고 del_un_df에 저장
    del_un_id_df = del_un_df.drop(['id'], axis='columns')   # del_un_df에서 'id' 열 del_un_id_df에 저장
    removed_df = del_un_id_df.dropna()  # del_un_id_df에 NaN 있으면 없애고 removed_df에 저장
    return removed_df

def get_negative_index(list_data):
    neg_idx = []
    for i, value in enumerate(list_data):
        if value < 0:   # 값이 음수면
            neg_idx.append(list_data.index[i])  # 초기화한 neg_idx에 값이 음수인 index 추가
    return neg_idx

def outlier_index():
    idx_fare_amount = get_negative_index(fare_amount)   # idx_fare_amount에다가 'fare_amount'열 값을 집어넣고 그 값들이 음수인 index 저장
    idx_passenger_count = get_negative_index(passenger_count)   # idx_passenger_count에다가 'passenger_count'열 집어넣고 그 값이 음수인 index 저장
    
    idx_zero_distance = []
    idx = [i for i in range(len(passenger_count))]
    zipped = zip(idx, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
    
    for i, x, y, _x, _y in zipped:
        if (x == _x) and (y == _y): # 탑승 x와 하차 x가 같고, 탑승 y와 하차 y가 같으면
            idx_zero_distance.append(i) # 초기화한 idx_zero_distance에 index 넘버 추가
    
    total_index4remove = list(set(idx_fare_amount+idx_passenger_count+idx_zero_distance))   # 지금까지 구한 운임요금이상, 고객수이상, 거리0들의 index를 모두 합쳐서 하나의 리스트로 만들기
    return total_index4remove

# 인덱스를 기반으로 DataFrame 내의 데이터를 제거하고, 제거된 DataFrame을 반환하는 함수를 만듭니다.
def remove_outlier(dataframe, list_idx):
    return dataframe.drop(list_idx)

# load_csv 함수를 사용하여 데이터를 불러와 df에 저장합니다.
df = load_csv(DATA_PATH)

# 1-1. del_missing 함수로 df의 결측치을 처리하여 df에 덮어씌웁니다.
df = del_missing(df) # df는 바로 위의 df에서 id열과 Unnamed:0열을 제거한 상태 removed_df가 리턴되어 df에 저장

# 불러온 DataFrame의 각 인덱스의 값들을 변수로 저장합니다.
fare_amount = df['fare_amount']
passenger_count = df['passenger_count']
pickup_longitude = df['pickup_longitude']
pickup_latitude = df['pickup_latitude']
dropoff_longitude = df['dropoff_longitude']
dropoff_latitude = df['dropoff_latitude']

# 1-2. remove_outlier()을 사용하여 이상치를 제거합니다.
# remove_outlier()가 어떤 인자들을 받는지 확인하세요.
remove_index = outlier_index()
df = remove_outlier(df,remove_index)

# 2. df.corr()을 사용하여 상관 계수 값 계산
corr_df = df.drop(columns=['pickup_datetime']).corr()

# seaborn을 사용하여 heatmap 출력
plt.figure(figsize=(15,10))
sns.heatmap(corr_df, annot=True, cmap='PuBu')
plt.savefig("plot.png")
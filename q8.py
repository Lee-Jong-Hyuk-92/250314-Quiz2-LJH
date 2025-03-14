import pandas as pd

WEEK_KOR = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}


def load_csv(path: str) -> pd.DataFrame:
    """pandas를 이용하여 path의 데이터를 DataFrame의 형태로 반환합니다."""
    df = pd.read_csv(path)
    return df


def cvt_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """df의 DateTime 칼럼을 datetime 형태로 변환합니다."""
    df["DateTime"] = pd.to_datetime(df['DateTime'])
    return df


def add_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    """df에 DateTime 칼럼의 요일이 저장된 "요일" 칼럼을 새로 추가합니다."""
    df["요일"] = df["DateTime"].dt.dayofweek  # 0(월) ~ 6(일)
    df["요일"] = df["요일"].map(WEEK_KOR)  # 한글 요일 변환
    return df


def get_mean_consumption(df: pd.DataFrame) -> pd.Series:
    """df의 요일별 전력 소비량의 평균을 구하여 반환합니다."""
    series_mean = df.groupby("요일")["Consumption"].mean()
    return series_mean


def main():
    data_path = "data/electronic.csv"
    df = load_csv(data_path)


    df = cvt_to_datetime(df)
    print(df)


    df = add_dayofweek(df)
    print(df)

    s_mean = get_mean_consumption(df)
    print(s_mean)

if __name__ == "__main__":
    main()

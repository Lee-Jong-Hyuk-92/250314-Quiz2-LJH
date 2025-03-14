import numpy as np
import pandas as pd

DATA_PATH = "./data/data.csv"


def get_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def add_type(df: pd.DataFrame) -> pd.DataFrame:
    # 지시사항에 따라 df에 Type칼럼을 추가하고 반환합니다.
    # 나이가 19세 이상이면 “adult”로 설정합니다.
    # 나이가 19세 미만이면서 여성(female)이라면 “girl”로 설정합니다.
    # 나이가 19세 미만이면서 남성(male)이라면 “boy”로 설정합니다.

    condi = [
        (df["Age"] >= 19),  # 19 이상이면
        (df["Age"] < 19) & (df["Sex"] == "female"),  # 19세 미만이면서  여성
        (df["Age"] < 19) & (df["Sex"] == "male")  # 19세 미만이면서 남성
    ]
    
    # 조건마다 뭐라고 붙일지
    type_condi = ["adult", "girl", "boy"]
    
    # Type 열 추가
    df["Type"] = np.select(condi, type_condi)

    return df


def main():
    # 데이터 불러오기
    df = get_data()
    print("추가 전\n", df.head())

    # 1. 새로운 특성 생성
    df_new = add_type(df.copy())
    print("추가 후\n", df_new.head(20))


if __name__ == "__main__":
    main()

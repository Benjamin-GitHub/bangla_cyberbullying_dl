import pandas as pd
from src.config import DATA_DIR
from src.data_utils import prepare_dataframe, stratified_splits

def main():
    data_path = DATA_DIR / "CyberBulling_Dataset_Bangla.xlsx"
    df_raw = pd.read_excel(data_path)

    df = prepare_dataframe(df_raw)
    print("Prepared df shape:", df.shape)
    print(df.head())

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(df)
    print("Train / Val / Test sizes:", len(X_train), len(X_val), len(X_test))

if __name__ == "__main__":
    main()
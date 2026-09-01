from pathlib import Path 
import pandas as pd 

BRONZE_PATH = Path("data/bronze")
DATASET = BRONZE_PATH / "sample_network_traffic.csv"

def create_bronze_directory():
    BRONZE_PATH.mkdir(parents = True,exist_ok = True)
    print(f"Bronze Directory Ready : {BRONZE_PATH}")



def load_bronze_data():

    df = pd.read_csv(DATASET)
    parquet_path = BRONZE_PATH / "sample_network_traffic.parquet"
    df.to_parquet(parquet_path,index = False)
   
    print("Dataset Loaded Successfully")
    print("Rows : ",len(df))
    print("Columns : ",len(df.columns))
    print(f"Parquet file created: {parquet_path}")
    return df

if __name__ == "__main__":
    create_bronze_directory()
    load_bronze_data()
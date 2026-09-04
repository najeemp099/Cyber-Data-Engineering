def clean_data():
    df = pd.read_parquet(BRONZE_FILE)

    print("Bronze data loaded")
    print(f"Rows before cleaning: {len(df)}")

    df = df.drop_duplicates()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce")
    df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce")
    df["packet_count"] = pd.to_numeric(df["packet_count"], errors="coerce")
    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce")

    df["protocol"] = df["protocol"].str.upper().str.strip()
    df["label"] = df["label"].str.upper().str.strip()

    SILVER_PATH.mkdir(parents=True, exist_ok=True)

    silver_file = SILVER_PATH / "network_traffic_cleaned.parquet"
    df.to_parquet(silver_file, index=False)

    print(f"Rows after removing duplicates: {len(df)}")
    print(f"Silver file created: {silver_file}")

    return df
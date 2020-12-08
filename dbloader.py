import os
import pandas as pd
from lambdb.dbengine import create_engine, get_time_series_into_db, DBEngine


if __name__ == "__main__":
    DATA_PATH = "../data"
    TS_PATH = "train"
    engine = create_engine(ENGINE)
    train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    train.to_sql("training_set",
                 con=engine,
                 if_exists="replace",
                 index=False,
                 method="multi")
    print("Finished loading training_set")
    csv_dir = os.path.join(DATA_PATH, TS_PATH)
    engine = DBEngine(ENGINE, TS, csv_dir=csv_dir, func=get_time_series_into_db)
    engine.to_sql(train, num_threads=8)
    print("Finished loading time_series")

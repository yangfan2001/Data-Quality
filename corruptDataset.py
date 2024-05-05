import pandas as pd
import numpy as np
import random
import string

def corrupt_dataset(old_file_name, corrupted_cols, new_file_name):
    """
    This function reads the raw dataset, generates a corrupted dataset based on the corrupted columns,
    and creates a corresponding CSV file indicating the type of corruption applied to each cell.

    :param old_file_name: String, the file path of the original dataset.
    :param corrupted_cols: List of tuples (int, int), where the first int is the column index
                           and the second int is the percentage of the data in that column to be corrupted.
    :param new_file_name: String, the file path where the corrupted dataset will be saved.
    :return: None, saves two files - the corrupted dataset and a tracker file.
    """
    df = pd.read_csv(old_file_name)
    corruption_tracker = pd.DataFrame(1, index=np.arange(len(df)), columns=df.columns)

    for col_idx, corruption_percentage in corrupted_cols:
        column_type = df.dtypes[col_idx]
        num_corrupt = int(len(df) * corruption_percentage / 100)
        corrupt_indices = random.sample(range(len(df)), num_corrupt)

        if column_type in ['float64', 'float32']:
            noise_level = 10 * df.iloc[:, col_idx].std()
            noise_func = np.random.normal
        elif column_type in ['int64', 'int32']:
            noise_level = int(10 * df.iloc[:, col_idx].std())
            noise_func = lambda mean, std: int(np.random.normal(mean, std))
        else:
            noise_level = None  # 不应用噪声于非数值列

        for idx in corrupt_indices:
            if column_type in ['float64', 'float32', 'int64', 'int32']:
                action = random.choice(['noise', 'null'])
            else:
                action = random.choice(['add_remove', 'nonsense', 'null'])

            if action == 'noise' and noise_level is not None:
                original_val = df.iloc[idx, col_idx]
                noise = noise_func(0, noise_level)
                df.iloc[idx, col_idx] = original_val + noise
                corruption_tracker.iloc[idx, col_idx] = 5  # Mark as Noise added

            elif action == 'null':
                df.iloc[idx, col_idx] = np.nan
                corruption_tracker.iloc[idx, col_idx] = 4  # Mark as NULL value

            elif action == 'add_remove':
                # 对非数值列处理逻辑
                pass
            elif action == 'nonsense':
                # 对非数值列处理逻辑
                pass

    df.to_csv(new_file_name, index=False)
    tracker_file_name = new_file_name.replace(".csv", "_tracker.csv")
    corruption_tracker.to_csv(tracker_file_name, index=False)


if __name__ == '__main__':
    corrupt_dataset('./Dataset/2018_Central_Park_Squirrel_Census_-_Squirrel_Data_20240427.csv',
                    [(2, 5), (1, 5),(6,5)], "test_num1.csv")
    corrupt_dataset('./Dataset/SAT__College_Board__2010_School_Level_Results_20240504.csv',
                    [(2, 10), (3, 5),(4,5),(5,5)], "test_num2.csv")
    corrupt_dataset('./Dataset/Air_Quality_20240504.csv',
                    [(10, 10)], "test_num3.csv")
    corrupt_dataset('./Dataset/Current_Reservoir_Levels_20240505.csv',
                    [(1, 10), (3, 10)], "test_num4.csv")

#%%

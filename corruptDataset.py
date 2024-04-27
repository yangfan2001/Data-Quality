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
    # Create a DataFrame to track changes, initialized with '1' (Valid value)
    corruption_tracker = pd.DataFrame(1, index=np.arange(len(df)), columns=df.columns)

    for col_idx, corruption_percentage in corrupted_cols:
        num_corrupt = int(len(df) * corruption_percentage / 100)
        corrupt_indices = random.sample(range(len(df)), num_corrupt)

        for idx in corrupt_indices:
            action = random.choice(['add_remove', 'nonsense', 'null'])

            if action == 'add_remove':
                original_str = str(df.iloc[idx, col_idx])
                if len(original_str) > 1 and random.choice([True, False]):
                    position_to_remove = random.randint(0, len(original_str) - 1)
                    modified_str = original_str[:position_to_remove] + original_str[position_to_remove + 1:]
                else:
                    char_to_add = random.choice(string.ascii_letters)
                    position_to_add = random.randint(0, len(original_str))
                    modified_str = original_str[:position_to_add] + char_to_add + original_str[position_to_add:]
                df.iloc[idx, col_idx] = modified_str
                corruption_tracker.iloc[idx, col_idx] = 2  # Mark as Misspelling/Abbreviation

            elif action == 'nonsense':
                nonsense_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                df.iloc[idx, col_idx] = nonsense_str
                corruption_tracker.iloc[idx, col_idx] = 3  # Mark as Invalid value

            elif action == 'null':
                df.iloc[idx, col_idx] = np.nan
                corruption_tracker.iloc[idx, col_idx] = 4  # Mark as NULL value

    df.to_csv(new_file_name, index=False)
    tracker_file_name = new_file_name.replace(".csv", "_tracker.csv")
    corruption_tracker.to_csv(tracker_file_name, index=False)

if __name__ == '__main__':
    corrupt_dataset('Public_Recycling_Bins_20240427.csv', [(1, 10), (2, 20)], "test.csv")

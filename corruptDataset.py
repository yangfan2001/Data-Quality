import pandas as pd
import numpy as np
import random
import string

def corrupt_dataset(old_file_name, corrupted_cols, new_file_name):
    """
    This function read the raw dataset and generate a corrupted dataset based on the corrupted columns
    :param old_file_name: String
    :param corrupted_cols: list of (int,int)
    :param new_file_name: String
    :return: None
    """
    df = pd.read_csv(old_file_name)

    for col_idx, corruption_percentage in corrupted_cols:

        num_corrupt = int(len(df) * corruption_percentage / 100)

        corrupt_indices = random.sample(range(len(df)), num_corrupt)

        for idx in corrupt_indices:
            action = random.choice(['add_remove', 'nonsense', 'null']) # random select the way

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

            elif action == 'nonsense':
                # to some meaningless random string
                nonsense_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                df.iloc[idx, col_idx] = nonsense_str

            elif action == 'null':

                df.iloc[idx, col_idx] = np.nan
    df.to_csv(new_file_name, index=False)


if __name__ == '__main__':
    corrupt_dataset('Public_Recycling_Bins_20240427.csv',[(1,10),(2,20)],"test.csv")


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance

'''
The link for dataset
https://data.cityofnewyork.us/Environment/Public-Recycling-Bins/sxx4-xhzg/about_data
'''

df = pd.read_csv('bin.csv')

column_to_count = 'DSNY Zone'

counts = df[column_to_count].value_counts()

total_counts = counts.sum()
proportions = counts / total_counts

print("Proportions of each unique value:")
print(proportions)

threshold = 0.01
anomaly_threshold = 0.7 

low_proportion_values = proportions[proportions < threshold].index
high_proportion_values = proportions[proportions >= threshold].index
print(low_proportion_values)
print(high_proportion_values)

average_distances = {}

for low_val in low_proportion_values:
    distances = []
    for high_val in high_proportion_values:
        max_len = max(len(low_val), len(high_val))
        edit_dist = levenshtein_distance(low_val, high_val)
        normalized_dist = edit_dist / max_len
        distances.append(normalized_dist)
        #print(f"Edit distance between '{low_val}' and '{high_val}': {normalized_dist}")
    average_distance = sum(distances) / len(high_proportion_values)
    average_distances[low_val] = average_distance

anomalies = {low_val: avg_dist for low_val, avg_dist in average_distances.items() if avg_dist > anomaly_threshold}


print(f"Average normalized edit distances for each low proportion value:")
for low_val, avg_dist in average_distances.items():
    print(f"{low_val}: {avg_dist}")

print(f"\nAnomalies (average distance > {anomaly_threshold}):")
for anomaly, avg_dist in anomalies.items():
    print(f"{anomaly}: {avg_dist}")


# Similarity
# TF-IDF and cos
#vectorizer = TfidfVectorizer()
#text_values = list(low_proportion_values) + list(high_proportion_values)
#print(text_values)
#text_vectors = vectorizer.fit_transform(text_values)

#similarity_matrix = cosine_similarity(text_vectors)

#print("Similarity matrix between low and high proportion values:")
#print(similarity_matrix)

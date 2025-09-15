import pandas as pd
from collections import Counter

data = pd.read_excel("data_BT_tinh_E.xlsx")

X = data.iloc[:,0]
Y = data.iloc[:,1]


def find_most_probable_value(data):
    counts = Counter(data)
    
    total_count = len(data)
    probabilities = {value: count / total_count for value, count in counts.items()}
    
    most_probable_value = max(probabilities, key=probabilities.get)
    highest_probability = probabilities[most_probable_value]
    
    return most_probable_value, highest_probability

most_probable_x, prob_x = find_most_probable_value(X)
most_probable_y, prob_y = find_most_probable_value(Y)

most_probable_x, prob_x, most_probable_y, prob_y

print(most_probable_x, prob_x)
print(most_probable_y, prob_y)

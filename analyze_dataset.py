import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Data/dataset1.csv')

print("=" * 80)
print("DATASET ANALYSIS - Understanding Class Distribution")
print("=" * 80)

# Analyze each class
for class_id in [0, 1, 2]:
    class_name = {0: 'Good', 1: 'Bad', 2: 'Medium'}[class_id]
    class_data = df[df['Output'] == class_id]

    print(f"\n{class_name} Soil (Class {class_id}) - {len(class_data)} samples")
    print("-" * 80)

    # Stats for key features
    features = ['N', 'P', 'K', 'pH', 'EC', 'OC']
    for feature in features:
        min_val = class_data[feature].min()
        max_val = class_data[feature].max()
        mean_val = class_data[feature].mean()
        print(f"  {feature:3s}: Min={min_val:7.2f}, Max={max_val:7.2f}, Mean={mean_val:7.2f}")

print("\n" + "=" * 80)
print("COMPARISON: Average Feature Values by Class")
print("=" * 80)

data_summary = pd.DataFrame()
for class_id in [0, 1, 2]:
    class_name = {0: 'Good', 1: 'Bad', 2: 'Medium'}[class_id]
    data_summary[class_name] = df[df['Output'] == class_id][df.columns[:-1]].mean()

print(data_summary.T)

print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)
print("\nTo create 'Bad' soil predictions, you need to match the feature ranges")
print("that are typical for Bad soil class in the dataset.\n")

for feature in ['N', 'P', 'K', 'pH', 'EC', 'OC']:
    good_data = df[df['Output'] == 0][feature]
    bad_data = df[df['Output'] == 1][feature]

    print(f"{feature}:")
    print(f"  Good range: {good_data.min():.1f} - {good_data.max():.1f} (mean: {good_data.mean():.1f})")
    print(f"  Bad range:  {bad_data.min():.1f} - {bad_data.max():.1f} (mean: {bad_data.mean():.1f})")

    if good_data.mean() > bad_data.mean():
        print(f"  -> BAD soil has LOWER {feature} than GOOD soil")
    else:
        print(f"  -> BAD soil has HIGHER {feature} than GOOD soil")
    print()

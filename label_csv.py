import os
import pandas as pd

# Define the paths for the Training and Testing folders
training_folder = r'C:\Users\Buranon\OneDrive\Desktop\Data\Training image'  # Replace with the actual path to the Training folder
testing_folder = r'C:\Users\Buranon\OneDrive\Desktop\Data\Testing image'   # Replace with the actual path to the Testing folder

# Helper function to get file paths and labels
def get_file_list_with_labels(folder_path):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Image file types
                # Use the subfolder name as the label
                label = os.path.basename(root)
                file_list.append({
                    'label': label,
                    'file': os.path.join(root, file)
                })
    return file_list

# Get file lists for training and testing with labels
train_files_with_labels = get_file_list_with_labels(training_folder)
test_files_with_labels = get_file_list_with_labels(testing_folder)

# Create separate DataFrames for training and testing
train_data = pd.DataFrame(train_files_with_labels)
test_data = pd.DataFrame(test_files_with_labels)

# Define paths for the output CSV files
train_csv_path = r'C:\Users\Buranon\OneDrive\Desktop\Data\train_split.csv'  # Replace with desired output path for the training CSV
test_csv_path = r'C:\Users\Buranon\OneDrive\Desktop\Data\test_split.csv'   # Replace with desired output path for the testing CSV

# Save each DataFrame to its respective CSV file
train_data.to_csv(train_csv_path, index=False)
test_data.to_csv(test_csv_path, index=False)

print(f"Training dataset CSV saved at: {train_csv_path}")
print(f"Testing dataset CSV saved at: {test_csv_path}")

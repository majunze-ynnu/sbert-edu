import pandas as pd
from torch.utils.data import Dataset
from sentence_transformers import InputExample

class SyllabusDataset(Dataset):
    """Syllabus-Text Consistency Dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        return InputExample(texts=[row['sentence1'], row['sentence2']], label=float(row['label']))

def main():
    """Main function to test the dataset loader."""
    train_dataset = SyllabusDataset(csv_file='data/train.csv')
    print(f"Loaded train dataset with {len(train_dataset)} samples.")

    # Print a sample
    sample = train_dataset[0]
    print("Sample 0:")
    print(f"  Sentence 1: {sample.texts[0]}")
    print(f"  Sentence 2: {sample.texts[1]}")
    print(f"  Label: {sample.label}")

if __name__ == '__main__':
    main()

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, util
from src.dataset import SyllabusDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, test_df, threshold=0.6):
    """Evaluates the model on the test set."""
    sentences1 = test_df['sentence1'].tolist()
    sentences2 = test_df['sentence2'].tolist()
    labels = test_df['label'].tolist()

    embeddings1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=False)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=False)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    pred_scores = [cosine_scores[i][i] for i in range(len(sentences1))]

    predictions = [1 if score > threshold else 0 for score in pred_scores]

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)

    print("\n------ Evaluation Results ------")
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("---------------------------------")

def main():
    """Main function to train and evaluate the model."""
    # 1. Load Model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # 2. Load Dataset
    train_dataset = SyllabusDataset(csv_file='data/train.csv')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    test_df = pd.read_csv('data/test.csv')

    # 3. Define Loss
    train_loss = losses.CosineSimilarityLoss(model)

    # 4. Train Model
    num_epochs = 1
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    print("Starting model training...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              show_progress_bar=True)
    print("Training complete.")

    # 5. Evaluate Model
    print("Evaluating model...")
    evaluate_model(model, test_df)

if __name__ == '__main__':
    main()

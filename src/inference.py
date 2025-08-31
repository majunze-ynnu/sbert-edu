from sentence_transformers import SentenceTransformer, util

def check_consistency(model, syllabus, text):
    """
    Checks the consistency between a syllabus and a teaching text.

    Args:
        model: The SentenceTransformer model.
        syllabus (str): The syllabus text.
        text (str): The teaching text.

    Returns:
        float: The consistency score (cosine similarity).
    """
    # Encode the syllabus and the text
    embedding1 = model.encode(syllabus, convert_to_tensor=True)
    embedding2 = model.encode(text, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.cos_sim(embedding1, embedding2)

    return cosine_scores.item()

def main():
    """Main function to run inference examples."""
    # NOTE: In a real application, you would load your fine-tuned model.
    # Due to environment limitations, we use the base model here.
    # model = SentenceTransformer('output/model')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # --- Example 1: Consistent Pair ---
    syllabus_1 = "Syllabus: Introduction to Python"
    text_1 = "This text covers the fundamental concepts of Python, including variables, data types, and control flow."

    consistency_1 = check_consistency(model, syllabus_1, text_1)

    print("------ Inference Example 1 (Consistent) ------")
    print(f"Syllabus: {syllabus_1}")
    print(f"Text: {text_1}")
    print(f"Consistency Score: {consistency_1:.4f}")
    print("----------------------------------------------")

    # --- Example 2: Inconsistent Pair ---
    syllabus_2 = "Syllabus: Data Structures in Python"
    text_2 = "This article discusses the history of the internet and the development of web protocols."

    consistency_2 = check_consistency(model, syllabus_2, text_2)

    print("\n------ Inference Example 2 (Inconsistent) ------")
    print(f"Syllabus: {syllabus_2}")
    print(f"Text: {text_2}")
    print(f"Consistency Score: {consistency_2:.4f}")
    print("------------------------------------------------")

if __name__ == '__main__':
    main()

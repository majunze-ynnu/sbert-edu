from sentence_transformers import SentenceTransformer, util

# This dictionary is copied from generate_dataset.py to make this script self-contained.
KNOWLEDGE_DATA = {
    "李白《静夜思》": [
        "此诗描写了秋日夜晚诗人于屋内抬头望月而思念故乡的深切感受。",
        "“举头望明月，低头思故乡”是诗中表达游子思乡之情的千古名句。",
        "这首诗是唐代诗人李白所作的一首五言绝句。"
    ],
    "二次函数": [
        "二次函数是指变量的最高次数为二的多项式函数。",
        "二次函数的图像是一条对称的抛物线。",
        "抛物线的开口方向由二次项系数的正负决定。"
    ],
    "秦始皇": [
        "秦始皇是中国历史上首位完成华夏大一统的政治人物。",
        "他推行了统一文字、货币、度量衡等一系列重要改革。",
        "他下令修建了万里长城以抵御北方游牧民族的侵袭。"
    ]
}

def check_consistency(model, knowledge_point, text):
    """
    Checks the consistency between a knowledge point and a teaching text.
    """
    embedding1 = model.encode(knowledge_point, convert_to_tensor=True)
    embedding2 = model.encode(text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding1, embedding2)
    return cosine_scores.item()

def main():
    """Main function to run inference examples."""
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # --- Example 1: Consistent Pair ---
    knowledge_point_1 = "李白《静夜思》"
    explanation_1 = "此诗描写了秋日夜晚诗人于屋内抬头望月而思念故乡的深切感受。"

    consistency_1 = check_consistency(model, knowledge_point_1, explanation_1)

    print("------ 推理示例 1 (内容一致) ------")
    print(f"教学知识点: {knowledge_point_1}")
    print(f"教学讲解: {explanation_1}")
    print(f"一致性得分: {consistency_1:.4f}")
    print("------------------------------------")

    # --- Example 2: Inconsistent Pair ---
    knowledge_point_2 = "二次函数"
    explanation_2 = "秦始皇是中国历史上首位完成华夏大一统的政治人物。"

    consistency_2 = check_consistency(model, knowledge_point_2, explanation_2)

    print("\n------ 推理示例 2 (内容不一致) ------")
    print(f"教学知识点: {knowledge_point_2}")
    print(f"教学讲解: {explanation_2}")
    print(f"一致性得分: {consistency_2:.4f}")
    print("--------------------------------------")

if __name__ == '__main__':
    main()

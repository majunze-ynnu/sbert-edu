from sentence_transformers import SentenceTransformer, util

# This dictionary is copied from generate_dataset.py to make this script self-contained.
KNOWLEDGE_DATA = {
    "李白《静夜思》赏析": "《静夜思》是唐代诗人李白所作的一首五言绝句。此诗描写了秋日夜晚，诗人于屋内抬头望月而思念故乡的深切感受。诗中“举头望明月，低头思故乡”是千古名句，表达了游子的思乡之情。",
    "二次函数的定义与性质": "在数学中，二次函数是变量的最高次数为二的多项式函数。其图像是一条对称的抛物线，开口方向由二次项系数的正负决定。二次函数在物理学、工程学等领域有广泛应用，例如描述抛物运动。",
    "秦始皇的主要功绩": "秦始皇是中国历史上首位完成华夏大一统的政治人物，自称“始皇帝”。他推行了一系列重要改革，如统一文字、货币、度量衡，巩固了中央集权。他还下令修建了万里长城，以抵御北方游牧民族的侵袭。",
    "光合作用的原理": "光合作用是指绿色植物利用光能，将二氧化碳和水转化为储存能量的有机物，并释放出氧气的过程。叶绿体是进行光合作用的主要场所，叶绿素是吸收光能的关键色素。这个过程对于维持大气中的氧气平衡和所有生物的生存至关重要。",
    "勾股定理": "勾股定理，又称毕达哥拉斯定理，是一个基本的几何定理。它说明了在直角三角形中，两条直角边的平方和等于斜边的平方。数学表达式为 a² + b² = c²，其中a和b为直角边，c为斜边。"
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

    # --- Example 1: Consistent Pair (from dataset) ---
    topic1 = "李白《静夜思》赏析"
    knowledge_point_1 = f"知识点：{topic1}"
    explanation_1 = f"讲解：{KNOWLEDGE_DATA[topic1]}"

    consistency_1 = check_consistency(model, knowledge_point_1, explanation_1)

    print("------ 推理示例 1 (内容一致) ------")
    print(f"教学知识点: {knowledge_point_1}")
    print(f"教学讲解: {explanation_1}")
    print(f"一致性得分: {consistency_1:.4f}")
    print("------------------------------------")

    # --- Example 2: Inconsistent Pair (from dataset) ---
    topic2 = "二次函数的定义与性质"
    topic3 = "勾股定理"
    knowledge_point_2 = f"知识点：{topic2}"
    explanation_2 = f"讲解：{KNOWLEDGE_DATA[topic3]}" # Mismatched explanation

    consistency_2 = check_consistency(model, knowledge_point_2, explanation_2)

    print("\n------ 推理示例 2 (内容不一致) ------")
    print(f"教学知识点: {knowledge_point_2}")
    print(f"教学讲解: {explanation_2}")
    print(f"一致性得分: {consistency_2:.4f}")
    print("--------------------------------------")

if __name__ == '__main__':
    main()

import pandas as pd
import random

# Define Chinese knowledge points and corresponding single-sentence explanations
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
    ],
    "光合作用": [
        "光合作用是指绿色植物利用光能将二氧化碳和水转化为有机物的过程。",
        "叶绿体是植物进行光合作用的主要场所。",
        "光合作用对于维持大气中的氧气平衡至关重要。"
    ],
    "勾股定理": [
        "勾股定理是一个说明直角三角形三边关系的几何定理。",
        "该定理指出两条直角边的平方和等于斜边的平方。",
        "其数学表达式为 a² + b² = c²。"
    ]
}

def generate_sentence_pairs(num_samples):
    """Generates positive and negative sentence pairs without prefixes."""
    pairs = []
    labels = []

    topics = list(KNOWLEDGE_DATA.keys())

    for _ in range(num_samples):
        # Create a positive pair (50% chance)
        if random.random() > 0.5:
            topic = random.choice(topics)
            knowledge_point = topic  # No prefix
            explanation = random.choice(KNOWLEDGE_DATA[topic])  # Single sentence
            pairs.append([knowledge_point, explanation])
            labels.append(1)
        # Create a negative pair (50% chance)
        else:
            topic1, topic2 = random.sample(topics, 2)
            knowledge_point = topic1  # No prefix
            explanation = random.choice(KNOWLEDGE_DATA[topic2])  # Single sentence from another topic
            pairs.append([knowledge_point, explanation])
            labels.append(0)

    return pairs, labels

def main():
    """Main function to generate and save datasets."""
    # Generate training data
    train_pairs, train_labels = generate_sentence_pairs(1000)
    train_df = pd.DataFrame(train_pairs, columns=['sentence1', 'sentence2'])
    train_df['label'] = train_labels
    train_df.to_csv('data/train.csv', index=False, encoding='utf-8')

    # Generate test data
    test_pairs, test_labels = generate_sentence_pairs(200)
    test_df = pd.DataFrame(test_pairs, columns=['sentence1', 'sentence2'])
    test_df['label'] = test_labels
    test_df.to_csv('data/test.csv', index=False, encoding='utf-8')

    print("新的中文数据集已成功生成在 'data/' 目录下 (单句，无前缀)。")

if __name__ == "__main__":
    main()

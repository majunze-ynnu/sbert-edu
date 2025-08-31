import pandas as pd
import random

# Define Chinese knowledge points and corresponding explanations
KNOWLEDGE_DATA = {
    "李白《静夜思》赏析": "《静夜思》是唐代诗人李白所作的一首五言绝句。此诗描写了秋日夜晚，诗人于屋内抬头望月而思念故乡的深切感受。诗中“举头望明月，低头思故乡”是千古名句，表达了游子的思乡之情。",
    "二次函数的定义与性质": "在数学中，二次函数是变量的最高次数为二的多项式函数。其图像是一条对称的抛物线，开口方向由二次项系数的正负决定。二次函数在物理学、工程学等领域有广泛应用，例如描述抛物运动。",
    "秦始皇的主要功绩": "秦始皇是中国历史上首位完成华夏大一统的政治人物，自称“始皇帝”。他推行了一系列重要改革，如统一文字、货币、度量衡，巩固了中央集权。他还下令修建了万里长城，以抵御北方游牧民族的侵袭。",
    "光合作用的原理": "光合作用是指绿色植物利用光能，将二氧化碳和水转化为储存能量的有机物，并释放出氧气的过程。叶绿体是进行光合作用的主要场所，叶绿素是吸收光能的关键色素。这个过程对于维持大气中的氧气平衡和所有生物的生存至关重要。",
    "勾股定理": "勾股定理，又称毕达哥拉斯定理，是一个基本的几何定理。它说明了在直角三角形中，两条直角边的平方和等于斜边的平方。数学表达式为 a² + b² = c²，其中a和b为直角边，c为斜边。"
}

def generate_sentence_pairs(num_samples):
    """Generates positive and negative sentence pairs in Chinese."""
    pairs = []
    labels = []

    topics = list(KNOWLEDGE_DATA.keys())

    for _ in range(num_samples):
        if random.random() > 0.5:
            topic = random.choice(topics)
            knowledge_point = f"知识点：{topic}"
            explanation = KNOWLEDGE_DATA[topic]
            pairs.append([knowledge_point, f"讲解：{explanation}"])
            labels.append(1)
        else:
            topic1, topic2 = random.sample(topics, 2)
            knowledge_point = f"知识点：{topic1}"
            explanation = KNOWLEDGE_DATA[topic2]
            pairs.append([knowledge_point, f"讲解：{explanation}"])
            labels.append(0)

    return pairs, labels

def main():
    """Main function to generate and save datasets."""
    train_pairs, train_labels = generate_sentence_pairs(1000)
    train_df = pd.DataFrame(train_pairs, columns=['sentence1', 'sentence2'])
    train_df['label'] = train_labels
    train_df.to_csv('data/train.csv', index=False, encoding='utf-8')

    test_pairs, test_labels = generate_sentence_pairs(200)
    test_df = pd.DataFrame(test_pairs, columns=['sentence1', 'sentence2'])
    test_df['label'] = test_labels
    test_df.to_csv('data/test.csv', index=False, encoding='utf-8')

    print("中文数据集已成功生成在 'data/' 目录下。")

if __name__ == "__main__":
    main()

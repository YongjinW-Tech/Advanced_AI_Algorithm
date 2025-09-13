import os
import re
import jieba
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_file = './training_data.txt'

class Word2VecTrainer:
    """ Word2Vec Trainer """
    
    def __init__(self, data_file=data_file):
        """ Initialize the trainer with the data file path """
        self.data_file = data_file
        self.sentences = []
        self.model_skipgram = None  # Skip-Gram
        self.model_cbow = None      # CBOW 

    def load_and_preprocess_data(self):
        """ Load and preprocess the training data """
        print("Loading and preprocessing training data...")
        
        if not os.path.exists(self.data_file):
            print(f"Error: Data file {self.data_file} does not exist")
            return False
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        print(f"Row count: {len(texts)} lines of text")
        
        # Preprocess the text data
        self.sentences = []
        for text in texts:
            text = text.strip() # Remove leading/trailing whitespace
            if text:
                # Use jieba to segment the text
                words = jieba.lcut(text)
                # Filter out short words（小于 2） and non-alphanumeric（非字母数字字符） tokens
                filtered_words = [stripped for word in words if (stripped := word.strip()) and len(stripped) >= 2 and stripped.isalnum()]
                if len(filtered_words) >= 3:  # At least 3 words are needed
                    self.sentences.append(filtered_words)

        print(f"Preprocessing complete, obtained {len(self.sentences)} valid sentences")

        # Count the vocabulary
        all_words = []
        for sentence in self.sentences:
            all_words.extend(sentence)
        
        unique_words = set(all_words) # Unique words
        print(f"Total unique words (V)：{len(unique_words)} words")

        # Show some example sentences
        print("\nExample sentences:")
        for i, sentence in enumerate(self.sentences[:5]):
            print(f"{i+1}: {' '.join(sentence)}")
        
        return True
    
    def train_models(self, vector_size=100, window=5, min_count=1, workers=4, epochs=200):
        """Train Word2Vec models (Skip-Gram and CBOW)"""
        print(f"\nStarting training Word2Vec models...")
        print(f"Parameter settings: \n \t vector_size={vector_size}, window={window}, min_count={min_count}, epochs={epochs}")
        
        # Train Skip-Gram model
        print("\n1. Training Skip-Gram model...")
        self.model_skipgram = Word2Vec(
            sentences=self.sentences,   # 训练语料（已分词的句子列表）
            vector_size=vector_size,    # 词向量维度（默认100维）
            window=window,              # 上下文窗口大小（默认5）
            min_count=min_count,        # 最小词频阈值（默认1） min_count=1：保留所有词汇
            workers=workers,            # 并行训练线程数（默认4）
            sg=1,                       # Skip-Gram. sg=0 for CBOW
            epochs=epochs,              # 训练迭代次数（默认5）
            seed=42                      # 随机种子（确保结果可复现）
        )

        # Train CBOW model
        print("2. Training CBOW model...")
        self.model_cbow = Word2Vec(
            sentences=self.sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=0,  # CBOW
            epochs=epochs,
            seed=42
        )
        
        print("Training completed!")

        # Save models
        # 模型保存路径：./model-checkpoint 目录下
        if not os.path.exists('./model-checkpoint'):
            os.makedirs('./model-checkpoint')
        model_paths = {
            "Skip-Gram": './model-checkpoint/word2vec_skipgram.model',
            "CBOW": './model-checkpoint/word2vec_cbow.model'
        }
        self.model_skipgram.save(model_paths["Skip-Gram"])
        self.model_cbow.save(model_paths["CBOW"])
        print("Models saved!")
    
    def analyze_vocabulary(self):
        """ Analyze the vocabulary """
        print(f"\n=== Vocabulary Analysis ===")

        # Extract vocabularies
        # self.model_skipgram.wv 其中 wv = Word Vectors（词向量）
        #  - 这是Gensim Word2Vec模型中存储词向量的核心对象 self.wv = KeyedVectors(vector_size)
        #  - 包含所有学习到的词汇及其对应的向量表示
        # key_to_index() 是一个字典（Dictionary），存储词汇到索引的映射
        #  - 结构：{"词汇": 索引编号}
        #  - 例如：{"国王": 0, "王后": 1, "医生": 2, ...}
        # .keys() 获取字典中所有的键（key），即所有词汇，返回一个 <class 'list'>
        vocab_sg = list(self.model_skipgram.wv.key_to_index.keys())
        vocab_cbow = list(self.model_cbow.wv.key_to_index.keys())

        print(type(vocab_sg))

        print(f"Skip-Gram vocabulary size: {len(vocab_sg)}")
        print(f"CBOW vocabulary size: {len(vocab_cbow)}")
        
        # 输出部分词汇表样例
        # print(f"\n词汇表样例（前30个）:")
        # for i, word in enumerate(vocab_sg[:30]):
        #     if i % 10 == 0 and i > 0:
        #         print()
        #     print(f"{word:8s}", end=" ")
        # print("\n")
        
        return vocab_sg
    
    def analyze_semantic_relations(self):
        """ Analyze Semantic Relations """
        print(f"\n=== Analyze Semantic Relations ===")

        # Define more test cases
        test_cases = [
            # 原有的核心测试案例
            ("国王", "男人", "女人", "王后", "王室性别关系"),
            ("父亲", "男人", "女人", "母亲", "家庭性别关系"),
            ("王子", "男人", "女人", "公主", "王室后代性别关系"),
            ("儿子", "男人", "女人", "女儿", "子女性别关系"),
            ("丈夫", "男人", "女人", "妻子", "配偶性别关系"),
            ("哥哥", "男人", "女人", "姐姐", "兄弟姐妹性别关系"),
            ("爷爷", "男人", "女人", "奶奶", "祖辈性别关系"),
            ("北京", "中国", "日本", "东京", "国家首都关系"),
            ("苹果", "红色", "黄色", "香蕉", "水果颜色关系"),
            ("医生", "医院", "学校", "老师", "场所职业关系"),
            
            # 新增的语义关系测试案例
            # 扩展家庭关系
            ("外公", "男人", "女人", "外婆", "外祖父母性别关系"),
            ("叔叔", "男人", "女人", "阿姨", "父系亲属性别关系"),
            ("舅舅", "男人", "女人", "舅妈", "母系亲属性别关系"),
            
            # 世界首都关系
            ("华盛顿", "美国", "法国", "巴黎", "国家首都关系"),
            ("伦敦", "英国", "德国", "柏林", "国家首都关系"),
            ("莫斯科", "俄罗斯", "意大利", "罗马", "国家首都关系"),
            ("首尔", "韩国", "泰国", "曼谷", "国家首都关系"),
            
            # 水果颜色关系
            ("橙子", "橙色", "紫色", "葡萄", "水果颜色关系"),
            ("草莓", "红色", "蓝色", "蓝莓", "水果颜色关系"),
            ("柠檬", "黄色", "绿色", "苹果", "水果颜色关系"),
            
            # 职业场所关系
            ("护士", "医院", "学校", "教师", "场所职业关系"),
            ("程序员", "公司", "厨房", "厨师", "场所职业关系"),
            ("飞行员", "机场", "农田", "农民", "场所职业关系"),
            
            # 动物栖息地关系
            ("老虎", "森林", "海洋", "鲸鱼", "栖息地动物关系"),
            ("熊猫", "竹林", "草原", "狮子", "栖息地动物关系"),
            
            # 季节时间关系
            ("春天", "温暖", "寒冷", "冬天", "季节温度关系"),
            ("白天", "明亮", "黑暗", "夜晚", "时间光线关系"),
            ("早晨", "日出", "日落", "傍晚", "时间太阳关系"),
            
            # 交通工具环境关系
            ("汽车", "马路", "铁轨", "火车", "交通环境关系"),
            ("轮船", "海洋", "天空", "飞机", "交通环境关系"),
            ("自行车", "陆地", "水中", "鱼", "环境适应关系"),
            
            # 颜色对比关系
            ("红色", "热情", "冷静", "蓝色", "颜色情感关系"),
            ("黑色", "深沉", "纯洁", "白色", "颜色特质关系"),
            ("绿色", "自然", "活力", "橙色", "颜色联想关系"),
            
            # 情感对比关系
            ("快乐", "积极", "消极", "悲伤", "情感极性关系"),
            ("愤怒", "激烈", "平和", "平静", "情感强度关系"),
            ("恐惧", "退缩", "前进", "勇敢", "情感行为关系"),
            
            # 运动类型关系
            ("游泳", "水中", "陆地", "跑步", "运动环境关系"),
            ("足球", "团队", "个人", "网球", "运动类型关系"),
        ]

        for model_name, model in [("Skip-Gram", self.model_skipgram), ("CBOW", self.model_cbow)]:
            print(f"\n{model_name} model semantic relations analysis:")
            successful_tests = 0
            total_tests = 0
            
            for word1, word2, word3, expected, description in test_cases[:5]:
                try:
                    # 检查所有词是否在词汇表中
                    vocab = model.wv.key_to_index
                    if all(word in vocab for word in [word1, word2, word3]):
                        total_tests += 1
                        # 计算 word1 - word2 + word3
                        result = model.wv.most_similar(
                            positive=[word1, word3], 
                            negative=[word2], 
                            topn=5
                        )
                        
                        print(f"  测试: {word1} - {word2} + {word3} = ?")
                        print(f"  描述: {description}")
                        print(f"  预期: {expected}")
                        result_words = [word for word, score in result]
                        print(f"  结果: {result_words}")
                        
                        # 检查预期词是否在结果中
                        if expected in result_words:
                            rank = result_words.index(expected) + 1
                            print(f"  ✓ 预期词 '{expected}' 排名第 {rank}")
                            successful_tests += 1
                        else:
                            print(f"  ✗ 预期词 '{expected}' 不在前5名中")
                        print()
                    else:
                        missing = [word for word in [word1, word2, word3] if word not in vocab]
                        print(f"  跳过测试 {word1}-{word2}+{word3}: 缺少词汇 {missing}")
                        
                except Exception as e:
                    print(f"  测试失败 {word1}-{word2}+{word3}: {e}")
            
            if total_tests > 0:
                success_rate = successful_tests / total_tests * 100
                print(f"  语义关系测试成功率: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
    
    def analyze_semantic_clusters(self):
        """ Analyze Semantic Clusters """
        print(f"\n=== Semantic Clusters Analysis ===")

        vocab = list(self.model_skipgram.wv.key_to_index.keys())

        # Define semantic categories
        semantic_categories = {
            "家庭关系": ["父亲", "母亲", "儿子", "女儿", "哥哥", "姐姐", "弟弟", "妹妹", "爷爷", "奶奶", "外公", "外婆", "丈夫", "妻子"],
            "王室关系": ["国王", "王后", "王子", "公主"],
            "职业类别": ["医生", "老师", "教师", "工程师", "司机", "厨师", "警察", "消防员", "农民", "程序员", "护士", "建筑师", "飞行员", "服务员", "科学家"],
            "颜色词汇": ["红色", "蓝色", "绿色", "黄色", "黑色", "白色", "紫色", "粉色", "橙色", "灰色"],
            "水果类别": ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "草莓", "柠檬", "梨", "桃子", "樱桃", "菠萝", "椰子", "杏子"],
            "动物世界": ["狗", "猫", "老虎", "狮子", "大象", "兔子", "熊猫", "鱼", "鸟", "鲸鱼", "乌龟", "蜜蜂", "蚂蚁", "蝴蝶", "蜻蜓", "考拉"],
            "世界首都": ["北京", "东京", "华盛顿", "巴黎", "伦敦", "柏林", "莫斯科", "罗马", "渥太华", "堪培拉", "巴西利亚", "新德里", "首尔", "曼谷", "开罗", "开普敦"],
            "季节时间": ["春天", "夏天", "秋天", "冬天", "白天", "夜晚", "早晨", "傍晚", "上午", "下午"],
            "情感表达": ["快乐", "悲伤", "愤怒", "恐惧", "爱情", "友情", "兴奋", "沮丧", "平静", "勇敢"],
            "地理地貌": ["高山", "小溪", "海洋", "湖泊", "森林", "草原", "沙漠", "雪山", "城市", "乡村"],
            "交通工具": ["汽车", "火车", "飞机", "轮船", "自行车", "摩托车", "公交车", "出租车", "地铁", "高铁"],
            "公共场所": ["医院", "学校", "银行", "邮局", "商店", "餐厅", "图书馆", "博物馆", "电影院", "剧院"],
            "运动项目": ["足球", "篮球", "网球", "乒乓球", "游泳", "跑步", "登山", "瑜伽", "太极拳", "武术"],
            "学科领域": ["数学", "物理", "化学", "生物", "历史", "地理", "文学", "艺术", "音乐", "舞蹈"],
            "电子设备": ["手机", "电脑", "电视", "收音机", "相机", "摄像机", "冰箱", "洗衣机", "空调", "电风扇"],
            "饮食营养": ["早餐", "午餐", "晚餐", "夜宵", "米饭", "面条", "蔬菜", "肉类", "牛奶", "鸡蛋"],
            "抽象概念": ["工作", "学习", "成功", "失败", "健康", "智慧", "财富", "幸福", "友谊", "包容"]
        }
        
        for category, words in semantic_categories.items():
            available_words = [word for word in words if word in vocab]
            if len(available_words) >= 2:
                print(f"\n'{category}' class similarity matrix:")
                print(f"Available words: {available_words}")

                # Compute average similarity within the category
                similarities = []
                for i, word1 in enumerate(available_words):
                    for j, word2 in enumerate(available_words):
                        if i < j:  # Avoid redundant calculations
                            try:
                                sim = self.model_skipgram.wv.similarity(word1, word2)
                                similarities.append(sim)
                                print(f"  {word1} ↔ {word2}: {sim:.3f}")
                            except:
                                pass
                
                if similarities:
                    avg_sim = np.mean(similarities)
                    print(f"  Average similarity: {avg_sim:.3f}")

    def visualize_semantic_space(self, categories=None):
        """ Visualize the semantic space using PCA and t-SNE """
        print(f"\n=== Semantic Space Visualization ===")

        vocab = list(self.model_skipgram.wv.key_to_index.keys())
        
        if categories is None:
            # 选择有代表性的词汇进行可视化
            target_words = []
            word_sets = [
                ["国王", "王后", "王子", "公主"],           # 王室
                ["父亲", "母亲", "儿子", "女儿", "爷爷", "奶奶"],  # 家庭
                ["红色", "蓝色", "绿色", "黄色", "白色", "黑色"],  # 颜色
                ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "草莓"],  # 水果
                ["医生", "老师", "工程师", "司机", "厨师", "程序员"],  # 职业
                ["北京", "东京", "华盛顿", "巴黎", "伦敦", "莫斯科"],  # 首都
                ["春天", "夏天", "秋天", "冬天", "白天", "夜晚"],    # 季节时间
                ["快乐", "悲伤", "愤怒", "恐惧", "爱情", "友情"],   # 情感
                ["狗", "猫", "老虎", "狮子", "大象", "熊猫"],      # 动物
                ["汽车", "火车", "飞机", "轮船", "自行车", "地铁"], # 交通工具
                ["足球", "篮球", "网球", "游泳", "跑步", "瑜伽"],   # 运动
                ["手机", "电脑", "电视", "相机", "冰箱", "空调"]    # 电子设备
            ]
            
            for word_set in word_sets:
                available = [w for w in word_set if w in vocab]
                if len(available) >= 2:
                    target_words.extend(available[:3])  # 每类最多3个词
        else:
            target_words = [word for word in categories if word in vocab]
        
        if len(target_words) < 5:
            print("可用词汇不足，跳过可视化")
            return
        
        print(f"可视化词汇: {target_words}")
        
        # 获取词向量
        vectors = np.array([self.model_skipgram.wv[word] for word in target_words])
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # PCA可视化
        pca = PCA(n_components=2, random_state=42)
        vectors_pca = pca.fit_transform(vectors)
        
        ax1 = axes[0]
        colors = plt.cm.tab20(np.linspace(0, 1, len(target_words)))
        
        for i, word in enumerate(target_words):
            ax1.scatter(vectors_pca[i, 0], vectors_pca[i, 1], 
                       c=[colors[i]], s=120, alpha=0.7)
            ax1.annotate(word, (vectors_pca[i, 0], vectors_pca[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        ax1.set_title('PCA降维可视化', fontsize=14)
        ax1.set_xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
        ax1.grid(True, alpha=0.3)
        
        # t-SNE可视化
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(target_words)-1))
        vectors_tsne = tsne.fit_transform(vectors)
        
        ax2 = axes[1]
        for i, word in enumerate(target_words):
            ax2.scatter(vectors_tsne[i, 0], vectors_tsne[i, 1], 
                       c=[colors[i]], s=120, alpha=0.7)
            ax2.annotate(word, (vectors_tsne[i, 0], vectors_tsne[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        ax2.set_title('t-SNE降维可视化', fontsize=14)
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_semantic_space_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("语义空间可视化已保存为: enhanced_semantic_space_visualization.png")        


def main():
    """主函数"""
    trainer = Word2VecTrainer()
    
    # Step 1: Load and preprocess data
    if not trainer.load_and_preprocess_data():
        return

    # Step 2: Train models
    trainer.train_models()

    # Step 3: Run enhanced analysis
    # vocab = trainer.analyze_vocabulary()

    # Step 4: Analyze semantic relations
    trainer.analyze_semantic_relations()

    # # Step 5: Analyze semantic clusters
    # trainer.analyze_semantic_clusters()

    # # Step 6: Visualize semantic space
    # trainer.visualize_semantic_space()


if __name__ == "__main__":
    main()

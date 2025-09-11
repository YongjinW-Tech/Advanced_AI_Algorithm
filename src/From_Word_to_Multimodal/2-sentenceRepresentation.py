import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib font for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Check transformers library and try to import
try:
    from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
    print("The Transformers library has been imported successfully.")
except ImportError:
    print("Please install the transformers library: pip install transformers")
    exit(1) # Exit if transformers is not available

def load_sentence_data(json_file='./sentence_data.json'):
    """
    Load sentence data from a JSON file.
    Args:
        json_file: JSON文件路径
    Returns:
        tuple: (sentences, categories, category_stats)
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sentences = []
        categories = []
        
        for item in data['sentences']:
            sentences.append(item['text'])
            categories.append(item['category'])
        
        # Calculate category statistics
        category_stats = {}
        for category in categories:
            # 利用字典的 get 方法实现一个简单的计数器逻辑
            # category_stats.get(category, 0)：尝试从字典 category_stats 中获取键 category 的值。
            # 如果键不存在，则返回默认值 0。
            # + 1：将获取到的值加 1
            category_stats[category] = category_stats.get(category, 0) + 1

        print(f"Loaded {len(sentences)} sentences")
        print("Category distribution:")
        for category, count in category_stats.items():
            print(f"  {category}: {count} sentences")

        return sentences, categories, category_stats
        
    except FileNotFoundError:
        print(f"Error: File not found {json_file}")
        print("Using default test sentences...")
        
        # Provide default sentences as fallback
        default_sentences = [
            "今天天气真不错，阳光明媚。",
            "我很喜欢在阳光下散步。",
            "雨天让人感到忧郁。",
            "The weather is beautiful today.",
            "I love walking in the sunshine.",
            "Apple released its first smartwatch in 2015.",
            "I like eating a fresh apple every morning.",
            "机器学习是人工智能的重要分支。"
        ]
        default_categories = ['天气', '天气', '天气', '天气', '天气', '科技', '饮食', '教育']
        default_stats = {'天气': 5, '科技': 1, '饮食': 1, '教育': 1}
        
        return default_sentences, default_categories, default_stats
    
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return [], [], {}
    
    except Exception as e:
        print(f"Error: Problem loading data - {e}")
        return [], [], {}

# Define the SentenceRepresentationExtractor class
class SentenceRepresentationExtractor:
    """
    基于 BERT 的句子表示提取器
    功能：
        1. 加载预训练BERT模型
        2. 提取句子级别的向量表示
        3. 分析句子语义相似性
        4. 可视化句子语义空间
    """
    
    def __init__(self, model_name='bert-base-chinese', max_length=512):
        """
        Init the extractor with pretrained model and tokenizer.
        Args:
            model_name: the name of the pretrained BERT model, e.g., 'bert-base-chinese'(default), 'bert-base-uncased'
            max_length: maximum token length for BERT input
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Use device: {self.device}")
        print(f"Loading model: {model_name}")

        # Create model save directory
        self.checkpoint_dir = './model-checkpoint'
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        self.tokenizer = None
        self.model = None
        self._load_model() # Call self defined loading model method
    
    def _load_model(self):
        """ Load the pretrained BERT model and tokenizer """
        try:
            # Try to load from local first
            local_model_path = os.path.join(self.checkpoint_dir, 'bert_model')
            if os.path.exists(local_model_path):
                print(f"Loading model from local: {local_model_path}")
                self.tokenizer = BertTokenizer.from_pretrained(local_model_path)
                self.model = BertModel.from_pretrained(local_model_path)
            else:
                print(f"Download model from Hugging Face: {self.model_name}")
                # If download fails, fallback to 'bert-base-uncased'
                try:
                    # Return a BertTokenizer Object, which includes the pre-trained BERT tokenizer
                    self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                    # Return a BertModel Object, which includes the pre-trained BERT model architecture and weights
                    self.model = BertModel.from_pretrained(self.model_name)
                except Exception as e:
                    print(f"Failed to download {self.model_name}, trying bert-base-uncased")
                    self.model_name = 'bert-base-uncased'
                    self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                    self.model = BertModel.from_pretrained(self.model_name)
                
                # Save the model locally
                self.tokenizer.save_pretrained(local_model_path)
                self.model.save_pretrained(local_model_path)
                print(f"Model has been saved to: {local_model_path}")

            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")

        except Exception as e:
            print(f"Model loading failed: {e}")
    
    def encode_sentences(self, sentences, pooling_strategy='cls'):
        """
        Encode sentences to vector representations.
        Args:
            sentences: List of sentences
            pooling_strategy: Pooling strategy ('cls', 'mean', 'max')
        Returns:
            numpy array: Sentence vector matrix [num_sentences, hidden_size]
        """

        print(f"Encoding {len(sentences)} sentences...")

        sentence_embeddings = []
        
        with torch.no_grad():
            for sentence in tqdm(sentences, desc="Encoding sentences"): # tqdm 是一个 Python 库，用于显示循环进度条，让用户能够直观地看到任务的执行进度。
                # Tokenization
                inputs = self.tokenizer(
                    sentence,                   # input text to be tokenized
                    padding=True,               # pad to the longest sequence in the batch
                    truncation=True,            # truncate sequences longer than max_length
                    max_length=self.max_length, # maximum length of the tokenized input sequence
                    return_tensors='pt'         # return PyTorch tensors
                ).to(self.device)
                
                # Forward pass
                # 使用 **inputs 字典解包调用模型，因为 Tokenizer 返回的 inputs 是一个字典
                # **inputs 会自动展开为：       
                # self.model(
                #     input_ids=inputs['input_ids'],
                #     attention_mask=inputs['attention_mask']
                # )
                outputs = self.model(**inputs)
                
                # 提取表示
                if pooling_strategy == 'cls':
                    # 使用[CLS] token的表示
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif pooling_strategy == 'mean':
                    # 使用所有token的平均表示
                    attention_mask = inputs['attention_mask'].cpu().numpy()
                    token_embeddings = outputs.last_hidden_state.cpu().numpy()
                    
                    # 计算掩码加权平均
                    input_mask_expanded = np.expand_dims(attention_mask, -1)
                    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
                    sum_mask = np.sum(input_mask_expanded, axis=1)
                    embedding = sum_embeddings / sum_mask
                elif pooling_strategy == 'max':
                    # 使用最大池化
                    embedding = np.max(outputs.last_hidden_state.cpu().numpy(), axis=1)
                else:
                    raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")

                sentence_embeddings.append(embedding[0])
        
        return np.array(sentence_embeddings)
    
    def analyze_semantic_similarity(self, sentences, embeddings):
        """
        Analyze semantic similarity between sentences.
        Args:
            sentences: List of sentences
            embeddings: Sentence embedding matrix
        """
        print("\n=== Sentence Semantic Similarity Analysis ===")

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Visualize similarity matrix
        plt.figure(figsize=(12, 10))

        # Create heatmap
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1) # k=1 means the upper triangle
        sns.heatmap(
            similarity_matrix,      # The data matrix to visualize
            mask=mask,              # Mask to hide the upper triangle
            annot=True,             # Show values in each cell
            fmt='.3f',              # Number format: keep 3 decimal places
            cmap='RdYlBu_r',        # Color map: Red-Yellow-Blue reversed
            # High similarity (close to 1) → Red/Dark Red; Medium similarity (around 0.5) → Yellow; Low similarity (close to 0) → Blue
            center=0,               # Center value for color mapping
            square=True,            # Make each cell square
            xticklabels=[f"S{i+1}" for i in range(len(sentences))], # X-axis labels
            yticklabels=[f"S{i+1}" for i in range(len(sentences))], # Y-axis labels
            cbar_kws={"shrink": .8} # Color bar size is 80% of the original
        )

        plt.title('Sentence Semantic Similarity Matrix', fontsize=16, pad=20)
        plt.xlabel('Sentence Index', fontsize=12)
        plt.ylabel('Sentence Index', fontsize=12)
        plt.tight_layout()
        plt.savefig('./results/sentence_similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 找出最相似和最不相似的句子对
        n_sentences = len(sentences)
        max_sim = -1
        min_sim = 2
        max_pair = None
        min_pair = None
        
        for i in range(n_sentences):
            for j in range(i+1, n_sentences):
                sim = similarity_matrix[i, j]
                if sim > max_sim:
                    max_sim = sim
                    max_pair = (i, j)
                if sim < min_sim:
                    min_sim = sim
                    min_pair = (i, j)

        print(f"\nThe most similar sentence pair (Similarity: {max_sim:.3f}):")
        if max_pair:
            print(f"  Sentence {max_pair[0]+1}: {sentences[max_pair[0]]}")
            print(f"  Sentence {max_pair[1]+1}: {sentences[max_pair[1]]}")

        print(f"\nThe most dissimilar sentence pair (Similarity: {min_sim:.3f}):")
        if min_pair:
            print(f"  Sentence {min_pair[0]+1}: {sentences[min_pair[0]]}")
            print(f"  Sentence {min_pair[1]+1}: {sentences[min_pair[1]]}")

        return similarity_matrix
    
    def visualize_semantic_space(self, sentences, embeddings, method='both'):
        """
        Visualize the semantic space of sentences
        Args:
            sentences: List of sentences
            embeddings: Sentence embedding matrix
            method: Visualization method ('pca', 'tsne', 'both')
        """
        print("\n=== Sentence Semantic Space Visualization ===")

        if method in ['pca', 'both']:
            # PCA降维
            pca = PCA(n_components=2, random_state=42)
            embeddings_pca = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                                s=100, alpha=0.7, c=range(len(sentences)), cmap='tab10')
            
            # 添加句子标签
            for i, sentence in enumerate(sentences):
                # 显示句子前20个字符
                label = sentence[:20] + "..." if len(sentence) > 20 else sentence
                plt.annotate(f"S{i+1}: {label}", 
                           (embeddings_pca[i, 0], embeddings_pca[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            plt.title('PCA降维后的句子语义空间', fontsize=14)
            plt.xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('./results/sentence_semantic_space_pca.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        if method in ['tsne', 'both'] and len(sentences) > 3:
            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, 
                       perplexity=min(5, len(sentences)-1))
            embeddings_tsne = tsne.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                                s=100, alpha=0.7, c=range(len(sentences)), cmap='tab10')
            
            # 添加句子标签
            for i, sentence in enumerate(sentences):
                label = sentence[:20] + "..." if len(sentence) > 20 else sentence
                plt.annotate(f"S{i+1}: {label}", 
                           (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            plt.title('t-SNE降维后的句子语义空间', fontsize=14)
            plt.xlabel('t-SNE 维度 1')
            plt.ylabel('t-SNE 维度 2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('./results/sentence_semantic_space_tsne.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def semantic_clustering(self, sentences, embeddings, n_clusters=3):
        """
        Perform semantic clustering on sentences

        Args:
            sentences: List of sentences
            embeddings: Sentence embedding matrix
            n_clusters: Number of clusters
        """
        print(f"\n=== Sentence Semantic Clustering (k={n_clusters}) ===")

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Display clustering results
        for cluster_id in range(n_clusters):
            cluster_sentences = [sentences[i] for i in range(len(sentences)) 
                               if cluster_labels[i] == cluster_id]
            print(f"\nCluster {cluster_id + 1}:")
            for i, sentence in enumerate(cluster_sentences):
                print(f"  {i+1}. {sentence}")

        # Visualize clustering results
        if len(sentences) > 3:
            pca = PCA(n_components=2, random_state=42)
            embeddings_pca = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                                c=cluster_labels, cmap='tab10', s=100, alpha=0.7)

            # Plot cluster centers
            centers_pca = pca.transform(kmeans.cluster_centers_)
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')

            # Add sentence labels
            for i, sentence in enumerate(sentences):
                label = sentence[:15] + "..." if len(sentence) > 15 else sentence
                plt.annotate(f"S{i+1}", 
                           (embeddings_pca[i, 0], embeddings_pca[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, alpha=0.8)

            plt.title('Sentence Semantic Clustering Results', fontsize=14)
            plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('./results/sentence_semantic_clustering.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return cluster_labels
    
    def analyze_contextual_sensitivity(self):
        """
        Analyze BERT's contextual sensitivity
        Show different representations of the same word in different contexts
        """
        print("\n=== Contextual Sensitivity Analysis ===")

        # Test cases: meanings of the same word in different contexts
        test_cases = [
            {
                "word": "方便",
                "sentences": [
                    "这个小卖部有方便面售卖吗？",
                    "请问最近的地铁站在哪里？很方便吗？",
                    "他是一个很方便的人",
                    "我们应该选择一个方便的时间开会",
                    "这个软件使用起来非常方便",
                    "我觉得住在市中心很方便"
                ]
            },
            {
                "word": "苹果",
                "sentences": [
                    "我喜欢吃苹果",
                    "苹果是我最喜欢的水果",
                    "苹果公司发布了新产品",
                    "我买了一个新的苹果手机",
                    "这个苹果很甜",
                    "她在苹果园工作"
                ]
            },
            {
                "word": "开放",
                "sentences": [
                    "图书馆今天开放",
                    "这个城市的公园开放时间很长",
                    "他是一个很开放的人",
                    "政府决定开放市场",
                    "开放心态有助于个人成长",
                    "我们应该开放心态去接受新事物"
                ]
            }
        ]
        
        for case in test_cases:
            word = case["word"]
            sentences = case["sentences"]

            print(f"\nAnalysis of '{word}' in Different Contexts:")

            # Encode sentences
            embeddings = self.encode_sentences(sentences, pooling_strategy='mean')

            # Compute similarity
            similarity_matrix = cosine_similarity(embeddings)

            print("The similarity scores are:")
            for i in range(len(sentences)):
                for j in range(i+1, len(sentences)):
                    sim = similarity_matrix[i, j]
                    print(f"  Sentence {i+1} vs Sentence {j+1}: {sim:.3f}")
                    print(f"    '{sentences[i]}'")
                    print(f"    '{sentences[j]}'")
            
            # Visualize in 2D space if there are at least 3 sentences
            if len(sentences) >= 3:
                pca = PCA(n_components=2, random_state=42)
                embeddings_pca = pca.fit_transform(embeddings)
                
                plt.figure(figsize=(10, 6))
                plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                           s=100, alpha=0.7, c=['red', 'blue', 'green', 'purple', 'orange', 'cyan'])
                
                for i, sentence in enumerate(sentences):
                    plt.annotate(f"S{i+1}: {sentence}", 
                               (embeddings_pca[i, 0], embeddings_pca[i, 1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, alpha=0.8,
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', alpha=0.7))
                
                plt.title(f"The Semantic Space of '{word}' in Different Contexts", fontsize=12)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'./results/contextual_analysis_{word}.png', dpi=300, bbox_inches='tight')
                plt.show()

def analyze_category_clustering(sentences, true_categories, cluster_labels, category_stats):
    """
    Analyze and compare true categories with clustering results.
    Args:
        sentences: List of sentences
        true_categories: List of true categories
        cluster_labels: List of cluster labels
        category_stats: Category statistics
    """
    print("=== True Categories vs Clustering Results ===")

    # Create a mapping from clusters to categories
    cluster_to_category = {}
    n_clusters = len(set(cluster_labels))

    print(f"\nClustering Results Analysis (Total {n_clusters} Clusters):")
    
    for cluster_id in range(n_clusters):
        # Find all sentences belonging to the current cluster
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_sentences = [sentences[i] for i in cluster_indices]
        cluster_categories = [true_categories[i] for i in cluster_indices]

        # Count the distribution of categories in the cluster
        category_count = {}
        for cat in cluster_categories:
            category_count[cat] = category_count.get(cat, 0) + 1

        # Find the dominant category
        dominant_category = max(category_count.items(), key=lambda x: x[1])

        print(f"\nCluster {cluster_id + 1} (Total {len(cluster_sentences)} Sentences):")
        print(f"  Dominant Category: {dominant_category[0]} ({dominant_category[1]}/{len(cluster_sentences)})")
        print(f"  Category Distribution: {dict(category_count)}")

        # Show a few example sentences
        print("  Example Sentences:")
        for i, (sentence, category) in enumerate(zip(cluster_sentences[:3], cluster_categories[:3])):
            print(f"    {i+1}. [{category}] {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
        if len(cluster_sentences) > 3:
            print(f"    ... {len(cluster_sentences) - 3} more sentences")

    # Calculate clustering purity
    print(f"\nClustering Quality Assessment:")
    total_sentences = len(sentences)
    correctly_clustered = 0
    
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_categories = [true_categories[i] for i in cluster_indices]
        
        if cluster_categories:
            # 找到最频繁的类别
            category_count = {}
            for cat in cluster_categories:
                category_count[cat] = category_count.get(cat, 0) + 1
            
            most_frequent_count = max(category_count.values())
            correctly_clustered += most_frequent_count
    
    purity = correctly_clustered / total_sentences
    print(f"Clustering Purity: {purity:.3f} ({correctly_clustered}/{total_sentences})")

    # Visualize category distribution
    plt.figure(figsize=(15, 6))

    # Subplot 1: True category distribution
    plt.subplot(1, 2, 1)
    categories = list(category_stats.keys())
    counts = list(category_stats.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    bars = plt.bar(range(len(categories)), counts, color=colors)
    plt.title('True Category Distribution', fontsize=14)
    plt.xlabel('Category')
    plt.ylabel('Number of Sentences')
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right')

    # Add value labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')

    # Subplot 2: Clustering results distribution
    plt.subplot(1, 2, 2)
    cluster_counts = [sum(1 for label in cluster_labels if label == i) for i in range(n_clusters)]
    cluster_names = [f'Cluster {i+1}' for i in range(n_clusters)]

    bars = plt.bar(range(n_clusters), cluster_counts, color=colors[:n_clusters])
    plt.title('Clustering Results Distribution', fontsize=14)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Sentences')
    plt.xticks(range(n_clusters), cluster_names, rotation=45, ha='right')

    # Add value labels
    for bar, count in zip(bars, cluster_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('./results/category_clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create confusion matrix style table
    plt.figure(figsize=(12, 8))

    # Create category to cluster mapping matrix
    category_list = list(category_stats.keys())
    matrix = np.zeros((len(category_list), n_clusters))
    
    for i, sentence_cat in enumerate(true_categories):
        cat_idx = category_list.index(sentence_cat)
        cluster_idx = cluster_labels[i]
        matrix[cat_idx, cluster_idx] += 1

    # Plot heatmap
    im = plt.imshow(matrix, cmap='Blues', aspect='auto')

    # Set axis labels
    plt.xticks(range(n_clusters), [f'Cluster {i+1}' for i in range(n_clusters)])
    plt.yticks(range(len(category_list)), category_list)
    plt.xlabel('Clustering Results')
    plt.ylabel('True Categories')
    plt.title('Category-Cluster Distribution Matrix')

    # Add value labels
    for i in range(len(category_list)):
        for j in range(n_clusters):
            if matrix[i, j] > 0:
                plt.text(j, i, int(matrix[i, j]), ha='center', va='center',
                        color='white' if matrix[i, j] > matrix.max()/2 else 'black')

    # Add color bar
    plt.colorbar(im, label='Number of Sentences')
    plt.tight_layout()
    plt.savefig('./results/category_cluster_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Sentence Representation Based on BERT ===\n")
    
    # Initialize the extractor with Chinese BERT model
    extractor = SentenceRepresentationExtractor(
                    model_name='bert-base-chinese', 
                    max_length=512
                )

    # Loading data
    print("=== Loading data ===")
    test_sentences, categories, category_stats = load_sentence_data('sentence_data.json')
    
    if not test_sentences:
        print("Loading data failed, exiting...")
        return

    print(f"\nTest sentences: {len(test_sentences)}")
    print(f"Categories: {len(category_stats)}")

    print("\nFirst 10 test sentences:")
    for i, sentence in enumerate(test_sentences[:10]):
        category = categories[i] if i < len(categories) else "Unknown"
        print(f"{i+1:2d}. [{category}] {sentence}")
    
    if len(test_sentences) > 10:
        print(f"... {len(test_sentences) - 10} more sentences")

    # Step 1: Extract sentence representations
    print(f"\n{'='*50}")
    print("Step 1: Extract sentence representations")
    sentence_embeddings = extractor.encode_sentences(test_sentences, pooling_strategy='cls')
    print(f"Sentence embeddings shape: {sentence_embeddings.shape}")

    # Step 2: Analyze semantic similarity
    print(f"\n{'='*50}")
    print("Step 2: Analyze semantic similarity")
    similarity_matrix = extractor.analyze_semantic_similarity(test_sentences, sentence_embeddings)

    # Step 3: Visualize semantic space
    print(f"\n{'='*50}")
    print("Step 3: Visualize semantic space")
    extractor.visualize_semantic_space(test_sentences, sentence_embeddings, method='both')

    # Step 4: Semantic Clustering
    print(f"\n{'='*50}")
    print("Step 4: Semantic Clustering")
    # Determine number of clusters based on category stats and sentence count
    n_clusters = min(len(category_stats), len(test_sentences) // 3, 10)  # 最多10个聚类
    print(f"数据类别数: {len(category_stats)}, 使用聚类数: {n_clusters}")
    cluster_labels = extractor.semantic_clustering(test_sentences, sentence_embeddings, n_clusters=n_clusters)

    # Step 5: Contextual Sensitivity Analysis
    print(f"\n{'='*50}")
    print("Step 5: Contextual Sensitivity Analysis")
    extractor.analyze_contextual_sensitivity()

    # Step 6: Category Analysis
    print(f"\n{'='*50}")
    print("Step 6: Analyze True Categories vs Clustering Results")
    analyze_category_clustering(test_sentences, categories, cluster_labels, category_stats)
    
    # Step 7: Compare different pooling strategies
    print(f"\n{'='*50}")
    print("Step 7: Compare different pooling strategies")

    pooling_strategies = ['cls', 'mean', 'max']
    sample_sentences = test_sentences[:5]  # Using first 5 sentences for quick comparison
    
    pooling_results = {}
    for strategy in pooling_strategies:
        print(f"\nUsing {strategy} pooling strategy:")
        embeddings = extractor.encode_sentences(sample_sentences, pooling_strategy=strategy)
        similarity = cosine_similarity(embeddings)

        # Compute average similarity
        n = len(sample_sentences)
        avg_similarity = (similarity.sum() - np.trace(similarity)) / (n * (n - 1))
        pooling_results[strategy] = avg_similarity
        print(f"Average sentence similarity: {avg_similarity:.3f}")

    print(f"\nPooling strategy comparison results:")
    for strategy, avg_sim in pooling_results.items():
        print(f"  {strategy:4s} pooling: {avg_sim:.3f}")

if __name__ == "__main__":
    main()
import json
import os
# 在所有其他导入之前设置环境变量
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 或者 "true"
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    set_seed
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import traceback
import logging
import warnings
warnings.filterwarnings("ignore")

# 配置日志记录
# Python logging 模块的默认行为是将 ERROR 级别的消息输出到控制台
# 我们通过 basicConfig 来配置日志记录
logging.basicConfig(
    level=logging.INFO, # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s', # 日志格式
    handlers=[ 
        logging.FileHandler('3-nextTokenPrediction.log', encoding='utf-8'),  # 保存到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class NextTokenPredictor:
    """Next Token Predictor"""
    
    def __init__(self, model_name: str = "facebook/opt-350m", device: str = None):
        """
        Initialize the model and tokenizer
        Args:
            model_name: Pre-trained model name
            device: Computing device
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logging.info("Initializing Next Token Predictor...")
        logging.info(f"Using device: {self.device}")
        logging.info(f"Model: {model_name}")
        
        # 加载分词器和模型
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        # 创建生成pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        logging.info("Model loading completed successfully")
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer"""
        try:
            logging.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # 设置pad_token（如果没有的话）
            # 在自然语言处理（NLP）任务中，pad_token 通常用于对齐不同长度的输入序列。
            # 例如，当我们将多个句子输入到模型中时，句子的长度可能不同。
            # 为了让它们具有相同的长度，我们会在较短的句子末尾填充一个特殊的标记（pad_token）。
            if tokenizer.pad_token is None:
                # eos_token 是一个特殊标记，通常表示句子的结束。
                # 这里的逻辑是，如果没有专门的填充标记，就复用句子结束标记作为填充标记。
                tokenizer.pad_token = tokenizer.eos_token
            logging.info("Tokenizer loaded successfully")
            return tokenizer
        except Exception as e:
            logging.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self) -> AutoModelForCausalLM:
        """Load model"""
        try:
            logging.info("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
                # device_map="auto"，可以自动将模型分配到多个设备（如多 GPU）
            )
            # 在深度学习中，模型通常有两种模式：
            # - 训练模式（Training Mode）：用于训练模型，启用诸如 dropout 和 batch normalization 的行为，这些操作在训练时会随机化或动态调整。
            # - 评估模式（Evaluation Mode）：用于评估或推理，关闭训练时的随机行为（如 dropout），并使用固定的参数（如 batch normalization 的均值和方差）。
            model.eval() # 设置模型为评估模式
            logging.info("Model loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def predict_next_token(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict the next token and its probability  
        Args:
            text: input text
            top_k: return top-k most likely tokens
        Returns:
            list of tokens and their probabilities
        """
        # Encode input text
        inputs = self.tokenizer.encode(
            text, 
            return_tensors="pt",
            truncation=True,      # 明确启用截断
            max_length=512        # 设置最大长度
        ).to(self.device)
        
        with torch.no_grad(): # 禁用梯度计算，只需要推理（预测），不需要反向传播
            # Get model output
            outputs = self.model(inputs)
            # outputs.logits 是模型的输出张量，通常是一个三维张量，形状为 (batch_size, sequence_length, vocab_size)：
            # - batch_size：表示输入批次的大小。
            # - sequence_length：表示输入序列的长度（即每个样本的 token 数量）。
            # - vocab_size：表示模型词汇表的大小（即可能的 token 数量）。
            # 从模型的输出中提取 第一个样本 [0, ...] 的 最后一个[... , -1, ...] token 的 词汇表分数 [..., ..., :]。
            # 提取的结果是一个一维张量，形状为 (vocab_size,)，表示模型对词汇表中每个 token 的预测分数。
            # 这段代码通常用于 下一词预测 或 语言模型生成，例如：
            # - 根据最后一个 token 的分数，通过 softmax 转换为概率分布。
            # - 从概率分布中采样或选择最高概率的 token，生成下一个词。
            logits = outputs.logits[0, -1, :]  # 最后一个position的logits
            
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            
            # 从概率分布中选出概率最高的 top_k 个值
            top_probs, top_indices = torch.topk(probs, top_k) # top_probs：对应的概率值；top_indices：对应的 token 索引。
            
            # 解码token
            results = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                token = self.tokenizer.decode([idx])
                results.append((token, float(prob)))
            
            return results

    def generate_text(self, prompt: str, max_length: int = 100,  max_new_tokens: int = 50,
                     temperature: float = 0.7, top_p: float = 0.9,
                     num_return_sequences: int = 1) -> List[str]:
        """
        Generate text based on the prompt
        Args:
            prompt: the input prompt
            max_length: the maximum length of the generated text
            temperature: the temperature parameter (controls randomness)
            top_p: the nucleus sampling parameter
            num_return_sequences: the number of generated sequences
        Returns:
            the list of generated texts
        """
        try:
            # Set generation parameters
            generation_params = {
                # "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "num_return_sequences": num_return_sequences,
                "do_sample": True, # when temperature > 0, we usually set do_sample=True
                "pad_token_id": self.tokenizer.eos_token_id, # padding token id
                "eos_token_id": self.tokenizer.eos_token_id, # end of sequence token id
                "return_full_text": False # only return the generated part
            }

            # Generate text
            # **generation_params: unpack the dictionary into keyword arguments
            outputs = self.generator(prompt, **generation_params)

            # Extract generated texts
            generated_texts = []
            for output in outputs:
                generated_text = output['generated_text']
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            logging.error(f"Text generation failed: {e}")
            return [f"Error: {str(e)}"]

    def analyze_generation_process(self, text: str, num_steps: int = 10) -> Dict:
        """
        Analyze the text generation process step by step
        
        Args:
            text: 起始文本
            num_steps: 分析步数
            
        Returns:
            生成过程分析结果
        """
        analysis_results = {
            "steps": [],
            "tokens": [],
            "probabilities": [],
            "generated_text": text
        }
        
        current_text = text
        
        for step in range(num_steps):
            # 预测下一个token
            next_tokens = self.predict_next_token(current_text, top_k=5)
            
            # 选择概率最高的token
            best_token, best_prob = next_tokens[0]
            
            # 记录信息
            analysis_results["steps"].append(step + 1)
            analysis_results["tokens"].append(best_token)
            analysis_results["probabilities"].append(best_prob)
            
            # 更新文本
            current_text += best_token
            analysis_results["generated_text"] = current_text
            
            logging.info(f"Step {step + 1}: '{best_token}' (probability: {best_prob:.4f})")
        
        return analysis_results
    
    def save_model_checkpoint(self, save_path: str):
        """保存模型检查点"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型和分词器
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存配置信息
        config = {
            "model_name": self.model_name,
            "device": self.device,
            "model_type": "causal_lm"
        }
        
        with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logging.info(f"Model has been saved to: {save_path}")

class TextGenerationExperiment:
    """Text Generation Experiment Class"""
    
    def __init__(self, data_file: str):
        """
        Initialize the experiment setup
        Args:
            data_file: Path to the data file
        """
        self.data_file = data_file
        self.data = self._load_data()
        self.predictor = None
        self.results = {}
    
    def _load_data(self) -> Dict:
        """Load experiment data"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Data loaded successfully, containing {len(data.get('prompts', []))} prompts")
            return data
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def setup_model(self, model_name: str = "facebook/opt-350m") -> None:
        """Set up the model"""
        logging.info(f"Setting up model: {model_name}")
        self.predictor = NextTokenPredictor(model_name)
    
    def run_generation_experiments(self) -> Dict:
        """Run text generation experiments with different parameters"""
        if not self.predictor:
            raise ValueError("Please set up the model \"NextTokenPredictor\" first")

        logging.info("Beginning text generation experiments...")
        
        results = {
            "prompts": [],
            "generated_texts": [],
            "generation_params": [],
            "analysis": []
        }
        
        # Define different generation parameter sets
        param_sets = [
            # temperature: 是一个控制生成文本随机性的参数。
            # - 值越低（如 0.3），生成的文本越保守，模型“更倾向于选择概率最高”的词语，输出更确定但可能缺乏多样性。
            # - 值越高（如 1.0），生成的文本越随机，模型“更可能选择概率较低”的词语，输出更有创意但可能不够连贯。
            # top_p: 是 核采样（nucleus sampling） 的参数，用于限制模型选择的词语范围。它的作用是限制模型在生成下一个词时，
            #        只考虑概率累积到某个阈值（即 top_p）的词汇集合，而忽略其他概率较低的词。
            # - 假设模型预测下一个词时，所有可能的词都有一个概率分布。
            #   top_p 会动态选择一个概率“累积”到 top_p 的子集（例如，前 90% 的概率质量），然后从这个子集中随机采样。
            # - 较高的 top_p（如 0.9）允许更多的词语参与采样，生成的文本更丰富。
            # - 较低的 top_p（如 0.8）限制了采样范围，生成的文本更集中。
            {"temperature": 0.3, "top_p": 0.9, "name": "保守生成"},
            {"temperature": 0.7, "top_p": 0.9, "name": "平衡生成"},
            {"temperature": 1.0, "top_p": 0.8, "name": "创意生成"}
        ]
        
        prompts = self.data.get("prompts", [])[:5]  # 取前5个提示进行实验
        # 如果 "prompts" 不存在于 self.data 中，则返回默认值 []（空列表）
        
        for i, prompt in enumerate(tqdm(prompts, desc="生成文本")):
            prompt_results = {
                "prompt": prompt,
                "generations": {}
            }
            
            for params in param_sets:
                # 生成文本
                generated = self.predictor.generate_text(
                    prompt=prompt,
                    max_length=len(prompt.split()) + 30,  # 动态设置长度
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    num_return_sequences=1  # 每次只生成一个序列
                )
                
                prompt_results["generations"][params["name"]] = {
                    "text": generated[0] if generated else "",
                    "params": params
                }
            
            results["prompts"].append(prompt)
            results["generated_texts"].append(prompt_results)
        
        self.results["generation"] = results
        return results
    
    def run_next_token_analysis(self) -> Dict:
        """ Next Token prediction analysis"""
        if not self.predictor:
            raise ValueError("Please set up the model \"NextTokenPredictor\" first")

        logging.info("Beginning Next Token prediction analysis...")

        analysis_results = {
            "token_predictions": [],
            "generation_processes": []
        }
        
        # Choose a few contexts for analysis
        contexts = self.data.get("contexts", [])[:3]

        for context in tqdm(contexts, desc="Analyzing predictions"):
            # Predict the next token
            next_tokens = self.predictor.predict_next_token(context, top_k=10)
            
            analysis_results["token_predictions"].append({
                "context": context,
                "predictions": next_tokens
            })
            
            # 分析生成过程
            generation_process = self.predictor.analyze_generation_process(
                context[:50] + "...",  # 使用context的前50个字符
                num_steps=5
            )
            
            analysis_results["generation_processes"].append({
                "initial_text": context[:50] + "...",
                "process": generation_process
            })
        
        self.results["analysis"] = analysis_results
        return analysis_results
    
    def visualize_results(self, save_dir: str):
        """可视化实验结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 可视化token预测概率
        self._plot_token_predictions(save_dir)
        
        # 2. 可视化生成过程
        self._plot_generation_process(save_dir)
        
        # 3. 生成参数对比
        self._plot_parameter_comparison(save_dir)
    
    def _plot_token_predictions(self, save_dir: str):
        """绘制token预测概率图"""
        if "analysis" not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        predictions = self.results["analysis"]["token_predictions"]
        
        for i, pred_data in enumerate(predictions[:4]):
            if i >= 4:
                break
                
            tokens = [item[0] for item in pred_data["predictions"]]
            probs = [item[1] for item in pred_data["predictions"]]
            
            # 清理token显示
            clean_tokens = [token.replace('\n', '\\n').replace(' ', '_') for token in tokens]
            
            axes[i].bar(range(len(clean_tokens)), probs, color='skyblue', alpha=0.7)
            axes[i].set_xlabel('Token排名')
            axes[i].set_ylabel('预测概率')
            axes[i].set_title(f'上下文 {i+1} 的Token预测')
            axes[i].set_xticks(range(len(clean_tokens)))
            axes[i].set_xticklabels(clean_tokens, rotation=45, ha='right')
            
            # 添加数值标签
            for j, prob in enumerate(probs):
                axes[i].text(j, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'token_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logging.info("Token prediction probability plot saved")

    def _plot_generation_process(self, save_dir: str):
        """绘制生成过程图"""
        if "analysis" not in self.results:
            return
        
        processes = self.results["analysis"]["generation_processes"]
        
        fig, axes = plt.subplots(len(processes), 1, figsize=(12, 4*len(processes)))
        if len(processes) == 1:
            axes = [axes]
        
        for i, process_data in enumerate(processes):
            process = process_data["process"]
            
            steps = process["steps"]
            probs = process["probabilities"]
            tokens = process["tokens"]
            
            # 绘制概率变化
            axes[i].plot(steps, probs, marker='o', linewidth=2, markersize=6)
            axes[i].set_xlabel('生成步骤')
            axes[i].set_ylabel('预测概率')
            axes[i].set_title(f'生成过程 {i+1}: 概率变化')
            axes[i].grid(True, alpha=0.3)
            
            # 添加token标签
            for j, (step, prob, token) in enumerate(zip(steps, probs, tokens)):
                clean_token = token.replace('\n', '\\n').replace(' ', '_')
                axes[i].annotate(clean_token, (step, prob), 
                               textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'generation_process.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Generation process plot saved")
    
    def _plot_parameter_comparison(self, save_dir: str):
        """绘制参数对比图"""
        if "generation" not in self.results:
            return
        
        # 统计不同参数设置的文本长度
        param_stats = {}
        
        for prompt_result in self.results["generation"]["generated_texts"]:
            for param_name, gen_data in prompt_result["generations"].items():
                if param_name not in param_stats:
                    param_stats[param_name] = []
                
                text_length = len(gen_data["text"].split())
                param_stats[param_name].append(text_length)
        
        # 绘制箱线图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 文本长度对比
        param_names = list(param_stats.keys())
        lengths_data = [param_stats[name] for name in param_names]
        
        ax1.boxplot(lengths_data, labels=param_names)
        ax1.set_ylabel('生成文本长度 (词数)')
        ax1.set_title('不同参数设置的文本长度对比')
        ax1.grid(True, alpha=0.3)
        
        # 参数设置对比
        temps = []
        top_ps = []
        names = []
        
        for prompt_result in self.results["generation"]["generated_texts"]:
            for param_name, gen_data in prompt_result["generations"].items():
                params = gen_data["params"]
                temps.append(params["temperature"])
                top_ps.append(params["top_p"])
                names.append(param_name)
        
        # 创建散点图
        unique_names = list(set(names))
        colors = ['red', 'blue', 'green']
        
        for i, name in enumerate(unique_names):
            mask = [n == name for n in names]
            temp_vals = [t for t, m in zip(temps, mask) if m]
            top_p_vals = [p for p, m in zip(top_ps, mask) if m]
            
            ax2.scatter(temp_vals, top_p_vals, label=name, 
                       color=colors[i % len(colors)], s=100, alpha=0.7)
        
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Top-p')
        ax2.set_title('生成参数设置对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'parameter_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logging.info("Parameter comparison plot saved")

    def save_results(self, save_dir: str):
        """保存实验结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存详细结果
        with open(os.path.join(save_dir, 'experiment_results.json'), 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 生成实验报告
        self._generate_report(save_dir)

        logging.info(f"Experiment results saved to: {save_dir}")

    def _generate_report(self, save_dir: str):
        """生成实验报告"""
        report = []
        report.append("# Next Token Prediction 实验报告\n")
        report.append(f"实验时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 模型信息
        if self.predictor:
            report.append("## 模型信息\n")
            report.append(f"- 模型名称: {self.predictor.model_name}\n")
            report.append(f"- 计算设备: {self.predictor.device}\n\n")
        
        # 实验概述
        if "generation" in self.results:
            gen_results = self.results["generation"]
            report.append("## 文本生成实验\n")
            report.append(f"- 测试提示数量: {len(gen_results['prompts'])}\n")
            report.append("- 参数设置: 保守生成、平衡生成、创意生成\n\n")
            
            # 展示部分生成结果
            report.append("### 生成示例\n")
            for i, prompt_result in enumerate(gen_results["generated_texts"][:2]):
                report.append(f"**提示 {i+1}**: {prompt_result['prompt']}\n\n")
                for param_name, gen_data in prompt_result["generations"].items():
                    report.append(f"- {param_name}: {gen_data['text'][:100]}...\n")
                report.append("\n")
        
        # Token预测分析
        if "analysis" in self.results:
            analysis_results = self.results["analysis"]
            report.append("## Next Token 预测分析\n")
            report.append(f"- 分析上下文数量: {len(analysis_results['token_predictions'])}\n")
            report.append(f"- 生成过程分析: {len(analysis_results['generation_processes'])}\n\n")
        
        # 核心发现
        report.append("## 核心发现\n")
        report.append("1. **温度参数影响**: 低温度生成更保守，高温度生成更有创意\n")
        report.append("2. **Next Token机制**: 模型基于上下文概率分布选择下一个词\n")
        report.append("3. **生成质量**: 模型能够生成连贯的中文文本\n")
        report.append("4. **概率分布**: Top-k预测显示了模型的不确定性\n\n")
        
        report.append("## 文件说明\n")
        report.append("- `experiment_results.json`: 详细实验数据\n")
        report.append("- `token_predictions.png`: Token预测概率可视化\n")
        report.append("- `generation_process.png`: 生成过程分析\n")
        report.append("- `parameter_comparison.png`: 参数设置对比\n")
        
        # 写入报告
        with open(os.path.join(save_dir, 'experiment_report.md'), 'w', encoding='utf-8') as f:
            f.writelines(report)

def main():
    # 设置调试模式
    DEBUG_MODE = True  # 开发时设为True，生产时设为False
    
    logging.info("Next Token Prediction experiment begins!")
    set_seed(42)

    # define paths 
    current_dir = os.path.dirname(os.path.abspath(__file__)) # get current directory
    data_file = os.path.join(current_dir, "text_generation_data.json")
    results_dir = os.path.join(current_dir, "results")
    checkpoint_dir = os.path.join(current_dir, "model-checkpoint")
    
    logging.info(f"Working directory: {current_dir}")
    logging.info(f"Data file: {data_file}")
    logging.info(f"Results directory: {results_dir}")
    logging.info(f"Checkpoint directory: {checkpoint_dir}")
    
    try:
        # 1. Initialize
        logging.info("Initializing experiment...")
        experiment = TextGenerationExperiment(data_file)

        # 2. Set up model (use a smaller model for quick experiments)
        # NextTokenPredictor is initialized here
        logging.info("Setting up model...")
        experiment.setup_model("facebook/opt-350m") 

        # 3. Text generation
        logging.info("Running text generation experiments...")
        generation_results = experiment.run_generation_experiments()

        # 4. Next Token analysis
        logging.info("Running Next Token prediction analysis...")
        analysis_results = experiment.run_next_token_analysis()
        
        # 5. Visualize results
        logging.info("Generating visualization results...")
        experiment.visualize_results(results_dir)
        
        # 6. Save results
        logging.info("Saving experiment results...")
        experiment.save_results(results_dir)

        # 7. Save model checkpoint
        logging.info("Saving model checkpoint...")
        experiment.predictor.save_model_checkpoint(checkpoint_dir)
        
        logging.info("=" * 50)
        logging.info("Next Token Prediction experiment completed!")
        logging.info(f"Results saved in: {results_dir}")
        logging.info(f"Model saved in: {checkpoint_dir}")
        
        # Print experiment summary
        if generation_results:
            logging.info(f"Completed {len(generation_results['prompts'])} text generation experiments")
        if analysis_results:
            logging.info(f"Completed {len(analysis_results['token_predictions'])} token prediction analyses")
        
    except Exception as e:
        if DEBUG_MODE:
            # 开发模式：显示完整错误
            logging.error(f"Error: {e}")
            logging.error("Error details:")
            logging.error("", exc_info=True)
        else:
            # 生产模式：只显示友好信息
            logging.error(f"The program encountered a problem: {e}")
            logging.error("Please check your network connection or contact technical support")
            
            # 同时记录详细错误到日志
            # exc_info=True 参数会同时输出完整的错误堆栈信息
            logging.error(f"Experiment failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()

# 运行过程中提示：
# Both `max_new_tokens` (=256) and `max_length`(=32) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation )

# 这个报错是什么原因
# 它的意思是：
# 你同时给模型传了 max_new_tokens=256 和 max_length=32 两个参数，但这两个参数在生成文本时只能有一个生效。
# 当前 max_new_tokens 优先级更高，所以 真正生效的是 256 个新 token，而 max_length=32 被忽略了。

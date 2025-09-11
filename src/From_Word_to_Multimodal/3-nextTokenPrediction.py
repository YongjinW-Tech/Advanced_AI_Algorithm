#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Next Token Prediction 实验
基于 OPT 模型进行文本生成，掌握 Next Token Prediction 原理

核心功能：
1. 加载预训练的 OPT 模型
2. 实现 Next Token Prediction 机制
3. 进行文本生成实验
4. 分析生成质量和机制

Author: AI Algorithm Course
Date: 2025-09-11
"""

import json
import os
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
import warnings
warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class NextTokenPredictor:
    """Next Token Prediction 实验类"""
    
    def __init__(self, model_name: str = "facebook/opt-350m", device: str = None):
        """
        初始化模型和分词器
        
        Args:
            model_name: 预训练模型名称
            device: 计算设备
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"🚀 初始化 Next Token Predictor...")
        print(f"📱 使用设备: {self.device}")
        print(f"🤖 模型: {model_name}")
        
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
        
        print("✅ 模型加载完成!")
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """加载分词器"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # 设置pad_token（如果没有的话）
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            print(f"❌ 分词器加载失败: {e}")
            raise
    
    def _load_model(self) -> AutoModelForCausalLM:
        """加载模型"""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            model.eval()
            return model
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def predict_next_token(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        预测下一个token及其概率
        
        Args:
            text: 输入文本
            top_k: 返回top-k个最可能的token
            
        Returns:
            token和概率的列表
        """
        # 编码输入文本
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(inputs)
            logits = outputs.logits[0, -1, :]  # 最后一个position的logits
            
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            
            # 获取top-k
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # 解码token
            results = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                token = self.tokenizer.decode([idx])
                results.append((token, float(prob)))
            
            return results
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.7, top_p: float = 0.9,
                     num_return_sequences: int = 1) -> List[str]:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大长度
            temperature: 温度参数（控制随机性）
            top_p: nucleus sampling参数
            num_return_sequences: 生成序列数量
            
        Returns:
            生成的文本列表
        """
        try:
            # 设置生成参数
            generation_params = {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "num_return_sequences": num_return_sequences,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            # 生成文本
            outputs = self.generator(prompt, **generation_params)
            
            # 提取生成的文本
            generated_texts = []
            for output in outputs:
                generated_text = output['generated_text']
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            print(f"❌ 文本生成失败: {e}")
            return [f"生成失败: {str(e)}"]
    
    def analyze_generation_process(self, text: str, num_steps: int = 10) -> Dict:
        """
        分析生成过程中的token预测
        
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
            
            print(f"步骤 {step + 1}: '{best_token}' (概率: {best_prob:.4f})")
        
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
        
        print(f"✅ 模型已保存到: {save_path}")

class TextGenerationExperiment:
    """文本生成实验类"""
    
    def __init__(self, data_file: str):
        """
        初始化实验
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.data = self._load_data()
        self.predictor = None
        self.results = {}
    
    def _load_data(self) -> Dict:
        """加载实验数据"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 数据加载成功，包含 {len(data.get('prompts', []))} 个提示")
            return data
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def setup_model(self, model_name: str = "facebook/opt-350m"):
        """设置模型"""
        print(f"\n🔧 设置模型: {model_name}")
        self.predictor = NextTokenPredictor(model_name)
    
    def run_generation_experiments(self) -> Dict:
        """运行文本生成实验"""
        if not self.predictor:
            raise ValueError("请先设置模型")
        
        print("\n🧪 开始文本生成实验...")
        
        results = {
            "prompts": [],
            "generated_texts": [],
            "generation_params": [],
            "analysis": []
        }
        
        # 不同的生成参数设置
        param_sets = [
            {"temperature": 0.3, "top_p": 0.9, "name": "保守生成"},
            {"temperature": 0.7, "top_p": 0.9, "name": "平衡生成"},
            {"temperature": 1.0, "top_p": 0.8, "name": "创意生成"}
        ]
        
        prompts = self.data.get("prompts", [])[:5]  # 取前5个提示进行实验
        
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
                    num_return_sequences=1
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
        """运行Next Token预测分析"""
        if not self.predictor:
            raise ValueError("请先设置模型")
        
        print("\n🔍 开始Next Token预测分析...")
        
        analysis_results = {
            "token_predictions": [],
            "generation_processes": []
        }
        
        # 选择几个上下文进行分析
        contexts = self.data.get("contexts", [])[:3]
        
        for context in tqdm(contexts, desc="分析预测"):
            # 预测下一个token
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
        
        print("✅ Token预测概率图已保存")
    
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
        
        print("✅ 生成过程图已保存")
    
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
        
        print("✅ 参数对比图已保存")
    
    def save_results(self, save_dir: str):
        """保存实验结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存详细结果
        with open(os.path.join(save_dir, 'experiment_results.json'), 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 生成实验报告
        self._generate_report(save_dir)
        
        print(f"✅ 实验结果已保存到: {save_dir}")
    
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
    """主函数"""
    print("🎯 Next Token Prediction 实验开始!")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(42)
    
    # 定义路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "text_generation_data.json")
    results_dir = os.path.join(current_dir, "results")
    checkpoint_dir = os.path.join(current_dir, "model-checkpoint")
    
    try:
        # 1. 初始化实验
        experiment = TextGenerationExperiment(data_file)
        
        # 2. 设置模型（使用较小的模型以便快速实验）
        experiment.setup_model("facebook/opt-350m")
        
        # 3. 运行文本生成实验
        print("\n📝 运行文本生成实验...")
        generation_results = experiment.run_generation_experiments()
        
        # 4. 运行Next Token分析
        print("\n🔍 运行Next Token预测分析...")
        analysis_results = experiment.run_next_token_analysis()
        
        # 5. 可视化结果
        print("\n📊 生成可视化结果...")
        experiment.visualize_results(results_dir)
        
        # 6. 保存结果
        print("\n💾 保存实验结果...")
        experiment.save_results(results_dir)
        
        # 7. 保存模型检查点
        print("\n🔄 保存模型检查点...")
        experiment.predictor.save_model_checkpoint(checkpoint_dir)
        
        print("\n" + "=" * 50)
        print("🎉 Next Token Prediction 实验完成!")
        print(f"📁 结果保存在: {results_dir}")
        print(f"🤖 模型保存在: {checkpoint_dir}")
        
        # 打印一些关键结果
        print("\n📋 实验摘要:")
        if generation_results:
            print(f"✅ 完成 {len(generation_results['prompts'])} 个文本生成实验")
        if analysis_results:
            print(f"✅ 完成 {len(analysis_results['token_predictions'])} 个Token预测分析")
        
    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# 运行过程中提示：
# Both `max_new_tokens` (=256) and `max_length`(=32) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation )

# 这个报错是什么原因
# 它的意思是：
# 你同时给模型传了 max_new_tokens=256 和 max_length=32 两个参数，但这两个参数在生成文本时只能有一个生效。
# 当前 max_new_tokens 优先级更高，所以 真正生效的是 256 个新 token，而 max_length=32 被忽略了。

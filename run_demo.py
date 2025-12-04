# run_demo.py
#!/usr/bin/env python3
"""
ToT-PromptOptimizer 演示脚本
"""

import asyncio
import json
import yaml
import os
import sys
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

from vllm_server import VLLMServer
from evaluator import PromptEvaluator, PaperData, SummaryEvaluator
from tot_optimizer import ToTPromptOptimizer
from component_lib import ComponentLibrary
from search_visualizer import visualize_search_tree
from prompt_builder import PromptBuilder

class ToTPromptOptimizerDemo:
    """ToT提示优化器演示类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化演示"""
        self.config_path = config_path
        self.config = self.load_config()
        
        # 创建结果目录
        self.results_dir = Path(self.config["data"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型实例
        self.target_llm = None
        self.evaluator_llm = None
        
        # 数据
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        # 组件库
        self.components = []
        
        # 优化器实例
        self.optimizer = None
        
    def load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    async def initialize_models(self):
        """初始化LLM模型"""
        print("=" * 60)
        print("初始化LLM模型")
        print("=" * 60)
        
        # 初始化目标LLM
        print(f"\n1. 初始化目标LLM...")
        target_config = self.config["models"]["target_model"]
        self.target_llm = VLLMServer(target_config)
        await self.target_llm.initialize()
        
        # 初始化评估LLM
        print(f"\n2. 初始化评估LLM...")
        evaluator_config = self.config["models"]["evaluator_model"]
        self.evaluator_llm = VLLMServer(evaluator_config)
        await self.evaluator_llm.initialize()
        
        print(f"\n✅ 模型初始化完成")
        print(f"   目标LLM: {target_config['path']} on {target_config['device']}")
        print(f"   评估LLM: {evaluator_config['path']} on {evaluator_config['device']}")
    
    def load_data(self):
        """加载数据"""
        print("\n" + "=" * 60)
        print("加载数据集")
        print("=" * 60)
        
        data_dir = Path(self.config["data"].get("cache_dir", "./data"))
        train_file_name = self.config["data"].get("train_file", "train_papers.json")
        test_file_name = self.config["data"].get("test_file", "test_papers.json")
        val_file_name = self.config["data"].get("val_file", "val_papers.json")
        
        # 加载训练数据
        train_file = data_dir / train_file_name
        if train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                train_dicts = json.load(f)
            self.train_data = [PaperData(**paper) for paper in train_dicts]
            print(f"✅ 加载训练数据: {len(self.train_data)} 篇论文")
        else:
            print(f"⚠️  训练数据文件不存在: {train_file}")
            # 生成模拟数据
            self.generate_mock_data()
        
        # 加载验证数据
        val_file = data_dir / test_file_name
        if not self.val_data and  val_file.exists():
            with open(val_file, 'r', encoding='utf-8') as f:
                val_dicts = json.load(f)
            self.val_data = [PaperData(**paper) for paper in val_dicts]
            print(f"✅ 加载验证数据: {len(self.val_data)} 篇论文")
        
        # 加载测试数据
        test_file = data_dir / val_file_name
        if not self.test_data and test_file.exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                test_dicts = json.load(f)
            self.test_data = [PaperData(**paper) for paper in test_dicts]
            print(f"✅ 加载测试数据: {len(self.test_data)} 篇论文")
    
    def generate_mock_data(self):
        """生成模拟数据（用于演示）"""
        print("生成模拟数据...")
        domains = ["Computer Vision", "NLP", "Reinforcement Learning"]
        train_size = int(self.config["data"].get("train_size", 50))
        val_size = int(self.config["data"].get("val_size", 15))
        test_size = int(self.config["data"].get("test_size", 10))

        papers = []
        for i in range(train_size+val_size+test_size):
            paper = PaperData(
                paper_id=f"demo_paper_{i:03d}",
                title=f"Deep Learning Approach for {domains[i % len(domains)]}",
                content=f"This is a demo paper about {domains[i % len(domains)]}. It proposes a novel method with significant improvements.",
                domain=domains[i % len(domains)],
                key_points=["novel method", "experimental validation", "performance improvement"],
                gold_summary=f"Demo paper {i} summary in {domains[i % len(domains)]}."
            )
            papers.append(paper)
        
            
        self.train_data = papers[:train_size]
        self.test_data = papers[train_size:train_size+test_size]
        self.val_data = papers[train_size+test_size:]
        
        print(f"✅ 生成模拟训练数据: {len(self.train_data)} 篇论文;验证数据：{len(self.val_data)};测试数据：{len(self.test_data)}")

    def load_components(self):
        """加载组件库"""
        print("\n" + "=" * 60)
        print("加载提示组件库")
        print("=" * 60)
        
        data_dir = Path(self.config["data"].get("cache_dir", "./data"))
        component_file_name = self.config["data"].get("components_file", "component_library.json")
        
        component_file = data_dir / component_file_name
        
        if component_file.exists():
            self.components = ComponentLibrary.load_from_file(str(component_file))
        else:
            self.components = ComponentLibrary.load_default()
            # 保存默认组件库
            ComponentLibrary.save_to_file(self.components, str(component_file))
        
        print(f"✅ 加载组件: {len(self.components)} 个")
        
        # 分析组件库
        analysis = ComponentLibrary.analyze_components(self.components)
        print(f"   冲突关系: {len(analysis['conflict_graph'])} 组")
        print(f"   依赖关系: {len(analysis['dependency_graph'])} 组")
        
        # 显示一些组件
        print(f"\n示例组件:")
        for i, comp in enumerate(self.components[:3]):
            print(f"  {i+1}. {comp['id']}: {comp['text']}")
    
    async def create_evaluator(self, split: str = "train") -> PromptEvaluator:
        """创建评估器"""
        if split == "train":
            papers = self.train_data
        elif split == "val":
            papers = self.val_data
        elif split == "test":
            papers = self.test_data
        else:
            papers = self.train_data
        
        evaluator = PromptEvaluator(
            target_llm=self.target_llm,
            evaluator_llm=self.evaluator_llm,
            papers=papers,
            config=self.config["evaluation"]
        )
        
        return evaluator
    
    async def run_optimization(self):
        """运行优化"""
        print("\n" + "=" * 60)
        print("开始Tree of Thoughts优化")
        print("=" * 60)
        
        # 创建评估器
        train_evaluator = await self.create_evaluator("train")
        
        # 创建优化器
        self.optimizer = ToTPromptOptimizer(
            components=self.components,
            evaluator=train_evaluator,
            config=self.config["optimization"]
        )
        
        # 运行优化
        await self.optimizer.optimize(
            iterations=self.config["optimization"]["max_iterations"],
            search_method="mcts"  # 或 "beam"
        )
        
        # 保存结果
        results_file = self.results_dir / "optimization_results.json"
        self.optimizer.save_results(str(results_file))
        
        # 可视化搜索树
        await self.visualize_results()
    
    async def visualize_results(self):
        """可视化结果"""
        print("\n" + "=" * 60)
        print("可视化结果")
        print("=" * 60)
        
        # 获取最佳提示
        best_prompt, best_details = self.optimizer.get_best_prompt()
        
        print(f"\n🎯 最佳提示:")
        print(f"   {best_prompt}")
        
        print(f"\n📊 评估分数: {best_details.get('score', 0):.3f}")
        
        if "metrics" in best_details:
            print(f"\n📈 详细指标:")
            metrics = best_details["metrics"]
            for metric, score in metrics.items():
                print(f"   {metric}: {score:.3f}")
        
        # 显示示例摘要
        if "summaries" in best_details and best_details["summaries"]:
            print(f"\n📝 示例生成的摘要:")
            print(f"   {best_details['summaries'][0][:200]}...")
        
        # 可视化搜索树
        try:
            tree_data = self.optimizer.get_search_tree()
            
            # 保存可视化
            vis_file = self.results_dir / "search_tree.html"
            visualize_search_tree(tree_data, str(vis_file))
            print(f"\n🌳 搜索树可视化已保存到: {vis_file}")
            
        except ImportError:
            print("\n⚠️  搜索树可视化需要额外的依赖")
        
        # 绘制优化曲线
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            history = self.optimizer.search_history
            if history:
                iterations = [h["iteration"] for h in history]
                scores = [h["best_score"] for h in history]
                
                plt.figure(figsize=(10, 6))
                plt.plot(iterations, scores, 'b-', linewidth=2, marker='o', markersize=4)
                plt.xlabel(' Iteration Time')
                plt.ylabel('Best Score')
                plt.title('ToT Optimization Progress')
                plt.grid(True, alpha=0.3)
                
                # 保存图表
                chart_file = self.results_dir / "optimization_progress.png"
                plt.savefig(str(chart_file), dpi=150, bbox_inches='tight')
                print(f"📈 优化进度图表已保存到: {chart_file}")
                
                plt.close()
        
        except ImportError:
            print("⚠️  图表绘制需要matplotlib")
    

    async def evaluate_on_test_set(self):
        """在测试集上评估最佳提示"""
        print("\n" + "=" * 60)
        print("测试集最终评估")
        print("=" * 60)
        
        if not self.optimizer:
            print("❌ 请先运行优化")
            return
        
        # 获取最佳提示
        best_prompt, _ = self.optimizer.get_best_prompt()
        
        # 创建测试集评估器
        test_evaluator = await self.create_evaluator("test")
        
        print(f"\n评估提示: {best_prompt}")
        
        # 评估
        test_result = await test_evaluator.evaluate_prompt(
            best_prompt,
            num_samples=min(5, len(self.test_data))
        )
        
        # 获取测试集中的gold summaries
        paper_ids = test_result.paper_ids
        generated_summaries = test_result.summaries
        gold_summaries = [ ]
        for p_id in paper_ids:
            for i in self.test_data:
                if i.id  == p_id :
                    gold_summaries.append(i.gold_summary)
                    break
         
        # 初始化摘要评估器
        summary_evaluator = SummaryEvaluator()
        summary_metrics = {}
        if len(generated_summaries) == len(gold_summaries):
            # print(f"generated_summaries:{len(generated_summaries)}, {generated_summaries}")
            # print(f"gold_summaries:{len(gold_summaries)}, {gold_summaries}")
            # 计算指标
            summary_metrics = summary_evaluator.compute_batch_metrics(
                    generated_summaries, 
                    gold_summaries
            )
            # summary_evaluator.print_result()

        
        print(f"\n📊 测试集结果:")
        print(f"   总体分数: {test_result.score:.3f}")
        
        print(f"\n📈 详细指标:")
        metrics_dict = test_result.metrics.to_dict()
        for metric, score in metrics_dict.items():
            if metric != "overall":
                print(f"   {metric}: {score:.3f}")
        
        print(f"\n📋 评估论文数: {len(test_result.paper_ids)}")
        print(f"   领域分布: {test_result.details.get('domain_distribution', {})}")
        
        summary_evaluator.print_result() ## 打印 summary metric

        # 保存测试结果
        test_results = {
            "prompt": best_prompt,
            "test_score": test_result.score,
            "metrics": metrics_dict,
            "paper_ids": test_result.paper_ids,
            "details": test_result.details,
            "summary_metrics":summary_metrics['statistics']
        }
        
        test_file = self.results_dir / "test_evaluation.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 测试结果已保存到: {test_file}")
    
    async def compare_with_baseline(self):
        """与基线提示比较"""
        print("\n" + "=" * 60)
        print("与基线提示比较")
        print("=" * 60)
        
        # 基线提示
        baseline_prompts = {
            "simple": "Summary this paper",
            "detailed": "Please provide a detailed summary of the paper's core content, methodology, and contributions.",
            "structured": "Please summarize the paper from four aspects: background, methodology, experiments, and conclusions."
        }
        
        # 创建评估器
        test_evaluator = await self.create_evaluator("test")
        
        # 获取优化后的最佳提示
        if self.optimizer:
            best_prompt, _ = self.optimizer.get_best_prompt()
            baseline_prompts["optimized"] = best_prompt
        
        results = {}
        
        print("\n评估不同提示...")
        for name, prompt in baseline_prompts.items():
            print(f"\n评估: {name}...")
            result = await test_evaluator.evaluate_prompt(
                prompt,
                num_samples=3
            )
            results[name] = {
                "prompt": prompt,
                "score": result.score,
                "metrics": result.metrics.to_dict()
            }
            
            print(f"  分数: {result.score:.3f}")
        
        # 显示比较结果
        print("\n" + "=" * 60)
        print("提示比较结果")
        print("=" * 60)
        
        print("\n排名:")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
        
        for i, (name, data) in enumerate(sorted_results):
            print(f"{i+1}. {name}: {data['score']:.3f}")
            print(f"   提示: {data['prompt'][:80]}...")
        
        # 保存比较结果
        compare_file = self.results_dir / "prompt_comparison.json"
        with open(compare_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 比较结果已保存到: {compare_file}")
    
    async def interactive_demo(self):
        """交互式演示"""
        print("\n" + "=" * 60)
        print("交互式演示")
        print("=" * 60)
        
        if not self.optimizer:
            print("❌ 请先运行优化")
            return
        
        best_prompt, best_details = self.optimizer.get_best_prompt()
        
        print(f"\n当前最佳提示: {best_prompt}")
        print(f"分数: {best_details.get('score', 0):.3f}")
        
        while True:
            print("\n选项:")
            print("1. 测试新论文")
            print("2. 查看搜索树")
            print("3. 修改组件库")
            print("4. 运行更多优化")
            print("5. 退出")
            
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == "1":
                await self.test_custom_paper(best_prompt)
            elif choice == "2":
                await self.explore_search_tree()
            elif choice == "3":
                self.modify_component_library()
            elif choice == "4":
                more_iterations = int(input("输入额外迭代次数: ") or "10")
                await self.optimizer.optimize(iterations=more_iterations)
                best_prompt, best_details = self.optimizer.get_best_prompt()
                print(f"\n新最佳提示: {best_prompt}")
                print(f"新分数: {best_details.get('score', 0):.3f}")
            elif choice == "5":
                break
            else:
                print("无效选择")
    
    async def test_custom_paper(self, prompt: str):
        """测试自定义论文"""
        print("\n输入论文内容（输入END结束）:")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        
        paper_content = "\n".join(lines)
        
        if not paper_content:
            paper_content = "This is a paper on deep learning that proposes a new neural network architecture."
        
        full_prompt = f"{prompt}\n\nPaper Content：\n{paper_content}"
        print(f"\n生成摘要...")
        summary = await self.target_llm.generate([full_prompt])
        
        if summary:
            print(f"\n📝 生成的摘要:")
            print(f"{summary[0]}")
        else:
            print("❌ 生成失败")
    
    async def explore_search_tree(self):
        """探索搜索树"""
        if not self.optimizer:
            print("❌ 优化器未初始化")
            return
        
        tree = self.optimizer.get_search_tree()
        
        print(f"\n搜索树统计:")
        print(f"   总节点数: {len(tree['nodes'])}")
        print(f"   边数: {len(tree['edges'])}")
        
        # 显示高分节点
        nodes = list(tree["nodes"].items())
        nodes.sort(key=lambda x: x[1]["score"], reverse=True)
        
        print(f"\nTop 5 节点:")
        for i, (node_id, node_data) in enumerate(nodes[:5]):
            state = node_data["state"]
            state_str = " -> ".join(state) if state else "[空]"
            print(f"{i+1}. {state_str} (分数: {node_data['score']:.3f})")
    
    def modify_component_library(self):
        """修改组件库"""
        print(f"\n当前组件数: {len(self.components)}")
        print("1. 添加组件")
        print("2. 删除组件")
        print("3. 查看组件")
        print("4. 返回")
        
        choice = input("选择: ").strip()
        
        if choice == "1":
            comp_id = input("组件ID: ").strip()
            comp_text = input("组件文本: ").strip()
            
            new_component = {
                "id": comp_id,
                "text": comp_text,
                "effect_vector": {"conciseness": 0.5, "completeness": 0.5},
                "conflicts": [],
                "requires": [],
                "estimated_tokens": len(comp_text.split())
            }
            
            self.components.append(new_component)
            print(f"✅ 添加组件: {comp_id}")
        
        elif choice == "2":
            print("组件列表:")
            for i, comp in enumerate(self.components):
                print(f"{i+1}. {comp['id']}: {comp['text']}")
            
            idx = int(input("要删除的组件编号: ").strip()) - 1
            if 0 <= idx < len(self.components):
                removed = self.components.pop(idx)
                print(f"✅ 删除组件: {removed['id']}")
        
        elif choice == "3":
            print("\n组件详情:")
            for comp in self.components:
                print(f"\n{comp['id']}:")
                print(f"  文本: {comp['text']}")
                print(f"  效果: {comp.get('effect_vector', {})}")
                print(f"  冲突: {comp.get('conflicts', [])}")
                print(f"  依赖: {comp.get('requires', [])}")
    
    async def run_full_demo(self):
        """运行完整演示"""
        print("🚀 启动 ToT-PromptOptimizer 演示")
        print("=" * 60)
        
        # 1. 初始化
        await self.initialize_models()
        
        # 2. 加载数据
        self.load_data()
        
        # 3. 加载组件
        self.load_components()
        
        # 4. 运行优化
        await self.run_optimization()
        
        # 5. 测试集评估
        await self.evaluate_on_test_set()
        
        # 6. 与基线比较
        await self.compare_with_baseline()
        
        # # 7. 交互式演示（可选）
        # interactive = input("\n是否进入交互式演示? (y/n): ").strip().lower()
        # if interactive == 'y':
        #     await self.interactive_demo()
        
        print("\n" + "=" * 60)
        print("演示完成!")
        print("=" * 60)
        
        # 显示最终统计
        if self.target_llm:
            target_stats = self.target_llm.get_stats()
            print(f"\n目标LLM统计:")
            print(f"   调用次数: {target_stats['call_count']}")
            print(f"   生成token数: {target_stats['total_tokens']}")
        
        if self.optimizer:
            print(f"\n优化统计:")
            stats = self.optimizer.stats
            for key, value in stats.items():
                print(f"   {key}: {value}")

async def main():
    """主函数"""
    demo = ToTPromptOptimizerDemo("config.yaml")
    
    try:
        await demo.run_full_demo()
    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 检查并生成数据
    data_dir = Path("./data")
    if not (data_dir / "train_papers.json").exists():
        print("生成模拟数据...")
        import generate_data
        generate_data.save_data({})  # 这会生成数据
        
    # 运行演示
    asyncio.run(main())
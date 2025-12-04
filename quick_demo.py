# quick_demo.py
"""
快速演示脚本 - 不需要真实LLM模型
"""

import asyncio
import json
from pathlib import Path
import sys

# 添加src目录
sys.path.append(str(Path(__file__).parent / "src"))

from vllm_server import MockLLM
from evaluator import PromptEvaluator, PaperData
from tot_optimizer import ToTPromptOptimizer
from component_lib import ComponentLibrary

async def quick_demo():
    """快速演示"""
    print("🚀 ToT-PromptOptimizer 快速演示")
    print("=" * 60)
    
    # 使用模拟LLM
    target_llm = MockLLM("target-mock")
    evaluator_llm = MockLLM("evaluator-mock")
    
    # 创建模拟数据
    papers = []
    domains = ["CV", "NLP", "RL"]
    
    for i in range(20):
        paper = PaperData(
            paper_id=f"paper_{i:03d}",
            content=f"这是{i}号论文，关于{domains[i % len(domains)]}。提出了新方法，实验结果良好。",
            domain=domains[i % len(domains)],
            key_points=["创新方法", "实验验证", "性能提升"],
            gold_summary=f"论文{i}的摘要。"
        )
        papers.append(paper)
    
    # 加载组件
    components = ComponentLibrary.load_default()
    
    # 创建评估器
    evaluator = PromptEvaluator(
        target_llm=target_llm,
        evaluator_llm=evaluator_llm,
        papers=papers,
        config={
            "samples_per_eval": 2,
            "metric_weights": {
                "faithfulness": 0.25, "conciseness": 0.20, 
                "completeness": 0.25, "readability": 0.15, 
                "insightfulness": 0.15
            }
        }
    )
    
    # 创建优化器
    optimizer = ToTPromptOptimizer(
        components=components,
        evaluator=evaluator,
        config={
            "max_iterations": 15,
            "beam_width": 2,
            "exploration_weight": 1.41,
            "max_depth": 4,
            "max_tokens": 150,
            "task_description": "总结一篇技术论文",
            "train_samples_per_eval": 2
        }
    )
    
    # 运行优化
    print("\n开始优化...")
    await optimizer.optimize(iterations=10, search_method="beam")
    
    # 显示结果
    best_prompt, best_details = optimizer.get_best_prompt()
    
    print("\n" + "=" * 60)
    print("优化结果")
    print("=" * 60)
    
    print(f"\n🎯 最佳提示:")
    print(f"   {best_prompt}")
    
    print(f"\n📊 分数: {best_details.get('score', 0):.3f}")
    
    if "metrics" in best_details:
        print(f"\n📈 详细指标:")
        for metric, score in best_details["metrics"].items():
            print(f"   {metric}: {score:.3f}")
    
    print(f"\n📊 优化统计:")
    for key, value in optimizer.stats.items():
        print(f"   {key}: {value}")
    
    # 保存结果
    results = {
        "best_prompt": best_prompt,
        "best_details": best_details,
        "stats": optimizer.stats
    }
    
    with open("quick_demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到: quick_demo_results.json")
    print("\n演示完成!")

if __name__ == "__main__":
    asyncio.run(quick_demo())
# generate_data.py
import json
import random
from typing import List, Dict, Any


def generate_paper_data(num_papers: int = 100) -> List[Dict[str, Any]]:
    """生成模拟论文数据 - 英文版本"""
    
    domains = [
        "Computer Vision", "Natural Language Processing", "Reinforcement Learning", 
        "Machine Learning Theory", "Data Mining", "Computer Graphics",
        "Human-Computer Interaction", "Information Security", "Computational Biology"
    ]
    
    paper_templates = [
        {
            "template": "This paper proposes a {method}-based approach for {task}, named {model_name}. The method combines {technique1} and {technique2}, achieving {improvement}% performance improvement on the {dataset} dataset. Key innovations include: 1) {innovation1}; 2) {innovation2}; 3) {innovation3}.",
            "methods": ["Transformer", "CNN", "RNN", "GNN", "Diffusion Model", "Meta-Learning"],
            "tasks": ["image classification", "object detection", "semantic segmentation", "machine translation", "text generation", "game AI"],
            "techniques": ["attention mechanism", "residual connections", "batch normalization", "dropout", "data augmentation", "curriculum learning"],
            "datasets": ["ImageNet", "COCO", "GLUE", "Atari", "MNIST", "CIFAR-10"],
            "improvements": [5, 10, 15, 20, 25, 30],
            "innovations": [
                "novel architecture design",
                "efficient training strategy", 
                "improved loss function",
                "innovative data preprocessing method",
                "multi-task learning framework",
                "self-supervised learning paradigm"
            ]
        },
        {
            "template": "This study explores {approach} methods for the {problem} problem. Through {method}, we demonstrate that {insight}. Experimental results show that the method achieves {score} on the {metric} metric, representing a {improvement}% improvement over baselines.",
            "problems": ["model generalization", "training stability", "computational efficiency", "sample complexity", "adversarial robustness", "interpretability"],
            "approaches": ["theoretical analysis", "empirical study", "hybrid methods", "end-to-end learning", "multi-stage optimization"],
            "methods": ["mathematical proof", "extensive experiments", "simulation validation", "user studies", "case analysis"],
            "insights": [
                "log-linear relationship between model scale and performance",
                "critical role of attention mechanisms",
                "data quality is more important than quantity",
                "conditions for effective pre-training",
                "importance of regularization techniques"
            ],
            "metrics": ["accuracy", "F1 score", "BLEU score", "reward", "human rating", "inference time"],
            "scores": ["98.5%", "0.95", "45.2", "8500", "4.8/5.0", "23ms"],
            "improvements": [12, 18, 25, 32, 40, 50]
        }
    ]
    
    papers = []
    
    for i in range(num_papers):
        # 选择模板
        template = random.choice(paper_templates)
        
        # 填充模板
        domain = random.choice(domains)
        
        if template["template"] == paper_templates[0]["template"]:
            content = template["template"].format(
                method=random.choice(template["methods"]),
                task=random.choice(template["tasks"]),
                model_name=f"{random.choice(['Deep', 'Smart', 'Fast', 'Robust'])}Net",
                technique1=random.choice(template["techniques"]),
                technique2=random.choice(template["techniques"]),
                dataset=random.choice(template["datasets"]),
                improvement=random.choice(template["improvements"]),
                innovation1=random.choice(template["innovations"]),
                innovation2=random.choice(template["innovations"]),
                innovation3=random.choice(template["innovations"])
            )
            
            # 生成关键点
            key_points = [
                f"Proposes a novel {random.choice(template['methods'])} architecture",
                f"Demonstrates superior performance on {random.choice(template['datasets'])}",
                f"Achieves {random.choice(template['improvements'])}% improvement over baselines"
            ]
            
        else:
            content = template["template"].format(
                problem=random.choice(template["problems"]),
                approach=random.choice(template["approaches"]),
                method=random.choice(template["methods"]),
                insight=random.choice(template["insights"]),
                metric=random.choice(template["metrics"]),
                score=random.choice(template["scores"]),
                improvement=random.choice(template["improvements"])
            )
            
            key_points = [
                f"Investigates {random.choice(template['problems'])} problem",
                f"Uses {random.choice(template['approaches'])} approach",
                f"Key finding: {random.choice(template['insights'])[:50]}..."
            ]
        
        # 添加更多内容
        content += "\n\nRelated work discusses limitations of existing methods."
        content += "\n\nMethodology section details the proposed algorithm and technical details."
        content += "\n\nExperimental section presents results on multiple benchmark tests."
        content += "\n\nConclusion summarizes main contributions and future work directions."
        
        # 生成论文ID
        paper_id = f"paper_{i+1:03d}"
        
        # 生成参考摘要
        gold_summary = f"This paper proposes an innovative method that achieves significant progress in the field of {domain}. Main contributions include proposing a new architecture, designing efficient algorithms, and conducting extensive experimental validation. The method achieves state-of-the-art results on multiple benchmark tests."
        
        papers.append({
            "id": paper_id,
            "content": content,
            "domain": domain,
            "key_points": key_points,
            "gold_summary": gold_summary
        })
    
    return papers


def split_data(papers: List[Dict[str, Any]], 
               train_ratio: float = 0.7,
               val_ratio: float = 0.15) -> Dict[str, List[Dict[str, Any]]]:
    """划分数据集"""
    random.shuffle(papers)
    
    n = len(papers)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_papers = papers[:n_train]
    val_papers = papers[n_train:n_train + n_val]
    test_papers = papers[n_train + n_val:]
    
    return {
        "train": train_papers,
        "val": val_papers,
        "test": test_papers
    }

def save_data(data: Dict[str, List[Dict[str, Any]]], output_dir: str = "./data"):
    """保存数据到文件"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, papers in data.items():
        filepath = os.path.join(output_dir, f"{split_name}_papers.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(papers)} 篇论文到 {filepath}")
    
    # 保存组件库
    from src.component_lib import ComponentLibrary
    components = ComponentLibrary.load_default()
    component_file = os.path.join(output_dir, "component_library.json")
    ComponentLibrary.save_to_file(components, component_file)
    print(f"已保存 {len(components)} 个组件到 {component_file}")

if __name__ == "__main__":
    # 生成数据
    print("生成模拟论文数据...")
    papers = generate_paper_data(num_papers=100)
    
    # 划分数据集
    print("划分数据集...")
    split_data_dict = split_data(papers, train_ratio=0.6, val_ratio=0.2)
    
    # 保存数据
    print("保存数据到文件...")
    save_data(split_data_dict)
    
    print("数据生成完成!")
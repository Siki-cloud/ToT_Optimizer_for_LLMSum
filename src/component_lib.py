# src/component_lib.py
import json
from typing import List, Dict, Any

class ComponentLibrary:
    """英文提示组件库管理"""
    DEFAULT_COMPONENTS = [
        {
            "id": "conciseness",
            "text": "Please use concise language",
            "description": "Requires concise output",
            "effect_vector": {
                "conciseness": 0.8,
                "readability": 0.6,
                "completeness": -0.2
            },
            "conflicts": ["detail_oriented", "comprehensive"],
            "requires": [],
            "estimated_tokens": 8
        },
        {
            "id": "three_points",
            "text": "Please discuss in three main points",
            "description": "Structure output into three points",
            "effect_vector": {
                "completeness": 0.7,
                "readability": 0.5,
                "structure": 0.8
            },
            "conflicts": ["one_sentence"],
            "requires": [],
            "estimated_tokens": 10
        },
        {
            "id": "compare_works",
            "text": "Please compare with related work",
            "description": "Requires comparison with related work",
            "effect_vector": {
                "insightfulness": 0.9,
                "depth": 0.7,
                "completeness": 0.3
            },
            "conflicts": [],
            "requires": ["conciseness"],
            "estimated_tokens": 12
        },
        {
            "id": "key_formulas",
            "text": "Please list key formulas",
            "description": "Requires listing key formulas",
            "effect_vector": {
                "technical": 0.8,
                "precision": 0.7,
                "readability": -0.4
            },
            "conflicts": ["simple_language"],
            "requires": [],
            "estimated_tokens": 10
        },
        {
            "id": "metaphor",
            "text": "Please explain using simple metaphors",
            "description": "Explain complex concepts using metaphors",
            "effect_vector": {
                "readability": 0.9,
                "accessibility": 0.8,
                "technical": -0.5
            },
            "conflicts": ["key_formulas", "technical_terms"],
            "requires": [],
            "estimated_tokens": 12
        },
        {
            "id": "detail_oriented",
            "text": "Please provide a detailed and comprehensive summary",
            "description": "Requires detailed summary",
            "effect_vector": {
                "completeness": 0.9,
                "depth": 0.8,
                "conciseness": -0.7
            },
            "conflicts": ["conciseness", "one_sentence"],
            "requires": [],
            "estimated_tokens": 15
        },
        {
            "id": "one_sentence",
            "text": "Please summarize the core contribution in one sentence",
            "description": "Requires one-sentence summary",
            "effect_vector": {
                "conciseness": 0.9,
                "focus": 0.8,
                "completeness": -0.6
            },
            "conflicts": ["three_points", "detail_oriented"],
            "requires": [],
            "estimated_tokens": 15
        },
        {
            "id": "technical_terms",
            "text": "Please use professional academic terminology",
            "description": "Requires professional terminology",
            "effect_vector": {
                "technical": 0.9,
                "precision": 0.7,
                "readability": -0.5
            },
            "conflicts": ["metaphor", "simple_language"],
            "requires": [],
            "estimated_tokens": 12
        },
        {
            "id": "simple_language",
            "text": "Please use simple and understandable language",
            "description": "Requires simple language",
            "effect_vector": {
                "readability": 0.9,
                "accessibility": 0.8,
                "technical": -0.4
            },
            "conflicts": ["key_formulas", "technical_terms"],
            "requires": [],
            "estimated_tokens": 12
        },
        {
            "id": "structure",
            "text": "Please structure the summary with clear sections",
            "description": "Requires structured sections",
            "effect_vector": {
                "structure": 0.9,
                "readability": 0.6,
                "completeness": 0.5
            },
            "conflicts": ["one_sentence"],
            "requires": [],
            "estimated_tokens": 12
        },
        {
            "id": "limitations",
            "text": "Please include limitations of the work",
            "description": "Requires discussing limitations",
            "effect_vector": {
                "critical": 0.8,
                "completeness": 0.4,
                "insightfulness": 0.6
            },
            "conflicts": [],
            "requires": ["conciseness"],
            "estimated_tokens": 10
        },
        {
            "id": "future_work",
            "text": "Please suggest future research directions",
            "description": "Requires suggesting future work",
            "effect_vector": {
                "insightfulness": 0.8,
                "completeness": 0.3,
                "forward_looking": 0.9
            },
            "conflicts": [],
            "requires": ["conciseness"],
            "estimated_tokens": 12
        },
        {
            "id": "key_findings",
            "text": "Please highlight key findings",
            "description": "Requires highlighting key findings",
            "effect_vector": {
                "focus": 0.8,
                "readability": 0.5,
                "completeness": 0.4
            },
            "conflicts": [],
            "requires": [],
            "estimated_tokens": 10
        },
        {
            "id": "methodology",
            "text": "Please describe the methodology in detail",
            "description": "Requires detailed methodology description",
            "effect_vector": {
                "technical": 0.7,
                "completeness": 0.6,
                "conciseness": -0.5
            },
            "conflicts": ["conciseness"],
            "requires": [],
            "estimated_tokens": 12
        },
        {
            "id": "applications",
            "text": "Please discuss practical applications",
            "description": "Requires discussing applications",
            "effect_vector": {
                "practical": 0.9,
                "insightfulness": 0.6,
                "completeness": 0.3
            },
            "conflicts": [],
            "requires": ["conciseness"],
            "estimated_tokens": 12
        }
    ]
    
    @classmethod
    def load_default(cls) -> List[Dict[str, Any]]:
        """加载默认组件库"""
        return cls.DEFAULT_COMPONENTS
    
    @classmethod
    def load_from_file(cls, filepath: str) -> List[Dict[str, Any]]:
        """从文件加载组件库"""
        with open(filepath, 'r', encoding='utf-8') as f:
            components = json.load(f)
        return components
    
    @classmethod
    def save_to_file(cls, components: List[Dict[str, Any]], filepath: str):
        """保存组件库到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(components, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def analyze_components(cls, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析组件库"""
        analysis = {
            "total_components": len(components),
            "domains": set(),
            "conflict_graph": {},
            "dependency_graph": {},
            "effect_stats": {}
        }
        
        # 收集所有效果向量维度
        all_effects = set()
        for comp in components:
            effects = comp.get("effect_vector", {})
            all_effects.update(effects.keys())
        
        # 初始化效果统计
        for effect in all_effects:
            analysis["effect_stats"][effect] = {
                "min": float('inf'),
                "max": -float('inf'),
                "avg": 0.0,
                "count": 0
            }
        
        # 分析每个组件
        for comp in components:
            comp_id = comp["id"]
            
            # 冲突图
            conflicts = comp.get("conflicts", [])
            if conflicts:
                analysis["conflict_graph"][comp_id] = conflicts
            
            # 依赖图
            requires = comp.get("requires", [])
            if requires:
                analysis["dependency_graph"][comp_id] = requires
            
            # 效果统计
            effects = comp.get("effect_vector", {})
            for effect, value in effects.items():
                stats = analysis["effect_stats"][effect]
                stats["min"] = min(stats["min"], value)
                stats["max"] = max(stats["max"], value)
                stats["avg"] += value
                stats["count"] += 1
        
        # 计算平均效果
        for effect in all_effects:
            stats = analysis["effect_stats"][effect]
            if stats["count"] > 0:
                stats["avg"] /= stats["count"]
        
        return analysis
    
    @classmethod
    def get_component_categories(cls) -> Dict[str, List[Dict[str, Any]]]:
        """获取按类别分组的组件"""
        categories = {
            "style": [],      # 风格相关
            "structure": [],  # 结构相关
            "content": [],    # 内容相关
            "depth": []       # 深度相关
        }
        
        components = cls.load_default()
        
        style_ids = ["conciseness", "simple_language", "technical_terms"]
        structure_ids = ["three_points", "structure", "one_sentence"]
        content_ids = ["key_formulas", "key_findings", "methodology", "applications"]
        depth_ids = ["compare_works", "limitations", "future_work", "detail_oriented", "metaphor"]
        
        for comp in components:
            if comp["id"] in style_ids:
                categories["style"].append(comp)
            elif comp["id"] in structure_ids:
                categories["structure"].append(comp)
            elif comp["id"] in content_ids:
                categories["content"].append(comp)
            elif comp["id"] in depth_ids:
                categories["depth"].append(comp)
        
        return categories
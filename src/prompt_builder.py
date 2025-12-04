# src/prompt_builder.py
from typing import List, Dict, Any

class PromptBuilder:
    """英文提示构建器"""
    
    @staticmethod
    def build_prompt(component_ids: List[str], task_description: str = None) -> str:
        """
        构建英文提示
        
        Args:
            component_ids: 组件ID列表
            task_description: 任务描述，默认为"Summarize this technical paper"
            
        Returns:
            构建的完整提示
        """
        if task_description is None:
            task_description = "Summarize this technical paper"
        
        if not component_ids:
            return task_description
        
        # 需要组件库
        from component_lib import ComponentLibrary
        components = ComponentLibrary.load_default()
        component_dict = {c["id"]: c for c in components}
        
        # 收集组件文本
        component_texts = []
        for comp_id in component_ids:
            if comp_id in component_dict:
                component_texts.append(component_dict[comp_id]["text"])
        
        # 构建自然语言的提示
        if len(component_texts) == 1:
            return f"{task_description}. {component_texts[0]}"
        elif len(component_texts) == 2:
            return f"{task_description}. {component_texts[0]} and {component_texts[1]}"
        else:
            # 使用牛津逗号
            prompt = task_description + ". "
            prompt += ", ".join(component_texts[:-1])
            prompt += f", and {component_texts[-1]}"
            return prompt
    

    @staticmethod
    def build_evaluation_prompt(paper_content: str, summary: str, 
                               gold_summary: str = None) -> str:
        """
        构建英文评估提示
        
        Args:
            paper_content: 论文内容
            summary: 生成的摘要
            gold_summary: 参考摘要（可选）
            
        Returns:
            评估提示
        """
        prompt_parts = [
            "Please evaluate the quality of this paper summary:",
            "",
            "Original text excerpt:",
            paper_content[:500],
            "",
            "Generated summary:",
            summary,
            ""
        ]
        
        if gold_summary:
            prompt_parts.extend([
                "Reference summary (for comparison):",
                gold_summary,
                ""
            ])
        
        prompt_parts.extend([
            "Please rate on the following dimensions (0-10, 10 being best):",
            "1. Faithfulness: Does the summary accurately reflect the original text without adding or distorting information?",
            "2. Insightfulness: Does the summary provide valuable interpretation or depth of analysis?",
            "",
            "Return in JSON format:",
            '{"faithfulness": score, "insightfulness": score}',
            "",
            "Return only the JSON, no other text."
        ])
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def build_summary_prompt(paper_content: str, component_ids: List[str]) -> str:
        """
        构建摘要生成提示
        
        Args:
            paper_content: 论文内容
            component_ids: 组件ID列表
            
        Returns:
            摘要生成提示
        """
        base_prompt = PromptBuilder.build_prompt(component_ids)
        
        full_prompt = f"""
{base_prompt}

Paper content:
{paper_content[:1500]}
"""
        return full_prompt.strip()
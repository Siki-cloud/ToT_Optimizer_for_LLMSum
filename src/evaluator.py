# src/evaluator.py (修复版本)
import asyncio
import json
import random
import hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re
from tqdm import tqdm

import numpy as np
from rouge_score import rouge_scorer
from bert_score import BERTScorer


@dataclass
class EvaluationMetrics:
    """评估指标"""
    faithfulness: float = 0.0  # 忠实度
    conciseness: float = 0.0   # 简洁性
    completeness: float = 0.0  # 完整性
    readability: float = 0.0   # 可读性
    insightfulness: float = 0.0 # 洞察力
    overall: float = 0.0       # 总体评分
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "conciseness": self.conciseness,
            "completeness": self.completeness,
            "readability": self.readability,
            "insightfulness": self.insightfulness,
            "overall": self.overall
        }

@dataclass
class EvaluationResult:
    """评估结果"""
    prompt: str
    metrics: EvaluationMetrics
    summaries: List[str]
    paper_ids: List[str]
    details: Dict[str, Any]
    cache_key: str = ""
    
    @property
    def score(self) -> float:
        """获取加权总分"""
        weights = {
            "faithfulness": 0.25,
            "conciseness": 0.20,
            "completeness": 0.25,
            "readability": 0.15,
            "insightfulness": 0.15
        }
        
        total = 0.0
        metrics_dict = self.metrics.to_dict()
        for metric, weight in weights.items():
            if metric != "overall":
                total += metrics_dict.get(metric, 0.0) * weight
        
        return total

class PaperData:
    """论文数据类"""
    
    def __init__(self, paper_id: str, content: str, domain: str, title:str,
                 key_points: List[str] = None, gold_summary: str = None):
        self.id = paper_id
        self.title = title
        self.content = content
        self.domain = domain
        self.key_points = key_points or []
        self.gold_summary = gold_summary
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title":self.title,
            "content": self.content,
            "domain": self.domain,
            "key_points": self.key_points,
            "gold_summary": self.gold_summary
        }

class PromptEvaluator:
    """提示评估器"""
    
    def __init__(self, target_llm, evaluator_llm, papers: List[PaperData], 
                 config: Dict[str, Any]):
        self.target_llm = target_llm
        self.evaluator_llm = evaluator_llm
        self.papers = papers
        self.config = config
        
        # 缓存
        self.cache = {}
        self.cache_hits = 0
        self.total_evaluations = 0
        
        # 按领域分组
        self.domain_groups = {}
        for paper in papers:
            domain = paper.domain
            if domain not in self.domain_groups:
                self.domain_groups[domain] = []
            self.domain_groups[domain].append(paper)
    
    def _get_cache_key(self, prompt: str, paper_ids: List[str]) -> str:
        """生成缓存键"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        papers_hash = hashlib.md5("|".join(sorted(paper_ids)).encode()).hexdigest()
        return f"{prompt_hash}_{papers_hash}"
    
    def _sample_papers(self, num_samples: int) -> List[PaperData]:
        """采样论文（确保领域多样性）"""
        if num_samples >= len(self.papers) or num_samples < 0:
            return self.papers[:]
        
        # 分层采样
        selected = []
        domains = list(self.domain_groups.keys())
        
        # 确保每个领域至少有一篇
        papers_per_domain = max(1, num_samples // len(domains))
        
        for domain in domains:
            if domain in self.domain_groups:
                papers = self.domain_groups[domain]
                if len(papers) <= papers_per_domain:
                    selected.extend(papers)
                else:
                    selected.extend(random.sample(papers, papers_per_domain))
        
        # 如果还不够，随机补充
        if len(selected) < num_samples:
            selected_ids = {p.id for p in selected}
            available = [p for p in self.papers if p.id not in selected_ids]
            if available:
                additional = random.sample(available, min(num_samples - len(selected), len(available)))
                selected.extend(additional)
        
        return selected[:num_samples]
    
    async def evaluate_prompt(self, prompt: str, num_samples: int = None, 
                            use_cache: bool = True) -> EvaluationResult:
        """
        评估提示
        
        Args:
            prompt: 要评估的提示
            num_samples: 采样论文数量
            use_cache: 是否使用缓存
            
        Returns:
            评估结果
        """
        if num_samples is None:
            num_samples = self.config.get("samples_per_eval", 3)
        
        # 采样论文
        sampled_papers = self._sample_papers(num_samples)
        paper_ids = [p.id for p in sampled_papers]
        
        # 检查缓存
        cache_key = self._get_cache_key(prompt, paper_ids)
        if use_cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.total_evaluations += 1
        
        # 并行评估每篇论文
        tasks = []
        for paper in sampled_papers:
            task = self._evaluate_single(prompt, paper)
            tasks.append(task)
        
        paper_results = await asyncio.gather(*tasks)
        
        # 聚合结果
        result = self._aggregate_results(prompt, paper_results, sampled_papers, cache_key)
        
        # 缓存结果
        self.cache[cache_key] = result
        
        return result
    
    async def _evaluate_single(self, prompt: str, paper: PaperData) -> Dict[str, Any]:
        """评估单篇论文"""
        try:
            # 1. 生成摘要
            # 修复这里：使用切片而不是在f-string中使用反斜杠
            paper_content_preview = paper.content[:1500]
            full_prompt = f"{prompt}\n\n论文内容：\n{paper_content_preview}"
            
            summary_response = await self.target_llm.generate(
                [full_prompt],
                generation_config=None  # 使用默认配置
            )
            
            summary = summary_response[0] if summary_response else ""
            
            # 2. 计算指标
            metrics = await self._compute_metrics(paper, summary)
            
            return {
                "paper_id": paper.id,
                "domain": paper.domain,
                "summary": summary,
                "metrics": metrics.to_dict(),
                "success": True
            }
            
        except Exception as e:
            print(f"评估论文 {paper.id} 时出错: {e}")
            return {
                "paper_id": paper.id,
                "domain": paper.domain,
                "summary": "",
                "metrics": EvaluationMetrics().to_dict(),
                "success": False
            }
    
    async def _compute_metrics(self, paper: PaperData, summary: str) -> EvaluationMetrics:
        """计算评估指标"""
        metrics = EvaluationMetrics()
        
        # 1. 规则指标（快速）
        metrics.conciseness = self._compute_conciseness(summary)
        metrics.completeness = self._compute_completeness(summary, paper.key_points)
        metrics.readability = self._compute_readability(summary)
        
        # 2. LLM指标（准确但慢）
        llm_metrics = await self._compute_llm_metrics(paper.content, summary, paper.gold_summary)
        metrics.faithfulness = llm_metrics.get("faithfulness", 0.5)
        metrics.insightfulness = llm_metrics.get("insightfulness", 0.5)
        
        # 3. 计算总体评分
        weights = self.config.get("metric_weights", {
            "faithfulness": 0.25, "conciseness": 0.20, "completeness": 0.25,
            "readability": 0.15, "insightfulness": 0.15
        })
        
        total = 0.0
        metrics_dict = metrics.to_dict()
        for metric, weight in weights.items():
            if metric != "overall":
                total += metrics_dict.get(metric, 0.0) * weight
        
        metrics.overall = total
        
        return metrics
    
    def _compute_conciseness(self, summary: str) -> float:
        """计算简洁性"""
        if not summary:
            return 0.0
        
        word_count = len(summary.split())
        
        # 理想长度：100-200词
        if word_count <= 100:
            return 0.9  # 非常简洁
        elif word_count <= 150:
            return 0.8  # 简洁
        elif word_count <= 200:
            return 0.7  # 适中
        elif word_count <= 250:
            return 0.5  # 稍长
        elif word_count <= 300:
            return 0.3  # 过长
        else:
            return 0.1  # 非常长
    
    def _compute_completeness(self, summary: str, key_points: List[str]) -> float:
        """计算完整性（关键点覆盖）"""
        if not summary or not key_points:
            return 0.5
        
        summary_lower = summary.lower()
        covered = 0
        
        for point in key_points:
            # 检查关键点中的关键词是否出现在摘要中
            keywords = point.lower().split()[:3]
            if any(keyword in summary_lower for keyword in keywords if len(keyword) > 3):
                covered += 1
        
        return covered / len(key_points)
    
    def _compute_readability(self, summary: str) -> float:
        """计算可读性"""
        if not summary:
            return 0.0
        
        # 简单启发式：句子数量、平均句长
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        if len(sentences) == 0:
            return 0.5
        
        word_count = len(summary.split())
        avg_sentence_len = word_count / len(sentences)
        
        # 理想平均句长：15-25词
        if 15 <= avg_sentence_len <= 25:
            return 0.8
        elif 10 <= avg_sentence_len < 15 or 25 < avg_sentence_len <= 30:
            return 0.6
        elif 5 <= avg_sentence_len < 10 or 30 < avg_sentence_len <= 40:
            return 0.4
        else:
            return 0.2
    
    async def _compute_llm_metrics(self, paper_content: str, summary: str, 
                              gold_summary: str = None) -> Dict[str, float]:
        
        """使用LLM计算指标 - 英文版本"""
        # 导入英文提示构建器
        from prompt_builder import PromptBuilder
        
        eval_prompt = PromptBuilder.build_evaluation_prompt(
            paper_content=paper_content,
            summary=summary,
            gold_summary=gold_summary
        )
        # 创建GenerationConfig对象
        from vllm_server import GenerationConfig  # 确保导入正确的类
        
        generation_config = GenerationConfig(
            temperature=0.1, # 低温度，更确定
            max_tokens=150,# 限制长度
            stop=["\n\n", "##", "explain"]  # 提前停止词
            # stop参数可能需要特殊处理，如果GenerationConfig不支持stop参数
        )
        
        try:
            # 设置更严格的生成参数
            response = await self.evaluator_llm.generate(
                [eval_prompt],
                generation_config=generation_config
            )
            
            response_text = response[0] if response else ""
            
            # 调试日志
            # print(f"=== LLM评估响应 ===")
            # print(f"长度: {len(response_text)}")
            # print(f"内容: {response_text[:200]}")
            # print("=" * 40)
            
            # 多策略解析
            parsed_data = None
            
            # 策略1：直接JSON解析
            try:
                parsed_data = json.loads(response_text.strip())
            except json.JSONDecodeError:
                pass
            
            # 策略2：提取JSON块
            if not parsed_data:
                parsed_data = self._extract_json_from_text(response_text)
            
            # 策略3：正则提取数字
            if not parsed_data:
                parsed_data = self._extract_scores_with_regex(response_text)
            
            # 策略4：使用备用评分
            if not parsed_data:
                parsed_data = self._compute_fallback_scores(paper_content, summary)
            
            # 确保分数在合理范围
            faithfulness = self._normalize_score(parsed_data.get("faithfulness", 
                                                            parsed_data.get("f", 5.0)))
            insightfulness = self._normalize_score(parsed_data.get("insightfulness", 
                                                                parsed_data.get("i", 5.0)))
            
            return {
                "faithfulness": faithfulness / 10.0,
                "insightfulness": insightfulness / 10.0
            }
            
        except Exception as e:
            print(f"LLM指标计算错误: {e}")
            import traceback
            traceback.print_exc()
            return {"faithfulness": 0.5, "insightfulness": 0.5}

    def _extract_json_from_text(self, text: str) -> Dict:
        """从文本中提取JSON"""
        import re
        
        # 匹配JSON对象
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # 尝试清理和解析
                cleaned = match.strip()
                # 确保键有引号
                if '"' not in cleaned and "'" not in cleaned:
                    # 尝试添加引号
                    cleaned = re.sub(r'(\w+):', r'"\1":', cleaned)
                
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    return data
            except:
                continue
        
        return {}

    def _extract_scores_with_regex(self, text: str) -> Dict:
        """使用正则表达式提取分数"""
        import re
        
        # 匹配分数模式
        patterns = [
            r'"faithfulness"\s*:\s*(\d+(?:\.\d+)?)',
            r'"f"\s*:\s*(\d+(?:\.\d+)?)',
            r'faithfulness[:\s]+(\d+(?:\.\d+)?)',
            r'忠实度[:\s]+(\d+(?:\.\d+)?)',
        ]
        
        faithfulness_score = None
        insightfulness_score = None
        
        # 提取忠实度
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    faithfulness_score = float(match.group(1))
                    break
                except:
                    continue
        
        # 提取洞察力
        insight_patterns = [
            r'"insightfulness"\s*:\s*(\d+(?:\.\d+)?)',
            r'"i"\s*:\s*(\d+(?:\.\d+)?)',
            r'insightfulness[:\s]+(\d+(?:\.\d+)?)',
            r'洞察力[:\s]+(\d+(?:\.\d+)?)',
        ]
        
        for pattern in insight_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    insightfulness_score = float(match.group(1))
                    break
                except:
                    continue
        
        result = {}
        if faithfulness_score is not None:
            result["faithfulness"] = faithfulness_score
        if insightfulness_score is not None:
            result["insightfulness"] = insightfulness_score
        
        return result

    def _normalize_score(self, score) -> float:
        """标准化分数到0-10范围"""
        try:
            score = float(score)
            if score > 100:  # 可能是百分比
                score = score / 10.0
            elif score > 10:  # 可能是0-100
                score = score / 10.0
            elif score > 1 and score <= 5:  # 可能是1-5分制
                score = score * 2
            elif score <= 1:  # 可能是0-1
                score = score * 10
            
            return max(0, min(10, score))
        except:
            return 5.0

    def _compute_fallback_scores(self, paper_content: str, summary: str) -> Dict:
        """启发式评分（后备方案）"""
        # 基于文本相似性简单评分
        import difflib
        
        content_words = set(paper_content[:300].lower().split())
        summary_words = set(summary.lower().split())
        
        # Jaccard相似度
        if content_words and summary_words:
            intersection = content_words & summary_words
            union = content_words | summary_words
            similarity = len(intersection) / len(union) if union else 0
        else:
            similarity = 0.5
        
        # 转换到0-10分
        faithfulness_score = min(10, similarity * 12)
        insightfulness_score = 7.0  # 默认中等
        
        return {'faithfulness': faithfulness_score, 'insightfulness': insightfulness_score}
 
    def _aggregate_results(self, prompt: str, paper_results: List[Dict], 
                          papers: List[PaperData], cache_key: str) -> EvaluationResult:
        """聚合多个论文的结果"""
        successful = [r for r in paper_results if r["success"]]
        
        if not successful:
            return EvaluationResult(
                prompt=prompt,
                metrics=EvaluationMetrics(),
                summaries=[],
                paper_ids=[p.id for p in papers],
                details={"error": "所有评估都失败"},
                cache_key=cache_key
            )
        
        # 计算平均指标
        metrics_sum = {k: 0.0 for k in EvaluationMetrics().__dict__.keys()}
        count = 0
        
        for result in successful:
            for metric, value in result["metrics"].items():
                if metric in metrics_sum:
                    metrics_sum[metric] += value
            count += 1
        
        avg_metrics = EvaluationMetrics()
        for metric in metrics_sum:
            if count > 0:
                setattr(avg_metrics, metric, metrics_sum[metric] / count)
        
        # 收集摘要
        summaries = [r["summary"] for r in successful if r.get("summary")]
        
        # 计算领域分布
        domain_counts = {}
        for result in successful:
            domain = result.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return EvaluationResult(
            prompt=prompt,
            metrics=avg_metrics,
            summaries=summaries,  # 最多保留个摘要
            paper_ids=[p.id for p in papers],
            details={
                "total_papers": len(papers),
                "successful_evaluations": len(successful),
                "domain_distribution": domain_counts,
                "cache_key": cache_key
            },
            cache_key=cache_key
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取评估器统计信息"""
        return {
            "total_papers": len(self.papers),
            "domains": list(self.domain_groups.keys()),
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "total_evaluations": self.total_evaluations
        }
    



class SummaryEvaluator:
    """摘要评估指标计算器"""
    
    def __init__(self, model_type=None, language='en'):
        """
        初始化评估器
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
            use_stemmer=True
        )
        
        # 初始化BERTScorer
        if not model_type:
            model_type = "bert-base-multilingual-cased"
            # model_type = "/mnt/sharedata/ssd_large/common/SLMs/bert-base-multilingual-cased"
        self.bert_scorer = BERTScorer(
            model_type=model_type,
            lang=language,
            rescale_with_baseline=True,
            device='cpu'
        )

        self.summary_metrics=None
    
    def compute_metrics(self, generated_summary: str, gold_summary: str) -> Dict[str, float]:
        """
        计算单个摘要对的指标
        """
        metrics = {}
        
        # 1. 计算ROUGE分数
        rouge_scores = self.rouge_scorer.score(gold_summary, generated_summary)
        
        metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
        metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
        metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
        
        # 2. 计算BERTScore
        P, R, F1 = self.bert_scorer.score([generated_summary], [gold_summary])
        metrics['bertscore_f1'] = float(F1[0])
        metrics['bertscore_precision'] = float(P[0])
        metrics['bertscore_recall'] = float(R[0])
        
        # 3. 计算综合分数
        metrics['composite_score'] = 0.5 * metrics['rouge2'] + 0.5 * metrics['bertscore_f1']
        
        return metrics
    
    def compute_batch_metrics(self, generated_summaries: List[str], 
                            gold_summaries: List[str]) -> Dict[str, Any]:
        """
        批量计算指标并返回统计信息
        """
        if len(generated_summaries) != len(gold_summaries):
            raise ValueError("生成的摘要和参考摘要数量必须相同")
        
        all_metrics = []
        
        for gen, gold in zip(generated_summaries, gold_summaries):
            metrics = self.compute_metrics(gen, gold)
            all_metrics.append(metrics)
        
        # 计算平均值
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
            avg_metrics[f'min_{key}'] = np.min(values)
            avg_metrics[f'max_{key}'] = np.max(values)
        
        summary_metrics = {
            'per_sample': all_metrics,
            'statistics': avg_metrics,
        }
        self.summary_metrics = summary_metrics
        return summary_metrics
    
    def print_result(self):
        if self.summary_metrics is not None:
            # 打印结果
            stats = self.summary_metrics['statistics']
                
            # print(f"\n📊 评估样本数: {min_len}")
            print("\n📈 摘要质量指标:")
            print(f"{'指标':<20} {'平均值':<10} {'标准差':<10} {'范围':<15}")
            print("-" * 55)
            
            metrics_to_show = {
                    'rouge2': 'ROUGE-2 F1',
                    'rougeL': 'ROUGE-L F1',
                    'bertscore_f1': 'BERTScore F1',
                    'composite_score': '综合分数'
                }
                
            for key, display_name in metrics_to_show.items():
                    avg = stats[f'avg_{key}']
                    std = stats[f'std_{key}']
                    min_val = stats[f'min_{key}']
                    max_val = stats[f'max_{key}']
                    print(f"{display_name:<20} {avg:<10.4f} {std:<10.4f} [{min_val:.4f}-{max_val:.4f}]")
                

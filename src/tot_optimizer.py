# src/tot_optimizer.py
import asyncio
import json
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import heapq
from tqdm import tqdm


@dataclass
class SearchNode:
    """搜索树节点"""
    node_id: str
    state: List[str]  # 组件ID列表
    parent_id: Optional[str]
    depth: int
    visits: int = 0
    total_score: float = 0.0
    heuristic_score: float = 0.0
    full_evaluation_score: float = 0.0
    children_ids: List[str] = None
    is_fully_evaluated: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def avg_score(self) -> float:
        """平均分数（优先使用完整评估分数）"""
        if self.is_fully_evaluated and self.full_evaluation_score > 0:
            return self.full_evaluation_score
        elif self.visits > 0:
            return self.total_score / self.visits
        else:
            return self.heuristic_score
    
    def ucb_score(self, parent_visits: int, exploration_weight: float = 1.41) -> float:
        """计算UCB分数"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.avg_score
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration
    
    def state_hash(self) -> str:
        """生成状态哈希"""
        state_str = "|".join(sorted(self.state))
        return hashlib.md5(state_str.encode()).hexdigest()

class ToTPromptOptimizer:
    """基于Tree of Thoughts的提示优化器"""
    
    def __init__(self, components: List[Dict], evaluator, config: Dict[str, Any]):
        """
        初始化优化器
        
        Args:
            components: 提示组件列表
            evaluator: 评估器实例
            config: 配置字典
        """
        self.components = {c["id"]: c for c in components}
        self.evaluator = evaluator
        self.config = config
        
        # 使用英文任务描述
        self.task_description = config.get("task_description", "Summarize this technical paper")
        
        # 搜索状态
        self.nodes = {}  # node_id -> SearchNode
        self.state_to_node = {}  # state_hash -> node_id
        
        # 创建根节点
        root_node = SearchNode(
            node_id="root",
            state=[],
            parent_id=None,
            depth=0
        )
        self.nodes["root"] = root_node
        self.state_to_node[root_node.state_hash()] = "root"
        
        # 最佳节点
        self.best_node = root_node
        self.best_score = 0.0
        
        # 统计信息
        self.stats = {
            "total_iterations": 0,
            "nodes_created": 1,
            "full_evaluations": 0,
            "heuristic_evaluations": 0,
            "llm_calls": 0,
            "cache_hits": 0
        }
        
        # 搜索历史
        self.search_history = []
    
    def build_prompt(self, component_ids: List[str]) -> str:
        from prompt_builder import PromptBuilder

        """根据组件ID构建完整英文提示"""
        return PromptBuilder.build_prompt(component_ids=component_ids, task_description=self.task_description)
    
        
    def check_constraints(self, component_ids: List[str]) -> Tuple[bool, str]:
        """检查约束"""
        # 1. 检查互斥
        active_ids = set(component_ids)
        
        for comp_id in active_ids:
            comp = self.components.get(comp_id, {})
            conflicts = comp.get("conflicts", [])
            for conflict_id in conflicts:
                if conflict_id in active_ids:
                    return False, f"互斥冲突: {comp_id} 与 {conflict_id}"
        
        # 2. 检查依赖
        for comp_id in active_ids:
            comp = self.components.get(comp_id, {})
            requires = comp.get("requires", [])
            for required_id in requires:
                if required_id not in active_ids:
                    return False, f"缺少依赖: {comp_id} 需要 {required_id}"
        
        # 3. 检查token限制（近似）
        prompt = self.build_prompt(component_ids)
        token_estimate = len(prompt.split()) * 1.3
        max_tokens = self.config.get("max_tokens", 200)
        
        if token_estimate > max_tokens:
            return False, f"Token估计超限: {token_estimate:.1f} > {max_tokens}"
        
        return True, "满足约束"
    
    def generate_candidates(self, current_state: List[str]) -> List[List[str]]:
        """生成候选状态"""
        candidates = []
        current_set = set(current_state)
        
        # 操作1: 添加组件
        for comp_id, comp in self.components.items():
            if comp_id not in current_set:
                new_state = current_state + [comp_id]
                is_valid, _ = self.check_constraints(new_state)
                if is_valid:
                    candidates.append(new_state)
        
        # 操作2: 移除组件
        for i in range(len(current_state)):
            new_state = current_state[:i] + current_state[i+1:]
            candidates.append(new_state)
        
        # 操作3: 替换组件
        for i, current_id in enumerate(current_state):
            for new_id in self.components:
                if new_id != current_id and new_id not in current_set:
                    new_state = current_state.copy()
                    new_state[i] = new_id
                    is_valid, _ = self.check_constraints(new_state)
                    if is_valid:
                        candidates.append(new_state)
        
        # 去重
        unique_candidates = []
        seen = set()
        
        for cand in candidates:
            cand_hash = hashlib.md5("|".join(sorted(cand)).encode()).hexdigest()
            if cand_hash not in seen:
                seen.add(cand_hash)
                unique_candidates.append(cand)
        
        return unique_candidates
    
    async def evaluate_state_heuristic(self, state: List[str]) -> float:
        """启发式评估状态"""
        if not state:
            return 0.3
        
        # 基于组件效果向量计算
        total_effect = 0.0
        for comp_id in state:
            comp = self.components.get(comp_id, {})
            effects = comp.get("effect_vector", {})
            avg_effect = sum(effects.values()) / len(effects) if effects else 0.5
            total_effect += avg_effect
        
        avg_effect = total_effect / len(state)
        
        # 惩罚过长状态
        prompt = self.build_prompt(state)
        token_estimate = len(prompt.split()) * 1.3
        max_tokens = self.config.get("max_tokens", 200)
        
        if token_estimate > max_tokens * 0.8:  # 超过80%就惩罚
            penalty = (token_estimate - max_tokens * 0.8) / (max_tokens * 0.2)
            penalty = min(penalty, 0.5)  # 最多惩罚50%
            avg_effect *= (1 - penalty)
        
        return avg_effect
    
    async def evaluate_state_full(self, state: List[str]) -> Tuple[float, Dict[str, Any]]:
        """完整评估状态（使用测试数据）"""
        self.stats["full_evaluations"] += 1
        
        prompt = self.build_prompt(state)
        
        # 使用评估器评估提示
        eval_result = await self.evaluator.evaluate_prompt(
            prompt,
            num_samples=self.config.get("train_samples_per_eval", 3)
        )
        
        score = eval_result.score
        details = {
            "prompt": prompt,
            "metrics": eval_result.metrics.to_dict(),
            "score_details": eval_result.details,
            "summaries": eval_result.summaries[:1]  # 保留一个摘要示例
        }
        
        return score, details
    
    def create_node(self, state: List[str], parent_id: Optional[str] = None) -> SearchNode:
        """创建新节点"""
        state_hash = hashlib.md5("|".join(sorted(state)).encode()).hexdigest()
        
        # 如果状态已存在，返回现有节点
        if state_hash in self.state_to_node:
            return self.nodes[self.state_to_node[state_hash]]
        
        # 创建新节点
        node_id = f"node_{self.stats['nodes_created']}"
        parent_node = self.nodes.get(parent_id) if parent_id else None
        
        node = SearchNode(
            node_id=node_id,
            state=state,
            parent_id=parent_id,
            depth=parent_node.depth + 1 if parent_node else 0
        )
        
        self.nodes[node_id] = node
        self.state_to_node[state_hash] = node_id
        self.stats["nodes_created"] += 1
        
        # 添加到父节点的子节点列表
        if parent_node:
            parent_node.children_ids.append(node_id)
        
        return node
    
    async def mcts_iteration(self, exploration_weight: float = 1.41):
        """执行一次MCTS迭代"""
        self.stats["total_iterations"] += 1
        
        # 1. 选择阶段
        node = self.nodes["root"]
        path = [node]
        
        while node.children_ids:
            # 选择最佳子节点
            best_score = -float('inf')
            best_child = None
            
            for child_id in node.children_ids:
                child = self.nodes[child_id]
                child_score = child.ucb_score(node.visits, exploration_weight)
                
                if child_score > best_score:
                    best_score = child_score
                    best_child = child
            
            if best_child:
                node = best_child
                path.append(node)
            else:
                break
        
        # 2. 扩展阶段
        if not node.is_fully_evaluated and node.depth < self.config.get("max_depth", 5):
            candidates = self.generate_candidates(node.state)
            
            # 限制扩展数量
            max_expansions = self.config.get("beam_width", 3)
            candidates = candidates[:max_expansions * 2]  # 生成多一些，然后筛选
            
            for candidate_state in candidates:
                # 检查是否已存在
                cand_hash = hashlib.md5("|".join(sorted(candidate_state)).encode()).hexdigest()
                if cand_hash not in self.state_to_node:
                    child_node = self.create_node(candidate_state, node.node_id)
                    
                    # 启发式评估新节点
                    heuristic_score = await self.evaluate_state_heuristic(candidate_state)
                    child_node.heuristic_score = heuristic_score
                    child_node.total_score = heuristic_score
                    child_node.visits = 1
                    
                    self.stats["heuristic_evaluations"] += 1
        
        # 3. 评估阶段（对最有希望的节点进行完整评估）
        if node.depth > 0 and not node.is_fully_evaluated:
            # 检查节点是否有潜力
            if node.avg_score > self.best_score * 0.7:  # 比最佳分数的70%好
                full_score, details = await self.evaluate_state_full(node.state)
                
                node.full_evaluation_score = full_score
                node.is_fully_evaluated = True
                node.metadata.update(details)
                
                # 更新最佳节点
                if full_score > self.best_score:
                    self.best_score = full_score
                    self.best_node = node
                
                # 更新回溯分数
                score_for_backup = full_score
            else:
                score_for_backup = node.avg_score
        else:
            score_for_backup = node.avg_score
        
        # 4. 回溯阶段
        for path_node in reversed(path):
            path_node.visits += 1
            path_node.total_score += score_for_backup
    
    async def beam_search_iteration(self, beam_width: int = 3):
        """执行一次Beam Search迭代"""
        self.stats["total_iterations"] += 1
        
        # 获取当前beam（最佳节点）
        current_beam = []
        
        # 从所有节点中选择最佳的几个
        all_nodes = list(self.nodes.values())
        all_nodes.sort(key=lambda n: n.avg_score, reverse=True)
        
        for node in all_nodes[:beam_width]:
            current_beam.append(node)
        
        # 为每个beam节点扩展
        new_candidates = []
        
        for node in current_beam:
            if node.depth >= self.config.get("max_depth", 5):
                continue
            
            candidates = self.generate_candidates(node.state)
            
            for candidate_state in candidates:
                # 检查是否已存在
                cand_hash = hashlib.md5("|".join(sorted(candidate_state)).encode()).hexdigest()
                if cand_hash not in self.state_to_node:
                    # 创建新节点
                    child_node = self.create_node(candidate_state, node.node_id)
                    
                    # 启发式评估
                    heuristic_score = await self.evaluate_state_heuristic(candidate_state)
                    child_node.heuristic_score = heuristic_score
                    child_node.total_score = heuristic_score
                    child_node.visits = 1
                    
                    self.stats["heuristic_evaluations"] += 1
                    new_candidates.append(child_node)
        
        # 对新候选进行完整评估
        if new_candidates:
            new_candidates.sort(key=lambda n: n.heuristic_score, reverse=True)
            top_candidates = new_candidates[:beam_width]
            
            for candidate in top_candidates:
                if not candidate.is_fully_evaluated:
                    full_score, details = await self.evaluate_state_full(candidate.state)
                    
                    candidate.full_evaluation_score = full_score
                    candidate.is_fully_evaluated = True
                    candidate.metadata.update(details)
                    
                    # 更新最佳节点
                    if full_score > self.best_score:
                        self.best_score = full_score
                        self.best_node = candidate
    
    async def optimize(self, iterations: int = 30, search_method: str = "mcts"):
        """
        执行优化
        
        Args:
            iterations: 迭代次数
            search_method: 搜索方法 ("mcts" 或 "beam")
        """
        print(f"开始优化，方法: {search_method}, 迭代次数: {iterations}")
        print(f"初始状态: 空提示")
        
        progress_bar = tqdm(range(iterations), desc="优化进度")
        
        for i in progress_bar:
            if search_method == "mcts":
                await self.mcts_iteration(
                    exploration_weight=self.config.get("exploration_weight", 1.41)
                )
            else:  # beam search
                await self.beam_search_iteration(
                    beam_width=self.config.get("beam_width", 3)
                )
            
            # 更新进度条
            progress_bar.set_postfix({
                "最佳分数": f"{self.best_score:.3f}",
                "节点数": self.stats["nodes_created"],
                "完整评估": self.stats["full_evaluations"]
            })
            
            # 记录历史
            self.search_history.append({
                "iteration": i,
                "best_score": self.best_score,
                "best_state": self.best_node.state,
                "nodes_created": self.stats["nodes_created"],
                "full_evaluations": self.stats["full_evaluations"]
            })
        
        print(f"\n优化完成!")
        print(f"最佳分数: {self.best_score:.3f}")
        print(f"最佳提示组件: {self.best_node.state}")
        print(f"总节点数: {self.stats['nodes_created']}")
        print(f"完整评估次数: {self.stats['full_evaluations']}")
    
    def get_best_prompt(self) -> Tuple[str, Dict[str, Any]]:
        """获取最佳提示"""
        if self.best_node.node_id == "root":
            return self.build_prompt([]), {}
        
        prompt = self.build_prompt(self.best_node.state)
        details = self.best_node.metadata.copy()
        details["score"] = self.best_score
        details["components"] = self.best_node.state
        
        return prompt, details
    
    def get_search_tree(self) -> Dict[str, Any]:
        """获取搜索树结构"""
        tree = {
            "root": "root",
            "nodes": {},
            "edges": []
        }
        
        for node_id, node in self.nodes.items():
            tree["nodes"][node_id] = {
                "state": node.state,
                "depth": node.depth,
                "score": node.avg_score,
                "visits": node.visits,
                "is_fully_evaluated": node.is_fully_evaluated
            }
            
            for child_id in node.children_ids:
                tree["edges"].append({
                    "from": node_id,
                    "to": child_id
                })
        
        return tree
    
    def save_results(self, filepath: str):
        """保存结果到文件"""
        results = {
            "best_prompt": self.build_prompt(self.best_node.state),
            "best_components": self.best_node.state,
            "best_score": self.best_score,
            "best_node_details": self.best_node.metadata,
            "stats": self.stats,
            "search_history": self.search_history,
            "config": self.config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {filepath}")
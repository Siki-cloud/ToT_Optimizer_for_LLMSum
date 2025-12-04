# src/vllm_server.py
import asyncio
import torch
from typing import List, Dict, Any,Optional
from dataclasses import dataclass, field
import json
import time
import os



try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not installed. Using mock LLM for demonstration.")


@dataclass
class GenerationConfig:
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 300
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = field(default_factory=list)  # 添加stop参数
    
    def __post_init__(self):
        # 确保stop是列表
        if self.stop is None:
            self.stop = []


class MockLLM:
    """用于演示的模拟LLM"""
    
    def __init__(self, name="mock-llm"):
        self.name = name
        self.call_count = 0
        
    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """模拟生成文本"""
        self.call_count += 1
        
        # 模拟生成摘要
        responses = []
        for prompt in prompts:
            if "总结" in prompt or "summary" in prompt.lower():
                # 模拟论文摘要
                response = f"""这是模拟生成的论文摘要（调用#{self.call_count}）。
                本文提出了一个创新方法，在基准测试上达到了先进水平。
                主要贡献包括：1) 新架构设计；2) 高效训练策略；3) 广泛实验验证。"""
            elif "评估" in prompt or "evaluat" in prompt.lower():
                # 模拟评估结果
                response = '''{
                    "faithfulness": 8.2,
                    "conciseness": 7.5,
                    "completeness": 8.0,
                    "readability": 8.5,
                    "insightfulness": 7.8,
                    "overall": 8.0,
                    "comments": "摘要质量良好，覆盖了主要贡献"
                }'''
            else:
                response = f"这是对提示的响应: {prompt[:50]}..."
            
            responses.append(response)
        
        # 模拟延迟
        await asyncio.sleep(0.1)
        return responses

class VLLMServer:
    """vLLM服务器封装类"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        初始化vLLM服务器
        
        Args:
            model_config: 模型配置字典
        """
        self.config = model_config
        self.device = model_config.get("device", "cuda:0")
        self.engine = None
        self.call_count = 0
        self.total_tokens = 0
        
    async def initialize(self):
        """异步初始化引擎"""
        if not VLLM_AVAILABLE:
            print(f"vLLM不可用，为 {self.config['path']} 创建模拟LLM")
            self.engine = MockLLM(name=self.config["path"])
            return
        
        try:
            # 设置设备
            # device_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            # torch.cuda.set_device(device_id)
            
            # 解析GPU设备设置
            if "cuda:" in self.device:
                # 设置CUDA_VISIBLE_DEVICES环境变量
                gpu_ids = self.device.split(":")[1]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
                tensor_parallel_size = len(gpu_ids.split(','))
            else:
                tensor_parallel_size = 1
            # 创建引擎参数
            engine_args = AsyncEngineArgs(
                model=self.config["path"],
                tensor_parallel_size= self.config.get("tensor_parallel_size", tensor_parallel_size),
                gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.95),
                max_num_seqs=self.config.get("max_num_seqs", 64),
                # max_model_len=self.config.get("max_model_len", 4096),
                trust_remote_code=True,
                enforce_eager=True,
                dtype=self.config.get("dtype", "bfloat16"),
               
            )
            
            # 创建异步引擎
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            print(f"✅ vLLM服务器初始化成功: {self.config['path']} on {self.device}")
            
        except Exception as e:
            print(f"❌ vLLM初始化失败: {e}")
            print("使用模拟LLM作为后备")
            self.engine = MockLLM(name=self.config["path"])
    
    # async def generate(self, prompts: List[str], generation_config: GenerationConfig = None) -> List[str]:
    #     """
    #     异步生成文本
        
    #     Args:
    #         prompts: 输入提示列表
    #         generation_config: 生成配置
            
    #     Returns:
    #         生成的文本列表
    #     """
    #     if generation_config is None:
    #         generation_config = GenerationConfig()
        
    #     self.call_count += 1
        
    #     if isinstance(self.engine, MockLLM):
    #         return await self.engine.generate(prompts)
        
    #     try:
    #         # 创建采样参数
    #         sampling_params = SamplingParams(
    #             temperature=generation_config.temperature,
    #             top_p=generation_config.top_p,
    #             max_tokens=generation_config.max_tokens,
    #             frequency_penalty=generation_config.frequency_penalty,
    #             presence_penalty=generation_config.presence_penalty
    #         )
            
            # # 生成请求ID
            # request_id = f"req_{self.call_count}_{int(time.time())}"
            
    #         # 执行生成
    #         outputs = await self.engine.generate(
    #             prompt=prompts,
    #             sampling_params=sampling_params,
    #             request_id=request_id
    #         )
            
    #         # 提取结果
    #         results = []
    #         for output in outputs:
    #             generated_text = output.outputs[0].text
    #             self.total_tokens += len(generated_text.split())
    #             results.append(generated_text.strip())
            
    #         return results
            
    #     except Exception as e:
    #         print(f"生成错误: {e}")
    #         # 返回默认值
    #         return [""] * len(prompts)
    

    async def generate(self, prompts: List[str], generation_config: GenerationConfig = None) -> List[str]:
        """
        异步生成文本 - 修复的vLLM调用方式
        
        Args:
            prompts: 输入提示列表
            generation_config: 生成配置
            
        Returns:
            生成的文本列表
        """
        if generation_config is None:
            generation_config = GenerationConfig()
        
        self.call_count += 1
        
        # 如果引擎是模拟的，直接调用
        if isinstance(self.engine, MockLLM):
            return await self.engine.generate(prompts)
        
        try:
            # 创建采样参数
            sampling_params = SamplingParams(
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                max_tokens=generation_config.max_tokens,
                frequency_penalty=generation_config.frequency_penalty,
                presence_penalty=generation_config.presence_penalty
            )
            
            # 为每个提示创建生成任务
            tasks = []
            for prompt in prompts:
                request_id =  random_uuid()
            
                task = self._generate_single(prompt, sampling_params, request_id)
                tasks.append(task)
            
            # 并行执行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            final_results = []
            for result in results:
                if isinstance(result, Exception):
                    print(f"生成失败: {result}")
                    final_results.append("")
                else:
                    final_results.append(result)
            
            return final_results
            
        except Exception as e:
            print(f"生成错误: {e}")
            return [""] * len(prompts)
    
    async def _generate_single(self, prompt: str, sampling_params: SamplingParams, request_id: str) -> str:
        """生成单个提示 - 正确使用异步生成器"""
        try:
            # 正确方式：engine.generate返回AsyncGenerator，需要异步迭代
            # 注意：这里prompt是单个字符串，不是列表！
            results_generator: AsyncGenerator[RequestOutput, None] = self.engine.generate(
                prompt=prompt,  # 单个提示字符串
                sampling_params=sampling_params,
                request_id=request_id
            )
            
            # 异步迭代生成器获取结果
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output and final_output.outputs:
                generated_text = final_output.outputs[0].text
                self.total_tokens += len(generated_text.split())
                return generated_text.strip()
            
            return ""
            
        except Exception as e:
            print(f"单提示生成错误: {e}")
            return ""
    
    def get_stats(self):
        if isinstance(self.engine, MockLLM):
            return self.engine.get_stats()
        
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "model": self.config["path"],
            "device": self.device,
            "type": "vllm"
        }
    
    # def get_stats(self) -> Dict[str, Any]:
    #     """获取服务器统计信息"""
    #     return {
    #         "call_count": self.call_count,
    #         "total_tokens": self.total_tokens,
    #         "model": self.config["path"],
    #         "device": self.device
    #     }
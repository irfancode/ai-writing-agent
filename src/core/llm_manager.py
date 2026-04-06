"""LLM Manager - Multi-model support for AI Writing Agent"""

import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import httpx


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"


@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None


class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[BaseMessage],
        **kwargs
    ):
        """Generate a streaming response from the LLM"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM wrapper"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.api_base,
            streaming=False,
        )
    
    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        response = await self.model.agenerate([messages])
        result = response.generations[0][0]
        return LLMResponse(
            content=result.text,
            model=self.config.model_name,
            usage={
                "prompt_tokens": response.llm_output.get("token_usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.llm_output.get("token_usage", {}).get("completion_tokens", 0),
                "total_tokens": response.llm_output.get("token_usage", {}).get("total_tokens", 0),
            },
            finish_reason=result.generation_info.get("finish_reason") if result.generation_info else None,
        )
    
    async def generate_stream(self, messages: List[BaseMessage], **kwargs):
        async for chunk in self.model.astream(messages):
            yield chunk.content


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM wrapper"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            anthropic_api_key=api_key,
            base_url=config.api_base,
        )
    
    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        response = await self.model.agenerate([messages])
        result = response.generations[0][0]
        return LLMResponse(
            content=result.text,
            model=self.config.model_name,
            usage={
                "prompt_tokens": response.llm_output.get("token_usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.llm_output.get("token_usage", {}).get("completion_tokens", 0),
                "total_tokens": response.llm_output.get("token_usage", {}).get("total_tokens", 0),
            },
        )
    
    async def generate_stream(self, messages: List[BaseMessage], **kwargs):
        async for chunk in self.model.astream(messages):
            yield chunk.content


class OllamaLLM(BaseLLM):
    """Ollama local LLM wrapper"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.api_base or "http://localhost:11434",
        )
    
    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        response = await self.model.agenerate([messages])
        result = response.generations[0][0]
        return LLMResponse(
            content=result.text,
            model=self.config.model_name,
            usage={"total_tokens": result.text.__len__()},
        )
    
    async def generate_stream(self, messages: List[BaseMessage], **kwargs):
        async for chunk in self.model.astream(messages):
            yield chunk.content


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM wrapper"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=config.api_base or "https://api.deepseek.com",
        )
    
    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        response = await self.model.agenerate([messages])
        result = response.generations[0][0]
        return LLMResponse(
            content=result.text,
            model=self.config.model_name,
            usage={
                "prompt_tokens": response.llm_output.get("token_usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.llm_output.get("token_usage", {}).get("completion_tokens", 0),
                "total_tokens": response.llm_output.get("token_usage", {}).get("total_tokens", 0),
            },
        )
    
    async def generate_stream(self, messages: List[BaseMessage], **kwargs):
        async for chunk in self.model.astream(messages):
            yield chunk.content


class LLMManager:
    """Manages multiple LLM providers and models"""
    
    DEFAULT_CONFIGS = {
        "gpt-4o": ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=4096,
        ),
        "gpt-4o-mini": ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=4096,
        ),
        "gpt-4-turbo": ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4-turbo",
            temperature=0.7,
            max_tokens=4096,
        ),
        "claude-3-5-sonnet": ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=4096,
        ),
        "claude-3-5-haiku": ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-5-haiku-20241022",
            temperature=0.7,
            max_tokens=4096,
        ),
        "claude-3-opus": ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            temperature=0.7,
            max_tokens=4096,
        ),
        "deepseek-chat": ModelConfig(
            provider=ModelProvider.DEEPSEEK,
            model_name="deepseek-chat",
            temperature=0.7,
            max_tokens=4096,
        ),
        "llama3.1": ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="llama3.1",
            temperature=0.7,
            max_tokens=4096,
        ),
        "mistral": ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="mistral",
            temperature=0.7,
            max_tokens=4096,
        ),
        "codellama": ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="codellama",
            temperature=0.7,
            max_tokens=4096,
        ),
    }
    
    def __init__(self):
        self.llms: Dict[str, BaseLLM] = {}
        self._initialize_llms()
    
    def _initialize_llms(self):
        """Initialize available LLM instances"""
        for model_name, config in self.DEFAULT_CONFIGS.items():
            try:
                llm = self._create_llm(config)
                if llm:
                    self.llms[model_name] = llm
            except Exception as e:
                print(f"Warning: Could not initialize {model_name}: {e}")
    
    def _create_llm(self, config: ModelConfig) -> Optional[BaseLLM]:
        """Create an LLM instance based on provider"""
        try:
            if config.provider == ModelProvider.OPENAI:
                return OpenAILLM(config)
            elif config.provider == ModelProvider.ANTHROPIC:
                return AnthropicLLM(config)
            elif config.provider == ModelProvider.OLLAMA:
                return OllamaLLM(config)
            elif config.provider == ModelProvider.DEEPSEEK:
                return DeepSeekLLM(config)
        except Exception as e:
            print(f"Error creating {config.provider.value} LLM: {e}")
            return None
    
    def get_llm(self, model_name: str) -> Optional[BaseLLM]:
        """Get an LLM instance by name"""
        return self.llms.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """List all available models"""
        return list(self.llms.keys())
    
    def add_custom_model(self, name: str, config: ModelConfig):
        """Add a custom model configuration"""
        llm = self._create_llm(config)
        if llm:
            self.llms[name] = llm
    
    async def generate(
        self,
        messages: List[BaseMessage],
        model: str = "gpt-4o",
        **kwargs
    ) -> LLMResponse:
        """Generate a response using the specified model"""
        llm = self.get_llm(model)
        if not llm:
            raise ValueError(f"Model {model} not found. Available: {self.list_available_models()}")
        return await llm.generate(messages, **kwargs)
    
    async def generate_stream(
        self,
        messages: List[BaseMessage],
        model: str = "gpt-4o",
        **kwargs
    ):
        """Generate a streaming response using the specified model"""
        llm = self.get_llm(model)
        if not llm:
            raise ValueError(f"Model {model} not found. Available: {self.list_available_models()}")
        async for chunk in llm.generate_stream(messages, **kwargs):
            yield chunk


class LLMFactory:
    """Factory for creating LLM instances"""
    
    @staticmethod
    def create(
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Create an LLM instance from provider string"""
        provider_map = {
            "openai": ModelProvider.OPENAI,
            "anthropic": ModelProvider.ANTHROPIC,
            "ollama": ModelProvider.OLLAMA,
            "deepseek": ModelProvider.DEEPSEEK,
        }
        
        config = ModelConfig(
            provider=provider_map.get(provider.lower(), ModelProvider.OPENAI),
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs
        )
        
        if provider.lower() == "openai":
            return OpenAILLM(config)
        elif provider.lower() == "anthropic":
            return AnthropicLLM(config)
        elif provider.lower() == "ollama":
            return OllamaLLM(config)
        elif provider.lower() == "deepseek":
            return DeepSeekLLM(config)
        else:
            raise ValueError(f"Unknown provider: {provider}")

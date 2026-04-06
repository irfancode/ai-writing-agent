"""Base Writing Agent - Abstract base class for all writing agents"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from ..core.llm_manager import LLMManager
from ..core.memory import ConversationMemory


class ContentLength(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    EXTENDED = "extended"


class WritingTone(Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    HUMOROUS = "humorous"
    THOUGHT_LEADER = "thought_leader"
    INSPIRATIONAL = "inspirational"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    PERSUASIVE = "persuasive"


@dataclass
class AgentResponse:
    """Response from a writing agent"""
    content: str
    agent_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    tokens_used: int = 0
    generation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "agent_name": self.agent_name,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "tokens_used": self.tokens_used,
            "generation_time": self.generation_time,
        }


@dataclass
class WritingRequirements:
    """Requirements for content generation"""
    topic: str
    tone: Optional[WritingTone] = None
    audience: Optional[str] = None
    length: ContentLength = ContentLength.MEDIUM
    keywords: List[str] = field(default_factory=list)
    style: Optional[str] = None
    goal: Optional[str] = None
    additional_context: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def to_prompt_context(self) -> str:
        """Convert requirements to prompt context"""
        parts = []
        
        if self.tone:
            parts.append(f"Tone: {self.tone.value}")
        
        if self.audience:
            parts.append(f"Target Audience: {self.audience}")
        
        if self.keywords:
            parts.append(f"Keywords to include: {', '.join(self.keywords)}")
        
        if self.style:
            parts.append(f"Writing Style: {self.style}")
        
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        
        if self.additional_context:
            parts.append(f"Additional Context: {self.additional_context}")
        
        return "\n".join(parts)


class BaseWritingAgent(ABC):
    """Abstract base class for all writing agents"""
    
    name: str = "base_agent"
    description: str = "Base writing agent"
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        memory: Optional[ConversationMemory] = None,
        model: str = "claude-3-5-sonnet",
    ):
        self.llm_manager = llm_manager or LLMManager()
        self.memory = memory
        self.model = model
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    @abstractmethod
    def get_content_prompt(
        self,
        requirements: WritingRequirements
    ) -> str:
        """Get the content generation prompt"""
        pass
    
    async def generate(
        self,
        topic: str,
        tone: Optional[WritingTone] = None,
        audience: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        length: ContentLength = ContentLength.MEDIUM,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate content based on requirements"""
        import time
        start_time = time.time()
        
        requirements = WritingRequirements(
            topic=topic,
            tone=tone,
            audience=audience,
            keywords=keywords or [],
            length=length,
            additional_context=additional_context,
        )
        
        system_prompt = self.get_system_prompt()
        content_prompt = self.get_content_prompt(requirements)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content_prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=self._get_max_tokens(length),
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata=self._extract_metadata(response.content),
                tokens_used=response.usage.get("total_tokens", 0),
                generation_time=time.time() - start_time,
            )
        
        except Exception as e:
            return AgentResponse(
                content="",
                agent_name=self.name,
                metadata={"error": str(e)},
                warnings=[f"Generation failed: {str(e)}"],
                generation_time=time.time() - start_time,
            )
    
    def _get_max_tokens(self, length: ContentLength) -> int:
        """Get max tokens based on content length"""
        tokens_map = {
            ContentLength.SHORT: 500,
            ContentLength.MEDIUM: 1500,
            ContentLength.LONG: 3000,
            ContentLength.EXTENDED: 8000,
        }
        return tokens_map.get(length, 1500)
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from generated content"""
        return {
            "word_count": len(content.split()),
            "char_count": len(content),
            "generated_at": datetime.now().isoformat(),
        }
    
    async def improve(
        self,
        content: str,
        improvement_type: str = "clarity",
        **kwargs
    ) -> AgentResponse:
        """Improve existing content"""
        import time
        start_time = time.time()
        
        improvement_prompts = {
            "clarity": "Improve the clarity and readability of this content. Make sentences clearer and more direct.",
            "engagement": "Make this content more engaging and interesting. Add hooks and compelling language.",
            "seo": "Optimize this content for SEO. Naturally integrate keywords and improve searchability.",
            "grammar": "Check and fix any grammar, spelling, or punctuation errors.",
            "tone": "Adjust the tone to be more appropriate for the target audience while maintaining the core message.",
        }
        
        prompt = f"""Improve the following content for {improvement_type}.

{improvement_prompts.get(improvement_type, improvement_prompts['clarity'])}

Original Content:
{content}

Provide the improved version:"""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=f"{self.name}_improver",
                metadata={"improvement_type": improvement_type},
                tokens_used=response.usage.get("total_tokens", 0),
                generation_time=time.time() - start_time,
            )
        
        except Exception as e:
            return AgentResponse(
                content=content,
                agent_name=f"{self.name}_improver",
                warnings=[f"Improvement failed: {str(e)}"],
                generation_time=time.time() - start_time,
            )
    
    def validate_content(self, content: str) -> Dict[str, Any]:
        """Validate generated content"""
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "suggestions": [],
        }
        
        if len(content) < 50:
            validation_results["is_valid"] = False
            validation_results["warnings"].append("Content is very short")
        
        if not content.strip():
            validation_results["is_valid"] = False
            validation_results["warnings"].append("Content is empty")
        
        return validation_results

"""Writing Agent Orchestrator - LangGraph-powered multi-agent system"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm_manager import LLMManager
from .memory import ConversationMemory


class ContentType(Enum):
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    LINKEDIN = "linkedin"
    BLOG = "blog"
    TECHNICAL = "technical"
    DOCUMENTATION = "documentation"


class Tone(Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    HUMOROUS = "humorous"
    THOUGHT_LEADER = "thought_leader"
    INSPIRATIONAL = "inspirational"


class WritingAgentState(BaseModel):
    """State for the writing agent graph"""
    messages: List[BaseMessage] = Field(default_factory=list)
    content_type: Optional[ContentType] = None
    topic: Optional[str] = None
    tone: Optional[Tone] = None
    audience: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    generated_content: Optional[str] = None
    optimized_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    current_step: str = "start"
    iterations: int = 0
    error: Optional[str] = None


@dataclass
class GenerationResult:
    """Result from content generation"""
    content: str
    content_type: ContentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    generation_time: float = 0.0
    model: str = ""
    iterations: int = 1
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "content_type": self.content_type.value,
            "metadata": self.metadata,
            "tokens_used": self.tokens_used,
            "generation_time": self.generation_time,
            "model": self.model,
            "iterations": self.iterations,
            "warnings": self.warnings,
        }


class WritingAgentOrchestrator:
    """Orchestrates multiple writing agents using LangGraph"""
    
    def __init__(
        self,
        primary_model: str = "claude-3-5-sonnet",
        secondary_model: str = "gpt-4o",
        max_iterations: int = 3,
        memory_persist_path: Optional[str] = None,
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.max_iterations = max_iterations
        
        self.llm_manager = LLMManager()
        self.memory = ConversationMemory(persist_path=memory_persist_path)
        
        self.agents: Dict[ContentType, Any] = {}
        self.graph = self._build_graph()
    
    def register_agent(self, content_type: ContentType, agent: Any):
        """Register a writing agent for a specific content type"""
        self.agents[content_type] = agent
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        async def route_content_type(state: WritingAgentState) -> str:
            """Route to the appropriate agent based on content type"""
            return state.content_type.value if state.content_type else END
        
        async def generate_content(state: WritingAgentState) -> WritingAgentState:
            """Generate content using the appropriate agent"""
            if not state.content_type or state.content_type not in self.agents:
                state.error = f"No agent registered for {state.content_type}"
                return state
            
            agent = self.agents[state.content_type]
            try:
                result = await agent.generate(
                    topic=state.topic,
                    tone=state.tone,
                    audience=state.audience,
                    keywords=state.keywords,
                )
                state.generated_content = result.content
                state.metadata.update(result.metadata)
            except Exception as e:
                state.error = str(e)
            
            return state
        
        async def optimize_content(state: WritingAgentState) -> WritingAgentState:
            """Optimize the generated content"""
            if state.generated_content and not state.error:
                try:
                    system_prompt = self._get_optimization_prompt()
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Content to optimize:\n\n{state.generated_content}"),
                    ]
                    response = await self.llm_manager.generate(
                        messages,
                        model=self.secondary_model,
                    )
                    state.optimized_content = response.content
                except Exception as e:
                    state.warnings = state.warnings + [f"Optimization failed: {str(e)}"]
                    state.optimized_content = state.generated_content
            return state
        
        async def finalize_content(state: WritingAgentState) -> WritingAgentState:
            """Finalize the content"""
            if state.optimized_content:
                state.generated_content = state.optimized_content
            return state
        
        graph = StateGraph(WritingAgentState)
        
        graph.add_node("route", route_content_type)
        graph.add_node("generate", generate_content)
        graph.add_node("optimize", optimize_content)
        graph.add_node("finalize", finalize_content)
        
        graph.set_entry_point("route")
        graph.add_edge("route", "generate")
        graph.add_edge("generate", "optimize")
        graph.add_edge("optimize", "finalize")
        graph.add_edge("finalize", END)
        
        return graph.compile()
    
    def _get_optimization_prompt(self) -> str:
        return """You are an expert content optimizer. Your task is to improve the content while maintaining its core message and intent.

        Improve the content by:
        1. Enhancing clarity and readability
        2. Improving flow and transitions
        3. Making it more engaging
        4. Fixing any grammar or style issues
        5. Ensuring consistent tone throughout

        Do not add new information or change the meaning. Only refine and improve."""
    
    async def generate(
        self,
        content_type: ContentType,
        topic: str,
        tone: Optional[Tone] = None,
        audience: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        additional_instructions: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> GenerationResult:
        """Generate content using the orchestration pipeline"""
        import time
        start_time = time.time()
        
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.memory.create_session(
            session_id=session_id,
            topic=topic,
            tone=tone.value if tone else None,
            audience=audience,
            content_type=content_type.value,
        )
        
        initial_state = WritingAgentState(
            content_type=content_type,
            topic=topic,
            tone=tone,
            audience=audience,
            keywords=keywords or [],
            metadata={
                "additional_instructions": additional_instructions,
                "session_id": session_id,
            },
        )
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            if final_state.error:
                return GenerationResult(
                    content="",
                    content_type=content_type,
                    metadata={"error": final_state.error},
                    generation_time=time.time() - start_time,
                )
            
            return GenerationResult(
                content=final_state.generated_content or "",
                content_type=content_type,
                metadata=final_state.metadata,
                generation_time=time.time() - start_time,
                model=self.primary_model,
                iterations=final_state.iterations,
            )
        
        except Exception as e:
            return GenerationResult(
                content="",
                content_type=content_type,
                metadata={"error": str(e)},
                generation_time=time.time() - start_time,
            )
    
    async def generate_variants(
        self,
        content_type: ContentType,
        topic: str,
        tone: Optional[Tone] = None,
        audience: Optional[str] = None,
        num_variants: int = 3,
        **kwargs
    ) -> List[GenerationResult]:
        """Generate multiple content variants"""
        tasks = []
        for i in range(num_variants):
            tasks.append(
                self.generate(
                    content_type=content_type,
                    topic=topic,
                    tone=tone,
                    audience=audience,
                    additional_instructions=f"Variant {i+1}/{num_variants}",
                    session_id=f"variant_{i+1}",
                    **kwargs
                )
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, GenerationResult)]
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.llm_manager.list_available_models()
    
    def set_primary_model(self, model: str):
        """Set the primary model"""
        if model in self.llm_manager.list_available_models():
            self.primary_model = model
        else:
            raise ValueError(f"Model {model} not available")
    
    def set_secondary_model(self, model: str):
        """Set the secondary model"""
        if model in self.llm_manager.list_available_models():
            self.secondary_model = model
        else:
            raise ValueError(f"Model {model} not available")


class WritingPipeline:
    """Sequential pipeline for content generation and processing"""
    
    def __init__(self, orchestrator: WritingAgentOrchestrator):
        self.orchestrator = orchestrator
        self.processors: List[Callable] = []
    
    def add_processor(self, processor: Callable):
        """Add a processor to the pipeline"""
        self.processors.append(processor)
    
    async def execute(
        self,
        content_type: ContentType,
        topic: str,
        **kwargs
    ) -> GenerationResult:
        """Execute the pipeline"""
        result = await self.orchestrator.generate(
            content_type=content_type,
            topic=topic,
            **kwargs
        )
        
        for processor in self.processors:
            result = await processor(result)
        
        return result


class AgentRouter:
    """Intelligent routing based on content requirements"""
    
    CONTENT_TYPE_PATTERNS = {
        ContentType.SOCIAL_MEDIA: ["twitter", "instagram", "facebook", "tweet", "post", "social"],
        ContentType.EMAIL: ["email", "mail", "newsletter", "cold", "outreach"],
        ContentType.LINKEDIN: ["linkedin", "professional", "network", "connection"],
        ContentType.BLOG: ["blog", "article", "post", "write", "blogpost"],
        ContentType.TECHNICAL: ["technical", "research", "analysis", "study", "paper"],
        ContentType.DOCUMENTATION: ["doc", "documentation", "readme", "guide", "manual"],
    }
    
    @classmethod
    def route(cls, query: str) -> ContentType:
        """Route a query to the appropriate content type"""
        query_lower = query.lower()
        
        for content_type, patterns in cls.CONTENT_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return content_type
        
        return ContentType.BLOG
    
    @classmethod
    def extract_requirements(cls, query: str) -> Dict[str, Any]:
        """Extract requirements from a natural language query"""
        requirements = {
            "content_type": cls.route(query),
            "topic": query,
            "tone": None,
            "audience": None,
            "keywords": [],
        }
        
        tone_keywords = {
            "professional": ["professional", "formal", "business"],
            "casual": ["casual", "relaxed", "informal"],
            "friendly": ["friendly", "warm", "approachable"],
            "humorous": ["funny", "humor", "entertaining"],
            "thought_leader": ["thought leader", "insightful", "expert"],
        }
        
        for tone, keywords in tone_keywords.items():
            for keyword in keywords:
                if keyword in query.lower():
                    requirements["tone"] = Tone[tone.upper()]
                    break
        
        return requirements

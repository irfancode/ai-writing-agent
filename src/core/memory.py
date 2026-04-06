"""Conversation Memory for AI Writing Agent"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
import json
import os


@dataclass
class MemoryEntry:
    """A single memory entry"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationContext:
    """Context for a conversation session"""
    session_id: str
    topic: Optional[str] = None
    tone: Optional[str] = None
    audience: Optional[str] = None
    content_type: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "tone": self.tone,
            "audience": self.audience,
            "content_type": self.content_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 100, persist_path: Optional[str] = None):
        self.max_history = max_history
        self.persist_path = persist_path
        self.conversations: Dict[str, List[MemoryEntry]] = {}
        self.contexts: Dict[str, ConversationContext] = {}
        
        if persist_path and os.path.exists(persist_path):
            self._load()
    
    def create_session(
        self,
        session_id: str,
        topic: Optional[str] = None,
        tone: Optional[str] = None,
        audience: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> ConversationContext:
        """Create a new conversation session"""
        context = ConversationContext(
            session_id=session_id,
            topic=topic,
            tone=tone,
            audience=audience,
            content_type=content_type,
        )
        self.contexts[session_id] = context
        self.conversations[session_id] = []
        return context
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a message to the conversation history"""
        if session_id not in self.conversations:
            self.create_session(session_id)
        
        entry = MemoryEntry(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.conversations[session_id].append(entry)
        
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
        
        if session_id in self.contexts:
            self.contexts[session_id].updated_at = datetime.now()
        
        self._persist()
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Get conversation messages"""
        if session_id not in self.conversations:
            return []
        
        messages = self.conversations[session_id]
        if limit:
            return messages[-limit:]
        return messages
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context"""
        return self.contexts.get(session_id)
    
    def get_messages_for_llm(
        self,
        session_id: str,
        system_prompt: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BaseMessage]:
        """Get messages formatted for LLM consumption"""
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        entries = self.get_messages(session_id, limit)
        
        for entry in entries:
            if entry.role == "user":
                messages.append(HumanMessage(content=entry.content))
            elif entry.role == "assistant":
                messages.append(AIMessage(content=entry.content))
            elif entry.role == "system":
                messages.append(SystemMessage(content=entry.content))
        
        return messages
    
    def update_context(
        self,
        session_id: str,
        **kwargs
    ):
        """Update conversation context"""
        if session_id in self.contexts:
            context = self.contexts[session_id]
            for key, value in kwargs.items():
                if hasattr(context, key):
                    setattr(context, key, value)
            context.updated_at = datetime.now()
            self._persist()
    
    def clear_session(self, session_id: str):
        """Clear a conversation session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
        if session_id in self.contexts:
            del self.contexts[session_id]
        self._persist()
    
    def _persist(self):
        """Persist memory to disk"""
        if not self.persist_path:
            return
        
        data = {
            "conversations": {
                sid: [e.to_dict() for e in entries]
                for sid, entries in self.conversations.items()
            },
            "contexts": {
                sid: ctx.to_dict()
                for sid, ctx in self.contexts.items()
            },
        }
        
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load memory from disk"""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            
            self.conversations = {
                sid: [MemoryEntry.from_dict(e) for e in entries]
                for sid, entries in data.get("conversations", {}).items()
            }
            
            self.contexts = {
                sid: ConversationContext(**ctx)
                for sid, ctx in data.get("contexts", {}).items()
            }
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def get_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation session"""
        messages = self.get_messages(session_id)
        return {
            "total_messages": len(messages),
            "user_messages": sum(1 for m in messages if m.role == "user"),
            "assistant_messages": sum(1 for m in messages if m.role == "assistant"),
            "total_tokens_estimate": sum(len(m.content.split()) for m in messages),
        }


class VectorMemory:
    """Simple in-memory vector store for document retrieval"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents with their embeddings"""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
    ) -> List[Document]:
        """Simple similarity search using cosine similarity"""
        if not self.embeddings:
            return []
        
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, embedding)
            similarities.append((sim, i))
        
        similarities.sort(reverse=True)
        top_k = similarities[:k]
        
        return [self.documents[i] for _, i in top_k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def clear(self):
        """Clear all documents"""
        self.documents = []
        self.embeddings = []

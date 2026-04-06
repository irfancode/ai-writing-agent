"""Tests for the AI Writing Agent"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch


class TestLLMManager:
    """Test cases for LLM Manager"""
    
    def test_default_configs_exist(self):
        """Test that default model configurations exist"""
        from src.core.llm_manager import LLMManager
        
        manager = LLMManager()
        models = manager.list_available_models()
        
        assert isinstance(models, list)
    
    def test_get_unavailable_model_returns_none(self):
        """Test getting a model that doesn't exist"""
        from src.core.llm_manager import LLMManager
        
        manager = LLMManager()
        result = manager.get_llm("nonexistent-model-xyz")
        
        assert result is None


class TestMemory:
    """Test cases for Conversation Memory"""
    
    def test_create_session(self):
        """Test creating a conversation session"""
        from src.core.memory import ConversationMemory
        
        memory = ConversationMemory()
        context = memory.create_session(
            session_id="test-123",
            topic="Test topic",
            tone="professional",
        )
        
        assert context.session_id == "test-123"
        assert context.topic == "Test topic"
    
    def test_add_message(self):
        """Test adding a message to memory"""
        from src.core.memory import ConversationMemory
        
        memory = ConversationMemory()
        memory.create_session("test-456")
        memory.add_message("test-456", "user", "Hello")
        memory.add_message("test-456", "assistant", "Hi there")
        
        messages = memory.get_messages("test-456")
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there"
    
    def test_clear_session(self):
        """Test clearing a session"""
        from src.core.memory import ConversationMemory
        
        memory = ConversationMemory()
        memory.create_session("test-789")
        memory.add_message("test-789", "user", "Test")
        
        memory.clear_session("test-789")
        messages = memory.get_messages("test-789")
        assert len(messages) == 0


class TestBaseAgent:
    """Test cases for Base Writing Agent"""
    
    def test_validation_empty_content(self):
        """Test validation of empty content"""
        from src.agents.base_agent import BaseWritingAgent
        
        class DummyAgent(BaseWritingAgent):
            def get_system_prompt(self):
                return ""
            
            def get_content_prompt(self, requirements):
                return ""
        
        agent = DummyAgent()
        result = agent.validate_content("")
        
        assert result["is_valid"] is False
        assert len(result["warnings"]) > 0
    
    def test_validation_short_content(self):
        """Test validation of short content"""
        from src.agents.base_agent import BaseWritingAgent
        
        class DummyAgent(BaseWritingAgent):
            def get_system_prompt(self):
                return ""
            
            def get_content_prompt(self, requirements):
                return ""
        
        agent = DummyAgent()
        result = agent.validate_content("Hi")
        
        assert result["is_valid"] is False
        assert "short" in result["warnings"][0].lower()
    
    def test_validation_good_content(self):
        """Test validation of good content"""
        from src.agents.base_agent import BaseWritingAgent
        
        class DummyAgent(BaseWritingAgent):
            def get_system_prompt(self):
                return ""
            
            def get_content_prompt(self, requirements):
                return ""
        
        agent = DummyAgent()
        result = agent.validate_content("This is a reasonably long piece of content that should pass validation.")
        
        assert result["is_valid"] is True


class TestSocialMediaAgent:
    """Test cases for Social Media Agent"""
    
    def test_agent_initialization(self):
        """Test agent can be initialized"""
        from src.agents.social_media import SocialMediaAgent
        
        agent = SocialMediaAgent()
        assert agent.name == "social_media_agent"
    
    def test_platform_enum(self):
        """Test platform enum values"""
        from src.agents.social_media import Platform
        
        assert Platform.TWITTER.value == "twitter"
        assert Platform.LINKEDIN.value == "linkedin"
        assert Platform.INSTAGRAM.value == "instagram"


class TestEmailAgent:
    """Test cases for Email Agent"""
    
    def test_agent_initialization(self):
        """Test agent can be initialized"""
        from src.agents.email import EmailAgent
        
        agent = EmailAgent()
        assert agent.name == "email_agent"
    
    def test_email_type_enum(self):
        """Test email type enum values"""
        from src.agents.email import EmailType
        
        assert EmailType.COLD_OUTREACH.value == "cold_outreach"
        assert EmailType.NEWSLETTER.value == "newsletter"


class TestLinkedInAgent:
    """Test cases for LinkedIn Agent"""
    
    def test_agent_initialization(self):
        """Test agent can be initialized"""
        from src.agents.linkedin import LinkedInAgent
        
        agent = LinkedInAgent()
        assert agent.name == "linkedin_agent"


class TestBlogAgent:
    """Test cases for Blog Agent"""
    
    def test_agent_initialization(self):
        """Test agent can be initialized"""
        from src.agents.blog import BlogAgent
        
        agent = BlogAgent()
        assert agent.name == "blog_agent"
    
    def test_blog_type_structures(self):
        """Test blog type structures exist"""
        from src.agents.blog import BlogAgent, BlogType
        
        agent = BlogAgent()
        assert BlogType.HOW_TO in agent.BLOG_TYPE_STRUCTURES
        assert "sections" in agent.BLOG_TYPE_STRUCTURES[BlogType.HOW_TO]


class TestTechnicalAgent:
    """Test cases for Technical Agent"""
    
    def test_agent_initialization(self):
        """Test agent can be initialized"""
        from src.agents.technical import TechnicalAgent
        
        agent = TechnicalAgent()
        assert agent.name == "technical_agent"


class TestDocumentationAgent:
    """Test cases for Documentation Agent"""
    
    def test_agent_initialization(self):
        """Test agent can be initialized"""
        from src.agents.documentation import DocumentationAgent
        
        agent = DocumentationAgent()
        assert agent.name == "documentation_agent"


class TestOptimizer:
    """Test cases for Content Optimizer"""
    
    def test_readability_calculation(self):
        """Test readability score calculation"""
        from src.utils.optimizer import ContentOptimizer
        
        optimizer = ContentOptimizer()
        
        simple_text = "This is a simple sentence. It has short words."
        result = optimizer.calculate_readability(simple_text)
        
        assert "flesch_reading_ease" in result
        assert "flesch_kincaid_grade" in result
        assert result["total_words"] > 0


class TestSEOEnhancer:
    """Test cases for SEO Enhancer"""
    
    def test_keyword_density_calculation(self):
        """Test keyword density calculation"""
        from src.utils.seo import SEOEnhancer
        
        enhancer = SEOEnhancer()
        
        content = "AI writing tools are changing how we write. AI makes writing faster."
        density = enhancer.calculate_keyword_density(content, "AI")
        
        assert density > 0
        assert density <= 100


class TestExporter:
    """Test cases for Content Exporter"""
    
    def test_supported_formats(self):
        """Test supported export formats"""
        from src.utils.exporter import ContentExporter
        
        assert "md" in ContentExporter.SUPPORTED_FORMATS
        assert "html" in ContentExporter.SUPPORTED_FORMATS
        assert "txt" in ContentExporter.SUPPORTED_FORMATS
        assert "json" in ContentExporter.SUPPORTED_FORMATS
    
    def test_markdown_to_html(self):
        """Test basic markdown to HTML conversion"""
        from src.utils.exporter import ContentExporter
        
        exporter = ContentExporter()
        
        md_content = "# Hello\n\nThis is **bold** text."
        html = exporter._markdown_to_html(md_content)
        
        assert "<h1>" in html
        assert "<strong>" in html
    
    def test_strip_markdown(self):
        """Test markdown stripping"""
        from src.utils.exporter import ContentExporter
        
        exporter = ContentExporter()
        
        md_content = "# Title\n\nThis is **bold** and *italic*."
        text = exporter._strip_markdown(md_content)
        
        assert "#" not in text
        assert "**" not in text
        assert "Title" in text


class TestOrchestrator:
    """Test cases for Writing Agent Orchestrator"""
    
    def test_content_type_enum(self):
        """Test content type enum values"""
        from src.core.orchestrator import ContentType
        
        assert ContentType.SOCIAL_MEDIA.value == "social_media"
        assert ContentType.EMAIL.value == "email"
        assert ContentType.BLOG.value == "blog"


class TestAgentRouter:
    """Test cases for Agent Router"""
    
    def test_route_social_keywords(self):
        """Test routing for social media keywords"""
        from src.core.orchestrator import AgentRouter, ContentType
        
        result = AgentRouter.route("I want to post on twitter")
        assert result == ContentType.SOCIAL_MEDIA
    
    def test_route_email_keywords(self):
        """Test routing for email keywords"""
        from src.core.orchestrator import AgentRouter, ContentType
        
        result = AgentRouter.route("I need to write a cold email")
        assert result == ContentType.EMAIL
    
    def test_route_blog_keywords(self):
        """Test routing for blog keywords"""
        from src.core.orchestrator import AgentRouter, ContentType
        
        result = AgentRouter.route("Write an article about AI")
        assert result == ContentType.BLOG
    
    def test_route_technical_keywords(self):
        """Test routing for technical keywords"""
        from src.core.orchestrator import AgentRouter, ContentType
        
        result = AgentRouter.route("I need technical research")
        assert result == ContentType.TECHNICAL
    
    def test_route_default(self):
        """Test default routing"""
        from src.core.orchestrator import AgentRouter, ContentType
        
        result = AgentRouter.route("Hello world")
        assert result == ContentType.BLOG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

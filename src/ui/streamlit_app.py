"""Streamlit Web UI for AI Writing Agent"""

import asyncio
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ..core.llm_manager import LLMManager
from ..agents.social_media import SocialMediaAgent, Platform, WritingTone
from ..agents.email import EmailAgent, EmailType
from ..agents.linkedin import LinkedInAgent, LinkedInPostStyle
from ..agents.blog import BlogAgent, BlogType
from ..agents.technical import TechnicalAgent, TechnicalDepth
from ..agents.documentation import DocumentationAgent
from ..utils.optimizer import ContentOptimizer, OptimizationTarget
from ..utils.seo import SEOEnhancer


st.set_page_config(
    page_title="AI Writing Agent",
    page_icon="🤖",
    layout="wide",
)


class StreamlitUI:
    """Streamlit-based UI for AI Writing Agent"""
    
    def __init__(self):
        if "llm_manager" not in st.session_state:
            st.session_state.llm_manager = LLMManager()
        
        if "agents" not in st.session_state:
            st.session_state.agents = {
                "social": SocialMediaAgent(llm_manager=st.session_state.llm_manager),
                "email": EmailAgent(llm_manager=st.session_state.llm_manager),
                "linkedin": LinkedInAgent(llm_manager=st.session_state.llm_manager),
                "blog": BlogAgent(llm_manager=st.session_state.llm_manager),
                "technical": TechnicalAgent(llm_manager=st.session_state.llm_manager),
                "docs": DocumentationAgent(llm_manager=st.session_state.llm_manager),
            }
        
        if "optimizer" not in st.session_state:
            st.session_state.optimizer = ContentOptimizer(
                llm_manager=st.session_state.llm_manager
            )
        
        if "seo_enhancer" not in st.session_state:
            st.session_state.seo_enhancer = SEOEnhancer(
                llm_manager=st.session_state.llm_manager
            )
        
        if "history" not in st.session_state:
            st.session_state.history = []
    
    @staticmethod
    @st.cache_resource
    def get_models():
        """Get available models"""
        return st.session_state.llm_manager.list_available_models()


def render_social_media(ui: StreamlitUI):
    """Render social media generation tab"""
    st.header("📱 Social Media Content")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        topic = st.text_input("Topic", placeholder="What's your post about?")
        platform = st.selectbox("Platform", ["Twitter", "Instagram", "LinkedIn", "Facebook"])
        tone = st.selectbox("Tone", ["Friendly", "Professional", "Humorous", "Casual"])
        include_hashtags = st.checkbox("Include Hashtags", value=True)
        
        if st.button("Generate Post", type="primary", use_container_width=True):
            if topic:
                with st.spinner("Generating..."):
                    result = asyncio.run(
                        ui.agents["social"].generate_post(
                            topic=topic,
                            platform=Platform(platform.lower()),
                            tone=WritingTone[tone.upper()],
                            include_hashtags=include_hashtags,
                        )
                    )
                    st.session_state.history.append(("Social", topic, result.content))
                    st.rerun()
    
    with col2:
        if st.session_state.history:
            st.subheader("Recent Generation")
            recent = st.session_state.history[-1]
            st.text_area("Output", value=recent[2], height=200, disabled=True)


def render_email(ui: StreamlitUI):
    """Render email generation tab"""
    st.header("📧 Email Content")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        topic = st.text_input("Email Purpose", placeholder="What do you want to communicate?")
        email_type = st.selectbox("Email Type", ["Cold Outreach", "Follow Up", "Newsletter", "Meeting Request"])
        recipient = st.text_input("Recipient (Optional)", placeholder="John Smith, CTO")
        goal = st.text_input("Goal (Optional)", placeholder="Schedule a demo")
        
        if st.button("Generate Email", type="primary", use_container_width=True):
            if topic:
                with st.spinner("Generating..."):
                    email_type_map = {
                        "Cold Outreach": "cold_outreach",
                        "Follow Up": "follow_up",
                        "Newsletter": "newsletter",
                        "Meeting Request": "meeting_request",
                    }
                    result = asyncio.run(
                        ui.agents["email"].generate_email(
                            topic=topic,
                            email_type=EmailType(email_type_map[email_type]),
                            recipient_name=recipient if recipient else None,
                            goal=goal if goal else None,
                        )
                    )
                    st.session_state.history.append(("Email", topic, result.content))
                    st.rerun()
    
    with col2:
        if st.session_state.history:
            st.subheader("Recent Generation")
            recent = st.session_state.history[-1]
            st.text_area("Output", value=recent[2], height=250, disabled=True)


def render_linkedin(ui: StreamlitUI):
    """Render LinkedIn content tab"""
    st.header("💼 LinkedIn Content")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        topic = st.text_input("Topic", placeholder="What do you want to share?")
        style = st.selectbox(
            "Post Style",
            ["Thought Leadership", "Personal Story", "Industry Insight", "How To", "Listicle"]
        )
        tone = st.selectbox("Tone", ["Professional", "Friendly", "Casual"])
        
        style_map = {
            "Thought Leadership": "thought_leadership",
            "Personal Story": "personal_story",
            "Industry Insight": "industry_insight",
            "How To": "how_to",
            "Listicle": "listicle",
        }
        
        if st.button("Generate Post", type="primary", use_container_width=True):
            if topic:
                with st.spinner("Generating..."):
                    result = asyncio.run(
                        ui.agents["linkedin"].generate_post(
                            topic=topic,
                            style=LinkedInPostStyle(style_map[style]),
                            tone=WritingTone[tone.upper()],
                        )
                    )
                    st.session_state.history.append(("LinkedIn", topic, result.content))
                    st.rerun()
    
    with col2:
        if st.session_state.history:
            st.subheader("Recent Generation")
            recent = st.session_state.history[-1]
            st.text_area("Output", value=recent[2], height=200, disabled=True)


def render_blog(ui: StreamlitUI):
    """Render blog generation tab"""
    st.header("📝 Blog Articles")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        topic = st.text_input("Article Topic", placeholder="What should the article be about?")
        blog_type = st.selectbox("Article Type", ["How To", "Listicle", "Tutorial", "Opinion", "Case Study"])
        tone = st.selectbox("Tone", ["Casual", "Professional", "Technical"])
        word_count = st.slider("Target Word Count", 500, 3000, 1500, 100)
        keywords = st.text_input("Keywords (comma-separated)", placeholder="AI, writing, productivity")
        
        blog_type_map = {
            "How To": "how_to",
            "Listicle": "listicle",
            "Tutorial": "tutorial",
            "Opinion": "opinion",
            "Case Study": "case_study",
        }
        
        if st.button("Generate Article", type="primary", use_container_width=True):
            if topic:
                with st.spinner("Generating article (this may take a minute)..."):
                    kw_list = [k.strip() for k in keywords.split(",")] if keywords else None
                    result = asyncio.run(
                        ui.agents["blog"].generate_post(
                            topic=topic,
                            blog_type=BlogType(blog_type_map[blog_type]),
                            tone=WritingTone[tone.upper()],
                            word_count=word_count,
                            keywords=kw_list,
                        )
                    )
                    st.session_state.history.append(("Blog", topic, result.content))
                    st.rerun()
    
    with col2:
        if st.session_state.history:
            st.subheader("Recent Generation")
            recent = st.session_state.history[-1]
            st.text_area("Output", value=recent[2], height=400, disabled=True)


def render_technical(ui: StreamlitUI):
    """Render technical content tab"""
    st.header("🔬 Technical Content")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        topic = st.text_input("Technical Topic", placeholder="What technical concept?")
        depth = st.selectbox("Depth", ["Surface", "Intermediate", "Deep", "Expert"])
        
        depth_map = {
            "Surface": "surface",
            "Intermediate": "intermediate",
            "Deep": "deep",
            "Expert": "expert",
        }
        
        if st.button("Generate", type="primary", use_container_width=True):
            if topic:
                with st.spinner("Researching and generating..."):
                    result = asyncio.run(
                        ui.agents["technical"].generate_research_summary(
                            topic=topic,
                            depth=TechnicalDepth(depth_map[depth]),
                        )
                    )
                    st.session_state.history.append(("Technical", topic, result.content))
                    st.rerun()
    
    with col2:
        if st.session_state.history:
            st.subheader("Recent Generation")
            recent = st.session_state.history[-1]
            st.text_area("Output", value=recent[2], height=400, disabled=True)


def render_docs(ui: StreamlitUI):
    """Render documentation tab"""
    st.header("📚 Documentation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        project_name = st.text_input("Project Name", placeholder="My Awesome Project")
        doc_type = st.selectbox("Document Type", ["README", "User Guide", "API Docs"])
        description = st.text_area("Description", placeholder="What does this project do?", height=100)
        
        if st.button("Generate", type="primary", use_container_width=True):
            if project_name:
                with st.spinner("Generating documentation..."):
                    result = asyncio.run(
                        ui.agents["docs"].generate_readme(
                            project_name=project_name,
                            description=description,
                        )
                    )
                    st.session_state.history.append(("Docs", project_name, result.content))
                    st.rerun()
    
    with col2:
        if st.session_state.history:
            st.subheader("Recent Generation")
            recent = st.session_state.history[-1]
            st.text_area("Output", value=recent[2], height=400, disabled=True)


def render_optimizer(ui: StreamlitUI):
    """Render optimization tab"""
    st.header("✨ Content Optimizer")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        content = st.text_area("Content to Optimize", height=200, placeholder="Paste your content here...")
        target = st.selectbox("Optimization Target", ["Readability", "Engagement", "SEO", "Professional", "Clarity"])
        
        target_map = {
            "Readability": "readability",
            "Engagement": "engagement",
            "SEO": "seo",
            "Professional": "professional",
            "Clarity": "clarity",
        }
        
        if st.button("Optimize", type="primary", use_container_width=True):
            if content:
                with st.spinner("Optimizing..."):
                    result = asyncio.run(
                        ui.optimizer.optimize(
                            content=content,
                            target=OptimizationTarget(target_map[target]),
                        )
                    )
                    st.session_state.history.append(("Optimize", target, result))
                    st.rerun()
    
    with col2:
        st.subheader("Optimized Content")
        if st.session_state.history:
            recent = st.session_state.history[-1]
            st.text_area("Output", value=recent[2], height=250, disabled=True)


def main():
    """Main Streamlit app"""
    st.title("🤖 AI Writing Agent")
    st.caption("Your personal content creation assistant")
    
    ui = StreamlitUI()
    
    tabs = st.tabs([
        "📱 Social Media",
        "📧 Email",
        "💼 LinkedIn",
        "📝 Blog",
        "🔬 Technical",
        "📚 Documentation",
        "✨ Optimize",
    ])
    
    with tabs[0]:
        render_social_media(ui)
    with tabs[1]:
        render_email(ui)
    with tabs[2]:
        render_linkedin(ui)
    with tabs[3]:
        render_blog(ui)
    with tabs[4]:
        render_technical(ui)
    with tabs[5]:
        render_docs(ui)
    with tabs[6]:
        render_optimizer(ui)
    
    with st.sidebar:
        st.header("Settings")
        models = ui.get_models()
        selected_model = st.selectbox("Model", models if models else ["No models available"])
        
        st.header("About")
        st.info("""
        **AI Writing Agent**
        
        Generate high-quality content for:
        - Social Media
        - Email
        - LinkedIn
        - Blog
        - Technical
        - Documentation
        
        Built with LangChain, LangGraph, and Streamlit.
        """)


if __name__ == "__main__":
    main()

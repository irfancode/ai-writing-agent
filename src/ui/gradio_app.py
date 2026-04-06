"""Gradio Web UI for AI Writing Agent"""

import asyncio
import os
from typing import Optional
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from ..core.orchestrator import WritingAgentOrchestrator, ContentType, Tone
from ..core.llm_manager import LLMManager
from ..agents.social_media import SocialMediaAgent, Platform, PostType, WritingTone
from ..agents.email import EmailAgent, EmailType
from ..agents.linkedin import LinkedInAgent, LinkedInPostStyle
from ..agents.blog import BlogAgent, BlogType
from ..agents.technical import TechnicalAgent, TechnicalDepth
from ..agents.documentation import DocumentationAgent, DocType
from ..utils.optimizer import ContentOptimizer, OptimizationTarget
from ..utils.seo import SEOEnhancer
from ..utils.exporter import ContentExporter, ExportConfig


class WritingAgentUI:
    """Gradio-based UI for AI Writing Agent"""
    
    def __init__(self):
        self.llm_manager = LLMManager()
        self.orchestrator = WritingAgentOrchestrator(
            primary_model=os.getenv("DEFAULT_MODEL", "gpt-4o"),
            secondary_model=os.getenv("SECONDARY_MODEL", "claude-3-5-sonnet"),
        )
        
        self.agents = {
            "social": SocialMediaAgent(llm_manager=self.llm_manager),
            "email": EmailAgent(llm_manager=self.llm_manager),
            "linkedin": LinkedInAgent(llm_manager=self.llm_manager),
            "blog": BlogAgent(llm_manager=self.llm_manager),
            "technical": TechnicalAgent(llm_manager=self.llm_manager),
            "docs": DocumentationAgent(llm_manager=self.llm_manager),
        }
        
        self.optimizer = ContentOptimizer(llm_manager=self.llm_manager)
        self.seo_enhancer = SEOEnhancer(llm_manager=self.llm_manager)
        self.exporter = ContentExporter(ExportConfig())
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        return self.llm_manager.list_available_models()
    
    async def generate_social(self, topic, platform, tone, include_hashtags):
        """Generate social media content"""
        agent = self.agents["social"]
        result = await agent.generate_post(
            topic=topic,
            platform=Platform(platform.lower()),
            tone=WritingTone[tone.upper()],
            include_hashtags=include_hashtags,
        )
        return result.content, f"Generated {len(result.content.split())} words"
    
    async def generate_email(self, topic, email_type, recipient, goal):
        """Generate email content"""
        agent = self.agents["email"]
        result = await agent.generate_email(
            topic=topic,
            email_type=EmailType(email_type),
            recipient_name=recipient if recipient else None,
            goal=goal if goal else None,
        )
        return result.content, f"Subject: {result.metadata.get('subject', 'N/A')}"
    
    async def generate_linkedin(self, topic, style, tone):
        """Generate LinkedIn content"""
        agent = self.agents["linkedin"]
        result = await agent.generate_post(
            topic=topic,
            style=LinkedInPostStyle(style.lower()),
            tone=WritingTone[tone.upper()],
        )
        return result.content, f"Generated {len(result.content.split())} words"
    
    async def generate_blog(self, topic, blog_type, tone, word_count, keywords):
        """Generate blog post"""
        agent = self.agents["blog"]
        kw_list = [k.strip() for k in keywords.split(",")] if keywords else None
        result = await agent.generate_post(
            topic=topic,
            blog_type=BlogType(blog_type.lower()),
            tone=WritingTone[tone.upper()],
            word_count=int(word_count),
            keywords=kw_list,
        )
        return result.content, f"Generated {len(result.content.split())} words"
    
    async def generate_technical(self, topic, depth):
        """Generate technical content"""
        agent = self.agents["technical"]
        result = await agent.generate_research_summary(
            topic=topic,
            depth=TechnicalDepth(depth.lower()),
        )
        return result.content, f"Generated {len(result.content.split())} words"
    
    async def generate_docs(self, project_name, doc_type, description):
        """Generate documentation"""
        agent = self.agents["docs"]
        if doc_type == "readme":
            result = await agent.generate_readme(
                project_name=project_name,
                description=description,
            )
        else:
            result = await agent.generate_user_guide(
                title=project_name,
                purpose=description,
            )
        return result.content, f"Generated {len(result.content.split())} words"
    
    async def optimize_content(self, content, target):
        """Optimize existing content"""
        result = await self.optimizer.optimize(
            content=content,
            target=OptimizationTarget(target.lower()),
        )
        return result, "Content optimized successfully"
    
    async def analyze_seo(self, content, keyword):
        """Analyze SEO"""
        analysis = await self.seo_enhancer.analyze(content, keyword)
        return f"""SEO Analysis Results:
        
Overall Score: {analysis.score}/100
Readability: {analysis.readability_score}/100
Keyword: {analysis.keyword_score}/100
Structure: {analysis.structure_score}/100

Suggestions:
{chr(10).join(f"- {s}" for s in analysis.suggestions)}

Missing Elements:
{chr(10).join(f"- {m}" for m in analysis.missing_elements)}"""
    
    def export_content(self, content, formats):
        """Export content"""
        results = self.exporter.create_content_package(
            content=content,
            filename="exported_content",
            include_formats=formats,
        )
        return "\n".join(f"{fmt}: {path}" for fmt, path in results.items())


def create_ui() -> gr.Blocks:
    """Create the Gradio UI"""
    ui = WritingAgentUI()
    available_models = ui.get_available_models()
    
    with gr.Blocks(
        title="AI Writing Agent",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🤖 AI Writing Agent
        
        Your personal AI-powered content creation assistant. Generate high-quality content for:
        - **Social Media** - Engaging posts for Twitter, LinkedIn, Instagram
        - **Email** - Professional cold emails, newsletters, follow-ups
        - **LinkedIn** - Thought leadership posts and articles
        - **Blog** - Full articles optimized for SEO
        - **Technical** - Research summaries and technical documentation
        - **Documentation** - README files, guides, and API docs
        """)
        
        with gr.Tabs():
            with gr.TabItem("📱 Social Media"):
                with gr.Row():
                    with gr.Column():
                        social_topic = gr.Textbox(label="Topic", placeholder="What's your post about?")
                        social_platform = gr.Dropdown(
                            ["Twitter", "Instagram", "LinkedIn", "Facebook"],
                            label="Platform",
                            value="Twitter"
                        )
                        social_tone = gr.Dropdown(
                            ["Friendly", "Professional", "Humorous", "Casual"],
                            label="Tone",
                            value="Friendly"
                        )
                        social_hashtags = gr.Checkbox(label="Include Hashtags", value=True)
                        social_generate = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        social_output = gr.Textbox(label="Generated Content", lines=10)
                        social_stats = gr.Textbox(label="Stats")
                
                social_generate.click(
                    ui.generate_social,
                    inputs=[social_topic, social_platform, social_tone, social_hashtags],
                    outputs=[social_output, social_stats],
                )
            
            with gr.TabItem("📧 Email"):
                with gr.Row():
                    with gr.Column():
                        email_topic = gr.Textbox(label="Email Purpose/Topic", placeholder="What do you want to communicate?")
                        email_type = gr.Dropdown(
                            ["Cold Outreach", "Follow Up", "Newsletter", "Meeting Request"],
                            label="Email Type",
                            value="Cold Outreach"
                        )
                        email_recipient = gr.Textbox(label="Recipient (Optional)", placeholder="John Smith, CTO at TechCorp")
                        email_goal = gr.Textbox(label="Goal (Optional)", placeholder="Schedule a demo call")
                        email_generate = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        email_output = gr.Textbox(label="Generated Email", lines=15)
                        email_subject = gr.Textbox(label="Subject Line")
                
                email_generate.click(
                    ui.generate_email,
                    inputs=[email_topic, email_type, email_recipient, email_goal],
                    outputs=[email_output, email_subject],
                )
            
            with gr.TabItem("💼 LinkedIn"):
                with gr.Row():
                    with gr.Column():
                        linkedin_topic = gr.Textbox(label="Topic", placeholder="What do you want to share?")
                        linkedin_style = gr.Dropdown(
                            ["Thought Leadership", "Personal Story", "Industry Insight", "How To", "Listicle"],
                            label="Post Style",
                            value="Thought Leadership"
                        )
                        linkedin_tone = gr.Dropdown(
                            ["Professional", "Friendly", "Casual"],
                            label="Tone",
                            value="Professional"
                        )
                        linkedin_generate = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        linkedin_output = gr.Textbox(label="Generated Post", lines=12)
                        linkedin_stats = gr.Textbox(label="Stats")
                
                linkedin_generate.click(
                    ui.generate_linkedin,
                    inputs=[linkedin_topic, linkedin_style, linkedin_tone],
                    outputs=[linkedin_output, linkedin_stats],
                )
            
            with gr.TabItem("📝 Blog"):
                with gr.Row():
                    with gr.Column():
                        blog_topic = gr.Textbox(label="Article Topic", placeholder="What should the article be about?")
                        blog_type = gr.Dropdown(
                            ["How To", "Listicle", "Tutorial", "Opinion", "Case Study"],
                            label="Article Type",
                            value="How To"
                        )
                        blog_tone = gr.Dropdown(
                            ["Casual", "Professional", "Technical"],
                            label="Tone",
                            value="Casual"
                        )
                        blog_word_count = gr.Slider(500, 3000, value=1500, step=100, label="Target Word Count")
                        blog_keywords = gr.Textbox(label="Keywords (comma-separated)", placeholder="AI, writing, automation")
                        blog_generate = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        blog_output = gr.Textbox(label="Generated Article", lines=20)
                        blog_stats = gr.Textbox(label="Stats")
                
                blog_generate.click(
                    ui.generate_blog,
                    inputs=[blog_topic, blog_type, blog_tone, blog_word_count, blog_keywords],
                    outputs=[blog_output, blog_stats],
                )
            
            with gr.TabItem("🔬 Technical"):
                with gr.Row():
                    with gr.Column():
                        tech_topic = gr.Textbox(label="Topic", placeholder="What technical topic?")
                        tech_depth = gr.Dropdown(
                            ["Surface", "Intermediate", "Deep", "Expert"],
                            label="Depth",
                            value="Intermediate"
                        )
                        tech_generate = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        tech_output = gr.Textbox(label="Generated Content", lines=20)
                        tech_stats = gr.Textbox(label="Stats")
                
                tech_generate.click(
                    ui.generate_technical,
                    inputs=[tech_topic, tech_depth],
                    outputs=[tech_output, tech_stats],
                )
            
            with gr.TabItem("📚 Documentation"):
                with gr.Row():
                    with gr.Column():
                        docs_name = gr.Textbox(label="Project Name", placeholder="My Awesome Project")
                        docs_type = gr.Dropdown(
                            ["README", "User Guide", "API Docs"],
                            label="Document Type",
                            value="README"
                        )
                        docs_description = gr.Textbox(label="Description", placeholder="What does this project do?")
                        docs_generate = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        docs_output = gr.Textbox(label="Generated Documentation", lines=20)
                        docs_stats = gr.Textbox(label="Stats")
                
                docs_generate.click(
                    ui.generate_docs,
                    inputs=[docs_name, docs_type, docs_description],
                    outputs=[docs_output, docs_stats],
                )
            
            with gr.TabItem("✨ Optimize"):
                with gr.Row():
                    with gr.Column():
                        opt_content = gr.Textbox(label="Content to Optimize", lines=10, placeholder="Paste your content here...")
                        opt_target = gr.Dropdown(
                            ["Readability", "Engagement", "SEO", "Professional", "Clarity"],
                            label="Optimization Target",
                            value="Readability"
                        )
                        opt_button = gr.Button("Optimize", variant="primary")
                    with gr.Column():
                        opt_output = gr.Textbox(label="Optimized Content", lines=10)
                        opt_status = gr.Textbox(label="Status")
                
                opt_button.click(
                    ui.optimize_content,
                    inputs=[opt_content, opt_target],
                    outputs=[opt_output, opt_status],
                )
            
            with gr.TabItem("🔍 SEO Analysis"):
                with gr.Row():
                    with gr.Column():
                        seo_content = gr.Textbox(label="Content to Analyze", lines=10)
                        seo_keyword = gr.Textbox(label="Target Keyword")
                        seo_button = gr.Button("Analyze", variant="primary")
                    with gr.Column():
                        seo_output = gr.Textbox(label="Analysis Results", lines=15)
                
                seo_button.click(
                    ui.analyze_seo,
                    inputs=[seo_content, seo_keyword],
                    outputs=[seo_output],
                )
        
        gr.Markdown("""
        ---
        
        **Model Selection** (configure in `.env` file)
        
        Available models: GPT-4o, Claude 3.5, DeepSeek, Ollama (local)
        
        Built with ❤️ using LangChain, Gradio, and LangGraph
        """)
    
    return demo


def main():
    """Launch the Gradio UI"""
    port = int(os.getenv("GRADIO_PORT", 7860))
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()

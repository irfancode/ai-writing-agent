"""CLI Interface for AI Writing Agent"""

import asyncio
import os
import sys
from typing import Optional, List
from pathlib import Path
import argparse
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .core.orchestrator import WritingAgentOrchestrator, ContentType, Tone
from .core.llm_manager import LLMManager
from .agents.social_media import SocialMediaAgent, Platform, PostType
from .agents.email import EmailAgent, EmailType
from .agents.linkedin import LinkedInAgent, LinkedInPostStyle
from .agents.blog import BlogAgent, BlogType
from .agents.technical import TechnicalAgent, TechnicalDepth
from .agents.documentation import DocumentationAgent, DocType
from .utils.optimizer import ContentOptimizer, OptimizationTarget
from .utils.seo import SEOEnhancer
from .utils.exporter import ContentExporter, ExportConfig


console = Console() if HAS_RICH else None


@dataclass
class CLIConfig:
    primary_model: str = "claude-3-5-sonnet"
    secondary_model: str = "gpt-4o"
    default_output_dir: str = "./output"
    verbose: bool = False


class WritingAgentCLI:
    """Command-line interface for AI Writing Agent"""
    
    def __init__(self, config: Optional[CLIConfig] = None):
        self.config = config or CLIConfig()
        self.orchestrator = WritingAgentOrchestrator(
            primary_model=self.config.primary_model,
            secondary_model=self.config.secondary_model,
        )
        self.llm_manager = LLMManager()
        self.optimizer = ContentOptimizer(llm_manager=self.llm_manager)
        self.seo_enhancer = SEOEnhancer(llm_manager=self.llm_manager)
        self.exporter = ContentExporter(ExportConfig(output_dir=self.config.default_output_dir))
        
        self._register_agents()
    
    def _register_agents(self):
        """Register all specialized agents"""
        self.orchestrator.register_agent(ContentType.SOCIAL_MEDIA, SocialMediaAgent())
        self.orchestrator.register_agent(ContentType.EMAIL, EmailAgent())
        self.orchestrator.register_agent(ContentType.LINKEDIN, LinkedInAgent())
        self.orchestrator.register_agent(ContentType.BLOG, BlogAgent())
        self.orchestrator.register_agent(ContentType.TECHNICAL, TechnicalAgent())
        self.orchestrator.register_agent(ContentType.DOCUMENTATION, DocumentationAgent())
    
    def print_banner(self):
        """Print the application banner"""
        banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🤖  AI Writing Agent                                  ║
║     Your Personal Content Creation Assistant              ║
║                                                           ║
║     ✍️  Social Media  📧  Email  💼  LinkedIn              ║
║     📝  Blog  🔬  Technical  📚  Documentation            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
        """
        if console:
            console.print(banner, style="bold cyan")
        else:
            print("AI Writing Agent - Your Personal Content Creation Assistant")
    
    def print_available_models(self):
        """Print available LLM models"""
        models = self.llm_manager.list_available_models()
        
        if console:
            table = Table(title="Available Models")
            table.add_column("Model", style="cyan")
            table.add_column("Provider", style="green")
            
            for model in models:
                provider = "OpenAI" if "gpt" in model else "Anthropic" if "claude" in model else "Ollama"
                table.add_row(model, provider)
            
            console.print(table)
        else:
            print(f"Available models: {', '.join(models)}")
    
    async def generate_social_media(
        self,
        topic: str,
        platform: str = "twitter",
        tone: str = "friendly",
        num_posts: int = 1,
        include_hashtags: bool = True,
        **kwargs
    ):
        """Generate social media content"""
        agent = SocialMediaAgent(llm_manager=self.llm_manager)
        
        platform_enum = Platform(platform.lower())
        
        if num_posts > 1:
            result = await agent.generate_thread(
                topic=topic,
                platform=platform_enum,
                num_tweets=num_posts,
            )
        else:
            result = await agent.generate_post(
                topic=topic,
                platform=platform_enum,
                tone=WritingTone[tone.upper()],
                include_hashtags=include_hashtags,
            )
        
        return result
    
    async def generate_email(
        self,
        topic: str,
        email_type: str = "cold_outreach",
        recipient: Optional[str] = None,
        tone: str = "professional",
        goal: Optional[str] = None,
        **kwargs
    ):
        """Generate email content"""
        agent = EmailAgent(llm_manager=self.llm_manager)
        
        result = await agent.generate_email(
            topic=topic,
            email_type=EmailType(email_type),
            recipient_name=recipient,
            tone=WritingTone[tone.upper()],
            goal=goal,
        )
        
        return result
    
    async def generate_linkedin(
        self,
        topic: str,
        style: str = "thought_leadership",
        tone: str = "professional",
        **kwargs
    ):
        """Generate LinkedIn content"""
        agent = LinkedInAgent(llm_manager=self.llm_manager)
        
        result = await agent.generate_post(
            topic=topic,
            style=LinkedInPostStyle(style.lower()),
            tone=WritingTone[tone.upper()],
        )
        
        return result
    
    async def generate_blog(
        self,
        topic: str,
        blog_type: str = "how_to",
        tone: str = "casual",
        word_count: int = 1500,
        keywords: Optional[List[str]] = None,
        **kwargs
    ):
        """Generate blog post"""
        agent = BlogAgent(llm_manager=self.llm_manager)
        
        result = await agent.generate_post(
            topic=topic,
            blog_type=BlogType(blog_type.lower()),
            tone=WritingTone[tone.upper()],
            word_count=word_count,
            keywords=keywords,
        )
        
        return result
    
    async def generate_technical(
        self,
        topic: str,
        depth: str = "intermediate",
        **kwargs
    ):
        """Generate technical content"""
        agent = TechnicalAgent(llm_manager=self.llm_manager)
        
        result = await agent.generate_research_summary(
            topic=topic,
            depth=TechnicalDepth(depth.lower()),
        )
        
        return result
    
    async def generate_documentation(
        self,
        doc_type: str,
        project_name: str,
        description: str = "",
        **kwargs
    ):
        """Generate documentation"""
        agent = DocumentationAgent(llm_manager=self.llm_manager)
        
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
        
        return result
    
    async def optimize_content(
        self,
        content: str,
        target: str = "readability",
        **kwargs
    ):
        """Optimize existing content"""
        result = await self.optimizer.optimize(
            content=content,
            target=OptimizationTarget(target.lower()),
        )
        return result
    
    async def export_content(
        self,
        content: str,
        filename: str,
        formats: List[str] = None,
        **kwargs
    ):
        """Export content to various formats"""
        formats = formats or ["md"]
        results = self.exporter.create_content_package(
            content=content,
            filename=filename,
        )
        return results


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="AI Writing Agent - Your Personal Content Creation Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    generate_parser = subparsers.add_parser("generate", help="Generate content")
    generate_parser.add_argument(
        "content_type",
        choices=["social", "email", "linkedin", "blog", "technical", "docs"],
        help="Type of content to generate",
    )
    generate_parser.add_argument("--topic", "-t", required=True, help="Topic or subject")
    generate_parser.add_argument("--platform", "-p", default="twitter", help="Platform (for social)")
    generate_parser.add_argument("--tone", default="friendly", help="Tone of voice")
    generate_parser.add_argument("--output", "-o", help="Output file")
    generate_parser.add_argument("--count", "-c", type=int, default=1, help="Number of variants")
    
    optimize_parser = subparsers.add_parser("optimize", help="Optimize content")
    optimize_parser.add_argument("--input", "-i", required=True, help="Input file")
    optimize_parser.add_argument("--target", "-t", default="readability", help="Optimization target")
    optimize_parser.add_argument("--output", "-o", help="Output file")
    
    export_parser = subparsers.add_parser("export", help="Export content")
    export_parser.add_argument("--input", "-i", required=True, help="Input file")
    export_parser.add_argument("--formats", "-f", nargs="+", default=["md"], help="Export formats")
    export_parser.add_argument("--output-dir", "-o", default="./output", help="Output directory")
    
    models_parser = subparsers.add_parser("models", help="List available models")
    
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive mode")
    
    test_parser = subparsers.add_parser("test", help="Test the agent")
    test_parser.add_argument("--agent", "-a", help="Specific agent to test")
    
    return parser


async def run_interactive(cli: WritingAgentCLI):
    """Run interactive mode"""
    cli.print_banner()
    
    if console:
        console.print("\n[bold green]Welcome to Interactive Mode![/bold green]\n")
    else:
        print("\nWelcome to Interactive Mode!\n")
    
    while True:
        if console:
            choice = console.input("\nSelect content type:\n1. Social Media\n2. Email\n3. LinkedIn\n4. Blog\n5. Technical\n6. Documentation\n7. Optimize Content\n8. Exit\n\nChoice (1-8): ")
        else:
            choice = input("\nSelect content type (1-8): ")
        
        if choice == "8":
            break
        
        if console:
            topic = Prompt.ask("\n[bold cyan]What's the topic?[/bold cyan]")
        else:
            topic = input("\nWhat's the topic? ")
        
        try:
            if choice == "1":
                result = await cli.generate_social_media(topic)
            elif choice == "2":
                result = await cli.generate_email(topic)
            elif choice == "3":
                result = await cli.generate_linkedin(topic)
            elif choice == "4":
                result = await cli.generate_blog(topic)
            elif choice == "5":
                result = await cli.generate_technical(topic)
            elif choice == "6":
                result = await cli.generate_documentation("readme", topic, topic)
            elif choice == "7":
                result = await cli.optimize_content(topic)
            else:
                continue
            
            if console:
                console.print("\n[bold green]Generated Content:[/bold green]\n")
                console.print(Panel(result.content, title="Result", border_style="cyan"))
            else:
                print("\n--- Generated Content ---\n")
                print(result.content)
        
        except Exception as e:
            if console:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
            else:
                print(f"Error: {str(e)}")


async def run_generate(cli: WritingAgentCLI, args):
    """Run content generation"""
    content_type_map = {
        "social": ("social_media", cli.generate_social_media, {"platform": args.platform}),
        "email": ("email", cli.generate_email, {}),
        "linkedin": ("linkedin", cli.generate_linkedin, {}),
        "blog": ("blog", cli.generate_blog, {}),
        "technical": ("technical", cli.generate_technical, {}),
        "docs": ("documentation", cli.generate_documentation, {"doc_type": "readme"}),
    }
    
    name, func, extra_args = content_type_map[args.content_type]
    
    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Generating {name} content...", total=None)
            result = await func(args.topic, **extra_args)
    else:
        print(f"Generating {name} content...")
        result = await func(args.topic, **extra_args)
    
    if console:
        console.print(Panel(result.content, title=f"{name.title()} Content", border_style="cyan"))
    else:
        print(result.content)
    
    if args.output:
        cli.exporter.export_markdown(result.content, args.output)
        if console:
            console.print(f"[green]Saved to {args.output}.md[/green]")


async def run_optimize(cli: WritingAgentCLI, args):
    """Run content optimization"""
    with open(args.input, "r") as f:
        content = f.read()
    
    if console:
        with Progress(SpinnerColumn(), TextColumn("Optimizing..."), console=console) as progress:
            progress.add_task("task", total=None)
            result = await cli.optimize_content(content, args.target)
    else:
        print("Optimizing content...")
        result = await cli.optimize_content(content, args.target)
    
    if console:
        console.print(Panel(result, title="Optimized Content", border_style="green"))
    else:
        print(result)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(result)


async def run_test(cli: WritingAgentCLI, agent_name: Optional[str] = None):
    """Test the agent(s)"""
    if console:
        console.print("\n[bold]Testing AI Writing Agent...[/bold]\n")
    else:
        print("\nTesting AI Writing Agent...\n")
    
    tests = [
        ("Social Media", cli.generate_social_media("The future of AI in content creation")),
        ("Email", cli.generate_email("Introduction to our new product", email_type="cold_outreach")),
        ("LinkedIn", cli.generate_linkedin("Building a personal brand on LinkedIn")),
        ("Blog", cli.generate_blog("How to write better prompts for AI")),
        ("Technical", cli.generate_technical("Introduction to LangGraph")),
    ]
    
    passed = 0
    failed = 0
    
    for name, coro in tests:
        if agent_name and agent_name.lower() not in name.lower():
            continue
        
        try:
            if console:
                console.print(f"[cyan]Testing {name}...[/cyan]")
            else:
                print(f"Testing {name}...")
            
            result = await coro
            if result.content:
                passed += 1
                if console:
                    console.print(f"[green]✓ {name} passed[/green]")
                else:
                    print(f"✓ {name} passed")
            else:
                failed += 1
                if console:
                    console.print(f"[red]✗ {name} failed[/red]")
                else:
                    print(f"✗ {name} failed")
        except Exception as e:
            failed += 1
            if console:
                console.print(f"[red]✗ {name} failed: {str(e)}[/red]")
            else:
                print(f"✗ {name} failed: {str(e)}")
    
    if console:
        console.print(f"\n[bold]Results: {passed} passed, {failed} failed[/bold]")
    else:
        print(f"\nResults: {passed} passed, {failed} failed")
    
    return failed == 0


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = WritingAgentCLI()
    
    if args.command == "models":
        cli.print_available_models()
        return
    
    if args.command == "interactive":
        asyncio.run(run_interactive(cli))
        return
    
    if args.command == "test":
        success = asyncio.run(run_test(cli, args.agent))
        sys.exit(0 if success else 1)
    
    if args.command == "generate":
        asyncio.run(run_generate(cli, args))
        return
    
    if args.command == "optimize":
        asyncio.run(run_optimize(cli, args))
        return


if __name__ == "__main__":
    main()

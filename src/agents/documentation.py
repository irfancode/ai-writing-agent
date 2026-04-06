"""Documentation Agent - Generates technical documentation"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import BaseWritingAgent, WritingRequirements, AgentResponse, WritingTone, ContentLength


class DocType(Enum):
    README = "readme"
    API = "api"
    GUIDE = "guide"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    TROUBLESHOOTING = "troubleshooting"
    CHANGELOG = "changelog"
    CONTRIBUTING = "contributing"


class DocStyle(Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    DETAILED = "detailed"
    REFERENCE = "reference"


class DocumentationAgent(BaseWritingAgent):
    """Agent specialized in technical documentation"""
    
    name = "documentation_agent"
    description = "Generates comprehensive technical documentation"
    
    def get_system_prompt(self) -> str:
        return """You are an expert technical documentation writer with deep expertise in creating clear, comprehensive documentation.

Documentation principles:
1. Clarity - Every sentence should be easily understood
2. Completeness - Cover all use cases and edge cases
3. Consistency - Use the same terminology throughout
4. Scannability - Headers, lists, and code blocks for easy scanning
5. Accuracy - Every example should be correct and runnable
6. Maintainability - Write for future editors

Documentation should:
- Use active voice
- Address the reader directly ("you")
- Provide concrete examples
- Anticipate questions
- Link related topics
- Include warnings and caveats"""
    
    def get_content_prompt(self, requirements: WritingRequirements) -> str:
        doc_type = requirements.additional_context or DocType.README.value
        
        prompt_parts = [
            f"Document Type: {doc_type.replace('_', ' ').title()}",
            f"Topic/Subject: {requirements.topic}",
        ]
        
        if requirements.tone:
            prompt_parts.append(f"Style: {requirements.tone.value}")
        
        if requirements.audience:
            prompt_parts.append(f"Target Users: {requirements.audience}")
        
        if requirements.goal:
            prompt_parts.append(f"Goal: {requirements.goal}")
        
        return f"""Create documentation.

Requirements:
{chr(10).join(prompt_parts)}

Focus on clarity, completeness, and usability."""
    
    async def generate_readme(
        self,
        project_name: str,
        description: str,
        features: Optional[List[str]] = None,
        installation: Optional[str] = None,
        usage: Optional[str] = None,
        tech_stack: Optional[List[str]] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a comprehensive README file"""
        import time
        start_time = time.time()
        
        prompt = f"""Create a professional README.md for: {project_name}

Description: {description}

"""
        
        if features:
            prompt += "Features:\n"
            for f in features:
                prompt += f"- {f}\n"
            prompt += "\n"
        
        if tech_stack:
            prompt += f"Tech Stack: {', '.join(tech_stack)}\n\n"
        
        if installation:
            prompt += f"Installation:\n{installation}\n\n"
        
        if usage:
            prompt += f"Usage:\n{usage}\n\n"
        
        prompt += """
Include these standard sections:
- Badges (build, version, license)
- Description
- Features
- Installation
- Usage
- Configuration
- Contributing
- License
- Support/Contact

Make it comprehensive and professional. Use Markdown formatting."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=4000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "doc_type": "readme",
                    "project_name": project_name,
                },
                tokens_used=response.usage.get("total_tokens", 0),
                generation_time=time.time() - start_time,
            )
        
        except Exception as e:
            return AgentResponse(
                content="",
                agent_name=self.name,
                warnings=[str(e)],
                generation_time=time.time() - start_time,
            )
    
    async def generate_api_docs(
        self,
        api_name: str,
        endpoints: Optional[List[Dict]] = None,
        description: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate API documentation"""
        import time
        start_time = time.time()
        
        prompt = f"""Create comprehensive API documentation for: {api_name}

"""
        
        if description:
            prompt += f"Description: {description}\n\n"
        
        if base_url:
            prompt += f"Base URL: {base_url}\n\n"
        
        if endpoints:
            prompt += "Endpoints to document:\n"
            for ep in endpoints:
                prompt += f"- {ep.get('method', 'GET')} {ep.get('path', '/')}: {ep.get('description', '')}\n"
            prompt += "\n"
        
        prompt += """
Include:
- Overview and introduction
- Authentication methods
- Rate limiting
- Error codes
- For each endpoint:
  - Method and path
  - Description
  - Path/Query/Body parameters
  - Request examples
  - Response examples
  - Error responses

Use OpenAPI-like format with code examples in multiple languages (Python, JavaScript, cURL)."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=6000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "doc_type": "api",
                    "api_name": api_name,
                    "endpoint_count": len(endpoints) if endpoints else 0,
                },
                tokens_used=response.usage.get("total_tokens", 0),
                generation_time=time.time() - start_time,
            )
        
        except Exception as e:
            return AgentResponse(
                content="",
                agent_name=self.name,
                warnings=[str(e)],
                generation_time=time.time() - start_time,
            )
    
    async def generate_user_guide(
        self,
        title: str,
        purpose: str,
        sections: Optional[List[str]] = None,
        audience: str = "end users",
        **kwargs
    ) -> AgentResponse:
        """Generate a user guide"""
        import time
        start_time = time.time()
        
        sections = sections or [
            "Getting Started",
            "Core Features",
            "Configuration",
            "Advanced Usage",
            "Tips and Tricks",
            "FAQ",
        ]
        
        prompt = f"""Create a comprehensive user guide.

Title: {title}
Purpose: {purpose}
Audience: {audience}

Sections to cover:
{chr(10).join(f'- {s}' for s in sections)}

For each section:
- Clear headings
- Step-by-step instructions
- Screenshots/placeholders where appropriate
- Examples
- Troubleshooting tips

Make it user-friendly and accessible."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=5000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "doc_type": "guide",
                    "title": title,
                    "sections": sections,
                },
                tokens_used=response.usage.get("total_tokens", 0),
                generation_time=time.time() - start_time,
            )
        
        except Exception as e:
            return AgentResponse(
                content="",
                agent_name=self.name,
                warnings=[str(e)],
                generation_time=time.time() - start_time,
            )
    
    async def generate_troubleshooting_guide(
        self,
        product_name: str,
        common_issues: Optional[List[str]] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a troubleshooting guide"""
        import time
        start_time = time.time()
        
        prompt = f"""Create a troubleshooting guide for: {product_name}

"""
        
        if common_issues:
            prompt += "Common issues to address:\n"
            for issue in common_issues:
                prompt += f"- {issue}\n"
            prompt += "\n"
        
        prompt += """
For each issue include:
- Problem description
- Possible causes
- Step-by-step solutions
- Prevention tips
- When to contact support

Use clear headings and formatting for easy scanning."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=4000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "doc_type": "troubleshooting",
                    "product": product_name,
                },
                tokens_used=response.usage.get("total_tokens", 0),
                generation_time=time.time() - start_time,
            )
        
        except Exception as e:
            return AgentResponse(
                content="",
                agent_name=self.name,
                warnings=[str(e)],
                generation_time=time.time() - start_time,
            )
    
    async def generate_changelog(
        self,
        project_name: str,
        version: str,
        changes: Optional[List[Dict]] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a changelog entry"""
        import time
        start_time = time.time()
        
        prompt = f"""Generate a changelog entry for {project_name} version {version}.

"""
        
        if changes:
            prompt += "Changes:\n"
            for change in changes:
                change_type = change.get("type", "Changed")
                description = change.get("description", "")
                prompt += f"- **{change_type}**: {description}\n"
            prompt += "\n"
        
        prompt += """
Use semantic versioning categories:
- Added (new features)
- Changed (changes in existing functionality)
- Deprecated (soon-to-be removed features)
- Removed (removed features)
- Fixed (bug fixes)
- Security (vulnerability fixes)

Follow Keep a Changelog format."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=2000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "doc_type": "changelog",
                    "project": project_name,
                    "version": version,
                },
                tokens_used=response.usage.get("total_tokens", 0),
                generation_time=time.time() - start_time,
            )
        
        except Exception as e:
            return AgentResponse(
                content="",
                agent_name=self.name,
                warnings=[str(e)],
                generation_time=time.time() - start_time,
            )
    
    async def generate_contributing_guide(
        self,
        project_name: str,
        repo_url: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a CONTRIBUTING guide"""
        import time
        start_time = time.time()
        
        prompt = f"""Create a comprehensive CONTRIBUTING.md for: {project_name}

"""
        
        if repo_url:
            prompt += f"Repository: {repo_url}\n\n"
        
        prompt += """
Include:
- Welcome message
- Code of Conduct
- Getting Started
  - Prerequisites
  - Fork and Clone
  - Development Setup
- Development Workflow
  - Branch naming
  - Commit messages
  - Pull Request process
- Coding Standards
  - Style guides
  - Testing requirements
  - Documentation
- Reporting Issues
- Suggesting Features
- Review Process
- Community Guidelines

Make it welcoming to new contributors."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=4000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "doc_type": "contributing",
                    "project": project_name,
                },
                tokens_used=response.usage.get("total_tokens", 0),
                generation_time=time.time() - start_time,
            )
        
        except Exception as e:
            return AgentResponse(
                content="",
                agent_name=self.name,
                warnings=[str(e)],
                generation_time=time.time() - start_time,
            )
    
    async def analyze_codebase(
        self,
        file_path: str,
        code_content: str,
        **kwargs
    ) -> AgentResponse:
        """Analyze code and generate documentation"""
        import time
        start_time = time.time()
        
        prompt = f"""Analyze this code and generate documentation:

File: {file_path}

```{code_content[:500]}...``` (truncated)

Provide:
1. File summary
2. Key components (classes, functions)
3. Dependencies
4. Usage examples
5. Configuration options

Generate documentation in Markdown format."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=3000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "doc_type": "code_analysis",
                    "file": file_path,
                    "lines": code_content.count("\n") + 1,
                },
                tokens_used=response.usage.get("total_tokens", 0),
                generation_time=time.time() - start_time,
            )
        
        except Exception as e:
            return AgentResponse(
                content="",
                agent_name=self.name,
                warnings=[str(e)],
                generation_time=time.time() - start_time,
            )

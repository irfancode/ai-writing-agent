"""Technical Agent - Generates technical research and analysis content"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import BaseWritingAgent, WritingRequirements, AgentResponse, WritingTone, ContentLength


class TechnicalDocumentType(Enum):
    RESEARCH_PAPER = "research_paper"
    TECHNICAL_BLOG = "technical_blog"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    TUTORIAL = "tutorial"
    BENCHMARK = "benchmark"
    CASE_STUDY = "case_study"
    SURVEY = "survey"
    METHODOLOGY = "methodology"


class TechnicalDepth(Enum):
    SURFACE = "surface"
    INTERMEDIATE = "intermediate"
    DEEP = "deep"
    EXPERT = "expert"


class TechnicalAgent(BaseWritingAgent):
    """Agent specialized in technical writing and research"""
    
    name = "technical_agent"
    description = "Generates technical research, analysis, and documentation"
    
    DOCUMENT_PROMPTS = {
        TechnicalDocumentType.RESEARCH_PAPER: """You are writing a research paper. Structure includes:
- Abstract (150-300 words)
- Introduction (context, problem, contribution)
- Related Work/Literature Review
- Methodology
- Results/Findings
- Discussion
- Conclusion
- References (placeholder)

Be rigorous, cite concepts (not real papers unless specified), and maintain academic tone.""",
        
        TechnicalDocumentType.TECHNICAL_BLOG: """You are writing a technical blog post. Create content that:
- Explains complex concepts in accessible language
- Includes code examples where relevant
- Uses analogies and visual descriptions
- Balances depth with readability
- Provides practical takeaways""",
        
        TechnicalDocumentType.ANALYSIS: """You are writing a technical analysis. Structure includes:
- Executive Summary
- Background/Context
- Detailed Analysis
- Key Findings
- Implications
- Recommendations
- Conclusion

Be thorough, objective, and data-driven.""",
        
        TechnicalDocumentType.TUTORIAL: """You are writing a technical tutorial. Include:
- Prerequisites
- Learning Objectives
- Step-by-step instructions
- Code examples
- Expected output
- Troubleshooting
- Further reading

Make it actionable and easy to follow.""",
    }
    
    def get_system_prompt(self) -> str:
        return """You are an expert technical writer with deep expertise in software engineering, AI/ML, and technology topics.

Technical writing principles:
1. Accuracy - Every claim must be verifiable or clearly marked as opinion
2. Clarity - Use simple language for complex concepts
3. Structure - Clear hierarchy with headers and sections
4. Code - Include runnable examples when applicable
5. Citations - Reference authoritative sources
6. Balance - Mix depth with accessibility

Technical content should:
- Define all technical terms
- Provide context and background
- Show concrete examples
- Include pros/cons where relevant
- Be honest about limitations
- Offer actionable insights"""
    
    def get_content_prompt(self, requirements: WritingRequirements) -> str:
        doc_type = requirements.additional_context or TechnicalDocumentType.TECHNICAL_BLOG.value
        
        prompt_parts = [
            f"Document Type: {doc_type.replace('_', ' ').title()}",
            f"Topic: {requirements.topic}",
        ]
        
        if requirements.tone:
            prompt_parts.append(f"Tone: {requirements.tone.value}")
        
        if requirements.audience:
            prompt_parts.append(f"Target Audience: {requirements.audience}")
        
        if requirements.keywords:
            prompt_parts.append(f"Key Concepts: {', '.join(requirements.keywords)}")
        
        if requirements.goal:
            prompt_parts.append(f"Goal: {requirements.goal}")
        
        return f"""Create technical content.

Requirements:
{chr(10).join(prompt_parts)}

Focus on accuracy, depth, and practical value."""
    
    async def generate_research_summary(
        self,
        topic: str,
        depth: TechnicalDepth = TechnicalDepth.INTERMEDIATE,
        audience: Optional[str] = None,
        key_questions: Optional[List[str]] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a comprehensive research summary"""
        import time
        start_time = time.time()
        
        depth_descriptions = {
            TechnicalDepth.SURFACE: "High-level overview suitable for beginners",
            TechnicalDepth.INTERMEDIATE: "Detailed explanation for practitioners",
            TechnicalDepth.DEEP: "In-depth technical analysis for experts",
            TechnicalDepth.EXPERT: "Research-grade content with citations and nuances",
        }
        
        prompt = f"""Write a comprehensive research summary about: {topic}

Depth Level: {depth_descriptions.get(depth, depth_descriptions[TechnicalDepth.INTERMEDIATE])}

"""
        
        if audience:
            prompt += f"Target Audience: {audience}\n"
        
        if key_questions:
            prompt += f"\nKey Questions to Address:\n"
            for q in key_questions:
                prompt += f"- {q}\n"
        
        prompt += """

Structure your response as:

# [Topic] - Research Summary

## Executive Summary
[2-3 sentence overview]

## Background
[Context and history]

## Key Concepts
[Main technical concepts explained]

## Current State
[Where the field/technology stands today]

## Key Findings
[Main discoveries or insights]

## Implications
[What this means for practitioners/researchers]

## Limitations & Future Work
[Acknowledged gaps and future directions]

## References
[Placeholder references to relevant concepts]

Target length: 1500-2500 words
"""
        
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
                    "document_type": "research_summary",
                    "depth": depth.value,
                    "word_count": len(response.content.split()),
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
    
    async def generate_comparison(
        self,
        topic_a: str,
        topic_b: str,
        comparison_criteria: Optional[List[str]] = None,
        depth: TechnicalDepth = TechnicalDepth.INTERMEDIATE,
        **kwargs
    ) -> AgentResponse:
        """Generate a technical comparison"""
        import time
        start_time = time.time()
        
        criteria = comparison_criteria or ["Architecture", "Performance", "Ease of Use", "Community", "Use Cases"]
        
        prompt = f"""Write a comprehensive technical comparison between {topic_a} and {topic_b}.

Comparison Criteria:
{chr(10).join(f'- {c}' for c in criteria)}

Structure:
# {topic_a} vs {topic_b} - Technical Comparison

## Executive Summary
[Quick verdict and recommendation]

## Overview
[Brief intro to both options]

## {topic_a}
[Key features, strengths, weaknesses]

## {topic_b}
[Key features, strengths, weaknesses]

## Head-to-Head Comparison
[Comparison table]
| Criteria | {topic_a} | {topic_b} |
|----------|----------|----------|
"""
        
        for c in criteria:
            prompt += f"| {c} | Rating/Description | Rating/Description |\n"
        
        prompt += f"""
## Use Case Analysis
[When to choose each option]

## Conclusion & Recommendation
[Summary and guidance]

"""
        
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
                    "document_type": "comparison",
                    "topic_a": topic_a,
                    "topic_b": topic_b,
                    "criteria": criteria,
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
    
    async def generate_code_analysis(
        self,
        code: str,
        language: str = "python",
        analysis_type: str = "review",
        **kwargs
    ) -> AgentResponse:
        """Analyze code and provide insights"""
        import time
        start_time = time.time()
        
        prompt = f"""Analyze this {language} code:

```{language}
{code}
```

Analysis Type: {analysis_type}

Provide:
1. What the code does (summary)
2. How it works (breakdown)
3. Potential improvements
4. Best practices followed
5. Issues or bugs (if any)
6. Security considerations (if applicable)

Be specific and constructive in your feedback."""
        
        messages = [
            SystemMessage(content="You are an expert software engineer conducting a code review."),
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
                    "analysis_type": analysis_type,
                    "language": language,
                    "code_length": len(code),
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
    
    async def generate_benchmark_report(
        self,
        tool_or_system: str,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a benchmark report"""
        import time
        start_time = time.time()
        
        metrics = metrics or ["Speed", "Accuracy", "Resource Usage", "Scalability", "Cost"]
        
        prompt = f"""Create a benchmark framework for evaluating: {tool_or_system}

Metrics to benchmark:
{chr(10).join(f'- {m}' for m in metrics)}

Include:
1. Benchmark methodology
2. Test environment/conditions
3. Metric definitions
4. Expected results ranges
5. How to interpret results
6. Limitations of benchmarking

Format as a technical report."""
        
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
                    "document_type": "benchmark",
                    "tool": tool_or_system,
                    "metrics": metrics,
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
    
    def extract_key_concepts(self, content: str) -> List[str]:
        """Extract key technical concepts from content"""
        technical_indicators = [
            "API", "algorithm", "architecture", "async", "authentication",
            "benchmark", "cache", "CI/CD", "cloud", "cluster", "code",
            "compute", "container", "database", "debug", "deploy",
            "distributed", "Docker", "encryption", "endpoint", "framework",
            "function", "Git", "GPU", "HTTP", "inference", "infrastructure",
            "integration", "interface", "JSON", "Kubernetes", "latency",
            "learning", "load", "memory", "metadata", "microservice",
            "model", "monitoring", "network", "node", "optimization",
            "orchestration", "parallel", "pipeline", "protocol", "query",
            "queue", "rate", "REST", "runtime", "scale", "schema",
            "security", "server", "service", "storage", "stream",
            "sync", "system", "test", "throughput", "token", "trace",
            "train", "transform", "trigger", "tuning", "upload",
            "validation", "vector", "virtual", "workflow", "YAML",
        ]
        
        content_lower = content.lower()
        found_concepts = set()
        
        for concept in technical_indicators:
            if concept.lower() in content_lower:
                found_concepts.add(concept)
        
        return sorted(list(found_concepts))
    
    def calculate_readability_score(self, content: str) -> Dict[str, Any]:
        """Calculate technical readability metrics"""
        words = content.split()
        sentences = content.replace("!", ".").replace("?", ".").split(".")
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return {"score": 0, "grade": "N/A"}
        
        avg_words_per_sentence = len(words) / len(sentences)
        
        avg_chars_per_word = sum(len(w) for w in words) / len(words) if words else 0
        
        score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * (avg_chars_per_word / 6)
        
        grade = "College Graduate"
        if score >= 80:
            grade = "Middle School"
        elif score >= 60:
            grade = "High School"
        elif score >= 40:
            grade = "College"
        
        return {
            "score": round(score, 1),
            "grade_level": grade,
            "avg_words_per_sentence": round(avg_words_per_sentence, 1),
            "avg_chars_per_word": round(avg_chars_per_word, 1),
        }

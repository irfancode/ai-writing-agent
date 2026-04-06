"""Blog Agent - Generates blog posts and articles"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import BaseWritingAgent, WritingRequirements, AgentResponse, WritingTone, ContentLength


class BlogType(Enum):
    HOW_TO = "how_to"
    LISTICLE = "listicle"
    TUTORIAL = "tutorial"
    OPINION = "opinion"
    NEWS = "news"
    REVIEW = "review"
    COMPARISON = "comparison"
    CASE_STUDY = "case_study"
    INTERVIEW = "interview"
    ROUNDUP = "roundup"


class BlogTone(Enum):
    CASUAL = "casual"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    AUTHORITATIVE = "authoritative"
    FRIENDLY = "friendly"


class BlogAgent(BaseWritingAgent):
    """Agent specialized in blog content generation"""
    
    name = "blog_agent"
    description = "Generates high-quality blog posts and articles"
    
    BLOG_TYPE_STRUCTURES = {
        BlogType.HOW_TO: {
            "name": "How-To Guide",
            "sections": ["Introduction", "Prerequisites", "Step-by-Step Instructions", "Common Mistakes", "Conclusion"],
            "best_for": "Educational content, teaching processes",
        },
        BlogType.LISTICLE: {
            "name": "Listicle",
            "sections": ["Introduction", "Item 1", "Item 2", "...", "Item N", "Conclusion"],
            "best_for": "Shareable, scannable content, social traffic",
        },
        BlogType.TUTORIAL: {
            "name": "Tutorial",
            "sections": ["Introduction", "Overview", "Setup", "Core Concepts", "Exercise", "Solution", "Summary"],
            "best_for": "Learning-oriented, hands-on practice",
        },
        BlogType.OPINION: {
            "name": "Opinion Piece",
            "sections": ["Introduction (Hook)", "Context", "Main Argument", "Counterarguments", "Conclusion"],
            "best_for": "Thought leadership, engaging readers",
        },
        BlogType.CASE_STUDY: {
            "name": "Case Study",
            "sections": ["Introduction", "Challenge", "Approach", "Results", "Key Learnings", "Conclusion"],
            "best_for": "B2B content, demonstrating expertise",
        },
        BlogType.COMPARISON: {
            "name": "Comparison Article",
            "sections": ["Introduction", "Option A", "Option B", "Head-to-Head Comparison", "Conclusion"],
            "best_for": "Decision-making content, SEO value",
        },
    }
    
    def get_system_prompt(self) -> str:
        return """You are an expert blog writer with expertise in creating engaging, SEO-optimized content that ranks well and resonates with readers.

Blog writing best practices:
1. Compelling headline (H1) that includes primary keyword
2. Introductory hook that promises value
3. Well-structured body with H2, H3 headers
4. Short paragraphs (2-3 sentences max)
5. Bullet and numbered lists for scannability
6. Natural keyword integration
7. Internal/external link suggestions
8. Clear conclusion with CTA
9. Meta description

SEO principles:
- Primary keyword in first 100 words
- Include keywords in headers
- Use variations of keywords naturally
- Aim for 1500-2500 words for comprehensive coverage
- Include 1-2 images with alt text

Content should:
- Provide genuine value
- Answer the reader's questions
- Be well-researched
- Be original and not generic
- Engage the reader throughout"""
    
    def get_content_prompt(self, requirements: WritingRequirements) -> str:
        blog_type = requirements.additional_context or BlogType.HOW_TO.value
        blog_type_enum = BlogType(blog_type) if isinstance(blog_type, str) else blog_type
        
        structure = self.BLOG_TYPE_STRUCTURES.get(blog_type_enum, self.BLOG_TYPE_STRUCTURES[BlogType.HOW_TO])
        
        prompt_parts = [
            f"Blog Type: {structure['name']}",
            f"Topic: {requirements.topic}",
        ]
        
        if requirements.tone:
            prompt_parts.append(f"Tone: {requirements.tone.value}")
        
        if requirements.audience:
            prompt_parts.append(f"Target Audience: {requirements.audience}")
        
        if requirements.keywords:
            prompt_parts.append(f"Primary Keywords: {requirements.keywords[0]}")
            if len(requirements.keywords) > 1:
                prompt_parts.append(f"Secondary Keywords: {', '.join(requirements.keywords[1:])}")
        
        if requirements.goal:
            prompt_parts.append(f"Goal: {requirements.goal}")
        
        return f"""Write a blog post using the {structure['name']} format.

Structure: {' → '.join(structure['sections'])}

Requirements:
{chr(10).join(prompt_parts)}

Include:
- Compelling headline (H1)
- Introduction with hook
- Well-structured body sections (H2, H3)
- Bullet points and lists where appropriate
- Conclusion with takeaways
- Suggested meta description"""
    
    async def generate_post(
        self,
        topic: str,
        blog_type: BlogType = BlogType.HOW_TO,
        tone: WritingTone = WritingTone.CASUAL,
        audience: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        word_count: int = 1500,
        goal: Optional[str] = None,
        include_seo: bool = True,
        **kwargs
    ) -> AgentResponse:
        """Generate a complete blog post"""
        import time
        start_time = time.time()
        
        prompt = f"""Write a {word_count}+ word {blog_type.value.replace('_', ' ')} blog post.

Topic: {topic}

"""
        
        if tone:
            prompt += f"Tone: {tone.value}\n"
        
        if audience:
            prompt += f"Target Audience: {audience}\n"
        
        if keywords:
            prompt += f"Primary Keyword: {keywords[0]}\n"
            if len(keywords) > 1:
                prompt += f"Secondary Keywords: {', '.join(keywords[1:])}\n"
        
        if goal:
            prompt += f"Goal: {goal}\n"
        
        prompt += f"""

Format your response as follows:

# [Your Compelling Headline]

[Meta description - 150-160 characters]

## Introduction
[Hook and value proposition]

## [H2 Section Title]
[H2 content]

### [H3 Section Title]
[H3 content if needed]

[Continue with well-structured content...]

## Conclusion
[Key takeaways and CTA]

---
Word count target: {word_count}+ words
"""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=8000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "blog_type": blog_type.value,
                    "word_count": word_count,
                    "has_seo_elements": include_seo,
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
    
    async def generate_headline_variants(
        self,
        topic: str,
        num_variants: int = 5,
        **kwargs
    ) -> AgentResponse:
        """Generate multiple headline options"""
        import time
        start_time = time.time()
        
        prompt = f"""Generate {num_variants} headline variants for a blog post about: {topic}

Headline types to include:
1. How-to format (e.g., "How to X in 5 Steps")
2. Number list (e.g., "X Things You Need to Know About Y")
3. Question format (e.g., "Are You Making This Common Mistake?")
4. Benefit-driven (e.g., "Get X Results with This Simple Strategy")
5. Controversial/bold (e.g., "Why X is Overrated")

Each headline should:
- Be 50-60 characters (ideal for SEO)
- Include the main keyword
- Create curiosity or promise value
- Be click-worthy"""
        
        messages = [
            SystemMessage(content="You are an expert at writing SEO-optimized blog headlines."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.8,
                max_tokens=1000,
            )
            
            headlines = [h.strip() for h in response.content.split("\n") if h.strip()]
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "headline_count": len(headlines),
                    "headlines": headlines,
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
    
    async def generate_outline(
        self,
        topic: str,
        blog_type: BlogType = BlogType.HOW_TO,
        num_sections: int = 7,
        **kwargs
    ) -> AgentResponse:
        """Generate a blog post outline"""
        import time
        start_time = time.time()
        
        prompt = f"""Create a detailed outline for a {blog_type.value.replace('_', ' ')} blog post about: {topic}

Include:
- H1 headline (compelling)
- Introduction hook (3-5 sentences)
- {num_sections} main H2 sections with:
  - H2 title
  - 3-5 bullet points per section
  - Key points to cover
- H3 sub-sections where needed
- Conclusion with key takeaways
- CTA suggestion

Be specific and detailed in your outline."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=2500,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "blog_type": blog_type.value,
                    "sections_count": num_sections,
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
    
    async def expand_outline(
        self,
        outline: str,
        target_word_count: int = 1500,
        **kwargs
    ) -> AgentResponse:
        """Expand a blog outline into full content"""
        import time
        start_time = time.time()
        
        prompt = f"""Expand this blog outline into a full blog post:

{outline}

Requirements:
- Word count: {target_word_count}+ words
- Expand each section with detailed, valuable content
- Maintain consistent tone throughout
- Include examples, statistics, or quotes where appropriate
- Add transitions between sections
- Keep paragraphs short (2-3 sentences)
- Use bullet points and lists for readability

Write the complete blog post."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=8000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={"target_word_count": target_word_count},
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
    
    def analyze_seo_potential(
        self,
        content: str,
        target_keyword: str,
    ) -> Dict[str, Any]:
        """Analyze SEO potential of blog content"""
        words = content.lower().split()
        total_words = len(words)
        
        keyword_count = words.count(target_keyword.lower())
        keyword_density = (keyword_count / total_words * 100) if total_words > 0 else 0
        
        seo_score = 50.0
        
        if 1.0 <= keyword_density <= 2.0:
            seo_score += 25
        elif 0.5 <= keyword_density <= 3.0:
            seo_score += 15
        
        if content.lower().startswith(target_keyword.lower()):
            seo_score += 10
        
        h2_count = content.count("\n## ")
        if 3 <= h2_count <= 8:
            seo_score += 10
        
        if content.lower().find(target_keyword.lower()) < 300:
            seo_score += 5
        
        return {
            "seo_score": min(seo_score, 100),
            "keyword_density": round(keyword_density, 2),
            "keyword_count": keyword_count,
            "word_count": total_words,
            "recommendations": self._get_seo_recommendations(seo_score, keyword_density, h2_count),
        }
    
    def _get_seo_recommendations(
        self,
        score: float,
        density: float,
        h2_count: int,
    ) -> List[str]:
        """Get SEO recommendations based on analysis"""
        recommendations = []
        
        if density < 0.5:
            recommendations.append("Consider adding more instances of your target keyword")
        elif density > 3.0:
            recommendations.append("Keyword density is too high - reduce keyword usage")
        
        if h2_count < 3:
            recommendations.append("Add more section headers to improve structure")
        
        if score < 70:
            recommendations.append("Consider improving SEO elements like headers and keyword placement")
        
        return recommendations

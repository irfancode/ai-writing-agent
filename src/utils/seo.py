"""SEO Enhancer - Optimizes content for search engines"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..core.llm_manager import LLMManager
from langchain_core.messages import SystemMessage, HumanMessage


@dataclass
class SEOAnalysis:
    score: int
    readability_score: int
    keyword_score: int
    structure_score: int
    suggestions: List[str]
    missing_elements: List[str]


class SEOEnhancer:
    """SEO optimization and analysis tools"""
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model: str = "claude-3-5-sonnet",
    ):
        self.llm_manager = llm_manager or LLMManager()
        self.model = model
    
    async def optimize(
        self,
        content: str,
        keywords: List[str],
        target_search_intent: str = "informational",
        competition_level: str = "medium",
    ) -> str:
        """Optimize content for SEO"""
        
        primary_keyword = keywords[0] if keywords else ""
        secondary_keywords = keywords[1:] if len(keywords) > 1 else []
        
        prompt = f"""Optimize this content for SEO with the following keywords:
- Primary: {primary_keyword}
- Secondary: {', '.join(secondary_keywords)}

Search intent: {target_search_intent}
Competition: {competition_level}

Requirements:
1. Include primary keyword in:
   - Title (first 60 characters)
   - First paragraph (within first 100 words)
   - At least one H2 header
   - Throughout body naturally (1-2% density)
   - Meta description

2. Improve structure:
   - Clear H2/H3 hierarchy
   - Short paragraphs (2-3 sentences)
   - Bullet/numbered lists
   - Conclusion with CTA

3. Add SEO elements:
   - Internal link suggestions
   - External resource suggestions
   - FAQ section (optional)

Content:
{content}

Provide the optimized version with the title and meta description prefixed."""
        
        messages = [
            SystemMessage(content="You are an SEO expert with deep knowledge of search engine optimization best practices."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=6000,
            )
            return response.content
        
        except Exception as e:
            print(f"SEO optimization error: {e}")
            return content
    
    async def analyze(self, content: str, target_keyword: str) -> SEOAnalysis:
        """Analyze SEO potential of content"""
        
        prompt = f"""Analyze this content for SEO optimization:

Target keyword: {target_keyword}

Content:
{content[:3000]}...

Provide a detailed SEO analysis including:
1. Overall SEO score (0-100)
2. Readability score (0-100)
3. Keyword optimization score (0-100)
4. Structure score (0-100)
5. Specific suggestions for improvement
6. Missing SEO elements

Be specific and actionable in your suggestions."""
        
        messages = [
            SystemMessage(content="You are an SEO expert conducting a content audit."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=2000,
            )
            
            return self._parse_seo_analysis(response.content, target_keyword)
        
        except Exception as e:
            return SEOAnalysis(
                score=0,
                readability_score=0,
                keyword_score=0,
                structure_score=0,
                suggestions=[],
                missing_elements=[str(e)],
            )
    
    async def generate_meta_description(
        self,
        content: str,
        target_keyword: str,
        max_length: int = 160,
    ) -> str:
        """Generate an SEO-optimized meta description"""
        
        prompt = f"""Generate an SEO-optimized meta description for this content.

Target keyword: {target_keyword}
Max length: {max_length} characters

Requirements:
- Include target keyword naturally
- Include a call-to-action or value proposition
- Be compelling enough to earn clicks
- Accurately describe content

Content (first 500 words):
{content[:500]}

Provide only the meta description (no quotes)."""
        
        messages = [
            SystemMessage(content="You are an SEO expert specializing in meta descriptions."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=300,
            )
            return response.content.strip()
        
        except Exception as e:
            return ""
    
    async def suggest_keywords(
        self,
        topic: str,
        content: Optional[str] = None,
        num_suggestions: int = 10,
    ) -> List[Dict[str, Any]]:
        """Suggest related keywords for a topic"""
        
        content_hint = f"\n\nExisting content:\n{content[:500]}..." if content else ""
        
        prompt = f"""Suggest {num_suggestions} keywords for this topic:

Topic: {topic}
{content_hint}

For each keyword provide:
1. The keyword phrase
2. Search intent (informational, navigational, transactional, commercial)
3. Difficulty estimate (easy, medium, hard)
4. Why it's relevant

Format as a list."""
        
        messages = [
            SystemMessage(content="You are an SEO keyword research expert."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.6,
                max_tokens=2000,
            )
            
            return self._parse_keyword_suggestions(response.content)
        
        except Exception as e:
            return []
    
    async def generate_faq_section(
        self,
        topic: str,
        num_questions: int = 5,
    ) -> str:
        """Generate FAQ section for SEO"""
        
        prompt = f"""Generate {num_questions} frequently asked questions about: {topic}

Requirements:
- Questions should be what people actually search
- Include the main keyword in questions
- Provide concise, valuable answers
- Format as FAQ with Q&A pairs

This will be used for FAQ rich snippets in search results."""
        
        messages = [
            SystemMessage(content="You are an SEO expert creating FAQ content for featured snippets."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.6,
                max_tokens=2000,
            )
            return response.content
        
        except Exception as e:
            return ""
    
    async def improve_title(
        self,
        content: str,
        target_keyword: str,
        max_length: int = 60,
    ) -> str:
        """Improve the title for SEO"""
        
        prompt = f"""Suggest improvements for this title or generate a better SEO title.

Current title or topic: {content[:100]}...
Target keyword: {target_keyword}
Max length: {max_length} characters

Guidelines:
- Include primary keyword near the beginning
- Be compelling for click-through
- Be specific about what the content covers
- Avoid clickbait but include value proposition
- Numbers often perform well

Provide 3 title options, each on a new line."""
        
        messages = [
            SystemMessage(content="You are an SEO expert specializing in title optimization."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=500,
            )
            
            lines = [l.strip() for l in response.content.split("\n") if l.strip()]
            return lines[0] if lines else content
        
        except Exception as e:
            return content
    
    async def generate_schema_markup(
        self,
        content_type: str = "article",
        title: str = "",
        description: str = "",
        author: str = "",
        date_published: str = "",
        url: str = "",
    ) -> str:
        """Generate schema.org markup for content"""
        
        schema_templates = {
            "article": f'''```json
{{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "{title}",
  "description": "{description}",
  "author": {{
    "@type": "Person",
    "name": "{author}"
  }},
  "datePublished": "{date_published}",
  "url": "{url}"
}}
```''',
            "faq": '''```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "Question 1",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Answer to question 1."
      }
    }
  ]
}
```''',
            "howto": f'''```json
{{
  "@context": "https://schema.org",
  "@type": "HowTo",
  "title": "{title}",
  "description": "{description}",
  "author": {{
    "@type": "Person",
    "name": "{author}"
  }},
  "step": [
    {{
      "@type": "HowToStep",
      "name": "Step 1",
      "text": "Description of step 1."
    }}
  ]
}}
```''',
        }
        
        return schema_templates.get(content_type, schema_templates["article"])
    
    def calculate_keyword_density(self, content: str, keyword: str) -> float:
        """Calculate keyword density percentage"""
        words = content.lower().split()
        keyword_lower = keyword.lower()
        
        if not words:
            return 0.0
        
        keyword_count = sum(1 for w in words if keyword_lower in w)
        return round((keyword_count / len(words)) * 100, 2)
    
    def _parse_seo_analysis(self, content: str, keyword: str) -> SEOAnalysis:
        """Parse SEO analysis response"""
        score = 50
        readability = 70
        keyword_score = 50
        structure_score = 60
        suggestions = []
        missing = []
        
        lines = content.split("\n")
        for line in lines:
            line_lower = line.lower()
            if "score" in line_lower and any(c.isdigit() for c in line):
                nums = [int(s) for s in line if s.isdigit()]
                if nums:
                    score = nums[0]
        
        if "readability" in content.lower():
            suggestions.append("Consider simplifying sentence structure")
        if "keyword" in content.lower():
            suggestions.append("Ensure keyword appears in first 100 words")
        if "structure" in content.lower():
            missing.append("Consider adding more subheadings")
        
        return SEOAnalysis(
            score=min(score, 100),
            readability_score=min(readability, 100),
            keyword_score=min(keyword_score, 100),
            structure_score=min(structure_score, 100),
            suggestions=suggestions,
            missing_elements=missing,
        )
    
    def _parse_keyword_suggestions(self, content: str) -> List[Dict[str, Any]]:
        """Parse keyword suggestions"""
        suggestions = []
        lines = content.split("\n")
        
        for line in lines:
            if line.strip() and any(c.isdigit() for c in line[:5]):
                suggestions.append({
                    "keyword": line.strip(),
                    "intent": "informational",
                    "difficulty": "medium",
                })
        
        return suggestions[:10]

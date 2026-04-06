"""Content Optimizer - Enhances and refines written content"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from ..core.llm_manager import LLMManager
from langchain_core.messages import SystemMessage, HumanMessage


class OptimizationTarget(Enum):
    READABILITY = "readability"
    ENGAGEMENT = "engagement"
    SEO = "seo"
    PROFESSIONAL = "professional"
    CONVERSION = "conversion"
    CLARITY = "clarity"


class ContentOptimizer:
    """Optimizes content for various targets"""
    
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
        target: OptimizationTarget = OptimizationTarget.READABILITY,
        grade_level: int = 12,
        **kwargs
    ) -> str:
        """Optimize content for the specified target"""
        
        target_prompts = {
            OptimizationTarget.READABILITY: f"""Improve the readability of this content:
- Use shorter sentences where possible
- Break up long paragraphs
- Use active voice
- Remove jargon or explain it
- Target grade level: {grade_level}
- Keep meaning and value intact""",
            
            OptimizationTarget.ENGAGEMENT: """Make this content more engaging:
- Add hooks at the beginning
- Use compelling language
- Include questions to provoke thought
- Add emotional resonance
- Make it more memorable
- Keep the core message intact""",
            
            OptimizationTarget.SEO: """Optimize this content for SEO:
- Naturally integrate primary keywords
- Improve keyword distribution
- Enhance header structure
- Add semantic keywords
- Improve readability for search engines
- Keep it natural, not keyword-stuffed""",
            
            OptimizationTarget.PROFESSIONAL: """Make this content more professional:
- Use formal language appropriate for business
- Remove colloquialisms
- Improve structure and flow
- Add credibility signals
- Maintain authority throughout
- Keep it clear and direct""",
            
            OptimizationTarget.CONVERSION: """Optimize this content for conversion:
- Strengthen the value proposition
- Add urgency where appropriate
- Improve CTA effectiveness
- Remove friction points
- Increase desire and motivation
- Keep authenticity""",
            
            OptimizationTarget.CLARITY: """Improve the clarity of this content:
- Make complex ideas simpler
- Use concrete examples
- Remove ambiguity
- Structure logically
- Lead with the main point
- Ensure every sentence adds value""",
        }
        
        prompt = f"""{target_prompts.get(target, target_prompts[OptimizationTarget.READABILITY])}

Content to optimize:
{content}

Provide the optimized version."""
        
        messages = [
            SystemMessage(content="You are an expert content editor and optimizer."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
            )
            return response.content
        
        except Exception as e:
            print(f"Optimization error: {e}")
            return content
    
    async def check_grammar(self, content: str) -> Dict[str, Any]:
        """Check grammar and provide corrections"""
        
        prompt = f"""Check this content for grammar, spelling, and punctuation errors.
Also identify any style issues.

Content:
{content}

Provide:
1. Any corrections needed (with explanations)
2. Style suggestions
3. Overall quality rating (1-10)

Format your response as:
---
CORRECTIONS:
- [correction] → [if needed, explain]

STYLE SUGGESTIONS:
- [suggestion]

QUALITY RATING: [X/10]
---"""
        
        messages = [
            SystemMessage(content="You are an expert grammar and style editor."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.3,
            )
            
            return {
                "content": response.content,
                "suggestions": self._extract_suggestions(response.content),
            }
        
        except Exception as e:
            return {"content": content, "suggestions": [], "error": str(e)}
    
    async def simplify_content(
        self,
        content: str,
        target_grade: int = 8,
    ) -> str:
        """Simplify content for lower reading level"""
        
        prompt = f"""Simplify this content to an approximately {target_grade}th grade reading level.

Rules:
- Use common, everyday words
- Short sentences (average 12-15 words)
- Avoid jargon or define it simply
- Use active voice
- Break complex ideas into smaller parts
- Keep all important information
- Use analogies to familiar concepts

Content:
{content}

Provide the simplified version."""
        
        messages = [
            SystemMessage(content="You are an expert at simplifying complex content for general audiences."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
            )
            return response.content
        
        except Exception as e:
            return content
    
    async def expand_content(
        self,
        content: str,
        expansion_factor: float = 1.5,
    ) -> str:
        """Expand content with more detail"""
        
        target_words = int(len(content.split()) * expansion_factor)
        
        prompt = f"""Expand this content to approximately {target_words} words while adding more value.

How to expand:
- Add more examples
- Provide deeper explanations
- Include relevant statistics or facts
- Add context and background
- Explore nuances
- Include pros and cons if relevant

Content:
{content}

Provide the expanded version."""
        
        messages = [
            SystemMessage(content="You are an expert at expanding content while maintaining quality and adding value."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.6,
            )
            return response.content
        
        except Exception as e:
            return content
    
    async def condense_content(
        self,
        content: str,
        preserve_ratio: float = 0.6,
    ) -> str:
        """Condense content while preserving key points"""
        
        prompt = f"""Condense this content to approximately {int(preserve_ratio * 100)}% of its current length while preserving:
- Key messages
- Essential information
- Any important statistics or facts
- Core arguments

Remove:
- Redundancies
- Filler words
- Tangential points
- Repetitions

Content:
{content}

Provide the condensed version."""
        
        messages = [
            SystemMessage(content="You are an expert at condensing content without losing essential meaning."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.5,
            )
            return response.content
        
        except Exception as e:
            return content
    
    async def change_tone(
        self,
        content: str,
        from_tone: str,
        to_tone: str,
    ) -> str:
        """Change the tone of content"""
        
        prompt = f"""Rewrite this content in a different tone.

Original tone: {from_tone}
New tone: {to_tone}

Content:
{content}

Ensure:
- The core message remains the same
- The new tone is consistent throughout
- The content still feels natural
- Key points are preserved

Provide the rewritten version."""
        
        messages = [
            SystemMessage(content="You are an expert at adapting content to different tones while preserving meaning."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
            )
            return response.content
        
        except Exception as e:
            return content
    
    async def add_cta(
        self,
        content: str,
        cta_type: str = "general",
        position: str = "end",
    ) -> str:
        """Add a call-to-action to content"""
        
        cta_suggestions = {
            "general": "If you found this valuable, consider sharing it with others who might benefit.",
            "subscription": "Subscribe to get more insights like this delivered to your inbox.",
            "engagement": "What are your thoughts? Share in the comments below.",
            "purchase": "Ready to get started? Click here to learn more.",
            "social": "Follow me for more content like this.",
        }
        
        prompt = f"""Add an appropriate call-to-action to this content.

CTA type: {cta_type}
Position: {position}

Suggested CTA direction:
{cta_suggestions.get(cta_type, cta_suggestions['general'])}

Content:
{content}

Provide the content with the CTA added naturally."""
        
        messages = [
            SystemMessage(content="You are an expert at adding effective, natural call-to-actions."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.6,
            )
            return response.content
        
        except Exception as e:
            return content
    
    def _extract_suggestions(self, content: str) -> List[str]:
        """Extract suggestions from grammar check response"""
        suggestions = []
        lines = content.split("\n")
        
        for line in lines:
            if "SUGGESTION" in line.upper() or "- " in line:
                suggestions.append(line.strip())
        
        return suggestions[:10]
    
    def calculate_readability(self, content: str) -> Dict[str, Any]:
        """Calculate various readability metrics"""
        import re
        
        words = content.split()
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"flesch_reading_ease": 0, "flesch_kincaid_grade": 0}
        
        total_syllables = sum(self._count_syllables(w) for w in words)
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_syllables_per_word = total_syllables / len(words) if words else 0
        
        flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_kincaid_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        
        return {
            "flesch_reading_ease": round(max(0, min(100, flesch_reading_ease)), 1),
            "flesch_kincaid_grade": round(max(0, flesch_kincaid_grade), 1),
            "avg_sentence_length": round(avg_sentence_length, 1),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "total_words": len(words),
            "total_sentences": len(sentences),
        }
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word"""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        if word.endswith("e"):
            count -= 1
        
        return max(1, count)

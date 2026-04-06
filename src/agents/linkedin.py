"""LinkedIn Agent - Generates professional LinkedIn content"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import BaseWritingAgent, WritingRequirements, AgentResponse, WritingTone, ContentLength


class LinkedInContentType(Enum):
    POST = "post"
    ARTICLE = "article"
    CAROUSEL_IDEA = "carousel_idea"
    COMMENT = "comment"
    PROFILE_UPDATE = "profile_update"
    COMPANY_UPDATE = "company_update"
    JOB_POSTING = "job_posting"


class LinkedInPostStyle(Enum):
    THOUGHT_LEADERSHIP = "thought_leadership"
    PERSONAL_STORY = "personal_story"
    INDUSTRY_INSIGHT = "industry_insight"
    HOW_TO = "how_to"
    LISTICLE = "listicle"
    CASE_STUDY = "case_study"
    NEWS_REACTION = "news_reaction"


class LinkedInAgent(BaseWritingAgent):
    """Agent specialized in LinkedIn content generation"""
    
    name = "linkedin_agent"
    description = "Generates professional LinkedIn content for thought leadership and networking"
    
    POST_STYLE_PROMPTS = {
        LinkedInPostStyle.THOUGHT_LEADERSHIP: """You are a LinkedIn thought leader. Create content that:
- Challenges conventional wisdom
- Provides unique perspectives
- Back claims with reasoning or data
- Sparks meaningful discussion
- Positions you as an authority""",
        
        LinkedInPostStyle.PERSONAL_STORY: """You are sharing a personal story on LinkedIn. Create content that:
- Opens with a relatable hook
- Includes specific details and emotions
- Extracts lessons or insights
- Connects personal to professional
- Ends with a takeaway or question""",
        
        LinkedInPostStyle.INDUSTRY_INSIGHT: """You are an industry analyst on LinkedIn. Create content that:
- Provides data-driven insights
- Identifies trends and patterns
- Offers predictions or forecasts
- References current events
- Positions you as informed""",
        
        LinkedInPostStyle.HOW_TO: """You are sharing knowledge on LinkedIn. Create content that:
- Starts with a compelling problem
- Provides actionable steps
- Includes specific examples
- Offers tips and best practices
- Encourages questions""",
        
        LinkedInPostStyle.LISTICLE: """You are creating a listicle on LinkedIn. Create content that:
- Uses a numbered list format
- Provides brief, valuable points
- Each point stands alone
- Has a compelling intro and conclusion
- Drives engagement through curiosity""",
        
        LinkedInPostStyle.CASE_STUDY: """You are sharing a case study on LinkedIn. Create content that:
- Presents the challenge clearly
- Shows the approach/solution
- Includes measurable results
- Extracts key learnings
- Positions the author as capable""",
        
        LinkedInPostStyle.NEWS_REACTION: """You are reacting to industry news on LinkedIn. Create content that:
- References the specific news
- Provides immediate reaction
- Offers unique perspective
- Doesn't just repeat the news
- Invites discussion""",
    }
    
    def get_system_prompt(self) -> str:
        return """You are an expert LinkedIn content creator specializing in thought leadership and professional engagement.

LinkedIn content best practices:
1. First line must hook immediately (stop the scroll)
2. Use line breaks for readability (short paragraphs)
3. Include 1-2 relevant emojis strategically
4. End with a question or call-to-action
5. Use 3-5 relevant hashtags at the end
6. Keep posts between 1300-3000 characters
7. Be authentic, not salesy
8. Show expertise through insights, not claims

LinkedIn content should:
- Provide genuine value
- Spark conversation
- Build trust and credibility
- Reflect your professional brand
- Be appropriate for a professional network"""
    
    def get_content_prompt(self, requirements: WritingRequirements) -> str:
        style = requirements.additional_context or LinkedInPostStyle.THOUGHT_LEADERSHIP.value
        
        prompt_parts = [
            f"Content Style: {style.replace('_', ' ').title()}",
            f"Topic: {requirements.topic}",
        ]
        
        if requirements.tone:
            prompt_parts.append(f"Tone: {requirements.tone.value}")
        
        if requirements.audience:
            prompt_parts.append(f"Target Audience: {requirements.audience}")
        
        if requirements.goal:
            prompt_parts.append(f"Goal: {requirements.goal}")
        
        return f"""Create a LinkedIn post.

Requirements:
{chr(10).join(prompt_parts)}

Output format:
---
[Your LinkedIn post content]

---

Include relevant hashtags at the end (3-5)."""
    
    async def generate_post(
        self,
        topic: str,
        style: LinkedInPostStyle = LinkedInPostStyle.THOUGHT_LEADERSHIP,
        tone: WritingTone = WritingTone.PROFESSIONAL,
        audience: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        length: ContentLength = ContentLength.MEDIUM,
        goal: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a LinkedIn post"""
        import time
        start_time = time.time()
        
        prompt = self.POST_STYLE_PROMPTS.get(style, self.POST_STYLE_PROMPTS[LinkedInPostStyle.THOUGHT_LEADERSHIP])
        prompt += f"\n\nGenerate a LinkedIn post about: {topic}\n"
        
        if tone:
            prompt += f"Tone: {tone.value}\n"
        
        if audience:
            prompt += f"Target: {audience}\n"
        
        if goal:
            prompt += f"Goal: {goal}\n"
        
        if keywords:
            prompt += f"Key themes: {', '.join(keywords)}\n"
        
        prompt += "\nEnd with a question to encourage engagement."
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Create an engaging LinkedIn post. Include line breaks, strategic emojis, and hashtags."),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=2000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "content_type": "post",
                    "style": style.value,
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
    
    async def generate_article(
        self,
        topic: str,
        tone: WritingTone = WritingTone.PROFESSIONAL,
        audience: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a LinkedIn article"""
        import time
        start_time = time.time()
        
        prompt = f"""Write a LinkedIn article about: {topic}

Target Audience: {audience or 'Professionals in the field'}
Tone: {tone.value if tone else 'professional'}

Structure:
1. Compelling title
2. Introduction (hook the reader)
3. Main body with headers (3-5 sections)
4. Conclusion with key takeaways
5. Call-to-action

Articles should:
- Provide in-depth value
- Be well-structured with headers
- Include examples and data when relevant
- Be 1000-2000 words
- End with discussion questions

Include a featured image suggestion at the end."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=5000,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "content_type": "article",
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
    
    async def generate_carousel_idea(
        self,
        topic: str,
        num_slides: int = 10,
        **kwargs
    ) -> AgentResponse:
        """Generate carousel content ideas and outline"""
        import time
        start_time = time.time()
        
        prompt = f"""Create a {num_slides}-slide LinkedIn carousel outline about: {topic}

Format for each slide:
Slide # - [Title] - [Key points]

Structure:
- Slide 1: Hook/Title
- Slides 2-{num_slides-1}: Content slides
- Slide {num_slides}: Conclusion/CTA

Each slide should have:
- Clear, concise title
- 2-3 bullet points max
- Visual cue for design

For the final slide, include your CTA."""
        
        messages = [
            SystemMessage(content="You are a LinkedIn content expert specializing in carousel formats."),
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
                    "content_type": "carousel",
                    "slide_count": num_slides,
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
    
    async def generate_comment(
        self,
        post_content: str,
        comment_type: str = "insightful",
        **kwargs
    ) -> AgentResponse:
        """Generate a LinkedIn comment"""
        import time
        start_time = time.time()
        
        comment_types = {
            "insightful": "Provide a thoughtful, valuable perspective that adds to the discussion.",
            "question": "Ask a follow-up question to encourage more engagement.",
            "supportive": "Show genuine support or agreement with a personal angle.",
            "debate": "Offer a different perspective respectfully.",
            "celebratory": "Congratulate or celebrate an achievement.",
        }
        
        prompt = f"""Generate a LinkedIn comment on this post:

{post_content}

Comment Type: {comment_type}
Style: {comment_types.get(comment_type, comment_types['insightful'])}

Comment should:
- Be authentic and genuine
- Add value to the conversation
- Be 1-3 sentences
- Not be generic or spammy"""
        
        messages = [
            SystemMessage(content="You are an expert at writing authentic, engaging LinkedIn comments."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=500,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={"comment_type": comment_type},
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
    
    async def generate_profile_section(
        self,
        section: str,
        current_content: Optional[str] = None,
        target_audience: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate LinkedIn profile section content"""
        import time
        start_time = time.time()
        
        section_prompts = {
            "headline": "Create a compelling LinkedIn headline (max 220 characters) that showcases your value proposition.",
            "about": "Write a LinkedIn About section that tells your story, highlights achievements, and connects with your target audience.",
            "experience": "Write engaging experience descriptions that highlight impact and achievements.",
            "featured": "Suggest content for the Featured section.",
        }
        
        prompt = f"""Generate a LinkedIn {section} section.\n"""
        
        if current_content:
            prompt += f"Current content:\n{current_content}\n\n"
        
        if target_audience:
            prompt += f"Target Audience: {target_audience}\n"
        
        prompt += f"\n{section_prompts.get(section, section_prompts['about'])}"
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=1500,
            )
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={"profile_section": section},
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
    
    def estimate_engagement(
        self,
        content: str,
        has_image: bool = False,
        has_hashtags: bool = True,
    ) -> Dict[str, float]:
        """Estimate engagement potential"""
        base_score = 50.0
        
        if len(content) >= 1300:
            base_score += 10
        
        if has_image:
            base_score += 15
        
        if has_hashtags:
            base_score += 5
        
        hook_indicators = ["Here's", "The", "Why", "How", "I", "We", "This", "What"]
        if any(content.startswith(ind) for ind in hook_indicators):
            base_score += 10
        
        question_count = content.count("?")
        if question_count > 0:
            base_score += 5 * min(question_count, 3)
        
        return {
            "engagement_score": min(base_score, 98.0),
            "estimated_likes": int(base_score * 0.5),
            "estimated_comments": int(base_score * 0.1),
        }

"""Social Media Agent - Generates engaging social media content"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import BaseWritingAgent, WritingRequirements, AgentResponse, WritingTone, ContentLength


class Platform(Enum):
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"
    THREADS = "threads"


class PostType(Enum):
    SINGLE = "single"
    THREAD = "thread"
    CAROUSEL = "carousel"
    STORY = "story"
    REEL = "reel"


@dataclass
class SocialMediaRequirements:
    platform: Platform
    post_type: PostType = PostType.SINGLE
    include_hashtags: bool = True
    include_emoji: bool = True
    add_cta: bool = False
    call_to_action: Optional[str] = None
    thread_count: int = 5


class SocialMediaAgent(BaseWritingAgent):
    """Agent specialized in social media content generation"""
    
    name = "social_media_agent"
    description = "Generates engaging social media content for various platforms"
    
    PLATFORM_PROMPTS = {
        Platform.TWITTER: """You are an expert Twitter content creator. Create engaging, punchy tweets that capture attention in the first 2 seconds. Use:
- Hooks that create curiosity or emotion
- Clear value proposition
- Natural hashtags (max 3)
- A compelling CTA when appropriate
- Character limit awareness (280 chars)""",
        
        Platform.INSTAGRAM: """You are an expert Instagram content creator. Create visually descriptive captions that complement images. Use:
- Story-driven hooks
- Emojis strategically (not overused)
- Line breaks for readability
- Relevant hashtags (5-15)
- Engagement prompts""",
        
        Platform.FACEBOOK: """You are an expert Facebook content creator. Create community-focused posts that encourage interaction. Use:
- Conversational tone
- Questions or prompts for comments
- Shareable hooks
- Relevant hashtags
- Call-to-action for engagement""",
        
        Platform.TIKTOK: """You are an expert TikTok content creator. Create viral-worthy hooks and script ideas. Use:
- Attention-grabbing openings
- Trend integration
- Clear value or entertainment
- Duet-worthy elements
- Trendy language and sounds""",
        
        Platform.LINKEDIN: """You are an expert LinkedIn content creator. Create professional yet personable posts. Use:
- Professional insights
- Personal experiences
- Industry relevance
- Professional hashtags (3-5)
- Thought-provoking questions""",
        
        Platform.THREADS: """You are an expert Threads content creator. Create conversational, authentic content. Use:
- Casual, real tone
- Personal opinions
- Conversational hooks
- Minimal hashtags
- Genuine engagement""",
    }
    
    def get_system_prompt(self) -> str:
        return """You are an expert social media content creator with deep knowledge of viral content patterns, engagement strategies, and platform-specific best practices.

Your content should:
1. Hook immediately in the first line
2. Provide genuine value or spark emotion
3. Be authentic and not salesy
4. Encourage engagement through questions or reactions
5. Use platform-native language and style

Always consider: Would this make someone stop scrolling?"""
    
    def get_content_prompt(self, requirements: WritingRequirements) -> str:
        platform = requirements.additional_context or Platform.TWITTER.value
        platform_enum = Platform(platform) if isinstance(platform, str) else platform
        
        prompt_parts = [
            f"Platform: {platform_enum.value.upper()}",
            f"Topic: {requirements.topic}",
            f"Content Type: {requirements.length.value}",
        ]
        
        if requirements.tone:
            prompt_parts.append(f"Tone: {requirements.tone.value}")
        
        if requirements.audience:
            prompt_parts.append(f"Target Audience: {requirements.audience}")
        
        if requirements.keywords:
            prompt_parts.append(f"Key Themes: {', '.join(requirements.keywords)}")
        
        if requirements.goal:
            prompt_parts.append(f"Goal: {requirements.goal}")
        
        platform_instruction = self.PLATFORM_PROMPTS.get(
            platform_enum,
            self.PLATFORM_PROMPTS[Platform.TWITTER]
        )
        
        return f"""{platform_instruction}

Requirements:
{chr(10).join(prompt_parts)}

Generate {requirements.length.value} social media content for this topic. Make it engaging, authentic, and platform-native.

Output format:
---
[Your generated content here]
---

If generating a thread, separate each tweet with "---NEXT TWEET---"
If generating multiple options, separate with "---OPTION---" """
    
    async def generate_post(
        self,
        topic: str,
        platform: Platform = Platform.TWITTER,
        post_type: PostType = PostType.SINGLE,
        tone: WritingTone = WritingTone.FRIENDLY,
        audience: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        length: ContentLength = ContentLength.SHORT,
        include_hashtags: bool = True,
        include_emoji: bool = True,
        add_cta: bool = False,
        **kwargs
    ) -> AgentResponse:
        """Generate social media content with platform-specific optimizations"""
        import time
        start_time = time.time()
        
        context = {
            "platform": platform.value,
            "post_type": post_type.value,
            "include_hashtags": include_hashtags,
            "include_emoji": include_emoji,
            "add_cta": add_cta,
        }
        
        requirements = WritingRequirements(
            topic=topic,
            tone=tone,
            audience=audience,
            keywords=keywords or [],
            length=length,
            additional_context=platform.value,
        )
        
        return await self.generate(
            topic=topic,
            tone=tone,
            audience=audience,
            keywords=keywords,
            length=length,
            additional_context=str(context),
        )
    
    async def generate_thread(
        self,
        topic: str,
        platform: Platform = Platform.TWITTER,
        num_tweets: int = 5,
        thread_opener: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a thread of posts"""
        import time
        start_time = time.time()
        
        prompt = f"""Generate a {num_tweets}-tweet thread on: {topic}

{thread_opener if thread_opener else 'Create an engaging thread that tells a story or delivers value progressively.'}

Structure the thread as:
1. Hook/Setup tweet
2-{num_tweets-1}. Content tweets (delivering value)
{num_tweets}. CTA/conclusion tweet

Format: Separate each tweet with "---NEXT TWEET---"

Make each tweet valuable on its own but also build toward a conclusion."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.8,
                max_tokens=2000,
            )
            
            tweets = response.content.split("---NEXT TWEET---")
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "platform": platform.value,
                    "post_type": "thread",
                    "tweet_count": len(tweets),
                    "tweets": [t.strip() for t in tweets if t.strip()],
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
    
    async def generate_engagement_hooks(
        self,
        topic: str,
        platform: Platform = Platform.TWITTER,
        num_hooks: int = 5,
    ) -> AgentResponse:
        """Generate multiple engagement hook options"""
        import time
        start_time = time.time()
        
        prompt = f"""Generate {num_hooks} engaging opening hooks for a {platform.value} post about: {topic}

Types of hooks to use:
1. Question hook
2. Statistic/fact hook
3. Contrarian/opinion hook
4. Story hook
5. Contrast hook

Format each hook for {platform.value} (under 280 chars for Twitter)."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.9,
                max_tokens=1500,
            )
            
            hooks = [h.strip() for h in response.content.split("\n") if h.strip()]
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "platform": platform.value,
                    "hook_type": "engagement_hooks",
                    "hooks": hooks,
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
    
    def suggest_hashtags(
        self,
        topic: str,
        platform: Platform = Platform.TWITTER,
        num_hashtags: int = 5,
    ) -> List[str]:
        """Suggest relevant hashtags for a topic"""
        hashtag_strategies = {
            Platform.TWITTER: num_hashtags,
            Platform.INSTAGRAM: min(num_hashtags + 5, 15),
            Platform.LINKEDIN: min(num_hashtags, 5),
            Platform.FACEBOOK: min(num_hashtags, 3),
            Platform.TIKTOK: min(num_hashtags, 5),
            Platform.THREADS: min(num_hashtags, 2),
        }
        
        return [f"#{topic.replace(' ', '').lower()[:20]}"]

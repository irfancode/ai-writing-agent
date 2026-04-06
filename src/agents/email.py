"""Email Agent - Generates professional email content"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import BaseWritingAgent, WritingRequirements, AgentResponse, WritingTone, ContentLength


class EmailType(Enum):
    COLD_OUTREACH = "cold_outreach"
    WARM_INTRO = "warm_intro"
    FOLLOW_UP = "follow_up"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    INTERNAL = "internal"
    THANK_YOU = "thank_you"
    MEETING_REQUEST = "meeting_request"
    INTRODUCTION = "introduction"
    REENGAGEMENT = "reengagement"


class EmailTone(Enum):
    PROFESSIONAL = "professional"
    FRIENDLY_PROFESSIONAL = "friendly_professional"
    CASUAL = "casual"
    FORMAL = "formal"
    PERSUASIVE = "persuasive"
    URGENT = "urgent"


@dataclass
class EmailRequirements:
    email_type: EmailType
    recipient_name: Optional[str] = None
    recipient_title: Optional[str] = None
    company: Optional[str] = None
    sender_name: Optional[str] = None
    subject_line: Optional[str] = None
    goal: Optional[str] = None
    personalization_notes: Optional[str] = None


class EmailAgent(BaseWritingAgent):
    """Agent specialized in email copywriting"""
    
    name = "email_agent"
    description = "Generates professional email content for various purposes"
    
    EMAIL_FRAMEWORKS = {
        EmailType.COLD_OUTREACH: {
            "name": "AIDA (Attention-Interest-Desire-Action)",
            "description": "Best for sales and partnership cold emails",
            "structure": "Hook → Problem → Solution → CTA",
        },
        EmailType.FOLLOW_UP: {
            "name": "Simple Follow-up",
            "description": "Gentle reminder with added value",
            "structure": "Reminder → Value Add → Soft CTA",
        },
        EmailType.NEWSLETTER: {
            "name": "Newsletter Format",
            "description": "Engaging newsletter content",
            "structure": "Hook → Highlights → CTA → Unsubscribe",
        },
        EmailType.MEETING_REQUEST: {
            "name": "Meeting Request",
            "description": "Professional meeting request",
            "structure": "Introduction → Purpose → Proposed Solution → Meeting CTA",
        },
    }
    
    def get_system_prompt(self) -> str:
        return """You are an expert email copywriter with deep expertise in conversion-focused email marketing.

Your emails should:
1. Have compelling subject lines that increase open rates
2. Open with a strong hook that creates urgency or curiosity
3. Be scannable with short paragraphs and bullet points
4. Maintain a consistent voice throughout
5. Include clear, single CTA
6. Be personalized to the recipient

Email length guidelines:
- Cold outreach: 100-150 words
- Follow-up: 50-100 words
- Newsletter: 300-500 words
- Meeting request: 100-150 words

Always consider: Will this email get opened, read, and acted upon?"""
    
    def get_content_prompt(self, requirements: WritingRequirements) -> str:
        email_type = requirements.additional_context or EmailType.COLD_OUTREACH.value
        email_type_enum = EmailType(email_type) if isinstance(email_type, str) else email_type
        
        prompt_parts = [
            f"Email Type: {email_type_enum.value.replace('_', ' ').title()}",
            f"Topic/Purpose: {requirements.topic}",
        ]
        
        if requirements.tone:
            prompt_parts.append(f"Tone: {requirements.tone.value}")
        
        if requirements.audience:
            prompt_parts.append(f"Target Recipient: {requirements.audience}")
        
        if requirements.goal:
            prompt_parts.append(f"Goal: {requirements.goal}")
        
        if requirements.keywords:
            prompt_parts.append(f"Key Points: {', '.join(requirements.keywords)}")
        
        framework = self.EMAIL_FRAMEWORKS.get(email_type_enum, self.EMAIL_FRAMEWORKS[EmailType.COLD_OUTREACH])
        
        return f"""Generate a professional email using the {framework['name']} framework.

Framework: {framework['description']}
Structure: {framework['structure']}

Requirements:
{chr(10).join(prompt_parts)}

Output format:
---
Subject: [Compelling subject line]

[Email body]
---

Keep subject lines under 50 characters for best open rates.
Email body should be concise, scannable, and action-oriented."""
    
    async def generate_email(
        self,
        topic: str,
        email_type: EmailType = EmailType.COLD_OUTREACH,
        recipient_name: Optional[str] = None,
        recipient_title: Optional[str] = None,
        sender_name: Optional[str] = None,
        tone: WritingTone = WritingTone.PROFESSIONAL,
        goal: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        personalization_notes: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a complete email with subject line"""
        import time
        start_time = time.time()
        
        prompt = f"""Generate a {email_type.value.replace('_', ' ')} email.

"""
        
        if recipient_name:
            prompt += f"Recipient: {recipient_name}"
            if recipient_title:
                prompt += f" ({recipient_title})"
            prompt += "\n"
        
        if sender_name:
            prompt += f"Sender: {sender_name}\n"
        
        prompt += f"\nTopic/Purpose: {topic}\n"
        
        if tone:
            prompt += f"Tone: {tone.value}\n"
        
        if goal:
            prompt += f"Goal: {goal}\n"
        
        if keywords:
            prompt += f"Key points to include: {', '.join(keywords)}\n"
        
        if personalization_notes:
            prompt += f"\nPersonalization notes:\n{personalization_notes}\n"
        
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
            
            content = response.content
            subject = ""
            
            if "Subject:" in content:
                parts = content.split("Subject:")
                if len(parts) > 1:
                    subject_and_body = parts[1]
                    if "---" in subject_and_body:
                        subject_parts = subject_and_body.split("---")
                        subject = subject_parts[0].strip()
                        content = subject_parts[1].strip() if len(subject_parts) > 1 else content
                    elif "\n\n" in subject_and_body:
                        first_newline = subject_and_body.index("\n\n")
                        subject = subject_and_body[:first_newline].strip()
                        content = subject_and_body[first_newline + 2:].strip()
            
            return AgentResponse(
                content=content,
                agent_name=self.name,
                metadata={
                    "email_type": email_type.value,
                    "subject": subject,
                    "recipient": recipient_name,
                    "sender": sender_name,
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
    
    async def generate_sequence(
        self,
        topic: str,
        num_emails: int = 5,
        email_type: EmailType = EmailType.COLD_OUTREACH,
        **kwargs
    ) -> AgentResponse:
        """Generate an email sequence"""
        import time
        start_time = time.time()
        
        sequence_types = {
            EmailType.COLD_OUTREACH: "A 5-email cold outreach sequence",
            EmailType.FOLLOW_UP: "A 3-email follow-up sequence",
            EmailType.NEWSLETTER: "A weekly newsletter structure",
        }
        
        prompt = f"""Generate a {num_emails}-email {email_type.value.replace('_', ' ')} sequence.

Topic: {topic}

For each email in the sequence:
1. Email # (Day X)
2. Subject line
3. Email body (100-150 words each)
4. CTA

Sequence structure:
- Email 1: Hook/Introduction
- Email 2: Value proposition
- Email 3: Social proof/case study
- Email 4: Urgency/limitation
- Email 5: Final CTA

Separate each email with "---NEXT EMAIL---" """
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=4000,
            )
            
            emails = response.content.split("---NEXT EMAIL---")
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "sequence_type": email_type.value,
                    "email_count": len(emails),
                    "emails": [e.strip() for e in emails if e.strip()],
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
    
    async def generate_subject_lines(
        self,
        topic: str,
        num_variants: int = 10,
        goal: str = "get_opened",
    ) -> AgentResponse:
        """Generate multiple subject line variants"""
        import time
        start_time = time.time()
        
        prompt = f"""Generate {num_variants} subject line variations for an email about: {topic}

Goal: {goal}

Categories:
1. Curiosity-driven
2. Benefit-focused
3. Urgency-based
4. Personal
5. Question-based
6. Numbers/stats
7. Controversial/take
8. Short/punchy
9. Long/descriptive
10. Personalized

Keep each subject under 50 characters.
Number each subject line."""
        
        messages = [
            SystemMessage(content="You are an expert at writing email subject lines that maximize open rates."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm_manager.generate(
                messages=messages,
                model=self.model,
                temperature=0.8,
                max_tokens=1000,
            )
            
            subjects = [s.strip() for s in response.content.split("\n") if s.strip() and any(c.isdigit() for c in s[:3])]
            
            return AgentResponse(
                content=response.content,
                agent_name=self.name,
                metadata={
                    "subject_count": len(subjects),
                    "subjects": subjects,
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
    
    def estimate_open_rate(self, subject: str) -> float:
        """Estimate open rate potential (0-100%)"""
        score = 50.0
        
        if len(subject) <= 40:
            score += 15
        elif len(subject) <= 50:
            score += 10
        
        question_marks = subject.count("?")
        if question_marks > 0:
            score += 10
        
        personalization_indicators = ["you", "your"]
        if any(ind in subject.lower() for ind in personalization_indicators):
            score += 10
        
        urgency_words = ["today", "now", "limited", "last", "only", "don't miss"]
        if any(word in subject.lower() for word in urgency_words):
            score += 10
        
        numbers_indicators = ["numbers", "digits"]
        if any(c.isdigit() for c in subject):
            score += 5
        
        return min(score, 98.0)

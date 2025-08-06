# product_agent.py - LangGraph Implementation (Fixed)
import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, asdict
from datetime import datetime
import base64
from io import BytesIO

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Other imports
from groq import Groq
from huggingface_hub import InferenceClient
from PIL import Image
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Structures ---
@dataclass
class CompetitiveInsight:
    competitor: str
    product_features: List[str]
    pricing: Optional[str]
    market_position: str
    strengths: List[str]
    weaknesses: List[str]

@dataclass
class SentimentData:
    platform: str
    overall_sentiment: float
    positive_mentions: int
    negative_mentions: int
    neutral_mentions: int
    key_themes: List[str]
    explanation: str = ""
    confidence_score: float = 0.0

@dataclass
class LaunchStrategy:
    positioning_statement: str
    key_differentiators: List[str]
    target_audience: str
    pricing_recommendation: str
    launch_timeline: List[str]
    risk_mitigation: List[str]
    success_metrics: List[str]

@dataclass
class SocialMediaPost:
    platform: str
    post_type: str
    text_content: str
    image_prompt: str
    hashtags: List[str]
    image_url: Optional[str] = None

# --- LangGraph State Definition ---
class ProductLaunchState(TypedDict):
    product_info: Dict[str, Any]
    competitive_insights: List[Dict[str, Any]]
    sentiment_analysis: List[Dict[str, Any]]
    launch_strategy: Optional[Dict[str, Any]]
    social_media_posts: List[Dict[str, Any]]
    current_step: str
    errors: List[str]
    retry_count: int
    execution_metadata: Dict[str, Any]

# --- Agent Tools and Utilities ---
class AgentTools:
    def __init__(self, groq_api_key: str, tavily_api_key: str, hf_token: str):
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.tavily_search = TavilySearch(api_key=tavily_api_key, max_results=5) if tavily_api_key else None
        self.hf_client = InferenceClient(token=hf_token) if hf_token else None
        
    async def call_groq(self, prompt: str, system_prompt: str = "", model: str = "llama3-70b-8192", json_mode: bool = False) -> Optional[str]:
        if not self.groq_client:
            raise ValueError("Groq client not initialized")
        
        try:
            await asyncio.sleep(1)  # Rate limiting
            logger.info("Making Groq API call...")
            
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                temperature=0.4,
                response_format={"type": "json_object"} if json_mode else None
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

# --- Agent Implementations ---
async def competitive_research_agent(state: ProductLaunchState, tools: AgentTools) -> ProductLaunchState:
    """Analyze competitive landscape"""
    logger.info("🔍 Starting competitive research...")
    
    try:
        product_info = state["product_info"]
        
        # Step 1: Identify competitors
        competitors_prompt = f"""For a product named '{product_info['name']}' in '{product_info['category']}' category, find 3 main competitors. 
        Respond ONLY with a valid JSON object: {{"competitors": ["Competitor A", "Competitor B", "Competitor C"]}}"""
        
        competitors_response = await tools.call_groq(
            competitors_prompt, 
            "You are a market research expert. You only output valid JSON.",
            json_mode=True
        )
        
        competitors_data = json.loads(competitors_response)
        competitors = competitors_data.get("competitors", [])
        
        if not competitors:
            raise ValueError("No competitors identified")
        
        # Step 2: Analyze each competitor
        competitive_insights = []
        for competitor in competitors:
            analysis_prompt = f"""Analyze competitor '{competitor}' vs. our product '{product_info['name']}'. 
            Respond ONLY with a valid JSON object with EXACTLY these fields:
            {{
                "product_features": ["feature1", "feature2", "feature3"],
                "pricing": "pricing info or estimate",
                "market_position": "position description",
                "strengths": ["strength1", "strength2"],
                "weaknesses": ["weakness1", "weakness2"]
            }}
            
            Do NOT include any other fields like 'competitor_analysis' or additional data."""
            
            analysis_response = await tools.call_groq(
                analysis_prompt,
                "You are a competitive intelligence analyst. You only output valid JSON with the exact fields requested. Do not add extra fields.",
                json_mode=True
            )
            
            analysis_data = json.loads(analysis_response)
            
            # Validate and clean the response data
            cleaned_data = {
                "product_features": analysis_data.get("product_features", []),
                "pricing": analysis_data.get("pricing", "Unknown"),
                "market_position": analysis_data.get("market_position", "Unknown position"),
                "strengths": analysis_data.get("strengths", []),
                "weaknesses": analysis_data.get("weaknesses", [])
            }
            
            insight = CompetitiveInsight(
                competitor=competitor,
                **cleaned_data
            )
            competitive_insights.append(asdict(insight))
        
        logger.info(f"✅ Competitive research completed: {len(competitive_insights)} competitors analyzed")
        
        return {
            **state,
            "competitive_insights": competitive_insights,
            "current_step": "competitive_analysis_complete",
            "execution_metadata": {
                **state.get("execution_metadata", {}),
                "competitive_research_timestamp": datetime.now().isoformat(),
                "competitors_found": len(competitive_insights)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Competitive research failed: {e}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Competitive analysis failed: {str(e)}"],
            "current_step": "error",
            "retry_count": state.get("retry_count", 0) + 1
        }

async def sentiment_analysis_agent(state: ProductLaunchState, tools: AgentTools) -> ProductLaunchState:
    """Analyze market sentiment across platforms"""
    logger.info("📊 Starting sentiment analysis...")
    
    try:
        if not tools.tavily_search:
            raise ValueError("Tavily search tool not available")
        
        product_info = state["product_info"]
        platforms = ["Twitter", "Reddit", "LinkedIn"]
        sentiment_results = []
        
        for platform in platforms:
            logger.info(f"Analyzing sentiment on {platform}...")
            
            # Platform-specific search queries
            platform_queries = {
                "Twitter": f"recent tweets about \"{product_info['name']}\" OR \"{product_info['category']}\" products reviews opinions site:twitter.com",
                "Reddit": f"reddit discussions reviews \"{product_info['name']}\" OR \"{product_info['category']}\" products opinions site:reddit.com",
                "LinkedIn": f"professional opinions \"{product_info['name']}\" OR \"{product_info['category']}\" business reviews site:linkedin.com"
            }
            
            query = platform_queries.get(platform, f"recent discussions and reviews of \"{product_info['name']}\"")
            
            try:
                # Search for content
                search_results = await tools.tavily_search.ainvoke(query)
                
                if not search_results:
                    logger.warning(f"No search results for {platform}")
                    # Create default sentiment data for platforms with no results
                    sentiment_obj = SentimentData(
                        platform=platform,
                        overall_sentiment=0.0,
                        positive_mentions=0,
                        negative_mentions=0,
                        neutral_mentions=0,
                        key_themes=[f"Limited {platform} presence"],
                        explanation=f"No recent discussions found on {platform}",
                        confidence_score=0.1
                    )
                    sentiment_results.append(asdict(sentiment_obj))
                    continue
                
                # Prepare context for analysis
                context = str(search_results)[:4000] if search_results else ""
                
                # Analyze sentiment
                sentiment_prompt = f"""Based on these RECENT search results from '{platform}':\n{context}\n\n
                Analyze the public sentiment for a product like '{product_info['name']}' ({product_info['category']} category).
                
                Respond ONLY with a valid JSON object with EXACTLY these fields:
                {{
                    "overall_sentiment": 0.5,
                    "positive_mentions": 100,
                    "negative_mentions": 20,
                    "neutral_mentions": 80,
                    "key_themes": ["theme1", "theme2", "theme3"],
                    "explanation": "Brief explanation of sentiment drivers on {platform}",
                    "confidence_score": 0.8
                }}
                
                Do NOT include any other fields. overall_sentiment should be between -1.0 and 1.0."""
                
                sentiment_response = await tools.call_groq(
                    sentiment_prompt,
                    f"You are a social media analyst specializing in {platform}. Output only valid JSON with exact fields requested.",
                    json_mode=True
                )
                
                sentiment_data = json.loads(sentiment_response)
                
                # Validate and clean the response data
                cleaned_sentiment = {
                    "overall_sentiment": max(-1.0, min(1.0, sentiment_data.get("overall_sentiment", 0.0))),
                    "positive_mentions": max(0, sentiment_data.get("positive_mentions", 0)),
                    "negative_mentions": max(0, sentiment_data.get("negative_mentions", 0)),
                    "neutral_mentions": max(0, sentiment_data.get("neutral_mentions", 0)),
                    "key_themes": sentiment_data.get("key_themes", [f"{platform} themes"]),
                    "explanation": sentiment_data.get("explanation", f"Analysis based on {platform} data"),
                    "confidence_score": max(0.0, min(1.0, sentiment_data.get("confidence_score", 0.5)))
                }
                
                sentiment_obj = SentimentData(
                    platform=platform,
                    **cleaned_sentiment
                )
                
                sentiment_results.append(asdict(sentiment_obj))
                
            except Exception as platform_error:
                logger.warning(f"Failed to analyze {platform}: {platform_error}")
                # Create fallback data for failed platforms
                fallback_sentiment = SentimentData(
                    platform=platform,
                    overall_sentiment=0.0,
                    positive_mentions=0,
                    negative_mentions=0,
                    neutral_mentions=0,
                    key_themes=[f"{platform} analysis failed"],
                    explanation=f"Could not analyze {platform} due to technical issues",
                    confidence_score=0.0
                )
                sentiment_results.append(asdict(fallback_sentiment))
                continue
        
        if not sentiment_results:
            raise ValueError("No sentiment analysis results obtained")
        
        logger.info(f"✅ Sentiment analysis completed: {len(sentiment_results)} platforms analyzed")
        
        return {
            **state,
            "sentiment_analysis": sentiment_results,
            "current_step": "sentiment_analysis_complete",
            "execution_metadata": {
                **state.get("execution_metadata", {}),
                "sentiment_analysis_timestamp": datetime.now().isoformat(),
                "platforms_analyzed": len(sentiment_results)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Sentiment analysis failed: {e}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Sentiment analysis failed: {str(e)}"],
            "current_step": "error",
            "retry_count": state.get("retry_count", 0) + 1
        }

async def strategy_generator_agent(state: ProductLaunchState, tools: AgentTools) -> ProductLaunchState:
    """Generate launch strategy based on competitive and sentiment analysis"""
    logger.info("🎯 Starting strategy generation...")
    
    try:
        product_info = state["product_info"]
        competitive_insights = state.get("competitive_insights", [])
        sentiment_analysis = state.get("sentiment_analysis", [])
        
        context = json.dumps({
            "product": product_info,
            "competitors": competitive_insights,
            "sentiment": sentiment_analysis
        })
        
        # Generate launch strategy
        strategy_prompt = f"""Based on this comprehensive market data:\n{context}\n\n
        Create a tailored launch strategy for '{product_info['name']}'. 
        
        Respond ONLY with a valid JSON object with EXACTLY these fields:
        {{
            "positioning_statement": "Clear value proposition statement",
            "key_differentiators": ["differentiator1", "differentiator2", "differentiator3"],
            "target_audience": "Detailed target audience description",
            "pricing_recommendation": "Pricing strategy and rationale",
            "launch_timeline": ["phase1", "phase2", "phase3"],
            "risk_mitigation": ["risk1_mitigation", "risk2_mitigation"],
            "success_metrics": ["metric1", "metric2", "metric3"]
        }}
        
        Do NOT include any other fields."""
        
        strategy_response = await tools.call_groq(
            strategy_prompt,
            "You are a senior product marketing strategist. Create data-driven strategies. Output only valid JSON with exact fields requested.",
            json_mode=True
        )
        
        strategy_data = json.loads(strategy_response)
        
        # Validate and clean the response data
        cleaned_strategy = {
            "positioning_statement": strategy_data.get("positioning_statement", "Innovative product positioning"),
            "key_differentiators": strategy_data.get("key_differentiators", ["Unique features"]),
            "target_audience": strategy_data.get("target_audience", "General market"),
            "pricing_recommendation": strategy_data.get("pricing_recommendation", "Competitive pricing"),
            "launch_timeline": strategy_data.get("launch_timeline", ["Phase 1: Preparation"]),
            "risk_mitigation": strategy_data.get("risk_mitigation", ["Monitor market response"]),
            "success_metrics": strategy_data.get("success_metrics", ["User adoption"])
        }
        
        launch_strategy = LaunchStrategy(**cleaned_strategy)
        
        # Generate FAQ
        faq_prompt = f"""Based on this market data:\n{context}\n\n
        Generate 5 insightful FAQs that address common concerns about '{product_info['name']}' based on competitive landscape and market sentiment.
        
        Respond ONLY with a valid JSON object with EXACTLY this structure:
        {{
            "faq": [
                {{"question": "What makes this different from competitor X?", "answer": "Detailed answer"}},
                {{"question": "How does pricing compare?", "answer": "Detailed answer"}},
                {{"question": "Question 3?", "answer": "Answer 3"}},
                {{"question": "Question 4?", "answer": "Answer 4"}},
                {{"question": "Question 5?", "answer": "Answer 5"}}
            ]
        }}
        
        Do NOT include any other fields."""
        
        faq_response = await tools.call_groq(
            faq_prompt,
            "You are a customer success expert. Create helpful FAQs. Output only valid JSON with exact structure requested.",
            json_mode=True
        )
        
        faq_data = json.loads(faq_response)
        
        strategy_result = {
            "launch_strategy": asdict(launch_strategy),
            "faq": faq_data.get("faq", [])
        }
        
        logger.info("✅ Strategy generation completed")
        
        return {
            **state,
            "launch_strategy": strategy_result,
            "current_step": "strategy_complete",
            "execution_metadata": {
                **state.get("execution_metadata", {}),
                "strategy_generation_timestamp": datetime.now().isoformat(),
                "faq_items_generated": len(faq_data.get("faq", []))
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Strategy generation failed: {e}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Strategy generation failed: {str(e)}"],
            "current_step": "error",
            "retry_count": state.get("retry_count", 0) + 1
        }

async def content_creator_agent(state: ProductLaunchState, tools: AgentTools) -> ProductLaunchState:
    """Create social media content with AI-generated images"""
    logger.info("✍️ Starting content creation...")
    
    try:
        product_info = state["product_info"]
        sentiment_data = state.get("sentiment_analysis", [])
        launch_strategy = state.get("launch_strategy", {})
        
        if not sentiment_data:
            # Create default platforms if no sentiment data
            sentiment_data = [
                {"platform": "Twitter", "overall_sentiment": 0.1},
                {"platform": "LinkedIn", "overall_sentiment": 0.1}
            ]
        
        # Select platforms with reasonable sentiment (>-0.2)
        viable_platforms = [s for s in sentiment_data if s.get("overall_sentiment", 0) > -0.2]
        if not viable_platforms:
            logger.warning("No platforms with positive sentiment. Using all available platforms.")
            viable_platforms = sentiment_data
        
        social_posts = []
        
        for platform_data in viable_platforms:
            platform_name = platform_data.get("platform")
            logger.info(f"Creating content for {platform_name}...")
            
            context = json.dumps({
                "product": product_info,
                "platform_sentiment": platform_data,
                "overall_strategy": launch_strategy.get("launch_strategy", {})
            })
            
            # Create post content
            post_prompt = f"""Based on this data:\n{context}\n\n
            Create an engaging social media post for '{platform_name}' to launch '{product_info['name']}'. 
            
            For the image_prompt, create a HIGHLY REALISTIC, professional product photography prompt.
            
            Respond ONLY with a valid JSON object with EXACTLY these fields:
            {{
                "post_type": "Text & Image",
                "text_content": "Engaging post text tailored for {platform_name} audience and style",
                "image_prompt": "Detailed realistic product photography prompt with professional lighting and composition",
                "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3", "#hashtag4", "#hashtag5"]
            }}
            
            Do NOT include any other fields."""
            
            post_response = await tools.call_groq(
                post_prompt,
                f"You are a viral social media expert and professional photographer for {platform_name}. Output only valid JSON with exact fields requested.",
                json_mode=True
            )
            
            post_data = json.loads(post_response)
            
            # Validate and clean the response data
            cleaned_post = {
                "post_type": post_data.get("post_type", "Text & Image"),
                "text_content": post_data.get("text_content", f"Check out {product_info['name']}!"),
                "image_prompt": post_data.get("image_prompt", f"Professional photo of {product_info['name']}"),
                "hashtags": post_data.get("hashtags", [f"#{product_info['name'].replace(' ', '')}", f"#{product_info['category']}"])
            }
            
            # Generate image if possible
            image_url = None
            image_prompt = cleaned_post["image_prompt"]
            if image_prompt and tools.hf_client:
                try:
                    image_url = await generate_product_image(image_prompt, product_info, tools.hf_client)
                except Exception as img_error:
                    logger.warning(f"Image generation failed for {platform_name}: {img_error}")
            
            social_post = SocialMediaPost(
                platform=platform_name,
                post_type=cleaned_post["post_type"],
                text_content=cleaned_post["text_content"],
                image_prompt=image_prompt,
                hashtags=cleaned_post["hashtags"],
                image_url=image_url
            )
            
            social_posts.append(asdict(social_post))
        
        logger.info(f"✅ Content creation completed: {len(social_posts)} posts created")
        
        return {
            **state,
            "social_media_posts": social_posts,
            "current_step": "complete",
            "execution_metadata": {
                **state.get("execution_metadata", {}),
                "content_creation_timestamp": datetime.now().isoformat(),
                "posts_created": len(social_posts)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Content creation failed: {e}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Content creation failed: {str(e)}"],
            "current_step": "error",
            "retry_count": state.get("retry_count", 0) + 1
        }

async def generate_product_image(prompt: str, product_info: Dict, hf_client: InferenceClient) -> Optional[str]:
    """Generate realistic product images using Hugging Face models"""
    logger.info("🎨 Generating product image...")
    
    # Enhanced realistic prompts
    realistic_enhancements = [
        "photorealistic, professional product photography",
        "shot with Canon EOS R5, 85mm lens, f/2.8",
        "soft box lighting, clean white background",
        "high resolution, ultra detailed, sharp focus",
        "commercial photography style, studio lighting",
        "shallow depth of field, bokeh background",
        "professional color grading, vibrant colors",
        "product showcase, marketing photography"
    ]
    
    enhanced_prompt = f"{prompt}, {', '.join(realistic_enhancements[:4])}"
    
    # Try multiple models for best results
    models = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-2-1",
        "runwayml/stable-diffusion-v1-5"
    ]
    
    for model in models:
        try:
            logger.info(f"Trying model: {model}")
            
            image: Image.Image = await asyncio.to_thread(
                hf_client.text_to_image,
                prompt=enhanced_prompt,
                model=model
            )
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            data_uri = f"data:image/png;base64,{img_str}"
            
            logger.info(f"✅ Image generated successfully using {model}")
            return data_uri
            
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")
            continue
    
    logger.error("❌ All image generation attempts failed")
    return None

# --- Error Handling Agent ---
async def handle_error(state: ProductLaunchState) -> ProductLaunchState:
    """Handle errors and implement retry logic"""
    errors = state.get("errors", [])
    retry_count = state.get("retry_count", 0)
    max_retries = 2
    
    logger.error(f"🚨 Workflow error occurred. Retry count: {retry_count}/{max_retries}")
    logger.error(f"Errors: {errors}")
    
    if retry_count < max_retries:
        # Implement retry logic - could retry specific failed steps
        logger.info("🔄 Attempting retry...")
        return {
            **state,
            "current_step": "retry",
            "errors": []  # Clear errors for retry
        }
    else:
        logger.error("❌ Max retries exceeded. Workflow failed.")
        return {
            **state,
            "current_step": "failed",
            "execution_metadata": {
                **state.get("execution_metadata", {}),
                "final_status": "failed",
                "failure_timestamp": datetime.now().isoformat(),
                "total_errors": len(errors)
            }
        }

# --- Workflow Routing Logic ---
def determine_next_step(state: ProductLaunchState) -> str:
    """Determine the next step in the workflow based on current state"""
    current_step = state.get("current_step", "start")
    
    if current_step == "error":
        return "handle_error"
    elif current_step == "retry":
        return "competitive_analysis"  # Restart from beginning on retry
    elif current_step == "competitive_analysis_complete":
        return "sentiment_analysis"
    elif current_step == "sentiment_analysis_complete":
        return "strategy_generation"
    elif current_step == "strategy_complete":
        return "content_creation"
    elif current_step == "complete":
        return END
    elif current_step == "failed":
        return END
    else:
        return "competitive_analysis"

# --- LangGraph Workflow Builder ---
class ProductLaunchOrchestrator:
    def __init__(self, groq_api_key: str, tavily_api_key: str, hf_token: str):
        self.tools = AgentTools(groq_api_key, tavily_api_key, hf_token)
        self.checkpointer = MemorySaver()  # For state persistence
        self.workflow = self._build_workflow()
        
        logger.info("✅ ProductLaunchOrchestrator initialized with LangGraph")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ProductLaunchState)
        
        # Add agent nodes
        workflow.add_node("competitive_analysis", self._competitive_analysis_node)
        workflow.add_node("sentiment_analysis", self._sentiment_analysis_node)
        workflow.add_node("strategy_generation", self._strategy_generation_node)
        workflow.add_node("content_creation", self._content_creation_node)
        workflow.add_node("handle_error", self._error_handling_node)
        
        # Set entry point
        workflow.set_entry_point("competitive_analysis")
        
        # Add conditional edges for smart routing
        workflow.add_conditional_edges(
            "competitive_analysis",
            determine_next_step,
            {
                "sentiment_analysis": "sentiment_analysis",
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "sentiment_analysis",
            determine_next_step,
            {
                "strategy_generation": "strategy_generation",
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "strategy_generation",
            determine_next_step,
            {
                "content_creation": "content_creation",
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "content_creation",
            determine_next_step,
            {
                END: END,
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_error",
            determine_next_step,
            {
                "competitive_analysis": "competitive_analysis",
                END: END
            }
        )
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    # Node wrapper methods
    async def _competitive_analysis_node(self, state: ProductLaunchState) -> ProductLaunchState:
        return await competitive_research_agent(state, self.tools)
    
    async def _sentiment_analysis_node(self, state: ProductLaunchState) -> ProductLaunchState:
        return await sentiment_analysis_agent(state, self.tools)
    
    async def _strategy_generation_node(self, state: ProductLaunchState) -> ProductLaunchState:
        return await strategy_generator_agent(state, self.tools)
    
    async def _content_creation_node(self, state: ProductLaunchState) -> ProductLaunchState:
        return await content_creator_agent(state, self.tools)
    
    async def _error_handling_node(self, state: ProductLaunchState) -> ProductLaunchState:
        return await handle_error(state)
    
    async def analyze_product_launch(self, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete product launch analysis workflow"""
        logger.info("🚀 Starting LangGraph-powered product launch analysis")
        
        # Create initial state
        initial_state = ProductLaunchState(
            product_info=product_info,
            competitive_insights=[],
            sentiment_analysis=[],
            launch_strategy=None,
            social_media_posts=[],
            current_step="start",
            errors=[],
            retry_count=0,
            execution_metadata={
                "started_at": datetime.now().isoformat(),
                "workflow_version": "1.0.0"
            }
        )
        
        # Execute workflow with state management
        config = {"configurable": {"thread_id": f"product_launch_{datetime.now().timestamp()}"}}
        
        try:
            final_state = await self.workflow.ainvoke(initial_state, config)
            
            # Return formatted results
            return {
                "product_info": final_state["product_info"],
                "competitive_analysis": final_state.get("competitive_insights", []),
                "sentiment_analysis": final_state.get("sentiment_analysis", []),
                "launch_strategy": final_state.get("launch_strategy"),
                "social_media_posts": final_state.get("social_media_posts", []),
                "execution_metadata": final_state.get("execution_metadata", {}),
                "workflow_status": final_state.get("current_step", "unknown"),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            return {
                "error": str(e),
                "workflow_status": "failed",
                "generated_at": datetime.now().isoformat()
            }

# --- Main Function for Testing ---
async def main():
    """Test the LangGraph implementation"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not all([groq_api_key, tavily_api_key, hf_token]):
        logger.error("❌ Missing required API keys")
        return
    
    orchestrator = ProductLaunchOrchestrator(groq_api_key, tavily_api_key, hf_token)
    
    product_info = {
        "name": "TaskFlow Pro",
        "category": "productivity",
        "description": "AI-powered task management and workflow automation platform for modern teams."
    }
    
    print("🚀 Starting LangGraph Product Launch Intelligence Analysis...")
    results = await orchestrator.analyze_product_launch(product_info)
    
    if "error" not in results:
        print("\n📊 ANALYSIS RESULTS")
        print(json.dumps(results, indent=2, default=str))
        
        filename = f"langgraph_launch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {filename}")
    else:
        print(f"❌ Analysis failed: {results['error']}")

if __name__ == "__main__":
    asyncio.run(main())
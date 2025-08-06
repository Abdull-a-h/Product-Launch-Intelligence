# app.py - Fixed LangGraph Streamlit Application
import streamlit as st
import asyncio
import os
import base64
from datetime import datetime
from dotenv import load_dotenv
import json
import logging

# Fix for asyncio in Streamlit
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Load environment variables
load_dotenv()

# Import our fixed LangGraph implementation
from product_agent import ProductLaunchOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Async Helper Functions ---
def run_async(coro):
    """Helper function to run async code in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"Async execution error: {e}")
        raise

async def run_analysis_with_progress(orchestrator, product_info, progress_bar, status_text):
    """Run analysis with progress updates"""
    
    # This is a simplified progress tracking - in a real implementation,
    # you'd need to modify the orchestrator to emit progress events
    
    status_text.text("📊 Analyzing sentiment across platforms...")
    progress_bar.progress(50)
    await asyncio.sleep(0.1)  # Small delay for UI update
    
    status_text.text("🎯 Generating launch strategy...")
    progress_bar.progress(75)
    await asyncio.sleep(0.1)  # Small delay for UI update
    
    status_text.text("✍️ Creating social media content...")
    progress_bar.progress(90)
    
    # Run the actual analysis
    results = await orchestrator.analyze_product_launch(product_info)
    
    return results

# --- Page Configuration and Enhanced CSS ---
st.set_page_config(
    page_title="Launch Intelligence AI - LangGraph Edition", 
    page_icon="🚀", 
    layout="wide"
)

# Enhanced CSS for better UI
enhanced_css = """
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.post-card { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px; 
    padding: 20px; 
    margin-bottom: 20px; 
    border: 1px solid #3c4048;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.post-card h5 { 
    color: #ffffff; 
    border-bottom: 2px solid #00c497; 
    padding-bottom: 8px; 
    margin-top: 0;
    font-weight: 600;
}

.post-card p { 
    color: #f0f0f0; 
    line-height: 1.6;
}

.post-card .hashtags { 
    color: #00c497; 
    font-weight: bold;
    background-color: rgba(0, 196, 151, 0.1);
    padding: 8px;
    border-radius: 8px;
    margin-top: 10px;
}

.metric-container {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 4px solid #667eea;
}

.competitor-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #ff6b6b;
}

.strategy-card {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.workflow-status {
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
    font-weight: bold;
}

.status-complete {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.status-error {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.status-processing {
    background-color: #fff3cd;
    color: #856404;
    border: 1px solid #ffeaa7;
}

.execution-metadata {
    background-color: #e9ecef;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
    font-family: monospace;
    font-size: 0.9em;
}

.success-banner {
    background: linear-gradient(135deg, #52c234 0%, #61b15a 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
    text-align: center;
    font-weight: bold;
}

.error-banner {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
    text-align: center;
    font-weight: bold;
}
</style>
"""

st.markdown(enhanced_css, unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🚀 Product Launch Agents</h1>
    <h3>LangGraph-Powered Multi-Agent Product Analysis</h3>
    <p>Advanced workflow orchestration with state management, error handling, and retry logic</p>
</div>
""", unsafe_allow_html=True)

# --- API Keys Status in Sidebar ---
st.sidebar.title("🔧 System Configuration")

groq_key_loaded = os.getenv("GROQ_API_KEY") is not None
tavily_key_loaded = os.getenv("TAVILY_API_KEY") is not None
hf_key_loaded = os.getenv("HUGGING_FACE_HUB_TOKEN") is not None



if not all([groq_key_loaded, tavily_key_loaded, hf_key_loaded]):
    st.sidebar.error("⚠️ Some API keys are missing! Please check your .env file.")
    st.sidebar.info("Required keys: GROQ_API_KEY, TAVILY_API_KEY, HUGGING_FACE_HUB_TOKEN")

st.sidebar.subheader("🏗️ LangGraph Features")
st.sidebar.info("""
- **State Management**: Persistent workflow state
- **Error Handling**: Automatic retry logic with data validation
- **Conditional Routing**: Smart workflow decisions
- **Observability**: Detailed execution metadata
- **Memory Persistence**: Conversation checkpointing
""")



# --- Main Application ---
st.subheader("📝 Product Information")
st.markdown("Provide details about your product for comprehensive launch analysis:")

with st.form("product_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        product_name = st.text_input(
            "Product Name", 
            "Zenith Sparkling Water",
            help="The name of your product"
        )
        product_category = st.text_input(
            "Product Category", 
            "beverage",
            help="Industry category (e.g., beverage, software, fashion)"
        )
    
    with col2:
        product_description = st.text_area(
            "Product Description", 
            "A new line of premium sparkling water infused with natural botanical extracts.",
            height=100,
            help="Detailed description of your product's features and benefits"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        enable_image_generation = st.checkbox("Enable AI Image Generation", value=True, help="Generate product images for social media posts")
        max_retries = st.slider("Max Retry Attempts", 1, 5, 2, help="Number of retry attempts if agents fail")
        
    submitted = st.form_submit_button(
        "🚀 Launch Fixed LangGraph Analysis", 
        type="primary", 
        use_container_width=True
    )

# --- Analysis Execution with Fixed Async Handling ---
if submitted:
    if not all([groq_key_loaded, tavily_key_loaded, hf_key_loaded]):
        st.markdown('<div class="error-banner">❌ Missing required API keys. Please check your .env file.</div>', unsafe_allow_html=True)
    else:
        # Initialize the LangGraph orchestrator
        groq_api_key = os.getenv("GROQ_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        # Show progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🤖 LangGraph agents are collaborating with improved error handling..."):
            try:
                status_text.text("🔧 Initializing LangGraph orchestrator...")
                progress_bar.progress(10)
                
                orchestrator = ProductLaunchOrchestrator(groq_api_key, tavily_api_key, hf_token)
                
                product_info = {
                    "name": product_name,
                    "category": product_category,
                    "description": product_description,
                    "enable_images": enable_image_generation,
                    "max_retries": max_retries
                }
                
                status_text.text("🔍 Running competitive analysis...")
                progress_bar.progress(25)
                
                # Run the analysis with progress updates using the fixed async wrapper
                results = run_async(run_analysis_with_progress(orchestrator, product_info, progress_bar, status_text))
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis completed successfully!")
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.analysis_complete = True
                
                # Show completion status
                workflow_status = results.get("workflow_status", "unknown")
                if workflow_status == "failed":
                    st.markdown('<div class="error-banner">❌ Analysis failed. Check error details below.</div>', unsafe_allow_html=True)

                
            except Exception as e:
                st.markdown(f'<div class="error-banner">❌ System error: {str(e)}</div>', unsafe_allow_html=True)
                logger.error(f"Application error: {e}", exc_info=True)

# --- Enhanced Display Functions ---
def display_execution_metadata(metadata):
    """Display workflow execution metadata with enhanced styling"""
    st.subheader("🔍 Workflow Execution Details")
    
    with st.expander("📊 Execution Metadata & Performance", expanded=False):
        st.markdown('<div class="execution-metadata">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🚀 Workflow Information:**")
            st.write(f"• Started: {metadata.get('started_at', 'N/A')}")
            st.write(f"• Version: {metadata.get('workflow_version', 'N/A')}")
            st.write(f"• Status: {metadata.get('final_status', 'N/A')}")
            
            # Add execution time if available
            started = metadata.get('started_at')
            if started:
                try:
                    start_time = datetime.fromisoformat(started.replace('Z', '+00:00'))
                    duration = datetime.now() - start_time.replace(tzinfo=None)
                    st.write(f"• Duration: {duration.seconds}s")
                except:
                    pass
        
        with col2:
            st.markdown("**🤖 Agent Performance:**")
            st.write(f"• Competitors Found: {metadata.get('competitors_found', 'N/A')}")
            st.write(f"• Platforms Analyzed: {metadata.get('platforms_analyzed', 'N/A')}")
            st.write(f"• Posts Created: {metadata.get('posts_created', 'N/A')}")
            st.write(f"• FAQ Items: {metadata.get('faq_items_generated', 'N/A')}")
        
        with col3:
            st.markdown("**📈 Quality Metrics:**")
            
            # Calculate quality score based on results
            quality_score = 0
            if metadata.get('competitors_found', 0) > 0:
                quality_score += 25
            if metadata.get('platforms_analyzed', 0) > 0:
                quality_score += 25
            if metadata.get('posts_created', 0) > 0:
                quality_score += 25
            if metadata.get('faq_items_generated', 0) > 0:
                quality_score += 25
            
            st.write(f"• Quality Score: {quality_score}%")
            st.write(f"• Error Count: {metadata.get('total_errors', 0)}")
            
            if quality_score >= 75:
                st.success("🌟 Excellent Results")
            elif quality_score >= 50:
                st.info("👍 Good Results")
            else:
                st.warning("⚠️ Partial Results")
        
        if metadata.get('total_errors', 0) > 0:
            st.markdown("**🚨 Error Information:**")
            st.error(f"• Total Errors: {metadata.get('total_errors', 0)}")
            if metadata.get('failure_timestamp'):
                st.write(f"• Failure Time: {metadata.get('failure_timestamp', 'N/A')}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_competitive_analysis(analysis):
    """Enhanced competitive analysis display with better validation"""
    st.subheader("🏢 Competitive Landscape Analysis")
    
    if not isinstance(analysis, list) or not analysis:
        st.warning("⚠️ No competitive analysis data was generated. This might be due to API limitations or connectivity issues.")
        st.info("💡 Try running the analysis again or check your API keys.")
        return
    
    st.success(f"📊 Successfully analyzed {len(analysis)} competitors in the market")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_strengths = sum(len(comp.get('strengths', [])) for comp in analysis)
        st.metric("💪 Total Strengths Identified", total_strengths)
    with col2:
        total_weaknesses = sum(len(comp.get('weaknesses', [])) for comp in analysis)
        st.metric("🔍 Total Weaknesses Found", total_weaknesses)
    with col3:
        avg_features = sum(len(comp.get('product_features', [])) for comp in analysis) / len(analysis) if analysis else 0
        st.metric("📋 Avg Features per Competitor", f"{avg_features:.1f}")
    
    for i, competitor in enumerate(analysis):
        competitor_name = competitor.get('competitor', f'Competitor {i+1}')
        market_position = competitor.get('market_position', 'Position unknown')
        
        with st.expander(f"**{i+1}. {competitor_name}** - *{market_position}*", expanded=i==0):
            # Create competitor card
            st.markdown(f'<div class="competitor-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🟢 Strengths")
                strengths = competitor.get('strengths', [])
                if strengths:
                    for strength in strengths:
                        st.markdown(f"• {strength}")
                else:
                    st.markdown("*No strengths identified*")
                
                st.markdown("##### 📋 Product Features")
                features = competitor.get('product_features', [])
                if features:
                    for feature in features:
                        st.markdown(f"• {feature}")
                else:
                    st.markdown("*No features identified*")
            
            with col2:
                st.markdown("##### 🔴 Weaknesses")
                weaknesses = competitor.get('weaknesses', [])
                if weaknesses:
                    for weakness in weaknesses:
                        st.markdown(f"• {weakness}")
                else:
                    st.markdown("*No weaknesses identified*")
                
                st.markdown("##### 💰 Pricing")
                pricing = competitor.get('pricing', 'N/A')
                if pricing and pricing != 'N/A':
                    st.info(f"💵 {pricing}")
                else:
                    st.info("💵 Pricing information not available")
            
            st.markdown('</div>', unsafe_allow_html=True)

def display_sentiment_analysis(analysis):
    """Enhanced sentiment analysis with better error handling"""
    st.subheader("📱 Real-Time Market Sentiment Analysis")
    
    if not isinstance(analysis, list) or not analysis:
        st.warning("⚠️ No sentiment analysis data was generated.")
        st.info("💡 This could be due to search API limitations or network issues. Try again later.")
        return
    
    # Filter out failed analyses
    valid_analyses = [a for a in analysis if a.get('confidence_score', 0) > 0]
    
    if valid_analyses:
        st.success(f"📊 Successfully analyzed sentiment across {len(valid_analyses)} platforms using real-time search")
    else:
        st.warning(f"⚠️ Analyzed {len(analysis)} platforms but with limited confidence in results")
    
    # Create metrics row
    cols = st.columns(len(analysis))
    
    for i, platform_data in enumerate(analysis):
        with cols[i]:
            sentiment_score = platform_data.get("overall_sentiment", 0)
            confidence = platform_data.get("confidence_score", 0)
            platform_name = platform_data.get('platform', 'Unknown')
            
            # Enhanced emoji and color logic
            if sentiment_score > 0.3:
                emoji = "😊"
                delta_color = "normal"
            elif sentiment_score > 0:
                emoji = "🙂"
                delta_color = "normal"
            elif sentiment_score > -0.3:
                emoji = "😐"
                delta_color = "normal"
            else:
                emoji = "😟"
                delta_color = "inverse"
            
            # Display metrics with confidence indicator
            confidence_text = f"Confidence: {confidence:.1%}" if confidence > 0 else "Low confidence"
            st.metric(
                label=f"{platform_name} {emoji}", 
                value=f"{sentiment_score:.2f}",
                delta=confidence_text
            )
            
            # Add confidence warning for low-confidence results
            if confidence < 0.3:
                st.caption("⚠️ Limited data")
    
    # Detailed platform insights
    for platform_data in analysis:
        platform_name = platform_data.get('platform', 'Unknown')
        confidence = platform_data.get('confidence_score', 0)
        
        with st.expander(f"📊 {platform_name} Deep Insights", expanded=False):
            st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
            
            # Show confidence level
            if confidence > 0.7:
                st.success(f"🎯 High confidence analysis ({confidence:.1%})")
            elif confidence > 0.3:
                st.info(f"📊 Moderate confidence analysis ({confidence:.1%})")
            else:
                st.warning(f"⚠️ Low confidence analysis ({confidence:.1%}) - Limited data available")
            
            # Explanation
            explanation = platform_data.get('explanation', 'No explanation available')
            st.markdown(f"**🔍 Analysis:** {explanation}")
            
            # Mention breakdown
            pos_mentions = platform_data.get('positive_mentions', 0)
            neg_mentions = platform_data.get('negative_mentions', 0)
            neu_mentions = platform_data.get('neutral_mentions', 0)
            total_mentions = pos_mentions + neg_mentions + neu_mentions
            
            if total_mentions > 0:
                st.markdown("**📈 Mention Breakdown:**")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("✅ Positive", pos_mentions, delta=f"{pos_mentions/total_mentions:.1%}")
                with metric_col2:
                    st.metric("➖ Neutral", neu_mentions, delta=f"{neu_mentions/total_mentions:.1%}")
                with metric_col3:
                    st.metric("❌ Negative", neg_mentions, delta=f"{neg_mentions/total_mentions:.1%}")
            else:
                st.info("📊 No specific mention counts available")
            
            # Key themes with better visualization
            themes = platform_data.get('key_themes', [])
            if themes and themes != [f"{platform_name} themes"]:
                st.markdown(f"**🏷️ Key Themes on {platform_name}:**")
                
                theme_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3"]
                theme_html = ""
                
                for idx, theme in enumerate(themes):
                    if theme and theme.strip():  # Only show non-empty themes
                        color = theme_colors[idx % len(theme_colors)]
                        theme_html += f'<span style="background-color: {color}; color: white; padding: 6px 12px; margin: 4px; border-radius: 15px; font-size: 14px; display: inline-block; font-weight: 500;">{theme}</span> '
                
                if theme_html:
                    st.markdown(theme_html, unsafe_allow_html=True)
                else:
                    st.markdown("*No specific themes identified for this platform.*")
            else:
                st.markdown("*No specific themes identified for this platform.*")
            
            st.markdown('</div>', unsafe_allow_html=True)

def display_launch_strategy(strategy_data):
    """Enhanced launch strategy display with validation"""
    st.subheader("🎯 AI-Generated Launch Strategy")
    
    if not isinstance(strategy_data, dict) or "launch_strategy" not in strategy_data:
        st.warning("⚠️ No launch strategy was generated.")
        st.info("💡 This might be due to incomplete analysis data. Try running the analysis again.")
        return
    
    strategy = strategy_data["launch_strategy"]
    faq_data = strategy_data.get("faq", [])
    
    # Positioning statement highlight
    positioning = strategy.get('positioning_statement', 'N/A')
    if positioning and positioning != 'N/A':
        st.markdown(f'<div class="strategy-card">', unsafe_allow_html=True)
        st.markdown("### 🎪 Strategic Positioning Statement")
        st.success(f"💡 {positioning}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategy quality indicator
    strategy_completeness = 0
    total_fields = 7
    
    for field in ['key_differentiators', 'target_audience', 'pricing_recommendation', 'launch_timeline', 'risk_mitigation', 'success_metrics']:
        if strategy.get(field) and strategy[field] not in ['N/A', 'Unknown', '']:
            strategy_completeness += 1
    
    completeness_percent = (strategy_completeness / total_fields) * 100
    
    if completeness_percent >= 80:
        st.success(f"🌟 Comprehensive strategy generated ({completeness_percent:.0f}% complete)")
    elif completeness_percent >= 60:
        st.info(f"👍 Good strategy generated ({completeness_percent:.0f}% complete)")
    else:
        st.warning(f"⚠️ Partial strategy generated ({completeness_percent:.0f}% complete)")
    
    # Two-column layout for strategy details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔑 Key Differentiators")
        differentiators = strategy.get('key_differentiators', [])
        if differentiators and differentiators != ['Unique features']:
            for diff in differentiators:
                st.markdown(f"• ⭐ {diff}")
        else:
            st.markdown("*No specific differentiators identified*")
        
        st.markdown("#### 🎯 Target Audience")
        target_audience = strategy.get('target_audience', 'N/A')
        if target_audience and target_audience not in ['N/A', 'General market']:
            st.info(f"👥 {target_audience}")
        else:
            st.markdown("*Target audience analysis incomplete*")
        
        st.markdown("#### 📊 Success Metrics")
        metrics = strategy.get('success_metrics', [])
        if metrics and metrics != ['User adoption']:
            for metric in metrics:
                st.markdown(f"• 📈 {metric}")
        else:
            st.markdown("*No specific metrics defined*")
    
    with col2:
        st.markdown("#### 💰 Pricing Recommendation")
        pricing = strategy.get('pricing_recommendation', 'N/A')
        if pricing and pricing not in ['N/A', 'Competitive pricing']:
            st.info(f"💵 {pricing}")
        else:
            st.markdown("*Pricing strategy needs refinement*")
        
        st.markdown("#### ⚠️ Risk Mitigation")
        risks = strategy.get('risk_mitigation', [])
        if risks and risks != ['Monitor market response']:
            for risk in risks:
                st.markdown(f"• 🛡️ {risk}")
        else:
            st.markdown("*No specific risks identified*")
    
    # Launch timeline
    with st.expander("📅 Proposed Launch Timeline", expanded=False):
        timeline = strategy.get('launch_timeline', [])
        if timeline and timeline != ['Phase 1: Preparation']:
            for i, phase in enumerate(timeline, 1):
                st.markdown(f"**Phase {i}:** {phase}")
        else:
            st.markdown("*Timeline needs to be developed based on more specific requirements*")
    
    # FAQ section with enhanced validation
    with st.expander("❓ AI-Generated FAQ", expanded=False):
        if faq_data and isinstance(faq_data, list) and len(faq_data) > 0:
            st.success(f"📝 Generated {len(faq_data)} FAQ items based on comprehensive market analysis")
            for i, faq_item in enumerate(faq_data, 1):
                if isinstance(faq_item, dict):
                    question = faq_item.get('question', 'No question')
                    answer = faq_item.get('answer', 'No answer')
                    
                    if question != 'No question' and answer != 'No answer':
                        st.markdown(f"**Q{i}: {question}**")
                        st.markdown(f"**A:** {answer}")
                        st.markdown("---")
                    else:
                        st.markdown(f"**Q{i}:** *FAQ item incomplete*")
        else:
            st.info("*FAQ generation was skipped or failed. This could be improved with more market data.*")

def display_social_media_posts(posts):
    """Enhanced social media content display with better error handling"""
    st.subheader("✍️ AI-Generated Social Media Content")
    
    if not isinstance(posts, list) or not posts:
        st.warning("⚠️ No social media content was generated.")
        st.info("💡 This could be due to sentiment analysis issues or image generation failures. Try running the analysis again.")
        return
    
    # Filter out empty or invalid posts
    valid_posts = [post for post in posts if isinstance(post, dict) and post.get('text_content')]
    
    if valid_posts:
        st.success(f"📱 Successfully generated {len(valid_posts)} platform-specific posts")
        
        # Show image generation success rate
        posts_with_images = len([post for post in valid_posts if post.get('image_url')])
        if posts_with_images > 0:
            st.info(f"🖼️ {posts_with_images}/{len(valid_posts)} posts include AI-generated images")
        else:
            st.warning("🖼️ Image generation was not successful for any posts")
    else:
        st.error("❌ No valid social media posts were generated")
        return
    
    for i, post in enumerate(valid_posts):
        platform_name = post.get('platform', 'Unknown Platform')
        
        # Enhanced post card
        st.markdown(f"""
        <div class='post-card'>
            <h5>📱 {platform_name} Post #{i+1}</h5>
        </div>
        """, unsafe_allow_html=True)
        
        # Layout: Image and content side by side
        img_col, content_col = st.columns([1.2, 1])
        
        with img_col:
            # Display AI-generated image
            image_uri = post.get("image_url")
            if image_uri and image_uri.startswith("data:image"):
                try:
                    st.image(
                        image_uri, 
                        caption=f"AI-Generated Visual for {platform_name}",
                        use_container_width=True
                    )
                    
                    # Download button
                    img_data_b64 = image_uri.split(",")[1]
                    img_bytes = base64.b64decode(img_data_b64)
                    st.download_button(
                        label=f"⬇️ Download {platform_name} Image", 
                        data=img_bytes,
                        file_name=f"social_post_{platform_name.lower().replace(' ', '_')}_{i+1}.png", 
                        mime="image/png",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Image display error: {e}")
                    st.info("🖼️ Image data corrupted or invalid format")
            else:
                st.info("🖼️ No image was generated for this post")
                st.markdown("**Possible reasons:**")
                st.markdown("• Image generation API was unavailable")
                st.markdown("• Rate limits exceeded")
                st.markdown("• Content policy restrictions")
        
        with content_col:
            # Post content with validation
            st.markdown("**📝 Post Content:**")
            post_content = post.get('text_content', 'No content generated')
            if post_content and post_content != 'No content generated':
                st.write(post_content)
                
                # Content quality indicator
                word_count = len(post_content.split())
                if word_count > 20:
                    st.success(f"✅ Rich content ({word_count} words)")
                elif word_count > 10:
                    st.info(f"📝 Good content ({word_count} words)")
                else:
                    st.warning(f"⚠️ Brief content ({word_count} words)")
            else:
                st.error("❌ No content was generated")
            
            # Post type
            post_type = post.get('post_type', 'Unknown')
            st.markdown(f"**📋 Type:** {post_type}")
            
            # Hashtags with enhanced styling and validation
            hashtags = post.get('hashtags', [])
            if hashtags and isinstance(hashtags, list) and len(hashtags) > 0:
                # Filter out empty hashtags
                valid_hashtags = [tag for tag in hashtags if tag and tag.strip() and tag.startswith('#')]
                
                if valid_hashtags:
                    st.markdown("**#️⃣ Hashtags:**")
                    hashtag_str = " ".join(valid_hashtags)
                    st.markdown(
                        f'<p class="hashtags">{hashtag_str}</p>', 
                        unsafe_allow_html=True
                    )
                    
                    # Hashtag analytics
                    st.caption(f"📊 {len(valid_hashtags)} optimized hashtags for {platform_name}")
                else:
                    st.warning("⚠️ No valid hashtags generated")
            else:
                st.info("🏷️ No hashtags were generated for this post")
        
        # Image generation prompt in expandable section
        with st.expander("🎨 View AI Image Generation Prompt", expanded=False):
            image_prompt = post.get("image_prompt", "No prompt available")
            if image_prompt and image_prompt != "No prompt available":
                st.code(image_prompt, language="text")
                st.caption("This detailed prompt was used to generate the photorealistic product image above.")
                
                # Prompt quality indicator
                prompt_length = len(image_prompt.split())
                if prompt_length > 20:
                    st.success(f"🎯 Detailed prompt ({prompt_length} words)")
                else:
                    st.info(f"📝 Basic prompt ({prompt_length} words)")
            else:
                st.warning("⚠️ No image generation prompt was created")
        
        st.divider()

# --- Results Display with Enhanced Error Handling ---
if "analysis_complete" in st.session_state and st.session_state.analysis_complete:
    st.divider()
    
    results = st.session_state.results
    
    # Display workflow status with enhanced styling
    workflow_status = results.get("workflow_status", "unknown")
    if workflow_status == "failed":
        st.markdown('<div class="error-banner">❌ Analysis Failed - Check Error Details Below</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="workflow-status status-processing">
            🔄 Workflow Status: {workflow_status.upper()}
        </div>
        """, unsafe_allow_html=True)
    
    if "error" in results:
        st.error(f"❌ Analysis failed: {results['error']}")
        st.info("💡 Try running the analysis again or check your API keys and internet connection.")
    else:
        st.header("📊 Enhanced LangGraph Launch Intelligence Report")
        
        # Display execution metadata first
        execution_metadata = results.get("execution_metadata", {})
        if execution_metadata:
            display_execution_metadata(execution_metadata)
        
        # Main analysis results with enhanced tabs
        tabs = st.tabs([
            "🏢 Competitive Analysis", 
            "📱 Sentiment Analysis", 
            "🎯 Launch Strategy", 
            "✍️ Social Content",
            "📋 Executive Summary"
        ])
        
        with tabs[0]:
            display_competitive_analysis(results.get("competitive_analysis", []))
        
        with tabs[1]:
            display_sentiment_analysis(results.get("sentiment_analysis", []))
        
        with tabs[2]:
            display_launch_strategy(results.get("launch_strategy", {}))
        
        with tabs[3]:
            display_social_media_posts(results.get("social_media_posts", []))
        
        with tabs[4]:
            # Executive Summary Tab
            st.subheader("📋 Executive Summary")
            
            # Key insights
            competitive_count = len(results.get("competitive_analysis", []))
            sentiment_count = len(results.get("sentiment_analysis", []))
            social_count = len(results.get("social_media_posts", []))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏢 Competitors Analyzed", competitive_count)
            with col2:
                st.metric("📱 Platforms Monitored", sentiment_count)
            with col3:
                st.metric("✍️ Content Pieces Created", social_count)
            
            # Overall recommendations
            st.markdown("### 🎯 Key Recommendations")
            
            recommendations = []
            
            if competitive_count > 0:
                recommendations.append("✅ **Competitive Analysis Complete** - Use insights to refine positioning")
            else:
                recommendations.append("⚠️ **Competitive Analysis Incomplete** - Rerun analysis for better insights")
            
            if sentiment_count > 0:
                avg_sentiment = sum(s.get('overall_sentiment', 0) for s in results.get("sentiment_analysis", [])) / sentiment_count
                if avg_sentiment > 0.2:
                    recommendations.append("😊 **Positive Market Sentiment** - Good timing for launch")
                elif avg_sentiment > -0.2:
                    recommendations.append("😐 **Neutral Market Sentiment** - Consider targeted marketing")
                else:
                    recommendations.append("😟 **Challenging Market Sentiment** - Review positioning strategy")
            
            if social_count > 0:
                recommendations.append("📱 **Social Content Ready** - Content pieces prepared for launch")
            else:
                recommendations.append("⚠️ **Content Creation Needed** - Generate social media content")
            
            for rec in recommendations:
                st.markdown(rec)
            
            # Next steps
            st.markdown("### 🚀 Recommended Next Steps")
            st.markdown("""
            1. **Review Competitive Insights** - Analyze competitor strengths and weaknesses
            2. **Refine Positioning** - Use sentiment data to adjust messaging
            3. **Prepare Launch Campaign** - Use generated content as starting point
            4. **Monitor Performance** - Set up tracking for success metrics
            5. **Iterate Strategy** - Rerun analysis as market conditions change
            """)
        
        # Enhanced download section
        st.subheader("💾 Export & Share Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare download data
            download_data = json.dumps(results, indent=2, default=str)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.download_button(
                label="📄 Download Complete Analysis (JSON)",
                data=download_data,
                file_name=f"langgraph_launch_analysis_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Prepare summary report
            summary_report = f"""
# Launch Intelligence Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Product Information
- Name: {results.get('product_info', {}).get('name', 'N/A')}
- Category: {results.get('product_info', {}).get('category', 'N/A')}
- Description: {results.get('product_info', {}).get('description', 'N/A')}

## Analysis Summary
- Competitors Analyzed: {len(results.get('competitive_analysis', []))}
- Platforms Monitored: {len(results.get('sentiment_analysis', []))}
- Content Pieces Created: {len(results.get('social_media_posts', []))}
- Workflow Status: {results.get('workflow_status', 'Unknown')}

## Key Findings
{json.dumps(results.get('launch_strategy', {}).get('launch_strategy', {}), indent=2)}

---
Generated by LangGraph Launch Intelligence AI
"""
            
            st.download_button(
                label="📝 Download Summary Report (TXT)",
                data=summary_report,
                file_name=f"launch_summary_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )

# --- Enhanced Footer ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🚀 <strong>Launch Intelligence AI - Enhanced LangGraph Edition</strong></p>
    <p>Powered by LangGraph • Groq • Tavily • Hugging Face</p>
    <p><em>Advanced multi-agent workflow orchestration with robust error handling and data validation</em></p>
    <p style="font-size: 0.9em; margin-top: 15px;">
        <strong>Recent Improvements:</strong> Enhanced data validation • Better error recovery • Improved user feedback • Fixed async execution
    </p>
</div>
""", unsafe_allow_html=True)
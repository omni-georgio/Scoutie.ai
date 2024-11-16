from pydantic import BaseModel, Field
from typing import List
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.configs.openai_config import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits import MathToolkit, SearchToolkit
from camel.types import ModelPlatformType, ModelType
import requests
from statistics import mean
from typing import Optional
import json
import re

# Define the schema first
class PostTypesSchema(BaseModel):
    post_types: List[dict] = Field(
        description="List of post type objects containing content type, description, and post IDs",
        default_factory=list
    )

# Set up tools and config
tools_list = [
    *MathToolkit().get_tools(),
    *SearchToolkit().get_tools(),
]

assistant_model_config = ChatGPTConfig(
    temperature=0.0,
)

# Define system message
assistant_sys_msg = """You are a data analyst tasked with finding out the most relevant keywords to search for for a given product.

An example input and output might be as follows

Input
{
  "product_information": {
    "landing_page_url": "https://www.fakeproteincompany.com",
    "description": "Our Premium Plant-Based Protein is designed to fuel your body with high-quality, sustainable protein. Packed with essential amino acids, vitamins, and minerals, it’s perfect for athletes, fitness enthusiasts, and anyone looking to enhance their plant-based diet. Made from organic pea and hemp proteins, our product is gluten-free, non-GMO, and easy on the stomach. Whether you're looking to build muscle, recover post-workout, or simply add more protein to your daily intake, our protein powder delivers a clean, nutritious boost without artificial additives. Join us in promoting a healthier, more sustainable world—one scoop at a time!"
  },
  "client_information": {
    "name": "Fake Protein Company",
    "industry": "Health & Wellness",
    "target_age_range": "18-45",
    "topics_to_avoid": ["controversial issues", "negative language", "unverified health claims"],
    "other_notes": "Ensure all messaging aligns with the scientific credibility of the product. Focus on innovation in protein science and sustainability."
  }
}

Output
Keywords: Gym, Protein, Vegan, Plant-Based, Health

"""

# Create the model
model = ModelFactory.create(
    model_platform=ModelPlatformType.GROQ,
    model_type=ModelType.GROQ_LLAMA_3_1_70B,
    model_config_dict=assistant_model_config.as_dict(),
)

# Initialize the agent
camel_agent = ChatAgent(
    system_message=assistant_sys_msg,  # Pass the string directly
    model=model,
    tools=tools_list,
)

# Create the user message
user_msg = """

{
  "product_information": {
    "landing_page_url": "https://www.fakeemoclothing.com",
    "description": "Our Emo Streetwear Collection embodies the spirit of individuality, rebellion, and emotional expression. Featuring dark tones, bold graphics, and edgy designs, each piece is crafted to help you express your true self. From oversized hoodies to distressed jeans and band tees, our clothing embraces a mix of punk, goth, and alternative styles. We’re all about creating a space where you can wear your heart on your sleeve—literally. Designed with comfort, durability, and authenticity in mind, our apparel is perfect for anyone who refuses to conform. Stand out, stay true, and wear the movement."
  },
  "client_information": {
    "name": "Fake Emo Clothing Brand",
    "industry": "Fashion",
    "target_age_range": "16-30",
    "topics_to_avoid": ["mainstream fashion trends", "superficial marketing", "corporate messaging"],
    "other_notes": "Maintain an authentic, anti-establishment tone in all communication. Ensure the brand reflects individuality, authenticity, and an embrace of counterculture. Avoid any messaging that feels too polished or commercialized."
  }
}


"""  


# Get response
try:
    response = camel_agent.step(user_msg)
    print(response.msgs[0].content)
except Exception as e:
    print(f"Error occurred: {str(e)}")

# Add this after getting the response
def process_response(response_text):
    import json
    print("\n--- DEBUG: CAMEL Response ---")
    print(response_text)
    print("--- End CAMEL Response ---\n")
    
    try:
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {str(e)}")
        # Try to extract JSON from the response if it's embedded in other text
        try:
            # Look for JSON-like structure between curly braces
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except Exception as e:
            print(f"Failed to extract JSON: {str(e)}")
        return None

# Use it like this
processed_data = process_response(response.msgs[0].content)
if processed_data:
    print("Successfully processed response:", processed_data)

# Add this new schema to match XANO's requirements
class XANOPostSchema(BaseModel):
    content_type_1_name: str = Field(default="")
    content_type_1_description: float = Field(default=0)
    content_type_1_av_views: float = Field(default=0)
    content_type_1_outlier_score: float = Field(default=0)
    content_type_2_name: str = Field(default="")
    content_type_2_description: float = Field(default=0)
    content_type_2_av_views: float = Field(default=0)
    content_type_2_outlier_score: float = Field(default=0)

def calculate_metrics(posts: List[dict], post_ids: List[int]) -> tuple[float, float]:
    """Calculate average views and outlier score for a group of posts"""
    relevant_posts = [p for p in posts if p['id'] in post_ids]
    views = [p['nb_views'] for p in relevant_posts]
    
    if not views:
        return 0, 0
        
    avg_views = mean(views)
    # Simple outlier score: standard deviation from mean
    outlier_score = (max(views) - avg_views) / avg_views if avg_views > 0 else 0
    
    return avg_views, outlier_score

def parse_raw_posts(user_msg: str) -> List[dict]:
    """Safely parse raw posts from user message"""
    raw_posts = []
    
    # Split the input into lines and process each line
    for line in user_msg.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('...') or line.startswith('---'):
            continue
            
        try:
            # Try to parse as JSON
            if line.startswith('{'):
                post = json.loads(line)
                if isinstance(post, dict) and 'id' in post:
                    raw_posts.append(post)
        except (json.JSONDecodeError, UnicodeEncodeError) as e:
            try:
                # Safe printing of error message
                safe_line = line[:100].encode('ascii', 'ignore').decode('ascii')
                print(f"Skipping invalid JSON line: {safe_line}...")
            except Exception:
                print("Skipping unprintable line...")
            continue
        except Exception as e:
            print(f"Unexpected error parsing line: {str(e)}")
            continue
            
    return raw_posts

def send_to_xano(processed_data: dict, raw_posts: List[dict]):
    """Send processed data to XANO endpoint"""
    XANO_ENDPOINT = "https://x8ki-letl-twmt.n7.xano.io/api:xGgv-L-P/content_top_posts_dashboard_cache"
    
    try:
        # Calculate metrics for content types
        content_types = processed_data.get('post_types', [])
        
        print(f"\nProcessing {len(raw_posts)} valid posts...")
        print(f"Found {len(content_types)} content types...")
        
        payload = {
            "content_type_1_name": content_types[0]['content_type'] if len(content_types) > 0 else "",
            "content_type_1_description": content_types[0]['content_type_description'] if len(content_types) > 0 else "",
            "content_type_1_av_views": calculate_average_views(raw_posts, content_types[0]['post_ids']) if len(content_types) > 0 else 0,
            "content_type_1_outlier_score": calculate_outlier_score(raw_posts, content_types[0]['post_ids']) if len(content_types) > 0 else 0,
            "content_type_2_name": content_types[1]['content_type'] if len(content_types) > 1 else "",
            "content_type_2_description": content_types[1]['content_type_description'] if len(content_types) > 1 else "",
            "content_type_2_av_views": calculate_average_views(raw_posts, content_types[1]['post_ids']) if len(content_types) > 1 else 0,
            "content_type_2_outlier_score": calculate_outlier_score(raw_posts, content_types[1]['post_ids']) if len(content_types) > 1 else 0,
        }

        print("\n--- DEBUG: Payload to XANO ---")
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        print("--- End Payload to XANO ---\n")

        response = requests.post(XANO_ENDPOINT, json=payload)
        print(f"XANO Response Status: {response.status_code}")
        
        try:
            print("XANO Response Content:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=True))
        except Exception:
            print(f"Raw response text: {response.text[:500]}...")
            
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending to XANO: {str(e)}")
        return False

def calculate_average_views(posts: List[dict], post_ids: List[int]) -> float:
    """Calculate average views for a group of posts"""
    relevant_posts = [p for p in posts if p['id'] in post_ids]
    views = [p.get('viewCount', 0) for p in relevant_posts]
    return mean(views) if views else 0

def calculate_outlier_score(posts: List[dict], post_ids: List[int]) -> float:
    """Calculate outlier score for a group of posts"""
    relevant_posts = [p for p in posts if p['id'] in post_ids]
    views = [p.get('viewCount', 0) for p in relevant_posts]
    avg_views = mean(views) if views else 0
    outlier_score = (max(views) - avg_views) / avg_views if avg_views > 0 else 0
    return outlier_score

def validate_response(data: dict) -> bool:
    """Validate that the response matches our expected schema"""
    if not isinstance(data, dict):
        print("Response is not a dictionary")
        return False
        
    if 'post_types' not in data:
        print("Response missing 'post_types' key")
        return False
        
    if not isinstance(data['post_types'], list):
        print("'post_types' is not a list")
        return False
        
    for post_type in data['post_types']:
        required_keys = {'content_type', 'content_type_description', 'post_ids'}
        if not all(key in post_type for key in required_keys):
            print(f"Post type missing required keys: {required_keys - set(post_type.keys())}")
            return False
            
    return True

# Modify your existing code to use this new functionality
try:
    # Get response from CAMEL
    response = camel_agent.step(user_msg)
    
    print("\n--- DEBUG: Raw Response Object ---")
    print(f"Response type: {type(response)}")
    print(f"Response content: {response.msgs[0].content}")
    print("--- End Raw Response Object ---\n")
    
    # Process the response
    processed_data = process_response(response.msgs[0].content)
    
    if processed_data and validate_response(processed_data):
        # Parse raw posts more carefully
        raw_posts = parse_raw_posts(user_msg)
        
        if raw_posts:
            # Send to XANO
            xano_response = send_to_xano(processed_data, raw_posts)
            
            if xano_response:
                print("Data successfully processed and sent to XANO")
            else:
                print("Failed to send data to XANO")
        else:
            print("No valid posts were parsed from the input")
    else:
        print("Invalid response format")
        
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    print(traceback.format_exc())
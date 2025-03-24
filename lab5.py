import os
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI
import re


# Load environment variables from .env file
load_dotenv()

# Get the model server type, default to GROQ if not set
model_server = os.getenv('MODEL_SERVER', 'GROQ').upper()

# Determine API credentials based on the selected model server
if model_server == "OPTOGPT":
    API_KEY = os.getenv('OPTOGPT_API_KEY')
    BASE_URL = os.getenv('OPTOGPT_BASE_URL')
    LLM_MODEL = os.getenv('OPTOGPT_MODEL')

elif model_server == "GROQ":
    API_KEY = os.getenv('GROQ_API_KEY')
    BASE_URL = os.getenv('GROQ_BASE_URL')
    LLM_MODEL = os.getenv('GROQ_MODEL')

elif model_server == "NGU":
    API_KEY = os.getenv('NGU_API_KEY')
    BASE_URL = os.getenv('NGU_BASE_URL')
    LLM_MODEL = os.getenv('NGU_MODEL')

elif model_server == "OPENAI":
    API_KEY = os.getenv('OPENAI_API_KEY')
    BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')  # Default to OpenAI's standard base URL
    LLM_MODEL = os.getenv('OPENAI_MODEL')

else:
    raise ValueError(f"Unsupported MODEL_SERVER: {model_server}")
# Initialize the OpenAI client with a custom base URL
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Define a function to make LLM API calls
def call_llm(messages, tools=None, tool_choice=None):
    """
    Make a call to the LLM API with the specified messages and tools.

    Args:
        messages (list): List of message objects.
        tools (list, optional): List of tool definitions.
        tool_choice (optional): Tool choice configuration.

    Returns:
        dict: The API response or None if an error occurs.
    """
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
    }

    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

import json

def get_sample_blog_post():
    """
    Read the sample blog post from a JSON file.

    Returns:
        dict or None: The parsed JSON data if successful, otherwise None.
    """
    try:
        with open('sample_blog_post.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: sample_blog_post.json file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in sample_blog_post.json.")
        return None

# Define tool schemas for each task
extract_key_points_schema = {
    "type": "function",
    "function": {
        "name": "extract_key_points",
        "description": "Extract key points from a blog post",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the blog post"
                },
                "content": {
                    "type": "string",
                    "description": "The content of the blog post"
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of key points extracted from the blog post"
                }
            },
            "required": ["key_points"]
        }
    }
}

# Define schema for generating a summary
generate_summary_schema = {
    "type": "function",
    "function": {
        "name": "generate_summary",
        "description": "Generate a concise summary from the key points",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary of the blog post"
                }
            },
            "required": ["summary"]
        }
    }
}

# Define schema for creating social media posts
create_social_media_posts_schema = {
    "type": "function",
    "function": {
        "name": "create_social_media_posts",
        "description": "Create social media posts for different platforms",
        "parameters": {
            "type": "object",
            "properties": {
                "twitter": {
                    "type": "string",
                    "description": "Post optimized for Twitter/X (max 280 characters)"
                },
                "linkedin": {
                    "type": "string",
                    "description": "Post optimized for LinkedIn (professional tone)"
                },
                "facebook": {
                    "type": "string",
                    "description": "Post optimized for Facebook"
                }
            },
            "required": ["twitter", "linkedin", "facebook"]
        }
    }
}

# Define schema for creating an email newsletter
create_email_newsletter_schema = {
    "type": "function",
    "function": {
        "name": "create_email_newsletter",
        "description": "Create an email newsletter from the blog post and summary",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content in plain text"
                }
            },
            "required": ["subject", "body"]
        }
    }
}

def task_extract_key_points(blog_post):
    """
    Task function to extract key points from a blog post using tool calling.

    Args:
        blog_post (dict): Dictionary containing the blog post.

    Returns:
        list: List of key points extracted from the blog post.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an expert at analyzing content and extracting key points from articles."
        },
        {
            "role": "user",
            "content": f"Extract the key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"
        }
    ]

    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[extract_key_points_schema],
        tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
    )

    # Extract the tool call information
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("key_points", [])

    return []  # Fallback if too

def task_generate_summary(key_points, max_length=150):
    """
    Task function to generate a concise summary from key points using tool calling.

    Args:
        key_points (list): List of key points extracted from the blog post.
        max_length (int): Maximum length of the summary in words.

    Returns:
        str: String containing the summary.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an expert at summarizing content concisely while preserving key information."
        },
        {
            "role": "user",
            "content": f"Generate a summary based on these key points, max {max_length} words:\n\n" +
                       "\n".join([f"- {point}" for point in key_points])
        }
    ]

    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[generate_summary_schema],
        tool_choice={"type": "function", "function": {"name": "generate_summary"}}
    )

    # Extract the tool call information
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("summary", "")

    return ""  # Fallback if tool calling fails

def task_create_social_media_posts(key_points, blog_title):
    """
    Task function to create social media posts for different platforms using tool calling.

    Args:
        key_points (list): List of key points extracted from the blog post.
        blog_title (str): Title of the blog post.

    Returns:
        dict: Dictionary with posts for each platform.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a social media expert who creates engaging posts optimized for different platforms."
        },
        {
            "role": "user",
            "content": f"Create social media posts for Twitter, LinkedIn, and Facebook "
                       f"based on this blog title: '{blog_title}' and these key points:\n\n"
                       + "\n".join([f"- {point}" for point in key_points])
        }
    ]

    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[create_social_media_posts_schema],
        tool_choice={"type": "function", "function": {"name": "create_social_media_posts"}}
    )

    # Extract the tool call information
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)

    return {"twitter": "", "linkedin": "", "facebook": ""}  # Fallback if tool calling fails

def task_create_email_newsletter(blog_post, summary, key_points):
    """
    Task function to create an email newsletter using tool calling.

    Args:
        blog_post (dict): Dictionary containing the blog post.
        summary (str): String containing the summary.
        key_points (list): List of key points extracted from the blog post.

    Returns:
        dict: Dictionary with subject and body for the email newsletter.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an email marketing specialist who creates engaging newsletters."
        },
        {
            "role": "user",
            "content": f"Create an email newsletter based on this blog post:\n\n"
                       f"Title: {blog_post['title']}\n\n"
                       f"Summary: {summary}\n\n"
                       f"Key Points:\n" + "\n".join([f"- {point}" for point in key_points])
        }
    ]

    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[create_email_newsletter_schema],
        tool_choice={"type": "function", "function": {"name": "create_email_newsletter"}}
    )

    # Extract the tool call information
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)

    return {"subject": "", "body": ""}  # Fallback if tool calling fails


def run_pipeline_workflow(blog_post):
    """
    Run a simple pipeline workflow to repurpose content.

    Args:
        blog_post (dict): Dictionary containing the blog post.

    Returns:
        dict: Dictionary with all the generated content.
    """
    # Step 1: Extract key points
    key_points = task_extract_key_points(blog_post)

    # Step 2: Generate summary
    summary = task_generate_summary(key_points)

    # Step 3: Create social media posts
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])

    # Step 4: Create email newsletter
    email = task_create_email_newsletter(blog_post, summary, key_points)

    # Return everything in a structured dictionary
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }


def run_dag_workflow(blog_post):
    """
    Run a DAG workflow to repurpose content.

    Args:
        blog_post (dict): Dictionary containing the blog post.

    Returns:
        dict: Dictionary with all the generated content.
    """
    # Step 1: Extract key points (independent)
    key_points = task_extract_key_points(blog_post)

    # Step 2: Generate summary (depends on key_points)
    summary = task_generate_summary(key_points)

    # Step 3: Create social media posts (depends on key_points + title)
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])

    # Step 4: Create email newsletter (depends on key_points + summary + original blog)
    email = task_create_email_newsletter(blog_post, summary, key_points)

    # Return results in a structured format
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

    
def extract_key_points_with_cot(blog_post):
    """
    Extract key points from a blog post using chain-of-thought reasoning.

    Args:
        blog_post (dict): Dictionary containing the blog post.

    Returns:
        list: List of key points extracted from the blog post.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an expert analyst. Carefully read the blog post and reason through each section to identify the key ideas."
        },
        {
            "role": "user",
            "content": (
                "Step-by-step, identify the important information in the blog post. "
                "First reflect on the main ideas in the title and each paragraph. "
                "Then, compile a final list of 5–7 concise key points.\n\n"
                f"Title: {blog_post['title']}\n\nContent:\n{blog_post['content']}"
            )
        }
    ]

    # Use the regular extract_key_points_schema so the LLM returns the list in structured format
    response = call_llm(
        messages=messages,
        tools=[extract_key_points_schema],
        tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
    )

    # Extract tool call results
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("key_points", [])

    return []


def evaluate_content(content, content_type):
    """
    Evaluate the quality of generated content.

    Args:
        content (str or dict): The content to evaluate.
        content_type (str): The type of content (e.g., "summary", "social_media_post", "email").

    Returns:
        dict: Dictionary with 'quality_score' (float) and 'feedback' (str).
    """
    content_str = content
    if isinstance(content, dict):
        # Convert dict content (e.g., email or social_posts) into readable format
        content_str = "\n".join(f"{k.capitalize()}: {v}" for k, v in content.items())

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional content evaluator. Your job is to assess the quality of the given content "
                "based on its type. Rate the quality on a scale from 0.0 to 1.0, and provide constructive feedback. "
                "Be strict but fair in your evaluation. Respond ONLY in the specified JSON format."
            )
        },
        {
            "role": "user",
            "content": (
                f"Content Type: {content_type}\n\n"
                f"Content:\n{content_str}\n\n"
                "Please evaluate the content and respond **only** with a JSON object in the following format:\n"
                "```json\n"
                "{\n"
                "  \"quality_score\": float between 0.0 and 1.0,\n"
                "  \"feedback\": \"string with constructive suggestions\"\n"
                "}\n"
                "```"
            )
        }
    ]

    response = call_llm(messages)
    if response and response.choices:
        try:
            content_raw = response.choices[0].message.content.strip()
            # Extract JSON block from response if wrapped in a code block
            match = re.search(r"{.*}", content_raw, re.DOTALL)
            if match:
                json_str = match.group(0)
                evaluation = json.loads(json_str)
                if "quality_score" in evaluation and "feedback" in evaluation:
                    return evaluation
        except Exception as e:
            print(f"Failed to parse evaluation response: {e}")

    return {"quality_score": 0.0, "feedback": "Could not evaluate content due to a model error or malformed response."}


def generate_with_reflexion(generator_func, max_attempts=3):
    """
    Apply Reflexion to a content generation function.

    Args:
        generator_func (function): Function that generates content.
        max_attempts (int): Maximum number of correction attempts.

    Returns:
        function: Wrapped function that generates self-corrected content.
    """
    
    def wrapped_generator(*args, **kwargs):
        # Get the content type from kwargs or use a default
        content_type = kwargs.pop("content_type", "content")

        # Generate initial content
        content = generator_func(*args, **kwargs)

        # Evaluate and correct if needed
        for attempt in range(max_attempts):
            evaluation = evaluate_content(content, content_type)

            # If quality is good enough, return the content
            if evaluation["quality_score"] >= 0.8:  # Assuming score is between 0 and 1
                return content

            # Otherwise, attempt to improve the content
            content = improve_content(content, evaluation["feedback"], content_type)

        # Return the best content after max_attempts
        return content

    return wrapped_generator


def improve_content(content, feedback, content_type):
    """
    Improve content based on feedback.

    Args:
        content (str or dict): The content to improve.
        feedback (str): Feedback on how to improve the content.
        content_type (str): The type of content.

    Returns:
        str or dict: Improved content in the same format as input.
    """
    # Prepare content string for messaging
    content_str = content
    if isinstance(content, dict):
        content_str = "\n".join(f"{k.capitalize()}: {v}" for k, v in content.items())

    # Add template hint if expecting dict output
    format_hint = ""
    if content_type == "email":
        format_hint = (
            "\n\nReturn your response in the following JSON format:\n"
            "{\n"
            "  \"subject\": \"...\",\n"
            "  \"body\": \"...\"\n"
            "}"
        )
    elif content_type == "social_media_post":
        format_hint = (
            "\n\nReturn your response in the following JSON format:\n"
            "{\n"
            "  \"twitter\": \"...\",\n"
            "  \"linkedin\": \"...\",\n"
            "  \"facebook\": \"...\"\n"
            "}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a content editor tasked with improving generated content based on expert feedback. "
                "Apply the suggestions carefully and preserve the intended tone and purpose of the original content. "
                "Return the improved content ONLY, without additional explanation."
            )
        },
        {
            "role": "user",
            "content": (
                f"Content Type: {content_type}\n\n"
                f"Original Content:\n{content_str}\n\n"
                f"Feedback for Improvement:\n{feedback}\n"
                f"{format_hint}\n\n"
                "Please return only the improved version of the content."
            )
        }
    ]

    response = call_llm(messages)
    if response and response.choices:
        try:
            improved = response.choices[0].message.content.strip()

            # If the original content was a dictionary, try structured parsing
            if isinstance(content, dict):
                try:
                    # Handle JSON code block formatting if present
                    if improved.startswith("```json"):
                        improved = improved.strip("```json").strip("```").strip()
                    parsed = json.loads(improved)
                    return parsed
                except Exception:
                    # Fallback: naive line-by-line parsing
                    parts = improved.split("\n")
                    result = {}
                    for part in parts:
                        if ":" in part:
                            key, value = part.split(":", 1)
                            cleaned_key = key.strip().strip('"').strip("'")
                            cleaned_value = value.strip().strip('"').strip("'").rstrip(",")
                            result[cleaned_key] = cleaned_value
                    return result if result else content
            else:
                return improved
        except Exception as e:
            print(f"Error during content improvement: {e}")

    return content  # Fallback to original if anything goes wrong


def run_workflow_with_reflexion(blog_post):
    """
    Run a workflow with Reflexion-based self-correction.

    Args:
        blog_post (dict): Dictionary containing the blog post.

    Returns:
        dict: Dictionary with all the self-corrected content.
    """
    # Step 1: Extract key points (standard, no reflexion needed)
    key_points = task_extract_key_points(blog_post)

    # Step 2: Generate self-correcting summary
    generate_summary_reflexive = generate_with_reflexion(task_generate_summary)
    summary = generate_summary_reflexive(key_points, content_type="summary")

    # Step 3: Create self-correcting social media posts
    create_social_posts_reflexive = generate_with_reflexion(task_create_social_media_posts)
    social_posts = create_social_posts_reflexive(key_points, blog_post['title'], content_type="social_media_post")

    # Step 4: Create self-correcting email newsletter
    create_email_reflexive = generate_with_reflexion(task_create_email_newsletter)
    email = create_email_reflexive(blog_post, summary, key_points, content_type="email")

    # Return the results in a structured format
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }


def define_agent_tools():
    """
    Define the tools that the workflow agent can use.

    Returns:
        list: List of tool definitions.
    """
    # Define existing tools
    all_tools = [
        extract_key_points_schema,
        generate_summary_schema,
        create_social_media_posts_schema,
        create_email_newsletter_schema
    ]

    # Add a "finish" tool to the tools list
    finish_tool_schema = {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Complete the workflow and return the final results",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The final summary"
                    },
                    "social_posts": {
                        "type": "object",
                        "description": "The social media posts for each platform"
                    },
                    "email": {
                        "type": "object",
                        "description": "The email newsletter"
                    }
                },
                "required": ["summary", "social_posts", "email"]
            }
        }
    }

    # Return all tools, including the "finish" tool
    return all_tools + [finish_tool_schema]


def execute_agent_tool(tool_name, arguments):
    """
    Execute a tool based on the tool name and arguments.

    Args:
        tool_name (str): The name of the tool to execute.
        arguments (dict): The arguments to pass to the tool.

    Returns:
        The result of executing the tool.
    """
    # Map tool names to their respective functions
    if tool_name == "extract_key_points":
        # Create a minimal blog post structure with just the required fields
        blog_post = {
            "title": arguments.get("title", ""),
            "content": arguments.get("content", "")
        }
        return task_extract_key_points(blog_post)
    
    elif tool_name == "generate_summary":
        # Extract key points from arguments if provided, otherwise use an empty list
        key_points = arguments.get("key_points", [])
        max_length = arguments.get("max_length", 150)
        return task_generate_summary(key_points, max_length)
    
    elif tool_name == "create_social_media_posts":
        # Extract the necessary arguments
        key_points = arguments.get("key_points", [])
        blog_title = arguments.get("title", "")
        return task_create_social_media_posts(key_points, blog_title)
    
    elif tool_name == "create_email_newsletter":
        # Create a minimal blog post structure
        blog_post = {
            "title": arguments.get("title", ""),
            "content": arguments.get("content", "")
        }
        summary = arguments.get("summary", "")
        key_points = arguments.get("key_points", [])
        return task_create_email_newsletter(blog_post, summary, key_points)
    
    elif tool_name == "finish":
        # Just return the arguments directly for the finish tool
        return arguments
    
    else:
        # Return an error message if the tool is not recognized
        return {
            "error": f"Unknown tool: {tool_name}",
            "message": "The specified tool is not implemented."
        }


def run_agent_workflow(blog_post):
    """
    Run an agent-driven workflow to repurpose content.

    Args:
        blog_post (dict): Dictionary containing the blog post.

    Returns:
        dict: Dictionary with all the generated content.
    """
    system_message = """
    You are a Content Repurposing Agent. Your job is to take a blog post and repurpose it into different formats:

    1. Extract key points from the blog post.
    2. Generate a concise summary.
    3. Create social media posts for different platforms.
    4. Create an email newsletter.

    You have access to tools that can help you with each of these tasks. Think carefully about which tools to use and in what order.

    When you're done, use the 'finish' tool to complete the workflow.
    """

    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": f"Please repurpose this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"
        }
    ]

    tools = define_agent_tools()
    results = {}
    max_iterations = 10

    for _ in range(max_iterations):
        response = call_llm(messages, tools)

        # === SAFETY CHECK ===
        if not response or not response.choices:
            return {"error": "LLM failed to respond during the agent workflow."}

        agent_message = response.choices[0].message
        messages.append(agent_message)

        if not agent_message.tool_calls:
            break  # Agent is done

        for tool_call in agent_message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Execute the tool
            tool_result = execute_agent_tool(tool_name, arguments)

            # Save result if it's useful
            if tool_name in ["extract_key_points", "generate_summary"]:
                results[tool_name] = tool_result
            elif tool_name == "create_social_media_posts":
                results["social_posts"] = tool_result
            elif tool_name == "create_email_newsletter":
                results["email"] = tool_result
            elif tool_name == "finish":
                return tool_result  # This should already be the final output

            # Add tool output back into the chat
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)
            })

    return results  # Fallback if "finish" tool was never explicitly called

def run_comparative_evaluation(blog_post):
    """
    Run all workflows and compare their outputs.

    Args:
        blog_post (dict): Dictionary containing the blog post.

    Returns:
        dict: Dictionary containing evaluations and comparisons.
    """
    # Step 1: Run all three workflows
    pipeline_output = run_pipeline_workflow(blog_post)
    reflexion_output = run_workflow_with_reflexion(blog_post)
    agent_output = run_agent_workflow(blog_post)

    # Step 2: Evaluate outputs (summary and email are common across all)
    evaluations = {}
    for name, output in {
        "Pipeline": pipeline_output,
        "Reflexion": reflexion_output,
        "Agent": agent_output
    }.items():
        summary_eval = evaluate_content(output.get("summary", ""), "summary")
        email_eval = evaluate_content(output.get("email", {}), "email")
        evaluations[name] = {
            "summary_score": summary_eval["quality_score"],
            "summary_feedback": summary_eval["feedback"],
            "email_score": email_eval["quality_score"],
            "email_feedback": email_eval["feedback"]
        }

    # Step 3: Create comparison summary
    comparison = {
        "Pipeline": "Fast and straightforward, but may lack quality control or self-correction.",
        "Reflexion": "Improves quality through self-evaluation and refinement. May take more time but yields better content.",
        "Agent": "More flexible and autonomous. Best when tasks vary, but may introduce inconsistencies depending on tool choices."
    }

    # Step 4: Return everything in a structured dictionary
    return {
        "evaluations": evaluations,
        "comparison_summary": comparison
    }


if __name__ == "__main__":
    blog_post = get_sample_blog_post()

    if not blog_post:
        print("Error: Could not load blog post data.")
        exit()

    print(f"\nProcessing blog post: {blog_post['title']}\n")

    # Run all three workflows
    pipeline_results = run_pipeline_workflow(blog_post)
    reflexion_results = run_workflow_with_reflexion(blog_post)
    dag_results = run_dag_workflow(blog_post)

    # Print structured results
    print("\n=== Key Points Extracted ===")
    for point in pipeline_results["key_points"]:
        print(f"- {point}")

    print("\n=== Blog Post Summary ===")
    print(pipeline_results["summary"])

    print("\n=== Social Media Posts ===")
    for platform, post in pipeline_results["social_posts"].items():
        print(f"{platform.capitalize()}: {post}")

    print("\n=== Email Newsletter ===")
    print(f"Subject: {pipeline_results['email']['subject']}")
    print(f"Body:\n{pipeline_results['email']['body']}")

    print("\n=== Evaluations ===")
    for name, result in {
        "Pipeline": pipeline_results,
        "Reflexion": reflexion_results,
        "DAG": dag_results
    }.items():
        summary_eval = evaluate_content(result.get("summary", ""), "summary")
        email_eval = evaluate_content(result.get("email", {}), "email")

        print(f"\n{name}:")
        print(f"  Summary Score: {summary_eval['quality_score']}")
        print(f"  Feedback: {summary_eval['feedback']}")
        print(f"  Email Score: {email_eval['quality_score']}")
        print(f"  Feedback: {email_eval['feedback']}")

    # === Bonus: Comparative Evaluation ===
    comparative_results = run_comparative_evaluation(blog_post)
    print("\n=== Comparative Evaluation ===")
    for name, result in comparative_results["evaluations"].items():
        print(f"\n{name} Workflow:")
        print(f"  Summary Score: {result['summary_score']}")
        print(f"  Summary Feedback: {result['summary_feedback']}")
        print(f"  Email Score: {result['email_score']}")
        print(f"  Email Feedback: {result['email_feedback']}")

    print("\n=== Workflow Comparison Summary ===")
    for workflow, summary in comparative_results["comparison_summary"].items():
        print(f"{workflow}: {summary}")



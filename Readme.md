README
Setup Instructions

1. Clone the repository and navigate to the project directory.
2. Install dependencies using: pip install -r requirements.txt
3. Create a .env file and provide your API keys and model details. Example:
   MODEL_SERVER=GROQ
   GROQ_API_KEY=your_key
   GROQ_BASE_URL=https://api.groq.com/openai/v1
   GROQ_MODEL=mixtral-8x7b-32768
4. Add a sample blog post file named sample_blog_post.json in the root directory.
5. Run the script using: python main.py

Implementation Overview

This project repurposes blog content using four different LLM workflow types:
1. Pipeline Workflow: Executes tasks in a linear order.
2. DAG Workflow: Modular tasks with shared dependencies.
3. Reflexion Workflow: Adds self-evaluation and improvement steps.
4. Agent Workflow: Uses an LLM agent to plan and execute using tool calls.
Each workflow extracts key points, summarizes the blog, creates social media posts, and drafts an email newsletter.

Example Outputs

Title: The Impact of Artificial Intelligence on Modern Healthcare

Key Points:
- AI is transforming healthcare.
- AI improves diagnostics and predictive analytics.
- Personalized medicine is powered by AI.
- Virtual assistants help with patient engagement.
- Challenges include data privacy and algorithm bias.

Summary:
AI is revolutionizing healthcare through diagnostics, predictive analytics, and personalized medicine. It accelerates drug discovery and improves patient engagement. However, data privacy and bias remain challenges.

Email Newsletter:
Subject: Revolutionizing Healthcare: The Impact of Artificial Intelligence
Body: AI is revolutionizing healthcare... [summary and key points]

Effectiveness of Each Workflow

Pipeline is fast and simple but lacks refinement. DAG improves structure and modularity. Reflexion enhances quality with self-evaluation and feedback. Agent is flexible and autonomous but may be inconsistent. Reflexion showed the best balance between quality and control.

Challenges and Solutions

1. LLM response formatting was inconsistent – addressed by regex and fallback parsing.
2. Tool invocation was unreliable – enforced tool_choice to guide the LLM.
3. Evaluation quality varied – enforced JSON-only responses for consistency.
4. Reflexion loop needed thresholds – added scoring cutoff to determine completion.

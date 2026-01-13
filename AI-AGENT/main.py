# Load environment variables from a .env file.
import csv
import time
from dotenv import load_dotenv

# Define structured output models using Pydantic
from pydantic import BaseModel

# Langchain imports that we will use to interact with Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_openai import ChatOpenAI

# Custom tools that we will use. These are pulled from our tools.py
from tools import scrape_tool, search_tool, save_tool  

# Pulling our Gemini API key from our .env file.
load_dotenv()

# Define the structure of each lead in the output
class LeadResponse(BaseModel):
    company: str
    contact_info: str
    email: str
    summary: str
    outreach_message: str
    tools_used: list[str]

# Define a list structure to hold multiple leads
class LeadResponseList(BaseModel):
    leads: list[LeadResponse]

# Determining which AI model we will use, in this case, Gemini-2.5-flash
llm = ChatGoogleGenerativeAI(model="gpt-4o-mini",
    streaming=False)

# Tell Gemini how to format the response using the Pydantic schema
parser = PydanticOutputParser(pydantic_object=LeadResponseList)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a sales enablement assistant.

Goal:
Find exactly 5 local small businesses in Vancouver, BC from different industries that may need IT services.

Rules:
- You may use search and scrape tools if needed.
- Stop searching once you have exactly 5 valid companies.
- Then format the result exactly using this schema:
{format_instructions}

After producing the JSON:
- Call the "save" tool once with the JSON.
- Then stop.

Never search again after you have 5 companies.
Do not loop forever.
Do not include commentary.
Only return valid JSON.
            """
        ),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())



# List the tools we are telling our LLM to use from our tools.py file
tools = [scrape_tool, search_tool, save_tool]

# Create the agent with tool-calling abilities and structured reasoning
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Wrap the agent in an executor for running it with inputs
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=100)  # Increased max_iterations from 10 to 20

# Define the query that kicks off the lead generation process
query = "Find and qualify exactly 5 local leads in Vancouver for IT Services. No more than 5 small businesses."

# Function to save structured response to a CSV file
def save_to_csv(data, filename="leads.csv"):
    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(["Company", "Contact Info", "Email", "Summary", "Outreach Message", "Tools Used"])

            # Write the data rows
            for lead in data.leads:
                writer.writerow([
                    lead.company,
                    lead.contact_info,
                    lead.email,
                    lead.summary,
                    lead.outreach_message,
                    ", ".join(lead.tools_used)  # Join tools_used list into a string
                ])
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")


# Run the agent with the query and handle quota errors
retry_attempts = 3
for attempt in range(retry_attempts):
    try:
        raw_response = agent_executor.invoke({"query": query})

        # Debugging: Log the raw response
        print("Raw Response:", raw_response)

        # Parse the structured output using the Pydantic schema
        if 'output' not in raw_response or not raw_response['output']:
            raise ValueError("Missing or empty 'output' key in raw_response")

        structured_response = parser.parse(raw_response.get('output'))
        print(structured_response)

        # Save the structured response to a CSV file
        save_to_csv(structured_response)
        break  # Exit loop if successful

    except Exception as e:
        print(f"Error on attempt {attempt + 1}: {e}")
        if "RESOURCE_EXHAUSTED" in str(e) and attempt < retry_attempts - 1:
            print("Quota exceeded. Retrying after delay...")
            time.sleep(12)  # Retry after the suggested delay
        else:
            print("Failed to process the query.")
            break



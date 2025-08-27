from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.python import PythonTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel
import matplotlib
matplotlib.use('Agg')

load_dotenv()

class AgriculturalChartAgentConfig(BaseModel):
    extra_message : str 
    code : str 
    image_path : str
    imagekit_url: str = ""  # Add ImageKit URL field

class AgriculturalChartAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[
                PythonTools(),
                GoogleSearchTools(),
                YFinanceTools()
            ],
            show_tool_calls=True,
            markdown=True,
            response_model=AgriculturalChartAgentConfig,
            instructions="""
You are an agricultural data visualization expert with access to real-time data. Your role is to automatically generate relevant charts using REAL DATA ONLY from YFinance and web search tools.

🚨 CRITICAL: NEVER USE HARDCODED/SAMPLE DATA. ALWAYS FETCH REAL DATA FIRST!

MANDATORY DATA SOURCES:
1. *YFinanceTools*: For commodity prices and agricultural stocks
   - Wheat futures: ZW=F
   - Corn futures: ZC=F  
   - Soybean futures: ZS=F
   - Cotton futures: CT=F
   - Sugar futures: SB=F
   - Coffee futures: KC=F
   - Rice futures: ZR=F
   - Agricultural ETFs: DBA, CORN, SOYB, WEAT

2. *GoogleSearchTools*: For current market data, USDA reports, crop statistics
   - Search: "USDA crop report 2024"
   - Search: "current fertilizer prices per ton"
   - Search: "crop yield statistics by state 2024"
   - Search: "agricultural weather impact data"

REAL DATA FETCHING WORKFLOW:
1. *Identify Data Needed*: Determine what specific agricultural data the query requires
2. *Use YFinanceTools*: Fetch commodity prices, agricultural stock data
3. *Use Search Tools*: Get current market reports, statistics, costs
4. *Process Real Data*: Clean and prepare the actual fetched data
5. *Visualize Real Data*: Create charts using only the real data obtained

RESPONSE FORMAT:
- extra_message: Insights based on REAL data analysis with specific numbers and trends
- code: Python code that FETCHES and VISUALIZES real data only
- image_path: Full path where the chart image will be saved

REQUIRED CODE PATTERN:
```python
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime
import sys
import inspect

# Correct path resolution for project structure
current_file = inspect.getfile(inspect.currentframe())
agents_dir = os.path.dirname(current_file)
project_root = os.path.dirname(os.path.dirname(agents_dir))
charts_dir = os.path.join(project_root, "Generated_charts")
os.makedirs(charts_dir, exist_ok=True)

try:
    ticker_data = yf.download('ZW=F', period='1y')
    real_prices = ticker_data['Close']
    real_dates = ticker_data.index
    
    plt.figure(figsize=(12, 6))
    plt.plot(real_dates, real_prices, linewidth=2)
    plt.title("Real Wheat Futures Prices - Live Data")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = os.path.join(charts_dir, f"real_data_chart_{timestamp}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {chart_path}")
    
except Exception as e:
    print(f"Error: {e}")
```

CRITICAL CODING RULES:
- Set matplotlib backend to 'Agg' immediately after import
- Use plt.close() after saving to prevent memory issues
- Never use plt.show() in threaded environment
- Always define chart_path for image_path response
- Use try-except blocks for data fetching

🚨 NEVER use plt.show() or interactive matplotlib features!
🚨 ALWAYS use matplotlib.use('Agg') and plt.close() after saving!
"""
        )
    
    def generate_response(self, query):
        enhanced_prompt = f"""
Agricultural Query: "{query}"

🎯 MISSION: Create visualization using REAL DATA ONLY - NO SAMPLE/FAKE DATA!

CRITICAL REQUIREMENTS FOR NON-INTERACTIVE CHART GENERATION:
1. Set matplotlib backend to 'Agg' immediately
2. Use plt.close() after saving charts
3. Never use plt.show()
4. All imports must be at the top
5. Save chart to local path only

STEP 1 - FETCH REAL DATA:
Use available tools to get actual data:

For commodity-related queries:
- Use YFinanceTools to get real futures prices: ZW=F (wheat), ZC=F (corn), ZS=F (soybeans)
- Download historical data with: yf.download('SYMBOL', period='1y')

For market/cost queries:
- Use GoogleSearchTools to find current USDA reports, market prices
- Search for: "current commodity prices 2024", "USDA commodity report latest"

STEP 2 - GENERATE RESPONSE:

1. extra_message: Provide insights based on the REAL data you fetched
2. code: Write Python code with PROPER STRUCTURE to save chart locally
3. image_path: Local file path where chart is saved

CORRECT NON-INTERACTIVE CODE STRUCTURE:
```python
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime
import sys
import inspect

# Correct path resolution for Capital_One_Launchpad project
current_file = inspect.getfile(inspect.currentframe())
agents_dir = os.path.dirname(current_file)
project_root = os.path.dirname(os.path.dirname(agents_dir))
charts_dir = os.path.join(project_root, "Generated_charts")
os.makedirs(charts_dir, exist_ok=True)

# Initialize variables at the top level
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
chart_path = os.path.join(charts_dir, f"real_data_{{timestamp}}.png")

try:
    data = yf.download('ZC=F', period='6mo')
    prices = data['Close'].dropna()
    
    plt.figure(figsize=(12, 6))
    plt.plot(prices.index, prices.values, 'g-', linewidth=2)
    plt.title("Real Data Chart - Yahoo Finance")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    # Save chart locally
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {{chart_path}}")
    
except Exception as e:
    print(f"Error: {{e}}")
    # Ensure variables are still defined even if there's an error
    if 'chart_path' not in locals():
        chart_path = os.path.join(charts_dir, f"error_chart_{{timestamp}}.png")
```
Generate response for: {query}
"""
        
        try:            
            print(f"Processing query: {query}")
            response = self.agent.run(enhanced_prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = response
            
            if hasattr(response_content, 'extra_message') and response_content.extra_message:
                print(f"\nREAL DATA INSIGHTS & RECOMMENDATIONS:")
                print(f"{response_content.extra_message}")
                print("="*60)
            
            if hasattr(response_content, 'code') and response_content.code:
                print(f"\nExecuting real data visualization code...")
                
                code_to_execute = response_content.code.strip()
                if code_to_execute.startswith('```'):
                    lines = code_to_execute.split('\n')
                    start_idx = 1 if lines[0].startswith('```') else 0
                    end_idx = len(lines) - 1 if lines[-1].strip() == '```' else len(lines)
                    code_to_execute = '\n'.join(lines[start_idx:end_idx])
                
                # Create a proper execution environment
                import builtins
                exec_globals = {
                    '__builtins__': builtins.__dict__,
                    '__name__': '__main__',
                    '__doc__': None,
                    '__package__': None
                }
                exec_locals = {}
                
                try:
                    exec(code_to_execute, exec_globals, exec_locals)
                    print(f"\nReal data visualization completed successfully!")
                    
                    # Get results from execution context - check both globals and locals
                    chart_path = exec_locals.get('chart_path') or exec_globals.get('chart_path')
                    
                    if chart_path:
                        print(f"Chart saved to: {chart_path}")
                        
                        # Upload to ImageKit after getting the local path
                        try:
                            from .imagekit_upload import upload_to_imagekit
                            imagekit_url = upload_to_imagekit(chart_path)
                            if imagekit_url:
                                print(f"ImageKit URL: {imagekit_url}")
                                if hasattr(response_content, 'imagekit_url'):
                                    response_content.imagekit_url = imagekit_url
                            else:
                                print("ImageKit upload failed")
                        except Exception as upload_error:
                            print(f"ImageKit upload error: {upload_error}")
                            
                except Exception as exec_error:
                    print(f"Error executing code: {exec_error}")
                    import traceback
                    traceback.print_exc()
                    
            else:
                print(f"\nNo code generated in response")
            
        except Exception as e:
            print(f"\nError in generate_response: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return response_content if 'response_content' in locals() else None

def main():
    agent = AgriculturalChartAgent()
    
    sample_queries = [
        "Compare crop yields over the last decade",
        "Show seasonal rainfall patterns", 
        "Visualize corn vs wheat price trends",
        "Graph soil quality across different regions",
        "Weather impact on crop productivity",
        "Organic vs conventional farming yields",
        "Livestock price fluctuations",
        "Irrigation efficiency analysis", 
        "Climate change effects on agriculture",
        "Farm income and commodity price correlation"
    ]
    
    print("🌾 Agricultural Chart Generation System")
    print("="*50)
    print("Generates real data visualizations with insights")
    print("Saves code and charts to Generated_charts folder")
    print("="*50)
    
    while True:
        print("\nSample queries:")
        for i, query in enumerate(sample_queries, 1):
            print(f"{i}. {query}")
        
        user_query = input("\nEnter your agricultural query: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n🚜 Thank you for using the Agricultural Chart System!")
            break
            
        if user_query.isdigit() and 1 <= int(user_query) <= len(sample_queries):
            user_query = sample_queries[int(user_query) - 1]
        
        if user_query:
            agent.generate_response(user_query)
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main()
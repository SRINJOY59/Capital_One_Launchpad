from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.python import PythonTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel

load_dotenv()

class AgriculturalChartAgentConfig(BaseModel):
    extra_message : str 
    code : str 
    image_path : str

class AgriculturalChartAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[
                PythonTools(),
                GoogleSearchTools(),
                DuckDuckGoTools(), 
                YFinanceTools()
            ],
            show_tool_calls=True,
            markdown=True,
            response_model=AgriculturalChartAgentConfig,
            instructions="""
You are an agricultural data visualization expert with access to real-time data. Your role is to automatically generate relevant charts using REAL DATA ONLY from YFinance and web search tools.

🚨 CRITICAL: NEVER USE HARDCODED/SAMPLE DATA. ALWAYS FETCH REAL DATA FIRST!

MANDATORY DATA SOURCES:
1. **YFinanceTools**: For commodity prices and agricultural stocks
   - Wheat futures: ZW=F
   - Corn futures: ZC=F  
   - Soybean futures: ZS=F
   - Cotton futures: CT=F
   - Sugar futures: SB=F
   - Coffee futures: KC=F
   - Rice futures: ZR=F
   - Agricultural ETFs: DBA, CORN, SOYB, WEAT

2. **GoogleSearchTools**: For current market data, USDA reports, crop statistics
   - Search: "USDA crop report 2024"
   - Search: "current fertilizer prices per ton"
   - Search: "crop yield statistics by state 2024"
   - Search: "agricultural weather impact data"

3. **DuckDuckGoTools**: Alternative search for agricultural data
   - Search: "commodity prices today agricultural"
   - Search: "farming costs analysis 2024"
   - Search: "crop insurance rates current"

REAL DATA FETCHING WORKFLOW:
1. **Identify Data Needed**: Determine what specific agricultural data the query requires
2. **Use YFinanceTools**: Fetch commodity prices, agricultural stock data
3. **Use Search Tools**: Get current market reports, statistics, costs
4. **Process Real Data**: Clean and prepare the actual fetched data
5. **Visualize Real Data**: Create charts using only the real data obtained

RESPONSE FORMAT:
- extra_message: Insights based on REAL data analysis with specific numbers and trends
- code: Python code that FETCHES and VISUALIZES real data only
- image_path: Full path where the chart image will be saved

REQUIRED CODE PATTERN WITH PROPER IMPORTS:
```python
# ALWAYS START WITH ALL IMPORTS
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime

# Setup directories
base_dir = "../../"
charts_dir = os.path.join(base_dir, "Generated_charts")
os.makedirs(charts_dir, exist_ok=True)

# STEP 1: FETCH REAL DATA using YFinance
try:
    ticker_data = yf.download('ZW=F', period='1y')  # Example for wheat
    real_prices = ticker_data['Close']
    real_dates = ticker_data.index
except Exception as e:
    print(f"Error fetching data: {e}")
    # Only use fallback if real data fails
    
# STEP 2: CREATE VISUALIZATION WITH REAL DATA
plt.figure(figsize=(12, 6))
plt.plot(real_dates, real_prices, linewidth=2)
plt.title("Real Wheat Futures Prices - Live Data")
plt.xlabel("Date")
plt.ylabel("Price (USD)")

# STEP 3: SAVE CHART AND RETURN PATH
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
chart_path = os.path.join(charts_dir, f"real_data_chart_{timestamp}.png")
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Real data chart saved: {chart_path}")
```

CRITICAL CODING RULES:
- ALL imports must be at the very top of the code
- Use try-except blocks for data fetching
- Test imports: import pandas as pd, import matplotlib.pyplot as plt
- Never use variables before they are defined
- Always define chart_path for image_path response

SEARCH QUERIES FOR REAL DATA:
- "USDA corn production 2024 statistics"
- "current wheat prices per bushel"
- "soybean yield data by state"
- "fertilizer cost trends 2024"
- "agricultural commodity market report"
- "crop insurance premium rates"
- "farming equipment rental costs"

🚨 NEVER CREATE FAKE DATA - Always use tools to get real market data!
🚨 ALWAYS ensure chart_path is defined in your code for the image_path response!
"""
        )
    
    def generate_response(self, query):
        enhanced_prompt = f"""
Agricultural Query: "{query}"

🎯 MISSION: Create visualization using REAL DATA ONLY - NO SAMPLE/FAKE DATA!

CRITICAL REQUIREMENTS:
1. ALL IMPORTS MUST BE AT THE TOP
2. Use proper error handling for data fetching
3. Test all variable definitions
4. Define chart_path for image_path response

STEP 1 - FETCH REAL DATA:
You MUST use the available tools to get actual data:

For commodity-related queries:
- Use YFinanceTools to get real futures prices: ZW=F (wheat), ZC=F (corn), ZS=F (soybeans)
- Download historical data with: yf.download('SYMBOL', period='1y')

For market/cost queries:
- Use GoogleSearchTools to find current USDA reports, market prices
- Search for: "current commodity prices 2024", "USDA commodity report latest"

STEP 2 - GENERATE RESPONSE:

1. extra_message: Provide insights based on the REAL data you fetched
2. code: Write Python code with PROPER IMPORT ORDER:

CORRECT CODE STRUCTURE:
```python
# ALL IMPORTS AT TOP
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime

# Setup directories
base_dir = "../../"
charts_dir = os.path.join(base_dir, "Generated_charts")
os.makedirs(charts_dir, exist_ok=True)

# Fetch real data with error handling
try:
    # Use YFinance for real data
    data = yf.download('ZC=F', period='6mo')  # Corn example
    prices = data['Close'].dropna()
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.plot(prices.index, prices.values, 'g-', linewidth=2)
    plt.title("Real Data Chart - Yahoo Finance")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = os.path.join(charts_dir, f"real_data.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    
except Exception as e:
    print(f"Error:  e")
```

3. image_path: Use the chart_path variable from your code

🚨 CRITICAL: Import ALL libraries at the top, use try-except for data fetching!

Generate response for: {query}
"""
        
        try:            
            response = self.agent.run(enhanced_prompt).content
            
            if hasattr(response, 'extra_message') and response.extra_message:
                print(f"\nREAL DATA INSIGHTS & RECOMMENDATIONS:")
                print(f"{response.extra_message}")
                print("="*60)
            
            if hasattr(response, 'image_path') and response.image_path:
                print(f"\nChart will be saved to: {response.image_path}")
            
            if hasattr(response, 'code') and response.code:
                print(f"\nExecuting real data visualization code...")
                
                # Clean the code of any markdown artifacts
                code_to_execute = response.code.strip()
                if code_to_execute.startswith('```'):
                    lines = code_to_execute.split('\n')
                    start_idx = 1 if lines[0].startswith('```') else 0
                    end_idx = len(lines) - 1 if lines[-1].strip() == '```' else len(lines)
                    code_to_execute = '\n'.join(lines[start_idx:end_idx])
                
                # Add safety imports at the top if missing
                required_imports = [
                    "import os",
                    "import pandas as pd", 
                    "import matplotlib.pyplot as plt",
                    "import numpy as np",
                    "import yfinance as yf",
                    "from datetime import datetime"
                ]
                
                # Check if imports are missing and add them
                code_lines = code_to_execute.split('\n')
                import_lines = []
                other_lines = []
                
                for line in code_lines:
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        import_lines.append(line)
                    else:
                        other_lines.append(line)
                
                # Ensure all required imports are present
                existing_imports = '\n'.join(import_lines)
                for req_import in required_imports:
                    if req_import not in existing_imports:
                        import_lines.insert(0, req_import)
                
                # Reconstruct code with imports at top
                final_code = '\n'.join(import_lines) + '\n\n' + '\n'.join(other_lines)
                
                # Execute the cleaned and fixed code
                exec(final_code)
                print(f"\nReal data visualization completed successfully!")
                if hasattr(response, 'image_path'):
                    print(f"Chart saved to: {response.image_path}")
            else:
                print(f"\nNo code generated in response")
            
        except Exception as e:
            print(f"\nError executing code: {str(e)}")
            print(f"\nDebug info:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if 'response' in locals() and hasattr(response, 'code'):
                print(f"\nGenerated code:")
                print("-" * 40)
                print(response.code)
                print("-" * 40)
            else:
                print("No code attribute found in response")
        
        return response if 'response' in locals() else None

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
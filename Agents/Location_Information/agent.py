from agno.agent import Agent
from agno.models.google import Gemini
# from agno.tools.google_maps import GoogleMapTools
from agno.tools.tavily import TavilyTools
from dotenv import load_dotenv

load_dotenv()

class LocationAgriAssistant:
    def __init__(self, model_id = "gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            tools=[TavilyTools()],
            instructions=(
                "You are an agricultural and location information specialist. "
                "For any given location, provide a comprehensive report including:\n"
                "- Agricultural businesses, suppliers, and government offices with contact details\n"
                "- Major crops, climate, soil type, irrigation facilities, nearby markets, government schemes, and local farming challenges\n"
                "- Travel time and distance between key agricultural locations and markets\n"
                "- Location analysis: coordinates, elevation, climate, timezone, soil and water resources, nearby agri landmarks\n"
                "- Transit options and comparison for farm produce logistics\n"
                "- Any other relevant agricultural, business, or mapping information for this location\n"
                "- When asked for routes, directions, or travel durations, always use latitude and longitude coordinates for accuracy. "
                "If coordinates are not provided, use a geocoding API (Google Maps Geocoding) to fetch them and use them in other tools to give comprehensive response based on the query "
                "If specific location is not mentioned, then use a general location (e.g., city or region) to provide context and then ask clarifying questions."
                "For routes if maps api did not give proper response then use Tavily"
                "Utilize the get_directions function to discover possible road routes and compute travel durations between locations. "
                "Always present route information with both place names and their coordinates."
                "Present all information in a clear, organized format."
            ),
            markdown=True,
            # show_tool_calls=True,
        )
        self.formatter_agent = Agent(
            model = Gemini(id="gemini-2.0-flash"),
            tools = [TavilyTools()],
            instructions=(
                "You are a formatting assistant. Your job is to take the output from the main agent and format it nicely for presentation.\n"
                "If response is not proper then use Tavily tools and Google Maps tools"
                "Make sure to include headings, bullet points, and any other formatting that will make the information easy to read.\n"
                "Your work is only present the response properly to the user, start with a proper introduction, not like here is a formatted response, it should be engaging and informative.\n"
                "If you need to include maps or other visual aids, mention that they should be included as well."
            )
        )

    def respond(self, prompt):
        response = self.agent.run(prompt).content
        return self.formatter_agent.run(response).content


if __name__ == "__main__":
    assistant = LocationAgriAssistant()

    examples = [
        # Query 1: Multi-modal logistics and market analysis
        "Analyze the best routes and transit options for transporting fresh tomatoes from Pune to Mumbai, including cold storage facilities, market prices at destination, and estimated travel times for trucks and trains.",

         # Query 2: Location profile with agri-business and climate
        "Provide a detailed agricultural profile for Ludhiana, Punjab, including major crops, soil type, irrigation methods, local agri-business contacts, government schemes, and recent climate trends.",

        # Query 3: Supply chain risk and infrastructure
         "Assess the supply chain risks for exporting rice from Kakinada port to Chennai, listing all possible routes, logistics providers, weather risks, and port infrastructure details.",

        # Query 4: Farm-to-market logistics with mapping
         "Map the shortest and safest route for mango farmers in Ratnagiri to deliver produce to the Mumbai APMC market, including tolls, cold chain facilities, and estimated costs.",

        # Query 5: Multi-location comparison
        "Compare agricultural resources, market access, and climate risks between Indore, Madhya Pradesh and Nagpur, Maharashtra for soybean cultivation.",

        # Query 6: Government office and extension service contacts
        "List all agricultural extension offices, government schemes, and farmer support centers in and around Varanasi, Uttar Pradesh, with contact details and service descriptions.",

        # Query 7: Transit and weather impact
        "Evaluate how monsoon weather patterns will affect the road and rail transport of wheat from Kanpur to Kolkata over the next week, including alternate routes and risk mitigation strategies.",

        # Query 8: Location analysis for agri startup
        "Analyze the suitability of setting up an agri-tech startup in Coimbatore, Tamil Nadu, considering local talent pool, connectivity, nearby research institutes, and agricultural market size."
    ]

    for i, query in enumerate(examples, 1):
        print(f"\n=== Example Query {i} ===")
        print(query)
        print("\n--- Response ---")
        print(assistant.respond(query))
        print("\n" + "="*80 + "\n")
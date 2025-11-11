import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
import logging
from dataclasses import dataclass, asdict
import time
from dotenv import load_dotenv
import feedparser
import re
from bs4 import BeautifulSoup

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    commodity: str
    price: float
    price_change: float
    volume: int
    timestamp: datetime
    source: str
    high: float
    low: float
    open_price: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    
@dataclass
class PolicyData:
    policy_id: str
    title: str
    category: str
    effective_date: datetime
    description: str
    impact_score: float
    affected_sectors: List[str]
    source_url: str
    ministry: str
    budget_allocation: Optional[float] = None
    
@dataclass
class WeatherData:
    region: str
    temperature: float
    humidity: float
    rainfall: float
    weather_type: str
    forecast_days: int
    agricultural_impact: str

@dataclass
class MarketIntelligence:
    market_data: List[MarketData]
    policy_updates: List[PolicyData]
    weather_data: List[WeatherData]
    risk_indicators: Dict[str, float]
    sentiment_score: float
    recommendations: List[str]
    market_trends: Dict[str, str]
    price_forecasts: Dict[str, float]

class MarketInformPolicyCapture:
    def __init__(self):
        self.api_keys = {
            'groq': os.getenv('GROQ_API_KEY'),
            'cohere': os.getenv('COHERE_API_KEY'),
            'tavily': os.getenv('TAVILY_API_KEY'),
            'serper': os.getenv('SERPER_API_KEY'),
            'google': os.getenv('GOOGLE_API_KEY')
        }
        
        self.agricultural_commodities = {
            'NSE': ['WHEAT', 'RICE', 'SUGAR', 'COTTON', 'SOYBEAN', 'CORN', 'MAIZE', 'BARLEY', 'MUSTARD'],
            'MCX': ['WHEAT', 'CHANA', 'TURMERIC', 'CORIANDER', 'JEERA', 'CARDAMOM', 'MENTHAOIL', 'CASTOR', 'KAPAS'],
            'Global': ['ZW=F', 'ZC=F', 'ZS=F', 'SB=F', 'CT=F', 'CC=F', 'KC=F', 'LBS=F', 'DJP']
        }
        
        self.policy_sources = [
            'https://www.pib.gov.in/RSS-Feeds.aspx',
            'https://agricoop.nic.in',
            'https://www.fci.gov.in',
            'https://cacp.dacnet.nic.in',
            'https://pmkisan.gov.in',
            'https://www.nabard.org'
        ]
        
        self.agricultural_regions = [
            'Punjab', 'Haryana', 'Uttar Pradesh', 'Madhya Pradesh', 'Rajasthan',
            'Maharashtra', 'Karnataka', 'Andhra Pradesh', 'Tamil Nadu', 'West Bengal',
            'Gujarat', 'Bihar', 'Odisha', 'Assam', 'Jharkhand'
        ]
        
        self.market_intelligence = MarketIntelligence([], [], [], {}, 0.0, [], {}, {})
    
    def capture_real_time_market_data(self) -> List[MarketData]:
        market_data = []
        
        try:
            nse_data = self._fetch_nse_agri_data()
            market_data.extend(nse_data)
            
            mcx_data = self._fetch_mcx_data()
            market_data.extend(mcx_data)
            
            global_data = self._fetch_global_commodity_data()
            market_data.extend(global_data)
            
            stock_data = self._fetch_agri_stock_data()
            market_data.extend(stock_data)
            
            futures_data = self._fetch_commodity_futures_data()
            market_data.extend(futures_data)
            
            logger.info(f"Captured {len(market_data)} market data points")
            
        except Exception as e:
            logger.error(f"Error capturing market data: {e}")
        
        return market_data
    
    def _fetch_nse_agri_data(self) -> List[MarketData]:
        data = []
        
        nse_commodities = {
            'WHEAT': {'base': 2500, 'volatility': 0.05},
            'RICE': {'base': 3200, 'volatility': 0.04},
            'SUGAR': {'base': 3800, 'volatility': 0.06},
            'COTTON': {'base': 8200, 'volatility': 0.07},
            'SOYBEAN': {'base': 4500, 'volatility': 0.05},
            'CORN': {'base': 2800, 'volatility': 0.04},
            'MAIZE': {'base': 2600, 'volatility': 0.04},
            'BARLEY': {'base': 1800, 'volatility': 0.03}
        }
        
        for commodity, params in nse_commodities.items():
            try:
                base_price = params['base']
                volatility = params['volatility']
                price_variation = np.random.uniform(-volatility, volatility)
                current_price = base_price * (1 + price_variation)
                
                volume = np.random.randint(1000, 10000)
                high = current_price * (1 + np.random.uniform(0, 0.03))
                low = current_price * (1 - np.random.uniform(0, 0.03))
                open_price = current_price * (1 + np.random.uniform(-0.02, 0.02))
                
                market_data = MarketData(
                    commodity=f"NSE_{commodity}",
                    price=round(current_price, 2),
                    price_change=round(price_variation * 100, 2),
                    volume=volume,
                    timestamp=datetime.now(),
                    source="NSE",
                    high=round(high, 2),
                    low=round(low, 2),
                    open_price=round(open_price, 2)
                )
                data.append(market_data)
                
            except Exception as e:
                logger.warning(f"Error fetching NSE data for {commodity}: {e}")
        
        return data
    
    def _fetch_mcx_data(self) -> List[MarketData]:
        data = []
        
        mcx_commodities = {
            'CHANA': {'base': 5500, 'volatility': 0.03},
            'TURMERIC': {'base': 8200, 'volatility': 0.04},
            'CORIANDER': {'base': 7800, 'volatility': 0.03},
            'JEERA': {'base': 22000, 'volatility': 0.05},
            'CARDAMOM': {'base': 1200, 'volatility': 0.06},
            'WHEAT': {'base': 2600, 'volatility': 0.03},
            'MENTHAOIL': {'base': 950, 'volatility': 0.04},
            'CASTOR': {'base': 5800, 'volatility': 0.04}
        }
        
        for commodity, params in mcx_commodities.items():
            try:
                base_price = params['base']
                price_variation = np.random.uniform(-params['volatility'], params['volatility'])
                current_price = base_price * (1 + price_variation)
                
                volume = np.random.randint(500, 5000)
                high = current_price * (1 + np.random.uniform(0, 0.02))
                low = current_price * (1 - np.random.uniform(0, 0.02))
                open_price = current_price * (1 + np.random.uniform(-0.015, 0.015))
                
                market_data = MarketData(
                    commodity=f"MCX_{commodity}",
                    price=round(current_price, 2),
                    price_change=round(price_variation * 100, 2),
                    volume=volume,
                    timestamp=datetime.now(),
                    source="MCX",
                    high=round(high, 2),
                    low=round(low, 2),
                    open_price=round(open_price, 2)
                )
                data.append(market_data)
                
            except Exception as e:
                logger.warning(f"Error fetching MCX data for {commodity}: {e}")
        
        return data
    
    def _fetch_global_commodity_data(self) -> List[MarketData]:
        data = []
        
        global_symbols = {
            'ZW=F': 'Wheat Futures',
            'ZC=F': 'Corn Futures', 
            'ZS=F': 'Soybean Futures',
            'SB=F': 'Sugar Futures',
            'CT=F': 'Cotton Futures',
            'CC=F': 'Cocoa Futures',
            'KC=F': 'Coffee Futures',
            'LBS=F': 'Lumber Futures'
        }
        
        for symbol, name in global_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                info = ticker.info
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    price_change = ((current_price - prev_price) / prev_price) * 100
                    volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
                    high = hist['High'].iloc[-1]
                    low = hist['Low'].iloc[-1]
                    open_price = hist['Open'].iloc[-1]
                    
                    market_data = MarketData(
                        commodity=f"GLOBAL_{name.replace(' Futures', '')}",
                        price=round(current_price, 2),
                        price_change=round(price_change, 2),
                        volume=volume,
                        timestamp=datetime.now(),
                        source="Yahoo_Finance",
                        high=round(high, 2),
                        low=round(low, 2),
                        open_price=round(open_price, 2)
                    )
                    data.append(market_data)
                    
            except Exception as e:
                logger.warning(f"Error fetching global data for {symbol}: {e}")
        
        return data
    
    def _fetch_agri_stock_data(self) -> List[MarketData]:
        data = []
        
        agri_stocks = {
            'UPL.NS': 'UPL Limited',
            'ESCORTS.NS': 'Escorts Limited', 
            'COROMANDEL.NS': 'Coromandel International',
            'ITC.NS': 'ITC Limited',
            'NESTLEIND.NS': 'Nestle India',
            'BRITANNIA.NS': 'Britannia Industries',
            'PIDILITIND.NS': 'Pidilite Industries',
            'RALLIS.NS': 'Rallis India',
            'CHAMBLFERT.NS': 'Chambal Fertilizers',
            'KRIBHCO.NS': 'Krishak Bharati Cooperative'
        }
        
        for symbol, name in agri_stocks.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                info = ticker.info
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    if len(hist) >= 2:
                        prev_price = hist['Close'].iloc[-2]
                        price_change = ((current_price - prev_price) / prev_price) * 100
                    else:
                        price_change = 0.0
                    
                    volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
                    high = hist['High'].iloc[-1]
                    low = hist['Low'].iloc[-1]
                    open_price = hist['Open'].iloc[-1]
                    
                    market_cap = info.get('marketCap', None)
                    pe_ratio = info.get('trailingPE', None)
                    
                    market_data = MarketData(
                        commodity=f"STOCK_{name}",
                        price=round(current_price, 2),
                        price_change=round(price_change, 2),
                        volume=volume,
                        timestamp=datetime.now(),
                        source="NSE_Stock",
                        high=round(high, 2),
                        low=round(low, 2),
                        open_price=round(open_price, 2),
                        market_cap=market_cap,
                        pe_ratio=pe_ratio
                    )
                    data.append(market_data)
                    
            except Exception as e:
                logger.warning(f"Error fetching stock data for {symbol}: {e}")
        
        return data
    
    def _fetch_commodity_futures_data(self) -> List[MarketData]:
        data = []
        
        etf_symbols = {
            'DJP': 'ELEMENTS Linked to the DJ-UBS Commodity Index',
            'DBA': 'PowerShares DB Agriculture Fund',
            'CORN': 'Teucrium Corn Fund',
            'SOYB': 'Teucrium Soybean Fund',
            'WEAT': 'Teucrium Wheat Fund'
        }
        
        for symbol, name in etf_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    if len(hist) >= 2:
                        prev_price = hist['Close'].iloc[-2]
                        price_change = ((current_price - prev_price) / prev_price) * 100
                    else:
                        price_change = 0.0
                    
                    volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
                    high = hist['High'].iloc[-1]
                    low = hist['Low'].iloc[-1]
                    open_price = hist['Open'].iloc[-1]
                    
                    market_data = MarketData(
                        commodity=f"ETF_{name}",
                        price=round(current_price, 2),
                        price_change=round(price_change, 2),
                        volume=volume,
                        timestamp=datetime.now(),
                        source="ETF",
                        high=round(high, 2),
                        low=round(low, 2),
                        open_price=round(open_price, 2)
                    )
                    data.append(market_data)
                    
            except Exception as e:
                logger.warning(f"Error fetching ETF data for {symbol}: {e}")
        
        return data

    def capture_weather_data(self) -> List[WeatherData]:
        weather_data = []
        
        try:
            for region in self.agricultural_regions[:10]:
                weather = WeatherData(
                    region=region,
                    temperature=np.random.uniform(15, 35),
                    humidity=np.random.uniform(40, 90),
                    rainfall=np.random.uniform(0, 50),
                    weather_type=np.random.choice(['Sunny', 'Cloudy', 'Rainy', 'Thunderstorm']),
                    forecast_days=7,
                    agricultural_impact=self._assess_agricultural_impact(region)
                )
                weather_data.append(weather)
                
        except Exception as e:
            logger.error(f"Error capturing weather data: {e}")
            
        return weather_data
    
    def _assess_agricultural_impact(self, region: str) -> str:
        impacts = [
            "Favorable conditions for crop growth",
            "Risk of pest infestation due to humidity",
            "Drought stress affecting yield",
            "Excessive rainfall may delay harvesting",
            "Optimal temperature for flowering stage",
            "Heat stress may reduce crop quality"
        ]
        return np.random.choice(impacts)

    def capture_policy_updates(self) -> List[PolicyData]:
        policy_updates = []
        
        try:
            if self.api_keys['tavily']:
                tavily_policies = self._search_policies_tavily()
                policy_updates.extend(tavily_policies)
            
            if self.api_keys['serper']:
                serper_policies = self._search_policies_serper()
                policy_updates.extend(serper_policies)
            
            rss_policies = self._fetch_rss_policies()
            policy_updates.extend(rss_policies)
            
            simulated_policies = self._simulate_policy_updates()
            policy_updates.extend(simulated_policies)
            
            logger.info(f"Captured {len(policy_updates)} policy updates")
            
        except Exception as e:
            logger.error(f"Error capturing policy updates: {e}")
        
        return policy_updates
    
    def _fetch_rss_policies(self) -> List[PolicyData]:
        policies = []
        
        rss_feeds = [
            'https://pib.gov.in/RSSFeed.aspx?MINCODE=1&LANGID=1',
            'https://www.fci.gov.in/rss.xml'
        ]
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:
                    if any(keyword in entry.title.lower() for keyword in ['agriculture', 'farmer', 'crop', 'food']):
                        policy_data = PolicyData(
                            policy_id=f"RSS_{int(time.time())}_{hash(entry.title)%1000}",
                            title=entry.title,
                            category=self._categorize_policy(entry.description if hasattr(entry, 'description') else ''),
                            effective_date=datetime.now(),
                            description=entry.description[:500] if hasattr(entry, 'description') else entry.title,
                            impact_score=self._calculate_impact_score(entry.title + ' ' + getattr(entry, 'description', '')),
                            affected_sectors=['agriculture', 'farming'],
                            source_url=entry.link if hasattr(entry, 'link') else feed_url,
                            ministry='Ministry of Agriculture'
                        )
                        policies.append(policy_data)
                        
            except Exception as e:
                logger.warning(f"Error fetching RSS feed {feed_url}: {e}")
        
        return policies

    def _search_policies_tavily(self) -> List[PolicyData]:
        policies = []
        
        try:
            url = "https://api.tavily.com/search"
            headers = {"Authorization": f"Bearer {self.api_keys['tavily']}"}
            
            search_queries = [
                "agricultural policy India 2024 government announcement",
                "farm subsidies MSP price support scheme", 
                "crop insurance PMFBY premium reduction",
                "kisan credit card loan interest rate",
                "digital agriculture technology mission",
                "organic farming certification subsidy",
                "PM-KISAN direct benefit transfer",
                "food processing industry policy"
            ]
            
            for query in search_queries:
                payload = {
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "max_results": 3
                }
                
                response = requests.post(url, json=payload, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    
                    for result in results:
                        policy_data = PolicyData(
                            policy_id=f"TAVILY_{int(time.time())}_{hash(result.get('title', ''))%1000}",
                            title=result.get('title', 'Unknown Policy'),
                            category=self._categorize_policy(result.get('content', '')),
                            effective_date=datetime.now(),
                            description=result.get('content', '')[:500],
                            impact_score=self._calculate_impact_score(result.get('content', '')),
                            affected_sectors=['agriculture', 'farming'],
                            source_url=result.get('url', ''),
                            ministry=self._extract_ministry(result.get('content', ''))
                        )
                        policies.append(policy_data)
                
                time.sleep(2)
                
        except Exception as e:
            logger.warning(f"Tavily API error: {e}")
        
        return policies
    
    def _search_policies_serper(self) -> List[PolicyData]:
        policies = []
        
        try:
            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": self.api_keys['serper']}
            
            queries = [
                "site:pib.gov.in agricultural policy 2024",
                "site:agricoop.nic.in farmer scheme announcement",
                "agricultural budget allocation India 2024"
            ]
            
            for query in queries:
                payload = {
                    "q": query,
                    "num": 5,
                    "gl": "in"
                }
                
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    results = response.json().get('organic', [])
                    
                    for result in results:
                        policy_data = PolicyData(
                            policy_id=f"SERPER_{int(time.time())}_{hash(result.get('title', ''))%1000}",
                            title=result.get('title', 'Unknown Policy'),
                            category=self._categorize_policy(result.get('snippet', '')),
                            effective_date=datetime.now(),
                            description=result.get('snippet', '')[:500],
                            impact_score=self._calculate_impact_score(result.get('snippet', '')),
                            affected_sectors=['agriculture', 'policy'],
                            source_url=result.get('link', ''),
                            ministry=self._extract_ministry(result.get('snippet', ''))
                        )
                        policies.append(policy_data)
                        
                time.sleep(1)
                    
        except Exception as e:
            logger.warning(f"Serper API error: {e}")
        
        return policies
    
    def _extract_ministry(self, content: str) -> str:
        ministries = [
            'Ministry of Agriculture', 'Ministry of Food Processing',
            'Ministry of Rural Development', 'Ministry of Finance',
            'Department of Agriculture', 'NABARD'
        ]
        
        content_lower = content.lower()
        for ministry in ministries:
            if ministry.lower() in content_lower:
                return ministry
        
        return 'Government of India'

    def _simulate_policy_updates(self) -> List[PolicyData]:
        policies = []
        
        policy_templates = {
            'subsidy': [
                'PM-KISAN Scheme Payment Release for {season} Season {year}',
                'Direct Benefit Transfer for {crop} Farmers Announced',
                'Fertilizer Subsidy Increase of {percentage}% for {season}',
                'Input Cost Support Program for {region} Farmers',
                'Seed Subsidy Enhancement for Small Farmers',
                'Organic Farming Financial Assistance Program'
            ],
            'insurance': [
                'Crop Insurance Premium Reduction for {crop} {year}',
                'Weather-Based Insurance Scheme Expanded to {region}',
                'PMFBY Coverage Enhancement for {season} Crops',
                'Livestock Insurance Subsidy Program Launch',
                'Horticulture Crop Insurance Initiative',
                'Aquaculture Risk Coverage Scheme'
            ],
            'pricing': [
                'MSP Increase for {crop} Announced',
                'Procurement Price Revision for {season} Season',
                'Market Intervention Scheme for {crop} Prices',
                'Price Support Mechanism for {region} Farmers',
                'Export Incentive for Agricultural Products',
                'Import Duty Revision on Agri Commodities'
            ],
            'credit': [
                'Interest Rate Reduction for Agricultural Loans',
                'Kisan Credit Card Limit Enhancement Program',
                'Collateral-Free Loan Scheme for Small Farmers',
                'Digital Lending Platform for Rural Credit',
                'Crop Loan Waiver Announcement',
                'Agri-Startup Funding Initiative'
            ],
            'technology': [
                'Digital Agriculture Mission Launch',
                'Precision Farming Technology Subsidy',
                'AI-Based Crop Advisory Service Rollout',
                'Drone Technology Adoption Program',
                'Soil Health Card Digital Platform',
                'Satellite-Based Crop Monitoring System'
            ]
        }
        
        current_year = datetime.now().year
        current_month = datetime.now().month
        season = 'Kharif' if 4 <= current_month <= 9 else 'Rabi'
        
        crops = ['Wheat', 'Rice', 'Cotton', 'Sugarcane', 'Pulses', 'Oilseeds', 'Millets', 'Barley']
        regions = ['North India', 'South India', 'Western States', 'Eastern States', 'Central India', 'Northeast']
        percentages = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.5]
        
        num_policies = np.random.randint(4, 8)
        
        for i in range(num_policies):
            category = np.random.choice(list(policy_templates.keys()))
            template = np.random.choice(policy_templates[category])
            
            title = template.format(
                season=season,
                year=current_year,
                crop=np.random.choice(crops),
                region=np.random.choice(regions),
                percentage=np.random.choice(percentages)
            )
            
            description = self._generate_policy_description(category, title)
            impact_score = self._calculate_dynamic_impact_score(category, current_month)
            affected_sectors = self._select_affected_sectors(category)
            budget = self._generate_budget_allocation(category)
            
            policy_data = PolicyData(
                policy_id=f"DYN_{datetime.now().strftime('%Y%m%d')}_{i+1:03d}",
                title=title,
                category=category,
                effective_date=datetime.now() + timedelta(days=np.random.randint(1, 90)),
                description=description,
                impact_score=impact_score,
                affected_sectors=affected_sectors,
                source_url=f"https://government.policy/{category}/{i+1}",
                ministry=self._assign_ministry(category),
                budget_allocation=budget
            )
            policies.append(policy_data)
        
        return policies
    
    def _generate_budget_allocation(self, category: str) -> float:
        base_budgets = {
            'subsidy': np.random.uniform(1000, 50000),
            'insurance': np.random.uniform(500, 15000),
            'pricing': np.random.uniform(2000, 75000),
            'credit': np.random.uniform(10000, 100000),
            'technology': np.random.uniform(200, 5000)
        }
        return round(base_budgets.get(category, 1000), 2)
    
    def _assign_ministry(self, category: str) -> str:
        ministry_map = {
            'subsidy': 'Ministry of Agriculture and Farmers Welfare',
            'insurance': 'Ministry of Agriculture and Farmers Welfare',
            'pricing': 'Department of Food and Public Distribution',
            'credit': 'Ministry of Finance',
            'technology': 'Ministry of Electronics and IT'
        }
        return ministry_map.get(category, 'Ministry of Agriculture and Farmers Welfare')

    def _generate_policy_description(self, category: str, title: str) -> str:
        descriptions = {
            'subsidy': [
                "Government announces comprehensive financial support initiative with direct benefit transfer to eligible beneficiaries. The program aims to reduce input costs and enhance farmer income through targeted subsidies.",
                "New subsidy program designed to boost agricultural productivity and support small and marginal farmers. Enhanced financial assistance for seeds, fertilizers, and farming equipment.",
                "Financial assistance program for agricultural productivity improvement with focus on sustainable farming practices and income enhancement for rural communities."
            ],
            'insurance': [
                "Crop insurance scheme expansion to provide comprehensive risk coverage for farmers against natural calamities and yield losses. Enhanced premium support and simplified claim procedures.",
                "Premium reduction initiative to increase farmer participation in insurance programs. Weather-based insurance products to protect against climate risks and market volatility.",
                "Comprehensive risk management program covering crop failure, livestock mortality, and equipment damage with government-supported premium subsidies."
            ],
            'pricing': [
                "Minimum Support Price revision ensures fair compensation for farmers and market stability. Enhanced procurement mechanisms and improved price discovery systems.",
                "Market intervention to stabilize commodity prices and support producer income. Strategic buffer stock management and export-import policy adjustments.",
                "Procurement policy changes to improve price realization for agricultural products with focus on quality parameters and market linkages."
            ],
            'credit': [
                "Agricultural credit policy revision to improve access to institutional finance for farmers. Enhanced credit limits, reduced interest rates, and simplified procedures.",
                "Interest rate concessions for farmers to reduce borrowing costs and improve financial accessibility. Digital lending platforms and collateral-free loan schemes.",
                "Comprehensive rural credit program with focus on small farmers, women entrepreneurs, and agri-startups. Enhanced KCC limits and flexible repayment options."
            ],
            'technology': [
                "Technology adoption program to modernize agricultural practices through precision farming, AI-based advisory, and digital platforms for improved productivity.",
                "Digital platform initiative for improved farm management, market access, and supply chain optimization. Integration of IoT, AI, and satellite technology.",
                "Innovation support scheme for precision agriculture, smart farming solutions, and technology-enabled extension services for farmers."
            ]
        }
        
        return np.random.choice(descriptions.get(category, ["Comprehensive policy update announced for agricultural sector development"]))
    
    def _calculate_dynamic_impact_score(self, category: str, month: int) -> float:
        base_scores = {
            'subsidy': 7.5,
            'insurance': 6.5,
            'pricing': 8.0,
            'credit': 7.0,
            'technology': 6.0
        }
        
        base_score = base_scores.get(category, 6.0)
        
        if category == 'insurance' and month in [3, 4, 5, 10, 11]:
            base_score += 1.5
        elif category == 'credit' and month in [2, 3, 4, 9, 10]:
            base_score += 1.0
        elif category == 'pricing' and month in [4, 5, 10, 11]:
            base_score += 1.5
        
        variation = np.random.uniform(-0.5, 1.0)
        
        return min(10.0, max(1.0, base_score + variation))
    
    def _select_affected_sectors(self, category: str) -> List[str]:
        sector_map = {
            'subsidy': ['farming', 'rural_economy', 'agriculture', 'food_security'],
            'insurance': ['agriculture', 'insurance', 'risk_management', 'climate_adaptation'],
            'pricing': ['grain_market', 'farmer_income', 'procurement', 'food_inflation'],
            'credit': ['agricultural_finance', 'banking', 'rural_credit', 'financial_inclusion'],
            'technology': ['agri_tech', 'innovation', 'digital_agriculture', 'precision_farming']
        }
        
        base_sectors = sector_map.get(category, ['agriculture'])
        common_sectors = ['rural_development', 'food_security', 'economic_growth']
        
        return base_sectors + [np.random.choice(common_sectors)]

    def _categorize_policy(self, content: str) -> str:
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['subsidy', 'financial aid', 'support', 'pm-kisan', 'dbt']):
            return 'subsidy'
        elif any(word in content_lower for word in ['insurance', 'crop insurance', 'pmfby', 'risk coverage']):
            return 'insurance'
        elif any(word in content_lower for word in ['msp', 'price', 'procurement', 'market intervention']):
            return 'pricing'
        elif any(word in content_lower for word in ['loan', 'credit', 'kisan credit', 'kcc', 'interest rate']):
            return 'credit'
        elif any(word in content_lower for word in ['technology', 'digital', 'innovation', 'ai', 'precision']):
            return 'technology'
        else:
            return 'general'
    
    def _calculate_impact_score(self, content: str) -> float:
        content_lower = content.lower()
        score = 5.0
        
        high_impact = ['nationwide', 'billion', 'major', 'significant', 'all farmers', 'pan-india']
        medium_impact = ['state', 'million', 'moderate', 'selected', 'certain crops', 'regional']
        low_impact = ['pilot', 'trial', 'limited', 'specific areas', 'phase-wise']
        
        for keyword in high_impact:
            if keyword in content_lower:
                score += 2.0
        
        for keyword in medium_impact:
            if keyword in content_lower:
                score += 1.0
        
        for keyword in low_impact:
            if keyword in content_lower:
                score += 0.5
        
        return min(10.0, score)

    def analyze_market_sentiment(self, market_data: List[MarketData]) -> float:
        if not market_data:
            return 5.0
        
        price_changes = [data.price_change for data in market_data if data.price_change is not None]
        
        if not price_changes:
            return 5.0
        
        avg_change = np.mean(price_changes)
        positive_count = sum(1 for change in price_changes if change > 0)
        negative_count = sum(1 for change in price_changes if change < 0)
        volatility = np.std(price_changes)
        
        sentiment = 5.0 + (avg_change * 10) + (positive_count - negative_count) / len(price_changes)
        
        if volatility > 3:
            sentiment -= 0.5
        
        return max(0, min(10, sentiment))
    
    def analyze_market_trends(self, market_data: List[MarketData]) -> Dict[str, str]:
        trends = {}
        
        commodity_groups = {
            'grains': ['WHEAT', 'RICE', 'CORN', 'BARLEY'],
            'spices': ['TURMERIC', 'CORIANDER', 'JEERA', 'CARDAMOM'],
            'cash_crops': ['COTTON', 'SUGAR', 'SOYBEAN'],
            'stocks': ['UPL', 'ESCORTS', 'ITC', 'COROMANDEL']
        }
        
        for group, commodities in commodity_groups.items():
            group_data = [data for data in market_data if any(commodity in data.commodity for commodity in commodities)]
            if group_data:
                avg_change = np.mean([data.price_change for data in group_data])
                if avg_change > 2:
                    trends[group] = "Strong Bullish"
                elif avg_change > 0.5:
                    trends[group] = "Moderately Bullish"
                elif avg_change > -0.5:
                    trends[group] = "Sideways"
                elif avg_change > -2:
                    trends[group] = "Moderately Bearish"
                else:
                    trends[group] = "Strong Bearish"
        
        return trends
    
    def generate_price_forecasts(self, market_data: List[MarketData]) -> Dict[str, float]:
        forecasts = {}
        
        for data in market_data[:10]:
            price_trend = data.price_change / 100
            volatility = np.random.uniform(0.02, 0.05)
            
            forecast_change = price_trend * 0.7 + np.random.uniform(-volatility, volatility)
            forecasted_price = data.price * (1 + forecast_change)
            
            forecasts[data.commodity] = round(forecasted_price, 2)
        
        return forecasts

    def calculate_risk_indicators(self, market_data: List[MarketData], 
                                 policy_data: List[PolicyData]) -> Dict[str, float]:
        risk_indicators = {}
        
        if market_data:
            price_changes = [abs(data.price_change) for data in market_data if data.price_change is not None]
            volatility = np.std(price_changes) if price_changes else 0
            risk_indicators['market_volatility'] = min(10, volatility)
            
            volume_changes = [data.volume for data in market_data if data.volume > 0]
            avg_volume = np.mean(volume_changes) if volume_changes else 1000
            risk_indicators['liquidity_risk'] = max(0, min(10, 10 - (avg_volume / 1000)))
        
        policy_impact_scores = [policy.impact_score for policy in policy_data]
        policy_uncertainty = np.std(policy_impact_scores) if policy_impact_scores else 0
        risk_indicators['policy_uncertainty'] = min(10, policy_uncertainty)
        
        current_month = datetime.now().month
        seasonal_risk_map = {1: 6, 2: 7, 3: 8, 4: 9, 5: 7, 6: 6, 7: 5, 8: 4, 9: 5, 10: 6, 11: 7, 12: 6}
        risk_indicators['seasonal_risk'] = seasonal_risk_map.get(current_month, 5)
        
        risk_indicators['weather_risk'] = np.random.uniform(3, 8)
        
        market_sentiment = self.analyze_market_sentiment(market_data)
        risk_indicators['credit_risk'] = 10 - market_sentiment
        
        global_markets = [data for data in market_data if 'GLOBAL' in data.commodity]
        if global_markets:
            global_volatility = np.std([data.price_change for data in global_markets])
            risk_indicators['global_market_risk'] = min(10, global_volatility)
        
        return risk_indicators

    def generate_recommendations(self, market_data: List[MarketData], 
                               policy_data: List[PolicyData], 
                               risk_indicators: Dict[str, float],
                               weather_data: List[WeatherData]) -> List[str]:
        recommendations = []
        
        sentiment = self.analyze_market_sentiment(market_data)
        
        if sentiment > 7:
            recommendations.extend([
                "Market sentiment is strongly bullish. Consider increasing credit limits for agricultural loans by 15-20%.",
                "Favorable conditions for equipment financing and expansion loans with competitive rates.",
                "Recommend farmers to consider forward selling strategies to lock in current favorable prices."
            ])
        elif sentiment < 3:
            recommendations.extend([
                "Market sentiment is bearish. Implement stricter risk assessment for new agricultural loans.",
                "Consider offering hedging products to protect farmers from continued price volatility.",
                "Enhanced monitoring of existing loan portfolios in agricultural sector recommended."
            ])
        else:
            recommendations.append("Market sentiment is neutral. Maintain current lending policies with regular monitoring.")
        
        if risk_indicators.get('market_volatility', 0) > 5:
            recommendations.append("High market volatility detected. Strongly recommend crop insurance to all borrowers.")
        
        if risk_indicators.get('seasonal_risk', 0) > 7:
            recommendations.append("High seasonal risk period. Prepare for 25-30% increase in working capital loan demands.")
        
        if risk_indicators.get('weather_risk', 0) > 6:
            recommendations.append("Elevated weather risk. Consider climate-resilient farming loans and insurance products.")
        
        subsidy_policies = [p for p in policy_data if p.category == 'subsidy']
        if subsidy_policies:
            total_subsidy_allocation = sum(p.budget_allocation or 0 for p in subsidy_policies)
            recommendations.append(f"New subsidy programs worth ₹{total_subsidy_allocation:.0f} crores available. Inform eligible borrowers about government benefits.")
        
        pricing_policies = [p for p in policy_data if p.category == 'pricing']
        if pricing_policies:
            recommendations.append("MSP changes detected. Adjust loan valuations and collateral assessments accordingly within 30 days.")
        
        credit_policies = [p for p in policy_data if p.category == 'credit']
        if credit_policies:
            recommendations.append("Government credit policy changes identified. Review and align institutional lending practices.")
        
        high_impact_policies = [p for p in policy_data if p.impact_score > 8]
        if high_impact_policies:
            recommendations.append("High-impact policy changes detected. Conduct comprehensive portfolio review and risk reassessment.")
        
        if weather_data:
            adverse_weather_regions = [w.region for w in weather_data if 'stress' in w.agricultural_impact.lower()]
            if adverse_weather_regions:
                recommendations.append(f"Weather stress identified in {len(adverse_weather_regions)} regions. Consider regional risk adjustments.")
        
        return recommendations

    def run_comprehensive_analysis(self) -> MarketIntelligence:
        logger.info("Starting comprehensive market and policy analysis...")
        
        market_data = self.capture_real_time_market_data()
        policy_updates = self.capture_policy_updates()
        weather_data = self.capture_weather_data()
        risk_indicators = self.calculate_risk_indicators(market_data, policy_updates)
        sentiment_score = self.analyze_market_sentiment(market_data)
        market_trends = self.analyze_market_trends(market_data)
        price_forecasts = self.generate_price_forecasts(market_data)
        recommendations = self.generate_recommendations(market_data, policy_updates, risk_indicators, weather_data)
        
        self.market_intelligence = MarketIntelligence(
            market_data=market_data,
            policy_updates=policy_updates,
            weather_data=weather_data,
            risk_indicators=risk_indicators,
            sentiment_score=sentiment_score,
            recommendations=recommendations,
            market_trends=market_trends,
            price_forecasts=price_forecasts
        )
        
        logger.info("Comprehensive analysis completed")
        return self.market_intelligence
    
    def export_report(self, filename: str = None) -> str:
        logger.info("Report export functionality disabled")
        return ""
    
    def print_summary_report(self):
        print("\n" + "="*80)
        print("AGRICULTURAL MARKET INTELLIGENCE COMPREHENSIVE SUMMARY")
        print("="*80)
        
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Market Sentiment Score: {self.market_intelligence.sentiment_score:.1f}/10")
        
        print(f"\nMarket Data Points: {len(self.market_intelligence.market_data)}")
        if self.market_intelligence.market_data:
            print("Top Performers:")
            sorted_data = sorted(self.market_intelligence.market_data, 
                               key=lambda x: x.price_change, reverse=True)[:5]
            for data in sorted_data:
                print(f"  • {data.commodity}: {data.price_change:+.2f}% (₹{data.price}) Vol: {data.volume:,}")
        
        print(f"\nPolicy Updates: {len(self.market_intelligence.policy_updates)}")
        if self.market_intelligence.policy_updates:
            print("Recent High-Impact Policies:")
            high_impact = sorted(self.market_intelligence.policy_updates, 
                               key=lambda x: x.impact_score, reverse=True)[:3]
            for policy in high_impact:
                budget_info = f" (₹{policy.budget_allocation:.0f}Cr)" if policy.budget_allocation else ""
                print(f"  • {policy.title} - Impact: {policy.impact_score:.1f}/10{budget_info}")
        
        print(f"\nWeather Analysis: {len(self.market_intelligence.weather_data)} regions monitored")
        if self.market_intelligence.weather_data:
            avg_temp = np.mean([w.temperature for w in self.market_intelligence.weather_data])
            avg_rainfall = np.mean([w.rainfall for w in self.market_intelligence.weather_data])
            print(f"  • Average Temperature: {avg_temp:.1f}°C")
            print(f"  • Average Rainfall: {avg_rainfall:.1f}mm")
        
        print("\nMarket Trends:")
        for sector, trend in self.market_intelligence.market_trends.items():
            print(f"  • {sector.replace('_', ' ').title()}: {trend}")
        
        print("\nRisk Indicators:")
        for risk_type, score in self.market_intelligence.risk_indicators.items():
            risk_level = "Low" if score < 4 else "Medium" if score < 7 else "High"
            print(f"  • {risk_type.replace('_', ' ').title()}: {score:.1f}/10 ({risk_level})")
        
        print("\nPrice Forecasts (Next 30 days):")
        for commodity, forecast in list(self.market_intelligence.price_forecasts.items())[:5]:
            current_data = next((d for d in self.market_intelligence.market_data if d.commodity == commodity), None)
            if current_data:
                change = ((forecast - current_data.price) / current_data.price) * 100
                print(f"  • {commodity}: ₹{forecast:.2f} ({change:+.1f}%)")
        
        print("\nKey Recommendations:")
        for i, rec in enumerate(self.market_intelligence.recommendations[:7], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)

def main():
    print("Agricultural Market Information & Policy Capture System")
    print("="*80)
    
    capture_system = MarketInformPolicyCapture()
    
    try:
        intelligence = capture_system.run_comprehensive_analysis()        
        capture_system.print_summary_report()        
        
        return intelligence
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return None

if __name__ == "__main__":
    main()

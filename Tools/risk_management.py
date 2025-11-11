import numpy as np
import requests
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RiskMetric:
    metric_name: str
    current_value: float
    risk_level: str
    confidence: float

class RealTimeDataFetcher:
    def __init__(self):
        self.api_keys = {
            'tavily': os.getenv('TAVILY_API_KEY'),
            'serper': os.getenv('SERPER_API_KEY'), 
            'google': os.getenv('GOOGLE_API_KEY')
        }

    def fetch_complete_portfolio_data(self, commodity: str) -> Dict:
        portfolio_data = {"primary_commodity": commodity}

        market_data = self._fetch_market_indicators(commodity)
        weather_data = self._fetch_weather_indicators(commodity)
        financial_data = self._fetch_financial_indicators(commodity)
        operational_data = self._fetch_operational_indicators(commodity)

        portfolio_data.update(market_data)
        portfolio_data.update(weather_data)
        portfolio_data.update(financial_data)
        portfolio_data.update(operational_data)
        
        return portfolio_data

    def _fetch_market_indicators(self, commodity: str) -> Dict:
        try:
            if self.api_keys['serper']:
                headers = {'X-API-KEY': self.api_keys['serper'], 'Content-Type': 'application/json'}
                payload = {
                    'q': f"{commodity} commodity price volatility elasticity market analysis {datetime.now().year}",
                    'num': 8,
                    'type': 'search'
                }
                
                response = requests.post('https://google.serper.dev/search', 
                                       headers=headers, json=payload, timeout=15)
                
                if response.status_code == 200:
                    return self._parse_market_indicators(response.json())
            
            if self.api_keys['tavily']:
                headers = {'Content-Type': 'application/json'}
                payload = {
                    'api_key': self.api_keys['tavily'],
                    'query': f"{commodity} price volatility demand elasticity market trends",
                    'search_depth': 'advanced',
                    'max_results': 8
                }
                
                response = requests.post('https://api.tavily.com/search',
                                       headers=headers, json=payload, timeout=15)
                
                if response.status_code == 200:
                    return self._parse_tavily_market_data(response.json())
                    
        except Exception as e:
            print(f"Market data fetch error: {e}")
            
        return self._get_default_market_data(commodity)

    def _fetch_weather_indicators(self, commodity: str) -> Dict:
        try:
            if self.api_keys['serper']:
                headers = {'X-API-KEY': self.api_keys['serper'], 'Content-Type': 'application/json'}
                payload = {
                    'q': f"{commodity} climate precipitation temperature agriculture weather risk {datetime.now().year}",
                    'num': 6,
                    'type': 'search'
                }
                
                response = requests.post('https://google.serper.dev/search', 
                                       headers=headers, json=payload, timeout=15)
                
                if response.status_code == 200:
                    return self._parse_weather_indicators(response.json())
                    
        except Exception as e:
            print(f"Weather data fetch error: {e}")
            
        return self._get_default_weather_data(commodity)

    def _fetch_financial_indicators(self, commodity: str) -> Dict:
        try:
            if self.api_keys['serper']:
                headers = {'X-API-KEY': self.api_keys['serper'], 'Content-Type': 'application/json'}
                payload = {
                    'q': f"{commodity} agriculture debt ratio liquidity farming finance analysis",
                    'num': 5,
                    'type': 'search'
                }
                
                response = requests.post('https://google.serper.dev/search', 
                                       headers=headers, json=payload, timeout=15)
                
                if response.status_code == 200:
                    return self._parse_financial_indicators(response.json())
                    
        except Exception as e:
            print(f"Financial data fetch error: {e}")
            
        return self._get_default_financial_data(commodity)

    def _fetch_operational_indicators(self, commodity: str) -> Dict:
        try:
            if self.api_keys['serper']:
                headers = {'X-API-KEY': self.api_keys['serper'], 'Content-Type': 'application/json'}
                payload = {
                    'q': f"{commodity} supply chain technology agriculture operations efficiency",
                    'num': 5,
                    'type': 'search'
                }
                
                response = requests.post('https://google.serper.dev/search', 
                                       headers=headers, json=payload, timeout=15)
                
                if response.status_code == 200:
                    return self._parse_operational_indicators(response.json())
                    
        except Exception as e:
            print(f"Operational data fetch error: {e}")
            
        return self._get_default_operational_data(commodity)

    def _parse_market_indicators(self, results: Dict) -> Dict:
        volatility_score = 0.5
        elasticity_score = -1.2
        
        for result in results.get('organic', []):
            content = (result.get('snippet', '') + ' ' + result.get('title', '')).lower()
            
            if any(term in content for term in ['high volatility', 'very volatile', 'extremely volatile']):
                volatility_score = min(0.8, volatility_score + 0.15)
            elif any(term in content for term in ['moderate volatility', 'stable']):
                volatility_score = max(0.2, volatility_score - 0.1)
            elif any(term in content for term in ['low volatility', 'steady']):
                volatility_score = max(0.1, volatility_score - 0.2)
                
            if any(term in content for term in ['inelastic', 'price insensitive']):
                elasticity_score = max(-0.8, elasticity_score + 0.3)
            elif any(term in content for term in ['elastic', 'price sensitive']):
                elasticity_score = min(-2.0, elasticity_score - 0.3)
        
        return {
            'volatility_index': min(1.0, max(0.0, volatility_score)),
            'price_elasticity': max(-3.0, min(-0.5, elasticity_score))
        }

    def _parse_tavily_market_data(self, results: Dict) -> Dict:
        volatility_indicators = []
        elasticity_indicators = []
        
        for result in results.get('results', []):
            content = result.get('content', '').lower()
            
            if 'volatility' in content:
                if 'high' in content or 'extreme' in content:
                    volatility_indicators.append(0.7)
                elif 'moderate' in content:
                    volatility_indicators.append(0.4)
                elif 'low' in content:
                    volatility_indicators.append(0.2)
                    
        avg_volatility = np.mean(volatility_indicators) if volatility_indicators else 0.45
        
        return {
            'volatility_index': min(1.0, max(0.0, avg_volatility)),
            'price_elasticity': -1.5
        }

    def _parse_weather_indicators(self, results: Dict) -> Dict:
        precipitation_cv = 0.3
        temp_anomaly = 1.5
        geo_diversity = 0.5
        
        for result in results.get('organic', []):
            content = (result.get('snippet', '') + ' ' + result.get('title', '')).lower()
            
            if any(term in content for term in ['drought', 'dry conditions', 'water stress']):
                precipitation_cv = min(0.6, precipitation_cv + 0.1)
            elif any(term in content for term in ['flood', 'excessive rain', 'waterlogged']):
                precipitation_cv = min(0.6, precipitation_cv + 0.08)
                
            if any(term in content for term in ['temperature extreme', 'heat wave', 'cold snap']):
                temp_anomaly = min(3.0, temp_anomaly + 0.3)
                
            if any(term in content for term in ['regional', 'diversified', 'multiple locations']):
                geo_diversity = min(0.8, geo_diversity + 0.1)
        
        return {
            'precipitation_cv': precipitation_cv,
            'temp_anomaly_std': temp_anomaly,
            'geographic_diversity': geo_diversity
        }

    def _parse_financial_indicators(self, results: Dict) -> Dict:
        debt_ratio = 0.4
        liquidity_ratio = 1.3
        
        for result in results.get('organic', []):
            content = (result.get('snippet', '') + ' ' + result.get('title', '')).lower()
            
            if any(term in content for term in ['high debt', 'heavily leveraged', 'debt burden']):
                debt_ratio = min(0.7, debt_ratio + 0.1)
            elif any(term in content for term in ['low debt', 'debt-free', 'minimal leverage']):
                debt_ratio = max(0.1, debt_ratio - 0.1)
                
            if any(term in content for term in ['cash flow', 'liquidity crisis', 'working capital']):
                if 'strong' in content or 'positive' in content:
                    liquidity_ratio = min(2.0, liquidity_ratio + 0.2)
                elif 'weak' in content or 'negative' in content:
                    liquidity_ratio = max(0.8, liquidity_ratio - 0.2)
        
        return {
            'debt_ratio': debt_ratio,
            'liquidity_ratio': liquidity_ratio
        }

    def _parse_operational_indicators(self, results: Dict) -> Dict:
        supply_complexity = 0.5
        tech_resilience = 0.7
        
        for result in results.get('organic', []):
            content = (result.get('snippet', '') + ' ' + result.get('title', '')).lower()
            
            if any(term in content for term in ['complex supply chain', 'multiple suppliers', 'global supply']):
                supply_complexity = min(0.8, supply_complexity + 0.1)
            elif any(term in content for term in ['simple supply', 'local supply', 'direct supply']):
                supply_complexity = max(0.2, supply_complexity - 0.1)
                
            if any(term in content for term in ['digital', 'automation', 'precision agriculture', 'AI']):
                tech_resilience = min(0.9, tech_resilience + 0.1)
            elif any(term in content for term in ['traditional', 'manual', 'outdated']):
                tech_resilience = max(0.3, tech_resilience - 0.1)
        
        return {
            'supply_complexity': supply_complexity,
            'tech_resilience': tech_resilience
        }

    def _get_default_market_data(self, commodity: str) -> Dict:
        defaults = {
            'wheat': {'volatility_index': 0.45, 'price_elasticity': -1.8},
            'corn': {'volatility_index': 0.52, 'price_elasticity': -1.5},
            'soybeans': {'volatility_index': 0.48, 'price_elasticity': -1.6},
            'rice': {'volatility_index': 0.38, 'price_elasticity': -1.2},
            'cotton': {'volatility_index': 0.55, 'price_elasticity': -2.0}
        }
        return defaults.get(commodity.lower(), {'volatility_index': 0.45, 'price_elasticity': -1.5})

    def _get_default_weather_data(self, commodity: str) -> Dict:
        return {
            'precipitation_cv': 0.35,
            'temp_anomaly_std': 2.1,
            'geographic_diversity': 0.6
        }

    def _get_default_financial_data(self, commodity: str) -> Dict:
        return {
            'debt_ratio': 0.42,
            'liquidity_ratio': 1.35
        }

    def _get_default_operational_data(self, commodity: str) -> Dict:
        return {
            'supply_complexity': 0.55,
            'tech_resilience': 0.75
        }

class AgriculturalRiskCalculator:
    def __init__(self):
        self.weights = {
            'market': 0.35,
            'weather': 0.30,
            'financial': 0.25,
            'operational': 0.10
        }
        self.data_fetcher = RealTimeDataFetcher()
        
    def _sigmoid_transform(self, x: float, k: float = 10, x0: float = 0.5) -> float:
        return 1 / (1 + np.exp(-k * (x - x0)))
    
    def _calculate_market_risk(self, portfolio: Dict) -> float:
        volatility = portfolio.get('volatility_index', 0.5)
        price_elasticity = portfolio.get('price_elasticity', -1.2)
        
        normalized_elasticity = min(1.0, abs(price_elasticity) / 2.0)
        market_exposure = 0.7 * volatility + 0.3 * normalized_elasticity
        
        return self._sigmoid_transform(market_exposure, k=8, x0=0.4)
    
    def _calculate_weather_risk(self, portfolio: Dict) -> float:
        precipitation_variance = portfolio.get('precipitation_cv', 0.3)
        temperature_anomaly = portfolio.get('temp_anomaly_std', 1.5)
        
        climate_stress = np.sqrt(
            (precipitation_variance ** 2 + (temperature_anomaly / 3.0) ** 2) / 2
        )
        
        geographic_factor = 1.0 - portfolio.get('geographic_diversity', 0.5) * 0.3
        weather_risk = climate_stress * geographic_factor
        
        return min(1.0, weather_risk)
    
    def _calculate_financial_risk(self, portfolio: Dict) -> float:
        debt_ratio = portfolio.get('debt_ratio', 0.4)
        liquidity_ratio = portfolio.get('liquidity_ratio', 1.2)
        
        leverage_risk = self._sigmoid_transform(debt_ratio, k=12, x0=0.6)
        liquidity_risk = max(0, (1.5 - liquidity_ratio) / 1.5)
        
        financial_risk = np.sqrt(leverage_risk ** 2 + liquidity_risk ** 2)
        return min(1.0, financial_risk)
    
    def _calculate_operational_risk(self, portfolio: Dict) -> float:
        supply_complexity = portfolio.get('supply_complexity', 0.5)
        tech_resilience = portfolio.get('tech_resilience', 0.7)
        
        complexity_penalty = supply_complexity ** 1.5
        tech_bonus = 1.0 - tech_resilience * 0.4
        
        operational_risk = 0.6 * complexity_penalty + 0.4 * tech_bonus
        return min(1.0, operational_risk)
    
    def _calculate_overall_risk(self, individual_risks: Dict) -> float:
        risk_vector = np.array([
            individual_risks['market'],
            individual_risks['weather'], 
            individual_risks['financial'],
            individual_risks['operational']
        ])
        
        weight_vector = np.array([
            self.weights['market'],
            self.weights['weather'],
            self.weights['financial'], 
            self.weights['operational']
        ])
        
        correlation_matrix = np.array([
            [1.00, 0.25, 0.40, 0.15],
            [0.25, 1.00, 0.20, 0.30],
            [0.40, 0.20, 1.00, 0.35],
            [0.15, 0.30, 0.35, 1.00]
        ])
        
        portfolio_variance = np.dot(weight_vector, np.dot(correlation_matrix, 
                                   weight_vector * risk_vector ** 2))
        
        return min(1.0, np.sqrt(portfolio_variance))
    
    def _get_risk_level(self, risk_score: float) -> str:
        thresholds = [(0.2, "Very Low"), (0.4, "Low"), (0.6, "Moderate"), 
                     (0.8, "High"), (1.0, "Critical")]
        
        for threshold, level in thresholds:
            if risk_score <= threshold:
                return level
        return "Critical"
    
    def _calculate_confidence(self, portfolio: Dict, risk_type: str) -> float:
        data_quality_scores = {
            'market': 0.85,
            'weather': 0.75,
            'financial': 0.80,
            'operational': 0.80,
            'overall': 0.82
        }
        
        return data_quality_scores.get(risk_type, 0.75)
    
    def calculate_risk_metrics(self, commodity: str) -> List[RiskMetric]:
        portfolio = self.data_fetcher.fetch_complete_portfolio_data(commodity)
        individual_risks = {
            'market': self._calculate_market_risk(portfolio),
            'weather': self._calculate_weather_risk(portfolio),
            'financial': self._calculate_financial_risk(portfolio),
            'operational': self._calculate_operational_risk(portfolio)
        }
        
        overall_risk = self._calculate_overall_risk(individual_risks)
        
        metrics = [
            RiskMetric(
                metric_name="Overall Portfolio Risk",
                current_value=overall_risk,
                risk_level=self._get_risk_level(overall_risk),
                confidence=self._calculate_confidence(portfolio, 'overall')
            ),
            RiskMetric(
                metric_name="Market Risk",
                current_value=individual_risks['market'],
                risk_level=self._get_risk_level(individual_risks['market']),
                confidence=self._calculate_confidence(portfolio, 'market')
            ),
            RiskMetric(
                metric_name="Weather Risk",
                current_value=individual_risks['weather'],
                risk_level=self._get_risk_level(individual_risks['weather']),
                confidence=self._calculate_confidence(portfolio, 'weather')
            ),
            RiskMetric(
                metric_name="Financial Risk",
                current_value=individual_risks['financial'],
                risk_level=self._get_risk_level(individual_risks['financial']),
                confidence=self._calculate_confidence(portfolio, 'financial')
            ),
            RiskMetric(
                metric_name="Operational Risk",
                current_value=individual_risks['operational'],
                risk_level=self._get_risk_level(individual_risks['operational']),
                confidence=self._calculate_confidence(portfolio, 'operational')
            )
        ]
        
        return metrics

def get_agricultural_risk_metrics(primary_commodity: str) -> Dict:
    calculator = AgriculturalRiskCalculator()
    
    try:
        risk_metrics = calculator.calculate_risk_metrics(primary_commodity)

        return {
            "status": "success",
            "commodity": primary_commodity,
            "timestamp": datetime.now().isoformat(),
            "risk_metrics": [
                {
                    "name": metric.metric_name,
                    "value": round(metric.current_value, 3),
                    "level": metric.risk_level,
                    "confidence": round(metric.confidence, 3)
                }
                for metric in risk_metrics
            ]
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    commodity = "wheat"
    result = get_agricultural_risk_metrics(commodity)
        
    print(f"Status: {result['status']}")
    print(f"Commodity: {result.get('commodity', 'N/A')}")
    print(f"Timestamp: {result['timestamp']}")
    print("\nRisk Metrics:")
    for metric in result.get("risk_metrics", []):
        print(f"  {metric['name']}: {metric['value']} ({metric['level']}) - Confidence: {metric['confidence']}")

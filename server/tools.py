import httpx
import logging
import re
# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tool 1 - Fetch public IP address
async def get_public_ip() -> str:
    """Fetches the public IP address of the machine using api.ipify.org."""
    try:
        url = 'https://api.ipify.org?format=json'
        logger.info(f"Fetching public IP from {url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout = 5.0)
            response.raise_for_status()
            ip  = response.json().get("ip", "")
            logger.info(f"Successfully retrieved IP: {ip}")
            return ip
    except httpx.HTTPError as e:
        logger.error(f"Error fetching public IP: {e}")
        raise RuntimeError(f"Failed to retrieve public IP: {e}")
    
# Tool 2 - Get Geo-location from IP address
async def get_location_from_ip(ip_address: str) -> dict:
    """Retrieves the latitude and longitude for a given IP address using ip-api.com."""

    # Input validation: Check if IP address format is valid (IPv4 or IPv6)
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$'
        

    if not (re.match(ipv4_pattern, ip_address) or re.match(ipv6_pattern, ip_address)):
        error_msg = f"Invalid IP address format: {ip_address}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
        # Additional IPv4 validation: each octet must be 0-255
    if re.match(ipv4_pattern, ip_address):
       octets = [int(octet) for octet in ip_address.split('.')]
           
       if any(octet > 255 for octet in octets):
          error_msg = f"Invalid IPv4 address (octet out of range): {ip_address}"
          logger.error(error_msg)
          raise ValueError(error_msg)
           
    url = f"http://ip-api.com/json/{ip_address}"
    logger.info(f"Fetching location for IP: {ip_address}")
    try:
       async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=5.0)
        response.raise_for_status()
        data = response.json()
               
        if data.get("status") == "fail":
           error_msg = data.get("message", "Unknown error")
           logger.error(f"IP-API returned fail: {error_msg}")
           raise ValueError(f"Could not resolve location for IP {ip_address}: {error_msg}")
        else:  
           lat = data.get("lat")
           lon = data.get("lon")
           logger.info(f"Location found: {lat}, {lon}")
           return {"latitude": lat, "longitude": lon}
                    
    except Exception as e:
            logger.error(f"Error fetching location: {e}")
            raise RuntimeError(f"Failed to fetch location data: {e}")
    

# Tool 3 - Get Weather Data using Latitude and Longitude - https://open-meteo.com/en/docs
async def get_weather_forecast(latitude: float, longitude: float) -> str:
    """
    Fetches the current weather forecast for the given coordinates using Open-Meteo.
    Documentation used https://open-meteo.com/en/docs
    """
    # Input validation: It will verify latitude and longitude are within valid ranges
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        error_msg = f"Latitude and longitude must be numeric values"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not (-90 <= latitude <= 90):
        error_msg = f"Invalid latitude: {latitude}. Must be between -90 and 90"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not (-180 <= longitude <= 180):
        error_msg = f"Invalid longitude: {longitude}. Must be between -180 and 180"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true"
    }
    logger.info(f"Fetching weather for {latitude}, {longitude}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=5.0)
            response.raise_for_status()
            data = response.json()
            current_weather = data.get("current_weather", {})
            
            # Formatting a nice string for the LLM to digest
            weather_desc = (
                f"Temperature: {current_weather.get('temperature')} C, "
                f"Windspeed: {current_weather.get('windspeed')} km/h"
            )
            logger.info(f"Weather retrieved: {weather_desc}")
            return weather_desc
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        raise RuntimeError(f"Failed to get weather data: {e}")
import math
import requests
import numpy as np

# convert from miles to meters
def miles_to_meters(miles):
    """convert miles to meters"""
    return miles * 1609.34

def haversine_distance(lat1, lon1, lat2, lon2):
    """calculate the great circle distance between two points in meters"""
    R = 6371000  # Earth radius in meters
    
    # convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance


def get_location_info(place_name):
    """Get location info from Nominatim API"""
    print(f"Getting location info for '{place_name}'...")
    nominatim_url = f"https://nominatim.openstreetmap.org/search?q={place_name}&format=json&limit=1"
    
    headers = {'User-Agent': 'RadioPropagationSimulator/1.0'}
    
    response = requests.get(nominatim_url, headers=headers)
    response.raise_for_status()
    
    location_data = response.json()
    
    if not location_data:
        raise RuntimeError(f"Location '{place_name}' not found")
    
    location = location_data[0]
    lat = float(location["lat"])
    lon = float(location["lon"])
    display_name = location["display_name"]
    
    return {
        "lat": lat,
        "lon": lon,
        "name": display_name
    }

def calculate_bounding_box(lat, lon, radius_m):
    """calculate the bounding box given a center point and radius in meters"""
    # convert radius from meters to degrees of latitude
    lat_offset = radius_m / 111111
    
    # longitude offset varies with latitude
    lon_offset = radius_m / (111111 * math.cos(math.radians(lat)))
    
    # calculate the bounding box
    north = lat + lat_offset
    south = lat - lat_offset
    east = lon + lon_offset
    west = lon - lon_offset
    
    return {
        "north": north,
        "south": south,
        "east": east,
        "west": west
    }

# thermal noise power
def thermal_noise_power(bandwidth_mhz, noise_figure_db=8):
    """calculate thermal noise power in dBm"""
    # Boltzmann's constant
    k = 1.38e-23  # Joules/Kelvin
    
    # standard temperature in Kelvin
    T = 290
    
    # bandwidth in Hz
    B = bandwidth_mhz * 1e6
    
    # thermal noise power in watts
    N = k * T * B
    
    # convert to dBm
    N_dbm = 10 * math.log10(N * 1000) + noise_figure_db
    
    return N_dbm

def calculate_free_space_path_loss(distance_m, frequency_mhz):
    """
    Calculate free space path loss using the Friis formula
    
    Parameters:
    distance_m (float): Distance in meters
    frequency_mhz (float): Frequency in MHz
    
    Returns:
    float: Path loss in dB
    """
    # convert frequency to Hz   
    frequency_hz = frequency_mhz * 1e6
    
    # safeguard against zero distance
    distance_m = max(1.0, distance_m)
    
    # speed of light in meters per second
    c = 3e8
    
    # calculate wavelength
    wavelength = c / frequency_hz
    
    # calculate path loss
    # FSPL = (4πd/λ)² from paper 
    path_loss_linear = (4 * math.pi * distance_m / wavelength) ** 2
    
    # convert to dB
    path_loss_db = 10 * math.log10(path_loss_linear)
    
    return path_loss_db

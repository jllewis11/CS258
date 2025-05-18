import os
import numpy as np
import folium
import math
import json
import time
import tempfile
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import LinearSegmentedColormap
from branca.element import MacroElement, Template

try:
    from dotenv import load_dotenv
    load_dotenv()  
    env_available = True
except ImportError:
    env_available = False


from utils import miles_to_meters, haversine_distance, get_location_info
from utils import calculate_bounding_box, thermal_noise_power
from utils import calculate_free_space_path_loss

from osm_terrain_handler import fetch_osm_data, fetch_terrain_data
from osm_terrain_handler import create_clutter_raster, create_indoor_mask

from bs_optimizer import BaseStationOptimizer, tr38901_path_loss, calculate_5g_propagation

try:
    from llm_orchestrator import LLMOrchestrator
    llm_available = True
except ImportError:
    llm_available = False

# these configurations are based upon the paper's implementation
DEFAULT_CONFIG = {

    "location_name": "Tracy Hills, California, USA",
    "radius_miles": 0.5,            # radius of study area in miles
    "grid_resolution_m": 5,         # paper's maps use 5 m pixels
    
    # network settings
    "carrier_frequency_mhz": 5000,  # 5 GHz
    "bandwidth_mhz": 80,            # 80 MHz channel (paper's value)
    "tx_power_dbm": 30,             # transmit power in dBm (1 W = 30 dBm)
    "tx_height_m": 25,              # transmitter height (meters)
    "rx_height_m": 1.5,             # receiver height (meters)
    "noise_figure_db": 8,           # receiver noise figure
    "coverage_threshold_db": 5,     # minimum SINR for coverage
    
    # base station configuration
    "num_base_stations": 2,         # number of base stations (matches paper's seed)
    "min_base_stations": 2,         # minimum number of base stations to use during optimization
    "max_base_stations": 5,         # maximum number of base stations to use during optimization
    
    # default base station parameters (used when automatically creating base stations)
    "default_bs_height_m": 25,      # default height in meters
    "default_bs_power_dbm": 30,     # default power in dBm
    "default_bs_downtilt": 5,       # default antenna downtilt in degrees
    
    # propagation model settings
    "indoor_penalty_db": [5, 15],   # min/max indoor penetration loss range
    
    # optimization settings
    "optimize_bs_placement": True,  # automatically optimize BS placement
    "optimization_trials": 3,     # number of optimization trials (let's Optuna breathe)
    "optimization_metric": "balanced_coverage",  # metric to optimize (balanced_coverage, coverage_percent, avg_sinr, etc.)
    
    # LLM orchestration
    "use_llm_orchestration": False,  # use LLM to orchestrate the simulation
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),  # get OpenAI API key from environment
    "llm_model": "o4-mini-2025-04-16",  # LLM model to use (gpt-4, gpt-3.5-turbo, etc.)
    
    # output settings
    "output_html": "radio_map.html",
    "output_directory": "results"
}


def initialize_base_stations(center_lat, center_lon, radius_m, config):
    """
    Automatically initialize base stations within the study area
    
    Parameters:
    center_lat (float): Center latitude of the study area
    center_lon (float): Center longitude of the study area
    radius_m (float): Radius of the study area in meters
    config (dict): Configuration dictionary
    
    Returns:
    list: List of base stations [lat, lon, height_m, power_dbm, name, downtilt]
    """
    # get number of base stations from config
    num_stations = config.get("num_base_stations", 3)
    
    print(f"Initializing {num_stations} base stations...")
    
    base_stations = []
    
    # get default parameters from config
    default_height = config.get("default_bs_height_m", 25)
    default_power = config.get("default_bs_power_dbm", 30)
    default_downtilt = config.get("default_bs_downtilt", 5)
    
    # calculate bounding box
    bbox = calculate_bounding_box(center_lat, center_lon, radius_m)
    
    for i in range(num_stations):
        # create a simple grid-based placement with some randomness
        # this divides the study area roughly into equal parts plus some random offset
        
        if num_stations == 1:
            # if only one station, place it at the center
            bs_lat = center_lat
            bs_lon = center_lon
        else:
            # otherwise distribute in a grid pattern with randomness
            grid_size = int(np.ceil(np.sqrt(num_stations)))
            row = i // grid_size
            col = i % grid_size
            
            # calculate grid cell position (0-1 range)
            lat_pos = (row + 0.5) / grid_size
            lon_pos = (col + 0.5) / grid_size
            
            # add small random offset (±20% of a cell)
            lat_pos += (np.random.random() - 0.5) * 0.4 / grid_size
            lon_pos += (np.random.random() - 0.5) * 0.4 / grid_size
            
            # convert to actual coordinates
            lat_range = bbox["north"] - bbox["south"]
            lon_range = bbox["east"] - bbox["west"]
            
            bs_lat = bbox["south"] + lat_pos * lat_range
            bs_lon = bbox["west"] + lon_pos * lon_range
        
        # create a name for the base station
        bs_name = f"BS {i+1}"
        
        # create the base station entry
        base_station = [bs_lat, bs_lon, default_height, default_power, bs_name, default_downtilt]
        base_stations.append(base_station)
        
        print(f"  Created {bs_name} at ({bs_lat:.6f}, {bs_lon:.6f})")
    
    return base_stations

def calculate_enhanced_path_loss(center_lat, center_lon, base_stations, bbox, grid_resolution_m, 
                               osm_data, terrain_data, config):
    """
    Calculate path loss using enhanced models with terrain and building data

    Parameters:
    center_lat (float): Center latitude
    center_lon (float): Center longitude
    base_stations (list): List of base stations [lat, lon, height_m, power_dbm, name, downtilt]
    bbox (dict): Bounding box with north, south, east, west
    grid_resolution_m (float): Grid resolution in meters
    osm_data (dict): Dictionary with OSM data
    terrain_data (numpy.ndarray): Terrain elevation grid
    config (dict): Configuration dictionary

    Returns:
    tuple: (path_loss_list, grid_lats, grid_lons, grid_shape, indoor_mask)
    """
    print("Calculating enhanced path loss...")
    
    # determine if we should use 5G model based on frequency
    use_5g_model = True
    
    # if using 5G model, simply delegate to the calculate_5g_propagation function
    if use_5g_model:
        # make sure config has the necessary parameters
        if "frequency" not in config:
            # use carrier_frequency_mhz from config, but convert to GHz for TR38901
            config["frequency"] = config["carrier_frequency_mhz"] / 1000
        
        if "environment" not in config:
            # default to urban environment if not specified
            config["environment"] = "urban"
            
        if "frequency_band" not in config:
            # determine frequency band based on frequency
            freq_mhz = config["carrier_frequency_mhz"]
            if freq_mhz < 1000:  # < 1 GHz
                config["frequency_band"] = "low"
            elif freq_mhz < 6000:  # < 6 GHz
                config["frequency_band"] = "mid"
            else:  # >= 6 GHz
                config["frequency_band"] = "high"
                
        # use the TR38901 model through calculate_5g_propagation
        print(f"  Using 3GPP TR 38.901 model with {config['frequency']:.2f} GHz ({config['frequency_band']}-band) in {config['environment']} environment")
        return calculate_5g_propagation(
            center_lat, center_lon, base_stations, bbox, 
            grid_resolution_m, osm_data, terrain_data, config
        )
    
    # fall back to original implementation if not using 5G model
    # (keeping this as fallback just in case, but it shouldn't be used)
    print("  Warning: Using legacy path loss model. Consider using TR38901 model instead.")
    
    # calculate grid dimensions
    lat_range = bbox["north"] - bbox["south"]
    lon_range = bbox["east"] - bbox["west"]
    
    # convert grid_resolution_m to degrees
    lat_resolution = grid_resolution_m / 111111  # Approx meters per degree latitude
    lon_resolution = grid_resolution_m / (111111 * math.cos(math.radians(center_lat)))  # Adjust for longitude
    
    # calculate number of grid points
    lat_steps = max(2, int(np.ceil(lat_range / lat_resolution)))
    lon_steps = max(2, int(np.ceil(lon_range / lon_resolution)))
    
    # create grid using linspace
    grid_lats = np.linspace(bbox["south"], bbox["north"], lat_steps)
    grid_lons = np.linspace(bbox["west"], bbox["east"], lon_steps)
    
    print(f"Grid dimensions: {lat_steps}x{lon_steps} ({lat_steps*lon_steps} points)")
    
    # create clutter raster for path loss calculations
    grid_shape = (lat_steps, lon_steps)
    clutter_raster = create_clutter_raster(osm_data, bbox, grid_shape)
    
    # create indoor/outdoor mask
    indoor_mask = create_indoor_mask(osm_data, bbox, grid_shape)
    
    # initialize path loss grid for each base station
    path_loss_list = []
    
    # for each base station
    for bs_idx, bs in enumerate(base_stations):
        print(f"  Calculating path loss for base station {bs_idx+1}/{len(base_stations)}: {bs[4]}")
        
        # extract base station parameters
        bs_lat, bs_lon = bs[0], bs[1]
        bs_height = bs[2]
        bs_power = bs[3]
        bs_name = bs[4]
        bs_downtilt = bs[5] if len(bs) > 5 else 0
        
        # initialize path loss grid for this base station
        path_loss = np.zeros((lat_steps, lon_steps))
        
        # calculate path loss for each grid point
        for i, lat in enumerate(grid_lats):
            for j, lon in enumerate(grid_lons):
                # calculate distance in meters
                dx = haversine_distance(bs_lat, bs_lon, lat, lon)
                
                # get terrain height at this point
                # in a real implementation, you would interpolate from the terrain data
                if terrain_data is not None:
                    terrain_height = terrain_data[min(i, terrain_data.shape[0]-1), min(j, terrain_data.shape[1]-1)]
                else:
                    terrain_height = 0
                
                # calculate effective heights
                eff_tx_height = bs_height
                eff_rx_height = config["rx_height_m"] + terrain_height
                
                # calculate base path loss using appropriate model
                if config["carrier_frequency_mhz"] > 1500:
                    # for higher frequencies, use more appropriate model than free space
                    # here we're using a simplified approach with free space as base
                    base_pl = calculate_free_space_path_loss(dx, config["carrier_frequency_mhz"])

                
                # get clutter attenuation at this point
                clutter_attenuation = clutter_raster[i, j]
                
                # calculate antenna pattern effects (downtilt)
                # simple model: maximum gain in main beam direction, reduced gain elsewhere
                dx_m = haversine_distance(bs_lat, bs_lon, lat, lon)
                dh_m = terrain_height + config["rx_height_m"] - (bs_height)
                angle_degrees = math.degrees(math.atan2(dh_m, dx_m))
                
                # calculate gain reduction due to downtilt
                # simple model: 3dB reduction per 3 degrees off main beam
                tilt_loss = min(20, 3 * abs(angle_degrees - (-bs_downtilt)) / 3)
                
                # apply indoor penetration loss if this is an indoor point
                indoor_loss = 0
                if indoor_mask[i, j] == 1:
                    # random indoor loss between min and max values
                    min_loss, max_loss = config["indoor_penalty_db"]
                    indoor_loss = min_loss + (max_loss - min_loss) * np.random.random()
                
                # combine all loss factors
                total_pl = base_pl + clutter_attenuation + tilt_loss + indoor_loss
                
                path_loss[i, j] = total_pl
        
        path_loss_list.append(path_loss)
    
    return path_loss_list, grid_lats, grid_lons, (lat_steps, lon_steps), indoor_mask


def calculate_sinr(path_loss_list, base_stations, config):
    """
    Calculate SINR grid from path loss grids
    
    Parameters:
    path_loss_list (list): List of path loss grids for each base station
    base_stations (list): List of base stations 
    config (dict): Configuration dictionary
    
    Returns:
    numpy.ndarray: SINR grid in dB
    """
    print("calculating SINR grid...")
    
    # grid shape (assuming all path loss grids have the same shape)
    grid_shape = path_loss_list[0].shape
    
    # calculate noise power
    noise_power_dbm = thermal_noise_power(config["bandwidth_mhz"], config["noise_figure_db"])
    
    # special case for single base station (no interference, only SNR)
    if len(base_stations) == 1:
        print("  Single base station detected - calculating SNR (no interference)")
        # get base station power
        bs_power_dbm = base_stations[0][3]
        
        # calculate received power (P_rx = P_tx - L)
        received_power_dbm = bs_power_dbm - path_loss_list[0]
        
        # convert noise power and received power from dBm to mW for linear addition
        noise_power_mw = 10 ** (noise_power_dbm / 10)
        received_power_mw = 10 ** (received_power_dbm / 10)
        
        # calculate SNR (Signal to Noise Ratio)
        snr = received_power_mw / noise_power_mw
        
        # convert back to dB
        sinr_db = 10 * np.log10(snr)
        
        return sinr_db
    
    # standard case with multiple base stations - calculate SINR
    # initialize arrays for signal and interference
    signal_power = np.zeros(grid_shape)  # power in mW
    interference_power = np.zeros(grid_shape)  # power in mW
    
    # initialize array to track best serving cell for each pixel
    best_server_idx = np.zeros(grid_shape, dtype=int)
    best_server_power = -np.inf * np.ones(grid_shape)
    
    # calculate received power for each base station
    for bs_idx, (bs, path_loss) in enumerate(zip(base_stations, path_loss_list)):
        # get base station power
        bs_power_dbm = bs[3]
        
        # calculate received power (P_rx = P_tx - L)
        received_power_dbm = bs_power_dbm - path_loss
        
        # convert from dBm to mW for linear addition
        received_power_mw = 10 ** (received_power_dbm / 10)
        
        # update best server if this station provides stronger signal
        better_server_mask = received_power_dbm > best_server_power
        best_server_idx[better_server_mask] = bs_idx
        best_server_power[better_server_mask] = received_power_dbm[better_server_mask]
    
    # now calculate signal and interference based on best server
    for bs_idx, (bs, path_loss) in enumerate(zip(base_stations, path_loss_list)):
        # get base station power
        bs_power_dbm = bs[3]
        
        # calculate received power (P_rx = P_tx - L)
        received_power_dbm = bs_power_dbm - path_loss
        
        # convert from dBm to mW for linear addition
        received_power_mw = 10 ** (received_power_dbm / 10)
        
        # mask for pixels where this station is the best server
        is_best_server = (best_server_idx == bs_idx)
        
        # add power to signal or interference accordingly
        signal_power[is_best_server] = received_power_mw[is_best_server]
        interference_power[~is_best_server] += received_power_mw[~is_best_server]
    
    # convert noise power from dBm to mW
    noise_power_mw = 10 ** (noise_power_dbm / 10)
    
    # calculate SINR (Signal to Interference plus Noise Ratio)
    # SINR = S / (I + N)
    sinr = signal_power / (interference_power + noise_power_mw)
    
    # convert to dB
    sinr_db = 10 * np.log10(sinr)
    
    return sinr_db


def create_sinr_image(sinr_db, output_path="sinr_map.png", title="SINR (dB)", png_background_color=None):
    """
    Create a colored image of the SINR grid
    
    Parameters:
    sinr_db (numpy.ndarray): SINR grid in dB
    output_path (str): Output path for the image
    title (str): Title for the image
    png_background_color: Background color (None for transparent)
    
    Returns:
    str: Path to the created image
    """
    print(f"Creating {title} image...")
    
    # create a custom colormap for SINR
    # low SINR (bad) -> high SINR (good)
    # purple -> blue -> teal -> green -> yellow
    colors = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
    cmap = LinearSegmentedColormap.from_list("sinr_cmap", colors)
    
    # clip SINR values to a reasonable range for better visualization
    sinr_min = -10  # dB
    sinr_max = 30   # dB
    
    # handle NaN values (e.g., indoor mask)
    sinr_masked = np.ma.masked_invalid(sinr_db)
    
    # clip to range
    sinr_clipped = np.clip(sinr_masked, sinr_min, sinr_max)
    
    # create a figure with a specific aspect ratio
    height, width = sinr_clipped.shape
    plt.figure(figsize=(10, 10 * height / width))
    
    # plot the SINR grid with correct orientation
    plt.imshow(
        sinr_clipped,
        cmap=cmap,
        vmin=sinr_min,
        vmax=sinr_max,
        origin="lower",  # ensure correct orientation (south at bottom)
        interpolation="bicubic",  # smoother interpolation
        aspect="auto"  # maintain aspect ratio
    )
    
    # remove axes for cleaner overlay
    plt.axis('off')
    
    # add a colorbar
    cbar = plt.colorbar(label=title)
    
    # add a title
    plt.title(title)
    
    # save the figure with tight borders and high resolution
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=(png_background_color is None))
    plt.close()
    
    print(f"Image saved to {output_path}")
    return output_path

def create_coverage_map(sinr_db, grid_lats, grid_lons, base_stations, bbox, config, 
                       indoor_mask=None, output_path="radio_map.html"):
    """
    Create an enhanced interactive coverage map with Folium
    
    Parameters:
    sinr_db (numpy.ndarray): SINR grid in dB
    grid_lats (numpy.ndarray): Grid latitudes
    grid_lons (numpy.ndarray): Grid longitudes
    base_stations (list): List of base stations
    bbox (dict): Bounding box with north, south, east, west
    config (dict): Configuration dictionary
    indoor_mask (numpy.ndarray): Mask of indoor (1) vs outdoor (0) points
    output_path (str): Output path for the HTML map
    
    Returns:
    str: Path to the created map
    """
    print("Creating enhanced interactive coverage map...")
    
    # get center coordinates
    center_lat = (bbox["north"] + bbox["south"]) / 2
    center_lon = (bbox["east"] + bbox["west"]) / 2
    
    # calculate appropriate zoom level
    lat_range = bbox["north"] - bbox["south"]
    lon_range = bbox["east"] - bbox["west"]
    zoom_level = min(14, max(10, int(14 - np.log2(max(lat_range, lon_range) * 111111 / 500))))
    
    # create a Folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
        tiles="OpenStreetMap"
    )
    
    # define SINR value range for normalization
    sinr_min = -10  # dB
    sinr_max = 30   # dB
    
    # import HeatMap plugin
    from folium.plugins import HeatMap
    
    # check if arrays are 1D and reshape them if needed
    is_1d = len(sinr_db.shape) == 1
    if is_1d:
        # create meshgrid from 1D arrays
        lat_grid, lon_grid = np.meshgrid(grid_lats, grid_lons)
        grid_shape = lat_grid.shape
        
        # reshape 1D arrays to 2D
        sinr_2d = np.reshape(sinr_db, grid_shape)
        
        if indoor_mask is not None:
            indoor_2d = np.reshape(indoor_mask, grid_shape)
        else:
            indoor_2d = None
    else:
        # already 2D
        sinr_2d = sinr_db
        indoor_2d = indoor_mask
        grid_shape = sinr_db.shape
    
    # convert the SINR grid to heatmap format
    heatmap_data = []
    for i in range(len(grid_lats)):
        for j in range(len(grid_lons)):
            # clip and normalize SINR values
            sinr_value = np.clip(sinr_2d[j, i], sinr_min, sinr_max)  # Note: j, i for proper orientation
            intensity = (sinr_value - sinr_min) / (sinr_max - sinr_min)
            
            # skip NaN values (e.g. from buildings with no signal)
            if not np.isnan(intensity):
                heatmap_data.append([grid_lats[i], grid_lons[j], intensity])
    
    # create the heatmap layer for overall SINR
    heatmap = HeatMap(
        heatmap_data,
        name="SINR (dB)",
        min_opacity=0.4,  # reduced from 0.7 for better map visibility
        radius=int(config["grid_resolution_m"] * 1.5),  # adjust radius based on resolution, convert to int
        blur=int(config["grid_resolution_m"]),  # convert to int 
        gradient={
            "0.0": '#440154',  # purple (poor signal)
            "0.25": '#3b528b', # blue
            "0.5": '#21918c',  # teal
            "0.75": '#5ec962', # green
            "1.0": '#fde725'   # yellow (excellent signal)
        }
    )
    heatmap.add_to(m)
    
    # if we have indoor/outdoor mask, create a separate indoor heatmap
    if indoor_2d is not None:
        indoor_heatmap_data = []
        for i in range(len(grid_lats)):
            for j in range(len(grid_lons)):
                # only include indoor points
                if indoor_2d[j, i] == 1:  # Note: j, i for proper orientation
                    sinr_value = np.clip(sinr_2d[j, i], sinr_min, sinr_max)
                    intensity = (sinr_value - sinr_min) / (sinr_max - sinr_min)
                    if not np.isnan(intensity):
                        indoor_heatmap_data.append([grid_lats[i], grid_lons[j], intensity])
        
        # create indoor-only heatmap layer
        indoor_heatmap = HeatMap(
            indoor_heatmap_data,
            name="Indoor SINR",
            min_opacity=0.3,  # reduced opacity for better map visibility
            radius=int(config["grid_resolution_m"] * 1.5),  # convert to int
            blur=int(config["grid_resolution_m"]),  # convert to int
            gradient={
                "0.0": '#440154',  # purple (poor signal)
                "0.25": '#3b528b', # blue
                "0.5": '#21918c',  # teal
                "0.75": '#5ec962', # green
                "1.0": '#fde725'   # yellow (excellent signal)
            }
        )
        indoor_heatmap.add_to(m)
    
    # add base stations as a separate layer
    bs_group = folium.FeatureGroup(name="Base Stations")
    for bs in base_stations:
        bs_lat, bs_lon, bs_height, bs_power, bs_name = bs[:5]
        bs_downtilt = bs[5] if len(bs) > 5 else 0
        
        # create a more detailed popup
        popup_html = f"""
        <div style="min-width: 200px;">
            <h4>{bs_name}</h4>
            <table>
                <tr><td>Position:</td><td>{bs_lat:.6f}, {bs_lon:.6f}</td></tr>
                <tr><td>Height:</td><td>{bs_height} m</td></tr>
                <tr><td>Power:</td><td>{bs_power} dBm</td></tr>
                <tr><td>Downtilt:</td><td>{bs_downtilt}°</td></tr>
                <tr><td>Frequency:</td><td>{config['carrier_frequency_mhz']} MHz</td></tr>
            </table>
        </div>
        """
        
        # create a more visible marker for base stations
        folium.Marker(
            [bs_lat, bs_lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=bs_name,
            icon=folium.Icon(color="red", icon="antenna", prefix='fa')
        ).add_to(bs_group)
        
        # calculate a more realistic coverage radius based on frequency and power
        # lower frequency and higher power = larger coverage radius
        # this is a simplified model for visualization only
        # base radius is 300m at 5000 MHz and 30 dBm
        base_radius = 300
        # frequency factor: lower frequency travels further (roughly inversely proportional)
        freq_factor = 5000 / max(100, config['carrier_frequency_mhz'])
        # power factor: higher power travels further (logarithmic relationship)
        power_factor = 10 ** ((bs_power - 30) / 20)  # 6dB increase = 2x distance
        

        coverage_radius = base_radius * freq_factor * power_factor

    
    bs_group.add_to(m)
    
    # add a bounding box to show the study area
    box_coords = [
        [bbox["north"], bbox["west"]],
        [bbox["north"], bbox["east"]],
        [bbox["south"], bbox["east"]],
        [bbox["south"], bbox["west"]],
        [bbox["north"], bbox["west"]]
    ]
    folium.Polygon(
        locations=box_coords,
        color="blue",
        weight=3,
        fill=False,
        popup="Study Area",
        tooltip="Study Area"
    ).add_to(m)
    
    # add a custom Branca element legend instead of built-in legend
    # this provides more control over placement and doesn't block the map
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; 
        right: 50px; 
        width: 120px;
        height: 180px; 
        z-index: 1000;
        background-color: white; 
        padding: 10px; 
        border: 2px solid grey; 
        border-radius: 5px;
        opacity: 0.8;
    ">
        <p style="margin-top: 0; margin-bottom: 5px;"><b>SINR (dB)</b></p>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: #440154; width: 20px; height: 20px; margin-right: 5px;"></div>
            <span>-10</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: #3b528b; width: 20px; height: 20px; margin-right: 5px;"></div>
            <span>0</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: #21918c; width: 20px; height: 20px; margin-right: 5px;"></div>
            <span>10</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: #5ec962; width: 20px; height: 20px; margin-right: 5px;"></div>
            <span>20</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="background-color: #fde725; width: 20px; height: 20px; margin-right: 5px;"></div>
            <span>30</span>
        </div>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # add information about the simulation
    info_html = f"""
    <div style="
        position: fixed; 
        top: 10px; 
        right: 10px; 
        z-index: 1000; 
        background-color: white; 
        padding: 10px; 
        border: 2px solid grey; 
        border-radius: 5px;
        opacity: 0.9;
        max-width: 300px;
    ">
        <h4>Simulation Info</h4>
        <p>Location: {config['location_name']}</p>
        <p>Frequency: {config['carrier_frequency_mhz']} MHz</p>
        <p>Bandwidth: {config['bandwidth_mhz']} MHz</p>
        <p>Resolution: {config['grid_resolution_m']} m</p>
        <p>Base Stations: {len(base_stations)}</p>
        <p>Coverage Threshold: {config['coverage_threshold_db']} dB</p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(info_html))
    
    # add layer control
    folium.LayerControl().add_to(m)
    
    # create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # save the map
    m.save(output_path)
    print(f"Interactive map created at {os.path.abspath(output_path)}")
    
    return os.path.abspath(output_path)


def run_simulation(config=None, command_handlers=None, run_optimization=False):
    """
    Run the enhanced radio propagation simulation
    
    Parameters:
    config (dict): Configuration dictionary
    command_handlers (dict): Custom command handlers for LLM orchestration
    run_optimization (bool): Whether to run BS optimization
    
    Returns:
    dict: Simulation results
    """
    # use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG
    
    print("Starting enhanced radio propagation simulation...")
    print(f"Study area: {config['location_name']}")
    
    # initialize results dict
    results = {}
    
    # check if using LLM orchestration
    if config.get("use_llm_orchestration", False) and llm_available:
        print("Using LLM orchestration...")
        orchestrator = LLMOrchestrator(config)
        
        # define workflow steps
        workflow = ["import_osm"]
        if run_optimization or config.get("optimize_bs_placement", False):
            workflow.append("optimize_bs")
        workflow.append("generate_radio_map")
        
        # if no custom handlers, use our default simulation flow
        if command_handlers is None:
            command_handlers = {
                "import_osm": lambda state: handle_import_osm(state, config),
                "optimize_bs": lambda state: handle_optimize_bs(state, config),
                "generate_radio_map": lambda state: handle_generate_map(state, config)
            }
        
        # Run the simulation through the LLM orchestrator
        simulation_state = orchestrator.run_simulation(workflow, command_handlers=command_handlers)
        
        # Generate an explanation of the results
        if "sinr_db" in simulation_state:
            explanation = orchestrator.explain_results(simulation_state)
            simulation_state["explanation"] = explanation
        
        return simulation_state
    
    # if not using LLM orchestration, run the simulation directly
    return run_direct_simulation(config, run_optimization)

def handle_import_osm(state, config):
    """Handler for importing OSM data (used with LLM orchestration)"""
    print("Importing OSM data...")
    
    # get location information
    location_info = get_location_info(config['location_name'])
    center_lat = location_info['lat']
    center_lon = location_info['lon']
    location_name = location_info['name']
    
    # calculate radius in meters
    radius_m = miles_to_meters(config['radius_miles'])
    
    # calculate bounding box
    bbox = calculate_bounding_box(center_lat, center_lon, radius_m)
    
    # fetch OSM data
    osm_data = fetch_osm_data(center_lat, center_lon, radius_m)
    
    # fetch terrain data
    terrain_data = fetch_terrain_data(bbox)
    
    # initialize base stations if not explicitly defined in config
    if "base_stations" not in config:
        # initialize base stations using num_base_stations from config
        base_stations = initialize_base_stations(
            center_lat, center_lon, radius_m, config
        )
        # add base stations to state so other functions can use them
        state["base_stations"] = base_stations
    elif "base_stations" in config:
        # use explicitly defined base stations if provided
        state["base_stations"] = config["base_stations"]
    
    # update state
    state["center_lat"] = center_lat
    state["center_lon"] = center_lon
    state["radius_m"] = radius_m
    state["bbox"] = bbox
    state["osm_data"] = osm_data
    state["terrain_data"] = terrain_data
    state["location_name"] = location_name
    
    return "OSM data imported successfully", state

def handle_optimize_bs(state, config):
    """Handler for optimizing base stations (used with LLM orchestration)"""
    print("Optimizing base station placement...")
    
    # check if required data is available
    if not all(k in state for k in ["osm_data", "terrain_data", "center_lat", "center_lon", "bbox"]):
        return "Missing required data. Import OSM data first.", state
    
    # use base stations from state if available (from initialize_base_stations)
    if "base_stations" in state:
        base_stations = state["base_stations"]
    else:
        # fall back to config or create new ones if not defined
        if "base_stations" in config:
            base_stations = config["base_stations"]
        else:
            base_stations = initialize_base_stations(
                state["center_lat"], state["center_lon"], state["radius_m"], 
                config
            )
    
    # create a temporary config with the base stations for the optimizer
    temp_config = config.copy()
    temp_config["base_stations"] = base_stations
    
    # initialize the optimizer
    optimizer = BaseStationOptimizer(temp_config)
    
    # run the optimization
    optimized_config = optimizer.optimize(
        state["center_lat"],
        state["center_lon"],
        state["bbox"],
        config["grid_resolution_m"],
        state["osm_data"],
        state["terrain_data"],
        calculate_enhanced_path_loss,
        calculate_sinr
    )
    
    # update the state with optimized config
    state["optimized_config"] = optimized_config
    
    # also save optimization results
    state["optimization_results"] = optimizer.get_optimization_results()
    
    # export optimization plots
    os.makedirs(config["output_directory"], exist_ok=True)
    plot_paths = optimizer.export_optimization_plots(config["output_directory"])
    state["optimization_plots"] = plot_paths
    
    return "Base station optimization completed successfully", state

def handle_generate_map(state, config):
    """Handler for generating radio map (used with LLM orchestration)"""
    print("Generating radio propagation map...")
    
    # check if required data is available
    if not all(k in state for k in ["osm_data", "terrain_data", "center_lat", "center_lon", "bbox"]):
        return "Missing required data. Import OSM data first.", state
    
    # use optimized config if available
    if "optimized_config" in state:
        current_config = state["optimized_config"]
    else:
        current_config = config.copy()
        
        # use base stations from state if available (from initialize_base_stations)
        if "base_stations" in state:
            current_config["base_stations"] = state["base_stations"]
        # if base stations not in state or config, create them now
        elif "base_stations" not in current_config:
            base_stations = initialize_base_stations(
                state["center_lat"], state["center_lon"], state["radius_m"], 
                current_config
            )
            current_config["base_stations"] = base_stations
    
    # calculate path loss
    path_loss_list, grid_lats, grid_lons, grid_shape, indoor_mask = calculate_enhanced_path_loss(
        state["center_lat"],
        state["center_lon"],
        current_config["base_stations"],
        state["bbox"],
        current_config["grid_resolution_m"],
        state["osm_data"],
        state["terrain_data"],
        current_config
    )
    
    # calculate SINR
    sinr_db = calculate_sinr(path_loss_list, current_config["base_stations"], current_config)
    
    # save results to state
    state["path_loss_list"] = path_loss_list
    state["grid_lats"] = grid_lats
    state["grid_lons"] = grid_lons
    state["grid_shape"] = grid_shape
    state["indoor_mask"] = indoor_mask
    state["sinr_db"] = sinr_db
    
    # create output directory if it doesn't exist
    os.makedirs(current_config["output_directory"], exist_ok=True)
    
    # create map
    output_path = os.path.join(current_config["output_directory"], current_config["output_html"])
    map_path = create_coverage_map(
        sinr_db,
        grid_lats,
        grid_lons,
        current_config["base_stations"],
        state["bbox"],
        current_config,
        indoor_mask=indoor_mask,
        output_path=output_path
    )
    
    state["map_path"] = map_path
    
    # calculate coverage statistics
    coverage_threshold = current_config["coverage_threshold_db"]
    coverage_percent = np.mean(sinr_db >= coverage_threshold) * 100
    avg_sinr = np.mean(sinr_db)
    
    # calculate indoor and outdoor coverage if mask available
    if indoor_mask is not None:
        indoor_coverage = np.mean((sinr_db >= coverage_threshold) & (indoor_mask == 1)) * 100
        outdoor_coverage = np.mean((sinr_db >= coverage_threshold) & (indoor_mask == 0)) * 100
        state["indoor_coverage_percent"] = indoor_coverage
        state["outdoor_coverage_percent"] = outdoor_coverage
    
    # create water mask from natural features to calculate land-only coverage
    water_mask = np.zeros_like(indoor_mask)
    if 'natural' in state["osm_data"] and state["osm_data"]['natural'] is not None:
        try:
            # get water features from natural OSM data
            water_features = state["osm_data"]['natural'][state["osm_data"]['natural']['natural'] == 'water']
            
            if not water_features.empty:
                from shapely.geometry import Polygon
                
                # create water mask based on OSM water features
                for idx, feature in water_features.iterrows():
                    geom = feature.geometry
                    if geom.is_valid:
                        # convert to relative coordinates within bbox
                        if hasattr(geom, 'exterior'):
                            # for polygons, get coordinates
                            coords = list(geom.exterior.coords)
                            
                            # convert coordinates to grid indices
                            lat_idx_list = []
                            lon_idx_list = []
                            
                            for lon, lat in coords:
                                # convert lat/lon to relative position in grid
                                lat_ratio = (lat - state["bbox"]["south"]) / (state["bbox"]["north"] - state["bbox"]["south"])
                                lon_ratio = (lon - state["bbox"]["west"]) / (state["bbox"]["east"] - state["bbox"]["west"])
                                
                                # convert to grid indices
                                lat_idx = int(lat_ratio * grid_shape[0])
                                lon_idx = int(lon_ratio * grid_shape[1])
                                
                                # ensure indices are within bounds
                                lat_idx = max(0, min(lat_idx, grid_shape[0]-1))
                                lon_idx = max(0, min(lon_idx, grid_shape[1]-1))
                                
                                lat_idx_list.append(lat_idx)
                                lon_idx_list.append(lon_idx)
                            
                            # create a polygon from the points
                            if len(lat_idx_list) >= 3:  # need at least 3 points for a polygon
                                points = [(lon_idx_list[i], lat_idx_list[i]) for i in range(len(lat_idx_list))]
                                poly = Polygon(points)
                                
                                # create a binary mask of the polygon
                                from rasterio.features import rasterize
                                shapes = [(poly, 1)]
                                feature_mask = rasterize(shapes, out_shape=grid_shape)
                                
                                # add to water mask
                                water_mask = np.logical_or(water_mask, feature_mask)
        except Exception as e:
            print(f"  Warning: Could not create water mask: {e}")
    
    # calculate land-only coverage (excluding water areas)
    land_mask = np.logical_not(water_mask)
    land_coverage = np.mean((sinr_db >= coverage_threshold) & land_mask) * 100
    state["land_coverage_percent"] = land_coverage
    
    state["coverage_percent"] = coverage_percent
    state["avg_sinr_db"] = avg_sinr
    
    print(f"Coverage statistics:")
    print(f"  Overall coverage: {coverage_percent:.1f}%")
    if "indoor_coverage_percent" in state:
        print(f"  Indoor coverage: {state['indoor_coverage_percent']:.1f}%")
    if "outdoor_coverage_percent" in state:
        print(f"  Outdoor coverage: {state['outdoor_coverage_percent']:.1f}%")
    if "land_coverage_percent" in state:
        print(f"  Land-only coverage: {state['land_coverage_percent']:.1f}%")
    print(f"  Average SINR: {avg_sinr:.1f} dB")
    
    return f"Radio map generated successfully. Coverage: {coverage_percent:.1f}%, Land-only: {land_coverage:.1f}%, Avg SINR: {avg_sinr:.1f} dB", state

def run_direct_simulation(config, run_optimization=False):
    """
    Run simulation directly without LLM orchestration
    
    Parameters:
    config (dict): Configuration dictionary
    run_optimization (bool): Whether to run BS optimization
    
    Returns:
    dict: Simulation results
    """
    # create a copy of the config to avoid modifying the original
    config_copy = config.copy()
    
    # initialize results dict
    results = {}
    simulation_state = {}
    
    # import OSM data
    _, simulation_state = handle_import_osm(simulation_state, config_copy)
    
    # if base stations were initialized during import, add them to the config copy
    if "base_stations" in simulation_state:
        config_copy["base_stations"] = simulation_state["base_stations"]
    
    # run optimization if requested
    if run_optimization or config_copy.get("optimize_bs_placement", False):
        _, simulation_state = handle_optimize_bs(simulation_state, config_copy)
        results["optimized_config"] = simulation_state.get("optimized_config")
        results["optimization_results"] = simulation_state.get("optimization_results")
    
    # generate radio map
    _, simulation_state = handle_generate_map(simulation_state, config_copy)
    
    # combine results
    results.update(simulation_state)
    
    print("Simulation completed successfully!")
    print(f"Coverage: {simulation_state.get('coverage_percent', 0):.1f}%")
    print(f"Average SINR: {simulation_state.get('avg_sinr_db', 0):.1f} dB")
    
    # if indoor mask was used, show indoor/outdoor stats
    if "indoor_coverage_percent" in simulation_state:
        print(f"Indoor coverage: {simulation_state['indoor_coverage_percent']:.1f}%")
        print(f"Outdoor coverage: {simulation_state['outdoor_coverage_percent']:.1f}%")
    
    print(f"Map saved to: {simulation_state.get('map_path', '')}")
    
    return results

def main():
    """Run the simulation using the default configuration"""
    # use the configuration from DEFAULT_CONFIG
    config = DEFAULT_CONFIG.copy()
    
    # check if command-line arguments were provided
    import sys
    
    # process arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--optimize":
            print("Running with optimization enabled")
            run_simulation(config, run_optimization=True)
            return
        elif sys.argv[i] == "--interactive":
            print("Running in interactive mode")
            # create LLM orchestrator if available
            if llm_available:
                orchestrator = LLMOrchestrator(config)
                # define command handlers
                command_handlers = {
                    "import_osm": lambda state: handle_import_osm(state, config),
                    "optimize_bs": lambda state: handle_optimize_bs(state, config),
                    "generate_radio_map": lambda state: handle_generate_map(state, config)
                }
                # run interactive session
                orchestrator.run_interactive(command_handlers)
            else:
                print("Error: Interactive mode requires LLM orchestration module")
            return
        elif sys.argv[i] == "--llm-model" and i + 1 < len(sys.argv):
            # Set the LLM model
            config["llm_model"] = sys.argv[i + 1]
            config["use_llm_orchestration"] = True
            print(f"Using LLM model: {config['llm_model']}")
            i += 1  # Skip the next argument (the model name)
        elif sys.argv[i] == "--trials" and i + 1 < len(sys.argv):
            # Set number of optimization trials
            try:
                trials = int(sys.argv[i + 1])
                config["optimization_trials"] = max(1, trials)
                print(f"Setting optimization trials to: {config['optimization_trials']}")
                i += 1  # Skip the next argument
            except ValueError:
                print(f"Error: Invalid number of trials: {sys.argv[i+1]}")
                return
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            print("Usage: python cellular_network_simulator_enhanced.py "
                  "[--optimize] [--interactive] [--llm-model MODEL_NAME] [--trials NUM_TRIALS]")
            return
        i += 1
    
    # run standard simulation
    run_simulation(config)

if __name__ == "__main__":
    main() 
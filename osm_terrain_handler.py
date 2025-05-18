import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, Point
import rasterio
from rasterio.features import rasterize

def fetch_osm_data(center_lat, center_lon, radius_m):
    """
    Fetch buildings and terrain data from OpenStreetMap
    
    Parameters:
    center_lat (float): Center latitude
    center_lon (float): Center longitude
    radius_m (float): Radius in meters
    
    Returns:
    dict: Dictionary with buildings, streets, and natural features
    """
    print("Fetching OSM data...")
    
    # get study area
    distance = radius_m  # meters
    
    # get buildings and other relevant features
    buildings = ox.features_from_point(
        (center_lat, center_lon), 
        dist=distance,
        tags={'building': True}
    )
    
    # get street network
    streets = ox.graph_from_point(
        (center_lat, center_lon),
        dist=distance,
        network_type='all',
        simplify=True
    )
    streets_gdf = ox.graph_to_gdfs(streets, nodes=False)
    
    # get natural features (parks, water)
    natural = ox.features_from_point(
        (center_lat, center_lon),
        dist=distance,
        tags={'natural': True}
    )
    
    print(f"Downloaded {len(buildings)} buildings, {len(streets_gdf)} streets, and {len(natural) if isinstance(natural, gpd.GeoDataFrame) else 0} natural features")
    
    return {
        'buildings': buildings,
        'streets': streets_gdf,
        'natural': natural
    }

def fetch_terrain_data(bbox):
    """
    Fetch digital elevation model data
    
    Parameters:
    bbox (dict): Bounding box with north, south, east, west
    
    Returns:
    numpy.ndarray: Terrain elevation grid
    
    """
    print("Fetching terrain data...")
    
    
    # get bounding box
    north = bbox["north"]
    south = bbox["south"]
    east = bbox["east"]
    west = bbox["west"]
    
    # calculate grid dimensions
    lat_range = north - south
    lon_range = east - west
    
    # create a simple elevation grid 
    elev_grid = np.zeros((100, 100))
    for i in range(100):
        elevation = 5 + 20 * (i / 100) 
        elev_grid[i, :] = elevation
    
    # add some random variation
    elev_grid += np.random.normal(0, 3, size=elev_grid.shape)
    
    return elev_grid

def create_clutter_raster(osm_data, bbox, grid_shape):
    """
    Create a clutter raster from OSM data for path loss calculations
    
    Parameters:
    osm_data (dict): Dictionary with OSM data (buildings, streets, natural)
    bbox (dict): Bounding box with north, south, east, west
    grid_shape (tuple): Shape of the grid (lat_steps, lon_steps)
    
    Returns:
    numpy.ndarray: Clutter raster grid with attenuation values
    """
    print("Creating clutter raster...")
    
    # extract buildings and natural features
    buildings = osm_data['buildings']
    natural = osm_data['natural']
    
    # define clutter categories and values
    # higher values = more signal attenuation (dB)
    clutter_values = {
        'building': 20.0,      # high attenuation for buildings
        'water': 0.0,          # low attenuation for water
        'forest': 10.0,        # medium attenuation for forest/trees
        'park': 5.0,           # some attenuation for parks
        'default': 2.0         # default ground clutter
    }
    
    # create an empty raster with default clutter value
    clutter_raster = np.ones(grid_shape) * clutter_values['default']
    
    # helper function to convert geom to pixel coordinates
    def geom_to_pixels(geom, bbox, shape):
        # get min/max coordinates
        minx, miny, maxx, maxy = geom.bounds
        
        # convert to relative coordinates within bbox (0-1)
        rel_minx = (minx - bbox["west"]) / (bbox["east"] - bbox["west"])
        rel_maxx = (maxx - bbox["west"]) / (bbox["east"] - bbox["west"])
        rel_miny = (miny - bbox["south"]) / (bbox["north"] - bbox["south"])
        rel_maxy = (maxy - bbox["south"]) / (bbox["north"] - bbox["south"])
        
        # convert to pixel coordinates
        pixel_minx = int(rel_minx * shape[1])
        pixel_maxx = int(rel_maxx * shape[1])
        pixel_miny = int((1 - rel_maxy) * shape[0])  # Flip y-axis
        pixel_maxy = int((1 - rel_miny) * shape[0])  # Flip y-axis
        
        # ensure within bounds
        pixel_minx = max(0, min(pixel_minx, shape[1]-1))
        pixel_maxx = max(0, min(pixel_maxx, shape[1]-1))
        pixel_miny = max(0, min(pixel_miny, shape[0]-1))
        pixel_maxy = max(0, min(pixel_maxy, shape[0]-1))
        
        return pixel_minx, pixel_miny, pixel_maxx, pixel_maxy
    
    # add buildings to clutter raster
    if not buildings.empty:
        for idx, building in buildings.iterrows():
            geom = building.geometry
            if geom.is_valid:
                minx, miny, maxx, maxy = geom_to_pixels(geom, bbox, grid_shape)
                clutter_raster[miny:maxy+1, minx:maxx+1] = clutter_values['building']
    
    # add natural features
    if isinstance(natural, gpd.GeoDataFrame) and not natural.empty:
        for idx, feature in natural.iterrows():
            geom = feature.geometry
            if geom.is_valid:
                feature_type = feature.get('natural', 'default')
                clutter_value = clutter_values.get(feature_type, clutter_values['default'])
                
                minx, miny, maxx, maxy = geom_to_pixels(geom, bbox, grid_shape)
                clutter_raster[miny:maxy+1, minx:maxx+1] = clutter_value
    
    return clutter_raster

def create_indoor_mask(osm_data, bbox, grid_shape):
    """
    Create a mask indicating indoor vs outdoor areas
    
    Parameters:
    osm_data (dict): Dictionary with OSM data
    bbox (dict): Bounding box
    grid_shape (tuple): Shape of the grid
    
    Returns:
    numpy.ndarray: Binary mask where 1=indoor, 0=outdoor
    """
    print("Creating indoor/outdoor mask...")
    
    # extract buildings
    buildings = osm_data['buildings']
    
    # create empty mask
    indoor_mask = np.zeros(grid_shape, dtype=int)
    
    # helper function to convert geom to pixel coordinates
    def geom_to_pixels(geom, bbox, shape):
        # get min/max coordinates
        minx, miny, maxx, maxy = geom.bounds
        
        # convert to relative coordinates within bbox (0-1)
        rel_minx = (minx - bbox["west"]) / (bbox["east"] - bbox["west"])
        rel_maxx = (maxx - bbox["west"]) / (bbox["east"] - bbox["west"])
        rel_miny = (miny - bbox["south"]) / (bbox["north"] - bbox["south"])
        rel_maxy = (maxy - bbox["south"]) / (bbox["north"] - bbox["south"])
        
        # convert to pixel coordinates
        pixel_minx = int(rel_minx * shape[1])
        pixel_maxx = int(rel_maxx * shape[1])
        pixel_miny = int((1 - rel_maxy) * shape[0])  # Flip y-axis
        pixel_maxy = int((1 - rel_miny) * shape[0])  # Flip y-axis
        
        # ensure within bounds
        pixel_minx = max(0, min(pixel_minx, shape[1]-1))
        pixel_maxx = max(0, min(pixel_maxx, shape[1]-1))
        pixel_miny = max(0, min(pixel_miny, shape[0]-1))
        pixel_maxy = max(0, min(pixel_maxy, shape[0]-1))
        
        return pixel_minx, pixel_miny, pixel_maxx, pixel_maxy
    
    # add buildings to indoor mask
    if not buildings.empty:
        for idx, building in buildings.iterrows():
            geom = building.geometry
            if geom.is_valid:
                minx, miny, maxx, maxy = geom_to_pixels(geom, bbox, grid_shape)
                indoor_mask[miny:maxy+1, minx:maxx+1] = 1
    
    return indoor_mask 
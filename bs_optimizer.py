import optuna
import numpy as np
import functools
from typing import Dict, List, Tuple, Any, Callable
import math
from shapely.geometry import Point

def tr38901_path_loss(distance, frequency, height_bs, height_ue, environment, is_los):
    """
    Calculate path loss using 3GPP TR 38.901 model
    
    Parameters:
    distance: distance in meters
    frequency: frequency in GHz
    height_bs: base station height in meters
    height_ue: user equipment height in meters
    environment: 'urban', 'suburban', or 'rural'
    is_los: boolean indicating line-of-sight condition
    
    Returns:
    Path loss in dB
    """

    #these values are from the 3GPP TR 38.901 technical report 
    # breakpoint distance needing the given height, freq, and environment (given in paper   )
    d_bp = 4 * height_bs * height_ue * frequency * 10 / 3  
    
    # There is loss depending on environment and conditions
    if environment == 'urban':
        if is_los:
            # Urban LOS formula 
            # breakpoint is the distance where the loss is different
            if distance <= d_bp:
                # LOS before breakpoint
                pl = 28.0 + 22*np.log10(distance) + 20*np.log10(frequency)
            else:
                # LOS after breakpoint
                pl = 28.0 + 40*np.log10(distance/d_bp) + 22*np.log10(d_bp) + 20*np.log10(frequency)
        else:
            # if no breakpoint, the loss is different
            pl = 13.54 + 39.08*np.log10(distance) + 20*np.log10(frequency) - 0.6*(height_ue-1.5)
    elif environment == 'suburban':
        if is_los:
            # suburban has less loss than urban
            if distance <= d_bp:
                pl = 28.0 + 22*np.log10(distance) + 20*np.log10(frequency)
            else:
                pl = 28.0 + 40*np.log10(distance/d_bp) + 22*np.log10(d_bp) + 20*np.log10(frequency)
        else:
            # if no breakpoint, the loss is different
            pl = 13.54 + 37.08*np.log10(distance) + 20*np.log10(frequency) - 0.6*(height_ue-1.5)
    elif environment == 'rural':
        if is_los:
            # rural LOS should have less loss than urban
            if distance <= d_bp:
                pl = 28.0 + 22*np.log10(distance) + 20*np.log10(frequency)
            else:
                pl = 28.0 + 40*np.log10(distance/d_bp) + 22*np.log10(d_bp) + 20*np.log10(frequency)
        else:
            pl = 13.54 + 35.08*np.log10(distance) + 20*np.log10(frequency) - 0.6*(height_ue-1.5)
    else:
        # we default to most loss in case there is no environment (assumed the worst case)
        pl = 13.54 + 39.08*np.log10(distance) + 20*np.log10(frequency) - 0.6*(height_ue-1.5)
    
    # the paper also mentions an atmospheric loss, this is a simple implementation to account for that
    if frequency > 6:
        # approximately 0.05 dB/m at 28 GHz in moderate humidity
        atmospheric_loss = 0.002 * distance * (frequency / 10)
        pl += atmospheric_loss
    
    return pl

def calculate_5g_propagation(center_lat, center_lon, base_stations, bbox, 
                           grid_resolution_m, osm_data, terrain_data, config):
    """
    Calculate 5G propagation using 3GPP TR 38.901 model
    
    Parameters:
    center_lat, center_lon: Center coordinates
    base_stations: List of base station parameters [lat, lon, height, power, name, downtilt]
    bbox: Bounding box
    grid_resolution_m: Grid resolution in meters
    osm_data: OSM data dictionary
    terrain_data: Terrain data
    config: Configuration dictionary
    
    Returns:
    tuple: (path_loss_list, grid_lats, grid_lons, grid_shape, indoor_mask)
    """
    # parameters needed for 5g propagation
    frequency = config.get("frequency", 3.5)                # default to 3.5 GHz (mid-band 5G)
    frequency_band = config.get("frequency_band", "mid")    # default to mid-band
    environment = config.get("environment", "urban")        # default to urban environment
    user_height = config.get("user_height", 1.5)            # default UE height in meters
    
    # create grid points
    lat_min, lat_max = bbox["south"], bbox["north"]
    lon_min, lon_max = bbox["west"], bbox["east"]
    
    # calculate the number of points in each dimention
    # convert to meters per degree from the map point in center
    meters_per_degree_lat = 111111
    meters_per_degree_lon = 111111 * math.cos(math.radians(center_lat))
    
    # grid conversion from lat and lon to meters
    grid_lat_step = grid_resolution_m / meters_per_degree_lat
    grid_lon_step = grid_resolution_m / meters_per_degree_lon
    
    # get grid points for point placement
    grid_lats = np.arange(lat_min, lat_max, grid_lat_step)
    grid_lons = np.arange(lon_min, lon_max, grid_lon_step)
    
    lat_grid, lon_grid = np.meshgrid(grid_lats, grid_lons)
    grid_shape = lat_grid.shape
    
    # we disregard the curve of the earth and use a flat grid
    # for a simple calculations
    grid_points = np.column_stack((lat_grid.flatten(), lon_grid.flatten()))
    
    #folium can determine if it is a building or not, we need to know if a point is indoor or outdoor
    indoor_mask = np.zeros(len(grid_points), dtype=int)
    
    # process building data from OSM data if available
    if 'buildings' in osm_data and osm_data['buildings'] is not None:
        try:
            from shapely.strtree import STRtree
            import time
            
            start_time = time.time()
            print(f"  Processing {len(osm_data['buildings'])} buildings for indoor/outdoor classification...")
            
            # create spatial index for buildings
            building_geoms = list(osm_data['buildings'].geometry)
            if building_geoms:
                building_idx = STRtree(building_geoms)
                
                # process grid points in batches
                batch_size = 5000
                num_batches = (len(grid_points) + batch_size - 1) // batch_size
                
                for batch_num in range(num_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min((batch_num + 1) * batch_size, len(grid_points))
                    
                    # create points for this batch
                    points = [Point(grid_points[i][1], grid_points[i][0]) for i in range(start_idx, end_idx)]
                    
                    # query spatial index for potential buildings that contain each point
                    for i, point in enumerate(points):
                        global_idx = start_idx + i
                        # get indices of potential buildings that might contain the point
                        potential_indices = building_idx.query(point)
                        
                        # get the actual geometries using the indices and check contains
                        for idx in potential_indices:
                            if building_geoms[idx].contains(point):
                                indoor_mask[global_idx] = 1
                                break
                    
                    # print progress for large datasets
                    if num_batches > 1 and batch_num % 10 == 0:
                        print(f"    Processed {end_idx}/{len(grid_points)} points ({end_idx/len(grid_points)*100:.1f}%)...")
                
                print(f"  Building processing completed in {time.time() - start_time:.2f} seconds")
                print(f"  {np.sum(indoor_mask)} indoor points identified ({np.sum(indoor_mask)/len(indoor_mask)*100:.1f}% of total)")
            else:
                print("  No building geometries found")
                
        except (KeyError, AttributeError, TypeError, ImportError) as e:
            print(f"  Warning: Could not process buildings: {e}")
    
    # calculate path loss for each base station
    path_loss_list = []
    
    for bs in base_stations:
        bs_lat, bs_lon, bs_height, bs_power, _, bs_downtilt = bs
        
        # initialize path loss matrix
        path_loss = np.zeros(len(grid_points))
        
        # calculate LOS probability for each point
        # using 3GPP TR 38.901 Urban Macro LOS probability models
        los_prob = np.zeros(len(grid_points))
        
        for i, point in enumerate(grid_points):
            # calculate distance in meters
            point_lat, point_lon = point
            
            # calculate approximate distance in meters
            lat_distance_m = abs(bs_lat - point_lat) * meters_per_degree_lat
            lon_distance_m = abs(bs_lon - point_lon) * meters_per_degree_lon
            
            # euclidean distance
            distance_m = max((lat_distance_m**2 + lon_distance_m**2)**0.5, 1)  # avoid division by zero
            
            # calculate LOS probability based on 3GPP TR 38.901
            # these formulas are more accurate representations of the standard
            if environment == "urban":
                # urban macro (UMa) LOS probability
                if distance_m <= 18:
                    los_prob[i] = 1.0
                else:
                    los_prob[i] = (18/distance_m) + np.exp(-distance_m/63) * (1 - (18/distance_m))
            elif environment == "suburban":
                # suburban macro (RMa) LOS probability (adapted)
                if distance_m <= 36:
                    los_prob[i] = 1.0
                else:
                    los_prob[i] = (36/distance_m) + np.exp(-distance_m/72) * (1 - (36/distance_m))
            else:  # rural
                # rural macro LOS probability (adapted)
                if distance_m <= 45:
                    los_prob[i] = 1.0
                else:
                    los_prob[i] = (45/distance_m) + np.exp(-distance_m/90) * (1 - (45/distance_m))
            
            # determine if LOS based on probability
            # using deterministic seed based on position to ensure consistent results
            np.random.seed(int(point_lat * 1e6 + point_lon * 1e7) % 2**32)
            is_los = np.random.random() < los_prob[i]
            
            # apply building penetration loss for indoor points
            building_penetration_loss = 0
            if indoor_mask[i] == 1:
                # building penetration loss based on 3GPP TR 38.901
                if frequency_band == "low":  # low band (< 1 GHz)
                    # typically 10-15 dB for low frequencies
                    building_penetration_loss = 10 + 5 * np.random.random()
                elif frequency_band == "mid":  # mid band (1-6 GHz)
                    # standard glass: ~20-25 dB for mid-band frequencies
                    building_penetration_loss = 20 + 5 * np.random.random()
                else:  # High band / mmWave (> 6 GHz)
                    building_penetration_loss = 25 + 15 * np.random.random()
            
            # Use 3GPP TR 38.901 path loss model
            path_loss[i] = tr38901_path_loss(
                distance_m, frequency, bs_height, user_height, environment, is_los
            )
            
            # add building penetration loss
            path_loss[i] += building_penetration_loss
            
            # add terrain effect
            if terrain_data is not None:
                try:
                    # extract terrain heights for BS and UE positions starting from default values
                    bs_height_terrain = 0  
                    ue_height_terrain = 0  
                    
                    if bs_height_terrain > bs_height or ue_height_terrain > user_height:
                        terrain_loss = 20 
                    else:
                        terrain_loss = 0  

                    path_loss[i] += terrain_loss
                except Exception as e:
                    print(f"Warning: Could not process terrain data: {e}")
            
            # account for antenna downtilt (information given from paper)
            if bs_downtilt > 0:
                # calculate angle in vertical plane
                distance_2d = distance_m
                height_diff = bs_height - user_height
                elevation_angle = np.degrees(np.arctan2(height_diff, distance_2d))
                
                # calculate vertical antenna pattern attenuation
                tilt_angle_diff = abs(elevation_angle - bs_downtilt)
                if tilt_angle_diff <= 10:  
                    tilt_loss = 0.5 * tilt_angle_diff
                else:  
                    tilt_loss = 5 + 0.2 * (tilt_angle_diff - 10)
                
                path_loss[i] += tilt_loss
        
        # apply the transmit power
        effective_path_loss = path_loss - bs_power
        
        # add to the list of path losss 
        path_loss_list.append(effective_path_loss)
    
    return path_loss_list, grid_lats, grid_lons, grid_shape, indoor_mask

class BaseStationOptimizer:
    # this implementation using optuna to optimize the base station placements 
    def __init__(self, config):
        self.config = config
        self.best_config = None
        self.study = None
    
    def objective_function(self, trial, center_lat, center_lon, bbox, grid_resolution_m, 
                         osm_data, terrain_data, propagation_func, sinr_func, config):
        """
        Optimization objective function for base station placement
        
        Parameters:
        trial: Optuna trial object
        center_lat, center_lon: Center coordinates
        bbox: Bounding box
        grid_resolution_m: Grid resolution
        osm_data: OSM data dictionary
        terrain_data: Terrain data
        propagation_func: Function to calculate path loss (should be calculate_5g_propagation)
        sinr_func: Function to calculate SINR
        config: Configuration dictionary
        
        Returns:
        float: Performance metric value (higher is better)
        """
        print(f"Trial {trial.number}: Evaluating BS placement...")
        
        # get the base configuration
        opt_config = config.copy()
        
        # maximum number of potential base stations
        max_bs = config.get("max_base_stations", 5)         # maximum number of base stations allowed
        min_bs = config.get("min_base_stations", 1)         # minimum number of base stations required
        initial_bs_count = len(config["base_stations"])     # initial number of base stations
        
        # let the optimizer decide how many base stations to use 
        bs_count = trial.suggest_int("bs_count", min_bs, max_bs)
        
        # select 5G frequency band
        # in research, we determined that there are different frequency bands for 5G depending on the environment
        # we categorize the environments into low, mid, and high bands
        frequency_band = trial.suggest_categorical("frequency_band", ["low", "mid", "high"])
        if frequency_band == "low":
            # low-band 5G (600-700 MHz)
            frequency = trial.suggest_float("frequency", 0.6, 0.7)
        elif frequency_band == "mid":
            # mid-band 5G (2.5-3.7 GHz)
            frequency = trial.suggest_float("frequency", 2.5, 3.7)
        else:
            # high-band 5G / mmWave (24-40 GHz)
            frequency = trial.suggest_float("frequency", 24.0, 40.0)
        
        # select environment type
        environment = trial.suggest_categorical("environment", ["urban", "suburban", "rural"])
        
        # there was an issue with sometimes the base stations are placed in water,
        # OSM can determine if a point has water, so we just eliminate the points considered for placement
        water_polygons = []
        if 'natural' in osm_data and osm_data['natural'] is not None:
            try:
                water_features = osm_data['natural'][osm_data['natural']['natural'] == 'water']
                if not water_features.empty:
                    for _, water in water_features.iterrows():

                        # sometimes water has multiple points within the same polygon
                        if hasattr(water.geometry, 'geoms'):       
                            for geom in water.geometry.geoms:
                                water_polygons.append(geom)
                        else:
                            water_polygons.append(water.geometry)
            except (KeyError, AttributeError, TypeError) as e:
                print(f"  Warning: Could not process water features: {e}")
        
        # create a list of base stations 
        optimized_bs = []
        for i in range(bs_count):
            # determien a valid placement and double check that it is not in water
            valid_placement = False
            max_attempts = 10  # there was a problem with infinite loops, so there is a limit
            
            for attempt in range(max_attempts):
                # we need to ensure that the base station is within the bounding box we are considering
                bs_lat = trial.suggest_float(f"bs{i}_lat_attempt{attempt}", bbox["south"], bbox["north"])
                bs_lon = trial.suggest_float(f"bs{i}_lon_attempt{attempt}", bbox["west"], bbox["east"])
                
                # check if the point is in water
                is_in_water = False
                bs_point = Point(bs_lon, bs_lat)
                
                for water_poly in water_polygons:
                    if water_poly.contains(bs_point):
                        is_in_water = True
                        break
                
                # if the point is not in water, accept this placement
                if not is_in_water:
                    valid_placement = True
                    break
            
            # If we couldn't find a valid placement after max attempts, use the last one
            if environment == "urban":
                bs_height = trial.suggest_float(f"bs{i}_height", 10.0, 150.0)
            elif environment == "suburban":
                bs_height = trial.suggest_float(f"bs{i}_height", 15.0, 80.0)
            else:  # rural
                bs_height = trial.suggest_float(f"bs{i}_height", 20.0, 100.0)
            

            bs_power = trial.suggest_float(f"bs{i}_power", 25.0, 35.0)
            bs_downtilt = trial.suggest_float(f"bs{i}_downtilt", 0.0, 15.0)
            
            #
            if i < initial_bs_count:
                bs_name = config["base_stations"][i][4]
            else:
                bs_name = f"5G BS {i+1} ({frequency_band}-band)"
            
            # Add to optimized base stations
            optimized_bs.append([bs_lat, bs_lon, bs_height, bs_power, bs_name, bs_downtilt])
        
        # check distance between base stations, it doesnt make sense if they are too close together
        # a penalty system was added to help discurage different scenarios
        min_distance_m = 1000  
        penalty = 0
        
        for i in range(bs_count):
            for j in range(i+1, bs_count):
                # calculate the distance between the base stations
                lat1, lon1 = optimized_bs[i][0], optimized_bs[i][1]
                lat2, lon2 = optimized_bs[j][0], optimized_bs[j][1]
                
                # to determine the distance relative to lat and lon we need to convert.
                lat_distance_m = abs(lat1 - lat2) * 111111
                lon_distance_m = abs(lon1 - lon2) * 111111 * math.cos(math.radians((lat1 + lat2) / 2))
                
                # euclidean distance
                distance_m = (lat_distance_m**2 + lon_distance_m**2)**0.5
                
                # apply penalty if distance is less than minimum
                if distance_m < min_distance_m:
                    penalty += (min_distance_m - distance_m) / 100  # scale penalty
        
        # check if base stations are in water bodies and apply reduced penalty
        for i in range(bs_count):
            bs_lat = optimized_bs[i][0]  
            bs_lon = optimized_bs[i][1]
            
            # create a point object for the base station
            bs_point = Point(bs_lon, bs_lat)
            
            # check if the point is in water (using natural features from OSM)
            if 'natural' in osm_data and osm_data['natural'] is not None:
                try:
                    water_features = osm_data['natural'][osm_data['natural']['natural'] == 'water']
                    
                    if not water_features.empty:
                        # check if the point is in any water feature
                        is_in_water = False
                        for _, water in water_features.iterrows():
                            if water.geometry.contains(bs_point):
                                is_in_water = True
                                break
                        
                        if is_in_water:
                            # apply reduced penalty for water placement
                            penalty += 100
                            print(f"  Warning: Base station {i+1} is in water! Applying penalty.")
                except (KeyError, AttributeError, TypeError) as e:
                    print(f"  Warning: Could not check for water bodies: {e}")
        
        # update the config with optimized base stations
        opt_config["base_stations"] = optimized_bs
        
        # add 5G propagation parameters to config
        opt_config["frequency"] = frequency
        opt_config["environment"] = environment
        opt_config["frequency_band"] = frequency_band
        
        # calculate path loss for this configuration
        path_loss_list, grid_lats, grid_lons, grid_shape, indoor_mask = propagation_func(
            center_lat, center_lon, optimized_bs, bbox, grid_resolution_m, 
            osm_data, terrain_data, opt_config
        )
        
        # calculate SINR grid
        sinr_db = sinr_func(path_loss_list, optimized_bs, opt_config)
        
        # calculate performance metrics
        coverage_threshold = opt_config["coverage_threshold_db"]
        
        # calculate percent coverage
        coverage_percent = np.mean(sinr_db >= coverage_threshold) * 100
        
        # calculate average SINR (only for covered points)
        covered_points = sinr_db[sinr_db >= coverage_threshold]
        avg_sinr = np.mean(covered_points) if len(covered_points) > 0 else -float('inf')
        
        # calculate weighted metrics based on indoor/outdoor
        indoor_coverage = np.mean((sinr_db >= coverage_threshold) & (indoor_mask == 1)) * 100
        outdoor_coverage = np.mean((sinr_db >= coverage_threshold) & (indoor_mask == 0)) * 100
        
        # prioritize outdoor coverage more (reduce impact of indoor coverage)
        balanced_coverage = (outdoor_coverage * 0.8) + (indoor_coverage * 0.2)
        
        # choose metric based on configuration
        if opt_config["optimization_metric"] == "coverage_percent":
            metric = coverage_percent
        elif opt_config["optimization_metric"] == "avg_sinr":
            metric = avg_sinr
        elif opt_config["optimization_metric"] == "indoor_coverage":
            metric = indoor_coverage
        elif opt_config["optimization_metric"] == "outdoor_coverage":
            metric = outdoor_coverage
        elif opt_config["optimization_metric"] == "balanced_coverage":
            metric = balanced_coverage
        else:
            # by default, use balanced coverage
            metric = balanced_coverage
        
        # apply the distance penalty to the metric
        metric = metric - penalty
        
        # apply a cost penalty for each additional base station
        # this penalizes configurations with more base stations
        base_station_cost = 10  # cost penalty per base station
        cost_penalty = base_station_cost * bs_count
        
        # final score balances coverage and cost
        # using a weighted approach where we apply a smaller penalty for cost
        # the system will prefer to add base stations if they significantly improve coverage
        final_score = metric - (cost_penalty * 0.5)
        
        return final_score
    
    def optimize(self, center_lat, center_lon, bbox, grid_resolution_m, 
               osm_data, terrain_data, propagation_func, sinr_func):
        """
        Optimize base station placement
        
        Parameters:
        center_lat, center_lon: Center coordinates
        bbox: Bounding box
        grid_resolution_m: Grid resolution
        osm_data: OSM data dictionary
        terrain_data: Terrain data
        propagation_func: Function to calculate path loss (should be calculate_5g_propagation)
        sinr_func: Function to calculate SINR
        
        Returns:
        dict: Optimized configuration
        """
        print("Optimizing base station placement...")
        
        # maximization objective function from optuna
        self.study = optuna.create_study(direction="maximize")
        
        # use partial function to fix some parameters for the objective function
        objective = functools.partial(
            self.objective_function,
            center_lat=center_lat,
            center_lon=center_lon,
            bbox=bbox,
            grid_resolution_m=grid_resolution_m,
            osm_data=osm_data,
            terrain_data=terrain_data,
            propagation_func=propagation_func,
            sinr_func=sinr_func,
            config=self.config
        )
        
        # run the optimization
        n_trials = self.config.get("optimization_trials", 50)
        
        try:
            self.study.optimize(objective, n_trials=n_trials)
            print(f"Optimization completed successfully with {len(self.study.trials)} trials")
        except KeyboardInterrupt:
            print("\nOptimization interrupted! Using best result so far...")
        except Exception as e:
            print(f"Optimization error: {str(e)}. Using best result so far...")
        
        # check if we have any completed trials
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            print("No completed trials! Using default configuration.")
            return self.config
        
        # get best parameters
        best_params = self.study.best_params
        
        # get the optimized number of base stations
        bs_count = best_params["bs_count"]
        
        # get optimized 5G parameters
        frequency_band = best_params["frequency_band"]
        frequency = best_params["frequency"]
        environment = best_params["environment"]
        
        # create optimized base stations
        optimized_bs = []
        initial_bs_count = len(self.config["base_stations"])
        
        for i in range(bs_count):
            # look for parameters with attempt numbers (bs0_lat_attempt0, bs0_lat_attempt1, etc.)
            # try to find the successful attempt for latitude
            bs_lat = None
            bs_lon = None
            
            # find the highest attempt number that exists in the parameters
            for attempt in range(10):  # assuming max 10 attempts
                lat_param = f"bs{i}_lat_attempt{attempt}"
                lon_param = f"bs{i}_lon_attempt{attempt}"
                
                if lat_param in best_params and lon_param in best_params:
                    bs_lat = best_params[lat_param]
                    bs_lon = best_params[lon_param]
                    break
            
            # if we didn't find any attempt, try the old parameter format as fallback
            if bs_lat is None or bs_lon is None:
                if f"bs{i}_lat" in best_params and f"bs{i}_lon" in best_params:
                    bs_lat = best_params[f"bs{i}_lat"]
                    bs_lon = best_params[f"bs{i}_lon"]
                else:
                    print(f"Warning: Could not find valid coordinates for base station {i}!")
                    # use default values as a last resort
                    bs_lat = center_lat + (0.001 * i)  # slightly offset from center
                    bs_lon = center_lon + (0.001 * i)
            
            bs_height = best_params[f"bs{i}_height"]
            bs_power = best_params[f"bs{i}_power"]
            bs_downtilt = best_params[f"bs{i}_downtilt"]
            
            
            if i < initial_bs_count:
                bs_name = self.config["base_stations"][i][4]
            else:
                bs_name = f"5G BS {i+1} ({frequency_band}-band)"
            
            optimized_bs.append([bs_lat, bs_lon, bs_height, bs_power, bs_name, bs_downtilt])
        
        # create a new configuration with optimized base stations
        self.best_config = self.config.copy()
        self.best_config["base_stations"] = optimized_bs
        
        # add 5G parameters to best configuration
        self.best_config["frequency"] = frequency
        self.best_config["environment"] = environment
        self.best_config["frequency_band"] = frequency_band
        
        print(f"Optimization complete! Best {self.config['optimization_metric']}: {self.study.best_value:.2f}")
        print(f"Optimized number of base stations: {bs_count}")
        print(f"Optimized frequency band: {frequency_band} ({frequency:.2f} GHz)")
        print(f"Optimized environment type: {environment}")
        
        return self.best_config
    
    def get_optimization_results(self):
        """
        Get detailed optimization results
        
        Returns:
        dict: Dictionary with optimization results
        """
        if self.study is None:
            return {"error": "No optimization has been run yet"}
        
        # get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        # get all trials
        trials = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials.append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params
                })
        
        # sort trials by value (descending)
        trials.sort(key=lambda x: x["value"] if x["value"] is not None else -float('inf'), reverse=True)
        
        # generate cost-benefit analysis
        cost_benefit = {}
        if "bs_count" in best_params:
            # group trials by base station count
            bs_count_results = {}
            for trial in trials:
                if "bs_count" in trial["params"] and trial["value"] is not None:
                    bs_count = trial["params"]["bs_count"]
                    if bs_count not in bs_count_results:
                        bs_count_results[bs_count] = []
                    bs_count_results[bs_count].append(trial["value"])
            
            # calculate average performance for each base station count
            for bs_count, values in bs_count_results.items():
                avg_performance = sum(values) / len(values)
                # add cost-benefit ratio (performance per base station)
                cost_benefit[bs_count] = {
                    "avg_performance": avg_performance,
                    "performance_per_bs": avg_performance / bs_count,
                    "sample_count": len(values)
                }
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "trials": trials,
            "best_config": self.best_config,
            "cost_benefit_analysis": cost_benefit
        }
    
    def export_optimization_plots(self, output_dir="results"):
        """
        Export optimization plots
        
        Parameters:
        output_dir (str): Output directory
        
        Returns:
        dict: Dictionary with paths to saved plots
        """
        import os
        import matplotlib.pyplot as plt
        
        if self.study is None:
            return {"error": "No optimization has been run yet"}
        
        # create ouput dir 
        os.makedirs(output_dir, exist_ok=True)
        
        # plot optimization history
        history_path = os.path.join(output_dir, "optimization_history.png")
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.tight_layout()
        plt.savefig(history_path)
        plt.close()
        
        # plot parameter importances
        importance_path = os.path.join(output_dir, "parameter_importance.png")
        try:
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.tight_layout()
            plt.savefig(importance_path)
            plt.close()
        except:
            importance_path = "Error: Could not compute parameter importances"
        
        # create cost-benefit analysis plot
        cost_benefit_path = os.path.join(output_dir, "cost_benefit.png")
        try:
            # get optimization results
            results = self.get_optimization_results()
            cost_benefit = results.get("cost_benefit_analysis", {})
            
            if cost_benefit:
                plt.figure(figsize=(12, 10))
                
                # create subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                
                # data for plotting
                bs_counts = list(cost_benefit.keys())
                avg_performance = [data["avg_performance"] for data in cost_benefit.values()]
                performance_per_bs = [data["performance_per_bs"] for data in cost_benefit.values()]
                
                # plot 1: performance vs. BS count
                ax1.bar(bs_counts, avg_performance, alpha=0.7)
                ax1.set_xlabel('Number of Base Stations')
                ax1.set_ylabel('Average Performance')
                ax1.set_title('Performance vs. Number of Base Stations')
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # add value labels on top of each bar
                for i, v in enumerate(avg_performance):
                    ax1.text(bs_counts[i], v + 0.5, f"{v:.2f}", ha='center')
                
                # plot 2: efficiency (performance per BS) vs. BS count
                ax2.bar(bs_counts, performance_per_bs, color='green', alpha=0.7)
                ax2.set_xlabel('Number of Base Stations')
                ax2.set_ylabel('Performance per Base Station')
                ax2.set_title('Cost-Benefit Analysis: Efficiency vs. Number of Base Stations')
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                # add value labels on top of each bar
                for i, v in enumerate(performance_per_bs):
                    ax2.text(bs_counts[i], v + 0.2, f"{v:.2f}", ha='center')
                
                plt.tight_layout()
                plt.savefig(cost_benefit_path)
                plt.close()
            else:
                cost_benefit_path = "Error: No cost-benefit data available"
        except Exception as e:
            cost_benefit_path = f"Error: {str(e)}"
        
        return {
            "history": history_path,
            "importance": importance_path,
            "cost_benefit": cost_benefit_path
        } 
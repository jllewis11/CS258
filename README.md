#Cellular Network Simulator

A simulation of radio wave propagation for cellular network planning, incorporating OSM data, terrain modeling, and BS optimization.

## Features

- **Propagation Modeling**: Includes terrain effects, building penetration, and antenna patterns
- **OpenStreetMap Integration**: Uses real building/street data for accurate clutter modeling
- **Indoor/Outdoor Analysis**: Distinguishes between indoor and outdoor coverage areas
- **Base Station Optimization**: Automatically optimizes BS placement, height, power, and antenna tilt
- **LLM Orchestration**: Optional AI-driven workflow orchestration (requires OpenAI API key)
- **Interactive Visualization**: Generate detailed Folium maps with comprehensive information

## Installation



1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the simulator with default settings:

```bash
python main.py
```

This will generate a radio propagation map for Hyde Park, London with two base stations.

You must set the OpenAI API key in the configuration first:

```python
DEFAULT_CONFIG["use_llm_orchestration"] = True
DEFAULT_CONFIG["openai_api_key"] = "your-api-key"
```

## Configuration

Edit the `DEFAULT_CONFIG` in `main.py` to customize:

- Study area location and size
- Network parameters (frequency, bandwidth, etc.)
- Base station settings
- Propagation model options
- Optimization parameters
- Output settings

## Project Structure

- `main.py`: Main simulation script
- `utils.py`: Basic utility functions
- `osm_terrain_handler.py`: OSM data and terrain handling
- `bs_optimizer.py`: Base station optimization
- `llm_orchestrator.py`: LLM-based orchestration (optional)

## Output

The simulator generates:

1. An interactive HTML map with SINR overlay
2. Indoor/outdoor coverage analysis
3. Optimization results
4. Coverage statistics

Results are saved to the configured output directory (default: `results/`).

## License
[MIT License](LICENSE)

## Acknowledgments
- [Folium](https://github.com/python-visualization/folium) for visualization 
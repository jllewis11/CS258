#!/usr/bin/env python3
# ---------------------------------------------------------
# LLM Orchestration System
# AI-driven automation for cellular network simulation
# ---------------------------------------------------------

import os
import openai
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple

class LLMOrchestrator:
    """
    LLM-based orchestration system for automating the simulation workflow
    
    This class implements an AI agent that can:
    1. Understand high-level commands ("Import OSM data", "Generate radio map")
    2. Execute the corresponding functions
    3. Provide progress updates and explanations
    """
    
    def __init__(self, config):
        """
        Initialize the LLM orchestrator
        
        Parameters:
        config (dict): Configuration dictionary with API keys, etc.
        """
        self.config = config
        self.api_key = config.get("openai_api_key", "")
        if not self.api_key and config.get("use_llm_orchestration", False):
            print("Warning: LLM orchestration enabled but no API key provided")
        
        self.model = config.get("llm_model", "gpt-4")
        print(f"Using LLM model: {self.model}")
        
        self.messages = []
        self.setup_llm()
    
    def setup_llm(self):
        """Setup the LLM client"""
        if self.api_key:
            openai.api_key = self.api_key
    
    def add_system_message(self, content):
        """
        Add a system message to the conversation
        
        Parameters:
        content (str): Message content
        """
        self.messages.append({"role": "system", "content": content})
    
    def add_user_message(self, content):
        """
        Add a user message to the conversation
        
        Parameters:
        content (str): Message content
        """
        self.messages.append({"role": "user", "content": content})
    
    def get_llm_response(self, prompt):
        """
        Get a response from the LLM
        
        Parameters:
        prompt (str): User prompt
        
        Returns:
        str: LLM response
        """
        if not self.api_key:
            return "LLM orchestration is disabled or no API key provided"
        
        # Add the user prompt
        self.add_user_message(prompt)
        
        try:
            # Call the OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_completion_tokens=1024
            )
            
            # Extract and save the response
            response_text = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response_text})
            
            return response_text
        
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return f"Error: {e}"
    
    def parse_command(self, text):
        """
        Parse a command from text into actionable steps
        
        Parameters:
        text (str): User command text
        
        Returns:
        str: Parsed command type
        """
        if not self.api_key:
            # Simple rule-based parsing when LLM not available
            text = text.lower()
            if "import" in text and "osm" in text:
                return "import_osm"
            elif "generate" in text and "radio" in text and "map" in text:
                return "generate_radio_map"
            elif "optimize" in text:
                return "optimize_bs"
            else:
                return "unknown"
        
        # Use the LLM to parse the command
        prompt = f"""
        Parse the following command into one of these categories:
        - import_osm: Import OpenStreetMap data
        - generate_radio_map: Generate radio propagation map
        - optimize_bs: Optimize base station placement
        - unknown: Unknown command
        
        Command: {text}
        
        Category:
        """
        
        response = self.get_llm_response(prompt)
        return response.strip()
    
    def execute_command(self, command, simulation_state, command_handlers=None):
        """
        Execute a command in the simulation state
        
        Parameters:
        command (str): Command to execute
        simulation_state (dict): Current simulation state
        command_handlers (dict): Dictionary of command handler functions
        
        Returns:
        tuple: (result message, updated simulation state)
        """
        print(f"Executing command: {command}")
        
        # If no custom handlers provided, use default handlers
        if command_handlers is None:
            return self._default_execute_command(command, simulation_state)
        
        # Use custom handlers if provided
        if command in command_handlers:
            handler = command_handlers[command]
            return handler(simulation_state)
        else:
            return f"Unknown command: {command}", simulation_state
    
    def _default_execute_command(self, command, simulation_state):
        """
        Default command execution implementation
        
        Parameters:
        command (str): Command to execute
        simulation_state (dict): Current simulation state
        
        Returns:
        tuple: (result message, updated simulation state)
        """
        if command == "import_osm":
            # Placeholder - in a real implementation, you would call your OSM import function
            return "OSM import not implemented in default handler", simulation_state
        
        elif command == "generate_radio_map":
            # Placeholder - in a real implementation, you would call your radio map generation function
            return "Radio map generation not implemented in default handler", simulation_state
        
        elif command == "optimize_bs":
            # Placeholder - in a real implementation, you would call your optimization function
            return "BS optimization not implemented in default handler", simulation_state
        
        else:
            return f"Unknown command: {command}", simulation_state
    
    def run_interactive(self, command_handlers=None):
        """
        Run an interactive session with the LLM orchestrator
        
        Parameters:
        command_handlers (dict): Dictionary of command handler functions
        
        Returns:
        dict: Final simulation state
        """
        # Initialize simulation state
        simulation_state = {}
        
        # Welcome message
        print("=" * 80)
        print("Radio Propagation Simulator with LLM Orchestration")
        print("=" * 80)
        print("Type 'exit' to quit")
        print("")
        
        # Add system instructions
        self.add_system_message("""
        You are an AI assistant for a radio propagation simulator.
        You can help users:
        1. Import OSM data for an area
        2. Generate radio propagation maps
        3. Optimize base station placement
        
        Parse user commands and respond conversationally.
        """)
        
        # Interactive loop
        while True:
            user_input = input("Enter command: ")
            
            if user_input.lower() == "exit":
                break
            
            # Parse the command
            command_type = self.parse_command(user_input)
            
            # Execute the command
            result, simulation_state = self.execute_command(command_type, simulation_state, command_handlers)
            
            # Print the result
            print(result)
        
        return simulation_state
    
    def run_simulation(self, workflow_steps, simulation_state=None, command_handlers=None):
        """
        Run the full simulation workflow automatically
        
        Parameters:
        workflow_steps (list): List of steps to execute
        simulation_state (dict): Initial simulation state
        command_handlers (dict): Dictionary of command handler functions
        
        Returns:
        dict: Final simulation state
        """
        # Initialize simulation state if not provided
        if simulation_state is None:
            simulation_state = {}
        
        # Execute each step in the workflow
        for i, step in enumerate(workflow_steps):
            print(f"Step {i+1}/{len(workflow_steps)}: {step}")
            result, simulation_state = self.execute_command(step, simulation_state, command_handlers)
            print(result)
        
        return simulation_state
    
    def explain_results(self, simulation_state):
        """
        Generate an explanation of the simulation results using the LLM
        
        Parameters:
        simulation_state (dict): Simulation state with results
        
        Returns:
        str: Explanation of results
        """
        if not self.api_key:
            return "LLM explanation not available (no API key)"
        
        # Extract key metrics from simulation state
        metrics = {}
        if "sinr_db" in simulation_state:
            sinr = simulation_state["sinr_db"]
            if sinr is not None:
                metrics["avg_sinr"] = float(np.mean(sinr))
                metrics["min_sinr"] = float(np.min(sinr))
                metrics["max_sinr"] = float(np.max(sinr))
        
        # Extract base station info
        bs_info = []
        if "base_stations" in self.config:
            for bs in self.config["base_stations"]:
                bs_info.append({
                    "lat": bs[0],
                    "lon": bs[1],
                    "height": bs[2],
                    "power": bs[3],
                    "name": bs[4]
                })
        
        # Craft a prompt for the LLM
        prompt = f"""
        Please provide a concise analysis of these cellular network simulation results:
        
        Base Stations: {bs_info}
        
        Performance Metrics:
        {metrics}
        
        Explain the quality of coverage, any problem areas, and potential improvements in 3-5 sentences.
        """
        
        # Get the explanation from the LLM
        explanation = self.get_llm_response(prompt)
        
        return explanation 
"""
File: category_colors.py
Last modified: 03-11-2025

Shared category color mapping utilities.

This module provides consistent color generation for all 69 shape categories
using semantic grouping - similar categories get similar color families.
"""

import colorsys


# Categories organized by semantic groups
CATEGORY_GROUPS = {
    # AIRCRAFT & FLYING (Blue family - sky)
    'aircraft': ['AircraftBuoyant', 'Biplane', 'Helicopter', 'Jet', 'Monoplane', 'Rocket', 'Starship'],
    
    # GROUND VEHICLES (Red/Orange family - roads/machinery)
    'ground_vehicles': ['Bicycle', 'Bus', 'Car', 'MilitaryVehicle', 'Motorcycle', 'Train', 'Truck', 'TruckNonContainer'],
    
    # WATER VESSELS (Cyan/Teal family - water)
    'water': ['AquaticAnimal', 'Fish', 'Ship'],
    
    # BUILDINGS (Brown/Earth tones - construction)
    'buildings': ['Apartment', 'BuildingNonResidential', 'City', 'Door', 'House', 'Skyscraper'],
    
    # FURNITURE (Purple family - indoor living)
    'furniture': ['Bed', 'Chair', 'MultiSeat', 'NonWheelChair', 'RectangleTable', 'RoundTable', 'Shelf', 'WheelChair'],
    
    # MUSICAL INSTRUMENTS (Magenta/Pink family - arts)
    'music': ['ClassicPiano', 'Drum', 'Guitar', 'Musical_Instrument', 'PianoBoard', 'Violin'],
    
    # ELECTRONICS (Blue-gray family - technology)
    'electronics': ['Cellphone', 'Computer', 'ComputerKeyboard', 'DeskPhone', 'Monitor'],
    
    # LIGHTING (Yellow family - light sources)
    'lighting': ['DeskLamp', 'FloorLamp'],
    
    # SMALL OBJECTS (Lime/Green family - everyday items)
    'small_objects': ['Bookset', 'Bottle', 'Cup', 'Glasses', 'Mug', 'Spoon', 'Vase'],
    
    # WEAPONS & TOOLS (Dark red/crimson - danger/utility)
    'weapons_tools': ['Gun', 'Knife', 'SubmachineGun', 'Sword', 'Tool'],
    
    # NATURE (Green family - organic)
    'nature': ['Bird', 'PlantIndoors', 'PlantWildNonTree', 'Tree'],
    
    # ANIMALS & HUMANOID (Orange family - living beings)
    'living': ['Hand', 'HumanHead', 'Humanoid', 'Insect', 'Quadruped'],
    
    # MISCELLANEOUS (Remaining distinct hues)
    'misc': ['Chess', 'Hat', 'Sign', 'Wheel']
}

# Flatten to get complete list (same order as before for compatibility)
CATEGORIES_LIST = [
    'AircraftBuoyant', 'Apartment', 'AquaticAnimal', 'Bed', 'Bicycle', 'Biplane', 'Bird', 'Bookset', 'Bottle',
    'BuildingNonResidential', 'Bus', 'Car', 'Cellphone', 'Chess', 'City', 'ClassicPiano', 'Computer',
    'ComputerKeyboard', 'Cup', 'DeskLamp', 'DeskPhone', 'Door', 'Drum', 'Fish', 'FloorLamp', 'Glasses',
    'Guitar', 'Gun', 'Hand', 'Hat', 'Helicopter', 'House', 'HumanHead', 'Humanoid', 'Insect', 'Jet', 'Knife',
    'MilitaryVehicle', 'Monitor', 'Monoplane', 'Motorcycle', 'Mug', 'MultiSeat', 'Musical_Instrument',
    'NonWheelChair', 'PianoBoard', 'PlantIndoors', 'PlantWildNonTree', 'Quadruped', 'RectangleTable', 'Rocket',
    'RoundTable', 'Shelf', 'Ship', 'Sign', 'Skyscraper', 'Spoon', 'Starship', 'SubmachineGun', 'Sword', 'Tool',
    'Train', 'Tree', 'Truck', 'TruckNonContainer', 'Vase', 'Violin', 'Wheel', 'WheelChair'
]


def generate_category_color_map():
    """
    Generate consistent color mapping with semantic grouping.
    
    Similar categories get similar color families (same hue range),
    while different groups get maximally distinct hues.
    
    Returns:
        dict: Mapping of category name to hex color string (e.g., '#FF5733')
    """
    # Define base hue for each semantic group (maximally spread across color wheel)
    group_base_hues = {
        'aircraft': 210,          # Blue (sky)
        'ground_vehicles': 15,    # Red-Orange (roads/energy)
        'water': 180,             # Cyan (water)
        'buildings': 35,          # Orange-Brown (earth/construction)
        'furniture': 280,         # Purple (indoor/comfort)
        'music': 330,             # Magenta/Pink (arts/creativity)
        'electronics': 240,       # Blue-violet (technology)
        'lighting': 55,           # Yellow (light)
        'small_objects': 90,      # Yellow-green (everyday items)
        'weapons_tools': 0,       # Red/Crimson (danger/metal)
        'nature': 130,            # Green (organic)
        'living': 25,             # Orange (warmth/life)
        'misc': 300               # Magenta-violet (distinct)
    }
    
    category_color_map = {}
    
    for group_name, categories in CATEGORY_GROUPS.items():
        base_hue = group_base_hues[group_name]
        n_cats = len(categories)
        
        # Wider spread within family + use golden ratio for better distinction
        hue_spread = min(50, 80 / max(1, n_cats - 1)) if n_cats > 1 else 0
        golden_ratio = 137.508
        
        for idx, cat in enumerate(categories):
            # Use golden ratio offset for better color separation within group
            if n_cats == 1:
                hue = base_hue
            else:
                # Combine linear spread with golden ratio for maximum distinction
                linear_offset = (idx / (n_cats - 1) - 0.5) * hue_spread
                golden_offset = (idx * golden_ratio * 0.3) % hue_spread - hue_spread/2
                hue = base_hue + linear_offset + golden_offset * 0.3
            
            hue = hue % 360.0
            
            # More aggressive saturation and value variation for distinction
            # Use prime number patterns for better visual separation
            sat_levels = [0.95, 0.75, 0.85, 0.65, 0.90, 0.70, 0.80]
            val_levels = [0.95, 0.80, 0.90, 0.75, 0.85, 0.70, 0.92]
            
            sat = sat_levels[idx % len(sat_levels)]
            val = val_levels[idx % len(val_levels)]
            
            r, g, b = colorsys.hsv_to_rgb(hue / 360.0, sat, val)
            hex_color = '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))
            category_color_map[cat] = hex_color
    
    return category_color_map


# Global color map instance (computed once)
CATEGORY_COLOR_MAP = generate_category_color_map()


def get_category_color(category_name, default='#999999'):
    """
    Get the hex color for a specific category.
    
    Parameters:
        category_name (str): Name of the category
        default (str): Default color if category not found
        
    Returns:
        str: Hex color string (e.g., '#FF5733')
    """
    return CATEGORY_COLOR_MAP.get(category_name, default)

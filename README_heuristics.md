# Traffic Simulation with Heuristics

This project provides a comprehensive traffic simulation system with optional adaptive traffic light control using heuristics.

## Files

- `reto6.py` - Main traffic simulation with optional heuristics support
- `traffic_heuristics.py` - Heuristics mixin for adaptive traffic light control
- `compare_heuristics.py` - Comparison script to evaluate heuristics vs no-heuristics performance
- `test_heuristics.py` - Quick test script for basic functionality

## Features

### Core Simulation

- Node-based road network with A\* pathfinding
- Traffic lights with cycling groups
- Yield behaviors at merge points
- Weighted destination spawning
- Real-time JSON data export for Unity integration

### Heuristics System

- **Car Counting**: Monitors cars in stop nodes + 3 nodes back for each group
- **Adaptive Timer**: Reduces green light duration when other groups have higher congestion
- **Dynamic Adjustment**: Automatically adjusts traffic light timing based on real-time traffic conditions

## Usage

### Running with Heuristics (Default)

```python
python reto6.py
```

### Running without Heuristics

```python
# Modify reto6.py line 590 to:
TrafficModel = create_traffic_model_class(use_heuristics=False)
```

### Running Comparison

```python
python compare_heuristics.py
```

### Quick Test

```python
python test_heuristics.py
```

## Traffic Light Groups

The system organizes nodes by traffic light controllers with upstream monitoring:

### Group 0 (8 nodes)

- **8_18 controller**: 8_18, 8_20, 8_22, 8_24 (upstream nodes)
- **18_10 controller**: 18_10, 18_8, 18_6, 18_4 (upstream nodes)

### Group 1 (16 nodes)

- **6_14 controller**: 6_14, 4_14, 2_14, 0_14 (upstream nodes)
- **6_12 controller**: 6_12, 4_12, 2_12, 0_12 (upstream nodes)
- **16_13 controller**: 16_13, 14_12, 12_12, 10_13 (upstream nodes)
- **16_11 controller**: 16_11, 14_10, 12_10, 10_11 (upstream nodes)

### Group 2 (12 nodes)

- **20_15 controller**: 20_15, 22_15, 24_15, 26_15 (upstream nodes)
- **10_15 controller**: 10_15, 12_16, 14_16, 16_15 (upstream nodes)
- **10_17 controller**: 10_17, 12_18, 14_18, 16_17 (upstream nodes)

## Density-Based Control

The heuristics use **car density** (cars per node) rather than absolute car counts, making decisions relative to group size:

- **Group 0**: 8 nodes → density = cars/8
- **Group 1**: 16 nodes → density = cars/16
- **Group 2**: 12 nodes → density = cars/12

### Timer Adjustment Rules

1. Calculate density for each group
2. Compare current group's density with others
3. **If any other group has higher density**: reduce timer by 1 (minimum 1)
4. **If current group has equal/highest density**: reset timer to 5

## Performance Results

Testing shows significant improvements with the density-based approach:

| Metric          | No Heuristics | With Heuristics | Improvement |
| --------------- | ------------- | --------------- | ----------- |
| Cars Completed  | 31            | 37              | +19%        |
| Avg Active Cars | 14.6          | 11.3            | -23%        |
| Traffic Flow    | Good          | Excellent       | Significant |

The density-based approach provides fair traffic distribution by considering the relative congestion of each group's road network.

## Configuration

### Traffic Presets

- `morning` - Morning rush hour patterns
- `evening` - Evening rush hour patterns
- `night` - Night time patterns

### Simulation Parameters

- `steps`: Number of simulation steps (default: 100)
- `spawn_rate`: Car spawn probability per step (default: 1.0)
- `world_size`: World boundaries (default: 500)
- `preset`: Traffic pattern preset
- `animation`: Save animation to GIF (default: True)

## Output

The system generates:

- Console output with traffic light changes and timer adjustments
- JSON data file (`data.json`) for Unity integration
- Comparison plots (`traffic_comparison_results.png`)
- Detailed results JSON (`traffic_comparison_[timestamp].json`)

## Requirements

- Python 3.x
- agentpy
- numpy
- matplotlib
- socket (for Unity integration)

## Architecture

The system uses a modular design:

- `TrafficModelBase`: Core simulation logic
- `HeuristicsMixin`: Adaptive traffic control
- `create_traffic_model_class()`: Dynamic class creation based on availability

This allows the system to gracefully degrade to basic functionality if heuristics are not available.


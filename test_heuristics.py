#!/usr/bin/env python3
"""
Quick test script for heuristics comparison
"""

import compare_heuristics as comp

if __name__ == "__main__":
    # Run a quick comparison with fewer steps for testing
    print("Running quick heuristics comparison test...")

    # Override the main function to use fewer steps
    import compare_heuristics as comp

    # Run comparison with shorter simulation
    metrics_no_h, time_no_h = comp.run_simulation_with_heuristics(
        steps=50, preset="morning", use_heuristics=False
    )
    metrics_h, time_h = comp.run_simulation_with_heuristics(
        steps=50, preset="morning", use_heuristics=True
    )

    # Show basic results
    print("\nQuick Test Results:")
    print(f"No heuristics - Avg cars: {comp.np.mean(metrics_no_h['active_cars']):.1f}")
    print(f"With heuristics - Avg cars: {comp.np.mean(metrics_h['active_cars']):.1f}")
    print(f"Time - No heuristics: {time_no_h:.2f}s")
    print(f"Time - With heuristics: {time_h:.2f}s")

    print("\nTest completed successfully!")

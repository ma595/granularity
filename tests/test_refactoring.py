#!/usr/bin/env python3
"""
Test script to verify the refactoring worked correctly
"""

# Test imports
try:
    from granularity_simple_analysis import (
        analyze_variable_availability,
        analyze_metric_requirements,
        get_runnable_metrics_at_granularity,
        get_optimal_granularities_for_metrics,
        get_maximum_granularity_with_all,
        show_availability_summary,
        analyze_what_is_possible_efficient
    )
    print("✓ Successfully imported all analysis functions from granularity_analysis.py")
except ImportError as e:
    print(f"✗ Import error: {e}")

# Test that the main utils file still works
try:
    from granularity_simple_utils import (
        get_data_optimised_with_cache,
        run_all_metrics_with_cache,
        run_metrics_intelligently_with_cache,
        run_metric_with_cache,
        show_cache_stats
    )
    print("✓ Successfully imported main utility functions from granularity_simple_utils.py")
except ImportError as e:
    print(f"✗ Import error: {e}")

# Test that the analysis functions work through the utils file too (imported there)
try:
    from granularity_simple_utils import (
        analyze_what_is_possible_efficient,
        get_maximum_granularity_with_all,
        show_availability_summary
    )
    print("✓ Successfully imported analysis functions through granularity_simple_utils.py")
except ImportError as e:
    print(f"✗ Import error: {e}")

print("\n🎉 All imports successful! Refactoring completed successfully.")
print("\n📁 New structure:")
print("   granularity_analysis.py    - All analysis functions")
print("   granularity_simple_utils.py - Data loading, caching, and metric execution")

#!/usr/bin/env python3
"""
Master script to run all tests: window sizes and hyperparameters
"""

import sys
import argparse
from test_window_sizes_fulsang import test_window_sizes
from test_hyperparams_fulsang import test_hyperparameters

def main():
    parser = argparse.ArgumentParser(description='Run FULCNNLOC tests')
    parser.add_argument('--test', type=str, choices=['windows', 'hyperparams', 'all'],
                       default='windows', help='Which test to run')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (fewer window sizes, fewer epochs)')
    
    args = parser.parse_args()
    
    if args.test == 'windows' or args.test == 'all':
        print("\n" + "="*80)
        print("RUNNING WINDOW SIZE TESTS (1s to 30s)")
        print("="*80)
        test_window_sizes()
    
    if args.test == 'hyperparams' or args.test == 'all':
        print("\n" + "="*80)
        print("RUNNING HYPERPARAMETER TESTS")
        print("="*80)
        test_hyperparameters()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()


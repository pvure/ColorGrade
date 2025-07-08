#!/usr/bin/env python3
"""
Test script for the color analyzer
"""

import sys
import os
sys.path.append('src')

from color_analyzer import ColorAnalyzer

def test_color_analyzer():
    """Test the color analyzer with a sample video"""
    
    print("Testing Color Analyzer...")
    
    # Initialize analyzer
    analyzer = ColorAnalyzer(sample_frames=10)  # Use fewer frames for testing
    
    # You'll need to provide a test video file
    test_video = input("Enter the path to your video file: ").strip()
    
    if not os.path.exists(test_video):
        print(f"Please provide a test video file: {test_video}")
        print("You can use any MP4 video file for testing.")
        return
    
    try:
        # Analyze the video
        print("Analyzing video...")
        results = analyzer.analyze_video(test_video)
        
        # Print some results
        print("\n=== Analysis Results ===")
        print(f"Video dimensions: {results['video_info']['width']}x{results['video_info']['height']}")
        print(f"FPS: {results['video_info']['fps']}")
        print(f"Total frames: {results['video_info']['total_frames']}")
        
        # Print RGB mean values
        rgb_mean = results['color_stats']['rgb']['mean']
        print(f"\nRGB Mean: R={rgb_mean[0]:.1f}, G={rgb_mean[1]:.1f}, B={rgb_mean[2]:.1f}")
        
        # Create visualization
        print("\nGenerating visualization...")
        analyzer.visualize_analysis("analysis_test.png")
        
        # Save analysis
        print("Saving analysis...")
        analyzer.save_analysis("analysis_test.pkl")
        
        print("\n✅ Color analyzer test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_color_analyzer()
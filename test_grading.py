#!/usr/bin/env python3
"""
Test script for the color grading system
"""

import sys
import os
sys.path.append('src')

from color_grader import ColorGrader

def test_color_grading():
    """Test the complete color grading pipeline"""
    
    print("=== Color Grading Test ===")
    
    # Check if we have the reference analysis
    if not os.path.exists("color_analysis.pkl"):
        print("❌ Reference analysis not found!")
        print("Please run the color analyzer first with your reference video.")
        print("Run: python run_analyzer.py reference_video.mp4")
        return False
    
    # Get target video path
    target_video = input("Enter path to target video (to be graded): ").strip()
    
    if not os.path.exists(target_video):
        print(f"❌ Target video not found: {target_video}")
        return False
    
    # Initialize grader
    grader = ColorGrader()
    
    try:
        # Load reference analysis
        print("Loading reference analysis...")
        grader.load_reference_analysis("color_analysis.pkl")
        
        # Test 1: Create before/after comparison
        print("\n1. Creating before/after comparison...")
        grader.analyze_target_video(target_video, sample_frames=20)
        grader.generate_transformation_luts(method='histogram_matching')
        
        success = grader.create_before_after_comparison(
            target_video, 
            "before_after_comparison.jpg", 
            frame_number=50
        )
        
        if success:
            print("✅ Before/after comparison created: before_after_comparison.jpg")
        
        # Test 2: Process a short preview
        print("\n2. Creating preview (first 90 frames)...")
        preview_success = grader.grade_video(
            target_video,
            "preview_graded.mp4",
            method='histogram_matching',
            intensity=0.8,
            preview_frames=90
        )
        
        if preview_success:
            print("✅ Preview created: preview_graded.mp4")
        
        # Test 3: Different intensity levels
        print("\n3. Testing different intensity levels...")
        for intensity in [0.5, 1.0]:
            output_name = f"graded_intensity_{intensity}.mp4"
            print(f"   Processing with intensity {intensity}...")
            
            success = grader.grade_video(
                target_video,
                output_name,
                method='histogram_matching',
                intensity=intensity,
                preview_frames=60
            )
            
            if success:
                print(f"   ✅ Created: {output_name}")
        
        # Test 4: Try statistical matching method
        print("\n4. Testing statistical matching method...")
        grader.generate_transformation_luts(method='statistical_matching')
        
        success = grader.grade_video(
            target_video,
            "graded_statistical.mp4",
            method='statistical_matching',
            intensity=0.9,
            preview_frames=60
        )
        
        if success:
            print("✅ Statistical matching result: graded_statistical.mp4")
        
        # Save the transformation LUTs
        grader.save_luts("transformation_luts.pkl")
        
        print("\n🎉 All tests completed successfully!")
        print("\nFiles created:")
        print("  - before_after_comparison.jpg")
        print("  - preview_graded.mp4")
        print("  - graded_intensity_0.5.mp4")
        print("  - graded_intensity_1.0.mp4")
        print("  - graded_statistical.mp4")
        print("  - transformation_luts.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def grade_full_video():
    """Grade a full video with user-specified parameters"""
    
    print("=== Full Video Grading ===")
    
    # Check if we have the reference analysis
    if not os.path.exists("color_analysis.pkl"):
        print("❌ Reference analysis not found!")
        print("Please run the color analyzer first with your reference video.")
        return False
    
    # Get parameters from user
    target_video = input("Enter path to target video: ").strip()
    output_video = input("Enter output video path (e.g., graded_video.mp4): ").strip()
    
    method = input("Choose method (histogram_matching/statistical_matching) [histogram_matching]: ").strip()
    if not method:
        method = 'histogram_matching'
    
    intensity_input = input("Enter intensity (0.0-1.0) [0.8]: ").strip()
    intensity = float(intensity_input) if intensity_input else 0.8
    
    if not os.path.exists(target_video):
        print(f"❌ Target video not found: {target_video}")
        return False
    
    # Initialize grader
    grader = ColorGrader()
    
    try:
        # Load reference analysis
        print("Loading reference analysis...")
        grader.load_reference_analysis("color_analysis.pkl")
        
        # Grade the video
        print(f"Grading video with method: {method}, intensity: {intensity}")
        success = grader.grade_video(
            target_video,
            output_video,
            method=method,
            intensity=intensity
        )
        
        if success:
            print(f"🎉 Video grading completed successfully!")
            print(f"Output saved to: {output_video}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error during grading: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to choose test mode"""
    
    print("Choose an option:")
    print("1. Run quick tests (creates previews and comparisons)")
    print("2. Grade full video")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        test_color_grading()
    elif choice == '2':
        grade_full_video()
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
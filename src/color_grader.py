import cv2
import numpy as np
from typing import Dict, Tuple
from scipy import interpolate
import pickle


class ColorGrader:
    """
    Applies color grading transformations to match reference video characteristics.
    """
    
    def __init__(self, reference_stats: Dict = None):
        """
        Initialize the color grader.
        
        Args:
            reference_stats: Color statistics from reference video analysis
        """
        self.reference_stats = reference_stats
        self.target_stats = None
        self.transformation_luts = {}
    
    def load_reference_analysis(self, analysis_path: str):
        """Load reference video analysis from file."""
        with open(analysis_path, 'rb') as f:
            self.reference_stats = pickle.load(f)
        print(f"Generating transformation LUTs using {method}...")
        
        self.transformation_luts = {}
        
        if method == 'histogram_matching':
            # RGB histogram matching
            self.transformation_luts['rgb'] = {}
            rgb_channels = ['r', 'g', 'b']
            for i, channel in enumerate(rgb_channels):
                source_hist = self.target_stats['histograms']['rgb'][channel]
                target_hist = self.reference_stats['histograms']['rgb'][channel]
                self.transformation_luts['rgb'][channel] = self.create_histogram_matching_lut(
                    source_hist, target_hist)
            
            # LAB histogram matching (more perceptually uniform)
            self.transformation_luts['lab'] = {}
            lab_channels = ['l', 'a', 'b']
            for i, channel in enumerate(lab_channels):
                source_hist = self.target_stats['histograms']['lab'][channel]
                target_hist = self.reference_stats['histograms']['lab'][channel]
                self.transformation_luts['lab'][channel] = self.create_histogram_matching_lut(
                    source_hist, target_hist)
        
        elif method == 'statistical_matching':
            # Statistical matching in LAB color space
            lab_luts = self.create_statistical_matching_lut(
                self.target_stats['color_stats'],
                self.reference_stats['color_stats'],
                'lab'
            )
            self.transformation_luts['lab'] = {
                'l': lab_luts[0],
                'a': lab_luts[1],
                'b': lab_luts[2]
            }
        
        print("✅ Transformation LUTs generated successfully!")
    
    def apply_lut_to_frame(self, frame: np.ndarray, method: str = 'lab') -> np.ndarray:
        """
        Apply color transformation to a single frame.
        
        Args:
            frame: Input frame (BGR format)
            method: Color space to use ('rgb' or 'lab')
            
        Returns:
            Transformed frame
        """
        if method == 'rgb':
            # Apply RGB transformation
            if 'rgb' not in self.transformation_luts:
                raise ValueError("RGB transformation LUTs not available")
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply LUTs
            rgb_frame[:, :, 0] = cv2.LUT(rgb_frame[:, :, 0], self.transformation_luts['rgb']['r'])
            rgb_frame[:, :, 1] = cv2.LUT(rgb_frame[:, :, 1], self.transformation_luts['rgb']['g'])
            rgb_frame[:, :, 2] = cv2.LUT(rgb_frame[:, :, 2], self.transformation_luts['rgb']['b'])
            
            # Convert back to BGR
            return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        elif method == 'lab':
            # Apply LAB transformation (recommended for better color accuracy)
            if 'lab' not in self.transformation_luts:
                raise ValueError("LAB transformation LUTs not available")
            
            # Convert BGR to LAB
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Apply LUTs
            lab_frame[:, :, 0] = cv2.LUT(lab_frame[:, :, 0], self.transformation_luts['lab']['l'])
            lab_frame[:, :, 1] = cv2.LUT(lab_frame[:, :, 1], self.transformation_luts['lab']['a'])
            lab_frame[:, :, 2] = cv2.LUT(lab_frame[:, :, 2], self.transformation_luts['lab']['b'])
            
            # Convert back to BGR
            return cv2.cvtColor(lab_frame, cv2.COLOR_LAB2BGR)
        
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def apply_adaptive_color_grading(self, frame: np.ndarray, 
                                   intensity: float = 1.0) -> np.ndarray:
        """
        Apply adaptive color grading with blending.
        
        Args:
            frame: Input frame
            intensity: Grading intensity (0.0 to 1.0)
            
        Returns:
            Graded frame
        """
        # Apply LAB transformation
        graded_frame = self.apply_lut_to_frame(frame, method='lab')
        
        # Blend with original frame based on intensity
        if intensity < 1.0:
            graded_frame = cv2.addWeighted(
                frame, 1 - intensity,
                graded_frame, intensity,
                0
            )
        
        return graded_frame
    
    def grade_video(self, input_path: str, output_path: str, 
                   method: str = 'histogram_matching',
                   intensity: float = 1.0,
                   preview_frames: int = 0) -> bool:
        """
        Apply color grading to entire video.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            method: Transformation method
            intensity: Grading intensity (0.0 to 1.0)
            preview_frames: If > 0, process only first N frames for preview
            
        Returns:
            True if successful
        """
        # Analyze target video if not done already
        if not self.target_stats:
            print("Analyzing target video...")
            self.analyze_target_video(input_path)
        
        # Generate transformation LUTs
        self.generate_transformation_luts(method=method)
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if preview_frames > 0:
            total_frames = min(total_frames, preview_frames)
        
        print(f"Processing video: {width}x{height}, {total_frames} frames, {fps} FPS")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        
        try:
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply color grading
                graded_frame = self.apply_adaptive_color_grading(frame, intensity)
                
                # Write frame
                out.write(graded_frame)
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        except Exception as e:
            print(f"Error during processing: {e}")
            return False
        
        finally:
            cap.release()
            out.release()
        
        print(f"✅ Video grading completed! Output saved to: {output_path}")
        return True
    
    def create_before_after_comparison(self, input_path: str, 
                                     output_path: str,
                                     frame_number: int = 100) -> bool:
        """
        Create a side-by-side comparison of original vs graded frame.
        
        Args:
            input_path: Path to input video
            output_path: Path to save comparison image
            frame_number: Frame number to compare
            
        Returns:
            True if successful
        """
        if not self.transformation_luts:
            print("No transformation LUTs available. Generate them first.")
            return False
        
        # Open video and get specific frame
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Could not read frame {frame_number}")
            return False
        
        # Apply grading
        graded_frame = self.apply_adaptive_color_grading(frame)
        
        # Create side-by-side comparison
        comparison = np.hstack([frame, graded_frame])
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Graded", (frame.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save comparison
        cv2.imwrite(output_path, comparison)
        print(f"Comparison saved to: {output_path}")
        return True
    
    def save_luts(self, save_path: str):
        """Save transformation LUTs to file."""
        with open(save_path, 'wb') as f:
            pickle.dump(self.transformation_luts, f)
        print(f"Transformation LUTs saved to: {save_path}")
    
    def load_luts(self, load_path: str):
        """Load transformation LUTs from file."""
        with open(load_path, 'rb') as f:
            self.transformation_luts = pickle.load(f)
        print(f"Transformation LUTs loaded from: {load_path}")


# Example usage
if __name__ == "__main__":
    print("Color grader module loaded successfully!")
    print("Use this module by importing: from color_grader import ColorGrader")(f"Reference analysis loaded from: {analysis_path}")
    
    def analyze_target_video(self, video_path: str, sample_frames: int = 30) -> Dict:
        """
        Analyze the target video to understand its color characteristics.
        
        Args:
            video_path: Path to target video
            sample_frames: Number of frames to sample
            
        Returns:
            Color statistics for target video
        """
        from color_analyzer import ColorAnalyzer
        
        analyzer = ColorAnalyzer(sample_frames=sample_frames)
        self.target_stats = analyzer.analyze_video(video_path)
        
        print("Target video analyzed successfully!")
        return self.target_stats
    
    def create_histogram_matching_lut(self, source_hist: np.ndarray, 
                                    target_hist: np.ndarray) -> np.ndarray:
        """
        Create a Look-Up Table (LUT) for histogram matching.
        
        Args:
            source_hist: Histogram of source (target video)
            target_hist: Histogram of target (reference video)
            
        Returns:
            LUT array for pixel value mapping
        """
        # Normalize histograms
        source_hist = source_hist.astype(np.float64)
        target_hist = target_hist.astype(np.float64)
        
        # Calculate cumulative distribution functions (CDFs)
        source_cdf = np.cumsum(source_hist)
        target_cdf = np.cumsum(target_hist)
        
        # Normalize CDFs to [0, 1]
        source_cdf = source_cdf / source_cdf[-1]
        target_cdf = target_cdf / target_cdf[-1]
        
        # Create LUT by finding closest CDF values
        lut = np.zeros(256, dtype=np.uint8)
        
        for i in range(256):
            # Find the closest value in target CDF
            closest_idx = np.argmin(np.abs(target_cdf - source_cdf[i]))
            lut[i] = closest_idx
        
        return lut
    
    def create_statistical_matching_lut(self, source_stats: Dict, 
                                      target_stats: Dict, 
                                      color_space: str = 'lab') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create LUTs based on statistical matching (mean, std matching).
        
        Args:
            source_stats: Statistics from source video
            target_stats: Statistics from target video
            color_space: Color space to use ('rgb', 'hsv', 'lab')
            
        Returns:
            Tuple of LUTs for each channel
        """
        source_mean = source_stats[color_space]['mean']
        source_std = source_stats[color_space]['std']
        target_mean = target_stats[color_space]['mean']
        target_std = target_stats[color_space]['std']
        
        luts = []
        
        for i in range(3):
            # Calculate transformation parameters
            if source_std[i] > 0:
                scale = target_std[i] / source_std[i]
            else:
                scale = 1.0
            
            shift = target_mean[i] - source_mean[i] * scale
            
            # Create LUT
            lut = np.zeros(256, dtype=np.uint8)
            for j in range(256):
                new_val = j * scale + shift
                lut[j] = np.clip(new_val, 0, 255)
            
            luts.append(lut)
        
        return tuple(luts)
    
    def generate_transformation_luts(self, method: str = 'histogram_matching'):
        """
        Generate Look-Up Tables for color transformation.
        
        Args:
            method: Transformation method ('histogram_matching', 'statistical_matching')
        """
        if not self.reference_stats or not self.target_stats:
            raise ValueError("Both reference and target statistics are required")
        
        print
import cv2
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class ColorAnalyzer:
    """
    Analyzes color properties of a video to extract grading characteristics.
    """
    
    def __init__(self, sample_frames: int = 50):
        """
        Initialize the color analyzer.
        
        Args:
            sample_frames: Number of frames to sample for analysis
        """
        self.sample_frames = sample_frames
        self.color_stats = {}
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        Analyze color properties of a video.
        
        Args:
            video_path: Path to the reference video
            
        Returns:
            Dictionary containing color statistics and histograms
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Analyzing video: {width}x{height}, {total_frames} frames, {fps} FPS")
        
        # Sample frames evenly throughout the video
        frame_indices = np.linspace(0, total_frames - 1, self.sample_frames, dtype=int)
        
        # Initialize arrays to store color data
        rgb_pixels = []
        hsv_pixels = []
        lab_pixels = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Sample pixels (every 10th pixel to reduce memory usage)
            rgb_sample = rgb_frame[::10, ::10].reshape(-1, 3)
            hsv_sample = hsv_frame[::10, ::10].reshape(-1, 3)
            lab_sample = lab_frame[::10, ::10].reshape(-1, 3)
            
            rgb_pixels.append(rgb_sample)
            hsv_pixels.append(hsv_sample)
            lab_pixels.append(lab_sample)
        
        cap.release()
        
        # Combine all sampled pixels
        rgb_pixels = np.vstack(rgb_pixels)
        hsv_pixels = np.vstack(hsv_pixels)
        lab_pixels = np.vstack(lab_pixels)
        
        # Calculate color statistics
        color_stats = self._calculate_color_stats(rgb_pixels, hsv_pixels, lab_pixels)
        
        # Generate histograms
        histograms = self._generate_histograms(rgb_pixels, hsv_pixels, lab_pixels)
        
        # Store results
        self.color_stats = {
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames
            },
            'color_stats': color_stats,
            'histograms': histograms
        }
        
        return self.color_stats
    
    def _calculate_color_stats(self, rgb_pixels: np.ndarray, 
                             hsv_pixels: np.ndarray, 
                             lab_pixels: np.ndarray) -> Dict:
        """Calculate color statistics for different color spaces."""
        
        stats = {}
        
        # RGB statistics
        stats['rgb'] = {
            'mean': np.mean(rgb_pixels, axis=0),
            'std': np.std(rgb_pixels, axis=0),
            'median': np.median(rgb_pixels, axis=0),
            'percentile_25': np.percentile(rgb_pixels, 25, axis=0),
            'percentile_75': np.percentile(rgb_pixels, 75, axis=0)
        }
        
        # HSV statistics
        stats['hsv'] = {
            'mean': np.mean(hsv_pixels, axis=0),
            'std': np.std(hsv_pixels, axis=0),
            'median': np.median(hsv_pixels, axis=0),
            'percentile_25': np.percentile(hsv_pixels, 25, axis=0),
            'percentile_75': np.percentile(hsv_pixels, 75, axis=0)
        }
        
        # LAB statistics
        stats['lab'] = {
            'mean': np.mean(lab_pixels, axis=0),
            'std': np.std(lab_pixels, axis=0),
            'median': np.median(lab_pixels, axis=0),
            'percentile_25': np.percentile(lab_pixels, 25, axis=0),
            'percentile_75': np.percentile(lab_pixels, 75, axis=0)
        }
        
        return stats
    
    def _generate_histograms(self, rgb_pixels: np.ndarray, 
                           hsv_pixels: np.ndarray, 
                           lab_pixels: np.ndarray) -> Dict:
        """Generate color histograms for different color spaces."""
        
        histograms = {}
        
        # RGB histograms
        histograms['rgb'] = {
            'r': np.histogram(rgb_pixels[:, 0], bins=256, range=(0, 256))[0],
            'g': np.histogram(rgb_pixels[:, 1], bins=256, range=(0, 256))[0],
            'b': np.histogram(rgb_pixels[:, 2], bins=256, range=(0, 256))[0]
        }
        
        # HSV histograms
        histograms['hsv'] = {
            'h': np.histogram(hsv_pixels[:, 0], bins=180, range=(0, 180))[0],
            's': np.histogram(hsv_pixels[:, 1], bins=256, range=(0, 256))[0],
            'v': np.histogram(hsv_pixels[:, 2], bins=256, range=(0, 256))[0]
        }
        
        # LAB histograms
        histograms['lab'] = {
            'l': np.histogram(lab_pixels[:, 0], bins=256, range=(0, 256))[0],
            'a': np.histogram(lab_pixels[:, 1], bins=256, range=(0, 256))[0],
            'b': np.histogram(lab_pixels[:, 2], bins=256, range=(0, 256))[0]
        }
        
        return histograms
    
    def visualize_analysis(self, save_path: str = None):
        """
        Create visualizations of the color analysis.
        
        Args:
            save_path: Path to save the visualization (optional)
        """
        if not self.color_stats:
            raise ValueError("No color analysis data available. Run analyze_video() first.")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Color Analysis Results', fontsize=16)
        
        # RGB histograms
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            axes[0, i].hist(range(256), bins=256, weights=self.color_stats['histograms']['rgb'][color[0]], 
                          color=color, alpha=0.7)
            axes[0, i].set_title(f'RGB {color.upper()} Channel')
            axes[0, i].set_xlabel('Pixel Value')
            axes[0, i].set_ylabel('Frequency')
        
        # HSV histograms
        hsv_colors = ['orange', 'purple', 'gray']
        hsv_labels = ['Hue', 'Saturation', 'Value']
        for i, (color, label) in enumerate(zip(hsv_colors, hsv_labels)):
            channel = label.lower()[0]
            bins = 180 if channel == 'h' else 256
            axes[1, i].hist(range(bins), bins=bins, weights=self.color_stats['histograms']['hsv'][channel], 
                          color=color, alpha=0.7)
            axes[1, i].set_title(f'HSV {label} Channel')
            axes[1, i].set_xlabel('Value')
            axes[1, i].set_ylabel('Frequency')
        
        # LAB histograms
        lab_colors = ['black', 'green', 'blue']
        lab_labels = ['L*', 'a*', 'b*']
        for i, (color, label) in enumerate(zip(lab_colors, lab_labels)):
            channel = label.lower()[0]
            axes[2, i].hist(range(256), bins=256, weights=self.color_stats['histograms']['lab'][channel], 
                          color=color, alpha=0.7)
            axes[2, i].set_title(f'LAB {label} Channel')
            axes[2, i].set_xlabel('Value')
            axes[2, i].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_analysis(self, save_path: str):
        """Save the color analysis results to a file."""
        import pickle
        
        if not self.color_stats:
            raise ValueError("No color analysis data available. Run analyze_video() first.")
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.color_stats, f)
        
        print(f"Color analysis saved to: {save_path}")
    
    def load_analysis(self, load_path: str):
        """Load color analysis results from a file."""
        import pickle
        
        with open(load_path, 'rb') as f:
            self.color_stats = pickle.load(f)
        
        print(f"Color analysis loaded from: {load_path}")
        return self.color_stats


# Example usage
if __name__ == "__main__":
    print("Color analyzer module loaded successfully!")
    print("Use this module by importing: from color_analyzer import ColorAnalyzer")
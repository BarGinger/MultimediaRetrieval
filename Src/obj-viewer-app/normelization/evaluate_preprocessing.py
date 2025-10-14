"""
Evaluation Script for Unified Preprocessing & Normalization Pipeline
Analyzes and visualizes the quality of preprocessing and normalization results.

Generates:
- Comprehensive evaluation plots
- CSV files with detailed metrics
- Compliance reports according to technical tips
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import open3d as o3d
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.obj_parser import OBJParser
from core.file_index import get_file_tree

class PreprocessingEvaluator:
    """
    Comprehensive evaluator for preprocessing and normalization pipeline results.
    Follows technical tips guidelines for validation.
    """
    
    def __init__(self, 
                 original_dataset_dir="Datasets/Data_sampled_resampled",
                 processed_dataset_dir="Datasets/UnifiedPreprocessed/Data_sampled_resampled",
                 output_dir="normeliztion/evaluation_results"):
        """
        Initialize evaluator with dataset directories
        
        Args:
            original_dataset_dir: Path to original dataset
            processed_dataset_dir: Path to processed dataset
            output_dir: Directory to save evaluation results
        """
        # Resolve paths relative to project root
        base_dir = Path(__file__).parent.parent.parent.parent
        self.original_dataset_dir = base_dir / original_dataset_dir
        self.processed_dataset_dir = base_dir / processed_dataset_dir
        self.output_dir = base_dir / output_dir
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "csv").mkdir(exist_ok=True)
        
        print(f"📁 Original dataset: {self.original_dataset_dir}")
        print(f"📁 Processed dataset: {self.processed_dataset_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Initialize results storage
        self.evaluation_results = []
        self.face_area_results = []
        self.edge_length_results = []
        
    def load_mesh_safely(self, filepath):
        """Safely load a mesh file with error handling"""
        try:
            if filepath.suffix.lower() == '.obj':
                vertices, faces = OBJParser.parse_obj_file(str(filepath))
                return vertices, faces
            else:
                # Try Open3D for other formats
                mesh = o3d.io.read_triangle_mesh(str(filepath))
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                return vertices, faces
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None, None
    
    def compute_mesh_statistics(self, vertices, faces, label=""):
        """Compute comprehensive mesh statistics"""
        if vertices is None or len(vertices) == 0:
            return None
            
        stats = {
            'label': label,
            'num_vertices': len(vertices),
            'num_faces': len(faces) if faces is not None else 0
        }
        
        # Basic geometry statistics
        if len(vertices) > 0:
            # Center and dimensions
            center = np.mean(vertices, axis=0)
            stats['center_x'] = center[0]
            stats['center_y'] = center[1]
            stats['center_z'] = center[2]
            stats['center_distance_from_origin'] = np.linalg.norm(center)
            
            # Bounding box
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            dimensions = max_coords - min_coords
            stats['bbox_x'] = dimensions[0]
            stats['bbox_y'] = dimensions[1]
            stats['bbox_z'] = dimensions[2]
            stats['max_dimension'] = np.max(dimensions)
            stats['min_dimension'] = np.min(dimensions)
            stats['dimension_ratio'] = stats['max_dimension'] / max(stats['min_dimension'], 1e-10)
            
            # Volume estimation (bounding box volume)
            stats['bbox_volume'] = np.prod(dimensions)
            
        # Face-based statistics
        if faces is not None and len(faces) > 0:
            face_areas = []
            edge_lengths = []
            face_centers = []
            
            for face in faces:
                if len(face) >= 3:
                    # Get face vertices
                    face_verts = vertices[face[:3]]  # Use first 3 vertices
                    
                    # Face center
                    face_center = np.mean(face_verts, axis=0)
                    face_centers.append(face_center)
                    
                    # Face area using cross product
                    v1 = face_verts[1] - face_verts[0]
                    v2 = face_verts[2] - face_verts[0]
                    area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                    face_areas.append(area)
                    
                    # Edge lengths
                    for i in range(3):
                        edge = face_verts[(i+1)%3] - face_verts[i]
                        edge_lengths.append(np.linalg.norm(edge))
            
            if face_areas:
                stats['mean_face_area'] = np.mean(face_areas)
                stats['std_face_area'] = np.std(face_areas)
                stats['min_face_area'] = np.min(face_areas)
                stats['max_face_area'] = np.max(face_areas)
                stats['face_area_uniformity'] = 1.0 - (np.std(face_areas) / max(np.mean(face_areas), 1e-10))
                
            if edge_lengths:
                stats['mean_edge_length'] = np.mean(edge_lengths)
                stats['std_edge_length'] = np.std(edge_lengths)
                stats['min_edge_length'] = np.min(edge_lengths)
                stats['max_edge_length'] = np.max(edge_lengths)
                stats['edge_length_uniformity'] = 1.0 - (np.std(edge_lengths) / max(np.mean(edge_lengths), 1e-10))
        
        return stats
    
    def evaluate_normalization_compliance(self, stats):
        """Evaluate compliance with technical tips normalization requirements"""
        compliance = {}
        
        if stats is None:
            return compliance
        
        # Technical Tips Compliance Checks
        
        # 1. Centering compliance (barycenter should be at origin)
        center_distance = stats.get('center_distance_from_origin', float('inf'))
        compliance['centered'] = center_distance < 1e-10
        compliance['center_error'] = center_distance
        
        # 2. Scaling compliance (max dimension should be 1.0)
        max_dim = stats.get('max_dimension', 0)
        scaling_error = abs(max_dim - 1.0)
        compliance['scaled'] = scaling_error < 1e-6
        compliance['scaling_error'] = scaling_error
        
        # 3. Resolution uniformity (for remeshing evaluation)
        face_area_uniformity = stats.get('face_area_uniformity', 0)
        edge_length_uniformity = stats.get('edge_length_uniformity', 0)
        compliance['face_uniformity'] = face_area_uniformity
        compliance['edge_uniformity'] = edge_length_uniformity
        
        # 4. Overall compliance
        compliance['fully_compliant'] = compliance['centered'] and compliance['scaled']
        
        return compliance
    
    def compare_datasets(self):
        """Compare original vs processed datasets"""
        print("🔍 Analyzing dataset comparison...")
        
        # Get file lists
        if self.original_dataset_dir.exists():
            original_files = list(self.original_dataset_dir.rglob("*.obj"))
        else:
            print(f"⚠️ Original dataset not found: {self.original_dataset_dir}")
            original_files = []
            
        if self.processed_dataset_dir.exists():
            processed_files = list(self.processed_dataset_dir.rglob("*_unified.obj"))
        else:
            print(f"⚠️ Processed dataset not found: {self.processed_dataset_dir}")
            processed_files = []
        
        print(f"Found {len(original_files)} original files and {len(processed_files)} processed files")
        
        # Process files
        for processed_file in tqdm(processed_files, desc="Evaluating processed files"):
            # Extract base name and category
            base_name = processed_file.stem.replace('_unified', '')
            category = processed_file.parent.name
            
            # Find corresponding original file
            original_file = None
            for orig in original_files:
                if orig.stem == base_name and orig.parent.name == category:
                    original_file = orig
                    break
            
            # Load meshes
            processed_vertices, processed_faces = self.load_mesh_safely(processed_file)
            original_vertices, original_faces = None, None
            
            if original_file:
                original_vertices, original_faces = self.load_mesh_safely(original_file)
            
            # Compute statistics
            processed_stats = self.compute_mesh_statistics(processed_vertices, processed_faces, "processed")
            original_stats = self.compute_mesh_statistics(original_vertices, original_faces, "original") if original_vertices is not None else None
            
            if processed_stats:
                # Add metadata
                processed_stats['filename'] = base_name
                processed_stats['category'] = category
                processed_stats['original_file_found'] = original_file is not None
                
                # Add original statistics for comparison
                if original_stats:
                    for key, value in original_stats.items():
                        if key != 'label':
                            processed_stats[f'original_{key}'] = value
                
                # Evaluate compliance
                compliance = self.evaluate_normalization_compliance(processed_stats)
                processed_stats.update(compliance)
                
                # Load processing metadata if available
                metadata_file = processed_file.parent / f"{base_name}_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            processing_info = metadata.get('processing_info', {})
                            processed_stats['remeshing_applied'] = processing_info.get('remeshing_applied', False)
                            processed_stats['target_faces'] = processing_info.get('target_faces', 0)
                    except Exception as e:
                        print(f"⚠️ Error reading metadata for {base_name}: {e}")
                
                self.evaluation_results.append(processed_stats)
                
                # Store face area and edge length data for detailed analysis
                if processed_vertices is not None and processed_faces is not None:
                    face_areas, edge_lengths = self.extract_face_edge_data(processed_vertices, processed_faces)
                    
                    for area in face_areas:
                        self.face_area_results.append({
                            'filename': base_name,
                            'category': category,
                            'face_area': area,
                            'dataset_type': 'processed'
                        })
                    
                    for length in edge_lengths:
                        self.edge_length_results.append({
                            'filename': base_name,
                            'category': category,
                            'edge_length': length,
                            'dataset_type': 'processed'
                        })
                
                # Add original face/edge data if available
                if original_vertices is not None and original_faces is not None:
                    face_areas, edge_lengths = self.extract_face_edge_data(original_vertices, original_faces)
                    
                    for area in face_areas:
                        self.face_area_results.append({
                            'filename': base_name,
                            'category': category,
                            'face_area': area,
                            'dataset_type': 'original'
                        })
                    
                    for length in edge_lengths:
                        self.edge_length_results.append({
                            'filename': base_name,
                            'category': category,
                            'edge_length': length,
                            'dataset_type': 'original'
                        })
        
        print(f"✅ Processed {len(self.evaluation_results)} shapes")
    
    def extract_face_edge_data(self, vertices, faces):
        """Extract face areas and edge lengths for detailed analysis"""
        face_areas = []
        edge_lengths = []
        
        for face in faces:
            if len(face) >= 3:
                face_verts = vertices[face[:3]]
                
                # Face area
                v1 = face_verts[1] - face_verts[0]
                v2 = face_verts[2] - face_verts[0]
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                face_areas.append(area)
                
                # Edge lengths
                for i in range(3):
                    edge = face_verts[(i+1)%3] - face_verts[i]
                    edge_lengths.append(np.linalg.norm(edge))
        
        return face_areas, edge_lengths
    
    def save_csv_results(self):
        """Save evaluation results to CSV files"""
        print("💾 Saving CSV results...")
        
        # Main evaluation results
        if self.evaluation_results:
            df_main = pd.DataFrame(self.evaluation_results)
            df_main.to_csv(self.output_dir / "csv" / "preprocessing_evaluation.csv", index=False)
            print(f"   Saved: preprocessing_evaluation.csv ({len(df_main)} shapes)")
        
        # Face area results
        if self.face_area_results:
            df_faces = pd.DataFrame(self.face_area_results)
            df_faces.to_csv(self.output_dir / "csv" / "face_area_analysis.csv", index=False)
            print(f"   Saved: face_area_analysis.csv ({len(df_faces)} face areas)")
        
        # Edge length results
        if self.edge_length_results:
            df_edges = pd.DataFrame(self.edge_length_results)
            df_edges.to_csv(self.output_dir / "csv" / "edge_length_analysis.csv", index=False)
            print(f"   Saved: edge_length_analysis.csv ({len(df_edges)} edge lengths)")
        
        # Summary statistics by category
        if self.evaluation_results:
            df_main = pd.DataFrame(self.evaluation_results)
            summary_stats = df_main.groupby('category').agg({
                'centered': 'mean',
                'scaled': 'mean',
                'fully_compliant': 'mean',
                'center_error': ['mean', 'max'],
                'scaling_error': ['mean', 'max'],
                'face_uniformity': 'mean',
                'edge_uniformity': 'mean',
                'num_vertices': ['mean', 'std'],
                'num_faces': ['mean', 'std'],
                'remeshing_applied': 'mean'
            }).round(6)
            
            # Flatten column names
            summary_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary_stats.columns]
            summary_stats.to_csv(self.output_dir / "csv" / "category_summary.csv")
            print(f"   Saved: category_summary.csv ({len(summary_stats)} categories)")
    
    def generate_evaluation_plots(self):
        """Generate comprehensive evaluation plots"""
        print("📊 Generating evaluation plots...")
        
        if not self.evaluation_results:
            print("⚠️ No evaluation results to plot")
            return
        
        df = pd.DataFrame(self.evaluation_results)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Normalization Compliance Overview
        self.plot_compliance_overview(df)
        
        # 2. Technical Tips Compliance Histogram
        self.plot_compliance_histograms(df)
        
        # 3. Face Area Distributions (Technical Tips recommended)
        self.plot_face_area_distributions()
        
        # 4. Edge Length Distributions
        self.plot_edge_length_distributions()
        
        # 5. Remeshing Effectiveness
        self.plot_remeshing_effectiveness(df)
        
        # 6. Category-wise Analysis
        self.plot_category_analysis(df)
        
        # 7. Before/After Comparison
        self.plot_before_after_comparison(df)
        
        print("✅ All plots generated successfully!")
    
    def plot_compliance_overview(self, df):
        """Plot overall compliance with technical tips"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall compliance pie chart
        compliance_counts = df['fully_compliant'].value_counts()
        ax1.pie([compliance_counts.get(True, 0), compliance_counts.get(False, 0)], 
                labels=['Compliant', 'Non-Compliant'], 
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        ax1.set_title('Overall Technical Tips Compliance')
        
        # Centering error histogram
        ax2.hist(df['center_error'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(1e-10, color='red', linestyle='--', label='Target (<1e-10)')
        ax2.set_xlabel('Center Distance from Origin')
        ax2.set_ylabel('Count')
        ax2.set_title('Centering Quality')
        ax2.set_yscale('log')
        ax2.legend()
        
        # Scaling error histogram
        ax3.hist(df['scaling_error'], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(1e-6, color='red', linestyle='--', label='Target (<1e-6)')
        ax3.set_xlabel('|Max Dimension - 1.0|')
        ax3.set_ylabel('Count')
        ax3.set_title('Scaling Quality')
        ax3.set_yscale('log')
        ax3.legend()
        
        # Compliance by category
        category_compliance = df.groupby('category')['fully_compliant'].mean().sort_values(ascending=True)
        category_compliance.plot(kind='barh', ax=ax4, color='skyblue')
        ax4.set_xlabel('Compliance Rate')
        ax4.set_title('Compliance Rate by Category')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "compliance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_compliance_histograms(self, df):
        """Plot detailed compliance histograms following technical tips"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Center coordinates histograms
        ax1.hist(df['center_x'], bins=50, alpha=0.5, label='X', color='red')
        ax1.hist(df['center_y'], bins=50, alpha=0.5, label='Y', color='green')
        ax1.hist(df['center_z'], bins=50, alpha=0.5, label='Z', color='blue')
        ax1.axvline(0, color='black', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Coordinate Value')
        ax1.set_ylabel('Count')
        ax1.set_title('Barycenter Coordinates Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Bounding box dimensions
        ax2.hist(df['bbox_x'], bins=30, alpha=0.5, label='X Dimension', color='red')
        ax2.hist(df['bbox_y'], bins=30, alpha=0.5, label='Y Dimension', color='green')
        ax2.hist(df['bbox_z'], bins=30, alpha=0.5, label='Z Dimension', color='blue')
        ax2.axvline(1.0, color='black', linestyle='--', alpha=0.8, label='Target (1.0)')
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Count')
        ax2.set_title('Bounding Box Dimensions After Scaling')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Vertex count distribution
        ax3.hist(df['num_vertices'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_xlabel('Number of Vertices')
        ax3.set_ylabel('Count')
        ax3.set_title('Vertex Count Distribution')
        ax3.grid(alpha=0.3)
        
        # Face count distribution
        ax4.hist(df['num_faces'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(2000, color='red', linestyle='--', label='Target (2000)')
        ax4.set_xlabel('Number of Faces')
        ax4.set_ylabel('Count')
        ax4.set_title('Face Count Distribution After Remeshing')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "compliance_histograms.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_face_area_distributions(self):
        """Plot face area distributions as recommended in technical tips"""
        if not self.face_area_results:
            print("⚠️ No face area data available")
            return
        
        df_faces = pd.DataFrame(self.face_area_results)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Combined histogram (original vs processed)
        for dataset_type in ['original', 'processed']:
            data = df_faces[df_faces['dataset_type'] == dataset_type]['face_area']
            if len(data) > 0:
                ax1.hist(data, bins=50, alpha=0.7, label=f'{dataset_type.title()} Dataset', 
                        density=True, edgecolor='black')
        
        ax1.set_xlabel('Face Area')
        ax1.set_ylabel('Density')
        ax1.set_title('Face Area Distribution Comparison')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_yscale('log')
        
        # Log scale histogram for processed data
        processed_data = df_faces[df_faces['dataset_type'] == 'processed']['face_area']
        if len(processed_data) > 0:
            ax2.hist(processed_data, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Face Area')
            ax2.set_ylabel('Count')
            ax2.set_title('Processed Dataset: Face Area Distribution')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(alpha=0.3)
        
        # Box plot by category (processed only)
        processed_df = df_faces[df_faces['dataset_type'] == 'processed']
        if len(processed_df) > 0:
            # Select top categories for readability
            top_categories = processed_df['category'].value_counts().head(10).index
            plot_data = processed_df[processed_df['category'].isin(top_categories)]
            
            sns.boxplot(data=plot_data, x='category', y='face_area', ax=ax3)
            ax3.set_xlabel('Category')
            ax3.set_ylabel('Face Area')
            ax3.set_title('Face Area Distribution by Category (Top 10)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.set_yscale('log')
            ax3.grid(alpha=0.3)
        
        # Statistical summary
        if len(processed_data) > 0:
            stats_text = f"""Face Area Statistics (Processed):
Count: {len(processed_data):,}
Mean: {np.mean(processed_data):.2e}
Std: {np.std(processed_data):.2e}
Min: {np.min(processed_data):.2e}
Max: {np.max(processed_data):.2e}
Uniformity: {1 - np.std(processed_data)/max(np.mean(processed_data), 1e-10):.3f}"""
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax4.set_title('Face Area Statistics Summary')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "face_area_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_edge_length_distributions(self):
        """Plot edge length distributions"""
        if not self.edge_length_results:
            print("⚠️ No edge length data available")
            return
        
        df_edges = pd.DataFrame(self.edge_length_results)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Combined histogram (original vs processed)
        for dataset_type in ['original', 'processed']:
            data = df_edges[df_edges['dataset_type'] == dataset_type]['edge_length']
            if len(data) > 0:
                ax1.hist(data, bins=50, alpha=0.7, label=f'{dataset_type.title()} Dataset', 
                        density=True, edgecolor='black')
        
        ax1.set_xlabel('Edge Length')
        ax1.set_ylabel('Density')
        ax1.set_title('Edge Length Distribution Comparison')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_yscale('log')
        
        # Processed data detailed view
        processed_data = df_edges[df_edges['dataset_type'] == 'processed']['edge_length']
        if len(processed_data) > 0:
            ax2.hist(processed_data, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax2.set_xlabel('Edge Length')
            ax2.set_ylabel('Count')
            ax2.set_title('Processed Dataset: Edge Length Distribution')
            ax2.grid(alpha=0.3)
        
        # Cumulative distribution
        for dataset_type in ['original', 'processed']:
            data = df_edges[df_edges['dataset_type'] == dataset_type]['edge_length']
            if len(data) > 0:
                data_sorted = np.sort(data)
                y = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
                ax3.plot(data_sorted, y, label=f'{dataset_type.title()}', linewidth=2)
        
        ax3.set_xlabel('Edge Length')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Edge Length Cumulative Distribution')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Statistics summary
        if len(processed_data) > 0:
            stats_text = f"""Edge Length Statistics (Processed):
Count: {len(processed_data):,}
Mean: {np.mean(processed_data):.3f}
Std: {np.std(processed_data):.3f}
Min: {np.min(processed_data):.3f}
Max: {np.max(processed_data):.3f}
Uniformity: {1 - np.std(processed_data)/max(np.mean(processed_data), 1e-10):.3f}"""
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            ax4.set_title('Edge Length Statistics Summary')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "edge_length_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_remeshing_effectiveness(self, df):
        """Plot remeshing effectiveness analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Remeshing applied distribution
        remesh_counts = df['remeshing_applied'].value_counts()
        ax1.pie([remesh_counts.get(True, 0), remesh_counts.get(False, 0)], 
                labels=['Remeshed', 'Not Remeshed'], 
                autopct='%1.1f%%',
                colors=['lightblue', 'lightcoral'])
        ax1.set_title('Remeshing Application Rate')
        
        # Face count: original vs processed (where available)
        original_faces = df['original_num_faces'].dropna()
        processed_faces = df[df['original_num_faces'].notna()]['num_faces']
        
        if len(original_faces) > 0:
            ax2.scatter(original_faces, processed_faces, alpha=0.6, color='purple')
            ax2.plot([0, max(original_faces.max(), processed_faces.max())], 
                    [0, max(original_faces.max(), processed_faces.max())], 
                    'r--', alpha=0.8, label='No Change Line')
            ax2.axhline(2000, color='green', linestyle='--', alpha=0.8, label='Target (2000)')
            ax2.set_xlabel('Original Face Count')
            ax2.set_ylabel('Processed Face Count')
            ax2.set_title('Face Count: Original vs Processed')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # Face uniformity comparison
        remeshed_shapes = df[df['remeshing_applied'] == True]['face_uniformity'].dropna()
        not_remeshed_shapes = df[df['remeshing_applied'] == False]['face_uniformity'].dropna()
        
        uniformity_data = []
        labels = []
        if len(remeshed_shapes) > 0:
            uniformity_data.append(remeshed_shapes)
            labels.append('Remeshed')
        if len(not_remeshed_shapes) > 0:
            uniformity_data.append(not_remeshed_shapes)
            labels.append('Not Remeshed')
        
        if uniformity_data:
            ax3.boxplot(uniformity_data, labels=labels)
            ax3.set_ylabel('Face Area Uniformity')
            ax3.set_title('Face Uniformity: Remeshed vs Not Remeshed')
            ax3.grid(alpha=0.3)
        
        # Target compliance analysis
        target_proximity = np.abs(df['num_faces'] - 2000)
        ax4.hist(target_proximity, bins=30, alpha=0.7, color='cyan', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', label='Perfect Target (0)')
        ax4.set_xlabel('|Face Count - 2000|')
        ax4.set_ylabel('Count')
        ax4.set_title('Proximity to Target Face Count')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "remeshing_effectiveness.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_category_analysis(self, df):
        """Plot category-wise analysis"""
        # Calculate category statistics
        category_stats = df.groupby('category').agg({
            'fully_compliant': 'mean',
            'center_error': 'mean',
            'scaling_error': 'mean',
            'face_uniformity': 'mean',
            'num_faces': 'mean',
            'remeshing_applied': 'mean'
        }).round(3)
        
        # Sort by compliance rate
        category_stats = category_stats.sort_values('fully_compliant', ascending=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Compliance rate by category
        category_stats['fully_compliant'].plot(kind='barh', ax=ax1, color='lightgreen')
        ax1.set_xlabel('Compliance Rate')
        ax1.set_title('Technical Tips Compliance by Category')
        ax1.grid(axis='x', alpha=0.3)
        
        # Mean face count by category
        category_stats['num_faces'].plot(kind='barh', ax=ax2, color='lightblue')
        ax2.axvline(2000, color='red', linestyle='--', label='Target (2000)')
        ax2.set_xlabel('Mean Face Count')
        ax2.set_title('Mean Face Count by Category')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        # Remeshing rate by category
        category_stats['remeshing_applied'].plot(kind='barh', ax=ax3, color='orange')
        ax3.set_xlabel('Remeshing Rate')
        ax3.set_title('Remeshing Application Rate by Category')
        ax3.grid(axis='x', alpha=0.3)
        
        # Face uniformity by category
        category_stats['face_uniformity'].plot(kind='barh', ax=ax4, color='purple')
        ax4.set_xlabel('Face Area Uniformity')
        ax4.set_title('Face Area Uniformity by Category')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "category_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_before_after_comparison(self, df):
        """Plot before/after comparison where original data is available"""
        # Filter data where original information is available
        comparison_df = df[df['original_file_found'] == True].copy()
        
        if len(comparison_df) == 0:
            print("⚠️ No original data available for before/after comparison")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Center distance comparison
        original_centers = np.sqrt(comparison_df['original_center_x']**2 + 
                                 comparison_df['original_center_y']**2 + 
                                 comparison_df['original_center_z']**2)
        processed_centers = comparison_df['center_distance_from_origin']
        
        ax1.scatter(original_centers, processed_centers, alpha=0.6, color='red')
        ax1.set_xlabel('Original Center Distance from Origin')
        ax1.set_ylabel('Processed Center Distance from Origin')
        ax1.set_title('Centering Effectiveness')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.plot([original_centers.min(), original_centers.max()], 
                [original_centers.min(), original_centers.max()], 'k--', alpha=0.5)
        ax1.grid(alpha=0.3)
        
        # Dimension comparison
        original_max_dims = comparison_df['original_max_dimension']
        processed_max_dims = comparison_df['max_dimension']
        
        ax2.scatter(original_max_dims, processed_max_dims, alpha=0.6, color='blue')
        ax2.axhline(1.0, color='red', linestyle='--', label='Target (1.0)')
        ax2.set_xlabel('Original Max Dimension')
        ax2.set_ylabel('Processed Max Dimension')
        ax2.set_title('Scaling Effectiveness')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Face count comparison
        original_faces = comparison_df['original_num_faces']
        processed_faces = comparison_df['num_faces']
        
        ax3.scatter(original_faces, processed_faces, alpha=0.6, color='green')
        ax3.axhline(2000, color='red', linestyle='--', label='Target (2000)')
        ax3.plot([original_faces.min(), original_faces.max()], 
                [original_faces.min(), original_faces.max()], 'k--', alpha=0.5, label='No Change')
        ax3.set_xlabel('Original Face Count')
        ax3.set_ylabel('Processed Face Count')
        ax3.set_title('Remeshing Effectiveness')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Overall improvement summary
        improvements = {
            'Shapes Centered': (processed_centers < 1e-10).sum(),
            'Shapes Scaled': (np.abs(processed_max_dims - 1.0) < 1e-6).sum(),
            'Shapes Remeshed': comparison_df['remeshing_applied'].sum(),
            'Fully Compliant': comparison_df['fully_compliant'].sum()
        }
        
        improvement_text = "Processing Results Summary:\n\n"
        improvement_text += f"Total Shapes Analyzed: {len(comparison_df)}\n"
        for key, value in improvements.items():
            percentage = (value / len(comparison_df)) * 100
            improvement_text += f"{key}: {value} ({percentage:.1f}%)\n"
        
        ax4.text(0.1, 0.9, improvement_text, transform=ax4.transAxes, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax4.set_title('Processing Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "before_after_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.evaluation_results:
            print("⚠️ No evaluation results available for summary")
            return
        
        df = pd.DataFrame(self.evaluation_results)
        
        report_path = self.output_dir / "evaluation_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PREPROCESSING & NORMALIZATION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            total_shapes = len(df)
            f.write(f"OVERALL STATISTICS\n")
            f.write(f"-" * 40 + "\n")
            f.write(f"Total shapes analyzed: {total_shapes}\n")
            f.write(f"Categories analyzed: {df['category'].nunique()}\n")
            f.write(f"Original files found: {df['original_file_found'].sum()}\n\n")
            
            # Technical Tips Compliance
            f.write(f"TECHNICAL TIPS COMPLIANCE\n")
            f.write(f"-" * 40 + "\n")
            centered_shapes = df['centered'].sum()
            scaled_shapes = df['scaled'].sum()
            fully_compliant = df['fully_compliant'].sum()
            
            f.write(f"Properly centered shapes: {centered_shapes}/{total_shapes} ({centered_shapes/total_shapes*100:.1f}%)\n")
            f.write(f"Properly scaled shapes: {scaled_shapes}/{total_shapes} ({scaled_shapes/total_shapes*100:.1f}%)\n")
            f.write(f"Fully compliant shapes: {fully_compliant}/{total_shapes} ({fully_compliant/total_shapes*100:.1f}%)\n\n")
            
            f.write(f"Mean centering error: {df['center_error'].mean():.2e}\n")
            f.write(f"Max centering error: {df['center_error'].max():.2e}\n")
            f.write(f"Mean scaling error: {df['scaling_error'].mean():.2e}\n")
            f.write(f"Max scaling error: {df['scaling_error'].max():.2e}\n\n")
            
            # Remeshing Statistics
            f.write(f"REMESHING STATISTICS\n")
            f.write(f"-" * 40 + "\n")
            remeshed_shapes = df['remeshing_applied'].sum()
            f.write(f"Shapes remeshed: {remeshed_shapes}/{total_shapes} ({remeshed_shapes/total_shapes*100:.1f}%)\n")
            f.write(f"Mean face count: {df['num_faces'].mean():.0f}\n")
            f.write(f"Target face count: 2000\n")
            f.write(f"Mean face area uniformity: {df['face_uniformity'].mean():.3f}\n")
            f.write(f"Mean edge length uniformity: {df['edge_uniformity'].mean():.3f}\n\n")
            
            # Category Performance
            f.write(f"CATEGORY PERFORMANCE (Top 10 by Compliance)\n")
            f.write(f"-" * 40 + "\n")
            category_performance = df.groupby('category').agg({
                'fully_compliant': 'mean',
                'remeshing_applied': 'mean'
            }).round(3).sort_values('fully_compliant', ascending=False)
            
            f.write(f"{'Category':<20} {'Compliance':<12} {'Remesh Rate':<12}\n")
            f.write(f"-" * 45 + "\n")
            for category, row in category_performance.head(10).iterrows():
                f.write(f"{category:<20} {row['fully_compliant']:<12.3f} {row['remeshing_applied']:<12.3f}\n")
            
            f.write(f"\n" + "=" * 80 + "\n")
            f.write(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"=" * 80 + "\n")
        
        print(f"📄 Summary report saved: {report_path}")
    
    def run_full_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("🚀 Starting comprehensive preprocessing evaluation...")
        print("=" * 60)
        
        # Step 1: Compare datasets
        self.compare_datasets()
        
        # Step 2: Save CSV results
        self.save_csv_results()
        
        # Step 3: Generate plots
        self.generate_evaluation_plots()
        
        # Step 4: Generate summary report
        self.generate_summary_report()
        
        print("\n✅ Evaluation complete!")
        print(f"📁 Results saved in: {self.output_dir}")
        print(f"📊 Plots available in: {self.output_dir / 'plots'}")
        print(f"📈 CSV files available in: {self.output_dir / 'csv'}")
        print(f"📄 Summary report: {self.output_dir / 'evaluation_summary_report.txt'}")

def main():
    """Main evaluation function"""
    print("📊 Preprocessing & Normalization Evaluation Tool")
    print("Following Technical Tips guidelines for validation")
    
    # Initialize evaluator
    dataset = "Data"
    # dataset = "Data_sampled"  # For quick testing
    evaluator = PreprocessingEvaluator(        
        original_dataset_dir=f"Datasets/{dataset}",
        processed_dataset_dir=f"Datasets/UnifiedPreprocessed/{dataset}",
        output_dir=f"Src/obj-viewer-app/normeliztion/evaluation_{dataset}_results"
    )
    
    # Run full evaluation
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()
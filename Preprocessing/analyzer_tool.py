import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import csv

def analyze_shape(shape_path):
    # OBJ file parser: extracts vertices and faces
    vertices = []
    faces = []
    with open(shape_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    # Only take x, y, z
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                face = []
                for part in parts:
                    # OBJ faces can be like 'f 1 2 3' or 'f 1/1/1 2/2/2 3/3/3'
                    idx = part.split('/')[0]
                    face.append(int(idx) - 1)  # OBJ indices start at 1
                faces.append(face)

    num_vertices = len(vertices)
    num_faces = len(faces)

    # Determine face types
    face_types = set()
    for face in faces:
        if len(face) == 3:
            face_types.add('triangle')
        elif len(face) == 4:
            face_types.add('quad')
        else:
            face_types.add(f'{len(face)}-gon')

    # Bounding box
    if vertices:
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        bbox = {
            'min': [min(xs), min(ys), min(zs)],
            'max': [max(xs), max(ys), max(zs)]
        }
    else:
        bbox = None

    return {
        'num_vertices': num_vertices,
        'num_faces': num_faces,
        'face_types': list(face_types),
        'bounding_box': bbox
    }

def analyze_database(database_path):
    results = []
    for class_name in os.listdir(database_path):
        class_folder = os.path.join(database_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        for shape_file in os.listdir(class_folder):
            if not shape_file.endswith('.obj'):
                continue
            shape_path = os.path.join(class_folder, shape_file)
            analysis = analyze_shape(shape_path)
            results.append({
                'class': class_name,
                'shape_file': shape_file,
                **analysis
            })
    return results

if __name__ == "__main__":

    # --- Toggle for sampled dataset ---
    USE_SAMPLED_DATASET = False  # Set to True to use Data_sampled, False for full Data

    database_path = 'Datasets/UnifiedPreprocessed/Data'
    suffix = 'unifiedPreprocessed_data'
    csv_file = f'analysis_results{suffix}.csv'

    fieldnames = ['class', 'shape_file', 'num_vertices', 'num_faces', 'face_types', 'bounding_box']
    GENERATE_CSV = True  # Set to True to regenerate CSV, False to use existing

    if GENERATE_CSV:
        analysis_results = analyze_database(database_path)
        # Print results to terminal
        for result in analysis_results:
            print(json.dumps(result, indent=2))

        # Save results to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in analysis_results:
                # Convert face_types and bounding_box to strings for CSV
                row = result.copy()
                row['face_types'] = ','.join(row['face_types'])
                row['bounding_box'] = json.dumps(row['bounding_box'])
                writer.writerow(row)
        print(f"Results saved to {csv_file}")

    # Load CSV and analyze
    df = pd.read_csv(csv_file)

    # (a) Average shape in database
    avg_vertices = df['num_vertices'].mean()
    avg_faces = df['num_faces'].mean()
    print(f"Average number of vertices: {avg_vertices:.2f}")
    print(f"Average number of faces: {avg_faces:.2f}")

    # (b) Outlier detection (simple: >2 std dev from mean)
    v_std = df['num_vertices'].std()
    f_std = df['num_faces'].std()
    v_outliers = df[(df['num_vertices'] > avg_vertices + 2*v_std) | (df['num_vertices'] < avg_vertices - 2*v_std)]
    f_outliers = df[(df['num_faces'] > avg_faces + 2*f_std) | (df['num_faces'] < avg_faces - 2*f_std)]
    print(f"Vertex count outliers:")
    print(v_outliers[['class','shape_file','num_vertices']])
    print(f"Face count outliers:")
    print(f_outliers[['class','shape_file','num_faces']])

    # Vertex count histogram
    plt.figure(figsize=(8,5))
    plt.hist(df['num_vertices'], bins=30, color='skyblue', edgecolor='black')
    plt.axvline(avg_vertices, color='blue', linestyle='dashed', linewidth=2, label=f'Average: {avg_vertices:.1f}')
    plt.title(f'Histogram of Vertex Counts{suffix}')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Number of Shapes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'shape_histogram_vertices{suffix}.png')
    plt.close()
    print(f'Vertex histogram saved as shape_histogram_vertices{suffix}.png')

    # Face count histogram
    plt.figure(figsize=(8,5))
    plt.hist(df['num_faces'], bins=30, color='salmon', edgecolor='black')
    plt.axvline(avg_faces, color='red', linestyle='dashed', linewidth=2, label=f'Average: {avg_faces:.1f}')
    plt.title(f'Histogram of Face Counts{suffix}')
    plt.xlabel('Number of Faces')
    plt.ylabel('Number of Shapes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'shape_histogram_faces{suffix}.png')
    plt.close()
    print(f'Face histogram saved as shape_histogram_faces{suffix}.png')

    # Shape class histogram (horizontal bar for readability)
    plt.figure(figsize=(12,10))
    class_counts = df['class'].value_counts().sort_values()
    plt.barh(class_counts.index, class_counts.values, color='mediumseagreen', edgecolor='black')
    plt.title(f'Histogram of Shape Classes{suffix}')
    plt.xlabel('Number of Shapes')
    plt.ylabel('Shape Class')
    plt.tight_layout()
    plt.savefig(f'shape_histogram_classes{suffix}.png')
    plt.close()
    print(f'Class histogram saved as shape_histogram_classes{suffix}.png')
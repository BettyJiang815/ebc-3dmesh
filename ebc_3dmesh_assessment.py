import os
import csv
import numpy as np
import cv2
from plyfile import PlyData
import time
import traceback
import pandas as pd
from pathlib import Path
import glob

GLOBAL_CONFIG = {
    # Input configuration
    "input_root_dir": r"<YOUR_INPUT_DIRECTORY>",
    "textures_directory": r"<YOUR_TEXTURES_DIRECTORY>",

    # Output configuration
    "output_root_dir": r"<YOUR_OUTPUT_DIRECTORY>",
    "models_output_subdir": "models",
    "summary_output_subdir": "summary",

    # Processing configuration
    "ply_files_to_process": [],
    "recursive_search": True,
    "edge_shrink_pixels": 3,

    # Unified threshold configuration
    "unified_threshold_method": "global_iqr",
    "iqr_multiplier": 1.5,

    # Color mapping for FECI visualization
    "feci_color_min": 0.0,
    "feci_color_max": 100.0,

    # GNECI calculation method options
    "gneci_method": "neci_gmax",  # Options: "neci_gmax", "neci_gmean"

    # FECI calculation method options
    "feci_method": "max_gneci",  # Options: "max_gneci", "max_neci", "max_neci_gmean"

    # Execution flow configuration
    "skip_data_processing": False,
    "skip_analysis": False,
}

# Core data structures
class Face:
    """Represents a face (usually triangle) in the model"""

    def __init__(self, indices, normal, texcoord=None, texnumber=0):
        self.indices = indices  # Vertex indices list
        self.normal = normal  # Normal vector
        self.texcoord = texcoord  # Texture coordinates (u0,v0, u1,v1, u2,v2)
        self.texnumber = texnumber  # Texture index


class ProcessingResult:
    """Processing result data structure"""

    def __init__(self):
        self.model_id = 0
        self.model_path = ""
        self.model_name = ""
        self.total_faces = 0
        self.processed_faces = 0
        self.edge_records = []
        self.processing_time = 0
        self.success = False
        self.error_message = ""

# Utility functions
def setup_output_directories():
    """Setup output directory structure"""
    output_root = Path(GLOBAL_CONFIG["output_root_dir"])
    models_dir = output_root / GLOBAL_CONFIG["models_output_subdir"]
    summary_dir = output_root / GLOBAL_CONFIG["summary_output_subdir"]

    for dir_path in [output_root, models_dir, summary_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return {
        "root": output_root,
        "models": models_dir,
        "summary": summary_dir
    }


def setup_model_output_directory(base_models_dir, model_name, model_id):
    """Setup dedicated output directory for single model"""
    model_folder_name = f"{Path(model_name).stem}_{model_id:02d}"
    model_dir = base_models_dir / model_folder_name

    csv_dir = model_dir / "csv_results"
    ply_dir = model_dir / "colored_ply"

    for dir_path in [model_dir, csv_dir, ply_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return {
        "model_root": model_dir,
        "csv": csv_dir,
        "ply": ply_dir
    }


def find_ply_files(root_dir, recursive=True):
    """Recursively search for PLY files and return file info"""
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Input directory does not exist: {root_dir}")
        return []

    ply_files_info = []

    if recursive:
        for ply_file in root_path.rglob("*.ply"):
            relative_path = ply_file.relative_to(root_path)
            ply_files_info.append({
                'file_path': ply_file,
                'relative_path': relative_path,
                'parent_folder': ply_file.parent.name,
                'file_name': ply_file.name
            })
    else:
        for ply_file in root_path.glob("*.ply"):
            relative_path = ply_file.relative_to(root_path)
            ply_files_info.append({
                'file_path': ply_file,
                'relative_path': relative_path,
                'parent_folder': ply_file.parent.name,
                'file_name': ply_file.name
            })

    print(f"Found {len(ply_files_info)} PLY files in {root_dir}")
    return sorted(ply_files_info, key=lambda x: x['file_path'])


def compute_face_normal(v0, v1, v2):
    """Calculate face normal vector for triangle"""
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    return normal / norm if norm != 0 else np.array([0, 0, 0])


def calculate_triangle_3d_area(v0, v1, v2):
    """Calculate geometric area of triangle in 3D space using cross product"""
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_product = np.cross(edge1, edge2)
    area = np.linalg.norm(cross_product) / 2.0
    return max(area, 1e-10)  # Avoid zero area

# PLY file processing
def load_ply(filename):
    """Load vertices, faces and texture info from PLY file"""
    print(f"Loading PLY: {Path(filename).name}")
    try:
        ply_data = PlyData.read(filename)
    except Exception as e:
        print(f"Failed to read PLY file: {e}")
        return None, [], []

    if 'vertex' not in ply_data:
        print(f"Error: No vertex element in PLY file")
        return None, [], []

    vertices_data = ply_data['vertex'].data
    try:
        vertices = np.vstack([vertices_data['x'], vertices_data['y'], vertices_data['z']]).T
    except ValueError as ve:
        print(f"Error: Failed to read vertex coordinates: {ve}")
        return None, [], []

    faces_list = []
    if 'face' not in ply_data:
        print(f"Warning: No face element in PLY file")
        return vertices, [], []

    face_element_data = ply_data['face'].data
    available_face_fields = face_element_data.dtype.names

    vertex_indices_field_name = None
    if 'vertex_indices' in available_face_fields:
        vertex_indices_field_name = 'vertex_indices'
    elif 'vertex_index' in available_face_fields:
        vertex_indices_field_name = 'vertex_index'

    if vertex_indices_field_name is None:
        print(f"Error: Missing vertex index field in face data. Available fields: {available_face_fields}")
        return vertices, [], []

    has_texcoord_field = 'texcoord' in available_face_fields
    has_texnumber_field = 'texnumber' in available_face_fields

    if not has_texcoord_field:
        print(f"Warning: Missing 'texcoord' field in face data")

    for face_info in face_element_data:
        inds_array = face_info[vertex_indices_field_name]
        inds = list(inds_array)

        if len(inds) != 3:
            continue

        processed_texcoord = None
        if has_texcoord_field:
            raw_texcoord_array = face_info['texcoord']
            if raw_texcoord_array is not None and isinstance(raw_texcoord_array, np.ndarray):
                if raw_texcoord_array.ndim == 1 and len(raw_texcoord_array) == 6:
                    processed_texcoord = list(raw_texcoord_array)
                elif raw_texcoord_array.ndim == 2 and raw_texcoord_array.shape == (3, 2):
                    processed_texcoord = list(raw_texcoord_array.flatten())

        tex_num = 0
        if has_texnumber_field:
            try:
                tex_num_val = face_info['texnumber']
                if tex_num_val is not None:
                    tex_num = int(tex_num_val)
            except (ValueError, TypeError):
                tex_num = 0

        try:
            normal = compute_face_normal(vertices[inds[0]], vertices[inds[1]], vertices[inds[2]])
        except IndexError:
            print(f"Warning: Vertex indices {inds} out of range, skipping face")
            continue

        faces_list.append(Face(inds, normal, processed_texcoord, tex_num))

    tex_files = []
    if ply_data.comments:
        for c_bytes in ply_data.comments:
            try:
                c = c_bytes.decode('utf-8', errors='ignore')
            except AttributeError:
                c = c_bytes

            if c.lower().startswith('texturefile'):
                parts = c.split(None, 1)
                if len(parts) > 1:
                    tex_files.append(parts[1].strip())

    print(f"   Vertices={len(vertices)}, Faces={len(faces_list)}, Textures={len(tex_files)}")
    return vertices, faces_list, tex_files

# Gradient calculation and sampling
def compute_gradient_image(img_path):
    """Calculate gradient image using Sobel operator and Gaussian blur"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image {img_path}")

    blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1, sigmaY=1)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.hypot(gx, gy)

    return gradient_magnitude


def bresenham_line_sample(grad_img, u_start, v_start, u_end, v_end, shrink_pixels=3):
    """Sample gradient along line using Bresenham algorithm"""
    h, w = grad_img.shape

    x0, y0 = int(round(u_start * (w - 1))), int(round((1 - v_start) * (h - 1)))
    x1, y1 = int(round(u_end * (w - 1))), int(round((1 - v_end) * (h - 1)))

    full_sampled_points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    curr_x, curr_y = x0, y0

    max_pixels_to_sample = 2 * (w + h)
    count = 0

    while count < max_pixels_to_sample:
        px = np.clip(curr_x, 0, w - 1)
        py = np.clip(curr_y, 0, h - 1)
        full_sampled_points.append((px, py))

        if curr_x == x1 and curr_y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            curr_x += sx
        if e2 < dx:
            err += dx
            curr_y += sy
        count += 1

    if not full_sampled_points:
        px = np.clip(x0, 0, w - 1)
        py = np.clip(y0, 0, h - 1)
        return np.array([grad_img[py, px]])

    total_points = len(full_sampled_points)
    if total_points <= 2 * shrink_pixels:
        mid_idx = total_points // 2
        px, py = full_sampled_points[mid_idx]
        return np.array([grad_img[py, px]])

    shrunk_points = full_sampled_points[shrink_pixels:-shrink_pixels]

    sampled_gradients = []
    for px, py in shrunk_points:
        sampled_gradients.append(grad_img[py, px])

    return np.array(sampled_gradients)

# Core data processing
def process_single_model_data(file_info, textures_dir, model_id):
    """
    Process single PLY model file and generate detailed CSV data
    Uses 3D geometric area weighting and flexible NECI/FECI calculation
    """
    result = ProcessingResult()
    result.model_id = model_id
    result.model_path = str(file_info['file_path'])
    result.model_name = file_info['file_name']

    start_time = time.time()

    try:
        loaded_data = load_ply(result.model_path)
        if loaded_data is None or loaded_data[0] is None:
            result.error_message = "PLY file loading failed"
            return result

        vertices, faces, texture_filenames = loaded_data
        if not faces:
            result.error_message = "No face data to process"
            return result

        result.total_faces = len(faces)

        # Load texture gradient maps
        texture_gradient_maps = {}
        for idx, tex_name in enumerate(texture_filenames):
            actual_tex_path = os.path.join(textures_dir, tex_name)
            ply_dir = os.path.dirname(result.model_path)

            if not os.path.exists(actual_tex_path) and ply_dir != textures_dir:
                path_try_ply_dir = os.path.join(ply_dir, tex_name)
                if os.path.exists(path_try_ply_dir):
                    actual_tex_path = path_try_ply_dir

            if not os.path.exists(actual_tex_path):
                print(f"   Warning: Texture file '{tex_name}' not found")
                continue

            try:
                texture_gradient_maps[idx] = compute_gradient_image(actual_tex_path)
            except ValueError as e:
                print(f"   Error: Processing texture {actual_tex_path} failed: {e}")
                continue

        if not texture_gradient_maps and any(f.texcoord is not None for f in faces):
            result.error_message = "Failed to load any required texture gradient maps"
            return result

        # Process faces
        edge_unique_id_counter = 0
        model_data_rows = []

        print(f"   Processing {len(faces)} faces...")

        for face_idx, current_face in enumerate(faces):
            if (face_idx + 1) % 10000 == 0:
                print(f"     Processed {face_idx + 1}/{len(faces)} faces...")

            if current_face.texcoord is None:
                continue

            active_grad_map = texture_gradient_maps.get(current_face.texnumber)
            if active_grad_map is None:
                continue

            # Get 3D coordinates of triangle vertices
            try:
                v0 = vertices[current_face.indices[0]]
                v1 = vertices[current_face.indices[1]]
                v2 = vertices[current_face.indices[2]]
            except IndexError:
                print(f"   Warning: Face {face_idx} vertex indices out of range, skipping")
                continue

            # Calculate 3D geometric area
            triangle_3d_area = calculate_triangle_3d_area(v0, v1, v2)

            # Extract UV coordinates for texture sampling
            uv_points_for_face = []
            try:
                for i in range(3):
                    u_coord = current_face.texcoord[i * 2]
                    v_coord = current_face.texcoord[i * 2 + 1]
                    uv_points_for_face.append((u_coord, v_coord))
            except (TypeError, IndexError):
                continue

            # Process three edges
            edge_metrics_for_feci_calc = []
            edge_local_vertex_pairs = [(0, 1), (1, 2), (2, 0)]
            num_edges_processed = 0

            for edge_in_face_idx, (local_v_start_idx, local_v_end_idx) in enumerate(edge_local_vertex_pairs):
                uv_start = uv_points_for_face[local_v_start_idx]
                uv_end = uv_points_for_face[local_v_end_idx]

                # Bresenham sampling (using UV coordinates to sample on texture)
                edge_grad_values = bresenham_line_sample(
                    active_grad_map,
                    uv_start[0], uv_start[1],
                    uv_end[0], uv_end[1],
                    shrink_pixels=GLOBAL_CONFIG["edge_shrink_pixels"]
                )

                if len(edge_grad_values) == 0:
                    mean_g, max_g, neci_val, gneci_val = 0.0, 0.0, 0.0, 0.0
                else:
                    mean_g = np.mean(edge_grad_values)
                    max_g = np.max(edge_grad_values)
                    neci_val = (max_g - mean_g) / (max_g + mean_g) if (max_g + mean_g) != 0 else 0.0

                    # Calculate GNECI based on configuration
                    if GLOBAL_CONFIG["gneci_method"] == "neci_gmax":
                        gneci_val = neci_val * max_g  # Original: NECI * Gmax
                    elif GLOBAL_CONFIG["gneci_method"] == "neci_gmean":
                        gneci_val = neci_val * mean_g  # Alternative: NECI * Gmean
                    else:
                        gneci_val = neci_val * max_g  # Default to original

                # Collect metrics for FECI calculation based on method
                if GLOBAL_CONFIG["feci_method"] == "max_gneci":
                    edge_metrics_for_feci_calc.append(gneci_val)
                elif GLOBAL_CONFIG["feci_method"] == "max_neci":
                    edge_metrics_for_feci_calc.append(neci_val)
                elif GLOBAL_CONFIG["feci_method"] == "max_neci_gmean":
                    edge_metrics_for_feci_calc.append(neci_val * mean_g)
                else:
                    edge_metrics_for_feci_calc.append(gneci_val)  # Default

                # Build data row
                row = {
                    "faceid": face_idx,
                    "edgeid": edge_unique_id_counter,
                    "start_vertex": current_face.indices[local_v_start_idx],
                    "end_vertex": current_face.indices[local_v_end_idx],
                    "mean_grad": mean_g,
                    "max_grad": max_g,
                    "NECI": neci_val,  # Renamed from NDTI
                    "GNECI": gneci_val,  # Renamed from NDTI_x_Area, now using configurable method
                    "FECI": None,  # Renamed from TQI, will be filled later
                    "FaceArea": triangle_3d_area,  # 3D geometric area
                    "model_id": model_id,
                    "relative_path": str(file_info['relative_path'])
                }

                model_data_rows.append(row)
                edge_unique_id_counter += 1
                num_edges_processed += 1

            # Calculate face FECI value
            feci_face_value = np.max(edge_metrics_for_feci_calc) if edge_metrics_for_feci_calc else 0.0

            # Fill back FECI values
            if num_edges_processed > 0:
                for i in range(1, num_edges_processed + 1):
                    if model_data_rows:
                        model_data_rows[-i]["FECI"] = feci_face_value

            result.processed_faces += 1

        result.edge_records = model_data_rows
        result.success = True

    except Exception as e:
        result.error_message = f"Error during processing: {str(e)}"
        traceback.print_exc()

    finally:
        result.processing_time = time.time() - start_time

    return result

# Colored PLY generation with fixed FECI color mapping
def generate_colored_ply_with_header_update(original_ply_path, output_ply_path, face_feci_map):
    """Color PLY faces based on FECI values using fixed color mapping range"""
    print(f"   Generating colored PLY: {Path(output_ply_path).name}")

    try:
        with open(original_ply_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"   Error: Cannot read original PLY file: {e}")
        return

    vertex_count = 0
    face_count = 0
    header_end_line_idx = -1
    original_header_lines = []

    face_properties = []
    in_face_element = False

    # Parse header
    for i, line_content in enumerate(lines):
        original_header_lines.append(line_content)
        stripped_line = line_content.strip()

        if stripped_line.startswith('element vertex'):
            try:
                vertex_count = int(stripped_line.split()[-1])
            except ValueError:
                print(f"   Warning: Cannot parse vertex count: {stripped_line}")
                return
        elif stripped_line.startswith('element face'):
            try:
                face_count = int(stripped_line.split()[-1])
                in_face_element = True
            except ValueError:
                print(f"   Warning: Cannot parse face count: {stripped_line}")
                return
        elif stripped_line.startswith('element ') and not stripped_line.startswith('element face'):
            in_face_element = False
        elif stripped_line == 'end_header':
            header_end_line_idx = i
            break
        elif in_face_element and stripped_line.startswith('property'):
            face_properties.append(stripped_line)

    if header_end_line_idx == -1:
        print(f"   Error: Cannot find 'end_header'")
        return

    # Analyze face property structure
    has_original_colors = any('red' in prop or 'green' in prop or 'blue' in prop for prop in face_properties)

    # Separate data lines
    vertex_data_lines = lines[header_end_line_idx + 1: header_end_line_idx + 1 + vertex_count]
    original_face_data_lines = lines[
                               header_end_line_idx + 1 + vertex_count: header_end_line_idx + 1 + vertex_count + face_count]

    # Build new header
    new_header_sans_orig_color = []
    in_face_element_section = False

    for line in original_header_lines:
        stripped = line.strip()
        if stripped.startswith("element face"):
            in_face_element_section = True
        elif stripped.startswith("element") and not stripped.startswith("element face"):
            in_face_element_section = False

        if in_face_element_section and \
                (stripped.startswith("property uchar red") or \
                 stripped.startswith("property uchar green") or \
                 stripped.startswith("property uchar blue") or \
                 stripped.startswith("property uchar alpha")):
            continue
        new_header_sans_orig_color.append(line)

    # Add color properties at correct position
    final_new_header = []
    color_props_to_add = [
        "property uchar red\n",
        "property uchar green\n",
        "property uchar blue\n"
    ]
    color_props_inserted = False

    for i, line in enumerate(new_header_sans_orig_color):
        s_line = line.strip()

        if (s_line.startswith("property list") and
                ("vertex_indices" in s_line or "vertex_index" in s_line) and
                not color_props_inserted):
            final_new_header.append(line)
            final_new_header.extend(color_props_to_add)
            color_props_inserted = True
        elif s_line == "end_header" and not color_props_inserted and face_count > 0:
            final_new_header.extend(color_props_to_add)
            final_new_header.append(line)
            color_props_inserted = True
        else:
            final_new_header.append(line)

    # Use fixed FECI color mapping range
    feci_min = GLOBAL_CONFIG["feci_color_min"]
    feci_max = GLOBAL_CONFIG["feci_color_max"]

    def get_color_from_feci(feci_value):
        """Color mapping function based on fixed FECI range"""
        if feci_value is None:
            return 128, 128, 128  # Gray for no data

        # Clamp FECI value to fixed range
        feci_clamped = np.clip(feci_value, feci_min, feci_max)

        # Normalize based on fixed range
        if feci_max == feci_min:
            t = 0.5
        else:
            t = (feci_clamped - feci_min) / (feci_max - feci_min)

        t = np.clip(t, 0, 1)

        # Color interpolation: Low FECI -> High FECI
        # t=0 (Low FECI, high quality)
        # t=1 (High FECI, low quality)
        r = int(46 + (211 - 46) * t)
        g = int(125 - (125 - 47) * t)
        b = int(50 - (50 - 47) * t)

        # Ensure RGB values are in valid range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        return r, g, b

    # Reconstruct face data lines
    new_face_data_lines = []
    num_faces_to_process = min(face_count, len(original_face_data_lines)) if face_count > 0 else len(
        original_face_data_lines)

    for face_line_idx in range(num_faces_to_process):
        line_content = original_face_data_lines[face_line_idx]
        parts = line_content.strip().split()

        if not parts or not parts[0].isdigit():
            new_face_data_lines.append(line_content)
            continue

        num_verts_in_face = int(parts[0])
        current_feci = face_feci_map.get(face_line_idx, feci_min)  # Default to min FECI
        r_val, g_val, b_val = get_color_from_feci(current_feci)

        new_parts = []
        part_idx = 0

        # 1. Vertex count
        new_parts.append(parts[part_idx])
        part_idx += 1

        # 2. Vertex indices
        for _ in range(num_verts_in_face):
            if part_idx < len(parts):
                new_parts.append(parts[part_idx])
                part_idx += 1

        # 3. Skip original colors if exist
        if has_original_colors:
            part_idx += 3

        # 4. Add new colors
        new_parts.extend([str(r_val), str(g_val), str(b_val)])

        # 5. Add remaining properties
        remaining_parts = parts[part_idx:]
        new_parts.extend(remaining_parts)

        new_face_data_lines.append(' '.join(new_parts) + '\n')

    # Write new PLY file
    try:
        with open(output_ply_path, 'w', encoding='utf-8') as f:
            f.writelines(final_new_header)
            f.writelines(vertex_data_lines)
            f.writelines(new_face_data_lines)

        print(f"     Colored PLY generated (FECI range: {feci_min:.1f}-{feci_max:.1f})")

    except Exception as e:
        print(f"     Cannot write colored PLY file: {e}")


# =============================================================================
# Stage 2: Collect FECI data and calculate unified threshold
# =============================================================================
def collect_feci_from_csv_files(models_output_dir):
    """Collect all FECI data from saved CSV files"""
    print(f"\nCollecting FECI data from CSV files...")

    all_feci_values = []
    model_info_list = []

    # Find all detailed CSV files in model directories
    csv_pattern = models_output_dir / "*" / "csv_results" / "*_detailed_results.csv"
    csv_files = glob.glob(str(csv_pattern))

    print(f"Found {len(csv_files)} detailed CSV files")

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        model_folder = csv_path.parent.parent.name

        try:
            # Read CSV file
            df = pd.read_csv(csv_file)

            if 'FECI' not in df.columns:
                print(f"   Warning: No FECI column in {csv_path.name}")
                continue

            # Extract face-level FECI data (deduplicated)
            face_df = df.groupby('faceid').first()
            feci_values = face_df['FECI'].dropna().tolist()

            all_feci_values.extend(feci_values)

            # Record model info
            model_info = {
                'csv_file': csv_file,
                'model_folder': model_folder,
                'model_id': df['model_id'].iloc[0] if 'model_id' in df.columns else 0,
                'face_count': len(face_df),
                'feci_count': len(feci_values)
            }
            model_info_list.append(model_info)

            print(f"   {csv_path.name}: {len(feci_values)} FECI values")

        except Exception as e:
            print(f"   Failed to read {csv_path.name}: {e}")

    print(f"Total FECI data points: {len(all_feci_values)}")
    print(f"Models involved: {len(model_info_list)}")

    return all_feci_values, model_info_list


def calculate_unified_threshold(all_feci_data, method="global_iqr"):
    """Calculate unified FECI anomaly detection threshold"""
    print(f"Calculating unified threshold (method: {method})")

    if not all_feci_data:
        print("   No FECI data, using default threshold 1.0")
        return 1.0

    if method == "global_iqr":
        q25 = np.percentile(all_feci_data, 25)
        q75 = np.percentile(all_feci_data, 75)
        iqr = q75 - q25
        threshold = q75 + GLOBAL_CONFIG["iqr_multiplier"] * iqr

        print(f"   Global Q25: {q25:.6f}")
        print(f"   Global Q75: {q75:.6f}")
        print(f"   Global IQR: {iqr:.6f}")
        print(f"   Unified threshold: {threshold:.6f}")

    else:
        raise ValueError(f"Unsupported threshold calculation method: {method}")

    return threshold


# =============================================================================
# Stage 3: Generate quality analysis based on unified threshold
# =============================================================================
def analyze_model_quality_from_csv(csv_file, unified_threshold, model_output_dirs):
    """Generate quality analysis based on existing CSV file and unified threshold"""
    try:
        # Read CSV data
        df = pd.read_csv(csv_file)

        if df.empty or 'FECI' not in df.columns:
            print(f"   Invalid CSV file or missing FECI data")
            return None

        # Extract model info
        model_id = df['model_id'].iloc[0] if 'model_id' in df.columns else 0
        model_name = Path(csv_file).stem.replace('_detailed_results', '') + '.ply'

        # Face-level analysis (deduplicated)
        face_df = df.groupby('faceid').first().reset_index()

        # Basic statistics
        total_faces = len(face_df)
        total_area = face_df['FaceArea'].sum()  # 3D geometric area sum
        feci_values = face_df['FECI'].values

        # Low quality face identification (using FECI absolute value comparison)
        low_quality_mask = feci_values > unified_threshold
        low_quality_faces = np.sum(low_quality_mask)
        low_quality_areas = face_df.loc[low_quality_mask, 'FaceArea'].sum()  # Sum of low quality face areas
        low_quality_area_ratio = low_quality_areas / total_area if total_area > 0 else 0
        low_quality_face_ratio = low_quality_faces / total_faces if total_faces > 0 else 0

        # Calculate percentile statistics
        feci_percentiles = {
            'P25': float(np.percentile(feci_values, 25)),
            'P50': float(np.percentile(feci_values, 50)),
            'P75': float(np.percentile(feci_values, 75)),
            'P90': float(np.percentile(feci_values, 90)),
            'P95': float(np.percentile(feci_values, 95)),
            'P99': float(np.percentile(feci_values, 99))
        }

        # Build statistics result
        model_stats = {
            'Model_ID': model_id,
            'Model_Name': model_name,
            'Total_Faces': total_faces,
            'Total_Area_3D': total_area,
            'Low_Quality_Faces': low_quality_faces,
            'Low_Quality_Face_Ratio': low_quality_face_ratio,
            'Low_Quality_Area_3D': low_quality_areas,
            'Low_Quality_Area_Ratio': low_quality_area_ratio,
            'FECI_Mean': float(np.mean(feci_values)),
            'FECI_Max': float(np.max(feci_values)),
            'FECI_Std': float(np.std(feci_values)),
            'FECI_Threshold': unified_threshold
        }

        print(
            f"   Analysis complete - Low quality face ratio: {low_quality_face_ratio:.2%}, Low quality area ratio: {low_quality_area_ratio:.2%}")

        return model_stats

    except Exception as e:
        print(f"   Analysis failed: {e}")
        return None


# =============================================================================
# Stage 4: Global summary
# =============================================================================
def generate_global_summary(all_model_stats, summary_output_dir, unified_threshold):
    """Generate global summary report"""
    print(f"Generating global summary report...")

    if not all_model_stats:
        print("No model statistics data to summarize")
        return

    summary_df = pd.DataFrame(all_model_stats)

    # Save global statistics summary
    global_stats_path = summary_output_dir / "global_quality_summary.csv"
    summary_df.to_csv(global_stats_path, index=False)

    # Print summary info
    print(f"Global summary statistics:")
    print(f"   Total models: {len(all_model_stats)}")
    print(f"   Total faces: {summary_df['Total_Faces'].sum():,}")
    print(f"   Total area: {summary_df['Total_Area_3D'].sum():.6f}")
    print(
        f"   Global low quality face ratio: {summary_df['Low_Quality_Faces'].sum() / summary_df['Total_Faces'].sum():.2%}")
    print(
        f"   Global low quality area ratio: {summary_df['Low_Quality_Area_3D'].sum() / summary_df['Total_Area_3D'].sum():.2%}")

    print(f"Global summary file saved: {global_stats_path}")


# =============================================================================
# Main execution function (FECI absolute value judgment version)
# =============================================================================
def run_analysis():
    """Main function, execute entire analysis workflow """
    print(f"Input directory: {GLOBAL_CONFIG['input_root_dir']}")
    print(f"Recursive search: {GLOBAL_CONFIG['recursive_search']}")
    print(f"GNECI method: {GLOBAL_CONFIG['gneci_method']}")
    print(f"FECI method: {GLOBAL_CONFIG['feci_method']}")
    print(f"FECI color mapping range: {GLOBAL_CONFIG['feci_color_min']:.1f} - {GLOBAL_CONFIG['feci_color_max']:.1f}")

    # Setup output directories
    output_dirs = setup_output_directories()
    print(f"Output directory: {output_dirs['root']}")

    overall_start_time = time.time()

    # Check if skip data processing stage
    if not GLOBAL_CONFIG["skip_data_processing"]:
        # Stage 1: Process models individually, generate CSV and PLY
        print(f"\n{'=' * 60}")
        print("Stage 1: Process individual model data, save CSV and PLY immediately")
        print(f"{'=' * 60}")

        # Search PLY files
        ply_files_info = find_ply_files(
            GLOBAL_CONFIG["input_root_dir"],
            GLOBAL_CONFIG["recursive_search"]
        )

        if not ply_files_info:
            print("No PLY files found, program exit")
            return

        # Filter files to process
        if GLOBAL_CONFIG["ply_files_to_process"]:
            specified_names = set(GLOBAL_CONFIG["ply_files_to_process"])
            ply_files_info = [f for f in ply_files_info if f['file_name'] in specified_names]
            print(f"After configuration filtering, will process {len(ply_files_info)} files")

        if not ply_files_info:
            print("No PLY files match criteria, program exit")
            return

        # Process models individually
        for i, file_info in enumerate(ply_files_info, 1):
            print(f"\nProcessing progress: {i}/{len(ply_files_info)} - {file_info['file_name']}")

            # Process model data
            result = process_single_model_data(
                file_info=file_info,
                textures_dir=GLOBAL_CONFIG["textures_directory"],
                model_id=i
            )

            # Save results immediately
            if result.success:
                print(f"Data processing success: {result.processed_faces}/{result.total_faces} faces")
                print(f"   Generated edge records: {len(result.edge_records)} entries")
                print(f"   Processing time: {result.processing_time:.2f} seconds")

                # Setup model output directory
                model_output_dirs = setup_model_output_directory(
                    output_dirs["models"],
                    result.model_name,
                    result.model_id
                )

                # Save detailed data CSV
                detailed_csv_path = model_output_dirs["csv"] / f"{Path(result.model_name).stem}_detailed_results.csv"
                df = pd.DataFrame(result.edge_records)
                df.to_csv(detailed_csv_path, index=False)
                print(f"   Detailed CSV saved: {detailed_csv_path.name}")

                # Generate colored PLY
                if result.edge_records:
                    face_feci_map = {}
                    for row in result.edge_records:
                        face_id = row.get("faceid")
                        feci_val = row.get("FECI")
                        if face_id is not None and feci_val is not None:
                            face_feci_map[face_id] = feci_val

                    colored_ply_path = model_output_dirs["ply"] / f"{Path(result.model_name).stem}_colored.ply"
                    generate_colored_ply_with_header_update(
                        result.model_path,
                        str(colored_ply_path),
                        face_feci_map
                    )

            else:
                print(f"Processing failed: {result.error_message}")

    else:
        print(f"\nSkip data processing stage, use existing CSV files directly")

    # Check if skip analysis stage
    if GLOBAL_CONFIG["skip_analysis"]:
        print(f"\nSkip analysis stage, only complete data processing")
        total_time = time.time() - overall_start_time
        print(f"\nData processing complete! Total time: {total_time:.2f} seconds")
        return

    # Stage 2: Collect FECI data from CSV files and calculate unified threshold
    print(f"\n{'=' * 60}")
    print("Stage 2: Collect FECI data from CSV files and calculate unified threshold")
    print("(Based on FECI absolute values)")
    print(f"{'=' * 60}")

    all_feci_values, model_info_list = collect_feci_from_csv_files(output_dirs["models"])

    if not all_feci_values:
        print("No FECI data collected, cannot continue analysis")
        return

    unified_threshold = calculate_unified_threshold(
        all_feci_values,
        GLOBAL_CONFIG["unified_threshold_method"]
    )

    # Stage 3: Generate quality analysis for each model based on unified threshold
    print(f"\n{'=' * 60}")
    print("Stage 3: Generate quality analysis for each model based on unified threshold")
    print("(Using FECI absolute value judgment, statistics face count and area ratio)")
    print(f"{'=' * 60}")

    all_model_stats = []

    for model_info in model_info_list:
        csv_file = model_info['csv_file']
        model_folder = model_info['model_folder']

        print(f"\nAnalyzing model: {model_folder}")

        # Re-get model output directories
        model_output_dirs = {
            "csv": Path(csv_file).parent,
            "ply": Path(csv_file).parent.parent / "colored_ply",
            "model_root": Path(csv_file).parent.parent
        }

        # Quality analysis based on CSV and unified threshold
        analysis_result = analyze_model_quality_from_csv(
            csv_file, unified_threshold, model_output_dirs
        )

        if analysis_result is not None:
            all_model_stats.append(analysis_result)

    # Stage 4: Generate global summary
    print(f"\n{'=' * 60}")
    print("Stage 4: Generate global summary")
    print(f"{'=' * 60}")

    generate_global_summary(all_model_stats, output_dirs["summary"], unified_threshold)

    # Summary
    total_time = time.time() - overall_start_time
    successful_models = len(all_model_stats)

    print(f"\nAnalysis complete!")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Successfully analyzed: {successful_models} models")
    print(f"   Output root directory: {output_dirs['root']}")

    print(f"\nFECI absolute value judgment + fixed color mapping features:")
    print(f"   - Quality judgment based on FECI absolute values, not area-weighted values")
    print(f"   - Statistics of low quality face count ratio and area ratio")
    print(f"   - Fixed FECI color mapping range, supports cross-model comparison")
    print(f"   - Flexible NECI/FECI calculation methods")
    print(f"   - Using 3D geometric area for statistical weight calculation")

    # Print method configuration summary
    print(f"\nMethod Configuration:")
    print(f"   GNECI method: {GLOBAL_CONFIG['gneci_method']}")
    print(f"   FECI method: {GLOBAL_CONFIG['feci_method']}")
    print(f"   Threshold method: {GLOBAL_CONFIG['unified_threshold_method']}")


if __name__ == '__main__':
    run_analysis()
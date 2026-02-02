/**************************************************************************/
/*  usd_document.cpp                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "usd_document.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/importer_mesh_instance_3d.h"

// tinyusdz headers -- only included in this translation unit.
#include "tinyusdz.hh"
#include "prim-types.hh"
#include "value-types.hh"

void USDDocument::_bind_methods() {
	ClassDB::bind_method(D_METHOD("append_from_file", "path", "state", "flags", "base_path"), &USDDocument::append_from_file, DEFVAL(0), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("generate_scene", "state", "bake_fps", "trimming", "remove_immutable_tracks"), &USDDocument::generate_scene, DEFVAL(30.0), DEFVAL(false), DEFVAL(true));
}

// ---------------------------------------------------------------------------
// Coordinate system conversion helpers.
// ---------------------------------------------------------------------------

Vector3 USDDocument::_convert_position(double p_x, double p_y, double p_z, float p_meters_per_unit, bool p_z_up) {
	// Godot uses Y-up, right-handed, meters.
	// USD default is Z-up, right-handed, centimeters (metersPerUnit = 0.01).
	if (p_z_up) {
		// Z-up to Y-up: swap Y and Z, negate the new Z.
		return Vector3(
				static_cast<float>(p_x * p_meters_per_unit),
				static_cast<float>(p_z * p_meters_per_unit),
				static_cast<float>(-p_y * p_meters_per_unit));
	} else {
		return Vector3(
				static_cast<float>(p_x * p_meters_per_unit),
				static_cast<float>(p_y * p_meters_per_unit),
				static_cast<float>(p_z * p_meters_per_unit));
	}
}

Transform3D USDDocument::_convert_transform(const tinyusdz::value::matrix4d &p_mat, float p_meters_per_unit, bool p_z_up) {
	// tinyusdz uses row-major 4x4 doubles: m[row][col].
	// Godot's Basis is column-major: columns are basis vectors.

	// First, extract the 3x3 rotation+scale and the translation column.
	// Note: USD stores translation in the last row for row-major convention
	// (m[3][0], m[3][1], m[3][2]).

	// Extract basis vectors (columns of the 3x3 upper-left in column-major
	// interpretation of the row-major matrix, which means rows of the
	// row-major matrix are basis vectors when read as column-major).
	// Since USD is row-major: row i = basis vector i (X, Y, Z).
	// Godot column-major: column i = basis vector i.

	// Extract rotation/scale only (no translation) from the upper-left 3x3.
	Basis basis;
	// USD row 0 -> Godot column 0 (X axis).
	basis.set_column(0, _convert_position(p_mat.m[0][0], p_mat.m[0][1], p_mat.m[0][2], 1.0f, p_z_up));
	// USD row 1 -> Godot column 1 (Y axis).
	basis.set_column(1, _convert_position(p_mat.m[1][0], p_mat.m[1][1], p_mat.m[1][2], 1.0f, p_z_up));
	// USD row 2 -> Godot column 2 (Z axis).
	basis.set_column(2, _convert_position(p_mat.m[2][0], p_mat.m[2][1], p_mat.m[2][2], 1.0f, p_z_up));

	// Translation: last row in row-major (m[3][0..2]).
	Vector3 origin = _convert_position(p_mat.m[3][0], p_mat.m[3][1], p_mat.m[3][2], p_meters_per_unit, p_z_up);

	return Transform3D(basis, origin);
}

// ---------------------------------------------------------------------------
// append_from_file: Load a USD file and populate USDState.
// ---------------------------------------------------------------------------

Error USDDocument::append_from_file(const String &p_path, Ref<USDState> p_state,
		uint32_t p_flags, const String &p_base_path) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);

	// Resolve the Godot resource path to a filesystem path.
	String fs_path = p_path;
	if (p_path.begins_with("res://") || p_path.begins_with("user://")) {
		fs_path = ProjectSettings::get_singleton()->globalize_path(p_path);
	}

	// Store base path information in the state.
	String base = p_base_path;
	if (base.is_empty()) {
		base = p_path.get_base_dir();
	}
	p_state->base_path = base;
	p_state->filename = p_path.get_file();
	p_state->scene_name = p_path.get_file().get_basename();

	// Load via tinyusdz.
	tinyusdz::Stage stage;
	std::string warn;
	std::string err;

	std::string std_path = fs_path.utf8().get_data();

	bool ok = tinyusdz::LoadUSDFromFile(std_path, &stage, &warn, &err);
	if (!ok) {
		String error_msg = String::utf8(err.c_str());
		ERR_PRINT("USD: Failed to load file '" + p_path + "': " + error_msg);
		return ERR_FILE_CANT_READ;
	}

	if (!warn.empty()) {
		String warn_msg = String::utf8(warn.c_str());
		WARN_PRINT("USD: Warnings loading '" + p_path + "': " + warn_msg);
	}

	// Read stage metadata.
	const tinyusdz::StageMetas &metas = stage.metas();

	double mpu = metas.metersPerUnit.get_value();
	p_state->meters_per_unit = static_cast<float>(mpu);

	tinyusdz::Axis up = metas.upAxis.get_value();
	p_state->up_axis_is_z = (up == tinyusdz::Axis::Z);

	// Parse the prim hierarchy into USDState.
	Error parse_err = _parse_scene(p_state, stage);
	if (parse_err != OK) {
		return parse_err;
	}

	return OK;
}

// ---------------------------------------------------------------------------
// _parse_scene: Traverse root prims.
// ---------------------------------------------------------------------------

Error USDDocument::_parse_scene(Ref<USDState> p_state, const tinyusdz::Stage &p_stage) {
	const std::vector<tinyusdz::Prim> &root_prims = p_stage.root_prims();

	for (size_t i = 0; i < root_prims.size(); i++) {
		_parse_nodes_recursive(p_state, root_prims[i], -1);
	}

	return OK;
}

// ---------------------------------------------------------------------------
// _parse_nodes_recursive: Create USDNode for each Prim and recurse children.
// ---------------------------------------------------------------------------

void USDDocument::_parse_nodes_recursive(Ref<USDState> p_state, const tinyusdz::Prim &p_prim, int p_parent_idx) {
	int node_idx = p_state->nodes.size();

	Ref<USDNode> node;
	node.instantiate();

	// Set the node name from the prim's element name.
	std::string prim_name = p_prim.element_name();
	node->set_original_name(String::utf8(prim_name.c_str()));

	// Set parent.
	node->set_parent(p_parent_idx);

	// If this is a root node, record it.
	if (p_parent_idx == -1) {
		p_state->root_nodes.push_back(node_idx);
	} else {
		// Add this node as a child of the parent.
		Ref<USDNode> parent_node = p_state->nodes[p_parent_idx];
		parent_node->append_child_index(node_idx);
	}

	// Compute local transform from the prim's xformOps.
	// Use GetLocalTransform which evaluates all xformOps at default time.
	bool reset_xform_stack = false;
	tinyusdz::value::matrix4d local_mat = tinyusdz::GetLocalTransform(p_prim, &reset_xform_stack);
	node->set_xform(_convert_transform(local_mat, p_state->meters_per_unit, p_state->up_axis_is_z));

	// Add node to the state array (before processing type-specific data,
	// since mesh/light/camera parsing may reference this node index).
	p_state->nodes.push_back(node);

	// Check prim type and parse type-specific data.
	std::string type_name = p_prim.type_name();

	if (p_prim.is<tinyusdz::GeomMesh>()) {
		const tinyusdz::GeomMesh *mesh = p_prim.as<tinyusdz::GeomMesh>();
		if (mesh) {
			_parse_mesh(p_state, *mesh, node_idx);
		}
	} else if (p_prim.is<tinyusdz::GeomCamera>()) {
		const tinyusdz::GeomCamera *cam = p_prim.as<tinyusdz::GeomCamera>();
		if (cam) {
			_parse_camera(p_state, *cam, node_idx);
		}
	} else if (tinyusdz::IsLightPrim(p_prim)) {
		_parse_light(p_state, p_prim, node_idx);
	} else if (p_prim.is<tinyusdz::Material>()) {
		const tinyusdz::Material *mat = p_prim.as<tinyusdz::Material>();
		if (mat) {
			_parse_material(p_state, *mat);
		}
	}

	// Recurse into children.
	const std::vector<tinyusdz::Prim> &children = p_prim.children();
	for (size_t i = 0; i < children.size(); i++) {
		_parse_nodes_recursive(p_state, children[i], node_idx);
	}
}

// ---------------------------------------------------------------------------
// _parse_mesh: Extract geometry from GeomMesh and populate USDMesh.
// ---------------------------------------------------------------------------

void USDDocument::_parse_mesh(Ref<USDState> p_state, const tinyusdz::GeomMesh &p_mesh, int p_node_idx) {
	Ref<USDMesh> usd_mesh;
	usd_mesh.instantiate();

	const std::string &mesh_name = p_mesh.name;
	usd_mesh->set_original_name(String::utf8(mesh_name.c_str()));

	// Get vertex positions.
	std::vector<tinyusdz::value::point3f> points = p_mesh.get_points();
	if (points.empty()) {
		// No geometry data; skip.
		return;
	}

	// Get face vertex counts and indices for triangulation.
	std::vector<int32_t> face_vertex_counts = p_mesh.get_faceVertexCounts();
	std::vector<int32_t> face_vertex_indices = p_mesh.get_faceVertexIndices();

	if (face_vertex_counts.empty() || face_vertex_indices.empty()) {
		return;
	}

	// Get normals (may be empty).
	std::vector<tinyusdz::value::normal3f> normals = p_mesh.get_normals();
	tinyusdz::Interpolation normals_interp = p_mesh.get_normalsInterpolation();

	// Try to get UVs from primvar "st" (the standard USD UV primvar name).
	std::vector<tinyusdz::value::texcoord2f> uvs;
	tinyusdz::Interpolation uv_interp = tinyusdz::Interpolation::Vertex;
	{
		tinyusdz::GeomPrimvar pv;
		if (p_mesh.get_primvar("st", &pv)) {
			std::vector<tinyusdz::value::texcoord2f> pv_uvs;
			if (pv.flatten_with_indices(&pv_uvs)) {
				uvs = std::move(pv_uvs);
				if (pv.has_interpolation()) {
					uv_interp = pv.get_interpolation();
				}
			}
		}
	}

	float mpu = p_state->meters_per_unit;
	bool z_up = p_state->up_axis_is_z;

	// Triangulate and build Godot surface arrays.
	// USD faces can be arbitrary polygons; we do simple fan triangulation.
	Vector<Vector3> godot_vertices;
	Vector<Vector3> godot_normals;
	Vector<Vector2> godot_uvs;
	Vector<int> godot_indices;

	// First pass: count triangles for reservation.
	int total_triangles = 0;
	for (size_t i = 0; i < face_vertex_counts.size(); i++) {
		int n = face_vertex_counts[i];
		if (n >= 3) {
			total_triangles += (n - 2);
		}
	}

	int total_verts = total_triangles * 3;
	godot_vertices.resize(total_verts);
	if (!normals.empty()) {
		godot_normals.resize(total_verts);
	}
	if (!uvs.empty()) {
		godot_uvs.resize(total_verts);
	}

	int vert_write = 0;
	int fv_offset = 0; // Running offset into face_vertex_indices.

	for (size_t face_i = 0; face_i < face_vertex_counts.size(); face_i++) {
		int num_verts = face_vertex_counts[face_i];

		if (num_verts < 3) {
			fv_offset += num_verts;
			continue;
		}

		// Fan triangulation: vertex 0 is the pivot.
		for (int tri = 0; tri < num_verts - 2; tri++) {
			// Triangle indices within the face.
			int idx0 = 0;
			int idx1 = tri + 1;
			int idx2 = tri + 2;

			int fvi0 = fv_offset + idx0;
			int fvi1 = fv_offset + idx1;
			int fvi2 = fv_offset + idx2;

			if (fvi0 >= (int)face_vertex_indices.size() || fvi1 >= (int)face_vertex_indices.size() || fvi2 >= (int)face_vertex_indices.size()) {
				vert_write += 3;
				continue;
			}

			int vi0 = face_vertex_indices[fvi0];
			int vi1 = face_vertex_indices[fvi1];
			int vi2 = face_vertex_indices[fvi2];

			if (vi0 < 0 || vi0 >= (int)points.size() || vi1 < 0 || vi1 >= (int)points.size() || vi2 < 0 || vi2 >= (int)points.size()) {
				vert_write += 3;
				continue;
			}

			// Positions.
			const auto &p0 = points[vi0];
			const auto &p1 = points[vi1];
			const auto &p2 = points[vi2];

			godot_vertices.set(vert_write + 0, _convert_position(p0[0], p0[1], p0[2], mpu, z_up));
			godot_vertices.set(vert_write + 1, _convert_position(p1[0], p1[1], p1[2], mpu, z_up));
			godot_vertices.set(vert_write + 2, _convert_position(p2[0], p2[1], p2[2], mpu, z_up));

			// Normals.
			if (!normals.empty()) {
				if (normals_interp == tinyusdz::Interpolation::FaceVarying) {
					// Per face-vertex normals: indexed by the running fv offset.
					if (fvi0 < (int)normals.size() && fvi1 < (int)normals.size() && fvi2 < (int)normals.size()) {
						const auto &n0 = normals[fvi0];
						const auto &n1 = normals[fvi1];
						const auto &n2 = normals[fvi2];
						godot_normals.set(vert_write + 0, _convert_position(n0[0], n0[1], n0[2], 1.0f, z_up));
						godot_normals.set(vert_write + 1, _convert_position(n1[0], n1[1], n1[2], 1.0f, z_up));
						godot_normals.set(vert_write + 2, _convert_position(n2[0], n2[1], n2[2], 1.0f, z_up));
					}
				} else {
					// Vertex interpolation: indexed by vertex index.
					if (vi0 < (int)normals.size() && vi1 < (int)normals.size() && vi2 < (int)normals.size()) {
						const auto &n0 = normals[vi0];
						const auto &n1 = normals[vi1];
						const auto &n2 = normals[vi2];
						godot_normals.set(vert_write + 0, _convert_position(n0[0], n0[1], n0[2], 1.0f, z_up));
						godot_normals.set(vert_write + 1, _convert_position(n1[0], n1[1], n1[2], 1.0f, z_up));
						godot_normals.set(vert_write + 2, _convert_position(n2[0], n2[1], n2[2], 1.0f, z_up));
					}
				}
			}

			// UVs.
			if (!uvs.empty()) {
				int uvi0, uvi1, uvi2;
				if (uv_interp == tinyusdz::Interpolation::FaceVarying) {
					uvi0 = fvi0;
					uvi1 = fvi1;
					uvi2 = fvi2;
				} else {
					uvi0 = vi0;
					uvi1 = vi1;
					uvi2 = vi2;
				}

				if (uvi0 < (int)uvs.size() && uvi1 < (int)uvs.size() && uvi2 < (int)uvs.size()) {
					// USD UVs: (0,0) = bottom-left; Godot: (0,0) = top-left.
					// Flip V coordinate.
					godot_uvs.set(vert_write + 0, Vector2(uvs[uvi0][0], 1.0f - uvs[uvi0][1]));
					godot_uvs.set(vert_write + 1, Vector2(uvs[uvi1][0], 1.0f - uvs[uvi1][1]));
					godot_uvs.set(vert_write + 2, Vector2(uvs[uvi2][0], 1.0f - uvs[uvi2][1]));
				}
			}

			vert_write += 3;
		}

		fv_offset += num_verts;
	}

	// Trim arrays to actual size written (should match, but be safe).
	if (vert_write < total_verts) {
		godot_vertices.resize(vert_write);
		if (!normals.empty()) {
			godot_normals.resize(vert_write);
		}
		if (!uvs.empty()) {
			godot_uvs.resize(vert_write);
		}
	}

	if (godot_vertices.is_empty()) {
		return;
	}

	// Build index array (trivial sequential, since we expanded everything).
	godot_indices.resize(godot_vertices.size());
	for (int i = 0; i < godot_vertices.size(); i++) {
		godot_indices.set(i, i);
	}

	// Build the Godot mesh arrays.
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = godot_vertices;
	if (!godot_normals.is_empty()) {
		arrays[Mesh::ARRAY_NORMAL] = godot_normals;
	}
	if (!godot_uvs.is_empty()) {
		arrays[Mesh::ARRAY_TEX_UV] = godot_uvs;
	}
	arrays[Mesh::ARRAY_INDEX] = godot_indices;

	// Add the surface to the mesh.
	usd_mesh->add_surface(arrays, Mesh::PRIMITIVE_TRIANGLES, -1, String::utf8(mesh_name.c_str()), Vector<Array>());

	// Store the mesh and link it to the node.
	int mesh_idx = p_state->meshes.size();
	p_state->meshes.push_back(usd_mesh);
	p_state->nodes[p_node_idx]->set_mesh(mesh_idx);
}

// ---------------------------------------------------------------------------
// _parse_material: Extract UsdPreviewSurface properties into USDMaterial.
// ---------------------------------------------------------------------------

void USDDocument::_parse_material(Ref<USDState> p_state, const tinyusdz::Material &p_mat) {
	Ref<USDMaterial> usd_mat;
	usd_mat.instantiate();

	usd_mat->set_name(String::utf8(p_mat.name.c_str()));

	// Material parsing requires traversing the shader network to find
	// UsdPreviewSurface children. For now, store the material with
	// default values; the shader network traversal will be expanded
	// in a subsequent iteration.

	p_state->materials.push_back(usd_mat);
}

// ---------------------------------------------------------------------------
// _parse_light: Extract light properties from typed light prims.
// ---------------------------------------------------------------------------

void USDDocument::_parse_light(Ref<USDState> p_state, const tinyusdz::Prim &p_prim, int p_node_idx) {
	Ref<USDLight> usd_light;
	usd_light.instantiate();

	// Extract common light properties using type-specific casts.
	if (p_prim.is<tinyusdz::DistantLight>()) {
		const tinyusdz::DistantLight *light = p_prim.as<tinyusdz::DistantLight>();
		if (light) {
			usd_light->set_type(USDLight::DISTANT);
			const auto &col = light->color.get_value();
			tinyusdz::value::color3f c;
			if (col.get(tinyusdz::value::TimeCode::Default(), &c)) {
				usd_light->set_color(Color(c[0], c[1], c[2]));
			}
			float intensity_val = 0.0f;
			const auto &inten = light->intensity.get_value();
			if (inten.get(tinyusdz::value::TimeCode::Default(), &intensity_val)) {
				usd_light->set_intensity(intensity_val);
			}
		}
	} else if (p_prim.is<tinyusdz::SphereLight>()) {
		const tinyusdz::SphereLight *light = p_prim.as<tinyusdz::SphereLight>();
		if (light) {
			usd_light->set_type(USDLight::SPHERE);
			const auto &col = light->color.get_value();
			tinyusdz::value::color3f c;
			if (col.get(tinyusdz::value::TimeCode::Default(), &c)) {
				usd_light->set_color(Color(c[0], c[1], c[2]));
			}
			float intensity_val = 0.0f;
			const auto &inten = light->intensity.get_value();
			if (inten.get(tinyusdz::value::TimeCode::Default(), &intensity_val)) {
				usd_light->set_intensity(intensity_val);
			}
			float radius_val = 0.0f;
			const auto &rad = light->radius.get_value();
			if (rad.get(tinyusdz::value::TimeCode::Default(), &radius_val)) {
				usd_light->set_radius(radius_val * p_state->meters_per_unit);
			}
		}
	} else if (p_prim.is<tinyusdz::DiskLight>()) {
		const tinyusdz::DiskLight *light = p_prim.as<tinyusdz::DiskLight>();
		if (light) {
			usd_light->set_type(USDLight::DISK);
			const auto &col = light->color.get_value();
			tinyusdz::value::color3f c;
			if (col.get(tinyusdz::value::TimeCode::Default(), &c)) {
				usd_light->set_color(Color(c[0], c[1], c[2]));
			}
			float intensity_val = 0.0f;
			const auto &inten = light->intensity.get_value();
			if (inten.get(tinyusdz::value::TimeCode::Default(), &intensity_val)) {
				usd_light->set_intensity(intensity_val);
			}
			float radius_val = 0.0f;
			const auto &rad = light->radius.get_value();
			if (rad.get(tinyusdz::value::TimeCode::Default(), &radius_val)) {
				usd_light->set_radius(radius_val * p_state->meters_per_unit);
			}
		}
	} else if (p_prim.is<tinyusdz::RectLight>()) {
		const tinyusdz::RectLight *light = p_prim.as<tinyusdz::RectLight>();
		if (light) {
			usd_light->set_type(USDLight::RECT);
			const auto &col = light->color.get_value();
			tinyusdz::value::color3f c;
			if (col.get(tinyusdz::value::TimeCode::Default(), &c)) {
				usd_light->set_color(Color(c[0], c[1], c[2]));
			}
			float intensity_val = 0.0f;
			const auto &inten = light->intensity.get_value();
			if (inten.get(tinyusdz::value::TimeCode::Default(), &intensity_val)) {
				usd_light->set_intensity(intensity_val);
			}
			float width_val = 0.0f;
			const auto &w = light->width.get_value();
			if (w.get(tinyusdz::value::TimeCode::Default(), &width_val)) {
				usd_light->set_width(width_val * p_state->meters_per_unit);
			}
			float height_val = 0.0f;
			const auto &h = light->height.get_value();
			if (h.get(tinyusdz::value::TimeCode::Default(), &height_val)) {
				usd_light->set_height(height_val * p_state->meters_per_unit);
			}
		}
	} else if (p_prim.is<tinyusdz::DomeLight>()) {
		const tinyusdz::DomeLight *light = p_prim.as<tinyusdz::DomeLight>();
		if (light) {
			usd_light->set_type(USDLight::DOME);
			const auto &col = light->color.get_value();
			tinyusdz::value::color3f c;
			if (col.get(tinyusdz::value::TimeCode::Default(), &c)) {
				usd_light->set_color(Color(c[0], c[1], c[2]));
			}
			float intensity_val = 0.0f;
			const auto &inten = light->intensity.get_value();
			if (inten.get(tinyusdz::value::TimeCode::Default(), &intensity_val)) {
				usd_light->set_intensity(intensity_val);
			}
		}
	} else {
		// Unknown light type; store as a generic sphere light.
		usd_light->set_type(USDLight::SPHERE);
	}

	int light_idx = p_state->lights.size();
	p_state->lights.push_back(usd_light);
	p_state->nodes[p_node_idx]->set_light(light_idx);
}

// ---------------------------------------------------------------------------
// _parse_camera: Extract camera properties from GeomCamera.
// ---------------------------------------------------------------------------

void USDDocument::_parse_camera(Ref<USDState> p_state, const tinyusdz::GeomCamera &p_cam, int p_node_idx) {
	Ref<USDCamera> usd_cam;
	usd_cam.instantiate();

	// Projection: TypedAttributeWithFallback<Animatable<Projection>>.
	{
		const auto &proj_anim = p_cam.projection.get_value();
		tinyusdz::GeomCamera::Projection proj_val = tinyusdz::GeomCamera::Projection::Perspective;
		proj_anim.get(tinyusdz::value::TimeCode::Default(), &proj_val);
		if (proj_val == tinyusdz::GeomCamera::Projection::Orthographic) {
			usd_cam->set_projection(USDCamera::ORTHOGRAPHIC);
		} else {
			usd_cam->set_projection(USDCamera::PERSPECTIVE);
		}
	}

	// Focal length (mm): TypedAttributeWithFallback<Animatable<float>>.
	{
		const auto &fl_anim = p_cam.focalLength.get_value();
		float fl = 50.0f;
		fl_anim.get(tinyusdz::value::TimeCode::Default(), &fl);
		usd_cam->set_focal_length(fl);
	}

	// Aperture (mm).
	{
		const auto &ha_anim = p_cam.horizontalAperture.get_value();
		float ha = 20.965f;
		ha_anim.get(tinyusdz::value::TimeCode::Default(), &ha);
		usd_cam->set_horizontal_aperture(ha);

		const auto &va_anim = p_cam.verticalAperture.get_value();
		float va = 15.2908f;
		va_anim.get(tinyusdz::value::TimeCode::Default(), &va);
		usd_cam->set_vertical_aperture(va);
	}

	// Clipping range: TypedAttributeWithFallback<Animatable<value::float2>>.
	{
		const auto &clip_anim = p_cam.clippingRange.get_value();
		tinyusdz::value::float2 clip = { { 0.1f, 1000000.0f } };
		clip_anim.get(tinyusdz::value::TimeCode::Default(), &clip);
		// USD stores clipping in scene units; convert to Godot meters.
		usd_cam->set_near_clip(clip[0] * p_state->meters_per_unit);
		usd_cam->set_far_clip(clip[1] * p_state->meters_per_unit);
	}

	int cam_idx = p_state->cameras.size();
	p_state->cameras.push_back(usd_cam);
	p_state->nodes[p_node_idx]->set_camera(cam_idx);
}

// ---------------------------------------------------------------------------
// generate_scene: Build the Godot Node tree from USDState.
// ---------------------------------------------------------------------------

Node *USDDocument::generate_scene(Ref<USDState> p_state, float p_bake_fps,
		bool p_trimming, bool p_remove_immutable_tracks) {
	ERR_FAIL_COND_V(p_state.is_null(), nullptr);
	ERR_FAIL_COND_V(p_state->nodes.is_empty(), nullptr);

	p_state->bake_fps = p_bake_fps;

	// Create root node.
	Node3D *root = memnew(Node3D);
	root->set_name(p_state->scene_name);

	// Generate nodes for each root prim.
	for (int i = 0; i < p_state->root_nodes.size(); i++) {
		int root_node_idx = p_state->root_nodes[i];
		_generate_scene_recursive(p_state, root_node_idx, root, root);
	}

	return root;
}

// ---------------------------------------------------------------------------
// _generate_scene_recursive: Depth-first scene tree building.
// ---------------------------------------------------------------------------

void USDDocument::_generate_scene_recursive(Ref<USDState> p_state, int p_node_idx, Node *p_parent, Node *p_root) {
	ERR_FAIL_INDEX(p_node_idx, p_state->nodes.size());

	Node3D *godot_node = _generate_node(p_state, p_node_idx, p_parent, p_root);
	if (!godot_node) {
		return;
	}

	// Recurse into children.
	Ref<USDNode> usd_node = p_state->nodes[p_node_idx];
	Vector<int> children = usd_node->get_children();
	for (int i = 0; i < children.size(); i++) {
		_generate_scene_recursive(p_state, children[i], godot_node, p_root);
	}
}

// ---------------------------------------------------------------------------
// _generate_node: Create the appropriate Godot node type.
// ---------------------------------------------------------------------------

Node3D *USDDocument::_generate_node(Ref<USDState> p_state, int p_node_idx, Node *p_parent, Node *p_root) {
	Ref<USDNode> usd_node = p_state->nodes[p_node_idx];
	Node3D *godot_node = nullptr;

	int mesh_idx = usd_node->get_mesh();
	int camera_idx = usd_node->get_camera();
	int light_idx = usd_node->get_light();

	if (mesh_idx >= 0 && mesh_idx < p_state->meshes.size()) {
		// Mesh node.
		Ref<USDMesh> usd_mesh = p_state->meshes[mesh_idx];
		Ref<ImporterMesh> importer_mesh = usd_mesh->to_importer_mesh();

		// If the mesh has material references, apply them.
		for (int s = 0; s < usd_mesh->get_surface_count(); s++) {
			USDMesh::Surface surface = usd_mesh->get_surface(s);
			if (surface.material >= 0 && surface.material < p_state->materials.size()) {
				Ref<USDMaterial> usd_mat = p_state->materials[surface.material];
				Ref<StandardMaterial3D> mat = usd_mat->to_material(p_state->base_path);
				importer_mesh->set_surface_material(s, mat);
			}
		}

		ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
		mesh_instance->set_mesh(importer_mesh);
		godot_node = mesh_instance;

		p_state->scene_mesh_instances[p_node_idx] = mesh_instance;

	} else if (camera_idx >= 0 && camera_idx < p_state->cameras.size()) {
		// Camera node.
		Ref<USDCamera> usd_cam = p_state->cameras[camera_idx];

		Camera3D *camera = memnew(Camera3D);
		if (usd_cam->get_projection() == USDCamera::ORTHOGRAPHIC) {
			camera->set_projection(Camera3D::PROJECTION_ORTHOGONAL);
		} else {
			camera->set_projection(Camera3D::PROJECTION_PERSPECTIVE);
			camera->set_fov(usd_cam->get_fov());
		}
		camera->set_near(usd_cam->get_near_clip());
		camera->set_far(usd_cam->get_far_clip());

		godot_node = camera;

	} else if (light_idx >= 0 && light_idx < p_state->lights.size()) {
		// Light node.
		Ref<USDLight> usd_light = p_state->lights[light_idx];
		Light3D *light = nullptr;

		switch (usd_light->get_type()) {
			case USDLight::DISTANT: {
				DirectionalLight3D *dir = memnew(DirectionalLight3D);
				light = dir;
			} break;
			case USDLight::SPHERE:
			case USDLight::DISK: {
				OmniLight3D *omni = memnew(OmniLight3D);
				omni->set_param(Light3D::PARAM_RANGE, usd_light->get_radius() > 0.0f ? usd_light->get_radius() * 10.0f : 10.0f);
				light = omni;
			} break;
			case USDLight::RECT:
			case USDLight::CYLINDER: {
				// Godot does not have a native area light; approximate with SpotLight.
				SpotLight3D *spot = memnew(SpotLight3D);
				spot->set_param(Light3D::PARAM_RANGE, MAX(usd_light->get_width(), usd_light->get_height()) * 10.0f);
				light = spot;
			} break;
			case USDLight::DOME: {
				// Dome lights map to environment/sky in Godot.
				// For now, create a DirectionalLight as a placeholder.
				DirectionalLight3D *dir = memnew(DirectionalLight3D);
				light = dir;
			} break;
		}

		if (light) {
			light->set_color(usd_light->get_color());
			// USD intensity model differs from Godot; apply a basic mapping.
			// intensity * 2^exposure is the effective power in USD.
			float effective_intensity = usd_light->get_intensity() * Math::pow(2.0f, usd_light->get_exposure());
			light->set_param(Light3D::PARAM_ENERGY, effective_intensity);
			light->set_shadow(usd_light->get_cast_shadows());

			godot_node = light;
		}
	}

	// If no specialized node was created, make a plain Node3D.
	if (!godot_node) {
		godot_node = memnew(Node3D);
	}

	// Set the name and transform.
	String node_name = usd_node->get_original_name();
	if (node_name.is_empty()) {
		node_name = "Node";
	}
	godot_node->set_name(node_name);
	godot_node->set_transform(usd_node->get_xform());
	godot_node->set_visible(usd_node->get_visible());

	// Add to scene tree.
	p_parent->add_child(godot_node, true);
	godot_node->set_owner(p_root);

	// Record in the state mapping.
	p_state->scene_nodes[p_node_idx] = godot_node;

	return godot_node;
}

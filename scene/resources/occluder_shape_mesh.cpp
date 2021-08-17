/*************************************************************************/
/*  occluder_shape_mesh.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "occluder_shape_mesh.h"

#include "scene/3d/occluder.h"
#include "scene/3d/spatial.h"
#include "scene/3d/visual_instance.h"
#include "scene/resources/surface_tool.h"
#include "servers/visual/portals/portal_defines.h"
#include "servers/visual_server.h"

void OccluderShapeMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bake_path", "path"), &OccluderShapeMesh::set_bake_path);
	ClassDB::bind_method(D_METHOD("get_bake_path"), &OccluderShapeMesh::get_bake_path);

	ClassDB::bind_method(D_METHOD("set_threshold_input_size", "size"), &OccluderShapeMesh::set_threshold_input_size);
	ClassDB::bind_method(D_METHOD("get_threshold_input_size"), &OccluderShapeMesh::get_threshold_input_size);

	ClassDB::bind_method(D_METHOD("set_threshold_output_size", "size"), &OccluderShapeMesh::set_threshold_output_size);
	ClassDB::bind_method(D_METHOD("get_threshold_output_size"), &OccluderShapeMesh::get_threshold_output_size);

	ClassDB::bind_method(D_METHOD("set_simplify", "simplify"), &OccluderShapeMesh::set_simplify);
	ClassDB::bind_method(D_METHOD("get_simplify"), &OccluderShapeMesh::get_simplify);

	ClassDB::bind_method(D_METHOD("set_plane_simplify_angle", "angle"), &OccluderShapeMesh::set_plane_simplify_angle);
	ClassDB::bind_method(D_METHOD("get_plane_simplify_angle"), &OccluderShapeMesh::get_plane_simplify_angle);

	ClassDB::bind_method(D_METHOD("set_remove_floor", "angle"), &OccluderShapeMesh::set_remove_floor);
	ClassDB::bind_method(D_METHOD("get_remove_floor"), &OccluderShapeMesh::get_remove_floor);

	ClassDB::bind_method(D_METHOD("set_bake_mask", "mask"), &OccluderShapeMesh::set_bake_mask);
	ClassDB::bind_method(D_METHOD("get_bake_mask"), &OccluderShapeMesh::get_bake_mask);
	ClassDB::bind_method(D_METHOD("set_bake_mask_value", "layer_number", "value"), &OccluderShapeMesh::set_bake_mask_value);
	ClassDB::bind_method(D_METHOD("get_bake_mask_value", "layer_number"), &OccluderShapeMesh::get_bake_mask_value);

	ClassDB::bind_method(D_METHOD("_set_data"), &OccluderShapeMesh::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &OccluderShapeMesh::_get_data);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "bake_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Spatial"), "set_bake_path", "get_bake_path");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bake_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_bake_mask", "get_bake_mask");

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "threshold_input_size", PROPERTY_HINT_RANGE, "0.0,10.0,0.1"), "set_threshold_input_size", "get_threshold_input_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "threshold_output_size", PROPERTY_HINT_RANGE, "0.0,10.0,0.1"), "set_threshold_output_size", "get_threshold_output_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "plane_angle", PROPERTY_HINT_RANGE, "0.0,45.0,0.1"), "set_plane_simplify_angle", "get_plane_simplify_angle");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "simplify", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_simplify", "get_simplify");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "remove_floor", PROPERTY_HINT_RANGE, "0, 90, 1"), "set_remove_floor", "get_remove_floor");

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");
}

void OccluderShapeMesh::_set_data(const Dictionary &p_d) {
	ERR_FAIL_COND(!p_d.has("version"));

	int version = p_d["version"];
	ERR_FAIL_COND(version != 100);

	ERR_FAIL_COND(!p_d.has("verts"));
	ERR_FAIL_COND(!p_d.has("planes"));
	ERR_FAIL_COND(!p_d.has("indices"));
	ERR_FAIL_COND(!p_d.has("first_indices"));
	ERR_FAIL_COND(!p_d.has("num_indices"));

	_mesh_data.vertices = p_d["verts"];

	Vector<Plane> planes = p_d["planes"];
	Vector<int> indices = p_d["indices"];
	Vector<int> face_first_indices = p_d["first_indices"];
	Vector<int> face_num_indices = p_d["num_indices"];

	ERR_FAIL_COND(planes.size() != face_first_indices.size());
	ERR_FAIL_COND(planes.size() != face_num_indices.size());

	_mesh_data.faces.resize(planes.size());
	for (int n = 0; n < planes.size(); n++) {
		Geometry::MeshData::Face f;
		f.plane = planes[n];
		f.indices.resize(face_num_indices[n]);

		for (int i = 0; i < face_num_indices[n]; i++) {
			int ind_id = i + face_first_indices[n];

			// check for invalid
			if (ind_id > indices.size()) {
				_mesh_data.vertices.clear();
				_mesh_data.faces.clear();
				ERR_FAIL_COND(ind_id > indices.size());
			}

			int ind = indices[ind_id];
			f.indices.set(i, ind);
		}

		_mesh_data.faces.set(n, f);
	}
}

Dictionary OccluderShapeMesh::_get_data() const {
	Dictionary d;
	d["version"] = 100;
	d["verts"] = _mesh_data.vertices;

	Vector<Plane> planes;
	Vector<int> indices;
	Vector<int> face_first_indices;
	Vector<int> face_num_indices;
	for (int n = 0; n < _mesh_data.faces.size(); n++) {
		const Geometry::MeshData::Face &f = _mesh_data.faces[n];
		planes.push_back(f.plane);
		face_first_indices.push_back(indices.size());

		for (int i = 0; i < f.indices.size(); i++) {
			indices.push_back(f.indices[i]);
		}
		face_num_indices.push_back(f.indices.size());
	}

	d["planes"] = planes;
	d["indices"] = indices;
	d["first_indices"] = face_first_indices;
	d["num_indices"] = face_num_indices;
	return d;
}

void OccluderShapeMesh::set_remove_floor(int p_angle) {
	_settings_remove_floor_angle = p_angle;
	_settings_remove_floor_dot = Math::cos(Math::deg2rad((real_t)p_angle));
}

void OccluderShapeMesh::update_transform_to_visual_server(const Transform &p_global_xform) {
	VisualServer::get_singleton()->occluder_set_transform(get_shape(), p_global_xform);
}

void OccluderShapeMesh::update_shape_to_visual_server() {
	VisualServer::get_singleton()->occluder_mesh_update(get_shape(), _mesh_data.faces, _mesh_data.vertices);
}

Transform OccluderShapeMesh::center_node(const Transform &p_global_xform, const Transform &p_parent_xform, real_t p_snap) {
	return Transform();
}

void OccluderShapeMesh::set_bake_mask(uint32_t p_mask) {
	_settings_bake_mask = p_mask;
	//update_configuration_warnings();
}

uint32_t OccluderShapeMesh::get_bake_mask() const {
	return _settings_bake_mask;
}

void OccluderShapeMesh::set_bake_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Render layer number must be between 1 and 20 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 20, "Render layer number must be between 1 and 20 inclusive.");
	uint32_t mask = get_bake_mask();
	if (p_value) {
		mask |= 1 << (p_layer_number - 1);
	} else {
		mask &= ~(1 << (p_layer_number - 1));
	}
	set_bake_mask(mask);
}

bool OccluderShapeMesh::get_bake_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Render layer number must be between 1 and 20 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 20, false, "Render layer number must be between 1 and 20 inclusive.");
	return _settings_bake_mask & (1 << (p_layer_number - 1));
}

void OccluderShapeMesh::notification_enter_world(RID p_scenario) {
	VisualServer::get_singleton()->occluder_set_scenario(get_shape(), p_scenario, VisualServer::OCCLUDER_TYPE_MESH);
}

void OccluderShapeMesh::clear() {
	_mesh_data.vertices.clear();
	_mesh_data.faces.clear();
	_bd.clear();
}

void OccluderShapeMesh::_log(String p_string) {
	//print_line(p_string);
}

void OccluderShapeMesh::bake(Node *owner) {
	clear();

	// the owner must be the occluder
	Occluder *occ = Object::cast_to<Occluder>(owner);
	ERR_FAIL_NULL_MSG(occ, "bake argument must be the Occluder");

	// We want the points in local space to the occluder ...
	// so that if the occluder moves, the polys can also move with it.
	// First we bake in world space though, as the epsilons etc are set up for world space.
	Transform xform_inv = occ->get_global_transform().affine_inverse();

	// one off conversion
	_settings_plane_simplify_dot = Math::cos(Math::deg2rad(_settings_plane_simplify_degrees));

	// try to get the nodepath
	if (owner->has_node(_settings_bake_path)) {
		Node *node = owner->get_node(_settings_bake_path);

		Spatial *branch = Object::cast_to<Spatial>(node);
		if (!branch) {
			return;
		}

		_bake_recursive(branch);
	}

	_simplify_triangles();

	_verify_verts();

	uint32_t process_tick = 1;
	while (_make_faces(process_tick++)) {
		;
	}

	print_line("num process ticks : " + itos(process_tick));

	_verify_verts();

	_finalize_faces();

	// clear intermediate data
	_bd.clear();

	// if not an identity matrix
	if (xform_inv != Transform()) {
		// transform all the verts etc from world space to local space
		for (int n = 0; n < _mesh_data.vertices.size(); n++) {
			const Vector3 &pt = _mesh_data.vertices[n];
			_mesh_data.vertices.set(n, xform_inv.xform(pt));
		}

		// planes
		for (int n = 0; n < _mesh_data.faces.size(); n++) {
			Geometry::MeshData::Face face = _mesh_data.faces[n];
			face.plane = xform_inv.xform(face.plane);
			_mesh_data.faces.set(n, face);
		}
	}

	update_shape_to_visual_server();
}

void OccluderShapeMesh::_simplify_triangles() {
	// noop
	if (!_mesh_data.vertices.size()) {
		return;
	}

	// clear hash table to reuse
	_bd.hash_verts.clear();
	_bd.hash_triangles._table.clear();

	// get the data into a format that mesh optimizer can deal with
	LocalVector<uint32_t> inds_in;
	for (int n = 0; n < _bd.faces.size(); n++) {
		const BakeFace &face = _bd.faces[n];
		CRASH_COND(face.indices.size() != 3);

		inds_in.push_back(face.indices[0]);
		inds_in.push_back(face.indices[1]);
		inds_in.push_back(face.indices[2]);
	}

	LocalVector<uint32_t> inds_out;
	inds_out.resize(inds_in.size());

	struct Vec3f {
		void set(real_t xx, real_t yy, real_t zz) {
			x = xx;
			y = yy;
			z = zz;
		}
		float x, y, z;
	};

	// prevent a screwup if real_t is double,
	// as the mesh optimizer lib expects floats
	LocalVector<Vec3f> verts;
	verts.resize(_mesh_data.vertices.size());
	for (int n = 0; n < _mesh_data.vertices.size(); n++) {
		const Vector3 &p = _mesh_data.vertices[n];
		verts[n].set(p.x, p.y, p.z);
	}

	//	typedef size_t (*SimplifyFunc)(unsigned int *destination, const unsigned int *indices, size_t index_count, const float *vertex_positions, size_t vertex_count, size_t vertex_positions_stride, size_t target_index_count, float target_error, float *r_error);
	//real_t target_error = 0.001;
	//	real_t target_error = 0.01;

	size_t result = 0;
	if (_settings_simplify > 0.0) {
#define GODOT_OCCLUDER_POLY_USE_ERROR_METRIC
#ifdef GODOT_OCCLUDER_POLY_USE_ERROR_METRIC
		// by an error metric
		real_t target_error = (_settings_simplify * _settings_simplify * _settings_simplify);
		target_error *= 0.05;
		result = SurfaceTool::simplify_func(&inds_out[0], &inds_in[0], inds_in.size(), (const float *)&verts[0], verts.size(), sizeof(Vec3f), 1, target_error, nullptr);
#else
		// by a percentage of the indices
		size_t target_indices = inds_in.size() - (inds_in.size() * _settings_simplify);
		target_indices = CLAMP(target_indices, 3, inds_in.size());
		result = SurfaceTool::simplify_func(&inds_out[0], &inds_in[0], inds_in.size(), (const float *)&verts[0], verts.size(), sizeof(Vec3f), target_indices, 0.9, nullptr);
#endif

	} else {
		// no simplification, just copy
		result = inds_in.size();
		inds_out = inds_in;
	}

	// resave
	_bd.faces.clear();
	_bd.faces.resize(result / 3);

	int count = 0;
	for (int n = 0; n < _bd.faces.size(); n++) {
		BakeFace face;
		face.indices.resize(3);

		Face3 f3;

		for (int c = 0; c < 3; c++) {
			const Vector3 &pos = _mesh_data.vertices[inds_out[count++]];
			int new_ind = _bd.find_or_create_vert(pos, n);
			face.indices[c] = new_ind;
			f3.vertex[c] = pos;
		}

		face.plane = f3.get_plane();

		face.area = f3.get_area();
		_bd.faces[n] = face;
	}

	// delete the orig verts
	_mesh_data.vertices.clear();

	// clear hash table no longer need
	_bd.hash_verts.clear();

	// sort all the vertex faces by area
	for (int n = 0; n < _bd.verts.size(); n++) {
		_sort_vertex_faces_by_area(n);
	}

	_create_sort_faces();
}

void OccluderShapeMesh::_create_sort_faces() {
	_bd.sort_faces.resize(_bd.faces.size());
	for (int n = 0; n < _bd.sort_faces.size(); n++) {
		const BakeFace &face = _bd.faces[n];
		SortFace &sface = _bd.sort_faces[n];
		sface.id = n;
		sface.area = face.area;
	}

	_bd.sort_faces.sort();
}

void OccluderShapeMesh::_sort_vertex_faces_by_area(uint32_t p_vertex_id) {
	BakeVertex &vert = _bd.verts[p_vertex_id];
	int num_faces = vert.linked_faces.size();

	// fill sort list
	LocalVector<SortFace> faces;
	faces.resize(num_faces);

	for (int n = 0; n < num_faces; n++) {
		uint32_t id = vert.linked_faces[n];
		faces[n].id = id;
		faces[n].area = _bd.faces[id].area;
	}

	faces.sort();

	for (int n = 0; n < num_faces; n++) {
		vert.linked_faces[n] = faces[n].id;
	}
}

bool OccluderShapeMesh::_make_faces(uint32_t p_process_tick) {
	bool found_one = false;

	// for now just recreate this each time
	_create_sort_faces();

	// maintain a list of failed combos, so as not to keep trying the
	// same ones
	LocalVectori<LocalVectori<uint32_t>> disallowed;

	// go through from the largest faces to the smallest
	for (int sort_face_id = 0; sort_face_id < _bd.sort_faces.size(); sort_face_id++) {
		int face_id = _bd.sort_faces[sort_face_id].id;
		//_log("source_face_id " + itos(face_id));

		const BakeFace &source_face = _bd.faces[face_id];

		for (int which_vert = 0; which_vert < source_face.indices.size(); which_vert++) {
			int v = source_face.indices[which_vert];

			//for (int v = 0; v < _bd.verts.size(); v++) {
			BakeVertex &vert = _bd.verts[v];

			// has the vert been done already?
			if (vert.last_processed_tick == p_process_tick) {
				continue;
			}

			// mark as done
			vert.last_processed_tick = p_process_tick;

			if (!vert.is_active()) {
				continue;
			}

			// dirty?
			if (!vert.dirty) {
				continue;
			} else {
				vert.dirty = false;
			}

			int num_faces = vert.linked_faces.size();

			if (num_faces <= 1) {
				continue;
			}

			_log("vert_id " + itos(v));

			// find maximum number of matching faces
			LocalVectori<uint32_t> matching_faces;
			LocalVectori<uint32_t> test_matching_faces;
			LocalVectori<uint32_t> best_matching_faces;
			real_t best_matching_faces_area = 0.0;
			real_t fit = 0.0;

			for (int a = 0; a < num_faces; a++) {
				matching_faces.clear();
				matching_faces.push_back(vert.linked_faces[a]);
				const BakeFace &fa = _bd.faces[vert.linked_faces[a]];

				real_t matching_faces_area = 0.0;

				for (int b = a + 1; b < num_faces; b++) {
					const BakeFace &fb = _bd.faces[vert.linked_faces[b]];

					if (_are_faces_coplanar_for_merging(fa, fb, fit)) {
						_log("\tfrom face_id " + itos(vert.linked_faces[a]) + " to face_id " + itos(vert.linked_faces[b]));

						// must be a neighbour of an existing face
						bool is_neighbour = false;
						for (int n = 0; n < matching_faces.size(); n++) {
							const BakeFace &neigh_face = _bd.faces[matching_faces[n]];
							if (_are_faces_neighbours(neigh_face, fb)) {
								is_neighbour = true;
								break;
							}
						}

						if (is_neighbour) {
							_log("\t\tis_neighbour");
							test_matching_faces.resize(1);
							test_matching_faces[0] = matching_faces[0];
							test_matching_faces.push_back(vert.linked_faces[b]);

							if (!_are_faces_disallowed(disallowed, test_matching_faces)) {
								//_log("\t\tarea : " + rtos(_bd.faces[matching_faces

								// better fit?
								real_t area = _find_matching_faces_total_area(test_matching_faces);
								if (area > matching_faces_area) {
									_log("\t\tarea > matching_faces_area");
									matching_faces_area = area;
									matching_faces = test_matching_faces;
								} else {
									_log("\t\tarea <= matching_faces_area");
								}
							} else {
								_log("\t\tDISALLOWED");
							}
						}
					}
				} // for b

				if (matching_faces.size() < 2)
					continue;

				// better than the best so far?
				real_t area = _find_matching_faces_total_area(matching_faces);

				//bool good_to_add = matching_faces.size() > best_matching_faces.size();
				bool good_to_add = area > best_matching_faces_area;

				if (good_to_add) {
					_log("\t\tadding to best_matching_faces");
					best_matching_faces = matching_faces;
					best_matching_faces_area = area;
				}

			} // for a

			// no good, can't match any faces from this vertex
			if (best_matching_faces.size() < 2) {
				disallowed.clear();
				continue;
			}

			_log("\t\t_merge_matching_faces " + itos(best_matching_faces[0]) + " to " + itos(best_matching_faces[1]));
			if (_merge_matching_faces(best_matching_faces)) {
				disallowed.clear();
				return true;
			} else {
				// don't allow this combo next time
				disallowed.push_back(best_matching_faces);

				// repeat for this vertex
				v--;
			}

		} // for vertex in sort face
	} // for sort face

	return found_one;
}

real_t OccluderShapeMesh::_find_matching_faces_total_area(const LocalVectori<uint32_t> &p_faces) const {
	real_t area = 0.0;
	for (int n = 0; n < p_faces.size(); n++) {
		area += _bd.faces[p_faces[n]].area;
	}
	return area;
}

bool OccluderShapeMesh::_are_faces_disallowed(const LocalVectori<LocalVectori<uint32_t>> &p_disallowed, const LocalVectori<uint32_t> &p_faces) const {
	for (int n = 0; n < p_disallowed.size(); n++) {
		const LocalVectori<uint32_t> &testlist = p_disallowed[n];

		if (testlist.size() != p_faces.size()) {
			continue;
		}

		bool match = true;
		for (int i = 0; i < testlist.size(); i++) {
			if (testlist[i] != p_faces[i]) {
				match = false;
				break;
			}
		}

		if (match) {
			return true;
		}
	}

	return false;
}

void OccluderShapeMesh::_verify_verts() {
	return;
#ifdef TOOLS_ENABLED
	for (int n = 0; n < _bd.verts.size(); n++) {
		const BakeVertex &bv = _bd.verts[n];

		for (int i = 0; i < bv.linked_faces.size(); i++) {
			uint32_t linked_face_id = bv.linked_faces[i];
			const BakeFace &face = _bd.faces[linked_face_id];

			int found = face.indices.find(n);
			CRASH_COND(found == -1);
		}
	}
#endif
}

bool OccluderShapeMesh::_are_faces_neighbours(const BakeFace &p_a, const BakeFace &p_b) const {
	for (int n = 0; n < p_a.indices.size(); n++) {
		int a0 = p_a.indices[n];
		int a1 = p_a.indices[(n + 1) % p_a.indices.size()];

		if (a1 < a0) {
			SWAP(a0, a1);
		}

		for (int m = 0; m < p_b.indices.size(); m++) {
			int b0 = p_b.indices[m];
			int b1 = p_b.indices[(m + 1) % p_b.indices.size()];

			if (b1 < b0) {
				SWAP(b0, b1);
			}

			if ((a0 == b0) && (a1 == b1)) {
				return true;
			}
		}
	}

	return false;
}

bool OccluderShapeMesh::_merge_matching_faces(const LocalVectori<uint32_t> &p_faces) {
	// count indices
	int count = 0;
	for (int n = 0; n < p_faces.size(); n++) {
		int face_id = p_faces[n];
		const BakeFace &fa = _bd.faces[face_id];
		count += fa.indices.size();
	}

	// add ALL the verts
	BakeFace new_face;
	new_face.indices.resize(count);
	count = 0;

	new_face.plane.normal = Vector3(0, 0, 0);
	new_face.plane.d = 0;
	real_t old_area = 0.0;

	for (int n = 0; n < p_faces.size(); n++) {
		int face_id = p_faces[n];
		const BakeFace &fa = _bd.faces[face_id];

		real_t area = fa.area;

		// average this
		new_face.plane.normal += fa.plane.normal * area;
		new_face.plane.d += fa.plane.d * area;

		old_area += area;

		for (int i = 0; i < fa.indices.size(); i++) {
			new_face.indices[count++] = fa.indices[i];
		}
	}

	// average the new plane
	if (old_area > 0.001) {
		new_face.plane.normal /= old_area;
		new_face.plane.d /= old_area;
	} else {
		// else face too small .. can't merge these,
		// we'd end up getting float error...
		return false;
	}

	// now remove the central, and any duplicates
	_tri_face_remove_central_and_duplicates(new_face, -1);

	BakeFace new_face_before_simplify = new_face;

	// return false if not convex
	real_t new_total_area = 0.0;

	if (!_create_merged_convex_face(new_face, old_area, new_total_area)) {
		return false; // not convex etc
	}

	// Impose a maximum number of verts for a face
	// (this is applied in the visual server anyway, so we want to avoid
	// merging such faces).
	// The reason for this maximum is that runtime speed depends on the number of edges.
	if (new_face.indices.size() > PortalDefines::OCCLUSION_POLY_MAX_VERTS) {
		return false;
	}

	// resave
	int new_face_id = _bd.faces.size();
	new_face.area = new_total_area;
	_bd.faces.push_back(new_face);

	// we need to remove the old face links from all the old vertices BEFORE
	// simplification (as some verts may have been removed)
	for (int n = 0; n < new_face_before_simplify.indices.size(); n++) {
		int idx = new_face_before_simplify.indices[n];
		BakeVertex &bv = _bd.verts[idx];

		for (int i = 0; i < p_faces.size(); i++) {
			uint32_t old_linked_face = p_faces[i];
			bv.remove_linked_face(old_linked_face);
		}
	}

	// store the new linked face in the vertices AFTER simplification
	for (int n = 0; n < new_face.indices.size(); n++) {
		int idx = new_face.indices[n];
		BakeVertex &bv = _bd.verts[idx];

		// add the new face
		bv.add_linked_face(new_face_id);
	}

	// remove source faces
	BakeFace face_blank;
	for (int n = 0; n < p_faces.size(); n++) {
		_bd.faces[p_faces[n]] = face_blank;
	}

	return true;
}

bool OccluderShapeMesh::_face_replace_index(Geometry::MeshData::Face &r_face, uint32_t p_face_id, int p_del_index, int p_replace_index, bool p_fix_vertex_linked_faces) {
	int64_t found = r_face.indices.find(p_del_index);
	if (found != -1) {
		r_face.indices.set(found, p_replace_index);

		if (p_fix_vertex_linked_faces) {
			_bd.verts[p_del_index].remove_linked_face(p_face_id);
			_bd.verts[p_replace_index].add_linked_face(p_face_id);
		}

		return true;
	}
	return false;
}

void OccluderShapeMesh::_tri_face_remove_central_and_duplicates(BakeFace &p_face, uint32_t p_central_idx) const {
	for (int n = 0; n < p_face.indices.size(); n++) {
		uint32_t idx = p_face.indices[n];
		if (idx == p_central_idx) {
			// remove this index, and repeat the element
			p_face.indices.remove(n);
			n--;
		} else {
			// check for duplicates
			for (int i = n + 1; i < p_face.indices.size(); i++) {
				uint32_t idx2 = p_face.indices[i];

				// if we find a duplicate, remove the first occurrence
				if (idx == idx2) {
					// remove this index, and repeat the element
					p_face.indices.remove(n);
					n--;
					break;
				}
			}
		}
	}
}

void OccluderShapeMesh::_finalize_faces() {
	// save into the geometry::mesh
	// first delete blank faces in the mesh
	for (int n = 0; n < _bd.faces.size(); n++) {
		const BakeFace &face = _bd.faces[n];

		// delete the face if no indices, or the size below the threshold
		if (!face.indices.size() || (face.area < _settings_threshold_output_size)) {
			// face can be deleted
			_bd.faces.remove(n);

			// repeat this element
			n--;
		}
	}

	LocalVector<uint32_t> vertex_remaps;
	vertex_remaps.resize(_bd.verts.size());
	for (uint32_t n = 0; n < vertex_remaps.size(); n++) {
		vertex_remaps[n] = UINT32_MAX;
	}

	for (int n = 0; n < _bd.faces.size(); n++) {
		BakeFace face = _bd.faces[n];

		// remap the indices
		for (int c = 0; c < face.indices.size(); c++) {
			int index_before = face.indices[c];
			uint32_t &index_after = vertex_remaps[index_before];

			// vertex not done yet
			if (index_after == UINT32_MAX) {
				_mesh_data.vertices.push_back(_bd.verts[index_before].pos);
				index_after = _mesh_data.vertices.size() - 1;
			}

			// save the after index in the face
			face.indices[c] = index_after;

			// resave the face
			_bd.faces[n] = face;
		}
	}

	// copy the bake faces to the mesh faces
	_mesh_data.faces.resize(_bd.faces.size());
	for (int n = 0; n < _mesh_data.faces.size(); n++) {
		const BakeFace &src = _bd.faces[n];
		Geometry::MeshData::Face dest;
		dest.plane = src.plane;

		dest.indices.resize(src.indices.size());
		for (int i = 0; i < src.indices.size(); i++) {
			dest.indices.set(i, src.indices[i]);
		}

		_mesh_data.faces.set(n, dest);
	}

	// display some stats
	print_line("OccluderShapeMesh faces : " + itos(_mesh_data.faces.size()) + ", verts : " + itos(_mesh_data.vertices.size()));
}

String OccluderShapeMesh::_vec3_to_string(const Vector3 &p_pt) const {
	String str;
	str = "(";
	str += itos(p_pt.x);
	str += ", ";
	str += itos(p_pt.y);
	str += ", ";
	str += itos(p_pt.z);
	str += ") ";
	return str;
}

bool OccluderShapeMesh::_try_bake_face(const Face3 &p_face) {
	real_t area = p_face.get_twice_area_squared() * 0.5;
	if (area < _settings_threshold_input_size_squared) {
		return false;
	}

	//const real_t threshold = 1.0 * 1.0;

	//	Vector3 e[3];
	//	for (int n = 0; n < 3; n++) {
	//		const Vector3 &v0 = p_face.vertex[n];
	//		const Vector3 &v1 = p_face.vertex[(n + 1) % 3];

	//		e[n] = v1 - v0;
	//		if (e[n].length_squared() < _threshold_size_squared) {
	//			return false;
	//		}
	//	}

	// face is big enough to use as occluder
	BakeFace face;
	face.plane = Plane(p_face.vertex[0], p_face.vertex[1], p_face.vertex[2]);

	// reject floors
	if ((_settings_remove_floor_angle) && (Math::abs(face.plane.normal.y) > (_settings_remove_floor_dot))) {
		return false;
	}

	_log("try_bake_face " + _vec3_to_string(p_face.vertex[0]) + _vec3_to_string(p_face.vertex[1]) + _vec3_to_string(p_face.vertex[2]));

	uint32_t inds[3];
	face.indices.resize(3);
	for (int c = 0; c < 3; c++) {
		face.indices[c] = _find_or_create_vert(p_face.vertex[c]);
		inds[c] = face.indices[c];
	}

	// already baked this face? a duplicate so ignore
	uint32_t stored_face_id = _bd.hash_triangles.find(inds);
	if (stored_face_id != UINT32_MAX)
		return false;

	// special case, check for duplicate faces... not very efficient
	//	for (int n = 0; n < _mesh_data.faces.size(); n++) {
	//		const Geometry::MeshData::Face &face2 = _mesh_data.faces[n];

	//		for (int i = 0; i < 3; i++) {
	//			if (face.indices[i] == face2.indices[0]) {
	//				bool matched = true;
	//				for (int test = 1; test < 3; test++) {
	//					if (face.indices[(i + test) % 3] != face2.indices[test]) {
	//						matched = false;
	//						break;
	//					}
	//				}
	//				if (matched) {
	//					return false;
	//				}
	//			}
	//		}
	//	}

	_bd.hash_triangles.add(inds, _bd.faces.size());
	face.area = p_face.get_area();
	_bd.faces.push_back(face);

	_log("\tbaked initial face");
	return true;
}

real_t OccluderShapeMesh::_find_face_area(const Geometry::MeshData::Face &p_face) const {
	int num_inds = p_face.indices.size();

	Vector<Vector3> pts;
	for (int n = 0; n < num_inds; n++) {
		pts.push_back(_bd.verts[p_face.indices[n]].pos);
	}

	return Geometry::find_polygon_area(pts.ptr(), pts.size());
}

// return false if not convex
bool OccluderShapeMesh::_create_merged_convex_face(BakeFace &r_face, real_t p_old_face_area, real_t &r_new_total_area) {
	int num_inds = r_face.indices.size();

	Vector<Vector3> pts;
	pts.resize(num_inds);
	for (int n = 0; n < num_inds; n++) {
		pts.set(n, _bd.verts[r_face.indices[n]].pos);
	}
	Vector<Vector3> pts_changed = pts;

	// make the new face convex, remove any redundant verts, find the area and compare to the old
	// poly area plus the new face, to detect invalid joins
	if (!Geometry::make_polygon_convex(pts_changed, -r_face.plane.normal)) {
		return false;
	}
	CRASH_COND(pts_changed.size() > num_inds);

	// reverse order, as we want counter clockwise polys
	//pts_changed.invert();

	r_new_total_area = Geometry::find_polygon_area(pts_changed.ptr(), pts_changed.size());

	// conditions to disallow invalid joins
	real_t diff = p_old_face_area - r_new_total_area;

	// this epsilon here is crucial for tuning to prevent
	// artifact 'overlap'. Can this be automatically determined from
	// scale? NYI
	if (Math::abs(diff) > 0.01) { // 0.01
		return false;
	}

	LocalVectori<uint32_t> new_inds;

	for (int n = 0; n < pts_changed.size(); n++) {
		for (int i = 0; i < num_inds; i++) {
			if (pts_changed[n] == pts[i]) {
				new_inds.push_back(r_face.indices[i]);
				break;
			}
		}
	}

	// now should have the new indices
	r_face.indices = new_inds;

	return true;
}

bool OccluderShapeMesh::_bake_material_check(Ref<Material> p_material) {
	SpatialMaterial *spat = Object::cast_to<SpatialMaterial>(p_material.ptr());
	if (spat && spat->get_feature(SpatialMaterial::FEATURE_TRANSPARENT)) {
		return false;
	}

	return true;
}

void OccluderShapeMesh::_bake_recursive(Spatial *p_node) {
	VisualInstance *vi = Object::cast_to<VisualInstance>(p_node);

	// check the flags, visible, static etc
	if (vi && vi->is_visible_in_tree() && (vi->get_portal_mode() == CullInstance::PORTAL_MODE_STATIC)) {
		bool valid = true;

		// material check for transparency
		GeometryInstance *gi = Object::cast_to<GeometryInstance>(p_node);
		if (gi && valid && !_bake_material_check(gi->get_material_override())) {
			valid = false;
		}

		if ((vi->get_layer_mask() & _settings_bake_mask) == 0) {
			valid = false;
		}

		if (valid) {
			Transform tr = vi->get_global_transform();

			PoolVector<Face3> faces = vi->get_faces(VisualInstance::FACES_SOLID);

			if (faces.size()) {
				_log("baking : " + p_node->get_name());

				for (int n = 0; n < faces.size(); n++) {
					Face3 face = faces[n];

					// transform to world space
					for (int c = 0; c < 3; c++) {
						face.vertex[c] = tr.xform(face.vertex[c]);
					}

					_try_bake_face(face);
				}
			} // if there were faces

		} // if valid
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		Spatial *child = Object::cast_to<Spatial>(p_node->get_child(n));
		if (child) {
			_bake_recursive(child);
		}
	}
}

OccluderShapeMesh::OccluderShapeMesh() :
		OccluderShape(VisualServer::get_singleton()->occluder_create()) {
}

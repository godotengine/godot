/*************************************************************************/
/*  occluder_shape_mesh.h                                                */
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

#ifndef OCCLUDER_SHAPE_MESH_H
#define OCCLUDER_SHAPE_MESH_H

#include "core/hash_map.h"
#include "core/local_vector.h"
#include "core/math/geometry.h"
#include "occluder_shape.h"

class Spatial;
class Material;

class OccluderShapeMesh : public OccluderShape {
	GDCLASS(OccluderShapeMesh, OccluderShape);
	OBJ_SAVE_TYPE(OccluderShapeMesh);

	Geometry::MeshData _mesh_data;

	template <uint32_t NUM_BINS, class ELEMENT>
	struct HashTable {
		void add(const ELEMENT &p_ele, uint32_t p_hash) {
			_bins[p_hash % NUM_BINS].push_back(p_ele);
		}

		const LocalVectori<ELEMENT> &get_bin(uint32_t p_hash) const {
			return _bins[p_hash % NUM_BINS];
		}

		void clear() {
			for (uint32_t n = 0; n < NUM_BINS; n++) {
				_bins[n].clear();
			}
		}

	private:
		LocalVectori<ELEMENT> _bins[NUM_BINS];
	};

	struct HashTable_Tri {
		struct Element {
			uint32_t inds[3];
			uint32_t id;
		};
		HashTable<1024, Element> _table;

		uint32_t hash(uint32_t p_inds[3]) const {
			return p_inds[0] + p_inds[1] + p_inds[2];
		}

		void add(uint32_t p_inds[3], uint32_t p_id) {
			Element e;
			e.inds[0] = p_inds[0];
			e.inds[1] = p_inds[1];
			e.inds[2] = p_inds[2];
			e.id = p_id;

			uint32_t h = hash(p_inds);
			_table.add(e, h);
		}

		uint32_t find(uint32_t p_inds[3]) {
			uint32_t h = hash(p_inds);
			const LocalVectori<Element> &bin = _table.get_bin(h);
			uint32_t bsize = bin.size();
			for (uint32_t n = 0; n < bsize; n++) {
				const Element &e = bin[n];

				// match?

				for (int i = 0; i < 3; i++) {
					if (p_inds[i] == e.inds[0]) {
						bool matched = true;
						for (int test = 1; test < 3; test++) {
							if (p_inds[(i + test) % 3] != e.inds[test]) {
								matched = false;
								break;
							}
						}
						if (matched) {
							return e.id;
						}
					} // if found first match
				} // for i
			}
			// not found
			return UINT32_MAX;
		}
	};

	struct HashTable_Pos {
		struct Element {
			Vector3 pos;
			uint32_t id;
		};
		HashTable<4096, Element> _table;

		struct Vec3i {
			bool operator==(const Vec3i &p_o) const { return x == p_o.x && y == p_o.y && z == p_o.z; }
			bool operator!=(const Vec3i &p_o) const { return (*this == p_o) == false; }
			int32_t x, y, z;
		};

		Vec3i hash_to_i(Vector3 p_pos) const {
			Vec3i res;

			_process_pos(p_pos);
			res.x = hash(p_pos.x);
			res.y = hash(p_pos.y);
			res.z = hash(p_pos.z);
			return res;
		}

		void _process_pos(Vector3 &p_pos) const {
			p_pos.x *= 1309.2;
			p_pos.y *= 6053.5;
			p_pos.z *= 185.9;
		}

		bool hash_range(Vector3 p_pos, Vec3i &r_range_lo, Vec3i &r_range_hi) const {
			const real_t epsilon = 0.01;

			_process_pos(p_pos);
			r_range_lo.x = hash(p_pos.x - epsilon);
			r_range_lo.y = hash(p_pos.y - epsilon);
			r_range_lo.z = hash(p_pos.z - epsilon);
			r_range_hi.x = hash(p_pos.x + epsilon);
			r_range_hi.y = hash(p_pos.y + epsilon);
			r_range_hi.z = hash(p_pos.z + epsilon);
			return r_range_lo != r_range_hi;
		}

		uint32_t hash(real_t p_val) const {
			return p_val; // + 0.34;
		}

		uint32_t hash_3i(const Vec3i &p_vec) const {
			return p_vec.x + p_vec.y + p_vec.z;
		};

		uint32_t hash_pos(const Vector3 &p_pos) const {
			return hash_3i(hash_to_i(p_pos));
		}
		void add(const Vector3 &p_pos, uint32_t p_id) {
			uint32_t hash = hash_pos(p_pos);
			Element e;
			e.pos = p_pos;
			e.id = p_id;
			_table.add(e, hash);
		}

		uint32_t find_single_bin(uint32_t p_hash, const Vector3 &p_pos) const {
			const LocalVectori<Element> &bin = _table.get_bin(p_hash);
			uint32_t bsize = bin.size();

			for (uint32_t n = 0; n < bsize; n++) {
				const Element &e = bin[n];
				if (e.pos.is_equal_approx(p_pos)) {
					return e.id;
				}
			}
			return UINT32_MAX;
		}

		uint32_t find(const Vector3 &p_pos) const {
			Vec3i lo, hi;
			// if there is no range
			if (!hash_range(p_pos, lo, hi)) {
				return find_single_bin(hash_3i(lo), p_pos);
			}

			// test immediate neighbours that might be caused by
			// crossing a boundary... (often there is just one extra, there is no
			// need to test all boundaries of the 3x3)
			Vec3i test;
			for (test.z = lo.z; test.z <= hi.z; test.z++) {
				for (test.y = lo.y; test.y <= hi.y; test.y++) {
					for (test.x = lo.x; test.x <= hi.x; test.x++) {
						uint32_t id = find_single_bin(hash_3i(test), p_pos);
						if (id != UINT32_MAX) {
							return id;
						}
					}
				}
			}

			// not found
			return UINT32_MAX;
		}

		void clear() {
			_table.clear();
		}
	};

	struct BakeVertex {
		bool is_active() const { return linked_faces.size(); }
		bool remove_linked_face(uint32_t p_id) {
			int64_t found = linked_faces.find(p_id);
			if (found != -1) {
				linked_faces.remove_unordered(found);
				dirty = true;
				return true;
			}
			return false;
		}
		void add_linked_face(uint32_t p_id) {
			if (linked_faces.find(p_id) == -1) {
				linked_faces.push_back(p_id);
				dirty = true;
			}
		}
		BakeVertex() {
			last_processed_tick = 0;
			dirty = true;
		};
		Vector3 pos;
		// try and merge faces from a vertex once with each run
		uint32_t last_processed_tick;
		bool dirty;
		LocalVectori<uint32_t> linked_faces;
	};

	struct BakeFace {
		BakeFace() { area = 0.0; }
		Plane plane;
		real_t area;
		LocalVectori<uint32_t> indices;
	};

	struct SortFace {
		// used for sort .. we want the largest faces first in the list
		bool operator<(const SortFace &p_o) const { return area > p_o.area; }
		uint32_t id;
		real_t area;
	};

	// used in baking
	struct BakeData {
		void clear() {
			//face_areas.clear();
			verts.clear();
			faces.clear();
			sort_faces.clear();
			hash_verts.clear();
			hash_triangles._table.clear();
		}
		uint32_t find_or_create_vert(const Vector3 &p_pos, uint32_t p_face_id) {
			uint32_t id = hash_verts.find(p_pos);
			if (id != UINT32_MAX) {
				verts[id].linked_faces.push_back(p_face_id);
				return id;
			}

#if 0
			// use this to test the hash table is finding all cases
			for (uint32_t n = 0; n < verts.size(); n++) {
				if (verts[n].pos.is_equal_approx(p_pos)) {
					verts[n].linked_faces.push_back(p_face_id);
					return n;
				}
			}
#endif

			id = verts.size();
			verts.resize(id + 1);
			verts[id].pos = p_pos;
			verts[id].linked_faces.push_back(p_face_id);

			hash_verts.add(p_pos, id);

			return id;
		}

		LocalVectori<BakeVertex> verts;
		LocalVectori<BakeFace> faces;

		HashTable_Pos hash_verts;
		HashTable_Tri hash_triangles;

		// sorted by the largest face,
		// so we can process the largest faces first
		LocalVectori<SortFace> sort_faces;
	} _bd;

	NodePath _settings_bake_path;
	real_t _settings_threshold_input_size = 1.0;
	real_t _settings_threshold_input_size_squared = 1.0;
	real_t _settings_threshold_output_size = 1.0;
	real_t _settings_simplify = 0.0;
	real_t _settings_plane_simplify_degrees = 11.0;
	real_t _settings_plane_simplify_dot = 0.98;
	real_t _settings_remove_floor_dot = 0.0;
	int _settings_remove_floor_angle = 20;
	uint32_t _settings_bake_mask = 0xFFFFFFFF;

	bool _bake_material_check(Ref<Material> p_material);
	void _bake_recursive(Spatial *p_node);
	bool _try_bake_face(const Face3 &p_face);
	void _simplify_triangles();
	bool _make_faces(uint32_t p_process_tick);
	bool _merge_matching_faces(const LocalVectori<uint32_t> &p_faces);

	void _finalize_faces();
	bool _face_replace_index(Geometry::MeshData::Face &r_face, uint32_t p_face_id, int p_del_index, int p_replace_index, bool p_fix_vertex_linked_faces);
	real_t _find_face_area(const Geometry::MeshData::Face &p_face) const;
	bool _are_faces_neighbours(const BakeFace &p_a, const BakeFace &p_b) const;
	bool _are_faces_disallowed(const LocalVectori<LocalVectori<uint32_t>> &p_disallowed, const LocalVectori<uint32_t> &p_faces) const;
	real_t _find_matching_faces_total_area(const LocalVectori<uint32_t> &p_faces) const;
	void _sort_vertex_faces_by_area(uint32_t p_vertex_id);
	void _create_sort_faces();

	void _verify_verts();

	uint32_t _find_or_create_vert(const Vector3 &p_pos) {
		uint32_t id = _bd.hash_verts.find(p_pos);
		if (id != UINT32_MAX) {
			return id;
		}

#if 0
		// use this to detect any that are present but not found in the hash table
		// .. this should not happen.
		for (uint32_t n = 0; n < _mesh_data.vertices.size(); n++) {
			if (_mesh_data.vertices[n].is_equal_approx(p_pos)) {
				WARN_PRINT("FOUND WITH FULL LOOKUP");
				return n;
			}
		}
#endif

		id = _mesh_data.vertices.size();
		_mesh_data.vertices.push_back(p_pos);

		_bd.hash_verts.add(p_pos, id);

		return id;
	}

	bool _create_merged_convex_face(BakeFace &r_face, real_t p_old_face_area, real_t &r_new_total_area);
	String _vec3_to_string(const Vector3 &p_pt) const;
	void _tri_face_remove_central_and_duplicates(BakeFace &p_face, uint32_t p_central_idx) const;
	bool _are_faces_coplanar_for_merging(const BakeFace &p_a, const BakeFace &p_b, real_t &r_fit) const {
		r_fit = p_a.plane.normal.dot(p_b.plane.normal);
		return r_fit >= _settings_plane_simplify_dot;
	}
	int _get_wrapped_face_index(const Geometry::MeshData::Face &p_face, int p_index) const {
		return (p_index + p_face.indices.size()) % p_face.indices.size();
	}

	void set_bake_path(const NodePath &p_path) { _settings_bake_path = p_path; }
	NodePath get_bake_path() { return _settings_bake_path; }

	void set_threshold_input_size(real_t p_threshold) {
		_settings_threshold_input_size = p_threshold;
		// can't be zero, we want to prevent zero area triangles which could
		// cause divide by zero in the occlusion culler goodness of fit
		_settings_threshold_input_size_squared = MAX(p_threshold * p_threshold, 0.01);
	}
	real_t get_threshold_input_size() const { return _settings_threshold_input_size; }

	void set_threshold_output_size(real_t p_threshold) {
		_settings_threshold_output_size = p_threshold;
	}
	real_t get_threshold_output_size() const { return _settings_threshold_output_size; }

	void set_simplify(real_t p_simplify) { _settings_simplify = p_simplify; }
	real_t get_simplify() const { return _settings_simplify; }

	void set_plane_simplify_angle(real_t p_angle) { _settings_plane_simplify_degrees = p_angle; }
	real_t get_plane_simplify_angle() const { return _settings_plane_simplify_degrees; }

	void set_remove_floor(int p_angle);
	int get_remove_floor() const { return _settings_remove_floor_angle; }

	void set_bake_mask(uint32_t p_mask);
	uint32_t get_bake_mask() const;

	void set_bake_mask_value(int p_layer_number, bool p_value);
	bool get_bake_mask_value(int p_layer_number) const;

	// serializing
	void _set_data(const Dictionary &p_d);
	Dictionary _get_data() const;

	void _log(String p_string);

protected:
	static void _bind_methods();

public:
	const Geometry::MeshData &get_mesh_data() const { return _mesh_data; }

	virtual void notification_enter_world(RID p_scenario);
	virtual void update_shape_to_visual_server();
	virtual void update_transform_to_visual_server(const Transform &p_global_xform);
	virtual Transform center_node(const Transform &p_global_xform, const Transform &p_parent_xform, real_t p_snap);

	void clear();
	void bake(Node *owner);

	OccluderShapeMesh();
};

#endif // OCCLUDER_SHAPE_MESH_H

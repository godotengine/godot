/**************************************************************************/
/*  grid_map.hpp                                                          */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class MeshLibrary;
class PhysicsMaterial;
class Resource;

class GridMap : public Node3D {
	GDEXTENSION_CLASS(GridMap, Node3D)

public:
	static const int INVALID_CELL_ITEM = -1;

	void set_collision_layer(uint32_t p_layer);
	uint32_t get_collision_layer() const;
	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;
	void set_collision_mask_value(int32_t p_layer_number, bool p_value);
	bool get_collision_mask_value(int32_t p_layer_number) const;
	void set_collision_layer_value(int32_t p_layer_number, bool p_value);
	bool get_collision_layer_value(int32_t p_layer_number) const;
	void set_collision_priority(float p_priority);
	float get_collision_priority() const;
	void set_physics_material(const Ref<PhysicsMaterial> &p_material);
	Ref<PhysicsMaterial> get_physics_material() const;
	void set_bake_navigation(bool p_bake_navigation);
	bool is_baking_navigation();
	void set_navigation_map(const RID &p_navigation_map);
	RID get_navigation_map() const;
	void set_mesh_library(const Ref<MeshLibrary> &p_mesh_library);
	Ref<MeshLibrary> get_mesh_library() const;
	void set_cell_size(const Vector3 &p_size);
	Vector3 get_cell_size() const;
	void set_cell_scale(float p_scale);
	float get_cell_scale() const;
	void set_octant_size(int32_t p_size);
	int32_t get_octant_size() const;
	void set_cell_item(const Vector3i &p_position, int32_t p_item, int32_t p_orientation = 0);
	int32_t get_cell_item(const Vector3i &p_position) const;
	int32_t get_cell_item_orientation(const Vector3i &p_position) const;
	Basis get_cell_item_basis(const Vector3i &p_position) const;
	Basis get_basis_with_orthogonal_index(int32_t p_index) const;
	int32_t get_orthogonal_index_from_basis(const Basis &p_basis) const;
	Vector3i local_to_map(const Vector3 &p_local_position) const;
	Vector3 map_to_local(const Vector3i &p_map_position) const;
	void resource_changed(const Ref<Resource> &p_resource);
	void set_center_x(bool p_enable);
	bool get_center_x() const;
	void set_center_y(bool p_enable);
	bool get_center_y() const;
	void set_center_z(bool p_enable);
	bool get_center_z() const;
	void clear();
	TypedArray<Vector3i> get_used_cells() const;
	TypedArray<Vector3i> get_used_cells_by_item(int32_t p_item) const;
	Array get_meshes() const;
	Array get_bake_meshes();
	RID get_bake_mesh_instance(int32_t p_idx);
	void clear_baked_meshes();
	void make_baked_meshes(bool p_gen_lightmap_uv = false, float p_lightmap_uv_texel_size = 0.1);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


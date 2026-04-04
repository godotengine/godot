/**************************************************************************/
/*  mesh_instance3d.hpp                                                   */
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

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/geometry_instance3d.hpp>
#include <godot_cpp/classes/mesh_convex_decomposition_settings.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/node_path.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Material;
class Mesh;
class Skin;
class SkinReference;
class StringName;

class MeshInstance3D : public GeometryInstance3D {
	GDEXTENSION_CLASS(MeshInstance3D, GeometryInstance3D)

public:
	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh() const;
	void set_skeleton_path(const NodePath &p_skeleton_path);
	NodePath get_skeleton_path();
	void set_skin(const Ref<Skin> &p_skin);
	Ref<Skin> get_skin() const;
	Ref<SkinReference> get_skin_reference() const;
	int32_t get_surface_override_material_count() const;
	void set_surface_override_material(int32_t p_surface, const Ref<Material> &p_material);
	Ref<Material> get_surface_override_material(int32_t p_surface) const;
	Ref<Material> get_active_material(int32_t p_surface) const;
	void create_trimesh_collision();
	void create_convex_collision(bool p_clean = true, bool p_simplify = false);
	void create_multiple_convex_collisions(const Ref<MeshConvexDecompositionSettings> &p_settings = nullptr);
	int32_t get_blend_shape_count() const;
	int32_t find_blend_shape_by_name(const StringName &p_name);
	float get_blend_shape_value(int32_t p_blend_shape_idx) const;
	void set_blend_shape_value(int32_t p_blend_shape_idx, float p_value);
	void create_debug_tangents();
	Ref<ArrayMesh> bake_mesh_from_current_blend_shape_mix(const Ref<ArrayMesh> &p_existing = nullptr);
	Ref<ArrayMesh> bake_mesh_from_current_skeleton_pose(const Ref<ArrayMesh> &p_existing = nullptr);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		GeometryInstance3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


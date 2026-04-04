/**************************************************************************/
/*  importer_mesh_instance3d.hpp                                          */
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

#include <godot_cpp/classes/geometry_instance3d.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/node_path.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ImporterMesh;
class Skin;

class ImporterMeshInstance3D : public Node3D {
	GDEXTENSION_CLASS(ImporterMeshInstance3D, Node3D)

public:
	void set_mesh(const Ref<ImporterMesh> &p_mesh);
	Ref<ImporterMesh> get_mesh() const;
	void set_skin(const Ref<Skin> &p_skin);
	Ref<Skin> get_skin() const;
	void set_skeleton_path(const NodePath &p_skeleton_path);
	NodePath get_skeleton_path() const;
	void set_layer_mask(uint32_t p_layer_mask);
	uint32_t get_layer_mask() const;
	void set_cast_shadows_setting(GeometryInstance3D::ShadowCastingSetting p_shadow_casting_setting);
	GeometryInstance3D::ShadowCastingSetting get_cast_shadows_setting() const;
	void set_visibility_range_end_margin(float p_distance);
	float get_visibility_range_end_margin() const;
	void set_visibility_range_end(float p_distance);
	float get_visibility_range_end() const;
	void set_visibility_range_begin_margin(float p_distance);
	float get_visibility_range_begin_margin() const;
	void set_visibility_range_begin(float p_distance);
	float get_visibility_range_begin() const;
	void set_visibility_range_fade_mode(GeometryInstance3D::VisibilityRangeFadeMode p_mode);
	GeometryInstance3D::VisibilityRangeFadeMode get_visibility_range_fade_mode() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


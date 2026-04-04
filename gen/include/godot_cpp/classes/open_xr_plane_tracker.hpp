/**************************************************************************/
/*  open_xr_plane_tracker.hpp                                             */
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

#include <godot_cpp/classes/open_xr_spatial_component_plane_alignment_list.hpp>
#include <godot_cpp/classes/open_xr_spatial_entity_tracker.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Mesh;
class PackedVector2Array;
class Shape3D;

class OpenXRPlaneTracker : public OpenXRSpatialEntityTracker {
	GDEXTENSION_CLASS(OpenXRPlaneTracker, OpenXRSpatialEntityTracker)

public:
	void set_bounds_size(const Vector2 &p_bounds_size);
	Vector2 get_bounds_size() const;
	void set_plane_alignment(OpenXRSpatialComponentPlaneAlignmentList::PlaneAlignment p_plane_alignment);
	OpenXRSpatialComponentPlaneAlignmentList::PlaneAlignment get_plane_alignment() const;
	void set_plane_label(const String &p_plane_label);
	String get_plane_label() const;
	void set_mesh_data(const Transform3D &p_origin, const PackedVector2Array &p_vertices, const PackedInt32Array &p_indices = PackedInt32Array());
	void clear_mesh_data();
	Transform3D get_mesh_offset() const;
	Ref<Mesh> get_mesh();
	Ref<Shape3D> get_shape(float p_thickness = 0.01);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		OpenXRSpatialEntityTracker::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


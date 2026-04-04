/**************************************************************************/
/*  navigation_mesh_source_geometry_data2d.hpp                            */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class NavigationMeshSourceGeometryData2D : public Resource {
	GDEXTENSION_CLASS(NavigationMeshSourceGeometryData2D, Resource)

public:
	void clear();
	bool has_data();
	void set_traversable_outlines(const TypedArray<PackedVector2Array> &p_traversable_outlines);
	TypedArray<PackedVector2Array> get_traversable_outlines() const;
	void set_obstruction_outlines(const TypedArray<PackedVector2Array> &p_obstruction_outlines);
	TypedArray<PackedVector2Array> get_obstruction_outlines() const;
	void append_traversable_outlines(const TypedArray<PackedVector2Array> &p_traversable_outlines);
	void append_obstruction_outlines(const TypedArray<PackedVector2Array> &p_obstruction_outlines);
	void add_traversable_outline(const PackedVector2Array &p_shape_outline);
	void add_obstruction_outline(const PackedVector2Array &p_shape_outline);
	void merge(const Ref<NavigationMeshSourceGeometryData2D> &p_other_geometry);
	void add_projected_obstruction(const PackedVector2Array &p_vertices, bool p_carve);
	void clear_projected_obstructions();
	void set_projected_obstructions(const Array &p_projected_obstructions);
	Array get_projected_obstructions() const;
	Rect2 get_bounds();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


/**************************************************************************/
/*  tube_trail_mesh.hpp                                                   */
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

#include <godot_cpp/classes/primitive_mesh.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Curve;

class TubeTrailMesh : public PrimitiveMesh {
	GDEXTENSION_CLASS(TubeTrailMesh, PrimitiveMesh)

public:
	void set_radius(float p_radius);
	float get_radius() const;
	void set_radial_steps(int32_t p_radial_steps);
	int32_t get_radial_steps() const;
	void set_sections(int32_t p_sections);
	int32_t get_sections() const;
	void set_section_length(float p_section_length);
	float get_section_length() const;
	void set_section_rings(int32_t p_section_rings);
	int32_t get_section_rings() const;
	void set_cap_top(bool p_cap_top);
	bool is_cap_top() const;
	void set_cap_bottom(bool p_cap_bottom);
	bool is_cap_bottom() const;
	void set_curve(const Ref<Curve> &p_curve);
	Ref<Curve> get_curve() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PrimitiveMesh::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


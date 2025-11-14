/**************************************************************************/
/*  tapered_capsule_mesh.h                                                */
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

#pragma once

#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/mesh.h"

class TaperedCapsuleMesh : public PrimitiveMesh {
	GDCLASS(TaperedCapsuleMesh, PrimitiveMesh);

private:
	real_t radius_top = 0.5;
	real_t radius_bottom = 0.5;
	real_t mid_height = 1.0;
	int radial_segments = 64;
	int rings = 8;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const override;

	virtual void _update_lightmap_size() override;

public:
	static void create_mesh_array(Array &p_arr, real_t p_radius_top, real_t p_radius_bottom, real_t p_mid_height, int p_radial_segments = 64, int p_rings = 8, bool p_add_uv2 = false, const real_t p_uv2_padding = 1.0);

	void set_radius_top(const real_t p_radius_top);
	real_t get_radius_top() const;

	void set_radius_bottom(const real_t p_radius_bottom);
	real_t get_radius_bottom() const;

	void set_mid_height(const real_t p_mid_height);
	real_t get_mid_height() const;

	void set_height(const real_t p_height);
	real_t get_height() const;

	void set_radial_segments(const int p_segments);
	int get_radial_segments() const;

	void set_rings(const int p_rings);
	int get_rings() const;
};

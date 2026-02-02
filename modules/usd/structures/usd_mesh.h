/**************************************************************************/
/*  usd_mesh.h                                                            */
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

#include "core/io/resource.h"
#include "scene/resources/3d/importer_mesh.h"

class USDMesh : public Resource {
	GDCLASS(USDMesh, Resource);

public:
	struct Surface {
		Array arrays;
		Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;
		int material = -1;
		String name;
		Vector<Array> blend_shape_arrays;
	};

private:
	Vector<Surface> surfaces;
	Vector<String> blend_shapes;
	String original_name;

protected:
	static void _bind_methods();

public:
	String get_original_name() const;
	void set_original_name(const String &p_name);

	int get_surface_count() const;
	Vector<String> get_blend_shapes() const;
	void set_blend_shapes(const Vector<String> &p_blend_shapes);

	void add_surface(const Array &p_arrays, Mesh::PrimitiveType p_primitive, int p_material, const String &p_name, const Vector<Array> &p_blend_shape_arrays);
	Surface get_surface(int p_index) const;

	Ref<ImporterMesh> to_importer_mesh() const;
};

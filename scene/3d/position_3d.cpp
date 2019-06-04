/*************************************************************************/
/*  position_3d.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "position_3d.h"
#include "scene/resources/mesh.h"

RES Position3D::_get_gizmo_geometry() const {

	Ref<Mesh> mesh = memnew(Mesh);

	DVector<Vector3> cursor_points;
	DVector<Color> cursor_colors;
	float cs = 0.25;
	cursor_points.push_back(Vector3(+cs, 0, 0));
	cursor_points.push_back(Vector3(-cs, 0, 0));
	cursor_points.push_back(Vector3(0, +cs, 0));
	cursor_points.push_back(Vector3(0, -cs, 0));
	cursor_points.push_back(Vector3(0, 0, +cs));
	cursor_points.push_back(Vector3(0, 0, -cs));
	cursor_colors.push_back(Color(1, 0.5, 0.5, 1));
	cursor_colors.push_back(Color(1, 0.5, 0.5, 1));
	cursor_colors.push_back(Color(0.5, 1, 0.5, 1));
	cursor_colors.push_back(Color(0.5, 1, 0.5, 1));
	cursor_colors.push_back(Color(0.5, 0.5, 1, 1));
	cursor_colors.push_back(Color(0.5, 0.5, 1, 1));

	Ref<FixedMaterial> mat = memnew(FixedMaterial);
	mat->set_flag(Material::FLAG_UNSHADED, true);
	mat->set_line_width(3);
	Array d;
	d[Mesh::ARRAY_VERTEX] = cursor_points;
	d[Mesh::ARRAY_COLOR] = cursor_colors;
	mesh->add_surface(Mesh::PRIMITIVE_LINES, d);
	mesh->surface_set_material(0, mat);
	return mesh;
}

Position3D::Position3D() {
}

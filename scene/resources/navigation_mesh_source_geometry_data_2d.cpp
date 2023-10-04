/**************************************************************************/
/*  navigation_mesh_source_geometry_data_2d.cpp                           */
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

#include "navigation_mesh_source_geometry_data_2d.h"

#include "scene/resources/mesh.h"

void NavigationMeshSourceGeometryData2D::clear() {
	traversable_outlines.clear();
	obstruction_outlines.clear();
}

void NavigationMeshSourceGeometryData2D::_set_traversable_outlines(const Vector<Vector<Vector2>> &p_traversable_outlines) {
	traversable_outlines = p_traversable_outlines;
}

void NavigationMeshSourceGeometryData2D::_set_obstruction_outlines(const Vector<Vector<Vector2>> &p_obstruction_outlines) {
	obstruction_outlines = p_obstruction_outlines;
}

void NavigationMeshSourceGeometryData2D::_add_traversable_outline(const Vector<Vector2> &p_shape_outline) {
	if (p_shape_outline.size() > 1) {
		traversable_outlines.push_back(p_shape_outline);
	}
}

void NavigationMeshSourceGeometryData2D::_add_obstruction_outline(const Vector<Vector2> &p_shape_outline) {
	if (p_shape_outline.size() > 1) {
		obstruction_outlines.push_back(p_shape_outline);
	}
}

void NavigationMeshSourceGeometryData2D::set_traversable_outlines(const TypedArray<Vector<Vector2>> &p_traversable_outlines) {
	traversable_outlines.resize(p_traversable_outlines.size());
	for (int i = 0; i < p_traversable_outlines.size(); i++) {
		traversable_outlines.write[i] = p_traversable_outlines[i];
	}
}

TypedArray<Vector<Vector2>> NavigationMeshSourceGeometryData2D::get_traversable_outlines() const {
	TypedArray<Vector<Vector2>> typed_array_traversable_outlines;
	typed_array_traversable_outlines.resize(traversable_outlines.size());
	for (int i = 0; i < typed_array_traversable_outlines.size(); i++) {
		typed_array_traversable_outlines[i] = traversable_outlines[i];
	}

	return typed_array_traversable_outlines;
}

void NavigationMeshSourceGeometryData2D::set_obstruction_outlines(const TypedArray<Vector<Vector2>> &p_obstruction_outlines) {
	obstruction_outlines.resize(p_obstruction_outlines.size());
	for (int i = 0; i < p_obstruction_outlines.size(); i++) {
		obstruction_outlines.write[i] = p_obstruction_outlines[i];
	}
}

TypedArray<Vector<Vector2>> NavigationMeshSourceGeometryData2D::get_obstruction_outlines() const {
	TypedArray<Vector<Vector2>> typed_array_obstruction_outlines;
	typed_array_obstruction_outlines.resize(obstruction_outlines.size());
	for (int i = 0; i < typed_array_obstruction_outlines.size(); i++) {
		typed_array_obstruction_outlines[i] = obstruction_outlines[i];
	}

	return typed_array_obstruction_outlines;
}

void NavigationMeshSourceGeometryData2D::add_traversable_outline(const PackedVector2Array &p_shape_outline) {
	if (p_shape_outline.size() > 1) {
		Vector<Vector2> traversable_outline;
		traversable_outline.resize(p_shape_outline.size());
		for (int i = 0; i < p_shape_outline.size(); i++) {
			traversable_outline.write[i] = p_shape_outline[i];
		}
		traversable_outlines.push_back(traversable_outline);
	}
}

void NavigationMeshSourceGeometryData2D::add_obstruction_outline(const PackedVector2Array &p_shape_outline) {
	if (p_shape_outline.size() > 1) {
		Vector<Vector2> obstruction_outline;
		obstruction_outline.resize(p_shape_outline.size());
		for (int i = 0; i < p_shape_outline.size(); i++) {
			obstruction_outline.write[i] = p_shape_outline[i];
		}
		obstruction_outlines.push_back(obstruction_outline);
	}
}

void NavigationMeshSourceGeometryData2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("clear"), &NavigationMeshSourceGeometryData2D::clear);
	ClassDB::bind_method(D_METHOD("has_data"), &NavigationMeshSourceGeometryData2D::has_data);

	ClassDB::bind_method(D_METHOD("set_traversable_outlines", "traversable_outlines"), &NavigationMeshSourceGeometryData2D::set_traversable_outlines);
	ClassDB::bind_method(D_METHOD("get_traversable_outlines"), &NavigationMeshSourceGeometryData2D::get_traversable_outlines);

	ClassDB::bind_method(D_METHOD("set_obstruction_outlines", "obstruction_outlines"), &NavigationMeshSourceGeometryData2D::set_obstruction_outlines);
	ClassDB::bind_method(D_METHOD("get_obstruction_outlines"), &NavigationMeshSourceGeometryData2D::get_obstruction_outlines);

	ClassDB::bind_method(D_METHOD("add_traversable_outline", "shape_outline"), &NavigationMeshSourceGeometryData2D::add_traversable_outline);
	ClassDB::bind_method(D_METHOD("add_obstruction_outline", "shape_outline"), &NavigationMeshSourceGeometryData2D::add_obstruction_outline);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "traversable_outlines", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "set_traversable_outlines", "get_traversable_outlines");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "obstruction_outlines", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "set_obstruction_outlines", "get_obstruction_outlines");
}

NavigationMeshSourceGeometryData2D::NavigationMeshSourceGeometryData2D() {
}

NavigationMeshSourceGeometryData2D::~NavigationMeshSourceGeometryData2D() {
	clear();
}

/**************************************************************************/
/*  polygon2d_navigation_geometry_parser_2d.cpp                           */
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

#include "polygon2d_navigation_geometry_parser_2d.h"

#include "scene/2d/polygon_2d.h"

bool Polygon2DNavigationGeometryParser2D::parses_node(Node *p_node) {
	return (Object::cast_to<Polygon2D>(p_node) != nullptr);
}

void Polygon2DNavigationGeometryParser2D::parse_geometry(Node *p_node, Ref<NavigationPolygon> p_navigation_polygon, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry) {
	NavigationPolygon::ParsedGeometryType parsed_geometry_type = p_navigation_polygon->get_parsed_geometry_type();

	if (Object::cast_to<Polygon2D>(p_node) && parsed_geometry_type != NavigationPolygon::PARSED_GEOMETRY_STATIC_COLLIDERS) {
		Polygon2D *polygon_2d = Object::cast_to<Polygon2D>(p_node);

		const Transform2D transform = polygon_2d->get_global_transform();

		Vector<Vector2> shape_outline = polygon_2d->get_polygon();
		for (int i = 0; i < shape_outline.size(); i++) {
			shape_outline.write[i] = transform.xform(shape_outline[i]);
		}

		p_source_geometry->add_obstruction_outline(shape_outline);
	}
}

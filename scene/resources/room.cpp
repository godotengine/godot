/*************************************************************************/
/*  room.cpp                                                             */
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
#include "room.h"

#include "servers/visual_server.h"

RID RoomBounds::get_rid() const {

	return area;
}

void RoomBounds::set_bounds(const BSP_Tree &p_bounds) {

	VisualServer::get_singleton()->room_set_bounds(area, p_bounds);
	emit_signal("changed");
}

BSP_Tree RoomBounds::get_bounds() const {

	return VisualServer::get_singleton()->room_get_bounds(area);
}

void RoomBounds::set_geometry_hint(const DVector<Face3> &p_geometry_hint) {

	geometry_hint = p_geometry_hint;
}

DVector<Face3> RoomBounds::get_geometry_hint() const {

	return geometry_hint;
}

void RoomBounds::_regenerate_bsp_cubic() {

	if (geometry_hint.size()) {

		float err = 0;
		geometry_hint = Geometry::wrap_geometry(geometry_hint, &err); ///< create a "wrap" that encloses the given geometry

		BSP_Tree new_bounds(geometry_hint, err);
		set_bounds(new_bounds);
	}
}

void RoomBounds::_regenerate_bsp() {

	if (geometry_hint.size()) {

		BSP_Tree new_bounds(geometry_hint, 0);
		set_bounds(new_bounds);
	}
}

void RoomBounds::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_bounds", "bsp_tree"), &RoomBounds::set_bounds);
	ObjectTypeDB::bind_method(_MD("get_bounds"), &RoomBounds::get_bounds);

	ObjectTypeDB::bind_method(_MD("set_geometry_hint", "triangles"), &RoomBounds::set_geometry_hint);
	ObjectTypeDB::bind_method(_MD("get_geometry_hint"), &RoomBounds::get_geometry_hint);
	ObjectTypeDB::bind_method(_MD("regenerate_bsp"), &RoomBounds::_regenerate_bsp);
	ObjectTypeDB::set_method_flags(get_type_static(), _SCS("regenerate_bsp"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);
	ObjectTypeDB::bind_method(_MD("regenerate_bsp_cubic"), &RoomBounds::_regenerate_bsp_cubic);
	ObjectTypeDB::set_method_flags(get_type_static(), _SCS("regenerate_bsp_cubic"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "bounds"), _SCS("set_bounds"), _SCS("get_bounds"));
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3_ARRAY, "geometry_hint"), _SCS("set_geometry_hint"), _SCS("get_geometry_hint"));
}

RoomBounds::RoomBounds() {

	area = VisualServer::get_singleton()->room_create();
}

RoomBounds::~RoomBounds() {

	VisualServer::get_singleton()->free(area);
}

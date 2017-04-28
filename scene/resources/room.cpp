/*************************************************************************/
/*  room.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

void RoomBounds::set_geometry_hint(const PoolVector<Face3> &p_geometry_hint) {

	geometry_hint = p_geometry_hint;
}

PoolVector<Face3> RoomBounds::get_geometry_hint() const {

	return geometry_hint;
}

void RoomBounds::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_geometry_hint", "triangles"), &RoomBounds::set_geometry_hint);
	ClassDB::bind_method(D_METHOD("get_geometry_hint"), &RoomBounds::get_geometry_hint);

	//ADD_PROPERTY( PropertyInfo( Variant::DICTIONARY, "bounds"), "set_bounds","get_bounds") ;
	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR3_ARRAY, "geometry_hint"), "set_geometry_hint", "get_geometry_hint");
}

RoomBounds::RoomBounds() {

	area = VisualServer::get_singleton()->room_create();
}

RoomBounds::~RoomBounds() {

	VisualServer::get_singleton()->free(area);
}

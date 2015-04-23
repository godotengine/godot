/*************************************************************************/
/*  concave_polygon_shape.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#include "concave_polygon_shape.h"

#include "servers/physics_server.h"

bool ConcavePolygonShape::_set(const StringName& p_name, const Variant& p_value) {

	if (p_name=="data")
		PhysicsServer::get_singleton()->shape_set_data(get_shape(),p_value);
	else
		return false;

	return true;

}

bool ConcavePolygonShape::_get(const StringName& p_name,Variant &r_ret) const {

	if (p_name=="data")
		r_ret=PhysicsServer::get_singleton()->shape_get_data(get_shape());
	else
		return false;
	return true;

}
void ConcavePolygonShape::_get_property_list( List<PropertyInfo> *p_list) const {

	p_list->push_back( PropertyInfo(Variant::ARRAY,"data") );
}


void ConcavePolygonShape::_update_shape() {

}

void ConcavePolygonShape::set_faces(const DVector<Vector3>& p_faces) {

	PhysicsServer::get_singleton()->shape_set_data(get_shape(),p_faces);
	notify_change_to_owners();
}

DVector<Vector3> ConcavePolygonShape::get_faces() const {

	return PhysicsServer::get_singleton()->shape_get_data(get_shape());

}



void ConcavePolygonShape::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_faces","faces"),&ConcavePolygonShape::set_faces);
	ObjectTypeDB::bind_method(_MD("get_faces"),&ConcavePolygonShape::get_faces);
}

ConcavePolygonShape::ConcavePolygonShape() : Shape( PhysicsServer::get_singleton()->shape_create(PhysicsServer::SHAPE_CONCAVE_POLYGON)) {

	//set_planes(Vector3(1,1,1));
}

/*************************************************************************/
/*  capsule_shape_2d.cpp                                                 */
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
#include "capsule_shape_2d.h"

#include "servers/physics_2d_server.h"

void CapsuleShape2D::_update_shape() {

	Physics2DServer::get_singleton()->shape_set_data(get_rid(),Vector2(radius,height));
	emit_changed();
}


void CapsuleShape2D::set_radius(real_t p_radius) {

	radius=p_radius;
	_update_shape();
}

real_t CapsuleShape2D::get_radius() const {

	return radius;
}

void CapsuleShape2D::set_height(real_t p_height) {

	height=p_height;
	_update_shape();
}

real_t CapsuleShape2D::get_height() const {

	return height;
}


void CapsuleShape2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_radius","radius"),&CapsuleShape2D::set_radius);
	ObjectTypeDB::bind_method(_MD("get_radius"),&CapsuleShape2D::get_radius);

	ObjectTypeDB::bind_method(_MD("set_height","height"),&CapsuleShape2D::set_height);
	ObjectTypeDB::bind_method(_MD("get_height"),&CapsuleShape2D::get_height);


	ADD_PROPERTY( PropertyInfo(Variant::REAL,"radius"),_SCS("set_radius"),_SCS("get_radius") );
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"height"),_SCS("set_height"),_SCS("get_height") );

}

CapsuleShape2D::CapsuleShape2D() : Shape2D( Physics2DServer::get_singleton()->shape_create(Physics2DServer::SHAPE_CAPSULE)) {

	radius=10;
	height=20;
	_update_shape();
}

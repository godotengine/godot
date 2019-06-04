/*************************************************************************/
/*  space_2d.cpp                                                         */
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
#include "space_2d.h"

RID Space2D::get_rid() const {

	return space;
}

void Space2D::set_active(bool p_active) {

	active = p_active;
	Physics2DServer::get_singleton()->space_set_active(space, active);
}

bool Space2D::is_active() const {

	return active;
}

void Space2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_active", "active"), &Space2D::set_active);
	ObjectTypeDB::bind_method(_MD("is_active"), &Space2D::is_active);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), _SCS("set_active"), _SCS("is_active"));
}

Space2D::Space2D() {

	active = false;
	space = Physics2DServer::get_singleton()->space_create();
}

Space2D::~Space2D() {

	Physics2DServer::get_singleton()->free(space);
}

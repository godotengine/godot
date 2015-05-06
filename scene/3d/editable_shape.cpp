/*************************************************************************/
/*  editable_shape.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "editable_shape.h"


void EditableShape::_notification(int p_what) {



}


void EditableShape::set_bsp_tree(const BSP_Tree& p_bsp) {

	bsp=p_bsp;
}

void EditableShape::set_shape(const Ref<Shape>& p_shape) {

	shape=p_shape;
}



EditableShape::EditableShape()
{
}



/////////////////////////


void EditableSphere::set_radius(float p_radius) {

	radius=p_radius;
	update_gizmo();
	_change_notify("params/radius");
}


float EditableSphere::get_radius() const{

	return radius;
}


void EditableSphere::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_radius","radius"),&EditableSphere::set_radius);
	ObjectTypeDB::bind_method(_MD("get_radius"),&EditableSphere::get_radius);

	ADD_PROPERTY( PropertyInfo(Variant::REAL,"params/radius",PROPERTY_HINT_EXP_RANGE,"0.001,16384,0.001"),_SCS("set_radius"),_SCS("get_radius"));
}

EditableSphere::EditableSphere() {

	radius=1.0;
}

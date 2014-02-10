/*************************************************************************/
/*  path_2d.cpp                                                          */
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
#include "path_2d.h"


void Path2D::_notification(int p_what) {

	if (p_what==NOTIFICATION_DRAW && curve.is_valid() && is_inside_scene() && get_scene()->is_editor_hint()) {
		//draw the curve!!

		for(int i=0;i<curve->get_point_count();i++) {

			Vector2 prev_p=curve->get_point_pos(i);

			for(int j=1;j<=8;j++) {

				real_t frac = j/8.0;
				Vector2 p = curve->interpolate(i,frac);
				draw_line(prev_p,p,Color(0.5,0.6,1.0,0.7),2);
				prev_p=p;
			}
		}
	}
}

void Path2D::_curve_changed() {


	if (is_inside_scene() && get_scene()->is_editor_hint())
		update();

}


void Path2D::set_curve(const Ref<Curve2D>& p_curve) {

	if (curve.is_valid()) {
		curve->disconnect("changed",this,"_curve_changed");
	}

	curve=p_curve;

	if (curve.is_valid()) {
		curve->connect("changed",this,"_curve_changed");
	}

}

Ref<Curve2D> Path2D::get_curve() const{

	return curve;
}

void Path2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_curve","curve:Curve2D"),&Path2D::set_curve);
	ObjectTypeDB::bind_method(_MD("get_curve:Curve2D","curve"),&Path2D::get_curve);
	ObjectTypeDB::bind_method(_MD("_curve_changed"),&Path2D::_curve_changed);

	ADD_PROPERTY( PropertyInfo( Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve2D"), _SCS("set_curve"),_SCS("get_curve"));
}

Path2D::Path2D() {

	set_curve(Ref<Curve2D>( memnew( Curve2D ))); //create one by default
}

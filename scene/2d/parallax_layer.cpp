/*************************************************************************/
/*  parallax_layer.cpp                                                   */
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
#include "parallax_layer.h"
#include "parallax_background.h"

void ParallaxLayer::set_motion_scale(const Size2& p_scale) {

	motion_scale=p_scale;
}

Size2 ParallaxLayer::get_motion_scale() const {

	return motion_scale;

}


void ParallaxLayer::_update_mirroring() {

	if (!get_parent())
		return;

	ParallaxBackground *pb = get_parent()->cast_to<ParallaxBackground>();
	if (pb) {

		RID c = pb->get_world_2d()->get_canvas();
		RID ci = get_canvas_item();
		VisualServer::get_singleton()->canvas_set_item_mirroring(c,ci,mirroring);
	}

}

void ParallaxLayer::set_mirroring(const Size2& p_mirroring) {

	mirroring=p_mirroring;
	_update_mirroring();

}

Size2 ParallaxLayer::get_mirroring() const{

	return mirroring;
}



void ParallaxLayer::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			orig_offset=get_pos();
			orig_scale=get_scale();
			_update_mirroring();
		} break;
	}
}

void ParallaxLayer::set_base_offset_and_scale(const Point2& p_offset,float p_scale) {

	if (!is_inside_tree())
		return;
	if (get_tree()->is_editor_hint())
		return;
	Point2 new_ofs = ((orig_offset+p_offset)*motion_scale)*p_scale;

	if (mirroring.x) {

		while( new_ofs.x>=0) {
			new_ofs.x -= mirroring.x*p_scale;
		}
		while(new_ofs.x < -mirroring.x*p_scale) {
			new_ofs.x += mirroring.x*p_scale;
		}
	}

	if (mirroring.y) {

		while( new_ofs.y>=0) {
			new_ofs.y -= mirroring.y*p_scale;
		}
		while(new_ofs.y < -mirroring.y*p_scale) {
			new_ofs.y += mirroring.y*p_scale;
		}
	}


	set_pos(new_ofs);
	set_scale(Vector2(1,1)*p_scale);


}

void ParallaxLayer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_motion_scale","scale"),&ParallaxLayer::set_motion_scale);
	ObjectTypeDB::bind_method(_MD("get_motion_scale"),&ParallaxLayer::get_motion_scale);
	ObjectTypeDB::bind_method(_MD("set_mirroring","mirror"),&ParallaxLayer::set_mirroring);
	ObjectTypeDB::bind_method(_MD("get_mirroring"),&ParallaxLayer::get_mirroring);

	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2,"motion/scale"),_SCS("set_motion_scale"),_SCS("get_motion_scale"));
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2,"motion/mirroring"),_SCS("set_mirroring"),_SCS("get_mirroring"));

}



ParallaxLayer::ParallaxLayer()
{
	motion_scale=Size2(1,1);
}

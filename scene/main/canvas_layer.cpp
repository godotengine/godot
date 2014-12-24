/*************************************************************************/
/*  canvas_layer.cpp                                                     */
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
#include "canvas_layer.h"
#include "viewport.h"


void CanvasLayer::set_layer(int p_xform) {

	layer=p_xform;
	if (viewport.is_valid())
		VisualServer::get_singleton()->viewport_set_canvas_layer(viewport,canvas->get_canvas(),layer);

}

int CanvasLayer::get_layer() const{

	return layer;
}

void CanvasLayer::set_transform(const Matrix32& p_xform) {

	transform=p_xform;
	locrotscale_dirty=true;
	if (viewport.is_valid())
		VisualServer::get_singleton()->viewport_set_canvas_transform(viewport,canvas->get_canvas(),transform);

}

Matrix32 CanvasLayer::get_transform() const {

	return transform;
}

void CanvasLayer::_update_xform() {

	transform.set_rotation_and_scale(rot,scale);
	transform.set_origin(ofs);
	if (viewport.is_valid())
		VisualServer::get_singleton()->viewport_set_canvas_transform(viewport,canvas->get_canvas(),transform);

}

void CanvasLayer::_update_locrotscale() {

	ofs=transform.elements[2];
	rot=transform.get_rotation();
	scale=transform.get_scale();
	locrotscale_dirty=false;
}


void CanvasLayer::set_offset(const Vector2& p_offset) {

	if (locrotscale_dirty)
		_update_locrotscale();

	ofs=p_offset;
	_update_xform();

}

Vector2 CanvasLayer::get_offset() const {

	if (locrotscale_dirty)
		const_cast<CanvasLayer*>(this)->_update_locrotscale();

	return ofs;
}


void CanvasLayer::set_rotation(real_t p_rotation) {

	if (locrotscale_dirty)
		_update_locrotscale();


	rot=p_rotation;
	_update_xform();

}

real_t CanvasLayer::get_rotation() const {

	if (locrotscale_dirty)
		const_cast<CanvasLayer*>(this)->_update_locrotscale();

	return rot;
}


void CanvasLayer::set_scale(const Vector2& p_scale) {

	if (locrotscale_dirty)
		_update_locrotscale();

	scale=p_scale;
	_update_xform();


}

Vector2 CanvasLayer::get_scale() const {

	if (locrotscale_dirty)
		const_cast<CanvasLayer*>(this)->_update_locrotscale();

	return scale;
}



Ref<World2D> CanvasLayer::get_world_2d() const {

	return canvas;
}

void CanvasLayer::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			Node *n = this;
			vp=NULL;

			while(n) {

				if (n->cast_to<Viewport>()) {

					vp = n->cast_to<Viewport>();
					break;
				}
				n=n->get_parent();
			}


			ERR_FAIL_COND(!vp);
			viewport=vp->get_viewport();

			VisualServer::get_singleton()->viewport_attach_canvas(viewport,canvas->get_canvas());
			VisualServer::get_singleton()->viewport_set_canvas_layer(viewport,canvas->get_canvas(),layer);
			VisualServer::get_singleton()->viewport_set_canvas_transform(viewport,canvas->get_canvas(),transform);


		} break;
		case NOTIFICATION_EXIT_TREE: {

			VisualServer::get_singleton()->viewport_remove_canvas(viewport,canvas->get_canvas());
			viewport=RID();

		} break;
	}
}

Size2 CanvasLayer::get_viewport_size() const {

	if (!is_inside_tree())
		return Size2(1,1);

	Rect2 r = vp->get_visible_rect();
	return r.size;
}


RID CanvasLayer::get_viewport() const {

	return viewport;
}

void CanvasLayer::_set_rotationd(real_t p_rotation) {

	set_rotation(Math::deg2rad(p_rotation));
}

real_t CanvasLayer::_get_rotationd() const {

	return Math::rad2deg(get_rotation());
}


void CanvasLayer::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("set_layer","layer"),&CanvasLayer::set_layer);
	ObjectTypeDB::bind_method(_MD("get_layer"),&CanvasLayer::get_layer);

	ObjectTypeDB::bind_method(_MD("set_transform","transform"),&CanvasLayer::set_transform);
	ObjectTypeDB::bind_method(_MD("get_transform"),&CanvasLayer::get_transform);

	ObjectTypeDB::bind_method(_MD("set_offset","offset"),&CanvasLayer::set_offset);
	ObjectTypeDB::bind_method(_MD("get_offset"),&CanvasLayer::get_offset);

	ObjectTypeDB::bind_method(_MD("set_rotation","rotation"),&CanvasLayer::set_rotation);
	ObjectTypeDB::bind_method(_MD("get_rotation"),&CanvasLayer::get_rotation);

	ObjectTypeDB::bind_method(_MD("_set_rotationd","rotationd"),&CanvasLayer::_set_rotationd);
	ObjectTypeDB::bind_method(_MD("_get_rotationd"),&CanvasLayer::_get_rotationd);

	ObjectTypeDB::bind_method(_MD("set_scale","scale"),&CanvasLayer::set_scale);
	ObjectTypeDB::bind_method(_MD("get_scale"),&CanvasLayer::get_scale);

	ObjectTypeDB::bind_method(_MD("get_world_2d:Canvas"),&CanvasLayer::get_world_2d);
	ObjectTypeDB::bind_method(_MD("get_viewport"),&CanvasLayer::get_viewport);

	ADD_PROPERTY( PropertyInfo(Variant::INT,"layer",PROPERTY_HINT_RANGE,"-128,128,1"),_SCS("set_layer"),_SCS("get_layer") );
	//ADD_PROPERTY( PropertyInfo(Variant::MATRIX32,"transform",PROPERTY_HINT_RANGE),_SCS("set_transform"),_SCS("get_transform") );
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2,"offset"),_SCS("set_offset"),_SCS("get_offset") );
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"rotation"),_SCS("_set_rotationd"),_SCS("_get_rotationd") );
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2,"scale"),_SCS("set_scale"),_SCS("get_scale") );

}

CanvasLayer::CanvasLayer() {

	vp=NULL;
	scale=Vector2(1,1);
	rot=0;
	locrotscale_dirty=false;
	layer=1;
	canvas = Ref<World2D>( memnew(World2D) );
}

/*************************************************************************/
/*  camera_2d.cpp                                                        */
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
#include "camera_2d.h"
#include "scene/scene_string_names.h"
#include "servers/visual_server.h"

void Camera2D::_update_scroll() {


	if (!is_inside_tree())
		return;

	if (get_tree()->is_editor_hint()) {
		update(); //will just be drawn
		return;
	}

	if (current) {
		Matrix32 xform = get_camera_transform();

		RID vp =  viewport->get_viewport();
		if (viewport) {
		       viewport->set_canvas_transform( xform );
		}
		get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,group_name,"_camera_moved",xform);
	};

}

void Camera2D::set_zoom(const Vector2 &p_zoom) {

	zoom = p_zoom;
	_update_scroll();
};

Vector2 Camera2D::get_zoom() const {

	return zoom;
};


Matrix32 Camera2D::get_camera_transform()  {

	if (!get_tree())
		return Matrix32();

	Size2 screen_size = get_viewport_rect().size;
	screen_size=get_viewport_rect().size;


	Point2 new_camera_pos = get_global_transform().get_origin();
	Point2 ret_camera_pos;

	if (!first) {


		if (centered) {

            if (h_drag_enabled) {
                camera_pos.x = MIN( camera_pos.x, (new_camera_pos.x + screen_size.x * 0.5 * drag_margin[MARGIN_RIGHT]));
                camera_pos.x = MAX( camera_pos.x, (new_camera_pos.x - screen_size.x * 0.5 * drag_margin[MARGIN_LEFT]));
            } else {

		if (h_ofs<0) {
                    camera_pos.x = new_camera_pos.x + screen_size.x * 0.5 * drag_margin[MARGIN_RIGHT] * h_ofs;
                } else {
                    camera_pos.x = new_camera_pos.x + screen_size.x * 0.5 * drag_margin[MARGIN_LEFT] * h_ofs;
                }
            }

            if (v_drag_enabled) {

                camera_pos.y = MIN( camera_pos.y, (new_camera_pos.y + screen_size.y * 0.5 * drag_margin[MARGIN_BOTTOM]));
                camera_pos.y = MAX( camera_pos.y, (new_camera_pos.y - screen_size.y * 0.5 * drag_margin[MARGIN_TOP]));

            } else {

                if (v_ofs<0) {
                    camera_pos.y = new_camera_pos.y + screen_size.y * 0.5 * drag_margin[MARGIN_TOP] * v_ofs;
                } else {
                    camera_pos.y = new_camera_pos.y + screen_size.y * 0.5 * drag_margin[MARGIN_BOTTOM] * v_ofs;
                }
            }

		}


		if (smoothing>0.0) {

			float c = smoothing*get_fixed_process_delta_time();
			smoothed_camera_pos = ((new_camera_pos-smoothed_camera_pos)*c)+smoothed_camera_pos;
			ret_camera_pos=smoothed_camera_pos;
//			camera_pos=camera_pos*(1.0-smoothing)+new_camera_pos*smoothing;
		} else {

			ret_camera_pos=smoothed_camera_pos=camera_pos;

		}



	} else {
		ret_camera_pos=smoothed_camera_pos=camera_pos=new_camera_pos;
		first=false;
	}


	Point2 screen_offset = (centered ? (screen_size * 0.5 * zoom) : Point2());;
	screen_offset;

	float angle = get_global_transform().get_rotation();
	if(rotating){
		screen_offset = screen_offset.rotated(angle);
	}

	Rect2 screen_rect(-screen_offset+ret_camera_pos,screen_size);
	if (screen_rect.pos.x + screen_rect.size.x > limit[MARGIN_RIGHT])
		screen_rect.pos.x = limit[MARGIN_RIGHT] - screen_rect.size.x;

	if (screen_rect.pos.y + screen_rect.size.y > limit[MARGIN_BOTTOM])
		screen_rect.pos.y = limit[MARGIN_BOTTOM] - screen_rect.size.y;


	if (screen_rect.pos.x < limit[MARGIN_LEFT])
		screen_rect.pos.x=limit[MARGIN_LEFT];

	if (screen_rect.pos.y < limit[MARGIN_TOP])
		screen_rect.pos.y =limit[MARGIN_TOP];

	if (offset!=Vector2()) {

		screen_rect.pos+=offset;
		if (screen_rect.pos.x + screen_rect.size.x > limit[MARGIN_RIGHT])
			screen_rect.pos.x = limit[MARGIN_RIGHT] - screen_rect.size.x;

		if (screen_rect.pos.y + screen_rect.size.y > limit[MARGIN_BOTTOM])
			screen_rect.pos.y = limit[MARGIN_BOTTOM] - screen_rect.size.y;


		if (screen_rect.pos.x < limit[MARGIN_LEFT])
			screen_rect.pos.x=limit[MARGIN_LEFT];

		if (screen_rect.pos.y < limit[MARGIN_TOP])
			screen_rect.pos.y =limit[MARGIN_TOP];

	}

	camera_screen_center=screen_rect.pos+screen_rect.size*0.5;

	Matrix32 xform;
	if(rotating){
		xform.set_rotation(angle);
	}
	xform.scale_basis(zoom);
	xform.set_origin(screen_rect.pos/*.floor()*/);


/*
	if (0) {

		xform = get_global_transform() * xform;
	} else {

		xform.elements[2]+=get_global_transform().get_origin();
	}
*/


	return (xform).affine_inverse();
}



void Camera2D::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_FIXED_PROCESS: {

			_update_scroll();

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {


			if (!is_fixed_processing())
				_update_scroll();

		} break;
		case NOTIFICATION_ENTER_TREE: {

			viewport = NULL;
			Node *n=this;
			while(n){

				viewport = n->cast_to<Viewport>();
				if (viewport)
					break;
				n=n->get_parent();
			}

			canvas = get_canvas();

			RID vp = viewport->get_viewport();

			group_name = "__cameras_"+itos(vp.get_id());
			canvas_group_name ="__cameras_c"+itos(canvas.get_id());
			add_to_group(group_name);
			add_to_group(canvas_group_name);

			_update_scroll();
			first=true;


		} break;
		case NOTIFICATION_EXIT_TREE: {

			if (is_current()) {
				if (viewport) {
					viewport->set_canvas_transform( Matrix32() );
				}
			}
			remove_from_group(group_name);
			remove_from_group(canvas_group_name);
			viewport=NULL;

		} break;
	}
}

void Camera2D::set_offset(const Vector2& p_offset) {

	offset=p_offset;
	_update_scroll();

}

Vector2 Camera2D::get_offset() const{

	return offset;
}

void Camera2D::set_centered(bool p_centered){

	centered=p_centered;
	_update_scroll();
}

bool Camera2D::is_centered() const {

	return centered;
}

void Camera2D::set_rotating(bool p_rotating){

	rotating=p_rotating;
	_update_scroll();
}

bool Camera2D::is_rotating() const {

	return rotating;
}


void Camera2D::_make_current(Object *p_which) {

	if (p_which==this) {

		current=true;
		_update_scroll();
	} else {
		current=false;
	}
}


void Camera2D::_set_current(bool p_current) {

	if (p_current)
		make_current();

	current=p_current;
}

bool Camera2D::is_current() const {

	return current;
}

void Camera2D::make_current() {

	if (!is_inside_tree()) {
		current=true;
	} else {
		get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,group_name,"_make_current",this);
	}
}

void Camera2D::set_limit(Margin p_margin,int p_limit) {

	ERR_FAIL_INDEX(p_margin,4);
	limit[p_margin]=p_limit;
}

int Camera2D::get_limit(Margin p_margin) const{

	ERR_FAIL_INDEX_V(p_margin,4,0);
	return limit[p_margin];

}

void Camera2D::set_drag_margin(Margin p_margin,float p_drag_margin) {

	ERR_FAIL_INDEX(p_margin,4);
	drag_margin[p_margin]=p_drag_margin;
}

float Camera2D::get_drag_margin(Margin p_margin) const{

	ERR_FAIL_INDEX_V(p_margin,4,0);
	return drag_margin[p_margin];

}


Vector2 Camera2D::get_camera_pos() const {


	return camera_pos;
}

void Camera2D::force_update_scroll() {


	_update_scroll();
}


void Camera2D::set_follow_smoothing(float p_speed) {

	smoothing=p_speed;
	if (smoothing>0)
		set_fixed_process(true);
	else
		set_fixed_process(false);
}

float Camera2D::get_follow_smoothing() const{

	return smoothing;
}

Point2 Camera2D::get_camera_screen_center() const {

       return camera_screen_center;
}


void Camera2D::set_h_drag_enabled(bool p_enabled) {

    h_drag_enabled=p_enabled;
}

bool Camera2D::is_h_drag_enabled() const{

    return h_drag_enabled;
}

void Camera2D::set_v_drag_enabled(bool p_enabled){

    v_drag_enabled=p_enabled;
}

bool Camera2D::is_v_drag_enabled() const{

    return v_drag_enabled;
}

void Camera2D::set_v_offset(float p_offset) {

    v_ofs=p_offset;
}

float Camera2D::get_v_offset() const{

    return v_ofs;
}

void Camera2D::set_h_offset(float p_offset){

    h_ofs=p_offset;
}
float Camera2D::get_h_offset() const{

    return h_ofs;
}


void Camera2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_offset","offset"),&Camera2D::set_offset);
	ObjectTypeDB::bind_method(_MD("get_offset"),&Camera2D::get_offset);

	ObjectTypeDB::bind_method(_MD("set_centered","centered"),&Camera2D::set_centered);
	ObjectTypeDB::bind_method(_MD("is_centered"),&Camera2D::is_centered);

	ObjectTypeDB::bind_method(_MD("set_rotating","rotating"),&Camera2D::set_rotating);
	ObjectTypeDB::bind_method(_MD("is_rotating"),&Camera2D::is_rotating);

	ObjectTypeDB::bind_method(_MD("make_current"),&Camera2D::make_current);
	ObjectTypeDB::bind_method(_MD("_make_current"),&Camera2D::_make_current);

	ObjectTypeDB::bind_method(_MD("_update_scroll"),&Camera2D::_update_scroll);


	ObjectTypeDB::bind_method(_MD("_set_current","current"),&Camera2D::_set_current);
	ObjectTypeDB::bind_method(_MD("is_current"),&Camera2D::is_current);

	ObjectTypeDB::bind_method(_MD("set_limit","margin","limit"),&Camera2D::set_limit);
	ObjectTypeDB::bind_method(_MD("get_limit","margin"),&Camera2D::get_limit);

	ObjectTypeDB::bind_method(_MD("set_v_drag_enabled","enabled"),&Camera2D::set_v_drag_enabled);
	ObjectTypeDB::bind_method(_MD("is_v_drag_enabled"),&Camera2D::is_v_drag_enabled);

	ObjectTypeDB::bind_method(_MD("set_h_drag_enabled","enabled"),&Camera2D::set_h_drag_enabled);
	ObjectTypeDB::bind_method(_MD("is_h_drag_enabled"),&Camera2D::is_h_drag_enabled);

	ObjectTypeDB::bind_method(_MD("set_v_offset","ofs"),&Camera2D::set_v_offset);
	ObjectTypeDB::bind_method(_MD("get_v_offset"),&Camera2D::get_v_offset);

	ObjectTypeDB::bind_method(_MD("set_h_offset","ofs"),&Camera2D::set_h_offset);
	ObjectTypeDB::bind_method(_MD("get_h_offset"),&Camera2D::get_h_offset);

	ObjectTypeDB::bind_method(_MD("set_drag_margin","margin","drag_margin"),&Camera2D::set_drag_margin);
	ObjectTypeDB::bind_method(_MD("get_drag_margin","margin"),&Camera2D::get_drag_margin);

	ObjectTypeDB::bind_method(_MD("get_camera_pos"),&Camera2D::get_camera_pos);
	ObjectTypeDB::bind_method(_MD("get_camera_screen_center"),&Camera2D::get_camera_screen_center);

	ObjectTypeDB::bind_method(_MD("set_zoom"),&Camera2D::set_zoom);
	ObjectTypeDB::bind_method(_MD("get_zoom"),&Camera2D::get_zoom);


	ObjectTypeDB::bind_method(_MD("set_follow_smoothing","follow_smoothing"),&Camera2D::set_follow_smoothing);
	ObjectTypeDB::bind_method(_MD("get_follow_smoothing"),&Camera2D::get_follow_smoothing);

	ObjectTypeDB::bind_method(_MD("force_update_scroll"),&Camera2D::force_update_scroll);


	ADD_PROPERTYNZ( PropertyInfo(Variant::VECTOR2,"offset"),_SCS("set_offset"),_SCS("get_offset"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"centered"),_SCS("set_centered"),_SCS("is_centered"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"rotating"),_SCS("set_rotating"),_SCS("is_rotating"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"current"),_SCS("_set_current"),_SCS("is_current"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"smoothing"),_SCS("set_follow_smoothing"),_SCS("get_follow_smoothing") );
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2,"zoom"),_SCS("set_zoom"),_SCS("get_zoom") );

	ADD_PROPERTYI( PropertyInfo(Variant::INT,"limit/left"),_SCS("set_limit"),_SCS("get_limit"),MARGIN_LEFT);
	ADD_PROPERTYI( PropertyInfo(Variant::INT,"limit/top"),_SCS("set_limit"),_SCS("get_limit"),MARGIN_TOP);
	ADD_PROPERTYI( PropertyInfo(Variant::INT,"limit/right"),_SCS("set_limit"),_SCS("get_limit"),MARGIN_RIGHT);
	ADD_PROPERTYI( PropertyInfo(Variant::INT,"limit/bottom"),_SCS("set_limit"),_SCS("get_limit"),MARGIN_BOTTOM);

	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"drag_margin/h_enabled"),_SCS("set_h_drag_enabled"),_SCS("is_h_drag_enabled") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"drag_margin/v_enabled"),_SCS("set_v_drag_enabled"),_SCS("is_v_drag_enabled") );

	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"drag_margin/left",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_drag_margin"),_SCS("get_drag_margin"),MARGIN_LEFT);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"drag_margin/top",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_drag_margin"),_SCS("get_drag_margin"),MARGIN_TOP);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"drag_margin/right",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_drag_margin"),_SCS("get_drag_margin"),MARGIN_RIGHT);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"drag_margin/bottom",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_drag_margin"),_SCS("get_drag_margin"),MARGIN_BOTTOM);



}

Camera2D::Camera2D() {



	centered=true;
	rotating=false;
	current=false;
	limit[MARGIN_LEFT]=-10000000;
	limit[MARGIN_TOP]=-10000000;
	limit[MARGIN_RIGHT]=10000000;
	limit[MARGIN_BOTTOM]=10000000;
	drag_margin[MARGIN_LEFT]=0.2;
	drag_margin[MARGIN_TOP]=0.2;
	drag_margin[MARGIN_RIGHT]=0.2;
	drag_margin[MARGIN_BOTTOM]=0.2;
	camera_pos=Vector2();

	smoothing=0.0;
	zoom = Vector2(1, 1);

	h_drag_enabled=true;
	v_drag_enabled=true;
	h_ofs=0;
	v_ofs=0;

}

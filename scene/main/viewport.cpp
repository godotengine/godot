/*************************************************************************/
/*  viewport.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "viewport.h"
#include "os/os.h"
#include "scene/3d/spatial.h"
#include "os/input.h"
#include "servers/physics_2d_server.h"
//#include "scene/3d/camera.h"

#include "servers/spatial_sound_server.h"
#include "servers/spatial_sound_2d_server.h"
#include "scene/gui/control.h"
#include "scene/3d/camera.h"
#include "scene/3d/listener.h"
#include "scene/resources/mesh.h"
#include "scene/3d/spatial_indexer.h"
#include "scene/3d/collision_object.h"

#include "scene/2d/collision_object_2d.h"

#include "scene/gui/panel.h"
#include "scene/gui/label.h"
#include "scene/main/timer.h"
#include "scene/scene_string_names.h"

#include "globals.h"

int RenderTargetTexture::get_width() const {

	ERR_FAIL_COND_V(!vp,0);
	return vp->rect.size.width;
}
int RenderTargetTexture::get_height() const{

	ERR_FAIL_COND_V(!vp,0);
	return vp->rect.size.height;
}
Size2 RenderTargetTexture::get_size() const{

	ERR_FAIL_COND_V(!vp,Size2());
	return vp->rect.size;
}
RID RenderTargetTexture::get_rid() const{

	ERR_FAIL_COND_V(!vp,RID());
	return vp->render_target_texture_rid;
}

bool RenderTargetTexture::has_alpha() const{

	return false;
}

void RenderTargetTexture::set_flags(uint32_t p_flags){

	ERR_FAIL_COND(!vp);
	if (p_flags&FLAG_FILTER)
		flags=FLAG_FILTER;
	else
		flags=0;

	VS::get_singleton()->texture_set_flags(vp->render_target_texture_rid,flags);

}

uint32_t RenderTargetTexture::get_flags() const{

	return flags;
}

RenderTargetTexture::RenderTargetTexture(Viewport *p_vp){

	vp=p_vp;
	flags=0;
}

/////////////////////////////////////

class TooltipPanel : public Panel {

	OBJ_TYPE(TooltipPanel,Panel)
public:
	TooltipPanel() {};

};

class TooltipLabel : public Label {

	OBJ_TYPE(TooltipLabel,Label)
public:
	TooltipLabel() {};

};


Viewport::GUI::GUI() {


	mouse_focus=NULL;
	mouse_focus_button=-1;
	key_focus=NULL;
	mouse_over=NULL;

	cancelled_input_ID=0;
	tooltip=NULL;
	tooltip_popup=NULL;
	tooltip_label=NULL;
	subwindow_order_dirty=false;
}


/////////////////////////////////////
void Viewport::_update_stretch_transform() {

	if (size_override_stretch && size_override) {

		//print_line("sive override size "+size_override_size);
		//print_line("rect size "+rect.size);
		stretch_transform=Matrix32();
		Size2 scale = rect.size/(size_override_size+size_override_margin*2);
		stretch_transform.scale(scale);
		stretch_transform.elements[2]=size_override_margin*scale;


	} else {


		stretch_transform=Matrix32();
	}

	_update_global_transform();

}

void Viewport::_update_rect() {

	if (!is_inside_tree())
		return;


	if (!render_target && parent_control) {

		Control *c = parent_control;

		rect.pos=Point2();
		rect.size=c->get_size();
	}

	VisualServer::ViewportRect vr;
	vr.x=rect.pos.x;
	vr.y=rect.pos.y;

	if (render_target) {
		vr.x=0;
		vr.y=0;
	}
	vr.width=rect.size.width;
	vr.height=rect.size.height;

	VisualServer::get_singleton()->viewport_set_rect(viewport,vr);
	last_vp_rect=rect;

	if (canvas_item.is_valid()) {
		VisualServer::get_singleton()->canvas_item_set_custom_rect(canvas_item,true,rect);
	}

	emit_signal("size_changed");
	render_target_texture->emit_changed();


}

void Viewport::_parent_resized() {

	_update_rect();
}

void Viewport::_parent_draw() {

}

void Viewport::_parent_visibility_changed() {


	if (parent_control) {

		Control *c = parent_control;
		VisualServer::get_singleton()->canvas_item_set_visible(canvas_item,c->is_visible());

		_update_listener();
		_update_listener_2d();
	}


}


void Viewport::_vp_enter_tree() {

	if (parent_control) {

		Control *cparent=parent_control;
		RID parent_ci = cparent->get_canvas_item();
		ERR_FAIL_COND(!parent_ci.is_valid());
		canvas_item = VisualServer::get_singleton()->canvas_item_create();

		VisualServer::get_singleton()->canvas_item_set_parent(canvas_item,parent_ci);
		VisualServer::get_singleton()->canvas_item_set_visible(canvas_item,false);
		VisualServer::get_singleton()->canvas_item_attach_viewport(canvas_item,viewport);
		parent_control->connect("resized",this,"_parent_resized");
		parent_control->connect("visibility_changed",this,"_parent_visibility_changed");
	} else if (!parent){

		VisualServer::get_singleton()->viewport_attach_to_screen(viewport,0);

	}


}

void Viewport::_vp_exit_tree() {

	if (parent_control) {

		parent_control->disconnect("resized",this,"_parent_resized");
	}

	if (parent_control) {

		parent_control->disconnect("visibility_changed",this,"_parent_visibility_changed");
	}

	if (canvas_item.is_valid()) {

		VisualServer::get_singleton()->free(canvas_item);
		canvas_item=RID();

	}

	if (!parent) {

		VisualServer::get_singleton()->viewport_detach(viewport);

	}

}


void Viewport::update_worlds() {

	if (!is_inside_tree())
		return;

	Rect2 xformed_rect = (global_canvas_transform * canvas_transform).affine_inverse().xform(get_visible_rect());
	find_world_2d()->_update_viewport(this,xformed_rect);
	find_world_2d()->_update();

	find_world()->_update(get_tree()->get_frame());
}


void Viewport::_test_new_mouseover(ObjectID new_collider) {
#ifndef _3D_DISABLED
	if (new_collider!=physics_object_over) {

		if (physics_object_over) {
			Object *obj = ObjectDB::get_instance(physics_object_over);
			if (obj) {
				CollisionObject *co = obj->cast_to<CollisionObject>();
				if (co) {
					co->_mouse_exit();
				}
			}
		}

		if (new_collider) {
			Object *obj = ObjectDB::get_instance(new_collider);
			if (obj) {
				CollisionObject *co = obj->cast_to<CollisionObject>();
				if (co) {
					co->_mouse_enter();

				}
			}

		}

		physics_object_over=new_collider;

	}
#endif

}

void Viewport::_notification(int p_what) {


	switch( p_what ) {

		case NOTIFICATION_ENTER_TREE: {

			if (get_parent()) {
				Node *parent=get_parent();
				if (parent) {
					parent_control=parent->cast_to<Control>();
				}
			}


			parent=NULL;
			Node *parent_node=get_parent();


			while(parent_node) {

				parent = parent_node->cast_to<Viewport>();
				if (parent)
					break;

				parent_node=parent_node->get_parent();
			}


			if (!render_target)
				_vp_enter_tree();


			current_canvas=find_world_2d()->get_canvas();
			VisualServer::get_singleton()->viewport_set_scenario(viewport,find_world()->get_scenario());
			VisualServer::get_singleton()->viewport_attach_canvas(viewport,current_canvas);

			_update_listener();
			_update_listener_2d();
			_update_rect();

			find_world_2d()->_register_viewport(this,Rect2());

			add_to_group("_viewports");
			if (get_tree()->is_debugging_collisions_hint()) {
				//2D
				Physics2DServer::get_singleton()->space_set_debug_contacts(find_world_2d()->get_space(),get_tree()->get_collision_debug_contact_count());
				contact_2d_debug=VisualServer::get_singleton()->canvas_item_create();
				VisualServer::get_singleton()->canvas_item_set_parent(contact_2d_debug,find_world_2d()->get_canvas());
				//3D
				PhysicsServer::get_singleton()->space_set_debug_contacts(find_world()->get_space(),get_tree()->get_collision_debug_contact_count());
				contact_3d_debug_multimesh=VisualServer::get_singleton()->multimesh_create();
				VisualServer::get_singleton()->multimesh_set_instance_count(contact_3d_debug_multimesh,get_tree()->get_collision_debug_contact_count());
				VisualServer::get_singleton()->multimesh_set_visible_instances(contact_3d_debug_multimesh,0);
				VisualServer::get_singleton()->multimesh_set_mesh(contact_3d_debug_multimesh,get_tree()->get_debug_contact_mesh()->get_rid());
				contact_3d_debug_instance=VisualServer::get_singleton()->instance_create();
				VisualServer::get_singleton()->instance_set_base(contact_3d_debug_instance,contact_3d_debug_multimesh);
				VisualServer::get_singleton()->instance_set_scenario(contact_3d_debug_instance,find_world()->get_scenario());
				VisualServer::get_singleton()->instance_geometry_set_flag(contact_3d_debug_instance,VS::INSTANCE_FLAG_VISIBLE_IN_ALL_ROOMS,true);

			}

		} break;
		case NOTIFICATION_READY: {
#ifndef _3D_DISABLED
			if (listeners.size() && !listener) {
				Listener *first=NULL;
				for(Set<Listener*>::Element *E=listeners.front();E;E=E->next()) {

					if (first==NULL || first->is_greater_than(E->get())) {
						first=E->get();
					}
				}

				if (first)
					first->make_current();
			}

			if (cameras.size() && !camera) {
				//there are cameras but no current camera, pick first in tree and make it current
				Camera *first=NULL;
				for(Set<Camera*>::Element *E=cameras.front();E;E=E->next()) {

					if (first==NULL || first->is_greater_than(E->get())) {
						first=E->get();
					}
				}

				if (first)
					first->make_current();
			}
#endif
		} break;
		case NOTIFICATION_EXIT_TREE: {


			_gui_cancel_tooltip();
			if (world_2d.is_valid())
				world_2d->_remove_viewport(this);

			if (!render_target)
				_vp_exit_tree();

			VisualServer::get_singleton()->viewport_set_scenario(viewport,RID());
			SpatialSoundServer::get_singleton()->listener_set_space(internal_listener, RID());
			VisualServer::get_singleton()->viewport_remove_canvas(viewport,current_canvas);
			if (contact_2d_debug.is_valid()) {
				VisualServer::get_singleton()->free(contact_2d_debug);
				contact_2d_debug=RID();
			}

			if (contact_3d_debug_multimesh.is_valid()) {
				VisualServer::get_singleton()->free(contact_3d_debug_multimesh);
				VisualServer::get_singleton()->free(contact_3d_debug_instance);
				contact_3d_debug_instance=RID();
				contact_3d_debug_multimesh=RID();
			}

			remove_from_group("_viewports");
			parent_control=NULL;

		} break;
		case NOTIFICATION_FIXED_PROCESS: {

			if (gui.tooltip_timer>=0) {
				gui.tooltip_timer-=get_fixed_process_delta_time();
				if (gui.tooltip_timer<0) {
					_gui_show_tooltip();
				}
			}

			if (get_tree()->is_debugging_collisions_hint() && contact_2d_debug.is_valid()) {

				VisualServer::get_singleton()->canvas_item_clear(contact_2d_debug);
				VisualServer::get_singleton()->canvas_item_raise(contact_2d_debug);

				Vector<Vector2> points = Physics2DServer::get_singleton()->space_get_contacts(find_world_2d()->get_space());
				int point_count = Physics2DServer::get_singleton()->space_get_contact_count(find_world_2d()->get_space());
				Color ccol = get_tree()->get_debug_collision_contact_color();


				for(int i=0;i<point_count;i++) {

					VisualServer::get_singleton()->canvas_item_add_rect(contact_2d_debug,Rect2(points[i]-Vector2(2,2),Vector2(5,5)),ccol);
				}
			}

			if (get_tree()->is_debugging_collisions_hint() && contact_3d_debug_multimesh.is_valid()) {


				Vector<Vector3> points = PhysicsServer::get_singleton()->space_get_contacts(find_world()->get_space());
				int point_count = PhysicsServer::get_singleton()->space_get_contact_count(find_world()->get_space());


				VS::get_singleton()->multimesh_set_visible_instances(contact_3d_debug_multimesh,point_count);

				if (point_count>0) {
					AABB aabb;

					Transform t;
					for(int i=0;i<point_count;i++) {

						if (i==0)
							aabb.pos=points[i];
						else
							aabb.expand_to(points[i]);
						t.origin=points[i];
						VisualServer::get_singleton()->multimesh_instance_set_transform(contact_3d_debug_multimesh,i,t);
					}
					aabb.grow(aabb.get_longest_axis_size()*0.01);
					VisualServer::get_singleton()->multimesh_set_aabb(contact_3d_debug_multimesh,aabb);
				}
			}



			if (physics_object_picking && (render_target || Input::get_singleton()->get_mouse_mode()!=Input::MOUSE_MODE_CAPTURED)) {

				Vector2 last_pos(1e20,1e20);
				CollisionObject *last_object;
				ObjectID last_id=0;
				PhysicsDirectSpaceState::RayResult result;
				Physics2DDirectSpaceState *ss2d=Physics2DServer::get_singleton()->space_get_direct_state(find_world_2d()->get_space());

				bool motion_tested=false;

				while(physics_picking_events.size()) {

					InputEvent ev = physics_picking_events.front()->get();
					physics_picking_events.pop_front();

					Vector2 pos;
					switch(ev.type) {
						case InputEvent::MOUSE_MOTION: {
							pos.x=ev.mouse_motion.x;
							pos.y=ev.mouse_motion.y;
							motion_tested=true;
							physics_last_mousepos=pos;
						} break;
						case InputEvent::MOUSE_BUTTON: {
							pos.x=ev.mouse_button.x;
							pos.y=ev.mouse_button.y;

						} break;
						case InputEvent::SCREEN_DRAG: {
							pos.x=ev.screen_drag.x;
							pos.y=ev.screen_drag.y;
						} break;
						case InputEvent::SCREEN_TOUCH: {
							pos.x=ev.screen_touch.x;
							pos.y=ev.screen_touch.y;
						} break;

					}

					if (ss2d) {
						//send to 2D


						uint64_t frame = get_tree()->get_frame();

						Vector2 point = get_canvas_transform().affine_inverse().xform(pos);
						Physics2DDirectSpaceState::ShapeResult res[64];
						int rc = ss2d->intersect_point(point,res,64,Set<RID>(),0xFFFFFFFF,0xFFFFFFFF,true);
						for(int i=0;i<rc;i++) {

							if (res[i].collider_id && res[i].collider) {
								CollisionObject2D *co=res[i].collider->cast_to<CollisionObject2D>();
								if (co) {

									Map<ObjectID,uint64_t>::Element *E=physics_2d_mouseover.find(res[i].collider_id);
									if (!E) {
										E=physics_2d_mouseover.insert(res[i].collider_id,frame);
										co->_mouse_enter();
									} else {
										E->get()=frame;
									}

									co->_input_event(this,ev,res[i].shape);
								}
							}
						}

						List<Map<ObjectID,uint64_t>::Element*> to_erase;

						for (Map<ObjectID,uint64_t>::Element*E=physics_2d_mouseover.front();E;E=E->next()) {
							if (E->get()!=frame) {
								Object *o=ObjectDB::get_instance(E->key());
								if (o) {

									CollisionObject2D *co=o->cast_to<CollisionObject2D>();
									if (co) {
										co->_mouse_exit();
									}
								}
								to_erase.push_back(E);
							}
						}

						while(to_erase.size()) {
							physics_2d_mouseover.erase(to_erase.front()->get());
							to_erase.pop_front();
						}

					}



#ifndef _3D_DISABLED
					bool captured=false;

					if (physics_object_capture!=0) {


						Object *obj = ObjectDB::get_instance(physics_object_capture);
						if (obj) {
							CollisionObject *co = obj->cast_to<CollisionObject>();
							if (co) {
								co->_input_event(camera,ev,Vector3(),Vector3(),0);
								captured=true;
								if (ev.type==InputEvent::MOUSE_BUTTON && ev.mouse_button.button_index==1 && !ev.mouse_button.pressed) {
									physics_object_capture=0;
								}

							} else {
								physics_object_capture=0;
							}
						} else {
							physics_object_capture=0;
						}
					}


					if (captured) {
						//none
					} else if (pos==last_pos) {

						if (last_id) {
							if (ObjectDB::get_instance(last_id)) {
								//good, exists
								last_object->_input_event(camera,ev,result.position,result.normal,result.shape);
								if (last_object->get_capture_input_on_drag() && ev.type==InputEvent::MOUSE_BUTTON && ev.mouse_button.button_index==1 && ev.mouse_button.pressed) {
									physics_object_capture=last_id;
								}


							}
						}
					} else {




						if (camera) {

							Vector3 from = camera->project_ray_origin(pos);
							Vector3 dir = camera->project_ray_normal(pos);

							PhysicsDirectSpaceState *space = PhysicsServer::get_singleton()->space_get_direct_state(find_world()->get_space());
							if (space) {

								bool col = space->intersect_ray(from,from+dir*10000,result,Set<RID>(),0xFFFFFFFF,0xFFFFFFFF,true);
								ObjectID new_collider=0;
								if (col) {

									if (result.collider) {

										CollisionObject *co = result.collider->cast_to<CollisionObject>();
										if (co) {

											co->_input_event(camera,ev,result.position,result.normal,result.shape);
											last_object=co;
											last_id=result.collider_id;
											new_collider=last_id;
											if (co->get_capture_input_on_drag() && ev.type==InputEvent::MOUSE_BUTTON && ev.mouse_button.button_index==1 && ev.mouse_button.pressed) {
												physics_object_capture=last_id;
											}

										}
									}
								}

								if (ev.type==InputEvent::MOUSE_MOTION) {
									_test_new_mouseover(new_collider);
								}
							}

							last_pos=pos;
						}
					}
				}

				if (!motion_tested && camera && physics_last_mousepos!=Vector2(1e20,1e20)) {

					//test anyway for mouseenter/exit because objects might move
					Vector3 from = camera->project_ray_origin(physics_last_mousepos);
					Vector3 dir = camera->project_ray_normal(physics_last_mousepos);

					PhysicsDirectSpaceState *space = PhysicsServer::get_singleton()->space_get_direct_state(find_world()->get_space());
					if (space) {

						bool col = space->intersect_ray(from,from+dir*10000,result,Set<RID>(),0xFFFFFFFF,0xFFFFFFFF,true);
						ObjectID new_collider=0;
						if (col) {
							if (result.collider) {
								CollisionObject *co = result.collider->cast_to<CollisionObject>();
								if (co) {
									new_collider=result.collider_id;

								}
							}
						}

						_test_new_mouseover(new_collider);

					}
#endif
				}

			}

		} break;
	}
}

RID Viewport::get_viewport() const {

	return viewport;
}

void Viewport::set_rect(const Rect2& p_rect) {

	if (rect==p_rect)
		return;
	rect=p_rect;

	_update_rect();
	_update_stretch_transform();

}

Rect2 Viewport::get_visible_rect() const {


	Rect2 r;

	if (rect.pos==Vector2() && rect.size==Size2()) {

		r=Rect2( Point2(), Size2( OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height ) );
	} else {

		r=Rect2( rect.pos , rect.size );
	}

	if (size_override) {
		r.size=size_override_size;
	}


	return r;
}

Rect2 Viewport::get_rect() const {

	return rect;
}


void Viewport::_update_listener() {

	if (is_inside_tree() && audio_listener && (camera || listener) && (!get_parent() || (get_parent()->cast_to<Control>() && get_parent()->cast_to<Control>()->is_visible())))  {
		SpatialSoundServer::get_singleton()->listener_set_space(internal_listener, find_world()->get_sound_space());
	} else {
		SpatialSoundServer::get_singleton()->listener_set_space(internal_listener, RID());
	}


}

void Viewport::_update_listener_2d() {

	if (is_inside_tree() && audio_listener && (!get_parent() || (get_parent()->cast_to<Control>() && get_parent()->cast_to<Control>()->is_visible())))
		SpatialSound2DServer::get_singleton()->listener_set_space(internal_listener_2d, find_world_2d()->get_sound_space());
	else
		SpatialSound2DServer::get_singleton()->listener_set_space(internal_listener_2d, RID());

}


void Viewport::set_as_audio_listener(bool p_enable) {

	if (p_enable==audio_listener)
		return;

	audio_listener=p_enable;
	_update_listener();

}

bool Viewport::is_audio_listener() const {

	return  audio_listener;
}

void Viewport::set_as_audio_listener_2d(bool p_enable) {

	if (p_enable==audio_listener_2d)
		return;

	audio_listener_2d=p_enable;

	_update_listener_2d();


}

bool Viewport::is_audio_listener_2d() const {

	return  audio_listener_2d;
}

void Viewport::set_canvas_transform(const Matrix32& p_transform) {

	canvas_transform=p_transform;
	VisualServer::get_singleton()->viewport_set_canvas_transform(viewport,find_world_2d()->get_canvas(),canvas_transform);

	Matrix32 xform = (global_canvas_transform * canvas_transform).affine_inverse();
	Size2 ss = get_visible_rect().size;
	SpatialSound2DServer::get_singleton()->listener_set_transform(internal_listener_2d, Matrix32(0, xform.xform(ss*0.5)));
	Vector2 ss2 = ss*xform.get_scale();
	float panrange = MAX(ss2.x,ss2.y);

	SpatialSound2DServer::get_singleton()->listener_set_param(internal_listener_2d, SpatialSound2DServer::LISTENER_PARAM_PAN_RANGE, panrange);


}

Matrix32 Viewport::get_canvas_transform() const{

	return canvas_transform;
}



void Viewport::_update_global_transform() {


	Matrix32 sxform = stretch_transform * global_canvas_transform;

	VisualServer::get_singleton()->viewport_set_global_canvas_transform(viewport,sxform);

	Matrix32 xform = (sxform * canvas_transform).affine_inverse();
	Size2 ss = get_visible_rect().size;
	SpatialSound2DServer::get_singleton()->listener_set_transform(internal_listener_2d, Matrix32(0, xform.xform(ss*0.5)));
	Vector2 ss2 = ss*xform.get_scale();
	float panrange = MAX(ss2.x,ss2.y);

	SpatialSound2DServer::get_singleton()->listener_set_param(internal_listener_2d, SpatialSound2DServer::LISTENER_PARAM_PAN_RANGE, panrange);

}


void Viewport::set_global_canvas_transform(const Matrix32& p_transform) {

	global_canvas_transform=p_transform;

	_update_global_transform();


}

Matrix32 Viewport::get_global_canvas_transform() const{

	return global_canvas_transform;
}

void Viewport::_listener_transform_changed_notify() {

#ifndef _3D_DISABLED
	if (listener)
		SpatialSoundServer::get_singleton()->listener_set_transform(internal_listener, listener->get_listener_transform());
#endif
}

void Viewport::_listener_set(Listener* p_listener) {

#ifndef _3D_DISABLED

	if (listener == p_listener)
		return;

	listener = p_listener;

	_update_listener();
	_listener_transform_changed_notify();
#endif
}

bool Viewport::_listener_add(Listener* p_listener) {

	listeners.insert(p_listener);
	return listeners.size() == 1;
}

void Viewport::_listener_remove(Listener* p_listener) {

	listeners.erase(p_listener);
	if (listener == p_listener) {
		listener = NULL;
	}
}

#ifndef _3D_DISABLED
void Viewport::_listener_make_next_current(Listener* p_exclude) {

	if (listeners.size() > 0) {
		for (Set<Listener*>::Element *E = listeners.front(); E; E = E->next()) {

			if (p_exclude == E->get())
				continue;
			if (!E->get()->is_inside_tree())
				continue;
			if (listener != NULL)
				return;

			E->get()->make_current();

		}
	}
	else {
		// Attempt to reset listener to the camera position
		if (camera != NULL) {
			_update_listener();
			_camera_transform_changed_notify();
		}
	}
}
#endif

void Viewport::_camera_transform_changed_notify() {

#ifndef _3D_DISABLED
	// If there is an active listener in the scene, it takes priority over the camera
	if (camera && !listener)
		SpatialSoundServer::get_singleton()->listener_set_transform(internal_listener, camera->get_camera_transform());
#endif
}

void Viewport::_camera_set(Camera* p_camera) {

#ifndef _3D_DISABLED

	if (camera==p_camera)
		return;

	if (camera && find_world().is_valid()) {
		camera->notification(Camera::NOTIFICATION_LOST_CURRENT);
	}
	camera=p_camera;
	if (camera)
		VisualServer::get_singleton()->viewport_attach_camera(viewport,camera->get_camera());
	else
		VisualServer::get_singleton()->viewport_attach_camera(viewport,RID());

	if (camera && find_world().is_valid()) {
		camera->notification(Camera::NOTIFICATION_BECAME_CURRENT);
	}

	_update_listener();
	_camera_transform_changed_notify();
#endif
}

bool Viewport::_camera_add(Camera* p_camera) {

	cameras.insert(p_camera);
	return cameras.size()==1;
}

void Viewport::_camera_remove(Camera* p_camera) {

	cameras.erase(p_camera);
	if (camera==p_camera) {
		camera=NULL;
	}
}

#ifndef _3D_DISABLED
void Viewport::_camera_make_next_current(Camera* p_exclude) {

	for(Set<Camera*>::Element *E=cameras.front();E;E=E->next()) {

		if (p_exclude==E->get())
			continue;
		if (!E->get()->is_inside_tree())
			continue;
		if (camera!=NULL)
			return;

		E->get()->make_current();

	}
}
#endif

void Viewport::set_transparent_background(bool p_enable) {

	transparent_bg=p_enable;
	VS::get_singleton()->viewport_set_transparent_background(viewport,p_enable);

}

bool Viewport::has_transparent_background() const {

	return transparent_bg;
}

void Viewport::set_world_2d(const Ref<World2D>& p_world_2d) {
	if (world_2d==p_world_2d)
		return;

	if (parent && parent->find_world_2d()==p_world_2d) {
		WARN_PRINT("Unable to use parent world as world_2d");
		return;
	}

	if (is_inside_tree()) {
		find_world_2d()->_remove_viewport(this);
		VisualServer::get_singleton()->viewport_remove_canvas(viewport,current_canvas);
	}

	if (p_world_2d.is_valid())
		world_2d=p_world_2d;
	else {
		WARN_PRINT("Invalid world");
		world_2d=Ref<World2D>( memnew( World2D ));
	}

	_update_listener_2d();

	if (is_inside_tree()) {
		current_canvas=find_world_2d()->get_canvas();
		VisualServer::get_singleton()->viewport_attach_canvas(viewport,current_canvas);
		find_world_2d()->_register_viewport(this,Rect2());
	}
}

Ref<World2D> Viewport::find_world_2d() const{

	if (world_2d.is_valid())
		return world_2d;
	else if (parent)
		return parent->find_world_2d();
	else
		return Ref<World2D>();
}

void Viewport::_propagate_enter_world(Node *p_node) {


	if (p_node!=this) {

		if (!p_node->is_inside_tree()) //may not have entered scene yet
			return;

		Spatial *s = p_node->cast_to<Spatial>();
		if (s) {

			s->notification(Spatial::NOTIFICATION_ENTER_WORLD);
		} else {
			Viewport *v = p_node->cast_to<Viewport>();
			if (v) {

				if (v->world.is_valid())
					return;
			}
		}
	}


	for(int i=0;i<p_node->get_child_count();i++) {

		_propagate_enter_world(p_node->get_child(i));
	}
}

void Viewport::_propagate_viewport_notification(Node* p_node,int p_what) {

	p_node->notification(p_what);
	for(int i=0;i<p_node->get_child_count();i++) {
		Node *c = p_node->get_child(i);
		if (c->cast_to<Viewport>())
			continue;
		_propagate_viewport_notification(c,p_what);
	}
}

void Viewport::_propagate_exit_world(Node *p_node) {

	if (p_node!=this) {

		if (!p_node->is_inside_tree()) //may have exited scene already
			return;

		Spatial *s = p_node->cast_to<Spatial>();
		if (s) {

			s->notification(Spatial::NOTIFICATION_EXIT_WORLD, true);
		} else {
			Viewport *v = p_node->cast_to<Viewport>();
			if (v) {

				if (v->world.is_valid())
					return;
			}
		}
	}


	for(int i=0;i<p_node->get_child_count();i++) {

		_propagate_exit_world(p_node->get_child(i));
	}

}


void Viewport::set_world(const Ref<World>& p_world) {

	if (world==p_world)
		return;

	if (is_inside_tree())
		_propagate_exit_world(this);

#ifndef _3D_DISABLED
	if (find_world().is_valid() && camera)
		camera->notification(Camera::NOTIFICATION_LOST_CURRENT);
#endif

	world=p_world;

	if (is_inside_tree())
		_propagate_enter_world(this);

#ifndef _3D_DISABLED
	if (find_world().is_valid() && camera)
		camera->notification(Camera::NOTIFICATION_BECAME_CURRENT);
#endif

	//propagate exit

	if (is_inside_tree()) {
		VisualServer::get_singleton()->viewport_set_scenario(viewport,find_world()->get_scenario());
	}

	_update_listener();

}

Ref<World> Viewport::get_world() const{

	return world;
}

Ref<World2D> Viewport::get_world_2d() const{

	return world_2d;
}

Ref<World> Viewport::find_world() const{

	if (own_world.is_valid())
		return own_world;
	else if (world.is_valid())
		return world;
	else if (parent)
		return parent->find_world();
	else
		return Ref<World>();
}

Listener* Viewport::get_listener() const {

	return listener;
}

Camera* Viewport::get_camera() const {

	return camera;
}


Matrix32 Viewport::get_final_transform() const {

	return stretch_transform * global_canvas_transform;
}

void Viewport::set_size_override(bool p_enable, const Size2& p_size, const Vector2 &p_margin) {

	if (size_override==p_enable && p_size==size_override_size)
		return;

	size_override=p_enable;
	if (p_size.x>=0 || p_size.y>=0) {
		size_override_size=p_size;
	}
	size_override_margin=p_margin;
	_update_rect();
	_update_stretch_transform();


}

Size2 Viewport::get_size_override() const {

	return size_override_size;
}
bool Viewport::is_size_override_enabled() const {

	return size_override;
}
void Viewport::set_size_override_stretch(bool p_enable) {

	if (p_enable==size_override_stretch)
		return;

	size_override_stretch=p_enable;
	if (size_override) {
		_update_rect();
	}


	_update_stretch_transform();
}


bool Viewport::is_size_override_stretch_enabled() const {

	return size_override_stretch;
}

void Viewport::set_as_render_target(bool p_enable){

	if (render_target==p_enable)
		return;

	render_target=p_enable;

	VS::get_singleton()->viewport_set_as_render_target(viewport,p_enable);
	if (is_inside_tree()) {

		if (p_enable)
			_vp_exit_tree();
		else
			_vp_enter_tree();
	}

	if (p_enable) {

		render_target_texture_rid = VS::get_singleton()->viewport_get_render_target_texture(viewport);
	} else {

		render_target_texture_rid=RID();
	}

	render_target_texture->set_flags(render_target_texture->flags);
	render_target_texture->emit_changed();

	update_configuration_warning();
}

bool Viewport::is_set_as_render_target() const{

	return render_target;
}
void Viewport::set_render_target_update_mode(RenderTargetUpdateMode p_mode){

	render_target_update_mode=p_mode;
	VS::get_singleton()->viewport_set_render_target_update_mode(viewport,VS::RenderTargetUpdateMode(p_mode));

}
Viewport::RenderTargetUpdateMode Viewport::get_render_target_update_mode() const{

	return render_target_update_mode;
}
//RID get_render_target_texture() const;

void Viewport::queue_screen_capture(){

	VS::get_singleton()->viewport_queue_screen_capture(viewport);
}
Image Viewport::get_screen_capture() const {

	return VS::get_singleton()->viewport_get_screen_capture(viewport);
}

Ref<RenderTargetTexture> Viewport::get_render_target_texture() const {

	return render_target_texture;
}

void Viewport::set_render_target_vflip(bool p_enable) {

	render_target_vflip=p_enable;
	VisualServer::get_singleton()->viewport_set_render_target_vflip(viewport,p_enable);
}

bool Viewport::get_render_target_vflip() const{

	return render_target_vflip;
}

void Viewport::set_render_target_clear_on_new_frame(bool p_enable) {

	render_target_clear_on_new_frame=p_enable;
	VisualServer::get_singleton()->viewport_set_render_target_clear_on_new_frame(viewport,p_enable);
}

bool Viewport::get_render_target_clear_on_new_frame() const{

	return render_target_clear_on_new_frame;
}

void Viewport::render_target_clear() {

	//render_target_clear=true;
	VisualServer::get_singleton()->viewport_render_target_clear(viewport);
}

void Viewport::set_render_target_filter(bool p_enable) {

	if(!render_target)
		return;

	render_target_texture->set_flags(p_enable?int(Texture::FLAG_FILTER):int(0));

}

bool Viewport::get_render_target_filter() const{

	return (render_target_texture->get_flags()&Texture::FLAG_FILTER)!=0;
}

void Viewport::set_render_target_gen_mipmaps(bool p_enable) {

	//render_target_texture->set_flags(p_enable?int(Texture::FLAG_FILTER):int(0));
	render_target_gen_mipmaps=p_enable;

}

bool Viewport::get_render_target_gen_mipmaps() const{

	//return (render_target_texture->get_flags()&Texture::FLAG_FILTER)!=0;
	return render_target_gen_mipmaps;
}


Matrix32 Viewport::_get_input_pre_xform() const {

	Matrix32 pre_xf;
	if (render_target) {

		if (to_screen_rect!=Rect2()) {

			pre_xf.elements[2]=-to_screen_rect.pos;
			pre_xf.scale(rect.size/to_screen_rect.size);
		}
	} else {

		pre_xf.elements[2]=-rect.pos;
	}

	return pre_xf;
}

Vector2 Viewport::_get_window_offset() const {

	if (parent_control) {
		return (parent_control->get_viewport()->get_final_transform() * parent_control->get_global_transform_with_canvas()).get_origin();
	}

	return Vector2();
}

void Viewport::_make_input_local(InputEvent& ev) {


	switch(ev.type) {

		case InputEvent::MOUSE_BUTTON: {

			Vector2 vp_ofs = _get_window_offset();

			Matrix32 ai = get_final_transform().affine_inverse() * _get_input_pre_xform();
			Vector2 g = ai.xform(Vector2(ev.mouse_button.global_x,ev.mouse_button.global_y));
			Vector2 l = ai.xform(Vector2(ev.mouse_button.x,ev.mouse_button.y)-vp_ofs);


			ev.mouse_button.x=l.x;
			ev.mouse_button.y=l.y;
			ev.mouse_button.global_x=g.x;
			ev.mouse_button.global_y=g.y;

		} break;
		case InputEvent::MOUSE_MOTION: {

			Vector2 vp_ofs = _get_window_offset();

			Matrix32 ai = get_final_transform().affine_inverse() * _get_input_pre_xform();
			Vector2 g = ai.xform(Vector2(ev.mouse_motion.global_x,ev.mouse_motion.global_y));
			Vector2 l = ai.xform(Vector2(ev.mouse_motion.x,ev.mouse_motion.y)-vp_ofs);
			Vector2 r = ai.basis_xform(Vector2(ev.mouse_motion.relative_x,ev.mouse_motion.relative_y));
			Vector2 s = ai.basis_xform(Vector2(ev.mouse_motion.speed_x,ev.mouse_motion.speed_y));


			ev.mouse_motion.x=l.x;
			ev.mouse_motion.y=l.y;
			ev.mouse_motion.global_x=g.x;
			ev.mouse_motion.global_y=g.y;
			ev.mouse_motion.relative_x=r.x;
			ev.mouse_motion.relative_y=r.y;
			ev.mouse_motion.speed_x=s.x;
			ev.mouse_motion.speed_y=s.y;

		} break;
		case InputEvent::SCREEN_TOUCH: {

			Vector2 vp_ofs = _get_window_offset();

			Matrix32 ai = get_final_transform().affine_inverse() * _get_input_pre_xform();
			Vector2 t = ai.xform(Vector2(ev.screen_touch.x,ev.screen_touch.y)-vp_ofs);


			ev.screen_touch.x=t.x;
			ev.screen_touch.y=t.y;

		} break;
		case InputEvent::SCREEN_DRAG: {

			Vector2 vp_ofs = _get_window_offset();

			Matrix32 ai = get_final_transform().affine_inverse() * _get_input_pre_xform();
			Vector2 t = ai.xform(Vector2(ev.screen_drag.x,ev.screen_drag.y)-vp_ofs);
			Vector2 r = ai.basis_xform(Vector2(ev.screen_drag.relative_x,ev.screen_drag.relative_y));
			Vector2 s = ai.basis_xform(Vector2(ev.screen_drag.speed_x,ev.screen_drag.speed_y));
			ev.screen_drag.x=t.x;
			ev.screen_drag.y=t.y;
			ev.screen_drag.relative_x=r.x;
			ev.screen_drag.relative_y=r.y;
			ev.screen_drag.speed_x=s.x;
			ev.screen_drag.speed_y=s.y;
		} break;
	}


}

void Viewport::_vp_input_text(const String& p_text) {

	if (gui.key_focus) {
		gui.key_focus->call("set_text",p_text);
	}
}

void Viewport::_vp_input(const InputEvent& p_ev) {

	if (disable_input)
		return;

#ifdef TOOLS_ENABLED
	if (get_tree()->is_editor_hint() && get_tree()->get_edited_scene_root()->is_a_parent_of(this)) {
		return;
	}
#endif

	if (parent_control && !parent_control->is_visible())
		return;

	if (render_target && to_screen_rect==Rect2())
		return; //if render target, can't get input events

	//this one handles system input, p_ev are in system coordinates
	//they are converted to viewport coordinates


	InputEvent ev = p_ev;
	_make_input_local(ev);
	input(ev);

}

void Viewport::_vp_unhandled_input(const InputEvent& p_ev) {

	if (disable_input)
		return;
#ifdef TOOLS_ENABLED
	if (get_tree()->is_editor_hint() && get_tree()->get_edited_scene_root()->is_a_parent_of(this)) {
		return;
	}
#endif

	if (parent_control && !parent_control->is_visible())
		return;

	if (render_target && to_screen_rect==Rect2())
		return; //if render target, can't get input events

	//this one handles system input, p_ev are in system coordinates
	//they are converted to viewport coordinates

	InputEvent ev = p_ev;
	_make_input_local(ev);
	unhandled_input(ev);

}

Vector2 Viewport::get_mouse_pos() const {

	return (get_final_transform().affine_inverse() * _get_input_pre_xform()).xform(Input::get_singleton()->get_mouse_pos() - _get_window_offset());
}

void Viewport::warp_mouse(const Vector2& p_pos) {

	Vector2 gpos = (get_final_transform().affine_inverse() * _get_input_pre_xform()).affine_inverse().xform(p_pos);
	Input::get_singleton()->warp_mouse_pos(gpos);
}



void Viewport::_gui_sort_subwindows() {

	if (!gui.subwindow_order_dirty)
		return;


	gui.modal_stack.sort_custom<Control::CComparator>();
	gui.subwindows.sort_custom<Control::CComparator>();

	gui.subwindow_order_dirty=false;
}

void Viewport::_gui_sort_modal_stack() {

	gui.modal_stack.sort_custom<Control::CComparator>();
}


void Viewport::_gui_sort_roots() {

	if (!gui.roots_order_dirty)
		return;

	gui.roots.sort_custom<Control::CComparator>();

	gui.roots_order_dirty=false;
}


void Viewport::_gui_cancel_tooltip() {

	gui.tooltip=NULL;
	gui.tooltip_timer=-1;
	if (gui.tooltip_popup) {
		gui.tooltip_popup->queue_delete();
		gui.tooltip_popup=NULL;
	}

}

void Viewport::_gui_show_tooltip() {

	if (!gui.tooltip) {
		return;
	}

	String tooltip = gui.tooltip->get_tooltip( gui.tooltip->get_global_transform().xform_inv(gui.tooltip_pos) );
	if (tooltip.length()==0)
		return; // bye

	if (gui.tooltip_popup) {
		memdelete(gui.tooltip_popup);
		gui.tooltip_popup=NULL;
	}

	Control *rp = gui.tooltip->get_root_parent_control();
	if (!rp)
		return;


	gui.tooltip_popup = memnew( TooltipPanel );

	rp->add_child(gui.tooltip_popup);
	gui.tooltip_popup->force_parent_owned();
	gui.tooltip_label = memnew( TooltipLabel );
	gui.tooltip_popup->add_child(gui.tooltip_label);
	gui.tooltip_popup->set_as_toplevel(true);
	gui.tooltip_popup->hide();

	Ref<StyleBox> ttp = gui.tooltip_label->get_stylebox("panel","TooltipPanel");

	gui.tooltip_label->set_anchor_and_margin(MARGIN_LEFT,Control::ANCHOR_BEGIN,ttp->get_margin(MARGIN_LEFT));
	gui.tooltip_label->set_anchor_and_margin(MARGIN_TOP,Control::ANCHOR_BEGIN,ttp->get_margin(MARGIN_TOP));
	gui.tooltip_label->set_anchor_and_margin(MARGIN_RIGHT,Control::ANCHOR_END,ttp->get_margin(MARGIN_RIGHT));
	gui.tooltip_label->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_END,ttp->get_margin(MARGIN_BOTTOM));
	gui.tooltip_label->set_text(tooltip);
	Rect2 r(gui.tooltip_pos+Point2(10,10),gui.tooltip_label->get_combined_minimum_size()+ttp->get_minimum_size());
	Rect2 vr = gui.tooltip_label->get_viewport_rect();
	if (r.size.x+r.pos.x>vr.size.x)
		r.pos.x=vr.size.x-r.size.x;
	else if (r.pos.x<0)
		r.pos.x=0;

	if (r.size.y+r.pos.y>vr.size.y)
		r.pos.y=vr.size.y-r.size.y;
	else if (r.pos.y<0)
		r.pos.y=0;

	gui.tooltip_popup->set_global_pos(r.pos);
	gui.tooltip_popup->set_size(r.size);

	gui.tooltip_popup->raise();
	gui.tooltip_popup->show();
}


void Viewport::_gui_call_input(Control *p_control,const InputEvent& p_input) {

//	_block();


	InputEvent ev = p_input;

	//mouse wheel events can't be stopped
	bool cant_stop_me_now = (ev.type==InputEvent::MOUSE_BUTTON &&
				 (ev.mouse_button.button_index==BUTTON_WHEEL_DOWN ||
				  ev.mouse_button.button_index==BUTTON_WHEEL_UP ||
				  ev.mouse_button.button_index==BUTTON_WHEEL_LEFT ||
				  ev.mouse_button.button_index==BUTTON_WHEEL_RIGHT ) );

	CanvasItem *ci=p_control;
	while(ci) {

		Control *control = ci->cast_to<Control>();
		if (control) {
			control->call_multilevel(SceneStringNames::get_singleton()->_input_event,ev);
			if (gui.key_event_accepted)
				break;
			if (!control->is_inside_tree())
				break;
			control->emit_signal(SceneStringNames::get_singleton()->input_event,ev);
			if (!control->is_inside_tree() || control->is_set_as_toplevel())
				break;
			if (gui.key_event_accepted)
				break;
			if (!cant_stop_me_now && control->data.stop_mouse && (ev.type==InputEvent::MOUSE_BUTTON || ev.type==InputEvent::MOUSE_MOTION))
				break;
		}

		if (ci->is_set_as_toplevel())
			break;

		ev=ev.xform_by(ci->get_transform()); //transform event upwards
		ci=ci->get_parent_item();
	}

	//_unblock();

}

Control* Viewport::_gui_find_control(const Point2& p_global)  {

	_gui_sort_subwindows();

	for (List<Control*>::Element *E=gui.subwindows.back();E;E=E->prev()) {

		Control *sw = E->get();
		if (!sw->is_visible())
			continue;

		Matrix32 xform;
		CanvasItem *pci = sw->get_parent_item();
		if (pci)
			xform=pci->get_global_transform_with_canvas();
		else
			xform=sw->get_canvas_transform();

		Control *ret = _gui_find_control_at_pos(sw,p_global,xform,gui.focus_inv_xform);
		if (ret)
			return ret;
	}

	_gui_sort_roots();

	for (List<Control*>::Element *E=gui.roots.back();E;E=E->prev()) {

		Control *sw = E->get();
		if (!sw->is_visible())
			continue;

		Matrix32 xform;
		CanvasItem *pci = sw->get_parent_item();
		if (pci)
			xform=pci->get_global_transform_with_canvas();
		else
			xform=sw->get_canvas_transform();


		Control *ret = _gui_find_control_at_pos(sw,p_global,xform,gui.focus_inv_xform);
		if (ret)
			return ret;
	}

	return NULL;

}


Control* Viewport::_gui_find_control_at_pos(CanvasItem* p_node,const Point2& p_global,const Matrix32& p_xform,Matrix32& r_inv_xform)  {

	if (p_node->cast_to<Viewport>())
		return NULL;

	Control *c=p_node->cast_to<Control>();

	if (c) {
	//	print_line("at "+String(c->get_path())+" POS "+c->get_pos()+" bt "+p_xform);
	}

	//subwindows first!!

	if (p_node->is_hidden()) {
		//return _find_next_visible_control_at_pos(p_node,p_global,r_inv_xform);
		return NULL; //canvas item hidden, discard
	}

	Matrix32 matrix = p_xform * p_node->get_transform();
	// matrix.basis_determinant() == 0.0f implies that node does not exist on scene
	if(matrix.basis_determinant() == 0.0f)
		return NULL;

	if (!c || !c->clips_input() || c->has_point(matrix.affine_inverse().xform(p_global))) {

		for(int i=p_node->get_child_count()-1;i>=0;i--) {

			if (p_node==gui.tooltip_popup)
				continue;

			CanvasItem *ci = p_node->get_child(i)->cast_to<CanvasItem>();
			if (!ci || ci->is_set_as_toplevel())
				continue;

			Control *ret=_gui_find_control_at_pos(ci,p_global,matrix,r_inv_xform);;
			if (ret)
				return ret;
		}
	}

	if (!c)
		return NULL;

	matrix.affine_invert();

	//conditions for considering this as a valid control for return
	if (!c->data.ignore_mouse && c->has_point(matrix.xform(p_global)) && (!gui.drag_preview || (c!=gui.drag_preview && !gui.drag_preview->is_a_parent_of(c)))) {
		r_inv_xform=matrix;
		return c;
	} else
		return NULL;
}

void Viewport::_gui_input_event(InputEvent p_event) {



	if (p_event.ID==gui.cancelled_input_ID) {
		return;
	}
	//?
//	if (!is_visible()) {
//		return; //simple and plain
//	}


	switch(p_event.type) {

		case InputEvent::MOUSE_BUTTON: {


			gui.key_event_accepted=false;

			Point2 mpos=Point2(p_event.mouse_button.x,p_event.mouse_button.y);
			if (p_event.mouse_button.pressed) {



				Size2 pos = mpos;
				if (gui.mouse_focus && p_event.mouse_button.button_index!=gui.mouse_focus_button) {

					//do not steal mouse focus and stuff

				} else {


					_gui_sort_modal_stack();
					while (!gui.modal_stack.empty()) {

						Control *top = gui.modal_stack.back()->get();
						Vector2 pos = top->get_global_transform_with_canvas().affine_inverse().xform(mpos);
						if (!top->has_point(pos)) {

							if (top->data.modal_exclusive || top->data.modal_frame==OS::get_singleton()->get_frames_drawn()) {
								//cancel event, sorry, modal exclusive EATS UP ALL
								//alternative, you can't pop out a window the same frame it was made modal (fixes many issues)
								get_tree()->set_input_as_handled();
								return; // no one gets the event if exclusive NO ONE
							}

							top->notification(Control::NOTIFICATION_MODAL_CLOSE);
							top->_modal_stack_remove();
							top->hide();
						} else {
							break;
						}
					}



					//Matrix32 parent_xform;

					//if (data.parent_canvas_item)
					//	parent_xform=data.parent_canvas_item->get_global_transform();



					gui.mouse_focus = _gui_find_control(pos);
					//print_line("has mf "+itos(gui.mouse_focus!=NULL));
					gui.mouse_focus_button=p_event.mouse_button.button_index;

					if (!gui.mouse_focus) {
						break;
					}

					if (p_event.mouse_button.button_index==BUTTON_LEFT) {
						gui.drag_accum=Vector2();
						gui.drag_attempted=false;
					}


				}


				p_event.mouse_button.global_x = pos.x;
				p_event.mouse_button.global_y = pos.y;

				pos = gui.focus_inv_xform.xform(pos);
				p_event.mouse_button.x = pos.x;
				p_event.mouse_button.y = pos.y;

#ifdef DEBUG_ENABLED
				if (ScriptDebugger::get_singleton()) {

					Array arr;
					arr.push_back(gui.mouse_focus->get_path());
					arr.push_back(gui.mouse_focus->get_type());
					ScriptDebugger::get_singleton()->send_message("click_ctrl",arr);
				}

				/*if (bool(GLOBAL_DEF("debug/print_clicked_control",false))) {

						print_line(String(gui.mouse_focus->get_path())+" - "+pos);
					}*/
#endif

				if (gui.mouse_focus->get_focus_mode()!=Control::FOCUS_NONE && gui.mouse_focus!=gui.key_focus && p_event.mouse_button.button_index==BUTTON_LEFT) {
					// also get keyboard focus
					gui.mouse_focus->grab_focus();
				}


				if (gui.mouse_focus->can_process()) {
					_gui_call_input(gui.mouse_focus,p_event);
				}

				get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_cancel_input_ID",p_event.ID);
				get_tree()->set_input_as_handled();


				if (gui.drag_data.get_type()!=Variant::NIL && p_event.mouse_button.button_index==BUTTON_LEFT) {

					//alternate drop use (when using force_drag(), as proposed by #5342
					if (gui.mouse_focus && gui.mouse_focus->can_drop_data(pos,gui.drag_data)) {
						gui.mouse_focus->drop_data(pos,gui.drag_data);
					}

					gui.drag_data=Variant();

					if (gui.drag_preview) {
						memdelete( gui.drag_preview );
						gui.drag_preview=NULL;
					}
					_propagate_viewport_notification(this,NOTIFICATION_DRAG_END);
					//change mouse accordingly
				}



				_gui_cancel_tooltip();
				//gui.tooltip_popup->hide();

			} else {



				if (gui.drag_data.get_type()!=Variant::NIL && p_event.mouse_button.button_index==BUTTON_LEFT) {

					if (gui.mouse_over) {
						Size2 pos = mpos;
						pos = gui.focus_inv_xform.xform(pos);
						if (gui.mouse_over->can_drop_data(pos,gui.drag_data)) {
							gui.mouse_over->drop_data(pos,gui.drag_data);
						}
					}

					if (gui.drag_preview && p_event.mouse_button.button_index==BUTTON_LEFT) {
						memdelete( gui.drag_preview );
						gui.drag_preview=NULL;
					}

					gui.drag_data=Variant();
					_propagate_viewport_notification(this,NOTIFICATION_DRAG_END);
					//change mouse accordingly
				}

				if (!gui.mouse_focus) {
					//release event is only sent if a mouse focus (previously pressed button) exists
					break;
				}

				Size2 pos = mpos;
				p_event.mouse_button.global_x = pos.x;
				p_event.mouse_button.global_y = pos.y;
				pos = gui.focus_inv_xform.xform(pos);
				p_event.mouse_button.x = pos.x;
				p_event.mouse_button.y = pos.y;

				if (gui.mouse_focus->can_process()) {
					_gui_call_input(gui.mouse_focus,p_event);
				}

				if (p_event.mouse_button.button_index==gui.mouse_focus_button) {
					gui.mouse_focus=NULL;
					gui.mouse_focus_button=-1;
				}

				/*if (gui.drag_data.get_type()!=Variant::NIL && p_event.mouse_button.button_index==BUTTON_LEFT) {
					_propagate_viewport_notification(this,NOTIFICATION_DRAG_END);
					gui.drag_data=Variant(); //always clear
				}*/


				get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_cancel_input_ID",p_event.ID);
				get_tree()->set_input_as_handled();

			}
		} break;
		case InputEvent::MOUSE_MOTION: {

			gui.key_event_accepted=false;
			Point2 mpos=Point2(p_event.mouse_motion.x,p_event.mouse_motion.y);

			gui.last_mouse_pos=mpos;

			Control *over = NULL;


			// D&D
			if (!gui.drag_attempted && gui.mouse_focus && p_event.mouse_motion.button_mask&BUTTON_MASK_LEFT) {

				gui.drag_accum+=Point2(p_event.mouse_motion.relative_x,p_event.mouse_motion.relative_y);;
				float len = gui.drag_accum.length();
				if (len>10) {
					gui.drag_data=gui.mouse_focus->get_drag_data(gui.focus_inv_xform.xform(mpos)-gui.drag_accum);
					if (gui.drag_data.get_type()!=Variant::NIL) {

						gui.mouse_focus=NULL;
					}
					gui.drag_attempted=true;
					if (gui.drag_data.get_type()!=Variant::NIL) {

						_propagate_viewport_notification(this,NOTIFICATION_DRAG_BEGIN);
					}
				}
			}


			if (gui.mouse_focus) {
				over=gui.mouse_focus;
				//recompute focus_inv_xform again here

			} else {

				over = _gui_find_control(mpos);
			}



			if (gui.drag_data.get_type()==Variant::NIL && over && !gui.modal_stack.empty()) {

				Control *top = gui.modal_stack.back()->get();
				if (over!=top && !top->is_a_parent_of(over)) {

					break; // don't send motion event to anything below modal stack top
				}
			}


			if (over!=gui.mouse_over) {

				if (gui.mouse_over)
					gui.mouse_over->notification(Control::NOTIFICATION_MOUSE_EXIT);

				_gui_cancel_tooltip();

				if (over)
					over->notification(Control::NOTIFICATION_MOUSE_ENTER);

			}

			gui.mouse_over=over;

			if (gui.drag_preview) {
				gui.drag_preview->set_pos(mpos);
			}

			if (!over) {
				OS::get_singleton()->set_cursor_shape(OS::CURSOR_ARROW);
				break;
			}


			Matrix32 localizer = over->get_global_transform_with_canvas().affine_inverse();
			Size2 pos = localizer.xform(mpos);
			Vector2 speed = localizer.basis_xform(Point2(p_event.mouse_motion.speed_x,p_event.mouse_motion.speed_y));
			Vector2 rel = localizer.basis_xform(Point2(p_event.mouse_motion.relative_x,p_event.mouse_motion.relative_y));


			p_event.mouse_motion.global_x = mpos.x;
			p_event.mouse_motion.global_y = mpos.y;
			p_event.mouse_motion.speed_x=speed.x;
			p_event.mouse_motion.speed_y=speed.y;
			p_event.mouse_motion.relative_x=rel.x;
			p_event.mouse_motion.relative_y=rel.y;

			if (p_event.mouse_motion.button_mask==0) {
				//nothing pressed

				bool can_tooltip=true;

				if (!gui.modal_stack.empty()) {
					if (gui.modal_stack.back()->get()!=over && !gui.modal_stack.back()->get()->is_a_parent_of(over))
						can_tooltip=false;

				}

				bool is_tooltip_shown = false;

				if (gui.tooltip_popup) {
					if (can_tooltip) {
						String tooltip = over->get_tooltip(gui.tooltip->get_global_transform().xform_inv(mpos));

						if (tooltip.length() == 0)
							_gui_cancel_tooltip();
						else if (tooltip == gui.tooltip_label->get_text())
							is_tooltip_shown = true;
					}
					else
						_gui_cancel_tooltip();
				}

				if (can_tooltip && !is_tooltip_shown) {

					gui.tooltip=over;
					gui.tooltip_pos=mpos;//(parent_xform * get_transform()).affine_inverse().xform(pos);
					gui.tooltip_timer=gui.tooltip_delay;

				}
			}


			//pos = gui.focus_inv_xform.xform(pos);


			p_event.mouse_motion.x = pos.x;
			p_event.mouse_motion.y = pos.y;


			Control::CursorShape cursor_shape = over->get_cursor_shape(pos);
			OS::get_singleton()->set_cursor_shape( (OS::CursorShape)cursor_shape );


			if (over->can_process()) {
				_gui_call_input(over,p_event);
			}



			get_tree()->set_input_as_handled();


			if (gui.drag_data.get_type()!=Variant::NIL && p_event.mouse_motion.button_mask&BUTTON_MASK_LEFT) {


				bool can_drop = over->can_drop_data(pos,gui.drag_data);

				if (!can_drop) {
					OS::get_singleton()->set_cursor_shape( OS::CURSOR_FORBIDDEN );
				} else {
					OS::get_singleton()->set_cursor_shape( OS::CURSOR_CAN_DROP );

				}
				//change mouse accordingly i guess
			}

		} break;
		case InputEvent::ACTION:
		case InputEvent::JOYSTICK_BUTTON:
		case InputEvent::JOYSTICK_MOTION:
		case InputEvent::KEY: {


			if (gui.key_focus && !gui.key_focus->is_visible()) {
				gui.key_focus->release_focus();
			}

			if (gui.key_focus) {				

				gui.key_event_accepted=false;
				if (gui.key_focus->can_process()) {
					gui.key_focus->call_multilevel("_input_event",p_event);
					if (gui.key_focus) //maybe lost it
						gui.key_focus->emit_signal(SceneStringNames::get_singleton()->input_event,p_event);
				}


				if (gui.key_event_accepted) {

					get_tree()->set_input_as_handled();
					break;
				}
			}


			if (p_event.is_pressed() && p_event.is_action("ui_cancel") && !gui.modal_stack.empty()) {

				_gui_sort_modal_stack();
				Control *top = gui.modal_stack.back()->get();
				if (!top->data.modal_exclusive) {

					top->notification(Control::NOTIFICATION_MODAL_CLOSE);
					top->_modal_stack_remove();
					top->hide();
				}
			}


			Control * from = gui.key_focus ? gui.key_focus : NULL; //hmm

			//keyboard focus
			//if (from && p_event.key.pressed && !p_event.key.mod.alt && !p_event.key.mod.meta && !p_event.key.mod.command) {

			if (from && p_event.is_pressed()) {
				Control * next=NULL;

				if (p_event.is_action("ui_focus_next")) {

					next = from->find_next_valid_focus();
				}

				if (p_event.is_action("ui_focus_prev")) {

					next = from->find_prev_valid_focus();
				}

				if (p_event.is_action("ui_up")) {

					next = from->_get_focus_neighbour(MARGIN_TOP);
				}

				if (p_event.is_action("ui_left")) {

					next = from->_get_focus_neighbour(MARGIN_LEFT);
				}

				if (p_event.is_action("ui_right")) {

					next = from->_get_focus_neighbour(MARGIN_RIGHT);
				}

				if (p_event.is_action("ui_down")) {

					next = from->_get_focus_neighbour(MARGIN_BOTTOM);
				}


				if (next) {
					next->grab_focus();
					get_tree()->set_input_as_handled();
				}
			}

		} break;
	}
}


List<Control*>::Element* Viewport::_gui_add_root_control(Control* p_control) {

	gui.roots_order_dirty=true;
	return gui.roots.push_back(p_control);
}

List<Control*>::Element* Viewport::_gui_add_subwindow_control(Control* p_control) {

	gui.subwindow_order_dirty=true;
	return gui.subwindows.push_back(p_control);

}

void Viewport::_gui_set_subwindow_order_dirty() {
	gui.subwindow_order_dirty=true;
}

void Viewport::_gui_set_root_order_dirty() {
	gui.roots_order_dirty=true;
}

void Viewport::_gui_remove_modal_control(List<Control*>::Element *MI) {

	gui.modal_stack.erase(MI);
}

void Viewport::_gui_remove_from_modal_stack(List<Control*>::Element *MI,ObjectID p_prev_focus_owner) {

	//transfer the focus stack to the next

	List<Control*>::Element *next = MI->next();

	gui.modal_stack.erase(MI);
	MI=NULL;

	if (p_prev_focus_owner) {

		// for previous window in stack, pass the focus so it feels more
		// natural

		if (!next) { //top of stack

			Object *pfo = ObjectDB::get_instance(p_prev_focus_owner);
			Control *pfoc = pfo->cast_to<Control>();
			if (!pfoc)
				return;

			if (!pfoc->is_inside_tree() || !pfoc->is_visible())
				return;
			pfoc->grab_focus();
		} else {

			next->get()->_modal_set_prev_focus_owner(p_prev_focus_owner);
		}
	}
}

void Viewport::_gui_force_drag(Control *p_base, const Variant& p_data, Control *p_control) {

	ERR_EXPLAIN("Drag data must be a value");
	ERR_FAIL_COND(p_data.get_type()==Variant::NIL);

	gui.drag_data=p_data;
	gui.mouse_focus=NULL;

	if (p_control) {
		_gui_set_drag_preview(p_base,p_control);
	}
}

void Viewport::_gui_set_drag_preview(Control *p_base, Control *p_control) {

	ERR_FAIL_NULL(p_control);
	ERR_FAIL_COND( !((Object*)p_control)->cast_to<Control>());
	ERR_FAIL_COND(p_control->is_inside_tree());
	ERR_FAIL_COND(p_control->get_parent()!=NULL);

	if (gui.drag_preview) {
		memdelete(gui.drag_preview);
	}
	p_control->set_as_toplevel(true);
	p_control->set_pos(gui.last_mouse_pos);
	p_base->get_root_parent_control()->add_child(p_control); //add as child of viewport
	p_control->raise();
	if (gui.drag_preview) {
		memdelete( gui.drag_preview );
	}
	gui.drag_preview=p_control;
}


void Viewport::_gui_remove_root_control(List<Control*>::Element *RI) {

	gui.roots.erase(RI);
}

void Viewport::_gui_remove_subwindow_control(List<Control*>::Element* SI){

	gui.subwindows.erase(SI);
}

void Viewport::_gui_unfocus_control(Control *p_control) {

	if (gui.key_focus==p_control) {
		gui.key_focus->release_focus();
	}
}

void Viewport::_gui_hid_control(Control *p_control) {

	if (gui.mouse_focus == p_control) {
		gui.mouse_focus=NULL;
	}

	/* ???
	if (data.window==p_control) {
		window->drag_data=Variant();
		if (window->drag_preview) {
			memdelete( window->drag_preview);
			window->drag_preview=NULL;
		}
	}
	*/

	if (gui.key_focus == p_control)
		gui.key_focus=NULL;
	if (gui.mouse_over == p_control)
		gui.mouse_over=NULL;
	if (gui.tooltip == p_control)
		gui.tooltip=NULL;
	if (gui.tooltip == p_control) {
		gui.tooltip=NULL;
		_gui_cancel_tooltip();
	}

}

void Viewport::_gui_remove_control(Control *p_control) {


	if (gui.mouse_focus == p_control)
		gui.mouse_focus=NULL;
	if (gui.key_focus == p_control)
		gui.key_focus=NULL;
	if (gui.mouse_over == p_control)
		gui.mouse_over=NULL;
	if (gui.tooltip == p_control)
		gui.tooltip=NULL;
	if (gui.tooltip_popup == p_control) {
		_gui_cancel_tooltip();
	}


}

void Viewport::_gui_remove_focus() {

	if (gui.key_focus) {
		Node *f=gui.key_focus;
		gui.key_focus=NULL;
		f->notification( Control::NOTIFICATION_FOCUS_EXIT,true );


	}
}

bool Viewport::_gui_is_modal_on_top(const Control* p_control) {

	return  (gui.modal_stack.size() && gui.modal_stack.back()->get()==p_control);

}

bool Viewport::_gui_control_has_focus(const Control* p_control) {

	return gui.key_focus==p_control;
}

void Viewport::_gui_control_grab_focus(Control* p_control) {


	//no need for change
	if (gui.key_focus && gui.key_focus==p_control)
		return;

	get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"_viewports","_gui_remove_focus");
	gui.key_focus=p_control;
	p_control->notification(Control::NOTIFICATION_FOCUS_ENTER);
	p_control->update();

}

void Viewport::_gui_accept_event() {

	gui.key_event_accepted=true;
	if (is_inside_tree())
		get_tree()->set_input_as_handled();
}


List<Control*>::Element* Viewport::_gui_show_modal(Control* p_control) {

	gui.modal_stack.push_back(p_control);
	if (gui.key_focus)
		p_control->_modal_set_prev_focus_owner(gui.key_focus->get_instance_ID());
	else
		p_control->_modal_set_prev_focus_owner(0);

	return gui.modal_stack.back();
}

Control *Viewport::_gui_get_focus_owner() {

	return gui.key_focus;
}

void Viewport::_gui_grab_click_focus(Control *p_control) {

	if (gui.mouse_focus) {


		if (gui.mouse_focus==p_control)
			return;
		InputEvent ie;
		ie.type=InputEvent::MOUSE_BUTTON;
		InputEventMouseButton &mb=ie.mouse_button;

		//send unclic

		Point2 click =gui.mouse_focus->get_global_transform_with_canvas().affine_inverse().xform(gui.last_mouse_pos);
		mb.x=click.x;
		mb.y=click.y;
		mb.button_index=gui.mouse_focus_button;
		mb.pressed=false;
		gui.mouse_focus->call_deferred("_input_event",ie);


		gui.mouse_focus=p_control;
		gui.focus_inv_xform=gui.mouse_focus->get_global_transform_with_canvas().affine_inverse();
		click =gui.mouse_focus->get_global_transform_with_canvas().affine_inverse().xform(gui.last_mouse_pos);
		mb.x=click.x;
		mb.y=click.y;
		mb.button_index=gui.mouse_focus_button;
		mb.pressed=true;
		gui.mouse_focus->call_deferred("_input_event",ie);

	}
}


///////////////////////////////


void Viewport::input(const InputEvent& p_event) {

	ERR_FAIL_COND(!is_inside_tree());


	get_tree()->_call_input_pause(input_group,"_input",p_event); //not a bug, must happen before GUI, order is _input -> gui input -> _unhandled input
	_gui_input_event(p_event);
	//get_tree()->call_group(SceneTree::GROUP_CALL_REVERSE|SceneTree::GROUP_CALL_REALTIME|SceneTree::GROUP_CALL_MULIILEVEL,gui_input_group,"_gui_input",p_event); //special one for GUI, as controls use their own process check
}

void Viewport::unhandled_input(const InputEvent& p_event) {

	ERR_FAIL_COND(!is_inside_tree());


	get_tree()->_call_input_pause(unhandled_input_group,"_unhandled_input",p_event);
	//call_group(GROUP_CALL_REVERSE|GROUP_CALL_REALTIME|GROUP_CALL_MULIILEVEL,"unhandled_input","_unhandled_input",ev);
	if (!get_tree()->input_handled && p_event.type==InputEvent::KEY) {
		get_tree()->_call_input_pause(unhandled_key_input_group,"_unhandled_key_input",p_event);
		//call_group(GROUP_CALL_REVERSE|GROUP_CALL_REALTIME|GROUP_CALL_MULIILEVEL,"unhandled_key_input","_unhandled_key_input",ev);
	}


	if (physics_object_picking && !get_tree()->input_handled) {

		if (p_event.type==InputEvent::MOUSE_BUTTON || p_event.type==InputEvent::MOUSE_MOTION || p_event.type==InputEvent::SCREEN_DRAG || p_event.type==InputEvent::SCREEN_TOUCH) {
			physics_picking_events.push_back(p_event);
		}
	}

}

void Viewport::set_use_own_world(bool p_world) {

	if (p_world==own_world.is_valid())
		return;


	if (is_inside_tree())
		_propagate_exit_world(this);

#ifndef _3D_DISABLED
	if (find_world().is_valid() && camera)
		camera->notification(Camera::NOTIFICATION_LOST_CURRENT);
#endif

	if (!p_world)
		own_world=Ref<World>();
	else
		own_world=Ref<World>( memnew( World ));

	if (is_inside_tree())
		_propagate_enter_world(this);

#ifndef _3D_DISABLED
	if (find_world().is_valid() && camera)
		camera->notification(Camera::NOTIFICATION_BECAME_CURRENT);
#endif

	//propagate exit

	if (is_inside_tree()) {
		VisualServer::get_singleton()->viewport_set_scenario(viewport,find_world()->get_scenario());
	}

	_update_listener();


}

bool Viewport::is_using_own_world() const {

	return own_world.is_valid();
}

void Viewport::set_render_target_to_screen_rect(const Rect2& p_rect) {

	to_screen_rect=p_rect;
	VisualServer::get_singleton()->viewport_set_render_target_to_screen_rect(viewport,to_screen_rect);
}

Rect2 Viewport::get_render_target_to_screen_rect() const{

	return to_screen_rect;
}

void Viewport::set_physics_object_picking(bool p_enable) {

	physics_object_picking=p_enable;
	set_fixed_process(physics_object_picking);
	if (!physics_object_picking)
		physics_picking_events.clear();


}


Vector2 Viewport::get_camera_coords(const Vector2 &p_viewport_coords) const {

	Matrix32 xf = get_final_transform();
	return xf.xform(p_viewport_coords);


}

Vector2 Viewport::get_camera_rect_size() const {

	return last_vp_rect.size;
}


bool Viewport::get_physics_object_picking() {


	return physics_object_picking;
}

bool Viewport::gui_has_modal_stack() const {

	return gui.modal_stack.size();
}

void Viewport::set_disable_input(bool p_disable) {
	disable_input=p_disable;
}

bool Viewport::is_input_disabled() const {

	return disable_input;
}

Variant Viewport::gui_get_drag_data() const {
	return gui.drag_data;
}

Control *Viewport::get_modal_stack_top() const {
	return gui.modal_stack.size()?gui.modal_stack.back()->get():NULL;
}

String Viewport::get_configuration_warning() const {

	if (get_parent() && !get_parent()->cast_to<Control>() && !render_target) {

		return TTR("This viewport is not set as render target. If you intend for it to display its contents directly to the screen, make it a child of a Control so it can obtain a size. Otherwise, make it a RenderTarget and assign its internal texture to some node for display.");
	}

	return String();
}

void Viewport::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("set_rect","rect"), &Viewport::set_rect);
	ObjectTypeDB::bind_method(_MD("get_rect"), &Viewport::get_rect);
	ObjectTypeDB::bind_method(_MD("set_world_2d","world_2d:World2D"), &Viewport::set_world_2d);
	ObjectTypeDB::bind_method(_MD("get_world_2d:World2D"), &Viewport::get_world_2d);
	ObjectTypeDB::bind_method(_MD("find_world_2d:World2D"), &Viewport::find_world_2d);
	ObjectTypeDB::bind_method(_MD("set_world","world:World"), &Viewport::set_world);
	ObjectTypeDB::bind_method(_MD("get_world:World"), &Viewport::get_world);
	ObjectTypeDB::bind_method(_MD("find_world:World"), &Viewport::find_world);

	ObjectTypeDB::bind_method(_MD("set_canvas_transform","xform"), &Viewport::set_canvas_transform);
	ObjectTypeDB::bind_method(_MD("get_canvas_transform"), &Viewport::get_canvas_transform);

	ObjectTypeDB::bind_method(_MD("set_global_canvas_transform","xform"), &Viewport::set_global_canvas_transform);
	ObjectTypeDB::bind_method(_MD("get_global_canvas_transform"), &Viewport::get_global_canvas_transform);
	ObjectTypeDB::bind_method(_MD("get_final_transform"), &Viewport::get_final_transform);

	ObjectTypeDB::bind_method(_MD("get_visible_rect"), &Viewport::get_visible_rect);
	ObjectTypeDB::bind_method(_MD("set_transparent_background","enable"), &Viewport::set_transparent_background);
	ObjectTypeDB::bind_method(_MD("has_transparent_background"), &Viewport::has_transparent_background);

	ObjectTypeDB::bind_method(_MD("_parent_visibility_changed"), &Viewport::_parent_visibility_changed);

	ObjectTypeDB::bind_method(_MD("_parent_resized"), &Viewport::_parent_resized);
	ObjectTypeDB::bind_method(_MD("_vp_input"), &Viewport::_vp_input);
	ObjectTypeDB::bind_method(_MD("_vp_input_text","text"), &Viewport::_vp_input_text);
	ObjectTypeDB::bind_method(_MD("_vp_unhandled_input"), &Viewport::_vp_unhandled_input);

	ObjectTypeDB::bind_method(_MD("set_size_override","enable","size","margin"), &Viewport::set_size_override,DEFVAL(Size2(-1,-1)),DEFVAL(Size2(0,0)));
	ObjectTypeDB::bind_method(_MD("get_size_override"), &Viewport::get_size_override);
	ObjectTypeDB::bind_method(_MD("is_size_override_enabled"), &Viewport::is_size_override_enabled);
	ObjectTypeDB::bind_method(_MD("set_size_override_stretch","enabled"), &Viewport::set_size_override_stretch);
	ObjectTypeDB::bind_method(_MD("is_size_override_stretch_enabled"), &Viewport::is_size_override_stretch_enabled);
	ObjectTypeDB::bind_method(_MD("queue_screen_capture"), &Viewport::queue_screen_capture);
	ObjectTypeDB::bind_method(_MD("get_screen_capture"), &Viewport::get_screen_capture);

	ObjectTypeDB::bind_method(_MD("set_as_render_target","enable"), &Viewport::set_as_render_target);
	ObjectTypeDB::bind_method(_MD("is_set_as_render_target"), &Viewport::is_set_as_render_target);

	ObjectTypeDB::bind_method(_MD("set_render_target_vflip","enable"), &Viewport::set_render_target_vflip);
	ObjectTypeDB::bind_method(_MD("get_render_target_vflip"), &Viewport::get_render_target_vflip);

	ObjectTypeDB::bind_method(_MD("set_render_target_clear_on_new_frame","enable"), &Viewport::set_render_target_clear_on_new_frame);
	ObjectTypeDB::bind_method(_MD("get_render_target_clear_on_new_frame"), &Viewport::get_render_target_clear_on_new_frame);

	ObjectTypeDB::bind_method(_MD("render_target_clear"), &Viewport::render_target_clear);

	ObjectTypeDB::bind_method(_MD("set_render_target_filter","enable"), &Viewport::set_render_target_filter);
	ObjectTypeDB::bind_method(_MD("get_render_target_filter"), &Viewport::get_render_target_filter);

	ObjectTypeDB::bind_method(_MD("set_render_target_gen_mipmaps","enable"), &Viewport::set_render_target_gen_mipmaps);
	ObjectTypeDB::bind_method(_MD("get_render_target_gen_mipmaps"), &Viewport::get_render_target_gen_mipmaps);

	ObjectTypeDB::bind_method(_MD("set_render_target_update_mode","mode"), &Viewport::set_render_target_update_mode);
	ObjectTypeDB::bind_method(_MD("get_render_target_update_mode"), &Viewport::get_render_target_update_mode);

	ObjectTypeDB::bind_method(_MD("get_render_target_texture:RenderTargetTexture"), &Viewport::get_render_target_texture);

	ObjectTypeDB::bind_method(_MD("set_physics_object_picking","enable"), &Viewport::set_physics_object_picking);
	ObjectTypeDB::bind_method(_MD("get_physics_object_picking"), &Viewport::get_physics_object_picking);

	ObjectTypeDB::bind_method(_MD("get_viewport"), &Viewport::get_viewport);
	ObjectTypeDB::bind_method(_MD("input","local_event"), &Viewport::input);
	ObjectTypeDB::bind_method(_MD("unhandled_input","local_event"), &Viewport::unhandled_input);

	ObjectTypeDB::bind_method(_MD("update_worlds"), &Viewport::update_worlds);

	ObjectTypeDB::bind_method(_MD("set_use_own_world","enable"), &Viewport::set_use_own_world);
	ObjectTypeDB::bind_method(_MD("is_using_own_world"), &Viewport::is_using_own_world);

	ObjectTypeDB::bind_method(_MD("get_camera:Camera"), &Viewport::get_camera);

	ObjectTypeDB::bind_method(_MD("set_as_audio_listener","enable"), &Viewport::set_as_audio_listener);
	ObjectTypeDB::bind_method(_MD("is_audio_listener","enable"), &Viewport::is_audio_listener);

	ObjectTypeDB::bind_method(_MD("set_as_audio_listener_2d","enable"), &Viewport::set_as_audio_listener_2d);
	ObjectTypeDB::bind_method(_MD("is_audio_listener_2d","enable"), &Viewport::is_audio_listener_2d);
	ObjectTypeDB::bind_method(_MD("set_render_target_to_screen_rect","rect"), &Viewport::set_render_target_to_screen_rect);

	ObjectTypeDB::bind_method(_MD("get_mouse_pos"), &Viewport::get_mouse_pos);
	ObjectTypeDB::bind_method(_MD("warp_mouse","to_pos"), &Viewport::warp_mouse);

	ObjectTypeDB::bind_method(_MD("gui_has_modal_stack"), &Viewport::gui_has_modal_stack);
	ObjectTypeDB::bind_method(_MD("gui_get_drag_data:Variant"), &Viewport::gui_get_drag_data);

	ObjectTypeDB::bind_method(_MD("set_disable_input","disable"), &Viewport::set_disable_input);
	ObjectTypeDB::bind_method(_MD("is_input_disabled"), &Viewport::is_input_disabled);

	ObjectTypeDB::bind_method(_MD("_gui_show_tooltip"), &Viewport::_gui_show_tooltip);
	ObjectTypeDB::bind_method(_MD("_gui_remove_focus"), &Viewport::_gui_remove_focus);

	ADD_PROPERTY( PropertyInfo(Variant::RECT2,"rect"), _SCS("set_rect"), _SCS("get_rect") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"own_world"), _SCS("set_use_own_world"), _SCS("is_using_own_world") );
	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"world",PROPERTY_HINT_RESOURCE_TYPE,"World"), _SCS("set_world"), _SCS("get_world") );
//	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"world_2d",PROPERTY_HINT_RESOURCE_TYPE,"World2D"), _SCS("set_world_2d"), _SCS("get_world_2d") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"transparent_bg"), _SCS("set_transparent_background"), _SCS("has_transparent_background") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"render_target/enabled"), _SCS("set_as_render_target"), _SCS("is_set_as_render_target") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"render_target/v_flip"), _SCS("set_render_target_vflip"), _SCS("get_render_target_vflip") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"render_target/clear_on_new_frame"), _SCS("set_render_target_clear_on_new_frame"), _SCS("get_render_target_clear_on_new_frame") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"render_target/filter"), _SCS("set_render_target_filter"), _SCS("get_render_target_filter") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"render_target/gen_mipmaps"), _SCS("set_render_target_gen_mipmaps"), _SCS("get_render_target_gen_mipmaps") );
	ADD_PROPERTY( PropertyInfo(Variant::INT,"render_target/update_mode",PROPERTY_HINT_ENUM,"Disabled,Once,When Visible,Always"), _SCS("set_render_target_update_mode"), _SCS("get_render_target_update_mode") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"audio_listener/enable_2d"), _SCS("set_as_audio_listener_2d"), _SCS("is_audio_listener_2d") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"audio_listener/enable_3d"), _SCS("set_as_audio_listener"), _SCS("is_audio_listener") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"physics/object_picking"), _SCS("set_physics_object_picking"), _SCS("get_physics_object_picking") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"gui/disable_input"), _SCS("set_disable_input"), _SCS("is_input_disabled") );

	ADD_SIGNAL(MethodInfo("size_changed"));

	BIND_CONSTANT( RENDER_TARGET_UPDATE_DISABLED );
	BIND_CONSTANT( RENDER_TARGET_UPDATE_ONCE  );
	BIND_CONSTANT( RENDER_TARGET_UPDATE_WHEN_VISIBLE  );
	BIND_CONSTANT( RENDER_TARGET_UPDATE_ALWAYS  );

}





Viewport::Viewport() {


	world_2d = Ref<World2D>( memnew( World2D ));

	viewport = VisualServer::get_singleton()->viewport_create();
	internal_listener = SpatialSoundServer::get_singleton()->listener_create();
	audio_listener=false;
	internal_listener_2d = SpatialSound2DServer::get_singleton()->listener_create();
	audio_listener_2d=false;
	transparent_bg=false;
	parent=NULL;
	listener=NULL;
	camera=NULL;
	size_override=false;
	size_override_stretch=false;
	size_override_size=Size2(1,1);
	render_target_gen_mipmaps=false;
	render_target=false;
	render_target_vflip=false;
	render_target_clear_on_new_frame=true;
	//render_target_clear=true;
	render_target_update_mode=RENDER_TARGET_UPDATE_WHEN_VISIBLE;
	render_target_texture = Ref<RenderTargetTexture>( memnew( RenderTargetTexture(this) ) );

	physics_object_picking=false;
	physics_object_capture=0;
	physics_object_over=0;
	physics_last_mousepos=Vector2(1e20,1e20);


	String id=itos(get_instance_ID());
	input_group = "_vp_input"+id;
	gui_input_group = "_vp_gui_input"+id;
	unhandled_input_group = "_vp_unhandled_input"+id;
	unhandled_key_input_group = "_vp_unhandled_key_input"+id;

	disable_input=false;

	//window tooltip
	gui.tooltip_timer = -1;

	//gui.tooltip_timer->force_parent_owned();
	gui.tooltip_delay=GLOBAL_DEF("display/tooltip_delay",0.7);

	gui.tooltip=NULL;
	gui.tooltip_label=NULL;
	gui.drag_preview=NULL;
	gui.drag_attempted=false;


	parent_control=NULL;


}


Viewport::~Viewport() {

	VisualServer::get_singleton()->free( viewport );
	SpatialSoundServer::get_singleton()->free(internal_listener);
	SpatialSound2DServer::get_singleton()->free(internal_listener_2d);
	if (render_target_texture.is_valid())
		render_target_texture->vp=NULL; //so if used, will crash
}



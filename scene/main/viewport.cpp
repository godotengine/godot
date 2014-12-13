/*************************************************************************/
/*  viewport.cpp                                                         */
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
#include "viewport.h"
#include "os/os.h"
#include "scene/3d/spatial.h"
//#include "scene/3d/camera.h"

#include "servers/spatial_sound_server.h"
#include "servers/spatial_sound_2d_server.h"
#include "scene/gui/control.h"
#include "scene/3d/camera.h"
#include "scene/3d/spatial_indexer.h"
#include "scene/3d/collision_object.h"



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

	Node *parent = get_parent();

	if (!render_target && parent && parent->cast_to<Control>()) {

		Control *c = parent->cast_to<Control>();

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

	Node *parent = get_parent();

	if (parent && parent->cast_to<Control>()) {

		Control *c = parent->cast_to<Control>();
		VisualServer::get_singleton()->canvas_item_set_visible(canvas_item,c->is_visible());

		_update_listener();
		_update_listener_2d();
	}


}


void Viewport::_vp_enter_tree() {

	Node *parent = get_parent();
		//none?
	if (parent && parent->cast_to<Control>()) {

		Control *cparent=parent->cast_to<Control>();
		RID parent_ci = cparent->get_canvas_item();
		ERR_FAIL_COND(!parent_ci.is_valid());
		canvas_item = VisualServer::get_singleton()->canvas_item_create();

		VisualServer::get_singleton()->canvas_item_set_parent(canvas_item,parent_ci);
		VisualServer::get_singleton()->canvas_item_set_visible(canvas_item,false);
		VisualServer::get_singleton()->canvas_item_attach_viewport(canvas_item,viewport);
		parent->connect("resized",this,"_parent_resized");
		parent->connect("visibility_changed",this,"_parent_visibility_changed");
	} else if (!parent){

		VisualServer::get_singleton()->viewport_attach_to_screen(viewport,0);

	}


}

void Viewport::_vp_exit_tree() {

	Node *parent = get_parent();
	if (parent && parent->cast_to<Control>()) {

		parent->disconnect("resized",this,"_parent_resized");
	}

	if (parent && parent->cast_to<Control>()) {

		parent->disconnect("visibility_changed",this,"_parent_visibility_changed");
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


			if (!render_target)
				_vp_enter_tree();

			this->parent=NULL;
			Node *parent=get_parent();

			if (parent) {


				while(parent && !(this->parent=parent->cast_to<Viewport>())) {

					parent=parent->get_parent();
				}
			}

			current_canvas=find_world_2d()->get_canvas();
			VisualServer::get_singleton()->viewport_set_scenario(viewport,find_world()->get_scenario());
			VisualServer::get_singleton()->viewport_attach_canvas(viewport,current_canvas);

			_update_listener();
			_update_listener_2d();
			_update_rect();

			if (world_2d.is_valid()) {
				find_world_2d()->_register_viewport(this,Rect2());
//best to defer this and not do it here, as it can annoy a lot of setup logic if user
//adds a node and then moves it, will get enter/exit screen/viewport notifications
//unnecesarily
//				update_worlds();
			}

			add_to_group("_viewports");

		} break;
		case NOTIFICATION_READY: {
#ifndef _3D_DISABLED
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



			if (world_2d.is_valid())
				world_2d->_remove_viewport(this);

			if (!render_target)
				_vp_exit_tree();

			VisualServer::get_singleton()->viewport_set_scenario(viewport,RID());
			SpatialSoundServer::get_singleton()->listener_set_space(listener,RID());
			VisualServer::get_singleton()->viewport_remove_canvas(viewport,current_canvas);
			remove_from_group("_viewports");

		} break;
		case NOTIFICATION_FIXED_PROCESS: {

			if (physics_object_picking) {
#ifndef _3D_DISABLED
				Vector2 last_pos(1e20,1e20);
				CollisionObject *last_object;
				ObjectID last_id=0;
				PhysicsDirectSpaceState::RayResult result;

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

								bool col = space->intersect_ray(from,from+dir*10000,result,Set<RID>(),0xFFFFFFFF,0xFFFFFFFF);
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

						bool col = space->intersect_ray(from,from+dir*10000,result,Set<RID>(),0xFFFFFFFF,0xFFFFFFFF);
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

				}
#endif
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

	if (is_inside_tree() && audio_listener && camera && (!get_parent() || (get_parent()->cast_to<Control>() && get_parent()->cast_to<Control>()->is_visible())))  {
		SpatialSoundServer::get_singleton()->listener_set_space(listener,find_world()->get_sound_space());
	} else {
		SpatialSoundServer::get_singleton()->listener_set_space(listener,RID());
	}


}

void Viewport::_update_listener_2d() {

	if (is_inside_tree() && audio_listener && (!get_parent() || (get_parent()->cast_to<Control>() && get_parent()->cast_to<Control>()->is_visible())))
		SpatialSound2DServer::get_singleton()->listener_set_space(listener_2d,find_world_2d()->get_sound_space());
	else
		SpatialSound2DServer::get_singleton()->listener_set_space(listener_2d,RID());

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
	SpatialSound2DServer::get_singleton()->listener_set_transform(listener_2d,Matrix32(0,xform.xform(ss*0.5)));
	Vector2 ss2 = ss*xform.get_scale();
	float panrange = MAX(ss2.x,ss2.y);

	SpatialSound2DServer::get_singleton()->listener_set_param(listener_2d,SpatialSound2DServer::LISTENER_PARAM_PAN_RANGE,panrange);


}

Matrix32 Viewport::get_canvas_transform() const{

	return canvas_transform;
}



void Viewport::_update_global_transform() {


	Matrix32 sxform = stretch_transform * global_canvas_transform;

	VisualServer::get_singleton()->viewport_set_global_canvas_transform(viewport,sxform);

	Matrix32 xform = (sxform * canvas_transform).affine_inverse();
	Size2 ss = get_visible_rect().size;
	SpatialSound2DServer::get_singleton()->listener_set_transform(listener_2d,Matrix32(0,xform.xform(ss*0.5)));
	Vector2 ss2 = ss*xform.get_scale();
	float panrange = MAX(ss2.x,ss2.y);

	SpatialSound2DServer::get_singleton()->listener_set_param(listener_2d,SpatialSound2DServer::LISTENER_PARAM_PAN_RANGE,panrange);

}


void Viewport::set_global_canvas_transform(const Matrix32& p_transform) {

	global_canvas_transform=p_transform;

	_update_global_transform();


}

Matrix32 Viewport::get_global_canvas_transform() const{

	return global_canvas_transform;
}


void Viewport::_camera_transform_changed_notify() {

#ifndef _3D_DISABLED
	if (camera)
		SpatialSoundServer::get_singleton()->listener_set_transform(listener,camera->get_camera_transform());
#endif
}

void Viewport::_set_camera(Camera* p_camera) {

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


void Viewport::set_transparent_background(bool p_enable) {

	transparent_bg=p_enable;
	VS::get_singleton()->viewport_set_transparent_background(viewport,p_enable);

}

bool Viewport::has_transparent_background() const {

	return transparent_bg;
}

#if 0
void Viewport::set_world_2d(const Ref<World2D>& p_world_2d) {

	world_2d=p_world_2d;
	_update_listener_2d();

	if (is_inside_scene()) {
		if (current_canvas.is_valid())
			VisualServer::get_singleton()->viewport_remove_canvas(viewport,current_canvas);
		current_canvas=find_world_2d()->get_canvas();
		VisualServer::get_singleton()->viewport_attach_canvas(viewport,current_canvas);
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
#endif

Ref<World2D> Viewport::find_world_2d() const{

	return world_2d;
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


void Viewport::_propagate_exit_world(Node *p_node) {

	if (p_node!=this) {

		if (!p_node->is_inside_tree()) //may have exited scene already
			return;

		Spatial *s = p_node->cast_to<Spatial>();
		if (s) {

			s->notification(Spatial::NOTIFICATION_EXIT_WORLD,false);
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

	return size_override;
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


void Viewport::set_render_target_filter(bool p_enable) {

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

		ERR_FAIL_COND_V(to_screen_rect.size.x==0,pre_xf);
		ERR_FAIL_COND_V(to_screen_rect.size.y==0,pre_xf);
		pre_xf.scale(rect.size/to_screen_rect.size);
		pre_xf.elements[2]=-to_screen_rect.pos;
	} else {

		pre_xf.elements[2]=-rect.pos;
	}

	return pre_xf;
}

void Viewport::_make_input_local(InputEvent& ev) {

	switch(ev.type) {

		case InputEvent::MOUSE_BUTTON: {

			Matrix32 ai = get_final_transform().affine_inverse() * _get_input_pre_xform();
			Vector2 g = ai.xform(Vector2(ev.mouse_button.global_x,ev.mouse_button.global_y));
			Vector2 l = ai.xform(Vector2(ev.mouse_button.x,ev.mouse_button.y));
			ev.mouse_button.x=l.x;
			ev.mouse_button.y=l.y;
			ev.mouse_button.global_x=g.x;
			ev.mouse_button.global_y=g.y;

		} break;
		case InputEvent::MOUSE_MOTION: {

			Matrix32 ai = get_final_transform().affine_inverse() * _get_input_pre_xform();
			Vector2 g = ai.xform(Vector2(ev.mouse_motion.global_x,ev.mouse_motion.global_y));
			Vector2 l = ai.xform(Vector2(ev.mouse_motion.x,ev.mouse_motion.y));
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

			Matrix32 ai = get_final_transform().affine_inverse() * _get_input_pre_xform();
			Vector2 t = ai.xform(Vector2(ev.screen_touch.x,ev.screen_touch.y));
			ev.screen_touch.x=t.x;
			ev.screen_touch.y=t.y;

		} break;
		case InputEvent::SCREEN_DRAG: {

			Matrix32 ai = get_final_transform().affine_inverse() * _get_input_pre_xform();
			Vector2 t = ai.xform(Vector2(ev.screen_drag.x,ev.screen_drag.y));
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


void Viewport::_vp_input(const InputEvent& p_ev) {

	if (render_target && to_screen_rect==Rect2())
		return; //if render target, can't get input events

	//this one handles system input, p_ev are in system coordinates
	//they are converted to viewport coordinates

	InputEvent ev = p_ev;	
	_make_input_local(ev);
	input(ev);

}

void Viewport::_vp_unhandled_input(const InputEvent& p_ev) {


	if (render_target && to_screen_rect==Rect2())
		return; //if render target, can't get input events

	//this one handles system input, p_ev are in system coordinates
	//they are converted to viewport coordinates

	InputEvent ev = p_ev;
	_make_input_local(ev);
	unhandled_input(ev);

}

void Viewport::input(const InputEvent& p_event) {

	ERR_FAIL_COND(!is_inside_tree());
	get_tree()->_call_input_pause(input_group,"_input",p_event);
	get_tree()->call_group(SceneTree::GROUP_CALL_REVERSE|SceneTree::GROUP_CALL_REALTIME|SceneTree::GROUP_CALL_MULIILEVEL,gui_input_group,"_gui_input",p_event); //special one for GUI, as controls use their own process check
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


void Viewport::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("set_rect","rect"), &Viewport::set_rect);
	ObjectTypeDB::bind_method(_MD("get_rect"), &Viewport::get_rect);
	//ObjectTypeDB::bind_method(_MD("set_world_2d","world_2d:World2D"), &Viewport::set_world_2d);
	//ObjectTypeDB::bind_method(_MD("get_world_2d:World2D"), &Viewport::get_world_2d);
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
	ObjectTypeDB::bind_method(_MD("set_render_target_to_screen_rect"), &Viewport::set_render_target_to_screen_rect);


	ADD_PROPERTY( PropertyInfo(Variant::RECT2,"rect"), _SCS("set_rect"), _SCS("get_rect") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"own_world"), _SCS("set_use_own_world"), _SCS("is_using_own_world") );
	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"world",PROPERTY_HINT_RESOURCE_TYPE,"World"), _SCS("set_world"), _SCS("get_world") );
//	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"world_2d",PROPERTY_HINT_RESOURCE_TYPE,"World2D"), _SCS("set_world_2d"), _SCS("get_world_2d") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"transparent_bg"), _SCS("set_transparent_background"), _SCS("has_transparent_background") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"render_target/enabled"), _SCS("set_as_render_target"), _SCS("is_set_as_render_target") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"render_target/v_flip"), _SCS("set_render_target_vflip"), _SCS("get_render_target_vflip") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"render_target/filter"), _SCS("set_render_target_filter"), _SCS("get_render_target_filter") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"render_target/gen_mipmaps"), _SCS("set_render_target_gen_mipmaps"), _SCS("get_render_target_gen_mipmaps") );
	ADD_PROPERTY( PropertyInfo(Variant::INT,"render_target/update_mode",PROPERTY_HINT_ENUM,"Disabled,Once,When Visible,Always"), _SCS("set_render_target_update_mode"), _SCS("get_render_target_update_mode") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"audio_listener/enable_2d"), _SCS("set_as_audio_listener_2d"), _SCS("is_audio_listener_2d") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"audio_listener/enable_3d"), _SCS("set_as_audio_listener"), _SCS("is_audio_listener") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"physics/object_picking"), _SCS("set_physics_object_picking"), _SCS("get_physics_object_picking") );

	ADD_SIGNAL(MethodInfo("size_changed"));

	BIND_CONSTANT( RENDER_TARGET_UPDATE_DISABLED );
	BIND_CONSTANT( RENDER_TARGET_UPDATE_ONCE  );
	BIND_CONSTANT( RENDER_TARGET_UPDATE_WHEN_VISIBLE  );
	BIND_CONSTANT( RENDER_TARGET_UPDATE_ALWAYS  );

}





Viewport::Viewport() {

	world_2d = Ref<World2D>( memnew( World2D ));

	viewport = VisualServer::get_singleton()->viewport_create();
	listener=SpatialSoundServer::get_singleton()->listener_create();
	audio_listener=false;
	listener_2d=SpatialSound2DServer::get_singleton()->listener_create();
	audio_listener_2d=false;
	transparent_bg=false;
	parent=NULL;
	camera=NULL;
	size_override=false;
	size_override_stretch=false;
	size_override_size=Size2(1,1);
	render_target_gen_mipmaps=false;
	render_target=false;
	render_target_vflip=false;
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


}


Viewport::~Viewport() {

	VisualServer::get_singleton()->free( viewport );
	SpatialSoundServer::get_singleton()->free(listener);
	if (render_target_texture.is_valid())
		render_target_texture->vp=NULL; //so if used, will crash
}



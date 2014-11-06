/*************************************************************************/
/*  visibility_notifier_2d.cpp                                           */
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
#include "visibility_notifier_2d.h"

#include "scene/scene_string_names.h"
#include "scene/2d/physics_body_2d.h"
#include "scene/animation/animation_player.h"
#include "scene/scene_string_names.h"

void VisibilityNotifier2D::_enter_viewport(Viewport* p_viewport) {

	ERR_FAIL_COND(viewports.has(p_viewport));
	viewports.insert(p_viewport);

	if (viewports.size()==1) {
		emit_signal(SceneStringNames::get_singleton()->enter_screen);

		_screen_enter();
	}
	emit_signal(SceneStringNames::get_singleton()->enter_viewport,p_viewport);

}

void VisibilityNotifier2D::_exit_viewport(Viewport* p_viewport){

	ERR_FAIL_COND(!viewports.has(p_viewport));
	viewports.erase(p_viewport);

	emit_signal(SceneStringNames::get_singleton()->exit_viewport,p_viewport);
	if (viewports.size()==0) {
		emit_signal(SceneStringNames::get_singleton()->exit_screen);

		_screen_exit();
	}
}


void VisibilityNotifier2D::set_rect(const Rect2& p_rect){

	rect=p_rect;
	if (is_inside_tree())
		get_world_2d()->_update_notifier(this,get_global_transform().xform(rect));

	_change_notify("rect");
}

Rect2 VisibilityNotifier2D::get_item_rect() const {

	return rect;
}

Rect2 VisibilityNotifier2D::get_rect() const{

	return rect;
}


void VisibilityNotifier2D::_notification(int p_what) {


	switch(p_what) {
		case NOTIFICATION_ENTER_TREE: {

			//get_world_2d()->
			get_world_2d()->_register_notifier(this,get_global_transform().xform(rect));
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			//get_world_2d()->
			get_world_2d()->_update_notifier(this,get_global_transform().xform(rect));
		} break;
		case NOTIFICATION_DRAW: {

			if (get_tree()->is_editor_hint()) {

				draw_rect(rect,Color(1,0.5,1,0.2));
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {

			get_world_2d()->_remove_notifier(this);
		} break;
	}
}

bool VisibilityNotifier2D::is_on_screen() const {

	return viewports.size()>0;
}

void VisibilityNotifier2D::_bind_methods(){

	ObjectTypeDB::bind_method(_MD("set_rect","rect"),&VisibilityNotifier2D::set_rect);
	ObjectTypeDB::bind_method(_MD("get_rect"),&VisibilityNotifier2D::get_rect);
	ObjectTypeDB::bind_method(_MD("is_on_screen"),&VisibilityNotifier2D::is_on_screen);

	ADD_PROPERTY( PropertyInfo(Variant::RECT2,"rect"),_SCS("set_rect"),_SCS("get_rect"));

	ADD_SIGNAL( MethodInfo("enter_viewport",PropertyInfo(Variant::OBJECT,"viewport",PROPERTY_HINT_RESOURCE_TYPE,"Viewport")) );
	ADD_SIGNAL( MethodInfo("exit_viewport",PropertyInfo(Variant::OBJECT,"viewport",PROPERTY_HINT_RESOURCE_TYPE,"Viewport")) );
	ADD_SIGNAL( MethodInfo("enter_screen"));
	ADD_SIGNAL( MethodInfo("exit_screen"));
}


VisibilityNotifier2D::VisibilityNotifier2D() {

	rect=Rect2(-10,-10,20,20);
}





//////////////////////////////////////


void VisibilityEnabler2D::_screen_enter() {


	for(Map<Node*,Variant>::Element *E=nodes.front();E;E=E->next()) {

		_change_node_state(E->key(),true);
	}

	visible=true;
}

void VisibilityEnabler2D::_screen_exit(){

	for(Map<Node*,Variant>::Element *E=nodes.front();E;E=E->next()) {

		_change_node_state(E->key(),false);
	}

	visible=false;
}

void VisibilityEnabler2D::_find_nodes(Node* p_node) {


	bool add=false;
	Variant meta;

	if (enabler[ENABLER_FREEZE_BODIES]) {

		RigidBody2D *rb2d = p_node->cast_to<RigidBody2D>();
		if (rb2d && ((rb2d->get_mode()==RigidBody2D::MODE_CHARACTER || (rb2d->get_mode()==RigidBody2D::MODE_RIGID && !rb2d->is_able_to_sleep())))) {


			add=true;
			meta=rb2d->get_mode();
		}
	}

	if (enabler[ENABLER_PAUSE_ANIMATIONS]) {

		AnimationPlayer *ap = p_node->cast_to<AnimationPlayer>();
		if (ap) {
			add=true;
		}

	}

	if (add) {

		p_node->connect(SceneStringNames::get_singleton()->exit_tree,this,"_node_removed",varray(p_node),CONNECT_ONESHOT);
		nodes[p_node]=meta;
		_change_node_state(p_node,false);
	}

	for(int i=0;i<p_node->get_child_count();i++) {
		Node *c = p_node->get_child(i);
		if (c->get_filename()!=String())
			continue; //skip, instance

		_find_nodes(c);
	}

}

void VisibilityEnabler2D::_notification(int p_what){

	if (p_what==NOTIFICATION_ENTER_TREE) {

		if (get_tree()->is_editor_hint())
			return;


		Node *from = this;
		//find where current scene starts
		while(from->get_parent() && from->get_filename()==String())
			from=from->get_parent();

		_find_nodes(from);

	}

	if (p_what==NOTIFICATION_EXIT_TREE) {

		if (get_tree()->is_editor_hint())
			return;


		Node *from = this;
		//find where current scene starts

		for (Map<Node*,Variant>::Element *E=nodes.front();E;E=E->next()) {

			if (!visible)
				_change_node_state(E->key(),true);
			E->key()->disconnect(SceneStringNames::get_singleton()->exit_tree,this,"_node_removed");
		}

		nodes.clear();

	}
}

void VisibilityEnabler2D::_change_node_state(Node* p_node,bool p_enabled) {

	ERR_FAIL_COND(!nodes.has(p_node));

	{
		RigidBody2D *rb = p_node->cast_to<RigidBody2D>();
		if (rb) {

			if (p_enabled) {
				RigidBody2D::Mode mode = RigidBody2D::Mode(nodes[p_node].operator int());
				//rb->set_mode(mode);
				rb->set_sleeping(false);
			} else {
				//rb->set_mode(RigidBody2D::MODE_STATIC);
				rb->set_sleeping(true);
			}
		}
	}

	{
		AnimationPlayer *ap=p_node->cast_to<AnimationPlayer>();

		if (ap) {

			ap->set_active(p_enabled);
		}
	}

}


void VisibilityEnabler2D::_node_removed(Node* p_node) {

	if (!visible)
		_change_node_state(p_node,true);
	//changed to one shot, not needed
	//p_node->disconnect(SceneStringNames::get_singleton()->exit_scene,this,"_node_removed");
	nodes.erase(p_node);

}

void VisibilityEnabler2D::_bind_methods(){

	ObjectTypeDB::bind_method(_MD("set_enabler","enabler","enabled"),&VisibilityEnabler2D::set_enabler);
	ObjectTypeDB::bind_method(_MD("is_enabler_enabled","enabler"),&VisibilityEnabler2D::is_enabler_enabled);
	ObjectTypeDB::bind_method(_MD("_node_removed"),&VisibilityEnabler2D::_node_removed);

	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"enabler/pause_animations"),_SCS("set_enabler"),_SCS("is_enabler_enabled"), ENABLER_PAUSE_ANIMATIONS );
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"enabler/freeze_bodies"),_SCS("set_enabler"),_SCS("is_enabler_enabled"), ENABLER_FREEZE_BODIES);

	BIND_CONSTANT( ENABLER_FREEZE_BODIES );
	BIND_CONSTANT( ENABLER_PAUSE_ANIMATIONS );
	BIND_CONSTANT( ENABLER_MAX);
}

void VisibilityEnabler2D::set_enabler(Enabler p_enabler,bool p_enable){

	ERR_FAIL_INDEX(p_enabler,ENABLER_MAX);
	enabler[p_enabler]=p_enable;

}
bool VisibilityEnabler2D::is_enabler_enabled(Enabler p_enabler) const{

	ERR_FAIL_INDEX_V(p_enabler,ENABLER_MAX,false);
	return enabler[p_enabler];

}

VisibilityEnabler2D::VisibilityEnabler2D() {

	for(int i=0;i<ENABLER_MAX;i++)
		enabler[i]=true;

	visible=false;

}


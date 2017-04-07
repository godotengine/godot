/*************************************************************************/
/*  visibility_notifier.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "visibility_notifier.h"

#include "scene/3d/physics_body.h"
#include "scene/animation/animation_player.h"
#include "scene/scene_string_names.h"
#include "scene/scene_string_names.h"

void VisibilityNotifier::_enter_camera(Camera *p_camera) {

	ERR_FAIL_COND(cameras.has(p_camera));
	cameras.insert(p_camera);
	if (cameras.size() == 1) {
		emit_signal(SceneStringNames::get_singleton()->screen_entered);
		_screen_enter();
	}
	emit_signal(SceneStringNames::get_singleton()->camera_entered, p_camera);
}

void VisibilityNotifier::_exit_camera(Camera *p_camera) {

	ERR_FAIL_COND(!cameras.has(p_camera));
	cameras.erase(p_camera);

	emit_signal(SceneStringNames::get_singleton()->camera_exited, p_camera);
	if (cameras.size() == 0) {
		emit_signal(SceneStringNames::get_singleton()->screen_exited);

		_screen_exit();
	}
}

void VisibilityNotifier::set_aabb(const Rect3 &p_aabb) {

	if (aabb == p_aabb)
		return;
	aabb = p_aabb;

	if (is_inside_world()) {
		get_world()->_update_notifier(this, get_global_transform().xform(aabb));
	}

	_change_notify("aabb");
	update_gizmo();
}

Rect3 VisibilityNotifier::get_aabb() const {

	return aabb;
}

void VisibilityNotifier::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_WORLD: {

			get_world()->_register_notifier(this, get_global_transform().xform(aabb));
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			get_world()->_update_notifier(this, get_global_transform().xform(aabb));
		} break;
		case NOTIFICATION_EXIT_WORLD: {

			get_world()->_remove_notifier(this);
		} break;
	}
}

bool VisibilityNotifier::is_on_screen() const {

	return cameras.size() != 0;
}

void VisibilityNotifier::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_aabb", "rect"), &VisibilityNotifier::set_aabb);
	ClassDB::bind_method(D_METHOD("get_aabb"), &VisibilityNotifier::get_aabb);
	ClassDB::bind_method(D_METHOD("is_on_screen"), &VisibilityNotifier::is_on_screen);

	ADD_PROPERTY(PropertyInfo(Variant::RECT3, "aabb"), "set_aabb", "get_aabb");

	ADD_SIGNAL(MethodInfo("camera_entered", PropertyInfo(Variant::OBJECT, "camera", PROPERTY_HINT_RESOURCE_TYPE, "Camera")));
	ADD_SIGNAL(MethodInfo("camera_exited", PropertyInfo(Variant::OBJECT, "camera", PROPERTY_HINT_RESOURCE_TYPE, "Camera")));
	ADD_SIGNAL(MethodInfo("screen_entered"));
	ADD_SIGNAL(MethodInfo("screen_exited"));
}

VisibilityNotifier::VisibilityNotifier() {

	aabb = Rect3(Vector3(-1, -1, -1), Vector3(2, 2, 2));
	set_notify_transform(true);
}

//////////////////////////////////////

void VisibilityEnabler::_screen_enter() {

	for (Map<Node *, Variant>::Element *E = nodes.front(); E; E = E->next()) {

		_change_node_state(E->key(), true);
	}

	visible = true;
}

void VisibilityEnabler::_screen_exit() {

	for (Map<Node *, Variant>::Element *E = nodes.front(); E; E = E->next()) {

		_change_node_state(E->key(), false);
	}

	visible = false;
}

void VisibilityEnabler::_find_nodes(Node *p_node) {

	bool add = false;
	Variant meta;

	if (enabler[ENABLER_FREEZE_BODIES]) {

		RigidBody *rb = p_node->cast_to<RigidBody>();
		if (rb && ((rb->get_mode() == RigidBody::MODE_CHARACTER || (rb->get_mode() == RigidBody::MODE_RIGID && !rb->is_able_to_sleep())))) {

			add = true;
			meta = rb->get_mode();
		}
	}

	if (enabler[ENABLER_PAUSE_ANIMATIONS]) {

		AnimationPlayer *ap = p_node->cast_to<AnimationPlayer>();
		if (ap) {
			add = true;
		}
	}

	if (add) {

		p_node->connect(SceneStringNames::get_singleton()->tree_exited, this, "_node_removed", varray(p_node), CONNECT_ONESHOT);
		nodes[p_node] = meta;
		_change_node_state(p_node, false);
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *c = p_node->get_child(i);
		if (c->get_filename() != String())
			continue; //skip, instance

		_find_nodes(c);
	}
}

void VisibilityEnabler::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		if (get_tree()->is_editor_hint())
			return;

		Node *from = this;
		//find where current scene starts
		while (from->get_parent() && from->get_filename() == String())
			from = from->get_parent();

		_find_nodes(from);
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {

		if (get_tree()->is_editor_hint())
			return;

		for (Map<Node *, Variant>::Element *E = nodes.front(); E; E = E->next()) {

			if (!visible)
				_change_node_state(E->key(), true);
			E->key()->disconnect(SceneStringNames::get_singleton()->tree_exited, this, "_node_removed");
		}

		nodes.clear();
	}
}

void VisibilityEnabler::_change_node_state(Node *p_node, bool p_enabled) {

	ERR_FAIL_COND(!nodes.has(p_node));

	{
		RigidBody *rb = p_node->cast_to<RigidBody>();
		if (rb)

			rb->set_sleeping(!p_enabled);
	}

	{
		AnimationPlayer *ap = p_node->cast_to<AnimationPlayer>();

		if (ap) {

			ap->set_active(p_enabled);
		}
	}
}

void VisibilityEnabler::_node_removed(Node *p_node) {

	if (!visible)
		_change_node_state(p_node, true);
	p_node->disconnect(SceneStringNames::get_singleton()->tree_exited, this, "_node_removed");
	nodes.erase(p_node);
}

void VisibilityEnabler::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_enabler", "enabler", "enabled"), &VisibilityEnabler::set_enabler);
	ClassDB::bind_method(D_METHOD("is_enabler_enabled", "enabler"), &VisibilityEnabler::is_enabler_enabled);
	ClassDB::bind_method(D_METHOD("_node_removed"), &VisibilityEnabler::_node_removed);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "pause_animations"), "set_enabler", "is_enabler_enabled", ENABLER_PAUSE_ANIMATIONS);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "freeze_bodies"), "set_enabler", "is_enabler_enabled", ENABLER_FREEZE_BODIES);

	BIND_CONSTANT(ENABLER_FREEZE_BODIES);
	BIND_CONSTANT(ENABLER_PAUSE_ANIMATIONS);
	BIND_CONSTANT(ENABLER_MAX);
}

void VisibilityEnabler::set_enabler(Enabler p_enabler, bool p_enable) {

	ERR_FAIL_INDEX(p_enabler, ENABLER_MAX);
	enabler[p_enabler] = p_enable;
}
bool VisibilityEnabler::is_enabler_enabled(Enabler p_enabler) const {

	ERR_FAIL_INDEX_V(p_enabler, ENABLER_MAX, false);
	return enabler[p_enabler];
}

VisibilityEnabler::VisibilityEnabler() {

	for (int i = 0; i < ENABLER_MAX; i++)
		enabler[i] = true;

	visible = false;
}

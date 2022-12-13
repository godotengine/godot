/*************************************************************************/
/*  visibility_notifier.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/engine.h"
#include "scene/3d/camera.h"
#include "scene/3d/physics_body.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/animation/animation_tree_player.h"
#include "scene/scene_string_names.h"

void VisibilityNotifier::_enter_camera(Camera *p_camera) {
	ERR_FAIL_COND(cameras.has(p_camera));
	cameras.insert(p_camera);

	bool in_gameplay = _in_gameplay;
	if (!Engine::get_singleton()->are_portals_active()) {
		in_gameplay = true;
	}

	if ((cameras.size() == 1) && in_gameplay) {
		emit_signal(SceneStringNames::get_singleton()->screen_entered);
		_screen_enter();
	}

	emit_signal(SceneStringNames::get_singleton()->camera_entered, p_camera);
}

void VisibilityNotifier::_exit_camera(Camera *p_camera) {
	ERR_FAIL_COND(!cameras.has(p_camera));
	cameras.erase(p_camera);

	bool in_gameplay = _in_gameplay;
	if (!Engine::get_singleton()->are_portals_active()) {
		in_gameplay = true;
	}

	emit_signal(SceneStringNames::get_singleton()->camera_exited, p_camera);
	if ((cameras.size() == 0) && (in_gameplay)) {
		emit_signal(SceneStringNames::get_singleton()->screen_exited);

		_screen_exit();
	}
}

void VisibilityNotifier::set_max_distance(real_t p_max_distance) {
	if (p_max_distance > CMP_EPSILON) {
		_max_distance = p_max_distance;
		_max_distance_squared = _max_distance * _max_distance;
		_max_distance_active = true;

		// make sure world aabb centre is up to date
		if (is_inside_world()) {
			AABB world_aabb = get_global_transform().xform(aabb);
			_world_aabb_center = world_aabb.get_center();
		}
	} else {
		_max_distance = 0.0;
		_max_distance_squared = 0.0;
		_max_distance_active = false;
	}
}

void VisibilityNotifier::set_aabb(const AABB &p_aabb) {
	if (aabb == p_aabb) {
		return;
	}
	aabb = p_aabb;

	if (is_inside_world()) {
		AABB world_aabb = get_global_transform().xform(aabb);
		get_world()->_update_notifier(this, world_aabb);
		_world_aabb_center = world_aabb.get_center();
	}

	_change_notify("aabb");
	update_gizmo();
}

AABB VisibilityNotifier::get_aabb() const {
	return aabb;
}

void VisibilityNotifier::_refresh_portal_mode() {
	// only create in the visual server if we are roaming.
	// All other cases don't require a visual server rep.
	// Global and ignore are the same (existing client side functionality only).
	// Static and dynamic require only a one off creation at conversion.
	if (get_portal_mode() == PORTAL_MODE_ROAMING) {
		if (is_inside_world()) {
			if (_cull_instance_rid == RID()) {
				_cull_instance_rid = RID_PRIME(VisualServer::get_singleton()->ghost_create());
			}

			if (is_inside_world() && get_world().is_valid() && get_world()->get_scenario().is_valid() && is_inside_tree()) {
				AABB world_aabb = get_global_transform().xform(aabb);
				VisualServer::get_singleton()->ghost_set_scenario(_cull_instance_rid, get_world()->get_scenario(), get_instance_id(), world_aabb);
			}
		} else {
			if (_cull_instance_rid != RID()) {
				VisualServer::get_singleton()->free(_cull_instance_rid);
				_cull_instance_rid = RID();
			}
		}

	} else {
		if (_cull_instance_rid != RID()) {
			VisualServer::get_singleton()->free(_cull_instance_rid);
			_cull_instance_rid = RID();
		}
	}
}

void VisibilityNotifier::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_WORLD: {
			world = get_world();
			ERR_FAIL_COND(!world.is_valid());

			AABB world_aabb = get_global_transform().xform(aabb);
#ifdef TOOLS_ENABLED
			if (!Engine::get_singleton()->is_editor_hint()) {
				world->_register_notifier(this, world_aabb);
			}
#else
			world->_register_notifier(this, world_aabb);
#endif
			_world_aabb_center = world_aabb.get_center();
			_refresh_portal_mode();
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			AABB world_aabb = get_global_transform().xform(aabb);
#ifdef TOOLS_ENABLED
			if (!Engine::get_singleton()->is_editor_hint()) {
				world->_update_notifier(this, world_aabb);
			}
#else
			world->_update_notifier(this, world_aabb);
#endif
			if (_max_distance_active) {
				_world_aabb_center = world_aabb.get_center();
			}

			if (_cull_instance_rid != RID()) {
				VisualServer::get_singleton()->ghost_update(_cull_instance_rid, world_aabb);
			}
		} break;
		case NOTIFICATION_EXIT_WORLD: {
			ERR_FAIL_COND(!world.is_valid());
#ifdef TOOLS_ENABLED
			if (!Engine::get_singleton()->is_editor_hint()) {
				world->_remove_notifier(this);
			}
#else
			world->_remove_notifier(this);
#endif

			if (_cull_instance_rid != RID()) {
				VisualServer::get_singleton()->ghost_set_scenario(_cull_instance_rid, RID(), get_instance_id(), AABB());
			}
		} break;
		case NOTIFICATION_ENTER_GAMEPLAY: {
			_in_gameplay = true;
			if (cameras.size() && Engine::get_singleton()->are_portals_active()) {
				emit_signal(SceneStringNames::get_singleton()->screen_entered);
				_screen_enter();
			}
		} break;
		case NOTIFICATION_EXIT_GAMEPLAY: {
			_in_gameplay = false;
			if (cameras.size() && Engine::get_singleton()->are_portals_active()) {
				emit_signal(SceneStringNames::get_singleton()->screen_exited);
				_screen_exit();
			}
		} break;
	}
}

bool VisibilityNotifier::is_on_screen() const {
	if (!Engine::get_singleton()->are_portals_active()) {
		return cameras.size() != 0;
	}

	return (cameras.size() != 0) && _in_gameplay;
}

void VisibilityNotifier::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_aabb", "rect"), &VisibilityNotifier::set_aabb);
	ClassDB::bind_method(D_METHOD("get_aabb"), &VisibilityNotifier::get_aabb);
	ClassDB::bind_method(D_METHOD("is_on_screen"), &VisibilityNotifier::is_on_screen);

	ClassDB::bind_method(D_METHOD("set_max_distance", "distance"), &VisibilityNotifier::set_max_distance);
	ClassDB::bind_method(D_METHOD("get_max_distance"), &VisibilityNotifier::get_max_distance);

	ADD_PROPERTY(PropertyInfo(Variant::AABB, "aabb"), "set_aabb", "get_aabb");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_distance", PROPERTY_HINT_RANGE, "0,32768,0.01"), "set_max_distance", "get_max_distance");

	ADD_SIGNAL(MethodInfo("camera_entered", PropertyInfo(Variant::OBJECT, "camera", PROPERTY_HINT_RESOURCE_TYPE, "Camera")));
	ADD_SIGNAL(MethodInfo("camera_exited", PropertyInfo(Variant::OBJECT, "camera", PROPERTY_HINT_RESOURCE_TYPE, "Camera")));
	ADD_SIGNAL(MethodInfo("screen_entered"));
	ADD_SIGNAL(MethodInfo("screen_exited"));
}

VisibilityNotifier::VisibilityNotifier() {
	aabb = AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));
	set_notify_transform(true);
	_in_gameplay = false;
	_max_distance_active = false;
	_max_distance = 0.0;
	_max_distance_squared = 0.0;
	_max_distance_leadin_counter = 1; // this could later be exposed as a property if necessary
}

VisibilityNotifier::~VisibilityNotifier() {
	if (_cull_instance_rid != RID()) {
		VisualServer::get_singleton()->free(_cull_instance_rid);
	}
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

	{
		RigidBody *rb = Object::cast_to<RigidBody>(p_node);
		if (rb && ((rb->get_mode() == RigidBody::MODE_CHARACTER || rb->get_mode() == RigidBody::MODE_RIGID))) {
			add = true;
			meta = rb->get_mode();
		}
	}

	if (Object::cast_to<AnimationPlayer>(p_node) || Object::cast_to<AnimationTree>(p_node) || Object::cast_to<AnimationTreePlayer>(p_node)) {
		add = true;
	}

	{
		AnimationTree *at = Object::cast_to<AnimationTree>(p_node);
		if (at) {
			add = true;
		}
	}

	{
		AnimationTreePlayer *atp = Object::cast_to<AnimationTreePlayer>(p_node);
		if (atp) {
			add = true;
		}
	}

	if (add) {
		p_node->connect(SceneStringNames::get_singleton()->tree_exiting, this, "_node_removed", varray(p_node), CONNECT_ONESHOT);
		nodes[p_node] = meta;
		_change_node_state(p_node, false);
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *c = p_node->get_child(i);
		if (c->get_filename() != String()) {
			continue; //skip, instance
		}

		_find_nodes(c);
	}
}

void VisibilityEnabler::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint()) {
			return;
		}
#endif

		Node *from = this;
		//find where current scene starts
		while (from->get_parent() && from->get_filename() == String()) {
			from = from->get_parent();
		}

		_find_nodes(from);
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint()) {
			return;
		}
#endif

		for (Map<Node *, Variant>::Element *E = nodes.front(); E; E = E->next()) {
			if (!visible) {
				_change_node_state(E->key(), true);
			}
			E->key()->disconnect(SceneStringNames::get_singleton()->tree_exiting, this, "_node_removed");
		}

		nodes.clear();
	}
}

void VisibilityEnabler::_change_node_state(Node *p_node, bool p_enabled) {
	ERR_FAIL_COND(!nodes.has(p_node));

	if (enabler[ENABLER_FREEZE_BODIES]) {
		RigidBody *rb = Object::cast_to<RigidBody>(p_node);
		if (rb) {
			rb->set_sleeping(!p_enabled);
		}
	}

	if (enabler[ENABLER_PAUSE_ANIMATIONS]) {
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(p_node);
		if (ap) {
			ap->set_active(p_enabled);
		} else {
			AnimationTree *at = Object::cast_to<AnimationTree>(p_node);
			if (at) {
				at->set_active(p_enabled);
			} else {
				AnimationTreePlayer *atp = Object::cast_to<AnimationTreePlayer>(p_node);
				if (atp) {
					atp->set_active(p_enabled);
				}
			}
		}
	}
}

void VisibilityEnabler::_node_removed(Node *p_node) {
	if (!visible) {
		_change_node_state(p_node, true);
	}
	nodes.erase(p_node);
}

void VisibilityEnabler::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabler", "enabler", "enabled"), &VisibilityEnabler::set_enabler);
	ClassDB::bind_method(D_METHOD("is_enabler_enabled", "enabler"), &VisibilityEnabler::is_enabler_enabled);
	ClassDB::bind_method(D_METHOD("_node_removed"), &VisibilityEnabler::_node_removed);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "pause_animations"), "set_enabler", "is_enabler_enabled", ENABLER_PAUSE_ANIMATIONS);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "freeze_bodies"), "set_enabler", "is_enabler_enabled", ENABLER_FREEZE_BODIES);

	BIND_ENUM_CONSTANT(ENABLER_PAUSE_ANIMATIONS);
	BIND_ENUM_CONSTANT(ENABLER_FREEZE_BODIES);
	BIND_ENUM_CONSTANT(ENABLER_MAX);
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
	for (int i = 0; i < ENABLER_MAX; i++) {
		enabler[i] = true;
	}

	visible = false;
}

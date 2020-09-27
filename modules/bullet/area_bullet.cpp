/*************************************************************************/
/*  area_bullet.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "area_bullet.h"

#include "bullet_physics_server.h"
#include "bullet_types_converter.h"
#include "bullet_utilities.h"
#include "collision_object_bullet.h"
#include "space_bullet.h"

#include <BulletCollision/CollisionDispatch/btGhostObject.h>
#include <btBulletCollisionCommon.h>

/**
	@author AndreaCatania
*/

AreaBullet::AreaBullet() :
		RigidCollisionObjectBullet(CollisionObjectBullet::TYPE_AREA),
		monitorable(true),
		spOv_mode(PhysicsServer::AREA_SPACE_OVERRIDE_DISABLED),
		spOv_gravityPoint(false),
		spOv_gravityPointDistanceScale(0),
		spOv_gravityPointAttenuation(1),
		spOv_gravityVec(0, -1, 0),
		spOv_gravityMag(10),
		spOv_linearDump(0.1),
		spOv_angularDump(0.1),
		spOv_priority(0),
		isScratched(false) {
	btGhost = bulletnew(btGhostObject);
	reload_shapes();
	setupBulletCollisionObject(btGhost);
	/// Collision objects with a callback still have collision response with dynamic rigid bodies.
	/// In order to use collision objects as trigger, you have to disable the collision response.
	set_collision_enabled(false);

	for (int i = 0; i < 5; ++i) {
		call_event_res_ptr[i] = &call_event_res[i];
	}
}

AreaBullet::~AreaBullet() {
	// signal are handled by godot, so just clear without notify
	for (int i = overlappingObjects.size() - 1; 0 <= i; --i) {
		overlappingObjects[i].object->on_exit_area(this);
	}
}

void AreaBullet::dispatch_callbacks() {
	if (!isScratched) {
		return;
	}
	isScratched = false;

	// Reverse order because I've to remove EXIT objects
	for (int i = overlappingObjects.size() - 1; 0 <= i; --i) {
		OverlappingObjectData &otherObj = overlappingObjects.write[i];

		switch (otherObj.state) {
			case OVERLAP_STATE_ENTER:
				otherObj.state = OVERLAP_STATE_INSIDE;
				call_event(otherObj.object, PhysicsServer::AREA_BODY_ADDED);
				otherObj.object->on_enter_area(this);
				break;
			case OVERLAP_STATE_EXIT:
				call_event(otherObj.object, PhysicsServer::AREA_BODY_REMOVED);
				otherObj.object->on_exit_area(this);
				overlappingObjects.remove(i); // Remove after callback
				break;
			case OVERLAP_STATE_INSIDE: {
				if (otherObj.object->getType() == TYPE_RIGID_BODY) {
					RigidBodyBullet *body = static_cast<RigidBodyBullet *>(otherObj.object);
					body->scratch_space_override_modificator();
				}
				break;
			}
			case OVERLAP_STATE_DIRTY:
				break;
		}
	}
}

void AreaBullet::call_event(CollisionObjectBullet *p_otherObject, PhysicsServer::AreaBodyStatus p_status) {
	InOutEventCallback &event = eventsCallbacks[static_cast<int>(p_otherObject->getType())];
	Object *areaGodoObject = ObjectDB::get_instance(event.event_callback_id);

	if (!areaGodoObject) {
		event.event_callback_id = 0;
		return;
	}

	call_event_res[0] = p_status;
	call_event_res[1] = p_otherObject->get_self(); // Other body
	call_event_res[2] = p_otherObject->get_instance_id(); // instance ID
	call_event_res[3] = 0; // other_body_shape ID
	call_event_res[4] = 0; // self_shape ID

	Variant::CallError outResp;
	areaGodoObject->call(event.event_callback_method, (const Variant **)call_event_res_ptr, 5, outResp);
}

void AreaBullet::scratch() {
	if (isScratched) {
		return;
	}
	isScratched = true;
}

void AreaBullet::clear_overlaps(bool p_notify) {
	for (int i = overlappingObjects.size() - 1; 0 <= i; --i) {
		if (p_notify) {
			call_event(overlappingObjects[i].object, PhysicsServer::AREA_BODY_REMOVED);
		}
		overlappingObjects[i].object->on_exit_area(this);
	}
	overlappingObjects.clear();
}

void AreaBullet::remove_overlap(CollisionObjectBullet *p_object, bool p_notify) {
	for (int i = overlappingObjects.size() - 1; 0 <= i; --i) {
		if (overlappingObjects[i].object == p_object) {
			if (p_notify) {
				call_event(overlappingObjects[i].object, PhysicsServer::AREA_BODY_REMOVED);
			}
			overlappingObjects[i].object->on_exit_area(this);
			overlappingObjects.remove(i);
			break;
		}
	}
}

int AreaBullet::find_overlapping_object(CollisionObjectBullet *p_colObj) {
	const int size = overlappingObjects.size();
	for (int i = 0; i < size; ++i) {
		if (overlappingObjects[i].object == p_colObj) {
			return i;
		}
	}
	return -1;
}

void AreaBullet::set_monitorable(bool p_monitorable) {
	monitorable = p_monitorable;
	updated = true;
}

bool AreaBullet::is_monitoring() const {
	return get_godot_object_flags() & GOF_IS_MONITORING_AREA;
}

void AreaBullet::main_shape_changed() {
	CRASH_COND(!get_main_shape());
	btGhost->setCollisionShape(get_main_shape());
	updated = true;
}

void AreaBullet::reload_body() {
	if (space) {
		space->remove_area(this);
		space->add_area(this);
	}
}

void AreaBullet::set_space(SpaceBullet *p_space) {
	// Clear the old space if there is one
	if (space) {
		clear_overlaps(false);
		isScratched = false;

		// Remove this object form the physics world
		space->remove_area(this);
	}

	space = p_space;

	if (space) {
		space->add_area(this);
	}
}

void AreaBullet::on_collision_filters_change() {
	if (space) {
		space->reload_collision_filters(this);
	}
	updated = true;
}

void AreaBullet::add_overlap(CollisionObjectBullet *p_otherObject) {
	scratch();
	overlappingObjects.push_back(OverlappingObjectData(p_otherObject, OVERLAP_STATE_ENTER));
	p_otherObject->notify_new_overlap(this);
}

void AreaBullet::put_overlap_as_exit(int p_index) {
	scratch();
	overlappingObjects.write[p_index].state = OVERLAP_STATE_EXIT;
}

void AreaBullet::put_overlap_as_inside(int p_index) {
	// This check is required to be sure this body was inside
	if (OVERLAP_STATE_DIRTY == overlappingObjects[p_index].state) {
		overlappingObjects.write[p_index].state = OVERLAP_STATE_INSIDE;
	}
}

void AreaBullet::set_param(PhysicsServer::AreaParameter p_param, const Variant &p_value) {
	switch (p_param) {
		case PhysicsServer::AREA_PARAM_GRAVITY:
			set_spOv_gravityMag(p_value);
			break;
		case PhysicsServer::AREA_PARAM_GRAVITY_VECTOR:
			set_spOv_gravityVec(p_value);
			break;
		case PhysicsServer::AREA_PARAM_LINEAR_DAMP:
			set_spOv_linearDump(p_value);
			break;
		case PhysicsServer::AREA_PARAM_ANGULAR_DAMP:
			set_spOv_angularDump(p_value);
			break;
		case PhysicsServer::AREA_PARAM_PRIORITY:
			set_spOv_priority(p_value);
			break;
		case PhysicsServer::AREA_PARAM_GRAVITY_IS_POINT:
			set_spOv_gravityPoint(p_value);
			break;
		case PhysicsServer::AREA_PARAM_GRAVITY_DISTANCE_SCALE:
			set_spOv_gravityPointDistanceScale(p_value);
			break;
		case PhysicsServer::AREA_PARAM_GRAVITY_POINT_ATTENUATION:
			set_spOv_gravityPointAttenuation(p_value);
			break;
		default:
			WARN_PRINT("Area doesn't support this parameter in the Bullet backend: " + itos(p_param));
	}
	scratch();
}

Variant AreaBullet::get_param(PhysicsServer::AreaParameter p_param) const {
	switch (p_param) {
		case PhysicsServer::AREA_PARAM_GRAVITY:
			return spOv_gravityMag;
		case PhysicsServer::AREA_PARAM_GRAVITY_VECTOR:
			return spOv_gravityVec;
		case PhysicsServer::AREA_PARAM_LINEAR_DAMP:
			return spOv_linearDump;
		case PhysicsServer::AREA_PARAM_ANGULAR_DAMP:
			return spOv_angularDump;
		case PhysicsServer::AREA_PARAM_PRIORITY:
			return spOv_priority;
		case PhysicsServer::AREA_PARAM_GRAVITY_IS_POINT:
			return spOv_gravityPoint;
		case PhysicsServer::AREA_PARAM_GRAVITY_DISTANCE_SCALE:
			return spOv_gravityPointDistanceScale;
		case PhysicsServer::AREA_PARAM_GRAVITY_POINT_ATTENUATION:
			return spOv_gravityPointAttenuation;
		default:
			WARN_PRINT("Area doesn't support this parameter in the Bullet backend: " + itos(p_param));
			return Variant();
	}
}

void AreaBullet::set_event_callback(Type p_callbackObjectType, ObjectID p_id, const StringName &p_method) {
	InOutEventCallback &ev = eventsCallbacks[static_cast<int>(p_callbackObjectType)];
	ev.event_callback_id = p_id;
	ev.event_callback_method = p_method;

	/// Set if monitoring
	if (eventsCallbacks[0].event_callback_id || eventsCallbacks[1].event_callback_id) {
		set_godot_object_flags(get_godot_object_flags() | GOF_IS_MONITORING_AREA);
	} else {
		set_godot_object_flags(get_godot_object_flags() & (~GOF_IS_MONITORING_AREA));
		clear_overlaps(true);
	}
}

bool AreaBullet::has_event_callback(Type p_callbackObjectType) {
	return eventsCallbacks[static_cast<int>(p_callbackObjectType)].event_callback_id;
}

void AreaBullet::on_enter_area(AreaBullet *p_area) {
}

void AreaBullet::on_exit_area(AreaBullet *p_area) {
	CollisionObjectBullet::on_exit_area(p_area);
}

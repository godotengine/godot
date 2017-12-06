/*************************************************************************/
/*  area_bullet.cpp                                                      */
/*  Author: AndreaCatania                                                */
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

#include "area_bullet.h"
#include "BulletCollision/CollisionDispatch/btGhostObject.h"
#include "btBulletCollisionCommon.h"
#include "bullet_types_converter.h"
#include "bullet_utilities.h"
#include "collision_object_bullet.h"
#include "space_bullet.h"

AreaBullet::AreaBullet() :
		RigidCollisionObjectBullet(CollisionObjectBullet::TYPE_AREA),
		monitorable(true),
		isScratched(false),
		spOv_mode(PhysicsServer::AREA_SPACE_OVERRIDE_DISABLED),
		spOv_gravityPoint(false),
		spOv_gravityPointDistanceScale(0),
		spOv_gravityPointAttenuation(1),
		spOv_gravityVec(0, -1, 0),
		spOv_gravityMag(10),
		spOv_linearDump(0.1),
		spOv_angularDump(1),
		spOv_priority(0) {

	btGhost = bulletnew(btGhostObject);
	btGhost->setCollisionShape(compoundShape);
	setupBulletCollisionObject(btGhost);
	/// Collision objects with a callback still have collision response with dynamic rigid bodies.
	/// In order to use collision objects as trigger, you have to disable the collision response.
	set_collision_enabled(false);

	for (int i = 0; i < 5; ++i)
		call_event_res_ptr[i] = &call_event_res[i];
}

AreaBullet::~AreaBullet() {
	remove_all_overlapping_instantly();
}

void AreaBullet::dispatch_callbacks() {
	if (!isScratched)
		return;
	isScratched = false;

	// Reverse order because I've to remove EXIT objects
	for (int i = overlappingObjects.size() - 1; 0 <= i; --i) {
		OverlappingObjectData &otherObj = overlappingObjects[i];

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
	if (isScratched)
		return;
	isScratched = true;
}

void AreaBullet::remove_all_overlapping_instantly() {
	CollisionObjectBullet *supportObject;
	for (int i = overlappingObjects.size() - 1; 0 <= i; --i) {
		supportObject = overlappingObjects[i].object;
		call_event(supportObject, PhysicsServer::AREA_BODY_REMOVED);
		supportObject->on_exit_area(this);
	}
	overlappingObjects.clear();
}

void AreaBullet::remove_overlapping_instantly(CollisionObjectBullet *p_object) {
	CollisionObjectBullet *supportObject;
	for (int i = overlappingObjects.size() - 1; 0 <= i; --i) {
		supportObject = overlappingObjects[i].object;
		if (supportObject == p_object) {
			call_event(supportObject, PhysicsServer::AREA_BODY_REMOVED);
			supportObject->on_exit_area(this);
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
}

bool AreaBullet::is_monitoring() const {
	return get_godot_object_flags() & GOF_IS_MONITORING_AREA;
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
}

void AreaBullet::add_overlap(CollisionObjectBullet *p_otherObject) {
	scratch();
	overlappingObjects.push_back(OverlappingObjectData(p_otherObject, OVERLAP_STATE_ENTER));
	p_otherObject->notify_new_overlap(this);
}

void AreaBullet::put_overlap_as_exit(int p_index) {
	scratch();
	overlappingObjects[p_index].state = OVERLAP_STATE_EXIT;
}

void AreaBullet::put_overlap_as_inside(int p_index) {
	// This check is required to be sure this body was inside
	if (OVERLAP_STATE_DIRTY == overlappingObjects[p_index].state) {
		overlappingObjects[p_index].state = OVERLAP_STATE_INSIDE;
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
			print_line("The Bullet areas dosn't suppot this param: " + itos(p_param));
	}
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
			print_line("The Bullet areas dosn't suppot this param: " + itos(p_param));
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

/*************************************************************************/
/*  area_bullet.cpp                                                      */
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
		overlaps_changed(false),
		is_scratched(false) {
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
	for (uint32_t i = 0; i < overlapping_shapes.size(); i++) {
		overlapping_shapes[i].other_object->on_exit_area(this);
	}
}

void AreaBullet::dispatch_callbacks() {
	if (!is_scratched && !overlaps_changed) {
		return;
	}

	// Reverse order so items can be removed.
	for (int64_t i = int64_t(overlapping_shapes.size()) - 1; i >= 0; i--) {
		OverlappingShapeData &overlapping_shape = overlapping_shapes[i];

		switch (overlapping_shape.state) {
			case OVERLAP_STATE_ENTER: {
				overlapping_shape.state = OVERLAP_STATE_INSIDE;
				call_event(overlapping_shape, PhysicsServer::AREA_BODY_ADDED);

				if (overlapping_shape.overlap_object_data->shape_count == 0) {
					// This object's first shape being added.
					overlapping_shape.other_object->on_enter_area(this);
				}

				overlapping_shape.overlap_object_data->shape_count += 1;
				break;
			}
			case OVERLAP_STATE_EXIT: {
				call_event(overlapping_shape, PhysicsServer::AREA_BODY_REMOVED);
				if (overlapping_shape.overlap_object_data->shape_count == 1) {
					// This object's last shape being removed.
					overlapping_shape.other_object->on_exit_area(this);
					delete overlapping_shape.overlap_object_data;

				} else {
					overlapping_shape.overlap_object_data->shape_count -= 1;
				}

				overlapping_shapes.remove_unordered(i); // Remove after callback
				break;
			}
			case OVERLAP_STATE_INSIDE: {
				if (
						is_scratched &&
						overlapping_shape.other_object->getType() == TYPE_RIGID_BODY) {
					RigidBodyBullet *body = static_cast<RigidBodyBullet *>(overlapping_shape.other_object);
					body->scratch_space_override_modificator();
				}
				break;
			}
			case OVERLAP_STATE_DIRTY:
			default:
				CRASH_NOW_MSG("At this point all the state should be `Enter`, `Exit`, or `Inside`. Otherwise there is a bug.");
		}
	}

	is_scratched = false;
	overlaps_changed = false;
}

void AreaBullet::call_event(const OverlappingShapeData &p_overlapping_shape, PhysicsServer::AreaBodyStatus p_status) {
	InOutEventCallback &event = eventsCallbacks[static_cast<int>(p_overlapping_shape.other_object->getType())];
	Object *areaGodoObject = ObjectDB::get_instance(event.event_callback_id);

	if (!areaGodoObject) {
		event.event_callback_id = 0;
		return;
	}

	call_event_res[0] = p_status;
	call_event_res[1] = p_overlapping_shape.other_object->get_self(); // RID
	call_event_res[2] = p_overlapping_shape.other_object->get_instance_id(); // Object ID
	call_event_res[3] = p_overlapping_shape.other_shape_id; // Other object's shape ID
	call_event_res[4] = p_overlapping_shape.our_shape_id; // This area's shape ID

	Variant::CallError outResp;
	areaGodoObject->call(event.event_callback_method, (const Variant **)call_event_res_ptr, 5, outResp);
}

void AreaBullet::mark_all_overlaps_dirty() {
	for (uint32_t i = 0; i < overlapping_shapes.size(); i++) {
		overlapping_shapes[i].state = OVERLAP_STATE_DIRTY;
	}
}

void AreaBullet::mark_object_overlaps_inside(CollisionObjectBullet *p_other_object) {
	for (uint32_t i = 0; i < overlapping_shapes.size(); i++) {
		if (overlapping_shapes[i].other_object == p_other_object) {
			overlapping_shapes[i].state = OVERLAP_STATE_INSIDE;
		}
	}
}

void AreaBullet::mark_object_shape_overlap_inside(CollisionObjectBullet *p_other_object, uint32_t p_other_shape_id, uint32_t p_our_shape_id) {
	uint32_t overlap_index = UINT32_MAX;
	OverlappingObjectData *overlap_object_data = nullptr;

	// Search the `overlap_index` and fetch the `OverlappingObjectData` (which is used to share common data)
	for (uint32_t i = 0; i < overlapping_shapes.size(); i++) {
		const OverlappingShapeData &overlapping_shape = overlapping_shapes[i];
		if (overlapping_shape.other_object == p_other_object) {
			overlap_object_data = overlapping_shape.overlap_object_data;
			if (overlapping_shape.other_shape_id == p_other_shape_id && overlapping_shape.our_shape_id == p_our_shape_id) {
				overlap_index = i;
				break;
			}
		}
	}

	if (overlap_index == UINT32_MAX) {
		// This is shape started the overlap just now.
		overlap_index = overlapping_shapes.size();
		overlapping_shapes.resize(overlap_index + 1);

		if (overlap_object_data == nullptr) {
			// Initialize a new `overlap_object_data`.
			overlap_object_data = new OverlappingObjectData;
			overlap_object_data->shape_count = 0;
		}

		overlapping_shapes[overlap_index].state = OVERLAP_STATE_ENTER;
		overlapping_shapes[overlap_index].overlap_object_data = overlap_object_data;
		overlapping_shapes[overlap_index].other_object = p_other_object;
		overlapping_shapes[overlap_index].other_shape_id = p_other_shape_id;
		overlapping_shapes[overlap_index].our_shape_id = p_our_shape_id;
		p_other_object->notify_new_overlap(this);
		overlaps_changed = true;
	} else {
		// It was overlapping.
		overlapping_shapes[overlap_index].state = OVERLAP_STATE_INSIDE;
	}
}

void AreaBullet::mark_all_dirty_overlaps_as_exit() {
	for (uint32_t i = 0; i < overlapping_shapes.size(); i++) {
		// Mark all `DIRTY` overlap as exit.
		if (overlapping_shapes[i].state == OVERLAP_STATE_DIRTY) {
			overlapping_shapes[i].state = OVERLAP_STATE_EXIT;
			overlaps_changed = true;
		}
	}
}

void AreaBullet::remove_object_overlaps(CollisionObjectBullet *p_object) {
	// Reverse order so items can be removed.
	for (int64_t i = int64_t(overlapping_shapes.size()) - 1; i >= 0; i--) {
		if (overlapping_shapes[i].other_object == p_object) {
			overlapping_shapes.remove_unordered(i);
		}
	}
}

void AreaBullet::clear_overlaps() {
	for (uint32_t i = 0; i < overlapping_shapes.size(); i++) {
		call_event(overlapping_shapes[i], PhysicsServer::AREA_BODY_REMOVED);
		overlapping_shapes[i].other_object->on_exit_area(this);
		if (overlapping_shapes[i].overlap_object_data->shape_count == 1) {
			delete overlapping_shapes[i].overlap_object_data;
		} else {
			overlapping_shapes[i].overlap_object_data->shape_count -= 1;
		}
	}
	overlapping_shapes.clear();
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
		clear_overlaps();
		is_scratched = false;

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
	is_scratched = true;
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
		clear_overlaps();
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

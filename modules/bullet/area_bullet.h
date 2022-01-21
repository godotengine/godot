/*************************************************************************/
/*  area_bullet.h                                                        */
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

#ifndef AREA_BULLET_H
#define AREA_BULLET_H

#include "collision_object_bullet.h"
#include "core/templates/vector.h"
#include "servers/physics_server_3d.h"
#include "space_bullet.h"

class btGhostObject;

class AreaBullet : public RigidCollisionObjectBullet {
public:
	struct InOutEventCallback {
		Callable event_callback;

		InOutEventCallback() {}
	};

	enum OverlapState {
		OVERLAP_STATE_DIRTY = 0, // Mark processed overlaps
		OVERLAP_STATE_INSIDE, // Mark old overlap
		OVERLAP_STATE_ENTER, // Mark just enter overlap
		OVERLAP_STATE_EXIT // Mark ended overlaps
	};

	struct OverlappingShapeData {
		CollisionObjectBullet *other_object = nullptr;
		OverlapState state = OVERLAP_STATE_DIRTY;
		uint32_t other_shape_id = 0;
		uint32_t our_shape_id = 0;

		OverlappingShapeData() {}

		OverlappingShapeData(CollisionObjectBullet *p_other_object, OverlapState p_state, uint32_t p_other_shape_id, uint32_t p_our_shape_id) :
				other_object(p_other_object),
				state(p_state),
				other_shape_id(p_other_shape_id),
				our_shape_id(p_our_shape_id) {}
	};

private:
	// These are used by function callEvent. Instead to create this each call I create if one time.
	Variant call_event_res[5];
	Variant *call_event_res_ptr[5] = {};

	btGhostObject *btGhost = nullptr;
	Vector<OverlappingShapeData> overlapping_shapes;
	int _overlapping_shape_count(CollisionObjectBullet *p_other_object);
	int _find_overlapping_shape(CollisionObjectBullet *p_other_object, uint32_t p_other_shape_id, uint32_t p_our_shape_id);
	bool monitorable = true;

	PhysicsServer3D::AreaSpaceOverrideMode spOv_mode = PhysicsServer3D::AREA_SPACE_OVERRIDE_DISABLED;
	bool spOv_gravityPoint = false;
	real_t spOv_gravityPointDistanceScale = 0.0;
	real_t spOv_gravityPointAttenuation = 1.0;
	Vector3 spOv_gravityVec = Vector3(0, -1, 0);
	real_t spOv_gravityMag = 10.0;
	real_t spOv_linearDump = 0.1;
	real_t spOv_angularDump = 0.1;
	int spOv_priority = 0;

	bool isScratched = false;

	InOutEventCallback eventsCallbacks[2];

public:
	AreaBullet();
	~AreaBullet();

	_FORCE_INLINE_ btGhostObject *get_bt_ghost() const { return btGhost; }

	void set_monitorable(bool p_monitorable);
	_FORCE_INLINE_ bool is_monitorable() const { return monitorable; }

	bool is_monitoring() const;

	_FORCE_INLINE_ void set_spOv_mode(PhysicsServer3D::AreaSpaceOverrideMode p_mode) { spOv_mode = p_mode; }
	_FORCE_INLINE_ PhysicsServer3D::AreaSpaceOverrideMode get_spOv_mode() { return spOv_mode; }

	_FORCE_INLINE_ void set_spOv_gravityPoint(bool p_isGP) { spOv_gravityPoint = p_isGP; }
	_FORCE_INLINE_ bool is_spOv_gravityPoint() { return spOv_gravityPoint; }

	_FORCE_INLINE_ void set_spOv_gravityPointDistanceScale(real_t p_GPDS) { spOv_gravityPointDistanceScale = p_GPDS; }
	_FORCE_INLINE_ real_t get_spOv_gravityPointDistanceScale() { return spOv_gravityPointDistanceScale; }

	_FORCE_INLINE_ void set_spOv_gravityPointAttenuation(real_t p_GPA) { spOv_gravityPointAttenuation = p_GPA; }
	_FORCE_INLINE_ real_t get_spOv_gravityPointAttenuation() { return spOv_gravityPointAttenuation; }

	_FORCE_INLINE_ void set_spOv_gravityVec(Vector3 p_vec) { spOv_gravityVec = p_vec; }
	_FORCE_INLINE_ const Vector3 &get_spOv_gravityVec() const { return spOv_gravityVec; }

	_FORCE_INLINE_ void set_spOv_gravityMag(real_t p_gravityMag) { spOv_gravityMag = p_gravityMag; }
	_FORCE_INLINE_ real_t get_spOv_gravityMag() { return spOv_gravityMag; }

	_FORCE_INLINE_ void set_spOv_linearDump(real_t p_linearDump) { spOv_linearDump = p_linearDump; }
	_FORCE_INLINE_ real_t get_spOv_linearDamp() { return spOv_linearDump; }

	_FORCE_INLINE_ void set_spOv_angularDump(real_t p_angularDump) { spOv_angularDump = p_angularDump; }
	_FORCE_INLINE_ real_t get_spOv_angularDamp() { return spOv_angularDump; }

	_FORCE_INLINE_ void set_spOv_priority(int p_priority) { spOv_priority = p_priority; }
	_FORCE_INLINE_ int get_spOv_priority() { return spOv_priority; }

	virtual void main_shape_changed();
	virtual void reload_body();
	virtual void set_space(SpaceBullet *p_space);

	virtual void dispatch_callbacks();
	void call_event(const OverlappingShapeData &p_overlapping_shape, PhysicsServer3D::AreaBodyStatus p_status);

	virtual void on_collision_filters_change();
	virtual void on_collision_checker_start() {}
	virtual void on_collision_checker_end() { updated = false; }

	void mark_all_overlaps_dirty();
	void mark_object_overlaps_inside(CollisionObjectBullet *p_other_object);
	void set_overlap(CollisionObjectBullet *p_other_object, uint32_t p_other_shape_id, uint32_t p_our_shape_id);
	void mark_all_dirty_overlaps_as_exit();
	void remove_object_overlaps(CollisionObjectBullet *p_object);
	void clear_overlaps();

	void set_param(PhysicsServer3D::AreaParameter p_param, const Variant &p_value);
	Variant get_param(PhysicsServer3D::AreaParameter p_param) const;

	void set_event_callback(Type p_callbackObjectType, const Callable &p_callback);
	bool has_event_callback(Type p_callbackObjectType);

	virtual void on_enter_area(AreaBullet *p_area);
	virtual void on_exit_area(AreaBullet *p_area);
};

#endif // AREA_BULLET_H

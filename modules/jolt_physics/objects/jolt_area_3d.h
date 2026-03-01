/**************************************************************************/
/*  jolt_area_3d.h                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "jolt_shaped_object_3d.h"

#include "servers/physics_3d/physics_server_3d.h"

class JoltBody3D;
class JoltSoftBody3D;

class JoltArea3D final : public JoltShapedObject3D {
public:
	typedef PhysicsServer3D::AreaSpaceOverrideMode OverrideMode;

private:
	struct BodyIDHasher {
		static uint32_t hash(const JPH::BodyID &p_id) { return hash_fmix32(p_id.GetIndexAndSequenceNumber()); }
	};

	struct ShapeIDPair {
		JPH::SubShapeID other;
		JPH::SubShapeID self;

		ShapeIDPair(JPH::SubShapeID p_other, JPH::SubShapeID p_self) :
				other(p_other), self(p_self) {}

		static uint32_t hash(const ShapeIDPair &p_pair) {
			uint32_t hash = hash_murmur3_one_32(p_pair.other.GetValue());
			hash = hash_murmur3_one_32(p_pair.self.GetValue(), hash);
			return hash_fmix32(hash);
		}

		friend bool operator==(const ShapeIDPair &p_lhs, const ShapeIDPair &p_rhs) {
			return (p_lhs.other == p_rhs.other) && (p_lhs.self == p_rhs.self);
		}
	};

	struct ShapeIndexPair {
		int other = -1;
		int self = -1;

		ShapeIndexPair() = default;

		ShapeIndexPair(int p_other, int p_self) :
				other(p_other), self(p_self) {}

		static uint32_t hash(const ShapeIndexPair &p_pair) {
			uint32_t hash = hash_murmur3_one_32(p_pair.other);
			hash = hash_murmur3_one_32(p_pair.self, hash);
			return hash_fmix32(hash);
		}

		friend bool operator==(const ShapeIndexPair &p_lhs, const ShapeIndexPair &p_rhs) {
			return (p_lhs.other == p_rhs.other) && (p_lhs.self == p_rhs.self);
		}
	};

	struct Overlap {
		HashMap<ShapeIDPair, ShapeIndexPair, ShapeIDPair> shape_pairs;
		HashMap<ShapeIndexPair, int, ShapeIndexPair> ref_counts;
		LocalVector<ShapeIndexPair> pending_added;
		LocalVector<ShapeIndexPair> pending_removed;
		RID rid;
		ObjectID instance_id;
	};

	typedef HashMap<JPH::BodyID, Overlap, BodyIDHasher> OverlapsById;

	SelfList<JoltArea3D> call_queries_element;

	OverlapsById bodies_by_id;
	OverlapsById areas_by_id;

	Vector3 gravity_vector = Vector3(0, -1, 0);
	Vector3 wind_source;
	Vector3 wind_direction;

	Callable body_monitor_callback;
	Callable area_monitor_callback;

	float priority = 0.0f;
	float gravity = 9.8f;
	float point_gravity_distance = 0.0f;
	float linear_damp = 0.1f;
	float angular_damp = 0.1f;
	float wind_pressure = 0.0f;
	float wind_attenuation_factor = 0.0f;

	OverrideMode gravity_mode = PhysicsServer3D::AREA_SPACE_OVERRIDE_DISABLED;
	OverrideMode linear_damp_mode = PhysicsServer3D::AREA_SPACE_OVERRIDE_DISABLED;
	OverrideMode angular_damp_mode = PhysicsServer3D::AREA_SPACE_OVERRIDE_DISABLED;

	bool monitorable = false;
	bool point_gravity = false;

	virtual JPH::BroadPhaseLayer _get_broad_phase_layer() const override;
	virtual JPH::ObjectLayer _get_object_layer() const override;

	virtual JPH::EMotionType _get_motion_type() const override { return JPH::EMotionType::Kinematic; }

	bool _should_sleep() const { return !is_monitoring(); }

	virtual void _add_to_space() override;

	void _enqueue_call_queries();
	void _dequeue_call_queries();

	void _add_shape_pair(Overlap &p_overlap, const JPH::BodyID &p_body_id, const JPH::SubShapeID &p_other_shape_id, const JPH::SubShapeID &p_self_shape_id);
	bool _remove_shape_pair(Overlap &p_overlap, const JPH::SubShapeID &p_other_shape_id, const JPH::SubShapeID &p_self_shape_id);

	void _flush_events(OverlapsById &p_objects, const Callable &p_callback);

	void _report_event(const Callable &p_callback, PhysicsServer3D::AreaBodyStatus p_status, const RID &p_other_rid, ObjectID p_other_instance_id, int p_other_shape_index, int p_self_shape_index) const;

	void _notify_body_entered(const JPH::BodyID &p_body_id);
	void _notify_body_exited(const JPH::BodyID &p_body_id);

	void _remove_all_overlaps();

	void _update_sleeping();
	void _update_group_filter();

	virtual void _space_changing() override;
	virtual void _space_changed() override;
	void _events_changed();
	void _body_monitoring_changed();
	void _area_monitoring_changed();
	void _monitorable_changed();

public:
	JoltArea3D();

	void set_transform(Transform3D p_transform);

	Variant get_param(PhysicsServer3D::AreaParameter p_param) const;
	void set_param(PhysicsServer3D::AreaParameter p_param, const Variant &p_value);

	bool has_body_monitor_callback() const { return body_monitor_callback.is_valid(); }
	void set_body_monitor_callback(const Callable &p_callback);

	bool has_area_monitor_callback() const { return area_monitor_callback.is_valid(); }
	void set_area_monitor_callback(const Callable &p_callback);

	bool is_monitoring_bodies() const { return has_body_monitor_callback(); }
	bool is_monitoring_areas() const { return has_area_monitor_callback(); }
	bool is_monitoring() const { return is_monitoring_bodies() || is_monitoring_areas(); }

	bool is_monitorable() const { return monitorable; }
	void set_monitorable(bool p_monitorable);

	bool can_monitor(const JoltBody3D &p_other) const;
	bool can_monitor(const JoltSoftBody3D &p_other) const;
	bool can_monitor(const JoltArea3D &p_other) const;

	virtual bool can_interact_with(const JoltBody3D &p_other) const override;
	virtual bool can_interact_with(const JoltSoftBody3D &p_other) const override;
	virtual bool can_interact_with(const JoltArea3D &p_other) const override;

	virtual Vector3 get_velocity_at_position(const Vector3 &p_position) const override { return Vector3(); }

	virtual bool reports_contacts() const override { return false; }

	bool is_point_gravity() const { return point_gravity; }
	void set_point_gravity(bool p_enabled) { point_gravity = p_enabled; }

	float get_priority() const { return priority; }
	void set_priority(float p_priority) { priority = p_priority; }

	float get_gravity() const { return gravity; }
	void set_gravity(float p_gravity) { gravity = p_gravity; }

	float get_point_gravity_distance() const { return point_gravity_distance; }
	void set_point_gravity_distance(float p_distance) { point_gravity_distance = p_distance; }

	float get_linear_damp() const { return linear_damp; }
	void set_area_linear_damp(float p_damp) { linear_damp = p_damp; }

	float get_angular_damp() const { return angular_damp; }
	void set_area_angular_damp(float p_damp) { angular_damp = p_damp; }

	OverrideMode get_gravity_mode() const { return gravity_mode; }
	void set_gravity_mode(OverrideMode p_mode) { gravity_mode = p_mode; }

	OverrideMode get_linear_damp_mode() const { return linear_damp_mode; }
	void set_linear_damp_mode(OverrideMode p_mode) { linear_damp_mode = p_mode; }

	OverrideMode get_angular_damp_mode() const { return angular_damp_mode; }
	void set_angular_damp_mode(OverrideMode p_mode) { angular_damp_mode = p_mode; }

	Vector3 get_gravity_vector() const { return gravity_vector; }
	void set_gravity_vector(const Vector3 &p_vector) { gravity_vector = p_vector; }

	float get_wind_pressure() const { return wind_pressure; }
	void set_wind_pressure(float p_wind_pressure) { wind_pressure = p_wind_pressure; }

	float get_wind_attenuation_factor() const { return wind_attenuation_factor; }
	void set_wind_attenuation_factor(float p_wind_attenuation_factor) { wind_attenuation_factor = p_wind_attenuation_factor; }

	const Vector3 &get_wind_source() const { return wind_source; }
	void set_wind_source(const Vector3 &p_wind_source) { wind_source = p_wind_source; }

	const Vector3 &get_wind_direction() const { return wind_direction; }
	void set_wind_direction(const Vector3 &p_wind_direction) { wind_direction = p_wind_direction; }

	Vector3 compute_gravity(const Vector3 &p_position) const;

	void body_shape_entered(const JPH::BodyID &p_body_id, const JPH::SubShapeID &p_other_shape_id, const JPH::SubShapeID &p_self_shape_id);
	bool body_shape_exited(const JPH::BodyID &p_body_id, const JPH::SubShapeID &p_other_shape_id, const JPH::SubShapeID &p_self_shape_id);

	void area_shape_entered(const JPH::BodyID &p_body_id, const JPH::SubShapeID &p_other_shape_id, const JPH::SubShapeID &p_self_shape_id);
	bool area_shape_exited(const JPH::BodyID &p_body_id, const JPH::SubShapeID &p_other_shape_id, const JPH::SubShapeID &p_self_shape_id);

	bool shape_exited(const JPH::BodyID &p_body_id, const JPH::SubShapeID &p_other_shape_id, const JPH::SubShapeID &p_self_shape_id);
	void body_exited(const JPH::BodyID &p_body_id, bool p_notify = true);
	void area_exited(const JPH::BodyID &p_body_id);

	void call_queries();

	virtual bool has_custom_center_of_mass() const override { return false; }
	virtual Vector3 get_center_of_mass_custom() const override { return Vector3(); }

	// Incorporates the value provided by `p_getter` into `p_value` according to the override mode `p_mode`.
	// Returns true if further calls to this function should stop (i.e. value has been replaced entirely).
	template <typename TValue, typename TGetter>
	static bool apply_override(TValue &p_value, PhysicsServer3D::AreaSpaceOverrideMode p_mode, TGetter &&p_getter);
};

template <typename TValue, typename TGetter>
inline bool JoltArea3D::apply_override(TValue &p_value, PhysicsServer3D::AreaSpaceOverrideMode p_mode, TGetter &&p_getter) {
	switch (p_mode) {
		case PhysicsServer3D::AREA_SPACE_OVERRIDE_DISABLED: {
			return false;
		}
		case PhysicsServer3D::AREA_SPACE_OVERRIDE_COMBINE: {
			p_value += p_getter();
			return false;
		}
		case PhysicsServer3D::AREA_SPACE_OVERRIDE_COMBINE_REPLACE: {
			p_value += p_getter();
			return true;
		}
		case PhysicsServer3D::AREA_SPACE_OVERRIDE_REPLACE: {
			p_value = p_getter();
			return true;
		}
		case PhysicsServer3D::AREA_SPACE_OVERRIDE_REPLACE_COMBINE: {
			p_value = p_getter();
			return false;
		}
		default: {
			ERR_FAIL_V_MSG(false, vformat("Unhandled override mode: '%d'. This should not happen. Please report this.", p_mode));
		}
	}
}

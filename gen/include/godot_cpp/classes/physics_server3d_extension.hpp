/**************************************************************************/
/*  physics_server3d_extension.hpp                                        */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/physics_server3d.hpp>
#include <godot_cpp/classes/physics_server3d_extension_motion_result.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class PhysicsDirectBodyState3D;
class PhysicsDirectSpaceState3D;
class PhysicsServer3DRenderingServerHandler;

class PhysicsServer3DExtension : public PhysicsServer3D {
	GDEXTENSION_CLASS(PhysicsServer3DExtension, PhysicsServer3D)

public:
	bool body_test_motion_is_excluding_body(const RID &p_body) const;
	bool body_test_motion_is_excluding_object(uint64_t p_object) const;
	virtual RID _world_boundary_shape_create();
	virtual RID _separation_ray_shape_create();
	virtual RID _sphere_shape_create();
	virtual RID _box_shape_create();
	virtual RID _capsule_shape_create();
	virtual RID _cylinder_shape_create();
	virtual RID _convex_polygon_shape_create();
	virtual RID _concave_polygon_shape_create();
	virtual RID _heightmap_shape_create();
	virtual RID _custom_shape_create();
	virtual void _shape_set_data(const RID &p_shape, const Variant &p_data);
	virtual void _shape_set_custom_solver_bias(const RID &p_shape, float p_bias);
	virtual void _shape_set_margin(const RID &p_shape, float p_margin);
	virtual float _shape_get_margin(const RID &p_shape) const;
	virtual PhysicsServer3D::ShapeType _shape_get_type(const RID &p_shape) const;
	virtual Variant _shape_get_data(const RID &p_shape) const;
	virtual float _shape_get_custom_solver_bias(const RID &p_shape) const;
	virtual RID _space_create();
	virtual void _space_set_active(const RID &p_space, bool p_active);
	virtual bool _space_is_active(const RID &p_space) const;
	virtual void _space_set_param(const RID &p_space, PhysicsServer3D::SpaceParameter p_param, float p_value);
	virtual float _space_get_param(const RID &p_space, PhysicsServer3D::SpaceParameter p_param) const;
	virtual PhysicsDirectSpaceState3D *_space_get_direct_state(const RID &p_space);
	virtual void _space_set_debug_contacts(const RID &p_space, int32_t p_max_contacts);
	virtual PackedVector3Array _space_get_contacts(const RID &p_space) const;
	virtual int32_t _space_get_contact_count(const RID &p_space) const;
	virtual RID _area_create();
	virtual void _area_set_space(const RID &p_area, const RID &p_space);
	virtual RID _area_get_space(const RID &p_area) const;
	virtual void _area_add_shape(const RID &p_area, const RID &p_shape, const Transform3D &p_transform, bool p_disabled);
	virtual void _area_set_shape(const RID &p_area, int32_t p_shape_idx, const RID &p_shape);
	virtual void _area_set_shape_transform(const RID &p_area, int32_t p_shape_idx, const Transform3D &p_transform);
	virtual void _area_set_shape_disabled(const RID &p_area, int32_t p_shape_idx, bool p_disabled);
	virtual int32_t _area_get_shape_count(const RID &p_area) const;
	virtual RID _area_get_shape(const RID &p_area, int32_t p_shape_idx) const;
	virtual Transform3D _area_get_shape_transform(const RID &p_area, int32_t p_shape_idx) const;
	virtual void _area_remove_shape(const RID &p_area, int32_t p_shape_idx);
	virtual void _area_clear_shapes(const RID &p_area);
	virtual void _area_attach_object_instance_id(const RID &p_area, uint64_t p_id);
	virtual uint64_t _area_get_object_instance_id(const RID &p_area) const;
	virtual void _area_set_param(const RID &p_area, PhysicsServer3D::AreaParameter p_param, const Variant &p_value);
	virtual void _area_set_transform(const RID &p_area, const Transform3D &p_transform);
	virtual Variant _area_get_param(const RID &p_area, PhysicsServer3D::AreaParameter p_param) const;
	virtual Transform3D _area_get_transform(const RID &p_area) const;
	virtual void _area_set_collision_layer(const RID &p_area, uint32_t p_layer);
	virtual uint32_t _area_get_collision_layer(const RID &p_area) const;
	virtual void _area_set_collision_mask(const RID &p_area, uint32_t p_mask);
	virtual uint32_t _area_get_collision_mask(const RID &p_area) const;
	virtual void _area_set_monitorable(const RID &p_area, bool p_monitorable);
	virtual void _area_set_ray_pickable(const RID &p_area, bool p_enable);
	virtual void _area_set_monitor_callback(const RID &p_area, const Callable &p_callback);
	virtual void _area_set_area_monitor_callback(const RID &p_area, const Callable &p_callback);
	virtual RID _body_create();
	virtual void _body_set_space(const RID &p_body, const RID &p_space);
	virtual RID _body_get_space(const RID &p_body) const;
	virtual void _body_set_mode(const RID &p_body, PhysicsServer3D::BodyMode p_mode);
	virtual PhysicsServer3D::BodyMode _body_get_mode(const RID &p_body) const;
	virtual void _body_add_shape(const RID &p_body, const RID &p_shape, const Transform3D &p_transform, bool p_disabled);
	virtual void _body_set_shape(const RID &p_body, int32_t p_shape_idx, const RID &p_shape);
	virtual void _body_set_shape_transform(const RID &p_body, int32_t p_shape_idx, const Transform3D &p_transform);
	virtual void _body_set_shape_disabled(const RID &p_body, int32_t p_shape_idx, bool p_disabled);
	virtual int32_t _body_get_shape_count(const RID &p_body) const;
	virtual RID _body_get_shape(const RID &p_body, int32_t p_shape_idx) const;
	virtual Transform3D _body_get_shape_transform(const RID &p_body, int32_t p_shape_idx) const;
	virtual void _body_remove_shape(const RID &p_body, int32_t p_shape_idx);
	virtual void _body_clear_shapes(const RID &p_body);
	virtual void _body_attach_object_instance_id(const RID &p_body, uint64_t p_id);
	virtual uint64_t _body_get_object_instance_id(const RID &p_body) const;
	virtual void _body_set_enable_continuous_collision_detection(const RID &p_body, bool p_enable);
	virtual bool _body_is_continuous_collision_detection_enabled(const RID &p_body) const;
	virtual void _body_set_collision_layer(const RID &p_body, uint32_t p_layer);
	virtual uint32_t _body_get_collision_layer(const RID &p_body) const;
	virtual void _body_set_collision_mask(const RID &p_body, uint32_t p_mask);
	virtual uint32_t _body_get_collision_mask(const RID &p_body) const;
	virtual void _body_set_collision_priority(const RID &p_body, float p_priority);
	virtual float _body_get_collision_priority(const RID &p_body) const;
	virtual void _body_set_user_flags(const RID &p_body, uint32_t p_flags);
	virtual uint32_t _body_get_user_flags(const RID &p_body) const;
	virtual void _body_set_param(const RID &p_body, PhysicsServer3D::BodyParameter p_param, const Variant &p_value);
	virtual Variant _body_get_param(const RID &p_body, PhysicsServer3D::BodyParameter p_param) const;
	virtual void _body_reset_mass_properties(const RID &p_body);
	virtual void _body_set_state(const RID &p_body, PhysicsServer3D::BodyState p_state, const Variant &p_value);
	virtual Variant _body_get_state(const RID &p_body, PhysicsServer3D::BodyState p_state) const;
	virtual void _body_apply_central_impulse(const RID &p_body, const Vector3 &p_impulse);
	virtual void _body_apply_impulse(const RID &p_body, const Vector3 &p_impulse, const Vector3 &p_position);
	virtual void _body_apply_torque_impulse(const RID &p_body, const Vector3 &p_impulse);
	virtual void _body_apply_central_force(const RID &p_body, const Vector3 &p_force);
	virtual void _body_apply_force(const RID &p_body, const Vector3 &p_force, const Vector3 &p_position);
	virtual void _body_apply_torque(const RID &p_body, const Vector3 &p_torque);
	virtual void _body_add_constant_central_force(const RID &p_body, const Vector3 &p_force);
	virtual void _body_add_constant_force(const RID &p_body, const Vector3 &p_force, const Vector3 &p_position);
	virtual void _body_add_constant_torque(const RID &p_body, const Vector3 &p_torque);
	virtual void _body_set_constant_force(const RID &p_body, const Vector3 &p_force);
	virtual Vector3 _body_get_constant_force(const RID &p_body) const;
	virtual void _body_set_constant_torque(const RID &p_body, const Vector3 &p_torque);
	virtual Vector3 _body_get_constant_torque(const RID &p_body) const;
	virtual void _body_set_axis_velocity(const RID &p_body, const Vector3 &p_axis_velocity);
	virtual void _body_set_axis_lock(const RID &p_body, PhysicsServer3D::BodyAxis p_axis, bool p_lock);
	virtual bool _body_is_axis_locked(const RID &p_body, PhysicsServer3D::BodyAxis p_axis) const;
	virtual void _body_add_collision_exception(const RID &p_body, const RID &p_excepted_body);
	virtual void _body_remove_collision_exception(const RID &p_body, const RID &p_excepted_body);
	virtual TypedArray<RID> _body_get_collision_exceptions(const RID &p_body) const;
	virtual void _body_set_max_contacts_reported(const RID &p_body, int32_t p_amount);
	virtual int32_t _body_get_max_contacts_reported(const RID &p_body) const;
	virtual void _body_set_contacts_reported_depth_threshold(const RID &p_body, float p_threshold);
	virtual float _body_get_contacts_reported_depth_threshold(const RID &p_body) const;
	virtual void _body_set_omit_force_integration(const RID &p_body, bool p_enable);
	virtual bool _body_is_omitting_force_integration(const RID &p_body) const;
	virtual void _body_set_state_sync_callback(const RID &p_body, const Callable &p_callable);
	virtual void _body_set_force_integration_callback(const RID &p_body, const Callable &p_callable, const Variant &p_userdata);
	virtual void _body_set_ray_pickable(const RID &p_body, bool p_enable);
	virtual bool _body_test_motion(const RID &p_body, const Transform3D &p_from, const Vector3 &p_motion, float p_margin, int32_t p_max_collisions, bool p_collide_separation_ray, bool p_recovery_as_collision, PhysicsServer3DExtensionMotionResult *p_result) const;
	virtual PhysicsDirectBodyState3D *_body_get_direct_state(const RID &p_body);
	virtual RID _soft_body_create();
	virtual void _soft_body_update_rendering_server(const RID &p_body, PhysicsServer3DRenderingServerHandler *p_rendering_server_handler);
	virtual void _soft_body_set_space(const RID &p_body, const RID &p_space);
	virtual RID _soft_body_get_space(const RID &p_body) const;
	virtual void _soft_body_set_ray_pickable(const RID &p_body, bool p_enable);
	virtual void _soft_body_set_collision_layer(const RID &p_body, uint32_t p_layer);
	virtual uint32_t _soft_body_get_collision_layer(const RID &p_body) const;
	virtual void _soft_body_set_collision_mask(const RID &p_body, uint32_t p_mask);
	virtual uint32_t _soft_body_get_collision_mask(const RID &p_body) const;
	virtual void _soft_body_add_collision_exception(const RID &p_body, const RID &p_body_b);
	virtual void _soft_body_remove_collision_exception(const RID &p_body, const RID &p_body_b);
	virtual TypedArray<RID> _soft_body_get_collision_exceptions(const RID &p_body) const;
	virtual void _soft_body_set_state(const RID &p_body, PhysicsServer3D::BodyState p_state, const Variant &p_variant);
	virtual Variant _soft_body_get_state(const RID &p_body, PhysicsServer3D::BodyState p_state) const;
	virtual void _soft_body_set_transform(const RID &p_body, const Transform3D &p_transform);
	virtual void _soft_body_set_simulation_precision(const RID &p_body, int32_t p_simulation_precision);
	virtual int32_t _soft_body_get_simulation_precision(const RID &p_body) const;
	virtual void _soft_body_set_total_mass(const RID &p_body, float p_total_mass);
	virtual float _soft_body_get_total_mass(const RID &p_body) const;
	virtual void _soft_body_set_linear_stiffness(const RID &p_body, float p_linear_stiffness);
	virtual float _soft_body_get_linear_stiffness(const RID &p_body) const;
	virtual void _soft_body_set_shrinking_factor(const RID &p_body, float p_shrinking_factor);
	virtual float _soft_body_get_shrinking_factor(const RID &p_body) const;
	virtual void _soft_body_set_pressure_coefficient(const RID &p_body, float p_pressure_coefficient);
	virtual float _soft_body_get_pressure_coefficient(const RID &p_body) const;
	virtual void _soft_body_set_damping_coefficient(const RID &p_body, float p_damping_coefficient);
	virtual float _soft_body_get_damping_coefficient(const RID &p_body) const;
	virtual void _soft_body_set_drag_coefficient(const RID &p_body, float p_drag_coefficient);
	virtual float _soft_body_get_drag_coefficient(const RID &p_body) const;
	virtual void _soft_body_set_mesh(const RID &p_body, const RID &p_mesh);
	virtual AABB _soft_body_get_bounds(const RID &p_body) const;
	virtual void _soft_body_move_point(const RID &p_body, int32_t p_point_index, const Vector3 &p_global_position);
	virtual Vector3 _soft_body_get_point_global_position(const RID &p_body, int32_t p_point_index) const;
	virtual void _soft_body_remove_all_pinned_points(const RID &p_body);
	virtual void _soft_body_pin_point(const RID &p_body, int32_t p_point_index, bool p_pin);
	virtual bool _soft_body_is_point_pinned(const RID &p_body, int32_t p_point_index) const;
	virtual void _soft_body_apply_point_impulse(const RID &p_body, int32_t p_point_index, const Vector3 &p_impulse);
	virtual void _soft_body_apply_point_force(const RID &p_body, int32_t p_point_index, const Vector3 &p_force);
	virtual void _soft_body_apply_central_impulse(const RID &p_body, const Vector3 &p_impulse);
	virtual void _soft_body_apply_central_force(const RID &p_body, const Vector3 &p_force);
	virtual RID _joint_create();
	virtual void _joint_clear(const RID &p_joint);
	virtual void _joint_make_pin(const RID &p_joint, const RID &p_body_A, const Vector3 &p_local_A, const RID &p_body_B, const Vector3 &p_local_B);
	virtual void _pin_joint_set_param(const RID &p_joint, PhysicsServer3D::PinJointParam p_param, float p_value);
	virtual float _pin_joint_get_param(const RID &p_joint, PhysicsServer3D::PinJointParam p_param) const;
	virtual void _pin_joint_set_local_a(const RID &p_joint, const Vector3 &p_local_A);
	virtual Vector3 _pin_joint_get_local_a(const RID &p_joint) const;
	virtual void _pin_joint_set_local_b(const RID &p_joint, const Vector3 &p_local_B);
	virtual Vector3 _pin_joint_get_local_b(const RID &p_joint) const;
	virtual void _joint_make_hinge(const RID &p_joint, const RID &p_body_A, const Transform3D &p_hinge_A, const RID &p_body_B, const Transform3D &p_hinge_B);
	virtual void _joint_make_hinge_simple(const RID &p_joint, const RID &p_body_A, const Vector3 &p_pivot_A, const Vector3 &p_axis_A, const RID &p_body_B, const Vector3 &p_pivot_B, const Vector3 &p_axis_B);
	virtual void _hinge_joint_set_param(const RID &p_joint, PhysicsServer3D::HingeJointParam p_param, float p_value);
	virtual float _hinge_joint_get_param(const RID &p_joint, PhysicsServer3D::HingeJointParam p_param) const;
	virtual void _hinge_joint_set_flag(const RID &p_joint, PhysicsServer3D::HingeJointFlag p_flag, bool p_enabled);
	virtual bool _hinge_joint_get_flag(const RID &p_joint, PhysicsServer3D::HingeJointFlag p_flag) const;
	virtual void _joint_make_slider(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B);
	virtual void _slider_joint_set_param(const RID &p_joint, PhysicsServer3D::SliderJointParam p_param, float p_value);
	virtual float _slider_joint_get_param(const RID &p_joint, PhysicsServer3D::SliderJointParam p_param) const;
	virtual void _joint_make_cone_twist(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B);
	virtual void _cone_twist_joint_set_param(const RID &p_joint, PhysicsServer3D::ConeTwistJointParam p_param, float p_value);
	virtual float _cone_twist_joint_get_param(const RID &p_joint, PhysicsServer3D::ConeTwistJointParam p_param) const;
	virtual void _joint_make_generic_6dof(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B);
	virtual void _generic_6dof_joint_set_param(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param, float p_value);
	virtual float _generic_6dof_joint_get_param(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param) const;
	virtual void _generic_6dof_joint_set_flag(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag, bool p_enable);
	virtual bool _generic_6dof_joint_get_flag(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag) const;
	virtual PhysicsServer3D::JointType _joint_get_type(const RID &p_joint) const;
	virtual void _joint_set_solver_priority(const RID &p_joint, int32_t p_priority);
	virtual int32_t _joint_get_solver_priority(const RID &p_joint) const;
	virtual void _joint_disable_collisions_between_bodies(const RID &p_joint, bool p_disable);
	virtual bool _joint_is_disabled_collisions_between_bodies(const RID &p_joint) const;
	virtual void _free_rid(const RID &p_rid);
	virtual void _set_active(bool p_active);
	virtual void _init();
	virtual void _step(float p_step);
	virtual void _sync();
	virtual void _flush_queries();
	virtual void _end_sync();
	virtual void _finish();
	virtual bool _is_flushing_queries() const;
	virtual int32_t _get_process_info(PhysicsServer3D::ProcessInfo p_process_info);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PhysicsServer3D::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_world_boundary_shape_create), decltype(&T::_world_boundary_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _world_boundary_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_separation_ray_shape_create), decltype(&T::_separation_ray_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _separation_ray_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_sphere_shape_create), decltype(&T::_sphere_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _sphere_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_box_shape_create), decltype(&T::_box_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _box_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_capsule_shape_create), decltype(&T::_capsule_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _capsule_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_cylinder_shape_create), decltype(&T::_cylinder_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _cylinder_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_convex_polygon_shape_create), decltype(&T::_convex_polygon_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _convex_polygon_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_concave_polygon_shape_create), decltype(&T::_concave_polygon_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _concave_polygon_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_heightmap_shape_create), decltype(&T::_heightmap_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _heightmap_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_custom_shape_create), decltype(&T::_custom_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _custom_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_set_data), decltype(&T::_shape_set_data)>) {
			BIND_VIRTUAL_METHOD(T, _shape_set_data, 3175752987);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_set_custom_solver_bias), decltype(&T::_shape_set_custom_solver_bias)>) {
			BIND_VIRTUAL_METHOD(T, _shape_set_custom_solver_bias, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_set_margin), decltype(&T::_shape_set_margin)>) {
			BIND_VIRTUAL_METHOD(T, _shape_set_margin, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_get_margin), decltype(&T::_shape_get_margin)>) {
			BIND_VIRTUAL_METHOD(T, _shape_get_margin, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_get_type), decltype(&T::_shape_get_type)>) {
			BIND_VIRTUAL_METHOD(T, _shape_get_type, 3418923367);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_get_data), decltype(&T::_shape_get_data)>) {
			BIND_VIRTUAL_METHOD(T, _shape_get_data, 4171304767);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_get_custom_solver_bias), decltype(&T::_shape_get_custom_solver_bias)>) {
			BIND_VIRTUAL_METHOD(T, _shape_get_custom_solver_bias, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_create), decltype(&T::_space_create)>) {
			BIND_VIRTUAL_METHOD(T, _space_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_set_active), decltype(&T::_space_set_active)>) {
			BIND_VIRTUAL_METHOD(T, _space_set_active, 1265174801);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_is_active), decltype(&T::_space_is_active)>) {
			BIND_VIRTUAL_METHOD(T, _space_is_active, 4155700596);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_set_param), decltype(&T::_space_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _space_set_param, 2406017470);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_get_param), decltype(&T::_space_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _space_get_param, 1523206731);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_get_direct_state), decltype(&T::_space_get_direct_state)>) {
			BIND_VIRTUAL_METHOD(T, _space_get_direct_state, 2048616813);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_set_debug_contacts), decltype(&T::_space_set_debug_contacts)>) {
			BIND_VIRTUAL_METHOD(T, _space_set_debug_contacts, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_get_contacts), decltype(&T::_space_get_contacts)>) {
			BIND_VIRTUAL_METHOD(T, _space_get_contacts, 808965560);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_get_contact_count), decltype(&T::_space_get_contact_count)>) {
			BIND_VIRTUAL_METHOD(T, _space_get_contact_count, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_create), decltype(&T::_area_create)>) {
			BIND_VIRTUAL_METHOD(T, _area_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_space), decltype(&T::_area_set_space)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_space, 395945892);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_space), decltype(&T::_area_get_space)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_space, 3814569979);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_add_shape), decltype(&T::_area_add_shape)>) {
			BIND_VIRTUAL_METHOD(T, _area_add_shape, 2153848567);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_shape), decltype(&T::_area_set_shape)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_shape, 2310537182);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_shape_transform), decltype(&T::_area_set_shape_transform)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_shape_transform, 675327471);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_shape_disabled), decltype(&T::_area_set_shape_disabled)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_shape_disabled, 2658558584);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_shape_count), decltype(&T::_area_get_shape_count)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_shape_count, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_shape), decltype(&T::_area_get_shape)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_shape, 1066463050);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_shape_transform), decltype(&T::_area_get_shape_transform)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_shape_transform, 1050775521);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_remove_shape), decltype(&T::_area_remove_shape)>) {
			BIND_VIRTUAL_METHOD(T, _area_remove_shape, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_clear_shapes), decltype(&T::_area_clear_shapes)>) {
			BIND_VIRTUAL_METHOD(T, _area_clear_shapes, 2722037293);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_attach_object_instance_id), decltype(&T::_area_attach_object_instance_id)>) {
			BIND_VIRTUAL_METHOD(T, _area_attach_object_instance_id, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_object_instance_id), decltype(&T::_area_get_object_instance_id)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_object_instance_id, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_param), decltype(&T::_area_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_param, 2980114638);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_transform), decltype(&T::_area_set_transform)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_transform, 3935195649);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_param), decltype(&T::_area_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_param, 890056067);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_transform), decltype(&T::_area_get_transform)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_transform, 1128465797);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_collision_layer), decltype(&T::_area_set_collision_layer)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_collision_layer, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_collision_layer), decltype(&T::_area_get_collision_layer)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_collision_layer, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_collision_mask), decltype(&T::_area_set_collision_mask)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_collision_mask, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_collision_mask), decltype(&T::_area_get_collision_mask)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_collision_mask, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_monitorable), decltype(&T::_area_set_monitorable)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_monitorable, 1265174801);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_ray_pickable), decltype(&T::_area_set_ray_pickable)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_ray_pickable, 1265174801);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_monitor_callback), decltype(&T::_area_set_monitor_callback)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_monitor_callback, 3379118538);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_area_monitor_callback), decltype(&T::_area_set_area_monitor_callback)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_area_monitor_callback, 3379118538);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_create), decltype(&T::_body_create)>) {
			BIND_VIRTUAL_METHOD(T, _body_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_space), decltype(&T::_body_set_space)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_space, 395945892);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_space), decltype(&T::_body_get_space)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_space, 3814569979);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_mode), decltype(&T::_body_set_mode)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_mode, 606803466);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_mode), decltype(&T::_body_get_mode)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_mode, 2488819728);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_add_shape), decltype(&T::_body_add_shape)>) {
			BIND_VIRTUAL_METHOD(T, _body_add_shape, 2153848567);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_shape), decltype(&T::_body_set_shape)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_shape, 2310537182);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_shape_transform), decltype(&T::_body_set_shape_transform)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_shape_transform, 675327471);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_shape_disabled), decltype(&T::_body_set_shape_disabled)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_shape_disabled, 2658558584);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_shape_count), decltype(&T::_body_get_shape_count)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_shape_count, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_shape), decltype(&T::_body_get_shape)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_shape, 1066463050);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_shape_transform), decltype(&T::_body_get_shape_transform)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_shape_transform, 1050775521);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_remove_shape), decltype(&T::_body_remove_shape)>) {
			BIND_VIRTUAL_METHOD(T, _body_remove_shape, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_clear_shapes), decltype(&T::_body_clear_shapes)>) {
			BIND_VIRTUAL_METHOD(T, _body_clear_shapes, 2722037293);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_attach_object_instance_id), decltype(&T::_body_attach_object_instance_id)>) {
			BIND_VIRTUAL_METHOD(T, _body_attach_object_instance_id, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_object_instance_id), decltype(&T::_body_get_object_instance_id)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_object_instance_id, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_enable_continuous_collision_detection), decltype(&T::_body_set_enable_continuous_collision_detection)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_enable_continuous_collision_detection, 1265174801);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_is_continuous_collision_detection_enabled), decltype(&T::_body_is_continuous_collision_detection_enabled)>) {
			BIND_VIRTUAL_METHOD(T, _body_is_continuous_collision_detection_enabled, 4155700596);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_collision_layer), decltype(&T::_body_set_collision_layer)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_collision_layer, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_collision_layer), decltype(&T::_body_get_collision_layer)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_collision_layer, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_collision_mask), decltype(&T::_body_set_collision_mask)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_collision_mask, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_collision_mask), decltype(&T::_body_get_collision_mask)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_collision_mask, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_collision_priority), decltype(&T::_body_set_collision_priority)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_collision_priority, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_collision_priority), decltype(&T::_body_get_collision_priority)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_collision_priority, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_user_flags), decltype(&T::_body_set_user_flags)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_user_flags, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_user_flags), decltype(&T::_body_get_user_flags)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_user_flags, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_param), decltype(&T::_body_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_param, 910941953);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_param), decltype(&T::_body_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_param, 3385027841);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_reset_mass_properties), decltype(&T::_body_reset_mass_properties)>) {
			BIND_VIRTUAL_METHOD(T, _body_reset_mass_properties, 2722037293);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_state), decltype(&T::_body_set_state)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_state, 599977762);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_state), decltype(&T::_body_get_state)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_state, 1850449534);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_central_impulse), decltype(&T::_body_apply_central_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_central_impulse, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_impulse), decltype(&T::_body_apply_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_impulse, 3214966418);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_torque_impulse), decltype(&T::_body_apply_torque_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_torque_impulse, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_central_force), decltype(&T::_body_apply_central_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_central_force, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_force), decltype(&T::_body_apply_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_force, 3214966418);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_torque), decltype(&T::_body_apply_torque)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_torque, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_add_constant_central_force), decltype(&T::_body_add_constant_central_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_add_constant_central_force, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_add_constant_force), decltype(&T::_body_add_constant_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_add_constant_force, 3214966418);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_add_constant_torque), decltype(&T::_body_add_constant_torque)>) {
			BIND_VIRTUAL_METHOD(T, _body_add_constant_torque, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_constant_force), decltype(&T::_body_set_constant_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_constant_force, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_constant_force), decltype(&T::_body_get_constant_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_constant_force, 531438156);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_constant_torque), decltype(&T::_body_set_constant_torque)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_constant_torque, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_constant_torque), decltype(&T::_body_get_constant_torque)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_constant_torque, 531438156);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_axis_velocity), decltype(&T::_body_set_axis_velocity)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_axis_velocity, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_axis_lock), decltype(&T::_body_set_axis_lock)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_axis_lock, 2020836892);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_is_axis_locked), decltype(&T::_body_is_axis_locked)>) {
			BIND_VIRTUAL_METHOD(T, _body_is_axis_locked, 587853580);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_add_collision_exception), decltype(&T::_body_add_collision_exception)>) {
			BIND_VIRTUAL_METHOD(T, _body_add_collision_exception, 395945892);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_remove_collision_exception), decltype(&T::_body_remove_collision_exception)>) {
			BIND_VIRTUAL_METHOD(T, _body_remove_collision_exception, 395945892);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_collision_exceptions), decltype(&T::_body_get_collision_exceptions)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_collision_exceptions, 2684255073);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_max_contacts_reported), decltype(&T::_body_set_max_contacts_reported)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_max_contacts_reported, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_max_contacts_reported), decltype(&T::_body_get_max_contacts_reported)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_max_contacts_reported, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_contacts_reported_depth_threshold), decltype(&T::_body_set_contacts_reported_depth_threshold)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_contacts_reported_depth_threshold, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_contacts_reported_depth_threshold), decltype(&T::_body_get_contacts_reported_depth_threshold)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_contacts_reported_depth_threshold, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_omit_force_integration), decltype(&T::_body_set_omit_force_integration)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_omit_force_integration, 1265174801);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_is_omitting_force_integration), decltype(&T::_body_is_omitting_force_integration)>) {
			BIND_VIRTUAL_METHOD(T, _body_is_omitting_force_integration, 4155700596);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_state_sync_callback), decltype(&T::_body_set_state_sync_callback)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_state_sync_callback, 3379118538);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_force_integration_callback), decltype(&T::_body_set_force_integration_callback)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_force_integration_callback, 2828036238);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_ray_pickable), decltype(&T::_body_set_ray_pickable)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_ray_pickable, 1265174801);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_test_motion), decltype(&T::_body_test_motion)>) {
			BIND_VIRTUAL_METHOD(T, _body_test_motion, 3627463434);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_direct_state), decltype(&T::_body_get_direct_state)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_direct_state, 3029727957);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_create), decltype(&T::_soft_body_create)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_update_rendering_server), decltype(&T::_soft_body_update_rendering_server)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_update_rendering_server, 2218179753);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_space), decltype(&T::_soft_body_set_space)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_space, 395945892);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_space), decltype(&T::_soft_body_get_space)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_space, 3814569979);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_ray_pickable), decltype(&T::_soft_body_set_ray_pickable)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_ray_pickable, 1265174801);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_collision_layer), decltype(&T::_soft_body_set_collision_layer)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_collision_layer, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_collision_layer), decltype(&T::_soft_body_get_collision_layer)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_collision_layer, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_collision_mask), decltype(&T::_soft_body_set_collision_mask)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_collision_mask, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_collision_mask), decltype(&T::_soft_body_get_collision_mask)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_collision_mask, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_add_collision_exception), decltype(&T::_soft_body_add_collision_exception)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_add_collision_exception, 395945892);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_remove_collision_exception), decltype(&T::_soft_body_remove_collision_exception)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_remove_collision_exception, 395945892);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_collision_exceptions), decltype(&T::_soft_body_get_collision_exceptions)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_collision_exceptions, 2684255073);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_state), decltype(&T::_soft_body_set_state)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_state, 599977762);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_state), decltype(&T::_soft_body_get_state)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_state, 1850449534);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_transform), decltype(&T::_soft_body_set_transform)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_transform, 3935195649);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_simulation_precision), decltype(&T::_soft_body_set_simulation_precision)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_simulation_precision, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_simulation_precision), decltype(&T::_soft_body_get_simulation_precision)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_simulation_precision, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_total_mass), decltype(&T::_soft_body_set_total_mass)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_total_mass, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_total_mass), decltype(&T::_soft_body_get_total_mass)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_total_mass, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_linear_stiffness), decltype(&T::_soft_body_set_linear_stiffness)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_linear_stiffness, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_linear_stiffness), decltype(&T::_soft_body_get_linear_stiffness)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_linear_stiffness, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_shrinking_factor), decltype(&T::_soft_body_set_shrinking_factor)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_shrinking_factor, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_shrinking_factor), decltype(&T::_soft_body_get_shrinking_factor)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_shrinking_factor, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_pressure_coefficient), decltype(&T::_soft_body_set_pressure_coefficient)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_pressure_coefficient, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_pressure_coefficient), decltype(&T::_soft_body_get_pressure_coefficient)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_pressure_coefficient, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_damping_coefficient), decltype(&T::_soft_body_set_damping_coefficient)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_damping_coefficient, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_damping_coefficient), decltype(&T::_soft_body_get_damping_coefficient)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_damping_coefficient, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_drag_coefficient), decltype(&T::_soft_body_set_drag_coefficient)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_drag_coefficient, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_drag_coefficient), decltype(&T::_soft_body_get_drag_coefficient)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_drag_coefficient, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_set_mesh), decltype(&T::_soft_body_set_mesh)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_set_mesh, 395945892);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_bounds), decltype(&T::_soft_body_get_bounds)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_bounds, 974181306);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_move_point), decltype(&T::_soft_body_move_point)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_move_point, 831953689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_get_point_global_position), decltype(&T::_soft_body_get_point_global_position)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_get_point_global_position, 3440143363);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_remove_all_pinned_points), decltype(&T::_soft_body_remove_all_pinned_points)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_remove_all_pinned_points, 2722037293);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_pin_point), decltype(&T::_soft_body_pin_point)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_pin_point, 2658558584);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_is_point_pinned), decltype(&T::_soft_body_is_point_pinned)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_is_point_pinned, 3120086654);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_apply_point_impulse), decltype(&T::_soft_body_apply_point_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_apply_point_impulse, 831953689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_apply_point_force), decltype(&T::_soft_body_apply_point_force)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_apply_point_force, 831953689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_apply_central_impulse), decltype(&T::_soft_body_apply_central_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_apply_central_impulse, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_soft_body_apply_central_force), decltype(&T::_soft_body_apply_central_force)>) {
			BIND_VIRTUAL_METHOD(T, _soft_body_apply_central_force, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_create), decltype(&T::_joint_create)>) {
			BIND_VIRTUAL_METHOD(T, _joint_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_clear), decltype(&T::_joint_clear)>) {
			BIND_VIRTUAL_METHOD(T, _joint_clear, 2722037293);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_make_pin), decltype(&T::_joint_make_pin)>) {
			BIND_VIRTUAL_METHOD(T, _joint_make_pin, 4280171926);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pin_joint_set_param), decltype(&T::_pin_joint_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _pin_joint_set_param, 810685294);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pin_joint_get_param), decltype(&T::_pin_joint_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _pin_joint_get_param, 2817972347);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pin_joint_set_local_a), decltype(&T::_pin_joint_set_local_a)>) {
			BIND_VIRTUAL_METHOD(T, _pin_joint_set_local_a, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pin_joint_get_local_a), decltype(&T::_pin_joint_get_local_a)>) {
			BIND_VIRTUAL_METHOD(T, _pin_joint_get_local_a, 531438156);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pin_joint_set_local_b), decltype(&T::_pin_joint_set_local_b)>) {
			BIND_VIRTUAL_METHOD(T, _pin_joint_set_local_b, 3227306858);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pin_joint_get_local_b), decltype(&T::_pin_joint_get_local_b)>) {
			BIND_VIRTUAL_METHOD(T, _pin_joint_get_local_b, 531438156);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_make_hinge), decltype(&T::_joint_make_hinge)>) {
			BIND_VIRTUAL_METHOD(T, _joint_make_hinge, 1684107643);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_make_hinge_simple), decltype(&T::_joint_make_hinge_simple)>) {
			BIND_VIRTUAL_METHOD(T, _joint_make_hinge_simple, 4069547571);
		}
		if constexpr (!std::is_same_v<decltype(&B::_hinge_joint_set_param), decltype(&T::_hinge_joint_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _hinge_joint_set_param, 3165502333);
		}
		if constexpr (!std::is_same_v<decltype(&B::_hinge_joint_get_param), decltype(&T::_hinge_joint_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _hinge_joint_get_param, 2129207581);
		}
		if constexpr (!std::is_same_v<decltype(&B::_hinge_joint_set_flag), decltype(&T::_hinge_joint_set_flag)>) {
			BIND_VIRTUAL_METHOD(T, _hinge_joint_set_flag, 1601626188);
		}
		if constexpr (!std::is_same_v<decltype(&B::_hinge_joint_get_flag), decltype(&T::_hinge_joint_get_flag)>) {
			BIND_VIRTUAL_METHOD(T, _hinge_joint_get_flag, 4165147865);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_make_slider), decltype(&T::_joint_make_slider)>) {
			BIND_VIRTUAL_METHOD(T, _joint_make_slider, 1684107643);
		}
		if constexpr (!std::is_same_v<decltype(&B::_slider_joint_set_param), decltype(&T::_slider_joint_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _slider_joint_set_param, 2264833593);
		}
		if constexpr (!std::is_same_v<decltype(&B::_slider_joint_get_param), decltype(&T::_slider_joint_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _slider_joint_get_param, 3498644957);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_make_cone_twist), decltype(&T::_joint_make_cone_twist)>) {
			BIND_VIRTUAL_METHOD(T, _joint_make_cone_twist, 1684107643);
		}
		if constexpr (!std::is_same_v<decltype(&B::_cone_twist_joint_set_param), decltype(&T::_cone_twist_joint_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _cone_twist_joint_set_param, 808587618);
		}
		if constexpr (!std::is_same_v<decltype(&B::_cone_twist_joint_get_param), decltype(&T::_cone_twist_joint_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _cone_twist_joint_get_param, 1134789658);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_make_generic_6dof), decltype(&T::_joint_make_generic_6dof)>) {
			BIND_VIRTUAL_METHOD(T, _joint_make_generic_6dof, 1684107643);
		}
		if constexpr (!std::is_same_v<decltype(&B::_generic_6dof_joint_set_param), decltype(&T::_generic_6dof_joint_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _generic_6dof_joint_set_param, 2600081391);
		}
		if constexpr (!std::is_same_v<decltype(&B::_generic_6dof_joint_get_param), decltype(&T::_generic_6dof_joint_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _generic_6dof_joint_get_param, 467122058);
		}
		if constexpr (!std::is_same_v<decltype(&B::_generic_6dof_joint_set_flag), decltype(&T::_generic_6dof_joint_set_flag)>) {
			BIND_VIRTUAL_METHOD(T, _generic_6dof_joint_set_flag, 3570926903);
		}
		if constexpr (!std::is_same_v<decltype(&B::_generic_6dof_joint_get_flag), decltype(&T::_generic_6dof_joint_get_flag)>) {
			BIND_VIRTUAL_METHOD(T, _generic_6dof_joint_get_flag, 4158090196);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_get_type), decltype(&T::_joint_get_type)>) {
			BIND_VIRTUAL_METHOD(T, _joint_get_type, 4290791900);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_set_solver_priority), decltype(&T::_joint_set_solver_priority)>) {
			BIND_VIRTUAL_METHOD(T, _joint_set_solver_priority, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_get_solver_priority), decltype(&T::_joint_get_solver_priority)>) {
			BIND_VIRTUAL_METHOD(T, _joint_get_solver_priority, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_disable_collisions_between_bodies), decltype(&T::_joint_disable_collisions_between_bodies)>) {
			BIND_VIRTUAL_METHOD(T, _joint_disable_collisions_between_bodies, 1265174801);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_is_disabled_collisions_between_bodies), decltype(&T::_joint_is_disabled_collisions_between_bodies)>) {
			BIND_VIRTUAL_METHOD(T, _joint_is_disabled_collisions_between_bodies, 4155700596);
		}
		if constexpr (!std::is_same_v<decltype(&B::_free_rid), decltype(&T::_free_rid)>) {
			BIND_VIRTUAL_METHOD(T, _free_rid, 2722037293);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_active), decltype(&T::_set_active)>) {
			BIND_VIRTUAL_METHOD(T, _set_active, 2586408642);
		}
		if constexpr (!std::is_same_v<decltype(&B::_init), decltype(&T::_init)>) {
			BIND_VIRTUAL_METHOD(T, _init, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_step), decltype(&T::_step)>) {
			BIND_VIRTUAL_METHOD(T, _step, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_sync), decltype(&T::_sync)>) {
			BIND_VIRTUAL_METHOD(T, _sync, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_flush_queries), decltype(&T::_flush_queries)>) {
			BIND_VIRTUAL_METHOD(T, _flush_queries, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_end_sync), decltype(&T::_end_sync)>) {
			BIND_VIRTUAL_METHOD(T, _end_sync, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_finish), decltype(&T::_finish)>) {
			BIND_VIRTUAL_METHOD(T, _finish, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_flushing_queries), decltype(&T::_is_flushing_queries)>) {
			BIND_VIRTUAL_METHOD(T, _is_flushing_queries, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_process_info), decltype(&T::_get_process_info)>) {
			BIND_VIRTUAL_METHOD(T, _get_process_info, 1332958745);
		}
	}

public:
};

} // namespace godot


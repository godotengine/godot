/**************************************************************************/
/*  physics_server2d_extension.hpp                                        */
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

#include <godot_cpp/classes/physics_server2d.hpp>
#include <godot_cpp/classes/physics_server2d_extension_motion_result.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class PhysicsDirectBodyState2D;
class PhysicsDirectSpaceState2D;

class PhysicsServer2DExtension : public PhysicsServer2D {
	GDEXTENSION_CLASS(PhysicsServer2DExtension, PhysicsServer2D)

public:
	bool body_test_motion_is_excluding_body(const RID &p_body) const;
	bool body_test_motion_is_excluding_object(uint64_t p_object) const;
	virtual RID _world_boundary_shape_create();
	virtual RID _separation_ray_shape_create();
	virtual RID _segment_shape_create();
	virtual RID _circle_shape_create();
	virtual RID _rectangle_shape_create();
	virtual RID _capsule_shape_create();
	virtual RID _convex_polygon_shape_create();
	virtual RID _concave_polygon_shape_create();
	virtual void _shape_set_data(const RID &p_shape, const Variant &p_data);
	virtual void _shape_set_custom_solver_bias(const RID &p_shape, float p_bias);
	virtual PhysicsServer2D::ShapeType _shape_get_type(const RID &p_shape) const;
	virtual Variant _shape_get_data(const RID &p_shape) const;
	virtual float _shape_get_custom_solver_bias(const RID &p_shape) const;
	virtual bool _shape_collide(const RID &p_shape_A, const Transform2D &p_xform_A, const Vector2 &p_motion_A, const RID &p_shape_B, const Transform2D &p_xform_B, const Vector2 &p_motion_B, void *p_results, int32_t p_result_max, int32_t *p_result_count);
	virtual RID _space_create();
	virtual void _space_set_active(const RID &p_space, bool p_active);
	virtual bool _space_is_active(const RID &p_space) const;
	virtual void _space_set_param(const RID &p_space, PhysicsServer2D::SpaceParameter p_param, float p_value);
	virtual float _space_get_param(const RID &p_space, PhysicsServer2D::SpaceParameter p_param) const;
	virtual PhysicsDirectSpaceState2D *_space_get_direct_state(const RID &p_space);
	virtual void _space_set_debug_contacts(const RID &p_space, int32_t p_max_contacts);
	virtual PackedVector2Array _space_get_contacts(const RID &p_space) const;
	virtual int32_t _space_get_contact_count(const RID &p_space) const;
	virtual RID _area_create();
	virtual void _area_set_space(const RID &p_area, const RID &p_space);
	virtual RID _area_get_space(const RID &p_area) const;
	virtual void _area_add_shape(const RID &p_area, const RID &p_shape, const Transform2D &p_transform, bool p_disabled);
	virtual void _area_set_shape(const RID &p_area, int32_t p_shape_idx, const RID &p_shape);
	virtual void _area_set_shape_transform(const RID &p_area, int32_t p_shape_idx, const Transform2D &p_transform);
	virtual void _area_set_shape_disabled(const RID &p_area, int32_t p_shape_idx, bool p_disabled);
	virtual int32_t _area_get_shape_count(const RID &p_area) const;
	virtual RID _area_get_shape(const RID &p_area, int32_t p_shape_idx) const;
	virtual Transform2D _area_get_shape_transform(const RID &p_area, int32_t p_shape_idx) const;
	virtual void _area_remove_shape(const RID &p_area, int32_t p_shape_idx);
	virtual void _area_clear_shapes(const RID &p_area);
	virtual void _area_attach_object_instance_id(const RID &p_area, uint64_t p_id);
	virtual uint64_t _area_get_object_instance_id(const RID &p_area) const;
	virtual void _area_attach_canvas_instance_id(const RID &p_area, uint64_t p_id);
	virtual uint64_t _area_get_canvas_instance_id(const RID &p_area) const;
	virtual void _area_set_param(const RID &p_area, PhysicsServer2D::AreaParameter p_param, const Variant &p_value);
	virtual void _area_set_transform(const RID &p_area, const Transform2D &p_transform);
	virtual Variant _area_get_param(const RID &p_area, PhysicsServer2D::AreaParameter p_param) const;
	virtual Transform2D _area_get_transform(const RID &p_area) const;
	virtual void _area_set_collision_layer(const RID &p_area, uint32_t p_layer);
	virtual uint32_t _area_get_collision_layer(const RID &p_area) const;
	virtual void _area_set_collision_mask(const RID &p_area, uint32_t p_mask);
	virtual uint32_t _area_get_collision_mask(const RID &p_area) const;
	virtual void _area_set_monitorable(const RID &p_area, bool p_monitorable);
	virtual void _area_set_pickable(const RID &p_area, bool p_pickable);
	virtual void _area_set_monitor_callback(const RID &p_area, const Callable &p_callback);
	virtual void _area_set_area_monitor_callback(const RID &p_area, const Callable &p_callback);
	virtual RID _body_create();
	virtual void _body_set_space(const RID &p_body, const RID &p_space);
	virtual RID _body_get_space(const RID &p_body) const;
	virtual void _body_set_mode(const RID &p_body, PhysicsServer2D::BodyMode p_mode);
	virtual PhysicsServer2D::BodyMode _body_get_mode(const RID &p_body) const;
	virtual void _body_add_shape(const RID &p_body, const RID &p_shape, const Transform2D &p_transform, bool p_disabled);
	virtual void _body_set_shape(const RID &p_body, int32_t p_shape_idx, const RID &p_shape);
	virtual void _body_set_shape_transform(const RID &p_body, int32_t p_shape_idx, const Transform2D &p_transform);
	virtual int32_t _body_get_shape_count(const RID &p_body) const;
	virtual RID _body_get_shape(const RID &p_body, int32_t p_shape_idx) const;
	virtual Transform2D _body_get_shape_transform(const RID &p_body, int32_t p_shape_idx) const;
	virtual void _body_set_shape_disabled(const RID &p_body, int32_t p_shape_idx, bool p_disabled);
	virtual void _body_set_shape_as_one_way_collision(const RID &p_body, int32_t p_shape_idx, bool p_enable, float p_margin);
	virtual void _body_remove_shape(const RID &p_body, int32_t p_shape_idx);
	virtual void _body_clear_shapes(const RID &p_body);
	virtual void _body_attach_object_instance_id(const RID &p_body, uint64_t p_id);
	virtual uint64_t _body_get_object_instance_id(const RID &p_body) const;
	virtual void _body_attach_canvas_instance_id(const RID &p_body, uint64_t p_id);
	virtual uint64_t _body_get_canvas_instance_id(const RID &p_body) const;
	virtual void _body_set_continuous_collision_detection_mode(const RID &p_body, PhysicsServer2D::CCDMode p_mode);
	virtual PhysicsServer2D::CCDMode _body_get_continuous_collision_detection_mode(const RID &p_body) const;
	virtual void _body_set_collision_layer(const RID &p_body, uint32_t p_layer);
	virtual uint32_t _body_get_collision_layer(const RID &p_body) const;
	virtual void _body_set_collision_mask(const RID &p_body, uint32_t p_mask);
	virtual uint32_t _body_get_collision_mask(const RID &p_body) const;
	virtual void _body_set_collision_priority(const RID &p_body, float p_priority);
	virtual float _body_get_collision_priority(const RID &p_body) const;
	virtual void _body_set_param(const RID &p_body, PhysicsServer2D::BodyParameter p_param, const Variant &p_value);
	virtual Variant _body_get_param(const RID &p_body, PhysicsServer2D::BodyParameter p_param) const;
	virtual void _body_reset_mass_properties(const RID &p_body);
	virtual void _body_set_state(const RID &p_body, PhysicsServer2D::BodyState p_state, const Variant &p_value);
	virtual Variant _body_get_state(const RID &p_body, PhysicsServer2D::BodyState p_state) const;
	virtual void _body_apply_central_impulse(const RID &p_body, const Vector2 &p_impulse);
	virtual void _body_apply_torque_impulse(const RID &p_body, float p_impulse);
	virtual void _body_apply_impulse(const RID &p_body, const Vector2 &p_impulse, const Vector2 &p_position);
	virtual void _body_apply_central_force(const RID &p_body, const Vector2 &p_force);
	virtual void _body_apply_force(const RID &p_body, const Vector2 &p_force, const Vector2 &p_position);
	virtual void _body_apply_torque(const RID &p_body, float p_torque);
	virtual void _body_add_constant_central_force(const RID &p_body, const Vector2 &p_force);
	virtual void _body_add_constant_force(const RID &p_body, const Vector2 &p_force, const Vector2 &p_position);
	virtual void _body_add_constant_torque(const RID &p_body, float p_torque);
	virtual void _body_set_constant_force(const RID &p_body, const Vector2 &p_force);
	virtual Vector2 _body_get_constant_force(const RID &p_body) const;
	virtual void _body_set_constant_torque(const RID &p_body, float p_torque);
	virtual float _body_get_constant_torque(const RID &p_body) const;
	virtual void _body_set_axis_velocity(const RID &p_body, const Vector2 &p_axis_velocity);
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
	virtual bool _body_collide_shape(const RID &p_body, int32_t p_body_shape, const RID &p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, void *p_results, int32_t p_result_max, int32_t *p_result_count);
	virtual void _body_set_pickable(const RID &p_body, bool p_pickable);
	virtual PhysicsDirectBodyState2D *_body_get_direct_state(const RID &p_body);
	virtual bool _body_test_motion(const RID &p_body, const Transform2D &p_from, const Vector2 &p_motion, float p_margin, bool p_collide_separation_ray, bool p_recovery_as_collision, PhysicsServer2DExtensionMotionResult *p_result) const;
	virtual RID _joint_create();
	virtual void _joint_clear(const RID &p_joint);
	virtual void _joint_set_param(const RID &p_joint, PhysicsServer2D::JointParam p_param, float p_value);
	virtual float _joint_get_param(const RID &p_joint, PhysicsServer2D::JointParam p_param) const;
	virtual void _joint_disable_collisions_between_bodies(const RID &p_joint, bool p_disable);
	virtual bool _joint_is_disabled_collisions_between_bodies(const RID &p_joint) const;
	virtual void _joint_make_pin(const RID &p_joint, const Vector2 &p_anchor, const RID &p_body_a, const RID &p_body_b);
	virtual void _joint_make_groove(const RID &p_joint, const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, const RID &p_body_a, const RID &p_body_b);
	virtual void _joint_make_damped_spring(const RID &p_joint, const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, const RID &p_body_a, const RID &p_body_b);
	virtual void _pin_joint_set_flag(const RID &p_joint, PhysicsServer2D::PinJointFlag p_flag, bool p_enabled);
	virtual bool _pin_joint_get_flag(const RID &p_joint, PhysicsServer2D::PinJointFlag p_flag) const;
	virtual void _pin_joint_set_param(const RID &p_joint, PhysicsServer2D::PinJointParam p_param, float p_value);
	virtual float _pin_joint_get_param(const RID &p_joint, PhysicsServer2D::PinJointParam p_param) const;
	virtual void _damped_spring_joint_set_param(const RID &p_joint, PhysicsServer2D::DampedSpringParam p_param, float p_value);
	virtual float _damped_spring_joint_get_param(const RID &p_joint, PhysicsServer2D::DampedSpringParam p_param) const;
	virtual PhysicsServer2D::JointType _joint_get_type(const RID &p_joint) const;
	virtual void _free_rid(const RID &p_rid);
	virtual void _set_active(bool p_active);
	virtual void _init();
	virtual void _step(float p_step);
	virtual void _sync();
	virtual void _flush_queries();
	virtual void _end_sync();
	virtual void _finish();
	virtual bool _is_flushing_queries() const;
	virtual int32_t _get_process_info(PhysicsServer2D::ProcessInfo p_process_info);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PhysicsServer2D::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_world_boundary_shape_create), decltype(&T::_world_boundary_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _world_boundary_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_separation_ray_shape_create), decltype(&T::_separation_ray_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _separation_ray_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_segment_shape_create), decltype(&T::_segment_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _segment_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_circle_shape_create), decltype(&T::_circle_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _circle_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_rectangle_shape_create), decltype(&T::_rectangle_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _rectangle_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_capsule_shape_create), decltype(&T::_capsule_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _capsule_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_convex_polygon_shape_create), decltype(&T::_convex_polygon_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _convex_polygon_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_concave_polygon_shape_create), decltype(&T::_concave_polygon_shape_create)>) {
			BIND_VIRTUAL_METHOD(T, _concave_polygon_shape_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_set_data), decltype(&T::_shape_set_data)>) {
			BIND_VIRTUAL_METHOD(T, _shape_set_data, 3175752987);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_set_custom_solver_bias), decltype(&T::_shape_set_custom_solver_bias)>) {
			BIND_VIRTUAL_METHOD(T, _shape_set_custom_solver_bias, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_get_type), decltype(&T::_shape_get_type)>) {
			BIND_VIRTUAL_METHOD(T, _shape_get_type, 1240598777);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_get_data), decltype(&T::_shape_get_data)>) {
			BIND_VIRTUAL_METHOD(T, _shape_get_data, 4171304767);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_get_custom_solver_bias), decltype(&T::_shape_get_custom_solver_bias)>) {
			BIND_VIRTUAL_METHOD(T, _shape_get_custom_solver_bias, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shape_collide), decltype(&T::_shape_collide)>) {
			BIND_VIRTUAL_METHOD(T, _shape_collide, 738864683);
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
			BIND_VIRTUAL_METHOD(T, _space_set_param, 949194586);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_get_param), decltype(&T::_space_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _space_get_param, 874111783);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_get_direct_state), decltype(&T::_space_get_direct_state)>) {
			BIND_VIRTUAL_METHOD(T, _space_get_direct_state, 3160173886);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_set_debug_contacts), decltype(&T::_space_set_debug_contacts)>) {
			BIND_VIRTUAL_METHOD(T, _space_set_debug_contacts, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_space_get_contacts), decltype(&T::_space_get_contacts)>) {
			BIND_VIRTUAL_METHOD(T, _space_get_contacts, 2222557395);
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
			BIND_VIRTUAL_METHOD(T, _area_add_shape, 888317420);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_shape), decltype(&T::_area_set_shape)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_shape, 2310537182);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_shape_transform), decltype(&T::_area_set_shape_transform)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_shape_transform, 736082694);
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
			BIND_VIRTUAL_METHOD(T, _area_get_shape_transform, 1324854622);
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
		if constexpr (!std::is_same_v<decltype(&B::_area_attach_canvas_instance_id), decltype(&T::_area_attach_canvas_instance_id)>) {
			BIND_VIRTUAL_METHOD(T, _area_attach_canvas_instance_id, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_canvas_instance_id), decltype(&T::_area_get_canvas_instance_id)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_canvas_instance_id, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_param), decltype(&T::_area_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_param, 1257146028);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_set_transform), decltype(&T::_area_set_transform)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_transform, 1246044741);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_param), decltype(&T::_area_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_param, 3047435120);
		}
		if constexpr (!std::is_same_v<decltype(&B::_area_get_transform), decltype(&T::_area_get_transform)>) {
			BIND_VIRTUAL_METHOD(T, _area_get_transform, 213527486);
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
		if constexpr (!std::is_same_v<decltype(&B::_area_set_pickable), decltype(&T::_area_set_pickable)>) {
			BIND_VIRTUAL_METHOD(T, _area_set_pickable, 1265174801);
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
			BIND_VIRTUAL_METHOD(T, _body_set_mode, 1658067650);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_mode), decltype(&T::_body_get_mode)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_mode, 3261702585);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_add_shape), decltype(&T::_body_add_shape)>) {
			BIND_VIRTUAL_METHOD(T, _body_add_shape, 888317420);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_shape), decltype(&T::_body_set_shape)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_shape, 2310537182);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_shape_transform), decltype(&T::_body_set_shape_transform)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_shape_transform, 736082694);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_shape_count), decltype(&T::_body_get_shape_count)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_shape_count, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_shape), decltype(&T::_body_get_shape)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_shape, 1066463050);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_shape_transform), decltype(&T::_body_get_shape_transform)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_shape_transform, 1324854622);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_shape_disabled), decltype(&T::_body_set_shape_disabled)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_shape_disabled, 2658558584);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_shape_as_one_way_collision), decltype(&T::_body_set_shape_as_one_way_collision)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_shape_as_one_way_collision, 2556489974);
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
		if constexpr (!std::is_same_v<decltype(&B::_body_attach_canvas_instance_id), decltype(&T::_body_attach_canvas_instance_id)>) {
			BIND_VIRTUAL_METHOD(T, _body_attach_canvas_instance_id, 3411492887);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_canvas_instance_id), decltype(&T::_body_get_canvas_instance_id)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_canvas_instance_id, 2198884583);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_continuous_collision_detection_mode), decltype(&T::_body_set_continuous_collision_detection_mode)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_continuous_collision_detection_mode, 1882257015);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_continuous_collision_detection_mode), decltype(&T::_body_get_continuous_collision_detection_mode)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_continuous_collision_detection_mode, 2661282217);
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
		if constexpr (!std::is_same_v<decltype(&B::_body_set_param), decltype(&T::_body_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_param, 2715630609);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_param), decltype(&T::_body_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_param, 3208033526);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_reset_mass_properties), decltype(&T::_body_reset_mass_properties)>) {
			BIND_VIRTUAL_METHOD(T, _body_reset_mass_properties, 2722037293);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_state), decltype(&T::_body_set_state)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_state, 1706355209);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_state), decltype(&T::_body_get_state)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_state, 4036367961);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_central_impulse), decltype(&T::_body_apply_central_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_central_impulse, 3201125042);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_torque_impulse), decltype(&T::_body_apply_torque_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_torque_impulse, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_impulse), decltype(&T::_body_apply_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_impulse, 2762675110);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_central_force), decltype(&T::_body_apply_central_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_central_force, 3201125042);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_force), decltype(&T::_body_apply_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_force, 2762675110);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_apply_torque), decltype(&T::_body_apply_torque)>) {
			BIND_VIRTUAL_METHOD(T, _body_apply_torque, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_add_constant_central_force), decltype(&T::_body_add_constant_central_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_add_constant_central_force, 3201125042);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_add_constant_force), decltype(&T::_body_add_constant_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_add_constant_force, 2762675110);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_add_constant_torque), decltype(&T::_body_add_constant_torque)>) {
			BIND_VIRTUAL_METHOD(T, _body_add_constant_torque, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_constant_force), decltype(&T::_body_set_constant_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_constant_force, 3201125042);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_constant_force), decltype(&T::_body_get_constant_force)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_constant_force, 2440833711);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_constant_torque), decltype(&T::_body_set_constant_torque)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_constant_torque, 1794382983);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_constant_torque), decltype(&T::_body_get_constant_torque)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_constant_torque, 866169185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_axis_velocity), decltype(&T::_body_set_axis_velocity)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_axis_velocity, 3201125042);
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
		if constexpr (!std::is_same_v<decltype(&B::_body_collide_shape), decltype(&T::_body_collide_shape)>) {
			BIND_VIRTUAL_METHOD(T, _body_collide_shape, 2131476465);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_set_pickable), decltype(&T::_body_set_pickable)>) {
			BIND_VIRTUAL_METHOD(T, _body_set_pickable, 1265174801);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_get_direct_state), decltype(&T::_body_get_direct_state)>) {
			BIND_VIRTUAL_METHOD(T, _body_get_direct_state, 1191931871);
		}
		if constexpr (!std::is_same_v<decltype(&B::_body_test_motion), decltype(&T::_body_test_motion)>) {
			BIND_VIRTUAL_METHOD(T, _body_test_motion, 104979818);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_create), decltype(&T::_joint_create)>) {
			BIND_VIRTUAL_METHOD(T, _joint_create, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_clear), decltype(&T::_joint_clear)>) {
			BIND_VIRTUAL_METHOD(T, _joint_clear, 2722037293);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_set_param), decltype(&T::_joint_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _joint_set_param, 3972556514);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_get_param), decltype(&T::_joint_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _joint_get_param, 4016448949);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_disable_collisions_between_bodies), decltype(&T::_joint_disable_collisions_between_bodies)>) {
			BIND_VIRTUAL_METHOD(T, _joint_disable_collisions_between_bodies, 1265174801);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_is_disabled_collisions_between_bodies), decltype(&T::_joint_is_disabled_collisions_between_bodies)>) {
			BIND_VIRTUAL_METHOD(T, _joint_is_disabled_collisions_between_bodies, 4155700596);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_make_pin), decltype(&T::_joint_make_pin)>) {
			BIND_VIRTUAL_METHOD(T, _joint_make_pin, 2607799521);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_make_groove), decltype(&T::_joint_make_groove)>) {
			BIND_VIRTUAL_METHOD(T, _joint_make_groove, 438649616);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_make_damped_spring), decltype(&T::_joint_make_damped_spring)>) {
			BIND_VIRTUAL_METHOD(T, _joint_make_damped_spring, 1276049561);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pin_joint_set_flag), decltype(&T::_pin_joint_set_flag)>) {
			BIND_VIRTUAL_METHOD(T, _pin_joint_set_flag, 3520002352);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pin_joint_get_flag), decltype(&T::_pin_joint_get_flag)>) {
			BIND_VIRTUAL_METHOD(T, _pin_joint_get_flag, 2647867364);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pin_joint_set_param), decltype(&T::_pin_joint_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _pin_joint_set_param, 550574241);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pin_joint_get_param), decltype(&T::_pin_joint_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _pin_joint_get_param, 348281383);
		}
		if constexpr (!std::is_same_v<decltype(&B::_damped_spring_joint_set_param), decltype(&T::_damped_spring_joint_set_param)>) {
			BIND_VIRTUAL_METHOD(T, _damped_spring_joint_set_param, 220564071);
		}
		if constexpr (!std::is_same_v<decltype(&B::_damped_spring_joint_get_param), decltype(&T::_damped_spring_joint_get_param)>) {
			BIND_VIRTUAL_METHOD(T, _damped_spring_joint_get_param, 2075871277);
		}
		if constexpr (!std::is_same_v<decltype(&B::_joint_get_type), decltype(&T::_joint_get_type)>) {
			BIND_VIRTUAL_METHOD(T, _joint_get_type, 4262502231);
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
			BIND_VIRTUAL_METHOD(T, _get_process_info, 576496006);
		}
	}

public:
};

} // namespace godot


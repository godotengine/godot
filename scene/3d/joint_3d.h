/*************************************************************************/
/*  joint_3d.h                                                           */
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

#ifndef JOINT_3D_H
#define JOINT_3D_H

#include "scene/3d/node_3d.h"
#include "scene/3d/physics_body_3d.h"

class Joint3D : public Node3D {
	GDCLASS(Joint3D, Node3D);

	RID ba, bb;

	RID joint;

	NodePath a;
	NodePath b;

	int solver_priority = 1;
	bool exclude_from_collision = true;
	String warning;
	bool configured = false;

protected:
	void _disconnect_signals();
	void _body_exit_tree();
	void _update_joint(bool p_only_free = false);

	void _notification(int p_what);

	virtual void _configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) = 0;

	static void _bind_methods();

	_FORCE_INLINE_ bool is_configured() const { return configured; }

public:
	virtual TypedArray<String> get_configuration_warnings() const override;

	void set_node_a(const NodePath &p_node_a);
	NodePath get_node_a() const;

	void set_node_b(const NodePath &p_node_b);
	NodePath get_node_b() const;

	void set_solver_priority(int p_priority);
	int get_solver_priority() const;

	void set_exclude_nodes_from_collision(bool p_enable);
	bool get_exclude_nodes_from_collision() const;

	RID get_joint() const { return joint; }
	Joint3D();
	~Joint3D();
};

///////////////////////////////////////////

class PinJoint3D : public Joint3D {
	GDCLASS(PinJoint3D, Joint3D);

public:
	enum Param {
		PARAM_BIAS = PhysicsServer3D::PIN_JOINT_BIAS,
		PARAM_DAMPING = PhysicsServer3D::PIN_JOINT_DAMPING,
		PARAM_IMPULSE_CLAMP = PhysicsServer3D::PIN_JOINT_IMPULSE_CLAMP
	};

protected:
	real_t params[3];
	virtual void _configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) override;
	static void _bind_methods();

public:
	void set_param(Param p_param, real_t p_value);
	real_t get_param(Param p_param) const;

	PinJoint3D();
};

VARIANT_ENUM_CAST(PinJoint3D::Param);

class HingeJoint3D : public Joint3D {
	GDCLASS(HingeJoint3D, Joint3D);

public:
	enum Param {
		PARAM_BIAS = PhysicsServer3D::HINGE_JOINT_BIAS,
		PARAM_LIMIT_UPPER = PhysicsServer3D::HINGE_JOINT_LIMIT_UPPER,
		PARAM_LIMIT_LOWER = PhysicsServer3D::HINGE_JOINT_LIMIT_LOWER,
		PARAM_LIMIT_BIAS = PhysicsServer3D::HINGE_JOINT_LIMIT_BIAS,
		PARAM_LIMIT_SOFTNESS = PhysicsServer3D::HINGE_JOINT_LIMIT_SOFTNESS,
		PARAM_LIMIT_RELAXATION = PhysicsServer3D::HINGE_JOINT_LIMIT_RELAXATION,
		PARAM_MOTOR_TARGET_VELOCITY = PhysicsServer3D::HINGE_JOINT_MOTOR_TARGET_VELOCITY,
		PARAM_MOTOR_MAX_IMPULSE = PhysicsServer3D::HINGE_JOINT_MOTOR_MAX_IMPULSE,
		PARAM_MAX = PhysicsServer3D::HINGE_JOINT_MAX
	};

	enum Flag {
		FLAG_USE_LIMIT = PhysicsServer3D::HINGE_JOINT_FLAG_USE_LIMIT,
		FLAG_ENABLE_MOTOR = PhysicsServer3D::HINGE_JOINT_FLAG_ENABLE_MOTOR,
		FLAG_MAX = PhysicsServer3D::HINGE_JOINT_FLAG_MAX
	};

protected:
	real_t params[PARAM_MAX];
	bool flags[FLAG_MAX];
	virtual void _configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) override;
	static void _bind_methods();

	void _set_upper_limit(real_t p_limit);
	real_t _get_upper_limit() const;

	void _set_lower_limit(real_t p_limit);
	real_t _get_lower_limit() const;

public:
	void set_param(Param p_param, real_t p_value);
	real_t get_param(Param p_param) const;

	void set_flag(Flag p_flag, bool p_value);
	bool get_flag(Flag p_flag) const;

	HingeJoint3D();
};

VARIANT_ENUM_CAST(HingeJoint3D::Param);
VARIANT_ENUM_CAST(HingeJoint3D::Flag);

class SliderJoint3D : public Joint3D {
	GDCLASS(SliderJoint3D, Joint3D);

public:
	enum Param {
		PARAM_LINEAR_LIMIT_UPPER = PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_UPPER,
		PARAM_LINEAR_LIMIT_LOWER = PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_LOWER,
		PARAM_LINEAR_LIMIT_SOFTNESS = PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS,
		PARAM_LINEAR_LIMIT_RESTITUTION = PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION,
		PARAM_LINEAR_LIMIT_DAMPING = PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_DAMPING,
		PARAM_LINEAR_MOTION_SOFTNESS = PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_SOFTNESS,
		PARAM_LINEAR_MOTION_RESTITUTION = PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_RESTITUTION,
		PARAM_LINEAR_MOTION_DAMPING = PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_DAMPING,
		PARAM_LINEAR_ORTHOGONAL_SOFTNESS = PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_SOFTNESS,
		PARAM_LINEAR_ORTHOGONAL_RESTITUTION = PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_RESTITUTION,
		PARAM_LINEAR_ORTHOGONAL_DAMPING = PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_DAMPING,

		PARAM_ANGULAR_LIMIT_UPPER = PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_UPPER,
		PARAM_ANGULAR_LIMIT_LOWER = PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_LOWER,
		PARAM_ANGULAR_LIMIT_SOFTNESS = PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS,
		PARAM_ANGULAR_LIMIT_RESTITUTION = PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_RESTITUTION,
		PARAM_ANGULAR_LIMIT_DAMPING = PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING,
		PARAM_ANGULAR_MOTION_SOFTNESS = PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_SOFTNESS,
		PARAM_ANGULAR_MOTION_RESTITUTION = PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_RESTITUTION,
		PARAM_ANGULAR_MOTION_DAMPING = PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_DAMPING,
		PARAM_ANGULAR_ORTHOGONAL_SOFTNESS = PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_SOFTNESS,
		PARAM_ANGULAR_ORTHOGONAL_RESTITUTION = PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_RESTITUTION,
		PARAM_ANGULAR_ORTHOGONAL_DAMPING = PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_DAMPING,
		PARAM_MAX = PhysicsServer3D::SLIDER_JOINT_MAX

	};

protected:
	void _set_upper_limit_angular(real_t p_limit_angular);
	real_t _get_upper_limit_angular() const;

	void _set_lower_limit_angular(real_t p_limit_angular);
	real_t _get_lower_limit_angular() const;

	real_t params[PARAM_MAX];
	virtual void _configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) override;
	static void _bind_methods();

public:
	void set_param(Param p_param, real_t p_value);
	real_t get_param(Param p_param) const;

	SliderJoint3D();
};

VARIANT_ENUM_CAST(SliderJoint3D::Param);

class ConeTwistJoint3D : public Joint3D {
	GDCLASS(ConeTwistJoint3D, Joint3D);

public:
	enum Param {
		PARAM_SWING_SPAN,
		PARAM_TWIST_SPAN,
		PARAM_BIAS,
		PARAM_SOFTNESS,
		PARAM_RELAXATION,
		PARAM_MAX
	};

protected:
	void _set_swing_span(real_t p_limit_angular);
	real_t _get_swing_span() const;

	void _set_twist_span(real_t p_limit_angular);
	real_t _get_twist_span() const;

	real_t params[PARAM_MAX];
	virtual void _configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) override;
	static void _bind_methods();

public:
	void set_param(Param p_param, real_t p_value);
	real_t get_param(Param p_param) const;

	ConeTwistJoint3D();
};

VARIANT_ENUM_CAST(ConeTwistJoint3D::Param);

class Generic6DOFJoint3D : public Joint3D {
	GDCLASS(Generic6DOFJoint3D, Joint3D);

public:
	enum Param {
		PARAM_LINEAR_LOWER_LIMIT = PhysicsServer3D::G6DOF_JOINT_LINEAR_LOWER_LIMIT,
		PARAM_LINEAR_UPPER_LIMIT = PhysicsServer3D::G6DOF_JOINT_LINEAR_UPPER_LIMIT,
		PARAM_LINEAR_LIMIT_SOFTNESS = PhysicsServer3D::G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS,
		PARAM_LINEAR_RESTITUTION = PhysicsServer3D::G6DOF_JOINT_LINEAR_RESTITUTION,
		PARAM_LINEAR_DAMPING = PhysicsServer3D::G6DOF_JOINT_LINEAR_DAMPING,
		PARAM_LINEAR_MOTOR_TARGET_VELOCITY = PhysicsServer3D::G6DOF_JOINT_LINEAR_MOTOR_TARGET_VELOCITY,
		PARAM_LINEAR_MOTOR_FORCE_LIMIT = PhysicsServer3D::G6DOF_JOINT_LINEAR_MOTOR_FORCE_LIMIT,
		PARAM_LINEAR_SPRING_STIFFNESS = PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_STIFFNESS,
		PARAM_LINEAR_SPRING_DAMPING = PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_DAMPING,
		PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT = PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_EQUILIBRIUM_POINT,
		PARAM_ANGULAR_LOWER_LIMIT = PhysicsServer3D::G6DOF_JOINT_ANGULAR_LOWER_LIMIT,
		PARAM_ANGULAR_UPPER_LIMIT = PhysicsServer3D::G6DOF_JOINT_ANGULAR_UPPER_LIMIT,
		PARAM_ANGULAR_LIMIT_SOFTNESS = PhysicsServer3D::G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS,
		PARAM_ANGULAR_DAMPING = PhysicsServer3D::G6DOF_JOINT_ANGULAR_DAMPING,
		PARAM_ANGULAR_RESTITUTION = PhysicsServer3D::G6DOF_JOINT_ANGULAR_RESTITUTION,
		PARAM_ANGULAR_FORCE_LIMIT = PhysicsServer3D::G6DOF_JOINT_ANGULAR_FORCE_LIMIT,
		PARAM_ANGULAR_ERP = PhysicsServer3D::G6DOF_JOINT_ANGULAR_ERP,
		PARAM_ANGULAR_MOTOR_TARGET_VELOCITY = PhysicsServer3D::G6DOF_JOINT_ANGULAR_MOTOR_TARGET_VELOCITY,
		PARAM_ANGULAR_MOTOR_FORCE_LIMIT = PhysicsServer3D::G6DOF_JOINT_ANGULAR_MOTOR_FORCE_LIMIT,
		PARAM_ANGULAR_SPRING_STIFFNESS = PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_STIFFNESS,
		PARAM_ANGULAR_SPRING_DAMPING = PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_DAMPING,
		PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT = PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_EQUILIBRIUM_POINT,
		PARAM_MAX = PhysicsServer3D::G6DOF_JOINT_MAX,
	};

	enum Flag {
		FLAG_ENABLE_LINEAR_LIMIT = PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT,
		FLAG_ENABLE_ANGULAR_LIMIT = PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT,
		FLAG_ENABLE_LINEAR_SPRING = PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING,
		FLAG_ENABLE_ANGULAR_SPRING = PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING,
		FLAG_ENABLE_MOTOR = PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_MOTOR,
		FLAG_ENABLE_LINEAR_MOTOR = PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_MOTOR,
		FLAG_MAX = PhysicsServer3D::G6DOF_JOINT_FLAG_MAX
	};

protected:
	void _set_angular_hi_limit_x(real_t p_limit_angular);
	real_t _get_angular_hi_limit_x() const;

	void _set_angular_hi_limit_y(real_t p_limit_angular);
	real_t _get_angular_hi_limit_y() const;

	void _set_angular_hi_limit_z(real_t p_limit_angular);
	real_t _get_angular_hi_limit_z() const;

	void _set_angular_lo_limit_x(real_t p_limit_angular);
	real_t _get_angular_lo_limit_x() const;

	void _set_angular_lo_limit_y(real_t p_limit_angular);
	real_t _get_angular_lo_limit_y() const;

	void _set_angular_lo_limit_z(real_t p_limit_angular);
	real_t _get_angular_lo_limit_z() const;

	real_t params_x[PARAM_MAX];
	bool flags_x[FLAG_MAX];
	real_t params_y[PARAM_MAX];
	bool flags_y[FLAG_MAX];
	real_t params_z[PARAM_MAX];
	bool flags_z[FLAG_MAX];

	virtual void _configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) override;
	static void _bind_methods();

public:
	void set_param_x(Param p_param, real_t p_value);
	real_t get_param_x(Param p_param) const;

	void set_param_y(Param p_param, real_t p_value);
	real_t get_param_y(Param p_param) const;

	void set_param_z(Param p_param, real_t p_value);
	real_t get_param_z(Param p_param) const;

	void set_flag_x(Flag p_flag, bool p_enabled);
	bool get_flag_x(Flag p_flag) const;

	void set_flag_y(Flag p_flag, bool p_enabled);
	bool get_flag_y(Flag p_flag) const;

	void set_flag_z(Flag p_flag, bool p_enabled);
	bool get_flag_z(Flag p_flag) const;

	Generic6DOFJoint3D();
};

VARIANT_ENUM_CAST(Generic6DOFJoint3D::Param);
VARIANT_ENUM_CAST(Generic6DOFJoint3D::Flag);

#endif // JOINT_3D_H

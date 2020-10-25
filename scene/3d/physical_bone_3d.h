/*************************************************************************/
/*  physical_bone_3d.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef PHYSICAL_BONE_3D_H
#define PHYSICAL_BONE_3D_H

#include "rigid_body_3d.h"

class Skeleton3D;

class PhysicalBone3D : public RigidBody3D {
	GDCLASS(PhysicalBone3D, RigidBody3D);

public:
	enum JointType {
		JOINT_TYPE_NONE,
		JOINT_TYPE_PIN,
		JOINT_TYPE_CONE,
		JOINT_TYPE_HINGE,
		JOINT_TYPE_SLIDER,
		JOINT_TYPE_6DOF
	};

	struct JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_NONE; }

		/// "j" is used to set the parameter inside the PhysicsServer3D
		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j = RID());
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		virtual ~JointData() {}
	};

	struct PinJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_PIN; }

		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j = RID());
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		real_t bias = 0.3;
		real_t damping = 1.;
		real_t impulse_clamp = 0;

		PinJointData() {}
	};

	struct ConeJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_CONE; }

		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j = RID());
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		real_t swing_span;
		real_t twist_span = Math_PI;
		real_t bias = 0.3;
		real_t softness = 0.8;
		real_t relaxation = 1.;

		ConeJointData() :
				swing_span(Math_PI * 0.25) {}
	};

	struct HingeJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_HINGE; }

		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j = RID());
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		bool angular_limit_enabled = false;
		real_t angular_limit_upper;
		real_t angular_limit_lower;
		real_t angular_limit_bias = 0.3;
		real_t angular_limit_softness = 0.9;
		real_t angular_limit_relaxation = 1.;

		HingeJointData() :

				angular_limit_upper(Math_PI * 0.5),
				angular_limit_lower(-Math_PI * 0.5) {}
	};

	struct SliderJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_SLIDER; }

		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j = RID());
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		real_t linear_limit_upper = 1.;
		real_t linear_limit_lower = -1.;
		real_t linear_limit_softness = 1.;
		real_t linear_limit_restitution = 0.7;
		real_t linear_limit_damping = 1.;
		real_t angular_limit_upper = 0;
		real_t angular_limit_lower = 0;
		real_t angular_limit_softness = 1.;
		real_t angular_limit_restitution = 0.7;
		real_t angular_limit_damping = 1.;

		SliderJointData() {}
	};

	struct SixDOFJointData : public JointData {
		struct SixDOFAxisData {
			bool linear_limit_enabled = true;
			real_t linear_limit_upper = 0;
			real_t linear_limit_lower = 0;
			real_t linear_limit_softness = 0.7;
			real_t linear_restitution = 0.5;
			real_t linear_damping = 1.;
			bool linear_spring_enabled = false;
			real_t linear_spring_stiffness = 0;
			real_t linear_spring_damping = 0;
			real_t linear_equilibrium_point = 0;
			bool angular_limit_enabled = true;
			real_t angular_limit_upper = 0;
			real_t angular_limit_lower = 0;
			real_t angular_limit_softness = 0.5;
			real_t angular_restitution = 0;
			real_t angular_damping = 1.;
			real_t erp = 0.5;
			bool angular_spring_enabled = false;
			real_t angular_spring_stiffness = 0;
			real_t angular_spring_damping = 0.;
			real_t angular_equilibrium_point = 0;

			SixDOFAxisData() {}
		};

		virtual JointType get_joint_type() { return JOINT_TYPE_6DOF; }

		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j = RID());
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		SixDOFAxisData axis_data[3];

		SixDOFJointData() {}
	};

private:
#ifdef TOOLS_ENABLED
	// if false gizmo move body
	bool gizmo_move_joint = false;
#endif

	JointData *joint_data = nullptr;
	Transform joint_offset;
	RID joint;

	Skeleton3D *parent_skeleton = nullptr;
	Transform body_offset;
	Transform body_offset_inverse;
	bool simulate_physics = false;
	bool _internal_simulate_physics = false;
	int bone_id = -1;

	String bone_name;
	real_t bounce = 0;
	real_t mass = 1;
	real_t friction = 1;
	real_t gravity_scale = 1;
	real_t linear_damp = -1;
	real_t angular_damp = -1;
	bool can_sleep = true;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);
	virtual void _direct_state_changed(Object *p_state) override;

	static void _bind_methods();

private:
	static Skeleton3D *find_skeleton_parent(Node *p_parent);

	void _update_joint_offset();
	void _fix_joint_offset();
	void _reload_joint();

public:
	void _on_bone_parent_changed();
	void _set_gizmo_move_joint(bool p_move_joint);

public:
#ifdef TOOLS_ENABLED
	virtual Transform get_global_gizmo_transform() const override;
	virtual Transform get_local_gizmo_transform() const override;
#endif

	const JointData *get_joint_data() const;
	Skeleton3D *find_skeleton_parent();

	int get_bone_id() const { return bone_id; }

	void set_joint_type(JointType p_joint_type);
	JointType get_joint_type() const;

	void set_joint_offset(const Transform &p_offset);
	const Transform &get_joint_offset() const;

	void set_joint_rotation(const Vector3 &p_euler_rad);
	Vector3 get_joint_rotation() const;

	void set_joint_rotation_degrees(const Vector3 &p_euler_deg);
	Vector3 get_joint_rotation_degrees() const;

	void set_body_offset(const Transform &p_offset);
	const Transform &get_body_offset() const;

	void set_simulate_physics(bool p_simulate);
	bool get_simulate_physics();
	bool is_simulating_physics();

	void set_bone_name(const String &p_name);
	const String &get_bone_name() const;

	void reset_physics_simulation_state();
	void reset_to_rest_position();

	PhysicalBone3D();
	~PhysicalBone3D();

private:
	void update_bone_id();
	void update_offset();

	void _start_physics_simulation();
	void _stop_physics_simulation();
};

VARIANT_ENUM_CAST(PhysicalBone3D::JointType);

#endif // PHYSICAL_BONE_3D_H

/**************************************************************************/
/*  vehicle_wing_3d.h                                                     */
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

#include "core/math/math_funcs.h"
#include "scene/3d/node_3d.h"

class RigidBody3D;

class VehicleWing3D : public Node3D {
	GDCLASS(VehicleWing3D, Node3D);

public:
	enum ControlSurfaceType {
		None,
		Flap,
		Aileron
	};

	struct ControlSurface {
		ControlSurfaceType type;
		real_t start;
		real_t end;
		real_t fraction;
	};

	struct Section {
		ControlSurfaceType type;
		Transform3D transform;
		real_t length;
		real_t chord;
		real_t control_surface_fraction;
		bool mirror;

		// dynamic data
		Transform3D global_transform;
		Vector3 force;
		Vector3 torque;
		real_t lift_factor;
		real_t drag_factor;
		real_t torque_factor;
		real_t angle_of_attack;
		real_t control_surface_lift;
		real_t control_surface_angle;
		real_t corrected_lift_slope;
		real_t corrected_zero_lift_angle;
		real_t corrected_stall_angle_max;
		real_t corrected_stall_angle_min;
		bool stall_warning;
		bool stall;
		real_t restore_stall_angle_max;
		real_t restore_stall_angle_min;
	};

private:
	RigidBody3D *m_body = nullptr;

	Vector3 m_force;
	Vector3 m_torque;

	real_t m_chord = 0.5;
	real_t m_span = 4.0;
	real_t m_taper = 1.0;
	real_t m_twist = 0.0;
	real_t m_sweep = 0.0;
	real_t m_dihedral = 0.0;
	real_t m_offset = 0.0;
	bool m_mirror = true;

	real_t m_lift_slope = Math::TAU;
	real_t m_zero_lift_angle = 0.0;
	real_t m_stall_angle_max = Math::deg_to_rad(16.0);
	real_t m_stall_angle_min = Math::deg_to_rad(-16.0);
	real_t m_stall_width = Math::deg_to_rad(6.0);
	real_t m_surface_friction = 0.023;
	real_t m_density = 1.2255;
	real_t m_restore_stall_angle = Math::deg_to_rad(5.0);
	bool m_alternative_drag = false;

	real_t m_flap_start = 0.1;
	real_t m_flap_end = 0.4;
	real_t m_flap_fraction = 0.3;
	real_t m_flap_angle_max = Math::deg_to_rad(30.0);
	real_t m_flap_angle_min = Math::deg_to_rad(-30.0);

	real_t m_aileron_start = 0.5;
	real_t m_aileron_end = 0.9;
	real_t m_aileron_fraction = 0.2;
	real_t m_aileron_angle_max = Math::deg_to_rad(15.0);
	real_t m_aileron_angle_min = Math::deg_to_rad(-15.0);

	real_t m_aileron_value = 0.0;
	real_t m_flap_value = 0.0;

	bool m_dirty = true;
	Vector<Section> m_sections;
	real_t m_aspect_ratio = 1.0;

	void apply_forces();
	void try_rebuild();
	void update_aspect_ratio();
	void calculate_section_forces(Section &p_section, const Vector3 &p_wind);
	void update_section_parameters(Section &p_section, const Vector3 p_wind);
	void update_section_hysteresis_stall(Section &p_section, const Vector3 p_wind);
	void calculate_section_factors(Section &p_section, const Vector3 &p_wind);
	void calculate_normal_coefficients(const Section &p_section, real_t p_angle_of_attack, real_t &p_lift, real_t &p_drag, real_t &p_torque) const;
	void calculate_stall_coefficients(const Section &p_section, real_t p_angle_of_attack, real_t &p_lift, real_t &p_drag, real_t &p_torque) const;

	real_t get_control_surface_angle(ControlSurfaceType p_type, bool p_mirror) const;
	real_t get_torque_factor(real_t p_effective_angle) const;
	real_t get_control_surface_lift_max(real_t p_control_surface_fraction) const;
	real_t get_drag_max(real_t p_control_surface_angle) const;
	real_t get_control_surface_lift_factor(real_t p_control_surface_angle) const;

	static real_t get_angle_of_attack(const Vector3 &p_wind);

	PackedStringArray get_configuration_warnings() const override;
	void _notification(int p_what);
	void on_enter_tree();
	void on_exit_tree();

protected:
	static void _bind_methods();

public:
	void calculate(const Vector3 &p_linear_velocity, const Vector3 &p_angular_velocity, const Vector3 &p_center_of_mass);

	Vector3 get_tip() const;
	bool has_flap() const;
	bool has_aileron() const;
	real_t get_mac() const;
	real_t get_mac_right_position() const;
	real_t get_mac_forward_position() const;
	real_t get_console_length() const;

	int get_section_count() const { return m_sections.size(); }
	real_t get_section_length(int p_index) const;
	real_t get_section_chord(int p_index) const;
	Transform3D get_section_transform(int p_index) const;
	real_t get_section_angle_of_attack(int p_index) const;
	real_t get_section_control_surface_fraction(int p_index) const;
	real_t get_section_control_surface_angle(int p_index) const;
	bool is_section_stall_warning(int p_index) const;
	bool is_section_stall(int p_index) const;

	_ALWAYS_INLINE_ Vector3 get_force() const { return m_force; }
	_ALWAYS_INLINE_ Vector3 get_torque() const { return m_torque; }
	_ALWAYS_INLINE_ Vector3 get_base() const { return Vector3(m_offset, 0.0, 0.0); }

	void set_span(real_t p_span);
	_ALWAYS_INLINE_ real_t get_span() const { return m_span; }

	void set_chord(real_t p_chord);
	_ALWAYS_INLINE_ real_t get_chord() const { return m_chord; }

	void set_taper(real_t p_taper);
	_ALWAYS_INLINE_ real_t get_taper() const { return m_taper; }

	void set_twist(real_t p_twist);
	_ALWAYS_INLINE_ real_t get_twist() const { return m_twist; }

	void set_sweep(real_t p_sweep);
	_ALWAYS_INLINE_ real_t get_sweep() const { return m_sweep; }

	void set_dihedral(real_t p_dihedral);
	_ALWAYS_INLINE_ real_t get_dihedral() const { return m_dihedral; }

	void set_offset(real_t p_offset);
	_ALWAYS_INLINE_ real_t get_offset() const { return m_offset; }

	void set_mirror(bool p_mirror);
	_ALWAYS_INLINE_ bool is_mirror() const { return m_mirror; }

	_ALWAYS_INLINE_ void set_lift_slope(real_t p_lift_slope) { m_lift_slope = p_lift_slope; }
	_ALWAYS_INLINE_ real_t get_lift_slope() const { return m_lift_slope; }

	_ALWAYS_INLINE_ void set_zero_lift_angle(real_t p_zero_lift_angle) { m_zero_lift_angle = p_zero_lift_angle; }
	_ALWAYS_INLINE_ real_t get_zero_lift_angle() const { return m_zero_lift_angle; }

	_ALWAYS_INLINE_ void set_stall_angle_max(real_t p_stall_angle_max) { m_stall_angle_max = p_stall_angle_max; }
	_ALWAYS_INLINE_ real_t get_stall_angle_max() const { return m_stall_angle_max; }

	_ALWAYS_INLINE_ void set_stall_angle_min(real_t p_stall_angle_min) { m_stall_angle_min = p_stall_angle_min; }
	_ALWAYS_INLINE_ real_t get_stall_angle_min() const { return m_stall_angle_min; }

	_ALWAYS_INLINE_ void set_stall_width(real_t p_stall_width) { m_stall_width = p_stall_width; }
	_ALWAYS_INLINE_ real_t get_stall_width() const { return m_stall_width; }

	_ALWAYS_INLINE_ void set_surface_friction(real_t p_surface_friction) { m_surface_friction = p_surface_friction; }
	_ALWAYS_INLINE_ real_t get_surface_friction() const { return m_surface_friction; }

	_ALWAYS_INLINE_ void set_density(real_t p_density) { m_density = p_density; }
	_ALWAYS_INLINE_ real_t get_density() const { return m_density; }

	_ALWAYS_INLINE_ void set_restore_stall_angle(real_t p_restore_stall_angle) { m_restore_stall_angle = p_restore_stall_angle; }
	_ALWAYS_INLINE_ real_t get_restore_stall_angle() const { return m_restore_stall_angle; }

	_ALWAYS_INLINE_ void set_alternative_drag(bool p_alternative_drag) { m_alternative_drag = p_alternative_drag; }
	_ALWAYS_INLINE_ bool get_alternative_drag() const { return m_alternative_drag; }

	void set_flap_start(real_t p_flap_start);
	_ALWAYS_INLINE_ real_t get_flap_start() const { return m_flap_start; }

	void set_flap_end(real_t p_flap_end);
	_ALWAYS_INLINE_ real_t get_flap_end() const { return m_flap_end; }

	void set_flap_fraction(real_t p_flap_fraction);
	_ALWAYS_INLINE_ real_t get_flap_fraction() const { return m_flap_fraction; }

	_ALWAYS_INLINE_ void set_flap_angle_max(real_t p_flap_angle_max) { m_flap_angle_max = p_flap_angle_max; }
	_ALWAYS_INLINE_ real_t get_flap_angle_max() const { return m_flap_angle_max; }

	_ALWAYS_INLINE_ void set_flap_angle_min(real_t p_flap_angle_min) { m_flap_angle_min = p_flap_angle_min; }
	_ALWAYS_INLINE_ real_t get_flap_angle_min() const { return m_flap_angle_min; }

	void set_aileron_start(real_t p_aileron_start);
	_ALWAYS_INLINE_ real_t get_aileron_start() const { return m_aileron_start; }

	void set_aileron_end(real_t p_aileron_end);
	_ALWAYS_INLINE_ real_t get_aileron_end() const { return m_aileron_end; }

	void set_aileron_fraction(real_t p_aileron_fraction);
	_ALWAYS_INLINE_ real_t get_aileron_fraction() const { return m_aileron_fraction; }

	_ALWAYS_INLINE_ void set_aileron_angle_max(real_t p_aileron_angle_max) { m_aileron_angle_max = p_aileron_angle_max; }
	_ALWAYS_INLINE_ real_t get_aileron_angle_max() const { return m_aileron_angle_max; }

	_ALWAYS_INLINE_ void set_aileron_angle_min(real_t p_aileron_angle_min) { m_aileron_angle_min = p_aileron_angle_min; }
	_ALWAYS_INLINE_ real_t get_aileron_angle_min() const { return m_aileron_angle_min; }

	void set_aileron_value(real_t p_aileron_value);
	_ALWAYS_INLINE_ real_t get_aileron_value() const { return m_aileron_value; }

	void set_flap_value(real_t p_flap_value);
	_ALWAYS_INLINE_ real_t get_flap_value() const { return m_flap_value; }

	real_t get_aileron_angle(bool p_mirror) const;
	real_t get_flap_angle() const;
};

class WingBuilder {
public:
	static void build_wing_sections(const VehicleWing3D *p_wing, Vector<VehicleWing3D::Section> &p_sections);
	static void build_control_surface_sections(const VehicleWing3D *p_wing, Vector<VehicleWing3D::ControlSurface> &p_sections);

private:
	static real_t get_nominal_section_length(const VehicleWing3D *p_wing);
	static VehicleWing3D::Section create_wing_section(const VehicleWing3D *p_wing, const VehicleWing3D::ControlSurface &p_control_surface, const Vector3 &p_position, real_t p_chord, real_t p_length, real_t p_twist, bool p_mirror);

	struct DeflectorComparator {
		_ALWAYS_INLINE_ bool operator()(const VehicleWing3D::ControlSurface &p_a, const VehicleWing3D::ControlSurface &p_b) const { return (p_a.start < p_b.start); }
	};
};

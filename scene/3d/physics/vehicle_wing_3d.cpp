/**************************************************************************/
/*  vehicle_wing_3d.cpp                                                   */
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

#include "vehicle_wing_3d.h"
#include "scene/3d/physics/rigid_body_3d.h"

void VehicleWing3D::apply_forces() {
	if (m_body == nullptr) {
		return;
	}

	PhysicsServer3D *physics_server = PhysicsServer3D::get_singleton();
	if (physics_server == nullptr) {
		return;
	}

	PhysicsDirectBodyState3D *state = physics_server->body_get_direct_state(m_body->get_rid());
	if (state != nullptr) {
		calculate(state->get_linear_velocity(), state->get_angular_velocity(), m_body->get_transform().xform(state->get_center_of_mass_local()));
		m_body->apply_central_force(m_force);
		m_body->apply_torque(m_torque);
	}
}

void VehicleWing3D::try_rebuild() {
	if (!m_dirty) {
		return;
	}

	m_sections.clear();
	m_dirty = false;

	WingBuilder::build_wing_sections(this, m_sections);
	update_aspect_ratio();
}

void VehicleWing3D::update_aspect_ratio() {
	real_t mean_chord = (m_chord + m_chord * m_taper) * 0.5;
	if (Math::abs(mean_chord) < 0.001) {
		m_aspect_ratio = 1.0;
		return;
	}

	m_aspect_ratio = m_span / mean_chord;

	if (Math::abs(m_aspect_ratio) < 0.001) {
		m_aspect_ratio = 1.0;
	}
}

void VehicleWing3D::calculate(const Vector3 &p_linear_velocity, const Vector3 &p_angular_velocity, const Vector3 &p_center_of_mass) {
	try_rebuild();
	m_force = Vector3(0.0, 0.0, 0.0);
	m_torque = Vector3(0.0, 0.0, 0.0);

	Transform3D global_transform = get_global_transform();

	for (Section &section : m_sections) {
		section.global_transform = global_transform * section.transform;
		Vector3 arm = section.global_transform.get_origin() - p_center_of_mass;
		Vector3 wind = -(p_linear_velocity + p_angular_velocity.cross(arm));

		calculate_section_forces(section, wind);

		m_force += section.force;
		m_torque += section.torque + arm.cross(section.force);
	}
}

void VehicleWing3D::calculate_section_forces(Section &p_section, const Vector3 &p_wind) {
	Vector3 right = p_section.global_transform.get_basis().xform(Vector3(1.0, 0.0, 0.0));
	Vector3 drag_direction = p_wind.normalized();
	Vector3 lift_direction = drag_direction.cross(right);

	update_section_parameters(p_section, p_wind);
	calculate_section_factors(p_section, p_wind);

	real_t pressure = 0.5 * m_density * p_wind.length_squared() * p_section.chord * p_section.length;
	Vector3 lift = lift_direction * p_section.lift_factor * pressure;
	Vector3 drag = drag_direction * p_section.drag_factor * pressure;
	Vector3 force = lift + drag;
	Vector3 torque = -right * p_section.torque_factor * pressure * p_section.chord;

	force = p_section.force + (force - p_section.force) * 0.5;
	torque = p_section.torque + (torque - p_section.torque) * 0.5;
	p_section.force = force;
	p_section.torque = torque;
}

void VehicleWing3D::update_section_parameters(Section &p_section, const Vector3 p_wind) {
	Transform3D to_local = p_section.global_transform.affine_inverse();
	Vector3 local_wind = to_local.xform(p_wind) - to_local.xform(Vector3(0.0, 0.0, 0.0));

	p_section.angle_of_attack = get_angle_of_attack(local_wind);
	p_section.control_surface_angle = get_control_surface_angle(p_section.type, p_section.mirror);
	p_section.corrected_lift_slope = m_lift_slope * m_aspect_ratio / (m_aspect_ratio + 2.0 * (m_aspect_ratio + 4.0) / (m_aspect_ratio + 2.0));

	real_t control_surface_effectivness_factor = Math::acos(2.0 * p_section.control_surface_fraction - 1.0);
	real_t control_surface_effectivness = 1.0 - (control_surface_effectivness_factor - Math::sin(control_surface_effectivness_factor)) / Math::PI;

	p_section.control_surface_lift = p_section.corrected_lift_slope * control_surface_effectivness * get_control_surface_lift_factor(p_section.control_surface_angle) * p_section.control_surface_angle;
	p_section.corrected_zero_lift_angle = m_zero_lift_angle - p_section.control_surface_lift / p_section.corrected_lift_slope;

	real_t control_surface_lift_max = get_control_surface_lift_max(p_section.control_surface_fraction);
	real_t lift_max = p_section.corrected_lift_slope * (m_stall_angle_max - m_zero_lift_angle) + p_section.control_surface_lift * control_surface_lift_max;
	real_t lift_min = p_section.corrected_lift_slope * (m_stall_angle_min - m_zero_lift_angle) + p_section.control_surface_lift * control_surface_lift_max;

	p_section.corrected_stall_angle_max = p_section.corrected_zero_lift_angle + lift_max / p_section.corrected_lift_slope;
	p_section.corrected_stall_angle_min = p_section.corrected_zero_lift_angle + lift_min / p_section.corrected_lift_slope;

	update_section_hysteresis_stall(p_section, p_wind);
}

void VehicleWing3D::update_section_hysteresis_stall(Section &p_section, const Vector3 p_wind) {
	if (p_wind.length_squared() < m_chord * m_chord) {
		p_section.stall = false;
		return;
	}

	real_t start_hysteresis_angle_max = p_section.corrected_stall_angle_max + m_stall_width;
	real_t start_hysteresis_angle_min = p_section.corrected_stall_angle_min - m_stall_width;
	p_section.restore_stall_angle_max = MIN(p_section.corrected_stall_angle_max, m_restore_stall_angle);
	p_section.restore_stall_angle_min = MAX(p_section.corrected_stall_angle_min, -m_restore_stall_angle);
	if (!p_section.stall && (p_section.angle_of_attack >= start_hysteresis_angle_max || p_section.angle_of_attack <= start_hysteresis_angle_min)) {
		p_section.stall = true;
	} else if (p_section.stall && p_section.angle_of_attack <= m_restore_stall_angle && p_section.angle_of_attack >= -m_restore_stall_angle) {
		p_section.stall = false;
	}
}

void VehicleWing3D::calculate_section_factors(Section &p_section, const Vector3 &p_wind) {
	real_t stall_angle_max = p_section.stall ? p_section.restore_stall_angle_max : p_section.corrected_stall_angle_max;
	real_t stall_angle_min = p_section.stall ? p_section.restore_stall_angle_min : p_section.corrected_stall_angle_min;

	if (p_section.angle_of_attack >= stall_angle_min && p_section.angle_of_attack <= stall_angle_max) {
		calculate_normal_coefficients(p_section, p_section.angle_of_attack, p_section.lift_factor, p_section.drag_factor, p_section.torque_factor);
		p_section.stall_warning = false;
		return;
	}

	p_section.stall_warning = p_wind.length_squared() >= m_chord * m_chord;

	real_t full_stall_angle_max = p_section.corrected_stall_angle_max + m_stall_width;
	real_t full_stall_angle_min = p_section.corrected_stall_angle_min - m_stall_width;

	if (p_section.angle_of_attack > full_stall_angle_max || p_section.angle_of_attack < full_stall_angle_min) {
		calculate_stall_coefficients(p_section, p_section.angle_of_attack, p_section.lift_factor, p_section.drag_factor, p_section.torque_factor);
		return;
	}

	real_t lift1, lift2, drag1, drag2, torque1, torque2, w;

	if (p_section.angle_of_attack > stall_angle_max) {
		calculate_normal_coefficients(p_section, stall_angle_max, lift1, drag1, torque1);
		calculate_stall_coefficients(p_section, full_stall_angle_max, lift2, drag2, torque2);
		w = (p_section.angle_of_attack - stall_angle_max) / (full_stall_angle_max - stall_angle_max);
	} else {
		calculate_normal_coefficients(p_section, stall_angle_min, lift1, drag1, torque1);
		calculate_stall_coefficients(p_section, full_stall_angle_min, lift2, drag2, torque2);
		w = (p_section.angle_of_attack - stall_angle_min) / (full_stall_angle_min - stall_angle_min);
	}

	w = w * w * (3 - 2 * w);
	p_section.lift_factor = Math::lerp(lift1, lift2, w);
	p_section.drag_factor = Math::lerp(drag1, drag2, w);
	p_section.torque_factor = Math::lerp(torque1, torque2, w);
}

void VehicleWing3D::calculate_normal_coefficients(const Section &p_section, real_t p_angle_of_attack, real_t &p_lift, real_t &p_drag, real_t &p_torque) const {
	p_lift = p_section.corrected_lift_slope * (p_angle_of_attack - p_section.corrected_zero_lift_angle);

	real_t induced_angle = p_lift / (Math::PI * m_aspect_ratio);
	real_t effective_angle = p_angle_of_attack - p_section.corrected_zero_lift_angle - induced_angle;
	real_t cos_ea = Math::cos(effective_angle);
	real_t sin_ea = Math::sin(effective_angle);
	real_t tangent = m_surface_friction * cos_ea;
	real_t normal = Math::abs(cos_ea) >= 0.001 ? (p_lift + sin_ea * tangent) / cos_ea : 0.0;

	if (m_alternative_drag) {
		real_t k = 1.0 / (Math::PI * m_aspect_ratio * 0.8);
		p_drag = m_surface_friction + k * p_lift * p_lift;
	} else {
		p_drag = normal * sin_ea + tangent * cos_ea;
	}

	p_torque = p_section.control_surface_lift / 6.0 - normal * get_torque_factor(effective_angle);
}

void VehicleWing3D::calculate_stall_coefficients(const Section &p_section, real_t p_angle_of_attack, real_t &p_lift, real_t &p_drag, real_t &p_torque) const {
	real_t stall_angle = p_angle_of_attack > p_section.corrected_stall_angle_max
			? p_section.corrected_stall_angle_max
			: p_section.corrected_stall_angle_min;

	real_t stall_lift = p_section.corrected_lift_slope * (stall_angle - p_section.corrected_zero_lift_angle);
	real_t induced_angle = stall_lift / (Math::PI * m_aspect_ratio);

	real_t half_pi = Math::PI / 2.0;
	real_t z = p_angle_of_attack > p_section.corrected_stall_angle_max
			? half_pi - p_section.corrected_stall_angle_max
			: -half_pi - p_section.corrected_stall_angle_min;

	real_t w = Math::abs(z) >= 0.001 ? (half_pi - CLAMP(p_angle_of_attack, -half_pi, half_pi)) / z : 0.0;

	induced_angle = Math::lerp((real_t)0.0, induced_angle, w);

	real_t effective_angle = p_angle_of_attack - p_section.corrected_zero_lift_angle - induced_angle;
	real_t sin_ea = Math::sin(effective_angle);
	real_t cos_ea = Math::cos(effective_angle);

	real_t normal = get_drag_max(p_section.control_surface_angle) * sin_ea * (1.0 / (0.56 + 0.44 * Math::abs(sin_ea)) - 0.41 * (1.0 - exp(-17.0 / m_aspect_ratio)));
	real_t tangent = 0.5 * m_surface_friction * cos_ea;

	p_lift = normal * cos_ea - tangent * sin_ea;
	p_drag = normal * sin_ea + tangent * cos_ea;
	p_torque = -normal * get_torque_factor(effective_angle);
}

real_t VehicleWing3D::get_control_surface_angle(ControlSurfaceType p_type, bool p_mirror) const {
	if (p_type == ControlSurfaceType::Aileron) {
		return get_aileron_angle(p_mirror);
	} else if (p_type == ControlSurfaceType::Flap) {
		return get_flap_angle();
	}

	return 0.0;
}

real_t VehicleWing3D::get_torque_factor(real_t p_effective_angle) const {
	return 0.25 - 0.175 * (1.0 - 2.0 * p_effective_angle / Math::PI);
}

real_t VehicleWing3D::get_control_surface_lift_max(real_t p_control_surface_fraction) const {
	return CLAMP(1.0 - 0.5 * (p_control_surface_fraction - 0.1) / 0.3, 0.0, 1.0);
}

real_t VehicleWing3D::get_drag_max(real_t p_control_surface_angle) const {
	return 1.98 - 4.26e-2 * p_control_surface_angle * p_control_surface_angle + 2.1e-1 * p_control_surface_angle;
}

real_t VehicleWing3D::get_control_surface_lift_factor(real_t p_control_surface_angle) const {
	return Math::lerp(0.8, 0.4, (Math::abs(Math::rad_to_deg(p_control_surface_angle)) - 10.0) / 50.0);
}

real_t VehicleWing3D::get_angle_of_attack(const Vector3 &p_wind) {
	real_t angle = Math::atan2(p_wind.y, p_wind.z);

	if (angle > Math::PI / 2.0) {
		angle -= Math::PI;
	} else if (angle < -Math::PI / 2.0) {
		angle += Math::PI;
	}

	return angle;
}

PackedStringArray VehicleWing3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();

	if (Object::cast_to<RigidBody3D>(get_parent()) == nullptr) {
		warnings.push_back(RTR("VehicleWing3D serves to provide a aerodynamic to a VehicleBody3D or RigidBody3D. Please use it as a child of a VehicleBody3D or RigidBody3D."));
	}

	return warnings;
}

void VehicleWing3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
			on_enter_tree();
			break;
		case NOTIFICATION_EXIT_TREE:
			on_exit_tree();
			break;
		case NOTIFICATION_PHYSICS_PROCESS:
			apply_forces();
			break;
	}
}

void VehicleWing3D::on_enter_tree() {
	m_body = Object::cast_to<RigidBody3D>(get_parent());
	try_rebuild();
	set_physics_process(m_body != nullptr && !Engine::get_singleton()->is_editor_hint());
}

void VehicleWing3D::on_exit_tree() {
	m_body = nullptr;
}

Vector3 VehicleWing3D::get_tip() const {
	Vector3 direction = Vector3(1.0, 0.0, 0.0);
	direction.rotate(Vector3(0.0, -1.0, 0.0), m_sweep);
	direction.rotate(Vector3(0.0, 0.0, 1.0), m_dihedral);
	Vector3 tip = Math::abs(direction.x) > 0.001 ? direction * get_console_length() / direction.x : direction * get_console_length();
	return get_base() + tip;
}

bool VehicleWing3D::has_aileron() const {
	return m_aileron_start != m_aileron_end && m_aileron_fraction > 0.0;
}

bool VehicleWing3D::has_flap() const {
	return m_flap_start != m_flap_end && m_flap_fraction > 0.0;
}

real_t VehicleWing3D::get_mac() const {
	return 2.0 / 3.0 * m_chord * (1.0 + m_taper + m_taper * m_taper) / (1.0 + m_taper);
}

real_t VehicleWing3D::get_mac_right_position() const {
	real_t length = m_mirror ? get_console_length() * 2.0 : get_console_length();
	return length / 6.0 * (1.0 + 2.0 * m_taper) / (1.0 + m_taper);
}

real_t VehicleWing3D::get_mac_forward_position() const {
	real_t mac = get_mac();
	real_t position = mac / 4.0 * (1.0 - m_taper);

	if (m_sweep != 0.0) {
		position += Math::tan(m_sweep) * get_mac_right_position();
	}

	return position - (m_chord - mac) * 0.5;
}

real_t VehicleWing3D::get_console_length() const {
	if (m_mirror) {
		return m_span / 2.0 - m_offset;
	}

	return m_span - get_offset();
}

real_t VehicleWing3D::get_section_length(int p_index) const {
	if (p_index < 0) {
		p_index += m_sections.size();
	}
	ERR_FAIL_INDEX_V(p_index, (int)m_sections.size(), 0.0);
	return m_sections[p_index].length;
}

real_t VehicleWing3D::get_section_chord(int p_index) const {
	if (p_index < 0) {
		p_index += m_sections.size();
	}
	ERR_FAIL_INDEX_V(p_index, (int)m_sections.size(), 0.0);
	return m_sections[p_index].chord;
}

Transform3D VehicleWing3D::get_section_transform(int p_index) const {
	if (p_index < 0) {
		p_index += m_sections.size();
	}
	ERR_FAIL_INDEX_V(p_index, (int)m_sections.size(), get_transform());
	return m_sections[p_index].transform;
}

real_t VehicleWing3D::get_section_angle_of_attack(int p_index) const {
	if (p_index < 0) {
		p_index += m_sections.size();
	}
	ERR_FAIL_INDEX_V(p_index, (int)m_sections.size(), 0.0);
	return m_sections[p_index].angle_of_attack;
}

real_t VehicleWing3D::get_section_control_surface_fraction(int p_index) const {
	if (p_index < 0) {
		p_index += m_sections.size();
	}
	ERR_FAIL_INDEX_V(p_index, (int)m_sections.size(), 0.0);
	return m_sections[p_index].control_surface_fraction;
}

real_t VehicleWing3D::get_section_control_surface_angle(int p_index) const {
	if (p_index < 0) {
		p_index += m_sections.size();
	}
	ERR_FAIL_INDEX_V(p_index, (int)m_sections.size(), 0.0);
	return m_sections[p_index].control_surface_angle;
}

bool VehicleWing3D::is_section_stall_warning(int p_index) const {
	if (p_index < 0) {
		p_index += m_sections.size();
	}
	ERR_FAIL_INDEX_V(p_index, (int)m_sections.size(), false);
	return m_sections[p_index].stall_warning;
}

bool VehicleWing3D::is_section_stall(int p_index) const {
	if (p_index < 0) {
		p_index += m_sections.size();
	}
	ERR_FAIL_INDEX_V(p_index, (int)m_sections.size(), false);
	return m_sections[p_index].stall;
}

void VehicleWing3D::set_span(real_t p_span) {
	m_span = p_span;
	m_dirty = true;
	update_gizmos();
}

void VehicleWing3D::set_chord(real_t p_chord) {
	m_chord = p_chord;
	m_dirty = true;
	update_gizmos();
}

void VehicleWing3D::set_taper(real_t p_taper) {
	m_taper = p_taper;
	m_dirty = true;
	update_gizmos();
}

void VehicleWing3D::set_twist(real_t p_twist) {
	m_twist = p_twist;
	m_dirty = true;
	update_gizmos();
}

void VehicleWing3D::set_sweep(real_t p_sweep) {
	m_sweep = p_sweep;
	m_dirty = true;
	update_gizmos();
}

void VehicleWing3D::set_dihedral(real_t dihedral) {
	m_dihedral = dihedral;
	m_dirty = true;
	update_gizmos();
}

void VehicleWing3D::set_offset(real_t p_offset) {
	m_offset = p_offset;
	m_dirty = true;
	update_gizmos();
}

void VehicleWing3D::set_mirror(bool p_mirror) {
	m_mirror = p_mirror;
	m_dirty = true;
	update_gizmos();
}

void VehicleWing3D::set_flap_start(real_t p_flap_start) {
	m_flap_start = p_flap_start;
	update_gizmos();
}

void VehicleWing3D::set_flap_end(real_t p_flap_end) {
	m_flap_end = p_flap_end;
	update_gizmos();
}

void VehicleWing3D::set_flap_fraction(real_t p_flap_fraction) {
	m_flap_fraction = p_flap_fraction;
	update_gizmos();
}

void VehicleWing3D::set_aileron_start(real_t p_aileron_start) {
	m_aileron_start = p_aileron_start;
	update_gizmos();
}

void VehicleWing3D::set_aileron_end(real_t p_aileron_end) {
	m_aileron_end = p_aileron_end;
	update_gizmos();
}

void VehicleWing3D::set_aileron_fraction(real_t p_aileron_fraction) {
	m_aileron_fraction = p_aileron_fraction;
	update_gizmos();
}

void VehicleWing3D::set_aileron_value(real_t p_aileron_value) {
	m_aileron_value = p_aileron_value;
	update_gizmos();
}

void VehicleWing3D::set_flap_value(real_t p_flap_value) {
	m_flap_value = p_flap_value;
	update_gizmos();
}

real_t VehicleWing3D::get_aileron_angle(bool p_mirror) const {
	if (p_mirror) {
		return (m_aileron_value > 0.0 ? m_aileron_angle_min : -m_aileron_angle_max) * m_aileron_value;
	}

	return (m_aileron_value > 0.0 ? m_aileron_angle_max : -m_aileron_angle_min) * m_aileron_value;
}

real_t VehicleWing3D::get_flap_angle() const {
	return (m_flap_value >= 0.0 ? m_flap_angle_max : -m_flap_angle_min) * m_flap_value;
}

void VehicleWing3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("calculate", "linear_velocity", "angular_velocity", "p_center_of_mass"), &VehicleWing3D::calculate);
	ClassDB::bind_method(D_METHOD("get_force"), &VehicleWing3D::get_force);
	ClassDB::bind_method(D_METHOD("get_torque"), &VehicleWing3D::get_torque);
	ClassDB::bind_method(D_METHOD("get_mac"), &VehicleWing3D::get_mac);

	ClassDB::bind_method(D_METHOD("get_section_count"), &VehicleWing3D::get_section_count);
	ClassDB::bind_method(D_METHOD("get_section_length", "index"), &VehicleWing3D::get_section_length);
	ClassDB::bind_method(D_METHOD("get_section_chord", "index"), &VehicleWing3D::get_section_chord);
	ClassDB::bind_method(D_METHOD("get_section_transform", "index"), &VehicleWing3D::get_section_transform);
	ClassDB::bind_method(D_METHOD("get_section_angle_of_attack", "index"), &VehicleWing3D::get_section_angle_of_attack);
	ClassDB::bind_method(D_METHOD("get_section_control_surface_fraction", "index"), &VehicleWing3D::get_section_control_surface_fraction);
	ClassDB::bind_method(D_METHOD("get_section_control_surface_angle", "index"), &VehicleWing3D::get_section_control_surface_angle);
	ClassDB::bind_method(D_METHOD("is_section_stall_warning", "index"), &VehicleWing3D::is_section_stall_warning);
	ClassDB::bind_method(D_METHOD("is_section_stall", "index"), &VehicleWing3D::is_section_stall);

	ClassDB::bind_method(D_METHOD("set_span", "span"), &VehicleWing3D::set_span);
	ClassDB::bind_method(D_METHOD("get_span"), &VehicleWing3D::get_span);
	ClassDB::bind_method(D_METHOD("set_chord", "chord"), &VehicleWing3D::set_chord);
	ClassDB::bind_method(D_METHOD("get_chord"), &VehicleWing3D::get_chord);
	ClassDB::bind_method(D_METHOD("set_taper", "taper"), &VehicleWing3D::set_taper);
	ClassDB::bind_method(D_METHOD("get_taper"), &VehicleWing3D::get_taper);
	ClassDB::bind_method(D_METHOD("set_twist", "twist"), &VehicleWing3D::set_twist);
	ClassDB::bind_method(D_METHOD("get_twist"), &VehicleWing3D::get_twist);
	ClassDB::bind_method(D_METHOD("set_sweep", "sweep"), &VehicleWing3D::set_sweep);
	ClassDB::bind_method(D_METHOD("get_sweep"), &VehicleWing3D::get_sweep);
	ClassDB::bind_method(D_METHOD("set_dihedral", "dihedral"), &VehicleWing3D::set_dihedral);
	ClassDB::bind_method(D_METHOD("get_dihedral"), &VehicleWing3D::get_dihedral);
	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &VehicleWing3D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &VehicleWing3D::get_offset);
	ClassDB::bind_method(D_METHOD("set_mirror", "mirror"), &VehicleWing3D::set_mirror);
	ClassDB::bind_method(D_METHOD("is_mirror"), &VehicleWing3D::is_mirror);
	ClassDB::bind_method(D_METHOD("set_lift_slope", "lift_slope"), &VehicleWing3D::set_lift_slope);
	ClassDB::bind_method(D_METHOD("get_lift_slope"), &VehicleWing3D::get_lift_slope);
	ClassDB::bind_method(D_METHOD("set_zero_lift_angle", "zero_lift_angle"), &VehicleWing3D::set_zero_lift_angle);
	ClassDB::bind_method(D_METHOD("get_zero_lift_angle"), &VehicleWing3D::get_zero_lift_angle);
	ClassDB::bind_method(D_METHOD("set_stall_angle_max", "stall_angle_max"), &VehicleWing3D::set_stall_angle_max);
	ClassDB::bind_method(D_METHOD("get_stall_angle_max"), &VehicleWing3D::get_stall_angle_max);
	ClassDB::bind_method(D_METHOD("set_stall_angle_min", "stall_angle_min"), &VehicleWing3D::set_stall_angle_min);
	ClassDB::bind_method(D_METHOD("get_stall_angle_min"), &VehicleWing3D::get_stall_angle_min);
	ClassDB::bind_method(D_METHOD("set_stall_width", "stall_width"), &VehicleWing3D::set_stall_width);
	ClassDB::bind_method(D_METHOD("get_stall_width"), &VehicleWing3D::get_stall_width);
	ClassDB::bind_method(D_METHOD("set_surface_friction", "surface_friction"), &VehicleWing3D::set_surface_friction);
	ClassDB::bind_method(D_METHOD("get_surface_friction"), &VehicleWing3D::get_surface_friction);
	ClassDB::bind_method(D_METHOD("set_density", "density"), &VehicleWing3D::set_density);
	ClassDB::bind_method(D_METHOD("get_density"), &VehicleWing3D::get_density);
	ClassDB::bind_method(D_METHOD("set_restore_stall_angle", "restore_stall_angle"), &VehicleWing3D::set_restore_stall_angle);
	ClassDB::bind_method(D_METHOD("get_restore_stall_angle"), &VehicleWing3D::get_restore_stall_angle);
	ClassDB::bind_method(D_METHOD("set_alternative_drag", "alternative_drag"), &VehicleWing3D::set_alternative_drag);
	ClassDB::bind_method(D_METHOD("get_alternative_drag"), &VehicleWing3D::get_alternative_drag);
	ClassDB::bind_method(D_METHOD("set_flap_start", "flap_start"), &VehicleWing3D::set_flap_start);
	ClassDB::bind_method(D_METHOD("get_flap_start"), &VehicleWing3D::get_flap_start);
	ClassDB::bind_method(D_METHOD("set_flap_end", "flap_end"), &VehicleWing3D::set_flap_end);
	ClassDB::bind_method(D_METHOD("get_flap_end"), &VehicleWing3D::get_flap_end);
	ClassDB::bind_method(D_METHOD("set_flap_fraction", "flap_fraction"), &VehicleWing3D::set_flap_fraction);
	ClassDB::bind_method(D_METHOD("get_flap_fraction"), &VehicleWing3D::get_flap_fraction);
	ClassDB::bind_method(D_METHOD("set_flap_angle_max", "flap_angle_max"), &VehicleWing3D::set_flap_angle_max);
	ClassDB::bind_method(D_METHOD("get_flap_angle_max"), &VehicleWing3D::get_flap_angle_max);
	ClassDB::bind_method(D_METHOD("set_flap_angle_min", "flap_angle_min"), &VehicleWing3D::set_flap_angle_min);
	ClassDB::bind_method(D_METHOD("get_flap_angle_min"), &VehicleWing3D::get_flap_angle_min);
	ClassDB::bind_method(D_METHOD("set_aileron_start", "aileron_start"), &VehicleWing3D::set_aileron_start);
	ClassDB::bind_method(D_METHOD("get_aileron_start"), &VehicleWing3D::get_aileron_start);
	ClassDB::bind_method(D_METHOD("set_aileron_end", "aileron_end"), &VehicleWing3D::set_aileron_end);
	ClassDB::bind_method(D_METHOD("get_aileron_end"), &VehicleWing3D::get_aileron_end);
	ClassDB::bind_method(D_METHOD("set_aileron_fraction", "aileron_fraction"), &VehicleWing3D::set_aileron_fraction);
	ClassDB::bind_method(D_METHOD("get_aileron_fraction"), &VehicleWing3D::get_aileron_fraction);
	ClassDB::bind_method(D_METHOD("set_aileron_angle_max", "aileron_angle_max"), &VehicleWing3D::set_aileron_angle_max);
	ClassDB::bind_method(D_METHOD("get_aileron_angle_max"), &VehicleWing3D::get_aileron_angle_max);
	ClassDB::bind_method(D_METHOD("set_aileron_angle_min", "aileron_angle_min"), &VehicleWing3D::set_aileron_angle_min);
	ClassDB::bind_method(D_METHOD("get_aileron_angle_min"), &VehicleWing3D::get_aileron_angle_min);
	ClassDB::bind_method(D_METHOD("set_aileron_value", "aileron_value"), &VehicleWing3D::set_aileron_value);
	ClassDB::bind_method(D_METHOD("get_aileron_value"), &VehicleWing3D::get_aileron_value);
	ClassDB::bind_method(D_METHOD("set_flap_value", "flap_value"), &VehicleWing3D::set_flap_value);
	ClassDB::bind_method(D_METHOD("get_flap_value"), &VehicleWing3D::get_flap_value);

	ADD_GROUP("Wing shape", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "span"), "set_span", "get_span");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "chord"), "set_chord", "get_chord");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "taper", PROPERTY_HINT_RANGE, "0,1"), "set_taper", "get_taper");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "twist", PROPERTY_HINT_RANGE, "-15,15,radians_as_degrees"), "set_twist", "get_twist");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sweep", PROPERTY_HINT_RANGE, "-70,70,radians_as_degrees"), "set_sweep", "get_sweep");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dihedral", PROPERTY_HINT_RANGE, "-30,30,radians_as_degrees"), "set_dihedral", "get_dihedral");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mirror"), "set_mirror", "is_mirror");

	ADD_GROUP("Aerodynamic", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lift_slope"), "set_lift_slope", "get_lift_slope");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zero_lift_angle", PROPERTY_HINT_RANGE, "-10,10,radians_as_degrees"), "set_zero_lift_angle", "get_zero_lift_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stall_angle_max", PROPERTY_HINT_RANGE, "0,30,radians_as_degrees"), "set_stall_angle_max", "get_stall_angle_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stall_angle_min", PROPERTY_HINT_RANGE, "-30,0,radians_as_degrees"), "set_stall_angle_min", "get_stall_angle_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stall_width", PROPERTY_HINT_RANGE, "0,30,radians_as_degrees"), "set_stall_width", "get_stall_width");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "surface_friction", PROPERTY_HINT_RANGE, "0,1"), "set_surface_friction", "get_surface_friction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "density"), "set_density", "get_density");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "restore_stall_angle", PROPERTY_HINT_RANGE, "0,30,radians_as_degrees"), "set_restore_stall_angle", "get_restore_stall_angle");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alternative_drag"), "set_alternative_drag", "get_alternative_drag");

	ADD_GROUP("Control surfaces", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "flap_start", PROPERTY_HINT_RANGE, "0,1"), "set_flap_start", "get_flap_start");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "flap_end", PROPERTY_HINT_RANGE, "0,1"), "set_flap_end", "get_flap_end");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "flap_fraction", PROPERTY_HINT_RANGE, "0,0.9"), "set_flap_fraction", "get_flap_fraction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "flap_angle_max", PROPERTY_HINT_RANGE, "0,90,radians_as_degrees"), "set_flap_angle_max", "get_flap_angle_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "flap_angle_min", PROPERTY_HINT_RANGE, "-90,0,radians_as_degrees"), "set_flap_angle_min", "get_flap_angle_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "aileron_start", PROPERTY_HINT_RANGE, "0,1"), "set_aileron_start", "get_aileron_start");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "aileron_end", PROPERTY_HINT_RANGE, "0,1"), "set_aileron_end", "get_aileron_end");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "aileron_fraction", PROPERTY_HINT_RANGE, "0,0.9"), "set_aileron_fraction", "get_aileron_fraction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "aileron_angle_max", PROPERTY_HINT_RANGE, "0,90,radians_as_degrees"), "set_aileron_angle_max", "get_aileron_angle_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "aileron_angle_min", PROPERTY_HINT_RANGE, "-90,0,radians_as_degrees"), "set_aileron_angle_min", "get_aileron_angle_min");

	ADD_GROUP("Input", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "aileron_value", PROPERTY_HINT_RANGE, "-1,1"), "set_aileron_value", "get_aileron_value");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "flap_value", PROPERTY_HINT_RANGE, "-1,1"), "set_flap_value", "get_flap_value");
}

void WingBuilder::build_wing_sections(const VehicleWing3D *p_wing, Vector<VehicleWing3D::Section> &p_sections) {
	p_sections.clear();

	real_t nominal_section_length = get_nominal_section_length(p_wing);
	if (nominal_section_length <= 0) {
		return;
	}

	Vector<VehicleWing3D::ControlSurface> control_surface_sections;
	build_control_surface_sections(p_wing, control_surface_sections);

	Vector3 base = p_wing->get_base();
	Vector3 tip = p_wing->get_tip();
	real_t mac_z = p_wing->get_mac_forward_position();
	real_t offset_z = base.z - mac_z;
	base.z += offset_z;
	tip.z += offset_z;

	for (const VehicleWing3D::ControlSurface &control_surface : control_surface_sections) {
		real_t bound_size = control_surface.end - control_surface.start;
		int section_count = (int)Math::ceil(bound_size / nominal_section_length);
		if (section_count <= 0) {
			continue;
		}

		real_t section_length = p_wing->get_console_length() * bound_size / section_count;

		for (int i = 0; i < section_count; i++) {
			real_t fraction = control_surface.start + (i + 0.5) * bound_size / section_count;
			Vector3 position = base + (tip - base) * fraction;
			real_t chord = p_wing->get_chord() * (1.0 - (1.0 - p_wing->get_taper()) * fraction);
			real_t twist = p_wing->get_twist() * fraction;

			p_sections.append(create_wing_section(p_wing, control_surface, position, chord, section_length, twist, false));

			if (p_wing->is_mirror()) {
				position.x = -position.x;
				p_sections.append(create_wing_section(p_wing, control_surface, position, chord, section_length, twist, true));
			}
		}
	}
}

void WingBuilder::build_control_surface_sections(const VehicleWing3D *p_wing, Vector<VehicleWing3D::ControlSurface> &p_sections) {
	Vector<VehicleWing3D::ControlSurface> control_surfaces;
	p_sections.clear();

	if (p_wing->has_flap()) {
		VehicleWing3D::ControlSurface flap;
		flap.type = VehicleWing3D::ControlSurfaceType::Flap;
		flap.start = p_wing->get_flap_start();
		flap.end = p_wing->get_flap_end();
		flap.fraction = p_wing->get_flap_fraction();
		control_surfaces.append(flap);
	}

	if (p_wing->has_aileron()) {
		VehicleWing3D::ControlSurface aileron;
		aileron.type = VehicleWing3D::ControlSurfaceType::Aileron;
		aileron.start = p_wing->get_aileron_start();
		aileron.end = p_wing->get_aileron_end();
		aileron.fraction = p_wing->get_aileron_fraction();
		control_surfaces.append(aileron);
	}

	if (control_surfaces.size() <= 0) {
		VehicleWing3D::ControlSurface empty;
		empty.type = VehicleWing3D::ControlSurfaceType::None;
		empty.start = 0.0;
		empty.end = 1.0;
		empty.fraction = 0.0;
		p_sections.append(empty);
		return;
	}

	control_surfaces.sort_custom<DeflectorComparator>();

	real_t pos = 0.0;

	for (int i = 0; i < control_surfaces.size(); i++) {
		const VehicleWing3D::ControlSurface &control_surface = control_surfaces[i];
		if (pos < control_surface.start) {
			VehicleWing3D::ControlSurface section;
			section.type = VehicleWing3D::ControlSurfaceType::None;
			section.start = pos;
			section.end = control_surface.start;
			section.fraction = 0.0;
			p_sections.push_back(section);
		}

		p_sections.push_back(control_surface);
		pos = control_surface.end;

		if (i == control_surfaces.size() - 1 && pos < 1.0) {
			VehicleWing3D::ControlSurface section;
			section.type = VehicleWing3D::ControlSurfaceType::None;
			section.start = pos;
			section.end = 1.0;
			section.fraction = 0.0;
			p_sections.push_back(section);
		}
	}

	p_sections.sort_custom<DeflectorComparator>();
}

real_t WingBuilder::get_nominal_section_length(const VehicleWing3D *p_wing) {
	real_t tip_chord = p_wing->get_chord() * p_wing->get_taper();
	real_t mid_chord = (p_wing->get_chord() + tip_chord) / 2.0;
	real_t section_length = mid_chord / p_wing->get_console_length() / 2.0;
	if (section_length <= 0.0) {
		return 0.0;
	}

	int section_count = (int)Math::ceil(1.0 / section_length);

	return p_wing->get_twist() != 0.0 && section_count < 8 ? 1.0 / 8.0 : section_length;
}

VehicleWing3D::Section WingBuilder::create_wing_section(const VehicleWing3D *p_wing, const VehicleWing3D::ControlSurface &p_control_surface, const Vector3 &p_position, real_t p_chord, real_t p_length, real_t p_twist, bool p_mirror) {
	VehicleWing3D::Section section;

	section.transform = Transform3D(Basis::from_euler(Vector3(p_twist, 0.0, p_mirror ? -p_wing->get_dihedral() : p_wing->get_dihedral())), p_position);
	section.chord = p_chord;
	section.length = p_length;
	section.type = p_control_surface.type;
	section.mirror = p_mirror;
	section.stall = false;

	if (p_control_surface.type != VehicleWing3D::ControlSurfaceType::None && p_control_surface.fraction > 0.0) {
		section.control_surface_fraction = p_control_surface.fraction;
	} else {
		section.control_surface_fraction = 0.0;
	}

	return section;
}

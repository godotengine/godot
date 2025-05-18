/**************************************************************************/
/*  vehicle_wing_3d_gizmo_plugin.cpp                                      */
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

#include "vehicle_wing_3d_gizmo_plugin.h"
#include "core/math/math_funcs.h"

VehicleWing3DGizmoPlugin::VehicleWing3DGizmoPlugin() {
	create_material("wing_material", Color(0.0, 1.0, 0.0, 1.0));
	create_material("error_material", Color(1.0, 0.0, 0.0, 1.0));
	create_material("mac_material", Color(1.0, 1.0, 0.0, 1.0));
}

bool VehicleWing3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<VehicleWing3D>(p_spatial) != nullptr;
}

String VehicleWing3DGizmoPlugin::get_gizmo_name() const {
	return "VehicleWing3D";
}

int VehicleWing3DGizmoPlugin::get_priority() const {
	return -1;
}

void VehicleWing3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	const VehicleWing3D *wing = Object::cast_to<VehicleWing3D>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	if (wing == nullptr) {
		return;
	}

	Vector<Vector3> lines;

	bool valid = add_wing_lines(wing, false, lines);
	if (wing->is_mirror()) {
		add_wing_lines(wing, true, lines);
	}

	String material = valid ? "wing_material" : "error_material";
	p_gizmo->add_lines(lines, get_material(material, p_gizmo), false, Color(1.0, 1.0, 1.0, 1.0));

	lines.clear();
	add_mac(wing, lines);
	p_gizmo->add_lines(lines, get_material("mac_material", p_gizmo), false, Color(1.0, 1.0, 1.0, 1.0));
}

bool VehicleWing3DGizmoPlugin::add_wing_lines(const VehicleWing3D *p_wing, bool p_mirror, Vector<Vector3> &p_lines) {
	Vector3 base = p_wing->get_base();
	Vector3 tip = p_wing->get_tip();
	real_t chord = p_wing->get_chord();
	real_t taper = p_wing->get_taper();
	real_t twist = p_wing->get_twist();
	real_t mac = p_wing->get_mac();
	real_t mac_z = p_wing->get_mac_forward_position();

	bool valid = p_wing->get_span() > 0.0 && chord > 0.0 && taper >= 0.0 && p_wing->get_offset() >= 0.0 && p_wing->get_console_length() > 0.0;

	base.z += mac * 0.25 - mac_z;
	tip += (mac * 0.25 - mac_z) * Vector3(0.0, 0.0, 1.0).rotated(Vector3(1.0, 0.0, 0.0), twist);

	if (p_mirror) {
		base.x *= -1.0;
		tip.x *= -1.0;
	}

	Vector3 w1 = get_wing_point(base, tip, 0.0, -chord * 0.5, 0.0);
	Vector3 w2 = get_wing_point(base, tip, 1.0, -chord * 0.5 * taper, twist);

	p_lines.append(w1);
	p_lines.append(w2);

	Vector<VehicleWing3D::ControlSurface> sections;
	WingBuilder::build_control_surface_sections(p_wing, sections);

	real_t last_end = 0.0;

	for (int i = 0; i < sections.size(); i++) {
		const VehicleWing3D::ControlSurface &section = sections[i];
		real_t start_chord = Math::lerp(chord, chord * taper, section.start);
		real_t end_chord = Math::lerp(chord, chord * taper, section.end);

		if (section.type == VehicleWing3D::ControlSurfaceType::None) {
			Vector3 d1 = get_wing_point(base, tip, section.start, start_chord * 0.5, twist * section.start);
			Vector3 d2 = get_wing_point(base, tip, section.end, end_chord * 0.5, twist * section.end);
			p_lines.append(d1);
			p_lines.append(d2);

			if (i == 0) {
				p_lines.append(w1);
				p_lines.append(d1);
			}
			if (i == sections.size() - 1) {
				p_lines.append(w2);
				p_lines.append(d2);
			}
		} else {
			Vector3 d1 = get_wing_point(base, tip, section.start, start_chord * 0.5, twist * section.start);
			Vector3 d2 = get_wing_point(base, tip, section.start, start_chord * 0.5 - start_chord * section.fraction, twist * section.start);
			Vector3 d3 = get_wing_point(base, tip, section.end, end_chord * 0.5, twist * section.end);
			Vector3 d4 = get_wing_point(base, tip, section.end, end_chord * 0.5 - end_chord * section.fraction, twist * section.end);

			if (section.start > 0.0) {
				p_lines.append(d1);
				p_lines.append(d2);
			}
			if (section.end < 1.0) {
				p_lines.append(d3);
				p_lines.append(d4);
			}
			p_lines.append(d2);
			p_lines.append(d4);

			real_t angle = section.type == VehicleWing3D::ControlSurfaceType::Flap ? p_wing->get_flap_angle() : p_wing->get_aileron_angle(p_mirror);
			if (p_mirror) {
				angle *= -1.0;
			}

			Vector3 axis = (d4 - d2).normalized();
			Vector3 r1 = d2 + (d1 - d2).rotated(axis, angle);
			Vector3 r2 = d4 + (d3 - d4).rotated(axis, angle);
			p_lines.append(d2);
			p_lines.append(r1);
			p_lines.append(r1);
			p_lines.append(r2);
			p_lines.append(r2);
			p_lines.append(d4);

			if (i == 0) {
				p_lines.append(w1);
				p_lines.append(d2);
			}
			if (i == sections.size() - 1) {
				p_lines.append(w2);
				p_lines.append(d4);
			}
		}

		valid = valid && section.start <= section.end && section.start >= 0.0 && section.start <= 1.0 && section.end <= 1.0 && section.end >= 0.0 && section.fraction >= 0.0 && section.fraction < 1.0 && section.start >= last_end;
		last_end = section.end;
	}

	return valid;
}

void VehicleWing3DGizmoPlugin::add_mac(const VehicleWing3D *p_wing, Vector<Vector3> &p_lines) {
	real_t mac = p_wing->get_mac();
	real_t half_mac = mac * 0.5;
	real_t mac_z = 0.25 * mac;

	p_lines.append(Vector3(0.1, 0.0, mac_z + half_mac));
	p_lines.append(Vector3(-0.1, 0.0, mac_z + half_mac));

	p_lines.append(Vector3(0.1, 0.0, mac_z - half_mac));
	p_lines.append(Vector3(-0.1, 0.0, mac_z - half_mac));
}

Vector3 VehicleWing3DGizmoPlugin::get_wing_point(const Vector3 &p_base, const Vector3 &p_tip, real_t p_pos, real_t p_chord, real_t p_twist) {
	Vector3 point = p_base + (p_tip - p_base) * p_pos;
	Vector3 twist_dir = Vector3(0.0, 0.0, 1.0).rotated(Vector3(1.0, 0.0, 0.0), p_twist);
	point += twist_dir * p_chord;
	return point;
}

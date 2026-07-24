/**************************************************************************/
/*  audio_stream_player_3d_gizmo_plugin.cpp                               */
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

#include "audio_stream_player_3d_gizmo_plugin.h"

#include "core/math/geometry_3d.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "scene/3d/audio_stream_player_3d.h"

// IDs for the gizmo's draggable handles.
enum HandleID {
	HANDLE_MAX_DISTANCE = 0,
	HANDLE_EMISSION_ANGLE = 1,
};

AudioStreamPlayer3DGizmoPlugin::AudioStreamPlayer3DGizmoPlugin() {
	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/stream_player_3d");

	create_icon_material("stream_player_3d_icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Gizmo3DSamplePlayer"), EditorStringName(EditorIcons)));
	create_material("stream_player_3d_material_primary", gizmo_color);
	create_material("stream_player_3d_material_secondary", gizmo_color * Color(1, 1, 1, 0.35));
	// Enable vertex colors so the range circles can reflect the attenuation model
	// and whether Max Distance is capping the range (see redraw()).
	create_material("stream_player_3d_material_lines", Color(1, 1, 1), false, false, true);
	create_material("stream_player_3d_material_billboard", Color(1, 1, 1), true, false, true);
	create_handle_material("handles");
	create_handle_material("handles_billboard", true);
}

bool AudioStreamPlayer3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<AudioStreamPlayer3D>(p_spatial) != nullptr;
}

String AudioStreamPlayer3DGizmoPlugin::get_gizmo_name() const {
	return "AudioStreamPlayer3D";
}

int AudioStreamPlayer3DGizmoPlugin::get_priority() const {
	return -1;
}

String AudioStreamPlayer3DGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	if (p_id == HANDLE_MAX_DISTANCE) {
		return "Max Distance";
	}
	return "Emission Radius";
}

Variant AudioStreamPlayer3DGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	AudioStreamPlayer3D *player = Object::cast_to<AudioStreamPlayer3D>(p_gizmo->get_node_3d());
	if (p_id == HANDLE_MAX_DISTANCE) {
		return player->get_max_distance();
	}
	return player->get_emission_angle();
}

void AudioStreamPlayer3DGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	AudioStreamPlayer3D *player = Object::cast_to<AudioStreamPlayer3D>(p_gizmo->get_node_3d());

	Transform3D gt = player->get_global_transform();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	if (p_id == HANDLE_MAX_DISTANCE) {
		// Max Distance: use the distance from the node to the cursor on a camera-facing plane.
		Plane cp = Plane(p_camera->get_transform().basis.get_column(2), gt.origin);
		Vector3 inters;
		if (cp.intersects_ray(ray_from, ray_dir, &inters)) {
			float r = inters.distance_to(gt.origin);
			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				r = Math::snapped(r, Node3DEditor::get_singleton()->get_translate_snap());
			}
			player->set_max_distance(r);
		}
		return;
	}

	// Emission Radius (angle).
	Transform3D gi = gt.affine_inverse();
	Vector3 ray_to = ray_from + ray_dir * 4096;

	ray_from = gi.xform(ray_from);
	ray_to = gi.xform(ray_to);

	float closest_dist = 1e20;
	float closest_angle = 1e20;

	for (int i = 0; i < 180; i++) {
		float a = Math::deg_to_rad((float)i);
		float an = Math::deg_to_rad((float)(i + 1));

		Vector3 from(Math::sin(a), 0, -Math::cos(a));
		Vector3 to(Math::sin(an), 0, -Math::cos(an));

		Vector3 r1, r2;
		Geometry3D::get_closest_points_between_segments(from, to, ray_from, ray_to, r1, r2);
		float d = r1.distance_to(r2);
		if (d < closest_dist) {
			closest_dist = d;
			closest_angle = i;
		}
	}

	if (closest_angle < 91) {
		player->set_emission_angle(closest_angle);
	}
}

void AudioStreamPlayer3DGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	AudioStreamPlayer3D *player = Object::cast_to<AudioStreamPlayer3D>(p_gizmo->get_node_3d());

	if (p_id == HANDLE_MAX_DISTANCE) {
		if (p_cancel) {
			player->set_max_distance(p_restore);
		} else {
			EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
			ur->create_action(TTR("Change AudioStreamPlayer3D Max Distance"));
			ur->add_do_method(player, "set_max_distance", player->get_max_distance());
			ur->add_undo_method(player, "set_max_distance", p_restore);
			ur->commit_action();
		}
		return;
	}

	if (p_cancel) {
		player->set_emission_angle(p_restore);

	} else {
		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Change AudioStreamPlayer3D Emission Angle"));
		ur->add_do_method(player, "set_emission_angle", player->get_emission_angle());
		ur->add_undo_method(player, "set_emission_angle", p_restore);
		ur->commit_action();
	}
}

void AudioStreamPlayer3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	p_gizmo->clear();

	if (p_gizmo->is_selected()) {
		const AudioStreamPlayer3D *player = Object::cast_to<AudioStreamPlayer3D>(p_gizmo->get_node_3d());

		if (player->get_attenuation_model() != AudioStreamPlayer3D::ATTENUATION_DISABLED || player->get_max_distance() > CMP_EPSILON) {
			// Draw the attenuation range as three axis-aligned circles plus a billboard circle,
			// giving a sphere-like representation (matching OmniLight3D).
			const Ref<Material> lines_material = get_material("stream_player_3d_material_lines", p_gizmo);
			const Ref<Material> lines_billboard_material = get_material("stream_player_3d_material_billboard", p_gizmo);

			// Soft distance cap varies depending on attenuation model, as some will fade out more aggressively than others.
			// Multipliers were empirically determined through testing.
			float soft_multiplier;
			switch (player->get_attenuation_model()) {
				case AudioStreamPlayer3D::ATTENUATION_INVERSE_DISTANCE:
					soft_multiplier = 12.0;
					break;
				case AudioStreamPlayer3D::ATTENUATION_INVERSE_SQUARE_DISTANCE:
					soft_multiplier = 4.0;
					break;
				case AudioStreamPlayer3D::ATTENUATION_LOGARITHMIC:
					soft_multiplier = 3.25;
					break;
				default:
					// Ensures Max Distance's radius visualization is not capped by Unit Size
					// (when the attenuation mode is Disabled).
					soft_multiplier = 10000.0;
					break;
			}

			// With a hard cap, draw the circle at Max Distance so it lines up with the draggable handle;
			// otherwise fall back to the Unit Size-derived soft radius (where the sound fades to near silence).
			float radius;
			if (player->get_max_distance() > CMP_EPSILON) {
				radius = player->get_max_distance();
			} else {
				radius = player->get_unit_size() * soft_multiplier;
			}

			Vector<Vector3> points;
			Vector<Vector3> points_billboard;
			for (int i = 0; i < 120; i++) {
				const float ra = Math::deg_to_rad((float)(i * 3));
				const float rb = Math::deg_to_rad((float)((i + 1) * 3));
				const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * radius;
				const Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * radius;

				// One segment on each of the three axis-aligned circles.
				points.push_back(Vector3(a.x, 0, a.y));
				points.push_back(Vector3(b.x, 0, b.y));
				points.push_back(Vector3(0, a.x, a.y));
				points.push_back(Vector3(0, b.x, b.y));
				points.push_back(Vector3(a.x, a.y, 0));
				points.push_back(Vector3(b.x, b.y, 0));

				// Billboard circle.
				points_billboard.push_back(Vector3(a.x, a.y, 0));
				points_billboard.push_back(Vector3(b.x, b.y, 0));
			}

			Color color;
			switch (player->get_attenuation_model()) {
				// Pick cold colors for all attenuation models (except Disabled),
				// so that soft caps can be easily distinguished from hard caps
				// (which use warm colors).
				case AudioStreamPlayer3D::ATTENUATION_INVERSE_DISTANCE:
					color = Color(0.4, 0.8, 1);
					break;
				case AudioStreamPlayer3D::ATTENUATION_INVERSE_SQUARE_DISTANCE:
					color = Color(0.4, 0.5, 1);
					break;
				case AudioStreamPlayer3D::ATTENUATION_LOGARITHMIC:
					color = Color(0.4, 0.2, 1);
					break;
				default:
					// Disabled attenuation mode.
					// This is never reached when Max Distance is 0, but the
					// hue-inverted form of this color will be used if Max Distance is greater than 0.
					color = Color(1, 1, 1);
					break;
			}

			if (player->get_max_distance() > CMP_EPSILON) {
				// Sound is hard-capped by max distance. The attenuation model still matters,
				// so invert the hue of the color that was chosen above.
				color.set_h(color.get_h() + 0.5);
			}

			p_gizmo->add_lines(points, lines_material, false, color);
			p_gizmo->add_lines(points_billboard, lines_billboard_material, true, color);

			if (player->get_max_distance() > CMP_EPSILON) {
				// Handle to edit Max Distance directly in the viewport, placed on the billboard circle.
				Vector<Vector3> distance_handles;
				distance_handles.push_back(Vector3(player->get_max_distance(), 0, 0));
				// A gizmo uses one billboard flag for all handles, so only billboard the Max Distance
				// handle when the fixed emission-angle handle isn't also present (keeps picking consistent).
				if (player->is_emission_angle_enabled()) {
					p_gizmo->add_handles(distance_handles, get_material("handles"), { HANDLE_MAX_DISTANCE });
				} else {
					p_gizmo->add_handles(distance_handles, get_material("handles_billboard"), { HANDLE_MAX_DISTANCE }, true);
				}
			}
		}

		if (player->is_emission_angle_enabled()) {
			const float ha = Math::deg_to_rad(player->get_emission_angle());
			const float ofs = -Math::cos(ha);
			const float radius = Math::sin(ha);

			const uint32_t points_in_octant = 7;
			const real_t octant_angle = Math::PI / 4;
			const real_t inc = (Math::PI / (4 * points_in_octant));
			const real_t radius_squared = radius * radius;
			real_t r = 0;

			Vector<Vector3> points_primary;
			points_primary.resize(8 * points_in_octant * 2);
			Vector3 *points_ptrw = points_primary.ptrw();

			uint32_t index = 0;
			float previous_x = radius;
			float previous_y = 0.f;
#define PUSH_QUARTER(m_from_x, m_from_y, m_to_x, m_to_y, m_y) \
	points_ptrw[index++] = Vector3(m_from_x, -m_from_y, m_y); \
	points_ptrw[index++] = Vector3(m_to_x, -m_to_y, m_y); \
	points_ptrw[index++] = Vector3(m_from_x, m_from_y, m_y); \
	points_ptrw[index++] = Vector3(m_to_x, m_to_y, m_y); \
	points_ptrw[index++] = Vector3(-m_from_x, -m_from_y, m_y); \
	points_ptrw[index++] = Vector3(-m_to_x, -m_to_y, m_y); \
	points_ptrw[index++] = Vector3(-m_from_x, m_from_y, m_y); \
	points_ptrw[index++] = Vector3(-m_to_x, m_to_y, m_y);

			for (uint32_t i = 0; i < points_in_octant; i++) {
				r += inc;
				real_t x = Math::cos((i == points_in_octant - 1) ? octant_angle : r) * radius;
				real_t y = Math::sqrt(radius_squared - (x * x));

				PUSH_QUARTER(previous_x, previous_y, x, y, ofs);
				PUSH_QUARTER(previous_y, previous_x, y, x, ofs);

				previous_x = x;
				previous_y = y;
			}
#undef PUSH_QUARTER

			const Ref<Material> material_primary = get_material("stream_player_3d_material_primary", p_gizmo);
			p_gizmo->add_lines(points_primary, material_primary);

			Vector<Vector3> points_secondary;
			points_secondary.resize(16);
			Vector3 *points_second_ptrw = points_secondary.ptrw();
			uint32_t index2 = 0;
			// Lines to the circle.
			points_second_ptrw[index2++] = Vector3();
			points_second_ptrw[index2++] = Vector3(radius, 0, ofs);
			points_second_ptrw[index2++] = Vector3();
			points_second_ptrw[index2++] = Vector3(-radius, 0, ofs);
			points_second_ptrw[index2++] = Vector3();
			points_second_ptrw[index2++] = Vector3(0, radius, ofs);
			points_second_ptrw[index2++] = Vector3();
			points_second_ptrw[index2++] = Vector3(0, -radius, ofs);
			real_t octant_value = Math::cos(octant_angle) * radius;
			points_second_ptrw[index2++] = Vector3();
			points_second_ptrw[index2++] = Vector3(octant_value, octant_value, ofs);
			points_second_ptrw[index2++] = Vector3();
			points_second_ptrw[index2++] = Vector3(-octant_value, octant_value, ofs);
			points_second_ptrw[index2++] = Vector3();
			points_second_ptrw[index2++] = Vector3(-octant_value, -octant_value, ofs);
			points_second_ptrw[index2++] = Vector3();
			points_second_ptrw[index2++] = Vector3(octant_value, -octant_value, ofs);

			const Ref<Material> material_secondary = get_material("stream_player_3d_material_secondary", p_gizmo);
			p_gizmo->add_lines(points_secondary, material_secondary);

			Vector<Vector3> handles;
			handles.push_back(Vector3(Math::sin(ha), 0, -Math::cos(ha)));
			p_gizmo->add_handles(handles, get_material("handles"), { HANDLE_EMISSION_ANGLE });
		}
	}

	const Ref<Material> icon = get_material("stream_player_3d_icon", p_gizmo);
	p_gizmo->add_unscaled_billboard(icon, 0.05);
}

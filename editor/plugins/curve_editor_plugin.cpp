/*************************************************************************/
/*  curve_editor_plugin.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "curve_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "core/core_string_names.h"
#include "core/input/input.h"
#include "core/os/keyboard.h"
#include "editor/editor_scale.h"

CurveEditor::CurveEditor() {
	_selected_point = -1;
	_hover_point = -1;
	_selected_tangent = TANGENT_NONE;
	_hover_radius = 6;
	_tangents_length = 40;
	_dragging = false;
	_has_undo_data = false;

	set_focus_mode(FOCUS_ALL);
	set_clip_contents(true);

	_context_menu = memnew(PopupMenu);
	_context_menu->connect("id_pressed", callable_mp(this, &CurveEditor::on_context_menu_item_selected));
	add_child(_context_menu);

	_presets_menu = memnew(PopupMenu);
	_presets_menu->set_name("_presets_menu");
	_presets_menu->add_item(TTR("Flat 0"), PRESET_FLAT0);
	_presets_menu->add_item(TTR("Flat 1"), PRESET_FLAT1);
	_presets_menu->add_item(TTR("Linear"), PRESET_LINEAR);
	_presets_menu->add_item(TTR("Ease In"), PRESET_EASE_IN);
	_presets_menu->add_item(TTR("Ease Out"), PRESET_EASE_OUT);
	_presets_menu->add_item(TTR("Smoothstep"), PRESET_SMOOTHSTEP);
	_presets_menu->connect("id_pressed", callable_mp(this, &CurveEditor::on_preset_item_selected));
	_context_menu->add_child(_presets_menu);
}

void CurveEditor::set_curve(Ref<Curve> curve) {
	if (curve == _curve_ref) {
		return;
	}

	if (_curve_ref.is_valid()) {
		_curve_ref->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &CurveEditor::_curve_changed));
		_curve_ref->disconnect(Curve::SIGNAL_RANGE_CHANGED, callable_mp(this, &CurveEditor::_curve_changed));
	}

	_curve_ref = curve;

	if (_curve_ref.is_valid()) {
		_curve_ref->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &CurveEditor::_curve_changed));
		_curve_ref->connect(Curve::SIGNAL_RANGE_CHANGED, callable_mp(this, &CurveEditor::_curve_changed));
	}

	_selected_point = -1;
	_hover_point = -1;
	_selected_tangent = TANGENT_NONE;

	update();

	// Note: if you edit a curve, then set another, and try to undo,
	// it will normally apply on the previous curve, but you won't see it
}

Size2 CurveEditor::get_minimum_size() const {
	return Vector2(64, 150) * EDSCALE;
}

void CurveEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		_draw();
	}
}

void CurveEditor::gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb_ref = p_event;
	if (mb_ref.is_valid()) {
		const InputEventMouseButton &mb = **mb_ref;

		if (mb.is_pressed() && !_dragging) {
			Vector2 mpos = mb.get_position();

			_selected_tangent = get_tangent_at(mpos);
			if (_selected_tangent == TANGENT_NONE) {
				set_selected_point(get_point_at(mpos));
			}

			switch (mb.get_button_index()) {
				case MouseButton::RIGHT:
					_context_click_pos = mpos;
					open_context_menu(get_global_transform().xform(mpos));
					break;

				case MouseButton::MIDDLE:
					remove_point(_hover_point);
					break;

				case MouseButton::LEFT:
					_dragging = true;
					break;
				default:
					break;
			}
		}

		if (!mb.is_pressed() && _dragging && mb.get_button_index() == MouseButton::LEFT) {
			_dragging = false;
			if (_has_undo_data) {
				UndoRedo &ur = *EditorNode::get_singleton()->get_undo_redo();

				ur.create_action(_selected_tangent == TANGENT_NONE ? TTR("Modify Curve Point") : TTR("Modify Curve Tangent"));
				ur.add_do_method(*_curve_ref, "_set_data", _curve_ref->get_data());
				ur.add_undo_method(*_curve_ref, "_set_data", _undo_data);
				// Note: this will trigger one more "changed" signal even if nothing changes,
				// but it's ok since it would have fired every frame during the drag anyways
				ur.commit_action();

				_has_undo_data = false;
			}
		}
	}

	Ref<InputEventMouseMotion> mm_ref = p_event;
	if (mm_ref.is_valid()) {
		const InputEventMouseMotion &mm = **mm_ref;

		Vector2 mpos = mm.get_position();

		if (_dragging && _curve_ref.is_valid()) {
			if (_selected_point != -1) {
				Curve &curve = **_curve_ref;

				if (!_has_undo_data) {
					// Save full curve state before dragging points,
					// because this operation can modify their order
					_undo_data = curve.get_data();
					_has_undo_data = true;
				}

				const float curve_amplitude = curve.get_max_value() - curve.get_min_value();
				// Snap to "round" coordinates when holding Ctrl.
				// Be more precise when holding Shift as well.
				float snap_threshold;
				if (mm.is_ctrl_pressed()) {
					snap_threshold = mm.is_shift_pressed() ? 0.025 : 0.1;
				} else {
					snap_threshold = 0.0;
				}

				if (_selected_tangent == TANGENT_NONE) {
					// Drag point

					Vector2 point_pos = get_world_pos(mpos).snapped(Vector2(snap_threshold, snap_threshold * curve_amplitude));

					int i = curve.set_point_offset(_selected_point, point_pos.x);
					// The index may change if the point is dragged across another one
					set_hover_point_index(i);
					set_selected_point(i);

					// This is to prevent the user from losing a point out of view.
					if (point_pos.y < curve.get_min_value()) {
						point_pos.y = curve.get_min_value();
					} else if (point_pos.y > curve.get_max_value()) {
						point_pos.y = curve.get_max_value();
					}

					curve.set_point_value(_selected_point, point_pos.y);

				} else {
					// Drag tangent

					const Vector2 point_pos = curve.get_point_position(_selected_point);
					const Vector2 control_pos = get_world_pos(mpos).snapped(Vector2(snap_threshold, snap_threshold * curve_amplitude));

					Vector2 dir = (control_pos - point_pos).normalized();

					real_t tangent;
					if (!Math::is_zero_approx(dir.x)) {
						tangent = dir.y / dir.x;
					} else {
						tangent = 9999 * (dir.y >= 0 ? 1 : -1);
					}

					bool link = !Input::get_singleton()->is_key_pressed(Key::SHIFT);

					if (_selected_tangent == TANGENT_LEFT) {
						curve.set_point_left_tangent(_selected_point, tangent);

						// Note: if a tangent is set to linear, it shouldn't be linked to the other
						if (link && _selected_point != (curve.get_point_count() - 1) && curve.get_point_right_mode(_selected_point) != Curve::TANGENT_LINEAR) {
							curve.set_point_right_tangent(_selected_point, tangent);
						}

					} else {
						curve.set_point_right_tangent(_selected_point, tangent);

						if (link && _selected_point != 0 && curve.get_point_left_mode(_selected_point) != Curve::TANGENT_LINEAR) {
							curve.set_point_left_tangent(_selected_point, tangent);
						}
					}
				}
			}

		} else {
			set_hover_point_index(get_point_at(mpos));
		}
	}

	Ref<InputEventKey> key_ref = p_event;
	if (key_ref.is_valid()) {
		const InputEventKey &key = **key_ref;

		if (key.is_pressed() && _selected_point != -1) {
			if (key.get_keycode() == Key::KEY_DELETE) {
				remove_point(_selected_point);
			}
		}
	}
}

void CurveEditor::on_preset_item_selected(int preset_id) {
	ERR_FAIL_COND(preset_id < 0 || preset_id >= PRESET_COUNT);
	ERR_FAIL_COND(_curve_ref.is_null());

	Curve &curve = **_curve_ref;
	Array previous_data = curve.get_data();

	curve.clear_points();

	switch (preset_id) {
		case PRESET_FLAT0:
			curve.add_point(Vector2(0, 0));
			curve.add_point(Vector2(1, 0));
			curve.set_point_right_mode(0, Curve::TANGENT_LINEAR);
			curve.set_point_left_mode(1, Curve::TANGENT_LINEAR);
			break;

		case PRESET_FLAT1:
			curve.add_point(Vector2(0, 1));
			curve.add_point(Vector2(1, 1));
			curve.set_point_right_mode(0, Curve::TANGENT_LINEAR);
			curve.set_point_left_mode(1, Curve::TANGENT_LINEAR);
			break;

		case PRESET_LINEAR:
			curve.add_point(Vector2(0, 0));
			curve.add_point(Vector2(1, 1));
			curve.set_point_right_mode(0, Curve::TANGENT_LINEAR);
			curve.set_point_left_mode(1, Curve::TANGENT_LINEAR);
			break;

		case PRESET_EASE_IN:
			curve.add_point(Vector2(0, 0));
			curve.add_point(Vector2(1, 1), (curve.get_max_value() - curve.get_min_value()) * 1.4, 0);
			break;

		case PRESET_EASE_OUT:
			curve.add_point(Vector2(0, 0), 0, (curve.get_max_value() - curve.get_min_value()) * 1.4);
			curve.add_point(Vector2(1, 1));
			break;

		case PRESET_SMOOTHSTEP:
			curve.add_point(Vector2(0, 0));
			curve.add_point(Vector2(1, 1));
			break;

		default:
			break;
	}

	UndoRedo &ur = *EditorNode::get_singleton()->get_undo_redo();
	ur.create_action(TTR("Load Curve Preset"));

	ur.add_do_method(&curve, "_set_data", curve.get_data());
	ur.add_undo_method(&curve, "_set_data", previous_data);

	ur.commit_action();
}

void CurveEditor::_curve_changed() {
	update();
	// Point count can change in case of undo
	if (_selected_point >= _curve_ref->get_point_count()) {
		set_selected_point(-1);
	}
}

void CurveEditor::on_context_menu_item_selected(int action_id) {
	switch (action_id) {
		case CONTEXT_ADD_POINT:
			add_point(_context_click_pos);
			break;

		case CONTEXT_REMOVE_POINT:
			remove_point(_selected_point);
			break;

		case CONTEXT_LINEAR:
			toggle_linear();
			break;

		case CONTEXT_LEFT_LINEAR:
			toggle_linear(TANGENT_LEFT);
			break;

		case CONTEXT_RIGHT_LINEAR:
			toggle_linear(TANGENT_RIGHT);
			break;
	}
}

void CurveEditor::open_context_menu(Vector2 pos) {
	_context_menu->set_position(pos);

	_context_menu->clear();

	if (_curve_ref.is_valid()) {
		_context_menu->add_item(TTR("Add Point"), CONTEXT_ADD_POINT);

		if (_selected_point >= 0) {
			_context_menu->add_item(TTR("Remove Point"), CONTEXT_REMOVE_POINT);

			if (_selected_tangent != TANGENT_NONE) {
				_context_menu->add_separator();

				_context_menu->add_check_item(TTR("Linear"), CONTEXT_LINEAR);

				bool is_linear = _selected_tangent == TANGENT_LEFT
						? _curve_ref->get_point_left_mode(_selected_point) == Curve::TANGENT_LINEAR
						: _curve_ref->get_point_right_mode(_selected_point) == Curve::TANGENT_LINEAR;

				_context_menu->set_item_checked(_context_menu->get_item_index(CONTEXT_LINEAR), is_linear);

			} else {
				if (_selected_point > 0 || _selected_point + 1 < _curve_ref->get_point_count()) {
					_context_menu->add_separator();
				}

				if (_selected_point > 0) {
					_context_menu->add_check_item(TTR("Left Linear"), CONTEXT_LEFT_LINEAR);
					_context_menu->set_item_checked(_context_menu->get_item_index(CONTEXT_LEFT_LINEAR),
							_curve_ref->get_point_left_mode(_selected_point) == Curve::TANGENT_LINEAR);
				}
				if (_selected_point + 1 < _curve_ref->get_point_count()) {
					_context_menu->add_check_item(TTR("Right Linear"), CONTEXT_RIGHT_LINEAR);
					_context_menu->set_item_checked(_context_menu->get_item_index(CONTEXT_RIGHT_LINEAR),
							_curve_ref->get_point_right_mode(_selected_point) == Curve::TANGENT_LINEAR);
				}
			}
		}

		_context_menu->add_separator();
	}

	_context_menu->add_submenu_item(TTR("Load Preset"), _presets_menu->get_name());

	_context_menu->reset_size();
	_context_menu->popup();
}

int CurveEditor::get_point_at(Vector2 pos) const {
	if (_curve_ref.is_null()) {
		return -1;
	}
	const Curve &curve = **_curve_ref;

	const float true_hover_radius = Math::round(_hover_radius * EDSCALE);
	const float r = true_hover_radius * true_hover_radius;

	for (int i = 0; i < curve.get_point_count(); ++i) {
		Vector2 p = get_view_pos(curve.get_point_position(i));
		if (p.distance_squared_to(pos) <= r) {
			return i;
		}
	}

	return -1;
}

CurveEditor::TangentIndex CurveEditor::get_tangent_at(Vector2 pos) const {
	if (_curve_ref.is_null() || _selected_point < 0) {
		return TANGENT_NONE;
	}

	if (_selected_point != 0) {
		Vector2 control_pos = get_tangent_view_pos(_selected_point, TANGENT_LEFT);
		if (control_pos.distance_to(pos) < _hover_radius) {
			return TANGENT_LEFT;
		}
	}

	if (_selected_point != _curve_ref->get_point_count() - 1) {
		Vector2 control_pos = get_tangent_view_pos(_selected_point, TANGENT_RIGHT);
		if (control_pos.distance_to(pos) < _hover_radius) {
			return TANGENT_RIGHT;
		}
	}

	return TANGENT_NONE;
}

void CurveEditor::add_point(Vector2 pos) {
	ERR_FAIL_COND(_curve_ref.is_null());

	UndoRedo &ur = *EditorNode::get_singleton()->get_undo_redo();
	ur.create_action(TTR("Remove Curve Point"));

	Vector2 point_pos = get_world_pos(pos);
	if (point_pos.y < 0.0) {
		point_pos.y = 0.0;
	} else if (point_pos.y > 1.0) {
		point_pos.y = 1.0;
	}

	// Small trick to get the point index to feed the undo method
	int i = _curve_ref->add_point(point_pos);
	_curve_ref->remove_point(i);

	ur.add_do_method(*_curve_ref, "add_point", point_pos);
	ur.add_undo_method(*_curve_ref, "remove_point", i);

	ur.commit_action();
}

void CurveEditor::remove_point(int index) {
	ERR_FAIL_COND(_curve_ref.is_null());

	UndoRedo &ur = *EditorNode::get_singleton()->get_undo_redo();
	ur.create_action(TTR("Remove Curve Point"));

	Curve::Point p = _curve_ref->get_point(index);

	ur.add_do_method(*_curve_ref, "remove_point", index);
	ur.add_undo_method(*_curve_ref, "add_point", p.position, p.left_tangent, p.right_tangent, p.left_mode, p.right_mode);

	if (index == _selected_point) {
		set_selected_point(-1);
	}

	if (index == _hover_point) {
		set_hover_point_index(-1);
	}

	ur.commit_action();
}

void CurveEditor::toggle_linear(TangentIndex tangent) {
	ERR_FAIL_COND(_curve_ref.is_null());

	UndoRedo &ur = *EditorNode::get_singleton()->get_undo_redo();
	ur.create_action(TTR("Toggle Curve Linear Tangent"));

	if (tangent == TANGENT_NONE) {
		tangent = _selected_tangent;
	}

	if (tangent == TANGENT_LEFT) {
		bool is_linear = _curve_ref->get_point_left_mode(_selected_point) == Curve::TANGENT_LINEAR;

		Curve::TangentMode prev_mode = _curve_ref->get_point_left_mode(_selected_point);
		Curve::TangentMode mode = is_linear ? Curve::TANGENT_FREE : Curve::TANGENT_LINEAR;

		ur.add_do_method(*_curve_ref, "set_point_left_mode", _selected_point, mode);
		ur.add_undo_method(*_curve_ref, "set_point_left_mode", _selected_point, prev_mode);

	} else {
		bool is_linear = _curve_ref->get_point_right_mode(_selected_point) == Curve::TANGENT_LINEAR;

		Curve::TangentMode prev_mode = _curve_ref->get_point_right_mode(_selected_point);
		Curve::TangentMode mode = is_linear ? Curve::TANGENT_FREE : Curve::TANGENT_LINEAR;

		ur.add_do_method(*_curve_ref, "set_point_right_mode", _selected_point, mode);
		ur.add_undo_method(*_curve_ref, "set_point_right_mode", _selected_point, prev_mode);
	}

	ur.commit_action();
}

void CurveEditor::set_selected_point(int index) {
	if (index != _selected_point) {
		_selected_point = index;
		update();
	}
}

void CurveEditor::set_hover_point_index(int index) {
	if (index != _hover_point) {
		_hover_point = index;
		update();
	}
}

void CurveEditor::update_view_transform() {
	Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
	int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Label"));

	const real_t margin = font->get_height(font_size) + 2 * EDSCALE;

	float min_y = 0;
	float max_y = 1;

	if (_curve_ref.is_valid()) {
		min_y = _curve_ref->get_min_value();
		max_y = _curve_ref->get_max_value();
	}

	const Rect2 world_rect = Rect2(Curve::MIN_X, min_y, Curve::MAX_X, max_y - min_y);
	const Size2 view_margin(margin, margin);
	const Size2 view_size = get_size() - view_margin * 2;
	const Vector2 scale = view_size / world_rect.size;

	Transform2D world_trans;
	world_trans.translate(-world_rect.position - Vector2(0, world_rect.size.y));
	world_trans.scale(Vector2(scale.x, -scale.y));

	Transform2D view_trans;
	view_trans.translate(view_margin);

	_world_to_view = view_trans * world_trans;
}

Vector2 CurveEditor::get_tangent_view_pos(int i, TangentIndex tangent) const {
	Vector2 dir;
	if (tangent == TANGENT_LEFT) {
		dir = -Vector2(1, _curve_ref->get_point_left_tangent(i));
	} else {
		dir = Vector2(1, _curve_ref->get_point_right_tangent(i));
	}

	Vector2 point_pos = get_view_pos(_curve_ref->get_point_position(i));
	Vector2 control_pos = get_view_pos(_curve_ref->get_point_position(i) + dir);

	return point_pos + Math::round(_tangents_length * EDSCALE) * (control_pos - point_pos).normalized();
}

Vector2 CurveEditor::get_view_pos(Vector2 world_pos) const {
	return _world_to_view.xform(world_pos);
}

Vector2 CurveEditor::get_world_pos(Vector2 view_pos) const {
	return _world_to_view.affine_inverse().xform(view_pos);
}

// Uses non-baked points, but takes advantage of ordered iteration to be faster
template <typename T>
static void plot_curve_accurate(const Curve &curve, float step, T plot_func) {
	if (curve.get_point_count() <= 1) {
		// Not enough points to make a curve, so it's just a straight line
		float y = curve.interpolate(0);
		plot_func(Vector2(0, y), Vector2(1.f, y), true);

	} else {
		Vector2 first_point = curve.get_point_position(0);
		Vector2 last_point = curve.get_point_position(curve.get_point_count() - 1);

		// Edge lines
		plot_func(Vector2(0, first_point.y), first_point, false);
		plot_func(Vector2(Curve::MAX_X, last_point.y), last_point, false);

		// Draw section by section, so that we get maximum precision near points.
		// It's an accurate representation, but slower than using the baked one.
		for (int i = 1; i < curve.get_point_count(); ++i) {
			Vector2 a = curve.get_point_position(i - 1);
			Vector2 b = curve.get_point_position(i);

			Vector2 pos = a;
			Vector2 prev_pos = a;

			float len = b.x - a.x;

			for (float x = step; x < len; x += step) {
				pos.x = a.x + x;
				pos.y = curve.interpolate_local_nocheck(i - 1, x);
				plot_func(prev_pos, pos, true);
				prev_pos = pos;
			}

			plot_func(prev_pos, b, true);
		}
	}
}

struct CanvasItemPlotCurve {
	CanvasItem &ci;
	Color color1;
	Color color2;

	CanvasItemPlotCurve(CanvasItem &p_ci, Color p_color1, Color p_color2) :
			ci(p_ci),
			color1(p_color1),
			color2(p_color2) {}

	void operator()(Vector2 pos0, Vector2 pos1, bool in_definition) {
		// FIXME: Using a line width greater than 1 breaks curve rendering
		ci.draw_line(pos0, pos1, in_definition ? color1 : color2, 1);
	}
};

void CurveEditor::_draw() {
	if (_curve_ref.is_null()) {
		return;
	}
	Curve &curve = **_curve_ref;

	update_view_transform();

	// Background

	Vector2 view_size = get_rect().size;
	draw_style_box(get_theme_stylebox(SNAME("bg"), SNAME("Tree")), Rect2(Point2(), view_size));

	// Grid

	draw_set_transform_matrix(_world_to_view);

	Vector2 min_edge = get_world_pos(Vector2(0, view_size.y));
	Vector2 max_edge = get_world_pos(Vector2(view_size.x, 0));

	const Color grid_color0 = get_theme_color(SNAME("mono_color"), SNAME("Editor")) * Color(1, 1, 1, 0.15);
	const Color grid_color1 = get_theme_color(SNAME("mono_color"), SNAME("Editor")) * Color(1, 1, 1, 0.07);
	draw_line(Vector2(min_edge.x, curve.get_min_value()), Vector2(max_edge.x, curve.get_min_value()), grid_color0);
	draw_line(Vector2(max_edge.x, curve.get_max_value()), Vector2(min_edge.x, curve.get_max_value()), grid_color0);
	draw_line(Vector2(0, min_edge.y), Vector2(0, max_edge.y), grid_color0);
	draw_line(Vector2(1, max_edge.y), Vector2(1, min_edge.y), grid_color0);

	float curve_height = (curve.get_max_value() - curve.get_min_value());
	const Vector2 grid_step(0.25, 0.5 * curve_height);

	for (real_t x = 0; x < 1.0; x += grid_step.x) {
		draw_line(Vector2(x, min_edge.y), Vector2(x, max_edge.y), grid_color1);
	}
	for (real_t y = curve.get_min_value(); y < curve.get_max_value(); y += grid_step.y) {
		draw_line(Vector2(min_edge.x, y), Vector2(max_edge.x, y), grid_color1);
	}

	// Markings

	draw_set_transform_matrix(Transform2D());

	Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
	int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Label"));
	float font_height = font->get_height(font_size);
	Color text_color = get_theme_color(SNAME("font_color"), SNAME("Editor"));

	{
		// X axis
		float y = curve.get_min_value();
		Vector2 off(0, font_height - 1);
		draw_string(font, get_view_pos(Vector2(0, y)) + off, "0.0", HALIGN_LEFT, -1, font_size, text_color);
		draw_string(font, get_view_pos(Vector2(0.25, y)) + off, "0.25", HALIGN_LEFT, -1, font_size, text_color);
		draw_string(font, get_view_pos(Vector2(0.5, y)) + off, "0.5", HALIGN_LEFT, -1, font_size, text_color);
		draw_string(font, get_view_pos(Vector2(0.75, y)) + off, "0.75", HALIGN_LEFT, -1, font_size, text_color);
		draw_string(font, get_view_pos(Vector2(1, y)) + off, "1.0", HALIGN_LEFT, -1, font_size, text_color);
	}

	{
		// Y axis
		float m0 = curve.get_min_value();
		float m1 = 0.5 * (curve.get_min_value() + curve.get_max_value());
		float m2 = curve.get_max_value();
		Vector2 off(1, -1);
		draw_string(font, get_view_pos(Vector2(0, m0)) + off, String::num(m0, 2), HALIGN_LEFT, -1, font_size, text_color);
		draw_string(font, get_view_pos(Vector2(0, m1)) + off, String::num(m1, 2), HALIGN_LEFT, -1, font_size, text_color);
		draw_string(font, get_view_pos(Vector2(0, m2)) + off, String::num(m2, 3), HALIGN_LEFT, -1, font_size, text_color);
	}

	// Draw tangents for current point

	if (_selected_point >= 0) {
		const Color tangent_color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));

		int i = _selected_point;
		Vector2 pos = curve.get_point_position(i);

		if (i != 0) {
			Vector2 control_pos = get_tangent_view_pos(i, TANGENT_LEFT);
			draw_line(get_view_pos(pos), control_pos, tangent_color, Math::round(EDSCALE));
			draw_rect(Rect2(control_pos, Vector2(1, 1)).grow(Math::round(2 * EDSCALE)), tangent_color);
		}

		if (i != curve.get_point_count() - 1) {
			Vector2 control_pos = get_tangent_view_pos(i, TANGENT_RIGHT);
			draw_line(get_view_pos(pos), control_pos, tangent_color, Math::round(EDSCALE));
			draw_rect(Rect2(control_pos, Vector2(1, 1)).grow(Math::round(2 * EDSCALE)), tangent_color);
		}
	}

	// Draw lines

	draw_set_transform_matrix(_world_to_view);

	const Color line_color = get_theme_color(SNAME("font_color"), SNAME("Editor"));
	const Color edge_line_color = get_theme_color(SNAME("highlight_color"), SNAME("Editor"));

	CanvasItemPlotCurve plot_func(*this, line_color, edge_line_color);
	plot_curve_accurate(curve, 4.f / view_size.x, plot_func);

	// Draw points

	draw_set_transform_matrix(Transform2D());

	const Color point_color = get_theme_color(SNAME("font_color"), SNAME("Editor"));
	const Color selected_point_color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));

	for (int i = 0; i < curve.get_point_count(); ++i) {
		Vector2 pos = curve.get_point_position(i);
		draw_rect(Rect2(get_view_pos(pos), Vector2(1, 1)).grow(Math::round(3 * EDSCALE)), i == _selected_point ? selected_point_color : point_color);
		// TODO Circles are prettier. Needs a fix! Or a texture
		//draw_circle(pos, 2, point_color);
	}

	// Hover

	if (_hover_point != -1) {
		const Color hover_color = line_color;
		Vector2 pos = curve.get_point_position(_hover_point);
		draw_rect(Rect2(get_view_pos(pos), Vector2(1, 1)).grow(Math::round(_hover_radius * EDSCALE)), hover_color, false, Math::round(EDSCALE));
	}

	// Help text

	if (_selected_point > 0 && _selected_point + 1 < curve.get_point_count()) {
		text_color.a *= 0.4;
		draw_string(font, Vector2(50 * EDSCALE, font_height), TTR("Hold Shift to edit tangents individually"), HALIGN_LEFT, -1, font_size, text_color);
	} else if (curve.get_point_count() == 0) {
		text_color.a *= 0.4;
		draw_string(font, Vector2(50 * EDSCALE, font_height), TTR("Right click to add point"), HALIGN_LEFT, -1, font_size, text_color);
	}
}

//---------------

bool EditorInspectorPluginCurve::can_handle(Object *p_object) {
	return Object::cast_to<Curve>(p_object) != nullptr;
}

void EditorInspectorPluginCurve::parse_begin(Object *p_object) {
	Curve *curve = Object::cast_to<Curve>(p_object);
	ERR_FAIL_COND(!curve);
	Ref<Curve> c(curve);

	CurveEditor *editor = memnew(CurveEditor);
	editor->set_curve(curve);
	add_custom_control(editor);
}

CurveEditorPlugin::CurveEditorPlugin(EditorNode *p_node) {
	Ref<EditorInspectorPluginCurve> curve_plugin;
	curve_plugin.instantiate();
	EditorInspector::add_inspector_plugin(curve_plugin);

	get_editor_interface()->get_resource_previewer()->add_preview_generator(memnew(CurvePreviewGenerator));
}

//-----------------------------------
// Preview generator

bool CurvePreviewGenerator::handles(const String &p_type) const {
	return p_type == "Curve";
}

Ref<Texture2D> CurvePreviewGenerator::generate(const Ref<Resource> &p_from, const Size2 &p_size) const {
	Ref<Curve> curve_ref = p_from;
	ERR_FAIL_COND_V_MSG(curve_ref.is_null(), Ref<Texture2D>(), "It's not a reference to a valid Resource object.");
	Curve &curve = **curve_ref;

	// FIXME: Should be ported to use p_size as done in b2633a97
	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;
	Ref<Image> img_ref;
	img_ref.instantiate();
	Image &im = **img_ref;

	im.create(thumbnail_size, thumbnail_size / 2, false, Image::FORMAT_RGBA8);

	Color bg_color(0.1, 0.1, 0.1, 1.0);
	for (int i = 0; i < thumbnail_size; i++) {
		for (int j = 0; j < thumbnail_size / 2; j++) {
			im.set_pixel(i, j, bg_color);
		}
	}

	Color line_color(0.8, 0.8, 0.8, 1.0);
	float range_y = curve.get_max_value() - curve.get_min_value();

	int prev_y = 0;
	for (int x = 0; x < im.get_width(); ++x) {
		float t = static_cast<float>(x) / im.get_width();
		float v = (curve.interpolate_baked(t) - curve.get_min_value()) / range_y;
		int y = CLAMP(im.get_height() - v * im.get_height(), 0, im.get_height());

		// Plot point
		if (y >= 0 && y < im.get_height()) {
			im.set_pixel(x, y, line_color);
		}

		// Plot vertical line to fix discontinuity (not 100% correct but enough for a preview)
		if (x != 0 && Math::abs(y - prev_y) > 1) {
			int y0, y1;
			if (y < prev_y) {
				y0 = y;
				y1 = prev_y;
			} else {
				y0 = prev_y;
				y1 = y;
			}
			for (int ly = y0; ly < y1; ++ly) {
				im.set_pixel(x, ly, line_color);
			}
		}

		prev_y = y;
	}

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));

	ptex->create_from_image(img_ref);
	return ptex;
}

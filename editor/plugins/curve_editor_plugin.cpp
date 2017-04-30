/*************************************************************************/
/*  curve_editor_plugin.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "core_string_names.h"
#include "os/input.h"
#include "os/keyboard.h"

CurveEditor::CurveEditor() {
	_selected_point = -1;
	_hover_point = -1;
	_selected_tangent = TANGENT_NONE;
	_hover_radius = 6;
	_tangents_length = 40;
	_dragging = false;
	_has_undo_data = false;
	_world_rect = Rect2(0, 0, 1, 1);

	set_focus_mode(FOCUS_ALL);
	set_clip_contents(true);

	_context_menu = memnew(PopupMenu);
	_context_menu->connect("id_pressed", this, "_on_context_menu_item_selected");
	add_child(_context_menu);

	_presets_menu = memnew(PopupMenu);
	_presets_menu->set_name("_presets_menu");
	_presets_menu->add_item("Flat0", PRESET_FLAT0);
	_presets_menu->add_item("Flat1", PRESET_FLAT1);
	_presets_menu->add_item("Linear", PRESET_LINEAR);
	_presets_menu->add_item("Ease in", PRESET_EASE_IN);
	_presets_menu->add_item("Ease out", PRESET_EASE_OUT);
	_presets_menu->add_item("Smoothstep", PRESET_SMOOTHSTEP);
	_presets_menu->connect("id_pressed", this, "_on_preset_item_selected");
	_context_menu->add_child(_presets_menu);
}

void CurveEditor::set_curve(Ref<Curve> curve) {

	if (curve == _curve_ref)
		return;

	if (_curve_ref.is_valid()) {
		_curve_ref->disconnect("changed", this, "_curve_changed");
	}
	_curve_ref = curve;
	if (_curve_ref.is_valid()) {
		_curve_ref->connect("changed", this, "_curve_changed");
	}

	_selected_point = -1;
	_hover_point = -1;
	_selected_tangent = TANGENT_NONE;

	update();

	// Note: if you edit a curve, then set another, and try to undo,
	// it will normally apply on the previous curve, but you won't see it
}

Size2 CurveEditor::get_minimum_size() const {
	return Vector2(64, 64);
}

void CurveEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW)
		_draw();
}

void CurveEditor::on_gui_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseButton> mb_ref = p_event;
	if (mb_ref.is_valid()) {

		const InputEventMouseButton &mb = **mb_ref;

		if (mb.is_pressed() && !_dragging) {

			Vector2 mpos = mb.get_position();

			_selected_tangent = get_tangent_at(mpos);
			if (_selected_tangent == TANGENT_NONE)
				set_selected_point(get_point_at(mpos));

			switch (mb.get_button_index()) {
				case BUTTON_RIGHT:
					_context_click_pos = mpos;
					open_context_menu(get_global_transform().xform(mpos));
					break;

				case BUTTON_MIDDLE:
					remove_point(_hover_point);
					break;

				case BUTTON_LEFT:
					_dragging = true;
					break;
			}
		}

		if (!mb.is_pressed() && _dragging && mb.get_button_index() == BUTTON_LEFT) {
			_dragging = false;
			if (_has_undo_data) {
				push_undo(_undo_data);
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

				if (!_has_undo_data) {
					// Save curve state before dragging points
					_undo_data = _curve_ref->get_data();
					_has_undo_data = true;
				}

				if (_selected_tangent == TANGENT_NONE) {
					// Drag point

					Vector2 point_pos = get_world_pos(mpos);

					int i = _curve_ref->set_point_offset(_selected_point, point_pos.x);
					// The index may change if the point is dragged across another one
					set_hover_point_index(i);
					set_selected_point(i);

					// TODO Get rid of this clamp if zoom is implemented in this editor.
					// This is to prevent the user from loosing a point out of view.
					if (point_pos.y < 0.0)
						point_pos.y = 0.0;
					else if (point_pos.y > 1.0)
						point_pos.y = 1.0;

					_curve_ref->set_point_value(_selected_point, point_pos.y);

					//auto_calculate_tangents(i);

				} else {
					// Drag tangent

					Vector2 point_pos = _curve_ref->get_point_pos(_selected_point);
					Vector2 control_pos = get_world_pos(mpos);

					Vector2 dir = (control_pos - point_pos).normalized();

					real_t tangent;
					if (Math::abs(dir.x) > CMP_EPSILON)
						tangent = dir.y / dir.x;
					else
						tangent = 9999 * (dir.y >= 0 ? 1 : -1);

					bool link = !Input::get_singleton()->is_key_pressed(KEY_SHIFT);

					if (_selected_tangent == TANGENT_LEFT) {
						_curve_ref->set_point_left_tangent(_selected_point, tangent);
						if (link && _selected_point != _curve_ref->get_point_count() - 1)
							_curve_ref->set_point_right_tangent(_selected_point, tangent);
					} else {
						_curve_ref->set_point_right_tangent(_selected_point, tangent);
						if (link && _selected_point != 0)
							_curve_ref->set_point_left_tangent(_selected_point, tangent);
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
			if (key.get_scancode() == KEY_DELETE)
				remove_point(_selected_point);
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
			break;

		case PRESET_FLAT1:
			curve.add_point(Vector2(0, 1));
			curve.add_point(Vector2(1, 1));
			break;

		case PRESET_LINEAR:
			curve.add_point(Vector2(0, 0), 0, 1);
			curve.add_point(Vector2(1, 1), 1, 0);
			break;

		case PRESET_EASE_IN:
			curve.add_point(Vector2(0, 0));
			curve.add_point(Vector2(1, 1), 1.4, 0);
			break;

		case PRESET_EASE_OUT:
			curve.add_point(Vector2(0, 0), 0, 1.4);
			curve.add_point(Vector2(1, 1));
			break;

		case PRESET_SMOOTHSTEP:
			curve.add_point(Vector2(0, 0));
			curve.add_point(Vector2(1, 1));
			break;

		default:
			break;
	}

	push_undo(previous_data);
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
	}
}

void CurveEditor::open_context_menu(Vector2 pos) {
	_context_menu->set_position(pos);

	_context_menu->clear();

	if (_curve_ref.is_valid()) {
		_context_menu->add_item(TTR("Add point"), CONTEXT_ADD_POINT);
		if (_selected_point >= 0) {
			_context_menu->add_item(TTR("Remove point"), CONTEXT_REMOVE_POINT);
		}
		_context_menu->add_separator();
	}

	_context_menu->add_submenu_item(TTR("Load preset"), _presets_menu->get_name());

	_context_menu->popup();
}

int CurveEditor::get_point_at(Vector2 pos) const {
	if (_curve_ref.is_null())
		return -1;
	const Curve &curve = **_curve_ref;

	const float r = _hover_radius * _hover_radius;

	for (int i = 0; i < curve.get_point_count(); ++i) {
		Vector2 p = get_view_pos(curve.get_point_pos(i));
		if (p.distance_squared_to(pos) <= r) {
			return i;
		}
	}

	return -1;
}

int CurveEditor::get_tangent_at(Vector2 pos) const {
	if (_curve_ref.is_null() || _selected_point < 0)
		return TANGENT_NONE;

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

	Array prev_data = _curve_ref->get_data();

	Vector2 point_pos = get_world_pos(pos);
	if (point_pos.y < 0.0)
		point_pos.y = 0.0;
	else if (point_pos.y > 1.0)
		point_pos.y = 1.0;

	_curve_ref->add_point(point_pos);

	push_undo(prev_data);
}

void CurveEditor::remove_point(int index) {
	ERR_FAIL_COND(_curve_ref.is_null());

	Array prev_data = _curve_ref->get_data();

	_curve_ref->remove_point(index);

	if (index == _selected_point)
		set_selected_point(-1);

	push_undo(prev_data);
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

void CurveEditor::push_undo(Array previous_curve_data) {
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

	ur->create_action(TTR("Modify Curve"));
	ur->add_do_method(*_curve_ref, "_set_data", _curve_ref->get_data());
	ur->add_undo_method(*_curve_ref, "_set_data", previous_curve_data);

	// This boolean is to prevent commit_action from executing the do method,
	// because at this point it's already done, there is no point in doing it twice
	_curve_ref->_disable_set_data = true;
	ur->commit_action();
	_curve_ref->_disable_set_data = false;
}

void CurveEditor::update_view_transform() {
	Vector2 control_size = get_size();
	const real_t margin = 24;

	_world_rect = Rect2(Curve::MIN_X, 0, Curve::MAX_X, 1);
	Vector2 wm = Vector2(margin, margin) / control_size;
	_world_rect.position -= wm;
	_world_rect.size += 2.0 * wm;

	_world_to_view = Transform2D();
	_world_to_view.translate(-_world_rect.position - Vector2(0, _world_rect.size.y));
	_world_to_view.scale(Vector2(control_size.x, -control_size.y) / _world_rect.size);
}

Vector2 CurveEditor::get_tangent_view_pos(int i, TangentIndex tangent) const {

	Vector2 dir;
	if (tangent == TANGENT_LEFT)
		dir = -Vector2(1, _curve_ref->get_point_left_tangent(i));
	else
		dir = Vector2(1, _curve_ref->get_point_right_tangent(i));

	Vector2 point_pos = get_view_pos(_curve_ref->get_point_pos(i));
	Vector2 control_pos = get_view_pos(_curve_ref->get_point_pos(i) + dir);

	return point_pos + _tangents_length * (control_pos - point_pos).normalized();
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
		Vector2 first_point = curve.get_point_pos(0);
		Vector2 last_point = curve.get_point_pos(curve.get_point_count() - 1);

		// Edge lines
		plot_func(Vector2(0, first_point.y), first_point, false);
		plot_func(Vector2(Curve::MAX_X, last_point.y), last_point, false);

		// Draw section by section, so that we get maximum precision near points.
		// It's an accurate representation, but slower than using the baked one.
		for (int i = 1; i < curve.get_point_count(); ++i) {
			Vector2 a = curve.get_point_pos(i - 1);
			Vector2 b = curve.get_point_pos(i);

			Vector2 pos = a;
			Vector2 prev_pos = a;

			float len = b.x - a.x;
			//float step = 4.f / view_size.x;

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

	CanvasItemPlotCurve(CanvasItem &p_ci, Color p_color1, Color p_color2)
		: ci(p_ci), color1(p_color1), color2(p_color2) {}

	void operator()(Vector2 pos0, Vector2 pos1, bool in_definition) {
		ci.draw_line(pos0, pos1, in_definition ? color1 : color2);
	}
};

void CurveEditor::_draw() {
	if (_curve_ref.is_null())
		return;
	Curve &curve = **_curve_ref;

	update_view_transform();

	// Background

	Vector2 view_size = get_rect().size;
	draw_style_box(get_stylebox("bg", "Tree"), Rect2(Point2(), view_size));

	// Grid

	draw_set_transform_matrix(_world_to_view);

	Vector2 min_edge = get_world_pos(Vector2(0, view_size.y));
	Vector2 max_edge = get_world_pos(Vector2(view_size.x, 0));

	const Color grid_color0(0, 0, 0, 0.5);
	const Color grid_color1(0, 0, 0, 0.15);
	draw_line(Vector2(min_edge.x, 0), Vector2(max_edge.x, 0), grid_color0);
	draw_line(Vector2(0, min_edge.y), Vector2(0, max_edge.y), grid_color0);
	draw_line(Vector2(1, max_edge.y), Vector2(1, min_edge.y), grid_color0);
	draw_line(Vector2(max_edge.x, 1), Vector2(min_edge.x, 1), grid_color0);

	const Vector2 grid_step(0.25, 0.5);

	for (real_t x = 0; x < 1.0; x += grid_step.x) {
		draw_line(Vector2(x, min_edge.y), Vector2(x, max_edge.y), grid_color1);
	}
	for (real_t y = 0; y < 1.0; y += grid_step.y) {
		draw_line(Vector2(min_edge.x, y), Vector2(max_edge.x, y), grid_color1);
	}

	// Markings

	draw_set_transform_matrix(Transform2D());

	Ref<Font> font = get_font("font", "Label");
	const Color text_color(1, 1, 1, 0.3);

	draw_string(font, get_view_pos(Vector2(0, 0)), "0.0", text_color);

	draw_string(font, get_view_pos(Vector2(0.25, 0)), "0.25", text_color);
	draw_string(font, get_view_pos(Vector2(0.5, 0)), "0.5", text_color);
	draw_string(font, get_view_pos(Vector2(0.75, 0)), "0.75", text_color);
	draw_string(font, get_view_pos(Vector2(1, 0)), "1.0", text_color);

	draw_string(font, get_view_pos(Vector2(0, 0.5)), "0.5", text_color);
	draw_string(font, get_view_pos(Vector2(0, 1)), "1.0", text_color);

	// Draw tangents for current point

	if (_selected_point >= 0) {

		const Color tangent_color(0.5, 0.5, 1, 1);

		int i = _selected_point;
		Vector2 pos = curve.get_point_pos(i);

		if (i != 0) {
			Vector2 control_pos = get_tangent_view_pos(i, TANGENT_LEFT);
			draw_line(get_view_pos(pos), control_pos, tangent_color);
			draw_rect(Rect2(control_pos, Vector2(1, 1)).grow(2), tangent_color);
		}

		if (i != curve.get_point_count() - 1) {
			Vector2 control_pos = get_tangent_view_pos(i, TANGENT_RIGHT);
			draw_line(get_view_pos(pos), control_pos, tangent_color);
			draw_rect(Rect2(control_pos, Vector2(1, 1)).grow(2), tangent_color);
		}
	}

	// Draw lines

	draw_set_transform_matrix(_world_to_view);

	const Color line_color(1, 1, 1, 0.85);
	const Color edge_line_color(1, 1, 1, 0.4);

	CanvasItemPlotCurve plot_func(*this, line_color, edge_line_color);
	plot_curve_accurate(curve, 4.f / view_size.x, plot_func);

	/*// TEST draw baked curve
	{
		Vector2 pos = Vector2(0, curve.interpolate_baked(0));
		Vector2 prev_pos = pos;

		float len = 1.0;
		float step = 4.f / view_size.x;

		for(float x = step; x < len; x += step) {
			pos.x = x;
			pos.y = curve.interpolate_baked(x);
			draw_line(get_point_view_pos(prev_pos), get_point_view_pos(pos), Color(0,1,0));
			prev_pos = pos;
		}

		draw_line(get_point_view_pos(prev_pos), get_point_view_pos(Vector2(1, curve.interpolate_baked(1))), Color(0,1,0));
	}//*/

	// Draw points

	draw_set_transform_matrix(Transform2D());

	const Color point_color(1, 1, 1);
	const Color selected_point_color(1, 0.5, 0.5);

	for (int i = 0; i < curve.get_point_count(); ++i) {
		Vector2 pos = curve.get_point_pos(i);
		draw_rect(Rect2(get_view_pos(pos), Vector2(1, 1)).grow(3), i == _selected_point ? selected_point_color : point_color);
		// TODO Circles are prettier. Needs a fix! Or a texture
		//draw_circle(pos, 2, point_color);
	}

	// Hover

	if (_hover_point != -1) {
		const Color hover_color = line_color;
		Vector2 pos = curve.get_point_pos(_hover_point);
		stroke_rect(Rect2(get_view_pos(pos), Vector2(1, 1)).grow(_hover_radius), hover_color);
	}
}

// TODO That should be part of the drawing API...
void CurveEditor::stroke_rect(Rect2 rect, Color color) {

	// a---b
	// |   |
	// c---d
	Vector2 a(rect.position);
	Vector2 b(rect.position.x + rect.size.x, rect.position.y);
	Vector2 c(rect.position.x, rect.position.y + rect.size.y);
	Vector2 d(rect.position + rect.size);

	draw_line(a, b, color);
	draw_line(b, d, color);
	draw_line(d, c, color);
	draw_line(c, a, color);
}

void CurveEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_gui_input"), &CurveEditor::on_gui_input);
	ClassDB::bind_method(D_METHOD("_on_preset_item_selected"), &CurveEditor::on_preset_item_selected);
	ClassDB::bind_method(D_METHOD("_curve_changed"), &CurveEditor::_curve_changed);
	ClassDB::bind_method(D_METHOD("_on_context_menu_item_selected"), &CurveEditor::on_context_menu_item_selected);
}

//---------------

CurveEditorPlugin::CurveEditorPlugin(EditorNode *p_node) {
	_editor_node = p_node;

	_view = memnew(CurveEditor);
	_view->set_custom_minimum_size(Size2(100, 128 * EDSCALE));
	_view->hide();

	_toggle_button = _editor_node->add_bottom_panel_item(get_name(), _view);
	_toggle_button->hide();
}

CurveEditorPlugin::~CurveEditorPlugin() {
}

void CurveEditorPlugin::edit(Object *p_object) {

	Ref<Curve> curve_ref;

	if (_current_ref.is_valid()) {
		CurveTexture *ct = _current_ref->cast_to<CurveTexture>();
		if (ct)
			ct->disconnect(CoreStringNames::get_singleton()->changed, this, "_curve_texture_changed");
	}

	if (p_object) {
		Resource *res = p_object->cast_to<Resource>();
		ERR_FAIL_COND(res == NULL);
		ERR_FAIL_COND(!handles(p_object));

		_current_ref = Ref<Resource>(p_object->cast_to<Resource>());

		if (_current_ref.is_valid()) {
			Curve *curve = _current_ref->cast_to<Curve>();
			if (curve)
				curve_ref = Ref<Curve>(curve);
			else {
				CurveTexture *ct = _current_ref->cast_to<CurveTexture>();
				if (ct) {
					ct->connect(CoreStringNames::get_singleton()->changed, this, "_curve_texture_changed");
					curve_ref = ct->get_curve();
				}
			}
		}

	} else {
		_current_ref = Ref<Resource>();
	}

	_view->set_curve(curve_ref);
}

bool CurveEditorPlugin::handles(Object *p_object) const {
	// Both handled so that we can keep the curve editor open
	return p_object->cast_to<Curve>() || p_object->cast_to<CurveTexture>();
}

void CurveEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		_toggle_button->show();
		_editor_node->make_bottom_panel_item_visible(_view);
	} else {
		_toggle_button->hide();
		if (_view->is_visible_in_tree())
			_editor_node->hide_bottom_panel();
	}
}

void CurveEditorPlugin::_curve_texture_changed() {
	// If the curve is shown indirectly as a CurveTexture is edited,
	// we need to monitor when the curve property gets assigned
	CurveTexture *ct = _current_ref->cast_to<CurveTexture>();
	if (ct) {
		_view->set_curve(ct->get_curve());
	}
}

void CurveEditorPlugin::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_curve_texture_changed"), &CurveEditorPlugin::_curve_texture_changed);
}

/**************************************************************************/
/*  gradient_editor.cpp                                                   */
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

#include "gradient_editor.h"

#include "core/os/keyboard.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/resources/gradient_texture.h"

void GradientEditor::set_gradient(const Ref<Gradient> &p_gradient) {
	gradient = p_gradient;
	connect("ramp_changed", callable_mp(this, &GradientEditor::_ramp_changed));
	gradient->connect_changed(callable_mp(this, &GradientEditor::_gradient_changed));
	set_points(gradient->get_points());
	set_interpolation_mode(gradient->get_interpolation_mode());
	set_interpolation_color_space(gradient->get_interpolation_color_space());
}

void GradientEditor::reverse_gradient() {
	gradient->reverse();
	set_points(gradient->get_points());
	emit_signal(SNAME("ramp_changed"));
	queue_redraw();
}

int GradientEditor::_get_point_from_pos(int x) {
	int result = -1;
	int total_w = get_size().width - get_size().height - draw_spacing - handle_width;
	float min_distance = 1e20;
	for (int i = 0; i < points.size(); i++) {
		// Check if we clicked at point.
		float distance = ABS(x - points[i].offset * total_w);
		float min = handle_width * 0.85; // Allow the mouse to be more than half a handle width away for ease of grabbing.
		if (distance <= min && distance < min_distance) {
			result = i;
			min_distance = distance;
		}
	}
	return result;
}

void GradientEditor::_show_color_picker() {
	if (grabbed == -1) {
		return;
	}
	picker->set_pick_color(points[grabbed].color);
	Size2 minsize = popup->get_contents_minimum_size();
	bool show_above = false;
	if (get_global_position().y + get_size().y + minsize.y > get_viewport_rect().size.y) {
		show_above = true;
	}
	if (show_above) {
		popup->set_position(get_screen_position() - Vector2(0, minsize.y));
	} else {
		popup->set_position(get_screen_position() + Vector2(0, get_size().y));
	}
	popup->popup();
}

void GradientEditor::_gradient_changed() {
	if (editing) {
		return;
	}

	editing = true;
	Vector<Gradient::Point> grad_points = gradient->get_points();
	set_points(grad_points);
	set_interpolation_mode(gradient->get_interpolation_mode());
	set_interpolation_color_space(gradient->get_interpolation_color_space());
	queue_redraw();
	editing = false;
}

void GradientEditor::_ramp_changed() {
	editing = true;
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Gradient Edited"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(gradient.ptr(), "set_offsets", get_offsets());
	undo_redo->add_do_method(gradient.ptr(), "set_colors", get_colors());
	undo_redo->add_do_method(gradient.ptr(), "set_interpolation_mode", get_interpolation_mode());
	undo_redo->add_do_method(gradient.ptr(), "set_interpolation_color_space", get_interpolation_color_space());
	undo_redo->add_undo_method(gradient.ptr(), "set_offsets", gradient->get_offsets());
	undo_redo->add_undo_method(gradient.ptr(), "set_colors", gradient->get_colors());
	undo_redo->add_undo_method(gradient.ptr(), "set_interpolation_mode", gradient->get_interpolation_mode());
	undo_redo->add_undo_method(gradient.ptr(), "set_interpolation_color_space", gradient->get_interpolation_color_space());
	undo_redo->commit_action();
	editing = false;
}

void GradientEditor::_color_changed(const Color &p_color) {
	if (grabbed == -1) {
		return;
	}
	points.write[grabbed].color = p_color;
	queue_redraw();
	emit_signal(SNAME("ramp_changed"));
}

void GradientEditor::set_ramp(const Vector<float> &p_offsets, const Vector<Color> &p_colors) {
	ERR_FAIL_COND(p_offsets.size() != p_colors.size());
	points.clear();
	for (int i = 0; i < p_offsets.size(); i++) {
		Gradient::Point p;
		p.offset = p_offsets[i];
		p.color = p_colors[i];
		points.push_back(p);
	}

	points.sort();
	queue_redraw();
}

Vector<float> GradientEditor::get_offsets() const {
	Vector<float> ret;
	for (int i = 0; i < points.size(); i++) {
		ret.push_back(points[i].offset);
	}
	return ret;
}

Vector<Color> GradientEditor::get_colors() const {
	Vector<Color> ret;
	for (int i = 0; i < points.size(); i++) {
		ret.push_back(points[i].color);
	}
	return ret;
}

void GradientEditor::set_points(Vector<Gradient::Point> &p_points) {
	if (points.size() != p_points.size()) {
		grabbed = -1;
	}
	points.clear();
	points = p_points;
	points.sort();
}

Vector<Gradient::Point> &GradientEditor::get_points() {
	return points;
}

void GradientEditor::set_interpolation_mode(Gradient::InterpolationMode p_interp_mode) {
	interpolation_mode = p_interp_mode;
}

Gradient::InterpolationMode GradientEditor::get_interpolation_mode() {
	return interpolation_mode;
}

void GradientEditor::set_interpolation_color_space(Gradient::ColorSpace p_color_space) {
	interpolation_color_space = p_color_space;
}

Gradient::ColorSpace GradientEditor::get_interpolation_color_space() {
	return interpolation_color_space;
}

ColorPicker *GradientEditor::get_picker() {
	return picker;
}

PopupPanel *GradientEditor::get_popup() {
	return popup;
}

Size2 GradientEditor::get_minimum_size() const {
	return Size2(0, 60) * EDSCALE;
}

void GradientEditor::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->is_pressed() && k->get_keycode() == Key::KEY_DELETE && grabbed != -1) {
		points.remove_at(grabbed);
		grabbed = -1;
		grabbing = false;
		queue_redraw();
		emit_signal(SNAME("ramp_changed"));
		accept_event();
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed()) {
		float adjusted_mb_x = mb->get_position().x - handle_width / 2;

		// Delete point on right click.
		if (mb->get_button_index() == MouseButton::RIGHT) {
			grabbed = _get_point_from_pos(adjusted_mb_x);
			if (grabbed != -1) {
				points.remove_at(grabbed);
				grabbed = -1;
				grabbing = false;
				queue_redraw();
				emit_signal(SNAME("ramp_changed"));
				accept_event();
			}
		}

		// Hold Alt key to duplicate selected color.
		if (mb->get_button_index() == MouseButton::LEFT && mb->is_alt_pressed()) {
			grabbed = _get_point_from_pos(adjusted_mb_x);

			if (grabbed != -1) {
				int total_w = get_size().width - get_size().height - draw_spacing - handle_width;
				Gradient::Point new_point = points[grabbed];
				new_point.offset = CLAMP(adjusted_mb_x / float(total_w), 0, 1);
				points.push_back(new_point);
				points.sort();
				for (int i = 0; i < points.size(); ++i) {
					if (points[i].offset == new_point.offset) {
						grabbed = i;
						break;
					}
				}

				emit_signal(SNAME("ramp_changed"));
				queue_redraw();
			}
		}

		// Select.
		if (mb->get_button_index() == MouseButton::LEFT) {
			queue_redraw();
			int total_w = get_size().width - get_size().height - draw_spacing - handle_width;

			// Check if color selector was clicked or ramp was double-clicked.
			if (adjusted_mb_x > total_w + draw_spacing) {
				if (!mb->is_double_click()) {
					_show_color_picker();
				}
				return;
			} else if (mb->is_double_click()) {
				grabbed = _get_point_from_pos(adjusted_mb_x);
				_show_color_picker();
				accept_event();
				return;
			}

			grabbing = true;
			grabbed = _get_point_from_pos(adjusted_mb_x);

			// Grab or select.
			if (grabbed != -1) {
				return;
			}

			// Insert point.
			Gradient::Point new_point;
			new_point.offset = CLAMP(adjusted_mb_x / float(total_w), 0, 1);
			new_point.color = gradient->get_color_at_offset(new_point.offset);

			points.push_back(new_point);
			points.sort();
			for (int i = 0; i < points.size(); i++) {
				if (points[i].offset == new_point.offset) {
					grabbed = i;
					break;
				}
			}

			emit_signal(SNAME("ramp_changed"));
		}
	}

	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && !mb->is_pressed()) {
		if (grabbing) {
			grabbing = false;
			emit_signal(SNAME("ramp_changed"));
		}
		queue_redraw();
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid() && grabbing) {
		float adjusted_mm_x = mm->get_position().x - handle_width / 2;
		int total_w = get_size().width - get_size().height - draw_spacing - handle_width;
		float newofs = CLAMP(adjusted_mm_x / float(total_w), 0, 1);

		// Snap to "round" coordinates if holding Ctrl.
		// Be more precise if holding Shift as well.
		if (mm->is_command_or_control_pressed()) {
			newofs = Math::snapped(newofs, mm->is_shift_pressed() ? 0.025 : 0.1);
		} else if (mm->is_shift_pressed()) {
			// Snap to nearest point if holding just Shift.
			const float snap_threshold = 0.03;
			float smallest_ofs = snap_threshold;
			bool found = false;
			int nearest_point = 0;
			for (int i = 0; i < points.size(); ++i) {
				if (i != grabbed) {
					float temp_ofs = ABS(points[i].offset - newofs);
					if (temp_ofs < smallest_ofs) {
						smallest_ofs = temp_ofs;
						nearest_point = i;
						if (found) {
							break;
						}
						found = true;
					}
				}
			}
			if (found) {
				if (points[nearest_point].offset < newofs) {
					newofs = points[nearest_point].offset + 0.00001;
				} else {
					newofs = points[nearest_point].offset - 0.00001;
				}
				newofs = CLAMP(newofs, 0, 1);
			}
		}

		bool valid = true;
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset == newofs && i != grabbed) {
				valid = false;
				break;
			}
		}

		if (!valid || grabbed == -1) {
			return;
		}
		points.write[grabbed].offset = newofs;

		points.sort();
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset == newofs) {
				grabbed = i;
				break;
			}
		}

		emit_signal(SNAME("ramp_changed"));

		queue_redraw();
	}
}

void GradientEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (!picker->is_connected("color_changed", callable_mp(this, &GradientEditor::_color_changed))) {
				picker->connect("color_changed", callable_mp(this, &GradientEditor::_color_changed));
			}
			[[fallthrough]];
		}
		case NOTIFICATION_THEME_CHANGED: {
			draw_spacing = BASE_SPACING * get_theme_default_base_scale();
			handle_width = BASE_HANDLE_WIDTH * get_theme_default_base_scale();
		} break;

		case NOTIFICATION_DRAW: {
			int w = get_size().x;
			int h = get_size().y;

			if (w == 0 || h == 0) {
				return; // Safety check. We have division by 'h'. And in any case there is nothing to draw with such size.
			}

			int total_w = get_size().width - get_size().height - draw_spacing - handle_width;

			// Draw checker pattern for ramp.
			draw_texture_rect(get_editor_theme_icon(SNAME("GuiMiniCheckerboard")), Rect2(handle_width / 2, 0, total_w, h), true);

			// Draw color ramp.
			gradient_cache->set_points(points);
			gradient_cache->set_interpolation_mode(interpolation_mode);
			gradient_cache->set_interpolation_color_space(interpolation_color_space);
			preview_texture->set_gradient(gradient_cache);
			draw_texture_rect(preview_texture, Rect2(handle_width / 2, 0, total_w, h));

			// Draw borders around color ramp if in focus.
			if (has_focus()) {
				draw_rect(Rect2(handle_width / 2, 0, total_w, h), Color(1, 1, 1, 0.9), false, 1);
			}

			// Draw point markers.
			for (int i = 0; i < points.size(); i++) {
				Color col = points[i].color.get_v() > 0.5 ? Color(0, 0, 0) : Color(1, 1, 1);
				col.a = 0.9;

				draw_line(Vector2(points[i].offset * total_w + handle_width / 2, 0), Vector2(points[i].offset * total_w + handle_width / 2, h / 2), col);
				Rect2 rect = Rect2(points[i].offset * total_w, h / 2, handle_width, h / 2);
				draw_rect(rect, points[i].color, true);
				draw_rect(rect, col, false, 1);
				if (grabbed == i) {
					const Color focus_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
					rect = rect.grow(-1);
					if (has_focus()) {
						draw_rect(rect, focus_color, false, 1);
					} else {
						draw_rect(rect, focus_color.darkened(0.4), false, 1);
					}

					rect = rect.grow(-1);
					draw_rect(rect, col, false, 1);
				}
			}

			// Draw "button" for color selector.
			int button_offset = total_w + handle_width + draw_spacing;
			draw_texture_rect(get_editor_theme_icon(SNAME("GuiMiniCheckerboard")), Rect2(button_offset, 0, h, h), true);
			if (grabbed != -1) {
				// Draw with selection color.
				draw_rect(Rect2(button_offset, 0, h, h), points[grabbed].color);
			} else {
				// If no color selected draw gray color with 'X' on top.
				draw_rect(Rect2(button_offset, 0, h, h), Color(0.5, 0.5, 0.5, 1));
				draw_line(Vector2(button_offset, 0), Vector2(button_offset + h, h), Color(1, 1, 1, 0.6));
				draw_line(Vector2(button_offset, h), Vector2(button_offset + h, 0), Color(1, 1, 1, 0.6));
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				grabbing = false;
			}
		} break;
	}
}

void GradientEditor::_bind_methods() {
	ADD_SIGNAL(MethodInfo("ramp_changed"));
}

GradientEditor::GradientEditor() {
	set_focus_mode(FOCUS_ALL);

	popup = memnew(PopupPanel);
	picker = memnew(ColorPicker);
	popup->add_child(picker);
	popup->connect("about_to_popup", callable_mp(EditorNode::get_singleton(), &EditorNode::setup_color_picker).bind(GradientEditor::get_picker()));

	gradient_cache.instantiate();
	preview_texture.instantiate();

	preview_texture->set_width(1024);
	add_child(popup, false, INTERNAL_MODE_FRONT);
}

GradientEditor::~GradientEditor() {
}

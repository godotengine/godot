/*************************************************************************/
/*  editor_spin_slider.cpp                                               */
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

#include "editor_spin_slider.h"
#include "core/math/expression.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "editor_node.h"
#include "editor_scale.h"

String EditorSpinSlider::get_tooltip(const Point2 &p_pos) const {
	if (grabber->is_visible()) {
#ifdef OSX_ENABLED
		const int key = KEY_META;
#else
		const int key = KEY_CONTROL;
#endif
		return rtos(get_value()) + "\n\n" + vformat(TTR("Hold %s to round to integers. Hold Shift for more precise changes."), find_keycode_name(key));
	}
	return rtos(get_value());
}

String EditorSpinSlider::get_text_value() const {
	return String::num(get_value(), Math::range_step_decimals(get_step()));
}

void EditorSpinSlider::_gui_input(const Ref<InputEvent> &p_event) {
	if (read_only) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->get_button_index() == BUTTON_LEFT) {
			if (mb->is_pressed()) {
				if (updown_offset != -1 && mb->get_position().x > updown_offset) {
					//there is an updown, so use it.
					if (mb->get_position().y < get_size().height / 2) {
						set_value(get_value() + get_step());
					} else {
						set_value(get_value() - get_step());
					}
					return;
				} else {
					grabbing_spinner_attempt = true;
					grabbing_spinner_dist_cache = 0;
					pre_grab_value = get_value();
					grabbing_spinner = false;
					grabbing_spinner_mouse_pos = Input::get_singleton()->get_mouse_position();
				}
			} else {
				if (grabbing_spinner_attempt) {
					if (grabbing_spinner) {
						Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
						Input::get_singleton()->warp_mouse_position(grabbing_spinner_mouse_pos);
						update();
					} else {
						_focus_entered();
					}

					grabbing_spinner = false;
					grabbing_spinner_attempt = false;
				}
			}
		} else if (mb->get_button_index() == BUTTON_WHEEL_UP || mb->get_button_index() == BUTTON_WHEEL_DOWN) {
			if (grabber->is_visible()) {
				call_deferred("update");
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		if (grabbing_spinner_attempt) {
			double diff_x = mm->get_relative().x;
			if (mm->get_shift() && grabbing_spinner) {
				diff_x *= 0.1;
			}
			grabbing_spinner_dist_cache += diff_x;

			if (!grabbing_spinner && ABS(grabbing_spinner_dist_cache) > 4 * EDSCALE) {
				Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
				grabbing_spinner = true;
			}

			if (grabbing_spinner) {
				// Don't make the user scroll all the way back to 'in range' if they went off the end.
				if (pre_grab_value < get_min() && !is_lesser_allowed()) {
					pre_grab_value = get_min();
				}
				if (pre_grab_value > get_max() && !is_greater_allowed()) {
					pre_grab_value = get_max();
				}

				if (mm->get_command()) {
					// If control was just pressed, don't make the value do a huge jump in magnitude.
					if (grabbing_spinner_dist_cache != 0) {
						pre_grab_value += grabbing_spinner_dist_cache * get_step();
						grabbing_spinner_dist_cache = 0;
					}

					set_value(Math::round(pre_grab_value + get_step() * grabbing_spinner_dist_cache * 10));
				} else {
					set_value(pre_grab_value + get_step() * grabbing_spinner_dist_cache);
				}
			}
		} else if (updown_offset != -1) {
			bool new_hover = (mm->get_position().x > updown_offset);
			if (new_hover != hover_updown) {
				hover_updown = new_hover;
				update();
			}
		}
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed() && k->is_action("ui_accept")) {
		_focus_entered();
	}
}

void EditorSpinSlider::_grabber_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;

	if (grabbing_grabber) {
		if (mb.is_valid()) {
			if (mb->get_button_index() == BUTTON_WHEEL_UP) {
				set_value(get_value() + get_step());
				mousewheel_over_grabber = true;
			} else if (mb->get_button_index() == BUTTON_WHEEL_DOWN) {
				set_value(get_value() - get_step());
				mousewheel_over_grabber = true;
			}
		}
	}

	if (mb.is_valid() && mb->get_button_index() == BUTTON_LEFT) {
		if (mb->is_pressed()) {
			grabbing_grabber = true;
			if (!mousewheel_over_grabber) {
				grabbing_ratio = get_as_ratio();
				grabbing_from = grabber->get_transform().xform(mb->get_position()).x;
			}
		} else {
			grabbing_grabber = false;
			mousewheel_over_grabber = false;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && grabbing_grabber) {
		if (mousewheel_over_grabber) {
			return;
		}

		float scale_x = get_global_transform_with_canvas().get_scale().x;
		ERR_FAIL_COND(Math::is_zero_approx(scale_x));
		float grabbing_ofs = (grabber->get_transform().xform(mm->get_position()).x - grabbing_from) / float(grabber_range) / scale_x;
		set_as_ratio(grabbing_ratio + grabbing_ofs);
		update();
	}
}

void EditorSpinSlider::_value_input_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed()) {
		double step = get_step();
		double real_step = step;
		if (step < 1) {
			double divisor = 1.0 / get_step();

			if (trunc(divisor) == divisor) {
				step = 1.0;
			}
		}

#ifdef APPLE_STYLE_KEYS
		if (k->get_command()) {
#else
		if (k->get_control()) {
#endif
			step *= 100.0;
		} else if (k->get_shift()) {
			step *= 10.0;
#ifdef APPLE_STYLE_KEYS
		} else if (k->get_metakey()) {
#else
		} else if (k->get_alt()) {
#endif
			step *= 0.1;
		}

		uint32_t code = k->get_scancode();
		switch (code) {
			case KEY_UP: {
				_evaluate_input_text();

				double last_value = get_value();
				set_value(last_value + step);
				double new_value = get_value();

				if (new_value < CLAMP(last_value + step, get_min(), get_max())) {
					set_value(last_value + real_step);
				}

				value_input_dirty = true;
				set_process_internal(true);
			} break;
			case KEY_DOWN: {
				_evaluate_input_text();

				double last_value = get_value();
				set_value(last_value - step);
				double new_value = get_value();

				if (new_value > CLAMP(last_value - step, get_min(), get_max())) {
					set_value(last_value - real_step);
				}

				value_input_dirty = true;
				set_process_internal(true);
			} break;
		}
	}
}

void EditorSpinSlider::_draw_spin_slider() {
	updown_offset = -1;

	Ref<StyleBox> sb = get_stylebox("normal", "LineEdit");
	if (!flat) {
		draw_style_box(sb, Rect2(Vector2(), get_size()));
	}
	Ref<Font> font = get_font("font", "LineEdit");
	int sep_base = 4 * EDSCALE;
	int sep = sep_base + sb->get_offset().x; //make it have the same margin on both sides, looks better

	int string_width = font->get_string_size(label).width;
	int number_width = get_size().width - sb->get_minimum_size().width - string_width - sep;

	Ref<Texture> updown = get_icon("updown", "SpinBox");

	if (get_step() == 1) {
		number_width -= updown->get_width();
	}

	String numstr = get_text_value();

	int vofs = (get_size().height - font->get_height()) / 2 + font->get_ascent();

	Color fc = get_color("font_color", "LineEdit");
	Color lc;
	if (use_custom_label_color) {
		lc = custom_label_color;
	} else {
		lc = fc;
	}

	if (flat && label != String()) {
		Color label_bg_color = get_color("dark_color_3", "Editor");
		draw_rect(Rect2(Vector2(), Vector2(sb->get_offset().x * 2 + string_width, get_size().height)), label_bg_color);
	}

	if (has_focus()) {
		Ref<StyleBox> focus = get_stylebox("focus", "LineEdit");
		draw_style_box(focus, Rect2(Vector2(), get_size()));
	}

	draw_string(font, Vector2(Math::round(sb->get_offset().x), vofs), label, lc * Color(1, 1, 1, 0.5));

	draw_string(font, Vector2(Math::round(sb->get_offset().x + string_width + sep), vofs), numstr, fc, number_width);

	if (get_step() == 1) {
		Ref<Texture> updown2 = get_icon("updown", "SpinBox");
		int updown_vofs = (get_size().height - updown2->get_height()) / 2;
		updown_offset = get_size().width - sb->get_margin(MARGIN_RIGHT) - updown2->get_width();
		Color c(1, 1, 1);
		if (hover_updown) {
			c *= Color(1.2, 1.2, 1.2);
		}
		draw_texture(updown2, Vector2(updown_offset, updown_vofs), c);
		if (grabber->is_visible()) {
			grabber->hide();
		}
	} else if (!hide_slider) {
		const int grabber_w = 4 * EDSCALE;
		const int width = get_size().width - sb->get_minimum_size().width - grabber_w;
		const int ofs = sb->get_offset().x;
		const int svofs = (get_size().height + vofs) / 2 - 1;
		Color c = fc;

		// Draw the horizontal slider's background.
		c.a = 0.2;
		draw_rect(Rect2(ofs, svofs + 1, width, 2 * EDSCALE), c);

		// Draw the horizontal slider's filled part on the left.
		const int gofs = get_as_ratio() * width;
		c.a = 0.45;
		draw_rect(Rect2(ofs, svofs + 1, gofs, 2 * EDSCALE), c);

		// Draw the horizontal slider's grabber.
		c.a = 0.9;
		const Rect2 grabber_rect = Rect2(ofs + gofs, svofs + 1, grabber_w, 2 * EDSCALE);
		draw_rect(grabber_rect, c);

		bool display_grabber = (mouse_over_spin || mouse_over_grabber) && !grabbing_spinner && !value_input->is_visible();
		if (grabber->is_visible() != display_grabber) {
			if (display_grabber) {
				grabber->show();
			} else {
				grabber->hide();
			}
		}

		if (display_grabber) {
			Ref<Texture> grabber_tex;
			if (mouse_over_grabber) {
				grabber_tex = get_icon("grabber_highlight", "HSlider");
			} else {
				grabber_tex = get_icon("grabber", "HSlider");
			}

			if (grabber->get_texture() != grabber_tex) {
				grabber->set_texture(grabber_tex);
			}

			Vector2 scale = get_global_transform_with_canvas().get_scale();
			grabber->set_scale(scale);
			grabber->set_size(Size2(0, 0));
			grabber->set_position(get_global_position() + (grabber_rect.position + grabber_rect.size * 0.5 - grabber->get_size() * 0.5) * scale);

			if (mousewheel_over_grabber) {
				Input::get_singleton()->warp_mouse_position(grabber->get_position() + grabber_rect.size);
			}

			grabber_range = width;
		}
	}
}

void EditorSpinSlider::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			// Add a left margin to the stylebox to make the number align with the Label
			// when it's edited. The LineEdit "focus" stylebox uses the "normal" stylebox's
			// default margins.
			Ref<StyleBox> stylebox = get_stylebox("normal", "LineEdit")->duplicate();
			// EditorSpinSliders with a label have more space on the left, so add an
			// higher margin to match the location where the text begins.
			// The margin values below were determined by empirical testing.
			stylebox->set_default_margin(MARGIN_LEFT, (get_label() != String() ? 23 : 16) * EDSCALE);
			value_input->add_style_override("normal", stylebox);
		} break;

		case NOTIFICATION_INTERNAL_PROCESS:
			if (value_input_dirty) {
				value_input_dirty = false;
				value_input->set_text(get_text_value());
			}
			set_process_internal(false);
			break;

		case NOTIFICATION_DRAW:
			_draw_spin_slider();
			break;

		case MainLoop::NOTIFICATION_WM_FOCUS_IN:
		case MainLoop::NOTIFICATION_WM_FOCUS_OUT:
		case NOTIFICATION_EXIT_TREE:
			if (grabbing_spinner) {
				grabber->hide();
				Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
				grabbing_spinner = false;
				grabbing_spinner_attempt = false;
			}
			break;

		case NOTIFICATION_MOUSE_ENTER:
			mouse_over_spin = true;
			update();
			break;
		case NOTIFICATION_MOUSE_EXIT:
			mouse_over_spin = false;
			update();
			break;
		case NOTIFICATION_FOCUS_ENTER:
			/* Sorry, I don't like this, it makes navigating the different fields with arrows more difficult.
			* Just press enter to edit.
			* if (Input::get_singleton()->is_mouse_button_pressed(BUTTON_LEFT) && !value_input_just_closed) {
				_focus_entered();
			}*/
			if ((Input::get_singleton()->is_action_pressed("ui_focus_next") || Input::get_singleton()->is_action_pressed("ui_focus_prev")) && !value_input_just_closed) {
				_focus_entered();
			}
			value_input_just_closed = false;
			break;
	}
}

Size2 EditorSpinSlider::get_minimum_size() const {
	Ref<StyleBox> sb = get_stylebox("normal", "LineEdit");
	Ref<Font> font = get_font("font", "LineEdit");

	Size2 ms = sb->get_minimum_size();
	ms.height += font->get_height();

	return ms;
}

void EditorSpinSlider::set_hide_slider(bool p_hide) {
	hide_slider = p_hide;
	update();
}

bool EditorSpinSlider::is_hiding_slider() const {
	return hide_slider;
}

void EditorSpinSlider::set_label(const String &p_label) {
	label = p_label;
	update();
}

String EditorSpinSlider::get_label() const {
	return label;
}

void EditorSpinSlider::_evaluate_input_text() {
	// Replace comma with dot to support it as decimal separator (GH-6028).
	// This prevents using functions like `pow()`, but using functions
	// in EditorSpinSlider is a barely known (and barely used) feature.
	// Instead, we'd rather support German/French keyboard layouts out of the box.
	const String text = value_input->get_text().replace(",", ".");

	Ref<Expression> expr;
	expr.instance();
	Error err = expr->parse(text);
	if (err != OK) {
		return;
	}

	Variant v = expr->execute(Array(), nullptr, false);
	if (v.get_type() == Variant::NIL) {
		return;
	}
	set_value(v);
}

//text_entered signal
void EditorSpinSlider::_value_input_entered(const String &p_text) {
	value_input_just_closed = true;
	value_input->hide();
}

//modal_closed signal
void EditorSpinSlider::_value_input_closed() {
	_evaluate_input_text();
	value_input_just_closed = true;
}

//focus_exited signal
void EditorSpinSlider::_value_focus_exited() {
	// discontinue because the focus_exit was caused by right-click context menu
	if (value_input->get_menu()->is_visible()) {
		return;
	}

	_evaluate_input_text();
	// focus is not on the same element after the vlalue_input was exited
	// -> focus is on next element
	// -> TAB was pressed
	// -> modal_close was not called
	// -> need to close/hide manually
	if (!value_input_just_closed) { //value_input_just_closed should do the same
		value_input->hide();
		//tab was pressed
	} else {
		//enter, click, esc
	}
}

void EditorSpinSlider::_grabber_mouse_entered() {
	mouse_over_grabber = true;
	update();
}

void EditorSpinSlider::_grabber_mouse_exited() {
	mouse_over_grabber = false;
	update();
}

void EditorSpinSlider::set_read_only(bool p_enable) {
	read_only = p_enable;
	update();
}

bool EditorSpinSlider::is_read_only() const {
	return read_only;
}

void EditorSpinSlider::set_flat(bool p_enable) {
	flat = p_enable;
	update();
}

bool EditorSpinSlider::is_flat() const {
	return flat;
}

void EditorSpinSlider::set_custom_label_color(bool p_use_custom_label_color, Color p_custom_label_color) {
	use_custom_label_color = p_use_custom_label_color;
	custom_label_color = p_custom_label_color;
}

void EditorSpinSlider::_focus_entered() {
	Rect2 gr = get_global_rect();
	value_input->set_text(get_text_value());
	value_input->set_position(gr.position);
	value_input->set_size(gr.size);
	value_input->show_modal();
	value_input->select_all();
	value_input->call_deferred("grab_focus"); // deferred to avoid losing focus
	value_input->set_focus_next(find_next_valid_focus()->get_path());
	value_input->set_focus_previous(find_prev_valid_focus()->get_path());
}

void EditorSpinSlider::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_label", "label"), &EditorSpinSlider::set_label);
	ClassDB::bind_method(D_METHOD("get_label"), &EditorSpinSlider::get_label);

	ClassDB::bind_method(D_METHOD("set_read_only", "read_only"), &EditorSpinSlider::set_read_only);
	ClassDB::bind_method(D_METHOD("is_read_only"), &EditorSpinSlider::is_read_only);

	ClassDB::bind_method(D_METHOD("set_flat", "flat"), &EditorSpinSlider::set_flat);
	ClassDB::bind_method(D_METHOD("is_flat"), &EditorSpinSlider::is_flat);

	ClassDB::bind_method(D_METHOD("_gui_input"), &EditorSpinSlider::_gui_input);
	ClassDB::bind_method(D_METHOD("_value_input_gui_input", "event"), &EditorSpinSlider::_value_input_gui_input);
	ClassDB::bind_method(D_METHOD("_grabber_mouse_entered"), &EditorSpinSlider::_grabber_mouse_entered);
	ClassDB::bind_method(D_METHOD("_grabber_mouse_exited"), &EditorSpinSlider::_grabber_mouse_exited);
	ClassDB::bind_method(D_METHOD("_grabber_gui_input"), &EditorSpinSlider::_grabber_gui_input);
	ClassDB::bind_method(D_METHOD("_value_input_closed"), &EditorSpinSlider::_value_input_closed);
	ClassDB::bind_method(D_METHOD("_value_input_entered"), &EditorSpinSlider::_value_input_entered);
	ClassDB::bind_method(D_METHOD("_value_focus_exited"), &EditorSpinSlider::_value_focus_exited);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "label"), "set_label", "get_label");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "read_only"), "set_read_only", "is_read_only");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");
}

EditorSpinSlider::EditorSpinSlider() {
	flat = false;
	grabbing_spinner_attempt = false;
	grabbing_spinner = false;
	grabbing_spinner_dist_cache = 0;
	pre_grab_value = 0;
	set_focus_mode(FOCUS_ALL);
	updown_offset = -1;
	hover_updown = false;
	grabber = memnew(TextureRect);
	add_child(grabber);
	grabber->hide();
	grabber->set_as_toplevel(true);
	grabber->set_mouse_filter(MOUSE_FILTER_STOP);
	grabber->connect("mouse_entered", this, "_grabber_mouse_entered");
	grabber->connect("mouse_exited", this, "_grabber_mouse_exited");
	grabber->connect("gui_input", this, "_grabber_gui_input");
	mouse_over_spin = false;
	mouse_over_grabber = false;
	mousewheel_over_grabber = false;
	grabbing_grabber = false;
	grabber_range = 1;
	value_input = memnew(LineEdit);
	add_child(value_input);
	value_input->set_as_toplevel(true);
	value_input->hide();
	value_input->connect("modal_closed", this, "_value_input_closed");
	value_input->connect("text_entered", this, "_value_input_entered");
	value_input->connect("focus_exited", this, "_value_focus_exited");
	value_input->connect("gui_input", this, "_value_input_gui_input");
	value_input_just_closed = false;
	value_input_dirty = false;
	hide_slider = false;
	read_only = false;
	use_custom_label_color = false;
}

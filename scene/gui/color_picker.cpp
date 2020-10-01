/*************************************************************************/
/*  color_picker.cpp                                                     */
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

#include "color_picker.h"

#include "core/input/input.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#endif
#include "scene/main/window.h"

void ColorPicker::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			btn_pick->set_icon(get_theme_icon("screen_picker", "ColorPicker"));
			bt_add_preset->set_icon(get_theme_icon("add_preset"));

			_update_controls();
		} break;
		case NOTIFICATION_ENTER_TREE: {
			btn_pick->set_icon(get_theme_icon("screen_picker", "ColorPicker"));
			bt_add_preset->set_icon(get_theme_icon("add_preset"));

			_update_color();

#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				PackedColorArray saved_presets = EditorSettings::get_singleton()->get_project_metadata("color_picker", "presets", PackedColorArray());

				for (int i = 0; i < saved_presets.size(); i++) {
					add_preset(saved_presets[i]);
				}
			}
#endif
		} break;
		case NOTIFICATION_PARENTED: {
			for (int i = 0; i < 4; i++) {
				set_margin((Margin)i, get_margin((Margin)i) + get_theme_constant("margin"));
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			Popup *p = Object::cast_to<Popup>(get_parent());
			if (p) {
				p->set_size(Size2(get_combined_minimum_size().width + get_theme_constant("margin") * 2, get_combined_minimum_size().height + get_theme_constant("margin") * 2));
			}
		} break;
		case NOTIFICATION_WM_CLOSE_REQUEST: {
			if (screen != nullptr && screen->is_visible()) {
				screen->hide();
			}
		} break;
	}
}

void ColorPicker::set_focus_on_line_edit() {
	c_text->call_deferred("grab_focus");
}

void ColorPicker::_update_controls() {
	const char *rgb[3] = { "R", "G", "B" };
	const char *hsv[3] = { "H", "S", "V" };

	if (hsv_mode_enabled) {
		for (int i = 0; i < 3; i++) {
			labels[i]->set_text(hsv[i]);
		}
	} else {
		for (int i = 0; i < 3; i++) {
			labels[i]->set_text(rgb[i]);
		}
	}

	if (hsv_mode_enabled) {
		set_raw_mode(false);
		btn_raw->set_disabled(true);
	} else if (raw_mode_enabled) {
		set_hsv_mode(false);
		btn_hsv->set_disabled(true);
	} else {
		btn_raw->set_disabled(false);
		btn_hsv->set_disabled(false);
	}

	if (edit_alpha) {
		values[3]->show();
		scroll[3]->show();
		labels[3]->show();
	} else {
		values[3]->hide();
		scroll[3]->hide();
		labels[3]->hide();
	}
}

void ColorPicker::_set_pick_color(const Color &p_color, bool p_update_sliders) {
	color = p_color;
	if (color != last_hsv) {
		h = color.get_h();
		s = color.get_s();
		v = color.get_v();
		last_hsv = color;
	}

	if (!is_inside_tree()) {
		return;
	}

	_update_color(p_update_sliders);
}

void ColorPicker::set_pick_color(const Color &p_color) {
	_set_pick_color(p_color, true); //because setters can't have more arguments
}

void ColorPicker::set_edit_alpha(bool p_show) {
	edit_alpha = p_show;
	_update_controls();

	if (!is_inside_tree()) {
		return;
	}

	_update_color();
	sample->update();
}

bool ColorPicker::is_editing_alpha() const {
	return edit_alpha;
}

void ColorPicker::_value_changed(double) {
	if (updating) {
		return;
	}

	if (hsv_mode_enabled) {
		color.set_hsv(scroll[0]->get_value() / 360.0,
				scroll[1]->get_value() / 100.0,
				scroll[2]->get_value() / 100.0,
				scroll[3]->get_value() / 255.0);
	} else {
		for (int i = 0; i < 4; i++) {
			color.components[i] = scroll[i]->get_value() / (raw_mode_enabled ? 1.0 : 255.0);
		}
	}

	_set_pick_color(color, false);
	emit_signal("color_changed", color);
}

void ColorPicker::_html_entered(const String &p_html) {
	if (updating || text_is_constructor || !c_text->is_visible()) {
		return;
	}

	float last_alpha = color.a;
	color = Color::html(p_html);
	if (!is_editing_alpha()) {
		color.a = last_alpha;
	}

	if (!is_inside_tree()) {
		return;
	}

	set_pick_color(color);
	emit_signal("color_changed", color);
}

void ColorPicker::_update_color(bool p_update_sliders) {
	updating = true;

	if (p_update_sliders) {
		if (hsv_mode_enabled) {
			for (int i = 0; i < 4; i++) {
				scroll[i]->set_step(1.0);
			}

			scroll[0]->set_max(359);
			scroll[0]->set_value(h * 360.0);
			scroll[1]->set_max(100);
			scroll[1]->set_value(s * 100.0);
			scroll[2]->set_max(100);
			scroll[2]->set_value(v * 100.0);
			scroll[3]->set_max(255);
			scroll[3]->set_value(color.components[3] * 255.0);
		} else {
			for (int i = 0; i < 4; i++) {
				if (raw_mode_enabled) {
					scroll[i]->set_step(0.01);
					scroll[i]->set_max(100);
					if (i == 3) {
						scroll[i]->set_max(1);
					}
					scroll[i]->set_value(color.components[i]);
				} else {
					scroll[i]->set_step(1);
					const float byte_value = color.components[i] * 255.0;
					scroll[i]->set_max(next_power_of_2(MAX(255, byte_value)) - 1);
					scroll[i]->set_value(byte_value);
				}
			}
		}
	}

	_update_text_value();

	sample->update();
	uv_edit->update();
	w_edit->update();
	updating = false;
}

void ColorPicker::_update_presets() {
	return;
	//presets should be shown using buttons or something else, this method is not a good idea

	presets_per_row = 10;
	Size2 size = bt_add_preset->get_size();
	Size2 preset_size = Size2(MIN(size.width * presets.size(), presets_per_row * size.width), size.height * (Math::ceil((float)presets.size() / presets_per_row)));
	preset->set_custom_minimum_size(preset_size);
	preset_container->set_custom_minimum_size(preset_size);
	preset->draw_rect(Rect2(Point2(), preset_size), Color(1, 1, 1, 0));

	for (int i = 0; i < presets.size(); i++) {
		int x = (i % presets_per_row) * size.width;
		int y = (Math::floor((float)i / presets_per_row)) * size.height;
		preset->draw_rect(Rect2(Point2(x, y), size), presets[i]);
	}
	_notification(NOTIFICATION_VISIBILITY_CHANGED);
}

void ColorPicker::_text_type_toggled() {
	text_is_constructor = !text_is_constructor;
	if (text_is_constructor) {
		text_type->set_text("");
		text_type->set_icon(get_theme_icon("Script", "EditorIcons"));

		c_text->set_editable(false);
	} else {
		text_type->set_text("#");
		text_type->set_icon(nullptr);

		c_text->set_editable(true);
	}
	_update_color();
}

Color ColorPicker::get_pick_color() const {
	return color;
}

void ColorPicker::add_preset(const Color &p_color) {
	if (presets.find(p_color)) {
		presets.move_to_back(presets.find(p_color));
	} else {
		presets.push_back(p_color);
	}
	preset->update();

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		PackedColorArray arr_to_save = get_presets();
		EditorSettings::get_singleton()->set_project_metadata("color_picker", "presets", arr_to_save);
	}
#endif
}

void ColorPicker::erase_preset(const Color &p_color) {
	if (presets.find(p_color)) {
		presets.erase(presets.find(p_color));
		preset->update();

#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint()) {
			PackedColorArray arr_to_save = get_presets();
			EditorSettings::get_singleton()->set_project_metadata("color_picker", "presets", arr_to_save);
		}
#endif
	}
}

PackedColorArray ColorPicker::get_presets() const {
	PackedColorArray arr;
	arr.resize(presets.size());
	for (int i = 0; i < presets.size(); i++) {
		arr.set(i, presets[i]);
	}
	return arr;
}

void ColorPicker::set_hsv_mode(bool p_enabled) {
	if (hsv_mode_enabled == p_enabled || raw_mode_enabled) {
		return;
	}
	hsv_mode_enabled = p_enabled;
	if (btn_hsv->is_pressed() != p_enabled) {
		btn_hsv->set_pressed(p_enabled);
	}

	if (!is_inside_tree()) {
		return;
	}

	_update_controls();
	_update_color();
}

bool ColorPicker::is_hsv_mode() const {
	return hsv_mode_enabled;
}

void ColorPicker::set_raw_mode(bool p_enabled) {
	if (raw_mode_enabled == p_enabled || hsv_mode_enabled) {
		return;
	}
	raw_mode_enabled = p_enabled;
	if (btn_raw->is_pressed() != p_enabled) {
		btn_raw->set_pressed(p_enabled);
	}

	if (!is_inside_tree()) {
		return;
	}

	_update_controls();
	_update_color();
}

bool ColorPicker::is_raw_mode() const {
	return raw_mode_enabled;
}

void ColorPicker::set_deferred_mode(bool p_enabled) {
	deferred_mode_enabled = p_enabled;
}

bool ColorPicker::is_deferred_mode() const {
	return deferred_mode_enabled;
}

void ColorPicker::_update_text_value() {
	bool visible = true;
	if (text_is_constructor) {
		String t = "Color(" + String::num(color.r) + ", " + String::num(color.g) + ", " + String::num(color.b);
		if (edit_alpha && color.a < 1) {
			t += ", " + String::num(color.a) + ")";
		} else {
			t += ")";
		}
		c_text->set_text(t);
	}

	if (color.r > 1 || color.g > 1 || color.b > 1 || color.r < 0 || color.g < 0 || color.b < 0) {
		visible = false;
	} else if (!text_is_constructor) {
		c_text->set_text(color.to_html(edit_alpha && color.a < 1));
	}

	text_type->set_visible(visible);
	c_text->set_visible(visible);
}

void ColorPicker::_sample_draw() {
	const Rect2 r = Rect2(Point2(), Size2(uv_edit->get_size().width, sample->get_size().height * 0.95));

	if (color.a < 1.0) {
		sample->draw_texture_rect(get_theme_icon("preset_bg", "ColorPicker"), r, true);
	}

	sample->draw_rect(r, color);

	if (color.r > 1 || color.g > 1 || color.b > 1) {
		// Draw an indicator to denote that the color is "overbright" and can't be displayed accurately in the preview
		sample->draw_texture(get_theme_icon("overbright_indicator", "ColorPicker"), Point2());
	}
}

void ColorPicker::_hsv_draw(int p_which, Control *c) {
	if (!c) {
		return;
	}
	if (p_which == 0) {
		Vector<Point2> points;
		points.push_back(Vector2());
		points.push_back(Vector2(c->get_size().x, 0));
		points.push_back(c->get_size());
		points.push_back(Vector2(0, c->get_size().y));
		Vector<Color> colors;
		colors.push_back(Color(1, 1, 1, 1));
		colors.push_back(Color(1, 1, 1, 1));
		colors.push_back(Color(0, 0, 0, 1));
		colors.push_back(Color(0, 0, 0, 1));
		c->draw_polygon(points, colors);
		Vector<Color> colors2;
		Color col = color;
		col.set_hsv(h, 1, 1);
		col.a = 0;
		colors2.push_back(col);
		col.a = 1;
		colors2.push_back(col);
		col.set_hsv(h, 1, 0);
		colors2.push_back(col);
		col.a = 0;
		colors2.push_back(col);
		c->draw_polygon(points, colors2);
		int x = CLAMP(c->get_size().x * s, 0, c->get_size().x);
		int y = CLAMP(c->get_size().y - c->get_size().y * v, 0, c->get_size().y);
		col = color;
		col.a = 1;
		c->draw_line(Point2(x, 0), Point2(x, c->get_size().y), col.inverted());
		c->draw_line(Point2(0, y), Point2(c->get_size().x, y), col.inverted());
		c->draw_line(Point2(x, y), Point2(x, y), Color(1, 1, 1), 2);
	} else if (p_which == 1) {
		Ref<Texture2D> hue = get_theme_icon("color_hue", "ColorPicker");
		c->draw_texture_rect(hue, Rect2(Point2(), c->get_size()));
		int y = c->get_size().y - c->get_size().y * (1.0 - h);
		Color col = Color();
		col.set_hsv(h, 1, 1);
		c->draw_line(Point2(0, y), Point2(c->get_size().x, y), col.inverted());
	}
}

void ColorPicker::_uv_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> bev = p_event;

	if (bev.is_valid()) {
		if (bev->is_pressed() && bev->get_button_index() == BUTTON_LEFT) {
			changing_color = true;
			float x = CLAMP((float)bev->get_position().x, 0, uv_edit->get_size().width);
			float y = CLAMP((float)bev->get_position().y, 0, uv_edit->get_size().height);
			s = x / uv_edit->get_size().width;
			v = 1.0 - y / uv_edit->get_size().height;
			color.set_hsv(h, s, v, color.a);
			last_hsv = color;
			set_pick_color(color);
			_update_color();
			if (!deferred_mode_enabled) {
				emit_signal("color_changed", color);
			}
		} else if (deferred_mode_enabled && !bev->is_pressed() && bev->get_button_index() == BUTTON_LEFT) {
			emit_signal("color_changed", color);
			changing_color = false;
		} else {
			changing_color = false;
		}
	}

	Ref<InputEventMouseMotion> mev = p_event;

	if (mev.is_valid()) {
		if (!changing_color) {
			return;
		}
		float x = CLAMP((float)mev->get_position().x, 0, uv_edit->get_size().width);
		float y = CLAMP((float)mev->get_position().y, 0, uv_edit->get_size().height);
		s = x / uv_edit->get_size().width;
		v = 1.0 - y / uv_edit->get_size().height;
		color.set_hsv(h, s, v, color.a);
		last_hsv = color;
		set_pick_color(color);
		_update_color();
		if (!deferred_mode_enabled) {
			emit_signal("color_changed", color);
		}
	}
}

void ColorPicker::_w_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> bev = p_event;

	if (bev.is_valid()) {
		if (bev->is_pressed() && bev->get_button_index() == BUTTON_LEFT) {
			changing_color = true;
			float y = CLAMP((float)bev->get_position().y, 0, w_edit->get_size().height);
			h = y / w_edit->get_size().height;
		} else {
			changing_color = false;
		}
		color.set_hsv(h, s, v, color.a);
		last_hsv = color;
		set_pick_color(color);
		_update_color();
		if (!deferred_mode_enabled) {
			emit_signal("color_changed", color);
		} else if (!bev->is_pressed() && bev->get_button_index() == BUTTON_LEFT) {
			emit_signal("color_changed", color);
		}
	}

	Ref<InputEventMouseMotion> mev = p_event;

	if (mev.is_valid()) {
		if (!changing_color) {
			return;
		}
		float y = CLAMP((float)mev->get_position().y, 0, w_edit->get_size().height);
		h = y / w_edit->get_size().height;
		color.set_hsv(h, s, v, color.a);
		last_hsv = color;
		set_pick_color(color);
		_update_color();
		if (!deferred_mode_enabled) {
			emit_signal("color_changed", color);
		}
	}
}

void ColorPicker::_preset_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> bev = p_event;

	if (bev.is_valid()) {
		int index = 0;
		if (bev->is_pressed() && bev->get_button_index() == BUTTON_LEFT) {
			for (int i = 0; i < presets.size(); i++) {
				int x = (i % presets_per_row) * bt_add_preset->get_size().x;
				int y = (Math::floor((float)i / presets_per_row)) * bt_add_preset->get_size().y;
				if (bev->get_position().x > x && bev->get_position().x < x + preset->get_size().x && bev->get_position().y > y && bev->get_position().y < y + preset->get_size().y) {
					index = i;
				}
			}
			set_pick_color(presets[index]);
			_update_color();
			emit_signal("color_changed", color);
		} else if (bev->is_pressed() && bev->get_button_index() == BUTTON_RIGHT && presets_enabled) {
			index = bev->get_position().x / (preset->get_size().x / presets.size());
			Color clicked_preset = presets[index];
			erase_preset(clicked_preset);
			emit_signal("preset_removed", clicked_preset);
			bt_add_preset->show();
		}
	}

	Ref<InputEventMouseMotion> mev = p_event;

	if (mev.is_valid()) {
		int index = mev->get_position().x * presets.size();
		if (preset->get_size().x != 0) {
			index /= preset->get_size().x;
		}
		if (index < 0 || index >= presets.size()) {
			return;
		}
		preset->set_tooltip(vformat(RTR("Color: #%s\nLMB: Set color\nRMB: Remove preset"), presets[index].to_html(presets[index].a < 1)));
	}
}

void ColorPicker::_screen_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> bev = p_event;
	if (bev.is_valid() && bev->get_button_index() == BUTTON_LEFT && !bev->is_pressed()) {
		emit_signal("color_changed", color);
		screen->hide();
	}

	Ref<InputEventMouseMotion> mev = p_event;
	if (mev.is_valid()) {
		Viewport *r = get_tree()->get_root();
		if (!r->get_visible_rect().has_point(Point2(mev->get_global_position().x, mev->get_global_position().y))) {
			return;
		}

		Ref<Image> img = r->get_texture()->get_data();
		if (img.is_valid() && !img->empty()) {
			Vector2 ofs = mev->get_global_position() - r->get_visible_rect().get_position();
			Color c = img->get_pixel(ofs.x, r->get_visible_rect().size.height - ofs.y);

			set_pick_color(c);
		}
	}
}

void ColorPicker::_add_preset_pressed() {
	add_preset(color);
	emit_signal("preset_added", color);
}

void ColorPicker::_screen_pick_pressed() {
	Viewport *r = get_tree()->get_root();
	if (!screen) {
		screen = memnew(Control);
		r->add_child(screen);
		screen->set_as_top_level(true);
		screen->set_anchors_and_margins_preset(Control::PRESET_WIDE);
		screen->set_default_cursor_shape(CURSOR_POINTING_HAND);
		screen->connect("gui_input", callable_mp(this, &ColorPicker::_screen_input));
		// It immediately toggles off in the first press otherwise.
		screen->call_deferred("connect", "hide", Callable(btn_pick, "set_pressed"), varray(false));
	}
	screen->raise();
#ifndef _MSC_VER
#warning show modal no longer works, needs to be converted to a popup
#endif
	//screen->show_modal();
}

void ColorPicker::_focus_enter() {
	bool has_ctext_focus = c_text->has_focus();
	if (has_ctext_focus) {
		c_text->select_all();
	} else {
		c_text->select(0, 0);
	}

	for (int i = 0; i < 4; i++) {
		if (values[i]->get_line_edit()->has_focus() && !has_ctext_focus) {
			values[i]->get_line_edit()->select_all();
		} else {
			values[i]->get_line_edit()->select(0, 0);
		}
	}
}

void ColorPicker::_focus_exit() {
	for (int i = 0; i < 4; i++) {
		if (!values[i]->get_line_edit()->get_menu()->is_visible()) {
			values[i]->get_line_edit()->select(0, 0);
		}
	}
	c_text->select(0, 0);
}

void ColorPicker::_html_focus_exit() {
	if (c_text->get_menu()->is_visible()) {
		return;
	}
	_html_entered(c_text->get_text());
	_focus_exit();
}

void ColorPicker::set_presets_enabled(bool p_enabled) {
	presets_enabled = p_enabled;
	if (!p_enabled) {
		bt_add_preset->set_disabled(true);
		bt_add_preset->set_focus_mode(FOCUS_NONE);
	} else {
		bt_add_preset->set_disabled(false);
		bt_add_preset->set_focus_mode(FOCUS_ALL);
	}
}

bool ColorPicker::are_presets_enabled() const {
	return presets_enabled;
}

void ColorPicker::set_presets_visible(bool p_visible) {
	presets_visible = p_visible;
	preset_separator->set_visible(p_visible);
	preset_container->set_visible(p_visible);
	preset_container2->set_visible(p_visible);
}

bool ColorPicker::are_presets_visible() const {
	return presets_visible;
}

void ColorPicker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_pick_color", "color"), &ColorPicker::set_pick_color);
	ClassDB::bind_method(D_METHOD("get_pick_color"), &ColorPicker::get_pick_color);
	ClassDB::bind_method(D_METHOD("set_hsv_mode"), &ColorPicker::set_hsv_mode);
	ClassDB::bind_method(D_METHOD("is_hsv_mode"), &ColorPicker::is_hsv_mode);
	ClassDB::bind_method(D_METHOD("set_raw_mode"), &ColorPicker::set_raw_mode);
	ClassDB::bind_method(D_METHOD("is_raw_mode"), &ColorPicker::is_raw_mode);
	ClassDB::bind_method(D_METHOD("set_deferred_mode", "mode"), &ColorPicker::set_deferred_mode);
	ClassDB::bind_method(D_METHOD("is_deferred_mode"), &ColorPicker::is_deferred_mode);
	ClassDB::bind_method(D_METHOD("set_edit_alpha", "show"), &ColorPicker::set_edit_alpha);
	ClassDB::bind_method(D_METHOD("is_editing_alpha"), &ColorPicker::is_editing_alpha);
	ClassDB::bind_method(D_METHOD("set_presets_enabled", "enabled"), &ColorPicker::set_presets_enabled);
	ClassDB::bind_method(D_METHOD("are_presets_enabled"), &ColorPicker::are_presets_enabled);
	ClassDB::bind_method(D_METHOD("set_presets_visible", "visible"), &ColorPicker::set_presets_visible);
	ClassDB::bind_method(D_METHOD("are_presets_visible"), &ColorPicker::are_presets_visible);
	ClassDB::bind_method(D_METHOD("add_preset", "color"), &ColorPicker::add_preset);
	ClassDB::bind_method(D_METHOD("erase_preset", "color"), &ColorPicker::erase_preset);
	ClassDB::bind_method(D_METHOD("get_presets"), &ColorPicker::get_presets);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_pick_color", "get_pick_color");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "edit_alpha"), "set_edit_alpha", "is_editing_alpha");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hsv_mode"), "set_hsv_mode", "is_hsv_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "raw_mode"), "set_raw_mode", "is_raw_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deferred_mode"), "set_deferred_mode", "is_deferred_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "presets_enabled"), "set_presets_enabled", "are_presets_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "presets_visible"), "set_presets_visible", "are_presets_visible");

	ADD_SIGNAL(MethodInfo("color_changed", PropertyInfo(Variant::COLOR, "color")));
	ADD_SIGNAL(MethodInfo("preset_added", PropertyInfo(Variant::COLOR, "color")));
	ADD_SIGNAL(MethodInfo("preset_removed", PropertyInfo(Variant::COLOR, "color")));
}

ColorPicker::ColorPicker() :
		BoxContainer(true) {
	updating = true;
	edit_alpha = true;
	text_is_constructor = false;
	hsv_mode_enabled = false;
	raw_mode_enabled = false;
	deferred_mode_enabled = false;
	changing_color = false;
	presets_enabled = true;
	presets_visible = true;
	screen = nullptr;

	HBoxContainer *hb_edit = memnew(HBoxContainer);
	add_child(hb_edit);
	hb_edit->set_v_size_flags(SIZE_EXPAND_FILL);

	uv_edit = memnew(Control);
	hb_edit->add_child(uv_edit);
	uv_edit->connect("gui_input", callable_mp(this, &ColorPicker::_uv_input));
	uv_edit->set_mouse_filter(MOUSE_FILTER_PASS);
	uv_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	uv_edit->set_v_size_flags(SIZE_EXPAND_FILL);
	uv_edit->set_custom_minimum_size(Size2(get_theme_constant("sv_width"), get_theme_constant("sv_height")));
	uv_edit->connect("draw", callable_mp(this, &ColorPicker::_hsv_draw), make_binds(0, uv_edit));

	w_edit = memnew(Control);
	hb_edit->add_child(w_edit);
	w_edit->set_custom_minimum_size(Size2(get_theme_constant("h_width"), 0));
	w_edit->set_h_size_flags(SIZE_FILL);
	w_edit->set_v_size_flags(SIZE_EXPAND_FILL);
	w_edit->connect("gui_input", callable_mp(this, &ColorPicker::_w_input));
	w_edit->connect("draw", callable_mp(this, &ColorPicker::_hsv_draw), make_binds(1, w_edit));

	HBoxContainer *hb_smpl = memnew(HBoxContainer);
	add_child(hb_smpl);

	sample = memnew(TextureRect);
	hb_smpl->add_child(sample);
	sample->set_h_size_flags(SIZE_EXPAND_FILL);
	sample->connect("draw", callable_mp(this, &ColorPicker::_sample_draw));

	btn_pick = memnew(Button);
	btn_pick->set_flat(true);
	hb_smpl->add_child(btn_pick);
	btn_pick->set_toggle_mode(true);
	btn_pick->set_tooltip(TTR("Pick a color from the editor window."));
	btn_pick->connect("pressed", callable_mp(this, &ColorPicker::_screen_pick_pressed));

	VBoxContainer *vbl = memnew(VBoxContainer);
	add_child(vbl);

	add_child(memnew(HSeparator));

	VBoxContainer *vbr = memnew(VBoxContainer);
	add_child(vbr);
	vbr->set_h_size_flags(SIZE_EXPAND_FILL);

	for (int i = 0; i < 4; i++) {
		HBoxContainer *hbc = memnew(HBoxContainer);

		labels[i] = memnew(Label());
		labels[i]->set_custom_minimum_size(Size2(get_theme_constant("label_width"), 0));
		labels[i]->set_v_size_flags(SIZE_SHRINK_CENTER);
		hbc->add_child(labels[i]);

		scroll[i] = memnew(HSlider);
		scroll[i]->set_v_size_flags(SIZE_SHRINK_CENTER);
		scroll[i]->set_focus_mode(FOCUS_NONE);
		hbc->add_child(scroll[i]);

		values[i] = memnew(SpinBox);
		scroll[i]->share(values[i]);
		hbc->add_child(values[i]);
		values[i]->get_line_edit()->connect("focus_entered", callable_mp(this, &ColorPicker::_focus_enter));
		values[i]->get_line_edit()->connect("focus_exited", callable_mp(this, &ColorPicker::_focus_exit));

		scroll[i]->set_min(0);
		scroll[i]->set_page(0);
		scroll[i]->set_h_size_flags(SIZE_EXPAND_FILL);

		scroll[i]->connect("value_changed", callable_mp(this, &ColorPicker::_value_changed));

		vbr->add_child(hbc);
	}
	labels[3]->set_text("A");

	HBoxContainer *hhb = memnew(HBoxContainer);
	vbr->add_child(hhb);

	btn_hsv = memnew(CheckButton);
	hhb->add_child(btn_hsv);
	btn_hsv->set_text(TTR("HSV"));
	btn_hsv->connect("toggled", callable_mp(this, &ColorPicker::set_hsv_mode));

	btn_raw = memnew(CheckButton);
	hhb->add_child(btn_raw);
	btn_raw->set_text(TTR("Raw"));
	btn_raw->connect("toggled", callable_mp(this, &ColorPicker::set_raw_mode));

	text_type = memnew(Button);
	hhb->add_child(text_type);
	text_type->set_text("#");
	text_type->set_tooltip(TTR("Switch between hexadecimal and code values."));
	if (Engine::get_singleton()->is_editor_hint()) {
#ifdef TOOLS_ENABLED
		text_type->set_custom_minimum_size(Size2(28 * EDSCALE, 0)); // Adjust for the width of the "Script" icon.
#endif
		text_type->connect("pressed", callable_mp(this, &ColorPicker::_text_type_toggled));
	} else {
		text_type->set_flat(true);
		text_type->set_mouse_filter(MOUSE_FILTER_IGNORE);
	}

	c_text = memnew(LineEdit);
	hhb->add_child(c_text);
	c_text->set_h_size_flags(SIZE_EXPAND_FILL);
	c_text->connect("text_entered", callable_mp(this, &ColorPicker::_html_entered));
	c_text->connect("focus_entered", callable_mp(this, &ColorPicker::_focus_enter));
	c_text->connect("focus_exited", callable_mp(this, &ColorPicker::_html_focus_exit));

	_update_controls();
	updating = false;

	set_pick_color(Color(1, 1, 1));

	preset_separator = memnew(HSeparator);
	add_child(preset_separator);

	preset_container = memnew(HBoxContainer);
	preset_container->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(preset_container);

	preset = memnew(TextureRect);
	preset_container->add_child(preset);
	preset->connect("gui_input", callable_mp(this, &ColorPicker::_preset_input));
	preset->connect("draw", callable_mp(this, &ColorPicker::_update_presets));

	preset_container2 = memnew(HBoxContainer);
	preset_container2->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(preset_container2);
	bt_add_preset = memnew(Button);
	preset_container2->add_child(bt_add_preset);
	bt_add_preset->set_tooltip(TTR("Add current color as a preset."));
	bt_add_preset->connect("pressed", callable_mp(this, &ColorPicker::_add_preset_pressed));
}

/////////////////

void ColorPickerButton::_color_changed(const Color &p_color) {
	color = p_color;
	update();
	emit_signal("color_changed", color);
}

void ColorPickerButton::_modal_closed() {
	emit_signal("popup_closed");
	set_pressed(false);
}

void ColorPickerButton::pressed() {
	_update_picker();

	popup->set_as_minsize();

	Rect2i usable_rect = popup->get_usable_parent_rect();
	//let's try different positions to see which one we can use

	Rect2i cp_rect(Point2i(), popup->get_size());
	for (int i = 0; i < 4; i++) {
		if (i > 1) {
			cp_rect.position.y = get_screen_position().y - cp_rect.size.y;
		} else {
			cp_rect.position.y = get_screen_position().y + get_size().height;
		}

		if (i & 1) {
			cp_rect.position.x = get_screen_position().x;
		} else {
			cp_rect.position.x = get_screen_position().x - MAX(0, (cp_rect.size.x - get_size().x));
		}

		if (usable_rect.encloses(cp_rect)) {
			break;
		}
	}
	popup->set_position(cp_rect.position);
	popup->popup();
	picker->set_focus_on_line_edit();
}

void ColorPickerButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			const Ref<StyleBox> normal = get_theme_stylebox("normal");
			const Rect2 r = Rect2(normal->get_offset(), get_size() - normal->get_minimum_size());
			draw_texture_rect(Control::get_theme_icon("bg", "ColorPickerButton"), r, true);
			draw_rect(r, color);

			if (color.r > 1 || color.g > 1 || color.b > 1) {
				// Draw an indicator to denote that the color is "overbright" and can't be displayed accurately in the preview
				draw_texture(Control::get_theme_icon("overbright_indicator", "ColorPicker"), normal->get_offset());
			}
		} break;
		case NOTIFICATION_WM_CLOSE_REQUEST: {
			if (popup) {
				popup->hide();
			}
		} break;
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (popup && !is_visible_in_tree()) {
			popup->hide();
		}
	}
}

void ColorPickerButton::set_pick_color(const Color &p_color) {
	color = p_color;
	if (picker) {
		picker->set_pick_color(p_color);
	}

	update();
}

Color ColorPickerButton::get_pick_color() const {
	return color;
}

void ColorPickerButton::set_edit_alpha(bool p_show) {
	edit_alpha = p_show;
	if (picker) {
		picker->set_edit_alpha(p_show);
	}
}

bool ColorPickerButton::is_editing_alpha() const {
	return edit_alpha;
}

ColorPicker *ColorPickerButton::get_picker() {
	_update_picker();
	return picker;
}

PopupPanel *ColorPickerButton::get_popup() {
	_update_picker();
	return popup;
}

void ColorPickerButton::_update_picker() {
	if (!picker) {
		popup = memnew(PopupPanel);
		popup->set_wrap_controls(true);
		picker = memnew(ColorPicker);
		picker->set_anchors_and_margins_preset(PRESET_WIDE);
		popup->add_child(picker);
		add_child(popup);
		picker->connect("color_changed", callable_mp(this, &ColorPickerButton::_color_changed));
		popup->connect("about_to_popup", callable_mp((BaseButton *)this, &BaseButton::set_pressed), varray(true));
		popup->connect("popup_hide", callable_mp(this, &ColorPickerButton::_modal_closed));
		picker->set_pick_color(color);
		picker->set_edit_alpha(edit_alpha);
		emit_signal("picker_created");
	}
}

void ColorPickerButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_pick_color", "color"), &ColorPickerButton::set_pick_color);
	ClassDB::bind_method(D_METHOD("get_pick_color"), &ColorPickerButton::get_pick_color);
	ClassDB::bind_method(D_METHOD("get_picker"), &ColorPickerButton::get_picker);
	ClassDB::bind_method(D_METHOD("get_popup"), &ColorPickerButton::get_popup);
	ClassDB::bind_method(D_METHOD("set_edit_alpha", "show"), &ColorPickerButton::set_edit_alpha);
	ClassDB::bind_method(D_METHOD("is_editing_alpha"), &ColorPickerButton::is_editing_alpha);

	ADD_SIGNAL(MethodInfo("color_changed", PropertyInfo(Variant::COLOR, "color")));
	ADD_SIGNAL(MethodInfo("popup_closed"));
	ADD_SIGNAL(MethodInfo("picker_created"));
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_pick_color", "get_pick_color");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "edit_alpha"), "set_edit_alpha", "is_editing_alpha");
}

ColorPickerButton::ColorPickerButton() {
	// Initialization is now done deferred,
	// this improves performance in the inspector as the color picker
	// can be expensive to initialize.
	picker = nullptr;
	popup = nullptr;
	edit_alpha = true;

	set_toggle_mode(true);
}

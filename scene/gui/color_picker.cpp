/*************************************************************************/
/*  color_picker.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "editor/editor_settings.h"
#endif
#include "scene/main/window.h"

List<Color> ColorPicker::preset_cache;

void ColorPicker::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_color();

#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				if (preset_cache.is_empty()) {
					PackedColorArray saved_presets = EditorSettings::get_singleton()->get_project_metadata("color_picker", "presets", PackedColorArray());
					for (int i = 0; i < saved_presets.size(); i++) {
						preset_cache.push_back(saved_presets[i]);
					}
				}

				for (int i = 0; i < preset_cache.size(); i++) {
					presets.push_back(preset_cache[i]);
				}
			}
#endif
			[[fallthrough]];
		}
		case NOTIFICATION_THEME_CHANGED: {
			btn_pick->set_icon(get_theme_icon(SNAME("screen_picker"), SNAME("ColorPicker")));
			btn_add_preset->set_icon(get_theme_icon(SNAME("add_preset")));

			uv_edit->set_custom_minimum_size(Size2(get_theme_constant(SNAME("sv_width")), get_theme_constant(SNAME("sv_height"))));
			w_edit->set_custom_minimum_size(Size2(get_theme_constant(SNAME("h_width")), 0));

			wheel_edit->set_custom_minimum_size(Size2(get_theme_constant(SNAME("sv_width")), get_theme_constant(SNAME("sv_height"))));
			wheel_margin->add_theme_constant_override("margin_bottom", 8 * get_theme_default_base_scale());

			for (int i = 0; i < 4; i++) {
				labels[i]->set_custom_minimum_size(Size2(get_theme_constant(SNAME("label_width")), 0));
				set_offset((Side)i, get_offset((Side)i) + get_theme_constant(SNAME("margin")));
			}

			if (Engine::get_singleton()->is_editor_hint()) {
				// Adjust for the width of the "Script" icon.
				text_type->set_custom_minimum_size(Size2(28 * get_theme_default_base_scale(), 0));
			}

			_update_presets();
			_update_controls();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			Popup *p = Object::cast_to<Popup>(get_parent());
			if (p) {
				p->set_size(Size2(get_combined_minimum_size().width + get_theme_constant(SNAME("margin")) * 2, get_combined_minimum_size().height + get_theme_constant(SNAME("margin")) * 2));
			}
		} break;

		case NOTIFICATION_WM_CLOSE_REQUEST: {
			if (screen != nullptr && screen->is_visible()) {
				screen->hide();
			}
		} break;
	}
}

Ref<Shader> ColorPicker::wheel_shader;
Ref<Shader> ColorPicker::circle_shader;

void ColorPicker::init_shaders() {
	wheel_shader.instantiate();
	wheel_shader->set_code(R"(
// ColorPicker wheel shader.

shader_type canvas_item;

void fragment() {
	float x = UV.x - 0.5;
	float y = UV.y - 0.5;
	float a = atan(y, x);
	x += 0.001;
	y += 0.001;
	float b = float(sqrt(x * x + y * y) < 0.5) * float(sqrt(x * x + y * y) > 0.42);
	x -= 0.002;
	float b2 = float(sqrt(x * x + y * y) < 0.5) * float(sqrt(x * x + y * y) > 0.42);
	y -= 0.002;
	float b3 = float(sqrt(x * x + y * y) < 0.5) * float(sqrt(x * x + y * y) > 0.42);
	x += 0.002;
	float b4 = float(sqrt(x * x + y * y) < 0.5) * float(sqrt(x * x + y * y) > 0.42);

	COLOR = vec4(clamp((abs(fract(((a - TAU) / TAU) + vec3(3.0, 2.0, 1.0) / 3.0) * 6.0 - 3.0) - 1.0), 0.0, 1.0), (b + b2 + b3 + b4) / 4.00);
}
)");

	circle_shader.instantiate();
	circle_shader->set_code(R"(
// ColorPicker circle shader.

shader_type canvas_item;

uniform float v = 1.0;

void fragment() {
	float x = UV.x - 0.5;
	float y = UV.y - 0.5;
	float a = atan(y, x);
	x += 0.001;
	y += 0.001;
	float b = float(sqrt(x * x + y * y) < 0.5);
	x -= 0.002;
	float b2 = float(sqrt(x * x + y * y) < 0.5);
	y -= 0.002;
	float b3 = float(sqrt(x * x + y * y) < 0.5);
	x += 0.002;
	float b4 = float(sqrt(x * x + y * y) < 0.5);

	COLOR = vec4(mix(vec3(1.0), clamp(abs(fract(vec3((a - TAU) / TAU) + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - vec3(3.0)) - vec3(1.0), 0.0, 1.0), ((float(sqrt(x * x + y * y)) * 2.0)) / 1.0) * vec3(v), (b + b2 + b3 + b4) / 4.00);
})");
}

void ColorPicker::finish_shaders() {
	wheel_shader.unref();
	circle_shader.unref();
}

void ColorPicker::set_focus_on_line_edit() {
	c_text->call_deferred(SNAME("grab_focus"));
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

	if (raw_mode_enabled) {
		for (int i = 0; i < 3; i++) {
			scroll[i]->remove_theme_icon_override("grabber");
			scroll[i]->remove_theme_icon_override("grabber_highlight");
			scroll[i]->remove_theme_style_override("slider");
			scroll[i]->remove_theme_style_override("grabber_area");
			scroll[i]->remove_theme_style_override("grabber_area_highlight");
		}
	} else {
		Ref<StyleBoxEmpty> style_box_empty(memnew(StyleBoxEmpty));
		Ref<Texture2D> bar_arrow = get_theme_icon(SNAME("bar_arrow"));

		for (int i = 0; i < 4; i++) {
			scroll[i]->add_theme_icon_override("grabber", bar_arrow);
			scroll[i]->add_theme_icon_override("grabber_highlight", bar_arrow);
			scroll[i]->add_theme_style_override("slider", style_box_empty);
			scroll[i]->add_theme_style_override("grabber_area", style_box_empty);
			scroll[i]->add_theme_style_override("grabber_area_highlight", style_box_empty);
		}
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

	switch (picker_type) {
		case SHAPE_HSV_RECTANGLE:
			wheel_edit->hide();
			w_edit->show();
			uv_edit->show();
			break;
		case SHAPE_HSV_WHEEL:
			wheel_edit->show();
			w_edit->hide();
			uv_edit->hide();

			wheel->set_material(wheel_mat);
			break;
		case SHAPE_VHS_CIRCLE:
			wheel_edit->show();
			w_edit->show();
			uv_edit->hide();

			wheel->set_material(circle_mat);
			break;
		default: {
		}
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

void ColorPicker::set_old_color(const Color &p_color) {
	old_color = p_color;
}

void ColorPicker::set_display_old_color(bool p_enabled) {
	display_old_color = p_enabled;
}

bool ColorPicker::is_displaying_old_color() const {
	return display_old_color;
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
		h = scroll[0]->get_value() / 360.0;
		s = scroll[1]->get_value() / 100.0;
		v = scroll[2]->get_value() / 100.0;
		color.set_hsv(h, s, v, scroll[3]->get_value() / 255.0);

		last_hsv = color;

	} else {
		for (int i = 0; i < 4; i++) {
			color.components[i] = scroll[i]->get_value() / (raw_mode_enabled ? 1.0 : 255.0);
		}
	}

	_set_pick_color(color, false);
	emit_signal(SNAME("color_changed"), color);
}

void ColorPicker::_html_submitted(const String &p_html) {
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
	emit_signal(SNAME("color_changed"), color);
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
	for (int i = 0; i < 4; i++) {
		scroll[i]->update();
	}
	wheel->update();
	wheel_uv->update();
	updating = false;
}

void ColorPicker::_update_presets() {
	int preset_size = _get_preset_size();
	// Only update the preset button size if it has changed.
	if (preset_size != prev_preset_size) {
		prev_preset_size = preset_size;
		btn_add_preset->set_custom_minimum_size(Size2(preset_size, preset_size));
		for (int i = 1; i < preset_container->get_child_count(); i++) {
			ColorPresetButton *cpb = Object::cast_to<ColorPresetButton>(preset_container->get_child(i));
			cpb->set_custom_minimum_size(Size2(preset_size, preset_size));
		}
	}
	// Only load preset buttons when the only child is the add-preset button.
	if (preset_container->get_child_count() == 1) {
		for (int i = 0; i < preset_cache.size(); i++) {
			_add_preset_button(preset_size, preset_cache[i]);
		}
		_notification(NOTIFICATION_VISIBILITY_CHANGED);
	}
}

void ColorPicker::_text_type_toggled() {
	text_is_constructor = !text_is_constructor;
	if (text_is_constructor) {
		text_type->set_text("");
		text_type->set_icon(get_theme_icon(SNAME("Script"), SNAME("EditorIcons")));

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

void ColorPicker::set_picker_shape(PickerShapeType p_picker_type) {
	ERR_FAIL_INDEX(p_picker_type, SHAPE_MAX);
	picker_type = p_picker_type;

	_update_controls();
	_update_color();
}

ColorPicker::PickerShapeType ColorPicker::get_picker_shape() const {
	return picker_type;
}

inline int ColorPicker::_get_preset_size() {
	return (int(get_minimum_size().width) - (preset_container->get_theme_constant(SNAME("hseparation")) * (preset_column_count - 1))) / preset_column_count;
}

void ColorPicker::_add_preset_button(int p_size, const Color &p_color) {
	ColorPresetButton *btn_preset = memnew(ColorPresetButton(p_color));
	btn_preset->set_preset_color(p_color);
	btn_preset->set_custom_minimum_size(Size2(p_size, p_size));
	btn_preset->connect("gui_input", callable_mp(this, &ColorPicker::_preset_input), varray(p_color));
	btn_preset->set_tooltip(vformat(RTR("Color: #%s\nLMB: Apply color\nRMB: Remove preset"), p_color.to_html(p_color.a < 1)));
	preset_container->add_child(btn_preset);
}

void ColorPicker::add_preset(const Color &p_color) {
	if (presets.find(p_color)) {
		presets.move_to_back(presets.find(p_color));

		// Find button to move to the end.
		for (int i = 1; i < preset_container->get_child_count(); i++) {
			ColorPresetButton *current_btn = Object::cast_to<ColorPresetButton>(preset_container->get_child(i));
			if (current_btn && p_color == current_btn->get_preset_color()) {
				preset_container->move_child(current_btn, preset_container->get_child_count() - 1);
				break;
			}
		}
	} else {
		presets.push_back(p_color);
		preset_cache.push_back(p_color);

		_add_preset_button(_get_preset_size(), p_color);
	}

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
		preset_cache.erase(preset_cache.find(p_color));

		// Find preset button to remove.
		for (int i = 1; i < preset_container->get_child_count(); i++) {
			ColorPresetButton *current_btn = Object::cast_to<ColorPresetButton>(preset_container->get_child(i));
			if (current_btn && p_color == current_btn->get_preset_color()) {
				current_btn->queue_delete();
				break;
			}
		}

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

void ColorPicker::_sample_input(const Ref<InputEvent> &p_event) {
	const Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		const Rect2 rect_old = Rect2(Point2(), Size2(sample->get_size().width * 0.5, sample->get_size().height * 0.95));
		if (rect_old.has_point(mb->get_position())) {
			// Revert to the old color when left-clicking the old color sample.
			color = old_color;
			_update_color();
			emit_signal(SNAME("color_changed"), color);
		}
	}
}

void ColorPicker::_sample_draw() {
	// Covers the right half of the sample if the old color is being displayed,
	// or the whole sample if it's not being displayed.
	Rect2 rect_new;

	if (display_old_color) {
		rect_new = Rect2(Point2(sample->get_size().width * 0.5, 0), Size2(sample->get_size().width * 0.5, sample->get_size().height * 0.95));

		// Draw both old and new colors for easier comparison (only if spawned from a ColorPickerButton).
		const Rect2 rect_old = Rect2(Point2(), Size2(sample->get_size().width * 0.5, sample->get_size().height * 0.95));

		if (display_old_color && old_color.a < 1.0) {
			sample->draw_texture_rect(get_theme_icon(SNAME("sample_bg"), SNAME("ColorPicker")), rect_old, true);
		}

		sample->draw_rect(rect_old, old_color);

		if (old_color.r > 1 || old_color.g > 1 || old_color.b > 1) {
			// Draw an indicator to denote that the old color is "overbright" and can't be displayed accurately in the preview.
			sample->draw_texture(get_theme_icon(SNAME("overbright_indicator"), SNAME("ColorPicker")), Point2());
		}
	} else {
		rect_new = Rect2(Point2(), Size2(sample->get_size().width, sample->get_size().height * 0.95));
	}

	if (color.a < 1.0) {
		sample->draw_texture_rect(get_theme_icon(SNAME("sample_bg"), SNAME("ColorPicker")), rect_new, true);
	}

	sample->draw_rect(rect_new, color);

	if (color.r > 1 || color.g > 1 || color.b > 1) {
		// Draw an indicator to denote that the new color is "overbright" and can't be displayed accurately in the preview.
		sample->draw_texture(get_theme_icon(SNAME("overbright_indicator"), SNAME("ColorPicker")), Point2(uv_edit->get_size().width * 0.5, 0));
	}
}

void ColorPicker::_hsv_draw(int p_which, Control *c) {
	if (!c) {
		return;
	}
	if (p_which == 0) {
		Vector<Point2> points;
		Vector<Color> colors;
		Vector<Color> colors2;
		Color col = color;
		Vector2 center = c->get_size() / 2.0;

		switch (picker_type) {
			case SHAPE_HSV_WHEEL: {
				points.resize(4);
				colors.resize(4);
				colors2.resize(4);
				real_t ring_radius_x = Math_SQRT12 * c->get_size().width * 0.42;
				real_t ring_radius_y = Math_SQRT12 * c->get_size().height * 0.42;

				points.set(0, center - Vector2(ring_radius_x, ring_radius_y));
				points.set(1, center + Vector2(ring_radius_x, -ring_radius_y));
				points.set(2, center + Vector2(ring_radius_x, ring_radius_y));
				points.set(3, center + Vector2(-ring_radius_x, ring_radius_y));
				colors.set(0, Color(1, 1, 1, 1));
				colors.set(1, Color(1, 1, 1, 1));
				colors.set(2, Color(0, 0, 0, 1));
				colors.set(3, Color(0, 0, 0, 1));
				c->draw_polygon(points, colors);

				col.set_hsv(h, 1, 1);
				col.a = 0;
				colors2.set(0, col);
				col.a = 1;
				colors2.set(1, col);
				col.set_hsv(h, 1, 0);
				colors2.set(2, col);
				col.a = 0;
				colors2.set(3, col);
				c->draw_polygon(points, colors2);
				break;
			}
			case SHAPE_HSV_RECTANGLE: {
				points.resize(4);
				colors.resize(4);
				colors2.resize(4);
				points.set(0, Vector2());
				points.set(1, Vector2(c->get_size().x, 0));
				points.set(2, c->get_size());
				points.set(3, Vector2(0, c->get_size().y));
				colors.set(0, Color(1, 1, 1, 1));
				colors.set(1, Color(1, 1, 1, 1));
				colors.set(2, Color(0, 0, 0, 1));
				colors.set(3, Color(0, 0, 0, 1));
				c->draw_polygon(points, colors);
				col = color;
				col.set_hsv(h, 1, 1);
				col.a = 0;
				colors2.set(0, col);
				col.a = 1;
				colors2.set(1, col);
				col.set_hsv(h, 1, 0);
				colors2.set(2, col);
				col.a = 0;
				colors2.set(3, col);
				c->draw_polygon(points, colors2);
				break;
			}
			default: {
			}
		}
		Ref<Texture2D> cursor = get_theme_icon(SNAME("picker_cursor"), SNAME("ColorPicker"));
		int x;
		int y;
		if (picker_type == SHAPE_VHS_CIRCLE) {
			x = center.x + (center.x * Math::cos(h * Math_TAU) * s) - (cursor->get_width() / 2);
			y = center.y + (center.y * Math::sin(h * Math_TAU) * s) - (cursor->get_height() / 2);
		} else {
			real_t corner_x = (c == wheel_uv) ? center.x - Math_SQRT12 * c->get_size().width * 0.42 : 0;
			real_t corner_y = (c == wheel_uv) ? center.y - Math_SQRT12 * c->get_size().height * 0.42 : 0;

			Size2 real_size(c->get_size().x - corner_x * 2, c->get_size().y - corner_y * 2);
			x = CLAMP(real_size.x * s, 0, real_size.x) + corner_x - (cursor->get_width() / 2);
			y = CLAMP(real_size.y - real_size.y * v, 0, real_size.y) + corner_y - (cursor->get_height() / 2);
		}
		c->draw_texture(cursor, Point2(x, y));

		col.set_hsv(h, 1, 1);
		if (picker_type == SHAPE_HSV_WHEEL) {
			points.resize(4);
			double h1 = h - (0.5 / 360);
			double h2 = h + (0.5 / 360);
			points.set(0, Point2(center.x + (center.x * Math::cos(h1 * Math_TAU)), center.y + (center.y * Math::sin(h1 * Math_TAU))));
			points.set(1, Point2(center.x + (center.x * Math::cos(h1 * Math_TAU) * 0.84), center.y + (center.y * Math::sin(h1 * Math_TAU) * 0.84)));
			points.set(2, Point2(center.x + (center.x * Math::cos(h2 * Math_TAU)), center.y + (center.y * Math::sin(h2 * Math_TAU))));
			points.set(3, Point2(center.x + (center.x * Math::cos(h2 * Math_TAU) * 0.84), center.y + (center.y * Math::sin(h2 * Math_TAU) * 0.84)));
			c->draw_multiline(points, col.inverted());
		}

	} else if (p_which == 1) {
		if (picker_type == SHAPE_HSV_RECTANGLE) {
			Ref<Texture2D> hue = get_theme_icon(SNAME("color_hue"), SNAME("ColorPicker"));
			c->draw_texture_rect(hue, Rect2(Point2(), c->get_size()));
			int y = c->get_size().y - c->get_size().y * (1.0 - h);
			Color col;
			col.set_hsv(h, 1, 1);
			c->draw_line(Point2(0, y), Point2(c->get_size().x, y), col.inverted());
		} else if (picker_type == SHAPE_VHS_CIRCLE) {
			Vector<Point2> points;
			Vector<Color> colors;
			Color col;
			col.set_hsv(h, s, 1);
			points.resize(4);
			colors.resize(4);
			points.set(0, Vector2());
			points.set(1, Vector2(c->get_size().x, 0));
			points.set(2, c->get_size());
			points.set(3, Vector2(0, c->get_size().y));
			colors.set(0, col);
			colors.set(1, col);
			colors.set(2, Color(0, 0, 0));
			colors.set(3, Color(0, 0, 0));
			c->draw_polygon(points, colors);
			int y = c->get_size().y - c->get_size().y * CLAMP(v, 0, 1);
			col.set_hsv(h, 1, v);
			c->draw_line(Point2(0, y), Point2(c->get_size().x, y), col.inverted());
		}
	} else if (p_which == 2) {
		c->draw_rect(Rect2(Point2(), c->get_size()), Color(1, 1, 1));
		if (picker_type == SHAPE_VHS_CIRCLE) {
			circle_mat->set_shader_param("v", v);
		}
	}
}

void ColorPicker::_slider_draw(int p_which) {
	Vector<Vector2> pos;
	pos.resize(4);
	Vector<Color> col;
	col.resize(4);
	Size2 size = scroll[p_which]->get_size();
	Color left_color;
	Color right_color;
	const real_t margin = 4 * get_theme_default_base_scale();

	if (p_which == 3) {
		scroll[p_which]->draw_texture_rect(get_theme_icon(SNAME("sample_bg"), SNAME("ColorPicker")), Rect2(Point2(0, margin), Size2(size.x, margin)), true);

		left_color = color;
		left_color.a = 0;
		right_color = color;
		right_color.a = 1;
	} else {
		if (raw_mode_enabled) {
			return;
		}
		if (hsv_mode_enabled) {
			if (p_which == 0) {
				Ref<Texture2D> hue = get_theme_icon(SNAME("color_hue"), SNAME("ColorPicker"));
				scroll[p_which]->draw_set_transform(Point2(), -Math_PI / 2, Size2(1.0, 1.0));
				scroll[p_which]->draw_texture_rect(hue, Rect2(Vector2(margin * -2, 0), Vector2(scroll[p_which]->get_size().x, margin)), false, Color(1, 1, 1), true);
				return;
			}
			Color s_col;
			Color v_col;
			s_col.set_hsv(h, 0, v);
			left_color = (p_which == 1) ? s_col : Color(0, 0, 0);
			s_col.set_hsv(h, 1, v);
			v_col.set_hsv(h, s, 1);
			right_color = (p_which == 1) ? s_col : v_col;
		} else {
			left_color = Color(
					p_which == 0 ? 0 : color.r,
					p_which == 1 ? 0 : color.g,
					p_which == 2 ? 0 : color.b);
			right_color = Color(
					p_which == 0 ? 1 : color.r,
					p_which == 1 ? 1 : color.g,
					p_which == 2 ? 1 : color.b);
		}
	}

	col.set(0, left_color);
	col.set(1, right_color);
	col.set(2, right_color);
	col.set(3, left_color);
	pos.set(0, Vector2(0, margin));
	pos.set(1, Vector2(size.x, margin));
	pos.set(2, Vector2(size.x, margin * 2));
	pos.set(3, Vector2(0, margin * 2));

	scroll[p_which]->draw_polygon(pos, col);
}

void ColorPicker::_uv_input(const Ref<InputEvent> &p_event, Control *c) {
	Ref<InputEventMouseButton> bev = p_event;

	if (bev.is_valid()) {
		if (bev->is_pressed() && bev->get_button_index() == MouseButton::LEFT) {
			Vector2 center = c->get_size() / 2.0;
			if (picker_type == SHAPE_VHS_CIRCLE) {
				real_t dist = center.distance_to(bev->get_position());

				if (dist <= center.x) {
					real_t rad = center.angle_to_point(bev->get_position());
					h = ((rad >= 0) ? rad : (Math_TAU + rad)) / Math_TAU;
					s = CLAMP(dist / center.x, 0, 1);
				} else {
					return;
				}
			} else {
				real_t corner_x = (c == wheel_uv) ? center.x - Math_SQRT12 * c->get_size().width * 0.42 : 0;
				real_t corner_y = (c == wheel_uv) ? center.y - Math_SQRT12 * c->get_size().height * 0.42 : 0;
				Size2 real_size(c->get_size().x - corner_x * 2, c->get_size().y - corner_y * 2);

				if (bev->get_position().x < corner_x || bev->get_position().x > c->get_size().x - corner_x ||
						bev->get_position().y < corner_y || bev->get_position().y > c->get_size().y - corner_y) {
					{
						real_t dist = center.distance_to(bev->get_position());

						if (dist >= center.x * 0.84 && dist <= center.x) {
							real_t rad = center.angle_to_point(bev->get_position());
							h = ((rad >= 0) ? rad : (Math_TAU + rad)) / Math_TAU;
							spinning = true;
						} else {
							return;
						}
					}
				}

				if (!spinning) {
					real_t x = CLAMP(bev->get_position().x, corner_x, c->get_size().x - corner_x);
					real_t y = CLAMP(bev->get_position().y, corner_x, c->get_size().y - corner_y);

					s = (x - c->get_position().x - corner_x) / real_size.x;
					v = 1.0 - (y - c->get_position().y - corner_y) / real_size.y;
				}
			}
			changing_color = true;
			color.set_hsv(h, s, v, color.a);
			last_hsv = color;
			set_pick_color(color);
			_update_color();
			if (!deferred_mode_enabled) {
				emit_signal(SNAME("color_changed"), color);
			}
		} else if (deferred_mode_enabled && !bev->is_pressed() && bev->get_button_index() == MouseButton::LEFT) {
			emit_signal(SNAME("color_changed"), color);
			changing_color = false;
			spinning = false;
		} else {
			changing_color = false;
			spinning = false;
		}
	}

	Ref<InputEventMouseMotion> mev = p_event;

	if (mev.is_valid()) {
		if (!changing_color) {
			return;
		}

		Vector2 center = c->get_size() / 2.0;
		if (picker_type == SHAPE_VHS_CIRCLE) {
			real_t dist = center.distance_to(mev->get_position());
			real_t rad = center.angle_to_point(mev->get_position());
			h = ((rad >= 0) ? rad : (Math_TAU + rad)) / Math_TAU;
			s = CLAMP(dist / center.x, 0, 1);
		} else {
			if (spinning) {
				real_t rad = center.angle_to_point(mev->get_position());
				h = ((rad >= 0) ? rad : (Math_TAU + rad)) / Math_TAU;
			} else {
				real_t corner_x = (c == wheel_uv) ? center.x - Math_SQRT12 * c->get_size().width * 0.42 : 0;
				real_t corner_y = (c == wheel_uv) ? center.y - Math_SQRT12 * c->get_size().height * 0.42 : 0;
				Size2 real_size(c->get_size().x - corner_x * 2, c->get_size().y - corner_y * 2);

				real_t x = CLAMP(mev->get_position().x, corner_x, c->get_size().x - corner_x);
				real_t y = CLAMP(mev->get_position().y, corner_x, c->get_size().y - corner_y);

				s = (x - corner_x) / real_size.x;
				v = 1.0 - (y - corner_y) / real_size.y;
			}
		}

		color.set_hsv(h, s, v, color.a);
		last_hsv = color;
		set_pick_color(color);
		_update_color();
		if (!deferred_mode_enabled) {
			emit_signal(SNAME("color_changed"), color);
		}
	}
}

void ColorPicker::_w_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> bev = p_event;

	if (bev.is_valid()) {
		if (bev->is_pressed() && bev->get_button_index() == MouseButton::LEFT) {
			changing_color = true;
			float y = CLAMP((float)bev->get_position().y, 0, w_edit->get_size().height);
			if (picker_type == SHAPE_VHS_CIRCLE) {
				v = 1.0 - (y / w_edit->get_size().height);
			} else {
				h = y / w_edit->get_size().height;
			}
		} else {
			changing_color = false;
		}
		color.set_hsv(h, s, v, color.a);
		last_hsv = color;
		set_pick_color(color);
		_update_color();
		if (!deferred_mode_enabled) {
			emit_signal(SNAME("color_changed"), color);
		} else if (!bev->is_pressed() && bev->get_button_index() == MouseButton::LEFT) {
			emit_signal(SNAME("color_changed"), color);
		}
	}

	Ref<InputEventMouseMotion> mev = p_event;

	if (mev.is_valid()) {
		if (!changing_color) {
			return;
		}
		float y = CLAMP((float)mev->get_position().y, 0, w_edit->get_size().height);
		if (picker_type == SHAPE_VHS_CIRCLE) {
			v = 1.0 - (y / w_edit->get_size().height);
		} else {
			h = y / w_edit->get_size().height;
		}
		color.set_hsv(h, s, v, color.a);
		last_hsv = color;
		set_pick_color(color);
		_update_color();
		if (!deferred_mode_enabled) {
			emit_signal(SNAME("color_changed"), color);
		}
	}
}

void ColorPicker::_preset_input(const Ref<InputEvent> &p_event, const Color &p_color) {
	Ref<InputEventMouseButton> bev = p_event;

	if (bev.is_valid()) {
		if (bev->is_pressed() && bev->get_button_index() == MouseButton::LEFT) {
			set_pick_color(p_color);
			_update_color();
			emit_signal(SNAME("color_changed"), p_color);
		} else if (bev->is_pressed() && bev->get_button_index() == MouseButton::RIGHT && presets_enabled) {
			erase_preset(p_color);
			emit_signal(SNAME("preset_removed"), p_color);
		}
	}
}

void ColorPicker::_screen_input(const Ref<InputEvent> &p_event) {
	if (!is_inside_tree()) {
		return;
	}

	Ref<InputEventMouseButton> bev = p_event;
	if (bev.is_valid() && bev->get_button_index() == MouseButton::LEFT && !bev->is_pressed()) {
		emit_signal(SNAME("color_changed"), color);
		screen->hide();
	}

	Ref<InputEventMouseMotion> mev = p_event;
	if (mev.is_valid()) {
		Viewport *r = get_tree()->get_root();
		if (!r->get_visible_rect().has_point(mev->get_global_position())) {
			return;
		}

		Ref<Image> img = r->get_texture()->get_image();
		if (img.is_valid() && !img->is_empty()) {
			Vector2 ofs = mev->get_global_position() - r->get_visible_rect().get_position();
			Color c = img->get_pixel(ofs.x, r->get_visible_rect().size.height - ofs.y);

			set_pick_color(c);
		}
	}
}

void ColorPicker::_add_preset_pressed() {
	add_preset(color);
	emit_signal(SNAME("preset_added"), color);
}

void ColorPicker::_screen_pick_pressed() {
	if (!is_inside_tree()) {
		return;
	}

	Viewport *r = get_tree()->get_root();
	if (!screen) {
		screen = memnew(Control);
		r->add_child(screen);
		screen->set_as_top_level(true);
		screen->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
		screen->set_default_cursor_shape(CURSOR_POINTING_HAND);
		screen->connect("gui_input", callable_mp(this, &ColorPicker::_screen_input));
		// It immediately toggles off in the first press otherwise.
		screen->call_deferred(SNAME("connect"), "hidden", Callable(btn_pick, "set_pressed"), varray(false));
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
	if (c_text->is_menu_visible()) {
		return;
	}
	_html_submitted(c_text->get_text());
	_focus_exit();
}

void ColorPicker::set_presets_enabled(bool p_enabled) {
	presets_enabled = p_enabled;
	if (!p_enabled) {
		btn_add_preset->set_disabled(true);
		btn_add_preset->set_focus_mode(FOCUS_NONE);
	} else {
		btn_add_preset->set_disabled(false);
		btn_add_preset->set_focus_mode(FOCUS_ALL);
	}
}

bool ColorPicker::are_presets_enabled() const {
	return presets_enabled;
}

void ColorPicker::set_presets_visible(bool p_visible) {
	presets_visible = p_visible;
	preset_separator->set_visible(p_visible);
	preset_container->set_visible(p_visible);
}

bool ColorPicker::are_presets_visible() const {
	return presets_visible;
}

void ColorPicker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_pick_color", "color"), &ColorPicker::set_pick_color);
	ClassDB::bind_method(D_METHOD("get_pick_color"), &ColorPicker::get_pick_color);
	ClassDB::bind_method(D_METHOD("set_hsv_mode", "enabled"), &ColorPicker::set_hsv_mode);
	ClassDB::bind_method(D_METHOD("is_hsv_mode"), &ColorPicker::is_hsv_mode);
	ClassDB::bind_method(D_METHOD("set_raw_mode", "enabled"), &ColorPicker::set_raw_mode);
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
	ClassDB::bind_method(D_METHOD("set_picker_shape", "picker"), &ColorPicker::set_picker_shape);
	ClassDB::bind_method(D_METHOD("get_picker_shape"), &ColorPicker::get_picker_shape);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_pick_color", "get_pick_color");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "edit_alpha"), "set_edit_alpha", "is_editing_alpha");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hsv_mode"), "set_hsv_mode", "is_hsv_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "raw_mode"), "set_raw_mode", "is_raw_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deferred_mode"), "set_deferred_mode", "is_deferred_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "picker_shape", PROPERTY_HINT_ENUM, "HSV Rectangle,HSV Rectangle Wheel,VHS Circle"), "set_picker_shape", "get_picker_shape");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "presets_enabled"), "set_presets_enabled", "are_presets_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "presets_visible"), "set_presets_visible", "are_presets_visible");

	ADD_SIGNAL(MethodInfo("color_changed", PropertyInfo(Variant::COLOR, "color")));
	ADD_SIGNAL(MethodInfo("preset_added", PropertyInfo(Variant::COLOR, "color")));
	ADD_SIGNAL(MethodInfo("preset_removed", PropertyInfo(Variant::COLOR, "color")));

	BIND_ENUM_CONSTANT(SHAPE_HSV_RECTANGLE);
	BIND_ENUM_CONSTANT(SHAPE_HSV_WHEEL);
	BIND_ENUM_CONSTANT(SHAPE_VHS_CIRCLE);
}

ColorPicker::ColorPicker() :
		BoxContainer(true) {
	HBoxContainer *hb_edit = memnew(HBoxContainer);
	add_child(hb_edit, false, INTERNAL_MODE_FRONT);
	hb_edit->set_v_size_flags(SIZE_EXPAND_FILL);

	hb_edit->add_child(uv_edit);
	uv_edit->connect("gui_input", callable_mp(this, &ColorPicker::_uv_input), make_binds(uv_edit));
	uv_edit->set_mouse_filter(MOUSE_FILTER_PASS);
	uv_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	uv_edit->set_v_size_flags(SIZE_EXPAND_FILL);
	uv_edit->connect("draw", callable_mp(this, &ColorPicker::_hsv_draw), make_binds(0, uv_edit));

	HBoxContainer *hb_smpl = memnew(HBoxContainer);
	add_child(hb_smpl, false, INTERNAL_MODE_FRONT);

	hb_smpl->add_child(sample);
	sample->set_h_size_flags(SIZE_EXPAND_FILL);
	sample->connect("gui_input", callable_mp(this, &ColorPicker::_sample_input));
	sample->connect("draw", callable_mp(this, &ColorPicker::_sample_draw));

	btn_pick->set_flat(true);
	hb_smpl->add_child(btn_pick);
	btn_pick->set_toggle_mode(true);
	btn_pick->set_tooltip(RTR("Pick a color from the editor window."));
	btn_pick->connect("pressed", callable_mp(this, &ColorPicker::_screen_pick_pressed));

	VBoxContainer *vbl = memnew(VBoxContainer);
	add_child(vbl, false, INTERNAL_MODE_FRONT);

	add_child(memnew(HSeparator), false, INTERNAL_MODE_FRONT);

	VBoxContainer *vbr = memnew(VBoxContainer);
	add_child(vbr, false, INTERNAL_MODE_FRONT);
	vbr->set_h_size_flags(SIZE_EXPAND_FILL);

	for (int i = 0; i < 4; i++) {
		HBoxContainer *hbc = memnew(HBoxContainer);

		labels[i] = memnew(Label());
		labels[i]->set_custom_minimum_size(Size2(get_theme_constant(SNAME("label_width")), 0));
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
		scroll[i]->connect("draw", callable_mp(this, &ColorPicker::_slider_draw), make_binds(i));

		vbr->add_child(hbc);
	}

	labels[3]->set_text("A");

	HBoxContainer *hhb = memnew(HBoxContainer);
	vbr->add_child(hhb);

	hhb->add_child(btn_hsv);
	btn_hsv->set_text(RTR("HSV"));
	btn_hsv->connect("toggled", callable_mp(this, &ColorPicker::set_hsv_mode));

	hhb->add_child(btn_raw);
	btn_raw->set_text(RTR("Raw"));
	btn_raw->connect("toggled", callable_mp(this, &ColorPicker::set_raw_mode));

	hhb->add_child(text_type);
	text_type->set_text("#");
	text_type->set_tooltip(TTR("Switch between hexadecimal and code values."));
	if (Engine::get_singleton()->is_editor_hint()) {
		text_type->connect("pressed", callable_mp(this, &ColorPicker::_text_type_toggled));
	} else {
		text_type->set_flat(true);
		text_type->set_mouse_filter(MOUSE_FILTER_IGNORE);
	}

	hhb->add_child(c_text);
	c_text->set_h_size_flags(SIZE_EXPAND_FILL);
	c_text->connect("text_submitted", callable_mp(this, &ColorPicker::_html_submitted));
	c_text->connect("focus_entered", callable_mp(this, &ColorPicker::_focus_enter));
	c_text->connect("focus_exited", callable_mp(this, &ColorPicker::_html_focus_exit));

	wheel_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	wheel_edit->set_v_size_flags(SIZE_EXPAND_FILL);
	hb_edit->add_child(wheel_edit);

	wheel_mat.instantiate();
	wheel_mat->set_shader(wheel_shader);
	circle_mat.instantiate();
	circle_mat->set_shader(circle_shader);

	wheel_margin->add_theme_constant_override("margin_bottom", 8);
	wheel_edit->add_child(wheel_margin);

	wheel_margin->add_child(wheel);
	wheel->set_mouse_filter(MOUSE_FILTER_PASS);
	wheel->connect("draw", callable_mp(this, &ColorPicker::_hsv_draw), make_binds(2, wheel));

	wheel_margin->add_child(wheel_uv);
	wheel_uv->connect("gui_input", callable_mp(this, &ColorPicker::_uv_input), make_binds(wheel_uv));
	wheel_uv->connect("draw", callable_mp(this, &ColorPicker::_hsv_draw), make_binds(0, wheel_uv));

	hb_edit->add_child(w_edit);
	w_edit->set_h_size_flags(SIZE_FILL);
	w_edit->set_v_size_flags(SIZE_EXPAND_FILL);
	w_edit->connect("gui_input", callable_mp(this, &ColorPicker::_w_input));
	w_edit->connect("draw", callable_mp(this, &ColorPicker::_hsv_draw), make_binds(1, w_edit));

	picker_type = SHAPE_HSV_RECTANGLE;
	_update_controls();
	updating = false;

	set_pick_color(Color(1, 1, 1));

	add_child(preset_separator, false, INTERNAL_MODE_FRONT);

	preset_container->set_h_size_flags(SIZE_EXPAND_FILL);
	preset_container->set_columns(preset_column_count);
	add_child(preset_container, false, INTERNAL_MODE_FRONT);

	btn_add_preset->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	btn_add_preset->set_tooltip(RTR("Add current color as a preset."));
	btn_add_preset->connect("pressed", callable_mp(this, &ColorPicker::_add_preset_pressed));
	preset_container->add_child(btn_add_preset);
}

/////////////////

void ColorPickerButton::_about_to_popup() {
	set_pressed(true);
	if (picker) {
		picker->set_old_color(color);
	}
}

void ColorPickerButton::_color_changed(const Color &p_color) {
	color = p_color;
	update();
	emit_signal(SNAME("color_changed"), color);
}

void ColorPickerButton::_modal_closed() {
	emit_signal(SNAME("popup_closed"));
	set_pressed(false);
}

void ColorPickerButton::pressed() {
	_update_picker();

	Size2 size = get_size() * get_viewport()->get_canvas_transform().get_scale();

	popup->set_as_minsize();
	picker->_update_presets();

	Rect2i usable_rect = popup->get_usable_parent_rect();
	//let's try different positions to see which one we can use

	Rect2i cp_rect(Point2i(), popup->get_size());
	for (int i = 0; i < 4; i++) {
		if (i > 1) {
			cp_rect.position.y = get_screen_position().y - cp_rect.size.y;
		} else {
			cp_rect.position.y = get_screen_position().y + size.height;
		}

		if (i & 1) {
			cp_rect.position.x = get_screen_position().x;
		} else {
			cp_rect.position.x = get_screen_position().x - MAX(0, (cp_rect.size.x - size.x));
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
			const Ref<StyleBox> normal = get_theme_stylebox(SNAME("normal"));
			const Rect2 r = Rect2(normal->get_offset(), get_size() - normal->get_minimum_size());
			draw_texture_rect(Control::get_theme_icon(SNAME("bg"), SNAME("ColorPickerButton")), r, true);
			draw_rect(r, color);

			if (color.r > 1 || color.g > 1 || color.b > 1) {
				// Draw an indicator to denote that the color is "overbright" and can't be displayed accurately in the preview
				draw_texture(Control::get_theme_icon(SNAME("overbright_indicator"), SNAME("ColorPicker")), normal->get_offset());
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
		picker->set_anchors_and_offsets_preset(PRESET_WIDE);
		popup->add_child(picker);
		add_child(popup, false, INTERNAL_MODE_FRONT);
		picker->connect("color_changed", callable_mp(this, &ColorPickerButton::_color_changed));
		popup->connect("about_to_popup", callable_mp(this, &ColorPickerButton::_about_to_popup));
		popup->connect("popup_hide", callable_mp(this, &ColorPickerButton::_modal_closed));
		picker->set_pick_color(color);
		picker->set_edit_alpha(edit_alpha);
		picker->set_display_old_color(true);
		emit_signal(SNAME("picker_created"));
	}
}

void ColorPickerButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_pick_color", "color"), &ColorPickerButton::set_pick_color);
	ClassDB::bind_method(D_METHOD("get_pick_color"), &ColorPickerButton::get_pick_color);
	ClassDB::bind_method(D_METHOD("get_picker"), &ColorPickerButton::get_picker);
	ClassDB::bind_method(D_METHOD("get_popup"), &ColorPickerButton::get_popup);
	ClassDB::bind_method(D_METHOD("set_edit_alpha", "show"), &ColorPickerButton::set_edit_alpha);
	ClassDB::bind_method(D_METHOD("is_editing_alpha"), &ColorPickerButton::is_editing_alpha);
	ClassDB::bind_method(D_METHOD("_about_to_popup"), &ColorPickerButton::_about_to_popup);

	ADD_SIGNAL(MethodInfo("color_changed", PropertyInfo(Variant::COLOR, "color")));
	ADD_SIGNAL(MethodInfo("popup_closed"));
	ADD_SIGNAL(MethodInfo("picker_created"));
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_pick_color", "get_pick_color");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "edit_alpha"), "set_edit_alpha", "is_editing_alpha");
}

ColorPickerButton::ColorPickerButton() {
	set_toggle_mode(true);
}

/////////////////

void ColorPresetButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			const Rect2 r = Rect2(Point2(0, 0), get_size());
			Ref<StyleBox> sb_raw = get_theme_stylebox(SNAME("preset_fg"), SNAME("ColorPresetButton"))->duplicate();
			Ref<StyleBoxFlat> sb_flat = sb_raw;
			Ref<StyleBoxTexture> sb_texture = sb_raw;

			if (sb_flat.is_valid()) {
				if (preset_color.a < 1) {
					// Draw a background pattern when the color is transparent.
					sb_flat->set_bg_color(Color(1, 1, 1));
					sb_flat->draw(get_canvas_item(), r);

					Rect2 bg_texture_rect = r.grow_side(SIDE_LEFT, -sb_flat->get_margin(SIDE_LEFT));
					bg_texture_rect = bg_texture_rect.grow_side(SIDE_RIGHT, -sb_flat->get_margin(SIDE_RIGHT));
					bg_texture_rect = bg_texture_rect.grow_side(SIDE_TOP, -sb_flat->get_margin(SIDE_TOP));
					bg_texture_rect = bg_texture_rect.grow_side(SIDE_BOTTOM, -sb_flat->get_margin(SIDE_BOTTOM));

					draw_texture_rect(get_theme_icon(SNAME("preset_bg"), SNAME("ColorPresetButton")), bg_texture_rect, true);
					sb_flat->set_bg_color(preset_color);
				}
				sb_flat->set_bg_color(preset_color);
				sb_flat->draw(get_canvas_item(), r);
			} else if (sb_texture.is_valid()) {
				if (preset_color.a < 1) {
					// Draw a background pattern when the color is transparent.
					bool use_tile_texture = (sb_texture->get_h_axis_stretch_mode() == StyleBoxTexture::AxisStretchMode::AXIS_STRETCH_MODE_TILE) || (sb_texture->get_h_axis_stretch_mode() == StyleBoxTexture::AxisStretchMode::AXIS_STRETCH_MODE_TILE_FIT);
					draw_texture_rect(get_theme_icon(SNAME("preset_bg"), SNAME("ColorPresetButton")), r, use_tile_texture);
				}
				sb_texture->set_modulate(preset_color);
				sb_texture->draw(get_canvas_item(), r);
			} else {
				WARN_PRINT("Unsupported StyleBox used for ColorPresetButton. Use StyleBoxFlat or StyleBoxTexture instead.");
			}
			if (preset_color.r > 1 || preset_color.g > 1 || preset_color.b > 1) {
				// Draw an indicator to denote that the color is "overbright" and can't be displayed accurately in the preview
				draw_texture(Control::get_theme_icon(SNAME("overbright_indicator"), SNAME("ColorPresetButton")), Vector2(0, 0));
			}

		} break;
	}
}

void ColorPresetButton::set_preset_color(const Color &p_color) {
	preset_color = p_color;
}

Color ColorPresetButton::get_preset_color() const {
	return preset_color;
}

ColorPresetButton::ColorPresetButton(Color p_color) {
	preset_color = p_color;
}

ColorPresetButton::~ColorPresetButton() {
}

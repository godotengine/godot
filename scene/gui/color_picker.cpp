/**************************************************************************/
/*  color_picker.cpp                                                      */
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

#include "color_picker.h"

#include "core/io/image.h"
#include "core/math/expression.h"
#include "scene/gui/color_mode.h"
#include "scene/gui/color_picker_shape.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/link_button.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/atlas_texture.h"
#include "scene/resources/color_palette.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/style_box_texture.h"
#include "scene/theme/theme_db.h"

static inline bool is_color_overbright(const Color &color) {
	return (color.r > 1.0) || (color.g > 1.0) || (color.b > 1.0);
}

static inline bool is_color_valid_hex(const Color &color) {
	return !is_color_overbright(color) && color.r >= 0 && color.g >= 0 && color.b >= 0;
}

static inline String color_to_string(const Color &color, bool show_alpha = true, bool force_value_format = false) {
	if (!force_value_format && !is_color_overbright(color)) {
		return "#" + color.to_html(show_alpha);
	}
	String t = "(" + String::num(color.r, 3) + ", " + String::num(color.g, 3) + ", " + String::num(color.b, 3);
	if (show_alpha) {
		t += ", " + String::num(color.a, 3) + ")";
	} else {
		t += ")";
	}
	return t;
}

void ColorPicker::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			DisplayServer::get_singleton()->accessibility_update_set_role(ae, DisplayServer::AccessibilityRole::ROLE_COLOR_PICKER);
			DisplayServer::get_singleton()->accessibility_update_set_color_value(ae, color);
		} break;

		case NOTIFICATION_ENTER_TREE: {
			_update_color();
		} break;

#ifdef MACOS_ENABLED
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible_in_tree()) {
				perm_hb->set_visible(!OS::get_singleton()->get_granted_permissions().has("macos.permission.RECORD_SCREEN"));
			}
		} break;
#endif

		case NOTIFICATION_READY: {
			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_COLOR_PICKER)) {
				btn_pick->set_tooltip_text(ETR("Pick a color from the screen."));
				btn_pick->connect(SceneStringName(pressed), callable_mp(this, &ColorPicker::_pick_button_pressed_native));
			} else if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_SCREEN_CAPTURE) && !get_tree()->get_root()->is_embedding_subwindows()) {
				// FIXME: The embedding check is needed to fix a bug in single-window mode (GH-93718).
				btn_pick->set_tooltip_text(ETR("Pick a color from the screen."));
				btn_pick->connect(SceneStringName(pressed), callable_mp(this, &ColorPicker::_pick_button_pressed));
			} else {
				// On unsupported platforms, use a legacy method for color picking.
				btn_pick->set_tooltip_text(ETR("Pick a color from the application window."));
				btn_pick->connect(SceneStringName(pressed), callable_mp(this, &ColorPicker::_pick_button_pressed_legacy));
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			btn_pick->set_button_icon(theme_cache.screen_picker);
			_update_drop_down_arrow(btn_preset->is_pressed(), btn_preset);
			_update_drop_down_arrow(btn_recent_preset->is_pressed(), btn_recent_preset);
			btn_add_preset->set_button_icon(theme_cache.add_preset);
			menu_btn->set_button_icon(theme_cache.menu_option);
			btn_mode->set_button_icon(theme_cache.menu_option);

			btn_pick->set_custom_minimum_size(Size2(28 * theme_cache.base_scale, 0));
			btn_shape->set_custom_minimum_size(Size2(28 * theme_cache.base_scale, 0));
			btn_mode->set_custom_minimum_size(Size2(28 * theme_cache.base_scale, 0));

			{
				int i = 0;
				for (ColorPickerShape *shape : shapes) {
					if (shape->is_initialized) {
						shape->update_theme();
						for (Control *c : shape->controls) {
							c->queue_redraw();
						}
					}
					shape_popup->set_item_icon(i, shape->get_icon());
					i++;
				}
			}

			if (current_shape != SHAPE_NONE) {
				btn_shape->set_button_icon(shape_popup->get_item_icon(get_current_shape_index()));
			}

			for (int i = 0; i < MODE_SLIDER_COUNT; i++) {
				labels[i]->set_custom_minimum_size(Size2(theme_cache.label_width, 0));
				sliders[i]->add_theme_constant_override(SNAME("center_grabber"), theme_cache.center_slider_grabbers);
			}
			alpha_label->set_custom_minimum_size(Size2(theme_cache.label_width, 0));
			alpha_slider->add_theme_constant_override(SNAME("center_grabber"), theme_cache.center_slider_grabbers);
			intensity_label->set_custom_minimum_size(Size2(theme_cache.label_width, 0));

			for (int i = 0; i < MODE_BUTTON_COUNT; i++) {
				mode_btns[i]->begin_bulk_theme_override();
				mode_btns[i]->add_theme_style_override(SceneStringName(pressed), theme_cache.mode_button_pressed);
				mode_btns[i]->add_theme_style_override(CoreStringName(normal), theme_cache.mode_button_normal);
				mode_btns[i]->add_theme_style_override(SceneStringName(hover), theme_cache.mode_button_hover);
				mode_btns[i]->end_bulk_theme_override();
			}

			internal_margin->begin_bulk_theme_override();
			internal_margin->add_theme_constant_override(SNAME("margin_bottom"), theme_cache.content_margin);
			internal_margin->add_theme_constant_override(SNAME("margin_left"), theme_cache.content_margin);
			internal_margin->add_theme_constant_override(SNAME("margin_right"), theme_cache.content_margin);
			internal_margin->add_theme_constant_override(SNAME("margin_top"), theme_cache.content_margin);
			internal_margin->end_bulk_theme_override();

			_reset_sliders_theme();

			hex_label->set_custom_minimum_size(Size2(38 * theme_cache.base_scale, 0));
			// Adjust for the width of the "script" icon.
			text_type->set_custom_minimum_size(Size2(28 * theme_cache.base_scale, 0));

			_update_controls();
			// HACK: Deferring updating presets to ensure their size is correct when creating ColorPicker at runtime.
			callable_mp(this, &ColorPicker::_update_presets).call_deferred();
			callable_mp(this, &ColorPicker::_update_recent_presets).call_deferred();
		} break;

		case NOTIFICATION_WM_CLOSE_REQUEST: {
			if (picker_window != nullptr && picker_window->is_visible()) {
				picker_window->hide();
			}
		} break;

		case NOTIFICATION_FOCUS_ENTER:
		case NOTIFICATION_FOCUS_EXIT: {
			if (current_shape != SHAPE_NONE) {
				shapes[get_current_shape_index()]->cursor_editing = false;
			}
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (!is_picking_color) {
				Input *input = Input::get_singleton();
				if (input->is_action_just_released("ui_left") ||
						input->is_action_just_released("ui_right") ||
						input->is_action_just_released("ui_up") ||
						input->is_action_just_released("ui_down")) {
					gamepad_event_delay_ms = DEFAULT_GAMEPAD_EVENT_DELAY_MS;
					if (current_shape == SHAPE_NONE) {
						shapes[get_current_shape_index()]->echo_multiplier = 1;
					}
					accept_event();
					set_process_internal(false);
					return;
				}

				if (current_shape == SHAPE_NONE) {
					return;
				}

				gamepad_event_delay_ms -= get_process_delta_time();
				if (gamepad_event_delay_ms <= 0) {
					gamepad_event_delay_ms = GAMEPAD_EVENT_REPEAT_RATE_MS + gamepad_event_delay_ms;
					// Treat any input from joypad axis as -1, 0, or 1, as the value is added to Vector2i and would be lost.
					Vector2 color_change_vector = Vector2(
							input->is_action_pressed("ui_right") - input->is_action_pressed("ui_left"),
							input->is_action_pressed("ui_down") - input->is_action_pressed("ui_up"));

					shapes[get_current_shape_index()]->update_cursor(color_change_vector, true);
					accept_event();
				}
				return;
			}
			DisplayServer *ds = DisplayServer::get_singleton();
			Vector2 ofs = ds->mouse_get_position();

			Color c = DisplayServer::get_singleton()->screen_get_pixel(ofs);

			picker_preview_style_box_color->set_bg_color(c);
			picker_preview_style_box->set_bg_color(c.get_luminance() < 0.5 ? Color(1.0f, 1.0f, 1.0f) : Color(0.0f, 0.0f, 0.0f));

			if (ds->has_feature(DisplayServer::FEATURE_SCREEN_EXCLUDE_FROM_CAPTURE)) {
				Ref<Image> zoom_preview_img = ds->screen_get_image_rect(Rect2i(ofs.x - 8, ofs.y - 8, 17, 17));
				picker_window->set_position(ofs - Vector2(28, 28));
				picker_texture_zoom->set_texture(ImageTexture::create_from_image(zoom_preview_img));
			} else {
				Size2i screen_size = ds->screen_get_size(DisplayServer::SCREEN_WITH_MOUSE_FOCUS);
				Vector2i screen_position = ds->screen_get_position(DisplayServer::SCREEN_WITH_MOUSE_FOCUS);

				float ofs_decal_x = (ofs.x < screen_position.x + screen_size.width - 51) ? 8 : -36;
				float ofs_decal_y = (ofs.y < screen_position.y + screen_size.height - 51) ? 8 : -36;

				picker_window->set_position(ofs + Vector2(ofs_decal_x, ofs_decal_y));
			}

			set_pick_color(c);
		} break;
	}
}

void ColorPicker::_update_theme_item_cache() {
	VBoxContainer::_update_theme_item_cache();

	theme_cache.base_scale = get_theme_default_base_scale();
}

void ColorPicker::set_focus_on_line_edit() {
	callable_mp((Control *)c_text, &Control::grab_focus).call_deferred(false);
}

void ColorPicker::set_focus_on_picker_shape() {
	shapes[get_current_shape_index()]->grab_focus();
}

void ColorPicker::_update_controls() {
	int mode_sliders_count = modes[current_mode]->get_slider_count();

	for (int i = current_slider_count; i < mode_sliders_count; i++) {
		sliders[i]->show();
		labels[i]->show();
		values[i]->show();
	}
	for (int i = mode_sliders_count; i < current_slider_count; i++) {
		sliders[i]->hide();
		labels[i]->hide();
		values[i]->hide();
	}
	current_slider_count = mode_sliders_count;

	for (int i = 0; i < current_slider_count; i++) {
		labels[i]->set_text(modes[current_mode]->get_slider_label(i));
		sliders[i]->set_accessibility_name(modes[current_mode]->get_slider_label(i));
		values[i]->set_accessibility_name(modes[current_mode]->get_slider_label(i));
	}
	alpha_label->set_text("A");
	alpha_slider->set_accessibility_name(ETR("Alpha"));
	alpha_value->set_accessibility_name(ETR("Alpha"));

	intensity_label->set_text("I");
	intensity_slider->set_accessibility_name(ETR("Intensity"));
	intensity_value->set_accessibility_name(ETR("Intensity"));

	alpha_value->set_visible(edit_alpha);
	alpha_slider->set_visible(edit_alpha);
	alpha_label->set_visible(edit_alpha);

	intensity_value->set_visible(edit_intensity);
	intensity_slider->set_visible(edit_intensity);
	intensity_label->set_visible(edit_intensity);

	int i = 0;
	for (ColorPickerShape *shape : shapes) {
		bool is_active = get_current_shape_index() == i;
		i++;

		if (!shape->is_initialized) {
			if (is_active) {
				// Controls are initialized on demand, because ColorPicker does not need them all at once.
				shape->initialize_controls();
			} else {
				continue;
			}
		}

		for (Control *control : shape->controls) {
			control->set_visible(is_active);
		}
	}
	btn_shape->set_visible(current_shape != SHAPE_NONE);
}

void ColorPicker::_set_pick_color(const Color &p_color, bool p_update_sliders, bool p_calc_intensity) {
	if (text_changed) {
		add_recent_preset(color);
		text_changed = false;
	}

	color = p_color;
	if (p_calc_intensity) {
		_copy_color_to_normalized_and_intensity();
	}
	_copy_normalized_to_hsv_okhsl();

	if (!is_inside_tree()) {
		return;
	}

	_update_color(p_update_sliders);
}

void ColorPicker::set_pick_color(const Color &p_color) {
	_set_pick_color(p_color, true, true); // Because setters can't have more arguments.
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
	if (edit_alpha == p_show) {
		return;
	}
	edit_alpha = p_show;
	_update_controls();

	if (!is_inside_tree()) {
		return;
	}

	_update_color();
	sample->queue_redraw();
}

bool ColorPicker::is_editing_alpha() const {
	return edit_alpha;
}

void ColorPicker::set_edit_intensity(bool p_show) {
	if (edit_intensity == p_show) {
		return;
	}
	if (p_show) {
		set_pick_color(color);
	} else {
		_normalized_apply_intensity_to_color();
		color_normalized = color;
		intensity = 0;
	}
	edit_intensity = p_show;
	_update_controls();

	if (!is_inside_tree()) {
		return;
	}

	_update_color();
	sample->queue_redraw();
}

bool ColorPicker::is_editing_intensity() const {
	return edit_intensity;
}

void ColorPicker::_slider_drag_started() {
	currently_dragging = true;
}

void ColorPicker::_slider_value_changed() {
	if (updating) {
		return;
	}

	intensity = intensity_value->get_value();
	color_normalized = modes[current_mode]->get_color();
	if (edit_intensity && is_color_overbright(color_normalized)) {
		modes[current_mode]->_greater_value_inputted();
		color_normalized = modes[current_mode]->get_color();
	}
	_normalized_apply_intensity_to_color();
	intensity_value->set_prefix(intensity < 0 ? "" : "+");

	modes[current_mode]->_value_changed();

	_set_pick_color(color, false, false);
	if (!deferred_mode_enabled || !currently_dragging) {
		emit_signal(SNAME("color_changed"), color);
	}
}

void ColorPicker::_slider_drag_ended() {
	currently_dragging = false;
	if (deferred_mode_enabled) {
		emit_signal(SNAME("color_changed"), color);
	}
}

void ColorPicker::add_mode(ColorMode *p_mode) {
	modes.push_back(p_mode);
}

void ColorPicker::add_shape(ColorPickerShape *p_shape) {
	shapes.push_back(p_shape);
}

void ColorPicker::create_slider(GridContainer *gc, int idx) {
	Label *lbl = memnew(Label);
	lbl->set_v_size_flags(SIZE_SHRINK_CENTER);
	lbl->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	gc->add_child(lbl);

	HSlider *slider = memnew(HSlider);
	slider->set_v_size_flags(SIZE_SHRINK_CENTER);
	slider->set_focus_mode(FOCUS_ACCESSIBILITY);
	gc->add_child(slider);

	SpinBox *val = memnew(SpinBox);
	slider->share(val);
	val->set_select_all_on_focus(true);
	gc->add_child(val);

	LineEdit *vle = val->get_line_edit();
	vle->connect(SceneStringName(text_changed), callable_mp(this, &ColorPicker::_text_changed));
	vle->connect(SceneStringName(gui_input), callable_mp(this, &ColorPicker::_line_edit_input));
	vle->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_RIGHT);

	val->connect(SceneStringName(gui_input), callable_mp(this, &ColorPicker::_slider_or_spin_input));

	slider->set_h_size_flags(SIZE_EXPAND_FILL);

	slider->connect("drag_started", callable_mp(this, &ColorPicker::_slider_drag_started));
	slider->connect(SceneStringName(value_changed), callable_mp(this, &ColorPicker::_slider_value_changed).unbind(1));
	slider->connect("drag_ended", callable_mp(this, &ColorPicker::_slider_drag_ended).unbind(1));
	if (idx < MODE_SLIDER_COUNT) {
		slider->connect(SceneStringName(draw), callable_mp(this, &ColorPicker::_slider_draw).bind(idx));
	} else if (idx == SLIDER_ALPHA) {
		slider->connect(SceneStringName(draw), callable_mp(this, &ColorPicker::_alpha_slider_draw));
	}
	slider->connect(SceneStringName(gui_input), callable_mp(this, &ColorPicker::_slider_or_spin_input));

	if (idx < MODE_SLIDER_COUNT) {
		sliders[idx] = slider;
		values[idx] = val;
		labels[idx] = lbl;
	} else if (idx == SLIDER_INTENSITY) {
		intensity_slider = slider;
		intensity_value = val;
		intensity_label = lbl;
	} else if (idx == SLIDER_ALPHA) {
		alpha_slider = slider;
		alpha_value = val;
		alpha_label = lbl;
	}
}

#ifdef TOOLS_ENABLED
void ColorPicker::set_editor_settings(Object *p_editor_settings) {
	if (editor_settings) {
		return;
	}
	editor_settings = p_editor_settings;

	if (preset_cache.is_empty()) {
		PackedColorArray saved_presets = editor_settings->call(SNAME("get_project_metadata"), "color_picker", "presets", PackedColorArray());
		for (int i = 0; i < saved_presets.size(); i++) {
			preset_cache.push_back(saved_presets[i]);
		}
	}

	for (const Color &preset : preset_cache) {
		presets.push_back(preset);
	}

	if (recent_preset_cache.is_empty()) {
		PackedColorArray saved_recent_presets = editor_settings->call(SNAME("get_project_metadata"), "color_picker", "recent_presets", PackedColorArray());
		for (int i = 0; i < saved_recent_presets.size(); i++) {
			recent_preset_cache.push_back(saved_recent_presets[i]);
		}
	}

	for (const Color &preset : recent_preset_cache) {
		recent_presets.push_back(preset);
	}

	_update_presets();
	_update_recent_presets();
}

void ColorPicker::set_quick_open_callback(const Callable &p_file_selected) {
	quick_open_callback = p_file_selected;
}

void ColorPicker::set_palette_saved_callback(const Callable &p_palette_saved) {
	palette_saved_callback = p_palette_saved;
}

#endif

HSlider *ColorPicker::get_slider(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, MODE_MAX, nullptr);
	return sliders[p_idx];
}

Vector<float> ColorPicker::get_active_slider_values() {
	Vector<float> cur_values;
	for (int i = 0; i < current_slider_count; i++) {
		cur_values.push_back(sliders[i]->get_value());
	}
	cur_values.push_back(alpha_slider->get_value());
	return cur_values;
}

void ColorPicker::_copy_normalized_to_hsv_okhsl() {
	if (!okhsl_cached) {
		ok_hsl_h = color_normalized.get_ok_hsl_h();
		ok_hsl_s = color_normalized.get_ok_hsl_s();
		ok_hsl_l = color_normalized.get_ok_hsl_l();
	}
	if (!hsv_cached) {
		h = color_normalized.get_h();
		s = color_normalized.get_s();
		v = color_normalized.get_v();
	}
	hsv_cached = false;
	okhsl_cached = false;
}

void ColorPicker::_copy_hsv_okhsl_to_normalized() {
	if (current_shape != SHAPE_NONE && shapes[get_current_shape_index()]->is_ok_hsl()) {
		color_normalized.set_ok_hsl(ok_hsl_h, ok_hsl_s, ok_hsl_l, color_normalized.a);
	} else {
		color_normalized.set_hsv(h, s, v, color_normalized.a);
	}
}

Color ColorPicker::_color_apply_intensity(const Color &col) const {
	if (intensity == 0.0f) {
		return col;
	}
	Color linear_color = col.srgb_to_linear();
	Color result;
	float multiplier = Math::pow(2, intensity);
	for (int i = 0; i < 3; i++) {
		result.components[i] = linear_color.components[i] * multiplier;
	}
	result.a = col.a;
	return result.linear_to_srgb();
}

void ColorPicker::_normalized_apply_intensity_to_color() {
	color = _color_apply_intensity(color_normalized);
}

void ColorPicker::_copy_color_to_normalized_and_intensity() {
	Color linear_color = color.srgb_to_linear();
	float multiplier = MAX(1, MAX(MAX(linear_color.r, linear_color.g), linear_color.b));
	for (int i = 0; i < 3; i++) {
		color_normalized.components[i] = linear_color.components[i] / multiplier;
	}
	color_normalized.a = linear_color.a;
	color_normalized = color_normalized.linear_to_srgb();
	intensity = Math::log2(multiplier);
}

void ColorPicker::_select_from_preset_container(const Color &p_color) {
	if (preset_group->get_pressed_button()) {
		preset_group->get_pressed_button()->set_pressed(false);
	}

	for (int i = 1; i < preset_container->get_child_count(); i++) {
		ColorPresetButton *current_btn = Object::cast_to<ColorPresetButton>(preset_container->get_child(i));
		if (current_btn && p_color == current_btn->get_preset_color()) {
			current_btn->set_pressed(true);
			break;
		}
	}
}

bool ColorPicker::_select_from_recent_preset_hbc(const Color &p_color) {
	for (int i = 0; i < recent_preset_hbc->get_child_count(); i++) {
		ColorPresetButton *current_btn = Object::cast_to<ColorPresetButton>(recent_preset_hbc->get_child(i));
		if (current_btn && p_color == current_btn->get_preset_color()) {
			current_btn->set_pressed(true);
			return true;
		}
	}
	return false;
}

void ColorPicker::_reset_sliders_theme() {
	Ref<StyleBoxFlat> style_box_flat(memnew(StyleBoxFlat));
	style_box_flat->set_content_margin(SIDE_TOP, 16 * theme_cache.base_scale);
	style_box_flat->set_bg_color(Color(0.2, 0.23, 0.31).lerp(Color(0, 0, 0, 1), 0.3).clamp());

	for (int i = 0; i < MODE_SLIDER_COUNT; i++) {
		sliders[i]->begin_bulk_theme_override();
		sliders[i]->add_theme_icon_override(SNAME("grabber"), theme_cache.bar_arrow);
		sliders[i]->add_theme_icon_override(SNAME("grabber_highlight"), theme_cache.bar_arrow);
		sliders[i]->add_theme_constant_override(SNAME("grabber_offset"), 8 * theme_cache.base_scale);
		if (!colorize_sliders) {
			sliders[i]->add_theme_style_override(SNAME("slider"), style_box_flat);
		}
		sliders[i]->end_bulk_theme_override();
	}

	alpha_slider->begin_bulk_theme_override();
	alpha_slider->add_theme_icon_override(SNAME("grabber"), theme_cache.bar_arrow);
	alpha_slider->add_theme_icon_override(SNAME("grabber_highlight"), theme_cache.bar_arrow);
	alpha_slider->add_theme_constant_override(SNAME("grabber_offset"), 8 * theme_cache.base_scale);
	if (!colorize_sliders) {
		alpha_slider->add_theme_style_override(SNAME("slider"), style_box_flat);
	}
	alpha_slider->end_bulk_theme_override();
}

void ColorPicker::_html_submitted(const String &p_html) {
	if (updating) {
		return;
	}
	Color new_color = color;
	if (text_is_constructor || !is_color_valid_hex(color)) {
		Ref<Expression> expr;
		expr.instantiate();
		Error err = expr->parse(p_html);
		if (err == OK) {
			Variant result = expr->execute(Array(), nullptr, false, true);
			// This is basically the same as Variant::operator Color(), but remains original color if Color::from_string() fails
			if (result.get_type() == Variant::COLOR) {
				new_color = result;
			} else if (result.get_type() == Variant::STRING) {
				new_color = Color::from_string(result, color);
			} else if (result.get_type() == Variant::INT) {
				new_color = Color::hex(result);
			}
		}
	} else {
		new_color = Color::from_string(p_html.strip_edges(), color);
		String html_no_prefix = p_html.strip_edges().trim_prefix("#");
		if (html_no_prefix.is_valid_hex_number(false)) {
			// Convert invalid HTML color codes that software like Figma supports.
			if (html_no_prefix.length() == 1) {
				// Turn `#1` into `#111111`.
				html_no_prefix = html_no_prefix.repeat(6);
			} else if (html_no_prefix.length() == 2) {
				// Turn `#12` into `#121212`.
				html_no_prefix = html_no_prefix.repeat(3);
			} else if (html_no_prefix.length() == 5) {
				// Turn `#12345` into `#11223344`.
				html_no_prefix = html_no_prefix.left(4);
			} else if (html_no_prefix.length() == 7) {
				// Turn `#1234567` into `#123456`.
				html_no_prefix = html_no_prefix.left(6);
			}
		}
		new_color = Color::from_string(html_no_prefix, new_color);
	}

	if (!is_editing_alpha()) {
		new_color.a = color.a;
	}

	if (new_color == color) {
		return;
	}
	color = new_color;

	if (!is_inside_tree()) {
		return;
	}

	set_pick_color(color);
	emit_signal(SNAME("color_changed"), color);
}

void ColorPicker::_update_color(bool p_update_sliders) {
	updating = true;

	if (p_update_sliders) {
		float step = modes[current_mode]->get_slider_step();
		float spinbox_arrow_step = modes[current_mode]->get_spinbox_arrow_step();
		for (int i = 0; i < current_slider_count; i++) {
			sliders[i]->set_max(modes[current_mode]->get_slider_max(i));
			sliders[i]->set_step(step);
			sliders[i]->set_value(modes[current_mode]->get_slider_value(i));
			values[i]->set_custom_arrow_step(spinbox_arrow_step);
			values[i]->set_allow_greater(modes[current_mode]->get_allow_greater());
		}
		alpha_slider->set_max(modes[current_mode]->get_alpha_slider_max());
		alpha_slider->set_step(step);
		alpha_slider->set_value(modes[current_mode]->get_alpha_slider_value());
		intensity_slider->set_value(intensity);
		intensity_value->set_prefix(intensity < 0 ? "" : "+");
	}

	_update_text_value();

	if (current_shape != SHAPE_NONE) {
		for (Control *control : shapes[get_current_shape_index()]->controls) {
			control->queue_redraw();
		}
	}

	sample->queue_redraw();

	for (int i = 0; i < current_slider_count; i++) {
		sliders[i]->queue_redraw();
	}
	alpha_slider->queue_redraw();
	updating = false;
	queue_accessibility_update();
}

void ColorPicker::_update_presets() {
	int preset_size = _get_preset_size();
	btn_add_preset->set_custom_minimum_size(Size2(preset_size, preset_size));
	// Only update the preset button size if it has changed.
	if (preset_size != prev_preset_size) {
		prev_preset_size = preset_size;
		for (int i = 1; i < preset_container->get_child_count(); i++) {
			ColorPresetButton *cpb = Object::cast_to<ColorPresetButton>(preset_container->get_child(i));
			cpb->set_custom_minimum_size(Size2(preset_size, preset_size));
		}
	}

#ifdef TOOLS_ENABLED
	if (editor_settings) {
		String cached_name = editor_settings->call(SNAME("get_project_metadata"), "color_picker", "palette_name", String());
		palette_path = editor_settings->call(SNAME("get_project_metadata"), "color_picker", "palette_path", String());
		bool palette_edited = editor_settings->call(SNAME("get_project_metadata"), "color_picker", "palette_edited", false);
		if (cached_name.is_empty()) {
			palette_path = String();
			palette_name->hide();
		} else {
			palette_name->set_text(cached_name);
			if (btn_preset->is_pressed() && !presets.is_empty()) {
				palette_name->show();
			}

			if (palette_edited) {
				palette_name->set_text(vformat("%s*", palette_name->get_text().remove_char('*')));
				palette_name->set_tooltip_text(TTRC("The changes to this palette have not been saved to a file."));
			}
		}
	}
#endif

	if (presets_just_loaded || presets.is_empty() || Engine::get_singleton()->is_editor_hint()) {
		// Rebuild swatch color buttons, keeping the add-preset button in the first position.
		for (int i = 1; i < preset_container->get_child_count(); i++) {
			preset_container->get_child(i)->queue_free();
		}

		presets = preset_cache;
		for (const Color &preset : presets) {
			_add_preset_button(preset_size, preset);
		}
		presets_just_loaded = false;
	}

	_notification(NOTIFICATION_VISIBILITY_CHANGED);
}

void ColorPicker::_update_recent_presets() {
#ifdef TOOLS_ENABLED
	if (editor_settings) {
		int recent_preset_count = recent_preset_hbc->get_child_count();
		for (int i = 0; i < recent_preset_count; i++) {
			memdelete(recent_preset_hbc->get_child(0));
		}

		recent_presets.clear();
		for (const Color &preset : recent_preset_cache) {
			recent_presets.push_back(preset);
		}

		int preset_size = _get_preset_size();
		for (const Color &preset : recent_presets) {
			_add_recent_preset_button(preset_size, preset);
		}

		_notification(NOTIFICATION_VISIBILITY_CHANGED);
	}
#endif
}

#ifdef TOOLS_ENABLED
void ColorPicker::_text_type_toggled() {
	text_is_constructor = !text_is_constructor;
	if (text_is_constructor) {
		hex_label->set_text(ETR("Expr"));
		text_type->set_text("");
		text_type->set_button_icon(theme_cache.color_script);

		c_text->set_tooltip_text(RTR("Execute an expression as a color."));
	} else {
		hex_label->set_text(ETR("Hex"));
		text_type->set_text("#");
		text_type->set_button_icon(nullptr);

		c_text->set_tooltip_text(ETR("Enter a hex code (\"#ff0000\") or named color (\"red\")."));
	}
	_update_color();
}
#endif // TOOLS_ENABLED

Color ColorPicker::get_pick_color() const {
	return color;
}

Color ColorPicker::get_old_color() const {
	return old_color;
}

void ColorPicker::set_picker_shape(PickerShapeType p_shape) {
	ERR_FAIL_INDEX(p_shape, SHAPE_MAX);
	if (p_shape == current_shape) {
		return;
	}
	if (current_shape != SHAPE_NONE) {
		shape_popup->set_item_checked(get_current_shape_index(), false);
	}
	if (p_shape != SHAPE_NONE) {
		shape_popup->set_item_checked(shape_to_index(p_shape), true);
		btn_shape->set_button_icon(shape_popup->get_item_icon(shape_to_index(p_shape)));
	}

	current_shape = p_shape;

#ifdef TOOLS_ENABLED
	if (editor_settings) {
		editor_settings->call(SNAME("set_project_metadata"), "color_picker", "picker_shape", current_shape);
	}
#endif

	_copy_normalized_to_hsv_okhsl();
	_update_controls();
	_update_color();
}

ColorPicker::PickerShapeType ColorPicker::get_picker_shape() const {
	return current_shape;
}

inline int ColorPicker::_get_preset_size() {
	return (int(get_minimum_size().width) - (preset_container->get_h_separation() * (PRESET_COLUMN_COUNT - 1))) / PRESET_COLUMN_COUNT;
}

void ColorPicker::_add_preset_button(int p_size, const Color &p_color) {
	ColorPresetButton *btn_preset_new = memnew(ColorPresetButton(p_color, p_size, false));
	SET_DRAG_FORWARDING_GCDU(btn_preset_new, ColorPicker);
	btn_preset_new->set_button_group(preset_group);
	preset_container->add_child(btn_preset_new);
	btn_preset_new->set_pressed(true);
	btn_preset_new->connect(SceneStringName(gui_input), callable_mp(this, &ColorPicker::_preset_input).bind(p_color));
}

void ColorPicker::_add_recent_preset_button(int p_size, const Color &p_color) {
	ColorPresetButton *btn_preset_new = memnew(ColorPresetButton(p_color, p_size, true));
	btn_preset_new->set_button_group(recent_preset_group);
	recent_preset_hbc->add_child(btn_preset_new);
	recent_preset_hbc->move_child(btn_preset_new, 0);
	btn_preset_new->set_pressed(true);
	btn_preset_new->connect(SceneStringName(toggled), callable_mp(this, &ColorPicker::_recent_preset_pressed).bind(btn_preset_new));
}

void ColorPicker::_load_palette() {
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("ColorPalette", &extensions);

	file_dialog->set_title(ETR("Load Color Palette"));
	file_dialog->clear_filters();
	for (const String &K : extensions) {
		file_dialog->add_filter("*." + K);
	}

	file_dialog->set_file_mode(FileDialog::FILE_MODE_OPEN_FILE);
	file_dialog->set_current_file("");
	file_dialog->popup_centered_ratio();
}

void ColorPicker::_save_palette(bool p_is_save_as) {
	if (!p_is_save_as && !palette_path.is_empty()) {
		file_dialog->set_file_mode(FileDialog::FILE_MODE_SAVE_FILE);
		_palette_file_selected(palette_path);
		return;
	} else {
		List<String> extensions;
		ResourceLoader::get_recognized_extensions_for_type("ColorPalette", &extensions);

		file_dialog->set_title(ETR("Save Color Palette"));
		file_dialog->clear_filters();
		for (const String &K : extensions) {
			file_dialog->add_filter("*." + K);
		}

		file_dialog->set_file_mode(FileDialog::FILE_MODE_SAVE_FILE);
		file_dialog->set_current_file("new_palette.tres");
		file_dialog->popup_centered_ratio();
	}
}

#ifdef TOOLS_ENABLED
void ColorPicker::_quick_open_palette_file_selected(const String &p_path) {
	_ensure_file_dialog();
	file_dialog->set_file_mode(FileDialog::FILE_MODE_OPEN_FILE);
	_palette_file_selected(p_path);
}

GridContainer *ColorPicker::get_slider_container() {
	return slider_gc;
}

#endif // ifdef TOOLS_ENABLED

void ColorPicker::_palette_file_selected(const String &p_path) {
	switch (file_dialog->get_file_mode()) {
		case FileDialog::FileMode::FILE_MODE_OPEN_FILE: {
			Ref<ColorPalette> palette = ResourceLoader::load(p_path, "", ResourceFormatLoader::CACHE_MODE_IGNORE);
			ERR_FAIL_COND_MSG(palette.is_null(), vformat("Cannot open color palette file for reading at: %s", p_path));
			preset_cache.clear();
			presets.clear();

			PackedColorArray saved_presets = palette->get_colors();
			for (const Color &saved_preset : saved_presets) {
				preset_cache.push_back(saved_preset);
				presets.push_back(saved_preset);
			}
			presets_just_loaded = true;

#ifdef TOOLS_ENABLED
			if (editor_settings) {
				const StringName set_project_metadata = SNAME("set_project_metadata");
				editor_settings->call(set_project_metadata, "color_picker", "presets", saved_presets);
				editor_settings->call(set_project_metadata, "color_picker", "palette_edited", false);
			}
#endif
		} break;
		case FileDialog::FileMode::FILE_MODE_SAVE_FILE: {
			Ref<ColorPalette> palette;
			palette.instantiate();
			palette->set_colors(get_presets());
			Error error = ResourceSaver::save(palette, p_path);
			ERR_FAIL_COND_MSG(error != Error::OK, vformat("Cannot open color palette file for writing at: %s", p_path));
#ifdef TOOLS_ENABLED
			if (palette_saved_callback.is_valid()) {
				palette_saved_callback.call_deferred(p_path);
			}
#endif // TOOLS_ENABLED
		} break;
		default:
			break;
	}

	palette_name->set_text(p_path.get_file().get_basename());
	palette_name->set_tooltip_text("");
	palette_name->show();
	palette_path = p_path;
	btn_preset->set_pressed(true);
#ifdef TOOLS_ENABLED
	if (editor_settings) {
		editor_settings->call(SNAME("set_project_metadata"), "color_picker", "palette_name", palette_name->get_text());
		editor_settings->call(SNAME("set_project_metadata"), "color_picker", "palette_path", palette_path);
		editor_settings->call(SNAME("set_project_metadata"), "color_picker", "palette_edited", false);
	}
#endif
	if (file_dialog->get_file_mode() == FileDialog::FileMode::FILE_MODE_OPEN_FILE) {
		_update_presets();
	}
}

void ColorPicker::_show_hide_preset(const bool &p_is_btn_pressed, Button *p_btn_preset, Container *p_preset_container) {
	if (p_is_btn_pressed) {
		p_preset_container->show();
	} else {
		p_preset_container->hide();
	}
	_update_drop_down_arrow(p_is_btn_pressed, p_btn_preset);

	palette_name->hide();
	if (btn_preset->is_pressed() && !palette_name->get_text().is_empty()) {
		palette_name->show();
	}
}

void ColorPicker::_update_drop_down_arrow(const bool &p_is_btn_pressed, Button *p_btn_preset) {
	if (p_is_btn_pressed) {
		p_btn_preset->set_button_icon(theme_cache.expanded_arrow);
	} else {
		p_btn_preset->set_button_icon(theme_cache.folded_arrow);
	}
}

void ColorPicker::_set_mode_popup_value(ColorModeType p_mode) {
	ERR_FAIL_INDEX(p_mode, MODE_MAX + 1);

	if (p_mode == MODE_MAX) {
		set_colorize_sliders(!colorize_sliders);
	} else {
		set_color_mode(p_mode);
	}
}

Variant ColorPicker::_get_drag_data_fw(const Point2 &p_point, Control *p_from_control) {
	ColorPresetButton *dragged_preset_button = Object::cast_to<ColorPresetButton>(p_from_control);

	if (!dragged_preset_button) {
		return Variant();
	}

	ColorPresetButton *drag_preview = memnew(ColorPresetButton(dragged_preset_button->get_preset_color(), _get_preset_size(), false));
	set_drag_preview(drag_preview);

	Dictionary drag_data;
	drag_data["type"] = "color_preset";
	drag_data["color_preset"] = dragged_preset_button->get_index();

	return drag_data;
}

bool ColorPicker::_can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from_control) const {
	Dictionary d = p_data;
	if (!d.has("type") || String(d["type"]) != "color_preset") {
		return false;
	}
	return true;
}

void ColorPicker::_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from_control) {
	Dictionary d = p_data;
	if (!d.has("type")) {
		return;
	}

	if (String(d["type"]) == "color_preset") {
		int preset_from_id = d["color_preset"];
		int hover_now = p_from_control->get_index();

		if (preset_from_id == hover_now || hover_now == -1) {
			return;
		}

		{
			List<Color>::Element *from = presets.front();
			List<Color>::Element *to = presets.front();
			{
				int c = 0;
				while (c < preset_from_id - 1) {
					from = from->next();
					c++;
				}
			}
			{
				int c = 0;
				while (c < hover_now - 1) {
					to = to->next();
					c++;
				}
			}
			if (hover_now == presets.size()) {
				presets.move_to_back(from);
			} else if (hover_now == 1) {
				presets.move_to_front(from);
			} else {
				presets.move_before(from, to->next());
			}
		}

		{
			List<Color>::Element *from = preset_cache.front();
			List<Color>::Element *to = preset_cache.front();
			{
				int c = 0;
				while (c < preset_from_id - 1) {
					from = from->next();
					c++;
				}
			}
			{
				int c = 0;
				while (c < hover_now - 1) {
					to = to->next();
					c++;
				}
			}
			if (hover_now == presets.size()) {
				preset_cache.move_to_back(from);
			} else if (hover_now == 1) {
				preset_cache.move_to_front(from);
			} else {
				preset_cache.move_before(from, to->next());
			}
		}

		preset_container->move_child(preset_container->get_child(preset_from_id), hover_now);
	}
}

void ColorPicker::_ensure_file_dialog() {
	if (file_dialog) {
		return;
	}

	file_dialog = memnew(FileDialog);
	file_dialog->set_mode_overrides_title(false);
	file_dialog->set_access(FileDialog::ACCESS_FILESYSTEM);
	file_dialog->set_current_dir(Engine::get_singleton()->is_editor_hint() ? "res://" : "user://");
	add_child(file_dialog, false, INTERNAL_MODE_FRONT);
	file_dialog->connect("file_selected", callable_mp(this, &ColorPicker::_palette_file_selected));
}

void ColorPicker::add_preset(const Color &p_color) {
	List<Color>::Element *e = presets.find(p_color);
	if (e) {
		presets.move_to_back(e);

		for (int i = 1; i < preset_container->get_child_count(); i++) {
			ColorPresetButton *current_btn = Object::cast_to<ColorPresetButton>(preset_container->get_child(i));
			if (current_btn && p_color == current_btn->get_preset_color()) {
				preset_container->move_child(current_btn, preset_container->get_child_count() - 1);
				current_btn->set_pressed(true);
				break;
			}
		}
	} else {
		presets.push_back(p_color);

		_add_preset_button(_get_preset_size(), p_color);
	}

	List<Color>::Element *cache_e = preset_cache.find(p_color);
	if (cache_e) {
		preset_cache.move_to_back(cache_e);
	} else {
		preset_cache.push_back(p_color);
	}

	if (!palette_name->get_text().is_empty()) {
		palette_name->set_text(vformat("%s*", palette_name->get_text().trim_suffix("*")));
		palette_name->set_tooltip_text(ETR("The changes to this palette have not been saved to a file."));
	}

#ifdef TOOLS_ENABLED
	if (editor_settings) {
		PackedColorArray arr_to_save = get_presets();
		const StringName set_project_metadata = SNAME("set_project_metadata");
		editor_settings->call(set_project_metadata, "color_picker", "presets", arr_to_save);
		editor_settings->call(set_project_metadata, "color_picker", "palette_edited", true);
	}
#endif
}

void ColorPicker::add_recent_preset(const Color &p_color) {
	if (!_select_from_recent_preset_hbc(p_color)) {
		if (recent_preset_hbc->get_child_count() >= PRESET_COLUMN_COUNT) {
			recent_preset_cache.pop_front();
			recent_presets.pop_front();
			recent_preset_hbc->get_child(PRESET_COLUMN_COUNT - 1)->queue_free();
		}
		recent_presets.push_back(p_color);
		recent_preset_cache.push_back(p_color);
		_add_recent_preset_button(_get_preset_size(), p_color);
	}
	_select_from_preset_container(p_color);

#ifdef TOOLS_ENABLED
	if (editor_settings) {
		PackedColorArray arr_to_save = get_recent_presets();
		editor_settings->call(SNAME("set_project_metadata"), "color_picker", "recent_presets", arr_to_save);
	}
#endif
}

void ColorPicker::erase_preset(const Color &p_color) {
	List<Color>::Element *e = presets.find(p_color);
	if (e) {
		presets.erase(e);
		preset_cache.erase(preset_cache.find(p_color));

		// Find preset button to remove.
		for (int i = 1; i < preset_container->get_child_count(); i++) {
			ColorPresetButton *current_btn = Object::cast_to<ColorPresetButton>(preset_container->get_child(i));
			if (current_btn && p_color == current_btn->get_preset_color()) {
				current_btn->queue_free();
				// Removing focused control loose the focus totally. We focus on previous button to keep it possible to navigate with keyboard/joypad.
				Control *focus_target = Object::cast_to<Control>(preset_container->get_child(i - 1));
				focus_target->grab_focus();
				break;
			}
		}

		palette_name->set_text(vformat("%s*", palette_name->get_text().remove_char('*')));
		palette_name->set_tooltip_text(ETR("The changes to this palette have not been saved to a file."));
		if (presets.is_empty()) {
			palette_name->set_text("");
			palette_path = String();
			palette_name->hide();
		}

#ifdef TOOLS_ENABLED
		if (editor_settings) {
			PackedColorArray arr_to_save = get_presets();
			const StringName set_project_metadata = SNAME("set_project_metadata");
			editor_settings->call(set_project_metadata, "color_picker", "presets", arr_to_save);
			editor_settings->call(set_project_metadata, "color_picker", "palette_edited", true);
			editor_settings->call(set_project_metadata, "color_picker", "palette_name", palette_name->get_text());
			editor_settings->call(set_project_metadata, "color_picker", "palette_path", palette_path);
		}
#endif
	}
}

void ColorPicker::erase_recent_preset(const Color &p_color) {
	List<Color>::Element *e = recent_presets.find(p_color);
	if (e) {
		recent_presets.erase(e);
		recent_preset_cache.erase(recent_preset_cache.find(p_color));

		// Find recent preset button to remove.
		for (int i = 1; i < recent_preset_hbc->get_child_count(); i++) {
			ColorPresetButton *current_btn = Object::cast_to<ColorPresetButton>(recent_preset_hbc->get_child(i));
			if (current_btn && p_color == current_btn->get_preset_color()) {
				current_btn->queue_free();
				break;
			}
		}

#ifdef TOOLS_ENABLED
		if (editor_settings) {
			PackedColorArray arr_to_save = get_recent_presets();
			editor_settings->call(SNAME("set_project_metadata"), "color_picker", "recent_presets", arr_to_save);
		}
#endif
	}
}

PackedColorArray ColorPicker::get_presets() const {
	PackedColorArray arr;
	arr.resize(presets.size());
	int i = 0;
	for (List<Color>::ConstIterator itr = presets.begin(); itr != presets.end(); ++itr, ++i) {
		arr.set(i, *itr);
	}
	return arr;
}

PackedColorArray ColorPicker::get_recent_presets() const {
	PackedColorArray arr;
	arr.resize(recent_presets.size());
	int i = 0;
	for (List<Color>::ConstIterator itr = recent_presets.begin(); itr != recent_presets.end(); ++itr, ++i) {
		arr.set(i, *itr);
	}
	return arr;
}

void ColorPicker::set_color_mode(ColorModeType p_mode) {
	ERR_FAIL_INDEX(p_mode, MODE_MAX);

	if (current_mode == p_mode) {
		return;
	}

	mode_popup->set_item_checked(current_mode, false);
	mode_popup->set_item_checked(p_mode, true);

	if (p_mode < MODE_BUTTON_COUNT) {
		mode_btns[p_mode]->set_pressed(true);
	} else if (current_mode < MODE_BUTTON_COUNT) {
		mode_btns[current_mode]->set_pressed(false);
	}

	current_mode = p_mode;

#ifdef TOOLS_ENABLED
	if (editor_settings) {
		editor_settings->call(SNAME("set_project_metadata"), "color_picker", "color_mode", current_mode);
	}
#endif

	if (!is_inside_tree()) {
		return;
	}

	_update_controls();
	_update_color();
}

ColorPicker::ColorModeType ColorPicker::get_color_mode() const {
	return current_mode;
}

void ColorPicker::set_colorize_sliders(bool p_colorize_sliders) {
	if (colorize_sliders == p_colorize_sliders) {
		return;
	}

	colorize_sliders = p_colorize_sliders;
	mode_popup->set_item_checked(MODE_MAX + 1, colorize_sliders);

	if (colorize_sliders) {
		Ref<StyleBoxEmpty> style_box_empty(memnew(StyleBoxEmpty));

		for (int i = 0; i < MODE_SLIDER_COUNT; i++) {
			sliders[i]->add_theme_style_override("slider", style_box_empty);
		}

		alpha_slider->add_theme_style_override("slider", style_box_empty);
	} else {
		Ref<StyleBoxFlat> style_box_flat(memnew(StyleBoxFlat));
		style_box_flat->set_content_margin(SIDE_TOP, 16 * theme_cache.base_scale);
		style_box_flat->set_bg_color(Color(0.2, 0.23, 0.31).lerp(Color(0, 0, 0, 1), 0.3).clamp());

		for (int i = 0; i < MODE_SLIDER_COUNT; i++) {
			sliders[i]->add_theme_style_override("slider", style_box_flat);
		}

		alpha_slider->add_theme_style_override("slider", style_box_flat);
	}
}

bool ColorPicker::is_colorizing_sliders() const {
	return colorize_sliders;
}

void ColorPicker::set_deferred_mode(bool p_enabled) {
	deferred_mode_enabled = p_enabled;
}

bool ColorPicker::is_deferred_mode() const {
	return deferred_mode_enabled;
}

void ColorPicker::_update_text_value() {
	if (text_is_constructor || !is_color_valid_hex(color)) {
		String t = "Color" + color_to_string(color, edit_alpha && color.a < 1, true);

		text_type->set_text("");
		text_type->set_button_icon(theme_cache.color_script);
		text_type->set_disabled(!is_color_valid_hex(color));
		hex_label->set_text(ETR("Expr"));
		c_text->set_text(t);
		c_text->set_tooltip_text(RTR("Execute an expression as a color."));
	} else {
		text_type->set_text("#");
		text_type->set_button_icon(nullptr);
		text_type->set_disabled(false);
		hex_label->set_text(ETR("Hex"));
		c_text->set_text(color.to_html(edit_alpha && color.a < 1));
		c_text->set_tooltip_text(ETR("Enter a hex code (\"#ff0000\") or named color (\"red\")."));
	}
}

void ColorPicker::_sample_input(const Ref<InputEvent> &p_event) {
	if (!display_old_color) {
		return;
	}

	const Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		const Rect2 rect_old = Rect2(Point2(), Size2(sample->get_size().width * 0.5, sample->get_size().height * 0.95));
		if (rect_old.has_point(mb->get_position())) {
			// Revert to the old color when left-clicking the old color sample.
			set_pick_color(old_color);

			sample->set_focus_mode(FOCUS_NONE);
			emit_signal(SNAME("color_changed"), color);
		}
	}

	if (p_event->is_action_pressed(SNAME("ui_accept"), false, true)) {
		set_pick_color(old_color);
		emit_signal(SNAME("color_changed"), color);
	}
}

void ColorPicker::_sample_draw() {
	// Covers the right half of the sample if the old color is being displayed,
	// or the whole sample if it's not being displayed.
	Rect2 rect_new;
	Rect2 rect_old;

	if (display_old_color) {
		rect_new = Rect2(Point2(sample->get_size().width * 0.5, 0), Size2(sample->get_size().width * 0.5, sample->get_size().height * 0.95));

		// Draw both old and new colors for easier comparison (only if spawned from a ColorPickerButton).
		rect_old = Rect2(Point2(), Size2(sample->get_size().width * 0.5, sample->get_size().height * 0.95));

		if (old_color.a < 1.0) {
			sample->draw_texture_rect(theme_cache.sample_bg, rect_old, true);
		}

		sample->draw_rect(rect_old, old_color);

		if (!old_color.is_equal_approx(color)) {
			// Draw a revert indicator to indicate that the old sample can be clicked to revert to this old color.
			// Adapt icon color to the background color (taking alpha checkerboard into account) so that it's always visible.
			sample->draw_texture(theme_cache.sample_revert,
					rect_old.size * 0.5 - theme_cache.sample_revert->get_size() * 0.5,
					Math::lerp(0.75f, old_color.get_luminance(), old_color.a) < 0.455 ? Color(1, 1, 1) : (Color(0.01, 0.01, 0.01)));

			sample->set_focus_mode(FOCUS_ALL);
		} else {
			sample->set_focus_mode(FOCUS_NONE);
		}

		if (is_color_overbright(color)) {
			// Draw an indicator to denote that the old color is "overbright" and can't be displayed accurately in the preview.
			sample->draw_texture(theme_cache.overbright_indicator, Point2());
		}
	} else {
		rect_new = Rect2(Point2(), Size2(sample->get_size().width, sample->get_size().height * 0.95));
	}

	if (color.a < 1.0) {
		sample->draw_texture_rect(theme_cache.sample_bg, rect_new, true);
	}

	sample->draw_rect(rect_new, color);

	if (display_old_color && !old_color.is_equal_approx(color) && sample->has_focus(true)) {
		RID ci = sample->get_canvas_item();
		theme_cache.sample_focus->draw(ci, rect_old);
	}

	if (is_color_overbright(color)) {
		// Draw an indicator to denote that the new color is "overbright" and can't be displayed accurately in the preview.
		sample->draw_texture(theme_cache.overbright_indicator, Point2(sample->get_size().width * 0.5, 0));
	}
}

void ColorPicker::_slider_draw(int p_which) {
	if (colorize_sliders) {
		modes[current_mode]->slider_draw(p_which);
	}
}

void ColorPicker::_alpha_slider_draw() {
	if (!colorize_sliders) {
		return;
	}
	Vector<Vector2> pos;
	pos.resize(4);
	Vector<Color> col;
	col.resize(4);
	Size2 size = alpha_slider->get_size();
	Color left_color;
	Color right_color;
	const real_t margin = 16 * theme_cache.base_scale;
	alpha_slider->draw_texture_rect(theme_cache.sample_bg, Rect2(Point2(0, 0), Size2(size.x, margin)), true);

	left_color = color_normalized;
	left_color.a = 0;
	right_color = color_normalized;
	right_color.a = 1;

	col.set(0, left_color);
	col.set(1, right_color);
	col.set(2, right_color);
	col.set(3, left_color);
	pos.set(0, Vector2(0, 0));
	pos.set(1, Vector2(size.x, 0));
	pos.set(2, Vector2(size.x, margin));
	pos.set(3, Vector2(0, margin));

	alpha_slider->draw_polygon(pos, col);
}

void ColorPicker::_slider_or_spin_input(const Ref<InputEvent> &p_event) {
	if (line_edit_mouse_release) {
		line_edit_mouse_release = false;
		return;
	}
	Ref<InputEventMouseButton> bev = p_event;
	if (bev.is_valid() && !bev->is_pressed() && bev->get_button_index() == MouseButton::LEFT) {
		add_recent_preset(color);
	}
}

void ColorPicker::_line_edit_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> bev = p_event;
	if (bev.is_valid() && !bev->is_pressed() && bev->get_button_index() == MouseButton::LEFT) {
		line_edit_mouse_release = true;
	}
}

void ColorPicker::_preset_input(const Ref<InputEvent> &p_event, const Color &p_color) {
	Ref<InputEventMouseButton> bev = p_event;

	if (bev.is_valid()) {
		if (bev->is_pressed() && bev->get_button_index() == MouseButton::LEFT) {
			set_pick_color(p_color);
			add_recent_preset(color);
			emit_signal(SNAME("color_changed"), p_color);
		} else if (bev->is_pressed() && bev->get_button_index() == MouseButton::RIGHT && can_add_swatches) {
			erase_preset(p_color);
			emit_signal(SNAME("preset_removed"), p_color);
		}
	}

	if (p_event->is_action_pressed(SNAME("ui_accept"), false, true)) {
		set_pick_color(p_color);
		add_recent_preset(color);
		emit_signal(SNAME("color_changed"), p_color);
	} else if (p_event->is_action_pressed(SNAME("ui_colorpicker_delete_preset"), false, true) && can_add_swatches) {
		erase_preset(p_color);
		emit_signal(SNAME("preset_removed"), p_color);
	}
}

void ColorPicker::_recent_preset_pressed(const bool p_pressed, ColorPresetButton *p_preset) {
	if (!p_pressed) {
		return;
	}

	// Avoid applying and recalculating the intensity for non-overbright color if it doesn't change.
	if (color != p_preset->get_preset_color()) {
		set_pick_color(p_preset->get_preset_color());
	}

	recent_presets.move_to_back(recent_presets.find(p_preset->get_preset_color()));
	List<Color>::Element *e = recent_preset_cache.find(p_preset->get_preset_color());
	if (e) {
		recent_preset_cache.move_to_back(e);
	}

	recent_preset_hbc->move_child(p_preset, 0);
	emit_signal(SNAME("color_changed"), p_preset->get_preset_color());
}

void ColorPicker::_text_changed(const String &) {
	text_changed = true;
}

void ColorPicker::_add_preset_pressed() {
	add_preset(color);
	emit_signal(SNAME("preset_added"), color);
}

void ColorPicker::_pick_button_pressed_native() {
	if (!DisplayServer::get_singleton()->color_picker(callable_mp(this, &ColorPicker::_native_cb))) {
		// Fallback to default/legacy picker.
		if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_SCREEN_CAPTURE) && !get_tree()->get_root()->is_embedding_subwindows()) {
			_pick_button_pressed();
		} else {
			_pick_button_pressed_legacy();
		}
	}
}

void ColorPicker::_native_cb(bool p_status, const Color &p_color) {
	if (p_status) {
		set_pick_color(p_color);
		if (!deferred_mode_enabled) {
			emit_signal(SNAME("color_changed"), color);
		}
	}
}

void ColorPicker::_pick_button_pressed() {
	is_picking_color = true;
	pre_picking_color = color;

	if (!picker_window) {
		picker_window = memnew(Popup);
		bool has_feature_exclude_from_capture = DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_SCREEN_EXCLUDE_FROM_CAPTURE);
		if (!has_feature_exclude_from_capture) {
			picker_window->set_size(Vector2i(28, 28));
		} else {
			picker_window->set_size(Vector2i(55, 72));
			picker_window->set_flag(Window::FLAG_EXCLUDE_FROM_CAPTURE, true); // Only supported on MacOS and Windows.
		}
		picker_window->connect(SceneStringName(visibility_changed), callable_mp(this, &ColorPicker::_pick_finished));
		picker_window->connect(SceneStringName(window_input), callable_mp(this, &ColorPicker::_target_gui_input));

		picker_preview = memnew(Panel);
		picker_preview->set_mouse_filter(MOUSE_FILTER_IGNORE);
		picker_preview->set_size(Vector2i(55, 72));
		picker_window->add_child(picker_preview);

		picker_preview_color = memnew(Panel);
		picker_preview_color->set_mouse_filter(MOUSE_FILTER_IGNORE);
		if (!has_feature_exclude_from_capture) {
			picker_preview_color->set_size(Vector2i(24, 24));
			picker_preview_color->set_position(Vector2i(2, 2));
		} else {
			picker_preview_color->set_size(Vector2i(51, 15));
			picker_preview_color->set_position(Vector2i(2, 55));
		}
		picker_preview->add_child(picker_preview_color);

		if (has_feature_exclude_from_capture) {
			picker_texture_zoom = memnew(TextureRect);
			picker_texture_zoom->set_mouse_filter(MOUSE_FILTER_IGNORE);
			picker_texture_zoom->set_custom_minimum_size(Vector2i(51, 51));
			picker_texture_zoom->set_position(Vector2i(2, 2));
			picker_texture_zoom->set_texture_filter(CanvasItem::TEXTURE_FILTER_NEAREST);
			picker_preview->add_child(picker_texture_zoom);
		}

		picker_preview_style_box.instantiate();
		picker_preview->add_theme_style_override(SceneStringName(panel), picker_preview_style_box);

		picker_preview_style_box_color.instantiate();
		picker_preview_color->add_theme_style_override(SceneStringName(panel), picker_preview_style_box_color);

		add_child(picker_window, false, INTERNAL_MODE_FRONT);
	}
	set_process_internal(true);

	picker_window->popup();
}

void ColorPicker::_target_gui_input(const Ref<InputEvent> &p_event) {
	const Ref<InputEventMouseButton> mouse_event = p_event;
	if (mouse_event.is_null()) {
		return;
	}
	if (mouse_event->get_button_index() == MouseButton::LEFT) {
		if (mouse_event->is_pressed()) {
			picker_window->hide();
			_pick_finished();
		}
	} else if (mouse_event->get_button_index() == MouseButton::RIGHT) {
		set_pick_color(pre_picking_color); // Cancel.
		is_picking_color = false;
		set_process_internal(false);
		picker_window->hide();
	} else {
		Window *w = picker_window->get_parent_visible_window();
		while (w) {
			Point2i win_mpos = w->get_mouse_position(); // Mouse position local to the window.
			Size2i win_size = w->get_size();
			if (win_mpos.x >= 0 && win_mpos.y >= 0 && win_mpos.x <= win_size.x && win_mpos.y <= win_size.y) {
				// Mouse event inside window bounds, forward this event to the window.
				Ref<InputEventMouseButton> new_ev = p_event->duplicate();
				new_ev->set_position(win_mpos);
				new_ev->set_global_position(win_mpos);
				w->push_input(new_ev, true);
				return;
			}
			w = w->get_parent_visible_window();
		}
	}
}

void ColorPicker::_pick_finished() {
	if (picker_window->is_visible()) {
		return;
	}

	if (Input::get_singleton()->is_action_just_pressed(SNAME("ui_cancel"))) {
		set_pick_color(pre_picking_color);
	} else {
		emit_signal(SNAME("color_changed"), color);
	}
	is_picking_color = false;
	set_process_internal(false);
	picker_window->hide();
}

void ColorPicker::_update_menu_items() {
	options_menu->clear();
	options_menu->reset_size();

	options_menu->add_icon_item(get_theme_icon(SNAME("save"), SNAME("FileDialog")), ETR("Save"), static_cast<int>(MenuOption::MENU_SAVE));
	options_menu->set_item_tooltip(-1, ETR("Save the current color palette to reuse later."));
	options_menu->set_item_disabled(-1, presets.is_empty());

	options_menu->add_icon_item(get_theme_icon(SNAME("save"), SNAME("FileDialog")), ETR("Save As"), static_cast<int>(MenuOption::MENU_SAVE_AS));
	options_menu->set_item_tooltip(-1, ETR("Save the current color palette as a new to reuse later."));
	options_menu->set_item_disabled(-1, palette_path.is_empty());

	options_menu->add_icon_item(get_theme_icon(SNAME("load"), SNAME("FileDialog")), ETR("Load"), static_cast<int>(MenuOption::MENU_LOAD));
	options_menu->set_item_tooltip(-1, ETR("Load existing color palette."));

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		options_menu->add_icon_item(get_theme_icon(SNAME("load"), SNAME("FileDialog")), TTRC("Quick Load"), static_cast<int>(MenuOption::MENU_QUICKLOAD));
		options_menu->set_item_tooltip(-1, TTRC("Load existing color palette."));
	}
#endif // TOOLS_ENABLED

	options_menu->add_icon_item(get_theme_icon(SNAME("clear"), SNAME("FileDialog")), ETR("Clear"), static_cast<int>(MenuOption::MENU_CLEAR));
	options_menu->set_item_tooltip(-1, ETR("Clear the currently loaded color palettes in the picker."));
	options_menu->set_item_disabled(-1, presets.is_empty());
}

void ColorPicker::_options_menu_cbk(int p_which) {
	_ensure_file_dialog();

	MenuOption option = static_cast<MenuOption>(p_which);
	switch (option) {
		case MenuOption::MENU_SAVE:
			_save_palette(false);
			break;
		case MenuOption::MENU_SAVE_AS:
			_save_palette(true);
			break;
		case MenuOption::MENU_LOAD:
			_load_palette();
			break;

#ifdef TOOLS_ENABLED
		case MenuOption::MENU_QUICKLOAD:
			if (quick_open_callback.is_valid()) {
				file_dialog->set_file_mode(FileDialog::FILE_MODE_OPEN_FILE);
				quick_open_callback.call_deferred();
			}
			break;
#endif // TOOLS_ENABLED
		case MenuOption::MENU_CLEAR: {
			PackedColorArray colors = get_presets();
			for (Color c : colors) {
				erase_preset(c);
			}

			palette_name->set_text("");
			palette_name->set_tooltip_text("");
			palette_path = String();
			btn_preset->set_pressed(false);

#ifdef TOOLS_ENABLED
			if (editor_settings) {
				editor_settings->call(SNAME("set_project_metadata"), "color_picker", "palette_name", palette_name->get_text());
				editor_settings->call(SNAME("set_project_metadata"), "color_picker", "palette_path", palette_path);
				editor_settings->call(SNAME("set_project_metadata"), "color_picker", "palette_edited", false);
			}
#endif // TOOLS_ENABLED

		}

		break;
		default:
			break;
	}
}

void ColorPicker::_block_input_on_popup_show() {
	if (!get_tree()->get_root()->is_embedding_subwindows()) {
		get_viewport()->set_disable_input(true);
	}
}

void ColorPicker::_enable_input_on_popup_hide() {
	if (!get_tree()->get_root()->is_embedding_subwindows()) {
		get_viewport()->set_disable_input(false);
	}
}

void ColorPicker::_pick_button_pressed_legacy() {
	if (!is_inside_tree()) {
		return;
	}
	pre_picking_color = color;

	if (!picker_window) {
		picker_window = memnew(Popup);
		picker_window->hide();
		picker_window->set_transient(true);
		add_child(picker_window, false, INTERNAL_MODE_FRONT);

		picker_texture_rect = memnew(TextureRect);
		picker_texture_rect->set_anchors_preset(Control::PRESET_FULL_RECT);
		picker_texture_rect->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
		picker_texture_rect->set_default_cursor_shape(Control::CURSOR_CROSS);
		picker_window->add_child(picker_texture_rect);
		picker_texture_rect->connect(SceneStringName(gui_input), callable_mp(this, &ColorPicker::_picker_texture_input));

		picker_preview = memnew(Panel);
		picker_preview->set_mouse_filter(MOUSE_FILTER_IGNORE);
		picker_preview->set_size(Vector2i(55, 72));
		picker_window->add_child(picker_preview);

		picker_preview_color = memnew(Panel);
		picker_preview_color->set_mouse_filter(MOUSE_FILTER_IGNORE);
		picker_preview_color->set_size(Vector2i(51, 15));
		picker_preview_color->set_position(Vector2i(2, 55));
		picker_preview->add_child(picker_preview_color);

		picker_texture_zoom = memnew(TextureRect);
		picker_texture_zoom->set_mouse_filter(MOUSE_FILTER_IGNORE);
		picker_texture_zoom->set_custom_minimum_size(Vector2i(51, 51));
		picker_texture_zoom->set_position(Vector2i(2, 2));
		picker_texture_zoom->set_texture_filter(CanvasItem::TEXTURE_FILTER_NEAREST);
		picker_preview->add_child(picker_texture_zoom);

		picker_preview_style_box.instantiate();
		picker_preview->add_theme_style_override(SceneStringName(panel), picker_preview_style_box);

		picker_preview_style_box_color.instantiate();
		picker_preview_color->add_theme_style_override(SceneStringName(panel), picker_preview_style_box_color);
	}

	Rect2i screen_rect;
	if (picker_window->is_embedded()) {
		Ref<ImageTexture> tx = ImageTexture::create_from_image(picker_window->get_embedder()->get_texture()->get_image());
		screen_rect = picker_window->get_embedder()->get_visible_rect();
		picker_window->set_position(Point2i());
		picker_texture_rect->set_texture(tx);

		Vector2 ofs = picker_window->get_mouse_position();
		picker_preview->set_position(ofs - Vector2(28, 28));

		Vector2 scale = screen_rect.size / tx->get_image()->get_size();
		ofs /= scale;

		Ref<AtlasTexture> atlas;
		atlas.instantiate();
		atlas->set_atlas(tx);
		atlas->set_region(Rect2i(ofs.x - 8, ofs.y - 8, 17, 17));
		picker_texture_zoom->set_texture(atlas);
	} else {
		screen_rect = picker_window->get_parent_rect();
		picker_window->set_position(screen_rect.position);

		Ref<Image> target_image = Image::create_empty(screen_rect.size.x, screen_rect.size.y, false, Image::FORMAT_RGB8);
		DisplayServer *ds = DisplayServer::get_singleton();

		// Add the Texture of each Window to the Image.
		Vector<DisplayServer::WindowID> wl = ds->get_window_list();
		// FIXME: sort windows by visibility.
		for (const DisplayServer::WindowID &window_id : wl) {
			Window *w = Window::get_from_id(window_id);
			if (!w) {
				continue;
			}

			Ref<Image> img = w->get_texture()->get_image();
			if (img.is_null() || img->is_empty()) {
				continue;
			}
			img->convert(Image::FORMAT_RGB8);
			target_image->blit_rect(img, Rect2i(Point2i(0, 0), img->get_size()), w->get_position());
		}

		Ref<ImageTexture> tx = ImageTexture::create_from_image(target_image);
		picker_texture_rect->set_texture(tx);

		Vector2 ofs = screen_rect.position - DisplayServer::get_singleton()->mouse_get_position();
		picker_preview->set_position(ofs - Vector2(28, 28));

		Ref<AtlasTexture> atlas;
		atlas.instantiate();
		atlas->set_atlas(tx);
		atlas->set_region(Rect2i(ofs.x - 8, ofs.y - 8, 17, 17));
		picker_texture_zoom->set_texture(atlas);
	}

	picker_window->set_size(screen_rect.size);
	picker_window->popup();
}

void ColorPicker::_picker_texture_input(const Ref<InputEvent> &p_event) {
	if (!is_inside_tree()) {
		return;
	}

	Ref<InputEventMouseButton> bev = p_event;
	if (bev.is_valid() && bev->get_button_index() == MouseButton::LEFT && !bev->is_pressed()) {
		set_pick_color(picker_color);
		emit_signal(SNAME("color_changed"), color);
		picker_window->hide();
	}

	Ref<InputEventMouseMotion> mev = p_event;
	if (mev.is_valid()) {
		Ref<Image> img = picker_texture_rect->get_texture()->get_image();
		if (img.is_valid() && !img->is_empty()) {
			Vector2 ofs = mev->get_position();
			picker_preview->set_position(ofs - Vector2(28, 28));
			Vector2 scale = picker_texture_rect->get_size() / img->get_size();
			ofs /= scale;
			picker_color = img->get_pixel(ofs.x, ofs.y);
			picker_preview_style_box_color->set_bg_color(picker_color);
			picker_preview_style_box->set_bg_color(picker_color.get_luminance() < 0.5 ? Color(1.0f, 1.0f, 1.0f) : Color(0.0f, 0.0f, 0.0f));

			Ref<AtlasTexture> atlas = picker_texture_zoom->get_texture();
			if (atlas.is_valid()) {
				atlas->set_region(Rect2i(ofs.x - 8, ofs.y - 8, 17, 17));
			}
		}
	}
}

void ColorPicker::_html_focus_exit() {
	if (c_text->is_menu_visible()) {
		return;
	}

	if (is_visible_in_tree()) {
		_html_submitted(c_text->get_text());
	} else {
		_update_text_value();
	}
}

void ColorPicker::set_can_add_swatches(bool p_enabled) {
	if (can_add_swatches == p_enabled) {
		return;
	}
	can_add_swatches = p_enabled;
	if (!p_enabled) {
		btn_add_preset->set_disabled(true);
		btn_add_preset->set_focus_mode(FOCUS_NONE);
	} else {
		btn_add_preset->set_disabled(false);
		btn_add_preset->set_focus_mode(FOCUS_ALL);
	}
}

bool ColorPicker::are_swatches_enabled() const {
	return can_add_swatches;
}

void ColorPicker::set_presets_visible(bool p_visible) {
	if (presets_visible == p_visible) {
		return;
	}
	presets_visible = p_visible;
	swatches_vbc->set_visible(p_visible);
}

bool ColorPicker::are_presets_visible() const {
	return presets_visible;
}

void ColorPicker::set_modes_visible(bool p_visible) {
	if (color_modes_visible == p_visible) {
		return;
	}
	color_modes_visible = p_visible;
	mode_hbc->set_visible(p_visible);
}

bool ColorPicker::are_modes_visible() const {
	return color_modes_visible;
}

void ColorPicker::set_sampler_visible(bool p_visible) {
	if (sampler_visible == p_visible) {
		return;
	}
	sampler_visible = p_visible;
	sample_hbc->set_visible(p_visible);
}

bool ColorPicker::is_sampler_visible() const {
	return sampler_visible;
}

void ColorPicker::set_sliders_visible(bool p_visible) {
	if (sliders_visible == p_visible) {
		return;
	}
	sliders_visible = p_visible;
	slider_gc->set_visible(p_visible);
}

bool ColorPicker::are_sliders_visible() const {
	return sliders_visible;
}

void ColorPicker::set_hex_visible(bool p_visible) {
	if (hex_visible == p_visible) {
		return;
	}
	hex_visible = p_visible;
	hex_hbc->set_visible(p_visible);
}

bool ColorPicker::is_hex_visible() const {
	return hex_visible;
}

void ColorPicker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_pick_color", "color"), &ColorPicker::set_pick_color);
	ClassDB::bind_method(D_METHOD("get_pick_color"), &ColorPicker::get_pick_color);
	ClassDB::bind_method(D_METHOD("set_deferred_mode", "mode"), &ColorPicker::set_deferred_mode);
	ClassDB::bind_method(D_METHOD("is_deferred_mode"), &ColorPicker::is_deferred_mode);
	ClassDB::bind_method(D_METHOD("set_color_mode", "color_mode"), &ColorPicker::set_color_mode);
	ClassDB::bind_method(D_METHOD("get_color_mode"), &ColorPicker::get_color_mode);
	ClassDB::bind_method(D_METHOD("set_edit_alpha", "show"), &ColorPicker::set_edit_alpha);
	ClassDB::bind_method(D_METHOD("is_editing_alpha"), &ColorPicker::is_editing_alpha);
	ClassDB::bind_method(D_METHOD("set_edit_intensity", "show"), &ColorPicker::set_edit_intensity);
	ClassDB::bind_method(D_METHOD("is_editing_intensity"), &ColorPicker::is_editing_intensity);
	ClassDB::bind_method(D_METHOD("set_can_add_swatches", "enabled"), &ColorPicker::set_can_add_swatches);
	ClassDB::bind_method(D_METHOD("are_swatches_enabled"), &ColorPicker::are_swatches_enabled);
	ClassDB::bind_method(D_METHOD("set_presets_visible", "visible"), &ColorPicker::set_presets_visible);
	ClassDB::bind_method(D_METHOD("are_presets_visible"), &ColorPicker::are_presets_visible);
	ClassDB::bind_method(D_METHOD("set_modes_visible", "visible"), &ColorPicker::set_modes_visible);
	ClassDB::bind_method(D_METHOD("are_modes_visible"), &ColorPicker::are_modes_visible);
	ClassDB::bind_method(D_METHOD("set_sampler_visible", "visible"), &ColorPicker::set_sampler_visible);
	ClassDB::bind_method(D_METHOD("is_sampler_visible"), &ColorPicker::is_sampler_visible);
	ClassDB::bind_method(D_METHOD("set_sliders_visible", "visible"), &ColorPicker::set_sliders_visible);
	ClassDB::bind_method(D_METHOD("are_sliders_visible"), &ColorPicker::are_sliders_visible);
	ClassDB::bind_method(D_METHOD("set_hex_visible", "visible"), &ColorPicker::set_hex_visible);
	ClassDB::bind_method(D_METHOD("is_hex_visible"), &ColorPicker::is_hex_visible);
	ClassDB::bind_method(D_METHOD("add_preset", "color"), &ColorPicker::add_preset);
	ClassDB::bind_method(D_METHOD("erase_preset", "color"), &ColorPicker::erase_preset);
	ClassDB::bind_method(D_METHOD("get_presets"), &ColorPicker::get_presets);
	ClassDB::bind_method(D_METHOD("add_recent_preset", "color"), &ColorPicker::add_recent_preset);
	ClassDB::bind_method(D_METHOD("erase_recent_preset", "color"), &ColorPicker::erase_recent_preset);
	ClassDB::bind_method(D_METHOD("get_recent_presets"), &ColorPicker::get_recent_presets);
	ClassDB::bind_method(D_METHOD("set_picker_shape", "shape"), &ColorPicker::set_picker_shape);
	ClassDB::bind_method(D_METHOD("get_picker_shape"), &ColorPicker::get_picker_shape);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_pick_color", "get_pick_color");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "edit_alpha"), "set_edit_alpha", "is_editing_alpha");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "edit_intensity"), "set_edit_intensity", "is_editing_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "color_mode", PROPERTY_HINT_ENUM, "RGB,HSV,LINEAR,OKHSL"), "set_color_mode", "get_color_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deferred_mode"), "set_deferred_mode", "is_deferred_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "picker_shape", PROPERTY_HINT_ENUM, "HSV Rectangle,HSV Rectangle Wheel,VHS Circle,OKHSL Circle,OK HS Rectangle:5,OK HL Rectangle,None:4"), "set_picker_shape", "get_picker_shape");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "can_add_swatches"), "set_can_add_swatches", "are_swatches_enabled");
	ADD_GROUP("Customization", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sampler_visible"), "set_sampler_visible", "is_sampler_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "color_modes_visible"), "set_modes_visible", "are_modes_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sliders_visible"), "set_sliders_visible", "are_sliders_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hex_visible"), "set_hex_visible", "is_hex_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "presets_visible"), "set_presets_visible", "are_presets_visible");

	ADD_SIGNAL(MethodInfo("color_changed", PropertyInfo(Variant::COLOR, "color")));
	ADD_SIGNAL(MethodInfo("preset_added", PropertyInfo(Variant::COLOR, "color")));
	ADD_SIGNAL(MethodInfo("preset_removed", PropertyInfo(Variant::COLOR, "color")));

	BIND_ENUM_CONSTANT(MODE_RGB);
	BIND_ENUM_CONSTANT(MODE_HSV);
#ifndef DISABLE_DEPRECATED
	BIND_ENUM_CONSTANT(MODE_RAW);
#endif
	BIND_ENUM_CONSTANT(MODE_LINEAR);
	BIND_ENUM_CONSTANT(MODE_OKHSL);

	BIND_ENUM_CONSTANT(SHAPE_HSV_RECTANGLE);
	BIND_ENUM_CONSTANT(SHAPE_HSV_WHEEL);
	BIND_ENUM_CONSTANT(SHAPE_VHS_CIRCLE);
	BIND_ENUM_CONSTANT(SHAPE_OKHSL_CIRCLE);
	BIND_ENUM_CONSTANT(SHAPE_NONE);
	BIND_ENUM_CONSTANT(SHAPE_OK_HS_RECTANGLE);
	BIND_ENUM_CONSTANT(SHAPE_OK_HL_RECTANGLE);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_CONSTANT, ColorPicker, content_margin, "margin");
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, ColorPicker, label_width);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, ColorPicker, sv_width);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, ColorPicker, sv_height);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, ColorPicker, h_width);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, ColorPicker, center_slider_grabbers);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, ColorPicker, sample_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, ColorPicker, picker_focus_rectangle);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, ColorPicker, picker_focus_circle);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, ColorPicker, focused_not_editing_cursor_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, menu_option);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, screen_picker);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, expanded_arrow);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, folded_arrow);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, add_preset);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, shape_rect);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, shape_rect_wheel);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, shape_circle);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, bar_arrow);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, sample_bg);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, sample_revert);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, overbright_indicator);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, picker_cursor);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, picker_cursor_bg);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, color_hue);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPicker, color_script);

	BIND_THEME_ITEM_EXT(Theme::DATA_TYPE_STYLEBOX, ColorPicker, mode_button_normal, "tab_unselected", "TabContainer");
	BIND_THEME_ITEM_EXT(Theme::DATA_TYPE_STYLEBOX, ColorPicker, mode_button_pressed, "tab_selected", "TabContainer");
	BIND_THEME_ITEM_EXT(Theme::DATA_TYPE_STYLEBOX, ColorPicker, mode_button_hover, "tab_selected", "TabContainer");

	ADD_CLASS_DEPENDENCY("LineEdit");
	ADD_CLASS_DEPENDENCY("MenuButton");
	ADD_CLASS_DEPENDENCY("PopupMenu");
}

ColorPicker::ColorPicker() {
	internal_margin = memnew(MarginContainer);
	add_child(internal_margin, false, INTERNAL_MODE_FRONT);

	VBoxContainer *real_vbox = memnew(VBoxContainer);
	internal_margin->add_child(real_vbox);

	shape_container = memnew(HBoxContainer);
	shape_container->set_alignment(ALIGNMENT_CENTER);
	real_vbox->add_child(shape_container);

	sample_hbc = memnew(HBoxContainer);
	real_vbox->add_child(sample_hbc);

	btn_pick = memnew(Button);
	btn_pick->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	btn_pick->set_accessibility_name(ETR("Pick"));
	sample_hbc->add_child(btn_pick);

	sample = memnew(TextureRect);
	sample_hbc->add_child(sample);
	sample->set_h_size_flags(SIZE_EXPAND_FILL);
	sample->connect(SceneStringName(gui_input), callable_mp(this, &ColorPicker::_sample_input));
	sample->connect(SceneStringName(draw), callable_mp(this, &ColorPicker::_sample_draw));

	btn_shape = memnew(MenuButton);
	btn_shape->set_flat(false);
	sample_hbc->add_child(btn_shape);
	btn_shape->set_toggle_mode(true);
	btn_shape->set_tooltip_text(ETR("Select a picker shape."));
	btn_shape->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	btn_shape->set_focus_mode(FOCUS_ALL);

	add_shape(memnew(ColorPickerShapeRectangle(this)));
	add_shape(memnew(ColorPickerShapeWheel(this)));
	add_shape(memnew(ColorPickerShapeVHSCircle(this)));
	add_shape(memnew(ColorPickerShapeOKHSLCircle(this)));
	add_shape(memnew(ColorPickerShapeOKHSRectangle(this)));
	add_shape(memnew(ColorPickerShapeOKHLRectangle(this)));

	shape_popup = btn_shape->get_popup();
	{
		int i = 0;
		for (const ColorPickerShape *shape : shapes) {
			shape_popup->add_radio_check_item(shape->get_name(), index_to_shape(i));
			i++;
		}
	}
	shape_popup->set_item_checked(get_current_shape_index(), true);
	shape_popup->connect(SceneStringName(id_pressed), callable_mp(this, &ColorPicker::set_picker_shape));
	shape_popup->connect("about_to_popup", callable_mp(this, &ColorPicker::_block_input_on_popup_show));
	shape_popup->connect(SNAME("popup_hide"), callable_mp(this, &ColorPicker::_enable_input_on_popup_hide));

	add_mode(memnew(ColorModeRGB(this)));
	add_mode(memnew(ColorModeHSV(this)));
	add_mode(memnew(ColorModeLinear(this)));
	add_mode(memnew(ColorModeOKHSL(this)));

	mode_hbc = memnew(HBoxContainer);
	real_vbox->add_child(mode_hbc);

	mode_group.instantiate();

	for (int i = 0; i < MODE_BUTTON_COUNT; i++) {
		mode_btns[i] = memnew(Button);
		mode_hbc->add_child(mode_btns[i]);
		mode_btns[i]->set_focus_mode(FOCUS_ALL);
		mode_btns[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		mode_btns[i]->set_toggle_mode(true);
		mode_btns[i]->set_text(modes[i]->get_name());
		mode_btns[i]->set_button_group(mode_group);
		mode_btns[i]->connect(SceneStringName(pressed), callable_mp(this, &ColorPicker::set_color_mode).bind((ColorModeType)i));
	}
	mode_btns[0]->set_pressed(true);

	btn_mode = memnew(MenuButton);
	btn_mode->set_flat(false);
	mode_hbc->add_child(btn_mode);
	btn_mode->set_toggle_mode(true);
	btn_mode->set_accessibility_name(ETR("Select a picker mode."));
	btn_mode->set_tooltip_text(ETR("Select a picker mode."));
	btn_mode->set_focus_mode(FOCUS_ALL);

	mode_popup = btn_mode->get_popup();
	{
		int i = 0;
		for (const ColorMode *mode : modes) {
			mode_popup->add_radio_check_item(mode->get_name(), i);
			i++;
		}
	}
	mode_popup->add_separator();
	mode_popup->add_check_item(ETR("Colorized Sliders"), MODE_MAX);
	mode_popup->set_item_checked(current_mode, true);
	mode_popup->set_item_checked(MODE_MAX + 1, true);
	mode_popup->connect(SceneStringName(id_pressed), callable_mp(this, &ColorPicker::_set_mode_popup_value));
	mode_popup->connect("about_to_popup", callable_mp(this, &ColorPicker::_block_input_on_popup_show));
	mode_popup->connect(SNAME("popup_hide"), callable_mp(this, &ColorPicker::_enable_input_on_popup_hide));

	slider_gc = memnew(GridContainer);

	real_vbox->add_child(slider_gc);
	slider_gc->set_h_size_flags(SIZE_EXPAND_FILL);
	slider_gc->set_columns(3);

	for (int i = 0; i < SLIDER_MAX; i++) {
		create_slider(slider_gc, i);
	}
	alpha_label->set_text("A");

	intensity_label->set_text("I");
	intensity_slider->set_min(-10);
	intensity_slider->set_max(10);
	intensity_slider->set_step(0.001);
	intensity_value->set_allow_greater(true);
	intensity_value->set_custom_arrow_step(1);
	intensity_value->set_custom_arrow_round(true);

	hex_hbc = memnew(HBoxContainer);
	hex_hbc->set_alignment(ALIGNMENT_BEGIN);
	real_vbox->add_child(hex_hbc);
	hex_label = memnew(Label(ETR("Hex")));
	hex_hbc->add_child(hex_label);

	text_type = memnew(Button);
	hex_hbc->add_child(text_type);
	text_type->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	text_type->set_tooltip_auto_translate_mode(AUTO_TRANSLATE_MODE_ALWAYS);
	text_type->set_text("#");
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		text_type->set_tooltip_text(TTRC("Switch between hexadecimal and code values."));
		text_type->connect(SceneStringName(pressed), callable_mp(this, &ColorPicker::_text_type_toggled));
	} else {
		text_type->set_accessibility_name(ETR("Hexadecimal Values"));
#else
	{
		text_type->set_accessibility_name(ETR("Hexadecimal Values"));
#endif // TOOLS_ENABLED
		text_type->set_flat(true);
	}

	c_text = memnew(LineEdit);
	hex_hbc->add_child(c_text);
	c_text->set_h_size_flags(SIZE_EXPAND_FILL);
	c_text->set_select_all_on_focus(true);
	c_text->set_accessibility_name(ETR("Hex code or named color"));
	c_text->set_tooltip_text(ETR("Enter a hex code (\"#ff0000\") or named color (\"red\")."));
	c_text->set_placeholder(ETR("Hex code or named color"));
	c_text->connect(SceneStringName(text_submitted), callable_mp(this, &ColorPicker::_html_submitted));
	c_text->connect(SceneStringName(text_changed), callable_mp(this, &ColorPicker::_text_changed));
	c_text->connect(SceneStringName(focus_exited), callable_mp(this, &ColorPicker::_html_focus_exit));

	_update_controls();
	updating = false;

	swatches_vbc = memnew(VBoxContainer);
	real_vbox->add_child(swatches_vbc);

	preset_container = memnew(GridContainer);
	preset_container->set_h_size_flags(SIZE_EXPAND_FILL);
	preset_container->set_columns(PRESET_COLUMN_COUNT);
	preset_container->hide();

	preset_group.instantiate();

	HBoxContainer *palette_box = memnew(HBoxContainer);
	palette_box->set_h_size_flags(SIZE_EXPAND_FILL);
	swatches_vbc->add_child(palette_box);

	btn_preset = memnew(Button);
	btn_preset->set_text(ETR("Swatches"));
	btn_preset->set_flat(true);
	btn_preset->set_toggle_mode(true);
	btn_preset->set_focus_mode(FOCUS_ALL);
	btn_preset->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	btn_preset->set_h_size_flags(SIZE_EXPAND_FILL);
	btn_preset->connect(SceneStringName(toggled), callable_mp(this, &ColorPicker::_show_hide_preset).bind(btn_preset, preset_container));
	palette_box->add_child(btn_preset);

	menu_btn = memnew(MenuButton);
	menu_btn->set_flat(false);
	menu_btn->set_focus_mode(FOCUS_ALL);
	menu_btn->set_tooltip_text(ETR("Show all options available."));
	menu_btn->connect("about_to_popup", callable_mp(this, &ColorPicker::_update_menu_items));
	palette_box->add_child(menu_btn);

	options_menu = menu_btn->get_popup();
	options_menu->connect(SceneStringName(id_pressed), callable_mp(this, &ColorPicker::_options_menu_cbk));
	options_menu->connect("about_to_popup", callable_mp(this, &ColorPicker::_block_input_on_popup_show));
	options_menu->connect(SNAME("popup_hide"), callable_mp(this, &ColorPicker::_enable_input_on_popup_hide));

	palette_name = memnew(Label);
	palette_name->hide();
	palette_name->set_mouse_filter(MOUSE_FILTER_PASS);
	swatches_vbc->add_child(palette_name);

	swatches_vbc->add_child(preset_container);

	recent_preset_hbc = memnew(HBoxContainer);
	recent_preset_hbc->set_v_size_flags(SIZE_SHRINK_BEGIN);
	recent_preset_hbc->hide();

	recent_preset_group.instantiate();

	btn_recent_preset = memnew(Button(ETR("Recent Colors")));
	btn_recent_preset->set_flat(true);
	btn_recent_preset->set_toggle_mode(true);
	btn_recent_preset->set_focus_mode(FOCUS_ALL);
	btn_recent_preset->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	btn_recent_preset->connect(SceneStringName(toggled), callable_mp(this, &ColorPicker::_show_hide_preset).bind(btn_recent_preset, recent_preset_hbc));
	swatches_vbc->add_child(btn_recent_preset);

	swatches_vbc->add_child(recent_preset_hbc);

	set_pick_color(Color(1, 1, 1));

	btn_add_preset = memnew(Button);
	btn_add_preset->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	btn_add_preset->set_tooltip_text(ETR("Add current color as a preset."));
	btn_add_preset->connect(SceneStringName(pressed), callable_mp(this, &ColorPicker::_add_preset_pressed));
	preset_container->add_child(btn_add_preset);

	perm_hb = memnew(HBoxContainer);
	perm_hb->set_alignment(BoxContainer::ALIGNMENT_CENTER);

	LinkButton *perm_link = memnew(LinkButton);
	perm_link->set_text(ETR("Screen Recording permission missing!"));
	perm_link->set_tooltip_text(ETR("Screen Recording permission is required to pick colors from the other application windows.\nClick here to request access..."));
	perm_link->connect(SceneStringName(pressed), callable_mp(this, &ColorPicker::_req_permission));
	perm_hb->add_child(perm_link);
	real_vbox->add_child(perm_hb);
	perm_hb->set_visible(false);
}

void ColorPicker::_req_permission() {
#ifdef MACOS_ENABLED
	OS::get_singleton()->request_permission("macos.permission.RECORD_SCREEN");
#endif
}

ColorPicker::~ColorPicker() {
	for (ColorMode *mode : modes) {
		memdelete(mode);
	}
	for (ColorPickerShape *shape : shapes) {
		memdelete(shape);
	}
}

/////////////////

void ColorPickerPopupPanel::_input_from_window(const Ref<InputEvent> &p_event) {
	if (p_event->is_action_pressed(SNAME("ui_accept"), false, true)) {
		_close_pressed();
	}
	PopupPanel::_input_from_window(p_event);
}

/////////////////

void ColorPickerButton::_about_to_popup() {
	if (!get_tree()->get_root()->is_embedding_subwindows()) {
		get_viewport()->set_disable_input(true);
	}
	set_pressed(true);
	if (picker) {
		picker->set_old_color(color);
	}
}

void ColorPickerButton::_color_changed(const Color &p_color) {
	color = p_color;
	queue_accessibility_update();
	queue_redraw();
	emit_signal(SNAME("color_changed"), color);
}

void ColorPickerButton::_modal_closed() {
	if (picker->is_visible_in_tree()) {
		if (Input::get_singleton()->is_action_just_pressed(SNAME("ui_cancel"))) {
			set_pick_color(picker->get_old_color());
			emit_signal(SNAME("color_changed"), color);
		}
		emit_signal(SNAME("popup_closed"));
		set_pressed(false);
	}
	if (!get_tree()->get_root()->is_embedding_subwindows()) {
		get_viewport()->set_disable_input(false);
	}
}

void ColorPickerButton::pressed() {
	_update_picker();

	// Checking if the popup was open before, so we can keep it closed instead of reopening it.
	// Popups get closed when it's clicked outside of them.
	if (popup_was_open) {
		// Reset popup_was_open value.
		popup_was_open = popup->is_visible();
		return;
	}

	Size2 minsize = popup->get_contents_minimum_size();
	float viewport_height = get_viewport_rect().size.y;

	popup->reset_size();
	picker->_update_presets();
	picker->_update_recent_presets();

	// Determine in which direction to show the popup. By default popup horizontally centered below the button.
	// But if the popup doesn't fit below and the button is in the bottom half of the viewport, show above.
	bool show_above = false;
	if (get_global_position().y + get_size().y + minsize.y > viewport_height && get_global_position().y * 2 + get_size().y > viewport_height) {
		show_above = true;
	}

	float h_offset = (get_size().x - minsize.x) / 2;
	float v_offset = show_above ? -minsize.y : get_size().y;
	popup->set_position(get_screen_position() + Vector2(h_offset, v_offset));
	popup->popup();
	if (!picker->is_hex_visible() && picker->get_picker_shape() != ColorPicker::SHAPE_NONE) {
		callable_mp(picker, &ColorPicker::set_focus_on_picker_shape).call_deferred();
	} else if (DisplayServer::get_singleton()->has_hardware_keyboard()) {
		picker->set_focus_on_line_edit();
	}
}

void ColorPickerButton::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseButton> mouse_button = p_event;
	bool ui_accept = p_event->is_action("ui_accept", true) && !p_event->is_echo();
	bool mouse_left_pressed = mouse_button.is_valid() && mouse_button->get_button_index() == MouseButton::LEFT && mouse_button->is_pressed();
	if (mouse_left_pressed || ui_accept) {
		popup_was_open = popup && popup->is_visible();
	}

	BaseButton::gui_input(p_event);
}

void ColorPickerButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			DisplayServer::get_singleton()->accessibility_update_set_role(ae, DisplayServer::AccessibilityRole::ROLE_BUTTON);
			DisplayServer::get_singleton()->accessibility_update_set_popup_type(ae, DisplayServer::AccessibilityPopupType::POPUP_DIALOG);
			DisplayServer::get_singleton()->accessibility_update_set_color_value(ae, color);
		} break;

		case NOTIFICATION_DRAW: {
			const Rect2 r = Rect2(theme_cache.normal_style->get_offset(), get_size() - theme_cache.normal_style->get_minimum_size());
			draw_texture_rect(theme_cache.background_icon, r, true);
			draw_rect(r, color);

			if (color.r > 1 || color.g > 1 || color.b > 1) {
				// Draw an indicator to denote that the color is "overbright" and can't be displayed accurately in the preview
				draw_texture(theme_cache.overbright_indicator, theme_cache.normal_style->get_offset());
			}
		} break;

		case NOTIFICATION_WM_CLOSE_REQUEST: {
			if (popup) {
				popup->hide();
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (popup && !is_visible_in_tree()) {
				popup->hide();
			}
		} break;
	}
}

void ColorPickerButton::set_pick_color(const Color &p_color) {
	if (color == p_color) {
		return;
	}
	color = p_color;
	if (picker) {
		picker->set_pick_color(p_color);
	}
	queue_accessibility_update();
	queue_redraw();
}

Color ColorPickerButton::get_pick_color() const {
	return color;
}

void ColorPickerButton::set_edit_alpha(bool p_show) {
	if (edit_alpha == p_show) {
		return;
	}
	edit_alpha = p_show;
	if (picker) {
		picker->set_edit_alpha(p_show);
	}
}

bool ColorPickerButton::is_editing_alpha() const {
	return edit_alpha;
}

void ColorPickerButton::set_edit_intensity(bool p_show) {
	if (edit_intensity == p_show) {
		return;
	}
	edit_intensity = p_show;
	if (picker) {
		picker->set_edit_intensity(p_show);
	}
}

bool ColorPickerButton::is_editing_intensity() const {
	return edit_intensity;
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
		popup = memnew(ColorPickerPopupPanel);
		popup->set_wrap_controls(true);
		picker = memnew(ColorPicker);
		picker->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
		popup->add_child(picker);
		add_child(popup, false, INTERNAL_MODE_FRONT);
		picker->connect("color_changed", callable_mp(this, &ColorPickerButton::_color_changed));
		popup->connect("about_to_popup", callable_mp(this, &ColorPickerButton::_about_to_popup));
		popup->connect("popup_hide", callable_mp(this, &ColorPickerButton::_modal_closed));
		popup->connect("tree_exiting", callable_mp(this, &ColorPickerButton::_modal_closed));
		picker->connect(SceneStringName(minimum_size_changed), callable_mp((Window *)popup, &Window::reset_size));
		picker->set_pick_color(color);
		picker->set_edit_alpha(edit_alpha);
		picker->set_edit_intensity(edit_intensity);
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
	ClassDB::bind_method(D_METHOD("set_edit_intensity", "show"), &ColorPickerButton::set_edit_intensity);
	ClassDB::bind_method(D_METHOD("is_editing_intensity"), &ColorPickerButton::is_editing_intensity);
	ClassDB::bind_method(D_METHOD("_about_to_popup"), &ColorPickerButton::_about_to_popup);

	ADD_SIGNAL(MethodInfo("color_changed", PropertyInfo(Variant::COLOR, "color")));
	ADD_SIGNAL(MethodInfo("popup_closed"));
	ADD_SIGNAL(MethodInfo("picker_created"));
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_pick_color", "get_pick_color");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "edit_alpha"), "set_edit_alpha", "is_editing_alpha");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "edit_intensity"), "set_edit_intensity", "is_editing_intensity");

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, ColorPickerButton, normal_style, "normal");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, ColorPickerButton, background_icon, "bg");
	BIND_THEME_ITEM_EXT(Theme::DATA_TYPE_ICON, ColorPickerButton, overbright_indicator, "overbright_indicator", "ColorPicker");
}

ColorPickerButton::ColorPickerButton(const String &p_text) :
		Button(p_text) {
	set_toggle_mode(true);
}

/////////////////

void ColorPresetButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			DisplayServer::get_singleton()->accessibility_update_set_role(ae, DisplayServer::AccessibilityRole::ROLE_BUTTON);
			DisplayServer::get_singleton()->accessibility_update_set_color_value(ae, preset_color);
		} break;

		case NOTIFICATION_DRAW: {
			const Rect2 r = Rect2(Point2(0, 0), get_size());
			Ref<StyleBox> sb_raw = theme_cache.foreground_style->duplicate();
			Ref<StyleBoxFlat> sb_flat = sb_raw;
			Ref<StyleBoxTexture> sb_texture = sb_raw;

			if (sb_flat.is_valid()) {
				sb_flat->set_border_width(SIDE_BOTTOM, 2);
				if (get_draw_mode() == DRAW_PRESSED || get_draw_mode() == DRAW_HOVER_PRESSED) {
					sb_flat->set_border_color(Color(1, 1, 1, 1));
				} else {
					sb_flat->set_border_color(Color(0, 0, 0, 1));
				}

				if (preset_color.a < 1) {
					// Draw a background pattern when the color is transparent.
					sb_flat->set_bg_color(Color(1, 1, 1));
					sb_flat->draw(get_canvas_item(), r);

					Rect2 bg_texture_rect = r.grow_side(SIDE_LEFT, -sb_flat->get_margin(SIDE_LEFT));
					bg_texture_rect = bg_texture_rect.grow_side(SIDE_RIGHT, -sb_flat->get_margin(SIDE_RIGHT));
					bg_texture_rect = bg_texture_rect.grow_side(SIDE_TOP, -sb_flat->get_margin(SIDE_TOP));
					bg_texture_rect = bg_texture_rect.grow_side(SIDE_BOTTOM, -sb_flat->get_margin(SIDE_BOTTOM));

					draw_texture_rect(theme_cache.background_icon, bg_texture_rect, true);
					sb_flat->set_bg_color(preset_color);
				}
				sb_flat->set_bg_color(preset_color);
				sb_flat->draw(get_canvas_item(), r);
			} else if (sb_texture.is_valid()) {
				if (preset_color.a < 1) {
					// Draw a background pattern when the color is transparent.
					bool use_tile_texture = (sb_texture->get_h_axis_stretch_mode() == StyleBoxTexture::AxisStretchMode::AXIS_STRETCH_MODE_TILE) || (sb_texture->get_h_axis_stretch_mode() == StyleBoxTexture::AxisStretchMode::AXIS_STRETCH_MODE_TILE_FIT);
					draw_texture_rect(theme_cache.background_icon, r, use_tile_texture);
				}
				sb_texture->set_modulate(preset_color);
				sb_texture->draw(get_canvas_item(), r);
			} else {
				WARN_PRINT("Unsupported StyleBox used for ColorPresetButton. Use StyleBoxFlat or StyleBoxTexture instead.");
			}

			if (has_focus(true)) {
				RID ci = get_canvas_item();
				theme_cache.focus_style->draw(ci, Rect2(Point2(), get_size()));
			}

			if (is_color_overbright(preset_color)) {
				// Draw an indicator to denote that the color is "overbright" and can't be displayed accurately in the preview
				draw_texture(theme_cache.overbright_indicator, Vector2(0, 0));
			}

		} break;
	}
}

void ColorPresetButton::set_preset_color(const Color &p_color) {
	preset_color = p_color;
	queue_accessibility_update();
}

Color ColorPresetButton::get_preset_color() const {
	return preset_color;
}

String ColorPresetButton::get_tooltip(const Point2 &p_pos) const {
	Color color = get_preset_color();
	if (recent) {
		return vformat(atr(ETR("Color: %s\nLMB: Apply color")), color_to_string(color, color.a < 1));
	}
	return vformat(atr(ETR("Color: %s\nLMB: Apply color\nRMB: Remove preset")), color_to_string(color, color.a < 1));
}

void ColorPresetButton::_bind_methods() {
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, ColorPresetButton, foreground_style, "preset_fg");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, ColorPresetButton, focus_style, "preset_focus");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, ColorPresetButton, background_icon, "preset_bg");
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ColorPresetButton, overbright_indicator);
}

ColorPresetButton::ColorPresetButton(Color p_color, int p_size, bool p_recent) {
	preset_color = p_color;
	recent = p_recent;
	set_toggle_mode(true);
	set_custom_minimum_size(Size2(p_size, p_size));
	set_accessibility_name(vformat(atr(ETR("Color: %s")), color_to_string(p_color, p_color.a < 1)));
	set_tooltip_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
}

ColorPresetButton::~ColorPresetButton() {
}

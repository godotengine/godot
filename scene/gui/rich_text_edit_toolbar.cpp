/**************************************************************************/
/*  rich_text_edit_toolbar.cpp                                            */
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

#include "rich_text_edit_toolbar.h"

#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "scene/gui/button.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/rich_text_edit.h"
#include "scene/gui/separator.h"
#include "scene/gui/spin_box.h"
#include "scene/resources/style_box.h"
#include "scene/theme/theme_db.h"

enum {
	ALIGNMENT_LEFT,
	ALIGNMENT_CENTER,
	ALIGNMENT_RIGHT,
};

RichTextEdit *RichTextEditToolbar::_get_rich_text_edit() const {
	return ObjectDB::get_instance<RichTextEdit>(rich_text_edit_id);
}

void RichTextEditToolbar::_resolve_rich_text_edit() {
	if (RichTextEdit *old_target = _get_rich_text_edit()) {
		if (old_target->is_connected("caret_changed", callable_mp(this, &RichTextEditToolbar::_target_caret_changed))) {
			old_target->disconnect("caret_changed", callable_mp(this, &RichTextEditToolbar::_target_caret_changed));
		}
		if (old_target->is_connected(SceneStringName(text_changed), callable_mp(this, &RichTextEditToolbar::_target_caret_changed))) {
			old_target->disconnect(SceneStringName(text_changed), callable_mp(this, &RichTextEditToolbar::_target_caret_changed));
		}
	}

	RichTextEdit *target = nullptr;
	if (!rich_text_edit_path.is_empty() && has_node(rich_text_edit_path)) {
		target = Object::cast_to<RichTextEdit>(get_node(rich_text_edit_path));
	}

	if (target == nullptr && get_parent() != nullptr) {
		for (int i = 0; i < get_parent()->get_child_count(); i++) {
			target = Object::cast_to<RichTextEdit>(get_parent()->get_child(i));
			if (target != nullptr) {
				break;
			}
		}
	}

	rich_text_edit_id = target != nullptr ? target->get_instance_id() : ObjectID();
	if (target != nullptr) {
		if (!target->is_connected("caret_changed", callable_mp(this, &RichTextEditToolbar::_target_caret_changed))) {
			target->connect("caret_changed", callable_mp(this, &RichTextEditToolbar::_target_caret_changed));
		}
		if (!target->is_connected(SceneStringName(text_changed), callable_mp(this, &RichTextEditToolbar::_target_caret_changed))) {
			target->connect(SceneStringName(text_changed), callable_mp(this, &RichTextEditToolbar::_target_caret_changed));
		}
	}
	_update_controls_from_target();
}

void RichTextEditToolbar::_update_controls_from_target() {
	RichTextEdit *target = _get_rich_text_edit();
	if (target == nullptr || font_size_spin == nullptr) {
		return;
	}

	updating_controls = true;
	font_size_spin->set_value(target->get_current_font_size());
	updating_controls = false;
}

void RichTextEditToolbar::_target_caret_changed() {
	_update_controls_from_target();
}

void RichTextEditToolbar::_set_button_icon_or_text(Button *p_button, const StringName &p_icon_name, const String &p_fallback_text) {
	if (p_button == nullptr) {
		return;
	}

	p_button->set_custom_minimum_size(Size2(32, 32));
	p_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	_apply_toolbar_button_style(p_button);
	if (p_button->get_button_icon().is_null()) {
		p_button->set_text(String());
	}
	p_button->add_theme_constant_override("h_separation", 2);
	p_button->set_expand_icon(false);
	p_button->set_clip_text(false);
	p_button->set_text_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	p_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	p_button->set_vertical_icon_alignment(VERTICAL_ALIGNMENT_CENTER);
	p_button->set_text_overrun_behavior(TextServer::OVERRUN_NO_TRIMMING);
	const Color ink = Color::html("1c1c1a");
	p_button->add_theme_color_override(SceneStringName(font_color), ink);
	p_button->add_theme_color_override("font_hover_color", ink);
	p_button->add_theme_color_override("font_pressed_color", ink);
	p_button->add_theme_color_override("icon_normal_color", ink);
	p_button->add_theme_color_override("icon_hover_color", ink);
	p_button->add_theme_color_override("icon_pressed_color", ink);
	if (has_theme_icon(p_icon_name, SNAME("RichTextEditToolbar"))) {
		p_button->set_button_icon(get_theme_icon(p_icon_name, SNAME("RichTextEditToolbar")));
		p_button->set_text(String());
	} else {
		p_button->set_button_icon(Ref<Texture2D>());
		p_button->set_text(p_fallback_text);
	}
}

void RichTextEditToolbar::_apply_toolbar_button_style(Button *p_button) {
	if (p_button == nullptr) {
		return;
	}

	p_button->set_flat(false);
	Ref<StyleBox> normal = get_theme_stylebox(SNAME("button_normal"), SNAME("RichTextEditToolbar"));
	Ref<StyleBox> hover = get_theme_stylebox(SNAME("button_hover"), SNAME("RichTextEditToolbar"));
	Ref<StyleBox> pressed = get_theme_stylebox(SNAME("button_pressed"), SNAME("RichTextEditToolbar"));
	if (normal.is_valid()) {
		p_button->add_theme_style_override(CoreStringName(normal), normal);
		p_button->add_theme_style_override("disabled", normal);
		p_button->add_theme_style_override("focus", normal);
	}
	if (hover.is_valid()) {
		p_button->add_theme_style_override(SceneStringName(hover), hover);
	}
	if (pressed.is_valid()) {
		p_button->add_theme_style_override(SceneStringName(pressed), pressed);
		p_button->add_theme_style_override("hover_pressed", pressed);
	}
}

void RichTextEditToolbar::_apply_toolbar_color_button_style(Button *p_button) {
	if (p_button == nullptr) {
		return;
	}

	p_button->set_flat(false);
	Ref<StyleBox> normal = get_theme_stylebox(SNAME("color_button_normal"), SNAME("RichTextEditToolbar"));
	Ref<StyleBox> hover = get_theme_stylebox(SNAME("color_button_hover"), SNAME("RichTextEditToolbar"));
	Ref<StyleBox> pressed = get_theme_stylebox(SNAME("color_button_pressed"), SNAME("RichTextEditToolbar"));
	if (normal.is_valid()) {
		p_button->add_theme_style_override(CoreStringName(normal), normal);
		p_button->add_theme_style_override("disabled", normal);
		p_button->add_theme_style_override("focus", normal);
	}
	if (hover.is_valid()) {
		p_button->add_theme_style_override(SceneStringName(hover), hover);
	}
	if (pressed.is_valid()) {
		p_button->add_theme_style_override(SceneStringName(pressed), pressed);
		p_button->add_theme_style_override("hover_pressed", pressed);
	}
}

void RichTextEditToolbar::_apply_outline_button_color() {
	if (outline_button == nullptr) {
		return;
	}

	const Color outline_color = outline_color_button != nullptr ? outline_color_button->get_pick_color() : Color(0, 0, 0);
	outline_button->add_theme_color_override(SceneStringName(font_color), outline_color);
	outline_button->add_theme_color_override("font_hover_color", outline_color);
	outline_button->add_theme_color_override("font_pressed_color", outline_color);
	outline_button->add_theme_color_override("icon_normal_color", outline_color);
	outline_button->add_theme_color_override("icon_hover_color", outline_color);
	outline_button->add_theme_color_override("icon_pressed_color", outline_color);
}

void RichTextEditToolbar::_apply_dropdown_font_color() {
	const Color dropdown_font_color = get_theme_color(SNAME("dropdown_font_color"), SNAME("RichTextEditToolbar"));
	if (outline_color_label != nullptr) {
		outline_color_label->add_theme_color_override(SceneStringName(font_color), dropdown_font_color);
	}
	if (outline_size_label != nullptr) {
		outline_size_label->add_theme_color_override(SceneStringName(font_color), dropdown_font_color);
	}
}

void RichTextEditToolbar::_set_dropdown_button_arrow(Button *p_button, const String &p_fallback_text) {
	if (p_button == nullptr) {
		return;
	}

	const String arrow = String::chr(0x25BE);
	p_button->add_theme_constant_override("h_separation", 2);
	p_button->set_custom_minimum_size(Size2(40, 32));
	if (p_button->get_button_icon().is_valid()) {
		p_button->set_text(arrow);
	} else {
		p_button->set_text(p_fallback_text + " " + arrow);
	}
}

void RichTextEditToolbar::_apply_toolbar_icons() {
	_set_button_icon_or_text(bold_button, SNAME("bold"), "B");
	_set_button_icon_or_text(italic_button, SNAME("italic"), "I");
	_set_button_icon_or_text(underline_button, SNAME("underline"), "U");
	_set_button_icon_or_text(strikethrough_button, SNAME("strikethrough"), "S");
	_set_button_icon_or_text(quote_button, SNAME("quote"), "\"");
	_set_button_icon_or_text(alignment_button, SNAME("align"), RTR("Align"));
	_set_button_icon_or_text(indent_decrease_button, SNAME("indent_decrease"), "<");
	_set_button_icon_or_text(indent_increase_button, SNAME("indent_increase"), ">");
	_set_button_icon_or_text(link_button, SNAME("link"), RTR("Link"));
	_set_button_icon_or_text(ordered_list_button, SNAME("ordered_list"), "OL");
	_set_button_icon_or_text(unordered_list_button, SNAME("unordered_list"), "UL");
	_set_button_icon_or_text(align_left_button, SNAME("align_left"), RTR("Left"));
	_set_button_icon_or_text(align_center_button, SNAME("align_center"), RTR("Center"));
	_set_button_icon_or_text(align_right_button, SNAME("align_right"), RTR("Right"));
	_set_dropdown_button_arrow(alignment_button, RTR("Align"));
	_set_dropdown_button_arrow(outline_button, String::chr(0x25A1));
}

void RichTextEditToolbar::_apply_toolbar_style() {
	Ref<StyleBox> dropdown_panel = get_theme_stylebox(SNAME("dropdown_panel"), SNAME("RichTextEditToolbar"));
	if (dropdown_panel.is_valid()) {
		if (outline_popup != nullptr) {
			outline_popup->add_theme_style_override(SceneStringName(panel), dropdown_panel);
		}
		if (alignment_popup != nullptr) {
			alignment_popup->add_theme_style_override(SceneStringName(panel), dropdown_panel);
		}
		if (link_popup != nullptr) {
			link_popup->add_theme_style_override(SceneStringName(panel), dropdown_panel);
		}
	}

	_apply_toolbar_color_button_style(color_button);
	_apply_toolbar_color_button_style(bg_color_button);
	_apply_toolbar_color_button_style(outline_color_button);
	_apply_toolbar_button_style(outline_button);
	_apply_outline_button_color();
	_apply_dropdown_font_color();

	update_minimum_size();
	queue_redraw();
}

void RichTextEditToolbar::_pressed_bold() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_bold();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_italic() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_italic();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_underline() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_underline();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_strikethrough() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_strikethrough();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_quote() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_quote();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_outline() {
	if (outline_popup == nullptr || outline_button == nullptr) {
		return;
	}

	const Size2 min_size = outline_popup->get_contents_minimum_size();
	outline_popup->reset_size();
	outline_popup->set_position(outline_button->get_screen_position() + Vector2((outline_button->get_size().x - min_size.x) / 2.0, outline_button->get_size().y));
	outline_popup->popup();
}

void RichTextEditToolbar::_pressed_alignment() {
	if (alignment_popup == nullptr || alignment_button == nullptr) {
		return;
	}

	const Size2 min_size = alignment_popup->get_contents_minimum_size();
	alignment_popup->reset_size();
	alignment_popup->set_position(alignment_button->get_screen_position() + Vector2((alignment_button->get_size().x - min_size.x) / 2.0, alignment_button->get_size().y));
	alignment_popup->popup();
}

void RichTextEditToolbar::_pressed_link() {
	if (link_popup == nullptr || link_button == nullptr || link_line_edit == nullptr) {
		return;
	}

	const Size2 min_size = link_popup->get_contents_minimum_size();
	link_popup->reset_size();
	link_popup->set_position(link_button->get_screen_position() + Vector2((link_button->get_size().x - min_size.x) / 2.0, link_button->get_size().y));
	link_popup->popup();
	link_line_edit->grab_focus();
	link_line_edit->select_all();
}

void RichTextEditToolbar::_link_apply_pressed() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->set_selection_url(link_line_edit != nullptr ? link_line_edit->get_text() : String());
		target->grab_focus();
	}
	if (link_popup != nullptr) {
		link_popup->hide();
	}
}

void RichTextEditToolbar::_link_clear_pressed() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->clear_selection_url();
		target->grab_focus();
	}
	if (link_line_edit != nullptr) {
		link_line_edit->clear();
	}
	if (link_popup != nullptr) {
		link_popup->hide();
	}
}

void RichTextEditToolbar::_alignment_selected(int p_id) {
	RichTextEdit *target = _get_rich_text_edit();
	if (target == nullptr) {
		return;
	}

	switch (p_id) {
		case ALIGNMENT_LEFT:
			target->set_alignment(HORIZONTAL_ALIGNMENT_LEFT);
			break;
		case ALIGNMENT_CENTER:
			target->set_alignment(HORIZONTAL_ALIGNMENT_CENTER);
			break;
		case ALIGNMENT_RIGHT:
			target->set_alignment(HORIZONTAL_ALIGNMENT_RIGHT);
			break;
	}
	if (alignment_popup != nullptr) {
		alignment_popup->hide();
	}
	target->grab_focus();
}

void RichTextEditToolbar::_pressed_unordered_list() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_unordered_list();
	}
}

void RichTextEditToolbar::_pressed_ordered_list() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_ordered_list();
	}
}

void RichTextEditToolbar::_color_picker_created() {
	if (color_button == nullptr) {
		return;
	}

	PopupPanel *popup = color_button->get_popup();
	if (popup != nullptr && !popup->is_connected(SNAME("about_to_popup"), callable_mp(this, &RichTextEditToolbar::_color_popup_about_to_popup))) {
		popup->connect(SNAME("about_to_popup"), callable_mp(this, &RichTextEditToolbar::_color_popup_about_to_popup));
	}
}

void RichTextEditToolbar::_color_popup_about_to_popup() {
	color_picker_open = true;
	color_before_picker_open = color_button->get_pick_color();
	pending_picker_color = color_before_picker_open;
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->begin_selection_color_preview();
	}
}

void RichTextEditToolbar::_color_popup_closed() {
	if (!color_picker_open) {
		return;
	}

	color_picker_open = false;
	const Color final_color = color_button->get_pick_color();
	pending_picker_color = final_color;
	const bool commit_color = !final_color.is_equal_approx(color_before_picker_open);

	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->end_selection_color_preview(commit_color);
		target->grab_focus();
	}
}

void RichTextEditToolbar::_color_changed(const Color &p_color) {
	pending_picker_color = p_color;
	if (color_picker_open) {
		if (RichTextEdit *target = _get_rich_text_edit()) {
			target->preview_selection_color(p_color);
		}
	}
}

void RichTextEditToolbar::_bg_color_changed(const Color &p_color) {
	pending_bg_picker_color = p_color;
	if (bg_color_picker_open) {
		if (RichTextEdit *target = _get_rich_text_edit()) {
			target->preview_selection_bg_color(p_color);
		}
	}
}

void RichTextEditToolbar::_bg_color_picker_created() {
	if (bg_color_button == nullptr) {
		return;
	}

	PopupPanel *popup = bg_color_button->get_popup();
	if (popup != nullptr && !popup->is_connected(SNAME("about_to_popup"), callable_mp(this, &RichTextEditToolbar::_bg_color_popup_about_to_popup))) {
		popup->connect(SNAME("about_to_popup"), callable_mp(this, &RichTextEditToolbar::_bg_color_popup_about_to_popup));
	}
}

void RichTextEditToolbar::_bg_color_popup_about_to_popup() {
	bg_color_picker_open = true;
	bg_color_before_picker_open = bg_color_button->get_pick_color();
	pending_bg_picker_color = bg_color_before_picker_open;
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->begin_selection_bg_color_preview();
	}
}

void RichTextEditToolbar::_bg_color_popup_closed() {
	if (!bg_color_picker_open) {
		return;
	}

	bg_color_picker_open = false;
	const Color final_color = bg_color_button->get_pick_color();
	pending_bg_picker_color = final_color;
	const bool commit_color = !final_color.is_equal_approx(bg_color_before_picker_open);

	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->end_selection_bg_color_preview(commit_color);
		target->grab_focus();
	}
}

void RichTextEditToolbar::_outline_color_changed(const Color &p_color) {
	_apply_outline_button_color();
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->set_selection_outline_color(p_color);
		target->grab_focus();
	}
}

void RichTextEditToolbar::_font_size_changed(double p_value) {
	if (updating_controls) {
		return;
	}
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->set_selection_font_size(p_value);
		target->grab_focus();
	}
}

void RichTextEditToolbar::_outline_size_changed(double p_value) {
	if (updating_controls) {
		return;
	}
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->set_selection_outline_size(p_value);
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_decrease_indent() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->decrease_indent();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_increase_indent() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->increase_indent();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			Ref<StyleBox> panel = get_theme_stylebox(SceneStringName(panel), SNAME("RichTextEditToolbar"));
			if (panel.is_valid()) {
				panel->draw(get_canvas_item(), Rect2(Point2(), get_size()));
			}
		} break;

		case NOTIFICATION_READY:
			_resolve_rich_text_edit();
			_apply_toolbar_icons();
			_apply_toolbar_style();
			break;

		case NOTIFICATION_THEME_CHANGED:
			_apply_toolbar_icons();
			_apply_toolbar_style();
			break;
	}
}

void RichTextEditToolbar::set_rich_text_edit_path(const NodePath &p_path) {
	rich_text_edit_path = p_path;
	if (is_inside_tree()) {
		_resolve_rich_text_edit();
	}
}

NodePath RichTextEditToolbar::get_rich_text_edit_path() const {
	return rich_text_edit_path;
}

RichTextEdit *RichTextEditToolbar::get_rich_text_edit() const {
	return _get_rich_text_edit();
}

void RichTextEditToolbar::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_rich_text_edit_path", "path"), &RichTextEditToolbar::set_rich_text_edit_path);
	ClassDB::bind_method(D_METHOD("get_rich_text_edit_path"), &RichTextEditToolbar::get_rich_text_edit_path);
	ClassDB::bind_method(D_METHOD("get_rich_text_edit"), &RichTextEditToolbar::get_rich_text_edit);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "rich_text_edit_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, RichTextEdit::get_class_static()), "set_rich_text_edit_path", "get_rich_text_edit_path");

	auto update_icons = [](Node *p_instance, const StringName &, const StringName &) {
		RichTextEditToolbar *toolbar = Object::cast_to<RichTextEditToolbar>(p_instance);
		if (toolbar != nullptr) {
			toolbar->_apply_toolbar_icons();
		}
	};
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "bold", "bold", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "italic", "italic", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "underline", "underline", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "strikethrough", "strikethrough", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "quote", "quote", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "align", "align", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "align_left", "align_left", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "align_center", "align_center", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "align_right", "align_right", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "indent_decrease", "indent_decrease", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "indent_increase", "indent_increase", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "link", "link", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "ordered_list", "ordered_list", update_icons);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), "unordered_list", "unordered_list", update_icons);

	auto update_style = [](Node *p_instance, const StringName &, const StringName &) {
		RichTextEditToolbar *toolbar = Object::cast_to<RichTextEditToolbar>(p_instance);
		if (toolbar != nullptr) {
			toolbar->_apply_toolbar_style();
		}
	};
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_STYLEBOX, get_class_static(), "panel", "panel", update_style);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_STYLEBOX, get_class_static(), "dropdown_panel", "dropdown_panel", update_style);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_STYLEBOX, get_class_static(), "button_normal", "button_normal", update_style);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_STYLEBOX, get_class_static(), "button_hover", "button_hover", update_style);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_STYLEBOX, get_class_static(), "button_pressed", "button_pressed", update_style);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_STYLEBOX, get_class_static(), "color_button_normal", "color_button_normal", update_style);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_STYLEBOX, get_class_static(), "color_button_hover", "color_button_hover", update_style);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_STYLEBOX, get_class_static(), "color_button_pressed", "color_button_pressed", update_style);
	ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_COLOR, get_class_static(), "dropdown_font_color", "dropdown_font_color", update_style);
}

RichTextEditToolbar::RichTextEditToolbar() {
	add_theme_constant_override("separation", 4);
	set_custom_minimum_size(Size2(0, 44));

	auto add_toolbar_separator = [&]() {
		VSeparator *separator = memnew(VSeparator);
		add_child(separator, false, INTERNAL_MODE_FRONT);
	};

	auto add_toolbar_padding = [&]() {
		Control *padding = memnew(Control);
		padding->set_custom_minimum_size(Size2(4, 0));
		add_child(padding, false, INTERNAL_MODE_FRONT);
	};

	auto add_button = [&](Button *&r_button, const String &p_text, const String &p_tooltip, void (RichTextEditToolbar::*p_callback)()) {
		r_button = memnew(Button(p_text));
		r_button->set_focus_mode(FOCUS_NONE);
		r_button->set_flat(true);
		r_button->set_custom_minimum_size(Size2(32, 32));
		r_button->set_tooltip_text(p_tooltip);
		add_child(r_button, false, INTERNAL_MODE_FRONT);
		r_button->connect(SceneStringName(pressed), callable_mp(this, p_callback));
	};

	add_toolbar_padding();

	bold_button = memnew(Button("B"));
	bold_button->set_focus_mode(FOCUS_NONE);
	bold_button->set_flat(true);
	bold_button->set_custom_minimum_size(Size2(32, 32));
	bold_button->set_tooltip_text(RTR("Bold"));
	add_child(bold_button, false, INTERNAL_MODE_FRONT);
	bold_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_bold));

	italic_button = memnew(Button("I"));
	italic_button->set_focus_mode(FOCUS_NONE);
	italic_button->set_flat(true);
	italic_button->set_custom_minimum_size(Size2(32, 32));
	italic_button->set_tooltip_text(RTR("Italic"));
	add_child(italic_button, false, INTERNAL_MODE_FRONT);
	italic_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_italic));

	underline_button = memnew(Button("U"));
	underline_button->set_focus_mode(FOCUS_NONE);
	underline_button->set_flat(true);
	underline_button->set_custom_minimum_size(Size2(32, 32));
	underline_button->set_tooltip_text(RTR("Underline"));
	add_child(underline_button, false, INTERNAL_MODE_FRONT);
	underline_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_underline));

	strikethrough_button = memnew(Button("S"));
	strikethrough_button->set_focus_mode(FOCUS_NONE);
	strikethrough_button->set_flat(true);
	strikethrough_button->set_custom_minimum_size(Size2(32, 32));
	strikethrough_button->set_tooltip_text(RTR("Strikethrough"));
	add_child(strikethrough_button, false, INTERNAL_MODE_FRONT);
	strikethrough_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_strikethrough));

	add_toolbar_separator();

	font_size_spin = memnew(SpinBox);
	font_size_spin->set_min(1);
	font_size_spin->set_max(256);
	font_size_spin->set_step(1);
	font_size_spin->set_value(16);
	font_size_spin->set_custom_minimum_size(Size2(44, 32));
	font_size_spin->set_tooltip_text(RTR("Font Size"));
	add_child(font_size_spin, false, INTERNAL_MODE_FRONT);
	font_size_spin->connect(SceneStringName(value_changed), callable_mp(this, &RichTextEditToolbar::_font_size_changed));

	color_button = memnew(ColorPickerButton);
	color_button->set_focus_mode(FOCUS_NONE);
	color_button->set_custom_minimum_size(Size2(32, 32));
	color_button->set_pick_color(Color(1, 1, 1));
	color_button->set_tooltip_text(RTR("Font Color"));
	add_child(color_button, false, INTERNAL_MODE_FRONT);
	color_button->connect("picker_created", callable_mp(this, &RichTextEditToolbar::_color_picker_created));
	color_button->connect("popup_closed", callable_mp(this, &RichTextEditToolbar::_color_popup_closed));
	color_button->connect("color_changed", callable_mp(this, &RichTextEditToolbar::_color_changed));

	bg_color_button = memnew(ColorPickerButton);
	bg_color_button->set_focus_mode(FOCUS_NONE);
	bg_color_button->set_custom_minimum_size(Size2(32, 32));
	bg_color_button->set_pick_color(Color(0, 0, 0, 1));
	bg_color_button->set_tooltip_text(RTR("Background Color"));
	add_child(bg_color_button, false, INTERNAL_MODE_FRONT);
	bg_color_button->connect("picker_created", callable_mp(this, &RichTextEditToolbar::_bg_color_picker_created));
	bg_color_button->connect("popup_closed", callable_mp(this, &RichTextEditToolbar::_bg_color_popup_closed));
	bg_color_button->connect("color_changed", callable_mp(this, &RichTextEditToolbar::_bg_color_changed));

	outline_button = memnew(Button(String::chr(0x25A1)));
	outline_button->set_focus_mode(FOCUS_NONE);
	outline_button->set_flat(true);
	outline_button->set_custom_minimum_size(Size2(32, 32));
	outline_button->set_tooltip_text(RTR("Outline Color and Width"));
	add_child(outline_button, false, INTERNAL_MODE_FRONT);
	outline_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_outline));

	outline_popup = memnew(PopupPanel);
	VBoxContainer *outline_box = memnew(VBoxContainer);
	outline_popup->add_child(outline_box, false, INTERNAL_MODE_FRONT);
	HBoxContainer *outline_color_row = memnew(HBoxContainer);
	outline_box->add_child(outline_color_row, false, INTERNAL_MODE_FRONT);
	outline_color_label = memnew(Label(RTR("Color")));
	outline_color_row->add_child(outline_color_label, false, INTERNAL_MODE_FRONT);
	outline_color_button = memnew(ColorPickerButton);
	outline_color_button->set_focus_mode(FOCUS_NONE);
	outline_color_button->set_custom_minimum_size(Size2(48, 32));
	outline_color_button->set_pick_color(Color(0, 0, 0));
	outline_color_button->set_tooltip_text(RTR("Outline Color"));
	outline_color_row->add_child(outline_color_button, false, INTERNAL_MODE_FRONT);
	outline_color_button->connect("color_changed", callable_mp(this, &RichTextEditToolbar::_outline_color_changed));

	HBoxContainer *outline_size_row = memnew(HBoxContainer);
	outline_box->add_child(outline_size_row, false, INTERNAL_MODE_FRONT);
	outline_size_label = memnew(Label(RTR("Width")));
	outline_size_row->add_child(outline_size_label, false, INTERNAL_MODE_FRONT);
	outline_size_spin = memnew(SpinBox);
	outline_size_spin->set_min(0);
	outline_size_spin->set_max(32);
	outline_size_spin->set_step(1);
	outline_size_spin->set_value(0);
	outline_size_spin->set_tooltip_text(RTR("Outline Size"));
	outline_size_row->add_child(outline_size_spin, false, INTERNAL_MODE_FRONT);
	outline_size_spin->connect(SceneStringName(value_changed), callable_mp(this, &RichTextEditToolbar::_outline_size_changed));
	add_child(outline_popup, false, INTERNAL_MODE_FRONT);

	add_toolbar_separator();

	add_button(quote_button, "\"", RTR("Quote"), &RichTextEditToolbar::_pressed_quote);

	alignment_button = memnew(Button(RTR("Align")));
	alignment_button->set_focus_mode(FOCUS_NONE);
	alignment_button->set_flat(true);
	alignment_button->set_custom_minimum_size(Size2(32, 32));
	alignment_button->set_tooltip_text(RTR("Paragraph Alignment"));
	add_child(alignment_button, false, INTERNAL_MODE_FRONT);
	alignment_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_alignment));

	alignment_popup = memnew(PopupPanel);
	HBoxContainer *alignment_box = memnew(HBoxContainer);
	alignment_box->add_theme_constant_override("separation", 4);
	alignment_popup->add_child(alignment_box, false, INTERNAL_MODE_FRONT);

	auto add_alignment_button = [&](Button *&r_button, const String &p_text, const String &p_tooltip, int p_alignment) {
		r_button = memnew(Button(p_text));
		r_button->set_focus_mode(FOCUS_NONE);
		r_button->set_flat(true);
		r_button->set_custom_minimum_size(Size2(32, 32));
		r_button->set_tooltip_text(p_tooltip);
		alignment_box->add_child(r_button, false, INTERNAL_MODE_FRONT);
		r_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_alignment_selected).bind(p_alignment));
	};

	add_alignment_button(align_left_button, RTR("Left"), RTR("Align Left"), ALIGNMENT_LEFT);
	add_alignment_button(align_center_button, RTR("Center"), RTR("Align Center"), ALIGNMENT_CENTER);
	add_alignment_button(align_right_button, RTR("Right"), RTR("Align Right"), ALIGNMENT_RIGHT);
	add_child(alignment_popup, false, INTERNAL_MODE_FRONT);

	add_button(indent_decrease_button, "<", RTR("Decrease Indent"), &RichTextEditToolbar::_pressed_decrease_indent);
	add_button(indent_increase_button, ">", RTR("Increase Indent"), &RichTextEditToolbar::_pressed_increase_indent);

	add_toolbar_separator();

	link_button = memnew(Button(RTR("Link")));
	link_button->set_focus_mode(FOCUS_NONE);
	link_button->set_tooltip_text(RTR("URL Link"));
	add_child(link_button, false, INTERNAL_MODE_FRONT);
	link_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_link));

	link_popup = memnew(PopupPanel);
	VBoxContainer *link_box = memnew(VBoxContainer);
	link_popup->add_child(link_box, false, INTERNAL_MODE_FRONT);
	link_line_edit = memnew(LineEdit);
	link_line_edit->set_custom_minimum_size(Size2(300, 0));
	link_line_edit->set_placeholder(RTR("https://example.com"));
	link_line_edit->set_tooltip_text(RTR("URL"));
	link_box->add_child(link_line_edit, false, INTERNAL_MODE_FRONT);
	HBoxContainer *link_actions = memnew(HBoxContainer);
	link_box->add_child(link_actions, false, INTERNAL_MODE_FRONT);
	Button *link_apply_button = memnew(Button(RTR("Apply")));
	link_apply_button->set_focus_mode(FOCUS_NONE);
	link_apply_button->set_tooltip_text(RTR("Apply URL Link"));
	link_actions->add_child(link_apply_button, false, INTERNAL_MODE_FRONT);
	link_apply_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_link_apply_pressed));
	Button *link_clear_button = memnew(Button(RTR("Clear")));
	link_clear_button->set_focus_mode(FOCUS_NONE);
	link_clear_button->set_tooltip_text(RTR("Clear URL Link"));
	link_actions->add_child(link_clear_button, false, INTERNAL_MODE_FRONT);
	link_clear_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_link_clear_pressed));
	add_child(link_popup, false, INTERNAL_MODE_FRONT);

	add_button(ordered_list_button, "OL", RTR("Ordered List"), &RichTextEditToolbar::_pressed_ordered_list);
	add_button(unordered_list_button, "UL", RTR("Unordered List"), &RichTextEditToolbar::_pressed_unordered_list);

	add_toolbar_padding();
}

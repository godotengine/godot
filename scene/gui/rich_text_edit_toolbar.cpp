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

#include "core/config/project_settings.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup.h"
#include "scene/gui/separator.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/style_box_flat.h"
#include "scene/theme/theme_db.h"

enum {
	ALIGNMENT_LEFT,
	ALIGNMENT_CENTER,
	ALIGNMENT_RIGHT,
};

enum {
	DROPDOWN_FONT_COLOR,
	DROPDOWN_BG_COLOR,
	DROPDOWN_OUTLINE_COLOR,
	DROPDOWN_ALIGNMENT,
	DROPDOWN_LINK,
	DROPDOWN_LINE_HEIGHT,
};

// Layout metrics taken from the toolbar design reference. Only the values that
// have no visual meaning on their own are kept here; everything that defines
// the look (colors, styleboxes, font sizes) comes from the theme.
namespace {
constexpr int DEFAULT_BUTTON_SIZE = 28;
constexpr int DEFAULT_ICON_MAX_WIDTH = 16;
constexpr int DEFAULT_SEPARATOR_HEIGHT = 20;
constexpr int GROUP_SEPARATION = 2;

// A dropdown button is padded on both sides, with the caret sitting a short gap
// to the right of the icon. The design's dropdown buttons come out 38-41px wide;
// reserving the strip without also padding the left made them 45px and left a
// hole between icon and caret.
constexpr int CARET_SIZE = 8;
constexpr int CARET_LEFT_GAP = 3;
constexpr int DROPDOWN_SIDE_PADDING = 6;
constexpr int CARET_STRIP_WIDTH = CARET_LEFT_GAP + CARET_SIZE + DROPDOWN_SIDE_PADDING;

// Field width matches the design: 38px text box, 16px spinner column, 1px border.
constexpr int NUMBER_FIELD_TEXT_WIDTH = 38;
constexpr int NUMBER_FIELD_BUTTONS_WIDTH = 16;
constexpr int NUMBER_FIELD_BORDER = 1;
constexpr int NUMBER_FIELD_WIDTH = NUMBER_FIELD_TEXT_WIDTH + NUMBER_FIELD_BUTTONS_WIDTH + NUMBER_FIELD_BORDER * 2;
// Keeps the text box near its designed width instead of LineEdit's own default
// of four 'W' glyphs, which alone is wider than the whole field.
constexpr int NUMBER_FIELD_MIN_CHARS = 3;

constexpr int ALIGNMENT_PANEL_WIDTH = 150;
constexpr int LINK_PANEL_WIDTH = 220;
// The reference panel is 220px wide inside a 10px border/padding frame.
constexpr int LINE_HEIGHT_PANEL_WIDTH = 240;

// Color preview strips inside the two color buttons.
constexpr int COLOR_PREVIEW_INSET = 5;
constexpr int COLOR_PREVIEW_BOTTOM = 4;
constexpr int FONT_COLOR_BAR_HEIGHT = 3;
constexpr int BG_COLOR_SWATCH_HEIGHT = 9;

constexpr int DROPDOWN_GAP = 6;

// "No Color" row: a swatch the same size as a design palette tile, inset from
// the row's left edge, with the label clearing it.
constexpr int NONE_SWATCH_SIZE = 20;
constexpr int NONE_SWATCH_MARGIN = 8;
} // namespace

Control *RichTextEditToolbarButton::make_toolbar_tooltip(const Control *p_owner, const String &p_text) {
	if (p_text.is_empty()) {
		return nullptr;
	}

	PanelContainer *panel = memnew(PanelContainer);
	Ref<StyleBox> panel_style = p_owner->get_theme_stylebox(SNAME("tooltip_panel"), SNAME("RichTextEditToolbar"));
	if (panel_style.is_null()) {
		panel_style = p_owner->get_theme_stylebox(SceneStringName(panel), SNAME("TooltipPanel"));
	}
	if (panel_style.is_valid()) {
		panel->add_theme_style_override(SceneStringName(panel), panel_style);
	}

	Label *label = memnew(Label(p_text));
	label->add_theme_color_override(SceneStringName(font_color), p_owner->get_theme_color(SNAME("tooltip_font_color"), SNAME("RichTextEditToolbar")));
	const int font_size = p_owner->get_theme_font_size(SNAME("tooltip_font_size"), SNAME("RichTextEditToolbar"));
	if (font_size > 0) {
		label->add_theme_font_size_override(SceneStringName(font_size), font_size);
	}

	panel->add_child(label);
	return panel;
}

Control *RichTextEditToolbarButton::make_custom_tooltip(const String &p_text) const {
	return make_toolbar_tooltip(this, p_text);
}

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
		if (old_target->is_connected("text_style_changed", callable_mp(this, &RichTextEditToolbar::_target_caret_changed))) {
			old_target->disconnect("text_style_changed", callable_mp(this, &RichTextEditToolbar::_target_caret_changed));
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
		if (!target->is_connected("text_style_changed", callable_mp(this, &RichTextEditToolbar::_target_caret_changed))) {
			target->connect("text_style_changed", callable_mp(this, &RichTextEditToolbar::_target_caret_changed));
		}
	}
	_update_controls_from_target();
}

bool RichTextEditToolbar::_get_current_style(RichTextEdit::TextStyle &r_style) const {
	RichTextEdit *target = _get_rich_text_edit();
	if (target == nullptr) {
		return false;
	}

	// Resolve the same offset RichTextEdit::get_current_font_size() uses: the
	// start of the selection, or the character before the caret.
	const int caret_line = target->has_selection() ? target->get_selection_from_line() : target->get_caret_line();
	int offset = 0;
	for (int line = 0; line < caret_line; line++) {
		offset += target->get_line(line).length() + 1;
	}
	if (target->has_selection()) {
		offset += target->get_selection_from_column();
	} else {
		const int caret_column = target->get_caret_column();
		offset += caret_column > 0 ? caret_column - 1 : 0;
	}

	r_style = RichTextEdit::TextStyle();
	for (const RichTextEdit::StyleSpan &span : target->get_style_spans()) {
		if (offset >= span.from && offset < span.to) {
			r_style = span.style;
			break;
		}
	}
	return true;
}

void RichTextEditToolbar::_update_controls_from_target() {
	if (font_size_spin == nullptr) {
		return;
	}

	RichTextEdit *target = _get_rich_text_edit();
	RichTextEdit::TextStyle style;
	const bool has_target = _get_current_style(style);

	updating_controls = true;

	if (has_target) {
		font_size_spin->set_value(target->get_current_font_size());
	}

	bold_button->set_pressed_no_signal(has_target && style.bold);
	italic_button->set_pressed_no_signal(has_target && style.italic);
	underline_button->set_pressed_no_signal(has_target && style.has_underline && style.underline);
	strikethrough_button->set_pressed_no_signal(has_target && style.strikethrough);
	overline_button->set_pressed_no_signal(has_target && style.overline);

	// Quote, lists, alignment and indentation all share the block tag slot.
	const String block_tag = has_target ? style.block_tag : String();
	quote_button->set_pressed_no_signal(block_tag == "quote");
	ordered_list_button->set_pressed_no_signal(block_tag == "ol");
	unordered_list_button->set_pressed_no_signal(block_tag == "ul");
	align_center_button->set_pressed_no_signal(block_tag == "center");
	align_right_button->set_pressed_no_signal(block_tag == "right");
	align_left_button->set_pressed_no_signal(block_tag != "center" && block_tag != "right");
	_set_button_icon(alignment_button, block_tag == "center" ? SNAME("align_center") : (block_tag == "right" ? SNAME("align_right") : SNAME("align_left")));
	if (target != nullptr && line_height_button != nullptr) {
		// The button itself is only highlighted while its popup is open; the
		// active line height is shown by the checked preset inside the popup.
		const String line_height = target->get_current_line_height();
		if (line_height_line_edit != nullptr && (line_height_popup == nullptr || !line_height_popup->is_visible())) {
			line_height_line_edit->set_text(line_height == "normal" ? "1" : line_height);
		}
		for (Button *preset_button : line_height_preset_buttons) {
			if (preset_button->has_meta("line_height_value")) {
				const String preset_value = preset_button->get_meta("line_height_value");
				preset_button->set_pressed_no_signal(preset_value == (line_height == "normal" ? "1" : line_height));
			}
		}
	}

	// An open picker owns the color it is editing; the previews it drives must
	// not be overwritten by the caret updates its own live preview triggers.
	if (has_target && !color_picker_open && !bg_color_picker_open) {
		if (style.has_color) {
			font_color_value = style.color;
		}
		bg_color_assigned = style.has_bg_color;
		if (style.has_bg_color) {
			bg_color_value = style.bg_color;
		}
		// Assigned unconditionally: text without an outline has to read back as
		// no outline, otherwise the controls keep describing the previous caret
		// position.
		outline_color_value = style.has_outline_color ? style.outline_color : Color(0, 0, 0);
		outline_size_spin->set_value(style.has_outline_size ? style.outline_size : 0);
	}

	_apply_color_button_previews();

	updating_controls = false;
}

void RichTextEditToolbar::_target_caret_changed() {
	_update_controls_from_target();
}

String RichTextEditToolbar::get_tooltip(const Point2 &p_pos) const {
	// The number fields let the mouse pass through so the toolbar can own their
	// tooltip and give it the same look as the buttons'.
	if (font_size_field != nullptr && font_size_field->get_rect().has_point(p_pos)) {
		return RTR("Font Size");
	}
	return HBoxContainer::get_tooltip(p_pos);
}

Control *RichTextEditToolbar::make_custom_tooltip(const String &p_text) const {
	return RichTextEditToolbarButton::make_toolbar_tooltip(this, p_text);
}

void RichTextEditToolbar::_set_button_icon(Button *p_button, const StringName &p_icon_name) {
	if (p_button == nullptr) {
		return;
	}

	if (has_theme_icon(p_icon_name, SNAME("RichTextEditToolbar"))) {
		p_button->set_button_icon(get_theme_icon(p_icon_name, SNAME("RichTextEditToolbar")));
	} else {
		p_button->set_button_icon(Ref<Texture2D>());
	}
}

void RichTextEditToolbar::_style_tool_button(Button *p_button) {
	if (p_button == nullptr) {
		return;
	}

	const int button_size = get_theme_constant(SNAME("button_size"), SNAME("RichTextEditToolbar"));
	const int icon_max_width = get_theme_constant(SNAME("icon_max_width"), SNAME("RichTextEditToolbar"));
	const Color font_color = get_theme_color(SNAME("button_font_color"), SNAME("RichTextEditToolbar"));
	const Color accent_color = get_theme_color(SNAME("accent_color"), SNAME("RichTextEditToolbar"));

	p_button->set_flat(false);
	p_button->set_custom_minimum_size(Size2(button_size, button_size));
	p_button->set_v_size_flags(SIZE_SHRINK_CENTER);
	p_button->add_theme_constant_override("icon_max_width", icon_max_width);
	p_button->add_theme_constant_override("h_separation", CARET_LEFT_GAP);
	p_button->set_expand_icon(false);
	p_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	p_button->set_vertical_icon_alignment(VERTICAL_ALIGNMENT_CENTER);

	const Ref<StyleBox> normal = get_theme_stylebox(SNAME("button_normal"), SNAME("RichTextEditToolbar"));
	const Ref<StyleBox> hover = get_theme_stylebox(SNAME("button_hover"), SNAME("RichTextEditToolbar"));
	const Ref<StyleBox> pressed = get_theme_stylebox(SNAME("button_pressed"), SNAME("RichTextEditToolbar"));
	const Ref<StyleBox> focus = get_theme_stylebox(SNAME("button_focus"), SNAME("RichTextEditToolbar"));
	if (normal.is_valid()) {
		p_button->add_theme_style_override(CoreStringName(normal), normal);
		p_button->add_theme_style_override("disabled", normal);
	}
	if (hover.is_valid()) {
		p_button->add_theme_style_override(SceneStringName(hover), hover);
	}
	if (pressed.is_valid()) {
		p_button->add_theme_style_override(SceneStringName(pressed), pressed);
		p_button->add_theme_style_override("hover_pressed", pressed);
	}
	if (focus.is_valid()) {
		p_button->add_theme_style_override("focus", focus);
	}

	// The selected state tints icon and label with the accent color; the caret
	// glyph of dropdown buttons keeps its own muted color (see below).
	p_button->add_theme_color_override("icon_normal_color", font_color);
	p_button->add_theme_color_override("icon_hover_color", font_color);
	p_button->add_theme_color_override("icon_focus_color", font_color);
	p_button->add_theme_color_override("icon_pressed_color", accent_color);
	p_button->add_theme_color_override("icon_hover_pressed_color", accent_color);
	p_button->add_theme_color_override(SceneStringName(font_color), font_color);
	p_button->add_theme_color_override("font_hover_color", font_color);
	p_button->add_theme_color_override("font_focus_color", font_color);
	p_button->add_theme_color_override("font_pressed_color", accent_color);
	p_button->add_theme_color_override("font_hover_pressed_color", accent_color);
}

void RichTextEditToolbar::_style_dropdown_button(Button *p_button) {
	if (p_button == nullptr) {
		return;
	}

	_style_tool_button(p_button);

	const int button_size = get_theme_constant(SNAME("button_size"), SNAME("RichTextEditToolbar"));
	const int icon_max_width = get_theme_constant(SNAME("icon_max_width"), SNAME("RichTextEditToolbar"));
	p_button->set_custom_minimum_size(Size2(DROPDOWN_SIDE_PADDING + icon_max_width + CARET_STRIP_WIDTH, button_size));

	// Pad both sides so the content box is exactly the icon's width: the icon
	// then lands against the caret's gap instead of floating in slack space.
	const StringName style_names[] = { CoreStringName(normal), SceneStringName(hover), SceneStringName(pressed), SNAME("hover_pressed"), SNAME("disabled") };
	for (const StringName &style_name : style_names) {
		const Ref<StyleBox> base = p_button->get_theme_stylebox(style_name);
		if (base.is_null()) {
			continue;
		}
		Ref<StyleBox> reserved = base->duplicate();
		reserved->set_content_margin(SIDE_LEFT, DROPDOWN_SIDE_PADDING);
		reserved->set_content_margin(SIDE_RIGHT, CARET_STRIP_WIDTH);
		p_button->add_theme_style_override(style_name, reserved);
	}
}

void RichTextEditToolbar::_style_menu_item_button(Button *p_button) {
	if (p_button == nullptr) {
		return;
	}

	const int icon_max_width = get_theme_constant(SNAME("icon_max_width"), SNAME("RichTextEditToolbar"));
	const Color font_color = get_theme_color(SNAME("dropdown_font_color"), SNAME("RichTextEditToolbar"));
	const Color muted_color = get_theme_color(SNAME("muted_font_color"), SNAME("RichTextEditToolbar"));
	const Color accent_color = get_theme_color(SNAME("accent_color"), SNAME("RichTextEditToolbar"));

	p_button->set_flat(false);
	p_button->set_h_size_flags(SIZE_EXPAND_FILL);
	p_button->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	p_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	p_button->set_vertical_icon_alignment(VERTICAL_ALIGNMENT_CENTER);
	p_button->set_expand_icon(false);
	p_button->add_theme_constant_override("icon_max_width", icon_max_width);
	p_button->add_theme_constant_override("h_separation", 8);

	const Ref<StyleBox> normal = get_theme_stylebox(SNAME("menu_item_normal"), SNAME("RichTextEditToolbar"));
	const Ref<StyleBox> hover = get_theme_stylebox(SNAME("menu_item_hover"), SNAME("RichTextEditToolbar"));
	const Ref<StyleBox> pressed = get_theme_stylebox(SNAME("menu_item_pressed"), SNAME("RichTextEditToolbar"));
	const Ref<StyleBox> focus = get_theme_stylebox(SNAME("button_focus"), SNAME("RichTextEditToolbar"));
	if (normal.is_valid()) {
		p_button->add_theme_style_override(CoreStringName(normal), normal);
		p_button->add_theme_style_override("disabled", normal);
	}
	if (hover.is_valid()) {
		p_button->add_theme_style_override(SceneStringName(hover), hover);
	}
	if (pressed.is_valid()) {
		p_button->add_theme_style_override(SceneStringName(pressed), pressed);
		p_button->add_theme_style_override("hover_pressed", pressed);
	}
	if (focus.is_valid()) {
		p_button->add_theme_style_override("focus", focus);
	}

	const int font_size = get_theme_font_size(SNAME("menu_item_font_size"), SNAME("RichTextEditToolbar"));
	if (font_size > 0) {
		p_button->add_theme_font_size_override(SceneStringName(font_size), font_size);
	}

	p_button->add_theme_color_override(SceneStringName(font_color), font_color);
	p_button->add_theme_color_override("font_hover_color", font_color);
	p_button->add_theme_color_override("font_focus_color", font_color);
	p_button->add_theme_color_override("font_pressed_color", accent_color);
	p_button->add_theme_color_override("font_hover_pressed_color", accent_color);
	p_button->add_theme_color_override("icon_normal_color", muted_color);
	p_button->add_theme_color_override("icon_hover_color", font_color);
	p_button->add_theme_color_override("icon_focus_color", font_color);
	p_button->add_theme_color_override("icon_pressed_color", accent_color);
	p_button->add_theme_color_override("icon_hover_pressed_color", accent_color);
}

void RichTextEditToolbar::_style_none_button() {
	if (bg_color_none_button == nullptr) {
		return;
	}

	_style_menu_item_button(bg_color_none_button);

	// Widen the row's left padding so the label clears the swatch.
	const StringName style_names[] = { CoreStringName(normal), SceneStringName(hover), SceneStringName(pressed), SNAME("hover_pressed"), SNAME("disabled") };
	for (const StringName &style_name : style_names) {
		const Ref<StyleBox> base = bg_color_none_button->get_theme_stylebox(style_name);
		if (base.is_null()) {
			continue;
		}
		Ref<StyleBox> inset = base->duplicate();
		inset->set_content_margin(SIDE_LEFT, NONE_SWATCH_MARGIN * 2 + NONE_SWATCH_SIZE);
		bg_color_none_button->add_theme_style_override(style_name, inset);
	}
	bg_color_none_button->set_custom_minimum_size(Size2(0, get_theme_constant(SNAME("button_size"), SNAME("RichTextEditToolbar"))));

	Ref<StyleBoxFlat> swatch;
	swatch.instantiate();
	swatch->set_bg_color(Color(0, 0, 0, 0));
	swatch->set_draw_center(false);
	swatch->set_border_width_all(1);
	swatch->set_border_color(get_theme_color(SNAME("muted_font_color"), SNAME("RichTextEditToolbar")));
	swatch->set_corner_radius_all(3);
	bg_color_none_swatch->add_theme_style_override(SceneStringName(panel), swatch);
	if (has_theme_icon(SNAME("color_checkerboard"), SNAME("RichTextEditToolbar"))) {
		bg_color_none_checker->set_texture(get_theme_icon(SNAME("color_checkerboard"), SNAME("RichTextEditToolbar")));
	}

	const Ref<StyleBox> menu_separator = get_theme_stylebox(SNAME("menu_separator"), SNAME("RichTextEditToolbar"));
	if (menu_separator.is_valid() && bg_color_none_separator != nullptr) {
		bg_color_none_separator->add_theme_style_override(SNAME("separator"), menu_separator);
		bg_color_none_separator->add_theme_constant_override(SNAME("separation"), DROPDOWN_GAP * 2 + 1);
	}
}

void RichTextEditToolbar::_style_action_button(Button *p_button, const StringName &p_normal, const StringName &p_hover, const Color &p_font_color, const Color &p_hover_font_color) {
	if (p_button == nullptr) {
		return;
	}

	p_button->set_flat(false);

	const Ref<StyleBox> normal = get_theme_stylebox(p_normal, SNAME("RichTextEditToolbar"));
	const Ref<StyleBox> hover = get_theme_stylebox(p_hover, SNAME("RichTextEditToolbar"));
	const Ref<StyleBox> focus = get_theme_stylebox(SNAME("button_focus"), SNAME("RichTextEditToolbar"));
	if (normal.is_valid()) {
		p_button->add_theme_style_override(CoreStringName(normal), normal);
		p_button->add_theme_style_override("disabled", normal);
	}
	if (hover.is_valid()) {
		p_button->add_theme_style_override(SceneStringName(hover), hover);
		p_button->add_theme_style_override(SceneStringName(pressed), hover);
		p_button->add_theme_style_override("hover_pressed", hover);
	}
	if (focus.is_valid()) {
		p_button->add_theme_style_override("focus", focus);
	}

	const int font_size = get_theme_font_size(SNAME("action_button_font_size"), SNAME("RichTextEditToolbar"));
	if (font_size > 0) {
		p_button->add_theme_font_size_override(SceneStringName(font_size), font_size);
	}

	p_button->add_theme_color_override(SceneStringName(font_color), p_font_color);
	p_button->add_theme_color_override("font_focus_color", p_font_color);
	p_button->add_theme_color_override("font_hover_color", p_hover_font_color);
	p_button->add_theme_color_override("font_pressed_color", p_hover_font_color);
}

void RichTextEditToolbar::_style_number_field(PanelContainer *p_field, SpinBox *p_spin) {
	if (p_field == nullptr || p_spin == nullptr) {
		return;
	}

	const int button_size = get_theme_constant(SNAME("button_size"), SNAME("RichTextEditToolbar"));
	const Color muted_color = get_theme_color(SNAME("muted_font_color"), SNAME("RichTextEditToolbar"));
	const Color font_color = get_theme_color(SNAME("dropdown_font_color"), SNAME("RichTextEditToolbar"));

	const Ref<StyleBox> field_panel = get_theme_stylebox(SNAME("field_panel"), SNAME("RichTextEditToolbar"));
	if (field_panel.is_valid()) {
		p_field->add_theme_style_override(SceneStringName(panel), field_panel);
	}
	p_field->set_custom_minimum_size(Size2(NUMBER_FIELD_WIDTH, button_size));
	p_field->set_v_size_flags(SIZE_SHRINK_CENTER);
	// Keeps its width inside the outline dropdown's vertical box rather than
	// stretching to the panel width.
	p_field->set_h_size_flags(SIZE_SHRINK_BEGIN);

	const Ref<StyleBox> separator = get_theme_stylebox(SNAME("field_separator"), SNAME("RichTextEditToolbar"));
	const Ref<StyleBox> empty = memnew(StyleBoxEmpty);
	const Ref<StyleBox> spinner_hover = get_theme_stylebox(SNAME("button_hover"), SNAME("RichTextEditToolbar"));

	p_spin->add_theme_constant_override("buttons_width", NUMBER_FIELD_BUTTONS_WIDTH);
	p_spin->add_theme_constant_override("field_and_buttons_separation", 1);
	p_spin->add_theme_constant_override("buttons_vertical_separation", 1);
	p_spin->add_theme_constant_override("set_min_buttons_width_from_icons", 0);
	if (separator.is_valid()) {
		p_spin->add_theme_style_override("field_and_buttons_separator", separator);
		p_spin->add_theme_style_override("up_down_buttons_separator", separator);
	}
	p_spin->add_theme_style_override("up_background", empty);
	p_spin->add_theme_style_override("down_background", empty);
	p_spin->add_theme_style_override("up_background_disabled", empty);
	p_spin->add_theme_style_override("down_background_disabled", empty);
	if (spinner_hover.is_valid()) {
		p_spin->add_theme_style_override("up_background_hovered", spinner_hover);
		p_spin->add_theme_style_override("down_background_hovered", spinner_hover);
		p_spin->add_theme_style_override("up_background_pressed", spinner_hover);
		p_spin->add_theme_style_override("down_background_pressed", spinner_hover);
	}
	p_spin->add_theme_color_override("up_icon_modulate", muted_color);
	p_spin->add_theme_color_override("down_icon_modulate", muted_color);
	p_spin->add_theme_color_override("up_hover_icon_modulate", font_color);
	p_spin->add_theme_color_override("down_hover_icon_modulate", font_color);
	p_spin->add_theme_color_override("up_pressed_icon_modulate", font_color);
	p_spin->add_theme_color_override("down_pressed_icon_modulate", font_color);

	_style_line_edit(p_spin->get_line_edit(), StringName());
	p_spin->get_line_edit()->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	p_spin->get_line_edit()->add_theme_constant_override("minimum_character_width", NUMBER_FIELD_MIN_CHARS);
}

void RichTextEditToolbar::_style_line_edit(LineEdit *p_line_edit, const StringName &p_normal) {
	if (p_line_edit == nullptr) {
		return;
	}

	const Ref<StyleBox> normal = p_normal == StringName() ? Ref<StyleBox>(memnew(StyleBoxEmpty)) : get_theme_stylebox(p_normal, SNAME("RichTextEditToolbar"));
	const Ref<StyleBox> focus = get_theme_stylebox(SNAME("field_focus"), SNAME("RichTextEditToolbar"));
	if (normal.is_valid()) {
		p_line_edit->add_theme_style_override(CoreStringName(normal), normal);
		p_line_edit->add_theme_style_override("read_only", normal);
	}
	if (focus.is_valid()) {
		p_line_edit->add_theme_style_override("focus", focus);
	}

	const int font_size = get_theme_font_size(SNAME("input_font_size"), SNAME("RichTextEditToolbar"));
	if (font_size > 0) {
		p_line_edit->add_theme_font_size_override(SceneStringName(font_size), font_size);
	}

	const Color font_color = get_theme_color(SNAME("dropdown_font_color"), SNAME("RichTextEditToolbar"));
	const Color muted_color = get_theme_color(SNAME("muted_font_color"), SNAME("RichTextEditToolbar"));
	const Color accent_color = get_theme_color(SNAME("accent_color"), SNAME("RichTextEditToolbar"));
	p_line_edit->add_theme_color_override(SceneStringName(font_color), font_color);
	p_line_edit->add_theme_color_override("font_placeholder_color", muted_color);
	p_line_edit->add_theme_color_override("caret_color", accent_color);
}

void RichTextEditToolbar::_style_popup(PopupPanel *p_popup, const StringName &p_panel) {
	if (p_popup == nullptr) {
		return;
	}

	const Ref<StyleBox> panel_style = get_theme_stylebox(p_panel, SNAME("RichTextEditToolbar"));
	if (panel_style.is_valid()) {
		p_popup->add_theme_style_override(SceneStringName(panel), panel_style);
	}
}

void RichTextEditToolbar::_style_dropdown_label(Label *p_label) {
	if (p_label == nullptr) {
		return;
	}

	p_label->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("dropdown_label_color"), SNAME("RichTextEditToolbar")));
	const int font_size = get_theme_font_size(SNAME("dropdown_label_font_size"), SNAME("RichTextEditToolbar"));
	if (font_size > 0) {
		p_label->add_theme_font_size_override(SceneStringName(font_size), font_size);
	}
}

void RichTextEditToolbar::_apply_color_button_previews() {
	const Color border_color = get_theme_color(SNAME("muted_font_color"), SNAME("RichTextEditToolbar"));

	if (font_color_bar != nullptr) {
		Ref<StyleBoxFlat> bar;
		bar.instantiate();
		bar->set_bg_color(font_color_value);
		bar->set_corner_radius_all(1);
		font_color_bar->add_theme_style_override(SceneStringName(panel), bar);
	}

	if (outline_color_bar != nullptr) {
		Ref<StyleBoxFlat> bar;
		bar.instantiate();
		bar->set_bg_color(outline_color_value);
		bar->set_corner_radius_all(1);
		outline_color_bar->add_theme_style_override(SceneStringName(panel), bar);
	}

	if (bg_color_swatch != nullptr) {
		Ref<StyleBoxFlat> swatch;
		swatch.instantiate();
		swatch->set_bg_color(bg_color_assigned ? bg_color_value : Color(0, 0, 0, 0));
		swatch->set_draw_center(bg_color_assigned);
		swatch->set_border_width_all(1);
		swatch->set_border_color(Color(border_color, 0.6));
		swatch->set_corner_radius_all(1);
		bg_color_swatch->add_theme_style_override(SceneStringName(panel), swatch);
	}
	if (bg_color_none_button != nullptr) {
		bg_color_none_button->set_pressed_no_signal(!bg_color_assigned);
	}
	if (bg_color_checker != nullptr) {
		bg_color_checker->set_visible(!bg_color_assigned);
		if (has_theme_icon(SNAME("color_checkerboard"), SNAME("RichTextEditToolbar"))) {
			bg_color_checker->set_texture(get_theme_icon(SNAME("color_checkerboard"), SNAME("RichTextEditToolbar")));
		}
	}
}

void RichTextEditToolbar::_apply_toolbar_icons() {
	_set_button_icon(bold_button, SNAME("bold"));
	_set_button_icon(italic_button, SNAME("italic"));
	_set_button_icon(underline_button, SNAME("underline"));
	_set_button_icon(strikethrough_button, SNAME("strikethrough"));
	_set_button_icon(overline_button, SNAME("overline"));
	_set_button_icon(quote_button, SNAME("quote"));
	_set_button_icon(horizontal_rule_button, SNAME("horizontal_rule"));
	_set_button_icon(indent_decrease_button, SNAME("indent_decrease"));
	_set_button_icon(indent_increase_button, SNAME("indent_increase"));
	_set_button_icon(ordered_list_button, SNAME("ordered_list"));
	_set_button_icon(unordered_list_button, SNAME("unordered_list"));
	_set_button_icon(link_button, SNAME("link"));
	_set_button_icon(insert_image_button, SNAME("image"));
	_set_button_icon(line_height_button, SNAME("line_height"));
	_set_button_icon(outline_button, SNAME("outline"));
	_set_button_icon(font_color_button, SNAME("font_color"));
	_set_button_icon(bg_color_button, SNAME("background_color"));
	_set_button_icon(align_left_button, SNAME("align_left"));
	_set_button_icon(align_center_button, SNAME("align_center"));
	_set_button_icon(align_right_button, SNAME("align_right"));
	_set_button_icon(alignment_button, SNAME("align_left"));

	const Ref<Texture2D> caret = has_theme_icon(SNAME("dropdown_caret"), SNAME("RichTextEditToolbar")) ? get_theme_icon(SNAME("dropdown_caret"), SNAME("RichTextEditToolbar")) : Ref<Texture2D>();
	const Color muted_color = get_theme_color(SNAME("muted_font_color"), SNAME("RichTextEditToolbar"));
	for (TextureRect *dropdown_caret : dropdown_carets) {
		dropdown_caret->set_texture(caret);
		dropdown_caret->set_modulate(muted_color);
	}
}

void RichTextEditToolbar::_apply_row_layout() {
	if (first_row == nullptr || second_row == nullptr) {
		return;
	}
	// Detach the whole tail before reattaching it so the controls keep their
	// authored order whichever row they end up on.
	HBoxContainer *row = two_row_layout ? second_row : first_row;
	for (Control *item : second_row_items) {
		if (Node *parent = item->get_parent()) {
			parent->remove_child(item);
		}
	}
	for (Control *item : second_row_items) {
		row->add_child(item, false, INTERNAL_MODE_FRONT);
	}
	second_row->set_visible(two_row_layout);
	// In two-row layout the row break itself separates the two groups.
	if (row_break_separator != nullptr) {
		row_break_separator->set_visible(!two_row_layout);
	}
	// The minimum height depends on the row count. Outside the tree the ready
	// notification applies the style anyway, and theme lookups are not usable.
	if (is_inside_tree()) {
		_apply_toolbar_style();
	}
}

void RichTextEditToolbar::_apply_toolbar_style() {
	const Ref<StyleBox> panel = get_theme_stylebox(SceneStringName(panel), SNAME("RichTextEditToolbar"));
	const int button_size = get_theme_constant(SNAME("button_size"), SNAME("RichTextEditToolbar"));
	const int separation = get_theme_constant(SNAME("separation"));
	if (rows_box != nullptr) {
		rows_box->add_theme_constant_override(SNAME("separation"), separation);
	}
	if (panel.is_valid()) {
		const int rows = two_row_layout ? 2 : 1;
		const int rows_height = button_size * rows + separation * (rows - 1);
		set_custom_minimum_size(Size2(0, rows_height + panel->get_margin(SIDE_TOP) + panel->get_margin(SIDE_BOTTOM)));
		// HBoxContainer does not inset for the background it draws, so the
		// panel's horizontal padding is reproduced with spacers.
		if (paddings.size() == 2) {
			paddings[0]->set_custom_minimum_size(Size2(MAX(0, panel->get_margin(SIDE_LEFT) - separation), 0));
			paddings[1]->set_custom_minimum_size(Size2(MAX(0, panel->get_margin(SIDE_RIGHT) - separation), 0));
		}
	}

	_style_tool_button(bold_button);
	_style_tool_button(italic_button);
	_style_tool_button(underline_button);
	_style_tool_button(strikethrough_button);
	_style_tool_button(overline_button);
	_style_tool_button(quote_button);
	_style_tool_button(horizontal_rule_button);
	_style_tool_button(indent_decrease_button);
	_style_tool_button(indent_increase_button);
	_style_tool_button(ordered_list_button);
	_style_tool_button(unordered_list_button);
	_style_tool_button(font_color_button);
	_style_tool_button(bg_color_button);
	_style_tool_button(outline_button);
	_style_tool_button(insert_image_button);
	_style_dropdown_button(alignment_button);
	_style_dropdown_button(link_button);
	_style_dropdown_button(line_height_button);

	_style_menu_item_button(align_left_button);
	_style_menu_item_button(align_center_button);
	_style_menu_item_button(align_right_button);
	// Null until the background color picker has been opened at least once.
	_style_none_button();

	_style_popup(outline_color_popup, SNAME("dropdown_panel"));
	_style_popup(link_popup, SNAME("dropdown_panel"));
	_style_popup(alignment_popup, SNAME("menu_panel"));
	_style_popup(font_color_popup, SNAME("dropdown_panel"));
	_style_popup(bg_color_popup, SNAME("dropdown_panel"));
	_style_popup(line_height_popup, SNAME("dropdown_panel"));

	_style_number_field(font_size_field, font_size_spin);
	_style_number_field(outline_size_field, outline_size_spin);
	_style_line_edit(link_line_edit, SNAME("text_field_panel"));
	_style_line_edit(line_height_line_edit, SNAME("text_field_panel"));

	_style_dropdown_label(link_url_label);
	_style_dropdown_label(line_height_presets_label);
	_style_dropdown_label(line_height_custom_label);
	for (Button *preset_button : line_height_preset_buttons) {
		_style_menu_item_button(preset_button);
	}

	// Cancel brightens its muted label on hover; Apply is already accent-filled.
	const Color dropdown_font_color = get_theme_color(SNAME("dropdown_font_color"), SNAME("RichTextEditToolbar"));
	const Color muted_font_color = get_theme_color(SNAME("muted_font_color"), SNAME("RichTextEditToolbar"));
	const Color apply_font_color = get_theme_color(SNAME("apply_button_font_color"), SNAME("RichTextEditToolbar"));
	_style_action_button(link_cancel_button, SNAME("cancel_button_normal"), SNAME("cancel_button_hover"), muted_font_color, dropdown_font_color);
	_style_action_button(link_apply_button, SNAME("apply_button_normal"), SNAME("apply_button_hover"), apply_font_color, apply_font_color);
	_style_action_button(line_height_cancel_button, SNAME("cancel_button_normal"), SNAME("cancel_button_hover"), muted_font_color, dropdown_font_color);
	_style_action_button(line_height_apply_button, SNAME("apply_button_normal"), SNAME("apply_button_hover"), apply_font_color, apply_font_color);

	const Ref<StyleBox> separator = get_theme_stylebox(SNAME("separator"), SNAME("RichTextEditToolbar"));
	const int separator_height = get_theme_constant(SNAME("separator_height"), SNAME("RichTextEditToolbar"));
	for (VSeparator *toolbar_separator : separators) {
		if (separator.is_valid()) {
			toolbar_separator->add_theme_style_override(SNAME("separator"), separator);
		}
		toolbar_separator->add_theme_constant_override("separation", 1);
		toolbar_separator->set_custom_minimum_size(Size2(1, separator_height));
		toolbar_separator->set_v_size_flags(SIZE_SHRINK_CENTER);
	}

	_apply_color_button_previews();

	update_minimum_size();
	// The toolbar is fit-content: the panel has to end at the last control, so
	// shrink back to the new minimum instead of keeping a width that was sized
	// for a different row count or button size. A container re-fits its child
	// on the next layout pass anyway, so this only matters when the toolbar is
	// placed freely.
	reset_size();
	queue_redraw();
}

// ColorPicker is expensive to build, so both pickers are created on first use.
void RichTextEditToolbar::_ensure_font_color_picker() {
	if (font_color_popup != nullptr) {
		return;
	}

	font_color_popup = memnew(PopupPanel);
	font_color_picker = memnew(ColorPicker);
	font_color_picker->set_edit_alpha(false);
	font_color_popup->add_child(font_color_picker, false, INTERNAL_MODE_FRONT);
	add_child(font_color_popup, false, INTERNAL_MODE_FRONT);
	font_color_picker->connect("color_changed", callable_mp(this, &RichTextEditToolbar::_color_changed));
	font_color_popup->connect(SNAME("about_to_popup"), callable_mp(this, &RichTextEditToolbar::_color_popup_about_to_popup));
	font_color_popup->connect(SNAME("popup_hide"), callable_mp(this, &RichTextEditToolbar::_color_popup_closed));
	_style_popup(font_color_popup, SNAME("dropdown_panel"));
}

void RichTextEditToolbar::_ensure_bg_color_picker() {
	if (bg_color_popup != nullptr) {
		return;
	}

	bg_color_popup = memnew(PopupPanel);
	VBoxContainer *bg_color_box = memnew(VBoxContainer);
	bg_color_popup->add_child(bg_color_box, false, INTERNAL_MODE_FRONT);

	// Stands in for the design's "no color" palette tile, the palette grid having
	// been replaced by the engine's ColorPicker. It leads the dropdown so it is
	// the first thing seen, and shows the selected state while no color is set.
	bg_color_none_button = memnew(RichTextEditToolbarButton);
	bg_color_none_button->set_text(RTR("No Color"));
	bg_color_none_button->set_tooltip_text(RTR("Remove the background color"));
	bg_color_none_button->set_focus_mode(FOCUS_ALL);
	bg_color_none_button->set_toggle_mode(true);
	bg_color_box->add_child(bg_color_none_button, false, INTERNAL_MODE_FRONT);
	bg_color_none_swatch = memnew(Panel);
	bg_color_none_swatch->set_mouse_filter(MOUSE_FILTER_IGNORE);
	bg_color_none_swatch->set_anchors_preset(PRESET_CENTER_LEFT);
	bg_color_none_swatch->set_offset(SIDE_LEFT, NONE_SWATCH_MARGIN);
	bg_color_none_swatch->set_offset(SIDE_RIGHT, NONE_SWATCH_MARGIN + NONE_SWATCH_SIZE);
	bg_color_none_swatch->set_offset(SIDE_TOP, -NONE_SWATCH_SIZE / 2);
	bg_color_none_swatch->set_offset(SIDE_BOTTOM, NONE_SWATCH_SIZE / 2);
	bg_color_none_button->add_child(bg_color_none_swatch, false, INTERNAL_MODE_FRONT);
	bg_color_none_checker = memnew(TextureRect);
	bg_color_none_checker->set_mouse_filter(MOUSE_FILTER_IGNORE);
	bg_color_none_checker->set_stretch_mode(TextureRect::STRETCH_TILE);
	bg_color_none_checker->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	bg_color_none_checker->set_anchors_preset(PRESET_FULL_RECT);
	bg_color_none_checker->set_offset(SIDE_LEFT, 1);
	bg_color_none_checker->set_offset(SIDE_TOP, 1);
	bg_color_none_checker->set_offset(SIDE_RIGHT, -1);
	bg_color_none_checker->set_offset(SIDE_BOTTOM, -1);
	bg_color_none_swatch->add_child(bg_color_none_checker, false, INTERNAL_MODE_FRONT);

	bg_color_none_separator = memnew(HSeparator);
	bg_color_box->add_child(bg_color_none_separator, false, INTERNAL_MODE_FRONT);

	bg_color_picker = memnew(ColorPicker);
	bg_color_picker->set_edit_alpha(false);
	bg_color_box->add_child(bg_color_picker, false, INTERNAL_MODE_FRONT);
	add_child(bg_color_popup, false, INTERNAL_MODE_FRONT);
	bg_color_picker->connect("color_changed", callable_mp(this, &RichTextEditToolbar::_bg_color_changed));
	bg_color_popup->connect(SNAME("about_to_popup"), callable_mp(this, &RichTextEditToolbar::_bg_color_popup_about_to_popup));
	bg_color_popup->connect(SNAME("popup_hide"), callable_mp(this, &RichTextEditToolbar::_bg_color_popup_closed));
	bg_color_none_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_bg_color_cleared));
	_style_popup(bg_color_popup, SNAME("dropdown_panel"));
	_style_none_button();
	// The row was just built, so give it the current "no color" selected state.
	_apply_color_button_previews();
}

void RichTextEditToolbar::_ensure_outline_color_picker() {
	if (outline_color_popup != nullptr) {
		return;
	}

	outline_color_popup = memnew(PopupPanel);
	outline_color_picker = memnew(ColorPicker);
	outline_color_picker->set_edit_alpha(false);
	outline_color_popup->add_child(outline_color_picker, false, INTERNAL_MODE_FRONT);
	add_child(outline_color_popup, false, INTERNAL_MODE_FRONT);
	outline_color_picker->connect("color_changed", callable_mp(this, &RichTextEditToolbar::_outline_color_changed));
	outline_color_popup->connect(SNAME("popup_hide"), callable_mp(this, &RichTextEditToolbar::_outline_color_popup_closed));
	_style_popup(outline_color_popup, SNAME("dropdown_panel"));
}

Button *RichTextEditToolbar::_get_dropdown_button(int p_dropdown) const {
	switch (p_dropdown) {
		case DROPDOWN_FONT_COLOR:
			return font_color_button;
		case DROPDOWN_BG_COLOR:
			return bg_color_button;
		case DROPDOWN_OUTLINE_COLOR:
			return outline_button;
		case DROPDOWN_ALIGNMENT:
			return alignment_button;
		case DROPDOWN_LINK:
			return link_button;
		case DROPDOWN_LINE_HEIGHT:
			return line_height_button;
	}
	return nullptr;
}

PopupPanel *RichTextEditToolbar::_get_dropdown_popup(int p_dropdown) const {
	switch (p_dropdown) {
		case DROPDOWN_FONT_COLOR:
			return font_color_popup;
		case DROPDOWN_BG_COLOR:
			return bg_color_popup;
		case DROPDOWN_OUTLINE_COLOR:
			return outline_color_popup;
		case DROPDOWN_ALIGNMENT:
			return alignment_popup;
		case DROPDOWN_LINK:
			return link_popup;
		case DROPDOWN_LINE_HEIGHT:
			return line_height_popup;
	}
	return nullptr;
}

void RichTextEditToolbar::_dropdown_button_down(int p_dropdown) {
	const PopupPanel *popup = _get_dropdown_popup(p_dropdown);
	const bool was_open = popup != nullptr && popup->is_visible();
	switch (p_dropdown) {
		case DROPDOWN_FONT_COLOR:
			font_color_popup_was_open = was_open;
			break;
		case DROPDOWN_BG_COLOR:
			bg_color_popup_was_open = was_open;
			break;
		case DROPDOWN_OUTLINE_COLOR:
			outline_color_popup_was_open = was_open;
			break;
		case DROPDOWN_ALIGNMENT:
			alignment_popup_was_open = was_open;
			break;
		case DROPDOWN_LINK:
			link_popup_was_open = was_open;
			break;
		case DROPDOWN_LINE_HEIGHT:
			line_height_popup_was_open = was_open;
			break;
	}
}

void RichTextEditToolbar::_dropdown_popup_hidden(int p_dropdown) {
	if (Button *button = _get_dropdown_button(p_dropdown)) {
		button->set_pressed_no_signal(false);
		// A dropdown button that keeps keyboard focus after its popup closes
		// still looks active and takes the next click as a plain focus click.
		button->release_focus();
	}
}

void RichTextEditToolbar::_popup_below(PopupPanel *p_popup, Button *p_button, int p_min_width) {
	if (p_popup == nullptr || p_button == nullptr) {
		return;
	}

	p_popup->set_min_size(Size2(p_min_width, 0));
	p_popup->reset_size();
	p_popup->set_position(p_button->get_screen_position() + Vector2(0, p_button->get_size().y + DROPDOWN_GAP));
	p_popup->popup();
}

void RichTextEditToolbar::_popup_below_right(PopupPanel *p_popup, Button *p_button, int p_min_width) {
	if (p_popup == nullptr || p_button == nullptr) {
		return;
	}

	p_popup->set_min_size(Size2(p_min_width, 0));
	p_popup->reset_size();
	const Point2 button_position = p_button->get_screen_position();
	p_popup->set_position(button_position + Vector2(p_button->get_size().x - p_popup->get_size().x, p_button->get_size().y + DROPDOWN_GAP));
	p_popup->popup();
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

void RichTextEditToolbar::_pressed_overline() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_overline();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_quote() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_quote();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_horizontal_rule() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->insert_horizontal_rule();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_outline_color() {
	_ensure_outline_color_picker();
	if (outline_color_popup_was_open) {
		outline_color_popup_was_open = false;
		outline_button->set_pressed_no_signal(false);
		return;
	}
	if (!outline_button->is_pressed()) {
		outline_color_popup->hide();
		return;
	}
	outline_color_picker->set_pick_color(outline_color_value);
	_popup_below(outline_color_popup, outline_button, 0);
}

void RichTextEditToolbar::_pressed_alignment() {
	if (alignment_popup_was_open) {
		alignment_popup_was_open = false;
		alignment_button->set_pressed_no_signal(false);
		return;
	}
	if (!alignment_button->is_pressed()) {
		alignment_popup->hide();
		return;
	}
	_popup_below(alignment_popup, alignment_button, ALIGNMENT_PANEL_WIDTH);
}

void RichTextEditToolbar::_pressed_link() {
	if (link_popup_was_open) {
		link_popup_was_open = false;
		link_button->set_pressed_no_signal(false);
		return;
	}
	if (!link_button->is_pressed()) {
		link_popup->hide();
		return;
	}
	_popup_below(link_popup, link_button, LINK_PANEL_WIDTH);
	link_line_edit->grab_focus();
	link_line_edit->select_all();
}

void RichTextEditToolbar::_pressed_insert_image() {
	if (image_file_dialog == nullptr) {
		image_file_dialog = memnew(FileDialog);
		image_file_dialog->set_file_mode(FileDialog::FILE_MODE_OPEN_FILE);
		image_file_dialog->set_access(FileDialog::ACCESS_FILESYSTEM);
		image_file_dialog->set_use_native_dialog(true);
		image_file_dialog->set_title(RTR("Insert Image"));
		Vector<String> filters;
		filters.push_back("*.png,*.jpg,*.jpeg,*.gif,*.bmp,*.webp,*.svg ; " + RTR("Images"));
		image_file_dialog->set_filters(filters);
		add_child(image_file_dialog, false, INTERNAL_MODE_FRONT);
		image_file_dialog->connect(SNAME("file_selected"), callable_mp(this, &RichTextEditToolbar::_image_file_selected));
	}
	image_file_dialog->popup_file_dialog();
}

void RichTextEditToolbar::_image_file_selected(const String &p_path) {
	RichTextEdit *target = _get_rich_text_edit();
	if (target == nullptr || p_path.is_empty()) {
		return;
	}

	// FileDialog::ACCESS_FILESYSTEM intentionally returns an absolute path.
	// Localize only files inside this project; external files remain absolute
	// so no copy or import step is introduced.
	String source = p_path;
	const String localized = ProjectSettings::get_singleton()->localize_path(p_path);
	if (localized.begins_with("res://")) {
		source = localized;
	}
	target->insert_image(source, -1, -1, String());
	target->grab_focus();
}

void RichTextEditToolbar::_pressed_line_height() {
	if (line_height_popup_was_open) {
		// The click that reached the button had already dismissed the popup, and
		// the toggle it performed on the way has to be undone.
		line_height_popup_was_open = false;
		line_height_button->set_pressed_no_signal(false);
		return;
	}
	line_height_button->set_pressed_no_signal(true);
	if (RichTextEdit *target = _get_rich_text_edit()) {
		const String line_height = target->get_current_line_height();
		line_height_line_edit->set_text(line_height == "normal" ? "1" : line_height);
	}
	_popup_below_right(line_height_popup, line_height_button, LINE_HEIGHT_PANEL_WIDTH);
	line_height_line_edit->grab_focus();
	line_height_line_edit->select_all();
}

void RichTextEditToolbar::_line_height_preset_selected(const String &p_value) {
	if (line_height_popup != nullptr) {
		line_height_popup->hide();
	}
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->set_selection_line_height(p_value);
		target->grab_focus();
	}
}

void RichTextEditToolbar::_line_height_apply_pressed() {
	const String value = line_height_line_edit != nullptr ? line_height_line_edit->get_text().strip_edges() : String();
	if (!RichTextDocument::is_valid_line_height(value)) {
		if (line_height_line_edit != nullptr) {
			line_height_line_edit->grab_focus();
		}
		return;
	}
	if (line_height_popup != nullptr) {
		line_height_popup->hide();
	}
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->set_selection_line_height(value);
		target->grab_focus();
	}
}

void RichTextEditToolbar::_line_height_cancel_pressed() {
	if (line_height_popup != nullptr) {
		line_height_popup->hide();
	}
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_font_color() {
	_ensure_font_color_picker();
	if (font_color_popup_was_open) {
		font_color_popup_was_open = false;
		font_color_button->set_pressed_no_signal(false);
		return;
	}
	if (!font_color_button->is_pressed()) {
		font_color_popup->hide();
		return;
	}
	font_color_picker->set_pick_color(font_color_value);
	_popup_below(font_color_popup, font_color_button, 0);
}

void RichTextEditToolbar::_pressed_bg_color() {
	_ensure_bg_color_picker();
	if (bg_color_popup_was_open) {
		bg_color_popup_was_open = false;
		bg_color_button->set_pressed_no_signal(false);
		return;
	}
	if (!bg_color_button->is_pressed()) {
		bg_color_popup->hide();
		return;
	}
	bg_color_picker->set_pick_color(bg_color_value);
	_popup_below(bg_color_popup, bg_color_button, 0);
}

void RichTextEditToolbar::_link_apply_pressed() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		const String url = link_line_edit != nullptr ? link_line_edit->get_text().strip_edges() : String();
		if (url.is_empty()) {
			target->clear_selection_url();
		} else {
			target->set_selection_url(url);
		}
		target->grab_focus();
	}
	if (link_popup != nullptr) {
		link_popup->hide();
	}
}

void RichTextEditToolbar::_link_cancel_pressed() {
	if (link_line_edit != nullptr) {
		link_line_edit->clear();
	}
	if (link_popup != nullptr) {
		link_popup->hide();
	}
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->grab_focus();
	}
}

void RichTextEditToolbar::_alignment_selected(int p_id) {
	RichTextEdit *target = _get_rich_text_edit();
	if (alignment_popup != nullptr) {
		alignment_popup->hide();
	}
	if (target == nullptr) {
		_update_controls_from_target();
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
	target->grab_focus();
}

void RichTextEditToolbar::_pressed_unordered_list() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_unordered_list();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_pressed_ordered_list() {
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->toggle_ordered_list();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_color_popup_about_to_popup() {
	color_picker_open = true;
	color_before_picker_open = font_color_value;
	pending_picker_color = color_before_picker_open;
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->begin_selection_color_preview();
	}
}

void RichTextEditToolbar::_color_popup_closed() {
	font_color_button->set_pressed_no_signal(false);
	if (!color_picker_open) {
		return;
	}

	color_picker_open = false;
	const Color final_color = font_color_picker->get_pick_color();
	pending_picker_color = final_color;
	font_color_value = final_color;
	_apply_color_button_previews();
	const bool commit_color = !final_color.is_equal_approx(color_before_picker_open);

	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->end_selection_color_preview(commit_color);
		target->grab_focus();
	}
}

void RichTextEditToolbar::_color_changed(const Color &p_color) {
	pending_picker_color = p_color;
	font_color_value = p_color;
	_apply_color_button_previews();
	if (color_picker_open) {
		if (RichTextEdit *target = _get_rich_text_edit()) {
			target->preview_selection_color(p_color);
		}
	}
}

void RichTextEditToolbar::_bg_color_popup_about_to_popup() {
	bg_color_picker_open = true;
	bg_color_before_picker_open = bg_color_value;
	bg_color_assigned_before_picker_open = bg_color_assigned;
	pending_bg_picker_color = bg_color_before_picker_open;
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->begin_selection_bg_color_preview();
	}
}

void RichTextEditToolbar::_bg_color_popup_closed() {
	bg_color_button->set_pressed_no_signal(false);
	if (!bg_color_picker_open) {
		return;
	}

	bg_color_picker_open = false;
	const Color final_color = bg_color_picker->get_pick_color();
	pending_bg_picker_color = final_color;
	// Going from "no color" to a color must commit even when that color happens
	// to match the swatch's last remembered value.
	const bool commit_color = bg_color_assigned && (!bg_color_assigned_before_picker_open || !final_color.is_equal_approx(bg_color_before_picker_open));

	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->end_selection_bg_color_preview(commit_color);
		target->grab_focus();
	}
}

void RichTextEditToolbar::_bg_color_changed(const Color &p_color) {
	pending_bg_picker_color = p_color;
	bg_color_value = p_color;
	bg_color_assigned = true;
	_apply_color_button_previews();
	if (bg_color_picker_open) {
		if (RichTextEdit *target = _get_rich_text_edit()) {
			target->preview_selection_bg_color(p_color);
		}
	}
}

void RichTextEditToolbar::_bg_color_cleared() {
	bg_color_assigned = false;
	_apply_color_button_previews();
	if (bg_color_popup != nullptr) {
		bg_color_popup->hide();
	}
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->clear_selection_bg_color();
		target->grab_focus();
	}
}

void RichTextEditToolbar::_outline_color_changed(const Color &p_color) {
	outline_color_value = p_color;
	_apply_color_button_previews();
	if (updating_controls) {
		return;
	}
	// Successive changes coalesce into one undo step, so dragging the picker
	// does not fill the history with intermediate colors.
	if (RichTextEdit *target = _get_rich_text_edit()) {
		target->set_selection_outline_color(p_color);
		target->grab_focus();
	}
}

void RichTextEditToolbar::_outline_color_popup_closed() {
	outline_button->set_pressed_no_signal(false);
	if (RichTextEdit *target = _get_rich_text_edit()) {
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

void RichTextEditToolbar::set_two_row_layout(bool p_enabled) {
	if (two_row_layout == p_enabled) {
		return;
	}
	two_row_layout = p_enabled;
	_apply_row_layout();
}

bool RichTextEditToolbar::is_two_row_layout() const {
	return two_row_layout;
}

RichTextEdit *RichTextEditToolbar::get_rich_text_edit() const {
	return _get_rich_text_edit();
}

void RichTextEditToolbar::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_rich_text_edit_path", "path"), &RichTextEditToolbar::set_rich_text_edit_path);
	ClassDB::bind_method(D_METHOD("get_rich_text_edit_path"), &RichTextEditToolbar::get_rich_text_edit_path);
	ClassDB::bind_method(D_METHOD("get_rich_text_edit"), &RichTextEditToolbar::get_rich_text_edit);
	ClassDB::bind_method(D_METHOD("set_two_row_layout", "enabled"), &RichTextEditToolbar::set_two_row_layout);
	ClassDB::bind_method(D_METHOD("is_two_row_layout"), &RichTextEditToolbar::is_two_row_layout);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "rich_text_edit_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, RichTextEdit::get_class_static()), "set_rich_text_edit_path", "get_rich_text_edit_path");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "two_row_layout"), "set_two_row_layout", "is_two_row_layout");

	auto update_icons = [](Node *p_instance, const StringName &, const StringName &) {
		RichTextEditToolbar *toolbar = Object::cast_to<RichTextEditToolbar>(p_instance);
		if (toolbar != nullptr) {
			toolbar->_apply_toolbar_icons();
		}
	};
	const char *icon_items[] = {
		"bold", "italic", "underline", "strikethrough", "overline", "quote", "align",
		"align_left", "align_center", "align_right", "indent_decrease",
		"indent_increase", "link", "ordered_list", "unordered_list",
		"font_color", "background_color", "outline", "dropdown_caret",
		"color_checkerboard", "image", "line_height"
	};
	for (const char *icon_item : icon_items) {
		ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_ICON, get_class_static(), icon_item, icon_item, update_icons);
	}

	auto update_style = [](Node *p_instance, const StringName &, const StringName &) {
		RichTextEditToolbar *toolbar = Object::cast_to<RichTextEditToolbar>(p_instance);
		if (toolbar != nullptr) {
			toolbar->_apply_toolbar_style();
		}
	};
	const char *stylebox_items[] = {
		"panel", "dropdown_panel", "menu_panel", "button_normal", "button_hover",
		"button_pressed", "button_focus", "menu_item_normal", "menu_item_hover",
		"menu_item_pressed", "field_panel", "text_field_panel", "field_focus",
		"field_separator", "cancel_button_normal", "cancel_button_hover",
		"apply_button_normal", "apply_button_hover", "separator", "menu_separator",
		"tooltip_panel"
	};
	for (const char *stylebox_item : stylebox_items) {
		ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_STYLEBOX, get_class_static(), stylebox_item, stylebox_item, update_style);
	}
	const char *color_items[] = {
		"accent_color", "button_font_color", "muted_font_color", "dropdown_font_color",
		"dropdown_label_color", "apply_button_font_color", "tooltip_font_color"
	};
	for (const char *color_item : color_items) {
		ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_COLOR, get_class_static(), color_item, color_item, update_style);
	}
	const char *font_size_items[] = {
		"dropdown_label_font_size", "input_font_size", "menu_item_font_size",
		"action_button_font_size", "tooltip_font_size"
	};
	for (const char *font_size_item : font_size_items) {
		ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_FONT_SIZE, get_class_static(), font_size_item, font_size_item, update_style);
	}
	const char *constant_items[] = { "button_size", "icon_max_width", "separator_height" };
	for (const char *constant_item : constant_items) {
		ThemeDB::get_singleton()->bind_class_item(Theme::DATA_TYPE_CONSTANT, get_class_static(), constant_item, constant_item, update_style);
	}

	ADD_CLASS_DEPENDENCY("Button");
	ADD_CLASS_DEPENDENCY("ColorPicker");
	ADD_CLASS_DEPENDENCY("LineEdit");
	ADD_CLASS_DEPENDENCY("PopupPanel");
	ADD_CLASS_DEPENDENCY("SpinBox");
	ADD_CLASS_DEPENDENCY("FileDialog");
}

RichTextEditToolbar::RichTextEditToolbar() {
	set_custom_minimum_size(Size2(0, DEFAULT_BUTTON_SIZE + 12));
	// The design's toolbar is a fit-content row, so the panel ends at the last
	// button instead of stretching across its parent.
	set_h_size_flags(SIZE_SHRINK_BEGIN);

	// Horizontal padding matching the toolbar panel's content margins is added
	// as spacers because HBoxContainer does not inset for its background.
	auto add_toolbar_padding = [&]() {
		Control *padding = memnew(Control);
		padding->set_mouse_filter(MOUSE_FILTER_IGNORE);
		padding->set_custom_minimum_size(Size2(8, 0));
		add_child(padding, false, INTERNAL_MODE_FRONT);
		paddings.push_back(padding);
	};

	auto add_toolbar_separator = [&]() {
		VSeparator *separator = memnew(VSeparator);
		separator->set_custom_minimum_size(Size2(1, DEFAULT_SEPARATOR_HEIGHT));
		separator->set_v_size_flags(SIZE_SHRINK_CENTER);
		first_row->add_child(separator, false, INTERNAL_MODE_FRONT);
		separators.push_back(separator);
		return separator;
	};

	// Buttons within one design group sit closer together than the groups do.
	auto add_group = [&]() {
		HBoxContainer *group = memnew(HBoxContainer);
		group->add_theme_constant_override("separation", GROUP_SEPARATION);
		first_row->add_child(group, false, INTERNAL_MODE_FRONT);
		return group;
	};

	auto make_button = [&](Node *p_parent, const String &p_tooltip, bool p_toggle) {
		RichTextEditToolbarButton *button = memnew(RichTextEditToolbarButton);
		button->set_focus_mode(FOCUS_ALL);
		button->set_toggle_mode(p_toggle);
		button->set_custom_minimum_size(Size2(DEFAULT_BUTTON_SIZE, DEFAULT_BUTTON_SIZE));
		button->set_v_size_flags(SIZE_SHRINK_CENTER);
		button->set_tooltip_text(p_tooltip);
		p_parent->add_child(button, false, INTERNAL_MODE_FRONT);
		return button;
	};

	// Adds the small caret that marks a button as opening a dropdown.
	auto add_dropdown_caret = [&](Button *p_button) {
		TextureRect *caret = memnew(TextureRect);
		caret->set_mouse_filter(MOUSE_FILTER_IGNORE);
		caret->set_stretch_mode(TextureRect::STRETCH_SCALE);
		// The texture must not contribute a minimum size, or the anchored rect
		// grows to the icon's native size and drifts out of the button.
		caret->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
		caret->set_anchors_preset(PRESET_CENTER_RIGHT);
		caret->set_offset(SIDE_RIGHT, -DROPDOWN_SIDE_PADDING);
		caret->set_offset(SIDE_LEFT, -DROPDOWN_SIDE_PADDING - CARET_SIZE);
		caret->set_offset(SIDE_TOP, -CARET_SIZE / 2);
		caret->set_offset(SIDE_BOTTOM, CARET_SIZE / 2);
		p_button->add_child(caret, false, INTERNAL_MODE_FRONT);
		dropdown_carets.push_back(caret);
	};

	auto make_number_field = [&](Node *p_parent, PanelContainer *&r_field, SpinBox *&r_spin) {
		r_field = memnew(PanelContainer);
		r_field->set_custom_minimum_size(Size2(NUMBER_FIELD_WIDTH, DEFAULT_BUTTON_SIZE));
		r_field->set_v_size_flags(SIZE_SHRINK_CENTER);
		r_field->set_mouse_filter(MOUSE_FILTER_PASS);
		p_parent->add_child(r_field, false, INTERNAL_MODE_FRONT);

		r_spin = memnew(SpinBox);
		r_spin->set_step(1);
		r_spin->set_mouse_filter(MOUSE_FILTER_PASS);
		r_field->add_child(r_spin, false, INTERNAL_MODE_FRONT);
		r_spin->get_line_edit()->set_mouse_filter(MOUSE_FILTER_PASS);
	};

	add_toolbar_padding();

	// Every control lives in a row so that switching to the two-row layout only
	// has to move the second half across, never rebuild it.
	rows_box = memnew(VBoxContainer);
	rows_box->set_v_size_flags(SIZE_SHRINK_CENTER);
	add_child(rows_box, false, INTERNAL_MODE_FRONT);
	first_row = memnew(HBoxContainer);
	rows_box->add_child(first_row, false, INTERNAL_MODE_FRONT);
	second_row = memnew(HBoxContainer);
	second_row->hide();
	rows_box->add_child(second_row, false, INTERNAL_MODE_FRONT);

	// Bold / Italic / Underline / Strikethrough / Overline.
	HBoxContainer *format_group = add_group();
	bold_button = make_button(format_group, RTR("Bold"), true);
	bold_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_bold));
	italic_button = make_button(format_group, RTR("Italic"), true);
	italic_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_italic));
	underline_button = make_button(format_group, RTR("Underline"), true);
	underline_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_underline));
	strikethrough_button = make_button(format_group, RTR("Strikethrough"), true);
	strikethrough_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_strikethrough));
	overline_button = make_button(format_group, RTR("Overline"), true);
	overline_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_overline));

	add_toolbar_separator();

	// Font size.
	make_number_field(first_row, font_size_field, font_size_spin);
	font_size_spin->set_min(1);
	font_size_spin->set_max(256);
	font_size_spin->set_value(16);
	font_size_spin->connect(SceneStringName(value_changed), callable_mp(this, &RichTextEditToolbar::_font_size_changed));

	add_toolbar_separator();

	// Font color / Background color.
	HBoxContainer *color_group = add_group();
	font_color_button = make_button(color_group, RTR("Font Color"), true);
	font_color_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_font_color));
	font_color_button->connect(SNAME("button_down"), callable_mp(this, &RichTextEditToolbar::_dropdown_button_down).bind(DROPDOWN_FONT_COLOR));
	font_color_bar = memnew(Panel);
	font_color_bar->set_mouse_filter(MOUSE_FILTER_IGNORE);
	font_color_bar->set_anchors_preset(PRESET_BOTTOM_WIDE);
	font_color_bar->set_offset(SIDE_LEFT, COLOR_PREVIEW_INSET);
	font_color_bar->set_offset(SIDE_RIGHT, -COLOR_PREVIEW_INSET);
	font_color_bar->set_offset(SIDE_TOP, -(COLOR_PREVIEW_BOTTOM + FONT_COLOR_BAR_HEIGHT));
	font_color_bar->set_offset(SIDE_BOTTOM, -COLOR_PREVIEW_BOTTOM);
	font_color_button->add_child(font_color_bar, false, INTERNAL_MODE_FRONT);

	bg_color_button = make_button(color_group, RTR("Background Color"), true);
	bg_color_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_bg_color));
	bg_color_button->connect(SNAME("button_down"), callable_mp(this, &RichTextEditToolbar::_dropdown_button_down).bind(DROPDOWN_BG_COLOR));
	bg_color_swatch = memnew(Panel);
	bg_color_swatch->set_mouse_filter(MOUSE_FILTER_IGNORE);
	bg_color_swatch->set_anchors_preset(PRESET_BOTTOM_WIDE);
	bg_color_swatch->set_offset(SIDE_LEFT, COLOR_PREVIEW_INSET);
	bg_color_swatch->set_offset(SIDE_RIGHT, -COLOR_PREVIEW_INSET);
	bg_color_swatch->set_offset(SIDE_TOP, -(COLOR_PREVIEW_BOTTOM + BG_COLOR_SWATCH_HEIGHT));
	bg_color_swatch->set_offset(SIDE_BOTTOM, -COLOR_PREVIEW_BOTTOM);
	bg_color_button->add_child(bg_color_swatch, false, INTERNAL_MODE_FRONT);
	bg_color_checker = memnew(TextureRect);
	bg_color_checker->set_mouse_filter(MOUSE_FILTER_IGNORE);
	bg_color_checker->set_stretch_mode(TextureRect::STRETCH_TILE);
	// Without this the checkerboard texture's size becomes the minimum size and
	// the swatch grows past the bottom of the button.
	bg_color_checker->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	bg_color_checker->set_anchors_preset(PRESET_FULL_RECT);
	bg_color_checker->set_offset(SIDE_LEFT, 1);
	bg_color_checker->set_offset(SIDE_TOP, 1);
	bg_color_checker->set_offset(SIDE_RIGHT, -1);
	bg_color_checker->set_offset(SIDE_BOTTOM, -1);
	bg_color_swatch->add_child(bg_color_checker, false, INTERNAL_MODE_FRONT);

	add_toolbar_separator();

	// Outline: a color button and a width field sitting directly on the toolbar.
	outline_button = make_button(first_row, RTR("Outline Color"), true);
	outline_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_outline_color));
	outline_button->connect(SNAME("button_down"), callable_mp(this, &RichTextEditToolbar::_dropdown_button_down).bind(DROPDOWN_OUTLINE_COLOR));

	outline_color_bar = memnew(Panel);
	outline_color_bar->set_mouse_filter(MOUSE_FILTER_IGNORE);
	outline_color_bar->set_anchors_preset(PRESET_BOTTOM_WIDE);
	outline_color_bar->set_offset(SIDE_LEFT, COLOR_PREVIEW_INSET);
	outline_color_bar->set_offset(SIDE_RIGHT, -COLOR_PREVIEW_INSET);
	outline_color_bar->set_offset(SIDE_TOP, -(COLOR_PREVIEW_BOTTOM + FONT_COLOR_BAR_HEIGHT));
	outline_color_bar->set_offset(SIDE_BOTTOM, -COLOR_PREVIEW_BOTTOM);
	outline_button->add_child(outline_color_bar, false, INTERNAL_MODE_FRONT);

	make_number_field(first_row, outline_size_field, outline_size_spin);
	outline_size_spin->set_min(0);
	outline_size_spin->set_max(32);
	outline_size_spin->set_value(0);
	outline_size_spin->connect(SceneStringName(value_changed), callable_mp(this, &RichTextEditToolbar::_outline_size_changed));

	// The design's two-row layout breaks here, so everything from this point on
	// is what moves to the second row.
	row_break_separator = add_toolbar_separator();
	const int second_row_start = first_row->get_child_count(true) - 1;

	// Horizontal rule: a one-shot insertion action, like image insertion.
	horizontal_rule_button = make_button(first_row, RTR("Horizontal Rule"), false);
	horizontal_rule_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_horizontal_rule));

	add_toolbar_separator();

	// Quote.
	quote_button = make_button(first_row, RTR("Quote"), true);
	quote_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_quote));

	add_toolbar_separator();

	// Paragraph alignment.
	alignment_button = make_button(first_row, RTR("Paragraph Alignment"), true);
	alignment_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_alignment));
	alignment_button->connect(SNAME("button_down"), callable_mp(this, &RichTextEditToolbar::_dropdown_button_down).bind(DROPDOWN_ALIGNMENT));
	add_dropdown_caret(alignment_button);

	alignment_popup = memnew(PopupPanel);
	VBoxContainer *alignment_box = memnew(VBoxContainer);
	alignment_box->add_theme_constant_override("separation", 2);
	alignment_popup->add_child(alignment_box, false, INTERNAL_MODE_FRONT);

	auto add_alignment_button = [&](Button *&r_button, const String &p_text, int p_alignment) {
		RichTextEditToolbarButton *button = memnew(RichTextEditToolbarButton);
		button->set_focus_mode(FOCUS_ALL);
		button->set_toggle_mode(true);
		button->set_text(p_text);
		alignment_box->add_child(button, false, INTERNAL_MODE_FRONT);
		button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_alignment_selected).bind(p_alignment));
		r_button = button;
	};

	add_alignment_button(align_left_button, RTR("Align Left"), ALIGNMENT_LEFT);
	add_alignment_button(align_center_button, RTR("Align Center"), ALIGNMENT_CENTER);
	add_alignment_button(align_right_button, RTR("Align Right"), ALIGNMENT_RIGHT);
	add_child(alignment_popup, false, INTERNAL_MODE_FRONT);

	add_toolbar_separator();

	// Indentation and lists.
	HBoxContainer *list_group = add_group();
	indent_decrease_button = make_button(list_group, RTR("Decrease Indent"), false);
	indent_decrease_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_decrease_indent));
	indent_increase_button = make_button(list_group, RTR("Increase Indent"), false);
	indent_increase_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_increase_indent));
	ordered_list_button = make_button(list_group, RTR("Ordered List"), true);
	ordered_list_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_ordered_list));
	unordered_list_button = make_button(list_group, RTR("Unordered List"), true);
	unordered_list_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_unordered_list));

	add_toolbar_separator();

	// URL link.
	link_button = make_button(first_row, RTR("Insert Link"), true);
	link_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_link));
	link_button->connect(SNAME("button_down"), callable_mp(this, &RichTextEditToolbar::_dropdown_button_down).bind(DROPDOWN_LINK));
	add_dropdown_caret(link_button);

	link_popup = memnew(PopupPanel);
	VBoxContainer *link_box = memnew(VBoxContainer);
	link_box->add_theme_constant_override("separation", 6);
	link_popup->add_child(link_box, false, INTERNAL_MODE_FRONT);
	link_url_label = memnew(Label(RTR("URL")));
	link_box->add_child(link_url_label, false, INTERNAL_MODE_FRONT);
	link_line_edit = memnew(LineEdit);
	link_line_edit->set_placeholder("https://example.com");
	link_line_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	link_box->add_child(link_line_edit, false, INTERNAL_MODE_FRONT);
	link_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &RichTextEditToolbar::_link_apply_pressed).unbind(1));

	HBoxContainer *link_actions = memnew(HBoxContainer);
	link_actions->add_theme_constant_override("separation", 6);
	link_actions->set_alignment(BoxContainer::ALIGNMENT_END);
	link_box->add_child(link_actions, false, INTERNAL_MODE_FRONT);
	link_cancel_button = memnew(RichTextEditToolbarButton);
	link_cancel_button->set_text(RTR("Cancel"));
	link_cancel_button->set_focus_mode(FOCUS_ALL);
	link_actions->add_child(link_cancel_button, false, INTERNAL_MODE_FRONT);
	link_cancel_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_link_cancel_pressed));
	link_apply_button = memnew(RichTextEditToolbarButton);
	link_apply_button->set_text(RTR("Apply"));
	link_apply_button->set_tooltip_text(RTR("Apply the link. Leave the field empty to remove an existing link."));
	link_apply_button->set_focus_mode(FOCUS_ALL);
	link_actions->add_child(link_apply_button, false, INTERNAL_MODE_FRONT);
	link_apply_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_link_apply_pressed));
	add_child(link_popup, false, INTERNAL_MODE_FRONT);

	add_toolbar_separator();

	// Image insertion opens the native file browser when the platform supports
	// it, while keeping the same compact toolbar button geometry as the design.
	insert_image_button = make_button(first_row, RTR("Insert Image"), false);
	insert_image_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_insert_image));

	add_toolbar_separator();

	// CSS line-height dropdown: presets are direct actions, while the custom
	// field accepts unitless values, percentages, px values, and "normal".
	line_height_button = make_button(first_row, RTR("Line Height"), true);
	line_height_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_pressed_line_height));
	line_height_button->connect(SNAME("button_down"), callable_mp(this, &RichTextEditToolbar::_dropdown_button_down).bind(DROPDOWN_LINE_HEIGHT));
	add_dropdown_caret(line_height_button);

	line_height_popup = memnew(PopupPanel);
	VBoxContainer *line_height_box = memnew(VBoxContainer);
	line_height_box->add_theme_constant_override("separation", 0);
	line_height_popup->add_child(line_height_box, false, INTERNAL_MODE_FRONT);
	MarginContainer *line_height_presets_label_margin = memnew(MarginContainer);
	line_height_presets_label_margin->add_theme_constant_override("margin_top", 4);
	line_height_presets_label_margin->add_theme_constant_override("margin_bottom", 6);
	line_height_box->add_child(line_height_presets_label_margin, false, INTERNAL_MODE_FRONT);
	line_height_presets_label = memnew(Label(RTR("PRESETS")));
	line_height_presets_label_margin->add_child(line_height_presets_label, false, INTERNAL_MODE_FRONT);
	VBoxContainer *line_height_presets = memnew(VBoxContainer);
	line_height_presets->add_theme_constant_override("separation", 2);
	MarginContainer *line_height_presets_margin = memnew(MarginContainer);
	line_height_presets_margin->add_theme_constant_override("margin_bottom", 10);
	line_height_presets_margin->add_child(line_height_presets, false, INTERNAL_MODE_FRONT);
	line_height_box->add_child(line_height_presets_margin, false, INTERNAL_MODE_FRONT);
	const struct {
		const char *label;
		const char *value;
	} line_height_values[] = {
		{ "1 (Single)", "1" },
		{ "1.15", "1.15" },
		{ "1.5 (1.5x)", "1.5" },
		{ "2 (Double)", "2" },
	};
	for (const auto &preset : line_height_values) {
		RichTextEditToolbarButton *preset_button = memnew(RichTextEditToolbarButton);
		preset_button->set_text(String(preset.label));
		preset_button->set_custom_minimum_size(Size2(0, 26));
		preset_button->set_toggle_mode(true);
		preset_button->set_meta("line_height_value", String(preset.value));
		preset_button->set_focus_mode(FOCUS_ALL);
		line_height_presets->add_child(preset_button, false, INTERNAL_MODE_FRONT);
		preset_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_line_height_preset_selected).bind(String(preset.value)));
		line_height_preset_buttons.push_back(preset_button);
	}
	MarginContainer *line_height_custom_label_margin = memnew(MarginContainer);
	line_height_custom_label_margin->add_theme_constant_override("margin_bottom", 6);
	line_height_box->add_child(line_height_custom_label_margin, false, INTERNAL_MODE_FRONT);
	line_height_custom_label = memnew(Label(RTR("CUSTOM (CSS LINE-HEIGHT)")));
	line_height_custom_label_margin->add_child(line_height_custom_label, false, INTERNAL_MODE_FRONT);
	line_height_line_edit = memnew(LineEdit);
	line_height_line_edit->set_placeholder(RTR("e.g. 1.5, 150%, 24px, normal"));
	line_height_line_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	line_height_box->add_child(line_height_line_edit, false, INTERNAL_MODE_FRONT);
	line_height_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &RichTextEditToolbar::_line_height_apply_pressed).unbind(1));
	HBoxContainer *line_height_actions = memnew(HBoxContainer);
	line_height_actions->add_theme_constant_override("separation", 6);
	line_height_actions->set_alignment(BoxContainer::ALIGNMENT_END);
	MarginContainer *line_height_actions_margin = memnew(MarginContainer);
	line_height_actions_margin->add_theme_constant_override("margin_top", 8);
	line_height_actions_margin->add_child(line_height_actions, false, INTERNAL_MODE_FRONT);
	line_height_box->add_child(line_height_actions_margin, false, INTERNAL_MODE_FRONT);
	line_height_cancel_button = memnew(RichTextEditToolbarButton);
	line_height_cancel_button->set_text(RTR("Cancel"));
	line_height_cancel_button->set_focus_mode(FOCUS_ALL);
	line_height_actions->add_child(line_height_cancel_button, false, INTERNAL_MODE_FRONT);
	line_height_cancel_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_line_height_cancel_pressed));
	line_height_apply_button = memnew(RichTextEditToolbarButton);
	line_height_apply_button->set_text(RTR("Apply"));
	line_height_apply_button->set_focus_mode(FOCUS_ALL);
	line_height_actions->add_child(line_height_apply_button, false, INTERNAL_MODE_FRONT);
	line_height_apply_button->connect(SceneStringName(pressed), callable_mp(this, &RichTextEditToolbar::_line_height_apply_pressed));
	add_child(line_height_popup, false, INTERNAL_MODE_FRONT);

	alignment_popup->connect(SNAME("popup_hide"), callable_mp(this, &RichTextEditToolbar::_dropdown_popup_hidden).bind(DROPDOWN_ALIGNMENT));
	link_popup->connect(SNAME("popup_hide"), callable_mp(this, &RichTextEditToolbar::_dropdown_popup_hidden).bind(DROPDOWN_LINK));
	line_height_popup->connect(SNAME("popup_hide"), callable_mp(this, &RichTextEditToolbar::_dropdown_popup_hidden).bind(DROPDOWN_LINE_HEIGHT));

	for (int i = second_row_start; i < first_row->get_child_count(true); i++) {
		if (Control *item = Object::cast_to<Control>(first_row->get_child(i, true))) {
			second_row_items.push_back(item);
		}
	}

	add_toolbar_padding();
}

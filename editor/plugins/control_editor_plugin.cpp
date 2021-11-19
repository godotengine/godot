/*************************************************************************/
/*  control_editor_plugin.cpp                                            */
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

#include "control_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_tool_drawer.h"
#include "editor/plugins/canvas_item_editor_plugin.h"
#include "scene/gui/check_box.h"
#include "scene/gui/control.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"

void ControlPositioningWarning::_update_warning() {
	if (!control_node) {
		title_icon->set_texture(nullptr);
		title_label->set_text("");
		hint_label->set_text("");
		return;
	}

	Node *parent_node = control_node->get_parent_control();
	if (!parent_node) {
		title_icon->set_texture(get_theme_icon(SNAME("SubViewport"), SNAME("EditorIcons")));
		title_label->set_text(TTR("This node doesn't have a control parent."));
		hint_label->set_text(TTR("Use the appropriate layout properties depending on where you are going to put it."));
	} else if (Object::cast_to<Container>(parent_node)) {
		title_icon->set_texture(get_theme_icon(SNAME("Container"), SNAME("EditorIcons")));
		title_label->set_text(TTR("This node is a child of a container."));
		hint_label->set_text(TTR("Use container properties for positioning."));
	} else {
		title_icon->set_texture(get_theme_icon(SNAME("ControlLayout"), SNAME("EditorIcons")));
		title_label->set_text(TTR("This node is a child of a regular control."));
		hint_label->set_text(TTR("Use anchors and the rectangle for positioning."));
	}

	bg_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("bg_group_note"), SNAME("EditorProperty")));
}

void ControlPositioningWarning::_update_toggler() {
	Ref<Texture2D> arrow;
	if (hint_label->is_visible()) {
		arrow = get_theme_icon(SNAME("arrow"), SNAME("Tree"));
		set_tooltip(TTR("Collapse positioning hint."));
	} else {
		if (is_layout_rtl()) {
			arrow = get_theme_icon(SNAME("arrow_collapsed"), SNAME("Tree"));
		} else {
			arrow = get_theme_icon(SNAME("arrow_collapsed_mirrored"), SNAME("Tree"));
		}
		set_tooltip(TTR("Expand positioning hint."));
	}

	hint_icon->set_texture(arrow);
}

void ControlPositioningWarning::set_control(Control *p_node) {
	control_node = p_node;
	_update_warning();
}

void ControlPositioningWarning::gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		bool state = !hint_label->is_visible();

		hint_filler_left->set_visible(state);
		hint_label->set_visible(state);
		hint_filler_right->set_visible(state);

		_update_toggler();
	}
}

void ControlPositioningWarning::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
			_update_warning();
			_update_toggler();
			break;
	}
}

ControlPositioningWarning::ControlPositioningWarning() {
	set_mouse_filter(MOUSE_FILTER_STOP);

	bg_panel = memnew(PanelContainer);
	bg_panel->set_mouse_filter(MOUSE_FILTER_IGNORE);
	add_child(bg_panel);

	grid = memnew(GridContainer);
	grid->set_columns(3);
	bg_panel->add_child(grid);

	title_icon = memnew(TextureRect);
	title_icon->set_stretch_mode(TextureRect::StretchMode::STRETCH_KEEP_CENTERED);
	grid->add_child(title_icon);

	title_label = memnew(Label);
	title_label->set_autowrap_mode(Label::AutowrapMode::AUTOWRAP_WORD);
	title_label->set_h_size_flags(SIZE_EXPAND_FILL);
	title_label->set_vertical_alignment(VerticalAlignment::VERTICAL_ALIGNMENT_CENTER);
	grid->add_child(title_label);

	hint_icon = memnew(TextureRect);
	hint_icon->set_stretch_mode(TextureRect::StretchMode::STRETCH_KEEP_CENTERED);
	grid->add_child(hint_icon);

	// Filler.
	hint_filler_left = memnew(Control);
	hint_filler_left->hide();
	grid->add_child(hint_filler_left);

	hint_label = memnew(Label);
	hint_label->set_autowrap_mode(Label::AutowrapMode::AUTOWRAP_WORD);
	hint_label->set_h_size_flags(SIZE_EXPAND_FILL);
	hint_label->set_vertical_alignment(VerticalAlignment::VERTICAL_ALIGNMENT_CENTER);
	hint_label->hide();
	grid->add_child(hint_label);

	// Filler.
	hint_filler_right = memnew(Control);
	hint_filler_right->hide();
	grid->add_child(hint_filler_right);
}

void EditorPropertyAnchorsPreset::_set_read_only(bool p_read_only) {
	options->set_disabled(p_read_only);
};

void EditorPropertyAnchorsPreset::_option_selected(int p_which) {
	int64_t val = options->get_item_metadata(p_which);
	emit_changed(get_edited_property(), val);
}

void EditorPropertyAnchorsPreset::update_property() {
	int64_t which = get_edited_object()->get(get_edited_property());

	for (int i = 0; i < options->get_item_count(); i++) {
		Variant val = options->get_item_metadata(i);
		if (val != Variant() && which == (int64_t)val) {
			options->select(i);
			return;
		}
	}
}

void EditorPropertyAnchorsPreset::setup(const Vector<String> &p_options) {
	options->clear();

	Vector<String> split_after;
	split_after.append("Custom");
	split_after.append("PresetWide");
	split_after.append("PresetBottomLeft");
	split_after.append("PresetCenter");

	for (int i = 0, j = 0; i < p_options.size(); i++, j++) {
		Vector<String> text_split = p_options[i].split(":");
		int64_t current_val = text_split[1].to_int();

		String humanized_name = text_split[0];
		if (humanized_name.begins_with("Preset")) {
			if (humanized_name == "PresetWide") {
				humanized_name = "Full Rect";
			} else {
				humanized_name = humanized_name.trim_prefix("Preset");
				humanized_name = humanized_name.capitalize();
			}

			String icon_name = text_split[0].trim_prefix("Preset");
			icon_name = "ControlAlign" + icon_name;
			options->add_icon_item(EditorNode::get_singleton()->get_gui_base()->get_theme_icon(icon_name, "EditorIcons"), humanized_name);
		} else {
			options->add_item(humanized_name);
		}

		options->set_item_metadata(j, current_val);
		if (split_after.has(text_split[0])) {
			options->add_separator();
			j++;
		}
	}
}

EditorPropertyAnchorsPreset::EditorPropertyAnchorsPreset() {
	options = memnew(OptionButton);
	options->set_clip_text(true);
	options->set_flat(true);
	add_child(options);
	add_focusable(options);
	options->connect("item_selected", callable_mp(this, &EditorPropertyAnchorsPreset::_option_selected));
}

void EditorPropertySizeFlags::_set_read_only(bool p_read_only) {
	for (CheckBox *check : flag_checks) {
		check->set_disabled(p_read_only);
	}
	flag_presets->set_disabled(p_read_only);
};

void EditorPropertySizeFlags::_preset_selected(int p_which) {
	int preset = flag_presets->get_item_id(p_which);
	if (preset == SIZE_FLAGS_PRESET_CUSTOM) {
		flag_options->set_visible(true);
		return;
	}
	flag_options->set_visible(false);

	uint32_t value = 0;
	switch (preset) {
		case SIZE_FLAGS_PRESET_FILL:
			value = SIZE_FILL;
			break;
		case SIZE_FLAGS_PRESET_SHRINK_BEGIN:
			value = SIZE_SHRINK_BEGIN;
			break;
		case SIZE_FLAGS_PRESET_SHRINK_CENTER:
			value = SIZE_SHRINK_CENTER;
			break;
		case SIZE_FLAGS_PRESET_SHRINK_END:
			value = SIZE_SHRINK_END;
			break;
	}

	bool is_expand = flag_expand->is_visible() && flag_expand->is_pressed();
	if (is_expand) {
		value |= SIZE_EXPAND;
	}

	emit_changed(get_edited_property(), value);
}

void EditorPropertySizeFlags::_expand_toggled() {
	uint32_t value = get_edited_object()->get(get_edited_property());

	if (flag_expand->is_visible() && flag_expand->is_pressed()) {
		value |= SIZE_EXPAND;
	} else {
		value ^= SIZE_EXPAND;
	}

	// Keep the custom preset selected as we toggle individual flags.
	keep_selected_preset = true;
	emit_changed(get_edited_property(), value);
}

void EditorPropertySizeFlags::_flag_toggled() {
	uint32_t value = 0;
	for (int i = 0; i < flag_checks.size(); i++) {
		if (flag_checks[i]->is_pressed()) {
			int flag_value = flag_checks[i]->get_meta("_value");
			value |= flag_value;
		}
	}

	bool is_expand = flag_expand->is_visible() && flag_expand->is_pressed();
	if (is_expand) {
		value |= SIZE_EXPAND;
	}

	// Keep the custom preset selected as we toggle individual flags.
	keep_selected_preset = true;
	emit_changed(get_edited_property(), value);
}

void EditorPropertySizeFlags::update_property() {
	uint32_t value = get_edited_object()->get(get_edited_property());

	for (int i = 0; i < flag_checks.size(); i++) {
		int flag_value = flag_checks[i]->get_meta("_value");
		if (value & flag_value) {
			flag_checks[i]->set_pressed(true);
		} else {
			flag_checks[i]->set_pressed(false);
		}
	}

	bool is_expand = value & SIZE_EXPAND;
	flag_expand->set_pressed(is_expand);

	if (keep_selected_preset) {
		keep_selected_preset = false;
		return;
	}

	FlagPreset preset = SIZE_FLAGS_PRESET_CUSTOM;
	if (value == SIZE_FILL || value == (SIZE_FILL | SIZE_EXPAND)) {
		preset = SIZE_FLAGS_PRESET_FILL;
	} else if (value == SIZE_SHRINK_BEGIN || value == (SIZE_SHRINK_BEGIN | SIZE_EXPAND)) {
		preset = SIZE_FLAGS_PRESET_SHRINK_BEGIN;
	} else if (value == SIZE_SHRINK_CENTER || value == (SIZE_SHRINK_CENTER | SIZE_EXPAND)) {
		preset = SIZE_FLAGS_PRESET_SHRINK_CENTER;
	} else if (value == SIZE_SHRINK_END || value == (SIZE_SHRINK_END | SIZE_EXPAND)) {
		preset = SIZE_FLAGS_PRESET_SHRINK_END;
	}

	int preset_idx = flag_presets->get_item_index(preset);
	if (preset_idx >= 0) {
		flag_presets->select(preset_idx);
	}
	flag_options->set_visible(preset == SIZE_FLAGS_PRESET_CUSTOM);
}

void EditorPropertySizeFlags::setup(const Vector<String> &p_options, bool p_vertical) {
	vertical = p_vertical;

	if (p_options.size() == 0) {
		flag_presets->clear();
		flag_presets->add_item(TTR("Container Default"));
		flag_presets->set_disabled(true);
		flag_expand->set_visible(false);
		return;
	}

	Map<int, String> flags;
	for (int i = 0, j = 0; i < p_options.size(); i++, j++) {
		Vector<String> text_split = p_options[i].split(":");
		int64_t current_val = text_split[1].to_int();
		flags[current_val] = text_split[0];

		if (current_val == SIZE_EXPAND) {
			continue;
		}

		CheckBox *cb = memnew(CheckBox);
		cb->set_text(text_split[0]);
		cb->set_clip_text(true);
		cb->set_meta("_value", current_val);
		cb->connect("pressed", callable_mp(this, &EditorPropertySizeFlags::_flag_toggled));
		add_focusable(cb);

		flag_options->add_child(cb);
		flag_checks.append(cb);
	}

	Control *gui_base = EditorNode::get_singleton()->get_gui_base();
	String wide_preset_icon = SNAME("ControlAlignHCenterWide");
	if (vertical) {
		wide_preset_icon = SNAME("ControlAlignVCenterWide");
	}

	flag_presets->clear();
	if (flags.has(SIZE_FILL)) {
		flag_presets->add_icon_item(gui_base->get_theme_icon(wide_preset_icon, SNAME("EditorIcons")), TTR("Fill"), SIZE_FLAGS_PRESET_FILL);
	}
	// Shrink Begin is the same as no flags at all, as such it cannot be disabled.
	flag_presets->add_icon_item(gui_base->get_theme_icon(SNAME("ControlAlignCenterLeft"), SNAME("EditorIcons")), TTR("Shrink Begin"), SIZE_FLAGS_PRESET_SHRINK_BEGIN);
	if (flags.has(SIZE_SHRINK_CENTER)) {
		flag_presets->add_icon_item(gui_base->get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")), TTR("Shrink Center"), SIZE_FLAGS_PRESET_SHRINK_CENTER);
	}
	if (flags.has(SIZE_SHRINK_END)) {
		flag_presets->add_icon_item(gui_base->get_theme_icon(SNAME("ControlAlignCenterRight"), SNAME("EditorIcons")), TTR("Shrink End"), SIZE_FLAGS_PRESET_SHRINK_END);
	}
	flag_presets->add_separator();
	flag_presets->add_item(TTR("Custom"), SIZE_FLAGS_PRESET_CUSTOM);

	flag_expand->set_visible(flags.has(SIZE_EXPAND));
}

EditorPropertySizeFlags::EditorPropertySizeFlags() {
	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	flag_presets = memnew(OptionButton);
	flag_presets->set_clip_text(true);
	flag_presets->set_flat(true);
	vb->add_child(flag_presets);
	add_focusable(flag_presets);
	set_label_reference(flag_presets);
	flag_presets->connect("item_selected", callable_mp(this, &EditorPropertySizeFlags::_preset_selected));

	flag_options = memnew(VBoxContainer);
	flag_options->hide();
	vb->add_child(flag_options);

	flag_expand = memnew(CheckBox);
	flag_expand->set_text(TTR("Expand"));
	vb->add_child(flag_expand);
	add_focusable(flag_expand);
	flag_expand->connect("pressed", callable_mp(this, &EditorPropertySizeFlags::_expand_toggled));
}

bool EditorInspectorPluginControl::can_handle(Object *p_object) {
	return Object::cast_to<Control>(p_object) != nullptr;
}

void EditorInspectorPluginControl::parse_group(Object *p_object, const String &p_group) {
	Control *control = Object::cast_to<Control>(p_object);
	if (!control || p_group != "Layout") {
		return;
	}

	ControlPositioningWarning *pos_warning = memnew(ControlPositioningWarning);
	pos_warning->set_control(control);
	add_custom_control(pos_warning);
}

bool EditorInspectorPluginControl::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide) {
	Control *control = Object::cast_to<Control>(p_object);
	if (!control) {
		return false;
	}

	if (p_path == "anchors_preset") {
		EditorPropertyAnchorsPreset *prop_editor = memnew(EditorPropertyAnchorsPreset);
		Vector<String> options = p_hint_text.split(",");
		prop_editor->setup(options);
		add_property_editor(p_path, prop_editor);

		return true;
	}

	if (p_path == "size_flags_horizontal" || p_path == "size_flags_vertical") {
		EditorPropertySizeFlags *prop_editor = memnew(EditorPropertySizeFlags);
		Vector<String> options;
		if (!p_hint_text.is_empty()) {
			options = p_hint_text.split(",");
		}
		prop_editor->setup(options, p_path == "size_flags_vertical");
		add_property_editor(p_path, prop_editor);

		return true;
	}

	return false;
}

void ControlAnchorPresetPicker::_add_row_button(HBoxContainer *p_row, const LayoutPreset p_preset, const String &p_name) {
	ERR_FAIL_COND(preset_buttons.has(p_preset));

	Button *b = memnew(Button);
	b->set_custom_minimum_size(Size2i(36, 36) * EDSCALE);
	b->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	b->set_tooltip(p_name);
	p_row->add_child(b);
	b->connect("pressed", callable_mp(this, &ControlAnchorPresetPicker::_preset_button_pressed), varray(p_preset));

	preset_buttons[p_preset] = b;
}

void ControlAnchorPresetPicker::_add_separator(BoxContainer *p_box, Separator *p_separator) {
	p_separator->add_theme_constant_override("separation", grid_separation);
	p_separator->set_custom_minimum_size(Size2i(1, 1));
	p_box->add_child(p_separator);
}

void ControlAnchorPresetPicker::_preset_button_pressed(const LayoutPreset p_preset) {
	emit_signal("preset_selected", p_preset);
}

void ControlAnchorPresetPicker::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			preset_buttons[PRESET_TOP_LEFT]->set_icon(get_theme_icon(SNAME("ControlAlignTopLeft"), SNAME("EditorIcons")));
			preset_buttons[PRESET_CENTER_TOP]->set_icon(get_theme_icon(SNAME("ControlAlignCenterTop"), SNAME("EditorIcons")));
			preset_buttons[PRESET_TOP_RIGHT]->set_icon(get_theme_icon(SNAME("ControlAlignTopRight"), SNAME("EditorIcons")));

			preset_buttons[PRESET_CENTER_LEFT]->set_icon(get_theme_icon(SNAME("ControlAlignCenterLeft"), SNAME("EditorIcons")));
			preset_buttons[PRESET_CENTER]->set_icon(get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")));
			preset_buttons[PRESET_CENTER_RIGHT]->set_icon(get_theme_icon(SNAME("ControlAlignCenterRight"), SNAME("EditorIcons")));

			preset_buttons[PRESET_BOTTOM_LEFT]->set_icon(get_theme_icon(SNAME("ControlAlignBottomLeft"), SNAME("EditorIcons")));
			preset_buttons[PRESET_CENTER_BOTTOM]->set_icon(get_theme_icon(SNAME("ControlAlignCenterBottom"), SNAME("EditorIcons")));
			preset_buttons[PRESET_BOTTOM_RIGHT]->set_icon(get_theme_icon(SNAME("ControlAlignBottomRight"), SNAME("EditorIcons")));

			preset_buttons[PRESET_TOP_WIDE]->set_icon(get_theme_icon(SNAME("ControlAlignTopWide"), SNAME("EditorIcons")));
			preset_buttons[PRESET_HCENTER_WIDE]->set_icon(get_theme_icon(SNAME("ControlAlignHCenterWide"), SNAME("EditorIcons")));
			preset_buttons[PRESET_BOTTOM_WIDE]->set_icon(get_theme_icon(SNAME("ControlAlignBottomWide"), SNAME("EditorIcons")));

			preset_buttons[PRESET_LEFT_WIDE]->set_icon(get_theme_icon(SNAME("ControlAlignLeftWide"), SNAME("EditorIcons")));
			preset_buttons[PRESET_VCENTER_WIDE]->set_icon(get_theme_icon(SNAME("ControlAlignVCenterWide"), SNAME("EditorIcons")));
			preset_buttons[PRESET_RIGHT_WIDE]->set_icon(get_theme_icon(SNAME("ControlAlignRightWide"), SNAME("EditorIcons")));

			preset_buttons[PRESET_WIDE]->set_icon(get_theme_icon(SNAME("ControlAlignWide"), SNAME("EditorIcons")));
		} break;
	}
}

void ControlAnchorPresetPicker::_bind_methods() {
	ADD_SIGNAL(MethodInfo("preset_selected", PropertyInfo(Variant::INT, "preset")));
}

ControlAnchorPresetPicker::ControlAnchorPresetPicker() {
	VBoxContainer *main_vb = memnew(VBoxContainer);
	main_vb->add_theme_constant_override("separation", grid_separation);
	add_child(main_vb);

	HBoxContainer *top_row = memnew(HBoxContainer);
	top_row->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	top_row->add_theme_constant_override("separation", grid_separation);
	main_vb->add_child(top_row);

	_add_row_button(top_row, PRESET_TOP_LEFT, TTR("Top Left"));
	_add_row_button(top_row, PRESET_CENTER_TOP, TTR("Center Top"));
	_add_row_button(top_row, PRESET_TOP_RIGHT, TTR("Top Right"));
	_add_separator(top_row, memnew(VSeparator));
	_add_row_button(top_row, PRESET_TOP_WIDE, TTR("Top Wide"));

	HBoxContainer *mid_row = memnew(HBoxContainer);
	mid_row->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	mid_row->add_theme_constant_override("separation", grid_separation);
	main_vb->add_child(mid_row);

	_add_row_button(mid_row, PRESET_CENTER_LEFT, TTR("Center Left"));
	_add_row_button(mid_row, PRESET_CENTER, TTR("Center"));
	_add_row_button(mid_row, PRESET_CENTER_RIGHT, TTR("Center Right"));
	_add_separator(mid_row, memnew(VSeparator));
	_add_row_button(mid_row, PRESET_HCENTER_WIDE, TTR("HCenter Wide"));

	HBoxContainer *bot_row = memnew(HBoxContainer);
	bot_row->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	bot_row->add_theme_constant_override("separation", grid_separation);
	main_vb->add_child(bot_row);

	_add_row_button(bot_row, PRESET_BOTTOM_LEFT, TTR("Bottom Left"));
	_add_row_button(bot_row, PRESET_CENTER_BOTTOM, TTR("Center Bottom"));
	_add_row_button(bot_row, PRESET_BOTTOM_RIGHT, TTR("Bottom Right"));
	_add_separator(bot_row, memnew(VSeparator));
	_add_row_button(bot_row, PRESET_BOTTOM_WIDE, TTR("Bottom Wide"));

	_add_separator(main_vb, memnew(HSeparator));

	HBoxContainer *extra_row = memnew(HBoxContainer);
	extra_row->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	extra_row->add_theme_constant_override("separation", grid_separation);
	main_vb->add_child(extra_row);

	_add_row_button(extra_row, PRESET_LEFT_WIDE, TTR("Left Wide"));
	_add_row_button(extra_row, PRESET_VCENTER_WIDE, TTR("VCenter Wide"));
	_add_row_button(extra_row, PRESET_RIGHT_WIDE, TTR("Right Wide"));
	_add_separator(extra_row, memnew(VSeparator));
	_add_row_button(extra_row, PRESET_WIDE, TTR("Full Rect"));
}

void ControlAnchorsEditorToolbar::_set_anchors_and_offsets_preset(LayoutPreset p_preset) {
	List<Node *> selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Change Anchors and Offsets"));

	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (control) {
			undo_redo->add_do_method(control, "set_anchors_preset", p_preset);
			switch (p_preset) {
				case PRESET_TOP_LEFT:
				case PRESET_TOP_RIGHT:
				case PRESET_BOTTOM_LEFT:
				case PRESET_BOTTOM_RIGHT:
				case PRESET_CENTER_LEFT:
				case PRESET_CENTER_TOP:
				case PRESET_CENTER_RIGHT:
				case PRESET_CENTER_BOTTOM:
				case PRESET_CENTER:
					undo_redo->add_do_method(control, "set_offsets_preset", p_preset, PRESET_MODE_KEEP_SIZE);
					break;
				case PRESET_LEFT_WIDE:
				case PRESET_TOP_WIDE:
				case PRESET_RIGHT_WIDE:
				case PRESET_BOTTOM_WIDE:
				case PRESET_VCENTER_WIDE:
				case PRESET_HCENTER_WIDE:
				case PRESET_WIDE:
					undo_redo->add_do_method(control, "set_offsets_preset", p_preset, PRESET_MODE_MINSIZE);
					break;
			}
			undo_redo->add_undo_method(control, "_edit_set_state", control->_edit_get_state());
		}
	}

	undo_redo->commit_action();

	anchors_mode = false;
	anchor_mode_button->set_pressed(anchors_mode);
}

void ControlAnchorsEditorToolbar::_set_anchors_and_offsets_to_keep_ratio() {
	List<Node *> selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Change Anchors and Offsets"));

	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (control) {
			Point2 top_left_anchor = _position_to_anchor(control, Point2());
			Point2 bottom_right_anchor = _position_to_anchor(control, control->get_size());
			undo_redo->add_do_method(control, "set_anchor", SIDE_LEFT, top_left_anchor.x, false, true);
			undo_redo->add_do_method(control, "set_anchor", SIDE_RIGHT, bottom_right_anchor.x, false, true);
			undo_redo->add_do_method(control, "set_anchor", SIDE_TOP, top_left_anchor.y, false, true);
			undo_redo->add_do_method(control, "set_anchor", SIDE_BOTTOM, bottom_right_anchor.y, false, true);
			undo_redo->add_do_method(control, "set_meta", "_edit_use_anchors_", true);

			const bool use_anchors = control->has_meta("_edit_use_anchors_") && control->get_meta("_edit_use_anchors_");
			undo_redo->add_undo_method(control, "_edit_set_state", control->_edit_get_state());
			if (use_anchors) {
				undo_redo->add_undo_method(control, "set_meta", "_edit_use_anchors_", true);
			} else {
				undo_redo->add_undo_method(control, "remove_meta", "_edit_use_anchors_");
			}

			anchors_mode = true;
			anchor_mode_button->set_pressed(anchors_mode);
		}
	}

	undo_redo->commit_action();
}

void ControlAnchorsEditorToolbar::_set_anchors_preset(LayoutPreset p_preset) {
	List<Node *> selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Change Anchors"));
	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (control) {
			undo_redo->add_do_method(control, "set_anchors_preset", p_preset);
			undo_redo->add_undo_method(control, "_edit_set_state", control->_edit_get_state());
		}
	}

	undo_redo->commit_action();
}

void ControlAnchorsEditorToolbar::_button_toggle_picker_mode(bool p_pressed) {
	pick_anchors_and_offsets = p_pressed;
}

void ControlAnchorsEditorToolbar::_picker_preset_selected(int p_preset) {
	LayoutPreset preset = (LayoutPreset)p_preset;
	if (pick_anchors_and_offsets) {
		_set_anchors_and_offsets_preset(preset);
	} else {
		_set_anchors_preset(preset);
	}
}

void ControlAnchorsEditorToolbar::_button_toggle_anchor_mode(bool p_status) {
	List<Control *> selection = _get_edited_controls();
	for (Control *E : selection) {
		if (Object::cast_to<Container>(E->get_parent())) {
			continue;
		}

		if (p_status) {
			E->set_meta("_edit_use_anchors_", true);
		} else {
			E->remove_meta("_edit_use_anchors_");
		}
	}

	anchors_mode = p_status;
	CanvasItemEditor::get_singleton()->update_viewport();
}

Vector2 ControlAnchorsEditorToolbar::_anchor_to_position(const Control *p_control, Vector2 anchor) {
	ERR_FAIL_COND_V(!p_control, Vector2());

	Transform2D parent_transform = p_control->get_transform().affine_inverse();
	Rect2 parent_rect = p_control->get_parent_anchorable_rect();

	if (p_control->is_layout_rtl()) {
		return parent_transform.xform(parent_rect.position + Vector2(parent_rect.size.x - parent_rect.size.x * anchor.x, parent_rect.size.y * anchor.y));
	} else {
		return parent_transform.xform(parent_rect.position + Vector2(parent_rect.size.x * anchor.x, parent_rect.size.y * anchor.y));
	}
}

Vector2 ControlAnchorsEditorToolbar::_position_to_anchor(const Control *p_control, Vector2 position) {
	ERR_FAIL_COND_V(!p_control, Vector2());

	Rect2 parent_rect = p_control->get_parent_anchorable_rect();

	Vector2 output = Vector2();
	if (p_control->is_layout_rtl()) {
		output.x = (parent_rect.size.x == 0) ? 0.0 : (parent_rect.size.x - p_control->get_transform().xform(position).x - parent_rect.position.x) / parent_rect.size.x;
	} else {
		output.x = (parent_rect.size.x == 0) ? 0.0 : (p_control->get_transform().xform(position).x - parent_rect.position.x) / parent_rect.size.x;
	}
	output.y = (parent_rect.size.y == 0) ? 0.0 : (p_control->get_transform().xform(position).y - parent_rect.position.y) / parent_rect.size.y;
	return output;
}

bool ControlAnchorsEditorToolbar::_is_node_locked(const Node *p_node) {
	return p_node->has_meta("_edit_lock_") && p_node->get_meta("_edit_lock_");
}

List<Control *> ControlAnchorsEditorToolbar::_get_edited_controls() {
	List<Control *> selection;
	for (const KeyValue<Node *, Object *> &E : editor_selection->get_selection()) {
		Control *control = Object::cast_to<Control>(E.key);
		if (control && control->is_visible_in_tree() && control->get_viewport() == EditorNode::get_singleton()->get_scene_root() && !_is_node_locked(control)) {
			selection.push_back(control);
		}
	}

	return selection;
}

void ControlAnchorsEditorToolbar::_selection_changed() {
	// Update toolbar visibility.
	bool has_controls = false;
	bool has_control_parents = false;
	bool has_container_parents = false;

	for (const KeyValue<Node *, Object *> &E : editor_selection->get_selection()) {
		Control *control = Object::cast_to<Control>(E.key);
		if (!control) {
			continue;
		}
		has_controls = true;

		if (Object::cast_to<Control>(control->get_parent())) {
			has_control_parents = true;
		}
		if (Object::cast_to<Container>(control->get_parent())) {
			has_container_parents = true;
		}
	}

	if (has_controls && (!has_control_parents || !has_container_parents)) {
		CanvasItemEditor::get_singleton()->set_tool_drawer_visible(this, true);
	} else {
		CanvasItemEditor::get_singleton()->set_tool_drawer_visible(this, false);
		anchor_mode_button->set_pressed(false);
		return;
	}

	// Update anchor mode.
	int nb_valid_controls = 0;
	int nb_anchors_mode = 0;

	List<Node *> selection = editor_selection->get_selected_node_list();
	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (!control) {
			continue;
		}
		if (Object::cast_to<Container>(control->get_parent())) {
			continue;
		}

		nb_valid_controls++;
		if (control->has_meta("_edit_use_anchors_") && control->get_meta("_edit_use_anchors_")) {
			nb_anchors_mode++;
		}
	}

	anchors_mode = (nb_valid_controls == nb_anchors_mode);
	anchor_mode_button->set_pressed(anchors_mode);
}

ControlAnchorsEditorToolbar::ControlAnchorsEditorToolbar() {
	picker_mode_button = memnew(CheckBox);
	picker_mode_button->set_flat(true);
	picker_mode_button->set_pressed(pick_anchors_and_offsets);
	picker_mode_button->set_text(TTR("Anchors and Offsets"));
	picker_mode_button->set_tooltip(TTR("Enable to change both anchors and offsets.\nDisable to change only anchors."));
	add_child(picker_mode_button);
	picker_mode_button->connect("toggled", callable_mp(this, &ControlAnchorsEditorToolbar::_button_toggle_picker_mode));

	ControlAnchorPresetPicker *picker = memnew(ControlAnchorPresetPicker);
	picker->set_h_size_flags(SIZE_SHRINK_CENTER);
	add_child(picker);
	picker->connect("preset_selected", callable_mp(this, &ControlAnchorsEditorToolbar::_picker_preset_selected));

	keep_ratio_button = memnew(Button);
	keep_ratio_button->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	keep_ratio_button->set_text(TTR("Set to Current Ratio"));
	keep_ratio_button->set_tooltip(TTR("Adjust anchors and offsets to match the current rect size."));
	add_child(keep_ratio_button);
	keep_ratio_button->connect("pressed", callable_mp(this, &ControlAnchorsEditorToolbar::_set_anchors_and_offsets_to_keep_ratio));

	anchor_mode_button = memnew(CheckBox);
	anchor_mode_button->set_flat(true);
	anchor_mode_button->set_toggle_mode(true);
	anchor_mode_button->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	anchor_mode_button->set_text(TTR("Resize with Anchors"));
	anchor_mode_button->set_tooltip(TTR("Resizing the control will also affect its anchors."));
	anchor_mode_button->add_theme_constant_override("hseparation", 6 * EDSCALE);
	add_child(anchor_mode_button);
	anchor_mode_button->connect("toggled", callable_mp(this, &ControlAnchorsEditorToolbar::_button_toggle_anchor_mode));

	undo_redo = EditorNode::get_singleton()->get_undo_redo();
	editor_selection = EditorNode::get_singleton()->get_editor_selection();
	editor_selection->add_editor_plugin(this);
	editor_selection->connect("selection_changed", callable_mp(this, &ControlAnchorsEditorToolbar::_selection_changed));

	singleton = this;
}

ControlAnchorsEditorToolbar *ControlAnchorsEditorToolbar::singleton = nullptr;

void ContainerSizeFlagPicker::_add_row_button(HBoxContainer *p_row, const SizeFlags p_flag, const String &p_name) {
	ERR_FAIL_COND(flag_buttons.has(p_flag));

	Button *b = memnew(Button);
	b->set_custom_minimum_size(Size2i(36, 36) * EDSCALE);
	b->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	b->set_tooltip(p_name);
	p_row->add_child(b);
	b->connect("pressed", callable_mp(this, &ContainerSizeFlagPicker::_flag_button_pressed), varray(p_flag));

	flag_buttons[p_flag] = b;
}

void ContainerSizeFlagPicker::_add_separator(BoxContainer *p_box, Separator *p_separator) {
	p_separator->add_theme_constant_override("separation", grid_separation);
	p_separator->set_custom_minimum_size(Size2i(1, 1));
	p_box->add_child(p_separator);
}

void ContainerSizeFlagPicker::_flag_button_pressed(const SizeFlags p_flag) {
	int flags = p_flag;
	if (expand_button->is_pressed()) {
		flags |= SIZE_EXPAND;
	}

	emit_signal("size_flags_selected", flags);
}

void ContainerSizeFlagPicker::set_allowed_flags(Vector<SizeFlags> &p_flags) {
	flag_buttons[SIZE_SHRINK_BEGIN]->set_disabled(!p_flags.has(SIZE_SHRINK_BEGIN));
	flag_buttons[SIZE_SHRINK_CENTER]->set_disabled(!p_flags.has(SIZE_SHRINK_CENTER));
	flag_buttons[SIZE_SHRINK_END]->set_disabled(!p_flags.has(SIZE_SHRINK_END));
	flag_buttons[SIZE_FILL]->set_disabled(!p_flags.has(SIZE_FILL));

	expand_button->set_disabled(!p_flags.has(SIZE_EXPAND));
	if (p_flags.has(SIZE_EXPAND)) {
		expand_button->set_tooltip(TTR("Enable to also set the Expand flag.\nDisable to only set Shrink/Fill flags."));
	} else {
		expand_button->set_pressed(false);
		expand_button->set_tooltip(TTR("Some parents of the selected nodes do not support the Expand flag."));
	}
}

void ContainerSizeFlagPicker::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			if (vertical) {
				flag_buttons[SIZE_SHRINK_BEGIN]->set_icon(get_theme_icon(SNAME("ControlAlignCenterTop"), SNAME("EditorIcons")));
				flag_buttons[SIZE_SHRINK_CENTER]->set_icon(get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")));
				flag_buttons[SIZE_SHRINK_END]->set_icon(get_theme_icon(SNAME("ControlAlignCenterBottom"), SNAME("EditorIcons")));

				flag_buttons[SIZE_FILL]->set_icon(get_theme_icon(SNAME("ControlAlignVCenterWide"), SNAME("EditorIcons")));
			} else {
				flag_buttons[SIZE_SHRINK_BEGIN]->set_icon(get_theme_icon(SNAME("ControlAlignCenterLeft"), SNAME("EditorIcons")));
				flag_buttons[SIZE_SHRINK_CENTER]->set_icon(get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")));
				flag_buttons[SIZE_SHRINK_END]->set_icon(get_theme_icon(SNAME("ControlAlignCenterRight"), SNAME("EditorIcons")));

				flag_buttons[SIZE_FILL]->set_icon(get_theme_icon(SNAME("ControlAlignHCenterWide"), SNAME("EditorIcons")));
			}
		} break;
	}
}

void ContainerSizeFlagPicker::_bind_methods() {
	ADD_SIGNAL(MethodInfo("size_flags_selected", PropertyInfo(Variant::INT, "size_flags")));
}

ContainerSizeFlagPicker::ContainerSizeFlagPicker(bool p_vertical) {
	vertical = p_vertical;

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);

	expand_button = memnew(CheckBox);
	expand_button->set_flat(true);
	expand_button->set_text(TTR("Align with Expand"));
	expand_button->set_tooltip(TTR("Enable to also set the Expand flag.\nDisable to only set Shrink/Fill flags."));
	main_vb->add_child(expand_button);

	HBoxContainer *main_row = memnew(HBoxContainer);
	main_row->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	main_row->add_theme_constant_override("separation", grid_separation);
	main_vb->add_child(main_row);

	_add_row_button(main_row, SIZE_SHRINK_BEGIN, TTR("Shrink Begin"));
	_add_row_button(main_row, SIZE_SHRINK_CENTER, TTR("Shrink Center"));
	_add_row_button(main_row, SIZE_SHRINK_END, TTR("Shrink End"));
	_add_separator(main_row, memnew(VSeparator));
	_add_row_button(main_row, SIZE_FILL, TTR("Fill"));
}

void ContainerEditorToolbar::_container_flags_selected(int p_flags, bool p_vertical) {
	List<Node *> selection = editor_selection->get_selected_node_list();

	if (p_vertical) {
		undo_redo->create_action(TTR("Change Vertical Size Flags"));
	} else {
		undo_redo->create_action(TTR("Change Horizontal Size Flags"));
	}

	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (control) {
			if (p_vertical) {
				undo_redo->add_do_method(control, "set_v_size_flags", p_flags);
			} else {
				undo_redo->add_do_method(control, "set_h_size_flags", p_flags);
			}
			undo_redo->add_undo_method(control, "_edit_set_state", control->_edit_get_state());
		}
	}

	undo_redo->commit_action();
}

void ContainerEditorToolbar::_selection_changed() {
	// Update toolbar visibility.
	bool has_controls = false;
	bool has_control_parents = false;
	bool has_container_parents = false;

	// Also update which size flags can be configured for the selected nodes.
	Vector<SizeFlags> allowed_h_flags = {
		SIZE_SHRINK_BEGIN,
		SIZE_SHRINK_CENTER,
		SIZE_SHRINK_END,
		SIZE_FILL,
		SIZE_EXPAND,
	};
	Vector<SizeFlags> allowed_v_flags = {
		SIZE_SHRINK_BEGIN,
		SIZE_SHRINK_CENTER,
		SIZE_SHRINK_END,
		SIZE_FILL,
		SIZE_EXPAND,
	};

	for (const KeyValue<Node *, Object *> &E : editor_selection->get_selection()) {
		Control *control = Object::cast_to<Control>(E.key);
		if (!control) {
			continue;
		}
		has_controls = true;

		if (Object::cast_to<Control>(control->get_parent())) {
			has_control_parents = true;
		}
		if (Object::cast_to<Container>(control->get_parent())) {
			has_container_parents = true;

			Container *parent_container = Object::cast_to<Container>(control->get_parent());

			Vector<int> container_h_flags = parent_container->get_allowed_size_flags_horizontal();
			Vector<SizeFlags> tmp_flags = allowed_h_flags.duplicate();
			for (int i = 0; i < allowed_h_flags.size(); i++) {
				if (!container_h_flags.has((int)allowed_h_flags[i])) {
					tmp_flags.erase(allowed_h_flags[i]);
				}
			}
			allowed_h_flags = tmp_flags;

			Vector<int> container_v_flags = parent_container->get_allowed_size_flags_vertical();
			tmp_flags = allowed_v_flags.duplicate();
			for (int i = 0; i < allowed_v_flags.size(); i++) {
				if (!container_v_flags.has((int)allowed_v_flags[i])) {
					tmp_flags.erase(allowed_v_flags[i]);
				}
			}
			allowed_v_flags = tmp_flags;
		}
	}

	if (has_controls && (!has_control_parents || has_container_parents)) {
		CanvasItemEditor::get_singleton()->set_tool_drawer_visible(this, true);

		if (has_container_parents) {
			container_h_picker->set_allowed_flags(allowed_h_flags);
			container_v_picker->set_allowed_flags(allowed_v_flags);
		} else {
			Vector<SizeFlags> allowed_all_flags = {
				SIZE_SHRINK_BEGIN,
				SIZE_SHRINK_CENTER,
				SIZE_SHRINK_END,
				SIZE_FILL,
				SIZE_EXPAND,
			};

			container_h_picker->set_allowed_flags(allowed_all_flags);
			container_v_picker->set_allowed_flags(allowed_all_flags);
		}
	} else {
		CanvasItemEditor::get_singleton()->set_tool_drawer_visible(this, false);
	}
}

ContainerEditorToolbar::ContainerEditorToolbar() {
	EditorToolDrawerItemGroup *container_h_presets_group = memnew(EditorToolDrawerItemGroup);
	container_h_presets_group->set_title(TTR("Horizontal alignment"));
	add_child(container_h_presets_group);

	container_h_picker = memnew(ContainerSizeFlagPicker(false));
	container_h_presets_group->add_child(container_h_picker);
	container_h_picker->connect("size_flags_selected", callable_mp(this, &ContainerEditorToolbar::_container_flags_selected), varray(false));

	EditorToolDrawerItemGroup *container_v_presets_group = memnew(EditorToolDrawerItemGroup);
	container_v_presets_group->set_title(TTR("Vertical alignment"));
	add_child(container_v_presets_group);

	container_v_picker = memnew(ContainerSizeFlagPicker(true));
	container_v_presets_group->add_child(container_v_picker);
	container_v_picker->connect("size_flags_selected", callable_mp(this, &ContainerEditorToolbar::_container_flags_selected), varray(true));

	undo_redo = EditorNode::get_singleton()->get_undo_redo();
	editor_selection = EditorNode::get_singleton()->get_editor_selection();
	editor_selection->add_editor_plugin(this);
	editor_selection->connect("selection_changed", callable_mp(this, &ContainerEditorToolbar::_selection_changed));
}

ControlEditorPlugin::ControlEditorPlugin() {
	EditorNode *editor = EditorNode::get_singleton();

	anchors_toolbar = memnew(ControlAnchorsEditorToolbar());
	CanvasItemEditor::get_singleton()->add_tool_drawer(TTR("Control Anchors"), editor->get_gui_base()->get_theme_icon(SNAME("ControlLayout"), SNAME("EditorIcons")), anchors_toolbar);

	container_toolbar = memnew(ContainerEditorToolbar());
	CanvasItemEditor::get_singleton()->add_tool_drawer(TTR("Container Layout"), editor->get_gui_base()->get_theme_icon(SNAME("Container"), SNAME("EditorIcons")), container_toolbar);

	Ref<EditorInspectorPluginControl> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}

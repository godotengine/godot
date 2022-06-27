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
#include "editor/plugins/canvas_item_editor_plugin.h"

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
	title_label->set_autowrap_mode(TextServer::AutowrapMode::AUTOWRAP_WORD);
	title_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
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
	hint_label->set_autowrap_mode(TextServer::AutowrapMode::AUTOWRAP_WORD);
	hint_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
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
			value = Control::SIZE_FILL;
			break;
		case SIZE_FLAGS_PRESET_SHRINK_BEGIN:
			value = Control::SIZE_SHRINK_BEGIN;
			break;
		case SIZE_FLAGS_PRESET_SHRINK_CENTER:
			value = Control::SIZE_SHRINK_CENTER;
			break;
		case SIZE_FLAGS_PRESET_SHRINK_END:
			value = Control::SIZE_SHRINK_END;
			break;
	}

	bool is_expand = flag_expand->is_visible() && flag_expand->is_pressed();
	if (is_expand) {
		value |= Control::SIZE_EXPAND;
	}

	emit_changed(get_edited_property(), value);
}

void EditorPropertySizeFlags::_expand_toggled() {
	uint32_t value = get_edited_object()->get(get_edited_property());

	if (flag_expand->is_visible() && flag_expand->is_pressed()) {
		value |= Control::SIZE_EXPAND;
	} else {
		value ^= Control::SIZE_EXPAND;
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
		value |= Control::SIZE_EXPAND;
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

	bool is_expand = value & Control::SIZE_EXPAND;
	flag_expand->set_pressed(is_expand);

	if (keep_selected_preset) {
		keep_selected_preset = false;
		return;
	}

	FlagPreset preset = SIZE_FLAGS_PRESET_CUSTOM;
	if (value == Control::SIZE_FILL || value == (Control::SIZE_FILL | Control::SIZE_EXPAND)) {
		preset = SIZE_FLAGS_PRESET_FILL;
	} else if (value == Control::SIZE_SHRINK_BEGIN || value == (Control::SIZE_SHRINK_BEGIN | Control::SIZE_EXPAND)) {
		preset = SIZE_FLAGS_PRESET_SHRINK_BEGIN;
	} else if (value == Control::SIZE_SHRINK_CENTER || value == (Control::SIZE_SHRINK_CENTER | Control::SIZE_EXPAND)) {
		preset = SIZE_FLAGS_PRESET_SHRINK_CENTER;
	} else if (value == Control::SIZE_SHRINK_END || value == (Control::SIZE_SHRINK_END | Control::SIZE_EXPAND)) {
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

	HashMap<int, String> flags;
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
	String begin_preset_icon = SNAME("ControlAlignCenterLeft");
	String end_preset_icon = SNAME("ControlAlignCenterRight");
	if (vertical) {
		wide_preset_icon = SNAME("ControlAlignVCenterWide");
		begin_preset_icon = SNAME("ControlAlignCenterTop");
		end_preset_icon = SNAME("ControlAlignCenterBottom");
	}

	flag_presets->clear();
	if (flags.has(SIZE_FILL)) {
		flag_presets->add_icon_item(gui_base->get_theme_icon(wide_preset_icon, SNAME("EditorIcons")), TTR("Fill"), SIZE_FLAGS_PRESET_FILL);
	}
	// Shrink Begin is the same as no flags at all, as such it cannot be disabled.
	flag_presets->add_icon_item(gui_base->get_theme_icon(begin_preset_icon, SNAME("EditorIcons")), TTR("Shrink Begin"), SIZE_FLAGS_PRESET_SHRINK_BEGIN);
	if (flags.has(SIZE_SHRINK_CENTER)) {
		flag_presets->add_icon_item(gui_base->get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")), TTR("Shrink Center"), SIZE_FLAGS_PRESET_SHRINK_CENTER);
	}
	if (flags.has(SIZE_SHRINK_END)) {
		flag_presets->add_icon_item(gui_base->get_theme_icon(end_preset_icon, SNAME("EditorIcons")), TTR("Shrink End"), SIZE_FLAGS_PRESET_SHRINK_END);
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

void ControlEditorToolbar::_set_anchors_and_offsets_preset(Control::LayoutPreset p_preset) {
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
					undo_redo->add_do_method(control, "set_offsets_preset", p_preset, Control::PRESET_MODE_KEEP_SIZE);
					break;
				case PRESET_LEFT_WIDE:
				case PRESET_TOP_WIDE:
				case PRESET_RIGHT_WIDE:
				case PRESET_BOTTOM_WIDE:
				case PRESET_VCENTER_WIDE:
				case PRESET_HCENTER_WIDE:
				case PRESET_WIDE:
					undo_redo->add_do_method(control, "set_offsets_preset", p_preset, Control::PRESET_MODE_MINSIZE);
					break;
			}
			undo_redo->add_undo_method(control, "_edit_set_state", control->_edit_get_state());
		}
	}

	undo_redo->commit_action();

	anchors_mode = false;
	anchor_mode_button->set_pressed(anchors_mode);
}

void ControlEditorToolbar::_set_anchors_and_offsets_to_keep_ratio() {
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

			const bool use_anchors = control->get_meta("_edit_use_anchors_", false);
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

void ControlEditorToolbar::_set_anchors_preset(Control::LayoutPreset p_preset) {
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

void ControlEditorToolbar::_set_container_h_preset(Control::SizeFlags p_preset) {
	List<Node *> selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Change Horizontal Size Flags"));
	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (control) {
			undo_redo->add_do_method(control, "set_h_size_flags", p_preset);
			undo_redo->add_undo_method(control, "_edit_set_state", control->_edit_get_state());
		}
	}

	undo_redo->commit_action();
}

void ControlEditorToolbar::_set_container_v_preset(Control::SizeFlags p_preset) {
	List<Node *> selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Change Horizontal Size Flags"));
	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (control) {
			undo_redo->add_do_method(control, "set_v_size_flags", p_preset);
			undo_redo->add_undo_method(control, "_edit_set_state", control->_edit_get_state());
		}
	}

	undo_redo->commit_action();
}

Vector2 ControlEditorToolbar::_anchor_to_position(const Control *p_control, Vector2 anchor) {
	ERR_FAIL_COND_V(!p_control, Vector2());

	Transform2D parent_transform = p_control->get_transform().affine_inverse();
	Rect2 parent_rect = p_control->get_parent_anchorable_rect();

	if (p_control->is_layout_rtl()) {
		return parent_transform.xform(parent_rect.position + Vector2(parent_rect.size.x - parent_rect.size.x * anchor.x, parent_rect.size.y * anchor.y));
	} else {
		return parent_transform.xform(parent_rect.position + Vector2(parent_rect.size.x * anchor.x, parent_rect.size.y * anchor.y));
	}
}

Vector2 ControlEditorToolbar::_position_to_anchor(const Control *p_control, Vector2 position) {
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

void ControlEditorToolbar::_button_toggle_anchor_mode(bool p_status) {
	List<Control *> selection = _get_edited_controls(false, false);
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

bool ControlEditorToolbar::_is_node_locked(const Node *p_node) {
	return p_node->get_meta("_edit_lock_", false);
}

List<Control *> ControlEditorToolbar::_get_edited_controls(bool retrieve_locked, bool remove_controls_if_parent_in_selection) {
	List<Control *> selection;
	for (const KeyValue<Node *, Object *> &E : editor_selection->get_selection()) {
		Control *control = Object::cast_to<Control>(E.key);
		if (control && control->is_visible_in_tree() && control->get_viewport() == EditorNode::get_singleton()->get_scene_root() && (retrieve_locked || !_is_node_locked(control))) {
			selection.push_back(control);
		}
	}

	if (remove_controls_if_parent_in_selection) {
		List<Control *> filtered_selection;
		for (Control *E : selection) {
			if (!selection.find(E->get_parent())) {
				filtered_selection.push_back(E);
			}
		}
		return filtered_selection;
	}

	return selection;
}

void ControlEditorToolbar::_popup_callback(int p_op) {
	switch (p_op) {
		case ANCHORS_AND_OFFSETS_PRESET_TOP_LEFT: {
			_set_anchors_and_offsets_preset(PRESET_TOP_LEFT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_TOP_RIGHT: {
			_set_anchors_and_offsets_preset(PRESET_TOP_RIGHT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_BOTTOM_LEFT: {
			_set_anchors_and_offsets_preset(PRESET_BOTTOM_LEFT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_BOTTOM_RIGHT: {
			_set_anchors_and_offsets_preset(PRESET_BOTTOM_RIGHT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_CENTER_LEFT: {
			_set_anchors_and_offsets_preset(PRESET_CENTER_LEFT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_CENTER_RIGHT: {
			_set_anchors_and_offsets_preset(PRESET_CENTER_RIGHT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_CENTER_TOP: {
			_set_anchors_and_offsets_preset(PRESET_CENTER_TOP);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_CENTER_BOTTOM: {
			_set_anchors_and_offsets_preset(PRESET_CENTER_BOTTOM);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_CENTER: {
			_set_anchors_and_offsets_preset(PRESET_CENTER);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_TOP_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_TOP_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_LEFT_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_LEFT_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_RIGHT_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_RIGHT_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_BOTTOM_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_BOTTOM_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_VCENTER_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_VCENTER_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_HCENTER_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_HCENTER_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_WIDE: {
			_set_anchors_and_offsets_preset(Control::PRESET_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_KEEP_RATIO: {
			_set_anchors_and_offsets_to_keep_ratio();
		} break;

		case ANCHORS_PRESET_TOP_LEFT: {
			_set_anchors_preset(PRESET_TOP_LEFT);
		} break;
		case ANCHORS_PRESET_TOP_RIGHT: {
			_set_anchors_preset(PRESET_TOP_RIGHT);
		} break;
		case ANCHORS_PRESET_BOTTOM_LEFT: {
			_set_anchors_preset(PRESET_BOTTOM_LEFT);
		} break;
		case ANCHORS_PRESET_BOTTOM_RIGHT: {
			_set_anchors_preset(PRESET_BOTTOM_RIGHT);
		} break;
		case ANCHORS_PRESET_CENTER_LEFT: {
			_set_anchors_preset(PRESET_CENTER_LEFT);
		} break;
		case ANCHORS_PRESET_CENTER_RIGHT: {
			_set_anchors_preset(PRESET_CENTER_RIGHT);
		} break;
		case ANCHORS_PRESET_CENTER_TOP: {
			_set_anchors_preset(PRESET_CENTER_TOP);
		} break;
		case ANCHORS_PRESET_CENTER_BOTTOM: {
			_set_anchors_preset(PRESET_CENTER_BOTTOM);
		} break;
		case ANCHORS_PRESET_CENTER: {
			_set_anchors_preset(PRESET_CENTER);
		} break;
		case ANCHORS_PRESET_TOP_WIDE: {
			_set_anchors_preset(PRESET_TOP_WIDE);
		} break;
		case ANCHORS_PRESET_LEFT_WIDE: {
			_set_anchors_preset(PRESET_LEFT_WIDE);
		} break;
		case ANCHORS_PRESET_RIGHT_WIDE: {
			_set_anchors_preset(PRESET_RIGHT_WIDE);
		} break;
		case ANCHORS_PRESET_BOTTOM_WIDE: {
			_set_anchors_preset(PRESET_BOTTOM_WIDE);
		} break;
		case ANCHORS_PRESET_VCENTER_WIDE: {
			_set_anchors_preset(PRESET_VCENTER_WIDE);
		} break;
		case ANCHORS_PRESET_HCENTER_WIDE: {
			_set_anchors_preset(PRESET_HCENTER_WIDE);
		} break;
		case ANCHORS_PRESET_WIDE: {
			_set_anchors_preset(Control::PRESET_WIDE);
		} break;

		case CONTAINERS_H_PRESET_FILL: {
			_set_container_h_preset(Control::SIZE_FILL);
		} break;
		case CONTAINERS_H_PRESET_FILL_EXPAND: {
			_set_container_h_preset(Control::SIZE_EXPAND_FILL);
		} break;
		case CONTAINERS_H_PRESET_SHRINK_BEGIN: {
			_set_container_h_preset(Control::SIZE_SHRINK_BEGIN);
		} break;
		case CONTAINERS_H_PRESET_SHRINK_CENTER: {
			_set_container_h_preset(Control::SIZE_SHRINK_CENTER);
		} break;
		case CONTAINERS_H_PRESET_SHRINK_END: {
			_set_container_h_preset(Control::SIZE_SHRINK_END);
		} break;

		case CONTAINERS_V_PRESET_FILL: {
			_set_container_v_preset(Control::SIZE_FILL);
		} break;
		case CONTAINERS_V_PRESET_FILL_EXPAND: {
			_set_container_v_preset(Control::SIZE_EXPAND_FILL);
		} break;
		case CONTAINERS_V_PRESET_SHRINK_BEGIN: {
			_set_container_v_preset(Control::SIZE_SHRINK_BEGIN);
		} break;
		case CONTAINERS_V_PRESET_SHRINK_CENTER: {
			_set_container_v_preset(Control::SIZE_SHRINK_CENTER);
		} break;
		case CONTAINERS_V_PRESET_SHRINK_END: {
			_set_container_v_preset(Control::SIZE_SHRINK_END);
		} break;
	}
}

void ControlEditorToolbar::_selection_changed() {
	// Update the anchors_mode.
	int nb_controls = 0;
	int nb_valid_controls = 0;
	int nb_anchors_mode = 0;

	List<Node *> selection = editor_selection->get_selected_node_list();
	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (!control) {
			continue;
		}

		nb_controls++;
		if (Object::cast_to<Container>(control->get_parent())) {
			continue;
		}

		nb_valid_controls++;
		if (control->get_meta("_edit_use_anchors_", false)) {
			nb_anchors_mode++;
		}
	}

	anchors_mode = (nb_valid_controls == nb_anchors_mode);
	anchor_mode_button->set_pressed(anchors_mode);

	if (nb_controls > 0) {
		set_physics_process(true);
	} else {
		set_physics_process(false);
		set_visible(false);
	}
}

void ControlEditorToolbar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			anchor_presets_menu->set_icon(get_theme_icon(SNAME("ControlLayout"), SNAME("EditorIcons")));

			PopupMenu *p = anchor_presets_menu->get_popup();
			p->clear();
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignTopLeft"), SNAME("EditorIcons")), TTR("Top Left"), ANCHORS_AND_OFFSETS_PRESET_TOP_LEFT);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignTopRight"), SNAME("EditorIcons")), TTR("Top Right"), ANCHORS_AND_OFFSETS_PRESET_TOP_RIGHT);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomRight"), SNAME("EditorIcons")), TTR("Bottom Right"), ANCHORS_AND_OFFSETS_PRESET_BOTTOM_RIGHT);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomLeft"), SNAME("EditorIcons")), TTR("Bottom Left"), ANCHORS_AND_OFFSETS_PRESET_BOTTOM_LEFT);
			p->add_separator();
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterLeft"), SNAME("EditorIcons")), TTR("Center Left"), ANCHORS_AND_OFFSETS_PRESET_CENTER_LEFT);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterTop"), SNAME("EditorIcons")), TTR("Center Top"), ANCHORS_AND_OFFSETS_PRESET_CENTER_TOP);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterRight"), SNAME("EditorIcons")), TTR("Center Right"), ANCHORS_AND_OFFSETS_PRESET_CENTER_RIGHT);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterBottom"), SNAME("EditorIcons")), TTR("Center Bottom"), ANCHORS_AND_OFFSETS_PRESET_CENTER_BOTTOM);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")), TTR("Center"), ANCHORS_AND_OFFSETS_PRESET_CENTER);
			p->add_separator();
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignLeftWide"), SNAME("EditorIcons")), TTR("Left Wide"), ANCHORS_AND_OFFSETS_PRESET_LEFT_WIDE);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignTopWide"), SNAME("EditorIcons")), TTR("Top Wide"), ANCHORS_AND_OFFSETS_PRESET_TOP_WIDE);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignRightWide"), SNAME("EditorIcons")), TTR("Right Wide"), ANCHORS_AND_OFFSETS_PRESET_RIGHT_WIDE);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomWide"), SNAME("EditorIcons")), TTR("Bottom Wide"), ANCHORS_AND_OFFSETS_PRESET_BOTTOM_WIDE);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignVCenterWide"), SNAME("EditorIcons")), TTR("VCenter Wide"), ANCHORS_AND_OFFSETS_PRESET_VCENTER_WIDE);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignHCenterWide"), SNAME("EditorIcons")), TTR("HCenter Wide"), ANCHORS_AND_OFFSETS_PRESET_HCENTER_WIDE);
			p->add_separator();
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignWide"), SNAME("EditorIcons")), TTR("Full Rect"), ANCHORS_AND_OFFSETS_PRESET_WIDE);
			p->add_icon_item(get_theme_icon(SNAME("Anchor"), SNAME("EditorIcons")), TTR("Keep Current Ratio"), ANCHORS_AND_OFFSETS_PRESET_KEEP_RATIO);
			p->set_item_tooltip(19, TTR("Adjust anchors and offsets to match the current rect size."));

			p->add_separator();
			p->add_submenu_item(TTR("Anchors only"), "Anchors");
			p->set_item_icon(21, get_theme_icon(SNAME("Anchor"), SNAME("EditorIcons")));

			anchors_popup->clear();
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignTopLeft"), SNAME("EditorIcons")), TTR("Top Left"), ANCHORS_PRESET_TOP_LEFT);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignTopRight"), SNAME("EditorIcons")), TTR("Top Right"), ANCHORS_PRESET_TOP_RIGHT);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomRight"), SNAME("EditorIcons")), TTR("Bottom Right"), ANCHORS_PRESET_BOTTOM_RIGHT);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomLeft"), SNAME("EditorIcons")), TTR("Bottom Left"), ANCHORS_PRESET_BOTTOM_LEFT);
			anchors_popup->add_separator();
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterLeft"), SNAME("EditorIcons")), TTR("Center Left"), ANCHORS_PRESET_CENTER_LEFT);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterTop"), SNAME("EditorIcons")), TTR("Center Top"), ANCHORS_PRESET_CENTER_TOP);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterRight"), SNAME("EditorIcons")), TTR("Center Right"), ANCHORS_PRESET_CENTER_RIGHT);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterBottom"), SNAME("EditorIcons")), TTR("Center Bottom"), ANCHORS_PRESET_CENTER_BOTTOM);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")), TTR("Center"), ANCHORS_PRESET_CENTER);
			anchors_popup->add_separator();
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignLeftWide"), SNAME("EditorIcons")), TTR("Left Wide"), ANCHORS_PRESET_LEFT_WIDE);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignTopWide"), SNAME("EditorIcons")), TTR("Top Wide"), ANCHORS_PRESET_TOP_WIDE);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignRightWide"), SNAME("EditorIcons")), TTR("Right Wide"), ANCHORS_PRESET_RIGHT_WIDE);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomWide"), SNAME("EditorIcons")), TTR("Bottom Wide"), ANCHORS_PRESET_BOTTOM_WIDE);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignVCenterWide"), SNAME("EditorIcons")), TTR("VCenter Wide"), ANCHORS_PRESET_VCENTER_WIDE);
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignHCenterWide"), SNAME("EditorIcons")), TTR("HCenter Wide"), ANCHORS_PRESET_HCENTER_WIDE);
			anchors_popup->add_separator();
			anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignWide"), SNAME("EditorIcons")), TTR("Full Rect"), ANCHORS_PRESET_WIDE);

			anchor_mode_button->set_icon(get_theme_icon(SNAME("Anchor"), SNAME("EditorIcons")));

			container_h_presets_menu->set_icon(get_theme_icon(SNAME("Container"), SNAME("EditorIcons")));
			container_v_presets_menu->set_icon(get_theme_icon(SNAME("Container"), SNAME("EditorIcons")));

			p = container_h_presets_menu->get_popup();
			p->clear();
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignHCenterWide"), SNAME("EditorIcons")), TTR("Fill"), CONTAINERS_H_PRESET_FILL);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignHCenterWide"), SNAME("EditorIcons")), TTR("Fill & Expand"), CONTAINERS_H_PRESET_FILL_EXPAND);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterLeft"), SNAME("EditorIcons")), TTR("Shrink Begin"), CONTAINERS_H_PRESET_SHRINK_BEGIN);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")), TTR("Shrink Center"), CONTAINERS_H_PRESET_SHRINK_CENTER);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterRight"), SNAME("EditorIcons")), TTR("Shrink End"), CONTAINERS_H_PRESET_SHRINK_END);

			p = container_v_presets_menu->get_popup();
			p->clear();
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignVCenterWide"), SNAME("EditorIcons")), TTR("Fill"), CONTAINERS_V_PRESET_FILL);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignVCenterWide"), SNAME("EditorIcons")), TTR("Fill & Expand"), CONTAINERS_V_PRESET_FILL_EXPAND);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterTop"), SNAME("EditorIcons")), TTR("Shrink Begin"), CONTAINERS_V_PRESET_SHRINK_BEGIN);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")), TTR("Shrink Center"), CONTAINERS_V_PRESET_SHRINK_CENTER);
			p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenterBottom"), SNAME("EditorIcons")), TTR("Shrink End"), CONTAINERS_V_PRESET_SHRINK_END);
		} break;

		case NOTIFICATION_PHYSICS_PROCESS: {
			bool has_control_parents = false;
			bool has_container_parents = false;

			// Update the viewport if the canvas_item changes
			List<Control *> selection = _get_edited_controls(true);
			for (Control *control : selection) {
				if (Object::cast_to<Control>(control->get_parent())) {
					has_control_parents = true;
				}
				if (Object::cast_to<Container>(control->get_parent())) {
					has_container_parents = true;
				}
			}

			// Show / Hide the control layout buttons.
			if (selection.size() > 0) {
				set_visible(true);

				// Toggle anchor and container layout buttons depending on parents of the selected nodes.
				// - If there are no control parents, enable everything.
				// - If there are container parents, then enable only container buttons.
				// - If there are NO container parents, then enable only anchor buttons.
				bool enable_anchors = false;
				bool enable_containers = false;
				if (!has_control_parents) {
					enable_anchors = true;
					enable_containers = true;
				} else if (has_container_parents) {
					enable_containers = true;
				} else {
					enable_anchors = true;
				}

				if (enable_anchors) {
					anchor_presets_menu->set_visible(true);
					anchor_mode_button->set_visible(true);
				} else {
					anchor_presets_menu->set_visible(false);
					anchor_mode_button->set_visible(false);
				}

				if (enable_containers) {
					container_h_presets_menu->set_visible(true);
					container_v_presets_menu->set_visible(true);
				} else {
					container_h_presets_menu->set_visible(false);
					container_v_presets_menu->set_visible(false);
				}
			} else {
				set_visible(false);
			}
		} break;
	}
}

ControlEditorToolbar::ControlEditorToolbar() {
	anchor_presets_menu = memnew(MenuButton);
	anchor_presets_menu->set_shortcut_context(this);
	anchor_presets_menu->set_text(TTR("Anchors"));
	anchor_presets_menu->set_tooltip(TTR("Presets for the anchor and offset values of a Control node."));
	add_child(anchor_presets_menu);
	anchor_presets_menu->set_switch_on_hover(true);

	PopupMenu *p = anchor_presets_menu->get_popup();
	p->connect("id_pressed", callable_mp(this, &ControlEditorToolbar::_popup_callback));

	anchors_popup = memnew(PopupMenu);
	p->add_child(anchors_popup);
	anchors_popup->set_name("Anchors");
	anchors_popup->connect("id_pressed", callable_mp(this, &ControlEditorToolbar::_popup_callback));

	anchor_mode_button = memnew(Button);
	anchor_mode_button->set_flat(true);
	anchor_mode_button->set_toggle_mode(true);
	anchor_mode_button->set_tooltip(TTR("When active, moving Control nodes changes their anchors instead of their offsets."));
	add_child(anchor_mode_button);
	anchor_mode_button->connect("toggled", callable_mp(this, &ControlEditorToolbar::_button_toggle_anchor_mode));

	add_child(memnew(VSeparator));

	container_h_presets_menu = memnew(MenuButton);
	container_h_presets_menu->set_shortcut_context(this);
	container_h_presets_menu->set_text(TTR("Horizontal"));
	container_h_presets_menu->set_tooltip(TTR("Horizontal sizing setting for children of a Container node."));
	add_child(container_h_presets_menu);
	container_h_presets_menu->set_switch_on_hover(true);

	p = container_h_presets_menu->get_popup();
	p->connect("id_pressed", callable_mp(this, &ControlEditorToolbar::_popup_callback));

	container_v_presets_menu = memnew(MenuButton);
	container_v_presets_menu->set_shortcut_context(this);
	container_v_presets_menu->set_text(TTR("Vertical"));
	container_v_presets_menu->set_tooltip(TTR("Vertical sizing setting for children of a Container node."));
	add_child(container_v_presets_menu);
	container_v_presets_menu->set_switch_on_hover(true);

	p = container_v_presets_menu->get_popup();
	p->connect("id_pressed", callable_mp(this, &ControlEditorToolbar::_popup_callback));

	undo_redo = EditorNode::get_singleton()->get_undo_redo();
	editor_selection = EditorNode::get_singleton()->get_editor_selection();
	editor_selection->add_editor_plugin(this);
	editor_selection->connect("selection_changed", callable_mp(this, &ControlEditorToolbar::_selection_changed));

	singleton = this;
}

ControlEditorToolbar *ControlEditorToolbar::singleton = nullptr;

ControlEditorPlugin::ControlEditorPlugin() {
	toolbar = memnew(ControlEditorToolbar);
	toolbar->hide();
	add_control_to_container(CONTAINER_CANVAS_EDITOR_MENU, toolbar);

	Ref<EditorInspectorPluginControl> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}

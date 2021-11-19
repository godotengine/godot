/*************************************************************************/
/*  control_editor_plugin.h                                              */
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

#ifndef CONTROL_EDITOR_PLUGIN_H
#define CONTROL_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/margin_container.h"

class CheckBox;
class GridContainer;
class Label;
class OptionButton;
class PanelContainer;
class Separator;
class TextureRect;

class ControlPositioningWarning : public MarginContainer {
	GDCLASS(ControlPositioningWarning, MarginContainer);

	Control *control_node = nullptr;

	PanelContainer *bg_panel = nullptr;
	GridContainer *grid = nullptr;
	TextureRect *title_icon = nullptr;
	TextureRect *hint_icon = nullptr;
	Label *title_label = nullptr;
	Label *hint_label = nullptr;
	Control *hint_filler_left = nullptr;
	Control *hint_filler_right = nullptr;

	void _update_warning();
	void _update_toggler();
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

protected:
	void _notification(int p_notification);

public:
	void set_control(Control *p_node);

	ControlPositioningWarning();
};

class EditorPropertyAnchorsPreset : public EditorProperty {
	GDCLASS(EditorPropertyAnchorsPreset, EditorProperty);
	OptionButton *options;

	void _option_selected(int p_which);

protected:
	virtual void _set_read_only(bool p_read_only) override;

public:
	void setup(const Vector<String> &p_options);
	virtual void update_property() override;
	EditorPropertyAnchorsPreset();
};

class EditorPropertySizeFlags : public EditorProperty {
	GDCLASS(EditorPropertySizeFlags, EditorProperty);

	enum FlagPreset {
		SIZE_FLAGS_PRESET_FILL,
		SIZE_FLAGS_PRESET_SHRINK_BEGIN,
		SIZE_FLAGS_PRESET_SHRINK_CENTER,
		SIZE_FLAGS_PRESET_SHRINK_END,
		SIZE_FLAGS_PRESET_CUSTOM,
	};

	OptionButton *flag_presets;
	CheckBox *flag_expand;
	VBoxContainer *flag_options;
	Vector<CheckBox *> flag_checks;

	bool vertical = false;

	bool keep_selected_preset = false;

	void _preset_selected(int p_which);
	void _expand_toggled();
	void _flag_toggled();

protected:
	virtual void _set_read_only(bool p_read_only) override;

public:
	void setup(const Vector<String> &p_options, bool p_vertical);
	virtual void update_property() override;
	EditorPropertySizeFlags();
};

class EditorInspectorPluginControl : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginControl, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_group(Object *p_object, const String &p_group) override;
	virtual bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide = false) override;
};

class ControlAnchorPresetPicker : public MarginContainer {
	GDCLASS(ControlAnchorPresetPicker, MarginContainer);

	Map<LayoutPreset, Button *> preset_buttons;

	static constexpr int grid_separation = 0;

	void _add_row_button(HBoxContainer *p_row, const LayoutPreset p_preset, const String &p_name);
	void _add_separator(BoxContainer *p_box, Separator *p_separator);
	void _preset_button_pressed(const LayoutPreset p_preset);

protected:
	void _notification(int p_notification);
	static void _bind_methods();

public:
	ControlAnchorPresetPicker();
};

class ControlAnchorsEditorToolbar : public VBoxContainer {
	GDCLASS(ControlAnchorsEditorToolbar, VBoxContainer);

	UndoRedo *undo_redo;
	EditorSelection *editor_selection;

	CheckBox *picker_mode_button;
	Button *keep_ratio_button;
	CheckBox *anchor_mode_button;

	bool pick_anchors_and_offsets = true;
	bool anchors_mode = false;

	void _button_toggle_picker_mode(bool p_pressed);
	void _picker_preset_selected(int p_preset);
	void _button_toggle_anchor_mode(bool p_status);

	void _set_anchors_preset(LayoutPreset p_preset);
	void _set_anchors_and_offsets_preset(LayoutPreset p_preset);
	void _set_anchors_and_offsets_to_keep_ratio();

	Vector2 _anchor_to_position(const Control *p_control, Vector2 anchor);
	Vector2 _position_to_anchor(const Control *p_control, Vector2 position);
	bool _is_node_locked(const Node *p_node);
	List<Control *> _get_edited_controls();
	void _selection_changed();

protected:
	static ControlAnchorsEditorToolbar *singleton;

public:
	bool is_anchors_mode_enabled() { return anchors_mode; };

	static ControlAnchorsEditorToolbar *get_singleton() { return singleton; }

	ControlAnchorsEditorToolbar();
};

class ContainerSizeFlagPicker : public MarginContainer {
	GDCLASS(ContainerSizeFlagPicker, MarginContainer);

	Map<SizeFlags, Button *> flag_buttons;
	CheckBox *expand_button;

	static constexpr int grid_separation = 0;
	bool vertical = false;

	void _add_row_button(HBoxContainer *p_row, const SizeFlags p_flag, const String &p_name);
	void _add_separator(BoxContainer *p_box, Separator *p_separator);
	void _flag_button_pressed(const SizeFlags p_flag);

protected:
	void _notification(int p_notification);
	static void _bind_methods();

public:
	void set_allowed_flags(Vector<SizeFlags> &p_flags);

	ContainerSizeFlagPicker(bool p_vertical);
};

class ContainerEditorToolbar : public VBoxContainer {
	GDCLASS(ContainerEditorToolbar, VBoxContainer);

	UndoRedo *undo_redo;
	EditorSelection *editor_selection;

	ContainerSizeFlagPicker *container_h_picker;
	ContainerSizeFlagPicker *container_v_picker;

	void _container_flags_selected(int p_flags, bool p_vertical);
	void _selection_changed();

public:
	ContainerEditorToolbar();
};

class ControlEditorPlugin : public EditorPlugin {
	GDCLASS(ControlEditorPlugin, EditorPlugin);

	ControlAnchorsEditorToolbar *anchors_toolbar;
	ContainerEditorToolbar *container_toolbar;

public:
	virtual String get_name() const override { return "Control"; }

	ControlEditorPlugin();
};

#endif //CONTROL_EDITOR_PLUGIN_H

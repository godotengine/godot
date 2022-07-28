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
#include "scene/gui/check_box.h"
#include "scene/gui/control.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/texture_rect.h"

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
	OptionButton *options = nullptr;

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

	OptionButton *flag_presets = nullptr;
	CheckBox *flag_expand = nullptr;
	VBoxContainer *flag_options = nullptr;
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

class ControlEditorToolbar : public HBoxContainer {
	GDCLASS(ControlEditorToolbar, HBoxContainer);

	UndoRedo *undo_redo = nullptr;
	EditorSelection *editor_selection = nullptr;

	enum MenuOption {
		ANCHORS_AND_OFFSETS_PRESET_TOP_LEFT,
		ANCHORS_AND_OFFSETS_PRESET_TOP_RIGHT,
		ANCHORS_AND_OFFSETS_PRESET_BOTTOM_LEFT,
		ANCHORS_AND_OFFSETS_PRESET_BOTTOM_RIGHT,
		ANCHORS_AND_OFFSETS_PRESET_CENTER_LEFT,
		ANCHORS_AND_OFFSETS_PRESET_CENTER_RIGHT,
		ANCHORS_AND_OFFSETS_PRESET_CENTER_TOP,
		ANCHORS_AND_OFFSETS_PRESET_CENTER_BOTTOM,
		ANCHORS_AND_OFFSETS_PRESET_CENTER,
		ANCHORS_AND_OFFSETS_PRESET_TOP_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_LEFT_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_RIGHT_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_BOTTOM_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_VCENTER_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_HCENTER_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_FULL_RECT,

		ANCHORS_AND_OFFSETS_PRESET_KEEP_RATIO,

		ANCHORS_PRESET_TOP_LEFT,
		ANCHORS_PRESET_TOP_RIGHT,
		ANCHORS_PRESET_BOTTOM_LEFT,
		ANCHORS_PRESET_BOTTOM_RIGHT,
		ANCHORS_PRESET_CENTER_LEFT,
		ANCHORS_PRESET_CENTER_RIGHT,
		ANCHORS_PRESET_CENTER_TOP,
		ANCHORS_PRESET_CENTER_BOTTOM,
		ANCHORS_PRESET_CENTER,
		ANCHORS_PRESET_TOP_WIDE,
		ANCHORS_PRESET_LEFT_WIDE,
		ANCHORS_PRESET_RIGHT_WIDE,
		ANCHORS_PRESET_BOTTOM_WIDE,
		ANCHORS_PRESET_VCENTER_WIDE,
		ANCHORS_PRESET_HCENTER_WIDE,
		ANCHORS_PRESET_FULL_RECT,

		// Offsets Presets are not currently in use.
		OFFSETS_PRESET_TOP_LEFT,
		OFFSETS_PRESET_TOP_RIGHT,
		OFFSETS_PRESET_BOTTOM_LEFT,
		OFFSETS_PRESET_BOTTOM_RIGHT,
		OFFSETS_PRESET_CENTER_LEFT,
		OFFSETS_PRESET_CENTER_RIGHT,
		OFFSETS_PRESET_CENTER_TOP,
		OFFSETS_PRESET_CENTER_BOTTOM,
		OFFSETS_PRESET_CENTER,
		OFFSETS_PRESET_TOP_WIDE,
		OFFSETS_PRESET_LEFT_WIDE,
		OFFSETS_PRESET_RIGHT_WIDE,
		OFFSETS_PRESET_BOTTOM_WIDE,
		OFFSETS_PRESET_VCENTER_WIDE,
		OFFSETS_PRESET_HCENTER_WIDE,
		OFFSETS_PRESET_FULL_RECT,

		CONTAINERS_H_PRESET_FILL,
		CONTAINERS_H_PRESET_FILL_EXPAND,
		CONTAINERS_H_PRESET_SHRINK_BEGIN,
		CONTAINERS_H_PRESET_SHRINK_CENTER,
		CONTAINERS_H_PRESET_SHRINK_END,
		CONTAINERS_V_PRESET_FILL,
		CONTAINERS_V_PRESET_FILL_EXPAND,
		CONTAINERS_V_PRESET_SHRINK_BEGIN,
		CONTAINERS_V_PRESET_SHRINK_CENTER,
		CONTAINERS_V_PRESET_SHRINK_END,
	};

	MenuButton *anchor_presets_menu = nullptr;
	PopupMenu *anchors_popup = nullptr;
	MenuButton *container_h_presets_menu = nullptr;
	MenuButton *container_v_presets_menu = nullptr;

	Button *anchor_mode_button = nullptr;

	bool anchors_mode = false;

	void _set_anchors_preset(Control::LayoutPreset p_preset);
	void _set_anchors_and_offsets_preset(Control::LayoutPreset p_preset);
	void _set_anchors_and_offsets_to_keep_ratio();
	void _set_container_h_preset(Control::SizeFlags p_preset);
	void _set_container_v_preset(Control::SizeFlags p_preset);

	Vector2 _anchor_to_position(const Control *p_control, Vector2 anchor);
	Vector2 _position_to_anchor(const Control *p_control, Vector2 position);

	void _button_toggle_anchor_mode(bool p_status);

	bool _is_node_locked(const Node *p_node);
	List<Control *> _get_edited_controls(bool retrieve_locked = false, bool remove_controls_if_parent_in_selection = true);
	void _popup_callback(int p_op);
	void _selection_changed();

protected:
	void _notification(int p_notification);

	static ControlEditorToolbar *singleton;

public:
	bool is_anchors_mode_enabled() { return anchors_mode; };

	static ControlEditorToolbar *get_singleton() { return singleton; }

	ControlEditorToolbar();
};

class ControlEditorPlugin : public EditorPlugin {
	GDCLASS(ControlEditorPlugin, EditorPlugin);

	ControlEditorToolbar *toolbar = nullptr;

public:
	virtual String get_name() const override { return "Control"; }

	ControlEditorPlugin();
};

#endif // CONTROL_EDITOR_PLUGIN_H

/**************************************************************************/
/*  project_settings_editor.h                                             */
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

#pragma once

#include "core/config/project_settings.h"
#include "editor/editor_data.h"
#include "editor/import/import_defaults_editor.h"
#include "editor/inspector/editor_sectioned_inspector.h"
#include "editor/plugins/editor_plugin_settings.h"
#include "editor/scene/group_settings_editor.h"
#include "editor/settings/action_map_editor.h"
#include "editor/settings/editor_autoload_settings.h"
#include "editor/shader/shader_globals_editor.h"
#include "editor/translations/localization_editor.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"

class EditorVariantTypeOptionButton;
class FileSystemDock;

class ProjectSettingsEditor : public AcceptDialog {
	GDCLASS(ProjectSettingsEditor, AcceptDialog);

	inline static ProjectSettingsEditor *singleton = nullptr;

	struct Preset {
	private:
		static inline bool updating = false;
		static inline bool custom = false;

	public:
		enum StretchMode {
			DISABLED,
			CANVAS_ITEMS,
			VIEWPORT,
		};

		enum StretchAspect {
			IGNORE,
			KEEP,
			KEEP_WIDTH,
			KEEP_HEIGHT,
			EXPAND,
		};

		enum StretchScaleMode {
			FRACTIONAL,
			INTEGER,
		};

		enum Orientation {
			LANDSCAPE,
			PORTRAIT,
			REVERSE_LANDSCAPE,
			REVERSE_PORTRAIT,
			SENSOR_LANDSCAPE,
			SENSOR_PORTRAIT,
			SENSOR,
		};

		const Vector<String> STRETCH_MODE = {
			"disabled",
			"canvas_items",
			"viewport",
		};
		const Vector<String> STRETCH_ASPECT = {
			"ignore",
			"keep",
			"keep_width",
			"keep_height",
			"expand",
		};
		const Vector<String> STRETCH_SCALE_MODE = {
			"fractional",
			"integer",
		};

		HashMap<StringName, Variant> data;

		Preset(int p_width = 1152, int p_height = 648, StretchMode p_stretch_mode = DISABLED,
				StretchAspect p_stretch_aspect = KEEP, StretchScaleMode p_scale_mode = FRACTIONAL,
				Orientation p_orientation = LANDSCAPE, float p_theme_scale = 1.0f) {
			data["display/window/size/viewport_width"] = p_width;
			data["display/window/size/viewport_height"] = p_height;
			data["display/window/size/mode"] = GLOBAL_GET_INIT("display/window/size/mode");
			data["display/window/size/initial_position_type"] = GLOBAL_GET_INIT("display/window/size/initial_position_type");
			data["display/window/size/initial_position"] = GLOBAL_GET_INIT("display/window/size/initial_position");
			data["display/window/size/initial_screen"] = GLOBAL_GET_INIT("display/window/size/initial_screen");
			data["display/window/size/resizable"] = GLOBAL_GET_INIT("display/window/size/resizable");
			data["display/window/size/borderless"] = GLOBAL_GET_INIT("display/window/size/borderless");
			data["display/window/size/always_on_top"] = GLOBAL_GET_INIT("display/window/size/always_on_top");
			data["display/window/size/transparent"] = GLOBAL_GET_INIT("display/window/size/transparent");
			data["display/window/size/extend_to_title"] = GLOBAL_GET_INIT("display/window/size/extend_to_title");
			data["display/window/size/no_focus"] = GLOBAL_GET_INIT("display/window/size/no_focus");
			data["display/window/size/sharp_corners"] = GLOBAL_GET_INIT("display/window/size/sharp_corners");
			data["display/window/size/minimize_disabled"] = GLOBAL_GET_INIT("display/window/size/minimize_disabled");
			data["display/window/size/maximize_disabled"] = GLOBAL_GET_INIT("display/window/size/maximize_disabled");
			data["display/window/size/window_width_override"] = GLOBAL_GET_INIT("display/window/size/window_width_override");
			data["display/window/size/window_height_override"] = GLOBAL_GET_INIT("display/window/size/window_height_override");
			data["display/window/energy_saving/keep_screen_on"] = GLOBAL_GET_INIT("display/window/energy_saving/keep_screen_on");
			data["display/window/subwindows/embed_subwindows"] = GLOBAL_GET_INIT("display/window/subwindows/embed_subwindows");
			data["display/window/frame_pacing/android/enable_frame_pacing"] = GLOBAL_GET_INIT("display/window/frame_pacing/android/enable_frame_pacin");
			data["display/window/frame_pacing/android/swappy_mode"] = GLOBAL_GET_INIT("display/window/frame_pacing/android/swappy_mode");
			data["display/window/stretch/mode"] = STRETCH_MODE[p_stretch_mode];
			data["display/window/stretch/aspect"] = STRETCH_ASPECT[p_stretch_aspect];
			data["display/window/stretch/scale"] = GLOBAL_GET_INIT("display/window/stretch/scale");
			data["display/window/stretch/scale_mode"] = STRETCH_SCALE_MODE[p_scale_mode];
			data["display/window/dpi/allow_hidpi"] = GLOBAL_GET_INIT("display/window/dpi/allow_hidpi");
			data["display/window/per_pixel_transparency/allowed"] = GLOBAL_GET_INIT("display/window/per_pixel_transparency/allowed");
			data["display/window/handheld/orientation"] = p_orientation;
			data["display/window/vsync/vsync_mode"] = GLOBAL_GET_INIT("display/window/vsync/vsync_mode");
			data["display/window/ios/allow_high_refresh_rate"] = GLOBAL_GET_INIT("display/window/ios/allow_high_refresh_rate");
			data["display/window/ios/hide_home_indicator"] = GLOBAL_GET_INIT("display/window/ios/hide_home_indicator");
			data["display/window/ios/hide_status_bar"] = GLOBAL_GET_INIT("display/window/ios/hide_status_bar");
			data["display/window/ios/suppress_ui_gesture"] = GLOBAL_GET_INIT("display/window/ios/suppress_ui_gesture");
			data["gui/theme/default_theme_scale"] = p_theme_scale;
		}

		static void set_updating(bool p_updating) { updating = p_updating; }
		static bool is_updating() { return updating; }
		static void set_custom(bool p_custom) { custom = p_custom; }
		static bool is_custom() { return custom; }
	};

	enum {
		FEATURE_ALL,
		FEATURE_CUSTOM,
		FEATURE_FIRST,
	};

	Vector<Preset> window_presets;

	ProjectSettings *ps = nullptr;
	Timer *timer = nullptr;

	TabContainer *tab_container = nullptr;
	VBoxContainer *general_editor = nullptr;
	SectionedInspector *general_settings_inspector = nullptr;
	ActionMapEditor *action_map_editor = nullptr;
	LocalizationEditor *localization_editor = nullptr;
	EditorAutoloadSettings *autoload_settings = nullptr;
	ShaderGlobalsEditor *shaders_global_shader_uniforms_editor = nullptr;
	GroupSettingsEditor *group_settings = nullptr;
	EditorPluginSettings *plugin_settings = nullptr;

	LineEdit *search_box = nullptr;
	CheckButton *advanced = nullptr;

	HBoxContainer *custom_properties = nullptr;
	LineEdit *property_box = nullptr;
	OptionButton *feature_box = nullptr;
	EditorVariantTypeOptionButton *type_box = nullptr;
	Button *add_button = nullptr;
	Button *del_button = nullptr;

	Label *restart_label = nullptr;
	TextureRect *restart_icon = nullptr;
	PanelContainer *restart_container = nullptr;
	Button *restart_close_button = nullptr;

	ImportDefaultsEditor *import_defaults_editor = nullptr;
	EditorData *data = nullptr;

	bool settings_changed = false;
	bool pending_override_notify = false;

	void _on_category_changed(const String &p_new_category);
	void _on_editor_override_deleted(const String &p_setting);

	void _advanced_toggled(bool p_button_pressed);
	void _update_advanced(bool p_is_advanced);
	void _property_box_changed(const String &p_text);
	void _update_property_box();
	void _feature_selected(int p_index);
	void _select_type(Variant::Type p_type);

	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

	String _get_setting_name() const;
	void _setting_edited(const String &p_name);
	void _setting_selected(const String &p_path);
	void _settings_popup();
	void _settings_changed();
	void _add_setting();
	void _delete_setting();

	void _tabs_tab_changed(int p_tab);
	void _focus_current_search_box();
	void _focus_current_path_box();

	void _editor_restart_request();
	void _editor_restart();
	void _editor_restart_close();

	void _add_feature_overrides();

	void _action_added(const String &p_name);
	void _action_edited(const String &p_name, const Dictionary &p_action);
	void _action_removed(const String &p_name);
	void _action_renamed(const String &p_old_name, const String &p_new_name);
	void _action_reordered(const String &p_action_name, const String &p_relative_to, bool p_before);
	void _update_action_map_editor();
	void _update_theme();
	void _save();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static ProjectSettingsEditor *get_singleton() { return singleton; }

	void popup_project_settings(bool p_clear_filter = false);
	void popup_for_override(const String &p_override);

	void set_plugins_page();
	void set_general_page(const String &p_category);
	void update_plugins();
	void init_autoloads();

	void set_filter(const String &p_filter);

	EditorAutoloadSettings *get_autoload_settings() { return autoload_settings; }
	GroupSettingsEditor *get_group_settings() { return group_settings; }
	TabContainer *get_tabs() { return tab_container; }

	void queue_save();
	void connect_filesystem_dock_signals(FileSystemDock *p_fs_dock);

	ProjectSettingsEditor(EditorData *p_data);
};

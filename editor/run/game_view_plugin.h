/**************************************************************************/
/*  game_view_plugin.h                                                    */
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

#include "editor/debugger/editor_debugger_node.h"
#include "editor/debugger/editor_debugger_plugin.h"
#include "editor/editor_main_screen.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/debugger/scene_debugger.h"
#include "scene/gui/box_container.h"

class EmbeddedProcessBase;
class VSeparator;
class WindowWrapper;
class ScriptEditorDebugger;

class GameViewDebugger : public EditorDebuggerPlugin {
	GDCLASS(GameViewDebugger, EditorDebuggerPlugin);

private:
	Vector<Ref<EditorDebuggerSession>> sessions;

	bool is_feature_enabled = true;
	int node_type = RuntimeNodeSelect::NODE_TYPE_NONE;
	bool selection_visible = true;
	int select_mode = RuntimeNodeSelect::SELECT_MODE_SINGLE;
	bool mute_audio = false;
	EditorDebuggerNode::CameraOverride camera_override_mode = EditorDebuggerNode::OVERRIDE_INGAME;

	bool selection_avoid_locked = false;
	bool selection_prefer_group = false;

	void _session_started(Ref<EditorDebuggerSession> p_session);
	void _session_stopped();

	void _feature_profile_changed();

	struct ScreenshotCB {
		Callable cb;
		Rect2i rect;
	};

	int64_t scr_rq_id = 0;
	HashMap<uint64_t, ScreenshotCB> screenshot_callbacks;

	bool _msg_get_screenshot(const Array &p_args);

protected:
	static void _bind_methods();

public:
	virtual bool capture(const String &p_message, const Array &p_data, int p_session) override;
	virtual bool has_capture(const String &p_capture) const override;

	bool add_screenshot_callback(const Callable &p_callaback, const Rect2i &p_rect);

	void set_suspend(bool p_enabled);
	void next_frame();

	void set_time_scale(double p_scale);
	void reset_time_scale();

	void set_node_type(int p_type);
	void set_select_mode(int p_mode);

	void set_selection_visible(bool p_visible);

	void set_selection_avoid_locked(bool p_enabled);
	void set_selection_prefer_group(bool p_enabled);

	void set_debug_mute_audio(bool p_enabled);

	void set_camera_override(bool p_enabled);
	void set_camera_manipulate_mode(EditorDebuggerNode::CameraOverride p_mode);

	void reset_camera_2d_position();
	void reset_camera_3d_position();

	virtual void setup_session(int p_session_id) override;

	GameViewDebugger();
};

class GameView : public VBoxContainer {
	GDCLASS(GameView, VBoxContainer);

	enum {
		CAMERA_RESET_2D,
		CAMERA_RESET_3D,
		CAMERA_MODE_INGAME,
		CAMERA_MODE_EDITORS,
		EMBED_RUN_GAME_EMBEDDED,
		EMBED_MAKE_FLOATING_ON_PLAY,
		SELECTION_AVOID_LOCKED,
		SELECTION_PREFER_GROUP,
	};

	enum EmbedSizeMode {
		SIZE_MODE_FIXED,
		SIZE_MODE_KEEP_ASPECT,
		SIZE_MODE_STRETCH,
	};

	enum EmbedAvailability {
		EMBED_AVAILABLE,
		EMBED_NOT_AVAILABLE_FEATURE_NOT_SUPPORTED,
		EMBED_NOT_AVAILABLE_MINIMIZED,
		EMBED_NOT_AVAILABLE_MAXIMIZED,
		EMBED_NOT_AVAILABLE_FULLSCREEN,
		EMBED_NOT_AVAILABLE_SINGLE_WINDOW_MODE,
		EMBED_NOT_AVAILABLE_PROJECT_DISPLAY_DRIVER,
		EMBED_NOT_AVAILABLE_HEADLESS,
	};

	inline static GameView *singleton = nullptr;

	Ref<GameViewDebugger> debugger;
	WindowWrapper *window_wrapper = nullptr;

	bool is_feature_enabled = true;
	int active_sessions = 0;
	int screen_index_before_start = -1;
	ScriptEditorDebugger *embedded_script_debugger = nullptr;

	bool embed_on_play = true;
	bool make_floating_on_play = true;
	EmbedSizeMode embed_size_mode = SIZE_MODE_FIXED;
	bool paused = false;
	Size2 size_paused;

	Rect2i floating_window_rect;
	int floating_window_screen = -1;

	bool debug_mute_audio = false;

	bool selection_avoid_locked = false;
	bool selection_prefer_group = false;

	Button *suspend_button = nullptr;
	Button *next_frame_button = nullptr;

	Button *node_type_button[RuntimeNodeSelect::NODE_TYPE_MAX];
	Button *select_mode_button[RuntimeNodeSelect::SELECT_MODE_MAX];

	Button *hide_selection = nullptr;
	MenuButton *selection_options_menu = nullptr;

	Button *debug_mute_audio_button = nullptr;

	Button *camera_override_button = nullptr;
	MenuButton *camera_override_menu = nullptr;

	HBoxContainer *embedding_hb = nullptr;
	MenuButton *embed_options_menu = nullptr;
	Label *game_size_label = nullptr;
	PanelContainer *panel = nullptr;
	EmbeddedProcessBase *embedded_process = nullptr;
	Label *state_label = nullptr;

	int const DEFAULT_TIME_SCALE_INDEX = 5;
	Array time_scale_range = { 0.0625f, 0.125f, 0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f, 4.0f, 8.0f, 16.0f };
	Array time_scale_label = { "1/16", "1/8", "1/4", "1/2", "3/4", "1.0", "1.25", "1.5", "1.75", "2.0", "4.0", "8.0", "16.0" };
	int time_scale_index = DEFAULT_TIME_SCALE_INDEX;

	MenuButton *speed_state_button = nullptr;
	Button *reset_speed_button = nullptr;

	void _sessions_changed();

	void _update_debugger_buttons();

	void _handle_shortcut_requested(int p_embed_action);
	void _toggle_suspend_button();
	void _suspend_button_toggled(bool p_pressed);

	void _node_type_pressed(int p_option);
	void _select_mode_pressed(int p_option);
	void _selection_options_menu_id_pressed(int p_id);
	void _embed_options_menu_menu_id_pressed(int p_id);

	void _reset_time_scales();
	void _speed_state_menu_pressed(int p_id);
	void _update_speed_buttons();
	void _update_speed_state_color();
	void _update_speed_state_size();

	void _play_pressed();
	static void _instance_starting_static(int p_idx, List<String> &r_arguments);
	void _instance_starting(int p_idx, List<String> &r_arguments);
	static bool _instance_rq_screenshot_static(const Callable &p_callback);
	bool _instance_rq_screenshot(const Callable &p_callback);
	void _stop_pressed();
	void _embedding_completed();
	void _embedding_failed();
	void _embedded_process_updated();
	void _embedded_process_focused();
	void _editor_or_project_settings_changed();

	EmbedAvailability _get_embed_available();
	void _update_ui();
	void _update_embed_menu_options();
	void _update_embed_window_size();
	void _update_arguments_for_instance(int p_idx, List<String> &r_arguments);
	void _show_update_window_wrapper();

	void _hide_selection_toggled(bool p_pressed);

	void _debug_mute_audio_button_pressed();

	void _camera_override_button_toggled(bool p_pressed);
	void _camera_override_menu_id_pressed(int p_id);

	void _window_close_request();
	void _update_floating_window_settings();
	void _attach_script_debugger();
	void _detach_script_debugger();
	void _remote_window_title_changed(String title);

	void _debugger_breaked(bool p_breaked, bool p_can_debug);

	void _feature_profile_changed();

protected:
	void _notification(int p_what);

public:
	void set_state(const Dictionary &p_state);
	Dictionary get_state() const;

	void set_window_layout(Ref<ConfigFile> p_layout);
	void get_window_layout(Ref<ConfigFile> p_layout);

	GameView(Ref<GameViewDebugger> p_debugger, EmbeddedProcessBase *p_embedded_process, WindowWrapper *p_wrapper);
};

class GameViewPluginBase : public EditorPlugin {
	GDCLASS(GameViewPluginBase, EditorPlugin);

#ifndef ANDROID_ENABLED
	GameView *game_view = nullptr;
	WindowWrapper *window_wrapper = nullptr;
#endif // ANDROID_ENABLED

	Ref<GameViewDebugger> debugger;

	String last_editor;

#ifndef ANDROID_ENABLED
	void _window_visibility_changed(bool p_visible);
#endif // ANDROID_ENABLED
	void _save_last_editor(const String &p_editor);
	void _focus_another_editor();
	bool _is_window_wrapper_enabled() const;

protected:
	void _notification(int p_what);
#ifndef ANDROID_ENABLED
	void setup(Ref<GameViewDebugger> p_debugger, EmbeddedProcessBase *p_embedded_process);
#endif

public:
	virtual String get_plugin_name() const override { return TTRC("Game"); }
	bool has_main_screen() const override { return true; }
	virtual void edit(Object *p_object) override {}
	virtual bool handles(Object *p_object) const override { return false; }
	virtual void selected_notify() override;

	Ref<GameViewDebugger> get_debugger() const { return debugger; }

#ifndef ANDROID_ENABLED
	virtual void make_visible(bool p_visible) override;

	virtual void set_window_layout(Ref<ConfigFile> p_layout) override;
	virtual void get_window_layout(Ref<ConfigFile> p_layout) override;
#endif // ANDROID_ENABLED
	GameViewPluginBase();
};

class GameViewPlugin : public GameViewPluginBase {
	GDCLASS(GameViewPlugin, GameViewPluginBase);

public:
	GameViewPlugin();
};

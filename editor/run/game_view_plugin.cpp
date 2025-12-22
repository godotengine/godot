/**************************************************************************/
/*  game_view_plugin.cpp                                                  */
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

#include "game_view_plugin.h"

#include "core/config/project_settings.h"
#include "core/debugger/debugger_marshalls.h"
#include "core/debugger/engine_debugger.h"
#include "core/string/translation_server.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/editor_interface.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/gui/window_wrapper.h"
#include "editor/run/editor_run_bar.h"
#include "editor/run/embedded_process.h"
#include "editor/run/run_instances_dialog.h"
#include "editor/settings/editor_feature_profile.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/separator.h"

void GameViewDebugger::_session_started(Ref<EditorDebuggerSession> p_session) {
	if (!is_feature_enabled) {
		return;
	}

	Dictionary settings;
	settings["debugger/max_node_selection"] = EDITOR_GET("debugger/max_node_selection");
	settings["editors/panning/2d_editor_panning_scheme"] = EDITOR_GET("editors/panning/2d_editor_panning_scheme");
	settings["editors/panning/simple_panning"] = EDITOR_GET("editors/panning/simple_panning");
	settings["editors/panning/warped_mouse_panning"] = EDITOR_GET("editors/panning/warped_mouse_panning");
	settings["editors/panning/2d_editor_pan_speed"] = EDITOR_GET("editors/panning/2d_editor_pan_speed");
	settings["editors/polygon_editor/point_grab_radius"] = EDITOR_GET("editors/polygon_editor/point_grab_radius");
	settings["canvas_item_editor/pan_view"] = DebuggerMarshalls::serialize_key_shortcut(ED_GET_SHORTCUT("canvas_item_editor/pan_view"));
	settings["box_selection_fill_color"] = EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("box_selection_fill_color"), EditorStringName(Editor));
	settings["box_selection_stroke_color"] = EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("box_selection_stroke_color"), EditorStringName(Editor));
	settings["editors/3d/default_fov"] = EDITOR_GET("editors/3d/default_fov");
	settings["editors/3d/default_z_near"] = EDITOR_GET("editors/3d/default_z_near");
	settings["editors/3d/default_z_far"] = EDITOR_GET("editors/3d/default_z_far");
	settings["editors/3d/navigation/invert_x_axis"] = EDITOR_GET("editors/3d/navigation/invert_x_axis");
	settings["editors/3d/navigation/invert_y_axis"] = EDITOR_GET("editors/3d/navigation/invert_y_axis");
	settings["editors/3d/navigation/warped_mouse_panning"] = EDITOR_GET("editors/3d/navigation/warped_mouse_panning");
	settings["editors/3d/freelook/freelook_base_speed"] = EDITOR_GET("editors/3d/freelook/freelook_base_speed");
	settings["editors/3d/freelook/freelook_sensitivity"] = EDITOR_GET("editors/3d/freelook/freelook_sensitivity");
	settings["editors/3d/navigation_feel/orbit_sensitivity"] = EDITOR_GET("editors/3d/navigation_feel/orbit_sensitivity");
	settings["editors/3d/navigation_feel/translation_sensitivity"] = EDITOR_GET("editors/3d/navigation_feel/translation_sensitivity");
	settings["editors/3d/selection_box_color"] = EDITOR_GET("editors/3d/selection_box_color");
	settings["editors/3d/freelook/freelook_base_speed"] = EDITOR_GET("editors/3d/freelook/freelook_base_speed");

	Array setup_data;
	setup_data.append(settings);
	p_session->send_message("scene:runtime_node_select_setup", setup_data);

	Array type;
	type.append(node_type);
	p_session->send_message("scene:runtime_node_select_set_type", type);
	Array visible;
	visible.append(selection_visible);
	p_session->send_message("scene:runtime_node_select_set_visible", visible);
	Array mode;
	mode.append(select_mode);
	p_session->send_message("scene:runtime_node_select_set_mode", mode);
	Array avoid_locked;
	avoid_locked.append(selection_avoid_locked);
	p_session->send_message("scene:runtime_node_select_set_avoid_locked", avoid_locked);
	Array prefer_group;
	prefer_group.append(selection_prefer_group);
	p_session->send_message("scene:runtime_node_select_set_prefer_group", prefer_group);
	Array mute_audio_data;
	mute_audio_data.append(mute_audio);
	p_session->send_message("scene:debug_mute_audio", mute_audio_data);

	Dictionary shortcut_settings;
	shortcut_settings["editor/suspend_resume_embedded_project"] = DebuggerMarshalls::serialize_key_shortcut(ED_GET_SHORTCUT("editor/suspend_resume_embedded_project"));
	shortcut_settings["editor/next_frame_embedded_project"] = DebuggerMarshalls::serialize_key_shortcut(ED_GET_SHORTCUT("editor/next_frame_embedded_project"));

	p_session->send_message("scene:setup_embedded_shortcuts", { shortcut_settings });

	emit_signal(SNAME("session_started"));
}

void GameViewDebugger::_session_stopped() {
	if (!is_feature_enabled) {
		return;
	}

	emit_signal(SNAME("session_stopped"));
}

void GameViewDebugger::set_suspend(bool p_enabled) {
	Array message;
	message.append(p_enabled);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:suspend_changed", message);
		}
	}
}

void GameViewDebugger::next_frame() {
	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:next_frame", Array());
		}
	}
}

void GameViewDebugger::set_time_scale(double p_scale) {
	Array message;
	message.append(p_scale);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:speed_changed", message);
		}
	}
}

void GameViewDebugger::reset_time_scale() {
	Array message;
	message.append(1.0);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:speed_changed", message);
		}
	}
}

void GameViewDebugger::set_node_type(int p_type) {
	node_type = p_type;

	Array message;
	message.append(p_type);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:runtime_node_select_set_type", message);
		}
	}
}

void GameViewDebugger::set_selection_visible(bool p_visible) {
	selection_visible = p_visible;

	Array message;
	message.append(p_visible);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:runtime_node_select_set_visible", message);
		}
	}
}

void GameViewDebugger::set_selection_avoid_locked(bool p_enabled) {
	selection_avoid_locked = p_enabled;

	Array message;
	message.append(p_enabled);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:runtime_node_select_set_avoid_locked", message);
		}
	}
}

void GameViewDebugger::set_selection_prefer_group(bool p_enabled) {
	selection_prefer_group = p_enabled;

	Array message;
	message.append(p_enabled);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:runtime_node_select_set_prefer_group", message);
		}
	}
}

void GameViewDebugger::set_select_mode(int p_mode) {
	select_mode = p_mode;

	Array message;
	message.append(p_mode);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:runtime_node_select_set_mode", message);
		}
	}
}

void GameViewDebugger::set_debug_mute_audio(bool p_enabled) {
	mute_audio = p_enabled;
	EditorDebuggerNode::get_singleton()->set_debug_mute_audio(p_enabled);
}

void GameViewDebugger::set_camera_override(bool p_enabled) {
	EditorDebuggerNode::get_singleton()->set_camera_override(p_enabled ? camera_override_mode : EditorDebuggerNode::OVERRIDE_NONE);
}

void GameViewDebugger::set_camera_manipulate_mode(EditorDebuggerNode::CameraOverride p_mode) {
	camera_override_mode = p_mode;

	if (EditorDebuggerNode::get_singleton()->get_camera_override() != EditorDebuggerNode::OVERRIDE_NONE) {
		set_camera_override(true);
	}
}

void GameViewDebugger::reset_camera_2d_position() {
	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:runtime_node_select_reset_camera_2d", Array());
		}
	}
}

void GameViewDebugger::reset_camera_3d_position() {
	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:runtime_node_select_reset_camera_3d", Array());
		}
	}
}

void GameViewDebugger::setup_session(int p_session_id) {
	Ref<EditorDebuggerSession> session = get_session(p_session_id);
	ERR_FAIL_COND(session.is_null());

	sessions.append(session);

	session->connect("started", callable_mp(this, &GameViewDebugger::_session_started).bind(session));
	session->connect("stopped", callable_mp(this, &GameViewDebugger::_session_stopped));
}

void GameViewDebugger::_feature_profile_changed() {
	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();
	is_feature_enabled = profile.is_null() || !profile->is_feature_disabled(EditorFeatureProfile::FEATURE_GAME);
}

void GameViewDebugger::_bind_methods() {
	ADD_SIGNAL(MethodInfo("session_started"));
	ADD_SIGNAL(MethodInfo("session_stopped"));
}

bool GameViewDebugger::add_screenshot_callback(const Callable &p_callaback, const Rect2i &p_rect) {
	bool found = false;
	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			ScreenshotCB sd;
			sd.cb = p_callaback;
			sd.rect = p_rect;
			screenshot_callbacks[scr_rq_id] = sd;

			Array arr;
			arr.append(scr_rq_id);
			I->send_message("scene:rq_screenshot", arr);
			scr_rq_id++;
			found = true;
		}
	}
	return found;
}

bool GameViewDebugger::_msg_get_screenshot(const Array &p_args) {
	ERR_FAIL_COND_V_MSG(p_args.size() != 4, false, "get_screenshot: invalid number of arguments");

	int64_t id = p_args[0];
	int64_t w = p_args[1];
	int64_t h = p_args[2];
	const String &path = p_args[3];

	if (screenshot_callbacks.has(id)) {
		if (screenshot_callbacks[id].cb.is_valid()) {
			screenshot_callbacks[id].cb.call(w, h, path, screenshot_callbacks[id].rect);
		}
		screenshot_callbacks.erase(id);
	}
	return true;
}

bool GameViewDebugger::capture(const String &p_message, const Array &p_data, int p_session) {
	Ref<EditorDebuggerSession> session = get_session(p_session);
	ERR_FAIL_COND_V(session.is_null(), true);

	if (p_message == "game_view:get_screenshot") {
		return _msg_get_screenshot(p_data);
	} else {
		// Any other messages with this prefix should be ignored.
		WARN_PRINT("GameViewDebugger unknown message: " + p_message);
		return false;
	}

	return true;
}

bool GameViewDebugger::has_capture(const String &p_capture) const {
	return p_capture == "game_view";
}

GameViewDebugger::GameViewDebugger() {
	EditorFeatureProfileManager::get_singleton()->connect("current_feature_profile_changed", callable_mp(this, &GameViewDebugger::_feature_profile_changed));

	ED_SHORTCUT("editor/suspend_resume_embedded_project", TTRC("Suspend/Resume Embedded Project"), Key::F9);
	ED_SHORTCUT_OVERRIDE("editor/suspend_resume_embedded_project", "macos", KeyModifierMask::META | KeyModifierMask::SHIFT | Key::B);

	ED_SHORTCUT("editor/next_frame_embedded_project", TTRC("Next Frame"), Key::F10);
}

///////

void GameView::_sessions_changed() {
	// The debugger session's `session_started/stopped` signal can be unreliable, so count it manually.
	active_sessions = 0;
	Array sessions = debugger->get_sessions();
	for (int i = 0; i < sessions.size(); i++) {
		if (Object::cast_to<EditorDebuggerSession>(sessions[i])->is_active()) {
			active_sessions++;
		}
	}

	_update_debugger_buttons();

#ifdef MACOS_ENABLED
	if (!embedded_script_debugger || !embedded_script_debugger->is_session_active() || embedded_script_debugger->get_remote_pid() != embedded_process->get_embedded_pid()) {
		_attach_script_debugger();
	}
#else
	if (embedded_process->is_embedding_completed()) {
		if (!embedded_script_debugger || !embedded_script_debugger->is_session_active() || embedded_script_debugger->get_remote_pid() != embedded_process->get_embedded_pid()) {
			_attach_script_debugger();
		}
	}
#endif
}

void GameView::_instance_starting_static(int p_idx, List<String> &r_arguments) {
	ERR_FAIL_NULL(singleton);
	singleton->_instance_starting(p_idx, r_arguments);
}

void GameView::_instance_starting(int p_idx, List<String> &r_arguments) {
	if (!is_feature_enabled) {
		return;
	}

	if (p_idx == 0 && embed_on_play && make_floating_on_play && window_wrapper->is_window_available() && !window_wrapper->get_window_enabled() && _get_embed_available() == EMBED_AVAILABLE) {
		// Set the Floating Window default title. Always considered in DEBUG mode, same as in Window::set_title.
		String appname = GLOBAL_GET("application/config/name");
		appname = vformat("%s (DEBUG)", TranslationServer::get_singleton()->translate(appname));
		window_wrapper->set_window_title(appname);

		_show_update_window_wrapper();

		if (embedded_process->get_focus_mode_with_override() != FOCUS_NONE) {
			embedded_process->grab_focus();
		}
	}

	_update_arguments_for_instance(p_idx, r_arguments);
}

bool GameView::_instance_rq_screenshot_static(const Callable &p_callback) {
	ERR_FAIL_NULL_V(singleton, false);
	return singleton->_instance_rq_screenshot(p_callback);
}

bool GameView::_instance_rq_screenshot(const Callable &p_callback) {
	if (debugger.is_null() || window_wrapper->get_window_enabled() || !embedded_process || !embedded_process->is_embedding_completed()) {
		return false;
	}
	Rect2 r = embedded_process->get_adjusted_embedded_window_rect(embedded_process->get_rect());
	r.position += embedded_process->get_global_position();
#ifndef MACOS_ENABLED
	r.position -= embedded_process->get_window()->get_position();
#endif
	return debugger->add_screenshot_callback(p_callback, r);
}

void GameView::_show_update_window_wrapper() {
	EditorRun::WindowPlacement placement = EditorRun::get_window_placement();
	Point2 position = floating_window_rect.position;
	Size2i size = floating_window_rect.size;
	int screen = floating_window_screen;

	// Obtain the size around the embedded process control. Usually, the difference between the game view's get_size
	// and the embedded control should work. However, when the control is hidden and has never been displayed,
	// the size of the embedded control is not calculated.
	Size2 old_min_size = embedded_process->get_custom_minimum_size();
	embedded_process->set_custom_minimum_size(Size2i());

	Size2 embedded_process_min_size = get_minimum_size();
	Size2 wrapped_margins_size = window_wrapper->get_margins_size();
	Size2 wrapped_min_size = window_wrapper->get_minimum_size();
	Point2 offset_embedded_process = embedded_process->get_global_position() - get_global_position();

	// On the first startup, the global position of the embedded process control is invalid because it was
	// never displayed. We will calculate it manually using the minimum size of the window.
	if (offset_embedded_process == Point2()) {
		offset_embedded_process.y = wrapped_min_size.y;
	}
	offset_embedded_process.x += embedded_process->get_margin_size(SIDE_LEFT);
	offset_embedded_process.y += embedded_process->get_margin_size(SIDE_TOP);
	offset_embedded_process += window_wrapper->get_margins_top_left();

	embedded_process->set_custom_minimum_size(old_min_size);

	Point2 size_diff_embedded_process = Point2(0, embedded_process_min_size.y) + embedded_process->get_margins_size();

	if (placement.position != Point2i(INT_MAX, INT_MAX)) {
		position = placement.position - offset_embedded_process;
		screen = placement.screen;
	}
	if (placement.size != Size2i()) {
		size = placement.size + size_diff_embedded_process + wrapped_margins_size;
	}
	window_wrapper->restore_window_from_saved_position(Rect2(position, size), screen, Rect2i());
}

void GameView::_play_pressed() {
	if (!is_feature_enabled) {
		return;
	}

	OS::ProcessID current_process_id = EditorRunBar::get_singleton()->get_current_process();
	if (current_process_id == 0) {
		return;
	}

	if (!window_wrapper->get_window_enabled()) {
		screen_index_before_start = EditorNode::get_singleton()->get_editor_main_screen()->get_selected_index();
	}

	if (embed_on_play && _get_embed_available() == EMBED_AVAILABLE) {
		// It's important to disable the low power mode when unfocused because otherwise
		// the button in the editor are not responsive and if the user moves the mouse quickly,
		// the mouse clicks are not registered.
		EditorNode::get_singleton()->set_unfocused_low_processor_usage_mode_enabled(false);
		_update_embed_window_size();
		if (!window_wrapper->get_window_enabled()) {
			EditorNode::get_singleton()->get_editor_main_screen()->select(EditorMainScreen::EDITOR_GAME);
			// Reset the normal size of the bottom panel when fully expanded.
			EditorNode::get_singleton()->get_bottom_panel()->set_expanded(false);

			if (embedded_process->get_focus_mode_with_override() != FOCUS_NONE) {
				embedded_process->grab_focus();
			}
		}
		embedded_process->embed_process(current_process_id);
		_update_ui();
	}
}

void GameView::_stop_pressed() {
	if (!is_feature_enabled) {
		return;
	}

	_detach_script_debugger();
	paused = false;

	EditorNode::get_singleton()->set_unfocused_low_processor_usage_mode_enabled(true);
	embedded_process->reset();
	_update_ui();

	if (window_wrapper->get_window_enabled()) {
		window_wrapper->set_window_enabled(false);
	}

	if (screen_index_before_start >= 0 && EditorNode::get_singleton()->get_editor_main_screen()->get_selected_index() == EditorMainScreen::EDITOR_GAME) {
		// We go back to the screen where the user was before starting the game.
		EditorNode::get_singleton()->get_editor_main_screen()->select(screen_index_before_start);
	}

	screen_index_before_start = -1;
}

void GameView::_embedding_completed() {
#ifndef MACOS_ENABLED
	_attach_script_debugger();
#endif
	_update_ui();
	if (make_floating_on_play) {
		get_window()->set_flag(Window::FLAG_ALWAYS_ON_TOP, bool(GLOBAL_GET("display/window/size/always_on_top")));
	}
}

void GameView::_embedding_failed() {
	state_label->set_text(TTRC("Connection impossible to the game process."));
}

void GameView::_embedded_process_updated() {
	const Rect2i game_rect = embedded_process->get_screen_embedded_window_rect();
	game_size_label->set_text(vformat("%dx%d", game_rect.size.x, game_rect.size.y));
}

void GameView::_embedded_process_focused() {
	if (embed_on_play && !window_wrapper->get_window_enabled()) {
		EditorNode::get_singleton()->get_editor_main_screen()->select(EditorMainScreen::EDITOR_GAME);
	}
}

void GameView::_editor_or_project_settings_changed() {
	if (!is_inside_tree()) {
		return;
	}

	// Update the window size and aspect ratio.
	_update_embed_window_size();

	if (window_wrapper->get_window_enabled()) {
		_show_update_window_wrapper();
		if (embedded_process->is_embedding_completed()) {
			embedded_process->queue_update_embedded_process();
		}
	}

	_update_ui();
}

void GameView::_update_debugger_buttons() {
	bool empty = active_sessions == 0;

	suspend_button->set_disabled(empty);
	camera_override_button->set_disabled(empty);
	speed_state_button->set_disabled(empty);
	reset_speed_button->set_disabled(empty);

	PopupMenu *menu = camera_override_menu->get_popup();

	bool disable_camera_reset = empty || !camera_override_button->is_pressed() || !menu->is_item_checked(menu->get_item_index(CAMERA_MODE_INGAME));
	menu->set_item_disabled(CAMERA_RESET_2D, disable_camera_reset);
	menu->set_item_disabled(CAMERA_RESET_3D, disable_camera_reset);

	if (empty) {
		suspend_button->set_pressed(false);
		camera_override_button->set_pressed(false);
	}
	next_frame_button->set_disabled(!suspend_button->is_pressed());

	_reset_time_scales();
}

void GameView::_handle_shortcut_requested(int p_embed_action) {
	switch (p_embed_action) {
		case ScriptEditorDebugger::EMBED_SUSPEND_TOGGLE: {
			_toggle_suspend_button();
		} break;
		case ScriptEditorDebugger::EMBED_NEXT_FRAME: {
			debugger->next_frame();
		} break;
	}
}

void GameView::_toggle_suspend_button() {
	const bool new_pressed = !suspend_button->is_pressed();
	suspend_button->set_pressed(new_pressed);
	_suspend_button_toggled(new_pressed);
}

void GameView::_suspend_button_toggled(bool p_pressed) {
	_update_debugger_buttons();

	debugger->set_suspend(p_pressed);
}

void GameView::_node_type_pressed(int p_option) {
	RuntimeNodeSelect::NodeType type = (RuntimeNodeSelect::NodeType)p_option;
	for (int i = 0; i < RuntimeNodeSelect::NODE_TYPE_MAX; i++) {
		node_type_button[i]->set_pressed_no_signal(i == type);
	}

	_update_debugger_buttons();

	debugger->set_node_type(type);
}

void GameView::_select_mode_pressed(int p_option) {
	RuntimeNodeSelect::SelectMode mode = (RuntimeNodeSelect::SelectMode)p_option;
	if (!select_mode_button[mode]->is_visible()) {
		return;
	}

	for (int i = 0; i < RuntimeNodeSelect::SELECT_MODE_MAX; i++) {
		select_mode_button[i]->set_pressed_no_signal(i == mode);
	}

	debugger->set_select_mode(mode);

	EditorSettings::get_singleton()->set_project_metadata("game_view", "select_mode", mode);
}

void GameView::_selection_options_menu_id_pressed(int p_id) {
	switch (p_id) {
		case SELECTION_AVOID_LOCKED: {
			selection_avoid_locked = !selection_avoid_locked;
			debugger->set_selection_avoid_locked(selection_avoid_locked);
			EditorSettings::get_singleton()->set_project_metadata("game_view", "selection_avoid_locked", selection_avoid_locked);
		} break;
		case SELECTION_PREFER_GROUP: {
			selection_prefer_group = !selection_prefer_group;
			debugger->set_selection_prefer_group(selection_prefer_group);
			EditorSettings::get_singleton()->set_project_metadata("game_view", "selection_prefer_group", selection_prefer_group);
		} break;
	}

	PopupMenu *menu = selection_options_menu->get_popup();
	menu->set_item_checked(menu->get_item_index(SELECTION_AVOID_LOCKED), selection_avoid_locked);
	menu->set_item_checked(menu->get_item_index(SELECTION_PREFER_GROUP), selection_prefer_group);
}

void GameView::_embed_options_menu_menu_id_pressed(int p_id) {
	switch (p_id) {
		case EMBED_RUN_GAME_EMBEDDED: {
			embed_on_play = !embed_on_play;
			int game_mode = EDITOR_GET("run/window_placement/game_embed_mode");
			if (game_mode == 0) { // Save only if not overridden by editor.
				EditorSettings::get_singleton()->set_project_metadata("game_view", "embed_on_play", embed_on_play);
			}
		} break;
		case EMBED_MAKE_FLOATING_ON_PLAY: {
			make_floating_on_play = !make_floating_on_play;
			int game_mode = EDITOR_GET("run/window_placement/game_embed_mode");
			if (game_mode == 0) { // Save only if not overridden by editor.
				EditorSettings::get_singleton()->set_project_metadata("game_view", "make_floating_on_play", make_floating_on_play);
			}
		} break;
		case SIZE_MODE_FIXED:
		case SIZE_MODE_KEEP_ASPECT:
		case SIZE_MODE_STRETCH: {
			embed_size_mode = (EmbedSizeMode)p_id;
			EditorSettings::get_singleton()->set_project_metadata("game_view", "embed_size_mode", p_id);

			_update_embed_window_size();
		} break;
	}
	_update_embed_menu_options();
	_update_ui();
}

void GameView::_reset_time_scales() {
	if (!is_visible_in_tree()) {
		return;
	}
	time_scale_index = DEFAULT_TIME_SCALE_INDEX;
	debugger->reset_time_scale();
	_update_speed_buttons();
}

void GameView::_speed_state_menu_pressed(int p_id) {
	time_scale_index = p_id;
	debugger->set_time_scale(time_scale_range[time_scale_index]);
	_update_speed_buttons();
}

void GameView::_update_speed_buttons() {
	bool disabled = time_scale_index == DEFAULT_TIME_SCALE_INDEX;
	reset_speed_button->set_disabled(disabled);
	speed_state_button->set_text(vformat(U"%s×", time_scale_label[time_scale_index]));
	_update_speed_state_color();
}

void GameView::_update_speed_state_color() {
	Color text_color;
	if (time_scale_index == DEFAULT_TIME_SCALE_INDEX) {
		text_color = get_theme_color(SceneStringName(font_color), EditorStringName(Editor));
	} else if (time_scale_index > DEFAULT_TIME_SCALE_INDEX) {
		text_color = get_theme_color(SNAME("success_color"), EditorStringName(Editor));
	} else if (time_scale_index < DEFAULT_TIME_SCALE_INDEX) {
		text_color = get_theme_color(SNAME("warning_color"), EditorStringName(Editor));
	}
	speed_state_button->add_theme_color_override(SceneStringName(font_color), text_color);
}

void GameView::_update_speed_state_size() {
	if (!speed_state_button) {
		return;
	}
	float min_size = 0;
	for (const String lbl : time_scale_label) {
		min_size = MAX(speed_state_button->get_minimum_size_for_text_and_icon(vformat(U"%s×", lbl), Ref<Texture2D>()).x, min_size);
	}
	speed_state_button->set_custom_minimum_size(Vector2(min_size, 0));
}

GameView::EmbedAvailability GameView::_get_embed_available() {
	if (!DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_WINDOW_EMBEDDING)) {
		return EMBED_NOT_AVAILABLE_FEATURE_NOT_SUPPORTED;
	}
	if (get_tree()->get_root()->is_embedding_subwindows()) {
		return EMBED_NOT_AVAILABLE_SINGLE_WINDOW_MODE;
	}
	String display_driver = GLOBAL_GET("display/display_server/driver");
	if (display_driver == "headless") {
		return EMBED_NOT_AVAILABLE_PROJECT_DISPLAY_DRIVER;
	}

	if (RunInstancesDialog::get_singleton()) {
		List<String> instance_args;
		RunInstancesDialog::get_singleton()->get_argument_list_for_instance(0, instance_args);
		if (instance_args.find("--headless")) {
			return EMBED_NOT_AVAILABLE_HEADLESS;
		}
	}

	EditorRun::WindowPlacement placement = EditorRun::get_window_placement();
	if (placement.force_fullscreen) {
		return EMBED_NOT_AVAILABLE_FULLSCREEN;
	}
	if (placement.force_maximized) {
		return EMBED_NOT_AVAILABLE_MAXIMIZED;
	}

	DisplayServer::WindowMode window_mode = (DisplayServer::WindowMode)(GLOBAL_GET("display/window/size/mode").operator int());
	if (window_mode == DisplayServer::WindowMode::WINDOW_MODE_MINIMIZED) {
		return EMBED_NOT_AVAILABLE_MINIMIZED;
	}
	if (window_mode == DisplayServer::WindowMode::WINDOW_MODE_MAXIMIZED) {
		return EMBED_NOT_AVAILABLE_MAXIMIZED;
	}
	if (window_mode == DisplayServer::WindowMode::WINDOW_MODE_FULLSCREEN || window_mode == DisplayServer::WindowMode::WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
		return EMBED_NOT_AVAILABLE_FULLSCREEN;
	}

	return EMBED_AVAILABLE;
}

void GameView::_update_ui() {
	bool show_game_size = false;
	EmbedAvailability available = _get_embed_available();

	switch (available) {
		case EMBED_AVAILABLE:
			if (embedded_process->is_embedding_completed()) {
				state_label->set_text("");
				show_game_size = true;
			} else if (embedded_process->is_embedding_in_progress()) {
				state_label->set_text(TTRC("Game starting..."));
			} else if (EditorRunBar::get_singleton()->is_playing()) {
				state_label->set_text(TTRC("Game running not embedded."));
			} else if (embed_on_play) {
				state_label->set_text(TTRC("Press play to start the game."));
			} else {
				state_label->set_text(TTRC("Embedding is disabled."));
			}
			break;
		case EMBED_NOT_AVAILABLE_FEATURE_NOT_SUPPORTED:
			state_label->set_text(TTRC("Game embedding not available on your OS."));
			break;
		case EMBED_NOT_AVAILABLE_PROJECT_DISPLAY_DRIVER:
			state_label->set_text(vformat(TTR("Game embedding not available for the Display Server: '%s'.\nDisplay Server can be modified in the Project Settings (Display > Display Server > Driver)."), GLOBAL_GET("display/display_server/driver")));
			break;
		case EMBED_NOT_AVAILABLE_MINIMIZED:
			state_label->set_text(TTR("Game embedding not available when the game starts minimized.") + "\n" + TTR("Consider overriding the window mode project setting with the editor feature tag to Windowed to use game embedding while leaving the exported project intact."));
			break;
		case EMBED_NOT_AVAILABLE_MAXIMIZED:
			state_label->set_text(TTR("Game embedding not available when the game starts maximized.") + "\n" + TTR("Consider overriding the window mode project setting with the editor feature tag to Windowed to use game embedding while leaving the exported project intact."));
			break;
		case EMBED_NOT_AVAILABLE_FULLSCREEN:
			state_label->set_text(TTR("Game embedding not available when the game starts in fullscreen.") + "\n" + TTR("Consider overriding the window mode project setting with the editor feature tag to Windowed to use game embedding while leaving the exported project intact."));
			break;
		case EMBED_NOT_AVAILABLE_SINGLE_WINDOW_MODE:
			state_label->set_text(TTRC("Game embedding not available in single window mode."));
			break;
		case EMBED_NOT_AVAILABLE_HEADLESS:
			state_label->set_text(TTRC("Game embedding not available when the game starts in headless mode."));
			break;
	}

	if (available == EMBED_AVAILABLE) {
		if (state_label->has_theme_color_override(SceneStringName(font_color))) {
			state_label->remove_theme_color_override(SceneStringName(font_color));
		}
	} else {
		state_label->add_theme_color_override(SceneStringName(font_color), state_label->get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
	}

	game_size_label->set_visible(show_game_size);
}

void GameView::_update_embed_menu_options() {
	bool is_multi_window = window_wrapper->is_window_available();
	PopupMenu *menu = embed_options_menu->get_popup();
	menu->set_item_checked(menu->get_item_index(EMBED_RUN_GAME_EMBEDDED), embed_on_play);
	menu->set_item_checked(menu->get_item_index(EMBED_MAKE_FLOATING_ON_PLAY), make_floating_on_play && is_multi_window);

	menu->set_item_checked(menu->get_item_index(SIZE_MODE_FIXED), embed_size_mode == SIZE_MODE_FIXED);
	menu->set_item_checked(menu->get_item_index(SIZE_MODE_KEEP_ASPECT), embed_size_mode == SIZE_MODE_KEEP_ASPECT);
	menu->set_item_checked(menu->get_item_index(SIZE_MODE_STRETCH), embed_size_mode == SIZE_MODE_STRETCH);

	menu->set_item_disabled(menu->get_item_index(EMBED_MAKE_FLOATING_ON_PLAY), !embed_on_play || !is_multi_window);
}

void GameView::_update_embed_window_size() {
	if (paused) {
		// When paused, Godot does not re-render. As a result, resizing the game window to a larger size
		// causes artifacts and flickering. However, resizing to a smaller size seems fine.
		// To prevent artifacts and flickering, we will force the game window to maintain its size.
		// Using the same technique as SIZE_MODE_FIXED, the embedded process control will
		// prevent resizing the game to a larger size while maintaining the aspect ratio.
		embedded_process->set_window_size(size_paused);
		embedded_process->set_keep_aspect(false);

	} else {
		if (embed_size_mode == SIZE_MODE_FIXED || embed_size_mode == SIZE_MODE_KEEP_ASPECT) {
			// The embedded process control will need the desired window size.
			EditorRun::WindowPlacement placement = EditorRun::get_window_placement();
			embedded_process->set_window_size(placement.size);
		} else {
			// Stretch... No need for the window size.
			embedded_process->set_window_size(Size2i());
		}
		embedded_process->set_keep_aspect(embed_size_mode == SIZE_MODE_KEEP_ASPECT);
	}
}

void GameView::_hide_selection_toggled(bool p_pressed) {
	hide_selection->set_button_icon(get_editor_theme_icon(p_pressed ? SNAME("GuiVisibilityHidden") : SNAME("GuiVisibilityVisible")));

	debugger->set_selection_visible(!p_pressed);

	EditorSettings::get_singleton()->set_project_metadata("game_view", "hide_selection", p_pressed);
}

void GameView::_debug_mute_audio_button_pressed() {
	debug_mute_audio = !debug_mute_audio;
	debug_mute_audio_button->set_button_icon(get_editor_theme_icon(debug_mute_audio ? SNAME("AudioMute") : SNAME("AudioStreamPlayer")));
	debug_mute_audio_button->set_tooltip_text(debug_mute_audio ? TTRC("Unmute game audio.") : TTRC("Mute game audio."));
	debugger->set_debug_mute_audio(debug_mute_audio);
}

void GameView::_camera_override_button_toggled(bool p_pressed) {
	_update_debugger_buttons();

	debugger->set_camera_override(p_pressed);
}

void GameView::_camera_override_menu_id_pressed(int p_id) {
	PopupMenu *menu = camera_override_menu->get_popup();
	if (p_id != CAMERA_RESET_2D && p_id != CAMERA_RESET_3D) {
		for (int i = 0; i < menu->get_item_count(); i++) {
			menu->set_item_checked(i, false);
		}
	}

	switch (p_id) {
		case CAMERA_RESET_2D: {
			debugger->reset_camera_2d_position();
		} break;
		case CAMERA_RESET_3D: {
			debugger->reset_camera_3d_position();
		} break;
		case CAMERA_MODE_INGAME: {
			debugger->set_camera_manipulate_mode(EditorDebuggerNode::OVERRIDE_INGAME);
			menu->set_item_checked(menu->get_item_index(p_id), true);

			_update_debugger_buttons();

			EditorSettings::get_singleton()->set_project_metadata("game_view", "camera_override_mode", p_id);
		} break;
		case CAMERA_MODE_EDITORS: {
			debugger->set_camera_manipulate_mode(EditorDebuggerNode::OVERRIDE_EDITORS);
			menu->set_item_checked(menu->get_item_index(p_id), true);

			_update_debugger_buttons();

			EditorSettings::get_singleton()->set_project_metadata("game_view", "camera_override_mode", p_id);
		} break;
	}
}

void GameView::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_tooltip_text(vformat(TTR("%s+Alt+RMB: Show list of all nodes at position clicked."), keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL)));
			_update_ui();
		} break;

		case NOTIFICATION_POST_ENTER_TREE: {
			_update_speed_state_size();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			suspend_button->set_button_icon(get_editor_theme_icon(SNAME("Suspend")));
			next_frame_button->set_button_icon(get_editor_theme_icon(SNAME("NextFrame")));
			reset_speed_button->set_button_icon(get_editor_theme_icon(SNAME("Reload")));

			node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_button_icon(get_editor_theme_icon(SNAME("InputEventJoypadMotion")));
			node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_button_icon(get_editor_theme_icon(SNAME("2DNodes")));
			node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_button_icon(get_editor_theme_icon(SNAME("Node3D")));

			select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_button_icon(get_editor_theme_icon(SNAME("ToolSelect")));
			select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_button_icon(get_editor_theme_icon(SNAME("ListSelect")));

			hide_selection->set_button_icon(get_editor_theme_icon(hide_selection->is_pressed() ? SNAME("GuiVisibilityHidden") : SNAME("GuiVisibilityVisible")));
			selection_options_menu->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));
			embed_options_menu->set_button_icon(get_editor_theme_icon(SNAME("KeepAspect")));

			debug_mute_audio_button->set_button_icon(get_editor_theme_icon(debug_mute_audio ? SNAME("AudioMute") : SNAME("AudioStreamPlayer")));

			camera_override_button->set_button_icon(get_editor_theme_icon(SNAME("Camera")));
			camera_override_menu->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));

			_update_speed_state_size();
			_update_speed_state_color();
		} break;

		case NOTIFICATION_READY: {
			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_WINDOW_EMBEDDING)) {
				// Embedding available.
				int game_mode = EDITOR_GET("run/window_placement/game_embed_mode");
				switch (game_mode) {
					case -1: { // Disabled.
						embed_on_play = false;
						make_floating_on_play = false;
					} break;
					case 1: { // Embed.
						embed_on_play = true;
						make_floating_on_play = false;
					} break;
					case 2: { // Floating.
						embed_on_play = true;
						make_floating_on_play = true;
					} break;
					default: {
						embed_on_play = EditorSettings::get_singleton()->get_project_metadata("game_view", "embed_on_play", true);
						make_floating_on_play = EditorSettings::get_singleton()->get_project_metadata("game_view", "make_floating_on_play", true);
					} break;
				}
				embed_size_mode = (EmbedSizeMode)(int)EditorSettings::get_singleton()->get_project_metadata("game_view", "embed_size_mode", SIZE_MODE_FIXED);
				_update_embed_menu_options();

				EditorRunBar::get_singleton()->connect("play_pressed", callable_mp(this, &GameView::_play_pressed));
				EditorRunBar::get_singleton()->connect("stop_pressed", callable_mp(this, &GameView::_stop_pressed));
				EditorRun::instance_starting_callback = _instance_starting_static;
				EditorRun::instance_rq_screenshot_callback = _instance_rq_screenshot_static;

				// Listen for project settings changes to update the window size and aspect ratio.
				ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &GameView::_editor_or_project_settings_changed));
				EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &GameView::_editor_or_project_settings_changed));
			} else {
				// Embedding not available.
				embedding_hb->hide();
			}

			_update_ui();
		} break;
		case NOTIFICATION_WM_POSITION_CHANGED: {
			if (window_wrapper->get_window_enabled()) {
				_update_floating_window_settings();
			}
		} break;
	}
}

void GameView::set_window_layout(Ref<ConfigFile> p_layout) {
	floating_window_rect = p_layout->get_value("GameView", "floating_window_rect", Rect2i());
	floating_window_screen = p_layout->get_value("GameView", "floating_window_screen", -1);
}

void GameView::get_window_layout(Ref<ConfigFile> p_layout) {
	if (window_wrapper->get_window_enabled()) {
		_update_floating_window_settings();
	}

	p_layout->set_value("GameView", "floating_window_rect", floating_window_rect);
	p_layout->set_value("GameView", "floating_window_screen", floating_window_screen);
}

void GameView::_update_floating_window_settings() {
	if (window_wrapper->get_window_enabled()) {
		floating_window_rect = window_wrapper->get_window_rect();
		floating_window_screen = window_wrapper->get_window_screen();
	}
}

void GameView::_attach_script_debugger() {
	_detach_script_debugger();

	int i = 0;
	while (ScriptEditorDebugger *script_debugger = EditorDebuggerNode::get_singleton()->get_debugger(i)) {
		if (script_debugger->is_session_active() && script_debugger->get_remote_pid() == embedded_process->get_embedded_pid()) {
			embedded_script_debugger = script_debugger;
			break;
		}
		i++;
	}

#ifdef MACOS_ENABLED
	embedded_process->set_script_debugger(embedded_script_debugger);
#endif

	if (embedded_script_debugger) {
		embedded_script_debugger->connect("remote_window_title_changed", callable_mp(this, &GameView::_remote_window_title_changed));
		embedded_script_debugger->connect("embed_shortcut_requested", callable_mp(this, &GameView::_handle_shortcut_requested));
	}
}

void GameView::_detach_script_debugger() {
	if (embedded_script_debugger) {
		embedded_script_debugger->disconnect("remote_window_title_changed", callable_mp(this, &GameView::_remote_window_title_changed));
		embedded_script_debugger->disconnect("embed_shortcut_requested", callable_mp(this, &GameView::_handle_shortcut_requested));
		embedded_script_debugger = nullptr;
	}
	embedded_process->set_script_debugger(nullptr);
}

void GameView::_remote_window_title_changed(String title) {
	window_wrapper->set_window_title(title);
}

void GameView::_update_arguments_for_instance(int p_idx, List<String> &r_arguments) {
	if (p_idx != 0 || !embed_on_play || _get_embed_available() != EMBED_AVAILABLE) {
		return;
	}

	// Remove duplicates/unwanted parameters.
	List<String>::Element *E = r_arguments.front();
	List<String>::Element *user_args_element = nullptr;
	HashSet<String> remove_args({ "--position", "--resolution", "--screen" });
#ifdef MACOS_ENABLED
	// macOS requires the embedded display driver.
	remove_args.insert("--display-driver");
#endif

#ifdef WAYLAND_ENABLED
	// Wayland requires its display driver.
	if (DisplayServer::get_singleton()->get_name() == "Wayland") {
		remove_args.insert("--display-driver");
	}
#endif

#ifdef X11_ENABLED
	// X11 requires its display driver.
	if (DisplayServer::get_singleton()->get_name() == "X11") {
		remove_args.insert("--display-driver");
	}
#endif

	while (E) {
		List<String>::Element *N = E->next();

		// For these parameters, we need to also remove the value.
		if (remove_args.has(E->get())) {
			r_arguments.erase(E);
			if (N) {
				List<String>::Element *V = N->next();
				r_arguments.erase(N);
				N = V;
			}
		} else if (E->get() == "-f" || E->get() == "--fullscreen" || E->get() == "-m" || E->get() == "--maximized" || E->get() == "-t" || E->get() == "-always-on-top") {
			r_arguments.erase(E);
		} else if (E->get() == "--" || E->get() == "++") {
			user_args_element = E;
			break;
		}

		E = N;
	}

	// Add the editor window's native ID so the started game can directly set it as its parent.
	List<String>::Element *N = r_arguments.insert_before(user_args_element, "--wid");
	N = r_arguments.insert_after(N, itos(DisplayServer::get_singleton()->window_get_native_handle(DisplayServer::WINDOW_HANDLE, get_window()->get_window_id())));

#if MACOS_ENABLED
	N = r_arguments.insert_after(N, "--embedded");
#endif

#ifdef WAYLAND_ENABLED
	if (DisplayServer::get_singleton()->get_name() == "Wayland") {
		N = r_arguments.insert_after(N, "--display-driver");
		N = r_arguments.insert_after(N, "wayland");
	}
#endif

#ifdef X11_ENABLED
	if (DisplayServer::get_singleton()->get_name() == "X11") {
		N = r_arguments.insert_after(N, "--display-driver");
		N = r_arguments.insert_after(N, "x11");
	}
#endif

	// Be sure to have the correct window size in the embedded_process control.
	_update_embed_window_size();
	Rect2i rect = embedded_process->get_screen_embedded_window_rect();

	// Usually, the global rect of the embedded process control is invalid because it was hidden. We will calculate it manually.
	if (!window_wrapper->get_window_enabled()) {
		Size2 old_min_size = embedded_process->get_custom_minimum_size();
		embedded_process->set_custom_minimum_size(Size2i());

		Control *container = EditorNode::get_singleton()->get_editor_main_screen()->get_control();
		rect = container->get_global_rect();

		Size2 wrapped_min_size = window_wrapper->get_minimum_size();
		rect.position.y += wrapped_min_size.y;
		rect.size.y -= wrapped_min_size.y;

		rect = embedded_process->get_adjusted_embedded_window_rect(rect);

		embedded_process->set_custom_minimum_size(old_min_size);
	}

	// When using the floating window, we need to force the position and size from the
	// editor/project settings, because the get_screen_embedded_window_rect of the
	// embedded_process will be updated only on the next frame.
	if (window_wrapper->get_window_enabled()) {
		EditorRun::WindowPlacement placement = EditorRun::get_window_placement();
		if (placement.position != Point2i(INT_MAX, INT_MAX)) {
			rect.position = placement.position;
		}
		if (placement.size != Size2i()) {
			rect.size = placement.size;
		}
	}

	N = r_arguments.insert_after(N, "--position");
	N = r_arguments.insert_after(N, itos(rect.position.x) + "," + itos(rect.position.y));
	N = r_arguments.insert_after(N, "--resolution");
	r_arguments.insert_after(N, itos(rect.size.x) + "x" + itos(rect.size.y));
}

void GameView::_window_close_request() {
	if (window_wrapper->get_window_enabled()) {
		window_wrapper->set_window_enabled(false);
	}

	// Before the parent window closed, we close the embedded game. That prevents
	// the embedded game to be seen without a parent window for a fraction of second.
	if (EditorRunBar::get_singleton()->is_playing() && (embedded_process->is_embedding_completed() || embedded_process->is_embedding_in_progress())) {
		// When the embedding is not complete, we need to kill the process.
		// If the game is paused, the close request will not be processed by the game, so it's better to kill the process.
		if (paused || embedded_process->is_embedding_in_progress()) {
			embedded_process->reset();
			// Call deferred to prevent the _stop_pressed callback to be executed before the wrapper window
			// actually closes.
			callable_mp(EditorRunBar::get_singleton(), &EditorRunBar::stop_playing).call_deferred();
		} else {
			// Try to gracefully close the window. That way, the NOTIFICATION_WM_CLOSE_REQUEST
			// notification should be propagated in the game process.
			embedded_process->request_close();
		}
	}
}

void GameView::_debugger_breaked(bool p_breaked, bool p_can_debug) {
	if (p_breaked == paused) {
		return;
	}

	paused = p_breaked;

	if (paused) {
		size_paused = embedded_process->get_screen_embedded_window_rect().size;
	}

	_update_embed_window_size();
}

void GameView::_feature_profile_changed() {
	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();
	bool is_profile_null = profile.is_null();

	is_feature_enabled = is_profile_null || !profile->is_feature_disabled(EditorFeatureProfile::FEATURE_GAME);

	bool is_3d_enabled = is_profile_null || !profile->is_feature_disabled(EditorFeatureProfile::FEATURE_3D);
	if (!is_3d_enabled && node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->is_pressed()) {
		_node_type_pressed(RuntimeNodeSelect::NODE_TYPE_NONE);
	}
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_visible(is_3d_enabled);
}

GameView::GameView(Ref<GameViewDebugger> p_debugger, EmbeddedProcessBase *p_embedded_process, WindowWrapper *p_wrapper) {
	singleton = this;

	debugger = p_debugger;
	window_wrapper = p_wrapper;
	embedded_process = p_embedded_process;

	MarginContainer *toolbar_margin = memnew(MarginContainer);
	toolbar_margin->set_theme_type_variation("MainToolBarMargin");
	add_child(toolbar_margin);

	FlowContainer *main_menu_fc = memnew(FlowContainer);
	toolbar_margin->add_child(main_menu_fc);

	HBoxContainer *process_hb = memnew(HBoxContainer);
	main_menu_fc->add_child(process_hb);
	suspend_button = memnew(Button);
	process_hb->add_child(suspend_button);
	suspend_button->set_toggle_mode(true);
	suspend_button->set_theme_type_variation(SceneStringName(FlatButton));
	suspend_button->connect(SceneStringName(toggled), callable_mp(this, &GameView::_suspend_button_toggled));
	suspend_button->set_accessibility_name(TTRC("Suspend"));
	suspend_button->set_shortcut(ED_GET_SHORTCUT("editor/suspend_resume_embedded_project"));
	suspend_button->set_tooltip_text(TTRC("Force pause at SceneTree level. Stops all processing, but you can still interact with the project."));

	next_frame_button = memnew(Button);
	process_hb->add_child(next_frame_button);
	next_frame_button->set_theme_type_variation(SceneStringName(FlatButton));
	next_frame_button->connect(SceneStringName(pressed), callable_mp(*debugger, &GameViewDebugger::next_frame));
	next_frame_button->set_accessibility_name(TTRC("Next Frame"));
	next_frame_button->set_shortcut(ED_GET_SHORTCUT("editor/next_frame_embedded_project"));

	speed_state_button = memnew(MenuButton);
	process_hb->add_child(speed_state_button);
	speed_state_button->set_text(U"1.0×");
	speed_state_button->set_flat(false);
	speed_state_button->set_theme_type_variation("FlatMenuButton");
	speed_state_button->set_tooltip_text(TTRC("Change the game speed."));
	speed_state_button->set_accessibility_name(TTRC("Speed State"));

	PopupMenu *menu = speed_state_button->get_popup();
	menu->connect(SceneStringName(id_pressed), callable_mp(this, &GameView::_speed_state_menu_pressed));
	for (String lbl : time_scale_label) {
		menu->add_item(vformat(U"%s×", lbl));
	}

	reset_speed_button = memnew(Button);
	process_hb->add_child(reset_speed_button);
	reset_speed_button->set_theme_type_variation(SceneStringName(FlatButton));
	reset_speed_button->set_tooltip_text(TTRC("Reset the game speed."));
	reset_speed_button->set_accessibility_name(TTRC("Reset Speed"));
	reset_speed_button->connect(SceneStringName(pressed), callable_mp(this, &GameView::_reset_time_scales));

	process_hb->add_child(memnew(VSeparator));

	HBoxContainer *input_hb = memnew(HBoxContainer);
	main_menu_fc->add_child(input_hb);

	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE] = memnew(Button);
	input_hb->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_text(TTRC("Input"));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_pressed(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_theme_type_variation(SceneStringName(FlatButton));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->connect(SceneStringName(pressed), callable_mp(this, &GameView::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_NONE));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_tooltip_text(TTRC("Allow game input."));

	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D] = memnew(Button);
	input_hb->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_text(TTRC("2D"));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_theme_type_variation(SceneStringName(FlatButton));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->connect(SceneStringName(pressed), callable_mp(this, &GameView::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_2D));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_tooltip_text(TTRC("Disable game input and allow to select Node2Ds, Controls, and manipulate the 2D camera."));

	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D] = memnew(Button);
	input_hb->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_text(TTRC("3D"));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_theme_type_variation(SceneStringName(FlatButton));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->connect(SceneStringName(pressed), callable_mp(this, &GameView::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_3D));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_tooltip_text(TTRC("Disable game input and allow to select Node3Ds and manipulate the 3D camera."));

	input_hb->add_child(memnew(VSeparator));

	HBoxContainer *selection_hb = memnew(HBoxContainer);
	main_menu_fc->add_child(selection_hb);

	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE] = memnew(Button);
	selection_hb->add_child(select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_toggle_mode(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_pressed(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_theme_type_variation(SceneStringName(FlatButton));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->connect(SceneStringName(pressed), callable_mp(this, &GameView::_select_mode_pressed).bind(RuntimeNodeSelect::SELECT_MODE_SINGLE));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_shortcut(ED_GET_SHORTCUT("spatial_editor/tool_select"));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_shortcut_context(this);

	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST] = memnew(Button);
	selection_hb->add_child(select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_toggle_mode(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_theme_type_variation(SceneStringName(FlatButton));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->connect(SceneStringName(pressed), callable_mp(this, &GameView::_select_mode_pressed).bind(RuntimeNodeSelect::SELECT_MODE_LIST));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_tooltip_text(TTRC("Show list of selectable nodes at position clicked."));

	_select_mode_pressed(EditorSettings::get_singleton()->get_project_metadata("game_view", "select_mode", 0));

	hide_selection = memnew(Button);
	selection_hb->add_child(hide_selection);
	hide_selection->set_toggle_mode(true);
	hide_selection->set_theme_type_variation(SceneStringName(FlatButton));
	hide_selection->set_tooltip_text(TTRC("Toggle Selection Visibility"));
	hide_selection->set_pressed(EditorSettings::get_singleton()->get_project_metadata("game_view", "hide_selection", false));
	if (hide_selection->is_pressed()) {
		debugger->set_selection_visible(false);
	}
	hide_selection->connect(SceneStringName(toggled), callable_mp(this, &GameView::_hide_selection_toggled));

	selection_options_menu = memnew(MenuButton);
	selection_hb->add_child(selection_options_menu);
	selection_options_menu->set_flat(false);
	selection_options_menu->set_theme_type_variation("FlatMenuButton");
	selection_options_menu->set_h_size_flags(SIZE_SHRINK_END);
	selection_options_menu->set_tooltip_text(TTRC("Selection Options"));

	PopupMenu *selection_menu = selection_options_menu->get_popup();
	selection_menu->connect(SceneStringName(id_pressed), callable_mp(this, &GameView::_selection_options_menu_id_pressed));
	selection_menu->add_check_item(TTRC("Don't Select Locked Nodes"), SELECTION_AVOID_LOCKED);
	selection_menu->add_check_item(TTRC("Select Group Over Children"), SELECTION_PREFER_GROUP);

	selection_avoid_locked = EditorSettings::get_singleton()->get_project_metadata("game_view", "selection_avoid_locked", false);
	selection_prefer_group = EditorSettings::get_singleton()->get_project_metadata("game_view", "selection_prefer_group", false);
	selection_menu->set_item_checked(selection_menu->get_item_index(SELECTION_AVOID_LOCKED), selection_avoid_locked);
	selection_menu->set_item_checked(selection_menu->get_item_index(SELECTION_PREFER_GROUP), selection_prefer_group);

	debugger->set_selection_avoid_locked(selection_avoid_locked);
	debugger->set_selection_prefer_group(selection_prefer_group);

	selection_hb->add_child(memnew(VSeparator));

	HBoxContainer *audio_hb = memnew(HBoxContainer);
	main_menu_fc->add_child(audio_hb);

	debug_mute_audio_button = memnew(Button);
	audio_hb->add_child(debug_mute_audio_button);
	debug_mute_audio_button->set_theme_type_variation("FlatButton");
	debug_mute_audio_button->connect(SceneStringName(pressed), callable_mp(this, &GameView::_debug_mute_audio_button_pressed));
	debug_mute_audio_button->set_tooltip_text(debug_mute_audio ? TTRC("Unmute game audio.") : TTRC("Mute game audio."));

	audio_hb->add_child(memnew(VSeparator));

	HBoxContainer *camera_hb = memnew(HBoxContainer);
	main_menu_fc->add_child(camera_hb);

	camera_override_button = memnew(Button);
	camera_hb->add_child(camera_override_button);
	camera_override_button->set_toggle_mode(true);
	camera_override_button->set_theme_type_variation(SceneStringName(FlatButton));
	camera_override_button->set_tooltip_text(TTRC("Override the in-game camera."));
	camera_override_button->connect(SceneStringName(toggled), callable_mp(this, &GameView::_camera_override_button_toggled));

	camera_override_menu = memnew(MenuButton);
	camera_hb->add_child(camera_override_menu);
	camera_override_menu->set_flat(false);
	camera_override_menu->set_theme_type_variation("FlatMenuButton");
	camera_override_menu->set_h_size_flags(SIZE_SHRINK_END);
	camera_override_menu->set_tooltip_text(TTRC("Camera Override Options"));

	menu = camera_override_menu->get_popup();
	menu->connect(SceneStringName(id_pressed), callable_mp(this, &GameView::_camera_override_menu_id_pressed));
	menu->add_item(TTRC("Reset 2D Camera"), CAMERA_RESET_2D);
	menu->add_item(TTRC("Reset 3D Camera"), CAMERA_RESET_3D);
	menu->add_separator();
	menu->add_radio_check_item(TTRC("Manipulate In-Game"), CAMERA_MODE_INGAME);
	menu->set_item_checked(menu->get_item_index(CAMERA_MODE_INGAME), true);
	menu->add_radio_check_item(TTRC("Manipulate From Editors"), CAMERA_MODE_EDITORS);
	_camera_override_menu_id_pressed(EditorSettings::get_singleton()->get_project_metadata("game_view", "camera_override_mode", 0));

	camera_hb->add_child(memnew(VSeparator));

	embedding_hb = memnew(HBoxContainer);
	embedding_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_menu_fc->add_child(embedding_hb);

	embed_options_menu = memnew(MenuButton);
	embedding_hb->add_child(embed_options_menu);
	embed_options_menu->set_flat(false);
	embed_options_menu->set_theme_type_variation("FlatMenuButton");
	embed_options_menu->set_h_size_flags(SIZE_SHRINK_END);
	embed_options_menu->set_tooltip_text(TTRC("Embedding Options"));

	menu = embed_options_menu->get_popup();
	menu->connect(SceneStringName(id_pressed), callable_mp(this, &GameView::_embed_options_menu_menu_id_pressed));
	menu->add_check_item(TTRC("Embed Game on Next Play"), EMBED_RUN_GAME_EMBEDDED);
	menu->add_check_item(TTRC("Make Game Workspace Floating on Next Play"), EMBED_MAKE_FLOATING_ON_PLAY);
	menu->add_separator(TTRC("Embedded Window Sizing"));

	menu->add_radio_check_item(TTRC("Fixed Size"), SIZE_MODE_FIXED);
	menu->set_item_tooltip(menu->get_item_index(SIZE_MODE_FIXED), TTRC("Embedded game size is based on project settings.\nThe 'Keep Aspect' mode is used when the Game Workspace is smaller than the desired size."));
	menu->add_radio_check_item(TTRC("Keep Aspect Ratio"), SIZE_MODE_KEEP_ASPECT);
	menu->set_item_tooltip(menu->get_item_index(SIZE_MODE_KEEP_ASPECT), TTRC("Keep the aspect ratio of the embedded game."));
	menu->add_radio_check_item(TTRC("Stretch to Fit"), SIZE_MODE_STRETCH);
	menu->set_item_tooltip(menu->get_item_index(SIZE_MODE_STRETCH), TTRC("Embedded game size stretches to fit the Game Workspace."));

	game_size_label = memnew(Label());
	embedding_hb->add_child(game_size_label);
	game_size_label->hide();
	// Setting the minimum size prevents the game workspace from resizing indefinitely
	// due to the label size oscillating by a few pixels when the game is in stretch mode
	// and the game workspace is at its minimum size.
	game_size_label->set_custom_minimum_size(Size2(80 * EDSCALE, 0));
	game_size_label->set_h_size_flags(SIZE_EXPAND_FILL);
	game_size_label->set_horizontal_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_RIGHT);

	panel = memnew(PanelContainer);
	add_child(panel);
	panel->set_theme_type_variation("GamePanel");
	panel->set_v_size_flags(SIZE_EXPAND_FILL);
#ifdef MACOS_ENABLED
	panel->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
#endif

	panel->add_child(embedded_process);
	embedded_process->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	embedded_process->connect("embedding_failed", callable_mp(this, &GameView::_embedding_failed));
	embedded_process->connect("embedding_completed", callable_mp(this, &GameView::_embedding_completed));
	embedded_process->connect("embedded_process_updated", callable_mp(this, &GameView::_embedded_process_updated));
	embedded_process->connect("embedded_process_focused", callable_mp(this, &GameView::_embedded_process_focused));
	embedded_process->set_custom_minimum_size(Size2i(100, 100));

	state_label = memnew(Label());
	panel->add_child(state_label);
	state_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	state_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	state_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
	state_label->set_anchors_and_offsets_preset(PRESET_FULL_RECT);

	_update_debugger_buttons();

	p_debugger->connect("session_started", callable_mp(this, &GameView::_sessions_changed));
	p_debugger->connect("session_stopped", callable_mp(this, &GameView::_sessions_changed));

	p_wrapper->set_override_close_request(true);
	p_wrapper->connect("window_close_requested", callable_mp(this, &GameView::_window_close_request));
	p_wrapper->connect("window_size_changed", callable_mp(this, &GameView::_update_floating_window_settings));

	EditorDebuggerNode::get_singleton()->connect("breaked", callable_mp(this, &GameView::_debugger_breaked));

	EditorFeatureProfileManager::get_singleton()->connect("current_feature_profile_changed", callable_mp(this, &GameView::_feature_profile_changed));
}

///////

void GameViewPluginBase::selected_notify() {
	if (_is_window_wrapper_enabled()) {
#ifdef ANDROID_ENABLED
		notify_main_screen_changed(get_plugin_name());
#else
		window_wrapper->grab_window_focus();
#endif // ANDROID_ENABLED
		_focus_another_editor();
	}
}

#ifndef ANDROID_ENABLED
void GameViewPluginBase::make_visible(bool p_visible) {
	if (p_visible) {
		window_wrapper->show();
	} else {
		window_wrapper->hide();
	}
}

void GameViewPluginBase::set_window_layout(Ref<ConfigFile> p_layout) {
	game_view->set_window_layout(p_layout);
}

void GameViewPluginBase::get_window_layout(Ref<ConfigFile> p_layout) {
	game_view->get_window_layout(p_layout);
}

void GameViewPluginBase::setup(Ref<GameViewDebugger> p_debugger, EmbeddedProcessBase *p_embedded_process) {
	debugger = p_debugger;

	window_wrapper = memnew(WindowWrapper);
	window_wrapper->set_margins_enabled(true);

	game_view = memnew(GameView(debugger, p_embedded_process, window_wrapper));
	game_view->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	window_wrapper->set_wrapped_control(game_view, nullptr);

	EditorNode::get_singleton()->get_editor_main_screen()->get_control()->add_child(window_wrapper);
	window_wrapper->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	window_wrapper->hide();
	window_wrapper->connect("window_visibility_changed", callable_mp(this, &GameViewPlugin::_focus_another_editor).unbind(1));
}

#endif // ANDROID_ENABLED

void GameViewPluginBase::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
#ifndef ANDROID_ENABLED
			window_wrapper->set_window_title(vformat(TTR("%s - Godot Engine"), TTR("Game Workspace")));
#endif
		} break;
		case NOTIFICATION_ENTER_TREE: {
			add_debugger_plugin(debugger);
			connect("main_screen_changed", callable_mp(this, &GameViewPluginBase::_save_last_editor));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			remove_debugger_plugin(debugger);
			disconnect("main_screen_changed", callable_mp(this, &GameViewPluginBase::_save_last_editor));
		} break;
	}
}

void GameViewPluginBase::_save_last_editor(const String &p_editor) {
	if (p_editor != get_plugin_name()) {
		last_editor = p_editor;
	}
}

void GameViewPluginBase::_focus_another_editor() {
	if (_is_window_wrapper_enabled()) {
		if (last_editor.is_empty()) {
			EditorNode::get_singleton()->get_editor_main_screen()->select(EditorMainScreen::EDITOR_2D);
		} else {
			EditorInterface::get_singleton()->set_main_screen_editor(last_editor);
		}
	}
}

bool GameViewPluginBase::_is_window_wrapper_enabled() const {
#ifdef ANDROID_ENABLED
	return true;
#else
	return window_wrapper->get_window_enabled();
#endif // ANDROID_ENABLED
}

GameViewPluginBase::GameViewPluginBase() {
#ifdef ANDROID_ENABLED
	debugger.instantiate();
#endif
}

GameViewPlugin::GameViewPlugin() :
		GameViewPluginBase() {
#ifndef ANDROID_ENABLED
	Ref<GameViewDebugger> game_view_debugger;
	game_view_debugger.instantiate();
	EmbeddedProcess *embedded_process = memnew(EmbeddedProcess);
	setup(game_view_debugger, embedded_process);
#endif
}

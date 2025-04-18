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
#include "core/string/translation_server.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/editor_feature_profile.h"
#include "editor/editor_interface.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/gui/editor_run_bar.h"
#include "editor/plugins/embedded_process.h"
#include "editor/themes/editor_scale.h"
#include "editor/window_wrapper.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"

void GameViewDebugger::_session_started(Ref<EditorDebuggerSession> p_session, GameView *p_game_view) {
	if (!is_feature_enabled) {
		return;
	}
	if (!p_game_view->is_view_embedding()) {
		return;
	}
	if (p_game_view->get_plugin()->get_main_game_view() != p_game_view) {
		//TODO - Allow the debugger controls to work on both GameViews. For now, only use the main one.
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
	settings["embedded_window_title"] = p_game_view->get_embedded_window_title();

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
	Array mute_audio_data;
	mute_audio_data.append(mute_audio);
	p_session->send_message("scene:debug_mute_audio", mute_audio_data);

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

void GameViewDebugger::set_game_view(GameView *p_game_view) {
	game_view = p_game_view;
}

void GameViewDebugger::setup_session(int p_session_id) {
	Ref<EditorDebuggerSession> session = get_session(p_session_id);
	ERR_FAIL_COND(session.is_null());

	sessions.append(session);

	session->connect("started", callable_mp(this, &GameViewDebugger::_session_started).bind(session, game_view));
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

GameViewDebugger::GameViewDebugger() {
	EditorFeatureProfileManager::get_singleton()->connect("current_feature_profile_changed", callable_mp(this, &GameViewDebugger::_feature_profile_changed));
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

	if (embedded_process->is_embedding_completed()) {
		if (!embedded_script_debugger || !embedded_script_debugger->is_session_active() || embedded_script_debugger->get_remote_pid() != embedded_process->get_embedded_pid()) {
			_attach_script_debugger();
		}
	}
}

void GameView::show_update_window_wrapper() {
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

bool GameView::is_view_embedding() {
	if (!embed_on_play || plugin->get_embed_available() != GameViewPlugin::EMBED_AVAILABLE || !window_wrapper->is_visible() || !is_visible()) {
		return false;
	}
	if (get_embedded_window_title().is_empty()) {
		return plugin->get_game_view_with_main_screen_embed() == this;
	}
	return plugin->get_game_view_with_selected_window(get_embedded_window_title()) == this;
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

	if (is_view_embedding()) {
		// It's important to disable the low power mode when unfocused because otherwise
		// the button in the editor are not responsive and if the user moves the mouse quickly,
		// the mouse clicks are not registered.
		EditorNode::get_singleton()->set_unfocused_low_processor_usage_mode_enabled(false);
		update_embed_window_size();
		if (!window_wrapper->get_window_enabled()) {
			EditorNode::get_singleton()->get_editor_main_screen()->select(EditorMainScreen::EDITOR_GAME);
			// Reset the normal size of the bottom panel when fully expanded.
			EditorNode::get_singleton()->get_bottom_panel()->set_expanded(false);
			embedded_process->grab_focus();
		}
		embedded_process->embed_process(current_process_id, get_embedded_window_title());
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
	_attach_script_debugger();
	_update_ui();
	if (make_floating_on_play) {
		get_window()->set_flag(Window::FLAG_ALWAYS_ON_TOP, bool(GLOBAL_GET("display/window/size/always_on_top")));
	}
}

void GameView::_embedding_failed() {
	state_label->set_text(TTR("Connection impossible to the game process."));
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
	// Update the window size and aspect ratio.
	update_embed_window_size();

	if (window_wrapper->get_window_enabled()) {
		show_update_window_wrapper();
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

	PopupMenu *menu = camera_override_menu->get_popup();
	if (menu->get_item_index(CAMERA_MODE_INGAME) > -1 && menu->get_item_index(CAMERA_RESET_2D) > -1 && menu->get_item_index(CAMERA_RESET_3D)) {
		bool disable_camera_reset = empty || !camera_override_button->is_pressed() || !menu->is_item_checked(menu->get_item_index(CAMERA_MODE_INGAME));
		menu->set_item_disabled(menu->get_item_index(CAMERA_RESET_2D), disable_camera_reset);
		menu->set_item_disabled(menu->get_item_index(CAMERA_RESET_3D), disable_camera_reset);
	}

	if (empty) {
		suspend_button->set_pressed(false);
		camera_override_button->set_pressed(false);
	}
	next_frame_button->set_disabled(!suspend_button->is_pressed());
}

void GameView::_update_window_selector_controls() {
	bool has_subwindow_embedding = DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_SUBWINDOW_EMBEDDING);
	window_select_dropdown->set_visible(has_subwindow_embedding);
	if (!has_subwindow_embedding) {
		window_select_text->set_text("");
		window_select_text->hide();
		set_embedded_window_title("");
		show_embedded_window_title = false;
	}
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

	EditorSettings::get_singleton()->set_project_metadata(setting_prefix, "select_mode", mode);
}

void GameView::_embed_options_menu_menu_id_pressed(int p_id) {
	switch (p_id) {
		case EMBED_RUN_GAME_EMBEDDED: {
			embed_on_play = !embed_on_play;
			int game_mode = EDITOR_GET("run/window_placement/game_embed_mode");
			if (game_mode == 0) { // Save only if not overridden by editor.
				EditorSettings::get_singleton()->set_project_metadata(setting_prefix, "embed_on_play", embed_on_play);
			}
		} break;
		case EMBED_MAKE_FLOATING_ON_PLAY: {
			make_floating_on_play = !make_floating_on_play;
			int game_mode = EDITOR_GET("run/window_placement/game_embed_mode");
			if (game_mode == 0) { // Save only if not overridden by editor.
				EditorSettings::get_singleton()->set_project_metadata(setting_prefix, "make_floating_on_play", make_floating_on_play);
			}
		} break;
		case GAME_VIEW_SINGLE: {
			// The dual view setting is global for both game views.
			EditorSettings::get_singleton()->set_project_metadata("game_view", "dual_game_view", false);
			plugin->enable_dual_game_view(false);
		} break;
		case GAME_VIEW_DUAL: {
			EditorSettings::get_singleton()->set_project_metadata("game_view", "dual_game_view", true);
			plugin->enable_dual_game_view(true);
		} break;
	}
	_update_embed_menu_options();
	_update_ui();
}

void GameView::_size_mode_button_pressed(int size_mode) {
	embed_size_mode = (EmbedSizeMode)size_mode;
	EditorSettings::get_singleton()->set_project_metadata(setting_prefix, "embed_size_mode", size_mode);

	_update_embed_menu_options();
	update_embed_window_size();
}

void GameView::_select_window_dropdown_pressed(int p_id) {
	show_embedded_window_title = p_id == WINDOW_SELECT_CUSTOM;
	if (p_id == WINDOW_SELECT_MAIN) {
		set_embedded_window_title("");
	} else if (p_id == WINDOW_SELECT_CUSTOM) {
		set_embedded_window_title(window_select_text->get_text());
	}
	// Update UI label for all GameViews.
	plugin->game_view_changed_window_target();
}

void GameView::_window_select_text_changed(String p_text) {
	set_embedded_window_title(p_text);
	plugin->game_view_changed_window_target();
}

void GameView::_update_ui() {
	bool show_game_size = false;
	GameViewPlugin::EmbedAvailability available = plugin->get_embed_available();
	GameView *main_window_game_view = plugin->get_game_view_with_main_screen_embed();

	switch (available) {
		case GameViewPlugin::EMBED_AVAILABLE:
			if (embedded_process->is_embedding_completed()) {
				state_label->set_text("");
				show_game_size = true;
			} else if (embedded_process->is_embedding_in_progress()) {
				state_label->set_text(TTR("Game starting..."));
			} else if (EditorRunBar::get_singleton()->is_playing()) {
				state_label->set_text(TTR("Game running not embedded."));
			} else if (embed_on_play) {
				if (get_embedded_window_title().is_empty() && main_window_game_view != nullptr && main_window_game_view != this) {
					state_label->set_text(TTR("Only one Game View can embed the main window."));
				} else {
					state_label->set_text(TTR("Press play to start the game."));
				}
			} else {
				state_label->set_text(TTR("Embedding is disabled."));
			}
			break;
		case GameViewPlugin::EMBED_NOT_AVAILABLE_FEATURE_NOT_SUPPORTED:
			if (DisplayServer::get_singleton()->get_name() == "Wayland") {
				state_label->set_text(TTR("Game embedding not available on Wayland.\nWayland can be disabled in the Editor Settings (Run > Platforms > Linux/*BSD > Prefer Wayland)."));
			} else {
				state_label->set_text(TTR("Game embedding not available on your OS."));
			}
			break;
		case GameViewPlugin::EMBED_NOT_AVAILABLE_PROJECT_DISPLAY_DRIVER:
			state_label->set_text(vformat(TTR("Game embedding not available for the Display Server: '%s'.\nDisplay Server can be modified in the Project Settings (Display > Display Server > Driver)."), GLOBAL_GET("display/display_server/driver")));
			break;
		case GameViewPlugin::EMBED_NOT_AVAILABLE_MINIMIZED:
			state_label->set_text(TTR("Game embedding not available when the game starts minimized.") + "\n" + TTR("Consider overriding the window mode project setting with the editor feature tag to Windowed to use game embedding while leaving the exported project intact."));
			break;
		case GameViewPlugin::EMBED_NOT_AVAILABLE_MAXIMIZED:
			state_label->set_text(TTR("Game embedding not available when the game starts maximized.") + "\n" + TTR("Consider overriding the window mode project setting with the editor feature tag to Windowed to use game embedding while leaving the exported project intact."));
			break;
		case GameViewPlugin::EMBED_NOT_AVAILABLE_FULLSCREEN:
			state_label->set_text(TTR("Game embedding not available when the game starts in fullscreen.") + "\n" + TTR("Consider overriding the window mode project setting with the editor feature tag to Windowed to use game embedding while leaving the exported project intact."));
			break;
		case GameViewPlugin::EMBED_NOT_AVAILABLE_SINGLE_WINDOW_MODE:
			state_label->set_text(TTR("Game embedding not available in single window mode."));
			break;
	}

	if (available == GameViewPlugin::EMBED_AVAILABLE) {
		if (state_label->has_theme_color_override(SceneStringName(font_color))) {
			state_label->remove_theme_color_override(SceneStringName(font_color));
		}
	} else {
		state_label->add_theme_color_override(SceneStringName(font_color), state_label->get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
	}

	game_size_label->set_visible(show_game_size);
	window_select_text->set_visible(show_embedded_window_title);
}

void GameView::_update_embed_menu_options() {
	bool is_multi_window = window_wrapper->is_window_available();
	PopupMenu *menu = embed_options_menu->get_popup();
	menu->set_item_checked(menu->get_item_index(EMBED_RUN_GAME_EMBEDDED), embed_on_play);
	menu->set_item_checked(menu->get_item_index(EMBED_MAKE_FLOATING_ON_PLAY), make_floating_on_play && is_multi_window);

	menu->set_item_disabled(menu->get_item_index(EMBED_MAKE_FLOATING_ON_PLAY), !embed_on_play || !is_multi_window);

	bool dual_view_enabled = (bool)EditorSettings::get_singleton()->get_project_metadata("game_view", "dual_game_view", false);
	menu->set_item_checked(menu->get_item_index(GAME_VIEW_SINGLE), !dual_view_enabled);
	menu->set_item_checked(menu->get_item_index(GAME_VIEW_DUAL), dual_view_enabled);

	fixed_size_button->set_pressed(embed_size_mode == SIZE_MODE_FIXED);
	keep_aspect_button->set_pressed(embed_size_mode == SIZE_MODE_KEEP_ASPECT);
	stretch_button->set_pressed(embed_size_mode == SIZE_MODE_STRETCH);
}

void GameView::update_all_ui() {
	_update_embed_menu_options();
	_update_ui();
}

void GameView::update_embed_window_size() {
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

	EditorSettings::get_singleton()->set_project_metadata(setting_prefix, "hide_selection", p_pressed);
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
			if (menu->get_item_index(p_id) > -1) {
				menu->set_item_checked(menu->get_item_index(p_id), true);
			}

			_update_debugger_buttons();

			EditorSettings::get_singleton()->set_project_metadata(setting_prefix, "camera_override_mode", p_id);
		} break;
		case CAMERA_MODE_EDITORS: {
			debugger->set_camera_manipulate_mode(EditorDebuggerNode::OVERRIDE_EDITORS);
			if (menu->get_item_index(p_id) > -1) {
				menu->set_item_checked(menu->get_item_index(p_id), true);
			}

			_update_debugger_buttons();

			EditorSettings::get_singleton()->set_project_metadata(setting_prefix, "camera_override_mode", p_id);
		} break;
	}
}

void GameView::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			suspend_button->set_button_icon(get_editor_theme_icon(SNAME("Pause")));
			next_frame_button->set_button_icon(get_editor_theme_icon(SNAME("NextFrame")));

			node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_button_icon(get_editor_theme_icon(SNAME("InputEventJoypadMotion")));
			node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_button_icon(get_editor_theme_icon(SNAME("2DNodes")));
			node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_button_icon(get_editor_theme_icon(SNAME("Node3D")));

			select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_button_icon(get_editor_theme_icon(SNAME("ToolSelect")));
			select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_button_icon(get_editor_theme_icon(SNAME("ListSelect")));

			hide_selection->set_button_icon(get_editor_theme_icon(hide_selection->is_pressed() ? SNAME("GuiVisibilityHidden") : SNAME("GuiVisibilityVisible")));
			fixed_size_button->set_button_icon(get_editor_theme_icon(SNAME("FixedSize")));
			keep_aspect_button->set_button_icon(get_editor_theme_icon(SNAME("KeepAspect")));
			stretch_button->set_button_icon(get_editor_theme_icon(SNAME("Stretch")));
			embed_options_menu->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));

			PopupMenu *embed_popup = embed_options_menu->get_popup();
			embed_popup->set_item_icon(embed_popup->get_item_index(GAME_VIEW_SINGLE), get_editor_theme_icon(SNAME("Panels1")));
			embed_popup->set_item_icon(embed_popup->get_item_index(GAME_VIEW_DUAL), get_editor_theme_icon(SNAME("Panels2")));

			debug_mute_audio_button->set_button_icon(get_editor_theme_icon(debug_mute_audio ? SNAME("AudioMute") : SNAME("AudioStreamPlayer")));
			camera_override_button->set_button_icon(get_editor_theme_icon(SNAME("Camera")));
			camera_override_menu->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));
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
						embed_on_play = EditorSettings::get_singleton()->get_project_metadata(setting_prefix, "embed_on_play", true);
						make_floating_on_play = EditorSettings::get_singleton()->get_project_metadata(setting_prefix, "make_floating_on_play", true);
					} break;
				}
				embed_size_mode = (EmbedSizeMode)(int)EditorSettings::get_singleton()->get_project_metadata(setting_prefix, "embed_size_mode", SIZE_MODE_FIXED);
				keep_aspect_button->set_pressed(EditorSettings::get_singleton()->get_project_metadata(setting_prefix, "keep_aspect", true));
				_update_embed_menu_options();

				EditorRunBar::get_singleton()->connect("play_pressed", callable_mp(this, &GameView::_play_pressed));
				EditorRunBar::get_singleton()->connect("stop_pressed", callable_mp(this, &GameView::_stop_pressed));

				// Listen for project settings changes to update the window size and aspect ratio.
				ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &GameView::_editor_or_project_settings_changed));
				EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &GameView::_editor_or_project_settings_changed));
			} else {
				// Embedding not available.
				embedding_separator->hide();
				embed_options_menu->hide();
				fixed_size_button->hide();
				keep_aspect_button->hide();
				stretch_button->hide();
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

void GameView::set_is_feature_enabled(bool p_enabled) {
	is_feature_enabled = p_enabled;
}

void GameView::set_time_buttons_enabled(bool p_enabled) {
	suspend_button->set_visible(p_enabled);
	next_frame_button->set_visible(p_enabled);
	time_buttons_separator->set_visible(p_enabled);
}

void GameView::set_debugger_controls_enabled(bool p_enabled) {
	debugger_controls_enabled = p_enabled;

	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_visible(p_enabled);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_visible(p_enabled);
	node_type_separator->set_visible(p_enabled);
	hide_selection->set_visible(p_enabled);
	hide_selection_separator->set_visible(p_enabled);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_visible(p_enabled);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_visible(p_enabled);
	select_mode_separator->set_visible(p_enabled);
	debug_mute_audio_button->set_visible(p_enabled);
	debug_mute_audio_separator->set_visible(p_enabled);
	camera_override_button->set_visible(p_enabled);
	camera_override_menu->set_visible(p_enabled);
	if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_WINDOW_EMBEDDING)) {
		embedding_separator->set_visible(p_enabled);
	}
	// Also update the 3D mode button.
	_feature_profile_changed();
}

void GameView::set_embedded_window_title(String p_embedded_window_title) {
	embedded_window_title = p_embedded_window_title;
}

String GameView::get_embedded_window_title() {
	return embedded_window_title;
}

bool GameView::get_embed_on_play() {
	return embed_on_play;
}

bool GameView::get_make_floating_on_play() {
	return make_floating_on_play;
}

EmbeddedProcess *GameView::get_embedded_process() {
	return embedded_process;
}

GameViewPlugin *GameView::get_plugin() {
	return plugin;
}

void GameView::set_window_layout(Ref<ConfigFile> p_layout) {
	floating_window_rect = p_layout->get_value(setting_prefix, "floating_window_rect", Rect2i());
	floating_window_screen = p_layout->get_value(setting_prefix, "floating_window_screen", -1);
	window_select_text->set_text(p_layout->get_value(setting_prefix, "embedded_window_title", ""));
	if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_SUBWINDOW_EMBEDDING)) {
		int window_dropdown_selection = p_layout->get_value(setting_prefix, "show_embedded_window", WINDOW_SELECT_MAIN);
		if (window_dropdown_selection == WINDOW_SELECT_MAIN) {
			window_select_dropdown->select(0);
		} else {
			window_select_dropdown->select(1);
		}
		_select_window_dropdown_pressed(window_dropdown_selection);
	}
}

void GameView::get_window_layout(Ref<ConfigFile> p_layout) {
	if (window_wrapper->get_window_enabled()) {
		_update_floating_window_settings();
	}

	p_layout->set_value(setting_prefix, "floating_window_rect", floating_window_rect);
	p_layout->set_value(setting_prefix, "floating_window_screen", floating_window_screen);
	int show_window_value = show_embedded_window_title ? WINDOW_SELECT_CUSTOM : WINDOW_SELECT_MAIN;
	p_layout->set_value(setting_prefix, "show_embedded_window", show_window_value);
	p_layout->set_value(setting_prefix, "embedded_window_title", window_select_text->get_text());
}

void GameView::_update_floating_window_settings() {
	if (window_wrapper->get_window_enabled()) {
		floating_window_rect = window_wrapper->get_window_rect();
		floating_window_screen = window_wrapper->get_window_screen();
	}
}

void GameView::_attach_script_debugger() {
	if (embedded_script_debugger) {
		_detach_script_debugger();
	}

	embedded_script_debugger = nullptr;
	for (int i = 0; EditorDebuggerNode::get_singleton()->get_debugger(i); i++) {
		ScriptEditorDebugger *script_debugger = EditorDebuggerNode::get_singleton()->get_debugger(i);
		if (script_debugger->is_session_active() && script_debugger->get_remote_pid() == embedded_process->get_embedded_pid()) {
			embedded_script_debugger = script_debugger;
			break;
		}
	}

	if (embedded_script_debugger) {
		embedded_script_debugger->connect("remote_window_title_changed", callable_mp(this, &GameView::_remote_window_title_changed));
	}
}

void GameView::_detach_script_debugger() {
	if (embedded_script_debugger) {
		embedded_script_debugger->disconnect("remote_window_title_changed", callable_mp(this, &GameView::_remote_window_title_changed));
		embedded_script_debugger = nullptr;
	}
}

void GameView::_remote_window_title_changed(String title) {
	window_wrapper->set_window_title(title);
}

void GameView::_window_close_request() {
	// Before the parent window closed, we close the embedded game. That prevents
	// the embedded game to be seen without a parent window for a fraction of second.
	if (EditorRunBar::get_singleton()->is_playing() && (embedded_process->is_embedding_completed() || embedded_process->is_embedding_in_progress())) {
		// Try to gracefully close the window. That way, the NOTIFICATION_WM_CLOSE_REQUEST
		// notification should be propagated in the game process.
		if (embedded_process->is_embedding_completed()) {
			embedded_process->request_close();
		} else {
			embedded_process->reset();
		}

		if (EditorRunBar::get_singleton()->is_playing()) {
			// Then, since we could have closed an unhandled secondary window,
			// ensure that we stop the game process.
			callable_mp(EditorRunBar::get_singleton(), &EditorRunBar::stop_playing).call_deferred();
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

	update_embed_window_size();
}

void GameView::_feature_profile_changed() {
	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();
	bool is_profile_null = profile.is_null();

	is_feature_enabled = is_profile_null || !profile->is_feature_disabled(EditorFeatureProfile::FEATURE_GAME);

	bool is_3d_enabled = is_profile_null || !profile->is_feature_disabled(EditorFeatureProfile::FEATURE_3D);
	if (!is_3d_enabled && node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->is_pressed()) {
		_node_type_pressed(RuntimeNodeSelect::NODE_TYPE_NONE);
	}
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_visible(is_3d_enabled && debugger_controls_enabled);
}

GameView::GameView(Ref<GameViewDebugger> p_debugger, GameViewPlugin *p_plugin, WindowWrapper *p_wrapper, String p_setting_prefix) {
	debugger = p_debugger;
	plugin = p_plugin;
	window_wrapper = p_wrapper;
	setting_prefix = p_setting_prefix;

	// Add some margin to the sides for better aesthetics.
	// This prevents the first button's hover/pressed effect from "touching" the panel's border,
	// which looks ugly.
	MarginContainer *toolbar_margin = memnew(MarginContainer);
	toolbar_margin->add_theme_constant_override("margin_left", 4 * EDSCALE);
	toolbar_margin->add_theme_constant_override("margin_right", 4 * EDSCALE);
	add_child(toolbar_margin);

	HBoxContainer *main_menu_hbox = memnew(HBoxContainer);
	toolbar_margin->add_child(main_menu_hbox);

	suspend_button = memnew(Button);
	main_menu_hbox->add_child(suspend_button);
	suspend_button->set_toggle_mode(true);
	suspend_button->set_theme_type_variation(SceneStringName(FlatButton));
	suspend_button->connect(SceneStringName(toggled), callable_mp(this, &GameView::_suspend_button_toggled));
	suspend_button->set_tooltip_text(TTR("Suspend"));
	suspend_button->set_accessibility_name(TTRC("Suspend"));

	next_frame_button = memnew(Button);
	main_menu_hbox->add_child(next_frame_button);
	next_frame_button->set_theme_type_variation(SceneStringName(FlatButton));
	next_frame_button->connect(SceneStringName(pressed), callable_mp(*debugger, &GameViewDebugger::next_frame));
	next_frame_button->set_tooltip_text(TTR("Next Frame"));
	next_frame_button->set_accessibility_name(TTRC("Next Frame"));

	time_buttons_separator = memnew(VSeparator);
	main_menu_hbox->add_child(time_buttons_separator);

	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE] = memnew(Button);
	main_menu_hbox->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_text(TTR("Input"));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_pressed(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_theme_type_variation(SceneStringName(FlatButton));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->connect(SceneStringName(pressed), callable_mp(this, &GameView::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_NONE));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_tooltip_text(TTR("Allow game input."));

	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D] = memnew(Button);
	main_menu_hbox->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_text(TTR("2D"));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_theme_type_variation(SceneStringName(FlatButton));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->connect(SceneStringName(pressed), callable_mp(this, &GameView::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_2D));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_tooltip_text(TTR("Disable game input and allow to select Node2Ds, Controls, and manipulate the 2D camera."));

	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D] = memnew(Button);
	main_menu_hbox->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_text(TTR("3D"));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_theme_type_variation(SceneStringName(FlatButton));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->connect(SceneStringName(pressed), callable_mp(this, &GameView::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_3D));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_tooltip_text(TTR("Disable game input and allow to select Node3Ds and manipulate the 3D camera."));

	node_type_separator = memnew(VSeparator);
	main_menu_hbox->add_child(node_type_separator);

	hide_selection = memnew(Button);
	main_menu_hbox->add_child(hide_selection);
	hide_selection->set_toggle_mode(true);
	hide_selection->set_theme_type_variation(SceneStringName(FlatButton));
	hide_selection->connect(SceneStringName(toggled), callable_mp(this, &GameView::_hide_selection_toggled));
	hide_selection->set_tooltip_text(TTR("Toggle Selection Visibility"));
	hide_selection->set_accessibility_name(TTRC("Selection Visibility"));
	hide_selection->set_pressed(EditorSettings::get_singleton()->get_project_metadata(setting_prefix, "hide_selection", false));

	hide_selection_separator = memnew(VSeparator);
	main_menu_hbox->add_child(hide_selection_separator);

	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE] = memnew(Button);
	main_menu_hbox->add_child(select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_toggle_mode(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_pressed(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_theme_type_variation(SceneStringName(FlatButton));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->connect(SceneStringName(pressed), callable_mp(this, &GameView::_select_mode_pressed).bind(RuntimeNodeSelect::SELECT_MODE_SINGLE));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_select", TTRC("Select Mode"), Key::Q));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_shortcut_context(this);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_tooltip_text(keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL) + TTR("Alt+RMB: Show list of all nodes at position clicked."));

	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST] = memnew(Button);
	main_menu_hbox->add_child(select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_toggle_mode(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_theme_type_variation(SceneStringName(FlatButton));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->connect(SceneStringName(pressed), callable_mp(this, &GameView::_select_mode_pressed).bind(RuntimeNodeSelect::SELECT_MODE_LIST));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_tooltip_text(TTR("Show list of selectable nodes at position clicked."));

	_select_mode_pressed(EditorSettings::get_singleton()->get_project_metadata(setting_prefix, "select_mode", 0));
	select_mode_separator = memnew(VSeparator);
	main_menu_hbox->add_child(select_mode_separator);

	debug_mute_audio_button = memnew(Button);
	main_menu_hbox->add_child(debug_mute_audio_button);
	debug_mute_audio_button->set_theme_type_variation("FlatButton");
	debug_mute_audio_button->connect(SceneStringName(pressed), callable_mp(this, &GameView::_debug_mute_audio_button_pressed));
	debug_mute_audio_button->set_tooltip_text(debug_mute_audio ? TTRC("Unmute game audio.") : TTRC("Mute game audio."));

	debug_mute_audio_separator = memnew(VSeparator);
	main_menu_hbox->add_child(debug_mute_audio_separator);

	camera_override_button = memnew(Button);
	main_menu_hbox->add_child(camera_override_button);
	camera_override_button->set_toggle_mode(true);
	camera_override_button->set_theme_type_variation(SceneStringName(FlatButton));
	camera_override_button->set_tooltip_text(TTR("Override the in-game camera."));
	camera_override_button->set_accessibility_name(TTRC("Override In-game Camera"));
	camera_override_button->connect(SceneStringName(toggled), callable_mp(this, &GameView::_camera_override_button_toggled));

	camera_override_menu = memnew(MenuButton);
	main_menu_hbox->add_child(camera_override_menu);
	camera_override_menu->set_flat(false);
	camera_override_menu->set_theme_type_variation("FlatMenuButton");
	camera_override_menu->set_h_size_flags(SIZE_SHRINK_END);
	camera_override_menu->set_tooltip_text(TTR("Camera Override Options"));
	camera_override_menu->set_accessibility_name(TTRC("Camera Override Options"));
	_camera_override_menu_id_pressed(EditorSettings::get_singleton()->get_project_metadata(setting_prefix, "camera_override_mode", 0));

	PopupMenu *menu = camera_override_menu->get_popup();
	menu->connect(SceneStringName(id_pressed), callable_mp(this, &GameView::_camera_override_menu_id_pressed));
	menu->add_item(TTR("Reset 2D Camera"), CAMERA_RESET_2D);
	menu->add_item(TTR("Reset 3D Camera"), CAMERA_RESET_3D);
	menu->add_separator();
	menu->add_radio_check_item(TTR("Manipulate In-Game"), CAMERA_MODE_INGAME);
	menu->set_item_checked(menu->get_item_index(CAMERA_MODE_INGAME), true);
	menu->add_radio_check_item(TTR("Manipulate From Editors"), CAMERA_MODE_EDITORS);

	embedding_separator = memnew(VSeparator);
	main_menu_hbox->add_child(embedding_separator);

	fixed_size_button = memnew(Button);
	main_menu_hbox->add_child(fixed_size_button);
	fixed_size_button->set_toggle_mode(true);
	fixed_size_button->set_theme_type_variation("FlatButton");
	fixed_size_button->set_tooltip_text(TTR("Embedded game size is based on project settings.\nThe 'Keep Aspect' mode is used when the Game Workspace is smaller than the desired size."));
	fixed_size_button->set_accessibility_name(TTRC("Fixed Size"));
	fixed_size_button->connect(SceneStringName(pressed), callable_mp(this, &GameView::_size_mode_button_pressed).bind(SIZE_MODE_FIXED));

	keep_aspect_button = memnew(Button);
	main_menu_hbox->add_child(keep_aspect_button);
	keep_aspect_button->set_toggle_mode(true);
	keep_aspect_button->set_theme_type_variation("FlatButton");
	keep_aspect_button->set_tooltip_text(TTR("Keep the aspect ratio of the embedded game."));
	keep_aspect_button->set_accessibility_name(TTRC("Keep Aspect Ratio"));
	keep_aspect_button->connect(SceneStringName(pressed), callable_mp(this, &GameView::_size_mode_button_pressed).bind(SIZE_MODE_KEEP_ASPECT));

	stretch_button = memnew(Button);
	main_menu_hbox->add_child(stretch_button);
	stretch_button->set_toggle_mode(true);
	stretch_button->set_theme_type_variation("FlatButton");
	stretch_button->set_tooltip_text(TTR("Embedded game size stretches to fit the Game Workspace."));
	stretch_button->set_accessibility_name(TTRC("Stretch"));
	stretch_button->connect(SceneStringName(pressed), callable_mp(this, &GameView::_size_mode_button_pressed).bind(SIZE_MODE_STRETCH));

	embed_options_menu = memnew(MenuButton);
	main_menu_hbox->add_child(embed_options_menu);
	embed_options_menu->set_flat(false);
	embed_options_menu->set_theme_type_variation("FlatMenuButton");
	embed_options_menu->set_h_size_flags(SIZE_SHRINK_END);
	embed_options_menu->set_tooltip_text(TTR("Embedding Options"));
	embed_options_menu->set_accessibility_name(TTRC("Embedding Options"));

	menu = embed_options_menu->get_popup();
	menu->connect(SceneStringName(id_pressed), callable_mp(this, &GameView::_embed_options_menu_menu_id_pressed));
	menu->add_check_item(TTR("Embed Game on Next Play"), EMBED_RUN_GAME_EMBEDDED);
	menu->add_check_item(TTR("Make Game Workspace Floating on Next Play"), EMBED_MAKE_FLOATING_ON_PLAY);
	menu->add_separator();
	menu->add_radio_check_item(TTR("Single Game View"), GAME_VIEW_SINGLE);
	menu->add_radio_check_item(TTR("Dual Game View"), GAME_VIEW_DUAL);

	main_menu_hbox->add_spacer();

	game_size_label = memnew(Label());
	main_menu_hbox->add_child(game_size_label);
	game_size_label->hide();
	// Setting the minimum size prevents the game workspace from resizing indefinitely
	// due to the label size oscillating by a few pixels when the game is in stretch mode
	// and the game workspace is at its minimum size.
	game_size_label->set_custom_minimum_size(Size2(80 * EDSCALE, 0));
	game_size_label->set_horizontal_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_RIGHT);

	window_select_dropdown = memnew(OptionButton());
	main_menu_hbox->add_child(window_select_dropdown);
	window_select_dropdown->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &GameView::_select_window_dropdown_pressed));
	window_select_dropdown->add_item(TTR("Main Window"), WINDOW_SELECT_MAIN);
	window_select_dropdown->add_item(TTR("Custom Window"), WINDOW_SELECT_CUSTOM);

	window_select_text = memnew(LineEdit());
	main_menu_hbox->add_child(window_select_text);
	window_select_text->set_custom_minimum_size(Size2(180, 0));
	window_select_text->set_tooltip_text(ETR("Enter the title of the window that will be embedded on next play."));
	window_select_text->set_placeholder(ETR("Window Title"));
	window_select_text->connect(SceneStringName(text_changed), callable_mp(this, &GameView::_window_select_text_changed));
	window_select_text->hide();

	panel = memnew(Panel);
	add_child(panel);
	panel->set_theme_type_variation("GamePanel");
	panel->set_v_size_flags(SIZE_EXPAND_FILL);
	panel->set_custom_minimum_size(Size2(0, 80));

	embedded_process = memnew(EmbeddedProcess);
	panel->add_child(embedded_process);
	embedded_process->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	embedded_process->connect("embedding_failed", callable_mp(this, &GameView::_embedding_failed));
	embedded_process->connect("embedding_completed", callable_mp(this, &GameView::_embedding_completed));
	embedded_process->connect("embedded_process_updated", callable_mp(this, &GameView::_embedded_process_updated));
	embedded_process->connect("embedded_process_focused", callable_mp(this, &GameView::_embedded_process_focused));
	embedded_process->set_custom_minimum_size(Size2i(100, 100));

	MarginContainer *state_container = memnew(MarginContainer);
	state_container->add_theme_constant_override("margin_left", 8 * EDSCALE);
	state_container->add_theme_constant_override("margin_right", 8 * EDSCALE);
	state_container->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	panel->add_child(state_container);

	state_label = memnew(Label());
	state_container->add_child(state_label);
	state_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	state_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	state_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
	state_label->set_anchors_and_offsets_preset(PRESET_FULL_RECT);

	_update_debugger_buttons();
	_update_window_selector_controls();

	p_debugger->set_game_view(this);
	p_debugger->connect("session_started", callable_mp(this, &GameView::_sessions_changed));
	p_debugger->connect("session_stopped", callable_mp(this, &GameView::_sessions_changed));

	p_wrapper->set_override_close_request(true);
	p_wrapper->connect("window_close_requested", callable_mp(this, &GameView::_window_close_request));
	p_wrapper->connect("window_size_changed", callable_mp(this, &GameView::_update_floating_window_settings));

	EditorDebuggerNode::get_singleton()->connect("breaked", callable_mp(this, &GameView::_debugger_breaked));

	EditorFeatureProfileManager::get_singleton()->connect("current_feature_profile_changed", callable_mp(this, &GameView::_feature_profile_changed));
}

///////

#ifndef ANDROID_ENABLED
void GameViewPlugin::make_visible(bool p_visible) {
	game_view_layout->set_visible(p_visible);
}
#endif

void GameViewPlugin::selected_notify() {
	if (_is_window_wrapper_enabled()) {
#ifdef ANDROID_ENABLED
		notify_main_screen_changed(get_plugin_name());
#else
		top_window_wrapper->grab_window_focus();
		window_wrapper->grab_window_focus();
#endif // ANDROID_ENABLED
		_focus_another_editor();
	}
}

#ifndef ANDROID_ENABLED
void GameViewPlugin::set_window_layout(Ref<ConfigFile> p_layout) {
	game_view->set_window_layout(p_layout);
	top_game_view->set_window_layout(p_layout);
}

void GameViewPlugin::get_window_layout(Ref<ConfigFile> p_layout) {
	game_view->get_window_layout(p_layout);
	top_game_view->get_window_layout(p_layout);
}
#endif // ANDROID_ENABLED

void GameViewPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			add_debugger_plugin(debugger);
			add_debugger_plugin(top_debugger);
			connect("main_screen_changed", callable_mp(this, &GameViewPlugin::_save_last_editor));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			remove_debugger_plugin(debugger);
			remove_debugger_plugin(top_debugger);
			disconnect("main_screen_changed", callable_mp(this, &GameViewPlugin::_save_last_editor));
		} break;
		case NOTIFICATION_READY: {
			EditorRun::instance_starting_callback = _instance_starting_static;
		} break;
	}
}

void GameViewPlugin::_feature_profile_changed() {
	is_feature_enabled = true;
	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();
	if (profile.is_valid()) {
		is_feature_enabled = !profile->is_feature_disabled(EditorFeatureProfile::FEATURE_GAME);
	}

#ifndef ANDROID_ENABLED
	if (game_view) {
		game_view->set_is_feature_enabled(is_feature_enabled);
	}
	if (top_game_view) {
		top_game_view->set_is_feature_enabled(is_feature_enabled);
	}
#endif // ANDROID_ENABLED
}

void GameViewPlugin::_window_visibility_changed(bool p_visible) {
	_focus_another_editor();
}

GameViewPlugin::EmbedAvailability GameViewPlugin::get_embed_available() {
	if (!DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_WINDOW_EMBEDDING)) {
		return EMBED_NOT_AVAILABLE_FEATURE_NOT_SUPPORTED;
	}
	if (is_inside_tree() && get_tree()->get_root()->is_embedding_subwindows()) {
		return EMBED_NOT_AVAILABLE_SINGLE_WINDOW_MODE;
	}
	String display_driver = GLOBAL_GET("display/display_server/driver");
	if (display_driver == "headless" || display_driver == "wayland") {
		return EMBED_NOT_AVAILABLE_PROJECT_DISPLAY_DRIVER;
	}
	if (!EditorNode::get_singleton()->is_multi_window_enabled()) {
		return EMBED_NOT_AVAILABLE_SINGLE_WINDOW_MODE;
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

GameView *GameViewPlugin::get_top_game_view() {
#ifndef ANDROID_ENABLED
	return top_game_view;
#else
	return nullptr;
#endif //ANDROID_ENABLED
}

GameView *GameViewPlugin::get_main_game_view() {
#ifndef ANDROID_ENABLED
	return game_view;
#else
	return nullptr;
#endif //ANDROID_ENABLED
}

void GameViewPlugin::game_view_changed_window_target() {
#ifndef ANDROID_ENABLED
	top_game_view->update_all_ui();
	game_view->update_all_ui();
#endif //ANDROID_ENABLED
}

void GameViewPlugin::enable_dual_game_view(bool p_enabled) {
	dual_view_enabled = p_enabled;
	determine_dual_view_visible();
}

void GameViewPlugin::determine_dual_view_visible() {
#ifndef ANDROID_ENABLED
	top_game_view->set_visible(dual_view_enabled);
	top_window_wrapper->set_visible(dual_view_enabled);
	if (dual_view_enabled) {
		game_view_layout->set_split_offset(0);
		game_view_layout->set_dragging_enabled(true);
		game_view_layout->set_collapsed(false);
	} else {
		game_view_layout->set_dragging_enabled(false);
		game_view_layout->set_collapsed(true);
	}

	top_game_view->update_all_ui();
	game_view->update_all_ui();
#endif //ANDROID_ENABLED
}

void GameViewPlugin::_instance_starting_static(int p_idx, List<String> &r_arguments) {
	ERR_FAIL_NULL(singleton);
	singleton->_instance_starting(p_idx, r_arguments);
}

void GameViewPlugin::_instance_starting(int p_idx, List<String> &r_arguments) {
#ifndef ANDROID_ENABLED
	if (!is_feature_enabled) {
		return;
	}
	// Set the Floating Window default title for both Game Views.
	if (p_idx == 0 && get_embed_available() == GameViewPlugin::EMBED_AVAILABLE) {
		String appname = GLOBAL_GET("application/config/name");
		appname = vformat("%s (DEBUG)", TranslationServer::get_singleton()->translate(appname));

		GameView *main_window_game_view = get_game_view_with_main_screen_embed();

		if (game_view->is_view_embedding() && game_view->get_make_floating_on_play() && !window_wrapper->get_window_enabled()) {
			if (main_window_game_view == game_view) {
				window_wrapper->set_window_title(appname);
			} else {
				window_wrapper->set_window_title(game_view->get_embedded_window_title());
			}
			game_view->show_update_window_wrapper();
		}
		if (top_game_view->is_view_embedding() && top_game_view->get_make_floating_on_play() && !top_window_wrapper->get_window_enabled()) {
			if (main_window_game_view == top_game_view) {
				top_window_wrapper->set_window_title(appname);
			} else {
				top_window_wrapper->set_window_title(top_game_view->get_embedded_window_title());
			}
			top_game_view->show_update_window_wrapper();
		}
	}
	_update_arguments_for_game_instance(p_idx, r_arguments);
#endif //ANDROID_ENABLED
}

void GameViewPlugin::_update_arguments_for_game_instance(int p_idx, List<String> &r_arguments) {
	if (p_idx != 0 || get_embed_available() != EMBED_AVAILABLE) {
		return;
	}
	// Update game instance arguments for main window and custom subwindows.
	GameView *main_window_game_view = get_game_view_with_main_screen_embed();

	List<String>::Element *E = r_arguments.front();
	List<String>::Element *user_args_element = nullptr;

	if (main_window_game_view != nullptr && main_window_game_view->is_view_embedding()) {
		// Remove duplicates/unwanted parameters on main window.
		while (E) {
			List<String>::Element *N = E->next();

			//For these parameters, we need to also renove the value.
			if (E->get() == "--position" || E->get() == "--resolution" || E->get() == "--screen") {
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
		N = r_arguments.insert_after(N, itos(DisplayServer::get_singleton()->window_get_native_handle(DisplayServer::WINDOW_HANDLE, main_window_game_view->get_window()->get_window_id())));

		// Be sure to have the correct window size in the embedded_process control.
		main_window_game_view->update_embed_window_size();

		Rect2i rect = main_window_game_view->get_embedded_process()->get_screen_embedded_window_rect();
		N = r_arguments.insert_after(N, "--position");
		N = r_arguments.insert_after(N, itos(rect.position.x) + "," + itos(rect.position.y));
		N = r_arguments.insert_after(N, "--resolution");
		r_arguments.insert_after(N, itos(rect.size.x) + "x" + itos(rect.size.y));
	}

	if (!get_main_game_view()->get_embedded_window_title().is_empty() && get_main_game_view()->is_view_embedding()) {
		// Add the editor window's native ID as a subwindow parameter.
		List<String>::Element *N = r_arguments.insert_before(user_args_element, "--swid");
		int64_t native_handle = DisplayServer::get_singleton()->window_get_native_handle(DisplayServer::WINDOW_HANDLE, get_main_game_view()->get_window()->get_window_id());
		N = r_arguments.insert_after(N, get_main_game_view()->get_embedded_window_title() + "," + itos(native_handle));
	}
	if (!get_top_game_view()->get_embedded_window_title().is_empty() && get_top_game_view()->is_view_embedding()) {
		// Add the editor window's native ID as a subwindow parameter.
		List<String>::Element *N = r_arguments.insert_before(user_args_element, "--swid");
		int64_t native_handle = DisplayServer::get_singleton()->window_get_native_handle(DisplayServer::WINDOW_HANDLE, get_top_game_view()->get_window()->get_window_id());
		N = r_arguments.insert_after(N, get_top_game_view()->get_embedded_window_title() + "," + itos(native_handle));
	}
}

void GameViewPlugin::_save_last_editor(const String &p_editor) {
	if (p_editor != get_plugin_name()) {
		last_editor = p_editor;
	}
}

void GameViewPlugin::_focus_another_editor() {
	bool change_editor_focus = false;
#ifndef ANDROID_ENABLED
	change_editor_focus = window_wrapper->get_window_enabled();
	if (dual_view_enabled) {
		change_editor_focus = window_wrapper->get_window_enabled() && top_window_wrapper->get_window_enabled();
	}
#else
	change_editor_focus = true;
#endif // ANDROID_ENABLED

	if (change_editor_focus) {
		if (last_editor.is_empty()) {
			EditorNode::get_singleton()->get_editor_main_screen()->select(EditorMainScreen::EDITOR_2D);
		} else {
			EditorInterface::get_singleton()->set_main_screen_editor(last_editor);
		}
	}
}

GameView *GameViewPlugin::get_game_view_with_main_screen_embed() {
#ifndef ANDROID_ENABLED
	if (game_view->get_embedded_window_title().is_empty() && game_view->get_embed_on_play()) {
		// Bottom game_view takes priority.
		return game_view;
	}
	if (top_game_view->get_embedded_window_title().is_empty() && top_game_view->get_embed_on_play()) {
		return top_game_view;
	}
#endif

	return nullptr;
}

GameView *GameViewPlugin::get_game_view_with_selected_window(String p_window_title) {
#ifndef ANDROID_ENABLED
	if (game_view->get_embedded_window_title() == p_window_title && game_view->get_embed_on_play()) {
		// Bottom game_view takes priority.
		return game_view;
	}
	if (top_game_view->get_embedded_window_title() == p_window_title && top_game_view->get_embed_on_play()) {
		return top_game_view;
	}
#endif

	return nullptr;
}

bool GameViewPlugin::_is_window_wrapper_enabled() const {
#ifdef ANDROID_ENABLED
	return true;
#else
	if (dual_view_enabled) {
		return window_wrapper->get_window_enabled() && top_window_wrapper->get_window_enabled();
	}
	return window_wrapper->get_window_enabled();
#endif // ANDROID_ENABLED
}

GameViewPlugin::GameViewPlugin() {
	singleton = this;
	top_debugger.instantiate();
	debugger.instantiate();

#ifndef ANDROID_ENABLED
	game_view_layout = memnew(VSplitContainer);

	top_window_wrapper = memnew(WindowWrapper);
	top_window_wrapper->set_window_title(vformat(TTR("%s - Godot Engine"), TTR("Game Workspace")));
	top_window_wrapper->set_margins_enabled(true);

	top_game_view = memnew(GameView(top_debugger, this, top_window_wrapper, "top_game_view"));
	top_game_view->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	top_game_view->set_time_buttons_enabled(false);
	top_game_view->set_debugger_controls_enabled(false);

	top_window_wrapper->set_wrapped_control(top_game_view, nullptr);

	game_view_layout->add_child(top_window_wrapper);
	top_window_wrapper->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	top_window_wrapper->show();
	top_window_wrapper->connect("window_visibility_changed", callable_mp(this, &GameViewPlugin::_focus_another_editor).unbind(1));

	window_wrapper = memnew(WindowWrapper);
	window_wrapper->set_window_title(vformat(TTR("%s - Godot Engine"), TTR("Game Workspace")));
	window_wrapper->set_margins_enabled(true);

	game_view = memnew(GameView(debugger, this, window_wrapper, "main_game_view"));
	game_view->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	window_wrapper->set_wrapped_control(game_view, nullptr);

	game_view_layout->add_child(window_wrapper);
	window_wrapper->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	window_wrapper->show();
	window_wrapper->connect("window_visibility_changed", callable_mp(this, &GameViewPlugin::_focus_another_editor).unbind(1));

	EditorNode::get_singleton()->get_editor_main_screen()->get_control()->add_child(game_view_layout);
	game_view_layout->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	game_view_layout->hide();

#endif // ANDROID_ENABLED
	EditorFeatureProfileManager::get_singleton()->connect("current_feature_profile_changed", callable_mp(this, &GameViewPlugin::_feature_profile_changed));

#ifndef ANDROID_ENABLED
	bool metadata_dual_view_enabled = (bool)EditorSettings::get_singleton()->get_project_metadata("game_view", "dual_game_view", false);
	enable_dual_game_view(metadata_dual_view_enabled);
#endif // ANDROID_ENABLED
}

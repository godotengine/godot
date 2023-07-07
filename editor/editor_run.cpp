/**************************************************************************/
/*  editor_run.cpp                                                        */
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

#include "editor_run.h"

#include "core/config/project_settings.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "main/main.h"
#include "servers/display_server.h"

/**
 * Separates command line arguments without splitting up quoted strings.
 */
Vector<String> EditorRun::_split_cmdline_args(const String &arg_string) {
	Vector<String> split_args;
	int arg_start = 0;
	bool is_quoted = false;
	char32_t quote_char = '-';
	char32_t arg_char;
	int arg_length;
	for (int i = 0; i < arg_string.length(); i++) {
		arg_char = arg_string[i];
		if (arg_char == '\"' || arg_char == '\'') {
			if (i == 0 || arg_string[i - 1] != '\\') {
				if (is_quoted) {
					if (arg_char == quote_char) {
						is_quoted = false;
						quote_char = '-';
					}
				} else {
					is_quoted = true;
					quote_char = arg_char;
				}
			}
		} else if (!is_quoted && arg_char == ' ') {
			arg_length = i - arg_start;
			if (arg_length > 0) {
				split_args.push_back(arg_string.substr(arg_start, arg_length));
			}
			arg_start = i + 1;
		}
	}
	arg_length = arg_string.length() - arg_start;
	if (arg_length > 0) {
		split_args.push_back(arg_string.substr(arg_start, arg_length));
	}
	return split_args;
}

EditorRun::Status EditorRun::get_status() const {
	return status;
}

String EditorRun::get_running_scene() const {
	return running_scene;
}

Error EditorRun::run(const String &p_scene, const String &p_write_movie) {
	List<String> args;

	for (const String &a : Main::get_forwardable_cli_arguments(Main::CLI_SCOPE_PROJECT)) {
		args.push_back(a);
	}

	String resource_path = ProjectSettings::get_singleton()->get_resource_path();
	if (!resource_path.is_empty()) {
		args.push_back("--path");
		args.push_back(resource_path.replace(" ", "%20"));
	}

	const String debug_uri = EditorDebuggerNode::get_singleton()->get_server_uri();
	if (debug_uri.size()) {
		args.push_back("--remote-debug");
		args.push_back(debug_uri);
	}

	args.push_back("--editor-pid");
	args.push_back(itos(OS::get_singleton()->get_process_id()));

	bool debug_collisions = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_collisions", false);
	bool debug_paths = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_paths", false);
	bool debug_navigation = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_navigation", false);
	bool debug_avoidance = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_avoidance", false);
	bool debug_canvas_redraw = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_canvas_redraw", false);

	if (debug_collisions) {
		args.push_back("--debug-collisions");
	}

	if (debug_paths) {
		args.push_back("--debug-paths");
	}

	if (debug_navigation) {
		args.push_back("--debug-navigation");
	}

	if (debug_avoidance) {
		args.push_back("--debug-avoidance");
	}

	if (debug_canvas_redraw) {
		args.push_back("--debug-canvas-item-redraw");
	}

	if (p_write_movie != "") {
		args.push_back("--write-movie");
		args.push_back(p_write_movie);
		args.push_back("--fixed-fps");
		args.push_back(itos(GLOBAL_GET("editor/movie_writer/fps")));
		if (bool(GLOBAL_GET("editor/movie_writer/disable_vsync"))) {
			args.push_back("--disable-vsync");
		}
	}

	int screen = EDITOR_GET("run/window_placement/screen");
	if (screen == -5) {
		// Same as editor
		screen = DisplayServer::get_singleton()->window_get_current_screen();
	} else if (screen == -4) {
		// Previous monitor (wrap to the other end if needed)
		screen = Math::wrapi(
				DisplayServer::get_singleton()->window_get_current_screen() - 1,
				0,
				DisplayServer::get_singleton()->get_screen_count());
	} else if (screen == -3) {
		// Next monitor (wrap to the other end if needed)
		screen = Math::wrapi(
				DisplayServer::get_singleton()->window_get_current_screen() + 1,
				0,
				DisplayServer::get_singleton()->get_screen_count());
	}

	Rect2 screen_rect = DisplayServer::get_singleton()->screen_get_usable_rect(screen);

	int window_placement = EDITOR_GET("run/window_placement/rect");
	if (screen_rect != Rect2()) {
		Size2 window_size;
		window_size.x = GLOBAL_GET("display/window/size/viewport_width");
		window_size.y = GLOBAL_GET("display/window/size/viewport_height");

		Size2 desired_size;
		desired_size.x = GLOBAL_GET("display/window/size/window_width_override");
		desired_size.y = GLOBAL_GET("display/window/size/window_height_override");
		if (desired_size.x > 0 && desired_size.y > 0) {
			window_size = desired_size;
		}

		if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_HIDPI)) {
			bool hidpi_proj = GLOBAL_GET("display/window/dpi/allow_hidpi");
			int display_scale = 1;

			if (OS::get_singleton()->is_hidpi_allowed()) {
				if (hidpi_proj) {
					display_scale = 1; // Both editor and project runs in hiDPI mode, do not scale.
				} else {
					display_scale = DisplayServer::get_singleton()->screen_get_max_scale(); // Editor is in hiDPI mode, project is not, scale down.
				}
			} else {
				if (hidpi_proj) {
					display_scale = (1.f / DisplayServer::get_singleton()->screen_get_max_scale()); // Editor is not in hiDPI mode, project is, scale up.
				} else {
					display_scale = 1; // Both editor and project runs in lowDPI mode, do not scale.
				}
			}
			screen_rect.position /= display_scale;
			screen_rect.size /= display_scale;
		}

		switch (window_placement) {
			case 0: { // top left
				args.push_back("--position");
				args.push_back(itos(screen_rect.position.x) + "," + itos(screen_rect.position.y));
			} break;
			case 1: { // centered
				Vector2 pos = (screen_rect.position) + ((screen_rect.size - window_size) / 2).floor();
				args.push_back("--position");
				args.push_back(itos(pos.x) + "," + itos(pos.y));
			} break;
			case 2: { // custom pos
				Vector2 pos = EDITOR_GET("run/window_placement/rect_custom_position");
				pos += screen_rect.position;
				args.push_back("--position");
				args.push_back(itos(pos.x) + "," + itos(pos.y));
			} break;
			case 3: { // force maximized
				Vector2 pos = screen_rect.position + screen_rect.size / 2;
				args.push_back("--position");
				args.push_back(itos(pos.x) + "," + itos(pos.y));
				args.push_back("--maximized");
			} break;
			case 4: { // force fullscreen
				Vector2 pos = screen_rect.position + screen_rect.size / 2;
				args.push_back("--position");
				args.push_back(itos(pos.x) + "," + itos(pos.y));
				args.push_back("--fullscreen");
			} break;
		}
	} else {
		// Unable to get screen info, skip setting position.
		switch (window_placement) {
			case 3: { // force maximized
				args.push_back("--maximized");
			} break;
			case 4: { // force fullscreen
				args.push_back("--fullscreen");
			} break;
		}
	}

	List<String> breakpoints;
	EditorNode::get_editor_data().get_editor_breakpoints(&breakpoints);

	if (!breakpoints.is_empty()) {
		args.push_back("--breakpoints");
		String bpoints;
		for (const List<String>::Element *E = breakpoints.front(); E; E = E->next()) {
			bpoints += E->get().replace(" ", "%20");
			if (E->next()) {
				bpoints += ",";
			}
		}

		args.push_back(bpoints);
	}

	if (EditorDebuggerNode::get_singleton()->is_skip_breakpoints()) {
		args.push_back("--skip-breakpoints");
	}

	if (!p_scene.is_empty()) {
		args.push_back(p_scene);
	}

	String exec = OS::get_singleton()->get_executable_path();

	const String raw_custom_args = GLOBAL_GET("editor/run/main_run_args");
	if (!raw_custom_args.is_empty()) {
		// Allow the user to specify a command to run, similar to Steam's launch options.
		// In this case, Godot will no longer be run directly; it's up to the underlying command
		// to run it. For instance, this can be used on Linux to force a running project
		// to use Optimus using `prime-run` or similar.
		// Example: `prime-run %command% --time-scale 0.5`
		const int placeholder_pos = raw_custom_args.find("%command%");

		Vector<String> custom_args;

		if (placeholder_pos != -1) {
			// Prepend executable-specific custom arguments.
			// If nothing is placed before `%command%`, behave as if no placeholder was specified.
			Vector<String> exec_args = _split_cmdline_args(raw_custom_args.substr(0, placeholder_pos));
			if (exec_args.size() >= 1) {
				exec = exec_args[0];
				exec_args.remove_at(0);

				// Append the Godot executable name before we append executable arguments
				// (since the order is reversed when using `push_front()`).
				args.push_front(OS::get_singleton()->get_executable_path());
			}

			for (int i = exec_args.size() - 1; i >= 0; i--) {
				// Iterate backwards as we're pushing items in the reverse order.
				args.push_front(exec_args[i].replace(" ", "%20"));
			}

			// Append Godot-specific custom arguments.
			custom_args = _split_cmdline_args(raw_custom_args.substr(placeholder_pos + String("%command%").size()));
			for (int i = 0; i < custom_args.size(); i++) {
				args.push_back(custom_args[i].replace(" ", "%20"));
			}
		} else {
			// Append Godot-specific custom arguments.
			custom_args = _split_cmdline_args(raw_custom_args);
			for (int i = 0; i < custom_args.size(); i++) {
				args.push_back(custom_args[i].replace(" ", "%20"));
			}
		}
	}

	// Pass the debugger stop shortcut to the running instance(s).
	String shortcut;
	VariantWriter::write_to_string(ED_GET_SHORTCUT("editor/stop_running_project"), shortcut);
	OS::get_singleton()->set_environment("__GODOT_EDITOR_STOP_SHORTCUT__", shortcut);

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line(vformat("Running: %s", exec));
		for (const String &E : args) {
			print_line(vformat(" %s", E));
		}
	}

	int instances = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_instances", 1);
	for (int i = 0; i < instances; i++) {
		OS::ProcessID pid = 0;
		Error err = OS::get_singleton()->create_instance(args, &pid);
		ERR_FAIL_COND_V(err, err);
		if (pid != 0) {
			pids.push_back(pid);
		}
	}

	status = STATUS_PLAY;
	if (!p_scene.is_empty()) {
		running_scene = p_scene;
	}

	return OK;
}

bool EditorRun::has_child_process(OS::ProcessID p_pid) const {
	for (const OS::ProcessID &E : pids) {
		if (E == p_pid) {
			return true;
		}
	}
	return false;
}

void EditorRun::stop_child_process(OS::ProcessID p_pid) {
	if (has_child_process(p_pid)) {
		OS::get_singleton()->kill(p_pid);
		pids.erase(p_pid);
	}
}

void EditorRun::stop() {
	if (status != STATUS_STOP && pids.size() > 0) {
		for (const OS::ProcessID &E : pids) {
			OS::get_singleton()->kill(E);
		}
		pids.clear();
	}

	status = STATUS_STOP;
	running_scene = "";
}

EditorRun::EditorRun() {
	status = STATUS_STOP;
	running_scene = "";
}

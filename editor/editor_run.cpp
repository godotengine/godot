/*************************************************************************/
/*  editor_run.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_run.h"

#include "editor_settings.h"
#include "project_settings.h"

EditorRun::Status EditorRun::get_status() const {

	return status;
}
Error EditorRun::run(const String &p_scene, const String p_custom_args, const List<String> &p_breakpoints) {

	List<String> args;

	String resource_path = ProjectSettings::get_singleton()->get_resource_path();
	String remote_host = EditorSettings::get_singleton()->get("network/debug/remote_host");
	int remote_port = (int)EditorSettings::get_singleton()->get("network/debug/remote_port");

	if (resource_path != "") {
		args.push_back("--path");
		args.push_back(resource_path.replace(" ", "%20"));
	}

	if (true) {
		args.push_back("--remote-debug");
		args.push_back(remote_host + ":" + String::num(remote_port));
	}

	args.push_back("--allow_focus_steal_pid");
	args.push_back(itos(OS::get_singleton()->get_process_id()));

	if (debug_collisions) {
		args.push_back("--debug-collisions");
	}

	if (debug_navigation) {
		args.push_back("--debug-navigation");
	}

	int screen = EditorSettings::get_singleton()->get("run/window_placement/screen");
	if (screen == 0) {
		screen = OS::get_singleton()->get_current_screen();
	} else {
		screen--;
	}

	if (OS::get_singleton()->is_disable_crash_handler()) {
		args.push_back("--disable-crash-handler");
	}

	Rect2 screen_rect;
	screen_rect.position = OS::get_singleton()->get_screen_position(screen);
	screen_rect.size = OS::get_singleton()->get_screen_size(screen);

	Size2 desired_size;
	desired_size.x = ProjectSettings::get_singleton()->get("display/window/size/width");
	desired_size.y = ProjectSettings::get_singleton()->get("display/window/size/height");

	Size2 test_size;
	test_size.x = ProjectSettings::get_singleton()->get("display/window/size/test_width");
	test_size.y = ProjectSettings::get_singleton()->get("display/window/size/test_height");
	if (test_size.x > 0 && test_size.y > 0) {

		desired_size = test_size;
	}

	int window_placement = EditorSettings::get_singleton()->get("run/window_placement/rect");

	switch (window_placement) {
		case 0: { // top left

			args.push_back("--position");
			args.push_back(itos(screen_rect.position.x) + "," + itos(screen_rect.position.y));
		} break;
		case 1: { // centered
			int display_scale = 1;
#ifdef OSX_ENABLED
			if (OS::get_singleton()->get_screen_dpi(screen) >= 192 && OS::get_singleton()->get_screen_size(screen).x > 2000) {
				display_scale = 2;
			}
#endif

			Vector2 pos = screen_rect.position + ((screen_rect.size / display_scale - desired_size) / 2).floor();
			args.push_back("--position");
			args.push_back(itos(pos.x) + "," + itos(pos.y));
		} break;
		case 2: { // custom pos
			Vector2 pos = EditorSettings::get_singleton()->get("run/window_placement/rect_custom_position");
			pos += screen_rect.position;
			args.push_back("--position");
			args.push_back(itos(pos.x) + "," + itos(pos.y));
		} break;
		case 3: { // force maximized
			Vector2 pos = screen_rect.position;
			args.push_back("--position");
			args.push_back(itos(pos.x) + "," + itos(pos.y));
			args.push_back("--maximized");

		} break;
		case 4: { // force fullscreen

			Vector2 pos = screen_rect.position;
			args.push_back("--position");
			args.push_back(itos(pos.x) + "," + itos(pos.y));
			args.push_back("--fullscreen");
		} break;
	}

	if (p_breakpoints.size()) {

		args.push_back("--breakpoints");
		String bpoints;
		for (const List<String>::Element *E = p_breakpoints.front(); E; E = E->next()) {

			bpoints += E->get().replace(" ", "%20");
			if (E->next())
				bpoints += ",";
		}

		args.push_back(bpoints);
	}

	if (p_scene != "") {
		args.push_back(p_scene);
	}

	if (p_custom_args != "") {
		Vector<String> cargs = p_custom_args.split(" ", false);
		for (int i = 0; i < cargs.size(); i++) {
			args.push_back(cargs[i].replace(" ", "%20"));
		}
	}

	String exec = OS::get_singleton()->get_executable_path();

	printf("Running: %ls", exec.c_str());
	for (List<String>::Element *E = args.front(); E; E = E->next()) {

		printf(" %ls", E->get().c_str());
	};
	printf("\n");

	pid = 0;
	Error err = OS::get_singleton()->execute(exec, args, false, &pid);
	ERR_FAIL_COND_V(err, err);

	status = STATUS_PLAY;

	return OK;
}

void EditorRun::stop() {

	if (status != STATUS_STOP && pid != 0) {

		OS::get_singleton()->kill(pid);
	}

	status = STATUS_STOP;
}

void EditorRun::set_debug_collisions(bool p_debug) {

	debug_collisions = p_debug;
}

bool EditorRun::get_debug_collisions() const {

	return debug_collisions;
}

void EditorRun::set_debug_navigation(bool p_debug) {

	debug_navigation = p_debug;
}

bool EditorRun::get_debug_navigation() const {

	return debug_navigation;
}

EditorRun::EditorRun() {

	status = STATUS_STOP;
	debug_collisions = false;
	debug_navigation = false;
}

/*************************************************************************/
/*  os_server.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "os_server.h"

#include "core/print_string.h"
#include "drivers/dummy/rasterizer_dummy.h"
#include "drivers/dummy/texture_loader_dummy.h"
#include "servers/rendering/rendering_server_raster.h"

#include "main/main.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int OS_Server::get_video_driver_count() const {
	return 1;
}

const char *OS_Server::get_video_driver_name(int p_driver) const {
	return "Dummy";
}

int OS_Server::get_audio_driver_count() const {
	return 1;
}

const char *OS_Server::get_audio_driver_name(int p_driver) const {
	return "Dummy";
}

int OS_Server::get_current_video_driver() const {
	return video_driver_index;
}

void OS_Server::initialize_core() {
	crash_handler.initialize();

	OS_Unix::initialize_core();
}

Error OS_Server::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {
	args = OS::get_singleton()->get_cmdline_args();
	current_videomode = p_desired;
	main_loop = nullptr;

	RasterizerDummy::make_current();

	video_driver_index = p_video_driver; // unused in server platform, but should still be initialized

	rendering_server = memnew(RenderingServerRaster);
	rendering_server->init();

	AudioDriverManager::initialize(p_audio_driver);

	input = memnew(InputDefault);

	_ensure_user_data_dir();

	resource_loader_dummy.instance();
	ResourceLoader::add_resource_format_loader(resource_loader_dummy);

	return OK;
}

void OS_Server::finalize() {
	if (main_loop)
		memdelete(main_loop);
	main_loop = nullptr;

	rendering_server->finish();
	memdelete(rendering_server);

	memdelete(input);

	ResourceLoader::remove_resource_format_loader(resource_loader_dummy);
	resource_loader_dummy.unref();

	args.clear();
}

void OS_Server::set_mouse_show(bool p_show) {
}

void OS_Server::set_mouse_grab(bool p_grab) {
	grab = p_grab;
}

bool OS_Server::is_mouse_grab_enabled() const {
	return grab;
}

int OS_Server::get_mouse_button_state() const {
	return 0;
}

Point2 OS_Server::get_mouse_position() const {
	return Point2();
}

void OS_Server::set_window_title(const String &p_title) {
}

void OS_Server::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
}

OS::VideoMode OS_Server::get_video_mode(int p_screen) const {
	return current_videomode;
}

Size2 OS_Server::get_window_size() const {
	return Vector2(current_videomode.width, current_videomode.height);
}

void OS_Server::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {
}

MainLoop *OS_Server::get_main_loop() const {
	return main_loop;
}

void OS_Server::delete_main_loop() {
	if (main_loop)
		memdelete(main_loop);
	main_loop = nullptr;
}

void OS_Server::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
}

bool OS_Server::can_draw() const {
	return false; //can never draw
};

String OS_Server::get_name() const {
	return "Server";
}

void OS_Server::move_window_to_foreground() {
}

bool OS_Server::_check_internal_feature_support(const String &p_feature) {
	return p_feature == "pc";
}

void OS_Server::run() {
	force_quit = false;

	if (!main_loop)
		return;

	main_loop->init();

	while (!force_quit) {
		if (Main::iteration())
			break;
	};

	main_loop->finish();
}

String OS_Server::get_config_path() const {
	if (has_environment("XDG_CONFIG_HOME")) {
		return get_environment("XDG_CONFIG_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".config");
	} else {
		return ".";
	}
}

String OS_Server::get_data_path() const {
	if (has_environment("XDG_DATA_HOME")) {
		return get_environment("XDG_DATA_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".local/share");
	} else {
		return get_config_path();
	}
}

String OS_Server::get_cache_path() const {
	if (has_environment("XDG_CACHE_HOME")) {
		return get_environment("XDG_CACHE_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".cache");
	} else {
		return get_config_path();
	}
}

String OS_Server::get_system_dir(SystemDir p_dir) const {
	String xdgparam;

	switch (p_dir) {
		case SYSTEM_DIR_DESKTOP: {
			xdgparam = "DESKTOP";
		} break;
		case SYSTEM_DIR_DCIM: {
			xdgparam = "PICTURES";

		} break;
		case SYSTEM_DIR_DOCUMENTS: {
			xdgparam = "DOCUMENTS";

		} break;
		case SYSTEM_DIR_DOWNLOADS: {
			xdgparam = "DOWNLOAD";

		} break;
		case SYSTEM_DIR_MOVIES: {
			xdgparam = "VIDEOS";

		} break;
		case SYSTEM_DIR_MUSIC: {
			xdgparam = "MUSIC";

		} break;
		case SYSTEM_DIR_PICTURES: {
			xdgparam = "PICTURES";

		} break;
		case SYSTEM_DIR_RINGTONES: {
			xdgparam = "MUSIC";

		} break;
	}

	String pipe;
	List<String> arg;
	arg.push_back(xdgparam);
	Error err = const_cast<OS_Server *>(this)->execute("xdg-user-dir", arg, true, nullptr, &pipe);
	if (err != OK)
		return ".";
	return pipe.strip_edges();
}

void OS_Server::disable_crash_handler() {
	crash_handler.disable();
}

bool OS_Server::is_disable_crash_handler() const {
	return crash_handler.is_disabled();
}

OS_Server::OS_Server() {
	//adriver here
	grab = false;
};

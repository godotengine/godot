/*************************************************************************/
/*  os_linux.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/os/displaydriver.h"
#include "os_freedesktop.h"

#include "core/print_string.h"
#include "errno.h"
#include "key_mapping_x11.h"

#ifdef HAVE_MNTENT
#include <mntent.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "main/main.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

//stupid linux.h
#ifdef KEY_TAB
#undef KEY_TAB
#endif

int OS_Linux::get_audio_driver_count() const {
	return AudioDriverManager::get_driver_count();
}

const char *OS_Linux::get_audio_driver_name(int p_driver) const {

	AudioDriver *driver = AudioDriverManager::get_driver(p_driver);
	ERR_FAIL_COND_V(!driver, "");
	return AudioDriverManager::get_driver(p_driver)->get_name();
}

void OS_Linux::initialize_core() {

	crash_handler.initialize();

	OS_Unix::initialize_core();
}

Error OS_Linux::initialize_os(int p_audio_driver) {
	args = OS::get_singleton()->get_cmdline_args();
	last_timestamp = 0;

	AudioDriverManager::initialize(p_audio_driver);

	_ensure_user_data_dir();

	power_manager = memnew(PowerX11);

	return OK;
}

String OS_Linux::get_unique_id() const {

	static String machine_id;
	if (machine_id.empty()) {
		if (FileAccess *f = FileAccess::open("/etc/machine-id", FileAccess::READ)) {
			while (machine_id.empty() && !f->eof_reached()) {
				machine_id = f->get_line().strip_edges();
			}
			f->close();
			memdelete(f);
		}
	}
	return machine_id;
}

void OS_Linux::finalize_os() {

	args.clear();
}

String OS_Linux::get_name() {

	return "X11";
}

Error OS_Linux::shell_open(String p_uri) {

	Error ok;
	List<String> args;
	args.push_back(p_uri);
	ok = execute("xdg-open", args, false);
	if (ok == OK)
		return OK;
	ok = execute("gnome-open", args, false);
	if (ok == OK)
		return OK;
	ok = execute("kde-open", args, false);
	return ok;
}

bool OS_Linux::_check_internal_feature_support(const String &p_feature) {

	return p_feature == "pc" || p_feature == "s3tc" || p_feature == "bptc";
}

String OS_Linux::get_config_path() const {

	if (has_environment("XDG_CONFIG_HOME")) {
		return get_environment("XDG_CONFIG_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".config");
	} else {
		return ".";
	}
}

String OS_Linux::get_data_path() const {

	if (has_environment("XDG_DATA_HOME")) {
		return get_environment("XDG_DATA_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".local/share");
	} else {
		return get_config_path();
	}
}

String OS_Linux::get_cache_path() const {

	if (has_environment("XDG_CACHE_HOME")) {
		return get_environment("XDG_CACHE_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".cache");
	} else {
		return get_config_path();
	}
}

String OS_Linux::get_system_dir(SystemDir p_dir) const {

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
	Error err = const_cast<OS_Linux *>(this)->execute("xdg-user-dir", arg, true, NULL, &pipe);
	if (err != OK)
		return ".";
	return pipe.strip_edges();
}

void OS_Linux::alert(const String &p_alert, const String &p_title) {

	List<String> args;
	args.push_back("-center");
	args.push_back("-title");
	args.push_back(p_title);
	args.push_back(p_alert);

	execute("xmessage", args, true);
}

void OS_Linux::run() {

	force_quit = false;

	MainLoop *main_loop = get_main_loop();

	if (!main_loop)
		return;

	main_loop->init();

	//uint64_t last_ticks=get_ticks_usec();

	//int frames=0;
	//uint64_t frame=0;

	while (!force_quit) {

		DisplayDriver::get_singleton()->process_events();
		// #ifdef JOYDEV_ENABLED
		//		joypad->process_joypads();
		// #endif
		if (Main::iteration())
			break;
	};

	main_loop->finish();
}

OS::PowerState OS_Linux::get_power_state() {
	return power_manager->get_power_state();
}

int OS_Linux::get_power_seconds_left() {
	return power_manager->get_power_seconds_left();
}

int OS_Linux::get_power_percent_left() {
	return power_manager->get_power_percent_left();
}

void OS_Linux::disable_crash_handler() {
	crash_handler.disable();
}

bool OS_Linux::is_disable_crash_handler() const {
	return crash_handler.is_disabled();
}

static String get_mountpoint(const String &p_path) {
	struct stat s;
	if (stat(p_path.utf8().get_data(), &s)) {
		return "";
	}

#ifdef HAVE_MNTENT
	dev_t dev = s.st_dev;
	FILE *fd = setmntent("/proc/mounts", "r");
	if (!fd) {
		return "";
	}

	struct mntent mnt;
	char buf[1024];
	size_t buflen = 1024;
	while (getmntent_r(fd, &mnt, buf, buflen)) {
		if (!stat(mnt.mnt_dir, &s) && s.st_dev == dev) {
			endmntent(fd);
			return String(mnt.mnt_dir);
		}
	}

	endmntent(fd);
#endif
	return "";
}

Error OS_Linux::move_to_trash(const String &p_path) {
	String trashcan = "";
	String mnt = get_mountpoint(p_path);

	if (mnt != "") {
		String path(mnt + "/.Trash-" + itos(getuid()) + "/files");
		struct stat s;
		if (!stat(path.utf8().get_data(), &s)) {
			trashcan = path;
		}
	}

	if (trashcan == "") {
		char *dhome = getenv("XDG_DATA_HOME");
		if (dhome) {
			trashcan = String(dhome) + "/Trash/files";
		}
	}

	if (trashcan == "") {
		char *home = getenv("HOME");
		if (home) {
			trashcan = String(home) + "/.local/share/Trash/files";
		}
	}

	if (trashcan == "") {
		ERR_PRINTS("move_to_trash: Could not determine trashcan location");
		return FAILED;
	}

	List<String> args;
	args.push_back("-p");
	args.push_back(trashcan);
	Error err = execute("mkdir", args, true);
	if (err == OK) {
		List<String> args2;
		args2.push_back(p_path);
		args2.push_back(trashcan);
		err = execute("mv", args2, true);
	}

	return err;
}

OS_Linux::OS_Linux() {

#ifdef PULSEAUDIO_ENABLED
	AudioDriverManager::add_driver(&driver_pulseaudio);
#endif

#ifdef ALSA_ENABLED
	AudioDriverManager::add_driver(&driver_alsa);
#endif

	minimized = false;
}
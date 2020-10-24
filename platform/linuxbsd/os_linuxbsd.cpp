/*************************************************************************/
/*  os_linuxbsd.cpp                                                      */
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

#include "os_linuxbsd.h"

#include "core/os/dir_access.h"
#include "main/main.h"

#ifdef X11_ENABLED
#include "display_server_x11.h"
#endif

#ifdef HAVE_MNTENT
#include <mntent.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

void OS_LinuxBSD::initialize() {
	crash_handler.initialize();

	OS_Unix::initialize_core();
}

void OS_LinuxBSD::initialize_joypads() {
#ifdef JOYDEV_ENABLED
	joypad = memnew(JoypadLinux(Input::get_singleton()));
#endif
}

String OS_LinuxBSD::get_unique_id() const {
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

void OS_LinuxBSD::finalize() {
	if (main_loop) {
		memdelete(main_loop);
	}
	main_loop = nullptr;

#ifdef ALSAMIDI_ENABLED
	driver_alsamidi.close();
#endif

#ifdef JOYDEV_ENABLED
	if (joypad) {
		memdelete(joypad);
	}
#endif
}

MainLoop *OS_LinuxBSD::get_main_loop() const {
	return main_loop;
}

void OS_LinuxBSD::delete_main_loop() {
	if (main_loop) {
		memdelete(main_loop);
	}
	main_loop = nullptr;
}

void OS_LinuxBSD::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

String OS_LinuxBSD::get_name() const {
#ifdef __linux__
	return "Linux";
#elif defined(__FreeBSD__)
	return "FreeBSD";
#elif defined(__NetBSD__)
	return "NetBSD";
#else
	return "BSD";
#endif
}

Error OS_LinuxBSD::shell_open(String p_uri) {
	Error ok;
	int err_code;
	List<String> args;
	args.push_back(p_uri);

	// Agnostic
	ok = execute("xdg-open", args, true, nullptr, nullptr, &err_code);
	if (ok == OK && !err_code) {
		return OK;
	} else if (err_code == 2) {
		return ERR_FILE_NOT_FOUND;
	}
	// GNOME
	args.push_front("open"); // The command is `gio open`, so we need to add it to args
	ok = execute("gio", args, true, nullptr, nullptr, &err_code);
	if (ok == OK && !err_code) {
		return OK;
	} else if (err_code == 2) {
		return ERR_FILE_NOT_FOUND;
	}
	args.pop_front();
	ok = execute("gvfs-open", args, true, nullptr, nullptr, &err_code);
	if (ok == OK && !err_code) {
		return OK;
	} else if (err_code == 2) {
		return ERR_FILE_NOT_FOUND;
	}
	// KDE
	ok = execute("kde-open5", args, true, nullptr, nullptr, &err_code);
	if (ok == OK && !err_code) {
		return OK;
	}
	ok = execute("kde-open", args, true, nullptr, nullptr, &err_code);
	return !err_code ? ok : FAILED;
}

bool OS_LinuxBSD::_check_internal_feature_support(const String &p_feature) {
	return p_feature == "pc";
}

String OS_LinuxBSD::get_config_path() const {
	if (has_environment("XDG_CONFIG_HOME")) {
		return get_environment("XDG_CONFIG_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".config");
	} else {
		return ".";
	}
}

String OS_LinuxBSD::get_data_path() const {
	if (has_environment("XDG_DATA_HOME")) {
		return get_environment("XDG_DATA_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".local/share");
	} else {
		return get_config_path();
	}
}

String OS_LinuxBSD::get_cache_path() const {
	if (has_environment("XDG_CACHE_HOME")) {
		return get_environment("XDG_CACHE_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".cache");
	} else {
		return get_config_path();
	}
}

String OS_LinuxBSD::get_system_dir(SystemDir p_dir) const {
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
	Error err = const_cast<OS_LinuxBSD *>(this)->execute("xdg-user-dir", arg, true, nullptr, &pipe);
	if (err != OK) {
		return ".";
	}
	return pipe.strip_edges();
}

void OS_LinuxBSD::run() {
	force_quit = false;

	if (!main_loop) {
		return;
	}

	main_loop->init();

	//uint64_t last_ticks=get_ticks_usec();

	//int frames=0;
	//uint64_t frame=0;

	while (!force_quit) {
		DisplayServer::get_singleton()->process_events(); // get rid of pending events
#ifdef JOYDEV_ENABLED
		joypad->process_joypads();
#endif
		if (Main::iteration()) {
			break;
		}
	};

	main_loop->finish();
}

void OS_LinuxBSD::disable_crash_handler() {
	crash_handler.disable();
}

bool OS_LinuxBSD::is_disable_crash_handler() const {
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

Error OS_LinuxBSD::move_to_trash(const String &p_path) {
	String trash_can = "";
	String mnt = get_mountpoint(p_path);

	// If there is a directory "[Mountpoint]/.Trash-[UID]/files", use it as the trash can.
	if (mnt != "") {
		String path(mnt + "/.Trash-" + itos(getuid()) + "/files");
		struct stat s;
		if (!stat(path.utf8().get_data(), &s)) {
			trash_can = path;
		}
	}

	// Otherwise, if ${XDG_DATA_HOME} is defined, use "${XDG_DATA_HOME}/Trash/files" as the trash can.
	if (trash_can == "") {
		char *dhome = getenv("XDG_DATA_HOME");
		if (dhome) {
			trash_can = String(dhome) + "/Trash/files";
		}
	}

	// Otherwise, if ${HOME} is defined, use "${HOME}/.local/share/Trash/files" as the trash can.
	if (trash_can == "") {
		char *home = getenv("HOME");
		if (home) {
			trash_can = String(home) + "/.local/share/Trash/files";
		}
	}

	// Issue an error if none of the previous locations is appropriate for the trash can.
	if (trash_can == "") {
		ERR_PRINT("move_to_trash: Could not determine the trash can location");
		return FAILED;
	}

	// Create needed directories for decided trash can location.
	DirAccess *dir_access = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = dir_access->make_dir_recursive(trash_can);
	memdelete(dir_access);

	// Issue an error if trash can is not created proprely.
	if (err != OK) {
		ERR_PRINT("move_to_trash: Could not create the trash can \"" + trash_can + "\"");
		return err;
	}

	// The trash can is successfully created, now move the given resource to it.
	// Do not use DirAccess:rename() because it can't move files across multiple mountpoints.
	List<String> mv_args;
	mv_args.push_back(p_path);
	mv_args.push_back(trash_can);
	int retval;
	err = execute("mv", mv_args, true, nullptr, nullptr, &retval);

	// Issue an error if "mv" failed to move the given resource to the trash can.
	if (err != OK || retval != 0) {
		ERR_PRINT("move_to_trash: Could not move the resource \"" + p_path + "\" to the trash can \"" + trash_can + "\"");
		return FAILED;
	}

	return OK;
}

OS_LinuxBSD::OS_LinuxBSD() {
	main_loop = nullptr;
	force_quit = false;

#ifdef PULSEAUDIO_ENABLED
	AudioDriverManager::add_driver(&driver_pulseaudio);
#endif

#ifdef ALSA_ENABLED
	AudioDriverManager::add_driver(&driver_alsa);
#endif

#ifdef X11_ENABLED
	DisplayServerX11::register_x11_driver();
#endif
}

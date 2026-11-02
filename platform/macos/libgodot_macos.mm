/**************************************************************************/
/*  libgodot_macos.mm                                                     */
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

#include "os_macos.h"

#include "core/core_globals.h"
#include "core/extension/godot_instance.h"
#include "core/extension/libgodot.h"
#include "main/main.h"

static OS_MacOS *os = nullptr;

static GodotInstance *instance = nullptr;

GDExtensionObjectPtr libgodot_create_godot_instance(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func) {
	ERR_FAIL_COND_V_MSG(instance != nullptr, nullptr, "Only one Godot Instance may be created.");

	CoreGlobals::global_init_func_libgodot = p_init_func;
	CoreGlobals::engine_reinit_enabled = true;
	CoreGlobals::libgodot_host_instance = true;

	// 2dog: avoid AppKit when the embedded host requests a headless display driver.
	bool is_headless = false;
	for (int i = 1; i < p_argc; i++) {
		for (size_t j = 0; j < std::size(OS_MacOS::headless_args); j++) {
			if (strcmp(OS_MacOS::headless_args[j], p_argv[i]) == 0) {
				is_headless = true;
				break;
			}
		}
		if (i < p_argc - 1 && strcmp("--display-driver", p_argv[i]) == 0 && strcmp("headless", p_argv[i + 1]) == 0) {
			is_headless = true;
		}
		if (is_headless) {
			break;
		}
	}

	uint32_t remaining_args = p_argc - 1;
	if (is_headless) {
		os = new OS_MacOS_Headless(p_argv[0], remaining_args, remaining_args > 0 ? &p_argv[1] : nullptr);
	} else {
		os = new OS_MacOS_NSApp(p_argv[0], remaining_args, remaining_args > 0 ? &p_argv[1] : nullptr);
	}

	@autoreleasepool {
		Error err = Main::setup(p_argv[0], remaining_args, remaining_args > 0 ? &p_argv[1] : nullptr, false);
		if (err != OK) {
			delete os;
			os = nullptr;
			return nullptr;
		}

		instance = memnew(GodotInstance);
		if (!instance->initialize()) {
			memdelete(instance);
			instance = nullptr;
			Main::cleanup();
			delete os;
			os = nullptr;
			return nullptr;
		}

		return (GDExtensionObjectPtr)instance;
	}
}

void libgodot_destroy_godot_instance(GDExtensionObjectPtr p_godot_instance) {
	GodotInstance *godot_instance = (GodotInstance *)p_godot_instance;
	if (instance == godot_instance) {
		godot_instance->stop();
		memdelete(godot_instance);
		instance = nullptr;
		Main::cleanup();
		delete os;
		os = nullptr;
		CoreGlobals::global_init_func_libgodot = nullptr;
		CoreGlobals::libgodot_host_instance = false;
	}
}

int libgodot_import_project(const char *p_project_path, int p_extra_argc, const char *p_extra_argv[]) {
#ifndef TOOLS_ENABLED
	return -1;
#else
	ERR_FAIL_NULL_V(p_project_path, EXIT_FAILURE);
	ERR_FAIL_COND_V_MSG(instance != nullptr || os != nullptr, EXIT_FAILURE,
			"libgodot_import_project cannot run while a Godot instance exists in this process.");

	// 2dog: keep ClassDB valid for later engine instances; never load a previous host's extension.
	CoreGlobals::global_init_func_libgodot = nullptr;
	CoreGlobals::engine_reinit_enabled = true;
	CoreGlobals::libgodot_host_instance = false;

	Vector<char *> argv;
	argv.push_back(const_cast<char *>("--headless"));
	argv.push_back(const_cast<char *>("--import"));
	argv.push_back(const_cast<char *>("--path"));
	argv.push_back(const_cast<char *>(p_project_path));
	for (int i = 0; i < p_extra_argc; i++) {
		argv.push_back(const_cast<char *>(p_extra_argv[i]));
	}

	// 2dog: import through the AppKit-free headless OS.
	OS_MacOS_Headless import_os("libgodot", argv.size(), argv.ptrw());

	// 2dog: OS_MacOS_Headless::run owns setup and cleanup.
	import_os.run();
	return import_os.get_exit_code();
#endif
}

int libgodot_export_pack(const char *p_project_path, const char *p_preset, const char *p_output_path, int p_extra_argc, const char *p_extra_argv[]) {
#ifndef TOOLS_ENABLED
	return -1;
#else
	ERR_FAIL_NULL_V(p_project_path, EXIT_FAILURE);
	ERR_FAIL_NULL_V(p_preset, EXIT_FAILURE);
	ERR_FAIL_NULL_V(p_output_path, EXIT_FAILURE);
	ERR_FAIL_COND_V_MSG(instance != nullptr || os != nullptr, EXIT_FAILURE,
			"libgodot_export_pack cannot run while a Godot instance exists in this process.");

	// 2dog: keep ClassDB valid for later engine instances; never load a previous host's extension.
	CoreGlobals::global_init_func_libgodot = nullptr;
	CoreGlobals::engine_reinit_enabled = true;
	CoreGlobals::libgodot_host_instance = false;

	Vector<char *> argv;
	// 2dog: an explicit --rendering-driver in the extra arguments keeps a window-less rendering device for the shader baker.
	argv.push_back(const_cast<char *>("--headless"));
	argv.push_back(const_cast<char *>("--export-pack"));
	argv.push_back(const_cast<char *>(p_preset));
	argv.push_back(const_cast<char *>(p_output_path));
	argv.push_back(const_cast<char *>("--path"));
	argv.push_back(const_cast<char *>(p_project_path));
	for (int i = 0; i < p_extra_argc; i++) {
		argv.push_back(const_cast<char *>(p_extra_argv[i]));
	}
	// 2dog: marks a host pack export; see EditorExportPlatform::export_pack and GodotTools' ExportPlugin.
	argv.push_back(const_cast<char *>("++"));
	argv.push_back(const_cast<char *>("--2dog-host-pack"));

	// 2dog: OS_MacOS_Headless::run owns setup and cleanup.
	OS_MacOS_Headless export_os("libgodot", argv.size(), argv.ptrw());

	export_os.run();
	return export_os.get_exit_code();
#endif
}

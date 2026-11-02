/**************************************************************************/
/*  libgodot_windows.cpp                                                  */
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

#include "os_windows.h"

#include "core/core_globals.h"
#include "core/extension/godot_instance.h"
#include "core/extension/libgodot.h"
#include "main/main.h"

static OS_Windows *os = nullptr;

static GodotInstance *instance = nullptr;

GDExtensionObjectPtr libgodot_create_godot_instance(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func) {
	ERR_FAIL_COND_V_MSG(instance != nullptr, nullptr, "Only one Godot Instance may be created at a time.");

	CoreGlobals::global_init_func_libgodot = p_init_func;
	CoreGlobals::engine_reinit_enabled = true;
	CoreGlobals::libgodot_host_instance = true;

	os = new OS_Windows(GetModuleHandle(nullptr));

	Error err = Main::setup(p_argv[0], p_argc - 1, &p_argv[1], false);
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

	OS_Windows import_os(GetModuleHandle(nullptr));

	Vector<char *> argv;
	argv.push_back(const_cast<char *>("--headless"));
	argv.push_back(const_cast<char *>("--import"));
	argv.push_back(const_cast<char *>("--path"));
	argv.push_back(const_cast<char *>(p_project_path));
	for (int i = 0; i < p_extra_argc; i++) {
		argv.push_back(const_cast<char *>(p_extra_argv[i]));
	}

	// 2dog: run the headless editor lifecycle in-process.
	Error err = Main::setup("libgodot", argv.size(), argv.ptrw());
	if (err != OK) {
		return (err == ERR_HELP) ? EXIT_SUCCESS : EXIT_FAILURE;
	}

	if (Main::start() == EXIT_SUCCESS) {
		import_os.run();
	} else {
		import_os.set_exit_code(EXIT_FAILURE);
	}
	Main::cleanup();

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

	OS_Windows export_os(GetModuleHandle(nullptr));

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

	// 2dog: run the editor lifecycle in-process.
	Error err = Main::setup("libgodot", argv.size(), argv.ptrw());
	if (err != OK) {
		return (err == ERR_HELP) ? EXIT_SUCCESS : EXIT_FAILURE;
	}

	if (Main::start() == EXIT_SUCCESS) {
		export_os.run();
	} else {
		export_os.set_exit_code(EXIT_FAILURE);
	}
	Main::cleanup();

	return export_os.get_exit_code();
#endif
}

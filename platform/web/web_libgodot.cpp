/**************************************************************************/
/*  web_libgodot.cpp                                                      */
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

#include "godot_js.h"
#include "os_web.h"

#include "core/extension/godot_instance.h"
#include "core/extension/libgodot.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/variant/variant.h"
#include "main/main.h"

#include <cstdlib>

[[maybe_unused]] static OS_Web *os = nullptr;

static GodotInstance *instance = nullptr;

void print_web_header() {
	// Emscripten.
	char *emscripten_version_char = godot_js_emscripten_get_version();
	String emscripten_version = vformat("Emscripten %s", emscripten_version_char);
	// `free()` is used here because it's not memory that was allocated by Godot.
	free(emscripten_version_char);

	// Build features.
	String thread_support = OS::get_singleton()->has_feature("threads")
			? "multi-threaded"
			: "single-threaded";
	String extensions_support = OS::get_singleton()->has_feature("web_extensions")
			? "GDExtension support"
			: "no GDExtension support";

	Vector<String> build_configuration = { emscripten_version, thread_support, extensions_support };
	print_line(vformat("Build configuration: %s.", String(", ").join(build_configuration)));
}

GDExtensionObjectPtr libgodot_create_godot_instance(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func) {
	ERR_FAIL_COND_V_MSG(instance != nullptr, nullptr, "Only one Godot Instance may be created.");

	os = new OS_Web();

	Error err = Main::setup(p_argv[0], p_argc - 1, &p_argv[1], false);
	if (err != OK) {
		return nullptr;
	}

	instance = memnew(GodotInstance);
	if (!instance->initialize(p_init_func)) {
		memdelete(instance);
		// Note: When Godot Engine supports reinitialization, clear the instance pointer here.
		//instance = nullptr;
		return nullptr;
	}

	print_web_header();

	// Ease up compatibility.
	ResourceLoader::set_abort_on_missing_resources(false);

	return (GDExtensionObjectPtr)instance;
}

void libgodot_destroy_godot_instance(GDExtensionObjectPtr p_godot_instance) {
	GodotInstance *godot_instance = (GodotInstance *)p_godot_instance;
	if (instance == godot_instance) {
		godot_instance->stop();
		memdelete(godot_instance);
		instance = nullptr;
		Main::cleanup();
	}
}

/**************************************************************************/
/*  libgodot.cpp                                                          */
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

#ifdef LIBRARY_ENABLED
#include "libgodot.h"
#include "core/extension/gdextension_manager.h"

#ifdef __cplusplus
extern "C" {
#endif

void *godotsharp_game_main_init;
GDExtensionInitializationFunction initialization_function;
void (*scene_load_function)(void *);

void *libgodot_sharp_main_init() {
	return godotsharp_game_main_init;
}

LIBGODOT_API void libgodot_mono_bind(void *sharp_main_init, void (*scene_function_bind)(void *)) {
	godotsharp_game_main_init = sharp_main_init;
	scene_load_function = scene_function_bind;
}

LIBGODOT_API void libgodot_gdextension_bind(GDExtensionInitializationFunction initialization_bind, void (*scene_function_bind)(void *)) {
	initialization_function = initialization_bind;
	scene_load_function = scene_function_bind;
}

void libgodot_scene_load(void *scene) {
	if (scene_load_function != nullptr) {
		scene_load_function(scene);
	}
}

int libgodot_is_scene_loadable() {
	return scene_load_function != nullptr;
}

void libgodot_init_resource() {
	if (initialization_function != nullptr) {
		Ref<GDExtension> libgodot;
		libgodot.instantiate();
		Error err = libgodot->initialize_extension_function(initialization_function, "LibGodot");
		if (err != OK) {
			ERR_PRINT("LibGodot Had an error initialize_extension_function'");
		} else {
			print_verbose("LibGodot initialization");
			libgodot->set_path("res://LibGodotGDExtension");
			GDExtensionManager::get_singleton()->load_extension("res://LibGodotGDExtension");
		}
	}
}

#ifdef __cplusplus
}
#endif
#endif

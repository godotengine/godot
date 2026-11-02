/**************************************************************************/
/*  libgodot.h                                                            */
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

#pragma once

#include "core/extension/gdextension_interface.gen.h"

#ifndef __cplusplus
#include <stdbool.h> // 2dog: libgodot_error_callback takes a bool.
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Export macros for DLL visibility
#if defined(_MSC_VER) || defined(__MINGW32__)
#define LIBGODOT_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define LIBGODOT_API __attribute__((visibility("default")))
#else
#define LIBGODOT_API
#endif

/**
 * @name libgodot_create_godot_instance
 * @since 4.6
 *
 * Creates a new Godot instance.
 *
 * @param p_argc The number of command line arguments.
 * @param p_argv The C-style array of command line arguments.
 * @param p_init_func GDExtension initialization function of the host application.
 *
 * @return A pointer to created \ref GodotInstance GDExtension object or nullptr if there was an error.
 */
LIBGODOT_API GDExtensionObjectPtr libgodot_create_godot_instance(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func);

/**
 * @name libgodot_destroy_godot_instance
 * @since 4.6
 *
 * Destroys an existing Godot instance.
 *
 * @param p_godot_instance The reference to the GodotInstance object to destroy.
 *
 */
LIBGODOT_API void libgodot_destroy_godot_instance(GDExtensionObjectPtr p_godot_instance);

/**
 * @name libgodot_import_project
 * @since 4.6
 *
 * Runs the editor import lifecycle for p_project_path. Requires an editor
 * build; template builds return -1. Do not call while an instance exists.
 *
 * @param p_project_path Path to the directory containing project.godot.
 * @param p_extra_argc   Number of extra command line arguments.
 * @param p_extra_argv   Extra arguments appended after the built-in
 *                       ["--headless", "--import", "--path", p_project_path].
 *
 * @return Process-style exit code: 0 on success, non-zero on failure,
 *         -1 if this build has no editor.
 */
LIBGODOT_API int libgodot_import_project(const char *p_project_path, int p_extra_argc, const char *p_extra_argv[]);

/**
 * @name libgodot_export_pack
 * @since 4.7
 *
 * Runs the editor pack export lifecycle. The project must already be
 * imported. Requires an editor build; template builds return -1. Do not call
 * while an instance exists. Runs headless unless p_extra_argv passes
 * --rendering-driver, as the shader baker needs a renderer.
 *
 * @param p_project_path Path to the directory containing project.godot.
 * @param p_preset       Name of the export preset (from export_presets.cfg).
 * @param p_output_path  Path of the .pck (or .zip) file to write.
 * @param p_extra_argc   Number of extra command line arguments.
 * @param p_extra_argv   Extra arguments appended after the built-in argument list.
 *
 * @return Process-style exit code: 0 on success, non-zero on failure,
 *         -1 if this build has no editor.
 */
LIBGODOT_API int libgodot_export_pack(const char *p_project_path, const char *p_preset, const char *p_output_path, int p_extra_argc, const char *p_extra_argv[]);

// 2dog: receives every error and warning on the reporting thread, under Godot's error lock; must not call into Godot.
// 2dog: p_type is an ErrorHandlerType (0 error, 1 warning, 2 script, 3 shader).
typedef void (*libgodot_error_callback)(void *p_userdata, const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_message, bool p_editor_notify, int p_type);

// 2dog: replaces the process-wide error callback (nullptr unregisters); the old one is not running once this returns.
LIBGODOT_API void libgodot_set_error_callback(libgodot_error_callback p_callback, void *p_userdata);

#ifdef __cplusplus
}
#endif // __cplusplus

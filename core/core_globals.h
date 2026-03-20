/**************************************************************************/
/*  core_globals.h                                                        */
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

typedef GDExtensionBool (*GDInitFunction)();
typedef void (*GDWorldInitFunction)();

// Home for state needed from global functions
// that cannot be stored in Engine or OS due to e.g. circular includes

class CoreGlobals {
public:
	static inline bool leak_reporting_enabled = true;
	static inline bool print_line_enabled = true;
	static inline bool print_error_enabled = true;

	static inline GDInitFunction global_project_settings_function = nullptr;
	static inline GDWorldInitFunction global_world_init_function = nullptr;

	static inline bool run_global_project_settings_function() {
		if (global_project_settings_function == nullptr) {
			return false;
		}
		return global_project_settings_function();
	}

	static inline bool run_global_world_init_function() {
		if (global_world_init_function == nullptr) {
			return false;
		}
		global_world_init_function();
		return true;
	}

	static inline GDExtensionInitializationFunction global_init_func_libgodot = nullptr;
	static inline int32_t global_load_status_libgodot = 0;
};

/**************************************************************************/
/*  renames_map_3_to_4.h                                                  */
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

#ifndef RENAMES_MAP_3_TO_4_H
#define RENAMES_MAP_3_TO_4_H

#ifndef DISABLE_DEPRECATED

struct RenamesMap3To4 {
	static const char *enum_renames[][2];
	static const char *gdscript_function_renames[][2];
	static const char *csharp_function_renames[][2];
	static const char *gdscript_properties_renames[][2];
	static const char *csharp_properties_renames[][2];
	static const char *gdscript_signals_renames[][2];
	static const char *csharp_signals_renames[][2];
	static const char *project_settings_renames[][2];
	static const char *project_godot_renames[][2];
	static const char *input_map_renames[][2];
	static const char *builtin_types_renames[][2];
	static const char *shaders_renames[][2];
	static const char *class_renames[][2];
	static const char *color_renames[][2];
	static const char *theme_override_renames[][2];
};

#endif // DISABLE_DEPRECATED

#endif // RENAMES_MAP_3_TO_4_H

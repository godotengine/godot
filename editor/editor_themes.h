/*************************************************************************/
/*  editor_themes.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef EDITOR_THEMES_H
#define EDITOR_THEMES_H

#include "scene/resources/texture.h"
#include "scene/resources/theme.h"

// The default icon theme is designed to be used for a dark theme. This map stores
// Color values to convert to other colors for better readability on a light theme.
class EditorColorMap {
	// Godot Color values are used to avoid the ambiguity of strings
	// (where "#ffffff", "fff", and "white" are all equivalent).
	static HashMap<Color, Color> editor_color_map;

public:
	static void create();
	static void add_color_pair(const String p_from_color, const String p_to_color);

	static HashMap<Color, Color> &get() { return editor_color_map; };
};

Ref<Theme> create_editor_theme(Ref<Theme> p_theme = nullptr);

Ref<Theme> create_custom_theme(Ref<Theme> p_theme = nullptr);

String get_default_project_icon();

#endif // EDITOR_THEMES_H

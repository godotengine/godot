/**************************************************************************/
/*  editor_theme.h                                                        */
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

#ifndef EDITOR_THEME_H
#define EDITOR_THEME_H

#include "scene/resources/theme.h"

class EditorTheme : public Theme {
	GDCLASS(EditorTheme, Theme);

	static Vector<StringName> editor_theme_types;

public:
	virtual Color get_color(const StringName &p_name, const StringName &p_theme_type) const override;
	virtual int get_constant(const StringName &p_name, const StringName &p_theme_type) const override;
	virtual Ref<Font> get_font(const StringName &p_name, const StringName &p_theme_type) const override;
	virtual int get_font_size(const StringName &p_name, const StringName &p_theme_type) const override;
	virtual Ref<Texture2D> get_icon(const StringName &p_name, const StringName &p_theme_type) const override;
	virtual Ref<StyleBox> get_stylebox(const StringName &p_name, const StringName &p_theme_type) const override;

	static void initialize();
	static void finalize();
};

#endif // EDITOR_THEME_H

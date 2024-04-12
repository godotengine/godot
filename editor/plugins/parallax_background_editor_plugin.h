/**************************************************************************/
/*  parallax_background_editor_plugin.h                                   */
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

#ifndef PARALLAX_BACKGROUND_EDITOR_PLUGIN_H
#define PARALLAX_BACKGROUND_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"

class HBoxContainer;
class MenuButton;
class ParallaxBackground;

class ParallaxBackgroundEditorPlugin : public EditorPlugin {
	GDCLASS(ParallaxBackgroundEditorPlugin, EditorPlugin);

	enum {
		MENU_CONVERT_TO_PARALLAX_2D,
	};

	ParallaxBackground *parallax_background = nullptr;
	HBoxContainer *toolbar = nullptr;
	MenuButton *menu = nullptr;

	void _menu_callback(int p_idx);
	void convert_to_parallax2d();

protected:
	void _notification(int p_what);

public:
	virtual String get_name() const override { return "ParallaxBackground"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	ParallaxBackgroundEditorPlugin();
};

#endif // PARALLAX_BACKGROUND_EDITOR_PLUGIN_H

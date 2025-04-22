/**************************************************************************/
/*  editor_quick_movie_maker_config.h                                     */
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

#include "editor/gui/editor_file_dialog.h"
#include "editor/gui/editor_filepath_select.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/popup.h"

class EditorQuickMovieMakerConfig : public PopupPanel {
	GDCLASS(EditorQuickMovieMakerConfig, PopupPanel);

	VBoxContainer *parts_container = nullptr;

	Label *path_label = nullptr;
	VBoxContainer *path_container = nullptr;
	EditorFilepathSelect *filepath_select = nullptr;
	Button *open_settings_button = nullptr;
	bool movie_path_was_changed = false;

	void _notification(int p_what);
	void _open_settings_pressed();

public:
	void _close_requested();
	void _path_edit_focus_exited();
	void _path_edit_text_submitted(const String &p_new_text);
	void _path_edit_text_changed(const String &p_new_text);
	void _visibility_changed();
	void _update_movie_file_path(const String &p_new_text);

	EditorQuickMovieMakerConfig();
};

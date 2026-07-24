/**************************************************************************/
/*  viewport_bookmarks.h                                                  */
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

#include "scene/gui/dialogs.h"

class Button;
class ConfirmationDialog;
class ItemList;
class Label;
class LineEdit;
class Node;

class ViewportBookmarks {
public:
	enum Type {
		TYPE_2D,
		TYPE_3D,
	};

	static constexpr const char *META_KEY = "_edit_viewport_bookmarks_";

	static Array get_bookmarks(const Node *p_scene_root, Type p_type);
	static Dictionary make_metadata(const Node *p_scene_root, Type p_type, const Array &p_bookmarks);
	static bool is_valid_name(const Array &p_bookmarks, const String &p_name, int p_except = -1);
	static String make_unique_name(const Array &p_bookmarks);
};

class ViewportBookmarkManager : public AcceptDialog {
	GDCLASS(ViewportBookmarkManager, AcceptDialog);

	ViewportBookmarks::Type type = ViewportBookmarks::TYPE_2D;
	Callable capture_view;
	Callable activate_view;

	ItemList *bookmark_list = nullptr;
	Button *go_to_button = nullptr;
	Button *overwrite_button = nullptr;
	Button *rename_button = nullptr;
	Button *delete_button = nullptr;

	ConfirmationDialog *name_dialog = nullptr;
	LineEdit *name_edit = nullptr;
	Label *name_error = nullptr;
	ConfirmationDialog *delete_dialog = nullptr;

	int rename_index = -1;
	int active_index = -1;

	Node *_get_scene_root() const;
	Array _get_bookmarks() const;
	void _commit_bookmarks(const Array &p_bookmarks, const String &p_action);
	void _selection_changed(int p_index);
	void _item_activated(int p_index);
	void _go_to_selected();
	void _overwrite_selected();
	void _rename_selected();
	void _show_name_dialog(int p_index);
	void _name_changed(const String &p_name);
	void _name_confirmed();
	void _show_delete_dialog();
	void _delete_confirmed();

protected:
	static void _bind_methods();

public:
	void refresh();
	void popup_add();
	void popup_manage();
	void activate(int p_index);
	void activate_next();
	void activate_previous();

	ViewportBookmarkManager(ViewportBookmarks::Type p_type, const Callable &p_capture_view, const Callable &p_activate_view);
};

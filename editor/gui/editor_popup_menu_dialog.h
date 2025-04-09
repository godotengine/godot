/**************************************************************************/
/*  editor_popup_menu_dialog.h                                            */
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

#include "core/os/keyboard.h"
#include "core/string/fuzzy_search.h"
#include "scene/gui/dialogs.h"

class LineEdit;
class ItemList;
class PopupMenu;

class EditorPopupMenuDialog : public AcceptDialog {
	GDCLASS(EditorPopupMenuDialog, AcceptDialog);

public:
	void popup_dialog(PopupMenu *p_popup_menu, const String &p_title, const String &p_search_placeholder);

	EditorPopupMenuDialog();

	struct PopupMenuItem {
		String text;
		Ref<Texture2D> icon;
		int popup_menu_index;
	};

protected:
	virtual void cancel_pressed() override;
	virtual void ok_pressed() override;

private:
	Vector<PopupMenuItem> items;
	Vector<bool> item_visible_status;

	Vector<FuzzySearchResult> search_results;

	PopupMenu *popup_menu = nullptr;

	LineEdit *search_box = nullptr;
	ItemList *item_list = nullptr;

	int popup_menu_offset = 0;
	int num_visible_results = 0;

	void cleanup();

	void handle_search_box_input(const Ref<InputEvent> &p_ie);
	void handle_item_activated(int p_index);

	void _move_selection_index(Key p_key);
	void _update_fuzzy_search_results(const String &p_query);
	void _search_box_text_changed(const String &p_query);
};

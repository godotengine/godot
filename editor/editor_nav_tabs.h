/**************************************************************************/
/*  editor_nav_tabs.h                                                     */
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

#ifndef EDITOR_NAV_TABS_H
#define EDITOR_NAV_TABS_H

#include "editor/editor_tab.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/item_list.h"

class EditorNavTabs : public ConfirmationDialog {
	GDCLASS(EditorNavTabs, ConfirmationDialog);

	static Rect2i prev_rect;
	static bool was_showed;

	Vector<EditorTab *> _tabs;

	ItemList *tab_list = nullptr;

	void _update_tab_list();
	void _tab_list_clicked(int p_item, Vector2 p_local_mouse_pos, MouseButton p_mouse_button_index);
	void _select_tab(int p_item);
	void _close();
	void _select_next();
	void _select_prev();

	void _list_gui_input(const Ref<InputEvent> &p_event);

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

public:
	void popup_dialog(Vector<EditorTab *> p_tabs, bool p_next_tab);

	EditorNavTabs();
};

#endif // EDITOR_NAV_TABS_H

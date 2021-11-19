/*************************************************************************/
/*  editor_tool_drawer.h                                                 */
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

#ifndef EDITOR_TOOL_DRAWER_H
#define EDITOR_TOOL_DRAWER_H

#include "editor_node.h"

#include "scene/gui/box_container.h"
#include "scene/gui/control.h"
#include "scene/gui/margin_container.h"

class Button;
class Label;
class PanelContainer;
class ScrollContainer;
class TextureRect;

class EditorToolDrawer : public MarginContainer {
	GDCLASS(EditorToolDrawer, MarginContainer);

	PanelContainer *bg_panel;
	VBoxContainer *main_vb;
	PanelContainer *title_panel;
	TextureRect *title_icon;
	Label *title_label;

protected:
	void _notification(int p_notification);

public:
	void set_title(const String p_title);
	void set_title_icon(const Ref<Texture2D> &p_icon);
	void add_content(Control *p_content);

	EditorToolDrawer();
};

class EditorToolDrawerItemGroup : public VBoxContainer {
	GDCLASS(EditorToolDrawerItemGroup, VBoxContainer);

	Label *title_label;

protected:
	void _notification(int p_notification);

public:
	void set_title(const String p_title);

	EditorToolDrawerItemGroup();
};

class EditorToolDrawerContainer : public Control {
	GDCLASS(EditorToolDrawerContainer, Control);

	Button *tool_drawer_mass_toggle;
	VBoxContainer *tool_drawer_bar;
	ScrollContainer *tool_drawer_scroll;
	VBoxContainer *tool_drawer_vb;

	struct DrawerItem {
		Button *button = nullptr;
		Label *label = nullptr;
		EditorToolDrawer *drawer = nullptr;
		bool expanded = false;
	};

	Map<Control *, DrawerItem> tool_drawer_map;

	void _toggle_drawer(Control *p_control);
	void _toggle_all_drawers();
	void _set_drawers_toggled_nocheck(bool p_expanded);

	void _update_tool_drawer();
	void _update_mass_toggle_visibility();
	void _update_mass_toggle_icon();

	void _update_scroll_area();

protected:
	void _notification(int p_notification);

public:
	void add_drawer(const String p_name, const Ref<Texture2D> &p_icon, Control *p_control);
	void remove_drawer(Control *p_control);
	void set_drawer_visible(Control *p_control, bool p_visible);

	EditorToolDrawerContainer();
};

#endif //EDITOR_TOOL_DRAWER_H

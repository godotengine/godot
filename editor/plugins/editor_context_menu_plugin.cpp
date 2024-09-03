/**************************************************************************/
/*  editor_context_menu_plugin.cpp                                        */
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

#include "editor_context_menu_plugin.h"

#include "core/input/shortcut.h"
#include "editor/editor_node.h"
#include "scene/resources/texture.h"

void EditorContextMenuPlugin::add_options(const Vector<String> &p_paths) {
	GDVIRTUAL_CALL(_popup_menu, p_paths);
}

void EditorContextMenuPlugin::add_menu_shortcut(const Ref<Shortcut> &p_shortcut, const Callable &p_callable) {
	context_menu_shortcuts.insert(p_shortcut, p_callable);
}

void EditorContextMenuPlugin::add_context_menu_item(const String &p_name, const Callable &p_callable, const Ref<Texture2D> &p_texture, const Ref<Shortcut> &p_shortcut) {
	ERR_FAIL_COND_MSG(context_menu_items.has(p_name), "Context menu item already registered.");
	ERR_FAIL_COND_MSG(context_menu_items.size() == MAX_ITEMS, "Maximum number of context menu items reached.");
	ContextMenuItem item;
	item.item_name = p_name;
	item.callable = p_callable;
	item.icon = p_texture;
	item.shortcut = p_shortcut;
	item.idx = EditorData::CONTEXT_MENU_ITEM_ID_BASE + start_idx + context_menu_shortcuts.size() + context_menu_items.size();
	context_menu_items.insert(p_name, item);
}

void EditorContextMenuPlugin::clear_context_menu_items() {
	context_menu_items.clear();
}

void EditorContextMenuPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_menu_shortcut", "shortcut", "callback"), &EditorContextMenuPlugin::add_menu_shortcut);
	ClassDB::bind_method(D_METHOD("add_context_menu_item", "name", "callback", "icon", "shortcut"), &EditorContextMenuPlugin::add_context_menu_item, DEFVAL(Ref<Texture2D>()), DEFVAL(Ref<Shortcut>()));
	GDVIRTUAL_BIND(_popup_menu, "paths");
}

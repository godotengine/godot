/**************************************************************************/
/*  editor_context_menu_plugin.h                                          */
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

#ifndef EDITOR_CONTEXT_MENU_PLUGIN_H
#define EDITOR_CONTEXT_MENU_PLUGIN_H

#include "core/object/gdvirtual.gen.inc"
#include "core/object/ref_counted.h"

class Texture2D;
class Shortcut;

class EditorContextMenuPlugin : public RefCounted {
	GDCLASS(EditorContextMenuPlugin, RefCounted);

public:
	int start_idx;

	inline static constexpr int MAX_ITEMS = 100;

	struct ContextMenuItem {
		int idx = 0;
		String item_name;
		Callable callable;
		Ref<Texture2D> icon;
		Ref<Shortcut> shortcut;
	};
	HashMap<String, ContextMenuItem> context_menu_items;
	HashMap<Ref<Shortcut>, Callable> context_menu_shortcuts;

protected:
	static void _bind_methods();

	GDVIRTUAL1(_popup_menu, Vector<String>);

public:
	virtual void add_options(const Vector<String> &p_paths);
	void add_menu_shortcut(const Ref<Shortcut> &p_shortcut, const Callable &p_callable);
	void add_context_menu_item(const String &p_name, const Callable &p_callable, const Ref<Texture2D> &p_texture, const Ref<Shortcut> &p_shortcut);
	void clear_context_menu_items();
};

#endif // EDITOR_CONTEXT_MENU_PLUGIN_H

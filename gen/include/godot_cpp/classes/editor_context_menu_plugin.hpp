/**************************************************************************/
/*  editor_context_menu_plugin.hpp                                        */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/texture2d.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class PackedStringArray;
class PopupMenu;
class Shortcut;
class String;

class EditorContextMenuPlugin : public RefCounted {
	GDEXTENSION_CLASS(EditorContextMenuPlugin, RefCounted)

public:
	enum ContextMenuSlot {
		CONTEXT_SLOT_SCENE_TREE = 0,
		CONTEXT_SLOT_FILESYSTEM = 1,
		CONTEXT_SLOT_SCRIPT_EDITOR = 2,
		CONTEXT_SLOT_FILESYSTEM_CREATE = 3,
		CONTEXT_SLOT_SCRIPT_EDITOR_CODE = 4,
		CONTEXT_SLOT_SCENE_TABS = 5,
		CONTEXT_SLOT_2D_EDITOR = 6,
	};

	void add_menu_shortcut(const Ref<Shortcut> &p_shortcut, const Callable &p_callback);
	void add_context_menu_item(const String &p_name, const Callable &p_callback, const Ref<Texture2D> &p_icon = nullptr);
	void add_context_menu_item_from_shortcut(const String &p_name, const Ref<Shortcut> &p_shortcut, const Ref<Texture2D> &p_icon = nullptr);
	void add_context_submenu_item(const String &p_name, PopupMenu *p_menu, const Ref<Texture2D> &p_icon = nullptr);
	virtual void _popup_menu(const PackedStringArray &p_paths);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_popup_menu), decltype(&T::_popup_menu)>) {
			BIND_VIRTUAL_METHOD(T, _popup_menu, 4015028928);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(EditorContextMenuPlugin::ContextMenuSlot);


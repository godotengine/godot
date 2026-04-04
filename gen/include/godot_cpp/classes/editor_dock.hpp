/**************************************************************************/
/*  editor_dock.hpp                                                       */
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

#include <godot_cpp/classes/margin_container.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ConfigFile;
class Shortcut;
class Texture2D;

class EditorDock : public MarginContainer {
	GDEXTENSION_CLASS(EditorDock, MarginContainer)

public:
	enum DockLayout : uint64_t {
		DOCK_LAYOUT_VERTICAL = 1,
		DOCK_LAYOUT_HORIZONTAL = 2,
		DOCK_LAYOUT_FLOATING = 4,
		DOCK_LAYOUT_ALL = 7,
	};

	enum DockSlot {
		DOCK_SLOT_NONE = -1,
		DOCK_SLOT_LEFT_UL = 0,
		DOCK_SLOT_LEFT_BL = 1,
		DOCK_SLOT_LEFT_UR = 2,
		DOCK_SLOT_LEFT_BR = 3,
		DOCK_SLOT_RIGHT_UL = 4,
		DOCK_SLOT_RIGHT_BL = 5,
		DOCK_SLOT_RIGHT_UR = 6,
		DOCK_SLOT_RIGHT_BR = 7,
		DOCK_SLOT_BOTTOM = 8,
		DOCK_SLOT_MAX = 9,
	};

	void open();
	void make_visible();
	void close();
	void set_title(const String &p_title);
	String get_title() const;
	void set_layout_key(const String &p_layout_key);
	String get_layout_key() const;
	void set_global(bool p_global);
	bool is_global() const;
	void set_transient(bool p_transient);
	bool is_transient() const;
	void set_closable(bool p_closable);
	bool is_closable() const;
	void set_icon_name(const StringName &p_icon_name);
	StringName get_icon_name() const;
	void set_dock_icon(const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_dock_icon() const;
	void set_force_show_icon(bool p_force);
	bool get_force_show_icon() const;
	void set_title_color(const Color &p_color);
	Color get_title_color() const;
	void set_dock_shortcut(const Ref<Shortcut> &p_shortcut);
	Ref<Shortcut> get_dock_shortcut() const;
	void set_default_slot(EditorDock::DockSlot p_slot);
	EditorDock::DockSlot get_default_slot() const;
	void set_available_layouts(BitField<EditorDock::DockLayout> p_layouts);
	BitField<EditorDock::DockLayout> get_available_layouts() const;
	virtual void _update_layout(int32_t p_layout);
	virtual void _save_layout_to_config(const Ref<ConfigFile> &p_config, const String &p_section) const;
	virtual void _load_layout_from_config(const Ref<ConfigFile> &p_config, const String &p_section);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		MarginContainer::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_update_layout), decltype(&T::_update_layout)>) {
			BIND_VIRTUAL_METHOD(T, _update_layout, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_save_layout_to_config), decltype(&T::_save_layout_to_config)>) {
			BIND_VIRTUAL_METHOD(T, _save_layout_to_config, 3076455711);
		}
		if constexpr (!std::is_same_v<decltype(&B::_load_layout_from_config), decltype(&T::_load_layout_from_config)>) {
			BIND_VIRTUAL_METHOD(T, _load_layout_from_config, 2838822993);
		}
	}

public:
};

} // namespace godot

VARIANT_BITFIELD_CAST(EditorDock::DockLayout);
VARIANT_ENUM_CAST(EditorDock::DockSlot);


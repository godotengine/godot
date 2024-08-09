/**************************************************************************/
/*  editor_tab.h                                                          */
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

#ifndef EDITOR_TAB_H
#define EDITOR_TAB_H

#include "core/variant/variant.h"
#include "scene/resources/image_texture.h"

class EditorTab : public Object {
	GDCLASS(EditorTab, Object);

	friend class EditorSceneTabs;

private:
	String _name;
	String _resource_path;
	Ref<Texture2D> _icon;
	Ref<Texture2D> _tab_button_icon;
	Variant _state;
	bool _closing = false;
	bool _cancel = false;
	uint64_t _last_used = 0;

protected:
	static void _bind_methods();

public:
	struct EditorTabComparator {
		_FORCE_INLINE_ bool operator()(const EditorTab *a, const EditorTab *b) const {
			return a->get_last_used() > b->get_last_used();
		}
	};

	String get_name() const;
	void set_name(String p_name);
	String get_resource_path() const;
	void set_resource_path(String p_resource_path);
	Ref<Texture2D> get_icon() const;
	void set_icon(Ref<Texture2D> p_icon);
	Ref<Texture2D> get_tab_button_icon() const;
	void set_tab_button_icon(Ref<Texture2D> p_tab_button_icon);
	Variant get_state() const;
	void set_state(Variant p_state);
	bool get_closing() const;
	void set_closing(bool p_closing);
	bool get_cancel() const;
	void set_cancel(bool p_cancel);
	uint64_t get_last_used() const;
	void set_last_used(uint64_t p_last_used);
	void update_last_used();
};

#endif // EDITOR_TAB_H

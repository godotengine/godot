/*************************************************************************/
/*  script_tree_list.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef SCRIPT_TREE_LIST_H
#define SCRIPT_TREE_LIST_H

#include "scene/gui/box_container.h"
#include "scene/gui/control.h"
#include "scene/gui/tree.h"

class ScriptTreeList : public VBoxContainer {

	GDCLASS(ScriptTreeList, VBoxContainer);

	Tree *tree;
	TreeItem *root;

	int item_count;
	int current;

	TreeItem *_get_item(const int p_idx);
	void _item_selected();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void clear();
	void add_item(const String &p_item, const Ref<Texture> &p_texture = Ref<Texture>());
	void remove_item(const int p_idx);
	int get_item_count() const;

	void set_item_metadata(const int p_idx, const Variant &p_metadata);
	int get_item_metadata(const int p_idx);
	int find_metadata(const Variant &p_idx) const;

	void set_item_tooltip(const int p_idx, const String &p_tooltip);
	String get_item_tooltip(const int p_idx);

	void set_item_custom_font_color(const int p_idx, const Color &p_font_color);
	Color get_item_custom_font_color(const int p_idx);

	void set_item_custom_bg_color(const int p_idx, const Color &p_custom_bg_color);
	Color get_item_custom_bg_color(const int p_idx);

	void set_item_collapsed(const int p_idx, const bool p_collapsed);
	bool is_item_collapsed(const int p_idx);

	void add_functions(const int p_idx, const Vector<String> p_functions);
	void clear_functions(const int p_idx);

	void select(const int p_idx);
	int get_current() const;

	void update_settings();

	ScriptTreeList();
	~ScriptTreeList();
};

#endif

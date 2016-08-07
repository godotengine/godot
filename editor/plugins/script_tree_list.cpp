/*************************************************************************/
/*  script_tree_list.cpp                                                 */
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

#include "script_tree_list.h"
#include "editor/editor_settings.h"

void ScriptTreeList::clear() {
	root->clear_children();
	item_count = 0;
	current = -1;
}

void ScriptTreeList::add_item(const String &p_item, const Ref<Texture> &p_texture) {
	TreeItem *item = tree->create_item(root);
	item->set_text(0, p_item);
	item->set_icon(0, p_texture);
	item->set_selectable(0, true);
	item->set_collapsed(true);
	item_count++;
}

void ScriptTreeList::remove_item(const int p_idx) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		root->remove_child(item);
		item_count--;
	}
}

int ScriptTreeList::get_item_count() const {
	return item_count;
}

void ScriptTreeList::set_item_metadata(const int p_idx, const Variant &p_metadata) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		item->set_metadata(0, p_metadata);
	}
}

int ScriptTreeList::get_item_metadata(const int p_idx) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		return item->get_metadata(0);
	}
	return -1;
}

int ScriptTreeList::find_metadata(const Variant &p_idx) const {
	TreeItem *item = root->get_children();
	int i = 0;

	while (item) {

		if (item->get_metadata(0) == p_idx) {
			return i;
		}
		item = item->get_next();
		i++;
	}
	return -1;
}

void ScriptTreeList::set_item_tooltip(const int p_idx, const String &p_tooltip) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		item->set_tooltip(0, p_tooltip);
	}
}

String ScriptTreeList::get_item_tooltip(const int p_idx) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		return item->get_tooltip(0);
	}
	return "";
}

void ScriptTreeList::set_item_custom_font_color(const int p_idx, const Color &p_font_color) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		item->set_custom_color(0, p_font_color);
	}
}

Color ScriptTreeList::get_item_custom_font_color(const int p_idx) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		return item->get_custom_color(0);
	}
}

void ScriptTreeList::set_item_custom_bg_color(const int p_idx, const Color &p_custom_bg_color) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		item->set_custom_bg_color(0, p_custom_bg_color);
	}
}

Color ScriptTreeList::get_item_custom_bg_color(const int p_idx) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		item->get_custom_bg_color(0);
	}
	return Color(0, 0, 0, 0);
}

void ScriptTreeList::set_item_collapsed(const int p_idx, const bool p_collapsed) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		item->set_collapsed(p_collapsed);
	}
}

bool ScriptTreeList::is_item_collapsed(const int p_idx) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		return item->is_collapsed();
	}
	return false;
}

void ScriptTreeList::add_functions(const int p_idx, const Vector<String> p_functions) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		for (int i = 0; i < p_functions.size(); i++) {
			TreeItem *function = tree->create_item(item);
			function->set_text(0, p_functions[i].get_slice(":", 0));
			function->set_selectable(0, true);
		}
	}
}

void ScriptTreeList::clear_functions(const int p_idx) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		item->clear_children();
	}
}

void ScriptTreeList::select(const int p_idx) {
	TreeItem *item = _get_item(p_idx);
	if (item) {
		item->select(0);
		current = p_idx;
	}
}

int ScriptTreeList::get_current() const {
	return current;
}

TreeItem *ScriptTreeList::_get_item(const int p_idx) {
	if (p_idx < item_count && p_idx >= 0) {
		TreeItem *item = root->get_children();

		for (int i = 0; i < p_idx; i++) {

			if (!item) {
				break;
			}
			item = item->get_next();
		}
		return item;
	}
	return NULL;
}

void ScriptTreeList::_item_selected() {
	TreeItem *item = tree->get_selected();
	if (item) {
		if (item->get_parent() == root) {
			emit_signal("script_selected", find_metadata(item->get_metadata(0)));
		} else {
			emit_signal("function_selected", find_metadata(item->get_parent()->get_metadata(0)), item->get_text(0));
		}
	}
}

void ScriptTreeList::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		tree->connect("cell_selected", this, "_item_selected");
		update_settings();
	}
}

void ScriptTreeList::_bind_methods() {
	ClassDB::bind_method("_item_selected", &ScriptTreeList::_item_selected);

	ADD_SIGNAL(MethodInfo("script_selected", PropertyInfo(Variant::INT, "index")));
	ADD_SIGNAL(MethodInfo("function_selected", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::STRING, "function")));
}

void ScriptTreeList::update_settings() {
	tree->set_hide_folding(EditorSettings::get_singleton()->get("text_editor/open_scripts/use_list_mode"));
	if (EditorSettings::get_singleton()->get("text_editor/open_scripts/use_list_mode")) {
		TreeItem *item = root->get_children();
		while (item) {
			item->set_collapsed(true);
			item = item->get_next();
		}
	}
}

ScriptTreeList::ScriptTreeList() {
	VBoxContainer *vbc = this;

	item_count = 0;
	current = -1;

	tree = memnew(Tree);
	tree->set_hide_root(true);
	tree->set_select_mode(Tree::SELECT_SINGLE);
	vbc->add_child(tree);
	tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);

	root = tree->create_item();
}

ScriptTreeList::~ScriptTreeList() {
}

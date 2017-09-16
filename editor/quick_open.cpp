/*************************************************************************/
/*  quick_open.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "quick_open.h"

#include "os/keyboard.h"

void EditorQuickOpen::popup(const StringName &p_base, bool p_enable_multi, bool p_add_dirs, bool p_dontclear) {

	add_directories = p_add_dirs;
	popup_centered_ratio(0.6);
	if (p_dontclear)
		search_box->select_all();
	else
		search_box->clear();
	if (p_enable_multi)
		search_options->set_select_mode(Tree::SELECT_MULTI);
	else
		search_options->set_select_mode(Tree::SELECT_SINGLE);
	search_box->grab_focus();
	base_type = p_base;
	_update_search();
}

String EditorQuickOpen::get_selected() const {

	TreeItem *ti = search_options->get_selected();
	if (!ti)
		return String();

	return "res://" + ti->get_text(0);
}

Vector<String> EditorQuickOpen::get_selected_files() const {

	Vector<String> files;

	TreeItem *item = search_options->get_next_selected(search_options->get_root());
	while (item) {

		files.push_back("res://" + item->get_text(0));

		item = search_options->get_next_selected(item);
	}

	return files;
}

void EditorQuickOpen::_text_changed(const String &p_newtext) {

	_update_search();
}

void EditorQuickOpen::_sbox_input(const Ref<InputEvent> &p_ie) {

	Ref<InputEventKey> k = p_ie;
	if (k.is_valid()) {

		switch (k->get_scancode()) {
			case KEY_UP:
			case KEY_DOWN:
			case KEY_PAGEUP:
			case KEY_PAGEDOWN: {

				search_options->call("_gui_input", k);
				search_box->accept_event();

				TreeItem *root = search_options->get_root();
				if (!root->get_children())
					break;

				TreeItem *current = search_options->get_selected();

				TreeItem *item = search_options->get_next_selected(root);
				while (item) {
					item->deselect(0);
					item = search_options->get_next_selected(item);
				}

				current->select(0);

			} break;
		}
	}
}

float EditorQuickOpen::_path_cmp(String search, String path) const {

	if (search == path) {
		return 1.2f;
	}
	if (path.findn(search) != -1) {
		return 1.1f;
	}
	return path.to_lower().similarity(search.to_lower());
}

void EditorQuickOpen::_parse_fs(EditorFileSystemDirectory *efsd, Vector<Pair<String, Ref<Texture> > > &list) {

	if (!add_directories) {
		for (int i = 0; i < efsd->get_subdir_count(); i++) {

			_parse_fs(efsd->get_subdir(i), list);
		}
	}

	String search_text = search_box->get_text();

	if (add_directories) {
		String path = efsd->get_path();
		if (!path.ends_with("/"))
			path += "/";
		if (path != "res://") {
			path = path.substr(6, path.length());
			if (search_text.is_subsequence_ofi(path)) {
				Pair<String, Ref<Texture> > pair;
				pair.first = path;
				pair.second = get_icon("folder", "FileDialog");

				if (search_text != String() && list.size() > 0) {

					float this_sim = _path_cmp(search_text, path);
					float other_sim = _path_cmp(list[0].first, path);
					int pos = 1;

					while (pos < list.size() && this_sim <= other_sim) {
						other_sim = _path_cmp(list[pos++].first, path);
					}

					pos = this_sim >= other_sim ? pos - 1 : pos;
					list.insert(pos, pair);

				} else {
					list.push_back(pair);
				}
			}
		}
	}
	for (int i = 0; i < efsd->get_file_count(); i++) {

		String file = efsd->get_file_path(i);
		file = file.substr(6, file.length());

		if (ClassDB::is_parent_class(efsd->get_file_type(i), base_type) && (search_text.is_subsequence_ofi(file))) {
			Pair<String, Ref<Texture> > pair;
			pair.first = file;
			pair.second = get_icon((has_icon(efsd->get_file_type(i), ei) ? efsd->get_file_type(i) : ot), ei);
			list.push_back(pair);
		}
	}

	if (add_directories) {
		for (int i = 0; i < efsd->get_subdir_count(); i++) {

			_parse_fs(efsd->get_subdir(i), list);
		}
	}
}

Vector<Pair<String, Ref<Texture> > > EditorQuickOpen::_sort_fs(Vector<Pair<String, Ref<Texture> > > &list) {

	String search_text = search_box->get_text();
	Vector<Pair<String, Ref<Texture> > > sorted_list;

	if (search_text == String() || list.size() == 0)
		return list;

	Vector<float> scores;
	scores.resize(list.size());
	for (int i = 0; i < list.size(); i++)
		scores[i] = _path_cmp(search_text, list[i].first);

	while (list.size() > 0) {

		float best_score = 0.0f;
		int best_idx = 0;

		for (int i = 0; i < list.size(); i++) {
			float current_score = scores[i];
			if (current_score > best_score) {
				best_score = current_score;
				best_idx = i;
			}
		}

		sorted_list.push_back(list[best_idx]);
		list.remove(best_idx);
		scores.remove(best_idx);
	}

	return sorted_list;
}

void EditorQuickOpen::_update_search() {

	search_options->clear();
	TreeItem *root = search_options->create_item();
	EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->get_filesystem();
	Vector<Pair<String, Ref<Texture> > > list;

	_parse_fs(efsd, list);
	list = _sort_fs(list);

	for (int i = 0; i < list.size(); i++) {
		TreeItem *ti = search_options->create_item(root);
		ti->set_text(0, list[i].first);
		ti->set_icon(0, list[i].second);
	}

	if (root->get_children()) {
		TreeItem *ti = root->get_children();

		ti->select(0);
		ti->set_as_cursor(0);
	}

	get_ok()->set_disabled(root->get_children() == NULL);
}

void EditorQuickOpen::_confirmed() {

	TreeItem *ti = search_options->get_selected();
	if (!ti)
		return;
	emit_signal("quick_open");
	hide();
}

void EditorQuickOpen::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		connect("confirmed", this, "_confirmed");
	}
}

StringName EditorQuickOpen::get_base_type() const {

	return base_type;
}

void EditorQuickOpen::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_text_changed"), &EditorQuickOpen::_text_changed);
	ClassDB::bind_method(D_METHOD("_confirmed"), &EditorQuickOpen::_confirmed);
	ClassDB::bind_method(D_METHOD("_sbox_input"), &EditorQuickOpen::_sbox_input);

	ADD_SIGNAL(MethodInfo("quick_open"));
}

EditorQuickOpen::EditorQuickOpen() {

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);
	//set_child_rect(vbc);
	search_box = memnew(LineEdit);
	vbc->add_margin_child(TTR("Search:"), search_box);
	search_box->connect("text_changed", this, "_text_changed");
	search_box->connect("gui_input", this, "_sbox_input");
	search_options = memnew(Tree);
	vbc->add_margin_child(TTR("Matches:"), search_options, true);
	get_ok()->set_text(TTR("Open"));
	get_ok()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated", this, "_confirmed");
	search_options->set_hide_root(true);
	ei = "EditorIcons";
	ot = "Object";
	add_directories = false;
}

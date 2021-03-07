/*************************************************************************/
/*  quick_open.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/os/keyboard.h"

void EditorQuickOpen::popup_dialog(const StringName &p_base, bool p_enable_multi, bool p_dontclear) {
	base_type = p_base;
	allow_multi_select = p_enable_multi;
	search_options->set_select_mode(allow_multi_select ? Tree::SELECT_MULTI : Tree::SELECT_SINGLE);
	popup_centered_clamped(Size2i(600, 440), 0.8f);

	EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->get_filesystem();
	_build_search_cache(efsd);

	if (p_dontclear) {
		search_box->select_all();
		_update_search();
	} else {
		search_box->clear(); // This will emit text_changed.
	}
	search_box->grab_focus();
}

void EditorQuickOpen::_build_search_cache(EditorFileSystemDirectory *p_efsd) {
	for (int i = 0; i < p_efsd->get_subdir_count(); i++) {
		_build_search_cache(p_efsd->get_subdir(i));
	}

	for (int i = 0; i < p_efsd->get_file_count(); i++) {
		String file_type = p_efsd->get_file_type(i);
		if (ClassDB::is_parent_class(file_type, base_type)) {
			String file = p_efsd->get_file_path(i);
			files.push_back(file.substr(6, file.length()));

			// Store refs to used icons.
			String ext = file.get_extension();
			if (!icons.has(ext)) {
				icons.insert(ext, get_theme_icon((has_theme_icon(file_type, "EditorIcons") ? file_type : "Object"), "EditorIcons"));
			}
		}
	}
}

void EditorQuickOpen::_update_search() {
	const String search_text = search_box->get_text();
	const bool empty_search = search_text == "";

	// Filter possible candidates.
	Vector<Entry> entries;
	for (int i = 0; i < files.size(); i++) {
		if (empty_search || search_text.is_subsequence_ofi(files[i])) {
			Entry r;
			r.path = files[i];
			r.score = empty_search ? 0 : _score_path(search_text, files[i].to_lower());
			entries.push_back(r);
		}
	}

	// Display results
	TreeItem *root = search_options->get_root();
	root->clear_children();

	if (entries.size() > 0) {
		if (!empty_search) {
			SortArray<Entry, EntryComparator> sorter;
			sorter.sort(entries.ptrw(), entries.size());
		}

		const int entry_limit = MIN(entries.size(), 300);
		for (int i = 0; i < entry_limit; i++) {
			TreeItem *ti = search_options->create_item(root);
			ti->set_text(0, entries[i].path);
			ti->set_icon(0, *icons.lookup_ptr(entries[i].path.get_extension()));
		}

		TreeItem *to_select = root->get_first_child();
		to_select->select(0);
		to_select->set_as_cursor(0);
		search_options->scroll_to_item(to_select);

		get_ok_button()->set_disabled(false);
	} else {
		search_options->deselect_all();

		get_ok_button()->set_disabled(true);
	}
}

float EditorQuickOpen::_score_path(const String &p_search, const String &p_path) {
	float score = 0.9f + .1f * (p_search.length() / (float)p_path.length());

	// Positive bias for matches close to the beginning of the file name.
	String file = p_path.get_file();
	int pos = file.findn(p_search);
	if (pos != -1) {
		return score * (1.0f - 0.1f * (float(pos) / file.length()));
	}

	// Positive bias for matches close to the end of the path.
	pos = p_path.rfindn(p_search);
	if (pos != -1) {
		return score * (0.8f - 0.1f * (float(p_path.length() - pos) / p_path.length()));
	}

	// Remaining results belong to the same class of results.
	return score * 0.69f;
}

void EditorQuickOpen::_confirmed() {
	if (!search_options->get_selected()) {
		return;
	}
	_cleanup();
	emit_signal("quick_open");
	hide();
}

void EditorQuickOpen::cancel_pressed() {
	_cleanup();
}

void EditorQuickOpen::_cleanup() {
	files.clear();
	icons.clear();
}

void EditorQuickOpen::_text_changed(const String &p_newtext) {
	_update_search();
}

void EditorQuickOpen::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;
	if (k.is_valid()) {
		switch (k->get_keycode()) {
			case KEY_UP:
			case KEY_DOWN:
			case KEY_PAGEUP:
			case KEY_PAGEDOWN: {
				search_options->call("_gui_input", k);
				search_box->accept_event();

				if (allow_multi_select) {
					TreeItem *root = search_options->get_root();
					if (!root->get_first_child()) {
						break;
					}

					TreeItem *current = search_options->get_selected();
					TreeItem *item = search_options->get_next_selected(root);
					while (item) {
						item->deselect(0);
						item = search_options->get_next_selected(item);
					}

					current->select(0);
					current->set_as_cursor(0);
				}
			} break;
		}
	}
}

String EditorQuickOpen::get_selected() const {
	TreeItem *ti = search_options->get_selected();
	if (!ti) {
		return String();
	}

	return "res://" + ti->get_text(0);
}

Vector<String> EditorQuickOpen::get_selected_files() const {
	Vector<String> selected_files;

	TreeItem *item = search_options->get_next_selected(search_options->get_root());
	while (item) {
		selected_files.push_back("res://" + item->get_text(0));
		item = search_options->get_next_selected(item);
	}

	return selected_files;
}

StringName EditorQuickOpen::get_base_type() const {
	return base_type;
}

void EditorQuickOpen::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect("confirmed", callable_mp(this, &EditorQuickOpen::_confirmed));

			search_box->set_clear_button_enabled(true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			disconnect("confirmed", callable_mp(this, &EditorQuickOpen::_confirmed));
		} break;
	}
}

void EditorQuickOpen::_theme_changed() {
	search_box->set_right_icon(search_options->get_theme_icon("Search", "EditorIcons"));
}

void EditorQuickOpen::_bind_methods() {
	ADD_SIGNAL(MethodInfo("quick_open"));
}

EditorQuickOpen::EditorQuickOpen() {
	allow_multi_select = false;

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->connect("theme_changed", callable_mp(this, &EditorQuickOpen::_theme_changed));
	add_child(vbc);

	search_box = memnew(LineEdit);
	search_box->connect("text_changed", callable_mp(this, &EditorQuickOpen::_text_changed));
	search_box->connect("gui_input", callable_mp(this, &EditorQuickOpen::_sbox_input));
	vbc->add_margin_child(TTR("Search:"), search_box);
	register_text_enter(search_box);

	search_options = memnew(Tree);
	search_options->connect("item_activated", callable_mp(this, &EditorQuickOpen::_confirmed));
	search_options->create_item();
	search_options->set_hide_root(true);
	search_options->set_hide_folding(true);
	search_options->add_theme_constant_override("draw_guides", 1);
	vbc->add_margin_child(TTR("Matches:"), search_options, true);

	get_ok_button()->set_text(TTR("Open"));
	set_hide_on_ok(false);
}

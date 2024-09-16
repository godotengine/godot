/**************************************************************************/
/*  editor_quick_open.cpp                                                 */
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

#include "editor_quick_open.h"

#include "core/os/keyboard.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/themes/editor_scale.h"

Rect2i EditorQuickOpen::prev_rect = Rect2i();
bool EditorQuickOpen::was_showed = false;

void EditorQuickOpen::popup_dialog(const String &p_base, bool p_enable_multi, bool p_dont_clear) {
	base_type = p_base;
	allow_multi_select = p_enable_multi;
	search_options->set_select_mode(allow_multi_select ? Tree::SELECT_MULTI : Tree::SELECT_SINGLE);

	if (was_showed) {
		popup(prev_rect);
	} else {
		popup_centered_clamped(Size2(600, 440) * EDSCALE, 0.8f);
	}

	EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->get_filesystem();
	_build_search_cache(efsd);

	if (p_dont_clear) {
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

	Vector<String> base_types = base_type.split(",");
	for (int i = 0; i < p_efsd->get_file_count(); i++) {
		String file = p_efsd->get_file_path(i);
		String engine_type = p_efsd->get_file_type(i);
		String script_type = p_efsd->get_file_resource_script_class(i);
		String actual_type = script_type.is_empty() ? engine_type : script_type;

		// Iterate all possible base types.
		for (String &parent_type : base_types) {
			if (ClassDB::is_parent_class(engine_type, parent_type) || EditorNode::get_editor_data().script_class_is_parent(script_type, parent_type)) {
				files.push_back(file.substr(6, file.length()));

				// Store refs to used icons.
				String ext = file.get_extension();
				if (!icons.has(ext)) {
					icons.insert(ext, EditorNode::get_singleton()->get_class_icon(actual_type, "Object"));
				}

				// Stop testing base types as soon as we got a match.
				break;
			}
		}
	}
}

void EditorQuickOpen::_update_search() {
	const PackedStringArray search_tokens = search_box->get_text().to_lower().replace("/", " ").split(" ", false);
	const bool empty_search = search_tokens.is_empty();

	// Filter possible candidates.
	Vector<Entry> entries;
	for (int i = 0; i < files.size(); i++) {
		Entry r;
		r.path = files[i];
		if (empty_search) {
			entries.push_back(r);
		} else {
			r.score = _score_search_result(search_tokens, r.path.to_lower());
			if (r.score > 0) {
				entries.push_back(r);
			}
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

		const int class_icon_size = search_options->get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor));
		const int entry_limit = MIN(entries.size(), 300);
		for (int i = 0; i < entry_limit; i++) {
			TreeItem *ti = search_options->create_item(root);
			ti->set_text(0, entries[i].path);
			ti->set_icon_max_width(0, class_icon_size);
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

float EditorQuickOpen::_score_search_result(const PackedStringArray &p_search_tokens, const String &p_path) {
	float score = 0.0f;
	int prev_min_match_idx = -1;

	for (const String &s : p_search_tokens) {
		int min_match_idx = p_path.find(s);

		if (min_match_idx == -1) {
			return 0.0f;
		}

		float token_score = s.length();

		int max_match_idx = p_path.rfind(s);

		// Prioritize the actual file name over folder.
		if (max_match_idx > p_path.rfind("/")) {
			token_score *= 2.0f;
		}

		// Prioritize matches at the front of the path token.
		if (min_match_idx == 0 || p_path.contains("/" + s)) {
			token_score += 1.0f;
		}

		score += token_score;

		// Prioritize tokens which appear in order.
		if (prev_min_match_idx != -1 && max_match_idx > prev_min_match_idx) {
			score += 1.0f;
		}

		prev_min_match_idx = min_match_idx;
	}

	return score;
}

void EditorQuickOpen::_confirmed() {
	if (!search_options->get_selected()) {
		return;
	}
	_cleanup();
	hide();
	emit_signal(SNAME("quick_open"));
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

void EditorQuickOpen::_sbox_input(const Ref<InputEvent> &p_event) {
	// Redirect navigational key events to the tree.
	Ref<InputEventKey> key = p_event;
	if (key.is_valid()) {
		if (key->is_action("ui_up", true) || key->is_action("ui_down", true) || key->is_action("ui_page_up") || key->is_action("ui_page_down")) {
			search_options->gui_input(key);
			search_box->accept_event();

			if (allow_multi_select) {
				TreeItem *root = search_options->get_root();
				if (!root->get_first_child()) {
					return;
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

String EditorQuickOpen::get_base_type() const {
	return base_type;
}

void EditorQuickOpen::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect(SceneStringName(confirmed), callable_mp(this, &EditorQuickOpen::_confirmed));

			search_box->set_clear_button_enabled(true);
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				prev_rect = Rect2i(get_position(), get_size());
				was_showed = true;
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			search_box->set_right_icon(get_editor_theme_icon(SNAME("Search")));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			disconnect(SceneStringName(confirmed), callable_mp(this, &EditorQuickOpen::_confirmed));
		} break;
	}
}

void EditorQuickOpen::_bind_methods() {
	ADD_SIGNAL(MethodInfo("quick_open"));
}

EditorQuickOpen::EditorQuickOpen() {
	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	search_box = memnew(LineEdit);
	search_box->connect(SceneStringName(text_changed), callable_mp(this, &EditorQuickOpen::_text_changed));
	search_box->connect(SceneStringName(gui_input), callable_mp(this, &EditorQuickOpen::_sbox_input));
	vbc->add_margin_child(TTR("Search:"), search_box);
	register_text_enter(search_box);

	search_options = memnew(Tree);
	search_options->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	search_options->connect("item_activated", callable_mp(this, &EditorQuickOpen::_confirmed));
	search_options->create_item();
	search_options->set_hide_root(true);
	search_options->set_hide_folding(true);
	search_options->add_theme_constant_override("draw_guides", 1);
	vbc->add_margin_child(TTR("Matches:"), search_options, true);

	set_ok_button_text(TTR("Open"));
	set_hide_on_ok(false);
}

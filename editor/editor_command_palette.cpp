/*************************************************************************/
/*  editor_command_palette.cpp                                           */
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

#include "editor/editor_command_palette.h"
#include "core/os/keyboard.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "scene/gui/control.h"
#include "scene/gui/tree.h"

EditorCommandPalette *EditorCommandPalette::singleton = nullptr;

float EditorCommandPalette::_score_path(const String &p_search, const String &p_path) {
	float score = 0.9f + .1f * (p_search.length() / (float)p_path.length());

	// Positive bias for matches close to the beginning of the file name.
	int pos = p_path.findn(p_search);
	if (pos != -1) {
		return score * (1.0f - 0.1f * (float(pos) / p_path.length()));
	}

	// Positive bias for matches close to the end of the path.
	pos = p_path.rfindn(p_search);
	if (pos != -1) {
		return score * (0.8f - 0.1f * (float(p_path.length() - pos) / p_path.length()));
	}

	// Remaining results belong to the same class of results.
	return score * 0.69f;
}

void EditorCommandPalette::_update_command_search(const String &search_text) {
	commands.get_key_list(&command_keys);
	ERR_FAIL_COND(command_keys.is_empty());

	Map<String, TreeItem *> sections;
	TreeItem *first_section = nullptr;

	// Filter possible candidates.
	Vector<CommandEntry> entries;
	for (int i = 0; i < command_keys.size(); i++) {
		CommandEntry r;
		r.key_name = command_keys[i];
		r.display_name = commands[r.key_name].name;
		r.shortcut_text = commands[r.key_name].shortcut;
		r.last_used = commands[r.key_name].last_used;

		if (search_text.is_subsequence_ofi(r.display_name)) {
			if (!search_text.is_empty()) {
				r.score = _score_path(search_text, r.display_name.to_lower());
			}

			entries.push_back(r);
		}
	}

	command_keys.clear();

	TreeItem *root = search_options->get_root();
	root->clear_children();

	if (entries.is_empty()) {
		get_ok_button()->set_disabled(true);

		return;
	}

	if (!search_text.is_empty()) {
		SortArray<CommandEntry, CommandEntryComparator> sorter;
		sorter.sort(entries.ptrw(), entries.size());
	} else {
		SortArray<CommandEntry, CommandHistoryComparator> sorter;
		sorter.sort(entries.ptrw(), entries.size());
	}

	const int entry_limit = MIN(entries.size(), 300);
	for (int i = 0; i < entry_limit; i++) {
		String section_name = entries[i].key_name.get_slice("/", 0);
		TreeItem *section;

		if (sections.has(section_name)) {
			section = sections[section_name];
		} else {
			section = search_options->create_item(root);

			if (!first_section) {
				first_section = section;
			}

			String item_name = section_name.capitalize();
			section->set_text(0, item_name);
			section->set_selectable(0, false);
			section->set_selectable(1, false);
			section->set_custom_bg_color(0, search_options->get_theme_color(SNAME("prop_subsection"), SNAME("Editor")));
			section->set_custom_bg_color(1, search_options->get_theme_color(SNAME("prop_subsection"), SNAME("Editor")));

			sections[section_name] = section;
		}

		TreeItem *ti = search_options->create_item(section);
		String shortcut_text = entries[i].shortcut_text == "None" ? "" : entries[i].shortcut_text;
		ti->set_text(0, entries[i].display_name);
		ti->set_metadata(0, entries[i].key_name);
		ti->set_text_alignment(1, HORIZONTAL_ALIGNMENT_RIGHT);
		ti->set_text(1, shortcut_text);
		Color c = Color(1, 1, 1, 0.5);
		ti->set_custom_color(1, c);
	}

	TreeItem *to_select = first_section->get_first_child();
	to_select->select(0);
	to_select->set_as_cursor(0);
	search_options->ensure_cursor_is_visible();
}

void EditorCommandPalette::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_command", "command_name", "key_name", "binded_callable", "shortcut_text"), &EditorCommandPalette::_add_command, DEFVAL("None"));
	ClassDB::bind_method(D_METHOD("remove_command", "key_name"), &EditorCommandPalette::remove_command);
}

void EditorCommandPalette::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;
	if (k.is_valid()) {
		switch (k->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				search_options->gui_input(k);
			} break;
			default:
				break;
		}
	}
}

void EditorCommandPalette::_confirmed() {
	TreeItem *selected_option = search_options->get_selected();
	String command_key = selected_option != nullptr ? selected_option->get_metadata(0) : "";
	if (!command_key.is_empty()) {
		hide();
		execute_command(command_key);
	}
}

void EditorCommandPalette::open_popup() {
	popup_centered_clamped(Size2i(600, 440), 0.8f);

	command_search_box->clear();
	command_search_box->grab_focus();

	search_options->scroll_to_item(search_options->get_root());
}

void EditorCommandPalette::get_actions_list(List<String> *p_list) const {
	commands.get_key_list(p_list);
}

void EditorCommandPalette::remove_command(String p_key_name) {
	ERR_FAIL_COND_MSG(!commands.has(p_key_name), "The Command '" + String(p_key_name) + "' doesn't exists. Unable to remove it.");

	commands.erase(p_key_name);
}

void EditorCommandPalette::add_command(String p_command_name, String p_key_name, Callable p_action, Vector<Variant> arguments, String p_shortcut_text) {
	ERR_FAIL_COND_MSG(commands.has(p_key_name), "The Command '" + String(p_command_name) + "' already exists. Unable to add it.");

	const Variant **argptrs = (const Variant **)alloca(sizeof(Variant *) * arguments.size());
	for (int i = 0; i < arguments.size(); i++) {
		argptrs[i] = &arguments[i];
	}
	Command command;
	command.name = p_command_name;
	command.callable = p_action.bind(argptrs, arguments.size());
	command.shortcut = p_shortcut_text;

	commands[p_key_name] = command;
}

void EditorCommandPalette::_add_command(String p_command_name, String p_key_name, Callable p_binded_action, String p_shortcut_text) {
	ERR_FAIL_COND_MSG(commands.has(p_key_name), "The Command '" + String(p_command_name) + "' already exists. Unable to add it.");

	Command command;
	command.name = p_command_name;
	command.callable = p_binded_action;
	command.shortcut = p_shortcut_text;

	// Commands added from plugins don't exist yet when the history is loaded, so we assign the last use time here if it was recorded.
	Dictionary command_history = EditorSettings::get_singleton()->get_project_metadata("command_palette", "command_history", Dictionary());
	if (command_history.has(p_key_name)) {
		command.last_used = command_history[p_key_name];
	}

	commands[p_key_name] = command;
}

void EditorCommandPalette::execute_command(String &p_command_key) {
	ERR_FAIL_COND_MSG(!commands.has(p_command_key), p_command_key + " not found.");
	commands[p_command_key].last_used = OS::get_singleton()->get_unix_time();
	commands[p_command_key].callable.call_deferred(nullptr, 0);
	_save_history();
}

void EditorCommandPalette::register_shortcuts_as_command() {
	const String *key = nullptr;
	key = unregistered_shortcuts.next(key);
	while (key != nullptr) {
		String command_name = unregistered_shortcuts[*key].first;
		Ref<Shortcut> shortcut = unregistered_shortcuts[*key].second;
		Ref<InputEventShortcut> ev;
		ev.instantiate();
		ev->set_shortcut(shortcut);
		String shortcut_text = String(shortcut->get_as_text());
		add_command(command_name, *key, callable_mp(EditorNode::get_singleton()->get_viewport(), &Viewport::push_unhandled_input), varray(ev, false), shortcut_text);
		key = unregistered_shortcuts.next(key);
	}
	unregistered_shortcuts.clear();

	// Load command use history.
	Dictionary command_history = EditorSettings::get_singleton()->get_project_metadata("command_palette", "command_history", Dictionary());
	Array history_entries = command_history.keys();
	for (int i = 0; i < history_entries.size(); i++) {
		const String &history_key = history_entries[i];
		if (commands.has(history_key)) {
			commands[history_key].last_used = command_history[history_key];
		}
	}
}

Ref<Shortcut> EditorCommandPalette::add_shortcut_command(const String &p_command, const String &p_key, Ref<Shortcut> p_shortcut) {
	if (is_inside_tree()) {
		Ref<InputEventShortcut> ev;
		ev.instantiate();
		ev->set_shortcut(p_shortcut);
		String shortcut_text = String(p_shortcut->get_as_text());
		add_command(p_command, p_key, callable_mp(EditorNode::get_singleton()->get_viewport(), &Viewport::push_unhandled_input), varray(ev, false), shortcut_text);
	} else {
		const String key_name = String(p_key);
		const String command_name = String(p_command);
		Pair pair = Pair(command_name, p_shortcut);
		unregistered_shortcuts[key_name] = pair;
	}
	return p_shortcut;
}

void EditorCommandPalette::_theme_changed() {
	command_search_box->set_right_icon(search_options->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
}

void EditorCommandPalette::_save_history() const {
	Dictionary command_history;
	List<String> command_keys;
	commands.get_key_list(&command_keys);

	for (const String &key : command_keys) {
		if (commands[key].last_used > 0) {
			command_history[key] = commands[key].last_used;
		}
	}
	EditorSettings::get_singleton()->set_project_metadata("command_palette", "command_history", command_history);
}

EditorCommandPalette *EditorCommandPalette::get_singleton() {
	if (singleton == nullptr) {
		singleton = memnew(EditorCommandPalette);
	}
	return singleton;
}

EditorCommandPalette::EditorCommandPalette() {
	set_hide_on_ok(false);
	connect("confirmed", callable_mp(this, &EditorCommandPalette::_confirmed));

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->connect("theme_changed", callable_mp(this, &EditorCommandPalette::_theme_changed));
	add_child(vbc);

	command_search_box = memnew(LineEdit);
	command_search_box->set_placeholder(TTR("Filter commands"));
	command_search_box->connect("gui_input", callable_mp(this, &EditorCommandPalette::_sbox_input));
	command_search_box->connect("text_changed", callable_mp(this, &EditorCommandPalette::_update_command_search));
	command_search_box->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	command_search_box->set_clear_button_enabled(true);
	MarginContainer *margin_container_csb = memnew(MarginContainer);
	margin_container_csb->add_child(command_search_box);
	vbc->add_child(margin_container_csb);
	register_text_enter(command_search_box);

	search_options = memnew(Tree);
	search_options->connect("item_activated", callable_mp(this, &EditorCommandPalette::_confirmed));
	search_options->connect("item_selected", callable_mp((BaseButton *)get_ok_button(), &BaseButton::set_disabled), varray(false));
	search_options->connect("nothing_selected", callable_mp((BaseButton *)get_ok_button(), &BaseButton::set_disabled), varray(true));
	search_options->create_item();
	search_options->set_hide_root(true);
	search_options->set_columns(2);
	search_options->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	search_options->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_options->set_column_custom_minimum_width(0, int(8 * EDSCALE));
	vbc->add_child(search_options, true);
}

Ref<Shortcut> ED_SHORTCUT_AND_COMMAND(const String &p_path, const String &p_name, Key p_keycode, String p_command_name) {
	if (p_command_name.is_empty()) {
		p_command_name = p_name;
	}

	Ref<Shortcut> shortcut = ED_SHORTCUT(p_path, p_name, p_keycode);
	EditorCommandPalette::get_singleton()->add_shortcut_command(p_command_name, p_path, shortcut);
	return shortcut;
}

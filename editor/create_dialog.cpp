/*************************************************************************/
/*  create_dialog.cpp                                                    */
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

#include "create_dialog.h"

#include "core/object/class_db.h"
#include "core/os/keyboard.h"
#include "editor_feature_profile.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "editor_settings.h"

bool FavoriteList::toggle_favorite(const String &p_type) {
	favorites_changed = true;

	const bool already_favorited = favorites.has(p_type);
	if (!already_favorited) {
		favorites.push_back(p_type);
	} else {
		favorites.erase(p_type);
	}

	_update_tree();

	return already_favorited;
}

void FavoriteList::load_favorites(const String &p_file_id, const String &p_icon_fallback) {
	icon_fallback = p_icon_fallback;

	String dir = EditorSettings::get_singleton()->get_project_settings_dir();

	FileAccess *f = FileAccess::open(dir.plus_file("favorites." + p_file_id), FileAccess::READ);
	if (f) {
		while (!f->eof_reached()) {
			String l = f->get_line().strip_edges();

			if (l != String()) {
				favorites.push_back(l);
			}
		}

		memdelete(f);
	}

	_update_tree();
}

bool FavoriteList::save_favorites(const String &p_file_id) {
	if (favorites_changed) {
		FileAccess *f = FileAccess::open(EditorSettings::get_singleton()->get_project_settings_dir().plus_file("favorites." + p_file_id), FileAccess::WRITE);
		if (f) {
			for (int i = 0; i < favorites.size(); i++) {
				String favorite = favorites[i];
				String name = favorite.get_slicec(' ', 0);
				if (!(ClassDB::class_exists(name) || ScriptServer::is_global_class(name))) {
					continue;
				}
				f->store_line(favorite);

				if (EditorFeatureProfileManager::get_singleton()->is_class_disabled(name)) {
					continue;
				}
			}
			memdelete(f);
		}
	}

	favorites.clear();
	clear();
	return favorites_changed;
}

void FavoriteList::_update_tree() {
	clear();
	TreeItem *root = create_item();

	for (int i = 0; i < favorites.size(); i++) {
		String favorite = favorites[i];
		String name = favorite.get_slicec(' ', 0);
		if (!(ClassDB::class_exists(name) || ScriptServer::is_global_class(name))) {
			continue;
		}

		if (EditorFeatureProfileManager::get_singleton()->is_class_disabled(name)) {
			continue;
		}

		TreeItem *ti = create_item(root);
		ti->set_text(0, favorite);
		ti->set_icon(0, EditorNode::get_singleton()->get_class_icon(name, icon_fallback));
	}
}

Variant FavoriteList::_get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	TreeItem *ti = get_item_at_position(p_point);
	if (ti) {
		Dictionary d;
		d["type"] = "create_favorite_drag";
		d["class"] = ti->get_text(0);

		Button *tb = memnew(Button);
		tb->set_flat(true);
		tb->set_icon(ti->get_icon(0));
		tb->set_text(ti->get_text(0));
		set_drag_preview(tb);

		return d;
	}

	return Variant();
}

bool FavoriteList::_can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	Dictionary d = p_data;
	if (d.has("type") && String(d["type"]) == "create_favorite_drag") {
		set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN);
		return true;
	}

	return false;
}

void FavoriteList::_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	Dictionary d = p_data;

	TreeItem *ti = get_item_at_position(p_point);
	if (!ti) {
		return;
	}

	String drop_at = ti->get_text(0);
	int ds = get_drop_section_at_position(p_point);

	int drop_idx = favorites.find(drop_at);
	if (drop_idx < 0) {
		return;
	}

	String type = d["class"];

	int from_idx = favorites.find(type);
	if (from_idx < 0) {
		return;
	}

	if (drop_idx == from_idx) {
		ds = -1; //cause it will be gone
	} else if (drop_idx > from_idx) {
		drop_idx--;
	}

	favorites.remove_at(from_idx);

	if (ds < 0) {
		favorites.insert(drop_idx, type);
	} else {
		if (drop_idx >= favorites.size() - 1) {
			favorites.push_back(type);
		} else {
			favorites.insert(drop_idx + 1, type);
		}
	}

	_update_tree();
}

void FavoriteList::_bind_methods() {
	ClassDB::bind_method("_get_drag_data_fw", &FavoriteList::_get_drag_data_fw);
	ClassDB::bind_method("_can_drop_data_fw", &FavoriteList::_can_drop_data_fw);
	ClassDB::bind_method("_drop_data_fw", &FavoriteList::_drop_data_fw);
}

FavoriteList::FavoriteList() {
	set_hide_root(true);
	set_hide_folding(true);
	set_allow_reselect(true);
	add_theme_constant_override("draw_guides", 1);
#ifndef _MSC_VER
#warning cannot forward drag data to a non control, must be fixed
#endif
	//favorite_tree->set_drag_forwarding(this);
}

void HistoryList::save_to_history(const String &p_file_id, const String &p_item) {
	FileAccess *f = FileAccess::open(EditorSettings::get_singleton()->get_project_settings_dir().plus_file("create_recent." + p_file_id), FileAccess::WRITE);
	if (f) {
		f->store_line(p_item);

		for (int i = 0; i < MIN(32, get_item_count()); i++) {
			if (get_item_text(i) != p_item) {
				f->store_line(get_item_text(i));
			}
		}

		memdelete(f);
	}
}

void HistoryList::load_history(const String &p_file_id, const String &p_icon_fallback) {
	String dir = EditorSettings::get_singleton()->get_project_settings_dir();
	FileAccess *f = FileAccess::open(dir.plus_file("create_recent." + p_file_id), FileAccess::READ);
	if (f) {
		while (!f->eof_reached()) {
			String l = f->get_line().strip_edges();
			String name = l.get_slicec(' ', 0);

			if ((ClassDB::class_exists(name) || ScriptServer::is_global_class(name)) && !EditorFeatureProfileManager::get_singleton()->is_class_disabled(name)) {
				history.insert(l);
				add_item(l, EditorNode::get_singleton()->get_class_icon(name, p_icon_fallback));
			}
		}

		memdelete(f);
	}
}

void HistoryList::clear_history() {
	history.clear();
	clear();
}

HistoryList::HistoryList() {
	set_allow_reselect(true);
	add_theme_constant_override("draw_guides", 1);
}

struct CandidateAlphComparator {
	NoCaseComparator compare;

	bool operator()(const CreateDialogCandidate &p_a, const CreateDialogCandidate &p_b) const {
		return compare(p_a.get_type(), p_b.get_type());
	}
};

float CreateDialogCandidate::_word_score(const String &p_word, const String &p_query) const {
	const int start_pos = p_word.findn(p_query);
	const float inverse_length = 1.f / float(p_word.length());

	// Favor shorter items: they resemble the search term more.
	float w = 0.1f;
	float score = (1.0f - w) + w * (p_query.length() * inverse_length);

	// Favor types where search term is a substring close to the start of the word.
	w = 0.5f;
	score *= (start_pos > -1) ? 1.0f - w * MIN(1.0, 3.0 * start_pos * inverse_length) : MAX(0.f, .9f - w);

	return score;
}

float CreateDialogCandidate::compute_score(const String &p_query) const {
	// Custom types include the file name in their path.
	const String _type = type.get_slicec(' ', 0);
	float score = _word_score(_type, p_query);
	if (score == 1.0) {
		return score;
	}

	score *= _word_score(wb_chars, p_query);

	score *= is_preferred_type ? 1.0f : 0.6f;
	score *= in_favorites ? 1.0f : 0.7f;
	score *= in_recent ? 1.0f : 0.8f;

	return score;
}

bool CreateDialogCandidate::is_valid(const String &p_query) const {
	return type.findn(p_query) > -1 || p_query.is_subsequence_ofi(wb_chars);
}

String CreateDialogCandidate::_compute_word_boundary_characters() const {
	Vector<char32_t> wb_chars;

	const char32_t *src = type.get_data();
	const int len = type.length();

	for (int i = 0; i < len; i++) {
		char32_t c = src[i];

		if (is_upper_case(c) || is_digit(c)) {
			wb_chars.push_back(c);
		}
	}

	wb_chars.push_back(0); // Terminate string.

	return String(wb_chars.ptr());
}

CreateDialogCandidate::CreateDialogCandidate(const String &p_type, const bool p_is_preferred_type, const bool p_in_favorites, const bool p_in_recent) {
	type = p_type;
	wb_chars = _compute_word_boundary_characters();
	is_preferred_type = p_is_preferred_type;
	in_favorites = p_in_favorites;
	in_recent = p_in_recent;
}

void CreateDialog::popup_create(bool p_dont_clear, bool p_replace_mode, const String &p_select_type) {
	favorite_list->load_favorites(base_type, icon_fallback);
	history_list->load_history(base_type, icon_fallback);

	candidates = _compute_candidates();

	icon_fallback = result_tree->has_theme_icon(base_type, SNAME("EditorIcons")) ? base_type : "Object";

	if (p_dont_clear) {
		search_box->select_all();
	} else {
		search_box->clear();
	}

	if (p_replace_mode) {
		search_box->set_text(p_select_type);
	}

	search_box->grab_focus();
	_update_result_tree();

	if (p_replace_mode) {
		set_title(vformat(TTR("Change %s Type"), base_type));
		get_ok_button()->set_text(TTR("Change"));
	} else {
		set_title(vformat(TTR("Create New %s"), base_type));
		get_ok_button()->set_text(TTR("Create"));
	}

	// Restore valid window bounds or pop up at default size.
	Rect2 saved_size = EditorSettings::get_singleton()->get_project_metadata("dialog_bounds", "create_new_node", Rect2());
	if (saved_size != Rect2()) {
		popup(saved_size);
	} else {
		popup_centered_clamped(Size2(900, 700) * EDSCALE, 0.8);
	}
}

String CreateDialog::get_selected_type() {
	TreeItem *selected = result_tree->get_selected();
	return selected ? selected->get_text(0) : String();
}

Variant CreateDialog::instance_selected() {
	TreeItem *selected = result_tree->get_selected();

	if (!selected) {
		return Variant();
	}

	Variant md = selected->get_metadata(0);
	Variant obj;
	if (md.get_type() != Variant::NIL) {
		String custom = md;
		if (ScriptServer::is_global_class(custom)) {
			obj = EditorNode::get_editor_data().script_class_instance(custom);
			Node *n = Object::cast_to<Node>(obj);
			if (n) {
				n->set_name(custom);
			}
		} else {
			obj = EditorNode::get_editor_data().instance_custom_type(selected->get_text(0), custom);
		}
	} else {
		obj = ClassDB::instantiate(selected->get_text(0));
	}

	// Check if any Object-type property should be instantiated.
	List<PropertyInfo> pinfo;
	((Object *)obj)->get_property_list(&pinfo);

	for (const PropertyInfo &pi : pinfo) {
		if (pi.type == Variant::OBJECT && pi.usage & PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT) {
			Object *prop = ClassDB::instantiate(pi.class_name);
			((Object *)obj)->set(pi.name, prop);
		}
	}

	return obj;
}

Vector<CreateDialogCandidate> CreateDialog::_compute_candidates() {
	List<StringName> complete_type_list;
	ClassDB::get_class_list(&complete_type_list);
	ScriptServer::get_global_class_list(&complete_type_list);

	EditorData &ed = EditorNode::get_editor_data();

	Vector<CreateDialogCandidate> candidates;

	for (List<StringName>::Element *I = complete_type_list.front(); I; I = I->next()) {
		String type = I->get();
		if (!_should_hide_type(type)) {
			const bool is_preferred_type = ClassDB::class_exists(type) ? ClassDB::is_parent_class(type, preferred_search_result_type) : ed.script_class_is_parent(type, preferred_search_result_type);

			candidates.push_back(CreateDialogCandidate(type, is_preferred_type, favorite_list->has_favorite(type), history_list->has_history(type)));

			if (ed.get_custom_types().has(type)) {
				const Vector<EditorData::CustomType> &custom_types = ed.get_custom_types()[type];
				for (int i = 0; i < custom_types.size(); i++) {
					const String custom_type = custom_types[i].name;
					candidates.push_back(CreateDialogCandidate(custom_type, is_preferred_type, favorite_list->has_favorite(custom_type), history_list->has_history(custom_type)));

					custom_type_parents[custom_type] = type;
					custom_type_indices[custom_type] = i;
				}
			}
		}
	}
	candidates.sort_custom<CandidateAlphComparator>();
	return candidates;
}

bool CreateDialog::_should_hide_type(const String &p_type) const {
	if (EditorFeatureProfileManager::get_singleton()->is_class_disabled(p_type)) {
		return true;
	}

	if (base_type == "Node" && p_type.begins_with("Editor")) {
		return true; // Do not show editor nodes.
	}

	if (ClassDB::class_exists(p_type)) {
		if (!ClassDB::can_instantiate(p_type)) {
			return true; // Can't create abstract class.
		}

		if (!ClassDB::is_parent_class(p_type, base_type)) {
			return true; // Wrong inheritance.
		}

		for (Set<StringName>::Element *E = blacklisted_types.front(); E; E = E->next()) {
			if (ClassDB::is_parent_class(p_type, E->get())) {
				return true; // Parent type is blacklisted.
			}
		}
	} else {
		if (!EditorNode::get_editor_data().script_class_is_parent(p_type, base_type)) {
			return true; // Wrong inheritance.
		}
		if (!ScriptServer::is_global_class(p_type)) {
			return true;
		}

		String script_path = ScriptServer::get_global_class_path(p_type);
		if (script_path.begins_with("res://addons/")) {
			if (!EditorNode::get_singleton()->is_addon_plugin_enabled(script_path.get_slicec('/', 3))) {
				return true; // Plugin is not enabled.
			}
		}
	}

	return false;
}

void CreateDialog::_update_result_tree() {
	result_tree->clear();
	result_tree_types.clear();

	const String search_text = search_box->get_text();
	bool empty_search = search_text.is_empty();

	Vector<CreateDialogCandidate> valid_candidates;
	for (int i = 0; i < candidates.size(); i++) {
		CreateDialogCandidate c = candidates[i];
		if (empty_search || c.is_valid(search_text)) {
			valid_candidates.push_back(c);
		}
	}

	if (valid_candidates.size() > 0) {
		// Build the type tree and select the best result.
		result_tree_types[base_type] = _create_type(base_type, nullptr, true);
		for (int i = 0; i < valid_candidates.size(); i++) {
			const String type = valid_candidates[i].get_type();
			_add_type(type, ClassDB::class_exists(type));
		}

		_select_type(empty_search ? base_type : _top_result(valid_candidates, search_text));
	} else {
		result_tree->deselect_all();
		favorite->set_disabled(true);
		help_bit->set_text(vformat(TTR("No results for \"%s\"."), search_text));
		help_bit->get_rich_text()->set_self_modulate(Color(1, 1, 1, 0.5));

		get_ok_button()->set_disabled(true);
	}
}

void CreateDialog::_add_type(const String &p_type, bool p_cpp_type) {
	if (result_tree_types.has(p_type)) {
		return;
	}

	String parent_type;
	if (p_cpp_type) {
		parent_type = ClassDB::get_parent_class(p_type);
	} else if (ScriptServer::is_global_class(p_type)) {
		parent_type = EditorNode::get_editor_data().script_class_get_base(p_type);
	} else {
		parent_type = custom_type_parents[p_type];
	}

	// First create a branch to the root type.
	_add_type(parent_type, p_cpp_type || ClassDB::class_exists(parent_type));

	// Once a branch has been constructed, add the type as a child.
	result_tree_types[p_type] = _create_type(p_type, result_tree_types[parent_type], p_cpp_type);
}

TreeItem *CreateDialog::_create_type(const String &p_type, TreeItem *p_parent_type, const bool p_cpp_type) {
	TreeItem *item = result_tree->create_item(p_parent_type);

	bool script_type = ScriptServer::is_global_class(p_type);
	if (p_cpp_type) {
		item->set_text(0, p_type);
	} else if (script_type) {
		item->set_metadata(0, p_type);
		item->set_text(0, p_type + " (" + ScriptServer::get_global_class_path(p_type).get_file() + ")");
	} else {
		item->set_metadata(0, custom_type_parents[p_type]);
		item->set_text(0, p_type);
	}

	bool can_instantiate = (p_cpp_type && ClassDB::can_instantiate(p_type)) || !p_cpp_type;
	if (!can_instantiate) {
		item->set_custom_color(0, result_tree->get_theme_color(SNAME("disabled_font_color"), SNAME("Editor")));
		item->set_icon(0, EditorNode::get_singleton()->get_class_icon(p_type, "NodeDisabled"));
		item->set_selectable(0, false);
	} else {
		item->set_icon(0, EditorNode::get_singleton()->get_class_icon(p_type, icon_fallback));
	}

	if (!search_box->get_text().is_empty()) {
		item->set_collapsed(false);
	} else {
		// Don't collapse the root node or an abstract node on the first tree level.
		bool should_collapse = p_type != base_type && (item->get_parent()->get_text(0) != base_type || can_instantiate);

		if (should_collapse && bool(EditorSettings::get_singleton()->get("docks/scene_tree/start_create_dialog_fully_expanded"))) {
			should_collapse = false;
		}
		item->set_collapsed(should_collapse);
	}

	const String &description = DTR(EditorHelp::get_doc_data()->class_list[p_type].brief_description);
	item->set_tooltip(0, description);

	if (!p_cpp_type && !script_type) {
		Ref<Texture2D> icon = EditorNode::get_editor_data().get_custom_types()[custom_type_parents[p_type]][custom_type_indices[p_type]].icon;
		if (icon.is_valid()) {
			item->set_icon(0, icon);
		}
	}

	return item;
}

String CreateDialog::_top_result(const Vector<CreateDialogCandidate> p_candidates, const String &p_query) const {
	float highest_score = 0;
	int highest_index = 0;
	for (int i = 0; i < p_candidates.size(); i++) {
		float score = p_candidates[i].compute_score(p_query);
		if (score > highest_score) {
			highest_score = score;
			highest_index = i;
			if (highest_score == 1.0f) {
				break; // Cannot find a better match.
			}
		}
	}

	return p_candidates[highest_index].get_type();
}

void CreateDialog::_confirmed() {
	String selected_item = get_selected_type();
	if (selected_item.is_empty()) {
		return;
	}

	history_list->save_to_history(base_type, selected_item);

	// To prevent, emitting an error from the transient window (shader dialog for example) hide this dialog before emitting the "create" signal.
	hide();

	emit_signal(SNAME("create"));
	_cleanup();
}

void CreateDialog::_text_changed(const String &p_newtext) {
	_update_result_tree();
}

void CreateDialog::_search_box_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;
	if (k.is_valid()) {
		switch (k->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				result_tree->gui_input(k);
				search_box->accept_event();
			} break;
			default:
				break;
		}
	}
}

void CreateDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect("confirmed", callable_mp(this, &CreateDialog::_confirmed));
			search_box->set_right_icon(result_tree->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
			search_box->set_clear_button_enabled(true);
			favorite->set_icon(result_tree->get_theme_icon(SNAME("Favorites"), SNAME("EditorIcons")));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			disconnect("confirmed", callable_mp(this, &CreateDialog::_confirmed));
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				search_box->call_deferred(SNAME("grab_focus")); // still not visible
				search_box->select_all();
			} else {
				EditorSettings::get_singleton()->get_project_metadata("dialog_bounds", "create_new_node", Rect2(get_position(), get_size()));
			}
		} break;
	}
}

void CreateDialog::_select_type(const String &p_type) {
	if (!result_tree_types.has(p_type)) {
		return;
	}

	TreeItem *to_select = result_tree_types[p_type];
	to_select->select(0);
	result_tree->scroll_to_item(to_select);

	if (EditorHelp::get_doc_data()->class_list.has(p_type) && !DTR(EditorHelp::get_doc_data()->class_list[p_type].brief_description).is_empty()) {
		// Display both class name and description, since the help bit may be displayed
		// far away from the location (especially if the dialog was resized to be taller).
		help_bit->set_text(vformat("[b]%s[/b]: %s", p_type, DTR(EditorHelp::get_doc_data()->class_list[p_type].brief_description)));
		help_bit->get_rich_text()->set_self_modulate(Color(1, 1, 1, 1));
	} else {
		// Use nested `vformat()` as translators shouldn't interfere with BBCode tags.
		help_bit->set_text(vformat(TTR("No description available for %s."), vformat("[b]%s[/b]", p_type)));
		help_bit->get_rich_text()->set_self_modulate(Color(1, 1, 1, 0.5));
	}

	favorite->set_disabled(false);
	favorite->set_pressed(favorite_list->has_favorite(p_type));
	get_ok_button()->set_disabled(false);
}

void CreateDialog::_item_selected() {
	_select_type(get_selected_type());
}

void CreateDialog::_hide_requested() {
	_cancel_pressed(); // From AcceptDialog.
}

void CreateDialog::cancel_pressed() {
	_cleanup();
}

void CreateDialog::_favorite_toggled() {
	TreeItem *item = result_tree->get_selected();
	if (!item) {
		return;
	}

	favorite->set_pressed(favorite_list->toggle_favorite(item->get_text(0)));
}

void CreateDialog::_history_selected(int p_idx) {
	search_box->set_text(history_list->get_item_text(p_idx).get_slicec(' ', 0));
	favorite_list->deselect_all();
	_update_result_tree();
}

void CreateDialog::_favorite_selected() {
	TreeItem *item = favorite_list->get_selected();
	if (!item) {
		return;
	}

	search_box->set_text(item->get_text(0).get_slicec(' ', 0));
	history_list->deselect_all();
	_update_result_tree();
}

void CreateDialog::_history_activated(int p_idx) {
	_history_selected(p_idx);
	_confirmed();
}

void CreateDialog::_favorite_activated() {
	_favorite_selected();
	_confirmed();
}

void CreateDialog::_cleanup() {
	candidates.clear();
	custom_type_parents.clear();
	custom_type_indices.clear();

	bool favorites_changed = favorite_list->save_favorites(base_type);
	if (favorites_changed) {
		emit_signal(SNAME("favorites_updated"));
	}

	history_list->clear_history();
}

void CreateDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("create"));
	ADD_SIGNAL(MethodInfo("favorites_updated"));
}

CreateDialog::CreateDialog() {
	base_type = "Object";
	preferred_search_result_type = "";

	blacklisted_types.insert("PluginScript"); // PluginScript must be initialized before use, which is not possible here.
	blacklisted_types.insert("ScriptCreateDialog"); // This is an exposed editor Node that doesn't have an Editor prefix.

	HSplitContainer *hsc = memnew(HSplitContainer);
	add_child(hsc);

	VSplitContainer *vsc = memnew(VSplitContainer);
	hsc->add_child(vsc);

	{
		VBoxContainer *fav_vb = memnew(VBoxContainer);
		fav_vb->set_custom_minimum_size(Size2(150, 100) * EDSCALE);
		fav_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		vsc->add_child(fav_vb);

		favorite_list = memnew(FavoriteList);
		favorite_list->connect("cell_selected", callable_mp(this, &CreateDialog::_favorite_selected));
		favorite_list->connect("item_activated", callable_mp(this, &CreateDialog::_favorite_activated));
		fav_vb->add_margin_child(TTR("Favorites:"), favorite_list, true);
	}

	{
		VBoxContainer *rec_vb = memnew(VBoxContainer);
		vsc->add_child(rec_vb);
		rec_vb->set_custom_minimum_size(Size2(150, 100) * EDSCALE);
		rec_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);

		history_list = memnew(HistoryList);
		history_list->connect("item_selected", callable_mp(this, &CreateDialog::_history_selected));
		history_list->connect("item_activated", callable_mp(this, &CreateDialog::_history_activated));
		rec_vb->add_margin_child(TTR("Recent:"), history_list, true);
	}

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_custom_minimum_size(Size2(300, 0) * EDSCALE);
	vbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hsc->add_child(vbc);

	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_box->connect("text_changed", callable_mp(this, &CreateDialog::_text_changed));
	search_box->connect("gui_input", callable_mp(this, &CreateDialog::_search_box_input));

	HBoxContainer *search_hb = memnew(HBoxContainer);
	search_hb->add_child(search_box);

	favorite = memnew(Button);
	favorite->set_toggle_mode(true);
	favorite->set_tooltip(TTR("(Un)favorite selected item."));
	favorite->connect("pressed", callable_mp(this, &CreateDialog::_favorite_toggled));
	search_hb->add_child(favorite);
	vbc->add_margin_child(TTR("Search:"), search_hb);

	result_tree = memnew(Tree);
	result_tree->connect("item_activated", callable_mp(this, &CreateDialog::_confirmed));
	result_tree->connect("cell_selected", callable_mp(this, &CreateDialog::_item_selected));
	vbc->add_margin_child(TTR("Matches:"), result_tree, true);

	help_bit = memnew(EditorHelpBit);
	help_bit->connect("request_hide", callable_mp(this, &CreateDialog::_hide_requested));
	vbc->add_margin_child(TTR("Description:"), help_bit);

	register_text_enter(search_box);
	set_hide_on_ok(false);
}

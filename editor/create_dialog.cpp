/*************************************************************************/
/*  create_dialog.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

void CreateDialog::popup_create(bool p_dont_clear, bool p_replace_mode, const String &p_select_type) {
	_fill_type_list();

	icon_fallback = search_options->has_theme_icon(base_type, SNAME("EditorIcons")) ? base_type : "Object";

	if (p_dont_clear) {
		search_box->select_all();
	} else {
		search_box->clear();
	}

	if (p_replace_mode) {
		search_box->set_text(p_select_type);
	}

	search_box->grab_focus();
	_update_search();

	if (p_replace_mode) {
		set_title(vformat(TTR("Change %s Type"), base_type));
		get_ok_button()->set_text(TTR("Change"));
	} else {
		set_title(vformat(TTR("Create New %s"), base_type));
		get_ok_button()->set_text(TTR("Create"));
	}

	_load_favorites_and_history();
	_save_and_update_favorite_list();

	// Restore valid window bounds or pop up at default size.
	Rect2 saved_size = EditorSettings::get_singleton()->get_project_metadata("dialog_bounds", "create_new_node", Rect2());
	if (saved_size != Rect2()) {
		popup(saved_size);
	} else {
		popup_centered_clamped(Size2(900, 700) * EDSCALE, 0.8);
	}
}

void CreateDialog::_fill_type_list() {
	List<StringName> complete_type_list;
	ClassDB::get_class_list(&complete_type_list);
	ScriptServer::get_global_class_list(&complete_type_list);

	EditorData &ed = EditorNode::get_editor_data();

	for (List<StringName>::Element *I = complete_type_list.front(); I; I = I->next()) {
		String type = I->get();
		if (!_should_hide_type(type)) {
			type_list.push_back(type);

			if (!ed.get_custom_types().has(type)) {
				continue;
			}

			const Vector<EditorData::CustomType> &ct = ed.get_custom_types()[type];
			for (int i = 0; i < ct.size(); i++) {
				custom_type_parents[ct[i].name] = type;
				custom_type_indices[ct[i].name] = i;
				type_list.push_back(ct[i].name);
			}
		}
	}
	type_list.sort_custom<StringName::AlphCompare>();
}

bool CreateDialog::_is_type_preferred(const String &p_type) const {
	if (ClassDB::class_exists(p_type)) {
		return ClassDB::is_parent_class(p_type, preferred_search_result_type);
	}

	return EditorNode::get_editor_data().script_class_is_parent(p_type, preferred_search_result_type);
}

bool CreateDialog::_is_class_disabled_by_feature_profile(const StringName &p_class) const {
	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();

	return !profile.is_null() && profile->is_class_disabled(p_class);
}

bool CreateDialog::_should_hide_type(const String &p_type) const {
	if (_is_class_disabled_by_feature_profile(p_type)) {
		return true;
	}

	if (base_type == "Node" && p_type.begins_with("Editor")) {
		return true; // Do not show editor nodes.
	}

	if (p_type == base_type) {
		return true; // Root is already added.
	}

	if (ClassDB::class_exists(p_type)) {
		if (!ClassDB::can_instantiate(p_type)) {
			return true; // Can't create abstract class.
		}

		if (!ClassDB::is_parent_class(p_type, base_type)) {
			return true; // Wrong inheritance.
		}

		for (Set<StringName>::Element *E = type_blacklist.front(); E; E = E->next()) {
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

void CreateDialog::_update_search() {
	search_options->clear();
	search_options_types.clear();

	TreeItem *root = search_options->create_item();
	root->set_text(0, base_type);
	root->set_icon(0, search_options->get_theme_icon(icon_fallback, SNAME("EditorIcons")));
	search_options_types[base_type] = root;
	_configure_search_option_item(root, base_type, ClassDB::class_exists(base_type) ? TypeCategory::CPP_TYPE : TypeCategory::OTHER_TYPE);

	const String search_text = search_box->get_text();
	bool empty_search = search_text.is_empty();

	// Filter all candidate results.
	Vector<String> candidates;
	for (List<StringName>::Element *I = type_list.front(); I; I = I->next()) {
		if (empty_search || search_text.is_subsequence_ofi(I->get())) {
			candidates.push_back(I->get());
		}
	}

	// Build the type tree.
	for (int i = 0; i < candidates.size(); i++) {
		_add_type(candidates[i], ClassDB::class_exists(candidates[i]) ? TypeCategory::CPP_TYPE : TypeCategory::OTHER_TYPE);
	}

	// Select the best result.
	if (empty_search) {
		select_type(base_type);
	} else if (candidates.size() > 0) {
		select_type(_top_result(candidates, search_text));
	} else {
		favorite->set_disabled(true);
		help_bit->set_text(vformat(TTR("No results for \"%s\"."), search_text));
		help_bit->get_rich_text()->set_self_modulate(Color(1, 1, 1, 0.5));
		get_ok_button()->set_disabled(true);
		search_options->deselect_all();
	}
}

void CreateDialog::_add_type(const String &p_type, const TypeCategory p_type_category) {
	if (search_options_types.has(p_type)) {
		return;
	}

	String inherits;

	TypeCategory inherited_type = TypeCategory::OTHER_TYPE;

	if (p_type_category == TypeCategory::CPP_TYPE) {
		inherits = ClassDB::get_parent_class(p_type);
		inherited_type = TypeCategory::CPP_TYPE;
	} else if (p_type_category == TypeCategory::PATH_TYPE) {
		ERR_FAIL_COND(!ResourceLoader::exists(p_type, "Script"));
		Ref<Script> script = ResourceLoader::load(p_type, "Script");
		ERR_FAIL_COND(script.is_null());

		Ref<Script> base = script->get_base_script();
		if (base.is_null()) {
			String extends;
			script->get_language()->get_global_class_name(script->get_path(), &extends);

			inherits = extends;
			inherited_type = TypeCategory::CPP_TYPE;
		} else {
			inherits = script->get_language()->get_global_class_name(base->get_path());
			if (inherits.is_empty()) {
				inherits = base->get_path();
				inherited_type = TypeCategory::PATH_TYPE;
			}
		}
	} else {
		if (ScriptServer::is_global_class(p_type)) {
			inherits = EditorNode::get_editor_data().script_class_get_base(p_type);
			if (inherits.is_empty()) {
				Ref<Script> script = EditorNode::get_editor_data().script_class_load_script(p_type);
				ERR_FAIL_COND(script.is_null());

				Ref<Script> base = script->get_base_script();
				if (base.is_null()) {
					String extends;
					script->get_language()->get_global_class_name(script->get_path(), &extends);

					inherits = extends;
					inherited_type = TypeCategory::CPP_TYPE;
				} else {
					inherits = base->get_path();
					inherited_type = TypeCategory::PATH_TYPE;
				}
			}
		} else {
			inherits = custom_type_parents[p_type];
			if (ClassDB::class_exists(inherits)) {
				inherited_type = TypeCategory::CPP_TYPE;
			}
		}
	}

	// Should never happen, but just in case...
	ERR_FAIL_COND(inherits.is_empty());

	_add_type(inherits, inherited_type);

	TreeItem *item = search_options->create_item(search_options_types[inherits]);
	search_options_types[p_type] = item;
	_configure_search_option_item(item, p_type, p_type_category);
}

void CreateDialog::_configure_search_option_item(TreeItem *r_item, const String &p_type, const TypeCategory p_type_category) {
	bool script_type = ScriptServer::is_global_class(p_type);
	if (p_type_category == TypeCategory::CPP_TYPE) {
		r_item->set_text(0, p_type);
	} else if (p_type_category == TypeCategory::PATH_TYPE) {
		r_item->set_text(0, "\"" + p_type + "\"");
	} else if (script_type) {
		r_item->set_metadata(0, p_type);
		r_item->set_text(0, p_type + " (" + ScriptServer::get_global_class_path(p_type).get_file() + ")");
	} else {
		r_item->set_metadata(0, custom_type_parents[p_type]);
		r_item->set_text(0, p_type);
	}

	bool can_instantiate = (p_type_category == TypeCategory::CPP_TYPE && ClassDB::can_instantiate(p_type)) ||
			p_type_category == TypeCategory::OTHER_TYPE;

	if (!can_instantiate) {
		r_item->set_custom_color(0, search_options->get_theme_color(SNAME("disabled_font_color"), SNAME("Editor")));
		r_item->set_icon(0, EditorNode::get_singleton()->get_class_icon(p_type, "NodeDisabled"));
		r_item->set_selectable(0, false);
	} else {
		r_item->set_icon(0, EditorNode::get_singleton()->get_class_icon(p_type, icon_fallback));
	}

	if (!search_box->get_text().is_empty()) {
		r_item->set_collapsed(false);
	} else {
		// Don't collapse the root node or an abstract node on the first tree level.
		bool should_collapse = p_type != base_type && (r_item->get_parent()->get_text(0) != base_type || can_instantiate);

		if (should_collapse && bool(EditorSettings::get_singleton()->get("docks/scene_tree/start_create_dialog_fully_expanded"))) {
			should_collapse = false; // Collapse all nodes anyway.
		}
		r_item->set_collapsed(should_collapse);
	}

	const String &description = DTR(EditorHelp::get_doc_data()->class_list[p_type].brief_description);
	r_item->set_tooltip(0, description);

	if (p_type_category == TypeCategory::OTHER_TYPE && !script_type) {
		Ref<Texture2D> icon = EditorNode::get_editor_data().get_custom_types()[custom_type_parents[p_type]][custom_type_indices[p_type]].icon;
		if (icon.is_valid()) {
			r_item->set_icon(0, icon);
		}
	}
}

String CreateDialog::_top_result(const Vector<String> p_candidates, const String &p_search_text) const {
	float highest_score = 0;
	int highest_index = 0;
	for (int i = 0; i < p_candidates.size(); i++) {
		float score = _score_type(p_candidates[i].get_slicec(' ', 0), p_search_text);
		if (score > highest_score) {
			highest_score = score;
			highest_index = i;
		}
	}

	return p_candidates[highest_index];
}

float CreateDialog::_score_type(const String &p_type, const String &p_search) const {
	float inverse_length = 1.f / float(p_type.length());

	// Favor types where search term is a substring close to the start of the type.
	float w = 0.5f;
	int pos = p_type.findn(p_search);
	float score = (pos > -1) ? 1.0f - w * MIN(1, 3 * pos * inverse_length) : MAX(0.f, .9f - w);

	// Favor shorter items: they resemble the search term more.
	w = 0.1f;
	score *= (1 - w) + w * (p_search.length() * inverse_length);

	score *= _is_type_preferred(p_type) ? 1.0f : 0.8f;

	// Add score for being a favorite type.
	score *= (favorite_list.find(p_type) > -1) ? 1.0f : 0.7f;

	// Look through at most 5 recent items
	bool in_recent = false;
	for (int i = 0; i < MIN(5, recent->get_item_count()); i++) {
		if (recent->get_item_text(i) == p_type) {
			in_recent = true;
			break;
		}
	}
	score *= in_recent ? 1.0f : 0.8f;

	return score;
}

void CreateDialog::_cleanup() {
	type_list.clear();
	favorite_list.clear();
	favorites->clear();
	recent->clear();
	custom_type_parents.clear();
	custom_type_indices.clear();
}

void CreateDialog::_confirmed() {
	String selected_item = get_selected_type();
	if (selected_item.is_empty()) {
		return;
	}

	FileAccess *f = FileAccess::open(EditorSettings::get_singleton()->get_project_settings_dir().plus_file("create_recent." + base_type), FileAccess::WRITE);
	if (f) {
		f->store_line(selected_item);

		for (int i = 0; i < MIN(32, recent->get_item_count()); i++) {
			if (recent->get_item_text(i) != selected_item) {
				f->store_line(recent->get_item_text(i));
			}
		}

		memdelete(f);
	}

	// To prevent, emitting an error from the transient window (shader dialog for example) hide this dialog before emitting the "create" signal.
	hide();

	emit_signal(SNAME("create"));
	_cleanup();
}

void CreateDialog::_text_changed(const String &p_newtext) {
	_update_search();
}

void CreateDialog::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;
	if (k.is_valid()) {
		switch (k->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				search_options->gui_input(k);
				search_box->accept_event();
			} break;
			default:
				break;
		}
	}
}

void CreateDialog::_update_theme() {
	search_box->set_right_icon(search_options->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
	favorite->set_icon(search_options->get_theme_icon(SNAME("Favorites"), SNAME("EditorIcons")));
}

void CreateDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect("confirmed", callable_mp(this, &CreateDialog::_confirmed));
			_update_theme();
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
		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
		} break;
	}
}

void CreateDialog::select_type(const String &p_type) {
	if (!search_options_types.has(p_type)) {
		return;
	}

	TreeItem *to_select = search_options_types[p_type];
	to_select->select(0);
	search_options->scroll_to_item(to_select);

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
	favorite->set_pressed(favorite_list.find(p_type) != -1);
	get_ok_button()->set_disabled(false);
}

String CreateDialog::get_selected_type() {
	TreeItem *selected = search_options->get_selected();
	if (!selected) {
		return String();
	}

	return selected->get_text(0);
}

Variant CreateDialog::instance_selected() {
	TreeItem *selected = search_options->get_selected();

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

void CreateDialog::_item_selected() {
	String name = get_selected_type();
	select_type(name);
}

void CreateDialog::_hide_requested() {
	_cancel_pressed(); // From AcceptDialog.
}

void CreateDialog::cancel_pressed() {
	_cleanup();
}

void CreateDialog::_favorite_toggled() {
	TreeItem *item = search_options->get_selected();
	if (!item) {
		return;
	}

	String name = item->get_text(0);

	if (favorite_list.find(name) == -1) {
		favorite_list.push_back(name);
		favorite->set_pressed(true);
	} else {
		favorite_list.erase(name);
		favorite->set_pressed(false);
	}

	_save_and_update_favorite_list();
}

void CreateDialog::_history_selected(int p_idx) {
	search_box->set_text(recent->get_item_text(p_idx).get_slicec(' ', 0));
	favorites->deselect_all();
	_update_search();
}

void CreateDialog::_favorite_selected() {
	TreeItem *item = favorites->get_selected();
	if (!item) {
		return;
	}

	search_box->set_text(item->get_text(0).get_slicec(' ', 0));
	recent->deselect_all();
	_update_search();
}

void CreateDialog::_history_activated(int p_idx) {
	_history_selected(p_idx);
	_confirmed();
}

void CreateDialog::_favorite_activated() {
	_favorite_selected();
	_confirmed();
}

Variant CreateDialog::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	TreeItem *ti = favorites->get_item_at_position(p_point);
	if (ti) {
		Dictionary d;
		d["type"] = "create_favorite_drag";
		d["class"] = ti->get_text(0);

		Button *tb = memnew(Button);
		tb->set_flat(true);
		tb->set_icon(ti->get_icon(0));
		tb->set_text(ti->get_text(0));
		favorites->set_drag_preview(tb);

		return d;
	}

	return Variant();
}

bool CreateDialog::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	Dictionary d = p_data;
	if (d.has("type") && String(d["type"]) == "create_favorite_drag") {
		favorites->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN);
		return true;
	}

	return false;
}

void CreateDialog::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	Dictionary d = p_data;

	TreeItem *ti = favorites->get_item_at_position(p_point);
	if (!ti) {
		return;
	}

	String drop_at = ti->get_text(0);
	int ds = favorites->get_drop_section_at_position(p_point);

	int drop_idx = favorite_list.find(drop_at);
	if (drop_idx < 0) {
		return;
	}

	String type = d["class"];

	int from_idx = favorite_list.find(type);
	if (from_idx < 0) {
		return;
	}

	if (drop_idx == from_idx) {
		ds = -1; //cause it will be gone
	} else if (drop_idx > from_idx) {
		drop_idx--;
	}

	favorite_list.remove_at(from_idx);

	if (ds < 0) {
		favorite_list.insert(drop_idx, type);
	} else {
		if (drop_idx >= favorite_list.size() - 1) {
			favorite_list.push_back(type);
		} else {
			favorite_list.insert(drop_idx + 1, type);
		}
	}

	_save_and_update_favorite_list();
}

void CreateDialog::_save_and_update_favorite_list() {
	favorites->clear();
	TreeItem *root = favorites->create_item();

	FileAccess *f = FileAccess::open(EditorSettings::get_singleton()->get_project_settings_dir().plus_file("favorites." + base_type), FileAccess::WRITE);
	if (f) {
		for (int i = 0; i < favorite_list.size(); i++) {
			String l = favorite_list[i];
			String name = l.get_slicec(' ', 0);
			if (!(ClassDB::class_exists(name) || ScriptServer::is_global_class(name))) {
				continue;
			}
			f->store_line(l);

			if (_is_class_disabled_by_feature_profile(name)) {
				continue;
			}

			TreeItem *ti = favorites->create_item(root);
			ti->set_text(0, l);
			ti->set_icon(0, EditorNode::get_singleton()->get_class_icon(name, icon_fallback));
		}
		memdelete(f);
	}

	emit_signal(SNAME("favorites_updated"));
}

void CreateDialog::_load_favorites_and_history() {
	String dir = EditorSettings::get_singleton()->get_project_settings_dir();
	FileAccess *f = FileAccess::open(dir.plus_file("create_recent." + base_type), FileAccess::READ);
	if (f) {
		while (!f->eof_reached()) {
			String l = f->get_line().strip_edges();
			String name = l.get_slicec(' ', 0);

			if ((ClassDB::class_exists(name) || ScriptServer::is_global_class(name)) && !_is_class_disabled_by_feature_profile(name)) {
				recent->add_item(l, EditorNode::get_singleton()->get_class_icon(name, icon_fallback));
			}
		}

		memdelete(f);
	}

	f = FileAccess::open(dir.plus_file("favorites." + base_type), FileAccess::READ);
	if (f) {
		while (!f->eof_reached()) {
			String l = f->get_line().strip_edges();

			if (!l.is_empty()) {
				favorite_list.push_back(l);
			}
		}

		memdelete(f);
	}
}

void CreateDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_save_and_update_favorite_list"), &CreateDialog::_save_and_update_favorite_list);

	ClassDB::bind_method("_get_drag_data_fw", &CreateDialog::get_drag_data_fw);
	ClassDB::bind_method("_can_drop_data_fw", &CreateDialog::can_drop_data_fw);
	ClassDB::bind_method("_drop_data_fw", &CreateDialog::drop_data_fw);

	ADD_SIGNAL(MethodInfo("create"));
	ADD_SIGNAL(MethodInfo("favorites_updated"));
}

CreateDialog::CreateDialog() {
	base_type = "Object";
	preferred_search_result_type = "";

	type_blacklist.insert("PluginScript"); // PluginScript must be initialized before use, which is not possible here.
	type_blacklist.insert("ScriptCreateDialog"); // This is an exposed editor Node that doesn't have an Editor prefix.

	HSplitContainer *hsc = memnew(HSplitContainer);
	add_child(hsc);

	VSplitContainer *vsc = memnew(VSplitContainer);
	hsc->add_child(vsc);

	VBoxContainer *fav_vb = memnew(VBoxContainer);
	fav_vb->set_custom_minimum_size(Size2(150, 100) * EDSCALE);
	fav_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vsc->add_child(fav_vb);

	favorites = memnew(Tree);
	favorites->set_hide_root(true);
	favorites->set_hide_folding(true);
	favorites->set_allow_reselect(true);
	favorites->connect("cell_selected", callable_mp(this, &CreateDialog::_favorite_selected));
	favorites->connect("item_activated", callable_mp(this, &CreateDialog::_favorite_activated));
	favorites->add_theme_constant_override("draw_guides", 1);
#ifndef _MSC_VER
#warning cannot forward drag data to a non control, must be fixed
#endif
	//favorites->set_drag_forwarding(this);
	fav_vb->add_margin_child(TTR("Favorites:"), favorites, true);

	VBoxContainer *rec_vb = memnew(VBoxContainer);
	vsc->add_child(rec_vb);
	rec_vb->set_custom_minimum_size(Size2(150, 100) * EDSCALE);
	rec_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	recent = memnew(ItemList);
	rec_vb->add_margin_child(TTR("Recent:"), recent, true);
	recent->set_allow_reselect(true);
	recent->connect("item_selected", callable_mp(this, &CreateDialog::_history_selected));
	recent->connect("item_activated", callable_mp(this, &CreateDialog::_history_activated));
	recent->add_theme_constant_override("draw_guides", 1);

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_custom_minimum_size(Size2(300, 0) * EDSCALE);
	vbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hsc->add_child(vbc);

	search_box = memnew(LineEdit);
	search_box->set_clear_button_enabled(true);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_box->connect("text_changed", callable_mp(this, &CreateDialog::_text_changed));
	search_box->connect("gui_input", callable_mp(this, &CreateDialog::_sbox_input));

	HBoxContainer *search_hb = memnew(HBoxContainer);
	search_hb->add_child(search_box);

	favorite = memnew(Button);
	favorite->set_toggle_mode(true);
	favorite->set_tooltip(TTR("(Un)favorite selected item."));
	favorite->connect("pressed", callable_mp(this, &CreateDialog::_favorite_toggled));
	search_hb->add_child(favorite);
	vbc->add_margin_child(TTR("Search:"), search_hb);

	search_options = memnew(Tree);
	search_options->connect("item_activated", callable_mp(this, &CreateDialog::_confirmed));
	search_options->connect("cell_selected", callable_mp(this, &CreateDialog::_item_selected));
	vbc->add_margin_child(TTR("Matches:"), search_options, true);

	help_bit = memnew(EditorHelpBit);
	help_bit->connect("request_hide", callable_mp(this, &CreateDialog::_hide_requested));
	vbc->add_margin_child(TTR("Description:"), help_bit);

	register_text_enter(search_box);
	set_hide_on_ok(false);
}

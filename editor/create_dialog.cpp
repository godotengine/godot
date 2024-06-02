/**************************************************************************/
/*  create_dialog.cpp                                                     */
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

#include "create_dialog.h"

#include "core/io/json.h"
#include "core/object/class_db.h"
#include "core/os/keyboard.h"
#include "editor/editor_feature_profile.h"
#include "editor/editor_node.h"
#include "editor/editor_paths.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_checklist_dialog.h"
#include "editor/themes/editor_scale.h"

void CreateDialog::popup_create(bool p_dont_clear, bool p_replace_mode, const String &p_current_type, const String &p_current_name) {
	_fill_type_list();

	icon_fallback = search_options->has_theme_icon(base_type, EditorStringName(EditorIcons)) ? base_type : "Object";

	if (p_dont_clear) {
		search_box->select_all();
	} else {
		search_box->clear();
	}

	if (p_replace_mode) {
		search_box->set_text(p_current_type);
	}

	search_box->grab_focus();
	_update_search();

	if (p_replace_mode) {
		set_title(vformat(TTR("Change Type of \"%s\""), p_current_name));
		set_ok_button_text(TTR("Change"));
	} else {
		set_title(vformat(TTR("Create New %s"), base_type));
		set_ok_button_text(TTR("Create"));
	}

	_load_favorites_and_history_and_tabs();
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
		StringName type = I->get();
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

bool CreateDialog::_should_hide_type(const StringName &p_type) const {
	if (_is_class_disabled_by_feature_profile(p_type)) {
		return true;
	}

	if (is_base_type_node && p_type.operator String().begins_with("Editor")) {
		return true; // Do not show editor nodes.
	}

	if (ClassDB::class_exists(p_type)) {
		if (!ClassDB::can_instantiate(p_type) || ClassDB::is_virtual(p_type)) {
			return true; // Can't create abstract or virtual class.
		}

		if (!ClassDB::is_parent_class(p_type, base_type)) {
			return true; // Wrong inheritance.
		}

		if (!ClassDB::is_class_exposed(p_type)) {
			return true; // Unexposed types.
		}

		for (const StringName &E : type_blacklist) {
			if (ClassDB::is_parent_class(p_type, E)) {
				return true; // Parent type is blacklisted.
			}
		}
	} else {
		if (!ScriptServer::is_global_class(p_type)) {
			return true;
		}
		if (!EditorNode::get_editor_data().script_class_is_parent(p_type, base_type)) {
			return true; // Wrong inheritance.
		}

		StringName native_type = ScriptServer::get_global_class_native_base(p_type);
		if (ClassDB::class_exists(native_type) && !ClassDB::can_instantiate(native_type)) {
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
	root->set_icon(0, search_options->get_editor_theme_icon(icon_fallback));
	search_options_types[base_type] = root;
	_configure_search_option_item(root, base_type, ClassDB::class_exists(base_type) ? TypeCategory::CPP_TYPE : TypeCategory::OTHER_TYPE);

	Tree *current_tree = Object::cast_to<Tree>(tabs->get_tab_control(tabs->get_current_tab()));
	Array current_metadata = static_cast<Array>(tabs->get_tab_metadata(tabs->get_current_tab()));

	if (current_tree != search_options) {
		current_tree->clear();

		Dictionary custom_search_options_types = current_metadata[0];
		custom_search_options_types.clear();

		TreeItem *custom_root = current_tree->create_item();
		custom_root->set_text(0, base_type);
		custom_root->set_icon(0, current_tree->get_editor_theme_icon(icon_fallback));
		custom_search_options_types[base_type] = custom_root;
	}

	const String search_text = search_box->get_text();
	bool empty_search = search_text.is_empty();

	float highest_score = 0.0f;
	StringName best_match;

	for (List<StringName>::Element *I = type_list.front(); I; I = I->next()) {
		StringName candidate = I->get();
		if (empty_search || search_text.is_subsequence_ofn(candidate)) {
			_add_type(candidate, ClassDB::class_exists(candidate) ? TypeCategory::CPP_TYPE : TypeCategory::OTHER_TYPE, current_tree, current_metadata);

			// Determine the best match for an non-empty search.
			if (!empty_search) {
				float score = _score_type(candidate.operator String().get_slicec(' ', 0), search_text);
				if (score > highest_score) {
					highest_score = score;
					best_match = candidate;
				}
			}
		}
	}

	// Select the best result.
	if (empty_search) {
		select_type(base_type);
		search_options->scroll_to_item(root); // for some reason the scroll is going down a little bit the first time the dialog is opened after this Tree was added to the TabContainer. This line fixes that.
	} else if (best_match != StringName()) {
		select_type(best_match);
	} else {
		favorite->set_disabled(true);
		help_bit->set_custom_text(String(), String(), vformat(TTR("No results for \"%s\"."), search_text.replace("[", "[lb]")));
		get_ok_button()->set_disabled(true);
		search_options->deselect_all();
	}
}

void CreateDialog::_add_type(const StringName &p_type, TypeCategory p_type_category, Tree *current_tree, Array current_metadata) {
	if (search_options_types.has(p_type)) {
		return;
	}

	TypeCategory inherited_type = TypeCategory::OTHER_TYPE;

	StringName inherits;
	if (p_type_category == TypeCategory::CPP_TYPE) {
		inherits = ClassDB::get_parent_class(p_type);
		inherited_type = TypeCategory::CPP_TYPE;
	} else {
		if (p_type_category == TypeCategory::PATH_TYPE || ScriptServer::is_global_class(p_type)) {
			Ref<Script> scr;
			if (p_type_category == TypeCategory::PATH_TYPE) {
				ERR_FAIL_COND(!ResourceLoader::exists(p_type, "Script"));
				scr = ResourceLoader::load(p_type, "Script");
			} else {
				scr = EditorNode::get_editor_data().script_class_load_script(p_type);
			}
			ERR_FAIL_COND(scr.is_null());

			Ref<Script> base = scr->get_base_script();
			if (base.is_null()) {
				// Must be a native base type.
				StringName extends = scr->get_instance_base_type();
				if (extends == StringName()) {
					// Not a valid script (has compile errors), we therefore ignore it as it can not be instantiated anyway (when selected).
					return;
				}

				inherits = extends;
				inherited_type = TypeCategory::CPP_TYPE;
			} else {
				inherits = base->get_global_name();

				if (inherits == StringName()) {
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
	ERR_FAIL_COND(inherits == StringName());

	_add_type(inherits, inherited_type, current_tree, current_metadata);

	TreeItem *item = search_options->create_item(search_options_types[inherits]);
	search_options_types[p_type] = item;
	_configure_search_option_item(item, p_type, p_type_category);

	if ((current_tree != search_options) && static_cast<const Dictionary &>(current_metadata[1]).has(p_type)) {
		Dictionary custom_search_options_types = current_metadata[0];

		TreeItem *custom_item = current_tree->create_item(Object::cast_to<TreeItem>(custom_search_options_types[inherits]));
		custom_search_options_types[p_type] = custom_item;
		_configure_search_option_item(custom_item, p_type, p_type_category);
	}
}

void CreateDialog::_configure_search_option_item(TreeItem *r_item, const StringName &p_type, TypeCategory p_type_category) {
	bool script_type = ScriptServer::is_global_class(p_type);
	bool is_abstract = false;
	if (p_type_category == TypeCategory::CPP_TYPE) {
		r_item->set_text(0, p_type);
	} else if (p_type_category == TypeCategory::PATH_TYPE) {
		r_item->set_text(0, "\"" + p_type + "\"");
	} else if (script_type) {
		r_item->set_metadata(0, p_type);
		r_item->set_text(0, p_type);
		String script_path = ScriptServer::get_global_class_path(p_type);
		r_item->set_suffix(0, "(" + script_path.get_file() + ")");

		Ref<Script> scr = ResourceLoader::load(script_path, "Script");
		ERR_FAIL_COND(!scr.is_valid());
		is_abstract = scr->is_abstract();
	} else {
		r_item->set_metadata(0, custom_type_parents[p_type]);
		r_item->set_text(0, p_type);
	}

	bool can_instantiate = (p_type_category == TypeCategory::CPP_TYPE && ClassDB::can_instantiate(p_type)) ||
			(p_type_category == TypeCategory::OTHER_TYPE && !is_abstract);
	bool instantiable = can_instantiate && !(ClassDB::class_exists(p_type) && ClassDB::is_virtual(p_type));

	r_item->set_meta(SNAME("__instantiable"), instantiable);

	r_item->set_icon(0, EditorNode::get_singleton()->get_class_icon(p_type));
	if (!instantiable) {
		r_item->set_custom_color(0, search_options->get_theme_color(SNAME("font_disabled_color"), EditorStringName(Editor)));
	}

	HashMap<String, DocData::ClassDoc>::Iterator class_doc = EditorHelp::get_doc_data()->class_list.find(p_type);

	bool is_deprecated = (class_doc && class_doc->value.is_deprecated);
	bool is_experimental = (class_doc && class_doc->value.is_experimental);

	if (is_deprecated) {
		r_item->add_button(0, get_editor_theme_icon("StatusError"), 0, false, TTR("This class is marked as deprecated."));
	} else if (is_experimental) {
		r_item->add_button(0, get_editor_theme_icon("NodeWarning"), 0, false, TTR("This class is marked as experimental."));
	}

	if (!search_box->get_text().is_empty()) {
		r_item->set_collapsed(false);
	} else {
		// Don't collapse the root node or an abstract node on the first tree level.
		bool should_collapse = p_type != base_type && (r_item->get_parent()->get_text(0) != base_type || can_instantiate);

		if (should_collapse && bool(EDITOR_GET("docks/scene_tree/start_create_dialog_fully_expanded"))) {
			should_collapse = false; // Collapse all nodes anyway.
		}
		r_item->set_collapsed(should_collapse);
	}

	const String &description = DTR(class_doc ? class_doc->value.brief_description : "");
	r_item->set_tooltip_text(0, description);

	if (p_type_category == TypeCategory::OTHER_TYPE && !script_type) {
		Ref<Texture2D> icon = EditorNode::get_editor_data().get_custom_types()[custom_type_parents[p_type]][custom_type_indices[p_type]].icon;
		if (icon.is_valid()) {
			r_item->set_icon(0, icon);
		}
	}
}

float CreateDialog::_score_type(const String &p_type, const String &p_search) const {
	if (p_type == p_search) {
		// Always favor an exact match (case-sensitive), since clicking a favorite will set the search text to the type.
		return 1.0f;
	}

	float inverse_length = 1.f / float(p_type.length());

	// Favor types where search term is a substring close to the start of the type.
	float w = 0.5f;
	int pos = p_type.findn(p_search);
	float score = (pos > -1) ? 1.0f - w * MIN(1, 3 * pos * inverse_length) : MAX(0.f, .9f - w);

	// Favor shorter items: they resemble the search term more.
	w = 0.9f;
	score *= (1 - w) + w * MIN(1.0f, p_search.length() * inverse_length);

	score *= _is_type_preferred(p_type) ? 1.0f : 0.9f;

	// Add score for being a favorite type.
	score *= favorite_list.has(p_type) ? 1.0f : 0.8f;

	// Look through at most 5 recent items
	bool in_recent = false;
	constexpr int RECENT_COMPLETION_SIZE = 5;
	for (int i = 0; i < MIN(RECENT_COMPLETION_SIZE - 1, recent->get_item_count()); i++) {
		if (recent->get_item_text(i) == p_type) {
			in_recent = true;
			break;
		}
	}
	score *= in_recent ? 1.0f : 0.9f;

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

	TreeItem *selected = search_options->get_selected();
	if (!selected->get_meta("__instantiable", true)) {
		return;
	}

	{
		Ref<FileAccess> f = FileAccess::open(EditorPaths::get_singleton()->get_project_settings_dir().path_join("create_recent." + base_type), FileAccess::WRITE);
		if (f.is_valid()) {
			f->store_line(selected_item);

			constexpr int RECENT_HISTORY_SIZE = 15;
			for (int i = 0; i < MIN(RECENT_HISTORY_SIZE - 1, recent->get_item_count()); i++) {
				if (recent->get_item_text(i) != selected_item) {
					f->store_line(recent->get_item_text(i));
				}
			}
		}
	}

	// To prevent, emitting an error from the transient window (shader dialog for example) hide this dialog before emitting the "create" signal.
	hide();

	emit_signal(SNAME("create"));
	_cleanup();
}

void CreateDialog::_custom_confirmed() {
	search_options->set_selected(search_options_types[Object::cast_to<Tree>(tabs->get_tab_control(tabs->get_current_tab()))->get_selected()->get_text(0)]);
	_confirmed();
}

void CreateDialog::_text_changed(const String &p_newtext) {
	_update_search();
}

void CreateDialog::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;
	if (k.is_valid() && k->is_pressed()) {
		switch (k->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				search_options->gui_input(k);
				search_box->accept_event();
			} break;
			case Key::SPACE: {
				TreeItem *ti = search_options->get_selected();
				if (ti) {
					ti->set_collapsed(!ti->is_collapsed());
				}
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
		} break;

		case NOTIFICATION_EXIT_TREE: {
			disconnect("confirmed", callable_mp(this, &CreateDialog::_confirmed));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				callable_mp((Control *)search_box, &Control::grab_focus).call_deferred(); // Still not visible.
				search_box->select_all();
			} else {
				EditorSettings::get_singleton()->set_project_metadata("dialog_bounds", "create_new_node", Rect2(get_position(), get_size()));
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			const int icon_width = get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor));
			search_options->add_theme_constant_override("icon_max_width", icon_width);
			favorites->add_theme_constant_override("icon_max_width", icon_width);
			recent->set_fixed_icon_size(Size2(icon_width, icon_width));

			search_box->set_right_icon(get_editor_theme_icon(SNAME("Search")));
			favorite->set_icon(get_editor_theme_icon(SNAME("Favorites")));
			add_category->set_icon(get_editor_theme_icon(SNAME("Add")));
			tabs->set_tab_icon(0, get_editor_theme_icon(SNAME("Lock")));
		} break;
	}
}

void CreateDialog::select_type(const String &p_type, bool p_center_on_item) {
	if (!search_options_types.has(p_type)) {
		return;
	}

	TreeItem *to_select = search_options_types[p_type];
	to_select->select(0);
	search_options->scroll_to_item(to_select, p_center_on_item);

	help_bit->parse_symbol("class|" + p_type + "|");

	favorite->set_disabled(false);
	favorite->set_pressed(favorite_list.has(p_type));

	if (to_select->get_meta("__instantiable", true)) {
		get_ok_button()->set_disabled(false);
		get_ok_button()->set_tooltip_text(String());
	} else {
		get_ok_button()->set_disabled(true);
		get_ok_button()->set_tooltip_text(TTR("The selected class can't be instantiated."));
	}
}

void CreateDialog::select_base() {
	if (search_options_types.is_empty()) {
		_update_search();
	}
	select_type(base_type, false);
}

String CreateDialog::get_selected_type() {
	TreeItem *selected = search_options->get_selected();
	if (!selected) {
		return String();
	}

	return selected->get_text(0);
}

void CreateDialog::set_base_type(const String &p_base) {
	base_type = p_base;
	is_base_type_node = ClassDB::is_parent_class(p_base, "Node");
}

Variant CreateDialog::instantiate_selected() {
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
			obj = EditorNode::get_editor_data().instantiate_custom_type(selected->get_text(0), custom);
		}
	} else {
		obj = ClassDB::instantiate(selected->get_text(0));
	}
	EditorNode::get_editor_data().instantiate_object_properties(obj);

	return obj;
}

void CreateDialog::_item_selected() {
	String name = get_selected_type();
	select_type(name, false);
}
void CreateDialog::_custom_item_selected() {
	search_options->set_selected(search_options_types[Object::cast_to<Tree>(tabs->get_tab_control(tabs->get_current_tab()))->get_selected()->get_text(0)]);
	_item_selected();
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

	if (favorite_list.has(name)) {
		favorite_list.erase(name);
		favorite->set_pressed(false);
	} else {
		favorite_list.push_back(name);
		favorite->set_pressed(true);
	}

	_save_and_update_favorite_list();
}

Variant CreateDialog::_create_tab_metadata(const Dictionary &p_node_names) const {
	Array tab_metadata;
	tab_metadata.append(Dictionary{}); // for the custom search options types
	tab_metadata.append(p_node_names);

	return tab_metadata;
}

void CreateDialog::_add_tab(const String &p_name, const Dictionary &p_node_names) {
	Tree *new_search_options = memnew(Tree);
	new_search_options->set_name(p_name);
	new_search_options->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	new_search_options->connect("item_activated", callable_mp(this, &CreateDialog::_custom_confirmed));
	new_search_options->connect("cell_selected", callable_mp(this, &CreateDialog::_custom_item_selected));
	tabs->add_child(new_search_options);
	tabs->set_tab_metadata(tabs->get_tab_count() - 1, _create_tab_metadata(p_node_names));
}

void CreateDialog::_add_category_pressed() {
	String category_name = category_box->get_text();

	category_box->clear();

	if (category_name.is_empty() || tabs->has_node(category_name)) {
		return;
	}

	Dictionary node_names;
	_add_tab(category_name, node_names);

	Dictionary tab_data;
	tab_data["name"] = category_name;
	tab_data["nodes"] = node_names;
	static_cast<Array>(json->get_data()).push_back(tab_data);
	_save_and_update_tabs();
}

void CreateDialog::_add_node_pressed() {
	int tabs_child_count = tabs->get_child_count(false);

	if ((tabs_child_count == 1) || !Object::cast_to<Tree>(tabs->get_tab_control(tabs->get_current_tab()))->get_selected()) {
		return;
	}

	Tree *tree = memnew(Tree);

	TreeItem *root = tree->create_item();
	root->set_text(0, "Categories");

	for (int i = 0; i < tabs_child_count; ++i) {
		Node *tab_child = tabs->get_child(i, false);

		if (tab_child == search_options) {
			continue;
		}

		TreeItem *item = tree->create_item(root);
		item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		item->set_text(0, tab_child->get_name());
		item->set_editable(0, true);
	}

	checklist_dialog->reload(tree);
	checklist_dialog->popup_centered();
}

void CreateDialog::_remove_node_pressed() {
	Tree *current_tree = Object::cast_to<Tree>(tabs->get_tab_control(tabs->get_current_tab()));
	TreeItem *selected = current_tree->get_selected();

	if ((current_tree == search_options) || !selected) {
		return;
	}

	static_cast<Dictionary>(static_cast<Array>(tabs->get_tab_metadata(tabs->get_current_tab()))[1]).erase(selected->get_text(0));

	_save_and_update_tabs();
	_update_search();
}

void CreateDialog::_checklist_confirmed() {
	String node_name = Object::cast_to<Tree>(tabs->get_tab_control(tabs->get_current_tab()))->get_selected()->get_text(0);

	Vector<TreeItem *> checked = checklist_dialog->get_all_checked();

	for (const TreeItem *item : checked) {
		static_cast<Dictionary>(static_cast<Array>(tabs->get_tab_metadata(tabs->get_node(item->get_text(0))->get_index(false)))[1])[node_name] = 1;
	}

	_save_and_update_tabs();
}

void CreateDialog::_tab_changed(int p_idx) {
	if (tabs->get_tab_control(p_idx) == search_options) {
		tabs->get_tab_bar()->set_tab_close_display_policy(TabBar::CLOSE_BUTTON_SHOW_NEVER);
	} else {
		tabs->get_tab_bar()->set_tab_close_display_policy(TabBar::CLOSE_BUTTON_SHOW_ACTIVE_ONLY);
	}

	_update_search();
}

void CreateDialog::_tab_closed(int p_idx) {
	String node_name = tabs->get_child(p_idx, false)->get_name();
	tabs->get_child(p_idx, false)->queue_free();

	Array tab_data = json->get_data();
	for (int i = 0; i < tab_data.size(); ++i) {
		if (static_cast<const Dictionary &>(tab_data[i])["name"] == node_name) {
			tab_data.remove_at(i);
			break;
		}
	}
	
	_save_and_update_tabs();
}

void CreateDialog::_tab_rearranged(int p_idx_to) {
	if (tabs->get_child(p_idx_to, false) == search_options) {
		return;
	}

	Array tab_data = json->get_data();
	tab_data.clear();

	for (int i = 0, tabs_child_count = tabs->get_child_count(false); i < tabs_child_count; ++i) {
		Node *node = tabs->get_child(i, false);

		if (node == search_options) {
			continue;
		}

		Dictionary new_data;
		new_data["name"] = node->get_name();
		new_data["nodes"] = static_cast<Array>(tabs->get_tab_metadata(i))[1];
		tab_data.push_back(new_data);
	}

	_save_and_update_tabs();
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

	{
		Ref<FileAccess> f = FileAccess::open(EditorPaths::get_singleton()->get_project_settings_dir().path_join("favorites." + base_type), FileAccess::WRITE);
		if (f.is_valid()) {
			for (int i = 0; i < favorite_list.size(); i++) {
				String l = favorite_list[i];
				String name = l.get_slicec(' ', 0);
				if (!EditorNode::get_editor_data().is_type_recognized(name)) {
					continue;
				}
				f->store_line(l);

				if (_is_class_disabled_by_feature_profile(name)) {
					continue;
				}

				TreeItem *ti = favorites->create_item(root);
				ti->set_text(0, l);
				ti->set_icon(0, EditorNode::get_singleton()->get_class_icon(name));
			}
		}
	}

	emit_signal(SNAME("favorites_updated"));
}

void CreateDialog::_save_and_update_tabs() {
	Ref<FileAccess> f = FileAccess::open(EditorPaths::get_singleton()->get_project_settings_dir().path_join("tabs." + base_type), FileAccess::WRITE);
	if (f.is_valid()) {
		f->store_string(json->stringify(json->get_data()));
	}

	emit_signal(SNAME("tabs_updated"));
}

void CreateDialog::_load_favorites_and_history_and_tabs() {
	String dir = EditorPaths::get_singleton()->get_project_settings_dir();
	Ref<FileAccess> f = FileAccess::open(dir.path_join("create_recent." + base_type), FileAccess::READ);
	if (f.is_valid()) {
		while (!f->eof_reached()) {
			String l = f->get_line().strip_edges();
			String name = l.get_slicec(' ', 0);

			if (EditorNode::get_editor_data().is_type_recognized(name) && !_is_class_disabled_by_feature_profile(name)) {
				recent->add_item(l, EditorNode::get_singleton()->get_class_icon(name));
			}
		}
	}

	f = FileAccess::open(dir.path_join("favorites." + base_type), FileAccess::READ);
	if (f.is_valid()) {
		while (!f->eof_reached()) {
			String l = f->get_line().strip_edges();

			if (!l.is_empty()) {
				favorite_list.push_back(l);
			}
		}
	}

	if (json.is_null()) {
		json.instantiate();
		f = FileAccess::open(dir.path_join("tabs." + base_type), FileAccess::READ);
		if (f.is_valid()) {
			Error err = json->parse(f->get_as_text());
			if (err != OK) {
				String err_text = "Error parsing tabs file on line " + itos(json->get_error_line()) + ": " + json->get_error_message();
				WARN_PRINT(err_text);
			}

			Array tab_data = json->get_data();
			for (const Dictionary &tab_metadata : tab_data) {
				_add_tab(tab_metadata["name"], tab_metadata["nodes"]);
			}
		} else {
			json->parse("[]");
		}
	}
}

void CreateDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("create"));
	ADD_SIGNAL(MethodInfo("favorites_updated"));
	ADD_SIGNAL(MethodInfo("tabs_updated"));
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
	favorites->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	favorites->set_hide_root(true);
	favorites->set_hide_folding(true);
	favorites->set_allow_reselect(true);
	favorites->connect("cell_selected", callable_mp(this, &CreateDialog::_favorite_selected));
	favorites->connect("item_activated", callable_mp(this, &CreateDialog::_favorite_activated));
	favorites->add_theme_constant_override("draw_guides", 1);
	SET_DRAG_FORWARDING_GCD(favorites, CreateDialog);
	fav_vb->add_margin_child(TTR("Favorites:"), favorites, true);

	VBoxContainer *rec_vb = memnew(VBoxContainer);
	vsc->add_child(rec_vb);
	rec_vb->set_custom_minimum_size(Size2(150, 100) * EDSCALE);
	rec_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	recent = memnew(ItemList);
	recent->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
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
	search_box->connect(SceneStringName(gui_input), callable_mp(this, &CreateDialog::_sbox_input));

	HBoxContainer *search_hb = memnew(HBoxContainer);
	search_hb->add_child(search_box);

	favorite = memnew(Button);
	favorite->set_toggle_mode(true);
	favorite->set_tooltip_text(TTR("(Un)favorite selected item."));
	favorite->connect(SceneStringName(pressed), callable_mp(this, &CreateDialog::_favorite_toggled));
	search_hb->add_child(favorite);
	vbc->add_margin_child(TTR("Search:"), search_hb);

	category_box = memnew(LineEdit);
	category_box->set_clear_button_enabled(true);
	category_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	add_category = memnew(Button);
	add_category->set_tooltip_text(TTR("Add a new category tab."));
	add_category->connect(SceneStringName(pressed), callable_mp(this, &CreateDialog::_add_category_pressed));

	Button *add_node = memnew(Button);
	add_node->set_text(TTR("Add node"));
	add_node->set_tooltip_text(TTR("Add node into selected category tab."));
	add_node->connect(SceneStringName(pressed), callable_mp(this, &CreateDialog::_add_node_pressed));

	Button *remove_node = memnew(Button);
	remove_node->set_text(TTR("Remove node"));
	remove_node->set_tooltip_text(TTR("Remove node from selected category tab."));
	remove_node->connect(SceneStringName(pressed), callable_mp(this, &CreateDialog::_remove_node_pressed));

	HBoxContainer *category_hb = memnew(HBoxContainer);
	category_hb->add_child(category_box);
	category_hb->add_child(add_category);
	VSeparator *v_separator = memnew(VSeparator);
	category_hb->add_child(v_separator);
	category_hb->add_child(add_node);
	category_hb->add_child(remove_node);
	vbc->add_margin_child(TTR("Categories:"), category_hb);

	tabs = memnew(TabContainer);
	tabs->set_drag_to_rearrange_enabled(true);
	tabs->connect("tab_changed", callable_mp(this, &CreateDialog::_tab_changed));
	tabs->connect("active_tab_rearranged", callable_mp(this, &CreateDialog::_tab_rearranged));
	tabs->get_tab_bar()->connect("tab_close_pressed", callable_mp(this, &CreateDialog::_tab_closed));
	vbc->add_margin_child(TTR("Matches:"), tabs, true);

	search_options = memnew(Tree);
	search_options->set_name(TTR("Built-in"));
	search_options->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	search_options->connect("item_activated", callable_mp(this, &CreateDialog::_confirmed));
	search_options->connect("cell_selected", callable_mp(this, &CreateDialog::_item_selected));
	tabs->add_child(search_options);
	tabs->set_tab_metadata(0, _create_tab_metadata());

	checklist_dialog = memnew(EditorChecklistDialog);
	checklist_dialog->connect("confirmed", callable_mp(this, &CreateDialog::_checklist_confirmed));
	add_child(checklist_dialog);

	help_bit = memnew(EditorHelpBit);
	help_bit->set_content_height_limits(64 * EDSCALE, 64 * EDSCALE);
	help_bit->connect("request_hide", callable_mp(this, &CreateDialog::_hide_requested));
	vbc->add_margin_child(TTR("Description:"), help_bit);

	register_text_enter(search_box);
	set_hide_on_ok(false);
	set_clamp_to_embedder(true);
}

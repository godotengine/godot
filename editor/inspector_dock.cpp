/*************************************************************************/
/*  inspector_dock.cpp                                                   */
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

#include "inspector_dock.h"

#include "editor/editor_scale.h"
#include "editor/plugins/animation_player_editor_plugin.h"

void InspectorDock::_menu_option(int p_option) {
	_menu_option_confirm(p_option, false);
}

void InspectorDock::_menu_confirm_current() {
	_menu_option_confirm(current_option, true);
}

void InspectorDock::_menu_option_confirm(int p_option, bool p_confirmed) {
	if (!p_confirmed) {
		current_option = p_option;
	}

	switch (p_option) {
		case EXPAND_ALL: {
			_menu_expandall();
		} break;
		case COLLAPSE_ALL: {
			_menu_collapseall();
		} break;

		case RESOURCE_SAVE: {
			_save_resource(false);
		} break;
		case RESOURCE_SAVE_AS: {
			_save_resource(true);
		} break;

		case RESOURCE_MAKE_BUILT_IN: {
			_unref_resource();
		} break;
		case RESOURCE_COPY: {
			_copy_resource();
		} break;
		case RESOURCE_EDIT_CLIPBOARD: {
			_paste_resource();
		} break;

		case OBJECT_REQUEST_HELP: {
			if (current) {
				editor->set_visible_editor(EditorNode::EDITOR_SCRIPT);
				emit_signal(SNAME("request_help"), current->get_class());
			}
		} break;

		case OBJECT_COPY_PARAMS: {
			editor_data->apply_changes_in_editors();
			if (current) {
				editor_data->copy_object_params(current);
			}
		} break;

		case OBJECT_PASTE_PARAMS: {
			editor_data->apply_changes_in_editors();
			if (current) {
				editor_data->paste_object_params(current);
			}
		} break;

		case OBJECT_UNIQUE_RESOURCES: {
			if (!p_confirmed) {
				Vector<String> resource_propnames;

				if (current) {
					List<PropertyInfo> props;
					current->get_property_list(&props);

					for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
						if (!(E->get().usage & PROPERTY_USAGE_STORAGE)) {
							continue;
						}

						Variant v = current->get(E->get().name);
						REF ref = v;
						RES res = ref;
						if (v.is_ref() && ref.is_valid() && res.is_valid()) {
							// Valid resource which would be duplicated if action is confirmed.
							resource_propnames.append(E->get().name);
						}
					}
				}

				if (resource_propnames.size()) {
					unique_resources_list_tree->clear();
					TreeItem *root = unique_resources_list_tree->create_item();

					for (int i = 0; i < resource_propnames.size(); i++) {
						String propname = resource_propnames[i].replace("/", " / ");

						TreeItem *ti = unique_resources_list_tree->create_item(root);
						ti->set_text(0, bool(EDITOR_GET("interface/inspector/capitalize_properties")) ? propname.capitalize() : propname);
					}

					unique_resources_confirmation->popup_centered();
				} else {
					unique_resources_confirmation->set_text(TTR("This object has no resources."));
					current_option = -1;
					unique_resources_confirmation->popup_centered();
				}
			} else {
				editor_data->apply_changes_in_editors();

				if (current) {
					List<PropertyInfo> props;
					current->get_property_list(&props);
					Map<RES, RES> duplicates;
					for (const PropertyInfo &prop_info : props) {
						if (!(prop_info.usage & PROPERTY_USAGE_STORAGE)) {
							continue;
						}

						Variant v = current->get(prop_info.name);
						if (v.is_ref()) {
							REF ref = v;
							if (ref.is_valid()) {
								RES res = ref;
								if (res.is_valid()) {
									if (!duplicates.has(res)) {
										duplicates[res] = res->duplicate();
									}
									res = duplicates[res];

									current->set(prop_info.name, res);
									editor->get_inspector()->update_property(prop_info.name);
								}
							}
						}
					}
				}

				editor_data->get_undo_redo().clear_history();

				editor->get_editor_plugins_over()->edit(nullptr);
				editor->get_editor_plugins_over()->edit(current);
			}

		} break;

		default: {
			if (p_option >= OBJECT_METHOD_BASE) {
				ERR_FAIL_COND(!current);

				int idx = p_option - OBJECT_METHOD_BASE;

				List<MethodInfo> methods;
				current->get_method_list(&methods);

				ERR_FAIL_INDEX(idx, methods.size());
				String name = methods[idx].name;

				current->call(name);
			}
		}
	}
}

void InspectorDock::_new_resource() {
	new_resource_dialog->popup_create(true);
}

void InspectorDock::_load_resource(const String &p_type) {
	load_resource_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);

	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type(p_type, &extensions);

	load_resource_dialog->clear_filters();
	for (int i = 0; i < extensions.size(); i++) {
		load_resource_dialog->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
	}

	const Vector<String> textfile_ext = ((String)(EditorSettings::get_singleton()->get("docks/filesystem/textfile_extensions"))).split(",", false);
	for (int i = 0; i < textfile_ext.size(); i++) {
		load_resource_dialog->add_filter("*." + textfile_ext[i] + " ; " + textfile_ext[i].to_upper());
	}

	load_resource_dialog->popup_file_dialog();
}

void InspectorDock::_resource_file_selected(String p_file) {
	RES res;
	if (ResourceLoader::exists(p_file, "")) {
		res = ResourceLoader::load(p_file);
	} else {
		const Vector<String> textfile_ext = ((String)(EditorSettings::get_singleton()->get("docks/filesystem/textfile_extensions"))).split(",", false);
		if (textfile_ext.has(p_file.get_extension())) {
			res = ScriptEditor::get_singleton()->open_file(p_file);
		}
	}

	if (res.is_null()) {
		warning_dialog->set_text(TTR("Failed to load resource."));
		return;
	};

	editor->push_item(res.operator->());
}

void InspectorDock::_save_resource(bool save_as) {
	ObjectID current = EditorNode::get_singleton()->get_editor_history()->get_current();
	Object *current_obj = current.is_valid() ? ObjectDB::get_instance(current) : nullptr;

	ERR_FAIL_COND(!Object::cast_to<Resource>(current_obj));

	RES current_res = RES(Object::cast_to<Resource>(current_obj));

	if (save_as) {
		editor->save_resource_as(current_res);
	} else {
		editor->save_resource(current_res);
	}
}

void InspectorDock::_unref_resource() {
	ObjectID current = EditorNode::get_singleton()->get_editor_history()->get_current();
	Object *current_obj = current.is_valid() ? ObjectDB::get_instance(current) : nullptr;

	ERR_FAIL_COND(!Object::cast_to<Resource>(current_obj));

	RES current_res = RES(Object::cast_to<Resource>(current_obj));
	current_res->set_path("");
	editor->edit_current();
}

void InspectorDock::_copy_resource() {
	ObjectID current = EditorNode::get_singleton()->get_editor_history()->get_current();
	Object *current_obj = current.is_valid() ? ObjectDB::get_instance(current) : nullptr;

	ERR_FAIL_COND(!Object::cast_to<Resource>(current_obj));

	RES current_res = RES(Object::cast_to<Resource>(current_obj));

	EditorSettings::get_singleton()->set_resource_clipboard(current_res);
}

void InspectorDock::_paste_resource() {
	RES r = EditorSettings::get_singleton()->get_resource_clipboard();
	if (r.is_valid()) {
		editor->push_item(EditorSettings::get_singleton()->get_resource_clipboard().ptr(), String());
	}
}

void InspectorDock::_prepare_resource_extra_popup() {
	RES r = EditorSettings::get_singleton()->get_resource_clipboard();
	PopupMenu *popup = resource_extra_button->get_popup();
	popup->set_item_disabled(popup->get_item_index(RESOURCE_EDIT_CLIPBOARD), r.is_null());
}

void InspectorDock::_prepare_history() {
	EditorHistory *editor_history = EditorNode::get_singleton()->get_editor_history();

	int history_to = MAX(0, editor_history->get_history_len() - 25);

	history_menu->get_popup()->clear();

	Ref<Texture2D> base_icon = get_theme_icon(SNAME("Object"), SNAME("EditorIcons"));
	Set<ObjectID> already;
	for (int i = editor_history->get_history_len() - 1; i >= history_to; i--) {
		ObjectID id = editor_history->get_history_obj(i);
		Object *obj = ObjectDB::get_instance(id);
		if (!obj || already.has(id)) {
			if (history_to > 0) {
				history_to--;
			}
			continue;
		}

		already.insert(id);

		Ref<Texture2D> icon = EditorNode::get_singleton()->get_object_icon(obj, "");
		if (icon.is_null()) {
			icon = base_icon;
		}

		String text;
		if (Object::cast_to<Resource>(obj)) {
			Resource *r = Object::cast_to<Resource>(obj);
			if (r->get_path().is_resource_file()) {
				text = r->get_path().get_file();
			} else if (!r->get_name().is_empty()) {
				text = r->get_name();
			} else {
				text = r->get_class();
			}
		} else if (Object::cast_to<Node>(obj)) {
			text = Object::cast_to<Node>(obj)->get_name();
		} else if (obj->is_class("EditorDebuggerRemoteObject")) {
			text = obj->call("get_title");
		} else {
			text = obj->get_class();
		}

		if (i == editor_history->get_history_pos() && current) {
			text = "[" + text + "]";
		}
		history_menu->get_popup()->add_icon_item(icon, text, i);
	}
}

void InspectorDock::_select_history(int p_idx) {
	//push it to the top, it is not correct, but it's more useful
	ObjectID id = EditorNode::get_singleton()->get_editor_history()->get_history_obj(p_idx);
	Object *obj = ObjectDB::get_instance(id);
	if (!obj) {
		return;
	}
	editor->push_item(obj);
}

void InspectorDock::_resource_created() {
	Variant c = new_resource_dialog->instance_selected();

	ERR_FAIL_COND(!c);
	Resource *r = Object::cast_to<Resource>(c);
	ERR_FAIL_COND(!r);

	editor->push_item(r);
}

void InspectorDock::_resource_selected(const RES &p_res, const String &p_property) {
	if (p_res.is_null()) {
		return;
	}

	RES r = p_res;
	editor->push_item(r.operator->(), p_property);
}

void InspectorDock::_edit_forward() {
	if (EditorNode::get_singleton()->get_editor_history()->next()) {
		editor->edit_current();
	}
}

void InspectorDock::_edit_back() {
	EditorHistory *editor_history = EditorNode::get_singleton()->get_editor_history();
	if ((current && editor_history->previous()) || editor_history->get_path_size() == 1) {
		editor->edit_current();
	}
}

void InspectorDock::_menu_collapseall() {
	inspector->collapse_all_folding();
}

void InspectorDock::_menu_expandall() {
	inspector->expand_all_folding();
}

void InspectorDock::_property_keyed(const String &p_keyed, const Variant &p_value, bool p_advance) {
	AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_value_key(p_keyed, p_value, p_advance);
}

void InspectorDock::_transform_keyed(Object *sp, const String &p_sub, const Transform3D &p_key) {
	Node3D *s = Object::cast_to<Node3D>(sp);
	if (!s) {
		return;
	}
	AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_transform_key(s, p_sub, Animation::TYPE_POSITION_3D, p_key.origin);
	AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_transform_key(s, p_sub, Animation::TYPE_ROTATION_3D, p_key.basis.get_rotation_quaternion());
	AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_transform_key(s, p_sub, Animation::TYPE_SCALE_3D, p_key.basis.get_scale());
}

void InspectorDock::_warning_pressed() {
	warning_dialog->popup_centered();
}

Container *InspectorDock::get_addon_area() {
	return this;
}

void InspectorDock::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			set_theme(editor->get_gui_base()->get_theme());

			resource_new_button->set_icon(get_theme_icon(SNAME("New"), SNAME("EditorIcons")));
			resource_load_button->set_icon(get_theme_icon(SNAME("Load"), SNAME("EditorIcons")));
			resource_save_button->set_icon(get_theme_icon(SNAME("Save"), SNAME("EditorIcons")));
			resource_extra_button->set_icon(get_theme_icon(SNAME("GuiTabMenuHl"), SNAME("EditorIcons")));

			PopupMenu *resource_extra_popup = resource_extra_button->get_popup();
			resource_extra_popup->set_item_icon(resource_extra_popup->get_item_index(RESOURCE_EDIT_CLIPBOARD), get_theme_icon(SNAME("ActionPaste"), SNAME("EditorIcons")));
			resource_extra_popup->set_item_icon(resource_extra_popup->get_item_index(RESOURCE_COPY), get_theme_icon(SNAME("ActionCopy"), SNAME("EditorIcons")));

			if (is_layout_rtl()) {
				backward_button->set_icon(get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
				forward_button->set_icon(get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
			} else {
				backward_button->set_icon(get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
				forward_button->set_icon(get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
			}

			history_menu->set_icon(get_theme_icon(SNAME("History"), SNAME("EditorIcons")));
			object_menu->set_icon(get_theme_icon(SNAME("Tools"), SNAME("EditorIcons")));
			warning->set_icon(get_theme_icon(SNAME("NodeWarning"), SNAME("EditorIcons")));
			warning->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), SNAME("Editor")));
		} break;
	}
}

void InspectorDock::_bind_methods() {
	ClassDB::bind_method("update_keying", &InspectorDock::update_keying);
	ClassDB::bind_method("_transform_keyed", &InspectorDock::_transform_keyed); // Still used by some connect_compat.

	ClassDB::bind_method("_unref_resource", &InspectorDock::_unref_resource);
	ClassDB::bind_method("_paste_resource", &InspectorDock::_paste_resource);
	ClassDB::bind_method("_copy_resource", &InspectorDock::_copy_resource);

	ClassDB::bind_method("_menu_collapseall", &InspectorDock::_menu_collapseall);
	ClassDB::bind_method("_menu_expandall", &InspectorDock::_menu_expandall);

	ADD_SIGNAL(MethodInfo("request_help"));
}

void InspectorDock::edit_resource(const Ref<Resource> &p_resource) {
	_resource_selected(p_resource, "");
}

void InspectorDock::open_resource(const String &p_type) {
	_load_resource(p_type);
}

void InspectorDock::set_warning(const String &p_message) {
	warning->hide();
	if (!p_message.is_empty()) {
		warning->show();
		warning_dialog->set_text(p_message);
	}
}

void InspectorDock::clear() {
}

void InspectorDock::update(Object *p_object) {
	EditorHistory *editor_history = EditorNode::get_singleton()->get_editor_history();
	backward_button->set_disabled(editor_history->is_at_beginning());
	forward_button->set_disabled(editor_history->is_at_end());

	history_menu->set_disabled(true);
	if (editor_history->get_history_len() > 0) {
		history_menu->set_disabled(false);
	}
	editor_path->update_path();

	current = p_object;

	const bool is_object = p_object != nullptr;
	const bool is_resource = is_object && p_object->is_class("Resource");
	const bool is_text_file = is_object && p_object->is_class("TextFile");
	const bool is_node = is_object && p_object->is_class("Node");

	object_menu->set_disabled(!is_object || is_text_file);
	search->set_editable(is_object && !is_text_file);
	resource_save_button->set_disabled(!is_resource || is_text_file);
	open_docs_button->set_disabled(is_text_file || (!is_resource && !is_node));

	PopupMenu *resource_extra_popup = resource_extra_button->get_popup();
	resource_extra_popup->set_item_disabled(resource_extra_popup->get_item_index(RESOURCE_COPY), !is_resource || is_text_file);
	resource_extra_popup->set_item_disabled(resource_extra_popup->get_item_index(RESOURCE_MAKE_BUILT_IN), !is_resource || is_text_file);

	if (!is_object || is_text_file) {
		warning->hide();
		editor_path->clear_path();
		return;
	}

	editor_path->enable_path();

	PopupMenu *p = object_menu->get_popup();

	p->clear();
	p->add_icon_shortcut(get_theme_icon(SNAME("GuiTreeArrowDown"), SNAME("EditorIcons")), ED_SHORTCUT("property_editor/expand_all", TTR("Expand All")), EXPAND_ALL);
	p->add_icon_shortcut(get_theme_icon(SNAME("GuiTreeArrowRight"), SNAME("EditorIcons")), ED_SHORTCUT("property_editor/collapse_all", TTR("Collapse All")), COLLAPSE_ALL);
	p->add_separator();

	p->add_shortcut(ED_SHORTCUT("property_editor/copy_params", TTR("Copy Properties")), OBJECT_COPY_PARAMS);
	p->add_shortcut(ED_SHORTCUT("property_editor/paste_params", TTR("Paste Properties")), OBJECT_PASTE_PARAMS);

	if (is_resource || is_node) {
		p->add_separator();
		p->add_shortcut(ED_SHORTCUT("property_editor/make_subresources_unique", TTR("Make Sub-Resources Unique")), OBJECT_UNIQUE_RESOURCES);
	}

	List<MethodInfo> methods;
	p_object->get_method_list(&methods);

	if (!methods.is_empty()) {
		bool found = false;
		List<MethodInfo>::Element *I = methods.front();
		int i = 0;
		while (I) {
			if (I->get().flags & METHOD_FLAG_EDITOR) {
				if (!found) {
					p->add_separator();
					found = true;
				}
				p->add_item(I->get().name.capitalize(), OBJECT_METHOD_BASE + i);
			}
			i++;
			I = I->next();
		}
	}
}

void InspectorDock::go_back() {
	_edit_back();
}

void InspectorDock::update_keying() {
	bool valid = false;

	if (AnimationPlayerEditor::get_singleton()->get_track_editor()->has_keying()) {
		EditorHistory *editor_history = EditorNode::get_singleton()->get_editor_history();
		if (editor_history->get_path_size() >= 1) {
			Object *obj = ObjectDB::get_instance(editor_history->get_path_object(0));
			if (Object::cast_to<Node>(obj)) {
				valid = true;
			}
		}
	}

	inspector->set_keying(valid);
}

InspectorDock::InspectorDock(EditorNode *p_editor, EditorData &p_editor_data) {
	set_name("Inspector");
	set_theme(p_editor->get_gui_base()->get_theme());

	editor = p_editor;
	editor_data = &p_editor_data;

	HBoxContainer *general_options_hb = memnew(HBoxContainer);
	add_child(general_options_hb);

	resource_new_button = memnew(Button);
	resource_new_button->set_flat(true);
	resource_new_button->set_tooltip(TTR("Create a new resource in memory and edit it."));
	resource_new_button->set_icon(get_theme_icon(SNAME("New"), SNAME("EditorIcons")));
	general_options_hb->add_child(resource_new_button);
	resource_new_button->connect("pressed", callable_mp(this, &InspectorDock::_new_resource));
	resource_new_button->set_focus_mode(Control::FOCUS_NONE);

	resource_load_button = memnew(Button);
	resource_load_button->set_flat(true);
	resource_load_button->set_tooltip(TTR("Load an existing resource from disk and edit it."));
	resource_load_button->set_icon(get_theme_icon(SNAME("Load"), SNAME("EditorIcons")));
	general_options_hb->add_child(resource_load_button);
	resource_load_button->connect("pressed", callable_mp(this, &InspectorDock::_open_resource_selector));
	resource_load_button->set_focus_mode(Control::FOCUS_NONE);

	resource_save_button = memnew(MenuButton);
	resource_save_button->set_tooltip(TTR("Save the currently edited resource."));
	resource_save_button->set_icon(get_theme_icon(SNAME("Save"), SNAME("EditorIcons")));
	general_options_hb->add_child(resource_save_button);
	resource_save_button->get_popup()->add_item(TTR("Save"), RESOURCE_SAVE);
	resource_save_button->get_popup()->add_item(TTR("Save As..."), RESOURCE_SAVE_AS);
	resource_save_button->get_popup()->connect("id_pressed", callable_mp(this, &InspectorDock::_menu_option));
	resource_save_button->set_focus_mode(Control::FOCUS_NONE);
	resource_save_button->set_disabled(true);

	resource_extra_button = memnew(MenuButton);
	resource_extra_button->set_icon(get_theme_icon(SNAME("GuiTabMenuHl"), SNAME("EditorIcons")));
	resource_extra_button->set_tooltip(TTR("Extra resource options."));
	general_options_hb->add_child(resource_extra_button);
	resource_extra_button->connect("about_to_popup", callable_mp(this, &InspectorDock::_prepare_resource_extra_popup));
	resource_extra_button->get_popup()->add_icon_shortcut(get_theme_icon(SNAME("ActionPaste"), SNAME("EditorIcons")), ED_SHORTCUT("property_editor/paste_resource", TTR("Edit Resource from Clipboard")), RESOURCE_EDIT_CLIPBOARD);
	resource_extra_button->get_popup()->add_icon_shortcut(get_theme_icon(SNAME("ActionCopy"), SNAME("EditorIcons")), ED_SHORTCUT("property_editor/copy_resource", TTR("Copy Resource")), RESOURCE_COPY);
	resource_extra_button->get_popup()->set_item_disabled(1, true);
	resource_extra_button->get_popup()->add_separator();
	resource_extra_button->get_popup()->add_shortcut(ED_SHORTCUT("property_editor/unref_resource", TTR("Make Resource Built-In")), RESOURCE_MAKE_BUILT_IN);
	resource_extra_button->get_popup()->set_item_disabled(3, true);
	resource_extra_button->get_popup()->connect("id_pressed", callable_mp(this, &InspectorDock::_menu_option));

	general_options_hb->add_spacer();

	backward_button = memnew(Button);
	backward_button->set_flat(true);
	general_options_hb->add_child(backward_button);
	if (is_layout_rtl()) {
		backward_button->set_icon(get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
	} else {
		backward_button->set_icon(get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
	}
	backward_button->set_tooltip(TTR("Go to the previous edited object in history."));
	backward_button->set_disabled(true);
	backward_button->connect("pressed", callable_mp(this, &InspectorDock::_edit_back));

	forward_button = memnew(Button);
	forward_button->set_flat(true);
	general_options_hb->add_child(forward_button);
	if (is_layout_rtl()) {
		forward_button->set_icon(get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
	} else {
		forward_button->set_icon(get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
	}
	forward_button->set_tooltip(TTR("Go to the next edited object in history."));
	forward_button->set_disabled(true);
	forward_button->connect("pressed", callable_mp(this, &InspectorDock::_edit_forward));

	history_menu = memnew(MenuButton);
	history_menu->set_tooltip(TTR("History of recently edited objects."));
	history_menu->set_icon(get_theme_icon(SNAME("History"), SNAME("EditorIcons")));
	general_options_hb->add_child(history_menu);
	history_menu->connect("about_to_popup", callable_mp(this, &InspectorDock::_prepare_history));
	history_menu->get_popup()->connect("id_pressed", callable_mp(this, &InspectorDock::_select_history));

	HBoxContainer *subresource_hb = memnew(HBoxContainer);
	add_child(subresource_hb);
	editor_path = memnew(EditorPath(editor->get_editor_history()));
	editor_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	subresource_hb->add_child(editor_path);

	open_docs_button = memnew(Button);
	open_docs_button->set_flat(true);
	open_docs_button->set_disabled(true);
	open_docs_button->set_tooltip(TTR("Open documentation for this object."));
	open_docs_button->set_icon(get_theme_icon(SNAME("HelpSearch"), SNAME("EditorIcons")));
	open_docs_button->set_shortcut(ED_SHORTCUT("property_editor/open_help", TTR("Open Documentation")));
	subresource_hb->add_child(open_docs_button);
	open_docs_button->connect("pressed", callable_mp(this, &InspectorDock::_menu_option), varray(OBJECT_REQUEST_HELP));

	new_resource_dialog = memnew(CreateDialog);
	editor->get_gui_base()->add_child(new_resource_dialog);
	new_resource_dialog->set_base_type("Resource");
	new_resource_dialog->connect("create", callable_mp(this, &InspectorDock::_resource_created));

	HBoxContainer *property_tools_hb = memnew(HBoxContainer);
	add_child(property_tools_hb);

	search = memnew(LineEdit);
	search->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search->set_placeholder(TTR("Filter properties"));
	search->set_right_icon(get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
	search->set_clear_button_enabled(true);
	property_tools_hb->add_child(search);

	object_menu = memnew(MenuButton);
	object_menu->set_shortcut_context(this);
	object_menu->set_icon(get_theme_icon(SNAME("Tools"), SNAME("EditorIcons")));
	property_tools_hb->add_child(object_menu);
	object_menu->set_tooltip(TTR("Manage object properties."));
	object_menu->get_popup()->connect("id_pressed", callable_mp(this, &InspectorDock::_menu_option));

	warning = memnew(Button);
	add_child(warning);
	warning->set_text(TTR("Changes may be lost!"));
	warning->set_icon(get_theme_icon(SNAME("NodeWarning"), SNAME("EditorIcons")));
	warning->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), SNAME("Editor")));
	warning->set_clip_text(true);
	warning->hide();
	warning->connect("pressed", callable_mp(this, &InspectorDock::_warning_pressed));

	unique_resources_confirmation = memnew(ConfirmationDialog);
	add_child(unique_resources_confirmation);

	VBoxContainer *container = memnew(VBoxContainer);
	unique_resources_confirmation->add_child(container);

	Label *top_label = memnew(Label);
	top_label->set_text(TTR("The following resources will be duplicated and embedded within this resource/object."));
	container->add_child(top_label);

	unique_resources_list_tree = memnew(Tree);
	unique_resources_list_tree->set_hide_root(true);
	unique_resources_list_tree->set_columns(1);
	unique_resources_list_tree->set_column_title(0, TTR("Property"));
	unique_resources_list_tree->set_custom_minimum_size(Size2(0, 200 * EDSCALE));
	container->add_child(unique_resources_list_tree);

	Label *bottom_label = memnew(Label);
	bottom_label->set_text(TTR("This cannot be undone. Are you sure?"));
	container->add_child(bottom_label);

	unique_resources_confirmation->connect("confirmed", callable_mp(this, &InspectorDock::_menu_confirm_current));

	warning_dialog = memnew(AcceptDialog);
	editor->get_gui_base()->add_child(warning_dialog);

	load_resource_dialog = memnew(EditorFileDialog);
	add_child(load_resource_dialog);
	load_resource_dialog->set_current_dir("res://");
	load_resource_dialog->connect("file_selected", callable_mp(this, &InspectorDock::_resource_file_selected));

	inspector = memnew(EditorInspector);
	add_child(inspector);
	inspector->set_autoclear(true);
	inspector->set_show_categories(true);
	inspector->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	inspector->set_use_doc_hints(true);
	inspector->set_hide_script(false);
	inspector->set_enable_capitalize_paths(bool(EDITOR_GET("interface/inspector/capitalize_properties")));
	inspector->set_use_folding(!bool(EDITOR_GET("interface/inspector/disable_folding")));
	inspector->register_text_enter(search);
	inspector->set_undo_redo(&editor_data->get_undo_redo());

	inspector->set_use_filter(true); // TODO: check me

	inspector->connect("resource_selected", callable_mp(this, &InspectorDock::_resource_selected));
	inspector->connect("property_keyed", callable_mp(this, &InspectorDock::_property_keyed));
}

InspectorDock::~InspectorDock() {
}

/**************************************************************************/
/*  editor_resource_picker.cpp                                            */
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

#include "editor_resource_picker.h"

#include "editor/audio_stream_preview.h"
#include "editor/editor_node.h"
#include "editor/editor_quick_open.h"
#include "editor/editor_resource_preview.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/filesystem_dock.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/plugins/editor_resource_conversion_plugin.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/scene_tree_dock.h"
#include "scene/gui/button.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/gradient_texture.h"
#include "scene/resources/image_texture.h"

void EditorResourcePicker::_update_resource() {
	String resource_path;
	if (edited_resource.is_valid() && edited_resource->get_path().is_resource_file()) {
		resource_path = edited_resource->get_path() + "\n";
	}
	String class_name = _get_resource_type(edited_resource);

	if (preview_rect) {
		preview_rect->set_texture(Ref<Texture2D>());

		assign_button->set_custom_minimum_size(assign_button_min_size);

		if (edited_resource == Ref<Resource>()) {
			assign_button->set_icon(Ref<Texture2D>());
			assign_button->set_text(TTR("<empty>"));
			assign_button->set_tooltip_text("");
		} else {
			assign_button->set_icon(EditorNode::get_singleton()->get_object_icon(edited_resource.operator->(), SNAME("Object")));

			if (!edited_resource->get_name().is_empty()) {
				assign_button->set_text(edited_resource->get_name());
			} else if (edited_resource->get_path().is_resource_file()) {
				assign_button->set_text(edited_resource->get_path().get_file());
			} else {
				assign_button->set_text(class_name);
			}

			if (edited_resource->get_path().is_resource_file()) {
				resource_path = edited_resource->get_path() + "\n";
			}
			assign_button->set_tooltip_text(resource_path + TTR("Type:") + " " + class_name);

			// Preview will override the above, so called at the end.
			EditorResourcePreview::get_singleton()->queue_edited_resource_preview(edited_resource, this, "_update_resource_preview", edited_resource->get_instance_id());
		}
	} else if (edited_resource.is_valid()) {
		assign_button->set_tooltip_text(resource_path + TTR("Type:") + " " + edited_resource->get_class());
	}

	assign_button->set_disabled(!editable && !edited_resource.is_valid());
}

void EditorResourcePicker::_update_resource_preview(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, ObjectID p_obj) {
	if (!edited_resource.is_valid() || edited_resource->get_instance_id() != p_obj) {
		return;
	}

	if (preview_rect) {
		Ref<Script> scr = edited_resource;
		if (scr.is_valid()) {
			assign_button->set_text(scr->get_path().get_file());
			return;
		}

		if (p_preview.is_valid()) {
			preview_rect->set_offset(SIDE_LEFT, assign_button->get_icon()->get_width() + assign_button->get_theme_stylebox(SNAME("normal"))->get_content_margin(SIDE_LEFT) + get_theme_constant(SNAME("h_separation"), SNAME("Button")));

			// Resource-specific stretching.
			if (Ref<GradientTexture1D>(edited_resource).is_valid() || Ref<Gradient>(edited_resource).is_valid()) {
				preview_rect->set_stretch_mode(TextureRect::STRETCH_SCALE);
				assign_button->set_custom_minimum_size(assign_button_min_size);
			} else {
				preview_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
				int thumbnail_size = EDITOR_GET("filesystem/file_dialog/thumbnail_size");
				thumbnail_size *= EDSCALE;
				assign_button->set_custom_minimum_size(Size2(MAX(1, assign_button_min_size.x), MAX(thumbnail_size, assign_button_min_size.y)));
			}

			preview_rect->set_texture(p_preview);
			assign_button->set_text("");
		}
	}
}

void EditorResourcePicker::_resource_selected() {
	if (edited_resource.is_null()) {
		edit_button->set_pressed(true);
		_update_menu();
		return;
	}

	emit_signal(SNAME("resource_selected"), edited_resource, false);
}

void EditorResourcePicker::_file_selected(const String &p_path) {
	Ref<Resource> loaded_resource = ResourceLoader::load(p_path);
	ERR_FAIL_COND_MSG(loaded_resource.is_null(), "Cannot load resource from path '" + p_path + "'.");

	if (!base_type.is_empty()) {
		bool any_type_matches = false;

		String res_type = loaded_resource->get_class();
		Ref<Script> res_script = loaded_resource->get_script();
		bool is_global_class = false;
		if (res_script.is_valid()) {
			String script_type = EditorNode::get_editor_data().script_class_get_name(res_script->get_path());
			if (!script_type.is_empty()) {
				is_global_class = true;
				res_type = script_type;
			}
		}

		for (int i = 0; i < base_type.get_slice_count(","); i++) {
			String base = base_type.get_slice(",", i);

			any_type_matches = is_global_class ? EditorNode::get_editor_data().script_class_is_parent(res_type, base) : loaded_resource->is_class(base);

			if (any_type_matches) {
				break;
			}
		}

		if (!any_type_matches) {
			EditorNode::get_singleton()->show_warning(vformat(TTR("The selected resource (%s) does not match any type expected for this property (%s)."), res_type, base_type));
			return;
		}
	}

	edited_resource = loaded_resource;
	emit_signal(SNAME("resource_changed"), edited_resource);
	_update_resource();
}

void EditorResourcePicker::_file_quick_selected() {
	_file_selected(quick_open->get_selected());
}

void EditorResourcePicker::_update_menu() {
	_update_menu_items();

	Rect2 gt = edit_button->get_screen_rect();
	edit_menu->reset_size();
	int ms = edit_menu->get_contents_minimum_size().width;
	Vector2 popup_pos = gt.get_end() - Vector2(ms, 0);
	edit_menu->set_position(popup_pos);
	edit_menu->popup();
}

void EditorResourcePicker::_update_menu_items() {
	_ensure_resource_menu();
	edit_menu->clear();

	// Add options for creating specific subtypes of the base resource type.
	if (is_editable()) {
		set_create_options(edit_menu);

		// Add an option to load a resource from a file using the QuickOpen dialog.
		edit_menu->add_icon_item(get_editor_theme_icon(SNAME("Load")), TTR("Quick Load"), OBJ_MENU_QUICKLOAD);

		// Add an option to load a resource from a file using the regular file dialog.
		edit_menu->add_icon_item(get_editor_theme_icon(SNAME("Load")), TTR("Load"), OBJ_MENU_LOAD);
	}

	// Add options for changing existing value of the resource.
	if (edited_resource.is_valid()) {
		// Determine if the edited resource is part of another scene (foreign) which was imported
		bool is_edited_resource_foreign_import = EditorNode::get_singleton()->is_resource_read_only(edited_resource, true);

		// If the resource is determined to be foreign and imported, change the menu entry's description to 'inspect' rather than 'edit'
		// since will only be able to view its properties in read-only mode.
		if (is_edited_resource_foreign_import) {
			// The 'Search' icon is a magnifying glass, which seems appropriate, but maybe a bespoke icon is preferred here.
			edit_menu->add_icon_item(get_editor_theme_icon(SNAME("Search")), TTR("Inspect"), OBJ_MENU_INSPECT);
		} else {
			edit_menu->add_icon_item(get_editor_theme_icon(SNAME("Edit")), TTR("Edit"), OBJ_MENU_INSPECT);
		}

		if (is_editable()) {
			edit_menu->add_icon_item(get_editor_theme_icon(SNAME("Clear")), TTR("Clear"), OBJ_MENU_CLEAR);
			edit_menu->add_icon_item(get_editor_theme_icon(SNAME("Duplicate")), TTR("Make Unique"), OBJ_MENU_MAKE_UNIQUE);

			// Check whether the resource has subresources.
			List<PropertyInfo> property_list;
			edited_resource->get_property_list(&property_list);
			bool has_subresources = false;
			for (PropertyInfo &p : property_list) {
				if ((p.type == Variant::OBJECT) && (p.hint == PROPERTY_HINT_RESOURCE_TYPE) && (p.name != "script") && ((Object *)edited_resource->get(p.name) != nullptr)) {
					has_subresources = true;
					break;
				}
			}
			if (has_subresources) {
				edit_menu->add_icon_item(get_editor_theme_icon(SNAME("Duplicate")), TTR("Make Unique (Recursive)"), OBJ_MENU_MAKE_UNIQUE_RECURSIVE);
			}

			edit_menu->add_icon_item(get_editor_theme_icon(SNAME("Save")), TTR("Save"), OBJ_MENU_SAVE);
			edit_menu->add_icon_item(get_editor_theme_icon(SNAME("Save")), TTR("Save As..."), OBJ_MENU_SAVE_AS);
		}

		if (edited_resource->get_path().is_resource_file()) {
			edit_menu->add_separator();
			edit_menu->add_item(TTR("Show in FileSystem"), OBJ_MENU_SHOW_IN_FILE_SYSTEM);
		}
	}

	// Add options to copy/paste resource.
	Ref<Resource> cb = EditorSettings::get_singleton()->get_resource_clipboard();
	bool paste_valid = false;
	if (is_editable() && cb.is_valid()) {
		if (base_type.is_empty()) {
			paste_valid = true;
		} else {
			String res_type = _get_resource_type(cb);

			for (int i = 0; i < base_type.get_slice_count(","); i++) {
				String base = base_type.get_slice(",", i);

				paste_valid = ClassDB::is_parent_class(res_type, base) || EditorNode::get_editor_data().script_class_is_parent(res_type, base);

				if (paste_valid) {
					break;
				}
			}
		}
	}

	if (edited_resource.is_valid() || paste_valid) {
		edit_menu->add_separator();

		if (edited_resource.is_valid()) {
			edit_menu->add_item(TTR("Copy"), OBJ_MENU_COPY);
		}

		if (paste_valid) {
			edit_menu->add_item(TTR("Paste"), OBJ_MENU_PASTE);
		}
	}

	// Add options to convert existing resource to another type of resource.
	if (is_editable() && edited_resource.is_valid()) {
		Vector<Ref<EditorResourceConversionPlugin>> conversions = EditorNode::get_singleton()->find_resource_conversion_plugin(edited_resource);
		if (conversions.size()) {
			edit_menu->add_separator();
		}
		for (int i = 0; i < conversions.size(); i++) {
			String what = conversions[i]->converts_to();
			Ref<Texture2D> icon;
			if (has_theme_icon(what, EditorStringName(EditorIcons))) {
				icon = get_editor_theme_icon(what);
			} else {
				icon = get_theme_icon(what, SNAME("Resource"));
			}

			edit_menu->add_icon_item(icon, vformat(TTR("Convert to %s"), what), CONVERT_BASE_ID + i);
		}
	}
}

void EditorResourcePicker::_edit_menu_cbk(int p_which) {
	switch (p_which) {
		case OBJ_MENU_LOAD: {
			List<String> extensions;
			for (int i = 0; i < base_type.get_slice_count(","); i++) {
				String base = base_type.get_slice(",", i);
				ResourceLoader::get_recognized_extensions_for_type(base, &extensions);
				if (ScriptServer::is_global_class(base)) {
					ResourceLoader::get_recognized_extensions_for_type(ScriptServer::get_global_class_native_base(base), &extensions);
				}
			}

			HashSet<String> valid_extensions;
			for (const String &E : extensions) {
				valid_extensions.insert(E);
			}

			if (!file_dialog) {
				file_dialog = memnew(EditorFileDialog);
				file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
				add_child(file_dialog);
				file_dialog->connect("file_selected", callable_mp(this, &EditorResourcePicker::_file_selected));
			}

			file_dialog->clear_filters();
			for (const String &E : valid_extensions) {
				file_dialog->add_filter("*." + E, E.to_upper());
			}

			file_dialog->popup_file_dialog();
		} break;

		case OBJ_MENU_QUICKLOAD: {
			if (!quick_open) {
				quick_open = memnew(EditorQuickOpen);
				add_child(quick_open);
				quick_open->connect("quick_open", callable_mp(this, &EditorResourcePicker::_file_quick_selected));
			}

			quick_open->popup_dialog(base_type);
			quick_open->set_title(TTR("Resource"));
		} break;

		case OBJ_MENU_INSPECT: {
			if (edited_resource.is_valid()) {
				emit_signal(SNAME("resource_selected"), edited_resource, true);
			}
		} break;

		case OBJ_MENU_CLEAR: {
			edited_resource = Ref<Resource>();
			emit_signal(SNAME("resource_changed"), edited_resource);
			_update_resource();
		} break;

		case OBJ_MENU_MAKE_UNIQUE: {
			if (edited_resource.is_null()) {
				return;
			}

			Ref<Resource> unique_resource = edited_resource->duplicate();
			ERR_FAIL_COND(unique_resource.is_null()); // duplicate() may fail.

			edited_resource = unique_resource;
			emit_signal(SNAME("resource_changed"), edited_resource);
			_update_resource();
		} break;

		case OBJ_MENU_MAKE_UNIQUE_RECURSIVE: {
			if (edited_resource.is_null()) {
				return;
			}

			if (!duplicate_resources_dialog) {
				duplicate_resources_dialog = memnew(ConfirmationDialog);
				add_child(duplicate_resources_dialog);
				duplicate_resources_dialog->set_title(TTR("Make Unique (Recursive)"));
				duplicate_resources_dialog->connect("confirmed", callable_mp(this, &EditorResourcePicker::_duplicate_selected_resources));

				VBoxContainer *vb = memnew(VBoxContainer);
				duplicate_resources_dialog->add_child(vb);

				Label *label = memnew(Label(TTR("Select resources to make unique:")));
				vb->add_child(label);

				duplicate_resources_tree = memnew(Tree);
				vb->add_child(duplicate_resources_tree);
				duplicate_resources_tree->set_columns(2);
				duplicate_resources_tree->set_v_size_flags(SIZE_EXPAND_FILL);
			}

			duplicate_resources_tree->clear();
			TreeItem *root = duplicate_resources_tree->create_item();
			_gather_resources_to_duplicate(edited_resource, root);

			duplicate_resources_dialog->reset_size();
			duplicate_resources_dialog->popup_centered(Vector2(500, 400) * EDSCALE);
		} break;

		case OBJ_MENU_SAVE: {
			if (edited_resource.is_null()) {
				return;
			}
			EditorNode::get_singleton()->save_resource(edited_resource);
		} break;

		case OBJ_MENU_SAVE_AS: {
			if (edited_resource.is_null()) {
				return;
			}
			EditorNode::get_singleton()->save_resource_as(edited_resource);
		} break;

		case OBJ_MENU_COPY: {
			EditorSettings::get_singleton()->set_resource_clipboard(edited_resource);
		} break;

		case OBJ_MENU_PASTE: {
			edited_resource = EditorSettings::get_singleton()->get_resource_clipboard();
			if (edited_resource->is_built_in() && EditorNode::get_singleton()->get_edited_scene() &&
					edited_resource->get_path().get_slice("::", 0) != EditorNode::get_singleton()->get_edited_scene()->get_scene_file_path()) {
				// Automatically make resource unique if it belongs to another scene.
				_edit_menu_cbk(OBJ_MENU_MAKE_UNIQUE);
				return;
			}

			emit_signal(SNAME("resource_changed"), edited_resource);
			_update_resource();
		} break;

		case OBJ_MENU_SHOW_IN_FILE_SYSTEM: {
			FileSystemDock *file_system_dock = FileSystemDock::get_singleton();
			file_system_dock->navigate_to_path(edited_resource->get_path());

			// Ensure that the FileSystem dock is visible.
			if (file_system_dock->get_window() == get_tree()->get_root()) {
				TabContainer *tab_container = (TabContainer *)file_system_dock->get_parent_control();
				tab_container->set_current_tab(tab_container->get_tab_idx_from_control(file_system_dock));
			} else {
				file_system_dock->get_window()->grab_focus();
			}
		} break;

		default: {
			// Allow subclasses to handle their own options first, only then fallback on the default branch logic.
			if (handle_menu_selected(p_which)) {
				break;
			}

			if (p_which >= CONVERT_BASE_ID) {
				int to_type = p_which - CONVERT_BASE_ID;
				Vector<Ref<EditorResourceConversionPlugin>> conversions = EditorNode::get_singleton()->find_resource_conversion_plugin(edited_resource);
				ERR_FAIL_INDEX(to_type, conversions.size());

				edited_resource = conversions[to_type]->convert(edited_resource);
				emit_signal(SNAME("resource_changed"), edited_resource);
				_update_resource();
				break;
			}

			ERR_FAIL_COND(inheritors_array.is_empty());

			String intype = inheritors_array[p_which - TYPE_BASE_ID];
			Variant obj;

			if (ScriptServer::is_global_class(intype)) {
				obj = EditorNode::get_editor_data().script_class_instance(intype);
			} else {
				obj = ClassDB::instantiate(intype);
			}

			if (!obj) {
				obj = EditorNode::get_editor_data().instantiate_custom_type(intype, "Resource");
			}

			Resource *resp = Object::cast_to<Resource>(obj);
			ERR_BREAK(!resp);

			EditorNode::get_editor_data().instantiate_object_properties(obj);

			edited_resource = Ref<Resource>(resp);
			emit_signal(SNAME("resource_changed"), edited_resource);
			_update_resource();
		} break;
	}
}

void EditorResourcePicker::set_create_options(Object *p_menu_node) {
	_ensure_resource_menu();
	// If a subclass implements this method, use it to replace all create items.
	if (GDVIRTUAL_CALL(_set_create_options, p_menu_node)) {
		return;
	}

	// By default provide generic "New ..." options.
	if (!base_type.is_empty()) {
		int idx = 0;

		HashSet<StringName> allowed_types;
		_get_allowed_types(false, &allowed_types);

		Vector<EditorData::CustomType> custom_resources;
		if (EditorNode::get_editor_data().get_custom_types().has("Resource")) {
			custom_resources = EditorNode::get_editor_data().get_custom_types()["Resource"];
		}

		for (const StringName &E : allowed_types) {
			const String &t = E;

			bool is_custom_resource = false;
			Ref<Texture2D> icon;
			if (!custom_resources.is_empty()) {
				for (int j = 0; j < custom_resources.size(); j++) {
					if (custom_resources[j].name == t) {
						is_custom_resource = true;
						if (custom_resources[j].icon.is_valid()) {
							icon = custom_resources[j].icon;
						}
						break;
					}
				}
			}

			if (!is_custom_resource && !ClassDB::can_instantiate(t)) {
				continue;
			}

			inheritors_array.push_back(t);

			if (!icon.is_valid()) {
				icon = get_editor_theme_icon(has_theme_icon(t, EditorStringName(EditorIcons)) ? t : String("Object"));
			}

			int id = TYPE_BASE_ID + idx;
			edit_menu->add_icon_item(icon, vformat(TTR("New %s"), t), id);

			idx++;
		}

		if (edit_menu->get_item_count()) {
			edit_menu->add_separator();
		}
	}
}

bool EditorResourcePicker::handle_menu_selected(int p_which) {
	bool success = false;
	GDVIRTUAL_CALL(_handle_menu_selected, p_which, success);
	return success;
}

void EditorResourcePicker::_button_draw() {
	if (dropping) {
		Color color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
		assign_button->draw_rect(Rect2(Point2(), assign_button->get_size()), color, false);
	}
}

void EditorResourcePicker::_button_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::RIGHT) {
		// Only attempt to update and show the menu if we have
		// a valid resource or the Picker is editable, as
		// there will otherwise be nothing to display.
		if (edited_resource.is_valid() || is_editable()) {
			_update_menu_items();

			Vector2 pos = get_screen_position() + mb->get_position();
			edit_menu->reset_size();
			edit_menu->set_position(pos);
			edit_menu->popup();
		}
	}
}

String EditorResourcePicker::_get_resource_type(const Ref<Resource> &p_resource) const {
	if (p_resource.is_null()) {
		return String();
	}
	String res_type = p_resource->get_class();

	Ref<Script> res_script = p_resource->get_script();
	if (res_script.is_null()) {
		return res_type;
	}

	// TODO: Replace with EditorFileSystem when PR #60606 is merged to use cached resource type.
	String script_type = EditorNode::get_editor_data().script_class_get_name(res_script->get_path());
	if (!script_type.is_empty()) {
		res_type = script_type;
	}
	return res_type;
}

static void _add_allowed_type(const StringName &p_type, HashSet<StringName> *p_vector) {
	if (p_vector->has(p_type)) {
		// Already added
		return;
	}

	if (ClassDB::class_exists(p_type)) {
		// Engine class,

		if (!ClassDB::is_virtual(p_type)) {
			p_vector->insert(p_type);
		}

		List<StringName> inheriters;
		ClassDB::get_inheriters_from_class(p_type, &inheriters);
		for (const StringName &S : inheriters) {
			_add_allowed_type(S, p_vector);
		}
	} else {
		// Script class.
		p_vector->insert(p_type);
	}

	List<StringName> inheriters;
	ScriptServer::get_inheriters_list(p_type, &inheriters);
	for (const StringName &S : inheriters) {
		_add_allowed_type(S, p_vector);
	}
}

void EditorResourcePicker::_get_allowed_types(bool p_with_convert, HashSet<StringName> *p_vector) const {
	Vector<String> allowed_types = base_type.split(",");
	int size = allowed_types.size();

	for (int i = 0; i < size; i++) {
		String base = allowed_types[i].strip_edges();

		_add_allowed_type(base, p_vector);

		if (p_with_convert) {
			if (base == "BaseMaterial3D") {
				p_vector->insert("Texture2D");
			} else if (base == "ShaderMaterial") {
				p_vector->insert("Shader");
			} else if (base == "Texture2D") {
				p_vector->insert("Image");
			}
		}
	}
}

bool EditorResourcePicker::_is_drop_valid(const Dictionary &p_drag_data) const {
	if (base_type.is_empty()) {
		return true;
	}

	Dictionary drag_data = p_drag_data;

	Ref<Resource> res;
	if (drag_data.has("type") && String(drag_data["type"]) == "script_list_element") {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(drag_data["script_list_element"]);
		if (se) {
			res = se->get_edited_resource();
		}
	} else if (drag_data.has("type") && String(drag_data["type"]) == "resource") {
		res = drag_data["resource"];
	} else if (drag_data.has("type") && String(drag_data["type"]) == "files") {
		Vector<String> files = drag_data["files"];

		// TODO: Extract the typename of the dropped filepath's resource in a more performant way, without fully loading it.
		if (files.size() == 1) {
			String file = files[0];
			res = ResourceLoader::load(file);
		}
	}

	HashSet<StringName> allowed_types;
	_get_allowed_types(true, &allowed_types);

	if (res.is_valid()) {
		String res_type = _get_resource_type(res);

		if (_is_type_valid(res_type, allowed_types)) {
			return true;
		}

		StringName custom_class = EditorNode::get_singleton()->get_object_custom_type_name(res.ptr());
		if (_is_type_valid(custom_class, allowed_types)) {
			return true;
		}
	}

	return false;
}

bool EditorResourcePicker::_is_type_valid(const String p_type_name, HashSet<StringName> p_allowed_types) const {
	for (const StringName &E : p_allowed_types) {
		String at = E;
		if (p_type_name == at || ClassDB::is_parent_class(p_type_name, at) || EditorNode::get_editor_data().script_class_is_parent(p_type_name, at)) {
			return true;
		}
	}

	return false;
}

Variant EditorResourcePicker::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	if (edited_resource.is_valid()) {
		return EditorNode::get_singleton()->drag_resource(edited_resource, p_from);
	}

	return Variant();
}

bool EditorResourcePicker::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	return editable && _is_drop_valid(p_data);
}

void EditorResourcePicker::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	ERR_FAIL_COND(!_is_drop_valid(p_data));

	Dictionary drag_data = p_data;

	Ref<Resource> dropped_resource;
	if (drag_data.has("type") && String(drag_data["type"]) == "script_list_element") {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(drag_data["script_list_element"]);
		if (se) {
			dropped_resource = se->get_edited_resource();
		}
	} else if (drag_data.has("type") && String(drag_data["type"]) == "resource") {
		dropped_resource = drag_data["resource"];
	}

	if (!dropped_resource.is_valid() && drag_data.has("type") && String(drag_data["type"]) == "files") {
		Vector<String> files = drag_data["files"];

		if (files.size() == 1) {
			String file = files[0];
			dropped_resource = ResourceLoader::load(file);
		}
	}

	if (dropped_resource.is_valid()) {
		HashSet<StringName> allowed_types;
		_get_allowed_types(false, &allowed_types);

		String res_type = _get_resource_type(dropped_resource);

		// If the accepted dropped resource is from the extended list, it requires conversion.
		if (!_is_type_valid(res_type, allowed_types)) {
			for (const StringName &E : allowed_types) {
				String at = E;

				if (at == "BaseMaterial3D" && Ref<Texture2D>(dropped_resource).is_valid()) {
					// Use existing resource if possible and only replace its data.
					Ref<StandardMaterial3D> mat = edited_resource;
					if (!mat.is_valid()) {
						mat.instantiate();
					}
					mat->set_texture(StandardMaterial3D::TextureParam::TEXTURE_ALBEDO, dropped_resource);
					dropped_resource = mat;
					break;
				}

				if (at == "ShaderMaterial" && Ref<Shader>(dropped_resource).is_valid()) {
					Ref<ShaderMaterial> mat = edited_resource;
					if (!mat.is_valid()) {
						mat.instantiate();
					}
					mat->set_shader(dropped_resource);
					dropped_resource = mat;
					break;
				}

				if (at == "ImageTexture" && Ref<Image>(dropped_resource).is_valid()) {
					Ref<ImageTexture> texture = edited_resource;
					if (!texture.is_valid()) {
						texture.instantiate();
					}
					texture->set_image(dropped_resource);
					dropped_resource = texture;
					break;
				}
			}
		}

		edited_resource = dropped_resource;
		emit_signal(SNAME("resource_changed"), edited_resource);
		_update_resource();
	}
}

void EditorResourcePicker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_resource_preview"), &EditorResourcePicker::_update_resource_preview);

	ClassDB::bind_method(D_METHOD("set_base_type", "base_type"), &EditorResourcePicker::set_base_type);
	ClassDB::bind_method(D_METHOD("get_base_type"), &EditorResourcePicker::get_base_type);
	ClassDB::bind_method(D_METHOD("get_allowed_types"), &EditorResourcePicker::get_allowed_types);
	ClassDB::bind_method(D_METHOD("set_edited_resource", "resource"), &EditorResourcePicker::set_edited_resource);
	ClassDB::bind_method(D_METHOD("get_edited_resource"), &EditorResourcePicker::get_edited_resource);
	ClassDB::bind_method(D_METHOD("set_toggle_mode", "enable"), &EditorResourcePicker::set_toggle_mode);
	ClassDB::bind_method(D_METHOD("is_toggle_mode"), &EditorResourcePicker::is_toggle_mode);
	ClassDB::bind_method(D_METHOD("set_toggle_pressed", "pressed"), &EditorResourcePicker::set_toggle_pressed);
	ClassDB::bind_method(D_METHOD("set_editable", "enable"), &EditorResourcePicker::set_editable);
	ClassDB::bind_method(D_METHOD("is_editable"), &EditorResourcePicker::is_editable);

	GDVIRTUAL_BIND(_set_create_options, "menu_node");
	GDVIRTUAL_BIND(_handle_menu_selected, "id");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_type"), "set_base_type", "get_base_type");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "edited_resource", PROPERTY_HINT_RESOURCE_TYPE, "Resource", PROPERTY_USAGE_NONE), "set_edited_resource", "get_edited_resource");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editable"), "set_editable", "is_editable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "toggle_mode"), "set_toggle_mode", "is_toggle_mode");

	ADD_SIGNAL(MethodInfo("resource_selected", PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "Resource"), PropertyInfo(Variant::BOOL, "inspect")));
	ADD_SIGNAL(MethodInfo("resource_changed", PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "Resource")));
}

void EditorResourcePicker::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_resource();
			[[fallthrough]];
		}
		case NOTIFICATION_THEME_CHANGED: {
			assign_button->add_theme_constant_override("icon_max_width", get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor)));
			edit_button->set_icon(get_theme_icon(SNAME("select_arrow"), SNAME("Tree")));
		} break;

		case NOTIFICATION_DRAW: {
			draw_style_box(get_theme_stylebox(SNAME("panel"), SNAME("Tree")), Rect2(Point2(), get_size()));
		} break;

		case NOTIFICATION_DRAG_BEGIN: {
			if (editable && _is_drop_valid(get_viewport()->gui_get_drag_data())) {
				dropping = true;
				assign_button->queue_redraw();
			}
		} break;

		case NOTIFICATION_DRAG_END: {
			if (dropping) {
				dropping = false;
				assign_button->queue_redraw();
			}
		} break;
	}
}

void EditorResourcePicker::set_assign_button_min_size(const Size2i &p_size) {
	assign_button_min_size = p_size;
	assign_button->set_custom_minimum_size(assign_button_min_size);
}

void EditorResourcePicker::set_base_type(const String &p_base_type) {
	base_type = p_base_type;

	// There is a possibility that the new base type is conflicting with the existing value.
	// Keep the value, but warn the user that there is a potential mistake.
	if (!base_type.is_empty() && edited_resource.is_valid()) {
		HashSet<StringName> allowed_types;
		_get_allowed_types(true, &allowed_types);

		StringName custom_class;
		bool is_custom = false;
		if (edited_resource->get_script()) {
			custom_class = EditorNode::get_singleton()->get_object_custom_type_name(edited_resource->get_script());
			is_custom = _is_type_valid(custom_class, allowed_types);
		}

		if (!is_custom && !_is_type_valid(edited_resource->get_class(), allowed_types)) {
			String class_str = (custom_class == StringName() ? edited_resource->get_class() : vformat("%s (%s)", custom_class, edited_resource->get_class()));
			WARN_PRINT(vformat("Value mismatch between the new base type of this EditorResourcePicker, '%s', and the type of the value it already has, '%s'.", base_type, class_str));
		}
	}
}

String EditorResourcePicker::get_base_type() const {
	return base_type;
}

Vector<String> EditorResourcePicker::get_allowed_types() const {
	HashSet<StringName> allowed_types;
	_get_allowed_types(false, &allowed_types);

	Vector<String> types;
	types.resize(allowed_types.size());

	int i = 0;
	String *w = types.ptrw();
	for (const StringName &E : allowed_types) {
		w[i] = E;
		i++;
	}

	return types;
}

void EditorResourcePicker::set_edited_resource(Ref<Resource> p_resource) {
	if (!p_resource.is_valid()) {
		edited_resource = Ref<Resource>();
		_update_resource();
		return;
	}

	if (!base_type.is_empty()) {
		HashSet<StringName> allowed_types;
		_get_allowed_types(true, &allowed_types);

		StringName custom_class;
		bool is_custom = false;
		if (p_resource->get_script()) {
			custom_class = EditorNode::get_singleton()->get_object_custom_type_name(p_resource->get_script());
			is_custom = _is_type_valid(custom_class, allowed_types);
		}

		if (!is_custom && !_is_type_valid(p_resource->get_class(), allowed_types)) {
			String class_str = (custom_class == StringName() ? p_resource->get_class() : vformat("%s (%s)", custom_class, p_resource->get_class()));
			ERR_FAIL_MSG(vformat("Failed to set a resource of the type '%s' because this EditorResourcePicker only accepts '%s' and its derivatives.", class_str, base_type));
		}
	}

	edited_resource = p_resource;
	_update_resource();
}

Ref<Resource> EditorResourcePicker::get_edited_resource() {
	return edited_resource;
}

void EditorResourcePicker::set_toggle_mode(bool p_enable) {
	assign_button->set_toggle_mode(p_enable);
}

bool EditorResourcePicker::is_toggle_mode() const {
	return assign_button->is_toggle_mode();
}

void EditorResourcePicker::set_toggle_pressed(bool p_pressed) {
	if (!is_toggle_mode()) {
		return;
	}

	assign_button->set_pressed(p_pressed);
}

void EditorResourcePicker::set_editable(bool p_editable) {
	editable = p_editable;
	assign_button->set_disabled(!editable && !edited_resource.is_valid());
	edit_button->set_visible(editable);
}

bool EditorResourcePicker::is_editable() const {
	return editable;
}

void EditorResourcePicker::_ensure_resource_menu() {
	if (edit_menu) {
		return;
	}
	edit_menu = memnew(PopupMenu);
	add_child(edit_menu);
	edit_menu->connect("id_pressed", callable_mp(this, &EditorResourcePicker::_edit_menu_cbk));
	edit_menu->connect("popup_hide", callable_mp((BaseButton *)edit_button, &BaseButton::set_pressed).bind(false));
}

void EditorResourcePicker::_gather_resources_to_duplicate(const Ref<Resource> p_resource, TreeItem *p_item, const String &p_property_name) const {
	p_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);

	String res_name = p_resource->get_name();
	if (res_name.is_empty() && !p_resource->is_built_in()) {
		res_name = p_resource->get_path().get_file();
	}

	if (res_name.is_empty()) {
		p_item->set_text(0, p_resource->get_class());
	} else {
		p_item->set_text(0, vformat("%s (%s)", p_resource->get_class(), res_name));
	}

	p_item->set_icon(0, EditorNode::get_singleton()->get_object_icon(p_resource.ptr()));
	p_item->set_editable(0, true);

	Array meta;
	meta.append(p_resource);
	p_item->set_metadata(0, meta);

	if (!p_property_name.is_empty()) {
		p_item->set_text(1, p_property_name);
	}

	static Vector<String> unique_exceptions = { "Image", "Shader", "Mesh", "FontFile" };
	if (!unique_exceptions.has(p_resource->get_class())) {
		// Automatically select resource, unless it's something that shouldn't be duplicated.
		p_item->set_checked(0, true);
	}

	List<PropertyInfo> plist;
	p_resource->get_property_list(&plist);

	for (const PropertyInfo &E : plist) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE) || E.type != Variant::OBJECT || E.hint != PROPERTY_HINT_RESOURCE_TYPE) {
			continue;
		}

		Ref<Resource> res = p_resource->get(E.name);
		if (res.is_null()) {
			continue;
		}

		TreeItem *child = p_item->create_child();
		_gather_resources_to_duplicate(res, child, E.name);

		meta = child->get_metadata(0);
		// Remember property name.
		meta.append(E.name);

		if ((E.usage & PROPERTY_USAGE_NEVER_DUPLICATE)) {
			// The resource can't be duplicated, but make it appear on the list anyway.
			child->set_checked(0, false);
			child->set_editable(0, false);
		}
	}
}

void EditorResourcePicker::_duplicate_selected_resources() {
	for (TreeItem *item = duplicate_resources_tree->get_root(); item; item = item->get_next_in_tree()) {
		if (!item->is_checked(0)) {
			continue;
		}

		Array meta = item->get_metadata(0);
		Ref<Resource> res = meta[0];
		Ref<Resource> unique_resource = res->duplicate();
		ERR_FAIL_COND(unique_resource.is_null()); // duplicate() may fail.
		meta[0] = unique_resource;

		if (meta.size() == 1) { // Root.
			edited_resource = unique_resource;
			emit_signal(SNAME("resource_changed"), edited_resource);
			_update_resource();
		} else {
			Array parent_meta = item->get_parent()->get_metadata(0);
			Ref<Resource> parent = parent_meta[0];
			parent->set(meta[1], unique_resource);
		}
	}
}

EditorResourcePicker::EditorResourcePicker(bool p_hide_assign_button_controls) {
	assign_button = memnew(Button);
	assign_button->set_flat(true);
	assign_button->set_h_size_flags(SIZE_EXPAND_FILL);
	assign_button->set_expand_icon(true);
	assign_button->set_clip_text(true);
	assign_button->set_auto_translate(false);
	SET_DRAG_FORWARDING_GCD(assign_button, EditorResourcePicker);
	add_child(assign_button);
	assign_button->connect("pressed", callable_mp(this, &EditorResourcePicker::_resource_selected));
	assign_button->connect("draw", callable_mp(this, &EditorResourcePicker::_button_draw));
	assign_button->connect("gui_input", callable_mp(this, &EditorResourcePicker::_button_input));

	if (!p_hide_assign_button_controls) {
		preview_rect = memnew(TextureRect);
		preview_rect->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
		preview_rect->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
		preview_rect->set_offset(SIDE_TOP, 1);
		preview_rect->set_offset(SIDE_BOTTOM, -1);
		preview_rect->set_offset(SIDE_RIGHT, -1);
		preview_rect->set_texture_filter(TEXTURE_FILTER_NEAREST_WITH_MIPMAPS);
		assign_button->add_child(preview_rect);
	}

	edit_button = memnew(Button);
	edit_button->set_flat(true);
	edit_button->set_toggle_mode(true);
	edit_button->connect("pressed", callable_mp(this, &EditorResourcePicker::_update_menu));
	add_child(edit_button);
	edit_button->connect("gui_input", callable_mp(this, &EditorResourcePicker::_button_input));

	add_theme_constant_override("separation", 0);
}

// EditorScriptPicker

void EditorScriptPicker::set_create_options(Object *p_menu_node) {
	PopupMenu *menu_node = Object::cast_to<PopupMenu>(p_menu_node);
	if (!menu_node) {
		return;
	}

	menu_node->add_icon_item(get_editor_theme_icon(SNAME("ScriptCreate")), TTR("New Script"), OBJ_MENU_NEW_SCRIPT);
	if (script_owner) {
		Ref<Script> scr = script_owner->get_script();
		if (scr.is_valid()) {
			menu_node->add_icon_item(get_editor_theme_icon(SNAME("ScriptExtend")), TTR("Extend Script"), OBJ_MENU_EXTEND_SCRIPT);
		}
	}
	menu_node->add_separator();
}

bool EditorScriptPicker::handle_menu_selected(int p_which) {
	switch (p_which) {
		case OBJ_MENU_NEW_SCRIPT: {
			if (script_owner) {
				SceneTreeDock::get_singleton()->open_script_dialog(script_owner, false);
			}
			return true;
		}

		case OBJ_MENU_EXTEND_SCRIPT: {
			if (script_owner) {
				SceneTreeDock::get_singleton()->open_script_dialog(script_owner, true);
			}
			return true;
		}
	}

	return false;
}

void EditorScriptPicker::set_script_owner(Node *p_owner) {
	script_owner = p_owner;
}

Node *EditorScriptPicker::get_script_owner() const {
	return script_owner;
}

void EditorScriptPicker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_script_owner", "owner_node"), &EditorScriptPicker::set_script_owner);
	ClassDB::bind_method(D_METHOD("get_script_owner"), &EditorScriptPicker::get_script_owner);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "script_owner", PROPERTY_HINT_RESOURCE_TYPE, "Node", PROPERTY_USAGE_NONE), "set_script_owner", "get_script_owner");
}

EditorScriptPicker::EditorScriptPicker() {
}

// EditorShaderPicker

void EditorShaderPicker::set_create_options(Object *p_menu_node) {
	PopupMenu *menu_node = Object::cast_to<PopupMenu>(p_menu_node);
	if (!menu_node) {
		return;
	}

	menu_node->add_icon_item(get_editor_theme_icon(SNAME("Shader")), TTR("New Shader"), OBJ_MENU_NEW_SHADER);
	menu_node->add_separator();
}

bool EditorShaderPicker::handle_menu_selected(int p_which) {
	Ref<ShaderMaterial> ed_material = Ref<ShaderMaterial>(get_edited_material());

	switch (p_which) {
		case OBJ_MENU_NEW_SHADER: {
			if (ed_material.is_valid()) {
				SceneTreeDock::get_singleton()->open_shader_dialog(ed_material, preferred_mode);
				return true;
			}
		} break;
		default:
			break;
	}
	return false;
}

void EditorShaderPicker::set_edited_material(ShaderMaterial *p_material) {
	edited_material = p_material;
}

ShaderMaterial *EditorShaderPicker::get_edited_material() const {
	return edited_material;
}

void EditorShaderPicker::set_preferred_mode(int p_mode) {
	preferred_mode = p_mode;
}

EditorShaderPicker::EditorShaderPicker() {
}

//////////////

void EditorAudioStreamPicker::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
		case NOTIFICATION_THEME_CHANGED: {
			_update_resource();
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			Ref<AudioStream> audio_stream = get_edited_resource();
			if (audio_stream.is_valid()) {
				if (audio_stream->get_length() > 0) {
					Ref<AudioStreamPreview> preview = AudioStreamPreviewGenerator::get_singleton()->generate_preview(audio_stream);
					if (preview.is_valid()) {
						if (preview->get_version() != last_preview_version) {
							stream_preview_rect->queue_redraw();
							last_preview_version = preview->get_version();
						}
					}
				}

				uint64_t tagged_frame = audio_stream->get_tagged_frame();
				uint64_t diff_frames = AudioServer::get_singleton()->get_mixed_frames() - tagged_frame;
				uint64_t diff_msec = diff_frames * 1000 / AudioServer::get_singleton()->get_mix_rate();

				if (diff_msec < 300) {
					uint32_t count = audio_stream->get_tagged_frame_count();

					bool differ = false;

					if (count != tagged_frame_offset_count) {
						differ = true;
					}
					float offsets[MAX_TAGGED_FRAMES];

					for (uint32_t i = 0; i < MIN(count, uint32_t(MAX_TAGGED_FRAMES)); i++) {
						offsets[i] = audio_stream->get_tagged_frame_offset(i);
						if (offsets[i] != tagged_frame_offsets[i]) {
							differ = true;
						}
					}

					if (differ) {
						tagged_frame_offset_count = count;
						for (uint32_t i = 0; i < count; i++) {
							tagged_frame_offsets[i] = offsets[i];
						}
					}

					stream_preview_rect->queue_redraw();
				} else {
					if (tagged_frame_offset_count != 0) {
						stream_preview_rect->queue_redraw();
					}
					tagged_frame_offset_count = 0;
				}
			}
		} break;
	}
}

void EditorAudioStreamPicker::_update_resource() {
	EditorResourcePicker::_update_resource();

	Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
	int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Label"));
	Ref<AudioStream> audio_stream = get_edited_resource();
	if (audio_stream.is_valid() && audio_stream->get_length() > 0.0) {
		set_assign_button_min_size(Size2(1, font->get_height(font_size) * 3));
	} else {
		set_assign_button_min_size(Size2(1, font->get_height(font_size) * 1.5));
	}

	stream_preview_rect->queue_redraw();
}

void EditorAudioStreamPicker::_preview_draw() {
	Ref<AudioStream> audio_stream = get_edited_resource();
	if (!audio_stream.is_valid()) {
		get_assign_button()->set_text(TTR("<empty>"));
		return;
	}

	int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Label"));

	get_assign_button()->set_text("");

	Size2i size = stream_preview_rect->get_size();
	Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));

	Rect2 rect(Point2(), size);

	if (audio_stream->get_length() > 0 && size.width > 0) {
		rect.size.height *= 0.5;

		stream_preview_rect->draw_rect(rect, Color(0, 0, 0, 1));

		Ref<AudioStreamPreview> preview = AudioStreamPreviewGenerator::get_singleton()->generate_preview(audio_stream);
		float preview_len = preview->get_length();

		Vector<Vector2> points;
		points.resize(size.width * 2);

		for (int i = 0; i < size.width; i++) {
			float ofs = i * preview_len / size.width;
			float ofs_n = (i + 1) * preview_len / size.width;
			float max = preview->get_max(ofs, ofs_n) * 0.5 + 0.5;
			float min = preview->get_min(ofs, ofs_n) * 0.5 + 0.5;

			int idx = i;
			points.write[idx * 2 + 0] = Vector2(i + 1, rect.position.y + min * rect.size.y);
			points.write[idx * 2 + 1] = Vector2(i + 1, rect.position.y + max * rect.size.y);
		}

		Vector<Color> colors = { get_theme_color(SNAME("contrast_color_2"), EditorStringName(Editor)) };

		RS::get_singleton()->canvas_item_add_multiline(stream_preview_rect->get_canvas_item(), points, colors);

		if (tagged_frame_offset_count) {
			Color accent = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));

			for (uint32_t i = 0; i < tagged_frame_offset_count; i++) {
				int x = CLAMP(tagged_frame_offsets[i] * size.width / preview_len, 0, size.width);
				if (x == 0) {
					continue; // Because some may always return 0, ignore offset 0.
				}
				stream_preview_rect->draw_rect(Rect2i(x, 0, 2, rect.size.height), accent);
			}
		}
		rect.position.y += rect.size.height;
	}

	Ref<Texture2D> icon;
	Color icon_modulate(1, 1, 1, 1);

	if (tagged_frame_offset_count > 0) {
		icon = get_editor_theme_icon(SNAME("Play"));
		if ((OS::get_singleton()->get_ticks_msec() % 500) > 250) {
			icon_modulate = Color(1, 0.5, 0.5, 1); // get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
		}
	} else {
		icon = EditorNode::get_singleton()->get_object_icon(audio_stream.operator->(), "Object");
	}
	String text;
	if (!audio_stream->get_name().is_empty()) {
		text = audio_stream->get_name();
	} else if (audio_stream->get_path().is_resource_file()) {
		text = audio_stream->get_path().get_file();
	} else {
		text = audio_stream->get_class().replace_first("AudioStream", "");
	}

	stream_preview_rect->draw_texture(icon, Point2i(EDSCALE * 2, rect.position.y + (rect.size.height - icon->get_height()) / 2), icon_modulate);
	stream_preview_rect->draw_string(font, Point2i(EDSCALE * 2 + icon->get_width(), rect.position.y + font->get_ascent(font_size) + (rect.size.height - font->get_height(font_size)) / 2), text, HORIZONTAL_ALIGNMENT_CENTER, size.width - 4 * EDSCALE - icon->get_width());
}

EditorAudioStreamPicker::EditorAudioStreamPicker() :
		EditorResourcePicker(true) {
	stream_preview_rect = memnew(Control);

	stream_preview_rect->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	stream_preview_rect->set_offset(SIDE_TOP, 1);
	stream_preview_rect->set_offset(SIDE_BOTTOM, -1);
	stream_preview_rect->set_offset(SIDE_RIGHT, -1);
	stream_preview_rect->set_mouse_filter(MOUSE_FILTER_IGNORE);
	stream_preview_rect->connect("draw", callable_mp(this, &EditorAudioStreamPicker::_preview_draw));

	get_assign_button()->add_child(stream_preview_rect);
	get_assign_button()->move_child(stream_preview_rect, 0);
	set_process_internal(true);
}

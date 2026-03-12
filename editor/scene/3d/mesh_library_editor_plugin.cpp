/**************************************************************************/
/*  mesh_library_editor_plugin.cpp                                        */
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

#include "mesh_library_editor_plugin.h"

#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/docks/inspector_dock.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/gui/editor_zoom_widget.h"
#include "editor/gui/filter_line_edit.h"
#include "editor/inspector/editor_inspector.h"
#include "editor/inspector/editor_resource_preview.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/navigation/navigation_region_3d.h"
#include "scene/3d/physics/static_body_3d.h"
#include "scene/gui/button.h"
#include "scene/gui/item_list.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/main/timer.h"
#include "scene/resources/3d/mesh_library.h"
#include "scene/resources/packed_scene.h"

bool MeshLibraryEditor::MeshLibraryItem::_set(const StringName &p_name, const Variant &p_value) {
	ERR_FAIL_COND_V(mesh_library.is_null(), false);
	ERR_FAIL_COND_V(mesh_id == -1, false);
	ERR_FAIL_COND_V(!mesh_library->has_item(mesh_id), false);

	if (p_name == "name") {
		mesh_library->set_item_name(mesh_id, p_value);
	} else if (p_name == "mesh") {
		mesh_library->set_item_mesh(mesh_id, p_value);
	} else if (p_name == "mesh_transform") {
		mesh_library->set_item_mesh_transform(mesh_id, p_value);
	} else if (p_name == "mesh_cast_shadow") {
		switch ((int)p_value) {
			case 0: {
				mesh_library->set_item_mesh_cast_shadow(mesh_id, RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_OFF);
			} break;
			case 1: {
				mesh_library->set_item_mesh_cast_shadow(mesh_id, RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_ON);
			} break;
			case 2: {
				mesh_library->set_item_mesh_cast_shadow(mesh_id, RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_DOUBLE_SIDED);
			} break;
			case 3: {
				mesh_library->set_item_mesh_cast_shadow(mesh_id, RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
			} break;
			default: {
				mesh_library->set_item_mesh_cast_shadow(mesh_id, RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_ON);
			} break;
		}
#ifndef PHYSICS_3D_DISABLED
	} else if (p_name == "shapes") {
		mesh_library->call("set_item_shapes", mesh_id, p_value);
#endif // PHYSICS_3D_DISABLED
	} else if (p_name == "preview") {
		mesh_library->set_item_preview(mesh_id, p_value);
	} else if (p_name == "navigation_mesh") {
		mesh_library->set_item_navigation_mesh(mesh_id, p_value);
	} else if (p_name == "navigation_mesh_transform") {
		mesh_library->set_item_navigation_mesh_transform(mesh_id, p_value);
	} else if (p_name == "navigation_layers") {
		mesh_library->set_item_navigation_layers(mesh_id, p_value);
	} else {
		return false;
	}

	return false;
}

bool MeshLibraryEditor::MeshLibraryItem::_get(const StringName &p_name, Variant &r_ret) const {
	ERR_FAIL_COND_V(mesh_library.is_null(), false);
	ERR_FAIL_COND_V(mesh_id == -1, false);
	ERR_FAIL_COND_V(!mesh_library->has_item(mesh_id), false);

	if (p_name == "name") {
		r_ret = mesh_library->get_item_name(mesh_id);
	} else if (p_name == "mesh") {
		r_ret = mesh_library->get_item_mesh(mesh_id);
	} else if (p_name == "mesh_transform") {
		r_ret = mesh_library->get_item_mesh_transform(mesh_id);
	} else if (p_name == "mesh_cast_shadow") {
		r_ret = (int)mesh_library->get_item_mesh_cast_shadow(mesh_id);
#ifndef PHYSICS_3D_DISABLED
	} else if (p_name == "shapes") {
		r_ret = mesh_library->call("get_item_shapes", mesh_id);
#endif // PHYSICS_3D_DISABLED
	} else if (p_name == "navigation_mesh") {
		r_ret = mesh_library->get_item_navigation_mesh(mesh_id);
	} else if (p_name == "navigation_mesh_transform") {
		r_ret = mesh_library->get_item_navigation_mesh_transform(mesh_id);
	} else if (p_name == "navigation_layers") {
		r_ret = mesh_library->get_item_navigation_layers(mesh_id);
	} else if (p_name == "preview") {
		r_ret = mesh_library->get_item_preview(mesh_id);
	} else {
		return false;
	}

	return true;
}

void MeshLibraryEditor::MeshLibraryItem::_get_property_list(List<PropertyInfo> *p_list) const {
	if (mesh_id == -1 || mesh_library.is_null() || !mesh_library->has_item(mesh_id)) {
		return;
	}

	p_list->push_back(PropertyInfo(Variant::STRING, PNAME("name")));
	p_list->push_back(PropertyInfo(Variant::OBJECT, PNAME("mesh"), PROPERTY_HINT_RESOURCE_TYPE, Mesh::get_class_static()));
	p_list->push_back(PropertyInfo(Variant::TRANSFORM3D, PNAME("mesh_transform"), PROPERTY_HINT_NONE, "suffix:m"));
	p_list->push_back(PropertyInfo(Variant::INT, PNAME("mesh_cast_shadow"), PROPERTY_HINT_ENUM, "Off,On,Double-Sided,Shadows Only"));
	p_list->push_back(PropertyInfo(Variant::ARRAY, PNAME("shapes")));
	p_list->push_back(PropertyInfo(Variant::OBJECT, PNAME("navigation_mesh"), PROPERTY_HINT_RESOURCE_TYPE, NavigationMesh::get_class_static()));
	p_list->push_back(PropertyInfo(Variant::TRANSFORM3D, PNAME("navigation_mesh_transform"), PROPERTY_HINT_NONE, "suffix:m"));
	p_list->push_back(PropertyInfo(Variant::INT, PNAME("navigation_layers"), PROPERTY_HINT_LAYERS_3D_NAVIGATION));
	p_list->push_back(PropertyInfo(Variant::OBJECT, PNAME("preview"), PROPERTY_HINT_RESOURCE_TYPE, Texture2D::get_class_static(), PROPERTY_USAGE_DEFAULT));
}

////////////////

void MeshLibraryEditor::edit(const Ref<MeshLibrary> &p_mesh_library) {
	if (mesh_library.is_valid()) {
		mesh_library->disconnect_changed(callable_mp(update_items_delay, &Timer::start));
	}

	selected_item = -1;

	mesh_library = p_mesh_library;
	if (mesh_library.is_valid()) {
		// Avoid updating multiple times at once.
		mesh_library->connect_changed(callable_mp(update_items_delay, &Timer::start).bind(UPDATE_ITEMS_DELAY_TIMEOUT));
		_update_mesh_items();
	}
}

Error MeshLibraryEditor::update_library_file(Node *p_base_scene, Ref<MeshLibrary> ml, bool p_merge, bool p_apply_xforms) {
	_import_scene(p_base_scene, ml, p_merge, p_apply_xforms);
	return OK;
}

void MeshLibraryEditor::_update_mesh_items(bool p_reselect, Ref<MeshLibrary> p_lib_check) {
	if (p_lib_check.is_valid() && mesh_library != p_lib_check) {
		return;
	}

	mesh_library->set_edited(true);
	mesh_items->clear();
	remove_item->set_disabled(true);
	update_items_delay->stop();

	if (mesh_library->get_item_count() == 0) {
		_select_item(-1);

		item_split->hide();
		empty_lib->show();

		return;
	}

	empty_lib->hide();
	item_split->show();

	float min_size = EDITOR_GET("editors/grid_map/preview_size");
	min_size *= EDSCALE;
	mesh_items->set_fixed_column_width(min_size * MAX(zoom_widget->get_zoom(), 1.5));
	mesh_items->set_fixed_icon_size(Size2(min_size, min_size));

	Vector<int> ids;
	ids = mesh_library->get_item_list();

	struct _CGMEItemSort {
		String name;
		int id = 0;
		_FORCE_INLINE_ bool operator<(const _CGMEItemSort &r_it) const { return name < r_it.name; }
	};

	List<_CGMEItemSort> il;
	for (const int &E : ids) {
		_CGMEItemSort is;
		is.id = E;
		is.name = mesh_library->get_item_name(E);
		il.push_back(is);
	}
	il.sort();

	int item = 0;
	bool selection_found = false;
	String filter = search_box->get_text().strip_edges();
	for (_CGMEItemSort &E : il) {
		int id = E.id;
		String name = mesh_library->get_item_name(id);

		if (!filter.is_empty() && !filter.is_subsequence_ofn(name)) {
			if (selected_item == id) {
				selection_found = true;
			}

			continue;
		}

		if (name.is_empty()) {
			name = "#" + itos(id);
		}

		mesh_items->add_item("");

		Ref<Texture2D> preview = mesh_library->get_item_preview(id);
		if (preview.is_valid()) {
			mesh_items->set_item_icon(item, preview);
			mesh_items->set_item_tooltip(item, name);
		} else {
			Ref<Mesh> mesh = mesh_library->get_item_mesh(id);
			if (mesh.is_valid()) {
				// Fallback to the item's mesh preview.
				EditorResourcePreview::get_singleton()->queue_edited_resource_preview(mesh, callable_mp(this, &MeshLibraryEditor::_update_resource_preview).bind(item));
			}
		}

		mesh_items->set_item_text(item, name);
		mesh_items->set_item_metadata(item, id);

		if (selected_item == id) {
			selection_found = true;

			if (p_reselect) {
				_select_item(id);
				mesh_items->select(item);
				remove_item->set_disabled(false);
			}
		}

		item++;
	}

	if (!selection_found) {
		if (p_reselect && item > 0) {
			_select_item(mesh_items->get_item_metadata(0));
			mesh_items->select(0);
			remove_item->set_disabled(false);
		} else {
			_select_item(-1);
		}
	}
}

void MeshLibraryEditor::_update_resource_preview(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, int p_idx) {
	if (p_idx < mesh_items->get_item_count()) {
		mesh_items->set_item_icon(p_idx, p_preview);
	}
}

void MeshLibraryEditor::_select_item(int p_id, Ref<MeshLibrary> p_lib_check) {
	if (p_lib_check.is_valid() && mesh_library != p_lib_check) {
		return;
	}

	selected_item = p_id;
	if (selected_item == -1) {
		inspector->edit(nullptr);
		return;
	}

	Object *edited = inspector->get_edited_object();
	if (edited) {
		Ref<MeshLibraryItem> item(edited);
		if (item->mesh_id == selected_item) {
			return; // Already inspecting it.
		}
	}

	mesh_library_item.instantiate();
	mesh_library_item->mesh_library = mesh_library;
	mesh_library_item->mesh_id = selected_item;
	inspector->edit(*mesh_library_item);
}

void MeshLibraryEditor::_select_item_and_button(int p_id, Ref<MeshLibrary> p_lib_check) {
	if (p_lib_check.is_valid() && mesh_library != p_lib_check) {
		return;
	}

	_select_item(p_id);

	for (int i = 0; i < mesh_items->get_item_count(); i++) {
		int id = mesh_items->get_item_metadata(i);
		if (id == p_id) {
			mesh_items->select(i);
			remove_item->set_disabled(false);
			return;
		}
	}
}

void MeshLibraryEditor::_select_prev_item_and_button(int p_id, Ref<MeshLibrary> p_lib_check) {
	int idx = -1;
	for (int i = 0; i < mesh_items->get_item_count(); i++) {
		int id = mesh_items->get_item_metadata(i);
		if (id == p_id) {
			idx = MAX(i - 1, 0);
			break;
		}
	}

	if (idx == -1 && mesh_items->get_item_count() > 0) {
		idx = 0;
	}

	if (idx > -1) {
		int id = mesh_items->get_item_metadata(idx);
		_select_item(id);
		mesh_items->select(idx);
		remove_item->set_disabled(false);
	}
}

void MeshLibraryEditor::_mesh_items_cbk(int p_idx) {
	int id = mesh_items->get_item_metadata(p_idx);
	_select_item(id);
	remove_item->set_disabled(false);
}

void MeshLibraryEditor::_menu_cbk(int p_option) {
	switch (p_option) {
		case MENU_OPTION_ADD_ITEM: {
			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action(TTR("Add MeshLibrary item"));

			int to_create = mesh_library->get_last_unused_item_id();
			undo_redo->add_do_method(*mesh_library, "create_item", to_create);
			undo_redo->add_undo_method(*mesh_library, "remove_item", to_create);

			undo_redo->add_do_method(this, "_update_mesh_items", false, *mesh_library);
			undo_redo->add_undo_method(this, "_update_mesh_items", false, *mesh_library);

			undo_redo->add_do_method(this, "_select_item_and_button", to_create, *mesh_library);
			if (mesh_library->get_item_count() > 0) {
				undo_redo->add_undo_method(this, "_select_prev_item_and_button", to_create, *mesh_library);
			}

			undo_redo->commit_action();
		} break;

		case MENU_OPTION_REMOVE_ITEM: {
			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action(TTR("Remove MeshLibrary item"));

			if (mesh_library->get_item_count() - 1 > 0) {
				undo_redo->add_do_method(this, "_select_prev_item_and_button", selected_item, *mesh_library);
			}

			undo_redo->add_do_method(*mesh_library, "remove_item", selected_item);

			Ref<Mesh> mesh = mesh_library->get_item_mesh(selected_item);
			int mesh_shadow = mesh_library->get_item_mesh_cast_shadow(selected_item);
			Transform3D mesh_transform = mesh_library->get_item_mesh_transform(selected_item);
			String mesh_name = mesh_library->get_item_name(selected_item);
			int nav_layers = mesh_library->get_item_navigation_layers(selected_item);
			Ref<NavigationMesh> nav_mesh = mesh_library->get_item_navigation_mesh(selected_item);
			Transform3D nav_mesh_transform = mesh_library->get_item_navigation_mesh_transform(selected_item);
			Ref<Texture2D> preview = mesh_library->get_item_preview(selected_item);
			Array shapes = mesh_library->call("get_item_shapes", selected_item);

			undo_redo->add_undo_method(*mesh_library, "create_item", selected_item);
			undo_redo->add_undo_method(*mesh_library, "set_item_mesh", selected_item, mesh);
			undo_redo->add_undo_method(*mesh_library, "set_item_mesh_cast_shadow", selected_item, mesh_shadow);
			undo_redo->add_undo_method(*mesh_library, "set_item_mesh_transform", selected_item, mesh_transform);
			undo_redo->add_undo_method(*mesh_library, "set_item_name", selected_item, mesh_name);
			undo_redo->add_undo_method(*mesh_library, "set_item_navigation_layers", selected_item, nav_layers);
			undo_redo->add_undo_method(*mesh_library, "set_item_navigation_mesh", selected_item, nav_mesh);
			undo_redo->add_undo_method(*mesh_library, "set_item_navigation_mesh_transform", selected_item, nav_mesh_transform);
			undo_redo->add_undo_method(*mesh_library, "set_item_preview", selected_item, preview);
			undo_redo->add_undo_method(*mesh_library, "set_item_shapes", selected_item, shapes);

			undo_redo->add_do_method(this, "_update_mesh_items", true, *mesh_library);
			undo_redo->add_undo_method(this, "_update_mesh_items", false, *mesh_library);

			undo_redo->add_undo_method(this, "_select_item_and_button", selected_item, *mesh_library);

			undo_redo->commit_action();
		} break;

		case MENU_OPTION_IMPORT_FROM_SCENE: {
			apply_xforms = false;
			import_update = false;
			file->popup_file_dialog();
		} break;

		case MENU_OPTION_IMPORT_FROM_SCENE_APPLY_XFORMS: {
			apply_xforms = true;
			import_update = false;
			file->popup_file_dialog();
		} break;

		case MENU_OPTION_UPDATE_FROM_SCENE: {
			import_update = true;
			cd_update->set_text(vformat(TTR("Update from existing scene?:\n%s"), String(mesh_library->get_meta("_editor_source_scene"))));
			cd_update->popup_centered(Size2(500, 60));
		} break;
	}
}

void MeshLibraryEditor::_menu_update_confirm(bool p_apply_xforms) {
	cd_update->hide();
	apply_xforms = p_apply_xforms;
	String existing = mesh_library->get_meta("_editor_source_scene");
	ERR_FAIL_COND(existing.is_empty());
	_import_scene_cbk(existing);
}

void MeshLibraryEditor::_import_scene(Node *p_scene, Ref<MeshLibrary> p_library, bool p_merge, bool p_apply_xforms) {
	if (!p_merge) {
		p_library->clear();
	}

	HashMap<int, MeshInstance3D *> mesh_instances;
	for (int i = 0; i < p_scene->get_child_count(); i++) {
		_import_scene_parse_node(p_library, mesh_instances, p_scene->get_child(i), p_merge, p_apply_xforms);
	}

	// Generate previews.

	Vector<Ref<Mesh>> meshes;
	Vector<Transform3D> transforms;
	Vector<int> ids = p_library->get_item_list();
	for (const int &E : ids) {
		if (mesh_instances.find(E)) {
			meshes.push_back(p_library->get_item_mesh(E));
			transforms.push_back(mesh_instances[E]->get_transform());
		}
	}

	Vector<Ref<Texture2D>> textures = EditorInterface::get_singleton()->make_mesh_previews(meshes, &transforms, EDITOR_GET("editors/grid_map/preview_size"));
	int idx = 0;
	for (const int &E : ids) {
		if (mesh_instances.find(E)) {
			p_library->set_item_preview(E, textures[idx]);
			idx++;
		}
	}
}

void MeshLibraryEditor::_import_scene_cbk(const String &p_str) {
	Ref<PackedScene> ps = ResourceLoader::load(p_str, "PackedScene");
	ERR_FAIL_COND(ps.is_null());

	Node *scene = ps->instantiate();
	ERR_FAIL_NULL_MSG(scene, "Cannot create an instance from PackedScene '" + p_str + "'.");

	// Preserve the data from the current library.
	Ref<MeshLibrary> old_lib;
	old_lib = mesh_library->duplicate();

	_import_scene(scene, mesh_library, import_update, apply_xforms);

	memdelete(scene);
	mesh_library->set_meta("_editor_source_scene", p_str);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Import MeshLibrary from scene"));
	undo_redo->add_do_method(*mesh_library, "copy_from_resource", *mesh_library->duplicate());
	undo_redo->add_undo_method(*mesh_library, "copy_from_resource", old_lib);
	undo_redo->commit_action(false);

	import_scene->get_popup()->set_item_disabled(import_scene->get_popup()->get_item_index(MENU_OPTION_UPDATE_FROM_SCENE), false);
}

void MeshLibraryEditor::_import_scene_parse_node(Ref<MeshLibrary> p_library, HashMap<int, MeshInstance3D *> &p_mesh_instances, Node *p_node, bool p_merge, bool p_apply_xforms) {
	MeshInstance3D *mesh_instance_node = Object::cast_to<MeshInstance3D>(p_node);

	if (!mesh_instance_node) {
		// No MeshInstance so search deeper.
		for (int i = 0; i < p_node->get_child_count(); i++) {
			_import_scene_parse_node(p_library, p_mesh_instances, p_node->get_child(i), p_merge, p_apply_xforms);
		}
		return;
	}

	Ref<Mesh> source_mesh = mesh_instance_node->get_mesh();
	if (source_mesh.is_null()) {
		return;
	}

	int item_id = p_library->find_item_by_name(mesh_instance_node->get_name());
	if (item_id < 0) {
		item_id = p_library->get_last_unused_item_id();
		p_library->create_item(item_id);
		p_library->set_item_name(item_id, mesh_instance_node->get_name());
	} else if (!p_merge) {
		WARN_PRINT(vformat("MeshLibrary export found a MeshInstance3D with a duplicated name '%s' in the exported scene that overrides a previously parsed MeshInstance3D item with the same name.", mesh_instance_node->get_name()));
	}
	p_mesh_instances[item_id] = mesh_instance_node;

	Ref<Mesh> item_mesh = source_mesh->duplicate();
	for (int i = 0; i < item_mesh->get_surface_count(); i++) {
		Ref<Material> surface_override_material = mesh_instance_node->get_surface_override_material(i);
		if (surface_override_material.is_valid()) {
			item_mesh->surface_set_material(i, surface_override_material);
		}
	}
	p_library->set_item_mesh(item_id, item_mesh);

	GeometryInstance3D::ShadowCastingSetting gi3d_cast_shadows_setting = mesh_instance_node->get_cast_shadows_setting();
	switch (gi3d_cast_shadows_setting) {
		case GeometryInstance3D::ShadowCastingSetting::SHADOW_CASTING_SETTING_OFF: {
			p_library->set_item_mesh_cast_shadow(item_id, RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_OFF);
		} break;
		case GeometryInstance3D::ShadowCastingSetting::SHADOW_CASTING_SETTING_ON: {
			p_library->set_item_mesh_cast_shadow(item_id, RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_ON);
		} break;
		case GeometryInstance3D::ShadowCastingSetting::SHADOW_CASTING_SETTING_DOUBLE_SIDED: {
			p_library->set_item_mesh_cast_shadow(item_id, RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_DOUBLE_SIDED);
		} break;
		case GeometryInstance3D::ShadowCastingSetting::SHADOW_CASTING_SETTING_SHADOWS_ONLY: {
			p_library->set_item_mesh_cast_shadow(item_id, RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
		} break;
		default: {
			p_library->set_item_mesh_cast_shadow(item_id, RSE::ShadowCastingSetting::SHADOW_CASTING_SETTING_ON);
		} break;
	}

	Transform3D item_mesh_transform;
	if (p_apply_xforms) {
		item_mesh_transform = mesh_instance_node->get_transform();
	}
	p_library->set_item_mesh_transform(item_id, item_mesh_transform);

	Vector<MeshLibrary::ShapeData> collisions;
	for (int i = 0; i < mesh_instance_node->get_child_count(); i++) {
		StaticBody3D *static_body_node = Object::cast_to<StaticBody3D>(mesh_instance_node->get_child(i));
		if (!static_body_node) {
			continue;
		}
		List<uint32_t> shapes;
		static_body_node->get_shape_owners(&shapes);
		for (uint32_t &E : shapes) {
			if (static_body_node->is_shape_owner_disabled(E)) {
				continue;
			}
			Transform3D shape_transform;
			if (p_apply_xforms) {
				shape_transform = mesh_instance_node->get_transform();
			}
			shape_transform *= static_body_node->get_transform() * static_body_node->shape_owner_get_transform(E);
			for (int k = 0; k < static_body_node->shape_owner_get_shape_count(E); k++) {
				Ref<Shape3D> collision_shape = static_body_node->shape_owner_get_shape(E, k);
				if (collision_shape.is_null()) {
					continue;
				}
				MeshLibrary::ShapeData shape_data;
				shape_data.shape = collision_shape;
				shape_data.local_transform = shape_transform;
				collisions.push_back(shape_data);
			}
		}
	}
	p_library->set_item_shapes(item_id, collisions);

	for (int i = 0; i < mesh_instance_node->get_child_count(); i++) {
		NavigationRegion3D *navigation_region_node = Object::cast_to<NavigationRegion3D>(mesh_instance_node->get_child(i));
		if (!navigation_region_node) {
			continue;
		}
		Ref<NavigationMesh> navigation_mesh = navigation_region_node->get_navigation_mesh();
		if (navigation_mesh.is_valid()) {
			Transform3D navigation_mesh_transform = navigation_region_node->get_transform();
			p_library->set_item_navigation_mesh(item_id, navigation_mesh);
			p_library->set_item_navigation_mesh_transform(item_id, navigation_mesh_transform);
			break;
		}
	}
}

void MeshLibraryEditor::_icon_size_changed(float p_value) {
	mesh_items->set_icon_scale(p_value);
	_update_mesh_items();
}

void MeshLibraryEditor::_mesh_items_input(const Ref<InputEvent> &p_event) {
	// Zoom in/out using Ctrl + mouse wheel.
	const Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->is_command_or_control_pressed()) {
		if (mb->is_pressed() && mb->get_button_index() == MouseButton::WHEEL_UP) {
			zoom_widget->set_zoom(zoom_widget->get_zoom() + 0.2);
			zoom_widget->emit_signal(SNAME("zoom_changed"), zoom_widget->get_zoom());
			accept_event();
		}

		if (mb->is_pressed() && mb->get_button_index() == MouseButton::WHEEL_DOWN) {
			zoom_widget->set_zoom(zoom_widget->get_zoom() - 0.2);
			zoom_widget->emit_signal(SNAME("zoom_changed"), zoom_widget->get_zoom());
			accept_event();
		}
	}
}

void MeshLibraryEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_THEME_CHANGED) {
		add_item->set_button_icon(get_editor_theme_icon(SNAME("Add")));
		remove_item->set_button_icon(get_editor_theme_icon(SNAME("Remove")));
	}
}

void MeshLibraryEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_mesh_items", "reselect"), &MeshLibraryEditor::_update_mesh_items, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("_select_item_and_button", "id", "library"), &MeshLibraryEditor::_select_item_and_button, DEFVAL(Ref<MeshLibrary>()));
	ClassDB::bind_method(D_METHOD("_select_prev_item_and_button", "id", "library"), &MeshLibraryEditor::_select_prev_item_and_button, DEFVAL(Ref<MeshLibrary>()));
}

MeshLibraryEditor::MeshLibraryEditor() {
	set_name(TTRC("MeshLibrary"));
	set_icon_name("MeshLibraryEditor");
	set_default_slot(EditorDock::DOCK_SLOT_BOTTOM);
	set_available_layouts(EditorDock::DOCK_LAYOUT_HORIZONTAL | EditorDock::DOCK_LAYOUT_FLOATING);
	set_global(false);
	set_transient(true);
	set_custom_minimum_size(Size2(0, 200 * EDSCALE));

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	HBoxContainer *toolbar = memnew(HBoxContainer);
	vb->add_child(toolbar);

	add_item = memnew(Button);
	add_item->set_text(TTRC("Add"));
	add_item->set_theme_type_variation(SceneStringName(FlatButton));
	add_item->connect(SceneStringName(pressed), callable_mp(this, &MeshLibraryEditor::_menu_cbk).bind(MENU_OPTION_ADD_ITEM));
	toolbar->add_child(add_item);

	remove_item = memnew(Button);
	remove_item->set_text(TTRC("Remove"));
	remove_item->set_disabled(true);
	remove_item->set_theme_type_variation(SceneStringName(FlatButton));
	remove_item->connect(SceneStringName(pressed), callable_mp(this, &MeshLibraryEditor::_menu_cbk).bind(MENU_OPTION_REMOVE_ITEM));
	toolbar->add_child(remove_item);

	toolbar->add_child(memnew(VSeparator));

	import_scene = memnew(MenuButton);
	import_scene->set_text(TTRC("Import"));
	import_scene->set_flat(false);
	import_scene->set_theme_type_variation(SceneStringName(FlatButton));
	toolbar->add_child(import_scene);

	import_scene->get_popup()->add_item(TTRC("Import from Scene (Ignore Transforms)"), MENU_OPTION_IMPORT_FROM_SCENE);
	import_scene->get_popup()->add_item(TTRC("Import from Scene (Apply Transforms)"), MENU_OPTION_IMPORT_FROM_SCENE_APPLY_XFORMS);
	import_scene->get_popup()->add_separator();
	import_scene->get_popup()->add_item(TTRC("Update from Scene"), MENU_OPTION_UPDATE_FROM_SCENE);
	import_scene->get_popup()->set_item_disabled(-1, true);
	import_scene->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &MeshLibraryEditor::_menu_cbk));

	toolbar->add_spacer();

	update_items_delay = memnew(Timer);
	update_items_delay->set_wait_time(UPDATE_ITEMS_DELAY_TIMEOUT);
	update_items_delay->set_one_shot(true);
	update_items_delay->connect("timeout", callable_mp(this, &MeshLibraryEditor::_update_mesh_items).bind(true, Ref<MeshLibrary>()));
	add_child(update_items_delay);

	search_box = memnew(FilterLineEdit);
	search_box->set_placeholder(TTRC("Filter Meshes"));
	search_box->set_accessibility_name(TTRC("Filter Meshes"));
	search_box->add_theme_constant_override("minimum_character_width", 10);
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	toolbar->add_child(search_box);
	search_box->connect(SceneStringName(text_changed), callable_mp(update_items_delay, &Timer::start).bind(UPDATE_ITEMS_DELAY_TIMEOUT).unbind(1));

	zoom_widget = memnew(EditorZoomWidget);
	zoom_widget->setup_zoom_limits(0.2, 4);
	zoom_widget->connect("zoom_changed", callable_mp(this, &MeshLibraryEditor::_icon_size_changed));
	zoom_widget->set_shortcut_context(this);
	toolbar->add_child(zoom_widget);

	item_split = memnew(HSplitContainer);
	item_split->set_v_size_flags(SIZE_EXPAND_FILL);
	vb->add_child(item_split);
	item_split->hide();

	mesh_items = memnew(ItemList);
	mesh_items->set_max_columns(0);
	mesh_items->set_icon_mode(ItemList::ICON_MODE_TOP);
	mesh_items->set_max_text_lines(2);
	mesh_items->set_theme_type_variation("ItemListSecondary");
	mesh_items->set_h_size_flags(SIZE_EXPAND_FILL);
	mesh_items->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	mesh_items->set_auto_translate(false);
	item_split->add_child(mesh_items);
	mesh_items->connect(SceneStringName(item_selected), callable_mp(this, &MeshLibraryEditor::_mesh_items_cbk));
	mesh_items->connect(SceneStringName(gui_input), callable_mp(this, &MeshLibraryEditor::_mesh_items_input));

	search_box->set_forward_control(mesh_items);

	inspector = memnew(EditorInspector);
	inspector->set_use_doc_hints(true);
	inspector->set_theme_type_variation("ScrollContainerSecondary");
	inspector->set_h_size_flags(SIZE_EXPAND_FILL);
	inspector->set_stretch_ratio(0.3);
	inspector->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	item_split->add_child(inspector);

	inspector->add_custom_property_description("MeshLibraryItem", "name", TTRC("The item's name, shown in the editor. It can also be used to look up the item later using [method MeshLibrary.find_item_by_name]."));
	inspector->add_custom_property_description("MeshLibraryItem", "mesh", TTRC("The item's mesh. Used by other parts of the engine (e.g. [GridMap], which displays them in a 3D tile)."));
	inspector->add_custom_property_description("MeshLibraryItem", "mesh_transform", TTRC("The transform to apply to the item's mesh."));
	inspector->add_custom_property_description("MeshLibraryItem", "mesh_cast_shadow", TTRC("The shadow casting mode used by the item's mesh. See [enum RenderingServer.ShadowCastingSetting]"));
	inspector->add_custom_property_description("MeshLibraryItem", "shapes", TTRC("The item's collision shapes.\nThe array should consist of [Shape3D] objects, each followed by a [Transform3D] that will be applied to it. For shapes that should not have a transform, use [constant Transform3D.IDENTITY]."));
	inspector->add_custom_property_description("MeshLibraryItem", "navigation_mesh", TTRC("The item's navigation mesh."));
	inspector->add_custom_property_description("MeshLibraryItem", "navigation_mesh_transform", TTRC("The transform to apply to the item's navigation mesh."));
	inspector->add_custom_property_description("MeshLibraryItem", "navigation_layers", TTRC("The item's navigation layers bitmask."));
	inspector->add_custom_property_description("MeshLibraryItem", "preview", TTRC("The texture to use as the item's preview icon in the editor."));

	empty_lib = memnew(Label);
	empty_lib->set_text(TTRC("No items found inside the MeshLibrary.\nYou can add some by using the Add button on the left, or by exporting them from a scene file via the Export menu."));
	empty_lib->set_focus_mode(FOCUS_ACCESSIBILITY);
	empty_lib->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	empty_lib->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	empty_lib->set_v_size_flags(SIZE_EXPAND_FILL);
	empty_lib->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	empty_lib->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	vb->add_child(empty_lib);
	empty_lib->hide();

	file = memnew(EditorFileDialog);
	file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);
	file->clear_filters();
	file->set_title(TTRC("Import Scene"));
	for (const String &extension : extensions) {
		file->add_filter("*." + extension, extension.to_upper());
	}
	add_child(file);
	file->connect("file_selected", callable_mp(this, &MeshLibraryEditor::_import_scene_cbk));

	cd_update = memnew(ConfirmationDialog);
	add_child(cd_update);
	cd_update->set_ok_button_text(TTRC("Apply without Transforms"));
	cd_update->get_ok_button()->connect(SceneStringName(pressed), callable_mp(this, &MeshLibraryEditor::_menu_update_confirm).bind(false));
	cd_update->add_button(TTRC("Apply with Transforms"))->connect(SceneStringName(pressed), callable_mp(this, &MeshLibraryEditor::_menu_update_confirm).bind(true));
}

////////////////

void MeshLibraryEditorPlugin::edit(Object *p_node) {
	Ref<MeshLibrary> ml = Object::cast_to<MeshLibrary>(p_node);
	if (ml.is_valid()) {
		mesh_library_editor->edit(ml);
	}
}

bool MeshLibraryEditorPlugin::handles(Object *p_object) const {
	return p_object && p_object->is_class("MeshLibrary");
}

void MeshLibraryEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		mesh_library_editor->make_visible();
	} else {
		mesh_library_editor->close();
	}
}

MeshLibraryEditorPlugin::MeshLibraryEditorPlugin() {
	singleton = this;

	mesh_library_editor = memnew(MeshLibraryEditor);
	EditorDockManager::get_singleton()->add_dock(mesh_library_editor);
	mesh_library_editor->close();
}

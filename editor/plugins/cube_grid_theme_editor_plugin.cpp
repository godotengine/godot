/*************************************************************************/
/*  cube_grid_theme_editor_plugin.cpp                                    */
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
#include "cube_grid_theme_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "main/main.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/navigation_mesh.h"
#include "scene/3d/physics_body.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"

void MeshLibraryEditor::edit(const Ref<MeshLibrary> &p_theme) {

	theme = p_theme;
	if (theme.is_valid())
		menu->get_popup()->set_item_disabled(menu->get_popup()->get_item_index(MENU_OPTION_UPDATE_FROM_SCENE), !theme->has_meta("_editor_source_scene"));
}

void MeshLibraryEditor::_menu_confirm() {

	switch (option) {

		case MENU_OPTION_REMOVE_ITEM: {

			theme->remove_item(to_erase);
		} break;
		case MENU_OPTION_UPDATE_FROM_SCENE: {
			String existing = theme->get_meta("_editor_source_scene");
			ERR_FAIL_COND(existing == "");
			_import_scene_cbk(existing);

		} break;
		default: {};
	}
}

void MeshLibraryEditor::_import_scene(Node *p_scene, Ref<MeshLibrary> p_library, bool p_merge) {

	if (!p_merge)
		p_library->clear();

	for (int i = 0; i < p_scene->get_child_count(); i++) {

		Node *child = p_scene->get_child(i);

		if (!Object::cast_to<MeshInstance>(child)) {
			if (child->get_child_count() > 0) {
				child = child->get_child(0);
				if (!Object::cast_to<MeshInstance>(child)) {
					continue;
				}

			} else
				continue;
		}

		MeshInstance *mi = Object::cast_to<MeshInstance>(child);
		Ref<Mesh> mesh = mi->get_mesh();
		if (mesh.is_null())
			continue;

		int id = p_library->find_item_by_name(mi->get_name());
		if (id < 0) {

			id = p_library->get_last_unused_item_id();
			p_library->create_item(id);
			p_library->set_item_name(id, mi->get_name());
		}

		p_library->set_item_mesh(id, mesh);

		Vector<MeshLibrary::ShapeData> collisions;

		for (int j = 0; j < mi->get_child_count(); j++) {

			Node *child2 = mi->get_child(j);
			if (!Object::cast_to<StaticBody>(child2))
				continue;

			StaticBody *sb = Object::cast_to<StaticBody>(child2);
			List<uint32_t> shapes;
			sb->get_shape_owners(&shapes);

			for (List<uint32_t>::Element *E = shapes.front(); E; E = E->next()) {
				if (sb->is_shape_owner_disabled(E->get()))
					continue;

				//Transform shape_transform = sb->shape_owner_get_transform(E->get());

				//shape_transform.set_origin(shape_transform.get_origin() - phys_offset);

				for (int k = 0; k < sb->shape_owner_get_shape_count(E->get()); k++) {

					Ref<Shape> collision = sb->shape_owner_get_shape(E->get(), k);
					if (!collision.is_valid())
						continue;
					MeshLibrary::ShapeData shape_data;
					shape_data.shape = collision;
					shape_data.local_transform = sb->shape_owner_get_transform(E->get());
					collisions.push_back(shape_data);
				}
			}
		}

		p_library->set_item_shapes(id, collisions);

		Ref<NavigationMesh> navmesh;
		for (int j = 0; j < mi->get_child_count(); j++) {
			Node *child2 = mi->get_child(j);
			if (!Object::cast_to<NavigationMeshInstance>(child2))
				continue;
			NavigationMeshInstance *sb = Object::cast_to<NavigationMeshInstance>(child2);
			navmesh = sb->get_navigation_mesh();
			if (!navmesh.is_null())
				break;
		}
		if (!navmesh.is_null()) {
			p_library->set_item_navmesh(id, navmesh);
		}
	}

	//generate previews!

	if (1) {

		Vector<Ref<Mesh> > meshes;
		Vector<int> ids = p_library->get_item_list();
		for (int i = 0; i < ids.size(); i++) {
			meshes.push_back(p_library->get_item_mesh(ids[i]));
		}

		Vector<Ref<Texture> > textures = EditorInterface::get_singleton()->make_mesh_previews(meshes, EditorSettings::get_singleton()->get("editors/grid_map/preview_size"));
		for (int i = 0; i < ids.size(); i++) {
			p_library->set_item_preview(ids[i], textures[i]);
		}
	}
}

void MeshLibraryEditor::_import_scene_cbk(const String &p_str) {

	print_line("Impot Callback!");

	Ref<PackedScene> ps = ResourceLoader::load(p_str, "PackedScene");
	ERR_FAIL_COND(ps.is_null());
	Node *scene = ps->instance();

	_import_scene(scene, theme, option == MENU_OPTION_UPDATE_FROM_SCENE);

	memdelete(scene);
	theme->set_meta("_editor_source_scene", p_str);
	menu->get_popup()->set_item_disabled(menu->get_popup()->get_item_index(MENU_OPTION_UPDATE_FROM_SCENE), false);
}

Error MeshLibraryEditor::update_library_file(Node *p_base_scene, Ref<MeshLibrary> ml, bool p_merge) {

	_import_scene(p_base_scene, ml, p_merge);
	return OK;
}

void MeshLibraryEditor::_menu_cbk(int p_option) {

	option = p_option;
	switch (p_option) {

		case MENU_OPTION_ADD_ITEM: {

			theme->create_item(theme->get_last_unused_item_id());
		} break;
		case MENU_OPTION_REMOVE_ITEM: {

			String p = editor->get_property_editor()->get_selected_path();
			if (p.begins_with("/MeshLibrary/item") && p.get_slice_count("/") >= 3) {

				to_erase = p.get_slice("/", 3).to_int();
				cd->set_text(vformat(TTR("Remove item %d?"), to_erase));
				cd->popup_centered(Size2(300, 60));
			}
		} break;
		case MENU_OPTION_IMPORT_FROM_SCENE: {

			file->popup_centered_ratio();
		} break;
		case MENU_OPTION_UPDATE_FROM_SCENE: {

			cd->set_text("Update from existing scene?:\n" + String(theme->get_meta("_editor_source_scene")));
			cd->popup_centered(Size2(500, 60));
		} break;
	}
}

void MeshLibraryEditor::_bind_methods() {

	ClassDB::bind_method("_menu_cbk", &MeshLibraryEditor::_menu_cbk);
	ClassDB::bind_method("_menu_confirm", &MeshLibraryEditor::_menu_confirm);
	ClassDB::bind_method("_import_scene_cbk", &MeshLibraryEditor::_import_scene_cbk);
}

MeshLibraryEditor::MeshLibraryEditor(EditorNode *p_editor) {

	file = memnew(EditorFileDialog);
	file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	//not for now?
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);
	file->clear_filters();
	file->set_title(TTR("Import Scene"));
	for (int i = 0; i < extensions.size(); i++) {

		file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
	}
	add_child(file);
	file->connect("file_selected", this, "_import_scene_cbk");

	Panel *panel = memnew(Panel);
	panel->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	add_child(panel);
	MenuButton *options = memnew(MenuButton);
	panel->add_child(options);
	options->set_position(Point2(1, 1));
	options->set_text("Theme");
	options->get_popup()->add_item(TTR("Add Item"), MENU_OPTION_ADD_ITEM);
	options->get_popup()->add_item(TTR("Remove Selected Item"), MENU_OPTION_REMOVE_ITEM);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Import from Scene"), MENU_OPTION_IMPORT_FROM_SCENE);
	options->get_popup()->add_item(TTR("Update from Scene"), MENU_OPTION_UPDATE_FROM_SCENE);
	options->get_popup()->set_item_disabled(options->get_popup()->get_item_index(MENU_OPTION_UPDATE_FROM_SCENE), true);
	options->get_popup()->connect("id_pressed", this, "_menu_cbk");
	menu = options;
	editor = p_editor;
	cd = memnew(ConfirmationDialog);
	add_child(cd);
	cd->get_ok()->connect("pressed", this, "_menu_confirm");
}

void MeshLibraryEditorPlugin::edit(Object *p_node) {

	if (Object::cast_to<MeshLibrary>(p_node)) {
		theme_editor->edit(Object::cast_to<MeshLibrary>(p_node));
		theme_editor->show();
	} else
		theme_editor->hide();
}

bool MeshLibraryEditorPlugin::handles(Object *p_node) const {

	return p_node->is_class("MeshLibrary");
}

void MeshLibraryEditorPlugin::make_visible(bool p_visible) {

	if (p_visible)
		theme_editor->show();
	else
		theme_editor->hide();
}

MeshLibraryEditorPlugin::MeshLibraryEditorPlugin(EditorNode *p_node) {

	EDITOR_DEF("editors/grid_map/preview_size", 64);
	theme_editor = memnew(MeshLibraryEditor(p_node));

	p_node->get_viewport()->add_child(theme_editor);
	theme_editor->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	theme_editor->set_anchor(MARGIN_BOTTOM, Control::ANCHOR_BEGIN);
	theme_editor->set_end(Point2(0, 22));
	theme_editor->hide();
}

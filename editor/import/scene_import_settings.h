/*************************************************************************/
/*  scene_import_settings.h                                              */
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

#ifndef SCENEIMPORTSETTINGS_H
#define SCENEIMPORTSETTINGS_H

#include "editor/editor_file_dialog.h"
#include "editor/editor_inspector.h"
#include "editor/import/resource_importer_scene.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/item_list.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tree.h"
#include "scene/resources/primitive_meshes.h"

class SceneImportSettingsData;

class SceneImportSettings : public ConfirmationDialog {
	GDCLASS(SceneImportSettings, ConfirmationDialog)

	static SceneImportSettings *singleton;

	enum Actions {
		ACTION_EXTRACT_MATERIALS,
		ACTION_CHOOSE_MESH_SAVE_PATHS,
		ACTION_CHOOSE_ANIMATION_SAVE_PATHS,
	};

	Node *scene = nullptr;

	HSplitContainer *tree_split;
	HSplitContainer *property_split;
	TabContainer *data_mode;
	Tree *scene_tree;
	Tree *mesh_tree;
	Tree *material_tree;

	EditorInspector *inspector;

	SubViewport *base_viewport;

	Camera3D *camera;
	bool first_aabb = false;
	AABB contents_aabb;

	DirectionalLight3D *light;
	Ref<ArrayMesh> selection_mesh;
	MeshInstance3D *node_selected;

	MeshInstance3D *mesh_preview;
	Ref<SphereMesh> material_preview;

	Ref<StandardMaterial3D> collider_mat;

	float cam_rot_x;
	float cam_rot_y;
	float cam_zoom;

	void _update_scene();

	struct MaterialData {
		bool has_import_id;
		Ref<Material> material;
		TreeItem *scene_node;
		TreeItem *mesh_node;
		TreeItem *material_node;

		float cam_rot_x = -Math_PI / 4;
		float cam_rot_y = -Math_PI / 4;
		float cam_zoom = 1;

		Map<StringName, Variant> settings;
	};
	Map<String, MaterialData> material_map;

	struct MeshData {
		bool has_import_id;
		Ref<Mesh> mesh;
		TreeItem *scene_node;
		TreeItem *mesh_node;

		float cam_rot_x = -Math_PI / 4;
		float cam_rot_y = -Math_PI / 4;
		float cam_zoom = 1;
		Map<StringName, Variant> settings;
	};
	Map<String, MeshData> mesh_map;

	struct AnimationData {
		Ref<Animation> animation;
		TreeItem *scene_node;
		Map<StringName, Variant> settings;
	};
	Map<String, AnimationData> animation_map;

	struct NodeData {
		Node *node;
		TreeItem *scene_node;
		Map<StringName, Variant> settings;
	};
	Map<String, NodeData> node_map;

	void _fill_material(Tree *p_tree, const Ref<Material> &p_material, TreeItem *p_parent);
	void _fill_mesh(Tree *p_tree, const Ref<Mesh> &p_mesh, TreeItem *p_parent);
	void _fill_animation(Tree *p_tree, const Ref<Animation> &p_anim, const String &p_name, TreeItem *p_parent);
	void _fill_scene(Node *p_node, TreeItem *p_parent_item);

	Set<Ref<Mesh>> mesh_set;
	Set<Ref<Material>> material_set;

	String selected_type;
	String selected_id;

	bool selecting = false;

	void _update_view_gizmos();
	void _update_camera();
	void _select(Tree *p_from, String p_type, String p_id);
	void _material_tree_selected();
	void _mesh_tree_selected();
	void _scene_tree_selected();

	void _viewport_input(const Ref<InputEvent> &p_input);

	Map<StringName, Variant> defaults;

	SceneImportSettingsData *scene_import_settings_data;

	void _re_import();

	String base_path;

	MenuButton *action_menu;

	ConfirmationDialog *external_paths;
	Tree *external_path_tree;
	EditorFileDialog *save_path;
	OptionButton *external_extension_type;

	EditorFileDialog *item_save_path;

	void _menu_callback(int p_id);
	void _save_dir_callback(const String &p_path);

	int current_action;

	Vector<TreeItem *> save_path_items;

	TreeItem *save_path_item = nullptr;
	void _save_path_changed(const String &p_path);
	void _browse_save_callback(Object *p_item, int p_column, int p_id);
	void _save_dir_confirm();

	Dictionary base_subresource_settings;

	void _load_default_subresource_settings(Map<StringName, Variant> &settings, const String &p_type, const String &p_import_id, ResourceImporterScene::InternalImportCategory p_category);

protected:
	void _notification(int p_what);

public:
	void update_view();
	void open_settings(const String &p_path);
	static SceneImportSettings *get_singleton();
	SceneImportSettings();
	~SceneImportSettings();
};

#endif // SCENEIMPORTSETTINGS_H

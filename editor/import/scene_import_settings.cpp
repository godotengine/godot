/*************************************************************************/
/*  scene_import_settings.cpp                                            */
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

#include "scene_import_settings.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/resources/importer_mesh.h"
#include "scene/resources/surface_tool.h"

class SceneImportSettingsData : public Object {
	GDCLASS(SceneImportSettingsData, Object)
	friend class SceneImportSettings;
	Map<StringName, Variant> *settings = nullptr;
	Map<StringName, Variant> current;
	Map<StringName, Variant> defaults;
	List<ResourceImporter::ImportOption> options;

	ResourceImporterScene::InternalImportCategory category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_MAX;

	bool _set(const StringName &p_name, const Variant &p_value) {
		if (settings) {
			if (defaults.has(p_name) && defaults[p_name] == p_value) {
				settings->erase(p_name);
			} else {
				(*settings)[p_name] = p_value;
			}

			current[p_name] = p_value;

			if (ResourceImporterScene::get_singleton()->get_internal_option_update_view_required(category, p_name, current)) {
				SceneImportSettings::get_singleton()->update_view();
			}

			return true;
		}
		return false;
	}
	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (settings) {
			if (settings->has(p_name)) {
				r_ret = (*settings)[p_name];
				return true;
			}
		}
		if (defaults.has(p_name)) {
			r_ret = defaults[p_name];
			return true;
		}
		return false;
	}
	void _get_property_list(List<PropertyInfo> *p_list) const {
		for (const ResourceImporter::ImportOption &E : options) {
			if (ResourceImporterScene::get_singleton()->get_internal_option_visibility(category, E.option.name, current)) {
				p_list->push_back(E.option);
			}
		}
	}
};

void SceneImportSettings::_fill_material(Tree *p_tree, const Ref<Material> &p_material, TreeItem *p_parent) {
	String import_id;
	bool has_import_id = false;

	if (p_material->has_meta("import_id")) {
		import_id = p_material->get_meta("import_id");
		has_import_id = true;
	} else if (!p_material->get_name().is_empty()) {
		import_id = p_material->get_name();
		has_import_id = true;
	} else {
		import_id = "@MATERIAL:" + itos(material_set.size());
	}

	if (!material_map.has(import_id)) {
		MaterialData md;
		md.has_import_id = has_import_id;
		md.material = p_material;

		_load_default_subresource_settings(md.settings, "materials", import_id, ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_MATERIAL);

		material_map[import_id] = md;
	}

	MaterialData &material_data = material_map[import_id];

	Ref<Texture2D> icon = get_theme_icon(SNAME("StandardMaterial3D"), SNAME("EditorIcons"));

	TreeItem *item = p_tree->create_item(p_parent);
	item->set_text(0, p_material->get_name());
	item->set_icon(0, icon);

	bool created = false;
	if (!material_set.has(p_material)) {
		material_set.insert(p_material);
		created = true;
	}

	item->set_meta("type", "Material");
	item->set_meta("import_id", import_id);
	item->set_tooltip(0, vformat(TTR("Import ID: %s"), import_id));
	item->set_selectable(0, true);

	if (p_tree == scene_tree) {
		material_data.scene_node = item;
	} else if (p_tree == mesh_tree) {
		material_data.mesh_node = item;
	} else {
		material_data.material_node = item;
	}

	if (created) {
		_fill_material(material_tree, p_material, material_tree->get_root());
	}
}

void SceneImportSettings::_fill_mesh(Tree *p_tree, const Ref<Mesh> &p_mesh, TreeItem *p_parent) {
	String import_id;

	bool has_import_id = false;
	if (p_mesh->has_meta("import_id")) {
		import_id = p_mesh->get_meta("import_id");
		has_import_id = true;
	} else if (!p_mesh->get_name().is_empty()) {
		import_id = p_mesh->get_name();
		has_import_id = true;
	} else {
		import_id = "@MESH:" + itos(mesh_set.size());
	}

	if (!mesh_map.has(import_id)) {
		MeshData md;
		md.has_import_id = has_import_id;
		md.mesh = p_mesh;

		_load_default_subresource_settings(md.settings, "meshes", import_id, ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_MESH);

		mesh_map[import_id] = md;
	}

	MeshData &mesh_data = mesh_map[import_id];

	Ref<Texture2D> icon = get_theme_icon(SNAME("Mesh"), SNAME("EditorIcons"));

	TreeItem *item = p_tree->create_item(p_parent);
	item->set_text(0, p_mesh->get_name());
	item->set_icon(0, icon);

	bool created = false;
	if (!mesh_set.has(p_mesh)) {
		mesh_set.insert(p_mesh);
		created = true;
	}

	item->set_meta("type", "Mesh");
	item->set_meta("import_id", import_id);
	item->set_tooltip(0, vformat(TTR("Import ID: %s"), import_id));

	item->set_selectable(0, true);

	if (p_tree == scene_tree) {
		mesh_data.scene_node = item;
	} else {
		mesh_data.mesh_node = item;
	}

	item->set_collapsed(true);

	for (int i = 0; i < p_mesh->get_surface_count(); i++) {
		Ref<Material> mat = p_mesh->surface_get_material(i);
		if (mat.is_valid()) {
			_fill_material(p_tree, mat, item);
		}
	}

	if (created) {
		_fill_mesh(mesh_tree, p_mesh, mesh_tree->get_root());
	}
}

void SceneImportSettings::_fill_animation(Tree *p_tree, const Ref<Animation> &p_anim, const String &p_name, TreeItem *p_parent) {
	if (!animation_map.has(p_name)) {
		AnimationData ad;
		ad.animation = p_anim;

		_load_default_subresource_settings(ad.settings, "animations", p_name, ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_ANIMATION);

		animation_map[p_name] = ad;
	}

	AnimationData &animation_data = animation_map[p_name];

	Ref<Texture2D> icon = get_theme_icon(SNAME("Animation"), SNAME("EditorIcons"));

	TreeItem *item = p_tree->create_item(p_parent);
	item->set_text(0, p_name);
	item->set_icon(0, icon);

	item->set_meta("type", "Animation");
	item->set_meta("import_id", p_name);

	item->set_selectable(0, true);

	animation_data.scene_node = item;
}

void SceneImportSettings::_fill_scene(Node *p_node, TreeItem *p_parent_item) {
	String import_id;

	if (p_node->has_meta("import_id")) {
		import_id = p_node->get_meta("import_id");
	} else {
		import_id = "PATH:" + String(scene->get_path_to(p_node));
		p_node->set_meta("import_id", import_id);
	}

	ImporterMeshInstance3D *src_mesh_node = Object::cast_to<ImporterMeshInstance3D>(p_node);

	if (src_mesh_node) {
		MeshInstance3D *mesh_node = memnew(MeshInstance3D);
		mesh_node->set_name(src_mesh_node->get_name());
		mesh_node->set_transform(src_mesh_node->get_transform());
		mesh_node->set_skin(src_mesh_node->get_skin());
		mesh_node->set_skeleton_path(src_mesh_node->get_skeleton_path());
		if (src_mesh_node->get_mesh().is_valid()) {
			Ref<ImporterMesh> editor_mesh = src_mesh_node->get_mesh();
			mesh_node->set_mesh(editor_mesh->get_mesh());
		}

		p_node->replace_by(mesh_node);
		memdelete(p_node);
		p_node = mesh_node;
	}

	String type = p_node->get_class();

	if (!has_theme_icon(type, SNAME("EditorIcons"))) {
		type = "Node3D";
	}

	Ref<Texture2D> icon = get_theme_icon(type, SNAME("EditorIcons"));

	TreeItem *item = scene_tree->create_item(p_parent_item);
	item->set_text(0, p_node->get_name());

	if (p_node == scene) {
		icon = get_theme_icon(SNAME("PackedScene"), SNAME("EditorIcons"));
		item->set_text(0, "Scene");
	}

	item->set_icon(0, icon);

	item->set_meta("type", "Node");
	item->set_meta("class", type);
	item->set_meta("import_id", import_id);
	item->set_tooltip(0, vformat(TTR("Type: %s\nImport ID: %s"), type, import_id));

	item->set_selectable(0, true);

	if (!node_map.has(import_id)) {
		NodeData nd;

		if (p_node != scene) {
			ResourceImporterScene::InternalImportCategory category;
			if (src_mesh_node) {
				category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE;
			} else if (Object::cast_to<AnimationPlayer>(p_node)) {
				category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE;
			} else {
				category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_NODE;
			}

			_load_default_subresource_settings(nd.settings, "nodes", import_id, category);
		}

		node_map[import_id] = nd;
	}
	NodeData &node_data = node_map[import_id];

	node_data.node = p_node;
	node_data.scene_node = item;

	AnimationPlayer *anim_node = Object::cast_to<AnimationPlayer>(p_node);
	if (anim_node) {
		List<StringName> animations;
		anim_node->get_animation_list(&animations);
		for (const StringName &E : animations) {
			_fill_animation(scene_tree, anim_node->get_animation(E), E, item);
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_fill_scene(p_node->get_child(i), item);
	}
	MeshInstance3D *mesh_node = Object::cast_to<MeshInstance3D>(p_node);
	if (mesh_node && mesh_node->get_mesh().is_valid()) {
		_fill_mesh(scene_tree, mesh_node->get_mesh(), item);

		// Add the collider view.
		MeshInstance3D *collider_view = memnew(MeshInstance3D);
		collider_view->set_name("collider_view");
		collider_view->set_visible(false);
		mesh_node->add_child(collider_view, true);
		collider_view->set_owner(mesh_node);

		Transform3D accum_xform;
		Node3D *base = mesh_node;
		while (base) {
			accum_xform = base->get_transform() * accum_xform;
			base = Object::cast_to<Node3D>(base->get_parent());
		}

		AABB aabb = accum_xform.xform(mesh_node->get_mesh()->get_aabb());
		if (first_aabb) {
			contents_aabb = aabb;
			first_aabb = false;
		} else {
			contents_aabb.merge_with(aabb);
		}
	}
}

void SceneImportSettings::_update_scene() {
	scene_tree->clear();
	material_tree->clear();
	mesh_tree->clear();

	//hidden roots
	material_tree->create_item();
	mesh_tree->create_item();

	_fill_scene(scene, nullptr);
}

void SceneImportSettings::_update_view_gizmos() {
	for (const KeyValue<String, NodeData> &e : node_map) {
		bool generate_collider = false;
		if (e.value.settings.has(SNAME("generate/physics"))) {
			generate_collider = e.value.settings[SNAME("generate/physics")];
		}

		MeshInstance3D *mesh_node = Object::cast_to<MeshInstance3D>(e.value.node);
		if (mesh_node == nullptr || mesh_node->get_mesh().is_null()) {
			// Nothing to do
			continue;
		}

		MeshInstance3D *collider_view = static_cast<MeshInstance3D *>(mesh_node->find_node("collider_view"));
		CRASH_COND_MSG(collider_view == nullptr, "This is unreachable, since the collider view is always created even when the collision is not used! If this is triggered there is a bug on the function `_fill_scene`.");

		collider_view->set_visible(generate_collider);
		if (generate_collider) {
			// This collider_view doesn't have a mesh so we need to generate a new one.

			// Generate the mesh collider.
			Vector<Ref<Shape3D>> shapes = ResourceImporterScene::get_collision_shapes(mesh_node->get_mesh(), e.value.settings);
			const Transform3D transform = ResourceImporterScene::get_collision_shapes_transform(e.value.settings);

			Ref<ArrayMesh> collider_view_mesh;
			collider_view_mesh.instantiate();
			for (Ref<Shape3D> shape : shapes) {
				Ref<ArrayMesh> debug_shape_mesh;
				if (shape.is_valid()) {
					debug_shape_mesh = shape->get_debug_mesh();
				}
				if (debug_shape_mesh.is_valid()) {
					collider_view_mesh->add_surface_from_arrays(
							debug_shape_mesh->surface_get_primitive_type(0),
							debug_shape_mesh->surface_get_arrays(0));

					collider_view_mesh->surface_set_material(
							collider_view_mesh->get_surface_count() - 1,
							collider_mat);
				}
			}

			collider_view->set_mesh(collider_view_mesh);
			collider_view->set_transform(transform);
		}
	}
}

void SceneImportSettings::_update_camera() {
	AABB camera_aabb;

	float rot_x = cam_rot_x;
	float rot_y = cam_rot_y;
	float zoom = cam_zoom;

	if (selected_type == "Node" || selected_type.is_empty()) {
		camera_aabb = contents_aabb;
	} else {
		if (mesh_preview->get_mesh().is_valid()) {
			camera_aabb = mesh_preview->get_transform().xform(mesh_preview->get_mesh()->get_aabb());
		} else {
			camera_aabb = AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));
		}
		if (selected_type == "Mesh" && mesh_map.has(selected_id)) {
			const MeshData &md = mesh_map[selected_id];
			rot_x = md.cam_rot_x;
			rot_y = md.cam_rot_y;
			zoom = md.cam_zoom;
		} else if (selected_type == "Material" && material_map.has(selected_id)) {
			const MaterialData &md = material_map[selected_id];
			rot_x = md.cam_rot_x;
			rot_y = md.cam_rot_y;
			zoom = md.cam_zoom;
		}
	}

	Vector3 center = camera_aabb.get_center();
	float camera_size = camera_aabb.get_longest_axis_size();

	camera->set_orthogonal(camera_size * zoom, 0.0001, camera_size * 2);

	Transform3D xf;
	xf.basis = Basis(Vector3(0, 1, 0), rot_y) * Basis(Vector3(1, 0, 0), rot_x);
	xf.origin = center;
	xf.translate(0, 0, camera_size);

	camera->set_transform(xf);
}

void SceneImportSettings::_load_default_subresource_settings(Map<StringName, Variant> &settings, const String &p_type, const String &p_import_id, ResourceImporterScene::InternalImportCategory p_category) {
	if (base_subresource_settings.has(p_type)) {
		Dictionary d = base_subresource_settings[p_type];
		if (d.has(p_import_id)) {
			d = d[p_import_id];
			List<ResourceImporterScene::ImportOption> options;
			ResourceImporterScene::get_singleton()->get_internal_import_options(p_category, &options);
			for (const ResourceImporterScene::ImportOption &E : options) {
				String key = E.option.name;
				if (d.has(key)) {
					settings[key] = d[key];
				}
			}
		}
	}
}

void SceneImportSettings::update_view() {
	_update_view_gizmos();
}

void SceneImportSettings::open_settings(const String &p_path) {
	if (scene) {
		memdelete(scene);
		scene = nullptr;
	}
	scene_import_settings_data->settings = nullptr;
	scene = ResourceImporterScene::get_singleton()->pre_import(p_path);
	if (scene == nullptr) {
		EditorNode::get_singleton()->show_warning(TTR("Error opening scene"));
		return;
	}

	base_path = p_path;

	material_set.clear();
	mesh_set.clear();
	material_map.clear();
	mesh_map.clear();
	node_map.clear();
	defaults.clear();

	selected_id = "";
	selected_type = "";

	cam_rot_x = -Math_PI / 4;
	cam_rot_y = -Math_PI / 4;
	cam_zoom = 1;

	{
		base_subresource_settings.clear();

		Ref<ConfigFile> config;
		config.instantiate();
		Error err = config->load(p_path + ".import");
		if (err == OK) {
			List<String> keys;
			config->get_section_keys("params", &keys);
			for (const String &E : keys) {
				Variant value = config->get_value("params", E);
				if (E == "_subresources") {
					base_subresource_settings = value;
				} else {
					defaults[E] = value;
				}
			}
		}
	}

	first_aabb = true;

	_update_scene();

	base_viewport->add_child(scene);

	if (first_aabb) {
		contents_aabb = AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));
		first_aabb = false;
	}

	popup_centered_ratio();
	_update_view_gizmos();
	_update_camera();

	set_title(vformat(TTR("Advanced Import Settings for '%s'"), base_path.get_file()));
}

SceneImportSettings *SceneImportSettings::singleton = nullptr;

SceneImportSettings *SceneImportSettings::get_singleton() {
	return singleton;
}

void SceneImportSettings::_select(Tree *p_from, String p_type, String p_id) {
	selecting = true;

	if (p_type == "Node") {
		node_selected->hide(); //always hide just in case
		mesh_preview->hide();
		if (Object::cast_to<Node3D>(scene)) {
			Object::cast_to<Node3D>(scene)->show();
		}
		//NodeData &nd=node_map[p_id];
		material_tree->deselect_all();
		mesh_tree->deselect_all();
		NodeData &nd = node_map[p_id];

		MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(nd.node);
		if (mi) {
			Ref<Mesh> base_mesh = mi->get_mesh();
			if (base_mesh.is_valid()) {
				AABB aabb = base_mesh->get_aabb();
				Transform3D aabb_xf;
				aabb_xf.basis.scale(aabb.size);
				aabb_xf.origin = aabb.position;

				aabb_xf = mi->get_global_transform() * aabb_xf;
				node_selected->set_transform(aabb_xf);
				node_selected->show();
			}
		}

		if (nd.node == scene) {
			scene_import_settings_data->settings = &defaults;
			scene_import_settings_data->category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_MAX;
		} else {
			scene_import_settings_data->settings = &nd.settings;
			if (mi) {
				scene_import_settings_data->category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE;
			} else if (Object::cast_to<AnimationPlayer>(nd.node)) {
				scene_import_settings_data->category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE;
			} else {
				scene_import_settings_data->category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_NODE;
			}
		}
	} else if (p_type == "Animation") {
		node_selected->hide(); //always hide just in case
		mesh_preview->hide();
		if (Object::cast_to<Node3D>(scene)) {
			Object::cast_to<Node3D>(scene)->show();
		}
		//NodeData &nd=node_map[p_id];
		material_tree->deselect_all();
		mesh_tree->deselect_all();
		AnimationData &ad = animation_map[p_id];

		scene_import_settings_data->settings = &ad.settings;
		scene_import_settings_data->category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_ANIMATION;
	} else if (p_type == "Mesh") {
		node_selected->hide();
		if (Object::cast_to<Node3D>(scene)) {
			Object::cast_to<Node3D>(scene)->hide();
		}

		MeshData &md = mesh_map[p_id];
		if (p_from != mesh_tree) {
			md.mesh_node->uncollapse_tree();
			md.mesh_node->select(0);
			mesh_tree->ensure_cursor_is_visible();
		}
		if (p_from != scene_tree) {
			md.scene_node->uncollapse_tree();
			md.scene_node->select(0);
			scene_tree->ensure_cursor_is_visible();
		}

		mesh_preview->set_mesh(md.mesh);
		mesh_preview->show();

		material_tree->deselect_all();

		scene_import_settings_data->settings = &md.settings;
		scene_import_settings_data->category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_MESH;
	} else if (p_type == "Material") {
		node_selected->hide();
		if (Object::cast_to<Node3D>(scene)) {
			Object::cast_to<Node3D>(scene)->hide();
		}

		mesh_preview->show();

		MaterialData &md = material_map[p_id];

		material_preview->set_material(md.material);
		mesh_preview->set_mesh(material_preview);

		if (p_from != mesh_tree) {
			md.mesh_node->uncollapse_tree();
			md.mesh_node->select(0);
			mesh_tree->ensure_cursor_is_visible();
		}
		if (p_from != scene_tree) {
			md.scene_node->uncollapse_tree();
			md.scene_node->select(0);
			scene_tree->ensure_cursor_is_visible();
		}
		if (p_from != material_tree) {
			md.material_node->uncollapse_tree();
			md.material_node->select(0);
			material_tree->ensure_cursor_is_visible();
		}

		scene_import_settings_data->settings = &md.settings;
		scene_import_settings_data->category = ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_MATERIAL;
	}

	selected_type = p_type;
	selected_id = p_id;

	selecting = false;

	_update_camera();

	List<ResourceImporter::ImportOption> options;

	if (scene_import_settings_data->category == ResourceImporterScene::INTERNAL_IMPORT_CATEGORY_MAX) {
		ResourceImporterScene::get_singleton()->get_import_options(base_path, &options);
	} else {
		ResourceImporterScene::get_singleton()->get_internal_import_options(scene_import_settings_data->category, &options);
	}

	scene_import_settings_data->defaults.clear();
	scene_import_settings_data->current.clear();

	for (const ResourceImporter::ImportOption &E : options) {
		scene_import_settings_data->defaults[E.option.name] = E.default_value;
		//needed for visibility toggling (fails if something is missing)
		if (scene_import_settings_data->settings->has(E.option.name)) {
			scene_import_settings_data->current[E.option.name] = (*scene_import_settings_data->settings)[E.option.name];
		} else {
			scene_import_settings_data->current[E.option.name] = E.default_value;
		}
	}
	scene_import_settings_data->options = options;
	inspector->edit(scene_import_settings_data);
	scene_import_settings_data->notify_property_list_changed();
}

void SceneImportSettings::_material_tree_selected() {
	if (selecting) {
		return;
	}
	TreeItem *item = material_tree->get_selected();
	String type = item->get_meta("type");
	String import_id = item->get_meta("import_id");

	_select(material_tree, type, import_id);
}

void SceneImportSettings::_mesh_tree_selected() {
	if (selecting) {
		return;
	}

	TreeItem *item = mesh_tree->get_selected();
	String type = item->get_meta("type");
	String import_id = item->get_meta("import_id");

	_select(mesh_tree, type, import_id);
}

void SceneImportSettings::_scene_tree_selected() {
	if (selecting) {
		return;
	}
	TreeItem *item = scene_tree->get_selected();
	String type = item->get_meta("type");
	String import_id = item->get_meta("import_id");

	_select(scene_tree, type, import_id);
}

void SceneImportSettings::_viewport_input(const Ref<InputEvent> &p_input) {
	float *rot_x = &cam_rot_x;
	float *rot_y = &cam_rot_y;
	float *zoom = &cam_zoom;

	if (selected_type == "Mesh" && mesh_map.has(selected_id)) {
		MeshData &md = mesh_map[selected_id];
		rot_x = &md.cam_rot_x;
		rot_y = &md.cam_rot_y;
		zoom = &md.cam_zoom;
	} else if (selected_type == "Material" && material_map.has(selected_id)) {
		MaterialData &md = material_map[selected_id];
		rot_x = &md.cam_rot_x;
		rot_y = &md.cam_rot_y;
		zoom = &md.cam_zoom;
	}
	Ref<InputEventMouseMotion> mm = p_input;
	if (mm.is_valid() && (mm->get_button_mask() & MouseButton::MASK_LEFT) != MouseButton::NONE) {
		(*rot_x) -= mm->get_relative().y * 0.01 * EDSCALE;
		(*rot_y) -= mm->get_relative().x * 0.01 * EDSCALE;
		(*rot_x) = CLAMP((*rot_x), -Math_PI / 2, Math_PI / 2);
		_update_camera();
	}
	Ref<InputEventMouseButton> mb = p_input;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::WHEEL_DOWN) {
		(*zoom) *= 1.1;
		if ((*zoom) > 10.0) {
			(*zoom) = 10.0;
		}
		_update_camera();
	}
	if (mb.is_valid() && mb->get_button_index() == MouseButton::WHEEL_UP) {
		(*zoom) /= 1.1;
		if ((*zoom) < 0.1) {
			(*zoom) = 0.1;
		}
		_update_camera();
	}
}

void SceneImportSettings::_re_import() {
	Map<StringName, Variant> main_settings;

	main_settings = defaults;
	main_settings.erase("_subresources");
	Dictionary nodes;
	Dictionary materials;
	Dictionary meshes;
	Dictionary animations;

	Dictionary subresources;

	for (KeyValue<String, NodeData> &E : node_map) {
		if (E.value.settings.size()) {
			Dictionary d;
			for (const KeyValue<StringName, Variant> &F : E.value.settings) {
				d[String(F.key)] = F.value;
			}
			nodes[E.key] = d;
		}
	}
	if (nodes.size()) {
		subresources["nodes"] = nodes;
	}

	for (KeyValue<String, MaterialData> &E : material_map) {
		if (E.value.settings.size()) {
			Dictionary d;
			for (const KeyValue<StringName, Variant> &F : E.value.settings) {
				d[String(F.key)] = F.value;
			}
			materials[E.key] = d;
		}
	}
	if (materials.size()) {
		subresources["materials"] = materials;
	}

	for (KeyValue<String, MeshData> &E : mesh_map) {
		if (E.value.settings.size()) {
			Dictionary d;
			for (const KeyValue<StringName, Variant> &F : E.value.settings) {
				d[String(F.key)] = F.value;
			}
			meshes[E.key] = d;
		}
	}
	if (meshes.size()) {
		subresources["meshes"] = meshes;
	}

	for (KeyValue<String, AnimationData> &E : animation_map) {
		if (E.value.settings.size()) {
			Dictionary d;
			for (const KeyValue<StringName, Variant> &F : E.value.settings) {
				d[String(F.key)] = F.value;
			}
			animations[E.key] = d;
		}
	}
	if (animations.size()) {
		subresources["animations"] = animations;
	}

	if (subresources.size()) {
		main_settings["_subresources"] = subresources;
	}

	EditorFileSystem::get_singleton()->reimport_file_with_custom_parameters(base_path, "scene", main_settings);
}

void SceneImportSettings::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		connect("confirmed", callable_mp(this, &SceneImportSettings::_re_import));
	}
}

void SceneImportSettings::_menu_callback(int p_id) {
	switch (p_id) {
		case ACTION_EXTRACT_MATERIALS: {
			save_path->set_text(TTR("Select folder to extract material resources"));
			external_extension_type->select(0);
		} break;
		case ACTION_CHOOSE_MESH_SAVE_PATHS: {
			save_path->set_text(TTR("Select folder where mesh resources will save on import"));
			external_extension_type->select(1);
		} break;
		case ACTION_CHOOSE_ANIMATION_SAVE_PATHS: {
			save_path->set_text(TTR("Select folder where animations will save on import"));
			external_extension_type->select(1);
		} break;
	}

	save_path->set_current_dir(base_path.get_base_dir());
	current_action = p_id;
	save_path->popup_centered_ratio();
}

void SceneImportSettings::_save_path_changed(const String &p_path) {
	save_path_item->set_text(1, p_path);

	if (FileAccess::exists(p_path)) {
		save_path_item->set_text(2, "Warning: File exists");
		save_path_item->set_tooltip(2, TTR("Existing file with the same name will be replaced."));
		save_path_item->set_icon(2, get_theme_icon(SNAME("StatusWarning"), SNAME("EditorIcons")));

	} else {
		save_path_item->set_text(2, "Will create new File");
		save_path_item->set_icon(2, get_theme_icon(SNAME("StatusSuccess"), SNAME("EditorIcons")));
	}
}

void SceneImportSettings::_browse_save_callback(Object *p_item, int p_column, int p_id) {
	TreeItem *item = Object::cast_to<TreeItem>(p_item);

	String path = item->get_text(1);

	item_save_path->set_current_file(path);
	save_path_item = item;

	item_save_path->popup_centered_ratio();
}

void SceneImportSettings::_save_dir_callback(const String &p_path) {
	external_path_tree->clear();
	TreeItem *root = external_path_tree->create_item();
	save_path_items.clear();

	switch (current_action) {
		case ACTION_EXTRACT_MATERIALS: {
			for (const KeyValue<String, MaterialData> &E : material_map) {
				MaterialData &md = material_map[E.key];

				TreeItem *item = external_path_tree->create_item(root);

				String name = md.material_node->get_text(0);

				item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
				item->set_icon(0, get_theme_icon(SNAME("StandardMaterial3D"), SNAME("EditorIcons")));
				item->set_text(0, name);

				if (md.has_import_id) {
					if (md.settings.has("use_external/enabled") && bool(md.settings["use_external/enabled"])) {
						item->set_text(2, "Already External");
						item->set_tooltip(2, TTR("This material already references an external file, no action will be taken.\nDisable the external property for it to be extracted again."));
					} else {
						item->set_metadata(0, E.key);
						item->set_editable(0, true);
						item->set_checked(0, true);
						String path = p_path.plus_file(name);
						if (external_extension_type->get_selected() == 0) {
							path += ".tres";
						} else {
							path += ".res";
						}

						item->set_text(1, path);
						if (FileAccess::exists(path)) {
							item->set_text(2, "Warning: File exists");
							item->set_tooltip(2, TTR("Existing file with the same name will be replaced."));
							item->set_icon(2, get_theme_icon(SNAME("StatusWarning"), SNAME("EditorIcons")));

						} else {
							item->set_text(2, "Will create new File");
							item->set_icon(2, get_theme_icon(SNAME("StatusSuccess"), SNAME("EditorIcons")));
						}

						item->add_button(1, get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")));
					}

				} else {
					item->set_text(2, "No import ID");
					item->set_tooltip(2, TTR("Material has no name nor any other way to identify on re-import.\nPlease name it or ensure it is exported with an unique ID."));
					item->set_icon(2, get_theme_icon(SNAME("StatusError"), SNAME("EditorIcons")));
				}

				save_path_items.push_back(item);
			}

			external_paths->set_title(TTR("Extract Materials to Resource Files"));
			external_paths->get_ok_button()->set_text(TTR("Extract"));
		} break;
		case ACTION_CHOOSE_MESH_SAVE_PATHS: {
			for (const KeyValue<String, MeshData> &E : mesh_map) {
				MeshData &md = mesh_map[E.key];

				TreeItem *item = external_path_tree->create_item(root);

				String name = md.mesh_node->get_text(0);

				item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
				item->set_icon(0, get_theme_icon(SNAME("Mesh"), SNAME("EditorIcons")));
				item->set_text(0, name);

				if (md.has_import_id) {
					if (md.settings.has("save_to_file/enabled") && bool(md.settings["save_to_file/enabled"])) {
						item->set_text(2, "Already Saving");
						item->set_tooltip(2, TTR("This mesh already saves to an external resource, no action will be taken."));
					} else {
						item->set_metadata(0, E.key);
						item->set_editable(0, true);
						item->set_checked(0, true);
						String path = p_path.plus_file(name);
						if (external_extension_type->get_selected() == 0) {
							path += ".tres";
						} else {
							path += ".res";
						}

						item->set_text(1, path);
						if (FileAccess::exists(path)) {
							item->set_text(2, "Warning: File exists");
							item->set_tooltip(2, TTR("Existing file with the same name will be replaced on import."));
							item->set_icon(2, get_theme_icon(SNAME("StatusWarning"), SNAME("EditorIcons")));

						} else {
							item->set_text(2, "Will save to new File");
							item->set_icon(2, get_theme_icon(SNAME("StatusSuccess"), SNAME("EditorIcons")));
						}

						item->add_button(1, get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")));
					}

				} else {
					item->set_text(2, "No import ID");
					item->set_tooltip(2, TTR("Mesh has no name nor any other way to identify on re-import.\nPlease name it or ensure it is exported with an unique ID."));
					item->set_icon(2, get_theme_icon(SNAME("StatusError"), SNAME("EditorIcons")));
				}

				save_path_items.push_back(item);
			}

			external_paths->set_title(TTR("Set paths to save meshes as resource files on Reimport"));
			external_paths->get_ok_button()->set_text(TTR("Set Paths"));
		} break;
		case ACTION_CHOOSE_ANIMATION_SAVE_PATHS: {
			for (const KeyValue<String, AnimationData> &E : animation_map) {
				AnimationData &ad = animation_map[E.key];

				TreeItem *item = external_path_tree->create_item(root);

				String name = ad.scene_node->get_text(0);

				item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
				item->set_icon(0, get_theme_icon(SNAME("Animation"), SNAME("EditorIcons")));
				item->set_text(0, name);

				if (ad.settings.has("save_to_file/enabled") && bool(ad.settings["save_to_file/enabled"])) {
					item->set_text(2, "Already Saving");
					item->set_tooltip(2, TTR("This animation already saves to an external resource, no action will be taken."));
				} else {
					item->set_metadata(0, E.key);
					item->set_editable(0, true);
					item->set_checked(0, true);
					String path = p_path.plus_file(name);
					if (external_extension_type->get_selected() == 0) {
						path += ".tres";
					} else {
						path += ".res";
					}

					item->set_text(1, path);
					if (FileAccess::exists(path)) {
						item->set_text(2, "Warning: File exists");
						item->set_tooltip(2, TTR("Existing file with the same name will be replaced on import."));
						item->set_icon(2, get_theme_icon(SNAME("StatusWarning"), SNAME("EditorIcons")));

					} else {
						item->set_text(2, "Will save to new File");
						item->set_icon(2, get_theme_icon(SNAME("StatusSuccess"), SNAME("EditorIcons")));
					}

					item->add_button(1, get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")));
				}

				save_path_items.push_back(item);
			}

			external_paths->set_title(TTR("Set paths to save animations as resource files on Reimport"));
			external_paths->get_ok_button()->set_text(TTR("Set Paths"));

		} break;
	}

	external_paths->popup_centered_ratio();
}

void SceneImportSettings::_save_dir_confirm() {
	for (int i = 0; i < save_path_items.size(); i++) {
		TreeItem *item = save_path_items[i];
		if (!item->is_checked(0)) {
			continue; //ignore
		}
		String path = item->get_text(1);
		if (!path.is_resource_file()) {
			continue;
		}

		String id = item->get_metadata(0);

		switch (current_action) {
			case ACTION_EXTRACT_MATERIALS: {
				ERR_CONTINUE(!material_map.has(id));
				MaterialData &md = material_map[id];

				Error err = ResourceSaver::save(path, md.material);
				if (err != OK) {
					EditorNode::get_singleton()->add_io_error(TTR("Can't make material external to file, write error:") + "\n\t" + path);
					continue;
				}

				md.settings["use_external/enabled"] = true;
				md.settings["use_external/path"] = path;

			} break;
			case ACTION_CHOOSE_MESH_SAVE_PATHS: {
				ERR_CONTINUE(!mesh_map.has(id));
				MeshData &md = mesh_map[id];

				md.settings["save_to_file/enabled"] = true;
				md.settings["save_to_file/path"] = path;
			} break;
			case ACTION_CHOOSE_ANIMATION_SAVE_PATHS: {
				ERR_CONTINUE(!animation_map.has(id));
				AnimationData &ad = animation_map[id];

				ad.settings["save_to_file/enabled"] = true;
				ad.settings["save_to_file/path"] = path;

			} break;
		}
	}

	if (current_action == ACTION_EXTRACT_MATERIALS) {
		//as this happens right now, the scene needs to be saved and reimported.
		_re_import();
		open_settings(base_path);
	} else {
		scene_import_settings_data->notify_property_list_changed();
	}
}

SceneImportSettings::SceneImportSettings() {
	singleton = this;

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);
	HBoxContainer *menu_hb = memnew(HBoxContainer);
	main_vb->add_child(menu_hb);

	action_menu = memnew(MenuButton);
	action_menu->set_text(TTR("Actions..."));
	menu_hb->add_child(action_menu);

	action_menu->get_popup()->add_item(TTR("Extract Materials"), ACTION_EXTRACT_MATERIALS);
	action_menu->get_popup()->add_separator();
	action_menu->get_popup()->add_item(TTR("Set Animation Save Paths"), ACTION_CHOOSE_ANIMATION_SAVE_PATHS);
	action_menu->get_popup()->add_item(TTR("Set Mesh Save Paths"), ACTION_CHOOSE_MESH_SAVE_PATHS);

	action_menu->get_popup()->connect("id_pressed", callable_mp(this, &SceneImportSettings::_menu_callback));

	tree_split = memnew(HSplitContainer);
	main_vb->add_child(tree_split);
	tree_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	data_mode = memnew(TabContainer);
	tree_split->add_child(data_mode);
	data_mode->set_custom_minimum_size(Size2(300 * EDSCALE, 0));

	property_split = memnew(HSplitContainer);
	tree_split->add_child(property_split);
	property_split->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	scene_tree = memnew(Tree);
	scene_tree->set_name(TTR("Scene"));
	data_mode->add_child(scene_tree);
	scene_tree->connect("cell_selected", callable_mp(this, &SceneImportSettings::_scene_tree_selected));

	mesh_tree = memnew(Tree);
	mesh_tree->set_name(TTR("Meshes"));
	data_mode->add_child(mesh_tree);
	mesh_tree->set_hide_root(true);
	mesh_tree->connect("cell_selected", callable_mp(this, &SceneImportSettings::_mesh_tree_selected));

	material_tree = memnew(Tree);
	material_tree->set_name(TTR("Materials"));
	data_mode->add_child(material_tree);
	material_tree->connect("cell_selected", callable_mp(this, &SceneImportSettings::_material_tree_selected));

	material_tree->set_hide_root(true);

	SubViewportContainer *vp_container = memnew(SubViewportContainer);
	vp_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	vp_container->set_custom_minimum_size(Size2(10, 10));
	vp_container->set_stretch(true);
	vp_container->connect("gui_input", callable_mp(this, &SceneImportSettings::_viewport_input));
	property_split->add_child(vp_container);

	base_viewport = memnew(SubViewport);
	vp_container->add_child(base_viewport);

	base_viewport->set_use_own_world_3d(true);

	camera = memnew(Camera3D);
	base_viewport->add_child(camera);
	camera->make_current();

	light = memnew(DirectionalLight3D);
	light->set_transform(Transform3D().looking_at(Vector3(-1, -2, -0.6), Vector3(0, 1, 0)));
	base_viewport->add_child(light);
	light->set_shadow(true);

	{
		Ref<StandardMaterial3D> selection_mat;
		selection_mat.instantiate();
		selection_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
		selection_mat->set_albedo(Color(1, 0.8, 1.0));

		Ref<SurfaceTool> st;
		st.instantiate();
		st->begin(Mesh::PRIMITIVE_LINES);

		AABB base_aabb;
		base_aabb.size = Vector3(1, 1, 1);

		for (int i = 0; i < 12; i++) {
			Vector3 a, b;
			base_aabb.get_edge(i, a, b);

			st->add_vertex(a);
			st->add_vertex(a.lerp(b, 0.2));
			st->add_vertex(b);
			st->add_vertex(b.lerp(a, 0.2));
		}

		selection_mesh.instantiate();
		st->commit(selection_mesh);
		selection_mesh->surface_set_material(0, selection_mat);

		node_selected = memnew(MeshInstance3D);
		node_selected->set_mesh(selection_mesh);
		base_viewport->add_child(node_selected);
		node_selected->hide();
	}

	{
		mesh_preview = memnew(MeshInstance3D);
		base_viewport->add_child(mesh_preview);
		mesh_preview->hide();

		material_preview.instantiate();
	}

	{
		collider_mat.instantiate();
		collider_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
		collider_mat->set_albedo(Color(0.5, 0.5, 1.0));
	}

	inspector = memnew(EditorInspector);
	inspector->set_custom_minimum_size(Size2(300 * EDSCALE, 0));

	property_split->add_child(inspector);

	scene_import_settings_data = memnew(SceneImportSettingsData);

	get_ok_button()->set_text(TTR("Reimport"));
	get_cancel_button()->set_text(TTR("Close"));

	external_paths = memnew(ConfirmationDialog);
	add_child(external_paths);
	external_path_tree = memnew(Tree);
	external_paths->add_child(external_path_tree);
	external_path_tree->connect("button_pressed", callable_mp(this, &SceneImportSettings::_browse_save_callback));
	external_paths->connect("confirmed", callable_mp(this, &SceneImportSettings::_save_dir_confirm));
	external_path_tree->set_columns(3);
	external_path_tree->set_column_titles_visible(true);
	external_path_tree->set_column_expand(0, true);
	external_path_tree->set_column_custom_minimum_width(0, 100 * EDSCALE);
	external_path_tree->set_column_title(0, TTR("Resource"));
	external_path_tree->set_column_expand(1, true);
	external_path_tree->set_column_custom_minimum_width(1, 100 * EDSCALE);
	external_path_tree->set_column_title(1, TTR("Path"));
	external_path_tree->set_column_expand(2, false);
	external_path_tree->set_column_custom_minimum_width(2, 200 * EDSCALE);
	external_path_tree->set_column_title(2, TTR("Status"));
	save_path = memnew(EditorFileDialog);
	save_path->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
	HBoxContainer *extension_hb = memnew(HBoxContainer);
	save_path->get_vbox()->add_child(extension_hb);
	extension_hb->add_spacer();
	extension_hb->add_child(memnew(Label(TTR("Save Extension: "))));
	external_extension_type = memnew(OptionButton);
	extension_hb->add_child(external_extension_type);
	external_extension_type->add_item(TTR("Text: *.tres"));
	external_extension_type->add_item(TTR("Binary: *.res"));
	external_path_tree->set_hide_root(true);
	add_child(save_path);

	item_save_path = memnew(EditorFileDialog);
	item_save_path->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	item_save_path->add_filter("*.tres;Text Resource");
	item_save_path->add_filter("*.res;Binary Resource");
	add_child(item_save_path);
	item_save_path->connect("file_selected", callable_mp(this, &SceneImportSettings::_save_path_changed));

	save_path->connect("dir_selected", callable_mp(this, &SceneImportSettings::_save_dir_callback));
}

SceneImportSettings::~SceneImportSettings() {
	memdelete(scene_import_settings_data);
}

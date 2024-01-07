// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

// #include <godot_cpp/classes/collision_shape3d.hpp>
// #include <godot_cpp/classes/editor_interface.hpp>
// #include <godot_cpp/classes/editor_script.hpp>
// #include <godot_cpp/classes/engine.hpp>
// #include <godot_cpp/classes/height_map_shape3d.hpp>
// #include <godot_cpp/classes/os.hpp>
// #include <godot_cpp/classes/project_settings.hpp>
// #include <godot_cpp/classes/rendering_server.hpp>
// #include <godot_cpp/classes/surface_tool.hpp>
// #include <godot_cpp/classes/time.hpp>
// #include <godot_cpp/classes/v_box_container.hpp> // for get_editor_main_screen()
// #include <godot_cpp/classes/viewport.hpp>
// #include <godot_cpp/classes/world3d.hpp>
// #include <godot_cpp/core/class_db.hpp>

#include "scene/3d/collision_shape_3d.h"
#include "editor/editor_interface.h"
#include "editor/editor_script.h"
#include "core/config/engine.h"
#include "scene/resources/height_map_shape_3d.h"
#include "core/os/os.h"
#include "core/config/project_settings.h"
#include "servers/rendering_server.h"
#include "scene/resources/surface_tool.h"
#include "core/os/time.h"
#include "scene/main/viewport.h"
#include "scene/resources/world_3d.h"
#include "scene/gui/box_container.h"



#include "geoclipmap.h"
#include "logger.h"
#include "terrain_3d.h"
#include "util.h"

///////////////////////////
// Private Functions
///////////////////////////

// Initialize static member variable
int Terrain3D::debug_level{ ERROR };

void Terrain3D::_initialize() {
	LOG(INFO, "Checking material, storage, texture_list, signal, and mesh initialization");

	// Make blank objects if needed
	if (_material.is_null()) {
		LOG(DEBUG, "Creating blank material");
		_material.instantiate();
	}
	if (_storage.is_null()) {
		LOG(DEBUG, "Creating blank storage");
		_storage.instantiate();
		_storage->set_version(Terrain3DStorage::CURRENT_VERSION);
	}
	if (_texture_list.is_null()) {
		LOG(DEBUG, "Creating blank texture list");
		_texture_list.instantiate();
	}

	// Connect signals
	if (!_texture_list->is_connected("textures_changed", Callable(_material.ptr(), "_update_texture_arrays"))) {
		LOG(DEBUG, "Connecting texture_list.textures_changed to _material->_update_texture_arrays()");
		_texture_list->connect("textures_changed", Callable(_material.ptr(), "_update_texture_arrays"));
	}
	if (!_storage->is_connected("region_size_changed", Callable(_material.ptr(), "_set_region_size"))) {
		LOG(DEBUG, "Connecting region_size_changed signal to _material->_set_region_size()");
		_storage->connect("region_size_changed", Callable(_material.ptr(), "_set_region_size"));
	}
	if (!_storage->is_connected("regions_changed", Callable(_material.ptr(), "_update_regions"))) {
		LOG(DEBUG, "Connecting regions_changed signal to _material->_update_regions()");
		_storage->connect("regions_changed", Callable(_material.ptr(), "_update_regions"));
	}
	if (!_storage->is_connected("height_maps_changed", Callable(this, "update_aabbs"))) {
		LOG(DEBUG, "Connecting height_maps_changed signal to update_aabbs()");
		_storage->connect("height_maps_changed", Callable(this, "update_aabbs"));
	}

	// Initialize the system
	if (!_initialized && _is_inside_world && is_inside_tree()) {
		_material->initialize(_storage->get_region_size());
		_storage->update_regions(true); // generate map arrays
		_texture_list->update_list(); // generate texture arrays
		_build(_mesh_lods, _mesh_size);
		_build_collision();
		_initialized = true;
	}
	update_configuration_warnings();
}

void Terrain3D::__ready() {
	_initialize();
	set_process(true);
}

/**
 * This is a proxy for _process(delta) called by _notification() due to
 * https://github.com/godotengine/godot-cpp/issues/1022
 */
void Terrain3D::__process(double delta) {
	if (!_initialized)
		return;

	// If the game/editor camera is not set, find it
	if (!VariantUtilityFunctions::is_instance_valid(_camera)) {
		LOG(DEBUG, "camera is null, getting the current one");
		_grab_camera();
	}

	// If camera has moved enough, re-center the terrain on it.
	if (VariantUtilityFunctions::is_instance_valid(_camera) && _camera->is_inside_tree()) {
		Vector3 cam_pos = _camera->get_global_position();
		Vector2 cam_pos_2d = Vector2(cam_pos.x, cam_pos.z);
		if (_camera_last_position.distance_to(cam_pos_2d) > 0.2f) {
			snap(cam_pos);
			_camera_last_position = cam_pos_2d;
		}
	}
}

/**
 * If running in the editor, recurses into the editor scene tree to find the editor cameras and grabs the first one.
 * The edited_scene_root is excluded in case the user already has a Camera3D in their scene.
 */
void Terrain3D::_grab_camera() {
	if (Engine::get_singleton()->is_editor_hint()) {
		EditorScript temp_editor_script;
		EditorInterface *editor_interface = temp_editor_script.get_editor_interface();
		TypedArray<Camera3D> cam_array = TypedArray<Camera3D>();
		_find_cameras(editor_interface->get_editor_main_screen()->get_children(), editor_interface->get_edited_scene_root(), cam_array);
		if (!cam_array.is_empty()) {
			LOG(DEBUG, "Connecting to the first editor camera");
			_camera = Object::cast_to<Camera3D>(cam_array[0]);
		}
	} else {
		LOG(DEBUG, "Connecting to the in-game viewport camera");
		_camera = get_viewport()->get_camera_3d();
	}
	if (!_camera) {
		set_process(false);
		LOG(ERROR, "Cannot find active camera. Stopping _process()");
	}
}

/**
 * Recursive helper function for _grab_camera().
 */
void Terrain3D::_find_cameras(TypedArray<Node> from_nodes, Node *excluded_node, TypedArray<Camera3D> &cam_array) {
	for (int i = 0; i < from_nodes.size(); i++) {
		Node *node = Object::cast_to<Node>(from_nodes[i]);
		if (node != excluded_node) {
			_find_cameras(node->get_children(), excluded_node, cam_array);
		}
		if (node->is_class("Camera3D")) {
			LOG(DEBUG, "Found a Camera3D at: ", node->get_path());
			cam_array.push_back(node);
		}
	}
}

void Terrain3D::_clear(bool p_clear_meshes, bool p_clear_collision) {
	LOG(INFO, "Clearing the terrain");
	if (p_clear_meshes) {
		for (const RID rid : _meshes) {
			RSS->free(rid);
		}
		RSS->free(_data.cross);
		for (const RID rid : _data.tiles) {
			RSS->free(rid);
		}
		for (const RID rid : _data.fillers) {
			RSS->free(rid);
		}
		for (const RID rid : _data.trims) {
			RSS->free(rid);
		}
		for (const RID rid : _data.seams) {
			RSS->free(rid);
		}

		_meshes.clear();
		_data.tiles.clear();
		_data.fillers.clear();
		_data.trims.clear();
		_data.seams.clear();
		_initialized = false;
	}

	if (p_clear_collision) {
		_destroy_collision();
	}
}

void Terrain3D::_build(int p_mesh_lods, int p_mesh_size) {
	if (!is_inside_tree() || !_storage.is_valid()) {
		LOG(DEBUG, "Not inside the tree or no valid storage, skipping build");
		return;
	}
	LOG(INFO, "Building the terrain meshes");

	// Generate terrain meshes, lods, seams
	_meshes = GeoClipMap::generate(p_mesh_size, p_mesh_lods);
	ERR_FAIL_COND(_meshes.is_empty());

	// Set the current terrain material on all meshes
	RID material_rid = _material->get_material_rid();
	for (const RID rid : _meshes) {
		RSS->mesh_surface_set_material(rid, 0, material_rid);
	}

	LOG(DEBUG, "Creating mesh instances");

	// Get current visual scenario so the instances appear in the scene
	RID scenario = get_world_3d()->get_scenario();

	_data.cross = RSS->instance_create2(_meshes[GeoClipMap::CROSS], scenario);
	RSS->instance_geometry_set_cast_shadows_setting(_data.cross, RenderingServer::ShadowCastingSetting(_shadow_casting));
	RSS->instance_set_layer_mask(_data.cross, _render_layers);

	for (int l = 0; l < p_mesh_lods; l++) {
		for (int x = 0; x < 4; x++) {
			for (int y = 0; y < 4; y++) {
				if (l != 0 && (x == 1 || x == 2) && (y == 1 || y == 2)) {
					continue;
				}

				RID tile = RSS->instance_create2(_meshes[GeoClipMap::TILE], scenario);
				RSS->instance_geometry_set_cast_shadows_setting(tile, RenderingServer::ShadowCastingSetting(_shadow_casting));
				RSS->instance_set_layer_mask(tile, _render_layers);
				_data.tiles.push_back(tile);
			}
		}

		RID filler = RSS->instance_create2(_meshes[GeoClipMap::FILLER], scenario);
		RSS->instance_geometry_set_cast_shadows_setting(filler, RenderingServer::ShadowCastingSetting(_shadow_casting));
		RSS->instance_set_layer_mask(filler, _render_layers);
		_data.fillers.push_back(filler);

		if (l != p_mesh_lods - 1) {
			RID trim = RSS->instance_create2(_meshes[GeoClipMap::TRIM], scenario);
			RSS->instance_geometry_set_cast_shadows_setting(trim, RenderingServer::ShadowCastingSetting(_shadow_casting));
			RSS->instance_set_layer_mask(trim, _render_layers);
			_data.trims.push_back(trim);

			RID seam = RSS->instance_create2(_meshes[GeoClipMap::SEAM], scenario);
			RSS->instance_geometry_set_cast_shadows_setting(seam, RenderingServer::ShadowCastingSetting(_shadow_casting));
			RSS->instance_set_layer_mask(seam, _render_layers);
			_data.seams.push_back(seam);
		}
	}

	update_aabbs();
	// Force a snap update
	_camera_last_position = Vector2(__FLT_MAX__, __FLT_MAX__);
}

void Terrain3D::_build_collision() {
	if (!_collision_enabled || !_is_inside_world || !is_inside_tree()) {
		return;
	}
	// Create collision only in game, unless showing debug
	if (Engine::get_singleton()->is_editor_hint() && !_show_debug_collision) {
		return;
	}
	if (_storage.is_null()) {
		LOG(ERROR, "Storage missing, cannot create collision");
		return;
	}
	_destroy_collision();

	if (!_show_debug_collision) {
		LOG(INFO, "Building collision with physics server");
		_static_body = PhysicsServer3D::get_singleton()->body_create();
		PhysicsServer3D::get_singleton()->body_set_mode(_static_body, PhysicsServer3D::BODY_MODE_STATIC);
		PhysicsServer3D::get_singleton()->body_set_space(_static_body, get_world_3d()->get_space());
		PhysicsServer3D::get_singleton()->body_attach_object_instance_id(_static_body, get_instance_id());
	} else {
		LOG(WARN, "Building debug collision. Disable this mode for releases");
		_debug_static_body = memnew(StaticBody3D);
		_debug_static_body->set_name("StaticBody3D");
		add_child(_debug_static_body, true);
		_debug_static_body->set_owner(this);
	}
	_update_collision();
}

/* Eventually this should be callable to update collision on changes,
 * and especially updating only the ones that have changed. However it's not there yet, so
 * destroy and recreate for now.
 */
void Terrain3D::_update_collision() {
	if (!_collision_enabled || !is_inside_tree()) {
		return;
	}
	// Create collision only in game, unless showing debug
	if (Engine::get_singleton()->is_editor_hint() && !_show_debug_collision) {
		return;
	}
	if ((!_show_debug_collision && !_static_body.is_valid()) ||
			(_show_debug_collision && _debug_static_body == nullptr)) {
		_build_collision();
	}

	int time = Time::get_singleton()->get_ticks_msec();
	int region_size = _storage->get_region_size();
	int shape_size = region_size + 1;
	float hole_const = NAN;
	if (ProjectSettings::get_singleton()->get_setting("physics/3d/physics_engine") == "JoltPhysics3D") {
		hole_const = __FLT_MAX__;
	}

	for (int i = 0; i < _storage->get_region_count(); i++) {
		PackedFloat32Array map_data = PackedFloat32Array();
		map_data.resize(shape_size * shape_size);

		Vector2i global_offset = Vector2i(_storage->get_region_offsets()[i]) * region_size;
		Vector3 global_pos = Vector3(global_offset.x, 0, global_offset.y);

		Ref<Image> map, map_x, map_z, map_xz;
		Ref<Image> cmap, cmap_x, cmap_z, cmap_xz;
		map = _storage->get_map_region(Terrain3DStorage::TYPE_HEIGHT, i);
		cmap = _storage->get_map_region(Terrain3DStorage::TYPE_CONTROL, i);
		int region = _storage->get_region_index(Vector3(global_pos.x + region_size, 0, global_pos.z));
		if (region >= 0) {
			map_x = _storage->get_map_region(Terrain3DStorage::TYPE_HEIGHT, region);
			cmap_x = _storage->get_map_region(Terrain3DStorage::TYPE_CONTROL, region);
		}
		region = _storage->get_region_index(Vector3(global_pos.x, 0, global_pos.z + region_size));
		if (region >= 0) {
			map_z = _storage->get_map_region(Terrain3DStorage::TYPE_HEIGHT, region);
			cmap_z = _storage->get_map_region(Terrain3DStorage::TYPE_CONTROL, region);
		}
		region = _storage->get_region_index(Vector3(global_pos.x + region_size, 0, global_pos.z + region_size));
		if (region >= 0) {
			map_xz = _storage->get_map_region(Terrain3DStorage::TYPE_HEIGHT, region);
			cmap_xz = _storage->get_map_region(Terrain3DStorage::TYPE_CONTROL, region);
		}

		for (int z = 0; z < shape_size; z++) {
			for (int x = 0; x < shape_size; x++) {
				// Choose array indexing to match triangulation of heightmapshape with the mesh
				// https://stackoverflow.com/questions/16684856/rotating-a-2d-pixel-array-by-90-degrees
				// Normal array index rotated Y=0 - shape rotation Y=0 (xform below)
				// int index = z * shape_size + x;
				// Array Index Rotated Y=-90 - must rotate shape Y=+90 (xform below)
				int index = shape_size - 1 - z + x * shape_size;

				// Set heights on local map, or adjacent maps if on the last row/col
				if (x < region_size && z < region_size) {
					map_data.write[index] = (Util::is_hole(cmap->get_pixel(x, z).r)) ? hole_const : map->get_pixel(x, z).r;
				} else if (x == region_size && z < region_size) {
					if (map_x.is_valid()) {
						map_data.write[index] = (Util::is_hole(cmap_x->get_pixel(0, z).r)) ? hole_const : map_x->get_pixel(0, z).r;
					} else {
						map_data.write[index] = 0.0f;
					}
				} else if (z == region_size && x < region_size) {
					if (map_z.is_valid()) {
						map_data.write[index] = (Util::is_hole(cmap_z->get_pixel(x, 0).r)) ? hole_const : map_z->get_pixel(x, 0).r;
					} else {
						map_data.write[index] = 0.0f;
					}
				} else if (x == region_size && z == region_size) {
					if (map_xz.is_valid()) {
						map_data.write[index] = (Util::is_hole(cmap_xz->get_pixel(0, 0).r)) ? hole_const : map_xz->get_pixel(0, 0).r;
					} else {
						map_data.write[index] = 0.0f;
					}
				}
			}
		}

		// Non rotated shape for normal array index above
		//Transform3D xform = Transform3D(Basis(), global_pos);
		// Rotated shape Y=90 for -90 rotated array index
		Transform3D xform = Transform3D(Basis(Vector3(0, 1.0, 0), Math_PI * .5),
				global_pos + Vector3(region_size, 0, region_size) * .5);

		if (!_show_debug_collision) {
			RID shape = PhysicsServer3D::get_singleton()->heightmap_shape_create();
			Dictionary shape_data;
			shape_data["width"] = shape_size;
			shape_data["depth"] = shape_size;
			shape_data["heights"] = map_data;
			Vector2 min_max = _storage->get_height_range();
			shape_data["min_height"] = min_max.x;
			shape_data["max_height"] = min_max.y;
			PhysicsServer3D::get_singleton()->shape_set_data(shape, shape_data);
			PhysicsServer3D::get_singleton()->body_add_shape(_static_body, shape);
			PhysicsServer3D::get_singleton()->body_set_shape_transform(_static_body, i, xform);
			PhysicsServer3D::get_singleton()->body_set_collision_mask(_static_body, _collision_mask);
			PhysicsServer3D::get_singleton()->body_set_collision_layer(_static_body, _collision_layer);
			PhysicsServer3D::get_singleton()->body_set_collision_priority(_static_body, _collision_priority);
		} else {
			CollisionShape3D *debug_col_shape;
			debug_col_shape = memnew(CollisionShape3D);
			debug_col_shape->set_name("CollisionShape3D");
			_debug_static_body->add_child(debug_col_shape, true);
			debug_col_shape->set_owner(this);

			Ref<HeightMapShape3D> hshape;
			hshape.instantiate();
			hshape->set_map_width(shape_size);
			hshape->set_map_depth(shape_size);
			hshape->set_map_data(map_data);
			debug_col_shape->set_shape(hshape);
			debug_col_shape->set_global_transform(xform);
			_debug_static_body->set_collision_mask(_collision_mask);
			_debug_static_body->set_collision_layer(_collision_layer);
			_debug_static_body->set_collision_priority(_collision_priority);
		}
	}
	LOG(DEBUG, "Collision creation time: ", Time::get_singleton()->get_ticks_msec() - time, " ms");
}

void Terrain3D::_destroy_collision() {
	if (_static_body.is_valid()) {
		LOG(INFO, "Freeing physics body");
		RID shape = PhysicsServer3D::get_singleton()->body_get_shape(_static_body, 0);
		PhysicsServer3D::get_singleton()->free(shape);
		PhysicsServer3D::get_singleton()->free(_static_body);
		_static_body = RID();
	}

	if (_debug_static_body != nullptr) {
		LOG(INFO, "Freeing debug static body");
		for (int i = 0; i < _debug_static_body->get_child_count(); i++) {
			Node *child = _debug_static_body->get_child(i);
			LOG(DEBUG, "Freeing dsb child ", i, " ", child->get_name());
			_debug_static_body->remove_child(child);
			memfree(child);
		}

		LOG(DEBUG, "Freeing static body");
		remove_child(_debug_static_body);
		memfree(_debug_static_body);
		_debug_static_body = nullptr;
	}
}

/**
 * Make all mesh instances visible or not
 * Update all mesh instances with the new world scenario so they appear
 */
void Terrain3D::_update_instances() {
	if (!_initialized || !_is_inside_world || !is_inside_tree()) {
		return;
	}
	if (_static_body.is_valid()) {
		RID _space = get_world_3d()->get_space();
		PhysicsServer3D::get_singleton()->body_set_space(_static_body, _space);
	}

	RID _scenario = get_world_3d()->get_scenario();

	bool v = is_visible_in_tree();
	RSS->instance_set_visible(_data.cross, v);
	RSS->instance_set_scenario(_data.cross, _scenario);
	RSS->instance_geometry_set_cast_shadows_setting(_data.cross, RenderingServer::ShadowCastingSetting(_shadow_casting));
	RSS->instance_set_layer_mask(_data.cross, _render_layers);

	for (const RID rid : _data.tiles) {
		RSS->instance_set_visible(rid, v);
		RSS->instance_set_scenario(rid, _scenario);
		RSS->instance_geometry_set_cast_shadows_setting(rid, RenderingServer::ShadowCastingSetting(_shadow_casting));
		RSS->instance_set_layer_mask(rid, _render_layers);
	}

	for (const RID rid : _data.fillers) {
		RSS->instance_set_visible(rid, v);
		RSS->instance_set_scenario(rid, _scenario);
		RSS->instance_geometry_set_cast_shadows_setting(rid, RenderingServer::ShadowCastingSetting(_shadow_casting));
		RSS->instance_set_layer_mask(rid, _render_layers);
	}

	for (const RID rid : _data.trims) {
		RSS->instance_set_visible(rid, v);
		RSS->instance_set_scenario(rid, _scenario);
		RSS->instance_geometry_set_cast_shadows_setting(rid, RenderingServer::ShadowCastingSetting(_shadow_casting));
		RSS->instance_set_layer_mask(rid, _render_layers);
	}

	for (const RID rid : _data.seams) {
		RSS->instance_set_visible(rid, v);
		RSS->instance_set_scenario(rid, _scenario);
		RSS->instance_geometry_set_cast_shadows_setting(rid, RenderingServer::ShadowCastingSetting(_shadow_casting));
		RSS->instance_set_layer_mask(rid, _render_layers);
	}
}

void Terrain3D::_generate_triangles(PackedVector3Array &p_vertices, PackedVector2Array *p_uvs, int32_t p_lod, Terrain3DStorage::HeightFilter p_filter, bool p_require_nav, AABB const &p_global_aabb) const {
	ERR_FAIL_COND(!_storage.is_valid());
	int32_t step = 1 << CLAMP(p_lod, 0, 8);

	if (!p_global_aabb.has_volume()) {
		int32_t region_size = (int)_storage->get_region_size();

		TypedArray<Vector2i> region_offsets = _storage->get_region_offsets();
		for (int r = 0; r < region_offsets.size(); ++r) {
			Vector2i region_offset = (Vector2i)region_offsets[r] * region_size;

			for (int32_t z = region_offset.y; z < region_offset.y + region_size; z += step) {
				for (int32_t x = region_offset.x; x < region_offset.x + region_size; x += step) {
					_generate_triangle_pair(p_vertices, p_uvs, p_lod, p_filter, p_require_nav, x, z);
				}
			}
		}
	} else {
		int32_t z_start = (int32_t)Math::ceil(p_global_aabb.position.z);
		int32_t z_end = (int32_t)Math::floor(p_global_aabb.get_end().z) + 1;
		int32_t x_start = (int32_t)Math::ceil(p_global_aabb.position.x);
		int32_t x_end = (int32_t)Math::floor(p_global_aabb.get_end().x) + 1;

		for (int32_t z = z_start; z < z_end; ++z) {
			for (int32_t x = x_start; x < x_end; ++x) {
				real_t height = _storage->get_height(Vector3(x, 0.0, z));
				if (height >= p_global_aabb.position.y && height <= p_global_aabb.get_end().y) {
					_generate_triangle_pair(p_vertices, p_uvs, p_lod, p_filter, p_require_nav, x, z);
				}
			}
		}
	}
}

void Terrain3D::_generate_triangle_pair(PackedVector3Array &p_vertices, PackedVector2Array *p_uvs, int32_t p_lod, Terrain3DStorage::HeightFilter p_filter, bool p_require_nav, int32_t x, int32_t z) const {
	int32_t step = 1 << CLAMP(p_lod, 0, 8);

	uint32_t control1 = _storage->get_control(Vector3(x, 0.0, z));
	uint32_t control2 = _storage->get_control(Vector3(x + step, 0.0, z + step));
	uint32_t control3 = _storage->get_control(Vector3(x, 0.0, z + step));
	if (!Util::is_hole(control1) && !Util::is_hole(control2) && !Util::is_hole(control3)) {
		if (!p_require_nav || (Util::is_nav(control1) && Util::is_nav(control2) && Util::is_nav(control3))) {
			Vector3 v1 = _storage->get_mesh_vertex(p_lod, p_filter, Vector3(x, 0.0, z));
			Vector3 v2 = _storage->get_mesh_vertex(p_lod, p_filter, Vector3(x + step, 0.0, z + step));
			Vector3 v3 = _storage->get_mesh_vertex(p_lod, p_filter, Vector3(x, 0.0, z + step));
			p_vertices.push_back(v1);
			p_vertices.push_back(v2);
			p_vertices.push_back(v3);
			if (p_uvs != nullptr) {
				p_uvs->push_back(Vector2(v1.x, v1.z));
				p_uvs->push_back(Vector2(v2.x, v2.z));
				p_uvs->push_back(Vector2(v3.x, v3.z));
			}
		}
	}

	control1 = _storage->get_control(Vector3(x, 0.0, z));
	control2 = _storage->get_control(Vector3(x + step, 0.0, z));
	control3 = _storage->get_control(Vector3(x + step, 0.0, z + step));
	if (!Util::is_hole(control1) && !Util::is_hole(control2) && !Util::is_hole(control3)) {
		if (!p_require_nav || (Util::is_nav(control1) && Util::is_nav(control2) && Util::is_nav(control3))) {
			Vector3 v1 = _storage->get_mesh_vertex(p_lod, p_filter, Vector3(x, 0.0, z));
			Vector3 v2 = _storage->get_mesh_vertex(p_lod, p_filter, Vector3(x + step, 0.0, z));
			Vector3 v3 = _storage->get_mesh_vertex(p_lod, p_filter, Vector3(x + step, 0.0, z + step));
			p_vertices.push_back(v1);
			p_vertices.push_back(v2);
			p_vertices.push_back(v3);
			if (p_uvs != nullptr) {
				p_uvs->push_back(Vector2(v1.x, v1.z));
				p_uvs->push_back(Vector2(v2.x, v2.z));
				p_uvs->push_back(Vector2(v3.x, v3.z));
			}
		}
	}
}

///////////////////////////
// Public Functions
///////////////////////////

Terrain3D::Terrain3D() {
	set_notify_transform(true);
	List<String> args = OS::get_singleton()->get_cmdline_args();
	// for (int i = args.size() - 1; i >= 0; i--) {
	// 	String arg = args[i];
	// 	if (arg.begins_with("--terrain3d-debug=")) {
	// 		String value = arg.lstrip("--terrain3d-debug=");
	// 		if (value == "ERROR") {
	// 			set_debug_level(ERROR);
	// 		} else if (value == "INFO") {
	// 			set_debug_level(INFO);
	// 		} else if (value == "DEBUG") {
	// 			set_debug_level(DEBUG);
	// 		} else if (value == "DEBUG_CONT") {
	// 			set_debug_level(DEBUG_CONT);
	// 		}
	// 		break;
	// 	}
	// }
}

Terrain3D::~Terrain3D() {
	_destroy_collision();
}

void Terrain3D::set_debug_level(int p_level) {
	LOG(INFO, "Setting debug level: ", p_level);
	debug_level = CLAMP(p_level, 0, DEBUG_MAX);
}

void Terrain3D::set_mesh_lods(int p_count) {
	if (_mesh_lods != p_count) {
		LOG(INFO, "Setting mesh levels: ", p_count);
		_mesh_lods = p_count;
		_clear();
		_initialize();
	}
}

void Terrain3D::set_mesh_size(int p_size) {
	if (_mesh_size != p_size) {
		LOG(INFO, "Setting mesh size: ", p_size);
		_mesh_size = p_size;
		_clear();
		_initialize();
	}
}

void Terrain3D::set_material(const Ref<Terrain3DMaterial> &p_material) {
	if (_storage != p_material) {
		LOG(INFO, "Setting material");
		_material = p_material;
		_clear();
		_initialize();
		emit_signal("material_changed");
	}
}

// This is run after the object has loaded and initialized
void Terrain3D::set_storage(const Ref<Terrain3DStorage> &p_storage) {
	if (_storage != p_storage) {
		_storage = p_storage;
		if (_storage.is_null()) {
			LOG(INFO, "Clearing storage");
		}
		_clear();
		_initialize();
		emit_signal("storage_changed");
	}
}

void Terrain3D::set_texture_list(const Ref<Terrain3DTextureList> &p_texture_list) {
	if (_texture_list != p_texture_list) {
		LOG(INFO, "Setting texture list");
		_texture_list = p_texture_list;
		_clear();
		_initialize();
		emit_signal("texture_list_changed");
	}
}

void Terrain3D::set_plugin(EditorPlugin *p_plugin) {
	_plugin = p_plugin;
	LOG(DEBUG, "Received editor plugin: ", p_plugin);
}

void Terrain3D::set_camera(Camera3D *p_camera) {
	_camera = p_camera;
	if (p_camera == nullptr) {
		LOG(DEBUG, "Received null camera. Calling _grab_camera");
		_grab_camera();
	} else {
		LOG(DEBUG, "Setting camera: ", p_camera);
		_camera = p_camera;
	}
}

void Terrain3D::set_render_layers(uint32_t p_layers) {
	_render_layers = p_layers;
	_update_instances();
}

void Terrain3D::set_cast_shadows(GeometryInstance3D::ShadowCastingSetting p_shadow_casting) {
	_shadow_casting = p_shadow_casting;
	_update_instances();
}

void Terrain3D::set_cull_margin(real_t p_margin) {
	LOG(INFO, "Setting extra cull margin: ", p_margin);
	_cull_margin = p_margin;
	update_aabbs();
}

void Terrain3D::set_collision_enabled(bool p_enabled) {
	LOG(INFO, "Setting collision enabled: ", p_enabled);
	_collision_enabled = p_enabled;
	if (_collision_enabled) {
		_build_collision();
	} else {
		_destroy_collision();
	}
}

void Terrain3D::set_show_debug_collision(bool p_enabled) {
	LOG(INFO, "Setting show collision: ", p_enabled);
	_show_debug_collision = p_enabled;
	_destroy_collision();
	if (_storage.is_valid() && _show_debug_collision) {
		_build_collision();
	}
}

/**
 * Centers the terrain and LODs on a provided position. Y height is ignored.
 */
void Terrain3D::snap(Vector3 p_cam_pos) {
	p_cam_pos.y = 0;
	LOG(DEBUG_CONT, "Snapping terrain to: ", String(p_cam_pos));

	Transform3D t = Transform3D(Basis(), p_cam_pos.floor());
	RSS->instance_set_transform(_data.cross, t);

	int edge = 0;
	int tile = 0;

	for (int l = 0; l < _mesh_lods; l++) {
		real_t scale = real_t(1 << l);
		Vector3 snapped_pos = (p_cam_pos / scale).floor() * scale;
		Vector3 tile_size = Vector3(real_t(_mesh_size << l), 0, real_t(_mesh_size << l));
		Vector3 base = snapped_pos - Vector3(real_t(_mesh_size << (l + 1)), 0, real_t(_mesh_size << (l + 1)));

		// Position tiles
		for (int x = 0; x < 4; x++) {
			for (int y = 0; y < 4; y++) {
				if (l != 0 && (x == 1 || x == 2) && (y == 1 || y == 2)) {
					continue;
				}

				Vector3 fill = Vector3(x >= 2 ? 1 : 0, 0, y >= 2 ? 1 : 0) * scale;
				Vector3 tile_tl = base + Vector3(x, 0, y) * tile_size + fill;
				//Vector3 tile_br = tile_tl + tile_size;

				Transform3D t = Transform3D().scaled(Vector3(scale, 1, scale));
				t.origin = tile_tl;

				RSS->instance_set_transform(_data.tiles[tile], t);

				tile++;
			}
		}
		{
			Transform3D t = Transform3D().scaled(Vector3(scale, 1, scale));
			t.origin = snapped_pos;
			RSS->instance_set_transform(_data.fillers[l], t);
		}

		if (l != _mesh_lods - 1) {
			real_t next_scale = scale * 2.0f;
			Vector3 next_snapped_pos = (p_cam_pos / next_scale).floor() * next_scale;

			// Position trims
			{
				Vector3 tile_center = snapped_pos + (Vector3(scale, 0, scale) * 0.5f);
				Vector3 d = p_cam_pos - next_snapped_pos;

				int r = 0;
				r |= d.x >= scale ? 0 : 2;
				r |= d.z >= scale ? 0 : 1;

				real_t rotations[4] = { 0.0, 270.0, 90, 180.0 };

				real_t angle = VariantUtilityFunctions::deg_to_rad(rotations[r]);
				Transform3D t = Transform3D().rotated(Vector3(0, 1, 0), -angle);
				t = t.scaled(Vector3(scale, 1, scale));
				t.origin = tile_center;
				RSS->instance_set_transform(_data.trims[edge], t);
			}

			// Position seams
			{
				Vector3 next_base = next_snapped_pos - Vector3(real_t(_mesh_size << (l + 1)), 0, real_t(_mesh_size << (l + 1)));
				Transform3D t = Transform3D().scaled(Vector3(scale, 1, scale));
				t.origin = next_base;
				RSS->instance_set_transform(_data.seams[edge], t);
			}
			edge++;
		}
	}
}

void Terrain3D::update_aabbs() {
	if (_meshes.is_empty() || _storage.is_null()) {
		LOG(DEBUG, "Update AABB called before terrain meshes built. Returning.");
		return;
	}

	Vector2 height_range = _storage->get_height_range();
	LOG(DEBUG_CONT, "Updating AABBs. Total height range: ", height_range, ", extra cull margin: ", _cull_margin);
	height_range.y += abs(height_range.x); // Add below zero to total size

	AABB aabb = RSS->mesh_get_custom_aabb(_meshes[GeoClipMap::CROSS]);
	aabb.position.y = height_range.x;
	aabb.size.y = height_range.y;
	RSS->instance_set_custom_aabb(_data.cross, aabb);
	RSS->instance_set_extra_visibility_margin(_data.cross, _cull_margin);

	aabb = RSS->mesh_get_custom_aabb(_meshes[GeoClipMap::TILE]);
	aabb.position.y = height_range.x;
	aabb.size.y = height_range.y;
	for (int i = 0; i < _data.tiles.size(); i++) {
		RSS->instance_set_custom_aabb(_data.tiles[i], aabb);
		RSS->instance_set_extra_visibility_margin(_data.tiles[i], _cull_margin);
	}

	aabb = RSS->mesh_get_custom_aabb(_meshes[GeoClipMap::FILLER]);
	aabb.position.y = height_range.x;
	aabb.size.y = height_range.y;
	for (int i = 0; i < _data.fillers.size(); i++) {
		RSS->instance_set_custom_aabb(_data.fillers[i], aabb);
		RSS->instance_set_extra_visibility_margin(_data.fillers[i], _cull_margin);
	}

	aabb = RSS->mesh_get_custom_aabb(_meshes[GeoClipMap::TRIM]);
	aabb.position.y = height_range.x;
	aabb.size.y = height_range.y;
	for (int i = 0; i < _data.trims.size(); i++) {
		RSS->instance_set_custom_aabb(_data.trims[i], aabb);
		RSS->instance_set_extra_visibility_margin(_data.trims[i], _cull_margin);
	}

	aabb = RSS->mesh_get_custom_aabb(_meshes[GeoClipMap::SEAM]);
	aabb.position.y = height_range.x;
	aabb.size.y = height_range.y;
	for (int i = 0; i < _data.seams.size(); i++) {
		RSS->instance_set_custom_aabb(_data.seams[i], aabb);
		RSS->instance_set_extra_visibility_margin(_data.seams[i], _cull_margin);
	}
}

/* Iterate over ground to find intersection point between two rays:
 *	p_position (camera position)
 *	p_direction (camera direction looking at the terrain)
 *	test_dir (camera direction 0 Y, traversing terrain along height
 * Returns vec3(Double max 3.402823466e+38F) on no intersection. Test w/ if (var.x < 3.4e38)
 */
Vector3 Terrain3D::get_intersection(Vector3 p_position, Vector3 p_direction) {
	Vector3 test_dir = Vector3(p_direction.x, 0., p_direction.z).normalized();
	Vector3 test_point = p_position;
	p_direction.normalize();

	real_t highest_dotp = 0.f;
	Vector3 highest_point;

	if (_storage.is_valid()) {
		for (int i = 0; i < 3000; i++) {
			test_point += test_dir;
			test_point.y = _storage->get_height(test_point);
			Vector3 test_vec = (test_point - p_position).normalized();

			real_t test_dotp = p_direction.dot(test_vec);
			if (test_dotp > highest_dotp) {
				highest_dotp = test_dotp;
				highest_point = test_point;
			}
			// Highest accuracy hits most of the time
			if (test_dotp > 0.9999) {
				return test_point;
			}
		}

		// Good enough fallback (the above test often overshoots, so this grabs the highest we did hit)
		if (highest_dotp >= 0.999) {
			return highest_point;
		}
	}
	return Vector3(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);
}

/**
 * Generates a static ArrayMesh for the terrain.
 * p_lod (0-8): Determines the granularity of the generated mesh.
 * p_filter: Controls how vertices' Y coordinates are generated from the height map.
 *  HEIGHT_FILTER_NEAREST: Samples the height map in a 'nearest neighbour' fashion.
 *  HEIGHT_FILTER_MINIMUM: Samples a range of heights around each vertex and returns the lowest.
 *   This takes longer than ..._NEAREST, but can be used to create occluders, since it can guarantee the
 *   generated mesh will not extend above or outside the clipmap at any LOD.
 */
Ref<Mesh> Terrain3D::bake_mesh(int p_lod, Terrain3DStorage::HeightFilter p_filter) const {
	LOG(INFO, "Baking mesh at lod: ", p_lod, " with filter: ", p_filter);
	Ref<Mesh> result;
	ERR_FAIL_COND_V(!_storage.is_valid(), result);

	Ref<SurfaceTool> st;
	st.instantiate();
	st->begin(Mesh::PRIMITIVE_TRIANGLES);

	PackedVector3Array vertices;
	PackedVector2Array uvs;
	_generate_triangles(vertices, &uvs, p_lod, p_filter, false, AABB());

	ERR_FAIL_COND_V(vertices.size() != uvs.size(), result);
	for (int i = 0; i < vertices.size(); ++i) {
		st->set_uv(uvs[i]);
		st->add_vertex(vertices[i]);
	}

	st->index();
	st->generate_normals();
	st->generate_tangents();
	st->optimize_indices_for_cache();
	result = st->commit();
	return result;
}

/**
 * Generates source geometry faces for input to nav mesh baking. Geometry is only generated where there
 * are no holes and the terrain has been painted as navigable.
 * p_global_aabb: If non-empty, geometry will be generated only within this AABB. If empty, geometry
 *  will be generated for the entire terrain.
 * p_require_nav: If true, this function will only generate geometry for terrain marked navigable.
 *  Otherwise, geometry is generated for the entire terrain within the AABB (which can be useful for
 *  dynamic and/or runtime nav mesh baking).
 */
PackedVector3Array Terrain3D::generate_nav_mesh_source_geometry(AABB const &p_global_aabb, bool p_require_nav) const {
	LOG(INFO, "Generating NavMesh source geometry from terrain");
	PackedVector3Array faces;
	_generate_triangles(faces, nullptr, 0, Terrain3DStorage::HEIGHT_FILTER_NEAREST, p_require_nav, p_global_aabb);
	return faces;
}

PackedStringArray Terrain3D::_get_configuration_warnings() const {
	PackedStringArray psa;
	if (_storage.is_valid()) {
		String ext = _storage->get_path().get_extension();
		if (ext.is_empty()) {
			psa.push_back("Storage resource is saved in the scene. Click the arrow to the right of Storage, Save as a .res file.");
		} else if (ext != "res") {
			psa.push_back("Storage resource is saved as a ." + ext + ". Click the arrow to the right of Storage, Make Unique, then Save as a .res file.");
		}
	}
	if (!psa.is_empty()) {
		psa.push_back("Deselect Terrain3D and reselect in the Scene Tree to update this message.");
	}
	return psa;
}

///////////////////////////
// Protected Functions
///////////////////////////

void Terrain3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			LOG(INFO, "NOTIFICATION_READY");
			__ready();
			break;
		}

		case NOTIFICATION_PROCESS: {
			__process(get_process_delta_time());
			break;
		}

		case NOTIFICATION_PREDELETE: {
			LOG(INFO, "NOTIFICATION_PREDELETE");
			_clear();
			break;
		}

		case NOTIFICATION_ENTER_TREE: {
			LOG(INFO, "NOTIFICATION_ENTER_TREE");
			_initialize();
			break;
		}

		case NOTIFICATION_EXIT_TREE: {
			LOG(INFO, "NOTIFICATION_EXIT_TREE");
			_clear();
			break;
		}

		case NOTIFICATION_ENTER_WORLD: {
			LOG(INFO, "NOTIFICATION_ENTER_WORLD");
			_is_inside_world = true;
			_update_instances();
			break;
		}

		case NOTIFICATION_TRANSFORM_CHANGED: {
			//LOG(INFO, "NOTIFICATION_TRANSFORM_CHANGED");
			break;
		}

		case NOTIFICATION_EXIT_WORLD: {
			LOG(INFO, "NOTIFICATION_EXIT_WORLD");
			_is_inside_world = false;
			break;
		}

		case NOTIFICATION_VISIBILITY_CHANGED: {
			LOG(INFO, "NOTIFICATION_VISIBILITY_CHANGED");
			_update_instances();
			break;
		}

		case NOTIFICATION_EDITOR_PRE_SAVE: {
			LOG(INFO, "NOTIFICATION_EDITOR_PRE_SAVE");
			if (!_storage.is_valid()) {
				LOG(DEBUG, "Save requested, but no valid storage. Skipping");
			} else {
				_storage->save();
			}
			if (!_material.is_valid()) {
				LOG(DEBUG, "Save requested, but no valid material. Skipping");
			} else {
				_material->save();
			}
			if (!_texture_list.is_valid()) {
				LOG(DEBUG, "Save requested, but no valid texture list. Skipping");
			} else {
				_texture_list->save();
			}
			break;
		}

		case NOTIFICATION_EDITOR_POST_SAVE: {
			//LOG(INFO, "NOTIFICATION_EDITOR_POST_SAVE");
			break;
		}
	}
}

void Terrain3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_version"), &Terrain3D::get_version);
	ClassDB::bind_method(D_METHOD("set_debug_level", "level"), &Terrain3D::set_debug_level);
	ClassDB::bind_method(D_METHOD("get_debug_level"), &Terrain3D::get_debug_level);
	ClassDB::bind_method(D_METHOD("set_mesh_lods", "count"), &Terrain3D::set_mesh_lods);
	ClassDB::bind_method(D_METHOD("get_mesh_lods"), &Terrain3D::get_mesh_lods);
	ClassDB::bind_method(D_METHOD("set_mesh_size", "size"), &Terrain3D::set_mesh_size);
	ClassDB::bind_method(D_METHOD("get_mesh_size"), &Terrain3D::get_mesh_size);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &Terrain3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &Terrain3D::get_material);
	ClassDB::bind_method(D_METHOD("set_storage", "storage"), &Terrain3D::set_storage);
	ClassDB::bind_method(D_METHOD("get_storage"), &Terrain3D::get_storage);
	ClassDB::bind_method(D_METHOD("set_texture_list", "texture_list"), &Terrain3D::set_texture_list);
	ClassDB::bind_method(D_METHOD("get_texture_list"), &Terrain3D::get_texture_list);

	ClassDB::bind_method(D_METHOD("set_plugin", "plugin"), &Terrain3D::set_plugin);
	ClassDB::bind_method(D_METHOD("get_plugin"), &Terrain3D::get_plugin);
	ClassDB::bind_method(D_METHOD("set_camera", "camera"), &Terrain3D::set_camera);
	ClassDB::bind_method(D_METHOD("get_camera"), &Terrain3D::get_camera);

	ClassDB::bind_method(D_METHOD("set_render_layers", "layers"), &Terrain3D::set_render_layers);
	ClassDB::bind_method(D_METHOD("get_render_layers"), &Terrain3D::get_render_layers);
	ClassDB::bind_method(D_METHOD("set_cast_shadows", "shadow_casting_setting"), &Terrain3D::set_cast_shadows);
	ClassDB::bind_method(D_METHOD("get_cast_shadows"), &Terrain3D::get_cast_shadows);
	ClassDB::bind_method(D_METHOD("set_cull_margin", "margin"), &Terrain3D::set_cull_margin);
	ClassDB::bind_method(D_METHOD("get_cull_margin"), &Terrain3D::get_cull_margin);

	ClassDB::bind_method(D_METHOD("set_collision_enabled", "enabled"), &Terrain3D::set_collision_enabled);
	ClassDB::bind_method(D_METHOD("get_collision_enabled"), &Terrain3D::get_collision_enabled);
	ClassDB::bind_method(D_METHOD("set_show_debug_collision", "enabled"), &Terrain3D::set_show_debug_collision);
	ClassDB::bind_method(D_METHOD("get_show_debug_collision"), &Terrain3D::get_show_debug_collision);
	ClassDB::bind_method(D_METHOD("set_collision_layer", "layers"), &Terrain3D::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &Terrain3D::get_collision_layer);
	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &Terrain3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &Terrain3D::get_collision_mask);
	ClassDB::bind_method(D_METHOD("set_collision_priority", "priority"), &Terrain3D::set_collision_priority);
	ClassDB::bind_method(D_METHOD("get_collision_priority"), &Terrain3D::get_collision_priority);

	// Utility functions
	ClassDB::bind_static_method("Terrain3D", D_METHOD("get_min_max", "image"), &Util::get_min_max);
	ClassDB::bind_static_method("Terrain3D", D_METHOD("get_thumbnail", "image", "size"), &Util::get_thumbnail, DEFVAL(Vector2i(256, 256)));
	ClassDB::bind_static_method("Terrain3D", D_METHOD("get_filled_image", "size", "color", "create_mipmaps", "format"), &Util::get_filled_image); //, DEFVAL(Vector2i(256, 256)));

	// Expose 'update_aabbs' so it can be used in Callable. Not ideal.
	ClassDB::bind_method(D_METHOD("update_aabbs"), &Terrain3D::update_aabbs);
	ClassDB::bind_method(D_METHOD("get_intersection", "position", "direction"), &Terrain3D::get_intersection);
	ClassDB::bind_method(D_METHOD("bake_mesh", "lod", "filter"), &Terrain3D::bake_mesh);
	ClassDB::bind_method(D_METHOD("generate_nav_mesh_source_geometry", "global_aabb", "require_nav"), &Terrain3D::generate_nav_mesh_source_geometry, DEFVAL(true));

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "version", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_version");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "storage", PROPERTY_HINT_RESOURCE_TYPE, "Terrain3DStorage"), "set_storage", "get_storage");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "Terrain3DMaterial"), "set_material", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture_list", PROPERTY_HINT_RESOURCE_TYPE, "Terrain3DTextureList"), "set_texture_list", "get_texture_list");

	ADD_GROUP("Renderer", "render_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_layers", PROPERTY_HINT_LAYERS_3D_RENDER), "set_render_layers", "get_render_layers");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_cast_shadows", PROPERTY_HINT_ENUM, "Off,On,Double-Sided,Shadows Only"), "set_cast_shadows", "get_cast_shadows");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "render_cull_margin", PROPERTY_HINT_RANGE, "0, 10000, 1, or_greater"), "set_cull_margin", "get_cull_margin");

	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collision_enabled"), "set_collision_enabled", "get_collision_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_priority"), "set_collision_priority", "get_collision_priority");

	ADD_GROUP("Mesh", "mesh_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh_lods", PROPERTY_HINT_RANGE, "1,10,1"), "set_mesh_lods", "get_mesh_lods");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh_size", PROPERTY_HINT_RANGE, "8,64,1"), "set_mesh_size", "get_mesh_size");

	ADD_GROUP("Debug", "debug_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_level", PROPERTY_HINT_ENUM, "Errors,Info,Debug,Debug Continuous"), "set_debug_level", "get_debug_level");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_show_collision"), "set_show_debug_collision", "get_show_debug_collision");

	ADD_SIGNAL(MethodInfo("material_changed"));
	ADD_SIGNAL(MethodInfo("storage_changed"));
	ADD_SIGNAL(MethodInfo("texture_list_changed"));
}

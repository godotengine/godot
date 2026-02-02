/**************************************************************************/
/*  usd_state.cpp                                                         */
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

#include "usd_state.h"

#include "scene/main/node.h"

void USDState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_base_path"), &USDState::get_base_path);
	ClassDB::bind_method(D_METHOD("set_base_path", "base_path"), &USDState::set_base_path);
	ClassDB::bind_method(D_METHOD("get_filename"), &USDState::get_filename);
	ClassDB::bind_method(D_METHOD("set_filename", "filename"), &USDState::set_filename);
	ClassDB::bind_method(D_METHOD("get_scene_name"), &USDState::get_scene_name);
	ClassDB::bind_method(D_METHOD("set_scene_name", "scene_name"), &USDState::set_scene_name);
	ClassDB::bind_method(D_METHOD("get_root_nodes"), &USDState::get_root_nodes);
	ClassDB::bind_method(D_METHOD("set_root_nodes", "root_nodes"), &USDState::set_root_nodes);
	ClassDB::bind_method(D_METHOD("get_bake_fps"), &USDState::get_bake_fps);
	ClassDB::bind_method(D_METHOD("set_bake_fps", "bake_fps"), &USDState::set_bake_fps);
	ClassDB::bind_method(D_METHOD("get_create_animations"), &USDState::get_create_animations);
	ClassDB::bind_method(D_METHOD("set_create_animations", "create_animations"), &USDState::set_create_animations);
	ClassDB::bind_method(D_METHOD("get_meters_per_unit"), &USDState::get_meters_per_unit);
	ClassDB::bind_method(D_METHOD("set_meters_per_unit", "meters_per_unit"), &USDState::set_meters_per_unit);
	ClassDB::bind_method(D_METHOD("get_up_axis_is_z"), &USDState::get_up_axis_is_z);
	ClassDB::bind_method(D_METHOD("set_up_axis_is_z", "up_axis_is_z"), &USDState::set_up_axis_is_z);
	ClassDB::bind_method(D_METHOD("get_node_count"), &USDState::get_node_count);
	ClassDB::bind_method(D_METHOD("get_node", "index"), &USDState::get_node);
	ClassDB::bind_method(D_METHOD("get_mesh_count"), &USDState::get_mesh_count);
	ClassDB::bind_method(D_METHOD("get_mesh", "index"), &USDState::get_mesh);
	ClassDB::bind_method(D_METHOD("get_material_count"), &USDState::get_material_count);
	ClassDB::bind_method(D_METHOD("get_material", "index"), &USDState::get_material);
	ClassDB::bind_method(D_METHOD("get_light_count"), &USDState::get_light_count);
	ClassDB::bind_method(D_METHOD("get_light", "index"), &USDState::get_light);
	ClassDB::bind_method(D_METHOD("get_camera_count"), &USDState::get_camera_count);
	ClassDB::bind_method(D_METHOD("get_camera", "index"), &USDState::get_camera);
	ClassDB::bind_method(D_METHOD("get_skeleton_count"), &USDState::get_skeleton_count);
	ClassDB::bind_method(D_METHOD("get_skeleton", "index"), &USDState::get_skeleton);
	ClassDB::bind_method(D_METHOD("get_animation_count"), &USDState::get_animation_count);
	ClassDB::bind_method(D_METHOD("get_animation", "index"), &USDState::get_animation);
	ClassDB::bind_method(D_METHOD("get_image_count"), &USDState::get_image_count);
	ClassDB::bind_method(D_METHOD("get_image", "index"), &USDState::get_image);
	ClassDB::bind_method(D_METHOD("get_scene_node", "node_index"), &USDState::get_scene_node);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_path"), "set_base_path", "get_base_path");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "filename"), "set_filename", "get_filename");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "scene_name"), "set_scene_name", "get_scene_name");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "root_nodes"), "set_root_nodes", "get_root_nodes");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bake_fps"), "set_bake_fps", "get_bake_fps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "create_animations"), "set_create_animations", "get_create_animations");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "meters_per_unit"), "set_meters_per_unit", "get_meters_per_unit");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "up_axis_is_z"), "set_up_axis_is_z", "get_up_axis_is_z");
}

String USDState::get_base_path() const {
	return base_path;
}

void USDState::set_base_path(const String &p_base_path) {
	base_path = p_base_path;
}

String USDState::get_filename() const {
	return filename;
}

void USDState::set_filename(const String &p_filename) {
	filename = p_filename;
}

String USDState::get_scene_name() const {
	return scene_name;
}

void USDState::set_scene_name(const String &p_scene_name) {
	scene_name = p_scene_name;
}

PackedInt32Array USDState::get_root_nodes() const {
	PackedInt32Array arr;
	arr.resize(root_nodes.size());
	for (int i = 0; i < root_nodes.size(); i++) {
		arr.set(i, root_nodes[i]);
	}
	return arr;
}

void USDState::set_root_nodes(const PackedInt32Array &p_root_nodes) {
	root_nodes.clear();
	root_nodes.resize(p_root_nodes.size());
	for (int i = 0; i < p_root_nodes.size(); i++) {
		root_nodes.set(i, p_root_nodes[i]);
	}
}

double USDState::get_bake_fps() const {
	return bake_fps;
}

void USDState::set_bake_fps(double p_bake_fps) {
	bake_fps = p_bake_fps;
}

bool USDState::get_create_animations() const {
	return create_animations;
}

void USDState::set_create_animations(bool p_create_animations) {
	create_animations = p_create_animations;
}

float USDState::get_meters_per_unit() const {
	return meters_per_unit;
}

void USDState::set_meters_per_unit(float p_meters_per_unit) {
	meters_per_unit = p_meters_per_unit;
}

bool USDState::get_up_axis_is_z() const {
	return up_axis_is_z;
}

void USDState::set_up_axis_is_z(bool p_up_axis_is_z) {
	up_axis_is_z = p_up_axis_is_z;
}

int USDState::get_node_count() const {
	return nodes.size();
}

Ref<USDNode> USDState::get_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, nodes.size(), Ref<USDNode>());
	return nodes[p_index];
}

int USDState::get_mesh_count() const {
	return meshes.size();
}

Ref<USDMesh> USDState::get_mesh(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, meshes.size(), Ref<USDMesh>());
	return meshes[p_index];
}

int USDState::get_material_count() const {
	return materials.size();
}

Ref<USDMaterial> USDState::get_material(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, materials.size(), Ref<USDMaterial>());
	return materials[p_index];
}

int USDState::get_light_count() const {
	return lights.size();
}

Ref<USDLight> USDState::get_light(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, lights.size(), Ref<USDLight>());
	return lights[p_index];
}

int USDState::get_camera_count() const {
	return cameras.size();
}

Ref<USDCamera> USDState::get_camera(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, cameras.size(), Ref<USDCamera>());
	return cameras[p_index];
}

int USDState::get_skeleton_count() const {
	return skeletons.size();
}

Ref<USDSkeleton> USDState::get_skeleton(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, skeletons.size(), Ref<USDSkeleton>());
	return skeletons[p_index];
}

int USDState::get_animation_count() const {
	return animations.size();
}

Ref<USDAnimation> USDState::get_animation(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, animations.size(), Ref<USDAnimation>());
	return animations[p_index];
}

int USDState::get_image_count() const {
	return images.size();
}

Ref<Texture2D> USDState::get_image(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, images.size(), Ref<Texture2D>());
	return images[p_index];
}

Node *USDState::get_scene_node(int p_node_index) const {
	if (!scene_nodes.has(p_node_index)) {
		return nullptr;
	}
	return scene_nodes[p_node_index];
}

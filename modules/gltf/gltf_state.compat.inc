/**************************************************************************/
/*  gltf_state.compat.inc                                                 */
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

#ifndef DISABLE_DEPRECATED

// Non-const getters for compatibility.
int32_t GLTFState::_get_handle_binary_image_bind_compat_113172() {
	return get_handle_binary_image();
}

Dictionary GLTFState::_get_json_bind_compat_113172() {
	return get_json();
}

int GLTFState::_get_major_version_bind_compat_113172() {
	return get_major_version();
}

int GLTFState::_get_minor_version_bind_compat_113172() {
	return get_minor_version();
}

Vector<uint8_t> GLTFState::_get_glb_data_bind_compat_113172() {
	return get_glb_data();
}

bool GLTFState::_get_use_named_skin_binds_bind_compat_113172() {
	return get_use_named_skin_binds();
}

bool GLTFState::_get_discard_meshes_and_materials_bind_compat_113172() {
	return get_discard_meshes_and_materials();
}

TypedArray<GLTFNode> GLTFState::_get_nodes_bind_compat_113172() {
	return get_nodes_bind();
}

TypedArray<PackedByteArray> GLTFState::_get_buffers_bind_compat_113172() {
	return get_buffers_bind();
}

TypedArray<GLTFBufferView> GLTFState::_get_buffer_views_bind_compat_113172() {
	return get_buffer_views_bind();
}

TypedArray<GLTFAccessor> GLTFState::_get_accessors_bind_compat_113172() {
	return get_accessors_bind();
}

TypedArray<GLTFMesh> GLTFState::_get_meshes_bind_compat_113172() {
	return get_meshes_bind();
}

TypedArray<Material> GLTFState::_get_materials_bind_compat_113172() {
	return get_materials_bind();
}

String GLTFState::_get_scene_name_bind_compat_113172() {
	return get_scene_name();
}

String GLTFState::_get_base_path_bind_compat_113172() {
	return get_base_path();
}

String GLTFState::_get_extract_path_bind_compat_113172() {
	return get_extract_path();
}

String GLTFState::_get_extract_prefix_bind_compat_113172() {
	return get_extract_prefix();
}

PackedInt32Array GLTFState::_get_root_nodes_bind_compat_113172() {
	return get_root_nodes();
}

TypedArray<GLTFTexture> GLTFState::_get_textures_bind_compat_113172() {
	return get_textures_bind();
}

TypedArray<GLTFTextureSampler> GLTFState::_get_texture_samplers_bind_compat_113172() {
	return get_texture_samplers_bind();
}

TypedArray<Texture2D> GLTFState::_get_images_bind_compat_113172() {
	return get_images_bind();
}

TypedArray<GLTFSkin> GLTFState::_get_skins_bind_compat_113172() {
	return get_skins_bind();
}

TypedArray<GLTFCamera> GLTFState::_get_cameras_bind_compat_113172() {
	return get_cameras_bind();
}

TypedArray<GLTFLight> GLTFState::_get_lights_bind_compat_113172() {
	return get_lights_bind();
}

TypedArray<String> GLTFState::_get_unique_names_bind_compat_113172() {
	return get_unique_names_bind();
}

TypedArray<String> GLTFState::_get_unique_animation_names_bind_compat_113172() {
	return get_unique_animation_names_bind();
}

TypedArray<GLTFSkeleton> GLTFState::_get_skeletons_bind_compat_113172() {
	return get_skeletons_bind();
}

bool GLTFState::_get_create_animations_bind_compat_113172() {
	return get_create_animations();
}

bool GLTFState::_get_import_as_skeleton_bones_bind_compat_113172() {
	return get_import_as_skeleton_bones();
}

TypedArray<GLTFAnimation> GLTFState::_get_animations_bind_compat_113172() {
	return get_animations_bind();
}

Node *GLTFState::_get_scene_node_bind_compat_113172(GLTFNodeIndex p_gltf_node_index) {
	return get_scene_node(p_gltf_node_index);
}

GLTFNodeIndex GLTFState::_get_node_index_bind_compat_113172(Node *p_node) {
	return get_node_index(p_node);
}

int GLTFState::_get_animation_players_count_bind_compat_113172(int p_anim_player_index) {
	return get_animation_players_count(p_anim_player_index);
}

AnimationPlayer *GLTFState::_get_animation_player_bind_compat_113172(int p_anim_player_index) {
	return get_animation_player(p_anim_player_index);
}

Variant GLTFState::_get_additional_data_bind_compat_113172(const StringName &p_extension_name) {
	return get_additional_data(p_extension_name);
}

void GLTFState::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("get_handle_binary_image"), &GLTFState::_get_handle_binary_image_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_json"), &GLTFState::_get_json_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_major_version"), &GLTFState::_get_major_version_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_minor_version"), &GLTFState::_get_minor_version_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_glb_data"), &GLTFState::_get_glb_data_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_use_named_skin_binds"), &GLTFState::_get_use_named_skin_binds_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_discard_meshes_and_materials"), &GLTFState::_get_discard_meshes_and_materials_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_nodes"), &GLTFState::_get_nodes_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_buffers"), &GLTFState::_get_buffers_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_buffer_views"), &GLTFState::_get_buffer_views_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_accessors"), &GLTFState::_get_accessors_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_meshes"), &GLTFState::_get_meshes_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_materials"), &GLTFState::_get_materials_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_scene_name"), &GLTFState::_get_scene_name_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_base_path"), &GLTFState::_get_base_path_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_extract_path"), &GLTFState::_get_extract_path_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_extract_prefix"), &GLTFState::_get_extract_prefix_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_root_nodes"), &GLTFState::_get_root_nodes_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_textures"), &GLTFState::_get_textures_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_texture_samplers"), &GLTFState::_get_texture_samplers_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_images"), &GLTFState::_get_images_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_skins"), &GLTFState::_get_skins_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_cameras"), &GLTFState::_get_cameras_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_lights"), &GLTFState::_get_lights_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_unique_names"), &GLTFState::_get_unique_names_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_unique_animation_names"), &GLTFState::_get_unique_animation_names_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_skeletons"), &GLTFState::_get_skeletons_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_create_animations"), &GLTFState::_get_create_animations_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_import_as_skeleton_bones"), &GLTFState::_get_import_as_skeleton_bones_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_animations"), &GLTFState::_get_animations_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_scene_node", "idx"), &GLTFState::_get_scene_node_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_node_index", "node"), &GLTFState::_get_node_index_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_animation_players_count", "idx"), &GLTFState::_get_animation_players_count_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_animation_player", "idx"), &GLTFState::_get_animation_player_bind_compat_113172);
	ClassDB::bind_compatibility_method(D_METHOD("get_additional_data", "extension_name"), &GLTFState::_get_additional_data_bind_compat_113172);
}

#endif // DISABLE_DEPRECATED

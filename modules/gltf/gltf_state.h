/**************************************************************************/
/*  gltf_state.h                                                          */
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

#pragma once

#include "extensions/gltf_light.h"
#include "structures/gltf_accessor.h"
#include "structures/gltf_animation.h"
#include "structures/gltf_buffer_view.h"
#include "structures/gltf_camera.h"
#include "structures/gltf_mesh.h"
#include "structures/gltf_node.h"
#include "structures/gltf_object_model_property.h"
#include "structures/gltf_skeleton.h"
#include "structures/gltf_skin.h"
#include "structures/gltf_texture.h"
#include "structures/gltf_texture_sampler.h"

#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/animation/animation_player.h"

class GLTFState : public Resource {
	GDCLASS(GLTFState, Resource);
	friend class GLTFDocument;
	friend class GLTFNode;

public:
	enum ExternalDataMode {
		EXTERNAL_DATA_MODE_AUTOMATIC,
		EXTERNAL_DATA_MODE_EMBED_EVERYTHING,
		EXTERNAL_DATA_MODE_SEPARATE_ALL_FILES,
		EXTERNAL_DATA_MODE_SEPARATE_BINARY_BLOBS,
		EXTERNAL_DATA_MODE_SEPARATE_RESOURCE_FILES,
	};
	enum HandleBinaryImageMode {
		HANDLE_BINARY_IMAGE_MODE_DISCARD_TEXTURES = 0,
		HANDLE_BINARY_IMAGE_MODE_EXTRACT_TEXTURES,
		HANDLE_BINARY_IMAGE_MODE_EMBED_AS_BASISU,
		HANDLE_BINARY_IMAGE_MODE_EMBED_AS_UNCOMPRESSED, // If this value changes from 3, ResourceImporterScene::pre_import must be changed as well.
	};

protected:
	String base_path;
	String extract_path;
	String extract_prefix;
	String filename;
	Dictionary json;
	int major_version = 0;
	int minor_version = 0;
	String copyright;
	Vector<uint8_t> glb_data;
	double bake_fps = 30.0;

	bool use_named_skin_binds = false;
	bool use_khr_texture_transform = false;
	bool discard_meshes_and_materials = false;
	bool force_generate_tangents = false;
	bool create_animations = true;
	bool force_disable_compression = false;
	bool import_as_skeleton_bones = false;

	ExternalDataMode external_data_mode = ExternalDataMode::EXTERNAL_DATA_MODE_AUTOMATIC;
	HandleBinaryImageMode handle_binary_image_mode = HANDLE_BINARY_IMAGE_MODE_EXTRACT_TEXTURES;

	Vector<Ref<GLTFNode>> nodes;
	Vector<Vector<uint8_t>> buffers;
	Vector<Ref<GLTFBufferView>> buffer_views;
	Vector<Ref<GLTFAccessor>> accessors;

	Vector<Ref<GLTFMesh>> meshes; // Meshes are loaded directly, no reason not to.

	Vector<AnimationPlayer *> animation_players;
	HashMap<Ref<Material>, GLTFMaterialIndex> material_cache;
	Vector<Ref<Material>> materials;

	String scene_name;
	Vector<int> root_nodes;
	Vector<Ref<GLTFTexture>> textures;
	Vector<Ref<GLTFTextureSampler>> texture_samplers;
	Ref<GLTFTextureSampler> default_texture_sampler;
	Vector<Ref<Texture2D>> images;
	Vector<String> extensions_used;
	Vector<String> extensions_required;
	Vector<Ref<Image>> source_images;

	Vector<Ref<GLTFSkin>> skins;
	Vector<Ref<GLTFCamera>> cameras;
	Vector<Ref<GLTFLight>> lights;
	HashSet<String> unique_names;
	HashSet<String> unique_animation_names;

	Vector<Ref<GLTFSkeleton>> skeletons;
	Vector<Ref<GLTFAnimation>> animations;
	HashMap<GLTFNodeIndex, Node *> scene_nodes;
	HashMap<GLTFNodeIndex, ImporterMeshInstance3D *> scene_mesh_instances;
	HashMap<String, Ref<GLTFObjectModelProperty>> object_model_properties;

	HashMap<ObjectID, GLTFSkeletonIndex> skeleton3d_to_gltf_skeleton;
	HashMap<ObjectID, HashMap<ObjectID, GLTFSkinIndex>> skin_and_skeleton3d_to_gltf_skin;
	Dictionary additional_data;

protected:
	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	// Non-const getters for compatibility.
	int32_t _get_handle_binary_image_bind_compat_113172();
	Dictionary _get_json_bind_compat_113172();
	int _get_major_version_bind_compat_113172();
	int _get_minor_version_bind_compat_113172();
	Vector<uint8_t> _get_glb_data_bind_compat_113172();
	bool _get_use_named_skin_binds_bind_compat_113172();
	bool _get_discard_meshes_and_materials_bind_compat_113172();
	TypedArray<GLTFNode> _get_nodes_bind_compat_113172();
	TypedArray<PackedByteArray> _get_buffers_bind_compat_113172();
	TypedArray<GLTFBufferView> _get_buffer_views_bind_compat_113172();
	TypedArray<GLTFAccessor> _get_accessors_bind_compat_113172();
	TypedArray<GLTFMesh> _get_meshes_bind_compat_113172();
	TypedArray<Material> _get_materials_bind_compat_113172();
	String _get_scene_name_bind_compat_113172();
	String _get_base_path_bind_compat_113172();
	String _get_extract_path_bind_compat_113172();
	String _get_extract_prefix_bind_compat_113172();
	PackedInt32Array _get_root_nodes_bind_compat_113172();
	TypedArray<GLTFTexture> _get_textures_bind_compat_113172();
	TypedArray<GLTFTextureSampler> _get_texture_samplers_bind_compat_113172();
	TypedArray<Texture2D> _get_images_bind_compat_113172();
	TypedArray<GLTFSkin> _get_skins_bind_compat_113172();
	TypedArray<GLTFCamera> _get_cameras_bind_compat_113172();
	TypedArray<GLTFLight> _get_lights_bind_compat_113172();
	TypedArray<String> _get_unique_names_bind_compat_113172();
	TypedArray<String> _get_unique_animation_names_bind_compat_113172();
	TypedArray<GLTFSkeleton> _get_skeletons_bind_compat_113172();
	bool _get_create_animations_bind_compat_113172();
	bool _get_import_as_skeleton_bones_bind_compat_113172();
	TypedArray<GLTFAnimation> _get_animations_bind_compat_113172();
	Node *_get_scene_node_bind_compat_113172(GLTFNodeIndex p_gltf_node_index);
	GLTFNodeIndex _get_node_index_bind_compat_113172(Node *p_node);
	int _get_animation_players_count_bind_compat_113172(int p_anim_player_index);
	AnimationPlayer *_get_animation_player_bind_compat_113172(int p_anim_player_index);
	Variant _get_additional_data_bind_compat_113172(const StringName &p_extension_name);
	static void _bind_compatibility_methods();
#endif // DISABLE_DEPRECATED

public:
	double get_bake_fps() const {
		return bake_fps;
	}

	void set_bake_fps(double value) {
		bake_fps = value;
	}

	void add_used_extension(const String &p_extension, bool p_required = false);
	GLTFBufferViewIndex append_data_to_buffers(const Vector<uint8_t> &p_data, const bool p_deduplication);
	GLTFNodeIndex append_gltf_node(Ref<GLTFNode> p_gltf_node, Node *p_godot_scene_node, GLTFNodeIndex p_parent_node_index);

	// Deprecated, use HandleBinaryImageMode instead.
	enum GLTFHandleBinary {
		HANDLE_BINARY_DISCARD_TEXTURES = 0,
		HANDLE_BINARY_EXTRACT_TEXTURES,
		HANDLE_BINARY_EMBED_AS_BASISU,
		HANDLE_BINARY_EMBED_AS_UNCOMPRESSED, // If this value changes from 3, ResourceImporterScene::pre_import must be changed as well.
	};
	int32_t get_handle_binary_image() const {
		return handle_binary_image_mode;
	}
	void set_handle_binary_image(int32_t p_handle_binary_image) {
		handle_binary_image_mode = (HandleBinaryImageMode)p_handle_binary_image;
	}
	HandleBinaryImageMode get_handle_binary_image_mode() const { return handle_binary_image_mode; }
	void set_handle_binary_image_mode(HandleBinaryImageMode p_handle_binary_image) { handle_binary_image_mode = p_handle_binary_image; }

	Dictionary get_json() const;
	void set_json(const Dictionary &p_json);

	int get_major_version() const;
	void set_major_version(int p_major_version);

	int get_minor_version() const;
	void set_minor_version(int p_minor_version);

	String get_copyright() const;
	void set_copyright(const String &p_copyright);

	Vector<uint8_t> get_glb_data() const;
	void set_glb_data(const Vector<uint8_t> &p_glb_data);

	bool get_use_named_skin_binds() const;
	void set_use_named_skin_binds(bool p_use_named_skin_binds);

	bool get_discard_meshes_and_materials() const;
	void set_discard_meshes_and_materials(bool p_discard_meshes_and_materials);

	const Vector<Ref<GLTFNode>> &get_nodes() const { return nodes; }
	void set_nodes(const Vector<Ref<GLTFNode>> &p_nodes) { nodes = p_nodes; }
	TypedArray<GLTFNode> get_nodes_bind() const;
	void set_nodes_bind(const TypedArray<GLTFNode> &p_nodes);

	const Vector<PackedByteArray> &get_buffers() const { return buffers; }
	void set_buffers(const Vector<PackedByteArray> &p_buffers) { buffers = p_buffers; }
	TypedArray<PackedByteArray> get_buffers_bind() const;
	void set_buffers_bind(const TypedArray<PackedByteArray> &p_buffers);

	const Vector<Ref<GLTFBufferView>> &get_buffer_views() const { return buffer_views; }
	void set_buffer_views(const Vector<Ref<GLTFBufferView>> &p_buffer_views) { buffer_views = p_buffer_views; }
	TypedArray<GLTFBufferView> get_buffer_views_bind() const;
	void set_buffer_views_bind(const TypedArray<GLTFBufferView> &p_buffer_views);

	const Vector<Ref<GLTFAccessor>> &get_accessors() const { return accessors; }
	void set_accessors(const Vector<Ref<GLTFAccessor>> &p_accessors) { accessors = p_accessors; }
	TypedArray<GLTFAccessor> get_accessors_bind() const;
	void set_accessors_bind(const TypedArray<GLTFAccessor> &p_accessors);

	const Vector<Ref<GLTFMesh>> &get_meshes() const { return meshes; }
	void set_meshes(const Vector<Ref<GLTFMesh>> &p_meshes) { meshes = p_meshes; }
	TypedArray<GLTFMesh> get_meshes_bind() const;
	void set_meshes_bind(const TypedArray<GLTFMesh> &p_meshes);

	const Vector<Ref<Material>> &get_materials() const { return materials; }
	void set_materials(const Vector<Ref<Material>> &p_materials) { materials = p_materials; }
	TypedArray<Material> get_materials_bind() const;
	void set_materials_bind(const TypedArray<Material> &p_materials);

	String get_scene_name() const;
	void set_scene_name(const String &p_scene_name);

	String get_base_path() const;
	void set_base_path(const String &p_base_path);

	String get_extract_path() const;
	void set_extract_path(const String &p_extract_path);

	String get_extract_prefix() const;
	void set_extract_prefix(const String &p_extract_prefix);

	String get_filename() const;
	void set_filename(const String &p_filename);
	bool is_text_file() const;

	ExternalDataMode get_external_data_mode() const { return external_data_mode; }
	void set_external_data_mode(ExternalDataMode p_external_data_mode) { external_data_mode = p_external_data_mode; }
	bool should_separate_binary_blobs() const;
	bool should_separate_resource_files() const;

	PackedInt32Array get_root_nodes() const;
	void set_root_nodes(const PackedInt32Array &p_root_nodes);

	const Vector<Ref<GLTFTexture>> &get_textures() const { return textures; }
	void set_textures(const Vector<Ref<GLTFTexture>> &p_textures) { textures = p_textures; }
	TypedArray<GLTFTexture> get_textures_bind() const;
	void set_textures_bind(const TypedArray<GLTFTexture> &p_textures);

	const Vector<Ref<GLTFTextureSampler>> &get_texture_samplers() const { return texture_samplers; }
	void set_texture_samplers(const Vector<Ref<GLTFTextureSampler>> &p_texture_samplers) { texture_samplers = p_texture_samplers; }
	TypedArray<GLTFTextureSampler> get_texture_samplers_bind() const;
	void set_texture_samplers_bind(const TypedArray<GLTFTextureSampler> &p_texture_samplers);

	const Vector<Ref<Texture2D>> &get_images() const { return images; }
	void set_images(const Vector<Ref<Texture2D>> &p_images) { images = p_images; }
	TypedArray<Texture2D> get_images_bind() const;
	void set_images_bind(const TypedArray<Texture2D> &p_images);

	const Vector<Ref<GLTFSkin>> &get_skins() const { return skins; }
	void set_skins(const Vector<Ref<GLTFSkin>> &p_skins) { skins = p_skins; }
	TypedArray<GLTFSkin> get_skins_bind() const;
	void set_skins_bind(const TypedArray<GLTFSkin> &p_skins);

	const Vector<Ref<GLTFCamera>> &get_cameras() const { return cameras; }
	void set_cameras(const Vector<Ref<GLTFCamera>> &p_cameras) { cameras = p_cameras; }
	TypedArray<GLTFCamera> get_cameras_bind() const;
	void set_cameras_bind(const TypedArray<GLTFCamera> &p_cameras);

	const Vector<Ref<GLTFLight>> &get_lights() const { return lights; }
	void set_lights(const Vector<Ref<GLTFLight>> &p_lights) { lights = p_lights; }
	TypedArray<GLTFLight> get_lights_bind() const;
	void set_lights_bind(const TypedArray<GLTFLight> &p_lights);

	const HashSet<String> &get_unique_names() const { return unique_names; }
	void set_unique_names(const HashSet<String> &p_unique_names) { unique_names = p_unique_names; }
	TypedArray<String> get_unique_names_bind() const;
	void set_unique_names_bind(const TypedArray<String> &p_unique_names);

	const HashSet<String> &get_unique_animation_names() const { return unique_animation_names; }
	void set_unique_animation_names(const HashSet<String> &p_unique_animation_names) { unique_animation_names = p_unique_animation_names; }
	TypedArray<String> get_unique_animation_names_bind() const;
	void set_unique_animation_names_bind(const TypedArray<String> &p_unique_names);

	const Vector<Ref<GLTFSkeleton>> &get_skeletons() const { return skeletons; }
	void set_skeletons(const Vector<Ref<GLTFSkeleton>> &p_skeletons) { skeletons = p_skeletons; }
	TypedArray<GLTFSkeleton> get_skeletons_bind() const;
	void set_skeletons_bind(const TypedArray<GLTFSkeleton> &p_skeletons);

	bool get_create_animations() const;
	void set_create_animations(bool p_create_animations);

	bool get_import_as_skeleton_bones() const;
	void set_import_as_skeleton_bones(bool p_import_as_skeleton_bones);

	const Vector<Ref<GLTFAnimation>> &get_animations() const { return animations; }
	void set_animations(const Vector<Ref<GLTFAnimation>> &p_animations) { animations = p_animations; }
	TypedArray<GLTFAnimation> get_animations_bind() const;
	void set_animations_bind(const TypedArray<GLTFAnimation> &p_animations);

	Node *get_scene_node(GLTFNodeIndex p_gltf_node_index) const;
	GLTFNodeIndex get_node_index(Node *p_node) const;

	int get_animation_players_count(int p_anim_player_index) const;

	AnimationPlayer *get_animation_player(int p_anim_player_index) const;

	Variant get_additional_data(const StringName &p_extension_name) const;
	void set_additional_data(const StringName &p_extension_name, Variant p_additional_data);
};

VARIANT_ENUM_CAST(GLTFState::ExternalDataMode);
VARIANT_ENUM_CAST(GLTFState::HandleBinaryImageMode);

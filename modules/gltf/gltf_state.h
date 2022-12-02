/*************************************************************************/
/*  gltf_state.h                                                         */
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

#ifndef GLTF_STATE_H
#define GLTF_STATE_H

#include "extensions/gltf_light.h"
#include "gltf_template_convert.h"
#include "structures/gltf_accessor.h"
#include "structures/gltf_animation.h"
#include "structures/gltf_buffer_view.h"
#include "structures/gltf_camera.h"
#include "structures/gltf_mesh.h"
#include "structures/gltf_node.h"
#include "structures/gltf_skeleton.h"
#include "structures/gltf_skin.h"
#include "structures/gltf_texture.h"
#include "structures/gltf_texture_sampler.h"

#include "core/templates/rb_map.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/texture.h"

class GLTFState : public Resource {
	GDCLASS(GLTFState, Resource);
	friend class GLTFDocument;

	String filename;
	String base_path;
	Dictionary json;
	int major_version = 0;
	int minor_version = 0;
	Vector<uint8_t> glb_data;

	bool use_named_skin_binds = false;
	bool use_khr_texture_transform = false;
	bool discard_meshes_and_materials = false;
	bool create_animations = true;

	Vector<Ref<GLTFNode>> nodes;
	Vector<Vector<uint8_t>> buffers;
	Vector<Ref<GLTFBufferView>> buffer_views;
	Vector<Ref<GLTFAccessor>> accessors;

	Vector<Ref<GLTFMesh>> meshes; // meshes are loaded directly, no reason not to.

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

	Vector<Ref<GLTFSkin>> skins;
	Vector<Ref<GLTFCamera>> cameras;
	Vector<Ref<GLTFLight>> lights;
	HashSet<String> unique_names;
	HashSet<String> unique_animation_names;

	Vector<Ref<GLTFSkeleton>> skeletons;
	HashMap<GLTFSkeletonIndex, GLTFNodeIndex> skeleton_to_node;
	Vector<Ref<GLTFAnimation>> animations;
	HashMap<GLTFNodeIndex, Node *> scene_nodes;

	HashMap<ObjectID, GLTFSkeletonIndex> skeleton3d_to_gltf_skeleton;
	HashMap<ObjectID, HashMap<ObjectID, GLTFSkinIndex>> skin_and_skeleton3d_to_gltf_skin;
	Dictionary additional_data;

protected:
	static void _bind_methods();

public:
	void add_used_extension(const String &p_extension, bool p_required = false);

	Dictionary get_json();
	void set_json(Dictionary p_json);

	int get_major_version();
	void set_major_version(int p_major_version);

	int get_minor_version();
	void set_minor_version(int p_minor_version);

	Vector<uint8_t> get_glb_data();
	void set_glb_data(Vector<uint8_t> p_glb_data);

	bool get_use_named_skin_binds();
	void set_use_named_skin_binds(bool p_use_named_skin_binds);

	bool get_discard_meshes_and_materials();
	void set_discard_meshes_and_materials(bool p_discard_meshes_and_materials);

	TypedArray<GLTFNode> get_nodes();
	void set_nodes(TypedArray<GLTFNode> p_nodes);

	TypedArray<PackedByteArray> get_buffers();
	void set_buffers(TypedArray<PackedByteArray> p_buffers);

	TypedArray<GLTFBufferView> get_buffer_views();
	void set_buffer_views(TypedArray<GLTFBufferView> p_buffer_views);

	TypedArray<GLTFAccessor> get_accessors();
	void set_accessors(TypedArray<GLTFAccessor> p_accessors);

	TypedArray<GLTFMesh> get_meshes();
	void set_meshes(TypedArray<GLTFMesh> p_meshes);

	TypedArray<Material> get_materials();
	void set_materials(TypedArray<Material> p_materials);

	String get_scene_name();
	void set_scene_name(String p_scene_name);

	String get_base_path();
	void set_base_path(String p_base_path);

	PackedInt32Array get_root_nodes();
	void set_root_nodes(PackedInt32Array p_root_nodes);

	TypedArray<GLTFTexture> get_textures();
	void set_textures(TypedArray<GLTFTexture> p_textures);

	TypedArray<GLTFTextureSampler> get_texture_samplers();
	void set_texture_samplers(TypedArray<GLTFTextureSampler> p_texture_samplers);

	TypedArray<Texture2D> get_images();
	void set_images(TypedArray<Texture2D> p_images);

	TypedArray<GLTFSkin> get_skins();
	void set_skins(TypedArray<GLTFSkin> p_skins);

	TypedArray<GLTFCamera> get_cameras();
	void set_cameras(TypedArray<GLTFCamera> p_cameras);

	TypedArray<GLTFLight> get_lights();
	void set_lights(TypedArray<GLTFLight> p_lights);

	TypedArray<String> get_unique_names();
	void set_unique_names(TypedArray<String> p_unique_names);

	TypedArray<String> get_unique_animation_names();
	void set_unique_animation_names(TypedArray<String> p_unique_names);

	TypedArray<GLTFSkeleton> get_skeletons();
	void set_skeletons(TypedArray<GLTFSkeleton> p_skeletons);

	Dictionary get_skeleton_to_node();
	void set_skeleton_to_node(Dictionary p_skeleton_to_node);

	bool get_create_animations();
	void set_create_animations(bool p_create_animations);

	TypedArray<GLTFAnimation> get_animations();
	void set_animations(TypedArray<GLTFAnimation> p_animations);

	Node *get_scene_node(GLTFNodeIndex idx);

	int get_animation_players_count(int idx);

	AnimationPlayer *get_animation_player(int idx);

	Variant get_additional_data(const StringName &p_extension_name);
	void set_additional_data(const StringName &p_extension_name, Variant p_additional_data);

	//void set_scene_nodes(RBMap<GLTFNodeIndex, Node *> p_scene_nodes) {
	//	this->scene_nodes = p_scene_nodes;
	//}

	//void set_animation_players(Vector<AnimationPlayer *> p_animation_players) {
	//	this->animation_players = p_animation_players;
	//}

	//RBMap<Ref<Material>, GLTFMaterialIndex> get_material_cache() {
	//	return this->material_cache;
	//}
	//void set_material_cache(RBMap<Ref<Material>, GLTFMaterialIndex> p_material_cache) {
	//	this->material_cache = p_material_cache;
	//}
};

#endif // GLTF_STATE_H

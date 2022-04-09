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

#include "gltf_accessor.h"
#include "gltf_animation.h"
#include "gltf_buffer_view.h"
#include "gltf_camera.h"
#include "gltf_document.h"
#include "gltf_document_extension.h"
#include "gltf_light.h"
#include "gltf_mesh.h"
#include "gltf_node.h"
#include "gltf_skeleton.h"
#include "gltf_skin.h"
#include "gltf_texture.h"

#include "core/io/resource.h"
#include "core/templates/map.h"
#include "core/templates/pair.h"
#include "core/templates/vector.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/texture.h"

class GLTFState : public Resource {
	GDCLASS(GLTFState, Resource);
	friend class GLTFDocument;

	String filename;
	Dictionary json;
	int major_version = 0;
	int minor_version = 0;
	Vector<uint8_t> glb_data;

	bool use_named_skin_binds = false;

	Vector<Ref<GLTFNode>> nodes;
	Vector<Vector<uint8_t>> buffers;
	Vector<Ref<GLTFBufferView>> buffer_views;
	Vector<Ref<GLTFAccessor>> accessors;

	Vector<Ref<GLTFMesh>> meshes; // meshes are loaded directly, no reason not to.

	Vector<AnimationPlayer *> animation_players;
	Map<Ref<BaseMaterial3D>, GLTFMaterialIndex> material_cache;
	Vector<Ref<BaseMaterial3D>> materials;

	String scene_name;
	Vector<int> root_nodes;
	Vector<Ref<GLTFTexture>> textures;
	Vector<Ref<Texture2D>> images;

	Vector<Ref<GLTFSkin>> skins;
	Vector<Ref<GLTFCamera>> cameras;
	Vector<Ref<GLTFLight>> lights;
	Set<String> unique_names;
	Set<String> unique_animation_names;

	Vector<Ref<GLTFSkeleton>> skeletons;
	Map<GLTFSkeletonIndex, GLTFNodeIndex> skeleton_to_node;
	Vector<Ref<GLTFAnimation>> animations;
	Map<GLTFNodeIndex, Node *> scene_nodes;

	Map<ObjectID, GLTFSkeletonIndex> skeleton3d_to_gltf_skeleton;
	Map<ObjectID, Map<ObjectID, GLTFSkinIndex>> skin_and_skeleton3d_to_gltf_skin;

protected:
	static void _bind_methods();

public:
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

	Array get_nodes();
	void set_nodes(Array p_nodes);

	Array get_buffers();
	void set_buffers(Array p_buffers);

	Array get_buffer_views();
	void set_buffer_views(Array p_buffer_views);

	Array get_accessors();
	void set_accessors(Array p_accessors);

	Array get_meshes();
	void set_meshes(Array p_meshes);

	Array get_materials();
	void set_materials(Array p_materials);

	String get_scene_name();
	void set_scene_name(String p_scene_name);

	Array get_root_nodes();
	void set_root_nodes(Array p_root_nodes);

	Array get_textures();
	void set_textures(Array p_textures);

	Array get_images();
	void set_images(Array p_images);

	Array get_skins();
	void set_skins(Array p_skins);

	Array get_cameras();
	void set_cameras(Array p_cameras);

	Array get_lights();
	void set_lights(Array p_lights);

	Array get_unique_names();
	void set_unique_names(Array p_unique_names);

	Array get_unique_animation_names();
	void set_unique_animation_names(Array p_unique_names);

	Array get_skeletons();
	void set_skeletons(Array p_skeletons);

	Dictionary get_skeleton_to_node();
	void set_skeleton_to_node(Dictionary p_skeleton_to_node);

	Array get_animations();
	void set_animations(Array p_animations);

	Node *get_scene_node(GLTFNodeIndex idx);

	int get_animation_players_count(int idx);

	AnimationPlayer *get_animation_player(int idx);

	//void set_scene_nodes(Map<GLTFNodeIndex, Node *> p_scene_nodes) {
	//	this->scene_nodes = p_scene_nodes;
	//}

	//void set_animation_players(Vector<AnimationPlayer *> p_animation_players) {
	//	this->animation_players = p_animation_players;
	//}

	//Map<Ref<Material>, GLTFMaterialIndex> get_material_cache() {
	//	return this->material_cache;
	//}
	//void set_material_cache(Map<Ref<Material>, GLTFMaterialIndex> p_material_cache) {
	//	this->material_cache = p_material_cache;
	//}
};
#endif // GLTF_STATE_H

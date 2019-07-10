/*************************************************************************/
/*  editor_scene_importer_assimp.cpp                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "assimp/DefaultLogger.hpp"
#include "assimp/Importer.hpp"
#include "assimp/LogStream.hpp"
#include "assimp/Logger.hpp"
#include "assimp/SceneCombiner.h"
#include "assimp/cexport.h"
#include "assimp/cimport.h"
#include "assimp/matrix4x4.h"
#include "assimp/pbrmaterial.h"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

#include "core/bind/core_bind.h"
#include "core/io/image_loader.h"
#include "editor/editor_file_system.h"
#include "editor/import/resource_importer_scene.h"
#include "editor_scene_importer_assimp.h"
#include "editor_settings.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/animation/animation_player.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"
#include "scene/resources/surface_tool.h"
#include "zutil.h"
#include <string>

void EditorSceneImporterAssimp::get_extensions(List<String> *r_extensions) const {

	const String import_setting_string = "filesystem/import/open_asset_import/";

	Map<String, ImportFormat> import_format;
	{
		Vector<String> exts;
		exts.push_back("fbx");
		ImportFormat import = { exts, true };
		import_format.insert("fbx", import);
	}
	{
		Vector<String> exts;
		exts.push_back("pmx");
		ImportFormat import = { exts, true };
		import_format.insert("mmd", import);
	}
	for (Map<String, ImportFormat>::Element *E = import_format.front(); E; E = E->next()) {
		_register_project_setting_import(E->key(), import_setting_string, E->get().extensions, r_extensions, E->get().is_default);
	}
}

void EditorSceneImporterAssimp::_register_project_setting_import(const String generic, const String import_setting_string, const Vector<String> &exts, List<String> *r_extensions, const bool p_enabled) const {
	const String use_generic = "use_" + generic;
	_GLOBAL_DEF(import_setting_string + use_generic, p_enabled, true);
	if (ProjectSettings::get_singleton()->get(import_setting_string + use_generic)) {
		for (int32_t i = 0; i < exts.size(); i++) {
			r_extensions->push_back(exts[i]);
		}
	}
}

uint32_t EditorSceneImporterAssimp::get_import_flags() const {
	return IMPORT_SCENE;
}

AssimpStream::AssimpStream() {
	// empty
}

AssimpStream::~AssimpStream() {
	// empty
}

void AssimpStream::write(const char *message) {
	print_verbose(String("Open Asset Import: ") + String(message).strip_edges());
}

void EditorSceneImporterAssimp::_bind_methods() {
}

Node *EditorSceneImporterAssimp::import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err) {
	Assimp::Importer importer;
	std::wstring w_path = ProjectSettings::get_singleton()->globalize_path(p_path).c_str();
	std::string s_path(w_path.begin(), w_path.end());
	importer.SetPropertyBool(AI_CONFIG_PP_FD_REMOVE, true);
	// Cannot remove pivot points because the static mesh will be in the wrong place
	importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, false);
	int32_t max_bone_weights = 4;
	//if (p_flags & IMPORT_ANIMATION_EIGHT_WEIGHTS) {
	//	const int eight_bones = 8;
	//	importer.SetPropertyBool(AI_CONFIG_PP_LBW_MAX_WEIGHTS, eight_bones);
	//	max_bone_weights = eight_bones;
	//}

	importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
	//importer.SetPropertyFloat(AI_CONFIG_PP_DB_THRESHOLD, 1.0f);
	int32_t post_process_Steps = aiProcess_CalcTangentSpace |
								 //aiProcess_FlipUVs |
								 //aiProcess_FlipWindingOrder |
								 //aiProcess_DropNormals |
								 //aiProcess_GenSmoothNormals |
								 aiProcess_JoinIdenticalVertices |
								 aiProcess_ImproveCacheLocality |
								 aiProcess_LimitBoneWeights |
								 //aiProcess_RemoveRedundantMaterials | // Causes a crash
								 aiProcess_SplitLargeMeshes |
								 aiProcess_Triangulate |
								 aiProcess_GenUVCoords |
								 //aiProcess_FindDegenerates |
								 aiProcess_SortByPType |
								 aiProcess_FindInvalidData |
								 aiProcess_TransformUVCoords |
								 aiProcess_FindInstances |
								 //aiProcess_FixInfacingNormals |
								 //aiProcess_ValidateDataStructure |
								 aiProcess_OptimizeMeshes |
								 //aiProcess_OptimizeGraph |
								 //aiProcess_Debone |
								 aiProcess_EmbedTextures |
								 aiProcess_SplitByBoneCount |
								 0;
	const aiScene *scene = importer.ReadFile(s_path.c_str(),
			post_process_Steps);
	ERR_EXPLAIN(String("Open Asset Import failed to open: ") + String(importer.GetErrorString()));
	ERR_FAIL_COND_V(scene == NULL, NULL);
	return _generate_scene(p_path, scene, p_flags, p_bake_fps, max_bone_weights);
}

template <class T>
struct EditorSceneImporterAssetImportInterpolate {

	T lerp(const T &a, const T &b, float c) const {

		return a + (b - a) * c;
	}

	T catmull_rom(const T &p0, const T &p1, const T &p2, const T &p3, float t) {

		float t2 = t * t;
		float t3 = t2 * t;

		return 0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4 * p2 - p3) * t2 + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
	}

	T bezier(T start, T control_1, T control_2, T end, float t) {
		/* Formula from Wikipedia article on Bezier curves. */
		real_t omt = (1.0 - t);
		real_t omt2 = omt * omt;
		real_t omt3 = omt2 * omt;
		real_t t2 = t * t;
		real_t t3 = t2 * t;

		return start * omt3 + control_1 * omt2 * t * 3.0 + control_2 * omt * t2 * 3.0 + end * t3;
	}
};

//thank you for existing, partial specialization
template <>
struct EditorSceneImporterAssetImportInterpolate<Quat> {

	Quat lerp(const Quat &a, const Quat &b, float c) const {
		ERR_FAIL_COND_V(!a.is_normalized(), Quat());
		ERR_FAIL_COND_V(!b.is_normalized(), Quat());

		return a.slerp(b, c).normalized();
	}

	Quat catmull_rom(const Quat &p0, const Quat &p1, const Quat &p2, const Quat &p3, float c) {
		ERR_FAIL_COND_V(!p1.is_normalized(), Quat());
		ERR_FAIL_COND_V(!p2.is_normalized(), Quat());

		return p1.slerp(p2, c).normalized();
	}

	Quat bezier(Quat start, Quat control_1, Quat control_2, Quat end, float t) {
		ERR_FAIL_COND_V(!start.is_normalized(), Quat());
		ERR_FAIL_COND_V(!end.is_normalized(), Quat());

		return start.slerp(end, t).normalized();
	}
};

template <class T>
T EditorSceneImporterAssimp::_interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time, AssetImportAnimation::Interpolation p_interp) {
	//could use binary search, worth it?
	int idx = -1;
	for (int i = 0; i < p_times.size(); i++) {
		if (p_times[i] > p_time)
			break;
		idx++;
	}

	EditorSceneImporterAssetImportInterpolate<T> interp;

	switch (p_interp) {
		case AssetImportAnimation::INTERP_LINEAR: {

			if (idx == -1) {
				return p_values[0];
			} else if (idx >= p_times.size() - 1) {
				return p_values[p_times.size() - 1];
			}

			float c = (p_time - p_times[idx]) / (p_times[idx + 1] - p_times[idx]);

			return interp.lerp(p_values[idx], p_values[idx + 1], c);

		} break;
		case AssetImportAnimation::INTERP_STEP: {

			if (idx == -1) {
				return p_values[0];
			} else if (idx >= p_times.size() - 1) {
				return p_values[p_times.size() - 1];
			}

			return p_values[idx];

		} break;
		case AssetImportAnimation::INTERP_CATMULLROMSPLINE: {

			if (idx == -1) {
				return p_values[1];
			} else if (idx >= p_times.size() - 1) {
				return p_values[1 + p_times.size() - 1];
			}

			float c = (p_time - p_times[idx]) / (p_times[idx + 1] - p_times[idx]);

			return interp.catmull_rom(p_values[idx - 1], p_values[idx], p_values[idx + 1], p_values[idx + 3], c);

		} break;
		case AssetImportAnimation::INTERP_CUBIC_SPLINE: {

			if (idx == -1) {
				return p_values[1];
			} else if (idx >= p_times.size() - 1) {
				return p_values[(p_times.size() - 1) * 3 + 1];
			}

			float c = (p_time - p_times[idx]) / (p_times[idx + 1] - p_times[idx]);

			T from = p_values[idx * 3 + 1];
			T c1 = from + p_values[idx * 3 + 2];
			T to = p_values[idx * 3 + 4];
			T c2 = to + p_values[idx * 3 + 3];

			return interp.bezier(from, c1, c2, to, c);

		} break;
	}

	ERR_FAIL_V(p_values[0]);
}

void EditorSceneImporterAssimp::_generate_bone_groups(ImportState &state, const aiNode *p_assimp_node, Map<String, int> &ownership, Map<String, Transform> &bind_xforms) {

	Transform mesh_offset = _get_global_assimp_node_transform(p_assimp_node);
	//mesh_offset.basis = Basis();
	for (uint32_t i = 0; i < p_assimp_node->mNumMeshes; i++) {
		const aiMesh *mesh = state.assimp_scene->mMeshes[i];
		int owned_by = -1;
		for (uint32_t j = 0; j < mesh->mNumBones; j++) {
			const aiBone *bone = mesh->mBones[j];
			String name = _assimp_get_string(bone->mName);

			if (ownership.has(name)) {
				owned_by = ownership[name];
				break;
			}
		}

		if (owned_by == -1) { //no owned, create new unique id
			owned_by = 1;
			for (Map<String, int>::Element *E = ownership.front(); E; E = E->next()) {
				owned_by = MAX(E->get() + 1, owned_by);
			}
		}

		for (uint32_t j = 0; j < mesh->mNumBones; j++) {
			const aiBone *bone = mesh->mBones[j];
			String name = _assimp_get_string(bone->mName);
			ownership[name] = owned_by;
			//store the actual full path for the bone transform
			//when skeleton finds its place in the tree, it will be restored
			bind_xforms[name] = mesh_offset * _assimp_matrix_transform(bone->mOffsetMatrix);
		}
	}

	for (size_t i = 0; i < p_assimp_node->mNumChildren; i++) {
		_generate_bone_groups(state, p_assimp_node->mChildren[i], ownership, bind_xforms);
	}
}

void EditorSceneImporterAssimp::_fill_node_relationships(ImportState &state, const aiNode *p_assimp_node, Map<String, int> &ownership, Map<int, int> &skeleton_map, int p_skeleton_id, Skeleton *p_skeleton, const String &p_parent_name, int &holecount, const Vector<SkeletonHole> &p_holes, const Map<String, Transform> &bind_xforms) {

	String name = _assimp_get_string(p_assimp_node->mName);
	if (name == String()) {
		name = "AuxiliaryBone" + itos(holecount++);
	}

	Transform pose = _assimp_matrix_transform(p_assimp_node->mTransformation);

	if (!ownership.has(name)) {
		//not a bone, it's a hole
		Vector<SkeletonHole> holes = p_holes;
		SkeletonHole hole; //add a new one
		hole.name = name;
		hole.pose = pose;
		hole.node = p_assimp_node;
		hole.parent = p_parent_name;
		holes.push_back(hole);

		for (size_t i = 0; i < p_assimp_node->mNumChildren; i++) {
			_fill_node_relationships(state, p_assimp_node->mChildren[i], ownership, skeleton_map, p_skeleton_id, p_skeleton, name, holecount, holes, bind_xforms);
		}

		return;
	} else if (ownership[name] != p_skeleton_id) {
		//oh, it's from another skeleton? fine.. reparent all bones to this skeleton.
		int prev_owner = ownership[name];
		ERR_EXPLAIN("A previous skeleton exists for bone '" + name + "', this type of skeleton layout is unsupported.");
		ERR_FAIL_COND(skeleton_map.has(prev_owner));
		for (Map<String, int>::Element *E = ownership.front(); E; E = E->next()) {
			if (E->get() == prev_owner) {
				E->get() = p_skeleton_id;
			}
		}
	}

	//valid bone, first fill holes if needed
	for (int i = 0; i < p_holes.size(); i++) {

		int bone_idx = p_skeleton->get_bone_count();
		p_skeleton->add_bone(p_holes[i].name);
		int parent_idx = p_skeleton->find_bone(p_holes[i].parent);
		if (parent_idx >= 0) {
			p_skeleton->set_bone_parent(bone_idx, parent_idx);
		}

		Transform pose_transform = _get_global_assimp_node_transform(p_holes[i].node);
		p_skeleton->set_bone_rest(bone_idx, pose_transform);

		state.bone_owners[p_holes[i].name] = skeleton_map[p_skeleton_id];
	}

	//finally fill bone

	int bone_idx = p_skeleton->get_bone_count();
	p_skeleton->add_bone(name);
	int parent_idx = p_skeleton->find_bone(p_parent_name);
	if (parent_idx >= 0) {
		p_skeleton->set_bone_parent(bone_idx, parent_idx);
	}
	//p_skeleton->set_bone_pose(bone_idx, pose);
	if (bind_xforms.has(name)) {
		//for now this is the full path to the bone in rest pose
		//when skeleton finds it's place in the tree, it will get fixed
		p_skeleton->set_bone_rest(bone_idx, bind_xforms[name]);
	}
	state.bone_owners[name] = skeleton_map[p_skeleton_id];
	//go to children
	for (size_t i = 0; i < p_assimp_node->mNumChildren; i++) {
		_fill_node_relationships(state, p_assimp_node->mChildren[i], ownership, skeleton_map, p_skeleton_id, p_skeleton, name, holecount, Vector<SkeletonHole>(), bind_xforms);
	}
}

void EditorSceneImporterAssimp::_generate_skeletons(ImportState &state, const aiNode *p_assimp_node, Map<String, int> &ownership, Map<int, int> &skeleton_map, const Map<String, Transform> &bind_xforms) {

	//find skeletons at this level, there may be multiple root nodes for each
	Map<int, List<aiNode *> > skeletons_found;
	for (size_t i = 0; i < p_assimp_node->mNumChildren; i++) {
		String name = _assimp_get_string(p_assimp_node->mChildren[i]->mName);
		if (ownership.has(name)) {
			int skeleton = ownership[name];
			if (!skeletons_found.has(skeleton)) {
				skeletons_found[skeleton] = List<aiNode *>();
			}
			skeletons_found[skeleton].push_back(p_assimp_node->mChildren[i]);
		}
	}

	//go via the potential skeletons found and generate the actual skeleton
	for (Map<int, List<aiNode *> >::Element *E = skeletons_found.front(); E; E = E->next()) {
		ERR_CONTINUE(skeleton_map.has(E->key())); //skeleton already exists? this can't be.. skip
		Skeleton *skeleton = memnew(Skeleton);
		//this the only way to reliably use multiple meshes with one skeleton, at the cost of less precision
		skeleton->set_use_bones_in_world_transform(true);
		skeleton_map[E->key()] = state.skeletons.size();
		state.skeletons.push_back(skeleton);
		int holecount = 1;
		//fill the bones and their relationships
		for (List<aiNode *>::Element *F = E->get().front(); F; F = F->next()) {
			_fill_node_relationships(state, F->get(), ownership, skeleton_map, E->key(), skeleton, "", holecount, Vector<SkeletonHole>(), bind_xforms);
		}
	}

	//go to the children
	for (uint32_t i = 0; i < p_assimp_node->mNumChildren; i++) {
		String name = _assimp_get_string(p_assimp_node->mChildren[i]->mName);
		if (ownership.has(name)) {
			continue; //a bone, so don't bother with this
		}
		_generate_skeletons(state, p_assimp_node->mChildren[i], ownership, skeleton_map, bind_xforms);
	}
}

Spatial *EditorSceneImporterAssimp::_generate_scene(const String &p_path, const aiScene *scene, const uint32_t p_flags, int p_bake_fps, const int32_t p_max_bone_weights) {
	ERR_FAIL_COND_V(scene == NULL, NULL);

	ImportState state;
	state.path = p_path;
	state.assimp_scene = scene;
	state.max_bone_weights = p_max_bone_weights;
	state.root = memnew(Spatial);
	state.fbx = false;
	state.animation_player = NULL;

	real_t scale_factor = 1.0f;
	{
		//handle scale
		String ext = p_path.get_file().get_extension().to_lower();
		if (ext == "fbx") {
			if (scene->mMetaData != NULL) {
				float factor = 1.0;
				scene->mMetaData->Get("UnitScaleFactor", factor);
				scale_factor = factor * 0.01f;
			}
			state.fbx = true;
		}
	}

	state.root->set_scale(Vector3(scale_factor, scale_factor, scale_factor));

	//fill light map cache
	for (size_t l = 0; l < scene->mNumLights; l++) {

		aiLight *ai_light = scene->mLights[l];
		ERR_CONTINUE(ai_light == NULL);
		state.light_cache[_assimp_get_string(ai_light->mName)] = l;
	}

	//fill camera cache
	for (size_t c = 0; c < scene->mNumCameras; c++) {
		aiCamera *ai_camera = scene->mCameras[c];
		ERR_CONTINUE(ai_camera == NULL);
		state.camera_cache[_assimp_get_string(ai_camera->mName)] = c;
	}

	if (scene->mRootNode) {
		Map<String, Transform> bind_xforms; //temporary map to store bind transforms
		//guess the skeletons, since assimp does not really support them directly
		Map<String, int> ownership; //bone names to groups
		//fill this map with bone names and which group where they detected to, going mesh by mesh
		_generate_bone_groups(state, state.assimp_scene->mRootNode, ownership, bind_xforms);
		Map<int, int> skeleton_map; //maps previously created groups to actual skeletons
		//generates the skeletons when bones are found in the hierarchy, and follows them (including gaps/holes).
		_generate_skeletons(state, state.assimp_scene->mRootNode, ownership, skeleton_map, bind_xforms);

		//generate nodes
		for (uint32_t i = 0; i < scene->mRootNode->mNumChildren; i++) {
			_generate_node(state, scene->mRootNode->mChildren[i], state.root);
		}

		//assign skeletons to nodes

		for (Map<MeshInstance *, Skeleton *>::Element *E = state.mesh_skeletons.front(); E; E = E->next()) {
			MeshInstance *mesh = E->key();
			Skeleton *skeleton = E->get();
			NodePath skeleton_path = mesh->get_path_to(skeleton);
			mesh->set_skeleton_path(skeleton_path);
		}
	}

	if (p_flags & IMPORT_ANIMATION && scene->mNumAnimations) {

		state.animation_player = memnew(AnimationPlayer);
		state.root->add_child(state.animation_player);
		state.animation_player->set_owner(state.root);

		for (uint32_t i = 0; i < scene->mNumAnimations; i++) {
			_import_animation(state, i, p_bake_fps);
		}
	}

	return state.root;
}

void EditorSceneImporterAssimp::_insert_animation_track(ImportState &scene, const aiAnimation *assimp_anim, int p_track, int p_bake_fps, Ref<Animation> animation, float ticks_per_second, Skeleton *p_skeleton, const NodePath &p_path, const String &p_name) {

	const aiNodeAnim *assimp_track = assimp_anim->mChannels[p_track];
	//make transform track
	int track_idx = animation->get_track_count();
	animation->add_track(Animation::TYPE_TRANSFORM);
	animation->track_set_path(track_idx, p_path);
	//first determine animation length

	float increment = 1.0 / float(p_bake_fps);
	float time = 0.0;

	bool last = false;

	int skeleton_bone = -1;

	if (p_skeleton) {
		skeleton_bone = p_skeleton->find_bone(p_name);
	}

	Vector<Vector3> pos_values;
	Vector<float> pos_times;
	Vector<Vector3> scale_values;
	Vector<float> scale_times;
	Vector<Quat> rot_values;
	Vector<float> rot_times;

	for (size_t p = 0; p < assimp_track->mNumPositionKeys; p++) {
		aiVector3D pos = assimp_track->mPositionKeys[p].mValue;
		pos_values.push_back(Vector3(pos.x, pos.y, pos.z));
		pos_times.push_back(assimp_track->mPositionKeys[p].mTime / ticks_per_second);
	}

	for (size_t r = 0; r < assimp_track->mNumRotationKeys; r++) {
		aiQuaternion quat = assimp_track->mRotationKeys[r].mValue;
		rot_values.push_back(Quat(quat.x, quat.y, quat.z, quat.w).normalized());
		rot_times.push_back(assimp_track->mRotationKeys[r].mTime / ticks_per_second);
	}

	for (size_t sc = 0; sc < assimp_track->mNumScalingKeys; sc++) {
		aiVector3D scale = assimp_track->mScalingKeys[sc].mValue;
		scale_values.push_back(Vector3(scale.x, scale.y, scale.z));
		scale_times.push_back(assimp_track->mScalingKeys[sc].mTime / ticks_per_second);
	}
	while (true) {
		Vector3 pos;
		Quat rot;
		Vector3 scale(1, 1, 1);

		if (pos_values.size()) {
			pos = _interpolate_track<Vector3>(pos_times, pos_values, time, AssetImportAnimation::INTERP_LINEAR);
		}

		if (rot_values.size()) {
			rot = _interpolate_track<Quat>(rot_times, rot_values, time, AssetImportAnimation::INTERP_LINEAR).normalized();
		}

		if (scale_values.size()) {
			scale = _interpolate_track<Vector3>(scale_times, scale_values, time, AssetImportAnimation::INTERP_LINEAR);
		}

		if (skeleton_bone >= 0) {
			Transform xform;
			xform.basis.set_quat_scale(rot, scale);
			xform.origin = pos;

			Transform rest_xform = p_skeleton->get_bone_rest(skeleton_bone);
			xform = rest_xform.affine_inverse() * xform;
			rot = xform.basis.get_rotation_quat();
			scale = xform.basis.get_scale();
			pos = xform.origin;
		}

		rot.normalize();

		animation->track_set_interpolation_type(track_idx, Animation::INTERPOLATION_LINEAR);
		animation->transform_track_insert_key(track_idx, time, pos, rot, scale);

		if (last) { //done this way so a key is always inserted past the end (for proper interpolation)
			break;
		}
		time += increment;
		if (time >= animation->get_length()) {
			last = true;
		}
	}
}

void EditorSceneImporterAssimp::_import_animation(ImportState &state, int p_animation_index, int p_bake_fps) {

	ERR_FAIL_INDEX(p_animation_index, (int)state.assimp_scene->mNumAnimations);

	const aiAnimation *anim = state.assimp_scene->mAnimations[p_animation_index];
	String name = _assimp_anim_string_to_string(anim->mName);
	if (name == String()) {
		name = "Animation " + itos(p_animation_index + 1);
	}

	float ticks_per_second = anim->mTicksPerSecond;

	if (state.assimp_scene->mMetaData != NULL && Math::is_equal_approx(ticks_per_second, 0.0f)) {
		int32_t time_mode = 0;
		state.assimp_scene->mMetaData->Get("TimeMode", time_mode);
		ticks_per_second = _get_fbx_fps(time_mode, state.assimp_scene);
	}

	//?
	//if ((p_path.get_file().get_extension().to_lower() == "glb" || p_path.get_file().get_extension().to_lower() == "gltf") && Math::is_equal_approx(ticks_per_second, 0.0f)) {
	//	ticks_per_second = 1000.0f;
	//}

	if (Math::is_equal_approx(ticks_per_second, 0.0f)) {
		ticks_per_second = 25.0f;
	}

	Ref<Animation> animation;
	animation.instance();
	animation->set_name(name);
	animation->set_length(anim->mDuration / ticks_per_second);

	//regular tracks

	for (size_t i = 0; i < anim->mNumChannels; i++) {
		const aiNodeAnim *track = anim->mChannels[i];
		String node_name = _assimp_get_string(track->mNodeName);
		/*
		if (node_name.find(ASSIMP_FBX_KEY) != -1) {
			String p_track_type = node_name.get_slice(ASSIMP_FBX_KEY, 1);
			if (p_track_type == "_Translation" || p_track_type == "_Rotation" || p_track_type == "_Scaling") {
				continue;
			}
		}
*/
		if (track->mNumRotationKeys == 0 && track->mNumPositionKeys == 0 && track->mNumScalingKeys == 0) {
			continue; //do not bother
		}

		bool is_bone = state.bone_owners.has(node_name);
		NodePath node_path;
		Skeleton *skeleton = NULL;

		if (is_bone) {
			skeleton = state.skeletons[state.bone_owners[node_name]];
			String path = state.root->get_path_to(skeleton);
			path += ":" + node_name;
			node_path = path;
		} else {

			ERR_CONTINUE(!state.node_map.has(node_name));
			Node *node = state.node_map[node_name];
			node_path = state.root->get_path_to(node);
		}

		_insert_animation_track(state, anim, i, p_bake_fps, animation, ticks_per_second, skeleton, node_path, node_name);
	}

	//blend shape tracks

	for (size_t i = 0; i < anim->mNumMorphMeshChannels; i++) {

		const aiMeshMorphAnim *anim_mesh = anim->mMorphMeshChannels[i];

		const String prop_name = _assimp_get_string(anim_mesh->mName);
		const String mesh_name = prop_name.split("*")[0];

		ERR_CONTINUE(prop_name.split("*").size() != 2);

		ERR_CONTINUE(!state.node_map.has(mesh_name));

		const MeshInstance *mesh_instance = Object::cast_to<MeshInstance>(state.node_map[mesh_name]);

		ERR_CONTINUE(mesh_instance == NULL);

		String base_path = state.root->get_path_to(mesh_instance);

		Ref<Mesh> mesh = mesh_instance->get_mesh();
		ERR_CONTINUE(mesh.is_null());

		//add the tracks for this mesh
		int base_track = animation->get_track_count();
		for (int j = 0; j < mesh->get_blend_shape_count(); j++) {

			animation->add_track(Animation::TYPE_VALUE);
			animation->track_set_path(base_track + j, base_path + ":blend_shapes/" + mesh->get_blend_shape_name(j));
		}

		for (size_t k = 0; k < anim_mesh->mNumKeys; k++) {
			for (size_t j = 0; j < anim_mesh->mKeys[k].mNumValuesAndWeights; j++) {

				float t = anim_mesh->mKeys[k].mTime / ticks_per_second;
				float w = anim_mesh->mKeys[k].mWeights[j];

				animation->track_insert_key(base_track + j, t, w);
			}
		}
	}

	if (animation->get_track_count()) {
		state.animation_player->add_animation(name, animation);
	}
}

float EditorSceneImporterAssimp::_get_fbx_fps(int32_t time_mode, const aiScene *p_scene) {
	switch (time_mode) {
		case AssetImportFbx::TIME_MODE_DEFAULT: return 24; //hack
		case AssetImportFbx::TIME_MODE_120: return 120;
		case AssetImportFbx::TIME_MODE_100: return 100;
		case AssetImportFbx::TIME_MODE_60: return 60;
		case AssetImportFbx::TIME_MODE_50: return 50;
		case AssetImportFbx::TIME_MODE_48: return 48;
		case AssetImportFbx::TIME_MODE_30: return 30;
		case AssetImportFbx::TIME_MODE_30_DROP: return 30;
		case AssetImportFbx::TIME_MODE_NTSC_DROP_FRAME: return 29.9700262f;
		case AssetImportFbx::TIME_MODE_NTSC_FULL_FRAME: return 29.9700262f;
		case AssetImportFbx::TIME_MODE_PAL: return 25;
		case AssetImportFbx::TIME_MODE_CINEMA: return 24;
		case AssetImportFbx::TIME_MODE_1000: return 1000;
		case AssetImportFbx::TIME_MODE_CINEMA_ND: return 23.976f;
		case AssetImportFbx::TIME_MODE_CUSTOM:
			int32_t frame_rate;
			p_scene->mMetaData->Get("FrameRate", frame_rate);
			return frame_rate;
	}
	return 0;
}

Transform EditorSceneImporterAssimp::_get_global_assimp_node_transform(const aiNode *p_current_node) {
	aiNode const *current_node = p_current_node;
	Transform xform;
	while (current_node != NULL) {
		xform = _assimp_matrix_transform(current_node->mTransformation) * xform;
		current_node = current_node->mParent;
	}
	return xform;
}

Ref<Texture> EditorSceneImporterAssimp::_load_texture(ImportState &state, String p_path) {
	Vector<String> split_path = p_path.get_basename().split("*");
	if (split_path.size() == 2) {
		size_t texture_idx = split_path[1].to_int();
		ERR_FAIL_COND_V(texture_idx >= state.assimp_scene->mNumTextures, Ref<Texture>());
		aiTexture *tex = state.assimp_scene->mTextures[texture_idx];
		String filename = _assimp_raw_string_to_string(tex->mFilename);
		filename = filename.get_file();
		print_verbose("Open Asset Import: Loading embedded texture " + filename);
		if (tex->mHeight == 0) {
			if (tex->CheckFormat("png")) {
				Ref<Image> img = Image::_png_mem_loader_func((uint8_t *)tex->pcData, tex->mWidth);
				ERR_FAIL_COND_V(img.is_null(), Ref<Texture>());

				Ref<ImageTexture> t;
				t.instance();
				t->create_from_image(img);
				t->set_storage(ImageTexture::STORAGE_COMPRESS_LOSSY);
				return t;
			} else if (tex->CheckFormat("jpg")) {
				Ref<Image> img = Image::_jpg_mem_loader_func((uint8_t *)tex->pcData, tex->mWidth);
				ERR_FAIL_COND_V(img.is_null(), Ref<Texture>());
				Ref<ImageTexture> t;
				t.instance();
				t->create_from_image(img);
				t->set_storage(ImageTexture::STORAGE_COMPRESS_LOSSY);
				return t;
			} else if (tex->CheckFormat("dds")) {
				ERR_EXPLAIN("Open Asset Import: Embedded dds not implemented");
				ERR_FAIL_COND_V(true, Ref<Texture>());
				//Ref<Image> img = Image::_dds_mem_loader_func((uint8_t *)tex->pcData, tex->mWidth);
				//ERR_FAIL_COND_V(img.is_null(), Ref<Texture>());
				//Ref<ImageTexture> t;
				//t.instance();
				//t->create_from_image(img);
				//t->set_storage(ImageTexture::STORAGE_COMPRESS_LOSSY);
				//return t;
			}
		} else {
			Ref<Image> img;
			img.instance();
			PoolByteArray arr;
			uint32_t size = tex->mWidth * tex->mHeight;
			arr.resize(size);
			memcpy(arr.write().ptr(), tex->pcData, size);
			ERR_FAIL_COND_V(arr.size() % 4 != 0, Ref<Texture>());
			//ARGB8888 to RGBA8888
			for (int32_t i = 0; i < arr.size() / 4; i++) {
				arr.write().ptr()[(4 * i) + 3] = arr[(4 * i) + 0];
				arr.write().ptr()[(4 * i) + 0] = arr[(4 * i) + 1];
				arr.write().ptr()[(4 * i) + 1] = arr[(4 * i) + 2];
				arr.write().ptr()[(4 * i) + 2] = arr[(4 * i) + 3];
			}
			img->create(tex->mWidth, tex->mHeight, true, Image::FORMAT_RGBA8, arr);
			ERR_FAIL_COND_V(img.is_null(), Ref<Texture>());

			Ref<ImageTexture> t;
			t.instance();
			t->create_from_image(img);
			t->set_storage(ImageTexture::STORAGE_COMPRESS_LOSSY);
			return t;
		}
		return Ref<Texture>();
	}
	Ref<Texture> p_texture = ResourceLoader::load(p_path, "Texture");
	return p_texture;
}

Ref<Material> EditorSceneImporterAssimp::_generate_material_from_index(ImportState &state, int p_index, bool p_double_sided) {

	ERR_FAIL_INDEX_V(p_index, (int)state.assimp_scene->mNumMaterials, Ref<Material>());

	aiMaterial *ai_material = state.assimp_scene->mMaterials[p_index];
	Ref<SpatialMaterial> mat;
	mat.instance();

	int32_t mat_two_sided = 0;
	if (AI_SUCCESS == ai_material->Get(AI_MATKEY_TWOSIDED, mat_two_sided)) {
		if (mat_two_sided > 0) {
			mat->set_cull_mode(SpatialMaterial::CULL_DISABLED);
		}
	}

	//const String mesh_name = _assimp_get_string(ai_mesh->mName);
	aiString mat_name;
	if (AI_SUCCESS == ai_material->Get(AI_MATKEY_NAME, mat_name)) {
		mat->set_name(_assimp_get_string(mat_name));
	}

	aiTextureType tex_normal = aiTextureType_NORMALS;
	{
		aiString ai_filename = aiString();
		String filename = "";
		aiTextureMapMode map_mode[2];

		if (AI_SUCCESS == ai_material->GetTexture(tex_normal, 0, &ai_filename, NULL, NULL, NULL, NULL, map_mode)) {
			filename = _assimp_raw_string_to_string(ai_filename);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);

				if (texture.is_valid()) {
					_set_texture_mapping_mode(map_mode, texture);
					mat->set_feature(SpatialMaterial::Feature::FEATURE_NORMAL_MAPPING, true);
					mat->set_texture(SpatialMaterial::TEXTURE_NORMAL, texture);
				}
			}
		}
	}

	{
		aiString ai_filename = aiString();
		String filename = "";

		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_NORMAL_TEXTURE, ai_filename)) {
			filename = _assimp_raw_string_to_string(ai_filename);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				if (texture != NULL) {
					mat->set_feature(SpatialMaterial::Feature::FEATURE_NORMAL_MAPPING, true);
					mat->set_texture(SpatialMaterial::TEXTURE_NORMAL, texture);
				}
			}
		}
	}

	aiTextureType tex_emissive = aiTextureType_EMISSIVE;

	if (ai_material->GetTextureCount(tex_emissive) > 0) {

		aiString ai_filename = aiString();
		String filename = "";
		aiTextureMapMode map_mode[2];

		if (AI_SUCCESS == ai_material->GetTexture(tex_emissive, 0, &ai_filename, NULL, NULL, NULL, NULL, map_mode)) {
			filename = _assimp_raw_string_to_string(ai_filename);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				if (texture != NULL) {
					_set_texture_mapping_mode(map_mode, texture);
					mat->set_feature(SpatialMaterial::FEATURE_EMISSION, true);
					mat->set_texture(SpatialMaterial::TEXTURE_EMISSION, texture);
				}
			}
		}
	}

	aiTextureType tex_albedo = aiTextureType_DIFFUSE;
	if (ai_material->GetTextureCount(tex_albedo) > 0) {

		aiString ai_filename = aiString();
		String filename = "";
		aiTextureMapMode map_mode[2];
		if (AI_SUCCESS == ai_material->GetTexture(tex_albedo, 0, &ai_filename, NULL, NULL, NULL, NULL, map_mode)) {
			filename = _assimp_raw_string_to_string(ai_filename);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				if (texture != NULL) {
					if (texture->get_data()->detect_alpha() != Image::ALPHA_NONE) {
						_set_texture_mapping_mode(map_mode, texture);
						mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
						mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
					}
					mat->set_texture(SpatialMaterial::TEXTURE_ALBEDO, texture);
				}
			}
		}
	} else {
		aiColor4D clr_diffuse;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_COLOR_DIFFUSE, clr_diffuse)) {
			if (Math::is_equal_approx(clr_diffuse.a, 1.0f) == false) {
				mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
				mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
			}
			mat->set_albedo(Color(clr_diffuse.r, clr_diffuse.g, clr_diffuse.b, clr_diffuse.a));
		}
	}

	aiString tex_gltf_base_color_path = aiString();
	aiTextureMapMode map_mode[2];
	if (AI_SUCCESS == ai_material->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_TEXTURE, &tex_gltf_base_color_path, NULL, NULL, NULL, NULL, map_mode)) {
		String filename = _assimp_raw_string_to_string(tex_gltf_base_color_path);
		String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
		bool found = false;
		_find_texture_path(state.path, path, found);
		if (found) {
			Ref<Texture> texture = _load_texture(state, path);
			_find_texture_path(state.path, path, found);
			if (texture != NULL) {
				if (texture->get_data()->detect_alpha() == Image::ALPHA_BLEND) {
					_set_texture_mapping_mode(map_mode, texture);
					mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
					mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
				}
				mat->set_texture(SpatialMaterial::TEXTURE_ALBEDO, texture);
			}
		}
	} else {
		aiColor4D pbr_base_color;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_FACTOR, pbr_base_color)) {
			if (Math::is_equal_approx(pbr_base_color.a, 1.0f) == false) {
				mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
				mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
			}
			mat->set_albedo(Color(pbr_base_color.r, pbr_base_color.g, pbr_base_color.b, pbr_base_color.a));
		}
	}
	{
		aiString tex_fbx_pbs_base_color_path = aiString();
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_BASE_COLOR_TEXTURE, tex_fbx_pbs_base_color_path)) {
			String filename = _assimp_raw_string_to_string(tex_fbx_pbs_base_color_path);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				_find_texture_path(state.path, path, found);
				if (texture != NULL) {
					if (texture->get_data()->detect_alpha() == Image::ALPHA_BLEND) {
						mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
						mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
					}
					mat->set_texture(SpatialMaterial::TEXTURE_ALBEDO, texture);
				}
			}
		} else {
			aiColor4D pbr_base_color;
			if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_BASE_COLOR_FACTOR, pbr_base_color)) {
				mat->set_albedo(Color(pbr_base_color.r, pbr_base_color.g, pbr_base_color.b, pbr_base_color.a));
			}
		}

		aiUVTransform pbr_base_color_uv_xform;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_BASE_COLOR_UV_XFORM, pbr_base_color_uv_xform)) {
			mat->set_uv1_offset(Vector3(pbr_base_color_uv_xform.mTranslation.x, pbr_base_color_uv_xform.mTranslation.y, 0.0f));
			mat->set_uv1_scale(Vector3(pbr_base_color_uv_xform.mScaling.x, pbr_base_color_uv_xform.mScaling.y, 1.0f));
		}
	}

	{
		aiString tex_fbx_pbs_normal_path = aiString();
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_NORMAL_TEXTURE, tex_fbx_pbs_normal_path)) {
			String filename = _assimp_raw_string_to_string(tex_fbx_pbs_normal_path);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				_find_texture_path(state.path, path, found);
				if (texture != NULL) {
					mat->set_feature(SpatialMaterial::Feature::FEATURE_NORMAL_MAPPING, true);
					mat->set_texture(SpatialMaterial::TEXTURE_NORMAL, texture);
				}
			}
		}
	}

	if (p_double_sided) {
		mat->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	}

	{
		aiString tex_fbx_stingray_normal_path = aiString();
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_NORMAL_TEXTURE, tex_fbx_stingray_normal_path)) {
			String filename = _assimp_raw_string_to_string(tex_fbx_stingray_normal_path);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				_find_texture_path(state.path, path, found);
				if (texture != NULL) {
					mat->set_feature(SpatialMaterial::Feature::FEATURE_NORMAL_MAPPING, true);
					mat->set_texture(SpatialMaterial::TEXTURE_NORMAL, texture);
				}
			}
		}
	}

	{
		aiString tex_fbx_pbs_base_color_path = aiString();
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_COLOR_TEXTURE, tex_fbx_pbs_base_color_path)) {
			String filename = _assimp_raw_string_to_string(tex_fbx_pbs_base_color_path);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				_find_texture_path(state.path, path, found);
				if (texture != NULL) {
					if (texture->get_data()->detect_alpha() == Image::ALPHA_BLEND) {
						mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
						mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
					}
					mat->set_texture(SpatialMaterial::TEXTURE_ALBEDO, texture);
				}
			}
		} else {
			aiColor4D pbr_base_color;
			if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_BASE_COLOR_FACTOR, pbr_base_color)) {
				mat->set_albedo(Color(pbr_base_color.r, pbr_base_color.g, pbr_base_color.b, pbr_base_color.a));
			}
		}

		aiUVTransform pbr_base_color_uv_xform;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_COLOR_UV_XFORM, pbr_base_color_uv_xform)) {
			mat->set_uv1_offset(Vector3(pbr_base_color_uv_xform.mTranslation.x, pbr_base_color_uv_xform.mTranslation.y, 0.0f));
			mat->set_uv1_scale(Vector3(pbr_base_color_uv_xform.mScaling.x, pbr_base_color_uv_xform.mScaling.y, 1.0f));
		}
	}

	{
		aiString tex_fbx_pbs_emissive_path = aiString();
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_TEXTURE, tex_fbx_pbs_emissive_path)) {
			String filename = _assimp_raw_string_to_string(tex_fbx_pbs_emissive_path);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				_find_texture_path(state.path, path, found);
				if (texture != NULL) {
					if (texture->get_data()->detect_alpha() == Image::ALPHA_BLEND) {
						mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
						mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
					}
					mat->set_texture(SpatialMaterial::TEXTURE_ALBEDO, texture);
				}
			}
		} else {
			aiColor4D pbr_emmissive_color;
			if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_FACTOR, pbr_emmissive_color)) {
				mat->set_emission(Color(pbr_emmissive_color.r, pbr_emmissive_color.g, pbr_emmissive_color.b, pbr_emmissive_color.a));
			}
		}

		real_t pbr_emission_intensity;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_INTENSITY_FACTOR, pbr_emission_intensity)) {
			mat->set_emission_energy(pbr_emission_intensity);
		}
	}

	aiString tex_gltf_pbr_metallicroughness_path;
	if (AI_SUCCESS == ai_material->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, &tex_gltf_pbr_metallicroughness_path)) {
		String filename = _assimp_raw_string_to_string(tex_gltf_pbr_metallicroughness_path);
		String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
		bool found = false;
		_find_texture_path(state.path, path, found);
		if (found) {
			Ref<Texture> texture = _load_texture(state, path);
			if (texture != NULL) {
				mat->set_texture(SpatialMaterial::TEXTURE_METALLIC, texture);
				mat->set_metallic_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_BLUE);
				mat->set_texture(SpatialMaterial::TEXTURE_ROUGHNESS, texture);
				mat->set_roughness_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_GREEN);
			}
		}
	} else {
		float pbr_roughness = 0.0f;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR, pbr_roughness)) {
			mat->set_roughness(pbr_roughness);
		}
		float pbr_metallic = 0.0f;

		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR, pbr_metallic)) {
			mat->set_metallic(pbr_metallic);
		}
	}
	{
		aiString tex_fbx_pbs_metallic_path;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_METALLIC_TEXTURE, tex_fbx_pbs_metallic_path)) {
			String filename = _assimp_raw_string_to_string(tex_fbx_pbs_metallic_path);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				if (texture != NULL) {
					mat->set_texture(SpatialMaterial::TEXTURE_METALLIC, texture);
					mat->set_metallic_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_GRAYSCALE);
				}
			}
		} else {
			float pbr_metallic = 0.0f;
			if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_METALLIC_FACTOR, pbr_metallic)) {
				mat->set_metallic(pbr_metallic);
			}
		}

		aiString tex_fbx_pbs_rough_path;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_ROUGHNESS_TEXTURE, tex_fbx_pbs_rough_path)) {
			String filename = _assimp_raw_string_to_string(tex_fbx_pbs_rough_path);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				if (texture != NULL) {
					mat->set_texture(SpatialMaterial::TEXTURE_ROUGHNESS, texture);
					mat->set_roughness_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_GRAYSCALE);
				}
			}
		} else {
			float pbr_roughness = 0.04f;

			if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_STINGRAY_ROUGHNESS_FACTOR, pbr_roughness)) {
				mat->set_roughness(pbr_roughness);
			}
		}
	}

	{
		aiString tex_fbx_pbs_metallic_path;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_METALNESS_TEXTURE, tex_fbx_pbs_metallic_path)) {
			String filename = _assimp_raw_string_to_string(tex_fbx_pbs_metallic_path);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				if (texture != NULL) {
					mat->set_texture(SpatialMaterial::TEXTURE_METALLIC, texture);
					mat->set_metallic_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_GRAYSCALE);
				}
			}
		} else {
			float pbr_metallic = 0.0f;
			if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_METALNESS_FACTOR, pbr_metallic)) {
				mat->set_metallic(pbr_metallic);
			}
		}

		aiString tex_fbx_pbs_rough_path;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_DIFFUSE_ROUGHNESS_TEXTURE, tex_fbx_pbs_rough_path)) {
			String filename = _assimp_raw_string_to_string(tex_fbx_pbs_rough_path);
			String path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
			bool found = false;
			_find_texture_path(state.path, path, found);
			if (found) {
				Ref<Texture> texture = _load_texture(state, path);
				if (texture != NULL) {
					mat->set_texture(SpatialMaterial::TEXTURE_ROUGHNESS, texture);
					mat->set_roughness_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_GRAYSCALE);
				}
			}
		} else {
			float pbr_roughness = 0.04f;

			if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_DIFFUSE_ROUGHNESS_FACTOR, pbr_roughness)) {
				mat->set_roughness(pbr_roughness);
			}
		}
	}

	return mat;
}

Ref<Mesh> EditorSceneImporterAssimp::_generate_mesh_from_surface_indices(ImportState &state, const Vector<int> &p_surface_indices, Skeleton *p_skeleton, bool p_double_sided_material) {

	Ref<ArrayMesh> mesh;
	mesh.instance();
	bool has_uvs = false;

	for (int i = 0; i < p_surface_indices.size(); i++) {
		const unsigned int mesh_idx = p_surface_indices[i];
		const aiMesh *ai_mesh = state.assimp_scene->mMeshes[mesh_idx];

		Map<uint32_t, Vector<BoneInfo> > vertex_weights;

		if (p_skeleton) {
			for (size_t b = 0; b < ai_mesh->mNumBones; b++) {
				aiBone *bone = ai_mesh->mBones[b];
				String bone_name = _assimp_get_string(bone->mName);
				int bone_index = p_skeleton->find_bone(bone_name);
				ERR_CONTINUE(bone_index == -1); //bone refers to an unexisting index, wtf.

				for (size_t w = 0; w < bone->mNumWeights; w++) {

					aiVertexWeight ai_weights = bone->mWeights[w];

					BoneInfo bi;

					uint32_t vertex_index = ai_weights.mVertexId;
					bi.bone = bone_index;
					bi.weight = ai_weights.mWeight;
					;

					if (!vertex_weights.has(vertex_index)) {
						vertex_weights[vertex_index] = Vector<BoneInfo>();
					}

					vertex_weights[vertex_index].push_back(bi);
				}
			}
		}

		Ref<SurfaceTool> st;
		st.instance();
		st->begin(Mesh::PRIMITIVE_TRIANGLES);

		for (size_t j = 0; j < ai_mesh->mNumVertices; j++) {
			if (ai_mesh->HasTextureCoords(0)) {
				has_uvs = true;
				st->add_uv(Vector2(ai_mesh->mTextureCoords[0][j].x, 1.0f - ai_mesh->mTextureCoords[0][j].y));
			}
			if (ai_mesh->HasTextureCoords(1)) {
				has_uvs = true;
				st->add_uv2(Vector2(ai_mesh->mTextureCoords[1][j].x, 1.0f - ai_mesh->mTextureCoords[1][j].y));
			}
			if (ai_mesh->HasVertexColors(0)) {
				Color color = Color(ai_mesh->mColors[0]->r, ai_mesh->mColors[0]->g, ai_mesh->mColors[0]->b, ai_mesh->mColors[0]->a);
				st->add_color(color);
			}
			if (ai_mesh->mNormals != NULL) {
				const aiVector3D normals = ai_mesh->mNormals[j];
				const Vector3 godot_normal = Vector3(normals.x, normals.y, normals.z);
				st->add_normal(godot_normal);
				if (ai_mesh->HasTangentsAndBitangents()) {
					const aiVector3D tangents = ai_mesh->mTangents[j];
					const Vector3 godot_tangent = Vector3(tangents.x, tangents.y, tangents.z);
					const aiVector3D bitangent = ai_mesh->mBitangents[j];
					const Vector3 godot_bitangent = Vector3(bitangent.x, bitangent.y, bitangent.z);
					float d = godot_normal.cross(godot_tangent).dot(godot_bitangent) > 0.0f ? 1.0f : -1.0f;
					st->add_tangent(Plane(tangents.x, tangents.y, tangents.z, d));
				}
			}

			if (vertex_weights.has(j)) {

				Vector<BoneInfo> bone_info = vertex_weights[j];
				Vector<int> bones;
				bones.resize(bone_info.size());
				Vector<float> weights;
				weights.resize(bone_info.size());
				for (int k = 0; k < bone_info.size(); k++) {
					bones.write[k] = bone_info[k].bone;
					weights.write[k] = bone_info[k].weight;
				}

				st->add_bones(bones);
				st->add_weights(weights);
			}

			const aiVector3D pos = ai_mesh->mVertices[j];
			Vector3 godot_pos = Vector3(pos.x, pos.y, pos.z);
			st->add_vertex(godot_pos);
		}

		for (size_t j = 0; j < ai_mesh->mNumFaces; j++) {
			const aiFace face = ai_mesh->mFaces[j];
			ERR_CONTINUE(face.mNumIndices != 3);
			Vector<size_t> order;
			order.push_back(2);
			order.push_back(1);
			order.push_back(0);
			for (int32_t k = 0; k < order.size(); k++) {
				st->add_index(face.mIndices[order[k]]);
			}
		}
		if (ai_mesh->HasTangentsAndBitangents() == false && has_uvs) {
			st->generate_tangents();
		}

		Ref<Material> material;

		if (!state.material_cache.has(ai_mesh->mMaterialIndex)) {
			material = _generate_material_from_index(state, ai_mesh->mMaterialIndex, p_double_sided_material);
		}

		Array array_mesh = st->commit_to_arrays();
		Array morphs;
		morphs.resize(ai_mesh->mNumAnimMeshes);
		Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;
		Map<uint32_t, String> morph_mesh_idx_names;
		for (size_t j = 0; j < ai_mesh->mNumAnimMeshes; j++) {

			if (i == 0) {
				//only do this the first time
				String ai_anim_mesh_name = _assimp_get_string(ai_mesh->mAnimMeshes[j]->mName);
				mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);
				if (ai_anim_mesh_name.empty()) {
					ai_anim_mesh_name = String("morph_") + itos(j);
				}
				mesh->add_blend_shape(ai_anim_mesh_name);
			}

			Array array_copy;
			array_copy.resize(VisualServer::ARRAY_MAX);

			for (int l = 0; l < VisualServer::ARRAY_MAX; l++) {
				array_copy[l] = array_mesh[l].duplicate(true);
			}

			const size_t num_vertices = ai_mesh->mAnimMeshes[j]->mNumVertices;
			array_copy[Mesh::ARRAY_INDEX] = Variant();
			if (ai_mesh->mAnimMeshes[j]->HasPositions()) {
				PoolVector3Array vertices;
				vertices.resize(num_vertices);
				for (size_t l = 0; l < num_vertices; l++) {
					const aiVector3D ai_pos = ai_mesh->mAnimMeshes[j]->mVertices[l];
					Vector3 position = Vector3(ai_pos.x, ai_pos.y, ai_pos.z);
					vertices.write()[l] = position;
				}
				PoolVector3Array new_vertices = array_copy[VisualServer::ARRAY_VERTEX].duplicate(true);

				for (int32_t l = 0; l < vertices.size(); l++) {
					PoolVector3Array::Write w = new_vertices.write();
					w[l] = vertices[l];
				}
				ERR_CONTINUE(vertices.size() != new_vertices.size());
				array_copy[VisualServer::ARRAY_VERTEX] = new_vertices;
			}

			int32_t color_set = 0;
			if (ai_mesh->mAnimMeshes[j]->HasVertexColors(color_set)) {
				PoolColorArray colors;
				colors.resize(num_vertices);
				for (size_t l = 0; l < num_vertices; l++) {
					const aiColor4D ai_color = ai_mesh->mAnimMeshes[j]->mColors[color_set][l];
					Color color = Color(ai_color.r, ai_color.g, ai_color.b, ai_color.a);
					colors.write()[l] = color;
				}
				PoolColorArray new_colors = array_copy[VisualServer::ARRAY_COLOR].duplicate(true);

				for (int32_t l = 0; l < colors.size(); l++) {
					PoolColorArray::Write w = new_colors.write();
					w[l] = colors[l];
				}
				array_copy[VisualServer::ARRAY_COLOR] = new_colors;
			}

			if (ai_mesh->mAnimMeshes[j]->HasNormals()) {
				PoolVector3Array normals;
				normals.resize(num_vertices);
				for (size_t l = 0; l < num_vertices; l++) {
					const aiVector3D ai_normal = ai_mesh->mAnimMeshes[i]->mNormals[l];
					Vector3 normal = Vector3(ai_normal.x, ai_normal.y, ai_normal.z);
					normals.write()[l] = normal;
				}
				PoolVector3Array new_normals = array_copy[VisualServer::ARRAY_NORMAL].duplicate(true);

				for (int l = 0; l < normals.size(); l++) {
					PoolVector3Array::Write w = new_normals.write();
					w[l] = normals[l];
				}
				array_copy[VisualServer::ARRAY_NORMAL] = new_normals;
			}

			if (ai_mesh->mAnimMeshes[j]->HasTangentsAndBitangents()) {
				PoolColorArray tangents;
				tangents.resize(num_vertices);
				PoolColorArray::Write w = tangents.write();
				for (size_t l = 0; l < num_vertices; l++) {
					_calc_tangent_from_mesh(ai_mesh, j, l, l, w);
				}
				PoolRealArray new_tangents = array_copy[VisualServer::ARRAY_TANGENT].duplicate(true);
				ERR_CONTINUE(new_tangents.size() != tangents.size() * 4);
				for (int32_t l = 0; l < tangents.size(); l++) {
					new_tangents.write()[l + 0] = tangents[l].r;
					new_tangents.write()[l + 1] = tangents[l].g;
					new_tangents.write()[l + 2] = tangents[l].b;
					new_tangents.write()[l + 3] = tangents[l].a;
				}

				array_copy[VisualServer::ARRAY_TANGENT] = new_tangents;
			}

			morphs[j] = array_copy;
		}

		mesh->add_surface_from_arrays(primitive, array_mesh, morphs);
		mesh->surface_set_material(i, material);
		mesh->surface_set_name(i, _assimp_get_string(ai_mesh->mName));
	}

	return mesh;
}

void EditorSceneImporterAssimp::_generate_node(ImportState &state, const aiNode *p_assimp_node, Node *p_parent) {

	Spatial *new_node = NULL;
	String node_name = _assimp_get_string(p_assimp_node->mName);
	Transform node_transform = _assimp_matrix_transform(p_assimp_node->mTransformation);

	if (p_assimp_node->mNumMeshes > 0) {
		/* MESH NODE */
		Ref<Mesh> mesh;
		Skeleton *skeleton = NULL;
		{

			//see if we have mesh cache for this.
			Vector<int> surface_indices;
			for (uint32_t i = 0; i < p_assimp_node->mNumMeshes; i++) {
				int mesh_index = p_assimp_node->mMeshes[i];
				surface_indices.push_back(mesh_index);

				//take the chance and attempt to find the skeleton from the bones
				if (!skeleton) {
					aiMesh *ai_mesh = state.assimp_scene->mMeshes[p_assimp_node->mMeshes[i]];
					for (uint32_t j = 0; j < ai_mesh->mNumBones; j++) {
						aiBone *bone = ai_mesh->mBones[j];
						String bone_name = _assimp_get_string(bone->mName);
						if (state.bone_owners.has(bone_name)) {
							skeleton = state.skeletons[state.bone_owners[bone_name]];
							break;
						}
					}
				}
			}
			surface_indices.sort();
			String mesh_key;
			for (int i = 0; i < surface_indices.size(); i++) {
				if (i > 0) {
					mesh_key += ":";
				}
				mesh_key += itos(surface_indices[i]);
			}

			if (!state.mesh_cache.has(mesh_key)) {
				//adding cache
				aiString cull_mode; //cull is on mesh, which is kind of stupid tbh
				bool double_sided_material = false;
				if (p_assimp_node->mMetaData) {
					p_assimp_node->mMetaData->Get("Culling", cull_mode);
				}
				if (cull_mode.length != 0 && cull_mode == aiString("CullingOff")) {
					double_sided_material = true;
				}

				mesh = _generate_mesh_from_surface_indices(state, surface_indices, skeleton, double_sided_material);
				state.mesh_cache[mesh_key] = mesh;
			}

			mesh = state.mesh_cache[mesh_key];
		}

		MeshInstance *mesh_node = memnew(MeshInstance);
		if (skeleton) {
			state.mesh_skeletons[mesh_node] = skeleton;
		}
		mesh_node->set_mesh(mesh);
		new_node = mesh_node;

	} else if (state.light_cache.has(node_name)) {

		Light *light = NULL;
		aiLight *ai_light = state.assimp_scene->mLights[state.light_cache[node_name]];
		ERR_FAIL_COND(!ai_light);

		if (ai_light->mType == aiLightSource_DIRECTIONAL) {
			light = memnew(DirectionalLight);
			Vector3 dir = Vector3(ai_light->mDirection.y, ai_light->mDirection.x, ai_light->mDirection.z);
			dir.normalize();
			Vector3 pos = Vector3(ai_light->mPosition.x, ai_light->mPosition.y, ai_light->mPosition.z);
			Vector3 up = Vector3(ai_light->mUp.x, ai_light->mUp.y, ai_light->mUp.z);
			up.normalize();

			Transform light_transform;
			light_transform.set_look_at(pos, pos + dir, up);

			node_transform *= light_transform;

		} else if (ai_light->mType == aiLightSource_POINT) {
			light = memnew(OmniLight);
			Vector3 pos = Vector3(ai_light->mPosition.x, ai_light->mPosition.y, ai_light->mPosition.z);
			Transform xform;
			xform.origin = pos;

			node_transform *= xform;

			light->set_transform(xform);

			//light->set_param(Light::PARAM_ATTENUATION, 1);
		} else if (ai_light->mType == aiLightSource_SPOT) {
			light = memnew(SpotLight);

			Vector3 dir = Vector3(ai_light->mDirection.y, ai_light->mDirection.x, ai_light->mDirection.z);
			dir.normalize();
			Vector3 pos = Vector3(ai_light->mPosition.x, ai_light->mPosition.y, ai_light->mPosition.z);
			Vector3 up = Vector3(ai_light->mUp.x, ai_light->mUp.y, ai_light->mUp.z);
			up.normalize();

			Transform light_transform;
			light_transform.set_look_at(pos, pos + dir, up);
			node_transform *= light_transform;

			//light->set_param(Light::PARAM_ATTENUATION, 0.0f);
		}
		ERR_FAIL_COND(light == NULL);
		light->set_color(Color(ai_light->mColorDiffuse.r, ai_light->mColorDiffuse.g, ai_light->mColorDiffuse.b));
		new_node = light;
	} else if (state.camera_cache.has(node_name)) {

		aiCamera *ai_camera = state.assimp_scene->mCameras[state.camera_cache[node_name]];
		ERR_FAIL_COND(!ai_camera);

		Camera *camera = memnew(Camera);

		float near = ai_camera->mClipPlaneNear;
		if (Math::is_equal_approx(near, 0.0f)) {
			near = 0.1f;
		}
		camera->set_perspective(Math::rad2deg(ai_camera->mHorizontalFOV) * 2.0f, near, ai_camera->mClipPlaneFar);

		Vector3 pos = Vector3(ai_camera->mPosition.x, ai_camera->mPosition.y, ai_camera->mPosition.z);
		Vector3 look_at = Vector3(ai_camera->mLookAt.y, ai_camera->mLookAt.x, ai_camera->mLookAt.z).normalized();
		Vector3 up = Vector3(ai_camera->mUp.x, ai_camera->mUp.y, ai_camera->mUp.z);

		Transform xform;
		xform.set_look_at(pos, look_at, up);

		new_node = camera;
	} else if (state.bone_owners.has(node_name)) {

		//have to actually put the skeleton somewhere, you know.
		Skeleton *skeleton = state.skeletons[state.bone_owners[node_name]];
		if (skeleton->get_parent()) {
			//a bone for a skeleton already added..
			//could go downwards here to add meshes children of skeleton bones
			//but let's not support it for now.
			return;
		}
		//restore rest poses to local, now that we know where the skeleton finally is
		Transform skeleton_transform;
		if (p_assimp_node->mParent) {
			skeleton_transform = _get_global_assimp_node_transform(p_assimp_node->mParent);
		}
		for (int i = 0; i < skeleton->get_bone_count(); i++) {
			Transform rest = skeleton_transform.affine_inverse() * skeleton->get_bone_rest(i);
			skeleton->set_bone_rest(i, rest.affine_inverse());
		}

		skeleton->localize_rests();
		node_name = "Skeleton"; //don't use the bone root name
		node_transform = Transform(); //don't transform

		new_node = skeleton;
	} else {
		//generic node
		new_node = memnew(Spatial);
	}

	{

		new_node->set_name(node_name);
		new_node->set_transform(node_transform);
		p_parent->add_child(new_node);
		new_node->set_owner(state.root);
	}

	state.node_map[node_name] = new_node;

	for (size_t i = 0; i < p_assimp_node->mNumChildren; i++) {
		_generate_node(state, p_assimp_node->mChildren[i], new_node);
	}
}

void EditorSceneImporterAssimp::_calc_tangent_from_mesh(const aiMesh *ai_mesh, int i, int tri_index, int index, PoolColorArray::Write &w) {
	const aiVector3D normals = ai_mesh->mAnimMeshes[i]->mNormals[tri_index];
	const Vector3 godot_normal = Vector3(normals.x, normals.y, normals.z);
	const aiVector3D tangent = ai_mesh->mAnimMeshes[i]->mTangents[tri_index];
	const Vector3 godot_tangent = Vector3(tangent.x, tangent.y, tangent.z);
	const aiVector3D bitangent = ai_mesh->mAnimMeshes[i]->mBitangents[tri_index];
	const Vector3 godot_bitangent = Vector3(bitangent.x, bitangent.y, bitangent.z);
	float d = godot_normal.cross(godot_tangent).dot(godot_bitangent) > 0.0f ? 1.0f : -1.0f;
	Color plane_tangent = Color(tangent.x, tangent.y, tangent.z, d);
	w[index] = plane_tangent;
}

void EditorSceneImporterAssimp::_set_texture_mapping_mode(aiTextureMapMode *map_mode, Ref<Texture> texture) {
	ERR_FAIL_COND(map_mode == NULL);
	aiTextureMapMode tex_mode = aiTextureMapMode::aiTextureMapMode_Wrap;
	//for (size_t i = 0; i < 3; i++) {
	tex_mode = map_mode[0];
	//}
	int32_t flags = Texture::FLAGS_DEFAULT;
	if (tex_mode == aiTextureMapMode_Wrap) {
		//Default
	} else if (tex_mode == aiTextureMapMode_Clamp) {
		flags = flags & ~Texture::FLAG_REPEAT;
	} else if (tex_mode == aiTextureMapMode_Mirror) {
		flags = flags | Texture::FLAG_MIRRORED_REPEAT;
	}
	texture->set_flags(flags);
}

void EditorSceneImporterAssimp::_find_texture_path(const String &r_p_path, String &r_path, bool &r_found) {

	_Directory dir;

	List<String> exts;
	ImageLoader::get_recognized_extensions(&exts);

	Vector<String> split_path = r_path.get_basename().split("*");
	if (split_path.size() == 2) {
		r_found = true;
		return;
	}

	if (dir.file_exists(r_p_path.get_base_dir() + r_path.get_file())) {
		r_path = r_p_path.get_base_dir() + r_path.get_file();
		r_found = true;
		return;
	}

	for (int32_t i = 0; i < exts.size(); i++) {
		if (r_found) {
			return;
		}
		if (r_found == false) {
			_find_texture_path(r_p_path, dir, r_path, r_found, "." + exts[i]);
		}
	}
}

void EditorSceneImporterAssimp::_find_texture_path(const String &p_path, _Directory &dir, String &path, bool &found, String extension) {
	String name = path.get_basename() + extension;
	if (dir.file_exists(name)) {
		found = true;
		path = name;
		return;
	}
	String name_ignore_sub_directory = p_path.get_base_dir().plus_file(path.get_file().get_basename()) + extension;
	if (dir.file_exists(name_ignore_sub_directory)) {
		found = true;
		path = name_ignore_sub_directory;
		return;
	}

	String name_find_texture_sub_directory = p_path.get_base_dir() + "/textures/" + path.get_file().get_basename() + extension;
	if (dir.file_exists(name_find_texture_sub_directory)) {
		found = true;
		path = name_find_texture_sub_directory;
		return;
	}
	String name_find_texture_upper_sub_directory = p_path.get_base_dir() + "/Textures/" + path.get_file().get_basename() + extension;
	if (dir.file_exists(name_find_texture_upper_sub_directory)) {
		found = true;
		path = name_find_texture_upper_sub_directory;
		return;
	}
	String name_find_texture_outside_sub_directory = p_path.get_base_dir() + "/../textures/" + path.get_file().get_basename() + extension;
	if (dir.file_exists(name_find_texture_outside_sub_directory)) {
		found = true;
		path = name_find_texture_outside_sub_directory;
		return;
	}

	String name_find_upper_texture_outside_sub_directory = p_path.get_base_dir() + "/../Textures/" + path.get_file().get_basename() + extension;
	if (dir.file_exists(name_find_upper_texture_outside_sub_directory)) {
		found = true;
		path = name_find_upper_texture_outside_sub_directory;
		return;
	}
}

String EditorSceneImporterAssimp::_assimp_get_string(const aiString p_string) const {
	//convert an assimp String to a Godot String
	String name;
	name.parse_utf8(p_string.C_Str() /*,p_string.length*/);
	if (name.find(":") != -1) {
		String replaced_name = name.split(":")[1];
		print_verbose("Replacing " + name + " containing : with " + replaced_name);
		name = replaced_name;
	}

	name = name.replace(".", ""); //can break things, specially bone names

	return name;
}

String EditorSceneImporterAssimp::_assimp_anim_string_to_string(const aiString p_string) const {

	String name;
	name.parse_utf8(p_string.C_Str() /*,p_string.length*/);
	if (name.find(":") != -1) {
		String replaced_name = name.split(":")[1];
		print_verbose("Replacing " + name + " containing : with " + replaced_name);
		name = replaced_name;
	}
	return name;
}

String EditorSceneImporterAssimp::_assimp_raw_string_to_string(const aiString p_string) const {
	String name;
	name.parse_utf8(p_string.C_Str() /*,p_string.length*/);
	return name;
}

Ref<Animation> EditorSceneImporterAssimp::import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps) {
	return Ref<Animation>();
}

const Transform EditorSceneImporterAssimp::_assimp_matrix_transform(const aiMatrix4x4 p_matrix) {
	aiMatrix4x4 matrix = p_matrix;
	Transform xform;
	//xform.set(matrix.a1, matrix.b1, matrix.c1, matrix.a2, matrix.b2, matrix.c2, matrix.a3, matrix.b3, matrix.c3, matrix.a4, matrix.b4, matrix.c4);
	xform.set(matrix.a1, matrix.a2, matrix.a3, matrix.b1, matrix.b2, matrix.b3, matrix.c1, matrix.c2, matrix.c3, matrix.a4, matrix.b4, matrix.c4);
	return xform;
}

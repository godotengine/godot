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

#include "editor_scene_importer_assimp.h"
#include "core/bind/core_bind.h"
#include "core/io/image_loader.h"
#include "editor/editor_file_system.h"
#include "editor/import/resource_importer_scene.h"
#include "editor_settings.h"
#include "import_utils.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/animation/animation_player.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"
#include "scene/resources/surface_tool.h"

#include <assimp/SceneCombiner.h>
#include <assimp/cexport.h>
#include <assimp/cimport.h>
#include <assimp/matrix4x4.h>
#include <assimp/pbrmaterial.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <zutil.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/LogStream.hpp>
#include <assimp/Logger.hpp>
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
								 aiProcess_GlobalScale | // imports models and listens to their file scale for CM to M conversions
								 //aiProcess_FlipUVs |
								 aiProcess_FlipWindingOrder | // very important for culling so that it is done in the correct order.
								 //aiProcess_DropNormals |
								 //aiProcess_GenSmoothNormals |
								 //aiProcess_JoinIdenticalVertices |
								 aiProcess_ImproveCacheLocality |
								 //aiProcess_RemoveRedundantMaterials | // Causes a crash
								 //aiProcess_SplitLargeMeshes |
								 aiProcess_Triangulate |
								 aiProcess_GenUVCoords |
								 //aiProcess_FindDegenerates |
								 //aiProcess_SortByPType |
								 // aiProcess_FindInvalidData |
								 aiProcess_TransformUVCoords |
								 aiProcess_FindInstances |
								 //aiProcess_FixInfacingNormals |
								 aiProcess_ValidateDataStructure |
								 aiProcess_OptimizeMeshes |
								 //aiProcess_OptimizeGraph |
								 //aiProcess_Debone |
								 // aiProcess_EmbedTextures |
								 //aiProcess_SplitByBoneCount |
								 0;
	aiScene *scene = (aiScene *)importer.ReadFile(s_path.c_str(), post_process_Steps);
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

Spatial *EditorSceneImporterAssimp::_generate_scene(const String &p_path, aiScene *scene, const uint32_t p_flags, int p_bake_fps, const int32_t p_max_bone_weights) {
	ERR_FAIL_COND_V(scene == NULL, NULL);

	ImportState state;
	state.path = p_path;
	state.assimp_scene = scene;
	state.max_bone_weights = p_max_bone_weights;
	state.root = memnew(Spatial);
	state.fbx = false;
	state.animation_player = NULL;

	//fill light map cache
	for (size_t l = 0; l < scene->mNumLights; l++) {

		aiLight *ai_light = scene->mLights[l];
		ERR_CONTINUE(ai_light == NULL);
		state.light_cache[AssimpUtils::get_assimp_string(ai_light->mName)] = l;
	}

	//fill camera cache
	for (size_t c = 0; c < scene->mNumCameras; c++) {
		aiCamera *ai_camera = scene->mCameras[c];
		ERR_CONTINUE(ai_camera == NULL);
		state.camera_cache[AssimpUtils::get_assimp_string(ai_camera->mName)] = c;
	}

	if (scene->mRootNode) {

		//generate nodes
		for (uint32_t i = 0; i < scene->mRootNode->mNumChildren; i++) {
			_generate_node(state, NULL, scene->mRootNode->mChildren[i], state.root);
		}

		// finalize skeleton
		for (Map<Skeleton *, const Spatial *>::Element *key_value_pair = state.armature_skeletons.front(); key_value_pair; key_value_pair = key_value_pair->next()) {
			Skeleton *skeleton = key_value_pair->key();
			// convert world to local for skeleton bone rests
			skeleton->localize_rests();
		}

		print_verbose("generating mesh phase from skeletal mesh");
		generate_mesh_phase_from_skeletal_mesh(state);
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

// animation tracks are per bone

void EditorSceneImporterAssimp::_import_animation(ImportState &state, int p_animation_index, int p_bake_fps) {

	ERR_FAIL_INDEX(p_animation_index, (int)state.assimp_scene->mNumAnimations);

	const aiAnimation *anim = state.assimp_scene->mAnimations[p_animation_index];
	String name = AssimpUtils::get_anim_string_from_assimp(anim->mName);
	if (name == String()) {
		name = "Animation " + itos(p_animation_index + 1);
	}

	float ticks_per_second = anim->mTicksPerSecond;

	if (state.assimp_scene->mMetaData != NULL && Math::is_equal_approx(ticks_per_second, 0.0f)) {
		int32_t time_mode = 0;
		state.assimp_scene->mMetaData->Get("TimeMode", time_mode);
		ticks_per_second = AssimpUtils::get_fbx_fps(time_mode, state.assimp_scene);
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
		String node_name = AssimpUtils::get_assimp_string(track->mNodeName);

		if (track->mNumRotationKeys == 0 && track->mNumPositionKeys == 0 && track->mNumScalingKeys == 0) {
			continue; //do not bother
		}

		for (Map<Skeleton *, const Spatial *>::Element *key_value_pair = state.armature_skeletons.front(); key_value_pair; key_value_pair = key_value_pair->next()) {
			Skeleton *skeleton = key_value_pair->key();

			bool is_bone = skeleton->find_bone(node_name) != -1;
			//print_verbose("Bone " + node_name + " is bone? " + (is_bone ? "Yes" : "No"));
			NodePath node_path;

			if (is_bone) {
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
	}

	//blend shape tracks

	for (size_t i = 0; i < anim->mNumMorphMeshChannels; i++) {

		const aiMeshMorphAnim *anim_mesh = anim->mMorphMeshChannels[i];

		const String prop_name = AssimpUtils::get_assimp_string(anim_mesh->mName);
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

//
// Mesh Generation from indices ? why do we need so much mesh code
// [debt needs looked into]
Ref<Mesh> EditorSceneImporterAssimp::_generate_mesh_from_surface_indices(
		ImportState &state,
		const Vector<int> &p_surface_indices,
		const aiNode *assimp_node,
		Skeleton *p_skeleton) {

	Ref<ArrayMesh> mesh;
	mesh.instance();
	bool has_uvs = false;

	Map<String, uint32_t> morph_mesh_string_lookup;

	for (int i = 0; i < p_surface_indices.size(); i++) {
		const unsigned int mesh_idx = p_surface_indices[0];
		const aiMesh *ai_mesh = state.assimp_scene->mMeshes[mesh_idx];
		for (size_t j = 0; j < ai_mesh->mNumAnimMeshes; j++) {

			String ai_anim_mesh_name = AssimpUtils::get_assimp_string(ai_mesh->mAnimMeshes[j]->mName);
			if (!morph_mesh_string_lookup.has(ai_anim_mesh_name)) {
				morph_mesh_string_lookup.insert(ai_anim_mesh_name, j);
				mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);
				if (ai_anim_mesh_name.empty()) {
					ai_anim_mesh_name = String("morph_") + itos(j);
				}
				mesh->add_blend_shape(ai_anim_mesh_name);
			}
		}
	}

	//
	// Process Vertex Weights
	//
	for (int i = 0; i < p_surface_indices.size(); i++) {
		const unsigned int mesh_idx = p_surface_indices[i];
		const aiMesh *ai_mesh = state.assimp_scene->mMeshes[mesh_idx];

		Map<uint32_t, Vector<BoneInfo> > vertex_weights;

		if (p_skeleton) {
			for (size_t b = 0; b < ai_mesh->mNumBones; b++) {
				aiBone *bone = ai_mesh->mBones[b];
				String bone_name = AssimpUtils::get_assimp_string(bone->mName);
				int bone_index = p_skeleton->find_bone(bone_name);
				ERR_CONTINUE(bone_index == -1); //bone refers to an unexisting index, wtf.

				for (size_t w = 0; w < bone->mNumWeights; w++) {

					aiVertexWeight ai_weights = bone->mWeights[w];

					BoneInfo bi;

					uint32_t vertex_index = ai_weights.mVertexId;
					bi.bone = bone_index;
					bi.weight = ai_weights.mWeight;

					if (!vertex_weights.has(vertex_index)) {
						vertex_weights[vertex_index] = Vector<BoneInfo>();
					}

					vertex_weights[vertex_index].push_back(bi);
				}
			}
		}

		//
		// Create mesh from data from assimp
		//

		Ref<SurfaceTool> st;
		st.instance();
		st->begin(Mesh::PRIMITIVE_TRIANGLES);

		for (size_t j = 0; j < ai_mesh->mNumVertices; j++) {

			// Get the texture coordinates if they exist
			if (ai_mesh->HasTextureCoords(0)) {
				has_uvs = true;
				st->add_uv(Vector2(ai_mesh->mTextureCoords[0][j].x, 1.0f - ai_mesh->mTextureCoords[0][j].y));
			}

			if (ai_mesh->HasTextureCoords(1)) {
				has_uvs = true;
				st->add_uv2(Vector2(ai_mesh->mTextureCoords[1][j].x, 1.0f - ai_mesh->mTextureCoords[1][j].y));
			}

			// Assign vertex colors
			if (ai_mesh->HasVertexColors(0)) {
				Color color = Color(ai_mesh->mColors[0]->r, ai_mesh->mColors[0]->g, ai_mesh->mColors[0]->b, ai_mesh->mColors[0]->a);
				st->add_color(color);
			}

			// Work out normal calculations? - this needs work it doesn't work properly on huestos
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

			// We have vertex weights right?
			if (vertex_weights.has(j)) {

				Vector<BoneInfo> bone_info = vertex_weights[j];
				Vector<int> bones;
				bones.resize(bone_info.size());
				Vector<float> weights;
				weights.resize(bone_info.size());

				// todo? do we really need to loop over all bones? - assimp may have helper to find all influences on this vertex.
				for (int k = 0; k < bone_info.size(); k++) {
					bones.write[k] = bone_info[k].bone;
					weights.write[k] = bone_info[k].weight;
				}

				st->add_bones(bones);
				st->add_weights(weights);
			}

			// Assign vertex
			const aiVector3D pos = ai_mesh->mVertices[j];

			// note we must include node offset transform as this is relative to world space not local space.
			Vector3 godot_pos = Vector3(pos.x, pos.y, pos.z);
			st->add_vertex(godot_pos);
		}

		// fire replacement for face handling
		for (size_t j = 0; j < ai_mesh->mNumFaces; j++) {
			const aiFace face = ai_mesh->mFaces[j];
			for (unsigned int k = 0; k < face.mNumIndices; k++) {
				st->add_index(face.mIndices[k]);
			}
		}

		if (ai_mesh->HasTangentsAndBitangents() == false && has_uvs) {
			st->generate_tangents();
		}

		aiMaterial *ai_material = state.assimp_scene->mMaterials[ai_mesh->mMaterialIndex];
		Ref<SpatialMaterial> mat;
		mat.instance();

		int32_t mat_two_sided = 0;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_TWOSIDED, mat_two_sided)) {
			if (mat_two_sided > 0) {
				mat->set_cull_mode(SpatialMaterial::CULL_DISABLED);
			}
		}

		const String mesh_name = AssimpUtils::get_assimp_string(ai_mesh->mName);
		aiString mat_name;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_NAME, mat_name)) {
			mat->set_name(AssimpUtils::get_assimp_string(mat_name));
		}

		// Culling handling for meshes

		// cull all back faces
		mat->set_cull_mode(SpatialMaterial::CULL_BACK);

		// Now process materials
		aiTextureType base_color = aiTextureType_BASE_COLOR;
		{
			String filename, path;
			AssimpImageData image_data;

			if (AssimpUtils::GetAssimpTexture(state, ai_material, base_color, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);

				// anything transparent must be culled
				if (image_data.raw_image->detect_alpha() != Image::ALPHA_NONE) {
					mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
					mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
					mat->set_cull_mode(SpatialMaterial::CULL_DISABLED); // since you can see both sides in transparent mode
				}

				mat->set_texture(SpatialMaterial::TEXTURE_ALBEDO, image_data.texture);
			}
		}

		aiTextureType tex_diffuse = aiTextureType_DIFFUSE;
		{
			String filename, path;
			AssimpImageData image_data;

			if (AssimpUtils::GetAssimpTexture(state, ai_material, tex_diffuse, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);

				// anything transparent must be culled
				if (image_data.raw_image->detect_alpha() != Image::ALPHA_NONE) {
					mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
					mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
					mat->set_cull_mode(SpatialMaterial::CULL_DISABLED); // since you can see both sides in transparent mode
				}

				mat->set_texture(SpatialMaterial::TEXTURE_ALBEDO, image_data.texture);
			}

			aiColor4D clr_diffuse;
			if (AI_SUCCESS == ai_material->Get(AI_MATKEY_COLOR_DIFFUSE, clr_diffuse)) {
				if (Math::is_equal_approx(clr_diffuse.a, 1.0f) == false) {
					mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
					mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
					mat->set_cull_mode(SpatialMaterial::CULL_DISABLED); // since you can see both sides in transparent mode
				}
				mat->set_albedo(Color(clr_diffuse.r, clr_diffuse.g, clr_diffuse.b, clr_diffuse.a));
			}
		}

		aiTextureType tex_normal = aiTextureType_NORMALS;
		{
			String filename, path;
			Ref<ImageTexture> texture;
			AssimpImageData image_data;

			// Process texture normal map
			if (AssimpUtils::GetAssimpTexture(state, ai_material, tex_normal, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);
				mat->set_feature(SpatialMaterial::Feature::FEATURE_NORMAL_MAPPING, true);
				mat->set_texture(SpatialMaterial::TEXTURE_NORMAL, image_data.texture);
			} else {
				aiString texture_path;
				if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_NORMAL_TEXTURE, AI_PROPERTIES, texture_path)) {
					if (AssimpUtils::CreateAssimpTexture(state, texture_path, filename, path, image_data)) {
						mat->set_feature(SpatialMaterial::Feature::FEATURE_NORMAL_MAPPING, true);
						mat->set_texture(SpatialMaterial::TEXTURE_NORMAL, image_data.texture);
					}
				}
			}
		}

		aiTextureType tex_normal_camera = aiTextureType_NORMAL_CAMERA;
		{
			String filename, path;
			Ref<ImageTexture> texture;
			AssimpImageData image_data;

			// Process texture normal map
			if (AssimpUtils::GetAssimpTexture(state, ai_material, tex_normal_camera, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);
				mat->set_feature(SpatialMaterial::Feature::FEATURE_NORMAL_MAPPING, true);
				mat->set_texture(SpatialMaterial::TEXTURE_NORMAL, image_data.texture);
			}
		}

		aiTextureType tex_emission_color = aiTextureType_EMISSION_COLOR;
		{
			String filename, path;
			Ref<ImageTexture> texture;
			AssimpImageData image_data;

			// Process texture normal map
			if (AssimpUtils::GetAssimpTexture(state, ai_material, tex_emission_color, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);
				mat->set_feature(SpatialMaterial::Feature::FEATURE_NORMAL_MAPPING, true);
				mat->set_texture(SpatialMaterial::TEXTURE_NORMAL, image_data.texture);
			}
		}

		aiTextureType tex_metalness = aiTextureType_METALNESS;
		{
			String filename, path;
			Ref<ImageTexture> texture;
			AssimpImageData image_data;

			// Process texture normal map
			if (AssimpUtils::GetAssimpTexture(state, ai_material, tex_metalness, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);
				mat->set_texture(SpatialMaterial::TEXTURE_METALLIC, image_data.texture);
			}
		}

		aiTextureType tex_roughness = aiTextureType_DIFFUSE_ROUGHNESS;
		{
			String filename, path;
			Ref<ImageTexture> texture;
			AssimpImageData image_data;

			// Process texture normal map
			if (AssimpUtils::GetAssimpTexture(state, ai_material, tex_roughness, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);
				mat->set_texture(SpatialMaterial::TEXTURE_ROUGHNESS, image_data.texture);
			}
		}

		aiTextureType tex_emissive = aiTextureType_EMISSIVE;
		{
			String filename = "";
			String path = "";
			Ref<Image> texture;
			AssimpImageData image_data;

			if (AssimpUtils::GetAssimpTexture(state, ai_material, tex_emissive, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);
				mat->set_feature(SpatialMaterial::FEATURE_EMISSION, true);
				mat->set_texture(SpatialMaterial::TEXTURE_EMISSION, image_data.texture);
			} else {
				// Process emission textures
				aiString texture_emissive_path;
				if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_EMISSION_TEXTURE, AI_PROPERTIES, texture_emissive_path)) {
					if (AssimpUtils::CreateAssimpTexture(state, texture_emissive_path, filename, path, image_data)) {
						mat->set_feature(SpatialMaterial::FEATURE_EMISSION, true);
						mat->set_texture(SpatialMaterial::TEXTURE_EMISSION, image_data.texture);
					}
				} else {
					float pbr_emission = 0.0f;
					if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_MAYA_EMISSIVE_FACTOR, AI_NULL, pbr_emission)) {
						mat->set_emission(Color(pbr_emission, pbr_emission, pbr_emission, 1.0f));
					}
				}
			}
		}

		aiTextureType tex_specular = aiTextureType_SPECULAR;
		{
			String filename, path;
			Ref<ImageTexture> texture;
			AssimpImageData image_data;

			// Process texture normal map
			if (AssimpUtils::GetAssimpTexture(state, ai_material, tex_specular, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);
				mat->set_texture(SpatialMaterial::TEXTURE_METALLIC, image_data.texture);
			}
		}

		aiTextureType tex_ao_map = aiTextureType_AMBIENT_OCCLUSION;
		{
			String filename, path;
			Ref<ImageTexture> texture;
			AssimpImageData image_data;

			// Process texture normal map
			if (AssimpUtils::GetAssimpTexture(state, ai_material, tex_ao_map, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);
				mat->set_feature(SpatialMaterial::FEATURE_AMBIENT_OCCLUSION, true);
				mat->set_texture(SpatialMaterial::TEXTURE_AMBIENT_OCCLUSION, image_data.texture);
			}
		}

		Array array_mesh = st->commit_to_arrays();
		Array morphs;
		morphs.resize(ai_mesh->mNumAnimMeshes);
		Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;

		for (size_t j = 0; j < ai_mesh->mNumAnimMeshes; j++) {

			String ai_anim_mesh_name = AssimpUtils::get_assimp_string(ai_mesh->mAnimMeshes[j]->mName);

			if (ai_anim_mesh_name.empty()) {
				ai_anim_mesh_name = String("morph_") + itos(j);
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
				ERR_CONTINUE(vertices.size() != new_vertices.size());
				for (int32_t l = 0; l < new_vertices.size(); l++) {
					PoolVector3Array::Write w = new_vertices.write();
					w[l] = vertices[l];
				}
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
				ERR_CONTINUE(colors.size() != new_colors.size());
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
					const aiVector3D ai_normal = ai_mesh->mAnimMeshes[j]->mNormals[l];
					Vector3 normal = Vector3(ai_normal.x, ai_normal.y, ai_normal.z);
					normals.write()[l] = normal;
				}
				PoolVector3Array new_normals = array_copy[VisualServer::ARRAY_NORMAL].duplicate(true);
				ERR_CONTINUE(normals.size() != new_normals.size());
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
					AssimpUtils::calc_tangent_from_mesh(ai_mesh, j, l, l, w);
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
		mesh->surface_set_material(i, mat);
		mesh->surface_set_name(i, AssimpUtils::get_assimp_string(ai_mesh->mName));
	}

	return mesh;
}

/* to be moved into assimp */
aiBone *get_bone_by_name(const aiScene *scene, aiString bone_name) {
	for (unsigned int mesh_id = 0; mesh_id < scene->mNumMeshes; ++mesh_id) {
		aiMesh *mesh = scene->mMeshes[mesh_id];

		// iterate over all the bones on the mesh for this node only!
		for (unsigned int boneIndex = 0; boneIndex < mesh->mNumBones; boneIndex++) {

			aiBone *bone = mesh->mBones[boneIndex];
			if (bone->mName == bone_name) {
				printf("matched bone by name: %s\n", bone->mName.C_Str());
				return bone;
			}
		}
	}

	return NULL;
}

/**
 * Create a new mesh for the node supplied
 */
void EditorSceneImporterAssimp::create_mesh(ImportState &state, const aiNode *assimp_node, const String &node_name, Node *current_node, Node *parent_node, Transform node_transform) {
	/* MESH NODE */
	Ref<Mesh> mesh;
	Skeleton *skeleton = NULL;
	// see if we have mesh cache for this.
	Vector<int> surface_indices;
	for (uint32_t i = 0; i < assimp_node->mNumMeshes; i++) {
		int mesh_index = assimp_node->mMeshes[i];
		aiMesh *ai_mesh = state.assimp_scene->mMeshes[assimp_node->mMeshes[i]];

		// Map<aiBone*, Skeleton*> // this is what we need
		if (ai_mesh->mNumBones > 0) {
			// we only need the first bone to retrieve the skeleton
			const aiBone *first = ai_mesh->mBones[0];

			ERR_FAIL_COND(first == NULL);

			Map<const aiBone *, Skeleton *>::Element *match = state.bone_to_skeleton_lookup.find(first);
			if (match != NULL) {
				skeleton = match->value();

				if (skeleton == NULL) {
					print_error("failed to find bone skeleton for bone: " + AssimpUtils::get_assimp_string(first->mName));
				} else {
					print_verbose("successfully found skeleton for first bone on mesh, can properly handle animations now!");
				}
				// I really need the skeleton and bone to be known as this is something flaky in model exporters.
				ERR_FAIL_COND(skeleton == NULL); // should not happen if bone was successfully created in previous step.
			}
		}
		surface_indices.push_back(mesh_index);
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
		mesh = _generate_mesh_from_surface_indices(state, surface_indices, assimp_node, skeleton);
		state.mesh_cache[mesh_key] = mesh;
	}

	//Transform transform = recursive_state.node_transform;

	// we must unfortunately overwrite mesh and skeleton transform with armature data
	if (skeleton != NULL) {
		print_verbose("Applying mesh and skeleton to armature");
		// required for blender, maya etc
		Map<Skeleton *, const Spatial *>::Element *match = state.armature_skeletons.find(skeleton);
		node_transform = match->value()->get_transform();
	}

	MeshInstance *mesh_node = memnew(MeshInstance);
	mesh = state.mesh_cache[mesh_key];
	mesh_node->set_mesh(mesh);

	attach_new_node(state,
			mesh_node,
			assimp_node,
			parent_node,
			node_name,
			node_transform);

	// set this once and for all
	if (skeleton != NULL) {
		// root must be informed of its new child
		parent_node->add_child(skeleton);

		// owner must be set after adding to tree
		skeleton->set_owner(state.root);

		skeleton->set_transform(node_transform);

		// must be done after added to tree
		mesh_node->set_skeleton_path(mesh_node->get_path_to(skeleton));
	}
}

/** generate_mesh_phase_from_skeletal_mesh
 * This must be executed after generate_nodes because the skeleton doesn't exist until that has completed the first pass
 */
void EditorSceneImporterAssimp::generate_mesh_phase_from_skeletal_mesh(ImportState &state) {
	// prevent more than one skeleton existing per mesh
	// * multiple root bones have this
	// * this simply filters the node out if it has already been added then references the skeleton so we know the actual skeleton for this node
	for (Map<const aiNode *, const Node *>::Element *key_value_pair = state.assimp_node_map.front(); key_value_pair; key_value_pair = key_value_pair->next()) {
		const aiNode *assimp_node = key_value_pair->key();
		Node *current_node = (Node *)key_value_pair->value();
		Node *parent_node = current_node->get_parent();

		ERR_CONTINUE(assimp_node == NULL);
		ERR_CONTINUE(parent_node == NULL);

		String node_name = AssimpUtils::get_assimp_string(assimp_node->mName);
		Transform node_transform = AssimpUtils::assimp_matrix_transform(assimp_node->mTransformation);

		if (assimp_node->mNumMeshes > 0) {
			create_mesh(state, assimp_node, node_name, current_node, parent_node, node_transform);
		}
	}
}

/** 
 * attach_new_node
 * configures node, assigns parent node
**/
void EditorSceneImporterAssimp::attach_new_node(ImportState &state, Spatial *new_node, const aiNode *node, Node *parent_node, String Name, Transform &transform) {
	ERR_FAIL_COND(new_node == NULL);
	ERR_FAIL_COND(node == NULL);
	ERR_FAIL_COND(parent_node == NULL);
	ERR_FAIL_COND(state.root == NULL);

	// assign properties to new godot note
	new_node->set_name(Name);
	new_node->set_transform(transform);

	// add element as child to parent
	parent_node->add_child(new_node);

	// owner must be set after
	new_node->set_owner(state.root);

	// cache node mapping results by name and then by aiNode*
	state.node_map[Name] = new_node;
	state.assimp_node_map[node] = new_node;
}

/**
 * Create a light for the scene
 * Automatically caches lights for lookup later
 */
void EditorSceneImporterAssimp::create_light(ImportState &state, RecursiveState &recursive_state) {
	Light *light = NULL;
	aiLight *ai_light = state.assimp_scene->mLights[state.light_cache[recursive_state.node_name]];
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

		recursive_state.node_transform *= light_transform;

	} else if (ai_light->mType == aiLightSource_POINT) {
		light = memnew(OmniLight);
		Vector3 pos = Vector3(ai_light->mPosition.x, ai_light->mPosition.y, ai_light->mPosition.z);
		Transform xform;
		xform.origin = pos;

		recursive_state.node_transform *= xform;

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
		recursive_state.node_transform *= light_transform;

		//light->set_param(Light::PARAM_ATTENUATION, 0.0f);
	}
	ERR_FAIL_COND(light == NULL);

	light->set_color(Color(ai_light->mColorDiffuse.r, ai_light->mColorDiffuse.g, ai_light->mColorDiffuse.b));
	recursive_state.new_node = light;

	attach_new_node(state,
			recursive_state.new_node,
			recursive_state.assimp_node,
			recursive_state.parent_node,
			recursive_state.node_name,
			recursive_state.node_transform);
}

/**
 * Create camera for the scene
 */
void EditorSceneImporterAssimp::create_camera(ImportState &state, RecursiveState &recursive_state) {
	aiCamera *ai_camera = state.assimp_scene->mCameras[state.camera_cache[recursive_state.node_name]];
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

	recursive_state.new_node = camera;

	attach_new_node(state,
			recursive_state.new_node,
			recursive_state.assimp_node,
			recursive_state.parent_node,
			recursive_state.node_name,
			recursive_state.node_transform);
}

/**
 * Create Bone 
 * Create a bone in the scene
 */
void EditorSceneImporterAssimp::create_bone(ImportState &state, RecursiveState &recursive_state) {
	// for each armature node we must make a new skeleton but ensure it
	// has a bone in the child to ensure we don't make too many
	// the reason you must do this is because a skeleton exists per mesh?
	// and duplicate bone names are very bad for determining what is going on.
	aiBone *parent_bone_assimp = get_bone_by_name(state.assimp_scene, recursive_state.assimp_node->mParent->mName);

	// set to true when you want to use skeleton reference from cache.
	bool do_not_create_armature = false;

	// prevent more than one skeleton existing per mesh
	// * multiple root bones have this
	// * this simply filters the node out if it has already been added then references the skeleton so we know the actual skeleton for this node
	for (Map<Skeleton *, const Spatial *>::Element *key_value_pair = state.armature_skeletons.front(); key_value_pair; key_value_pair = key_value_pair->next()) {
		if (key_value_pair->value() == recursive_state.parent_node) {
			// apply the skeleton for this mesh
			recursive_state.skeleton = key_value_pair->key();

			// force this off
			do_not_create_armature = true;
		}
	}

	// check if parent was a bone
	// if parent was not a bone this is the first bone.
	// therefore parent is the 'armature'?
	// also for multi root bone support make sure we don't already have the skeleton cached.
	// if we do we must merge them - as this is all godot supports right now.
	if (!parent_bone_assimp && recursive_state.skeleton == NULL && !do_not_create_armature) {
		// create new skeleton on the root.
		recursive_state.skeleton = memnew(Skeleton);

		ERR_FAIL_COND(state.root == NULL);
		ERR_FAIL_COND(recursive_state.skeleton == NULL);

		print_verbose("Parent armature node is called " + recursive_state.parent_node->get_name());
		// store root node for this skeleton / used in animation playback and bone detection.

		state.armature_skeletons.insert(recursive_state.skeleton, Object::cast_to<Spatial>(recursive_state.parent_node));

		//skeleton->set_use_bones_in_world_transform(true);
		print_verbose("Created new FBX skeleton for armature node");
	}

	ERR_FAIL_COND_MSG(recursive_state.skeleton == NULL, "Mesh has invalid armature detection - report this");

	// this transform is a bone
	recursive_state.skeleton->add_bone(recursive_state.node_name);

	//ERR_FAIL_COND(recursive_state.skeleton->get_name() == "");
	print_verbose("Bone added to lookup: " + AssimpUtils::get_assimp_string(recursive_state.bone->mName));
	print_verbose("Skeleton attached to: " + recursive_state.skeleton->get_name());
	// make sure to write the bone lookup inverse so we can retrieve the mesh for this bone later
	state.bone_to_skeleton_lookup.insert(recursive_state.bone, recursive_state.skeleton);

	Transform xform = AssimpUtils::assimp_matrix_transform(recursive_state.bone->mOffsetMatrix);
	recursive_state.skeleton->set_bone_rest(recursive_state.skeleton->get_bone_count() - 1, xform.affine_inverse());

	// get parent node of assimp node
	const aiNode *parent_node_assimp = recursive_state.assimp_node->mParent;

	// ensure we have a parent
	if (parent_node_assimp != NULL) {
		int parent_bone_id = recursive_state.skeleton->find_bone(AssimpUtils::get_assimp_string(parent_node_assimp->mName));
		int current_bone_id = recursive_state.skeleton->find_bone(recursive_state.node_name);
		print_verbose("Parent bone id " + itos(parent_bone_id) + " current bone id" + itos(current_bone_id));
		print_verbose("Bone debug: " + AssimpUtils::get_assimp_string(parent_node_assimp->mName));
		recursive_state.skeleton->set_bone_parent(current_bone_id, parent_bone_id);
	}
}

/**
 * Generate node
 * Recursive call to iterate over all nodes
 */
void EditorSceneImporterAssimp::_generate_node(
		ImportState &state,
		Skeleton *skeleton,
		const aiNode *assimp_node, Node *parent_node) {

	// sanity check
	ERR_FAIL_COND(state.root == NULL);
	ERR_FAIL_COND(state.assimp_scene == NULL);
	ERR_FAIL_COND(assimp_node == NULL);
	ERR_FAIL_COND(parent_node == NULL);

	Spatial *new_node = NULL;
	String node_name = AssimpUtils::get_assimp_string(assimp_node->mName);
	Transform node_transform = AssimpUtils::assimp_matrix_transform(assimp_node->mTransformation);

	// can safely return null - is this node a bone?
	aiBone *bone = get_bone_by_name(state.assimp_scene, assimp_node->mName);

	// out arguments helper - for pushing state down into creation functions
	RecursiveState recursive_state(node_transform, skeleton, new_node, node_name, assimp_node, parent_node, bone);

	// Creation code
	if (state.light_cache.has(node_name)) {
		create_light(state, recursive_state);
	} else if (state.camera_cache.has(node_name)) {
		create_camera(state, recursive_state);
	} else if (bone != NULL) {
		create_bone(state, recursive_state);
	} else {
		//generic node
		recursive_state.new_node = memnew(Spatial);
		attach_new_node(state,
				recursive_state.new_node,
				recursive_state.assimp_node,
				recursive_state.parent_node,
				recursive_state.node_name,
				recursive_state.node_transform);
	}

	// recurse into all child elements
	for (size_t i = 0; i < recursive_state.assimp_node->mNumChildren; i++) {
		_generate_node(state, recursive_state.skeleton, recursive_state.assimp_node->mChildren[i],
				recursive_state.new_node != NULL ? recursive_state.new_node : recursive_state.parent_node);
	}
}
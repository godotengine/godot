/*************************************************************************/
/*  editor_scene_importer_assimp.cpp                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/io/image_loader.h"
#include "editor/import/resource_importer_scene.h"
#include "import_utils.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"
#include "scene/resources/surface_tool.h"

#include <assimp/matrix4x4.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/LogStream.hpp>

// move into assimp
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

	return nullptr;
}

void EditorSceneImporterAssimp::get_extensions(List<String> *r_extensions) const {
	const String import_setting_string = "filesystem/import/open_asset_import/";

	Map<String, ImportFormat> import_format;
	{
		Vector<String> exts;
		exts.push_back("fbx");
		ImportFormat import = { exts, true };
		import_format.insert("fbx", import);
	}
	for (Map<String, ImportFormat>::Element *E = import_format.front(); E; E = E->next()) {
		_register_project_setting_import(E->key(), import_setting_string, E->get().extensions, r_extensions,
				E->get().is_default);
	}
}

void EditorSceneImporterAssimp::_register_project_setting_import(const String generic, const String import_setting_string,
		const Vector<String> &exts, List<String> *r_extensions,
		const bool p_enabled) const {
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

Node *EditorSceneImporterAssimp::import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps,
		List<String> *r_missing_deps, Error *r_err) {
	Assimp::Importer importer;
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
								 aiProcess_GlobalScale |
								 // imports models and listens to their file scale for CM to M conversions
								 //aiProcess_FlipUVs |
								 aiProcess_FlipWindingOrder |
								 // very important for culling so that it is done in the correct order.
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
								 //aiProcess_ValidateDataStructure |
								 aiProcess_OptimizeMeshes |
								 aiProcess_PopulateArmatureData |
								 //aiProcess_OptimizeGraph |
								 //aiProcess_Debone |
								 // aiProcess_EmbedTextures |
								 //aiProcess_SplitByBoneCount |
								 0;
	String g_path = ProjectSettings::get_singleton()->globalize_path(p_path);
	aiScene *scene = (aiScene *)importer.ReadFile(g_path.utf8().ptr(), post_process_Steps);

	ERR_FAIL_COND_V_MSG(scene == nullptr, nullptr, String("Open Asset Import failed to open: ") + String(importer.GetErrorString()));

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

		return 0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
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
		ERR_FAIL_COND_V_MSG(!a.is_normalized(), Quat(), "The quaternion \"a\" must be normalized.");
		ERR_FAIL_COND_V_MSG(!b.is_normalized(), Quat(), "The quaternion \"b\" must be normalized.");

		return a.slerp(b, c).normalized();
	}

	Quat catmull_rom(const Quat &p0, const Quat &p1, const Quat &p2, const Quat &p3, float c) {
		ERR_FAIL_COND_V_MSG(!p1.is_normalized(), Quat(), "The quaternion \"p1\" must be normalized.");
		ERR_FAIL_COND_V_MSG(!p2.is_normalized(), Quat(), "The quaternion \"p2\" must be normalized.");

		return p1.slerp(p2, c).normalized();
	}

	Quat bezier(Quat start, Quat control_1, Quat control_2, Quat end, float t) {
		ERR_FAIL_COND_V_MSG(!start.is_normalized(), Quat(), "The start quaternion must be normalized.");
		ERR_FAIL_COND_V_MSG(!end.is_normalized(), Quat(), "The end quaternion must be normalized.");

		return start.slerp(end, t).normalized();
	}
};

template <class T>
T EditorSceneImporterAssimp::_interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time,
		AssetImportAnimation::Interpolation p_interp) {
	//could use binary search, worth it?
	int idx = -1;
	for (int i = 0; i < p_times.size(); i++) {
		if (p_times[i] > p_time) {
			break;
		}
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

aiBone *EditorSceneImporterAssimp::get_bone_from_stack(ImportState &state, aiString name) {
	List<aiBone *>::Element *iter;
	aiBone *bone = nullptr;
	for (iter = state.bone_stack.front(); iter; iter = iter->next()) {
		bone = (aiBone *)iter->get();

		if (bone && bone->mName == name) {
			state.bone_stack.erase(bone);
			return bone;
		}
	}

	return nullptr;
}

Node3D *
EditorSceneImporterAssimp::_generate_scene(const String &p_path, aiScene *scene, const uint32_t p_flags, int p_bake_fps,
		const int32_t p_max_bone_weights) {
	ERR_FAIL_COND_V(scene == nullptr, nullptr);

	ImportState state;
	state.path = p_path;
	state.assimp_scene = scene;
	state.max_bone_weights = p_max_bone_weights;
	state.animation_player = nullptr;
	state.import_flags = p_flags;

	// populate light map
	for (unsigned int l = 0; l < scene->mNumLights; l++) {
		aiLight *ai_light = scene->mLights[l];
		ERR_CONTINUE(ai_light == nullptr);
		state.light_cache[AssimpUtils::get_assimp_string(ai_light->mName)] = l;
	}

	// fill camera cache
	for (unsigned int c = 0; c < scene->mNumCameras; c++) {
		aiCamera *ai_camera = scene->mCameras[c];
		ERR_CONTINUE(ai_camera == nullptr);
		state.camera_cache[AssimpUtils::get_assimp_string(ai_camera->mName)] = c;
	}

	if (scene->mRootNode) {
		state.nodes.push_back(scene->mRootNode);

		// make flat node tree - in order to make processing deterministic
		for (unsigned int i = 0; i < scene->mRootNode->mNumChildren; i++) {
			_generate_node(state, scene->mRootNode->mChildren[i]);
		}

		RegenerateBoneStack(state);

		Node *last_valid_parent = nullptr;

		List<const aiNode *>::Element *iter;
		for (iter = state.nodes.front(); iter; iter = iter->next()) {
			const aiNode *element_assimp_node = iter->get();
			const aiNode *parent_assimp_node = element_assimp_node->mParent;

			String node_name = AssimpUtils::get_assimp_string(element_assimp_node->mName);
			//print_verbose("node: " + node_name);

			Node3D *spatial = nullptr;
			Transform transform = AssimpUtils::assimp_matrix_transform(element_assimp_node->mTransformation);

			// retrieve this node bone
			aiBone *bone = get_bone_from_stack(state, element_assimp_node->mName);

			if (state.light_cache.has(node_name)) {
				spatial = create_light(state, node_name, transform);
			} else if (state.camera_cache.has(node_name)) {
				spatial = create_camera(state, node_name, transform);
			} else if (state.armature_nodes.find(element_assimp_node)) {
				// create skeleton
				print_verbose("Making skeleton: " + node_name);
				Skeleton3D *skeleton = memnew(Skeleton3D);
				spatial = skeleton;
				if (!state.armature_skeletons.has(element_assimp_node)) {
					state.armature_skeletons.insert(element_assimp_node, skeleton);
				}
			} else if (bone != nullptr) {
				continue;
			} else {
				spatial = memnew(Node3D);
			}

			ERR_CONTINUE_MSG(spatial == nullptr, "FBX Import - are we out of ram?");
			// we on purpose set the transform and name after creating the node.

			spatial->set_name(node_name);
			spatial->set_global_transform(transform);

			// first element is root
			if (iter == state.nodes.front()) {
				state.root = spatial;
			}

			// flat node map parent lookup tool
			state.flat_node_map.insert(element_assimp_node, spatial);

			Map<const aiNode *, Node3D *>::Element *parent_lookup = state.flat_node_map.find(parent_assimp_node);

			// note: this always fails on the root node :) keep that in mind this is by design
			if (parent_lookup) {
				Node3D *parent_node = parent_lookup->value();

				ERR_FAIL_COND_V_MSG(parent_node == nullptr, state.root,
						"Parent node invalid even though lookup successful, out of ram?");

				if (spatial != state.root) {
					parent_node->add_child(spatial);
					spatial->set_owner(state.root);
				} else {
					// required - think about it root never has a parent yet is valid, anything else without a parent is not valid.
				}
			} else if (spatial != state.root) {
				// if the ainode is not in the tree
				// parent it to the last good parent found
				if (last_valid_parent) {
					last_valid_parent->add_child(spatial);
					spatial->set_owner(state.root);
				} else {
					// this is a serious error?
					memdelete(spatial);
				}
			}

			// update last valid parent
			last_valid_parent = spatial;
		}
		print_verbose("node counts: " + itos(state.nodes.size()));

		// make clean bone stack
		RegenerateBoneStack(state);

		print_verbose("generating godot bone data");

		print_verbose("Godot bone stack count: " + itos(state.bone_stack.size()));

		// This is a list of bones, duplicates are from other meshes and must be dealt with properly
		for (List<aiBone *>::Element *element = state.bone_stack.front(); element; element = element->next()) {
			aiBone *bone = element->get();

			ERR_CONTINUE_MSG(!bone, "invalid bone read from assimp?");

			// Utilities for armature lookup - for now only FBX makes these
			aiNode *armature_for_bone = bone->mArmature;

			// Utilities for bone node lookup - for now only FBX makes these
			aiNode *bone_node = bone->mNode;
			aiNode *parent_node = bone_node->mParent;

			String bone_name = AssimpUtils::get_anim_string_from_assimp(bone->mName);
			ERR_CONTINUE_MSG(armature_for_bone == nullptr, "Armature for bone invalid: " + bone_name);
			Skeleton3D *skeleton = state.armature_skeletons[armature_for_bone];

			state.skeleton_bone_map[bone] = skeleton;

			if (bone_name.empty()) {
				bone_name = "untitled_bone_name";
				WARN_PRINT("Untitled bone name detected... report with file please");
			}

			// todo: this is where skin support goes
			if (skeleton && skeleton->find_bone(bone_name) == -1) {
				print_verbose("[Godot Glue] Imported bone" + bone_name);
				int boneIdx = skeleton->get_bone_count();

				Transform pform = AssimpUtils::assimp_matrix_transform(bone->mNode->mTransformation);
				skeleton->add_bone(bone_name);
				skeleton->set_bone_rest(boneIdx, pform);

				if (parent_node != nullptr) {
					int parent_bone_id = skeleton->find_bone(AssimpUtils::get_anim_string_from_assimp(parent_node->mName));
					int current_bone_id = boneIdx;
					skeleton->set_bone_parent(current_bone_id, parent_bone_id);
				}
			}
		}

		print_verbose("generating mesh phase from skeletal mesh");

		List<Node3D *> cleanup_template_nodes;

		for (Map<const aiNode *, Node3D *>::Element *key_value_pair = state.flat_node_map.front(); key_value_pair; key_value_pair = key_value_pair->next()) {
			const aiNode *assimp_node = key_value_pair->key();
			Node3D *mesh_template = key_value_pair->value();

			ERR_CONTINUE(assimp_node == nullptr);
			ERR_CONTINUE(mesh_template == nullptr);

			Node *parent_node = mesh_template->get_parent();

			if (mesh_template == state.root) {
				continue;
			}

			if (parent_node == nullptr) {
				print_error("Found invalid parent node!");
				continue; // root node
			}

			String node_name = AssimpUtils::get_assimp_string(assimp_node->mName);
			Transform node_transform = AssimpUtils::assimp_matrix_transform(assimp_node->mTransformation);

			if (assimp_node->mNumMeshes > 0) {
				MeshInstance3D *mesh = create_mesh(state, assimp_node, node_name, parent_node, node_transform);
				if (mesh) {
					parent_node->remove_child(mesh_template);

					// re-parent children
					List<Node *> children;
					// re-parent all children to new node
					// note: since get_child_count will change during execution we must build a list first to be safe.
					for (int childId = 0; childId < mesh_template->get_child_count(); childId++) {
						// get child
						Node *child = mesh_template->get_child(childId);
						children.push_back(child);
					}

					for (List<Node *>::Element *element = children.front(); element; element = element->next()) {
						// reparent the children to the real mesh node.
						mesh_template->remove_child(element->get());
						mesh->add_child(element->get());
						element->get()->set_owner(state.root);
					}

					// update mesh in list so that each mesh node is available
					// this makes the template unavailable which is the desired behaviour
					state.flat_node_map[assimp_node] = mesh;

					cleanup_template_nodes.push_back(mesh_template);

					// clean up this list we don't need it
					children.clear();
				}
			}
		}

		for (List<Node3D *>::Element *element = cleanup_template_nodes.front(); element; element = element->next()) {
			if (element->get()) {
				memdelete(element->get());
			}
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

	//
	// Cleanup operations
	//

	state.mesh_cache.clear();
	state.material_cache.clear();
	state.light_cache.clear();
	state.camera_cache.clear();
	state.assimp_node_map.clear();
	state.path_to_image_cache.clear();
	state.nodes.clear();
	state.flat_node_map.clear();
	state.armature_skeletons.clear();
	state.bone_stack.clear();
	return state.root;
}

void EditorSceneImporterAssimp::_insert_animation_track(ImportState &scene, const aiAnimation *assimp_anim, int track_id,
		int anim_fps, Ref<Animation> animation, float ticks_per_second,
		Skeleton3D *skeleton, const NodePath &node_path,
		const String &node_name, aiBone *track_bone) {
	const aiNodeAnim *assimp_track = assimp_anim->mChannels[track_id];
	//make transform track
	int track_idx = animation->get_track_count();
	animation->add_track(Animation::TYPE_TRANSFORM);
	animation->track_set_path(track_idx, node_path);
	//first determine animation length

	float increment = 1.0 / float(anim_fps);
	float time = 0.0;

	bool last = false;

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
			rot = _interpolate_track<Quat>(rot_times, rot_values, time,
					AssetImportAnimation::INTERP_LINEAR)
						  .normalized();
		}

		if (scale_values.size()) {
			scale = _interpolate_track<Vector3>(scale_times, scale_values, time, AssetImportAnimation::INTERP_LINEAR);
		}

		if (skeleton) {
			int skeleton_bone = skeleton->find_bone(node_name);

			if (skeleton_bone >= 0 && track_bone) {
				Transform xform;
				xform.basis.set_quat_scale(rot, scale);
				xform.origin = pos;

				xform = skeleton->get_bone_rest(skeleton_bone).inverse() * xform;

				rot = xform.basis.get_rotation_quat();
				rot.normalize();
				scale = xform.basis.get_scale();
				pos = xform.origin;
			} else {
				ERR_FAIL_MSG("Skeleton bone lookup failed for skeleton: " + skeleton->get_name());
			}
		}

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

// I really do not like this but need to figure out a better way of removing it later.
Node *EditorSceneImporterAssimp::get_node_by_name(ImportState &state, String name) {
	for (Map<const aiNode *, Node3D *>::Element *key_value_pair = state.flat_node_map.front(); key_value_pair; key_value_pair = key_value_pair->next()) {
		const aiNode *assimp_node = key_value_pair->key();
		Node3D *node = key_value_pair->value();

		String node_name = AssimpUtils::get_assimp_string(assimp_node->mName);
		if (name == node_name && node) {
			return node;
		}
	}
	return nullptr;
}

/* Bone stack is a fifo handler for multiple armatures since armatures aren't a thing in assimp (yet) */
void EditorSceneImporterAssimp::RegenerateBoneStack(ImportState &state) {
	state.bone_stack.clear();
	// build bone stack list
	for (unsigned int mesh_id = 0; mesh_id < state.assimp_scene->mNumMeshes; ++mesh_id) {
		aiMesh *mesh = state.assimp_scene->mMeshes[mesh_id];

		// iterate over all the bones on the mesh for this node only!
		for (unsigned int boneIndex = 0; boneIndex < mesh->mNumBones; boneIndex++) {
			aiBone *bone = mesh->mBones[boneIndex];

			// doubtful this is required right now but best to check
			if (!state.bone_stack.find(bone)) {
				//print_verbose("[assimp] bone stack added: " + String(bone->mName.C_Str()) );
				state.bone_stack.push_back(bone);
			}
		}
	}
}

/* Bone stack is a fifo handler for multiple armatures since armatures aren't a thing in assimp (yet) */
void EditorSceneImporterAssimp::RegenerateBoneStack(ImportState &state, aiMesh *mesh) {
	state.bone_stack.clear();
	// iterate over all the bones on the mesh for this node only!
	for (unsigned int boneIndex = 0; boneIndex < mesh->mNumBones; boneIndex++) {
		aiBone *bone = mesh->mBones[boneIndex];
		if (state.bone_stack.find(bone) == nullptr) {
			state.bone_stack.push_back(bone);
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
	print_verbose("import animation: " + name);
	float ticks_per_second = anim->mTicksPerSecond;

	if (state.assimp_scene->mMetaData != nullptr && Math::is_equal_approx(ticks_per_second, 0.0f)) {
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

	if (name.begins_with("loop") || name.ends_with("loop") || name.begins_with("cycle") || name.ends_with("cycle")) {
		animation->set_loop(true);
	}

	// generate bone stack for animation import
	RegenerateBoneStack(state);

	//regular tracks
	for (size_t i = 0; i < anim->mNumChannels; i++) {
		const aiNodeAnim *track = anim->mChannels[i];
		String node_name = AssimpUtils::get_assimp_string(track->mNodeName);
		print_verbose("track name import: " + node_name);
		if (track->mNumRotationKeys == 0 && track->mNumPositionKeys == 0 && track->mNumScalingKeys == 0) {
			continue; //do not bother
		}

		Skeleton3D *skeleton = nullptr;
		NodePath node_path;
		aiBone *bone = nullptr;

		// Import skeleton bone animation for this track
		// Any bone will do, no point in processing more than just what is in the skeleton
		{
			bone = get_bone_from_stack(state, track->mNodeName);

			if (bone) {
				// get skeleton by bone
				skeleton = state.armature_skeletons[bone->mArmature];

				if (skeleton) {
					String path = state.root->get_path_to(skeleton);
					path += ":" + node_name;
					node_path = path;

					if (node_path != NodePath()) {
						_insert_animation_track(state, anim, i, p_bake_fps, animation, ticks_per_second, skeleton,
								node_path, node_name, bone);
					} else {
						print_error("Failed to find valid node path for animation");
					}
				}
			}
		}

		// not a bone
		// note this is flaky it uses node names which is unreliable
		Node *allocated_node = get_node_by_name(state, node_name);
		// todo: implement skeleton grabbing for node based animations too :)
		// check if node exists, if it does then also apply animation track for node and bones above are all handled.
		// this is now inclusive animation handling so that
		// we import all the data and do not miss anything.
		if (allocated_node) {
			node_path = state.root->get_path_to(allocated_node);

			if (node_path != NodePath()) {
				_insert_animation_track(state, anim, i, p_bake_fps, animation, ticks_per_second, skeleton,
						node_path, node_name, nullptr);
			}
		}
	}

	//blend shape tracks

	for (size_t i = 0; i < anim->mNumMorphMeshChannels; i++) {
		const aiMeshMorphAnim *anim_mesh = anim->mMorphMeshChannels[i];

		const String prop_name = AssimpUtils::get_assimp_string(anim_mesh->mName);
		const String mesh_name = prop_name.split("*")[0];

		ERR_CONTINUE(prop_name.split("*").size() != 2);

		Node *item = get_node_by_name(state, mesh_name);
		ERR_CONTINUE_MSG(!item, "failed to look up node by name");
		const MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(item);
		ERR_CONTINUE(mesh_instance == nullptr);

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
Ref<Mesh>
EditorSceneImporterAssimp::_generate_mesh_from_surface_indices(ImportState &state, const Vector<int> &p_surface_indices,
		const aiNode *assimp_node, Ref<Skin> &skin,
		Skeleton3D *&skeleton_assigned) {
	Ref<ArrayMesh> mesh;
	mesh.instance();
	bool has_uvs = false;
	bool compress_vert_data = state.import_flags & IMPORT_USE_COMPRESSION;
	uint32_t mesh_flags = compress_vert_data ? Mesh::ARRAY_COMPRESS_DEFAULT : 0;

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

		Map<uint32_t, Vector<BoneInfo>> vertex_weights;

		if (ai_mesh->mNumBones > 0) {
			for (size_t b = 0; b < ai_mesh->mNumBones; b++) {
				aiBone *bone = ai_mesh->mBones[b];

				if (!skeleton_assigned) {
					print_verbose("Assigned mesh skeleton during mesh creation");
					skeleton_assigned = state.skeleton_bone_map[bone];

					if (!skin.is_valid()) {
						print_verbose("Configured new skin");
						skin.instance();
					} else {
						print_verbose("Reusing existing skin!");
					}
				}
				//                skeleton_assigned =
				String bone_name = AssimpUtils::get_assimp_string(bone->mName);
				int bone_index = skeleton_assigned->find_bone(bone_name);
				ERR_CONTINUE(bone_index == -1);
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
				Color color = Color(ai_mesh->mColors[0]->r, ai_mesh->mColors[0]->g, ai_mesh->mColors[0]->b,
						ai_mesh->mColors[0]->a);
				st->add_color(color);
			}

			// Work out normal calculations? - this needs work it doesn't work properly on huestos
			if (ai_mesh->mNormals != nullptr) {
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
		Ref<StandardMaterial3D> mat;
		mat.instance();

		int32_t mat_two_sided = 0;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_TWOSIDED, mat_two_sided)) {
			if (mat_two_sided > 0) {
				mat->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
			} else {
				mat->set_cull_mode(StandardMaterial3D::CULL_BACK);
			}
		}

		aiString mat_name;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_NAME, mat_name)) {
			mat->set_name(AssimpUtils::get_assimp_string(mat_name));
		}

		// Culling handling for meshes

		// cull all back faces
		mat->set_cull_mode(StandardMaterial3D::CULL_DISABLED);

		// Now process materials
		aiTextureType base_color = aiTextureType_BASE_COLOR;
		{
			String filename, path;
			AssimpImageData image_data;

			if (AssimpUtils::GetAssimpTexture(state, ai_material, base_color, filename, path, image_data)) {
				AssimpUtils::set_texture_mapping_mode(image_data.map_mode, image_data.texture);

				// anything transparent must be culled
				if (image_data.raw_image->detect_alpha() != Image::ALPHA_NONE) {
					mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA_DEPTH_PRE_PASS);
					mat->set_cull_mode(StandardMaterial3D::CULL_DISABLED); // since you can see both sides in transparent mode
				}

				mat->set_texture(StandardMaterial3D::TEXTURE_ALBEDO, image_data.texture);
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
					mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA_DEPTH_PRE_PASS);
					mat->set_cull_mode(StandardMaterial3D::CULL_DISABLED); // since you can see both sides in transparent mode
				}

				mat->set_texture(StandardMaterial3D::TEXTURE_ALBEDO, image_data.texture);
			}

			aiColor4D clr_diffuse;
			if (AI_SUCCESS == ai_material->Get(AI_MATKEY_COLOR_DIFFUSE, clr_diffuse)) {
				if (Math::is_equal_approx(clr_diffuse.a, 1.0f) == false) {
					mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA_DEPTH_PRE_PASS);
					mat->set_cull_mode(StandardMaterial3D::CULL_DISABLED); // since you can see both sides in transparent mode
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
				mat->set_feature(StandardMaterial3D::Feature::FEATURE_NORMAL_MAPPING, true);
				mat->set_texture(StandardMaterial3D::TEXTURE_NORMAL, image_data.texture);
			} else {
				aiString texture_path;
				if (AI_SUCCESS == ai_material->Get(AI_MATKEY_FBX_NORMAL_TEXTURE, AI_PROPERTIES, texture_path)) {
					if (AssimpUtils::CreateAssimpTexture(state, texture_path, filename, path, image_data)) {
						mat->set_feature(StandardMaterial3D::Feature::FEATURE_NORMAL_MAPPING, true);
						mat->set_texture(StandardMaterial3D::TEXTURE_NORMAL, image_data.texture);
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
				mat->set_feature(StandardMaterial3D::Feature::FEATURE_NORMAL_MAPPING, true);
				mat->set_texture(StandardMaterial3D::TEXTURE_NORMAL, image_data.texture);
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
				mat->set_feature(StandardMaterial3D::Feature::FEATURE_NORMAL_MAPPING, true);
				mat->set_texture(StandardMaterial3D::TEXTURE_NORMAL, image_data.texture);
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
				mat->set_texture(StandardMaterial3D::TEXTURE_METALLIC, image_data.texture);
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
				mat->set_texture(StandardMaterial3D::TEXTURE_ROUGHNESS, image_data.texture);
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
				mat->set_feature(StandardMaterial3D::FEATURE_EMISSION, true);
				mat->set_texture(StandardMaterial3D::TEXTURE_EMISSION, image_data.texture);
			} else {
				// Process emission textures
				aiString texture_emissive_path;
				if (AI_SUCCESS ==
						ai_material->Get(AI_MATKEY_FBX_MAYA_EMISSION_TEXTURE, AI_PROPERTIES, texture_emissive_path)) {
					if (AssimpUtils::CreateAssimpTexture(state, texture_emissive_path, filename, path, image_data)) {
						mat->set_feature(StandardMaterial3D::FEATURE_EMISSION, true);
						mat->set_texture(StandardMaterial3D::TEXTURE_EMISSION, image_data.texture);
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
				mat->set_texture(StandardMaterial3D::TEXTURE_METALLIC, image_data.texture);
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
				mat->set_feature(StandardMaterial3D::FEATURE_AMBIENT_OCCLUSION, true);
				mat->set_texture(StandardMaterial3D::TEXTURE_AMBIENT_OCCLUSION, image_data.texture);
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
			array_copy.resize(RenderingServer::ARRAY_MAX);

			for (int l = 0; l < RenderingServer::ARRAY_MAX; l++) {
				array_copy[l] = array_mesh[l].duplicate(true);
			}

			const size_t num_vertices = ai_mesh->mAnimMeshes[j]->mNumVertices;
			array_copy[Mesh::ARRAY_INDEX] = Variant();
			if (ai_mesh->mAnimMeshes[j]->HasPositions()) {
				PackedVector3Array vertices;
				vertices.resize(num_vertices);
				for (size_t l = 0; l < num_vertices; l++) {
					const aiVector3D ai_pos = ai_mesh->mAnimMeshes[j]->mVertices[l];
					Vector3 position = Vector3(ai_pos.x, ai_pos.y, ai_pos.z);
					vertices.ptrw()[l] = position;
				}
				PackedVector3Array new_vertices = array_copy[RenderingServer::ARRAY_VERTEX].duplicate(true);
				ERR_CONTINUE(vertices.size() != new_vertices.size());
				for (int32_t l = 0; l < new_vertices.size(); l++) {
					Vector3 *w = new_vertices.ptrw();
					w[l] = vertices[l];
				}
				array_copy[RenderingServer::ARRAY_VERTEX] = new_vertices;
			}

			int32_t color_set = 0;
			if (ai_mesh->mAnimMeshes[j]->HasVertexColors(color_set)) {
				PackedColorArray colors;
				colors.resize(num_vertices);
				for (size_t l = 0; l < num_vertices; l++) {
					const aiColor4D ai_color = ai_mesh->mAnimMeshes[j]->mColors[color_set][l];
					Color color = Color(ai_color.r, ai_color.g, ai_color.b, ai_color.a);
					colors.ptrw()[l] = color;
				}
				PackedColorArray new_colors = array_copy[RenderingServer::ARRAY_COLOR].duplicate(true);
				ERR_CONTINUE(colors.size() != new_colors.size());
				for (int32_t l = 0; l < colors.size(); l++) {
					Color *w = new_colors.ptrw();
					w[l] = colors[l];
				}
				array_copy[RenderingServer::ARRAY_COLOR] = new_colors;
			}

			if (ai_mesh->mAnimMeshes[j]->HasNormals()) {
				PackedVector3Array normals;
				normals.resize(num_vertices);
				for (size_t l = 0; l < num_vertices; l++) {
					const aiVector3D ai_normal = ai_mesh->mAnimMeshes[j]->mNormals[l];
					Vector3 normal = Vector3(ai_normal.x, ai_normal.y, ai_normal.z);
					normals.ptrw()[l] = normal;
				}
				PackedVector3Array new_normals = array_copy[RenderingServer::ARRAY_NORMAL].duplicate(true);
				ERR_CONTINUE(normals.size() != new_normals.size());
				for (int l = 0; l < normals.size(); l++) {
					Vector3 *w = new_normals.ptrw();
					w[l] = normals[l];
				}
				array_copy[RenderingServer::ARRAY_NORMAL] = new_normals;
			}

			if (ai_mesh->mAnimMeshes[j]->HasTangentsAndBitangents()) {
				PackedColorArray tangents;
				tangents.resize(num_vertices);
				Color *w = tangents.ptrw();
				for (size_t l = 0; l < num_vertices; l++) {
					AssimpUtils::calc_tangent_from_mesh(ai_mesh, j, l, l, w);
				}
				PackedFloat32Array new_tangents = array_copy[RenderingServer::ARRAY_TANGENT].duplicate(true);
				ERR_CONTINUE(new_tangents.size() != tangents.size() * 4);
				for (int32_t l = 0; l < tangents.size(); l++) {
					new_tangents.ptrw()[l + 0] = tangents[l].r;
					new_tangents.ptrw()[l + 1] = tangents[l].g;
					new_tangents.ptrw()[l + 2] = tangents[l].b;
					new_tangents.ptrw()[l + 3] = tangents[l].a;
				}
				array_copy[RenderingServer::ARRAY_TANGENT] = new_tangents;
			}

			morphs[j] = array_copy;
		}
		mesh->add_surface_from_arrays(primitive, array_mesh, morphs, Dictionary(), mesh_flags);
		mesh->surface_set_material(i, mat);
		mesh->surface_set_name(i, AssimpUtils::get_assimp_string(ai_mesh->mName));
	}

	return mesh;
}

/**
 * Create a new mesh for the node supplied
 */
MeshInstance3D *
EditorSceneImporterAssimp::create_mesh(ImportState &state, const aiNode *assimp_node, const String &node_name, Node *active_node, Transform node_transform) {
	/* MESH NODE */
	Ref<Mesh> mesh;
	Ref<Skin> skin;
	// see if we have mesh cache for this.
	Vector<int> surface_indices;

	RegenerateBoneStack(state);

	// Configure indices
	for (uint32_t i = 0; i < assimp_node->mNumMeshes; i++) {
		int mesh_index = assimp_node->mMeshes[i];
		// create list of mesh indexes
		surface_indices.push_back(mesh_index);
	}

	//surface_indices.sort();
	String mesh_key;
	for (int i = 0; i < surface_indices.size(); i++) {
		if (i > 0) {
			mesh_key += ":";
		}
		mesh_key += itos(surface_indices[i]);
	}

	Skeleton3D *skeleton = nullptr;
	aiNode *armature = nullptr;

	if (!state.mesh_cache.has(mesh_key)) {
		mesh = _generate_mesh_from_surface_indices(state, surface_indices, assimp_node, skin, skeleton);
		state.mesh_cache[mesh_key] = mesh;
	}

	MeshInstance3D *mesh_node = memnew(MeshInstance3D);
	mesh = state.mesh_cache[mesh_key];
	mesh_node->set_mesh(mesh);

	// if we have a valid skeleton set it up
	if (skin.is_valid()) {
		for (uint32_t i = 0; i < assimp_node->mNumMeshes; i++) {
			unsigned int mesh_index = assimp_node->mMeshes[i];
			const aiMesh *ai_mesh = state.assimp_scene->mMeshes[mesh_index];

			// please remember bone id relative to the skin is NOT the mesh relative index.
			// it is the index relative to the skeleton that is why
			// we have state.bone_id_map, it allows for duplicate bone id's too :)
			// hope this makes sense

			int bind_count = 0;
			for (unsigned int boneId = 0; boneId < ai_mesh->mNumBones; ++boneId) {
				aiBone *iterBone = ai_mesh->mBones[boneId];

				// used to reparent mesh to the correct armature later on if assigned.
				if (!armature) {
					print_verbose("Configured mesh armature, will reparent later to armature");
					armature = iterBone->mArmature;
				}

				if (skeleton) {
					int id = skeleton->find_bone(AssimpUtils::get_assimp_string(iterBone->mName));
					if (id != -1) {
						print_verbose("Set bind bone: mesh: " + itos(mesh_index) + " bone index: " + itos(id));
						Transform t = AssimpUtils::assimp_matrix_transform(iterBone->mOffsetMatrix);

						skin->add_bind(bind_count, t);
						skin->set_bind_bone(bind_count, id);
						bind_count++;
					}
				}
			}
		}

		print_verbose("Finished configuring bind pose for skin mesh");
	}

	// this code parents all meshes with bones to the armature they are for
	// GLTF2 specification relies on this and we are enforcing it for FBX.
	if (armature && state.flat_node_map[armature]) {
		Node *armature_parent = state.flat_node_map[armature];
		print_verbose("Parented mesh " + node_name + " to armature " + armature_parent->get_name());
		// static mesh handling
		armature_parent->add_child(mesh_node);
		// transform must be identity
		mesh_node->set_global_transform(Transform());
		mesh_node->set_name(node_name);
		mesh_node->set_owner(state.root);
	} else {
		// static mesh handling
		active_node->add_child(mesh_node);
		mesh_node->set_global_transform(node_transform);
		mesh_node->set_name(node_name);
		mesh_node->set_owner(state.root);
	}

	if (skeleton) {
		print_verbose("Attempted to set skeleton path!");
		mesh_node->set_skeleton_path(mesh_node->get_path_to(skeleton));
		mesh_node->set_skin(skin);
	}

	return mesh_node;
}

/**
 * Create a light for the scene
 * Automatically caches lights for lookup later
 */
Node3D *EditorSceneImporterAssimp::create_light(
		ImportState &state,
		const String &node_name,
		Transform &look_at_transform) {
	Light3D *light = nullptr;
	aiLight *assimp_light = state.assimp_scene->mLights[state.light_cache[node_name]];
	ERR_FAIL_COND_V(!assimp_light, nullptr);

	if (assimp_light->mType == aiLightSource_DIRECTIONAL) {
		light = memnew(DirectionalLight3D);
	} else if (assimp_light->mType == aiLightSource_POINT) {
		light = memnew(OmniLight3D);
	} else if (assimp_light->mType == aiLightSource_SPOT) {
		light = memnew(SpotLight3D);
	}
	ERR_FAIL_COND_V(light == nullptr, nullptr);

	if (assimp_light->mType != aiLightSource_POINT) {
		Vector3 pos = Vector3(
				assimp_light->mPosition.x,
				assimp_light->mPosition.y,
				assimp_light->mPosition.z);
		Vector3 look_at = Vector3(
				assimp_light->mDirection.y,
				assimp_light->mDirection.x,
				assimp_light->mDirection.z)
								  .normalized();
		Vector3 up = Vector3(
				assimp_light->mUp.x,
				assimp_light->mUp.y,
				assimp_light->mUp.z);

		look_at_transform.set_look_at(pos, look_at, up);
	}
	// properties for light variables should be put here.
	// not really hugely important yet but we will need them in the future

	light->set_color(
			Color(assimp_light->mColorDiffuse.r, assimp_light->mColorDiffuse.g, assimp_light->mColorDiffuse.b));

	return light;
}

/**
 * Create camera for the scene
 */
Node3D *EditorSceneImporterAssimp::create_camera(
		ImportState &state,
		const String &node_name,
		Transform &look_at_transform) {
	aiCamera *camera = state.assimp_scene->mCameras[state.camera_cache[node_name]];
	ERR_FAIL_COND_V(!camera, nullptr);

	Camera3D *camera_node = memnew(Camera3D);
	ERR_FAIL_COND_V(!camera_node, nullptr);
	float near = camera->mClipPlaneNear;
	if (Math::is_equal_approx(near, 0.0f)) {
		near = 0.1f;
	}
	camera_node->set_perspective(Math::rad2deg(camera->mHorizontalFOV) * 2.0f, near, camera->mClipPlaneFar);
	Vector3 pos = Vector3(camera->mPosition.x, camera->mPosition.y, camera->mPosition.z);
	Vector3 look_at = Vector3(camera->mLookAt.y, camera->mLookAt.x, camera->mLookAt.z).normalized();
	Vector3 up = Vector3(camera->mUp.x, camera->mUp.y, camera->mUp.z);

	look_at_transform.set_look_at(pos + look_at_transform.origin, look_at, up);
	return camera_node;
}

/**
 * Generate node
 * Recursive call to iterate over all nodes
 */
void EditorSceneImporterAssimp::_generate_node(
		ImportState &state,
		const aiNode *assimp_node) {
	ERR_FAIL_COND(assimp_node == nullptr);
	state.nodes.push_back(assimp_node);
	String parent_name = AssimpUtils::get_assimp_string(assimp_node->mParent->mName);

	// please note
	// duplicate bone names exist
	// this is why we only check if the bone exists
	// so everything else is useless but the name
	// please do not copy any other values from get_bone_by_name.
	aiBone *parent_bone = get_bone_by_name(state.assimp_scene, assimp_node->mParent->mName);
	aiBone *current_bone = get_bone_by_name(state.assimp_scene, assimp_node->mName);

	// is this an armature
	// parent null
	// and this is the first bone :)
	if (parent_bone == nullptr && current_bone) {
		state.armature_nodes.push_back(assimp_node->mParent);
		print_verbose("found valid armature: " + parent_name);
	}

	for (size_t i = 0; i < assimp_node->mNumChildren; i++) {
		_generate_node(state, assimp_node->mChildren[i]);
	}
}

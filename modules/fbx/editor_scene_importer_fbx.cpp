/*************************************************************************/
/*  editor_scene_importer_fbx.cpp                                        */
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

#include "editor_scene_importer_fbx.h"

#include "data/fbx_anim_container.h"
#include "data/fbx_material.h"
#include "data/fbx_mesh_data.h"
#include "data/fbx_skeleton.h"
#include "tools/import_utils.h"

#include "core/io/image_loader.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/import/resource_importer_scene.h"
#include "scene/3d/bone_attachment_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"

#include "fbx_parser/FBXDocument.h"
#include "fbx_parser/FBXImportSettings.h"
#include "fbx_parser/FBXMeshGeometry.h"
#include "fbx_parser/FBXParser.h"
#include "fbx_parser/FBXProperties.h"
#include "fbx_parser/FBXTokenizer.h"

#include <string>

void EditorSceneFormatImporterFBX::get_extensions(List<String> *r_extensions) const {
	// register FBX as the one and only format for FBX importing
	const String import_setting_string = "filesystem/import/fbx/";
	const String fbx_str = "fbx";
	Vector<String> exts;
	exts.push_back(fbx_str);
	_register_project_setting_import(fbx_str, import_setting_string, exts, r_extensions, true);
}

void EditorSceneFormatImporterFBX::_register_project_setting_import(const String generic,
		const String import_setting_string,
		const Vector<String> &exts,
		List<String> *r_extensions,
		const bool p_enabled) const {
	const String use_generic = "use_" + generic;
	_GLOBAL_DEF(import_setting_string + use_generic, p_enabled, true);
	if (ProjectSettings::get_singleton()->get(import_setting_string + use_generic)) {
		for (int32_t i = 0; i < exts.size(); i++) {
			r_extensions->push_back(exts[i]);
		}
	}
}

uint32_t EditorSceneFormatImporterFBX::get_import_flags() const {
	return IMPORT_SCENE;
}

Node3D *EditorSceneFormatImporterFBX::import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps,
		List<String> *r_missing_deps, Error *r_err) {
	// done for performance when re-importing lots of files when testing importer in verbose only!
	if (OS::get_singleton()->is_stdout_verbose()) {
		EditorLog *log = EditorNode::get_log();
		log->clear();
	}
	Error err;
	FileAccessRef f = FileAccess::open(p_path, FileAccess::READ, &err);

	ERR_FAIL_COND_V(!f, nullptr);

	{
		PackedByteArray data;
		// broadphase tokenizing pass in which we identify the core
		// syntax elements of FBX (brackets, commas, key:value mappings)
		FBXDocParser::TokenList tokens;

		bool is_binary = false;
		data.resize(f->get_length());

		ERR_FAIL_COND_V(data.size() < 64, nullptr);

		f->get_buffer(data.ptrw(), data.size());
		PackedByteArray fbx_header;
		fbx_header.resize(64);
		for (int32_t byte_i = 0; byte_i < 64; byte_i++) {
			fbx_header.ptrw()[byte_i] = data.ptr()[byte_i];
		}

		String fbx_header_string;
		if (fbx_header.size() >= 0) {
			fbx_header_string.parse_utf8((const char *)fbx_header.ptr(), fbx_header.size());
		}

		print_verbose("[doc] opening fbx file: " + p_path);
		print_verbose("[doc] fbx header: " + fbx_header_string);
		bool corrupt = false;

		// safer to check this way as there can be different formatted headers
		if (fbx_header_string.find("Kaydara FBX Binary", 0) != -1) {
			is_binary = true;
			print_verbose("[doc] is binary");

			FBXDocParser::TokenizeBinary(tokens, (const char *)data.ptrw(), (size_t)data.size(), corrupt);

		} else {
			print_verbose("[doc] is ascii");
			FBXDocParser::Tokenize(tokens, (const char *)data.ptrw(), (size_t)data.size(), corrupt);
		}

		if (corrupt) {
			for (FBXDocParser::TokenPtr token : tokens) {
				delete token;
			}
			tokens.clear();
			ERR_PRINT(vformat("Cannot import FBX file: %s the file is corrupt so we safely exited parsing the file.", p_path));
			return memnew(Node3D);
		}

		// The import process explained:
		// 1. Tokens are made, these are then taken into the 'parser' below
		// 2. The parser constructs 'Elements' and all 'real' FBX Types.
		// 3. This creates a problem: shared_ptr ownership, should Elements later 'take ownership'
		// 4. No, it shouldn't so we should either a.) use weak ref for elements; but this is not correct.

		// use this information to construct a very rudimentary
		// parse-tree representing the FBX scope structure
		FBXDocParser::Parser parser(tokens, is_binary);

		if (parser.IsCorrupt()) {
			for (FBXDocParser::TokenPtr token : tokens) {
				delete token;
			}
			tokens.clear();
			ERR_PRINT(vformat("Cannot import FBX file: %s the file is corrupt so we safely exited parsing the file.", p_path));
			return memnew(Node3D);
		}

		FBXDocParser::ImportSettings settings;
		settings.strictMode = false;

		// this function leaks a lot
		FBXDocParser::Document doc(parser, settings);

		// yeah so closing the file is a good idea (prevents readonly states)
		f->close();

		// safety for version handling
		if (doc.IsSafeToImport()) {
			bool is_blender_fbx = false;
			const FBXDocParser::PropertyTable &import_props = doc.GetMetadataProperties();
			const FBXDocParser::PropertyPtr app_name = import_props.Get("Original|ApplicationName");
			const FBXDocParser::PropertyPtr app_vendor = import_props.Get("Original|ApplicationVendor");
			const FBXDocParser::PropertyPtr app_version = import_props.Get("Original|ApplicationVersion");
			//
			if (app_name) {
				const FBXDocParser::TypedProperty<std::string> *app_name_string = dynamic_cast<const FBXDocParser::TypedProperty<std::string> *>(app_name);
				if (app_name_string) {
					print_verbose("FBX App Name: " + String(app_name_string->Value().c_str()));
				}
			}

			if (app_vendor) {
				const FBXDocParser::TypedProperty<std::string> *app_vendor_string = dynamic_cast<const FBXDocParser::TypedProperty<std::string> *>(app_vendor);
				if (app_vendor_string) {
					print_verbose("FBX App Vendor: " + String(app_vendor_string->Value().c_str()));
					is_blender_fbx = app_vendor_string->Value().find("Blender") != std::string::npos;
				}
			}

			if (app_version) {
				const FBXDocParser::TypedProperty<std::string> *app_version_string = dynamic_cast<const FBXDocParser::TypedProperty<std::string> *>(app_version);
				if (app_version_string) {
					print_verbose("FBX App Version: " + String(app_version_string->Value().c_str()));
				}
			}

			if (is_blender_fbx) {
				WARN_PRINT("We don't officially support Blender FBX animations yet, due to issues with upstream Blender,\n"
						   "so please wait for us to work around remaining issues. We will continue to import the file but it may be broken.\n"
						   "For minimal breakage, please export FBX from Blender with -Z forward, and Y up.");
			}

			Node3D *spatial = _generate_scene(p_path, &doc, p_flags, p_bake_fps, 8, is_blender_fbx);
			// todo: move to document shutdown (will need to be validated after moving; this code has been validated already)
			for (FBXDocParser::TokenPtr token : tokens) {
				if (token) {
					delete token;
					token = nullptr;
				}
			}

			return spatial;

		} else {
			for (FBXDocParser::TokenPtr token : tokens) {
				delete token;
			}
			tokens.clear();

			ERR_PRINT(vformat("Cannot import FBX file: %s. It uses file format %d which is unsupported by Godot. Please re-export it or convert it to a newer format.", p_path, doc.FBXVersion()));
		}
	}

	return memnew(Node3D);
}

template <class T>
struct EditorSceneFormatImporterAssetImportInterpolate {
	T lerp(const T &a, const T &b, float c) const {
		return a + (b - a) * c;
	}

	T catmull_rom(const T &p0, const T &p1, const T &p2, const T &p3, float t) {
		const float t2 = t * t;
		const float t3 = t2 * t;

		return 0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
	}

	T bezier(T start, T control_1, T control_2, T end, float t) {
		/* Formula from Wikipedia article on Bezier curves. */
		const real_t omt = (1.0 - t);
		const real_t omt2 = omt * omt;
		const real_t omt3 = omt2 * omt;
		const real_t t2 = t * t;
		const real_t t3 = t2 * t;

		return start * omt3 + control_1 * omt2 * t * 3.0 + control_2 * omt * t2 * 3.0 + end * t3;
	}
};

//thank you for existing, partial specialization
template <>
struct EditorSceneFormatImporterAssetImportInterpolate<Quaternion> {
	Quaternion lerp(const Quaternion &a, const Quaternion &b, float c) const {
		ERR_FAIL_COND_V(!a.is_normalized(), Quaternion());
		ERR_FAIL_COND_V(!b.is_normalized(), Quaternion());

		return a.slerp(b, c).normalized();
	}

	Quaternion catmull_rom(const Quaternion &p0, const Quaternion &p1, const Quaternion &p2, const Quaternion &p3, float c) {
		ERR_FAIL_COND_V(!p1.is_normalized(), Quaternion());
		ERR_FAIL_COND_V(!p2.is_normalized(), Quaternion());

		return p1.slerp(p2, c).normalized();
	}

	Quaternion bezier(Quaternion start, Quaternion control_1, Quaternion control_2, Quaternion end, float t) {
		ERR_FAIL_COND_V(!start.is_normalized(), Quaternion());
		ERR_FAIL_COND_V(!end.is_normalized(), Quaternion());

		return start.slerp(end, t).normalized();
	}
};

template <class T>
T EditorSceneFormatImporterFBX::_interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time,
		AssetImportAnimation::Interpolation p_interp) {
	//could use binary search, worth it?
	int idx = -1;
	for (int i = 0; i < p_times.size(); i++) {
		if (p_times[i] > p_time) {
			break;
		}
		idx++;
	}

	EditorSceneFormatImporterAssetImportInterpolate<T> interp;

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

Node3D *EditorSceneFormatImporterFBX::_generate_scene(
		const String &p_path,
		const FBXDocParser::Document *p_document,
		const uint32_t p_flags,
		int p_bake_fps,
		const int32_t p_max_bone_weights,
		bool p_is_blender_fbx) {
	ImportState state;
	state.is_blender_fbx = p_is_blender_fbx;
	state.path = p_path;
	state.animation_player = nullptr;

	// create new root node for scene
	Node3D *scene_root = memnew(Node3D);
	state.root = memnew(Node3D);
	state.root_owner = scene_root; // the real scene root... sorry compatibility code is painful...

	state.root->set_name("RootNode");
	scene_root->add_child(state.root);
	state.root->set_owner(scene_root);

	state.fbx_root_node.instantiate();
	state.fbx_root_node->godot_node = state.root;

	// Size relative to cm.
	const real_t fbx_unit_scale = p_document->GlobalSettingsPtr()->UnitScaleFactor();

	print_verbose("FBX unit scale import value: " + rtos(fbx_unit_scale));
	// Set FBX file scale is relative to CM must be converted to M
	state.scale = fbx_unit_scale / 100.0;
	print_verbose("FBX unit scale is: " + rtos(state.scale));

	// Enabled by default.
	state.enable_material_import = true;
	// Enabled by default.
	state.enable_animation_import = true;
	Ref<FBXNode> root_node;
	root_node.instantiate();

	// make sure fake noFBXDocParser::PropertyPtr ptrde always has a transform too ;)
	Ref<PivotTransform> pivot_transform;
	pivot_transform.instantiate();
	root_node->pivot_transform = pivot_transform;
	root_node->node_name = "root node";
	root_node->current_node_id = 0;
	root_node->godot_node = state.root;

	// cache this node onto the fbx_target map.
	state.fbx_target_map.insert(0, root_node);

	// cache basic node information from FBX document
	// grabs all FBX bones
	BuildDocumentBones(Ref<FBXBone>(), state, p_document, 0L);
	BuildDocumentNodes(Ref<PivotTransform>(), state, p_document, 0L, nullptr);

	// Build document skinning information

	// Algorithm is this:
	// Get Deformer: object with "Skin" class.
	// Deformer:: has link to Geometry:: (correct mesh for skin)
	// Deformer:: has Source which is the SubDeformer:: (e.g. the Cluster)
	// Notes at the end it configures the vertex weight mapping.

	for (uint64_t skin_id : p_document->GetSkinIDs()) {
		// Validate the parser
		FBXDocParser::LazyObject *lazy_skin = p_document->GetObject(skin_id);
		ERR_CONTINUE_MSG(lazy_skin == nullptr, "invalid lazy object [serious parser bug]");

		// Validate the parser
		const FBXDocParser::Skin *skin = lazy_skin->Get<FBXDocParser::Skin>();
		ERR_CONTINUE_MSG(skin == nullptr, "invalid skin added to skin list [parser bug]");

		const std::vector<const FBXDocParser::Connection *> source_to_destination = p_document->GetConnectionsBySourceSequenced(skin_id);
		FBXDocParser::MeshGeometry *mesh = nullptr;
		uint64_t mesh_id = 0;

		// Most likely only contains the mesh link for the skin
		// The mesh geometry.
		for (const FBXDocParser::Connection *con : source_to_destination) {
			// do something
			print_verbose("src: " + itos(con->src));
			FBXDocParser::Object *ob = con->DestinationObject();
			mesh = dynamic_cast<FBXDocParser::MeshGeometry *>(ob);

			if (mesh) {
				mesh_id = mesh->ID();
				break;
			}
		}

		// Validate the mesh exists and was retrieved
		ERR_CONTINUE_MSG(mesh_id == 0, "mesh id is invalid");
		const std::vector<const FBXDocParser::Cluster *> clusters = skin->Clusters();

		// NOTE: this will ONLY work on skinned bones (it is by design.)
		// A cluster is a skinned bone so SKINS won't contain unskinned bones so we need to pre-add all bones and parent them in a step beforehand.
		for (const FBXDocParser::Cluster *cluster : clusters) {
			ERR_CONTINUE_MSG(cluster == nullptr, "invalid bone cluster");
			const uint64_t deformer_id = cluster->ID();
			std::vector<const FBXDocParser::Connection *> connections = p_document->GetConnectionsByDestinationSequenced(deformer_id);

			// Weight data always has a node in the scene lets grab the limb's node in the scene :) (reverse set to true since it's the opposite way around)
			const FBXDocParser::ModelLimbNode *limb_node = ProcessDOMConnection<FBXDocParser::ModelLimbNode>(p_document, deformer_id, true);

			ERR_CONTINUE_MSG(limb_node == nullptr, "unable to resolve model for skinned bone");

			const uint64_t model_id = limb_node->ID();

			// This will never happen, so if it does you know you fucked up.
			ERR_CONTINUE_MSG(!state.fbx_bone_map.has(model_id), "missing LimbNode detected");

			// new bone instance
			Ref<FBXBone> bone_element = state.fbx_bone_map[model_id];

			//
			// Bone Weight Information Configuration
			//

			// Cache Weight Information into bone for later usage if you want the raw data.
			const std::vector<unsigned int> &indexes = cluster->GetIndices();
			const std::vector<float> &weights = cluster->GetWeights();
			Ref<FBXMeshData> mesh_vertex_data;

			// this data will pre-exist if vertex weight information is found
			if (state.renderer_mesh_data.has(mesh_id)) {
				mesh_vertex_data = state.renderer_mesh_data[mesh_id];
			} else {
				mesh_vertex_data.instantiate();
				state.renderer_mesh_data.insert(mesh_id, mesh_vertex_data);
			}

			mesh_vertex_data->armature_id = bone_element->armature_id;
			mesh_vertex_data->valid_armature_id = true;

			//print_verbose("storing mesh vertex data for mesh to use later");
			ERR_CONTINUE_MSG(indexes.size() != weights.size(), "[doc] error mismatch between weight info");

			for (size_t idx = 0; idx < indexes.size(); idx++) {
				const size_t vertex_index = indexes[idx];
				const real_t influence_weight = weights[idx];

				VertexWeightMapping &vm = mesh_vertex_data->vertex_weights[vertex_index];
				vm.weights.push_back(influence_weight);
				vm.bones.push_back(0); // bone id is pushed on here during sanitization phase
				vm.bones_ref.push_back(bone_element);
			}

			for (const int *vertex_index = mesh_vertex_data->vertex_weights.next(nullptr);
					vertex_index != nullptr;
					vertex_index = mesh_vertex_data->vertex_weights.next(vertex_index)) {
				VertexWeightMapping *vm = mesh_vertex_data->vertex_weights.getptr(*vertex_index);
				const int influence_count = vm->weights.size();
				if (influence_count > mesh_vertex_data->max_weight_count) {
					mesh_vertex_data->max_weight_count = influence_count;
					mesh_vertex_data->valid_weight_count = true;
				}
			}

			if (mesh_vertex_data->max_weight_count > 4) {
				if (mesh_vertex_data->max_weight_count > 8) {
					ERR_PRINT("[doc] Serious: maximum bone influences is 8 in this branch.");
				}
				// Clamp to 8 bone vertex influences.
				mesh_vertex_data->max_weight_count = 8;
				print_verbose("[doc] Using 8 vertex bone influences configuration.");
			} else {
				mesh_vertex_data->max_weight_count = 4;
				print_verbose("[doc] Using 4 vertex bone influences configuration.");
			}
		}
	}

	// do we globally allow for import of materials
	// (prevents overwrite of materials; so you can handle them explicitly)
	if (state.enable_material_import) {
		const std::vector<uint64_t> &materials = p_document->GetMaterialIDs();

		for (uint64_t material_id : materials) {
			FBXDocParser::LazyObject *lazy_material = p_document->GetObject(material_id);
			FBXDocParser::Material *mat = (FBXDocParser::Material *)lazy_material->Get<FBXDocParser::Material>();
			ERR_CONTINUE_MSG(!mat, "Could not convert fbx material by id: " + itos(material_id));

			Ref<FBXMaterial> material;
			material.instantiate();
			material->set_imported_material(mat);

			Ref<StandardMaterial3D> godot_material = material->import_material(state);

			state.cached_materials.insert(material_id, godot_material);
		}
	}

	// build skin and skeleton information
	print_verbose("[doc] Skeleton3D Bone count: " + itos(state.fbx_bone_map.size()));

	// Importing bones using document based method from FBX directly
	// We do not use the assimp bone format to determine this information anymore.
	if (state.fbx_bone_map.size() > 0) {
		// We are using a single skeleton only method here
		// this is because we really have no concept of skeletons in FBX
		// their are bones in a scene but they have no specific armature
		// we can detect armatures but the issue lies in the complexity
		// we opted to merge the entire scene onto one skeleton for now
		// if we need to change this we have an archive of the old code.

		// bind pose normally only has 1 per mesh but can have more than one
		// this is the point of skins
		// in FBX first bind pose is the master for the first skin

		// In order to handle the FBX skeleton we must also inverse any parent transforms on the bones
		// just to rule out any parent node transforms in the bone data
		// this is trivial to do and allows us to use the single skeleton method and merge them
		// this means that the nodes from maya kLocators will be preserved as bones
		// in the same rig without having to match this across skeletons and merge by detection
		// we can just merge and undo any parent transforms
		for (KeyValue<uint64_t, Ref<FBXBone>> &bone_element : state.fbx_bone_map) {
			Ref<FBXBone> bone = bone_element.value;
			Ref<FBXSkeleton> fbx_skeleton_inst;

			uint64_t armature_id = bone->armature_id;
			if (state.skeleton_map.has(armature_id)) {
				fbx_skeleton_inst = state.skeleton_map[armature_id];
			} else {
				fbx_skeleton_inst.instantiate();
				state.skeleton_map.insert(armature_id, fbx_skeleton_inst);
			}

			print_verbose("populating skeleton with bone: " + bone->bone_name);

			//// populate bone skeleton - since fbx has no DOM for the skeleton just a node.
			//bone->bone_skeleton = fbx_skeleton_inst;

			// now populate bone on the armature node list
			fbx_skeleton_inst->skeleton_bones.push_back(bone);

			CRASH_COND_MSG(!state.fbx_target_map.has(armature_id), "invalid armature [serious]");

			Ref<FBXNode> node = state.fbx_target_map[armature_id];

			CRASH_COND_MSG(node.is_null(), "invalid node [serious]");
			CRASH_COND_MSG(node->pivot_transform.is_null(), "invalid pivot transform [serious]");
			fbx_skeleton_inst->fbx_node = node;

			ERR_CONTINUE_MSG(fbx_skeleton_inst->fbx_node.is_null(), "invalid skeleton node [serious]");

			// we need to have a valid armature id and the model configured for the bone to be assigned fully.
			// happens once per skeleton

			if (state.fbx_target_map.has(armature_id) && !fbx_skeleton_inst->fbx_node->has_model()) {
				print_verbose("allocated fbx skeleton primary / armature node for the level: " + fbx_skeleton_inst->fbx_node->node_name);
			} else if (!state.fbx_target_map.has(armature_id) && !fbx_skeleton_inst->fbx_node->has_model()) {
				print_error("bones are not mapped to an armature node for armature id: " + itos(armature_id) + " bone: " + bone->bone_name);
				// this means bone will be removed and not used, which is safe actually and no skeleton will be created.
			}
		}

		// setup skeleton instances if required :)
		for (KeyValue<uint64_t, Ref<FBXSkeleton>> &skeleton_node : state.skeleton_map) {
			Ref<FBXSkeleton> &skeleton = skeleton_node.value;
			skeleton->init_skeleton(state);

			ERR_CONTINUE_MSG(skeleton->fbx_node.is_null(), "invalid fbx target map, missing skeleton");
		}

		// This list is not populated
		for (Map<uint64_t, Ref<FBXNode>>::Element *skin_mesh = state.MeshNodes.front(); skin_mesh; skin_mesh = skin_mesh->next()) {
		}
	}

	// build godot node tree
	if (state.fbx_node_list.size() > 0) {
		for (List<Ref<FBXNode>>::Element *node_element = state.fbx_node_list.front();
				node_element;
				node_element = node_element->next()) {
			Ref<FBXNode> fbx_node = node_element->get();
			ImporterMeshInstance3D *mesh_node = nullptr;
			Ref<FBXMeshData> mesh_data_precached;

			// check for valid geometry
			if (fbx_node->fbx_model == nullptr) {
				print_error("[doc] fundamental flaw, submit bug immediately with full import log with verbose logging on");
			} else {
				const std::vector<const FBXDocParser::Geometry *> &geometry = fbx_node->fbx_model->GetGeometry();
				for (const FBXDocParser::Geometry *mesh : geometry) {
					print_verbose("[doc] [" + itos(mesh->ID()) + "] mesh: " + fbx_node->node_name);

					if (mesh == nullptr) {
						continue;
					}

					const FBXDocParser::MeshGeometry *mesh_geometry = dynamic_cast<const FBXDocParser::MeshGeometry *>(mesh);
					if (mesh_geometry) {
						uint64_t mesh_id = mesh_geometry->ID();

						// this data will pre-exist if vertex weight information is found
						if (state.renderer_mesh_data.has(mesh_id)) {
							mesh_data_precached = state.renderer_mesh_data[mesh_id];
						} else {
							mesh_data_precached.instantiate();
							state.renderer_mesh_data.insert(mesh_id, mesh_data_precached);
						}

						mesh_data_precached->mesh_node = fbx_node;

						// mesh node, mesh id
						mesh_node = mesh_data_precached->create_fbx_mesh(state, mesh_geometry, fbx_node->fbx_model, false);
						if (!state.MeshNodes.has(mesh_id)) {
							state.MeshNodes.insert(mesh_id, fbx_node);
						}
					}

					const FBXDocParser::ShapeGeometry *shape_geometry = dynamic_cast<const FBXDocParser::ShapeGeometry *>(mesh);
					if (shape_geometry != nullptr) {
						print_verbose("[doc] valid shape geometry converted");
					}
				}
			}

			Ref<FBXSkeleton> node_skeleton = fbx_node->skeleton_node;

			if (node_skeleton.is_valid()) {
				Skeleton3D *skel = node_skeleton->skeleton;
				fbx_node->godot_node = skel;
			} else if (mesh_node == nullptr) {
				fbx_node->godot_node = memnew(Node3D);
			} else {
				fbx_node->godot_node = mesh_node;
			}

			fbx_node->godot_node->set_name(fbx_node->node_name);

			// assign parent if valid
			if (fbx_node->fbx_parent.is_valid()) {
				fbx_node->fbx_parent->godot_node->add_child(fbx_node->godot_node);
				fbx_node->godot_node->set_owner(state.root_owner);
			}

			// Node Transform debug, set local xform data.
			fbx_node->godot_node->set_transform(get_unscaled_transform(fbx_node->pivot_transform->LocalTransform, state.scale));

			// populate our mesh node reference
			if (mesh_node != nullptr && mesh_data_precached.is_valid()) {
				mesh_data_precached->godot_mesh_instance = mesh_node;
			}
		}
	}

	for (KeyValue<uint64_t, Ref<FBXMeshData>> &mesh_data : state.renderer_mesh_data) {
		const uint64_t mesh_id = mesh_data.key;
		Ref<FBXMeshData> mesh = mesh_data.value;

		const FBXDocParser::MeshGeometry *mesh_geometry = p_document->GetObject(mesh_id)->Get<FBXDocParser::MeshGeometry>();

		ERR_CONTINUE_MSG(mesh->mesh_node.is_null(), "invalid mesh allocation");

		const FBXDocParser::Skin *mesh_skin = mesh_geometry->DeformerSkin();

		if (!mesh_skin) {
			continue; // safe to continue
		}

		//
		// Skin bone configuration
		//

		//
		// Get Mesh Node Xform only
		//
		//ERR_CONTINUE_MSG(!state.fbx_target_map.has(mesh_id), "invalid xform for the skin pose: " + itos(mesh_id));
		//Ref<FBXNode> mesh_node_xform_data = state.fbx_target_map[mesh_id];

		if (!mesh_skin) {
			continue; // not a deformer.
		}

		if (mesh_skin->Clusters().size() == 0) {
			continue; // possibly buggy mesh
		}

		// Lookup skin or create it if it's not found.
		Ref<Skin> skin;
		if (!state.MeshSkins.has(mesh_id)) {
			print_verbose("Created new skin");
			skin.instantiate();
			state.MeshSkins.insert(mesh_id, skin);
		} else {
			print_verbose("Grabbed skin");
			skin = state.MeshSkins[mesh_id];
		}

		for (const FBXDocParser::Cluster *cluster : mesh_skin->Clusters()) {
			// node or bone this cluster targets (in theory will only be a bone target)
			uint64_t skin_target_id = cluster->TargetNode()->ID();

			print_verbose("adding cluster [" + itos(cluster->ID()) + "] " + String(cluster->Name().c_str()) + " for target: [" + itos(skin_target_id) + "] " + String(cluster->TargetNode()->Name().c_str()));
			ERR_CONTINUE_MSG(!state.fbx_bone_map.has(skin_target_id), "no bone found by that ID? locator");

			const Ref<FBXBone> bone = state.fbx_bone_map[skin_target_id];
			const Ref<FBXSkeleton> skeleton = bone->fbx_skeleton;
			const Ref<FBXNode> skeleton_node = skeleton->fbx_node;

			skin->add_named_bind(
					bone->bone_name,
					get_unscaled_transform(
							skeleton_node->pivot_transform->GlobalTransform.affine_inverse() * cluster->TransformLink().affine_inverse(), state.scale));
		}

		print_verbose("cluster name / id: " + String(mesh_skin->Name().c_str()) + " [" + itos(mesh_skin->ID()) + "]");
		print_verbose("skeleton has " + itos(state.fbx_bone_map.size()) + " binds");
		print_verbose("fbx skin has " + itos(mesh_skin->Clusters().size()) + " binds");
	}

	// mesh data iteration for populating skeleton mapping
	for (KeyValue<uint64_t, Ref<FBXMeshData>> &mesh_data : state.renderer_mesh_data) {
		Ref<FBXMeshData> mesh = mesh_data.value;
		const uint64_t mesh_id = mesh_data.key;
		ImporterMeshInstance3D *mesh_instance = mesh->godot_mesh_instance;
		const int mesh_weights = mesh->max_weight_count;
		Ref<FBXSkeleton> skeleton;
		const bool valid_armature = mesh->valid_armature_id;
		const uint64_t armature = mesh->armature_id;

		if (mesh_weights > 0) {
			// this is a bug, it means the weights were found but the skeleton wasn't
			ERR_CONTINUE_MSG(!valid_armature, "[doc] fbx armature is missing");
		} else {
			continue; // safe to continue not a bug just a normal mesh
		}

		if (state.skeleton_map.has(armature)) {
			skeleton = state.skeleton_map[armature];
			print_verbose("[doc] armature mesh to skeleton mapping has been allocated");
		} else {
			print_error("[doc] unable to find armature mapping");
		}

		ERR_CONTINUE_MSG(!mesh_instance, "[doc] invalid mesh mapping for skeleton assignment");
		ERR_CONTINUE_MSG(skeleton.is_null(), "[doc] unable to resolve the correct skeleton but we have weights!");

		mesh_instance->set_skeleton_path(mesh_instance->get_path_to(skeleton->skeleton));
		print_verbose("[doc] allocated skeleton to mesh " + mesh_instance->get_name());

		// do we have a mesh skin for this mesh
		ERR_CONTINUE_MSG(!state.MeshSkins.has(mesh_id), "no skin found for mesh");

		Ref<Skin> mesh_skin = state.MeshSkins[mesh_id];

		ERR_CONTINUE_MSG(mesh_skin.is_null(), "invalid skin stored in map");
		print_verbose("[doc] allocated skin to mesh " + mesh_instance->get_name());
		mesh_instance->set_skin(mesh_skin);
	}

	// build skin and skeleton information
	print_verbose("[doc] Skeleton3D Bone count: " + itos(state.fbx_bone_map.size()));
	const FBXDocParser::FileGlobalSettings *FBXSettings = p_document->GlobalSettingsPtr();

	// Configure constraints
	// NOTE: constraints won't be added quite yet, we don't have a real need for them *yet*. (they can be supported later on)
	// const std::vector<uint64_t> fbx_constraints = p_document->GetConstraintStackIDs();

	// get the animation FPS
	float fps_setting = ImportUtils::get_fbx_fps(FBXSettings);

	// enable animation import, only if local animation is enabled
	if (state.enable_animation_import && (p_flags & IMPORT_ANIMATION)) {
		// document animation stack list - get by ID so we can unload any non used animation stack
		const std::vector<uint64_t> animation_stack = p_document->GetAnimationStackIDs();

		for (uint64_t anim_id : animation_stack) {
			FBXDocParser::LazyObject *lazyObject = p_document->GetObject(anim_id);
			const FBXDocParser::AnimationStack *stack = lazyObject->Get<FBXDocParser::AnimationStack>();

			if (stack != nullptr) {
				String animation_name = ImportUtils::FBXNodeToName(stack->Name());
				print_verbose("Valid animation stack has been found: " + animation_name);
				// ReferenceTime is the same for some animations?
				// LocalStop time is the start and end time
				float r_start = CONVERT_FBX_TIME(stack->ReferenceStart());
				float r_stop = CONVERT_FBX_TIME(stack->ReferenceStop());
				float start_time = CONVERT_FBX_TIME(stack->LocalStart());
				float end_time = CONVERT_FBX_TIME(stack->LocalStop());
				float duration = end_time - start_time;

				print_verbose("r_start " + rtos(r_start) + ", r_stop " + rtos(r_stop));
				print_verbose("start_time" + rtos(start_time) + " end_time " + rtos(end_time));
				print_verbose("anim duration : " + rtos(duration));

				// we can safely create the animation player
				if (state.animation_player == nullptr) {
					print_verbose("Creating animation player");
					state.animation_player = memnew(AnimationPlayer);
					state.root->add_child(state.animation_player, true);
					state.animation_player->set_owner(state.root_owner);
				}

				Ref<Animation> animation;
				animation.instantiate();
				animation->set_name(animation_name);
				animation->set_length(duration);

				print_verbose("Animation length: " + rtos(animation->get_length()) + " seconds");

				// i think assimp was duplicating things, this lets me know to just reference or ignore this to prevent duplicate information in tracks
				// this would mean that we would be doing three times as much work per track if my theory is correct.
				// this was not the case but this is a good sanity check for the animation handler from the document.
				// it also lets us know if the FBX specification massively changes the animation system, in theory such a change would make this show
				// an fbx specification error, so best keep it in
				// the overhead is tiny.
				Map<uint64_t, const FBXDocParser::AnimationCurve *> CheckForDuplication;

				const std::vector<const FBXDocParser::AnimationLayer *> &layers = stack->Layers();
				print_verbose("FBX Animation layers: " + itos(layers.size()));
				for (const FBXDocParser::AnimationLayer *layer : layers) {
					std::vector<const FBXDocParser::AnimationCurveNode *> node_list = layer->Nodes();
					print_verbose("Layer: " + ImportUtils::FBXNodeToName(layer->Name()) + ", " + " AnimCurveNode count " + itos(node_list.size()));

					// first thing to do here is that i need to first get the animcurvenode to a Vector3
					// we now need to put this into the track information for godot.
					// to do this we need to know which track is what?

					// target id, [ track name, [time index, vector] ]
					// new map needs to be [ track name, keyframe_data ]
					Map<uint64_t, Map<StringName, FBXTrack>> AnimCurveNodes;

					// struct AnimTrack {
					// 	// Animation track can be
					// 	// visible, T, R, S
					// 	Map<StringName, Map<uint64_t, Vector3> > animation_track;
					// };

					// Map<uint64_t, AnimTrack> AnimCurveNodes;

					// so really, what does this mean to make an animtion track.
					// we need to know what object the curves are for.
					// we need the target ID and the target name for the track reduction.

					FBXDocParser::Model::RotOrder quaternion_rotation_order = FBXDocParser::Model::RotOrder_EulerXYZ;

					// T:: R:: S:: Visible:: Custom::
					for (const FBXDocParser::AnimationCurveNode *curve_node : node_list) {
						// when Curves() is called the curves are actually read, we could replace this with our own ProcessDomConnection code here if required.
						// We may need to do this but ideally we use Curves
						// note: when you call this there might be a delay in opening it
						// uses mutable type to 'cache' the response until the AnimationCurveNode is cleaned up.
						std::map<std::string, const FBXDocParser::AnimationCurve *> curves = curve_node->Curves();
						const FBXDocParser::Object *object = curve_node->Target();
						const FBXDocParser::Model *target = curve_node->TargetAsModel();
						if (target == nullptr) {
							if (object != nullptr) {
								print_error("[doc] warning failed to find a target Model for curve: " + String(object->Name().c_str()));
							} else {
								//print_error("[doc] failed to resolve object");
								continue;
							}

							continue;
						} else {
							//print_verbose("[doc] applied rotation order: " + itos(target->RotationOrder()));
							quaternion_rotation_order = target->RotationOrder();
						}

						uint64_t target_id = target->ID();
						String target_name = ImportUtils::FBXNodeToName(target->Name());

						const FBXDocParser::PropertyTable *properties = curve_node;
						bool got_x = false, got_y = false, got_z = false;
						float offset_x = FBXDocParser::PropertyGet<float>(properties, "d|X", got_x);
						float offset_y = FBXDocParser::PropertyGet<float>(properties, "d|Y", got_y);
						float offset_z = FBXDocParser::PropertyGet<float>(properties, "d|Z", got_z);

						String curve_node_name = ImportUtils::FBXNodeToName(curve_node->Name());

						// Reduce all curves for this node into a single container
						// T, R, S is what we expect, although other tracks are possible
						// like for example visibility tracks.

						// We are not ordered here, we don't care about ordering, this happens automagically by godot when we insert with the
						// key time :), so order is unimportant because the insertion will happen at a time index
						// good to know: we do not need a list of these in another format :)
						//Map<String, Vector<const Assimp::FBX::AnimationCurve *> > unordered_track;

						// T
						// R
						// S
						// Map[String, List<VECTOR>]

						// So this is a reduction of the animation curve nodes
						// We build this as a lookup, this is essentially our 'animation track'
						//AnimCurveNodes.insert(curve_node_name, Map<uint64_t, Vector3>());

						// create the animation curve information with the target id
						// so the point of this makes a track with the name "T" for example
						// the target ID is also set here, this means we don't need to do anything extra when we are in the 'create all animation tracks' step
						FBXTrack &keyframe_map = AnimCurveNodes[target_id][StringName(curve_node_name)];

						if (got_x && got_y && got_z) {
							Vector3 default_value = Vector3(offset_x, offset_y, offset_z);
							keyframe_map.default_value = default_value;
							keyframe_map.has_default = true;
							//print_verbose("track name: " + curve_node_name);
							//print_verbose("xyz default: " + default_value);
						}
						// target id, [ track name, [time index, vector] ]
						// Map<uint64_t, Map<StringName, Map<uint64_t, Vector3> > > AnimCurveNodes;

						// we probably need the target id here.
						// so map[uint64_t map]...
						// Map<uint64_t, Vector3D> translation_keys, rotation_keys, scale_keys;

						// extra const required by C++11 colon/Range operator
						// note: do not use C++17 syntax here for dicts.
						// this is banned in Godot.
						for (std::pair<const std::string, const FBXDocParser::AnimationCurve *> &kvp : curves) {
							const String curve_element = ImportUtils::FBXNodeToName(kvp.first);
							const FBXDocParser::AnimationCurve *curve = kvp.second;
							String curve_name = ImportUtils::FBXNodeToName(curve->Name());
							uint64_t curve_id = curve->ID();

							if (CheckForDuplication.has(curve_id)) {
								print_error("(FBX spec changed?) We found a duplicate curve being used for an alternative node - report to godot issue tracker");
							} else {
								CheckForDuplication.insert(curve_id, curve);
							}

							// FBX has no name for AnimCurveNode::, most of the time, not seen any with valid name here.
							const std::map<int64_t, float> &track_time = curve->GetValueTimeTrack();

							if (track_time.size() > 0) {
								for (std::pair<int64_t, float> keyframe : track_time) {
									if (curve_element == "d|X") {
										keyframe_map.keyframes[keyframe.first].x = keyframe.second;
									} else if (curve_element == "d|Y") {
										keyframe_map.keyframes[keyframe.first].y = keyframe.second;
									} else if (curve_element == "d|Z") {
										keyframe_map.keyframes[keyframe.first].z = keyframe.second;
									} else {
										//print_error("FBX Unsupported element: " + curve_element);
									}

									//print_verbose("[" + itos(target_id) + "] Keyframe added:  " + itos(keyframe_map.size()));

									//print_verbose("Keyframe t:" + rtos(animation_track_time) + " v: " + rtos(keyframe.second));
								}
							}
						}
					}

					// Map<uint64_t, Map<StringName, Map<uint64_t, Vector3> > > AnimCurveNodes;
					// add this animation track here

					// target id, [ track name, [time index, vector] ]
					//std::map<uint64_t, std::map<StringName, FBXTrack > > AnimCurveNodes;
					for (KeyValue<uint64_t, Map<StringName, FBXTrack>> &track : AnimCurveNodes) {
						// 5 tracks
						// current track index
						// track count is 5
						// track count is 5.
						// next track id is 5.
						const uint64_t target_id = track.key;

						Ref<FBXBone> bone;

						// note we must not run the below code if the entry doesn't exist, it will create dummy entries which is very bad.
						// remember that state.fbx_bone_map[target_id] will create a new entry EVEN if you only read.
						// this would break node animation targets, so if you change this be warned. :)
						if (state.fbx_bone_map.has(target_id)) {
							bone = state.fbx_bone_map[target_id];
						}

						Transform3D target_transform;

						if (state.fbx_target_map.has(target_id)) {
							Ref<FBXNode> node_ref = state.fbx_target_map[target_id];
							target_transform = node_ref->pivot_transform->GlobalTransform;
							//print_verbose("[doc] allocated animation node transform");
						}

						//int size_targets = state.fbx_target_map.size();
						//print_verbose("Target ID map: " + itos(size_targets));
						//print_verbose("[doc] debug bone map size: " + itos(state.fbx_bone_map.size()));

						// if this is a skeleton mapped track we can just set the path for the track.
						// todo: implement node paths here at some
						NodePath track_path;
						if (state.fbx_bone_map.size() > 0 && state.fbx_bone_map.has(target_id)) {
							if (bone->fbx_skeleton.is_valid() && bone.is_valid()) {
								Ref<FBXSkeleton> fbx_skeleton = bone->fbx_skeleton;
								String bone_path = state.root->get_path_to(fbx_skeleton->skeleton);
								bone_path += ":" + fbx_skeleton->skeleton->get_bone_name(bone->godot_bone_id);
								print_verbose("[doc] track bone path: " + bone_path);
								track_path = bone_path;
							}
						} else if (state.fbx_target_map.has(target_id)) {
							//print_verbose("[doc] we have a valid target for a node animation");
							Ref<FBXNode> target_node = state.fbx_target_map[target_id];
							if (target_node.is_valid() && target_node->godot_node != nullptr) {
								String node_path = state.root->get_path_to(target_node->godot_node);
								track_path = node_path;
								//print_verbose("[doc] node animation path: " + node_path);
							}
						} else {
							// note: this could actually be unsafe this means we should be careful about continuing here, if we see bizarre effects later we should disable this.
							// I am not sure if this is unsafe or not, testing will tell us this.
							print_error("[doc] invalid fbx target detected for this track");
							continue;
						}

						// everything in FBX and Maya is a node therefore if this happens something is seriously broken.
						if (!state.fbx_target_map.has(target_id)) {
							print_error("unable to resolve this to an FBX object.");
							continue;
						}

						Ref<FBXNode> target_node = state.fbx_target_map[target_id];
						const FBXDocParser::Model *model = target_node->fbx_model;
						const FBXDocParser::PropertyTable *props = dynamic_cast<const FBXDocParser::PropertyTable *>(model);

						Map<StringName, FBXTrack> &track_data = track.value;
						FBXTrack &translation_keys = track_data[StringName("T")];
						FBXTrack &rotation_keys = track_data[StringName("R")];
						FBXTrack &scale_keys = track_data[StringName("S")];

						double increment = 1.0f / fps_setting;
						double time = 0.0f;

						bool last = false;

						Vector<Vector3> pos_values;
						Vector<float> pos_times;
						Vector<Vector3> scale_values;
						Vector<float> scale_times;
						Vector<Quaternion> rot_values;
						Vector<float> rot_times;

						double max_duration = 0;
						double anim_length = animation->get_length();

						for (std::pair<int64_t, Vector3> position_key : translation_keys.keyframes) {
							pos_values.push_back(position_key.second * state.scale);
							double animation_track_time = CONVERT_FBX_TIME(position_key.first);

							if (animation_track_time > max_duration) {
								max_duration = animation_track_time;
							}

							//print_verbose("pos keyframe: t:" + rtos(animation_track_time) + " value " + position_key.second);
							pos_times.push_back(animation_track_time);
						}

						for (std::pair<int64_t, Vector3> scale_key : scale_keys.keyframes) {
							scale_values.push_back(scale_key.second);
							double animation_track_time = CONVERT_FBX_TIME(scale_key.first);

							if (animation_track_time > max_duration) {
								max_duration = animation_track_time;
							}
							//print_verbose("scale keyframe t:" + rtos(animation_track_time));
							scale_times.push_back(animation_track_time);
						}

						//
						// Pre and Post keyframe rotation handler
						// -- Required because Maya and Autodesk <3 the pain when it comes to implementing animation code! enjoy <3

						bool got_pre = false;
						bool got_post = false;

						Quaternion post_rotation;
						Quaternion pre_rotation;

						// Rotation matrix
						const Vector3 &PreRotation = FBXDocParser::PropertyGet<Vector3>(props, "PreRotation", got_pre);
						const Vector3 &PostRotation = FBXDocParser::PropertyGet<Vector3>(props, "PostRotation", got_post);

						FBXDocParser::Model::RotOrder rot_order = model->RotationOrder();
						if (got_pre) {
							pre_rotation = ImportUtils::EulerToQuaternion(rot_order, ImportUtils::deg2rad(PreRotation));
						}
						if (got_post) {
							post_rotation = ImportUtils::EulerToQuaternion(rot_order, ImportUtils::deg2rad(PostRotation));
						}

						Quaternion lastQuaternion = Quaternion();

						for (std::pair<int64_t, Vector3> rotation_key : rotation_keys.keyframes) {
							double animation_track_time = CONVERT_FBX_TIME(rotation_key.first);

							//print_verbose("euler rotation key: " + rotation_key.second);
							Quaternion rot_key_value = ImportUtils::EulerToQuaternion(quaternion_rotation_order, ImportUtils::deg2rad(rotation_key.second));

							if (lastQuaternion != Quaternion() && rot_key_value.dot(lastQuaternion) < 0) {
								rot_key_value.x = -rot_key_value.x;
								rot_key_value.y = -rot_key_value.y;
								rot_key_value.z = -rot_key_value.z;
								rot_key_value.w = -rot_key_value.w;
							}
							// pre_post rotation possibly could fix orientation
							Quaternion final_rotation = pre_rotation * rot_key_value * post_rotation;

							lastQuaternion = final_rotation;

							if (animation_track_time > max_duration) {
								max_duration = animation_track_time;
							}

							rot_values.push_back(final_rotation.normalized());
							rot_times.push_back(animation_track_time);
						}

						bool valid_rest = false;
						Transform3D bone_rest;
						int skeleton_bone = -1;
						if (state.fbx_bone_map.has(target_id)) {
							if (bone.is_valid() && bone->fbx_skeleton.is_valid()) {
								skeleton_bone = bone->godot_bone_id;
								if (skeleton_bone >= 0) {
									bone_rest = bone->fbx_skeleton->skeleton->get_bone_rest(skeleton_bone);
									valid_rest = true;
								}
							}

							if (!valid_rest) {
								print_verbose("invalid rest!");
							}
						}

						const Vector3 def_pos = translation_keys.has_default ? (translation_keys.default_value * state.scale) : bone_rest.origin;
						const Quaternion def_rot = rotation_keys.has_default ? ImportUtils::EulerToQuaternion(quaternion_rotation_order, ImportUtils::deg2rad(rotation_keys.default_value)) : bone_rest.basis.get_rotation_quaternion();
						const Vector3 def_scale = scale_keys.has_default ? scale_keys.default_value : bone_rest.basis.get_scale();
						print_verbose("track defaults: p(" + def_pos + ") s(" + def_scale + ") r(" + def_rot + ")");

						int position_idx = -1;
						if (pos_values.size()) {
							position_idx = animation->get_track_count();
							animation->add_track(Animation::TYPE_POSITION_3D);
							animation->track_set_path(position_idx, track_path);
							animation->track_set_imported(position_idx, true);
						}

						int rotation_idx = -1;
						if (pos_values.size()) {
							rotation_idx = animation->get_track_count();
							animation->add_track(Animation::TYPE_ROTATION_3D);
							animation->track_set_path(rotation_idx, track_path);
							animation->track_set_imported(rotation_idx, true);
						}

						int scale_idx = -1;
						if (pos_values.size()) {
							scale_idx = animation->get_track_count();
							animation->add_track(Animation::TYPE_SCALE_3D);
							animation->track_set_path(scale_idx, track_path);
							animation->track_set_imported(scale_idx, true);
						}

						while (true) {
							Vector3 pos = def_pos;
							Quaternion rot = def_rot;
							Vector3 scale = def_scale;

							if (pos_values.size()) {
								pos = _interpolate_track<Vector3>(pos_times, pos_values, time,
										AssetImportAnimation::INTERP_LINEAR);
							}

							if (rot_values.size()) {
								rot = _interpolate_track<Quaternion>(rot_times, rot_values, time,
										AssetImportAnimation::INTERP_LINEAR);
							}

							if (scale_values.size()) {
								scale = _interpolate_track<Vector3>(scale_times, scale_values, time,
										AssetImportAnimation::INTERP_LINEAR);
							}

							if (position_idx >= 0) {
								animation->position_track_insert_key(position_idx, time, pos);
							}
							if (rotation_idx >= 0) {
								animation->rotation_track_insert_key(rotation_idx, time, rot);
							}
							if (scale_idx >= 0) {
								animation->scale_track_insert_key(scale_idx, time, scale);
							}

							if (last) {
								break;
							}

							time += increment;
							if (time > anim_length) {
								last = true;
								time = anim_length;
								break;
							}
						}
					}
				}
				state.animation_player->add_animation(animation_name, animation);
			}
		}

		// AnimStack elements contain start stop time and name of animation
		// AnimLayer is the current active layer of the animation (multiple layers can be active we only support 1)
		// AnimCurveNode has a OP link back to the model which is the real node.
		// AnimCurveNode has a direct link to AnimationCurve (of which it may have more than one)

		// Store animation stack in list
		// iterate over all AnimStacks like the cache node algorithm recursively
		// this can then be used with ProcessDomConnection<> to link from
		// AnimStack:: <-- (OO) --> AnimLayer:: <-- (OO) --> AnimCurveNode:: (which can OP resolve) to Model::
	}

	//
	// Cleanup operations - explicit to prevent errors on shutdown - found that ref to ref does behave badly sometimes.
	//

	state.renderer_mesh_data.clear();
	state.MeshSkins.clear();
	state.fbx_target_map.clear();
	state.fbx_node_list.clear();

	for (KeyValue<uint64_t, Ref<FBXBone>> &element : state.fbx_bone_map) {
		Ref<FBXBone> bone = element.value;
		bone->parent_bone.unref();
		bone->node.unref();
		bone->fbx_skeleton.unref();
	}

	for (KeyValue<uint64_t, Ref<FBXSkeleton>> &element : state.skeleton_map) {
		Ref<FBXSkeleton> skel = element.value;
		skel->fbx_node.unref();
		skel->skeleton_bones.clear();
	}

	state.fbx_bone_map.clear();
	state.skeleton_map.clear();
	state.fbx_root_node.unref();

	return scene_root;
}

void EditorSceneFormatImporterFBX::BuildDocumentBones(Ref<FBXBone> p_parent_bone,
		ImportState &state, const FBXDocParser::Document *p_doc,
		uint64_t p_id) {
	const std::vector<const FBXDocParser::Connection *> &conns = p_doc->GetConnectionsByDestinationSequenced(p_id, "Model");
	// FBX can do an join like this
	// Model -> SubDeformer (bone) -> Deformer (skin pose)
	// This is important because we need to somehow link skin back to bone id in skeleton :)
	// The rules are:
	// A subdeformer will exist if 'limbnode' class tag present
	// The subdeformer will not necessarily have a deformer as joints do not have one
	for (const FBXDocParser::Connection *con : conns) {
		// goto: bone creation
		//print_verbose("con: " + String(con->PropertyName().c_str()));

		// ignore object-property links we want the object to object links nothing else
		if (con->PropertyName().length()) {
			continue;
		}

		// convert connection source object into Object base class
		const FBXDocParser::Object *const object = con->SourceObject();

		if (nullptr == object) {
			print_verbose("failed to convert source object for Model link");
			continue;
		}

		// FBX Model::Cube, Model::Bone001, etc elements
		// This detects if we can cast the object into this model structure.
		const FBXDocParser::Model *const model = dynamic_cast<const FBXDocParser::Model *>(object);

		// declare our bone element reference (invalid, unless we create a bone in this step)
		// this lets us pass valid armature information into children objects and this is why we moved this up here
		// previously this was created .instantiated() on the same line.
		Ref<FBXBone> bone_element;

		if (model != nullptr) {
			// model marked with limb node / casted.
			const FBXDocParser::ModelLimbNode *const limb_node = dynamic_cast<const FBXDocParser::ModelLimbNode *>(model);
			if (limb_node != nullptr) {
				// Write bone into bone list for FBX

				ERR_FAIL_COND_MSG(state.fbx_bone_map.has(limb_node->ID()), "[serious] duplicate LimbNode detected");

				bool parent_is_bone = state.fbx_bone_map.find(p_id);
				bone_element.instantiate();

				// used to build the bone hierarchy in the skeleton
				bone_element->parent_bone_id = parent_is_bone ? p_id : 0;
				bone_element->valid_parent = parent_is_bone;
				bone_element->limb_node = limb_node;

				// parent is a node and this is the first bone
				if (!parent_is_bone) {
					uint64_t armature_id = p_id;
					bone_element->valid_armature_id = true;
					bone_element->armature_id = armature_id;
					print_verbose("[doc] valid armature has been configured for first child: " + itos(armature_id));
				} else if (p_parent_bone.is_valid()) {
					if (p_parent_bone->valid_armature_id) {
						bone_element->valid_armature_id = true;
						bone_element->armature_id = p_parent_bone->armature_id;
						print_verbose("[doc] bone has valid armature id:" + itos(bone_element->armature_id));
					} else {
						print_error("[doc] unassigned armature id: " + String(limb_node->Name().c_str()));
					}
				} else {
					print_error("[doc] error is this a bone? " + String(limb_node->Name().c_str()));
				}

				if (!parent_is_bone) {
					print_verbose("[doc] Root bone: " + bone_element->bone_name);
				}

				uint64_t limb_id = limb_node->ID();
				bone_element->bone_id = limb_id;
				bone_element->bone_name = ImportUtils::FBXNodeToName(model->Name());
				bone_element->parent_bone = p_parent_bone;

				// insert limb by ID into list.
				state.fbx_bone_map.insert(limb_node->ID(), bone_element);
			}

			// recursion call - child nodes
			BuildDocumentBones(bone_element, state, p_doc, model->ID());
		}
	}
}

void EditorSceneFormatImporterFBX::BuildDocumentNodes(
		Ref<PivotTransform> parent_transform,
		ImportState &state,
		const FBXDocParser::Document *p_doc,
		uint64_t id,
		Ref<FBXNode> parent_node) {
	// tree
	// here we get the node 0 on the root by default
	const std::vector<const FBXDocParser::Connection *> &conns = p_doc->GetConnectionsByDestinationSequenced(id, "Model");

	// branch
	for (const FBXDocParser::Connection *con : conns) {
		// ignore object-property links
		if (con->PropertyName().length()) {
			// really important we document why this is ignored.
			print_verbose("ignoring property link - no docs on why this is ignored");
			continue;
		}

		// convert connection source object into Object base class
		// Source objects can exist with 'null connections' this means that we only for sure know the source exists.
		const FBXDocParser::Object *const source_object = con->SourceObject();

		if (nullptr == source_object) {
			print_verbose("failed to convert source object for Model link");
			continue;
		}

		// FBX Model::Cube, Model::Bone001, etc elements
		// This detects if we can cast the object into this model structure.
		const FBXDocParser::Model *const model = dynamic_cast<const FBXDocParser::Model *>(source_object);
		// model is the current node
		if (nullptr != model) {
			uint64_t current_node_id = model->ID();

			Ref<FBXNode> new_node;
			new_node.instantiate();
			new_node->current_node_id = current_node_id;
			new_node->node_name = ImportUtils::FBXNodeToName(model->Name());

			Ref<PivotTransform> fbx_transform;
			fbx_transform.instantiate();
			fbx_transform->set_parent(parent_transform);
			fbx_transform->set_model(model);
			fbx_transform->debug_pivot_xform("name: " + new_node->node_name);
			fbx_transform->Execute();

			new_node->set_pivot_transform(fbx_transform);

			// check if this node is a bone
			if (state.fbx_bone_map.has(current_node_id)) {
				Ref<FBXBone> bone = state.fbx_bone_map[current_node_id];
				if (bone.is_valid()) {
					bone->set_node(new_node);
					print_verbose("allocated bone data: " + bone->bone_name);
				}
			}

			// set the model, we can't just assign this safely
			new_node->set_model(model);

			if (parent_node.is_valid()) {
				new_node->set_parent(parent_node);
			} else {
				new_node->set_parent(state.fbx_root_node);
			}

			CRASH_COND_MSG(new_node->pivot_transform.is_null(), "invalid fbx target map pivot transform [serious]");

			// populate lookup tables with references
			// [fbx_node_id, fbx_node]

			state.fbx_node_list.push_back(new_node);
			if (!state.fbx_target_map.has(new_node->current_node_id)) {
				state.fbx_target_map[new_node->current_node_id] = new_node;
			}

			// print node name
			print_verbose("[doc] new node " + new_node->node_name);

			// sub branches
			BuildDocumentNodes(new_node->pivot_transform, state, p_doc, current_node_id, new_node);
		}
	}
}

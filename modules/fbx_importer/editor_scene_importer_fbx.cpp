/*************************************************************************/
/*  editor_scene_importer_fbx.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "core/io/image_loader.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/import/resource_importer_scene.h"
#include "tools/import_utils.h"

#include "data/fbx_skeleton.h"
#include "scene/3d/bone_attachment.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/surface_tool.h"

#include <assimp/Importer.hpp>

#include <code/FBX/FBXDocument.h>
#include <code/FBX/FBXImportSettings.h>
#include <code/FBX/FBXParser.h>
#include <code/FBX/FBXProperties.h>
#include <code/FBX/FBXTokenizer.h>
#include <thirdparty/assimp/code/FBX/FBXMeshGeometry.h>
#include <string>

void EditorSceneImporterFBX::get_extensions(List<String> *r_extensions) const {

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

void EditorSceneImporterFBX::_register_project_setting_import(const String generic,
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

uint32_t EditorSceneImporterFBX::get_import_flags() const {
	return IMPORT_SCENE;
}

void EditorSceneImporterFBX::_bind_methods() {
}
Node *EditorSceneImporterFBX::import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps,
		List<String> *r_missing_deps, Error *r_err) {
	// done for performance when re-importing lots of files when testing importer in verbose only!
	if (OS::get_singleton()->is_stdout_verbose()) {
		EditorLog *log = EditorNode::get_log();
		log->clear();
	}
	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);
	ERR_FAIL_COND_V(!f, NULL);

	PoolByteArray data;
	// broadphase tokenizing pass in which we identify the core
	// syntax elements of FBX (brackets, commas, key:value mappings)
	Assimp::FBX::TokenList tokens;

	bool is_binary = false;
	data.resize(f->get_len());
	f->get_buffer(data.write().ptr(), data.size());
	PoolByteArray fbx_header;
	fbx_header.resize(64);
	for (int32_t byte_i = 0; byte_i < 64; byte_i++) {
		fbx_header.write()[byte_i] = data.read()[byte_i];
	}

	String fbx_header_string;
	if (fbx_header.size() >= 0) {
		PoolByteArray::Read r = fbx_header.read();
		fbx_header_string.parse_utf8((const char *)r.ptr(), fbx_header.size());
	}

	print_verbose("[doc] opening fbx file: " + p_path);
	print_verbose("[doc] fbx header: " + fbx_header_string);

	// safer to check this way as there can be different formatted headers
	if (fbx_header_string.find("Kaydara FBX Binary", 0) != -1) {
		is_binary = true;
		print_verbose("[doc] is binary");
		Assimp::FBX::TokenizeBinary(tokens, (const char *)data.write().ptr(), (size_t)data.size());
	} else {
		print_verbose("[doc] is ascii");
		Assimp::FBX::Tokenize(tokens, (const char *)data.write().ptr());
	}

	// use this information to construct a very rudimentary
	// parse-tree representing the FBX scope structure
	Assimp::FBX::Parser parser(tokens, is_binary);
	Assimp::FBX::ImportSettings settings;

	// 'strict' mode is dangerous this causes more fun asserts to crash engine, so don't enable it.
	settings.strictMode = false;

	// take the raw parse-tree and convert it to a FBX DOM
	Assimp::FBX::Document doc(parser, settings);

	// safety for version handling
	if (doc.IsSafeToImport()) {
		return _generate_scene(p_path, &doc, p_flags, p_bake_fps, 8);
	} else {
		print_error("Cannot import file: " + p_path + " version of file is unsupported, please re-export in your modelling package file version is: " + itos(doc.FBXVersion()));
	}

	return memnew(Spatial);
}

template <class T>
struct EditorSceneImporterAssetImportInterpolate {

	T lerp(const T &a, const T &b, float c) const {

		return a + (b - a) * c;
	}

	T catmull_rom(const T &p0, const T &p1, const T &p2, const T &p3, float t) {

		float t2 = t * t;
		float t3 = t2 * t;

		return 0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4 * p2 - p3) * t2 +
							  (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
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
T EditorSceneImporterFBX::_interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time,
		AssetImportAnimation::Interpolation p_interp) {
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

MeshInstance *EditorSceneImporterFBX::create_fbx_mesh(Ref<FBXMeshVertexData> renderer_mesh_data, const Assimp::FBX::MeshGeometry *mesh_geometry, const Assimp::FBX::Model *model) {

	print_verbose("[doc] FBX creating godot mesh for: " + FBXNodeToName(model->Name()));
	if (renderer_mesh_data.is_valid()) {
		print_error("[doc] mesh has " + itos(renderer_mesh_data->max_weight_count) + " bone weights");
	}

	Ref<ArrayMesh> mesh;
	mesh.instance();

	Ref<SurfaceTool> st;
	st.instance();

	const std::vector<int> &material_indices = mesh_geometry->GetMaterialIndices();

	bool no_material_found = material_indices.size() == 0;

	if (no_material_found) {
		print_error("no material is configured for mesh " + FBXNodeToName(model->Name()));
	}
	
	std::vector<Vector3> vertices = mesh_geometry->GetVertices();
	AssimpUtils::AlignMeshAxes(vertices);
	std::vector<uint32_t> face_vertex_counts = mesh_geometry->GetFaceIndexCounts();

	// godot has two uv coordinate channels
	const std::vector<Vector2> &uv_coordinates_0 = mesh_geometry->GetTextureCoords(0);
	const std::vector<Vector2> &uv_coordinates_1 = mesh_geometry->GetTextureCoords(1);
	const std::vector<Vector3> &normals = mesh_geometry->GetNormals();
	const std::vector<Color> &vertex_colors = mesh_geometry->GetVertexColors(0);

	// material id, primitive_type(triangles,lines, points etc), SurfaceData
	Map<int, Map<uint32_t, FBXSplitBySurfaceVertexMapping> > surface_split_by_material_primitive;

	// we need to map faces back to vertexes
	{
		// Map Reduce Algorithm
		// The problem: reduce face primitives and also reduce material indices without duplicating vertexes :D
		// vertex count (1,2,3,4, etc), FBX Surface Data (uvs and indices for faces...)
		// yeah two sets.. uhuh
		// you read that correct. <3
		//Map<uint32_t, FBXSplitBySurfaceVertexMapping> primitive_geometry; // triangles, points, lines, quads

		// material id, indices list
		//Map<int, Vector<int>> material_surfaces;

		//		// fbx vertex id - value stored in the array is the material number
		//		for(uint32_t fbx_vertex_id = 0; fbx_vertex_id < material_indices.size(); fbx_vertex_id++) {
		//			const int material_id = material_indices[fbx_vertex_id];
		//			material_surfaces[material_id].push_back(fbx_vertex_id);
		//		}

		// Mesh face data - split based on geometry type
		uint32_t cursor = 0;
		for (uint32_t face_id = 0; face_id < face_vertex_counts.size(); face_id++) {
			uint32_t vertex_count = face_vertex_counts[face_id];
			for (uint32_t y = 0; y < vertex_count; y++) {

				// some files don't have these configured at all :P
				int material_id = 0;
				if (cursor < material_indices.size()) {
					material_id = material_indices[cursor];
				}

				FBXSplitBySurfaceVertexMapping &mapping = surface_split_by_material_primitive[material_id][vertex_count];
				mapping.vertex_id.push_back(cursor);

				// ensure we aren't outside available indexes, some will be
				if (cursor < uv_coordinates_0.size()) {
					mapping.add_uv_0(uv_coordinates_0[cursor]);
				}

				if (cursor < uv_coordinates_1.size()) {
					mapping.add_uv_1(uv_coordinates_1[cursor]);
				}

				if (cursor < normals.size()) {
					mapping.normals.push_back(normals[cursor]);
				}

				if (cursor < vertex_colors.size()) {
					mapping.colors.push_back(vertex_colors[cursor]);
				}

				cursor++; // we manually increment cursor, we are essentially creating a new mesh.
				// each surface split is a mesh
			}
		}
	}

	print_verbose("[vertex count for mesh] " + itos(vertices.size()));

	// triangles surface for triangles
	for (int material = 0; material < surface_split_by_material_primitive.size(); material++) {
		if (surface_split_by_material_primitive[material].has(3)) {
			FBXSplitBySurfaceVertexMapping &mapping = surface_split_by_material_primitive[material][3];
			Vector<size_t> &mesh_vertex_ids = mapping.vertex_id;
			//Vector<size_t> &vertex_weight_mapping = mapping.vertex_weight_id;
			st->begin(Mesh::PrimitiveType::PRIMITIVE_TRIANGLES);

			// stream in vertexes
			for (int x = 0; x < mesh_vertex_ids.size(); x++) {
				size_t vertex_id = mapping.vertex_id[x];
				GenFBXWeightInfo(renderer_mesh_data, mesh_geometry, st, vertex_id);

				mapping.GenerateSurfaceMaterial(st, vertex_id);
				st->add_vertex(vertices[vertex_id]);
			}

			for (int x = 0; x < mesh_vertex_ids.size(); x += 3) {
				st->add_index(x + 2);
				st->add_index(x + 1);
				st->add_index(x);
			}

			Array triangle_mesh = st->commit_to_arrays();
			triangle_mesh.resize(VS::ARRAY_MAX);
			Array morphs;

			mesh->add_surface_from_arrays(Mesh::PrimitiveType::PRIMITIVE_TRIANGLES, triangle_mesh, morphs);
		}

		if (surface_split_by_material_primitive[material].has(4)) {
			FBXSplitBySurfaceVertexMapping &mapping = surface_split_by_material_primitive[material][4];
			Vector<size_t> &mesh_vertex_ids = mapping.vertex_id;

			print_verbose("quads: " + itos(mesh_vertex_ids.size()));
			st->begin(Mesh::PrimitiveType::PRIMITIVE_TRIANGLES);

			// stream in vertexes
			for (int x = 0; x < mesh_vertex_ids.size(); x++) {
				size_t vertex_id = mesh_vertex_ids[x];
				GenFBXWeightInfo(renderer_mesh_data, mesh_geometry, st, vertex_id);

				mapping.GenerateSurfaceMaterial(st, vertex_id);
				//print_verbose("vert: " + quads[x]);
				st->add_vertex(vertices[vertex_id]);
			}

			//int cursor = 0;
			for (int x = 0; x < mesh_vertex_ids.size(); x += 4) {
				// complete first side of triangle
				st->add_index(x + 2);
				st->add_index(x + 1);
				st->add_index(x);

				// complete second side of triangle

				// top right
				// bottom right
				// top left

				// first triangle is
				// (x+2), (x+1), (x)
				// second triangle is
				// (x+2), (x), (x+3)

				st->add_index(x + 2);
				st->add_index(x);
				st->add_index(x + 3);

				// anti clockwise rotation in indices
				// note had to reverse right from left here
				// [0](x) bottom right (-1,-1)
				// [1](x+1) bottom left (1,-1)
				// [2](x+2) top left (1,1)
				// [3](x+3) top right (-1,1)

				// we have 4 points
				// we have 2 triangles
				// we have CCW
			}

			Array triangle_mesh = st->commit_to_arrays();
			triangle_mesh.resize(VS::ARRAY_MAX);
			Array morphs;
			mesh->add_surface_from_arrays(Mesh::PrimitiveType::PRIMITIVE_TRIANGLES, triangle_mesh, morphs);
		}
	}

	if (normals.size() > 0) {
		st->generate_tangents();
	}
	// const std::vector<uint32_t> &face_primitives = mesh_geometry->GetFaceIndexCounts();

	// uint32_t cursor;
	// for (uint32_t indice_count : face_primitives) {
	// 	if (indice_count == 1) {
	// 		st->add_index(0 + cursor);
	// 		cursor++;
	// 	} else if (indice_count == 2) {
	// 		st->add_index(0 + cursor);
	// 		st->add_index(1 + cursor);
	// 		cursor += 2;
	// 	} else if (indice_count == 3) {
	// 		st->add_index(0 + cursor);
	// 		st->add_index(0 + cursor);
	// 		st->add_index(3 + cursor);
	// 		st->add_index(2 + cursor);

	// 		cursor += 6;
	// 	}
	// }

	// Ref<SpatialMaterial> material;
	// material.instance();
	// material->set_cull_mode(SpatialMaterial::CullMode::CULL_DISABLED);

	// mesh->add_surface_from_arrays(Mesh::PRIMITIVE_POINTS, array_mesh, morphs);
	//mesh->surface_set_material(0, material);
	// okay now enable it
	mesh->set_name(FBXNodeToName(mesh_geometry->Name()));

	MeshInstance *godot_mesh = memnew(MeshInstance);
	godot_mesh->set_mesh(mesh);

	return godot_mesh;
}

void EditorSceneImporterFBX::GenFBXWeightInfo(Ref<FBXMeshVertexData> &renderer_mesh_data,
		const Assimp::FBX::MeshGeometry *mesh_geometry, Ref<SurfaceTool> st,
		size_t vertex_id) const {
	// check for valid mesh weight data mapping - this stores our vertex data
	// it's pre-cached in the skin caching so we need to read that before we make our meshes
	// which is currently the case.
	if (renderer_mesh_data.is_valid()) {

		// mesh is de-indexed by FBX Mesh class so id's from document aren't the same
		// this convention will rewrite weight vertex ids safely, but most importantly only once :)
		if (!renderer_mesh_data->valid_weight_indexes) {
			renderer_mesh_data->FixWeightData(mesh_geometry);
		}

		if (renderer_mesh_data->vertex_weights.has(vertex_id)) {
			Ref<VertexMapping> VertexWeights = renderer_mesh_data->vertex_weights[vertex_id];
			int weight_size = VertexWeights->weights.size();
			int max_weight_count = renderer_mesh_data->max_weight_count;

			if (weight_size > 0) {

				//print_error("initial count: " + itos(weight_size));
				if (VertexWeights->weights.size() < max_weight_count) {
					// missing weight count - how many do we not have?
					int missing_count = max_weight_count - weight_size;
					//print_verbose("adding missing count : " + itos(missing_count));
					for (int empty_weight_id = 0; empty_weight_id < missing_count; empty_weight_id++) {
						VertexWeights->weights.push_back(0); // no weight
						VertexWeights->bones.push_back(Ref<FBXBone>()); // invalid entry on purpose
					}
				}

				//print_error("final count: " + itos(VertexWeights->weights.size()));

				Vector<float> valid_weights;
				Vector<int> valid_bone_ids;

				VertexWeights->GetValidatedBoneWeightInfo(valid_bone_ids, valid_weights);

				st->add_weights(valid_weights);
				st->add_bones(valid_bone_ids);

				//print_verbose("[doc] triangle added weights to mesh for bones");
			}
		} else {
			//print_error("no weight data for vertex: " + itos(vertex_id));
		}
	}
}

void set_owner_recursive(Node *root, Node *current_node) {
	current_node->set_owner(root);

	for (int child_id = 0; child_id < current_node->get_child_count(); child_id++) {
		Node *child = current_node->get_child(child_id);
		set_owner_recursive(root, child); // recursive
	}
}

// tool which can get the global transform for a scene which isn't loaded.
Transform get_global_transform(Spatial *root, Spatial *child_node) {
	// state.root is armature and you are using this for an armature check.
	if (root == child_node) {
		return root->get_transform();
	}

	Transform t = Transform();
	Node *iter = child_node;

	while (iter != nullptr && iter != root) {
		Spatial *spatial = Object::cast_to<Spatial>(iter);
		if (spatial) {
			t *= spatial->get_transform();
		}

		iter = iter->get_parent();
	}

	return t;
}

Spatial *
EditorSceneImporterFBX::_generate_scene(const String &p_path,
		const Assimp::FBX::Document *p_document, const uint32_t p_flags,
		int p_bake_fps, const int32_t p_max_bone_weights) {

	ImportState state;
	state.path = p_path;
	state.animation_player = NULL;

	// create new root node for scene
	state.root = memnew(Spatial);
	state.fbx_root_node.instance();
	state.fbx_root_node->godot_node = state.root;

	Ref<FBXNode> root_node;
	root_node.instance();
	root_node->node_name = "root node";
	root_node->current_node_id = 0;
	root_node->godot_node = state.root;

	// cache this node onto the fbx_target map.
	state.fbx_target_map.insert(0, root_node);

	// cache basic node information from FBX document
	// grabs all FBX bones
	CacheNodeInformation(Ref<FBXBone>(), state, p_document, 0L);
	BuildDocumentNodes(nullptr, state, p_document, 0L, NULL);

	//print_verbose("[doc] debug fbx_bone_map size: " + itos(state.fbx_bone_map.size()));

	// build skin and skeleton information
	print_verbose("[doc] Skeleton Bone count: " + itos(state.fbx_bone_map.size()));

	// Importing bones using document based method from FBX directly
	// We do not use the assimp bone format to determine this information anymore.
	if (state.fbx_bone_map.size() > 0) {
		// We are using a single skeleton only method here
		// this is because we really have no concept of skeletons in FBX
		// their are bones in a scene but they have no specific armature
		// we can detect armatures but the issue lies in the complexity
		// we opted to merge the entire scene onto one skeleton for now
		// if we need to change this we have an archive of the old code.

		const std::vector<uint64_t> &bind_pose_ids = p_document->GetBindPoseIDs();

		for (uint64_t skin_id : bind_pose_ids) {

			Assimp::FBX::LazyObject *lazy_skin = p_document->GetObject(skin_id);
			const Assimp::FBX::FbxPose *active_skin = lazy_skin->Get<Assimp::FBX::FbxPose>();

			if (active_skin) {
				const std::vector<std::shared_ptr<Assimp::FBX::FbxPoseNode> > &bind_poses = active_skin->GetBindPoses();

				for (std::shared_ptr<Assimp::FBX::FbxPoseNode> pose_node : bind_poses) {
					Transform t = pose_node->GetBindPose();
					uint64_t fbx_node_id = pose_node->GetNodeID();
					if (state.fbx_bone_map.has(fbx_node_id)) {
						Ref<FBXBone> bone = state.fbx_bone_map[fbx_node_id];
						if (bone.is_valid()) {
							print_verbose("assigned skin pose from the file for bone " + bone->bone_name + ", transform: " + t);
							bone->pose_node = t;
							bone->assigned_pose_node = true;
						}
					}
				}
			}
		}

		// bind pose normally only has 1 per mesh but can have more than one
		// this is the point of skins
		// in FBX first bind pose is the master for the first skin

		// In order to handle the FBX skeleton we must also inverse any parent transforms on the bones
		// just to rule out any parent node transforms in the bone data
		// this is trivial to do and allows us to use the single skeleton method and merge them
		// this means that the nodes from maya kLocators will be preserved as bones
		// in the same rig without having to match this across skeletons and merge by detection
		// we can just merge and undo any parent transforms
		for (Map<uint64_t, Ref<FBXBone> >::Element *bone_element = state.fbx_bone_map.front(); bone_element; bone_element = bone_element->next()) {
			Ref<FBXBone> bone = bone_element->value();
			Ref<FBXSkeleton> fbx_skeleton_inst;

			uint64_t armature_id = bone->armature_id;
			if (state.skeleton_map.has(armature_id)) {
				fbx_skeleton_inst = state.skeleton_map[armature_id];
			} else {
				fbx_skeleton_inst.instance();
				state.skeleton_map.insert(armature_id, fbx_skeleton_inst);
			}

			print_verbose("populating skeleton with bone: " + bone->bone_name);

			//			// populate bone skeleton - since fbx has no DOM for the skeleton just a node.
			//			bone->bone_skeleton = fbx_skeleton_inst;

			// now populate bone on the armature node list
			fbx_skeleton_inst->skeleton_bones.push_back(bone);

			// we need to have a valid armature id and the model configured for the bone to be assigned fully.
			// happens once per skeleton
			if (state.fbx_target_map.has(armature_id) && !fbx_skeleton_inst->has_model()) {
				Ref<FBXNode> node = state.fbx_target_map[armature_id];

				fbx_skeleton_inst->set_model(node->get_model());
				fbx_skeleton_inst->fbx_node = node;
				print_verbose("allocated fbx skeleton primary / armature node for the level: " + node->node_name);
			} else if (!state.fbx_target_map.has(armature_id) && !fbx_skeleton_inst->has_model()) {
				print_error("bones are not mapped to an armature node for armature id: " + itos(armature_id) + " bone: " + bone->bone_name);
				// this means bone will be removed and not used, which is safe actually and no skeleton will be created.
			}
		}

		// setup skeleton instances if required :)
		for (Map<uint64_t, Ref<FBXSkeleton> >::Element *skeleton_node = state.skeleton_map.front(); skeleton_node; skeleton_node = skeleton_node->next()) {
			skeleton_node->value()->init_skeleton(state);
		}
	}

	// build godot node tree
	if (state.fbx_node_list.size() > 0) {
		for (List<Ref<FBXNode> >::Element *node_element = state.fbx_node_list.front();
				node_element;
				node_element = node_element->next()) {
			Ref<FBXNode> fbx_node = node_element->get();

			MeshInstance *mesh_node = nullptr;

			Ref<FBXMeshVertexData> mesh_data_precached;

			// check for valid geometry
			if (fbx_node->fbx_model == nullptr) {
				print_error("[doc] fundamental flaw, submit bug immediately with full import log with verbose logging on");
			} else {
				const std::vector<const Assimp::FBX::Geometry *> &geometry = fbx_node->fbx_model->GetGeometry();
				for (const Assimp::FBX::Geometry *mesh : geometry) {
					print_verbose("[doc !important] found valid mesh: " + FBXNodeToName(mesh->Name()));
					if (mesh == nullptr)
						continue;

					const Assimp::FBX::MeshGeometry *mesh_geometry = dynamic_cast<const Assimp::FBX::MeshGeometry *>(mesh);
					if (mesh != nullptr && mesh_geometry != NULL) {
						uint64_t mesh_id = mesh_geometry->ID();
						get_mesh_data(state, fbx_node, mesh_data_precached, mesh_geometry);
						// mesh node, mesh id
						mesh_node = create_fbx_mesh(mesh_data_precached, mesh_geometry, fbx_node->fbx_model);

						if (!state.MeshNodes.has(mesh_id)) {
							print_verbose("caching skin creation call for later");
							state.MeshNodes.insert(mesh_id, fbx_node);
						}
					}

					const Assimp::FBX::ShapeGeometry *shape_geometry = dynamic_cast<const Assimp::FBX::ShapeGeometry *>(mesh);
					if (shape_geometry != nullptr && shape_geometry != NULL) {
						print_verbose("[doc] valid shape geometry converted");
					}
				}
			}

			Ref<FBXSkeleton> node_skeleton = fbx_node->skeleton_node;

			if (node_skeleton.is_valid()) {
				Skeleton *skel = node_skeleton->skeleton;
				fbx_node->godot_node = skel;
			} else if (mesh_node == nullptr) {
				fbx_node->godot_node = memnew(Spatial);
			} else {
				fbx_node->godot_node = mesh_node;
			}

			fbx_node->godot_node->set_name(fbx_node->node_name);

			// assign parent if valid
			if (fbx_node->fbx_parent.is_valid()) {
				fbx_node->fbx_parent->godot_node->add_child(fbx_node->godot_node);
				fbx_node->godot_node->set_owner(state.root);
			}

			// set local xform data
			fbx_node->godot_node->set_transform(fbx_node->pivot_transform->LocalTransform);

			// populate our mesh node reference
			if (mesh_node != nullptr && mesh_data_precached.is_valid()) {
				mesh_data_precached->godot_mesh_instance = mesh_node;
			}
		}
	}

	for (Map<uint64_t, Ref<FBXNode> >::Element *skin_mesh = state.MeshNodes.front(); skin_mesh; skin_mesh = skin_mesh->next()) {
		create_mesh_data_skin(state, skin_mesh->value(), skin_mesh->key());
	}

	// mesh data iteration for populating skeleton mapping
	for (Map<uint64_t, Ref<FBXMeshVertexData> >::Element *mesh_data = state.renderer_mesh_data.front(); mesh_data; mesh_data = mesh_data->next()) {
		Ref<FBXMeshVertexData> mesh = mesh_data->value();
		MeshInstance *mesh_instance = mesh->godot_mesh_instance;
		int mesh_weights = mesh->max_weight_count;
		Ref<FBXSkeleton> skeleton;
		bool valid_armature = mesh->valid_armature_id;
		uint64_t armature = mesh->armature_id;

		if (mesh_weights > 0 && valid_armature) {
			if (state.skeleton_map.has(armature)) {
				skeleton = state.skeleton_map[armature];
				print_verbose("[doc] armature mesh to skeleton mapping has been allocated");
			} else {
				print_error("[doc] unable to find armature mapping");
			}

			if (skeleton.is_valid()) {
				mesh_instance->set_skeleton_path(mesh_instance->get_path_to(skeleton->skeleton));
				print_verbose("[doc] allocated skeleton to mesh " + mesh_instance->get_name());

				// do we have a mesh skin for this mesh
				if (state.MeshSkins.has(mesh->mesh_id)) {
					Ref<Skin> mesh_skin = state.MeshSkins[mesh->mesh_id];
					if (mesh_skin.is_valid()) {
						print_verbose("[doc] allocated skin to mesh " + mesh_instance->get_name());
						mesh_instance->set_skin(mesh_skin);
					}
				}
			}
		}
	}

	// build skin and skeleton information
	print_verbose("[doc] Skeleton Bone count: " + itos(state.fbx_bone_map.size()));

	const Assimp::FBX::FileGlobalSettings *FBXSettings = p_document->GlobalSettingsPtr();

	// Configure constraints
	const std::vector<uint64_t> fbx_constraints = p_document->GetConstraintStackIDs();

	for (uint64_t constraint_id : fbx_constraints) {
		Assimp::FBX::LazyObject *lazyObject = p_document->GetObject(constraint_id);
		const Assimp::FBX::Constraint *constraint = lazyObject->Get<Assimp::FBX::Constraint>();

		if (constraint != nullptr) {
			print_verbose("[doc] FBX constraint detected");

			// We are 'inverting' our search to do inverse lookup of the constraint id where it attached.
			// todo: make ProcessDomConnections rather than singular
			std::vector<const Assimp::FBX::Connection *> connections = p_document->GetConnectionsByDestinationSequenced(constraint_id, "Model");

			if (connections.size()) {
				uint64_t parent_id = 0;
				std::vector<uint64_t> children;
				bool valid_parent = false;
				for (const Assimp::FBX::Connection *connection : connections) {
					if (connection->prop == "Source (Parent)") {
						parent_id = connection->src;
						valid_parent = true;
					} else if (connection->prop == "Constrained object (Child)") {
						children.push_back(connection->src);
					}
				}

				if (valid_parent && children.size() > 0) {

					bool parent_is_bone = state.fbx_bone_map.has(parent_id);
					bool parent_is_node = state.fbx_target_map.has(parent_id);

					print_error("[doc] FBX Constraint found these are banned in godot: parent: " + itos(parent_id) + " is bone " + (parent_is_bone ? "yes" : "no") + " is node " + (parent_is_node ? "yes" : "no"));
					// We can add these later if we want, they just require a lot of work, the code below can be used for this
					// for (uint64_t child_id : children) {
					// 	bool child_is_bone = state.fbx_bone_map.has(child_id);
					// 	bool child_is_node = state.fbx_target_map.has(child_id);
					// 	print_verbose("[doc] -- child: " + itos(child_id) + " is bone " + (child_is_bone ? "yes" : "no") + " is node " + (child_is_node ? "yes" : "no"));

					// 	// please note check bone list before you check node list.
					// 	// bones are always in node list but not the other way around.
					// 	if (!child_is_bone && !parent_is_bone) {

					// 		if (parent_is_node && child_is_node) {
					// 			Ref<FBXNode> parent = state.fbx_target_map[parent_id];
					// 			Ref<FBXNode> child = state.fbx_target_map[child_id];

					// 			print_verbose("[doc] Node to node constraint parent: " + parent->node_name + " child: " + child->node_name);

					// 			if (parent.is_valid() && child.is_valid()) {
					// 				if (child->godot_node && parent->godot_node) {
					// 					Transform t = get_global_transform(state.root, child->godot_node);
					// 					Transform constraint_parent_global = get_global_transform(state.root, parent->godot_node);
					// 					Node *previous_parent_of_child = child->godot_node->get_parent();

					// 					previous_parent_of_child->remove_child(child->godot_node);
					// 					parent->godot_node->add_child(child->godot_node);

					// 					// update owner for all children - otherwise children of child are invisible
					// 					set_owner_recursive(state.root, child->godot_node);
					// 					Transform final_matrix = constraint_parent_global.inverse() * t;
					// 					child->godot_node->set_global_transform(final_matrix);
					// 					print_verbose("[doc] reparenting completed, child count of children: " + itos(child->godot_node->get_child_count()));
					// 				} else {
					// 					print_error("[doc] invalid node instancing in godot");
					// 				}

					// 			} else {
					// 				print_error("[doc] can't resolve invalid parent!");
					// 			}
					// 		}
					// 	} else if (parent_is_bone && !child_is_bone) {
					// 		// // // bone attachment mode
					// 		// Ref<FBXBone> parent = state.fbx_bone_map[parent_id];
					// 		// Ref<FBXNode> child = state.fbx_target_map[child_id];

					// 		// print_verbose("[doc] Bone to node constraint parent bone: " + parent->bone_name + " child node: " + child->node_name);

					// 		// if (parent.is_valid() && child.is_valid()) {
					// 		// 	if (child->godot_node) {
					// 		// 		Transform t = get_global_transform(state.root, child->godot_node);
					// 		// 		Transform bone_matrix = parent->rest_pose;
					// 		// 		//Spatial *previous_parent_of_child = Object::cast_to<Spatial>(child->godot_node->get_parent());

					// 		// 		//print_verbose("t: " + t.origin + " bone matrix: " + bone_matrix + " parent: " + get_global_transform(state.root, previous_parent_of_child).origin);

					// 		// 		//print_verbose("bone: " + bone_matrix.origin + ", inverse:" + bone_matrix.inverse().origin);
					// 		// 		//print_verbose("transform: " + t.origin + ", inverse: " + t.inverse().origin);

					// 		// BoneAttachment *parent_attachment = memnew(BoneAttachment);
					// 		// 		// parent_attachment->set_bone_name(state.skeleton->get_bone_name(parent->godot_bone_id));
					// 		// 		// state.skeleton->add_child(parent_attachment);
					// 		// 		// parent_attachment->set_owner(state.root);

					// 		// 		// previous_parent_of_child->remove_child(child->godot_node);
					// 		// 		// parent_attachment->add_child(child->godot_node);

					// 		// 		// // update owner for all children - otherwise children of child are invisible
					// 		// 		// set_owner_recursive(state.root, child->godot_node);
					// 		// 		// child->godot_node->set_global_transform(t);

					// 		// 		print_verbose("[doc] reparenting completed, child count of children: " + itos(child->godot_node->get_child_count()));
					// 		// 	} else {
					// 		// 		print_error("[doc] invalid node instancing in godot");
					// 		// 	}

					// 		// } else {
					// 		// 	print_error("[doc] can't resolve invalid parent!");
					// 		// }
					// 	}
					// }
				}
			}
		}
	}

	print_verbose("Constraints count: " + itos(fbx_constraints.size()));

	// get the animation FPS
	float fps_setting = AssimpUtils::get_fbx_fps(FBXSettings);

	if (p_flags & IMPORT_ANIMATION) {
		// document animation stack list - get by ID so we can unload any non used animation stack
		const std::vector<uint64_t> animation_stack = p_document->GetAnimationStackIDs();

		for (uint64_t anim_id : animation_stack) {
			Assimp::FBX::LazyObject *lazyObject = p_document->GetObject(anim_id);
			const Assimp::FBX::AnimationStack *stack = lazyObject->Get<Assimp::FBX::AnimationStack>();

			if (stack != nullptr) {
				String animation_name = FBXNodeToName(stack->Name());
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
					state.root->add_child(state.animation_player);
					state.animation_player->set_owner(state.root);
				}

				Ref<Animation> animation;
				animation.instance();
				animation->set_name(animation_name);
				animation->set_length(duration);

				print_verbose("Animation length: " + rtos(animation->get_length()) + " seconds");

				// i think assimp was duplicating things, this lets me know to just reference or ignore this to prevent duplicate information in tracks
				// this would mean that we would be doing three times as much work per track if my theory is correct.
				// this was not the case but this is a good sanity check for the animation handler from the document.
				// it also lets us know if the FBX specification massively changes the animation system, in theory such a change would make this show
				// an fbx specification error, so best keep it in
				// the overhead is tiny.
				Map<uint64_t, const Assimp::FBX::AnimationCurve *> CheckForDuplication;

				const std::vector<const Assimp::FBX::AnimationLayer *> &layers = stack->Layers();
				print_verbose("FBX Animation layers: " + itos(layers.size()));
				for (const Assimp::FBX::AnimationLayer *layer : layers) {
					std::vector<const Assimp::FBX::AnimationCurveNode *> node_list = layer->Nodes();
					print_verbose("Layer: " + FBXNodeToName(layer->Name()) + ", " + " AnimCurveNode count " + itos(node_list.size()));

					// first thing to do here is that i need to first get the animcurvenode to a Vector3
					// we now need to put this into the track information for godot.
					// to do this we need to know which track is what?

					// target id, [ track name, [time index, vector] ]
					std::map<uint64_t, std::map<StringName, std::map<uint64_t, Vector3> > > AnimCurveNodes;

					// struct AnimTrack {
					// 	// Animation track can be
					// 	// visible, T, R, S
					// 	Map<StringName, Map<uint64_t, Vector3> > animation_track;
					// };

					// Map<uint64_t, AnimTrack> AnimCurveNodes;

					// so really, what does this mean to make an animtion track.
					// we need to know what object the curves are for.
					// we need the target ID and the target name for the track reduction.

					Assimp::FBX::Model::RotOrder quat_rotation_order = Assimp::FBX::Model::RotOrder_EulerXYZ;

					for (const Assimp::FBX::AnimationCurveNode *curve_node : node_list) {
						// when Curves() is called the curves are actually read, we could replace this with our own ProcessDomConnection code here if required.
						// We may need to do this but ideally we use Curves
						// note: when you call this there might be a delay in opening it
						// uses mutable type to 'cache' the response until the AnimationCurveNode is cleaned up.
						std::map<std::string, const Assimp::FBX::AnimationCurve *> curves = curve_node->Curves();
						const Assimp::FBX::Object *object = curve_node->Target();
						const Assimp::FBX::Model *target = curve_node->TargetAsModel();
						if (target == nullptr) {
							if (object != nullptr) {
								print_error("[doc] warning failed to find a target Model for curve: " + String(object->Name().c_str()));
							} else {
								print_error("[doc] failed to resolve object");
							}

							continue;
						} else {
							//print_verbose("[doc] applied rotation order: " + itos(target->RotationOrder()));
							quat_rotation_order = target->RotationOrder();
						}

						uint64_t target_id = target->ID();
						String target_name = FBXNodeToName(target->Name());

						const Assimp::FBX::PropertyTable &properties = curve_node->Props();
						bool got_x, got_y, got_z;
						float offset_x = Assimp::FBX::PropertyGet<float>(properties, "d|X", got_x);
						float offset_y = Assimp::FBX::PropertyGet<float>(properties, "d|Y", got_y);
						float offset_z = Assimp::FBX::PropertyGet<float>(properties, "d|Z", got_z);

						// Reduce all curves for this node into a single container
						// T, R, S is what we expect, although other tracks are possible
						// like for example visiblity tracks.

						// We are not ordered here, we don't care about ordering, this happens automagically by godot when we insert with the
						// key time :), so order is unimportant because the insertion will happen at a time index
						// good to know: we do not need a list of these in another format :)
						//Map<String, Vector<const Assimp::FBX::AnimationCurve *> > unordered_track;
						String curve_node_name = FBXNodeToName(curve_node->Name());
						//print_verbose("[doc] Target: [" + itos(target_id) + "] " + target_name + " curve node: " + curve_node_name);

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
						//AnimCurveNodes.insert(target_id,
						//		Map<StringName, Map<uint64_t, Vector3> >(StringName(curve_node_name), Map<uint64_t, Vector3>()));
						std::map<uint64_t, Vector3> &keyframe_map = AnimCurveNodes[target_id][StringName(curve_node_name)];

						// 	// target id, [ track name, [time index, vector] ]
						// Map<uint64_t, Map<StringName, Map<uint64_t, Vector3> > > AnimCurveNodes;

						// we probably need the target id here.
						// so map[uint64_t map]...
						//Map<uint64_t, Vector3D> translation_keys, rotation_keys, scale_keys;

						// extra const required by C++11 colon/Range operator
						// note: do not use C++17 syntax here for dicts.
						// this is banned in Godot.
						for (std::pair<const std::string, const Assimp::FBX::AnimationCurve *> &kvp : curves) {
							String curve_element = FBXNodeToName(kvp.first);
							const Assimp::FBX::AnimationCurve *curve = kvp.second;
							String curve_name = FBXNodeToName(curve->Name());
							uint64_t curve_id = curve->ID();

							if (CheckForDuplication.has(curve_id)) {
								print_error("(FBX spec changed?) We found a duplicate curve being used for an alternative node - report to godot issue tracker");
							} else {
								CheckForDuplication.insert(curve_id, curve);
							}

							// FBX in general
							// we have no name for AnimCurveNode::, most of the time, not seen any with valid name here.
							// maya has names here
							//if (curve_name != "") {
							// detect tracks with unique identifiers maybe these are OP links?
							//	print_verbose("custom curve: " + curve_name + ", property: " + curve_element);

							//} else {
							// normal handler can execute here
							//unordered_track[curve_element].push_back(curve);
							const std::map<uint64_t, float> track_time = curve->GetValueTimeTrack();

							// T R S tracks to map<timeindex, Vector(x,y,z)>
							// EulerQuat(x,y,z) R tracks should be handled this way

							if (track_time.size() > 0) {

								Vector3 default_offset;

								if (got_x && got_y && got_z && curve_node_name != "S") {
									default_offset.x = offset_x;
									default_offset.y = offset_y;
									default_offset.z = offset_z;
									//print_verbose("found track default values:" + rtos(offset_x) + "," + rtos(offset_y) + ", " + rtos(offset_z));
								}
								//print_verbose("importing standard curve: property: " + curve_element + " track size: " + itos(track_time.size()));

								for (std::pair<uint64_t, float> keyframe : track_time) {
									//double animation_track_time = CONVERT_FBX_TIME(keyframe.first) * fps_setting;

									// curve element can be d|X, d|Y, d|Z
									if (curve_element == "d|X") {
										keyframe_map[keyframe.first].x = keyframe.second;
									} else if (curve_element == "d|Y") {
										keyframe_map[keyframe.first].y = keyframe.second;
									} else if (curve_element == "d|Z") {
										keyframe_map[keyframe.first].z = keyframe.second;
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
					for (std::pair<uint64_t, std::map<StringName, std::map<uint64_t, Vector3> > > track_element : AnimCurveNodes) {

						// 5 tracks
						// current track index
						// 4
						// track count is 5
						// track count is 5.
						// next track id is 5.
						const uint64_t target_id = track_element.first;

						int track_idx = animation->add_track(Animation::TYPE_TRANSFORM);

						// animation->track_set_path(track_idx, node_path);
						// animation->track_set_path(track_idx, node_path);
						Ref<FBXBone> bone;

						// note we must not run the below code if the entry doesn't exist, it will create dummy entries which is very bad.
						// remember that state.fbx_bone_map[target_id] will create a new entry EVEN if you only read.
						// this would break node animation targets, so if you change this be warned. :)
						if (state.fbx_bone_map.has(target_id)) {
							bone = state.fbx_bone_map[target_id];
						}

						Transform target_transform;

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
						if (state.fbx_bone_map.size() > 0 && state.fbx_bone_map.has(target_id)) {

							if (bone->fbx_skeleton.is_valid() && bone.is_valid()) {
								Ref<FBXSkeleton> fbx_skeleton = bone->fbx_skeleton;
								String bone_path = state.root->get_path_to(fbx_skeleton->skeleton);
								bone_path += ":" + fbx_skeleton->skeleton->get_bone_name(bone->godot_bone_id);
								print_verbose("[doc] track bone path: " + bone_path);
								NodePath path = bone_path;
								animation->track_set_path(track_idx, path);
							}
						} else if (state.fbx_target_map.has(target_id)) {
							//print_verbose("[doc] we have a valid target for a node animation");
							Ref<FBXNode> target_node = state.fbx_target_map[target_id];
							if (target_node.is_valid() && target_node->godot_node != nullptr) {
								String node_path = state.root->get_path_to(target_node->godot_node);
								NodePath path = node_path;
								animation->track_set_path(track_idx, path);
								//print_verbose("[doc] node animation path: " + node_path);
							}
						} else {
							// note: this could actually be unsafe this means we should be careful about continuing here, if we see bizzare effects later we should disable this.
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
						const Assimp::FBX::Model *model = target_node->fbx_model;
						const Assimp::FBX::PropertyTable &props = model->Props();

						// animation->track_set_name()

						// todo: We absolutely need the node path here

						std::map<uint64_t, Vector3> &translation_keys = track_element.second[StringName("T")];
						// now how do we convert vector3 to quat? :D
						std::map<uint64_t, Vector3> &rotation_keys = track_element.second[StringName("R")];
						std::map<uint64_t, Vector3> &scale_keys = track_element.second[StringName("S")];

						// Rotation key can be converted with AssimpUtils::EulerToQuaternion()

						// animation->add_track(Animation::TYPE_TRANSFORM);
						// //first determine animation length

						double increment = 1.0f / fps_setting;
						double time = 0.0f;

						bool last = false;

						Vector<Vector3> pos_values;
						Vector<float> pos_times;
						Vector<Vector3> scale_values;
						Vector<float> scale_times;
						Vector<Quat> rot_values;
						Vector<float> rot_times;

						double max_duration = 0;
						double anim_length = animation->get_length();

						for (std::pair<uint64_t, Vector3> position_key : translation_keys) {
							pos_values.push_back(AssimpUtils::FixAxisConversions(position_key.second));
							double animation_track_time = CONVERT_FBX_TIME(position_key.first);

							if (animation_track_time > max_duration) {
								max_duration = animation_track_time;
							}

							//print_verbose("pos keyframe: t:" + rtos(animation_track_time));
							pos_times.push_back(animation_track_time);
						}

						for (std::pair<uint64_t, Vector3> scale_key : scale_keys) {
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

						Quat post_rotation;
						Quat pre_rotation;

						// Rotation matrix
						const Vector3 &PreRotation = Assimp::FBX::PropertyGet<Vector3>(props, "PreRotation", got_pre);
						const Vector3 &PostRotation = Assimp::FBX::PropertyGet<Vector3>(props, "PostRotation", got_post);

						Assimp::FBX::Model::RotOrder rot_order = model->RotationOrder();
						if (got_pre) {
							pre_rotation = AssimpUtils::EulerToQuaternion(rot_order, PreRotation);
						}
						if (got_post) {
							post_rotation = AssimpUtils::EulerToQuaternion(rot_order, PostRotation);
						}

						Quat lastQuat = Quat();

						for (std::pair<uint64_t, Vector3> rotation_key : rotation_keys) {
							double animation_track_time = CONVERT_FBX_TIME(rotation_key.first);

							//print_verbose("euler rotation key: " + rotation_key.second);
							Quat rot_key_value = AssimpUtils::EulerToQuaternion(quat_rotation_order, rotation_key.second);

							if (lastQuat != Quat() && rot_key_value.dot(lastQuat) < 0) {
								rot_key_value.x = -rot_key_value.x;
								rot_key_value.y = -rot_key_value.y;
								rot_key_value.z = -rot_key_value.z;
								rot_key_value.w = -rot_key_value.w;
							}

							//Quat final_rotation = pre_rotation * rot_key_value * post_rotation.inverse();
							Quat final_rotation = rot_key_value;

							lastQuat = final_rotation;

							if (bone.is_valid()) {
							}
							if (animation_track_time > max_duration) {
								max_duration = animation_track_time;
							}

							rot_values.push_back(final_rotation);
							rot_times.push_back(animation_track_time);
						}

						while (true) {
							Vector3 pos = Vector3();
							Quat rot = Quat();
							Vector3 scale(1, 1, 1);

							if (pos_values.size()) {
								pos = _interpolate_track<Vector3>(pos_times, pos_values, time,
										AssetImportAnimation::INTERP_LINEAR);
							}

							if (rot_values.size()) {
								rot = _interpolate_track<Quat>(rot_times, rot_values, time,
										AssetImportAnimation::INTERP_LINEAR);
							}

							if (scale_values.size()) {
								scale = _interpolate_track<Vector3>(scale_times, scale_values, time,
										AssetImportAnimation::INTERP_LINEAR);
							}
							//
							//							// node animations must also include pivots
							if (state.fbx_bone_map.has(target_id)) {
								//print_verbose("this animation is for bone on skeleton: " + itos(target_id));

								if (bone.is_valid() && bone->fbx_skeleton.is_valid()) {
									int skeleton_bone = bone->godot_bone_id;
									Ref<FBXSkeleton> fbx_skeleton = bone->fbx_skeleton;
									String bone_name = fbx_skeleton->skeleton->get_bone_name(skeleton_bone);

									if (skeleton_bone >= 0) {
										Transform bone_matrix = bone->pivot_xform->GlobalTransform.affine_inverse() * bone->transform_link;

										Transform keyframe_matrix;
										keyframe_matrix.set_basis(Basis(rot, scale));
										keyframe_matrix.set_origin(pos);

										Transform final_matrix = bone_matrix;

										//AssimpUtils::debug_xform("keyframe", keyframe_matrix);
										//AssimpUtils::debug_xform("bone", bone_matrix);

										// populate this again
										rot = final_matrix.basis.get_rotation_quat() * rot;
										scale = final_matrix.basis.get_scale() * scale;
										pos = final_matrix.origin + pos;
									} else {
										print_error("failed to resolve target to generate pivot track for");
									}
								}
							}

							animation->transform_track_insert_key(track_idx, time, pos, rot, scale);

							if (last) {
								break;
							}

							time += increment;
							if (time > anim_length) {
								if (bone.is_valid()) {
									//print_verbose("max duration: " + rtos(max_duration));
									//print_verbose("last frame time " + rtos(time) + " anim len: " + rtos(animation->get_length()));
								}
								last = true;
								time = anim_length;
								break;
							}
						}

						//print_verbose("Animation track count: " + itos(AnimCurveNodes.size()));
					}
				}

				state.animation_player->add_animation(animation_name, animation);
			}

			//import_document_animation(p_document);
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

	for (Map<uint64_t, Ref<FBXBone> >::Element *element = state.fbx_bone_map.front(); element; element = element->next()) {
		Ref<FBXBone> bone = element->value();
		bone->parent_bone.unref();
		bone->pivot_xform.unref();
		bone->fbx_skeleton.unref();
	}

	for (Map<uint64_t, Ref<FBXSkeleton> >::Element *element = state.skeleton_map.front(); element; element = element->next()) {
		Ref<FBXSkeleton> skel = element->value();
		skel->fbx_node.unref();
		skel->skeleton_bones.clear();
	}

	state.fbx_bone_map.clear();
	state.skeleton_map.clear();
	state.fbx_root_node.unref();

	return state.root;
}

void EditorSceneImporterFBX::get_mesh_data(const ImportState &state, const Ref<FBXNode> &fbx_node, Ref<FBXMeshVertexData> &mesh_data_precached, const Assimp::FBX::MeshGeometry *mesh_geometry) const {
	uint64_t mesh_id = mesh_geometry->ID();
	if (state.renderer_mesh_data.has(mesh_id)) {

		// skins are poses for bones, they are overriding poses which have mesh offsets.
		// a skeleton is a structure made of bones
		// a skin is a pose relative to the mesh, it is a inverse matrix but it's important to mention
		// a skin overrides what is in the skeleton, if configured
		// a skin will never contain locator bones (kLocators from maya)
		// therefore the distinction is we need both, non negotiable.
		// this is what assimp cannot handle unfortunately (yet)

		//create_mesh_data_skin(state, fbx_node, mesh_id);

		mesh_data_precached = state.renderer_mesh_data[mesh_id];
		mesh_data_precached->mesh_id = mesh_id;
		print_verbose("[doc] valid mesh weight data has been cached for " + itos(mesh_id));
	}
}
void EditorSceneImporterFBX::create_mesh_data_skin(ImportState &state, const Ref<FBXNode> &fbx_node, uint64_t mesh_id) {
	if (!state.MeshSkins.has(mesh_id)) {
		print_verbose("[doc] caching skin for " + itos(mesh_id) + ", mesh node name: " + fbx_node->node_name);
		Ref<Skin> skin;
		skin.instance();

		for (Map<uint64_t, Ref<FBXBone> >::Element *elem = state.fbx_bone_map.front(); elem; elem = elem->next()) {
			Ref<FBXBone> bone = elem->value();
			Transform ignore_t;
			Ref<FBXSkeleton> skeleton = bone->fbx_skeleton;

			if (bone->get_link(state).is_valid()) {
				skin->add_named_bind(bone->bone_name, bone->get_vertex_skin_xform(state, fbx_node->pivot_transform->GlobalTransform));
			}
		}
		state.MeshSkins[mesh_id] = skin;
	}
}

void EditorSceneImporterFBX::CacheNodeInformation(Ref<FBXBone> p_parent_bone,
		ImportState &state, const Assimp::FBX::Document *p_doc,
		uint64_t p_id) {
	const std::vector<const Assimp::FBX::Connection *> &conns = p_doc->GetConnectionsByDestinationSequenced(p_id, "Model");
	// FBX can do an join like this
	// Model -> SubDeformer (bone) -> Deformer (skin pose)
	// This is important because we need to somehow link skin back to bone id in skeleton :)
	// The rules are:
	// A subdeformer will exist if 'limbnode' classtag present
	// The subdeformer will not neccisarily have a deformer as joints do not have one
	for (const Assimp::FBX::Connection *con : conns) {
		// goto: bone creation
		//print_verbose("con: " + String(con->PropertyName().c_str()));

		// ignore object-property links we want the object to object links nothing else
		if (con->PropertyName().length()) {
			continue;
		}

		// convert connection source object into Object base class
		const Assimp::FBX::Object *const object = con->SourceObject();

		if (nullptr == object) {
			print_verbose("failed to convert source object for Model link");
			continue;
		}

		// FBX Model::Cube, Model::Bone001, etc elements
		// This detects if we can cast the object into this model structure.
		const Assimp::FBX::Model *const model = dynamic_cast<const Assimp::FBX::Model *>(object);

		// declare our bone element reference (invalid, unless we create a bone in this step)
		// this lets us pass valid armature information into children objects and this is why we moved this up here
		// previously this was created .instanced() on the same line.
		Ref<FBXBone> bone_element;

		if (nullptr != model) {
			const Assimp::FBX::LimbNodeMaya *const limb_node = dynamic_cast<const Assimp::FBX::LimbNodeMaya *>(model);
			if (limb_node != nullptr) {
				// write bone into bone list for FBX
				if (!state.fbx_bone_map.has(limb_node->ID())) {
					bool parent_is_bone = state.fbx_bone_map.find(p_id);

					bone_element.instance();

					// used to build the bone hierarchy in the skeleton
					bone_element->parent_bone_id = parent_is_bone ? p_id : 0;
					bone_element->valid_parent = parent_is_bone;

					// armature handling

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
							print_verbose("[doc] bone valid armature id assigned: " + itos(bone_element->armature_id));
						} else {
							print_error("[doc] unassigned armature id: " + String(limb_node->Name().c_str()));
						}
					} else {
						print_error("[doc] error is this a bone? " + String(limb_node->Name().c_str()));
					}

					if (!parent_is_bone) {
						print_verbose("[doc] Root bone: " + bone_element->bone_name);
					}

					// ;Deformer::, Geometry::
					// C: "OO",2188874359888,2190730951072
					uint64_t limb_id = limb_node->ID();

					// Model "limb node" to SubDeformer to Deformer (skin)
					const Assimp::FBX::Cluster *deformer = ProcessDOMConnection<Assimp::FBX::Cluster>(p_doc, "Deformer", limb_id);

					bone_element->bone_name = FBXNodeToName(model->Name()) + "-" + itos(limb_id);
					bone_element->parent_bone = p_parent_bone;

					if (deformer != nullptr) {
						uint64_t mesh_target_id = 0;

						// inverse lookup of mesh from skin, this is because we like to reduce information down so we can re-use skins.
						const Assimp::FBX::Skin *skin = ProcessDOMConnection<Assimp::FBX::Skin>(p_doc, "Skin", deformer->ID());
						if (skin) {
							bone_element->skin = skin; // do not copy info out
							//print_verbose("[doc] valid skin found! searching for geometry that this skin applies to");
							const Assimp::FBX::Geometry *geo = ProcessDOMConnection<Assimp::FBX::Geometry>(p_doc, "Mesh", skin->ID());
							if (geo) {
								bone_element->geometry = geo;
								//print_verbose("[doc] valid mesh found!");
								mesh_target_id = geo->ID();
							} else {
								print_error("skin element is not assigned properly to mesh - unsupported");
							}

						} else {
							print_error("invalid skin layout / unsupported fbx file");
						}

						// if (mesh != nullptr) {
						// 	print_verbose("cluster mesh : " + String(mesh->Name().c_str()));
						// }
						print_verbose("[doc] Mesh Cluster: " + String(deformer->Name().c_str()) + ", " + deformer->TransformLink());
						// basic bone information
						print_verbose("fbx node: debug name: " + String(model->Name().c_str()) + "bone name: " + String(deformer->Name().c_str()));
						// assign FBX animation bind pose compensation data;
						bone_element->transform_link = deformer->TransformLink();
						bone_element->transform_matrix = deformer->GetTransform();
						bone_element->cluster = deformer;
						// skin configures target node ID.
						bone_element->target_node_id = deformer->TargetNode()->ID();
						bone_element->valid_target = true;
						bone_element->bone_id = limb_id;

						// apply the vertex weight information
						const std::vector<unsigned int> &indexes = deformer->GetIndices();
						const std::vector<float> &weights = deformer->GetWeights();

						// // convert to godot
						// for (size_t idx = 0; idx < indexes.size(); idx++) {
						// 	bone_element->VertexWeightInfo[indexes[idx]] = weights[idx];
						// }

						// Pipeline to move vertex weight information
						// to the renderer
						Ref<FBXMeshVertexData> mesh_vertex_data;
						if (state.renderer_mesh_data.has(mesh_target_id)) {
							mesh_vertex_data = state.renderer_mesh_data[mesh_target_id];
						} else {
							mesh_vertex_data.instance();
							state.renderer_mesh_data.insert(mesh_target_id, mesh_vertex_data);
						}

						// mesh vertex data retrieved now to stream this deformer data into the
						// internal mesh storage
						if (mesh_vertex_data.is_valid()) {
							// tell the mesh what Armature it should use
							mesh_vertex_data->armature_id = bone_element->armature_id;
							mesh_vertex_data->valid_armature_id = true;

							//print_verbose("storing mesh vertex data for mesh to use later");
							ERR_FAIL_COND_MSG(indexes.size() != weights.size(), "[doc] error mismatch between weight info");

							for (size_t idx = 0; idx < indexes.size(); idx++) {

								size_t vertex_index = indexes[idx];
								//print_verbose("vertex index: " + itos(vertex_index));

								Ref<VertexMapping> vertex_weight;
								if (mesh_vertex_data->vertex_weights.has(vertex_index)) {
									vertex_weight = mesh_vertex_data->vertex_weights[vertex_index];
									//print_verbose("grabbed pre-existing vertex index for " + itos(vertex_index));
								} else {
									vertex_weight.instance();
									mesh_vertex_data->vertex_weights.insert(vertex_index, vertex_weight);
									//print_verbose("created new vertex index for " + itos(vertex_index));
								}

								float influence_weight = weights[idx];

								vertex_weight->weights.push_back(influence_weight);
								vertex_weight->bones.push_back(bone_element);
								//print_verbose("Weight debug: " + rtos(influence_weight) + " bone id:" + bone_element->bone_name);
							}

							for (size_t idx = 0; idx < indexes.size(); idx++) {
								size_t vertex_index = indexes[idx];
								Ref<VertexMapping> vertex_weight;
								// if we have a weight we must count and check if its larger than our 'maximum value'
								if (mesh_vertex_data->vertex_weights.has(vertex_index)) {
									vertex_weight = mesh_vertex_data->vertex_weights[vertex_index];
									int influence_count = vertex_weight->weights.size();
									if (influence_count > mesh_vertex_data->max_weight_count) {
										mesh_vertex_data->max_weight_count = influence_count;
										mesh_vertex_data->valid_weight_count = true;
									}
								} else {
									continue;
								}
							}

							if (mesh_vertex_data.is_valid()) {
								if (mesh_vertex_data->max_weight_count > 8) {
									print_error("[doc] serious: maximum bone influences is 8 in this branch");
								}
								if (mesh_vertex_data->max_weight_count > 4) {
									mesh_vertex_data->max_weight_count = 8; // clamp to 8 bone vertex influences
									print_verbose("[doc] using 8 vertex bone influences configuration");
								} else {
									mesh_vertex_data->max_weight_count = 4;
									print_verbose("[doc] using 4 vertex bone influences configuration");
								}
							}
						}
					} else {
						print_verbose("[doc] warning: no skin for the deformer found: " + bone_element->bone_name);
					}

					// insert limb by ID into list.
					state.fbx_bone_map.insert(limb_node->ID(), bone_element);
				}
			}

			// recursion call - child nodes
			CacheNodeInformation(bone_element, state, p_doc, model->ID());
		}
	}
}

void EditorSceneImporterFBX::BuildDocumentNodes(
		Ref<PivotTransform> parent_transform,
		ImportState &state,
		const Assimp::FBX::Document *p_doc,
		uint64_t id,
		Ref<FBXNode> parent_node) {

	// tree
	// here we get the node 0 on the root by default
	const std::vector<const Assimp::FBX::Connection *> &conns = p_doc->GetConnectionsByDestinationSequenced(id, "Model");

	// branch
	for (const Assimp::FBX::Connection *con : conns) {

		// ignore object-property links
		if (con->PropertyName().length()) {
			// really important we document why this is ignored.
			print_verbose("ignoring property link - no docs on why this is ignored");
			continue;
		}

		// convert connection source object into Object base class
		// Source objects can exist with 'null connections' this means that we only for sure know the source exists.
		const Assimp::FBX::Object *const source_object = con->SourceObject();

		if (nullptr == source_object) {
			print_verbose("failed to convert source object for Model link");
			continue;
		}

		// FBX Model::Cube, Model::Bone001, etc elements
		// This detects if we can cast the object into this model structure.
		const Assimp::FBX::Model *const model = dynamic_cast<const Assimp::FBX::Model *>(source_object);

		// model is the current node
		if (nullptr != model) {
			uint64_t current_node_id = model->ID();

			Ref<FBXNode> new_node;
			new_node.instance();
			new_node->current_node_id = current_node_id;
			new_node->node_name = FBXNodeToName(model->Name());

			Ref<PivotTransform> fbx_transform;
			fbx_transform.instance();
			fbx_transform->set_parent(parent_transform);
			fbx_transform->set_model(model);
			fbx_transform->debug_pivot_xform("name: " + new_node->node_name);
			fbx_transform->Execute();

			new_node->set_pivot_transform(fbx_transform);

			// check if this node is a bone
			if (state.fbx_bone_map.has(current_node_id)) {
				Ref<FBXBone> bone = state.fbx_bone_map[current_node_id];
				if (bone.is_valid()) {
					bone->set_pivot_xform(fbx_transform);
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
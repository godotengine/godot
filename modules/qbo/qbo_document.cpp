/**************************************************************************/
/*  qbo_document.cpp                                                      */
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

#include "qbo_document.h"

#include "core/io/file_access.h"
#include "core/io/file_access_memory.h"
#include "modules/gltf/skin_tool.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/mesh.h"
#include "scene/resources/surface_tool.h"

Error QBODocument::_parse_qbo_data(Ref<FileAccess> f, Ref<GLTFState> p_state, uint32_t p_flags, String p_base_path, String p_path) {
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, "Cannot open QBO file.");
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);

	HashMap<String, int> bone_name_map;
	Vector<int> parent_stack;
	Vector<Transform3D> transform_stack;
	Vector<Vector<int>> node_children;
	Ref<GLTFAnimation> current_animation;
	bool in_hierarchy = false;
	bool in_motion = false;
	double frame_time = 0.03333333;
	int64_t frame_count = 0;
	int64_t frames_parsed = 0;
	Vector<Vector<String>> frame_data;

	Vector<Vector3> vertices;
	Vector<Vector3> normals;
	Vector<Vector2> uvs;
	Vector<Color> colors;
	HashMap<int, Vector4i> joint_indices_map;
	HashMap<int, Vector4> joint_weights_map;
	Ref<SurfaceTool> surf_tool = memnew(SurfaceTool);
	Ref<GLTFMesh> gltf_mesh;
	gltf_mesh.instantiate();
	String current_material;

	surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);

	print_verbose("Starting QBO parsing...");

	while (!f->eof_reached()) {
		String l = f->get_line().strip_edges();
		if (l.is_empty()) {
			continue;
		}

		if (l.begins_with("HIERARCHY")) {
			print_verbose("\n--- Entering HIERARCHY section ---");
			in_hierarchy = true;
			in_motion = false;
			parent_stack.clear();
			transform_stack.clear();
			bone_name_map.clear();
			p_state->nodes.resize(0);
			node_children.resize(0);
		} else if (l.begins_with("MOTION")) {
			print_verbose("\n--- Entering MOTION section ---");
			in_hierarchy = false;
			in_motion = true;
			current_animation.instantiate();
			current_animation->set_name("animation");
			frames_parsed = 0;
			print_verbose(vformat("Motion: Frame Count = %d, Frame Time = %f", frame_count, frame_time));
		} else if (in_hierarchy) {
			if (l.begins_with("End Site")) {
				int brace_count = 0;
				while (!f->eof_reached()) {
					String el = f->get_line().strip_edges();
					if (el == "{") {
						brace_count++;
					} else if (el == "}") {
						brace_count--;
						if (brace_count <= 0) {
							break;
						}
					}
				}
				continue;
			} else if (l.begins_with("ROOT") || l.begins_with("JOINT")) {
				Ref<GLTFNode> node;
				node.instantiate();
				node->joint = true;
				String original_bone_name = l.substr(l.find(" ") + 1).strip_edges();
				node->set_original_name(original_bone_name);

				if (bone_name_map.has(original_bone_name)) {
					ERR_FAIL_V_MSG(ERR_PARSE_ERROR, vformat("Bone name '%s' is not unique. Bone names must be unique.", original_bone_name));
				}

				node->set_name(original_bone_name);

				GLTFNodeIndex parent = -1;
				if (!parent_stack.is_empty()) {
					parent = parent_stack[parent_stack.size() - 1];
					node_children.write[parent].push_back(p_state->nodes.size());
				}
				node->set_parent(parent);

				print_verbose(vformat("Created %s node '%s' with parent index %d. Parent stack size: %d",
						l.begins_with("ROOT") ? "ROOT" : "JOINT", original_bone_name, parent, parent_stack.size()));
				if (parent != -1 && (parent < 0 || parent >= p_state->nodes.size())) {
					ERR_PRINT("ERROR: Invalid parent index detected during node creation!");
				}
				bone_name_map[original_bone_name] = p_state->nodes.size();
				p_state->nodes.push_back(node);
				node_children.push_back(Vector<int>());
				parent_stack.push_back(p_state->nodes.size() - 1);
				transform_stack.push_back(Transform3D());

				String brace_check = f->get_line().strip_edges();
				if (brace_check != "{") {
					ERR_FAIL_V_MSG(ERR_PARSE_ERROR,
							"Missing '{' after joint declaration");
				}
			} else if (l == "}") {
				if (!parent_stack.is_empty()) {
					parent_stack.remove_at(parent_stack.size() - 1);
					transform_stack.remove_at(transform_stack.size() - 1);
				}
			} else if (l.begins_with("OFFSET")) {
				Vector<String> parts = l.split(" ");
				if (parts.size() >= 4 && !p_state->nodes.is_empty()) {
					Vector3 offset(
							parts[1].to_float(),
							parts[2].to_float(),
							parts[3].to_float());
					Ref<GLTFNode> node = p_state->nodes[p_state->nodes.size() - 1];
					node->set_xform(Transform3D(Basis(), offset));
					print_verbose(vformat("Node %d '%s' offset set to: %s",
							p_state->nodes.size() - 1,
							node->get_name(),
							String(offset)));
				}
			} else if (l.begins_with("ORIENT")) {
				Vector<String> parts = l.split(" ");
				if (parts.size() >= 5 && !p_state->nodes.is_empty()) {
					Quaternion rot(
							parts[1].to_float(),
							parts[2].to_float(),
							parts[3].to_float(),
							parts[4].to_float());
					Ref<GLTFNode> node = p_state->nodes[p_state->nodes.size() - 1];
					Transform3D t = node->get_xform();
					t.basis = Basis(rot);
					node->set_xform(t);
					print_verbose(vformat("Node %d '%s' orientation set to: %s",
							p_state->nodes.size() - 1,
							node->get_name(),
							String(rot)));
				}
			}
		} else if (in_motion) {
			if (l.begins_with("Frames:")) {
				frame_count = l.get_slice(" ", 1).to_int();
			} else if (l.begins_with("Frame Time:")) {
				frame_time = l.get_slice(" ", 2).to_float();
			} else {
				if (frames_parsed < frame_count) {
					frame_data.push_back(l.split(" "));
					frames_parsed++;
					print_verbose(vformat("Parsed frame %d/%d (data points: %d)",
							frames_parsed,
							frame_count,
							frame_data.size()));
				} else {
					in_motion = false;
				}
			}
		} else {
			if (l.begins_with("v ")) {
				Vector<String> parts = l.split(" ");
				Vector3 vertex(
						parts[1].to_float(),
						parts[2].to_float(),
						parts[3].to_float());
				vertices.push_back(vertex);

				if (vertices.size() % 1000 == 0) {
					print_verbose(vformat("Parsed %d vertices...", vertices.size()));
				}

				if (parts.size() >= 7) {
					Color color(
							parts[4].to_float(),
							parts[5].to_float(),
							parts[6].to_float());
					colors.resize(vertices.size());
					colors.write[vertices.size() - 1] = color;
				}
			} else if (l.begins_with("vt ")) {
				Vector<String> parts = l.split(" ");
				Vector2 uv(
						parts[1].to_float(),
						1.0 - parts[2].to_float());
				uvs.push_back(uv);
			} else if (l.begins_with("vn ")) {
				Vector<String> parts = l.split(" ");
				Vector3 normal(
						parts[1].to_float(),
						parts[2].to_float(),
						parts[3].to_float());
				normals.push_back(normal);
			} else if (l.begins_with("vw ")) {
				Vector<String> parts = l.split(" ");
				int vert_index = parts[1].to_int() - 1;
				Vector4i joints;
				Vector4 weights;
				int count = 0;

				for (int i = 2; i < parts.size(); i += 2) {
					if (count >= 4) {
						break;
					}

					String bone_name = parts[i];
					float weight = parts[i + 1].to_float();

					String original_bone_name = parts[i];
					if (bone_name_map.has(original_bone_name)) {
						joints[count] = bone_name_map[original_bone_name];
						weights[count] = weight;
						count++;
					} else {
						ERR_PRINT(vformat("ERROR: Vertex %d references unknown bone '%s'",
								vert_index, original_bone_name));
					}
				}

				float total = weights.x + weights.y + weights.z + weights.w;
				if (total > 0) {
					weights /= total;
				}

				joint_indices_map[vert_index] = joints;
				joint_weights_map[vert_index] = weights;
				print_verbose(vformat("Vertex %d weights: %s (joints: %s)",
						vert_index,
						String(weights),
						String(joints)));
			} else if (l.begins_with("f ")) {
				Vector<String> face_verts = l.substr(2).split(" ");
				for (int vert_idx = face_verts.size() - 1; vert_idx >= 0; vert_idx--) {
					const String &vert = face_verts[vert_idx];
					Vector<String> components = vert.split("/");
					int pos_idx = components[0].to_int() - 1;

					Vector3 normal;
					Vector2 uv;
					Color color;
					Vector4i joints;
					Vector4 weights;

					if (pos_idx >= 0) {
						if (components.size() > 1 && !components[1].is_empty()) {
							uv = uvs[components[1].to_int() - 1];
						}
						if (components.size() > 2 && !components[2].is_empty()) {
							normal = normals[components[2].to_int() - 1];
						}
					}

					if (pos_idx >= 0 && pos_idx < colors.size()) {
						color = colors[pos_idx];
					}

					if (joint_indices_map.has(pos_idx)) {
						joints = joint_indices_map[pos_idx];
					}
					if (joint_weights_map.has(pos_idx)) {
						weights = joint_weights_map[pos_idx];
					}

					PackedInt32Array bone_array;
					PackedFloat32Array weight_array;

					bone_array.resize(4);
					weight_array.resize(4);

					for (int i = 0; i < 4; i++) {
						bone_array.set(i, i < 4 ? joints[i] : 0);
						weight_array.set(i, i < 4 ? weights[i] : 0.0f);
					}

					surf_tool->set_normal(normal);
					surf_tool->set_uv(uv);
					surf_tool->set_color(color);
					surf_tool->set_bones(bone_array);
					surf_tool->set_weights(weight_array);

					surf_tool->add_vertex(vertices[pos_idx]);
				}
				print_verbose(vformat("Added face with %d vertices (total indices: %d)",
						face_verts.size(),
						surf_tool->get_vertex_array().size()));
			} else if (l.begins_with("usemtl ")) {
				current_material = l.substr(7).strip_edges();
			} else if (l.begins_with("mtllib ")) {
				String mtl_path = l.substr(7).strip_edges();
				// Material loading implementation would go here
			}
		}
	}

	for (GLTFNodeIndex i = 0; i < p_state->nodes.size(); i++) {
		Ref<GLTFNode> node = p_state->nodes[i];
		node->children = node_children[i];
	}

	print_verbose("\n--- Node Hierarchy Validation ---");
	for (GLTFNodeIndex i = 0; i < p_state->nodes.size(); i++) {
		Ref<GLTFNode> node = p_state->nodes[i];
		GLTFNodeIndex parent = node->get_parent();
		if (parent != -1) {
			if (parent < 0 || parent >= p_state->nodes.size()) {
				ERR_PRINT(vformat("ERROR: Node %d '%s' has invalid parent index %d",
						i, node->get_name(), parent));
			} else {
				print_verbose(vformat("Node %d '%s' parent: %d (%s)",
						i, node->get_name(), parent, p_state->nodes[parent]->get_name()));
			}
		} else {
			print_verbose(vformat("Node %d '%s' is a root node", i, node->get_name()));
		}
	}

	if (current_animation.is_valid()) {
		print_verbose(vformat("\n--- Processing Animation: %d frames, %f FPS ---",
				frame_count,
				1.0 / frame_time));
		for (int frame_idx = 0; frame_idx < frame_count; frame_idx++) {
			if (frame_idx >= frame_data.size()) {
				break;
			}
			const Vector<String> &frame = frame_data[frame_idx];
			int data_idx = 0;

			for (GLTFNodeIndex node_idx = 0; node_idx < p_state->nodes.size(); node_idx++) {
				if (data_idx + 7 > frame.size()) {
					break;
				}

				float pos_x = frame[data_idx++].to_float();
				float pos_y = frame[data_idx++].to_float();
				float pos_z = frame[data_idx++].to_float();
				Vector3 position(pos_x, pos_y, pos_z);

				float rot_x = frame[data_idx++].to_float();
				float rot_y = frame[data_idx++].to_float();
				float rot_z = frame[data_idx++].to_float();
				float rot_w = frame[data_idx++].to_float();
				Quaternion rotation(rot_x, rot_y, rot_z, rot_w);

				double time = frame_idx * frame_time;
				GLTFAnimation::NodeTrack &track = current_animation->get_node_tracks()[node_idx];
				track.position_track.times.push_back(time);
				track.position_track.values.push_back(position);
				track.rotation_track.times.push_back(time);
				track.rotation_track.values.push_back(rotation);
			}
		}
		p_state->animations.push_back(current_animation);
		print_verbose(vformat("Added animation '%s' with %d tracks",
				current_animation->get_name(),
				current_animation->get_node_tracks().size()));
	}

	print_verbose("\n--- Skin/Joint Validation ---");
	print_verbose(vformat("Total vertices with skin data: %d", joint_indices_map.size()));
	for (const KeyValue<int, Vector4i> &E : joint_indices_map) {
		Vector4i joints = E.value;
		for (int i = 0; i < 4; i++) {
			int joint_idx = joints[i];
			if (joint_idx < 0 || joint_idx >= p_state->nodes.size()) {
				ERR_PRINT(vformat("ERROR: Vertex %d uses invalid joint index %d",
						E.key, joint_idx));
			}
		}
	}

	if (surf_tool->get_vertex_array().size() > 0) {
		print_verbose(vformat("\n--- Mesh Details ---\nVertices: %d\nNormals: %d\nUVs: %d\nColors: %d",
				vertices.size(),
				normals.size(),
				uvs.size(),
				colors.size()));

		if (normals.is_empty()) {
			print_verbose("Generating normals...");
			surf_tool->generate_normals();
		}
		if (uvs.size() == vertices.size()) {
			print_verbose("Generating tangents...");
			surf_tool->generate_tangents();
		}

		Ref<ImporterMesh> importer_mesh;
		importer_mesh.instantiate();

		surf_tool->index();
		Array surface_arrays = surf_tool->commit_to_arrays();

		importer_mesh->add_surface(
				Mesh::PRIMITIVE_TRIANGLES,
				surface_arrays);

		gltf_mesh->set_mesh(importer_mesh);
		gltf_mesh->set_original_name(p_path.get_file().get_basename());

		p_state->meshes.push_back(gltf_mesh);
		print_verbose(vformat("Created mesh with %d surfaces (total vertices: %d)",
				importer_mesh->get_surface_count(),
				surf_tool->get_vertex_array().size()));

		Vector<int> used_joints;
		for (const KeyValue<int, Vector4i> &E : joint_indices_map) {
			Vector4i joints = E.value;
			for (int i = 0; i < 4; i++) {
				int joint_idx = joints[i];
				if (joint_idx >= 0 && joint_idx < p_state->nodes.size() && !used_joints.has(joint_idx)) {
					used_joints.push_back(joint_idx);
				}
			}
		}

		if (!used_joints.is_empty()) {
			Vector<Transform3D> global_transforms;
			global_transforms.resize(p_state->nodes.size());

			for (GLTFNodeIndex i = 0; i < p_state->nodes.size(); i++) {
				Ref<GLTFNode> node = p_state->nodes[i];
				GLTFNodeIndex parent = node->get_parent();
				node->set_additional_data("GODOT_rest_transform", node->get_xform());
				if (parent == -1) {
					global_transforms.write[i] = node->get_xform();
				} else {
					global_transforms.write[i] = global_transforms[parent] * node->get_xform();
				}
			}

			TypedArray<Transform3D> inverse_binds;
			for (int joint_idx : used_joints) {
				if (joint_idx >= 0 && joint_idx < global_transforms.size()) {
					inverse_binds.push_back(global_transforms[joint_idx].affine_inverse());
				}
			}

			Ref<GLTFSkin> skin;
			skin.instantiate();
			skin->set_joints(used_joints);
			skin->set_joints_original(used_joints);
			skin->set_inverse_binds(inverse_binds);
			p_state->skins.push_back(skin);

			Ref<GLTFNode> mesh_node;
			mesh_node.instantiate();
			mesh_node->set_mesh(p_state->meshes.size() - 1);
			mesh_node->set_skin(p_state->skins.size() - 1);
			GLTFNodeIndex mesh_node_idx = p_state->nodes.size();
			p_state->nodes.push_back(mesh_node);
			mesh_node.instantiate();
			mesh_node->set_parent(p_state->nodes.size() - 1);
			p_state->nodes.push_back(mesh_node);
			mesh_node = p_state->nodes[p_state->nodes.size() - 1];

			if (!p_state->root_nodes.is_empty()) {
				GLTFNodeIndex root_idx = p_state->root_nodes[0];
				mesh_node->set_parent(root_idx);
				p_state->nodes.write[root_idx]->get_children().push_back(mesh_node_idx);
			} else {
				p_state->root_nodes.push_back(mesh_node_idx);
			}
		} else {
			Ref<GLTFNode> mesh_node;
			mesh_node.instantiate();
			mesh_node->set_mesh(p_state->meshes.size() - 1);
			GLTFNodeIndex mesh_node_idx = p_state->nodes.size();
			p_state->nodes.push_back(mesh_node);
			if (p_state->root_nodes.is_empty()) {
				p_state->root_nodes.push_back(mesh_node_idx);
			} else {
				GLTFNodeIndex root_idx = p_state->root_nodes[0];
				mesh_node->set_parent(root_idx);
				p_state->nodes.write[root_idx]->get_children().push_back(mesh_node_idx);
			}
		}
	}

	print_verbose(vformat("\n--- Final Structure ---\nNodes: %d\nRoot Nodes: %d\nSkins: %d\nMeshes: %d\nAnimations: %d",
			p_state->nodes.size(),
			p_state->root_nodes.size(),
			p_state->skins.size(),
			p_state->meshes.size(),
			p_state->animations.size()));

	print_verbose(vformat("\n--- Pre-skeleton Determination ---\nSkins: %d, Nodes: %d, Root Nodes: %d",
			p_state->skins.size(), p_state->nodes.size(), p_state->root_nodes.size()));
	if (p_state->root_nodes.is_empty()) {
		print_verbose("WARNING: No root nodes detected!");
	}

	_compute_node_heights(p_state);
	Error err = SkinTool::_determine_skeletons(p_state->skins, p_state->nodes, p_state->skeletons,
			p_state->get_import_as_skeleton_bones() ? p_state->root_nodes : Vector<GLTFNodeIndex>());
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	err = SkinTool::_create_skins(p_state->skins, p_state->nodes, p_state->use_named_skin_binds, p_state->unique_names);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	return OK;
}

Error QBODocument::append_from_file(String p_path, Ref<GLTFState> p_state, uint32_t p_flags, String p_base_path) {
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_FILE_NOT_FOUND);
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, "Cannot open QBO file.");
	return _parse_qbo_data(f, p_state, p_flags, p_base_path, p_path);
}

Error QBODocument::append_from_buffer(PackedByteArray p_bytes, String p_base_path, Ref<GLTFState> p_state, uint32_t p_flags) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_bytes.is_empty(), ERR_INVALID_PARAMETER);
	Ref<FileAccessMemory> memfile;
	memfile.instantiate();
	const Error open_error = memfile->open_custom(p_bytes.ptr(), p_bytes.size());
	ERR_FAIL_COND_V_MSG(open_error != OK, open_error, "Could not create memory file for QBO buffer.");
	ERR_FAIL_COND_V_MSG(memfile.is_null(), ERR_CANT_OPEN, "Cannot open QBO file.");
	return _parse_qbo_data(memfile, p_state, p_flags, p_base_path, String());
}

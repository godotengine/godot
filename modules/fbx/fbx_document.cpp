/**************************************************************************/
/*  fbx_document.cpp                                                      */
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

#include "fbx_document.h"

#include "core/config/project_settings.h"
#include "core/crypto/crypto_core.h"
#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/file_access_memory.h"
#include "core/io/image.h"
#include "core/io/stream_peer.h"
#include "core/math/color.h"
#include "scene/3d/bone_attachment_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/material.h"
#include "scene/resources/portable_compressed_texture.h"
#include "scene/resources/surface_tool.h"

#include "modules/gltf/extensions/gltf_light.h"
#include "modules/gltf/gltf_defines.h"
#include "modules/gltf/skin_tool.h"
#include "modules/gltf/structures/gltf_animation.h"
#include "modules/gltf/structures/gltf_camera.h"

#ifdef TOOLS_ENABLED
#include "editor/file_system/editor_file_system.h"
#endif

// FIXME: Hardcoded to avoid editor dependency.
#define FBX_IMPORT_USE_NAMED_SKIN_BINDS 16
#define FBX_IMPORT_DISCARD_MESHES_AND_MATERIALS 32
#define FBX_IMPORT_FORCE_DISABLE_MESH_COMPRESSION 64

#include "test/ufbx/ufbx.h"
#include "ufbx_write.h"

static size_t _file_access_read_fn(void *user, void *data, size_t size) {
	FileAccess *file = static_cast<FileAccess *>(user);
	return (size_t)file->get_buffer((uint8_t *)data, (uint64_t)size);
}

static bool _file_access_skip_fn(void *user, size_t size) {
	FileAccess *file = static_cast<FileAccess *>(user);
	file->seek(file->get_position() + size);
	return true;
}

static Vector2 _as_vec2(const ufbx_vec2 &p_vector) {
	return Vector2(real_t(p_vector.x), real_t(p_vector.y));
}

static Color _as_color(const ufbx_vec4 &p_vector) {
	return Color(real_t(p_vector.x), real_t(p_vector.y), real_t(p_vector.z), real_t(p_vector.w));
}

static Quaternion _as_quaternion(const ufbx_quat &p_quat) {
	return Quaternion(real_t(p_quat.x), real_t(p_quat.y), real_t(p_quat.z), real_t(p_quat.w));
}

static Transform3D _as_transform(const ufbx_transform &p_xform) {
	Transform3D result;
	result.origin = FBXDocument::_as_vec3(p_xform.translation);
	result.basis.set_quaternion_scale(_as_quaternion(p_xform.rotation), FBXDocument::_as_vec3(p_xform.scale));
	return result;
}

static real_t _relative_error(const Vector3 &p_a, const Vector3 &p_b) {
	return p_a.distance_to(p_b) / MAX(p_a.length(), p_b.length());
}

static Color _material_color(const ufbx_material_map &p_map) {
	if (p_map.value_components == 1) {
		float r = float(p_map.value_real);
		return Color(r, r, r);
	} else if (p_map.value_components == 3) {
		float r = float(p_map.value_vec3.x);
		float g = float(p_map.value_vec3.y);
		float b = float(p_map.value_vec3.z);
		return Color(r, g, b);
	} else {
		float r = float(p_map.value_vec4.x);
		float g = float(p_map.value_vec4.y);
		float b = float(p_map.value_vec4.z);
		float a = float(p_map.value_vec4.z);
		return Color(r, g, b, a);
	}
}

static Color _material_color(const ufbx_material_map &p_map, const ufbx_material_map &p_factor) {
	Color color = _material_color(p_map);
	if (p_factor.has_value) {
		float factor = float(p_factor.value_real);
		color.r *= factor;
		color.g *= factor;
		color.b *= factor;
	}
	return color;
}

static const ufbx_texture *_get_file_texture(const ufbx_texture *p_texture) {
	if (!p_texture) {
		return nullptr;
	}
	for (const ufbx_texture *texture : p_texture->file_textures) {
		if (texture->file_index != UFBX_NO_INDEX) {
			return texture;
		}
	}
	return nullptr;
}

static Ref<Image> _get_decompressed_image(Ref<Texture2D> texture) {
	if (texture.is_null()) {
		return Ref<Image>();
	}
	Ref<Image> image = texture->get_image();
	if (image.is_null()) {
		return Ref<Image>();
	}
	image = image->duplicate();
	image->decompress();
	return image;
}

static Vector<Vector2> _decode_vertex_attrib_vec2(const ufbx_vertex_vec2 &p_attrib, const Vector<uint32_t> &p_indices) {
	Vector<Vector2> ret;

	int num_indices = p_indices.size();
	ret.resize(num_indices);
	for (int i = 0; i < num_indices; i++) {
		ret.write[i] = _as_vec2(p_attrib[p_indices[i]]);
	}
	return ret;
}

static Vector<Vector3> _decode_vertex_attrib_vec3(const ufbx_vertex_vec3 &p_attrib, const Vector<uint32_t> &p_indices) {
	Vector<Vector3> ret;

	int num_indices = p_indices.size();
	ret.resize(num_indices);
	for (int i = 0; i < num_indices; i++) {
		ret.write[i] = FBXDocument::_as_vec3(p_attrib[p_indices[i]]);
	}
	return ret;
}

static Vector<float> _decode_vertex_attrib_vec3_as_tangent(const ufbx_vertex_vec3 &p_attrib, const Vector<uint32_t> &p_indices) {
	Vector<float> ret;

	int num_indices = p_indices.size();
	ret.resize(num_indices * 4);
	for (int i = 0; i < num_indices; i++) {
		Vector3 v = FBXDocument::_as_vec3(p_attrib[p_indices[i]]);
		ret.write[i * 4 + 0] = v.x;
		ret.write[i * 4 + 1] = v.y;
		ret.write[i * 4 + 2] = v.z;
		ret.write[i * 4 + 3] = 1.0f;
	}
	return ret;
}

static Vector<Color> _decode_vertex_attrib_color(const ufbx_vertex_vec4 &p_attrib, const Vector<uint32_t> &p_indices) {
	Vector<Color> ret;

	int num_indices = p_indices.size();
	ret.resize(num_indices);
	for (int i = 0; i < num_indices; i++) {
		ret.write[i] = _as_color(p_attrib[p_indices[i]]);
	}
	return ret;
}

static Vector3 _encode_vertex_index(uint32_t p_index) {
	return Vector3(real_t(p_index & 0xffff), real_t(p_index >> 16), 0.0f);
}

static uint32_t _decode_vertex_index(const Vector3 &p_vertex) {
	return uint32_t(p_vertex.x) | uint32_t(p_vertex.y) << 16;
}

static ufbx_skin_deformer *_find_skin_deformer(ufbx_skin_cluster *p_cluster) {
	for (const ufbx_connection &conn : p_cluster->element.connections_src) {
		ufbx_skin_deformer *deformer = ufbx_as_skin_deformer(conn.dst);
		if (deformer) {
			return deformer;
		}
	}
	return nullptr;
}

static String _find_element_name(ufbx_element *p_element) {
	if (p_element->name.length > 0) {
		return FBXDocument::_as_string(p_element->name);
	} else if (p_element->instances.count > 0) {
		return _find_element_name(&p_element->instances[0]->element);
	} else {
		return "";
	}
}

struct ThreadPoolFBX {
	struct Group {
		ufbx_thread_pool_context ctx = {};
		WorkerThreadPool::GroupID task_id = {};
		uint32_t start_index = 0;
	};

	WorkerThreadPool *pool = nullptr;
	Group groups[UFBX_THREAD_GROUP_COUNT] = {};
};

static void _thread_pool_task(void *user, uint32_t index) {
	ThreadPoolFBX::Group *group = (ThreadPoolFBX::Group *)user;
	ufbx_thread_pool_run_task(group->ctx, group->start_index + index);
}

static bool _thread_pool_init_fn(void *user, ufbx_thread_pool_context ctx, const ufbx_thread_pool_info *info) {
	ThreadPoolFBX *pool = (ThreadPoolFBX *)user;
	for (ThreadPoolFBX::Group &group : pool->groups) {
		group.ctx = ctx;
	}
	return true;
}

static void _thread_pool_run_fn(void *user, ufbx_thread_pool_context ctx, uint32_t group, uint32_t start_index, uint32_t count) {
	ThreadPoolFBX *pool = (ThreadPoolFBX *)user;
	ThreadPoolFBX::Group &pool_group = pool->groups[group];
	pool_group.start_index = start_index;
	pool_group.task_id = pool->pool->add_native_group_task(_thread_pool_task, &pool_group, (int)count, -1, true, "ufbx");
}

static void _thread_pool_wait_fn(void *user, ufbx_thread_pool_context ctx, uint32_t group, uint32_t max_index) {
	ThreadPoolFBX *pool = (ThreadPoolFBX *)user;
	pool->pool->wait_for_group_task_completion(pool->groups[group].task_id);
}

String FBXDocument::_gen_unique_name(HashSet<String> &unique_names, const String &p_name) {
	const String s_name = p_name.validate_node_name();

	String u_name;
	int index = 1;
	while (true) {
		u_name = s_name;

		if (index > 1) {
			u_name += itos(index);
		}
		if (!unique_names.has(u_name)) {
			break;
		}
		index++;
	}

	unique_names.insert(u_name);

	return u_name;
}

String FBXDocument::_sanitize_animation_name(const String &p_name) {
	String anim_name = p_name.validate_node_name();
	return AnimationLibrary::validate_library_name(anim_name);
}

String FBXDocument::_gen_unique_animation_name(Ref<FBXState> p_state, const String &p_name) {
	const String s_name = _sanitize_animation_name(p_name);

	String u_name;
	int index = 1;
	while (true) {
		u_name = s_name;

		if (index > 1) {
			u_name += itos(index);
		}
		if (!p_state->unique_animation_names.has(u_name)) {
			break;
		}
		index++;
	}

	p_state->unique_animation_names.insert(u_name);

	return u_name;
}

Error FBXDocument::_parse_scenes(Ref<FBXState> p_state) {
	p_state->unique_names.insert("Skeleton3D"); // Reserve skeleton name.

	const ufbx_scene *fbx_scene = p_state->scene.get();

	// TODO: Multi-document support, would need test files for structure
	p_state->scene_name = "";

	// TODO: Append the root node directly if we use root-based space conversion
	for (const ufbx_node *root_node : fbx_scene->root_node->children) {
		p_state->root_nodes.push_back(int(root_node->typed_id));
	}

	return OK;
}

Error FBXDocument::_parse_nodes(Ref<FBXState> p_state) {
	const ufbx_scene *fbx_scene = p_state->scene.get();

	for (int node_i = 0; node_i < static_cast<int>(fbx_scene->nodes.count); node_i++) {
		const ufbx_node *fbx_node = fbx_scene->nodes[node_i];

		Ref<GLTFNode> node;
		node.instantiate();

		node->height = int(fbx_node->node_depth);

		if (fbx_node->name.length > 0) {
			node->set_name(_as_string(fbx_node->name));
			node->set_original_name(node->get_name());
		} else if (fbx_node->is_root) {
			node->set_name("RootNode");
		}
		if (fbx_node->camera) {
			node->camera = fbx_node->camera->typed_id;
		}
		if (fbx_node->light) {
			node->light = fbx_node->light->typed_id;
		}
		if (fbx_node->mesh) {
			node->mesh = fbx_node->mesh->typed_id;
		}

		{
			node->transform = _as_transform(fbx_node->local_transform);

			bool found_rest_xform = false;
			bool bad_rest_xform = false;
			Transform3D candidate_rest_xform;

			if (fbx_node->parent) {
				// Attempt to resolve a rest pose for bones: This uses internal FBX connections to find
				// all skin clusters connected to the bone.
				for (const ufbx_connection &child_conn : fbx_node->element.connections_src) {
					ufbx_skin_cluster *child_cluster = ufbx_as_skin_cluster(child_conn.dst);
					if (!child_cluster) {
						continue;
					}
					ufbx_skin_deformer *child_deformer = _find_skin_deformer(child_cluster);
					if (!child_deformer) {
						continue;
					}

					// Found a skin cluster: Now iterate through all the skin clusters of the parent and
					// try to find one that used by the same deformer.
					for (const ufbx_connection &parent_conn : fbx_node->parent->element.connections_src) {
						ufbx_skin_cluster *parent_cluster = ufbx_as_skin_cluster(parent_conn.dst);
						if (!parent_cluster) {
							continue;
						}
						ufbx_skin_deformer *parent_deformer = _find_skin_deformer(parent_cluster);
						if (parent_deformer != child_deformer) {
							continue;
						}

						// Success: Found two skin clusters from the same deformer, now we can resolve the
						// local bind pose from the difference between the two world-space bind poses.
						ufbx_matrix child_to_world = child_cluster->bind_to_world;
						ufbx_matrix world_to_parent = ufbx_matrix_invert(&parent_cluster->bind_to_world);
						ufbx_matrix child_to_parent = ufbx_matrix_mul(&world_to_parent, &child_to_world);
						Transform3D xform = _as_transform(ufbx_matrix_to_transform(&child_to_parent));

						if (!found_rest_xform) {
							// Found the first bind pose for the node, assume that this one is good
							found_rest_xform = true;
							candidate_rest_xform = xform;
						} else if (!bad_rest_xform) {
							// Found another: Let's hope it's similar to the previous one, if not warn and
							// use the initial pose, which is used by default if rest pose is not found.
							real_t error = 0.0f;
							error += _relative_error(candidate_rest_xform.origin, xform.origin);
							for (int i = 0; i < 3; i++) {
								error += _relative_error(candidate_rest_xform.basis.rows[i], xform.basis.rows[i]);
							}
							const real_t max_error = 0.01f;
							if (error >= max_error) {
								WARN_PRINT(vformat("FBX: Node '%s' has multiple bind poses, using initial pose as rest pose.", node->get_name()));
								bad_rest_xform = true;
							}
						}
					}
				}
			}

			Transform3D godot_rest_xform = node->transform;
			if (found_rest_xform && !bad_rest_xform) {
				godot_rest_xform = candidate_rest_xform;
			}
			node->set_additional_data("GODOT_rest_transform", godot_rest_xform);
		}

		for (const ufbx_node *child : fbx_node->children) {
			node->children.push_back(child->typed_id);
		}

		p_state->nodes.push_back(node);
	}

	// build the hierarchy
	for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); node_i++) {
		for (int j = 0; j < p_state->nodes[node_i]->children.size(); j++) {
			GLTFNodeIndex child_i = p_state->nodes[node_i]->children[j];

			ERR_FAIL_INDEX_V(child_i, p_state->nodes.size(), ERR_FILE_CORRUPT);
			ERR_CONTINUE(p_state->nodes[child_i]->parent != -1); //node already has a parent, wtf.

			p_state->nodes.write[child_i]->parent = node_i;
		}
	}

	return OK;
}

Error FBXDocument::_parse_meshes(Ref<FBXState> p_state) {
	ufbx_scene *fbx_scene = p_state->scene.get();

	LocalVector<int> nodes_by_mesh_id;
	nodes_by_mesh_id.reserve(fbx_scene->meshes.count);
	for (size_t i = 0; i < fbx_scene->meshes.count; i++) {
		nodes_by_mesh_id.push_back(-1);
	}
	for (int i = 0; i < p_state->nodes.size(); i++) {
		const Ref<GLTFNode> &node = p_state->nodes[i];
		if (node->mesh >= 0 && (unsigned)node->mesh < nodes_by_mesh_id.size()) {
			nodes_by_mesh_id[node->mesh] = i;
		}
	}

	for (const ufbx_mesh *fbx_mesh : fbx_scene->meshes) {
		print_verbose("FBX: Parsing mesh: " + itos(int64_t(fbx_mesh->typed_id)));

		static const Mesh::PrimitiveType primitive_types[] = {
			Mesh::PRIMITIVE_TRIANGLES,
			Mesh::PRIMITIVE_POINTS,
			Mesh::PRIMITIVE_LINES,
		};

		Ref<ImporterMesh> import_mesh;
		import_mesh.instantiate();
		String mesh_name = "mesh";
		String original_name;
		if (fbx_mesh->name.length > 0) {
			mesh_name = _as_string(fbx_mesh->name);
			original_name = mesh_name;
		} else if (fbx_mesh->typed_id < (unsigned)p_state->nodes.size() && nodes_by_mesh_id[fbx_mesh->typed_id] != -1) {
			const Ref<GLTFNode> &node = p_state->nodes[nodes_by_mesh_id[fbx_mesh->typed_id]];
			original_name = node->get_original_name();
			mesh_name = node->get_name();
		}
		import_mesh->set_name(_gen_unique_name(p_state->unique_mesh_names, mesh_name));

		bool use_blend_shapes = false;
		if (fbx_mesh->blend_deformers.count > 0) {
			use_blend_shapes = true;
		}

		Vector<float> blend_weights;
		Vector<int> blend_channels;
		if (use_blend_shapes) {
			print_verbose("FBX: Mesh has targets");

			import_mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);

			for (const ufbx_blend_deformer *fbx_deformer : fbx_mesh->blend_deformers) {
				for (const ufbx_blend_channel *fbx_channel : fbx_deformer->channels) {
					if (fbx_channel->keyframes.count == 0) {
						continue;
					}
					String bs_name;
					if (fbx_channel->name.length > 0) {
						bs_name = _as_string(fbx_channel->name);
					} else {
						bs_name = String("morph_") + itos(blend_channels.size());
					}
					import_mesh->add_blend_shape(bs_name);
					blend_weights.push_back(float(fbx_channel->weight));
					blend_channels.push_back(float(fbx_channel->typed_id));
				}
			}
		}

		for (const ufbx_mesh_part &fbx_mesh_part : fbx_mesh->material_parts) {
			for (Mesh::PrimitiveType primitive : primitive_types) {
				uint32_t num_indices = 0;
				switch (primitive) {
					case Mesh::PRIMITIVE_POINTS:
						num_indices = fbx_mesh_part.num_point_faces * 1;
						break;
					case Mesh::PRIMITIVE_LINES:
						num_indices = fbx_mesh_part.num_line_faces * 2;
						break;
					case Mesh::PRIMITIVE_TRIANGLES:
						num_indices = fbx_mesh_part.num_triangles * 3;
						break;
					case Mesh::PRIMITIVE_TRIANGLE_STRIP:
						// FIXME 2021-09-15 fire
						break;
					case Mesh::PRIMITIVE_LINE_STRIP:
						// FIXME 2021-09-15 fire
						break;
					default:
						// FIXME 2021-09-15 fire
						break;
				}
				if (num_indices == 0) {
					continue;
				}

				Vector<uint32_t> indices;
				indices.resize(num_indices);

				uint32_t offset = 0;
				for (uint32_t face_index : fbx_mesh_part.face_indices) {
					ufbx_face face = fbx_mesh->faces[face_index];
					switch (primitive) {
						case Mesh::PRIMITIVE_POINTS: {
							if (face.num_indices == 1) {
								indices.write[offset] = face.index_begin;
								offset += 1;
							}
						} break;
						case Mesh::PRIMITIVE_LINES:
							if (face.num_indices == 2) {
								indices.write[offset] = face.index_begin;
								indices.write[offset + 1] = face.index_begin + 1;
								offset += 2;
							}
							break;
						case Mesh::PRIMITIVE_TRIANGLES:
							if (face.num_indices >= 3) {
								uint32_t *dst = indices.ptrw() + offset;
								size_t space = indices.size() - offset;
								uint32_t num_triangles = ufbx_triangulate_face(dst, space, fbx_mesh, face);
								offset += num_triangles * 3;

								// Godot uses clockwise winding order!
								for (uint32_t i = 0; i < num_triangles; i++) {
									SWAP(dst[i * 3 + 0], dst[i * 3 + 2]);
								}
							}
							break;
						case Mesh::PRIMITIVE_TRIANGLE_STRIP:
							// FIXME 2021-09-15 fire
							break;
						case Mesh::PRIMITIVE_LINE_STRIP:
							// FIXME 2021-09-15 fire
							break;
						default:
							// FIXME 2021-09-15 fire
							break;
					}
				}
				ERR_CONTINUE((uint64_t)offset != (uint64_t)indices.size());

				int32_t vertex_num = indices.size();
				bool has_vertex_color = false;

				uint32_t flags = 0;

				Array array;
				array.resize(Mesh::ARRAY_MAX);

				// HACK: If we have blend shapes we cannot merge vertices at identical positions
				// if they have different indices in the file. To avoid this encode the vertex index
				// into the vertex position for the time being.
				// Ideally this would be an extra channel in the vertex but as the vertex format is
				// fixed and we already use user data for extra UV channels this'll do.
				if (use_blend_shapes) {
					Vector<Vector3> vertex_indices;
					int num_blend_shape_indices = indices.size();
					vertex_indices.resize(num_blend_shape_indices);
					for (int i = 0; i < num_blend_shape_indices; i++) {
						vertex_indices.write[i] = _encode_vertex_index(fbx_mesh->vertex_indices[indices[i]]);
					}
					array[Mesh::ARRAY_VERTEX] = vertex_indices;
				} else {
					array[Mesh::ARRAY_VERTEX] = _decode_vertex_attrib_vec3(fbx_mesh->vertex_position, indices);
				}

				// Normals always exist as they're generated if missing,
				// see `ufbx_load_opts.generate_missing_normals`.
				Vector<Vector3> normals = _decode_vertex_attrib_vec3(fbx_mesh->vertex_normal, indices);
				array[Mesh::ARRAY_NORMAL] = normals;

				if (fbx_mesh->vertex_tangent.exists) {
					Vector<float> tangents = _decode_vertex_attrib_vec3_as_tangent(fbx_mesh->vertex_tangent, indices);

					// Patch bitangent sign if available
					if (fbx_mesh->vertex_bitangent.exists) {
						for (int i = 0; i < vertex_num; i++) {
							Vector3 tangent = Vector3(tangents[i * 4], tangents[i * 4 + 1], tangents[i * 4 + 2]);
							Vector3 bitangent = _as_vec3(fbx_mesh->vertex_bitangent[indices[i]]);
							Vector3 generated_bitangent = normals[i].cross(tangent);
							if (generated_bitangent.dot(bitangent) < 0.0f) {
								tangents.write[i * 4 + 3] = -1.0f;
							}
						}
					}

					array[Mesh::ARRAY_TANGENT] = tangents;
				}

				if (fbx_mesh->vertex_uv.exists) {
					PackedVector2Array uv_array = _decode_vertex_attrib_vec2(fbx_mesh->vertex_uv, indices);
					_process_uv_set(uv_array);
					array[Mesh::ARRAY_TEX_UV] = uv_array;
				}

				if (fbx_mesh->uv_sets.count >= 2 && fbx_mesh->uv_sets[1].vertex_uv.exists) {
					PackedVector2Array uv2_array = _decode_vertex_attrib_vec2(fbx_mesh->uv_sets[1].vertex_uv, indices);
					_process_uv_set(uv2_array);
					array[Mesh::ARRAY_TEX_UV2] = uv2_array;
				}

				for (int uv_i = 2; uv_i < 8; uv_i += 2) {
					Vector<float> cur_custom;
					Vector<Vector2> texcoord_first;
					Vector<Vector2> texcoord_second;

					int texcoord_i = uv_i;
					int texcoord_next = texcoord_i + 1;
					int num_channels = 0;
					if (texcoord_i < static_cast<int>(fbx_mesh->uv_sets.count) && fbx_mesh->uv_sets[texcoord_i].vertex_uv.exists) {
						texcoord_first = _decode_vertex_attrib_vec2(fbx_mesh->uv_sets[texcoord_i].vertex_uv, indices);
						_process_uv_set(texcoord_first);
						num_channels = 2;
					}
					if (texcoord_next < static_cast<int>(fbx_mesh->uv_sets.count) && fbx_mesh->uv_sets[texcoord_next].vertex_uv.exists) {
						texcoord_second = _decode_vertex_attrib_vec2(fbx_mesh->uv_sets[texcoord_next].vertex_uv, indices);
						_process_uv_set(texcoord_second);
						num_channels = 4;
					}
					if (!num_channels) {
						break;
					}
					cur_custom.resize(vertex_num * num_channels);
					for (int32_t uv_first_i = 0; uv_first_i < texcoord_first.size() && uv_first_i < vertex_num; uv_first_i++) {
						int index = uv_first_i * num_channels;
						cur_custom.write[index] = texcoord_first[uv_first_i].x;
						cur_custom.write[index + 1] = texcoord_first[uv_first_i].y;
					}
					if (num_channels == 4) {
						for (int32_t uv_second_i = 0; uv_second_i < texcoord_second.size() && uv_second_i < vertex_num; uv_second_i++) {
							int index = uv_second_i * num_channels;
							cur_custom.write[index + 2] = texcoord_second[uv_second_i].x;
							cur_custom.write[index + 3] = texcoord_second[uv_second_i].y;
						}
						_zero_unused_elements(cur_custom, texcoord_second.size(), vertex_num, num_channels);
					} else if (num_channels == 2) {
						_zero_unused_elements(cur_custom, texcoord_first.size(), vertex_num, num_channels);
					}
					if (!cur_custom.is_empty()) {
						array[Mesh::ARRAY_CUSTOM0 + ((uv_i - 2) / 2)] = cur_custom; // Map uv2-uv7 to custom0-custom2
						int custom_shift = Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT + ((uv_i - 2) / 2) * Mesh::ARRAY_FORMAT_CUSTOM_BITS;
						flags |= (num_channels == 2 ? Mesh::ARRAY_CUSTOM_RG_FLOAT : Mesh::ARRAY_CUSTOM_RGBA_FLOAT) << custom_shift;
					}
				}

				if (fbx_mesh->vertex_color.exists) {
					array[Mesh::ARRAY_COLOR] = _decode_vertex_attrib_color(fbx_mesh->vertex_color, indices);
					has_vertex_color = true;
				}

				int32_t num_skin_weights = 0;

				// Find the first imported skin deformer
				for (ufbx_skin_deformer *fbx_skin : fbx_mesh->skin_deformers) {
					GLTFSkinIndex skin_i = p_state->original_skin_indices[fbx_skin->typed_id];
					if (skin_i < 0) {
						continue;
					}

					// Tag all nodes to use the skin
					for (const ufbx_node *node : fbx_mesh->instances) {
						p_state->nodes[node->typed_id]->skin = skin_i;
					}

					num_skin_weights = fbx_skin->max_weights_per_vertex > 4 ? 8 : 4;

					Vector<int32_t> bones;
					Vector<float> weights;

					bones.resize(vertex_num * num_skin_weights);
					weights.resize(vertex_num * num_skin_weights);
					for (int32_t vertex_i = 0; vertex_i < vertex_num; vertex_i++) {
						uint32_t fbx_vertex_index = fbx_mesh->vertex_indices[indices[vertex_i]];
						ufbx_skin_vertex skin_vertex = fbx_skin->vertices[fbx_vertex_index];
						float total_weight = 0.0f;
						int32_t num_weights = MIN(int32_t(skin_vertex.num_weights), num_skin_weights);
						for (int32_t i = 0; i < num_weights; i++) {
							ufbx_skin_weight skin_weight = fbx_skin->weights[skin_vertex.weight_begin + i];
							int index = vertex_i * num_skin_weights + i;
							float weight = float(skin_weight.weight);
							bones.write[index] = int(skin_weight.cluster_index);
							weights.write[index] = weight;
							total_weight += weight;
						}
						if (total_weight > 0.0f) {
							for (int32_t i = 0; i < num_weights; i++) {
								int index = vertex_i * num_skin_weights + i;
								weights.write[index] /= total_weight;
							}
						}
						// Pad the rest with empty weights
						for (int32_t i = num_weights; i < num_skin_weights; i++) {
							int index = vertex_i * num_skin_weights + i;
							bones.write[index] = 0; // TODO: What should this be padded with?
							weights.write[index] = 0.0f;
						}
					}
					array[Mesh::ARRAY_BONES] = bones;
					array[Mesh::ARRAY_WEIGHTS] = weights;

					if (num_skin_weights == 8) {
						flags |= Mesh::ARRAY_FLAG_USE_8_BONE_WEIGHTS;
					}

					// Only use the first found skin
					break;
				}

				bool generate_tangents = (primitive == Mesh::PRIMITIVE_TRIANGLES && !array[Mesh::ARRAY_TANGENT] && array[Mesh::ARRAY_TEX_UV] && array[Mesh::ARRAY_NORMAL]);

				Ref<SurfaceTool> mesh_surface_tool;
				mesh_surface_tool.instantiate();
				mesh_surface_tool->create_from_triangle_arrays(array);
				mesh_surface_tool->set_skin_weight_count(num_skin_weights == 8 ? SurfaceTool::SKIN_8_WEIGHTS : SurfaceTool::SKIN_4_WEIGHTS);
				mesh_surface_tool->index();
				if (generate_tangents) {
					//must generate mikktspace tangents.. ergh..
					mesh_surface_tool->generate_tangents();
				}
				array = mesh_surface_tool->commit_to_arrays();

				Array morphs;
				//blend shapes
				if (use_blend_shapes) {
					print_verbose("FBX: Mesh has targets");

					import_mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);

					for (const ufbx_blend_deformer *fbx_deformer : fbx_mesh->blend_deformers) {
						for (const ufbx_blend_channel *fbx_channel : fbx_deformer->channels) {
							if (fbx_channel->keyframes.count == 0) {
								continue;
							}

							// Use the last shape keyframe by default
							ufbx_blend_shape *fbx_shape = fbx_channel->keyframes[fbx_channel->keyframes.count - 1].shape;

							Array array_copy;
							array_copy.resize(Mesh::ARRAY_MAX);

							for (int l = 0; l < Mesh::ARRAY_MAX; l++) {
								array_copy[l] = array[l];
							}

							Vector<Vector3> varr;
							Vector<Vector3> narr;
							const Vector<Vector3> src_varr = array[Mesh::ARRAY_VERTEX];
							const Vector<Vector3> src_narr = array[Mesh::ARRAY_NORMAL];
							const int size = src_varr.size();
							ERR_FAIL_COND_V(size == 0, ERR_PARSE_ERROR);
							{
								varr.resize(size);
								narr.resize(size);

								Vector3 *w_varr = varr.ptrw();
								Vector3 *w_narr = narr.ptrw();
								const Vector3 *r_varr = src_varr.ptr();
								const Vector3 *r_narr = src_narr.ptr();
								for (int l = 0; l < size; l++) {
									uint32_t vertex_index = _decode_vertex_index(r_varr[l]);
									uint32_t offset_index = ufbx_get_blend_shape_offset_index(fbx_shape, vertex_index);
									Vector3 position = _as_vec3(fbx_mesh->vertices[vertex_index]);
									Vector3 normal = r_narr[l];

									if (offset_index != UFBX_NO_INDEX && offset_index < fbx_shape->position_offsets.count) {
										Vector3 blend_shape_position_offset = _as_vec3(fbx_shape->position_offsets[offset_index]);
										w_varr[l] = position + blend_shape_position_offset;
									} else {
										w_varr[l] = position;
									}

									if (offset_index != UFBX_NO_INDEX && offset_index < fbx_shape->normal_offsets.count) {
										w_narr[l] = (normal.normalized() + _as_vec3(fbx_shape->normal_offsets[offset_index])).normalized();
									} else {
										w_narr[l] = normal;
									}
								}
							}
							array_copy[Mesh::ARRAY_VERTEX] = varr;
							array_copy[Mesh::ARRAY_NORMAL] = narr;

							Ref<SurfaceTool> blend_surface_tool;
							blend_surface_tool.instantiate();
							blend_surface_tool->create_from_triangle_arrays(array_copy);
							blend_surface_tool->set_skin_weight_count(num_skin_weights == 8 ? SurfaceTool::SKIN_8_WEIGHTS : SurfaceTool::SKIN_4_WEIGHTS);
							if (generate_tangents) {
								//must generate mikktspace tangents.. ergh..
								blend_surface_tool->generate_tangents();
							}
							array_copy = blend_surface_tool->commit_to_arrays();

							// Enforce blend shape mask array format
							for (int l = 0; l < Mesh::ARRAY_MAX; l++) {
								if (!(Mesh::ARRAY_FORMAT_BLEND_SHAPE_MASK & (static_cast<int64_t>(1) << l))) {
									array_copy[l] = Variant();
								}
							}

							morphs.push_back(array_copy);
						}
					}
				}

				// Decode the original vertex positions now that we're done processing blend shapes.
				if (use_blend_shapes) {
					Vector<Vector3> varr = array[Mesh::ARRAY_VERTEX];
					Vector3 *w_varr = varr.ptrw();
					const int size = varr.size();
					for (int i = 0; i < size; i++) {
						uint32_t vertex_index = _decode_vertex_index(w_varr[i]);
						w_varr[i] = _as_vec3(fbx_mesh->vertices[vertex_index]);
					}
					array[Mesh::ARRAY_VERTEX] = varr;
				}

				Ref<Material> mat;
				String mat_name;
				if (!p_state->discard_meshes_and_materials) {
					ufbx_material *fbx_material = nullptr;
					if (fbx_mesh_part.index < fbx_mesh->materials.count) {
						fbx_material = fbx_mesh->materials[fbx_mesh_part.index];
					}
					if (fbx_material) {
						const int material = int(fbx_material->typed_id);
						ERR_FAIL_INDEX_V(material, p_state->materials.size(), ERR_FILE_CORRUPT);
						Ref<Material> mat3d = p_state->materials[material];
						ERR_FAIL_COND_V(mat3d.is_null(), ERR_FILE_CORRUPT);

						Ref<BaseMaterial3D> base_material = mat3d;
						if (has_vertex_color && base_material.is_valid()) {
							base_material->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
						}
						mat = mat3d;

					} else {
						Ref<StandardMaterial3D> mat3d;
						mat3d.instantiate();
						if (has_vertex_color) {
							mat3d->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
						}
						mat = mat3d;
					}
					ERR_FAIL_COND_V(mat.is_null(), ERR_FILE_CORRUPT);
					mat_name = mat->get_name();
				}
				import_mesh->add_surface(primitive, array, morphs,
						Dictionary(), mat, mat_name, flags);
			}
		}

		Ref<GLTFMesh> mesh;
		mesh.instantiate();
		Dictionary additional_data;
		additional_data["blend_channels"] = blend_channels;
		mesh->set_additional_data("GODOT_mesh_blend_channels", additional_data);
		mesh->set_blend_weights(blend_weights);
		mesh->set_mesh(import_mesh);
		mesh->set_name(import_mesh->get_name());
		mesh->set_original_name(original_name);

		p_state->meshes.push_back(mesh);
	}

	print_verbose("FBX: Total meshes: " + itos(p_state->meshes.size()));

	return OK;
}

Ref<Image> FBXDocument::_parse_image_bytes_into_image(Ref<FBXState> p_state, const Vector<uint8_t> &p_bytes, const String &p_filename, int p_index) {
	Ref<Image> r_image;
	r_image.instantiate();
	// Try to import first based on filename.
	String filename_lower = p_filename.to_lower();
	if (filename_lower.ends_with(".png")) {
		r_image->load_png_from_buffer(p_bytes);
	} else if (filename_lower.ends_with(".jpg")) {
		r_image->load_jpg_from_buffer(p_bytes);
	} else if (filename_lower.ends_with(".tga")) {
		r_image->load_tga_from_buffer(p_bytes);
	}
	// If we didn't pass the above tests, try loading as each option.
	if (r_image->is_empty()) { // Try PNG first.
		r_image->load_png_from_buffer(p_bytes);
	}
	if (r_image->is_empty()) { // And then JPEG.
		r_image->load_jpg_from_buffer(p_bytes);
	}
	if (r_image->is_empty()) { // And then TGA.
		r_image->load_jpg_from_buffer(p_bytes);
	}
	// If it still can't be loaded, give up and insert an empty image as placeholder.
	if (r_image->is_empty()) {
		ERR_PRINT(vformat("FBX: Couldn't load image index '%d'", p_index));
	}
	return r_image;
}

GLTFImageIndex FBXDocument::_parse_image_save_image(Ref<FBXState> p_state, const Vector<uint8_t> &p_bytes, const String &p_file_extension, int p_index, Ref<Image> p_image) {
	FBXState::HandleBinaryImageMode handling = FBXState::HandleBinaryImageMode(p_state->handle_binary_image_mode);
	if (p_image->is_empty() || handling == FBXState::HandleBinaryImageMode::HANDLE_BINARY_IMAGE_MODE_DISCARD_TEXTURES) {
		if (p_index < 0) {
			return -1;
		}
		p_state->images.push_back(Ref<Texture2D>());
		p_state->source_images.push_back(Ref<Image>());
		return p_state->images.size() - 1;
	}
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && handling == FBXState::HandleBinaryImageMode::HANDLE_BINARY_IMAGE_MODE_EXTRACT_TEXTURES) {
		if (p_state->base_path.is_empty()) {
			if (p_index < 0) {
				return -1;
			}
			p_state->images.push_back(Ref<Texture2D>());
			p_state->source_images.push_back(Ref<Image>());
		} else if (p_image->get_name().is_empty()) {
			if (p_index < 0) {
				return -1;
			}
			WARN_PRINT(vformat("FBX: Image index '%d' couldn't be named. Skipping it.", p_index));
			p_state->images.push_back(Ref<Texture2D>());
			p_state->source_images.push_back(Ref<Image>());
		} else {
			bool must_import = true;
			Vector<uint8_t> img_data = p_image->get_data();
			Dictionary generator_parameters;
			String file_path = p_state->get_base_path().path_join(p_state->filename.get_basename() + "_" + p_image->get_name());
			file_path += p_file_extension.is_empty() ? ".png" : p_file_extension;
			if (FileAccess::exists(file_path + ".import")) {
				Ref<ConfigFile> config;
				config.instantiate();
				config->load(file_path + ".import");
				if (config->has_section_key("remap", "generator_parameters")) {
					generator_parameters = (Dictionary)config->get_value("remap", "generator_parameters");
				}
				if (!generator_parameters.has("md5")) {
					must_import = false; // Didn't come from a gltf document; don't overwrite.
				}
			}
			if (must_import) {
				String existing_md5 = generator_parameters["md5"];
				unsigned char md5_hash[16];
				CryptoCore::md5(img_data.ptr(), img_data.size(), md5_hash);
				String new_md5 = String::hex_encode_buffer(md5_hash, 16);
				generator_parameters["md5"] = new_md5;
				if (new_md5 == existing_md5) {
					must_import = false;
				}
			}
			if (must_import) {
				Error err = OK;
				if (p_file_extension.is_empty()) {
					// If a file extension was not specified, save the image data to a PNG file.
					err = p_image->save_png(file_path);
					ERR_FAIL_COND_V(err != OK, -1);
				} else {
					// If a file extension was specified, save the original bytes to a file with that extension.
					Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::WRITE, &err);
					ERR_FAIL_COND_V(err != OK, -1);
					file->store_buffer(p_bytes);
					file->close();
				}
				// ResourceLoader::import will crash if not is_editor_hint(), so this case is protected above and will fall through to uncompressed.
				HashMap<StringName, Variant> custom_options;
				custom_options[SNAME("mipmaps/generate")] = true;
				// Will only use project settings defaults if custom_importer is empty.
				EditorFileSystem::get_singleton()->update_file(file_path);
				EditorFileSystem::get_singleton()->reimport_append(file_path, custom_options, String(), generator_parameters);
			}
			Ref<Texture2D> saved_image = ResourceLoader::load(_get_texture_path(p_state->get_base_path(), file_path), "Texture2D");
			if (saved_image.is_valid()) {
				p_state->images.push_back(saved_image);
				p_state->source_images.push_back(saved_image->get_image());
			} else if (p_index < 0) {
				return -1;
			} else {
				WARN_PRINT(vformat("FBX: Image index '%d' couldn't be loaded with the name: %s. Skipping it.", p_index, p_image->get_name()));
				// Placeholder to keep count.
				p_state->images.push_back(Ref<Texture2D>());
				p_state->source_images.push_back(Ref<Image>());
			}
		}
		return p_state->images.size() - 1;
	}
#endif // TOOLS_ENABLED
	if (handling == FBXState::HANDLE_BINARY_IMAGE_MODE_EMBED_AS_BASISU) {
		Ref<PortableCompressedTexture2D> tex;
		tex.instantiate();
		tex->set_name(p_image->get_name());
		tex->set_keep_compressed_buffer(true);
		tex->create_from_image(p_image, PortableCompressedTexture2D::COMPRESSION_MODE_BASIS_UNIVERSAL);
		p_state->images.push_back(tex);
		p_state->source_images.push_back(p_image);
		return p_state->images.size() - 1;
	}
	// This handles the case of HANDLE_BINARY_IMAGE_MODE_EMBED_AS_UNCOMPRESSED, and it also serves
	// as a fallback for HANDLE_BINARY_IMAGE_MODE_EXTRACT_TEXTURES when this is not the editor.
	Ref<ImageTexture> tex;
	tex.instantiate();
	tex->set_name(p_image->get_name());
	tex->set_image(p_image);
	p_state->images.push_back(tex);
	p_state->source_images.push_back(p_image);
	return p_state->images.size() - 1;
}

Error FBXDocument::_parse_images(Ref<FBXState> p_state, const String &p_base_path) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);

	const ufbx_scene *fbx_scene = p_state->scene.get();
	for (int texture_i = 0; texture_i < static_cast<int>(fbx_scene->texture_files.count); texture_i++) {
		const ufbx_texture_file &fbx_texture_file = fbx_scene->texture_files[texture_i];
		String path = _as_string(fbx_texture_file.filename);
		// Use only filename for absolute paths to avoid portability issues.
		if (path.is_absolute_path()) {
			path = path.get_file();
		}
		if (!p_base_path.is_empty()) {
			path = p_base_path.path_join(path);
		}
		path = path.simplify_path();
		Vector<uint8_t> data;
		if (fbx_texture_file.content.size > 0 && fbx_texture_file.content.size <= INT_MAX) {
			data.resize(int(fbx_texture_file.content.size));
			memcpy(data.ptrw(), fbx_texture_file.content.data, fbx_texture_file.content.size);
		} else {
			String base_dir = p_state->get_base_path();
			Ref<Texture2D> texture = ResourceLoader::load(_get_texture_path(base_dir, path), "Texture2D");
			if (texture.is_valid()) {
				p_state->images.push_back(texture);
				p_state->source_images.push_back(texture->get_image());
				continue;
			}
			// Fallback to loading as byte array.
			data = FileAccess::get_file_as_bytes(path);
			if (data.is_empty()) {
				WARN_PRINT(vformat("FBX: Image index '%d' couldn't be loaded from path: %s because there was no data to load. Skipping it.", texture_i, path));
				p_state->images.push_back(Ref<Texture2D>()); // Placeholder to keep count.
				p_state->source_images.push_back(Ref<Image>());
				continue;
			}
		}

		// Parse the image data from bytes into an Image resource and save if needed.
		String file_extension;
		Ref<Image> img = _parse_image_bytes_into_image(p_state, data, path, texture_i);
		img->set_name(itos(texture_i));
		_parse_image_save_image(p_state, data, file_extension, texture_i, img);
	}

	// Create a texture for each file texture.
	for (int texture_file_i = 0; texture_file_i < static_cast<int>(fbx_scene->texture_files.count); texture_file_i++) {
		Ref<GLTFTexture> texture;
		texture.instantiate();
		texture->set_src_image(GLTFImageIndex(texture_file_i));
		p_state->textures.push_back(texture);
	}

	print_verbose("FBX: Total images: " + itos(p_state->images.size()));

	return OK;
}

Ref<Texture2D> FBXDocument::_get_texture(Ref<FBXState> p_state, const GLTFTextureIndex p_texture, int p_texture_types) {
	ERR_FAIL_INDEX_V(p_texture, p_state->textures.size(), Ref<Texture2D>());
	const GLTFImageIndex image = p_state->textures[p_texture]->get_src_image();
	ERR_FAIL_INDEX_V(image, p_state->images.size(), Ref<Texture2D>());
	if (FBXState::HandleBinaryImageMode(p_state->handle_binary_image_mode) == FBXState::HANDLE_BINARY_IMAGE_MODE_EMBED_AS_BASISU) {
		ERR_FAIL_INDEX_V(image, p_state->source_images.size(), Ref<Texture2D>());
		Ref<PortableCompressedTexture2D> portable_texture;
		portable_texture.instantiate();
		portable_texture->set_keep_compressed_buffer(true);
		Ref<Image> new_img = p_state->source_images[image]->duplicate();
		ERR_FAIL_COND_V(new_img.is_null(), Ref<Texture2D>());
		new_img->generate_mipmaps();
		if (p_texture_types) {
			portable_texture->create_from_image(new_img, PortableCompressedTexture2D::COMPRESSION_MODE_BASIS_UNIVERSAL, true);
		} else {
			portable_texture->create_from_image(new_img, PortableCompressedTexture2D::COMPRESSION_MODE_BASIS_UNIVERSAL, false);
		}
		p_state->images.write[image] = portable_texture;
		p_state->source_images.write[image] = new_img;
	}
	return p_state->images[image];
}

Error FBXDocument::_parse_materials(Ref<FBXState> p_state) {
	const ufbx_scene *fbx_scene = p_state->scene.get();
	for (GLTFMaterialIndex material_i = 0; material_i < static_cast<GLTFMaterialIndex>(fbx_scene->materials.count); material_i++) {
		const ufbx_material *fbx_material = fbx_scene->materials[material_i];

		Ref<StandardMaterial3D> material;
		material.instantiate();
		if (fbx_material->name.length > 0) {
			material->set_name(_as_string(fbx_material->name));
		} else {
			material->set_name(vformat("material_%s", itos(material_i)));
		}
		material->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		Dictionary material_extensions;

		if (fbx_material->pbr.base_color.has_value) {
			Color albedo = _material_color(fbx_material->pbr.base_color, fbx_material->pbr.base_factor);
			material->set_albedo(albedo.linear_to_srgb());
		}

		if (fbx_material->features.double_sided.enabled) {
			material->set_cull_mode(BaseMaterial3D::CULL_DISABLED);
		}

		const ufbx_texture *base_texture = _get_file_texture(fbx_material->pbr.base_color.texture);
		if (base_texture) {
			bool wrap = base_texture->wrap_u == UFBX_WRAP_REPEAT && base_texture->wrap_v == UFBX_WRAP_REPEAT;
			material->set_flag(BaseMaterial3D::FLAG_USE_TEXTURE_REPEAT, wrap);

			Ref<Texture2D> albedo_texture = _get_texture(p_state, GLTFTextureIndex(base_texture->file_index), TEXTURE_TYPE_GENERIC);

			// Search for transparency map.
			Ref<Texture2D> transparency_texture;
			const ufbx_texture *transparency_sources[] = {
				fbx_material->pbr.opacity.texture,
				fbx_material->fbx.transparency_color.texture,
			};
			for (const ufbx_texture *transparency_source : transparency_sources) {
				const ufbx_texture *fbx_transparency_texture = _get_file_texture(transparency_source);
				if (fbx_transparency_texture) {
					transparency_texture = _get_texture(p_state, GLTFTextureIndex(fbx_transparency_texture->file_index), TEXTURE_TYPE_GENERIC);
					if (transparency_texture.is_valid()) {
						break;
					}
				}
			}

			// Multiply the albedo alpha with the transparency texture if necessary.
			if (albedo_texture.is_valid() && transparency_texture.is_valid() && albedo_texture != transparency_texture) {
				Pair<uint64_t, uint64_t> key = { albedo_texture->get_rid().get_id(), transparency_texture->get_rid().get_id() };
				GLTFTextureIndex *texture_index_ptr = p_state->albedo_transparency_textures.getptr(key);
				if (texture_index_ptr != nullptr) {
					if (*texture_index_ptr >= 0) {
						albedo_texture = _get_texture(p_state, *texture_index_ptr, TEXTURE_TYPE_GENERIC);
					}
				} else {
					Ref<Image> albedo_image = _get_decompressed_image(albedo_texture);
					Ref<Image> transparency_image = _get_decompressed_image(transparency_texture);

					if (albedo_image.is_valid() && transparency_image.is_valid()) {
						albedo_image->convert(Image::Format::FORMAT_RGBA8);
						transparency_image->resize(albedo_texture->get_width(), albedo_texture->get_height(), Image::INTERPOLATE_LANCZOS);
						for (int y = 0; y < albedo_image->get_height(); y++) {
							for (int x = 0; x < albedo_image->get_width(); x++) {
								Color albedo_pixel = albedo_image->get_pixel(x, y);
								Color transparency_pixel = transparency_image->get_pixel(x, y);
								albedo_pixel.a *= transparency_pixel.r;
								albedo_image->set_pixel(x, y, albedo_pixel);
							}
						}

						albedo_image->clear_mipmaps();
						albedo_image->generate_mipmaps();

						albedo_image->set_name(vformat("alpha_%d", p_state->albedo_transparency_textures.size()));

						GLTFImageIndex new_image = _parse_image_save_image(p_state, PackedByteArray(), "", -1, albedo_image);
						if (new_image >= 0) {
							Ref<GLTFTexture> new_texture;
							new_texture.instantiate();
							new_texture->set_src_image(GLTFImageIndex(new_image));
							p_state->textures.push_back(new_texture);

							GLTFTextureIndex texture_index = p_state->textures.size() - 1;
							p_state->albedo_transparency_textures[key] = texture_index;

							albedo_texture = _get_texture(p_state, texture_index, TEXTURE_TYPE_GENERIC);
						} else {
							WARN_PRINT(vformat("FBX: Could not save modified albedo texture from RID (%d, %d).", key.first, key.second));
							p_state->albedo_transparency_textures[key] = -1;
						}
					}
				}
			}

			Image::AlphaMode alpha_mode;
			if (albedo_texture.is_valid()) {
				Image::AlphaMode *alpha_mode_ptr = p_state->alpha_mode_cache.getptr(albedo_texture->get_rid().get_id());
				if (alpha_mode_ptr != nullptr) {
					alpha_mode = *alpha_mode_ptr;
				} else {
					Ref<Image> albedo_image = _get_decompressed_image(albedo_texture);
					alpha_mode = albedo_image->detect_alpha();
					p_state->alpha_mode_cache[albedo_texture->get_rid().get_id()] = alpha_mode;
				}

				if (alpha_mode == Image::ALPHA_BLEND) {
					material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA_DEPTH_PRE_PASS);
				} else if (alpha_mode == Image::ALPHA_BIT) {
					material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA_SCISSOR);
				}
				material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, albedo_texture);
			}

			// Combined textures and factors are very unreliable in FBX
			Color albedo_factor = Color(1, 1, 1);
			if (fbx_material->pbr.base_factor.has_value) {
				albedo_factor *= (float)fbx_material->pbr.base_factor.value_real;
			}
			material->set_albedo(albedo_factor.linear_to_srgb());

			// TODO: Does not support rotation, could be inverted?
			material->set_uv1_offset(_as_vec3(base_texture->uv_transform.translation));
			Vector3 scale = _as_vec3(base_texture->uv_transform.scale);
			material->set_uv1_scale(scale);
		}

		if (fbx_material->features.pbr.enabled) {
			if (fbx_material->pbr.metalness.has_value) {
				material->set_metallic(float(fbx_material->pbr.metalness.value_real));
			} else {
				material->set_metallic(1.0);
			}

			if (fbx_material->pbr.roughness.has_value) {
				material->set_roughness(float(fbx_material->pbr.roughness.value_real));
			} else {
				material->set_roughness(1.0);
			}

			const ufbx_texture *metalness_texture = _get_file_texture(fbx_material->pbr.metalness.texture);
			if (metalness_texture) {
				material->set_texture(BaseMaterial3D::TEXTURE_METALLIC, _get_texture(p_state, GLTFTextureIndex(metalness_texture->file_index), TEXTURE_TYPE_GENERIC));
				material->set_metallic_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_RED);
				material->set_metallic(1.0);
			}

			const ufbx_texture *roughness_texture = _get_file_texture(fbx_material->pbr.roughness.texture);
			if (roughness_texture) {
				material->set_texture(BaseMaterial3D::TEXTURE_ROUGHNESS, _get_texture(p_state, GLTFTextureIndex(roughness_texture->file_index), TEXTURE_TYPE_GENERIC));
				material->set_roughness_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_RED);
				material->set_roughness(1.0);
			}
		}

		const ufbx_texture *normal_texture = _get_file_texture(fbx_material->pbr.normal_map.texture);
		if (normal_texture) {
			material->set_texture(BaseMaterial3D::TEXTURE_NORMAL, _get_texture(p_state, GLTFTextureIndex(normal_texture->file_index), TEXTURE_TYPE_NORMAL));
			material->set_feature(BaseMaterial3D::FEATURE_NORMAL_MAPPING, true);
			if (fbx_material->pbr.normal_map.has_value) {
				material->set_normal_scale(fbx_material->pbr.normal_map.value_real);
			}
		}

		const ufbx_texture *occlusion_texture = _get_file_texture(fbx_material->pbr.ambient_occlusion.texture);
		if (occlusion_texture) {
			material->set_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION, _get_texture(p_state, GLTFTextureIndex(occlusion_texture->file_index), TEXTURE_TYPE_GENERIC));
			material->set_ao_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_RED);
			material->set_feature(BaseMaterial3D::FEATURE_AMBIENT_OCCLUSION, true);
		}

		if (fbx_material->pbr.emission_color.has_value) {
			material->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);
			material->set_emission(_material_color(fbx_material->pbr.emission_color).linear_to_srgb());
			material->set_emission_energy_multiplier(float(fbx_material->pbr.emission_factor.value_real));
		}

		const ufbx_texture *emission_texture = _get_file_texture(fbx_material->pbr.emission_color.texture);
		if (emission_texture) {
			material->set_texture(BaseMaterial3D::TEXTURE_EMISSION, _get_texture(p_state, GLTFTextureIndex(emission_texture->file_index), TEXTURE_TYPE_GENERIC));
			material->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);
			material->set_emission(Color(0, 0, 0));
		}

		if (fbx_material->features.double_sided.enabled && fbx_material->features.double_sided.is_explicit) {
			material->set_cull_mode(BaseMaterial3D::CULL_DISABLED);
		}
		p_state->materials.push_back(material);
	}

	print_verbose("Total materials: " + itos(p_state->materials.size()));

	return OK;
}
Error FBXDocument::_parse_cameras(Ref<FBXState> p_state) {
	const ufbx_scene *fbx_scene = p_state->scene.get();
	for (GLTFCameraIndex i = 0; i < static_cast<GLTFCameraIndex>(fbx_scene->cameras.count); i++) {
		const ufbx_camera *fbx_camera = fbx_scene->cameras[i];

		Ref<GLTFCamera> camera;
		camera.instantiate();
		camera->set_name(_as_string(fbx_camera->name));
		if (fbx_camera->projection_mode == UFBX_PROJECTION_MODE_PERSPECTIVE) {
			camera->set_perspective(true);
			camera->set_fov(Math::deg_to_rad(real_t(fbx_camera->field_of_view_deg.y)));
		} else {
			camera->set_perspective(false);
			camera->set_size_mag(real_t(fbx_camera->orthographic_size.y * 0.5f));
		}
		if (fbx_camera->near_plane != 0.0f) {
			camera->set_depth_near(fbx_camera->near_plane);
		}
		if (fbx_camera->far_plane != 0.0f) {
			camera->set_depth_far(fbx_camera->far_plane);
		}
		p_state->cameras.push_back(camera);
	}

	print_verbose("FBX: Total cameras: " + itos(p_state->cameras.size()));

	return OK;
}

Error FBXDocument::_parse_animations(Ref<FBXState> p_state) {
	const ufbx_scene *fbx_scene = p_state->scene.get();
	for (GLTFAnimationIndex animation_i = 0; animation_i < static_cast<GLTFAnimationIndex>(fbx_scene->anim_stacks.count); animation_i++) {
		const ufbx_anim_stack *fbx_anim_stack = fbx_scene->anim_stacks[animation_i];

		Ref<GLTFAnimation> animation;
		animation.instantiate();

		if (fbx_anim_stack->name.length > 0) {
			const String anim_name = _as_string(fbx_anim_stack->name);
			const String anim_name_lower = anim_name.to_lower();
			if (anim_name_lower.begins_with("loop") || anim_name_lower.ends_with("loop") || anim_name_lower.begins_with("cycle") || anim_name_lower.ends_with("cycle")) {
				animation->set_loop(true);
			}
			animation->set_original_name(anim_name);
			animation->set_name(_gen_unique_animation_name(p_state, anim_name));
		}

		Dictionary additional_data;
		additional_data["time_begin"] = fbx_anim_stack->time_begin;
		additional_data["time_end"] = fbx_anim_stack->time_end;
		animation->set_additional_data("GODOT_animation_time_begin_time_end", additional_data);
		ufbx_bake_opts opts = {};
		opts.resample_rate = p_state->get_bake_fps();
		opts.minimum_sample_rate = p_state->get_bake_fps();
		opts.max_keyframe_segments = 1024;

		ufbx_error error;
		ufbx_unique_ptr<ufbx_baked_anim> fbx_baked_anim{ ufbx_bake_anim(fbx_scene, fbx_anim_stack->anim, &opts, &error) };
		if (!fbx_baked_anim) {
			char err_buf[512];
			ufbx_format_error(err_buf, sizeof(err_buf), &error);
			ERR_FAIL_V_MSG(FAILED, err_buf);
		}

		for (const ufbx_baked_node &fbx_baked_node : fbx_baked_anim->nodes) {
			const GLTFNodeIndex node = fbx_baked_node.typed_id;
			GLTFAnimation::NodeTrack &track = animation->get_node_tracks()[node];

			for (const ufbx_baked_vec3 &key : fbx_baked_node.translation_keys) {
				track.position_track.times.push_back(float(key.time));
				track.position_track.values.push_back(_as_vec3(key.value));
			}

			for (const ufbx_baked_quat &key : fbx_baked_node.rotation_keys) {
				track.rotation_track.times.push_back(float(key.time));
				track.rotation_track.values.push_back(_as_quaternion(key.value));
			}

			for (const ufbx_baked_vec3 &key : fbx_baked_node.scale_keys) {
				track.scale_track.times.push_back(float(key.time));
				track.scale_track.values.push_back(_as_vec3(key.value));
			}
		}

		Dictionary blend_shape_animations;

		for (const ufbx_baked_element &fbx_baked_element : fbx_baked_anim->elements) {
			const ufbx_element *fbx_element = fbx_scene->elements[fbx_baked_element.element_id];

			for (const ufbx_baked_prop &fbx_baked_prop : fbx_baked_element.props) {
				String prop_name = _as_string(fbx_baked_prop.name);

				if (fbx_element->type == UFBX_ELEMENT_BLEND_CHANNEL && prop_name == UFBX_DeformPercent) {
					const ufbx_blend_channel *fbx_blend_channel = ufbx_as_blend_channel(fbx_element);

					int blend_i = fbx_blend_channel->typed_id;
					Vector<real_t> track_times;
					Vector<real_t> track_values;

					for (const ufbx_baked_vec3 &key : fbx_baked_prop.keys) {
						track_times.push_back(float(key.time));
						track_values.push_back(real_t(key.value.x / 100.0));
					}

					Dictionary track;
					track["times"] = track_times;
					track["values"] = track_values;
					blend_shape_animations[blend_i] = track;
				}
			}
		}

		animation->set_additional_data("GODOT_blend_shape_animations", blend_shape_animations);

		p_state->animations.push_back(animation);
	}

	print_verbose("FBX: Total animations '" + itos(p_state->animations.size()) + "'.");

	return OK;
}

void FBXDocument::_assign_node_names(Ref<FBXState> p_state) {
	for (int i = 0; i < p_state->nodes.size(); i++) {
		Ref<GLTFNode> fbx_node = p_state->nodes[i];

		// Any joints get unique names generated when the skeleton is made, unique to the skeleton
		if (fbx_node->skeleton >= 0) {
			continue;
		}

		if (fbx_node->get_name().is_empty()) {
			if (fbx_node->mesh >= 0) {
				fbx_node->set_name(_gen_unique_name(p_state->unique_names, "Mesh"));
			} else if (fbx_node->camera >= 0) {
				fbx_node->set_name(_gen_unique_name(p_state->unique_names, "Camera3D"));
			} else {
				fbx_node->set_name(_gen_unique_name(p_state->unique_names, "Node"));
			}
		}

		fbx_node->set_name(_gen_unique_name(p_state->unique_names, fbx_node->get_name()));
	}
}

BoneAttachment3D *FBXDocument::_generate_bone_attachment(Ref<FBXState> p_state, Skeleton3D *p_skeleton, const GLTFNodeIndex p_node_index, const GLTFNodeIndex p_bone_index) {
	Ref<GLTFNode> fbx_node = p_state->nodes[p_node_index];
	Ref<GLTFNode> bone_node = p_state->nodes[p_bone_index];
	BoneAttachment3D *bone_attachment = memnew(BoneAttachment3D);
	print_verbose("FBX: Creating bone attachment for: " + fbx_node->get_name());

	ERR_FAIL_COND_V(!bone_node->joint, nullptr);

	bone_attachment->set_bone_name(bone_node->get_name());

	return bone_attachment;
}

ImporterMeshInstance3D *FBXDocument::_generate_mesh_instance(Ref<FBXState> p_state, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> fbx_node = p_state->nodes[p_node_index];

	ERR_FAIL_INDEX_V(fbx_node->mesh, p_state->meshes.size(), nullptr);

	ImporterMeshInstance3D *mi = memnew(ImporterMeshInstance3D);
	print_verbose("FBX: Creating mesh for: " + fbx_node->get_name());

	p_state->scene_mesh_instances.insert(p_node_index, mi);
	Ref<GLTFMesh> mesh = p_state->meshes.write[fbx_node->mesh];
	if (mesh.is_null()) {
		return mi;
	}
	Ref<ImporterMesh> import_mesh = mesh->get_mesh();
	if (import_mesh.is_null()) {
		return mi;
	}
	mi->set_mesh(import_mesh);
	return mi;
}

Camera3D *FBXDocument::_generate_camera(Ref<FBXState> p_state, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> fbx_node = p_state->nodes[p_node_index];

	ERR_FAIL_INDEX_V(fbx_node->camera, p_state->cameras.size(), nullptr);

	print_verbose("FBX: Creating camera for: " + fbx_node->get_name());

	Ref<GLTFCamera> c = p_state->cameras[fbx_node->camera];
	return c->to_node();
}

Light3D *FBXDocument::_generate_light(Ref<FBXState> p_state, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> fbx_node = p_state->nodes[p_node_index];

	ERR_FAIL_INDEX_V(fbx_node->light, p_state->lights.size(), nullptr);

	print_verbose("FBX: Creating light for: " + fbx_node->get_name());

	Ref<GLTFLight> l = p_state->lights[fbx_node->light];
	Light3D *light = nullptr;

	if (l->get_light_type() == "point") {
		light = memnew(OmniLight3D);
	} else if (l->get_light_type() == "directional") {
		light = memnew(DirectionalLight3D);
	} else if (l->get_light_type() == "spot") {
		light = memnew(SpotLight3D);
	} else {
		ERR_FAIL_NULL_V(light, nullptr);
	}

	if (light) {
		light->set_name(l->get_name());
		light->set_color(l->get_color());
		light->set_param(Light3D::PARAM_ENERGY, l->get_intensity());
		Dictionary additional_data = l->get_additional_data("GODOT_fbx_light");
		if (additional_data.has("castShadows")) {
			light->set_shadow(additional_data["castShadows"]);
		}
		if (additional_data.has("castLight")) {
			light->set_visible(additional_data["castLight"]);
		}

		Transform3D transform;
		DirectionalLight3D *dir_light = Object::cast_to<DirectionalLight3D>(light);
		SpotLight3D *spot_light = Object::cast_to<SpotLight3D>(light);
		OmniLight3D *omni_light = Object::cast_to<OmniLight3D>(light);
		if (dir_light) {
			dir_light->set_transform(transform);
		} else if (spot_light) {
			spot_light->set_transform(transform);
			spot_light->set_param(SpotLight3D::PARAM_SPOT_ANGLE, l->get_outer_cone_angle() / 2.0f);
		}
		if (omni_light || spot_light) {
			light->set_param(OmniLight3D::PARAM_RANGE, 4096);
		}

// This is "correct", but FBX files may have unexpected decay modes.
// Also does not match with what FBX2glTF does, so it might be better to not do any of this..
#if 0
		if (omni_light || spot_light) {
			float attenuation = 1.0f;
			if (additional_data.has("decay")) {
				String decay_type = additional_data["decay"];
				if (decay_type == "none") {
					attenuation = 0.001f;
				} else if (decay_type == "linear") {
					attenuation = 1.0f;
				} else if (decay_type == "quadratic") {
					attenuation = 2.0f;
				} else if (decay_type == "cubic") {
					attenuation = 3.0f;
				}
			}
			light->set_param(Light3D::PARAM_ATTENUATION, attenuation);
		}
#endif

		if (spot_light) {
			// Line of best fit derived from guessing, see https://www.desmos.com/calculator/biiflubp8b
			// The points in desmos are not exact, except for (1, infinity).
			float angle_ratio = l->get_inner_cone_angle() / l->get_outer_cone_angle();
			float angle_attenuation = 0.2 / (1 - angle_ratio) - 0.1;
			light->set_param(SpotLight3D::PARAM_SPOT_ATTENUATION, angle_attenuation);
		}
	}

	return light;
}

Node3D *FBXDocument::_generate_spatial(Ref<FBXState> p_state, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> fbx_node = p_state->nodes[p_node_index];

	Node3D *spatial = memnew(Node3D);
	print_verbose("FBX: Converting spatial: " + fbx_node->get_name());

	return spatial;
}

void FBXDocument::_generate_scene_node(Ref<FBXState> p_state, const GLTFNodeIndex p_node_index, Node *p_scene_parent, Node *p_scene_root) {
	Ref<GLTFNode> fbx_node = p_state->nodes[p_node_index];

	if (fbx_node->skeleton >= 0) {
		_generate_skeleton_bone_node(p_state, p_node_index, p_scene_parent, p_scene_root);
		return;
	}

	Node3D *current_node = nullptr;

	// Is our parent a skeleton
	Skeleton3D *active_skeleton = Object::cast_to<Skeleton3D>(p_scene_parent);

	const bool non_bone_parented_to_skeleton = active_skeleton;

	// skinned meshes must not be placed in a bone attachment.
	if (non_bone_parented_to_skeleton && fbx_node->skin < 0) {
		// Bone Attachment - Parent Case
		BoneAttachment3D *bone_attachment = _generate_bone_attachment(p_state, active_skeleton, p_node_index, fbx_node->parent);

		p_scene_parent->add_child(bone_attachment, true);
		bone_attachment->set_owner(p_scene_root);

		// There is no fbx_node that represent this, so just directly create a unique name
		bone_attachment->set_name(fbx_node->get_name());

		// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
		// and attach it to the bone_attachment
		p_scene_parent = bone_attachment;
	}
	if (!current_node) {
		if (fbx_node->skin >= 0 && fbx_node->mesh >= 0 && !fbx_node->children.is_empty()) {
			current_node = _generate_spatial(p_state, p_node_index);
			Node3D *mesh_inst = _generate_mesh_instance(p_state, p_node_index);
			mesh_inst->set_name(fbx_node->get_name());

			current_node->add_child(mesh_inst, true);
		} else if (fbx_node->mesh >= 0) {
			current_node = _generate_mesh_instance(p_state, p_node_index);
		} else if (fbx_node->camera >= 0) {
			current_node = _generate_camera(p_state, p_node_index);
		} else if (fbx_node->light >= 0) {
			current_node = _generate_light(p_state, p_node_index);
		} else {
			current_node = _generate_spatial(p_state, p_node_index);
		}
	}

	ERR_FAIL_NULL(current_node);

	// Add the node we generated and set the owner to the scene root.
	p_scene_parent->add_child(current_node, true);
	if (current_node != p_scene_root) {
		Array args = { p_scene_root };
		current_node->propagate_call(StringName("set_owner"), args);
	}
	current_node->set_transform(fbx_node->transform);
	current_node->set_name(fbx_node->get_name());

	p_state->scene_nodes.insert(p_node_index, current_node);
	for (int i = 0; i < fbx_node->children.size(); ++i) {
		_generate_scene_node(p_state, fbx_node->children[i], current_node, p_scene_root);
	}
}

void FBXDocument::_generate_skeleton_bone_node(Ref<FBXState> p_state, const GLTFNodeIndex p_node_index, Node *p_scene_parent, Node *p_scene_root) {
	Ref<GLTFNode> fbx_node = p_state->nodes[p_node_index];

	Node3D *current_node = nullptr;

	Skeleton3D *skeleton = p_state->skeletons[fbx_node->skeleton]->godot_skeleton;
	// In this case, this node is already a bone in skeleton.
	const bool is_skinned_mesh = (fbx_node->skin >= 0 && fbx_node->mesh >= 0);
	const bool requires_extra_node = (fbx_node->mesh >= 0 || fbx_node->camera >= 0 || fbx_node->light >= 0);

	Skeleton3D *active_skeleton = Object::cast_to<Skeleton3D>(p_scene_parent);
	if (active_skeleton != skeleton) {
		if (active_skeleton) {
			// Should no longer be possible.
			ERR_PRINT(vformat("FBX: Generating scene detected direct parented Skeletons at node %d", p_node_index));
			BoneAttachment3D *bone_attachment = _generate_bone_attachment(p_state, active_skeleton, p_node_index, fbx_node->parent);
			p_scene_parent->add_child(bone_attachment, true);
			bone_attachment->set_owner(p_scene_root);
			// There is no fbx_node that represent this, so just directly create a unique name
			bone_attachment->set_name(_gen_unique_name(p_state->unique_names, "BoneAttachment3D"));
			// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
			// and attach it to the bone_attachment
			p_scene_parent = bone_attachment;
		}
		if (skeleton->get_parent() == nullptr) {
			p_scene_parent->add_child(skeleton, true);
			skeleton->set_owner(p_scene_root);
		}
	}

	active_skeleton = skeleton;
	current_node = active_skeleton;
	if (active_skeleton) {
		p_scene_parent = active_skeleton;
	}

	if (requires_extra_node) {
		current_node = nullptr;
		// skinned meshes must not be placed in a bone attachment.
		if (!is_skinned_mesh) {
			// Bone Attachment - Same Node Case
			BoneAttachment3D *bone_attachment = _generate_bone_attachment(p_state, active_skeleton, p_node_index, p_node_index);

			p_scene_parent->add_child(bone_attachment, true);
			bone_attachment->set_owner(p_scene_root);

			// There is no fbx_node that represent this, so just directly create a unique name
			bone_attachment->set_name(fbx_node->get_name());

			// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
			// and attach it to the bone_attachment
			p_scene_parent = bone_attachment;
		}
		// TODO: 20240118 // fire
		// // Check if any GLTFDocumentExtension classes want to generate a node for us.
		// for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		// 	ERR_CONTINUE(ext.is_null());
		// 	current_node = ext->generate_scene_node(p_state, fbx_node, p_scene_parent);
		// 	if (current_node) {
		// 		break;
		// 	}
		// }
		// If none of our GLTFDocumentExtension classes generated us a node, we generate one.
		if (!current_node) {
			if (fbx_node->mesh >= 0) {
				current_node = _generate_mesh_instance(p_state, p_node_index);
			} else if (fbx_node->camera >= 0) {
				current_node = _generate_camera(p_state, p_node_index);
			} else {
				current_node = _generate_spatial(p_state, p_node_index);
			}
		}
		// Add the node we generated and set the owner to the scene root.
		p_scene_parent->add_child(current_node, true);
		if (current_node != p_scene_root) {
			Array args = { p_scene_root };
			current_node->propagate_call(StringName("set_owner"), args);
		}
		// Do not set transform here. Transform is already applied to our bone.
		current_node->set_name(fbx_node->get_name());
	}

	p_state->scene_nodes.insert(p_node_index, current_node);

	for (int i = 0; i < fbx_node->children.size(); ++i) {
		_generate_scene_node(p_state, fbx_node->children[i], active_skeleton, p_scene_root);
	}
}

void FBXDocument::_import_animation(Ref<FBXState> p_state, AnimationPlayer *p_animation_player, const GLTFAnimationIndex p_index, const bool p_trimming, const bool p_remove_immutable_tracks) {
	Ref<GLTFAnimation> anim = p_state->animations[p_index];

	String anim_name = anim->get_name();
	if (anim_name.is_empty()) {
		// No node represent these, and they are not in the hierarchy, so just make a unique name
		anim_name = _gen_unique_name(p_state->unique_names, "Animation");
	}

	Ref<Animation> animation;
	animation.instantiate();
	animation->set_name(anim_name);
	animation->set_step(1.0 / p_state->get_bake_fps());

	if (anim->get_loop()) {
		animation->set_loop_mode(Animation::LOOP_LINEAR);
	}

	Dictionary additional_animation_data = anim->get_additional_data("GODOT_animation_time_begin_time_end");

	double anim_start_offset = p_trimming ? double(additional_animation_data["time_begin"]) : 0.0;

	for (const KeyValue<int, GLTFAnimation::NodeTrack> &track_i : anim->get_node_tracks()) {
		const GLTFAnimation::NodeTrack &track = track_i.value;
		//need to find the path: for skeletons, weight tracks will affect the mesh
		NodePath node_path;
		//for skeletons, transform tracks always affect bones
		NodePath transform_node_path;
		GLTFNodeIndex node_index = track_i.key;
		Node *root = p_animation_player->get_parent();
		ERR_FAIL_NULL(root);
		HashMap<GLTFNodeIndex, Node *>::Iterator node_element = p_state->scene_nodes.find(node_index);
		ERR_CONTINUE_MSG(!node_element, vformat("Unable to find node %d for animation.", node_index));
		node_path = root->get_path_to(node_element->value);

		const Ref<GLTFNode> fbx_node = p_state->nodes[track_i.key];

		if (fbx_node->skeleton >= 0) {
			const Skeleton3D *sk = p_state->skeletons[fbx_node->skeleton]->godot_skeleton;
			ERR_FAIL_NULL(sk);

			const String path = String(p_animation_player->get_parent()->get_path_to(sk));
			const String bone = fbx_node->get_name();
			transform_node_path = path + ":" + bone;
		} else {
			transform_node_path = node_path;
		}

		// Animated TRS properties will not affect a skinned mesh.
		const bool transform_affects_skinned_mesh_instance = fbx_node->skeleton < 0 && fbx_node->skin >= 0;
		if ((track.rotation_track.values.size() || track.position_track.values.size() || track.scale_track.values.size()) && !transform_affects_skinned_mesh_instance) {
			// Make a transform track.
			int base_idx = animation->get_track_count();
			int position_idx = -1;
			int rotation_idx = -1;
			int scale_idx = -1;

			if (track.position_track.values.size()) {
				bool is_default = true; // Discard the track if all it contains is default values.
				if (p_remove_immutable_tracks) {
					Vector3 base_pos = p_state->nodes[track_i.key]->transform.origin;
					for (int i = 0; i < track.position_track.times.size(); i++) {
						Vector3 value = track.position_track.values[track.position_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + i * 3) : i];
						if (!value.is_equal_approx(base_pos)) {
							is_default = false;
							break;
						}
					}
				}
				if (!p_remove_immutable_tracks || !is_default) {
					position_idx = base_idx;
					animation->add_track(Animation::TYPE_POSITION_3D);
					animation->track_set_path(position_idx, transform_node_path);
					animation->track_set_imported(position_idx, true); // Helps merging positions later.
					base_idx++;
				}
			}
			if (track.rotation_track.values.size()) {
				bool is_default = true; // Discard the track if all the track contains is the default values.
				if (p_remove_immutable_tracks) {
					Quaternion base_rot = p_state->nodes[track_i.key]->transform.basis.get_rotation_quaternion();
					for (int i = 0; i < track.rotation_track.times.size(); i++) {
						Quaternion value = track.rotation_track.values[track.rotation_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + i * 3) : i].normalized();
						if (!value.is_equal_approx(base_rot)) {
							is_default = false;
							break;
						}
					}
				}
				if (!p_remove_immutable_tracks || !is_default) {
					rotation_idx = base_idx;
					animation->add_track(Animation::TYPE_ROTATION_3D);
					animation->track_set_path(rotation_idx, transform_node_path);
					animation->track_set_imported(rotation_idx, true); //helps merging later
					base_idx++;
				}
			}
			if (track.scale_track.values.size()) {
				bool is_default = true; // Discard the track if all the track contains is the default values.
				if (p_remove_immutable_tracks) {
					Vector3 base_scale = p_state->nodes[track_i.key]->transform.basis.get_scale();
					for (int i = 0; i < track.scale_track.times.size(); i++) {
						Vector3 value = track.scale_track.values[track.scale_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + i * 3) : i];
						if (!value.is_equal_approx(base_scale)) {
							is_default = false;
							break;
						}
					}
				}
				if (!p_remove_immutable_tracks || !is_default) {
					scale_idx = base_idx;
					animation->add_track(Animation::TYPE_SCALE_3D);
					animation->track_set_path(scale_idx, transform_node_path);
					animation->track_set_imported(scale_idx, true); //helps merging later
					base_idx++;
				}
			}

			if (position_idx != -1) {
				animation->track_set_interpolation_type(position_idx, Animation::INTERPOLATION_LINEAR);
				for (int j = 0; j < track.position_track.times.size(); j++) {
					const float t = track.position_track.times[j] - anim_start_offset;
					const Vector3 value = track.position_track.values[j];
					animation->position_track_insert_key(position_idx, t, value);
				}
			}

			if (rotation_idx != -1) {
				animation->track_set_interpolation_type(rotation_idx, Animation::INTERPOLATION_LINEAR);
				for (int j = 0; j < track.rotation_track.times.size(); j++) {
					const float t = track.rotation_track.times[j] - anim_start_offset;
					const Quaternion value = track.rotation_track.values[j];
					animation->rotation_track_insert_key(rotation_idx, t, value);
				}
			}

			if (scale_idx != -1) {
				animation->track_set_interpolation_type(scale_idx, Animation::INTERPOLATION_LINEAR);
				for (int j = 0; j < track.scale_track.times.size(); j++) {
					const float t = track.scale_track.times[j] - anim_start_offset;
					const Vector3 value = track.scale_track.values[j];
					animation->scale_track_insert_key(scale_idx, t, value);
				}
			}
		}
	}

	Dictionary blend_shape_animations = anim->get_additional_data("GODOT_blend_shape_animations");

	for (GLTFNodeIndex node_index = 0; node_index < p_state->nodes.size(); node_index++) {
		Ref<GLTFNode> node = p_state->nodes[node_index];
		if (node->mesh < 0) {
			continue;
		}

		// For meshes, especially skinned meshes, there are cases where it will be added as a child.
		NodePath mesh_instance_node_path;

		Node *root = p_animation_player->get_parent();
		ERR_FAIL_NULL(root);
		HashMap<GLTFNodeIndex, Node *>::Iterator node_element = p_state->scene_nodes.find(node_index);
		ERR_CONTINUE_MSG(!node_element, vformat("Unable to find node %d for animation.", node_index));
		NodePath node_path = root->get_path_to(node_element->value);
		HashMap<GLTFNodeIndex, ImporterMeshInstance3D *>::Iterator mesh_instance_element = p_state->scene_mesh_instances.find(node_index);
		if (mesh_instance_element) {
			mesh_instance_node_path = root->get_path_to(mesh_instance_element->value);
		} else {
			mesh_instance_node_path = node_path;
		}

		Ref<GLTFMesh> mesh = p_state->meshes[node->mesh];
		ERR_CONTINUE(mesh.is_null());
		ERR_CONTINUE(mesh->get_mesh().is_null());
		ERR_CONTINUE(mesh->get_mesh()->get_mesh().is_null());

		Dictionary mesh_additional_data = mesh->get_additional_data("GODOT_mesh_blend_channels");
		Vector<int> blend_channels = mesh_additional_data["blend_channels"];

		for (int i = 0; i < blend_channels.size(); i++) {
			int blend_i = blend_channels[i];
			if (!blend_shape_animations.has(blend_i)) {
				continue;
			}
			Dictionary blend_track = blend_shape_animations[blend_i];

			GLTFAnimation::Channel<real_t> weights;
			weights.interpolation = GLTFAnimation::INTERP_LINEAR;
			weights.times = blend_track["times"];
			weights.values = blend_track["values"];

			const String blend_path = String(mesh_instance_node_path) + ":" + String(mesh->get_mesh()->get_blend_shape_name(i));
			const int track_idx = animation->get_track_count();
			animation->add_track(Animation::TYPE_BLEND_SHAPE);
			animation->track_set_path(track_idx, blend_path);
			animation->track_set_imported(track_idx, true); // Helps merging later.

			animation->track_set_interpolation_type(track_idx, Animation::INTERPOLATION_LINEAR);
			for (int j = 0; j < weights.times.size(); j++) {
				const double t = weights.times[j] - anim_start_offset;
				const real_t attribs = weights.values[j];
				animation->blend_shape_track_insert_key(track_idx, t, attribs);
			}
		}
	}
	double time_begin = additional_animation_data["time_begin"];
	double time_end = additional_animation_data["time_end"];
	double length = p_trimming ? time_end - time_begin : time_end;
	animation->set_length(length);

	Ref<AnimationLibrary> library;
	if (!p_animation_player->has_animation_library("")) {
		library.instantiate();
		p_animation_player->add_animation_library("", library);
	} else {
		library = p_animation_player->get_animation_library("");
	}
	library->add_animation(anim_name, animation);
}

void FBXDocument::_process_mesh_instances(Ref<FBXState> p_state, Node *p_scene_root) {
	for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); ++node_i) {
		Ref<GLTFNode> node = p_state->nodes[node_i];

		if (node.is_null() || !(node->skin >= 0 && node->mesh >= 0)) {
			continue;
		}

		const GLTFSkinIndex skin_i = node->skin;

		ImporterMeshInstance3D *mi = nullptr;
		HashMap<GLTFNodeIndex, ImporterMeshInstance3D *>::Iterator mi_element = p_state->scene_mesh_instances.find(node_i);
		if (!mi_element) {
			HashMap<GLTFNodeIndex, Node *>::Iterator si_element = p_state->scene_nodes.find(node_i);
			ERR_CONTINUE_MSG(!si_element, vformat("Unable to find node %d", node_i));
			mi = Object::cast_to<ImporterMeshInstance3D>(si_element->value);
			ERR_CONTINUE_MSG(mi == nullptr, vformat("Unable to cast node %d of type %s to ImporterMeshInstance3D", node_i, si_element->value->get_class_name()));
		} else {
			mi = mi_element->value;
		}

		bool is_skin_valid = node->skin >= 0;
		bool is_skin_accessible = is_skin_valid && node->skin < p_state->skins.size();
		bool is_valid = is_skin_accessible && p_state->skins.write[node->skin]->skeleton >= 0;

		if (!is_valid) {
			continue;
		}

		const GLTFSkeletonIndex skel_i = p_state->skins.write[node->skin]->skeleton;
		Ref<GLTFSkeleton> fbx_skeleton = p_state->skeletons.write[skel_i];
		Skeleton3D *skeleton = fbx_skeleton->godot_skeleton;
		ERR_CONTINUE_MSG(skeleton == nullptr, vformat("Unable to find Skeleton for node %d skin %d", node_i, skin_i));

		mi->get_parent()->remove_child(mi);
		mi->set_owner(nullptr);
		skeleton->add_child(mi, true);
		mi->set_owner(skeleton->get_owner());

		mi->set_skin(p_state->skins.write[skin_i]->godot_skin);
		mi->set_skeleton_path(mi->get_path_to(skeleton));
		mi->set_transform(Transform3D());
	}
}

Error FBXDocument::_parse(Ref<FBXState> p_state, const String &p_path, Ref<FileAccess> p_file) {
	p_state->scene.reset();

	Error err = ERR_INVALID_DATA;
	if (p_file.is_null()) {
		return FAILED;
	}

	ufbx_load_opts opts = {};
	opts.target_axes = ufbx_axes_right_handed_y_up;
	opts.target_unit_meters = 1.0f;
	opts.space_conversion = UFBX_SPACE_CONVERSION_MODIFY_GEOMETRY;
	if (!p_state->get_allow_geometry_helper_nodes()) {
		opts.geometry_transform_handling = UFBX_GEOMETRY_TRANSFORM_HANDLING_MODIFY_GEOMETRY_NO_FALLBACK;
		opts.inherit_mode_handling = UFBX_INHERIT_MODE_HANDLING_COMPENSATE_NO_FALLBACK;
	} else {
		opts.geometry_transform_handling = UFBX_GEOMETRY_TRANSFORM_HANDLING_HELPER_NODES;
		opts.inherit_mode_handling = UFBX_INHERIT_MODE_HANDLING_COMPENSATE;
	}
	opts.pivot_handling = UFBX_PIVOT_HANDLING_ADJUST_TO_PIVOT;
	opts.geometry_transform_helper_name.data = "GeometryTransformHelper";
	opts.geometry_transform_helper_name.length = SIZE_MAX;
	opts.scale_helper_name.data = "ScaleHelper";
	opts.scale_helper_name.length = SIZE_MAX;
	opts.node_depth_limit = 512;
	opts.target_camera_axes = ufbx_axes_right_handed_y_up;
	opts.target_light_axes = ufbx_axes_right_handed_y_up;
	opts.clean_skin_weights = true;
	if (p_state->discard_meshes_and_materials) {
		opts.ignore_geometry = true;
		opts.ignore_embedded = true;
	}
	opts.generate_missing_normals = true;

	ThreadPoolFBX thread_pool;
	thread_pool.pool = WorkerThreadPool::get_singleton();

	opts.thread_opts.pool.init_fn = &_thread_pool_init_fn;
	opts.thread_opts.pool.run_fn = &_thread_pool_run_fn;
	opts.thread_opts.pool.wait_fn = &_thread_pool_wait_fn;
	opts.thread_opts.pool.user = &thread_pool;
	opts.thread_opts.memory_limit = 64 * 1024 * 1024;

	ufbx_error error;
	ufbx_stream file_stream = {};
	file_stream.read_fn = &_file_access_read_fn;
	file_stream.skip_fn = &_file_access_skip_fn;
	file_stream.user = p_file.ptr();
	p_state->scene.reset(ufbx_load_stream(&file_stream, &opts, &error));

	if (!p_state->scene.get()) {
		char err_buf[512];
		ufbx_format_error(err_buf, sizeof(err_buf), &error);
		ERR_FAIL_V_MSG(ERR_PARSE_ERROR, err_buf);
	}

	const int max_warning_count = 10;
	int warning_count[UFBX_WARNING_TYPE_COUNT] = {};
	int ignored_warning_count = 0;
	for (const ufbx_warning &warning : p_state->scene->metadata.warnings) {
		if (warning_count[warning.type]++ < max_warning_count) {
			if (warning.count > 1) {
				WARN_PRINT(vformat("FBX: ufbx warning: %s (x%d)", _as_string(warning.description), (int)warning.count));
			} else {
				String element_name;
				if (warning.element_id != UFBX_NO_INDEX) {
					element_name = _find_element_name(p_state->scene->elements[warning.element_id]);
				}
				if (!element_name.is_empty()) {
					WARN_PRINT(vformat("FBX: ufbx warning in '%s': %s", element_name, _as_string(warning.description)));
				} else {
					WARN_PRINT(vformat("FBX: ufbx warning: %s", _as_string(warning.description)));
				}
			}
		} else {
			ignored_warning_count++;
		}
	}
	if (ignored_warning_count > 0) {
		WARN_PRINT(vformat("FBX: ignored %d further ufbx warnings", ignored_warning_count));
	}

	document_extensions.clear();
	for (Ref<GLTFDocumentExtension> ext : all_document_extensions) {
		ERR_CONTINUE(ext.is_null());
		err = ext->import_preflight(p_state, p_state->json["extensionsUsed"]);
		if (err == OK) {
			document_extensions.push_back(ext);
		}
	}

	err = _parse_fbx_state(p_state, p_path);
	ERR_FAIL_COND_V(err != OK, err);

	return OK;
}

Node *FBXDocument::generate_scene(Ref<GLTFState> p_state, float p_bake_fps, bool p_trimming, bool p_remove_immutable_tracks) {
	Ref<FBXState> state = p_state;
	ERR_FAIL_COND_V(state.is_null(), nullptr);
	ERR_FAIL_INDEX_V(0, state->root_nodes.size(), nullptr);
	p_state->set_bake_fps(p_bake_fps);
	GLTFNodeIndex fbx_root = state->root_nodes.write[0];
	Node *fbx_root_node = state->get_scene_node(fbx_root);
	Node *root = fbx_root_node;
	if (root && root->get_owner() && root->get_owner() != root) {
		root = root->get_owner();
	}
	ERR_FAIL_NULL_V(root, nullptr);
	_process_mesh_instances(state, root);
	if (state->get_create_animations() && state->animations.size()) {
		AnimationPlayer *ap = memnew(AnimationPlayer);
		root->add_child(ap, true);
		ap->set_owner(root);
		for (int i = 0; i < state->animations.size(); i++) {
			_import_animation(state, ap, i, p_trimming, p_remove_immutable_tracks);
		}
	}
	for (KeyValue<GLTFNodeIndex, Node *> E : state->scene_nodes) {
		ERR_CONTINUE(!E.value);
		for (Ref<GLTFDocumentExtension> ext : document_extensions) {
			ERR_CONTINUE(ext.is_null());
			Dictionary node_json;
			if (state->json.has("nodes")) {
				Array nodes = state->json["nodes"];
				if (0 <= E.key && E.key < nodes.size()) {
					node_json = nodes[E.key];
				}
			}
			Ref<GLTFNode> gltf_node = state->nodes[E.key];
			Error err = ext->import_node(p_state, gltf_node, node_json, E.value);
			ERR_CONTINUE(err != OK);
		}
	}
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		Error err = ext->import_post(p_state, root);
		ERR_CONTINUE(err != OK);
	}
	ERR_FAIL_NULL_V(root, nullptr);
	return root;
}

Error FBXDocument::append_from_buffer(const PackedByteArray &p_bytes, const String &p_base_path, Ref<GLTFState> p_state, uint32_t p_flags) {
	Ref<FBXState> state = p_state;
	ERR_FAIL_COND_V(state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_NULL_V(p_bytes.ptr(), ERR_INVALID_DATA);
	Error err = FAILED;
	state->use_named_skin_binds = p_flags & FBX_IMPORT_USE_NAMED_SKIN_BINDS;
	state->discard_meshes_and_materials = p_flags & FBX_IMPORT_DISCARD_MESHES_AND_MATERIALS;

	Ref<FileAccessMemory> file_access;
	file_access.instantiate();
	file_access->open_custom(p_bytes.ptr(), p_bytes.size());
	state->base_path = p_base_path.get_base_dir();
	err = _parse(state, state->base_path, file_access);
	ERR_FAIL_COND_V(err != OK, err);
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		err = ext->import_post_parse(state);
		ERR_FAIL_COND_V(err != OK, err);
	}
	return OK;
}

Error FBXDocument::_parse_fbx_state(Ref<FBXState> p_state, const String &p_search_path) {
	Error err;

	// Abort parsing if the scene is not loaded.
	ERR_FAIL_NULL_V(p_state->scene.get(), ERR_PARSE_ERROR);

	/* PARSE SCENE */
	err = _parse_scenes(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE NODES */
	err = _parse_nodes(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	if (!p_state->discard_meshes_and_materials) {
		/* PARSE IMAGES */
		err = _parse_images(p_state, p_search_path);

		ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

		/* PARSE MATERIALS */
		err = _parse_materials(p_state);

		ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);
	}

	/* PARSE SKINS */
	err = _parse_skins(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* DETERMINE SKELETONS */
	if (p_state->get_import_as_skeleton_bones()) {
		err = SkinTool::_determine_skeletons(p_state->skins, p_state->nodes, p_state->skeletons, p_state->root_nodes, true);
	} else {
		err = SkinTool::_determine_skeletons(p_state->skins, p_state->nodes, p_state->skeletons, Vector<GLTFNodeIndex>(), _naming_version < 2);
	}
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* CREATE SKELETONS */
	err = SkinTool::_create_skeletons(p_state->unique_names, p_state->skins, p_state->nodes, p_state->skeleton3d_to_fbx_skeleton, p_state->skeletons, p_state->scene_nodes, _naming_version);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* CREATE SKINS */
	err = SkinTool::_create_skins(p_state->skins, p_state->nodes, p_state->use_named_skin_binds, p_state->unique_names);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE MESHES (we have enough info now) */
	err = _parse_meshes(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE LIGHTS */
	err = _parse_lights(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE CAMERAS */
	err = _parse_cameras(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE ANIMATIONS */
	err = _parse_animations(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* ASSIGN SCENE NAMES */
	_assign_node_names(p_state);

	Node3D *root = memnew(Node3D);
	for (int32_t root_i = 0; root_i < p_state->root_nodes.size(); root_i++) {
		_generate_scene_node(p_state, p_state->root_nodes[root_i], root, root);
	}

	return OK;
}

Error FBXDocument::append_from_file(const String &p_path, Ref<GLTFState> p_state, uint32_t p_flags, const String &p_base_path) {
	Ref<FBXState> state = p_state;
	ERR_FAIL_COND_V(state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_FILE_NOT_FOUND);
	if (p_state == Ref<FBXState>()) {
		p_state.instantiate();
	}
	state->filename = p_path.get_file().get_basename();
	state->use_named_skin_binds = p_flags & FBX_IMPORT_USE_NAMED_SKIN_BINDS;
	state->discard_meshes_and_materials = p_flags & FBX_IMPORT_DISCARD_MESHES_AND_MATERIALS;
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ, &err);
	ERR_FAIL_COND_V(err != OK, ERR_FILE_CANT_OPEN);
	ERR_FAIL_COND_V(file.is_null(), ERR_FILE_CANT_OPEN);
	String base_path = p_base_path;
	if (base_path.is_empty()) {
		base_path = p_path.get_base_dir();
	}
	state->base_path = base_path;
	err = _parse(p_state, base_path, file);
	ERR_FAIL_COND_V(err != OK, err);
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		err = ext->import_post_parse(p_state);
		ERR_FAIL_COND_V(err != OK, err);
	}
	return OK;
}

void FBXDocument::_process_uv_set(PackedVector2Array &uv_array) {
	int uv_size = uv_array.size();
	for (int uv_i = 0; uv_i < uv_size; uv_i++) {
		Vector2 &uv = uv_array.write[uv_i];
		uv.y = 1.0 - uv.y;
	}
}

void FBXDocument::_zero_unused_elements(Vector<float> &cur_custom, int start, int end, int num_channels) {
	for (int32_t uv_i = start; uv_i < end; uv_i++) {
		int index = uv_i * num_channels;
		for (int channel = 0; channel < num_channels; channel++) {
			cur_custom.write[index + channel] = 0;
		}
	}
}

Error FBXDocument::_parse_lights(Ref<FBXState> p_state) {
	const ufbx_scene *fbx_scene = p_state->scene.get();
	for (size_t i = 0; i < fbx_scene->lights.count; i++) {
		const ufbx_light *fbx_light = fbx_scene->lights.data[i];
		Ref<GLTFLight> light;
		light.instantiate();
		light->set_name(_as_string(fbx_light->name));
		light->set_color(Color(fbx_light->color.x, fbx_light->color.y, fbx_light->color.z));
		light->set_intensity(fbx_light->intensity);
		switch (fbx_light->type) {
			case UFBX_LIGHT_POINT:
				light->set_light_type("point");
				break;
			case UFBX_LIGHT_DIRECTIONAL:
				light->set_light_type("directional");
				break;
			case UFBX_LIGHT_SPOT:
				light->set_light_type("spot");
				break;
			case UFBX_LIGHT_AREA:
				light->set_light_type("area");
				break;
			case UFBX_LIGHT_VOLUME:
				light->set_light_type("volume");
				break;
			default:
				light->set_light_type("unknown");
				break;
		}

		Dictionary additional_data;
		additional_data["shadow"] = fbx_light->cast_shadows;
		if (fbx_light->decay == UFBX_LIGHT_DECAY_NONE) {
			additional_data["decay"] = "none";

		} else if (fbx_light->decay == UFBX_LIGHT_DECAY_LINEAR) {
			additional_data["decay"] = "linear";

		} else if (fbx_light->decay == UFBX_LIGHT_DECAY_QUADRATIC) {
			additional_data["decay"] = "quadratic";

		} else if (fbx_light->decay == UFBX_LIGHT_DECAY_CUBIC) {
			additional_data["decay"] = "cubic";
		}

		if (fbx_light->area_shape == UFBX_LIGHT_AREA_SHAPE_RECTANGLE) {
			additional_data["areaShape"] = "rectangle";
		} else if (fbx_light->area_shape == UFBX_LIGHT_AREA_SHAPE_SPHERE) {
			additional_data["areaShape"] = "sphere";
		}

		light->set_inner_cone_angle(fbx_light->inner_angle);
		light->set_outer_cone_angle(fbx_light->outer_angle);

		additional_data["castLight"] = fbx_light->cast_light;
		additional_data["castShadows"] = fbx_light->cast_shadows;
		light->set_additional_data("GODOT_fbx_light", additional_data);
		p_state->lights.push_back(light);
	}
	print_verbose("FBX: Total lights: " + itos(p_state->lights.size()));
	return OK;
}

String FBXDocument::_get_texture_path(const String &p_base_dir, const String &p_source_file_path) const {
	// Check if the original path exists first.
	if (FileAccess::exists(p_source_file_path)) {
		return p_source_file_path.strip_edges();
	}
	const String tex_file_name = p_source_file_path.get_file();
	const Vector<String> subdirs = {
		"", "textures/", "Textures/", "images/",
		"Images/", "materials/", "Materials/",
		"maps/", "Maps/", "tex/", "Tex/"
	};
	String base_dir = p_base_dir;
	const String source_file_name = tex_file_name;
	while (!base_dir.is_empty()) {
		String old_base_dir = base_dir;
		for (int i = 0; i < subdirs.size(); ++i) {
			String full_path = base_dir.path_join(subdirs[i] + source_file_name);
			if (FileAccess::exists(full_path)) {
				return full_path.strip_edges();
			}
		}
		base_dir = base_dir.get_base_dir();
		if (base_dir == old_base_dir) {
			break;
		}
	}
	return String();
}

Error FBXDocument::_parse_skins(Ref<FBXState> p_state) {
	const ufbx_scene *fbx_scene = p_state->scene.get();
	HashMap<GLTFNodeIndex, bool> joint_mapping;

	for (const ufbx_skin_deformer *fbx_skin : fbx_scene->skin_deformers) {
		if (fbx_skin->clusters.count == 0 || fbx_skin->weights.count == 0) {
			p_state->skin_indices.push_back(-1);
			continue;
		}

		Ref<GLTFSkin> skin;
		skin.instantiate();

		skin->inverse_binds.resize(fbx_skin->clusters.count);
		for (int skin_i = 0; skin_i < static_cast<int>(fbx_skin->clusters.count); skin_i++) {
			const ufbx_skin_cluster *fbx_cluster = fbx_skin->clusters[skin_i];
			skin->inverse_binds.write[skin_i] = FBXDocument::_as_xform(fbx_cluster->geometry_to_bone);
			const GLTFNodeIndex node = fbx_cluster->bone_node->typed_id;

			skin->joints.push_back(node);
			skin->joints_original.push_back(node);
			p_state->nodes.write[node]->joint = true;
		}

		if (fbx_skin->name.length > 0) {
			skin->set_name(FBXDocument::_as_string(fbx_skin->name));
		} else {
			skin->set_name(vformat("skin_%s", itos(fbx_skin->typed_id)));
		}
		p_state->skin_indices.push_back(p_state->skins.size());
		p_state->skins.push_back(skin);
	}

	for (const ufbx_bone *fbx_bone : fbx_scene->bones) {
		for (const ufbx_node *fbx_node : fbx_bone->instances) {
			const GLTFNodeIndex node = fbx_node->typed_id;
			if (!p_state->nodes.write[node]->joint) {
				p_state->nodes.write[node]->joint = true;

				if (!(fbx_node->parent && fbx_node->parent->attrib_type == UFBX_ELEMENT_BONE)) {
					Ref<GLTFSkin> skin;
					skin.instantiate();
					skin->joints.push_back(node);
					skin->joints_original.push_back(node);
					skin->set_name(vformat("skin_%s", itos(p_state->skins.size())));
					p_state->skin_indices.push_back(p_state->skins.size());
					p_state->skins.push_back(skin);
				}
			}
		}
	}
	p_state->original_skin_indices = p_state->skin_indices.duplicate();
	Error err = SkinTool::_asset_parse_skins(
			p_state->original_skin_indices,
			p_state->skins.duplicate(),
			p_state->nodes.duplicate(),
			p_state->skin_indices,
			p_state->skins,
			joint_mapping);
	if (err != OK) {
		return err;
	}
	for (int i = 0; i < p_state->skins.size(); ++i) {
		Ref<GLTFSkin> skin = p_state->skins.write[i];
		ERR_FAIL_COND_V(skin.is_null(), ERR_PARSE_ERROR);
		// Expand and verify the skin
		ERR_FAIL_COND_V(SkinTool::_expand_skin(p_state->nodes, skin), ERR_PARSE_ERROR);
		ERR_FAIL_COND_V(SkinTool::_verify_skin(p_state->nodes, skin), ERR_PARSE_ERROR);
	}

	print_verbose("FBX: Total skins: " + itos(p_state->skins.size()));

	for (HashMap<GLTFNodeIndex, bool>::Iterator it = joint_mapping.begin(); it != joint_mapping.end(); ++it) {
		GLTFNodeIndex node_index = it->key;
		bool is_joint = it->value;
		if (is_joint) {
			if (p_state->nodes.size() > node_index) {
				p_state->nodes.write[node_index]->joint = true;
			}
		}
	}

	return OK;
}

PackedByteArray FBXDocument::generate_buffer(Ref<GLTFState> p_state) {
	return PackedByteArray();
}

void FBXDocument::_apply_scale_to_gltf_state(Ref<GLTFState> p_state, const Vector3 &p_scale) {
	ERR_FAIL_COND(p_state.is_null());

	// Scale all node transforms (translation/origin)
	for (int i = 0; i < p_state->nodes.size(); i++) {
		Ref<GLTFNode> node = p_state->nodes[i];
		if (node.is_null()) {
			continue;
		}
		Transform3D transform = node->transform;
		transform.origin = p_scale * transform.origin;
		node->transform = transform;
	}

	// Scale all mesh vertices
	for (int i = 0; i < p_state->meshes.size(); i++) {
		Ref<GLTFMesh> gltf_mesh = p_state->meshes[i];
		if (gltf_mesh.is_null()) {
			continue;
		}
		Ref<ImporterMesh> importer_mesh = gltf_mesh->get_mesh();
		if (importer_mesh.is_null()) {
			continue;
		}

		// Scale vertices in each surface
		// Use the same approach as _rescale_importer_mesh in resource_importer_scene.cpp
		const int surf_count = importer_mesh->get_surface_count();
		const int blendshape_count = importer_mesh->get_blend_shape_count();
		
		struct LocalSurfData {
			Mesh::PrimitiveType prim = {};
			Array arr;
			Array bsarr;
			Dictionary lods;
			String name;
			Ref<Material> mat;
			uint64_t fmt_compress_flags = 0;
		};
		
		Vector<LocalSurfData> surf_data_by_mesh;
		Vector<String> blendshape_names;
		
		for (int bsidx = 0; bsidx < blendshape_count; bsidx++) {
			blendshape_names.append(importer_mesh->get_blend_shape_name(bsidx));
		}
		
		for (int surf_idx = 0; surf_idx < surf_count; surf_idx++) {
			Mesh::PrimitiveType prim = importer_mesh->get_surface_primitive_type(surf_idx);
			const uint64_t fmt_compress_flags = importer_mesh->get_surface_format(surf_idx);
			Array arr = importer_mesh->get_surface_arrays(surf_idx);
			String surface_name = importer_mesh->get_surface_name(surf_idx);
			Dictionary lods;
			// Get LODs
			for (int lod_i = 0; lod_i < importer_mesh->get_surface_lod_count(surf_idx); lod_i++) {
				float lod_distance = importer_mesh->get_surface_lod_size(surf_idx, lod_i);
				Vector<int> lod_indices = importer_mesh->get_surface_lod_indices(surf_idx, lod_i);
				lods[lod_distance] = lod_indices;
			}
			Ref<Material> mat = importer_mesh->get_surface_material(surf_idx);
			
			// Scale vertices
			{
				Vector<Vector3> vertex_array = arr[Mesh::ARRAY_VERTEX];
				for (int vert_arr_i = 0; vert_arr_i < vertex_array.size(); vert_arr_i++) {
					vertex_array.write[vert_arr_i] = vertex_array[vert_arr_i] * p_scale;
				}
				arr[Mesh::ARRAY_VERTEX] = vertex_array;
			}
			
			// Scale blend shape vertices
			Array blendshapes;
			for (int bsidx = 0; bsidx < blendshape_count; bsidx++) {
				Array current_bsarr = importer_mesh->get_surface_blend_shape_arrays(surf_idx, bsidx);
				Vector<Vector3> current_bs_vertex_array = current_bsarr[Mesh::ARRAY_VERTEX];
				int current_bs_vert_arr_len = current_bs_vertex_array.size();
				for (int32_t bs_vert_arr_i = 0; bs_vert_arr_i < current_bs_vert_arr_len; bs_vert_arr_i++) {
					current_bs_vertex_array.write[bs_vert_arr_i] = current_bs_vertex_array[bs_vert_arr_i] * p_scale;
				}
				current_bsarr[Mesh::ARRAY_VERTEX] = current_bs_vertex_array;
				blendshapes.push_back(current_bsarr);
			}
			
			LocalSurfData surf_data_dictionary = LocalSurfData();
			surf_data_dictionary.prim = prim;
			surf_data_dictionary.arr = arr;
			surf_data_dictionary.bsarr = blendshapes;
			surf_data_dictionary.lods = lods;
			surf_data_dictionary.fmt_compress_flags = fmt_compress_flags;
			surf_data_dictionary.name = surface_name;
			surf_data_dictionary.mat = mat;
			
			surf_data_by_mesh.push_back(surf_data_dictionary);
		}
		
		// Clear and re-add surfaces (following _rescale_importer_mesh pattern)
		importer_mesh->clear();
		
		for (int bsidx = 0; bsidx < blendshape_count; bsidx++) {
			importer_mesh->add_blend_shape(blendshape_names[bsidx]);
		}
		
		for (int surf_idx = 0; surf_idx < surf_count; surf_idx++) {
			const Mesh::PrimitiveType prim = surf_data_by_mesh[surf_idx].prim;
			const Array arr = surf_data_by_mesh[surf_idx].arr;
			const Array bsarr = surf_data_by_mesh[surf_idx].bsarr;
			const Dictionary lods = surf_data_by_mesh[surf_idx].lods;
			const uint64_t fmt_compress_flags = surf_data_by_mesh[surf_idx].fmt_compress_flags;
			const String surface_name = surf_data_by_mesh[surf_idx].name;
			const Ref<Material> mat = surf_data_by_mesh[surf_idx].mat;
			
			importer_mesh->add_surface(prim, arr, bsarr, lods, mat, surface_name, fmt_compress_flags);
		}
	}

	// Scale all animation position tracks
	for (int i = 0; i < p_state->animations.size(); i++) {
		Ref<GLTFAnimation> gltf_anim = p_state->animations[i];
		if (gltf_anim.is_null()) {
			continue;
		}

		// Get node tracks (HashMap<int, NodeTrack>)
		HashMap<int, GLTFAnimation::NodeTrack> &node_tracks = gltf_anim->get_node_tracks();
		for (HashMap<int, GLTFAnimation::NodeTrack>::Iterator it = node_tracks.begin(); it != node_tracks.end(); ++it) {
			GLTFAnimation::NodeTrack &track = it->value;
			
			// Scale position track values
			for (int key_i = 0; key_i < track.position_track.values.size(); key_i++) {
				track.position_track.values.write[key_i] = p_scale * track.position_track.values[key_i];
			}
		}
	}

	// Scale all skin inverse bind poses
	for (int i = 0; i < p_state->skins.size(); i++) {
		Ref<GLTFSkin> gltf_skin = p_state->skins[i];
		if (gltf_skin.is_null()) {
			continue;
		}

		TypedArray<Transform3D> inverse_binds = gltf_skin->get_inverse_binds();
		for (int bind_i = 0; bind_i < inverse_binds.size(); bind_i++) {
			Transform3D bind = inverse_binds[bind_i];
			bind.origin = p_scale * bind.origin;
			inverse_binds[bind_i] = bind;
		}
		gltf_skin->set_inverse_binds(inverse_binds);
	}

	// Scale all camera properties that are in meters
	for (int i = 0; i < p_state->cameras.size(); i++) {
		Ref<GLTFCamera> gltf_camera = p_state->cameras[i];
		if (gltf_camera.is_null()) {
			continue;
		}

		// Scale depth_near, depth_far, and size_mag (all in meters)
		real_t depth_near = gltf_camera->get_depth_near();
		real_t depth_far = gltf_camera->get_depth_far();
		real_t size_mag = gltf_camera->get_size_mag();

		// Use uniform scale (assuming p_scale is uniform)
		real_t scale_factor = p_scale.x; // Use x component for uniform scaling
		gltf_camera->set_depth_near(depth_near * scale_factor);
		gltf_camera->set_depth_far(depth_far * scale_factor);
		gltf_camera->set_size_mag(size_mag * scale_factor);
	}

	// Scale all light properties that are in meters
	for (int i = 0; i < p_state->lights.size(); i++) {
		Ref<GLTFLight> gltf_light = p_state->lights[i];
		if (gltf_light.is_null()) {
			continue;
		}

		// Scale range (in meters)
		// Note: range can be INF, so check for that
		float range = gltf_light->get_range();
		if (range != Math::INF && range > 0.0f) {
			// Use uniform scale (assuming p_scale is uniform)
			real_t scale_factor = p_scale.x; // Use x component for uniform scaling
			gltf_light->set_range(range * scale_factor);
		}
		// Note: inner_cone_angle and outer_cone_angle are in radians, not meters, so no scaling needed
	}
}

// Helper struct and functions for memory-based FBX writing
// These must be at file scope (not inside write_to_filesystem) to avoid "function definition not allowed" errors
//
// NOTE: There is a known issue where ufbx_write library passes corrupted 'size' parameter
// on the second write callback call when ASAN is enabled. The corrupted value appears to be
// a pointer address (e.g., 0x1051e3dd0) instead of the actual size. This has been observed
// on ARM64 macOS with AddressSanitizer enabled.
//
// Attempted fixes (none resolved the root cause):
// 1. extern "C" linkage - ensures C calling convention
// 2. __attribute__((no_sanitize("address"))) - prevents ASAN from intercepting callback
// 3. C wrapper pattern - separates C/C++ boundary with explicit wrappers
//
// Current workaround: Sanity check rejects size > 256MB to prevent crashes.
// The export fails gracefully instead of crashing, but the underlying issue remains.
//
// Possible root causes:
// - ASAN instrumentation interfering with function pointer calls
// - ARM64 calling convention mismatch in ufbx_write library
// - Bug in ufbx_write when calling function pointers multiple times
//
// NOTE: We now use ufbxw_save_file() directly instead of stream callbacks
// to avoid all callback parameter corruption issues.

Error FBXDocument::write_to_filesystem(Ref<GLTFState> p_state, const String &p_path) {
#ifdef UFBX_WRITE_AVAILABLE
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_INVALID_PARAMETER);

	// First serialize the scene to GLTFState (similar to GLTFDocument)
	Ref<GLTFState> state = p_state;
	state->set_base_path(p_path.get_base_dir());
	state->filename = p_path.get_file();

	// Use parent class to serialize to GLTF format first
	// This populates the GLTFState with all scene data (nodes, meshes, materials, animations, etc.)
	Error err = _serialize(state);
	if (err != OK) {
		return err;
	}

	// Apply scale to convert from meters (Godot) to centimeters (FBX default)
	// This uses the same approach as Godot's import scale application
	// Scale factor: 1 meter = 100 centimeters
	const Vector3 scale_factor = Vector3(100.0, 100.0, 100.0);
	_apply_scale_to_gltf_state(state, scale_factor);

	// Convert GLTFState to FBX using ufbx_write

	// Create FBX write scene
	ufbxw_scene_opts scene_opts = {};
	ufbxw_scene *write_scene = ufbxw_create_scene(&scene_opts);
	if (!write_scene) {
		return ERR_CANT_CREATE;
	}

	// Convert GLTF nodes to FBX nodes
	// Create a mapping from GLTF node indices to FBX nodes
	HashMap<int, ufbxw_node> gltf_to_fbx_nodes;

	// First pass: Create all FBX nodes
	for (int i = 0; i < state->nodes.size(); i++) {
		Ref<GLTFNode> gltf_node = state->nodes[i];
		if (gltf_node.is_null()) {
			continue;
		}

		// Create FBX node
		ufbxw_node fbx_node = ufbxw_create_node(write_scene);
		if (fbx_node.id == 0) {
			ufbxw_free_scene(write_scene);
			return ERR_CANT_CREATE;
		}

		// Set node name
		String node_name = gltf_node->get_name();
		if (node_name.is_empty()) {
			node_name = "Node" + itos(i);
		}
		// Use utf8() to get CharString, then use length() to avoid strlen on boundary strings
		CharString node_name_utf8 = node_name.utf8();
		ufbxw_set_name_len(write_scene, fbx_node.id, node_name_utf8.get_data(), node_name_utf8.length());

		// Set transform (already scaled to centimeters in _apply_scale_to_gltf_state)
		Transform3D transform = gltf_node->transform;
		Vector3 translation = transform.origin;
		Quaternion rotation = transform.basis.get_rotation_quaternion();
		Vector3 scale = transform.basis.get_scale();

		ufbxw_vec3 fbx_translation = { (float)translation.x, (float)translation.y, (float)translation.z };
		ufbxw_vec3 fbx_scale = { (float)scale.x, (float)scale.y, (float)scale.z };

		ufbxw_node_set_translation(write_scene, fbx_node, fbx_translation);
		ufbxw_node_set_scaling(write_scene, fbx_node, fbx_scale);

		// Convert quaternion to Euler angles (degrees) for FBX
		Vector3 euler = rotation.get_euler();
		ufbxw_vec3 fbx_rotation = { Math::rad_to_deg((float)euler.x), Math::rad_to_deg((float)euler.y), Math::rad_to_deg((float)euler.z) };
		ufbxw_node_set_rotation(write_scene, fbx_node, fbx_rotation);

		// Set visibility
		ufbxw_node_set_visibility(write_scene, fbx_node, gltf_node->visible);

		// Store mapping
		gltf_to_fbx_nodes[i] = fbx_node;
	}

	// Second pass: Set parent-child relationships
	for (int i = 0; i < state->nodes.size(); i++) {
		Ref<GLTFNode> gltf_node = state->nodes[i];
		if (gltf_node.is_null()) {
			continue;
		}

		if (!gltf_to_fbx_nodes.has(i)) {
			continue;
		}

		ufbxw_node fbx_node = gltf_to_fbx_nodes[i];
		GLTFNodeIndex parent_idx = gltf_node->parent;

		if (parent_idx >= 0 && parent_idx < state->nodes.size() && gltf_to_fbx_nodes.has(parent_idx)) {
			ufbxw_node fbx_parent = gltf_to_fbx_nodes[parent_idx];
			ufbxw_node_set_parent(write_scene, fbx_node, fbx_parent);
		}
	}

	// Third pass: Export meshes
	// This converts GLTF meshes to FBX meshes and attaches them to nodes
	// Create one FBX mesh per surface (avoid combining surfaces)
	HashMap<GLTFMeshIndex, Vector<ufbxw_mesh>> gltf_to_fbx_meshes; // Multiple meshes per GLTF mesh (one per surface)
	// Store per-face material indices for each FBX mesh (to be applied after materials are created)
	HashMap<ufbxw_id, Vector<int32_t>> fbx_mesh_face_material_indices;

	for (int mesh_i = 0; mesh_i < state->meshes.size(); mesh_i++) {
		Ref<GLTFMesh> gltf_mesh = state->meshes[mesh_i];
		if (gltf_mesh.is_null()) {
			continue;
		}

		Ref<ImporterMesh> importer_mesh = gltf_mesh->get_mesh();
		if (importer_mesh.is_null() || importer_mesh->get_surface_count() == 0) {
			continue;
		}

		// Get instance materials to map surface indices to material indices
		TypedArray<Material> instance_materials = gltf_mesh->get_instance_materials();

		// Process each surface as a separate FBX mesh
		for (int surface_i = 0; surface_i < importer_mesh->get_surface_count(); surface_i++) {
			Array surface_arrays = importer_mesh->get_surface_arrays(surface_i);

			// Check if array is valid and has required size
			if (surface_arrays.size() < Mesh::ARRAY_MAX) {
				continue; // Invalid array, try next surface
			}

			// Get vertices - meshes must have vertices
			Variant vertex_var = surface_arrays[Mesh::ARRAY_VERTEX];
			if (vertex_var.get_type() != Variant::PACKED_VECTOR3_ARRAY) {
				continue; // Invalid vertex data, try next surface
			}
			Vector<Vector3> vertices = vertex_var;
			if (vertices.size() == 0) {
				continue; // Try next surface
			}

			// Get indices (triangles) - meshes should have triangles
			Variant index_var = surface_arrays[Mesh::ARRAY_INDEX];
			if (index_var.get_type() != Variant::PACKED_INT32_ARRAY && index_var.get_type() != Variant::NIL) {
				continue; // Invalid index data, try next surface
			}
			Vector<int> indices = index_var;
			if (indices.size() == 0) {
				continue; // Try next surface
			}

			// Create a separate FBX mesh for this surface
			ufbxw_mesh fbx_mesh = ufbxw_create_mesh(write_scene);
			if (fbx_mesh.id == 0) {
				continue; // Skip if creation failed
			}

			// Set mesh name with surface index
			String mesh_name = gltf_mesh->get_original_name();
			if (mesh_name.is_empty()) {
				mesh_name = "Mesh" + itos(mesh_i);
			}
			if (importer_mesh->get_surface_count() > 1) {
				mesh_name += "_Surface" + itos(surface_i);
			}
			CharString mesh_name_utf8 = mesh_name.utf8();
			ufbxw_set_name_len(write_scene, fbx_mesh.id, mesh_name_utf8.get_data(), mesh_name_utf8.length());

			// Convert vertices (already scaled to centimeters in _apply_scale_to_gltf_state)
			Vector<ufbxw_vec3> fbx_vertices;
			fbx_vertices.resize(vertices.size());
			for (int i = 0; i < vertices.size(); i++) {
				fbx_vertices.write[i] = { (ufbxw_real)vertices[i].x, (ufbxw_real)vertices[i].y, (ufbxw_real)vertices[i].z };
			}
			ufbxw_vec3_buffer vertices_buffer = ufbxw_copy_vec3_array(write_scene, fbx_vertices.ptr(), fbx_vertices.size());
			ufbxw_mesh_set_vertices(write_scene, fbx_mesh, vertices_buffer);

			// Convert indices
			// FBX uses counter-clockwise winding, but Godot uses clockwise
			// Swap first and third index of each triangle to convert winding order
			Vector<int32_t> fbx_indices;
			fbx_indices.resize(indices.size());
			int triangle_count = indices.size() / 3;
			for (int tri = 0; tri < triangle_count; tri++) {
				int base = tri * 3;
				// Swap index 0 and 2 to reverse winding order (Godot clockwise -> FBX counter-clockwise)
				fbx_indices.write[base + 0] = (int32_t)indices[base + 2];
				fbx_indices.write[base + 1] = (int32_t)indices[base + 1];
				fbx_indices.write[base + 2] = (int32_t)indices[base + 0];
			}
			ufbxw_int_buffer indices_buffer = ufbxw_copy_int_array(write_scene, fbx_indices.ptr(), fbx_indices.size());
			ufbxw_mesh_set_triangles(write_scene, fbx_mesh, indices_buffer);

			// Get and set normals
			Variant normal_var = surface_arrays[Mesh::ARRAY_NORMAL];
			if (normal_var.get_type() == Variant::PACKED_VECTOR3_ARRAY) {
				Vector<Vector3> normals = normal_var;
				if (normals.size() == vertices.size()) {
					Vector<ufbxw_vec3> fbx_normals;
					fbx_normals.resize(normals.size());
					for (int i = 0; i < normals.size(); i++) {
						fbx_normals.write[i] = { (ufbxw_real)normals[i].x, (ufbxw_real)normals[i].y, (ufbxw_real)normals[i].z };
					}
					ufbxw_vec3_buffer normals_buffer = ufbxw_copy_vec3_array(write_scene, fbx_normals.ptr(), fbx_normals.size());
					ufbxw_mesh_set_normals(write_scene, fbx_mesh, normals_buffer, UFBXW_ATTRIBUTE_MAPPING_VERTEX);
				}
			}

			// Get and set UVs (texture coordinates) - support up to 8 UV sets
			// UV set 0: ARRAY_TEX_UV
			Variant uv0_var = surface_arrays[Mesh::ARRAY_TEX_UV];
			if (uv0_var.get_type() == Variant::PACKED_VECTOR2_ARRAY) {
				Vector<Vector2> uv0 = uv0_var;
				if (uv0.size() == vertices.size()) {
					Vector<ufbxw_vec2> fbx_uvs;
					fbx_uvs.resize(uv0.size());
					for (int i = 0; i < uv0.size(); i++) {
						fbx_uvs.write[i] = { (ufbxw_real)uv0[i].x, (ufbxw_real)uv0[i].y };
					}
					ufbxw_vec2_buffer uvs_buffer = ufbxw_copy_vec2_array(write_scene, fbx_uvs.ptr(), fbx_uvs.size());
					Vector<int32_t> uv_indices;
					uv_indices.resize(uv0.size());
					for (int i = 0; i < uv0.size(); i++) {
						uv_indices.write[i] = i;
					}
					ufbxw_int_buffer uv_indices_buffer = ufbxw_copy_int_array(write_scene, uv_indices.ptr(), uv_indices.size());
					ufbxw_mesh_set_uvs_indexed(write_scene, fbx_mesh, 0, uvs_buffer, uv_indices_buffer, UFBXW_ATTRIBUTE_MAPPING_VERTEX);
				}
			}

			// UV set 1: ARRAY_TEX_UV2
			Variant uv1_var = surface_arrays[Mesh::ARRAY_TEX_UV2];
			if (uv1_var.get_type() == Variant::PACKED_VECTOR2_ARRAY) {
				Vector<Vector2> uv1 = uv1_var;
				if (uv1.size() == vertices.size()) {
					Vector<ufbxw_vec2> fbx_uvs;
					fbx_uvs.resize(uv1.size());
					for (int i = 0; i < uv1.size(); i++) {
						fbx_uvs.write[i] = { (ufbxw_real)uv1[i].x, (ufbxw_real)uv1[i].y };
					}
					ufbxw_vec2_buffer uvs_buffer = ufbxw_copy_vec2_array(write_scene, fbx_uvs.ptr(), fbx_uvs.size());
					Vector<int32_t> uv_indices;
					uv_indices.resize(uv1.size());
					for (int i = 0; i < uv1.size(); i++) {
						uv_indices.write[i] = i;
					}
					ufbxw_int_buffer uv_indices_buffer = ufbxw_copy_int_array(write_scene, uv_indices.ptr(), uv_indices.size());
					ufbxw_mesh_set_uvs_indexed(write_scene, fbx_mesh, 1, uvs_buffer, uv_indices_buffer, UFBXW_ATTRIBUTE_MAPPING_VERTEX);
				}
			}

			// UV sets 2-7: Extract from ARRAY_CUSTOM0, ARRAY_CUSTOM1, ARRAY_CUSTOM2
			// Each custom array can store 2 UV sets (RGBA format) or 1 UV set (RG format)
			uint64_t surface_format = importer_mesh->get_surface_format(surface_i);
			int fbx_uv_set = 2;
			
			for (int custom_i = 0; custom_i < 3 && fbx_uv_set < 8; custom_i++) {
				Variant custom_var = surface_arrays[Mesh::ARRAY_CUSTOM0 + custom_i];
				if (custom_var.get_type() != Variant::PACKED_FLOAT32_ARRAY) {
					continue;
				}
				
				PackedFloat32Array custom_data = custom_var;
				if (custom_data.size() == 0) {
					continue;
				}
				
				// Determine format: RG (2 channels) or RGBA (4 channels)
				int custom_shift = Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT + custom_i * Mesh::ARRAY_FORMAT_CUSTOM_BITS;
				uint64_t custom_format = (surface_format >> custom_shift) & Mesh::ARRAY_FORMAT_CUSTOM_MASK;
				int num_channels = 0;
				
				if (custom_format == Mesh::ARRAY_CUSTOM_RG_FLOAT) {
					num_channels = 2;
				} else if (custom_format == Mesh::ARRAY_CUSTOM_RGBA_FLOAT) {
					num_channels = 4;
				} else {
					continue; // Not a float format we can use for UVs
				}
				
				int vertex_count = custom_data.size() / num_channels;
				if (vertex_count != vertices.size()) {
					continue; // Size mismatch
				}
				
				// Extract first UV set from RG or first half of RGBA
				Vector<ufbxw_vec2> fbx_uvs_first;
				fbx_uvs_first.resize(vertex_count);
				for (int i = 0; i < vertex_count; i++) {
					fbx_uvs_first.write[i] = { (ufbxw_real)custom_data[i * num_channels + 0], (ufbxw_real)custom_data[i * num_channels + 1] };
				}
				ufbxw_vec2_buffer uvs_buffer_first = ufbxw_copy_vec2_array(write_scene, fbx_uvs_first.ptr(), fbx_uvs_first.size());
				Vector<int32_t> uv_indices;
				uv_indices.resize(vertex_count);
				for (int i = 0; i < vertex_count; i++) {
					uv_indices.write[i] = i;
				}
				ufbxw_int_buffer uv_indices_buffer = ufbxw_copy_int_array(write_scene, uv_indices.ptr(), uv_indices.size());
				ufbxw_mesh_set_uvs_indexed(write_scene, fbx_mesh, fbx_uv_set, uvs_buffer_first, uv_indices_buffer, UFBXW_ATTRIBUTE_MAPPING_VERTEX);
				fbx_uv_set++;
				
				// Extract second UV set from RGBA (if available)
				if (num_channels == 4 && fbx_uv_set < 8) {
					Vector<ufbxw_vec2> fbx_uvs_second;
					fbx_uvs_second.resize(vertex_count);
					for (int i = 0; i < vertex_count; i++) {
						fbx_uvs_second.write[i] = { (ufbxw_real)custom_data[i * num_channels + 2], (ufbxw_real)custom_data[i * num_channels + 3] };
					}
					ufbxw_vec2_buffer uvs_buffer_second = ufbxw_copy_vec2_array(write_scene, fbx_uvs_second.ptr(), fbx_uvs_second.size());
					ufbxw_mesh_set_uvs_indexed(write_scene, fbx_mesh, fbx_uv_set, uvs_buffer_second, uv_indices_buffer, UFBXW_ATTRIBUTE_MAPPING_VERTEX);
					fbx_uv_set++;
				}
			}

			// Get and set vertex colors if available (always indexed for consistency)
			Variant color_var = surface_arrays[Mesh::ARRAY_COLOR];
			if (color_var.get_type() == Variant::PACKED_COLOR_ARRAY) {
				Vector<Color> colors = color_var;
				if (colors.size() == vertices.size()) {
					Vector<ufbxw_vec4> fbx_colors;
					fbx_colors.resize(colors.size());
					for (int i = 0; i < colors.size(); i++) {
						fbx_colors.write[i] = { (ufbxw_real)colors[i].r, (ufbxw_real)colors[i].g, (ufbxw_real)colors[i].b, (ufbxw_real)colors[i].a };
					}
					ufbxw_vec4_buffer colors_buffer = ufbxw_copy_vec4_array(write_scene, fbx_colors.ptr(), fbx_colors.size());
					// Create indices matching vertex order
					Vector<int32_t> color_indices;
					color_indices.resize(colors.size());
					for (int i = 0; i < colors.size(); i++) {
						color_indices.write[i] = i;
					}
					ufbxw_int_buffer color_indices_buffer = ufbxw_copy_int_array(write_scene, color_indices.ptr(), color_indices.size());
					ufbxw_mesh_set_colors_indexed(write_scene, fbx_mesh, 0, colors_buffer, color_indices_buffer, UFBXW_ATTRIBUTE_MAPPING_VERTEX);
				}
			}

			// Track material index for this surface
			// Instance materials are the active materials from MeshInstance3D (primary source, like GLTF export)
			// Surface materials from ImporterMesh are the mesh defaults (fallback)
			int32_t surface_material_index = -1;
			Ref<Material> surface_material;
			
			// First try instance material (active material from MeshInstance3D node)
			if (surface_i < instance_materials.size()) {
				surface_material = instance_materials[surface_i];
			}
			
			// If no instance material, fall back to surface material from mesh
			if (surface_material.is_null()) {
				surface_material = importer_mesh->get_surface_material(surface_i);
			}
			
			if (surface_material.is_valid()) {
				// Find the material index in state->materials
				for (int mat_i = 0; mat_i < state->materials.size(); mat_i++) {
					if (state->materials[mat_i] == surface_material) {
						surface_material_index = mat_i;
						break;
					}
				}
			}

			// Store per-face material indices (all faces in this surface use the same material)
			int face_count = indices.size() / 3;
			Vector<int32_t> face_material_indices;
			face_material_indices.resize(face_count);
			for (int tri = 0; tri < face_count; tri++) {
				face_material_indices.write[tri] = surface_material_index;
			}
			if (face_material_indices.size() > 0) {
				fbx_mesh_face_material_indices[fbx_mesh.id] = face_material_indices;
			}

			// Store this FBX mesh for the GLTF mesh
			if (!gltf_to_fbx_meshes.has(mesh_i)) {
				gltf_to_fbx_meshes[mesh_i] = Vector<ufbxw_mesh>();
			}
			gltf_to_fbx_meshes[mesh_i].push_back(fbx_mesh);
		}
	}

	// Attach meshes to nodes (one instance per FBX mesh)
	for (int i = 0; i < state->nodes.size(); i++) {
		Ref<GLTFNode> gltf_node = state->nodes[i];
		if (gltf_node.is_null()) {
			continue;
		}

		if (!gltf_to_fbx_nodes.has(i)) {
			continue;
		}

		GLTFMeshIndex mesh_idx = gltf_node->mesh;
		if (mesh_idx >= 0 && mesh_idx < state->meshes.size() && gltf_to_fbx_meshes.has(mesh_idx)) {
			ufbxw_node fbx_node = gltf_to_fbx_nodes[i];
			Vector<ufbxw_mesh> fbx_meshes = gltf_to_fbx_meshes[mesh_idx];
			// Attach all FBX meshes (one per surface) to the node
			for (int j = 0; j < fbx_meshes.size(); j++) {
				ufbxw_mesh_add_instance(write_scene, fbx_meshes[j], fbx_node);
			}
		}
	}

	// Export materials, cameras, and lights
	// Materials: Create FBX materials and assign to meshes
	HashMap<GLTFMaterialIndex, ufbxw_material> gltf_to_fbx_materials;
	for (int mat_i = 0; mat_i < state->materials.size(); mat_i++) {
		Ref<Material> material = state->materials[mat_i];
		if (material.is_null()) {
			continue;
		}

		// Create FBX material using create_element
		ufbxw_id material_id = ufbxw_create_element(write_scene, UFBXW_ELEMENT_MATERIAL);
		if (material_id == 0) {
			continue;
		}

		ufbxw_material fbx_material = { material_id };

		// Set material name
		String mat_name = material->get_name();
		if (mat_name.is_empty()) {
			mat_name = "Material" + itos(mat_i);
		}
		// Use utf8() to get CharString, then use length() to avoid strlen on boundary strings
		CharString mat_name_utf8 = mat_name.utf8();
		ufbxw_set_name_len(write_scene, material_id, mat_name_utf8.get_data(), mat_name_utf8.length());

		gltf_to_fbx_materials[mat_i] = fbx_material;

		// Set material properties from BaseMaterial3D
		Ref<BaseMaterial3D> base_material = material;
		if (base_material.is_valid()) {
			// Diffuse/Albedo color
			Color albedo = base_material->get_albedo().linear_to_srgb();
			ufbxw_vec3 diffuse_color = { (ufbxw_real)albedo.r, (ufbxw_real)albedo.g, (ufbxw_real)albedo.b };
			ufbxw_set_vec3(write_scene, material_id, "DiffuseColor", diffuse_color);
			ufbxw_set_real(write_scene, material_id, "DiffuseFactor", (ufbxw_real)albedo.a);

			// Metallic and roughness (PBR properties)
			ufbxw_set_real(write_scene, material_id, "ReflectionFactor", (ufbxw_real)base_material->get_metallic());
			ufbxw_set_real(write_scene, material_id, "Shininess", (ufbxw_real)(1.0 - base_material->get_roughness()) * 100.0);

			// Emission
			if (base_material->get_feature(BaseMaterial3D::FEATURE_EMISSION)) {
				Color emission = base_material->get_emission().linear_to_srgb();
				ufbxw_vec3 emissive_color = { (ufbxw_real)emission.r, (ufbxw_real)emission.g, (ufbxw_real)emission.b };
				ufbxw_set_vec3(write_scene, material_id, "EmissiveColor", emissive_color);
				ufbxw_set_real(write_scene, material_id, "EmissiveFactor", (ufbxw_real)base_material->get_emission_energy_multiplier());
			}
		}
	}

	// Assign materials to meshes (always per-face)
	for (int mesh_i = 0; mesh_i < state->meshes.size(); mesh_i++) {
		if (!gltf_to_fbx_meshes.has(mesh_i)) {
			continue;
		}

		Ref<GLTFMesh> gltf_mesh = state->meshes[mesh_i];
		if (gltf_mesh.is_null()) {
			continue;
		}

		Vector<ufbxw_mesh> fbx_meshes = gltf_to_fbx_meshes[mesh_i];
		// Assign materials to each FBX mesh (one per surface)
		for (int j = 0; j < fbx_meshes.size(); j++) {
			ufbxw_mesh fbx_mesh = fbx_meshes[j];
			if (fbx_mesh_face_material_indices.has(fbx_mesh.id)) {
				Vector<int32_t> face_material_indices = fbx_mesh_face_material_indices[fbx_mesh.id];
				if (face_material_indices.size() > 0) {
					ufbxw_int_buffer material_indices_buffer = ufbxw_copy_int_array(write_scene, face_material_indices.ptr(), face_material_indices.size());
					ufbxw_mesh_set_face_material(write_scene, fbx_mesh, material_indices_buffer);
				}
			}
		}
	}

	// Export cameras
	for (int cam_i = 0; cam_i < state->cameras.size(); cam_i++) {
		Ref<GLTFCamera> gltf_camera = state->cameras[cam_i];
		if (gltf_camera.is_null()) {
			continue;
		}

		// Find nodes that use this camera
		for (int node_i = 0; node_i < state->nodes.size(); node_i++) {
			Ref<GLTFNode> node = state->nodes[node_i];
			if (node.is_null() || node->camera != cam_i) {
				continue;
			}

			if (!gltf_to_fbx_nodes.has(node_i)) {
				continue;
			}

			ufbxw_node fbx_node = gltf_to_fbx_nodes[node_i];

			// Create FBX camera
			ufbxw_camera fbx_camera = ufbxw_create_camera(write_scene, fbx_node);
			if (fbx_camera.id == 0) {
				continue;
			}

			// Set camera name (GLTFCamera inherits from Resource which has get_name())
			String cam_name = gltf_camera->get_name();
			if (cam_name.is_empty()) {
				cam_name = "Camera" + itos(cam_i);
			}
			// Use utf8() to get CharString, then use length() to avoid strlen on boundary strings
			CharString cam_name_utf8 = cam_name.utf8();
			ufbxw_set_name_len(write_scene, fbx_camera.id, cam_name_utf8.get_data(), cam_name_utf8.length());

			// Set camera properties
			if (gltf_camera->get_perspective()) {
				// Perspective camera
				ufbxw_set_real(write_scene, fbx_camera.id, "FieldOfView", Math::rad_to_deg((ufbxw_real)gltf_camera->get_fov()));
			} else {
				// Orthographic camera
				ufbxw_set_real(write_scene, fbx_camera.id, "OrthoZoom", (ufbxw_real)gltf_camera->get_size_mag() * 2.0);
			}
			ufbxw_set_real(write_scene, fbx_camera.id, "NearPlane", (ufbxw_real)gltf_camera->get_depth_near());
			ufbxw_set_real(write_scene, fbx_camera.id, "FarPlane", (ufbxw_real)gltf_camera->get_depth_far());
			ufbxw_set_bool(write_scene, fbx_camera.id, "ProjectionType", gltf_camera->get_perspective());
		}
	}

	// Export lights
	for (int light_i = 0; light_i < state->lights.size(); light_i++) {
		Ref<GLTFLight> gltf_light = state->lights[light_i];
		if (gltf_light.is_null()) {
			continue;
		}

		// Find nodes that use this light
		for (int node_i = 0; node_i < state->nodes.size(); node_i++) {
			Ref<GLTFNode> node = state->nodes[node_i];
			if (node.is_null() || node->light != light_i) {
				continue;
			}

			if (!gltf_to_fbx_nodes.has(node_i)) {
				continue;
			}

			ufbxw_node fbx_node = gltf_to_fbx_nodes[node_i];

			// Create FBX light
			ufbxw_light fbx_light = ufbxw_create_light(write_scene, fbx_node);
			if (fbx_light.id == 0) {
				continue;
			}

			// Set light name (GLTFLight inherits from Resource which has get_name())
			String light_name = gltf_light->get_name();
			if (light_name.is_empty()) {
				light_name = "Light" + itos(light_i);
			}
			// Use utf8() to get CharString, then use length() to avoid strlen on boundary strings
			CharString light_name_utf8 = light_name.utf8();
			ufbxw_set_name_len(write_scene, fbx_light.id, light_name_utf8.get_data(), light_name_utf8.length());

			// Set light properties
			Color light_color = gltf_light->get_color();
			ufbxw_vec3 fbx_color = { (ufbxw_real)light_color.r, (ufbxw_real)light_color.g, (ufbxw_real)light_color.b };
			ufbxw_light_set_color(write_scene, fbx_light, fbx_color);

			float intensity = gltf_light->get_intensity();
			// FBX intensity is usually multiplied by 100.0
			ufbxw_light_set_intensity(write_scene, fbx_light, (ufbxw_real)(intensity * 100.0));

			// Set light type
			String light_type = gltf_light->get_light_type();
			ufbxw_light_type fbx_light_type = UFBXW_LIGHT_POINT;
			if (light_type == "directional") {
				fbx_light_type = UFBXW_LIGHT_DIRECTIONAL;
			} else if (light_type == "spot") {
				fbx_light_type = UFBXW_LIGHT_SPOT;
			} else if (light_type == "point") {
				fbx_light_type = UFBXW_LIGHT_POINT;
			}
			ufbxw_light_set_type(write_scene, fbx_light, fbx_light_type);

			// Set spotlight angles if applicable
			if (light_type == "spot") {
				float inner_angle = gltf_light->get_inner_cone_angle();
				float outer_angle = gltf_light->get_outer_cone_angle();
				ufbxw_light_set_inner_angle(write_scene, fbx_light, Math::rad_to_deg((ufbxw_real)inner_angle));
				ufbxw_light_set_outer_angle(write_scene, fbx_light, Math::rad_to_deg((ufbxw_real)outer_angle));
			}

			// Set decay type (default to quadratic for physically accurate)
			ufbxw_light_set_decay(write_scene, fbx_light, UFBXW_LIGHT_DECAY_QUADRATIC);
		}
	}

	// Fourth pass: Export skins (inverse of import skin tool)
	// This converts GLTFSkin objects back to FBX skin deformers and clusters
	HashMap<GLTFSkinIndex, ufbxw_skin_deformer> gltf_to_fbx_skins;

	for (int skin_i = 0; skin_i < state->skins.size(); skin_i++) {
		Ref<GLTFSkin> gltf_skin = state->skins[skin_i];
		if (gltf_skin.is_null()) {
			continue;
		}

		// Find the mesh associated with this skin
		// A skin is associated with a mesh through nodes that have both mesh >= 0 and skin >= 0
		GLTFMeshIndex associated_mesh_idx = -1;
		for (int node_i = 0; node_i < state->nodes.size(); node_i++) {
			Ref<GLTFNode> node = state->nodes[node_i];
			if (node.is_null()) {
				continue;
			}
			if (node->skin == skin_i && node->mesh >= 0) {
				associated_mesh_idx = node->mesh;
				break;
			}
		}

		// Skip if no associated mesh found (mesh export not yet implemented)
		if (associated_mesh_idx < 0 || !gltf_to_fbx_meshes.has(associated_mesh_idx)) {
			continue;
		}

		Vector<ufbxw_mesh> fbx_meshes = gltf_to_fbx_meshes[associated_mesh_idx];
		if (fbx_meshes.size() == 0) {
			continue;
		}

		// Find the mesh node's transform for the bind pose
		Transform3D mesh_bind_transform = Transform3D();
		for (int node_i = 0; node_i < state->nodes.size(); node_i++) {
			Ref<GLTFNode> node = state->nodes[node_i];
			if (node.is_null()) {
				continue;
			}
			if (node->skin == skin_i && node->mesh == associated_mesh_idx) {
				mesh_bind_transform = node->transform;
				break;
			}
		}

		// Create skin deformer for each mesh (one per surface)
		// Track all created deformers per mesh
		HashMap<ufbxw_id, ufbxw_skin_deformer> mesh_to_deformer;
		ufbxw_skin_deformer primary_skin_deformer = { 0 };
		
		Vector<GLTFNodeIndex> joints_original = gltf_skin->get_joints_original();
		Vector<GLTFNodeIndex> joints = gltf_skin->get_joints();
		TypedArray<Transform3D> inverse_binds = gltf_skin->get_inverse_binds();

		// Build mapping from expanded joints array index to joints_original index
		// Mesh bone indices are indices into the expanded joints array, we need to map them to joints_original
		HashMap<int, int> joints_idx_to_original_idx;
		HashMap<GLTFNodeIndex, int> node_to_original_idx;
		for (int orig_i = 0; orig_i < joints_original.size(); orig_i++) {
			node_to_original_idx[joints_original[orig_i]] = orig_i;
		}
		
		// Map each joint in expanded array to its joints_original index
		for (int joints_i = 0; joints_i < joints.size(); joints_i++) {
			GLTFNodeIndex joint_node = joints[joints_i];
			int original_idx = -1;
			
			// Check if node is directly in joints_original
			if (node_to_original_idx.has(joint_node)) {
				original_idx = node_to_original_idx[joint_node];
			} else {
				// Walk up the tree to find the closest ancestor in joints_original
				GLTFNodeIndex current_node = joint_node;
				while (current_node >= 0 && current_node < state->nodes.size()) {
					if (node_to_original_idx.has(current_node)) {
						original_idx = node_to_original_idx[current_node];
						break;
					}
					
					Ref<GLTFNode> node = state->nodes[current_node];
					if (node.is_null()) {
						break;
					}
					
					current_node = node->parent;
				}
			}
			
			if (original_idx >= 0) {
				joints_idx_to_original_idx[joints_i] = original_idx;
			}
		}

		for (int mesh_idx = 0; mesh_idx < fbx_meshes.size(); mesh_idx++) {
			ufbxw_mesh fbx_mesh = fbx_meshes[mesh_idx];
			ufbxw_skin_deformer fbx_skin_deformer = ufbxw_create_skin_deformer(write_scene, fbx_mesh);
			if (fbx_skin_deformer.id == 0) {
				continue; // Skip if creation failed
			}
			
			mesh_to_deformer[fbx_mesh.id] = fbx_skin_deformer;
			
			// Set skin deformer properties
			ufbxw_skin_deformer_set_skinning_type(write_scene, fbx_skin_deformer, UFBXW_SKINNING_TYPE_LINEAR);

			// Set mesh bind transform (mesh's transform at bind time)
			// Note: ufbxw_create_skin_deformer already creates the connection, so we don't need ufbxw_skin_deformer_add_mesh
			ufbxw_matrix mesh_bind_matrix = _transform_to_ufbxw_matrix(mesh_bind_transform);
			ufbxw_skin_deformer_set_mesh_bind_transform(write_scene, fbx_skin_deformer, mesh_bind_matrix);

			// Set skin name if available
			String skin_name = gltf_skin->get_name();
			if (!skin_name.is_empty()) {
				CharString skin_name_utf8 = skin_name.utf8();
				String mesh_suffix = fbx_meshes.size() > 1 ? "_Surface" + itos(mesh_idx) : "";
				CharString full_name_utf8 = (skin_name + mesh_suffix).utf8();
				ufbxw_set_name_len(write_scene, fbx_skin_deformer.id, full_name_utf8.get_data(), full_name_utf8.length());
			}

			// Store the first deformer as primary for reference
			if (mesh_idx == 0) {
				primary_skin_deformer = fbx_skin_deformer;
			}

			// Create skin clusters for each joint in joints_original
			for (int joint_i = 0; joint_i < joints_original.size(); joint_i++) {
				GLTFNodeIndex joint_node_idx = joints_original[joint_i];
				if (joint_node_idx < 0 || joint_node_idx >= state->nodes.size()) {
					continue;
				}

				if (!gltf_to_fbx_nodes.has(joint_node_idx)) {
					continue;
				}

				ufbxw_node fbx_bone_node = gltf_to_fbx_nodes[joint_node_idx];

				// Create skin cluster for this deformer
				ufbxw_skin_cluster fbx_cluster = ufbxw_create_skin_cluster(write_scene, fbx_skin_deformer, fbx_bone_node);
				if (fbx_cluster.id == 0) {
					continue; // Skip if creation failed
				}

				// Explicitly set the bone node for this cluster (ensures proper linking)
				ufbxw_skin_cluster_set_node(write_scene, fbx_cluster, fbx_bone_node);

				// Get inverse bind matrix from joints_original
				Transform3D inverse_bind = Transform3D();
				if (joint_i < inverse_binds.size()) {
					inverse_bind = inverse_binds[joint_i];
				}

				// Convert Transform3D to ufbxw_matrix
				ufbxw_matrix fbx_matrix = _transform_to_ufbxw_matrix(inverse_bind);
				ufbxw_skin_cluster_set_transform(write_scene, fbx_cluster, fbx_matrix);

				// Extract and set vertex weights from mesh data for this cluster
				// Use the surface corresponding to this mesh index
				if (associated_mesh_idx >= 0 && associated_mesh_idx < state->meshes.size()) {
					Ref<GLTFMesh> gltf_mesh = state->meshes[associated_mesh_idx];
					if (!gltf_mesh.is_null()) {
						Ref<ImporterMesh> importer_mesh = gltf_mesh->get_mesh();
						if (!importer_mesh.is_null() && mesh_idx < importer_mesh->get_surface_count()) {
							// Get bone and weight arrays from the surface corresponding to this mesh
							Array surface_arrays = importer_mesh->get_surface_arrays(mesh_idx);
							if (surface_arrays.size() > Mesh::ARRAY_BONES && surface_arrays.size() > Mesh::ARRAY_WEIGHTS) {
								Variant bones_variant = surface_arrays[Mesh::ARRAY_BONES];
								Variant weights_variant = surface_arrays[Mesh::ARRAY_WEIGHTS];

								if (bones_variant.get_type() == Variant::PACKED_INT32_ARRAY && weights_variant.get_type() == Variant::PACKED_FLOAT32_ARRAY) {
									PackedInt32Array bones = bones_variant;
									PackedFloat32Array weights = weights_variant;

									if (bones.size() == weights.size() && bones.size() > 0) {
										// Determine number of weights per vertex (4 or 8)
										int num_weights_per_vertex = 4;
										uint64_t format = importer_mesh->get_surface_format(mesh_idx);
										if (format & Mesh::ARRAY_FLAG_USE_8_BONE_WEIGHTS) {
											num_weights_per_vertex = 8;
										}

										int vertex_count = bones.size() / num_weights_per_vertex;

										// Collect vertex indices and weights for this cluster
										Vector<int32_t> vertex_indices;
										Vector<ufbxw_real> cluster_weights;
										
										// Debug: track unique bone indices we see
										HashSet<int> seen_bone_indices;
										int total_weights_checked = 0;
										int weights_matched = 0;

										for (int vertex_i = 0; vertex_i < vertex_count; vertex_i++) {
											for (int weight_i = 0; weight_i < num_weights_per_vertex; weight_i++) {
												int bone_idx = bones[vertex_i * num_weights_per_vertex + weight_i];
												float weight = weights[vertex_i * num_weights_per_vertex + weight_i];

												if (weight <= 0.0f) {
													continue;
												}
												
												total_weights_checked++;
												seen_bone_indices.insert(bone_idx);

												// Map bone index (from expanded joints array) to joints_original index
												// Mesh bone indices are indices into the expanded joints array
												// We need to map them to joints_original indices to match clusters
												int mapped_original_idx = -1;
												if (bone_idx >= 0 && bone_idx < joints.size()) {
													if (joints_idx_to_original_idx.has(bone_idx)) {
														mapped_original_idx = joints_idx_to_original_idx[bone_idx];
													}
												}
												
												// Check if this matches the current cluster (joint_i is the joints_original index)
												if (mapped_original_idx == joint_i) {
													vertex_indices.push_back(vertex_i);
													cluster_weights.push_back((ufbxw_real)weight);
													weights_matched++;
												}
											}
										}
										
										// Debug output for first few clusters or when no weights found
										if (vertex_indices.size() == 0 && (joint_i < 10 || joint_i % 50 == 0)) {
											String bone_indices_str = "";
											int count = 0;
											for (HashSet<int>::Iterator it = seen_bone_indices.begin(); it != seen_bone_indices.end() && count < 20; ++it, count++) {
												if (count > 0) bone_indices_str += ", ";
												bone_indices_str += itos(*it);
											}
											if (seen_bone_indices.size() > 20) bone_indices_str += ", ...";
											
											ERR_PRINT(vformat("FBX export: Cluster %d (joint_%d, node %d): checked %d weights, matched %d, seen bone indices: [%s], joints_original size: %d, joints size: %d", 
												joint_i, joint_i, joint_node_idx, total_weights_checked, weights_matched, bone_indices_str, joints_original.size(), joints.size()));
										}

										// Set weights if we have any
										// vertex_indices and cluster_weights should always be the same size since we push to both in the same iteration
										if (vertex_indices.size() > 0) {
											// Safety check: arrays should match since we push to both in the same iteration
											if (vertex_indices.size() != cluster_weights.size()) {
												ERR_PRINT("FBX export: Vertex indices and weights arrays size mismatch");
												continue;
											}
											ufbxw_int_buffer indices_buffer = ufbxw_copy_int_array(write_scene, vertex_indices.ptr(), vertex_indices.size());
											ufbxw_real_buffer weights_buffer = ufbxw_copy_real_array(write_scene, cluster_weights.ptr(), cluster_weights.size());
											ufbxw_skin_cluster_set_weights(write_scene, fbx_cluster, indices_buffer, weights_buffer);
											print_verbose(vformat("FBX export: Set %d weights for cluster %d (joint_%d, node %d)", vertex_indices.size(), joint_i, joint_i, joint_node_idx));
										} else {
											ERR_PRINT(vformat("FBX export: No weights found for cluster %d (joint_%d, node %d)", joint_i, joint_i, joint_node_idx));
										}
									}
								}
							}
						}
					}
				}
			}
		}
		// Store the primary skin deformer for reference
		if (primary_skin_deformer.id != 0) {
			gltf_to_fbx_skins[skin_i] = primary_skin_deformer;
		}
	}

	// Fifth pass: Export animations
	// Convert GLTF animations to FBX animation stacks and layers
	// FBX uses ktime which is in 1/46186158000 seconds (FBX time units)
	// GLTF animation times are in seconds, so we need to convert
	const double FBX_TIME_UNIT = 46186158000.0;

	// Helper function to map GLTF interpolation to FBX keyframe type
	auto map_interpolation_type = [](GLTFAnimation::Interpolation gltf_interp) -> uint32_t {
		switch (gltf_interp) {
			case GLTFAnimation::INTERP_LINEAR:
				return UFBXW_KEYFRAME_LINEAR;
			case GLTFAnimation::INTERP_STEP:
				return UFBXW_KEYFRAME_CONSTANT;
			case GLTFAnimation::INTERP_CUBIC_SPLINE:
				return UFBXW_KEYFRAME_CUBIC_AUTO;
			case GLTFAnimation::INTERP_CATMULLROMSPLINE:
				return UFBXW_KEYFRAME_CUBIC_AUTO;
			default:
				return UFBXW_KEYFRAME_LINEAR;
		}
	};

	for (int anim_i = 0; anim_i < state->animations.size(); anim_i++) {
		Ref<GLTFAnimation> gltf_anim = state->animations[anim_i];
		if (gltf_anim.is_null()) {
			continue;
		}

		// Create animation stack
		ufbxw_anim_stack fbx_anim_stack = ufbxw_create_anim_stack(write_scene);
		if (fbx_anim_stack.id == 0) {
			continue; // Skip if creation failed
		}

		// Set animation stack name
		String anim_name = gltf_anim->get_original_name();
		if (anim_name.is_empty()) {
			anim_name = gltf_anim->get_name();
		}
		if (anim_name.is_empty()) {
			anim_name = "Animation" + itos(anim_i);
		}
		CharString anim_name_utf8 = anim_name.utf8();
		ufbxw_set_name_len(write_scene, fbx_anim_stack.id, anim_name_utf8.get_data(), anim_name_utf8.length());

		// Get time range from additional data or calculate from keyframes
		Dictionary time_data = gltf_anim->get_additional_data("GODOT_animation_time_begin_time_end");
		double time_begin = 0.0;
		double time_end = 0.0;
		if (time_data.has("time_begin") && time_data.has("time_end")) {
			time_begin = time_data["time_begin"];
			time_end = time_data["time_end"];
		} else {
			// Calculate time range from all keyframes
			for (const KeyValue<int, GLTFAnimation::NodeTrack> &track_pair : gltf_anim->get_node_tracks()) {
				const GLTFAnimation::NodeTrack &track = track_pair.value;
				if (track.position_track.times.size() > 0) {
					double first_time = track.position_track.times[0];
					double last_time = track.position_track.times[track.position_track.times.size() - 1];
					if (time_begin == 0.0 || first_time < time_begin) {
						time_begin = first_time;
					}
					if (last_time > time_end) {
						time_end = last_time;
					}
				}
				if (track.rotation_track.times.size() > 0) {
					double first_time = track.rotation_track.times[0];
					double last_time = track.rotation_track.times[track.rotation_track.times.size() - 1];
					if (time_begin == 0.0 || first_time < time_begin) {
						time_begin = first_time;
					}
					if (last_time > time_end) {
						time_end = last_time;
					}
				}
				if (track.scale_track.times.size() > 0) {
					double first_time = track.scale_track.times[0];
					double last_time = track.scale_track.times[track.scale_track.times.size() - 1];
					if (time_begin == 0.0 || first_time < time_begin) {
						time_begin = first_time;
					}
					if (last_time > time_end) {
						time_end = last_time;
					}
				}
			}
		}

		// Convert time to FBX ktime (seconds * FBX_TIME_UNIT)
		ufbxw_ktime fbx_time_begin = (ufbxw_ktime)(time_begin * FBX_TIME_UNIT);
		ufbxw_ktime fbx_time_end = (ufbxw_ktime)(time_end * FBX_TIME_UNIT);
		ufbxw_anim_stack_set_time_range(write_scene, fbx_anim_stack, fbx_time_begin, fbx_time_end);
		ufbxw_anim_stack_set_reference_time_range(write_scene, fbx_anim_stack, fbx_time_begin, fbx_time_end);

		// Create animation layer
		ufbxw_anim_layer fbx_anim_layer = ufbxw_create_anim_layer(write_scene, fbx_anim_stack);
		if (fbx_anim_layer.id == 0) {
			continue; // Skip if creation failed
		}

		// Export node tracks
		for (const KeyValue<int, GLTFAnimation::NodeTrack> &track_pair : gltf_anim->get_node_tracks()) {
			int node_idx = track_pair.key;
			const GLTFAnimation::NodeTrack &track = track_pair.value;

			// Skip if node doesn't exist in FBX scene
			if (!gltf_to_fbx_nodes.has(node_idx)) {
				continue;
			}

			ufbxw_node fbx_node = gltf_to_fbx_nodes[node_idx];

			// Export position track
			if (track.position_track.times.size() > 0 && track.position_track.values.size() > 0) {
				ufbxw_anim_prop anim_prop = ufbxw_node_animate_translation(write_scene, fbx_node, fbx_anim_layer);
				if (anim_prop.id != 0) {
					for (int key_i = 0; key_i < track.position_track.times.size(); key_i++) {
						int value_idx = track.position_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + key_i * 3) : key_i;
						if (value_idx >= track.position_track.values.size()) {
							continue;
						}
						Vector3 pos = track.position_track.values[value_idx]; // Already scaled to centimeters
						ufbxw_ktime fbx_time = (ufbxw_ktime)(track.position_track.times[key_i] * FBX_TIME_UNIT);
						ufbxw_vec3 fbx_pos = { (ufbxw_real)pos.x, (ufbxw_real)pos.y, (ufbxw_real)pos.z };
						uint32_t interp_type = map_interpolation_type(track.position_track.interpolation);
						ufbxw_anim_add_keyframe_vec3(write_scene, anim_prop, fbx_time, fbx_pos, interp_type);
					}
					ufbxw_anim_finish_keyframes(write_scene, anim_prop);
				}
			}

			// Export rotation track (convert quaternions to Euler angles)
			if (track.rotation_track.times.size() > 0 && track.rotation_track.values.size() > 0) {
				ufbxw_anim_prop anim_prop = ufbxw_node_animate_rotation(write_scene, fbx_node, fbx_anim_layer);
				if (anim_prop.id != 0) {
					for (int key_i = 0; key_i < track.rotation_track.times.size(); key_i++) {
						int value_idx = track.rotation_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + key_i * 3) : key_i;
						if (value_idx >= track.rotation_track.values.size()) {
							continue;
						}
						Quaternion rot = track.rotation_track.values[value_idx];
						Vector3 euler = rot.get_euler();
						ufbxw_ktime fbx_time = (ufbxw_ktime)(track.rotation_track.times[key_i] * FBX_TIME_UNIT);
						ufbxw_vec3 fbx_rot = { Math::rad_to_deg((ufbxw_real)euler.x), Math::rad_to_deg((ufbxw_real)euler.y), Math::rad_to_deg((ufbxw_real)euler.z) };
						uint32_t interp_type = map_interpolation_type(track.rotation_track.interpolation);
						ufbxw_anim_add_keyframe_vec3(write_scene, anim_prop, fbx_time, fbx_rot, interp_type);
					}
					ufbxw_anim_finish_keyframes(write_scene, anim_prop);
				}
			}

			// Export scale track
			if (track.scale_track.times.size() > 0 && track.scale_track.values.size() > 0) {
				ufbxw_anim_prop anim_prop = ufbxw_node_animate_scaling(write_scene, fbx_node, fbx_anim_layer);
				if (anim_prop.id != 0) {
					for (int key_i = 0; key_i < track.scale_track.times.size(); key_i++) {
						int value_idx = track.scale_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + key_i * 3) : key_i;
						if (value_idx >= track.scale_track.values.size()) {
							continue;
						}
						Vector3 scale = track.scale_track.values[value_idx];
						ufbxw_ktime fbx_time = (ufbxw_ktime)(track.scale_track.times[key_i] * FBX_TIME_UNIT);
						ufbxw_vec3 fbx_scale = { (ufbxw_real)scale.x, (ufbxw_real)scale.y, (ufbxw_real)scale.z };
						uint32_t interp_type = map_interpolation_type(track.scale_track.interpolation);
						ufbxw_anim_add_keyframe_vec3(write_scene, anim_prop, fbx_time, fbx_scale, interp_type);
					}
					ufbxw_anim_finish_keyframes(write_scene, anim_prop);
				}
			}
		}

		// Set as active animation stack if it's the first one
		if (anim_i == 0) {
			ufbxw_set_active_anim_stack(write_scene, fbx_anim_stack);
		}
	}

	// Sixth pass: Export textures
	// Convert GLTF textures/images to FBX textures
	HashMap<GLTFTextureIndex, ufbxw_id> gltf_to_fbx_textures;

	for (int tex_i = 0; tex_i < state->textures.size(); tex_i++) {
		Ref<GLTFTexture> gltf_texture = state->textures[tex_i];
		if (gltf_texture.is_null()) {
			continue;
		}

		GLTFImageIndex image_idx = gltf_texture->get_src_image();
		if (image_idx < 0 || image_idx >= state->images.size()) {
			continue;
		}

		Ref<Texture2D> texture = state->images[image_idx];
		if (texture.is_null()) {
			continue;
		}

		// Create FBX texture element
		ufbxw_id texture_id = ufbxw_create_element(write_scene, UFBXW_ELEMENT_TEXTURE);
		if (texture_id == 0) {
			continue; // Skip if creation failed
		}

		// Set texture name
		String tex_name = texture->get_name();
		if (tex_name.is_empty()) {
			tex_name = "Texture" + itos(tex_i);
		}
		CharString tex_name_utf8 = tex_name.utf8();
		ufbxw_set_name_len(write_scene, texture_id, tex_name_utf8.get_data(), tex_name_utf8.length());

		// Get image data from texture
		Ref<Image> image;
		if (image_idx < state->source_images.size()) {
			image = state->source_images[image_idx];
		}
		if (image.is_null()) {
			// Try to get image from texture
			Ref<ImageTexture> img_texture = texture;
			if (img_texture.is_valid()) {
				image = img_texture->get_image();
			}
		}

		// Embed image data directly in FBX file instead of saving as external file
		if (image.is_valid()) {
			// Save image to PNG buffer for embedding
			PackedByteArray png_data = image->save_png_to_buffer();
			if (png_data.size() > 0) {
				// Store PNG data in a persistent buffer (needs to survive until save)
				// Store it in a static map keyed by texture_id to keep it alive
				static HashMap<ufbxw_id, Vector<uint8_t>> embedded_texture_data;
				Vector<uint8_t> &stored_data = embedded_texture_data[texture_id];
				stored_data.resize(png_data.size());
				memcpy(stored_data.ptrw(), png_data.ptr(), png_data.size());
				
				// Create blob from stored PNG data
				ufbxw_blob content_blob;
				content_blob.data = stored_data.ptr();
				content_blob.size = stored_data.size();
				
				// Set filename (required by FBX format, even for embedded content)
				String filename_only = tex_name + ".png";
				CharString filename_utf8 = filename_only.utf8();
				ufbxw_set_string(write_scene, texture_id, "FileName", filename_utf8.get_data());
				
				// Add Content property as blob to embed image data in FBX file
				ufbxw_add_blob(write_scene, texture_id, "Content", UFBXW_PROP_TYPE_BLOB, content_blob);
			}
		}

		gltf_to_fbx_textures[tex_i] = texture_id;
	}

	// Connect textures to materials
	for (int mat_i = 0; mat_i < state->materials.size(); mat_i++) {
		if (!gltf_to_fbx_materials.has(mat_i)) {
			continue;
		}

		Ref<Material> material = state->materials[mat_i];
		if (material.is_null()) {
			continue;
		}

		Ref<BaseMaterial3D> base_material = material;
		if (base_material.is_null()) {
			continue;
		}

		ufbxw_material fbx_material = gltf_to_fbx_materials[mat_i];

		// Connect albedo/diffuse texture
		Ref<Texture2D> albedo_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_ALBEDO);
		if (albedo_texture.is_valid()) {
			// Find texture by comparing RID (texture objects may be different instances)
			RID albedo_rid = albedo_texture->get_rid();
			for (int tex_i = 0; tex_i < state->textures.size(); tex_i++) {
				Ref<GLTFTexture> gltf_texture = state->textures[tex_i];
				if (gltf_texture.is_null()) {
					continue;
				}
				GLTFImageIndex image_idx = gltf_texture->get_src_image();
				if (image_idx >= 0 && image_idx < state->images.size()) {
					Ref<Texture2D> state_texture = state->images[image_idx];
					if (state_texture.is_valid() && state_texture->get_rid() == albedo_rid) {
						if (gltf_to_fbx_textures.has(tex_i)) {
							ufbxw_id texture_id = gltf_to_fbx_textures[tex_i];
							// Connect texture to material's DiffuseColor property
							ufbxw_connect_prop(write_scene, texture_id, "", fbx_material.id, "DiffuseColor");
						}
						break;
					}
				}
			}
		}

		// Connect normal texture
		Ref<Texture2D> normal_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_NORMAL);
		if (normal_texture.is_valid()) {
			// Find texture by comparing RID (texture objects may be different instances)
			RID normal_rid = normal_texture->get_rid();
			for (int tex_i = 0; tex_i < state->textures.size(); tex_i++) {
				Ref<GLTFTexture> gltf_texture = state->textures[tex_i];
				if (gltf_texture.is_null()) {
					continue;
				}
				GLTFImageIndex image_idx = gltf_texture->get_src_image();
				if (image_idx >= 0 && image_idx < state->images.size()) {
					Ref<Texture2D> state_texture = state->images[image_idx];
					if (state_texture.is_valid() && state_texture->get_rid() == normal_rid) {
						if (gltf_to_fbx_textures.has(tex_i)) {
							ufbxw_id texture_id = gltf_to_fbx_textures[tex_i];
							ufbxw_connect_prop(write_scene, texture_id, "", fbx_material.id, "NormalMap");
						}
						break;
					}
				}
			}
		}

		// Connect metallic texture
		Ref<Texture2D> metallic_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_METALLIC);
		if (metallic_texture.is_valid()) {
			RID metallic_rid = metallic_texture->get_rid();
			for (int tex_i = 0; tex_i < state->textures.size(); tex_i++) {
				Ref<GLTFTexture> gltf_texture = state->textures[tex_i];
				if (gltf_texture.is_null()) {
					continue;
				}
				GLTFImageIndex image_idx = gltf_texture->get_src_image();
				if (image_idx >= 0 && image_idx < state->images.size()) {
					Ref<Texture2D> state_texture = state->images[image_idx];
					if (state_texture.is_valid() && state_texture->get_rid() == metallic_rid) {
						if (gltf_to_fbx_textures.has(tex_i)) {
							ufbxw_id texture_id = gltf_to_fbx_textures[tex_i];
							ufbxw_connect_prop(write_scene, texture_id, "", fbx_material.id, "ReflectionFactor");
						}
						break;
					}
				}
			}
		}

		// Connect roughness texture
		Ref<Texture2D> roughness_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_ROUGHNESS);
		if (roughness_texture.is_valid()) {
			RID roughness_rid = roughness_texture->get_rid();
			for (int tex_i = 0; tex_i < state->textures.size(); tex_i++) {
				Ref<GLTFTexture> gltf_texture = state->textures[tex_i];
				if (gltf_texture.is_null()) {
					continue;
				}
				GLTFImageIndex image_idx = gltf_texture->get_src_image();
				if (image_idx >= 0 && image_idx < state->images.size()) {
					Ref<Texture2D> state_texture = state->images[image_idx];
					if (state_texture.is_valid() && state_texture->get_rid() == roughness_rid) {
						if (gltf_to_fbx_textures.has(tex_i)) {
							ufbxw_id texture_id = gltf_to_fbx_textures[tex_i];
							ufbxw_connect_prop(write_scene, texture_id, "", fbx_material.id, "Shininess");
						}
						break;
					}
				}
			}
		}

		// Connect ambient occlusion texture
		if (base_material->get_feature(BaseMaterial3D::FEATURE_AMBIENT_OCCLUSION)) {
			Ref<Texture2D> ao_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION);
			if (ao_texture.is_valid()) {
				RID ao_rid = ao_texture->get_rid();
				for (int tex_i = 0; tex_i < state->textures.size(); tex_i++) {
					Ref<GLTFTexture> gltf_texture = state->textures[tex_i];
					if (gltf_texture.is_null()) {
						continue;
					}
					GLTFImageIndex image_idx = gltf_texture->get_src_image();
					if (image_idx >= 0 && image_idx < state->images.size()) {
						Ref<Texture2D> state_texture = state->images[image_idx];
						if (state_texture.is_valid() && state_texture->get_rid() == ao_rid) {
							if (gltf_to_fbx_textures.has(tex_i)) {
								ufbxw_id texture_id = gltf_to_fbx_textures[tex_i];
								ufbxw_connect_prop(write_scene, texture_id, "", fbx_material.id, "AmbientFactor");
							}
							break;
						}
					}
				}
			}
		}

		// Connect emission texture
		if (base_material->get_feature(BaseMaterial3D::FEATURE_EMISSION)) {
			Ref<Texture2D> emission_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_EMISSION);
			if (emission_texture.is_valid()) {
				RID emission_rid = emission_texture->get_rid();
				for (int tex_i = 0; tex_i < state->textures.size(); tex_i++) {
					Ref<GLTFTexture> gltf_texture = state->textures[tex_i];
					if (gltf_texture.is_null()) {
						continue;
					}
					GLTFImageIndex image_idx = gltf_texture->get_src_image();
					if (image_idx >= 0 && image_idx < state->images.size()) {
						Ref<Texture2D> state_texture = state->images[image_idx];
						if (state_texture.is_valid() && state_texture->get_rid() == emission_rid) {
							if (gltf_to_fbx_textures.has(tex_i)) {
								ufbxw_id texture_id = gltf_to_fbx_textures[tex_i];
								ufbxw_connect_prop(write_scene, texture_id, "", fbx_material.id, "EmissiveColor");
							}
							break;
						}
					}
				}
			}
		}
	}

	// Prepare scene before saving (recommended by ufbx_write API)
	// This validates and prepares the scene structure for export
	ufbxw_prepare_scene(write_scene, nullptr); // Use default prepare options

	// Check if scene has any content
	bool has_content = false;
	if (gltf_to_fbx_nodes.size() > 0 || gltf_to_fbx_meshes.size() > 0) {
		has_content = true;
	}
	if (!has_content) {
		ERR_PRINT("FBX export: Scene has no content to export (no nodes or meshes)");
		ufbxw_free_scene(write_scene);
		return ERR_INVALID_DATA;
	}

	// Convert path to absolute path for ufbx_write
	String abs_path = ProjectSettings::get_singleton()->globalize_path(p_path);
	if (abs_path.is_empty()) {
		abs_path = p_path; // Fallback to original path if globalize fails
	}

	// Ensure the output directory exists
	String dir = abs_path.get_base_dir();
	if (!dir.is_empty()) {
		// Try filesystem access first (for absolute paths), then resources
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (!da.is_valid()) {
			da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		}
		if (da.is_valid()) {
			String current_dir = da->get_current_dir();
			if (!da->dir_exists(dir)) {
				Error mkdir_err = da->make_dir_recursive(dir);
				if (mkdir_err != OK) {
					ERR_PRINT("FBX export: Failed to create directory: " + dir);
					ERR_PRINT("FBX export: Current directory: " + current_dir);
					ufbxw_free_scene(write_scene);
					return ERR_FILE_CANT_WRITE;
				}
			}
		}
	}

	// Initialize save options with format (0 = Binary, 1 = ASCII)
	// Version defaults to 7500 if not set (handled by ufbx_write internally)
	ufbxw_save_opts save_opts = {};
	save_opts.format = (ufbxw_save_format)export_format; // 0 = UFBXW_SAVE_FORMAT_BINARY, 1 = UFBXW_SAVE_FORMAT_ASCII
	save_opts.version = 7500; // FBX 7.5 format (commonly used and well-supported)

	// Disable threading for FBX writes to avoid callback parameter corruption with ASAN
	// Multi-threaded writes cause issues when ASAN is enabled, possibly due to thread-local
	// storage or calling convention differences across thread boundaries
	// Leave thread_pool empty (all function pointers NULL) to force single-threaded operation

	// Use direct file API (ufbxw_save_file) instead of stream callbacks
	// This avoids callback parameter corruption issues entirely by writing directly to file
	// Convert path to UTF-8 C string for ufbx_write
	CharString path_utf8 = abs_path.utf8();
	const char *path_cstr = path_utf8.get_data();

	// Write FBX directly to file using ufbx_write's file API
	// This bypasses all callback mechanisms and avoids stack corruption
	ufbxw_error error = {};
	bool success = ufbxw_save_file(write_scene, path_cstr, &save_opts, &error);
	ufbxw_free_scene(write_scene);

	if (!success) {
		if (error.type != UFBXW_ERROR_NONE) {
			String error_desc = String::utf8(error.description, (int)error.description_length);
			ERR_PRINT("FBX write error: " + error_desc);
			ERR_PRINT("FBX write path: " + abs_path);
			ERR_PRINT("FBX write error type: " + itos((int)error.type));
			ERR_PRINT("FBX write error function: " + String::utf8(error.function.data, (int)error.function.length));
		} else {
			ERR_PRINT("FBX write failed with unknown error. Path: " + abs_path);
		}
		return ERR_FILE_CANT_WRITE;
	}

	return OK;
#else
	ERR_PRINT("FBX writing requires ufbx_write library. Please add ufbx_write files (ufbx_write.h and ufbx_write.c) to thirdparty/ufbx_write/");
	return ERR_UNAVAILABLE;
#endif
}

Error FBXDocument::append_from_scene(Node *p_node, Ref<GLTFState> p_state, uint32_t p_flags) {
	// Use parent class implementation to convert Node to GLTFState
	// This works because FBXDocument inherits from GLTFDocument
	return GLTFDocument::append_from_scene(p_node, p_state, p_flags);
}

void FBXDocument::set_naming_version(int p_version) {
	_naming_version = p_version;
}

int FBXDocument::get_naming_version() const {
	return _naming_version;
}

ufbxw_matrix FBXDocument::_transform_to_ufbxw_matrix(const Transform3D &p_transform) {
	ufbxw_matrix fbx_matrix = {};
	Basis basis = p_transform.basis;
	Vector3 origin = p_transform.origin;

	// FBX uses column-major matrix format
	fbx_matrix.m00 = (ufbxw_real)basis.rows[0].x;
	fbx_matrix.m10 = (ufbxw_real)basis.rows[0].y;
	fbx_matrix.m20 = (ufbxw_real)basis.rows[0].z;
	fbx_matrix.m30 = 0.0;

	fbx_matrix.m01 = (ufbxw_real)basis.rows[1].x;
	fbx_matrix.m11 = (ufbxw_real)basis.rows[1].y;
	fbx_matrix.m21 = (ufbxw_real)basis.rows[1].z;
	fbx_matrix.m31 = 0.0;

	fbx_matrix.m02 = (ufbxw_real)basis.rows[2].x;
	fbx_matrix.m12 = (ufbxw_real)basis.rows[2].y;
	fbx_matrix.m22 = (ufbxw_real)basis.rows[2].z;
	fbx_matrix.m32 = 0.0;

	fbx_matrix.m03 = (ufbxw_real)origin.x;
	fbx_matrix.m13 = (ufbxw_real)origin.y;
	fbx_matrix.m23 = (ufbxw_real)origin.z;
	fbx_matrix.m33 = 1.0;

	return fbx_matrix;
}

Vector3 FBXDocument::_as_vec3(const ufbx_vec3 &p_vector) {
	return Vector3(real_t(p_vector.x), real_t(p_vector.y), real_t(p_vector.z));
}

String FBXDocument::_as_string(const ufbx_string &p_string) {
	return String::utf8(p_string.data, (int)p_string.length);
}

Transform3D FBXDocument::_as_xform(const ufbx_matrix &p_mat) {
	Transform3D xform;
	xform.basis.set_column(Vector3::AXIS_X, _as_vec3(p_mat.cols[0]));
	xform.basis.set_column(Vector3::AXIS_Y, _as_vec3(p_mat.cols[1]));
	xform.basis.set_column(Vector3::AXIS_Z, _as_vec3(p_mat.cols[2]));
	xform.set_origin(_as_vec3(p_mat.cols[3]));
	return xform;
}


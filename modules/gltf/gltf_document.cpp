/*************************************************************************/
/*  gltf_document.cpp                                                    */
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

#include "gltf_document.h"

#include "gltf_accessor.h"
#include "gltf_animation.h"
#include "gltf_camera.h"
#include "gltf_document_extension.h"
#include "gltf_document_extension_convert_importer_mesh.h"
#include "gltf_light.h"
#include "gltf_mesh.h"
#include "gltf_node.h"
#include "gltf_skeleton.h"
#include "gltf_skin.h"
#include "gltf_spec_gloss.h"
#include "gltf_state.h"
#include "gltf_texture.h"

#include "core/crypto/crypto_core.h"
#include "core/error/error_macros.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/file_access_memory.h"
#include "core/io/json.h"
#include "core/math/disjoint_set.h"
#include "core/math/vector2.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"
#include "core/version.h"
#include "core/version_hash.gen.h"
#include "drivers/png/png_driver_common.h"
#include "editor/import/resource_importer_scene.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/multimesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/importer_mesh.h"
#include "scene/resources/mesh.h"
#include "scene/resources/multimesh.h"
#include "scene/resources/surface_tool.h"

#include "modules/modules_enabled.gen.h" // For csg, gridmap.

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif // MODULE_CSG_ENABLED
#ifdef MODULE_GRIDMAP_ENABLED
#include "modules/gridmap/grid_map.h"
#endif // MODULE_GRIDMAP_ENABLED

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <limits>

Error GLTFDocument::_serialize(Ref<GLTFState> state, const String &p_path) {
	if (!state->buffers.size()) {
		state->buffers.push_back(Vector<uint8_t>());
	}

	/* STEP 1 CONVERT MESH INSTANCES */
	_convert_mesh_instances(state);

	/* STEP 2 SERIALIZE CAMERAS */
	Error err = _serialize_cameras(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 3 CREATE SKINS */
	err = _serialize_skins(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 5 SERIALIZE MESHES (we have enough info now) */
	err = _serialize_meshes(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 6 SERIALIZE TEXTURES */
	err = _serialize_materials(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 7 SERIALIZE ANIMATIONS */
	err = _serialize_animations(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 8 SERIALIZE ACCESSORS */
	err = _encode_accessors(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 9 SERIALIZE IMAGES */
	err = _serialize_images(state, p_path);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 10 SERIALIZE TEXTURES */
	err = _serialize_textures(state);
	if (err != OK) {
		return Error::FAILED;
	}

	for (GLTFBufferViewIndex i = 0; i < state->buffer_views.size(); i++) {
		state->buffer_views.write[i]->buffer = 0;
	}

	/* STEP 11 SERIALIZE BUFFER VIEWS */
	err = _encode_buffer_views(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 12 SERIALIZE NODES */
	err = _serialize_nodes(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 13 SERIALIZE SCENE */
	err = _serialize_scenes(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 14 SERIALIZE SCENE */
	err = _serialize_lights(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 15 SERIALIZE EXTENSIONS */
	err = _serialize_extensions(state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 16 SERIALIZE VERSION */
	err = _serialize_version(state);
	if (err != OK) {
		return Error::FAILED;
	}

	return OK;
}

Error GLTFDocument::_serialize_extensions(Ref<GLTFState> state) const {
	const String texture_transform = "KHR_texture_transform";
	const String punctual_lights = "KHR_lights_punctual";
	Array extensions_used;
	extensions_used.push_back(punctual_lights);
	extensions_used.push_back(texture_transform);
	state->json["extensionsUsed"] = extensions_used;
	Array extensions_required;
	extensions_required.push_back(texture_transform);
	state->json["extensionsRequired"] = extensions_required;
	return OK;
}

Error GLTFDocument::_serialize_scenes(Ref<GLTFState> state) {
	Array scenes;
	const int loaded_scene = 0;
	state->json["scene"] = loaded_scene;

	if (state->nodes.size()) {
		Dictionary s;
		if (!state->scene_name.is_empty()) {
			s["name"] = state->scene_name;
		}

		Array nodes;
		nodes.push_back(0);
		s["nodes"] = nodes;
		scenes.push_back(s);
	}
	state->json["scenes"] = scenes;

	return OK;
}

Error GLTFDocument::_parse_json(const String &p_path, Ref<GLTFState> state) {
	Error err;
	FileAccessRef f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (!f) {
		return err;
	}

	Vector<uint8_t> array;
	array.resize(f->get_length());
	f->get_buffer(array.ptrw(), array.size());
	String text;
	text.parse_utf8((const char *)array.ptr(), array.size());

	JSON json;
	err = json.parse(text);
	if (err != OK) {
		_err_print_error("", p_path.utf8().get_data(), json.get_error_line(), json.get_error_message().utf8().get_data(), false, ERR_HANDLER_SCRIPT);
		return err;
	}
	state->json = json.get_data();

	return OK;
}

Error GLTFDocument::_parse_glb(FileAccess *f, Ref<GLTFState> state) {
	ERR_FAIL_NULL_V(f, ERR_INVALID_PARAMETER);
	ERR_FAIL_NULL_V(state, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(f->get_position() != 0, ERR_FILE_CANT_READ);
	uint32_t magic = f->get_32();
	ERR_FAIL_COND_V(magic != 0x46546C67, ERR_FILE_UNRECOGNIZED); //glTF
	f->get_32(); // version
	f->get_32(); // length
	uint32_t chunk_length = f->get_32();
	uint32_t chunk_type = f->get_32();

	ERR_FAIL_COND_V(chunk_type != 0x4E4F534A, ERR_PARSE_ERROR); //JSON
	Vector<uint8_t> json_data;
	json_data.resize(chunk_length);
	uint32_t len = f->get_buffer(json_data.ptrw(), chunk_length);
	ERR_FAIL_COND_V(len != chunk_length, ERR_FILE_CORRUPT);

	String text;
	text.parse_utf8((const char *)json_data.ptr(), json_data.size());

	JSON json;
	Error err = json.parse(text);
	if (err != OK) {
		_err_print_error("", "", json.get_error_line(), json.get_error_message().utf8().get_data(), false, ERR_HANDLER_SCRIPT);
		return err;
	}

	state->json = json.get_data();

	//data?

	chunk_length = f->get_32();
	chunk_type = f->get_32();

	if (f->eof_reached()) {
		return OK; //all good
	}

	ERR_FAIL_COND_V(chunk_type != 0x004E4942, ERR_PARSE_ERROR); //BIN

	state->glb_data.resize(chunk_length);
	len = f->get_buffer(state->glb_data.ptrw(), chunk_length);
	ERR_FAIL_COND_V(len != chunk_length, ERR_FILE_CORRUPT);

	return OK;
}

static Array _vec3_to_arr(const Vector3 &p_vec3) {
	Array array;
	array.resize(3);
	array[0] = p_vec3.x;
	array[1] = p_vec3.y;
	array[2] = p_vec3.z;
	return array;
}

static Vector3 _arr_to_vec3(const Array &p_array) {
	ERR_FAIL_COND_V(p_array.size() != 3, Vector3());
	return Vector3(p_array[0], p_array[1], p_array[2]);
}

static Array _quaternion_to_array(const Quaternion &p_quaternion) {
	Array array;
	array.resize(4);
	array[0] = p_quaternion.x;
	array[1] = p_quaternion.y;
	array[2] = p_quaternion.z;
	array[3] = p_quaternion.w;
	return array;
}

static Quaternion _arr_to_quaternion(const Array &p_array) {
	ERR_FAIL_COND_V(p_array.size() != 4, Quaternion());
	return Quaternion(p_array[0], p_array[1], p_array[2], p_array[3]);
}

static Transform3D _arr_to_xform(const Array &p_array) {
	ERR_FAIL_COND_V(p_array.size() != 16, Transform3D());

	Transform3D xform;
	xform.basis.set_axis(Vector3::AXIS_X, Vector3(p_array[0], p_array[1], p_array[2]));
	xform.basis.set_axis(Vector3::AXIS_Y, Vector3(p_array[4], p_array[5], p_array[6]));
	xform.basis.set_axis(Vector3::AXIS_Z, Vector3(p_array[8], p_array[9], p_array[10]));
	xform.set_origin(Vector3(p_array[12], p_array[13], p_array[14]));

	return xform;
}

static Vector<real_t> _xform_to_array(const Transform3D p_transform) {
	Vector<real_t> array;
	array.resize(16);
	Vector3 axis_x = p_transform.get_basis().get_axis(Vector3::AXIS_X);
	array.write[0] = axis_x.x;
	array.write[1] = axis_x.y;
	array.write[2] = axis_x.z;
	array.write[3] = 0.0f;
	Vector3 axis_y = p_transform.get_basis().get_axis(Vector3::AXIS_Y);
	array.write[4] = axis_y.x;
	array.write[5] = axis_y.y;
	array.write[6] = axis_y.z;
	array.write[7] = 0.0f;
	Vector3 axis_z = p_transform.get_basis().get_axis(Vector3::AXIS_Z);
	array.write[8] = axis_z.x;
	array.write[9] = axis_z.y;
	array.write[10] = axis_z.z;
	array.write[11] = 0.0f;
	Vector3 origin = p_transform.get_origin();
	array.write[12] = origin.x;
	array.write[13] = origin.y;
	array.write[14] = origin.z;
	array.write[15] = 1.0f;
	return array;
}

Error GLTFDocument::_serialize_nodes(Ref<GLTFState> state) {
	Array nodes;
	for (int i = 0; i < state->nodes.size(); i++) {
		Dictionary node;
		Ref<GLTFNode> n = state->nodes[i];
		Dictionary extensions;
		node["extensions"] = extensions;
		if (!n->get_name().is_empty()) {
			node["name"] = n->get_name();
		}
		if (n->camera != -1) {
			node["camera"] = n->camera;
		}
		if (n->light != -1) {
			Dictionary lights_punctual;
			extensions["KHR_lights_punctual"] = lights_punctual;
			lights_punctual["light"] = n->light;
		}
		if (n->mesh != -1) {
			node["mesh"] = n->mesh;
		}
		if (n->skin != -1) {
			node["skin"] = n->skin;
		}
		if (n->skeleton != -1 && n->skin < 0) {
		}
		if (n->xform != Transform3D()) {
			node["matrix"] = _xform_to_array(n->xform);
		}

		if (!n->rotation.is_equal_approx(Quaternion())) {
			node["rotation"] = _quaternion_to_array(n->rotation);
		}

		if (!n->scale.is_equal_approx(Vector3(1.0f, 1.0f, 1.0f))) {
			node["scale"] = _vec3_to_arr(n->scale);
		}

		if (!n->position.is_equal_approx(Vector3())) {
			node["translation"] = _vec3_to_arr(n->position);
		}
		if (n->children.size()) {
			Array children;
			for (int j = 0; j < n->children.size(); j++) {
				children.push_back(n->children[j]);
			}
			node["children"] = children;
		}
		nodes.push_back(node);
	}
	state->json["nodes"] = nodes;
	return OK;
}

String GLTFDocument::_gen_unique_name(Ref<GLTFState> state, const String &p_name) {
	const String s_name = p_name.validate_node_name();

	String name;
	int index = 1;
	while (true) {
		name = s_name;

		if (index > 1) {
			name += itos(index);
		}
		if (!state->unique_names.has(name)) {
			break;
		}
		index++;
	}

	state->unique_names.insert(name);

	return name;
}

String GLTFDocument::_sanitize_animation_name(const String &p_name) {
	// Animations disallow the normal node invalid characters as well as  "," and "["
	// (See animation/animation_player.cpp::add_animation)

	// TODO: Consider adding invalid_characters or a validate_animation_name to animation_player to mirror Node.
	String name = p_name.validate_node_name();
	name = name.replace(",", "");
	name = name.replace("[", "");
	return name;
}

String GLTFDocument::_gen_unique_animation_name(Ref<GLTFState> state, const String &p_name) {
	const String s_name = _sanitize_animation_name(p_name);

	String name;
	int index = 1;
	while (true) {
		name = s_name;

		if (index > 1) {
			name += itos(index);
		}
		if (!state->unique_animation_names.has(name)) {
			break;
		}
		index++;
	}

	state->unique_animation_names.insert(name);

	return name;
}

String GLTFDocument::_sanitize_bone_name(const String &p_name) {
	String name = p_name;
	name = name.replace(":", "_");
	name = name.replace("/", "_");
	return name;
}

String GLTFDocument::_gen_unique_bone_name(Ref<GLTFState> state, const GLTFSkeletonIndex skel_i, const String &p_name) {
	String s_name = _sanitize_bone_name(p_name);
	if (s_name.is_empty()) {
		s_name = "bone";
	}
	String name;
	int index = 1;
	while (true) {
		name = s_name;

		if (index > 1) {
			name += "_" + itos(index);
		}
		if (!state->skeletons[skel_i]->unique_names.has(name)) {
			break;
		}
		index++;
	}

	state->skeletons.write[skel_i]->unique_names.insert(name);

	return name;
}

Error GLTFDocument::_parse_scenes(Ref<GLTFState> state) {
	ERR_FAIL_COND_V(!state->json.has("scenes"), ERR_FILE_CORRUPT);
	const Array &scenes = state->json["scenes"];
	int loaded_scene = 0;
	if (state->json.has("scene")) {
		loaded_scene = state->json["scene"];
	} else {
		WARN_PRINT("The load-time scene is not defined in the glTF2 file. Picking the first scene.");
	}

	if (scenes.size()) {
		ERR_FAIL_COND_V(loaded_scene >= scenes.size(), ERR_FILE_CORRUPT);
		const Dictionary &s = scenes[loaded_scene];
		ERR_FAIL_COND_V(!s.has("nodes"), ERR_UNAVAILABLE);
		const Array &nodes = s["nodes"];
		for (int j = 0; j < nodes.size(); j++) {
			state->root_nodes.push_back(nodes[j]);
		}

		if (s.has("name") && !String(s["name"]).is_empty() && !((String)s["name"]).begins_with("Scene")) {
			state->scene_name = _gen_unique_name(state, s["name"]);
		} else {
			state->scene_name = _gen_unique_name(state, state->filename);
		}
	}

	return OK;
}

Error GLTFDocument::_parse_nodes(Ref<GLTFState> state) {
	ERR_FAIL_COND_V(!state->json.has("nodes"), ERR_FILE_CORRUPT);
	const Array &nodes = state->json["nodes"];
	for (int i = 0; i < nodes.size(); i++) {
		Ref<GLTFNode> node;
		node.instantiate();
		const Dictionary &n = nodes[i];

		if (n.has("name")) {
			node->set_name(n["name"]);
		}
		if (n.has("camera")) {
			node->camera = n["camera"];
		}
		if (n.has("mesh")) {
			node->mesh = n["mesh"];
		}
		if (n.has("skin")) {
			node->skin = n["skin"];
		}
		if (n.has("matrix")) {
			node->xform = _arr_to_xform(n["matrix"]);
		} else {
			if (n.has("translation")) {
				node->position = _arr_to_vec3(n["translation"]);
			}
			if (n.has("rotation")) {
				node->rotation = _arr_to_quaternion(n["rotation"]);
			}
			if (n.has("scale")) {
				node->scale = _arr_to_vec3(n["scale"]);
			}

			node->xform.basis.set_quaternion_scale(node->rotation, node->scale);
			node->xform.origin = node->position;
		}

		if (n.has("extensions")) {
			Dictionary extensions = n["extensions"];
			if (extensions.has("KHR_lights_punctual")) {
				Dictionary lights_punctual = extensions["KHR_lights_punctual"];
				if (lights_punctual.has("light")) {
					GLTFLightIndex light = lights_punctual["light"];
					node->light = light;
				}
			}
		}

		if (n.has("children")) {
			const Array &children = n["children"];
			for (int j = 0; j < children.size(); j++) {
				node->children.push_back(children[j]);
			}
		}

		state->nodes.push_back(node);
	}

	// build the hierarchy
	for (GLTFNodeIndex node_i = 0; node_i < state->nodes.size(); node_i++) {
		for (int j = 0; j < state->nodes[node_i]->children.size(); j++) {
			GLTFNodeIndex child_i = state->nodes[node_i]->children[j];

			ERR_FAIL_INDEX_V(child_i, state->nodes.size(), ERR_FILE_CORRUPT);
			ERR_CONTINUE(state->nodes[child_i]->parent != -1); //node already has a parent, wtf.

			state->nodes.write[child_i]->parent = node_i;
		}
	}

	_compute_node_heights(state);

	return OK;
}

void GLTFDocument::_compute_node_heights(Ref<GLTFState> state) {
	state->root_nodes.clear();
	for (GLTFNodeIndex node_i = 0; node_i < state->nodes.size(); ++node_i) {
		Ref<GLTFNode> node = state->nodes[node_i];
		node->height = 0;

		GLTFNodeIndex current_i = node_i;
		while (current_i >= 0) {
			const GLTFNodeIndex parent_i = state->nodes[current_i]->parent;
			if (parent_i >= 0) {
				++node->height;
			}
			current_i = parent_i;
		}

		if (node->height == 0) {
			state->root_nodes.push_back(node_i);
		}
	}
}

static Vector<uint8_t> _parse_base64_uri(const String &uri) {
	int start = uri.find(",");
	ERR_FAIL_COND_V(start == -1, Vector<uint8_t>());

	CharString substr = uri.substr(start + 1).ascii();

	int strlen = substr.length();

	Vector<uint8_t> buf;
	buf.resize(strlen / 4 * 3 + 1 + 1);

	size_t len = 0;
	ERR_FAIL_COND_V(CryptoCore::b64_decode(buf.ptrw(), buf.size(), &len, (unsigned char *)substr.get_data(), strlen) != OK, Vector<uint8_t>());

	buf.resize(len);

	return buf;
}
Error GLTFDocument::_encode_buffer_glb(Ref<GLTFState> state, const String &p_path) {
	print_verbose("glTF: Total buffers: " + itos(state->buffers.size()));

	if (!state->buffers.size()) {
		return OK;
	}
	Array buffers;
	if (state->buffers.size()) {
		Vector<uint8_t> buffer_data = state->buffers[0];
		Dictionary gltf_buffer;

		gltf_buffer["byteLength"] = buffer_data.size();
		buffers.push_back(gltf_buffer);
	}

	for (GLTFBufferIndex i = 1; i < state->buffers.size() - 1; i++) {
		Vector<uint8_t> buffer_data = state->buffers[i];
		Dictionary gltf_buffer;
		String filename = p_path.get_basename().get_file() + itos(i) + ".bin";
		String path = p_path.get_base_dir() + "/" + filename;
		Error err;
		FileAccessRef f = FileAccess::open(path, FileAccess::WRITE, &err);
		if (!f) {
			return err;
		}
		if (buffer_data.size() == 0) {
			return OK;
		}
		f->create(FileAccess::ACCESS_RESOURCES);
		f->store_buffer(buffer_data.ptr(), buffer_data.size());
		f->close();
		gltf_buffer["uri"] = filename;
		gltf_buffer["byteLength"] = buffer_data.size();
		buffers.push_back(gltf_buffer);
	}
	state->json["buffers"] = buffers;

	return OK;
}

Error GLTFDocument::_encode_buffer_bins(Ref<GLTFState> state, const String &p_path) {
	print_verbose("glTF: Total buffers: " + itos(state->buffers.size()));

	if (!state->buffers.size()) {
		return OK;
	}
	Array buffers;

	for (GLTFBufferIndex i = 0; i < state->buffers.size(); i++) {
		Vector<uint8_t> buffer_data = state->buffers[i];
		Dictionary gltf_buffer;
		String filename = p_path.get_basename().get_file() + itos(i) + ".bin";
		String path = p_path.get_base_dir() + "/" + filename;
		Error err;
		FileAccessRef f = FileAccess::open(path, FileAccess::WRITE, &err);
		if (!f) {
			return err;
		}
		if (buffer_data.size() == 0) {
			return OK;
		}
		f->create(FileAccess::ACCESS_RESOURCES);
		f->store_buffer(buffer_data.ptr(), buffer_data.size());
		f->close();
		gltf_buffer["uri"] = filename;
		gltf_buffer["byteLength"] = buffer_data.size();
		buffers.push_back(gltf_buffer);
	}
	state->json["buffers"] = buffers;

	return OK;
}

Error GLTFDocument::_parse_buffers(Ref<GLTFState> state, const String &p_base_path) {
	if (!state->json.has("buffers")) {
		return OK;
	}

	const Array &buffers = state->json["buffers"];
	for (GLTFBufferIndex i = 0; i < buffers.size(); i++) {
		if (i == 0 && state->glb_data.size()) {
			state->buffers.push_back(state->glb_data);

		} else {
			const Dictionary &buffer = buffers[i];
			if (buffer.has("uri")) {
				Vector<uint8_t> buffer_data;
				String uri = buffer["uri"];

				if (uri.begins_with("data:")) { // Embedded data using base64.
					// Validate data MIME types and throw an error if it's one we don't know/support.
					if (!uri.begins_with("data:application/octet-stream;base64") &&
							!uri.begins_with("data:application/gltf-buffer;base64")) {
						ERR_PRINT("glTF: Got buffer with an unknown URI data type: " + uri);
					}
					buffer_data = _parse_base64_uri(uri);
				} else { // Relative path to an external image file.
					ERR_FAIL_COND_V(p_base_path.is_empty(), ERR_INVALID_PARAMETER);
					uri = uri.uri_decode();
					uri = p_base_path.plus_file(uri).replace("\\", "/"); // Fix for Windows.
					buffer_data = FileAccess::get_file_as_array(uri);
					ERR_FAIL_COND_V_MSG(buffer.size() == 0, ERR_PARSE_ERROR, "glTF: Couldn't load binary file as an array: " + uri);
				}

				ERR_FAIL_COND_V(!buffer.has("byteLength"), ERR_PARSE_ERROR);
				int byteLength = buffer["byteLength"];
				ERR_FAIL_COND_V(byteLength < buffer_data.size(), ERR_PARSE_ERROR);
				state->buffers.push_back(buffer_data);
			}
		}
	}

	print_verbose("glTF: Total buffers: " + itos(state->buffers.size()));

	return OK;
}

Error GLTFDocument::_encode_buffer_views(Ref<GLTFState> state) {
	Array buffers;
	for (GLTFBufferViewIndex i = 0; i < state->buffer_views.size(); i++) {
		Dictionary d;

		Ref<GLTFBufferView> buffer_view = state->buffer_views[i];

		d["buffer"] = buffer_view->buffer;
		d["byteLength"] = buffer_view->byte_length;

		d["byteOffset"] = buffer_view->byte_offset;

		if (buffer_view->byte_stride != -1) {
			d["byteStride"] = buffer_view->byte_stride;
		}

		// TODO Sparse
		// d["target"] = buffer_view->indices;

		ERR_FAIL_COND_V(!d.has("buffer"), ERR_INVALID_DATA);
		ERR_FAIL_COND_V(!d.has("byteLength"), ERR_INVALID_DATA);
		buffers.push_back(d);
	}
	print_verbose("glTF: Total buffer views: " + itos(state->buffer_views.size()));
	if (!buffers.size()) {
		return OK;
	}
	state->json["bufferViews"] = buffers;
	return OK;
}

Error GLTFDocument::_parse_buffer_views(Ref<GLTFState> state) {
	if (!state->json.has("bufferViews")) {
		return OK;
	}
	const Array &buffers = state->json["bufferViews"];
	for (GLTFBufferViewIndex i = 0; i < buffers.size(); i++) {
		const Dictionary &d = buffers[i];

		Ref<GLTFBufferView> buffer_view;
		buffer_view.instantiate();

		ERR_FAIL_COND_V(!d.has("buffer"), ERR_PARSE_ERROR);
		buffer_view->buffer = d["buffer"];
		ERR_FAIL_COND_V(!d.has("byteLength"), ERR_PARSE_ERROR);
		buffer_view->byte_length = d["byteLength"];

		if (d.has("byteOffset")) {
			buffer_view->byte_offset = d["byteOffset"];
		}

		if (d.has("byteStride")) {
			buffer_view->byte_stride = d["byteStride"];
		}

		if (d.has("target")) {
			const int target = d["target"];
			buffer_view->indices = target == GLTFDocument::ELEMENT_ARRAY_BUFFER;
		}

		state->buffer_views.push_back(buffer_view);
	}

	print_verbose("glTF: Total buffer views: " + itos(state->buffer_views.size()));

	return OK;
}

Error GLTFDocument::_encode_accessors(Ref<GLTFState> state) {
	Array accessors;
	for (GLTFAccessorIndex i = 0; i < state->accessors.size(); i++) {
		Dictionary d;

		Ref<GLTFAccessor> accessor = state->accessors[i];
		d["componentType"] = accessor->component_type;
		d["count"] = accessor->count;
		d["type"] = _get_accessor_type_name(accessor->type);
		d["byteOffset"] = accessor->byte_offset;
		d["normalized"] = accessor->normalized;
		d["max"] = accessor->max;
		d["min"] = accessor->min;
		d["bufferView"] = accessor->buffer_view; //optional because it may be sparse...

		// Dictionary s;
		// s["count"] = accessor->sparse_count;
		// ERR_FAIL_COND_V(!s.has("count"), ERR_PARSE_ERROR);

		// s["indices"] = accessor->sparse_accessors;
		// ERR_FAIL_COND_V(!s.has("indices"), ERR_PARSE_ERROR);

		// Dictionary si;

		// si["bufferView"] = accessor->sparse_indices_buffer_view;

		// ERR_FAIL_COND_V(!si.has("bufferView"), ERR_PARSE_ERROR);
		// si["componentType"] = accessor->sparse_indices_component_type;

		// if (si.has("byteOffset")) {
		// 	si["byteOffset"] = accessor->sparse_indices_byte_offset;
		// }

		// ERR_FAIL_COND_V(!si.has("componentType"), ERR_PARSE_ERROR);
		// s["indices"] = si;
		// Dictionary sv;

		// sv["bufferView"] = accessor->sparse_values_buffer_view;
		// if (sv.has("byteOffset")) {
		// 	sv["byteOffset"] = accessor->sparse_values_byte_offset;
		// }
		// ERR_FAIL_COND_V(!sv.has("bufferView"), ERR_PARSE_ERROR);
		// s["values"] = sv;
		// ERR_FAIL_COND_V(!s.has("values"), ERR_PARSE_ERROR);
		// d["sparse"] = s;
		accessors.push_back(d);
	}

	if (!accessors.size()) {
		return OK;
	}
	state->json["accessors"] = accessors;
	ERR_FAIL_COND_V(!state->json.has("accessors"), ERR_FILE_CORRUPT);
	print_verbose("glTF: Total accessors: " + itos(state->accessors.size()));

	return OK;
}

String GLTFDocument::_get_accessor_type_name(const GLTFDocument::GLTFType p_type) {
	if (p_type == GLTFDocument::TYPE_SCALAR) {
		return "SCALAR";
	}
	if (p_type == GLTFDocument::TYPE_VEC2) {
		return "VEC2";
	}
	if (p_type == GLTFDocument::TYPE_VEC3) {
		return "VEC3";
	}
	if (p_type == GLTFDocument::TYPE_VEC4) {
		return "VEC4";
	}

	if (p_type == GLTFDocument::TYPE_MAT2) {
		return "MAT2";
	}
	if (p_type == GLTFDocument::TYPE_MAT3) {
		return "MAT3";
	}
	if (p_type == GLTFDocument::TYPE_MAT4) {
		return "MAT4";
	}
	ERR_FAIL_V("SCALAR");
}

GLTFDocument::GLTFType GLTFDocument::_get_type_from_str(const String &p_string) {
	if (p_string == "SCALAR") {
		return GLTFDocument::TYPE_SCALAR;
	}

	if (p_string == "VEC2") {
		return GLTFDocument::TYPE_VEC2;
	}
	if (p_string == "VEC3") {
		return GLTFDocument::TYPE_VEC3;
	}
	if (p_string == "VEC4") {
		return GLTFDocument::TYPE_VEC4;
	}

	if (p_string == "MAT2") {
		return GLTFDocument::TYPE_MAT2;
	}
	if (p_string == "MAT3") {
		return GLTFDocument::TYPE_MAT3;
	}
	if (p_string == "MAT4") {
		return GLTFDocument::TYPE_MAT4;
	}

	ERR_FAIL_V(GLTFDocument::TYPE_SCALAR);
}

Error GLTFDocument::_parse_accessors(Ref<GLTFState> state) {
	if (!state->json.has("accessors")) {
		return OK;
	}
	const Array &accessors = state->json["accessors"];
	for (GLTFAccessorIndex i = 0; i < accessors.size(); i++) {
		const Dictionary &d = accessors[i];

		Ref<GLTFAccessor> accessor;
		accessor.instantiate();

		ERR_FAIL_COND_V(!d.has("componentType"), ERR_PARSE_ERROR);
		accessor->component_type = d["componentType"];
		ERR_FAIL_COND_V(!d.has("count"), ERR_PARSE_ERROR);
		accessor->count = d["count"];
		ERR_FAIL_COND_V(!d.has("type"), ERR_PARSE_ERROR);
		accessor->type = _get_type_from_str(d["type"]);

		if (d.has("bufferView")) {
			accessor->buffer_view = d["bufferView"]; //optional because it may be sparse...
		}

		if (d.has("byteOffset")) {
			accessor->byte_offset = d["byteOffset"];
		}

		if (d.has("normalized")) {
			accessor->normalized = d["normalized"];
		}

		if (d.has("max")) {
			accessor->max = d["max"];
		}

		if (d.has("min")) {
			accessor->min = d["min"];
		}

		if (d.has("sparse")) {
			//eeh..

			const Dictionary &s = d["sparse"];

			ERR_FAIL_COND_V(!s.has("count"), ERR_PARSE_ERROR);
			accessor->sparse_count = s["count"];
			ERR_FAIL_COND_V(!s.has("indices"), ERR_PARSE_ERROR);
			const Dictionary &si = s["indices"];

			ERR_FAIL_COND_V(!si.has("bufferView"), ERR_PARSE_ERROR);
			accessor->sparse_indices_buffer_view = si["bufferView"];
			ERR_FAIL_COND_V(!si.has("componentType"), ERR_PARSE_ERROR);
			accessor->sparse_indices_component_type = si["componentType"];

			if (si.has("byteOffset")) {
				accessor->sparse_indices_byte_offset = si["byteOffset"];
			}

			ERR_FAIL_COND_V(!s.has("values"), ERR_PARSE_ERROR);
			const Dictionary &sv = s["values"];

			ERR_FAIL_COND_V(!sv.has("bufferView"), ERR_PARSE_ERROR);
			accessor->sparse_values_buffer_view = sv["bufferView"];
			if (sv.has("byteOffset")) {
				accessor->sparse_values_byte_offset = sv["byteOffset"];
			}
		}

		state->accessors.push_back(accessor);
	}

	print_verbose("glTF: Total accessors: " + itos(state->accessors.size()));

	return OK;
}

double GLTFDocument::_filter_number(double p_float) {
	if (Math::is_nan(p_float)) {
		return 0.0f;
	}
	return p_float;
}

String GLTFDocument::_get_component_type_name(const uint32_t p_component) {
	switch (p_component) {
		case GLTFDocument::COMPONENT_TYPE_BYTE:
			return "Byte";
		case GLTFDocument::COMPONENT_TYPE_UNSIGNED_BYTE:
			return "UByte";
		case GLTFDocument::COMPONENT_TYPE_SHORT:
			return "Short";
		case GLTFDocument::COMPONENT_TYPE_UNSIGNED_SHORT:
			return "UShort";
		case GLTFDocument::COMPONENT_TYPE_INT:
			return "Int";
		case GLTFDocument::COMPONENT_TYPE_FLOAT:
			return "Float";
	}

	return "<Error>";
}

String GLTFDocument::_get_type_name(const GLTFType p_component) {
	static const char *names[] = {
		"float",
		"vec2",
		"vec3",
		"vec4",
		"mat2",
		"mat3",
		"mat4"
	};

	return names[p_component];
}

Error GLTFDocument::_encode_buffer_view(Ref<GLTFState> state, const double *src, const int count, const GLTFType type, const int component_type, const bool normalized, const int byte_offset, const bool for_vertex, GLTFBufferViewIndex &r_accessor) {
	const int component_count_for_type[7] = {
		1, 2, 3, 4, 4, 9, 16
	};

	const int component_count = component_count_for_type[type];
	const int component_size = _get_component_type_size(component_type);
	ERR_FAIL_COND_V(component_size == 0, FAILED);

	int skip_every = 0;
	int skip_bytes = 0;
	//special case of alignments, as described in spec
	switch (component_type) {
		case COMPONENT_TYPE_BYTE:
		case COMPONENT_TYPE_UNSIGNED_BYTE: {
			if (type == TYPE_MAT2) {
				skip_every = 2;
				skip_bytes = 2;
			}
			if (type == TYPE_MAT3) {
				skip_every = 3;
				skip_bytes = 1;
			}
		} break;
		case COMPONENT_TYPE_SHORT:
		case COMPONENT_TYPE_UNSIGNED_SHORT: {
			if (type == TYPE_MAT3) {
				skip_every = 6;
				skip_bytes = 4;
			}
		} break;
		default: {
		}
	}

	Ref<GLTFBufferView> bv;
	bv.instantiate();
	const uint32_t offset = bv->byte_offset = byte_offset;
	Vector<uint8_t> &gltf_buffer = state->buffers.write[0];

	int stride = _get_component_type_size(component_type);
	if (for_vertex && stride % 4) {
		stride += 4 - (stride % 4); //according to spec must be multiple of 4
	}
	//use to debug
	print_verbose("glTF: encoding type " + _get_type_name(type) + " component type: " + _get_component_type_name(component_type) + " stride: " + itos(stride) + " amount " + itos(count));

	print_verbose("glTF: encoding accessor offset " + itos(byte_offset) + " view offset: " + itos(bv->byte_offset) + " total buffer len: " + itos(gltf_buffer.size()) + " view len " + itos(bv->byte_length));

	const int buffer_end = (stride * (count - 1)) + _get_component_type_size(component_type);
	// TODO define bv->byte_stride
	bv->byte_offset = gltf_buffer.size();

	switch (component_type) {
		case COMPONENT_TYPE_BYTE: {
			Vector<int8_t> buffer;
			buffer.resize(count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *src;
					if (normalized) {
						buffer.write[dst_i] = d * 128.0;
					} else {
						buffer.write[dst_i] = d;
					}
					src++;
					dst_i++;
				}
			}
			int64_t old_size = gltf_buffer.size();
			gltf_buffer.resize(old_size + (buffer.size() * sizeof(int8_t)));
			memcpy(gltf_buffer.ptrw() + old_size, buffer.ptrw(), buffer.size() * sizeof(int8_t));
			bv->byte_length = buffer.size() * sizeof(int8_t);
		} break;
		case COMPONENT_TYPE_UNSIGNED_BYTE: {
			Vector<uint8_t> buffer;
			buffer.resize(count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *src;
					if (normalized) {
						buffer.write[dst_i] = d * 255.0;
					} else {
						buffer.write[dst_i] = d;
					}
					src++;
					dst_i++;
				}
			}
			gltf_buffer.append_array(buffer);
			bv->byte_length = buffer.size() * sizeof(uint8_t);
		} break;
		case COMPONENT_TYPE_SHORT: {
			Vector<int16_t> buffer;
			buffer.resize(count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *src;
					if (normalized) {
						buffer.write[dst_i] = d * 32768.0;
					} else {
						buffer.write[dst_i] = d;
					}
					src++;
					dst_i++;
				}
			}
			int64_t old_size = gltf_buffer.size();
			gltf_buffer.resize(old_size + (buffer.size() * sizeof(int16_t)));
			memcpy(gltf_buffer.ptrw() + old_size, buffer.ptrw(), buffer.size() * sizeof(int16_t));
			bv->byte_length = buffer.size() * sizeof(int16_t);
		} break;
		case COMPONENT_TYPE_UNSIGNED_SHORT: {
			Vector<uint16_t> buffer;
			buffer.resize(count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *src;
					if (normalized) {
						buffer.write[dst_i] = d * 65535.0;
					} else {
						buffer.write[dst_i] = d;
					}
					src++;
					dst_i++;
				}
			}
			int64_t old_size = gltf_buffer.size();
			gltf_buffer.resize(old_size + (buffer.size() * sizeof(uint16_t)));
			memcpy(gltf_buffer.ptrw() + old_size, buffer.ptrw(), buffer.size() * sizeof(uint16_t));
			bv->byte_length = buffer.size() * sizeof(uint16_t);
		} break;
		case COMPONENT_TYPE_INT: {
			Vector<int> buffer;
			buffer.resize(count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *src;
					buffer.write[dst_i] = d;
					src++;
					dst_i++;
				}
			}
			int64_t old_size = gltf_buffer.size();
			gltf_buffer.resize(old_size + (buffer.size() * sizeof(int32_t)));
			memcpy(gltf_buffer.ptrw() + old_size, buffer.ptrw(), buffer.size() * sizeof(int32_t));
			bv->byte_length = buffer.size() * sizeof(int32_t);
		} break;
		case COMPONENT_TYPE_FLOAT: {
			Vector<float> buffer;
			buffer.resize(count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *src;
					buffer.write[dst_i] = d;
					src++;
					dst_i++;
				}
			}
			int64_t old_size = gltf_buffer.size();
			gltf_buffer.resize(old_size + (buffer.size() * sizeof(float)));
			memcpy(gltf_buffer.ptrw() + old_size, buffer.ptrw(), buffer.size() * sizeof(float));
			bv->byte_length = buffer.size() * sizeof(float);
		} break;
	}
	ERR_FAIL_COND_V(buffer_end > bv->byte_length, ERR_INVALID_DATA);

	ERR_FAIL_COND_V((int)(offset + buffer_end) > gltf_buffer.size(), ERR_INVALID_DATA);
	r_accessor = bv->buffer = state->buffer_views.size();
	state->buffer_views.push_back(bv);
	return OK;
}

Error GLTFDocument::_decode_buffer_view(Ref<GLTFState> state, double *dst, const GLTFBufferViewIndex p_buffer_view, const int skip_every, const int skip_bytes, const int element_size, const int count, const GLTFType type, const int component_count, const int component_type, const int component_size, const bool normalized, const int byte_offset, const bool for_vertex) {
	const Ref<GLTFBufferView> bv = state->buffer_views[p_buffer_view];

	int stride = element_size;
	if (bv->byte_stride != -1) {
		stride = bv->byte_stride;
	}
	if (for_vertex && stride % 4) {
		stride += 4 - (stride % 4); //according to spec must be multiple of 4
	}

	ERR_FAIL_INDEX_V(bv->buffer, state->buffers.size(), ERR_PARSE_ERROR);

	const uint32_t offset = bv->byte_offset + byte_offset;
	Vector<uint8_t> buffer = state->buffers[bv->buffer]; //copy on write, so no performance hit
	const uint8_t *bufptr = buffer.ptr();

	//use to debug
	print_verbose("glTF: type " + _get_type_name(type) + " component type: " + _get_component_type_name(component_type) + " stride: " + itos(stride) + " amount " + itos(count));
	print_verbose("glTF: accessor offset " + itos(byte_offset) + " view offset: " + itos(bv->byte_offset) + " total buffer len: " + itos(buffer.size()) + " view len " + itos(bv->byte_length));

	const int buffer_end = (stride * (count - 1)) + element_size;
	ERR_FAIL_COND_V(buffer_end > bv->byte_length, ERR_PARSE_ERROR);

	ERR_FAIL_COND_V((int)(offset + buffer_end) > buffer.size(), ERR_PARSE_ERROR);

	//fill everything as doubles

	for (int i = 0; i < count; i++) {
		const uint8_t *src = &bufptr[offset + i * stride];

		for (int j = 0; j < component_count; j++) {
			if (skip_every && j > 0 && (j % skip_every) == 0) {
				src += skip_bytes;
			}

			double d = 0;

			switch (component_type) {
				case COMPONENT_TYPE_BYTE: {
					int8_t b = int8_t(*src);
					if (normalized) {
						d = (double(b) / 128.0);
					} else {
						d = double(b);
					}
				} break;
				case COMPONENT_TYPE_UNSIGNED_BYTE: {
					uint8_t b = *src;
					if (normalized) {
						d = (double(b) / 255.0);
					} else {
						d = double(b);
					}
				} break;
				case COMPONENT_TYPE_SHORT: {
					int16_t s = *(int16_t *)src;
					if (normalized) {
						d = (double(s) / 32768.0);
					} else {
						d = double(s);
					}
				} break;
				case COMPONENT_TYPE_UNSIGNED_SHORT: {
					uint16_t s = *(uint16_t *)src;
					if (normalized) {
						d = (double(s) / 65535.0);
					} else {
						d = double(s);
					}
				} break;
				case COMPONENT_TYPE_INT: {
					d = *(int *)src;
				} break;
				case COMPONENT_TYPE_FLOAT: {
					d = *(float *)src;
				} break;
			}

			*dst++ = d;
			src += component_size;
		}
	}

	return OK;
}

int GLTFDocument::_get_component_type_size(const int component_type) {
	switch (component_type) {
		case COMPONENT_TYPE_BYTE:
		case COMPONENT_TYPE_UNSIGNED_BYTE:
			return 1;
			break;
		case COMPONENT_TYPE_SHORT:
		case COMPONENT_TYPE_UNSIGNED_SHORT:
			return 2;
			break;
		case COMPONENT_TYPE_INT:
		case COMPONENT_TYPE_FLOAT:
			return 4;
			break;
		default: {
			ERR_FAIL_V(0);
		}
	}
	return 0;
}

Vector<double> GLTFDocument::_decode_accessor(Ref<GLTFState> state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	//spec, for reference:
	//https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#data-alignment

	ERR_FAIL_INDEX_V(p_accessor, state->accessors.size(), Vector<double>());

	const Ref<GLTFAccessor> a = state->accessors[p_accessor];

	const int component_count_for_type[7] = {
		1, 2, 3, 4, 4, 9, 16
	};

	const int component_count = component_count_for_type[a->type];
	const int component_size = _get_component_type_size(a->component_type);
	ERR_FAIL_COND_V(component_size == 0, Vector<double>());
	int element_size = component_count * component_size;

	int skip_every = 0;
	int skip_bytes = 0;
	//special case of alignments, as described in spec
	switch (a->component_type) {
		case COMPONENT_TYPE_BYTE:
		case COMPONENT_TYPE_UNSIGNED_BYTE: {
			if (a->type == TYPE_MAT2) {
				skip_every = 2;
				skip_bytes = 2;
				element_size = 8; //override for this case
			}
			if (a->type == TYPE_MAT3) {
				skip_every = 3;
				skip_bytes = 1;
				element_size = 12; //override for this case
			}
		} break;
		case COMPONENT_TYPE_SHORT:
		case COMPONENT_TYPE_UNSIGNED_SHORT: {
			if (a->type == TYPE_MAT3) {
				skip_every = 6;
				skip_bytes = 4;
				element_size = 16; //override for this case
			}
		} break;
		default: {
		}
	}

	Vector<double> dst_buffer;
	dst_buffer.resize(component_count * a->count);
	double *dst = dst_buffer.ptrw();

	if (a->buffer_view >= 0) {
		ERR_FAIL_INDEX_V(a->buffer_view, state->buffer_views.size(), Vector<double>());

		const Error err = _decode_buffer_view(state, dst, a->buffer_view, skip_every, skip_bytes, element_size, a->count, a->type, component_count, a->component_type, component_size, a->normalized, a->byte_offset, p_for_vertex);
		if (err != OK) {
			return Vector<double>();
		}
	} else {
		//fill with zeros, as bufferview is not defined.
		for (int i = 0; i < (a->count * component_count); i++) {
			dst_buffer.write[i] = 0;
		}
	}

	if (a->sparse_count > 0) {
		// I could not find any file using this, so this code is so far untested
		Vector<double> indices;
		indices.resize(a->sparse_count);
		const int indices_component_size = _get_component_type_size(a->sparse_indices_component_type);

		Error err = _decode_buffer_view(state, indices.ptrw(), a->sparse_indices_buffer_view, 0, 0, indices_component_size, a->sparse_count, TYPE_SCALAR, 1, a->sparse_indices_component_type, indices_component_size, false, a->sparse_indices_byte_offset, false);
		if (err != OK) {
			return Vector<double>();
		}

		Vector<double> data;
		data.resize(component_count * a->sparse_count);
		err = _decode_buffer_view(state, data.ptrw(), a->sparse_values_buffer_view, skip_every, skip_bytes, element_size, a->sparse_count, a->type, component_count, a->component_type, component_size, a->normalized, a->sparse_values_byte_offset, p_for_vertex);
		if (err != OK) {
			return Vector<double>();
		}

		for (int i = 0; i < indices.size(); i++) {
			const int write_offset = int(indices[i]) * component_count;

			for (int j = 0; j < component_count; j++) {
				dst[write_offset + j] = data[i * component_count + j];
			}
		}
	}

	return dst_buffer;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_ints(Ref<GLTFState> state, const Vector<int32_t> p_attribs, const bool p_for_vertex) {
	if (p_attribs.size() == 0) {
		return -1;
	}
	const int element_count = 1;
	const int ret_size = p_attribs.size();
	Vector<double> attribs;
	attribs.resize(ret_size);
	Vector<double> type_max;
	type_max.resize(element_count);
	Vector<double> type_min;
	type_min.resize(element_count);
	for (int i = 0; i < p_attribs.size(); i++) {
		attribs.write[i] = Math::snapped(p_attribs[i], 1.0);
		if (i == 0) {
			for (int32_t type_i = 0; type_i < element_count; type_i++) {
				type_max.write[type_i] = attribs[(i * element_count) + type_i];
				type_min.write[type_i] = attribs[(i * element_count) + type_i];
			}
		}
		for (int32_t type_i = 0; type_i < element_count; type_i++) {
			type_max.write[type_i] = MAX(attribs[(i * element_count) + type_i], type_max[type_i]);
			type_min.write[type_i] = MIN(attribs[(i * element_count) + type_i], type_min[type_i]);
			type_max.write[type_i] = _filter_number(type_max.write[type_i]);
			type_min.write[type_i] = _filter_number(type_min.write[type_i]);
		}
	}

	ERR_FAIL_COND_V(attribs.size() == 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	GLTFBufferIndex buffer_view_i;
	int64_t size = state->buffers[0].size();
	const GLTFDocument::GLTFType type = GLTFDocument::TYPE_SCALAR;
	const int component_type = GLTFDocument::COMPONENT_TYPE_INT;

	accessor->max = type_max;
	accessor->min = type_min;
	accessor->normalized = false;
	accessor->count = ret_size;
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(state, attribs.ptr(), attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	state->accessors.push_back(accessor);
	return state->accessors.size() - 1;
}

Vector<int> GLTFDocument::_decode_accessor_as_ints(Ref<GLTFState> state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<int> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	const double *attribs_ptr = attribs.ptr();
	const int ret_size = attribs.size();
	ret.resize(ret_size);
	{
		for (int i = 0; i < ret_size; i++) {
			ret.write[i] = int(attribs_ptr[i]);
		}
	}
	return ret;
}

Vector<float> GLTFDocument::_decode_accessor_as_floats(Ref<GLTFState> state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<float> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	const double *attribs_ptr = attribs.ptr();
	const int ret_size = attribs.size();
	ret.resize(ret_size);
	{
		for (int i = 0; i < ret_size; i++) {
			ret.write[i] = float(attribs_ptr[i]);
		}
	}
	return ret;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_vec2(Ref<GLTFState> state, const Vector<Vector2> p_attribs, const bool p_for_vertex) {
	if (p_attribs.size() == 0) {
		return -1;
	}
	const int element_count = 2;

	const int ret_size = p_attribs.size() * element_count;
	Vector<double> attribs;
	attribs.resize(ret_size);
	Vector<double> type_max;
	type_max.resize(element_count);
	Vector<double> type_min;
	type_min.resize(element_count);

	for (int i = 0; i < p_attribs.size(); i++) {
		Vector2 attrib = p_attribs[i];
		attribs.write[(i * element_count) + 0] = Math::snapped(attrib.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::snapped(attrib.y, CMP_NORMALIZE_TOLERANCE);
		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}

	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	GLTFBufferIndex buffer_view_i;
	int64_t size = state->buffers[0].size();
	const GLTFDocument::GLTFType type = GLTFDocument::TYPE_VEC2;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	accessor->max = type_max;
	accessor->min = type_min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	state->accessors.push_back(accessor);
	return state->accessors.size() - 1;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_color(Ref<GLTFState> state, const Vector<Color> p_attribs, const bool p_for_vertex) {
	if (p_attribs.size() == 0) {
		return -1;
	}

	const int ret_size = p_attribs.size() * 4;
	Vector<double> attribs;
	attribs.resize(ret_size);

	const int element_count = 4;
	Vector<double> type_max;
	type_max.resize(element_count);
	Vector<double> type_min;
	type_min.resize(element_count);
	for (int i = 0; i < p_attribs.size(); i++) {
		Color attrib = p_attribs[i];
		attribs.write[(i * element_count) + 0] = Math::snapped(attrib.r, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::snapped(attrib.g, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 2] = Math::snapped(attrib.b, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 3] = Math::snapped(attrib.a, CMP_NORMALIZE_TOLERANCE);

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}

	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	GLTFBufferIndex buffer_view_i;
	int64_t size = state->buffers[0].size();
	const GLTFDocument::GLTFType type = GLTFDocument::TYPE_VEC4;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	accessor->max = type_max;
	accessor->min = type_min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	state->accessors.push_back(accessor);
	return state->accessors.size() - 1;
}

void GLTFDocument::_calc_accessor_min_max(int i, const int element_count, Vector<double> &type_max, Vector<double> attribs, Vector<double> &type_min) {
	if (i == 0) {
		for (int32_t type_i = 0; type_i < element_count; type_i++) {
			type_max.write[type_i] = attribs[(i * element_count) + type_i];
			type_min.write[type_i] = attribs[(i * element_count) + type_i];
		}
	}
	for (int32_t type_i = 0; type_i < element_count; type_i++) {
		type_max.write[type_i] = MAX(attribs[(i * element_count) + type_i], type_max[type_i]);
		type_min.write[type_i] = MIN(attribs[(i * element_count) + type_i], type_min[type_i]);
		type_max.write[type_i] = _filter_number(type_max.write[type_i]);
		type_min.write[type_i] = _filter_number(type_min.write[type_i]);
	}
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_weights(Ref<GLTFState> state, const Vector<Color> p_attribs, const bool p_for_vertex) {
	if (p_attribs.size() == 0) {
		return -1;
	}

	const int ret_size = p_attribs.size() * 4;
	Vector<double> attribs;
	attribs.resize(ret_size);

	const int element_count = 4;

	Vector<double> type_max;
	type_max.resize(element_count);
	Vector<double> type_min;
	type_min.resize(element_count);
	for (int i = 0; i < p_attribs.size(); i++) {
		Color attrib = p_attribs[i];
		attribs.write[(i * element_count) + 0] = Math::snapped(attrib.r, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::snapped(attrib.g, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 2] = Math::snapped(attrib.b, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 3] = Math::snapped(attrib.a, CMP_NORMALIZE_TOLERANCE);

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}

	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	GLTFBufferIndex buffer_view_i;
	int64_t size = state->buffers[0].size();
	const GLTFDocument::GLTFType type = GLTFDocument::TYPE_VEC4;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	accessor->max = type_max;
	accessor->min = type_min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	state->accessors.push_back(accessor);
	return state->accessors.size() - 1;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_joints(Ref<GLTFState> state, const Vector<Color> p_attribs, const bool p_for_vertex) {
	if (p_attribs.size() == 0) {
		return -1;
	}

	const int element_count = 4;
	const int ret_size = p_attribs.size() * element_count;
	Vector<double> attribs;
	attribs.resize(ret_size);

	Vector<double> type_max;
	type_max.resize(element_count);
	Vector<double> type_min;
	type_min.resize(element_count);
	for (int i = 0; i < p_attribs.size(); i++) {
		Color attrib = p_attribs[i];
		attribs.write[(i * element_count) + 0] = Math::snapped(attrib.r, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::snapped(attrib.g, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 2] = Math::snapped(attrib.b, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 3] = Math::snapped(attrib.a, CMP_NORMALIZE_TOLERANCE);
		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}
	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	GLTFBufferIndex buffer_view_i;
	int64_t size = state->buffers[0].size();
	const GLTFDocument::GLTFType type = GLTFDocument::TYPE_VEC4;
	const int component_type = GLTFDocument::COMPONENT_TYPE_UNSIGNED_SHORT;

	accessor->max = type_max;
	accessor->min = type_min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	state->accessors.push_back(accessor);
	return state->accessors.size() - 1;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_quaternions(Ref<GLTFState> state, const Vector<Quaternion> p_attribs, const bool p_for_vertex) {
	if (p_attribs.size() == 0) {
		return -1;
	}
	const int element_count = 4;

	const int ret_size = p_attribs.size() * element_count;
	Vector<double> attribs;
	attribs.resize(ret_size);

	Vector<double> type_max;
	type_max.resize(element_count);
	Vector<double> type_min;
	type_min.resize(element_count);
	for (int i = 0; i < p_attribs.size(); i++) {
		Quaternion quaternion = p_attribs[i];
		attribs.write[(i * element_count) + 0] = Math::snapped(quaternion.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::snapped(quaternion.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 2] = Math::snapped(quaternion.z, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 3] = Math::snapped(quaternion.w, CMP_NORMALIZE_TOLERANCE);

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}

	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	GLTFBufferIndex buffer_view_i;
	int64_t size = state->buffers[0].size();
	const GLTFDocument::GLTFType type = GLTFDocument::TYPE_VEC4;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	accessor->max = type_max;
	accessor->min = type_min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	state->accessors.push_back(accessor);
	return state->accessors.size() - 1;
}

Vector<Vector2> GLTFDocument::_decode_accessor_as_vec2(Ref<GLTFState> state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Vector2> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	ERR_FAIL_COND_V(attribs.size() % 2 != 0, ret);
	const double *attribs_ptr = attribs.ptr();
	const int ret_size = attribs.size() / 2;
	ret.resize(ret_size);
	{
		for (int i = 0; i < ret_size; i++) {
			ret.write[i] = Vector2(attribs_ptr[i * 2 + 0], attribs_ptr[i * 2 + 1]);
		}
	}
	return ret;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_floats(Ref<GLTFState> state, const Vector<real_t> p_attribs, const bool p_for_vertex) {
	if (p_attribs.size() == 0) {
		return -1;
	}
	const int element_count = 1;
	const int ret_size = p_attribs.size();
	Vector<double> attribs;
	attribs.resize(ret_size);

	Vector<double> type_max;
	type_max.resize(element_count);
	Vector<double> type_min;
	type_min.resize(element_count);

	for (int i = 0; i < p_attribs.size(); i++) {
		attribs.write[i] = Math::snapped(p_attribs[i], CMP_NORMALIZE_TOLERANCE);

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}

	ERR_FAIL_COND_V(!attribs.size(), -1);

	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	GLTFBufferIndex buffer_view_i;
	int64_t size = state->buffers[0].size();
	const GLTFDocument::GLTFType type = GLTFDocument::TYPE_SCALAR;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	accessor->max = type_max;
	accessor->min = type_min;
	accessor->normalized = false;
	accessor->count = ret_size;
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(state, attribs.ptr(), attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	state->accessors.push_back(accessor);
	return state->accessors.size() - 1;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_vec3(Ref<GLTFState> state, const Vector<Vector3> p_attribs, const bool p_for_vertex) {
	if (p_attribs.size() == 0) {
		return -1;
	}
	const int element_count = 3;
	const int ret_size = p_attribs.size() * element_count;
	Vector<double> attribs;
	attribs.resize(ret_size);

	Vector<double> type_max;
	type_max.resize(element_count);
	Vector<double> type_min;
	type_min.resize(element_count);
	for (int i = 0; i < p_attribs.size(); i++) {
		Vector3 attrib = p_attribs[i];
		attribs.write[(i * element_count) + 0] = Math::snapped(attrib.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::snapped(attrib.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 2] = Math::snapped(attrib.z, CMP_NORMALIZE_TOLERANCE);

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}
	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	GLTFBufferIndex buffer_view_i;
	int64_t size = state->buffers[0].size();
	const GLTFDocument::GLTFType type = GLTFDocument::TYPE_VEC3;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	accessor->max = type_max;
	accessor->min = type_min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	state->accessors.push_back(accessor);
	return state->accessors.size() - 1;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_xform(Ref<GLTFState> state, const Vector<Transform3D> p_attribs, const bool p_for_vertex) {
	if (p_attribs.size() == 0) {
		return -1;
	}
	const int element_count = 16;
	const int ret_size = p_attribs.size() * element_count;
	Vector<double> attribs;
	attribs.resize(ret_size);

	Vector<double> type_max;
	type_max.resize(element_count);
	Vector<double> type_min;
	type_min.resize(element_count);
	for (int i = 0; i < p_attribs.size(); i++) {
		Transform3D attrib = p_attribs[i];
		Basis basis = attrib.get_basis();
		Vector3 axis_0 = basis.get_axis(Vector3::AXIS_X);

		attribs.write[i * element_count + 0] = Math::snapped(axis_0.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 1] = Math::snapped(axis_0.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 2] = Math::snapped(axis_0.z, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 3] = 0.0;

		Vector3 axis_1 = basis.get_axis(Vector3::AXIS_Y);
		attribs.write[i * element_count + 4] = Math::snapped(axis_1.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 5] = Math::snapped(axis_1.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 6] = Math::snapped(axis_1.z, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 7] = 0.0;

		Vector3 axis_2 = basis.get_axis(Vector3::AXIS_Z);
		attribs.write[i * element_count + 8] = Math::snapped(axis_2.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 9] = Math::snapped(axis_2.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 10] = Math::snapped(axis_2.z, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 11] = 0.0;

		Vector3 origin = attrib.get_origin();
		attribs.write[i * element_count + 12] = Math::snapped(origin.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 13] = Math::snapped(origin.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 14] = Math::snapped(origin.z, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 15] = 1.0;

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}
	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	GLTFBufferIndex buffer_view_i;
	int64_t size = state->buffers[0].size();
	const GLTFDocument::GLTFType type = GLTFDocument::TYPE_MAT4;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	accessor->max = type_max;
	accessor->min = type_min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	state->accessors.push_back(accessor);
	return state->accessors.size() - 1;
}

Vector<Vector3> GLTFDocument::_decode_accessor_as_vec3(Ref<GLTFState> state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Vector3> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	ERR_FAIL_COND_V(attribs.size() % 3 != 0, ret);
	const double *attribs_ptr = attribs.ptr();
	const int ret_size = attribs.size() / 3;
	ret.resize(ret_size);
	{
		for (int i = 0; i < ret_size; i++) {
			ret.write[i] = Vector3(attribs_ptr[i * 3 + 0], attribs_ptr[i * 3 + 1], attribs_ptr[i * 3 + 2]);
		}
	}
	return ret;
}

Vector<Color> GLTFDocument::_decode_accessor_as_color(Ref<GLTFState> state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Color> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	const int type = state->accessors[p_accessor]->type;
	ERR_FAIL_COND_V(!(type == TYPE_VEC3 || type == TYPE_VEC4), ret);
	int vec_len = 3;
	if (type == TYPE_VEC4) {
		vec_len = 4;
	}

	ERR_FAIL_COND_V(attribs.size() % vec_len != 0, ret);
	const double *attribs_ptr = attribs.ptr();
	const int ret_size = attribs.size() / vec_len;
	ret.resize(ret_size);
	{
		for (int i = 0; i < ret_size; i++) {
			ret.write[i] = Color(attribs_ptr[i * vec_len + 0], attribs_ptr[i * vec_len + 1], attribs_ptr[i * vec_len + 2], vec_len == 4 ? attribs_ptr[i * 4 + 3] : 1.0);
		}
	}
	return ret;
}
Vector<Quaternion> GLTFDocument::_decode_accessor_as_quaternion(Ref<GLTFState> state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Quaternion> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	ERR_FAIL_COND_V(attribs.size() % 4 != 0, ret);
	const double *attribs_ptr = attribs.ptr();
	const int ret_size = attribs.size() / 4;
	ret.resize(ret_size);
	{
		for (int i = 0; i < ret_size; i++) {
			ret.write[i] = Quaternion(attribs_ptr[i * 4 + 0], attribs_ptr[i * 4 + 1], attribs_ptr[i * 4 + 2], attribs_ptr[i * 4 + 3]).normalized();
		}
	}
	return ret;
}
Vector<Transform2D> GLTFDocument::_decode_accessor_as_xform2d(Ref<GLTFState> state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Transform2D> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	ERR_FAIL_COND_V(attribs.size() % 4 != 0, ret);
	ret.resize(attribs.size() / 4);
	for (int i = 0; i < ret.size(); i++) {
		ret.write[i][0] = Vector2(attribs[i * 4 + 0], attribs[i * 4 + 1]);
		ret.write[i][1] = Vector2(attribs[i * 4 + 2], attribs[i * 4 + 3]);
	}
	return ret;
}

Vector<Basis> GLTFDocument::_decode_accessor_as_basis(Ref<GLTFState> state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Basis> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	ERR_FAIL_COND_V(attribs.size() % 9 != 0, ret);
	ret.resize(attribs.size() / 9);
	for (int i = 0; i < ret.size(); i++) {
		ret.write[i].set_axis(0, Vector3(attribs[i * 9 + 0], attribs[i * 9 + 1], attribs[i * 9 + 2]));
		ret.write[i].set_axis(1, Vector3(attribs[i * 9 + 3], attribs[i * 9 + 4], attribs[i * 9 + 5]));
		ret.write[i].set_axis(2, Vector3(attribs[i * 9 + 6], attribs[i * 9 + 7], attribs[i * 9 + 8]));
	}
	return ret;
}

Vector<Transform3D> GLTFDocument::_decode_accessor_as_xform(Ref<GLTFState> state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Transform3D> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	ERR_FAIL_COND_V(attribs.size() % 16 != 0, ret);
	ret.resize(attribs.size() / 16);
	for (int i = 0; i < ret.size(); i++) {
		ret.write[i].basis.set_axis(0, Vector3(attribs[i * 16 + 0], attribs[i * 16 + 1], attribs[i * 16 + 2]));
		ret.write[i].basis.set_axis(1, Vector3(attribs[i * 16 + 4], attribs[i * 16 + 5], attribs[i * 16 + 6]));
		ret.write[i].basis.set_axis(2, Vector3(attribs[i * 16 + 8], attribs[i * 16 + 9], attribs[i * 16 + 10]));
		ret.write[i].set_origin(Vector3(attribs[i * 16 + 12], attribs[i * 16 + 13], attribs[i * 16 + 14]));
	}
	return ret;
}

Error GLTFDocument::_serialize_meshes(Ref<GLTFState> state) {
	Array meshes;
	for (GLTFMeshIndex gltf_mesh_i = 0; gltf_mesh_i < state->meshes.size(); gltf_mesh_i++) {
		print_verbose("glTF: Serializing mesh: " + itos(gltf_mesh_i));
		Ref<ImporterMesh> import_mesh = state->meshes.write[gltf_mesh_i]->get_mesh();
		if (import_mesh.is_null()) {
			continue;
		}
		Array instance_materials = state->meshes.write[gltf_mesh_i]->get_instance_materials();
		Array primitives;
		Dictionary gltf_mesh;
		Array target_names;
		Array weights;
		for (int morph_i = 0; morph_i < import_mesh->get_blend_shape_count(); morph_i++) {
			target_names.push_back(import_mesh->get_blend_shape_name(morph_i));
		}
		for (int surface_i = 0; surface_i < import_mesh->get_surface_count(); surface_i++) {
			Array targets;
			Dictionary primitive;
			Mesh::PrimitiveType primitive_type = import_mesh->get_surface_primitive_type(surface_i);
			switch (primitive_type) {
				case Mesh::PRIMITIVE_POINTS: {
					primitive["mode"] = 0;
					break;
				}
				case Mesh::PRIMITIVE_LINES: {
					primitive["mode"] = 1;
					break;
				}
				// case Mesh::PRIMITIVE_LINE_LOOP: {
				// 	primitive["mode"] = 2;
				// 	break;
				// }
				case Mesh::PRIMITIVE_LINE_STRIP: {
					primitive["mode"] = 3;
					break;
				}
				case Mesh::PRIMITIVE_TRIANGLES: {
					primitive["mode"] = 4;
					break;
				}
				case Mesh::PRIMITIVE_TRIANGLE_STRIP: {
					primitive["mode"] = 5;
					break;
				}
				// case Mesh::PRIMITIVE_TRIANGLE_FAN: {
				// 	primitive["mode"] = 6;
				// 	break;
				// }
				default: {
					ERR_FAIL_V(FAILED);
				}
			}

			Array array = import_mesh->get_surface_arrays(surface_i);
			uint32_t format = import_mesh->get_surface_format(surface_i);
			int32_t vertex_num = 0;
			Dictionary attributes;
			{
				Vector<Vector3> a = array[Mesh::ARRAY_VERTEX];
				ERR_FAIL_COND_V(!a.size(), ERR_INVALID_DATA);
				attributes["POSITION"] = _encode_accessor_as_vec3(state, a, true);
				vertex_num = a.size();
			}
			{
				Vector<real_t> a = array[Mesh::ARRAY_TANGENT];
				if (a.size()) {
					const int ret_size = a.size() / 4;
					Vector<Color> attribs;
					attribs.resize(ret_size);
					for (int i = 0; i < ret_size; i++) {
						Color out;
						out.r = a[(i * 4) + 0];
						out.g = a[(i * 4) + 1];
						out.b = a[(i * 4) + 2];
						out.a = a[(i * 4) + 3];
						attribs.write[i] = out;
					}
					attributes["TANGENT"] = _encode_accessor_as_color(state, attribs, true);
				}
			}
			{
				Vector<Vector3> a = array[Mesh::ARRAY_NORMAL];
				if (a.size()) {
					const int ret_size = a.size();
					Vector<Vector3> attribs;
					attribs.resize(ret_size);
					for (int i = 0; i < ret_size; i++) {
						attribs.write[i] = Vector3(a[i]).normalized();
					}
					attributes["NORMAL"] = _encode_accessor_as_vec3(state, attribs, true);
				}
			}
			{
				Vector<Vector2> a = array[Mesh::ARRAY_TEX_UV];
				if (a.size()) {
					attributes["TEXCOORD_0"] = _encode_accessor_as_vec2(state, a, true);
				}
			}
			{
				Vector<Vector2> a = array[Mesh::ARRAY_TEX_UV2];
				if (a.size()) {
					attributes["TEXCOORD_1"] = _encode_accessor_as_vec2(state, a, true);
				}
			}
			for (int custom_i = 0; custom_i < 3; custom_i++) {
				Vector<float> a = array[Mesh::ARRAY_CUSTOM0 + custom_i];
				if (a.size()) {
					int num_channels = 4;
					int custom_shift = Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT + custom_i * Mesh::ARRAY_FORMAT_CUSTOM_BITS;
					switch ((format >> custom_shift) & Mesh::ARRAY_FORMAT_CUSTOM_MASK) {
						case Mesh::ARRAY_CUSTOM_R_FLOAT:
							num_channels = 1;
							break;
						case Mesh::ARRAY_CUSTOM_RG_FLOAT:
							num_channels = 2;
							break;
						case Mesh::ARRAY_CUSTOM_RGB_FLOAT:
							num_channels = 3;
							break;
						case Mesh::ARRAY_CUSTOM_RGBA_FLOAT:
							num_channels = 4;
							break;
					}
					int texcoord_i = 2 + 2 * custom_i;
					String gltf_texcoord_key;
					for (int prev_texcoord_i = 0; prev_texcoord_i < texcoord_i; prev_texcoord_i++) {
						gltf_texcoord_key = vformat("TEXCOORD_%d", prev_texcoord_i);
						if (!attributes.has(gltf_texcoord_key)) {
							Vector<Vector2> empty;
							empty.resize(vertex_num);
							attributes[gltf_texcoord_key] = _encode_accessor_as_vec2(state, empty, true);
						}
					}

					LocalVector<Vector2> first_channel;
					first_channel.resize(vertex_num);
					LocalVector<Vector2> second_channel;
					second_channel.resize(vertex_num);
					for (int32_t vert_i = 0; vert_i < vertex_num; vert_i++) {
						float u = a[vert_i * num_channels + 0];
						float v = (num_channels == 1 ? 0.0f : a[vert_i * num_channels + 1]);
						first_channel[vert_i] = Vector2(u, v);
						u = 0;
						v = 0;
						if (num_channels >= 3) {
							u = a[vert_i * num_channels + 2];
							v = (num_channels == 3 ? 0.0f : a[vert_i * num_channels + 3]);
							second_channel[vert_i] = Vector2(u, v);
						}
					}
					gltf_texcoord_key = vformat("TEXCOORD_%d", texcoord_i);
					attributes[gltf_texcoord_key] = _encode_accessor_as_vec2(state, first_channel, true);
					gltf_texcoord_key = vformat("TEXCOORD_%d", texcoord_i + 1);
					attributes[gltf_texcoord_key] = _encode_accessor_as_vec2(state, second_channel, true);
				}
			}
			{
				Vector<Color> a = array[Mesh::ARRAY_COLOR];
				if (a.size()) {
					attributes["COLOR_0"] = _encode_accessor_as_color(state, a, true);
				}
			}
			Map<int, int> joint_i_to_bone_i;
			for (GLTFNodeIndex node_i = 0; node_i < state->nodes.size(); node_i++) {
				GLTFSkinIndex skin_i = -1;
				if (state->nodes[node_i]->mesh == gltf_mesh_i) {
					skin_i = state->nodes[node_i]->skin;
				}
				if (skin_i != -1) {
					joint_i_to_bone_i = state->skins[skin_i]->joint_i_to_bone_i;
					break;
				}
			}
			{
				const Array &a = array[Mesh::ARRAY_BONES];
				const Vector<Vector3> &vertex_array = array[Mesh::ARRAY_VERTEX];
				if ((a.size() / JOINT_GROUP_SIZE) == vertex_array.size()) {
					const int ret_size = a.size() / JOINT_GROUP_SIZE;
					Vector<Color> attribs;
					attribs.resize(ret_size);
					{
						for (int array_i = 0; array_i < attribs.size(); array_i++) {
							int32_t joint_0 = a[(array_i * JOINT_GROUP_SIZE) + 0];
							int32_t joint_1 = a[(array_i * JOINT_GROUP_SIZE) + 1];
							int32_t joint_2 = a[(array_i * JOINT_GROUP_SIZE) + 2];
							int32_t joint_3 = a[(array_i * JOINT_GROUP_SIZE) + 3];
							attribs.write[array_i] = Color(joint_0, joint_1, joint_2, joint_3);
						}
					}
					attributes["JOINTS_0"] = _encode_accessor_as_joints(state, attribs, true);
				} else if ((a.size() / (JOINT_GROUP_SIZE * 2)) >= vertex_array.size()) {
					Vector<Color> joints_0;
					joints_0.resize(vertex_num);
					Vector<Color> joints_1;
					joints_1.resize(vertex_num);
					int32_t weights_8_count = JOINT_GROUP_SIZE * 2;
					for (int32_t vertex_i = 0; vertex_i < vertex_num; vertex_i++) {
						Color joint_0;
						joint_0.r = a[vertex_i * weights_8_count + 0];
						joint_0.g = a[vertex_i * weights_8_count + 1];
						joint_0.b = a[vertex_i * weights_8_count + 2];
						joint_0.a = a[vertex_i * weights_8_count + 3];
						joints_0.write[vertex_i] = joint_0;
						Color joint_1;
						joint_1.r = a[vertex_i * weights_8_count + 4];
						joint_1.g = a[vertex_i * weights_8_count + 5];
						joint_1.b = a[vertex_i * weights_8_count + 6];
						joint_1.a = a[vertex_i * weights_8_count + 7];
						joints_1.write[vertex_i] = joint_1;
					}
					attributes["JOINTS_0"] = _encode_accessor_as_joints(state, joints_0, true);
					attributes["JOINTS_1"] = _encode_accessor_as_joints(state, joints_1, true);
				}
			}
			{
				const Array &a = array[Mesh::ARRAY_WEIGHTS];
				const Vector<Vector3> &vertex_array = array[Mesh::ARRAY_VERTEX];
				if ((a.size() / JOINT_GROUP_SIZE) == vertex_array.size()) {
					int32_t vertex_count = vertex_array.size();
					Vector<Color> attribs;
					attribs.resize(vertex_count);
					for (int i = 0; i < vertex_count; i++) {
						attribs.write[i] = Color(a[(i * JOINT_GROUP_SIZE) + 0], a[(i * JOINT_GROUP_SIZE) + 1], a[(i * JOINT_GROUP_SIZE) + 2], a[(i * JOINT_GROUP_SIZE) + 3]);
					}
					attributes["WEIGHTS_0"] = _encode_accessor_as_weights(state, attribs, true);
				} else if ((a.size() / (JOINT_GROUP_SIZE * 2)) >= vertex_array.size()) {
					Vector<Color> weights_0;
					weights_0.resize(vertex_num);
					Vector<Color> weights_1;
					weights_1.resize(vertex_num);
					int32_t weights_8_count = JOINT_GROUP_SIZE * 2;
					for (int32_t vertex_i = 0; vertex_i < vertex_num; vertex_i++) {
						Color weight_0;
						weight_0.r = a[vertex_i * weights_8_count + 0];
						weight_0.g = a[vertex_i * weights_8_count + 1];
						weight_0.b = a[vertex_i * weights_8_count + 2];
						weight_0.a = a[vertex_i * weights_8_count + 3];
						weights_0.write[vertex_i] = weight_0;
						Color weight_1;
						weight_1.r = a[vertex_i * weights_8_count + 4];
						weight_1.g = a[vertex_i * weights_8_count + 5];
						weight_1.b = a[vertex_i * weights_8_count + 6];
						weight_1.a = a[vertex_i * weights_8_count + 7];
						weights_1.write[vertex_i] = weight_1;
					}
					attributes["WEIGHTS_0"] = _encode_accessor_as_weights(state, weights_0, true);
					attributes["WEIGHTS_1"] = _encode_accessor_as_weights(state, weights_1, true);
				}
			}
			{
				Vector<int32_t> mesh_indices = array[Mesh::ARRAY_INDEX];
				if (mesh_indices.size()) {
					if (primitive_type == Mesh::PRIMITIVE_TRIANGLES) {
						//swap around indices, convert ccw to cw for front face
						const int is = mesh_indices.size();
						for (int k = 0; k < is; k += 3) {
							SWAP(mesh_indices.write[k + 0], mesh_indices.write[k + 2]);
						}
					}
					primitive["indices"] = _encode_accessor_as_ints(state, mesh_indices, true);
				} else {
					if (primitive_type == Mesh::PRIMITIVE_TRIANGLES) {
						//generate indices because they need to be swapped for CW/CCW
						const Vector<Vector3> &vertices = array[Mesh::ARRAY_VERTEX];
						Ref<SurfaceTool> st;
						st.instantiate();
						st->create_from_triangle_arrays(array);
						st->index();
						Vector<int32_t> generated_indices = st->commit_to_arrays()[Mesh::ARRAY_INDEX];
						const int vs = vertices.size();
						generated_indices.resize(vs);
						{
							for (int k = 0; k < vs; k += 3) {
								generated_indices.write[k] = k;
								generated_indices.write[k + 1] = k + 2;
								generated_indices.write[k + 2] = k + 1;
							}
						}
						primitive["indices"] = _encode_accessor_as_ints(state, generated_indices, true);
					}
				}
			}

			primitive["attributes"] = attributes;

			//blend shapes
			print_verbose("glTF: Mesh has targets");
			if (import_mesh->get_blend_shape_count()) {
				ArrayMesh::BlendShapeMode shape_mode = import_mesh->get_blend_shape_mode();
				for (int morph_i = 0; morph_i < import_mesh->get_blend_shape_count(); morph_i++) {
					Array array_morph = import_mesh->get_surface_blend_shape_arrays(surface_i, morph_i);
					Dictionary t;
					Vector<Vector3> varr = array_morph[Mesh::ARRAY_VERTEX];
					Array mesh_arrays = import_mesh->get_surface_arrays(surface_i);
					if (varr.size()) {
						Vector<Vector3> src_varr = array[Mesh::ARRAY_VERTEX];
						if (shape_mode == ArrayMesh::BlendShapeMode::BLEND_SHAPE_MODE_NORMALIZED) {
							const int max_idx = src_varr.size();
							for (int blend_i = 0; blend_i < max_idx; blend_i++) {
								varr.write[blend_i] = Vector3(varr[blend_i]) - src_varr[blend_i];
							}
						}

						t["POSITION"] = _encode_accessor_as_vec3(state, varr, true);
					}

					Vector<Vector3> narr = array_morph[Mesh::ARRAY_NORMAL];
					if (narr.size()) {
						t["NORMAL"] = _encode_accessor_as_vec3(state, narr, true);
					}
					Vector<real_t> tarr = array_morph[Mesh::ARRAY_TANGENT];
					if (tarr.size()) {
						const int ret_size = tarr.size() / 4;
						Vector<Vector3> attribs;
						attribs.resize(ret_size);
						for (int i = 0; i < ret_size; i++) {
							Vector3 vec3;
							vec3.x = tarr[(i * 4) + 0];
							vec3.y = tarr[(i * 4) + 1];
							vec3.z = tarr[(i * 4) + 2];
						}
						t["TANGENT"] = _encode_accessor_as_vec3(state, attribs, true);
					}
					targets.push_back(t);
				}
			}

			Variant v;
			if (surface_i < instance_materials.size()) {
				v = instance_materials.get(surface_i);
			}
			Ref<BaseMaterial3D> mat = v;
			if (!mat.is_valid()) {
				mat = import_mesh->get_surface_material(surface_i);
			}
			if (mat.is_valid()) {
				Map<Ref<BaseMaterial3D>, GLTFMaterialIndex>::Element *material_cache_i = state->material_cache.find(mat);
				if (material_cache_i && material_cache_i->get() != -1) {
					primitive["material"] = material_cache_i->get();
				} else {
					GLTFMaterialIndex mat_i = state->materials.size();
					state->materials.push_back(mat);
					primitive["material"] = mat_i;
					state->material_cache.insert(mat, mat_i);
				}
			}

			if (targets.size()) {
				primitive["targets"] = targets;
			}

			primitives.push_back(primitive);
		}

		Dictionary e;
		e["targetNames"] = target_names;

		weights.resize(target_names.size());
		for (int name_i = 0; name_i < target_names.size(); name_i++) {
			real_t weight = 0.0;
			if (name_i < state->meshes.write[gltf_mesh_i]->get_blend_weights().size()) {
				weight = state->meshes.write[gltf_mesh_i]->get_blend_weights()[name_i];
			}
			weights[name_i] = weight;
		}
		if (weights.size()) {
			gltf_mesh["weights"] = weights;
		}

		ERR_FAIL_COND_V(target_names.size() != weights.size(), FAILED);

		gltf_mesh["extras"] = e;

		gltf_mesh["primitives"] = primitives;

		meshes.push_back(gltf_mesh);
	}

	if (!meshes.size()) {
		return OK;
	}
	state->json["meshes"] = meshes;
	print_verbose("glTF: Total meshes: " + itos(meshes.size()));

	return OK;
}

Error GLTFDocument::_parse_meshes(Ref<GLTFState> state) {
	if (!state->json.has("meshes")) {
		return OK;
	}

	Array meshes = state->json["meshes"];
	for (GLTFMeshIndex i = 0; i < meshes.size(); i++) {
		print_verbose("glTF: Parsing mesh: " + itos(i));
		Dictionary d = meshes[i];

		Ref<GLTFMesh> mesh;
		mesh.instantiate();
		bool has_vertex_color = false;

		ERR_FAIL_COND_V(!d.has("primitives"), ERR_PARSE_ERROR);

		Array primitives = d["primitives"];
		const Dictionary &extras = d.has("extras") ? (Dictionary)d["extras"] : Dictionary();
		Ref<ImporterMesh> import_mesh;
		import_mesh.instantiate();
		String mesh_name = "mesh";
		if (d.has("name") && !String(d["name"]).is_empty()) {
			mesh_name = d["name"];
		}
		import_mesh->set_name(_gen_unique_name(state, vformat("%s_%s", state->scene_name, mesh_name)));

		for (int j = 0; j < primitives.size(); j++) {
			uint32_t flags = 0;
			Dictionary p = primitives[j];

			Array array;
			array.resize(Mesh::ARRAY_MAX);

			ERR_FAIL_COND_V(!p.has("attributes"), ERR_PARSE_ERROR);

			Dictionary a = p["attributes"];

			Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;
			if (p.has("mode")) {
				const int mode = p["mode"];
				ERR_FAIL_INDEX_V(mode, 7, ERR_FILE_CORRUPT);
				// Convert mesh.primitive.mode to Godot Mesh enum. See:
				// https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html#_mesh_primitive_mode
				static const Mesh::PrimitiveType primitives2[7] = {
					Mesh::PRIMITIVE_POINTS, // 0 POINTS
					Mesh::PRIMITIVE_LINES, // 1 LINES
					Mesh::PRIMITIVE_LINES, // 2 LINE_LOOP; loop not supported, should be converted
					Mesh::PRIMITIVE_LINE_STRIP, // 3 LINE_STRIP
					Mesh::PRIMITIVE_TRIANGLES, // 4 TRIANGLES
					Mesh::PRIMITIVE_TRIANGLE_STRIP, // 5 TRIANGLE_STRIP
					Mesh::PRIMITIVE_TRIANGLES, // 6 TRIANGLE_FAN fan not supported, should be converted
#ifndef _MSC_VER
#warning line loop and triangle fan are not supported and need to be converted to lines and triangles
#endif

				};

				primitive = primitives2[mode];
			}

			ERR_FAIL_COND_V(!a.has("POSITION"), ERR_PARSE_ERROR);
			int32_t vertex_num = 0;
			if (a.has("POSITION")) {
				PackedVector3Array vertices = _decode_accessor_as_vec3(state, a["POSITION"], true);
				array[Mesh::ARRAY_VERTEX] = vertices;
				vertex_num = vertices.size();
			}
			if (a.has("NORMAL")) {
				array[Mesh::ARRAY_NORMAL] = _decode_accessor_as_vec3(state, a["NORMAL"], true);
			}
			if (a.has("TANGENT")) {
				array[Mesh::ARRAY_TANGENT] = _decode_accessor_as_floats(state, a["TANGENT"], true);
			}
			if (a.has("TEXCOORD_0")) {
				array[Mesh::ARRAY_TEX_UV] = _decode_accessor_as_vec2(state, a["TEXCOORD_0"], true);
			}
			if (a.has("TEXCOORD_1")) {
				array[Mesh::ARRAY_TEX_UV2] = _decode_accessor_as_vec2(state, a["TEXCOORD_1"], true);
			}
			for (int custom_i = 0; custom_i < 3; custom_i++) {
				Vector<float> cur_custom;
				Vector<Vector2> texcoord_first;
				Vector<Vector2> texcoord_second;

				int texcoord_i = 2 + 2 * custom_i;
				String gltf_texcoord_key = vformat("TEXCOORD_%d", texcoord_i);
				int num_channels = 0;
				if (a.has(gltf_texcoord_key)) {
					texcoord_first = _decode_accessor_as_vec2(state, a[gltf_texcoord_key], true);
					num_channels = 2;
				}
				gltf_texcoord_key = vformat("TEXCOORD_%d", texcoord_i + 1);
				if (a.has(gltf_texcoord_key)) {
					texcoord_second = _decode_accessor_as_vec2(state, a[gltf_texcoord_key], true);
					num_channels = 4;
				}
				if (!num_channels) {
					break;
				}
				if (num_channels == 2 || num_channels == 4) {
					cur_custom.resize(vertex_num * num_channels);
					for (int32_t uv_i = 0; uv_i < texcoord_first.size() && uv_i < vertex_num; uv_i++) {
						cur_custom.write[uv_i * num_channels + 0] = texcoord_first[uv_i].x;
						cur_custom.write[uv_i * num_channels + 1] = texcoord_first[uv_i].y;
					}
					// Vector.resize seems to not zero-initialize. Ensure all unused elements are 0:
					for (int32_t uv_i = texcoord_first.size(); uv_i < vertex_num; uv_i++) {
						cur_custom.write[uv_i * num_channels + 0] = 0;
						cur_custom.write[uv_i * num_channels + 1] = 0;
					}
				}
				if (num_channels == 4) {
					for (int32_t uv_i = 0; uv_i < texcoord_second.size() && uv_i < vertex_num; uv_i++) {
						// num_channels must be 4
						cur_custom.write[uv_i * num_channels + 2] = texcoord_second[uv_i].x;
						cur_custom.write[uv_i * num_channels + 3] = texcoord_second[uv_i].y;
					}
					// Vector.resize seems to not zero-initialize. Ensure all unused elements are 0:
					for (int32_t uv_i = texcoord_second.size(); uv_i < vertex_num; uv_i++) {
						cur_custom.write[uv_i * num_channels + 2] = 0;
						cur_custom.write[uv_i * num_channels + 3] = 0;
					}
				}
				if (cur_custom.size() > 0) {
					array[Mesh::ARRAY_CUSTOM0 + custom_i] = cur_custom;
					int custom_shift = Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT + custom_i * Mesh::ARRAY_FORMAT_CUSTOM_BITS;
					if (num_channels == 2) {
						flags |= Mesh::ARRAY_CUSTOM_RG_FLOAT << custom_shift;
					} else {
						flags |= Mesh::ARRAY_CUSTOM_RGBA_FLOAT << custom_shift;
					}
				}
			}
			if (a.has("COLOR_0")) {
				array[Mesh::ARRAY_COLOR] = _decode_accessor_as_color(state, a["COLOR_0"], true);
				has_vertex_color = true;
			}
			if (a.has("JOINTS_0") && !a.has("JOINTS_1")) {
				array[Mesh::ARRAY_BONES] = _decode_accessor_as_ints(state, a["JOINTS_0"], true);
			} else if (a.has("JOINTS_0") && a.has("JOINTS_1")) {
				PackedInt32Array joints_0 = _decode_accessor_as_ints(state, a["JOINTS_0"], true);
				PackedInt32Array joints_1 = _decode_accessor_as_ints(state, a["JOINTS_1"], true);
				ERR_FAIL_COND_V(joints_0.size() != joints_0.size(), ERR_INVALID_DATA);
				int32_t weight_8_count = JOINT_GROUP_SIZE * 2;
				Vector<int> joints;
				joints.resize(vertex_num * weight_8_count);
				for (int32_t vertex_i = 0; vertex_i < vertex_num; vertex_i++) {
					joints.write[vertex_i * weight_8_count + 0] = joints_0[vertex_i * JOINT_GROUP_SIZE + 0];
					joints.write[vertex_i * weight_8_count + 1] = joints_0[vertex_i * JOINT_GROUP_SIZE + 1];
					joints.write[vertex_i * weight_8_count + 2] = joints_0[vertex_i * JOINT_GROUP_SIZE + 2];
					joints.write[vertex_i * weight_8_count + 3] = joints_0[vertex_i * JOINT_GROUP_SIZE + 3];
					joints.write[vertex_i * weight_8_count + 4] = joints_1[vertex_i * JOINT_GROUP_SIZE + 0];
					joints.write[vertex_i * weight_8_count + 5] = joints_1[vertex_i * JOINT_GROUP_SIZE + 1];
					joints.write[vertex_i * weight_8_count + 6] = joints_1[vertex_i * JOINT_GROUP_SIZE + 2];
					joints.write[vertex_i * weight_8_count + 7] = joints_1[vertex_i * JOINT_GROUP_SIZE + 3];
				}
				array[Mesh::ARRAY_BONES] = joints;
			}
			if (a.has("WEIGHTS_0") && !a.has("WEIGHTS_1")) {
				Vector<float> weights = _decode_accessor_as_floats(state, a["WEIGHTS_0"], true);
				{ //gltf does not seem to normalize the weights for some reason..
					int wc = weights.size();
					float *w = weights.ptrw();

					for (int k = 0; k < wc; k += 4) {
						float total = 0.0;
						total += w[k + 0];
						total += w[k + 1];
						total += w[k + 2];
						total += w[k + 3];
						if (total > 0.0) {
							w[k + 0] /= total;
							w[k + 1] /= total;
							w[k + 2] /= total;
							w[k + 3] /= total;
						}
					}
				}
				array[Mesh::ARRAY_WEIGHTS] = weights;
			} else if (a.has("WEIGHTS_0") && a.has("WEIGHTS_1")) {
				Vector<float> weights_0 = _decode_accessor_as_floats(state, a["WEIGHTS_0"], true);
				Vector<float> weights_1 = _decode_accessor_as_floats(state, a["WEIGHTS_1"], true);
				Vector<float> weights;
				ERR_FAIL_COND_V(weights_0.size() != weights_1.size(), ERR_INVALID_DATA);
				int32_t weight_8_count = JOINT_GROUP_SIZE * 2;
				weights.resize(vertex_num * weight_8_count);
				for (int32_t vertex_i = 0; vertex_i < vertex_num; vertex_i++) {
					weights.write[vertex_i * weight_8_count + 0] = weights_0[vertex_i * JOINT_GROUP_SIZE + 0];
					weights.write[vertex_i * weight_8_count + 1] = weights_0[vertex_i * JOINT_GROUP_SIZE + 1];
					weights.write[vertex_i * weight_8_count + 2] = weights_0[vertex_i * JOINT_GROUP_SIZE + 2];
					weights.write[vertex_i * weight_8_count + 3] = weights_0[vertex_i * JOINT_GROUP_SIZE + 3];
					weights.write[vertex_i * weight_8_count + 4] = weights_1[vertex_i * JOINT_GROUP_SIZE + 0];
					weights.write[vertex_i * weight_8_count + 5] = weights_1[vertex_i * JOINT_GROUP_SIZE + 1];
					weights.write[vertex_i * weight_8_count + 6] = weights_1[vertex_i * JOINT_GROUP_SIZE + 2];
					weights.write[vertex_i * weight_8_count + 7] = weights_1[vertex_i * JOINT_GROUP_SIZE + 3];
				}
				{ //gltf does not seem to normalize the weights for some reason..
					int wc = weights.size();
					float *w = weights.ptrw();

					for (int k = 0; k < wc; k += weight_8_count) {
						float total = 0.0;
						total += w[k + 0];
						total += w[k + 1];
						total += w[k + 2];
						total += w[k + 3];
						total += w[k + 4];
						total += w[k + 5];
						total += w[k + 6];
						total += w[k + 7];
						if (total > 0.0) {
							w[k + 0] /= total;
							w[k + 1] /= total;
							w[k + 2] /= total;
							w[k + 3] /= total;
							w[k + 4] /= total;
							w[k + 5] /= total;
							w[k + 6] /= total;
							w[k + 7] /= total;
						}
					}
				}
				array[Mesh::ARRAY_WEIGHTS] = weights;
			}

			if (p.has("indices")) {
				Vector<int> indices = _decode_accessor_as_ints(state, p["indices"], false);

				if (primitive == Mesh::PRIMITIVE_TRIANGLES) {
					//swap around indices, convert ccw to cw for front face

					const int is = indices.size();
					int *w = indices.ptrw();
					for (int k = 0; k < is; k += 3) {
						SWAP(w[k + 1], w[k + 2]);
					}
				}
				array[Mesh::ARRAY_INDEX] = indices;

			} else if (primitive == Mesh::PRIMITIVE_TRIANGLES) {
				//generate indices because they need to be swapped for CW/CCW
				const Vector<Vector3> &vertices = array[Mesh::ARRAY_VERTEX];
				ERR_FAIL_COND_V(vertices.size() == 0, ERR_PARSE_ERROR);
				Vector<int> indices;
				const int vs = vertices.size();
				indices.resize(vs);
				{
					int *w = indices.ptrw();
					for (int k = 0; k < vs; k += 3) {
						w[k] = k;
						w[k + 1] = k + 2;
						w[k + 2] = k + 1;
					}
				}
				array[Mesh::ARRAY_INDEX] = indices;
			}

			bool generate_tangents = (primitive == Mesh::PRIMITIVE_TRIANGLES && !a.has("TANGENT") && a.has("TEXCOORD_0") && a.has("NORMAL"));

			Ref<SurfaceTool> mesh_surface_tool;
			mesh_surface_tool.instantiate();
			mesh_surface_tool->create_from_triangle_arrays(array);
			if (a.has("JOINTS_0") && a.has("JOINTS_1")) {
				mesh_surface_tool->set_skin_weight_count(SurfaceTool::SKIN_8_WEIGHTS);
			}
			mesh_surface_tool->index();
			if (generate_tangents) {
				//must generate mikktspace tangents.. ergh..
				mesh_surface_tool->generate_tangents();
			}
			array = mesh_surface_tool->commit_to_arrays();

			Array morphs;
			//blend shapes
			if (p.has("targets")) {
				print_verbose("glTF: Mesh has targets");
				const Array &targets = p["targets"];

				//ideally BLEND_SHAPE_MODE_RELATIVE since gltf2 stores in displacement
				//but it could require a larger refactor?
				import_mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);

				if (j == 0) {
					const Array &target_names = extras.has("targetNames") ? (Array)extras["targetNames"] : Array();
					for (int k = 0; k < targets.size(); k++) {
						const String name = k < target_names.size() ? (String)target_names[k] : String("morph_") + itos(k);
						import_mesh->add_blend_shape(name);
					}
				}

				for (int k = 0; k < targets.size(); k++) {
					const Dictionary &t = targets[k];

					Array array_copy;
					array_copy.resize(Mesh::ARRAY_MAX);

					for (int l = 0; l < Mesh::ARRAY_MAX; l++) {
						array_copy[l] = array[l];
					}

					if (t.has("POSITION")) {
						Vector<Vector3> varr = _decode_accessor_as_vec3(state, t["POSITION"], true);
						const Vector<Vector3> src_varr = array[Mesh::ARRAY_VERTEX];
						const int size = src_varr.size();
						ERR_FAIL_COND_V(size == 0, ERR_PARSE_ERROR);
						{
							const int max_idx = varr.size();
							varr.resize(size);

							Vector3 *w_varr = varr.ptrw();
							const Vector3 *r_varr = varr.ptr();
							const Vector3 *r_src_varr = src_varr.ptr();
							for (int l = 0; l < size; l++) {
								if (l < max_idx) {
									w_varr[l] = r_varr[l] + r_src_varr[l];
								} else {
									w_varr[l] = r_src_varr[l];
								}
							}
						}
						array_copy[Mesh::ARRAY_VERTEX] = varr;
					}
					if (t.has("NORMAL")) {
						Vector<Vector3> narr = _decode_accessor_as_vec3(state, t["NORMAL"], true);
						const Vector<Vector3> src_narr = array[Mesh::ARRAY_NORMAL];
						int size = src_narr.size();
						ERR_FAIL_COND_V(size == 0, ERR_PARSE_ERROR);
						{
							int max_idx = narr.size();
							narr.resize(size);

							Vector3 *w_narr = narr.ptrw();
							const Vector3 *r_narr = narr.ptr();
							const Vector3 *r_src_narr = src_narr.ptr();
							for (int l = 0; l < size; l++) {
								if (l < max_idx) {
									w_narr[l] = r_narr[l] + r_src_narr[l];
								} else {
									w_narr[l] = r_src_narr[l];
								}
							}
						}
						array_copy[Mesh::ARRAY_NORMAL] = narr;
					}
					if (t.has("TANGENT")) {
						const Vector<Vector3> tangents_v3 = _decode_accessor_as_vec3(state, t["TANGENT"], true);
						const Vector<float> src_tangents = array[Mesh::ARRAY_TANGENT];
						ERR_FAIL_COND_V(src_tangents.size() == 0, ERR_PARSE_ERROR);

						Vector<float> tangents_v4;

						{
							int max_idx = tangents_v3.size();

							int size4 = src_tangents.size();
							tangents_v4.resize(size4);
							float *w4 = tangents_v4.ptrw();

							const Vector3 *r3 = tangents_v3.ptr();
							const float *r4 = src_tangents.ptr();

							for (int l = 0; l < size4 / 4; l++) {
								if (l < max_idx) {
									w4[l * 4 + 0] = r3[l].x + r4[l * 4 + 0];
									w4[l * 4 + 1] = r3[l].y + r4[l * 4 + 1];
									w4[l * 4 + 2] = r3[l].z + r4[l * 4 + 2];
								} else {
									w4[l * 4 + 0] = r4[l * 4 + 0];
									w4[l * 4 + 1] = r4[l * 4 + 1];
									w4[l * 4 + 2] = r4[l * 4 + 2];
								}
								w4[l * 4 + 3] = r4[l * 4 + 3]; //copy flip value
							}
						}

						array_copy[Mesh::ARRAY_TANGENT] = tangents_v4;
					}

					Ref<SurfaceTool> blend_surface_tool;
					blend_surface_tool.instantiate();
					blend_surface_tool->create_from_triangle_arrays(array_copy);
					if (a.has("JOINTS_0") && a.has("JOINTS_1")) {
						blend_surface_tool->set_skin_weight_count(SurfaceTool::SKIN_8_WEIGHTS);
					}
					blend_surface_tool->index();
					if (generate_tangents) {
						blend_surface_tool->generate_tangents();
					}
					array_copy = blend_surface_tool->commit_to_arrays();

					morphs.push_back(array_copy);
				}
			}

			//just add it

			Ref<BaseMaterial3D> mat;
			if (p.has("material")) {
				const int material = p["material"];
				ERR_FAIL_INDEX_V(material, state->materials.size(), ERR_FILE_CORRUPT);
				Ref<BaseMaterial3D> mat3d = state->materials[material];
				ERR_FAIL_NULL_V(mat3d, ERR_FILE_CORRUPT);
				if (has_vertex_color) {
					mat3d->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
				}
				mat = mat3d;

			} else {
				Ref<StandardMaterial3D> mat3d;
				mat3d.instantiate();
				if (has_vertex_color) {
					mat3d->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
				}
				mat = mat3d;
			}
			ERR_FAIL_NULL_V(mat, ERR_FILE_CORRUPT);
			import_mesh->add_surface(primitive, array, morphs,
					Dictionary(), mat, mat->get_name(), flags);
		}

		Vector<float> blend_weights;
		blend_weights.resize(import_mesh->get_blend_shape_count());
		for (int32_t weight_i = 0; weight_i < blend_weights.size(); weight_i++) {
			blend_weights.write[weight_i] = 0.0f;
		}

		if (d.has("weights")) {
			const Array &weights = d["weights"];
			for (int j = 0; j < weights.size(); j++) {
				if (j >= blend_weights.size()) {
					break;
				}
				blend_weights.write[j] = weights[j];
			}
		}
		mesh->set_blend_weights(blend_weights);
		mesh->set_mesh(import_mesh);

		state->meshes.push_back(mesh);
	}

	print_verbose("glTF: Total meshes: " + itos(state->meshes.size()));

	return OK;
}

Error GLTFDocument::_serialize_images(Ref<GLTFState> state, const String &p_path) {
	Array images;
	for (int i = 0; i < state->images.size(); i++) {
		Dictionary d;

		ERR_CONTINUE(state->images[i].is_null());

		Ref<Image> image = state->images[i]->get_image();
		ERR_CONTINUE(image.is_null());

		if (p_path.to_lower().ends_with("glb") || p_path.is_empty()) {
			GLTFBufferViewIndex bvi;

			Ref<GLTFBufferView> bv;
			bv.instantiate();

			const GLTFBufferIndex bi = 0;
			bv->buffer = bi;
			bv->byte_offset = state->buffers[bi].size();
			ERR_FAIL_INDEX_V(bi, state->buffers.size(), ERR_PARAMETER_RANGE_ERROR);

			Vector<uint8_t> buffer;
			Ref<ImageTexture> img_tex = image;
			if (img_tex.is_valid()) {
				image = img_tex->get_image();
			}
			Error err = PNGDriverCommon::image_to_png(image, buffer);
			ERR_FAIL_COND_V_MSG(err, err, "Can't convert image to PNG.");

			bv->byte_length = buffer.size();
			state->buffers.write[bi].resize(state->buffers[bi].size() + bv->byte_length);
			memcpy(&state->buffers.write[bi].write[bv->byte_offset], buffer.ptr(), buffer.size());
			ERR_FAIL_COND_V(bv->byte_offset + bv->byte_length > state->buffers[bi].size(), ERR_FILE_CORRUPT);

			state->buffer_views.push_back(bv);
			bvi = state->buffer_views.size() - 1;
			d["bufferView"] = bvi;
			d["mimeType"] = "image/png";
		} else {
			ERR_FAIL_COND_V(p_path.is_empty(), ERR_INVALID_PARAMETER);
			String name = state->images[i]->get_name();
			if (name.is_empty()) {
				name = itos(i);
			}
			name = _gen_unique_name(state, name);
			name = name.pad_zeros(3) + ".png";
			String texture_dir = "textures";
			String path = p_path.get_base_dir();
			String new_texture_dir = path + "/" + texture_dir;
			DirAccessRef da = DirAccess::open(path);
			if (!da->dir_exists(new_texture_dir)) {
				da->make_dir(new_texture_dir);
			}
			image->save_png(new_texture_dir.plus_file(name));
			d["uri"] = texture_dir.plus_file(name).uri_encode();
		}
		images.push_back(d);
	}

	print_verbose("Total images: " + itos(state->images.size()));

	if (!images.size()) {
		return OK;
	}
	state->json["images"] = images;

	return OK;
}

Error GLTFDocument::_parse_images(Ref<GLTFState> state, const String &p_base_path) {
	ERR_FAIL_NULL_V(state, ERR_INVALID_PARAMETER);
	if (!state->json.has("images")) {
		return OK;
	}

	// Ref: https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#images

	const Array &images = state->json["images"];
	for (int i = 0; i < images.size(); i++) {
		const Dictionary &d = images[i];

		// glTF 2.0 supports PNG and JPEG types, which can be specified as (from spec):
		// "- a URI to an external file in one of the supported images formats, or
		//  - a URI with embedded base64-encoded data, or
		//  - a reference to a bufferView; in that case mimeType must be defined."
		// Since mimeType is optional for external files and base64 data, we'll have to
		// fall back on letting Godot parse the data to figure out if it's PNG or JPEG.

		// We'll assume that we use either URI or bufferView, so let's warn the user
		// if their image somehow uses both. And fail if it has neither.
		ERR_CONTINUE_MSG(!d.has("uri") && !d.has("bufferView"), "Invalid image definition in glTF file, it should specific an 'uri' or 'bufferView'.");
		if (d.has("uri") && d.has("bufferView")) {
			WARN_PRINT("Invalid image definition in glTF file using both 'uri' and 'bufferView'. 'bufferView' will take precedence.");
		}

		String mimetype;
		if (d.has("mimeType")) { // Should be "image/png" or "image/jpeg".
			mimetype = d["mimeType"];
		}

		Vector<uint8_t> data;
		const uint8_t *data_ptr = nullptr;
		int data_size = 0;

		if (d.has("uri")) {
			// Handles the first two bullet points from the spec (embedded data, or external file).
			String uri = d["uri"];

			if (uri.begins_with("data:")) { // Embedded data using base64.
				// Validate data MIME types and throw a warning if it's one we don't know/support.
				if (!uri.begins_with("data:application/octet-stream;base64") &&
						!uri.begins_with("data:application/gltf-buffer;base64") &&
						!uri.begins_with("data:image/png;base64") &&
						!uri.begins_with("data:image/jpeg;base64")) {
					WARN_PRINT(vformat("glTF: Image index '%d' uses an unsupported URI data type: %s. Skipping it.", i, uri));
					state->images.push_back(Ref<Texture2D>()); // Placeholder to keep count.
					continue;
				}
				data = _parse_base64_uri(uri);
				data_ptr = data.ptr();
				data_size = data.size();
				// mimeType is optional, but if we have it defined in the URI, let's use it.
				if (mimetype.is_empty()) {
					if (uri.begins_with("data:image/png;base64")) {
						mimetype = "image/png";
					} else if (uri.begins_with("data:image/jpeg;base64")) {
						mimetype = "image/jpeg";
					}
				}
			} else { // Relative path to an external image file.
				ERR_FAIL_COND_V(p_base_path.is_empty(), ERR_INVALID_PARAMETER);
				uri = uri.uri_decode();
				uri = p_base_path.plus_file(uri).replace("\\", "/"); // Fix for Windows.
				// ResourceLoader will rely on the file extension to use the relevant loader.
				// The spec says that if mimeType is defined, it should take precedence (e.g.
				// there could be a `.png` image which is actually JPEG), but there's no easy
				// API for that in Godot, so we'd have to load as a buffer (i.e. embedded in
				// the material), so we do this only as fallback.
				Ref<Texture2D> texture = ResourceLoader::load(uri);
				if (texture.is_valid()) {
					state->images.push_back(texture);
					continue;
				} else if (mimetype == "image/png" || mimetype == "image/jpeg") {
					// Fallback to loading as byte array.
					// This enables us to support the spec's requirement that we honor mimetype
					// regardless of file URI.
					data = FileAccess::get_file_as_array(uri);
					if (data.size() == 0) {
						WARN_PRINT(vformat("glTF: Image index '%d' couldn't be loaded as a buffer of MIME type '%s' from URI: %s. Skipping it.", i, mimetype, uri));
						state->images.push_back(Ref<Texture2D>()); // Placeholder to keep count.
						continue;
					}
					data_ptr = data.ptr();
					data_size = data.size();
				} else {
					WARN_PRINT(vformat("glTF: Image index '%d' couldn't be loaded from URI: %s. Skipping it.", i, uri));
					state->images.push_back(Ref<Texture2D>()); // Placeholder to keep count.
					continue;
				}
			}
		} else if (d.has("bufferView")) {
			// Handles the third bullet point from the spec (bufferView).
			ERR_FAIL_COND_V_MSG(mimetype.is_empty(), ERR_FILE_CORRUPT,
					vformat("glTF: Image index '%d' specifies 'bufferView' but no 'mimeType', which is invalid.", i));

			const GLTFBufferViewIndex bvi = d["bufferView"];

			ERR_FAIL_INDEX_V(bvi, state->buffer_views.size(), ERR_PARAMETER_RANGE_ERROR);

			Ref<GLTFBufferView> bv = state->buffer_views[bvi];

			const GLTFBufferIndex bi = bv->buffer;
			ERR_FAIL_INDEX_V(bi, state->buffers.size(), ERR_PARAMETER_RANGE_ERROR);

			ERR_FAIL_COND_V(bv->byte_offset + bv->byte_length > state->buffers[bi].size(), ERR_FILE_CORRUPT);

			data_ptr = &state->buffers[bi][bv->byte_offset];
			data_size = bv->byte_length;
		}

		Ref<Image> img;

		// First we honor the mime types if they were defined.
		if (mimetype == "image/png") { // Load buffer as PNG.
			ERR_FAIL_COND_V(Image::_png_mem_loader_func == nullptr, ERR_UNAVAILABLE);
			img = Image::_png_mem_loader_func(data_ptr, data_size);
		} else if (mimetype == "image/jpeg") { // Loader buffer as JPEG.
			ERR_FAIL_COND_V(Image::_jpg_mem_loader_func == nullptr, ERR_UNAVAILABLE);
			img = Image::_jpg_mem_loader_func(data_ptr, data_size);
		}

		// If we didn't pass the above tests, we attempt loading as PNG and then
		// JPEG directly.
		// This covers URIs with base64-encoded data with application/* type but
		// no optional mimeType property, or bufferViews with a bogus mimeType
		// (e.g. `image/jpeg` but the data is actually PNG).
		// That's not *exactly* what the spec mandates but this lets us be
		// lenient with bogus glb files which do exist in production.
		if (img.is_null()) { // Try PNG first.
			ERR_FAIL_COND_V(Image::_png_mem_loader_func == nullptr, ERR_UNAVAILABLE);
			img = Image::_png_mem_loader_func(data_ptr, data_size);
		}
		if (img.is_null()) { // And then JPEG.
			ERR_FAIL_COND_V(Image::_jpg_mem_loader_func == nullptr, ERR_UNAVAILABLE);
			img = Image::_jpg_mem_loader_func(data_ptr, data_size);
		}
		// Now we've done our best, fix your scenes.
		if (img.is_null()) {
			ERR_PRINT(vformat("glTF: Couldn't load image index '%d' with its given mimetype: %s.", i, mimetype));
			state->images.push_back(Ref<Texture2D>());
			continue;
		}

		Ref<ImageTexture> t;
		t.instantiate();
		t->create_from_image(img);

		state->images.push_back(t);
	}

	print_verbose("glTF: Total images: " + itos(state->images.size()));

	return OK;
}

Error GLTFDocument::_serialize_textures(Ref<GLTFState> state) {
	if (!state->textures.size()) {
		return OK;
	}

	Array textures;
	for (int32_t i = 0; i < state->textures.size(); i++) {
		Dictionary d;
		Ref<GLTFTexture> t = state->textures[i];
		ERR_CONTINUE(t->get_src_image() == -1);
		d["source"] = t->get_src_image();
		textures.push_back(d);
	}
	state->json["textures"] = textures;

	return OK;
}

Error GLTFDocument::_parse_textures(Ref<GLTFState> state) {
	if (!state->json.has("textures")) {
		return OK;
	}

	const Array &textures = state->json["textures"];
	for (GLTFTextureIndex i = 0; i < textures.size(); i++) {
		const Dictionary &d = textures[i];

		ERR_FAIL_COND_V(!d.has("source"), ERR_PARSE_ERROR);

		Ref<GLTFTexture> t;
		t.instantiate();
		t->set_src_image(d["source"]);
		state->textures.push_back(t);
	}

	return OK;
}

GLTFTextureIndex GLTFDocument::_set_texture(Ref<GLTFState> state, Ref<Texture2D> p_texture) {
	ERR_FAIL_COND_V(p_texture.is_null(), -1);
	Ref<GLTFTexture> gltf_texture;
	gltf_texture.instantiate();
	ERR_FAIL_COND_V(p_texture->get_image().is_null(), -1);
	GLTFImageIndex gltf_src_image_i = state->images.size();
	state->images.push_back(p_texture);
	gltf_texture->set_src_image(gltf_src_image_i);
	GLTFTextureIndex gltf_texture_i = state->textures.size();
	state->textures.push_back(gltf_texture);
	return gltf_texture_i;
}

Ref<Texture2D> GLTFDocument::_get_texture(Ref<GLTFState> state, const GLTFTextureIndex p_texture) {
	ERR_FAIL_INDEX_V(p_texture, state->textures.size(), Ref<Texture2D>());
	const GLTFImageIndex image = state->textures[p_texture]->get_src_image();

	ERR_FAIL_INDEX_V(image, state->images.size(), Ref<Texture2D>());

	return state->images[image];
}

Error GLTFDocument::_serialize_materials(Ref<GLTFState> state) {
	Array materials;
	for (int32_t i = 0; i < state->materials.size(); i++) {
		Dictionary d;

		Ref<BaseMaterial3D> material = state->materials[i];
		if (material.is_null()) {
			materials.push_back(d);
			continue;
		}
		if (!material->get_name().is_empty()) {
			d["name"] = _gen_unique_name(state, material->get_name());
		}
		{
			Dictionary mr;
			{
				Array arr;
				const Color c = material->get_albedo().to_linear();
				arr.push_back(c.r);
				arr.push_back(c.g);
				arr.push_back(c.b);
				arr.push_back(c.a);
				mr["baseColorFactor"] = arr;
			}
			{
				Dictionary bct;
				Ref<Texture2D> albedo_texture = material->get_texture(BaseMaterial3D::TEXTURE_ALBEDO);
				GLTFTextureIndex gltf_texture_index = -1;

				if (albedo_texture.is_valid() && albedo_texture->get_image().is_valid()) {
					albedo_texture->set_name(material->get_name() + "_albedo");
					gltf_texture_index = _set_texture(state, albedo_texture);
				}
				if (gltf_texture_index != -1) {
					bct["index"] = gltf_texture_index;
					bct["extensions"] = _serialize_texture_transform_uv1(material);
					mr["baseColorTexture"] = bct;
				}
			}

			mr["metallicFactor"] = material->get_metallic();
			mr["roughnessFactor"] = material->get_roughness();
			bool has_roughness = material->get_texture(BaseMaterial3D::TEXTURE_ROUGHNESS).is_valid() && material->get_texture(BaseMaterial3D::TEXTURE_ROUGHNESS)->get_image().is_valid();
			bool has_ao = material->get_feature(BaseMaterial3D::FEATURE_AMBIENT_OCCLUSION) && material->get_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION).is_valid();
			bool has_metalness = material->get_texture(BaseMaterial3D::TEXTURE_METALLIC).is_valid() && material->get_texture(BaseMaterial3D::TEXTURE_METALLIC)->get_image().is_valid();
			if (has_ao || has_roughness || has_metalness) {
				Dictionary mrt;
				Ref<Texture2D> roughness_texture = material->get_texture(BaseMaterial3D::TEXTURE_ROUGHNESS);
				BaseMaterial3D::TextureChannel roughness_channel = material->get_roughness_texture_channel();
				Ref<Texture2D> metallic_texture = material->get_texture(BaseMaterial3D::TEXTURE_METALLIC);
				BaseMaterial3D::TextureChannel metalness_channel = material->get_metallic_texture_channel();
				Ref<Texture2D> ao_texture = material->get_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION);
				BaseMaterial3D::TextureChannel ao_channel = material->get_ao_texture_channel();
				Ref<ImageTexture> orm_texture;
				orm_texture.instantiate();
				Ref<Image> orm_image;
				orm_image.instantiate();
				int32_t height = 0;
				int32_t width = 0;
				Ref<Image> ao_image;
				if (has_ao) {
					height = ao_texture->get_height();
					width = ao_texture->get_width();
					ao_image = ao_texture->get_image();
					Ref<ImageTexture> img_tex = ao_image;
					if (img_tex.is_valid()) {
						ao_image = img_tex->get_image();
					}
					if (ao_image->is_compressed()) {
						ao_image->decompress();
					}
				}
				Ref<Image> roughness_image;
				if (has_roughness) {
					height = roughness_texture->get_height();
					width = roughness_texture->get_width();
					roughness_image = roughness_texture->get_image();
					Ref<ImageTexture> img_tex = roughness_image;
					if (img_tex.is_valid()) {
						roughness_image = img_tex->get_image();
					}
					if (roughness_image->is_compressed()) {
						roughness_image->decompress();
					}
				}
				Ref<Image> metallness_image;
				if (has_metalness) {
					height = metallic_texture->get_height();
					width = metallic_texture->get_width();
					metallness_image = metallic_texture->get_image();
					Ref<ImageTexture> img_tex = metallness_image;
					if (img_tex.is_valid()) {
						metallness_image = img_tex->get_image();
					}
					if (metallness_image->is_compressed()) {
						metallness_image->decompress();
					}
				}
				Ref<Texture2D> albedo_texture = material->get_texture(BaseMaterial3D::TEXTURE_ALBEDO);
				if (albedo_texture.is_valid() && albedo_texture->get_image().is_valid()) {
					height = albedo_texture->get_height();
					width = albedo_texture->get_width();
				}
				orm_image->create(width, height, false, Image::FORMAT_RGBA8);
				if (ao_image.is_valid() && ao_image->get_size() != Vector2(width, height)) {
					ao_image->resize(width, height, Image::INTERPOLATE_LANCZOS);
				}
				if (roughness_image.is_valid() && roughness_image->get_size() != Vector2(width, height)) {
					roughness_image->resize(width, height, Image::INTERPOLATE_LANCZOS);
				}
				if (metallness_image.is_valid() && metallness_image->get_size() != Vector2(width, height)) {
					metallness_image->resize(width, height, Image::INTERPOLATE_LANCZOS);
				}
				for (int32_t h = 0; h < height; h++) {
					for (int32_t w = 0; w < width; w++) {
						Color c = Color(1.0f, 1.0f, 1.0f);
						if (has_ao) {
							if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_RED == ao_channel) {
								c.r = ao_image->get_pixel(w, h).r;
							} else if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_GREEN == ao_channel) {
								c.r = ao_image->get_pixel(w, h).g;
							} else if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_BLUE == ao_channel) {
								c.r = ao_image->get_pixel(w, h).b;
							} else if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_ALPHA == ao_channel) {
								c.r = ao_image->get_pixel(w, h).a;
							}
						}
						if (has_roughness) {
							if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_RED == roughness_channel) {
								c.g = roughness_image->get_pixel(w, h).r;
							} else if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_GREEN == roughness_channel) {
								c.g = roughness_image->get_pixel(w, h).g;
							} else if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_BLUE == roughness_channel) {
								c.g = roughness_image->get_pixel(w, h).b;
							} else if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_ALPHA == roughness_channel) {
								c.g = roughness_image->get_pixel(w, h).a;
							}
						}
						if (has_metalness) {
							if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_RED == metalness_channel) {
								c.b = metallness_image->get_pixel(w, h).r;
							} else if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_GREEN == metalness_channel) {
								c.b = metallness_image->get_pixel(w, h).g;
							} else if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_BLUE == metalness_channel) {
								c.b = metallness_image->get_pixel(w, h).b;
							} else if (BaseMaterial3D::TextureChannel::TEXTURE_CHANNEL_ALPHA == metalness_channel) {
								c.b = metallness_image->get_pixel(w, h).a;
							}
						}
						orm_image->set_pixel(w, h, c);
					}
				}
				orm_image->generate_mipmaps();
				orm_texture->create_from_image(orm_image);
				GLTFTextureIndex orm_texture_index = -1;
				if (has_ao || has_roughness || has_metalness) {
					orm_texture->set_name(material->get_name() + "_orm");
					orm_texture_index = _set_texture(state, orm_texture);
				}
				if (has_ao) {
					Dictionary ot;
					ot["index"] = orm_texture_index;
					d["occlusionTexture"] = ot;
				}
				if (has_roughness || has_metalness) {
					mrt["index"] = orm_texture_index;
					mrt["extensions"] = _serialize_texture_transform_uv1(material);
					mr["metallicRoughnessTexture"] = mrt;
				}
			}
			d["pbrMetallicRoughness"] = mr;
		}

		if (material->get_feature(BaseMaterial3D::FEATURE_NORMAL_MAPPING)) {
			Dictionary nt;
			Ref<ImageTexture> tex;
			tex.instantiate();
			{
				Ref<Texture2D> normal_texture = material->get_texture(BaseMaterial3D::TEXTURE_NORMAL);
				if (normal_texture.is_valid()) {
					// Code for uncompressing RG normal maps
					Ref<Image> img = normal_texture->get_image();
					if (img.is_valid()) {
						Ref<ImageTexture> img_tex = img;
						if (img_tex.is_valid()) {
							img = img_tex->get_image();
						}
						img->decompress();
						img->convert(Image::FORMAT_RGBA8);
						img->convert_ra_rgba8_to_rg();
						for (int32_t y = 0; y < img->get_height(); y++) {
							for (int32_t x = 0; x < img->get_width(); x++) {
								Color c = img->get_pixel(x, y);
								Vector2 red_green = Vector2(c.r, c.g);
								red_green = red_green * Vector2(2.0f, 2.0f) - Vector2(1.0f, 1.0f);
								float blue = 1.0f - red_green.dot(red_green);
								blue = MAX(0.0f, blue);
								c.b = Math::sqrt(blue);
								img->set_pixel(x, y, c);
							}
						}
						tex->create_from_image(img);
					}
				}
			}
			GLTFTextureIndex gltf_texture_index = -1;
			if (tex.is_valid() && tex->get_image().is_valid()) {
				tex->set_name(material->get_name() + "_normal");
				gltf_texture_index = _set_texture(state, tex);
			}
			nt["scale"] = material->get_normal_scale();
			if (gltf_texture_index != -1) {
				nt["index"] = gltf_texture_index;
				d["normalTexture"] = nt;
			}
		}

		if (material->get_feature(BaseMaterial3D::FEATURE_EMISSION)) {
			const Color c = material->get_emission().to_srgb();
			Array arr;
			arr.push_back(c.r);
			arr.push_back(c.g);
			arr.push_back(c.b);
			d["emissiveFactor"] = arr;
		}
		if (material->get_feature(BaseMaterial3D::FEATURE_EMISSION)) {
			Dictionary et;
			Ref<Texture2D> emission_texture = material->get_texture(BaseMaterial3D::TEXTURE_EMISSION);
			GLTFTextureIndex gltf_texture_index = -1;
			if (emission_texture.is_valid() && emission_texture->get_image().is_valid()) {
				emission_texture->set_name(material->get_name() + "_emission");
				gltf_texture_index = _set_texture(state, emission_texture);
			}

			if (gltf_texture_index != -1) {
				et["index"] = gltf_texture_index;
				d["emissiveTexture"] = et;
			}
		}
		const bool ds = material->get_cull_mode() == BaseMaterial3D::CULL_DISABLED;
		if (ds) {
			d["doubleSided"] = ds;
		}
		if (material->get_transparency() == BaseMaterial3D::TRANSPARENCY_ALPHA_SCISSOR) {
			d["alphaMode"] = "MASK";
			d["alphaCutoff"] = material->get_alpha_scissor_threshold();
		} else if (material->get_transparency() != BaseMaterial3D::TRANSPARENCY_DISABLED) {
			d["alphaMode"] = "BLEND";
		}
		materials.push_back(d);
	}
	if (!materials.size()) {
		return OK;
	}
	state->json["materials"] = materials;
	print_verbose("Total materials: " + itos(state->materials.size()));

	return OK;
}

Error GLTFDocument::_parse_materials(Ref<GLTFState> state) {
	if (!state->json.has("materials")) {
		return OK;
	}

	const Array &materials = state->json["materials"];
	for (GLTFMaterialIndex i = 0; i < materials.size(); i++) {
		const Dictionary &d = materials[i];

		Ref<StandardMaterial3D> material;
		material.instantiate();
		if (d.has("name") && !String(d["name"]).is_empty()) {
			material->set_name(d["name"]);
		} else {
			material->set_name(vformat("material_%s", itos(i)));
		}
		material->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		Dictionary pbr_spec_gloss_extensions;
		if (d.has("extensions")) {
			pbr_spec_gloss_extensions = d["extensions"];
		}
		if (pbr_spec_gloss_extensions.has("KHR_materials_pbrSpecularGlossiness")) {
			WARN_PRINT("Material uses a specular and glossiness workflow. Textures will be converted to roughness and metallic workflow, which may not be 100% accurate.");
			Dictionary sgm = pbr_spec_gloss_extensions["KHR_materials_pbrSpecularGlossiness"];

			Ref<GLTFSpecGloss> spec_gloss;
			spec_gloss.instantiate();
			if (sgm.has("diffuseTexture")) {
				const Dictionary &diffuse_texture_dict = sgm["diffuseTexture"];
				if (diffuse_texture_dict.has("index")) {
					Ref<Texture2D> diffuse_texture = _get_texture(state, diffuse_texture_dict["index"]);
					if (diffuse_texture.is_valid()) {
						spec_gloss->diffuse_img = diffuse_texture->get_image();
						material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, diffuse_texture);
					}
				}
			}
			if (sgm.has("diffuseFactor")) {
				const Array &arr = sgm["diffuseFactor"];
				ERR_FAIL_COND_V(arr.size() != 4, ERR_PARSE_ERROR);
				const Color c = Color(arr[0], arr[1], arr[2], arr[3]).to_srgb();
				spec_gloss->diffuse_factor = c;
				material->set_albedo(spec_gloss->diffuse_factor);
			}

			if (sgm.has("specularFactor")) {
				const Array &arr = sgm["specularFactor"];
				ERR_FAIL_COND_V(arr.size() != 3, ERR_PARSE_ERROR);
				spec_gloss->specular_factor = Color(arr[0], arr[1], arr[2]);
			}

			if (sgm.has("glossinessFactor")) {
				spec_gloss->gloss_factor = sgm["glossinessFactor"];
				material->set_roughness(1.0f - CLAMP(spec_gloss->gloss_factor, 0.0f, 1.0f));
			}
			if (sgm.has("specularGlossinessTexture")) {
				const Dictionary &spec_gloss_texture = sgm["specularGlossinessTexture"];
				if (spec_gloss_texture.has("index")) {
					const Ref<Texture2D> orig_texture = _get_texture(state, spec_gloss_texture["index"]);
					if (orig_texture.is_valid()) {
						spec_gloss->spec_gloss_img = orig_texture->get_image();
					}
				}
			}
			spec_gloss_to_rough_metal(spec_gloss, material);

		} else if (d.has("pbrMetallicRoughness")) {
			const Dictionary &mr = d["pbrMetallicRoughness"];
			if (mr.has("baseColorFactor")) {
				const Array &arr = mr["baseColorFactor"];
				ERR_FAIL_COND_V(arr.size() != 4, ERR_PARSE_ERROR);
				const Color c = Color(arr[0], arr[1], arr[2], arr[3]).to_srgb();
				material->set_albedo(c);
			}

			if (mr.has("baseColorTexture")) {
				const Dictionary &bct = mr["baseColorTexture"];
				if (bct.has("index")) {
					material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, _get_texture(state, bct["index"]));
				}
				if (!mr.has("baseColorFactor")) {
					material->set_albedo(Color(1, 1, 1));
				}
				_set_texture_transform_uv1(bct, material);
			}

			if (mr.has("metallicFactor")) {
				material->set_metallic(mr["metallicFactor"]);
			} else {
				material->set_metallic(1.0);
			}

			if (mr.has("roughnessFactor")) {
				material->set_roughness(mr["roughnessFactor"]);
			} else {
				material->set_roughness(1.0);
			}

			if (mr.has("metallicRoughnessTexture")) {
				const Dictionary &bct = mr["metallicRoughnessTexture"];
				if (bct.has("index")) {
					const Ref<Texture2D> t = _get_texture(state, bct["index"]);
					material->set_texture(BaseMaterial3D::TEXTURE_METALLIC, t);
					material->set_metallic_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_BLUE);
					material->set_texture(BaseMaterial3D::TEXTURE_ROUGHNESS, t);
					material->set_roughness_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_GREEN);
					if (!mr.has("metallicFactor")) {
						material->set_metallic(1);
					}
					if (!mr.has("roughnessFactor")) {
						material->set_roughness(1);
					}
				}
			}
		}

		if (d.has("normalTexture")) {
			const Dictionary &bct = d["normalTexture"];
			if (bct.has("index")) {
				material->set_texture(BaseMaterial3D::TEXTURE_NORMAL, _get_texture(state, bct["index"]));
				material->set_feature(BaseMaterial3D::FEATURE_NORMAL_MAPPING, true);
			}
			if (bct.has("scale")) {
				material->set_normal_scale(bct["scale"]);
			}
		}
		if (d.has("occlusionTexture")) {
			const Dictionary &bct = d["occlusionTexture"];
			if (bct.has("index")) {
				material->set_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION, _get_texture(state, bct["index"]));
				material->set_ao_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_RED);
				material->set_feature(BaseMaterial3D::FEATURE_AMBIENT_OCCLUSION, true);
			}
		}

		if (d.has("emissiveFactor")) {
			const Array &arr = d["emissiveFactor"];
			ERR_FAIL_COND_V(arr.size() != 3, ERR_PARSE_ERROR);
			const Color c = Color(arr[0], arr[1], arr[2]).to_srgb();
			material->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);

			material->set_emission(c);
		}

		if (d.has("emissiveTexture")) {
			const Dictionary &bct = d["emissiveTexture"];
			if (bct.has("index")) {
				material->set_texture(BaseMaterial3D::TEXTURE_EMISSION, _get_texture(state, bct["index"]));
				material->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);
				material->set_emission(Color(0, 0, 0));
			}
		}

		if (d.has("doubleSided")) {
			const bool ds = d["doubleSided"];
			if (ds) {
				material->set_cull_mode(BaseMaterial3D::CULL_DISABLED);
			}
		}
		if (d.has("alphaMode")) {
			const String &am = d["alphaMode"];
			if (am == "BLEND") {
				material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA_DEPTH_PRE_PASS);
			} else if (am == "MASK") {
				material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA_SCISSOR);
				if (d.has("alphaCutoff")) {
					material->set_alpha_scissor_threshold(d["alphaCutoff"]);
				} else {
					material->set_alpha_scissor_threshold(0.5f);
				}
			}
		}
		state->materials.push_back(material);
	}

	print_verbose("Total materials: " + itos(state->materials.size()));

	return OK;
}

void GLTFDocument::_set_texture_transform_uv1(const Dictionary &d, Ref<BaseMaterial3D> material) {
	if (d.has("extensions")) {
		const Dictionary &extensions = d["extensions"];
		if (extensions.has("KHR_texture_transform")) {
			const Dictionary &texture_transform = extensions["KHR_texture_transform"];
			const Array &offset_arr = texture_transform["offset"];
			if (offset_arr.size() == 2) {
				const Vector3 offset_vector3 = Vector3(offset_arr[0], offset_arr[1], 0.0f);
				material->set_uv1_offset(offset_vector3);
			}

			const Array &scale_arr = texture_transform["scale"];
			if (scale_arr.size() == 2) {
				const Vector3 scale_vector3 = Vector3(scale_arr[0], scale_arr[1], 1.0f);
				material->set_uv1_scale(scale_vector3);
			}
		}
	}
}

void GLTFDocument::spec_gloss_to_rough_metal(Ref<GLTFSpecGloss> r_spec_gloss, Ref<BaseMaterial3D> p_material) {
	if (r_spec_gloss->spec_gloss_img.is_null()) {
		return;
	}
	if (r_spec_gloss->diffuse_img.is_null()) {
		return;
	}
	Ref<Image> rm_img;
	rm_img.instantiate();
	bool has_roughness = false;
	bool has_metal = false;
	p_material->set_roughness(1.0f);
	p_material->set_metallic(1.0f);
	rm_img->create(r_spec_gloss->spec_gloss_img->get_width(), r_spec_gloss->spec_gloss_img->get_height(), false, Image::FORMAT_RGBA8);
	r_spec_gloss->spec_gloss_img->decompress();
	if (r_spec_gloss->diffuse_img.is_valid()) {
		r_spec_gloss->diffuse_img->decompress();
		r_spec_gloss->diffuse_img->resize(r_spec_gloss->spec_gloss_img->get_width(), r_spec_gloss->spec_gloss_img->get_height(), Image::INTERPOLATE_LANCZOS);
		r_spec_gloss->spec_gloss_img->resize(r_spec_gloss->diffuse_img->get_width(), r_spec_gloss->diffuse_img->get_height(), Image::INTERPOLATE_LANCZOS);
	}
	for (int32_t y = 0; y < r_spec_gloss->spec_gloss_img->get_height(); y++) {
		for (int32_t x = 0; x < r_spec_gloss->spec_gloss_img->get_width(); x++) {
			const Color specular_pixel = r_spec_gloss->spec_gloss_img->get_pixel(x, y).to_linear();
			Color specular = Color(specular_pixel.r, specular_pixel.g, specular_pixel.b);
			specular *= r_spec_gloss->specular_factor;
			Color diffuse = Color(1.0f, 1.0f, 1.0f);
			diffuse *= r_spec_gloss->diffuse_img->get_pixel(x, y).to_linear();
			float metallic = 0.0f;
			Color base_color;
			spec_gloss_to_metal_base_color(specular, diffuse, base_color, metallic);
			Color mr = Color(1.0f, 1.0f, 1.0f);
			mr.g = specular_pixel.a;
			mr.b = metallic;
			if (!Math::is_equal_approx(mr.g, 1.0f)) {
				has_roughness = true;
			}
			if (!Math::is_zero_approx(mr.b)) {
				has_metal = true;
			}
			mr.g *= r_spec_gloss->gloss_factor;
			mr.g = 1.0f - mr.g;
			rm_img->set_pixel(x, y, mr);
			if (r_spec_gloss->diffuse_img.is_valid()) {
				r_spec_gloss->diffuse_img->set_pixel(x, y, base_color.to_srgb());
			}
		}
	}
	rm_img->generate_mipmaps();
	r_spec_gloss->diffuse_img->generate_mipmaps();
	Ref<ImageTexture> diffuse_image_texture;
	diffuse_image_texture.instantiate();
	diffuse_image_texture->create_from_image(r_spec_gloss->diffuse_img);
	p_material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, diffuse_image_texture);
	Ref<ImageTexture> rm_image_texture;
	rm_image_texture.instantiate();
	rm_image_texture->create_from_image(rm_img);
	if (has_roughness) {
		p_material->set_texture(BaseMaterial3D::TEXTURE_ROUGHNESS, rm_image_texture);
		p_material->set_roughness_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_GREEN);
	}

	if (has_metal) {
		p_material->set_texture(BaseMaterial3D::TEXTURE_METALLIC, rm_image_texture);
		p_material->set_metallic_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_BLUE);
	}
}

void GLTFDocument::spec_gloss_to_metal_base_color(const Color &p_specular_factor, const Color &p_diffuse, Color &r_base_color, float &r_metallic) {
	const Color DIELECTRIC_SPECULAR = Color(0.04f, 0.04f, 0.04f);
	Color specular = Color(p_specular_factor.r, p_specular_factor.g, p_specular_factor.b);
	const float one_minus_specular_strength = 1.0f - get_max_component(specular);
	const float dielectric_specular_red = DIELECTRIC_SPECULAR.r;
	float brightness_diffuse = get_perceived_brightness(p_diffuse);
	const float brightness_specular = get_perceived_brightness(specular);
	r_metallic = solve_metallic(dielectric_specular_red, brightness_diffuse, brightness_specular, one_minus_specular_strength);
	const float one_minus_metallic = 1.0f - r_metallic;
	const Color base_color_from_diffuse = p_diffuse * (one_minus_specular_strength / (1.0f - dielectric_specular_red) / MAX(one_minus_metallic, CMP_EPSILON));
	const Color base_color_from_specular = (specular - (DIELECTRIC_SPECULAR * (one_minus_metallic))) * (1.0f / MAX(r_metallic, CMP_EPSILON));
	r_base_color.r = Math::lerp(base_color_from_diffuse.r, base_color_from_specular.r, r_metallic * r_metallic);
	r_base_color.g = Math::lerp(base_color_from_diffuse.g, base_color_from_specular.g, r_metallic * r_metallic);
	r_base_color.b = Math::lerp(base_color_from_diffuse.b, base_color_from_specular.b, r_metallic * r_metallic);
	r_base_color.a = p_diffuse.a;
	r_base_color = r_base_color.clamp();
}

GLTFNodeIndex GLTFDocument::_find_highest_node(Ref<GLTFState> state, const Vector<GLTFNodeIndex> &subset) {
	int highest = -1;
	GLTFNodeIndex best_node = -1;

	for (int i = 0; i < subset.size(); ++i) {
		const GLTFNodeIndex node_i = subset[i];
		const Ref<GLTFNode> node = state->nodes[node_i];

		if (highest == -1 || node->height < highest) {
			highest = node->height;
			best_node = node_i;
		}
	}

	return best_node;
}

bool GLTFDocument::_capture_nodes_in_skin(Ref<GLTFState> state, Ref<GLTFSkin> skin, const GLTFNodeIndex node_index) {
	bool found_joint = false;

	for (int i = 0; i < state->nodes[node_index]->children.size(); ++i) {
		found_joint |= _capture_nodes_in_skin(state, skin, state->nodes[node_index]->children[i]);
	}

	if (found_joint) {
		// Mark it if we happen to find another skins joint...
		if (state->nodes[node_index]->joint && skin->joints.find(node_index) < 0) {
			skin->joints.push_back(node_index);
		} else if (skin->non_joints.find(node_index) < 0) {
			skin->non_joints.push_back(node_index);
		}
	}

	if (skin->joints.find(node_index) > 0) {
		return true;
	}

	return false;
}

void GLTFDocument::_capture_nodes_for_multirooted_skin(Ref<GLTFState> state, Ref<GLTFSkin> skin) {
	DisjointSet<GLTFNodeIndex> disjoint_set;

	for (int i = 0; i < skin->joints.size(); ++i) {
		const GLTFNodeIndex node_index = skin->joints[i];
		const GLTFNodeIndex parent = state->nodes[node_index]->parent;
		disjoint_set.insert(node_index);

		if (skin->joints.find(parent) >= 0) {
			disjoint_set.create_union(parent, node_index);
		}
	}

	Vector<GLTFNodeIndex> roots;
	disjoint_set.get_representatives(roots);

	if (roots.size() <= 1) {
		return;
	}

	int maxHeight = -1;

	// Determine the max height rooted tree
	for (int i = 0; i < roots.size(); ++i) {
		const GLTFNodeIndex root = roots[i];

		if (maxHeight == -1 || state->nodes[root]->height < maxHeight) {
			maxHeight = state->nodes[root]->height;
		}
	}

	// Go up the tree till all of the multiple roots of the skin are at the same hierarchy level.
	// This sucks, but 99% of all game engines (not just Godot) would have this same issue.
	for (int i = 0; i < roots.size(); ++i) {
		GLTFNodeIndex current_node = roots[i];
		while (state->nodes[current_node]->height > maxHeight) {
			GLTFNodeIndex parent = state->nodes[current_node]->parent;

			if (state->nodes[parent]->joint && skin->joints.find(parent) < 0) {
				skin->joints.push_back(parent);
			} else if (skin->non_joints.find(parent) < 0) {
				skin->non_joints.push_back(parent);
			}

			current_node = parent;
		}

		// replace the roots
		roots.write[i] = current_node;
	}

	// Climb up the tree until they all have the same parent
	bool all_same;

	do {
		all_same = true;
		const GLTFNodeIndex first_parent = state->nodes[roots[0]]->parent;

		for (int i = 1; i < roots.size(); ++i) {
			all_same &= (first_parent == state->nodes[roots[i]]->parent);
		}

		if (!all_same) {
			for (int i = 0; i < roots.size(); ++i) {
				const GLTFNodeIndex current_node = roots[i];
				const GLTFNodeIndex parent = state->nodes[current_node]->parent;

				if (state->nodes[parent]->joint && skin->joints.find(parent) < 0) {
					skin->joints.push_back(parent);
				} else if (skin->non_joints.find(parent) < 0) {
					skin->non_joints.push_back(parent);
				}

				roots.write[i] = parent;
			}
		}

	} while (!all_same);
}

Error GLTFDocument::_expand_skin(Ref<GLTFState> state, Ref<GLTFSkin> skin) {
	_capture_nodes_for_multirooted_skin(state, skin);

	// Grab all nodes that lay in between skin joints/nodes
	DisjointSet<GLTFNodeIndex> disjoint_set;

	Vector<GLTFNodeIndex> all_skin_nodes;
	all_skin_nodes.append_array(skin->joints);
	all_skin_nodes.append_array(skin->non_joints);

	for (int i = 0; i < all_skin_nodes.size(); ++i) {
		const GLTFNodeIndex node_index = all_skin_nodes[i];
		const GLTFNodeIndex parent = state->nodes[node_index]->parent;
		disjoint_set.insert(node_index);

		if (all_skin_nodes.find(parent) >= 0) {
			disjoint_set.create_union(parent, node_index);
		}
	}

	Vector<GLTFNodeIndex> out_owners;
	disjoint_set.get_representatives(out_owners);

	Vector<GLTFNodeIndex> out_roots;

	for (int i = 0; i < out_owners.size(); ++i) {
		Vector<GLTFNodeIndex> set;
		disjoint_set.get_members(set, out_owners[i]);

		const GLTFNodeIndex root = _find_highest_node(state, set);
		ERR_FAIL_COND_V(root < 0, FAILED);
		out_roots.push_back(root);
	}

	out_roots.sort();

	for (int i = 0; i < out_roots.size(); ++i) {
		_capture_nodes_in_skin(state, skin, out_roots[i]);
	}

	skin->roots = out_roots;

	return OK;
}

Error GLTFDocument::_verify_skin(Ref<GLTFState> state, Ref<GLTFSkin> skin) {
	// This may seem duplicated from expand_skins, but this is really a sanity check! (so it kinda is)
	// In case additional interpolating logic is added to the skins, this will help ensure that you
	// do not cause it to self implode into a fiery blaze

	// We are going to re-calculate the root nodes and compare them to the ones saved in the skin,
	// then ensure the multiple trees (if they exist) are on the same sublevel

	// Grab all nodes that lay in between skin joints/nodes
	DisjointSet<GLTFNodeIndex> disjoint_set;

	Vector<GLTFNodeIndex> all_skin_nodes;
	all_skin_nodes.append_array(skin->joints);
	all_skin_nodes.append_array(skin->non_joints);

	for (int i = 0; i < all_skin_nodes.size(); ++i) {
		const GLTFNodeIndex node_index = all_skin_nodes[i];
		const GLTFNodeIndex parent = state->nodes[node_index]->parent;
		disjoint_set.insert(node_index);

		if (all_skin_nodes.find(parent) >= 0) {
			disjoint_set.create_union(parent, node_index);
		}
	}

	Vector<GLTFNodeIndex> out_owners;
	disjoint_set.get_representatives(out_owners);

	Vector<GLTFNodeIndex> out_roots;

	for (int i = 0; i < out_owners.size(); ++i) {
		Vector<GLTFNodeIndex> set;
		disjoint_set.get_members(set, out_owners[i]);

		const GLTFNodeIndex root = _find_highest_node(state, set);
		ERR_FAIL_COND_V(root < 0, FAILED);
		out_roots.push_back(root);
	}

	out_roots.sort();

	ERR_FAIL_COND_V(out_roots.size() == 0, FAILED);

	// Make sure the roots are the exact same (they better be)
	ERR_FAIL_COND_V(out_roots.size() != skin->roots.size(), FAILED);
	for (int i = 0; i < out_roots.size(); ++i) {
		ERR_FAIL_COND_V(out_roots[i] != skin->roots[i], FAILED);
	}

	// Single rooted skin? Perfectly ok!
	if (out_roots.size() == 1) {
		return OK;
	}

	// Make sure all parents of a multi-rooted skin are the SAME
	const GLTFNodeIndex parent = state->nodes[out_roots[0]]->parent;
	for (int i = 1; i < out_roots.size(); ++i) {
		if (state->nodes[out_roots[i]]->parent != parent) {
			return FAILED;
		}
	}

	return OK;
}

Error GLTFDocument::_parse_skins(Ref<GLTFState> state) {
	if (!state->json.has("skins")) {
		return OK;
	}

	const Array &skins = state->json["skins"];

	// Create the base skins, and mark nodes that are joints
	for (int i = 0; i < skins.size(); i++) {
		const Dictionary &d = skins[i];

		Ref<GLTFSkin> skin;
		skin.instantiate();

		ERR_FAIL_COND_V(!d.has("joints"), ERR_PARSE_ERROR);

		const Array &joints = d["joints"];

		if (d.has("inverseBindMatrices")) {
			skin->inverse_binds = _decode_accessor_as_xform(state, d["inverseBindMatrices"], false);
			ERR_FAIL_COND_V(skin->inverse_binds.size() != joints.size(), ERR_PARSE_ERROR);
		}

		for (int j = 0; j < joints.size(); j++) {
			const GLTFNodeIndex node = joints[j];
			ERR_FAIL_INDEX_V(node, state->nodes.size(), ERR_PARSE_ERROR);

			skin->joints.push_back(node);
			skin->joints_original.push_back(node);

			state->nodes.write[node]->joint = true;
		}

		if (d.has("name") && !String(d["name"]).is_empty()) {
			skin->set_name(d["name"]);
		} else {
			skin->set_name(vformat("skin_%s", itos(i)));
		}

		if (d.has("skeleton")) {
			skin->skin_root = d["skeleton"];
		}

		state->skins.push_back(skin);
	}

	for (GLTFSkinIndex i = 0; i < state->skins.size(); ++i) {
		Ref<GLTFSkin> skin = state->skins.write[i];

		// Expand the skin to capture all the extra non-joints that lie in between the actual joints,
		// and expand the hierarchy to ensure multi-rooted trees lie on the same height level
		ERR_FAIL_COND_V(_expand_skin(state, skin), ERR_PARSE_ERROR);
		ERR_FAIL_COND_V(_verify_skin(state, skin), ERR_PARSE_ERROR);
	}

	print_verbose("glTF: Total skins: " + itos(state->skins.size()));

	return OK;
}

Error GLTFDocument::_determine_skeletons(Ref<GLTFState> state) {
	// Using a disjoint set, we are going to potentially combine all skins that are actually branches
	// of a main skeleton, or treat skins defining the same set of nodes as ONE skeleton.
	// This is another unclear issue caused by the current glTF specification.

	DisjointSet<GLTFNodeIndex> skeleton_sets;

	for (GLTFSkinIndex skin_i = 0; skin_i < state->skins.size(); ++skin_i) {
		const Ref<GLTFSkin> skin = state->skins[skin_i];

		Vector<GLTFNodeIndex> all_skin_nodes;
		all_skin_nodes.append_array(skin->joints);
		all_skin_nodes.append_array(skin->non_joints);

		for (int i = 0; i < all_skin_nodes.size(); ++i) {
			const GLTFNodeIndex node_index = all_skin_nodes[i];
			const GLTFNodeIndex parent = state->nodes[node_index]->parent;
			skeleton_sets.insert(node_index);

			if (all_skin_nodes.find(parent) >= 0) {
				skeleton_sets.create_union(parent, node_index);
			}
		}

		// We are going to connect the separate skin subtrees in each skin together
		// so that the final roots are entire sets of valid skin trees
		for (int i = 1; i < skin->roots.size(); ++i) {
			skeleton_sets.create_union(skin->roots[0], skin->roots[i]);
		}
	}

	{ // attempt to joint all touching subsets (siblings/parent are part of another skin)
		Vector<GLTFNodeIndex> groups_representatives;
		skeleton_sets.get_representatives(groups_representatives);

		Vector<GLTFNodeIndex> highest_group_members;
		Vector<Vector<GLTFNodeIndex>> groups;
		for (int i = 0; i < groups_representatives.size(); ++i) {
			Vector<GLTFNodeIndex> group;
			skeleton_sets.get_members(group, groups_representatives[i]);
			highest_group_members.push_back(_find_highest_node(state, group));
			groups.push_back(group);
		}

		for (int i = 0; i < highest_group_members.size(); ++i) {
			const GLTFNodeIndex node_i = highest_group_members[i];

			// Attach any siblings together (this needs to be done n^2/2 times)
			for (int j = i + 1; j < highest_group_members.size(); ++j) {
				const GLTFNodeIndex node_j = highest_group_members[j];

				// Even if they are siblings under the root! :)
				if (state->nodes[node_i]->parent == state->nodes[node_j]->parent) {
					skeleton_sets.create_union(node_i, node_j);
				}
			}

			// Attach any parenting going on together (we need to do this n^2 times)
			const GLTFNodeIndex node_i_parent = state->nodes[node_i]->parent;
			if (node_i_parent >= 0) {
				for (int j = 0; j < groups.size() && i != j; ++j) {
					const Vector<GLTFNodeIndex> &group = groups[j];

					if (group.find(node_i_parent) >= 0) {
						const GLTFNodeIndex node_j = highest_group_members[j];
						skeleton_sets.create_union(node_i, node_j);
					}
				}
			}
		}
	}

	// At this point, the skeleton groups should be finalized
	Vector<GLTFNodeIndex> skeleton_owners;
	skeleton_sets.get_representatives(skeleton_owners);

	// Mark all the skins actual skeletons, after we have merged them
	for (GLTFSkeletonIndex skel_i = 0; skel_i < skeleton_owners.size(); ++skel_i) {
		const GLTFNodeIndex skeleton_owner = skeleton_owners[skel_i];
		Ref<GLTFSkeleton> skeleton;
		skeleton.instantiate();

		Vector<GLTFNodeIndex> skeleton_nodes;
		skeleton_sets.get_members(skeleton_nodes, skeleton_owner);

		for (GLTFSkinIndex skin_i = 0; skin_i < state->skins.size(); ++skin_i) {
			Ref<GLTFSkin> skin = state->skins.write[skin_i];

			// If any of the the skeletons nodes exist in a skin, that skin now maps to the skeleton
			for (int i = 0; i < skeleton_nodes.size(); ++i) {
				GLTFNodeIndex skel_node_i = skeleton_nodes[i];
				if (skin->joints.find(skel_node_i) >= 0 || skin->non_joints.find(skel_node_i) >= 0) {
					skin->skeleton = skel_i;
					continue;
				}
			}
		}

		Vector<GLTFNodeIndex> non_joints;
		for (int i = 0; i < skeleton_nodes.size(); ++i) {
			const GLTFNodeIndex node_i = skeleton_nodes[i];

			if (state->nodes[node_i]->joint) {
				skeleton->joints.push_back(node_i);
			} else {
				non_joints.push_back(node_i);
			}
		}

		state->skeletons.push_back(skeleton);

		_reparent_non_joint_skeleton_subtrees(state, state->skeletons.write[skel_i], non_joints);
	}

	for (GLTFSkeletonIndex skel_i = 0; skel_i < state->skeletons.size(); ++skel_i) {
		Ref<GLTFSkeleton> skeleton = state->skeletons.write[skel_i];

		for (int i = 0; i < skeleton->joints.size(); ++i) {
			const GLTFNodeIndex node_i = skeleton->joints[i];
			Ref<GLTFNode> node = state->nodes[node_i];

			ERR_FAIL_COND_V(!node->joint, ERR_PARSE_ERROR);
			ERR_FAIL_COND_V(node->skeleton >= 0, ERR_PARSE_ERROR);
			node->skeleton = skel_i;
		}

		ERR_FAIL_COND_V(_determine_skeleton_roots(state, skel_i), ERR_PARSE_ERROR);
	}

	return OK;
}

Error GLTFDocument::_reparent_non_joint_skeleton_subtrees(Ref<GLTFState> state, Ref<GLTFSkeleton> skeleton, const Vector<GLTFNodeIndex> &non_joints) {
	DisjointSet<GLTFNodeIndex> subtree_set;

	// Populate the disjoint set with ONLY non joints that are in the skeleton hierarchy (non_joints vector)
	// This way we can find any joints that lie in between joints, as the current glTF specification
	// mentions nothing about non-joints being in between joints of the same skin. Hopefully one day we
	// can remove this code.

	// skinD depicted here explains this issue:
	// https://github.com/KhronosGroup/glTF-Asset-Generator/blob/master/Output/Positive/Animation_Skin

	for (int i = 0; i < non_joints.size(); ++i) {
		const GLTFNodeIndex node_i = non_joints[i];

		subtree_set.insert(node_i);

		const GLTFNodeIndex parent_i = state->nodes[node_i]->parent;
		if (parent_i >= 0 && non_joints.find(parent_i) >= 0 && !state->nodes[parent_i]->joint) {
			subtree_set.create_union(parent_i, node_i);
		}
	}

	// Find all the non joint subtrees and re-parent them to a new "fake" joint

	Vector<GLTFNodeIndex> non_joint_subtree_roots;
	subtree_set.get_representatives(non_joint_subtree_roots);

	for (int root_i = 0; root_i < non_joint_subtree_roots.size(); ++root_i) {
		const GLTFNodeIndex subtree_root = non_joint_subtree_roots[root_i];

		Vector<GLTFNodeIndex> subtree_nodes;
		subtree_set.get_members(subtree_nodes, subtree_root);

		for (int subtree_i = 0; subtree_i < subtree_nodes.size(); ++subtree_i) {
			Ref<GLTFNode> node = state->nodes[subtree_nodes[subtree_i]];
			node->joint = true;
			// Add the joint to the skeletons joints
			skeleton->joints.push_back(subtree_nodes[subtree_i]);
		}
	}

	return OK;
}

Error GLTFDocument::_determine_skeleton_roots(Ref<GLTFState> state, const GLTFSkeletonIndex skel_i) {
	DisjointSet<GLTFNodeIndex> disjoint_set;

	for (GLTFNodeIndex i = 0; i < state->nodes.size(); ++i) {
		const Ref<GLTFNode> node = state->nodes[i];

		if (node->skeleton != skel_i) {
			continue;
		}

		disjoint_set.insert(i);

		if (node->parent >= 0 && state->nodes[node->parent]->skeleton == skel_i) {
			disjoint_set.create_union(node->parent, i);
		}
	}

	Ref<GLTFSkeleton> skeleton = state->skeletons.write[skel_i];

	Vector<GLTFNodeIndex> owners;
	disjoint_set.get_representatives(owners);

	Vector<GLTFNodeIndex> roots;

	for (int i = 0; i < owners.size(); ++i) {
		Vector<GLTFNodeIndex> set;
		disjoint_set.get_members(set, owners[i]);
		const GLTFNodeIndex root = _find_highest_node(state, set);
		ERR_FAIL_COND_V(root < 0, FAILED);
		roots.push_back(root);
	}

	roots.sort();

	skeleton->roots = roots;

	if (roots.size() == 0) {
		return FAILED;
	} else if (roots.size() == 1) {
		return OK;
	}

	// Check that the subtrees have the same parent root
	const GLTFNodeIndex parent = state->nodes[roots[0]]->parent;
	for (int i = 1; i < roots.size(); ++i) {
		if (state->nodes[roots[i]]->parent != parent) {
			return FAILED;
		}
	}

	return OK;
}

Error GLTFDocument::_create_skeletons(Ref<GLTFState> state) {
	for (GLTFSkeletonIndex skel_i = 0; skel_i < state->skeletons.size(); ++skel_i) {
		Ref<GLTFSkeleton> gltf_skeleton = state->skeletons.write[skel_i];

		Skeleton3D *skeleton = memnew(Skeleton3D);
		gltf_skeleton->godot_skeleton = skeleton;
		state->skeleton3d_to_gltf_skeleton[skeleton->get_instance_id()] = skel_i;

		// Make a unique name, no gltf node represents this skeleton
		skeleton->set_name(_gen_unique_name(state, "Skeleton3D"));

		List<GLTFNodeIndex> bones;

		for (int i = 0; i < gltf_skeleton->roots.size(); ++i) {
			bones.push_back(gltf_skeleton->roots[i]);
		}

		// Make the skeleton creation deterministic by going through the roots in
		// a sorted order, and DEPTH FIRST
		bones.sort();

		while (!bones.is_empty()) {
			const GLTFNodeIndex node_i = bones.front()->get();
			bones.pop_front();

			Ref<GLTFNode> node = state->nodes[node_i];
			ERR_FAIL_COND_V(node->skeleton != skel_i, FAILED);

			{ // Add all child nodes to the stack (deterministically)
				Vector<GLTFNodeIndex> child_nodes;
				for (int i = 0; i < node->children.size(); ++i) {
					const GLTFNodeIndex child_i = node->children[i];
					if (state->nodes[child_i]->skeleton == skel_i) {
						child_nodes.push_back(child_i);
					}
				}

				// Depth first insertion
				child_nodes.sort();
				for (int i = child_nodes.size() - 1; i >= 0; --i) {
					bones.push_front(child_nodes[i]);
				}
			}

			const int bone_index = skeleton->get_bone_count();

			if (node->get_name().is_empty()) {
				node->set_name("bone");
			}

			node->set_name(_gen_unique_bone_name(state, skel_i, node->get_name()));

			skeleton->add_bone(node->get_name());
			skeleton->set_bone_rest(bone_index, node->xform);
			skeleton->set_bone_pose_position(bone_index, node->position);
			skeleton->set_bone_pose_rotation(bone_index, node->rotation.normalized());
			skeleton->set_bone_pose_scale(bone_index, node->scale);

			if (node->parent >= 0 && state->nodes[node->parent]->skeleton == skel_i) {
				const int bone_parent = skeleton->find_bone(state->nodes[node->parent]->get_name());
				ERR_FAIL_COND_V(bone_parent < 0, FAILED);
				skeleton->set_bone_parent(bone_index, skeleton->find_bone(state->nodes[node->parent]->get_name()));
			}

			state->scene_nodes.insert(node_i, skeleton);
		}
	}

	ERR_FAIL_COND_V(_map_skin_joints_indices_to_skeleton_bone_indices(state), ERR_PARSE_ERROR);

	return OK;
}

Error GLTFDocument::_map_skin_joints_indices_to_skeleton_bone_indices(Ref<GLTFState> state) {
	for (GLTFSkinIndex skin_i = 0; skin_i < state->skins.size(); ++skin_i) {
		Ref<GLTFSkin> skin = state->skins.write[skin_i];

		Ref<GLTFSkeleton> skeleton = state->skeletons[skin->skeleton];

		for (int joint_index = 0; joint_index < skin->joints_original.size(); ++joint_index) {
			const GLTFNodeIndex node_i = skin->joints_original[joint_index];
			const Ref<GLTFNode> node = state->nodes[node_i];

			const int bone_index = skeleton->godot_skeleton->find_bone(node->get_name());
			ERR_FAIL_COND_V(bone_index < 0, FAILED);

			skin->joint_i_to_bone_i.insert(joint_index, bone_index);
		}
	}

	return OK;
}

Error GLTFDocument::_serialize_skins(Ref<GLTFState> state) {
	_remove_duplicate_skins(state);
	Array json_skins;
	for (int skin_i = 0; skin_i < state->skins.size(); skin_i++) {
		Ref<GLTFSkin> gltf_skin = state->skins[skin_i];
		Dictionary json_skin;
		json_skin["inverseBindMatrices"] = _encode_accessor_as_xform(state, gltf_skin->inverse_binds, false);
		json_skin["joints"] = gltf_skin->get_joints();
		json_skin["name"] = gltf_skin->get_name();
		json_skins.push_back(json_skin);
	}
	if (!state->skins.size()) {
		return OK;
	}

	state->json["skins"] = json_skins;
	return OK;
}

Error GLTFDocument::_create_skins(Ref<GLTFState> state) {
	for (GLTFSkinIndex skin_i = 0; skin_i < state->skins.size(); ++skin_i) {
		Ref<GLTFSkin> gltf_skin = state->skins.write[skin_i];

		Ref<Skin> skin;
		skin.instantiate();

		// Some skins don't have IBM's! What absolute monsters!
		const bool has_ibms = !gltf_skin->inverse_binds.is_empty();

		for (int joint_i = 0; joint_i < gltf_skin->joints_original.size(); ++joint_i) {
			GLTFNodeIndex node = gltf_skin->joints_original[joint_i];
			String bone_name = state->nodes[node]->get_name();

			Transform3D xform;
			if (has_ibms) {
				xform = gltf_skin->inverse_binds[joint_i];
			}

			if (state->use_named_skin_binds) {
				skin->add_named_bind(bone_name, xform);
			} else {
				int32_t bone_i = gltf_skin->joint_i_to_bone_i[joint_i];
				skin->add_bind(bone_i, xform);
			}
		}

		gltf_skin->godot_skin = skin;
	}

	// Purge the duplicates!
	_remove_duplicate_skins(state);

	// Create unique names now, after removing duplicates
	for (GLTFSkinIndex skin_i = 0; skin_i < state->skins.size(); ++skin_i) {
		Ref<Skin> skin = state->skins.write[skin_i]->godot_skin;
		if (skin->get_name().is_empty()) {
			// Make a unique name, no gltf node represents this skin
			skin->set_name(_gen_unique_name(state, "Skin"));
		}
	}

	return OK;
}

bool GLTFDocument::_skins_are_same(const Ref<Skin> skin_a, const Ref<Skin> skin_b) {
	if (skin_a->get_bind_count() != skin_b->get_bind_count()) {
		return false;
	}

	for (int i = 0; i < skin_a->get_bind_count(); ++i) {
		if (skin_a->get_bind_bone(i) != skin_b->get_bind_bone(i)) {
			return false;
		}
		if (skin_a->get_bind_name(i) != skin_b->get_bind_name(i)) {
			return false;
		}

		Transform3D a_xform = skin_a->get_bind_pose(i);
		Transform3D b_xform = skin_b->get_bind_pose(i);

		if (a_xform != b_xform) {
			return false;
		}
	}

	return true;
}

void GLTFDocument::_remove_duplicate_skins(Ref<GLTFState> state) {
	for (int i = 0; i < state->skins.size(); ++i) {
		for (int j = i + 1; j < state->skins.size(); ++j) {
			const Ref<Skin> skin_i = state->skins[i]->godot_skin;
			const Ref<Skin> skin_j = state->skins[j]->godot_skin;

			if (_skins_are_same(skin_i, skin_j)) {
				// replace it and delete the old
				state->skins.write[j]->godot_skin = skin_i;
			}
		}
	}
}

Error GLTFDocument::_serialize_lights(Ref<GLTFState> state) {
	Array lights;
	for (GLTFLightIndex i = 0; i < state->lights.size(); i++) {
		Dictionary d;
		Ref<GLTFLight> light = state->lights[i];
		Array color;
		color.resize(3);
		color[0] = light->color.r;
		color[1] = light->color.g;
		color[2] = light->color.b;
		d["color"] = color;
		d["type"] = light->light_type;
		if (light->light_type == "spot") {
			Dictionary s;
			float inner_cone_angle = light->inner_cone_angle;
			s["innerConeAngle"] = inner_cone_angle;
			float outer_cone_angle = light->outer_cone_angle;
			s["outerConeAngle"] = outer_cone_angle;
			d["spot"] = s;
		}
		float intensity = light->intensity;
		d["intensity"] = intensity;
		float range = light->range;
		d["range"] = range;
		lights.push_back(d);
	}

	if (!state->lights.size()) {
		return OK;
	}

	Dictionary extensions;
	if (state->json.has("extensions")) {
		extensions = state->json["extensions"];
	} else {
		state->json["extensions"] = extensions;
	}
	Dictionary lights_punctual;
	extensions["KHR_lights_punctual"] = lights_punctual;
	lights_punctual["lights"] = lights;

	print_verbose("glTF: Total lights: " + itos(state->lights.size()));

	return OK;
}

Error GLTFDocument::_serialize_cameras(Ref<GLTFState> state) {
	Array cameras;
	cameras.resize(state->cameras.size());
	for (GLTFCameraIndex i = 0; i < state->cameras.size(); i++) {
		Dictionary d;

		Ref<GLTFCamera> camera = state->cameras[i];

		if (camera->get_perspective() == false) {
			Dictionary og;
			og["ymag"] = Math::deg2rad(camera->get_fov_size());
			og["xmag"] = Math::deg2rad(camera->get_fov_size());
			og["zfar"] = camera->get_depth_far();
			og["znear"] = camera->get_depth_near();
			d["orthographic"] = og;
			d["type"] = "orthographic";
		} else if (camera->get_perspective()) {
			Dictionary ppt;
			// GLTF spec is in radians, Godot's camera is in degrees.
			ppt["yfov"] = Math::deg2rad(camera->get_fov_size());
			ppt["zfar"] = camera->get_depth_far();
			ppt["znear"] = camera->get_depth_near();
			d["perspective"] = ppt;
			d["type"] = "perspective";
		}
		cameras[i] = d;
	}

	if (!state->cameras.size()) {
		return OK;
	}

	state->json["cameras"] = cameras;

	print_verbose("glTF: Total cameras: " + itos(state->cameras.size()));

	return OK;
}

Error GLTFDocument::_parse_lights(Ref<GLTFState> state) {
	if (!state->json.has("extensions")) {
		return OK;
	}
	Dictionary extensions = state->json["extensions"];
	if (!extensions.has("KHR_lights_punctual")) {
		return OK;
	}
	Dictionary lights_punctual = extensions["KHR_lights_punctual"];
	if (!lights_punctual.has("lights")) {
		return OK;
	}

	const Array &lights = lights_punctual["lights"];

	for (GLTFLightIndex light_i = 0; light_i < lights.size(); light_i++) {
		const Dictionary &d = lights[light_i];

		Ref<GLTFLight> light;
		light.instantiate();
		ERR_FAIL_COND_V(!d.has("type"), ERR_PARSE_ERROR);
		const String &type = d["type"];
		light->light_type = type;

		if (d.has("color")) {
			const Array &arr = d["color"];
			ERR_FAIL_COND_V(arr.size() != 3, ERR_PARSE_ERROR);
			const Color c = Color(arr[0], arr[1], arr[2]).to_srgb();
			light->color = c;
		}
		if (d.has("intensity")) {
			light->intensity = d["intensity"];
		}
		if (d.has("range")) {
			light->range = d["range"];
		}
		if (type == "spot") {
			const Dictionary &spot = d["spot"];
			light->inner_cone_angle = spot["innerConeAngle"];
			light->outer_cone_angle = spot["outerConeAngle"];
			ERR_CONTINUE_MSG(light->inner_cone_angle >= light->outer_cone_angle, "The inner angle must be smaller than the outer angle.");
		} else if (type != "point" && type != "directional") {
			ERR_CONTINUE_MSG(true, "Light type is unknown.");
		}

		state->lights.push_back(light);
	}

	print_verbose("glTF: Total lights: " + itos(state->lights.size()));

	return OK;
}

Error GLTFDocument::_parse_cameras(Ref<GLTFState> state) {
	if (!state->json.has("cameras")) {
		return OK;
	}

	const Array cameras = state->json["cameras"];

	for (GLTFCameraIndex i = 0; i < cameras.size(); i++) {
		const Dictionary &d = cameras[i];

		Ref<GLTFCamera> camera;
		camera.instantiate();
		ERR_FAIL_COND_V(!d.has("type"), ERR_PARSE_ERROR);
		const String &type = d["type"];
		if (type == "orthographic") {
			camera->set_perspective(false);
			if (d.has("orthographic")) {
				const Dictionary &og = d["orthographic"];
				// GLTF spec is in radians, Godot's camera is in degrees.
				camera->set_fov_size(Math::rad2deg(real_t(og["ymag"])));
				camera->set_depth_far(og["zfar"]);
				camera->set_depth_near(og["znear"]);
			} else {
				camera->set_fov_size(10);
			}
		} else if (type == "perspective") {
			camera->set_perspective(true);
			if (d.has("perspective")) {
				const Dictionary &ppt = d["perspective"];
				// GLTF spec is in radians, Godot's camera is in degrees.
				camera->set_fov_size(Math::rad2deg(real_t(ppt["yfov"])));
				camera->set_depth_far(ppt["zfar"]);
				camera->set_depth_near(ppt["znear"]);
			} else {
				camera->set_fov_size(10);
			}
		} else {
			ERR_FAIL_V_MSG(ERR_PARSE_ERROR, "Camera3D should be in 'orthographic' or 'perspective'");
		}

		state->cameras.push_back(camera);
	}

	print_verbose("glTF: Total cameras: " + itos(state->cameras.size()));

	return OK;
}

String GLTFDocument::interpolation_to_string(const GLTFAnimation::Interpolation p_interp) {
	String interp = "LINEAR";
	if (p_interp == GLTFAnimation::INTERP_STEP) {
		interp = "STEP";
	} else if (p_interp == GLTFAnimation::INTERP_LINEAR) {
		interp = "LINEAR";
	} else if (p_interp == GLTFAnimation::INTERP_CATMULLROMSPLINE) {
		interp = "CATMULLROMSPLINE";
	} else if (p_interp == GLTFAnimation::INTERP_CUBIC_SPLINE) {
		interp = "CUBICSPLINE";
	}

	return interp;
}

Error GLTFDocument::_serialize_animations(Ref<GLTFState> state) {
	if (!state->animation_players.size()) {
		return OK;
	}
	for (int32_t player_i = 0; player_i < state->animation_players.size(); player_i++) {
		List<StringName> animation_names;
		AnimationPlayer *animation_player = state->animation_players[player_i];
		animation_player->get_animation_list(&animation_names);
		if (animation_names.size()) {
			for (int animation_name_i = 0; animation_name_i < animation_names.size(); animation_name_i++) {
				_convert_animation(state, animation_player, animation_names[animation_name_i]);
			}
		}
	}
	Array animations;
	for (GLTFAnimationIndex animation_i = 0; animation_i < state->animations.size(); animation_i++) {
		Dictionary d;
		Ref<GLTFAnimation> gltf_animation = state->animations[animation_i];
		if (!gltf_animation->get_tracks().size()) {
			continue;
		}

		if (!gltf_animation->get_name().is_empty()) {
			d["name"] = gltf_animation->get_name();
		}
		Array channels;
		Array samplers;

		for (KeyValue<int, GLTFAnimation::Track> &track_i : gltf_animation->get_tracks()) {
			GLTFAnimation::Track track = track_i.value;
			if (track.position_track.times.size()) {
				Dictionary t;
				t["sampler"] = samplers.size();
				Dictionary s;

				s["interpolation"] = interpolation_to_string(track.position_track.interpolation);
				Vector<real_t> times = Variant(track.position_track.times);
				s["input"] = _encode_accessor_as_floats(state, times, false);
				Vector<Vector3> values = Variant(track.position_track.values);
				s["output"] = _encode_accessor_as_vec3(state, values, false);

				samplers.push_back(s);

				Dictionary target;
				target["path"] = "translation";
				target["node"] = track_i.key;

				t["target"] = target;
				channels.push_back(t);
			}
			if (track.rotation_track.times.size()) {
				Dictionary t;
				t["sampler"] = samplers.size();
				Dictionary s;

				s["interpolation"] = interpolation_to_string(track.rotation_track.interpolation);
				Vector<real_t> times = Variant(track.rotation_track.times);
				s["input"] = _encode_accessor_as_floats(state, times, false);
				Vector<Quaternion> values = track.rotation_track.values;
				s["output"] = _encode_accessor_as_quaternions(state, values, false);

				samplers.push_back(s);

				Dictionary target;
				target["path"] = "rotation";
				target["node"] = track_i.key;

				t["target"] = target;
				channels.push_back(t);
			}
			if (track.scale_track.times.size()) {
				Dictionary t;
				t["sampler"] = samplers.size();
				Dictionary s;

				s["interpolation"] = interpolation_to_string(track.scale_track.interpolation);
				Vector<real_t> times = Variant(track.scale_track.times);
				s["input"] = _encode_accessor_as_floats(state, times, false);
				Vector<Vector3> values = Variant(track.scale_track.values);
				s["output"] = _encode_accessor_as_vec3(state, values, false);

				samplers.push_back(s);

				Dictionary target;
				target["path"] = "scale";
				target["node"] = track_i.key;

				t["target"] = target;
				channels.push_back(t);
			}
			if (track.weight_tracks.size()) {
				double length = 0.0f;

				for (int32_t track_idx = 0; track_idx < track.weight_tracks.size(); track_idx++) {
					int32_t last_time_index = track.weight_tracks[track_idx].times.size() - 1;
					length = MAX(length, track.weight_tracks[track_idx].times[last_time_index]);
				}

				Dictionary t;
				t["sampler"] = samplers.size();
				Dictionary s;
				Vector<real_t> times;
				const double increment = 1.0 / BAKE_FPS;
				{
					double time = 0.0;
					bool last = false;
					while (true) {
						times.push_back(time);
						if (last) {
							break;
						}
						time += increment;
						if (time >= length) {
							last = true;
							time = length;
						}
					}
				}

				for (int32_t track_idx = 0; track_idx < track.weight_tracks.size(); track_idx++) {
					double time = 0.0;
					bool last = false;
					Vector<real_t> weight_track;
					while (true) {
						float weight = _interpolate_track<real_t>(track.weight_tracks[track_idx].times,
								track.weight_tracks[track_idx].values,
								time,
								track.weight_tracks[track_idx].interpolation);
						weight_track.push_back(weight);
						if (last) {
							break;
						}
						time += increment;
						if (time >= length) {
							last = true;
							time = length;
						}
					}
					track.weight_tracks.write[track_idx].times = times;
					track.weight_tracks.write[track_idx].values = weight_track;
				}

				Vector<real_t> all_track_times = times;
				Vector<real_t> all_track_values;
				int32_t values_size = track.weight_tracks[0].values.size();
				int32_t weight_tracks_size = track.weight_tracks.size();
				all_track_values.resize(weight_tracks_size * values_size);
				for (int k = 0; k < track.weight_tracks.size(); k++) {
					Vector<real_t> wdata = track.weight_tracks[k].values;
					for (int l = 0; l < wdata.size(); l++) {
						int32_t index = l * weight_tracks_size + k;
						ERR_BREAK(index >= all_track_values.size());
						all_track_values.write[index] = wdata.write[l];
					}
				}

				s["interpolation"] = interpolation_to_string(track.weight_tracks[track.weight_tracks.size() - 1].interpolation);
				s["input"] = _encode_accessor_as_floats(state, all_track_times, false);
				s["output"] = _encode_accessor_as_floats(state, all_track_values, false);

				samplers.push_back(s);

				Dictionary target;
				target["path"] = "weights";
				target["node"] = track_i.key;

				t["target"] = target;
				channels.push_back(t);
			}
		}
		if (channels.size() && samplers.size()) {
			d["channels"] = channels;
			d["samplers"] = samplers;
			animations.push_back(d);
		}
	}

	if (!animations.size()) {
		return OK;
	}
	state->json["animations"] = animations;

	print_verbose("glTF: Total animations '" + itos(state->animations.size()) + "'.");

	return OK;
}

Error GLTFDocument::_parse_animations(Ref<GLTFState> state) {
	if (!state->json.has("animations")) {
		return OK;
	}

	const Array &animations = state->json["animations"];

	for (GLTFAnimationIndex i = 0; i < animations.size(); i++) {
		const Dictionary &d = animations[i];

		Ref<GLTFAnimation> animation;
		animation.instantiate();

		if (!d.has("channels") || !d.has("samplers")) {
			continue;
		}

		Array channels = d["channels"];
		Array samplers = d["samplers"];

		if (d.has("name")) {
			const String name = d["name"];
			if (name.begins_with("loop") || name.ends_with("loop") || name.begins_with("cycle") || name.ends_with("cycle")) {
				animation->set_loop(true);
			}
			animation->set_name(_gen_unique_animation_name(state, name));
		}

		for (int j = 0; j < channels.size(); j++) {
			const Dictionary &c = channels[j];
			if (!c.has("target")) {
				continue;
			}

			const Dictionary &t = c["target"];
			if (!t.has("node") || !t.has("path")) {
				continue;
			}

			ERR_FAIL_COND_V(!c.has("sampler"), ERR_PARSE_ERROR);
			const int sampler = c["sampler"];
			ERR_FAIL_INDEX_V(sampler, samplers.size(), ERR_PARSE_ERROR);

			GLTFNodeIndex node = t["node"];
			String path = t["path"];

			ERR_FAIL_INDEX_V(node, state->nodes.size(), ERR_PARSE_ERROR);

			GLTFAnimation::Track *track = nullptr;

			if (!animation->get_tracks().has(node)) {
				animation->get_tracks()[node] = GLTFAnimation::Track();
			}

			track = &animation->get_tracks()[node];

			const Dictionary &s = samplers[sampler];

			ERR_FAIL_COND_V(!s.has("input"), ERR_PARSE_ERROR);
			ERR_FAIL_COND_V(!s.has("output"), ERR_PARSE_ERROR);

			const int input = s["input"];
			const int output = s["output"];

			GLTFAnimation::Interpolation interp = GLTFAnimation::INTERP_LINEAR;
			int output_count = 1;
			if (s.has("interpolation")) {
				const String &in = s["interpolation"];
				if (in == "STEP") {
					interp = GLTFAnimation::INTERP_STEP;
				} else if (in == "LINEAR") {
					interp = GLTFAnimation::INTERP_LINEAR;
				} else if (in == "CATMULLROMSPLINE") {
					interp = GLTFAnimation::INTERP_CATMULLROMSPLINE;
					output_count = 3;
				} else if (in == "CUBICSPLINE") {
					interp = GLTFAnimation::INTERP_CUBIC_SPLINE;
					output_count = 3;
				}
			}

			const Vector<float> times = _decode_accessor_as_floats(state, input, false);
			if (path == "translation") {
				const Vector<Vector3> positions = _decode_accessor_as_vec3(state, output, false);
				track->position_track.interpolation = interp;
				track->position_track.times = Variant(times); //convert via variant
				track->position_track.values = Variant(positions); //convert via variant
			} else if (path == "rotation") {
				const Vector<Quaternion> rotations = _decode_accessor_as_quaternion(state, output, false);
				track->rotation_track.interpolation = interp;
				track->rotation_track.times = Variant(times); //convert via variant
				track->rotation_track.values = rotations;
			} else if (path == "scale") {
				const Vector<Vector3> scales = _decode_accessor_as_vec3(state, output, false);
				track->scale_track.interpolation = interp;
				track->scale_track.times = Variant(times); //convert via variant
				track->scale_track.values = Variant(scales); //convert via variant
			} else if (path == "weights") {
				const Vector<float> weights = _decode_accessor_as_floats(state, output, false);

				ERR_FAIL_INDEX_V(state->nodes[node]->mesh, state->meshes.size(), ERR_PARSE_ERROR);
				Ref<GLTFMesh> mesh = state->meshes[state->nodes[node]->mesh];
				ERR_CONTINUE(!mesh->get_blend_weights().size());
				const int wc = mesh->get_blend_weights().size();

				track->weight_tracks.resize(wc);

				const int expected_value_count = times.size() * output_count * wc;
				ERR_CONTINUE_MSG(weights.size() != expected_value_count, "Invalid weight data, expected " + itos(expected_value_count) + " weight values, got " + itos(weights.size()) + " instead.");

				const int wlen = weights.size() / wc;
				for (int k = 0; k < wc; k++) { //separate tracks, having them together is not such a good idea
					GLTFAnimation::Channel<real_t> cf;
					cf.interpolation = interp;
					cf.times = Variant(times);
					Vector<real_t> wdata;
					wdata.resize(wlen);
					for (int l = 0; l < wlen; l++) {
						wdata.write[l] = weights[l * wc + k];
					}

					cf.values = wdata;
					track->weight_tracks.write[k] = cf;
				}
			} else {
				WARN_PRINT("Invalid path '" + path + "'.");
			}
		}

		state->animations.push_back(animation);
	}

	print_verbose("glTF: Total animations '" + itos(state->animations.size()) + "'.");

	return OK;
}

void GLTFDocument::_assign_scene_names(Ref<GLTFState> state) {
	for (int i = 0; i < state->nodes.size(); i++) {
		Ref<GLTFNode> n = state->nodes[i];

		// Any joints get unique names generated when the skeleton is made, unique to the skeleton
		if (n->skeleton >= 0) {
			continue;
		}

		if (n->get_name().is_empty()) {
			if (n->mesh >= 0) {
				n->set_name(_gen_unique_name(state, "Mesh"));
			} else if (n->camera >= 0) {
				n->set_name(_gen_unique_name(state, "Camera3D"));
			} else {
				n->set_name(_gen_unique_name(state, "Node"));
			}
		}

		n->set_name(_gen_unique_name(state, n->get_name()));
	}
}

BoneAttachment3D *GLTFDocument::_generate_bone_attachment(Ref<GLTFState> state, Skeleton3D *skeleton, const GLTFNodeIndex node_index, const GLTFNodeIndex bone_index) {
	Ref<GLTFNode> gltf_node = state->nodes[node_index];
	Ref<GLTFNode> bone_node = state->nodes[bone_index];
	BoneAttachment3D *bone_attachment = memnew(BoneAttachment3D);
	print_verbose("glTF: Creating bone attachment for: " + gltf_node->get_name());

	ERR_FAIL_COND_V(!bone_node->joint, nullptr);

	bone_attachment->set_bone_name(bone_node->get_name());

	return bone_attachment;
}

GLTFMeshIndex GLTFDocument::_convert_mesh_to_gltf(Ref<GLTFState> state, MeshInstance3D *p_mesh_instance) {
	ERR_FAIL_NULL_V(p_mesh_instance, -1);
	if (p_mesh_instance->get_mesh().is_null()) {
		return -1;
	}
	Ref<ImporterMesh> current_mesh;
	current_mesh.instantiate();
	Vector<float> blend_weights;
	{
		Ref<Mesh> import_mesh = p_mesh_instance->get_mesh();
		Ref<ArrayMesh> import_array_mesh = p_mesh_instance->get_mesh();
		if (import_mesh->get_blend_shape_count()) {
			ArrayMesh::BlendShapeMode shape_mode = ArrayMesh::BLEND_SHAPE_MODE_NORMALIZED;
			if (import_array_mesh.is_valid()) {
				shape_mode = import_array_mesh->get_blend_shape_mode();
			}
			current_mesh->set_blend_shape_mode(shape_mode);
			for (int morph_i = 0; morph_i < import_mesh->get_blend_shape_count(); morph_i++) {
				current_mesh->add_blend_shape(import_mesh->get_blend_shape_name(morph_i));
			}
		}
		for (int32_t surface_i = 0; surface_i < import_mesh->get_surface_count(); surface_i++) {
			Array array = import_mesh->surface_get_arrays(surface_i);
			Ref<Material> mat = import_mesh->surface_get_material(surface_i);
			String mat_name;
			if (mat.is_valid()) {
				mat_name = mat->get_name();
			}
			current_mesh->add_surface(import_mesh->surface_get_primitive_type(surface_i),
					array, import_mesh->surface_get_blend_shape_arrays(surface_i), import_mesh->surface_get_lods(surface_i), mat,
					mat_name, import_mesh->surface_get_format(surface_i));
		}
		int32_t blend_count = import_mesh->get_blend_shape_count();
		blend_weights.resize(blend_count);
		for (int32_t blend_i = 0; blend_i < blend_count; blend_i++) {
			blend_weights.write[blend_i] = 0.0f;
		}
	}
	Ref<GLTFMesh> gltf_mesh;
	gltf_mesh.instantiate();
	Array instance_materials;
	for (int32_t surface_i = 0; surface_i < current_mesh->get_surface_count(); surface_i++) {
		Ref<Material> mat = current_mesh->get_surface_material(surface_i);
		if (p_mesh_instance->get_surface_override_material(surface_i).is_valid()) {
			mat = p_mesh_instance->get_surface_override_material(surface_i);
		}
		if (p_mesh_instance->get_material_override().is_valid()) {
			mat = p_mesh_instance->get_material_override();
		}
		instance_materials.append(mat);
	}
	gltf_mesh->set_instance_materials(instance_materials);
	gltf_mesh->set_mesh(current_mesh);
	gltf_mesh->set_blend_weights(blend_weights);
	GLTFMeshIndex mesh_i = state->meshes.size();
	state->meshes.push_back(gltf_mesh);
	return mesh_i;
}

ImporterMeshInstance3D *GLTFDocument::_generate_mesh_instance(Ref<GLTFState> state, const GLTFNodeIndex node_index) {
	Ref<GLTFNode> gltf_node = state->nodes[node_index];

	ERR_FAIL_INDEX_V(gltf_node->mesh, state->meshes.size(), nullptr);

	ImporterMeshInstance3D *mi = memnew(ImporterMeshInstance3D);
	print_verbose("glTF: Creating mesh for: " + gltf_node->get_name());

	Ref<GLTFMesh> mesh = state->meshes.write[gltf_node->mesh];
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

Node3D *GLTFDocument::_generate_light(Ref<GLTFState> state, const GLTFNodeIndex node_index) {
	Ref<GLTFNode> gltf_node = state->nodes[node_index];

	ERR_FAIL_INDEX_V(gltf_node->light, state->lights.size(), nullptr);

	print_verbose("glTF: Creating light for: " + gltf_node->get_name());

	Ref<GLTFLight> l = state->lights[gltf_node->light];

	float intensity = l->intensity;
	if (intensity > 10) {
		// GLTF spec has the default around 1, but Blender defaults lights to 100.
		// The only sane way to handle this is to check where it came from and
		// handle it accordingly. If it's over 10, it probably came from Blender.
		intensity /= 100;
	}

	if (l->light_type == "directional") {
		DirectionalLight3D *light = memnew(DirectionalLight3D);
		light->set_param(Light3D::PARAM_ENERGY, intensity);
		light->set_color(l->color);
		return light;
	}

	const float range = CLAMP(l->range, 0, 4096);
	// Doubling the range will double the effective brightness, so we need double attenuation (half brightness).
	// We want to have double intensity give double brightness, so we need half the attenuation.
	const float attenuation = range / (intensity * 2048);
	if (l->light_type == "point") {
		OmniLight3D *light = memnew(OmniLight3D);
		light->set_param(OmniLight3D::PARAM_ATTENUATION, attenuation);
		light->set_param(OmniLight3D::PARAM_RANGE, range);
		light->set_color(l->color);
		return light;
	}
	if (l->light_type == "spot") {
		SpotLight3D *light = memnew(SpotLight3D);
		light->set_param(SpotLight3D::PARAM_ATTENUATION, attenuation);
		light->set_param(SpotLight3D::PARAM_RANGE, range);
		light->set_param(SpotLight3D::PARAM_SPOT_ANGLE, Math::rad2deg(l->outer_cone_angle));
		light->set_color(l->color);

		// Line of best fit derived from guessing, see https://www.desmos.com/calculator/biiflubp8b
		// The points in desmos are not exact, except for (1, infinity).
		float angle_ratio = l->inner_cone_angle / l->outer_cone_angle;
		float angle_attenuation = 0.2 / (1 - angle_ratio) - 0.1;
		light->set_param(SpotLight3D::PARAM_SPOT_ATTENUATION, angle_attenuation);
		return light;
	}
	return memnew(Node3D);
}

Camera3D *GLTFDocument::_generate_camera(Ref<GLTFState> state, const GLTFNodeIndex node_index) {
	Ref<GLTFNode> gltf_node = state->nodes[node_index];

	ERR_FAIL_INDEX_V(gltf_node->camera, state->cameras.size(), nullptr);

	Camera3D *camera = memnew(Camera3D);
	print_verbose("glTF: Creating camera for: " + gltf_node->get_name());

	Ref<GLTFCamera> c = state->cameras[gltf_node->camera];
	if (c->get_perspective()) {
		camera->set_perspective(c->get_fov_size(), c->get_depth_near(), c->get_depth_far());
	} else {
		camera->set_orthogonal(c->get_fov_size(), c->get_depth_near(), c->get_depth_far());
	}

	return camera;
}

GLTFCameraIndex GLTFDocument::_convert_camera(Ref<GLTFState> state, Camera3D *p_camera) {
	print_verbose("glTF: Converting camera: " + p_camera->get_name());

	Ref<GLTFCamera> c;
	c.instantiate();

	if (p_camera->get_projection() == Camera3D::Projection::PROJECTION_PERSPECTIVE) {
		c->set_perspective(true);
	}
	c->set_fov_size(p_camera->get_fov());
	c->set_depth_far(p_camera->get_far());
	c->set_depth_near(p_camera->get_near());
	GLTFCameraIndex camera_index = state->cameras.size();
	state->cameras.push_back(c);
	return camera_index;
}

GLTFLightIndex GLTFDocument::_convert_light(Ref<GLTFState> state, Light3D *p_light) {
	print_verbose("glTF: Converting light: " + p_light->get_name());

	Ref<GLTFLight> l;
	l.instantiate();
	l->color = p_light->get_color();
	if (cast_to<DirectionalLight3D>(p_light)) {
		l->light_type = "directional";
		DirectionalLight3D *light = cast_to<DirectionalLight3D>(p_light);
		l->intensity = light->get_param(DirectionalLight3D::PARAM_ENERGY);
		l->range = FLT_MAX; // Range for directional lights is infinite in Godot.
	} else if (cast_to<OmniLight3D>(p_light)) {
		l->light_type = "point";
		OmniLight3D *light = cast_to<OmniLight3D>(p_light);
		l->range = light->get_param(OmniLight3D::PARAM_RANGE);
		float attenuation = p_light->get_param(OmniLight3D::PARAM_ATTENUATION);
		l->intensity = l->range / (attenuation * 2048);
	} else if (cast_to<SpotLight3D>(p_light)) {
		l->light_type = "spot";
		SpotLight3D *light = cast_to<SpotLight3D>(p_light);
		l->range = light->get_param(SpotLight3D::PARAM_RANGE);
		float attenuation = light->get_param(SpotLight3D::PARAM_ATTENUATION);
		l->intensity = l->range / (attenuation * 2048);
		l->outer_cone_angle = Math::deg2rad(light->get_param(SpotLight3D::PARAM_SPOT_ANGLE));

		// This equation is the inverse of the import equation (which has a desmos link).
		float angle_ratio = 1 - (0.2 / (0.1 + light->get_param(SpotLight3D::PARAM_SPOT_ATTENUATION)));
		angle_ratio = MAX(0, angle_ratio);
		l->inner_cone_angle = l->outer_cone_angle * angle_ratio;
	}

	GLTFLightIndex light_index = state->lights.size();
	state->lights.push_back(l);
	return light_index;
}

void GLTFDocument::_convert_spatial(Ref<GLTFState> state, Node3D *p_spatial, Ref<GLTFNode> p_node) {
	Transform3D xform = p_spatial->get_transform();
	p_node->scale = xform.basis.get_scale();
	p_node->rotation = xform.basis.get_rotation_quaternion();
	p_node->position = xform.origin;
}

Node3D *GLTFDocument::_generate_spatial(Ref<GLTFState> state, const GLTFNodeIndex node_index) {
	Ref<GLTFNode> gltf_node = state->nodes[node_index];

	Node3D *spatial = memnew(Node3D);
	print_verbose("glTF: Converting spatial: " + gltf_node->get_name());

	return spatial;
}
void GLTFDocument::_convert_scene_node(Ref<GLTFState> state, Node *p_current, const GLTFNodeIndex p_gltf_parent, const GLTFNodeIndex p_gltf_root) {
	bool retflag = true;
	_check_visibility(p_current, retflag);
	if (retflag) {
		return;
	}
	Ref<GLTFNode> gltf_node;
	gltf_node.instantiate();
	gltf_node->set_name(_gen_unique_name(state, p_current->get_name()));
	if (cast_to<Node3D>(p_current)) {
		Node3D *spatial = cast_to<Node3D>(p_current);
		_convert_spatial(state, spatial, gltf_node);
	}
	if (cast_to<MeshInstance3D>(p_current)) {
		MeshInstance3D *mi = cast_to<MeshInstance3D>(p_current);
		_convert_mesh_instance_to_gltf(mi, state, gltf_node);
	} else if (cast_to<BoneAttachment3D>(p_current)) {
		BoneAttachment3D *bone = cast_to<BoneAttachment3D>(p_current);
		_convert_bone_attachment_to_gltf(bone, state, p_gltf_parent, p_gltf_root, gltf_node);
		return;
	} else if (cast_to<Skeleton3D>(p_current)) {
		Skeleton3D *skel = cast_to<Skeleton3D>(p_current);
		_convert_skeleton_to_gltf(skel, state, p_gltf_parent, p_gltf_root, gltf_node);
		// We ignore the Godot Engine node that is the skeleton.
		return;
	} else if (cast_to<MultiMeshInstance3D>(p_current)) {
		MultiMeshInstance3D *multi = cast_to<MultiMeshInstance3D>(p_current);
		_convert_multi_mesh_instance_to_gltf(multi, p_gltf_parent, p_gltf_root, gltf_node, state);
#ifdef MODULE_CSG_ENABLED
	} else if (cast_to<CSGShape3D>(p_current)) {
		CSGShape3D *shape = cast_to<CSGShape3D>(p_current);
		if (shape->get_parent() && shape->is_root_shape()) {
			_convert_csg_shape_to_gltf(shape, p_gltf_parent, gltf_node, state);
		}
#endif // MODULE_CSG_ENABLED
#ifdef MODULE_GRIDMAP_ENABLED
	} else if (cast_to<GridMap>(p_current)) {
		GridMap *gridmap = Object::cast_to<GridMap>(p_current);
		_convert_grid_map_to_gltf(gridmap, p_gltf_parent, p_gltf_root, gltf_node, state);
#endif // MODULE_GRIDMAP_ENABLED
	} else if (cast_to<Camera3D>(p_current)) {
		Camera3D *camera = Object::cast_to<Camera3D>(p_current);
		_convert_camera_to_gltf(camera, state, gltf_node);
	} else if (cast_to<Light3D>(p_current)) {
		Light3D *light = Object::cast_to<Light3D>(p_current);
		_convert_light_to_gltf(light, state, gltf_node);
	} else if (cast_to<AnimationPlayer>(p_current)) {
		AnimationPlayer *animation_player = Object::cast_to<AnimationPlayer>(p_current);
		_convert_animation_player_to_gltf(animation_player, state, p_gltf_parent, p_gltf_root, gltf_node, p_current);
	}
	GLTFNodeIndex current_node_i = state->nodes.size();
	GLTFNodeIndex gltf_root = p_gltf_root;
	if (gltf_root == -1) {
		gltf_root = current_node_i;
		Array scenes;
		scenes.push_back(gltf_root);
		state->json["scene"] = scenes;
	}
	_create_gltf_node(state, p_current, current_node_i, p_gltf_parent, gltf_root, gltf_node);
	for (int node_i = 0; node_i < p_current->get_child_count(); node_i++) {
		_convert_scene_node(state, p_current->get_child(node_i), current_node_i, gltf_root);
	}
}

#ifdef MODULE_CSG_ENABLED
void GLTFDocument::_convert_csg_shape_to_gltf(CSGShape3D *p_current, GLTFNodeIndex p_gltf_parent, Ref<GLTFNode> gltf_node, Ref<GLTFState> state) {
	CSGShape3D *csg = p_current;
	csg->call("_update_shape");
	Array meshes = csg->get_meshes();
	if (meshes.size() != 2) {
		return;
	}
	Ref<Material> mat;
	if (csg->get_material_override().is_valid()) {
		mat = csg->get_material_override();
	}
	Ref<GLTFMesh> gltf_mesh;
	gltf_mesh.instantiate();
	Ref<ImporterMesh> array_mesh = csg->get_meshes()[1];
	gltf_mesh->set_mesh(array_mesh);
	GLTFMeshIndex mesh_i = state->meshes.size();
	state->meshes.push_back(gltf_mesh);
	gltf_node->mesh = mesh_i;
	gltf_node->xform = csg->get_meshes()[0];
	gltf_node->set_name(_gen_unique_name(state, csg->get_name()));
}
#endif // MODULE_CSG_ENABLED

void GLTFDocument::_create_gltf_node(Ref<GLTFState> state, Node *p_scene_parent, GLTFNodeIndex current_node_i,
		GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_gltf_node, Ref<GLTFNode> gltf_node) {
	state->scene_nodes.insert(current_node_i, p_scene_parent);
	state->nodes.push_back(gltf_node);
	ERR_FAIL_COND(current_node_i == p_parent_node_index);
	state->nodes.write[current_node_i]->parent = p_parent_node_index;
	if (p_parent_node_index == -1) {
		return;
	}
	state->nodes.write[p_parent_node_index]->children.push_back(current_node_i);
}

void GLTFDocument::_convert_animation_player_to_gltf(AnimationPlayer *animation_player, Ref<GLTFState> state, GLTFNodeIndex p_gltf_current, GLTFNodeIndex p_gltf_root_index, Ref<GLTFNode> p_gltf_node, Node *p_scene_parent) {
	ERR_FAIL_COND(!animation_player);
	state->animation_players.push_back(animation_player);
	print_verbose(String("glTF: Converting animation player: ") + animation_player->get_name());
}

void GLTFDocument::_check_visibility(Node *p_node, bool &retflag) {
	retflag = true;
	Node3D *spatial = Object::cast_to<Node3D>(p_node);
	Node2D *node_2d = Object::cast_to<Node2D>(p_node);
	if (node_2d && !node_2d->is_visible()) {
		return;
	}
	if (spatial && !spatial->is_visible()) {
		return;
	}
	retflag = false;
}

void GLTFDocument::_convert_camera_to_gltf(Camera3D *camera, Ref<GLTFState> state, Ref<GLTFNode> gltf_node) {
	ERR_FAIL_COND(!camera);
	GLTFCameraIndex camera_index = _convert_camera(state, camera);
	if (camera_index != -1) {
		gltf_node->camera = camera_index;
	}
}

void GLTFDocument::_convert_light_to_gltf(Light3D *light, Ref<GLTFState> state, Ref<GLTFNode> gltf_node) {
	ERR_FAIL_COND(!light);
	GLTFLightIndex light_index = _convert_light(state, light);
	if (light_index != -1) {
		gltf_node->light = light_index;
	}
}

#ifdef MODULE_GRIDMAP_ENABLED
void GLTFDocument::_convert_grid_map_to_gltf(GridMap *p_grid_map, GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_node_index, Ref<GLTFNode> gltf_node, Ref<GLTFState> state) {
	Array cells = p_grid_map->get_used_cells();
	for (int32_t k = 0; k < cells.size(); k++) {
		GLTFNode *new_gltf_node = memnew(GLTFNode);
		gltf_node->children.push_back(state->nodes.size());
		state->nodes.push_back(new_gltf_node);
		Vector3 cell_location = cells[k];
		int32_t cell = p_grid_map->get_cell_item(
				Vector3(cell_location.x, cell_location.y, cell_location.z));
		ImporterMeshInstance3D *import_mesh_node = memnew(ImporterMeshInstance3D);
		import_mesh_node->set_mesh(p_grid_map->get_mesh_library()->get_item_mesh(cell));
		Transform3D cell_xform;
		cell_xform.basis.set_orthogonal_index(
				p_grid_map->get_cell_item_orientation(
						Vector3(cell_location.x, cell_location.y, cell_location.z)));
		cell_xform.basis.scale(Vector3(p_grid_map->get_cell_scale(),
				p_grid_map->get_cell_scale(),
				p_grid_map->get_cell_scale()));
		cell_xform.set_origin(p_grid_map->map_to_world(
				Vector3(cell_location.x, cell_location.y, cell_location.z)));
		Ref<GLTFMesh> gltf_mesh;
		gltf_mesh.instantiate();
		gltf_mesh = import_mesh_node;
		new_gltf_node->mesh = state->meshes.size();
		state->meshes.push_back(gltf_mesh);
		new_gltf_node->xform = cell_xform * p_grid_map->get_transform();
		new_gltf_node->set_name(_gen_unique_name(state, p_grid_map->get_mesh_library()->get_item_name(cell)));
	}
}
#endif // MODULE_GRIDMAP_ENABLED

void GLTFDocument::_convert_multi_mesh_instance_to_gltf(
		MultiMeshInstance3D *p_multi_mesh_instance,
		GLTFNodeIndex p_parent_node_index,
		GLTFNodeIndex p_root_node_index,
		Ref<GLTFNode> gltf_node, Ref<GLTFState> state) {
	ERR_FAIL_COND(!p_multi_mesh_instance);
	Ref<MultiMesh> multi_mesh = p_multi_mesh_instance->get_multimesh();
	if (multi_mesh.is_null()) {
		return;
	}
	Ref<GLTFMesh> gltf_mesh;
	gltf_mesh.instantiate();
	Ref<Mesh> mesh = multi_mesh->get_mesh();
	if (mesh.is_null()) {
		return;
	}
	gltf_mesh->set_name(multi_mesh->get_name());
	Ref<ImporterMesh> importer_mesh;
	importer_mesh.instantiate();
	Ref<ArrayMesh> array_mesh = multi_mesh->get_mesh();
	if (array_mesh.is_valid()) {
		importer_mesh->set_blend_shape_mode(array_mesh->get_blend_shape_mode());
		for (int32_t blend_i = 0; blend_i < array_mesh->get_blend_shape_count(); blend_i++) {
			importer_mesh->add_blend_shape(array_mesh->get_blend_shape_name(blend_i));
		}
	}
	for (int32_t surface_i = 0; surface_i < mesh->get_surface_count(); surface_i++) {
		Ref<Material> mat = mesh->surface_get_material(surface_i);
		String material_name;
		if (mat.is_valid()) {
			material_name = mat->get_name();
		}
		Array blend_arrays;
		if (array_mesh.is_valid()) {
			blend_arrays = array_mesh->surface_get_blend_shape_arrays(surface_i);
		}
		importer_mesh->add_surface(mesh->surface_get_primitive_type(surface_i), mesh->surface_get_arrays(surface_i),
				blend_arrays, mesh->surface_get_lods(surface_i), mat, material_name, mesh->surface_get_format(surface_i));
	}
	gltf_mesh->set_mesh(importer_mesh);
	GLTFMeshIndex mesh_index = state->meshes.size();
	state->meshes.push_back(gltf_mesh);
	for (int32_t instance_i = 0; instance_i < multi_mesh->get_instance_count();
			instance_i++) {
		Transform3D transform;
		if (multi_mesh->get_transform_format() == MultiMesh::TRANSFORM_2D) {
			Transform2D xform_2d = multi_mesh->get_instance_transform_2d(instance_i);
			transform.origin =
					Vector3(xform_2d.get_origin().x, 0, xform_2d.get_origin().y);
			real_t rotation = xform_2d.get_rotation();
			Quaternion quaternion(Vector3(0, 1, 0), rotation);
			Size2 scale = xform_2d.get_scale();
			transform.basis.set_quaternion_scale(quaternion,
					Vector3(scale.x, 0, scale.y));
			transform = p_multi_mesh_instance->get_transform() * transform;
		} else if (multi_mesh->get_transform_format() == MultiMesh::TRANSFORM_3D) {
			transform = p_multi_mesh_instance->get_transform() *
					multi_mesh->get_instance_transform(instance_i);
		}
		Ref<GLTFNode> new_gltf_node;
		new_gltf_node.instantiate();
		new_gltf_node->mesh = mesh_index;
		new_gltf_node->xform = transform;
		new_gltf_node->set_name(_gen_unique_name(state, p_multi_mesh_instance->get_name()));
		gltf_node->children.push_back(state->nodes.size());
		state->nodes.push_back(new_gltf_node);
	}
}

void GLTFDocument::_convert_skeleton_to_gltf(Skeleton3D *p_skeleton3d, Ref<GLTFState> state, GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_node_index, Ref<GLTFNode> gltf_node) {
	Skeleton3D *skeleton = p_skeleton3d;
	Ref<GLTFSkeleton> gltf_skeleton;
	gltf_skeleton.instantiate();
	// GLTFSkeleton is only used to hold internal state data. It will not be written to the document.
	//
	gltf_skeleton->godot_skeleton = skeleton;
	GLTFSkeletonIndex skeleton_i = state->skeletons.size();
	state->skeleton3d_to_gltf_skeleton[skeleton->get_instance_id()] = skeleton_i;
	state->skeletons.push_back(gltf_skeleton);

	BoneId bone_count = skeleton->get_bone_count();
	for (BoneId bone_i = 0; bone_i < bone_count; bone_i++) {
		Ref<GLTFNode> joint_node;
		joint_node.instantiate();
		// Note that we cannot use _gen_unique_bone_name here, because glTF spec requires all node
		// names to be unique regardless of whether or not they are used as joints.
		joint_node->set_name(_gen_unique_name(state, skeleton->get_bone_name(bone_i)));
		Transform3D xform = skeleton->get_bone_pose(bone_i);
		joint_node->scale = xform.basis.get_scale();
		joint_node->rotation = xform.basis.get_rotation_quaternion();
		joint_node->position = xform.origin;
		joint_node->joint = true;
		GLTFNodeIndex current_node_i = state->nodes.size();
		state->scene_nodes.insert(current_node_i, skeleton);
		state->nodes.push_back(joint_node);

		gltf_skeleton->joints.push_back(current_node_i);
		if (skeleton->get_bone_parent(bone_i) == -1) {
			gltf_skeleton->roots.push_back(current_node_i);
		}
		gltf_skeleton->godot_bone_node.insert(bone_i, current_node_i);
	}
	for (BoneId bone_i = 0; bone_i < bone_count; bone_i++) {
		GLTFNodeIndex current_node_i = gltf_skeleton->godot_bone_node[bone_i];
		BoneId parent_bone_id = skeleton->get_bone_parent(bone_i);
		if (parent_bone_id == -1) {
			if (p_parent_node_index != -1) {
				state->nodes.write[current_node_i]->parent = p_parent_node_index;
				state->nodes.write[p_parent_node_index]->children.push_back(current_node_i);
			}
		} else {
			GLTFNodeIndex parent_node_i = gltf_skeleton->godot_bone_node[parent_bone_id];
			state->nodes.write[current_node_i]->parent = parent_node_i;
			state->nodes.write[parent_node_i]->children.push_back(current_node_i);
		}
	}
	// Remove placeholder skeleton3d node by not creating the gltf node
	// Skins are per mesh
	for (int node_i = 0; node_i < skeleton->get_child_count(); node_i++) {
		_convert_scene_node(state, skeleton->get_child(node_i), p_parent_node_index, p_root_node_index);
	}
}

void GLTFDocument::_convert_bone_attachment_to_gltf(BoneAttachment3D *p_bone_attachment, Ref<GLTFState> state, GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_node_index, Ref<GLTFNode> gltf_node) {
	Skeleton3D *skeleton;
	// Note that relative transforms to external skeletons and pose overrides are not supported.
	if (p_bone_attachment->get_use_external_skeleton()) {
		skeleton = cast_to<Skeleton3D>(p_bone_attachment->get_node_or_null(p_bone_attachment->get_external_skeleton()));
	} else {
		skeleton = cast_to<Skeleton3D>(p_bone_attachment->get_parent());
	}
	GLTFSkeletonIndex skel_gltf_i = -1;
	if (skeleton != nullptr && state->skeleton3d_to_gltf_skeleton.has(skeleton->get_instance_id())) {
		skel_gltf_i = state->skeleton3d_to_gltf_skeleton[skeleton->get_instance_id()];
	}
	int bone_idx = -1;
	if (skeleton != nullptr) {
		bone_idx = p_bone_attachment->get_bone_idx();
		if (bone_idx == -1) {
			bone_idx = skeleton->find_bone(p_bone_attachment->get_bone_name());
		}
	}
	GLTFNodeIndex par_node_index = p_parent_node_index;
	if (skeleton != nullptr && bone_idx != -1 && skel_gltf_i != -1) {
		Ref<GLTFSkeleton> gltf_skeleton = state->skeletons.write[skel_gltf_i];
		gltf_skeleton->bone_attachments.push_back(p_bone_attachment);
		par_node_index = gltf_skeleton->joints[bone_idx];
	}

	for (int node_i = 0; node_i < p_bone_attachment->get_child_count(); node_i++) {
		_convert_scene_node(state, p_bone_attachment->get_child(node_i), par_node_index, p_root_node_index);
	}
}

void GLTFDocument::_convert_mesh_instance_to_gltf(MeshInstance3D *p_scene_parent, Ref<GLTFState> state, Ref<GLTFNode> gltf_node) {
	GLTFMeshIndex gltf_mesh_index = _convert_mesh_to_gltf(state, p_scene_parent);
	if (gltf_mesh_index != -1) {
		gltf_node->mesh = gltf_mesh_index;
	}
}

void GLTFDocument::_generate_scene_node(Ref<GLTFState> state, Node *scene_parent, Node3D *scene_root, const GLTFNodeIndex node_index) {
	Ref<GLTFNode> gltf_node = state->nodes[node_index];

	if (gltf_node->skeleton >= 0) {
		_generate_skeleton_bone_node(state, scene_parent, scene_root, node_index);
		return;
	}

	Node3D *current_node = nullptr;

	// Is our parent a skeleton
	Skeleton3D *active_skeleton = Object::cast_to<Skeleton3D>(scene_parent);

	const bool non_bone_parented_to_skeleton = active_skeleton;

	// skinned meshes must not be placed in a bone attachment.
	if (non_bone_parented_to_skeleton && gltf_node->skin < 0) {
		// Bone Attachment - Parent Case
		BoneAttachment3D *bone_attachment = _generate_bone_attachment(state, active_skeleton, node_index, gltf_node->parent);

		scene_parent->add_child(bone_attachment, true);
		bone_attachment->set_owner(scene_root);

		// There is no gltf_node that represent this, so just directly create a unique name
		bone_attachment->set_name(_gen_unique_name(state, "BoneAttachment3D"));

		// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
		// and attach it to the bone_attachment
		scene_parent = bone_attachment;
	}
	if (gltf_node->mesh >= 0) {
		current_node = _generate_mesh_instance(state, node_index);
	} else if (gltf_node->camera >= 0) {
		current_node = _generate_camera(state, node_index);
	} else if (gltf_node->light >= 0) {
		current_node = _generate_light(state, node_index);
	}

	// We still have not managed to make a node.
	if (!current_node) {
		current_node = _generate_spatial(state, node_index);
	}
	scene_parent->add_child(current_node);
	if (current_node != scene_root) {
		current_node->set_owner(scene_root);
	}
	current_node->set_transform(gltf_node->xform);
	current_node->set_name(gltf_node->get_name());

	state->scene_nodes.insert(node_index, current_node);

	for (int i = 0; i < gltf_node->children.size(); ++i) {
		_generate_scene_node(state, current_node, scene_root, gltf_node->children[i]);
	}
}

void GLTFDocument::_generate_skeleton_bone_node(Ref<GLTFState> state, Node *scene_parent, Node3D *scene_root, const GLTFNodeIndex node_index) {
	Ref<GLTFNode> gltf_node = state->nodes[node_index];

	Node3D *current_node = nullptr;

	Skeleton3D *skeleton = state->skeletons[gltf_node->skeleton]->godot_skeleton;
	// In this case, this node is already a bone in skeleton.
	const bool is_skinned_mesh = (gltf_node->skin >= 0 && gltf_node->mesh >= 0);
	const bool requires_extra_node = (gltf_node->mesh >= 0 || gltf_node->camera >= 0 || gltf_node->light >= 0);

	Skeleton3D *active_skeleton = Object::cast_to<Skeleton3D>(scene_parent);
	if (active_skeleton != skeleton) {
		if (active_skeleton) {
			// Bone Attachment - Direct Parented Skeleton Case
			BoneAttachment3D *bone_attachment = _generate_bone_attachment(state, active_skeleton, node_index, gltf_node->parent);

			scene_parent->add_child(bone_attachment, true);
			bone_attachment->set_owner(scene_root);

			// There is no gltf_node that represent this, so just directly create a unique name
			bone_attachment->set_name(_gen_unique_name(state, "BoneAttachment3D"));

			// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
			// and attach it to the bone_attachment
			scene_parent = bone_attachment;
			WARN_PRINT(vformat("glTF: Generating scene detected direct parented Skeletons at node %d", node_index));
		}

		// Add it to the scene if it has not already been added
		if (skeleton->get_parent() == nullptr) {
			scene_parent->add_child(skeleton, true);
			skeleton->set_owner(scene_root);
		}
	}

	active_skeleton = skeleton;
	current_node = skeleton;

	if (requires_extra_node) {
		// skinned meshes must not be placed in a bone attachment.
		if (!is_skinned_mesh) {
			// Bone Attachment - Same Node Case
			BoneAttachment3D *bone_attachment = _generate_bone_attachment(state, active_skeleton, node_index, node_index);

			scene_parent->add_child(bone_attachment, true);
			bone_attachment->set_owner(scene_root);

			// There is no gltf_node that represent this, so just directly create a unique name
			bone_attachment->set_name(_gen_unique_name(state, "BoneAttachment3D"));

			// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
			// and attach it to the bone_attachment
			scene_parent = bone_attachment;
		}

		// We still have not managed to make a node
		if (gltf_node->mesh >= 0) {
			current_node = _generate_mesh_instance(state, node_index);
		} else if (gltf_node->camera >= 0) {
			current_node = _generate_camera(state, node_index);
		} else if (gltf_node->light >= 0) {
			current_node = _generate_light(state, node_index);
		}

		scene_parent->add_child(current_node, true);
		if (current_node != scene_root) {
			current_node->set_owner(scene_root);
		}
		// Do not set transform here. Transform is already applied to our bone.
		current_node->set_name(gltf_node->get_name());
	}

	state->scene_nodes.insert(node_index, current_node);

	for (int i = 0; i < gltf_node->children.size(); ++i) {
		_generate_scene_node(state, active_skeleton, scene_root, gltf_node->children[i]);
	}
}

template <class T>
struct EditorSceneFormatImporterGLTFInterpolate {
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

// thank you for existing, partial specialization
template <>
struct EditorSceneFormatImporterGLTFInterpolate<Quaternion> {
	Quaternion lerp(const Quaternion &a, const Quaternion &b, const float c) const {
		ERR_FAIL_COND_V_MSG(!a.is_normalized(), Quaternion(), "The quaternion \"a\" must be normalized.");
		ERR_FAIL_COND_V_MSG(!b.is_normalized(), Quaternion(), "The quaternion \"b\" must be normalized.");

		return a.slerp(b, c).normalized();
	}

	Quaternion catmull_rom(const Quaternion &p0, const Quaternion &p1, const Quaternion &p2, const Quaternion &p3, const float c) {
		ERR_FAIL_COND_V_MSG(!p1.is_normalized(), Quaternion(), "The quaternion \"p1\" must be normalized.");
		ERR_FAIL_COND_V_MSG(!p2.is_normalized(), Quaternion(), "The quaternion \"p2\" must be normalized.");

		return p1.slerp(p2, c).normalized();
	}

	Quaternion bezier(const Quaternion start, const Quaternion control_1, const Quaternion control_2, const Quaternion end, const float t) {
		ERR_FAIL_COND_V_MSG(!start.is_normalized(), Quaternion(), "The start quaternion must be normalized.");
		ERR_FAIL_COND_V_MSG(!end.is_normalized(), Quaternion(), "The end quaternion must be normalized.");

		return start.slerp(end, t).normalized();
	}
};

template <class T>
T GLTFDocument::_interpolate_track(const Vector<real_t> &p_times, const Vector<T> &p_values, const float p_time, const GLTFAnimation::Interpolation p_interp) {
	ERR_FAIL_COND_V(!p_values.size(), T());
	if (p_times.size() != (p_values.size() / (p_interp == GLTFAnimation::INTERP_CUBIC_SPLINE ? 3 : 1))) {
		ERR_PRINT_ONCE("The interpolated values are not corresponding to its times.");
		return p_values[0];
	}
	//could use binary search, worth it?
	int idx = -1;
	for (int i = 0; i < p_times.size(); i++) {
		if (p_times[i] > p_time) {
			break;
		}
		idx++;
	}

	EditorSceneFormatImporterGLTFInterpolate<T> interp;

	switch (p_interp) {
		case GLTFAnimation::INTERP_LINEAR: {
			if (idx == -1) {
				return p_values[0];
			} else if (idx >= p_times.size() - 1) {
				return p_values[p_times.size() - 1];
			}

			const float c = (p_time - p_times[idx]) / (p_times[idx + 1] - p_times[idx]);

			return interp.lerp(p_values[idx], p_values[idx + 1], c);
		} break;
		case GLTFAnimation::INTERP_STEP: {
			if (idx == -1) {
				return p_values[0];
			} else if (idx >= p_times.size() - 1) {
				return p_values[p_times.size() - 1];
			}

			return p_values[idx];
		} break;
		case GLTFAnimation::INTERP_CATMULLROMSPLINE: {
			if (idx == -1) {
				return p_values[1];
			} else if (idx >= p_times.size() - 1) {
				return p_values[1 + p_times.size() - 1];
			}

			const float c = (p_time - p_times[idx]) / (p_times[idx + 1] - p_times[idx]);

			return interp.catmull_rom(p_values[idx - 1], p_values[idx], p_values[idx + 1], p_values[idx + 3], c);
		} break;
		case GLTFAnimation::INTERP_CUBIC_SPLINE: {
			if (idx == -1) {
				return p_values[1];
			} else if (idx >= p_times.size() - 1) {
				return p_values[(p_times.size() - 1) * 3 + 1];
			}

			const float c = (p_time - p_times[idx]) / (p_times[idx + 1] - p_times[idx]);

			const T from = p_values[idx * 3 + 1];
			const T c1 = from + p_values[idx * 3 + 2];
			const T to = p_values[idx * 3 + 4];
			const T c2 = to + p_values[idx * 3 + 3];

			return interp.bezier(from, c1, c2, to, c);
		} break;
	}

	ERR_FAIL_V(p_values[0]);
}

void GLTFDocument::_import_animation(Ref<GLTFState> state, AnimationPlayer *ap, const GLTFAnimationIndex index, const int bake_fps) {
	Ref<GLTFAnimation> anim = state->animations[index];

	String name = anim->get_name();
	if (name.is_empty()) {
		// No node represent these, and they are not in the hierarchy, so just make a unique name
		name = _gen_unique_name(state, "Animation");
	}

	Ref<Animation> animation;
	animation.instantiate();
	animation->set_name(name);

	if (anim->get_loop()) {
		animation->set_loop_mode(Animation::LOOP_LINEAR);
	}

	float length = 0.0;

	for (const KeyValue<int, GLTFAnimation::Track> &track_i : anim->get_tracks()) {
		const GLTFAnimation::Track &track = track_i.value;
		//need to find the path: for skeletons, weight tracks will affect the mesh
		NodePath node_path;
		//for skeletons, transform tracks always affect bones
		NodePath transform_node_path;

		GLTFNodeIndex node_index = track_i.key;

		const Ref<GLTFNode> gltf_node = state->nodes[track_i.key];

		Node *root = ap->get_parent();
		ERR_FAIL_COND(root == nullptr);
		Map<GLTFNodeIndex, Node *>::Element *node_element = state->scene_nodes.find(node_index);
		ERR_CONTINUE_MSG(node_element == nullptr, vformat("Unable to find node %d for animation", node_index));
		node_path = root->get_path_to(node_element->get());

		if (gltf_node->skeleton >= 0) {
			const Skeleton3D *sk = state->skeletons[gltf_node->skeleton]->godot_skeleton;
			ERR_FAIL_COND(sk == nullptr);

			const String path = ap->get_parent()->get_path_to(sk);
			const String bone = gltf_node->get_name();
			transform_node_path = path + ":" + bone;
		} else {
			transform_node_path = node_path;
		}

		for (int i = 0; i < track.rotation_track.times.size(); i++) {
			length = MAX(length, track.rotation_track.times[i]);
		}
		for (int i = 0; i < track.position_track.times.size(); i++) {
			length = MAX(length, track.position_track.times[i]);
		}
		for (int i = 0; i < track.scale_track.times.size(); i++) {
			length = MAX(length, track.scale_track.times[i]);
		}

		for (int i = 0; i < track.weight_tracks.size(); i++) {
			for (int j = 0; j < track.weight_tracks[i].times.size(); j++) {
				length = MAX(length, track.weight_tracks[i].times[j]);
			}
		}

		// Animated TRS properties will not affect a skinned mesh.
		const bool transform_affects_skinned_mesh_instance = gltf_node->skeleton < 0 && gltf_node->skin >= 0;
		if ((track.rotation_track.values.size() || track.position_track.values.size() || track.scale_track.values.size()) && !transform_affects_skinned_mesh_instance) {
			//make transform track
			int base_idx = animation->get_track_count();
			int position_idx = -1;
			int rotation_idx = -1;
			int scale_idx = -1;

			if (track.position_track.values.size()) {
				Vector3 base_pos = state->nodes[track_i.key]->position;
				bool not_default = false; //discard the track if all it contains is default values
				for (int i = 0; i < track.position_track.times.size(); i++) {
					Vector3 value = track.position_track.values[track.position_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + i * 3) : i];
					if (!value.is_equal_approx(base_pos)) {
						not_default = true;
						break;
					}
				}
				if (not_default) {
					position_idx = base_idx;
					animation->add_track(Animation::TYPE_POSITION_3D);
					animation->track_set_path(position_idx, transform_node_path);
					animation->track_set_imported(position_idx, true); //helps merging later

					base_idx++;
				}
			}
			if (track.rotation_track.values.size()) {
				Quaternion base_rot = state->nodes[track_i.key]->rotation.normalized();
				bool not_default = false; //discard the track if all it contains is default values
				for (int i = 0; i < track.rotation_track.times.size(); i++) {
					Quaternion value = track.rotation_track.values[track.rotation_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + i * 3) : i].normalized();
					if (!value.is_equal_approx(base_rot)) {
						not_default = true;
						break;
					}
				}
				if (not_default) {
					rotation_idx = base_idx;
					animation->add_track(Animation::TYPE_ROTATION_3D);
					animation->track_set_path(rotation_idx, transform_node_path);
					animation->track_set_imported(rotation_idx, true); //helps merging later
					base_idx++;
				}
			}
			if (track.scale_track.values.size()) {
				Vector3 base_scale = state->nodes[track_i.key]->scale;
				bool not_default = false; //discard the track if all it contains is default values
				for (int i = 0; i < track.scale_track.times.size(); i++) {
					Vector3 value = track.scale_track.values[track.scale_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + i * 3) : i];
					if (!value.is_equal_approx(base_scale)) {
						not_default = true;
						break;
					}
				}
				if (not_default) {
					scale_idx = base_idx;
					animation->add_track(Animation::TYPE_SCALE_3D);
					animation->track_set_path(scale_idx, transform_node_path);
					animation->track_set_imported(scale_idx, true); //helps merging later
					base_idx++;
				}
			}

			//first determine animation length

			const double increment = 1.0 / bake_fps;
			double time = 0.0;

			Vector3 base_pos;
			Quaternion base_rot;
			Vector3 base_scale = Vector3(1, 1, 1);

			if (rotation_idx == -1) {
				base_rot = state->nodes[track_i.key]->rotation.normalized();
			}

			if (position_idx == -1) {
				base_pos = state->nodes[track_i.key]->position;
			}

			if (scale_idx == -1) {
				base_scale = state->nodes[track_i.key]->scale;
			}

			bool last = false;
			while (true) {
				Vector3 pos = base_pos;
				Quaternion rot = base_rot;
				Vector3 scale = base_scale;

				if (position_idx >= 0) {
					pos = _interpolate_track<Vector3>(track.position_track.times, track.position_track.values, time, track.position_track.interpolation);
					animation->position_track_insert_key(position_idx, time, pos);
				}

				if (rotation_idx >= 0) {
					rot = _interpolate_track<Quaternion>(track.rotation_track.times, track.rotation_track.values, time, track.rotation_track.interpolation);
					animation->rotation_track_insert_key(rotation_idx, time, rot);
				}

				if (scale_idx >= 0) {
					scale = _interpolate_track<Vector3>(track.scale_track.times, track.scale_track.values, time, track.scale_track.interpolation);
					animation->scale_track_insert_key(scale_idx, time, scale);
				}

				if (last) {
					break;
				}
				time += increment;
				if (time >= length) {
					last = true;
					time = length;
				}
			}
		}

		for (int i = 0; i < track.weight_tracks.size(); i++) {
			ERR_CONTINUE(gltf_node->mesh < 0 || gltf_node->mesh >= state->meshes.size());
			Ref<GLTFMesh> mesh = state->meshes[gltf_node->mesh];
			ERR_CONTINUE(mesh.is_null());
			ERR_CONTINUE(mesh->get_mesh().is_null());
			ERR_CONTINUE(mesh->get_mesh()->get_mesh().is_null());

			const String blend_path = String(node_path) + ":" + String(mesh->get_mesh()->get_blend_shape_name(i));

			const int track_idx = animation->get_track_count();
			animation->add_track(Animation::TYPE_BLEND_SHAPE);
			animation->track_set_path(track_idx, blend_path);

			// Only LINEAR and STEP (NEAREST) can be supported out of the box by Godot's Animation,
			// the other modes have to be baked.
			GLTFAnimation::Interpolation gltf_interp = track.weight_tracks[i].interpolation;
			if (gltf_interp == GLTFAnimation::INTERP_LINEAR || gltf_interp == GLTFAnimation::INTERP_STEP) {
				animation->track_set_interpolation_type(track_idx, gltf_interp == GLTFAnimation::INTERP_STEP ? Animation::INTERPOLATION_NEAREST : Animation::INTERPOLATION_LINEAR);
				for (int j = 0; j < track.weight_tracks[i].times.size(); j++) {
					const float t = track.weight_tracks[i].times[j];
					const float attribs = track.weight_tracks[i].values[j];
					animation->blend_shape_track_insert_key(track_idx, t, attribs);
				}
			} else {
				// CATMULLROMSPLINE or CUBIC_SPLINE have to be baked, apologies.
				const double increment = 1.0 / bake_fps;
				double time = 0.0;
				bool last = false;
				while (true) {
					real_t blend = _interpolate_track<real_t>(track.weight_tracks[i].times, track.weight_tracks[i].values, time, gltf_interp);
					animation->blend_shape_track_insert_key(track_idx, time, blend);
					if (last) {
						break;
					}
					time += increment;
					if (time >= length) {
						last = true;
						time = length;
					}
				}
			}
		}
	}

	animation->set_length(length);

	ap->add_animation(name, animation);
}

void GLTFDocument::_convert_mesh_instances(Ref<GLTFState> state) {
	for (GLTFNodeIndex mi_node_i = 0; mi_node_i < state->nodes.size(); ++mi_node_i) {
		Ref<GLTFNode> node = state->nodes[mi_node_i];

		if (node->mesh < 0) {
			continue;
		}
		Map<GLTFNodeIndex, Node *>::Element *mi_element = state->scene_nodes.find(mi_node_i);
		if (!mi_element) {
			continue;
		}
		MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(mi_element->get());
		if (!mi) {
			continue;
		}
		Transform3D mi_xform = mi->get_transform();
		node->scale = mi_xform.basis.get_scale();
		node->rotation = mi_xform.basis.get_rotation_quaternion();
		node->position = mi_xform.origin;

		Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(mi->get_node(mi->get_skeleton_path()));
		if (!skeleton) {
			continue;
		}
		if (!skeleton->get_bone_count()) {
			continue;
		}
		Ref<Skin> skin = mi->get_skin();
		Ref<GLTFSkin> gltf_skin;
		gltf_skin.instantiate();
		Array json_joints;

		NodePath skeleton_path = mi->get_skeleton_path();
		Node *skel_node = mi->get_node_or_null(skeleton_path);
		Skeleton3D *godot_skeleton = nullptr;
		if (skel_node != nullptr) {
			godot_skeleton = cast_to<Skeleton3D>(skel_node);
		}
		if (godot_skeleton != nullptr && state->skeleton3d_to_gltf_skeleton.has(godot_skeleton->get_instance_id())) {
			// This is a skinned mesh. If the mesh has no ARRAY_WEIGHTS or ARRAY_BONES, it will be invisible.
			const GLTFSkeletonIndex skeleton_gltf_i = state->skeleton3d_to_gltf_skeleton[godot_skeleton->get_instance_id()];
			Ref<GLTFSkeleton> gltf_skeleton = state->skeletons[skeleton_gltf_i];
			int bone_cnt = skeleton->get_bone_count();
			ERR_FAIL_COND(bone_cnt != gltf_skeleton->joints.size());

			ObjectID gltf_skin_key;
			if (skin.is_valid()) {
				gltf_skin_key = skin->get_instance_id();
			}
			ObjectID gltf_skel_key = godot_skeleton->get_instance_id();
			GLTFSkinIndex skin_gltf_i = -1;
			GLTFNodeIndex root_gltf_i = -1;
			if (!gltf_skeleton->roots.is_empty()) {
				root_gltf_i = gltf_skeleton->roots[0];
			}
			if (state->skin_and_skeleton3d_to_gltf_skin.has(gltf_skin_key) && state->skin_and_skeleton3d_to_gltf_skin[gltf_skin_key].has(gltf_skel_key)) {
				skin_gltf_i = state->skin_and_skeleton3d_to_gltf_skin[gltf_skin_key][gltf_skel_key];
			} else {
				if (skin.is_null()) {
					// Note that gltf_skin_key should remain null, so these can share a reference.
					skin = skeleton->create_skin_from_rest_transforms();
				}
				gltf_skin.instantiate();
				gltf_skin->godot_skin = skin;
				gltf_skin->set_name(skin->get_name());
				gltf_skin->skeleton = skeleton_gltf_i;
				gltf_skin->skin_root = root_gltf_i;
				//gltf_state->godot_to_gltf_node[skel_node]
				HashMap<StringName, int> bone_name_to_idx;
				for (int bone_i = 0; bone_i < bone_cnt; bone_i++) {
					bone_name_to_idx[skeleton->get_bone_name(bone_i)] = bone_i;
				}
				for (int bind_i = 0, cnt = skin->get_bind_count(); bind_i < cnt; bind_i++) {
					int bone_i = skin->get_bind_bone(bind_i);
					Transform3D bind_pose = skin->get_bind_pose(bind_i);
					StringName bind_name = skin->get_bind_name(bind_i);
					if (bind_name != StringName()) {
						bone_i = bone_name_to_idx[bind_name];
					}
					ERR_CONTINUE(bone_i < 0 || bone_i >= bone_cnt);
					if (bind_name == StringName()) {
						bind_name = skeleton->get_bone_name(bone_i);
					}
					GLTFNodeIndex skeleton_bone_i = gltf_skeleton->joints[bone_i];
					gltf_skin->joints_original.push_back(skeleton_bone_i);
					gltf_skin->joints.push_back(skeleton_bone_i);
					gltf_skin->inverse_binds.push_back(bind_pose);
					if (skeleton->get_bone_parent(bone_i) == -1) {
						gltf_skin->roots.push_back(skeleton_bone_i);
					}
					gltf_skin->joint_i_to_bone_i[bind_i] = bone_i;
					gltf_skin->joint_i_to_name[bind_i] = bind_name;
				}
				skin_gltf_i = state->skins.size();
				state->skins.push_back(gltf_skin);
				state->skin_and_skeleton3d_to_gltf_skin[gltf_skin_key][gltf_skel_key] = skin_gltf_i;
			}
			node->skin = skin_gltf_i;
			node->skeleton = skeleton_gltf_i;
		}
	}
}

float GLTFDocument::solve_metallic(float p_dielectric_specular, float diffuse, float specular, float p_one_minus_specular_strength) {
	if (specular <= p_dielectric_specular) {
		return 0.0f;
	}

	const float a = p_dielectric_specular;
	const float b = diffuse * p_one_minus_specular_strength / (1.0f - p_dielectric_specular) + specular - 2.0f * p_dielectric_specular;
	const float c = p_dielectric_specular - specular;
	const float D = b * b - 4.0f * a * c;
	return CLAMP((-b + Math::sqrt(D)) / (2.0f * a), 0.0f, 1.0f);
}

float GLTFDocument::get_perceived_brightness(const Color p_color) {
	const Color coeff = Color(R_BRIGHTNESS_COEFF, G_BRIGHTNESS_COEFF, B_BRIGHTNESS_COEFF);
	const Color value = coeff * (p_color * p_color);

	const float r = value.r;
	const float g = value.g;
	const float b = value.b;

	return Math::sqrt(r + g + b);
}

float GLTFDocument::get_max_component(const Color &p_color) {
	const float r = p_color.r;
	const float g = p_color.g;
	const float b = p_color.b;

	return MAX(MAX(r, g), b);
}

void GLTFDocument::_process_mesh_instances(Ref<GLTFState> state, Node *scene_root) {
	for (GLTFNodeIndex node_i = 0; node_i < state->nodes.size(); ++node_i) {
		Ref<GLTFNode> node = state->nodes[node_i];

		if (node->skin >= 0 && node->mesh >= 0) {
			const GLTFSkinIndex skin_i = node->skin;

			Map<GLTFNodeIndex, Node *>::Element *mi_element = state->scene_nodes.find(node_i);
			ERR_CONTINUE_MSG(mi_element == nullptr, vformat("Unable to find node %d", node_i));

			ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(mi_element->get());
			ERR_CONTINUE_MSG(mi == nullptr, vformat("Unable to cast node %d of type %s to ImporterMeshInstance3D", node_i, mi_element->get()->get_class_name()));

			const GLTFSkeletonIndex skel_i = state->skins.write[node->skin]->skeleton;
			Ref<GLTFSkeleton> gltf_skeleton = state->skeletons.write[skel_i];
			Skeleton3D *skeleton = gltf_skeleton->godot_skeleton;
			ERR_CONTINUE_MSG(skeleton == nullptr, vformat("Unable to find Skeleton for node %d skin %d", node_i, skin_i));

			mi->get_parent()->remove_child(mi);
			skeleton->add_child(mi, true);
			mi->set_owner(skeleton->get_owner());

			mi->set_skin(state->skins.write[skin_i]->godot_skin);
			mi->set_skeleton_path(mi->get_path_to(skeleton));
			mi->set_transform(Transform3D());
		}
	}
}

GLTFAnimation::Track GLTFDocument::_convert_animation_track(Ref<GLTFState> state, GLTFAnimation::Track p_track, Ref<Animation> p_animation, int32_t p_track_i, GLTFNodeIndex p_node_i) {
	Animation::InterpolationType interpolation = p_animation->track_get_interpolation_type(p_track_i);

	GLTFAnimation::Interpolation gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
	if (interpolation == Animation::InterpolationType::INTERPOLATION_LINEAR) {
		gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
	} else if (interpolation == Animation::InterpolationType::INTERPOLATION_NEAREST) {
		gltf_interpolation = GLTFAnimation::INTERP_STEP;
	} else if (interpolation == Animation::InterpolationType::INTERPOLATION_CUBIC) {
		gltf_interpolation = GLTFAnimation::INTERP_CUBIC_SPLINE;
	}
	Animation::TrackType track_type = p_animation->track_get_type(p_track_i);
	int32_t key_count = p_animation->track_get_key_count(p_track_i);
	Vector<real_t> times;
	times.resize(key_count);
	String path = p_animation->track_get_path(p_track_i);
	for (int32_t key_i = 0; key_i < key_count; key_i++) {
		times.write[key_i] = p_animation->track_get_key_time(p_track_i, key_i);
	}
	if (track_type == Animation::TYPE_SCALE_3D) {
		p_track.scale_track.times = times;
		p_track.scale_track.interpolation = gltf_interpolation;
		p_track.scale_track.values.resize(key_count);
		for (int32_t key_i = 0; key_i < key_count; key_i++) {
			Vector3 scale;
			Error err = p_animation->scale_track_get_key(p_track_i, key_i, &scale);
			ERR_CONTINUE(err != OK);
			p_track.scale_track.values.write[key_i] = scale;
		}
	} else if (track_type == Animation::TYPE_POSITION_3D) {
		p_track.position_track.times = times;
		p_track.position_track.values.resize(key_count);
		p_track.position_track.interpolation = gltf_interpolation;
		for (int32_t key_i = 0; key_i < key_count; key_i++) {
			Vector3 position;
			Error err = p_animation->position_track_get_key(p_track_i, key_i, &position);
			ERR_CONTINUE(err != OK);
			p_track.position_track.values.write[key_i] = position;
		}
	} else if (track_type == Animation::TYPE_ROTATION_3D) {
		p_track.rotation_track.times = times;
		p_track.rotation_track.interpolation = gltf_interpolation;
		p_track.rotation_track.values.resize(key_count);
		for (int32_t key_i = 0; key_i < key_count; key_i++) {
			Quaternion rotation;
			Error err = p_animation->rotation_track_get_key(p_track_i, key_i, &rotation);
			ERR_CONTINUE(err != OK);
			p_track.rotation_track.values.write[key_i] = rotation;
		}
	} else if (track_type == Animation::TYPE_VALUE) {
		if (path.find(":position") != -1) {
			p_track.position_track.times = times;
			p_track.position_track.interpolation = gltf_interpolation;

			p_track.position_track.values.resize(key_count);
			p_track.position_track.interpolation = gltf_interpolation;

			for (int32_t key_i = 0; key_i < key_count; key_i++) {
				Vector3 position = p_animation->track_get_key_value(p_track_i, key_i);
				p_track.position_track.values.write[key_i] = position;
			}
		} else if (path.find(":rotation") != -1) {
			p_track.rotation_track.times = times;
			p_track.rotation_track.interpolation = gltf_interpolation;

			p_track.rotation_track.values.resize(key_count);
			p_track.rotation_track.interpolation = gltf_interpolation;

			for (int32_t key_i = 0; key_i < key_count; key_i++) {
				Vector3 rotation_radian = p_animation->track_get_key_value(p_track_i, key_i);
				p_track.rotation_track.values.write[key_i] = Quaternion(rotation_radian);
			}
		} else if (path.find(":scale") != -1) {
			p_track.scale_track.times = times;
			p_track.scale_track.interpolation = gltf_interpolation;

			p_track.scale_track.values.resize(key_count);
			p_track.scale_track.interpolation = gltf_interpolation;

			for (int32_t key_i = 0; key_i < key_count; key_i++) {
				Vector3 scale_track = p_animation->track_get_key_value(p_track_i, key_i);
				p_track.scale_track.values.write[key_i] = scale_track;
			}
		}
	} else if (track_type == Animation::TYPE_BEZIER) {
		if (path.find("/scale") != -1) {
			const int32_t keys = p_animation->track_get_key_time(p_track_i, key_count - 1) * BAKE_FPS;
			if (!p_track.scale_track.times.size()) {
				Vector<real_t> new_times;
				new_times.resize(keys);
				for (int32_t key_i = 0; key_i < keys; key_i++) {
					new_times.write[key_i] = key_i / BAKE_FPS;
				}
				p_track.scale_track.times = new_times;
				p_track.scale_track.interpolation = gltf_interpolation;

				p_track.scale_track.values.resize(keys);

				for (int32_t key_i = 0; key_i < keys; key_i++) {
					p_track.scale_track.values.write[key_i] = Vector3(1.0f, 1.0f, 1.0f);
				}
				p_track.scale_track.interpolation = gltf_interpolation;
			}

			for (int32_t key_i = 0; key_i < keys; key_i++) {
				Vector3 bezier_track = p_track.scale_track.values[key_i];
				if (path.find("/scale:x") != -1) {
					bezier_track.x = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
				} else if (path.find("/scale:y") != -1) {
					bezier_track.y = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
				} else if (path.find("/scale:z") != -1) {
					bezier_track.z = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
				}
				p_track.scale_track.values.write[key_i] = bezier_track;
			}
		} else if (path.find("/position") != -1) {
			const int32_t keys = p_animation->track_get_key_time(p_track_i, key_count - 1) * BAKE_FPS;
			if (!p_track.position_track.times.size()) {
				Vector<real_t> new_times;
				new_times.resize(keys);
				for (int32_t key_i = 0; key_i < keys; key_i++) {
					new_times.write[key_i] = key_i / BAKE_FPS;
				}
				p_track.position_track.times = new_times;
				p_track.position_track.interpolation = gltf_interpolation;

				p_track.position_track.values.resize(keys);
				p_track.position_track.interpolation = gltf_interpolation;
			}

			for (int32_t key_i = 0; key_i < keys; key_i++) {
				Vector3 bezier_track = p_track.position_track.values[key_i];
				if (path.find("/position:x") != -1) {
					bezier_track.x = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
				} else if (path.find("/position:y") != -1) {
					bezier_track.y = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
				} else if (path.find("/position:z") != -1) {
					bezier_track.z = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
				}
				p_track.position_track.values.write[key_i] = bezier_track;
			}
		}
	}
	return p_track;
}

void GLTFDocument::_convert_animation(Ref<GLTFState> state, AnimationPlayer *ap, String p_animation_track_name) {
	Ref<Animation> animation = ap->get_animation(p_animation_track_name);
	Ref<GLTFAnimation> gltf_animation;
	gltf_animation.instantiate();
	gltf_animation->set_name(_gen_unique_name(state, p_animation_track_name));

	for (int32_t track_i = 0; track_i < animation->get_track_count(); track_i++) {
		if (!animation->track_is_enabled(track_i)) {
			continue;
		}
		String orig_track_path = animation->track_get_path(track_i);
		if (String(orig_track_path).find(":position") != -1) {
			const Vector<String> node_suffix = String(orig_track_path).split(":position");
			const NodePath path = node_suffix[0];
			const Node *node = ap->get_parent()->get_node_or_null(path);
			for (const KeyValue<GLTFNodeIndex, Node *> &position_scene_node_i : state->scene_nodes) {
				if (position_scene_node_i.value == node) {
					GLTFNodeIndex node_index = position_scene_node_i.key;
					Map<int, GLTFAnimation::Track>::Element *position_track_i = gltf_animation->get_tracks().find(node_index);
					GLTFAnimation::Track track;
					if (position_track_i) {
						track = position_track_i->get();
					}
					track = _convert_animation_track(state, track, animation, track_i, node_index);
					gltf_animation->get_tracks().insert(node_index, track);
				}
			}
		} else if (String(orig_track_path).find(":rotation_degrees") != -1) {
			const Vector<String> node_suffix = String(orig_track_path).split(":rotation_degrees");
			const NodePath path = node_suffix[0];
			const Node *node = ap->get_parent()->get_node_or_null(path);
			for (const KeyValue<GLTFNodeIndex, Node *> &rotation_degree_scene_node_i : state->scene_nodes) {
				if (rotation_degree_scene_node_i.value == node) {
					GLTFNodeIndex node_index = rotation_degree_scene_node_i.key;
					Map<int, GLTFAnimation::Track>::Element *rotation_degree_track_i = gltf_animation->get_tracks().find(node_index);
					GLTFAnimation::Track track;
					if (rotation_degree_track_i) {
						track = rotation_degree_track_i->get();
					}
					track = _convert_animation_track(state, track, animation, track_i, node_index);
					gltf_animation->get_tracks().insert(node_index, track);
				}
			}
		} else if (String(orig_track_path).find(":scale") != -1) {
			const Vector<String> node_suffix = String(orig_track_path).split(":scale");
			const NodePath path = node_suffix[0];
			const Node *node = ap->get_parent()->get_node_or_null(path);
			for (const KeyValue<GLTFNodeIndex, Node *> &scale_scene_node_i : state->scene_nodes) {
				if (scale_scene_node_i.value == node) {
					GLTFNodeIndex node_index = scale_scene_node_i.key;
					Map<int, GLTFAnimation::Track>::Element *scale_track_i = gltf_animation->get_tracks().find(node_index);
					GLTFAnimation::Track track;
					if (scale_track_i) {
						track = scale_track_i->get();
					}
					track = _convert_animation_track(state, track, animation, track_i, node_index);
					gltf_animation->get_tracks().insert(node_index, track);
				}
			}
		} else if (String(orig_track_path).find(":transform") != -1) {
			const Vector<String> node_suffix = String(orig_track_path).split(":transform");
			const NodePath path = node_suffix[0];
			const Node *node = ap->get_parent()->get_node_or_null(path);
			for (const KeyValue<GLTFNodeIndex, Node *> &transform_track_i : state->scene_nodes) {
				if (transform_track_i.value == node) {
					GLTFAnimation::Track track;
					track = _convert_animation_track(state, track, animation, track_i, transform_track_i.key);
					gltf_animation->get_tracks().insert(transform_track_i.key, track);
				}
			}
		} else if (String(orig_track_path).find(":blend_shapes/") != -1) {
			const Vector<String> node_suffix = String(orig_track_path).split(":blend_shapes/");
			const NodePath path = node_suffix[0];
			const String suffix = node_suffix[1];
			Node *node = ap->get_parent()->get_node_or_null(path);
			MeshInstance3D *mi = cast_to<MeshInstance3D>(node);
			Ref<Mesh> mesh = mi->get_mesh();
			ERR_CONTINUE(mesh.is_null());
			int32_t mesh_index = -1;
			for (const KeyValue<GLTFNodeIndex, Node *> &mesh_track_i : state->scene_nodes) {
				if (mesh_track_i.value == node) {
					mesh_index = mesh_track_i.key;
				}
			}
			ERR_CONTINUE(mesh_index == -1);
			Map<int, GLTFAnimation::Track> &tracks = gltf_animation->get_tracks();
			GLTFAnimation::Track track = gltf_animation->get_tracks().has(mesh_index) ? gltf_animation->get_tracks()[mesh_index] : GLTFAnimation::Track();
			if (!tracks.has(mesh_index)) {
				for (int32_t shape_i = 0; shape_i < mesh->get_blend_shape_count(); shape_i++) {
					String shape_name = mesh->get_blend_shape_name(shape_i);
					NodePath shape_path = String(path) + ":" + shape_name;
					int32_t shape_track_i = animation->find_track(shape_path, Animation::TYPE_BLEND_SHAPE);
					if (shape_track_i == -1) {
						GLTFAnimation::Channel<real_t> weight;
						weight.interpolation = GLTFAnimation::INTERP_LINEAR;
						weight.times.push_back(0.0f);
						weight.times.push_back(0.0f);
						weight.values.push_back(0.0f);
						weight.values.push_back(0.0f);
						track.weight_tracks.push_back(weight);
						continue;
					}
					Animation::InterpolationType interpolation = animation->track_get_interpolation_type(track_i);
					GLTFAnimation::Interpolation gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
					if (interpolation == Animation::InterpolationType::INTERPOLATION_LINEAR) {
						gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
					} else if (interpolation == Animation::InterpolationType::INTERPOLATION_NEAREST) {
						gltf_interpolation = GLTFAnimation::INTERP_STEP;
					} else if (interpolation == Animation::InterpolationType::INTERPOLATION_CUBIC) {
						gltf_interpolation = GLTFAnimation::INTERP_CUBIC_SPLINE;
					}
					int32_t key_count = animation->track_get_key_count(shape_track_i);
					GLTFAnimation::Channel<real_t> weight;
					weight.interpolation = gltf_interpolation;
					weight.times.resize(key_count);
					for (int32_t time_i = 0; time_i < key_count; time_i++) {
						weight.times.write[time_i] = animation->track_get_key_time(shape_track_i, time_i);
					}
					weight.values.resize(key_count);
					for (int32_t value_i = 0; value_i < key_count; value_i++) {
						weight.values.write[value_i] = animation->track_get_key_value(shape_track_i, value_i);
					}
					track.weight_tracks.push_back(weight);
				}
				tracks[mesh_index] = track;
			}
		} else if (String(orig_track_path).find(":") != -1) {
			//Process skeleton
			const Vector<String> node_suffix = String(orig_track_path).split(":");
			const String node = node_suffix[0];
			const NodePath node_path = node;
			const String suffix = node_suffix[1];
			Node *godot_node = ap->get_parent()->get_node_or_null(node_path);
			Skeleton3D *skeleton = nullptr;
			GLTFSkeletonIndex skeleton_gltf_i = -1;
			for (GLTFSkeletonIndex skeleton_i = 0; skeleton_i < state->skeletons.size(); skeleton_i++) {
				if (state->skeletons[skeleton_i]->godot_skeleton == cast_to<Skeleton3D>(godot_node)) {
					skeleton = state->skeletons[skeleton_i]->godot_skeleton;
					skeleton_gltf_i = skeleton_i;
					ERR_CONTINUE(!skeleton);
					Ref<GLTFSkeleton> skeleton_gltf = state->skeletons[skeleton_gltf_i];
					int32_t bone = skeleton->find_bone(suffix);
					ERR_CONTINUE(bone == -1);
					if (!skeleton_gltf->godot_bone_node.has(bone)) {
						continue;
					}
					GLTFNodeIndex node_i = skeleton_gltf->godot_bone_node[bone];
					Map<int, GLTFAnimation::Track>::Element *property_track_i = gltf_animation->get_tracks().find(node_i);
					GLTFAnimation::Track track;
					if (property_track_i) {
						track = property_track_i->get();
					}
					track = _convert_animation_track(state, track, animation, track_i, node_i);
					gltf_animation->get_tracks()[node_i] = track;
				}
			}
		} else if (String(orig_track_path).find(":") == -1) {
			ERR_CONTINUE(!ap->get_parent());
			Node *godot_node = ap->get_parent()->get_node_or_null(orig_track_path);
			for (const KeyValue<GLTFNodeIndex, Node *> &scene_node_i : state->scene_nodes) {
				if (scene_node_i.value == godot_node) {
					GLTFNodeIndex node_i = scene_node_i.key;
					Map<int, GLTFAnimation::Track>::Element *node_track_i = gltf_animation->get_tracks().find(node_i);
					GLTFAnimation::Track track;
					if (node_track_i) {
						track = node_track_i->get();
					}
					track = _convert_animation_track(state, track, animation, track_i, node_i);
					gltf_animation->get_tracks()[node_i] = track;
					break;
				}
			}
		}
	}
	if (gltf_animation->get_tracks().size()) {
		state->animations.push_back(gltf_animation);
	}
}

Error GLTFDocument::_parse(Ref<GLTFState> state, String p_path, FileAccess *f, int p_bake_fps) {
	Error err;
	if (!f) {
		return FAILED;
	}
	f->seek(0);
	uint32_t magic = f->get_32();
	if (magic == 0x46546C67) {
		//binary file
		//text file
		f->seek(0);
		err = _parse_glb(f, state);
		if (err != OK) {
			return err;
		}
	} else {
		f->seek(0);
		String text = f->get_as_utf8_string();
		JSON json;
		err = json.parse(text);
		if (err != OK) {
			_err_print_error("", "", json.get_error_line(), json.get_error_message().utf8().get_data(), false, ERR_HANDLER_SCRIPT);
		}
		ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);
		state->json = json.get_data();
	}
	f->close();

	if (!state->json.has("asset")) {
		return ERR_PARSE_ERROR;
	}

	Dictionary asset = state->json["asset"];

	if (!asset.has("version")) {
		return ERR_PARSE_ERROR;
	}

	String version = asset["version"];

	state->major_version = version.get_slice(".", 0).to_int();
	state->minor_version = version.get_slice(".", 1).to_int();

	for (int32_t ext_i = 0; ext_i < document_extensions.size(); ext_i++) {
		Ref<GLTFDocumentExtension> ext = document_extensions[ext_i];
		ERR_CONTINUE(ext.is_null());
		err = ext->import_preflight(this);
		ERR_FAIL_COND_V(err != OK, FAILED);
	}
	err = _parse_gltf_state(state, p_path, p_bake_fps);
	ERR_FAIL_COND_V(err != OK, err);
	return OK;
}

Dictionary GLTFDocument::_serialize_texture_transform_uv2(Ref<BaseMaterial3D> p_material) {
	Dictionary extension;
	Ref<BaseMaterial3D> mat = p_material;
	if (mat.is_valid()) {
		Dictionary texture_transform;
		Array offset;
		offset.resize(2);
		offset[0] = mat->get_uv2_offset().x;
		offset[1] = mat->get_uv2_offset().y;
		texture_transform["offset"] = offset;
		Array scale;
		scale.resize(2);
		scale[0] = mat->get_uv2_scale().x;
		scale[1] = mat->get_uv2_scale().y;
		texture_transform["scale"] = scale;
		// Godot doesn't support texture rotation
		extension["KHR_texture_transform"] = texture_transform;
	}
	return extension;
}

Dictionary GLTFDocument::_serialize_texture_transform_uv1(Ref<BaseMaterial3D> p_material) {
	Dictionary extension;
	if (p_material.is_valid()) {
		Dictionary texture_transform;
		Array offset;
		offset.resize(2);
		offset[0] = p_material->get_uv1_offset().x;
		offset[1] = p_material->get_uv1_offset().y;
		texture_transform["offset"] = offset;
		Array scale;
		scale.resize(2);
		scale[0] = p_material->get_uv1_scale().x;
		scale[1] = p_material->get_uv1_scale().y;
		texture_transform["scale"] = scale;
		// Godot doesn't support texture rotation
		extension["KHR_texture_transform"] = texture_transform;
	}
	return extension;
}

Error GLTFDocument::_serialize_version(Ref<GLTFState> state) {
	const String version = "2.0";
	state->major_version = version.get_slice(".", 0).to_int();
	state->minor_version = version.get_slice(".", 1).to_int();
	Dictionary asset;
	asset["version"] = version;

	String hash = VERSION_HASH;
	asset["generator"] = String(VERSION_FULL_NAME) + String("@") + (hash.length() == 0 ? String("unknown") : hash);
	state->json["asset"] = asset;
	ERR_FAIL_COND_V(!asset.has("version"), Error::FAILED);
	ERR_FAIL_COND_V(!state->json.has("asset"), Error::FAILED);
	return OK;
}

Error GLTFDocument::_serialize_file(Ref<GLTFState> state, const String p_path) {
	Error err = FAILED;
	if (p_path.to_lower().ends_with("glb")) {
		err = _encode_buffer_glb(state, p_path);
		ERR_FAIL_COND_V(err != OK, err);
		FileAccessRef f = FileAccess::open(p_path, FileAccess::WRITE, &err);
		ERR_FAIL_COND_V(!f, FAILED);

		String json = Variant(state->json).to_json_string();

		const uint32_t magic = 0x46546C67; // GLTF
		const int32_t header_size = 12;
		const int32_t chunk_header_size = 8;
		CharString cs = json.utf8();
		const uint32_t text_data_length = cs.length();
		const uint32_t text_chunk_length = ((text_data_length + 3) & (~3));
		const uint32_t text_chunk_type = 0x4E4F534A; //JSON

		uint32_t binary_data_length = 0;
		if (state->buffers.size()) {
			binary_data_length = state->buffers[0].size();
		}
		const uint32_t binary_chunk_length = ((binary_data_length + 3) & (~3));
		const uint32_t binary_chunk_type = 0x004E4942; //BIN

		f->create(FileAccess::ACCESS_RESOURCES);
		f->store_32(magic);
		f->store_32(state->major_version); // version
		f->store_32(header_size + chunk_header_size + text_chunk_length + chunk_header_size + binary_chunk_length); // length
		f->store_32(text_chunk_length);
		f->store_32(text_chunk_type);
		f->store_buffer((uint8_t *)&cs[0], cs.length());
		for (uint32_t pad_i = text_data_length; pad_i < text_chunk_length; pad_i++) {
			f->store_8(' ');
		}
		if (binary_chunk_length) {
			f->store_32(binary_chunk_length);
			f->store_32(binary_chunk_type);
			f->store_buffer(state->buffers[0].ptr(), binary_data_length);
		}
		for (uint32_t pad_i = binary_data_length; pad_i < binary_chunk_length; pad_i++) {
			f->store_8(0);
		}

		f->close();
	} else {
		err = _encode_buffer_bins(state, p_path);
		ERR_FAIL_COND_V(err != OK, err);
		FileAccessRef f = FileAccess::open(p_path, FileAccess::WRITE, &err);
		ERR_FAIL_COND_V(!f, FAILED);

		f->create(FileAccess::ACCESS_RESOURCES);
		String json = Variant(state->json).to_json_string();
		f->store_string(json);
		f->close();
	}
	return err;
}

void GLTFDocument::_bind_methods() {
	ClassDB::bind_method(D_METHOD("append_from_file", "path", "state", "flags", "bake_fps"),
			&GLTFDocument::append_from_file, DEFVAL(0), DEFVAL(30));
	ClassDB::bind_method(D_METHOD("append_from_buffer", "bytes", "base_path", "state", "flags", "bake_fps"),
			&GLTFDocument::append_from_buffer, DEFVAL(0), DEFVAL(30));
	ClassDB::bind_method(D_METHOD("append_from_scene", "node", "state", "flags", "bake_fps"),
			&GLTFDocument::append_from_scene, DEFVAL(0), DEFVAL(30));
	ClassDB::bind_method(D_METHOD("generate_scene", "state", "bake_fps"),
			&GLTFDocument::generate_scene, DEFVAL(30));
	ClassDB::bind_method(D_METHOD("generate_buffer", "state"),
			&GLTFDocument::generate_buffer);
	ClassDB::bind_method(D_METHOD("write_to_filesystem", "state", "path"),
			&GLTFDocument::write_to_filesystem);

	ClassDB::bind_method(D_METHOD("set_extensions", "extensions"),
			&GLTFDocument::set_extensions);
	ClassDB::bind_method(D_METHOD("get_extensions"),
			&GLTFDocument::get_extensions);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "extensions", PROPERTY_HINT_ARRAY_TYPE,
						 vformat("%s/%s:%s", Variant::OBJECT, PROPERTY_HINT_RESOURCE_TYPE, "GLTFDocumentExtension"),
						 PROPERTY_USAGE_DEFAULT),
			"set_extensions", "get_extensions");
}

void GLTFDocument::_build_parent_hierachy(Ref<GLTFState> state) {
	// build the hierarchy
	for (GLTFNodeIndex node_i = 0; node_i < state->nodes.size(); node_i++) {
		for (int j = 0; j < state->nodes[node_i]->children.size(); j++) {
			GLTFNodeIndex child_i = state->nodes[node_i]->children[j];
			ERR_FAIL_INDEX(child_i, state->nodes.size());
			if (state->nodes.write[child_i]->parent != -1) {
				continue;
			}
			state->nodes.write[child_i]->parent = node_i;
		}
	}
}

void GLTFDocument::set_extensions(TypedArray<GLTFDocumentExtension> p_extensions) {
	document_extensions = p_extensions;
}

TypedArray<GLTFDocumentExtension> GLTFDocument::get_extensions() const {
	return document_extensions;
}

GLTFDocument::GLTFDocument() {
	bool is_editor = ::Engine::get_singleton()->is_editor_hint();
	if (is_editor) {
		return;
	}
	Ref<GLTFDocumentExtensionConvertImporterMesh> extension_editor;
	extension_editor.instantiate();
	document_extensions.push_back(extension_editor);
}

PackedByteArray GLTFDocument::_serialize_glb_buffer(Ref<GLTFState> state, Error *r_err) {
	Error err = _encode_buffer_glb(state, "");
	if (r_err) {
		*r_err = err;
	}
	ERR_FAIL_COND_V(err != OK, PackedByteArray());
	String json = Variant(state->json).to_json_string();

	const uint32_t magic = 0x46546C67; // GLTF
	const int32_t header_size = 12;
	const int32_t chunk_header_size = 8;

	for (int32_t pad_i = 0; pad_i < (chunk_header_size + json.utf8().length()) % 4; pad_i++) {
		json += " ";
	}
	CharString cs = json.utf8();
	const uint32_t text_chunk_length = cs.length();

	const uint32_t text_chunk_type = 0x4E4F534A; //JSON
	int32_t binary_data_length = 0;
	if (state->buffers.size()) {
		binary_data_length = state->buffers[0].size();
	}
	const int32_t binary_chunk_length = binary_data_length;
	const int32_t binary_chunk_type = 0x004E4942; //BIN

	Ref<StreamPeerBuffer> buffer;
	buffer.instantiate();
	buffer->put_32(magic);
	buffer->put_32(state->major_version); // version
	buffer->put_32(header_size + chunk_header_size + text_chunk_length + chunk_header_size + binary_data_length); // length
	buffer->put_32(text_chunk_length);
	buffer->put_32(text_chunk_type);
	buffer->put_data((uint8_t *)&cs[0], cs.length());
	if (binary_chunk_length) {
		buffer->put_32(binary_chunk_length);
		buffer->put_32(binary_chunk_type);
		buffer->put_data(state->buffers[0].ptr(), binary_data_length);
	}
	return buffer->get_data_array();
}

PackedByteArray GLTFDocument::generate_buffer(Ref<GLTFState> state) {
	ERR_FAIL_NULL_V(state, PackedByteArray());
	Error err = _serialize(state, "");
	ERR_FAIL_COND_V(err != OK, PackedByteArray());
	PackedByteArray bytes = _serialize_glb_buffer(state, &err);
	return bytes;
}

Error GLTFDocument::write_to_filesystem(Ref<GLTFState> state, const String &p_path) {
	ERR_FAIL_NULL_V(state, ERR_INVALID_PARAMETER);

	Error err = _serialize(state, p_path);
	if (err != OK) {
		return err;
	}

	err = _serialize_file(state, p_path);
	if (err != OK) {
		return Error::FAILED;
	}
	return OK;
}

Node *GLTFDocument::generate_scene(Ref<GLTFState> state, int32_t p_bake_fps) {
	ERR_FAIL_NULL_V(state, nullptr);
	ERR_FAIL_INDEX_V(0, state->root_nodes.size(), nullptr);
	GLTFNodeIndex gltf_root = state->root_nodes.write[0];
	Node *gltf_root_node = state->get_scene_node(gltf_root);
	Node *root = gltf_root_node->get_parent();
	ERR_FAIL_NULL_V(root, nullptr);
	_process_mesh_instances(state, root);
	if (state->animations.size()) {
		AnimationPlayer *ap = memnew(AnimationPlayer);
		root->add_child(ap, true);
		ap->set_owner(root);
		for (int i = 0; i < state->animations.size(); i++) {
			_import_animation(state, ap, i, p_bake_fps);
		}
	}

	for (int32_t ext_i = 0; ext_i < document_extensions.size(); ext_i++) {
		Ref<GLTFDocumentExtension> ext = document_extensions[ext_i];
		ERR_CONTINUE(ext.is_null());
		Error err = ext->import_post(this, root);
		ERR_FAIL_COND_V(err != OK, nullptr);
	}
	ERR_FAIL_NULL_V(root, nullptr);
	return root;
}

Error GLTFDocument::append_from_scene(Node *p_node, Ref<GLTFState> state, uint32_t p_flags, int32_t p_bake_fps) {
	ERR_FAIL_COND_V(state.is_null(), FAILED);
	state->use_named_skin_binds =
			p_flags & EditorSceneFormatImporter::IMPORT_USE_NAMED_SKIN_BINDS;

	_convert_scene_node(state, p_node, -1, -1);
	if (!state->buffers.size()) {
		state->buffers.push_back(Vector<uint8_t>());
	}

	/* STEP 1 CONVERT MESH INSTANCES */
	_convert_mesh_instances(state);

	/* STEP 2 CREATE SKINS */
	Error err = _serialize_skins(state);
	return err;
}

Error GLTFDocument::append_from_buffer(PackedByteArray p_bytes, String p_base_path, Ref<GLTFState> state, uint32_t p_flags, int32_t p_bake_fps) {
	ERR_FAIL_COND_V(state.is_null(), FAILED);
	// TODO Add missing texture and missing .bin file paths to r_missing_deps 2021-09-10 fire
	Error err = FAILED;
	state->use_named_skin_binds =
			p_flags & EditorSceneFormatImporter::IMPORT_USE_NAMED_SKIN_BINDS;
	FileAccessMemory *file_access = memnew(FileAccessMemory);
	file_access->open_custom(p_bytes.ptr(), p_bytes.size());
	err = _parse(state, p_base_path.get_base_dir(), file_access, p_bake_fps);
	ERR_FAIL_COND_V(err != OK, FAILED);
	return OK;
}

Error GLTFDocument::_parse_gltf_state(Ref<GLTFState> state, const String &p_search_path, float p_bake_fps) {
	Error err;

	/* PARSE EXTENSIONS */
	err = _parse_gltf_extensions(state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE SCENE */
	err = _parse_scenes(state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE NODES */
	err = _parse_nodes(state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE BUFFERS */
	err = _parse_buffers(state, p_search_path);

	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE BUFFER VIEWS */
	err = _parse_buffer_views(state);

	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE ACCESSORS */
	err = _parse_accessors(state);

	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE IMAGES */
	err = _parse_images(state, p_search_path);

	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE TEXTURES */
	err = _parse_textures(state);

	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE TEXTURES */
	err = _parse_materials(state);

	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE SKINS */
	err = _parse_skins(state);

	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* DETERMINE SKELETONS */
	err = _determine_skeletons(state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* CREATE SKELETONS */
	err = _create_skeletons(state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* CREATE SKINS */
	err = _create_skins(state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE MESHES (we have enough info now) */
	err = _parse_meshes(state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE LIGHTS */
	err = _parse_lights(state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE CAMERAS */
	err = _parse_cameras(state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE ANIMATIONS */
	err = _parse_animations(state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* ASSIGN SCENE NAMES */
	_assign_scene_names(state);

	Node3D *root = memnew(Node3D);
	for (int32_t root_i = 0; root_i < state->root_nodes.size(); root_i++) {
		_generate_scene_node(state, root, root, state->root_nodes[root_i]);
	}
	return OK;
}

Error GLTFDocument::append_from_file(String p_path, Ref<GLTFState> r_state, uint32_t p_flags, int32_t p_bake_fps) {
	// TODO Add missing texture and missing .bin file paths to r_missing_deps 2021-09-10 fire
	if (r_state == Ref<GLTFState>()) {
		r_state.instantiate();
	}
	r_state->filename = p_path.get_file().get_basename();
	r_state->use_named_skin_binds =
			p_flags & EditorSceneFormatImporter::IMPORT_USE_NAMED_SKIN_BINDS;
	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);
	ERR_FAIL_COND_V(err != OK, ERR_FILE_CANT_OPEN);
	ERR_FAIL_NULL_V(f, ERR_FILE_CANT_OPEN);

	err = _parse(r_state, p_path.get_base_dir(), f, p_bake_fps);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);
	return err;
}

Error GLTFDocument::_parse_gltf_extensions(Ref<GLTFState> state) {
	ERR_FAIL_NULL_V(state, ERR_PARSE_ERROR);
	if (state->json.has("extensionsRequired") && state->json["extensionsRequired"].get_type() == Variant::ARRAY) {
		Array extensions_required = state->json["extensionsRequired"];
		if (extensions_required.find("KHR_draco_mesh_compression") != -1) {
			ERR_PRINT("glTF2 extension KHR_draco_mesh_compression is not supported.");
			return ERR_UNAVAILABLE;
		}
	}
	return OK;
}

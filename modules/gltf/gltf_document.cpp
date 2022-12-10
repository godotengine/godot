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

#include "extensions/gltf_spec_gloss.h"
#include "gltf_state.h"

#include "core/bind/core_bind.h" // FIXME: Shouldn't use _Directory but DirAccess.
#include "core/crypto/crypto_core.h"
#include "core/io/json.h"
#include "core/math/disjoint_set.h"
#include "core/os/file_access.h"
#include "core/variant.h"
#include "core/version.h"
#include "drivers/png/png_driver_common.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/bone_attachment.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/multimesh_instance.h"
#include "scene/3d/spatial.h"
#include "scene/animation/animation_player.h"
#include "scene/main/node.h"
#include "scene/resources/surface_tool.h"

#include "modules/modules_enabled.gen.h" // For csg, gridmap, regex.

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif // MODULE_CSG_ENABLED
#ifdef MODULE_GRIDMAP_ENABLED
#include "modules/gridmap/grid_map.h"
#endif // MODULE_GRIDMAP_ENABLED
#ifdef MODULE_REGEX_ENABLED
#include "modules/regex/regex.h"
#endif // MODULE_REGEX_ENABLED

Ref<ArrayMesh> _mesh_to_array_mesh(Ref<Mesh> p_mesh) {
	Ref<ArrayMesh> array_mesh = p_mesh;
	if (array_mesh.is_valid()) {
		return array_mesh;
	}
	array_mesh.instance();
	if (p_mesh.is_null()) {
		return array_mesh;
	}

	for (int32_t surface_i = 0; surface_i < p_mesh->get_surface_count(); surface_i++) {
		Mesh::PrimitiveType primitive_type = p_mesh->surface_get_primitive_type(surface_i);
		Array arrays = p_mesh->surface_get_arrays(surface_i);
		Ref<Material> mat = p_mesh->surface_get_material(surface_i);
		int32_t mat_idx = array_mesh->get_surface_count();
		array_mesh->add_surface_from_arrays(primitive_type, arrays);
		array_mesh->surface_set_material(mat_idx, mat);
	}

	return array_mesh;
}

Error GLTFDocument::serialize(Ref<GLTFState> p_state, Node *p_root, const String &p_path) {
	uint64_t begin_time = OS::get_singleton()->get_ticks_usec();

	p_state->skeleton3d_to_gltf_skeleton.clear();
	p_state->skin_and_skeleton3d_to_gltf_skin.clear();

	_convert_scene_node(p_state, p_root, -1, -1);
	if (!p_state->buffers.size()) {
		p_state->buffers.push_back(Vector<uint8_t>());
	}

	/* STEP 1 CONVERT MESH INSTANCES */
	_convert_mesh_instances(p_state);

	/* STEP 2 SERIALIZE CAMERAS */
	Error err = _serialize_cameras(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 3 CREATE SKINS */
	err = _serialize_skins(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 5 SERIALIZE MESHES (we have enough info now) */
	err = _serialize_meshes(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 6 SERIALIZE TEXTURES */
	err = _serialize_materials(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 7 SERIALIZE TEXTURE SAMPLERS */
	err = _serialize_texture_samplers(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 8 SERIALIZE ANIMATIONS */
	err = _serialize_animations(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 9 SERIALIZE ACCESSORS */
	err = _encode_accessors(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 10 SERIALIZE IMAGES */
	err = _serialize_images(p_state, p_path);
	if (err != OK) {
		return Error::FAILED;
	}

	for (GLTFBufferViewIndex i = 0; i < p_state->buffer_views.size(); i++) {
		p_state->buffer_views.write[i]->buffer = 0;
	}

	/* STEP 12 SERIALIZE BUFFER VIEWS */
	err = _encode_buffer_views(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 13 SERIALIZE NODES */
	err = _serialize_nodes(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 15 SERIALIZE SCENE */
	err = _serialize_scenes(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 16 SERIALIZE SCENE */
	err = _serialize_lights(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 17 SERIALIZE EXTENSIONS */
	err = _serialize_extensions(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 18 SERIALIZE VERSION */
	err = _serialize_version(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 19 SERIALIZE FILE */
	err = _serialize_file(p_state, p_path);
	if (err != OK) {
		return Error::FAILED;
	}
	uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - begin_time;
	float elapsed_sec = double(elapsed) / 1000000.0;
	elapsed_sec = Math::stepify(elapsed_sec, 0.01f);
	print_line("glTF: Export time elapsed seconds " + rtos(elapsed_sec).pad_decimals(2));

	return OK;
}

Error GLTFDocument::_serialize_extensions(Ref<GLTFState> p_state) const {
	Array extensions_used;
	Array extensions_required;
	if (!p_state->lights.empty()) {
		extensions_used.push_back("KHR_lights_punctual");
	}
	if (p_state->use_khr_texture_transform) {
		extensions_used.push_back("KHR_texture_transform");
		extensions_required.push_back("KHR_texture_transform");
	}
	if (!extensions_used.empty()) {
		p_state->json["extensionsUsed"] = extensions_used;
	}
	if (!extensions_required.empty()) {
		p_state->json["extensionsRequired"] = extensions_required;
	}
	return OK;
}

Error GLTFDocument::_serialize_scenes(Ref<GLTFState> p_state) {
	Array scenes;
	const int loaded_scene = 0;
	p_state->json["scene"] = loaded_scene;

	if (p_state->nodes.size()) {
		Dictionary s;
		if (!p_state->scene_name.empty()) {
			s["name"] = p_state->scene_name;
		}

		Array nodes;
		nodes.push_back(0);
		s["nodes"] = nodes;
		scenes.push_back(s);
	}
	p_state->json["scenes"] = scenes;

	return OK;
}

Error GLTFDocument::_parse_json(const String &p_path, Ref<GLTFState> p_state) {
	Error err;
	FileAccessRef file = FileAccess::open(p_path, FileAccess::READ, &err);
	if (!file) {
		return err;
	}

	Vector<uint8_t> array;
	array.resize(file->get_len());
	file->get_buffer(array.ptrw(), array.size());
	String text;
	text.parse_utf8((const char *)array.ptr(), array.size());

	String err_txt;
	int err_line;
	Variant v;
	err = JSON::parse(text, v, err_txt, err_line);
	if (err != OK) {
		_err_print_error("", p_path.utf8().get_data(), err_line, err_txt.utf8().get_data(), ERR_HANDLER_SCRIPT);
		return err;
	}
	p_state->json = v;

	return OK;
}

Error GLTFDocument::_parse_glb(const String &p_path, Ref<GLTFState> p_state) {
	Error err;
	FileAccessRef file = FileAccess::open(p_path, FileAccess::READ, &err);
	if (!file) {
		return err;
	}

	uint32_t magic = file->get_32();
	ERR_FAIL_COND_V(magic != 0x46546C67, ERR_FILE_UNRECOGNIZED); //glTF
	file->get_32(); // version
	file->get_32(); // length

	uint32_t chunk_length = file->get_32();
	uint32_t chunk_type = file->get_32();

	ERR_FAIL_COND_V(chunk_type != 0x4E4F534A, ERR_PARSE_ERROR); //JSON
	Vector<uint8_t> json_data;
	json_data.resize(chunk_length);
	uint32_t len = file->get_buffer(json_data.ptrw(), chunk_length);
	ERR_FAIL_COND_V(len != chunk_length, ERR_FILE_CORRUPT);

	String text;
	text.parse_utf8((const char *)json_data.ptr(), json_data.size());

	String err_txt;
	int err_line;
	Variant v;
	err = JSON::parse(text, v, err_txt, err_line);
	if (err != OK) {
		_err_print_error("", p_path.utf8().get_data(), err_line, err_txt.utf8().get_data(), ERR_HANDLER_SCRIPT);
		return err;
	}

	p_state->json = v;

	//data?

	chunk_length = file->get_32();
	chunk_type = file->get_32();

	if (file->eof_reached()) {
		return OK; //all good
	}

	ERR_FAIL_COND_V(chunk_type != 0x004E4942, ERR_PARSE_ERROR); //BIN

	p_state->glb_data.resize(chunk_length);
	len = file->get_buffer(p_state->glb_data.ptrw(), chunk_length);
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

static Array _quat_to_array(const Quat &p_quat) {
	Array array;
	array.resize(4);
	array[0] = p_quat.x;
	array[1] = p_quat.y;
	array[2] = p_quat.z;
	array[3] = p_quat.w;
	return array;
}

static Quat _arr_to_quat(const Array &p_array) {
	ERR_FAIL_COND_V(p_array.size() != 4, Quat());
	return Quat(p_array[0], p_array[1], p_array[2], p_array[3]);
}

static Transform _arr_to_xform(const Array &p_array) {
	ERR_FAIL_COND_V(p_array.size() != 16, Transform());

	Transform xform;
	xform.basis.set_axis(Vector3::AXIS_X, Vector3(p_array[0], p_array[1], p_array[2]));
	xform.basis.set_axis(Vector3::AXIS_Y, Vector3(p_array[4], p_array[5], p_array[6]));
	xform.basis.set_axis(Vector3::AXIS_Z, Vector3(p_array[8], p_array[9], p_array[10]));
	xform.set_origin(Vector3(p_array[12], p_array[13], p_array[14]));

	return xform;
}

static Vector<real_t> _xform_to_array(const Transform p_transform) {
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

Error GLTFDocument::_serialize_nodes(Ref<GLTFState> p_state) {
	Array nodes;
	for (int i = 0; i < p_state->nodes.size(); i++) {
		Dictionary node;
		Ref<GLTFNode> gltf_node = p_state->nodes[i];
		Dictionary extensions;
		node["extensions"] = extensions;
		if (!gltf_node->get_name().empty()) {
			node["name"] = gltf_node->get_name();
		}
		if (gltf_node->camera != -1) {
			node["camera"] = gltf_node->camera;
		}
		if (gltf_node->light != -1) {
			Dictionary lights_punctual;
			extensions["KHR_lights_punctual"] = lights_punctual;
			lights_punctual["light"] = gltf_node->light;
		}
		if (gltf_node->mesh != -1) {
			node["mesh"] = gltf_node->mesh;
		}
		if (gltf_node->skin != -1) {
			node["skin"] = gltf_node->skin;
		}
		if (gltf_node->skeleton != -1 && gltf_node->skin < 0) {
		}
		if (gltf_node->xform != Transform()) {
			node["matrix"] = _xform_to_array(gltf_node->xform);
		}

		if (!gltf_node->rotation.is_equal_approx(Quat())) {
			node["rotation"] = _quat_to_array(gltf_node->rotation);
		}

		if (!gltf_node->scale.is_equal_approx(Vector3(1.0f, 1.0f, 1.0f))) {
			node["scale"] = _vec3_to_arr(gltf_node->scale);
		}

		if (!gltf_node->translation.is_equal_approx(Vector3())) {
			node["translation"] = _vec3_to_arr(gltf_node->translation);
		}
		if (gltf_node->children.size()) {
			Array children;
			for (int j = 0; j < gltf_node->children.size(); j++) {
				children.push_back(gltf_node->children[j]);
			}
			node["children"] = children;
		}
		nodes.push_back(node);
	}
	p_state->json["nodes"] = nodes;
	return OK;
}

String GLTFDocument::_sanitize_scene_name(Ref<GLTFState> p_state, const String &p_name) {
	if (p_state->use_legacy_names) {
#ifdef MODULE_REGEX_ENABLED
		RegEx regex("([^a-zA-Z0-9_ -]+)");
		String s_name = regex.sub(p_name, "", true);
		return s_name;
#else
		WARN_PRINT("GLTF: Legacy scene names are not supported without the RegEx module. Falling back to new names.");
#endif // MODULE_REGEX_ENABLED
	}
	return p_name.validate_node_name();
}

String GLTFDocument::_legacy_validate_node_name(const String &p_name) {
	String invalid_character = ". : @ / \"";
	String name = p_name;
	Vector<String> chars = invalid_character.split(" ");
	for (int i = 0; i < chars.size(); i++) {
		name = name.replace(chars[i], "");
	}
	return name;
}

String GLTFDocument::_gen_unique_name(Ref<GLTFState> p_state, const String &p_name) {
	const String s_name = _sanitize_scene_name(p_state, p_name);

	String name;
	int index = 1;
	while (true) {
		name = s_name;

		if (index > 1) {
			if (p_state->use_legacy_names) {
				name += " ";
			}
			name += itos(index);
		}
		if (!p_state->unique_names.has(name)) {
			break;
		}
		index++;
	}

	p_state->unique_names.insert(name);

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

String GLTFDocument::_gen_unique_animation_name(Ref<GLTFState> p_state, const String &p_name) {
	const String s_name = _sanitize_animation_name(p_name);

	String name;
	int index = 1;
	while (true) {
		name = s_name;

		if (index > 1) {
			name += itos(index);
		}
		if (!p_state->unique_animation_names.has(name)) {
			break;
		}
		index++;
	}

	p_state->unique_animation_names.insert(name);

	return name;
}

String GLTFDocument::_sanitize_bone_name(Ref<GLTFState> p_state, const String &p_name) {
	if (p_state->use_legacy_names) {
#ifdef MODULE_REGEX_ENABLED
		String name = p_name.camelcase_to_underscore(true);
		RegEx pattern_del("([^a-zA-Z0-9_ ])+");

		name = pattern_del.sub(name, "", true);

		RegEx pattern_nospace(" +");
		name = pattern_nospace.sub(name, "_", true);

		RegEx pattern_multiple("_+");
		name = pattern_multiple.sub(name, "_", true);

		RegEx pattern_padded("0+(\\d+)");
		name = pattern_padded.sub(name, "$1", true);

		return name;
#else
		WARN_PRINT("GLTF: Legacy bone names are not supported without the RegEx module. Falling back to new names.");
#endif // MODULE_REGEX_ENABLED
	}
	String name = p_name;
	name = name.replace(":", "_");
	name = name.replace("/", "_");
	if (name.empty()) {
		name = "bone";
	}
	return name;
}

String GLTFDocument::_gen_unique_bone_name(Ref<GLTFState> p_state, const GLTFSkeletonIndex p_skel_i, const String &p_name) {
	String s_name = _sanitize_bone_name(p_state, p_name);
	String name;
	int index = 1;
	while (true) {
		name = s_name;

		if (index > 1) {
			name += "_" + itos(index);
		}
		if (!p_state->skeletons[p_skel_i]->unique_names.has(name)) {
			break;
		}
		index++;
	}

	p_state->skeletons.write[p_skel_i]->unique_names.insert(name);

	return name;
}

Error GLTFDocument::_parse_scenes(Ref<GLTFState> p_state) {
	ERR_FAIL_COND_V(!p_state->json.has("scenes"), ERR_FILE_CORRUPT);
	const Array &scenes = p_state->json["scenes"];
	int loaded_scene = 0;
	if (p_state->json.has("scene")) {
		loaded_scene = p_state->json["scene"];
	} else {
		WARN_PRINT("The load-time scene is not defined in the glTF2 file. Picking the first scene.");
	}

	if (scenes.size()) {
		ERR_FAIL_COND_V(loaded_scene >= scenes.size(), ERR_FILE_CORRUPT);
		const Dictionary &s = scenes[loaded_scene];
		ERR_FAIL_COND_V(!s.has("nodes"), ERR_UNAVAILABLE);
		const Array &nodes = s["nodes"];
		for (int j = 0; j < nodes.size(); j++) {
			p_state->root_nodes.push_back(nodes[j]);
		}

		if (s.has("name") && !String(s["name"]).empty() && !((String)s["name"]).begins_with("Scene")) {
			p_state->scene_name = s["name"];
		} else {
			p_state->scene_name = p_state->filename;
		}
	}

	return OK;
}

Error GLTFDocument::_parse_nodes(Ref<GLTFState> p_state) {
	ERR_FAIL_COND_V(!p_state->json.has("nodes"), ERR_FILE_CORRUPT);
	const Array &nodes = p_state->json["nodes"];
	for (int i = 0; i < nodes.size(); i++) {
		Ref<GLTFNode> node;
		node.instance();
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
				node->translation = _arr_to_vec3(n["translation"]);
			}
			if (n.has("rotation")) {
				node->rotation = _arr_to_quat(n["rotation"]);
			}
			if (n.has("scale")) {
				node->scale = _arr_to_vec3(n["scale"]);
			}

			node->xform.basis.set_quat_scale(node->rotation, node->scale);
			node->xform.origin = node->translation;
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

	_compute_node_heights(p_state);

	return OK;
}

void GLTFDocument::_compute_node_heights(Ref<GLTFState> p_state) {
	p_state->root_nodes.clear();
	for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); ++node_i) {
		Ref<GLTFNode> node = p_state->nodes[node_i];
		node->height = 0;

		GLTFNodeIndex current_i = node_i;
		while (current_i >= 0) {
			const GLTFNodeIndex parent_i = p_state->nodes[current_i]->parent;
			if (parent_i >= 0) {
				++node->height;
			}
			current_i = parent_i;
		}

		if (node->height == 0) {
			p_state->root_nodes.push_back(node_i);
		}
	}
}

static Vector<uint8_t> _parse_base64_uri(const String &uri) {
	int start = uri.find(",");
	ERR_FAIL_COND_V(start == -1, Vector<uint8_t>());

	CharString substr = uri.right(start + 1).ascii();

	int strlen = substr.length();

	Vector<uint8_t> buf;
	buf.resize(strlen / 4 * 3 + 1 + 1);

	size_t len = 0;
	ERR_FAIL_COND_V(CryptoCore::b64_decode(buf.ptrw(), buf.size(), &len, (unsigned char *)substr.get_data(), strlen) != OK, Vector<uint8_t>());

	buf.resize(len);

	return buf;
}
Error GLTFDocument::_encode_buffer_glb(Ref<GLTFState> p_state, const String &p_path) {
	print_verbose("glTF: Total buffers: " + itos(p_state->buffers.size()));

	if (!p_state->buffers.size()) {
		return OK;
	}
	Array buffers;
	if (p_state->buffers.size()) {
		Vector<uint8_t> buffer_data = p_state->buffers[0];
		Dictionary gltf_buffer;

		gltf_buffer["byteLength"] = buffer_data.size();
		buffers.push_back(gltf_buffer);
	}

	for (GLTFBufferIndex i = 1; i < p_state->buffers.size() - 1; i++) {
		Vector<uint8_t> buffer_data = p_state->buffers[i];
		Dictionary gltf_buffer;
		String filename = p_path.get_basename().get_file() + itos(i) + ".bin";
		String path = p_path.get_base_dir() + "/" + filename;
		Error err;
		FileAccessRef file = FileAccess::open(path, FileAccess::WRITE, &err);
		if (!file) {
			return err;
		}
		if (buffer_data.size() == 0) {
			return OK;
		}
		file->create(FileAccess::ACCESS_RESOURCES);
		file->store_buffer(buffer_data.ptr(), buffer_data.size());
		file->close();
		gltf_buffer["uri"] = filename;
		gltf_buffer["byteLength"] = buffer_data.size();
		buffers.push_back(gltf_buffer);
	}
	if (!buffers.size()) {
		return OK;
	}
	p_state->json["buffers"] = buffers;

	return OK;
}

Error GLTFDocument::_encode_buffer_bins(Ref<GLTFState> p_state, const String &p_path) {
	print_verbose("glTF: Total buffers: " + itos(p_state->buffers.size()));

	if (!p_state->buffers.size()) {
		return OK;
	}
	Array buffers;

	for (GLTFBufferIndex i = 0; i < p_state->buffers.size(); i++) {
		Vector<uint8_t> buffer_data = p_state->buffers[i];
		Dictionary gltf_buffer;
		String filename = p_path.get_basename().get_file() + itos(i) + ".bin";
		String path = p_path.get_base_dir() + "/" + filename;
		Error err;
		FileAccessRef file = FileAccess::open(path, FileAccess::WRITE, &err);
		if (!file) {
			return err;
		}
		if (buffer_data.size() == 0) {
			return OK;
		}
		file->create(FileAccess::ACCESS_RESOURCES);
		file->store_buffer(buffer_data.ptr(), buffer_data.size());
		file->close();
		gltf_buffer["uri"] = filename;
		gltf_buffer["byteLength"] = buffer_data.size();
		buffers.push_back(gltf_buffer);
	}
	p_state->json["buffers"] = buffers;

	return OK;
}

Error GLTFDocument::_parse_buffers(Ref<GLTFState> p_state, const String &p_base_path) {
	if (!p_state->json.has("buffers")) {
		return OK;
	}

	const Array &buffers = p_state->json["buffers"];
	for (GLTFBufferIndex i = 0; i < buffers.size(); i++) {
		if (i == 0 && p_state->glb_data.size()) {
			p_state->buffers.push_back(p_state->glb_data);

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
					uri = uri.http_unescape();
					uri = p_base_path.plus_file(uri).replace("\\", "/"); // Fix for Windows.
					buffer_data = FileAccess::get_file_as_array(uri);
					ERR_FAIL_COND_V_MSG(buffer.size() == 0, ERR_PARSE_ERROR, "glTF: Couldn't load binary file as an array: " + uri);
				}

				ERR_FAIL_COND_V(!buffer.has("byteLength"), ERR_PARSE_ERROR);
				int byteLength = buffer["byteLength"];
				ERR_FAIL_COND_V(byteLength < buffer_data.size(), ERR_PARSE_ERROR);
				p_state->buffers.push_back(buffer_data);
			}
		}
	}

	print_verbose("glTF: Total buffers: " + itos(p_state->buffers.size()));

	return OK;
}

Error GLTFDocument::_encode_buffer_views(Ref<GLTFState> p_state) {
	Array buffers;
	for (GLTFBufferViewIndex i = 0; i < p_state->buffer_views.size(); i++) {
		Dictionary d;

		Ref<GLTFBufferView> buffer_view = p_state->buffer_views[i];

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
	print_verbose("glTF: Total buffer views: " + itos(p_state->buffer_views.size()));
	if (!buffers.size()) {
		return OK;
	}
	p_state->json["bufferViews"] = buffers;
	return OK;
}

Error GLTFDocument::_parse_buffer_views(Ref<GLTFState> p_state) {
	if (!p_state->json.has("bufferViews")) {
		return OK;
	}
	const Array &buffers = p_state->json["bufferViews"];
	for (GLTFBufferViewIndex i = 0; i < buffers.size(); i++) {
		const Dictionary &d = buffers[i];

		Ref<GLTFBufferView> buffer_view;
		buffer_view.instance();

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

		p_state->buffer_views.push_back(buffer_view);
	}

	print_verbose("glTF: Total buffer views: " + itos(p_state->buffer_views.size()));

	return OK;
}

Error GLTFDocument::_encode_accessors(Ref<GLTFState> p_state) {
	Array accessors;
	for (GLTFAccessorIndex i = 0; i < p_state->accessors.size(); i++) {
		Dictionary d;

		Ref<GLTFAccessor> accessor = p_state->accessors[i];
		d["componentType"] = accessor->component_type;
		d["count"] = accessor->count;
		d["type"] = _get_accessor_type_name(accessor->type);
		d["byteOffset"] = accessor->byte_offset;
		d["normalized"] = accessor->normalized;
		Array max;
		max.resize(accessor->max.size());
		for (int32_t max_i = 0; max_i < max.size(); max_i++) {
			max[max_i] = accessor->max[max_i];
		}
		d["max"] = max;
		Array min;
		min.resize(accessor->min.size());
		for (int32_t min_i = 0; min_i < min.size(); min_i++) {
			min[min_i] = accessor->min[min_i];
		}
		d["min"] = min;
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
	p_state->json["accessors"] = accessors;
	ERR_FAIL_COND_V(!p_state->json.has("accessors"), ERR_FILE_CORRUPT);
	print_verbose("glTF: Total accessors: " + itos(p_state->accessors.size()));

	return OK;
}

String GLTFDocument::_get_accessor_type_name(const GLTFType p_type) {
	if (p_type == GLTFType::TYPE_SCALAR) {
		return "SCALAR";
	}
	if (p_type == GLTFType::TYPE_VEC2) {
		return "VEC2";
	}
	if (p_type == GLTFType::TYPE_VEC3) {
		return "VEC3";
	}
	if (p_type == GLTFType::TYPE_VEC4) {
		return "VEC4";
	}

	if (p_type == GLTFType::TYPE_MAT2) {
		return "MAT2";
	}
	if (p_type == GLTFType::TYPE_MAT3) {
		return "MAT3";
	}
	if (p_type == GLTFType::TYPE_MAT4) {
		return "MAT4";
	}
	ERR_FAIL_V("SCALAR");
}

GLTFType GLTFDocument::_get_type_from_str(const String &p_string) {
	if (p_string == "SCALAR") {
		return GLTFType::TYPE_SCALAR;
	}

	if (p_string == "VEC2") {
		return GLTFType::TYPE_VEC2;
	}
	if (p_string == "VEC3") {
		return GLTFType::TYPE_VEC3;
	}
	if (p_string == "VEC4") {
		return GLTFType::TYPE_VEC4;
	}

	if (p_string == "MAT2") {
		return GLTFType::TYPE_MAT2;
	}
	if (p_string == "MAT3") {
		return GLTFType::TYPE_MAT3;
	}
	if (p_string == "MAT4") {
		return GLTFType::TYPE_MAT4;
	}

	ERR_FAIL_V(GLTFType::TYPE_SCALAR);
}

Error GLTFDocument::_parse_accessors(Ref<GLTFState> p_state) {
	if (!p_state->json.has("accessors")) {
		return OK;
	}
	const Array &accessors = p_state->json["accessors"];
	for (GLTFAccessorIndex i = 0; i < accessors.size(); i++) {
		const Dictionary &d = accessors[i];

		Ref<GLTFAccessor> accessor;
		accessor.instance();

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
			Array max = d["max"];
			accessor->max.resize(max.size());
			PoolVector<float>::Write max_write = accessor->max.write();
			for (int32_t max_i = 0; max_i < accessor->max.size(); max_i++) {
				max_write[max_i] = max[max_i];
			}
		}

		if (d.has("min")) {
			Array min = d["min"];
			accessor->min.resize(min.size());
			PoolVector<float>::Write min_write = accessor->min.write();
			for (int32_t min_i = 0; min_i < accessor->min.size(); min_i++) {
				min_write[min_i] = min[min_i];
			}
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

		p_state->accessors.push_back(accessor);
	}

	print_verbose("glTF: Total accessors: " + itos(p_state->accessors.size()));

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

Error GLTFDocument::_encode_buffer_view(Ref<GLTFState> p_state, const double *p_src, const int p_count, const GLTFType p_type, const int p_component_type, const bool p_normalized, const int p_byte_offset, const bool p_for_vertex, GLTFBufferViewIndex &r_accessor) {
	const int component_count_for_type[7] = {
		1, 2, 3, 4, 4, 9, 16
	};

	const int component_count = component_count_for_type[p_type];
	const int component_size = _get_component_type_size(p_component_type);
	ERR_FAIL_COND_V(component_size == 0, FAILED);

	int skip_every = 0;
	int skip_bytes = 0;
	//special case of alignments, as described in spec
	switch (p_component_type) {
		case COMPONENT_TYPE_BYTE:
		case COMPONENT_TYPE_UNSIGNED_BYTE: {
			if (p_type == TYPE_MAT2) {
				skip_every = 2;
				skip_bytes = 2;
			}
			if (p_type == TYPE_MAT3) {
				skip_every = 3;
				skip_bytes = 1;
			}
		} break;
		case COMPONENT_TYPE_SHORT:
		case COMPONENT_TYPE_UNSIGNED_SHORT: {
			if (p_type == TYPE_MAT3) {
				skip_every = 6;
				skip_bytes = 4;
			}
		} break;
		default: {
		}
	}

	Ref<GLTFBufferView> bv;
	bv.instance();
	const uint32_t offset = bv->byte_offset = p_byte_offset;
	Vector<uint8_t> &gltf_buffer = p_state->buffers.write[0];

	int stride = _get_component_type_size(p_component_type);
	if (p_for_vertex && stride % 4) {
		stride += 4 - (stride % 4); //according to spec must be multiple of 4
	}
	//use to debug
	print_verbose("glTF: encoding type " + _get_type_name(p_type) + " component type: " + _get_component_type_name(p_component_type) + " stride: " + itos(stride) + " amount " + itos(p_count));

	print_verbose("glTF: encoding accessor offset " + itos(p_byte_offset) + " view offset: " + itos(bv->byte_offset) + " total buffer len: " + itos(gltf_buffer.size()) + " view len " + itos(bv->byte_length));

	const int buffer_end = (stride * (p_count - 1)) + _get_component_type_size(p_component_type);
	// TODO define bv->byte_stride
	bv->byte_offset = gltf_buffer.size();

	switch (p_component_type) {
		case COMPONENT_TYPE_BYTE: {
			Vector<int8_t> buffer;
			buffer.resize(p_count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < p_count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *p_src;
					if (p_normalized) {
						buffer.write[dst_i] = d * 128.0;
					} else {
						buffer.write[dst_i] = d;
					}
					p_src++;
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
			buffer.resize(p_count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < p_count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *p_src;
					if (p_normalized) {
						buffer.write[dst_i] = d * 255.0;
					} else {
						buffer.write[dst_i] = d;
					}
					p_src++;
					dst_i++;
				}
			}
			gltf_buffer.append_array(buffer);
			bv->byte_length = buffer.size() * sizeof(uint8_t);
		} break;
		case COMPONENT_TYPE_SHORT: {
			Vector<int16_t> buffer;
			buffer.resize(p_count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < p_count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *p_src;
					if (p_normalized) {
						buffer.write[dst_i] = d * 32768.0;
					} else {
						buffer.write[dst_i] = d;
					}
					p_src++;
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
			buffer.resize(p_count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < p_count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *p_src;
					if (p_normalized) {
						buffer.write[dst_i] = d * 65535.0;
					} else {
						buffer.write[dst_i] = d;
					}
					p_src++;
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
			buffer.resize(p_count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < p_count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *p_src;
					buffer.write[dst_i] = d;
					p_src++;
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
			buffer.resize(p_count * component_count);
			int32_t dst_i = 0;
			for (int i = 0; i < p_count; i++) {
				for (int j = 0; j < component_count; j++) {
					if (skip_every && j > 0 && (j % skip_every) == 0) {
						dst_i += skip_bytes;
					}
					double d = *p_src;
					buffer.write[dst_i] = d;
					p_src++;
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
	r_accessor = bv->buffer = p_state->buffer_views.size();
	p_state->buffer_views.push_back(bv);
	return OK;
}

Error GLTFDocument::_decode_buffer_view(Ref<GLTFState> p_state, double *p_dst, const GLTFBufferViewIndex p_buffer_view, const int p_skip_every, const int p_skip_bytes, const int p_element_size, const int p_count, const GLTFType p_type, const int p_component_count, const int p_component_type, const int p_component_size, const bool p_normalized, const int p_byte_offset, const bool p_for_vertex) {
	const Ref<GLTFBufferView> bv = p_state->buffer_views[p_buffer_view];

	int stride = p_element_size;
	if (bv->byte_stride != -1) {
		stride = bv->byte_stride;
	}
	if (p_for_vertex && stride % 4) {
		stride += 4 - (stride % 4); //according to spec must be multiple of 4
	}

	ERR_FAIL_INDEX_V(bv->buffer, p_state->buffers.size(), ERR_PARSE_ERROR);

	const uint32_t offset = bv->byte_offset + p_byte_offset;
	Vector<uint8_t> buffer = p_state->buffers[bv->buffer]; //copy on write, so no performance hit
	const uint8_t *bufptr = buffer.ptr();

	//use to debug
	print_verbose("glTF: type " + _get_type_name(p_type) + " component type: " + _get_component_type_name(p_component_type) + " stride: " + itos(stride) + " amount " + itos(p_count));
	print_verbose("glTF: accessor offset " + itos(p_byte_offset) + " view offset: " + itos(bv->byte_offset) + " total buffer len: " + itos(buffer.size()) + " view len " + itos(bv->byte_length));

	const int buffer_end = (stride * (p_count - 1)) + p_element_size;
	ERR_FAIL_COND_V(buffer_end > bv->byte_length, ERR_PARSE_ERROR);

	ERR_FAIL_COND_V((int)(offset + buffer_end) > buffer.size(), ERR_PARSE_ERROR);

	//fill everything as doubles

	for (int i = 0; i < p_count; i++) {
		const uint8_t *src = &bufptr[offset + i * stride];

		for (int j = 0; j < p_component_count; j++) {
			if (p_skip_every && j > 0 && (j % p_skip_every) == 0) {
				src += p_skip_bytes;
			}

			double d = 0;

			switch (p_component_type) {
				case COMPONENT_TYPE_BYTE: {
					int8_t b = int8_t(*src);
					if (p_normalized) {
						d = (double(b) / 128.0);
					} else {
						d = double(b);
					}
				} break;
				case COMPONENT_TYPE_UNSIGNED_BYTE: {
					uint8_t b = *src;
					if (p_normalized) {
						d = (double(b) / 255.0);
					} else {
						d = double(b);
					}
				} break;
				case COMPONENT_TYPE_SHORT: {
					int16_t s = *(int16_t *)src;
					if (p_normalized) {
						d = (double(s) / 32768.0);
					} else {
						d = double(s);
					}
				} break;
				case COMPONENT_TYPE_UNSIGNED_SHORT: {
					uint16_t s = *(uint16_t *)src;
					if (p_normalized) {
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

			*p_dst++ = d;
			src += p_component_size;
		}
	}

	return OK;
}

int GLTFDocument::_get_component_type_size(const int p_component_type) {
	switch (p_component_type) {
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

Vector<double> GLTFDocument::_decode_accessor(Ref<GLTFState> p_state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	//spec, for reference:
	//https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#data-alignment

	ERR_FAIL_INDEX_V(p_accessor, p_state->accessors.size(), Vector<double>());

	const Ref<GLTFAccessor> a = p_state->accessors[p_accessor];

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
		ERR_FAIL_INDEX_V(a->buffer_view, p_state->buffer_views.size(), Vector<double>());

		const Error err = _decode_buffer_view(p_state, dst, a->buffer_view, skip_every, skip_bytes, element_size, a->count, a->type, component_count, a->component_type, component_size, a->normalized, a->byte_offset, p_for_vertex);
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

		Error err = _decode_buffer_view(p_state, indices.ptrw(), a->sparse_indices_buffer_view, 0, 0, indices_component_size, a->sparse_count, TYPE_SCALAR, 1, a->sparse_indices_component_type, indices_component_size, false, a->sparse_indices_byte_offset, false);
		if (err != OK) {
			return Vector<double>();
		}

		Vector<double> data;
		data.resize(component_count * a->sparse_count);
		err = _decode_buffer_view(p_state, data.ptrw(), a->sparse_values_buffer_view, skip_every, skip_bytes, element_size, a->sparse_count, a->type, component_count, a->component_type, component_size, a->normalized, a->sparse_values_byte_offset, p_for_vertex);
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

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_ints(Ref<GLTFState> p_state, const Vector<int32_t> p_attribs, const bool p_for_vertex) {
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
		attribs.write[i] = Math::stepify(p_attribs[i], 1.0);
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
	accessor.instance();
	GLTFBufferIndex buffer_view_i;
	int64_t size = p_state->buffers[0].size();
	const GLTFType type = GLTFType::TYPE_SCALAR;
	const int component_type = GLTFDocument::COMPONENT_TYPE_INT;

	PoolVector<float> max;
	max.resize(type_max.size());
	PoolVector<float>::Write write_max = max.write();
	for (int32_t max_i = 0; max_i < max.size(); max_i++) {
		write_max[max_i] = type_max[max_i];
	}
	accessor->max = max;
	PoolVector<float> min;
	min.resize(type_min.size());
	PoolVector<float>::Write write_min = min.write();
	for (int32_t min_i = 0; min_i < min.size(); min_i++) {
		write_min[min_i] = type_min[min_i];
	}
	accessor->min = min;
	accessor->normalized = false;
	accessor->count = ret_size;
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(p_state, attribs.ptr(), attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	p_state->accessors.push_back(accessor);
	return p_state->accessors.size() - 1;
}

Vector<int> GLTFDocument::_decode_accessor_as_ints(Ref<GLTFState> p_state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(p_state, p_accessor, p_for_vertex);
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

Vector<float> GLTFDocument::_decode_accessor_as_floats(Ref<GLTFState> p_state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(p_state, p_accessor, p_for_vertex);
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

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_vec2(Ref<GLTFState> p_state, const Vector<Vector2> p_attribs, const bool p_for_vertex) {
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
		attribs.write[(i * element_count) + 0] = Math::stepify(attrib.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::stepify(attrib.y, CMP_NORMALIZE_TOLERANCE);
		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}

	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instance();
	GLTFBufferIndex buffer_view_i;
	int64_t size = p_state->buffers[0].size();
	const GLTFType type = GLTFType::TYPE_VEC2;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	PoolVector<float> max;
	max.resize(type_max.size());
	PoolVector<float>::Write write_max = max.write();
	for (int32_t max_i = 0; max_i < max.size(); max_i++) {
		write_max[max_i] = type_max[max_i];
	}
	accessor->max = max;
	PoolVector<float> min;
	min.resize(type_min.size());
	PoolVector<float>::Write write_min = min.write();
	for (int32_t min_i = 0; min_i < min.size(); min_i++) {
		write_min[min_i] = type_min[min_i];
	}
	accessor->min = min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(p_state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	p_state->accessors.push_back(accessor);
	return p_state->accessors.size() - 1;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_color(Ref<GLTFState> p_state, const Vector<Color> p_attribs, const bool p_for_vertex) {
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
		attribs.write[(i * element_count) + 0] = Math::stepify(attrib.r, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::stepify(attrib.g, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 2] = Math::stepify(attrib.b, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 3] = Math::stepify(attrib.a, CMP_NORMALIZE_TOLERANCE);

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}

	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instance();
	GLTFBufferIndex buffer_view_i;
	int64_t size = p_state->buffers[0].size();
	const GLTFType type = GLTFType::TYPE_VEC4;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;
	PoolVector<float> max;
	max.resize(type_max.size());
	PoolVector<float>::Write write_max = max.write();
	for (int32_t max_i = 0; max_i < max.size(); max_i++) {
		write_max[max_i] = type_max[max_i];
	}
	accessor->max = max;
	PoolVector<float> min;
	min.resize(type_min.size());
	PoolVector<float>::Write write_min = min.write();
	for (int32_t min_i = 0; min_i < min.size(); min_i++) {
		write_min[min_i] = type_min[min_i];
	}
	accessor->min = min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(p_state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	p_state->accessors.push_back(accessor);
	return p_state->accessors.size() - 1;
}

void GLTFDocument::_calc_accessor_min_max(int p_i, const int p_element_count, Vector<double> &p_type_max, Vector<double> p_attribs, Vector<double> &p_type_min) {
	if (p_i == 0) {
		for (int32_t type_i = 0; type_i < p_element_count; type_i++) {
			p_type_max.write[type_i] = p_attribs[(p_i * p_element_count) + type_i];
			p_type_min.write[type_i] = p_attribs[(p_i * p_element_count) + type_i];
		}
	}
	for (int32_t type_i = 0; type_i < p_element_count; type_i++) {
		p_type_max.write[type_i] = MAX(p_attribs[(p_i * p_element_count) + type_i], p_type_max[type_i]);
		p_type_min.write[type_i] = MIN(p_attribs[(p_i * p_element_count) + type_i], p_type_min[type_i]);
		p_type_max.write[type_i] = _filter_number(p_type_max.write[type_i]);
		p_type_min.write[type_i] = _filter_number(p_type_min.write[type_i]);
	}
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_weights(Ref<GLTFState> p_state, const Vector<Color> p_attribs, const bool p_for_vertex) {
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
		attribs.write[(i * element_count) + 0] = Math::stepify(attrib.r, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::stepify(attrib.g, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 2] = Math::stepify(attrib.b, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 3] = Math::stepify(attrib.a, CMP_NORMALIZE_TOLERANCE);

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}

	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instance();
	GLTFBufferIndex buffer_view_i;
	int64_t size = p_state->buffers[0].size();
	const GLTFType type = GLTFType::TYPE_VEC4;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	PoolVector<float> max;
	max.resize(type_max.size());
	PoolVector<float>::Write write_max = max.write();
	for (int32_t max_i = 0; max_i < max.size(); max_i++) {
		write_max[max_i] = type_max[max_i];
	}
	accessor->max = max;
	PoolVector<float> min;
	min.resize(type_min.size());
	PoolVector<float>::Write write_min = min.write();
	for (int32_t min_i = 0; min_i < min.size(); min_i++) {
		write_min[min_i] = type_min[min_i];
	}
	accessor->min = min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(p_state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	p_state->accessors.push_back(accessor);
	return p_state->accessors.size() - 1;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_joints(Ref<GLTFState> p_state, const Vector<Color> p_attribs, const bool p_for_vertex) {
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
		attribs.write[(i * element_count) + 0] = Math::stepify(attrib.r, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::stepify(attrib.g, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 2] = Math::stepify(attrib.b, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 3] = Math::stepify(attrib.a, CMP_NORMALIZE_TOLERANCE);
		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}
	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instance();
	GLTFBufferIndex buffer_view_i;
	int64_t size = p_state->buffers[0].size();
	const GLTFType type = GLTFType::TYPE_VEC4;
	const int component_type = GLTFDocument::COMPONENT_TYPE_UNSIGNED_SHORT;

	PoolVector<float> max;
	max.resize(type_max.size());
	PoolVector<float>::Write write_max = max.write();
	for (int32_t max_i = 0; max_i < max.size(); max_i++) {
		write_max[max_i] = type_max[max_i];
	}
	accessor->max = max;
	PoolVector<float> min;
	min.resize(type_min.size());
	PoolVector<float>::Write write_min = min.write();
	for (int32_t min_i = 0; min_i < min.size(); min_i++) {
		write_min[min_i] = type_min[min_i];
	}
	accessor->min = min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(p_state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	p_state->accessors.push_back(accessor);
	return p_state->accessors.size() - 1;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_quats(Ref<GLTFState> p_state, const Vector<Quat> p_attribs, const bool p_for_vertex) {
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
		Quat quat = p_attribs[i];
		attribs.write[(i * element_count) + 0] = Math::stepify(quat.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::stepify(quat.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 2] = Math::stepify(quat.z, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 3] = Math::stepify(quat.w, CMP_NORMALIZE_TOLERANCE);

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}

	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instance();
	GLTFBufferIndex buffer_view_i;
	int64_t size = p_state->buffers[0].size();
	const GLTFType type = GLTFType::TYPE_VEC4;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	PoolVector<float> max;
	max.resize(type_max.size());
	PoolVector<float>::Write write_max = max.write();
	for (int32_t max_i = 0; max_i < max.size(); max_i++) {
		write_max[max_i] = type_max[max_i];
	}
	accessor->max = max;
	PoolVector<float> min;
	min.resize(type_min.size());
	PoolVector<float>::Write write_min = min.write();
	for (int32_t min_i = 0; min_i < min.size(); min_i++) {
		write_min[min_i] = type_min[min_i];
	}
	accessor->min = min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(p_state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	p_state->accessors.push_back(accessor);
	return p_state->accessors.size() - 1;
}

Vector<Vector2> GLTFDocument::_decode_accessor_as_vec2(Ref<GLTFState> p_state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(p_state, p_accessor, p_for_vertex);
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

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_floats(Ref<GLTFState> p_state, const Vector<real_t> p_attribs, const bool p_for_vertex) {
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
		attribs.write[i] = Math::stepify(p_attribs[i], CMP_NORMALIZE_TOLERANCE);

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}

	ERR_FAIL_COND_V(!attribs.size(), -1);

	Ref<GLTFAccessor> accessor;
	accessor.instance();
	GLTFBufferIndex buffer_view_i;
	int64_t size = p_state->buffers[0].size();
	const GLTFType type = GLTFType::TYPE_SCALAR;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	PoolVector<float> max;
	max.resize(type_max.size());
	PoolVector<float>::Write write_max = max.write();
	for (int32_t max_i = 0; max_i < max.size(); max_i++) {
		write_max[max_i] = type_max[max_i];
	}
	accessor->max = max;
	PoolVector<float> min;
	min.resize(type_min.size());
	PoolVector<float>::Write write_min = min.write();
	for (int32_t min_i = 0; min_i < min.size(); min_i++) {
		write_min[min_i] = type_min[min_i];
	}
	accessor->min = min;
	accessor->normalized = false;
	accessor->count = ret_size;
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(p_state, attribs.ptr(), attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	p_state->accessors.push_back(accessor);
	return p_state->accessors.size() - 1;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_vec3(Ref<GLTFState> p_state, const Vector<Vector3> p_attribs, const bool p_for_vertex) {
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
		attribs.write[(i * element_count) + 0] = Math::stepify(attrib.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 1] = Math::stepify(attrib.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[(i * element_count) + 2] = Math::stepify(attrib.z, CMP_NORMALIZE_TOLERANCE);

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}
	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instance();
	GLTFBufferIndex buffer_view_i;
	int64_t size = p_state->buffers[0].size();
	const GLTFType type = GLTFType::TYPE_VEC3;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	PoolVector<float> max;
	max.resize(type_max.size());
	PoolVector<float>::Write write_max = max.write();
	for (int32_t max_i = 0; max_i < max.size(); max_i++) {
		write_max[max_i] = type_max[max_i];
	}
	accessor->max = max;
	PoolVector<float> min;
	min.resize(type_min.size());
	PoolVector<float>::Write write_min = min.write();
	for (int32_t min_i = 0; min_i < min.size(); min_i++) {
		write_min[min_i] = type_min[min_i];
	}
	accessor->min = min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(p_state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	p_state->accessors.push_back(accessor);
	return p_state->accessors.size() - 1;
}

GLTFAccessorIndex GLTFDocument::_encode_accessor_as_xform(Ref<GLTFState> p_state, const Vector<Transform> p_attribs, const bool p_for_vertex) {
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
		Transform attrib = p_attribs[i];
		Basis basis = attrib.get_basis();
		Vector3 axis_0 = basis.get_axis(Vector3::AXIS_X);

		attribs.write[i * element_count + 0] = Math::stepify(axis_0.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 1] = Math::stepify(axis_0.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 2] = Math::stepify(axis_0.z, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 3] = 0.0;

		Vector3 axis_1 = basis.get_axis(Vector3::AXIS_Y);
		attribs.write[i * element_count + 4] = Math::stepify(axis_1.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 5] = Math::stepify(axis_1.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 6] = Math::stepify(axis_1.z, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 7] = 0.0;

		Vector3 axis_2 = basis.get_axis(Vector3::AXIS_Z);
		attribs.write[i * element_count + 8] = Math::stepify(axis_2.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 9] = Math::stepify(axis_2.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 10] = Math::stepify(axis_2.z, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 11] = 0.0;

		Vector3 origin = attrib.get_origin();
		attribs.write[i * element_count + 12] = Math::stepify(origin.x, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 13] = Math::stepify(origin.y, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 14] = Math::stepify(origin.z, CMP_NORMALIZE_TOLERANCE);
		attribs.write[i * element_count + 15] = 1.0;

		_calc_accessor_min_max(i, element_count, type_max, attribs, type_min);
	}
	ERR_FAIL_COND_V(attribs.size() % element_count != 0, -1);

	Ref<GLTFAccessor> accessor;
	accessor.instance();
	GLTFBufferIndex buffer_view_i;
	int64_t size = p_state->buffers[0].size();
	const GLTFType type = GLTFType::TYPE_MAT4;
	const int component_type = GLTFDocument::COMPONENT_TYPE_FLOAT;

	PoolVector<float> max;
	max.resize(type_max.size());
	PoolVector<float>::Write write_max = max.write();
	for (int32_t max_i = 0; max_i < max.size(); max_i++) {
		write_max[max_i] = type_max[max_i];
	}
	accessor->max = max;
	PoolVector<float> min;
	min.resize(type_min.size());
	PoolVector<float>::Write write_min = min.write();
	for (int32_t min_i = 0; min_i < min.size(); min_i++) {
		write_min[min_i] = type_min[min_i];
	}
	accessor->min = min;
	accessor->normalized = false;
	accessor->count = p_attribs.size();
	accessor->type = type;
	accessor->component_type = component_type;
	accessor->byte_offset = 0;
	Error err = _encode_buffer_view(p_state, attribs.ptr(), p_attribs.size(), type, component_type, accessor->normalized, size, p_for_vertex, buffer_view_i);
	if (err != OK) {
		return -1;
	}
	accessor->buffer_view = buffer_view_i;
	p_state->accessors.push_back(accessor);
	return p_state->accessors.size() - 1;
}

Vector<Vector3> GLTFDocument::_decode_accessor_as_vec3(Ref<GLTFState> p_state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(p_state, p_accessor, p_for_vertex);
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

Vector<Color> GLTFDocument::_decode_accessor_as_color(Ref<GLTFState> p_state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(p_state, p_accessor, p_for_vertex);
	Vector<Color> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	const int type = p_state->accessors[p_accessor]->type;
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
Vector<Quat> GLTFDocument::_decode_accessor_as_quat(Ref<GLTFState> p_state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(p_state, p_accessor, p_for_vertex);
	Vector<Quat> ret;

	if (attribs.size() == 0) {
		return ret;
	}

	ERR_FAIL_COND_V(attribs.size() % 4 != 0, ret);
	const double *attribs_ptr = attribs.ptr();
	const int ret_size = attribs.size() / 4;
	ret.resize(ret_size);
	{
		for (int i = 0; i < ret_size; i++) {
			ret.write[i] = Quat(attribs_ptr[i * 4 + 0], attribs_ptr[i * 4 + 1], attribs_ptr[i * 4 + 2], attribs_ptr[i * 4 + 3]).normalized();
		}
	}
	return ret;
}
Vector<Transform2D> GLTFDocument::_decode_accessor_as_xform2d(Ref<GLTFState> p_state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(p_state, p_accessor, p_for_vertex);
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

Vector<Basis> GLTFDocument::_decode_accessor_as_basis(Ref<GLTFState> p_state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(p_state, p_accessor, p_for_vertex);
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

Vector<Transform> GLTFDocument::_decode_accessor_as_xform(Ref<GLTFState> p_state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex) {
	const Vector<double> attribs = _decode_accessor(p_state, p_accessor, p_for_vertex);
	Vector<Transform> ret;

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

Error GLTFDocument::_serialize_meshes(Ref<GLTFState> p_state) {
	Array meshes;
	for (GLTFMeshIndex gltf_mesh_i = 0; gltf_mesh_i < p_state->meshes.size(); gltf_mesh_i++) {
		print_verbose("glTF: Serializing mesh: " + itos(gltf_mesh_i));
		Ref<ArrayMesh> import_mesh = p_state->meshes.write[gltf_mesh_i]->get_mesh();
		if (import_mesh.is_null()) {
			continue;
		}
		Array instance_materials = p_state->meshes.write[gltf_mesh_i]->get_instance_materials();
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
			Mesh::PrimitiveType primitive_type = import_mesh->surface_get_primitive_type(surface_i);
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

			Array array = import_mesh->surface_get_arrays(surface_i);
			Dictionary attributes;
			{
				Vector<Vector3> a = array[Mesh::ARRAY_VERTEX];
				ERR_FAIL_COND_V(!a.size(), ERR_INVALID_DATA);
				attributes["POSITION"] = _encode_accessor_as_vec3(p_state, a, true);
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
					attributes["TANGENT"] = _encode_accessor_as_color(p_state, attribs, true);
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
					attributes["NORMAL"] = _encode_accessor_as_vec3(p_state, attribs, true);
				}
			}
			{
				Vector<Vector2> a = array[Mesh::ARRAY_TEX_UV];
				if (a.size()) {
					attributes["TEXCOORD_0"] = _encode_accessor_as_vec2(p_state, a, true);
				}
			}
			{
				Vector<Vector2> a = array[Mesh::ARRAY_TEX_UV2];
				if (a.size()) {
					attributes["TEXCOORD_1"] = _encode_accessor_as_vec2(p_state, a, true);
				}
			}
			{
				Vector<Color> a = array[Mesh::ARRAY_COLOR];
				if (a.size()) {
					attributes["COLOR_0"] = _encode_accessor_as_color(p_state, a, true);
				}
			}
			Map<int, int> joint_i_to_bone_i;
			for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); node_i++) {
				GLTFSkinIndex skin_i = -1;
				if (p_state->nodes[node_i]->mesh == gltf_mesh_i) {
					skin_i = p_state->nodes[node_i]->skin;
				}
				if (skin_i != -1) {
					joint_i_to_bone_i = p_state->skins[skin_i]->joint_i_to_bone_i;
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
					attributes["JOINTS_0"] = _encode_accessor_as_joints(p_state, attribs, true);
				}
				ERR_FAIL_COND_V((a.size() / (JOINT_GROUP_SIZE * 2)) >= vertex_array.size(), FAILED);
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
					attributes["WEIGHTS_0"] = _encode_accessor_as_weights(p_state, attribs, true);
				} else if ((a.size() / (JOINT_GROUP_SIZE * 2)) >= vertex_array.size()) {
					int32_t vertex_count = vertex_array.size();
					Vector<Color> weights_0;
					weights_0.resize(vertex_count);
					Vector<Color> weights_1;
					weights_1.resize(vertex_count);
					int32_t weights_8_count = JOINT_GROUP_SIZE * 2;
					for (int32_t vertex_i = 0; vertex_i < vertex_count; vertex_i++) {
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
					attributes["WEIGHTS_0"] = _encode_accessor_as_weights(p_state, weights_0, true);
					attributes["WEIGHTS_1"] = _encode_accessor_as_weights(p_state, weights_1, true);
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
					primitive["indices"] = _encode_accessor_as_ints(p_state, mesh_indices, true);
				} else {
					if (primitive_type == Mesh::PRIMITIVE_TRIANGLES) {
						//generate indices because they need to be swapped for CW/CCW
						const Vector<Vector3> &vertices = array[Mesh::ARRAY_VERTEX];
						Ref<SurfaceTool> st;
						st.instance();
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
						primitive["indices"] = _encode_accessor_as_ints(p_state, generated_indices, true);
					}
				}
			}

			primitive["attributes"] = attributes;

			//blend shapes
			print_verbose("glTF: Mesh has targets");
			if (import_mesh->get_blend_shape_count()) {
				ArrayMesh::BlendShapeMode shape_mode = import_mesh->get_blend_shape_mode();
				Array array_morphs = import_mesh->surface_get_blend_shape_arrays(surface_i);
				for (int morph_i = 0; morph_i < array_morphs.size(); morph_i++) {
					Array array_morph = array_morphs[morph_i];
					Dictionary t;
					Vector<Vector3> varr = array_morph[Mesh::ARRAY_VERTEX];
					Array mesh_arrays = import_mesh->surface_get_arrays(surface_i);
					if (varr.size()) {
						Vector<Vector3> src_varr = array[Mesh::ARRAY_VERTEX];
						if (shape_mode == ArrayMesh::BlendShapeMode::BLEND_SHAPE_MODE_NORMALIZED) {
							const int max_idx = src_varr.size();
							for (int blend_i = 0; blend_i < max_idx; blend_i++) {
								varr.write[blend_i] = Vector3(varr[blend_i]) - src_varr[blend_i];
							}
						}

						t["POSITION"] = _encode_accessor_as_vec3(p_state, varr, true);
					}

					Vector<Vector3> narr = array_morph[Mesh::ARRAY_NORMAL];
					if (narr.size()) {
						t["NORMAL"] = _encode_accessor_as_vec3(p_state, narr, true);
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
						t["TANGENT"] = _encode_accessor_as_vec3(p_state, attribs, true);
					}
					targets.push_back(t);
				}
			}
			Variant v;
			if (surface_i < instance_materials.size()) {
				v = instance_materials.get(surface_i);
			}
			Ref<SpatialMaterial> mat = v;
			if (!mat.is_valid()) {
				mat = import_mesh->surface_get_material(surface_i);
			}
			if (mat.is_valid()) {
				Map<Ref<Material>, GLTFMaterialIndex>::Element *material_cache_i = p_state->material_cache.find(mat);
				if (material_cache_i && material_cache_i->get() != -1) {
					primitive["material"] = material_cache_i->get();
				} else {
					GLTFMaterialIndex mat_i = p_state->materials.size();
					p_state->materials.push_back(mat);
					primitive["material"] = mat_i;
					p_state->material_cache.insert(mat, mat_i);
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
			if (name_i < p_state->meshes.write[gltf_mesh_i]->get_blend_weights().size()) {
				weight = p_state->meshes.write[gltf_mesh_i]->get_blend_weights()[name_i];
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
	p_state->json["meshes"] = meshes;
	print_verbose("glTF: Total meshes: " + itos(meshes.size()));

	return OK;
}

Error GLTFDocument::_parse_meshes(Ref<GLTFState> p_state) {
	if (!p_state->json.has("meshes")) {
		return OK;
	}

	bool has_warned = false;
	Array meshes = p_state->json["meshes"];
	for (GLTFMeshIndex i = 0; i < meshes.size(); i++) {
		print_verbose("glTF: Parsing mesh: " + itos(i));
		Dictionary d = meshes[i];

		Ref<GLTFMesh> mesh;
		mesh.instance();
		bool has_vertex_color = false;

		ERR_FAIL_COND_V(!d.has("primitives"), ERR_PARSE_ERROR);

		Array primitives = d["primitives"];
		const Dictionary &extras = d.has("extras") ? (Dictionary)d["extras"] : Dictionary();
		Ref<ArrayMesh> import_mesh;
		import_mesh.instance();
		String mesh_name = "mesh";
		if (d.has("name") && !String(d["name"]).empty()) {
			mesh_name = d["name"];
		}
		import_mesh->set_name(_gen_unique_name(p_state, vformat("%s_%s", p_state->scene_name, mesh_name)));

		for (int j = 0; j < primitives.size(); j++) {
			Dictionary p = primitives[j];

			Array array;
			array.resize(Mesh::ARRAY_MAX);

			ERR_FAIL_COND_V(!p.has("attributes"), ERR_PARSE_ERROR);

			Dictionary a = p["attributes"];

			Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;
			if (p.has("mode")) {
				const int mode = p["mode"];
				ERR_FAIL_INDEX_V(mode, 7, ERR_FILE_CORRUPT);
				static const Mesh::PrimitiveType primitives2[7] = {
					Mesh::PRIMITIVE_POINTS,
					Mesh::PRIMITIVE_LINES,
					Mesh::PRIMITIVE_LINES, //loop not supported, should ce converted
					Mesh::PRIMITIVE_LINES,
					Mesh::PRIMITIVE_TRIANGLES,
					Mesh::PRIMITIVE_TRIANGLE_STRIP,
					Mesh::PRIMITIVE_TRIANGLES, //fan not supported, should be converted
#ifndef _MSC_VER
// #warning line loop and triangle fan are not supported and need to be converted to lines and triangles
#endif

				};

				primitive = primitives2[mode];
			}

			ERR_FAIL_COND_V(!a.has("POSITION"), ERR_PARSE_ERROR);
			if (a.has("POSITION")) {
				array[Mesh::ARRAY_VERTEX] = _decode_accessor_as_vec3(p_state, a["POSITION"], true);
			}
			if (a.has("NORMAL")) {
				array[Mesh::ARRAY_NORMAL] = _decode_accessor_as_vec3(p_state, a["NORMAL"], true);
			}
			if (a.has("TANGENT")) {
				array[Mesh::ARRAY_TANGENT] = _decode_accessor_as_floats(p_state, a["TANGENT"], true);
			}
			if (a.has("TEXCOORD_0")) {
				array[Mesh::ARRAY_TEX_UV] = _decode_accessor_as_vec2(p_state, a["TEXCOORD_0"], true);
			}
			if (a.has("TEXCOORD_1")) {
				array[Mesh::ARRAY_TEX_UV2] = _decode_accessor_as_vec2(p_state, a["TEXCOORD_1"], true);
			}
			if (a.has("COLOR_0")) {
				array[Mesh::ARRAY_COLOR] = _decode_accessor_as_color(p_state, a["COLOR_0"], true);
				has_vertex_color = true;
			}
			if (a.has("JOINTS_0")) {
				array[Mesh::ARRAY_BONES] = _decode_accessor_as_ints(p_state, a["JOINTS_0"], true);
			}
			if (a.has("WEIGHTS_1") && a.has("JOINTS_1")) {
				if (!has_warned) {
					WARN_PRINT("glTF: Meshes use more than 4 bone joints");
					has_warned = true;
				}
			}
			if (a.has("WEIGHTS_0")) {
				Vector<float> weights = _decode_accessor_as_floats(p_state, a["WEIGHTS_0"], true);
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
			}

			if (p.has("indices")) {
				Vector<int> indices = _decode_accessor_as_ints(p_state, p["indices"], false);

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

			if (generate_tangents) {
				//must generate mikktspace tangents.. ergh..
				Ref<SurfaceTool> st;
				st.instance();
				st->create_from_triangle_arrays(array);
				st->generate_tangents();
				array = st->commit_to_arrays();
			}

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

					array_copy[Mesh::ARRAY_INDEX] = Variant();

					if (t.has("POSITION")) {
						Vector<Vector3> varr = _decode_accessor_as_vec3(p_state, t["POSITION"], true);
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
						Vector<Vector3> narr = _decode_accessor_as_vec3(p_state, t["NORMAL"], true);
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
						const Vector<Vector3> tangents_v3 = _decode_accessor_as_vec3(p_state, t["TANGENT"], true);
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

					if (generate_tangents) {
						Ref<SurfaceTool> st;
						st.instance();
						st->create_from_triangle_arrays(array_copy);
						st->deindex();
						st->generate_tangents();
						array_copy = st->commit_to_arrays();
					}

					morphs.push_back(array_copy);
				}
			}

			//just add it

			Ref<SpatialMaterial> mat;
			if (p.has("material")) {
				const int material = p["material"];
				ERR_FAIL_INDEX_V(material, p_state->materials.size(), ERR_FILE_CORRUPT);
				Ref<SpatialMaterial> mat3d = p_state->materials[material];
				if (has_vertex_color) {
					mat3d->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
				}
				mat = mat3d;

			} else if (has_vertex_color) {
				Ref<SpatialMaterial> mat3d;
				mat3d.instance();
				mat3d->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
				mat = mat3d;
			}
			int32_t mat_idx = import_mesh->get_surface_count();
			import_mesh->add_surface_from_arrays(primitive, array, morphs, p_state->compress_flags);
			import_mesh->surface_set_material(mat_idx, mat);
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

		p_state->meshes.push_back(mesh);
	}

	print_verbose("glTF: Total meshes: " + itos(p_state->meshes.size()));

	return OK;
}

Error GLTFDocument::_serialize_images(Ref<GLTFState> p_state, const String &p_path) {
	Array images;
	for (int i = 0; i < p_state->images.size(); i++) {
		Dictionary d;

		ERR_CONTINUE(p_state->images[i].is_null());

		Ref<Image> image = p_state->images[i];
		ERR_CONTINUE(image.is_null());

		if (p_path.to_lower().ends_with("glb")) {
			GLTFBufferViewIndex bvi;

			Ref<GLTFBufferView> bv;
			bv.instance();

			const GLTFBufferIndex bi = 0;
			bv->buffer = bi;
			bv->byte_offset = p_state->buffers[bi].size();
			ERR_FAIL_INDEX_V(bi, p_state->buffers.size(), ERR_PARAMETER_RANGE_ERROR);

			PoolVector<uint8_t> buffer;
			Ref<ImageTexture> img_tex = image;
			if (img_tex.is_valid()) {
				image = img_tex->get_data();
			}
			Error err = PNGDriverCommon::image_to_png(image, buffer);
			ERR_FAIL_COND_V_MSG(err, err, "Can't convert image to PNG.");

			bv->byte_length = buffer.size();
			p_state->buffers.write[bi].resize(p_state->buffers[bi].size() + bv->byte_length);
			memcpy(&p_state->buffers.write[bi].write[bv->byte_offset], buffer.read().ptr(), buffer.size());
			ERR_FAIL_COND_V(bv->byte_offset + bv->byte_length > p_state->buffers[bi].size(), ERR_FILE_CORRUPT);

			p_state->buffer_views.push_back(bv);
			bvi = p_state->buffer_views.size() - 1;
			d["bufferView"] = bvi;
			d["mimeType"] = "image/png";
		} else {
			String name = p_state->images[i]->get_name();
			if (name.empty()) {
				name = itos(i);
			}
			name = _gen_unique_name(p_state, name);
			name = name.pad_zeros(3);
			Ref<_Directory> dir;
			dir.instance();
			String texture_dir = "textures";
			String new_texture_dir = p_path.get_base_dir() + "/" + texture_dir;
			dir->open(p_path.get_base_dir());
			if (!dir->dir_exists(new_texture_dir)) {
				dir->make_dir(new_texture_dir);
			}
			name = name + ".png";
			image->save_png(new_texture_dir.plus_file(name));
			d["uri"] = texture_dir.plus_file(name);
		}
		images.push_back(d);
	}

	print_verbose("Total images: " + itos(p_state->images.size()));

	if (!images.size()) {
		return OK;
	}
	p_state->json["images"] = images;

	return OK;
}

Error GLTFDocument::_parse_images(Ref<GLTFState> p_state, const String &p_base_path) {
	if (!p_state->json.has("images")) {
		return OK;
	}

	// Ref: https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#images

	const Array &images = p_state->json["images"];
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
		ERR_CONTINUE_MSG(!d.has("uri") && !d.has("bufferView"), "Invalid image definition in glTF file, it should specify an 'uri' or 'bufferView'.");
		if (d.has("uri") && d.has("bufferView")) {
			WARN_PRINT("Invalid image definition in glTF file using both 'uri' and 'bufferView'. 'uri' will take precedence.");
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
					p_state->images.push_back(Ref<Image>()); // Placeholder to keep count.
					continue;
				}
				data = _parse_base64_uri(uri);
				data_ptr = data.ptr();
				data_size = data.size();
				// mimeType is optional, but if we have it defined in the URI, let's use it.
				if (mimetype.empty()) {
					if (uri.begins_with("data:image/png;base64")) {
						mimetype = "image/png";
					} else if (uri.begins_with("data:image/jpeg;base64")) {
						mimetype = "image/jpeg";
					}
				}
			} else { // Relative path to an external image file.
				uri = uri.http_unescape();
				uri = p_base_path.plus_file(uri).replace("\\", "/"); // Fix for Windows.
				// ResourceLoader will rely on the file extension to use the relevant loader.
				// The spec says that if mimeType is defined, it should take precedence (e.g.
				// there could be a `.png` image which is actually JPEG), but there's no easy
				// API for that in Godot, so we'd have to load as a buffer (i.e. embedded in
				// the material), so we do this only as fallback.
				Ref<Texture> texture = ResourceLoader::load(uri);
				if (texture.is_valid()) {
					p_state->images.push_back(texture->get_data());
					continue;
				} else if (mimetype == "image/png" || mimetype == "image/jpeg") {
					// Fallback to loading as byte array.
					// This enables us to support the spec's requirement that we honor mimetype
					// regardless of file URI.
					data = FileAccess::get_file_as_array(uri);
					if (data.size() == 0) {
						WARN_PRINT(vformat("glTF: Image index '%d' couldn't be loaded as a buffer of MIME type '%s' from URI: %s. Skipping it.", i, mimetype, uri));
						p_state->images.push_back(Ref<Image>()); // Placeholder to keep count.
						continue;
					}
					data_ptr = data.ptr();
					data_size = data.size();
				} else {
					WARN_PRINT(vformat("glTF: Image index '%d' couldn't be loaded from URI: %s. Skipping it.", i, uri));
					p_state->images.push_back(Ref<Image>()); // Placeholder to keep count.
					continue;
				}
			}
		} else if (d.has("bufferView")) {
			// Handles the third bullet point from the spec (bufferView).
			ERR_FAIL_COND_V_MSG(mimetype.empty(), ERR_FILE_CORRUPT,
					vformat("glTF: Image index '%d' specifies 'bufferView' but no 'mimeType', which is invalid.", i));

			const GLTFBufferViewIndex bvi = d["bufferView"];

			ERR_FAIL_INDEX_V(bvi, p_state->buffer_views.size(), ERR_PARAMETER_RANGE_ERROR);

			Ref<GLTFBufferView> bv = p_state->buffer_views[bvi];

			const GLTFBufferIndex bi = bv->buffer;
			ERR_FAIL_INDEX_V(bi, p_state->buffers.size(), ERR_PARAMETER_RANGE_ERROR);

			ERR_FAIL_COND_V(bv->byte_offset + bv->byte_length > p_state->buffers[bi].size(), ERR_FILE_CORRUPT);

			data_ptr = &p_state->buffers[bi][bv->byte_offset];
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
			p_state->images.push_back(Ref<Image>());
			continue;
		}

		p_state->images.push_back(img);
	}

	print_verbose("glTF: Total images: " + itos(p_state->images.size()));

	return OK;
}

Error GLTFDocument::_serialize_textures(Ref<GLTFState> p_state) {
	if (!p_state->textures.size()) {
		return OK;
	}

	Array textures;
	for (int32_t i = 0; i < p_state->textures.size(); i++) {
		Dictionary d;
		Ref<GLTFTexture> t = p_state->textures[i];
		ERR_CONTINUE(t->get_src_image() == -1);
		d["source"] = t->get_src_image();

		GLTFTextureSamplerIndex sampler_index = t->get_sampler();
		if (sampler_index != -1) {
			d["sampler"] = sampler_index;
		}
		textures.push_back(d);
	}
	p_state->json["textures"] = textures;

	return OK;
}

Error GLTFDocument::_parse_textures(Ref<GLTFState> p_state) {
	if (!p_state->json.has("textures")) {
		return OK;
	}

	const Array &textures = p_state->json["textures"];
	for (GLTFTextureIndex i = 0; i < textures.size(); i++) {
		const Dictionary &d = textures[i];

		ERR_FAIL_COND_V(!d.has("source"), ERR_PARSE_ERROR);

		Ref<GLTFTexture> t;
		t.instance();

		GLTFImageIndex gltf_src_image_i = d["source"];
		ERR_FAIL_INDEX_V(gltf_src_image_i, p_state->images.size(), ERR_PARSE_ERROR);

		t->set_src_image(gltf_src_image_i);
		if (d.has("sampler")) {
			t->set_sampler(d["sampler"]);
		} else {
			t->set_sampler(-1);
		}
		p_state->textures.push_back(t);

		// Create and cache the texture used in the engine
		Ref<ImageTexture> imgTex;
		imgTex.instance();
		imgTex->create_from_image(p_state->images[t->get_src_image()]);

		// Set texture filter and repeat based on sampler settings
		const Ref<GLTFTextureSampler> sampler = _get_sampler_for_texture(p_state, i);
		Texture::Flags flags = sampler->get_texture_flags();
		imgTex->set_flags(flags);

		p_state->texture_cache.insert(i, imgTex);
	}

	return OK;
}

GLTFTextureIndex GLTFDocument::_set_texture(Ref<GLTFState> p_state, Ref<Texture> p_texture) {
	ERR_FAIL_COND_V(p_texture.is_null(), -1);
	ERR_FAIL_COND_V(p_texture->get_data().is_null(), -1);

	// Create GLTF data structures for the new texture
	Ref<GLTFTexture> gltf_texture;
	gltf_texture.instance();
	GLTFImageIndex gltf_src_image_i = p_state->images.size();

	p_state->images.push_back(p_texture->get_data());

	GLTFTextureSamplerIndex gltf_sampler_i = _set_sampler_for_mode(p_state, p_texture->get_flags());

	gltf_texture->set_src_image(gltf_src_image_i);
	gltf_texture->set_sampler(gltf_sampler_i);

	GLTFTextureIndex gltf_texture_i = p_state->textures.size();
	p_state->textures.push_back(gltf_texture);
	p_state->texture_cache[gltf_texture_i] = p_texture;
	return gltf_texture_i;
}

Ref<Texture> GLTFDocument::_get_texture(Ref<GLTFState> p_state, const GLTFTextureIndex p_texture) {
	ERR_FAIL_INDEX_V(p_texture, p_state->textures.size(), Ref<Texture>());
	return p_state->texture_cache[p_texture];
}

GLTFTextureSamplerIndex GLTFDocument::_set_sampler_for_mode(Ref<GLTFState> p_state, uint32_t p_mode) {
	for (int i = 0; i < p_state->texture_samplers.size(); ++i) {
		if (p_state->texture_samplers[i]->get_texture_flags() == p_mode) {
			return i;
		}
	}

	GLTFTextureSamplerIndex gltf_sampler_i = p_state->texture_samplers.size();
	Ref<GLTFTextureSampler> gltf_sampler;
	gltf_sampler.instance();
	gltf_sampler->set_texture_flags(p_mode);
	p_state->texture_samplers.push_back(gltf_sampler);
	return gltf_sampler_i;
}

Ref<GLTFTextureSampler> GLTFDocument::_get_sampler_for_texture(Ref<GLTFState> p_state, const GLTFTextureIndex p_texture) {
	ERR_FAIL_INDEX_V(p_texture, p_state->textures.size(), p_state->default_texture_sampler);
	const GLTFTextureSamplerIndex sampler = p_state->textures[p_texture]->get_sampler();

	if (sampler == -1) {
		return p_state->default_texture_sampler;
	} else {
		ERR_FAIL_INDEX_V(sampler, p_state->texture_samplers.size(), p_state->default_texture_sampler);

		return p_state->texture_samplers[sampler];
	}
}

Error GLTFDocument::_serialize_texture_samplers(Ref<GLTFState> p_state) {
	if (!p_state->texture_samplers.size()) {
		return OK;
	}

	Array samplers;
	for (int32_t i = 0; i < p_state->texture_samplers.size(); ++i) {
		Dictionary d;
		Ref<GLTFTextureSampler> s = p_state->texture_samplers[i];
		d["magFilter"] = s->get_mag_filter();
		d["minFilter"] = s->get_min_filter();
		d["wrapS"] = s->get_wrap_s();
		d["wrapT"] = s->get_wrap_t();
		samplers.push_back(d);
	}
	p_state->json["samplers"] = samplers;

	return OK;
}

Error GLTFDocument::_parse_texture_samplers(Ref<GLTFState> p_state) {
	p_state->default_texture_sampler.instance();
	p_state->default_texture_sampler->set_min_filter(GLTFTextureSampler::FilterMode::LINEAR_MIPMAP_LINEAR);
	p_state->default_texture_sampler->set_mag_filter(GLTFTextureSampler::FilterMode::LINEAR);
	p_state->default_texture_sampler->set_wrap_s(GLTFTextureSampler::WrapMode::REPEAT);
	p_state->default_texture_sampler->set_wrap_t(GLTFTextureSampler::WrapMode::REPEAT);

	if (!p_state->json.has("samplers")) {
		return OK;
	}

	const Array &samplers = p_state->json["samplers"];
	for (int i = 0; i < samplers.size(); ++i) {
		const Dictionary &d = samplers[i];

		Ref<GLTFTextureSampler> sampler;
		sampler.instance();

		if (d.has("minFilter")) {
			sampler->set_min_filter(d["minFilter"]);
		} else {
			sampler->set_min_filter(GLTFTextureSampler::FilterMode::LINEAR_MIPMAP_LINEAR);
		}
		if (d.has("magFilter")) {
			sampler->set_mag_filter(d["magFilter"]);
		} else {
			sampler->set_mag_filter(GLTFTextureSampler::FilterMode::LINEAR);
		}

		if (d.has("wrapS")) {
			sampler->set_wrap_s(d["wrapS"]);
		} else {
			sampler->set_wrap_s(GLTFTextureSampler::WrapMode::REPEAT);
		}

		if (d.has("wrapT")) {
			sampler->set_wrap_t(d["wrapT"]);
		} else {
			sampler->set_wrap_t(GLTFTextureSampler::WrapMode::REPEAT);
		}

		p_state->texture_samplers.push_back(sampler);
	}

	return OK;
}

Error GLTFDocument::_serialize_materials(Ref<GLTFState> p_state) {
	Array materials;
	for (int32_t i = 0; i < p_state->materials.size(); i++) {
		Dictionary d;

		Ref<SpatialMaterial> material = p_state->materials[i];
		if (material.is_null()) {
			materials.push_back(d);
			continue;
		}
		if (!material->get_name().empty()) {
			d["name"] = _gen_unique_name(p_state, material->get_name());
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
				Ref<Texture> albedo_texture = material->get_texture(SpatialMaterial::TEXTURE_ALBEDO);
				GLTFTextureIndex gltf_texture_index = -1;

				if (albedo_texture.is_valid() && albedo_texture->get_data().is_valid()) {
					albedo_texture->set_name(material->get_name() + "_albedo");
					gltf_texture_index = _set_texture(p_state, albedo_texture);
				}
				if (gltf_texture_index != -1) {
					bct["index"] = gltf_texture_index;
					Dictionary extensions = _serialize_texture_transform_uv1(material);
					if (!extensions.empty()) {
						bct["extensions"] = extensions;
						p_state->use_khr_texture_transform = true;
					}
					mr["baseColorTexture"] = bct;
				}
			}

			mr["metallicFactor"] = material->get_metallic();
			mr["roughnessFactor"] = material->get_roughness();
			bool has_roughness = material->get_texture(SpatialMaterial::TEXTURE_ROUGHNESS).is_valid() && material->get_texture(SpatialMaterial::TEXTURE_ROUGHNESS)->get_data().is_valid();
			bool has_ao = material->get_feature(SpatialMaterial::FEATURE_AMBIENT_OCCLUSION) && material->get_texture(SpatialMaterial::TEXTURE_AMBIENT_OCCLUSION).is_valid();
			bool has_metalness = material->get_texture(SpatialMaterial::TEXTURE_METALLIC).is_valid() && material->get_texture(SpatialMaterial::TEXTURE_METALLIC)->get_data().is_valid();
			if (has_ao || has_roughness || has_metalness) {
				Dictionary mrt;
				Ref<Texture> roughness_texture = material->get_texture(SpatialMaterial::TEXTURE_ROUGHNESS);
				SpatialMaterial::TextureChannel roughness_channel = material->get_roughness_texture_channel();
				Ref<Texture> metallic_texture = material->get_texture(SpatialMaterial::TEXTURE_METALLIC);
				SpatialMaterial::TextureChannel metalness_channel = material->get_metallic_texture_channel();
				Ref<Texture> ao_texture = material->get_texture(SpatialMaterial::TEXTURE_AMBIENT_OCCLUSION);
				SpatialMaterial::TextureChannel ao_channel = material->get_ao_texture_channel();
				Ref<ImageTexture> orm_texture;
				orm_texture.instance();
				Ref<Image> orm_image;
				orm_image.instance();
				int32_t height = 0;
				int32_t width = 0;
				Ref<Image> ao_image;
				if (has_ao) {
					height = ao_texture->get_height();
					width = ao_texture->get_width();
					ao_image = ao_texture->get_data();
					Ref<ImageTexture> img_tex = ao_image;
					if (img_tex.is_valid()) {
						ao_image = img_tex->get_data();
					}
					if (ao_image->is_compressed()) {
						ao_image->decompress();
					}
				}
				Ref<Image> roughness_image;
				if (has_roughness) {
					height = roughness_texture->get_height();
					width = roughness_texture->get_width();
					roughness_image = roughness_texture->get_data();
					Ref<ImageTexture> img_tex = roughness_image;
					if (img_tex.is_valid()) {
						roughness_image = img_tex->get_data();
					}
					if (roughness_image->is_compressed()) {
						roughness_image->decompress();
					}
				}
				Ref<Image> metallness_image;
				if (has_metalness) {
					height = metallic_texture->get_height();
					width = metallic_texture->get_width();
					metallness_image = metallic_texture->get_data();
					Ref<ImageTexture> img_tex = metallness_image;
					if (img_tex.is_valid()) {
						metallness_image = img_tex->get_data();
					}
					if (metallness_image->is_compressed()) {
						metallness_image->decompress();
					}
				}
				Ref<Texture> albedo_texture = material->get_texture(SpatialMaterial::TEXTURE_ALBEDO);
				if (albedo_texture.is_valid() && albedo_texture->get_data().is_valid()) {
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
				orm_image->lock();
				for (int32_t h = 0; h < height; h++) {
					for (int32_t w = 0; w < width; w++) {
						Color c = Color(1.0f, 1.0f, 1.0f);
						if (has_ao) {
							ao_image->lock();
							if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_RED == ao_channel) {
								c.r = ao_image->get_pixel(w, h).r;
							} else if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_GREEN == ao_channel) {
								c.r = ao_image->get_pixel(w, h).g;
							} else if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_BLUE == ao_channel) {
								c.r = ao_image->get_pixel(w, h).b;
							} else if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_ALPHA == ao_channel) {
								c.r = ao_image->get_pixel(w, h).a;
							}
							ao_image->lock();
						}
						if (has_roughness) {
							roughness_image->lock();
							if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_RED == roughness_channel) {
								c.g = roughness_image->get_pixel(w, h).r;
							} else if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_GREEN == roughness_channel) {
								c.g = roughness_image->get_pixel(w, h).g;
							} else if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_BLUE == roughness_channel) {
								c.g = roughness_image->get_pixel(w, h).b;
							} else if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_ALPHA == roughness_channel) {
								c.g = roughness_image->get_pixel(w, h).a;
							}
							roughness_image->unlock();
						}
						if (has_metalness) {
							metallness_image->lock();
							if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_RED == metalness_channel) {
								c.b = metallness_image->get_pixel(w, h).r;
							} else if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_GREEN == metalness_channel) {
								c.b = metallness_image->get_pixel(w, h).g;
							} else if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_BLUE == metalness_channel) {
								c.b = metallness_image->get_pixel(w, h).b;
							} else if (SpatialMaterial::TextureChannel::TEXTURE_CHANNEL_ALPHA == metalness_channel) {
								c.b = metallness_image->get_pixel(w, h).a;
							}
							metallness_image->unlock();
						}
						orm_image->set_pixel(w, h, c);
					}
				}
				orm_image->unlock();
				orm_image->generate_mipmaps();
				orm_texture->create_from_image(orm_image);
				GLTFTextureIndex orm_texture_index = -1;
				if (has_ao || has_roughness || has_metalness) {
					orm_texture->set_name(material->get_name() + "_orm");
					orm_texture_index = _set_texture(p_state, orm_texture);
				}
				if (has_ao) {
					Dictionary ot;
					ot["index"] = orm_texture_index;
					d["occlusionTexture"] = ot;
				}
				if (has_roughness || has_metalness) {
					mrt["index"] = orm_texture_index;
					Dictionary extensions = _serialize_texture_transform_uv1(material);
					if (!extensions.empty()) {
						mrt["extensions"] = extensions;
						p_state->use_khr_texture_transform = true;
					}
					mr["metallicRoughnessTexture"] = mrt;
				}
			}
			d["pbrMetallicRoughness"] = mr;
		}

		if (material->get_feature(SpatialMaterial::FEATURE_NORMAL_MAPPING)) {
			Dictionary nt;
			Ref<ImageTexture> tex;
			tex.instance();
			{
				Ref<Texture> normal_texture = material->get_texture(SpatialMaterial::TEXTURE_NORMAL);
				if (normal_texture.is_valid()) {
					// Code for uncompressing RG normal maps
					Ref<Image> img = normal_texture->get_data();
					if (img.is_valid()) {
						Ref<ImageTexture> img_tex = img;
						if (img_tex.is_valid()) {
							img = img_tex->get_data();
						}
						img->decompress();
						img->convert(Image::FORMAT_RGBA8);
						img->lock();
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
						img->unlock();
						tex->create_from_image(img);
					}
				}
			}
			GLTFTextureIndex gltf_texture_index = -1;
			if (tex.is_valid() && tex->get_data().is_valid()) {
				tex->set_name(material->get_name() + "_normal");
				gltf_texture_index = _set_texture(p_state, tex);
			}
			nt["scale"] = material->get_normal_scale();
			if (gltf_texture_index != -1) {
				nt["index"] = gltf_texture_index;
				d["normalTexture"] = nt;
			}
		}

		if (material->get_feature(SpatialMaterial::FEATURE_EMISSION)) {
			const Color c = material->get_emission().to_srgb();
			Array arr;
			arr.push_back(c.r);
			arr.push_back(c.g);
			arr.push_back(c.b);
			d["emissiveFactor"] = arr;
		}
		if (material->get_feature(SpatialMaterial::FEATURE_EMISSION)) {
			Dictionary et;
			Ref<Texture> emission_texture = material->get_texture(SpatialMaterial::TEXTURE_EMISSION);
			GLTFTextureIndex gltf_texture_index = -1;
			if (emission_texture.is_valid() && emission_texture->get_data().is_valid()) {
				emission_texture->set_name(material->get_name() + "_emission");
				gltf_texture_index = _set_texture(p_state, emission_texture);
			}

			if (gltf_texture_index != -1) {
				et["index"] = gltf_texture_index;
				d["emissiveTexture"] = et;
			}
		}
		const bool ds = material->get_cull_mode() == SpatialMaterial::CULL_DISABLED;
		if (ds) {
			d["doubleSided"] = ds;
		}
		if (material->get_feature(SpatialMaterial::FEATURE_TRANSPARENT)) {
			if (material->get_flag(SpatialMaterial::FLAG_USE_ALPHA_SCISSOR)) {
				d["alphaMode"] = "MASK";
				d["alphaCutoff"] = material->get_alpha_scissor_threshold();
			} else {
				d["alphaMode"] = "BLEND";
			}
		}
		materials.push_back(d);
	}
	if (!materials.size()) {
		return OK;
	}
	p_state->json["materials"] = materials;
	print_verbose("Total materials: " + itos(p_state->materials.size()));

	return OK;
}

Error GLTFDocument::_parse_materials(Ref<GLTFState> p_state) {
	if (!p_state->json.has("materials")) {
		return OK;
	}

	const Array &materials = p_state->json["materials"];
	for (GLTFMaterialIndex i = 0; i < materials.size(); i++) {
		const Dictionary &d = materials[i];

		Ref<SpatialMaterial> material;
		material.instance();
		if (d.has("name") && !String(d["name"]).empty()) {
			material->set_name(d["name"]);
		} else {
			material->set_name(vformat("material_%s", itos(i)));
		}

		material->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		Dictionary pbr_spec_gloss_extensions;
		if (d.has("extensions")) {
			pbr_spec_gloss_extensions = d["extensions"];
		}
		if (pbr_spec_gloss_extensions.has("KHR_materials_pbrSpecularGlossiness")) {
			WARN_PRINT("Material uses a specular and glossiness workflow. Textures will be converted to roughness and metallic workflow, which may not be 100% accurate.");
			Dictionary sgm = pbr_spec_gloss_extensions["KHR_materials_pbrSpecularGlossiness"];

			Ref<GLTFSpecGloss> spec_gloss;
			spec_gloss.instance();
			if (sgm.has("diffuseTexture")) {
				const Dictionary &diffuse_texture_dict = sgm["diffuseTexture"];
				if (diffuse_texture_dict.has("index")) {
					Ref<Texture> diffuse_texture = _get_texture(p_state, diffuse_texture_dict["index"]);
					if (diffuse_texture.is_valid()) {
						spec_gloss->diffuse_img = diffuse_texture->get_data();
						material->set_texture(SpatialMaterial::TEXTURE_ALBEDO, diffuse_texture);
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
					const Ref<Texture> orig_texture = _get_texture(p_state, spec_gloss_texture["index"]);
					if (orig_texture.is_valid()) {
						spec_gloss->spec_gloss_img = orig_texture->get_data();
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
					material->set_texture(SpatialMaterial::TEXTURE_ALBEDO, _get_texture(p_state, bct["index"]));
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
					const Ref<Texture> t = _get_texture(p_state, bct["index"]);
					material->set_texture(SpatialMaterial::TEXTURE_METALLIC, t);
					material->set_metallic_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_BLUE);
					material->set_texture(SpatialMaterial::TEXTURE_ROUGHNESS, t);
					material->set_roughness_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_GREEN);
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
				material->set_texture(SpatialMaterial::TEXTURE_NORMAL, _get_texture(p_state, bct["index"]));
				material->set_feature(SpatialMaterial::FEATURE_NORMAL_MAPPING, true);
			}
			if (bct.has("scale")) {
				material->set_normal_scale(bct["scale"]);
			}
		}
		if (d.has("occlusionTexture")) {
			const Dictionary &bct = d["occlusionTexture"];
			if (bct.has("index")) {
				material->set_texture(SpatialMaterial::TEXTURE_AMBIENT_OCCLUSION, _get_texture(p_state, bct["index"]));
				material->set_ao_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_RED);
				material->set_feature(SpatialMaterial::FEATURE_AMBIENT_OCCLUSION, true);
			}
		}

		if (d.has("emissiveFactor")) {
			const Array &arr = d["emissiveFactor"];
			ERR_FAIL_COND_V(arr.size() != 3, ERR_PARSE_ERROR);
			const Color c = Color(arr[0], arr[1], arr[2]).to_srgb();
			material->set_feature(SpatialMaterial::FEATURE_EMISSION, true);

			material->set_emission(c);
		}

		if (d.has("emissiveTexture")) {
			const Dictionary &bct = d["emissiveTexture"];
			if (bct.has("index")) {
				material->set_texture(SpatialMaterial::TEXTURE_EMISSION, _get_texture(p_state, bct["index"]));
				material->set_feature(SpatialMaterial::FEATURE_EMISSION, true);
				material->set_emission(Color(0, 0, 0));
			}
		}

		if (d.has("doubleSided")) {
			const bool ds = d["doubleSided"];
			if (ds) {
				material->set_cull_mode(SpatialMaterial::CULL_DISABLED);
			}
		}

		if (d.has("alphaMode")) {
			const String &am = d["alphaMode"];
			if (am == "BLEND") {
				material->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
				material->set_depth_draw_mode(SpatialMaterial::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
			} else if (am == "MASK") {
				material->set_flag(SpatialMaterial::FLAG_USE_ALPHA_SCISSOR, true);
				if (d.has("alphaCutoff")) {
					material->set_alpha_scissor_threshold(d["alphaCutoff"]);
				} else {
					material->set_alpha_scissor_threshold(0.5f);
				}
			}
		}
		p_state->materials.push_back(material);
	}

	print_verbose("Total materials: " + itos(p_state->materials.size()));

	return OK;
}

void GLTFDocument::_set_texture_transform_uv1(const Dictionary &p_dict, Ref<SpatialMaterial> p_material) {
	if (p_dict.has("extensions")) {
		const Dictionary &extensions = p_dict["extensions"];
		if (extensions.has("KHR_texture_transform")) {
			const Dictionary &texture_transform = extensions["KHR_texture_transform"];
			const Array &offset_arr = texture_transform["offset"];
			if (offset_arr.size() == 2) {
				const Vector3 offset_vector3 = Vector3(offset_arr[0], offset_arr[1], 0.0f);
				p_material->set_uv1_offset(offset_vector3);
			}

			const Array &scale_arr = texture_transform["scale"];
			if (scale_arr.size() == 2) {
				const Vector3 scale_vector3 = Vector3(scale_arr[0], scale_arr[1], 1.0f);
				p_material->set_uv1_scale(scale_vector3);
			}
		}
	}
}

void GLTFDocument::spec_gloss_to_rough_metal(Ref<GLTFSpecGloss> r_spec_gloss, Ref<SpatialMaterial> p_material) {
	if (r_spec_gloss->spec_gloss_img.is_null()) {
		return;
	}
	if (r_spec_gloss->diffuse_img.is_null()) {
		return;
	}
	Ref<Image> rm_img;
	rm_img.instance();
	bool has_roughness = false;
	bool has_metal = false;
	p_material->set_roughness(1.0f);
	p_material->set_metallic(1.0f);
	rm_img->create(r_spec_gloss->spec_gloss_img->get_width(), r_spec_gloss->spec_gloss_img->get_height(), false, Image::FORMAT_RGBA8);
	rm_img->lock();
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
			r_spec_gloss->diffuse_img->lock();
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
			if (!Math::is_equal_approx(mr.b, 0.0f)) {
				has_metal = true;
			}
			mr.g *= r_spec_gloss->gloss_factor;
			mr.g = 1.0f - mr.g;
			rm_img->set_pixel(x, y, mr);
			r_spec_gloss->diffuse_img->set_pixel(x, y, base_color.to_srgb());
			r_spec_gloss->diffuse_img->unlock();
		}
	}
	rm_img->unlock();
	rm_img->generate_mipmaps();
	r_spec_gloss->diffuse_img->generate_mipmaps();
	Ref<ImageTexture> diffuse_image_texture;
	diffuse_image_texture.instance();
	diffuse_image_texture->create_from_image(r_spec_gloss->diffuse_img);
	p_material->set_texture(SpatialMaterial::TEXTURE_ALBEDO, diffuse_image_texture);
	Ref<ImageTexture> rm_image_texture;
	rm_image_texture.instance();
	rm_image_texture->create_from_image(rm_img);
	if (has_roughness) {
		p_material->set_texture(SpatialMaterial::TEXTURE_ROUGHNESS, rm_image_texture);
		p_material->set_roughness_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_GREEN);
	}

	if (has_metal) {
		p_material->set_texture(SpatialMaterial::TEXTURE_METALLIC, rm_image_texture);
		p_material->set_metallic_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_BLUE);
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
	r_base_color.r = CLAMP(r_base_color.r, 0.0f, 1.0f);
	r_base_color.g = CLAMP(r_base_color.g, 0.0f, 1.0f);
	r_base_color.b = CLAMP(r_base_color.b, 0.0f, 1.0f);
	r_base_color.a = CLAMP(r_base_color.a, 0.0f, 1.0f);
}

GLTFNodeIndex GLTFDocument::_find_highest_node(Ref<GLTFState> p_state, const Vector<GLTFNodeIndex> &p_subset) {
	int highest = -1;
	GLTFNodeIndex best_node = -1;

	for (int i = 0; i < p_subset.size(); ++i) {
		const GLTFNodeIndex node_i = p_subset[i];
		const Ref<GLTFNode> node = p_state->nodes[node_i];

		if (highest == -1 || node->height < highest) {
			highest = node->height;
			best_node = node_i;
		}
	}

	return best_node;
}

bool GLTFDocument::_capture_nodes_in_skin(Ref<GLTFState> p_state, Ref<GLTFSkin> p_skin, const GLTFNodeIndex p_node_index) {
	bool found_joint = false;

	for (int i = 0; i < p_state->nodes[p_node_index]->children.size(); ++i) {
		found_joint |= _capture_nodes_in_skin(p_state, p_skin, p_state->nodes[p_node_index]->children[i]);
	}

	if (found_joint) {
		// Mark it if we happen to find another skins joint...
		if (p_state->nodes[p_node_index]->joint && p_skin->joints.find(p_node_index) < 0) {
			p_skin->joints.push_back(p_node_index);
		} else if (p_skin->non_joints.find(p_node_index) < 0) {
			p_skin->non_joints.push_back(p_node_index);
		}
	}

	if (p_skin->joints.find(p_node_index) > 0) {
		return true;
	}

	return false;
}

void GLTFDocument::_capture_nodes_for_multirooted_skin(Ref<GLTFState> p_state, Ref<GLTFSkin> p_skin) {
	DisjointSet<GLTFNodeIndex> disjoint_set;

	for (int i = 0; i < p_skin->joints.size(); ++i) {
		const GLTFNodeIndex node_index = p_skin->joints[i];
		const GLTFNodeIndex parent = p_state->nodes[node_index]->parent;
		disjoint_set.insert(node_index);

		if (p_skin->joints.find(parent) >= 0) {
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

		if (maxHeight == -1 || p_state->nodes[root]->height < maxHeight) {
			maxHeight = p_state->nodes[root]->height;
		}
	}

	// Go up the tree till all of the multiple roots of the skin are at the same hierarchy level.
	// This sucks, but 99% of all game engines (not just Godot) would have this same issue.
	for (int i = 0; i < roots.size(); ++i) {
		GLTFNodeIndex current_node = roots[i];
		while (p_state->nodes[current_node]->height > maxHeight) {
			GLTFNodeIndex parent = p_state->nodes[current_node]->parent;

			if (p_state->nodes[parent]->joint && p_skin->joints.find(parent) < 0) {
				p_skin->joints.push_back(parent);
			} else if (p_skin->non_joints.find(parent) < 0) {
				p_skin->non_joints.push_back(parent);
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
		const GLTFNodeIndex first_parent = p_state->nodes[roots[0]]->parent;

		for (int i = 1; i < roots.size(); ++i) {
			all_same &= (first_parent == p_state->nodes[roots[i]]->parent);
		}

		if (!all_same) {
			for (int i = 0; i < roots.size(); ++i) {
				const GLTFNodeIndex current_node = roots[i];
				const GLTFNodeIndex parent = p_state->nodes[current_node]->parent;

				if (p_state->nodes[parent]->joint && p_skin->joints.find(parent) < 0) {
					p_skin->joints.push_back(parent);
				} else if (p_skin->non_joints.find(parent) < 0) {
					p_skin->non_joints.push_back(parent);
				}

				roots.write[i] = parent;
			}
		}

	} while (!all_same);
}

Error GLTFDocument::_expand_skin(Ref<GLTFState> p_state, Ref<GLTFSkin> p_skin) {
	_capture_nodes_for_multirooted_skin(p_state, p_skin);

	// Grab all nodes that lay in between skin joints/nodes
	DisjointSet<GLTFNodeIndex> disjoint_set;

	Vector<GLTFNodeIndex> all_skin_nodes;
	all_skin_nodes.append_array(p_skin->joints);
	all_skin_nodes.append_array(p_skin->non_joints);

	for (int i = 0; i < all_skin_nodes.size(); ++i) {
		const GLTFNodeIndex node_index = all_skin_nodes[i];
		const GLTFNodeIndex parent = p_state->nodes[node_index]->parent;
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

		const GLTFNodeIndex root = _find_highest_node(p_state, set);
		ERR_FAIL_COND_V(root < 0, FAILED);
		out_roots.push_back(root);
	}

	out_roots.sort();

	for (int i = 0; i < out_roots.size(); ++i) {
		_capture_nodes_in_skin(p_state, p_skin, out_roots[i]);
	}

	p_skin->roots = out_roots;

	return OK;
}

Error GLTFDocument::_verify_skin(Ref<GLTFState> p_state, Ref<GLTFSkin> p_skin) {
	// This may seem duplicated from expand_skins, but this is really a sanity check! (so it kinda is)
	// In case additional interpolating logic is added to the skins, this will help ensure that you
	// do not cause it to self implode into a fiery blaze

	// We are going to re-calculate the root nodes and compare them to the ones saved in the skin,
	// then ensure the multiple trees (if they exist) are on the same sublevel

	// Grab all nodes that lay in between skin joints/nodes
	DisjointSet<GLTFNodeIndex> disjoint_set;

	Vector<GLTFNodeIndex> all_skin_nodes;
	all_skin_nodes.append_array(p_skin->joints);
	all_skin_nodes.append_array(p_skin->non_joints);

	for (int i = 0; i < all_skin_nodes.size(); ++i) {
		const GLTFNodeIndex node_index = all_skin_nodes[i];
		const GLTFNodeIndex parent = p_state->nodes[node_index]->parent;
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

		const GLTFNodeIndex root = _find_highest_node(p_state, set);
		ERR_FAIL_COND_V(root < 0, FAILED);
		out_roots.push_back(root);
	}

	out_roots.sort();

	ERR_FAIL_COND_V(out_roots.size() == 0, FAILED);

	// Make sure the roots are the exact same (they better be)
	ERR_FAIL_COND_V(out_roots.size() != p_skin->roots.size(), FAILED);
	for (int i = 0; i < out_roots.size(); ++i) {
		ERR_FAIL_COND_V(out_roots[i] != p_skin->roots[i], FAILED);
	}

	// Single rooted skin? Perfectly ok!
	if (out_roots.size() == 1) {
		return OK;
	}

	// Make sure all parents of a multi-rooted skin are the SAME
	const GLTFNodeIndex parent = p_state->nodes[out_roots[0]]->parent;
	for (int i = 1; i < out_roots.size(); ++i) {
		if (p_state->nodes[out_roots[i]]->parent != parent) {
			return FAILED;
		}
	}

	return OK;
}

Error GLTFDocument::_parse_skins(Ref<GLTFState> p_state) {
	if (!p_state->json.has("skins")) {
		return OK;
	}

	const Array &skins = p_state->json["skins"];

	// Create the base skins, and mark nodes that are joints
	for (int i = 0; i < skins.size(); i++) {
		const Dictionary &d = skins[i];

		Ref<GLTFSkin> skin;
		skin.instance();

		ERR_FAIL_COND_V(!d.has("joints"), ERR_PARSE_ERROR);

		const Array &joints = d["joints"];

		if (d.has("inverseBindMatrices")) {
			skin->inverse_binds = _decode_accessor_as_xform(p_state, d["inverseBindMatrices"], false);
			ERR_FAIL_COND_V(skin->inverse_binds.size() != joints.size(), ERR_PARSE_ERROR);
		}

		for (int j = 0; j < joints.size(); j++) {
			const GLTFNodeIndex node = joints[j];
			ERR_FAIL_INDEX_V(node, p_state->nodes.size(), ERR_PARSE_ERROR);

			skin->joints.push_back(node);
			skin->joints_original.push_back(node);

			p_state->nodes.write[node]->joint = true;
		}

		if (d.has("name") && !String(d["name"]).empty()) {
			skin->set_name(d["name"]);
		} else {
			skin->set_name(vformat("skin_%s", itos(i)));
		}

		if (d.has("skeleton")) {
			skin->skin_root = d["skeleton"];
		}

		p_state->skins.push_back(skin);
	}

	for (GLTFSkinIndex i = 0; i < p_state->skins.size(); ++i) {
		Ref<GLTFSkin> skin = p_state->skins.write[i];

		// Expand the skin to capture all the extra non-joints that lie in between the actual joints,
		// and expand the hierarchy to ensure multi-rooted trees lie on the same height level
		ERR_FAIL_COND_V(_expand_skin(p_state, skin), ERR_PARSE_ERROR);
		ERR_FAIL_COND_V(_verify_skin(p_state, skin), ERR_PARSE_ERROR);
	}

	print_verbose("glTF: Total skins: " + itos(p_state->skins.size()));

	return OK;
}

Error GLTFDocument::_determine_skeletons(Ref<GLTFState> p_state) {
	// Using a disjoint set, we are going to potentially combine all skins that are actually branches
	// of a main skeleton, or treat skins defining the same set of nodes as ONE skeleton.
	// This is another unclear issue caused by the current glTF specification.

	DisjointSet<GLTFNodeIndex> skeleton_sets;

	for (GLTFSkinIndex skin_i = 0; skin_i < p_state->skins.size(); ++skin_i) {
		const Ref<GLTFSkin> skin = p_state->skins[skin_i];

		Vector<GLTFNodeIndex> all_skin_nodes;
		all_skin_nodes.append_array(skin->joints);
		all_skin_nodes.append_array(skin->non_joints);

		for (int i = 0; i < all_skin_nodes.size(); ++i) {
			const GLTFNodeIndex node_index = all_skin_nodes[i];
			const GLTFNodeIndex parent = p_state->nodes[node_index]->parent;
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
			highest_group_members.push_back(_find_highest_node(p_state, group));
			groups.push_back(group);
		}

		for (int i = 0; i < highest_group_members.size(); ++i) {
			const GLTFNodeIndex node_i = highest_group_members[i];

			// Attach any siblings together (this needs to be done n^2/2 times)
			for (int j = i + 1; j < highest_group_members.size(); ++j) {
				const GLTFNodeIndex node_j = highest_group_members[j];

				// Even if they are siblings under the root! :)
				if (p_state->nodes[node_i]->parent == p_state->nodes[node_j]->parent) {
					skeleton_sets.create_union(node_i, node_j);
				}
			}

			// Attach any parenting going on together (we need to do this n^2 times)
			const GLTFNodeIndex node_i_parent = p_state->nodes[node_i]->parent;
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
		skeleton.instance();

		Vector<GLTFNodeIndex> skeleton_nodes;
		skeleton_sets.get_members(skeleton_nodes, skeleton_owner);

		for (GLTFSkinIndex skin_i = 0; skin_i < p_state->skins.size(); ++skin_i) {
			Ref<GLTFSkin> skin = p_state->skins.write[skin_i];

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

			if (p_state->nodes[node_i]->joint) {
				skeleton->joints.push_back(node_i);
			} else {
				non_joints.push_back(node_i);
			}
		}

		p_state->skeletons.push_back(skeleton);

		_reparent_non_joint_skeleton_subtrees(p_state, p_state->skeletons.write[skel_i], non_joints);
	}

	for (GLTFSkeletonIndex skel_i = 0; skel_i < p_state->skeletons.size(); ++skel_i) {
		Ref<GLTFSkeleton> skeleton = p_state->skeletons.write[skel_i];

		for (int i = 0; i < skeleton->joints.size(); ++i) {
			const GLTFNodeIndex node_i = skeleton->joints[i];
			Ref<GLTFNode> node = p_state->nodes[node_i];

			ERR_FAIL_COND_V(!node->joint, ERR_PARSE_ERROR);
			ERR_FAIL_COND_V(node->skeleton >= 0, ERR_PARSE_ERROR);
			node->skeleton = skel_i;
		}

		ERR_FAIL_COND_V(_determine_skeleton_roots(p_state, skel_i), ERR_PARSE_ERROR);
	}

	return OK;
}

Error GLTFDocument::_reparent_non_joint_skeleton_subtrees(Ref<GLTFState> p_state, Ref<GLTFSkeleton> p_skeleton, const Vector<GLTFNodeIndex> &p_non_joints) {
	DisjointSet<GLTFNodeIndex> subtree_set;

	// Populate the disjoint set with ONLY non joints that are in the skeleton hierarchy (non_joints vector)
	// This way we can find any joints that lie in between joints, as the current glTF specification
	// mentions nothing about non-joints being in between joints of the same skin. Hopefully one day we
	// can remove this code.

	// skinD depicted here explains this issue:
	// https://github.com/KhronosGroup/glTF-Asset-Generator/blob/master/Output/Positive/Animation_Skin

	for (int i = 0; i < p_non_joints.size(); ++i) {
		const GLTFNodeIndex node_i = p_non_joints[i];

		subtree_set.insert(node_i);

		const GLTFNodeIndex parent_i = p_state->nodes[node_i]->parent;
		if (parent_i >= 0 && p_non_joints.find(parent_i) >= 0 && !p_state->nodes[parent_i]->joint) {
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
			Ref<GLTFNode> node = p_state->nodes[subtree_nodes[subtree_i]];
			node->joint = true;
			// Add the joint to the skeletons joints
			p_skeleton->joints.push_back(subtree_nodes[subtree_i]);
		}
	}

	return OK;
}

Error GLTFDocument::_determine_skeleton_roots(Ref<GLTFState> p_state, const GLTFSkeletonIndex p_skel_i) {
	DisjointSet<GLTFNodeIndex> disjoint_set;

	for (GLTFNodeIndex i = 0; i < p_state->nodes.size(); ++i) {
		const Ref<GLTFNode> node = p_state->nodes[i];

		if (node->skeleton != p_skel_i) {
			continue;
		}

		disjoint_set.insert(i);

		if (node->parent >= 0 && p_state->nodes[node->parent]->skeleton == p_skel_i) {
			disjoint_set.create_union(node->parent, i);
		}
	}

	Ref<GLTFSkeleton> skeleton = p_state->skeletons.write[p_skel_i];

	Vector<GLTFNodeIndex> owners;
	disjoint_set.get_representatives(owners);

	Vector<GLTFNodeIndex> roots;

	for (int i = 0; i < owners.size(); ++i) {
		Vector<GLTFNodeIndex> set;
		disjoint_set.get_members(set, owners[i]);
		const GLTFNodeIndex root = _find_highest_node(p_state, set);
		ERR_FAIL_COND_V(root < 0, FAILED);
		roots.push_back(root);
	}

	roots.sort();
	PoolVector<GLTFNodeIndex> roots_array;
	roots_array.resize(roots.size());
	PoolVector<GLTFNodeIndex>::Write write_roots = roots_array.write();
	for (int32_t root_i = 0; root_i < roots_array.size(); root_i++) {
		write_roots[root_i] = roots[root_i];
	}
	skeleton->roots = roots_array;

	if (roots.size() == 0) {
		return FAILED;
	} else if (roots.size() == 1) {
		return OK;
	}

	// Check that the subtrees have the same parent root
	const GLTFNodeIndex parent = p_state->nodes[roots[0]]->parent;
	for (int i = 1; i < roots.size(); ++i) {
		if (p_state->nodes[roots[i]]->parent != parent) {
			return FAILED;
		}
	}

	return OK;
}

Error GLTFDocument::_create_skeletons(Ref<GLTFState> p_state) {
	for (GLTFSkeletonIndex skel_i = 0; skel_i < p_state->skeletons.size(); ++skel_i) {
		Ref<GLTFSkeleton> gltf_skeleton = p_state->skeletons.write[skel_i];

		Skeleton *skeleton = memnew(Skeleton);
		gltf_skeleton->godot_skeleton = skeleton;
		p_state->skeleton3d_to_gltf_skeleton[skeleton->get_instance_id()] = skel_i;

		// Make a unique name, no gltf node represents this skeleton
		skeleton->set_name(_gen_unique_name(p_state, "Skeleton"));

		List<GLTFNodeIndex> bones;

		for (int i = 0; i < gltf_skeleton->roots.size(); ++i) {
			bones.push_back(gltf_skeleton->roots[i]);
		}

		// Make the skeleton creation deterministic by going through the roots in
		// a sorted order, and DEPTH FIRST
		bones.sort();

		while (!bones.empty()) {
			const GLTFNodeIndex node_i = bones.front()->get();
			bones.pop_front();

			Ref<GLTFNode> node = p_state->nodes[node_i];
			ERR_FAIL_COND_V(node->skeleton != skel_i, FAILED);

			{ // Add all child nodes to the stack (deterministically)
				Vector<GLTFNodeIndex> child_nodes;
				for (int i = 0; i < node->children.size(); ++i) {
					const GLTFNodeIndex child_i = node->children[i];
					if (p_state->nodes[child_i]->skeleton == skel_i) {
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

			if (node->get_name().empty()) {
				node->set_name("bone");
			}

			node->set_name(_gen_unique_bone_name(p_state, skel_i, node->get_name()));

			skeleton->add_bone(node->get_name());
			skeleton->set_bone_rest(bone_index, node->xform);

			if (node->parent >= 0 && p_state->nodes[node->parent]->skeleton == skel_i) {
				const int bone_parent = skeleton->find_bone(p_state->nodes[node->parent]->get_name());
				ERR_FAIL_COND_V(bone_parent < 0, FAILED);
				skeleton->set_bone_parent(bone_index, skeleton->find_bone(p_state->nodes[node->parent]->get_name()));
			}

			p_state->scene_nodes.insert(node_i, skeleton);
		}
	}

	ERR_FAIL_COND_V(_map_skin_joints_indices_to_skeleton_bone_indices(p_state), ERR_PARSE_ERROR);

	return OK;
}

Error GLTFDocument::_map_skin_joints_indices_to_skeleton_bone_indices(Ref<GLTFState> p_state) {
	for (GLTFSkinIndex skin_i = 0; skin_i < p_state->skins.size(); ++skin_i) {
		Ref<GLTFSkin> skin = p_state->skins.write[skin_i];

		Ref<GLTFSkeleton> skeleton = p_state->skeletons[skin->skeleton];

		for (int joint_index = 0; joint_index < skin->joints_original.size(); ++joint_index) {
			const GLTFNodeIndex node_i = skin->joints_original[joint_index];
			const Ref<GLTFNode> node = p_state->nodes[node_i];

			const int bone_index = skeleton->godot_skeleton->find_bone(node->get_name());
			ERR_FAIL_COND_V(bone_index < 0, FAILED);

			skin->joint_i_to_bone_i.insert(joint_index, bone_index);
		}
	}

	return OK;
}

Error GLTFDocument::_serialize_skins(Ref<GLTFState> p_state) {
	_remove_duplicate_skins(p_state);
	Array json_skins;
	for (int skin_i = 0; skin_i < p_state->skins.size(); skin_i++) {
		Ref<GLTFSkin> gltf_skin = p_state->skins[skin_i];
		Dictionary json_skin;
		json_skin["inverseBindMatrices"] = _encode_accessor_as_xform(p_state, gltf_skin->inverse_binds, false);
		json_skin["joints"] = gltf_skin->get_joints();
		json_skin["name"] = gltf_skin->get_name();
		json_skins.push_back(json_skin);
	}
	if (!p_state->skins.size()) {
		return OK;
	}

	p_state->json["skins"] = json_skins;
	return OK;
}

Error GLTFDocument::_create_skins(Ref<GLTFState> p_state) {
	for (GLTFSkinIndex skin_i = 0; skin_i < p_state->skins.size(); ++skin_i) {
		Ref<GLTFSkin> gltf_skin = p_state->skins.write[skin_i];

		Ref<Skin> skin;
		skin.instance();

		// Some skins don't have IBM's! What absolute monsters!
		const bool has_ibms = !gltf_skin->inverse_binds.empty();

		for (int joint_i = 0; joint_i < gltf_skin->joints_original.size(); ++joint_i) {
			GLTFNodeIndex node = gltf_skin->joints_original[joint_i];
			String bone_name = p_state->nodes[node]->get_name();

			Transform xform;
			if (has_ibms) {
				xform = gltf_skin->inverse_binds[joint_i];
			}

			if (p_state->use_named_skin_binds) {
				skin->add_named_bind(bone_name, xform);
			} else {
				int32_t bone_i = gltf_skin->joint_i_to_bone_i[joint_i];
				skin->add_bind(bone_i, xform);
			}
		}

		gltf_skin->godot_skin = skin;
	}

	// Purge the duplicates!
	_remove_duplicate_skins(p_state);

	// Create unique names now, after removing duplicates
	for (GLTFSkinIndex skin_i = 0; skin_i < p_state->skins.size(); ++skin_i) {
		Ref<Skin> skin = p_state->skins.write[skin_i]->godot_skin;
		if (skin->get_name().empty()) {
			// Make a unique name, no gltf node represents this skin
			skin->set_name(_gen_unique_name(p_state, "Skin"));
		}
	}

	return OK;
}

bool GLTFDocument::_skins_are_same(const Ref<Skin> p_skin_a, const Ref<Skin> p_skin_b) {
	if (p_skin_a->get_bind_count() != p_skin_b->get_bind_count()) {
		return false;
	}

	for (int i = 0; i < p_skin_a->get_bind_count(); ++i) {
		if (p_skin_a->get_bind_bone(i) != p_skin_b->get_bind_bone(i)) {
			return false;
		}
		if (p_skin_a->get_bind_name(i) != p_skin_b->get_bind_name(i)) {
			return false;
		}

		Transform a_xform = p_skin_a->get_bind_pose(i);
		Transform b_xform = p_skin_b->get_bind_pose(i);

		if (a_xform != b_xform) {
			return false;
		}
	}

	return true;
}

void GLTFDocument::_remove_duplicate_skins(Ref<GLTFState> p_state) {
	for (int i = 0; i < p_state->skins.size(); ++i) {
		for (int j = i + 1; j < p_state->skins.size(); ++j) {
			const Ref<Skin> skin_i = p_state->skins[i]->godot_skin;
			const Ref<Skin> skin_j = p_state->skins[j]->godot_skin;

			if (_skins_are_same(skin_i, skin_j)) {
				// replace it and delete the old
				p_state->skins.write[j]->godot_skin = skin_i;
			}
		}
	}
}

Error GLTFDocument::_serialize_lights(Ref<GLTFState> p_state) {
	if (p_state->lights.empty()) {
		return OK;
	}
	Array lights;
	for (GLTFLightIndex i = 0; i < p_state->lights.size(); i++) {
		Dictionary d;
		Ref<GLTFLight> light = p_state->lights[i];
		Array color;
		color.resize(3);
		color[0] = light->color.r;
		color[1] = light->color.g;
		color[2] = light->color.b;
		d["color"] = color;
		d["type"] = light->type;
		if (light->type == "spot") {
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

	Dictionary extensions;
	if (p_state->json.has("extensions")) {
		extensions = p_state->json["extensions"];
	} else {
		p_state->json["extensions"] = extensions;
	}
	Dictionary lights_punctual;
	extensions["KHR_lights_punctual"] = lights_punctual;
	lights_punctual["lights"] = lights;

	print_verbose("glTF: Total lights: " + itos(p_state->lights.size()));

	return OK;
}

Error GLTFDocument::_serialize_cameras(Ref<GLTFState> p_state) {
	Array cameras;
	cameras.resize(p_state->cameras.size());
	for (GLTFCameraIndex i = 0; i < p_state->cameras.size(); i++) {
		Dictionary d;

		Ref<GLTFCamera> camera = p_state->cameras[i];

		if (camera->get_perspective() == false) {
			Dictionary og;
			og["ymag"] = Math::deg2rad(camera->get_fov_size());
			og["xmag"] = Math::deg2rad(camera->get_fov_size());
			og["zfar"] = camera->get_zfar();
			og["znear"] = camera->get_znear();
			d["orthographic"] = og;
			d["type"] = "orthographic";
		} else if (camera->get_perspective()) {
			Dictionary ppt;
			// GLTF spec is in radians, Godot's camera is in degrees.
			ppt["yfov"] = Math::deg2rad(camera->get_fov_size());
			ppt["zfar"] = camera->get_zfar();
			ppt["znear"] = camera->get_znear();
			d["perspective"] = ppt;
			d["type"] = "perspective";
		}
		cameras[i] = d;
	}

	if (!p_state->cameras.size()) {
		return OK;
	}

	p_state->json["cameras"] = cameras;

	print_verbose("glTF: Total cameras: " + itos(p_state->cameras.size()));

	return OK;
}

Error GLTFDocument::_parse_lights(Ref<GLTFState> p_state) {
	if (!p_state->json.has("extensions")) {
		return OK;
	}
	Dictionary extensions = p_state->json["extensions"];
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
		light.instance();
		ERR_FAIL_COND_V(!d.has("type"), ERR_PARSE_ERROR);
		const String &type = d["type"];
		light->type = type;

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

		p_state->lights.push_back(light);
	}

	print_verbose("glTF: Total lights: " + itos(p_state->lights.size()));

	return OK;
}

Error GLTFDocument::_parse_cameras(Ref<GLTFState> p_state) {
	if (!p_state->json.has("cameras")) {
		return OK;
	}

	const Array cameras = p_state->json["cameras"];

	for (GLTFCameraIndex i = 0; i < cameras.size(); i++) {
		const Dictionary &d = cameras[i];

		Ref<GLTFCamera> camera;
		camera.instance();
		ERR_FAIL_COND_V(!d.has("type"), ERR_PARSE_ERROR);
		const String &type = d["type"];
		if (type == "orthographic") {
			camera->set_perspective(false);
			if (d.has("orthographic")) {
				const Dictionary &og = d["orthographic"];
				// GLTF spec is in radians, Godot's camera is in degrees.
				camera->set_fov_size(Math::rad2deg(real_t(og["ymag"])));
				camera->set_zfar(og["zfar"]);
				camera->set_znear(og["znear"]);
			} else {
				camera->set_fov_size(10);
			}
		} else if (type == "perspective") {
			camera->set_perspective(true);
			if (d.has("perspective")) {
				const Dictionary &ppt = d["perspective"];
				// GLTF spec is in radians, Godot's camera is in degrees.
				camera->set_fov_size(Math::rad2deg(real_t(ppt["yfov"])));
				camera->set_zfar(ppt["zfar"]);
				camera->set_znear(ppt["znear"]);
			} else {
				camera->set_fov_size(10);
			}
		} else {
			ERR_FAIL_V_MSG(ERR_PARSE_ERROR, "Camera should be in 'orthographic' or 'perspective'");
		}

		p_state->cameras.push_back(camera);
	}

	print_verbose("glTF: Total cameras: " + itos(p_state->cameras.size()));

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

Error GLTFDocument::_serialize_animations(Ref<GLTFState> p_state) {
	if (!p_state->animation_players.size()) {
		return OK;
	}
	for (int32_t player_i = 0; player_i < p_state->animation_players.size(); player_i++) {
		List<StringName> animation_names;
		AnimationPlayer *animation_player = p_state->animation_players[player_i];
		animation_player->get_animation_list(&animation_names);
		if (animation_names.size()) {
			for (int animation_name_i = 0; animation_name_i < animation_names.size(); animation_name_i++) {
				_convert_animation(p_state, animation_player, animation_names[animation_name_i]);
			}
		}
	}
	Array animations;
	for (GLTFAnimationIndex animation_i = 0; animation_i < p_state->animations.size(); animation_i++) {
		Dictionary d;
		Ref<GLTFAnimation> gltf_animation = p_state->animations[animation_i];
		if (!gltf_animation->get_tracks().size()) {
			continue;
		}

		if (!gltf_animation->get_name().empty()) {
			d["name"] = gltf_animation->get_name();
		}
		Array channels;
		Array samplers;

		for (Map<int, GLTFAnimation::Track>::Element *track_i = gltf_animation->get_tracks().front(); track_i; track_i = track_i->next()) {
			GLTFAnimation::Track track = track_i->get();
			if (track.translation_track.times.size()) {
				Dictionary t;
				t["sampler"] = samplers.size();
				Dictionary s;

				s["interpolation"] = interpolation_to_string(track.translation_track.interpolation);
				Vector<real_t> times = Variant(track.translation_track.times);
				s["input"] = _encode_accessor_as_floats(p_state, times, false);
				Vector<Vector3> values = Variant(track.translation_track.values);
				s["output"] = _encode_accessor_as_vec3(p_state, values, false);

				samplers.push_back(s);

				Dictionary target;
				target["path"] = "translation";
				target["node"] = track_i->key();

				t["target"] = target;
				channels.push_back(t);
			}
			if (track.rotation_track.times.size()) {
				Dictionary t;
				t["sampler"] = samplers.size();
				Dictionary s;

				s["interpolation"] = interpolation_to_string(track.rotation_track.interpolation);
				Vector<real_t> times = Variant(track.rotation_track.times);
				s["input"] = _encode_accessor_as_floats(p_state, times, false);
				Vector<Quat> values = track.rotation_track.values;
				s["output"] = _encode_accessor_as_quats(p_state, values, false);

				samplers.push_back(s);

				Dictionary target;
				target["path"] = "rotation";
				target["node"] = track_i->key();

				t["target"] = target;
				channels.push_back(t);
			}
			if (track.scale_track.times.size()) {
				Dictionary t;
				t["sampler"] = samplers.size();
				Dictionary s;

				s["interpolation"] = interpolation_to_string(track.scale_track.interpolation);
				Vector<real_t> times = Variant(track.scale_track.times);
				s["input"] = _encode_accessor_as_floats(p_state, times, false);
				Vector<Vector3> values = Variant(track.scale_track.values);
				s["output"] = _encode_accessor_as_vec3(p_state, values, false);

				samplers.push_back(s);

				Dictionary target;
				target["path"] = "scale";
				target["node"] = track_i->key();

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
						float weight = _interpolate_track<float>(track.weight_tracks[track_idx].times,
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
					Vector<float> wdata = track.weight_tracks[k].values;
					for (int l = 0; l < wdata.size(); l++) {
						int32_t index = l * weight_tracks_size + k;
						ERR_BREAK(index >= all_track_values.size());
						all_track_values.write[index] = wdata.write[l];
					}
				}

				s["interpolation"] = interpolation_to_string(track.weight_tracks[track.weight_tracks.size() - 1].interpolation);
				s["input"] = _encode_accessor_as_floats(p_state, all_track_times, false);
				s["output"] = _encode_accessor_as_floats(p_state, all_track_values, false);

				samplers.push_back(s);

				Dictionary target;
				target["path"] = "weights";
				target["node"] = track_i->key();

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
	p_state->json["animations"] = animations;

	print_verbose("glTF: Total animations '" + itos(p_state->animations.size()) + "'.");

	return OK;
}

Error GLTFDocument::_parse_animations(Ref<GLTFState> p_state) {
	if (!p_state->json.has("animations")) {
		return OK;
	}

	const Array &animations = p_state->json["animations"];

	for (GLTFAnimationIndex i = 0; i < animations.size(); i++) {
		const Dictionary &d = animations[i];

		Ref<GLTFAnimation> animation;
		animation.instance();

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
			if (p_state->use_legacy_names) {
				animation->set_name(_sanitize_scene_name(p_state, name));
			} else {
				animation->set_name(_gen_unique_animation_name(p_state, name));
			}
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

			ERR_FAIL_INDEX_V(node, p_state->nodes.size(), ERR_PARSE_ERROR);

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

			const Vector<float> times = _decode_accessor_as_floats(p_state, input, false);
			if (path == "translation") {
				const Vector<Vector3> translations = _decode_accessor_as_vec3(p_state, output, false);
				track->translation_track.interpolation = interp;
				track->translation_track.times = Variant(times); //convert via variant
				track->translation_track.values = Variant(translations); //convert via variant
			} else if (path == "rotation") {
				const Vector<Quat> rotations = _decode_accessor_as_quat(p_state, output, false);
				track->rotation_track.interpolation = interp;
				track->rotation_track.times = Variant(times); //convert via variant
				track->rotation_track.values = rotations;
			} else if (path == "scale") {
				const Vector<Vector3> scales = _decode_accessor_as_vec3(p_state, output, false);
				track->scale_track.interpolation = interp;
				track->scale_track.times = Variant(times); //convert via variant
				track->scale_track.values = Variant(scales); //convert via variant
			} else if (path == "weights") {
				const Vector<float> weights = _decode_accessor_as_floats(p_state, output, false);

				ERR_FAIL_INDEX_V(p_state->nodes[node]->mesh, p_state->meshes.size(), ERR_PARSE_ERROR);
				Ref<GLTFMesh> mesh = p_state->meshes[p_state->nodes[node]->mesh];
				ERR_CONTINUE(!mesh->get_blend_weights().size());
				const int wc = mesh->get_blend_weights().size();

				track->weight_tracks.resize(wc);

				const int expected_value_count = times.size() * output_count * wc;
				ERR_CONTINUE_MSG(weights.size() != expected_value_count, "Invalid weight data, expected " + itos(expected_value_count) + " weight values, got " + itos(weights.size()) + " instead.");

				const int wlen = weights.size() / wc;
				for (int k = 0; k < wc; k++) { //separate tracks, having them together is not such a good idea
					GLTFAnimation::Channel<float> cf;
					cf.interpolation = interp;
					cf.times = Variant(times);
					Vector<float> wdata;
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

		p_state->animations.push_back(animation);
	}

	print_verbose("glTF: Total animations '" + itos(p_state->animations.size()) + "'.");

	return OK;
}

void GLTFDocument::_assign_scene_names(Ref<GLTFState> p_state) {
	for (int i = 0; i < p_state->nodes.size(); i++) {
		Ref<GLTFNode> gltf_node = p_state->nodes[i];

		// Any joints get unique names generated when the skeleton is made, unique to the skeleton
		if (gltf_node->skeleton >= 0) {
			continue;
		}

		if (gltf_node->get_name().empty()) {
			if (gltf_node->mesh >= 0) {
				gltf_node->set_name(_gen_unique_name(p_state, "Mesh"));
			} else if (gltf_node->camera >= 0) {
				gltf_node->set_name(_gen_unique_name(p_state, "Camera"));
			} else {
				gltf_node->set_name(_gen_unique_name(p_state, "Node"));
			}
		}

		gltf_node->set_name(_gen_unique_name(p_state, gltf_node->get_name()));
	}

	// Assign a unique name to the scene last to avoid naming conflicts with the root
	p_state->scene_name = _gen_unique_name(p_state, p_state->scene_name);
}

BoneAttachment *GLTFDocument::_generate_bone_attachment(Ref<GLTFState> p_state, Skeleton *p_skeleton, const GLTFNodeIndex p_node_index, const GLTFNodeIndex p_bone_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];
	Ref<GLTFNode> bone_node = p_state->nodes[p_bone_index];

	BoneAttachment *bone_attachment = memnew(BoneAttachment);
	print_verbose("glTF: Creating bone attachment for: " + gltf_node->get_name());

	ERR_FAIL_COND_V(!bone_node->joint, nullptr);

	bone_attachment->set_bone_name(bone_node->get_name());

	return bone_attachment;
}

GLTFMeshIndex GLTFDocument::_convert_mesh_to_gltf(Ref<GLTFState> p_state, MeshInstance *p_mesh_instance) {
	ERR_FAIL_NULL_V(p_mesh_instance, -1);
	if (p_mesh_instance->get_mesh().is_null()) {
		return -1;
	}
	Ref<ArrayMesh> import_mesh;
	import_mesh.instance();
	Ref<Mesh> godot_mesh = p_mesh_instance->get_mesh();
	if (godot_mesh.is_null()) {
		return -1;
	}
	int32_t blend_count = godot_mesh->get_blend_shape_count();
	Vector<float> blend_weights;
	blend_weights.resize(blend_count);
	Ref<ArrayMesh> am = godot_mesh;
	if (am != nullptr) {
		import_mesh = am;
	} else {
		for (int32_t surface_i = 0; surface_i < godot_mesh->get_surface_count(); surface_i++) {
			Mesh::PrimitiveType primitive_type = godot_mesh->surface_get_primitive_type(surface_i);
			Array arrays = godot_mesh->surface_get_arrays(surface_i);
			Ref<Material> mat = godot_mesh->surface_get_material(surface_i);
			if (p_mesh_instance->get_surface_material(surface_i).is_valid()) {
				mat = p_mesh_instance->get_surface_material(surface_i);
			}
			if (p_mesh_instance->get_material_override().is_valid()) {
				mat = p_mesh_instance->get_material_override();
			}
			int32_t mat_idx = import_mesh->get_surface_count();
			import_mesh->add_surface_from_arrays(primitive_type, arrays);
			import_mesh->surface_set_material(mat_idx, mat);
		}
	}
	for (int32_t blend_i = 0; blend_i < blend_count; blend_i++) {
		blend_weights.write[blend_i] = 0.0f;
	}
	Ref<GLTFMesh> gltf_mesh;
	gltf_mesh.instance();
	Array instance_materials;
	for (int32_t surface_i = 0; surface_i < import_mesh->get_surface_count(); surface_i++) {
		Ref<Material> mat = import_mesh->surface_get_material(surface_i);
		if (p_mesh_instance->get_surface_material(surface_i).is_valid()) {
			mat = p_mesh_instance->get_surface_material(surface_i);
		}
		if (p_mesh_instance->get_material_override().is_valid()) {
			mat = p_mesh_instance->get_material_override();
		}
		instance_materials.append(mat);
	}
	gltf_mesh->set_instance_materials(instance_materials);
	gltf_mesh->set_mesh(import_mesh);
	gltf_mesh->set_blend_weights(blend_weights);
	GLTFMeshIndex mesh_i = p_state->meshes.size();
	p_state->meshes.push_back(gltf_mesh);
	return mesh_i;
}

Spatial *GLTFDocument::_generate_mesh_instance(Ref<GLTFState> p_state, Node *p_scene_parent, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	ERR_FAIL_INDEX_V(gltf_node->mesh, p_state->meshes.size(), nullptr);

	MeshInstance *mi = memnew(MeshInstance);
	print_verbose("glTF: Creating mesh for: " + gltf_node->get_name());

	Ref<GLTFMesh> mesh = p_state->meshes.write[gltf_node->mesh];
	if (mesh.is_null()) {
		return mi;
	}
	Ref<ArrayMesh> import_mesh = mesh->get_mesh();
	if (import_mesh.is_null()) {
		return mi;
	}
	mi->set_mesh(import_mesh);
	for (int i = 0; i < mesh->get_blend_weights().size(); i++) {
		mi->set("blend_shapes/" + mesh->get_mesh()->get_blend_shape_name(i), mesh->get_blend_weights()[i]);
	}
	return mi;
}

Spatial *GLTFDocument::_generate_light(Ref<GLTFState> p_state, Node *p_scene_parent, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	ERR_FAIL_INDEX_V(gltf_node->light, p_state->lights.size(), nullptr);

	print_verbose("glTF: Creating light for: " + gltf_node->get_name());

	Ref<GLTFLight> l = p_state->lights[gltf_node->light];

	float intensity = l->intensity;
	if (intensity > 10) {
		// GLTF spec has the default around 1, but Blender defaults lights to 100.
		// The only sane way to handle this is to check where it came from and
		// handle it accordingly. If it's over 10, it probably came from Blender.
		intensity /= 100;
	}

	if (l->type == "directional") {
		DirectionalLight *light = memnew(DirectionalLight);
		light->set_param(Light::PARAM_ENERGY, intensity);
		light->set_color(l->color);
		return light;
	}

	const float range = CLAMP(l->range, 0, 4096);
	if (l->type == "point") {
		OmniLight *light = memnew(OmniLight);
		light->set_param(OmniLight::PARAM_ENERGY, intensity);
		light->set_param(OmniLight::PARAM_RANGE, range);
		light->set_color(l->color);
		return light;
	}
	if (l->type == "spot") {
		SpotLight *light = memnew(SpotLight);
		light->set_param(SpotLight::PARAM_ENERGY, intensity);
		light->set_param(SpotLight::PARAM_RANGE, range);
		light->set_param(SpotLight::PARAM_SPOT_ANGLE, Math::rad2deg(l->outer_cone_angle));
		light->set_color(l->color);

		// Line of best fit derived from guessing, see https://www.desmos.com/calculator/biiflubp8b
		// The points in desmos are not exact, except for (1, infinity).
		float angle_ratio = l->inner_cone_angle / l->outer_cone_angle;
		float angle_attenuation = 0.2 / (1 - angle_ratio) - 0.1;
		light->set_param(SpotLight::PARAM_SPOT_ATTENUATION, angle_attenuation);
		return light;
	}
	return memnew(Spatial);
}

Camera *GLTFDocument::_generate_camera(Ref<GLTFState> p_state, Node *p_scene_parent, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	ERR_FAIL_INDEX_V(gltf_node->camera, p_state->cameras.size(), nullptr);

	Camera *camera = memnew(Camera);
	print_verbose("glTF: Creating camera for: " + gltf_node->get_name());

	Ref<GLTFCamera> c = p_state->cameras[gltf_node->camera];
	if (c->get_perspective()) {
		camera->set_perspective(c->get_fov_size(), c->get_znear(), c->get_zfar());
	} else {
		camera->set_orthogonal(c->get_fov_size(), c->get_znear(), c->get_zfar());
	}

	return camera;
}

GLTFCameraIndex GLTFDocument::_convert_camera(Ref<GLTFState> p_state, Camera *p_camera) {
	print_verbose("glTF: Converting camera: " + p_camera->get_name());

	Ref<GLTFCamera> c;
	c.instance();

	if (p_camera->get_projection() == Camera::Projection::PROJECTION_PERSPECTIVE) {
		c->set_perspective(true);
		c->set_fov_size(p_camera->get_fov());
		c->set_zfar(p_camera->get_zfar());
		c->set_znear(p_camera->get_znear());
	} else {
		c->set_fov_size(p_camera->get_fov());
		c->set_zfar(p_camera->get_zfar());
		c->set_znear(p_camera->get_znear());
	}
	GLTFCameraIndex camera_index = p_state->cameras.size();
	p_state->cameras.push_back(c);
	return camera_index;
}

GLTFLightIndex GLTFDocument::_convert_light(Ref<GLTFState> p_state, Light *p_light) {
	print_verbose("glTF: Converting light: " + p_light->get_name());

	Ref<GLTFLight> l;
	l.instance();
	l->color = p_light->get_color();
	if (cast_to<DirectionalLight>(p_light)) {
		l->type = "directional";
		DirectionalLight *light = cast_to<DirectionalLight>(p_light);
		l->intensity = light->get_param(DirectionalLight::PARAM_ENERGY);
		l->range = FLT_MAX; // Range for directional lights is infinite in Godot.
	} else if (cast_to<OmniLight>(p_light)) {
		l->type = "point";
		OmniLight *light = cast_to<OmniLight>(p_light);
		l->range = light->get_param(OmniLight::PARAM_RANGE);
		l->intensity = light->get_param(OmniLight::PARAM_ENERGY);
	} else if (cast_to<SpotLight>(p_light)) {
		l->type = "spot";
		SpotLight *light = cast_to<SpotLight>(p_light);
		l->range = light->get_param(SpotLight::PARAM_RANGE);
		l->intensity = light->get_param(SpotLight::PARAM_ENERGY);
		l->outer_cone_angle = Math::deg2rad(light->get_param(SpotLight::PARAM_SPOT_ANGLE));

		// This equation is the inverse of the import equation (which has a desmos link).
		float angle_ratio = 1 - (0.2 / (0.1 + light->get_param(SpotLight::PARAM_SPOT_ATTENUATION)));
		angle_ratio = MAX(0, angle_ratio);
		l->inner_cone_angle = l->outer_cone_angle * angle_ratio;
	}

	GLTFLightIndex light_index = p_state->lights.size();
	p_state->lights.push_back(l);
	return light_index;
}

void GLTFDocument::_convert_spatial(Ref<GLTFState> p_state, Spatial *p_spatial, Ref<GLTFNode> p_node) {
	Transform xform = p_spatial->get_transform();
	p_node->scale = xform.basis.get_scale();
	p_node->rotation = xform.basis.get_rotation_quat();
	p_node->translation = xform.origin;
}

Spatial *GLTFDocument::_generate_spatial(Ref<GLTFState> p_state, Node *p_scene_parent, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	Spatial *spatial = memnew(Spatial);
	print_verbose("glTF: Converting spatial: " + gltf_node->get_name());

	return spatial;
}

void GLTFDocument::_convert_scene_node(Ref<GLTFState> p_state, Node *p_current, const GLTFNodeIndex p_gltf_parent, const GLTFNodeIndex p_gltf_root) {
	bool retflag = true;
	_check_visibility(p_current, retflag);
	if (retflag) {
		return;
	}
	Ref<GLTFNode> gltf_node;
	gltf_node.instance();
	gltf_node->set_name(_gen_unique_name(p_state, p_current->get_name()));
	if (cast_to<Spatial>(p_current)) {
		Spatial *spatial = cast_to<Spatial>(p_current);
		_convert_spatial(p_state, spatial, gltf_node);
	}
	if (cast_to<MeshInstance>(p_current)) {
		MeshInstance *mi = cast_to<MeshInstance>(p_current);
		_convert_mesh_instance_to_gltf(mi, p_state, gltf_node);
	} else if (cast_to<BoneAttachment>(p_current)) {
		BoneAttachment *bone = cast_to<BoneAttachment>(p_current);
		_convert_bone_attachment_to_gltf(bone, p_state, p_gltf_parent, p_gltf_root, gltf_node);
		return;
	} else if (cast_to<Skeleton>(p_current)) {
		Skeleton *skel = cast_to<Skeleton>(p_current);
		_convert_skeleton_to_gltf(skel, p_state, p_gltf_parent, p_gltf_root, gltf_node);
		// We ignore the Godot Engine node that is the skeleton.
		return;
	} else if (cast_to<MultiMeshInstance>(p_current)) {
		MultiMeshInstance *multi = cast_to<MultiMeshInstance>(p_current);
		_convert_multi_mesh_instance_to_gltf(multi, p_gltf_parent, p_gltf_root, gltf_node, p_state);
#ifdef MODULE_CSG_ENABLED
	} else if (cast_to<CSGShape>(p_current)) {
		CSGShape *shape = cast_to<CSGShape>(p_current);
		if (shape->get_parent() && shape->is_root_shape()) {
			_convert_csg_shape_to_gltf(shape, p_gltf_parent, gltf_node, p_state);
		}
#endif // MODULE_CSG_ENABLED
#ifdef MODULE_GRIDMAP_ENABLED
	} else if (cast_to<GridMap>(p_current)) {
		GridMap *gridmap = Object::cast_to<GridMap>(p_current);
		_convert_grid_map_to_gltf(gridmap, p_gltf_parent, p_gltf_root, gltf_node, p_state);
#endif // MODULE_GRIDMAP_ENABLED
	} else if (cast_to<Camera>(p_current)) {
		Camera *camera = Object::cast_to<Camera>(p_current);
		_convert_camera_to_gltf(camera, p_state, gltf_node);
	} else if (cast_to<Light>(p_current)) {
		Light *light = Object::cast_to<Light>(p_current);
		_convert_light_to_gltf(light, p_state, gltf_node);
	} else if (cast_to<AnimationPlayer>(p_current)) {
		AnimationPlayer *animation_player = Object::cast_to<AnimationPlayer>(p_current);
		_convert_animation_player_to_gltf(animation_player, p_state, p_gltf_parent, p_gltf_root, gltf_node, p_current);
	}
	GLTFNodeIndex current_node_i = p_state->nodes.size();
	GLTFNodeIndex gltf_root = p_gltf_root;
	if (gltf_root == -1) {
		gltf_root = current_node_i;
		Array scenes;
		scenes.push_back(gltf_root);
		p_state->json["scene"] = scenes;
	}
	_create_gltf_node(p_state, p_current, current_node_i, p_gltf_parent, gltf_root, gltf_node);
	for (int node_i = 0; node_i < p_current->get_child_count(); node_i++) {
		_convert_scene_node(p_state, p_current->get_child(node_i), current_node_i, gltf_root);
	}
}

#ifdef MODULE_CSG_ENABLED
void GLTFDocument::_convert_csg_shape_to_gltf(CSGShape *p_current, GLTFNodeIndex p_gltf_parent, Ref<GLTFNode> p_gltf_node, Ref<GLTFState> p_state) {
	CSGShape *csg = p_current;
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
	gltf_mesh.instance();
	Ref<ArrayMesh> import_mesh;
	import_mesh.instance();
	Ref<ArrayMesh> array_mesh = csg->get_meshes()[1];
	for (int32_t surface_i = 0; surface_i < array_mesh->get_surface_count(); surface_i++) {
		import_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, array_mesh->surface_get_arrays(surface_i));
	}
	gltf_mesh->set_mesh(import_mesh);
	GLTFMeshIndex mesh_i = p_state->meshes.size();
	p_state->meshes.push_back(gltf_mesh);
	p_gltf_node->mesh = mesh_i;
	p_gltf_node->xform = csg->get_meshes()[0];
	p_gltf_node->set_name(_gen_unique_name(p_state, csg->get_name()));
}
#endif // MODULE_CSG_ENABLED

void GLTFDocument::_create_gltf_node(Ref<GLTFState> p_state, Node *p_scene_parent, GLTFNodeIndex current_node_i,
		GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_gltf_node, Ref<GLTFNode> p_gltf_node) {
	p_state->scene_nodes.insert(current_node_i, p_scene_parent);
	p_state->nodes.push_back(p_gltf_node);
	ERR_FAIL_COND(current_node_i == p_parent_node_index);
	p_state->nodes.write[current_node_i]->parent = p_parent_node_index;
	if (p_parent_node_index == -1) {
		return;
	}
	p_state->nodes.write[p_parent_node_index]->children.push_back(current_node_i);
}

void GLTFDocument::_convert_animation_player_to_gltf(AnimationPlayer *p_animation_player, Ref<GLTFState> p_state, GLTFNodeIndex p_gltf_current, GLTFNodeIndex p_gltf_root_index, Ref<GLTFNode> p_gltf_node, Node *p_scene_parent) {
	ERR_FAIL_COND(!p_animation_player);
	p_state->animation_players.push_back(p_animation_player);
	print_verbose(String("glTF: Converting animation player: ") + p_animation_player->get_name());
}

void GLTFDocument::_check_visibility(Node *p_node, bool &r_retflag) {
	r_retflag = true;
	Spatial *spatial = Object::cast_to<Spatial>(p_node);
	Node2D *node_2d = Object::cast_to<Node2D>(p_node);
	if (node_2d && !node_2d->is_visible()) {
		return;
	}
	if (spatial && !spatial->is_visible()) {
		return;
	}
	r_retflag = false;
}

void GLTFDocument::_convert_camera_to_gltf(Camera *p_camera, Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node) {
	ERR_FAIL_COND(!p_camera);
	GLTFCameraIndex camera_index = _convert_camera(p_state, p_camera);
	if (camera_index != -1) {
		p_gltf_node->camera = camera_index;
	}
}

void GLTFDocument::_convert_light_to_gltf(Light *p_light, Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node) {
	ERR_FAIL_COND(!p_light);
	GLTFLightIndex light_index = _convert_light(p_state, p_light);
	if (light_index != -1) {
		p_gltf_node->light = light_index;
	}
}

#ifdef MODULE_GRIDMAP_ENABLED
void GLTFDocument::_convert_grid_map_to_gltf(GridMap *p_grid_map, GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_node_index, Ref<GLTFNode> p_gltf_node, Ref<GLTFState> p_state) {
	Array cells = p_grid_map->get_used_cells();
	for (int32_t k = 0; k < cells.size(); k++) {
		GLTFNode *new_gltf_node = memnew(GLTFNode);
		p_gltf_node->children.push_back(p_state->nodes.size());
		p_state->nodes.push_back(new_gltf_node);
		Vector3 cell_location = cells[k];
		int32_t cell = p_grid_map->get_cell_item(
				cell_location.x, cell_location.y, cell_location.z);
		Transform cell_xform;
		cell_xform.basis.set_orthogonal_index(
				p_grid_map->get_cell_item_orientation(
						cell_location.x, cell_location.y, cell_location.z));
		cell_xform.basis.scale(Vector3(p_grid_map->get_cell_scale(),
				p_grid_map->get_cell_scale(),
				p_grid_map->get_cell_scale()));
		cell_xform.set_origin(p_grid_map->map_to_world(
				cell_location.x, cell_location.y, cell_location.z));
		Ref<GLTFMesh> gltf_mesh;
		gltf_mesh.instance();
		gltf_mesh->set_mesh(_mesh_to_array_mesh(p_grid_map->get_mesh_library()->get_item_mesh(cell)));
		new_gltf_node->mesh = p_state->meshes.size();
		p_state->meshes.push_back(gltf_mesh);
		new_gltf_node->xform = cell_xform * p_grid_map->get_transform();
		new_gltf_node->set_name(_gen_unique_name(p_state, p_grid_map->get_mesh_library()->get_item_name(cell)));
	}
}
#endif // MODULE_GRIDMAP_ENABLED

void GLTFDocument::_convert_multi_mesh_instance_to_gltf(MultiMeshInstance *p_multi_mesh_instance, GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_node_index, Ref<GLTFNode> p_gltf_node, Ref<GLTFState> p_state) {
	Ref<MultiMesh> multi_mesh = p_multi_mesh_instance->get_multimesh();
	if (multi_mesh.is_valid()) {
		for (int32_t instance_i = 0; instance_i < multi_mesh->get_instance_count();
				instance_i++) {
			GLTFNode *new_gltf_node = memnew(GLTFNode);
			Transform transform;
			if (multi_mesh->get_transform_format() == MultiMesh::TRANSFORM_2D) {
				Transform2D xform_2d = multi_mesh->get_instance_transform_2d(instance_i);
				transform.origin =
						Vector3(xform_2d.get_origin().x, 0, xform_2d.get_origin().y);
				real_t rotation = xform_2d.get_rotation();
				Quat quat(Vector3(0, 1, 0), rotation);
				Size2 scale = xform_2d.get_scale();
				transform.basis.set_quat_scale(quat,
						Vector3(scale.x, 0, scale.y));
				transform =
						p_multi_mesh_instance->get_transform() * transform;
			} else if (multi_mesh->get_transform_format() == MultiMesh::TRANSFORM_3D) {
				transform = p_multi_mesh_instance->get_transform() *
						multi_mesh->get_instance_transform(instance_i);
			}
			Ref<ArrayMesh> mm = multi_mesh->get_mesh();
			if (mm.is_valid()) {
				Ref<ArrayMesh> mesh;
				mesh.instance();
				for (int32_t surface_i = 0; surface_i < mm->get_surface_count(); surface_i++) {
					Array surface = mm->surface_get_arrays(surface_i);
					mesh->add_surface_from_arrays(mm->surface_get_primitive_type(surface_i), surface);
				}
				Ref<GLTFMesh> gltf_mesh;
				gltf_mesh.instance();
				gltf_mesh->set_name(multi_mesh->get_name());
				gltf_mesh->set_mesh(mesh);
				new_gltf_node->mesh = p_state->meshes.size();
				p_state->meshes.push_back(gltf_mesh);
			}
			new_gltf_node->xform = transform;
			new_gltf_node->set_name(_gen_unique_name(p_state, p_multi_mesh_instance->get_name()));
			p_gltf_node->children.push_back(p_state->nodes.size());
			p_state->nodes.push_back(new_gltf_node);
		}
	}
}

void GLTFDocument::_convert_skeleton_to_gltf(Skeleton *p_skeleton3d, Ref<GLTFState> p_state, GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_node_index, Ref<GLTFNode> p_gltf_node) {
	Skeleton *skeleton = p_skeleton3d;
	Ref<GLTFSkeleton> gltf_skeleton;
	gltf_skeleton.instance();
	// GLTFSkeleton is only used to hold internal p_state data. It will not be written to the document.
	//
	gltf_skeleton->godot_skeleton = skeleton;
	GLTFSkeletonIndex skeleton_i = p_state->skeletons.size();
	p_state->skeleton3d_to_gltf_skeleton[skeleton->get_instance_id()] = skeleton_i;
	p_state->skeletons.push_back(gltf_skeleton);

	BoneId bone_count = skeleton->get_bone_count();
	for (BoneId bone_i = 0; bone_i < bone_count; bone_i++) {
		Ref<GLTFNode> joint_node;
		joint_node.instance();
		// Note that we cannot use _gen_unique_bone_name here, because glTF spec requires all node
		// names to be unique regardless of whether or not they are used as joints.
		joint_node->set_name(_gen_unique_name(p_state, skeleton->get_bone_name(bone_i)));
		Transform xform = skeleton->get_bone_rest(bone_i) * skeleton->get_bone_pose(bone_i);
		joint_node->scale = xform.basis.get_scale();
		joint_node->rotation = xform.basis.get_rotation_quat();
		joint_node->translation = xform.origin;
		joint_node->joint = true;
		GLTFNodeIndex current_node_i = p_state->nodes.size();
		p_state->scene_nodes.insert(current_node_i, skeleton);
		p_state->nodes.push_back(joint_node);

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
				p_state->nodes.write[current_node_i]->parent = p_parent_node_index;
				p_state->nodes.write[p_parent_node_index]->children.push_back(current_node_i);
			}
		} else {
			GLTFNodeIndex parent_node_i = gltf_skeleton->godot_bone_node[parent_bone_id];
			p_state->nodes.write[current_node_i]->parent = parent_node_i;
			p_state->nodes.write[parent_node_i]->children.push_back(current_node_i);
		}
	}
	// Remove placeholder skeleton3d node by not creating the gltf node
	// Skins are per mesh
	for (int node_i = 0; node_i < skeleton->get_child_count(); node_i++) {
		_convert_scene_node(p_state, skeleton->get_child(node_i), p_parent_node_index, p_root_node_index);
	}
}

void GLTFDocument::_convert_bone_attachment_to_gltf(BoneAttachment *p_bone_attachment, Ref<GLTFState> p_state, GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_node_index, Ref<GLTFNode> p_gltf_node) {
	Skeleton *skeleton;
	// Note that relative transforms to external skeletons and pose overrides are not supported.
	skeleton = cast_to<Skeleton>(p_bone_attachment->get_parent());
	GLTFSkeletonIndex skel_gltf_i = -1;
	if (skeleton != nullptr && p_state->skeleton3d_to_gltf_skeleton.has(skeleton->get_instance_id())) {
		skel_gltf_i = p_state->skeleton3d_to_gltf_skeleton[skeleton->get_instance_id()];
	}
	int bone_idx = -1;
	if (skeleton != nullptr) {
		bone_idx = skeleton->find_bone(p_bone_attachment->get_bone_name());
	}
	GLTFNodeIndex par_node_index = p_parent_node_index;
	if (skeleton != nullptr && bone_idx != -1 && skel_gltf_i != -1) {
		Ref<GLTFSkeleton> gltf_skeleton = p_state->skeletons.write[skel_gltf_i];
		gltf_skeleton->bone_attachments.push_back(p_bone_attachment);
		par_node_index = gltf_skeleton->joints[bone_idx];
	}

	for (int node_i = 0; node_i < p_bone_attachment->get_child_count(); node_i++) {
		_convert_scene_node(p_state, p_bone_attachment->get_child(node_i), par_node_index, p_root_node_index);
	}
}

void GLTFDocument::_convert_mesh_instance_to_gltf(MeshInstance *p_scene_parent, Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node) {
	GLTFMeshIndex gltf_mesh_index = _convert_mesh_to_gltf(p_state, p_scene_parent);
	if (gltf_mesh_index != -1) {
		p_gltf_node->mesh = gltf_mesh_index;
	}
}

void GLTFDocument::_generate_scene_node(Ref<GLTFState> p_state, Node *p_scene_parent, Spatial *p_scene_root, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	if (gltf_node->skeleton >= 0) {
		_generate_skeleton_bone_node(p_state, p_scene_parent, p_scene_root, p_node_index);
		return;
	}

	Spatial *current_node = nullptr;

	// Is our parent a skeleton
	Skeleton *active_skeleton = Object::cast_to<Skeleton>(p_scene_parent);

	const bool non_bone_parented_to_skeleton = active_skeleton;

	// If we have an active skeleton, and the node is node skinned, we need to create a bone attachment
	if (non_bone_parented_to_skeleton && gltf_node->skin < 0) {
		// Bone Attachment - Parent Case
		BoneAttachment *bone_attachment = _generate_bone_attachment(p_state, active_skeleton, p_node_index, gltf_node->parent);

		p_scene_parent->add_child(bone_attachment);
		bone_attachment->set_owner(p_scene_root);

		// There is no gltf_node that represent this, so just directly create a unique name
		bone_attachment->set_name(_gen_unique_name(p_state, "BoneAttachment"));

		// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
		// and attach it to the bone_attachment
		p_scene_parent = bone_attachment;
	}
	if (gltf_node->mesh >= 0) {
		current_node = _generate_mesh_instance(p_state, p_scene_parent, p_node_index);
	} else if (gltf_node->camera >= 0) {
		current_node = _generate_camera(p_state, p_scene_parent, p_node_index);
	} else if (gltf_node->light >= 0) {
		current_node = _generate_light(p_state, p_scene_parent, p_node_index);
	}

	// We still have not managed to make a node.
	if (!current_node) {
		current_node = _generate_spatial(p_state, p_scene_parent, p_node_index);
	}

	p_scene_parent->add_child(current_node);
	if (current_node != p_scene_root) {
		current_node->set_owner(p_scene_root);
	}
	current_node->set_transform(gltf_node->xform);
	current_node->set_name(gltf_node->get_name());

	p_state->scene_nodes.insert(p_node_index, current_node);

	for (int i = 0; i < gltf_node->children.size(); ++i) {
		_generate_scene_node(p_state, current_node, p_scene_root, gltf_node->children[i]);
	}
}

void GLTFDocument::_generate_skeleton_bone_node(Ref<GLTFState> p_state, Node *p_scene_parent, Spatial *p_scene_root, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	Spatial *current_node = nullptr;

	Skeleton *skeleton = p_state->skeletons[gltf_node->skeleton]->godot_skeleton;
	// In this case, this node is already a bone in skeleton.
	const bool is_skinned_mesh = (gltf_node->skin >= 0 && gltf_node->mesh >= 0);
	const bool requires_extra_node = (gltf_node->mesh >= 0 || gltf_node->camera >= 0 || gltf_node->light >= 0);

	Skeleton *active_skeleton = Object::cast_to<Skeleton>(p_scene_parent);
	if (active_skeleton != skeleton) {
		if (active_skeleton) {
			// Bone Attachment - Direct Parented Skeleton Case
			BoneAttachment *bone_attachment = _generate_bone_attachment(p_state, active_skeleton, p_node_index, gltf_node->parent);

			p_scene_parent->add_child(bone_attachment);
			bone_attachment->set_owner(p_scene_root);

			// There is no gltf_node that represent this, so just directly create a unique name
			bone_attachment->set_name(_gen_unique_name(p_state, "BoneAttachment"));

			// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
			// and attach it to the bone_attachment
			p_scene_parent = bone_attachment;
			WARN_PRINT(vformat("glTF: Generating scene detected direct parented Skeletons at node %d", p_node_index));
		}

		// Add it to the scene if it has not already been added
		if (skeleton->get_parent() == nullptr) {
			p_scene_parent->add_child(skeleton);
			skeleton->set_owner(p_scene_root);
		}
	}

	active_skeleton = skeleton;
	current_node = skeleton;

	if (requires_extra_node) {
		// skinned meshes must not be placed in a bone attachment.
		if (!is_skinned_mesh) {
			// Bone Attachment - Same Node Case
			BoneAttachment *bone_attachment = _generate_bone_attachment(p_state, active_skeleton, p_node_index, p_node_index);

			p_scene_parent->add_child(bone_attachment);
			bone_attachment->set_owner(p_scene_root);

			// There is no gltf_node that represent this, so just directly create a unique name
			bone_attachment->set_name(_gen_unique_name(p_state, "BoneAttachment"));

			// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
			// and attach it to the bone_attachment
			p_scene_parent = bone_attachment;
		}

		// We still have not managed to make a node
		if (gltf_node->mesh >= 0) {
			current_node = _generate_mesh_instance(p_state, p_scene_parent, p_node_index);
		} else if (gltf_node->camera >= 0) {
			current_node = _generate_camera(p_state, p_scene_parent, p_node_index);
		} else if (gltf_node->light >= 0) {
			current_node = _generate_light(p_state, p_scene_parent, p_node_index);
		}

		p_scene_parent->add_child(current_node);
		if (current_node != p_scene_root) {
			current_node->set_owner(p_scene_root);
		}
		// Do not set transform here. Transform is already applied to our bone.
		if (p_state->use_legacy_names) {
			current_node->set_name(_legacy_validate_node_name(gltf_node->get_name()));
		} else {
			current_node->set_name(gltf_node->get_name());
		}
	}

	p_state->scene_nodes.insert(p_node_index, current_node);

	for (int i = 0; i < gltf_node->children.size(); ++i) {
		_generate_scene_node(p_state, active_skeleton, p_scene_root, gltf_node->children[i]);
	}
}

template <class T>
struct EditorSceneImporterGLTFInterpolate {
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
struct EditorSceneImporterGLTFInterpolate<Quat> {
	Quat lerp(const Quat &a, const Quat &b, const float c) const {
		ERR_FAIL_COND_V_MSG(!a.is_normalized(), Quat(), "The quaternion \"a\" must be normalized.");
		ERR_FAIL_COND_V_MSG(!b.is_normalized(), Quat(), "The quaternion \"b\" must be normalized.");

		return a.slerp(b, c).normalized();
	}

	Quat catmull_rom(const Quat &p0, const Quat &p1, const Quat &p2, const Quat &p3, const float c) {
		ERR_FAIL_COND_V_MSG(!p1.is_normalized(), Quat(), "The quaternion \"p1\" must be normalized.");
		ERR_FAIL_COND_V_MSG(!p2.is_normalized(), Quat(), "The quaternion \"p2\" must be normalized.");

		return p1.slerp(p2, c).normalized();
	}

	Quat bezier(const Quat start, const Quat control_1, const Quat control_2, const Quat end, const float t) {
		ERR_FAIL_COND_V_MSG(!start.is_normalized(), Quat(), "The start quaternion must be normalized.");
		ERR_FAIL_COND_V_MSG(!end.is_normalized(), Quat(), "The end quaternion must be normalized.");

		return start.slerp(end, t).normalized();
	}
};

template <class T>
T GLTFDocument::_interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, const float p_time, const GLTFAnimation::Interpolation p_interp) {
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

	EditorSceneImporterGLTFInterpolate<T> interp;

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

void GLTFDocument::_import_animation(Ref<GLTFState> p_state, AnimationPlayer *p_animation_player, const GLTFAnimationIndex p_index, const int p_bake_fps) {
	Ref<GLTFAnimation> anim = p_state->animations[p_index];

	String name = anim->get_name();
	if (name.empty()) {
		// No node represent these, and they are not in the hierarchy, so just make a unique name
		name = _gen_unique_name(p_state, "Animation");
	}

	Ref<Animation> animation;
	animation.instance();
	animation->set_name(name);

	if (anim->get_loop()) {
		animation->set_loop(true);
	}

	float length = 0.0;

	for (Map<int, GLTFAnimation::Track>::Element *track_i = anim->get_tracks().front(); track_i; track_i = track_i->next()) {
		const GLTFAnimation::Track &track = track_i->get();
		//need to find the path: for skeletons, weight tracks will affect the mesh
		NodePath node_path;
		//for skeletons, transform tracks always affect bones
		NodePath transform_node_path;

		GLTFNodeIndex node_index = track_i->key();

		const Ref<GLTFNode> gltf_node = p_state->nodes[track_i->key()];

		Node *root = p_animation_player->get_parent();
		ERR_FAIL_COND(root == nullptr);
		Map<GLTFNodeIndex, Node *>::Element *node_element = p_state->scene_nodes.find(node_index);
		ERR_CONTINUE_MSG(node_element == nullptr, vformat("Unable to find node %d for animation", node_index));
		node_path = root->get_path_to(node_element->get());

		if (gltf_node->skeleton >= 0) {
			const Skeleton *sk = p_state->skeletons[gltf_node->skeleton]->godot_skeleton;
			ERR_FAIL_COND(sk == nullptr);

			const String path = p_animation_player->get_parent()->get_path_to(sk);
			const String bone = gltf_node->get_name();
			transform_node_path = path + ":" + bone;
		} else {
			transform_node_path = node_path;
		}

		for (int i = 0; i < track.rotation_track.times.size(); i++) {
			length = MAX(length, track.rotation_track.times[i]);
		}
		for (int i = 0; i < track.translation_track.times.size(); i++) {
			length = MAX(length, track.translation_track.times[i]);
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
		if ((track.rotation_track.values.size() || track.translation_track.values.size() || track.scale_track.values.size()) && !transform_affects_skinned_mesh_instance) {
			//make transform track
			int track_idx = animation->get_track_count();
			animation->add_track(Animation::TYPE_TRANSFORM);
			animation->track_set_path(track_idx, transform_node_path);
			//first determine animation length

			const double increment = 1.0 / p_bake_fps;
			double time = 0.0;

			Vector3 base_pos;
			Quat base_rot;
			Vector3 base_scale = Vector3(1, 1, 1);

			if (!track.rotation_track.values.size()) {
				base_rot = p_state->nodes[track_i->key()]->rotation.normalized();
			}

			if (!track.translation_track.values.size()) {
				base_pos = p_state->nodes[track_i->key()]->translation;
			}

			if (!track.scale_track.values.size()) {
				base_scale = p_state->nodes[track_i->key()]->scale;
			}

			bool last = false;
			while (true) {
				Vector3 pos = base_pos;
				Quat rot = base_rot;
				Vector3 scale = base_scale;

				if (track.translation_track.times.size()) {
					pos = _interpolate_track<Vector3>(track.translation_track.times, track.translation_track.values, time, track.translation_track.interpolation);
				}

				if (track.rotation_track.times.size()) {
					rot = _interpolate_track<Quat>(track.rotation_track.times, track.rotation_track.values, time, track.rotation_track.interpolation);
				}

				if (track.scale_track.times.size()) {
					scale = _interpolate_track<Vector3>(track.scale_track.times, track.scale_track.values, time, track.scale_track.interpolation);
				}

				if (gltf_node->skeleton >= 0) {
					Transform xform;
					xform.basis.set_quat_scale(rot, scale);
					xform.origin = pos;

					const Skeleton *skeleton = p_state->skeletons[gltf_node->skeleton]->godot_skeleton;
					const int bone_idx = skeleton->find_bone(gltf_node->get_name());
					xform = skeleton->get_bone_rest(bone_idx).affine_inverse() * xform;

					rot = xform.basis.get_rotation_quat();
					rot.normalize();
					scale = xform.basis.get_scale();
					pos = xform.origin;
				}

				animation->transform_track_insert_key(track_idx, time, pos, rot, scale);

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
			ERR_CONTINUE(gltf_node->mesh < 0 || gltf_node->mesh >= p_state->meshes.size());
			Ref<GLTFMesh> mesh = p_state->meshes[gltf_node->mesh];
			ERR_CONTINUE(mesh.is_null());
			ERR_CONTINUE(mesh->get_mesh().is_null());
			const String prop = "blend_shapes/" + mesh->get_mesh()->get_blend_shape_name(i);

			const String blend_path = String(node_path) + ":" + prop;

			const int track_idx = animation->get_track_count();
			animation->add_track(Animation::TYPE_VALUE);
			animation->track_set_path(track_idx, blend_path);

			// Only LINEAR and STEP (NEAREST) can be supported out of the box by Godot's Animation,
			// the other modes have to be baked.
			GLTFAnimation::Interpolation gltf_interp = track.weight_tracks[i].interpolation;
			if (gltf_interp == GLTFAnimation::INTERP_LINEAR || gltf_interp == GLTFAnimation::INTERP_STEP) {
				animation->track_set_interpolation_type(track_idx, gltf_interp == GLTFAnimation::INTERP_STEP ? Animation::INTERPOLATION_NEAREST : Animation::INTERPOLATION_LINEAR);
				for (int j = 0; j < track.weight_tracks[i].times.size(); j++) {
					const float t = track.weight_tracks[i].times[j];
					const float attribs = track.weight_tracks[i].values[j];
					animation->track_insert_key(track_idx, t, attribs);
				}
			} else {
				// CATMULLROMSPLINE or CUBIC_SPLINE have to be baked, apologies.
				const double increment = 1.0 / p_bake_fps;
				double time = 0.0;
				bool last = false;
				while (true) {
					_interpolate_track<float>(track.weight_tracks[i].times, track.weight_tracks[i].values, time, gltf_interp);
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

	p_animation_player->add_animation(name, animation);
}

void GLTFDocument::_convert_mesh_instances(Ref<GLTFState> p_state) {
	for (GLTFNodeIndex mi_node_i = 0; mi_node_i < p_state->nodes.size(); ++mi_node_i) {
		Ref<GLTFNode> node = p_state->nodes[mi_node_i];

		if (node->mesh < 0) {
			continue;
		}
		Map<GLTFNodeIndex, Node *>::Element *mi_element = p_state->scene_nodes.find(mi_node_i);
		if (!mi_element) {
			continue;
		}
		MeshInstance *mi = Object::cast_to<MeshInstance>(mi_element->get());
		ERR_CONTINUE(!mi);
		Transform mi_xform = mi->get_transform();
		node->scale = mi_xform.basis.get_scale();
		node->rotation = mi_xform.basis.get_rotation_quat();
		node->translation = mi_xform.origin;

		Skeleton *skeleton = Object::cast_to<Skeleton>(mi->get_node(mi->get_skeleton_path()));
		if (!skeleton) {
			continue;
		}
		if (!skeleton->get_bone_count()) {
			continue;
		}
		Ref<Skin> skin = mi->get_skin();
		Ref<GLTFSkin> gltf_skin;
		gltf_skin.instance();
		Array json_joints;

		NodePath skeleton_path = mi->get_skeleton_path();
		Node *skel_node = mi->get_node_or_null(skeleton_path);
		Skeleton *godot_skeleton = nullptr;
		if (skel_node != nullptr) {
			godot_skeleton = cast_to<Skeleton>(skel_node);
		}
		if (godot_skeleton != nullptr && p_state->skeleton3d_to_gltf_skeleton.has(godot_skeleton->get_instance_id())) {
			// This is a skinned mesh. If the mesh has no ARRAY_WEIGHTS or ARRAY_BONES, it will be invisible.
			const GLTFSkeletonIndex skeleton_gltf_i = p_state->skeleton3d_to_gltf_skeleton[godot_skeleton->get_instance_id()];
			Ref<GLTFSkeleton> gltf_skeleton = p_state->skeletons[skeleton_gltf_i];
			int bone_cnt = skeleton->get_bone_count();
			ERR_FAIL_COND(bone_cnt != gltf_skeleton->joints.size());

			ObjectID gltf_skin_key = 0;
			if (skin.is_valid()) {
				gltf_skin_key = skin->get_instance_id();
			}
			ObjectID gltf_skel_key = godot_skeleton->get_instance_id();
			GLTFSkinIndex skin_gltf_i = -1;
			GLTFNodeIndex root_gltf_i = -1;
			if (!gltf_skeleton->roots.empty()) {
				root_gltf_i = gltf_skeleton->roots[0];
			}
			if (p_state->skin_and_skeleton3d_to_gltf_skin.has(gltf_skin_key) && p_state->skin_and_skeleton3d_to_gltf_skin[gltf_skin_key].has(gltf_skel_key)) {
				skin_gltf_i = p_state->skin_and_skeleton3d_to_gltf_skin[gltf_skin_key][gltf_skel_key];
			} else {
				if (skin.is_null()) {
					// Note that gltf_skin_key should remain null, so these can share a reference.
					skin = skeleton->register_skin(nullptr)->get_skin();
				}
				gltf_skin.instance();
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
					Transform bind_pose = skin->get_bind_pose(bind_i);
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
				skin_gltf_i = p_state->skins.size();
				p_state->skins.push_back(gltf_skin);
				p_state->skin_and_skeleton3d_to_gltf_skin[gltf_skin_key][gltf_skel_key] = skin_gltf_i;
			}
			node->skin = skin_gltf_i;
			node->skeleton = skeleton_gltf_i;
		}
	}
}

float GLTFDocument::solve_metallic(float p_dielectric_specular, float p_diffuse, float p_specular, float p_one_minus_specular_strength) {
	if (p_specular <= p_dielectric_specular) {
		return 0.0f;
	}

	const float a = p_dielectric_specular;
	const float b = p_diffuse * p_one_minus_specular_strength / (1.0f - p_dielectric_specular) + p_specular - 2.0f * p_dielectric_specular;
	const float c = p_dielectric_specular - p_specular;
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

void GLTFDocument::_process_mesh_instances(Ref<GLTFState> p_state, Node *p_scene_root) {
	for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); ++node_i) {
		Ref<GLTFNode> node = p_state->nodes[node_i];

		if (node->skin >= 0 && node->mesh >= 0) {
			const GLTFSkinIndex skin_i = node->skin;

			Map<GLTFNodeIndex, Node *>::Element *mi_element = p_state->scene_nodes.find(node_i);
			ERR_CONTINUE_MSG(mi_element == nullptr, vformat("Unable to find node %d", node_i));

			MeshInstance *mi = Object::cast_to<MeshInstance>(mi_element->get());
			ERR_CONTINUE_MSG(mi == nullptr, vformat("Unable to cast node %d of type %s to MeshInstance", node_i, mi_element->get()->get_class_name()));

			const GLTFSkeletonIndex skel_i = p_state->skins.write[node->skin]->skeleton;
			Ref<GLTFSkeleton> gltf_skeleton = p_state->skeletons.write[skel_i];
			Skeleton *skeleton = gltf_skeleton->godot_skeleton;
			ERR_CONTINUE_MSG(skeleton == nullptr, vformat("Unable to find Skeleton for node %d skin %d", node_i, skin_i));

			mi->get_parent()->remove_child(mi);
			skeleton->add_child(mi);
			mi->set_owner(skeleton->get_owner());

			mi->set_skin(p_state->skins.write[skin_i]->godot_skin);
			mi->set_skeleton_path(mi->get_path_to(skeleton));
			mi->set_transform(Transform());
		}
	}
}

GLTFAnimation::Track GLTFDocument::_convert_animation_track(Ref<GLTFState> p_state, GLTFAnimation::Track p_track, Ref<Animation> p_animation, Transform p_bone_rest, int32_t p_track_i, GLTFNodeIndex p_node_i) {
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
	Vector<float> times;
	times.resize(key_count);
	String path = p_animation->track_get_path(p_track_i);
	for (int32_t key_i = 0; key_i < key_count; key_i++) {
		times.write[key_i] = p_animation->track_get_key_time(p_track_i, key_i);
	}
	if (track_type == Animation::TYPE_TRANSFORM) {
		p_track.translation_track.times = times;
		p_track.translation_track.interpolation = gltf_interpolation;
		p_track.rotation_track.times = times;
		p_track.rotation_track.interpolation = gltf_interpolation;
		p_track.scale_track.times = times;
		p_track.scale_track.interpolation = gltf_interpolation;

		p_track.scale_track.values.resize(key_count);
		p_track.scale_track.interpolation = gltf_interpolation;
		p_track.translation_track.values.resize(key_count);
		p_track.translation_track.interpolation = gltf_interpolation;
		p_track.rotation_track.values.resize(key_count);
		p_track.rotation_track.interpolation = gltf_interpolation;
		for (int32_t key_i = 0; key_i < key_count; key_i++) {
			Vector3 translation;
			Quat rotation;
			Vector3 scale;
			Error err = p_animation->transform_track_get_key(p_track_i, key_i, &translation, &rotation, &scale);
			ERR_CONTINUE(err != OK);
			Transform xform;
			xform.basis.set_quat_scale(rotation, scale);
			xform.origin = translation;
			xform = p_bone_rest * xform;
			p_track.translation_track.values.write[key_i] = xform.get_origin();
			p_track.rotation_track.values.write[key_i] = xform.basis.get_rotation_quat();
			p_track.scale_track.values.write[key_i] = xform.basis.get_scale();
		}
	} else if (path.find(":transform") != -1) {
		p_track.translation_track.times = times;
		p_track.translation_track.interpolation = gltf_interpolation;
		p_track.rotation_track.times = times;
		p_track.rotation_track.interpolation = gltf_interpolation;
		p_track.scale_track.times = times;
		p_track.scale_track.interpolation = gltf_interpolation;

		p_track.scale_track.values.resize(key_count);
		p_track.scale_track.interpolation = gltf_interpolation;
		p_track.translation_track.values.resize(key_count);
		p_track.translation_track.interpolation = gltf_interpolation;
		p_track.rotation_track.values.resize(key_count);
		p_track.rotation_track.interpolation = gltf_interpolation;
		for (int32_t key_i = 0; key_i < key_count; key_i++) {
			Transform xform = p_animation->track_get_key_value(p_track_i, key_i);
			p_track.translation_track.values.write[key_i] = xform.get_origin();
			p_track.rotation_track.values.write[key_i] = xform.basis.get_rotation_quat();
			p_track.scale_track.values.write[key_i] = xform.basis.get_scale();
		}
	} else if (track_type == Animation::TYPE_VALUE) {
		if (path.find("/rotation_quat") != -1) {
			p_track.rotation_track.times = times;
			p_track.rotation_track.interpolation = gltf_interpolation;

			p_track.rotation_track.values.resize(key_count);
			p_track.rotation_track.interpolation = gltf_interpolation;

			for (int32_t key_i = 0; key_i < key_count; key_i++) {
				Quat rotation_track = p_animation->track_get_key_value(p_track_i, key_i);
				p_track.rotation_track.values.write[key_i] = rotation_track;
			}
		} else if (path.find(":translation") != -1) {
			p_track.translation_track.times = times;
			p_track.translation_track.interpolation = gltf_interpolation;

			p_track.translation_track.values.resize(key_count);
			p_track.translation_track.interpolation = gltf_interpolation;

			for (int32_t key_i = 0; key_i < key_count; key_i++) {
				Vector3 translation = p_animation->track_get_key_value(p_track_i, key_i);
				p_track.translation_track.values.write[key_i] = translation;
			}
		} else if (path.find(":rotation_degrees") != -1) {
			p_track.rotation_track.times = times;
			p_track.rotation_track.interpolation = gltf_interpolation;

			p_track.rotation_track.values.resize(key_count);
			p_track.rotation_track.interpolation = gltf_interpolation;

			for (int32_t key_i = 0; key_i < key_count; key_i++) {
				Vector3 rotation_degrees = p_animation->track_get_key_value(p_track_i, key_i);
				Vector3 rotation_radian;
				rotation_radian.x = Math::deg2rad(rotation_degrees.x);
				rotation_radian.y = Math::deg2rad(rotation_degrees.y);
				rotation_radian.z = Math::deg2rad(rotation_degrees.z);
				p_track.rotation_track.values.write[key_i] = Quat(rotation_radian);
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
				Vector<float> new_times;
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
					bezier_track.x = p_bone_rest.affine_inverse().basis.get_scale().x * bezier_track.x;
				} else if (path.find("/scale:y") != -1) {
					bezier_track.y = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
					bezier_track.y = p_bone_rest.affine_inverse().basis.get_scale().y * bezier_track.y;
				} else if (path.find("/scale:z") != -1) {
					bezier_track.z = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
					bezier_track.z = p_bone_rest.affine_inverse().basis.get_scale().z * bezier_track.z;
				}
				p_track.scale_track.values.write[key_i] = bezier_track;
			}
		} else if (path.find("/translation") != -1) {
			const int32_t keys = p_animation->track_get_key_time(p_track_i, key_count - 1) * BAKE_FPS;
			if (!p_track.translation_track.times.size()) {
				Vector<float> new_times;
				new_times.resize(keys);
				for (int32_t key_i = 0; key_i < keys; key_i++) {
					new_times.write[key_i] = key_i / BAKE_FPS;
				}
				p_track.translation_track.times = new_times;
				p_track.translation_track.interpolation = gltf_interpolation;

				p_track.translation_track.values.resize(keys);
				p_track.translation_track.interpolation = gltf_interpolation;
			}

			for (int32_t key_i = 0; key_i < keys; key_i++) {
				Vector3 bezier_track = p_track.translation_track.values[key_i];
				if (path.find("/translation:x") != -1) {
					bezier_track.x = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
					bezier_track.x = p_bone_rest.affine_inverse().origin.x * bezier_track.x;
				} else if (path.find("/translation:y") != -1) {
					bezier_track.y = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
					bezier_track.y = p_bone_rest.affine_inverse().origin.y * bezier_track.y;
				} else if (path.find("/translation:z") != -1) {
					bezier_track.z = p_animation->bezier_track_interpolate(p_track_i, key_i / BAKE_FPS);
					bezier_track.z = p_bone_rest.affine_inverse().origin.z * bezier_track.z;
				}
				p_track.translation_track.values.write[key_i] = bezier_track;
			}
		}
	}

	return p_track;
}

void GLTFDocument::_convert_animation(Ref<GLTFState> p_state, AnimationPlayer *p_animation_player, String p_animation_track_name) {
	Ref<Animation> animation = p_animation_player->get_animation(p_animation_track_name);
	Ref<GLTFAnimation> gltf_animation;
	gltf_animation.instance();
	gltf_animation->set_name(_gen_unique_name(p_state, p_animation_track_name));

	for (int32_t track_i = 0; track_i < animation->get_track_count(); track_i++) {
		if (!animation->track_is_enabled(track_i)) {
			continue;
		}
		String orig_track_path = animation->track_get_path(track_i);
		if (String(orig_track_path).find(":translation") != -1) {
			const Vector<String> node_suffix = String(orig_track_path).split(":translation");
			const NodePath path = node_suffix[0];
			const Node *node = p_animation_player->get_parent()->get_node_or_null(path);
			for (Map<GLTFNodeIndex, Node *>::Element *translation_scene_node_i = p_state->scene_nodes.front(); translation_scene_node_i; translation_scene_node_i = translation_scene_node_i->next()) {
				if (translation_scene_node_i->get() == node) {
					GLTFNodeIndex node_index = translation_scene_node_i->key();
					Map<int, GLTFAnimation::Track>::Element *translation_track_i = gltf_animation->get_tracks().find(node_index);
					GLTFAnimation::Track track;
					if (translation_track_i) {
						track = translation_track_i->get();
					}
					track = _convert_animation_track(p_state, track, animation, Transform(), track_i, node_index);
					gltf_animation->get_tracks().insert(node_index, track);
				}
			}
		} else if (String(orig_track_path).find(":rotation_degrees") != -1) {
			const Vector<String> node_suffix = String(orig_track_path).split(":rotation_degrees");
			const NodePath path = node_suffix[0];
			const Node *node = p_animation_player->get_parent()->get_node_or_null(path);
			for (Map<GLTFNodeIndex, Node *>::Element *rotation_degree_scene_node_i = p_state->scene_nodes.front(); rotation_degree_scene_node_i; rotation_degree_scene_node_i = rotation_degree_scene_node_i->next()) {
				if (rotation_degree_scene_node_i->get() == node) {
					GLTFNodeIndex node_index = rotation_degree_scene_node_i->key();
					Map<int, GLTFAnimation::Track>::Element *rotation_degree_track_i = gltf_animation->get_tracks().find(node_index);
					GLTFAnimation::Track track;
					if (rotation_degree_track_i) {
						track = rotation_degree_track_i->get();
					}
					track = _convert_animation_track(p_state, track, animation, Transform(), track_i, node_index);
					gltf_animation->get_tracks().insert(node_index, track);
				}
			}
		} else if (String(orig_track_path).find(":scale") != -1) {
			const Vector<String> node_suffix = String(orig_track_path).split(":scale");
			const NodePath path = node_suffix[0];
			const Node *node = p_animation_player->get_parent()->get_node_or_null(path);
			for (Map<GLTFNodeIndex, Node *>::Element *scale_scene_node_i = p_state->scene_nodes.front(); scale_scene_node_i; scale_scene_node_i = scale_scene_node_i->next()) {
				if (scale_scene_node_i->get() == node) {
					GLTFNodeIndex node_index = scale_scene_node_i->key();
					Map<int, GLTFAnimation::Track>::Element *scale_track_i = gltf_animation->get_tracks().find(node_index);
					GLTFAnimation::Track track;
					if (scale_track_i) {
						track = scale_track_i->get();
					}
					track = _convert_animation_track(p_state, track, animation, Transform(), track_i, node_index);
					gltf_animation->get_tracks().insert(node_index, track);
				}
			}
		} else if (String(orig_track_path).find(":transform") != -1) {
			const Vector<String> node_suffix = String(orig_track_path).split(":transform");
			const NodePath path = node_suffix[0];
			const Node *node = p_animation_player->get_parent()->get_node_or_null(path);
			for (Map<GLTFNodeIndex, Node *>::Element *transform_track_i = p_state->scene_nodes.front(); transform_track_i; transform_track_i = transform_track_i->next()) {
				if (transform_track_i->get() == node) {
					GLTFAnimation::Track track;
					track = _convert_animation_track(p_state, track, animation, Transform(), track_i, transform_track_i->key());
					gltf_animation->get_tracks().insert(transform_track_i->key(), track);
				}
			}
		} else if (String(orig_track_path).find(":blend_shapes/") != -1) {
			const Vector<String> node_suffix = String(orig_track_path).split(":blend_shapes/");
			const NodePath path = node_suffix[0];
			const String suffix = node_suffix[1];
			Node *node = p_animation_player->get_parent()->get_node_or_null(path);
			MeshInstance *mi = cast_to<MeshInstance>(node);
			Ref<Mesh> mesh = mi->get_mesh();
			ERR_CONTINUE(mesh.is_null());
			int32_t mesh_index = -1;
			for (Map<GLTFNodeIndex, Node *>::Element *mesh_track_i = p_state->scene_nodes.front(); mesh_track_i; mesh_track_i = mesh_track_i->next()) {
				if (mesh_track_i->get() == node) {
					mesh_index = mesh_track_i->key();
				}
			}
			ERR_CONTINUE(mesh_index == -1);
			Map<int, GLTFAnimation::Track> &tracks = gltf_animation->get_tracks();
			GLTFAnimation::Track track = gltf_animation->get_tracks().has(mesh_index) ? gltf_animation->get_tracks()[mesh_index] : GLTFAnimation::Track();
			if (!tracks.has(mesh_index)) {
				for (int32_t shape_i = 0; shape_i < mesh->get_blend_shape_count(); shape_i++) {
					String shape_name = mesh->get_blend_shape_name(shape_i);
					NodePath shape_path = String(path) + ":blend_shapes/" + shape_name;
					int32_t shape_track_i = animation->find_track(shape_path);
					if (shape_track_i == -1) {
						GLTFAnimation::Channel<float> weight;
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
					GLTFAnimation::Channel<float> weight;
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
			Node *godot_node = p_animation_player->get_parent()->get_node_or_null(node_path);
			Skeleton *skeleton = nullptr;
			GLTFSkeletonIndex skeleton_gltf_i = -1;
			for (GLTFSkeletonIndex skeleton_i = 0; skeleton_i < p_state->skeletons.size(); skeleton_i++) {
				if (p_state->skeletons[skeleton_i]->godot_skeleton == cast_to<Skeleton>(godot_node)) {
					skeleton = p_state->skeletons[skeleton_i]->godot_skeleton;
					skeleton_gltf_i = skeleton_i;
					ERR_CONTINUE(!skeleton);
					Ref<GLTFSkeleton> skeleton_gltf = p_state->skeletons[skeleton_gltf_i];
					int32_t bone = skeleton->find_bone(suffix);
					ERR_CONTINUE(bone == -1);
					Transform xform = skeleton->get_bone_rest(bone);
					if (!skeleton_gltf->godot_bone_node.has(bone)) {
						continue;
					}
					GLTFNodeIndex node_i = skeleton_gltf->godot_bone_node[bone];
					Map<int, GLTFAnimation::Track>::Element *property_track_i = gltf_animation->get_tracks().find(node_i);
					GLTFAnimation::Track track;
					if (property_track_i) {
						track = property_track_i->get();
					}
					track = _convert_animation_track(p_state, track, animation, xform, track_i, node_i);
					gltf_animation->get_tracks()[node_i] = track;
				}
			}
		} else if (String(orig_track_path).find(":") == -1) {
			ERR_CONTINUE(!p_animation_player->get_parent());
			for (int32_t node_i = 0; node_i < p_animation_player->get_parent()->get_child_count(); node_i++) {
				const Node *child = p_animation_player->get_parent()->get_child(node_i);
				const Node *node = child->get_node_or_null(orig_track_path);
				for (Map<GLTFNodeIndex, Node *>::Element *scene_node_i = p_state->scene_nodes.front(); scene_node_i; scene_node_i = scene_node_i->next()) {
					if (scene_node_i->get() == node) {
						GLTFNodeIndex node_index = scene_node_i->key();
						Map<int, GLTFAnimation::Track>::Element *node_track_i = gltf_animation->get_tracks().find(node_index);
						GLTFAnimation::Track track;
						if (node_track_i) {
							track = node_track_i->get();
						}
						track = _convert_animation_track(p_state, track, animation, Transform(), track_i, node_index);
						gltf_animation->get_tracks().insert(node_index, track);
						break;
					}
				}
			}
		}
	}
	if (gltf_animation->get_tracks().size()) {
		p_state->animations.push_back(gltf_animation);
	}
}

Error GLTFDocument::parse(Ref<GLTFState> p_state, String p_path, bool p_read_binary) {
	Error err;
	FileAccessRef file = FileAccess::open(p_path, FileAccess::READ, &err);
	if (!file) {
		return err;
	}
	uint32_t magic = file->get_32();
	if (magic == 0x46546C67) {
		//binary file
		//text file
		err = _parse_glb(p_path, p_state);
		if (err) {
			return FAILED;
		}
	} else {
		//text file
		err = _parse_json(p_path, p_state);
		if (err) {
			return FAILED;
		}
	}
	file->close();

	// get file's name, use for scene name if none
	p_state->filename = p_path.get_file().get_slice(".", 0);

	ERR_FAIL_COND_V(!p_state->json.has("asset"), Error::FAILED);

	Dictionary asset = p_state->json["asset"];

	ERR_FAIL_COND_V(!asset.has("version"), Error::FAILED);

	String version = asset["version"];

	p_state->major_version = version.get_slice(".", 0).to_int();
	p_state->minor_version = version.get_slice(".", 1).to_int();

	/* PARSE EXTENSIONS */

	err = _parse_gltf_extensions(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE SCENE */
	err = _parse_scenes(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE NODES */
	err = _parse_nodes(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE BUFFERS */
	err = _parse_buffers(p_state, p_path.get_base_dir());
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE BUFFER VIEWS */
	err = _parse_buffer_views(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE ACCESSORS */
	err = _parse_accessors(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE IMAGES */
	err = _parse_images(p_state, p_path.get_base_dir());
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE TEXTURE SAMPLERS */
	err = _parse_texture_samplers(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE TEXTURES */
	err = _parse_textures(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE MATERIALS */
	err = _parse_materials(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE SKINS */
	err = _parse_skins(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* DETERMINE SKELETONS */
	err = _determine_skeletons(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* CREATE SKELETONS */
	err = _create_skeletons(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* CREATE SKINS */
	err = _create_skins(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE MESHES (we have enough info now) */
	err = _parse_meshes(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE LIGHTS */
	err = _parse_lights(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE CAMERAS */
	err = _parse_cameras(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* PARSE ANIMATIONS */
	err = _parse_animations(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* ASSIGN SCENE NAMES */
	_assign_scene_names(p_state);

	return OK;
}

Dictionary _serialize_texture_transform_uv(Vector2 p_offset, Vector2 p_scale) {
	Dictionary texture_transform;
	bool is_offset = p_offset != Vector2(0.0, 0.0);
	if (is_offset) {
		Array offset;
		offset.resize(2);
		offset[0] = p_offset.x;
		offset[1] = p_offset.y;
		texture_transform["offset"] = offset;
	}
	bool is_scaled = p_scale != Vector2(1.0, 1.0);
	if (is_scaled) {
		Array scale;
		scale.resize(2);
		scale[0] = p_scale.x;
		scale[1] = p_scale.y;
		texture_transform["scale"] = scale;
	}
	Dictionary extension;
	// Note: Godot doesn't support texture rotation.
	if (is_offset || is_scaled) {
		extension["KHR_texture_transform"] = texture_transform;
	}
	return extension;
}

Dictionary GLTFDocument::_serialize_texture_transform_uv1(Ref<SpatialMaterial> p_material) {
	if (p_material.is_valid()) {
		Vector3 offset = p_material->get_uv1_offset();
		Vector3 scale = p_material->get_uv1_scale();
		return _serialize_texture_transform_uv(Vector2(offset.x, offset.y), Vector2(scale.x, scale.y));
	}
	return Dictionary();
}

Dictionary GLTFDocument::_serialize_texture_transform_uv2(Ref<SpatialMaterial> p_material) {
	if (p_material.is_valid()) {
		Vector3 offset = p_material->get_uv2_offset();
		Vector3 scale = p_material->get_uv2_scale();
		return _serialize_texture_transform_uv(Vector2(offset.x, offset.y), Vector2(scale.x, scale.y));
	}
	return Dictionary();
}

Error GLTFDocument::_serialize_version(Ref<GLTFState> p_state) {
	const String version = "2.0";
	p_state->major_version = version.get_slice(".", 0).to_int();
	p_state->minor_version = version.get_slice(".", 1).to_int();
	Dictionary asset;
	asset["version"] = version;

	String hash = String(VERSION_HASH);
	asset["generator"] = String(VERSION_FULL_NAME) + String("@") + (hash.empty() ? String("unknown") : hash);
	p_state->json["asset"] = asset;
	ERR_FAIL_COND_V(!asset.has("version"), Error::FAILED);
	ERR_FAIL_COND_V(!p_state->json.has("asset"), Error::FAILED);
	return OK;
}

Error GLTFDocument::_serialize_file(Ref<GLTFState> p_state, const String p_path) {
	Error err = FAILED;
	if (p_path.to_lower().ends_with("glb")) {
		err = _encode_buffer_glb(p_state, p_path);
		ERR_FAIL_COND_V(err != OK, err);
		FileAccessRef file = FileAccess::open(p_path, FileAccess::WRITE, &err);
		ERR_FAIL_COND_V(!file, FAILED);

		String json = JSON::print(p_state->json);

		const uint32_t magic = 0x46546C67; // GLTF
		const int32_t header_size = 12;
		const int32_t chunk_header_size = 8;
		CharString cs = json.utf8();
		const uint32_t text_data_length = cs.length();
		const uint32_t text_chunk_length = ((text_data_length + 3) & (~3));
		const uint32_t text_chunk_type = 0x4E4F534A; //JSON

		uint32_t binary_data_length = 0;
		if (p_state->buffers.size()) {
			binary_data_length = p_state->buffers[0].size();
		}
		const uint32_t binary_chunk_length = ((binary_data_length + 3) & (~3));
		const uint32_t binary_chunk_type = 0x004E4942; //BIN

		file->create(FileAccess::ACCESS_RESOURCES);
		file->store_32(magic);
		file->store_32(p_state->major_version); // version
		file->store_32(header_size + chunk_header_size + text_chunk_length + chunk_header_size + binary_chunk_length); // length
		file->store_32(text_chunk_length);
		file->store_32(text_chunk_type);
		file->store_buffer((uint8_t *)&cs[0], cs.length());
		for (uint32_t pad_i = text_data_length; pad_i < text_chunk_length; pad_i++) {
			file->store_8(' ');
		}
		if (binary_chunk_length) {
			file->store_32(binary_chunk_length);
			file->store_32(binary_chunk_type);
			file->store_buffer(p_state->buffers[0].ptr(), binary_data_length);
		}
		for (uint32_t pad_i = binary_data_length; pad_i < binary_chunk_length; pad_i++) {
			file->store_8(0);
		}

		file->close();
	} else {
		err = _encode_buffer_bins(p_state, p_path);
		ERR_FAIL_COND_V(err != OK, err);
		FileAccessRef file = FileAccess::open(p_path, FileAccess::WRITE, &err);
		ERR_FAIL_COND_V(!file, FAILED);

		file->create(FileAccess::ACCESS_RESOURCES);
		String json = JSON::print(p_state->json);
		file->store_string(json);
		file->close();
	}
	return err;
}

Error GLTFDocument::_parse_gltf_extensions(Ref<GLTFState> p_state) {
	ERR_FAIL_COND_V(!p_state.is_valid(), ERR_PARSE_ERROR);
	if (p_state->json.has("extensionsRequired") && p_state->json["extensionsRequired"].get_type() == Variant::ARRAY) {
		Array extensions_required = p_state->json["extensionsRequired"];
		if (extensions_required.find("KHR_draco_mesh_compression") != -1) {
			ERR_PRINT("glTF2 extension KHR_draco_mesh_compression is not supported.");
			return ERR_UNAVAILABLE;
		}
	}
	return OK;
}

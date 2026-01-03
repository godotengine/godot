/**************************************************************************/
/*  gltf_document.cpp                                                     */
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

#include "gltf_document.h"

#include "extensions/gltf_document_extension_convert_importer_mesh.h"
#include "extensions/gltf_spec_gloss.h"
#include "gltf_state.h"
#include "gltf_template_convert.h"
#include "skin_tool.h"

#include "core/config/project_settings.h"
#include "core/crypto/crypto_core.h"
#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/file_access_memory.h"
#include "core/io/json.h"
#include "core/io/stream_peer.h"
#include "core/object/object_id.h"
#include "core/version.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/bone_attachment_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/multimesh_instance_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/3d/skin.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/portable_compressed_texture.h"
#include "scene/resources/surface_tool.h"

#ifdef TOOLS_ENABLED
#include "editor/file_system/editor_file_system.h"
#endif

#include "modules/modules_enabled.gen.h" // For csg, gridmap.

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif
#ifdef MODULE_GRIDMAP_ENABLED
#include "modules/gridmap/grid_map.h"
#endif

// FIXME: Hardcoded to avoid editor dependency.
#define GLTF_IMPORT_GENERATE_TANGENT_ARRAYS 8
#define GLTF_IMPORT_USE_NAMED_SKIN_BINDS 16
#define GLTF_IMPORT_DISCARD_MESHES_AND_MATERIALS 32
#define GLTF_IMPORT_FORCE_DISABLE_MESH_COMPRESSION 64

#include <cstdio>
#include <cstdlib>

static void _attach_extras_to_meta(const Dictionary &p_extras, Ref<Resource> p_node) {
	if (!p_extras.is_empty()) {
		p_node->set_meta("extras", p_extras);
	}
}

static void _attach_meta_to_extras(Ref<Resource> p_node, Dictionary &p_json) {
	if (p_node->has_meta("extras")) {
		Dictionary node_extras = p_node->get_meta("extras");
		if (p_json.has("extras")) {
			Dictionary extras = p_json["extras"];
			extras.merge(node_extras);
		} else {
			p_json["extras"] = node_extras;
		}
	}
}

Error GLTFDocument::_serialize(Ref<GLTFState> p_state) {
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		Error err = ext->export_preserialize(p_state);
		ERR_CONTINUE(err != OK);
	}

	/* STEP CONVERT MESH INSTANCES */
	_convert_mesh_instances(p_state);

	/* STEP SERIALIZE CAMERAS */
	Error err = _serialize_cameras(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP 3 CREATE SKINS */
	err = _serialize_skins(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE MESHES (we have enough info now) */
	err = _serialize_meshes(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE TEXTURES */
	err = _serialize_materials(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE TEXTURE SAMPLERS */
	err = _serialize_texture_samplers(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE ANIMATIONS */
	err = _serialize_animations(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE IMAGES */
	err = _serialize_images(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE TEXTURES */
	err = _serialize_textures(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE NODES */
	err = _serialize_nodes(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE SCENE */
	err = _serialize_scenes(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE LIGHTS */
	err = _serialize_lights(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE EXTENSIONS */
	err = _serialize_gltf_extensions(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE VERSION */
	err = _serialize_asset_header(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE ACCESSORS */
	err = _encode_accessors(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	/* STEP SERIALIZE BUFFER VIEWS */
	err = _encode_buffer_views(p_state);
	if (err != OK) {
		return Error::FAILED;
	}

	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		err = ext->export_post(p_state);
		ERR_FAIL_COND_V(err != OK, err);
	}

	return OK;
}

Error GLTFDocument::_serialize_gltf_extensions(Ref<GLTFState> p_state) const {
	Vector<String> extensions_used = p_state->extensions_used;
	Vector<String> extensions_required = p_state->extensions_required;
	if (!p_state->lights.is_empty()) {
		extensions_used.push_back("KHR_lights_punctual");
	}
	if (p_state->use_khr_texture_transform) {
		extensions_used.push_back("KHR_texture_transform");
		extensions_required.push_back("KHR_texture_transform");
	}
	if (!extensions_used.is_empty()) {
		extensions_used.sort();
		p_state->json["extensionsUsed"] = extensions_used;
	}
	if (!extensions_required.is_empty()) {
		extensions_required.sort();
		p_state->json["extensionsRequired"] = extensions_required;
	}
	return OK;
}

Error GLTFDocument::_serialize_scenes(Ref<GLTFState> p_state) {
	// Godot only supports one scene per glTF file.
	Array scenes;
	Dictionary scene_dict;
	scenes.append(scene_dict);
	p_state->json["scenes"] = scenes;
	p_state->json["scene"] = 0;
	// Add nodes to the scene dict.
	if (!p_state->root_nodes.is_empty()) {
		scene_dict["nodes"] = p_state->root_nodes;
	}
	if (!p_state->scene_name.is_empty()) {
		scene_dict["name"] = p_state->scene_name;
	}
	return OK;
}

Error GLTFDocument::_parse_json(const String &p_path, Ref<GLTFState> p_state) {
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ, &err);
	if (file.is_null()) {
		return err;
	}

	Vector<uint8_t> array;
	array.resize(file->get_length());
	file->get_buffer(array.ptrw(), array.size());
	String text = String::utf8((const char *)array.ptr(), array.size());

	JSON json;
	err = json.parse(text);
	if (err != OK) {
		_err_print_error("", p_path.utf8().get_data(), json.get_error_line(), json.get_error_message().utf8().get_data(), false, ERR_HANDLER_SCRIPT);
		return err;
	}
	p_state->json = json.get_data();

	return OK;
}

Error GLTFDocument::_parse_glb(Ref<FileAccess> p_file, Ref<GLTFState> p_state) {
	ERR_FAIL_COND_V(p_file.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_file->get_position() != 0, ERR_FILE_CANT_READ);
	uint32_t magic = p_file->get_32();
	ERR_FAIL_COND_V(magic != 0x46546C67, ERR_FILE_UNRECOGNIZED); //glTF
	p_file->get_32(); // version
	p_file->get_32(); // length
	uint32_t chunk_length = p_file->get_32();
	uint32_t chunk_type = p_file->get_32();

	ERR_FAIL_COND_V(chunk_type != 0x4E4F534A, ERR_PARSE_ERROR); //JSON
	Vector<uint8_t> json_data;
	json_data.resize(chunk_length);
	uint32_t len = p_file->get_buffer(json_data.ptrw(), chunk_length);
	ERR_FAIL_COND_V(len != chunk_length, ERR_FILE_CORRUPT);

	String text = String::utf8((const char *)json_data.ptr(), json_data.size());

	JSON json;
	Error err = json.parse(text);
	ERR_FAIL_COND_V_MSG(err != OK, err, "glTF Binary: Error parsing .glb file's JSON data: " + json.get_error_message() + " at line: " + itos(json.get_error_line()));

	p_state->json = json.get_data();

	//data?

	chunk_length = p_file->get_32();
	chunk_type = p_file->get_32();

	if (p_file->eof_reached()) {
		return OK; //all good
	}

	ERR_FAIL_COND_V(chunk_type != 0x004E4942, ERR_PARSE_ERROR); //BIN

	p_state->glb_data.resize(chunk_length);
	len = p_file->get_buffer(p_state->glb_data.ptrw(), chunk_length);
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
	xform.basis.set_column(Vector3::AXIS_X, Vector3(p_array[0], p_array[1], p_array[2]));
	xform.basis.set_column(Vector3::AXIS_Y, Vector3(p_array[4], p_array[5], p_array[6]));
	xform.basis.set_column(Vector3::AXIS_Z, Vector3(p_array[8], p_array[9], p_array[10]));
	xform.set_origin(Vector3(p_array[12], p_array[13], p_array[14]));

	return xform;
}

static Vector<real_t> _xform_to_array(const Transform3D p_transform) {
	Vector<real_t> array;
	array.resize(16);
	Vector3 axis_x = p_transform.get_basis().get_column(Vector3::AXIS_X);
	array.write[0] = axis_x.x;
	array.write[1] = axis_x.y;
	array.write[2] = axis_x.z;
	array.write[3] = 0.0f;
	Vector3 axis_y = p_transform.get_basis().get_column(Vector3::AXIS_Y);
	array.write[4] = axis_y.x;
	array.write[5] = axis_y.y;
	array.write[6] = axis_y.z;
	array.write[7] = 0.0f;
	Vector3 axis_z = p_transform.get_basis().get_column(Vector3::AXIS_Z);
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
		if (!gltf_node->get_name().is_empty()) {
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
		if (!gltf_node->visible) {
			Dictionary khr_node_visibility;
			extensions["KHR_node_visibility"] = khr_node_visibility;
			khr_node_visibility["visible"] = gltf_node->visible;
			if (!p_state->extensions_used.has("KHR_node_visibility")) {
				p_state->extensions_used.push_back("KHR_node_visibility");
				if (_visibility_mode == VISIBILITY_MODE_INCLUDE_REQUIRED) {
					p_state->extensions_required.push_back("KHR_node_visibility");
				}
			}
		}
		if (gltf_node->mesh != -1) {
			node["mesh"] = gltf_node->mesh;
		}
		if (gltf_node->skin != -1) {
			node["skin"] = gltf_node->skin;
		}
		if (gltf_node->skeleton != -1 && gltf_node->skin < 0) {
		}
		if (gltf_node->transform.basis.is_orthogonal()) {
			// An orthogonal transform is decomposable into TRS, so prefer that.
			const Vector3 position = gltf_node->get_position();
			if (!position.is_zero_approx()) {
				node["translation"] = _vec3_to_arr(position);
			}
			const Quaternion rotation = gltf_node->get_rotation();
			if (!rotation.is_equal_approx(Quaternion())) {
				node["rotation"] = _quaternion_to_array(rotation);
			}
			const Vector3 scale = gltf_node->get_scale();
			if (!scale.is_equal_approx(Vector3(1.0f, 1.0f, 1.0f))) {
				node["scale"] = _vec3_to_arr(scale);
			}
		} else {
			node["matrix"] = _xform_to_array(gltf_node->transform);
		}
		if (gltf_node->children.size()) {
			Array children;
			for (int j = 0; j < gltf_node->children.size(); j++) {
				children.push_back(gltf_node->children[j]);
			}
			node["children"] = children;
		}

		Node *scene_node = nullptr;
		if (i < (int)p_state->scene_nodes.size()) {
			scene_node = p_state->scene_nodes[i];
		}
		for (Ref<GLTFDocumentExtension> ext : document_extensions) {
			ERR_CONTINUE(ext.is_null());
			Error err = ext->export_node(p_state, gltf_node, node, scene_node);
			ERR_CONTINUE(err != OK);
		}

		if (extensions.is_empty()) {
			node.erase("extensions");
		}
		_attach_meta_to_extras(gltf_node, node);
		nodes.push_back(node);
	}
	if (!nodes.is_empty()) {
		p_state->json["nodes"] = nodes;
	}
	return OK;
}

String GLTFDocument::_gen_unique_name(Ref<GLTFState> p_state, const String &p_name) {
	return _gen_unique_name_static(p_state->unique_names, p_name);
}

String GLTFDocument::_sanitize_animation_name(const String &p_name) {
	String anim_name = p_name.validate_node_name();
	return AnimationLibrary::validate_library_name(anim_name);
}

String GLTFDocument::_gen_unique_animation_name(Ref<GLTFState> p_state, const String &p_name) {
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

String GLTFDocument::_sanitize_bone_name(const String &p_name) {
	String bone_name = p_name;
	bone_name = bone_name.replace_chars(":/", '_');
	return bone_name;
}

String GLTFDocument::_gen_unique_bone_name(Ref<GLTFState> p_state, const GLTFSkeletonIndex p_skel_i, const String &p_name) {
	String s_name = _sanitize_bone_name(p_name);
	if (s_name.is_empty()) {
		s_name = "bone";
	}
	String u_name;
	int index = 1;
	while (true) {
		u_name = s_name;

		if (index > 1) {
			u_name += "_" + itos(index);
		}
		if (!p_state->skeletons[p_skel_i]->unique_names.has(u_name)) {
			break;
		}
		index++;
	}

	p_state->skeletons.write[p_skel_i]->unique_names.insert(u_name);

	return u_name;
}

Error GLTFDocument::_parse_scenes(Ref<GLTFState> p_state) {
	p_state->unique_names.insert("Skeleton3D"); // Reserve skeleton name.
	if (!p_state->json.has("scenes")) {
		return OK; // No scenes.
	}
	const Array &scenes = p_state->json["scenes"];
	int loaded_scene = 0;
	if (p_state->json.has("scene")) {
		loaded_scene = p_state->json["scene"];
	} else {
		WARN_PRINT("The load-time scene is not defined in the glTF2 file. Picking the first scene.");
	}

	if (scenes.size()) {
		ERR_FAIL_COND_V(loaded_scene >= scenes.size(), ERR_FILE_CORRUPT);
		const Dictionary &scene_dict = scenes[loaded_scene];
		if (scene_dict.has("nodes")) {
			const Array &nodes = scene_dict["nodes"];
			for (const Variant &node : nodes) {
				p_state->root_nodes.push_back(node);
			}
		}
		// Determine what to use for the scene name.
		if (scene_dict.has("name") && !String(scene_dict["name"]).is_empty() && !((String)scene_dict["name"]).begins_with("Scene")) {
			p_state->scene_name = scene_dict["name"];
		} else if (p_state->scene_name.is_empty()) {
			p_state->scene_name = p_state->filename;
		}
		if (_naming_version == 0) {
			p_state->scene_name = _gen_unique_name(p_state, p_state->scene_name);
		}
	}

	return OK;
}

Error GLTFDocument::_parse_nodes(Ref<GLTFState> p_state) {
	if (!p_state->json.has("nodes")) {
		return OK; // No nodes to parse.
	}
	const Array &nodes = p_state->json["nodes"];
	for (int i = 0; i < nodes.size(); i++) {
		Ref<GLTFNode> node;
		node.instantiate();
		const Dictionary &n = nodes[i];

		if (n.has("name")) {
			node->set_original_name(n["name"]);
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
			node->transform = _arr_to_xform(n["matrix"]);
		} else {
			if (n.has("translation")) {
				node->set_position(_arr_to_vec3(n["translation"]));
			}
			if (n.has("rotation")) {
				node->set_rotation(_arr_to_quaternion(n["rotation"]));
			}
			if (n.has("scale")) {
				node->set_scale(_arr_to_vec3(n["scale"]));
			}
		}
		node->set_additional_data("GODOT_rest_transform", node->transform);

		if (n.has("extensions")) {
			Dictionary extensions = n["extensions"];
			if (extensions.has("KHR_lights_punctual")) {
				Dictionary lights_punctual = extensions["KHR_lights_punctual"];
				if (lights_punctual.has("light")) {
					GLTFLightIndex light = lights_punctual["light"];
					node->light = light;
				}
			}
			if (extensions.has("KHR_node_visibility")) {
				Dictionary khr_node_visibility = extensions["KHR_node_visibility"];
				if (khr_node_visibility.has("visible")) {
					node->visible = khr_node_visibility["visible"];
				}
			}
			for (Ref<GLTFDocumentExtension> ext : document_extensions) {
				ERR_CONTINUE(ext.is_null());
				Error err = ext->parse_node_extensions(p_state, node, extensions);
				ERR_CONTINUE_MSG(err != OK, "glTF: Encountered error " + itos(err) + " when parsing node extensions for node " + node->get_name() + " in file " + p_state->filename + ". Continuing.");
			}
		}

		if (n.has("extras")) {
			_attach_extras_to_meta(n["extras"], node);
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
	if (_naming_version < 2) {
		p_state->root_nodes.clear();
	}
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

		if (_naming_version < 2) {
			// This is incorrect, but required for compatibility with previous Godot versions.
			if (node->height == 0) {
				p_state->root_nodes.push_back(node_i);
			}
		}
	}
}

static Vector<uint8_t> _parse_base64_uri(const String &p_uri) {
	int start = p_uri.find_char(',');
	ERR_FAIL_COND_V(start == -1, Vector<uint8_t>());

	CharString substr = p_uri.substr(start + 1).ascii();

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

	if (p_state->buffers.is_empty()) {
		return OK;
	}
	Array buffers;
	if (!p_state->buffers.is_empty()) {
		Vector<uint8_t> buffer_data = p_state->buffers[0];
		Dictionary gltf_buffer;

		gltf_buffer["byteLength"] = buffer_data.size();
		buffers.push_back(gltf_buffer);
	}

	for (GLTFBufferIndex i = 1; i < p_state->buffers.size(); i++) {
		Vector<uint8_t> buffer_data = p_state->buffers[i];
		Dictionary gltf_buffer;
		String filename = p_path.get_basename().get_file() + itos(i) + ".bin";
		String path = p_path.get_base_dir() + "/" + filename;
		Error err;
		Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE, &err);
		if (file.is_null()) {
			return err;
		}
		if (buffer_data.is_empty()) {
			return OK;
		}
		file->create(FileAccess::ACCESS_RESOURCES);
		file->store_buffer(buffer_data.ptr(), buffer_data.size());
		gltf_buffer["uri"] = filename;
		gltf_buffer["byteLength"] = buffer_data.size();
		buffers.push_back(gltf_buffer);
	}
	p_state->json["buffers"] = buffers;

	return OK;
}

Error GLTFDocument::_encode_buffer_bins(Ref<GLTFState> p_state, const String &p_path) {
	print_verbose("glTF: Total buffers: " + itos(p_state->buffers.size()));

	if (p_state->buffers.is_empty()) {
		return OK;
	}
	Array buffers;

	for (GLTFBufferIndex i = 0; i < p_state->buffers.size(); i++) {
		Vector<uint8_t> buffer_data = p_state->buffers[i];
		Dictionary gltf_buffer;
		String filename = p_path.get_basename().get_file() + itos(i) + ".bin";
		String path = p_path.get_base_dir() + "/" + filename;
		Error err;
		Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE, &err);
		if (file.is_null()) {
			return err;
		}
		if (buffer_data.is_empty()) {
			return OK;
		}
		file->create(FileAccess::ACCESS_RESOURCES);
		file->store_buffer(buffer_data.ptr(), buffer_data.size());
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
		const Dictionary &buffer = buffers[i];
		Vector<uint8_t> buffer_data;
		if (buffer.has("uri")) {
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
				uri = uri.uri_file_decode();
				uri = p_base_path.path_join(uri).replace_char('\\', '/'); // Fix for Windows.
				ERR_FAIL_COND_V_MSG(!FileAccess::exists(uri), ERR_FILE_NOT_FOUND, "glTF: Binary file not found: " + uri);
				buffer_data = FileAccess::get_file_as_bytes(uri);
				ERR_FAIL_COND_V_MSG(buffer_data.is_empty(), ERR_PARSE_ERROR, "glTF: Couldn't load binary file as an array: " + uri);
			}

			ERR_FAIL_COND_V(!buffer.has("byteLength"), ERR_PARSE_ERROR);
			int64_t byteLength = buffer["byteLength"];
			ERR_FAIL_COND_V(byteLength < buffer_data.size(), ERR_PARSE_ERROR);
		} else if (i == 0 && p_state->glb_data.size()) {
			buffer_data = p_state->glb_data;
		} else {
			ERR_PRINT("glTF: Buffer " + itos(i) + " has no data and cannot be loaded.");
		}
		p_state->buffers.push_back(buffer_data);
	}

	print_verbose("glTF: Total buffers: " + itos(p_state->buffers.size()));

	return OK;
}

Error GLTFDocument::_encode_buffer_views(Ref<GLTFState> p_state) {
	Array buffers;
	for (GLTFBufferViewIndex i = 0; i < p_state->buffer_views.size(); i++) {
		const Ref<GLTFBufferView> buffer_view = p_state->buffer_views[i];
		buffers.push_back(buffer_view->to_dictionary());
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
		const Dictionary &dict = buffers[i];
		// Both "buffer" and "byteLength" are required by the spec.
		ERR_FAIL_COND_V(!dict.has("buffer"), ERR_PARSE_ERROR);
		ERR_FAIL_COND_V(!dict.has("byteLength"), ERR_PARSE_ERROR);
		Ref<GLTFBufferView> buffer_view = GLTFBufferView::from_dictionary(dict);
		p_state->buffer_views.push_back(buffer_view);
	}

	print_verbose("glTF: Total buffer views: " + itos(p_state->buffer_views.size()));

	return OK;
}

Error GLTFDocument::_encode_accessors(Ref<GLTFState> p_state) {
	Array accessors;
	for (GLTFAccessorIndex i = 0; i < p_state->accessors.size(); i++) {
		const Ref<GLTFAccessor> accessor = p_state->accessors[i];
		accessors.push_back(accessor->to_dictionary());
	}

	if (!accessors.size()) {
		return OK;
	}
	p_state->json["accessors"] = accessors;
	ERR_FAIL_COND_V(!p_state->json.has("accessors"), ERR_FILE_CORRUPT);
	print_verbose("glTF: Total accessors: " + itos(p_state->accessors.size()));

	return OK;
}

Error GLTFDocument::_parse_accessors(Ref<GLTFState> p_state) {
	if (!p_state->json.has("accessors")) {
		return OK;
	}
	const Array &accessors = p_state->json["accessors"];
	for (GLTFAccessorIndex i = 0; i < accessors.size(); i++) {
		const Dictionary &dict = accessors[i];
		// All of these fields are required by the spec.
		ERR_FAIL_COND_V(!dict.has("componentType"), ERR_PARSE_ERROR);
		ERR_FAIL_COND_V(!dict.has("count"), ERR_PARSE_ERROR);
		ERR_FAIL_COND_V(!dict.has("type"), ERR_PARSE_ERROR);
		Ref<GLTFAccessor> accessor = GLTFAccessor::from_dictionary(dict);
		p_state->accessors.push_back(accessor);
	}

	print_verbose("glTF: Total accessors: " + itos(p_state->accessors.size()));

	return OK;
}

template <typename T>
T GLTFDocument::_decode_unpack_indexed_data(const T &p_source, const PackedInt32Array &p_indices) {
	// Handle unpacking indexed data as if it was a regular array.
	// This isn't a feature of accessors, rather a feature of places using accessors like
	// indexed meshes, but GLTFDocument needs it in several places when reading accessors.
	T ret;
	const int64_t last_index = p_indices[p_indices.size() - 1];
	ERR_FAIL_COND_V(last_index >= p_source.size(), ret);
	ret.resize(p_indices.size());
	for (int64_t i = 0; i < p_indices.size(); i++) {
		const int64_t source_index = p_indices[i];
		ret.set(i, p_source[source_index]);
	}
	return ret;
}

PackedFloat32Array GLTFDocument::_decode_accessor_as_float32s(const Ref<GLTFState> p_gltf_state, GLTFAccessorIndex p_accessor_index, const PackedInt32Array &p_packed_vertex_ids) {
	ERR_FAIL_INDEX_V(p_accessor_index, p_gltf_state->accessors.size(), PackedFloat32Array());
	Ref<GLTFAccessor> accessor = p_gltf_state->accessors[p_accessor_index];
	PackedFloat32Array numbers = accessor->decode_as_float32s(p_gltf_state);
	if (p_packed_vertex_ids.is_empty()) {
		return numbers;
	}
	return _decode_unpack_indexed_data<PackedFloat32Array>(numbers, p_packed_vertex_ids);
}

PackedFloat64Array GLTFDocument::_decode_accessor_as_float64s(const Ref<GLTFState> p_gltf_state, GLTFAccessorIndex p_accessor_index, const PackedInt32Array &p_packed_vertex_ids) {
	ERR_FAIL_INDEX_V(p_accessor_index, p_gltf_state->accessors.size(), PackedFloat64Array());
	Ref<GLTFAccessor> accessor = p_gltf_state->accessors[p_accessor_index];
	PackedFloat64Array numbers = accessor->decode_as_float64s(p_gltf_state);
	if (p_packed_vertex_ids.is_empty()) {
		return numbers;
	}
	return _decode_unpack_indexed_data<PackedFloat64Array>(numbers, p_packed_vertex_ids);
}

PackedInt32Array GLTFDocument::_decode_accessor_as_int32s(const Ref<GLTFState> p_gltf_state, GLTFAccessorIndex p_accessor_index, const PackedInt32Array &p_packed_vertex_ids) {
	ERR_FAIL_INDEX_V(p_accessor_index, p_gltf_state->accessors.size(), PackedInt32Array());
	Ref<GLTFAccessor> accessor = p_gltf_state->accessors[p_accessor_index];
	PackedInt32Array numbers = accessor->decode_as_int32s(p_gltf_state);
	if (p_packed_vertex_ids.is_empty()) {
		return numbers;
	}
	return _decode_unpack_indexed_data<PackedInt32Array>(numbers, p_packed_vertex_ids);
}

PackedVector2Array GLTFDocument::_decode_accessor_as_vec2(const Ref<GLTFState> p_gltf_state, GLTFAccessorIndex p_accessor_index, const PackedInt32Array &p_packed_vertex_ids) {
	ERR_FAIL_INDEX_V(p_accessor_index, p_gltf_state->accessors.size(), PackedVector2Array());
	Ref<GLTFAccessor> accessor = p_gltf_state->accessors[p_accessor_index];
	PackedVector2Array vectors = accessor->decode_as_vector2s(p_gltf_state);
	if (p_packed_vertex_ids.is_empty()) {
		return vectors;
	}
	return _decode_unpack_indexed_data<PackedVector2Array>(vectors, p_packed_vertex_ids);
}

PackedVector3Array GLTFDocument::_decode_accessor_as_vec3(const Ref<GLTFState> p_gltf_state, GLTFAccessorIndex p_accessor_index, const PackedInt32Array &p_packed_vertex_ids) {
	ERR_FAIL_INDEX_V(p_accessor_index, p_gltf_state->accessors.size(), PackedVector3Array());
	Ref<GLTFAccessor> accessor = p_gltf_state->accessors[p_accessor_index];
	PackedVector3Array vectors = accessor->decode_as_vector3s(p_gltf_state);
	if (p_packed_vertex_ids.is_empty()) {
		return vectors;
	}
	return _decode_unpack_indexed_data<PackedVector3Array>(vectors, p_packed_vertex_ids);
}

PackedColorArray GLTFDocument::_decode_accessor_as_color(const Ref<GLTFState> p_gltf_state, GLTFAccessorIndex p_accessor_index, const PackedInt32Array &p_packed_vertex_ids) {
	ERR_FAIL_INDEX_V(p_accessor_index, p_gltf_state->accessors.size(), PackedColorArray());
	Ref<GLTFAccessor> accessor = p_gltf_state->accessors[p_accessor_index];
	PackedColorArray colors = accessor->decode_as_colors(p_gltf_state);
	if (p_packed_vertex_ids.is_empty()) {
		return colors;
	}
	return _decode_unpack_indexed_data<PackedColorArray>(colors, p_packed_vertex_ids);
}

Vector<Quaternion> GLTFDocument::_decode_accessor_as_quaternion(const Ref<GLTFState> p_gltf_state, GLTFAccessorIndex p_accessor_index) {
	ERR_FAIL_INDEX_V(p_accessor_index, p_gltf_state->accessors.size(), Vector<Quaternion>());
	Ref<GLTFAccessor> accessor = p_gltf_state->accessors[p_accessor_index];
	Vector<Quaternion> quaternions = accessor->decode_as_quaternions(p_gltf_state);
	return quaternions;
}

Array GLTFDocument::_decode_accessor_as_variants(const Ref<GLTFState> p_gltf_state, GLTFAccessorIndex p_accessor_index, Variant::Type p_variant_type) {
	ERR_FAIL_INDEX_V(p_accessor_index, p_gltf_state->accessors.size(), Array());
	Ref<GLTFAccessor> accessor = p_gltf_state->accessors[p_accessor_index];
	Array variants = accessor->decode_as_variants(p_gltf_state, p_variant_type);
	return variants;
}

Error GLTFDocument::_serialize_meshes(Ref<GLTFState> p_state) {
	Array meshes;
	for (GLTFMeshIndex gltf_mesh_i = 0; gltf_mesh_i < p_state->meshes.size(); gltf_mesh_i++) {
		print_verbose("glTF: Serializing mesh: " + itos(gltf_mesh_i));
		Ref<GLTFMesh> &gltf_mesh = p_state->meshes.write[gltf_mesh_i];
		Ref<ImporterMesh> import_mesh = gltf_mesh->get_mesh();
		if (import_mesh.is_null()) {
			continue;
		}
		const Array &instance_materials = gltf_mesh->get_instance_materials();
		Array primitives;
		Dictionary mesh_dict;
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
			uint64_t format = import_mesh->get_surface_format(surface_i);
			int32_t vertex_num = 0;
			Dictionary attributes;
			{
				Vector<Vector3> a = array[Mesh::ARRAY_VERTEX];
				ERR_FAIL_COND_V(a.is_empty(), ERR_INVALID_DATA);
				attributes["POSITION"] = GLTFAccessor::encode_new_accessor_from_vector3s(p_state, a, GLTFBufferView::TARGET_ARRAY_BUFFER);
				vertex_num = a.size();
			}
			{
				Vector<real_t> a = array[Mesh::ARRAY_TANGENT];
				if (a.size()) {
					const int64_t ret_size = a.size() / 4;
					Vector<Color> attribs;
					attribs.resize(ret_size);
					for (int64_t i = 0; i < ret_size; i++) {
						Color out;
						out.r = a[(i * 4) + 0];
						out.g = a[(i * 4) + 1];
						out.b = a[(i * 4) + 2];
						out.a = a[(i * 4) + 3];
						attribs.write[i] = out;
					}
					attributes["TANGENT"] = GLTFAccessor::encode_new_accessor_from_colors(p_state, attribs, GLTFBufferView::TARGET_ARRAY_BUFFER);
				}
			}
			{
				Vector<Vector3> a = array[Mesh::ARRAY_NORMAL];
				if (a.size()) {
					const int64_t ret_size = a.size();
					Vector<Vector3> attribs;
					attribs.resize(ret_size);
					for (int64_t i = 0; i < ret_size; i++) {
						attribs.write[i] = Vector3(a[i]).normalized();
					}
					attributes["NORMAL"] = GLTFAccessor::encode_new_accessor_from_vector3s(p_state, attribs, GLTFBufferView::TARGET_ARRAY_BUFFER);
				}
			}
			{
				Vector<Vector2> a = array[Mesh::ARRAY_TEX_UV];
				if (a.size()) {
					attributes["TEXCOORD_0"] = GLTFAccessor::encode_new_accessor_from_vector2s(p_state, a, GLTFBufferView::TARGET_ARRAY_BUFFER);
				}
			}
			{
				Vector<Vector2> a = array[Mesh::ARRAY_TEX_UV2];
				if (a.size()) {
					attributes["TEXCOORD_1"] = GLTFAccessor::encode_new_accessor_from_vector2s(p_state, a, GLTFBufferView::TARGET_ARRAY_BUFFER);
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
							attributes[gltf_texcoord_key] = GLTFAccessor::encode_new_accessor_from_vector2s(p_state, empty, GLTFBufferView::TARGET_ARRAY_BUFFER);
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
					attributes[gltf_texcoord_key] = GLTFAccessor::encode_new_accessor_from_vector2s(p_state, PackedVector2Array(first_channel), GLTFBufferView::TARGET_ARRAY_BUFFER);
					gltf_texcoord_key = vformat("TEXCOORD_%d", texcoord_i + 1);
					attributes[gltf_texcoord_key] = GLTFAccessor::encode_new_accessor_from_vector2s(p_state, PackedVector2Array(second_channel), GLTFBufferView::TARGET_ARRAY_BUFFER);
				}
			}
			{
				Vector<Color> a = array[Mesh::ARRAY_COLOR];
				if (a.size()) {
					attributes["COLOR_0"] = GLTFAccessor::encode_new_accessor_from_colors(p_state, a, GLTFBufferView::TARGET_ARRAY_BUFFER);
				}
			}
			HashMap<int, int> joint_i_to_bone_i;
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
					Vector<Vector4i> attribs;
					attribs.resize(ret_size);
					{
						for (int array_i = 0; array_i < attribs.size(); array_i++) {
							int32_t joint_0 = a[(array_i * JOINT_GROUP_SIZE) + 0];
							int32_t joint_1 = a[(array_i * JOINT_GROUP_SIZE) + 1];
							int32_t joint_2 = a[(array_i * JOINT_GROUP_SIZE) + 2];
							int32_t joint_3 = a[(array_i * JOINT_GROUP_SIZE) + 3];
							attribs.write[array_i] = Vector4i(joint_0, joint_1, joint_2, joint_3);
						}
					}
					attributes["JOINTS_0"] = GLTFAccessor::encode_new_accessor_from_vector4is(p_state, attribs, GLTFBufferView::TARGET_ARRAY_BUFFER);
				} else if ((a.size() / (JOINT_GROUP_SIZE * 2)) >= vertex_array.size()) {
					Vector<Vector4i> joints_0;
					joints_0.resize(vertex_num);
					Vector<Vector4i> joints_1;
					joints_1.resize(vertex_num);
					int32_t weights_8_count = JOINT_GROUP_SIZE * 2;
					for (int32_t vertex_i = 0; vertex_i < vertex_num; vertex_i++) {
						Vector4i joint_0;
						joint_0.x = a[vertex_i * weights_8_count + 0];
						joint_0.y = a[vertex_i * weights_8_count + 1];
						joint_0.z = a[vertex_i * weights_8_count + 2];
						joint_0.w = a[vertex_i * weights_8_count + 3];
						joints_0.write[vertex_i] = joint_0;
						Vector4i joint_1;
						joint_1.x = a[vertex_i * weights_8_count + 4];
						joint_1.y = a[vertex_i * weights_8_count + 5];
						joint_1.z = a[vertex_i * weights_8_count + 6];
						joint_1.w = a[vertex_i * weights_8_count + 7];
						joints_1.write[vertex_i] = joint_1;
					}
					attributes["JOINTS_0"] = GLTFAccessor::encode_new_accessor_from_vector4is(p_state, joints_0, GLTFBufferView::TARGET_ARRAY_BUFFER);
					attributes["JOINTS_1"] = GLTFAccessor::encode_new_accessor_from_vector4is(p_state, joints_1, GLTFBufferView::TARGET_ARRAY_BUFFER);
				}
			}
			{
				const PackedRealArray &a = array[Mesh::ARRAY_WEIGHTS];
				const Vector<Vector3> &vertex_array = array[Mesh::ARRAY_VERTEX];
				if ((a.size() / JOINT_GROUP_SIZE) == vertex_array.size()) {
					int32_t vertex_count = vertex_array.size();
					Vector<Vector4> attribs;
					attribs.resize(vertex_count);
					for (int i = 0; i < vertex_count; i++) {
						Vector4 weight_0(a[(i * JOINT_GROUP_SIZE) + 0], a[(i * JOINT_GROUP_SIZE) + 1], a[(i * JOINT_GROUP_SIZE) + 2], a[(i * JOINT_GROUP_SIZE) + 3]);
						float divisor = weight_0.x + weight_0.y + weight_0.z + weight_0.w;
						if (Math::is_zero_approx(divisor) || !Math::is_finite(divisor)) {
							attribs.write[i] = Vector4(1, 0, 0, 0);
						} else {
							attribs.write[i] = weight_0 / divisor;
						}
					}
					attributes["WEIGHTS_0"] = GLTFAccessor::encode_new_accessor_from_vector4s(p_state, attribs, GLTFBufferView::TARGET_ARRAY_BUFFER);
				} else if ((a.size() / (JOINT_GROUP_SIZE * 2)) >= vertex_array.size()) {
					Vector<Vector4> weights_0;
					weights_0.resize(vertex_num);
					Vector<Vector4> weights_1;
					weights_1.resize(vertex_num);
					int32_t weights_8_count = JOINT_GROUP_SIZE * 2;
					for (int32_t vertex_i = 0; vertex_i < vertex_num; vertex_i++) {
						Vector4 weight_0;
						weight_0.x = a[vertex_i * weights_8_count + 0];
						weight_0.y = a[vertex_i * weights_8_count + 1];
						weight_0.z = a[vertex_i * weights_8_count + 2];
						weight_0.w = a[vertex_i * weights_8_count + 3];
						Vector4 weight_1;
						weight_1.x = a[vertex_i * weights_8_count + 4];
						weight_1.y = a[vertex_i * weights_8_count + 5];
						weight_1.z = a[vertex_i * weights_8_count + 6];
						weight_1.w = a[vertex_i * weights_8_count + 7];
						float divisor = weight_0.x + weight_0.y + weight_0.z + weight_0.w + weight_1.x + weight_1.y + weight_1.z + weight_1.w;
						if (Math::is_zero_approx(divisor) || !Math::is_finite(divisor)) {
							weights_0.write[vertex_i] = Vector4(1, 0, 0, 0);
							weights_1.write[vertex_i] = Vector4(0, 0, 0, 0);
						} else {
							weights_0.write[vertex_i] = weight_0 / divisor;
							weights_1.write[vertex_i] = weight_1 / divisor;
						}
					}
					attributes["WEIGHTS_0"] = GLTFAccessor::encode_new_accessor_from_vector4s(p_state, weights_0, GLTFBufferView::TARGET_ARRAY_BUFFER);
					attributes["WEIGHTS_1"] = GLTFAccessor::encode_new_accessor_from_vector4s(p_state, weights_1, GLTFBufferView::TARGET_ARRAY_BUFFER);
				}
			}
			{
				Vector<int32_t> mesh_indices = array[Mesh::ARRAY_INDEX];
				if (mesh_indices.size()) {
					if (primitive_type == Mesh::PRIMITIVE_TRIANGLES) {
						// Swap around indices, convert ccw to cw for front face.
						const int is = mesh_indices.size();
						for (int k = 0; k < is; k += 3) {
							SWAP(mesh_indices.write[k + 0], mesh_indices.write[k + 2]);
						}
					}
					primitive["indices"] = GLTFAccessor::encode_new_accessor_from_int32s(p_state, mesh_indices, GLTFBufferView::TARGET_ELEMENT_ARRAY_BUFFER);
				} else {
					if (primitive_type == Mesh::PRIMITIVE_TRIANGLES) {
						// Generate indices because they need to be swapped for CW/CCW.
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
						primitive["indices"] = GLTFAccessor::encode_new_accessor_from_int32s(p_state, generated_indices, GLTFBufferView::TARGET_ELEMENT_ARRAY_BUFFER);
					}
				}
			}

			primitive["attributes"] = attributes;

			// Blend shapes
			print_verbose("glTF: Mesh has targets");
			if (import_mesh->get_blend_shape_count()) {
				ArrayMesh::BlendShapeMode shape_mode = import_mesh->get_blend_shape_mode();
				const float normal_tangent_sparse_rounding = 0.001;
				for (int morph_i = 0; morph_i < import_mesh->get_blend_shape_count(); morph_i++) {
					Array array_morph = import_mesh->get_surface_blend_shape_arrays(surface_i, morph_i);
					Dictionary t;
					Vector<Vector3> varr = array_morph[Mesh::ARRAY_VERTEX];
					Vector<Vector3> src_varr = array[Mesh::ARRAY_VERTEX];
					Array mesh_arrays = import_mesh->get_surface_arrays(surface_i);
					if (varr.size() && varr.size() == src_varr.size()) {
						if (shape_mode == ArrayMesh::BlendShapeMode::BLEND_SHAPE_MODE_NORMALIZED) {
							const int max_idx = src_varr.size();
							for (int blend_i = 0; blend_i < max_idx; blend_i++) {
								varr.write[blend_i] = varr[blend_i] - src_varr[blend_i];
							}
						}
						const GLTFAccessorIndex position_accessor = attributes["POSITION"];
						if (position_accessor != -1) {
							const GLTFAccessorIndex new_accessor = GLTFAccessor::encode_new_sparse_accessor_from_vec3s(p_state, varr, Vector<Vector3>(), 1.0, GLTFBufferView::TARGET_ARRAY_BUFFER);
							if (new_accessor != -1) {
								t["POSITION"] = new_accessor;
							}
						}
					}

					Vector<Vector3> narr = array_morph[Mesh::ARRAY_NORMAL];
					Vector<Vector3> src_narr = array[Mesh::ARRAY_NORMAL];
					if (narr.size() && narr.size() == src_narr.size()) {
						if (shape_mode == ArrayMesh::BlendShapeMode::BLEND_SHAPE_MODE_NORMALIZED) {
							const int max_idx = src_narr.size();
							for (int blend_i = 0; blend_i < max_idx; blend_i++) {
								narr.write[blend_i] = narr[blend_i] - src_narr[blend_i];
							}
						}
						const GLTFAccessorIndex normal_accessor = attributes["NORMAL"];
						if (normal_accessor != -1) {
							const GLTFAccessorIndex new_accessor = GLTFAccessor::encode_new_sparse_accessor_from_vec3s(p_state, narr, Vector<Vector3>(), normal_tangent_sparse_rounding, GLTFBufferView::TARGET_ARRAY_BUFFER);
							if (new_accessor != -1) {
								t["NORMAL"] = new_accessor;
							}
						}
					}
					Vector<real_t> tarr = array_morph[Mesh::ARRAY_TANGENT];
					Vector<real_t> src_tarr = array[Mesh::ARRAY_TANGENT];
					if (tarr.size() && tarr.size() == src_tarr.size()) {
						const int ret_size = tarr.size() / 4;
						Vector<Vector3> attribs;
						attribs.resize(ret_size);
						for (int i = 0; i < ret_size; i++) {
							Vector3 vec3;
							vec3.x = tarr[(i * 4) + 0] - src_tarr[(i * 4) + 0];
							vec3.y = tarr[(i * 4) + 1] - src_tarr[(i * 4) + 1];
							vec3.z = tarr[(i * 4) + 2] - src_tarr[(i * 4) + 2];
							attribs.write[i] = vec3;
						}
						const GLTFAccessorIndex tangent_accessor = attributes["TANGENT"];
						if (tangent_accessor != -1) {
							const GLTFAccessorIndex new_accessor = GLTFAccessor::encode_new_sparse_accessor_from_vec3s(p_state, attribs, Vector<Vector3>(), normal_tangent_sparse_rounding, GLTFBufferView::TARGET_ARRAY_BUFFER);
							if (new_accessor != -1) {
								t["TANGENT"] = new_accessor;
							}
						}
					}
					targets.push_back(t);
				}
			}

			Variant v;
			if (surface_i < instance_materials.size()) {
				v = instance_materials.get(surface_i);
			}
			Ref<Material> mat = v;
			if (mat.is_null()) {
				mat = import_mesh->get_surface_material(surface_i);
			}
			if (mat.is_valid()) {
				HashMap<Ref<Material>, GLTFMaterialIndex>::Iterator material_cache_i = p_state->material_cache.find(mat);
				if (material_cache_i && material_cache_i->value != -1) {
					primitive["material"] = material_cache_i->value;
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

		if (!target_names.is_empty()) {
			Dictionary e;
			e["targetNames"] = target_names;
			mesh_dict["extras"] = e;
		}
		_attach_meta_to_extras(import_mesh, mesh_dict);

		weights.resize(target_names.size());
		for (int name_i = 0; name_i < target_names.size(); name_i++) {
			real_t weight = 0.0;
			if (name_i < gltf_mesh->get_blend_weights().size()) {
				weight = gltf_mesh->get_blend_weights()[name_i];
			}
			weights[name_i] = weight;
		}
		if (weights.size()) {
			mesh_dict["weights"] = weights;
		}

		ERR_FAIL_COND_V(target_names.size() != weights.size(), FAILED);

		mesh_dict["primitives"] = primitives;

		if (!gltf_mesh->get_name().is_empty()) {
			mesh_dict["name"] = gltf_mesh->get_name();
		}

		meshes.push_back(mesh_dict);
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

	Array meshes = p_state->json["meshes"];
	for (GLTFMeshIndex i = 0; i < meshes.size(); i++) {
		print_verbose("glTF: Parsing mesh: " + itos(i));
		Dictionary mesh_dict = meshes[i];

		Ref<GLTFMesh> mesh;
		mesh.instantiate();
		bool has_vertex_color = false;

		ERR_FAIL_COND_V(!mesh_dict.has("primitives"), ERR_PARSE_ERROR);

		Array primitives = mesh_dict["primitives"];
		const Dictionary &extras = mesh_dict.has("extras") ? (Dictionary)mesh_dict["extras"] : Dictionary();
		_attach_extras_to_meta(extras, mesh);
		Ref<ImporterMesh> import_mesh;
		import_mesh.instantiate();
		String mesh_name = "mesh";
		if (mesh_dict.has("name") && !String(mesh_dict["name"]).is_empty()) {
			mesh_name = mesh_dict["name"];
			mesh->set_original_name(mesh_name);
		}
		import_mesh->set_name(_gen_unique_name(p_state, vformat("%s_%s", p_state->scene_name, mesh_name)));
		mesh->set_name(import_mesh->get_name());
		TypedArray<Material> instance_materials;

		for (int j = 0; j < primitives.size(); j++) {
			uint64_t flags = RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
			Dictionary mesh_prim = primitives[j];

			Array array;
			array.resize(Mesh::ARRAY_MAX);

			ERR_FAIL_COND_V(!mesh_prim.has("attributes"), ERR_PARSE_ERROR);

			Dictionary a = mesh_prim["attributes"];

			Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;
			if (mesh_prim.has("mode")) {
				const int mode = mesh_prim["mode"];
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
					// TODO: Line loop and triangle fan are not supported and need to be converted to lines and triangles.
				};

				primitive = primitives2[mode];
			}

			int32_t orig_vertex_num = 0;
			ERR_FAIL_COND_V(!a.has("POSITION"), ERR_PARSE_ERROR);
			if (a.has("POSITION")) {
				PackedVector3Array vertices = _decode_accessor_as_vec3(p_state, a["POSITION"]);
				array[Mesh::ARRAY_VERTEX] = vertices;
				orig_vertex_num = vertices.size();
			}
			int32_t vertex_num = orig_vertex_num;

			Vector<int> indices;
			Vector<int> indices_mapping;
			Vector<int> indices_rev_mapping;
			Vector<int> indices_vec4_mapping;
			if (mesh_prim.has("indices")) {
				indices = _decode_accessor_as_int32s(p_state, mesh_prim["indices"]);
				const int index_count = indices.size();

				if (primitive == Mesh::PRIMITIVE_TRIANGLES) {
					ERR_FAIL_COND_V_MSG(index_count % 3 != 0, ERR_PARSE_ERROR, "glTF import: Mesh " + itos(i) + " surface " + itos(j) + " in file " + p_state->filename + " is invalid. Indexed triangle meshes MUST have an index array with a size that is a multiple of 3, but got " + itos(index_count) + " indices.");
					// Swap around indices, convert ccw to cw for front face.

					int *w = indices.ptrw();
					for (int k = 0; k < index_count; k += 3) {
						SWAP(w[k + 1], w[k + 2]);
					}
				}

				const int *indices_w = indices.ptrw();
				Vector<bool> used_indices;
				used_indices.resize_initialized(orig_vertex_num);
				bool *used_w = used_indices.ptrw();
				for (int idx_i = 0; idx_i < index_count; idx_i++) {
					ERR_FAIL_INDEX_V(indices_w[idx_i], orig_vertex_num, ERR_INVALID_DATA);
					used_w[indices_w[idx_i]] = true;
				}
				indices_rev_mapping.resize_initialized(orig_vertex_num);
				int *rev_w = indices_rev_mapping.ptrw();
				vertex_num = 0;
				for (int vert_i = 0; vert_i < orig_vertex_num; vert_i++) {
					if (used_w[vert_i]) {
						rev_w[vert_i] = indices_mapping.size();
						indices_mapping.push_back(vert_i);
						indices_vec4_mapping.push_back(vert_i * 4 + 0);
						indices_vec4_mapping.push_back(vert_i * 4 + 1);
						indices_vec4_mapping.push_back(vert_i * 4 + 2);
						indices_vec4_mapping.push_back(vert_i * 4 + 3);
						vertex_num++;
					}
				}
			}
			ERR_FAIL_COND_V(vertex_num <= 0, ERR_INVALID_DECLARATION);

			if (a.has("POSITION")) {
				PackedVector3Array vertices = _decode_accessor_as_vec3(p_state, a["POSITION"], indices_mapping);
				array[Mesh::ARRAY_VERTEX] = vertices;
			}
			if (a.has("NORMAL")) {
				array[Mesh::ARRAY_NORMAL] = _decode_accessor_as_vec3(p_state, a["NORMAL"], indices_mapping);
			}
			if (a.has("TANGENT")) {
				array[Mesh::ARRAY_TANGENT] = _decode_accessor_as_float32s(p_state, a["TANGENT"], indices_vec4_mapping);
			}
			if (a.has("TEXCOORD_0")) {
				array[Mesh::ARRAY_TEX_UV] = _decode_accessor_as_vec2(p_state, a["TEXCOORD_0"], indices_mapping);
			}
			if (a.has("TEXCOORD_1")) {
				array[Mesh::ARRAY_TEX_UV2] = _decode_accessor_as_vec2(p_state, a["TEXCOORD_1"], indices_mapping);
			}
			for (int custom_i = 0; custom_i < 3; custom_i++) {
				Vector<float> cur_custom;
				Vector<Vector2> texcoord_first;
				Vector<Vector2> texcoord_second;

				int texcoord_i = 2 + 2 * custom_i;
				String gltf_texcoord_key = vformat("TEXCOORD_%d", texcoord_i);
				int num_channels = 0;
				if (a.has(gltf_texcoord_key)) {
					texcoord_first = _decode_accessor_as_vec2(p_state, a[gltf_texcoord_key], indices_mapping);
					num_channels = 2;
				}
				gltf_texcoord_key = vformat("TEXCOORD_%d", texcoord_i + 1);
				if (a.has(gltf_texcoord_key)) {
					texcoord_second = _decode_accessor_as_vec2(p_state, a[gltf_texcoord_key], indices_mapping);
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
				array[Mesh::ARRAY_COLOR] = _decode_accessor_as_color(p_state, a["COLOR_0"], indices_mapping);
				has_vertex_color = true;
			}
			if (a.has("JOINTS_0") && !a.has("JOINTS_1")) {
				PackedInt32Array joints_0 = _decode_accessor_as_int32s(p_state, a["JOINTS_0"], indices_vec4_mapping);
				ERR_FAIL_COND_V(joints_0.size() != 4 * vertex_num, ERR_INVALID_DATA);
				array[Mesh::ARRAY_BONES] = joints_0;
			} else if (a.has("JOINTS_0") && a.has("JOINTS_1")) {
				PackedInt32Array joints_0 = _decode_accessor_as_int32s(p_state, a["JOINTS_0"], indices_vec4_mapping);
				PackedInt32Array joints_1 = _decode_accessor_as_int32s(p_state, a["JOINTS_1"], indices_vec4_mapping);
				ERR_FAIL_COND_V(joints_0.size() != joints_1.size(), ERR_INVALID_DATA);
				ERR_FAIL_COND_V(joints_0.size() != 4 * vertex_num, ERR_INVALID_DATA);
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
			// glTF stores weights as a VEC4 array or multiple VEC4 arrays, but Godot's
			// ArrayMesh uses a flat array of either 4 or 8 floats per vertex.
			// Therefore, decode up to two glTF VEC4 arrays as float arrays.
			if (a.has("WEIGHTS_0") && !a.has("WEIGHTS_1")) {
				Vector<float> weights = _decode_accessor_as_float32s(p_state, a["WEIGHTS_0"], indices_vec4_mapping);
				ERR_FAIL_COND_V(weights.size() != 4 * vertex_num, ERR_INVALID_DATA);
				{ // glTF does not seem to normalize the weights for some reason.
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
				Vector<float> weights_0 = _decode_accessor_as_float32s(p_state, a["WEIGHTS_0"], indices_vec4_mapping);
				Vector<float> weights_1 = _decode_accessor_as_float32s(p_state, a["WEIGHTS_1"], indices_vec4_mapping);
				Vector<float> weights;
				ERR_FAIL_COND_V(weights_0.size() != weights_1.size(), ERR_INVALID_DATA);
				ERR_FAIL_COND_V(weights_0.size() != 4 * vertex_num, ERR_INVALID_DATA);
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
				{ // glTF does not seem to normalize the weights for some reason.
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
				flags |= Mesh::ARRAY_FLAG_USE_8_BONE_WEIGHTS;
			}

			if (!indices.is_empty()) {
				int *w = indices.ptrw();
				const int is = indices.size();
				for (int ind_i = 0; ind_i < is; ind_i++) {
					w[ind_i] = indices_rev_mapping[indices[ind_i]];
				}
				array[Mesh::ARRAY_INDEX] = indices;

			} else if (primitive == Mesh::PRIMITIVE_TRIANGLES) {
				// Generate indices because they need to be swapped for CW/CCW.
				const Vector<Vector3> &vertices = array[Mesh::ARRAY_VERTEX];
				ERR_FAIL_COND_V(vertices.is_empty(), ERR_PARSE_ERROR);
				const int vertex_count = vertices.size();
				ERR_FAIL_COND_V_MSG(vertex_count % 3 != 0, ERR_PARSE_ERROR, "glTF import: Mesh " + itos(i) + " surface " + itos(j) + " in file " + p_state->filename + " is invalid. Non-indexed triangle meshes MUST have a vertex array with a size that is a multiple of 3, but got " + itos(vertex_count) + " vertices.");
				indices.resize(vertex_count);
				{
					int *w = indices.ptrw();
					for (int k = 0; k < vertex_count; k += 3) {
						w[k] = k;
						w[k + 1] = k + 2;
						w[k + 2] = k + 1;
					}
				}
				array[Mesh::ARRAY_INDEX] = indices;
			}

			bool generate_tangents = p_state->force_generate_tangents && (primitive == Mesh::PRIMITIVE_TRIANGLES && !a.has("TANGENT") && a.has("NORMAL"));

			if (generate_tangents && !a.has("TEXCOORD_0")) {
				// If we don't have UVs we provide a dummy tangent array.
				Vector<float> tangents;
				tangents.resize(vertex_num * 4);
				float *tangentsw = tangents.ptrw();

				Vector<Vector3> normals = array[Mesh::ARRAY_NORMAL];
				for (int k = 0; k < vertex_num; k++) {
					Vector3 tan = Vector3(normals[k].z, -normals[k].x, normals[k].y).cross(normals[k].normalized()).normalized();
					tangentsw[k * 4 + 0] = tan.x;
					tangentsw[k * 4 + 1] = tan.y;
					tangentsw[k * 4 + 2] = tan.z;
					tangentsw[k * 4 + 3] = 1.0;
				}
				array[Mesh::ARRAY_TANGENT] = tangents;
			}

			// Disable compression if all z equals 0 (the mesh is 2D).
			const Vector<Vector3> &vertices = array[Mesh::ARRAY_VERTEX];
			bool is_mesh_2d = true;
			for (int k = 0; k < vertices.size(); k++) {
				if (!Math::is_zero_approx(vertices[k].z)) {
					is_mesh_2d = false;
					break;
				}
			}

			if (p_state->force_disable_compression || is_mesh_2d || !a.has("POSITION") || !a.has("NORMAL") || mesh_prim.has("targets") || (a.has("JOINTS_0") || a.has("JOINTS_1"))) {
				flags &= ~RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
			}

			Ref<SurfaceTool> mesh_surface_tool;
			mesh_surface_tool.instantiate();
			mesh_surface_tool->create_from_triangle_arrays(array);
			if (a.has("JOINTS_0") && a.has("JOINTS_1")) {
				mesh_surface_tool->set_skin_weight_count(SurfaceTool::SKIN_8_WEIGHTS);
			}
			mesh_surface_tool->index();
			if (generate_tangents && a.has("TEXCOORD_0")) {
				//must generate mikktspace tangents.. ergh..
				mesh_surface_tool->generate_tangents();
			}
			array = mesh_surface_tool->commit_to_arrays();

			if ((flags & RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES) && a.has("NORMAL") && (a.has("TANGENT") || generate_tangents)) {
				// Compression is enabled, so let's validate that the normals and tangents are correct.
				Vector<Vector3> normals = array[Mesh::ARRAY_NORMAL];
				Vector<float> tangents = array[Mesh::ARRAY_TANGENT];
				if (unlikely(tangents.size() < normals.size() * 4)) {
					ERR_PRINT("glTF import: Mesh " + itos(i) + " has invalid tangents.");
					flags &= ~RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
				} else {
					for (int vert = 0; vert < normals.size(); vert++) {
						Vector3 tan = Vector3(tangents[vert * 4 + 0], tangents[vert * 4 + 1], tangents[vert * 4 + 2]);
						if (std::abs(tan.dot(normals[vert])) > 0.0001) {
							// Tangent is not perpendicular to the normal, so we can't use compression.
							flags &= ~RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
						}
					}
				}
			}

			Array morphs;
			// Blend shapes
			if (mesh_prim.has("targets")) {
				print_verbose("glTF: Mesh has targets");
				const Array &targets = mesh_prim["targets"];

				import_mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);

				if (j == 0) {
					const Array &target_names = extras.has("targetNames") ? (Array)extras["targetNames"] : Array();
					for (int k = 0; k < targets.size(); k++) {
						String bs_name;
						if (k < target_names.size() && ((String)target_names[k]).size() != 0) {
							bs_name = (String)target_names[k];
						} else {
							bs_name = String("morph_") + itos(k);
						}
						import_mesh->add_blend_shape(bs_name);
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
						Vector<Vector3> varr = _decode_accessor_as_vec3(p_state, t["POSITION"], indices_mapping);
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
						Vector<Vector3> narr = _decode_accessor_as_vec3(p_state, t["NORMAL"], indices_mapping);
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
						const Vector<Vector3> tangents_v3 = _decode_accessor_as_vec3(p_state, t["TANGENT"], indices_mapping);
						const Vector<float> src_tangents = array[Mesh::ARRAY_TANGENT];
						ERR_FAIL_COND_V(src_tangents.is_empty(), ERR_PARSE_ERROR);

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

					// Enforce blend shape mask array format
					for (int l = 0; l < Mesh::ARRAY_MAX; l++) {
						if (!(Mesh::ARRAY_FORMAT_BLEND_SHAPE_MASK & (1ULL << l))) {
							array_copy[l] = Variant();
						}
					}

					morphs.push_back(array_copy);
				}
			}

			Ref<Material> mat;
			String mat_name;
			if (!p_state->discard_meshes_and_materials) {
				if (mesh_prim.has("material")) {
					const int material = mesh_prim["material"];
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
				instance_materials.append(mat);
				mat_name = mat->get_name();
			}
			import_mesh->add_surface(primitive, array, morphs,
					Dictionary(), mat, mat_name, flags);
		}

		Vector<float> blend_weights;
		blend_weights.resize(import_mesh->get_blend_shape_count());
		for (int32_t weight_i = 0; weight_i < blend_weights.size(); weight_i++) {
			blend_weights.write[weight_i] = 0.0f;
		}

		if (mesh_dict.has("weights")) {
			const Array &weights = mesh_dict["weights"];
			for (int j = 0; j < weights.size(); j++) {
				if (j >= blend_weights.size()) {
					break;
				}
				blend_weights.write[j] = weights[j];
			}
		}
		mesh->set_blend_weights(blend_weights);
		mesh->set_instance_materials(instance_materials);
		mesh->set_mesh(import_mesh);

		p_state->meshes.push_back(mesh);
	}

	print_verbose("glTF: Total meshes: " + itos(p_state->meshes.size()));

	return OK;
}

void GLTFDocument::set_naming_version(int p_version) {
	_naming_version = p_version;
}

int GLTFDocument::get_naming_version() const {
	return _naming_version;
}

void GLTFDocument::set_image_format(const String &p_image_format) {
	_image_format = p_image_format;
}

String GLTFDocument::get_image_format() const {
	return _image_format;
}

void GLTFDocument::set_lossy_quality(float p_lossy_quality) {
	_lossy_quality = p_lossy_quality;
}

float GLTFDocument::get_lossy_quality() const {
	return _lossy_quality;
}

void GLTFDocument::set_fallback_image_format(const String &p_fallback_image_format) {
	_fallback_image_format = p_fallback_image_format;
}

String GLTFDocument::get_fallback_image_format() const {
	return _fallback_image_format;
}

void GLTFDocument::set_fallback_image_quality(float p_fallback_image_quality) {
	_fallback_image_quality = p_fallback_image_quality;
}

float GLTFDocument::get_fallback_image_quality() const {
	return _fallback_image_quality;
}

Error GLTFDocument::_serialize_images(Ref<GLTFState> p_state) {
	Array images;
	// Check if any extension wants to be the image saver.
	_image_save_extension = Ref<GLTFDocumentExtension>();
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		Vector<String> image_formats = ext->get_saveable_image_formats();
		if (image_formats.has(_image_format)) {
			_image_save_extension = ext;
			break;
		}
	}
	// Serialize every image in the state's images array.
	for (int i = 0; i < p_state->images.size(); i++) {
		Dictionary image_dict;
		if (p_state->images[i].is_null()) {
			ERR_PRINT("glTF export: Image Texture2D is null.");
		} else {
			Ref<Image> image = p_state->images[i]->get_image();
			if (image.is_null()) {
				ERR_PRINT("glTF export: Image's image is null.");
			} else {
				String image_name = p_state->images[i]->get_name();
				if (image_name.is_empty()) {
					image_name = itos(i).pad_zeros(3);
				}
				image_name = _gen_unique_name(p_state, image_name);
				image->set_name(image_name);
				image_dict = _serialize_image(p_state, image, _image_format, _lossy_quality, _image_save_extension);
			}
		}
		images.push_back(image_dict);
	}

	print_verbose("Total images: " + itos(p_state->images.size()));

	if (!images.size()) {
		return OK;
	}
	p_state->json["images"] = images;

	return OK;
}

static inline Ref<Image> _duplicate_and_decompress_image(const Ref<Image> &p_image) {
	Ref<Image> img = p_image->duplicate();
	if (img->is_compressed()) {
		img->decompress();
	}
	return img;
}

Dictionary GLTFDocument::_serialize_image(Ref<GLTFState> p_state, Ref<Image> p_image, const String &p_image_format, float p_lossy_quality, Ref<GLTFDocumentExtension> p_image_save_extension) {
	Dictionary image_dict;
	if (p_image->is_compressed()) {
		p_image = _duplicate_and_decompress_image(p_image);
		ERR_FAIL_COND_V_MSG(p_image->is_compressed(), image_dict, "glTF: Image was compressed, but could not be decompressed.");
	}

	if (!p_image->get_name().is_empty()) {
		image_dict["name"] = p_image->get_name();
	}

	if (p_state->filename.to_lower().ends_with("gltf")) {
		String relative_texture_dir = "textures";
		String full_texture_dir = p_state->base_path.path_join(relative_texture_dir);
		Ref<DirAccess> da = DirAccess::open(p_state->base_path);
		ERR_FAIL_COND_V(da.is_null(), image_dict);

		if (!da->dir_exists(full_texture_dir)) {
			da->make_dir(full_texture_dir);
		}
		String image_file_name = p_image->get_name();
		if (p_image_save_extension.is_valid()) {
			image_file_name = image_file_name + p_image_save_extension->get_image_file_extension();
			Error err = p_image_save_extension->save_image_at_path(p_state, p_image, full_texture_dir.path_join(image_file_name), p_image_format, p_lossy_quality);
			ERR_FAIL_COND_V_MSG(err != OK, image_dict, "glTF: Failed to save image in '" + p_image_format + "' format as a separate file, error " + itos(err) + ".");
		} else if (p_image_format == "PNG") {
			image_file_name = image_file_name + ".png";
			p_image->save_png(full_texture_dir.path_join(image_file_name));
		} else if (p_image_format == "JPEG") {
			image_file_name = image_file_name + ".jpg";
			p_image->save_jpg(full_texture_dir.path_join(image_file_name), p_lossy_quality);
		} else {
			ERR_FAIL_V_MSG(image_dict, "glTF: Unknown image format '" + p_image_format + "'.");
		}
		image_dict["uri"] = relative_texture_dir.path_join(image_file_name).uri_encode();
	} else {
		GLTFBufferViewIndex bvi;

		Ref<GLTFBufferView> bv;
		bv.instantiate();

		const GLTFBufferIndex bi = 0;
		bv->buffer = bi;
		ERR_FAIL_INDEX_V(bi, p_state->buffers.size(), image_dict);
		bv->byte_offset = p_state->buffers[bi].size();

		Vector<uint8_t> buffer;
		// Save in various image formats. Note that if the format is "None",
		// the state's images will be empty, so this code will not be reached.
		if (_image_save_extension.is_valid()) {
			buffer = _image_save_extension->serialize_image_to_bytes(p_state, p_image, image_dict, p_image_format, p_lossy_quality);
		} else if (p_image_format == "PNG") {
			buffer = p_image->save_png_to_buffer();
			image_dict["mimeType"] = "image/png";
		} else if (p_image_format == "JPEG") {
			buffer = p_image->save_jpg_to_buffer(p_lossy_quality);
			image_dict["mimeType"] = "image/jpeg";
		} else {
			ERR_FAIL_V_MSG(image_dict, "glTF: Unknown image format '" + p_image_format + "'.");
		}
		ERR_FAIL_COND_V_MSG(buffer.is_empty(), image_dict, "glTF: Failed to save image in '" + p_image_format + "' format.");

		bv->byte_length = buffer.size();
		p_state->buffers.write[bi].resize(p_state->buffers[bi].size() + bv->byte_length);
		memcpy(&p_state->buffers.write[bi].write[bv->byte_offset], buffer.ptr(), buffer.size());
		ERR_FAIL_COND_V(bv->byte_offset + bv->byte_length > p_state->buffers[bi].size(), image_dict);

		p_state->buffer_views.push_back(bv);
		bvi = p_state->buffer_views.size() - 1;
		image_dict["bufferView"] = bvi;
	}
	return image_dict;
}

Ref<Image> GLTFDocument::_parse_image_bytes_into_image(Ref<GLTFState> p_state, const Vector<uint8_t> &p_bytes, const String &p_mime_type, int p_index, String &r_file_extension) {
	Ref<Image> r_image;
	r_image.instantiate();
	// Check if any GLTFDocumentExtensions want to import this data as an image.
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		Error err = ext->parse_image_data(p_state, p_bytes, p_mime_type, r_image);
		ERR_CONTINUE_MSG(err != OK, "glTF: Encountered error " + itos(err) + " when parsing image " + itos(p_index) + " in file " + p_state->filename + ". Continuing.");
		if (!r_image->is_empty()) {
			r_file_extension = ext->get_image_file_extension();
			return r_image;
		}
	}
	// If no extension wanted to import this data as an image, try to load a PNG or JPEG.
	// First we honor the mime types if they were defined.
	if (p_mime_type == "image/png") { // Load buffer as PNG.
		r_image->load_png_from_buffer(p_bytes);
		r_file_extension = ".png";
	} else if (p_mime_type == "image/jpeg") { // Loader buffer as JPEG.
		r_image->load_jpg_from_buffer(p_bytes);
		r_file_extension = ".jpg";
	}
	// If we didn't pass the above tests, we attempt loading as PNG and then JPEG directly.
	// This covers URIs with base64-encoded data with application/* type but
	// no optional mimeType property, or bufferViews with a bogus mimeType
	// (e.g. `image/jpeg` but the data is actually PNG).
	// That's not *exactly* what the spec mandates but this lets us be
	// lenient with bogus glb files which do exist in production.
	if (r_image->is_empty()) { // Try PNG first.
		r_image->load_png_from_buffer(p_bytes);
	}
	if (r_image->is_empty()) { // And then JPEG.
		r_image->load_jpg_from_buffer(p_bytes);
	}
	// If it still can't be loaded, give up and insert an empty image as placeholder.
	if (r_image->is_empty()) {
		ERR_PRINT(vformat("glTF: Couldn't load image index '%d' with its given mimetype: %s.", p_index, p_mime_type));
	}
	return r_image;
}

void GLTFDocument::_parse_image_save_image(Ref<GLTFState> p_state, const Vector<uint8_t> &p_bytes, const String &p_resource_uri, const String &p_file_extension, int p_index, Ref<Image> p_image) {
	GLTFState::HandleBinaryImageMode handling = GLTFState::HandleBinaryImageMode(p_state->handle_binary_image_mode);
	if (p_image->is_empty() || handling == GLTFState::HandleBinaryImageMode::HANDLE_BINARY_IMAGE_MODE_DISCARD_TEXTURES) {
		p_state->images.push_back(Ref<Texture2D>());
		p_state->source_images.push_back(Ref<Image>());
		return;
	}
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && handling == GLTFState::HandleBinaryImageMode::HANDLE_BINARY_IMAGE_MODE_EXTRACT_TEXTURES) {
		if (p_state->extract_path.is_empty()) {
			WARN_PRINT("glTF: Couldn't extract image because the base and extract paths are empty. It will be loaded directly instead, uncompressed.");
		} else if (p_state->extract_path.begins_with("res://.godot/imported")) {
			WARN_PRINT(vformat("glTF: Extract path is in the imported directory. Image index '%d' will be loaded directly, uncompressed.", p_index));
		} else {
			if (p_image->get_name().is_empty()) {
				WARN_PRINT(vformat("glTF: Image index '%d' did not have a name. It will be automatically given a name based on its index.", p_index));
				p_image->set_name(itos(p_index));
			}
			bool must_write = true; // If the resource does not exist on the disk within res:// directory write it.
			bool must_import = true; // Trigger import.
			Vector<uint8_t> img_data = p_image->get_data();
			Dictionary generator_parameters;
			String file_path;
			// If resource_uri is within res:// folder but outside of .godot/imported folder, use it.
			if (!p_resource_uri.is_empty() && !p_resource_uri.begins_with("res://.godot/imported") && !p_resource_uri.begins_with("res://..")) {
				file_path = p_resource_uri;
				must_import = true;
				must_write = !FileAccess::exists(file_path);
			} else {
				// Texture data has to be written to the res:// folder and imported.
				file_path = p_state->get_extract_path().path_join(p_state->get_extract_prefix() + "_" + p_image->get_name());
				file_path += p_file_extension.is_empty() ? ".png" : p_file_extension;
				if (FileAccess::exists(file_path + ".import")) {
					Ref<ConfigFile> config;
					config.instantiate();
					config->load(file_path + ".import");
					if (config->has_section_key("remap", "generator_parameters")) {
						generator_parameters = (Dictionary)config->get_value("remap", "generator_parameters");
					}
					if (!generator_parameters.has("md5")) {
						must_write = false; // Didn't come from a gltf document; don't overwrite.
						must_import = false; // And don't import.
					}
				}
			}

			if (must_write) {
				String existing_md5 = generator_parameters["md5"];
				unsigned char md5_hash[16];
				CryptoCore::md5(img_data.ptr(), img_data.size(), md5_hash);
				String new_md5 = String::hex_encode_buffer(md5_hash, 16);
				generator_parameters["md5"] = new_md5;
				if (new_md5 == existing_md5) {
					must_write = false;
					must_import = false;
				}
			}
			if (must_write) {
				Error err = OK;
				if (p_file_extension.is_empty()) {
					// If a file extension was not specified, save the image data to a PNG file.
					err = p_image->save_png(file_path);
					ERR_FAIL_COND(err != OK);
				} else {
					// If a file extension was specified, save the original bytes to a file with that extension.
					Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::WRITE, &err);
					ERR_FAIL_COND(err != OK);
					file->store_buffer(p_bytes);
					file->close();
				}
			}
			if (must_import) {
				// ResourceLoader::import will crash if not is_editor_hint(), so this case is protected above and will fall through to uncompressed.
				HashMap<StringName, Variant> custom_options;
				custom_options[SNAME("mipmaps/generate")] = true;
				// Will only use project settings defaults if custom_importer is empty.

				EditorFileSystem::get_singleton()->update_file(file_path);
				EditorFileSystem::get_singleton()->reimport_append(file_path, custom_options, String(), generator_parameters);
			}
			Ref<Texture2D> saved_image = ResourceLoader::load(file_path, "Texture2D");
			if (saved_image.is_valid()) {
				p_state->images.push_back(saved_image);
				p_state->source_images.push_back(saved_image->get_image());
				return;
			} else {
				WARN_PRINT(vformat("glTF: Image index '%d' with the name '%s' resolved to %s couldn't be imported. It will be loaded directly instead, uncompressed.", p_index, p_image->get_name(), file_path));
			}
		}
	}
#endif // TOOLS_ENABLED
	if (handling == GLTFState::HandleBinaryImageMode::HANDLE_BINARY_IMAGE_MODE_EMBED_AS_BASISU) {
		Ref<PortableCompressedTexture2D> tex;
		tex.instantiate();
		tex->set_name(p_image->get_name());
		tex->set_keep_compressed_buffer(true);
		tex->create_from_image(p_image, PortableCompressedTexture2D::COMPRESSION_MODE_BASIS_UNIVERSAL);
		p_state->images.push_back(tex);
		p_state->source_images.push_back(p_image);
		return;
	}
	// This handles the case of HANDLE_BINARY_IMAGE_MODE_EMBED_AS_UNCOMPRESSED, and it also serves
	// as a fallback for HANDLE_BINARY_IMAGE_MODE_EXTRACT_TEXTURES when this is not the editor.
	Ref<ImageTexture> tex;
	tex.instantiate();
	tex->set_name(p_image->get_name());
	tex->set_image(p_image);
	p_state->images.push_back(tex);
	p_state->source_images.push_back(p_image);
}

Error GLTFDocument::_parse_images(Ref<GLTFState> p_state, const String &p_base_path) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	if (!p_state->json.has("images")) {
		return OK;
	}

	// Ref: https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#images

	const Array &images = p_state->json["images"];
	HashSet<String> used_names;
	for (int i = 0; i < images.size(); i++) {
		const Dictionary &dict = images[i];

		// glTF 2.0 supports PNG and JPEG types, which can be specified as (from spec):
		// "- a URI to an external file in one of the supported images formats, or
		//  - a URI with embedded base64-encoded data, or
		//  - a reference to a bufferView; in that case mimeType must be defined."
		// Since mimeType is optional for external files and base64 data, we'll have to
		// fall back on letting Godot parse the data to figure out if it's PNG or JPEG.

		// We'll assume that we use either URI or bufferView, so let's warn the user
		// if their image somehow uses both. And fail if it has neither.
		ERR_CONTINUE_MSG(!dict.has("uri") && !dict.has("bufferView"), "Invalid image definition in glTF file, it should specify an 'uri' or 'bufferView'.");
		if (dict.has("uri") && dict.has("bufferView")) {
			WARN_PRINT("Invalid image definition in glTF file using both 'uri' and 'bufferView'. 'uri' will take precedence.");
		}

		String mime_type;
		if (dict.has("mimeType")) { // Should be "image/png", "image/jpeg", or something handled by an extension.
			mime_type = dict["mimeType"];
		}

		String image_name;
		if (dict.has("name")) {
			image_name = dict["name"];
			image_name = image_name.get_file().get_basename().validate_filename();
		}
		if (image_name.is_empty()) {
			image_name = itos(i);
		}
		while (used_names.has(image_name)) {
			image_name += "_" + itos(i);
		}

		String resource_uri;

		used_names.insert(image_name);
		// Load the image data. If we get a byte array, store here for later.
		Vector<uint8_t> data;
		if (dict.has("uri")) {
			// Handles the first two bullet points from the spec (embedded data, or external file).
			String uri = dict["uri"];
			if (uri.begins_with("data:")) { // Embedded data using base64.
				data = _parse_base64_uri(uri);
				// mimeType is optional, but if we have it defined in the URI, let's use it.
				if (mime_type.is_empty() && uri.contains_char(';')) {
					// Trim "data:" prefix which is 5 characters long, and end at ";base64".
					mime_type = uri.substr(5, uri.find(";base64") - 5);
				}
			} else { // Relative path to an external image file.
				ERR_FAIL_COND_V(p_base_path.is_empty(), ERR_INVALID_PARAMETER);
				uri = uri.uri_file_decode();
				uri = p_base_path.path_join(uri).replace_char('\\', '/'); // Fix for Windows.
				resource_uri = uri.simplify_path();
				// ResourceLoader will rely on the file extension to use the relevant loader.
				// The spec says that if mimeType is defined, it should take precedence (e.g.
				// there could be a `.png` image which is actually JPEG), but there's no easy
				// API for that in Godot, so we'd have to load as a buffer (i.e. embedded in
				// the material), so we only do that only as fallback.
				if (ResourceLoader::exists(resource_uri)) {
					Ref<Texture2D> texture = ResourceLoader::load(resource_uri, "Texture2D");
					if (texture.is_valid()) {
						p_state->images.push_back(texture);
						p_state->source_images.push_back(texture->get_image());
						continue;
					}
				}
				// mimeType is optional, but if we have it in the file extension, let's use it.
				// If the mimeType does not match with the file extension, either it should be
				// specified in the file, or the GLTFDocumentExtension should handle it.
				if (mime_type.is_empty()) {
					mime_type = "image/" + resource_uri.get_extension();
				}
				// Fallback to loading as byte array. This enables us to support the
				// spec's requirement that we honor mimetype regardless of file URI.
				data = FileAccess::get_file_as_bytes(resource_uri);
				if (data.is_empty()) {
					WARN_PRINT(vformat("glTF: Image index '%d' couldn't be loaded as a buffer of MIME type '%s' from URI: %s because there was no data to load. Skipping it.", i, mime_type, resource_uri));
					p_state->images.push_back(Ref<Texture2D>()); // Placeholder to keep count.
					p_state->source_images.push_back(Ref<Image>());
					continue;
				}
			}
		} else if (dict.has("bufferView")) {
			// Handles the third bullet point from the spec (bufferView).
			ERR_FAIL_COND_V_MSG(mime_type.is_empty(), ERR_FILE_CORRUPT, vformat("glTF: Image index '%d' specifies 'bufferView' but no 'mimeType', which is invalid.", i));
			const GLTFBufferViewIndex bvi = dict["bufferView"];
			ERR_FAIL_INDEX_V(bvi, p_state->buffer_views.size(), ERR_PARAMETER_RANGE_ERROR);
			Ref<GLTFBufferView> bv = p_state->buffer_views[bvi];
			const GLTFBufferIndex bi = bv->buffer;
			ERR_FAIL_INDEX_V(bi, p_state->buffers.size(), ERR_PARAMETER_RANGE_ERROR);
			ERR_FAIL_COND_V(bv->byte_offset + bv->byte_length > p_state->buffers[bi].size(), ERR_FILE_CORRUPT);
			const PackedByteArray &buffer = p_state->buffers[bi];
			data = buffer.slice(bv->byte_offset, bv->byte_offset + bv->byte_length);
		}
		// Done loading the image data bytes. Check that we actually got data to parse.
		// Note: There are paths above that return early, so this point might not be reached.
		if (data.is_empty()) {
			WARN_PRINT(vformat("glTF: Image index '%d' couldn't be loaded, no data found. Skipping it.", i));
			p_state->images.push_back(Ref<Texture2D>()); // Placeholder to keep count.
			p_state->source_images.push_back(Ref<Image>());
			continue;
		}
		// Parse the image data from bytes into an Image resource and save if needed.
		String file_extension;
		Ref<Image> img = _parse_image_bytes_into_image(p_state, data, mime_type, i, file_extension);
		img->set_name(image_name);
		_parse_image_save_image(p_state, data, resource_uri, file_extension, i, img);
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
		Dictionary texture_dict;
		Ref<GLTFTexture> gltf_texture = p_state->textures[i];
		if (_image_save_extension.is_valid()) {
			Error err = _image_save_extension->serialize_texture_json(p_state, texture_dict, gltf_texture, _image_format);
			ERR_FAIL_COND_V(err != OK, err);
			// If a fallback image format was specified, serialize another image for it.
			// Note: This must only be done after serializing other images to keep the indices of those consistent.
			if (_fallback_image_format != "None" && p_state->json.has("images")) {
				Array json_images = p_state->json["images"];
				texture_dict["source"] = json_images.size();
				Ref<Image> image = p_state->source_images[gltf_texture->get_src_image()];
				String fallback_name = _gen_unique_name(p_state, image->get_name() + "_fallback");
				image = image->duplicate();
				image->set_name(fallback_name);
				ERR_CONTINUE(image.is_null());
				if (_fallback_image_format == "PNG") {
					image->resize(image->get_width() * _fallback_image_quality, image->get_height() * _fallback_image_quality);
				}
				json_images.push_back(_serialize_image(p_state, image, _fallback_image_format, _fallback_image_quality, nullptr));
			}
		} else {
			ERR_CONTINUE(gltf_texture->get_src_image() == -1);
			texture_dict["source"] = gltf_texture->get_src_image();
		}
		GLTFTextureSamplerIndex sampler_index = gltf_texture->get_sampler();
		if (sampler_index != -1) {
			texture_dict["sampler"] = sampler_index;
		}
		textures.push_back(texture_dict);
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
		const Dictionary &texture_dict = textures[i];
		Ref<GLTFTexture> gltf_texture;
		gltf_texture.instantiate();
		// Check if any GLTFDocumentExtensions want to handle this texture JSON.
		for (Ref<GLTFDocumentExtension> ext : document_extensions) {
			ERR_CONTINUE(ext.is_null());
			Error err = ext->parse_texture_json(p_state, texture_dict, gltf_texture);
			ERR_CONTINUE_MSG(err != OK, "glTF: Encountered error " + itos(err) + " when parsing texture JSON " + String(Variant(texture_dict)) + " in file " + p_state->filename + ". Continuing.");
			if (gltf_texture->get_src_image() != -1) {
				break;
			}
		}
		if (gltf_texture->get_src_image() == -1) {
			// No extensions handled it, so use the base glTF source.
			// This may be the fallback, or the only option anyway.
			ERR_FAIL_COND_V(!texture_dict.has("source"), ERR_PARSE_ERROR);
			gltf_texture->set_src_image(texture_dict["source"]);
		}
		if (gltf_texture->get_sampler() == -1 && texture_dict.has("sampler")) {
			gltf_texture->set_sampler(texture_dict["sampler"]);
		}
		p_state->textures.push_back(gltf_texture);
	}

	return OK;
}

GLTFTextureIndex GLTFDocument::_set_texture(Ref<GLTFState> p_state, Ref<Texture2D> p_texture, StandardMaterial3D::TextureFilter p_filter_mode, bool p_repeats) {
	ERR_FAIL_COND_V(p_texture.is_null(), -1);
	Ref<GLTFTexture> gltf_texture;
	gltf_texture.instantiate();
	ERR_FAIL_COND_V(p_texture->get_image().is_null(), -1);
	GLTFImageIndex gltf_src_image_i = p_state->images.find(p_texture);
	if (gltf_src_image_i == -1) {
		gltf_src_image_i = p_state->images.size();
		p_state->images.push_back(p_texture);
		p_state->source_images.push_back(p_texture->get_image());
	}
	gltf_texture->set_src_image(gltf_src_image_i);
	gltf_texture->set_sampler(_set_sampler_for_mode(p_state, p_filter_mode, p_repeats));
	GLTFTextureIndex gltf_texture_i = p_state->textures.size();
	p_state->textures.push_back(gltf_texture);
	return gltf_texture_i;
}

Ref<Texture2D> GLTFDocument::_get_texture(Ref<GLTFState> p_state, const GLTFTextureIndex p_texture, int p_texture_types) {
	ERR_FAIL_COND_V_MSG(p_state->textures.is_empty(), Ref<Texture2D>(), "glTF import: Tried to read texture at index " + itos(p_texture) + ", but this glTF file does not contain any textures.");
	ERR_FAIL_INDEX_V(p_texture, p_state->textures.size(), Ref<Texture2D>());
	const GLTFImageIndex image = p_state->textures[p_texture]->get_src_image();
	ERR_FAIL_INDEX_V(image, p_state->images.size(), Ref<Texture2D>());
	if (GLTFState::HandleBinaryImageMode(p_state->handle_binary_image_mode) == GLTFState::HandleBinaryImageMode::HANDLE_BINARY_IMAGE_MODE_EMBED_AS_BASISU) {
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

GLTFTextureSamplerIndex GLTFDocument::_set_sampler_for_mode(Ref<GLTFState> p_state, StandardMaterial3D::TextureFilter p_filter_mode, bool p_repeats) {
	for (int i = 0; i < p_state->texture_samplers.size(); ++i) {
		if (p_state->texture_samplers[i]->get_filter_mode() == p_filter_mode) {
			return i;
		}
	}

	GLTFTextureSamplerIndex gltf_sampler_i = p_state->texture_samplers.size();
	Ref<GLTFTextureSampler> gltf_sampler;
	gltf_sampler.instantiate();
	gltf_sampler->set_filter_mode(p_filter_mode);
	gltf_sampler->set_wrap_mode(p_repeats);
	p_state->texture_samplers.push_back(gltf_sampler);
	return gltf_sampler_i;
}

Ref<GLTFTextureSampler> GLTFDocument::_get_sampler_for_texture(Ref<GLTFState> p_state, const GLTFTextureIndex p_texture) {
	ERR_FAIL_COND_V_MSG(p_state->textures.is_empty(), Ref<GLTFTextureSampler>(), "glTF import: Tried to read sampler for texture at index " + itos(p_texture) + ", but this glTF file does not contain any textures.");
	ERR_FAIL_INDEX_V(p_texture, p_state->textures.size(), Ref<GLTFTextureSampler>());
	const GLTFTextureSamplerIndex sampler = p_state->textures[p_texture]->get_sampler();

	if (sampler == -1) {
		return p_state->default_texture_sampler;
	} else {
		ERR_FAIL_INDEX_V(sampler, p_state->texture_samplers.size(), Ref<GLTFTextureSampler>());

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
	p_state->default_texture_sampler.instantiate();
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
		sampler.instantiate();

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
			sampler->set_wrap_s(GLTFTextureSampler::WrapMode::DEFAULT);
		}

		if (d.has("wrapT")) {
			sampler->set_wrap_t(d["wrapT"]);
		} else {
			sampler->set_wrap_t(GLTFTextureSampler::WrapMode::DEFAULT);
		}

		p_state->texture_samplers.push_back(sampler);
	}

	return OK;
}

static inline void _set_material_texture_name(const Ref<Texture2D> &p_texture, const String &p_path, const String &p_mat_name, const String &p_suffix) {
	if (p_texture->get_name().is_empty()) {
		if (p_path.is_empty()) {
			p_texture->set_name(p_mat_name + p_suffix);
		} else {
			p_texture->set_name(p_path.get_file().get_basename());
		}
	}
}

Error GLTFDocument::_serialize_materials(Ref<GLTFState> p_state) {
	Array materials;
	for (int32_t i = 0; i < p_state->materials.size(); i++) {
		Dictionary mat_dict;
		Ref<Material> material = p_state->materials[i];
		if (material.is_null()) {
			materials.push_back(mat_dict);
			continue;
		}
		String mat_name = material->get_name();
		if (mat_name.is_empty()) {
			const String &mat_path = material->get_path();
			if (!mat_path.is_empty() && !mat_path.contains("::")) {
				mat_name = mat_path.get_file().get_basename();
			}
		}
		if (!mat_name.is_empty()) {
			mat_dict["name"] = _gen_unique_name(p_state, mat_name);
		}

		Ref<BaseMaterial3D> base_material = material;
		if (base_material.is_null()) {
			materials.push_back(mat_dict);
			continue;
		}

		Dictionary mr;
		{
			const Color c = base_material->get_albedo().srgb_to_linear();
			Array arr = { c.r, c.g, c.b, c.a };
			mr["baseColorFactor"] = arr;
		}
		if (_image_format != "None") {
			Dictionary bct;
			Ref<Texture2D> albedo_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_ALBEDO);
			GLTFTextureIndex gltf_texture_index = -1;

			if (albedo_texture.is_valid() && albedo_texture->get_image().is_valid()) {
				_set_material_texture_name(albedo_texture, albedo_texture->get_path(), mat_name, "_albedo");
				gltf_texture_index = _set_texture(p_state, albedo_texture, base_material->get_texture_filter(), base_material->get_flag(BaseMaterial3D::FLAG_USE_TEXTURE_REPEAT));
			}
			if (gltf_texture_index != -1) {
				bct["index"] = gltf_texture_index;
				Dictionary extensions = _serialize_texture_transform_uv1(material);
				if (!extensions.is_empty()) {
					bct["extensions"] = extensions;
					p_state->use_khr_texture_transform = true;
				}
				mr["baseColorTexture"] = bct;
			}
		}

		// Godot allows setting BaseMaterial3D roughness/metallic above 1.0 (which has an effect on how
		// the roughness/metallic texture is interpreted), but the glTF specification doesn't allow this.
		mr["metallicFactor"] = MIN(base_material->get_metallic(), 1.0);
		mr["roughnessFactor"] = MIN(base_material->get_roughness(), 1.0);
		if (_image_format != "None") {
			bool has_roughness = base_material->get_texture(BaseMaterial3D::TEXTURE_ROUGHNESS).is_valid() && base_material->get_texture(BaseMaterial3D::TEXTURE_ROUGHNESS)->get_image().is_valid();
			bool has_ao = base_material->get_feature(BaseMaterial3D::FEATURE_AMBIENT_OCCLUSION) && base_material->get_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION).is_valid();
			bool has_metalness = base_material->get_texture(BaseMaterial3D::TEXTURE_METALLIC).is_valid() && base_material->get_texture(BaseMaterial3D::TEXTURE_METALLIC)->get_image().is_valid();
			Ref<Texture2D> original_orm_tex = base_material->get_texture(BaseMaterial3D::TEXTURE_ORM);
			GLTFTextureIndex orm_texture_index = -1;
			if (has_ao || has_roughness || has_metalness) {
				Ref<Texture2D> roughness_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_ROUGHNESS);
				BaseMaterial3D::TextureChannel roughness_channel = base_material->get_roughness_texture_channel();
				Ref<Texture2D> metallic_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_METALLIC);
				BaseMaterial3D::TextureChannel metalness_channel = base_material->get_metallic_texture_channel();
				Ref<Texture2D> ao_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION);
				BaseMaterial3D::TextureChannel ao_channel = base_material->get_ao_texture_channel();
				Ref<ImageTexture> orm_texture;
				orm_texture.instantiate();
				Ref<Image> orm_image;
				orm_image.instantiate();
				int32_t height = 0;
				int32_t width = 0;
				Ref<Image> ao_image;
				HashSet<String> common_paths; // For setting name
				if (has_ao) {
					height = ao_texture->get_height();
					width = ao_texture->get_width();
					ao_image = _duplicate_and_decompress_image(ao_texture->get_image());
					if (!ao_texture->get_path().is_empty()) {
						common_paths.insert(ao_texture->get_path());
					}
				}
				Ref<Image> roughness_image;
				if (has_roughness) {
					height = roughness_texture->get_height();
					width = roughness_texture->get_width();
					roughness_image = _duplicate_and_decompress_image(roughness_texture->get_image());
					if (!roughness_texture->get_path().is_empty()) {
						common_paths.insert(roughness_texture->get_path());
					}
				}
				Ref<Image> metallness_image;
				if (has_metalness) {
					height = metallic_texture->get_height();
					width = metallic_texture->get_width();
					metallness_image = _duplicate_and_decompress_image(metallic_texture->get_image());
					if (!metallic_texture->get_path().is_empty()) {
						common_paths.insert(metallic_texture->get_path());
					}
				}
				Ref<Texture2D> albedo_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_ALBEDO);
				if (albedo_texture.is_valid() && albedo_texture->get_image().is_valid()) {
					height = albedo_texture->get_height();
					width = albedo_texture->get_width();
				}
				orm_image->initialize_data(width, height, false, Image::FORMAT_RGBA8);
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
				orm_texture->set_image(orm_image);
				if (has_ao || has_roughness || has_metalness) {
					// If they all share the same path, use it for the name.
					const String path = common_paths.size() == 1 ? *common_paths.begin() : String();
					_set_material_texture_name(orm_texture, path, mat_name, "_orm");
					orm_texture_index = _set_texture(p_state, orm_texture, base_material->get_texture_filter(), base_material->get_flag(BaseMaterial3D::FLAG_USE_TEXTURE_REPEAT));
				}
			} else if (original_orm_tex.is_valid() && original_orm_tex->get_image().is_valid()) {
				has_ao = true;
				has_roughness = true;
				has_metalness = true;

				_set_material_texture_name(original_orm_tex, original_orm_tex->get_path(), mat_name, "_orm");
				orm_texture_index = _set_texture(p_state, original_orm_tex, base_material->get_texture_filter(), base_material->get_flag(BaseMaterial3D::FLAG_USE_TEXTURE_REPEAT));
			}
			if (orm_texture_index != -1) {
				if (has_ao) {
					Dictionary occt;
					occt["index"] = orm_texture_index;
					mat_dict["occlusionTexture"] = occt;
				}
				if (has_roughness || has_metalness) {
					Dictionary mrt;
					mrt["index"] = orm_texture_index;
					Dictionary extensions = _serialize_texture_transform_uv1(material);
					if (!extensions.is_empty()) {
						mrt["extensions"] = extensions;
						p_state->use_khr_texture_transform = true;
					}
					mr["metallicRoughnessTexture"] = mrt;
				}
			}
		}

		mat_dict["pbrMetallicRoughness"] = mr;
		if (base_material->get_feature(BaseMaterial3D::FEATURE_NORMAL_MAPPING) && _image_format != "None") {
			Dictionary nt;
			Ref<ImageTexture> tex;
			tex.instantiate();
			String path;
			{
				Ref<Texture2D> normal_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_NORMAL);
				if (normal_texture.is_valid() && normal_texture->get_image().is_valid()) {
					path = normal_texture->get_path();
					// Code for uncompressing RG normal maps
					Ref<Image> img = _duplicate_and_decompress_image(normal_texture->get_image());
					img->convert(Image::FORMAT_RGBA8);
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
					tex->set_image(img);
				}
			}
			GLTFTextureIndex gltf_texture_index = -1;
			if (tex.is_valid() && tex->get_image().is_valid()) {
				_set_material_texture_name(tex, path, mat_name, "_normal");
				gltf_texture_index = _set_texture(p_state, tex, base_material->get_texture_filter(), base_material->get_flag(BaseMaterial3D::FLAG_USE_TEXTURE_REPEAT));
			}
			nt["scale"] = base_material->get_normal_scale();
			if (gltf_texture_index != -1) {
				nt["index"] = gltf_texture_index;
				mat_dict["normalTexture"] = nt;
			}
		}

		if (base_material->get_feature(BaseMaterial3D::FEATURE_EMISSION)) {
			const Color c = base_material->get_emission().linear_to_srgb();
			Array arr = { c.r, c.g, c.b };
			mat_dict["emissiveFactor"] = arr;
		}

		if (base_material->get_feature(BaseMaterial3D::FEATURE_EMISSION) && _image_format != "None") {
			Dictionary et;
			Ref<Texture2D> emission_texture = base_material->get_texture(BaseMaterial3D::TEXTURE_EMISSION);
			GLTFTextureIndex gltf_texture_index = -1;
			if (emission_texture.is_valid() && emission_texture->get_image().is_valid()) {
				_set_material_texture_name(emission_texture, emission_texture->get_path(), mat_name, "_emission");
				gltf_texture_index = _set_texture(p_state, emission_texture, base_material->get_texture_filter(), base_material->get_flag(BaseMaterial3D::FLAG_USE_TEXTURE_REPEAT));
			}

			if (gltf_texture_index != -1) {
				et["index"] = gltf_texture_index;
				mat_dict["emissiveTexture"] = et;
			}
		}

		const bool ds = base_material->get_cull_mode() == BaseMaterial3D::CULL_DISABLED;
		if (ds) {
			mat_dict["doubleSided"] = ds;
		}

		if (base_material->get_transparency() == BaseMaterial3D::TRANSPARENCY_ALPHA_SCISSOR) {
			mat_dict["alphaMode"] = "MASK";
			mat_dict["alphaCutoff"] = base_material->get_alpha_scissor_threshold();
		} else if (base_material->get_transparency() != BaseMaterial3D::TRANSPARENCY_DISABLED) {
			mat_dict["alphaMode"] = "BLEND";
		}

		Dictionary extensions;
		if (base_material->get_shading_mode() == BaseMaterial3D::SHADING_MODE_UNSHADED) {
			Dictionary mat_unlit;
			extensions["KHR_materials_unlit"] = mat_unlit;
			p_state->add_used_extension("KHR_materials_unlit");
		}
		if (base_material->get_feature(BaseMaterial3D::FEATURE_EMISSION) && !Math::is_equal_approx(base_material->get_emission_energy_multiplier(), 1.0f)) {
			Dictionary mat_emissive_strength;
			mat_emissive_strength["emissiveStrength"] = base_material->get_emission_energy_multiplier();
			extensions["KHR_materials_emissive_strength"] = mat_emissive_strength;
			p_state->add_used_extension("KHR_materials_emissive_strength");
		}
		if (!extensions.is_empty()) {
			mat_dict["extensions"] = extensions;
		}

		_attach_meta_to_extras(material, mat_dict);
		materials.push_back(mat_dict);
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
		const Dictionary &material_dict = materials[i];

		Ref<StandardMaterial3D> material;
		material.instantiate();
		if (material_dict.has("name") && !String(material_dict["name"]).is_empty()) {
			material->set_name(material_dict["name"]);
		} else {
			material->set_name(vformat("material_%s", itos(i)));
		}
		Dictionary material_extensions;
		if (material_dict.has("extensions")) {
			material_extensions = material_dict["extensions"];
		}

		if (material_extensions.has("KHR_materials_unlit")) {
			material->set_shading_mode(BaseMaterial3D::SHADING_MODE_UNSHADED);
		}

		if (material_extensions.has("KHR_materials_emissive_strength")) {
			Dictionary emissive_strength = material_extensions["KHR_materials_emissive_strength"];
			if (emissive_strength.has("emissiveStrength")) {
				material->set_emission_energy_multiplier(emissive_strength["emissiveStrength"]);
			}
		}

		if (material_extensions.has("KHR_materials_pbrSpecularGlossiness")) {
			WARN_PRINT("Material uses a specular and glossiness workflow. Textures will be converted to roughness and metallic workflow, which may not be 100% accurate.");
			Dictionary sgm = material_extensions["KHR_materials_pbrSpecularGlossiness"];

			Ref<GLTFSpecGloss> spec_gloss;
			spec_gloss.instantiate();
			if (sgm.has("diffuseTexture")) {
				const Dictionary &diffuse_texture_dict = sgm["diffuseTexture"];
				if (diffuse_texture_dict.has("index")) {
					Ref<GLTFTextureSampler> diffuse_sampler = _get_sampler_for_texture(p_state, diffuse_texture_dict["index"]);
					if (diffuse_sampler.is_valid()) {
						material->set_texture_filter(diffuse_sampler->get_filter_mode());
						material->set_flag(BaseMaterial3D::FLAG_USE_TEXTURE_REPEAT, diffuse_sampler->get_wrap_mode());
					}
					Ref<Texture2D> diffuse_texture = _get_texture(p_state, diffuse_texture_dict["index"], TEXTURE_TYPE_GENERIC);
					if (diffuse_texture.is_valid()) {
						spec_gloss->diffuse_img = diffuse_texture->get_image();
						material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, diffuse_texture);
					}
				}
			}
			if (sgm.has("diffuseFactor")) {
				const Array &arr = sgm["diffuseFactor"];
				ERR_FAIL_COND_V(arr.size() != 4, ERR_PARSE_ERROR);
				const Color c = Color(arr[0], arr[1], arr[2], arr[3]).linear_to_srgb();
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
					const Ref<Texture2D> orig_texture = _get_texture(p_state, spec_gloss_texture["index"], TEXTURE_TYPE_GENERIC);
					if (orig_texture.is_valid()) {
						spec_gloss->spec_gloss_img = orig_texture->get_image();
					}
				}
			}
			spec_gloss_to_rough_metal(spec_gloss, material);

		} else if (material_dict.has("pbrMetallicRoughness")) {
			const Dictionary &mr = material_dict["pbrMetallicRoughness"];
			if (mr.has("baseColorFactor")) {
				const Array &arr = mr["baseColorFactor"];
				ERR_FAIL_COND_V(arr.size() != 4, ERR_PARSE_ERROR);
				const Color c = Color(arr[0], arr[1], arr[2], arr[3]).linear_to_srgb();
				material->set_albedo(c);
			}

			if (mr.has("baseColorTexture")) {
				const Dictionary &bct = mr["baseColorTexture"];
				if (bct.has("index")) {
					const GLTFTextureIndex base_color_texture_index = bct["index"];
					material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, _get_texture(p_state, base_color_texture_index, TEXTURE_TYPE_GENERIC));
					const Ref<GLTFTextureSampler> bct_sampler = _get_sampler_for_texture(p_state, base_color_texture_index);
					if (bct_sampler.is_valid()) {
						material->set_texture_filter(bct_sampler->get_filter_mode());
						material->set_flag(BaseMaterial3D::FLAG_USE_TEXTURE_REPEAT, bct_sampler->get_wrap_mode());
					}
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
					const Ref<Texture2D> t = _get_texture(p_state, bct["index"], TEXTURE_TYPE_GENERIC);
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

		if (material_dict.has("normalTexture")) {
			const Dictionary &bct = material_dict["normalTexture"];
			if (bct.has("index")) {
				material->set_texture(BaseMaterial3D::TEXTURE_NORMAL, _get_texture(p_state, bct["index"], TEXTURE_TYPE_NORMAL));
				material->set_feature(BaseMaterial3D::FEATURE_NORMAL_MAPPING, true);
			}
			if (bct.has("scale")) {
				material->set_normal_scale(bct["scale"]);
			}
		}
		if (material_dict.has("occlusionTexture")) {
			const Dictionary &bct = material_dict["occlusionTexture"];
			if (bct.has("index")) {
				material->set_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION, _get_texture(p_state, bct["index"], TEXTURE_TYPE_GENERIC));
				material->set_ao_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_RED);
				material->set_feature(BaseMaterial3D::FEATURE_AMBIENT_OCCLUSION, true);
			}
		}

		if (material_dict.has("emissiveFactor")) {
			const Array &arr = material_dict["emissiveFactor"];
			ERR_FAIL_COND_V(arr.size() != 3, ERR_PARSE_ERROR);
			const Color c = Color(arr[0], arr[1], arr[2]).linear_to_srgb();
			material->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);

			material->set_emission(c);
		}

		if (material_dict.has("emissiveTexture")) {
			const Dictionary &bct = material_dict["emissiveTexture"];
			if (bct.has("index")) {
				material->set_texture(BaseMaterial3D::TEXTURE_EMISSION, _get_texture(p_state, bct["index"], TEXTURE_TYPE_GENERIC));
				material->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);
				material->set_emission_operator(BaseMaterial3D::EMISSION_OP_MULTIPLY);
				// glTF spec: emissiveFactor × emissiveTexture. Use WHITE if no factor specified.
				if (!material_dict.has("emissiveFactor")) {
					material->set_emission(Color(1, 1, 1));
				}
			}
		}

		if (material_dict.has("doubleSided")) {
			const bool ds = material_dict["doubleSided"];
			if (ds) {
				material->set_cull_mode(BaseMaterial3D::CULL_DISABLED);
			}
		}
		if (material_dict.has("alphaMode")) {
			const String &am = material_dict["alphaMode"];
			if (am == "BLEND") {
				material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA_DEPTH_PRE_PASS);
			} else if (am == "MASK") {
				material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA_SCISSOR);
			}
		}
		if (material_dict.has("alphaCutoff")) {
			material->set_alpha_scissor_threshold(material_dict["alphaCutoff"]);
		} else {
			material->set_alpha_scissor_threshold(0.5f);
		}

		if (material_dict.has("extras")) {
			_attach_extras_to_meta(material_dict["extras"], material);
		}
		p_state->materials.push_back(material);
	}

	print_verbose("Total materials: " + itos(p_state->materials.size()));

	return OK;
}

void GLTFDocument::_set_texture_transform_uv1(const Dictionary &p_dict, Ref<BaseMaterial3D> p_material) {
	if (p_dict.has("extensions")) {
		const Dictionary &extensions = p_dict["extensions"];
		if (extensions.has("KHR_texture_transform")) {
			if (p_material.is_valid()) {
				const Dictionary &texture_transform = extensions["KHR_texture_transform"];
				if (texture_transform.has("offset")) {
					const Array offset_arr = texture_transform["offset"];
					if (offset_arr.size() == 2) {
						const Vector3 offset_vector3 = Vector3(offset_arr[0], offset_arr[1], 0.0f);
						p_material->set_uv1_offset(offset_vector3);
					}
				}

				if (texture_transform.has("scale")) {
					const Array scale_arr = texture_transform["scale"];
					if (scale_arr.size() == 2) {
						const Vector3 scale_vector3 = Vector3(scale_arr[0], scale_arr[1], 1.0f);
						p_material->set_uv1_scale(scale_vector3);
					}
				}
			}
		}
	}
}

void GLTFDocument::spec_gloss_to_rough_metal(Ref<GLTFSpecGloss> r_spec_gloss, Ref<BaseMaterial3D> p_material) {
	if (r_spec_gloss.is_null()) {
		return;
	}
	if (r_spec_gloss->spec_gloss_img.is_null()) {
		return;
	}
	if (r_spec_gloss->diffuse_img.is_null()) {
		return;
	}
	if (p_material.is_null()) {
		return;
	}
	bool has_roughness = false;
	bool has_metal = false;
	p_material->set_roughness(1.0f);
	p_material->set_metallic(1.0f);
	Ref<Image> rm_img = Image::create_empty(r_spec_gloss->spec_gloss_img->get_width(), r_spec_gloss->spec_gloss_img->get_height(), false, Image::FORMAT_RGBA8);
	r_spec_gloss->spec_gloss_img->decompress();
	if (r_spec_gloss->diffuse_img.is_valid()) {
		r_spec_gloss->diffuse_img->decompress();
		r_spec_gloss->diffuse_img->resize(r_spec_gloss->spec_gloss_img->get_width(), r_spec_gloss->spec_gloss_img->get_height(), Image::INTERPOLATE_LANCZOS);
		r_spec_gloss->spec_gloss_img->resize(r_spec_gloss->diffuse_img->get_width(), r_spec_gloss->diffuse_img->get_height(), Image::INTERPOLATE_LANCZOS);
	}
	for (int32_t y = 0; y < r_spec_gloss->spec_gloss_img->get_height(); y++) {
		for (int32_t x = 0; x < r_spec_gloss->spec_gloss_img->get_width(); x++) {
			const Color specular_pixel = r_spec_gloss->spec_gloss_img->get_pixel(x, y).srgb_to_linear();
			Color specular = Color(specular_pixel.r, specular_pixel.g, specular_pixel.b);
			specular *= r_spec_gloss->specular_factor;
			Color diffuse = Color(1.0f, 1.0f, 1.0f);
			diffuse *= r_spec_gloss->diffuse_img->get_pixel(x, y).srgb_to_linear();
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
				r_spec_gloss->diffuse_img->set_pixel(x, y, base_color.linear_to_srgb());
			}
		}
	}
	rm_img->generate_mipmaps();
	r_spec_gloss->diffuse_img->generate_mipmaps();
	p_material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, ImageTexture::create_from_image(r_spec_gloss->diffuse_img));
	Ref<ImageTexture> rm_image_texture = ImageTexture::create_from_image(rm_img);
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
Error GLTFDocument::_parse_skins(Ref<GLTFState> p_state) {
	if (!p_state->json.has("skins")) {
		return OK;
	}

	const Array &skins = p_state->json["skins"];

	// Create the base skins, and mark nodes that are joints
	for (int i = 0; i < skins.size(); i++) {
		const Dictionary &d = skins[i];

		Ref<GLTFSkin> skin;
		skin.instantiate();

		ERR_FAIL_COND_V(!d.has("joints"), ERR_PARSE_ERROR);

		const Array &joints = d["joints"];

		if (d.has("inverseBindMatrices")) {
			const GLTFAccessorIndex inv_bind_accessor_index = d["inverseBindMatrices"];
			Array inv_binds_arr = _decode_accessor_as_variants(p_state, inv_bind_accessor_index, Variant::TRANSFORM3D);
			ERR_FAIL_COND_V(inv_binds_arr.size() != joints.size(), ERR_PARSE_ERROR);
			GLTFTemplateConvert::set_from_array(skin->inverse_binds, inv_binds_arr);
		}

		for (int j = 0; j < joints.size(); j++) {
			const GLTFNodeIndex node = joints[j];
			ERR_FAIL_INDEX_V(node, p_state->nodes.size(), ERR_PARSE_ERROR);

			skin->joints.push_back(node);
			skin->joints_original.push_back(node);

			p_state->nodes.write[node]->joint = true;
		}

		if (d.has("name") && !String(d["name"]).is_empty()) {
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
		ERR_FAIL_COND_V(SkinTool::_expand_skin(p_state->nodes, skin), ERR_PARSE_ERROR);
		ERR_FAIL_COND_V(SkinTool::_verify_skin(p_state->nodes, skin), ERR_PARSE_ERROR);
	}

	print_verbose("glTF: Total skins: " + itos(p_state->skins.size()));

	return OK;
}
Error GLTFDocument::_serialize_skins(Ref<GLTFState> p_state) {
	_remove_duplicate_skins(p_state);
	Array json_skins;
	for (int skin_i = 0; skin_i < p_state->skins.size(); skin_i++) {
		Ref<GLTFSkin> gltf_skin = p_state->skins[skin_i];
		Dictionary json_skin;
		Array inv_binds_arr = GLTFTemplateConvert::to_array(gltf_skin->inverse_binds);
		json_skin["inverseBindMatrices"] = GLTFAccessor::encode_new_accessor_from_variants(p_state, inv_binds_arr, Variant::TRANSFORM3D, GLTFAccessor::TYPE_MAT4, GLTFAccessor::COMPONENT_TYPE_SINGLE_FLOAT);
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
		skin.instantiate();

		// Some skins don't have IBM's! What absolute monsters!
		const bool has_ibms = !gltf_skin->inverse_binds.is_empty();

		for (int joint_i = 0; joint_i < gltf_skin->joints_original.size(); ++joint_i) {
			GLTFNodeIndex node = gltf_skin->joints_original[joint_i];
			String bone_name = p_state->nodes[node]->get_name();

			Transform3D xform;
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
		if (skin->get_name().is_empty()) {
			// Make a unique name, no gltf node represents this skin
			skin->set_name(_gen_unique_name(p_state, "Skin"));
		}
	}

	return OK;
}

bool GLTFDocument::_skins_are_same(const Ref<Skin> &p_skin_a, const Ref<Skin> &p_skin_b) {
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

		Transform3D a_xform = p_skin_a->get_bind_pose(i);
		Transform3D b_xform = p_skin_b->get_bind_pose(i);

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
	if (p_state->lights.is_empty()) {
		return OK;
	}
	Array lights;
	for (GLTFLightIndex i = 0; i < p_state->lights.size(); i++) {
		lights.push_back(p_state->lights[i]->to_dictionary());
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
		cameras[i] = p_state->cameras[i]->to_dictionary();
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
		Ref<GLTFLight> light = GLTFLight::from_dictionary(lights[light_i]);
		if (light.is_null()) {
			return Error::ERR_PARSE_ERROR;
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
		p_state->cameras.push_back(GLTFCamera::from_dictionary(cameras[i]));
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
		AnimationPlayer *animation_player = p_state->animation_players[player_i];
		List<StringName> animations;
		animation_player->get_animation_list(&animations);
		for (const StringName &animation_name : animations) {
			_convert_animation(p_state, animation_player, animation_name);
		}
	}
	Array animations;
	for (GLTFAnimationIndex animation_i = 0; animation_i < p_state->animations.size(); animation_i++) {
		Dictionary d;
		Ref<GLTFAnimation> gltf_animation = p_state->animations[animation_i];
		if (gltf_animation->is_empty_of_tracks()) {
			continue;
		}

		if (!gltf_animation->get_name().is_empty()) {
			d["name"] = gltf_animation->get_name();
		}
		Array channels;
		Array samplers;
		// Serialize glTF node tracks with the vanilla glTF animation system.
		for (KeyValue<int, GLTFAnimation::NodeTrack> &track_i : gltf_animation->get_node_tracks()) {
			GLTFAnimation::NodeTrack track = track_i.value;
			if (track.position_track.times.size()) {
				Dictionary t;
				t["sampler"] = samplers.size();
				Dictionary s;

				s["interpolation"] = interpolation_to_string(track.position_track.interpolation);
				Vector<double> times = track.position_track.times;
				s["input"] = GLTFAccessor::encode_new_accessor_from_float64s(p_state, times);
				Vector<Vector3> values = track.position_track.values;
				s["output"] = GLTFAccessor::encode_new_accessor_from_vector3s(p_state, values);

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
				Vector<double> times = track.rotation_track.times;
				s["input"] = GLTFAccessor::encode_new_accessor_from_float64s(p_state, times);
				Vector<Quaternion> values = track.rotation_track.values;
				s["output"] = GLTFAccessor::encode_new_accessor_from_quaternions(p_state, values);

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
				Vector<double> times = track.scale_track.times;
				s["input"] = GLTFAccessor::encode_new_accessor_from_float64s(p_state, times);
				Vector<Vector3> values = track.scale_track.values;
				s["output"] = GLTFAccessor::encode_new_accessor_from_vector3s(p_state, values);

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
				Vector<double> times;
				const double increment = 1.0 / p_state->get_bake_fps();
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

				Vector<double> all_track_times = times;
				Vector<double> all_track_values;
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
				s["input"] = GLTFAccessor::encode_new_accessor_from_float64s(p_state, all_track_times);
				s["output"] = GLTFAccessor::encode_new_accessor_from_float64s(p_state, all_track_values);

				samplers.push_back(s);

				Dictionary target;
				target["path"] = "weights";
				target["node"] = track_i.key;

				t["target"] = target;
				channels.push_back(t);
			}
		}
		if (!gltf_animation->get_pointer_tracks().is_empty()) {
			// Serialize glTF pointer tracks with the KHR_animation_pointer extension.
			if (!p_state->extensions_used.has("KHR_animation_pointer")) {
				p_state->extensions_used.push_back("KHR_animation_pointer");
			}
			for (KeyValue<String, GLTFAnimation::Channel<Variant>> &pointer_track_iter : gltf_animation->get_pointer_tracks()) {
				const String &json_pointer = pointer_track_iter.key;
				const GLTFAnimation::Channel<Variant> &pointer_track = pointer_track_iter.value;
				const Ref<GLTFObjectModelProperty> &obj_model_prop = p_state->object_model_properties[json_pointer];
				Dictionary channel;
				channel["sampler"] = samplers.size();
				Dictionary channel_target;
				channel_target["path"] = "pointer";
				Dictionary channel_target_ext;
				Dictionary channel_target_ext_khr_anim_ptr;
				channel_target_ext_khr_anim_ptr["pointer"] = json_pointer;
				channel_target_ext["KHR_animation_pointer"] = channel_target_ext_khr_anim_ptr;
				channel_target["extensions"] = channel_target_ext;
				channel["target"] = channel_target;
				channels.push_back(channel);
				Dictionary sampler;
				sampler["input"] = GLTFAccessor::encode_new_accessor_from_float64s(p_state, pointer_track.times);
				sampler["interpolation"] = interpolation_to_string(pointer_track.interpolation);
				GLTFAccessor::GLTFComponentType component_type = obj_model_prop->get_component_type(pointer_track.values);
				// TODO: This can be made faster after this pull request is merged: https://github.com/godotengine/godot/pull/109003
				Array values_arr = GLTFTemplateConvert::to_array(pointer_track.values);
				sampler["output"] = GLTFAccessor::encode_new_accessor_from_variants(p_state, values_arr, obj_model_prop->get_variant_type(), obj_model_prop->get_accessor_type(), component_type);
				samplers.push_back(sampler);
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

	for (GLTFAnimationIndex anim_index = 0; anim_index < animations.size(); anim_index++) {
		const Dictionary &anim_dict = animations[anim_index];

		Ref<GLTFAnimation> animation;
		animation.instantiate();

		if (!anim_dict.has("channels") || !anim_dict.has("samplers")) {
			continue;
		}

		Array channels = anim_dict["channels"];
		Array samplers = anim_dict["samplers"];

		if (anim_dict.has("name")) {
			const String anim_name = anim_dict["name"];
			const String anim_name_lower = anim_name.to_lower();
			if (anim_name_lower.begins_with("loop") || anim_name_lower.ends_with("loop") || anim_name_lower.begins_with("cycle") || anim_name_lower.ends_with("cycle")) {
				animation->set_loop(true);
			}
			animation->set_original_name(anim_name);
			animation->set_name(_gen_unique_animation_name(p_state, anim_name));
		}

		for (int channel_index = 0; channel_index < channels.size(); channel_index++) {
			const Dictionary &anim_channel = channels[channel_index];
			ERR_FAIL_COND_V_MSG(!anim_channel.has("sampler"), ERR_PARSE_ERROR, "glTF: Animation channel missing required 'sampler' property.");
			ERR_FAIL_COND_V_MSG(!anim_channel.has("target"), ERR_PARSE_ERROR, "glTF: Animation channel missing required 'target' property.");
			// Parse sampler.
			const int sampler_index = anim_channel["sampler"];
			ERR_FAIL_INDEX_V(sampler_index, samplers.size(), ERR_PARSE_ERROR);
			const Dictionary &sampler_dict = samplers[sampler_index];
			ERR_FAIL_COND_V(!sampler_dict.has("input"), ERR_PARSE_ERROR);
			ERR_FAIL_COND_V(!sampler_dict.has("output"), ERR_PARSE_ERROR);
			const int input_time_accessor_index = sampler_dict["input"];
			const int output_value_accessor_index = sampler_dict["output"];
			GLTFAnimation::Interpolation interp = GLTFAnimation::INTERP_LINEAR;
			int output_count = 1;
			if (sampler_dict.has("interpolation")) {
				const String &in = sampler_dict["interpolation"];
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
			const PackedFloat64Array times = _decode_accessor_as_float64s(p_state, input_time_accessor_index);
			// Parse target.
			const Dictionary &anim_target = anim_channel["target"];
			ERR_FAIL_COND_V_MSG(!anim_target.has("path"), ERR_PARSE_ERROR, "glTF: Animation channel target missing required 'path' property.");
			String path = anim_target["path"];
			if (path == "pointer") {
				ERR_FAIL_COND_V(!anim_target.has("extensions"), ERR_PARSE_ERROR);
				Dictionary target_extensions = anim_target["extensions"];
				ERR_FAIL_COND_V(!target_extensions.has("KHR_animation_pointer"), ERR_PARSE_ERROR);
				Dictionary khr_anim_ptr = target_extensions["KHR_animation_pointer"];
				ERR_FAIL_COND_V(!khr_anim_ptr.has("pointer"), ERR_PARSE_ERROR);
				String anim_json_ptr = khr_anim_ptr["pointer"];
				_parse_animation_pointer(p_state, anim_json_ptr, animation, interp, times, output_value_accessor_index);
			} else {
				// If it's not a pointer, it's a regular animation channel from vanilla glTF (pos/rot/scale/weights).
				if (!anim_target.has("node")) {
					WARN_PRINT("glTF: Animation channel target missing 'node' property. Ignoring this channel.");
					continue;
				}

				GLTFNodeIndex node = anim_target["node"];

				ERR_FAIL_INDEX_V(node, p_state->nodes.size(), ERR_PARSE_ERROR);

				GLTFAnimation::NodeTrack *track = nullptr;

				if (!animation->get_node_tracks().has(node)) {
					animation->get_node_tracks()[node] = GLTFAnimation::NodeTrack();
				}

				track = &animation->get_node_tracks()[node];

				if (path == "translation") {
					const Vector<Vector3> positions = _decode_accessor_as_vec3(p_state, output_value_accessor_index);
					track->position_track.interpolation = interp;
					track->position_track.times = times;
					track->position_track.values = positions;
				} else if (path == "rotation") {
					const Vector<Quaternion> rotations = _decode_accessor_as_quaternion(p_state, output_value_accessor_index);
					track->rotation_track.interpolation = interp;
					track->rotation_track.times = times;
					track->rotation_track.values = rotations;
				} else if (path == "scale") {
					const Vector<Vector3> scales = _decode_accessor_as_vec3(p_state, output_value_accessor_index);
					track->scale_track.interpolation = interp;
					track->scale_track.times = times;
					track->scale_track.values = scales;
				} else if (path == "weights") {
					const Vector<float> weights = _decode_accessor_as_float32s(p_state, output_value_accessor_index);

					ERR_FAIL_INDEX_V(p_state->nodes[node]->mesh, p_state->meshes.size(), ERR_PARSE_ERROR);
					Ref<GLTFMesh> mesh = p_state->meshes[p_state->nodes[node]->mesh];
					const int wc = mesh->get_blend_weights().size();
					ERR_CONTINUE_MSG(wc == 0, "glTF: Animation tried to animate weights, but mesh has no weights.");

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
		}

		p_state->animations.push_back(animation);
	}

	print_verbose("glTF: Total animations '" + itos(p_state->animations.size()) + "'.");

	return OK;
}

void GLTFDocument::_parse_animation_pointer(Ref<GLTFState> p_state, const String &p_animation_json_pointer, const Ref<GLTFAnimation> p_gltf_animation, const GLTFAnimation::Interpolation p_interp, const Vector<double> &p_times, const int p_output_value_accessor_index) {
	// Special case: Convert TRS animation pointers to node track pos/rot/scale.
	// This is required to handle skeleton bones, and improves performance for regular nodes.
	// Mark this as unlikely because TRS animation pointers are not recommended,
	// since vanilla glTF animations can already animate TRS properties directly.
	// But having this code exist is required to be spec-compliant and handle all test files.
	// Note that TRS still needs to be handled in the general case as well, for KHR_interactivity.
	const PackedStringArray split = p_animation_json_pointer.split("/", false, 3);
	if (unlikely(split.size() == 3 && split[0] == "nodes" && (split[2] == "translation" || split[2] == "rotation" || split[2] == "scale" || split[2] == "matrix" || split[2] == "weights"))) {
		const GLTFNodeIndex node_index = split[1].to_int();
		HashMap<int, GLTFAnimation::NodeTrack> &node_tracks = p_gltf_animation->get_node_tracks();
		if (!node_tracks.has(node_index)) {
			node_tracks[node_index] = GLTFAnimation::NodeTrack();
		}
		GLTFAnimation::NodeTrack *track = &node_tracks[node_index];
		if (split[2] == "translation") {
			const Vector<Vector3> positions = _decode_accessor_as_vec3(p_state, p_output_value_accessor_index);
			track->position_track.interpolation = p_interp;
			track->position_track.times = p_times;
			track->position_track.values = positions;
		} else if (split[2] == "rotation") {
			const Vector<Quaternion> rotations = _decode_accessor_as_quaternion(p_state, p_output_value_accessor_index);
			track->rotation_track.interpolation = p_interp;
			track->rotation_track.times = p_times;
			track->rotation_track.values = rotations;
		} else if (split[2] == "scale") {
			const Vector<Vector3> scales = _decode_accessor_as_vec3(p_state, p_output_value_accessor_index);
			track->scale_track.interpolation = p_interp;
			track->scale_track.times = p_times;
			track->scale_track.values = scales;
		} else if (split[2] == "matrix") {
			Array transforms = _decode_accessor_as_variants(p_state, p_output_value_accessor_index, Variant::TRANSFORM3D);
			track->position_track.interpolation = p_interp;
			track->position_track.times = p_times;
			track->position_track.values.resize(transforms.size());
			track->rotation_track.interpolation = p_interp;
			track->rotation_track.times = p_times;
			track->rotation_track.values.resize(transforms.size());
			track->scale_track.interpolation = p_interp;
			track->scale_track.times = p_times;
			track->scale_track.values.resize(transforms.size());
			for (int i = 0; i < transforms.size(); i++) {
				Transform3D transform = transforms[i];
				track->position_track.values.write[i] = transform.get_origin();
				track->rotation_track.values.write[i] = transform.basis.get_rotation_quaternion();
				track->scale_track.values.write[i] = transform.basis.get_scale();
			}
		} else { // if (split[2] == "weights")
			const Vector<float> accessor_weights = _decode_accessor_as_float32s(p_state, p_output_value_accessor_index);
			const GLTFMeshIndex mesh_index = p_state->nodes[node_index]->mesh;
			ERR_FAIL_INDEX(mesh_index, p_state->meshes.size());
			const Ref<GLTFMesh> gltf_mesh = p_state->meshes[mesh_index];
			const Vector<float> &blend_weights = gltf_mesh->get_blend_weights();
			const int blend_weight_count = gltf_mesh->get_blend_weights().size();
			const int anim_weights_size = accessor_weights.size();
			// For example, if a mesh has 2 blend weights, and the accessor provides 10 values, then there are 5 frames of animation, each with 2 blend weights.
			ERR_FAIL_COND_MSG(blend_weight_count == 0 || ((anim_weights_size % blend_weight_count) != 0), "glTF: Cannot apply " + itos(accessor_weights.size()) + " weights to a mesh with " + itos(blend_weights.size()) + " blend weights.");
			const int frame_count = anim_weights_size / blend_weight_count;
			track->weight_tracks.resize(blend_weight_count);
			for (int blend_weight_index = 0; blend_weight_index < blend_weight_count; blend_weight_index++) {
				GLTFAnimation::Channel<real_t> weight_track;
				weight_track.interpolation = p_interp;
				weight_track.times = p_times;
				weight_track.values.resize(frame_count);
				for (int frame_index = 0; frame_index < frame_count; frame_index++) {
					// For example, if a mesh has 2 blend weights, and the accessor provides 10 values,
					// then the first frame has indices [0, 1], the second frame has [2, 3], and so on.
					// Here we process all frames of one blend weight, so we want [0, 2, 4, 6, 8] or [1, 3, 5, 7, 9].
					// For the fist one we calculate 0 * 2 + 0, 1 * 2 + 0, 2 * 2 + 0, etc, then for the second 0 * 2 + 1, 1 * 2 + 1, 2 * 2 + 1, etc.
					weight_track.values.write[frame_index] = accessor_weights[frame_index * blend_weight_count + blend_weight_index];
				}
				track->weight_tracks.write[blend_weight_index] = weight_track;
			}
		}
		// The special case was handled, return to skip the general case.
		return;
	}
	// General case: Convert animation pointers to Variant value pointer tracks.
	Ref<GLTFObjectModelProperty> obj_model_prop = import_object_model_property(p_state, p_animation_json_pointer);
	if (obj_model_prop.is_null() || !obj_model_prop->has_node_paths()) {
		// Exit quietly, `import_object_model_property` already prints a warning if the property is not found.
		return;
	}
	HashMap<String, GLTFAnimation::Channel<Variant>> &anim_ptr_map = p_gltf_animation->get_pointer_tracks();
	GLTFAnimation::Channel<Variant> channel;
	channel.interpolation = p_interp;
	channel.times = p_times;
	Array values_arr = _decode_accessor_as_variants(p_state, p_output_value_accessor_index, obj_model_prop->get_variant_type());
	// TODO: This can be made faster after this pull request is merged: https://github.com/godotengine/godot/pull/109003
	GLTFTemplateConvert::set_from_array(channel.values, values_arr);
	anim_ptr_map[p_animation_json_pointer] = channel;
}

void GLTFDocument::_assign_node_names(Ref<GLTFState> p_state) {
	for (int i = 0; i < p_state->nodes.size(); i++) {
		Ref<GLTFNode> gltf_node = p_state->nodes[i];
		// Any joints get unique names generated when the skeleton is made, unique to the skeleton
		if (gltf_node->skeleton >= 0) {
			continue;
		}
		String gltf_node_name = gltf_node->get_name();
		if (gltf_node_name.is_empty()) {
			if (_naming_version == 0) {
				if (gltf_node->mesh >= 0) {
					gltf_node_name = _gen_unique_name(p_state, "Mesh");
				} else if (gltf_node->camera >= 0) {
					gltf_node_name = _gen_unique_name(p_state, "Camera3D");
				} else {
					gltf_node_name = _gen_unique_name(p_state, "Node");
				}
			} else {
				if (gltf_node->mesh >= 0) {
					gltf_node_name = "Mesh";
				} else if (gltf_node->camera >= 0) {
					gltf_node_name = "Camera";
				} else {
					gltf_node_name = "Node";
				}
			}
		}
		gltf_node->set_name(_gen_unique_name(p_state, gltf_node_name));
	}
}

BoneAttachment3D *GLTFDocument::_generate_bone_attachment(Skeleton3D *p_godot_skeleton, const Ref<GLTFNode> &p_bone_node) {
	BoneAttachment3D *bone_attachment = memnew(BoneAttachment3D);
	print_verbose("glTF: Creating bone attachment for: " + p_bone_node->get_name());
	bone_attachment->set_name(p_bone_node->get_name());
	p_godot_skeleton->add_child(bone_attachment, true);
	bone_attachment->set_bone_name(p_bone_node->get_name());
	return bone_attachment;
}

BoneAttachment3D *GLTFDocument::_generate_bone_attachment_compat_4pt4(Ref<GLTFState> p_state, Skeleton3D *p_skeleton, const GLTFNodeIndex p_node_index, const GLTFNodeIndex p_bone_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];
	Ref<GLTFNode> bone_node = p_state->nodes[p_bone_index];
	BoneAttachment3D *bone_attachment = memnew(BoneAttachment3D);
	print_verbose("glTF: Creating bone attachment for: " + gltf_node->get_name());

	ERR_FAIL_COND_V(!bone_node->joint, nullptr);

	bone_attachment->set_bone_name(bone_node->get_name());

	return bone_attachment;
}

GLTFMeshIndex GLTFDocument::_convert_mesh_to_gltf(Ref<GLTFState> p_state, MeshInstance3D *p_mesh_instance) {
	ERR_FAIL_NULL_V(p_mesh_instance, -1);
	ERR_FAIL_COND_V_MSG(p_mesh_instance->get_mesh().is_null(), -1, "glTF: Tried to export a MeshInstance3D node named " + p_mesh_instance->get_name() + ", but it has no mesh. This node will be exported without a mesh.");
	Ref<Mesh> mesh_resource = p_mesh_instance->get_mesh();
	ERR_FAIL_COND_V_MSG(mesh_resource->get_surface_count() == 0, -1, "glTF: Tried to export a MeshInstance3D node named " + p_mesh_instance->get_name() + ", but its mesh has no surfaces. This node will be exported without a mesh.");
	TypedArray<Material> instance_materials;
	for (int32_t surface_i = 0; surface_i < mesh_resource->get_surface_count(); surface_i++) {
		Ref<Material> mat = p_mesh_instance->get_active_material(surface_i);
		instance_materials.append(mat);
	}
	Vector<float> blend_weights;
	int32_t blend_count = mesh_resource->get_blend_shape_count();
	blend_weights.resize(blend_count);
	for (int32_t blend_i = 0; blend_i < blend_count; blend_i++) {
		blend_weights.write[blend_i] = 0.0f;
	}

	Ref<GLTFMesh> gltf_mesh;
	gltf_mesh.instantiate();
	if (!mesh_resource->get_name().is_empty()) {
		gltf_mesh->set_original_name(mesh_resource->get_name());
		gltf_mesh->set_name(_gen_unique_name(p_state, mesh_resource->get_name()));
	}
	gltf_mesh->set_instance_materials(instance_materials);
	gltf_mesh->set_mesh(ImporterMesh::from_mesh(mesh_resource));
	gltf_mesh->set_blend_weights(blend_weights);
	GLTFMeshIndex mesh_i = p_state->meshes.size();
	p_state->meshes.push_back(gltf_mesh);
	return mesh_i;
}

ImporterMeshInstance3D *GLTFDocument::_generate_mesh_instance(Ref<GLTFState> p_state, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	ERR_FAIL_INDEX_V(gltf_node->mesh, p_state->meshes.size(), nullptr);

	ImporterMeshInstance3D *mi = memnew(ImporterMeshInstance3D);
	print_verbose("glTF: Creating mesh for: " + gltf_node->get_name());

	p_state->scene_mesh_instances.insert(p_node_index, mi);
	Ref<GLTFMesh> mesh = p_state->meshes.write[gltf_node->mesh];
	if (mesh.is_null()) {
		return mi;
	}
	Ref<ImporterMesh> import_mesh = mesh->get_mesh();
	if (import_mesh.is_null()) {
		return mi;
	}
	mi->set_mesh(import_mesh);
	import_mesh->merge_meta_from(*mesh);
	return mi;
}

Light3D *GLTFDocument::_generate_light(Ref<GLTFState> p_state, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	ERR_FAIL_INDEX_V(gltf_node->light, p_state->lights.size(), nullptr);

	print_verbose("glTF: Creating light for: " + gltf_node->get_name());

	Ref<GLTFLight> l = p_state->lights[gltf_node->light];
	return l->to_node();
}

Camera3D *GLTFDocument::_generate_camera(Ref<GLTFState> p_state, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	ERR_FAIL_INDEX_V(gltf_node->camera, p_state->cameras.size(), nullptr);

	print_verbose("glTF: Creating camera for: " + gltf_node->get_name());

	Ref<GLTFCamera> c = p_state->cameras[gltf_node->camera];
	return c->to_node();
}

GLTFCameraIndex GLTFDocument::_convert_camera(Ref<GLTFState> p_state, Camera3D *p_camera) {
	print_verbose("glTF: Converting camera: " + p_camera->get_name());

	Ref<GLTFCamera> c = GLTFCamera::from_node(p_camera);
	GLTFCameraIndex camera_index = p_state->cameras.size();
	p_state->cameras.push_back(c);
	return camera_index;
}

GLTFLightIndex GLTFDocument::_convert_light(Ref<GLTFState> p_state, Light3D *p_light) {
	print_verbose("glTF: Converting light: " + p_light->get_name());

	Ref<GLTFLight> l = GLTFLight::from_node(p_light);

	GLTFLightIndex light_index = p_state->lights.size();
	p_state->lights.push_back(l);
	return light_index;
}

void GLTFDocument::_convert_spatial(Ref<GLTFState> p_state, Node3D *p_spatial, Ref<GLTFNode> p_node) {
	p_node->transform = p_spatial->get_transform();
}

Node3D *GLTFDocument::_generate_spatial(Ref<GLTFState> p_state, const GLTFNodeIndex p_node_index) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	Node3D *spatial = memnew(Node3D);
	print_verbose("glTF: Converting spatial: " + gltf_node->get_name());

	return spatial;
}

void GLTFDocument::_convert_scene_node(Ref<GLTFState> p_state, Node *p_current, const GLTFNodeIndex p_gltf_parent, const GLTFNodeIndex p_gltf_root) {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && p_gltf_root != -1 && p_current->get_owner() == nullptr) {
		WARN_VERBOSE("glTF export warning: Node '" + p_current->get_name() + "' has no owner. This is likely a temporary node generated by a @tool script. This would not be saved when saving the Godot scene, therefore it will not be exported to glTF.");
		return;
	}
#endif // TOOLS_ENABLED
	Ref<GLTFNode> gltf_node;
	gltf_node.instantiate();
	if (p_current->has_method("is_visible")) {
		bool visible = p_current->call("is_visible");
		if (!visible && _visibility_mode == VISIBILITY_MODE_EXCLUDE) {
			return;
		}
		gltf_node->visible = visible;
	}
	gltf_node->set_original_name(p_current->get_name());
	gltf_node->set_name(_gen_unique_name(p_state, p_current->get_name()));
	gltf_node->merge_meta_from(p_current);
	if (Object::cast_to<Node3D>(p_current)) {
		Node3D *spatial = Object::cast_to<Node3D>(p_current);
		_convert_spatial(p_state, spatial, gltf_node);
	}
	if (Object::cast_to<MeshInstance3D>(p_current)) {
		MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(p_current);
		_convert_mesh_instance_to_gltf(mi, p_state, gltf_node);
	} else if (Object::cast_to<BoneAttachment3D>(p_current)) {
		BoneAttachment3D *bone = Object::cast_to<BoneAttachment3D>(p_current);
		_convert_bone_attachment_to_gltf(bone, p_state, p_gltf_parent, p_gltf_root, gltf_node);
		return;
	} else if (Object::cast_to<Skeleton3D>(p_current)) {
		Skeleton3D *skel = Object::cast_to<Skeleton3D>(p_current);
		_convert_skeleton_to_gltf(skel, p_state, p_gltf_parent, p_gltf_root, gltf_node);
		// We ignore the Godot Engine node that is the skeleton.
		return;
	} else if (Object::cast_to<MultiMeshInstance3D>(p_current)) {
		MultiMeshInstance3D *multi = Object::cast_to<MultiMeshInstance3D>(p_current);
		_convert_multi_mesh_instance_to_gltf(multi, p_gltf_parent, p_gltf_root, gltf_node, p_state);
#ifdef MODULE_CSG_ENABLED
	} else if (Object::cast_to<CSGShape3D>(p_current)) {
		CSGShape3D *shape = Object::cast_to<CSGShape3D>(p_current);
		if (shape->get_parent() && shape->is_root_shape()) {
			_convert_csg_shape_to_gltf(shape, p_gltf_parent, gltf_node, p_state);
		}
#endif // MODULE_CSG_ENABLED
#ifdef MODULE_GRIDMAP_ENABLED
	} else if (Object::cast_to<GridMap>(p_current)) {
		GridMap *gridmap = Object::cast_to<GridMap>(p_current);
		_convert_grid_map_to_gltf(gridmap, p_gltf_parent, p_gltf_root, gltf_node, p_state);
#endif // MODULE_GRIDMAP_ENABLED
	} else if (Object::cast_to<Camera3D>(p_current)) {
		Camera3D *camera = Object::cast_to<Camera3D>(p_current);
		_convert_camera_to_gltf(camera, p_state, gltf_node);
	} else if (Object::cast_to<Light3D>(p_current)) {
		Light3D *light = Object::cast_to<Light3D>(p_current);
		_convert_light_to_gltf(light, p_state, gltf_node);
	} else if (Object::cast_to<AnimationPlayer>(p_current)) {
		AnimationPlayer *animation_player = Object::cast_to<AnimationPlayer>(p_current);
		p_state->animation_players.push_back(animation_player);
		if (animation_player->get_child_count() == 0) {
			gltf_node->set_parent(-2); // Don't export AnimationPlayer nodes as glTF nodes (unless they have children).
		}
	}
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		ext->convert_scene_node(p_state, gltf_node, p_current);
	}
	GLTFNodeIndex current_node_i;
	if (gltf_node->get_parent() == -1) {
		current_node_i = p_state->append_gltf_node(gltf_node, p_current, p_gltf_parent);
	} else if (gltf_node->get_parent() < -1) {
		return;
	} else {
		current_node_i = p_state->nodes.size() - 1;
		while (gltf_node != p_state->nodes[current_node_i]) {
			current_node_i--;
		}
	}
	const GLTFNodeIndex gltf_root = (p_gltf_root == -1) ? current_node_i : p_gltf_root;
	for (int node_i = 0; node_i < p_current->get_child_count(); node_i++) {
		_convert_scene_node(p_state, p_current->get_child(node_i), current_node_i, gltf_root);
	}
}

void GLTFDocument::_convert_csg_shape_to_gltf(CSGShape3D *p_current, GLTFNodeIndex p_gltf_parent, Ref<GLTFNode> p_gltf_node, Ref<GLTFState> p_state) {
#ifndef MODULE_CSG_ENABLED
	ERR_FAIL_MSG("csg module is disabled.");
#else
	CSGShape3D *csg = p_current;
	csg->update_shape();
	Array meshes = csg->get_meshes();
	if (meshes.size() != 2) {
		return;
	}

	Ref<ImporterMesh> mesh;
	mesh.instantiate();
	{
		Ref<ArrayMesh> csg_mesh = csg->get_meshes()[1];
		for (int32_t surface_i = 0; surface_i < csg_mesh->get_surface_count(); surface_i++) {
			Array array = csg_mesh->surface_get_arrays(surface_i);

			Ref<Material> mat;

			Ref<Material> mat_override = csg->get_material_override();
			if (mat_override.is_valid()) {
				mat = mat_override;
			}

			Ref<Material> mat_surface_override = csg_mesh->surface_get_material(surface_i);
			if (mat_surface_override.is_valid() && mat.is_null()) {
				mat = mat_surface_override;
			}

			String mat_name;
			if (mat.is_valid()) {
				mat_name = mat->get_name();
			} else {
				// Assign default material when no material is assigned.
				mat.instantiate();
			}

			mesh->add_surface(csg_mesh->surface_get_primitive_type(surface_i),
					array, csg_mesh->surface_get_blend_shape_arrays(surface_i), csg_mesh->surface_get_lods(surface_i), mat,
					mat_name, csg_mesh->surface_get_format(surface_i));
		}
	}

	Ref<GLTFMesh> gltf_mesh;
	gltf_mesh.instantiate();
	gltf_mesh->set_mesh(mesh);
	gltf_mesh->set_original_name(csg->get_name());
	const String unique_name = _gen_unique_name(p_state, csg->get_name());
	gltf_mesh->set_name(unique_name);
	GLTFMeshIndex mesh_i = p_state->meshes.size();
	p_state->meshes.push_back(gltf_mesh);
	p_gltf_node->mesh = mesh_i;
	p_gltf_node->transform = csg->get_transform();
	p_gltf_node->set_original_name(csg->get_name());
	p_gltf_node->set_name(unique_name);
#endif // MODULE_CSG_ENABLED
}

void GLTFDocument::_convert_camera_to_gltf(Camera3D *camera, Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node) {
	ERR_FAIL_NULL(camera);
	GLTFCameraIndex camera_index = _convert_camera(p_state, camera);
	if (camera_index != -1) {
		p_gltf_node->camera = camera_index;
	}
}

void GLTFDocument::_convert_light_to_gltf(Light3D *light, Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node) {
	ERR_FAIL_NULL(light);
	GLTFLightIndex light_index = _convert_light(p_state, light);
	if (light_index != -1) {
		p_gltf_node->light = light_index;
	}
}

void GLTFDocument::_convert_grid_map_to_gltf(GridMap *p_grid_map, GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_node_index, Ref<GLTFNode> p_gltf_node, Ref<GLTFState> p_state) {
#ifndef MODULE_GRIDMAP_ENABLED
	ERR_FAIL_MSG("gridmap module is disabled.");
#else
	const Array &cells = p_grid_map->get_used_cells();
	for (int32_t k = 0; k < cells.size(); k++) {
		GLTFNode *new_gltf_node = memnew(GLTFNode);
		p_gltf_node->children.push_back(p_state->nodes.size());
		p_state->nodes.push_back(new_gltf_node);
		Vector3 cell_location = cells[k];
		int32_t cell = p_grid_map->get_cell_item(
				Vector3(cell_location.x, cell_location.y, cell_location.z));
		Transform3D cell_xform;
		cell_xform.basis = p_grid_map->get_basis_with_orthogonal_index(
				p_grid_map->get_cell_item_orientation(
						Vector3(cell_location.x, cell_location.y, cell_location.z)));
		cell_xform.basis.scale(Vector3(p_grid_map->get_cell_scale(),
				p_grid_map->get_cell_scale(),
				p_grid_map->get_cell_scale()));
		cell_xform.set_origin(p_grid_map->map_to_local(
				Vector3(cell_location.x, cell_location.y, cell_location.z)));
		Ref<GLTFMesh> gltf_mesh;
		gltf_mesh.instantiate();
		gltf_mesh->set_mesh(ImporterMesh::from_mesh(p_grid_map->get_mesh_library()->get_item_mesh(cell)));
		gltf_mesh->set_original_name(p_grid_map->get_mesh_library()->get_item_name(cell));
		const String unique_name = _gen_unique_name(p_state, p_grid_map->get_mesh_library()->get_item_name(cell));
		gltf_mesh->set_name(unique_name);
		new_gltf_node->mesh = p_state->meshes.size();
		p_state->meshes.push_back(gltf_mesh);
		new_gltf_node->transform = cell_xform * p_grid_map->get_transform();
		new_gltf_node->set_original_name(p_grid_map->get_mesh_library()->get_item_name(cell));
		new_gltf_node->set_name(unique_name);
	}
#endif // MODULE_GRIDMAP_ENABLED
}

void GLTFDocument::_convert_multi_mesh_instance_to_gltf(
		MultiMeshInstance3D *p_multi_mesh_instance,
		GLTFNodeIndex p_parent_node_index,
		GLTFNodeIndex p_root_node_index,
		Ref<GLTFNode> p_gltf_node, Ref<GLTFState> p_state) {
	ERR_FAIL_NULL(p_multi_mesh_instance);
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
	gltf_mesh->set_original_name(multi_mesh->get_name());
	gltf_mesh->set_name(multi_mesh->get_name());
	gltf_mesh->set_mesh(ImporterMesh::from_mesh(mesh));
	GLTFMeshIndex mesh_index = p_state->meshes.size();
	p_state->meshes.push_back(gltf_mesh);
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
		new_gltf_node->transform = transform;
		new_gltf_node->set_original_name(p_multi_mesh_instance->get_name());
		new_gltf_node->set_name(_gen_unique_name(p_state, p_multi_mesh_instance->get_name()));
		p_gltf_node->children.push_back(p_state->nodes.size());
		p_state->nodes.push_back(new_gltf_node);
	}
}

void GLTFDocument::_convert_skeleton_to_gltf(Skeleton3D *p_skeleton3d, Ref<GLTFState> p_state, GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_node_index, Ref<GLTFNode> p_gltf_node) {
	Skeleton3D *skeleton = p_skeleton3d;
	Ref<GLTFSkeleton> gltf_skeleton;
	gltf_skeleton.instantiate();
	// GLTFSkeleton is only used to hold internal p_state data. It will not be written to the document.
	//
	gltf_skeleton->godot_skeleton = skeleton;
	GLTFSkeletonIndex skeleton_i = p_state->skeletons.size();
	p_state->skeleton3d_to_gltf_skeleton[skeleton->get_instance_id()] = skeleton_i;
	p_state->skeletons.push_back(gltf_skeleton);

	BoneId bone_count = skeleton->get_bone_count();
	for (BoneId bone_i = 0; bone_i < bone_count; bone_i++) {
		Ref<GLTFNode> joint_node;
		joint_node.instantiate();
		// Note that we cannot use _gen_unique_bone_name here, because glTF spec requires all node
		// names to be unique regardless of whether or not they are used as joints.
		joint_node->set_original_name(skeleton->get_bone_name(bone_i));
		joint_node->set_name(_gen_unique_name(p_state, skeleton->get_bone_name(bone_i)));
		joint_node->transform = skeleton->get_bone_pose(bone_i);
		joint_node->joint = true;

		if (p_skeleton3d->has_bone_meta(bone_i, "extras")) {
			joint_node->set_meta("extras", p_skeleton3d->get_bone_meta(bone_i, "extras"));
		}
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

void GLTFDocument::_convert_bone_attachment_to_gltf(BoneAttachment3D *p_bone_attachment, Ref<GLTFState> p_state, GLTFNodeIndex p_parent_node_index, GLTFNodeIndex p_root_node_index, Ref<GLTFNode> p_gltf_node) {
	Skeleton3D *skeleton;
	// Note that relative transforms to external skeletons and pose overrides are not supported.
	if (p_bone_attachment->get_use_external_skeleton()) {
		skeleton = Object::cast_to<Skeleton3D>(p_bone_attachment->get_node_or_null(p_bone_attachment->get_external_skeleton()));
	} else {
		skeleton = Object::cast_to<Skeleton3D>(p_bone_attachment->get_parent());
	}
	GLTFSkeletonIndex skel_gltf_i = -1;
	if (skeleton != nullptr && p_state->skeleton3d_to_gltf_skeleton.has(skeleton->get_instance_id())) {
		skel_gltf_i = p_state->skeleton3d_to_gltf_skeleton[skeleton->get_instance_id()];
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
		Ref<GLTFSkeleton> gltf_skeleton = p_state->skeletons.write[skel_gltf_i];
		gltf_skeleton->bone_attachments.push_back(p_bone_attachment);
		par_node_index = gltf_skeleton->joints[bone_idx];
	}

	for (int node_i = 0; node_i < p_bone_attachment->get_child_count(); node_i++) {
		_convert_scene_node(p_state, p_bone_attachment->get_child(node_i), par_node_index, p_root_node_index);
	}
}

void GLTFDocument::_convert_mesh_instance_to_gltf(MeshInstance3D *p_scene_parent, Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node) {
	GLTFMeshIndex gltf_mesh_index = _convert_mesh_to_gltf(p_state, p_scene_parent);
	if (gltf_mesh_index != -1) {
		p_gltf_node->mesh = gltf_mesh_index;
	}
}

void _set_node_tree_owner(Node *p_current_node, Node *&p_scene_root) {
	// Note: p_scene_parent and p_scene_root must either both be null or both be valid.
	if (p_scene_root == nullptr) {
		// If the root node argument is null, this is the root node.
		p_scene_root = p_current_node;
		// If multiple nodes were generated under the root node, ensure they have the owner set.
		if (unlikely(p_current_node->get_child_count() > 0)) {
			Array args;
			args.append(p_scene_root);
			for (int i = 0; i < p_current_node->get_child_count(); i++) {
				Node *child = p_current_node->get_child(i);
				child->propagate_call(StringName("set_owner"), args);
			}
		}
	} else {
		// Add the node we generated and set the owner to the scene root.
		Array args;
		args.append(p_scene_root);
		p_current_node->propagate_call(StringName("set_owner"), args);
	}
}

bool GLTFDocument::_does_skinned_mesh_require_placeholder_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node) {
	if (p_gltf_node->skin < 0) {
		return false; // Not a skinned mesh.
	}
	// Check for child nodes that aren't joints/bones.
	for (int i = 0; i < p_gltf_node->children.size(); ++i) {
		Ref<GLTFNode> child = p_state->nodes[p_gltf_node->children[i]];
		if (!child->joint) {
			return true;
		}
		// Edge case: If a child's skeleton is not yet in the tree, then we must add it as a child of this node.
		// While the Skeleton3D node isn't a glTF node, it's still a case where we need a placeholder.
		// This is required to handle this issue: https://github.com/godotengine/godot/issues/67773
		const GLTFSkeletonIndex skel_index = child->skeleton;
		ERR_FAIL_INDEX_V(skel_index, p_state->skeletons.size(), false);
		if (p_state->skeletons[skel_index]->godot_skeleton->get_parent() == nullptr) {
			return true;
		}
	}
	return false;
}

void GLTFDocument::_generate_scene_node(Ref<GLTFState> p_state, const GLTFNodeIndex p_node_index, Node *p_scene_parent, Node *p_scene_root) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];
	Node3D *current_node = nullptr;
	// Check if any GLTFDocumentExtension classes want to generate a node for us.
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		current_node = ext->generate_scene_node(p_state, gltf_node, p_scene_parent);
		if (current_node) {
			break;
		}
	}
	// If none of our GLTFDocumentExtension classes generated us a node, try using built-in glTF types.
	if (!current_node) {
		if (gltf_node->mesh >= 0) {
			current_node = _generate_mesh_instance(p_state, p_node_index);
			// glTF specifies that skinned meshes should ignore their node transforms,
			// only being controlled by the skeleton, so Godot will reparent a skinned
			// mesh to its skeleton. However, we still need to ensure any child nodes
			// keep their place in the tree, so if there are any child nodes, the skinned
			// mesh must not be the base node, so generate an empty spatial base.
			if (_does_skinned_mesh_require_placeholder_node(p_state, gltf_node)) {
				Node3D *placeholder;
				// We need a placeholder, but maybe the Skeleton3D *is* the placeholder?
				const GLTFSkeletonIndex skel_index = gltf_node->skeleton;
				if (skel_index >= 0 && skel_index < p_state->skeletons.size() && p_state->skeletons[skel_index]->godot_skeleton->get_parent() == nullptr) {
					placeholder = p_state->skeletons[skel_index]->godot_skeleton;
				} else {
					placeholder = _generate_spatial(p_state, p_node_index);
				}
				current_node->set_name(gltf_node->get_name());
				placeholder->add_child(current_node, true);
				current_node = placeholder;
			}
		} else if (gltf_node->camera >= 0) {
			current_node = _generate_camera(p_state, p_node_index);
		} else if (gltf_node->light >= 0) {
			current_node = _generate_light(p_state, p_node_index);
		}
	}
	// The only case where current_node is a Skeleton3D is when it is the placeholder for a skinned mesh.
	// In that case, we don't set the name or possibly generate a bone attachment. But usually, we do.
	// It is also possible that user code generates a Skeleton3D node, and this code also works for that case.
	if (likely(!Object::cast_to<Skeleton3D>(current_node))) {
		if (current_node) {
			// Set the name of the Godot node to the name of the glTF node.
			String gltf_node_name = gltf_node->get_name();
			if (!gltf_node_name.is_empty()) {
				current_node->set_name(gltf_node_name);
			}
		}
		// Skeleton stuff: If this node is in a skeleton, we need to attach it to a bone attachment pointing to its bone.
		if (gltf_node->skeleton >= 0) {
			_generate_skeleton_bone_node(p_state, p_node_index, current_node, p_scene_parent, p_scene_root);
			return;
		}
	}
	// Skeleton stuff: If the parent node is in a skeleton, we need to attach this node to a bone attachment pointing to the parent's bone.
	if (Object::cast_to<Skeleton3D>(p_scene_parent)) {
		Skeleton3D *parent_skeleton = Object::cast_to<Skeleton3D>(p_scene_parent);
		_attach_node_to_skeleton(p_state, p_node_index, current_node, parent_skeleton, p_scene_root);
		return;
	}
	// Not a skeleton bone, so definitely some kind of node that goes in the Godot scene tree.
	if (current_node == nullptr) {
		current_node = _generate_spatial(p_state, p_node_index);
		// Set the name of the Godot node to the name of the glTF node.
		String gltf_node_name = gltf_node->get_name();
		if (!gltf_node_name.is_empty()) {
			current_node->set_name(gltf_node_name);
		}
	}
	if (p_scene_parent) {
		p_scene_parent->add_child(current_node, true);
	}
	// Set the owner of the nodes to the scene root.
	// Note: p_scene_parent and p_scene_root must either both be null or both be valid.
	_set_node_tree_owner(current_node, p_scene_root);
	current_node->set_transform(gltf_node->transform);
	current_node->set_visible(gltf_node->visible);
	current_node->merge_meta_from(*gltf_node);
	p_state->scene_nodes.insert(p_node_index, current_node);
	for (int i = 0; i < gltf_node->children.size(); ++i) {
		_generate_scene_node(p_state, gltf_node->children[i], current_node, p_scene_root);
	}
}

void GLTFDocument::_generate_skeleton_bone_node(Ref<GLTFState> p_state, const GLTFNodeIndex p_node_index, Node3D *p_current_node, Node *p_scene_parent, Node *p_scene_root) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];
	// Grab the current skeleton, and ensure it's added to the tree.
	Skeleton3D *godot_skeleton = p_state->skeletons[gltf_node->skeleton]->godot_skeleton;
	if (godot_skeleton->get_parent() == nullptr) {
		if (p_scene_root) {
			if (Object::cast_to<Skeleton3D>(p_scene_parent)) {
				Skeleton3D *parent_skeleton = Object::cast_to<Skeleton3D>(p_scene_parent);
				// Explicitly specifying the bone of the parent glTF node is required to
				// handle the edge case where a skeleton is a child of another skeleton.
				_attach_node_to_skeleton(p_state, p_node_index, godot_skeleton, parent_skeleton, p_scene_root, gltf_node->parent);
			} else {
				p_scene_parent->add_child(godot_skeleton, true);
				godot_skeleton->set_owner(p_scene_root);
			}
		} else {
			p_scene_root = godot_skeleton;
		}
	}
	_attach_node_to_skeleton(p_state, p_node_index, p_current_node, godot_skeleton, p_scene_root);
}

void GLTFDocument::_attach_node_to_skeleton(Ref<GLTFState> p_state, const GLTFNodeIndex p_node_index, Node3D *p_current_node, Skeleton3D *p_godot_skeleton, Node *p_scene_root, GLTFNodeIndex p_bone_node_index) {
	ERR_FAIL_NULL(p_godot_skeleton->get_parent());
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];
	if (Object::cast_to<ImporterMeshInstance3D>(p_current_node) && gltf_node->skin >= 0) {
		// Skinned meshes should be attached directly to the skeleton without a BoneAttachment3D.
		ERR_FAIL_COND_MSG(p_current_node->get_child_count() > 0, "Skinned mesh nodes passed to this function should not have children (a placeholder should be inserted by `_generate_scene_node`).");
		p_godot_skeleton->add_child(p_current_node, true);
	} else if (p_current_node || !gltf_node->joint) {
		// If we have a node in need of attaching, we need a BoneAttachment3D.
		// This happens when a node in Blender has Relations -> Parent set to a bone.
		GLTFNodeIndex attachment_node_index = likely(p_bone_node_index == -1) ? (gltf_node->joint ? p_node_index : gltf_node->parent) : p_bone_node_index;
		ERR_FAIL_COND(!p_state->scene_nodes.has(attachment_node_index));
		Node *attachment_godot_node = p_state->scene_nodes[attachment_node_index];
		// If the parent is a Skeleton3D, we need to make a BoneAttachment3D.
		if (Object::cast_to<Skeleton3D>(attachment_godot_node)) {
			Ref<GLTFNode> attachment_gltf_node = p_state->nodes[attachment_node_index];
			BoneAttachment3D *bone_attachment = _generate_bone_attachment(p_godot_skeleton, attachment_gltf_node);
			bone_attachment->set_owner(p_scene_root);
			bone_attachment->merge_meta_from(*p_state->nodes[attachment_node_index]);
			p_state->scene_nodes.insert(attachment_node_index, bone_attachment);
			attachment_godot_node = bone_attachment;
		}
		// By this point, `attachment_godot_node` is either a BoneAttachment3D or part of a BoneAttachment3D subtree.
		// If the node is a plain non-joint, we should generate a Godot node for it.
		if (p_current_node == nullptr) {
			DEV_ASSERT(!gltf_node->joint);
			p_current_node = _generate_spatial(p_state, p_node_index);
		}
		if (!gltf_node->joint) {
			p_current_node->set_transform(gltf_node->transform);
		}
		p_current_node->set_name(gltf_node->get_name());
		attachment_godot_node->add_child(p_current_node, true);
	} else {
		// If this glTF is a plain joint, this glTF node only becomes a Godot bone.
		// We refer to the skeleton itself as this glTF node's corresponding Godot node.
		// This may be overridden later if the joint has a non-joint as a child in need of an attachment.
		p_current_node = p_godot_skeleton;
	}
	_set_node_tree_owner(p_current_node, p_scene_root);
	p_current_node->merge_meta_from(*gltf_node);
	p_state->scene_nodes.insert(p_node_index, p_current_node);
	for (int i = 0; i < gltf_node->children.size(); ++i) {
		_generate_scene_node(p_state, gltf_node->children[i], p_current_node, p_scene_root);
	}
}

// Deprecated code used when naming_version is 0 or 1 (Godot 4.0 to 4.4).
void GLTFDocument::_generate_scene_node_compat_4pt4(Ref<GLTFState> p_state, const GLTFNodeIndex p_node_index, Node *p_scene_parent, Node *p_scene_root) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	if (gltf_node->skeleton >= 0) {
		_generate_skeleton_bone_node_compat_4pt4(p_state, p_node_index, p_scene_parent, p_scene_root);
		return;
	}

	Node3D *current_node = nullptr;

	// Is our parent a skeleton
	Skeleton3D *active_skeleton = Object::cast_to<Skeleton3D>(p_scene_parent);

	const bool non_bone_parented_to_skeleton = active_skeleton;

	// skinned meshes must not be placed in a bone attachment.
	if (non_bone_parented_to_skeleton && gltf_node->skin < 0) {
		// Bone Attachment - Parent Case
		BoneAttachment3D *bone_attachment = _generate_bone_attachment_compat_4pt4(p_state, active_skeleton, p_node_index, gltf_node->parent);

		p_scene_parent->add_child(bone_attachment, true);

		// Find the correct bone_idx so we can properly serialize it.
		bone_attachment->set_bone_idx(active_skeleton->find_bone(gltf_node->get_name()));

		bone_attachment->set_owner(p_scene_root);

		// There is no gltf_node that represent this, so just directly create a unique name
		bone_attachment->set_name(gltf_node->get_name());

		// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
		// and attach it to the bone_attachment
		p_scene_parent = bone_attachment;
	}
	// Check if any GLTFDocumentExtension classes want to generate a node for us.
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		current_node = ext->generate_scene_node(p_state, gltf_node, p_scene_parent);
		if (current_node) {
			break;
		}
	}
	// If none of our GLTFDocumentExtension classes generated us a node, we generate one.
	if (!current_node) {
		if (gltf_node->skin >= 0 && gltf_node->mesh >= 0 && !gltf_node->children.is_empty()) {
			// glTF specifies that skinned meshes should ignore their node transforms,
			// only being controlled by the skeleton, so Godot will reparent a skinned
			// mesh to its skeleton. However, we still need to ensure any child nodes
			// keep their place in the tree, so if there are any child nodes, the skinned
			// mesh must not be the base node, so generate an empty spatial base.
			current_node = _generate_spatial(p_state, p_node_index);
			Node3D *mesh_inst = _generate_mesh_instance(p_state, p_node_index);
			mesh_inst->set_name(gltf_node->get_name());
			current_node->add_child(mesh_inst, true);
		} else if (gltf_node->mesh >= 0) {
			current_node = _generate_mesh_instance(p_state, p_node_index);
		} else if (gltf_node->camera >= 0) {
			current_node = _generate_camera(p_state, p_node_index);
		} else if (gltf_node->light >= 0) {
			current_node = _generate_light(p_state, p_node_index);
		} else {
			current_node = _generate_spatial(p_state, p_node_index);
		}
	}
	String gltf_node_name = gltf_node->get_name();
	if (!gltf_node_name.is_empty()) {
		current_node->set_name(gltf_node_name);
	}
	current_node->set_visible(gltf_node->visible);
	// Note: p_scene_parent and p_scene_root must either both be null or both be valid.
	if (p_scene_root == nullptr) {
		// If the root node argument is null, this is the root node.
		p_scene_root = current_node;
		// If multiple nodes were generated under the root node, ensure they have the owner set.
		if (unlikely(current_node->get_child_count() > 0)) {
			Array args;
			args.append(p_scene_root);
			for (int i = 0; i < current_node->get_child_count(); i++) {
				Node *child = current_node->get_child(i);
				child->propagate_call(StringName("set_owner"), args);
			}
		}
	} else {
		// Add the node we generated and set the owner to the scene root.
		p_scene_parent->add_child(current_node, true);
		Array args;
		args.append(p_scene_root);
		current_node->propagate_call(StringName("set_owner"), args);
		current_node->set_transform(gltf_node->transform);
	}

	current_node->merge_meta_from(*gltf_node);

	p_state->scene_nodes.insert(p_node_index, current_node);
	for (int i = 0; i < gltf_node->children.size(); ++i) {
		_generate_scene_node_compat_4pt4(p_state, gltf_node->children[i], current_node, p_scene_root);
	}
}

// Deprecated code used when naming_version is 0 or 1 (Godot 4.0 to 4.4).
void GLTFDocument::_generate_skeleton_bone_node_compat_4pt4(Ref<GLTFState> p_state, const GLTFNodeIndex p_node_index, Node *p_scene_parent, Node *p_scene_root) {
	Ref<GLTFNode> gltf_node = p_state->nodes[p_node_index];

	Node3D *current_node = nullptr;

	Skeleton3D *skeleton = p_state->skeletons[gltf_node->skeleton]->godot_skeleton;
	// In this case, this node is already a bone in skeleton.
	const bool is_skinned_mesh = (gltf_node->skin >= 0 && gltf_node->mesh >= 0);
	const bool requires_extra_node = (gltf_node->mesh >= 0 || gltf_node->camera >= 0 || gltf_node->light >= 0);

	Skeleton3D *active_skeleton = Object::cast_to<Skeleton3D>(p_scene_parent);
	if (active_skeleton != skeleton) {
		if (active_skeleton) {
			// Should no longer be possible.
			ERR_PRINT(vformat("glTF: Generating scene detected direct parented Skeletons at node %d", p_node_index));
			BoneAttachment3D *bone_attachment = _generate_bone_attachment_compat_4pt4(p_state, active_skeleton, p_node_index, gltf_node->parent);
			p_scene_parent->add_child(bone_attachment, true);
			bone_attachment->set_owner(p_scene_root);
			// There is no gltf_node that represent this, so just directly create a unique name
			bone_attachment->set_name(_gen_unique_name(p_state, "BoneAttachment3D"));
			// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
			// and attach it to the bone_attachment
			p_scene_parent = bone_attachment;
		}
		if (skeleton->get_parent() == nullptr) {
			if (p_scene_root) {
				p_scene_parent->add_child(skeleton, true);
				skeleton->set_owner(p_scene_root);
			} else {
				p_scene_parent = skeleton;
				p_scene_root = skeleton;
			}
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
			BoneAttachment3D *bone_attachment = _generate_bone_attachment_compat_4pt4(p_state, active_skeleton, p_node_index, p_node_index);

			p_scene_parent->add_child(bone_attachment, true);

			// Find the correct bone_idx so we can properly serialize it.
			bone_attachment->set_bone_idx(active_skeleton->find_bone(gltf_node->get_name()));

			bone_attachment->set_owner(p_scene_root);

			// There is no gltf_node that represent this, so just directly create a unique name
			bone_attachment->set_name(gltf_node->get_name());

			// We change the scene_parent to our bone attachment now. We do not set current_node because we want to make the node
			// and attach it to the bone_attachment
			p_scene_parent = bone_attachment;
		}
		// Check if any GLTFDocumentExtension classes want to generate a node for us.
		for (Ref<GLTFDocumentExtension> ext : document_extensions) {
			ERR_CONTINUE(ext.is_null());
			current_node = ext->generate_scene_node(p_state, gltf_node, p_scene_parent);
			if (current_node) {
				break;
			}
		}
		// If none of our GLTFDocumentExtension classes generated us a node, we generate one.
		if (!current_node) {
			if (gltf_node->mesh >= 0) {
				current_node = _generate_mesh_instance(p_state, p_node_index);
			} else if (gltf_node->camera >= 0) {
				current_node = _generate_camera(p_state, p_node_index);
			} else if (gltf_node->light >= 0) {
				current_node = _generate_light(p_state, p_node_index);
			} else {
				current_node = _generate_spatial(p_state, p_node_index);
			}
		}
		// Add the node we generated and set the owner to the scene root.
		p_scene_parent->add_child(current_node, true);
		if (current_node != p_scene_root) {
			Array args;
			args.append(p_scene_root);
			current_node->propagate_call(StringName("set_owner"), args);
		}
		// Do not set transform here. Transform is already applied to our bone.
		current_node->set_name(gltf_node->get_name());
	}

	p_state->scene_nodes.insert(p_node_index, current_node);

	for (int i = 0; i < gltf_node->children.size(); ++i) {
		_generate_scene_node_compat_4pt4(p_state, gltf_node->children[i], active_skeleton, p_scene_root);
	}
}

template <typename T>
struct SceneFormatImporterGLTFInterpolate {
	T lerp(const T &a, const T &b, float c) const {
		return a + (b - a) * c;
	}

	T catmull_rom(const T &p0, const T &p1, const T &p2, const T &p3, float t) {
		const float t2 = t * t;
		const float t3 = t2 * t;

		return 0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
	}

	T hermite(T start, T tan_start, T end, T tan_end, float t) {
		/* Formula from the glTF 2.0 specification. */
		const real_t t2 = t * t;
		const real_t t3 = t2 * t;

		const real_t h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
		const real_t h10 = t3 - 2.0 * t2 + t;
		const real_t h01 = -2.0 * t3 + 3.0 * t2;
		const real_t h11 = t3 - t2;

		return start * h00 + tan_start * h10 + end * h01 + tan_end * h11;
	}
};

// thank you for existing, partial specialization
template <>
struct SceneFormatImporterGLTFInterpolate<Quaternion> {
	Quaternion lerp(const Quaternion &a, const Quaternion &b, const float c) const {
		ERR_FAIL_COND_V_MSG(!a.is_normalized(), Quaternion(), vformat("The quaternion \"a\" %s must be normalized.", a));
		ERR_FAIL_COND_V_MSG(!b.is_normalized(), Quaternion(), vformat("The quaternion \"b\" %s must be normalized.", b));

		return a.slerp(b, c).normalized();
	}

	Quaternion catmull_rom(const Quaternion &p0, const Quaternion &p1, const Quaternion &p2, const Quaternion &p3, const float c) {
		ERR_FAIL_COND_V_MSG(!p1.is_normalized(), Quaternion(), vformat("The quaternion \"p1\" (%s) must be normalized.", p1));
		ERR_FAIL_COND_V_MSG(!p2.is_normalized(), Quaternion(), vformat("The quaternion \"p2\" (%s) must be normalized.", p2));

		return p1.slerp(p2, c).normalized();
	}

	Quaternion hermite(const Quaternion start, const Quaternion tan_start, const Quaternion end, const Quaternion tan_end, const float t) {
		ERR_FAIL_COND_V_MSG(!start.is_normalized(), Quaternion(), vformat("The start quaternion %s must be normalized.", start));
		ERR_FAIL_COND_V_MSG(!end.is_normalized(), Quaternion(), vformat("The end quaternion %s must be normalized.", end));

		return start.slerp(end, t).normalized();
	}
};

template <typename T>
T GLTFDocument::_interpolate_track(const Vector<double> &p_times, const Vector<T> &p_values, const float p_time, const GLTFAnimation::Interpolation p_interp) {
	ERR_FAIL_COND_V(p_values.is_empty(), T());
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

	SceneFormatImporterGLTFInterpolate<T> interp;

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

			const float td = (p_times[idx + 1] - p_times[idx]);
			const float c = (p_time - p_times[idx]) / td;

			const T &from = p_values[idx * 3 + 1];
			const T tan_from = td * p_values[idx * 3 + 2];
			const T &to = p_values[idx * 3 + 4];
			const T tan_to = td * p_values[idx * 3 + 3];

			return interp.hermite(from, tan_from, to, tan_to, c);
		} break;
	}

	ERR_FAIL_V(p_values[0]);
}

NodePath GLTFDocument::_find_material_node_path(Ref<GLTFState> p_state, const Ref<Material> &p_material) {
	int mesh_index = 0;
	for (Ref<GLTFMesh> gltf_mesh : p_state->meshes) {
		TypedArray<Material> materials = gltf_mesh->get_instance_materials();
		for (int mat_index = 0; mat_index < materials.size(); mat_index++) {
			if (materials[mat_index] == p_material) {
				for (Ref<GLTFNode> gltf_node : p_state->nodes) {
					if (gltf_node->mesh == mesh_index) {
						NodePath node_path = gltf_node->get_scene_node_path(p_state);
						// Example: MyNode:mesh:surface_0/material:albedo_color, so we want the mesh:surface_0/material part.
						Vector<StringName> subpath;
						subpath.append("mesh");
						subpath.append("surface_" + itos(mat_index) + "/material");
						return NodePath(node_path.get_names(), subpath, false);
					}
				}
			}
		}
		mesh_index++;
	}
	return NodePath();
}

Ref<GLTFObjectModelProperty> GLTFDocument::import_object_model_property(Ref<GLTFState> p_state, const String &p_json_pointer) {
	if (p_state->object_model_properties.has(p_json_pointer)) {
		return p_state->object_model_properties[p_json_pointer];
	}
	Ref<GLTFObjectModelProperty> ret;
	// Split the JSON pointer into its components.
	const PackedStringArray split = p_json_pointer.split("/", false);
	ERR_FAIL_COND_V_MSG(split.size() < 3, ret, "glTF: Cannot use JSON pointer '" + p_json_pointer + "' because it does not contain enough elements. The only animatable properties are at least 3 levels deep (ex: '/nodes/0/translation' or '/materials/0/emissiveFactor').");
	ret.instantiate();
	ret->set_json_pointers({ split });
	// Partial paths are passed to GLTFDocumentExtension classes if GLTFDocument cannot handle a given JSON pointer.
	TypedArray<NodePath> partial_paths;
	// Note: This might not be an integer, but in that case, we don't use this value anyway.
	const int top_level_index = split[1].to_int();
	// For JSON pointers present in the core glTF Object Model, hard-code them in GLTFDocument.
	// https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/ObjectModel.adoc
	if (split[0] == "nodes") {
		ERR_FAIL_INDEX_V_MSG(top_level_index, p_state->nodes.size(), ret, vformat("glTF: Unable to find node %d for JSON pointer '%s'.", top_level_index, p_json_pointer));
		Ref<GLTFNode> pointed_gltf_node = p_state->nodes[top_level_index];
		NodePath node_path = pointed_gltf_node->get_scene_node_path(p_state);
		partial_paths.append(node_path);
		// Check if it's something we should be able to handle.
		const String &node_prop = split[2];
		if (node_prop == "translation") {
			ret->append_path_to_property(node_path, "position");
			ret->set_types(Variant::VECTOR3, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT3);
		} else if (node_prop == "rotation") {
			ret->append_path_to_property(node_path, "quaternion");
			ret->set_types(Variant::QUATERNION, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT4);
		} else if (node_prop == "scale") {
			ret->append_path_to_property(node_path, "scale");
			ret->set_types(Variant::VECTOR3, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT3);
		} else if (node_prop == "matrix") {
			ret->append_path_to_property(node_path, "transform");
			ret->set_types(Variant::TRANSFORM3D, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT4X4);
		} else if (node_prop == "globalMatrix") {
			ret->append_path_to_property(node_path, "global_transform");
			ret->set_types(Variant::TRANSFORM3D, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT4X4);
		} else if (node_prop == "weights") {
			if (split.size() > 3) {
				const String &weight_index_string = split[3];
				ret->append_path_to_property(node_path, "blend_shapes/morph_" + weight_index_string);
				ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
			}
			// Else, Godot's MeshInstance3D does not expose the blend shape weights as one property.
			// But that's fine, we handle this case in _parse_animation_pointer instead.
		} else if (node_prop == "extensions") {
			ERR_FAIL_COND_V(split.size() < 5, ret);
			const String &ext_name = split[3];
			const String &ext_prop = split[4];
			if (ext_name == "KHR_node_visibility" && ext_prop == "visible") {
				ret->append_path_to_property(node_path, "visible");
				ret->set_types(Variant::BOOL, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_BOOL);
			}
		}
	} else if (split[0] == "cameras") {
		const String &camera_prop = split[2];
		for (Ref<GLTFNode> gltf_node : p_state->nodes) {
			if (gltf_node->camera == top_level_index) {
				NodePath node_path = gltf_node->get_scene_node_path(p_state);
				partial_paths.append(node_path);
				// Check if it's something we should be able to handle.
				if (camera_prop == "orthographic" || camera_prop == "perspective") {
					ERR_FAIL_COND_V(split.size() < 4, ret);
					ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
					const String &sub_prop = split[3];
					if (sub_prop == "xmag" || sub_prop == "ymag") {
						ret->append_path_to_property(node_path, "size");
					} else if (sub_prop == "yfov") {
						ret->append_path_to_property(node_path, "fov");
						GLTFCamera::set_fov_conversion_expressions(ret);
					} else if (sub_prop == "zfar") {
						ret->append_path_to_property(node_path, "far");
					} else if (sub_prop == "znear") {
						ret->append_path_to_property(node_path, "near");
					}
				}
			}
		}
	} else if (split[0] == "materials") {
		ERR_FAIL_INDEX_V_MSG(top_level_index, p_state->materials.size(), ret, vformat("glTF: Unable to find material %d for JSON pointer '%s'.", top_level_index, p_json_pointer));
		Ref<Material> pointed_material = p_state->materials[top_level_index];
		NodePath mat_path = _find_material_node_path(p_state, pointed_material);
		if (mat_path.is_empty()) {
			WARN_PRINT(vformat("glTF: Unable to find a path to the material %d for JSON pointer '%s'. This is likely bad data but it's also possible this is intentional. Continuing anyway.", top_level_index, p_json_pointer));
		} else {
			partial_paths.append(mat_path);
			const String &mat_prop = split[2];
			if (mat_prop == "alphaCutoff") {
				ret->append_path_to_property(mat_path, "alpha_scissor_threshold");
				ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
			} else if (mat_prop == "emissiveFactor") {
				ret->append_path_to_property(mat_path, "emission");
				ret->set_types(Variant::COLOR, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT3);
			} else if (mat_prop == "extensions") {
				ERR_FAIL_COND_V(split.size() < 5, ret);
				const String &ext_name = split[3];
				const String &ext_prop = split[4];
				if (ext_name == "KHR_materials_emissive_strength" && ext_prop == "emissiveStrength") {
					ret->append_path_to_property(mat_path, "emission_energy_multiplier");
					ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
				}
			} else {
				ERR_FAIL_COND_V(split.size() < 4, ret);
				const String &sub_prop = split[3];
				if (mat_prop == "normalTexture") {
					if (sub_prop == "scale") {
						ret->append_path_to_property(mat_path, "normal_scale");
						ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
					}
				} else if (mat_prop == "occlusionTexture") {
					if (sub_prop == "strength") {
						// This is the closest thing Godot has to an occlusion strength property.
						ret->append_path_to_property(mat_path, "ao_light_affect");
						ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
					}
				} else if (mat_prop == "pbrMetallicRoughness") {
					if (sub_prop == "baseColorFactor") {
						ret->append_path_to_property(mat_path, "albedo_color");
						ret->set_types(Variant::COLOR, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT4);
					} else if (sub_prop == "metallicFactor") {
						ret->append_path_to_property(mat_path, "metallic");
						ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
					} else if (sub_prop == "roughnessFactor") {
						ret->append_path_to_property(mat_path, "roughness");
						ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
					} else if (sub_prop == "baseColorTexture") {
						ERR_FAIL_COND_V(split.size() < 6, ret);
						const String &tex_ext_dict = split[4];
						const String &tex_ext_name = split[5];
						const String &tex_ext_prop = split[6];
						if (tex_ext_dict == "extensions" && tex_ext_name == "KHR_texture_transform") {
							// Godot only supports UVs for the whole material, not per texture.
							// We treat the albedo texture as the main texture, and import as UV1.
							// Godot does not support texture rotation, only offset and scale.
							if (tex_ext_prop == "offset") {
								ret->append_path_to_property(mat_path, "uv1_offset");
								ret->set_types(Variant::VECTOR3, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT2);
							} else if (tex_ext_prop == "scale") {
								ret->append_path_to_property(mat_path, "uv1_scale");
								ret->set_types(Variant::VECTOR3, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT2);
							}
						}
					}
				}
			}
		}
	} else if (split[0] == "meshes") {
		for (Ref<GLTFNode> gltf_node : p_state->nodes) {
			if (gltf_node->mesh == top_level_index) {
				NodePath node_path = gltf_node->get_scene_node_path(p_state);
				Vector<StringName> subpath;
				subpath.append("mesh");
				partial_paths.append(NodePath(node_path.get_names(), subpath, false));
				break;
			}
		}
	} else if (split[0] == "extensions") {
		if (split[1] == "KHR_lights_punctual" && split[2] == "lights" && split.size() > 4) {
			const int light_index = split[3].to_int();
			ERR_FAIL_INDEX_V_MSG(light_index, p_state->lights.size(), ret, vformat("glTF: Unable to find light %d for JSON pointer '%s'.", light_index, p_json_pointer));
			const String &light_prop = split[4];
			const Ref<GLTFLight> pointed_light = p_state->lights[light_index];
			for (Ref<GLTFNode> gltf_node : p_state->nodes) {
				if (gltf_node->light == light_index) {
					NodePath node_path = gltf_node->get_scene_node_path(p_state);
					partial_paths.append(node_path);
					ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
					// Check if it's something we should be able to handle.
					if (light_prop == "color") {
						ret->append_path_to_property(node_path, "light_color");
						ret->set_types(Variant::COLOR, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT3);
					} else if (light_prop == "intensity") {
						ret->append_path_to_property(node_path, "light_energy");
					} else if (light_prop == "range") {
						const String &light_type = p_state->lights[light_index]->light_type;
						if (light_type == "spot") {
							ret->append_path_to_property(node_path, "spot_range");
						} else {
							ret->append_path_to_property(node_path, "omni_range");
						}
					} else if (light_prop == "spot") {
						ERR_FAIL_COND_V(split.size() < 6, ret);
						const String &sub_prop = split[5];
						if (sub_prop == "innerConeAngle") {
							ret->append_path_to_property(node_path, "spot_angle_attenuation");
							GLTFLight::set_cone_inner_attenuation_conversion_expressions(ret);
						} else if (sub_prop == "outerConeAngle") {
							ret->append_path_to_property(node_path, "spot_angle");
						}
					}
				}
			}
		}
	}
	// Additional JSON pointers can be added by GLTFDocumentExtension classes.
	// We only need this if no mapping has been found yet from GLTFDocument's internal code.
	// When available, we pass the partial paths to the extension to help it generate the full path.
	// For example, for `/nodes/3/extensions/MY_ext/prop`, we pass a NodePath that leads to node 3,
	// so the GLTFDocumentExtension class only needs to resolve the last `MY_ext/prop` part of the path.
	// It should check `split.size() > 4 and split[0] == "nodes" and split[2] == "extensions" and split[3] == "MY_ext"`
	// at the start of the function to check if this JSON pointer applies to it, then it can handle `split[4]`.
	if (!ret->has_node_paths()) {
		for (Ref<GLTFDocumentExtension> ext : all_document_extensions) {
			ret = ext->import_object_model_property(p_state, split, partial_paths);
			if (ret.is_valid() && ret->has_node_paths()) {
				if (!ret->has_json_pointers()) {
					ret->set_json_pointers({ split });
				}
				break;
			}
		}
		if (ret.is_null() || !ret->has_node_paths()) {
			if (split.has("KHR_texture_transform")) {
				WARN_VERBOSE(vformat("glTF: Texture transforms are only supported per material in Godot. All KHR_texture_transform properties will be ignored except for the albedo texture. Ignoring JSON pointer '%s'.", p_json_pointer));
			} else {
				WARN_PRINT(vformat("glTF: Animation contained JSON pointer '%s' which could not be resolved. This property will not be animated.", p_json_pointer));
			}
		}
	}
	p_state->object_model_properties[p_json_pointer] = ret;
	return ret;
}

Ref<GLTFObjectModelProperty> GLTFDocument::export_object_model_property(Ref<GLTFState> p_state, const NodePath &p_node_path, const Node *p_godot_node, GLTFNodeIndex p_gltf_node_index) {
	Ref<GLTFObjectModelProperty> ret;
	const Object *target_object = p_godot_node;
	const Vector<StringName> subpath = p_node_path.get_subnames();
	ERR_FAIL_COND_V_MSG(subpath.is_empty(), ret, "glTF: Cannot export empty property. No property was specified in the NodePath: " + String(p_node_path));
	int target_prop_depth = 0;
	for (int64_t i = 0; i < subpath.size() - 1; i++) {
		const StringName &subname = subpath[i];
		Variant target_property = target_object->get(subname);
		if (target_property.get_type() == Variant::OBJECT) {
			target_object = target_property;
			if (target_object) {
				target_prop_depth++;
				continue;
			}
		}
		break;
	}
	const String &target_prop = subpath[target_prop_depth];
	ret.instantiate();
	ret->set_node_paths({ p_node_path });
	Vector<PackedStringArray> split_json_pointers;
	PackedStringArray split_json_pointer;
	if (Object::cast_to<BaseMaterial3D>(target_object)) {
		for (int i = 0; i < p_state->materials.size(); i++) {
			if (p_state->materials[i].ptr() == target_object) {
				split_json_pointer.append("materials");
				split_json_pointer.append(itos(i));
				if (target_prop == "alpha_scissor_threshold") {
					split_json_pointer.append("alphaCutoff");
					ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
				} else if (target_prop == "emission") {
					split_json_pointer.append("emissiveFactor");
					ret->set_types(Variant::COLOR, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT3);
				} else if (target_prop == "emission_energy_multiplier") {
					split_json_pointer.append("extensions");
					split_json_pointer.append("KHR_materials_emissive_strength");
					split_json_pointer.append("emissiveStrength");
					ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
				} else if (target_prop == "normal_scale") {
					split_json_pointer.append("normalTexture");
					split_json_pointer.append("scale");
					ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
				} else if (target_prop == "ao_light_affect") {
					split_json_pointer.append("occlusionTexture");
					split_json_pointer.append("strength");
					ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
				} else if (target_prop == "albedo_color") {
					split_json_pointer.append("pbrMetallicRoughness");
					split_json_pointer.append("baseColorFactor");
					ret->set_types(Variant::COLOR, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT4);
				} else if (target_prop == "metallic") {
					split_json_pointer.append("pbrMetallicRoughness");
					split_json_pointer.append("metallicFactor");
					ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
				} else if (target_prop == "roughness") {
					split_json_pointer.append("pbrMetallicRoughness");
					split_json_pointer.append("roughnessFactor");
					ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
				} else if (target_prop == "uv1_offset" || target_prop == "uv1_scale") {
					split_json_pointer.append("pbrMetallicRoughness");
					split_json_pointer.append("baseColorTexture");
					split_json_pointer.append("extensions");
					split_json_pointer.append("KHR_texture_transform");
					if (target_prop == "uv1_offset") {
						split_json_pointer.append("offset");
					} else {
						split_json_pointer.append("scale");
					}
					ret->set_types(Variant::VECTOR3, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT2);
				} else {
					split_json_pointer.clear();
				}
				break;
			}
		}
	} else {
		// Properties directly on Godot nodes.
		Ref<GLTFNode> gltf_node = p_state->nodes[p_gltf_node_index];
		if (Object::cast_to<Camera3D>(target_object) && gltf_node->camera >= 0) {
			split_json_pointer.append("cameras");
			split_json_pointer.append(itos(gltf_node->camera));
			const Camera3D *camera_node = Object::cast_to<Camera3D>(target_object);
			const Camera3D::ProjectionType projection_type = camera_node->get_projection();
			if (projection_type == Camera3D::PROJECTION_PERSPECTIVE) {
				split_json_pointer.append("perspective");
			} else {
				split_json_pointer.append("orthographic");
			}
			ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
			if (target_prop == "size") {
				PackedStringArray xmag = split_json_pointer.duplicate();
				xmag.append("xmag");
				split_json_pointers.append(xmag);
				split_json_pointer.append("ymag");
			} else if (target_prop == "fov") {
				split_json_pointer.append("yfov");
				GLTFCamera::set_fov_conversion_expressions(ret);
			} else if (target_prop == "far") {
				split_json_pointer.append("zfar");
			} else if (target_prop == "near") {
				split_json_pointer.append("znear");
			} else {
				split_json_pointer.clear();
			}
		} else if (Object::cast_to<Light3D>(target_object) && gltf_node->light >= 0) {
			split_json_pointer.append("extensions");
			split_json_pointer.append("KHR_lights_punctual");
			split_json_pointer.append("lights");
			split_json_pointer.append(itos(gltf_node->light));
			ret->set_types(Variant::FLOAT, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT);
			if (target_prop == "light_color") {
				split_json_pointer.append("color");
				ret->set_types(Variant::COLOR, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT3);
			} else if (target_prop == "light_energy") {
				split_json_pointer.append("intensity");
			} else if (target_prop == "spot_range") {
				split_json_pointer.append("range");
			} else if (target_prop == "omni_range") {
				split_json_pointer.append("range");
			} else if (target_prop == "spot_angle") {
				split_json_pointer.append("spot");
				split_json_pointer.append("outerConeAngle");
			} else if (target_prop == "spot_angle_attenuation") {
				split_json_pointer.append("spot");
				split_json_pointer.append("innerConeAngle");
				GLTFLight::set_cone_inner_attenuation_conversion_expressions(ret);
			} else {
				split_json_pointer.clear();
			}
		} else if (Object::cast_to<MeshInstance3D>(target_object) && target_prop.begins_with("blend_shapes/morph_")) {
			const String &weight_index_string = target_prop.trim_prefix("blend_shapes/morph_");
			split_json_pointer.append("nodes");
			split_json_pointer.append(itos(p_gltf_node_index));
			split_json_pointer.append("weights");
			split_json_pointer.append(weight_index_string);
		}
		// Transform and visibility properties. Check for all 3D nodes if we haven't resolved the JSON pointer yet.
		// Note: Do not put this in an `else`, because otherwise this will not be reached.
		if (split_json_pointer.is_empty() && Object::cast_to<Node3D>(target_object)) {
			split_json_pointer.append("nodes");
			split_json_pointer.append(itos(p_gltf_node_index));
			if (target_prop == "position") {
				split_json_pointer.append("translation");
				ret->set_types(Variant::VECTOR3, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT3);
			} else if (target_prop == "quaternion") {
				// Note: Only Quaternion rotation can be converted from Godot in this mapping.
				// Struct methods like from_euler are not accessible from the Expression class. :(
				split_json_pointer.append("rotation");
				ret->set_types(Variant::QUATERNION, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT4);
			} else if (target_prop == "scale") {
				split_json_pointer.append("scale");
				ret->set_types(Variant::VECTOR3, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT3);
			} else if (target_prop == "transform") {
				split_json_pointer.append("matrix");
				ret->set_types(Variant::TRANSFORM3D, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT4X4);
			} else if (target_prop == "global_transform") {
				split_json_pointer.append("globalMatrix");
				ret->set_types(Variant::TRANSFORM3D, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_FLOAT4X4);
			} else if (target_prop == "visible") {
				split_json_pointer.append("extensions");
				split_json_pointer.append("KHR_node_visibility");
				split_json_pointer.append("visible");
				ret->set_types(Variant::BOOL, GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_BOOL);
			} else {
				split_json_pointer.clear();
			}
		}
	}
	// Additional JSON pointers can be added by GLTFDocumentExtension classes.
	// We only need this if no mapping has been found yet from GLTFDocument's internal code.
	// We pass as many pieces of information as we can to the extension to give it lots of context.
	if (split_json_pointer.is_empty()) {
		for (Ref<GLTFDocumentExtension> ext : all_document_extensions) {
			ret = ext->export_object_model_property(p_state, p_node_path, p_godot_node, p_gltf_node_index, target_object, target_prop_depth);
			if (ret.is_valid() && ret->has_json_pointers()) {
				if (!ret->has_node_paths()) {
					ret->set_node_paths({ p_node_path });
				}
				break;
			}
		}
	} else {
		// GLTFDocument's internal code found a mapping, so set it and return it.
		split_json_pointers.append(split_json_pointer);
		ret->set_json_pointers(split_json_pointers);
	}
	return ret;
}

void GLTFDocument::_import_animation(Ref<GLTFState> p_state, AnimationPlayer *p_animation_player, const GLTFAnimationIndex p_index, const bool p_trimming, const bool p_remove_immutable_tracks) {
	ERR_FAIL_COND(p_state.is_null());
	Node *scene_root = p_animation_player->get_parent();
	ERR_FAIL_NULL(scene_root);
	Ref<GLTFAnimation> anim = p_state->animations[p_index];

	String anim_name = anim->get_name();
	if (anim_name.is_empty()) {
		// No node represent these, and they are not in the hierarchy, so just make a unique name
		anim_name = _gen_unique_name(p_state, "Animation");
	}

	Ref<Animation> animation;
	animation.instantiate();
	animation->set_name(anim_name);
	animation->set_step(1.0 / p_state->get_bake_fps());

	if (anim->get_loop()) {
		animation->set_loop_mode(Animation::LOOP_LINEAR);
	}

	double anim_start = p_trimming ? Math::INF : 0.0;
	double anim_end = 0.0;

	for (const KeyValue<int, GLTFAnimation::NodeTrack> &track_i : anim->get_node_tracks()) {
		const GLTFAnimation::NodeTrack &track = track_i.value;
		//need to find the path: for skeletons, weight tracks will affect the mesh
		NodePath node_path;
		//for skeletons, transform tracks always affect bones
		NodePath transform_node_path;
		//for meshes, especially skinned meshes, there are cases where it will be added as a child
		NodePath mesh_instance_node_path;

		GLTFNodeIndex node_index = track_i.key;

		const Ref<GLTFNode> gltf_node = p_state->nodes[track_i.key];

		HashMap<GLTFNodeIndex, Node *>::Iterator node_element = p_state->scene_nodes.find(node_index);
		ERR_CONTINUE_MSG(!node_element, vformat("Unable to find node %d for animation.", node_index));
		node_path = scene_root->get_path_to(node_element->value);
		HashMap<GLTFNodeIndex, ImporterMeshInstance3D *>::Iterator mesh_instance_element = p_state->scene_mesh_instances.find(node_index);
		if (mesh_instance_element) {
			mesh_instance_node_path = scene_root->get_path_to(mesh_instance_element->value);
		} else {
			mesh_instance_node_path = node_path;
		}

		if (gltf_node->skeleton >= 0) {
			const Skeleton3D *sk = p_state->skeletons[gltf_node->skeleton]->godot_skeleton;
			ERR_FAIL_NULL(sk);

			const String path = String(p_animation_player->get_parent()->get_path_to(sk));
			const String bone = gltf_node->get_name();
			transform_node_path = path + ":" + bone;
		} else {
			transform_node_path = node_path;
		}

		if (p_trimming) {
			for (int i = 0; i < track.rotation_track.times.size(); i++) {
				anim_start = MIN(anim_start, track.rotation_track.times[i]);
				anim_end = MAX(anim_end, track.rotation_track.times[i]);
			}
			for (int i = 0; i < track.position_track.times.size(); i++) {
				anim_start = MIN(anim_start, track.position_track.times[i]);
				anim_end = MAX(anim_end, track.position_track.times[i]);
			}
			for (int i = 0; i < track.scale_track.times.size(); i++) {
				anim_start = MIN(anim_start, track.scale_track.times[i]);
				anim_end = MAX(anim_end, track.scale_track.times[i]);
			}
			for (int i = 0; i < track.weight_tracks.size(); i++) {
				for (int j = 0; j < track.weight_tracks[i].times.size(); j++) {
					anim_start = MIN(anim_start, track.weight_tracks[i].times[j]);
					anim_end = MAX(anim_end, track.weight_tracks[i].times[j]);
				}
			}
		} else {
			// If you don't use trimming and the first key time is not at 0.0, fake keys will be inserted.
			for (int i = 0; i < track.rotation_track.times.size(); i++) {
				anim_end = MAX(anim_end, track.rotation_track.times[i]);
			}
			for (int i = 0; i < track.position_track.times.size(); i++) {
				anim_end = MAX(anim_end, track.position_track.times[i]);
			}
			for (int i = 0; i < track.scale_track.times.size(); i++) {
				anim_end = MAX(anim_end, track.scale_track.times[i]);
			}
			for (int i = 0; i < track.weight_tracks.size(); i++) {
				for (int j = 0; j < track.weight_tracks[i].times.size(); j++) {
					anim_end = MAX(anim_end, track.weight_tracks[i].times[j]);
				}
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
				bool is_default = true; //discard the track if all it contains is default values
				if (p_remove_immutable_tracks) {
					Vector3 base_pos = gltf_node->get_position();
					for (int i = 0; i < track.position_track.times.size(); i++) {
						int value_index = track.position_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + i * 3) : i;
						ERR_FAIL_COND_MSG(value_index >= track.position_track.values.size(), "Animation sampler output accessor with 'CUBICSPLINE' interpolation doesn't have enough elements.");
						Vector3 value = track.position_track.values[value_index];
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
					animation->track_set_imported(position_idx, true); //helps merging later
					if (track.position_track.interpolation == GLTFAnimation::INTERP_STEP) {
						animation->track_set_interpolation_type(position_idx, Animation::InterpolationType::INTERPOLATION_NEAREST);
					}
					base_idx++;
				}
			}
			if (track.rotation_track.values.size()) {
				bool is_default = true; //discard the track if all it contains is default values
				if (p_remove_immutable_tracks) {
					Quaternion base_rot = gltf_node->get_rotation();
					for (int i = 0; i < track.rotation_track.times.size(); i++) {
						int value_index = track.rotation_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + i * 3) : i;
						ERR_FAIL_COND_MSG(value_index >= track.rotation_track.values.size(), "Animation sampler output accessor with 'CUBICSPLINE' interpolation doesn't have enough elements.");
						Quaternion value = track.rotation_track.values[value_index].normalized();
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
					if (track.rotation_track.interpolation == GLTFAnimation::INTERP_STEP) {
						animation->track_set_interpolation_type(rotation_idx, Animation::InterpolationType::INTERPOLATION_NEAREST);
					}
					base_idx++;
				}
			}
			if (track.scale_track.values.size()) {
				bool is_default = true; //discard the track if all it contains is default values
				if (p_remove_immutable_tracks) {
					Vector3 base_scale = gltf_node->get_scale();
					for (int i = 0; i < track.scale_track.times.size(); i++) {
						int value_index = track.scale_track.interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE ? (1 + i * 3) : i;
						ERR_FAIL_COND_MSG(value_index >= track.scale_track.values.size(), "Animation sampler output accessor with 'CUBICSPLINE' interpolation doesn't have enough elements.");
						Vector3 value = track.scale_track.values[value_index];
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
					if (track.scale_track.interpolation == GLTFAnimation::INTERP_STEP) {
						animation->track_set_interpolation_type(scale_idx, Animation::InterpolationType::INTERPOLATION_NEAREST);
					}
					base_idx++;
				}
			}

			const double increment = 1.0 / p_state->get_bake_fps();
			double time = anim_start;

			Vector3 base_pos;
			Quaternion base_rot;
			Vector3 base_scale = Vector3(1, 1, 1);

			if (rotation_idx == -1) {
				base_rot = gltf_node->get_rotation();
			}

			if (position_idx == -1) {
				base_pos = gltf_node->get_position();
			}

			if (scale_idx == -1) {
				base_scale = gltf_node->get_scale();
			}

			bool last = false;
			while (true) {
				Vector3 pos = base_pos;
				Quaternion rot = base_rot;
				Vector3 scale = base_scale;

				if (position_idx >= 0) {
					pos = _interpolate_track<Vector3>(track.position_track.times, track.position_track.values, time, track.position_track.interpolation);
					animation->position_track_insert_key(position_idx, time - anim_start, pos);
				}

				if (rotation_idx >= 0) {
					rot = _interpolate_track<Quaternion>(track.rotation_track.times, track.rotation_track.values, time, track.rotation_track.interpolation);
					animation->rotation_track_insert_key(rotation_idx, time - anim_start, rot);
				}

				if (scale_idx >= 0) {
					scale = _interpolate_track<Vector3>(track.scale_track.times, track.scale_track.values, time, track.scale_track.interpolation);
					animation->scale_track_insert_key(scale_idx, time - anim_start, scale);
				}

				if (last) {
					break;
				}
				time += increment;
				if (time >= anim_end) {
					last = true;
					time = anim_end;
				}
			}
		}

		for (int i = 0; i < track.weight_tracks.size(); i++) {
			ERR_CONTINUE(gltf_node->mesh < 0 || gltf_node->mesh >= p_state->meshes.size());
			Ref<GLTFMesh> mesh = p_state->meshes[gltf_node->mesh];
			ERR_CONTINUE(mesh.is_null());
			ERR_CONTINUE(mesh->get_mesh().is_null());
			ERR_CONTINUE(mesh->get_mesh()->get_mesh().is_null());

			const String blend_path = String(mesh_instance_node_path) + ":" + String(mesh->get_mesh()->get_blend_shape_name(i));

			const int track_idx = animation->get_track_count();
			animation->add_track(Animation::TYPE_BLEND_SHAPE);
			animation->track_set_path(track_idx, blend_path);
			animation->track_set_imported(track_idx, true); //helps merging later

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
				const double increment = 1.0 / p_state->get_bake_fps();
				double time = 0.0;
				bool last = false;
				while (true) {
					real_t blend = _interpolate_track<real_t>(track.weight_tracks[i].times, track.weight_tracks[i].values, time, gltf_interp);
					animation->blend_shape_track_insert_key(track_idx, time - anim_start, blend);
					if (last) {
						break;
					}
					time += increment;
					if (time >= anim_end) {
						last = true;
						time = anim_end;
					}
				}
			}
		}
	}

	for (const KeyValue<String, GLTFAnimation::Channel<Variant>> &track_iter : anim->get_pointer_tracks()) {
		// Determine the property to animate.
		const String json_pointer = track_iter.key;
		const Ref<GLTFObjectModelProperty> prop = import_object_model_property(p_state, json_pointer);
		ERR_FAIL_COND(prop.is_null());
		// Adjust the animation duration to encompass all keyframes.
		const GLTFAnimation::Channel<Variant> &channel = track_iter.value;
		ERR_CONTINUE_MSG(channel.times.size() != channel.values.size(), vformat("glTF: Animation pointer '%s' has mismatched keyframe times and values.", json_pointer));
		if (p_trimming) {
			for (int i = 0; i < channel.times.size(); i++) {
				anim_start = MIN(anim_start, channel.times[i]);
				anim_end = MAX(anim_end, channel.times[i]);
			}
		} else {
			for (int i = 0; i < channel.times.size(); i++) {
				anim_end = MAX(anim_end, channel.times[i]);
			}
		}
		// Begin converting the glTF animation to a Godot animation.
		const Ref<Expression> gltf_to_godot_expr = prop->get_gltf_to_godot_expression();
		const bool is_gltf_to_godot_expr_valid = gltf_to_godot_expr.is_valid();
		for (const NodePath node_path : prop->get_node_paths()) {
			// If using an expression, determine the base instance to pass to the expression.
			Object *base_instance = nullptr;
			if (is_gltf_to_godot_expr_valid) {
				Ref<Resource> resource;
				Vector<StringName> leftover_subpath;
				base_instance = scene_root->get_node_and_resource(node_path, resource, leftover_subpath);
				if (resource.is_valid()) {
					base_instance = resource.ptr();
				}
			}
			// Add a track and insert all keys and values.
			const int track_index = animation->get_track_count();
			animation->add_track(Animation::TYPE_VALUE);
			animation->track_set_interpolation_type(track_index, GLTFAnimation::gltf_to_godot_interpolation(channel.interpolation));
			animation->track_set_path(track_index, node_path);
			for (int i = 0; i < channel.times.size(); i++) {
				const double time = channel.times[i];
				Variant value = channel.values[i];
				if (is_gltf_to_godot_expr_valid) {
					Array inputs;
					inputs.append(value);
					value = gltf_to_godot_expr->execute(inputs, base_instance);
				}
				animation->track_insert_key(track_index, time, value);
			}
		}
	}

	animation->set_length(anim_end - anim_start);

	Ref<AnimationLibrary> library;
	if (!p_animation_player->has_animation_library("")) {
		library.instantiate();
		p_animation_player->add_animation_library("", library);
	} else {
		library = p_animation_player->get_animation_library("");
	}
	library->add_animation(anim_name, animation);
}

void GLTFDocument::_convert_mesh_instances(Ref<GLTFState> p_state) {
	for (GLTFNodeIndex mi_node_i = 0; mi_node_i < p_state->nodes.size(); ++mi_node_i) {
		Ref<GLTFNode> node = p_state->nodes[mi_node_i];

		if (node->mesh < 0) {
			continue;
		}
		HashMap<GLTFNodeIndex, Node *>::Iterator mi_element = p_state->scene_nodes.find(mi_node_i);
		if (!mi_element) {
			continue;
		}
		MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(mi_element->value);
		if (!mi) {
			continue;
		}
		node->transform = mi->get_transform();

		Node *skel_node = mi->get_node_or_null(mi->get_skeleton_path());
		Skeleton3D *godot_skeleton = Object::cast_to<Skeleton3D>(skel_node);
		if (!godot_skeleton || godot_skeleton->get_bone_count() == 0) {
			continue;
		}
		// At this point in the code, we know we have a Skeleton3D with at least one bone.
		Ref<Skin> skin = mi->get_skin();
		Ref<GLTFSkin> gltf_skin;
		gltf_skin.instantiate();
		Array json_joints;
		if (p_state->skeleton3d_to_gltf_skeleton.has(godot_skeleton->get_instance_id())) {
			// This is a skinned mesh. If the mesh has no ARRAY_WEIGHTS or ARRAY_BONES, it will be invisible.
			const GLTFSkeletonIndex skeleton_gltf_i = p_state->skeleton3d_to_gltf_skeleton[godot_skeleton->get_instance_id()];
			Ref<GLTFSkeleton> gltf_skeleton = p_state->skeletons[skeleton_gltf_i];
			int bone_cnt = godot_skeleton->get_bone_count();
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
			if (p_state->skin_and_skeleton3d_to_gltf_skin.has(gltf_skin_key) && p_state->skin_and_skeleton3d_to_gltf_skin[gltf_skin_key].has(gltf_skel_key)) {
				skin_gltf_i = p_state->skin_and_skeleton3d_to_gltf_skin[gltf_skin_key][gltf_skel_key];
			} else {
				if (skin.is_null()) {
					// Note that gltf_skin_key should remain null, so these can share a reference.
					skin = godot_skeleton->create_skin_from_rest_transforms();
				}
				gltf_skin.instantiate();
				gltf_skin->godot_skin = skin;
				gltf_skin->set_name(skin->get_name());
				gltf_skin->skeleton = skeleton_gltf_i;
				gltf_skin->skin_root = root_gltf_i;
				//gltf_state->godot_to_gltf_node[skel_node]
				HashMap<StringName, int> bone_name_to_idx;
				for (int bone_i = 0; bone_i < bone_cnt; bone_i++) {
					bone_name_to_idx[godot_skeleton->get_bone_name(bone_i)] = bone_i;
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
						bind_name = godot_skeleton->get_bone_name(bone_i);
					}
					GLTFNodeIndex skeleton_bone_i = gltf_skeleton->joints[bone_i];
					gltf_skin->joints_original.push_back(skeleton_bone_i);
					gltf_skin->joints.push_back(skeleton_bone_i);
					gltf_skin->inverse_binds.push_back(bind_pose);
					if (godot_skeleton->get_bone_parent(bone_i) == -1) {
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

			ImporterMeshInstance3D *mi = nullptr;
			HashMap<GLTFNodeIndex, ImporterMeshInstance3D *>::Iterator mi_element = p_state->scene_mesh_instances.find(node_i);
			if (mi_element) {
				mi = mi_element->value;
			} else {
				HashMap<GLTFNodeIndex, Node *>::Iterator si_element = p_state->scene_nodes.find(node_i);
				ERR_CONTINUE_MSG(!si_element, vformat("Unable to find node %d", node_i));
				mi = Object::cast_to<ImporterMeshInstance3D>(si_element->value);
				ERR_CONTINUE_MSG(mi == nullptr, vformat("Unable to cast node %d of type %s to ImporterMeshInstance3D", node_i, si_element->value->get_class_name()));
			}
			ERR_CONTINUE_MSG(mi->get_child_count() > 0, "The glTF importer must generate skinned mesh instances as leaf nodes without any children to allow them to be repositioned in the tree without affecting other nodes.");

			const GLTFSkeletonIndex skel_i = p_state->skins.write[node->skin]->skeleton;
			Ref<GLTFSkeleton> gltf_skeleton = p_state->skeletons.write[skel_i];
			Skeleton3D *skeleton = gltf_skeleton->godot_skeleton;
			ERR_CONTINUE_MSG(skeleton == nullptr, vformat("Unable to find Skeleton for node %d skin %d", node_i, skin_i));

			mi->get_parent()->remove_child(mi);
			mi->set_owner(nullptr);
			skeleton->add_child(mi, true);
			mi->set_owner(p_scene_root);

			mi->set_skin(p_state->skins.write[skin_i]->godot_skin);
			mi->set_skeleton_path(mi->get_path_to(skeleton));
			mi->set_transform(Transform3D());
		}
	}
}

GLTFNodeIndex GLTFDocument::_node_and_or_bone_to_gltf_node_index(Ref<GLTFState> p_state, const Vector<StringName> &p_node_subpath, const Node *p_godot_node) {
	const Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(p_godot_node);
	if (skeleton && p_node_subpath.size() == 1) {
		// Special case: Handle skeleton bone TRS tracks. They use the format `A/B/C/Skeleton3D:bone_name`.
		// We have a Skeleton3D, check if it has a bone with the same name as this subpath.
		const String &bone_name = p_node_subpath[0];
		const int32_t bone_index = skeleton->find_bone(bone_name);
		if (bone_index != -1) {
			// A bone was found! But we still need to figure out which glTF node it corresponds to.
			for (GLTFSkeletonIndex skeleton_i = 0; skeleton_i < p_state->skeletons.size(); skeleton_i++) {
				const Ref<GLTFSkeleton> &skeleton_gltf = p_state->skeletons[skeleton_i];
				if (skeleton == skeleton_gltf->godot_skeleton) {
					GLTFNodeIndex node_i = skeleton_gltf->godot_bone_node[bone_index];
					return node_i;
				}
			}
			ERR_FAIL_V_MSG(-1, vformat("glTF: Found a bone %s in a Skeleton3D that wasn't in the GLTFState. Ensure that all nodes referenced by the AnimationPlayer are in the scene you are exporting.", bone_name));
		}
	}
	// General case: Not a skeleton bone, usually this means a normal node, or it could be the Skeleton3D itself.
	for (const KeyValue<GLTFNodeIndex, Node *> &scene_node_i : p_state->scene_nodes) {
		if (scene_node_i.value == p_godot_node) {
			return scene_node_i.key;
		}
	}
	ERR_FAIL_V_MSG(-1, vformat("glTF: A node was animated, but it wasn't found in the GLTFState. Ensure that all nodes referenced by the AnimationPlayer are in the scene you are exporting."));
}

bool GLTFDocument::_convert_animation_node_track(Ref<GLTFState> p_state, GLTFAnimation::NodeTrack &p_gltf_node_track, const Ref<Animation> &p_godot_animation, int32_t p_godot_anim_track_index, Vector<double> &p_times) {
	GLTFAnimation::Interpolation gltf_interpolation = GLTFAnimation::godot_to_gltf_interpolation(p_godot_animation, p_godot_anim_track_index);
	const Animation::TrackType track_type = p_godot_animation->track_get_type(p_godot_anim_track_index);
	const int32_t key_count = p_godot_animation->track_get_key_count(p_godot_anim_track_index);
	const NodePath node_path = p_godot_animation->track_get_path(p_godot_anim_track_index);
	const Vector<StringName> subpath = node_path.get_subnames();
	double anim_end = p_godot_animation->get_length();
	if (track_type == Animation::TYPE_SCALE_3D) {
		if (gltf_interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE) {
			gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
			p_gltf_node_track.scale_track.times.clear();
			p_gltf_node_track.scale_track.values.clear();
			// CATMULLROMSPLINE or CUBIC_SPLINE have to be baked, apologies.
			const double increment = 1.0 / p_state->get_bake_fps();
			double time = 0.0;
			bool last = false;
			while (true) {
				Vector3 scale;
				Error err = p_godot_animation->try_scale_track_interpolate(p_godot_anim_track_index, time, &scale);
				if (err == OK) {
					p_gltf_node_track.scale_track.values.push_back(scale);
					p_gltf_node_track.scale_track.times.push_back(time);
				} else {
					ERR_PRINT(vformat("Error interpolating animation %s scale track %d at time %f", p_godot_animation->get_name(), p_godot_anim_track_index, time));
				}
				if (last) {
					break;
				}
				time += increment;
				if (time >= anim_end) {
					last = true;
					time = anim_end;
				}
			}
		} else {
			p_gltf_node_track.scale_track.times = p_times;
			p_gltf_node_track.scale_track.interpolation = gltf_interpolation;
			p_gltf_node_track.scale_track.values.resize(key_count);
			for (int32_t key_i = 0; key_i < key_count; key_i++) {
				Vector3 scale;
				Error err = p_godot_animation->scale_track_get_key(p_godot_anim_track_index, key_i, &scale);
				ERR_CONTINUE(err != OK);
				p_gltf_node_track.scale_track.values.write[key_i] = scale;
			}
		}
	} else if (track_type == Animation::TYPE_POSITION_3D) {
		if (gltf_interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE) {
			gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
			p_gltf_node_track.position_track.times.clear();
			p_gltf_node_track.position_track.values.clear();
			// CATMULLROMSPLINE or CUBIC_SPLINE have to be baked, apologies.
			const double increment = 1.0 / p_state->get_bake_fps();
			double time = 0.0;
			bool last = false;
			while (true) {
				Vector3 scale;
				Error err = p_godot_animation->try_position_track_interpolate(p_godot_anim_track_index, time, &scale);
				if (err == OK) {
					p_gltf_node_track.position_track.values.push_back(scale);
					p_gltf_node_track.position_track.times.push_back(time);
				} else {
					ERR_PRINT(vformat("Error interpolating animation %s position track %d at time %f", p_godot_animation->get_name(), p_godot_anim_track_index, time));
				}
				if (last) {
					break;
				}
				time += increment;
				if (time >= anim_end) {
					last = true;
					time = anim_end;
				}
			}
		} else {
			p_gltf_node_track.position_track.times = p_times;
			p_gltf_node_track.position_track.values.resize(key_count);
			p_gltf_node_track.position_track.interpolation = gltf_interpolation;
			for (int32_t key_i = 0; key_i < key_count; key_i++) {
				Vector3 position;
				Error err = p_godot_animation->position_track_get_key(p_godot_anim_track_index, key_i, &position);
				ERR_CONTINUE(err != OK);
				p_gltf_node_track.position_track.values.write[key_i] = position;
			}
		}
	} else if (track_type == Animation::TYPE_ROTATION_3D) {
		if (gltf_interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE) {
			gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
			p_gltf_node_track.rotation_track.times.clear();
			p_gltf_node_track.rotation_track.values.clear();
			// CATMULLROMSPLINE or CUBIC_SPLINE have to be baked, apologies.
			const double increment = 1.0 / p_state->get_bake_fps();
			double time = 0.0;
			bool last = false;
			while (true) {
				Quaternion rotation;
				Error err = p_godot_animation->try_rotation_track_interpolate(p_godot_anim_track_index, time, &rotation);
				if (err == OK) {
					p_gltf_node_track.rotation_track.values.push_back(rotation);
					p_gltf_node_track.rotation_track.times.push_back(time);
				} else {
					ERR_PRINT(vformat("Error interpolating animation %s value rotation track %d at time %f", p_godot_animation->get_name(), p_godot_anim_track_index, time));
				}
				if (last) {
					break;
				}
				time += increment;
				if (time >= anim_end) {
					last = true;
					time = anim_end;
				}
			}
		} else {
			p_gltf_node_track.rotation_track.times = p_times;
			p_gltf_node_track.rotation_track.values.resize(key_count);
			p_gltf_node_track.rotation_track.interpolation = gltf_interpolation;
			for (int32_t key_i = 0; key_i < key_count; key_i++) {
				Quaternion rotation;
				Error err = p_godot_animation->rotation_track_get_key(p_godot_anim_track_index, key_i, &rotation);
				ERR_CONTINUE(err != OK);
				p_gltf_node_track.rotation_track.values.write[key_i] = rotation;
			}
		}
	} else if (subpath.size() > 0) {
		const StringName &node_prop = subpath[0];
		if (track_type == Animation::TYPE_VALUE) {
			if (node_prop == "position") {
				p_gltf_node_track.position_track.interpolation = gltf_interpolation;
				p_gltf_node_track.position_track.times = p_times;
				p_gltf_node_track.position_track.values.resize(key_count);

				if (gltf_interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE) {
					gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
					p_gltf_node_track.position_track.times.clear();
					p_gltf_node_track.position_track.values.clear();
					// CATMULLROMSPLINE or CUBIC_SPLINE have to be baked, apologies.
					const double increment = 1.0 / p_state->get_bake_fps();
					double time = 0.0;
					bool last = false;
					while (true) {
						Vector3 position;
						Error err = p_godot_animation->try_position_track_interpolate(p_godot_anim_track_index, time, &position);
						if (err == OK) {
							p_gltf_node_track.position_track.values.push_back(position);
							p_gltf_node_track.position_track.times.push_back(time);
						} else {
							ERR_PRINT(vformat("Error interpolating animation %s value position track %d at time %f", p_godot_animation->get_name(), p_godot_anim_track_index, time));
						}
						if (last) {
							break;
						}
						time += increment;
						if (time >= anim_end) {
							last = true;
							time = anim_end;
						}
					}
				} else {
					for (int32_t key_i = 0; key_i < key_count; key_i++) {
						Vector3 position = p_godot_animation->track_get_key_value(p_godot_anim_track_index, key_i);
						p_gltf_node_track.position_track.values.write[key_i] = position;
					}
				}
			} else if (node_prop == "rotation" || node_prop == "rotation_degrees" || node_prop == "quaternion") {
				p_gltf_node_track.rotation_track.interpolation = gltf_interpolation;
				p_gltf_node_track.rotation_track.times = p_times;
				p_gltf_node_track.rotation_track.values.resize(key_count);
				if (gltf_interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE) {
					gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
					p_gltf_node_track.rotation_track.times.clear();
					p_gltf_node_track.rotation_track.values.clear();
					// CATMULLROMSPLINE or CUBIC_SPLINE have to be baked, apologies.
					const double increment = 1.0 / p_state->get_bake_fps();
					double time = 0.0;
					bool last = false;
					while (true) {
						Quaternion rotation;
						Error err = p_godot_animation->try_rotation_track_interpolate(p_godot_anim_track_index, time, &rotation);
						if (err == OK) {
							p_gltf_node_track.rotation_track.values.push_back(rotation);
							p_gltf_node_track.rotation_track.times.push_back(time);
						} else {
							ERR_PRINT(vformat("Error interpolating animation %s value rotation track %d at time %f", p_godot_animation->get_name(), p_godot_anim_track_index, time));
						}
						if (last) {
							break;
						}
						time += increment;
						if (time >= anim_end) {
							last = true;
							time = anim_end;
						}
					}
				} else {
					for (int32_t key_i = 0; key_i < key_count; key_i++) {
						Quaternion rotation_quaternion;
						if (node_prop == "quaternion") {
							rotation_quaternion = p_godot_animation->track_get_key_value(p_godot_anim_track_index, key_i);
						} else {
							Vector3 rotation_euler = p_godot_animation->track_get_key_value(p_godot_anim_track_index, key_i);
							if (node_prop == "rotation_degrees") {
								rotation_euler *= Math::TAU / 360.0;
							}
							rotation_quaternion = Quaternion::from_euler(rotation_euler);
						}
						p_gltf_node_track.rotation_track.values.write[key_i] = rotation_quaternion;
					}
				}
			} else if (node_prop == "scale") {
				p_gltf_node_track.scale_track.interpolation = gltf_interpolation;
				p_gltf_node_track.scale_track.times = p_times;
				p_gltf_node_track.scale_track.values.resize(key_count);

				if (gltf_interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE) {
					gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
					p_gltf_node_track.scale_track.times.clear();
					p_gltf_node_track.scale_track.values.clear();
					// CATMULLROMSPLINE or CUBIC_SPLINE have to be baked, apologies.
					const double increment = 1.0 / p_state->get_bake_fps();
					double time = 0.0;
					bool last = false;
					while (true) {
						Vector3 scale;
						Error err = p_godot_animation->try_scale_track_interpolate(p_godot_anim_track_index, time, &scale);
						if (err == OK) {
							p_gltf_node_track.scale_track.values.push_back(scale);
							p_gltf_node_track.scale_track.times.push_back(time);
						} else {
							ERR_PRINT(vformat("Error interpolating animation %s scale track %d at time %f", p_godot_animation->get_name(), p_godot_anim_track_index, time));
						}
						if (last) {
							break;
						}
						time += increment;
						if (time >= anim_end) {
							last = true;
							time = anim_end;
						}
					}
				} else {
					for (int32_t key_i = 0; key_i < key_count; key_i++) {
						Vector3 scale_track = p_godot_animation->track_get_key_value(p_godot_anim_track_index, key_i);
						p_gltf_node_track.scale_track.values.write[key_i] = scale_track;
					}
				}
			} else if (node_prop == "transform") {
				p_gltf_node_track.position_track.interpolation = gltf_interpolation;
				p_gltf_node_track.position_track.times = p_times;
				p_gltf_node_track.position_track.values.resize(key_count);
				p_gltf_node_track.rotation_track.interpolation = gltf_interpolation;
				p_gltf_node_track.rotation_track.times = p_times;
				p_gltf_node_track.rotation_track.values.resize(key_count);
				p_gltf_node_track.scale_track.interpolation = gltf_interpolation;
				p_gltf_node_track.scale_track.times = p_times;
				p_gltf_node_track.scale_track.values.resize(key_count);
				if (gltf_interpolation == GLTFAnimation::INTERP_CUBIC_SPLINE) {
					gltf_interpolation = GLTFAnimation::INTERP_LINEAR;
					p_gltf_node_track.position_track.times.clear();
					p_gltf_node_track.position_track.values.clear();
					p_gltf_node_track.rotation_track.times.clear();
					p_gltf_node_track.rotation_track.values.clear();
					p_gltf_node_track.scale_track.times.clear();
					p_gltf_node_track.scale_track.values.clear();
					// CATMULLROMSPLINE or CUBIC_SPLINE have to be baked, apologies.
					const double increment = 1.0 / p_state->get_bake_fps();
					double time = 0.0;
					bool last = false;
					while (true) {
						Vector3 position;
						Quaternion rotation;
						Vector3 scale;
						Error err = p_godot_animation->try_position_track_interpolate(p_godot_anim_track_index, time, &position);
						if (err == OK) {
							err = p_godot_animation->try_rotation_track_interpolate(p_godot_anim_track_index, time, &rotation);
							if (err == OK) {
								err = p_godot_animation->try_scale_track_interpolate(p_godot_anim_track_index, time, &scale);
							}
						}
						if (err == OK) {
							p_gltf_node_track.position_track.values.push_back(position);
							p_gltf_node_track.position_track.times.push_back(time);
							p_gltf_node_track.rotation_track.values.push_back(rotation);
							p_gltf_node_track.rotation_track.times.push_back(time);
							p_gltf_node_track.scale_track.values.push_back(scale);
							p_gltf_node_track.scale_track.times.push_back(time);
						} else {
							ERR_PRINT(vformat("Error interpolating animation %s transform track %d at time %f", p_godot_animation->get_name(), p_godot_anim_track_index, time));
						}
						if (last) {
							break;
						}
						time += increment;
						if (time >= anim_end) {
							last = true;
							time = anim_end;
						}
					}
				} else {
					for (int32_t key_i = 0; key_i < key_count; key_i++) {
						Transform3D transform = p_godot_animation->track_get_key_value(p_godot_anim_track_index, key_i);
						p_gltf_node_track.position_track.values.write[key_i] = transform.get_origin();
						p_gltf_node_track.rotation_track.values.write[key_i] = transform.basis.get_rotation_quaternion();
						p_gltf_node_track.scale_track.values.write[key_i] = transform.basis.get_scale();
					}
				}
			} else {
				// This is a Value track animating a property, but not a TRS property, so it can't be converted into a node track.
				return false;
			}
		} else if (track_type == Animation::TYPE_BEZIER) {
			const int32_t keys = anim_end * p_state->get_bake_fps();
			if (node_prop == "scale") {
				if (p_gltf_node_track.scale_track.times.is_empty()) {
					p_gltf_node_track.scale_track.interpolation = gltf_interpolation;
					Vector<double> new_times;
					new_times.resize(keys);
					for (int32_t key_i = 0; key_i < keys; key_i++) {
						new_times.write[key_i] = key_i / p_state->get_bake_fps();
					}
					p_gltf_node_track.scale_track.times = new_times;

					p_gltf_node_track.scale_track.values.resize(keys);

					for (int32_t key_i = 0; key_i < keys; key_i++) {
						p_gltf_node_track.scale_track.values.write[key_i] = Vector3(1.0f, 1.0f, 1.0f);
					}

					for (int32_t key_i = 0; key_i < keys; key_i++) {
						Vector3 bezier_track = p_gltf_node_track.scale_track.values[key_i];
						if (subpath.size() == 2) {
							if (subpath[1] == StringName("x")) {
								bezier_track.x = p_godot_animation->bezier_track_interpolate(p_godot_anim_track_index, key_i / p_state->get_bake_fps());
							} else if (subpath[1] == StringName("y")) {
								bezier_track.y = p_godot_animation->bezier_track_interpolate(p_godot_anim_track_index, key_i / p_state->get_bake_fps());
							} else if (subpath[1] == StringName("z")) {
								bezier_track.z = p_godot_animation->bezier_track_interpolate(p_godot_anim_track_index, key_i / p_state->get_bake_fps());
							}
						}
						p_gltf_node_track.scale_track.values.write[key_i] = bezier_track;
					}
				}
			} else if (node_prop == "position") {
				if (p_gltf_node_track.position_track.times.is_empty()) {
					p_gltf_node_track.position_track.interpolation = gltf_interpolation;
					Vector<double> new_times;
					new_times.resize(keys);
					for (int32_t key_i = 0; key_i < keys; key_i++) {
						new_times.write[key_i] = key_i / p_state->get_bake_fps();
					}
					p_gltf_node_track.position_track.times = new_times;

					p_gltf_node_track.position_track.values.resize(keys);
				}

				for (int32_t key_i = 0; key_i < keys; key_i++) {
					Vector3 bezier_track = p_gltf_node_track.position_track.values[key_i];
					if (subpath.size() == 2) {
						if (subpath[1] == StringName("x")) {
							bezier_track.x = p_godot_animation->bezier_track_interpolate(p_godot_anim_track_index, key_i / p_state->get_bake_fps());
						} else if (subpath[1] == StringName("y")) {
							bezier_track.y = p_godot_animation->bezier_track_interpolate(p_godot_anim_track_index, key_i / p_state->get_bake_fps());
						} else if (subpath[1] == StringName("z")) {
							bezier_track.z = p_godot_animation->bezier_track_interpolate(p_godot_anim_track_index, key_i / p_state->get_bake_fps());
						}
					}
					p_gltf_node_track.position_track.values.write[key_i] = bezier_track;
				}
			} else if (node_prop == "quaternion") {
				if (p_gltf_node_track.rotation_track.times.is_empty()) {
					p_gltf_node_track.rotation_track.interpolation = gltf_interpolation;
					Vector<double> new_times;
					new_times.resize(keys);
					for (int32_t key_i = 0; key_i < keys; key_i++) {
						new_times.write[key_i] = key_i / p_state->get_bake_fps();
					}
					p_gltf_node_track.rotation_track.times = new_times;

					p_gltf_node_track.rotation_track.values.resize(keys);
				}
				for (int32_t key_i = 0; key_i < keys; key_i++) {
					Quaternion bezier_track = p_gltf_node_track.rotation_track.values[key_i];
					if (subpath.size() == 2) {
						if (subpath[1] == StringName("x")) {
							bezier_track.x = p_godot_animation->bezier_track_interpolate(p_godot_anim_track_index, key_i / p_state->get_bake_fps());
						} else if (subpath[1] == StringName("y")) {
							bezier_track.y = p_godot_animation->bezier_track_interpolate(p_godot_anim_track_index, key_i / p_state->get_bake_fps());
						} else if (subpath[1] == StringName("z")) {
							bezier_track.z = p_godot_animation->bezier_track_interpolate(p_godot_anim_track_index, key_i / p_state->get_bake_fps());
						} else if (subpath[1] == StringName("w")) {
							bezier_track.w = p_godot_animation->bezier_track_interpolate(p_godot_anim_track_index, key_i / p_state->get_bake_fps());
						}
					}
					p_gltf_node_track.rotation_track.values.write[key_i] = bezier_track;
				}
			} else {
				// This is a Bezier track animating a property, but not a TRS property, so it can't be converted into a node track.
				return false;
			}
		} else {
			// This property track isn't a Value track or Bezier track, so it can't be converted into a node track.
			return false;
		}
	} else {
		// This isn't a TRS track or a property track, so it can't be converted into a node track.
		return false;
	}
	// If we reached this point, the track was some kind of TRS track and was successfully converted.
	// All failure paths should return false before this point to indicate this
	// isn't a node track so it can be handled by KHR_animation_pointer instead.
	return true;
}

void GLTFDocument::_convert_animation(Ref<GLTFState> p_state, AnimationPlayer *p_animation_player, const String &p_animation_track_name) {
	Ref<Animation> animation = p_animation_player->get_animation(p_animation_track_name);
	Ref<GLTFAnimation> gltf_animation;
	gltf_animation.instantiate();
	gltf_animation->set_original_name(p_animation_track_name);
	gltf_animation->set_name(_gen_unique_name(p_state, p_animation_track_name));
	HashMap<int, GLTFAnimation::NodeTrack> &node_tracks = gltf_animation->get_node_tracks();
	for (int32_t track_index = 0; track_index < animation->get_track_count(); track_index++) {
		if (!animation->track_is_enabled(track_index)) {
			continue;
		}
		// Get the Godot node and the glTF node index for the animation track.
		const NodePath track_path = animation->track_get_path(track_index);
		const Node *anim_player_parent = p_animation_player->get_parent();
		const Node *animated_node = anim_player_parent->get_node_or_null(track_path);
		ERR_CONTINUE_MSG(!animated_node, "glTF: Cannot get node for animated track using path: " + String(track_path));
		const GLTFAnimation::Interpolation gltf_interpolation = GLTFAnimation::godot_to_gltf_interpolation(animation, track_index);
		// First, check if it's a Blend Shape track.
		if (animation->track_get_type(track_index) == Animation::TYPE_BLEND_SHAPE) {
			const MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(animated_node);
			ERR_CONTINUE_MSG(!mesh_instance, "glTF: Animation had a Blend Shape track, but the node wasn't a MeshInstance3D. Ignoring this track.");
			Ref<Mesh> mesh = mesh_instance->get_mesh();
			ERR_CONTINUE(mesh.is_null());
			int32_t mesh_index = -1;
			for (const KeyValue<GLTFNodeIndex, Node *> &mesh_track_i : p_state->scene_nodes) {
				if (mesh_track_i.value == animated_node) {
					mesh_index = mesh_track_i.key;
				}
			}
			ERR_CONTINUE(mesh_index == -1);
			GLTFAnimation::NodeTrack track = node_tracks.has(mesh_index) ? node_tracks[mesh_index] : GLTFAnimation::NodeTrack();
			if (!node_tracks.has(mesh_index)) {
				for (int32_t shape_i = 0; shape_i < mesh->get_blend_shape_count(); shape_i++) {
					String shape_name = mesh->get_blend_shape_name(shape_i);
					NodePath shape_path = NodePath(track_path.get_names(), { shape_name }, false);
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
				node_tracks[mesh_index] = track;
			}
			continue;
		}
		// If it's not a Blend Shape track, it must either be a TRS track, a property Value track, or something we can't handle.
		// For the cases we can handle, we will need to know the glTF node index, glTF interpolation, and the times of the track.
		const Vector<StringName> subnames = track_path.get_subnames();
		const GLTFNodeIndex node_i = _node_and_or_bone_to_gltf_node_index(p_state, subnames, animated_node);
		ERR_CONTINUE_MSG(node_i == -1, "glTF: Cannot get glTF node index for animated track using path: " + String(track_path));
		const int anim_key_count = animation->track_get_key_count(track_index);
		Vector<double> times;
		times.resize(anim_key_count);
		for (int32_t key_i = 0; key_i < anim_key_count; key_i++) {
			times.write[key_i] = animation->track_get_key_time(track_index, key_i);
		}
		// Try converting the track to a TRS glTF node track. This will only succeed if the Godot animation is a TRS track.
		const HashMap<int, GLTFAnimation::NodeTrack>::Iterator node_track_iter = node_tracks.find(node_i);
		GLTFAnimation::NodeTrack track;
		if (node_track_iter) {
			track = node_track_iter->value;
		}
		if (_convert_animation_node_track(p_state, track, animation, track_index, times)) {
			// If the track was successfully converted, save it and continue to the next track.
			node_tracks[node_i] = track;
			continue;
		}
		// If the track wasn't a TRS track or Blend Shape track, it might be a Value track animating a property.
		// Then this is something that we need to handle with KHR_animation_pointer.
		Ref<GLTFObjectModelProperty> obj_model_prop = export_object_model_property(p_state, track_path, animated_node, node_i);
		if (obj_model_prop.is_valid() && obj_model_prop->has_json_pointers()) {
			// Insert the property track into the KHR_animation_pointer pointer tracks.
			GLTFAnimation::Channel<Variant> channel;
			// Animation samplers used with `int` or `bool` Object Model Data Types **MUST** use `STEP` interpolation.
			// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_animation_pointer
			switch (obj_model_prop->get_object_model_type()) {
				case GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_BOOL:
				case GLTFObjectModelProperty::GLTF_OBJECT_MODEL_TYPE_INT: {
					channel.interpolation = GLTFAnimation::INTERP_STEP;
					if (gltf_interpolation != GLTFAnimation::INTERP_STEP) {
						WARN_PRINT(vformat("glTF export: Animation track %d on property %s is animating an int or bool, so it MUST use STEP interpolation (Godot \"Nearest\"), but the track in the Godot AnimationPlayer is using a different interpolation. Forcing STEP interpolation. Correct this track's interpolation in the source AnimationPlayer to avoid this warning.", track_index, String(track_path)));
					}
				} break;
				default: {
					channel.interpolation = gltf_interpolation;
				} break;
			}
			channel.times = times;
			channel.values.resize(anim_key_count);
			// If using an expression, determine the base instance to pass to the expression.
			const Ref<Expression> godot_to_gltf_expr = obj_model_prop->get_godot_to_gltf_expression();
			const bool is_godot_to_gltf_expr_valid = godot_to_gltf_expr.is_valid();
			Object *base_instance = nullptr;
			if (is_godot_to_gltf_expr_valid) {
				Ref<Resource> resource;
				Vector<StringName> leftover_subpath;
				base_instance = anim_player_parent->get_node_and_resource(track_path, resource, leftover_subpath);
				if (resource.is_valid()) {
					base_instance = resource.ptr();
				}
			}
			// Convert the Godot animation values into glTF animation values (still Variant).
			for (int32_t key_i = 0; key_i < anim_key_count; key_i++) {
				Variant value = animation->track_get_key_value(track_index, key_i);
				if (is_godot_to_gltf_expr_valid) {
					Array inputs;
					inputs.append(value);
					value = godot_to_gltf_expr->execute(inputs, base_instance);
				}
				channel.values.write[key_i] = value;
			}
			// Use the JSON pointer to insert the property track into the pointer tracks. There will usually be just one JSON pointer.
			HashMap<String, GLTFAnimation::Channel<Variant>> &pointer_tracks = gltf_animation->get_pointer_tracks();
			Vector<PackedStringArray> split_json_pointers = obj_model_prop->get_json_pointers();
			for (const PackedStringArray &split_json_pointer : split_json_pointers) {
				String json_pointer_str = "/" + String("/").join(split_json_pointer);
				p_state->object_model_properties[json_pointer_str] = obj_model_prop;
				pointer_tracks[json_pointer_str] = channel;
			}
		}
	}
	if (!gltf_animation->is_empty_of_tracks()) {
		p_state->animations.push_back(gltf_animation);
	}
}

Error GLTFDocument::_parse(Ref<GLTFState> p_state, const String &p_path, Ref<FileAccess> p_file) {
	Error err;
	if (p_file.is_null()) {
		return FAILED;
	}
	p_file->seek(0);
	uint32_t magic = p_file->get_32();
	if (magic == 0x46546C67) {
		// Binary file.
		p_file->seek(0);
		err = _parse_glb(p_file, p_state);
		if (err != OK) {
			return err;
		}
	} else {
		// Text file.
		p_file->seek(0);
		String text = p_file->get_as_utf8_string();
		JSON json;
		err = json.parse(text);
		ERR_FAIL_COND_V_MSG(err != OK, err, "glTF: Error parsing .gltf JSON data: " + json.get_error_message() + " at line: " + itos(json.get_error_line()));
		p_state->json = json.get_data();
	}

	err = _parse_asset_header(p_state);
	ERR_FAIL_COND_V(err != OK, err);

	document_extensions.clear();
	for (Ref<GLTFDocumentExtension> ext : all_document_extensions) {
		ERR_CONTINUE(ext.is_null());
		err = ext->import_preflight(p_state, p_state->json["extensionsUsed"]);
		if (err == OK) {
			document_extensions.push_back(ext);
		}
	}

	err = _parse_gltf_state(p_state, p_path);
	ERR_FAIL_COND_V(err != OK, err);

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

Dictionary GLTFDocument::_serialize_texture_transform_uv1(const Ref<BaseMaterial3D> &p_material) {
	ERR_FAIL_COND_V(p_material.is_null(), Dictionary());
	Vector3 offset = p_material->get_uv1_offset();
	Vector3 scale = p_material->get_uv1_scale();
	return _serialize_texture_transform_uv(Vector2(offset.x, offset.y), Vector2(scale.x, scale.y));
}

Dictionary GLTFDocument::_serialize_texture_transform_uv2(const Ref<BaseMaterial3D> &p_material) {
	ERR_FAIL_COND_V(p_material.is_null(), Dictionary());
	Vector3 offset = p_material->get_uv2_offset();
	Vector3 scale = p_material->get_uv2_scale();
	return _serialize_texture_transform_uv(Vector2(offset.x, offset.y), Vector2(scale.x, scale.y));
}

Error GLTFDocument::_serialize_asset_header(Ref<GLTFState> p_state) {
	const String version = "2.0";
	p_state->major_version = version.get_slicec('.', 0).to_int();
	p_state->minor_version = version.get_slicec('.', 1).to_int();
	Dictionary asset;
	asset["version"] = version;
	if (!p_state->copyright.is_empty()) {
		asset["copyright"] = p_state->copyright;
	}
	String hash = String(GODOT_VERSION_HASH);
	asset["generator"] = String(GODOT_VERSION_FULL_NAME) + String("@") + (hash.is_empty() ? String("unknown") : hash);
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
		Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);
		ERR_FAIL_COND_V(file.is_null(), FAILED);

		constexpr uint64_t header_size = 12;
		constexpr uint64_t chunk_header_size = 8;
		constexpr uint32_t magic = 0x46546C67; // The byte sequence "glTF" as little-endian.
		constexpr uint32_t text_chunk_type = 0x4E4F534A; // The byte sequence "JSON" as little-endian.
		constexpr uint32_t binary_chunk_type = 0x004E4942; // The byte sequence "BIN\0" as little-endian.

		String json_string = JSON::stringify(p_state->json, "", true, true);
		CharString cs = json_string.utf8();
		uint64_t text_data_length = cs.length();
		uint64_t text_chunk_length = ((text_data_length + 3) & (~3));
		uint64_t total_file_length = header_size + chunk_header_size + text_chunk_length;
		uint64_t binary_data_length = 0;
		uint64_t binary_chunk_length = 0;
		if (p_state->buffers.size() > 0) {
			binary_data_length = p_state->buffers[0].size();
			binary_chunk_length = ((binary_data_length + 3) & (~3));
			const uint64_t file_length_with_buffer = total_file_length + chunk_header_size + binary_chunk_length;
			// Check if the file length with the buffer is greater than glTF's maximum of 4 GiB.
			// If it is, we can't write the buffer into the file, but can write it separately.
			if (unlikely(file_length_with_buffer > (uint64_t)UINT32_MAX)) {
				err = _encode_buffer_bins(p_state, p_path);
				ERR_FAIL_COND_V(err != OK, err);
				// Since the buffer bins were re-encoded, we need to re-convert the JSON to string.
				json_string = JSON::stringify(p_state->json, "", true, true);
				cs = json_string.utf8();
				text_data_length = cs.length();
				text_chunk_length = ((text_data_length + 3) & (~3));
				total_file_length = header_size + chunk_header_size + text_chunk_length;
				binary_data_length = 0;
				binary_chunk_length = 0;
			} else {
				total_file_length = file_length_with_buffer;
			}
		}
		ERR_FAIL_COND_V_MSG(total_file_length > (uint64_t)UINT32_MAX, ERR_CANT_CREATE,
				"glTF: File size exceeds glTF Binary's maximum of 4 GiB. Cannot serialize as a GLB file.");

		file->create(FileAccess::ACCESS_RESOURCES);
		file->store_32(magic);
		file->store_32(p_state->major_version); // version
		file->store_32(total_file_length);

		// Write the JSON text chunk.
		file->store_32(text_chunk_length);
		file->store_32(text_chunk_type);
		file->store_buffer((uint8_t *)&cs[0], text_data_length);
		for (uint64_t pad_i = text_data_length; pad_i < text_chunk_length; pad_i++) {
			file->store_8(' ');
		}

		// Write a single binary chunk.
		if (binary_chunk_length) {
			file->store_32((uint32_t)binary_chunk_length);
			file->store_32(binary_chunk_type);
			file->store_buffer(p_state->buffers[0].ptr(), binary_data_length);
			for (uint32_t pad_i = binary_data_length; pad_i < binary_chunk_length; pad_i++) {
				file->store_8(0);
			}
		}
	} else {
		err = _encode_buffer_bins(p_state, p_path);
		ERR_FAIL_COND_V(err != OK, err);
		Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);
		ERR_FAIL_COND_V(file.is_null(), FAILED);

		file->create(FileAccess::ACCESS_RESOURCES);
		String json = JSON::stringify(p_state->json, "", true, true);
		file->store_string(json);
	}
	return err;
}

void GLTFDocument::_bind_methods() {
	BIND_ENUM_CONSTANT(ROOT_NODE_MODE_SINGLE_ROOT);
	BIND_ENUM_CONSTANT(ROOT_NODE_MODE_KEEP_ROOT);
	BIND_ENUM_CONSTANT(ROOT_NODE_MODE_MULTI_ROOT);

	BIND_ENUM_CONSTANT(VISIBILITY_MODE_INCLUDE_REQUIRED);
	BIND_ENUM_CONSTANT(VISIBILITY_MODE_INCLUDE_OPTIONAL);
	BIND_ENUM_CONSTANT(VISIBILITY_MODE_EXCLUDE);

	ClassDB::bind_method(D_METHOD("set_image_format", "image_format"), &GLTFDocument::set_image_format);
	ClassDB::bind_method(D_METHOD("get_image_format"), &GLTFDocument::get_image_format);
	ClassDB::bind_method(D_METHOD("set_lossy_quality", "lossy_quality"), &GLTFDocument::set_lossy_quality);
	ClassDB::bind_method(D_METHOD("get_lossy_quality"), &GLTFDocument::get_lossy_quality);
	ClassDB::bind_method(D_METHOD("set_fallback_image_format", "fallback_image_format"), &GLTFDocument::set_fallback_image_format);
	ClassDB::bind_method(D_METHOD("get_fallback_image_format"), &GLTFDocument::get_fallback_image_format);
	ClassDB::bind_method(D_METHOD("set_fallback_image_quality", "fallback_image_quality"), &GLTFDocument::set_fallback_image_quality);
	ClassDB::bind_method(D_METHOD("get_fallback_image_quality"), &GLTFDocument::get_fallback_image_quality);
	ClassDB::bind_method(D_METHOD("set_root_node_mode", "root_node_mode"), &GLTFDocument::set_root_node_mode);
	ClassDB::bind_method(D_METHOD("get_root_node_mode"), &GLTFDocument::get_root_node_mode);
	ClassDB::bind_method(D_METHOD("set_visibility_mode", "visibility_mode"), &GLTFDocument::set_visibility_mode);
	ClassDB::bind_method(D_METHOD("get_visibility_mode"), &GLTFDocument::get_visibility_mode);
	ClassDB::bind_method(D_METHOD("append_from_file", "path", "state", "flags", "base_path"),
			&GLTFDocument::append_from_file, DEFVAL(0), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("append_from_buffer", "bytes", "base_path", "state", "flags"),
			&GLTFDocument::append_from_buffer, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("append_from_scene", "node", "state", "flags"),
			&GLTFDocument::append_from_scene, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("generate_scene", "state", "bake_fps", "trimming", "remove_immutable_tracks"),
			&GLTFDocument::generate_scene, DEFVAL(30), DEFVAL(false), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("generate_buffer", "state"),
			&GLTFDocument::generate_buffer);
	ClassDB::bind_method(D_METHOD("write_to_filesystem", "state", "path"),
			&GLTFDocument::write_to_filesystem);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "image_format"), "set_image_format", "get_image_format");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lossy_quality"), "set_lossy_quality", "get_lossy_quality");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "fallback_image_format"), "set_fallback_image_format", "get_fallback_image_format");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fallback_image_quality"), "set_fallback_image_quality", "get_fallback_image_quality");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "root_node_mode"), "set_root_node_mode", "get_root_node_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visibility_mode"), "set_visibility_mode", "get_visibility_mode");

	ClassDB::bind_static_method("GLTFDocument", D_METHOD("import_object_model_property", "state", "json_pointer"), &GLTFDocument::import_object_model_property);
	ClassDB::bind_static_method("GLTFDocument", D_METHOD("export_object_model_property", "state", "node_path", "godot_node", "gltf_node_index"), &GLTFDocument::export_object_model_property);

	ClassDB::bind_static_method("GLTFDocument", D_METHOD("register_gltf_document_extension", "extension", "first_priority"),
			&GLTFDocument::register_gltf_document_extension, DEFVAL(false));
	ClassDB::bind_static_method("GLTFDocument", D_METHOD("unregister_gltf_document_extension", "extension"),
			&GLTFDocument::unregister_gltf_document_extension);
	ClassDB::bind_static_method("GLTFDocument", D_METHOD("get_supported_gltf_extensions"),
			&GLTFDocument::get_supported_gltf_extensions);
}

void GLTFDocument::_build_parent_hierarchy(Ref<GLTFState> p_state) {
	// build the hierarchy
	for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); node_i++) {
		for (int j = 0; j < p_state->nodes[node_i]->children.size(); j++) {
			GLTFNodeIndex child_i = p_state->nodes[node_i]->children[j];
			ERR_FAIL_INDEX(child_i, p_state->nodes.size());
			if (p_state->nodes.write[child_i]->parent != -1) {
				continue;
			}
			p_state->nodes.write[child_i]->parent = node_i;
		}
	}
}

Vector<Ref<GLTFDocumentExtension>> GLTFDocument::all_document_extensions;

void GLTFDocument::register_gltf_document_extension(Ref<GLTFDocumentExtension> p_extension, bool p_first_priority) {
	if (!all_document_extensions.has(p_extension)) {
		if (p_first_priority) {
			all_document_extensions.insert(0, p_extension);
		} else {
			all_document_extensions.push_back(p_extension);
		}
	}
}

void GLTFDocument::unregister_gltf_document_extension(Ref<GLTFDocumentExtension> p_extension) {
	all_document_extensions.erase(p_extension);
}

void GLTFDocument::unregister_all_gltf_document_extensions() {
	all_document_extensions.clear();
}

Vector<Ref<GLTFDocumentExtension>> GLTFDocument::get_all_gltf_document_extensions() {
	return all_document_extensions;
}

Vector<String> GLTFDocument::get_supported_gltf_extensions() {
	HashSet<String> set = get_supported_gltf_extensions_hashset();
	Vector<String> vec;
	for (const String &s : set) {
		vec.append(s);
	}
	vec.sort();
	return vec;
}

HashSet<String> GLTFDocument::get_supported_gltf_extensions_hashset() {
	HashSet<String> supported_extensions;
	// If the extension is supported directly in GLTFDocument, list it here.
	// Other built-in extensions are supported by GLTFDocumentExtension classes.
	supported_extensions.insert("GODOT_single_root");
	supported_extensions.insert("KHR_animation_pointer");
	supported_extensions.insert("KHR_lights_punctual");
	supported_extensions.insert("KHR_materials_emissive_strength");
	supported_extensions.insert("KHR_materials_pbrSpecularGlossiness");
	supported_extensions.insert("KHR_materials_unlit");
	supported_extensions.insert("KHR_node_visibility");
	supported_extensions.insert("KHR_texture_transform");
	for (Ref<GLTFDocumentExtension> ext : all_document_extensions) {
		ERR_CONTINUE(ext.is_null());
		Vector<String> ext_supported_extensions = ext->get_supported_extensions();
		for (int i = 0; i < ext_supported_extensions.size(); ++i) {
			supported_extensions.insert(ext_supported_extensions[i]);
		}
	}
	return supported_extensions;
}

PackedByteArray GLTFDocument::_serialize_glb_buffer(Ref<GLTFState> p_state, Error *r_err) {
	Error err = _encode_buffer_glb(p_state, "");
	if (r_err) {
		*r_err = err;
	}
	ERR_FAIL_COND_V(err != OK, PackedByteArray());
	String json_string = JSON::stringify(p_state->json, "", true, true);

	constexpr uint64_t header_size = 12;
	constexpr uint64_t chunk_header_size = 8;
	constexpr uint32_t magic = 0x46546C67; // The byte sequence "glTF" as little-endian.
	constexpr uint32_t text_chunk_type = 0x4E4F534A; // The byte sequence "JSON" as little-endian.
	constexpr uint32_t binary_chunk_type = 0x004E4942; // The byte sequence "BIN\0" as little-endian.
	const CharString cs = json_string.utf8();
	const uint64_t text_data_length = cs.length();
	const uint64_t text_chunk_length = ((text_data_length + 3) & (~3));

	uint64_t total_file_length = header_size + chunk_header_size + text_chunk_length;
	ERR_FAIL_COND_V(total_file_length > (uint64_t)UINT32_MAX, PackedByteArray());
	uint64_t binary_data_length = 0;
	if (p_state->buffers.size() > 0) {
		binary_data_length = p_state->buffers[0].size();
		const uint64_t file_length_with_buffer = total_file_length + chunk_header_size + binary_data_length;
		total_file_length = file_length_with_buffer;
	}
	ERR_FAIL_COND_V_MSG(total_file_length > (uint64_t)UINT32_MAX, PackedByteArray(),
			"glTF: File size exceeds glTF Binary's maximum of 4 GiB. Cannot serialize as a single GLB in-memory buffer.");
	const uint32_t binary_chunk_length = binary_data_length;

	Ref<StreamPeerBuffer> buffer;
	buffer.instantiate();
	buffer->put_32(magic);
	buffer->put_32(p_state->major_version); // version
	buffer->put_32((uint32_t)total_file_length); // length
	buffer->put_32((uint32_t)text_chunk_length);
	buffer->put_32(text_chunk_type);
	buffer->put_data((uint8_t *)&cs[0], text_data_length);
	for (uint64_t pad_i = text_data_length; pad_i < text_chunk_length; pad_i++) {
		buffer->put_8(' ');
	}
	if (binary_chunk_length) {
		buffer->put_32(binary_chunk_length);
		buffer->put_32(binary_chunk_type);
		buffer->put_data(p_state->buffers[0].ptr(), binary_data_length);
	}
	return buffer->get_data_array();
}

Node *GLTFDocument::_generate_scene_node_tree(Ref<GLTFState> p_state) {
	// Generate the skeletons and skins (if any).
	HashMap<ObjectID, SkinSkeletonIndex> skeleton_map;
	Error err = SkinTool::_create_skeletons(p_state->unique_names, p_state->skins, p_state->nodes,
			skeleton_map, p_state->skeletons, p_state->scene_nodes, _naming_version);
	ERR_FAIL_COND_V_MSG(err != OK, nullptr, "glTF: Failed to create skeletons.");
	err = _create_skins(p_state);
	ERR_FAIL_COND_V_MSG(err != OK, nullptr, "glTF: Failed to create skins.");
	// Run pre-generate for each extension, in case an extension needs to do something before generating the scene.
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		err = ext->import_pre_generate(p_state);
		ERR_CONTINUE(err != OK);
	}
	// Generate the node tree.
	Node *single_root;
	if (p_state->extensions_used.has("GODOT_single_root")) {
		ERR_FAIL_COND_V_MSG(p_state->nodes.is_empty(), nullptr, "glTF: Single root file has no nodes. This glTF file is invalid.");
		if (_naming_version < 2) {
			_generate_scene_node_compat_4pt4(p_state, 0, nullptr, nullptr);
		} else {
			_generate_scene_node(p_state, 0, nullptr, nullptr);
		}
		single_root = p_state->scene_nodes[0];
		if (single_root && single_root->get_owner() && single_root->get_owner() != single_root) {
			single_root = single_root->get_owner();
		}
	} else {
		single_root = memnew(Node3D);
		for (int32_t root_i = 0; root_i < p_state->root_nodes.size(); root_i++) {
			if (_naming_version < 2) {
				_generate_scene_node_compat_4pt4(p_state, p_state->root_nodes[root_i], single_root, single_root);
			} else {
				_generate_scene_node(p_state, p_state->root_nodes[root_i], single_root, single_root);
			}
		}
	}
	// Assign the scene name and single root name to each other
	// if one is missing, or do nothing if both are already set.
	if (unlikely(p_state->scene_name.is_empty())) {
		p_state->scene_name = single_root->get_name();
	} else if (single_root->get_name() == StringName()) {
		if (_naming_version == 0) {
			single_root->set_name(p_state->scene_name);
		} else {
			single_root->set_name(_gen_unique_name(p_state, p_state->scene_name));
		}
	}
	return single_root;
}

Error GLTFDocument::_parse_asset_header(Ref<GLTFState> p_state) {
	if (!p_state->json.has("asset")) {
		return ERR_PARSE_ERROR;
	}
	Dictionary asset = p_state->json["asset"];
	if (!asset.has("version")) {
		return ERR_PARSE_ERROR;
	}
	String version = asset["version"];
	p_state->major_version = version.get_slicec('.', 0).to_int();
	p_state->minor_version = version.get_slicec('.', 1).to_int();
	if (asset.has("copyright")) {
		p_state->copyright = asset["copyright"];
	}
	return OK;
}

Error GLTFDocument::_parse_gltf_state(Ref<GLTFState> p_state, const String &p_search_path) {
	Error err;

	/* PARSE BUFFERS */
	err = _parse_buffers(p_state, p_search_path);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE BUFFER VIEWS */
	err = _parse_buffer_views(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE ACCESSORS */
	err = _parse_accessors(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE EXTENSIONS */
	err = _parse_gltf_extensions(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

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

		/* PARSE TEXTURE SAMPLERS */
		err = _parse_texture_samplers(p_state);

		ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

		/* PARSE TEXTURES */
		err = _parse_textures(p_state);

		ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

		/* PARSE TEXTURES */
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

	/* ASSIGN SCENE NODE NAMES */
	// This must be run AFTER determining skeletons, and BEFORE parsing animations.
	_assign_node_names(p_state);

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

	return OK;
}

PackedByteArray GLTFDocument::generate_buffer(Ref<GLTFState> p_state) {
	Ref<GLTFState> state = p_state;
	ERR_FAIL_COND_V(state.is_null(), PackedByteArray());
	// For buffers, set the state filename to an empty string, but
	// don't touch the base path, in case the user set it manually.
	state->filename = "";
	Error err = _serialize(state);
	ERR_FAIL_COND_V(err != OK, PackedByteArray());
	PackedByteArray bytes = _serialize_glb_buffer(state, &err);
	return bytes;
}

Error GLTFDocument::write_to_filesystem(Ref<GLTFState> p_state, const String &p_path) {
	Ref<GLTFState> state = p_state;
	ERR_FAIL_COND_V(state.is_null(), ERR_INVALID_PARAMETER);
	state->set_base_path(p_path.get_base_dir());
	state->filename = p_path.get_file();
	Error err = _serialize(state);
	if (err != OK) {
		return err;
	}
	err = _serialize_file(state, p_path);
	if (err != OK) {
		return Error::FAILED;
	}
	return OK;
}

Node *GLTFDocument::generate_scene(Ref<GLTFState> p_state, float p_bake_fps, bool p_trimming, bool p_remove_immutable_tracks) {
	ERR_FAIL_COND_V(p_state.is_null(), nullptr);
	// The glTF file must have nodes, and have some marked as root nodes, in order to generate a scene.
	if (p_state->nodes.is_empty()) {
		WARN_PRINT("glTF: This glTF file has no nodes, the generated Godot scene will be empty.");
	}
	// Now that we know that we have glTF nodes, we can begin generating a scene from the parsed glTF data.
	Error err = OK;
	p_state->set_bake_fps(p_bake_fps);
	Node *godot_root_node = _generate_scene_node_tree(p_state);
	ERR_FAIL_NULL_V(godot_root_node, nullptr);
	_process_mesh_instances(p_state, godot_root_node);
	if (p_state->get_create_animations() && p_state->animations.size()) {
		AnimationPlayer *anim_player = memnew(AnimationPlayer);
		godot_root_node->add_child(anim_player, true);
		anim_player->set_owner(godot_root_node);
		for (int i = 0; i < p_state->animations.size(); i++) {
			_import_animation(p_state, anim_player, i, p_trimming, p_remove_immutable_tracks);
		}
	}
	for (KeyValue<GLTFNodeIndex, Node *> E : p_state->scene_nodes) {
		ERR_CONTINUE(!E.value);
		for (Ref<GLTFDocumentExtension> ext : document_extensions) {
			ERR_CONTINUE(ext.is_null());
			Dictionary node_json;
			if (p_state->json.has("nodes")) {
				Array nodes = p_state->json["nodes"];
				if (0 <= E.key && E.key < nodes.size()) {
					node_json = nodes[E.key];
				}
			}
			Ref<GLTFNode> gltf_node = p_state->nodes[E.key];
			err = ext->import_node(p_state, gltf_node, node_json, E.value);
			ERR_CONTINUE(err != OK);
		}
	}
	ImporterMeshInstance3D *root_importer_mesh = Object::cast_to<ImporterMeshInstance3D>(godot_root_node);
	if (unlikely(root_importer_mesh)) {
		godot_root_node = GLTFDocumentExtensionConvertImporterMesh::convert_importer_mesh_instance_3d(root_importer_mesh);
		memdelete(root_importer_mesh);
	}
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		err = ext->import_post(p_state, godot_root_node);
		ERR_CONTINUE(err != OK);
	}
	ERR_FAIL_NULL_V(godot_root_node, nullptr);
	return godot_root_node;
}

Error GLTFDocument::append_from_scene(Node *p_node, Ref<GLTFState> p_state, uint32_t p_flags) {
	ERR_FAIL_NULL_V(p_node, FAILED);
	Ref<GLTFState> state = p_state;
	ERR_FAIL_COND_V(state.is_null(), FAILED);
	state->use_named_skin_binds = p_flags & GLTF_IMPORT_USE_NAMED_SKIN_BINDS;
	state->discard_meshes_and_materials = p_flags & GLTF_IMPORT_DISCARD_MESHES_AND_MATERIALS;
	state->force_generate_tangents = p_flags & GLTF_IMPORT_GENERATE_TANGENT_ARRAYS;
	state->force_disable_compression = p_flags & GLTF_IMPORT_FORCE_DISABLE_MESH_COMPRESSION;
	if (!state->buffers.size()) {
		state->buffers.push_back(Vector<uint8_t>());
	}
	// Perform export preflight for document extensions. Only extensions that
	// return OK will be used for the rest of the export steps.
	document_extensions.clear();
	for (Ref<GLTFDocumentExtension> ext : all_document_extensions) {
		ERR_CONTINUE(ext.is_null());
		Error err = ext->export_preflight(state, p_node);
		if (err == OK) {
			document_extensions.push_back(ext);
		}
	}
	// Add the root node(s) and their descendants to the state.
	if (_root_node_mode == RootNodeMode::ROOT_NODE_MODE_MULTI_ROOT) {
		const int child_count = p_node->get_child_count();
		for (int i = 0; i < child_count; i++) {
			_convert_scene_node(state, p_node->get_child(i), -1, -1);
		}
		state->scene_name = p_node->get_name();
	} else {
		if (_root_node_mode == RootNodeMode::ROOT_NODE_MODE_SINGLE_ROOT) {
			state->extensions_used.append("GODOT_single_root");
		}
		_convert_scene_node(state, p_node, -1, -1);
	}
	// Run post-convert for each extension, in case an extension needs to do something after converting the scene.
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		Error err = ext->export_post_convert(p_state, p_node);
		ERR_CONTINUE(err != OK);
	}
	return OK;
}

Error GLTFDocument::append_from_buffer(const PackedByteArray &p_bytes, const String &p_base_path, Ref<GLTFState> p_state, uint32_t p_flags) {
	Ref<GLTFState> state = p_state;
	ERR_FAIL_COND_V(state.is_null(), FAILED);
	// TODO Add missing texture and missing .bin file paths to r_missing_deps 2021-09-10 fire
	Error err = FAILED;
	state->use_named_skin_binds = p_flags & GLTF_IMPORT_USE_NAMED_SKIN_BINDS;
	state->discard_meshes_and_materials = p_flags & GLTF_IMPORT_DISCARD_MESHES_AND_MATERIALS;
	state->force_generate_tangents = p_flags & GLTF_IMPORT_GENERATE_TANGENT_ARRAYS;
	state->force_disable_compression = p_flags & GLTF_IMPORT_FORCE_DISABLE_MESH_COMPRESSION;

	Ref<FileAccessMemory> file_access;
	file_access.instantiate();
	file_access->open_custom(p_bytes.ptr(), p_bytes.size());
	state->set_base_path(p_base_path.get_base_dir());
	err = _parse(p_state, state->base_path, file_access);
	ERR_FAIL_COND_V(err != OK, err);
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		err = ext->import_post_parse(state);
		ERR_FAIL_COND_V(err != OK, err);
	}
	return OK;
}

Error GLTFDocument::append_from_file(const String &p_path, Ref<GLTFState> p_state, uint32_t p_flags, const String &p_base_path) {
	Ref<GLTFState> state = p_state;
	// TODO Add missing texture and missing .bin file paths to r_missing_deps 2021-09-10 fire
	if (state == Ref<GLTFState>()) {
		state.instantiate();
	}
	state->set_filename(p_path.get_file().get_basename());
	state->use_named_skin_binds = p_flags & GLTF_IMPORT_USE_NAMED_SKIN_BINDS;
	state->discard_meshes_and_materials = p_flags & GLTF_IMPORT_DISCARD_MESHES_AND_MATERIALS;
	state->force_generate_tangents = p_flags & GLTF_IMPORT_GENERATE_TANGENT_ARRAYS;
	state->force_disable_compression = p_flags & GLTF_IMPORT_FORCE_DISABLE_MESH_COMPRESSION;

	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ, &err);
	ERR_FAIL_COND_V_MSG(err != OK, err, vformat(R"(Can't open file at path "%s")", p_path));
	ERR_FAIL_COND_V(file.is_null(), ERR_FILE_CANT_OPEN);
	String base_path = p_base_path;
	if (base_path.is_empty()) {
		base_path = p_path.get_base_dir();
	}
	state->set_base_path(base_path);
	err = _parse(p_state, base_path, file);
	ERR_FAIL_COND_V(err != OK, err);
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		err = ext->import_post_parse(p_state);
		ERR_FAIL_COND_V(err != OK, err);
	}
	return OK;
}

Error GLTFDocument::_parse_gltf_extensions(Ref<GLTFState> p_state) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_PARSE_ERROR);
	if (p_state->json.has("extensionsUsed")) {
		Vector<String> ext_array = p_state->json["extensionsUsed"];
		p_state->extensions_used = ext_array;
	}
	if (p_state->json.has("extensionsRequired")) {
		Vector<String> ext_array = p_state->json["extensionsRequired"];
		p_state->extensions_required = ext_array;
	}
	HashSet<String> supported_extensions = get_supported_gltf_extensions_hashset();
	Error ret = OK;
	for (int i = 0; i < p_state->extensions_required.size(); i++) {
		if (!supported_extensions.has(p_state->extensions_required[i])) {
			ERR_PRINT("glTF: Can't import file '" + p_state->filename + "', required extension '" + String(p_state->extensions_required[i]) + "' is not supported. Are you missing a GLTFDocumentExtension plugin?");
			ret = ERR_UNAVAILABLE;
		}
	}
	return ret;
}

void GLTFDocument::set_root_node_mode(GLTFDocument::RootNodeMode p_root_node_mode) {
	_root_node_mode = p_root_node_mode;
}

GLTFDocument::RootNodeMode GLTFDocument::get_root_node_mode() const {
	return _root_node_mode;
}

void GLTFDocument::set_visibility_mode(VisibilityMode p_visibility_mode) {
	_visibility_mode = p_visibility_mode;
}

GLTFDocument::VisibilityMode GLTFDocument::get_visibility_mode() const {
	return _visibility_mode;
}

String GLTFDocument::_gen_unique_name_static(HashSet<String> &r_unique_names, const String &p_name) {
	const String s_name = p_name.validate_node_name();

	String u_name;
	int index = 1;
	while (true) {
		u_name = s_name;

		if (index > 1) {
			u_name += itos(index);
		}
		if (!r_unique_names.has(u_name)) {
			break;
		}
		index++;
	}

	r_unique_names.insert(u_name);

	return u_name;
}

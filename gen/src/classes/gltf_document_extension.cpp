/**************************************************************************/
/*  gltf_document_extension.cpp                                           */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/gltf_document_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/gltf_node.hpp>
#include <godot_cpp/classes/gltf_object_model_property.hpp>
#include <godot_cpp/classes/gltf_state.hpp>
#include <godot_cpp/classes/gltf_texture.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/node_path.hpp>

namespace godot {

Error GLTFDocumentExtension::_import_preflight(const Ref<GLTFState> &p_state, const PackedStringArray &p_extensions) {
	return Error(0);
}

PackedStringArray GLTFDocumentExtension::_get_supported_extensions() {
	return PackedStringArray();
}

Error GLTFDocumentExtension::_parse_node_extensions(const Ref<GLTFState> &p_state, const Ref<GLTFNode> &p_gltf_node, const Dictionary &p_extensions) {
	return Error(0);
}

Error GLTFDocumentExtension::_parse_image_data(const Ref<GLTFState> &p_state, const PackedByteArray &p_image_data, const String &p_mime_type, const Ref<Image> &p_ret_image) {
	return Error(0);
}

String GLTFDocumentExtension::_get_image_file_extension() {
	return String();
}

Error GLTFDocumentExtension::_parse_texture_json(const Ref<GLTFState> &p_state, const Dictionary &p_texture_json, const Ref<GLTFTexture> &p_ret_gltf_texture) {
	return Error(0);
}

Ref<GLTFObjectModelProperty> GLTFDocumentExtension::_import_object_model_property(const Ref<GLTFState> &p_state, const PackedStringArray &p_split_json_pointer, const TypedArray<NodePath> &p_partial_paths) {
	return Ref<GLTFObjectModelProperty>();
}

Error GLTFDocumentExtension::_import_post_parse(const Ref<GLTFState> &p_state) {
	return Error(0);
}

Error GLTFDocumentExtension::_import_pre_generate(const Ref<GLTFState> &p_state) {
	return Error(0);
}

Node3D *GLTFDocumentExtension::_generate_scene_node(const Ref<GLTFState> &p_state, const Ref<GLTFNode> &p_gltf_node, Node *p_scene_parent) {
	return nullptr;
}

Error GLTFDocumentExtension::_import_node(const Ref<GLTFState> &p_state, const Ref<GLTFNode> &p_gltf_node, const Dictionary &p_json, Node *p_node) {
	return Error(0);
}

Error GLTFDocumentExtension::_import_post(const Ref<GLTFState> &p_state, Node *p_root) {
	return Error(0);
}

Error GLTFDocumentExtension::_export_preflight(const Ref<GLTFState> &p_state, Node *p_root) {
	return Error(0);
}

void GLTFDocumentExtension::_convert_scene_node(const Ref<GLTFState> &p_state, const Ref<GLTFNode> &p_gltf_node, Node *p_scene_node) {}

Error GLTFDocumentExtension::_export_post_convert(const Ref<GLTFState> &p_state, Node *p_root) {
	return Error(0);
}

Error GLTFDocumentExtension::_export_preserialize(const Ref<GLTFState> &p_state) {
	return Error(0);
}

Ref<GLTFObjectModelProperty> GLTFDocumentExtension::_export_object_model_property(const Ref<GLTFState> &p_state, const NodePath &p_node_path, Node *p_godot_node, int32_t p_gltf_node_index, Object *p_target_object, int32_t p_target_depth) {
	return Ref<GLTFObjectModelProperty>();
}

PackedStringArray GLTFDocumentExtension::_get_saveable_image_formats() {
	return PackedStringArray();
}

PackedByteArray GLTFDocumentExtension::_serialize_image_to_bytes(const Ref<GLTFState> &p_state, const Ref<Image> &p_image, const Dictionary &p_image_dict, const String &p_image_format, float p_lossy_quality) {
	return PackedByteArray();
}

Error GLTFDocumentExtension::_save_image_at_path(const Ref<GLTFState> &p_state, const Ref<Image> &p_image, const String &p_file_path, const String &p_image_format, float p_lossy_quality) {
	return Error(0);
}

Error GLTFDocumentExtension::_serialize_texture_json(const Ref<GLTFState> &p_state, const Dictionary &p_texture_json, const Ref<GLTFTexture> &p_gltf_texture, const String &p_image_format) {
	return Error(0);
}

Error GLTFDocumentExtension::_export_node(const Ref<GLTFState> &p_state, const Ref<GLTFNode> &p_gltf_node, const Dictionary &p_json, Node *p_node) {
	return Error(0);
}

Error GLTFDocumentExtension::_export_post(const Ref<GLTFState> &p_state) {
	return Error(0);
}

} // namespace godot

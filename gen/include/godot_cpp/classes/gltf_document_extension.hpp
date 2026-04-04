/**************************************************************************/
/*  gltf_document_extension.hpp                                           */
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

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Dictionary;
class GLTFNode;
class GLTFObjectModelProperty;
class GLTFState;
class GLTFTexture;
class Image;
class Node;
class Node3D;
class NodePath;
class Object;

class GLTFDocumentExtension : public Resource {
	GDEXTENSION_CLASS(GLTFDocumentExtension, Resource)

public:
	virtual Error _import_preflight(const Ref<GLTFState> &p_state, const PackedStringArray &p_extensions);
	virtual PackedStringArray _get_supported_extensions();
	virtual Error _parse_node_extensions(const Ref<GLTFState> &p_state, const Ref<GLTFNode> &p_gltf_node, const Dictionary &p_extensions);
	virtual Error _parse_image_data(const Ref<GLTFState> &p_state, const PackedByteArray &p_image_data, const String &p_mime_type, const Ref<Image> &p_ret_image);
	virtual String _get_image_file_extension();
	virtual Error _parse_texture_json(const Ref<GLTFState> &p_state, const Dictionary &p_texture_json, const Ref<GLTFTexture> &p_ret_gltf_texture);
	virtual Ref<GLTFObjectModelProperty> _import_object_model_property(const Ref<GLTFState> &p_state, const PackedStringArray &p_split_json_pointer, const TypedArray<NodePath> &p_partial_paths);
	virtual Error _import_post_parse(const Ref<GLTFState> &p_state);
	virtual Error _import_pre_generate(const Ref<GLTFState> &p_state);
	virtual Node3D *_generate_scene_node(const Ref<GLTFState> &p_state, const Ref<GLTFNode> &p_gltf_node, Node *p_scene_parent);
	virtual Error _import_node(const Ref<GLTFState> &p_state, const Ref<GLTFNode> &p_gltf_node, const Dictionary &p_json, Node *p_node);
	virtual Error _import_post(const Ref<GLTFState> &p_state, Node *p_root);
	virtual Error _export_preflight(const Ref<GLTFState> &p_state, Node *p_root);
	virtual void _convert_scene_node(const Ref<GLTFState> &p_state, const Ref<GLTFNode> &p_gltf_node, Node *p_scene_node);
	virtual Error _export_post_convert(const Ref<GLTFState> &p_state, Node *p_root);
	virtual Error _export_preserialize(const Ref<GLTFState> &p_state);
	virtual Ref<GLTFObjectModelProperty> _export_object_model_property(const Ref<GLTFState> &p_state, const NodePath &p_node_path, Node *p_godot_node, int32_t p_gltf_node_index, Object *p_target_object, int32_t p_target_depth);
	virtual PackedStringArray _get_saveable_image_formats();
	virtual PackedByteArray _serialize_image_to_bytes(const Ref<GLTFState> &p_state, const Ref<Image> &p_image, const Dictionary &p_image_dict, const String &p_image_format, float p_lossy_quality);
	virtual Error _save_image_at_path(const Ref<GLTFState> &p_state, const Ref<Image> &p_image, const String &p_file_path, const String &p_image_format, float p_lossy_quality);
	virtual Error _serialize_texture_json(const Ref<GLTFState> &p_state, const Dictionary &p_texture_json, const Ref<GLTFTexture> &p_gltf_texture, const String &p_image_format);
	virtual Error _export_node(const Ref<GLTFState> &p_state, const Ref<GLTFNode> &p_gltf_node, const Dictionary &p_json, Node *p_node);
	virtual Error _export_post(const Ref<GLTFState> &p_state);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_import_preflight), decltype(&T::_import_preflight)>) {
			BIND_VIRTUAL_METHOD(T, _import_preflight, 412946943);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_supported_extensions), decltype(&T::_get_supported_extensions)>) {
			BIND_VIRTUAL_METHOD(T, _get_supported_extensions, 2981934095);
		}
		if constexpr (!std::is_same_v<decltype(&B::_parse_node_extensions), decltype(&T::_parse_node_extensions)>) {
			BIND_VIRTUAL_METHOD(T, _parse_node_extensions, 2067053794);
		}
		if constexpr (!std::is_same_v<decltype(&B::_parse_image_data), decltype(&T::_parse_image_data)>) {
			BIND_VIRTUAL_METHOD(T, _parse_image_data, 3201673288);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_image_file_extension), decltype(&T::_get_image_file_extension)>) {
			BIND_VIRTUAL_METHOD(T, _get_image_file_extension, 2841200299);
		}
		if constexpr (!std::is_same_v<decltype(&B::_parse_texture_json), decltype(&T::_parse_texture_json)>) {
			BIND_VIRTUAL_METHOD(T, _parse_texture_json, 1624327185);
		}
		if constexpr (!std::is_same_v<decltype(&B::_import_object_model_property), decltype(&T::_import_object_model_property)>) {
			BIND_VIRTUAL_METHOD(T, _import_object_model_property, 1446147484);
		}
		if constexpr (!std::is_same_v<decltype(&B::_import_post_parse), decltype(&T::_import_post_parse)>) {
			BIND_VIRTUAL_METHOD(T, _import_post_parse, 1704600462);
		}
		if constexpr (!std::is_same_v<decltype(&B::_import_pre_generate), decltype(&T::_import_pre_generate)>) {
			BIND_VIRTUAL_METHOD(T, _import_pre_generate, 1704600462);
		}
		if constexpr (!std::is_same_v<decltype(&B::_generate_scene_node), decltype(&T::_generate_scene_node)>) {
			BIND_VIRTUAL_METHOD(T, _generate_scene_node, 3810899026);
		}
		if constexpr (!std::is_same_v<decltype(&B::_import_node), decltype(&T::_import_node)>) {
			BIND_VIRTUAL_METHOD(T, _import_node, 4064279746);
		}
		if constexpr (!std::is_same_v<decltype(&B::_import_post), decltype(&T::_import_post)>) {
			BIND_VIRTUAL_METHOD(T, _import_post, 295478427);
		}
		if constexpr (!std::is_same_v<decltype(&B::_export_preflight), decltype(&T::_export_preflight)>) {
			BIND_VIRTUAL_METHOD(T, _export_preflight, 295478427);
		}
		if constexpr (!std::is_same_v<decltype(&B::_convert_scene_node), decltype(&T::_convert_scene_node)>) {
			BIND_VIRTUAL_METHOD(T, _convert_scene_node, 147612932);
		}
		if constexpr (!std::is_same_v<decltype(&B::_export_post_convert), decltype(&T::_export_post_convert)>) {
			BIND_VIRTUAL_METHOD(T, _export_post_convert, 295478427);
		}
		if constexpr (!std::is_same_v<decltype(&B::_export_preserialize), decltype(&T::_export_preserialize)>) {
			BIND_VIRTUAL_METHOD(T, _export_preserialize, 1704600462);
		}
		if constexpr (!std::is_same_v<decltype(&B::_export_object_model_property), decltype(&T::_export_object_model_property)>) {
			BIND_VIRTUAL_METHOD(T, _export_object_model_property, 4111022730);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_saveable_image_formats), decltype(&T::_get_saveable_image_formats)>) {
			BIND_VIRTUAL_METHOD(T, _get_saveable_image_formats, 2981934095);
		}
		if constexpr (!std::is_same_v<decltype(&B::_serialize_image_to_bytes), decltype(&T::_serialize_image_to_bytes)>) {
			BIND_VIRTUAL_METHOD(T, _serialize_image_to_bytes, 276886664);
		}
		if constexpr (!std::is_same_v<decltype(&B::_save_image_at_path), decltype(&T::_save_image_at_path)>) {
			BIND_VIRTUAL_METHOD(T, _save_image_at_path, 1844337242);
		}
		if constexpr (!std::is_same_v<decltype(&B::_serialize_texture_json), decltype(&T::_serialize_texture_json)>) {
			BIND_VIRTUAL_METHOD(T, _serialize_texture_json, 2565166506);
		}
		if constexpr (!std::is_same_v<decltype(&B::_export_node), decltype(&T::_export_node)>) {
			BIND_VIRTUAL_METHOD(T, _export_node, 4064279746);
		}
		if constexpr (!std::is_same_v<decltype(&B::_export_post), decltype(&T::_export_post)>) {
			BIND_VIRTUAL_METHOD(T, _export_post, 1704600462);
		}
	}

public:
};

} // namespace godot


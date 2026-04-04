/**************************************************************************/
/*  gltf_document.hpp                                                     */
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

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class GLTFDocumentExtension;
class GLTFObjectModelProperty;
class GLTFState;
class Node;
class NodePath;

class GLTFDocument : public Resource {
	GDEXTENSION_CLASS(GLTFDocument, Resource)

public:
	enum RootNodeMode {
		ROOT_NODE_MODE_SINGLE_ROOT = 0,
		ROOT_NODE_MODE_KEEP_ROOT = 1,
		ROOT_NODE_MODE_MULTI_ROOT = 2,
	};

	enum VisibilityMode {
		VISIBILITY_MODE_INCLUDE_REQUIRED = 0,
		VISIBILITY_MODE_INCLUDE_OPTIONAL = 1,
		VISIBILITY_MODE_EXCLUDE = 2,
	};

	void set_image_format(const String &p_image_format);
	String get_image_format() const;
	void set_lossy_quality(float p_lossy_quality);
	float get_lossy_quality() const;
	void set_fallback_image_format(const String &p_fallback_image_format);
	String get_fallback_image_format() const;
	void set_fallback_image_quality(float p_fallback_image_quality);
	float get_fallback_image_quality() const;
	void set_root_node_mode(GLTFDocument::RootNodeMode p_root_node_mode);
	GLTFDocument::RootNodeMode get_root_node_mode() const;
	void set_visibility_mode(GLTFDocument::VisibilityMode p_visibility_mode);
	GLTFDocument::VisibilityMode get_visibility_mode() const;
	Error append_from_file(const String &p_path, const Ref<GLTFState> &p_state, uint32_t p_flags = 0, const String &p_base_path = String());
	Error append_from_buffer(const PackedByteArray &p_bytes, const String &p_base_path, const Ref<GLTFState> &p_state, uint32_t p_flags = 0);
	Error append_from_scene(Node *p_node, const Ref<GLTFState> &p_state, uint32_t p_flags = 0);
	Node *generate_scene(const Ref<GLTFState> &p_state, float p_bake_fps = 30, bool p_trimming = false, bool p_remove_immutable_tracks = true);
	PackedByteArray generate_buffer(const Ref<GLTFState> &p_state);
	Error write_to_filesystem(const Ref<GLTFState> &p_state, const String &p_path);
	static Ref<GLTFObjectModelProperty> import_object_model_property(const Ref<GLTFState> &p_state, const String &p_json_pointer);
	static Ref<GLTFObjectModelProperty> export_object_model_property(const Ref<GLTFState> &p_state, const NodePath &p_node_path, Node *p_godot_node, int32_t p_gltf_node_index);
	static void register_gltf_document_extension(const Ref<GLTFDocumentExtension> &p_extension, bool p_first_priority = false);
	static void unregister_gltf_document_extension(const Ref<GLTFDocumentExtension> &p_extension);
	static PackedStringArray get_supported_gltf_extensions();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(GLTFDocument::RootNodeMode);
VARIANT_ENUM_CAST(GLTFDocument::VisibilityMode);


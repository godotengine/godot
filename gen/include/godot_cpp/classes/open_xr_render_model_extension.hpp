/**************************************************************************/
/*  open_xr_render_model_extension.hpp                                    */
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

#include <godot_cpp/classes/open_xr_extension_wrapper.hpp>
#include <godot_cpp/classes/xr_pose.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Node3D;

class OpenXRRenderModelExtension : public OpenXRExtensionWrapper {
	GDEXTENSION_CLASS(OpenXRRenderModelExtension, OpenXRExtensionWrapper)

public:
	bool is_active() const;
	RID render_model_create(uint64_t p_render_model_id);
	void render_model_destroy(const RID &p_render_model);
	TypedArray<RID> render_model_get_all();
	Node3D *render_model_new_scene_instance(const RID &p_render_model) const;
	PackedStringArray render_model_get_subaction_paths(const RID &p_render_model);
	String render_model_get_top_level_path(const RID &p_render_model) const;
	XRPose::TrackingConfidence render_model_get_confidence(const RID &p_render_model) const;
	Transform3D render_model_get_root_transform(const RID &p_render_model) const;
	uint32_t render_model_get_animatable_node_count(const RID &p_render_model) const;
	String render_model_get_animatable_node_name(const RID &p_render_model, uint32_t p_index) const;
	bool render_model_is_animatable_node_visible(const RID &p_render_model, uint32_t p_index) const;
	Transform3D render_model_get_animatable_node_transform(const RID &p_render_model, uint32_t p_index) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		OpenXRExtensionWrapper::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


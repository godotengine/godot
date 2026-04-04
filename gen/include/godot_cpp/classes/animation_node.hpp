/**************************************************************************/
/*  animation_node.hpp                                                    */
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

#include <godot_cpp/classes/animation.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class NodePath;
class StringName;

class AnimationNode : public Resource {
	GDEXTENSION_CLASS(AnimationNode, Resource)

public:
	enum FilterAction {
		FILTER_IGNORE = 0,
		FILTER_PASS = 1,
		FILTER_STOP = 2,
		FILTER_BLEND = 3,
	};

	bool add_input(const String &p_name);
	void remove_input(int32_t p_index);
	bool set_input_name(int32_t p_input, const String &p_name);
	String get_input_name(int32_t p_input) const;
	int32_t get_input_count() const;
	int32_t find_input(const String &p_name) const;
	void set_filter_path(const NodePath &p_path, bool p_enable);
	bool is_path_filtered(const NodePath &p_path) const;
	void set_filter_enabled(bool p_enable);
	bool is_filter_enabled() const;
	uint64_t get_processing_animation_tree_instance_id() const;
	bool is_process_testing() const;
	void blend_animation(const StringName &p_animation, double p_time, double p_delta, bool p_seeked, bool p_is_external_seeking, float p_blend, Animation::LoopedFlag p_looped_flag = (Animation::LoopedFlag)0);
	double blend_node(const StringName &p_name, const Ref<AnimationNode> &p_node, double p_time, bool p_seek, bool p_is_external_seeking, float p_blend, AnimationNode::FilterAction p_filter = (AnimationNode::FilterAction)0, bool p_sync = true, bool p_test_only = false);
	double blend_input(int32_t p_input_index, double p_time, bool p_seek, bool p_is_external_seeking, float p_blend, AnimationNode::FilterAction p_filter = (AnimationNode::FilterAction)0, bool p_sync = true, bool p_test_only = false);
	void set_parameter(const StringName &p_name, const Variant &p_value);
	Variant get_parameter(const StringName &p_name) const;
	virtual Dictionary _get_child_nodes() const;
	virtual Array _get_parameter_list() const;
	virtual Ref<AnimationNode> _get_child_by_name(const StringName &p_name) const;
	virtual Variant _get_parameter_default_value(const StringName &p_parameter) const;
	virtual bool _is_parameter_read_only(const StringName &p_parameter) const;
	virtual double _process(double p_time, bool p_seek, bool p_is_external_seeking, bool p_test_only);
	virtual String _get_caption() const;
	virtual bool _has_filter() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_child_nodes), decltype(&T::_get_child_nodes)>) {
			BIND_VIRTUAL_METHOD(T, _get_child_nodes, 3102165223);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_parameter_list), decltype(&T::_get_parameter_list)>) {
			BIND_VIRTUAL_METHOD(T, _get_parameter_list, 3995934104);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_child_by_name), decltype(&T::_get_child_by_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_child_by_name, 625644256);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_parameter_default_value), decltype(&T::_get_parameter_default_value)>) {
			BIND_VIRTUAL_METHOD(T, _get_parameter_default_value, 2760726917);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_parameter_read_only), decltype(&T::_is_parameter_read_only)>) {
			BIND_VIRTUAL_METHOD(T, _is_parameter_read_only, 2619796661);
		}
		if constexpr (!std::is_same_v<decltype(&B::_process), decltype(&T::_process)>) {
			BIND_VIRTUAL_METHOD(T, _process, 2139827523);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_caption), decltype(&T::_get_caption)>) {
			BIND_VIRTUAL_METHOD(T, _get_caption, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_has_filter), decltype(&T::_has_filter)>) {
			BIND_VIRTUAL_METHOD(T, _has_filter, 36873697);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AnimationNode::FilterAction);


/**************************************************************************/
/*  material.hpp                                                          */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/variant/rid.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Material : public Resource {
	GDEXTENSION_CLASS(Material, Resource)

public:
	static const int RENDER_PRIORITY_MAX = 127;
	static const int RENDER_PRIORITY_MIN = -128;

	void set_next_pass(const Ref<Material> &p_next_pass);
	Ref<Material> get_next_pass() const;
	void set_render_priority(int32_t p_priority);
	int32_t get_render_priority() const;
	void inspect_native_shader_code();
	Ref<Resource> create_placeholder() const;
	virtual RID _get_shader_rid() const;
	virtual Shader::Mode _get_shader_mode() const;
	virtual bool _can_do_next_pass() const;
	virtual bool _can_use_render_priority() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_shader_rid), decltype(&T::_get_shader_rid)>) {
			BIND_VIRTUAL_METHOD(T, _get_shader_rid, 2944877500);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_shader_mode), decltype(&T::_get_shader_mode)>) {
			BIND_VIRTUAL_METHOD(T, _get_shader_mode, 3392948163);
		}
		if constexpr (!std::is_same_v<decltype(&B::_can_do_next_pass), decltype(&T::_can_do_next_pass)>) {
			BIND_VIRTUAL_METHOD(T, _can_do_next_pass, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_can_use_render_priority), decltype(&T::_can_use_render_priority)>) {
			BIND_VIRTUAL_METHOD(T, _can_use_render_priority, 36873697);
		}
	}

public:
};

} // namespace godot


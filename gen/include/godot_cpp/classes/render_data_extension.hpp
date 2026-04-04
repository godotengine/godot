/**************************************************************************/
/*  render_data_extension.hpp                                             */
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
#include <godot_cpp/classes/render_data.hpp>
#include <godot_cpp/variant/rid.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class RenderSceneBuffers;
class RenderSceneData;

class RenderDataExtension : public RenderData {
	GDEXTENSION_CLASS(RenderDataExtension, RenderData)

public:
	virtual Ref<RenderSceneBuffers> _get_render_scene_buffers() const;
	virtual RenderSceneData *_get_render_scene_data() const;
	virtual RID _get_environment() const;
	virtual RID _get_camera_attributes() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RenderData::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_render_scene_buffers), decltype(&T::_get_render_scene_buffers)>) {
			BIND_VIRTUAL_METHOD(T, _get_render_scene_buffers, 2793216201);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_render_scene_data), decltype(&T::_get_render_scene_data)>) {
			BIND_VIRTUAL_METHOD(T, _get_render_scene_data, 1288715698);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_environment), decltype(&T::_get_environment)>) {
			BIND_VIRTUAL_METHOD(T, _get_environment, 2944877500);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_camera_attributes), decltype(&T::_get_camera_attributes)>) {
			BIND_VIRTUAL_METHOD(T, _get_camera_attributes, 2944877500);
		}
	}

public:
};

} // namespace godot


/**************************************************************************/
/*  render_scene_data_extension.hpp                                       */
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

#include <godot_cpp/classes/render_scene_data.hpp>
#include <godot_cpp/variant/projection.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class RenderSceneDataExtension : public RenderSceneData {
	GDEXTENSION_CLASS(RenderSceneDataExtension, RenderSceneData)

public:
	virtual Transform3D _get_cam_transform() const;
	virtual Projection _get_cam_projection() const;
	virtual uint32_t _get_view_count() const;
	virtual Vector3 _get_view_eye_offset(uint32_t p_view) const;
	virtual Projection _get_view_projection(uint32_t p_view) const;
	virtual RID _get_uniform_buffer() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RenderSceneData::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_cam_transform), decltype(&T::_get_cam_transform)>) {
			BIND_VIRTUAL_METHOD(T, _get_cam_transform, 3229777777);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_cam_projection), decltype(&T::_get_cam_projection)>) {
			BIND_VIRTUAL_METHOD(T, _get_cam_projection, 2910717950);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_view_count), decltype(&T::_get_view_count)>) {
			BIND_VIRTUAL_METHOD(T, _get_view_count, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_view_eye_offset), decltype(&T::_get_view_eye_offset)>) {
			BIND_VIRTUAL_METHOD(T, _get_view_eye_offset, 711720468);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_view_projection), decltype(&T::_get_view_projection)>) {
			BIND_VIRTUAL_METHOD(T, _get_view_projection, 3179846605);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_uniform_buffer), decltype(&T::_get_uniform_buffer)>) {
			BIND_VIRTUAL_METHOD(T, _get_uniform_buffer, 2944877500);
		}
	}

public:
};

} // namespace godot


/**************************************************************************/
/*  texture3d.hpp                                                         */
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

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/texture.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Resource;

class Texture3D : public Texture {
	GDEXTENSION_CLASS(Texture3D, Texture)

public:
	Image::Format get_format() const;
	int32_t get_width() const;
	int32_t get_height() const;
	int32_t get_depth() const;
	bool has_mipmaps() const;
	TypedArray<Ref<Image>> get_data() const;
	Ref<Resource> create_placeholder() const;
	virtual Image::Format _get_format() const;
	virtual int32_t _get_width() const;
	virtual int32_t _get_height() const;
	virtual int32_t _get_depth() const;
	virtual bool _has_mipmaps() const;
	virtual TypedArray<Ref<Image>> _get_data() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Texture::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_format), decltype(&T::_get_format)>) {
			BIND_VIRTUAL_METHOD(T, _get_format, 3847873762);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_width), decltype(&T::_get_width)>) {
			BIND_VIRTUAL_METHOD(T, _get_width, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_height), decltype(&T::_get_height)>) {
			BIND_VIRTUAL_METHOD(T, _get_height, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_depth), decltype(&T::_get_depth)>) {
			BIND_VIRTUAL_METHOD(T, _get_depth, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_has_mipmaps), decltype(&T::_has_mipmaps)>) {
			BIND_VIRTUAL_METHOD(T, _has_mipmaps, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_data), decltype(&T::_get_data)>) {
			BIND_VIRTUAL_METHOD(T, _get_data, 3995934104);
		}
	}

public:
};

} // namespace godot


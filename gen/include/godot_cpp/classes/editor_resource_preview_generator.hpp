/**************************************************************************/
/*  editor_resource_preview_generator.hpp                                 */
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
#include <godot_cpp/classes/ref_counted.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Dictionary;
class RID;
class Resource;
class String;
class Texture2D;
struct Vector2i;

class EditorResourcePreviewGenerator : public RefCounted {
	GDEXTENSION_CLASS(EditorResourcePreviewGenerator, RefCounted)

public:
	void request_draw_and_wait(const RID &p_viewport) const;
	virtual bool _handles(const String &p_type) const;
	virtual Ref<Texture2D> _generate(const Ref<Resource> &p_resource, const Vector2i &p_size, const Dictionary &p_metadata) const;
	virtual Ref<Texture2D> _generate_from_path(const String &p_path, const Vector2i &p_size, const Dictionary &p_metadata) const;
	virtual bool _generate_small_preview_automatically() const;
	virtual bool _can_generate_small_preview() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_handles), decltype(&T::_handles)>) {
			BIND_VIRTUAL_METHOD(T, _handles, 3927539163);
		}
		if constexpr (!std::is_same_v<decltype(&B::_generate), decltype(&T::_generate)>) {
			BIND_VIRTUAL_METHOD(T, _generate, 255939159);
		}
		if constexpr (!std::is_same_v<decltype(&B::_generate_from_path), decltype(&T::_generate_from_path)>) {
			BIND_VIRTUAL_METHOD(T, _generate_from_path, 1601192835);
		}
		if constexpr (!std::is_same_v<decltype(&B::_generate_small_preview_automatically), decltype(&T::_generate_small_preview_automatically)>) {
			BIND_VIRTUAL_METHOD(T, _generate_small_preview_automatically, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_can_generate_small_preview), decltype(&T::_can_generate_small_preview)>) {
			BIND_VIRTUAL_METHOD(T, _can_generate_small_preview, 36873697);
		}
	}

public:
};

} // namespace godot


/**************************************************************************/
/*  sub_viewport.hpp                                                      */
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

#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class SubViewport : public Viewport {
	GDEXTENSION_CLASS(SubViewport, Viewport)

public:
	enum ClearMode {
		CLEAR_MODE_ALWAYS = 0,
		CLEAR_MODE_NEVER = 1,
		CLEAR_MODE_ONCE = 2,
	};

	enum UpdateMode {
		UPDATE_DISABLED = 0,
		UPDATE_ONCE = 1,
		UPDATE_WHEN_VISIBLE = 2,
		UPDATE_WHEN_PARENT_VISIBLE = 3,
		UPDATE_ALWAYS = 4,
	};

	void set_size(const Vector2i &p_size);
	Vector2i get_size() const;
	void set_size_2d_override(const Vector2i &p_size);
	Vector2i get_size_2d_override() const;
	void set_size_2d_override_stretch(bool p_enable);
	bool is_size_2d_override_stretch_enabled() const;
	void set_update_mode(SubViewport::UpdateMode p_mode);
	SubViewport::UpdateMode get_update_mode() const;
	void set_clear_mode(SubViewport::ClearMode p_mode);
	SubViewport::ClearMode get_clear_mode() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Viewport::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SubViewport::ClearMode);
VARIANT_ENUM_CAST(SubViewport::UpdateMode);


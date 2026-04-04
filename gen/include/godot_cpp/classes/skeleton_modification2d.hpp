/**************************************************************************/
/*  skeleton_modification2d.hpp                                           */
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

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class SkeletonModificationStack2D;

class SkeletonModification2D : public Resource {
	GDEXTENSION_CLASS(SkeletonModification2D, Resource)

public:
	void set_enabled(bool p_enabled);
	bool get_enabled();
	Ref<SkeletonModificationStack2D> get_modification_stack();
	void set_is_setup(bool p_is_setup);
	bool get_is_setup() const;
	void set_execution_mode(int32_t p_execution_mode);
	int32_t get_execution_mode() const;
	float clamp_angle(float p_angle, float p_min, float p_max, bool p_invert);
	void set_editor_draw_gizmo(bool p_draw_gizmo);
	bool get_editor_draw_gizmo() const;
	virtual void _execute(double p_delta);
	virtual void _setup_modification(const Ref<SkeletonModificationStack2D> &p_modification_stack);
	virtual void _draw_editor_gizmo();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_execute), decltype(&T::_execute)>) {
			BIND_VIRTUAL_METHOD(T, _execute, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_setup_modification), decltype(&T::_setup_modification)>) {
			BIND_VIRTUAL_METHOD(T, _setup_modification, 3907307132);
		}
		if constexpr (!std::is_same_v<decltype(&B::_draw_editor_gizmo), decltype(&T::_draw_editor_gizmo)>) {
			BIND_VIRTUAL_METHOD(T, _draw_editor_gizmo, 3218959716);
		}
	}

public:
};

} // namespace godot


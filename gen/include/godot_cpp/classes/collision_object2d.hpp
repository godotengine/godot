/**************************************************************************/
/*  collision_object2d.hpp                                                */
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

#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform2d.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class InputEvent;
class Object;
class Shape2D;
class Viewport;

class CollisionObject2D : public Node2D {
	GDEXTENSION_CLASS(CollisionObject2D, Node2D)

public:
	enum DisableMode {
		DISABLE_MODE_REMOVE = 0,
		DISABLE_MODE_MAKE_STATIC = 1,
		DISABLE_MODE_KEEP_ACTIVE = 2,
	};

	RID get_rid() const;
	void set_collision_layer(uint32_t p_layer);
	uint32_t get_collision_layer() const;
	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;
	void set_collision_layer_value(int32_t p_layer_number, bool p_value);
	bool get_collision_layer_value(int32_t p_layer_number) const;
	void set_collision_mask_value(int32_t p_layer_number, bool p_value);
	bool get_collision_mask_value(int32_t p_layer_number) const;
	void set_collision_priority(float p_priority);
	float get_collision_priority() const;
	void set_disable_mode(CollisionObject2D::DisableMode p_mode);
	CollisionObject2D::DisableMode get_disable_mode() const;
	void set_pickable(bool p_enabled);
	bool is_pickable() const;
	uint32_t create_shape_owner(Object *p_owner);
	void remove_shape_owner(uint32_t p_owner_id);
	PackedInt32Array get_shape_owners();
	void shape_owner_set_transform(uint32_t p_owner_id, const Transform2D &p_transform);
	Transform2D shape_owner_get_transform(uint32_t p_owner_id) const;
	Object *shape_owner_get_owner(uint32_t p_owner_id) const;
	void shape_owner_set_disabled(uint32_t p_owner_id, bool p_disabled);
	bool is_shape_owner_disabled(uint32_t p_owner_id) const;
	void shape_owner_set_one_way_collision(uint32_t p_owner_id, bool p_enable);
	bool is_shape_owner_one_way_collision_enabled(uint32_t p_owner_id) const;
	void shape_owner_set_one_way_collision_margin(uint32_t p_owner_id, float p_margin);
	float get_shape_owner_one_way_collision_margin(uint32_t p_owner_id) const;
	void shape_owner_add_shape(uint32_t p_owner_id, const Ref<Shape2D> &p_shape);
	int32_t shape_owner_get_shape_count(uint32_t p_owner_id) const;
	Ref<Shape2D> shape_owner_get_shape(uint32_t p_owner_id, int32_t p_shape_id) const;
	int32_t shape_owner_get_shape_index(uint32_t p_owner_id, int32_t p_shape_id) const;
	void shape_owner_remove_shape(uint32_t p_owner_id, int32_t p_shape_id);
	void shape_owner_clear_shapes(uint32_t p_owner_id);
	uint32_t shape_find_owner(int32_t p_shape_index) const;
	virtual void _input_event(Viewport *p_viewport, const Ref<InputEvent> &p_event, int32_t p_shape_idx);
	virtual void _mouse_enter();
	virtual void _mouse_exit();
	virtual void _mouse_shape_enter(int32_t p_shape_idx);
	virtual void _mouse_shape_exit(int32_t p_shape_idx);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node2D::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_input_event), decltype(&T::_input_event)>) {
			BIND_VIRTUAL_METHOD(T, _input_event, 1847696837);
		}
		if constexpr (!std::is_same_v<decltype(&B::_mouse_enter), decltype(&T::_mouse_enter)>) {
			BIND_VIRTUAL_METHOD(T, _mouse_enter, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_mouse_exit), decltype(&T::_mouse_exit)>) {
			BIND_VIRTUAL_METHOD(T, _mouse_exit, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_mouse_shape_enter), decltype(&T::_mouse_shape_enter)>) {
			BIND_VIRTUAL_METHOD(T, _mouse_shape_enter, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_mouse_shape_exit), decltype(&T::_mouse_shape_exit)>) {
			BIND_VIRTUAL_METHOD(T, _mouse_shape_exit, 1286410249);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(CollisionObject2D::DisableMode);


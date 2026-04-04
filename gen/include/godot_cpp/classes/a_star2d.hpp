/**************************************************************************/
/*  a_star2d.hpp                                                          */
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
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AStar2D : public RefCounted {
	GDEXTENSION_CLASS(AStar2D, RefCounted)

public:
	int64_t get_available_point_id() const;
	void add_point(int64_t p_id, const Vector2 &p_position, float p_weight_scale = 1.0);
	Vector2 get_point_position(int64_t p_id) const;
	void set_point_position(int64_t p_id, const Vector2 &p_position);
	float get_point_weight_scale(int64_t p_id) const;
	void set_point_weight_scale(int64_t p_id, float p_weight_scale);
	void remove_point(int64_t p_id);
	bool has_point(int64_t p_id) const;
	PackedInt64Array get_point_connections(int64_t p_id);
	PackedInt64Array get_point_ids();
	void set_neighbor_filter_enabled(bool p_enabled);
	bool is_neighbor_filter_enabled() const;
	void set_point_disabled(int64_t p_id, bool p_disabled = true);
	bool is_point_disabled(int64_t p_id) const;
	void connect_points(int64_t p_id, int64_t p_to_id, bool p_bidirectional = true);
	void disconnect_points(int64_t p_id, int64_t p_to_id, bool p_bidirectional = true);
	bool are_points_connected(int64_t p_id, int64_t p_to_id, bool p_bidirectional = true) const;
	int64_t get_point_count() const;
	int64_t get_point_capacity() const;
	void reserve_space(int64_t p_num_nodes);
	void clear();
	int64_t get_closest_point(const Vector2 &p_to_position, bool p_include_disabled = false) const;
	Vector2 get_closest_position_in_segment(const Vector2 &p_to_position) const;
	PackedVector2Array get_point_path(int64_t p_from_id, int64_t p_to_id, bool p_allow_partial_path = false);
	PackedInt64Array get_id_path(int64_t p_from_id, int64_t p_to_id, bool p_allow_partial_path = false);
	virtual bool _filter_neighbor(int64_t p_from_id, int64_t p_neighbor_id) const;
	virtual float _estimate_cost(int64_t p_from_id, int64_t p_end_id) const;
	virtual float _compute_cost(int64_t p_from_id, int64_t p_to_id) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_filter_neighbor), decltype(&T::_filter_neighbor)>) {
			BIND_VIRTUAL_METHOD(T, _filter_neighbor, 2522259332);
		}
		if constexpr (!std::is_same_v<decltype(&B::_estimate_cost), decltype(&T::_estimate_cost)>) {
			BIND_VIRTUAL_METHOD(T, _estimate_cost, 3085491603);
		}
		if constexpr (!std::is_same_v<decltype(&B::_compute_cost), decltype(&T::_compute_cost)>) {
			BIND_VIRTUAL_METHOD(T, _compute_cost, 3085491603);
		}
	}

public:
};

} // namespace godot


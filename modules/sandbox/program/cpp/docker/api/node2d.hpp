/**************************************************************************/
/*  node2d.hpp                                                            */
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

#pragma once
#include "canvas_item.hpp"
struct Transform2D;

// Node2D: Contains 2D transformations.
// Such as: position, rotation, scale, and skew.
struct Node2D : public CanvasItem {
	/// @brief Construct a Node2D object from an existing in-scope Node object.
	/// @param addr The address of the Node2D object.
	constexpr Node2D(uint64_t addr) :
			CanvasItem(addr) {}
	constexpr Node2D(Object obj) :
			CanvasItem(obj) {}
	constexpr Node2D(Node node) :
			CanvasItem(node) {}

	/// @brief Construct a Node2D object from a path.
	/// @param path The path to the Node2D object.
	Node2D(std::string_view path) :
			CanvasItem(path) {}

	/// @brief Get the position of the node.
	/// @return The position of the node.
	Vector2 get_position() const;
	/// @brief Set the position of the node.
	/// @param value The new position of the node.
	void set_position(const Vector2 &value);

	/// @brief Get the rotation of the node.
	/// @return The rotation of the node.
	real_t get_rotation() const;
	/// @brief Set the rotation of the node.
	/// @param value The new rotation of the node.
	void set_rotation(real_t value);

	/// @brief Get the scale of the node.
	/// @return The scale of the node.
	Vector2 get_scale() const;
	/// @brief Set the scale of the node.
	/// @param value The new scale of the node.
	void set_scale(const Vector2 &value);

	/// @brief Get the skew of the node.
	/// @return The skew of the node.
	float get_skew() const;
	/// @brief Set the skew of the node.
	/// @param value The new skew of the node.
	void set_skew(const Variant &value);

	/// @brief Set the 2D transform of the node.
	/// @param value The new 2D transform of the node.
	void set_transform(const Transform2D &value);

	/// @brief Get the 2D transform of the node.
	/// @return The 2D transform of the node.
	Transform2D get_transform() const;

	/// @brief  Duplicate the node.
	/// @return A new Node2D with the same properties and children.
	Node2D duplicate(int flags = 15) const;

	/// @brief Create a new Node2D node.
	/// @param path The path to the Node2D node.
	/// @return The Node2D node.
	static Node2D Create(std::string_view path);
};

inline Node2D Variant::as_node2d() const {
	if (get_type() == Variant::OBJECT)
		return Node2D{ uintptr_t(v.i) };
	else if (get_type() == Variant::NODE_PATH)
		return Node2D{ this->internal_fetch_string() };

	api_throw("std::bad_cast", "Variant is not a Node2D or NodePath", this);
}

inline Node2D Object::as_node2d() const {
	return Node2D{ address() };
}

inline Variant::Variant(const Node2D &node) {
	m_type = OBJECT;
	v.i = node.address();
}

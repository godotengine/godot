/**************************************************************************/
/*  mesh_library.hpp                                                      */
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
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/transform3d.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Mesh;
class NavigationMesh;
class Texture2D;

class MeshLibrary : public Resource {
	GDEXTENSION_CLASS(MeshLibrary, Resource)

public:
	void create_item(int32_t p_id);
	void set_item_name(int32_t p_id, const String &p_name);
	void set_item_mesh(int32_t p_id, const Ref<Mesh> &p_mesh);
	void set_item_mesh_transform(int32_t p_id, const Transform3D &p_mesh_transform);
	void set_item_mesh_cast_shadow(int32_t p_id, RenderingServer::ShadowCastingSetting p_shadow_casting_setting);
	void set_item_navigation_mesh(int32_t p_id, const Ref<NavigationMesh> &p_navigation_mesh);
	void set_item_navigation_mesh_transform(int32_t p_id, const Transform3D &p_navigation_mesh);
	void set_item_navigation_layers(int32_t p_id, uint32_t p_navigation_layers);
	void set_item_shapes(int32_t p_id, const Array &p_shapes);
	void set_item_preview(int32_t p_id, const Ref<Texture2D> &p_texture);
	String get_item_name(int32_t p_id) const;
	Ref<Mesh> get_item_mesh(int32_t p_id) const;
	Transform3D get_item_mesh_transform(int32_t p_id) const;
	RenderingServer::ShadowCastingSetting get_item_mesh_cast_shadow(int32_t p_id) const;
	Ref<NavigationMesh> get_item_navigation_mesh(int32_t p_id) const;
	Transform3D get_item_navigation_mesh_transform(int32_t p_id) const;
	uint32_t get_item_navigation_layers(int32_t p_id) const;
	Array get_item_shapes(int32_t p_id) const;
	Ref<Texture2D> get_item_preview(int32_t p_id) const;
	void remove_item(int32_t p_id);
	int32_t find_item_by_name(const String &p_name) const;
	void clear();
	PackedInt32Array get_item_list() const;
	int32_t get_last_unused_item_id() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


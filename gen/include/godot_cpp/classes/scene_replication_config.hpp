/**************************************************************************/
/*  scene_replication_config.hpp                                          */
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
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class SceneReplicationConfig : public Resource {
	GDEXTENSION_CLASS(SceneReplicationConfig, Resource)

public:
	enum ReplicationMode {
		REPLICATION_MODE_NEVER = 0,
		REPLICATION_MODE_ALWAYS = 1,
		REPLICATION_MODE_ON_CHANGE = 2,
	};

	TypedArray<NodePath> get_properties() const;
	void add_property(const NodePath &p_path, int32_t p_index = -1);
	bool has_property(const NodePath &p_path) const;
	void remove_property(const NodePath &p_path);
	int32_t property_get_index(const NodePath &p_path) const;
	bool property_get_spawn(const NodePath &p_path);
	void property_set_spawn(const NodePath &p_path, bool p_enabled);
	SceneReplicationConfig::ReplicationMode property_get_replication_mode(const NodePath &p_path);
	void property_set_replication_mode(const NodePath &p_path, SceneReplicationConfig::ReplicationMode p_mode);
	bool property_get_sync(const NodePath &p_path);
	void property_set_sync(const NodePath &p_path, bool p_enabled);
	bool property_get_watch(const NodePath &p_path);
	void property_set_watch(const NodePath &p_path, bool p_enabled);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SceneReplicationConfig::ReplicationMode);


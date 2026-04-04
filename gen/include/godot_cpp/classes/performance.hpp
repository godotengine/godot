/**************************************************************************/
/*  performance.hpp                                                       */
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

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;

class Performance : public Object {
	GDEXTENSION_CLASS(Performance, Object)

	static Performance *singleton;

public:
	enum Monitor {
		TIME_FPS = 0,
		TIME_PROCESS = 1,
		TIME_PHYSICS_PROCESS = 2,
		TIME_NAVIGATION_PROCESS = 3,
		MEMORY_STATIC = 4,
		MEMORY_STATIC_MAX = 5,
		MEMORY_MESSAGE_BUFFER_MAX = 6,
		OBJECT_COUNT = 7,
		OBJECT_RESOURCE_COUNT = 8,
		OBJECT_NODE_COUNT = 9,
		OBJECT_ORPHAN_NODE_COUNT = 10,
		RENDER_TOTAL_OBJECTS_IN_FRAME = 11,
		RENDER_TOTAL_PRIMITIVES_IN_FRAME = 12,
		RENDER_TOTAL_DRAW_CALLS_IN_FRAME = 13,
		RENDER_VIDEO_MEM_USED = 14,
		RENDER_TEXTURE_MEM_USED = 15,
		RENDER_BUFFER_MEM_USED = 16,
		PHYSICS_2D_ACTIVE_OBJECTS = 17,
		PHYSICS_2D_COLLISION_PAIRS = 18,
		PHYSICS_2D_ISLAND_COUNT = 19,
		PHYSICS_3D_ACTIVE_OBJECTS = 20,
		PHYSICS_3D_COLLISION_PAIRS = 21,
		PHYSICS_3D_ISLAND_COUNT = 22,
		AUDIO_OUTPUT_LATENCY = 23,
		NAVIGATION_ACTIVE_MAPS = 24,
		NAVIGATION_REGION_COUNT = 25,
		NAVIGATION_AGENT_COUNT = 26,
		NAVIGATION_LINK_COUNT = 27,
		NAVIGATION_POLYGON_COUNT = 28,
		NAVIGATION_EDGE_COUNT = 29,
		NAVIGATION_EDGE_MERGE_COUNT = 30,
		NAVIGATION_EDGE_CONNECTION_COUNT = 31,
		NAVIGATION_EDGE_FREE_COUNT = 32,
		NAVIGATION_OBSTACLE_COUNT = 33,
		PIPELINE_COMPILATIONS_CANVAS = 34,
		PIPELINE_COMPILATIONS_MESH = 35,
		PIPELINE_COMPILATIONS_SURFACE = 36,
		PIPELINE_COMPILATIONS_DRAW = 37,
		PIPELINE_COMPILATIONS_SPECIALIZATION = 38,
		NAVIGATION_2D_ACTIVE_MAPS = 39,
		NAVIGATION_2D_REGION_COUNT = 40,
		NAVIGATION_2D_AGENT_COUNT = 41,
		NAVIGATION_2D_LINK_COUNT = 42,
		NAVIGATION_2D_POLYGON_COUNT = 43,
		NAVIGATION_2D_EDGE_COUNT = 44,
		NAVIGATION_2D_EDGE_MERGE_COUNT = 45,
		NAVIGATION_2D_EDGE_CONNECTION_COUNT = 46,
		NAVIGATION_2D_EDGE_FREE_COUNT = 47,
		NAVIGATION_2D_OBSTACLE_COUNT = 48,
		NAVIGATION_3D_ACTIVE_MAPS = 49,
		NAVIGATION_3D_REGION_COUNT = 50,
		NAVIGATION_3D_AGENT_COUNT = 51,
		NAVIGATION_3D_LINK_COUNT = 52,
		NAVIGATION_3D_POLYGON_COUNT = 53,
		NAVIGATION_3D_EDGE_COUNT = 54,
		NAVIGATION_3D_EDGE_MERGE_COUNT = 55,
		NAVIGATION_3D_EDGE_CONNECTION_COUNT = 56,
		NAVIGATION_3D_EDGE_FREE_COUNT = 57,
		NAVIGATION_3D_OBSTACLE_COUNT = 58,
		MONITOR_MAX = 59,
	};

	enum MonitorType {
		MONITOR_TYPE_QUANTITY = 0,
		MONITOR_TYPE_MEMORY = 1,
		MONITOR_TYPE_TIME = 2,
		MONITOR_TYPE_PERCENTAGE = 3,
	};

	static Performance *get_singleton();

	double get_monitor(Performance::Monitor p_monitor) const;
	void add_custom_monitor(const StringName &p_id, const Callable &p_callable, const Array &p_arguments = Array(), Performance::MonitorType p_type = (Performance::MonitorType)0);
	void remove_custom_monitor(const StringName &p_id);
	bool has_custom_monitor(const StringName &p_id);
	Variant get_custom_monitor(const StringName &p_id);
	uint64_t get_monitor_modification_time();
	TypedArray<StringName> get_custom_monitor_names();
	PackedInt32Array get_custom_monitor_types();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~Performance();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Performance::Monitor);
VARIANT_ENUM_CAST(Performance::MonitorType);


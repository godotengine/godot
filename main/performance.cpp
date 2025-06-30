/**************************************************************************/
/*  performance.cpp                                                       */
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

#include "performance.h"

#include "core/message_queue.h"
#include "core/os/os.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "servers/audio_server.h"
#include "servers/physics_2d_server.h"
#include "servers/physics_server.h"
#include "servers/visual_server.h"

Performance *Performance::singleton = nullptr;

void Performance::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_monitor", "monitor"), &Performance::get_monitor);

	BIND_ENUM_CONSTANT(TIME_FPS);
	BIND_ENUM_CONSTANT(TIME_PROCESS);
	BIND_ENUM_CONSTANT(TIME_PHYSICS_PROCESS);
	BIND_ENUM_CONSTANT(MEMORY_STATIC);
	BIND_ENUM_CONSTANT(MEMORY_DYNAMIC);
	BIND_ENUM_CONSTANT(MEMORY_STATIC_MAX);
	BIND_ENUM_CONSTANT(MEMORY_DYNAMIC_MAX);
	BIND_ENUM_CONSTANT(MEMORY_MESSAGE_BUFFER_MAX);
	BIND_ENUM_CONSTANT(OBJECT_COUNT);
	BIND_ENUM_CONSTANT(OBJECT_RESOURCE_COUNT);
	BIND_ENUM_CONSTANT(OBJECT_NODE_COUNT);
	BIND_ENUM_CONSTANT(OBJECT_ORPHAN_NODE_COUNT);
	BIND_ENUM_CONSTANT(RENDER_OBJECTS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_VERTICES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_MATERIAL_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_SHADER_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_SURFACE_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_DRAW_CALLS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_2D_ITEMS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_2D_DRAW_CALLS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_VIDEO_MEM_USED);
	BIND_ENUM_CONSTANT(RENDER_TEXTURE_MEM_USED);
	BIND_ENUM_CONSTANT(RENDER_VERTEX_MEM_USED);
	BIND_ENUM_CONSTANT(RENDER_USAGE_VIDEO_MEM_TOTAL);
	BIND_ENUM_CONSTANT(PHYSICS_2D_ACTIVE_OBJECTS);
	BIND_ENUM_CONSTANT(PHYSICS_2D_COLLISION_PAIRS);
	BIND_ENUM_CONSTANT(PHYSICS_2D_ISLAND_COUNT);
	BIND_ENUM_CONSTANT(PHYSICS_3D_ACTIVE_OBJECTS);
	BIND_ENUM_CONSTANT(PHYSICS_3D_COLLISION_PAIRS);
	BIND_ENUM_CONSTANT(PHYSICS_3D_ISLAND_COUNT);
	BIND_ENUM_CONSTANT(AUDIO_OUTPUT_LATENCY);

	BIND_ENUM_CONSTANT(MONITOR_MAX);
}

float Performance::_get_node_count() const {
	MainLoop *ml = OS::get_singleton()->get_main_loop();
	SceneTree *sml = Object::cast_to<SceneTree>(ml);
	if (!sml) {
		return 0;
	}
	return sml->get_node_count();
}

String Performance::get_monitor_name(Monitor p_monitor) const {
	ERR_FAIL_INDEX_V(p_monitor, MONITOR_MAX, String());
	static const char *names[MONITOR_MAX] = {

		"time/fps",
		"time/process",
		"time/physics_process",
		"memory/static",
		"memory/dynamic",
		"memory/static_max",
		"memory/dynamic_max",
		"memory/msg_buf_max",
		"object/objects",
		"object/resources",
		"object/nodes",
		"object/orphan_nodes",
		"raster/objects_drawn",
		"raster/vertices_drawn",
		"raster/mat_changes",
		"raster/shader_changes",
		"raster/surface_changes",
		"raster/draw_calls",
		"2d/items",
		"2d/draw_calls",
		"video/video_mem",
		"video/texture_mem",
		"video/vertex_mem",
		"video/video_mem_max",
		"physics_2d/active_objects",
		"physics_2d/collision_pairs",
		"physics_2d/islands",
		"physics_3d/active_objects",
		"physics_3d/collision_pairs",
		"physics_3d/islands",
		"audio/output_latency",

	};

	return names[p_monitor];
}

float Performance::get_monitor(Monitor p_monitor) const {
	switch (p_monitor) {
		case TIME_FPS:
			return Engine::get_singleton()->get_frames_per_second();
		case TIME_PROCESS:
			return _process_time;
		case TIME_PHYSICS_PROCESS:
			return _physics_process_time;
		case MEMORY_STATIC:
			return Memory::get_mem_usage();
		case MEMORY_DYNAMIC:
			return 0;
		case MEMORY_STATIC_MAX:
			return Memory::get_mem_max_usage();
		case MEMORY_DYNAMIC_MAX:
			return 0;
		case MEMORY_MESSAGE_BUFFER_MAX:
			return MessageQueue::get_singleton()->get_max_buffer_usage();
		case OBJECT_COUNT:
			return ObjectDB::get_object_count();
		case OBJECT_RESOURCE_COUNT:
			return ResourceCache::get_cached_resource_count();
		case OBJECT_NODE_COUNT:
			return _get_node_count();
		case OBJECT_ORPHAN_NODE_COUNT:
			return Node::orphan_node_count;
		case RENDER_OBJECTS_IN_FRAME:
			return VS::get_singleton()->get_render_info(VS::INFO_OBJECTS_IN_FRAME);
		case RENDER_VERTICES_IN_FRAME:
			return VS::get_singleton()->get_render_info(VS::INFO_VERTICES_IN_FRAME);
		case RENDER_MATERIAL_CHANGES_IN_FRAME:
			return VS::get_singleton()->get_render_info(VS::INFO_MATERIAL_CHANGES_IN_FRAME);
		case RENDER_SHADER_CHANGES_IN_FRAME:
			return VS::get_singleton()->get_render_info(VS::INFO_SHADER_CHANGES_IN_FRAME);
		case RENDER_SURFACE_CHANGES_IN_FRAME:
			return VS::get_singleton()->get_render_info(VS::INFO_SURFACE_CHANGES_IN_FRAME);
		case RENDER_DRAW_CALLS_IN_FRAME:
			return VS::get_singleton()->get_render_info(VS::INFO_DRAW_CALLS_IN_FRAME);
		case RENDER_2D_ITEMS_IN_FRAME:
			return VS::get_singleton()->get_render_info(VS::INFO_2D_ITEMS_IN_FRAME);
		case RENDER_2D_DRAW_CALLS_IN_FRAME:
			return VS::get_singleton()->get_render_info(VS::INFO_2D_DRAW_CALLS_IN_FRAME);
		case RENDER_VIDEO_MEM_USED:
			return VS::get_singleton()->get_render_info(VS::INFO_VIDEO_MEM_USED);
		case RENDER_TEXTURE_MEM_USED:
			return VS::get_singleton()->get_render_info(VS::INFO_TEXTURE_MEM_USED);
		case RENDER_VERTEX_MEM_USED:
			return VS::get_singleton()->get_render_info(VS::INFO_VERTEX_MEM_USED);
		case RENDER_USAGE_VIDEO_MEM_TOTAL:
			return VS::get_singleton()->get_render_info(VS::INFO_USAGE_VIDEO_MEM_TOTAL);
		case PHYSICS_2D_ACTIVE_OBJECTS:
			return Physics2DServer::get_singleton()->get_process_info(Physics2DServer::INFO_ACTIVE_OBJECTS);
		case PHYSICS_2D_COLLISION_PAIRS:
			return Physics2DServer::get_singleton()->get_process_info(Physics2DServer::INFO_COLLISION_PAIRS);
		case PHYSICS_2D_ISLAND_COUNT:
			return Physics2DServer::get_singleton()->get_process_info(Physics2DServer::INFO_ISLAND_COUNT);
		case PHYSICS_3D_ACTIVE_OBJECTS:
			return PhysicsServer::get_singleton()->get_process_info(PhysicsServer::INFO_ACTIVE_OBJECTS);
		case PHYSICS_3D_COLLISION_PAIRS:
			return PhysicsServer::get_singleton()->get_process_info(PhysicsServer::INFO_COLLISION_PAIRS);
		case PHYSICS_3D_ISLAND_COUNT:
			return PhysicsServer::get_singleton()->get_process_info(PhysicsServer::INFO_ISLAND_COUNT);
		case AUDIO_OUTPUT_LATENCY:
			return AudioServer::get_singleton()->get_output_latency();

		default: {
		}
	}

	return 0;
}

Performance::MonitorType Performance::get_monitor_type(Monitor p_monitor) const {
	ERR_FAIL_INDEX_V(p_monitor, MONITOR_MAX, MONITOR_TYPE_QUANTITY);
	// ugly
	static const MonitorType types[MONITOR_MAX] = {

		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_TIME,
		MONITOR_TYPE_TIME,
		MONITOR_TYPE_MEMORY,
		MONITOR_TYPE_MEMORY,
		MONITOR_TYPE_MEMORY,
		MONITOR_TYPE_MEMORY,
		MONITOR_TYPE_MEMORY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_MEMORY,
		MONITOR_TYPE_MEMORY,
		MONITOR_TYPE_MEMORY,
		MONITOR_TYPE_MEMORY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_QUANTITY,
		MONITOR_TYPE_TIME,

	};

	return types[p_monitor];
}

void Performance::set_process_time(float p_pt) {
	_process_time = p_pt;
}

void Performance::set_physics_process_time(float p_pt) {
	_physics_process_time = p_pt;
}

Performance::Performance() {
	_process_time = 0;
	_physics_process_time = 0;
	singleton = this;
}

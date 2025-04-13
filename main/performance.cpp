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

#include "core/os/os.h"
#include "core/variant/typed_array.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "servers/audio_server.h"
#ifndef NAVIGATION_2D_DISABLED
#include "servers/navigation_server_2d.h"
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
#include "servers/navigation_server_3d.h"
#endif // NAVIGATION_3D_DISABLED
#include "servers/rendering_server.h"

#ifndef PHYSICS_2D_DISABLED
#include "servers/physics_server_2d.h"
#endif // PHYSICS_2D_DISABLED

#ifndef PHYSICS_3D_DISABLED
#include "servers/physics_server_3d.h"
#endif // PHYSICS_3D_DISABLED

Performance *Performance::singleton = nullptr;

void Performance::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_monitor", "monitor"), &Performance::get_monitor);
	ClassDB::bind_method(D_METHOD("add_custom_monitor", "id", "callable", "arguments"), &Performance::add_custom_monitor, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("remove_custom_monitor", "id"), &Performance::remove_custom_monitor);
	ClassDB::bind_method(D_METHOD("has_custom_monitor", "id"), &Performance::has_custom_monitor);
	ClassDB::bind_method(D_METHOD("get_custom_monitor", "id"), &Performance::get_custom_monitor);
	ClassDB::bind_method(D_METHOD("get_monitor_modification_time"), &Performance::get_monitor_modification_time);
	ClassDB::bind_method(D_METHOD("get_custom_monitor_names"), &Performance::get_custom_monitor_names);

	BIND_ENUM_CONSTANT(TIME_FPS);
	BIND_ENUM_CONSTANT(TIME_PROCESS);
	BIND_ENUM_CONSTANT(TIME_PHYSICS_PROCESS);
	BIND_ENUM_CONSTANT(TIME_NAVIGATION_PROCESS);
	BIND_ENUM_CONSTANT(MEMORY_STATIC);
	BIND_ENUM_CONSTANT(MEMORY_STATIC_MAX);
	BIND_ENUM_CONSTANT(MEMORY_MESSAGE_BUFFER_MAX);
	BIND_ENUM_CONSTANT(OBJECT_COUNT);
	BIND_ENUM_CONSTANT(OBJECT_RESOURCE_COUNT);
	BIND_ENUM_CONSTANT(OBJECT_NODE_COUNT);
	BIND_ENUM_CONSTANT(OBJECT_ORPHAN_NODE_COUNT);
	BIND_ENUM_CONSTANT(RENDER_TOTAL_OBJECTS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_TOTAL_PRIMITIVES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_TOTAL_DRAW_CALLS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_VIDEO_MEM_USED);
	BIND_ENUM_CONSTANT(RENDER_TEXTURE_MEM_USED);
	BIND_ENUM_CONSTANT(RENDER_BUFFER_MEM_USED);
	BIND_ENUM_CONSTANT(PHYSICS_2D_ACTIVE_OBJECTS);
	BIND_ENUM_CONSTANT(PHYSICS_2D_COLLISION_PAIRS);
	BIND_ENUM_CONSTANT(PHYSICS_2D_ISLAND_COUNT);
#ifndef _3D_DISABLED
	BIND_ENUM_CONSTANT(PHYSICS_3D_ACTIVE_OBJECTS);
	BIND_ENUM_CONSTANT(PHYSICS_3D_COLLISION_PAIRS);
	BIND_ENUM_CONSTANT(PHYSICS_3D_ISLAND_COUNT);
#endif // _3D_DISABLED
	BIND_ENUM_CONSTANT(AUDIO_OUTPUT_LATENCY);
#if !defined(NAVIGATION_2D_DISABLED) || !defined(NAVIGATION_3D_DISABLED)
	BIND_ENUM_CONSTANT(NAVIGATION_ACTIVE_MAPS);
	BIND_ENUM_CONSTANT(NAVIGATION_REGION_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_AGENT_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_LINK_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_POLYGON_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_EDGE_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_EDGE_MERGE_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_EDGE_CONNECTION_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_EDGE_FREE_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_OBSTACLE_COUNT);
#endif // !defined(NAVIGATION_2D_DISABLED) || !defined(NAVIGATION_3D_DISABLED)
	BIND_ENUM_CONSTANT(PIPELINE_COMPILATIONS_CANVAS);
	BIND_ENUM_CONSTANT(PIPELINE_COMPILATIONS_MESH);
	BIND_ENUM_CONSTANT(PIPELINE_COMPILATIONS_SURFACE);
	BIND_ENUM_CONSTANT(PIPELINE_COMPILATIONS_DRAW);
	BIND_ENUM_CONSTANT(PIPELINE_COMPILATIONS_SPECIALIZATION);
#ifndef NAVIGATION_2D_DISABLED
	BIND_ENUM_CONSTANT(NAVIGATION_2D_ACTIVE_MAPS);
	BIND_ENUM_CONSTANT(NAVIGATION_2D_REGION_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_2D_AGENT_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_2D_LINK_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_2D_POLYGON_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_2D_EDGE_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_2D_EDGE_MERGE_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_2D_EDGE_CONNECTION_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_2D_EDGE_FREE_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_2D_OBSTACLE_COUNT);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
	BIND_ENUM_CONSTANT(NAVIGATION_3D_ACTIVE_MAPS);
	BIND_ENUM_CONSTANT(NAVIGATION_3D_REGION_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_3D_AGENT_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_3D_LINK_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_3D_POLYGON_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_3D_EDGE_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_3D_EDGE_MERGE_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_3D_EDGE_CONNECTION_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_3D_EDGE_FREE_COUNT);
	BIND_ENUM_CONSTANT(NAVIGATION_3D_OBSTACLE_COUNT);
#endif // NAVIGATION_3D_DISABLED
	BIND_ENUM_CONSTANT(TEXTSERVER_BUFFER_COUNT);
	BIND_ENUM_CONSTANT(TEXTSERVER_FONT_COUNT);
	BIND_ENUM_CONSTANT(TEXTSERVER_FONT_VARIATION_COUNT);
	BIND_ENUM_CONSTANT(TEXTSERVER_BUFFER_GLYPH_COUNT);
	BIND_ENUM_CONSTANT(TEXTSERVER_FONT_GLYPH_COUNT);
	BIND_ENUM_CONSTANT(TEXTSERVER_FONT_TEXTURE_BYTES);
	BIND_ENUM_CONSTANT(TEXTSERVER_FONT_OVERSAMPLED_GLYPH_COUNT);
	BIND_ENUM_CONSTANT(TEXTSERVER_FONT_OVERSAMPLED_TEXTURE_BYTES);
	BIND_ENUM_CONSTANT(TEXTSERVER_FONT_OVERSAMPLING_LEVELS);
	BIND_ENUM_CONSTANT(TEXTSERVER_FONT_SYSTEM_FONT_DATA);
	BIND_ENUM_CONSTANT(MONITOR_MAX);
}

int Performance::_get_node_count() const {
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
		PNAME("time/fps"), // TIME_FPS
		PNAME("time/process"), // TIME_PROCESS
		PNAME("time/physics_process"), // TIME_PHYSICS_PROCESS
		PNAME("time/navigation_process"), // TIME_NAVIGATION_PROCESS
		PNAME("memory/static"), // MEMORY_STATIC
		PNAME("memory/static_max"), // MEMORY_STATIC_MAX
		PNAME("memory/msg_buf_max"), // MEMORY_MESSAGE_BUFFER_MAX
		PNAME("object/objects"), // OBJECT_COUNT
		PNAME("object/resources"), // OBJECT_RESOURCE_COUNT
		PNAME("object/nodes"), // OBJECT_NODE_COUNT
		PNAME("object/orphan_nodes"), // OBJECT_ORPHAN_NODE_COUNT
		PNAME("raster/total_objects_drawn"), // RENDER_TOTAL_OBJECTS_IN_FRAME
		PNAME("raster/total_primitives_drawn"), // RENDER_TOTAL_PRIMITIVES_IN_FRAME
		PNAME("raster/total_draw_calls"), // RENDER_TOTAL_DRAW_CALLS_IN_FRAME
		PNAME("video/video_mem"), // RENDER_VIDEO_MEM_USED
		PNAME("video/texture_mem"), // RENDER_TEXTURE_MEM_USED
		PNAME("video/buffer_mem"), // RENDER_BUFFER_MEM_USED
		PNAME("physics_2d/active_objects"), // PHYSICS_2D_ACTIVE_OBJECTS
		PNAME("physics_2d/collision_pairs"), // PHYSICS_2D_COLLISION_PAIRS
		PNAME("physics_2d/islands"), // PHYSICS_2D_ISLAND_COUNT
		PNAME("physics_3d/active_objects"), // PHYSICS_3D_ACTIVE_OBJECTS
		PNAME("physics_3d/collision_pairs"), // PHYSICS_3D_COLLISION_PAIRS
		PNAME("physics_3d/islands"), // PHYSICS_3D_ISLAND_COUNT
		PNAME("audio/driver/output_latency"), // AUDIO_OUTPUT_LATENCY
#if !defined(NAVIGATION_2D_DISABLED) || !defined(NAVIGATION_3D_DISABLED)
		PNAME("navigation/active_maps"), // NAVIGATION_ACTIVE_MAPS
		PNAME("navigation/regions"), // NAVIGATION_REGION_COUNT
		PNAME("navigation/agents"), // NAVIGATION_AGENT_COUNT
		PNAME("navigation/links"), // NAVIGATION_LINK_COUNT
		PNAME("navigation/polygons"), // NAVIGATION_POLYGON_COUNT
		PNAME("navigation/edges"), // NAVIGATION_EDGE_COUNT
		PNAME("navigation/edges_merged"), // NAVIGATION_EDGE_MERGE_COUNT
		PNAME("navigation/edges_connected"), // NAVIGATION_EDGE_CONNECTION_COUNT
		PNAME("navigation/edges_free"), // NAVIGATION_EDGE_FREE_COUNT
		PNAME("navigation/obstacles"), // NAVIGATION_OBSTACLE_COUNT
#endif // !defined(NAVIGATION_2D_DISABLED) || !defined(NAVIGATION_3D_DISABLED)
		PNAME("pipeline/compilations_canvas"), // PIPELINE_COMPILATIONS_CANVAS
		PNAME("pipeline/compilations_mesh"), // PIPELINE_COMPILATIONS_MESH
		PNAME("pipeline/compilations_surface"), // PIPELINE_COMPILATIONS_SURFACE
		PNAME("pipeline/compilations_draw"), // PIPELINE_COMPILATIONS_DRAW
		PNAME("pipeline/compilations_specialization"), // PIPELINE_COMPILATIONS_SPECIALIZATION
#ifndef NAVIGATION_2D_DISABLED
		PNAME("navigation_2d/active_maps"), // NAVIGATION_2D_ACTIVE_MAPS
		PNAME("navigation_2d/regions"), // NAVIGATION_2D_REGION_COUNT
		PNAME("navigation_2d/agents"), // NAVIGATION_2D_AGENT_COUNT
		PNAME("navigation_2d/links"), // NAVIGATION_2D_LINK_COUNT
		PNAME("navigation_2d/polygons"), // NAVIGATION_2D_POLYGON_COUNT
		PNAME("navigation_2d/edges"), // NAVIGATION_2D_EDGE_COUNT
		PNAME("navigation_2d/edges_merged"), // NAVIGATION_2D_EDGE_MERGE_COUNT
		PNAME("navigation_2d/edges_connected"), // NAVIGATION_2D_EDGE_CONNECTION_COUNT
		PNAME("navigation_2d/edges_free"), // NAVIGATION_2D_EDGE_FREE_COUNT
		PNAME("navigation_2d/obstacles"), // NAVIGATION_2D_OBSTACLE_COUNT
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
		PNAME("navigation_3d/active_maps"), // NAVIGATION_3D_ACTIVE_MAPS
		PNAME("navigation_3d/regions"), // NAVIGATION_3D_REGION_COUNT
		PNAME("navigation_3d/agents"), // NAVIGATION_3D_AGENT_COUNT
		PNAME("navigation_3d/links"), // NAVIGATION_3D_LINK_COUNT
		PNAME("navigation_3d/polygons"), // NAVIGATION_3D_POLYGON_COUNT
		PNAME("navigation_3d/edges"), // NAVIGATION_3D_EDGE_COUNT
		PNAME("navigation_3d/edges_merged"), // NAVIGATION_3D_EDGE_MERGE_COUNT
		PNAME("navigation_3d/edges_connected"), // NAVIGATION_3D_EDGE_CONNECTION_COUNT
		PNAME("navigation_3d/edges_free"), // NAVIGATION_3D_EDGE_FREE_COUNT
		PNAME("navigation_3d/obstacles"), // NAVIGATION_3D_OBSTACLE_COUNT
#endif // NAVIGATION_3D_DISABLED
		PNAME("text_server/buffers"), // TEXTSERVER_BUFFER_COUNT,
		PNAME("text_server/fonts"), // TEXTSERVER_FONT_COUNT,
		PNAME("text_server/font_variations"), // TEXTSERVER_FONT_VARIATION_COUNT,
		PNAME("text_server/buffer_glyphs"), // TEXTSERVER_BUFFERGLYPH_COUNT,
		PNAME("text_server/font_total_glyphs"), // TEXTSERVER_FONT_GLYPH_COUNT
		PNAME("text_server/font_total_textures"), // TEXTSERVER_FONT_TEXTURE_BYTES
		PNAME("text_server/font_oversampled_glyphs"), // TEXTSERVER_FONT_OVERSAMPLED_GLYPH_COUNT
		PNAME("text_server/font_oversampled_textures"), // TEXTSERVER_FONT_OVERSAMPLED_TEXTURE_BYTES
		PNAME("text_server/font_oversampling_levels"), // TEXTSERVER_FONT_OVERSAMPLING_LEVELS
		PNAME("text_server/system_font_data"), // TEXTSERVER_FONT_SYSTEM_FONT_DATA
	};
	static_assert(std::size(names) == MONITOR_MAX);

	return names[p_monitor];
}

double Performance::get_monitor(Monitor p_monitor) const {
	int info = 0;

	switch (p_monitor) {
		case TIME_FPS:
			return Engine::get_singleton()->get_frames_per_second();
		case TIME_PROCESS:
			return _process_time;
		case TIME_PHYSICS_PROCESS:
			return _physics_process_time;
		case TIME_NAVIGATION_PROCESS:
			return _navigation_process_time;
		case MEMORY_STATIC:
			return Memory::get_mem_usage();
		case MEMORY_STATIC_MAX:
			return Memory::get_mem_max_usage();
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
		case RENDER_TOTAL_OBJECTS_IN_FRAME:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_TOTAL_OBJECTS_IN_FRAME);
		case RENDER_TOTAL_PRIMITIVES_IN_FRAME:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME);
		case RENDER_TOTAL_DRAW_CALLS_IN_FRAME:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME);
		case RENDER_VIDEO_MEM_USED:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_VIDEO_MEM_USED);
		case RENDER_TEXTURE_MEM_USED:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_TEXTURE_MEM_USED);
		case RENDER_BUFFER_MEM_USED:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_BUFFER_MEM_USED);
		case PIPELINE_COMPILATIONS_CANVAS:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_PIPELINE_COMPILATIONS_CANVAS);
		case PIPELINE_COMPILATIONS_MESH:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_PIPELINE_COMPILATIONS_MESH);
		case PIPELINE_COMPILATIONS_SURFACE:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_PIPELINE_COMPILATIONS_SURFACE);
		case PIPELINE_COMPILATIONS_DRAW:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_PIPELINE_COMPILATIONS_DRAW);
		case PIPELINE_COMPILATIONS_SPECIALIZATION:
			return RS::get_singleton()->get_rendering_info(RS::RENDERING_INFO_PIPELINE_COMPILATIONS_SPECIALIZATION);
#ifndef PHYSICS_2D_DISABLED
		case PHYSICS_2D_ACTIVE_OBJECTS:
			return PhysicsServer2D::get_singleton()->get_process_info(PhysicsServer2D::INFO_ACTIVE_OBJECTS);
		case PHYSICS_2D_COLLISION_PAIRS:
			return PhysicsServer2D::get_singleton()->get_process_info(PhysicsServer2D::INFO_COLLISION_PAIRS);
		case PHYSICS_2D_ISLAND_COUNT:
			return PhysicsServer2D::get_singleton()->get_process_info(PhysicsServer2D::INFO_ISLAND_COUNT);
#else
		case PHYSICS_2D_ACTIVE_OBJECTS:
			return 0;
		case PHYSICS_2D_COLLISION_PAIRS:
			return 0;
		case PHYSICS_2D_ISLAND_COUNT:
			return 0;
#endif // PHYSICS_2D_DISABLED
#ifndef PHYSICS_3D_DISABLED
		case PHYSICS_3D_ACTIVE_OBJECTS:
			return PhysicsServer3D::get_singleton()->get_process_info(PhysicsServer3D::INFO_ACTIVE_OBJECTS);
		case PHYSICS_3D_COLLISION_PAIRS:
			return PhysicsServer3D::get_singleton()->get_process_info(PhysicsServer3D::INFO_COLLISION_PAIRS);
		case PHYSICS_3D_ISLAND_COUNT:
			return PhysicsServer3D::get_singleton()->get_process_info(PhysicsServer3D::INFO_ISLAND_COUNT);
#else
		case PHYSICS_3D_ACTIVE_OBJECTS:
			return 0;
		case PHYSICS_3D_COLLISION_PAIRS:
			return 0;
		case PHYSICS_3D_ISLAND_COUNT:
			return 0;
#endif // PHYSICS_3D_DISABLED

		case AUDIO_OUTPUT_LATENCY:
			return AudioServer::get_singleton()->get_output_latency();

		case NAVIGATION_ACTIVE_MAPS:
#ifndef NAVIGATION_2D_DISABLED
			info = NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_ACTIVE_MAPS);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
			info += NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_ACTIVE_MAPS);
#endif // NAVIGATION_3D_DISABLED
			return info;

		case NAVIGATION_REGION_COUNT:
#ifndef NAVIGATION_2D_DISABLED
			info = NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_REGION_COUNT);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
			info += NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_REGION_COUNT);
#endif // NAVIGATION_3D_DISABLED
			return info;

		case NAVIGATION_AGENT_COUNT:
#ifndef NAVIGATION_2D_DISABLED
			info = NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_AGENT_COUNT);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
			info += NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_AGENT_COUNT);
#endif // NAVIGATION_3D_DISABLED
			return info;

		case NAVIGATION_LINK_COUNT:
#ifndef NAVIGATION_2D_DISABLED
			info = NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_LINK_COUNT);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
			info += NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_LINK_COUNT);
#endif // NAVIGATION_3D_DISABLED
			return info;

		case NAVIGATION_POLYGON_COUNT:
#ifndef NAVIGATION_2D_DISABLED
			info = NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_POLYGON_COUNT);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
			info += NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_POLYGON_COUNT);
#endif // NAVIGATION_3D_DISABLED
			return info;

		case NAVIGATION_EDGE_COUNT:
#ifndef NAVIGATION_2D_DISABLED
			info = NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_EDGE_COUNT);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
			info += NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_EDGE_COUNT);
#endif // NAVIGATION_3D_DISABLED
			return info;

		case NAVIGATION_EDGE_MERGE_COUNT:
#ifndef NAVIGATION_2D_DISABLED
			info = NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_EDGE_MERGE_COUNT);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
			info += NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_EDGE_MERGE_COUNT);
#endif // NAVIGATION_3D_DISABLED
			return info;

		case NAVIGATION_EDGE_CONNECTION_COUNT:
#ifndef NAVIGATION_2D_DISABLED
			info = NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_EDGE_CONNECTION_COUNT);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
			info += NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_EDGE_CONNECTION_COUNT);
#endif // NAVIGATION_3D_DISABLED
			return info;

		case NAVIGATION_EDGE_FREE_COUNT:
#ifndef NAVIGATION_2D_DISABLED
			info = NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_EDGE_FREE_COUNT);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
			info += NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_EDGE_FREE_COUNT);
#endif // NAVIGATION_3D_DISABLED
			return info;

		case NAVIGATION_OBSTACLE_COUNT:
#ifndef NAVIGATION_2D_DISABLED
			info = NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_OBSTACLE_COUNT);
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
			info += NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_OBSTACLE_COUNT);
#endif // NAVIGATION_3D_DISABLED
			return info;

#ifndef NAVIGATION_2D_DISABLED
		case NAVIGATION_2D_ACTIVE_MAPS:
			return NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_ACTIVE_MAPS);
		case NAVIGATION_2D_REGION_COUNT:
			return NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_REGION_COUNT);
		case NAVIGATION_2D_AGENT_COUNT:
			return NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_AGENT_COUNT);
		case NAVIGATION_2D_LINK_COUNT:
			return NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_LINK_COUNT);
		case NAVIGATION_2D_POLYGON_COUNT:
			return NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_POLYGON_COUNT);
		case NAVIGATION_2D_EDGE_COUNT:
			return NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_EDGE_COUNT);
		case NAVIGATION_2D_EDGE_MERGE_COUNT:
			return NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_EDGE_MERGE_COUNT);
		case NAVIGATION_2D_EDGE_CONNECTION_COUNT:
			return NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_EDGE_CONNECTION_COUNT);
		case NAVIGATION_2D_EDGE_FREE_COUNT:
			return NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_EDGE_FREE_COUNT);
		case NAVIGATION_2D_OBSTACLE_COUNT:
			return NavigationServer2D::get_singleton()->get_process_info(NavigationServer2D::INFO_OBSTACLE_COUNT);
#endif // NAVIGATION_2D_DISABLED

#ifndef NAVIGATION_3D_DISABLED
		case NAVIGATION_3D_ACTIVE_MAPS:
			return NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_ACTIVE_MAPS);
		case NAVIGATION_3D_REGION_COUNT:
			return NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_REGION_COUNT);
		case NAVIGATION_3D_AGENT_COUNT:
			return NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_AGENT_COUNT);
		case NAVIGATION_3D_LINK_COUNT:
			return NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_LINK_COUNT);
		case NAVIGATION_3D_POLYGON_COUNT:
			return NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_POLYGON_COUNT);
		case NAVIGATION_3D_EDGE_COUNT:
			return NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_EDGE_COUNT);
		case NAVIGATION_3D_EDGE_MERGE_COUNT:
			return NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_EDGE_MERGE_COUNT);
		case NAVIGATION_3D_EDGE_CONNECTION_COUNT:
			return NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_EDGE_CONNECTION_COUNT);
		case NAVIGATION_3D_EDGE_FREE_COUNT:
			return NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_EDGE_FREE_COUNT);
		case NAVIGATION_3D_OBSTACLE_COUNT:
			return NavigationServer3D::get_singleton()->get_process_info(NavigationServer3D::INFO_OBSTACLE_COUNT);
#endif // NAVIGATION_3D_DISABLED
		case TEXTSERVER_BUFFER_COUNT:
			return TS->get_process_info(TextServer::BUFFER_COUNT);
		case TEXTSERVER_FONT_COUNT:
			return TS->get_process_info(TextServer::FONT_COUNT);
		case TEXTSERVER_FONT_VARIATION_COUNT:
			return TS->get_process_info(TextServer::FONT_VARIATION_COUNT);
		case TEXTSERVER_BUFFER_GLYPH_COUNT:
			return TS->get_process_info(TextServer::BUFFER_GLYPH_COUNT);
		case TEXTSERVER_FONT_GLYPH_COUNT:
			return TS->get_process_info(TextServer::FONT_GLYPH_COUNT);
		case TEXTSERVER_FONT_TEXTURE_BYTES:
			return TS->get_process_info(TextServer::FONT_TEXTURE_BYTES);
		case TEXTSERVER_FONT_OVERSAMPLED_GLYPH_COUNT:
			return TS->get_process_info(TextServer::FONT_OVERSAMPLED_GLYPH_COUNT);
		case TEXTSERVER_FONT_OVERSAMPLED_TEXTURE_BYTES:
			return TS->get_process_info(TextServer::FONT_OVERSAMPLED_TEXTURE_BYTES);
		case TEXTSERVER_FONT_OVERSAMPLING_LEVELS:
			return TS->get_process_info(TextServer::FONT_OVERSAMPLING_LEVELS);
		case TEXTSERVER_FONT_SYSTEM_FONT_DATA:
			return TS->get_process_info(TextServer::FONT_SYSTEM_FONT_DATA);
		default: {
		}
	}

	return 0;
}

Performance::MonitorType Performance::get_monitor_type(Monitor p_monitor) const {
	ERR_FAIL_INDEX_V(p_monitor, MONITOR_MAX, MONITOR_TYPE_QUANTITY);
	// ugly
	static const MonitorType types[MONITOR_MAX] = {
		MONITOR_TYPE_QUANTITY, // TIME_FPS
		MONITOR_TYPE_TIME, // TIME_PROCESS
		MONITOR_TYPE_TIME, // TIME_PHYSICS_PROCESS
		MONITOR_TYPE_TIME, // TIME_NAVIGATION_PROCESS
		MONITOR_TYPE_MEMORY, // MEMORY_STATIC
		MONITOR_TYPE_MEMORY, // MEMORY_STATIC_MAX
		MONITOR_TYPE_MEMORY, // MEMORY_MESSAGE_BUFFER_MAX
		MONITOR_TYPE_QUANTITY, // OBJECT_COUNT
		MONITOR_TYPE_QUANTITY, // OBJECT_RESOURCE_COUNT
		MONITOR_TYPE_QUANTITY, // OBJECT_NODE_COUNT
		MONITOR_TYPE_QUANTITY, // OBJECT_ORPHAN_NODE_COUNT
		MONITOR_TYPE_QUANTITY, // RENDER_TOTAL_OBJECTS_IN_FRAME
		MONITOR_TYPE_QUANTITY, // RENDER_TOTAL_PRIMITIVES_IN_FRAME
		MONITOR_TYPE_QUANTITY, // RENDER_TOTAL_DRAW_CALLS_IN_FRAME
		MONITOR_TYPE_MEMORY, // RENDER_VIDEO_MEM_USED
		MONITOR_TYPE_MEMORY, // RENDER_TEXTURE_MEM_USED
		MONITOR_TYPE_MEMORY, // RENDER_BUFFER_MEM_USED
		MONITOR_TYPE_QUANTITY, // PHYSICS_2D_ACTIVE_OBJECTS
		MONITOR_TYPE_QUANTITY, // PHYSICS_2D_COLLISION_PAIRS
		MONITOR_TYPE_QUANTITY, // PHYSICS_2D_ISLAND_COUNT
		MONITOR_TYPE_QUANTITY, // PHYSICS_3D_ACTIVE_OBJECTS
		MONITOR_TYPE_QUANTITY, // PHYSICS_3D_COLLISION_PAIRS
		MONITOR_TYPE_QUANTITY, // PHYSICS_3D_ISLAND_COUNT
		MONITOR_TYPE_TIME, // AUDIO_OUTPUT_LATENCY
#if !defined(NAVIGATION_2D_DISABLED) || !defined(NAVIGATION_3D_DISABLED)
		MONITOR_TYPE_QUANTITY, // NAVIGATION_ACTIVE_MAPS
		MONITOR_TYPE_QUANTITY, // NAVIGATION_REGION_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_AGENT_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_LINK_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_POLYGON_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_EDGE_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_EDGE_MERGE_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_EDGE_CONNECTION_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_EDGE_FREE_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_OBSTACLE_COUNT
#endif // !defined(NAVIGATION_2D_DISABLED) || !defined(NAVIGATION_3D_DISABLED)
		MONITOR_TYPE_QUANTITY, // PIPELINE_COMPILATIONS_CANVAS
		MONITOR_TYPE_QUANTITY, // PIPELINE_COMPILATIONS_MESH
		MONITOR_TYPE_QUANTITY, // PIPELINE_COMPILATIONS_SURFACE
		MONITOR_TYPE_QUANTITY, // PIPELINE_COMPILATIONS_DRAW
		MONITOR_TYPE_QUANTITY, // PIPELINE_COMPILATIONS_SPECIALIZATION
#ifndef NAVIGATION_2D_DISABLED
		MONITOR_TYPE_QUANTITY, // NAVIGATION_2D_ACTIVE_MAPS
		MONITOR_TYPE_QUANTITY, // NAVIGATION_2D_REGION_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_2D_AGENT_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_2D_LINK_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_2D_POLYGON_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_2D_EDGE_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_2D_EDGE_MERGE_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_2D_EDGE_CONNECTION_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_2D_EDGE_FREE_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_2D_OBSTACLE_COUNT
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
		MONITOR_TYPE_QUANTITY, // NAVIGATION_3D_ACTIVE_MAPS
		MONITOR_TYPE_QUANTITY, // NAVIGATION_3D_REGION_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_3D_AGENT_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_3D_LINK_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_3D_POLYGON_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_3D_EDGE_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_3D_EDGE_MERGE_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_3D_EDGE_CONNECTION_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_3D_EDGE_FREE_COUNT
		MONITOR_TYPE_QUANTITY, // NAVIGATION_3D_OBSTACLE_COUNT
#endif // NAVIGATION_3D_DISABLED
		MONITOR_TYPE_QUANTITY, // TEXTSERVER_BUFFER_COUNT
		MONITOR_TYPE_QUANTITY, // TEXTSERVER_FONT_COUNT
		MONITOR_TYPE_QUANTITY, // TEXTSERVER_FONT_VARIATION_COUNT
		MONITOR_TYPE_QUANTITY, // TEXTSERVER_BUFFER_GLYPH_COUNT
		MONITOR_TYPE_QUANTITY, // TEXTSERVER_FONT_GLYPH_COUNT
		MONITOR_TYPE_MEMORY, // TEXTSERVER_FONT_TEXTURE_BYTES
		MONITOR_TYPE_QUANTITY, // TEXTSERVER_FONT_OVERSAMPLED_GLYPH_COUNT
		MONITOR_TYPE_MEMORY, // TEXTSERVER_FONT_OVERSAMPLED_TEXTURE_BYTES
		MONITOR_TYPE_QUANTITY, // TEXTSERVER_FONT_OVERSAMPLING_LEVELS
		MONITOR_TYPE_MEMORY, // TEXTSERVER_FONT_SYSTEM_FONT_DATA
	};
	static_assert((sizeof(types) / sizeof(MonitorType)) == MONITOR_MAX);

	return types[p_monitor];
}

void Performance::set_process_time(double p_pt) {
	_process_time = p_pt;
}

void Performance::set_physics_process_time(double p_pt) {
	_physics_process_time = p_pt;
}

void Performance::set_navigation_process_time(double p_pt) {
	_navigation_process_time = p_pt;
}

void Performance::add_custom_monitor(const StringName &p_id, const Callable &p_callable, const Vector<Variant> &p_args) {
	ERR_FAIL_COND_MSG(has_custom_monitor(p_id), "Custom monitor with id '" + String(p_id) + "' already exists.");
	_monitor_map.insert(p_id, MonitorCall(p_callable, p_args));
	_monitor_modification_time = OS::get_singleton()->get_ticks_usec();
}

void Performance::remove_custom_monitor(const StringName &p_id) {
	ERR_FAIL_COND_MSG(!has_custom_monitor(p_id), "Custom monitor with id '" + String(p_id) + "' doesn't exists.");
	_monitor_map.erase(p_id);
	_monitor_modification_time = OS::get_singleton()->get_ticks_usec();
}

bool Performance::has_custom_monitor(const StringName &p_id) {
	return _monitor_map.has(p_id);
}

Variant Performance::get_custom_monitor(const StringName &p_id) {
	ERR_FAIL_COND_V_MSG(!has_custom_monitor(p_id), Variant(), "Custom monitor with id '" + String(p_id) + "' doesn't exists.");
	bool error;
	String error_message;
	Variant return_value = _monitor_map[p_id].call(error, error_message);
	ERR_FAIL_COND_V_MSG(error, return_value, "Error calling from custom monitor '" + String(p_id) + "' to callable: " + error_message);
	return return_value;
}

TypedArray<StringName> Performance::get_custom_monitor_names() {
	if (!_monitor_map.size()) {
		return TypedArray<StringName>();
	}
	TypedArray<StringName> return_array;
	return_array.resize(_monitor_map.size());
	int index = 0;
	for (KeyValue<StringName, MonitorCall> i : _monitor_map) {
		return_array.set(index, i.key);
		index++;
	}
	return return_array;
}

uint64_t Performance::get_monitor_modification_time() {
	return _monitor_modification_time;
}

Performance::Performance() {
	_process_time = 0;
	_physics_process_time = 0;
	_navigation_process_time = 0;
	_monitor_modification_time = 0;
	singleton = this;
}

Performance::MonitorCall::MonitorCall(Callable p_callable, Vector<Variant> p_arguments) {
	_callable = p_callable;
	_arguments = p_arguments;
}

Performance::MonitorCall::MonitorCall() {
}

Variant Performance::MonitorCall::call(bool &r_error, String &r_error_message) {
	Vector<const Variant *> arguments_mem;
	arguments_mem.resize(_arguments.size());
	for (int i = 0; i < _arguments.size(); i++) {
		arguments_mem.write[i] = &_arguments[i];
	}
	const Variant **args = (const Variant **)arguments_mem.ptr();
	int argc = _arguments.size();
	Variant return_value;
	Callable::CallError error;
	_callable.callp(args, argc, return_value, error);
	r_error = (error.error != Callable::CallError::CALL_OK);
	if (r_error) {
		r_error_message = Variant::get_callable_error_text(_callable, args, argc, error);
	}
	return return_value;
}

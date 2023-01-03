/**************************************************************************/
/*  godot_navigation_mesh_generator.h                                     */
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

#ifndef GODOT_NAVIGATION_MESH_GENERATOR_H
#define GODOT_NAVIGATION_MESH_GENERATOR_H

#include "servers/navigation/navigation_mesh_generator.h"

#include "core/object/worker_thread_pool.h"
#include "scene/2d/navigation_geometry_parser_2d.h"
#include "scene/3d/navigation_geometry_parser_3d.h"
#include "scene/resources/navigation_mesh_source_geometry_data_2d.h"
#include "scene/resources/navigation_mesh_source_geometry_data_3d.h"

class GodotNavigationMeshGenerator : public NavigationMeshGenerator {
	Mutex generator_mutex;

	bool use_threads = true;
	bool parsing_use_multiple_threads = true;
	bool parsing_use_high_priority_threads = true;
	bool baking_use_multiple_threads = true;
	bool baking_use_high_priority_threads = true;

public:
	GodotNavigationMeshGenerator();
	~GodotNavigationMeshGenerator() override;

	struct NavigationGeneratorTask2D {
		enum TaskStatus {
			PARSING_REQUIRED,
			PARSING_STARTED,
			PARSING_FINISHED,
			PARSING_FAILED,
			BAKING_STARTED,
			BAKING_FINISHED,
			BAKING_FAILED,
			CALLBACK_DISPATCHED,
			CALLBACK_FAILED,
		};

		Ref<NavigationPolygon> navigation_polygon;
		ObjectID parse_root_object_id;
		Ref<NavigationMeshSourceGeometryData2D> source_geometry_data;
		Callable callback;
		NavigationGeneratorTask2D::TaskStatus status = NavigationGeneratorTask2D::TaskStatus::PARSING_REQUIRED;
		LocalVector<Ref<NavigationGeometryParser2D>> geometry_parsers;
	};

	static void _navigation_mesh_generator_2d_thread_bake(void *p_arg);

private:
	LocalVector<Ref<NavigationPolygon>> baking_navpolys;
	LocalVector<Ref<NavigationGeometryParser2D>> geometry_2d_parsers;
	HashMap<NavigationGeneratorTask2D *, WorkerThreadPool::TaskID> navigation_generator_2d_task_to_threadpool_task_id;

#ifndef _3D_DISABLED
public:
	struct NavigationGeneratorTask3D {
		enum TaskStatus {
			PARSING_REQUIRED,
			PARSING_STARTED,
			PARSING_FINISHED,
			PARSING_FAILED,
			BAKING_STARTED,
			BAKING_FINISHED,
			BAKING_FAILED,
			CALLBACK_DISPATCHED,
			CALLBACK_FAILED,
		};

		Ref<NavigationMesh> navigation_mesh;
		ObjectID parse_root_object_id;
		Ref<NavigationMeshSourceGeometryData3D> source_geometry_data;
		Callable callback;
		NavigationGeneratorTask3D::TaskStatus status = NavigationGeneratorTask3D::TaskStatus::PARSING_REQUIRED;
		LocalVector<Ref<NavigationGeometryParser3D>> geometry_parsers;
	};

	static void _navigation_mesh_generator_3d_thread_bake(void *p_arg);

private:
	LocalVector<Ref<NavigationMesh>> baking_navmeshes;
	LocalVector<Ref<NavigationGeometryParser3D>> geometry_3d_parsers;
	HashMap<NavigationGeneratorTask3D *, WorkerThreadPool::TaskID> navigation_generator_3d_task_to_threadpool_task_id;
#endif // _3D_DISABLED

public:
	virtual void process() override;
	virtual void cleanup() override;

	// 2D ////////////////////////////////////
	virtual void register_geometry_parser_2d(Ref<NavigationGeometryParser2D> p_geometry_parser) override;
	virtual void unregister_geometry_parser_2d(Ref<NavigationGeometryParser2D> p_geometry_parser) override;

	virtual Ref<NavigationMeshSourceGeometryData2D> parse_2d_source_geometry_data(Ref<NavigationPolygon> p_navigation_polygon, Node *p_root_node, Callable p_callback = Callable()) override;
	virtual void bake_2d_from_source_geometry_data(Ref<NavigationPolygon> p_navigation_polygon, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry_data, Callable p_callback = Callable()) override;

	virtual void parse_and_bake_2d(Ref<NavigationPolygon> p_navigation_polygon, Node *p_root_node, Callable p_callback = Callable()) override;

	static void _static_parse_2d_geometry_node(Ref<NavigationPolygon> p_navigation_polygon, Node *p_node, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry_data, bool p_recurse_children, LocalVector<Ref<NavigationGeometryParser2D>> &p_geometry_2d_parsers);
	static void _static_parse_2d_source_geometry_data(Ref<NavigationPolygon> p_navigation_polygon, Node *p_root_node, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry_data, LocalVector<Ref<NavigationGeometryParser2D>> &p_geometry_2d_parsers);
	static void _static_bake_2d_from_source_geometry_data(Ref<NavigationPolygon> p_navigation_polygon, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry_data);

	virtual bool is_navigation_polygon_baking(Ref<NavigationPolygon> p_navigation_polygon) const override;

#ifndef _3D_DISABLED
	virtual void register_geometry_parser_3d(Ref<NavigationGeometryParser3D> p_geometry_parser) override;
	virtual void unregister_geometry_parser_3d(Ref<NavigationGeometryParser3D> p_geometry_parser) override;

	virtual Ref<NavigationMeshSourceGeometryData3D> parse_3d_source_geometry_data(Ref<NavigationMesh> p_navigation_mesh, Node *p_root_node, Callable p_callback = Callable()) override;
	virtual void bake_3d_from_source_geometry_data(Ref<NavigationMesh> p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, Callable p_callback = Callable()) override;

	virtual void parse_and_bake_3d(Ref<NavigationMesh> p_navigation_mesh, Node *p_root_node, Callable p_callback = Callable()) override;

	static void _static_parse_3d_geometry_node(Ref<NavigationMesh> p_navigation_mesh, Node *p_node, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, bool p_recurse_children, LocalVector<Ref<NavigationGeometryParser3D>> &p_geometry_3d_parsers);
	static void _static_parse_3d_source_geometry_data(Ref<NavigationMesh> p_navigation_mesh, Node *p_root_node, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, LocalVector<Ref<NavigationGeometryParser3D>> &p_geometry_3d_parsers);
	static void _static_bake_3d_from_source_geometry_data(Ref<NavigationMesh> p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data);

	virtual bool is_navigation_mesh_baking(Ref<NavigationMesh> p_navigation_mesh) const override;
#endif // _3D_DISABLED

private:
	void _process_2d_tasks();
	void _process_2d_parse_tasks();
	void _process_2d_bake_tasks();
	void _process_2d_callbacks();
	void _process_2d_cleanup_tasks();
	void _parse_2d_scenetree_task(uint32_t index, NavigationGeneratorTask2D **parse_task);

#ifndef _3D_DISABLED
	void _process_3d_tasks();
	void _process_3d_parse_tasks();
	void _process_3d_bake_tasks();
	void _process_3d_callbacks();
	void _process_3d_cleanup_tasks();
	void _parse_3d_scenetree_task(uint32_t index, NavigationGeneratorTask3D **parse_task);
#endif // _3D_DISABLED
};

#endif // GODOT_NAVIGATION_MESH_GENERATOR_H

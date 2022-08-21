/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "register_types.h"

#include "core/config/engine.h"
#include "servers/navigation_server_3d.h"

#include "godot_navigation_server.h"

#include "modules/modules_enabled.gen.h" // For csg, gridmap.

#ifndef _3D_DISABLED
#include "modules/navigation/geometry_parser/meshinstance3d_navigation_geometry_parser_3d.h"
#include "modules/navigation/geometry_parser/multimeshinstance3d_navigation_geometry_parser_3d.h"
#include "modules/navigation/geometry_parser/staticbody3d_navigation_geometry_parser_3d.h"
#include "navigation_mesh_generator.h"
#ifdef MODULE_GRIDMAP_ENABLED
#include "modules/navigation/geometry_parser/gridmap_navigation_geometry_parser_3d.h"
#endif // MODULE_GRIDMAP_ENABLED
#ifdef MODULE_CSG_ENABLED
#include "modules/navigation/geometry_parser/csgshape3d_navigation_geometry_parser_3d.h"
#endif // MODULE_CSG_ENABLED
#endif // _3D_DISABLED

#ifdef TOOLS_ENABLED
#include "editor/navigation_mesh_editor_plugin.h"
#endif // TOOLS_ENABLED

#ifndef _3D_DISABLED
NavigationMeshGenerator *_nav_mesh_generator = nullptr;
#endif // _3D_DISABLED

NavigationServer3D *new_server() {
	return memnew(GodotNavigationServer);
}

void initialize_navigation_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		NavigationServer3DManager::set_default_server(new_server);

#ifndef _3D_DISABLED
		_nav_mesh_generator = memnew(NavigationMeshGenerator);
		GDREGISTER_CLASS(NavigationMeshGenerator);
		Engine::get_singleton()->add_singleton(Engine::Singleton("NavigationMeshGenerator", NavigationMeshGenerator::get_singleton()));

		GDREGISTER_CLASS(NavigationGeometryParser3D);
		// add default 3D node navigation geometry parsers
#ifdef MODULE_GRIDMAP_ENABLED
		NavigationMeshGenerator::get_singleton()->register_geometry_parser_3d(memnew(GridMap3DNavigationGeometryParser3D));
#endif // MODULE_GRIDMAP_ENABLED
#ifdef MODULE_CSG_ENABLED
		NavigationMeshGenerator::get_singleton()->register_geometry_parser_3d(memnew(CSGShape3DNavigationGeometryParser3D));
#endif // MODULE_CSG_ENABLED
		NavigationMeshGenerator::get_singleton()->register_geometry_parser_3d(memnew(MultiMeshInstance3DNavigationGeometryParser3D));
		NavigationMeshGenerator::get_singleton()->register_geometry_parser_3d(memnew(StaticBody3DNavigationGeometryParser3D));
		NavigationMeshGenerator::get_singleton()->register_geometry_parser_3d(memnew(MeshInstance3DNavigationGeometryParser3D));
#endif // _3D_DISABLED
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		EditorPlugins::add_by_type<NavigationMeshEditorPlugin>();
	}
#endif // TOOLS_ENABLED
}

void uninitialize_navigation_module(ModuleInitializationLevel p_level) {
#ifndef _3D_DISABLED
	if (_nav_mesh_generator) {
		if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
			// required at this early level to avoid crashes with e.g. GDScript
			NavigationMeshGenerator::get_singleton()->cleanup();
		}
		if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
			memdelete(_nav_mesh_generator);
		}
	}
#endif // _3D_DISABLED
}

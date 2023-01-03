/**************************************************************************/
/*  navigation_mesh_generator.cpp                                         */
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

#include "navigation_mesh_generator.h"

#include "core/config/project_settings.h"

NavigationMeshGenerator *NavigationMeshGenerator::singleton = nullptr;

void NavigationMeshGenerator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("register_geometry_parser_2d", "geometry_parser"), &NavigationMeshGenerator::register_geometry_parser_2d);
	ClassDB::bind_method(D_METHOD("unregister_geometry_parser_2d", "geometry_parser"), &NavigationMeshGenerator::unregister_geometry_parser_2d);

	ClassDB::bind_method(D_METHOD("parse_2d_source_geometry_data", "navigation_polygon", "root_node", "callback"), &NavigationMeshGenerator::parse_2d_source_geometry_data, DEFVAL(Callable()));
	ClassDB::bind_method(D_METHOD("bake_2d_from_source_geometry_data", "navigation_polygon", "source_geometry_data", "callback"), &NavigationMeshGenerator::bake_2d_from_source_geometry_data, DEFVAL(Callable()));
	ClassDB::bind_method(D_METHOD("parse_and_bake_2d", "navigation_polygon", "root_node", "callback"), &NavigationMeshGenerator::parse_and_bake_2d, DEFVAL(Callable()));

	ClassDB::bind_method(D_METHOD("is_navigation_polygon_baking", "navigation_polygon"), &NavigationMeshGenerator::is_navigation_polygon_baking);

#ifndef _3D_DISABLED
	ClassDB::bind_method(D_METHOD("register_geometry_parser_3d", "geometry_parser"), &NavigationMeshGenerator::register_geometry_parser_3d);
	ClassDB::bind_method(D_METHOD("unregister_geometry_parser_3d", "geometry_parser"), &NavigationMeshGenerator::unregister_geometry_parser_3d);

	ClassDB::bind_method(D_METHOD("parse_3d_source_geometry_data", "navigation_mesh", "root_node", "callback"), &NavigationMeshGenerator::parse_3d_source_geometry_data, DEFVAL(Callable()));
	ClassDB::bind_method(D_METHOD("bake_3d_from_source_geometry_data", "navigation_mesh", "source_geometry_data", "callback"), &NavigationMeshGenerator::bake_3d_from_source_geometry_data, DEFVAL(Callable()));
	ClassDB::bind_method(D_METHOD("parse_and_bake_3d", "navigation_mesh", "root_node", "callback"), &NavigationMeshGenerator::parse_and_bake_3d, DEFVAL(Callable()));

	ClassDB::bind_method(D_METHOD("is_navigation_mesh_baking", "navigation_mesh"), &NavigationMeshGenerator::is_navigation_mesh_baking);
#endif // _3D_DISABLED
}

NavigationMeshGenerator *NavigationMeshGenerator::get_singleton() {
	return singleton;
}

NavigationMeshGenerator::NavigationMeshGenerator() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

NavigationMeshGenerator::~NavigationMeshGenerator() {
	singleton = nullptr;
}

/// NavigationMeshGeneratorManager ////////////////////////////////////////////////////

NavigationMeshGeneratorManager *NavigationMeshGeneratorManager::singleton = nullptr;
const String NavigationMeshGeneratorManager::setting_property_name(PNAME("navigation/baking/generator/navigation_mesh_generator"));

void NavigationMeshGeneratorManager::on_servers_changed() {
	String nav_server_names("DEFAULT");
	for (const ClassInfo &server : navigation_mesh_generators) {
		nav_server_names += "," + server.name;
	}

	ProjectSettings::get_singleton()->set_custom_property_info(PropertyInfo(Variant::STRING, setting_property_name, PROPERTY_HINT_ENUM, nav_server_names));
}

void NavigationMeshGeneratorManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("register_server", "name", "create_callback"), &NavigationMeshGeneratorManager::register_server);
	ClassDB::bind_method(D_METHOD("set_default_server", "name", "priority"), &NavigationMeshGeneratorManager::set_default_server);
}

NavigationMeshGeneratorManager *NavigationMeshGeneratorManager::get_singleton() {
	return singleton;
}

void NavigationMeshGeneratorManager::register_server(const String &p_name, const Callable &p_create_callback) {
	// TODO: Enable check when is_valid() is fixed for static functions.
	//ERR_FAIL_COND(!p_create_callback.is_valid());
	ERR_FAIL_COND_MSG(find_server_id(p_name) != -1, "NavigationMeshGenerator with the same name was already registered.");

	navigation_mesh_generators.push_back(ClassInfo{ p_name, p_create_callback });
	on_servers_changed();
}

void NavigationMeshGeneratorManager::set_default_server(const String &p_name, int p_priority) {
	const int id = find_server_id(p_name);
	ERR_FAIL_COND_MSG(id == -1, "NavigationMeshGenerator not found"); // Not found

	// Only change the server if it is registered with a higher priority
	if (default_server_priority < p_priority) {
		default_server_id = id;
		default_server_priority = p_priority;
	}
}

int NavigationMeshGeneratorManager::find_server_id(const String &p_name) const {
	for (int i = 0; i < navigation_mesh_generators.size(); ++i) {
		if (p_name == navigation_mesh_generators[i].name) {
			return i;
		}
	}

	return -1;
}

NavigationMeshGenerator *NavigationMeshGeneratorManager::new_default_server() const {
	ERR_FAIL_COND_V(default_server_id == -1, nullptr);

	Variant ret;
	Callable::CallError ce;
	navigation_mesh_generators[default_server_id].create_callback.callp(nullptr, 0, ret, ce);

	ERR_FAIL_COND_V(ce.error != Callable::CallError::CALL_OK, nullptr);
	return Object::cast_to<NavigationMeshGenerator>(ret.get_validated_object());
}

NavigationMeshGenerator *NavigationMeshGeneratorManager::new_server(const String &p_name) const {
	const int id = find_server_id(p_name);

	if (id == -1) {
		return nullptr;
	}

	Variant ret;
	Callable::CallError ce;
	navigation_mesh_generators[id].create_callback.callp(nullptr, 0, ret, ce);

	ERR_FAIL_COND_V(ce.error != Callable::CallError::CALL_OK, nullptr);
	return Object::cast_to<NavigationMeshGenerator>(ret.get_validated_object());
}

NavigationMeshGeneratorManager::NavigationMeshGeneratorManager() {
	singleton = this;

	GLOBAL_DEF("navigation/baking/thread_model/parsing_use_multiple_threads", true);
	GLOBAL_DEF("navigation/baking/thread_model/parsing_use_high_priority_threads", true);
	GLOBAL_DEF("navigation/baking/thread_model/baking_use_multiple_threads", true);
	GLOBAL_DEF("navigation/baking/thread_model/baking_use_high_priority_threads", true);
}

NavigationMeshGeneratorManager::~NavigationMeshGeneratorManager() {
	singleton = nullptr;
}

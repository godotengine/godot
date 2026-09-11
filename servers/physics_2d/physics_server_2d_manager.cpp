/**************************************************************************/
/*  physics_server_2d_manager.cpp                                         */
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

#include "physics_server_2d_manager.h"

#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "servers/physics_2d/physics_server_2d.h"
#include "servers/physics_2d/physics_server_2d_dummy.h"

static PhysicsServer2D *physics_server_2d = nullptr;

PhysicsServer2DManager *PhysicsServer2DManager::singleton = nullptr;
const String PhysicsServer2DManager::setting_property_name(PNAME("physics/2d/physics_engine"));

void PhysicsServer2DManager::initialize_server_manager() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = memnew(PhysicsServer2DManager);
}

void PhysicsServer2DManager::finalize_server_manager() {
	ERR_FAIL_NULL(singleton);
	memdelete(singleton);
	singleton = nullptr;
}

void PhysicsServer2DManager::initialize_server() {
#ifndef PHYSICS_2D_DISABLED
	// 2D Physics server
	physics_server_2d = PhysicsServer2DManager::get_singleton()->new_server(
			GLOBAL_GET(PhysicsServer2DManager::get_singleton()->setting_property_name));
	if (!physics_server_2d) {
		// Physics server not found, Use the default physics
		physics_server_2d = PhysicsServer2DManager::get_singleton()->new_default_server();
	}

	// Fall back to dummy if no default server has been registered.
	if (!physics_server_2d) {
		WARN_PRINT(vformat("Falling back to dummy PhysicsServer2D; 2D physics functionality will be disabled. If this is intended, set the %s project setting to Dummy.", PhysicsServer2DManager::setting_property_name));
		physics_server_2d = memnew(PhysicsServer2DDummy);
	}

	// Should be impossible, but make sure it's not null.
	ERR_FAIL_NULL_MSG(physics_server_2d, "Failed to initialize PhysicsServer2D.");
	physics_server_2d->init();
#endif // PHYSICS_2D_DISABLED
}

void PhysicsServer2DManager::finalize_server() {
	ERR_FAIL_NULL(physics_server_2d);
	physics_server_2d->finish();
	memdelete(physics_server_2d);
	physics_server_2d = nullptr;
}

void PhysicsServer2DManager::on_servers_changed() {
	String physics_servers("DEFAULT");
	for (int i = get_servers_count() - 1; 0 <= i; --i) {
		physics_servers += "," + get_server_name(i);
	}
	ProjectSettings::get_singleton()->set_custom_property_info(PropertyInfo(Variant::STRING, setting_property_name, PROPERTY_HINT_ENUM, physics_servers));
	ProjectSettings::get_singleton()->set_restart_if_changed(setting_property_name, true);
	ProjectSettings::get_singleton()->set_as_basic(setting_property_name, true);
}

void PhysicsServer2DManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("register_server", "name", "create_callback"), &PhysicsServer2DManager::register_server);
	ClassDB::bind_method(D_METHOD("set_default_server", "name", "priority"), &PhysicsServer2DManager::set_default_server);
}

PhysicsServer2DManager *PhysicsServer2DManager::get_singleton() {
	return singleton;
}

void PhysicsServer2DManager::register_server(const String &p_name, const Callable &p_create_callback) {
	//ERR_FAIL_COND(!p_create_callback.is_valid());
	ERR_FAIL_COND(find_server_id(p_name) != -1);
	physics_2d_servers.push_back(ClassInfo(p_name, p_create_callback));
	on_servers_changed();
}

void PhysicsServer2DManager::set_default_server(const String &p_name, int p_priority) {
	const int id = find_server_id(p_name);
	ERR_FAIL_COND(id == -1); // Not found
	if (default_server_priority < p_priority) {
		default_server_id = id;
		default_server_priority = p_priority;
	}
}

int PhysicsServer2DManager::find_server_id(const String &p_name) {
	for (int i = physics_2d_servers.size() - 1; 0 <= i; --i) {
		if (p_name == physics_2d_servers[i].name) {
			return i;
		}
	}
	return -1;
}

int PhysicsServer2DManager::get_servers_count() {
	return physics_2d_servers.size();
}

String PhysicsServer2DManager::get_server_name(int p_id) {
	ERR_FAIL_INDEX_V(p_id, get_servers_count(), "");
	return physics_2d_servers[p_id].name;
}

PhysicsServer2D *PhysicsServer2DManager::new_default_server() {
	if (default_server_id == -1) {
		return nullptr;
	}
	Variant ret;
	Callable::CallError ce;
	physics_2d_servers[default_server_id].create_callback.callp(nullptr, 0, ret, ce);
	ERR_FAIL_COND_V(ce.error != Callable::CallError::CALL_OK, nullptr);
	return Object::cast_to<PhysicsServer2D>(ret.get_validated_object());
}

PhysicsServer2D *PhysicsServer2DManager::new_server(const String &p_name) {
	int id = find_server_id(p_name);
	if (id == -1) {
		return nullptr;
	} else {
		Variant ret;
		Callable::CallError ce;
		physics_2d_servers[id].create_callback.callp(nullptr, 0, ret, ce);
		ERR_FAIL_COND_V(ce.error != Callable::CallError::CALL_OK, nullptr);
		return Object::cast_to<PhysicsServer2D>(ret.get_validated_object());
	}
}

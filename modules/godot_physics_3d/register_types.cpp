/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "godot_physics_server_3d.h"
#include "servers/physics_server_3d.h"
#include "servers/physics_server_3d_wrap_mt.h"

static PhysicsServer3D *_createGodotPhysics3DCallback() {
#ifdef THREADS_ENABLED
	bool using_threads = GLOBAL_GET("physics/3d/run_on_separate_thread");
#else
	bool using_threads = false;
#endif

	PhysicsServer3D *physics_server_3d = memnew(GodotPhysicsServer3D(using_threads));

	return memnew(PhysicsServer3DWrapMT(physics_server_3d, using_threads));
}

void initialize_godot_physics_3d_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SERVERS) {
		return;
	}
	PhysicsServer3DManager::get_singleton()->register_server("GodotPhysics3D", callable_mp_static(_createGodotPhysics3DCallback));
	PhysicsServer3DManager::get_singleton()->set_default_server("GodotPhysics3D");

#ifndef DISABLE_DEPRECATED
#define MOVE_PROJECT_SETTING(m_old_setting, m_new_setting)                                                                               \
	if (!ProjectSettings::get_singleton()->has_setting(m_new_setting) && ProjectSettings::get_singleton()->has_setting(m_old_setting)) { \
		Variant value = GLOBAL_GET(m_old_setting);                                                                                       \
		ProjectSettings::get_singleton()->set_setting(m_new_setting, value);                                                             \
		ProjectSettings::get_singleton()->clear(m_old_setting);                                                                          \
	}
	MOVE_PROJECT_SETTING("physics/3d/sleep_threshold_linear", "physics/godot_physics_3d/sleep_threshold_linear")
	MOVE_PROJECT_SETTING("physics/3d/sleep_threshold_angular", "physics/godot_physics_3d/sleep_threshold_angular")
	MOVE_PROJECT_SETTING("physics/3d/time_before_sleep", "physics/godot_physics_3d/time_before_sleep")
	MOVE_PROJECT_SETTING("physics/3d/solver/solver_iterations", "physics/godot_physics_3d/solver/solver_iterations")
	MOVE_PROJECT_SETTING("physics/3d/solver/contact_recycle_radius", "physics/godot_physics_3d/solver/contact_recycle_radius")
	MOVE_PROJECT_SETTING("physics/3d/solver/contact_max_separation", "physics/godot_physics_3d/solver/contact_max_separation")
	MOVE_PROJECT_SETTING("physics/3d/solver/contact_max_allowed_penetration", "physics/godot_physics_3d/solver/contact_max_allowed_penetration")
	MOVE_PROJECT_SETTING("physics/3d/solver/default_contact_bias", "physics/godot_physics_3d/solver/default_contact_bias")
#endif

	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/godot_physics_3d/sleep_threshold_linear", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater"), 0.1);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/godot_physics_3d/sleep_threshold_angular", PROPERTY_HINT_RANGE, "0,90,0.1,radians_as_degrees"), Math::deg_to_rad(8.0));
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/godot_physics_3d/time_before_sleep", PROPERTY_HINT_RANGE, "0,5,0.01,or_greater"), 0.5);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "physics/godot_physics_3d/solver/solver_iterations", PROPERTY_HINT_RANGE, "1,32,1,or_greater"), 16);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/godot_physics_3d/solver/contact_recycle_radius", PROPERTY_HINT_RANGE, "0,0.1,0.001,or_greater"), 0.01);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/godot_physics_3d/solver/contact_max_separation", PROPERTY_HINT_RANGE, "0,0.1,0.001,or_greater"), 0.05);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/godot_physics_3d/solver/contact_max_allowed_penetration", PROPERTY_HINT_RANGE, "0.001,0.1,0.001,or_greater"), 0.01);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/godot_physics_3d/solver/default_contact_bias", PROPERTY_HINT_RANGE, "0,1,0.01"), 0.8);
}

void uninitialize_godot_physics_3d_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SERVERS) {
		return;
	}
}

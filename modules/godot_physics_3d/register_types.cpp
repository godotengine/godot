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
#include "servers/physics_3d/physics_server_3d.h"
#include "servers/physics_3d/physics_server_3d_wrap_mt.h"

#include "modules/modules_enabled.gen.h" // For jolt_physics.

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
#ifndef MODULE_JOLT_PHYSICS_ENABLED
	// Set the default explicitly as the call to `set_default_server()` in `modules/jolt_physics/` isn't compiled in.
	// Otherwise, we would fall back to the Dummy physics server and end up with no physics at all.
	PhysicsServer3DManager::get_singleton()->set_default_server("GodotPhysics3D");
#endif
}

void uninitialize_godot_physics_3d_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SERVERS) {
		return;
	}
}

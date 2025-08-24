/**************************************************************************/
/*  sandbox_globals.cpp                                                   */
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

#include "sandbox.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/debugger/engine_debugger.h"
#include "core/extension/gdextension_manager.h"
#include "core/input/input.h"
#include "core/input/input_map.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/io/resource_uid.h"
#include "core/math/geometry_2d.h"
#include "core/math/geometry_3d.h"
#include "core/object/worker_thread_pool.h"
#include "core/os/time.h"
#include "core/string/translation.h"
#include "core/string/translation_server.h"
#include "main/performance.h"
#include "scene/theme/theme_db.h"
#include "servers/audio_server.h"
#include "servers/display_server.h"
#include "servers/navigation_server_2d.h"
#include "servers/navigation_server_3d.h"
#include "servers/physics_server_2d.h"
#include "servers/physics_server_3d.h"
#include "servers/rendering_server.h"
#include "servers/text_server.h"
#include "servers/xr_server.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_interface.h"
#endif

#ifdef MODULE_NAVIGATION_ENABLED
#include "modules/navigation/3d/navigation_mesh_generator.h"
#endif
#include <functional>
#include <unordered_map>

namespace riscv {
std::unordered_map<std::string, std::function<uint64_t()>> global_singleton_list = {
	// { "OS", [] { return uint64_t(uintptr_t(OS::get_singleton())); } },
	// { "Geometry2D", [] { return uint64_t(uintptr_t(Geometry2D::get_singleton())); } },
	// { "Geometry3D", [] { return uint64_t(uintptr_t(Geometry3D::get_singleton())); } },
	// { "NavigationMeshGenerator", [] { return uint64_t(uintptr_t(NavigationMeshGenerator::get_singleton())); } },
	// { "ResourceLoader", [] { return uint64_t(uintptr_t(ResourceLoader::get_singleton())); } },
	// { "ResourceSaver", [] { return uint64_t(uintptr_t(ResourceSaver::get_singleton())); } },
	// { "Marshalls", [] { return uint64_t(uintptr_t(Marshalls::get_singleton())); } },
	{ "AudioServer", [] { return uint64_t(uintptr_t(AudioServer::get_singleton())); } },
#ifdef TOOLS_ENABLED
	{ "EditorInterface", [] { return uint64_t(uintptr_t(EditorInterface::get_singleton())); } },
#endif
	{ "DisplayServer", [] { return uint64_t(uintptr_t(DisplayServer::get_singleton())); } },
	{ "GDExtensionManager", [] { return uint64_t(uintptr_t(GDExtensionManager::get_singleton())); } },
	{ "EngineDebugger", [] { return uint64_t(uintptr_t(EngineDebugger::get_singleton())); } },
	{ "Engine", [] { return uint64_t(uintptr_t(Engine::get_singleton())); } },
	{ "Input", [] { return uint64_t(uintptr_t(Input::get_singleton())); } },
	{ "InputMap", [] { return uint64_t(uintptr_t(InputMap::get_singleton())); } },
	{ "NativeMenu", [] { return uint64_t(uintptr_t(NativeMenu::get_singleton())); } },
	{ "NavigationServer2D", [] { return uint64_t(uintptr_t(NavigationServer2D::get_singleton())); } },
	{ "NavigationServer3D", [] { return uint64_t(uintptr_t(NavigationServer3D::get_singleton())); } },
	{ "Performance", [] { return uint64_t(uintptr_t(Performance::get_singleton())); } },
	{ "PhysicsServer2D", [] { return uint64_t(uintptr_t(PhysicsServer2D::get_singleton())); } },
	{ "PhysicsServer3D", [] { return uint64_t(uintptr_t(PhysicsServer3D::get_singleton())); } },
	{ "PhysicsServer2DManager", [] { return uint64_t(uintptr_t(PhysicsServer2DManager::get_singleton())); } },
	{ "PhysicsServer3DManager", [] { return uint64_t(uintptr_t(PhysicsServer3DManager::get_singleton())); } },
	{ "ProjectSettings", [] { return uint64_t(uintptr_t(ProjectSettings::get_singleton())); } },
	{ "RenderingServer", [] { return uint64_t(uintptr_t(RenderingServer::get_singleton())); } },
	{ "ResourceUID", [] { return uint64_t(uintptr_t(ResourceUID::get_singleton())); } },
	{ "TextServerManager", [] { return uint64_t(uintptr_t(TextServerManager::get_singleton())); } },
	{ "ThemeDB", [] { return uint64_t(uintptr_t(ThemeDB::get_singleton())); } },
	{ "Time", [] { return uint64_t(uintptr_t(Time::get_singleton())); } },
	{ "TranslationServer", [] { return uint64_t(uintptr_t(TranslationServer::get_singleton())); } },
	{ "WorkerThreadPool", [] { return uint64_t(uintptr_t(WorkerThreadPool::get_singleton())); } },
	{ "XRServer", [] { return uint64_t(uintptr_t(XRServer::get_singleton())); } },
};
} // namespace riscv

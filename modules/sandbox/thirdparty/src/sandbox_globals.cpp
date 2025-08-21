#include "sandbox.h"

#include <godot_cpp/classes/audio_server.hpp>
#include <godot_cpp/classes/display_server.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/engine_debugger.hpp>
#include <godot_cpp/classes/gd_extension_manager.hpp>
#include <godot_cpp/classes/geometry2d.hpp>
#include <godot_cpp/classes/geometry3d.hpp>
#include <godot_cpp/classes/input.hpp>
#include <godot_cpp/classes/input_map.hpp>
#include <godot_cpp/classes/marshalls.hpp>
#include <godot_cpp/classes/native_menu.hpp>
#include <godot_cpp/classes/navigation_mesh_generator.hpp>
#include <godot_cpp/classes/navigation_server2d.hpp>
#include <godot_cpp/classes/navigation_server3d.hpp>
#include <godot_cpp/classes/performance.hpp>
#include <godot_cpp/classes/physics_server2d.hpp>
#include <godot_cpp/classes/physics_server2d_manager.hpp>
#include <godot_cpp/classes/physics_server3d.hpp>
#include <godot_cpp/classes/physics_server3d_manager.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/resource_saver.hpp>
#include <godot_cpp/classes/resource_uid.hpp>
#include <godot_cpp/classes/text_server_manager.hpp>
#include <godot_cpp/classes/theme_db.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/classes/translation_server.hpp>
#include <godot_cpp/classes/worker_thread_pool.hpp>
#include <godot_cpp/classes/xr_server.hpp>

namespace riscv {
std::unordered_map<std::string, std::function<uint64_t()>> global_singleton_list = {
	{ "AudioServer", [] { return uint64_t(uintptr_t(AudioServer::get_singleton())); } },
	{ "EditorInterface", [] { return uint64_t(uintptr_t(EditorInterface::get_singleton())); } },
	{ "DisplayServer", [] { return uint64_t(uintptr_t(DisplayServer::get_singleton())); } },
	{ "GDExtensionManager", [] { return uint64_t(uintptr_t(GDExtensionManager::get_singleton())); } },
	{ "Geometry2D", [] { return uint64_t(uintptr_t(Geometry2D::get_singleton())); } },
	{ "Geometry3D", [] { return uint64_t(uintptr_t(Geometry3D::get_singleton())); } },
	{ "EngineDebugger", [] { return uint64_t(uintptr_t(EngineDebugger::get_singleton())); } },
	{ "Engine", [] { return uint64_t(uintptr_t(Engine::get_singleton())); } },
	{ "Input", [] { return uint64_t(uintptr_t(Input::get_singleton())); } },
	{ "InputMap", [] { return uint64_t(uintptr_t(InputMap::get_singleton())); } },
	{ "Marshalls", [] { return uint64_t(uintptr_t(Marshalls::get_singleton())); } },
	{ "NativeMenu", [] { return uint64_t(uintptr_t(NativeMenu::get_singleton())); } },
	{ "NavigationMeshGenerator", [] { return uint64_t(uintptr_t(NavigationMeshGenerator::get_singleton())); } },
	{ "NavigationServer2D", [] { return uint64_t(uintptr_t(NavigationServer2D::get_singleton())); } },
	{ "NavigationServer3D", [] { return uint64_t(uintptr_t(NavigationServer3D::get_singleton())); } },
//	{ "OS", [] { return uint64_t(uintptr_t(OS::get_singleton())); } },
	{ "Performance", [] { return uint64_t(uintptr_t(Performance::get_singleton())); } },
	{ "PhysicsServer2D", [] { return uint64_t(uintptr_t(PhysicsServer2D::get_singleton())); } },
	{ "PhysicsServer3D", [] { return uint64_t(uintptr_t(PhysicsServer3D::get_singleton())); } },
	{ "PhysicsServer2DManager", [] { return uint64_t(uintptr_t(PhysicsServer2DManager::get_singleton())); } },
	{ "PhysicsServer3DManager", [] { return uint64_t(uintptr_t(PhysicsServer3DManager::get_singleton())); } },
	{ "ProjectSettings", [] { return uint64_t(uintptr_t(ProjectSettings::get_singleton())); } },
	{ "RenderingServer", [] { return uint64_t(uintptr_t(RenderingServer::get_singleton())); } },
	{ "ResourceLoader", [] { return uint64_t(uintptr_t(ResourceLoader::get_singleton())); } },
	{ "ResourceSaver", [] { return uint64_t(uintptr_t(ResourceSaver::get_singleton())); } },
	{ "ResourceUID", [] { return uint64_t(uintptr_t(ResourceUID::get_singleton())); } },
	{ "TextServerManager", [] { return uint64_t(uintptr_t(TextServerManager::get_singleton())); } },
	{ "ThemeDB", [] { return uint64_t(uintptr_t(ThemeDB::get_singleton())); } },
	{ "Time", [] { return uint64_t(uintptr_t(Time::get_singleton())); } },
	{ "TranslationServer", [] { return uint64_t(uintptr_t(TranslationServer::get_singleton())); } },
	{ "WorkerThreadPool", [] { return uint64_t(uintptr_t(WorkerThreadPool::get_singleton())); } },
	{ "XRServer", [] { return uint64_t(uintptr_t(XRServer::get_singleton())); } },
};

} // namespace riscv

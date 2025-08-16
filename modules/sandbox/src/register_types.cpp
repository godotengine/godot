#include "register_types.h"

#include <gdextension_interface.h>

#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include "elf/resource_loader_elf.h"
#include "elf/resource_saver_elf.h"
#include "elf/script_elf.h"
#include "elf/script_language_elf.h"
#include "sandbox.h"
#include "sandbox_project_settings.h"
#include "cpp/resource_loader_cpp.h"
#include "cpp/resource_saver_cpp.h"
#include "cpp/script_cpp.h"
#include "cpp/script_language_cpp.h"
#ifdef PLATFORM_HAS_EDITOR
#include "rust/resource_loader_rust.h"
#include "rust/resource_saver_rust.h"
#include "rust/script_language_rust.h"
#include "rust/script_rust.h"
#include "zig/resource_loader_zig.h"
#include "zig/resource_saver_zig.h"
#include "zig/script_language_zig.h"
#include "zig/script_zig.h"
#endif

using namespace godot;

static Ref<ResourceFormatLoaderELF> elf_loader;
static Ref<ResourceFormatSaverELF> elf_saver;

static ELFScriptLanguage *elf_language;

ScriptLanguage *get_elf_language() {
	return elf_language;
}

static void initialize_riscv_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ClassDB::register_class<Sandbox>();
	ClassDB::register_class<ELFScript>();
	ClassDB::register_class<ELFScriptLanguage>();
	ClassDB::register_class<ResourceFormatLoaderELF>();
	ClassDB::register_class<ResourceFormatSaverELF>();
	ClassDB::register_class<CPPScript>();
	ClassDB::register_class<CPPScriptLanguage>();
	ClassDB::register_class<ResourceFormatLoaderCPP>();
	ClassDB::register_class<ResourceFormatSaverCPP>();
#ifdef PLATFORM_HAS_EDITOR
	ClassDB::register_class<RustScript>();
	ClassDB::register_class<RustScriptLanguage>();
	ClassDB::register_class<ResourceFormatLoaderRust>();
	ClassDB::register_class<ResourceFormatSaverRust>();
	ClassDB::register_class<ZigScript>();
	ClassDB::register_class<ZigScriptLanguage>();
	ClassDB::register_class<ResourceFormatLoaderZig>();
	ClassDB::register_class<ResourceFormatSaverZig>();
#endif
	elf_loader.instantiate();
	elf_saver.instantiate();
	ResourceLoader::get_singleton()->add_resource_format_loader(elf_loader, true);
	ResourceSaver::get_singleton()->add_resource_format_saver(elf_saver);
	elf_language = memnew(ELFScriptLanguage);
	Engine::get_singleton()->register_script_language(elf_language);
	ResourceFormatLoaderCPP::init();
	ResourceFormatSaverCPP::init();
	CPPScriptLanguage::init();
#ifdef PLATFORM_HAS_EDITOR
	ResourceFormatLoaderRust::init();
	ResourceFormatSaverRust::init();
	RustScriptLanguage::init();
	ResourceFormatLoaderZig::init();
	ResourceFormatSaverZig::init();
	ZigScriptLanguage::init();
#endif
	SandboxProjectSettings::register_settings();
}

static void uninitialize_riscv_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	Engine *engine = Engine::get_singleton();
	CPPScriptLanguage::deinit();
#ifdef PLATFORM_HAS_EDITOR
	RustScriptLanguage::deinit();
	ZigScriptLanguage::deinit();
#endif
	if (elf_language) {
		engine->unregister_script_language(elf_language);
		memdelete(elf_language);
		elf_language = nullptr;
	}

	ResourceLoader::get_singleton()->remove_resource_format_loader(elf_loader);
	ResourceSaver::get_singleton()->remove_resource_format_saver(elf_saver);
	elf_loader.unref();
	elf_saver.unref();
	ResourceFormatLoaderCPP::deinit();
	ResourceFormatSaverCPP::deinit();
#ifdef PLATFORM_HAS_EDITOR
	ResourceFormatLoaderRust::deinit();
	ResourceFormatSaverRust::deinit();
	ResourceFormatLoaderZig::deinit();
	ResourceFormatSaverZig::deinit();
#endif
}

extern "C" {
// Initialization.
GDExtensionBool GDE_EXPORT riscv_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
	godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

	init_obj.register_initializer(initialize_riscv_module);
	init_obj.register_terminator(uninitialize_riscv_module);
	init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

	return init_obj.init();
}
}

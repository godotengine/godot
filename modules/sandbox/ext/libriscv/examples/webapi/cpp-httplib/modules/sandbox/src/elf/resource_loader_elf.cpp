#include "resource_loader_elf.h"
#include "../sandbox.h"
#include "script_elf.h"
#include <godot_cpp/classes/file_access.hpp>
static constexpr bool VERBOSE_LOADER = false;

Variant ResourceFormatLoaderELF::_load(const String &p_path, const String &original_path, bool use_sub_threads, int32_t cache_mode) const {
#ifdef RISCV_BINARY_TRANSLATION
	// We will automatically load .dll's or .so's with the same basename and path as the ELF file.
	String dllpath = p_path.get_basename();
# ifdef _WIN32
	dllpath += ".dll";
# elif defined(__APPLE__)
	dllpath += ".dylib";
# else
	dllpath += ".so";
# endif
	if (FileAccess::file_exists(dllpath)) {
		// Load the binary translation library.
		if (!Sandbox::load_binary_translation(dllpath, true)) {
			WARN_PRINT("Failed to auto-load binary translation library: " + dllpath);
		} else if constexpr (VERBOSE_LOADER) {
			WARN_PRINT("Auto-loaded binary translation library: " + dllpath);
		}
	} else if constexpr (VERBOSE_LOADER) {
		WARN_PRINT("Binary translation library not found: " + dllpath);
	}
#endif
	Ref<ELFScript> elf_model = memnew(ELFScript);
	elf_model->set_file(p_path);
	elf_model->reload(false);
	return elf_model;
}
PackedStringArray ResourceFormatLoaderELF::_get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("elf");
	return array;
}
bool ResourceFormatLoaderELF::_recognize_path(const godot::String &path, const godot::StringName &type) const {
	String el = path.get_extension().to_lower();
	if (el == "elf") {
		return true;
	}
	return false;
}
bool ResourceFormatLoaderELF::_handles_type(const StringName &type) const {
	String type_str = type;
	return type_str == "ELFScript" || type_str == "Script";
}
String ResourceFormatLoaderELF::_get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "elf") {
		return "ELFScript";
	}
	return "";
}

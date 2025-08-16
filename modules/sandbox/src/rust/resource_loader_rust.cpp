#include "resource_loader_rust.h"
#include "script_rust.h"
#include <godot_cpp/classes/file_access.hpp>

static Ref<ResourceFormatLoaderRust> rust_loader;

void ResourceFormatLoaderRust::init() {
	rust_loader.instantiate();
	ResourceLoader::get_singleton()->add_resource_format_loader(rust_loader);
}

void ResourceFormatLoaderRust::deinit() {
	ResourceLoader::get_singleton()->remove_resource_format_loader(rust_loader);
	rust_loader.unref();
}

Variant ResourceFormatLoaderRust::_load(const String &p_path, const String &original_path, bool use_sub_threads, int32_t cache_mode) const {
	Ref<RustScript> model = memnew(RustScript);
	model->_set_source_code(FileAccess::get_file_as_string(p_path));
	return model;
}
PackedStringArray ResourceFormatLoaderRust::_get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("rs");
	return array;
}
bool ResourceFormatLoaderRust::_handles_type(const StringName &type) const {
	String type_str = type;
	return type_str == "RustScript" || type_str == "Script";
}
String ResourceFormatLoaderRust::_get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "rs") {
		return "RustScript";
	}
	return "";
}

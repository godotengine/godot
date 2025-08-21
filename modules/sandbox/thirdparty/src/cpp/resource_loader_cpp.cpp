#include "resource_loader_cpp.h"
#include "script_cpp.h"
#include <godot_cpp/classes/file_access.hpp>

static Ref<ResourceFormatLoaderCPP> cpp_loader;

void ResourceFormatLoaderCPP::init() {
	cpp_loader.instantiate();
	ResourceLoader::get_singleton()->add_resource_format_loader(cpp_loader);
}

void ResourceFormatLoaderCPP::deinit() {
	ResourceLoader::get_singleton()->remove_resource_format_loader(cpp_loader);
	cpp_loader.unref();
}

Variant ResourceFormatLoaderCPP::_load(const String &p_path, const String &original_path, bool use_sub_threads, int32_t cache_mode) const {
	Ref<CPPScript> cpp_model = memnew(CPPScript);
	cpp_model->set_file(p_path);
	return cpp_model;
}
PackedStringArray ResourceFormatLoaderCPP::_get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("cpp");
	array.push_back("cc");
	array.push_back("hh");
	array.push_back("h");
	array.push_back("hpp");
	return array;
}
bool ResourceFormatLoaderCPP::_handles_type(const StringName &type) const {
	String type_str = type;
	return type_str == "CPPScript" || type_str == "Script";
}
String ResourceFormatLoaderCPP::_get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "hpp" || el == "cpp" || el == "h" || el == "cc" || el == "hh") {
		return "CPPScript";
	}
	return "";
}

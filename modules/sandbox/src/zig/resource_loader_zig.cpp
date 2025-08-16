#include "resource_loader_zig.h"
#include "script_zig.h"
#include <godot_cpp/classes/file_access.hpp>

static Ref<ResourceFormatLoaderZig> zig_loader;

void ResourceFormatLoaderZig::init() {
	zig_loader.instantiate();
	ResourceLoader::get_singleton()->add_resource_format_loader(zig_loader);
}

void ResourceFormatLoaderZig::deinit() {
	ResourceLoader::get_singleton()->remove_resource_format_loader(zig_loader);
	zig_loader.unref();
}

Variant ResourceFormatLoaderZig::_load(const String &p_path, const String &original_path, bool use_sub_threads, int32_t cache_mode) const {
	Ref<ZigScript> model = memnew(ZigScript);
	model->_set_source_code(FileAccess::get_file_as_string(p_path));
	return model;
}
PackedStringArray ResourceFormatLoaderZig::_get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("zig");
	return array;
}
bool ResourceFormatLoaderZig::_handles_type(const StringName &type) const {
	String type_str = type;
	return type_str == "ZigScript" || type_str == "Script";
}
String ResourceFormatLoaderZig::_get_resource_type(const String &p_path) const {
	if (p_path.get_extension() == "zig") {
		return "ZigScript";
	}
	return "";
}

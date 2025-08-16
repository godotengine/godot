#pragma once

#include <godot_cpp/classes/resource_format_loader.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

using namespace godot;

class ResourceFormatLoaderZig : public ResourceFormatLoader {
	GDCLASS(ResourceFormatLoaderZig, ResourceFormatLoader);

protected:
	static void _bind_methods() {}

public:
	static void init();
	static void deinit();
	virtual Variant _load(const String &path, const String &original_path, bool use_sub_threads, int32_t cache_mode) const override;
	virtual PackedStringArray _get_recognized_extensions() const override;
	virtual bool _handles_type(const StringName &type) const override;
	virtual String _get_resource_type(const String &p_path) const override;
};

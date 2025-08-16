#pragma once

#include <godot_cpp/classes/resource_format_saver.hpp>
#include <godot_cpp/classes/resource_saver.hpp>

using namespace godot;

class ResourceFormatSaverRust : public ResourceFormatSaver {
	GDCLASS(ResourceFormatSaverRust, ResourceFormatSaver);

protected:
	static void _bind_methods() {}

public:
	static void init();
	static void deinit();
	virtual Error _save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) override;
	virtual Error _set_uid(const String &p_path, int64_t p_uid) override;
	virtual bool _recognize(const Ref<Resource> &p_resource) const override;
	virtual PackedStringArray _get_recognized_extensions(const Ref<Resource> &p_resource) const override;
	virtual bool _recognize_path(const Ref<Resource> &p_resource, const String &p_path) const override;
};

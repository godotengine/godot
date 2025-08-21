#pragma once

#include <godot_cpp/classes/script_extension.hpp>
#include <godot_cpp/classes/script_language.hpp>
#include <godot_cpp/templates/hash_set.hpp>

using namespace godot;
class ELFScriptInstance;
class Sandbox;
namespace godot {
	class ScriptInstanceExtension;
}

class ELFScript : public ScriptExtension {
	GDCLASS(ELFScript, ScriptExtension);

protected:
	static void _bind_methods();
	PackedByteArray source_code;
	String global_name;
	String path;
	std::string std_path;
	int source_version = 0;
	int elf_api_version;
	String elf_programming_language;

	mutable HashSet<ELFScriptInstance *> instances;
	friend class ELFScriptInstance;

	static inline HashMap<String, HashSet<Sandbox *>> sandbox_map;

public:
	Array functions;
	PackedStringArray function_names;

	void set_public_api_functions(Array &&p_functions);
	void update_public_api_functions();
	String get_elf_programming_language() const;
	int get_elf_api_version() const noexcept { return elf_api_version; }
	int get_source_version() const noexcept { return source_version; }
	String get_dockerized_program_path() const;
	const String &get_path() const noexcept { return path; }
	const std::string &get_std_path() const noexcept { return std_path; }

	/// @brief Retrieve a Sandbox instance based on a given owner object.
	/// @param p_for_object The owner object.
	/// @return The Sandbox instance, or nullptr if not found.
	Sandbox *get_sandbox_for(Object *p_for_object) const;

	/// @brief Retrieve all Objects that share a Sandbox instance that uses this ELF resource.
	/// @return An array of Objects that share a Sandbox instance.
	Array get_sandbox_objects() const;

	/// @brief Retrieve the content of the ELF resource as a byte array.
	/// @return An ELF program as a byte array.
	const PackedByteArray &get_content();

	/// @brief Get an ELFScript instance using a Node as the owner.
	/// @param p_for_object The owner Node.
	/// @return A reference to the ELFScript instance.
	ELFScriptInstance *get_script_instance(Object *p_for_object) const;

	void register_instance(Sandbox *p_sandbox) { sandbox_map[path].insert(p_sandbox); }
	void unregister_instance(Sandbox *p_sandbox) { sandbox_map[path].erase(p_sandbox); }

	virtual bool _editor_can_reload_from_file() override;
	virtual void _placeholder_erased(void *p_placeholder) override;
	virtual bool _can_instantiate() const override;
	virtual Ref<Script> _get_base_script() const override;
	virtual StringName _get_global_name() const override;
	virtual bool _inherits_script(const Ref<Script> &p_script) const override;
	virtual StringName _get_instance_base_type() const override;
	virtual void *_instance_create(Object *p_for_object) const override;
	virtual void *_placeholder_instance_create(Object *p_for_object) const override;
	virtual bool _instance_has(Object *p_object) const override;
	virtual bool _has_source_code() const override;
	virtual String _get_source_code() const override;
	virtual void _set_source_code(const String &p_code) override;
	virtual Error _reload(bool p_keep_state) override;
	virtual TypedArray<Dictionary> _get_documentation() const override;
	virtual String _get_class_icon_path() const override;
	virtual bool _has_method(const StringName &p_method) const override;
	virtual bool _has_static_method(const StringName &p_method) const override;
	virtual Dictionary _get_method_info(const StringName &p_method) const override;
	virtual bool _is_tool() const override;
	virtual bool _is_valid() const override;
	virtual bool _is_abstract() const override;
	virtual ScriptLanguage *_get_language() const override;
	virtual bool _has_script_signal(const StringName &p_signal) const override;
	virtual TypedArray<Dictionary> _get_script_signal_list() const override;
	virtual bool _has_property_default_value(const StringName &p_property) const override;
	virtual Variant _get_property_default_value(const StringName &p_property) const override;
	virtual void _update_exports() override;
	virtual TypedArray<Dictionary> _get_script_method_list() const override;
	virtual TypedArray<Dictionary> _get_script_property_list() const override;
	virtual int32_t _get_member_line(const StringName &p_member) const override;
	virtual Dictionary _get_constants() const override;
	virtual TypedArray<StringName> _get_members() const override;
	virtual bool _is_placeholder_fallback_enabled() const override;
	virtual Variant _get_rpc_config() const override;

	void set_file(const String &path);

	ELFScript() {}
	~ELFScript() {}
};

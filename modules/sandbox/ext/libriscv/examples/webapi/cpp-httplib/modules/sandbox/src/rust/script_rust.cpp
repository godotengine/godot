#include "script_rust.h"

#include "script_language_rust.h"
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

bool RustScript::_editor_can_reload_from_file() {
	return true;
}
void RustScript::_placeholder_erased(void *p_placeholder) {}
bool RustScript::_can_instantiate() const {
	return false;
}
Ref<Script> RustScript::_get_base_script() const {
	return Ref<Script>();
}
StringName RustScript::_get_global_name() const {
	return StringName();
}
bool RustScript::_inherits_script(const Ref<Script> &p_script) const {
	return false;
}
StringName RustScript::_get_instance_base_type() const {
	return StringName();
}
void *RustScript::_instance_create(Object *p_for_object) const {
	return nullptr;
}
void *RustScript::_placeholder_instance_create(Object *p_for_object) const {
	return nullptr;
}
bool RustScript::_instance_has(Object *p_object) const {
	return false;
}
bool RustScript::_has_source_code() const {
	return true;
}
String RustScript::_get_source_code() const {
	return source_code;
}
void RustScript::_set_source_code(const String &p_code) {
	source_code = p_code;
}
Error RustScript::_reload(bool p_keep_state) {
	return Error::OK;
}
TypedArray<Dictionary> RustScript::_get_documentation() const {
	return TypedArray<Dictionary>();
}
String RustScript::_get_class_icon_path() const {
	return String("res://addons/godot_sandbox/RustScript.svg");
}
bool RustScript::_has_method(const StringName &p_method) const {
	return false;
}
bool RustScript::_has_static_method(const StringName &p_method) const {
	return false;
}
Dictionary RustScript::_get_method_info(const StringName &p_method) const {
	return Dictionary();
}
bool RustScript::_is_tool() const {
	return true;
}
bool RustScript::_is_valid() const {
	return true;
}
bool RustScript::_is_abstract() const {
	return true;
}
ScriptLanguage *RustScript::_get_language() const {
	return RustScriptLanguage::get_singleton();
}
bool RustScript::_has_script_signal(const StringName &p_signal) const {
	return false;
}
TypedArray<Dictionary> RustScript::_get_script_signal_list() const {
	return TypedArray<Dictionary>();
}
bool RustScript::_has_property_default_value(const StringName &p_property) const {
	return false;
}
Variant RustScript::_get_property_default_value(const StringName &p_property) const {
	return Variant();
}
void RustScript::_update_exports() {}
TypedArray<Dictionary> RustScript::_get_script_method_list() const {
	return TypedArray<Dictionary>();
}
TypedArray<Dictionary> RustScript::_get_script_property_list() const {
	return TypedArray<Dictionary>();
}
int32_t RustScript::_get_member_line(const StringName &p_member) const {
	return 0;
}
Dictionary RustScript::_get_constants() const {
	return Dictionary();
}
TypedArray<StringName> RustScript::_get_members() const {
	return TypedArray<StringName>();
}
bool RustScript::_is_placeholder_fallback_enabled() const {
	return false;
}
Variant RustScript::_get_rpc_config() const {
	return Variant();
}

RustScript::RustScript() {
	source_code = R"C0D3(mod godot;
use godot::variant::*;

pub fn main() {
}

#[no_mangle]
pub fn public_function() -> Variant {
	let v1 = Variant::new_integer(42);
	let v2 = Variant::new_float(3.14);
	let v3 = Variant::new_string("Hello from Rust!");
	print(&[v1, v2, v3]);

	return Variant::new_string("Rust in Godot");
}
)C0D3";
}

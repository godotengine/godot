
#pragma once
#define DECL_GODOT_PROPERTY(type,name,default_value) type name = default_value; void set_##name(const type& p_value) { name = p_value; } type get_##name() const { return name; }
#define IMP_GODOT_PROPERTY(type,name) {\
ClassDB::bind_method(D_METHOD("set_" #name,#name), &self_type::set_##name);\
ClassDB::bind_method(D_METHOD("get_" #name), &self_type::get_##name); \
static type temp_var;Variant v = temp_var; \
Variant::Type var_type = v.get_type(); \
ADD_PROPERTY(PropertyInfo(var_type, #name), "set_" #name, "get_" #name);\
}
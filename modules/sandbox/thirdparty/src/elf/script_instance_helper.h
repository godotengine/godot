#include "../register_types.h"
#include <godot_cpp/templates/local_vector.hpp>
#include <godot_cpp/templates/pair.hpp>

using namespace godot;

static int get_len_from_ptr(const void *p_ptr) {
	return *((int *)p_ptr - 1);
}

static void free_with_len(void *p_ptr) {
	memfree((int *)p_ptr - 1);
}

static void free_prop(const GDExtensionPropertyInfo &p_prop) {
	// smelly
	memdelete((StringName *)p_prop.name);
	memdelete((StringName *)p_prop.class_name);
	memdelete((String *)p_prop.hint_string);
}

static String *string_alloc(const String &p_str) {
	String *ptr = memnew(String);
	*ptr = p_str;

	return ptr;
}

static StringName *stringname_alloc(const String &p_str) {
	StringName *ptr = memnew(StringName);
	*ptr = p_str;

	return ptr;
}

static GDExtensionPropertyInfo create_property_type(const Dictionary &p_src) {
	GDExtensionPropertyInfo p_dst;
	p_dst.type = (GDExtensionVariantType) int(p_src["type"]);
	p_dst.name = stringname_alloc(p_src["name"]);
	p_dst.class_name = stringname_alloc(p_src["class_name"]);
	p_dst.hint = p_src["hint"];
	p_dst.hint_string = string_alloc(p_src["hint_string"]);
	p_dst.usage = p_src["usage"];
	return p_dst;
}

static GDExtensionMethodInfo create_method_info(const MethodInfo &method_info) {
	GDExtensionMethodInfo result{
	   .name = stringname_alloc(method_info.name),
	   .return_value = GDExtensionPropertyInfo{
			   .type = (GDExtensionVariantType)method_info.return_val.type,
			   .name = stringname_alloc(method_info.return_val.name),
			   .class_name = stringname_alloc(method_info.return_val.class_name),
			   .hint = method_info.return_val.hint,
			   .hint_string = stringname_alloc(method_info.return_val.hint_string),
			   .usage = method_info.return_val.usage },
	   .flags = method_info.flags,
	   .id = method_info.id,
	   .argument_count = (uint32_t)method_info.arguments.size(),
	   .arguments = nullptr,
	   .default_argument_count = 0,
	   .default_arguments = nullptr,
   };
   if (!method_info.arguments.empty()) {
	   result.arguments = memnew_arr(GDExtensionPropertyInfo, method_info.arguments.size());
	   for (int i = 0; i < method_info.arguments.size(); i++) {
		   const PropertyInfo &arg = method_info.arguments[i];
		   result.arguments[i] = GDExtensionPropertyInfo{
				   .type = (GDExtensionVariantType)arg.type,
				   .name = stringname_alloc(arg.name),
				   .class_name = stringname_alloc(arg.class_name),
				   .hint = arg.hint,
				   .hint_string = stringname_alloc(arg.hint_string),
				   .usage = arg.usage };
	   }
   }
   return result;
}

static void add_to_state(GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value, void *p_userdata) {
	List<Pair<StringName, Variant>> *list = reinterpret_cast<List<Pair<StringName, Variant>> *>(p_userdata);
	list->push_back({ *(const StringName *)p_name, *(const Variant *)p_value });
}

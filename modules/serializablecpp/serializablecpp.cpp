#include "serializablecpp.h"

SerializableCPP::SerializableCPP(){
	serializable_ver = String(CURRENT_SERIALIZABLE_VER);
}

void SerializableCPP::_bind_methods(){
	ClassDB::bind_method(D_METHOD("get_serializable_version"), &SerializableCPP::get_serializable_version);
	ClassDB::bind_method(D_METHOD("add_properties"), &SerializableCPP::add_properties);
	ClassDB::bind_method(D_METHOD("remove_properties"), &SerializableCPP::remove_properties);
	ClassDB::bind_method(D_METHOD("copy"), &SerializableCPP::copy);
	ClassDB::bind_method(D_METHOD("serializable_dup"), &SerializableCPP::serializable_dup);

	ClassDB::bind_method(D_METHOD("set_forbidden"), &SerializableCPP::set_forbidden);
	ClassDB::bind_method(D_METHOD("set_module_name", "module_name"), &SerializableCPP::set_module_name);
	ClassDB::bind_method(D_METHOD("get_module_name"), &SerializableCPP::get_module_name);
	ClassDB::bind_method(D_METHOD("set_property_list", "property_list"), &SerializableCPP::set_property_list);
	ClassDB::bind_method(D_METHOD("get_props_list"), &SerializableCPP::get_props_list);
	ClassDB::bind_method(D_METHOD("set_dv1"/*, "debug_value1"*/), &SerializableCPP::set_dv1);
	ClassDB::bind_method(D_METHOD("get_dv1"), &SerializableCPP::get_dv1);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "module_name"), "set_module_name", "get_module_name");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_STRING_ARRAY, "property_list"), "set_property_list", "get_props_list");

}

String SerializableCPP::get_serializable_version(){
	return serializable_ver;
}
void SerializableCPP::add_properties(Vector<String> plist){
	int size = plist.size();
	for (int i = 0; i < size; i++){
		property_list.push_back(plist[i]);
	}
}
bool SerializableCPP::remove_properties(Vector<String> plist){
	bool result = false;
	int size = plist.size();
	for (int i = 0; i < size; i++){
		int find_result = property_list.find(plist[i]);
		if (find_result > -1){
			property_list.remove(find_result);
			result = true;
		}
	}
	return result;
}
bool SerializableCPP::copy(const Ref<SerializableCPP>& from){
	bool full_completion = true;
	int size = property_list.size();
	for (int i = 0; i < size; i++){
		String prop = property_list[i];
		bool result1, result2;
		Variant value = from->get("prop", &result1);
		if (!result1){
			Hub::get_singleton()->print_fatal(String("Property does not exist: ") + prop);
			full_completion = false;
			continue;
		}
		set(prop, from->get("prop"), &result2);
		if (!result2){
			Hub::get_singleton()->print_fatal(String("Failed to copy property: ") + prop);
			full_completion = false;
			continue;
		}
	}
	return full_completion;
}
Ref<SerializableCPP> SerializableCPP::serializable_dup(bool dup_property){
	// SerializableCPP *dup = new SerializableCPP();
	Ref<SerializableCPP> dup = memnew(SerializableCPP());
	dup->set_script(get_script());
	if (dup_property){
		dup->copy(Ref<SerializableCPP>(this));
	}
	// return (SerializableCPP&) dup;
	return dup;
}

void SerializableCPP::set_forbidden(Variant value){
	return;
}
void SerializableCPP::set_module_name(String new_name){
	module_name = new_name;
	emit_changed();
}
String SerializableCPP::get_module_name(){
	return module_name;
}
void SerializableCPP::set_property_list(Vector<String> plist){
	property_list = plist;
	emit_changed();
}
Vector<String> SerializableCPP::get_props_list(){
	return property_list;
}
void SerializableCPP::set_dv1(Variant value){
	debug_value1 = value;
	emit_changed();
}
Variant SerializableCPP::get_dv1(){
	return debug_value1;
}
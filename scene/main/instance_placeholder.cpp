#include "instance_placeholder.h"

#include "scene/resources/packed_scene.h"
#include "io/resource_loader.h"

bool InstancePlaceholder::_set(const StringName& p_name, const Variant& p_value) {

	PropSet ps;
	ps.name=p_name;
	ps.value=p_value;
	stored_values.push_back(ps);
	return true;
}

bool InstancePlaceholder::_get(const StringName& p_name,Variant &r_ret) const{

	return false;
}
void InstancePlaceholder::_get_property_list( List<PropertyInfo> *p_list) const{


}


void InstancePlaceholder::set_path(const String& p_name) {

	path=p_name;
}

String InstancePlaceholder::get_path() const {

	return path;
}
void InstancePlaceholder::replace_by_instance(const Ref<PackedScene> &p_custom_scene){

	ERR_FAIL_COND(!is_inside_tree());

	Node *base = get_parent();
	if (!base)
		return;

	Ref<PackedScene> ps;
	if (p_custom_scene.is_valid())
		ps = p_custom_scene;
	else
		ps = ResourceLoader::load(path,"PackedScene");

	if (!ps.is_valid())
		return;
	Node *scene = ps->instance();
	scene->set_name(get_name());
	int pos = get_position_in_parent();

	for(List<PropSet>::Element *E=stored_values.front();E;E=E->next()) {
		scene->set(E->get().name,E->get().value);
	}

	queue_delete();

	base->remove_child(this);
	base->add_child(scene);
	base->move_child(scene,pos);

}

void InstancePlaceholder::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("replace_by_instance","custom_scene:PackedScene"),&InstancePlaceholder::replace_by_instance,DEFVAL(Variant()));
}

InstancePlaceholder::InstancePlaceholder() {


}

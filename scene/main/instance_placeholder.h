#ifndef INSTANCE_PLACEHOLDER_H
#define INSTANCE_PLACEHOLDER_H

#include "scene/main/node.h"

class PackedScene;

class InstancePlaceholder : public Node {

	OBJ_TYPE(InstancePlaceholder,Node);

	String path;
	struct PropSet {
		StringName name;
		Variant value;
	};

	List<PropSet> stored_values;

protected:
	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:

	void set_path(const String& p_name);
	String get_path() const;

	void replace_by_instance(const Ref<PackedScene>& p_custom_scene=Ref<PackedScene>());

	InstancePlaceholder();
};

#endif // INSTANCE_PLACEHOLDER_H

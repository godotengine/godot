#ifndef MULTI_NODE_EDIT_H
#define MULTI_NODE_EDIT_H

#include "scene/main/node.h"

class MultiNodeEdit : public Reference {

	OBJ_TYPE(MultiNodeEdit,Reference);

	List<NodePath> nodes;
	struct PLData {
		int uses;
		PropertyInfo info;
	};

protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

public:



	void clear_nodes();
	void add_node(const NodePath& p_node);

	MultiNodeEdit();
};

#endif // MULTI_NODE_EDIT_H

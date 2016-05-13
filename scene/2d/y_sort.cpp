#include "y_sort.h"



void YSort::set_sort_enabled(bool p_enabled) {

	sort_enabled=p_enabled;
	VS::get_singleton()->canvas_item_set_sort_children_by_y(get_canvas_item(),sort_enabled);
}

bool YSort::is_sort_enabled() const {

	return sort_enabled;
}

void YSort::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_sort_enabled","enabled"),&YSort::set_sort_enabled);
	ObjectTypeDB::bind_method(_MD("is_sort_enabled"),&YSort::is_sort_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"sort/enabled"),_SCS("set_sort_enabled"),_SCS("is_sort_enabled"));
}


YSort::YSort() {

	sort_enabled=true;
	VS::get_singleton()->canvas_item_set_sort_children_by_y(get_canvas_item(),true);
}

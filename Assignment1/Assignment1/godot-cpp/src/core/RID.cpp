#include "RID.hpp"

#include <gdnative/rid.h>

#include "GodotGlobal.hpp"

namespace godot {

RID::RID() {
	godot::api->godot_rid_new(&_godot_rid);
}

RID::RID(Object *p) {
	godot::api->godot_rid_new_with_resource(&_godot_rid, (const godot_object *)p);
}

godot_rid RID::_get_godot_rid() const {
	return _godot_rid;
}

int32_t RID::get_id() const {
	return godot::api->godot_rid_get_id(&_godot_rid);
}

bool RID::operator==(const RID &p_other) const {
	return godot::api->godot_rid_operator_equal(&_godot_rid, &p_other._godot_rid);
}

bool RID::operator!=(const RID &p_other) const {
	return !(*this == p_other);
}

bool RID::operator<(const RID &p_other) const {
	return godot::api->godot_rid_operator_less(&_godot_rid, &p_other._godot_rid);
}

bool RID::operator>(const RID &p_other) const {
	return !(*this < p_other) && *this != p_other;
}

bool RID::operator<=(const RID &p_other) const {
	return (*this < p_other) || *this == p_other;
}

bool RID::operator>=(const RID &p_other) const {
	return !(*this < p_other);
}

} // namespace godot

#include "listener_2d.h"
#include "scene/main/viewport.h"

bool Listener2D::_set(const StringName& p_name, const Variant& p_value) {

	if (p_name == "current") {
		set_current(p_value.operator bool());
	}
	else
		return false;

	return true;
}

bool Listener2D::_get(const StringName& p_name,Variant &r_ret) const {

	if (p_name == "current") {
		if (is_inside_tree() && get_tree()->is_node_being_edited(this)) {
			r_ret = current;
		}
		else {
			r_ret = is_current();
		}
	}
	else
		return false;

	return true;
}

void Listener2D::_update_listener() {

	if (is_inside_tree() && is_current()) {
		get_viewport()->_listener_2d_transform_changed_notify();
	}
}

void Listener2D::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {
			bool first_listener = get_viewport()->_listener_2d_add(this);
			if (!get_tree()->is_node_being_edited(this) && (current || first_listener))
				set_current(true);
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			_update_listener();
		} break;
		case NOTIFICATION_EXIT_TREE: {

			if (!get_tree()->is_node_being_edited(this)) {
				bool curr = is_current();
				set_current(false);
				current = curr;
			}
			
			get_viewport()->_listener_2d_remove(this);
		} break;
	}
}

void Listener2D::set_current(bool p_current) {
	
	current = p_current;
	if (!is_inside_tree())
		return;
	
	if (current) {
		get_viewport()->_listener_2d_set(this);
	} else if (get_viewport()->get_listener_2d() == this) {
		get_viewport()->_listener_2d_set(NULL);
		get_viewport()->_listener_2d_make_next_current(this);
	}
}

bool Listener2D::is_current() const {

	if (is_inside_tree() && !get_tree()->is_node_being_edited(this)) {
		return get_viewport()->get_listener_2d()==this;
	} else
		return current;

	return false;
}

void Listener2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_current","current"),&Listener2D::set_current);
	ObjectTypeDB::bind_method(_MD("is_current"),&Listener2D::is_current);
	
	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"current"),_SCS("set_current"),_SCS("is_current"));
}

Listener2D::Listener2D() {

	current = false;
}

Listener2D::~Listener2D() {

}

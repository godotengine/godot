#include "listener_2d.h"

void Listener2D::_update_audio_listener_state() {


}

void Listener2D::_request_listener_update() {

	_update_listener();
}

bool Listener2D::_set(const StringName& p_name, const Variant& p_value) {

	if (p_name == "current") {
		if (p_value.operator bool()) {
			make_current();
		}
		else {
			clear_current();
		}
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

void Listener2D::_get_property_list( List<PropertyInfo> *p_list) const {

	p_list->push_back( PropertyInfo( Variant::BOOL, "current" ) );
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
				make_current();
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			_request_listener_update();
		} break;
		case NOTIFICATION_EXIT_TREE: {

			if (!get_tree()->is_node_being_edited(this)) {
				if (is_current()) {
					clear_current();
					current=true; //keep it true

				} else {
					current=false;
				}
			}

			get_viewport()->_listener_2d_remove(this);


		} break;


	}

}

Matrix32 Listener2D::get_listener_transform() const {

	return get_global_transform();
}

void Listener2D::make_current() {

	current=true;

	if (!is_inside_tree())
		return;

	get_viewport()->_listener_2d_set(this);
}

void Listener2D::clear_current() {

	current=false;
	if (!is_inside_tree())
		return;

	if (get_viewport()->get_listener_2d()==this) {
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

bool Listener2D::_can_gizmo_scale() const {

	return false;
}

RES Listener2D::_get_gizmo_geometry() const {
	
	//Ref<Mesh> mesh = memnew(Mesh);
	//return mesh;
	return NULL;
}

void Listener2D::_bind_methods() {

	ObjectTypeDB::bind_method( _MD("make_current"),&Listener2D::make_current );
	ObjectTypeDB::bind_method( _MD("clear_current"),&Listener2D::clear_current );
	ObjectTypeDB::bind_method( _MD("is_current"),&Listener2D::is_current );
	ObjectTypeDB::bind_method( _MD("get_listener_transform"),&Listener2D::get_listener_transform );
}

Listener2D::Listener2D() {

	current=false;
	force_change=false;
}

Listener2D::~Listener2D() {

}

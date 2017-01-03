#include "viewport_container.h"
#include "scene/main/viewport.h"
Size2 ViewportContainer::get_minimum_size() const {


	if (stretch)
		return Size2();
	Size2 ms;
	for(int i=0;i<get_child_count();i++) {

		Viewport *c = get_child(i)->cast_to<Viewport>();
		if (!c)
			continue;

		Size2 minsize = c->get_size();
		ms.width = MAX(ms.width , minsize.width);
		ms.height = MAX(ms.height , minsize.height);
	}

	return ms;

}


void ViewportContainer::set_stretch(bool p_enable) {

	stretch=p_enable;
	queue_sort();
	update();

}

bool ViewportContainer::is_stretch_enabled() const {

	return stretch;
}


void ViewportContainer::_notification(int p_what) {


	if (p_what==NOTIFICATION_RESIZED) {

		if (!stretch)
			return;

		for(int i=0;i<get_child_count();i++) {

			Viewport *c = get_child(i)->cast_to<Viewport>();
			if (!c)
				continue;

			c->set_size(get_size());
		}
	}

	if (p_what==NOTIFICATION_ENTER_TREE || p_what==NOTIFICATION_VISIBILITY_CHANGED) {

		for(int i=0;i<get_child_count();i++) {

			Viewport *c = get_child(i)->cast_to<Viewport>();
			if (!c)
				continue;


			if (is_visible())
				c->set_update_mode(Viewport::UPDATE_ALWAYS);
			else
				c->set_update_mode(Viewport::UPDATE_DISABLED);
		}

	}

	if (p_what==NOTIFICATION_DRAW) {

		for(int i=0;i<get_child_count();i++) {


			Viewport *c = get_child(i)->cast_to<Viewport>();
			if (!c)
				continue;

			if (stretch)
				draw_texture_rect(c->get_texture(),Rect2(Vector2(),get_size()*Size2(1,-1)));
			else
				draw_texture_rect(c->get_texture(),Rect2(Vector2(),c->get_size()*Size2(1,-1)));
		}
	}

}

void ViewportContainer::_bind_methods() {

	ClassDB::bind_method(_MD("set_stretch","enable"),&ViewportContainer::set_stretch);
	ClassDB::bind_method(_MD("is_stretch_enabled"),&ViewportContainer::is_stretch_enabled);

	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"stretch"),_SCS("set_stretch"),_SCS("is_stretch_enabled"));
}

ViewportContainer::ViewportContainer() {

	stretch=false;
}

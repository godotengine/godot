#include "canvas_modulate.h"


void CanvasModulate::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_CANVAS) {

		if (is_visible())
			VS::get_singleton()->canvas_set_modulate(get_canvas(),color);
	} else if (p_what==NOTIFICATION_EXIT_CANVAS) {

		if (is_visible())
			VS::get_singleton()->canvas_set_modulate(get_canvas(),Color(1,1,1,1));
	} else if (p_what==NOTIFICATION_VISIBILITY_CHANGED) {

		if (is_visible()) {
			VS::get_singleton()->canvas_set_modulate(get_canvas(),color);
		} else {
			VS::get_singleton()->canvas_set_modulate(get_canvas(),Color(1,1,1,1));
		}
	}
}

void CanvasModulate::_bind_methods(){

	ObjectTypeDB::bind_method(_MD("set_color","color"),&CanvasModulate::set_color);
	ObjectTypeDB::bind_method(_MD("get_color"),&CanvasModulate::get_color);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR,"color"),_SCS("set_color"),_SCS("get_color"));
}


void CanvasModulate::set_color(const Color& p_color){

	color=p_color;
	if (is_inside_tree()) {
		VS::get_singleton()->canvas_set_modulate(get_canvas(),color);
	}
}
Color CanvasModulate::get_color() const {

	return color;
}


CanvasModulate::CanvasModulate()
{
	color=Color(1,1,1,1);
}

CanvasModulate::~CanvasModulate()
{

}


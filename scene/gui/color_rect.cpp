#include "color_rect.h"




void ColorRect::set_frame_color(const Color& p_color) {

	color=p_color;
	update();
}

Color ColorRect::get_frame_color() const{

	return color;
}

void ColorRect::_notification(int p_what) {

	if (p_what==NOTIFICATION_DRAW) {
		draw_rect(Rect2(Point2(),get_size()),color);
	}
}

void ColorRect::_bind_methods() {

	ClassDB::bind_method(_MD("set_frame_color","color"),&ColorRect::set_frame_color);
	ClassDB::bind_method(_MD("get_frame_color"),&ColorRect::get_frame_color);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR,"color"),_SCS("set_frame_color"),_SCS("get_frame_color") );
}

ColorRect::ColorRect() {

	color=Color(1,1,1);
}


#include "color_rect.h"

void ColorFrame::set_frame_color(const Color &p_color) {

	color = p_color;
	update();
}

Color ColorFrame::get_frame_color() const {

	return color;
}

void ColorFrame::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {
		draw_rect(Rect2(Point2(), get_size()), color);
	}
}

void ColorFrame::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_frame_color", "color"), &ColorFrame::set_frame_color);
	ObjectTypeDB::bind_method(_MD("get_frame_color"), &ColorFrame::get_frame_color);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), _SCS("set_frame_color"), _SCS("get_frame_color"));
}

ColorFrame::ColorFrame() {

	color = Color(1, 1, 1);
}

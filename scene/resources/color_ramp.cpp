/*
 * color_ramp.h
 */

#include "color_ramp.h"

//setter and getter names for property serialization
#define COLOR_RAMP_GET_OFFSETS "get_offsets"
#define COLOR_RAMP_GET_COLORS "get_colors"
#define COLOR_RAMP_SET_OFFSETS "set_offsets"
#define COLOR_RAMP_SET_COLORS "set_colors"

ColorRamp::ColorRamp() {
	//Set initial color ramp transition from black to white
	points.resize(2);
	points[0].color = Color(0,0,0,1);
	points[0].offset = 0;
	points[1].color = Color(1,1,1,1);
	points[1].offset = 1;
	is_sorted = true;
}

ColorRamp::~ColorRamp() {

}

void ColorRamp::_bind_methods() {

	ObjectTypeDB::bind_method(_MD(COLOR_RAMP_SET_OFFSETS,"offsets"),&ColorRamp::set_offsets);
	ObjectTypeDB::bind_method(_MD(COLOR_RAMP_GET_OFFSETS),&ColorRamp::get_offsets);

	ObjectTypeDB::bind_method(_MD(COLOR_RAMP_SET_COLORS,"colors"),&ColorRamp::set_colors);
	ObjectTypeDB::bind_method(_MD(COLOR_RAMP_GET_COLORS),&ColorRamp::get_colors);

	ADD_PROPERTY( PropertyInfo(Variant::REAL,"offsets"),_SCS(COLOR_RAMP_SET_OFFSETS),_SCS(COLOR_RAMP_GET_OFFSETS) );
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"colors"),_SCS(COLOR_RAMP_SET_COLORS),_SCS(COLOR_RAMP_GET_COLORS) );
}

Vector<float> ColorRamp::get_offsets() const {
	Vector<float> offsets;
	offsets.resize(points.size());
	for(int i = 0; i < points.size(); i++)
	{
		offsets[i] = points[i].offset;
	}
	return offsets;
}

Vector<Color> ColorRamp::get_colors() const {
	Vector<Color> colors;
	colors.resize(points.size());
	for(int i = 0; i < points.size(); i++)
	{
		colors[i] = points[i].color;
	}
	return colors;
}

void ColorRamp::set_offsets(const Vector<float>& p_offsets) {
	points.resize(p_offsets.size());
	for(int i = 0; i < points.size(); i++)
	{
		points[i].offset = p_offsets[i];
	}
	is_sorted = false;
}

void ColorRamp::set_colors(const Vector<Color>& p_colors) {
	if(points.size()<p_colors.size())
		is_sorted = false;
	points.resize(p_colors.size());
	for(int i = 0; i < points.size(); i++)
	{
		points[i].color = p_colors[i];
	}
}

Vector<ColorRamp::Point>& ColorRamp::get_points() {
	return points;
}

void ColorRamp::set_points(Vector<ColorRamp::Point>& p_points) {
	points = p_points;
	is_sorted = false;
}

void ColorRamp::set_offset(int pos, const float offset) {
	if(points.size() <= pos)
		points.resize(pos + 1);
	points[pos].offset = offset;
	is_sorted = false;
}

float ColorRamp::get_offset(int pos) const {
	if(points.size() > pos)
		return points[pos].offset;
	return 0;  //TODO: Maybe throw some error instead?
}

void ColorRamp::set_color(int pos, const Color& color) {
	if(points.size() <= pos)
	{
		points.resize(pos + 1);
		is_sorted = false;
	}
	points[pos].color = color;
}

Color ColorRamp::get_color(int pos) const {
	if(points.size() > pos)
		return points[pos].color;
	return Color(0,0,0,1); //TODO: Maybe throw some error instead?
}

int ColorRamp::get_points_count() const {
	return points.size();
}

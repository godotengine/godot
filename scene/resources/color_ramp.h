/*
 * color_ramp.h
 */

#ifndef SCENE_RESOURCES_COLOR_RAMP_H_
#define SCENE_RESOURCES_COLOR_RAMP_H_

#include "resource.h"

class ColorRamp: public Resource {
	OBJ_TYPE( ColorRamp, Resource );
	OBJ_SAVE_TYPE( ColorRamp );

public:
	struct Point {

		float offset;
		Color color;
		bool operator<(const Point& p_ponit) const {
			return offset<p_ponit.offset;
		}
	};

private:
	Vector<Point> points;
	bool is_sorted;

protected:
	static void _bind_methods();

public:
	ColorRamp();
	virtual ~ColorRamp();

	void set_points(Vector<Point>& points);
	Vector<Point>& get_points();

	void set_offset(int pos, const float offset);
	float get_offset(int pos) const;

	void set_color(int pos, const Color& color);
	Color get_color(int pos) const;

	void set_offsets(const Vector<float>& offsets);
	Vector<float> get_offsets() const;

	void set_colors(const Vector<Color>& colors);
	Vector<Color> get_colors() const;

	Color get_color_at_offset(float offset);

	int get_points_count() const;
};

#endif /* SCENE_RESOURCES_COLOR_RAMP_H_ */

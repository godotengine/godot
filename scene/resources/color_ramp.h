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

	void add_point(float p_offset, const Color& p_color);
	void remove_point(int p_index);

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

	_FORCE_INLINE_ Color get_color_at_offset(float p_offset) {

		if (points.empty())
			return Color(0,0,0,1);

		if(!is_sorted)
		{
			points.sort();
			is_sorted = true;
		}

		//binary search
		int low = 0;
		int high = points.size() -1;
		int middle;

		while( low <= high )
		{
			middle = ( low  + high ) / 2;
			Point& point = points[middle];
			if( point.offset > p_offset ) {
				high = middle - 1; //search low end of array
			} else if ( point.offset < p_offset) {
				low = middle + 1; //search high end of array
			} else {
				return point.color;
			}
		}

		//return interpolated value
		if (points[middle].offset>p_offset)
		{
			middle--;
		}
		int first=middle;
		int second=middle+1;
		if(second>=points.size())
			return points[points.size()-1].color;
		if(first<0)
			return points[0].color;
		Point& pointFirst = points[first];
		Point& pointSecond = points[second];
		return pointFirst.color.linear_interpolate(pointSecond.color, (p_offset-pointFirst.offset)/(pointSecond.offset - pointFirst.offset));
	}

	int get_points_count() const;
};

#endif /* SCENE_RESOURCES_COLOR_RAMP_H_ */

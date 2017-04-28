/*************************************************************************/
/*  color_ramp.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef SCENE_RESOURCES_COLOR_RAMP_H_
#define SCENE_RESOURCES_COLOR_RAMP_H_

#include "resource.h"

class ColorRamp : public Resource {
	GDCLASS(ColorRamp, Resource);
	OBJ_SAVE_TYPE(ColorRamp);

public:
	struct Point {

		float offset;
		Color color;
		bool operator<(const Point &p_ponit) const {
			return offset < p_ponit.offset;
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

	void add_point(float p_offset, const Color &p_color);
	void remove_point(int p_index);

	void set_points(Vector<Point> &points);
	Vector<Point> &get_points();

	void set_offset(int pos, const float offset);
	float get_offset(int pos) const;

	void set_color(int pos, const Color &color);
	Color get_color(int pos) const;

	void set_offsets(const Vector<float> &offsets);
	Vector<float> get_offsets() const;

	void set_colors(const Vector<Color> &colors);
	Vector<Color> get_colors() const;

	_FORCE_INLINE_ Color get_color_at_offset(float p_offset) {

		if (points.empty())
			return Color(0, 0, 0, 1);

		if (!is_sorted) {
			points.sort();
			is_sorted = true;
		}

		//binary search
		int low = 0;
		int high = points.size() - 1;
		int middle;

		while (low <= high) {
			middle = (low + high) / 2;
			Point &point = points[middle];
			if (point.offset > p_offset) {
				high = middle - 1; //search low end of array
			} else if (point.offset < p_offset) {
				low = middle + 1; //search high end of array
			} else {
				return point.color;
			}
		}

		//return interpolated value
		if (points[middle].offset > p_offset) {
			middle--;
		}
		int first = middle;
		int second = middle + 1;
		if (second >= points.size())
			return points[points.size() - 1].color;
		if (first < 0)
			return points[0].color;
		Point &pointFirst = points[first];
		Point &pointSecond = points[second];
		return pointFirst.color.linear_interpolate(pointSecond.color, (p_offset - pointFirst.offset) / (pointSecond.offset - pointFirst.offset));
	}

	int get_points_count() const;
};

#endif /* SCENE_RESOURCES_COLOR_RAMP_H_ */

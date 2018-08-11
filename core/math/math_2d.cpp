/*************************************************************************/
/*  math_2d.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "math_2d.h"

/* Point2i */

Point2i Point2i::operator+(const Point2i &p_v) const {

	return Point2i(x + p_v.x, y + p_v.y);
}
void Point2i::operator+=(const Point2i &p_v) {

	x += p_v.x;
	y += p_v.y;
}
Point2i Point2i::operator-(const Point2i &p_v) const {

	return Point2i(x - p_v.x, y - p_v.y);
}
void Point2i::operator-=(const Point2i &p_v) {

	x -= p_v.x;
	y -= p_v.y;
}

Point2i Point2i::operator*(const Point2i &p_v1) const {

	return Point2i(x * p_v1.x, y * p_v1.y);
};

Point2i Point2i::operator*(const int &rvalue) const {

	return Point2i(x * rvalue, y * rvalue);
};
void Point2i::operator*=(const int &rvalue) {

	x *= rvalue;
	y *= rvalue;
};

Point2i Point2i::operator/(const Point2i &p_v1) const {

	return Point2i(x / p_v1.x, y / p_v1.y);
};

Point2i Point2i::operator/(const int &rvalue) const {

	return Point2i(x / rvalue, y / rvalue);
};

void Point2i::operator/=(const int &rvalue) {

	x /= rvalue;
	y /= rvalue;
};

Point2i Point2i::operator-() const {

	return Point2i(-x, -y);
}

bool Point2i::operator==(const Point2i &p_vec2) const {

	return x == p_vec2.x && y == p_vec2.y;
}
bool Point2i::operator!=(const Point2i &p_vec2) const {

	return x != p_vec2.x || y != p_vec2.y;
}

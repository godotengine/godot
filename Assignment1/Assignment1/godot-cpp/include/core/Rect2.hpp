#ifndef RECT2_H
#define RECT2_H

#include "Vector2.hpp"

#include <cmath>

#include <cstdlib>

namespace godot {

class String;

typedef Vector2 Size2;
typedef Vector2 Point2;

struct Transform2D;

struct Rect2 {

	Point2 position;
	Size2 size;

	inline const Vector2 &get_position() const { return position; }
	inline void set_position(const Vector2 &p_position) { position = p_position; }
	inline const Vector2 &get_size() const { return size; }
	inline void set_size(const Vector2 &p_size) { size = p_size; }

	inline real_t get_area() const { return size.width * size.height; }

	inline bool intersects(const Rect2 &p_rect) const {
		if (position.x >= (p_rect.position.x + p_rect.size.width))
			return false;
		if ((position.x + size.width) <= p_rect.position.x)
			return false;
		if (position.y >= (p_rect.position.y + p_rect.size.height))
			return false;
		if ((position.y + size.height) <= p_rect.position.y)
			return false;

		return true;
	}

	real_t distance_to(const Vector2 &p_point) const;

	bool intersects_transformed(const Transform2D &p_xform, const Rect2 &p_rect) const;

	bool intersects_segment(const Point2 &p_from, const Point2 &p_to, Point2 *r_position = nullptr, Point2 *r_normal = nullptr) const;

	inline bool encloses(const Rect2 &p_rect) const {

		return (p_rect.position.x >= position.x) && (p_rect.position.y >= position.y) &&
			   ((p_rect.position.x + p_rect.size.x) < (position.x + size.x)) &&
			   ((p_rect.position.y + p_rect.size.y) < (position.y + size.y));
	}

	inline bool has_no_area() const {

		return (size.x <= 0 || size.y <= 0);
	}
	Rect2 clip(const Rect2 &p_rect) const;

	Rect2 merge(const Rect2 &p_rect) const;

	inline bool has_point(const Point2 &p_point) const {
		if (p_point.x < position.x)
			return false;
		if (p_point.y < position.y)
			return false;

		if (p_point.x >= (position.x + size.x))
			return false;
		if (p_point.y >= (position.y + size.y))
			return false;

		return true;
	}

	inline bool no_area() const { return (size.width <= 0 || size.height <= 0); }

	inline bool operator==(const Rect2 &p_rect) const { return position == p_rect.position && size == p_rect.size; }
	inline bool operator!=(const Rect2 &p_rect) const { return position != p_rect.position || size != p_rect.size; }

	inline Rect2 grow(real_t p_by) const {

		Rect2 g = *this;
		g.position.x -= p_by;
		g.position.y -= p_by;
		g.size.width += p_by * 2;
		g.size.height += p_by * 2;
		return g;
	}

	inline Rect2 expand(const Vector2 &p_vector) const {

		Rect2 r = *this;
		r.expand_to(p_vector);
		return r;
	}

	inline void expand_to(const Vector2 &p_vector) { //in place function for speed

		Vector2 begin = position;
		Vector2 end = position + size;

		if (p_vector.x < begin.x)
			begin.x = p_vector.x;
		if (p_vector.y < begin.y)
			begin.y = p_vector.y;

		if (p_vector.x > end.x)
			end.x = p_vector.x;
		if (p_vector.y > end.y)
			end.y = p_vector.y;

		position = begin;
		size = end - begin;
	}

	operator String() const;

	inline Rect2() {}
	inline Rect2(real_t p_x, real_t p_y, real_t p_width, real_t p_height) {
		position = Point2(p_x, p_y);
		size = Size2(p_width, p_height);
	}
	inline Rect2(const Point2 &p_position, const Size2 &p_size) {
		position = p_position;
		size = p_size;
	}
};

} // namespace godot

#endif // RECT2_H

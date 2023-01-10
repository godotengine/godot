/**************************************************************************/
/*  bit_map.cpp                                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "bit_map.h"

#include "core/io/image_loader.h"

void BitMap::create(const Size2 &p_size) {
	ERR_FAIL_COND(p_size.width < 1);
	ERR_FAIL_COND(p_size.height < 1);

	width = p_size.width;
	height = p_size.height;
	bitmask.resize((((width * height) - 1) / 8) + 1);
	memset(bitmask.ptrw(), 0, bitmask.size());
}

void BitMap::create_from_image_alpha(const Ref<Image> &p_image, float p_threshold) {
	ERR_FAIL_COND(p_image.is_null() || p_image->empty());
	Ref<Image> img = p_image->duplicate();
	img->convert(Image::FORMAT_LA8);
	ERR_FAIL_COND(img->get_format() != Image::FORMAT_LA8);

	create(Size2(img->get_width(), img->get_height()));

	PoolVector<uint8_t>::Read r = img->get_data().read();
	uint8_t *w = bitmask.ptrw();

	for (int i = 0; i < width * height; i++) {
		int bbyte = i / 8;
		int bbit = i % 8;
		if (r[i * 2 + 1] / 255.0 > p_threshold) {
			w[bbyte] |= (1 << bbit);
		}
	}
}

void BitMap::set_bit_rect(const Rect2 &p_rect, bool p_value) {
	Rect2i current = Rect2i(0, 0, width, height).clip(p_rect);
	uint8_t *data = bitmask.ptrw();

	for (int i = current.position.x; i < current.position.x + current.size.x; i++) {
		for (int j = current.position.y; j < current.position.y + current.size.y; j++) {
			int ofs = width * j + i;
			int bbyte = ofs / 8;
			int bbit = ofs % 8;

			uint8_t b = data[bbyte];

			if (p_value) {
				b |= (1 << bbit);
			} else {
				b &= ~(1 << bbit);
			}

			data[bbyte] = b;
		}
	}
}

int BitMap::get_true_bit_count() const {
	int ds = bitmask.size();
	const uint8_t *d = bitmask.ptr();
	int c = 0;

	//fast, almost branchless version

	for (int i = 0; i < ds; i++) {
		c += (d[i] & (1 << 7)) >> 7;
		c += (d[i] & (1 << 6)) >> 6;
		c += (d[i] & (1 << 5)) >> 5;
		c += (d[i] & (1 << 4)) >> 4;
		c += (d[i] & (1 << 3)) >> 3;
		c += (d[i] & (1 << 2)) >> 2;
		c += (d[i] & (1 << 1)) >> 1;
		c += d[i] & 1;
	}

	return c;
}

void BitMap::set_bit(const Point2 &p_pos, bool p_value) {
	int x = p_pos.x;
	int y = p_pos.y;

	ERR_FAIL_INDEX(x, width);
	ERR_FAIL_INDEX(y, height);

	int ofs = width * y + x;
	int bbyte = ofs / 8;
	int bbit = ofs % 8;

	uint8_t b = bitmask[bbyte];

	if (p_value) {
		b |= (1 << bbit);
	} else {
		b &= ~(1 << bbit);
	}

	bitmask.write[bbyte] = b;
}

bool BitMap::get_bit(const Point2 &p_pos) const {
	int x = Math::fast_ftoi(p_pos.x);
	int y = Math::fast_ftoi(p_pos.y);
	ERR_FAIL_INDEX_V(x, width, false);
	ERR_FAIL_INDEX_V(y, height, false);

	int ofs = width * y + x;
	int bbyte = ofs / 8;
	int bbit = ofs % 8;

	return (bitmask[bbyte] & (1 << bbit)) != 0;
}

Size2 BitMap::get_size() const {
	return Size2(width, height);
}

void BitMap::_set_data(const Dictionary &p_d) {
	ERR_FAIL_COND(!p_d.has("size"));
	ERR_FAIL_COND(!p_d.has("data"));

	create(p_d["size"]);
	bitmask = p_d["data"];
}

Dictionary BitMap::_get_data() const {
	Dictionary d;
	d["size"] = get_size();
	d["data"] = bitmask;
	return d;
}

Vector<Vector2> BitMap::_march_square(const Rect2i &rect, const Point2i &start) const {
	int stepx = 0;
	int stepy = 0;
	int prevx = 0;
	int prevy = 0;
	int startx = start.x;
	int starty = start.y;
	int curx = startx;
	int cury = starty;
	unsigned int count = 0;
	Set<Point2i> case9s;
	Set<Point2i> case6s;
	Vector<Vector2> _points;
	do {
		int sv = 0;
		{ //square value

			/*
			checking the 2x2 pixel grid, assigning these values to each pixel, if not transparent
			+---+---+
			| 1 | 2 |
			+---+---+
			| 4 | 8 | <- current pixel (curx,cury)
			+---+---+
			*/
			Point2i tl = Point2i(curx - 1, cury - 1);
			sv += (rect.has_point(tl) && get_bit(tl)) ? 1 : 0;
			Point2i tr = Point2i(curx, cury - 1);
			sv += (rect.has_point(tr) && get_bit(tr)) ? 2 : 0;
			Point2i bl = Point2i(curx - 1, cury);
			sv += (rect.has_point(bl) && get_bit(bl)) ? 4 : 0;
			Point2i br = Point2i(curx, cury);
			sv += (rect.has_point(br) && get_bit(br)) ? 8 : 0;
			ERR_FAIL_COND_V(sv == 0 || sv == 15, Vector<Vector2>());
		}

		switch (sv) {
			case 1:
			case 5:
			case 13:
				/* going UP with these cases:
				1          5           13
				+---+---+  +---+---+  +---+---+
				| 1 |   |  | 1 |   |  | 1 |   |
				+---+---+  +---+---+  +---+---+
				|   |   |  | 4 |   |  | 4 | 8 |
				+---+---+  +---+---+  +---+---+
				*/
				stepx = 0;
				stepy = -1;
				break;

			case 8:
			case 10:
			case 11:
				/* going DOWN with these cases:
				8          10         11
				+---+---+  +---+---+  +---+---+
				|   |   |  |   | 2 |  | 1 | 2 |
				+---+---+  +---+---+  +---+---+
				|   | 8 |  |   | 8 |  |   | 8 |
				+---+---+  +---+---+  +---+---+
				*/
				stepx = 0;
				stepy = 1;
				break;

			case 4:
			case 12:
			case 14:
				/* going LEFT with these cases:
				4          12         14
				+---+---+  +---+---+  +---+---+
				|   |   |  |   |   |  |   | 2 |
				+---+---+  +---+---+  +---+---+
				| 4 |   |  | 4 | 8 |  | 4 | 8 |
				+---+---+  +---+---+  +---+---+
				*/
				stepx = -1;
				stepy = 0;
				break;

			case 2:
			case 3:
			case 7:
				/* going RIGHT with these cases:
				2          3          7
				+---+---+  +---+---+  +---+---+
				|   | 2 |  | 1 | 2 |  | 1 | 2 |
				+---+---+  +---+---+  +---+---+
				|   |   |  |   |   |  | 4 |   |
				+---+---+  +---+---+  +---+---+
				*/
				stepx = 1;
				stepy = 0;
				break;
			case 9:
				/*
				+---+---+
				| 1 |   |
				+---+---+
				|   | 8 |
				+---+---+
				this should normally go UP, but if we already been here, we go down
				*/
				if (case9s.has(Point2i(curx, cury))) {
					//found, so we go down, and delete from case9s;
					stepx = 0;
					stepy = 1;
					case9s.erase(Point2i(curx, cury));
				} else {
					//not found, we go up, and add to case9s;
					stepx = 0;
					stepy = -1;
					case9s.insert(Point2i(curx, cury));
				}
				break;
			case 6:
				/*
				6
				+---+---+
				|   | 2 |
				+---+---+
				| 4 |   |
				+---+---+
				this normally go RIGHT, but if its coming from RIGHT, it should go LEFT
				*/
				if (case6s.has(Point2i(curx, cury))) {
					//found, so we go left, and delete from case6s;
					stepx = -1;
					stepy = 0;
					case6s.erase(Point2i(curx, cury));
				} else {
					//not found, we go right, and add to case6s;
					stepx = 1;
					stepy = 0;
					case6s.insert(Point2i(curx, cury));
				}
				break;
			default:
				ERR_PRINT("this shouldn't happen.");
		}
		//little optimization
		// if previous direction is same as current direction,
		// then we should modify the last vec to current
		curx += stepx;
		cury += stepy;
		if (stepx == prevx && stepy == prevy) {
			_points.write[_points.size() - 1].x = (float)(curx - rect.position.x);
			_points.write[_points.size() - 1].y = (float)(cury + rect.position.y);
		} else {
			_points.push_back(Vector2((float)(curx - rect.position.x), (float)(cury + rect.position.y)));
		}

		count++;
		prevx = stepx;
		prevy = stepy;

		ERR_FAIL_COND_V((int)count > width * height, _points);
	} while (curx != startx || cury != starty);
	return _points;
}

static float perpendicular_distance(const Vector2 &i, const Vector2 &start, const Vector2 &end) {
	float res;
	float slope;
	float intercept;

	if (start.x == end.x) {
		res = Math::absf(i.x - end.x);
	} else if (start.y == end.y) {
		res = Math::absf(i.y - end.y);
	} else {
		slope = (end.y - start.y) / (end.x - start.x);
		intercept = start.y - (slope * start.x);
		res = Math::absf(slope * i.x - i.y + intercept) / Math::sqrt(Math::pow(slope, 2.0f) + 1.0);
	}
	return res;
}

static Vector<Vector2> rdp(const Vector<Vector2> &v, float optimization) {
	if (v.size() < 3) {
		return v;
	}

	int index = -1;
	float dist = 0;
	//not looping first and last point
	for (size_t i = 1, size = v.size(); i < size - 1; ++i) {
		float cdist = perpendicular_distance(v[i], v[0], v[v.size() - 1]);
		if (cdist > dist) {
			dist = cdist;
			index = static_cast<int>(i);
		}
	}
	if (dist > optimization) {
		Vector<Vector2> left, right;
		left.resize(index);
		for (int i = 0; i < index; i++) {
			left.write[i] = v[i];
		}
		right.resize(v.size() - index);
		for (int i = 0; i < right.size(); i++) {
			right.write[i] = v[index + i];
		}
		Vector<Vector2> r1 = rdp(left, optimization);
		Vector<Vector2> r2 = rdp(right, optimization);

		int middle = r1.size();
		r1.resize(r1.size() + r2.size());
		for (int i = 0; i < r2.size(); i++) {
			r1.write[middle + i] = r2[i];
		}
		return r1;
	} else {
		Vector<Vector2> ret;
		ret.push_back(v[0]);
		ret.push_back(v[v.size() - 1]);
		return ret;
	}
}

static Vector<Vector2> reduce(const Vector<Vector2> &points, const Rect2i &rect, float epsilon) {
	int size = points.size();
	// if there are less than 3 points, then we have nothing
	ERR_FAIL_COND_V(size < 3, Vector<Vector2>());
	// if there are less than 9 points (but more than 3), then we don't need to reduce it
	if (size < 9) {
		return points;
	}

	float maxEp = MIN(rect.size.width, rect.size.height);
	float ep = CLAMP(epsilon, 0.0, maxEp / 2);
	Vector<Vector2> result = rdp(points, ep);

	Vector2 last = result[result.size() - 1];

	if (last.y > result[0].y && last.distance_to(result[0]) < ep * 0.5f) {
		result.write[0].y = last.y;
		result.resize(result.size() - 1);
	}
	return result;
}

struct FillBitsStackEntry {
	Point2i pos;
	int i;
	int j;
};

static void fill_bits(const BitMap *p_src, Ref<BitMap> &p_map, const Point2i &p_pos, const Rect2i &rect) {
	// Using a custom stack to work iteratively to avoid stack overflow on big bitmaps
	PoolVector<FillBitsStackEntry> stack;
	// Tracking size since we won't be shrinking the stack vector
	int stack_size = 0;

	Point2i pos = p_pos;
	int next_i = 0;
	int next_j = 0;

	bool reenter = true;
	bool popped = false;
	do {
		if (reenter) {
			next_i = pos.x - 1;
			next_j = pos.y - 1;
			reenter = false;
		}

		for (int i = next_i; i <= pos.x + 1; i++) {
			for (int j = next_j; j <= pos.y + 1; j++) {
				if (popped) {
					// The next loop over j must start normally
					next_j = pos.y;
					popped = false;
					// Skip because an iteration was already executed with current counter values
					continue;
				}

				if (i < rect.position.x || i >= rect.position.x + rect.size.x) {
					continue;
				}
				if (j < rect.position.y || j >= rect.position.y + rect.size.y) {
					continue;
				}

				if (p_map->get_bit(Vector2(i, j))) {
					continue;

				} else if (p_src->get_bit(Vector2(i, j))) {
					p_map->set_bit(Vector2(i, j), true);

					FillBitsStackEntry se = { pos, i, j };
					stack.resize(MAX(stack_size + 1, stack.size()));
					stack.set(stack_size, se);
					stack_size++;

					pos = Point2i(i, j);
					reenter = true;
					break;
				}
			}
			if (reenter) {
				break;
			}
		}
		if (!reenter) {
			if (stack_size) {
				FillBitsStackEntry se = stack.get(stack_size - 1);
				stack_size--;
				pos = se.pos;
				next_i = se.i;
				next_j = se.j;
				popped = true;
			}
		}
	} while (reenter || popped);

	print_verbose("BitMap: Max stack size: " + itos(stack.size()));
}

Vector<Vector<Vector2>> BitMap::clip_opaque_to_polygons(const Rect2 &p_rect, float p_epsilon) const {
	Rect2i r = Rect2i(0, 0, width, height).clip(p_rect);
	print_verbose("BitMap: Rect: " + r);

	Point2i from;
	Ref<BitMap> fill;
	fill.instance();
	fill->create(get_size());

	Vector<Vector<Vector2>> polygons;
	for (int i = r.position.y; i < r.position.y + r.size.height; i++) {
		for (int j = r.position.x; j < r.position.x + r.size.width; j++) {
			if (!fill->get_bit(Point2(j, i)) && get_bit(Point2(j, i))) {
				fill_bits(this, fill, Point2i(j, i), r);

				Vector<Vector2> polygon = _march_square(r, Point2i(j, i));
				print_verbose("BitMap: Pre reduce: " + itos(polygon.size()));
				polygon = reduce(polygon, r, p_epsilon);
				print_verbose("BitMap: Post reduce: " + itos(polygon.size()));

				if (polygon.size() < 3) {
					print_verbose("Invalid polygon, skipped");
					continue;
				}

				polygons.push_back(polygon);
			}
		}
	}

	return polygons;
}

void BitMap::grow_mask(int p_pixels, const Rect2 &p_rect) {
	if (p_pixels == 0) {
		return;
	}

	bool bit_value = p_pixels > 0;
	p_pixels = Math::abs(p_pixels);

	Rect2i r = Rect2i(0, 0, width, height).clip(p_rect);

	Ref<BitMap> copy;
	copy.instance();
	copy->create(get_size());
	copy->bitmask = bitmask;

	for (int i = r.position.y; i < r.position.y + r.size.height; i++) {
		for (int j = r.position.x; j < r.position.x + r.size.width; j++) {
			if (bit_value == get_bit(Point2(j, i))) {
				continue;
			}

			bool found = false;

			for (int y = i - p_pixels; y <= i + p_pixels; y++) {
				for (int x = j - p_pixels; x <= j + p_pixels; x++) {
					bool outside = false;

					if ((x < p_rect.position.x) || (x >= p_rect.position.x + p_rect.size.x) || (y < p_rect.position.y) || (y >= p_rect.position.y + p_rect.size.y)) {
						// outside of rectangle counts as bit not set
						if (!bit_value) {
							outside = true;
						} else {
							continue;
						}
					}

					float d = Point2(j, i).distance_to(Point2(x, y)) - CMP_EPSILON;
					if (d > p_pixels) {
						continue;
					}

					if (outside || (bit_value == copy->get_bit(Point2(x, y)))) {
						found = true;
						break;
					}
				}
				if (found) {
					break;
				}
			}

			if (found) {
				set_bit(Point2(j, i), bit_value);
			}
		}
	}
}

void BitMap::shrink_mask(int p_pixels, const Rect2 &p_rect) {
	grow_mask(-p_pixels, p_rect);
}

Array BitMap::_opaque_to_polygons_bind(const Rect2 &p_rect, float p_epsilon) const {
	Vector<Vector<Vector2>> result = clip_opaque_to_polygons(p_rect, p_epsilon);

	// Convert result to bindable types

	Array result_array;
	result_array.resize(result.size());
	for (int i = 0; i < result.size(); i++) {
		const Vector<Vector2> &polygon = result[i];

		PoolVector2Array polygon_array;
		polygon_array.resize(polygon.size());

		{
			PoolVector2Array::Write w = polygon_array.write();
			for (int j = 0; j < polygon.size(); j++) {
				w[j] = polygon[j];
			}
		}

		result_array[i] = polygon_array;
	}

	return result_array;
}

void BitMap::resize(const Size2 &p_new_size) {
	Ref<BitMap> new_bitmap;
	new_bitmap.instance();
	new_bitmap->create(p_new_size);
	int lw = MIN(width, new_bitmap->width);
	int lh = MIN(height, new_bitmap->height);
	for (int x = 0; x < lw; x++) {
		for (int y = 0; y < lh; y++) {
			new_bitmap->set_bit(Vector2(x, y), get_bit(Vector2(x, y)));
		}
	}

	width = new_bitmap->width;
	height = new_bitmap->height;
	bitmask = new_bitmap->bitmask;
}

Ref<Image> BitMap::convert_to_image() const {
	Ref<Image> image;
	image.instance();
	image->create(width, height, false, Image::FORMAT_L8);
	image->lock();
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			image->set_pixel(i, j, get_bit(Point2(i, j)) ? Color(1, 1, 1) : Color(0, 0, 0));
		}
	}

	image->unlock();

	return image;
}
void BitMap::blit(const Vector2 &p_pos, const Ref<BitMap> &p_bitmap) {
	int x = p_pos.x;
	int y = p_pos.y;
	int w = p_bitmap->get_size().width;
	int h = p_bitmap->get_size().height;

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			int px = x + i;
			int py = y + j;
			if (px < 0 || px >= width) {
				continue;
			}
			if (py < 0 || py >= height) {
				continue;
			}
			if (p_bitmap->get_bit(Vector2(i, j))) {
				set_bit(Vector2(x, y), true);
			}
		}
	}
}

void BitMap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create", "size"), &BitMap::create);
	ClassDB::bind_method(D_METHOD("create_from_image_alpha", "image", "threshold"), &BitMap::create_from_image_alpha, DEFVAL(0.1));

	ClassDB::bind_method(D_METHOD("set_bit", "position", "bit"), &BitMap::set_bit);
	ClassDB::bind_method(D_METHOD("get_bit", "position"), &BitMap::get_bit);

	ClassDB::bind_method(D_METHOD("set_bit_rect", "rect", "bit"), &BitMap::set_bit_rect);
	ClassDB::bind_method(D_METHOD("get_true_bit_count"), &BitMap::get_true_bit_count);

	ClassDB::bind_method(D_METHOD("get_size"), &BitMap::get_size);
	ClassDB::bind_method(D_METHOD("resize", "new_size"), &BitMap::resize);

	ClassDB::bind_method(D_METHOD("_set_data"), &BitMap::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &BitMap::_get_data);

	ClassDB::bind_method(D_METHOD("grow_mask", "pixels", "rect"), &BitMap::grow_mask);
	ClassDB::bind_method(D_METHOD("convert_to_image"), &BitMap::convert_to_image);
	ClassDB::bind_method(D_METHOD("opaque_to_polygons", "rect", "epsilon"), &BitMap::_opaque_to_polygons_bind, DEFVAL(2.0));

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");
}

BitMap::BitMap() {
	width = 0;
	height = 0;
}

//////////////////////////////////////

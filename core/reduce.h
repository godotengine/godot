/*************************************************************************/
/*  reduce.h                                                             */
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

struct Min {
	struct reduce_number {
		_FORCE_INLINE_ static Variant empty() {
			return Variant();
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			return p_a < p_b ? p_a : p_b;
		}
	};

	struct reduce_vector2 {
		_FORCE_INLINE_ static Variant empty() {
			return Variant();
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			const Vector2 u = p_a;
			const Vector2 v = p_b;
			return Vector2(MIN(u.x, v.x), MIN(u.y, v.y));
		}
	};

	struct reduce_vector3 {
		_FORCE_INLINE_ static Variant empty() {
			return Variant();
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			const Vector3 u = p_a;
			const Vector3 v = p_b;
			return Vector3(MIN(u.x, v.x), MIN(u.y, v.y), MIN(u.z, v.z));
		}
	};

	struct reduce_color {
		_FORCE_INLINE_ static Variant empty() {
			return Variant();
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			const Color u = p_a;
			const Color v = p_b;
			return Color(MIN(u.r, v.r), MIN(u.g, v.g), MIN(u.b, v.b), MIN(u.a, v.a));
		}
	};

	_FORCE_INLINE_ static Variant empty() {
		return Variant();
	}

	_FORCE_INLINE_ static Variant reduce(const Variant &p_a, const Variant &p_b, bool &r_valid) {
		const Variant::Type t_a = p_a.get_type();
		const Variant::Type t_b = p_b.get_type();
		if (t_a == t_b) {
			switch (t_a) {
				case Variant::INT: {
					return ((int64_t)p_a) < ((int64_t)p_b) ? p_a : p_b;
				} break;
				case Variant::REAL: {
					return ((real_t)p_a) < ((real_t)p_b) ? p_a : p_b;
				} break;
				case Variant::VECTOR2: {
					return reduce_vector2::reduce(p_a, p_b);
				} break;
				case Variant::VECTOR3: {
					return reduce_vector3::reduce(p_a, p_b);
				} break;
				case Variant::COLOR: {
					return reduce_color::reduce(p_a, p_b);
				} break;
			}
		} else if (p_a.is_num() && p_b.is_num()) {
			return ((real_t)p_a) < ((real_t)p_b) ? p_a : p_b;
		}

		r_valid = false;
		return Variant();
	}
};

struct Max {
	struct reduce_number {
		_FORCE_INLINE_ static Variant empty() {
			return Variant();
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			return p_a > p_b ? p_a : p_b;
		}
	};

	struct reduce_vector2 {
		_FORCE_INLINE_ static Variant empty() {
			return Variant();
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			const Vector2 u = p_a;
			const Vector2 v = p_b;
			return Vector2(MAX(u.x, v.x), MAX(u.y, v.y));
		}
	};

	struct reduce_vector3 {
		_FORCE_INLINE_ static Variant empty() {
			return Variant();
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			const Vector3 u = p_a;
			const Vector3 v = p_b;
			return Vector3(MAX(u.x, v.x), MAX(u.y, v.y), MAX(u.z, v.z));
		}
	};

	struct reduce_color {
		_FORCE_INLINE_ static Variant empty() {
			return Variant();
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			const Color u = p_a;
			const Color v = p_b;
			return Color(MAX(u.r, v.r), MAX(u.g, v.g), MAX(u.b, v.b), MAX(u.a, v.a));
		}
	};

	_FORCE_INLINE_ static Variant empty() {
		return Variant();
	}

	_FORCE_INLINE_ static Variant reduce(const Variant &p_a, const Variant &p_b, bool &r_valid) {
		const Variant::Type t_a = p_a.get_type();
		const Variant::Type t_b = p_b.get_type();
		if (t_a == t_b) {
			switch (t_a) {
				case Variant::INT: {
					return ((int64_t)p_a) > ((int64_t)p_b) ? p_a : p_b;
				} break;
				case Variant::REAL: {
					return ((real_t)p_a) > ((real_t)p_b) ? p_a : p_b;
				} break;
				case Variant::VECTOR2: {
					return reduce_vector2::reduce(p_a, p_b);
				} break;
				case Variant::VECTOR3: {
					return reduce_vector3::reduce(p_a, p_b);
				} break;
				case Variant::COLOR: {
					return reduce_color::reduce(p_a, p_b);
				} break;
			}
		} else if (p_a.is_num() && p_b.is_num()) {
			return ((real_t)p_a) > ((real_t)p_b) ? p_a : p_b;
		}

		r_valid = false;
		return Variant();
	}
};

struct Sum {
	struct reduce_number {
		_FORCE_INLINE_ static Variant empty() {
			return 0;
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			return p_a + p_b;
		}
	};

	struct reduce_vector2 {
		_FORCE_INLINE_ static Variant empty() {
			return Vector2(0, 0);
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			const Vector2 u = p_a;
			const Vector2 v = p_b;
			return Vector2(u.x + v.x, u.y + v.y);
		}
	};

	struct reduce_vector3 {
		_FORCE_INLINE_ static Variant empty() {
			return Vector3(0, 0, 0);
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			const Vector3 u = p_a;
			const Vector3 v = p_b;
			return Vector3(u.x + v.x, u.y + v.y, u.z + v.z);
		}
	};

	struct reduce_color {
		_FORCE_INLINE_ static Variant empty() {
			return Color(0, 0, 0, 0);
		}

		template <typename T>
		_FORCE_INLINE_ static T reduce(const T &p_a, const T &p_b) {
			const Color u = p_a;
			const Color v = p_b;
			return Color(u.r + v.r, u.g + v.g, u.b + v.b, u.a + v.a);
		}
	};

	_FORCE_INLINE_ static Variant empty() {
		return 0;
	}

	_FORCE_INLINE_ static Variant reduce(const Variant &p_a, const Variant &p_b, bool &r_valid) {
		const Variant::Type t_a = p_a.get_type();
		const Variant::Type t_b = p_b.get_type();
		if (t_a == t_b) {
			switch (t_a) {
				case Variant::INT: {
					return ((int64_t)p_a) + ((int64_t)p_b);
				} break;
				case Variant::REAL: {
					return ((real_t)p_a) + ((real_t)p_b);
				} break;
				case Variant::VECTOR2: {
					return reduce_vector2::reduce(p_a, p_b);
				} break;
				case Variant::VECTOR3: {
					return reduce_vector3::reduce(p_a, p_b);
				} break;
				case Variant::COLOR: {
					return reduce_color::reduce(p_a, p_b);
				} break;
			}
		} else if (p_a.is_num() && p_b.is_num()) {
			return ((real_t)p_a) + ((real_t)p_b);
		}

		r_valid = false;
		return Variant();
	}
};

template <typename R>
Variant reduce_array(const Variant &p_variant);

template <typename Reduce, typename Array>
Variant reduce_variants(const Array &p_arr, int p_len) {

	if (p_len < 1) {
		return Reduce::empty();
	}

	Variant value = p_arr[0];
	if (value.is_array()) {
		value = reduce_array<Reduce>(value);
	}

	Variant reduced;
	for (int i = 1; i < p_len; i++) {
		const Variant *x = &p_arr[i];

		// recursive descend, e.g. handle [1, [2, 3]]
		if (x->is_array()) {
			reduced = reduce_array<Reduce>(*x);
			x = &reduced;
		}

		bool valid = true;
		value = Reduce::reduce(value, *x, valid);
		if (!valid) {
			return Variant();
		}
	}

	return value;
}

template <typename T, typename Reduce, typename Array>
Variant reduce_pool_vector(const Array &p_arr, int p_len);

template <typename R>
Variant reduce_array(const Variant &p_variant) {

	switch (p_variant.get_type()) {
		case Variant::ARRAY: {

			const Array arr = p_variant.operator Array();
			return reduce_variants<R>(arr, arr.size());
		} break;
		case Variant::POOL_BYTE_ARRAY: {

			const PoolVector<uint8_t> arr = p_variant.operator PoolVector<uint8_t>();
			PoolVector<uint8_t>::Read r = arr.read();
			return reduce_pool_vector<int64_t, typename R::reduce_number>(r.ptr(), arr.size());
		} break;
		case Variant::POOL_INT_ARRAY: {

			const PoolVector<int> arr = p_variant.operator PoolVector<int>();
			PoolVector<int>::Read r = arr.read();
			return reduce_pool_vector<int64_t, typename R::reduce_number>(r.ptr(), arr.size());
		} break;
		case Variant::POOL_REAL_ARRAY: {

			const PoolVector<real_t> arr = p_variant.operator PoolVector<real_t>();
			PoolVector<real_t>::Read r = arr.read();
			return reduce_pool_vector<real_t, typename R::reduce_number>(r.ptr(), arr.size());
		} break;
		case Variant::POOL_VECTOR2_ARRAY: {

			const PoolVector<Vector2> arr = p_variant.operator PoolVector<Vector2>();
			PoolVector<Vector2>::Read r = arr.read();
			return reduce_pool_vector<Vector2, typename R::reduce_vector2>(r.ptr(), arr.size());
		} break;
		case Variant::POOL_VECTOR3_ARRAY: {

			const PoolVector<Vector3> arr = p_variant.operator PoolVector<Vector3>();
			PoolVector<Vector3>::Read r = arr.read();
			return reduce_pool_vector<Vector3, typename R::reduce_vector3>(r.ptr(), arr.size());
		} break;
		case Variant::POOL_COLOR_ARRAY: {

			const PoolVector<Color> arr = p_variant.operator PoolVector<Color>();
			PoolVector<Color>::Read r = arr.read();
			return reduce_pool_vector<Color, typename R::reduce_color>(r.ptr(), arr.size());
		} break;
	}

	return p_variant; // identity
}

template <typename T, typename Reduce, typename Array>
Variant reduce_pool_vector(const Array &p_arr, int p_len) {

	if (p_len < 1) {
		return Reduce::empty();
	}
	T value = p_arr[0];
	for (int i = 1; i < p_len; i++) {
		value = Reduce::template reduce<T>(value, p_arr[i]);
	}
	return value;
}

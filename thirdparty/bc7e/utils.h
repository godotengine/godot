// File: utils.h
#pragma once
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable:4127) // conditional expression is constant
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <time.h>
#include <vector>
#include <string>
#include <random>
#include <utility>
#include <limits.h>
#include "dds_defs.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ASSUME(c) static_assert(c, #c)
#define ARRAY_SIZE(a) (sizeof(a)/sizeof(a[0]))

#define VECTOR_TEXT_LINE_SIZE (30.0f)
#define VECTOR_TEXT_CORE_LINE_SIZE (21.0f)

#define UNUSED(x) (void)x

namespace utils
{
extern const uint32_t g_pretty_colors[];
extern const uint32_t g_num_pretty_colors;

const float cDegToRad = 0.01745329252f;
const float cRadToDeg = 57.29577951f;

enum eClear { cClear };
enum eZero { cZero };
enum eInitExpand { cInitExpand };

inline int iabs(int i) { if (i < 0) i = -i; return i; }
inline uint8_t clamp255(int32_t i) { return (uint8_t)((i & 0xFFFFFF00U) ? (~(i >> 31)) : i); }
template <typename S> inline S clamp(S value, S low, S high) { return (value < low) ? low : ((value > high) ? high : value); }
template<typename F> inline F lerp(F a, F b, F s) { return a + (b - a) * s; }
template<typename F> inline F square(F a) { return a * a; }

template <class T>
inline T prev_wrap(T i, T n)
{
	T temp = i - 1;
	if (temp < 0)
		temp = n - 1;
	return temp;
}

template <class T>
inline T next_wrap(T i, T n)
{
	T temp = i + 1;
	if (temp >= n)
		temp = 0;
	return temp;
}

inline int posmod(int x, int y)
{
	if (x >= 0)
		return (x < y) ? x : (x % y);
	int m = (-x) % y;
	return (m != 0) ? (y - m) : m;
}

inline float deg_to_rad(float f)
{
	return f * cDegToRad;
};

inline float rad_to_deg(float f)
{
	return f * cRadToDeg;
};

template <typename T>
struct rel_ops
{
	friend bool operator!=(const T& x, const T& y)
	{
		return (!(x == y));
	}
	friend bool operator>(const T& x, const T& y)
	{
		return (y < x);
	}
	friend bool operator<=(const T& x, const T& y)
	{
		return (!(y < x));
	}
	friend bool operator>=(const T& x, const T& y)
	{
		return (!(x < y));
	}
};

template <uint32_t N, typename T>
class vec : public rel_ops<vec<N, T> >
{
public:
	typedef T scalar_type;
	enum
	{
		num_elements = N
	};

	inline vec()
	{
	}

	inline vec(eClear)
	{
		clear();
	}

	inline vec(const vec& other)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] = other.m_s[i];
	}

	template <uint32_t O, typename U>
	inline vec(const vec<O, U>& other)
	{
		set(other);
	}

	template <uint32_t O, typename U>
	inline vec(const vec<O, U>& other, T w)
	{
		*this = other;
		m_s[N - 1] = w;
	}

	explicit inline vec(T val)
	{
		set(val);
	}

	inline vec(T val0, T val1)
	{
		set(val0, val1);
	}

	inline vec(T val0, T val1, T val2)
	{
		set(val0, val1, val2);
	}

	inline vec(T val0, T val1, T val2, T val3)
	{
		set(val0, val1, val2, val3);
	}

	inline vec(T val0, T val1, T val2, T val3, T val4, T val5)
	{
		set(val0, val1, val2, val3, val4, val5);
	}

	inline vec(
		T val0, T val1, T val2, T val3,
		T val4, T val5, T val6, T val7,
		T val8, T val9, T val10, T val11,
		T val12, T val13, T val14, T val15)
	{
		set(val0, val1, val2, val3,
			val4, val5, val6, val7,
			val8, val9, val10, val11,
			val12, val13, val14, val15);
	}

	inline vec(
		T val0, T val1, T val2, T val3,
		T val4, T val5, T val6, T val7,
		T val8, T val9, T val10, T val11,
		T val12, T val13, T val14, T val15,
		T val16, T val17, T val18, T val19)
	{
		set(val0, val1, val2, val3,
			val4, val5, val6, val7,
			val8, val9, val10, val11,
			val12, val13, val14, val15,
			val16, val17, val18, val19);
	}

	inline vec(
		T val0, T val1, T val2, T val3,
		T val4, T val5, T val6, T val7,
		T val8, T val9, T val10, T val11,
		T val12, T val13, T val14, T val15,
		T val16, T val17, T val18, T val19,
		T val20, T val21, T val22, T val23,
		T val24)
	{
		set(val0, val1, val2, val3,
			val4, val5, val6, val7,
			val8, val9, val10, val11,
			val12, val13, val14, val15,
			val16, val17, val18, val19,
			val20, val21, val22, val23,
			val24);
	}

	inline void clear()
	{
		if (N > 4)
			memset(m_s, 0, sizeof(m_s));
		else
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] = 0;
		}
	}

	template <uint32_t ON, typename OT>
	inline vec& set(const vec<ON, OT>& other)
	{
		if ((void*)this == (void*)&other)
			return *this;
		const uint32_t m = std::min(N, ON);
		uint32_t i;
		for (i = 0; i < m; i++)
			m_s[i] = static_cast<T>(other[i]);
		for (; i < N; i++)
			m_s[i] = 0;
		return *this;
	}

	inline vec& set_component(uint32_t index, T val)
	{
		assert(index < N);
		m_s[index] = val;
		return *this;
	}

	inline vec& set(T val)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] = val;
		return *this;
	}

	inline vec& set(T val0, T val1)
	{
		m_s[0] = val0;
		if (N >= 2)
		{
			m_s[1] = val1;

			for (uint32_t i = 2; i < N; i++)
				m_s[i] = 0;
		}
		return *this;
	}

	inline vec& set(T val0, T val1, T val2)
	{
		m_s[0] = val0;
		if (N >= 2)
		{
			m_s[1] = val1;

			if (N >= 3)
			{
				m_s[2] = val2;

				for (uint32_t i = 3; i < N; i++)
					m_s[i] = 0;
			}
		}
		return *this;
	}

	inline vec& set(T val0, T val1, T val2, T val3)
	{
		m_s[0] = val0;
		if (N >= 2)
		{
			m_s[1] = val1;

			if (N >= 3)
			{
				m_s[2] = val2;

				if (N >= 4)
				{
					m_s[3] = val3;

					for (uint32_t i = 4; i < N; i++)
						m_s[i] = 0;
				}
			}
		}
		return *this;
	}

	inline vec& set(T val0, T val1, T val2, T val3, T val4, T val5)
	{
		m_s[0] = val0;
		if (N >= 2)
		{
			m_s[1] = val1;

			if (N >= 3)
			{
				m_s[2] = val2;

				if (N >= 4)
				{
					m_s[3] = val3;

					if (N >= 5)
					{
						m_s[4] = val4;

						if (N >= 6)
						{
							m_s[5] = val5;

							for (uint32_t i = 6; i < N; i++)
								m_s[i] = 0;
						}
					}
				}
			}
		}
		return *this;
	}

	inline vec& set(
		T val0, T val1, T val2, T val3,
		T val4, T val5, T val6, T val7,
		T val8, T val9, T val10, T val11,
		T val12, T val13, T val14, T val15)
	{
		m_s[0] = val0;
		if (N >= 2)
			m_s[1] = val1;
		if (N >= 3)
			m_s[2] = val2;
		if (N >= 4)
			m_s[3] = val3;

		if (N >= 5)
			m_s[4] = val4;
		if (N >= 6)
			m_s[5] = val5;
		if (N >= 7)
			m_s[6] = val6;
		if (N >= 8)
			m_s[7] = val7;

		if (N >= 9)
			m_s[8] = val8;
		if (N >= 10)
			m_s[9] = val9;
		if (N >= 11)
			m_s[10] = val10;
		if (N >= 12)
			m_s[11] = val11;

		if (N >= 13)
			m_s[12] = val12;
		if (N >= 14)
			m_s[13] = val13;
		if (N >= 15)
			m_s[14] = val14;
		if (N >= 16)
			m_s[15] = val15;

		for (uint32_t i = 16; i < N; i++)
			m_s[i] = 0;

		return *this;
	}

	inline vec& set(
		T val0, T val1, T val2, T val3,
		T val4, T val5, T val6, T val7,
		T val8, T val9, T val10, T val11,
		T val12, T val13, T val14, T val15,
		T val16, T val17, T val18, T val19)
	{
		m_s[0] = val0;
		if (N >= 2)
			m_s[1] = val1;
		if (N >= 3)
			m_s[2] = val2;
		if (N >= 4)
			m_s[3] = val3;

		if (N >= 5)
			m_s[4] = val4;
		if (N >= 6)
			m_s[5] = val5;
		if (N >= 7)
			m_s[6] = val6;
		if (N >= 8)
			m_s[7] = val7;

		if (N >= 9)
			m_s[8] = val8;
		if (N >= 10)
			m_s[9] = val9;
		if (N >= 11)
			m_s[10] = val10;
		if (N >= 12)
			m_s[11] = val11;

		if (N >= 13)
			m_s[12] = val12;
		if (N >= 14)
			m_s[13] = val13;
		if (N >= 15)
			m_s[14] = val14;
		if (N >= 16)
			m_s[15] = val15;

		if (N >= 17)
			m_s[16] = val16;
		if (N >= 18)
			m_s[17] = val17;
		if (N >= 19)
			m_s[18] = val18;
		if (N >= 20)
			m_s[19] = val19;

		for (uint32_t i = 20; i < N; i++)
			m_s[i] = 0;

		return *this;
	}

	inline vec& set(
		T val0, T val1, T val2, T val3,
		T val4, T val5, T val6, T val7,
		T val8, T val9, T val10, T val11,
		T val12, T val13, T val14, T val15,
		T val16, T val17, T val18, T val19,
		T val20, T val21, T val22, T val23,
		T val24)
	{
		m_s[0] = val0;
		if (N >= 2)
			m_s[1] = val1;
		if (N >= 3)
			m_s[2] = val2;
		if (N >= 4)
			m_s[3] = val3;

		if (N >= 5)
			m_s[4] = val4;
		if (N >= 6)
			m_s[5] = val5;
		if (N >= 7)
			m_s[6] = val6;
		if (N >= 8)
			m_s[7] = val7;

		if (N >= 9)
			m_s[8] = val8;
		if (N >= 10)
			m_s[9] = val9;
		if (N >= 11)
			m_s[10] = val10;
		if (N >= 12)
			m_s[11] = val11;

		if (N >= 13)
			m_s[12] = val12;
		if (N >= 14)
			m_s[13] = val13;
		if (N >= 15)
			m_s[14] = val14;
		if (N >= 16)
			m_s[15] = val15;

		if (N >= 17)
			m_s[16] = val16;
		if (N >= 18)
			m_s[17] = val17;
		if (N >= 19)
			m_s[18] = val18;
		if (N >= 20)
			m_s[19] = val19;

		if (N >= 21)
			m_s[20] = val20;
		if (N >= 22)
			m_s[21] = val21;
		if (N >= 23)
			m_s[22] = val22;
		if (N >= 24)
			m_s[23] = val23;

		if (N >= 25)
			m_s[24] = val24;

		for (uint32_t i = 25; i < N; i++)
			m_s[i] = 0;

		return *this;
	}

	inline vec& set(const T* pValues)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] = pValues[i];
		return *this;
	}

	template <uint32_t ON, typename OT>
	inline vec& swizzle_set(const vec<ON, OT>& other, uint32_t i)
	{
		return set(static_cast<T>(other[i]));
	}

	template <uint32_t ON, typename OT>
	inline vec& swizzle_set(const vec<ON, OT>& other, uint32_t i, uint32_t j)
	{
		return set(static_cast<T>(other[i]), static_cast<T>(other[j]));
	}

	template <uint32_t ON, typename OT>
	inline vec& swizzle_set(const vec<ON, OT>& other, uint32_t i, uint32_t j, uint32_t k)
	{
		return set(static_cast<T>(other[i]), static_cast<T>(other[j]), static_cast<T>(other[k]));
	}

	template <uint32_t ON, typename OT>
	inline vec& swizzle_set(const vec<ON, OT>& other, uint32_t i, uint32_t j, uint32_t k, uint32_t l)
	{
		return set(static_cast<T>(other[i]), static_cast<T>(other[j]), static_cast<T>(other[k]), static_cast<T>(other[l]));
	}

	inline vec& operator=(const vec& rhs)
	{
		if (this != &rhs)
		{
			for (uint32_t i = 0; i < N; i++)
				m_s[i] = rhs.m_s[i];
		}
		return *this;
	}

	template <uint32_t O, typename U>
	inline vec& operator=(const vec<O, U>& other)
	{
		if ((void*)this == (void*)&other)
			return *this;

		uint32_t s = std::min(N, O);

		uint32_t i;
		for (i = 0; i < s; i++)
			m_s[i] = static_cast<T>(other[i]);

		for (; i < N; i++)
			m_s[i] = 0;

		return *this;
	}

	inline bool operator==(const vec& rhs) const
	{
		for (uint32_t i = 0; i < N; i++)
			if (!(m_s[i] == rhs.m_s[i]))
				return false;
		return true;
	}

	inline bool operator<(const vec& rhs) const
	{
		for (uint32_t i = 0; i < N; i++)
		{
			if (m_s[i] < rhs.m_s[i])
				return true;
			else if (!(m_s[i] == rhs.m_s[i]))
				return false;
		}

		return false;
	}

	inline T operator[](uint32_t i) const
	{
		assert(i < N);
		return m_s[i];
	}

	inline T& operator[](uint32_t i)
	{
		assert(i < N);
		return m_s[i];
	}

	template <uint32_t index>
	inline uint64_t get_component_as_uint() const
	{
		ASSUME(index < N);
		if (sizeof(T) == sizeof(float))
			return *reinterpret_cast<const uint32_t*>(&m_s[index]);
		else
			return *reinterpret_cast<const uint64_t*>(&m_s[index]);
	}

	inline T get_x(void) const
	{
		return m_s[0];
	}
	inline T get_y(void) const
	{
		ASSUME(N >= 2);
		return m_s[1];
	}
	inline T get_z(void) const
	{
		ASSUME(N >= 3);
		return m_s[2];
	}
	inline T get_w(void) const
	{
		ASSUME(N >= 4);
		return m_s[3];
	}

	inline vec get_x_vector() const
	{
		return broadcast<0>();
	}
	inline vec get_y_vector() const
	{
		return broadcast<1>();
	}
	inline vec get_z_vector() const
	{
		return broadcast<2>();
	}
	inline vec get_w_vector() const
	{
		return broadcast<3>();
	}

	inline T get_component(uint32_t i) const
	{
		return (*this)[i];
	}

	inline vec& set_x(T v)
	{
		m_s[0] = v;
		return *this;
	}
	inline vec& set_y(T v)
	{
		ASSUME(N >= 2);
		m_s[1] = v;
		return *this;
	}
	inline vec& set_z(T v)
	{
		ASSUME(N >= 3);
		m_s[2] = v;
		return *this;
	}
	inline vec& set_w(T v)
	{
		ASSUME(N >= 4);
		m_s[3] = v;
		return *this;
	}

	inline const T* get_ptr() const
	{
		return reinterpret_cast<const T*>(&m_s[0]);
	}
	inline T* get_ptr()
	{
		return reinterpret_cast<T*>(&m_s[0]);
	}

	inline vec as_point() const
	{
		vec result(*this);
		result[N - 1] = 1;
		return result;
	}

	inline vec as_dir() const
	{
		vec result(*this);
		result[N - 1] = 0;
		return result;
	}

	inline vec<2, T> select2(uint32_t i, uint32_t j) const
	{
		assert((i < N) && (j < N));
		return vec<2, T>(m_s[i], m_s[j]);
	}

	inline vec<3, T> select3(uint32_t i, uint32_t j, uint32_t k) const
	{
		assert((i < N) && (j < N) && (k < N));
		return vec<3, T>(m_s[i], m_s[j], m_s[k]);
	}

	inline vec<4, T> select4(uint32_t i, uint32_t j, uint32_t k, uint32_t l) const
	{
		assert((i < N) && (j < N) && (k < N) && (l < N));
		return vec<4, T>(m_s[i], m_s[j], m_s[k], m_s[l]);
	}

	inline bool is_dir() const
	{
		return m_s[N - 1] == 0;
	}
	inline bool is_vector() const
	{
		return is_dir();
	}
	inline bool is_point() const
	{
		return m_s[N - 1] == 1;
	}

	inline vec project() const
	{
		vec result(*this);
		if (result[N - 1])
			result /= result[N - 1];
		return result;
	}

	inline vec broadcast(unsigned i) const
	{
		return vec((*this)[i]);
	}

	template <uint32_t i>
	inline vec broadcast() const
	{
		return vec((*this)[i]);
	}

	inline vec swizzle(uint32_t i, uint32_t j) const
	{
		return vec((*this)[i], (*this)[j]);
	}

	inline vec swizzle(uint32_t i, uint32_t j, uint32_t k) const
	{
		return vec((*this)[i], (*this)[j], (*this)[k]);
	}

	inline vec swizzle(uint32_t i, uint32_t j, uint32_t k, uint32_t l) const
	{
		return vec((*this)[i], (*this)[j], (*this)[k], (*this)[l]);
	}

	inline vec operator-() const
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result.m_s[i] = -m_s[i];
		return result;
	}

	inline vec operator+() const
	{
		return *this;
	}

	inline vec& operator+=(const vec& other)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] += other.m_s[i];
		return *this;
	}

	inline vec& operator-=(const vec& other)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] -= other.m_s[i];
		return *this;
	}

	inline vec& operator*=(const vec& other)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] *= other.m_s[i];
		return *this;
	}

	inline vec& operator/=(const vec& other)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] /= other.m_s[i];
		return *this;
	}

	inline vec& operator*=(T s)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] *= s;
		return *this;
	}

	inline vec& operator/=(T s)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] /= s;
		return *this;
	}

	// component-wise multiply (not a dot product like in previous versions)
	// just remarking it out because it's too ambiguous, use dot() or mul_components() instead
#if 0
	friend inline vec operator*(const vec& lhs, const vec& rhs)
	{
		return vec::mul_components(lhs, rhs);
	}
#endif

	friend inline vec operator*(const vec& lhs, T val)
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result.m_s[i] = lhs.m_s[i] * val;
		return result;
	}

	friend inline vec operator*(T val, const vec& rhs)
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result.m_s[i] = val * rhs.m_s[i];
		return result;
	}

	friend inline vec operator/(const vec& lhs, const vec& rhs)
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result.m_s[i] = lhs.m_s[i] / rhs.m_s[i];
		return result;
	}

	friend inline vec operator/(const vec& lhs, T val)
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result.m_s[i] = lhs.m_s[i] / val;
		return result;
	}

	friend inline vec operator+(const vec& lhs, const vec& rhs)
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result.m_s[i] = lhs.m_s[i] + rhs.m_s[i];
		return result;
	}

	friend inline vec operator-(const vec& lhs, const vec& rhs)
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result.m_s[i] = lhs.m_s[i] - rhs.m_s[i];
		return result;
	}

	static inline vec<3, T> cross2(const vec& a, const vec& b)
	{
		ASSUME(N >= 2);
		return vec<3, T>(0, 0, a[0] * b[1] - a[1] * b[0]);
	}

	inline vec<3, T> cross2(const vec& b) const
	{
		return cross2(*this, b);
	}

	static inline vec<3, T> cross3(const vec& a, const vec& b)
	{
		ASSUME(N >= 3);
		return vec<3, T>(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
	}

	inline vec<3, T> cross3(const vec& b) const
	{
		return cross3(*this, b);
	}

	static inline vec<3, T> cross(const vec& a, const vec& b)
	{
		ASSUME(N >= 2);

		if (N == 2)
			return cross2(a, b);
		else
			return cross3(a, b);
	}

	inline vec<3, T> cross(const vec& b) const
	{
		ASSUME(N >= 2);
		return cross(*this, b);
	}

	inline T dot(const vec& rhs) const
	{
		return dot(*this, rhs);
	}

	inline vec dot_vector(const vec& rhs) const
	{
		return vec(dot(*this, rhs));
	}

	static inline T dot(const vec& lhs, const vec& rhs)
	{
		T result = lhs.m_s[0] * rhs.m_s[0];
		for (uint32_t i = 1; i < N; i++)
			result += lhs.m_s[i] * rhs.m_s[i];
		return result;
	}

	inline T dot2(const vec& rhs) const
	{
		ASSUME(N >= 2);
		return m_s[0] * rhs.m_s[0] + m_s[1] * rhs.m_s[1];
	}

	inline T dot3(const vec& rhs) const
	{
		ASSUME(N >= 3);
		return m_s[0] * rhs.m_s[0] + m_s[1] * rhs.m_s[1] + m_s[2] * rhs.m_s[2];
	}

	inline T dot4(const vec& rhs) const
	{
		ASSUME(N >= 4);
		return m_s[0] * rhs.m_s[0] + m_s[1] * rhs.m_s[1] + m_s[2] * rhs.m_s[2] + m_s[3] * rhs.m_s[3];
	}

	inline T norm(void) const
	{
		T sum = m_s[0] * m_s[0];
		for (uint32_t i = 1; i < N; i++)
			sum += m_s[i] * m_s[i];
		return sum;
	}

	inline T length(void) const
	{
		return sqrt(norm());
	}

	inline T squared_distance(const vec& rhs) const
	{
		T dist2 = 0;
		for (uint32_t i = 0; i < N; i++)
		{
			T d = m_s[i] - rhs.m_s[i];
			dist2 += d * d;
		}
		return dist2;
	}

	inline T squared_distance(const vec& rhs, T early_out) const
	{
		T dist2 = 0;
		for (uint32_t i = 0; i < N; i++)
		{
			T d = m_s[i] - rhs.m_s[i];
			dist2 += d * d;
			if (dist2 > early_out)
				break;
		}
		return dist2;
	}

	inline T distance(const vec& rhs) const
	{
		T dist2 = 0;
		for (uint32_t i = 0; i < N; i++)
		{
			T d = m_s[i] - rhs.m_s[i];
			dist2 += d * d;
		}
		return sqrt(dist2);
	}

	inline vec inverse() const
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result[i] = m_s[i] ? (1.0f / m_s[i]) : 0;
		return result;
	}

	// returns squared length (norm)
	inline double normalize(const vec* pDefaultVec = NULL)
	{
		double n = m_s[0] * m_s[0];
		for (uint32_t i = 1; i < N; i++)
			n += m_s[i] * m_s[i];

		if (n != 0)
			*this *= static_cast<T>(1.0f / sqrt(n));
		else if (pDefaultVec)
			*this = *pDefaultVec;
		return n;
	}

	inline double normalize3(const vec* pDefaultVec = NULL)
	{
		ASSUME(N >= 3);

		double n = m_s[0] * m_s[0] + m_s[1] * m_s[1] + m_s[2] * m_s[2];

		if (n != 0)
			*this *= static_cast<T>((1.0f / sqrt(n)));
		else if (pDefaultVec)
			*this = *pDefaultVec;
		return n;
	}

	inline vec& normalize_in_place(const vec* pDefaultVec = NULL)
	{
		normalize(pDefaultVec);
		return *this;
	}

	inline vec& normalize3_in_place(const vec* pDefaultVec = NULL)
	{
		normalize3(pDefaultVec);
		return *this;
	}

	inline vec get_normalized(const vec* pDefaultVec = NULL) const
	{
		vec result(*this);
		result.normalize(pDefaultVec);
		return result;
	}

	inline vec get_normalized3(const vec* pDefaultVec = NULL) const
	{
		vec result(*this);
		result.normalize3(pDefaultVec);
		return result;
	}

	inline vec& clamp(T l, T h)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] = static_cast<T>(clamp(m_s[i], l, h));
		return *this;
	}

	inline vec& saturate()
	{
		return clamp(0.0f, 1.0f);
	}

	inline vec& clamp(const vec& l, const vec& h)
	{
		for (uint32_t i = 0; i < N; i++)
			m_s[i] = static_cast<T>(clamp(m_s[i], l[i], h[i]));
		return *this;
	}

	inline bool is_within_bounds(const vec& l, const vec& h) const
	{
		for (uint32_t i = 0; i < N; i++)
			if ((m_s[i] < l[i]) || (m_s[i] > h[i]))
				return false;

		return true;
	}

	inline bool is_within_bounds(T l, T h) const
	{
		for (uint32_t i = 0; i < N; i++)
			if ((m_s[i] < l) || (m_s[i] > h))
				return false;

		return true;
	}

	inline uint32_t get_major_axis(void) const
	{
		T m = fabs(m_s[0]);
		uint32_t r = 0;
		for (uint32_t i = 1; i < N; i++)
		{
			const T c = fabs(m_s[i]);
			if (c > m)
			{
				m = c;
				r = i;
			}
		}
		return r;
	}

	inline uint32_t get_minor_axis(void) const
	{
		T m = fabs(m_s[0]);
		uint32_t r = 0;
		for (uint32_t i = 1; i < N; i++)
		{
			const T c = fabs(m_s[i]);
			if (c < m)
			{
				m = c;
				r = i;
			}
		}
		return r;
	}

	inline void get_projection_axes(uint32_t& u, uint32_t& v) const
	{
		const int axis = get_major_axis();
		if (m_s[axis] < 0.0f)
		{
			v = next_wrap<uint32_t>(axis, N);
			u = next_wrap<uint32_t>(v, N);
		}
		else
		{
			u = next_wrap<uint32_t>(axis, N);
			v = next_wrap<uint32_t>(u, N);
		}
	}

	inline T get_absolute_minimum(void) const
	{
		T result = fabs(m_s[0]);
		for (uint32_t i = 1; i < N; i++)
			result = std::min(result, fabs(m_s[i]));
		return result;
	}

	inline T get_absolute_maximum(void) const
	{
		T result = fabs(m_s[0]);
		for (uint32_t i = 1; i < N; i++)
			result = std::max(result, fabs(m_s[i]));
		return result;
	}

	inline T get_minimum(void) const
	{
		T result = m_s[0];
		for (uint32_t i = 1; i < N; i++)
			result = std::min(result, m_s[i]);
		return result;
	}

	inline T get_maximum(void) const
	{
		T result = m_s[0];
		for (uint32_t i = 1; i < N; i++)
			result = std::max(result, m_s[i]);
		return result;
	}

	inline vec& remove_unit_direction(const vec& dir)
	{
		*this -= (dot(dir) * dir);
		return *this;
	}

	inline vec get_remove_unit_direction(const vec& dir) const
	{
		return *this - (dot(dir) * dir);
	}

	inline bool all_less(const vec& b) const
	{
		for (uint32_t i = 0; i < N; i++)
			if (m_s[i] >= b.m_s[i])
				return false;
		return true;
	}

	inline bool all_less_equal(const vec& b) const
	{
		for (uint32_t i = 0; i < N; i++)
			if (m_s[i] > b.m_s[i])
				return false;
		return true;
	}

	inline bool all_greater(const vec& b) const
	{
		for (uint32_t i = 0; i < N; i++)
			if (m_s[i] <= b.m_s[i])
				return false;
		return true;
	}

	inline bool all_greater_equal(const vec& b) const
	{
		for (uint32_t i = 0; i < N; i++)
			if (m_s[i] < b.m_s[i])
				return false;
		return true;
	}

	inline vec negate_xyz() const
	{
		vec ret;

		ret[0] = -m_s[0];
		if (N >= 2)
			ret[1] = -m_s[1];
		if (N >= 3)
			ret[2] = -m_s[2];

		for (uint32_t i = 3; i < N; i++)
			ret[i] = m_s[i];

		return ret;
	}

	inline vec& invert()
	{
		for (uint32_t i = 0; i < N; i++)
			if (m_s[i] != 0.0f)
				m_s[i] = 1.0f / m_s[i];
		return *this;
	}

	inline scalar_type perp_dot(const vec& b) const
	{
		ASSUME(N == 2);
		return m_s[0] * b.m_s[1] - m_s[1] * b.m_s[0];
	}

	inline vec perp() const
	{
		ASSUME(N == 2);
		return vec(-m_s[1], m_s[0]);
	}

	inline vec get_floor() const
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result[i] = floor(m_s[i]);
		return result;
	}

	inline vec get_ceil() const
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result[i] = ceil(m_s[i]);
		return result;
	}

	// static helper methods

	static inline vec mul_components(const vec& lhs, const vec& rhs)
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result[i] = lhs.m_s[i] * rhs.m_s[i];
		return result;
	}

	static inline vec mul_add_components(const vec& a, const vec& b, const vec& c)
	{
		vec result;
		for (uint32_t i = 0; i < N; i++)
			result[i] = a.m_s[i] * b.m_s[i] + c.m_s[i];
		return result;
	}

	static inline vec make_axis(uint32_t i)
	{
		vec result;
		result.clear();
		result[i] = 1;
		return result;
	}

	static inline vec equals_mask(const vec& a, const vec& b)
	{
		vec ret;
		for (uint32_t i = 0; i < N; i++)
			ret[i] = (a[i] == b[i]);
		return ret;
	}

	static inline vec not_equals_mask(const vec& a, const vec& b)
	{
		vec ret;
		for (uint32_t i = 0; i < N; i++)
			ret[i] = (a[i] != b[i]);
		return ret;
	}

	static inline vec less_mask(const vec& a, const vec& b)
	{
		vec ret;
		for (uint32_t i = 0; i < N; i++)
			ret[i] = (a[i] < b[i]);
		return ret;
	}

	static inline vec less_equals_mask(const vec& a, const vec& b)
	{
		vec ret;
		for (uint32_t i = 0; i < N; i++)
			ret[i] = (a[i] <= b[i]);
		return ret;
	}

	static inline vec greater_equals_mask(const vec& a, const vec& b)
	{
		vec ret;
		for (uint32_t i = 0; i < N; i++)
			ret[i] = (a[i] >= b[i]);
		return ret;
	}

	static inline vec greater_mask(const vec& a, const vec& b)
	{
		vec ret;
		for (uint32_t i = 0; i < N; i++)
			ret[i] = (a[i] > b[i]);
		return ret;
	}

	static inline vec component_max(const vec& a, const vec& b)
	{
		vec ret;
		for (uint32_t i = 0; i < N; i++)
			ret.m_s[i] = std::max(a.m_s[i], b.m_s[i]);
		return ret;
	}

	static inline vec component_min(const vec& a, const vec& b)
	{
		vec ret;
		for (uint32_t i = 0; i < N; i++)
			ret.m_s[i] = std::min(a.m_s[i], b.m_s[i]);
		return ret;
	}

	static inline vec lerp(const vec& a, const vec& b, float t)
	{
		vec ret;
		for (uint32_t i = 0; i < N; i++)
			ret.m_s[i] = a.m_s[i] + (b.m_s[i] - a.m_s[i]) * t;
		return ret;
	}

	static inline bool equal_tol(const vec& a, const vec& b, float t)
	{
		for (uint32_t i = 0; i < N; i++)
			if (!equal_tol(a.m_s[i], b.m_s[i], t))
				return false;
		return true;
	}

	inline bool equal_tol(const vec& b, float t) const
	{
		return equal_tol(*this, b, t);
	}

protected:
	T m_s[N];
};

typedef vec<1, double> vec1D;
typedef vec<2, double> vec2D;
typedef vec<3, double> vec3D;
typedef vec<4, double> vec4D;

typedef vec<1, float> vec1F;

typedef vec<2, float> vec2F;
typedef std::vector<vec2F> vec2F_array;

typedef vec<3, float> vec3F;
typedef std::vector<vec3F> vec3F_array;

typedef vec<4, float> vec4F;
typedef std::vector<vec4F> vec4F_array;

typedef vec<2, uint32_t> vec2U;
typedef vec<3, uint32_t> vec3U;
typedef vec<2, int> vec2I;
typedef vec<3, int> vec3I;
typedef vec<4, int> vec4I;

typedef vec<2, int16_t> vec2I16;
typedef vec<3, int16_t> vec3I16;

inline vec2F rotate_point(const vec2F& p, float rad)
{
	float c = cos(rad);
	float s = sin(rad);

	float x = p[0];
	float y = p[1];

	return vec2F(x * c - y * s, x * s + y * c);
}

class rect
{
public:
	inline rect()
	{
	}

	inline rect(eClear)
	{
		clear();
	}

	inline rect(eInitExpand)
	{
		init_expand();
	}

	// up to, but not including right/bottom
	inline rect(int left, int top, int right, int bottom)
	{
		set(left, top, right, bottom);
	}

	inline rect(const vec2I& lo, const vec2I& hi)
	{
		m_corner[0] = lo;
		m_corner[1] = hi;
	}

	inline rect(const vec2I& point)
	{
		m_corner[0] = point;
		m_corner[1].set(point[0] + 1, point[1] + 1);
	}

	inline bool operator==(const rect& r) const
	{
		return (m_corner[0] == r.m_corner[0]) && (m_corner[1] == r.m_corner[1]);
	}

	inline bool operator<(const rect& r) const
	{
		for (uint32_t i = 0; i < 2; i++)
		{
			if (m_corner[i] < r.m_corner[i])
				return true;
			else if (!(m_corner[i] == r.m_corner[i]))
				return false;
		}

		return false;
	}

	inline void clear()
	{
		m_corner[0].clear();
		m_corner[1].clear();
	}

	inline void set(int left, int top, int right, int bottom)
	{
		m_corner[0].set(left, top);
		m_corner[1].set(right, bottom);
	}

	inline void set(const vec2I& lo, const vec2I& hi)
	{
		m_corner[0] = lo;
		m_corner[1] = hi;
	}

	inline void set(const vec2I& point)
	{
		m_corner[0] = point;
		m_corner[1].set(point[0] + 1, point[1] + 1);
	}

	inline uint32_t get_width() const
	{
		return m_corner[1][0] - m_corner[0][0];
	}
	inline uint32_t get_height() const
	{
		return m_corner[1][1] - m_corner[0][1];
	}

	inline int get_left() const
	{
		return m_corner[0][0];
	}
	inline int get_top() const
	{
		return m_corner[0][1];
	}
	inline int get_right() const
	{
		return m_corner[1][0];
	}
	inline int get_bottom() const
	{
		return m_corner[1][1];
	}

	inline bool is_empty() const
	{
		return (m_corner[1][0] <= m_corner[0][0]) || (m_corner[1][1] <= m_corner[0][1]);
	}

	inline uint32_t get_dimension(uint32_t axis) const
	{
		return m_corner[1][axis] - m_corner[0][axis];
	}
	inline uint32_t get_area() const
	{
		return get_dimension(0) * get_dimension(1);
	}

	inline const vec2I& operator[](uint32_t i) const
	{
		assert(i < 2);
		return m_corner[i];
	}
	inline vec2I& operator[](uint32_t i)
	{
		assert(i < 2);
		return m_corner[i];
	}

	inline rect& translate(int x_ofs, int y_ofs)
	{
		m_corner[0][0] += x_ofs;
		m_corner[0][1] += y_ofs;
		m_corner[1][0] += x_ofs;
		m_corner[1][1] += y_ofs;
		return *this;
	}

	inline rect& init_expand()
	{
		m_corner[0].set(INT_MAX);
		m_corner[1].set(INT_MIN);
		return *this;
	}

	inline rect& expand(int x, int y)
	{
		m_corner[0][0] = std::min(m_corner[0][0], x);
		m_corner[0][1] = std::min(m_corner[0][1], y);
		m_corner[1][0] = std::max(m_corner[1][0], x + 1);
		m_corner[1][1] = std::max(m_corner[1][1], y + 1);
		return *this;
	}

	inline rect& expand(const rect& r)
	{
		m_corner[0][0] = std::min(m_corner[0][0], r[0][0]);
		m_corner[0][1] = std::min(m_corner[0][1], r[0][1]);
		m_corner[1][0] = std::max(m_corner[1][0], r[1][0]);
		m_corner[1][1] = std::max(m_corner[1][1], r[1][1]);
		return *this;
	}

	inline bool touches(const rect& r) const
	{
		for (uint32_t i = 0; i < 2; i++)
		{
			if (r[1][i] <= m_corner[0][i])
				return false;
			else if (r[0][i] >= m_corner[1][i])
				return false;
		}

		return true;
	}

	inline bool fully_within(const rect& r) const
	{
		for (uint32_t i = 0; i < 2; i++)
		{
			if (m_corner[0][i] < r[0][i])
				return false;
			else if (m_corner[1][i] > r[1][i])
				return false;
		}

		return true;
	}

	inline bool intersect(const rect& r)
	{
		if (!touches(r))
		{
			clear();
			return false;
		}

		for (uint32_t i = 0; i < 2; i++)
		{
			m_corner[0][i] = std::max<int>(m_corner[0][i], r[0][i]);
			m_corner[1][i] = std::min<int>(m_corner[1][i], r[1][i]);
		}

		return true;
	}

	inline bool contains(int x, int y) const
	{
		return (x >= m_corner[0][0]) && (x < m_corner[1][0]) &&
			(y >= m_corner[0][1]) && (y < m_corner[1][1]);
	}

	inline bool contains(const vec2I& p) const
	{
		return contains(p[0], p[1]);
	}

private:
	vec2I m_corner[2];
};

inline rect make_rect(uint32_t width, uint32_t height)
{
	return rect(0, 0, width, height);
}

struct color_quad_u8
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4201)
#endif
	union
	{
		uint8_t m_c[4];
		struct
		{
			uint8_t r;
			uint8_t g;
			uint8_t b;
			uint8_t a;
		};
	};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

	inline color_quad_u8(eClear) : color_quad_u8(0, 0, 0, 0) { }

	inline color_quad_u8(uint8_t cr, uint8_t cg, uint8_t cb, uint8_t ca)
	{
		set(cr, cg, cb, ca);
	}

	inline color_quad_u8(uint8_t cy = 0, uint8_t ca = 255)
	{
		set(cy, ca);
	}

	inline void clear()
	{
		set(0, 0, 0, 0);
	}

	inline color_quad_u8& set(uint8_t cy, uint8_t ca = 255)
	{
		m_c[0] = cy;
		m_c[1] = cy;
		m_c[2] = cy;
		m_c[3] = ca;
		return *this;
	}

	inline color_quad_u8& set(uint8_t cr, uint8_t cg, uint8_t cb, uint8_t ca)
	{
		m_c[0] = cr;
		m_c[1] = cg;
		m_c[2] = cb;
		m_c[3] = ca;
		return *this;
	}

	inline color_quad_u8& set_clamped(int cr, int cg, int cb, int ca)
	{
		m_c[0] = (uint8_t)clamp(cr, 0, 255);
		m_c[1] = (uint8_t)clamp(cg, 0, 255);
		m_c[2] = (uint8_t)clamp(cb, 0, 255);
		m_c[3] = (uint8_t)clamp(ca, 0, 255);
		return *this;
	}

	color_quad_u8& set_alpha(int ca) { a = (uint8_t)clamp(ca, 0, 255); return *this; }

	inline uint8_t& operator[] (uint32_t i) { assert(i < 4);  return m_c[i]; }
	inline uint8_t operator[] (uint32_t i) const { assert(i < 4); return m_c[i]; }
		
	inline int get_luma() const { return (13938U * m_c[0] + 46869U * m_c[1] + 4729U * m_c[2] + 32768U) >> 16U; } // REC709 weightings

	inline bool operator== (const color_quad_u8& other) const
	{
		return (m_c[0] == other.m_c[0]) && (m_c[1] == other.m_c[1]) && (m_c[2] == other.m_c[2]) && (m_c[3] == other.m_c[3]);
	}

	inline bool operator!= (const color_quad_u8& other) const
	{
		return !(*this == other);
	}

	inline uint32_t squared_distance(const color_quad_u8& c, bool alpha = true) const
	{
		return square(r - c.r) + square(g - c.g) + square(b - c.b) + (alpha ? square(a - c.a) : 0);
	}

	inline bool rgb_equals(const color_quad_u8& rhs) const
	{
		return (r == rhs.r) && (g == rhs.g) && (b == rhs.b);
	}
};
typedef std::vector<color_quad_u8> color_quad_u8_vec;

inline uint32_t color_distance(bool perceptual, const color_quad_u8& e1, const color_quad_u8& e2, bool alpha)
{
	if (perceptual)
	{
		const float l1 = e1.r * .2126f + e1.g * .715f + e1.b * .0722f;
		const float cr1 = e1.r - l1;
		const float cb1 = e1.b - l1;

		const float l2 = e2.r * .2126f + e2.g * .715f + e2.b * .0722f;
		const float cr2 = e2.r - l2;
		const float cb2 = e2.b - l2;

		const float dl = l1 - l2;
		const float dcr = cr1 - cr2;
		const float dcb = cb1 - cb2;

		uint32_t d = static_cast<uint32_t>(
			32.0f * 4.0f * dl * dl + 
			32.0f * 2.0f * (.5f / (1.0f - .2126f)) * (.5f / (1.0f - .2126f)) * dcr * dcr + 
			32.0f * .25f * (.5f / (1.0f - .0722f)) * (.5f / (1.0f - .0722f)) * dcb * dcb);
		
		if (alpha)
		{
			int da = (int)e1.a - (int)e2.a;

			d += static_cast<uint32_t>(128.0f * da * da);
		}

		return d;
	}
	else
		return e1.squared_distance(e2, alpha);
}

extern color_quad_u8 g_white_color_u8, g_black_color_u8, g_red_color_u8, g_green_color_u8, g_blue_color_u8, g_yellow_color_u8, g_purple_color_u8, g_magenta_color_u8, g_cyan_color_u8;

class image_u8
{
public:
	image_u8() :
		m_width(0), m_height(0),
		m_clip_rect(cClear)
	{
	}

	image_u8(uint32_t width, uint32_t height) :
		m_width(width), m_height(height),
		m_clip_rect(0, 0, width, height)
	{
		m_pixels.resize(width * height);
	}

	inline const color_quad_u8_vec& get_pixels() const { return m_pixels; }
	inline color_quad_u8_vec& get_pixels() { return m_pixels; }

	inline uint32_t width() const { return m_width; }
	inline uint32_t height() const { return m_height; }
	inline uint32_t total_pixels() const { return m_width * m_height; }
	
	inline const rect& get_clip_rect() const { return m_clip_rect; }

	inline void set_clip_rect(const rect& r) 
	{ 
		assert((r.get_left() >= 0) && (r.get_top() >= 0) && (r.get_right() <= (int)m_width) && (r.get_bottom() <= (int)m_height));

		m_clip_rect = r; 
	}

	inline void clear_clip_rect() { m_clip_rect.set(0, 0, m_width, m_height); }

	inline bool is_clipped(int x, int y) const { return !m_clip_rect.contains(x, y); }
	
	inline rect get_bounds() const { return rect(0, 0, m_width, m_height); }

	inline color_quad_u8& operator()(uint32_t x, uint32_t y) { assert((x < m_width) && (y < m_height));  return m_pixels[x + m_width * y]; }
	inline const color_quad_u8& operator()(uint32_t x, uint32_t y) const { assert((x < m_width) && (y < m_height));  return m_pixels[x + m_width * y]; }

	image_u8& clear()
	{
		m_width = m_height = 0;
		m_clip_rect.clear();
		m_pixels.clear();
		return *this;
	}

	image_u8& init(uint32_t width, uint32_t height)
	{
		clear();

		m_width = width;
		m_height = height;
		m_clip_rect.set(0, 0, width, height);
		m_pixels.resize(width * height);
		return *this;
	}

	image_u8& set_all(const color_quad_u8& p)
	{
		for (uint32_t i = 0; i < m_pixels.size(); i++)
			m_pixels[i] = p;
		return *this;
	}

	inline const color_quad_u8& get_clamped(int x, int y) const { return (*this)(clamp<int>(x, 0, m_width - 1), clamp<int>(y, 0, m_height - 1)); }
	inline color_quad_u8& get_clamped(int x, int y) { return (*this)(clamp<int>(x, 0, m_width - 1), clamp<int>(y, 0, m_height - 1)); }

	inline image_u8& set_pixel_clipped(int x, int y, const color_quad_u8& c)
	{
		if (!is_clipped(x, y))
			(*this)(x, y) = c;
		return *this;
	}

	inline image_u8& fill_box(int x, int y, int w, int h, const color_quad_u8& c)
	{
		for (int y_ofs = 0; y_ofs < h; y_ofs++)
			for (int x_ofs = 0; x_ofs < w; x_ofs++)
				set_pixel_clipped(x + x_ofs, y + y_ofs, c);
		return *this;
	}

	void invert_box(int inX, int inY, int inW, int inH)
	{
		for (int y = 0; y < inH; y++)
		{
			const uint32_t yy = inY + y;

			for (int x = 0; x < inW; x++)
			{
				const uint32_t xx = inX + x;

				if (is_clipped(xx, yy))
					continue;

				color_quad_u8 c((*this)(xx, yy));

				c.r = 255 - c.r;
				c.g = 255 - c.g;
				c.b = 255 - c.b;

				set_pixel_clipped(xx, yy, c);
			}
		}
	}

	image_u8& crop_dup_borders(uint32_t w, uint32_t h)
	{
		const uint32_t orig_w = m_width, orig_h = m_height;

		crop(w, h);

		if (orig_w && orig_h)
		{
			if (m_width > orig_w)
			{
				for (uint32_t x = orig_w; x < m_width; x++)
					for (uint32_t y = 0; y < m_height; y++)
						set_pixel_clipped(x, y, get_clamped(std::min(x, orig_w - 1U), std::min(y, orig_h - 1U)));
			}

			if (m_height > orig_h)
			{
				for (uint32_t y = orig_h; y < m_height; y++)
					for (uint32_t x = 0; x < m_width; x++)
						set_pixel_clipped(x, y, get_clamped(std::min(x, orig_w - 1U), std::min(y, orig_h - 1U)));
			}
		}
		return *this;
	}

	image_u8& crop(uint32_t new_width, uint32_t new_height)
	{
		if ((m_width == new_width) && (m_height == new_height))
			return *this;

		image_u8 new_image(new_width, new_height);

		const uint32_t w = std::min(m_width, new_width);
		const uint32_t h = std::min(m_height, new_height);

		for (uint32_t y = 0; y < h; y++)
			for (uint32_t x = 0; x < w; x++)
				new_image(x, y) = (*this)(x, y);

		return swap(new_image);
	}

	image_u8& swap(image_u8& other)
	{
		std::swap(m_width, other.m_width);
		std::swap(m_height, other.m_height);
		std::swap(m_pixels, other.m_pixels);
		std::swap(m_clip_rect, other.m_clip_rect);
		return *this;
	}

	// No clipping
	inline void get_block(uint32_t bx, uint32_t by, uint32_t width, uint32_t height, color_quad_u8* pPixels) const
	{
		assert((bx * width + width) <= m_width);
		assert((by * height + height) <= m_height);

		for (uint32_t y = 0; y < height; y++)
			memcpy(pPixels + y * width, &(*this)(bx * width, by * height + y), width * sizeof(color_quad_u8));
	}

	inline void get_block_clamped(uint32_t bx, uint32_t by, uint32_t width, uint32_t height, color_quad_u8* pPixels) const
	{
		for (uint32_t y = 0; y < height; y++)
			for (uint32_t x = 0; x < width; x++)
				pPixels[x + y * width] = get_clamped(bx * width + x, by * height + y);
	}
		
	// No clipping
	inline void set_block(uint32_t bx, uint32_t by, uint32_t width, uint32_t height, const color_quad_u8* pPixels)
	{
		assert((bx * width + width) <= m_width);
		assert((by * height + height) <= m_height);

		for (uint32_t y = 0; y < height; y++)
			memcpy(&(*this)(bx * width, by * height + y), pPixels + y * width, width * sizeof(color_quad_u8));
	}

	image_u8& swizzle(uint32_t r, uint32_t g, uint32_t b, uint32_t a)
	{
		assert((r | g | b | a) <= 3);
		for (uint32_t y = 0; y < m_height; y++)
		{
			for (uint32_t x = 0; x < m_width; x++)
			{
				color_quad_u8 tmp((*this)(x, y));
				(*this)(x, y).set(tmp[r], tmp[g], tmp[b], tmp[a]);
			}
		}

		return *this;
	}

	struct pixel_coord
	{
		uint16_t m_x, m_y;
		pixel_coord() { }
		pixel_coord(uint32_t x, uint32_t y) : m_x((uint16_t)x), m_y((uint16_t)y) { }
	};
		
	uint32_t flood_fill(int x, int y, const color_quad_u8& c, const color_quad_u8& b, std::vector<pixel_coord>* pSet_pixels = nullptr);

	void draw_line(int xs, int ys, int xe, int ye, const color_quad_u8& color);
		
	inline void set_pixel_clipped_alphablend(int x, int y, const color_quad_u8& c)
	{
		if (is_clipped(x, y))
			return;

		color_quad_u8 ct(m_pixels[x + y * m_width]);

		ct.r = static_cast<uint8_t>(ct.r + ((c.r - ct.r) * c.a) / 255);
		ct.g = static_cast<uint8_t>(ct.g + ((c.g - ct.g) * c.a) / 255);
		ct.b = static_cast<uint8_t>(ct.b + ((c.b - ct.b) * c.a) / 255);
		
		m_pixels[x + y * m_width] = ct;
	}

private:
	color_quad_u8_vec m_pixels;
	uint32_t m_width, m_height;
	rect m_clip_rect;

	struct fill_segment
	{
		int16_t m_y, m_xl, m_xr, m_dy;

		fill_segment(int y, int xl, int xr, int dy) :
			m_y((int16_t)y), m_xl((int16_t)xl), m_xr((int16_t)xr), m_dy((int16_t)dy)
		{
		}
	};

	inline bool flood_fill_is_inside(int x, int y, const color_quad_u8& b) const
	{
		if (is_clipped(x, y))
			return false;

		return (*this)(x, y) == b;
	}

	void rasterize_line(int xs, int ys, int xe, int ye, int pred, int inc_dec, int e, int e_inc, int e_no_inc, const color_quad_u8& color);

	void draw_aaline_pixel(int x, int y, int a, color_quad_u8 color)
	{
		color.a = static_cast<uint8_t>(255 - a);
		set_pixel_clipped_alphablend(x, y, color);
	}
};

bool load_png(const char* pFilename, image_u8& img);

bool save_png(const char* pFilename, const image_u8& img, bool save_alpha);

class image_metrics
{
public:
	double m_max, m_mean, m_mean_squared, m_root_mean_squared, m_peak_snr;

	image_metrics()
	{
		clear();
	}

	void clear()
	{
		memset(this, 0, sizeof(*this));
	}

	void compute(const image_u8& a, const image_u8& b, uint32_t first_channel, uint32_t num_channels)
	{
		const bool average_component_error = true;

		const uint32_t width = std::min(a.width(), b.width());
		const uint32_t height = std::min(a.height(), b.height());

		assert((first_channel < 4U) && (first_channel + num_channels <= 4U));

		// Histogram approach originally due to Charles Bloom.
		double hist[256];
		memset(hist, 0, sizeof(hist));

		for (uint32_t y = 0; y < height; y++)
		{
			for (uint32_t x = 0; x < width; x++)
			{
				const color_quad_u8& ca = a(x, y);
				const color_quad_u8& cb = b(x, y);

				if (!num_channels)
					hist[iabs(ca.get_luma() - cb.get_luma())]++;
				else
				{
					for (uint32_t c = 0; c < num_channels; c++)
						hist[iabs(ca[first_channel + c] - cb[first_channel + c])]++;
				}
			}
		}

		m_max = 0;
		double sum = 0.0f, sum2 = 0.0f;
		for (uint32_t i = 0; i < 256; i++)
		{
			if (!hist[i])
				continue;

			m_max = std::max<double>(m_max, i);

			double x = i * hist[i];

			sum += x;
			sum2 += i * x;
		}

		// See http://richg42.blogspot.com/2016/09/how-to-compute-psnr-from-old-berkeley.html
		double total_values = width * height;

		if (average_component_error)
			total_values *= clamp<uint32_t>(num_channels, 1, 4);

		m_mean = clamp<double>(sum / total_values, 0.0f, 255.0f);
		m_mean_squared = clamp<double>(sum2 / total_values, 0.0f, 255.0f * 255.0f);

		m_root_mean_squared = sqrt(m_mean_squared);

		if (!m_root_mean_squared)
			m_peak_snr = 100.0f;
		else
			m_peak_snr = clamp<double>(log10(255.0f / m_root_mean_squared) * 20.0f, 0.0f, 100.0f);
	}
};

class imagef
{
public:
	imagef() :
		m_width(0), m_height(0), m_pitch(0)
	{
	}

	imagef(uint32_t w, uint32_t h, uint32_t p = UINT32_MAX) :
		m_width(0), m_height(0), m_pitch(0)
	{
		resize(w, h, p);
	}

	imagef(const imagef& other) :
		m_width(0), m_height(0), m_pitch(0)
	{
		*this = other;
	}

	imagef& swap(imagef& other)
	{
		std::swap(m_width, other.m_width);
		std::swap(m_height, other.m_height);
		std::swap(m_pitch, other.m_pitch);
		m_pixels.swap(other.m_pixels);
		return *this;
	}

	imagef& operator= (const imagef& rhs)
	{
		if (this != &rhs)
		{
			m_width = rhs.m_width;
			m_height = rhs.m_height;
			m_pitch = rhs.m_pitch;
			m_pixels = rhs.m_pixels;
		}
		return *this;
	}

	imagef& clear()
	{
		m_width = 0;
		m_height = 0;
		m_pitch = 0;
		m_pixels.resize(0);
		return *this;
	}

	imagef& set(const image_u8& src, const vec4F& scale = vec4F(1), const vec4F& bias = vec4F(0))
	{
		const uint32_t width = src.width();
		const uint32_t height = src.height();

		resize(width, height);

		for (int y = 0; y < (int)height; y++)
		{
			for (uint32_t x = 0; x < width; x++)
			{
				const color_quad_u8& src_pixel = src(x, y);
				(*this)(x, y).set((float)src_pixel.r * scale[0] + bias[0], (float)src_pixel.g * scale[1] + bias[1], (float)src_pixel.b * scale[2] + bias[2], (float)src_pixel.a * scale[3] + bias[3]);
			}
		}

		return *this;
	}

	imagef& resize(const imagef& other, uint32_t p = UINT32_MAX, const vec4F& background = vec4F(0, 0, 0, 1))
	{
		return resize(other.get_width(), other.get_height(), p, background);
	}

	imagef& resize(uint32_t w, uint32_t h, uint32_t p = UINT32_MAX, const vec4F& background = vec4F(0, 0, 0, 1))
	{
		return crop(w, h, p, background);
	}

	imagef& set_all(const vec4F& c)
	{
		for (uint32_t i = 0; i < m_pixels.size(); i++)
			m_pixels[i] = c;
		return *this;
	}

	imagef& fill_box(uint32_t x, uint32_t y, uint32_t w, uint32_t h, const vec4F& c)
	{
		for (uint32_t iy = 0; iy < h; iy++)
			for (uint32_t ix = 0; ix < w; ix++)
				set_pixel_clipped(x + ix, y + iy, c);
		return *this;
	}

	imagef& crop(uint32_t w, uint32_t h, uint32_t p = UINT32_MAX, const vec4F& background = vec4F(0, 0, 0, 1))
	{
		if (p == UINT32_MAX)
			p = w;

		if ((w == m_width) && (m_height == h) && (m_pitch == p))
			return *this;

		if ((!w) || (!h) || (!p))
		{
			clear();
			return *this;
		}

		vec4F_array cur_state;
		cur_state.swap(m_pixels);

		m_pixels.resize(p * h);

		for (uint32_t y = 0; y < h; y++)
		{
			for (uint32_t x = 0; x < w; x++)
			{
				if ((x < m_width) && (y < m_height))
					m_pixels[x + y * p] = cur_state[x + y * m_pitch];
				else
					m_pixels[x + y * p] = background;
			}
		}

		m_width = w;
		m_height = h;
		m_pitch = p;

		return *this;
	}

	inline const vec4F& operator() (uint32_t x, uint32_t y) const { assert(x < m_width&& y < m_height); return m_pixels[x + y * m_pitch]; }
	inline vec4F& operator() (uint32_t x, uint32_t y) { assert(x < m_width&& y < m_height); return m_pixels[x + y * m_pitch]; }

	inline const vec4F& get_clamped(int x, int y) const { return (*this)(clamp<int>(x, 0, m_width - 1), clamp<int>(y, 0, m_height - 1)); }
	inline vec4F& get_clamped(int x, int y) { return (*this)(clamp<int>(x, 0, m_width - 1), clamp<int>(y, 0, m_height - 1)); }

	inline const vec4F& get_clamped_or_wrapped(int x, int y, bool wrap_u, bool wrap_v) const
	{
		x = wrap_u ? posmod(x, m_width) : clamp<int>(x, 0, m_width - 1);
		y = wrap_v ? posmod(y, m_height) : clamp<int>(y, 0, m_height - 1);
		return m_pixels[x + y * m_pitch];
	}

	inline vec4F& get_clamped_or_wrapped(int x, int y, bool wrap_u, bool wrap_v)
	{
		x = wrap_u ? posmod(x, m_width) : clamp<int>(x, 0, m_width - 1);
		y = wrap_v ? posmod(y, m_height) : clamp<int>(y, 0, m_height - 1);
		return m_pixels[x + y * m_pitch];
	}

	inline imagef& set_pixel_clipped(int x, int y, const vec4F& c)
	{
		if ((static_cast<uint32_t>(x) < m_width) && (static_cast<uint32_t>(y) < m_height))
			(*this)(x, y) = c;
		return *this;
	}

	// Very straightforward blit with full clipping. Not fast, but it works.
	imagef& blit(const imagef& src, int src_x, int src_y, int src_w, int src_h, int dst_x, int dst_y)
	{
		for (int y = 0; y < src_h; y++)
		{
			const int sy = src_y + y;
			if (sy < 0)
				continue;
			else if (sy >= (int)src.get_height())
				break;

			for (int x = 0; x < src_w; x++)
			{
				const int sx = src_x + x;
				if (sx < 0)
					continue;
				else if (sx >= (int)src.get_height())
					break;

				set_pixel_clipped(dst_x + x, dst_y + y, src(sx, sy));
			}
		}

		return *this;
	}

	const imagef& extract_block_clamped(vec4F* pDst, uint32_t src_x, uint32_t src_y, uint32_t w, uint32_t h) const
	{
		for (uint32_t y = 0; y < h; y++)
			for (uint32_t x = 0; x < w; x++)
				*pDst++ = get_clamped(src_x + x, src_y + y);
		return *this;
	}

	imagef& set_block_clipped(const vec4F* pSrc, uint32_t dst_x, uint32_t dst_y, uint32_t w, uint32_t h)
	{
		for (uint32_t y = 0; y < h; y++)
			for (uint32_t x = 0; x < w; x++)
				set_pixel_clipped(dst_x + x, dst_y + y, *pSrc++);
		return *this;
	}

	inline uint32_t get_width() const { return m_width; }
	inline uint32_t get_height() const { return m_height; }
	inline uint32_t get_pitch() const { return m_pitch; }
	inline uint32_t get_total_pixels() const { return m_width * m_height; }

	inline uint32_t get_block_width(uint32_t w) const { return (m_width + (w - 1)) / w; }
	inline uint32_t get_block_height(uint32_t h) const { return (m_height + (h - 1)) / h; }
	inline uint32_t get_total_blocks(uint32_t w, uint32_t h) const { return get_block_width(w) * get_block_height(h); }

	inline const vec4F_array& get_pixels() const { return m_pixels; }
	inline vec4F_array& get_pixels() { return m_pixels; }

	inline const vec4F* get_ptr() const { return &m_pixels[0]; }
	inline vec4F* get_ptr() { return &m_pixels[0]; }

private:
	uint32_t m_width, m_height, m_pitch;  // all in pixels
	vec4F_array m_pixels;
};

enum
{
	cComputeGaussianFlagNormalize = 1,
	cComputeGaussianFlagPrint = 2,
	cComputeGaussianFlagNormalizeCenterToOne = 4
};

// size_x/y should be odd
void compute_gaussian_kernel(float* pDst, int size_x, int size_y, float sigma_sqr, uint32_t flags);

void gaussian_filter(imagef& dst, const imagef& orig_img, uint32_t odd_filter_width, float sigma_sqr, bool wrapping = false, uint32_t width_divisor = 1, uint32_t height_divisor = 1);

vec4F compute_ssim(const imagef& a, const imagef& b);

vec4F compute_ssim(const image_u8& a, const image_u8& b, bool luma);

struct block8
{
	uint64_t m_vals[1];
};

typedef std::vector<block8> block8_vec;

struct block16
{
	uint64_t m_vals[2];
};

typedef std::vector<block16> block16_vec;

bool save_dds(const char* pFilename, uint32_t width, uint32_t height, const void* pBlocks, uint32_t pixel_format_bpp, DXGI_FORMAT dxgi_format, bool srgb, bool force_dx10_header);

void strip_extension(std::string& s);
void strip_path(std::string& s);

uint32_t hash_hsieh(const uint8_t* pBuf, size_t len);

// https://www.johndcook.com/blog/standard_deviation/
// This class is for small numbers of integers, so precision shouldn't be an issue.
class tracked_stat
{
public:
	tracked_stat() { clear(); }

	void clear() { m_num = 0; m_total = 0; m_total2 = 0; }

	void update(uint32_t val) { m_num++; m_total += val; m_total2 += val * val; }

	tracked_stat& operator += (uint32_t val) { update(val); return *this; }

	uint32_t get_number_of_values() const { return m_num; }
	uint64_t get_total() const { return m_total; }
	uint64_t get_total2() const { return m_total2; }

	float get_mean() const { return m_num ? (float)m_total / m_num : 0.0f; };
		
	float get_variance() const { return m_num ? ((float)(m_num * m_total2 - m_total * m_total)) / (m_num * m_num) : 0.0f; }
	float get_std_dev() const { return m_num ? sqrtf((float)(m_num * m_total2 - m_total * m_total)) / m_num : 0.0f; }

	float get_sample_variance() const { return (m_num > 1) ? ((float)(m_num * m_total2 - m_total * m_total)) / (m_num * (m_num - 1)) : 0.0f; }
	float get_sample_std_dev() const { return (m_num > 1) ? sqrtf(get_sample_variance()) : 0.0f; }

private:
	uint32_t m_num;
	uint64_t m_total;
	uint64_t m_total2;
};

inline float compute_covariance(const float* pA, const float* pB, const tracked_stat& a, const tracked_stat& b, bool sample)
{
	const uint32_t n = a.get_number_of_values();
	assert(n == b.get_number_of_values());

	if (!n)
	{
		assert(0);
		return 0.0f;
	}
	if ((sample) && (n == 1))
	{
		assert(0);
		return 0;
	}

	const float mean_a = a.get_mean();
	const float mean_b = b.get_mean();
	
	float total = 0.0f;
	for (uint32_t i = 0; i < n; i++)
		total += (pA[i] - mean_a) * (pB[i] - mean_b);

	return total / (sample ? (n - 1) : n);
}

inline float compute_correlation_coefficient(const float* pA, const float* pB, const tracked_stat& a, const tracked_stat& b, float c, bool sample)
{
	if (!a.get_number_of_values())
		return 1.0f;

	float covar = compute_covariance(pA, pB, a, b, sample);
	float std_dev_a = sample ? a.get_sample_std_dev() : a.get_std_dev();
	float std_dev_b = sample ? b.get_sample_std_dev() : b.get_std_dev();
	float denom = std_dev_a * std_dev_b + c;

	if (denom < .0000125f)
		return 1.0f;

	float result = (covar + c) / denom;
	
	return clamp(result, -1.0f, 1.0f);
}

float compute_block_max_std_dev(const color_quad_u8* pPixels, uint32_t block_width, uint32_t block_height, uint32_t num_comps);

class rand
{
	std::mt19937 m_mt;

public:
	rand() {	}

	rand(uint32_t s) { seed(s); }
	void seed(uint32_t s) { m_mt.seed(s); }

	// between [l,h]
	int irand(int l, int h) { std::uniform_int_distribution<int> d(l, h); return d(m_mt); }

	uint32_t urand32() { return static_cast<uint32_t>(irand(INT32_MIN, INT32_MAX)); }

	bool bit() { return irand(0, 1) == 1; }

	uint8_t byte() { return static_cast<uint8_t>(urand32()); }

	// between [l,h)
	float frand(float l, float h) { std::uniform_real_distribution<float> d(l, h); return d(m_mt); }

	float gaussian(float mean, float stddev) { std::normal_distribution<float> d(mean, stddev); return d(m_mt); }
};

bool save_astc_file(const char* pFilename, block16_vec& blocks, uint32_t width, uint32_t height, uint32_t block_width, uint32_t block_height);
bool load_astc_file(const char* pFilename, block16_vec& blocks, uint32_t& width, uint32_t& height, uint32_t& block_width, uint32_t& block_height);

class value_stats
{
public:
	value_stats()
	{
		clear();
	}

	void clear()
	{
		m_sum = 0;
		m_sum2 = 0;
		m_num = 0;
		m_min = 1e+39;
		m_max = -1e+39;
		m_vals.clear();
	}

	void add(double val)
	{
		m_sum += val;
		m_sum2 += val * val;

		m_num++;

		m_min = std::min(m_min, val);
		m_max = std::max(m_max, val);

		m_vals.push_back(val);
	}

	void add(int val)
	{
		add(static_cast<double>(val));
	}

	void add(uint32_t val)
	{
		add(static_cast<double>(val));
	}

	void add(int64_t val)
	{
		add(static_cast<double>(val));
	}

	void add(uint64_t val)
	{
		add(static_cast<double>(val));
	}

	void print(const char* pPrefix = "")
	{
		if (!m_vals.size())
			printf("%s: Empty\n", pPrefix);
		else
			printf("%s: Samples: %llu, Total: %f, Avg: %f, Std Dev: %f, Min: %f, Max: %f, Mean: %f\n",
				pPrefix, (unsigned long long)get_num(), get_total(), get_average(), get_std_dev(), get_min(), get_max(), get_mean());
	}

	double get_total() const
	{
		return m_sum;
	}

	double get_average() const
	{
		return m_num ? (m_sum / m_num) : 0.0f;
	}

	double get_min() const
	{
		return m_min;
	}

	double get_max() const
	{
		return m_max;
	}

	uint64_t get_num() const
	{
		return m_num;
	}

	double get_val(uint32_t index) const
	{
		return m_vals[index];
	}

	// Returns population standard deviation
	double get_std_dev() const
	{
		if (!m_num)
			return 0.0f;
		
		// TODO: FP precision
		return sqrt((m_sum2 - ((m_sum * m_sum) / m_num)) / m_num);
	}

	double get_mean() const
	{
		if (!m_num)
			return 0.0f;

		std::vector<double> sorted_vals(m_vals);
		std::sort(sorted_vals.begin(), sorted_vals.end());
		
		return sorted_vals[sorted_vals.size() / 2];
	}

private:
	double m_sum;
	double m_sum2;

	uint64_t m_num;

	double m_min;
	double m_max;

	mutable std::vector<double> m_vals;
};

uint32_t get_deflate_size(const void* pData, size_t data_size);

} // namespace utils

#ifdef _MSC_VER
#pragma warning (pop)
#endif
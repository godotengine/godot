// This code is in the public domain -- castanyo@yahoo.es
#include "xatlas.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#undef min
#undef max

#ifndef xaAssert
#define xaAssert(exp)                                    \
	if (!(exp)) {                                        \
		xaPrint("%s %s %s\n", #exp, __FILE__, __LINE__); \
	}
#endif
#ifndef xaDebugAssert
#define xaDebugAssert(exp) assert(exp)
#endif
#ifndef xaPrint
#define xaPrint(...)                            \
	if (xatlas::internal::s_print) {            \
		xatlas::internal::s_print(__VA_ARGS__); \
	}
#endif

#ifdef _MSC_VER
// Ignore gcc attributes.
#define __attribute__(X)
#endif

#ifdef _MSC_VER
#define restrict
#define NV_FORCEINLINE __forceinline
#else
#define restrict __restrict__
#define NV_FORCEINLINE __attribute__((always_inline)) inline
#endif

#define NV_UINT32_MAX 0xffffffff
#define NV_FLOAT_MAX 3.402823466e+38F

#ifndef PI
#define PI float(3.1415926535897932384626433833)
#endif

#define NV_EPSILON (0.0001f)
#define NV_NORMAL_EPSILON (0.001f)

namespace xatlas {
namespace internal {

static PrintFunc s_print = NULL;

static int align(int x, int a) {
	return (x + a - 1) & ~(a - 1);
}

static bool isAligned(int x, int a) {
	return (x & (a - 1)) == 0;
}

/// Return the maximum of the three arguments.
template <typename T>
static T max3(const T &a, const T &b, const T &c) {
	return std::max(a, std::max(b, c));
}

/// Return the maximum of the three arguments.
template <typename T>
static T min3(const T &a, const T &b, const T &c) {
	return std::min(a, std::min(b, c));
}

/// Clamp between two values.
template <typename T>
static T clamp(const T &x, const T &a, const T &b) {
	return std::min(std::max(x, a), b);
}

static float saturate(float f) {
	return clamp(f, 0.0f, 1.0f);
}

// Robust floating point comparisons:
// http://realtimecollisiondetection.net/blog/?p=89
static bool equal(const float f0, const float f1, const float epsilon = NV_EPSILON) {
	//return fabs(f0-f1) <= epsilon;
	return fabs(f0 - f1) <= epsilon * max3(1.0f, fabsf(f0), fabsf(f1));
}

NV_FORCEINLINE static int ftoi_floor(float val) {
	return (int)val;
}

NV_FORCEINLINE static int ftoi_ceil(float val) {
	return (int)ceilf(val);
}

NV_FORCEINLINE static int ftoi_round(float f) {
	return int(floorf(f + 0.5f));
}

static bool isZero(const float f, const float epsilon = NV_EPSILON) {
	return fabs(f) <= epsilon;
}

static float lerp(float f0, float f1, float t) {
	const float s = 1.0f - t;
	return f0 * s + f1 * t;
}

static float square(float f) {
	return f * f;
}

static int square(int i) {
	return i * i;
}

/** Return the next power of two.
* @see http://graphics.stanford.edu/~seander/bithacks.html
* @warning Behaviour for 0 is undefined.
* @note isPowerOfTwo(x) == true -> nextPowerOfTwo(x) == x
* @note nextPowerOfTwo(x) = 2 << log2(x-1)
*/
static uint32_t nextPowerOfTwo(uint32_t x) {
	xaDebugAssert(x != 0);
	// On modern CPUs this is supposed to be as fast as using the bsr instruction.
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x + 1;
}

static uint64_t nextPowerOfTwo(uint64_t x) {
	xaDebugAssert(x != 0);
	uint32_t p = 1;
	while (x > p) {
		p += p;
	}
	return p;
}

static uint32_t sdbmHash(const void *data_in, uint32_t size, uint32_t h = 5381) {
	const uint8_t *data = (const uint8_t *)data_in;
	uint32_t i = 0;
	while (i < size) {
		h = (h << 16) + (h << 6) - h + (uint32_t)data[i++];
	}
	return h;
}

// Note that this hash does not handle NaN properly.
static uint32_t sdbmFloatHash(const float *f, uint32_t count, uint32_t h = 5381) {
	for (uint32_t i = 0; i < count; i++) {
		union {
			float f;
			uint32_t i;
		} x = { f[i] };
		if (x.i == 0x80000000) x.i = 0;
		h = sdbmHash(&x, 4, h);
	}
	return h;
}

template <typename T>
static uint32_t hash(const T &t, uint32_t h = 5381) {
	return sdbmHash(&t, sizeof(T), h);
}

static uint32_t hash(const float &f, uint32_t h) {
	return sdbmFloatHash(&f, 1, h);
}

// Functors for hash table:
template <typename Key>
struct Hash {
	uint32_t operator()(const Key &k) const { return hash(k); }
};

template <typename Key>
struct Equal {
	bool operator()(const Key &k0, const Key &k1) const { return k0 == k1; }
};

class Vector2 {
public:
	typedef Vector2 const &Arg;

	Vector2() {}
	explicit Vector2(float f) :
			x(f),
			y(f) {}
	Vector2(float x, float y) :
			x(x),
			y(y) {}
	Vector2(Vector2::Arg v) :
			x(v.x),
			y(v.y) {}

	const Vector2 &operator=(Vector2::Arg v) {
		x = v.x;
		y = v.y;
		return *this;
	}
	const float *ptr() const { return &x; }

	void set(float _x, float _y) {
		x = _x;
		y = _y;
	}

	Vector2 operator-() const {
		return Vector2(-x, -y);
	}

	void operator+=(Vector2::Arg v) {
		x += v.x;
		y += v.y;
	}

	void operator-=(Vector2::Arg v) {
		x -= v.x;
		y -= v.y;
	}

	void operator*=(float s) {
		x *= s;
		y *= s;
	}

	void operator*=(Vector2::Arg v) {
		x *= v.x;
		y *= v.y;
	}

	friend bool operator==(Vector2::Arg a, Vector2::Arg b) {
		return a.x == b.x && a.y == b.y;
	}

	friend bool operator!=(Vector2::Arg a, Vector2::Arg b) {
		return a.x != b.x || a.y != b.y;
	}

	union {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4201)
#endif
		struct
		{
			float x, y;
		};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

		float component[2];
	};
};

Vector2 operator+(Vector2::Arg a, Vector2::Arg b) {
	return Vector2(a.x + b.x, a.y + b.y);
}

Vector2 operator-(Vector2::Arg a, Vector2::Arg b) {
	return Vector2(a.x - b.x, a.y - b.y);
}

Vector2 operator*(Vector2::Arg v, float s) {
	return Vector2(v.x * s, v.y * s);
}

Vector2 operator*(Vector2::Arg v1, Vector2::Arg v2) {
	return Vector2(v1.x * v2.x, v1.y * v2.y);
}

Vector2 operator/(Vector2::Arg v, float s) {
	return Vector2(v.x / s, v.y / s);
}

Vector2 lerp(Vector2::Arg v1, Vector2::Arg v2, float t) {
	const float s = 1.0f - t;
	return Vector2(v1.x * s + t * v2.x, v1.y * s + t * v2.y);
}

float dot(Vector2::Arg a, Vector2::Arg b) {
	return a.x * b.x + a.y * b.y;
}

float lengthSquared(Vector2::Arg v) {
	return v.x * v.x + v.y * v.y;
}

float length(Vector2::Arg v) {
	return sqrtf(lengthSquared(v));
}

float distance(Vector2::Arg a, Vector2::Arg b) {
	return length(a - b);
}

bool isNormalized(Vector2::Arg v, float epsilon = NV_NORMAL_EPSILON) {
	return equal(length(v), 1, epsilon);
}

Vector2 normalize(Vector2::Arg v, float epsilon = NV_EPSILON) {
	float l = length(v);
	xaDebugAssert(!isZero(l, epsilon));
#ifdef NDEBUG
	epsilon = 0; // silence unused parameter warning
#endif
	Vector2 n = v * (1.0f / l);
	xaDebugAssert(isNormalized(n));
	return n;
}

Vector2 normalizeSafe(Vector2::Arg v, Vector2::Arg fallback, float epsilon = NV_EPSILON) {
	float l = length(v);
	if (isZero(l, epsilon)) {
		return fallback;
	}
	return v * (1.0f / l);
}

bool equal(Vector2::Arg v1, Vector2::Arg v2, float epsilon = NV_EPSILON) {
	return equal(v1.x, v2.x, epsilon) && equal(v1.y, v2.y, epsilon);
}

Vector2 max(Vector2::Arg a, Vector2::Arg b) {
	return Vector2(std::max(a.x, b.x), std::max(a.y, b.y));
}

bool isFinite(Vector2::Arg v) {
	return std::isfinite(v.x) && std::isfinite(v.y);
}

// Note, this is the area scaled by 2!
float triangleArea(Vector2::Arg v0, Vector2::Arg v1) {
	return (v0.x * v1.y - v0.y * v1.x); // * 0.5f;
}
float triangleArea(Vector2::Arg a, Vector2::Arg b, Vector2::Arg c) {
	// IC: While it may be appealing to use the following expression:
	//return (c.x * a.y + a.x * b.y + b.x * c.y - b.x * a.y - c.x * b.y - a.x * c.y); // * 0.5f;
	// That's actually a terrible idea. Small triangles far from the origin can end up producing fairly large floating point
	// numbers and the results becomes very unstable and dependent on the order of the factors.
	// Instead, it's preferable to subtract the vertices first, and multiply the resulting small values together. The result
	// in this case is always much more accurate (as long as the triangle is small) and less dependent of the location of
	// the triangle.
	//return ((a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)); // * 0.5f;
	return triangleArea(a - c, b - c);
}

float triangleArea2(Vector2::Arg v1, Vector2::Arg v2, Vector2::Arg v3) {
	return 0.5f * (v3.x * v1.y + v1.x * v2.y + v2.x * v3.y - v2.x * v1.y - v3.x * v2.y - v1.x * v3.y);
}

static uint32_t hash(const Vector2 &v, uint32_t h) {
	return sdbmFloatHash(v.component, 2, h);
}

class Vector3 {
public:
	typedef Vector3 const &Arg;

	Vector3() {}
	explicit Vector3(float f) :
			x(f),
			y(f),
			z(f) {}
	Vector3(float x, float y, float z) :
			x(x),
			y(y),
			z(z) {}
	Vector3(Vector2::Arg v, float z) :
			x(v.x),
			y(v.y),
			z(z) {}
	Vector3(Vector3::Arg v) :
			x(v.x),
			y(v.y),
			z(v.z) {}

	const Vector3 &operator=(Vector3::Arg v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	Vector2 xy() const {
		return Vector2(x, y);
	}

	const float *ptr() const { return &x; }

	void set(float _x, float _y, float _z) {
		x = _x;
		y = _y;
		z = _z;
	}

	Vector3 operator-() const {
		return Vector3(-x, -y, -z);
	}

	void operator+=(Vector3::Arg v) {
		x += v.x;
		y += v.y;
		z += v.z;
	}

	void operator-=(Vector3::Arg v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
	}

	void operator*=(float s) {
		x *= s;
		y *= s;
		z *= s;
	}

	void operator/=(float s) {
		float is = 1.0f / s;
		x *= is;
		y *= is;
		z *= is;
	}

	void operator*=(Vector3::Arg v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
	}

	void operator/=(Vector3::Arg v) {
		x /= v.x;
		y /= v.y;
		z /= v.z;
	}

	friend bool operator==(Vector3::Arg a, Vector3::Arg b) {
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}

	friend bool operator!=(Vector3::Arg a, Vector3::Arg b) {
		return a.x != b.x || a.y != b.y || a.z != b.z;
	}

	union {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4201)
#endif
		struct
		{
			float x, y, z;
		};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

		float component[3];
	};
};

Vector3 add(Vector3::Arg a, Vector3::Arg b) {
	return Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
}
Vector3 add(Vector3::Arg a, float b) {
	return Vector3(a.x + b, a.y + b, a.z + b);
}
Vector3 operator+(Vector3::Arg a, Vector3::Arg b) {
	return add(a, b);
}
Vector3 operator+(Vector3::Arg a, float b) {
	return add(a, b);
}

Vector3 sub(Vector3::Arg a, Vector3::Arg b) {
	return Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vector3 sub(Vector3::Arg a, float b) {
	return Vector3(a.x - b, a.y - b, a.z - b);
}

Vector3 operator-(Vector3::Arg a, Vector3::Arg b) {
	return sub(a, b);
}

Vector3 operator-(Vector3::Arg a, float b) {
	return sub(a, b);
}

Vector3 cross(Vector3::Arg a, Vector3::Arg b) {
	return Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

Vector3 operator*(Vector3::Arg v, float s) {
	return Vector3(v.x * s, v.y * s, v.z * s);
}

Vector3 operator*(float s, Vector3::Arg v) {
	return Vector3(v.x * s, v.y * s, v.z * s);
}

Vector3 operator*(Vector3::Arg v, Vector3::Arg s) {
	return Vector3(v.x * s.x, v.y * s.y, v.z * s.z);
}

Vector3 operator/(Vector3::Arg v, float s) {
	return v * (1.0f / s);
}

Vector3 lerp(Vector3::Arg v1, Vector3::Arg v2, float t) {
	const float s = 1.0f - t;
	return Vector3(v1.x * s + t * v2.x, v1.y * s + t * v2.y, v1.z * s + t * v2.z);
}

float dot(Vector3::Arg a, Vector3::Arg b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

float lengthSquared(Vector3::Arg v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

float length(Vector3::Arg v) {
	return sqrtf(lengthSquared(v));
}

float distance(Vector3::Arg a, Vector3::Arg b) {
	return length(a - b);
}

float distanceSquared(Vector3::Arg a, Vector3::Arg b) {
	return lengthSquared(a - b);
}

bool isNormalized(Vector3::Arg v, float epsilon = NV_NORMAL_EPSILON) {
	return equal(length(v), 1, epsilon);
}

Vector3 normalize(Vector3::Arg v, float epsilon = NV_EPSILON) {
	float l = length(v);
	xaDebugAssert(!isZero(l, epsilon));
#ifdef NDEBUG
	epsilon = 0; // silence unused parameter warning
#endif
	Vector3 n = v * (1.0f / l);
	xaDebugAssert(isNormalized(n));
	return n;
}

Vector3 normalizeSafe(Vector3::Arg v, Vector3::Arg fallback, float epsilon = NV_EPSILON) {
	float l = length(v);
	if (isZero(l, epsilon)) {
		return fallback;
	}
	return v * (1.0f / l);
}

bool equal(Vector3::Arg v1, Vector3::Arg v2, float epsilon = NV_EPSILON) {
	return equal(v1.x, v2.x, epsilon) && equal(v1.y, v2.y, epsilon) && equal(v1.z, v2.z, epsilon);
}

Vector3 min(Vector3::Arg a, Vector3::Arg b) {
	return Vector3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

Vector3 max(Vector3::Arg a, Vector3::Arg b) {
	return Vector3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

Vector3 clamp(Vector3::Arg v, float min, float max) {
	return Vector3(clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max));
}

Vector3 saturate(Vector3::Arg v) {
	return Vector3(saturate(v.x), saturate(v.y), saturate(v.z));
}

Vector3 floor(Vector3::Arg v) {
	return Vector3(floorf(v.x), floorf(v.y), floorf(v.z));
}

bool isFinite(Vector3::Arg v) {
	return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static uint32_t hash(const Vector3 &v, uint32_t h) {
	return sdbmFloatHash(v.component, 3, h);
}

/// Basis class to compute tangent space basis, ortogonalizations and to
/// transform vectors from one space to another.
class Basis {
public:
	/// Create a null basis.
	Basis() :
			tangent(0, 0, 0),
			bitangent(0, 0, 0),
			normal(0, 0, 0) {}

	void buildFrameForDirection(Vector3::Arg d, float angle = 0) {
		xaAssert(isNormalized(d));
		normal = d;
		// Choose minimum axis.
		if (fabsf(normal.x) < fabsf(normal.y) && fabsf(normal.x) < fabsf(normal.z)) {
			tangent = Vector3(1, 0, 0);
		} else if (fabsf(normal.y) < fabsf(normal.z)) {
			tangent = Vector3(0, 1, 0);
		} else {
			tangent = Vector3(0, 0, 1);
		}
		// Ortogonalize
		tangent -= normal * dot(normal, tangent);
		tangent = normalize(tangent);
		bitangent = cross(normal, tangent);
		// Rotate frame around normal according to angle.
		if (angle != 0.0f) {
			float c = cosf(angle);
			float s = sinf(angle);
			Vector3 tmp = c * tangent - s * bitangent;
			bitangent = s * tangent + c * bitangent;
			tangent = tmp;
		}
	}

	Vector3 tangent;
	Vector3 bitangent;
	Vector3 normal;
};

// Simple bit array.
class BitArray {
public:
	BitArray() :
			m_size(0) {}
	BitArray(uint32_t sz) {
		resize(sz);
	}

	uint32_t size() const {
		return m_size;
	}

	void clear() {
		resize(0);
	}

	void resize(uint32_t new_size) {
		m_size = new_size;
		m_wordArray.resize((m_size + 31) >> 5);
	}

	/// Get bit.
	bool bitAt(uint32_t b) const {
		xaDebugAssert(b < m_size);
		return (m_wordArray[b >> 5] & (1 << (b & 31))) != 0;
	}

	// Set a bit.
	void setBitAt(uint32_t idx) {
		xaDebugAssert(idx < m_size);
		m_wordArray[idx >> 5] |= (1 << (idx & 31));
	}

	// Toggle a bit.
	void toggleBitAt(uint32_t idx) {
		xaDebugAssert(idx < m_size);
		m_wordArray[idx >> 5] ^= (1 << (idx & 31));
	}

	// Set a bit to the given value. @@ Rename modifyBitAt?
	void setBitAt(uint32_t idx, bool b) {
		xaDebugAssert(idx < m_size);
		m_wordArray[idx >> 5] = setBits(m_wordArray[idx >> 5], 1 << (idx & 31), b);
		xaDebugAssert(bitAt(idx) == b);
	}

	// Clear all the bits.
	void clearAll() {
		memset(m_wordArray.data(), 0, m_wordArray.size() * sizeof(uint32_t));
	}

	// Set all the bits.
	void setAll() {
		memset(m_wordArray.data(), 0xFF, m_wordArray.size() * sizeof(uint32_t));
	}

private:
	// See "Conditionally set or clear bits without branching" at http://graphics.stanford.edu/~seander/bithacks.html
	uint32_t setBits(uint32_t w, uint32_t m, bool b) {
		return (w & ~m) | (-int(b) & m);
	}

	// Number of bits stored.
	uint32_t m_size;

	// Array of bits.
	std::vector<uint32_t> m_wordArray;
};

/// Bit map. This should probably be called BitImage.
class BitMap {
public:
	BitMap() :
			m_width(0),
			m_height(0) {}
	BitMap(uint32_t w, uint32_t h) :
			m_width(w),
			m_height(h),
			m_bitArray(w * h) {}

	uint32_t width() const {
		return m_width;
	}
	uint32_t height() const {
		return m_height;
	}

	void resize(uint32_t w, uint32_t h, bool initValue) {
		BitArray tmp(w * h);
		if (initValue)
			tmp.setAll();
		else
			tmp.clearAll();
		// @@ Copying one bit at a time. This could be much faster.
		for (uint32_t y = 0; y < m_height; y++) {
			for (uint32_t x = 0; x < m_width; x++) {
				//tmp.setBitAt(y*w + x, bitAt(x, y));
				if (bitAt(x, y) != initValue) tmp.toggleBitAt(y * w + x);
			}
		}
		std::swap(m_bitArray, tmp);
		m_width = w;
		m_height = h;
	}

	bool bitAt(uint32_t x, uint32_t y) const {
		xaDebugAssert(x < m_width && y < m_height);
		return m_bitArray.bitAt(y * m_width + x);
	}

	void setBitAt(uint32_t x, uint32_t y) {
		xaDebugAssert(x < m_width && y < m_height);
		m_bitArray.setBitAt(y * m_width + x);
	}

	void clearAll() {
		m_bitArray.clearAll();
	}

private:
	uint32_t m_width;
	uint32_t m_height;
	BitArray m_bitArray;
};

// Axis Aligned Bounding Box.
class Box {
public:
	Box() {}
	Box(const Box &b) :
			minCorner(b.minCorner),
			maxCorner(b.maxCorner) {}
	Box(const Vector3 &mins, const Vector3 &maxs) :
			minCorner(mins),
			maxCorner(maxs) {}

	operator const float *() const {
		return reinterpret_cast<const float *>(this);
	}

	// Clear the bounds.
	void clearBounds() {
		minCorner.set(FLT_MAX, FLT_MAX, FLT_MAX);
		maxCorner.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	}

	// Return extents of the box.
	Vector3 extents() const {
		return (maxCorner - minCorner) * 0.5f;
	}

	// Add a point to this box.
	void addPointToBounds(const Vector3 &p) {
		minCorner = min(minCorner, p);
		maxCorner = max(maxCorner, p);
	}

	// Get the volume of the box.
	float volume() const {
		Vector3 d = extents();
		return 8.0f * (d.x * d.y * d.z);
	}

	Vector3 minCorner;
	Vector3 maxCorner;
};

class Fit {
public:
	static Vector3 computeCentroid(int n, const Vector3 *__restrict points) {
		Vector3 centroid(0.0f);
		for (int i = 0; i < n; i++) {
			centroid += points[i];
		}
		centroid /= float(n);
		return centroid;
	}

	static Vector3 computeCovariance(int n, const Vector3 *__restrict points, float *__restrict covariance) {
		// compute the centroid
		Vector3 centroid = computeCentroid(n, points);
		// compute covariance matrix
		for (int i = 0; i < 6; i++) {
			covariance[i] = 0.0f;
		}
		for (int i = 0; i < n; i++) {
			Vector3 v = points[i] - centroid;
			covariance[0] += v.x * v.x;
			covariance[1] += v.x * v.y;
			covariance[2] += v.x * v.z;
			covariance[3] += v.y * v.y;
			covariance[4] += v.y * v.z;
			covariance[5] += v.z * v.z;
		}
		return centroid;
	}

	static bool isPlanar(int n, const Vector3 *points, float epsilon = NV_EPSILON) {
		// compute the centroid and covariance
		float matrix[6];
		computeCovariance(n, points, matrix);
		float eigenValues[3];
		Vector3 eigenVectors[3];
		if (!eigenSolveSymmetric3(matrix, eigenValues, eigenVectors)) {
			return false;
		}
		return eigenValues[2] < epsilon;
	}

	// Tridiagonal solver from Charles Bloom.
	// Householder transforms followed by QL decomposition.
	// Seems to be based on the code from Numerical Recipes in C.
	static bool eigenSolveSymmetric3(const float matrix[6], float eigenValues[3], Vector3 eigenVectors[3]) {
		xaDebugAssert(matrix != NULL && eigenValues != NULL && eigenVectors != NULL);
		float subd[3];
		float diag[3];
		float work[3][3];
		work[0][0] = matrix[0];
		work[0][1] = work[1][0] = matrix[1];
		work[0][2] = work[2][0] = matrix[2];
		work[1][1] = matrix[3];
		work[1][2] = work[2][1] = matrix[4];
		work[2][2] = matrix[5];
		EigenSolver3_Tridiagonal(work, diag, subd);
		if (!EigenSolver3_QLAlgorithm(work, diag, subd)) {
			for (int i = 0; i < 3; i++) {
				eigenValues[i] = 0;
				eigenVectors[i] = Vector3(0);
			}
			return false;
		}
		for (int i = 0; i < 3; i++) {
			eigenValues[i] = (float)diag[i];
		}
		// eigenvectors are the columns; make them the rows :
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				eigenVectors[j].component[i] = (float)work[i][j];
			}
		}
		// shuffle to sort by singular value :
		if (eigenValues[2] > eigenValues[0] && eigenValues[2] > eigenValues[1]) {
			std::swap(eigenValues[0], eigenValues[2]);
			std::swap(eigenVectors[0], eigenVectors[2]);
		}
		if (eigenValues[1] > eigenValues[0]) {
			std::swap(eigenValues[0], eigenValues[1]);
			std::swap(eigenVectors[0], eigenVectors[1]);
		}
		if (eigenValues[2] > eigenValues[1]) {
			std::swap(eigenValues[1], eigenValues[2]);
			std::swap(eigenVectors[1], eigenVectors[2]);
		}
		xaDebugAssert(eigenValues[0] >= eigenValues[1] && eigenValues[0] >= eigenValues[2]);
		xaDebugAssert(eigenValues[1] >= eigenValues[2]);
		return true;
	}

private:
	static void EigenSolver3_Tridiagonal(float mat[3][3], float *diag, float *subd) {
		// Householder reduction T = Q^t M Q
		//   Input:
		//     mat, symmetric 3x3 matrix M
		//   Output:
		//     mat, orthogonal matrix Q
		//     diag, diagonal entries of T
		//     subd, subdiagonal entries of T (T is symmetric)
		const float epsilon = 1e-08f;
		float a = mat[0][0];
		float b = mat[0][1];
		float c = mat[0][2];
		float d = mat[1][1];
		float e = mat[1][2];
		float f = mat[2][2];
		diag[0] = a;
		subd[2] = 0.f;
		if (fabsf(c) >= epsilon) {
			const float ell = sqrtf(b * b + c * c);
			b /= ell;
			c /= ell;
			const float q = 2 * b * e + c * (f - d);
			diag[1] = d + c * q;
			diag[2] = f - c * q;
			subd[0] = ell;
			subd[1] = e - b * q;
			mat[0][0] = 1;
			mat[0][1] = 0;
			mat[0][2] = 0;
			mat[1][0] = 0;
			mat[1][1] = b;
			mat[1][2] = c;
			mat[2][0] = 0;
			mat[2][1] = c;
			mat[2][2] = -b;
		} else {
			diag[1] = d;
			diag[2] = f;
			subd[0] = b;
			subd[1] = e;
			mat[0][0] = 1;
			mat[0][1] = 0;
			mat[0][2] = 0;
			mat[1][0] = 0;
			mat[1][1] = 1;
			mat[1][2] = 0;
			mat[2][0] = 0;
			mat[2][1] = 0;
			mat[2][2] = 1;
		}
	}

	static bool EigenSolver3_QLAlgorithm(float mat[3][3], float *diag, float *subd) {
		// QL iteration with implicit shifting to reduce matrix from tridiagonal
		// to diagonal
		const int maxiter = 32;
		for (int ell = 0; ell < 3; ell++) {
			int iter;
			for (iter = 0; iter < maxiter; iter++) {
				int m;
				for (m = ell; m <= 1; m++) {
					float dd = fabsf(diag[m]) + fabsf(diag[m + 1]);
					if (fabsf(subd[m]) + dd == dd)
						break;
				}
				if (m == ell)
					break;
				float g = (diag[ell + 1] - diag[ell]) / (2 * subd[ell]);
				float r = sqrtf(g * g + 1);
				if (g < 0)
					g = diag[m] - diag[ell] + subd[ell] / (g - r);
				else
					g = diag[m] - diag[ell] + subd[ell] / (g + r);
				float s = 1, c = 1, p = 0;
				for (int i = m - 1; i >= ell; i--) {
					float f = s * subd[i], b = c * subd[i];
					if (fabsf(f) >= fabsf(g)) {
						c = g / f;
						r = sqrtf(c * c + 1);
						subd[i + 1] = f * r;
						c *= (s = 1 / r);
					} else {
						s = f / g;
						r = sqrtf(s * s + 1);
						subd[i + 1] = g * r;
						s *= (c = 1 / r);
					}
					g = diag[i + 1] - p;
					r = (diag[i] - g) * s + 2 * b * c;
					p = s * r;
					diag[i + 1] = g + p;
					g = c * r - b;
					for (int k = 0; k < 3; k++) {
						f = mat[k][i + 1];
						mat[k][i + 1] = s * mat[k][i] + c * f;
						mat[k][i] = c * mat[k][i] - s * f;
					}
				}
				diag[ell] -= p;
				subd[ell] = g;
				subd[m] = 0;
			}
			if (iter == maxiter)
				// should not get here under normal circumstances
				return false;
		}
		return true;
	}
};

/// Fixed size vector class.
class FullVector {
public:
	FullVector(uint32_t dim) { m_array.resize(dim); }
	FullVector(const FullVector &v) :
			m_array(v.m_array) {}

	const FullVector &operator=(const FullVector &v) {
		xaAssert(dimension() == v.dimension());
		m_array = v.m_array;
		return *this;
	}

	uint32_t dimension() const { return m_array.size(); }
	const float &operator[](uint32_t index) const { return m_array[index]; }
	float &operator[](uint32_t index) { return m_array[index]; }

	void fill(float f) {
		const uint32_t dim = dimension();
		for (uint32_t i = 0; i < dim; i++) {
			m_array[i] = f;
		}
	}

	void operator+=(const FullVector &v) {
		xaDebugAssert(dimension() == v.dimension());
		const uint32_t dim = dimension();
		for (uint32_t i = 0; i < dim; i++) {
			m_array[i] += v.m_array[i];
		}
	}

	void operator-=(const FullVector &v) {
		xaDebugAssert(dimension() == v.dimension());
		const uint32_t dim = dimension();
		for (uint32_t i = 0; i < dim; i++) {
			m_array[i] -= v.m_array[i];
		}
	}

	void operator*=(const FullVector &v) {
		xaDebugAssert(dimension() == v.dimension());
		const uint32_t dim = dimension();
		for (uint32_t i = 0; i < dim; i++) {
			m_array[i] *= v.m_array[i];
		}
	}

	void operator+=(float f) {
		const uint32_t dim = dimension();
		for (uint32_t i = 0; i < dim; i++) {
			m_array[i] += f;
		}
	}

	void operator-=(float f) {
		const uint32_t dim = dimension();
		for (uint32_t i = 0; i < dim; i++) {
			m_array[i] -= f;
		}
	}

	void operator*=(float f) {
		const uint32_t dim = dimension();
		for (uint32_t i = 0; i < dim; i++) {
			m_array[i] *= f;
		}
	}

private:
	std::vector<float> m_array;
};

namespace halfedge {
class Face;
class Vertex;

class Edge {
public:
	uint32_t id;
	Edge *next;
	Edge *prev; // This is not strictly half-edge, but makes algorithms easier and faster.
	Edge *pair;
	Vertex *vertex;
	Face *face;

	// Default constructor.
	Edge(uint32_t id) :
			id(id),
			next(NULL),
			prev(NULL),
			pair(NULL),
			vertex(NULL),
			face(NULL) {}

	// Vertex queries.
	const Vertex *from() const {
		return vertex;
	}

	Vertex *from() {
		return vertex;
	}

	const Vertex *to() const {
		return pair->vertex; // This used to be 'next->vertex', but that changed often when the connectivity of the mesh changes.
	}

	Vertex *to() {
		return pair->vertex;
	}

	// Edge queries.
	void setNext(Edge *e) {
		next = e;
		if (e != NULL) e->prev = this;
	}
	void setPrev(Edge *e) {
		prev = e;
		if (e != NULL) e->next = this;
	}

	// @@ It would be more simple to only check m_pair == NULL
	// Face queries.
	bool isBoundary() const {
		return !(face && pair->face);
	}

	// @@ This is not exactly accurate, we should compare the texture coordinates...
	bool isSeam() const {
		return vertex != pair->next->vertex || next->vertex != pair->vertex;
	}

	bool isNormalSeam() const;
	bool isTextureSeam() const;

	bool isValid() const {
		// null face is OK.
		if (next == NULL || prev == NULL || pair == NULL || vertex == NULL) return false;
		if (next->prev != this) return false;
		if (prev->next != this) return false;
		if (pair->pair != this) return false;
		return true;
	}

	float length() const;

	// Return angle between this edge and the previous one.
	float angle() const;
};

class Vertex {
public:
	uint32_t id;
	uint32_t original_id;
	Edge *edge;
	Vertex *next;
	Vertex *prev;
	Vector3 pos;
	Vector3 nor;
	Vector2 tex;

	Vertex(uint32_t id) :
			id(id),
			original_id(id),
			edge(NULL),
			pos(0.0f),
			nor(0.0f),
			tex(0.0f) {
		next = this;
		prev = this;
	}

	// Set first edge of all colocals.
	void setEdge(Edge *e) {
		for (VertexIterator it(colocals()); !it.isDone(); it.advance()) {
			it.current()->edge = e;
		}
	}

	// Update position of all colocals.
	void setPos(const Vector3 &p) {
		for (VertexIterator it(colocals()); !it.isDone(); it.advance()) {
			it.current()->pos = p;
		}
	}

	bool isFirstColocal() const {
		return firstColocal() == this;
	}

	const Vertex *firstColocal() const {
		uint32_t firstId = id;
		const Vertex *vertex = this;
		for (ConstVertexIterator it(colocals()); !it.isDone(); it.advance()) {
			if (it.current()->id < firstId) {
				firstId = vertex->id;
				vertex = it.current();
			}
		}
		return vertex;
	}

	Vertex *firstColocal() {
		Vertex *vertex = this;
		uint32_t firstId = id;
		for (VertexIterator it(colocals()); !it.isDone(); it.advance()) {
			if (it.current()->id < firstId) {
				firstId = vertex->id;
				vertex = it.current();
			}
		}
		return vertex;
	}

	bool isColocal(const Vertex *v) const {
		if (this == v) return true;
		if (pos != v->pos) return false;
		for (ConstVertexIterator it(colocals()); !it.isDone(); it.advance()) {
			if (v == it.current()) {
				return true;
			}
		}
		return false;
	}

	void linkColocal(Vertex *v) {
		next->prev = v;
		v->next = next;
		next = v;
		v->prev = this;
	}
	void unlinkColocal() {
		next->prev = prev;
		prev->next = next;
		next = this;
		prev = this;
	}

	// @@ Note: This only works if linkBoundary has been called.
	bool isBoundary() const {
		return (edge && !edge->face);
	}

	// Iterator that visits the edges around this vertex in counterclockwise order.
	class EdgeIterator //: public Iterator<Edge *>
	{
	public:
		EdgeIterator(Edge *e) :
				m_end(NULL),
				m_current(e) {}

		virtual void advance() {
			if (m_end == NULL) m_end = m_current;
			m_current = m_current->pair->next;
			//m_current = m_current->prev->pair;
		}

		virtual bool isDone() const {
			return m_end == m_current;
		}
		virtual Edge *current() const {
			return m_current;
		}
		Vertex *vertex() const {
			return m_current->vertex;
		}

	private:
		Edge *m_end;
		Edge *m_current;
	};

	EdgeIterator edges() {
		return EdgeIterator(edge);
	}
	EdgeIterator edges(Edge *e) {
		return EdgeIterator(e);
	}

	// Iterator that visits the edges around this vertex in counterclockwise order.
	class ConstEdgeIterator //: public Iterator<Edge *>
	{
	public:
		ConstEdgeIterator(const Edge *e) :
				m_end(NULL),
				m_current(e) {}
		ConstEdgeIterator(EdgeIterator it) :
				m_end(NULL),
				m_current(it.current()) {}

		virtual void advance() {
			if (m_end == NULL) m_end = m_current;
			m_current = m_current->pair->next;
			//m_current = m_current->prev->pair;
		}

		virtual bool isDone() const {
			return m_end == m_current;
		}
		virtual const Edge *current() const {
			return m_current;
		}
		const Vertex *vertex() const {
			return m_current->to();
		}

	private:
		const Edge *m_end;
		const Edge *m_current;
	};

	ConstEdgeIterator edges() const {
		return ConstEdgeIterator(edge);
	}
	ConstEdgeIterator edges(const Edge *e) const {
		return ConstEdgeIterator(e);
	}

	// Iterator that visits all the colocal vertices.
	class VertexIterator //: public Iterator<Edge *>
	{
	public:
		VertexIterator(Vertex *v) :
				m_end(NULL),
				m_current(v) {}

		virtual void advance() {
			if (m_end == NULL) m_end = m_current;
			m_current = m_current->next;
		}

		virtual bool isDone() const {
			return m_end == m_current;
		}
		virtual Vertex *current() const {
			return m_current;
		}

	private:
		Vertex *m_end;
		Vertex *m_current;
	};

	VertexIterator colocals() {
		return VertexIterator(this);
	}

	// Iterator that visits all the colocal vertices.
	class ConstVertexIterator //: public Iterator<Edge *>
	{
	public:
		ConstVertexIterator(const Vertex *v) :
				m_end(NULL),
				m_current(v) {}

		virtual void advance() {
			if (m_end == NULL) m_end = m_current;
			m_current = m_current->next;
		}

		virtual bool isDone() const {
			return m_end == m_current;
		}
		virtual const Vertex *current() const {
			return m_current;
		}

	private:
		const Vertex *m_end;
		const Vertex *m_current;
	};

	ConstVertexIterator colocals() const {
		return ConstVertexIterator(this);
	}
};

bool Edge::isNormalSeam() const {
	return (vertex->nor != pair->next->vertex->nor || next->vertex->nor != pair->vertex->nor);
}

bool Edge::isTextureSeam() const {
	return (vertex->tex != pair->next->vertex->tex || next->vertex->tex != pair->vertex->tex);
}

float Edge::length() const {
	return internal::length(to()->pos - from()->pos);
}

float Edge::angle() const {
	Vector3 p = vertex->pos;
	Vector3 a = prev->vertex->pos;
	Vector3 b = next->vertex->pos;
	Vector3 v0 = a - p;
	Vector3 v1 = b - p;
	return acosf(dot(v0, v1) / (internal::length(v0) * internal::length(v1)));
}

class Face {
public:
	uint32_t id;
	uint16_t group;
	uint16_t material;
	Edge *edge;

	Face(uint32_t id) :
			id(id),
			group(uint16_t(~0)),
			material(uint16_t(~0)),
			edge(NULL) {}

	float area() const {
		float area = 0;
		const Vector3 &v0 = edge->from()->pos;
		for (ConstEdgeIterator it(edges(edge->next)); it.current() != edge->prev; it.advance()) {
			const Edge *e = it.current();
			const Vector3 &v1 = e->vertex->pos;
			const Vector3 &v2 = e->next->vertex->pos;
			area += length(cross(v1 - v0, v2 - v0));
		}
		return area * 0.5f;
	}

	float parametricArea() const {
		float area = 0;
		const Vector2 &v0 = edge->from()->tex;
		for (ConstEdgeIterator it(edges(edge->next)); it.current() != edge->prev; it.advance()) {
			const Edge *e = it.current();
			const Vector2 &v1 = e->vertex->tex;
			const Vector2 &v2 = e->next->vertex->tex;
			area += triangleArea(v0, v1, v2);
		}
		return area * 0.5f;
	}

	Vector3 normal() const {
		Vector3 n(0);
		const Vertex *vertex0 = NULL;
		for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance()) {
			const Edge *e = it.current();
			xaAssert(e != NULL);
			if (vertex0 == NULL) {
				vertex0 = e->vertex;
			} else if (e->next->vertex != vertex0) {
				const halfedge::Vertex *vertex1 = e->from();
				const halfedge::Vertex *vertex2 = e->to();
				const Vector3 &p0 = vertex0->pos;
				const Vector3 &p1 = vertex1->pos;
				const Vector3 &p2 = vertex2->pos;
				Vector3 v10 = p1 - p0;
				Vector3 v20 = p2 - p0;
				n += cross(v10, v20);
			}
		}
		return normalizeSafe(n, Vector3(0, 0, 1), 0.0f);
	}

	Vector3 centroid() const {
		Vector3 sum(0.0f);
		uint32_t count = 0;
		for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance()) {
			const Edge *e = it.current();
			sum += e->from()->pos;
			count++;
		}
		return sum / float(count);
	}

	// Unnormalized face normal assuming it's a triangle.
	Vector3 triangleNormal() const {
		Vector3 p0 = edge->vertex->pos;
		Vector3 p1 = edge->next->vertex->pos;
		Vector3 p2 = edge->next->next->vertex->pos;
		Vector3 e0 = p2 - p0;
		Vector3 e1 = p1 - p0;
		return normalizeSafe(cross(e0, e1), Vector3(0), 0.0f);
	}

	Vector3 triangleNormalAreaScaled() const {
		Vector3 p0 = edge->vertex->pos;
		Vector3 p1 = edge->next->vertex->pos;
		Vector3 p2 = edge->next->next->vertex->pos;
		Vector3 e0 = p2 - p0;
		Vector3 e1 = p1 - p0;
		return cross(e0, e1);
	}

	// Average of the edge midpoints weighted by the edge length.
	// I want a point inside the triangle, but closer to the cirumcenter.
	Vector3 triangleCenter() const {
		Vector3 p0 = edge->vertex->pos;
		Vector3 p1 = edge->next->vertex->pos;
		Vector3 p2 = edge->next->next->vertex->pos;
		float l0 = length(p1 - p0);
		float l1 = length(p2 - p1);
		float l2 = length(p0 - p2);
		Vector3 m0 = (p0 + p1) * l0 / (l0 + l1 + l2);
		Vector3 m1 = (p1 + p2) * l1 / (l0 + l1 + l2);
		Vector3 m2 = (p2 + p0) * l2 / (l0 + l1 + l2);
		return m0 + m1 + m2;
	}

	bool isValid() const {
		uint32_t count = 0;
		for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance()) {
			const Edge *e = it.current();
			if (e->face != this) return false;
			if (!e->isValid()) return false;
			if (!e->pair->isValid()) return false;
			count++;
		}
		if (count < 3) return false;
		return true;
	}

	bool contains(const Edge *e) const {
		for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance()) {
			if (it.current() == e) return true;
		}
		return false;
	}

	uint32_t edgeCount() const {
		uint32_t count = 0;
		for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance()) {
			++count;
		}
		return count;
	}

	// The iterator that visits the edges of this face in clockwise order.
	class EdgeIterator //: public Iterator<Edge *>
	{
	public:
		EdgeIterator(Edge *e) :
				m_end(NULL),
				m_current(e) {}

		virtual void advance() {
			if (m_end == NULL) m_end = m_current;
			m_current = m_current->next;
		}

		virtual bool isDone() const {
			return m_end == m_current;
		}
		virtual Edge *current() const {
			return m_current;
		}
		Vertex *vertex() const {
			return m_current->vertex;
		}

	private:
		Edge *m_end;
		Edge *m_current;
	};

	EdgeIterator edges() {
		return EdgeIterator(edge);
	}
	EdgeIterator edges(Edge *e) {
		xaDebugAssert(contains(e));
		return EdgeIterator(e);
	}

	// The iterator that visits the edges of this face in clockwise order.
	class ConstEdgeIterator //: public Iterator<const Edge *>
	{
	public:
		ConstEdgeIterator(const Edge *e) :
				m_end(NULL),
				m_current(e) {}
		ConstEdgeIterator(const EdgeIterator &it) :
				m_end(NULL),
				m_current(it.current()) {}

		virtual void advance() {
			if (m_end == NULL) m_end = m_current;
			m_current = m_current->next;
		}

		virtual bool isDone() const {
			return m_end == m_current;
		}
		virtual const Edge *current() const {
			return m_current;
		}
		const Vertex *vertex() const {
			return m_current->vertex;
		}

	private:
		const Edge *m_end;
		const Edge *m_current;
	};

	ConstEdgeIterator edges() const {
		return ConstEdgeIterator(edge);
	}
	ConstEdgeIterator edges(const Edge *e) const {
		xaDebugAssert(contains(e));
		return ConstEdgeIterator(e);
	}
};

/// Simple half edge mesh designed for dynamic mesh manipulation.
class Mesh {
public:
	Mesh() :
			m_colocalVertexCount(0) {}

	Mesh(const Mesh *mesh) {
		// Copy mesh vertices.
		const uint32_t vertexCount = mesh->vertexCount();
		m_vertexArray.resize(vertexCount);
		for (uint32_t v = 0; v < vertexCount; v++) {
			const Vertex *vertex = mesh->vertexAt(v);
			xaDebugAssert(vertex->id == v);
			m_vertexArray[v] = new Vertex(v);
			m_vertexArray[v]->pos = vertex->pos;
			m_vertexArray[v]->nor = vertex->nor;
			m_vertexArray[v]->tex = vertex->tex;
		}
		m_colocalVertexCount = vertexCount;
		// Copy mesh faces.
		const uint32_t faceCount = mesh->faceCount();
		std::vector<uint32_t> indexArray;
		indexArray.reserve(3);
		for (uint32_t f = 0; f < faceCount; f++) {
			const Face *face = mesh->faceAt(f);
			for (Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				const Vertex *vertex = it.current()->from();
				indexArray.push_back(vertex->id);
			}
			addFace(indexArray);
			indexArray.clear();
		}
	}

	~Mesh() {
		clear();
	}

	void clear() {
		for (size_t i = 0; i < m_vertexArray.size(); i++)
			delete m_vertexArray[i];
		m_vertexArray.clear();
		for (auto it = m_edgeMap.begin(); it != m_edgeMap.end(); it++)
			delete it->second;
		m_edgeArray.clear();
		m_edgeMap.clear();
		for (size_t i = 0; i < m_faceArray.size(); i++)
			delete m_faceArray[i];
		m_faceArray.clear();
	}

	Vertex *addVertex(const Vector3 &pos) {
		xaDebugAssert(isFinite(pos));
		Vertex *v = new Vertex(m_vertexArray.size());
		v->pos = pos;
		m_vertexArray.push_back(v);
		return v;
	}

	/// Link colocal vertices based on geometric location only.
	void linkColocals() {
		xaPrint("--- Linking colocals:\n");
		const uint32_t vertexCount = this->vertexCount();
		std::unordered_map<Vector3, Vertex *, Hash<Vector3>, Equal<Vector3> > vertexMap;
		vertexMap.reserve(vertexCount);
		for (uint32_t v = 0; v < vertexCount; v++) {
			Vertex *vertex = vertexAt(v);
			Vertex *colocal = vertexMap[vertex->pos];
			if (colocal) {
				colocal->linkColocal(vertex);
			} else {
				vertexMap[vertex->pos] = vertex;
			}
		}
		m_colocalVertexCount = vertexMap.size();
		xaPrint("---   %d vertex positions.\n", m_colocalVertexCount);
		// @@ Remove duplicated vertices? or just leave them as colocals?
	}

	void linkColocalsWithCanonicalMap(const std::vector<uint32_t> &canonicalMap) {
		xaPrint("--- Linking colocals:\n");
		uint32_t vertexMapSize = 0;
		for (uint32_t i = 0; i < canonicalMap.size(); i++) {
			vertexMapSize = std::max(vertexMapSize, canonicalMap[i] + 1);
		}
		std::vector<Vertex *> vertexMap;
		vertexMap.resize(vertexMapSize, NULL);
		m_colocalVertexCount = 0;
		const uint32_t vertexCount = this->vertexCount();
		for (uint32_t v = 0; v < vertexCount; v++) {
			Vertex *vertex = vertexAt(v);
			Vertex *colocal = vertexMap[canonicalMap[v]];
			if (colocal != NULL) {
				xaDebugAssert(vertex->pos == colocal->pos);
				colocal->linkColocal(vertex);
			} else {
				vertexMap[canonicalMap[v]] = vertex;
				m_colocalVertexCount++;
			}
		}
		xaPrint("---   %d vertex positions.\n", m_colocalVertexCount);
	}

	Face *addFace() {
		Face *f = new Face(m_faceArray.size());
		m_faceArray.push_back(f);
		return f;
	}

	Face *addFace(uint32_t v0, uint32_t v1, uint32_t v2) {
		uint32_t indexArray[3];
		indexArray[0] = v0;
		indexArray[1] = v1;
		indexArray[2] = v2;
		return addFace(indexArray, 3, 0, 3);
	}

	Face *addUniqueFace(uint32_t v0, uint32_t v1, uint32_t v2) {

		int base_vertex = m_vertexArray.size();

		uint32_t ids[3] = { v0, v1, v2 };

		Vector3 base[3] = {
			m_vertexArray[v0]->pos,
			m_vertexArray[v1]->pos,
			m_vertexArray[v2]->pos,
		};

		//make sure its not a degenerate
		bool degenerate = distanceSquared(base[0], base[1]) < NV_EPSILON || distanceSquared(base[0], base[2]) < NV_EPSILON || distanceSquared(base[1], base[2]) < NV_EPSILON;
		xaDebugAssert(!degenerate);

		float min_x = 0;

		for (int i = 0; i < 3; i++) {
			if (i == 0 || m_vertexArray[v0]->pos.x < min_x) {
				min_x = m_vertexArray[v0]->pos.x;
			}
		}

		float max_x = 0;

		for (int j = 0; j < m_vertexArray.size(); j++) {
			if (j == 0 || m_vertexArray[j]->pos.x > max_x) { //vertex already exists
				max_x = m_vertexArray[j]->pos.x;
			}
		}

		//separate from everything else, in x axis
		for (int i = 0; i < 3; i++) {

			base[i].x -= min_x;
			base[i].x += max_x + 10.0;
		}

		for (int i = 0; i < 3; i++) {
			Vertex *v = new Vertex(m_vertexArray.size());
			v->pos = base[i];
			v->nor = m_vertexArray[ids[i]]->nor,
			v->tex = m_vertexArray[ids[i]]->tex,

			v->original_id = ids[i];
			m_vertexArray.push_back(v);
		}

		uint32_t indexArray[3];
		indexArray[0] = base_vertex + 0;
		indexArray[1] = base_vertex + 1;
		indexArray[2] = base_vertex + 2;
		return addFace(indexArray, 3, 0, 3);
	}

	Face *addFace(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
		uint32_t indexArray[4];
		indexArray[0] = v0;
		indexArray[1] = v1;
		indexArray[2] = v2;
		indexArray[3] = v3;
		return addFace(indexArray, 4, 0, 4);
	}

	Face *addFace(const std::vector<uint32_t> &indexArray) {
		return addFace(indexArray, 0, indexArray.size());
	}

	Face *addFace(const std::vector<uint32_t> &indexArray, uint32_t first, uint32_t num) {
		return addFace(indexArray.data(), (uint32_t)indexArray.size(), first, num);
	}

	Face *addFace(const uint32_t *indexArray, uint32_t indexCount, uint32_t first, uint32_t num) {
		xaDebugAssert(first < indexCount);
		xaDebugAssert(num <= indexCount - first);
		xaDebugAssert(num > 2);
		if (!canAddFace(indexArray, first, num)) {
			return NULL;
		}
		Face *f = new Face(m_faceArray.size());
		Edge *firstEdge = NULL;
		Edge *last = NULL;
		Edge *current = NULL;
		for (uint32_t i = 0; i < num - 1; i++) {
			current = addEdge(indexArray[first + i], indexArray[first + i + 1]);
			xaAssert(current != NULL && current->face == NULL);
			current->face = f;
			if (last != NULL)
				last->setNext(current);
			else
				firstEdge = current;
			last = current;
		}
		current = addEdge(indexArray[first + num - 1], indexArray[first]);
		xaAssert(current != NULL && current->face == NULL);
		current->face = f;
		last->setNext(current);
		current->setNext(firstEdge);
		f->edge = firstEdge;
		m_faceArray.push_back(f);
		return f;
	}

	// These functions disconnect the given element from the mesh and delete it.

	// @@ We must always disconnect edge pairs simultaneously.
	void disconnect(Edge *edge) {
		xaDebugAssert(edge != NULL);
		// Remove from edge list.
		if ((edge->id & 1) == 0) {
			xaDebugAssert(m_edgeArray[edge->id / 2] == edge);
			m_edgeArray[edge->id / 2] = NULL;
		}
		// Remove edge from map. @@ Store map key inside edge?
		xaDebugAssert(edge->from() != NULL && edge->to() != NULL);
		size_t removed = m_edgeMap.erase(Key(edge->from()->id, edge->to()->id));
		xaDebugAssert(removed == 1);
#ifdef NDEBUG
		removed = 0; // silence unused parameter warning
#endif
		// Disconnect from vertex.
		if (edge->vertex != NULL) {
			if (edge->vertex->edge == edge) {
				if (edge->prev && edge->prev->pair) {
					edge->vertex->edge = edge->prev->pair;
				} else if (edge->pair && edge->pair->next) {
					edge->vertex->edge = edge->pair->next;
				} else {
					edge->vertex->edge = NULL;
					// @@ Remove disconnected vertex?
				}
			}
		}
		// Disconnect from face.
		if (edge->face != NULL) {
			if (edge->face->edge == edge) {
				if (edge->next != NULL && edge->next != edge) {
					edge->face->edge = edge->next;
				} else if (edge->prev != NULL && edge->prev != edge) {
					edge->face->edge = edge->prev;
				} else {
					edge->face->edge = NULL;
					// @@ Remove disconnected face?
				}
			}
		}
		// Disconnect from previous.
		if (edge->prev) {
			if (edge->prev->next == edge) {
				edge->prev->setNext(NULL);
			}
			//edge->setPrev(NULL);
		}
		// Disconnect from next.
		if (edge->next) {
			if (edge->next->prev == edge) {
				edge->next->setPrev(NULL);
			}
			//edge->setNext(NULL);
		}
	}

	void remove(Edge *edge) {
		xaDebugAssert(edge != NULL);
		disconnect(edge);
		delete edge;
	}

	void remove(Vertex *vertex) {
		xaDebugAssert(vertex != NULL);
		// Remove from vertex list.
		m_vertexArray[vertex->id] = NULL;
		// Disconnect from colocals.
		vertex->unlinkColocal();
		// Disconnect from edges.
		if (vertex->edge != NULL) {
			// @@ Removing a connected vertex is asking for trouble...
			if (vertex->edge->vertex == vertex) {
				// @@ Connect edge to a colocal?
				vertex->edge->vertex = NULL;
			}
			vertex->setEdge(NULL);
		}
		delete vertex;
	}

	void remove(Face *face) {
		xaDebugAssert(face != NULL);
		// Remove from face list.
		m_faceArray[face->id] = NULL;
		// Disconnect from edges.
		if (face->edge != NULL) {
			xaDebugAssert(face->edge->face == face);
			face->edge->face = NULL;
			face->edge = NULL;
		}
		delete face;
	}

	// Triangulate in place.
	void triangulate() {
		bool all_triangles = true;
		const uint32_t faceCount = m_faceArray.size();
		for (uint32_t f = 0; f < faceCount; f++) {
			Face *face = m_faceArray[f];
			if (face->edgeCount() != 3) {
				all_triangles = false;
				break;
			}
		}
		if (all_triangles) {
			return;
		}
		// Do not touch vertices, but rebuild edges and faces.
		std::vector<Edge *> edgeArray;
		std::vector<Face *> faceArray;
		std::swap(edgeArray, m_edgeArray);
		std::swap(faceArray, m_faceArray);
		m_edgeMap.clear();
		for (uint32_t f = 0; f < faceCount; f++) {
			Face *face = faceArray[f];
			// Trivial fan-like triangulation.
			const uint32_t v0 = face->edge->vertex->id;
			uint32_t v2, v1 = (uint32_t)-1;
			for (Face::EdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				Edge *edge = it.current();
				v2 = edge->to()->id;
				if (v2 == v0) break;
				if (v1 != -1) addFace(v0, v1, v2);
				v1 = v2;
			}
		}
		xaDebugAssert(m_faceArray.size() > faceCount); // triangle count > face count
		linkBoundary();
		for (size_t i = 0; i < edgeArray.size(); i++)
			delete edgeArray[i];
		for (size_t i = 0; i < faceArray.size(); i++)
			delete faceArray[i];
	}

	/// Link boundary edges once the mesh has been created.
	void linkBoundary() {
		xaPrint("--- Linking boundaries:\n");
		int num = 0;
		// Create boundary edges.
		uint32_t edgeCount = this->edgeCount();
		for (uint32_t e = 0; e < edgeCount; e++) {
			Edge *edge = edgeAt(e);
			if (edge != NULL && edge->pair == NULL) {
				Edge *pair = new Edge(edge->id + 1);
				uint32_t i = edge->from()->id;
				uint32_t j = edge->next->from()->id;
				Key key(j, i);
				xaAssert(m_edgeMap.find(key) == m_edgeMap.end());
				pair->vertex = m_vertexArray[j];
				m_edgeMap[key] = pair;
				edge->pair = pair;
				pair->pair = edge;
				num++;
			}
		}
		// Link boundary edges.
		for (uint32_t e = 0; e < edgeCount; e++) {
			Edge *edge = edgeAt(e);
			if (edge != NULL && edge->pair->face == NULL) {
				linkBoundaryEdge(edge->pair);
			}
		}
		xaPrint("---   %d boundary edges.\n", num);
	}

	/*
	Fixing T-junctions.

	- Find T-junctions. Find  vertices that are on an edge.
		- This test is approximate.
		- Insert edges on a spatial index to speedup queries.
		- Consider only open edges, that is edges that have no pairs.
		- Consider only vertices on boundaries.
	- Close T-junction.
		- Split edge.

	*/
	bool splitBoundaryEdges() // Returns true if any split was made.
	{
		std::vector<Vertex *> boundaryVertices;
		for (uint32_t i = 0; i < m_vertexArray.size(); i++) {
			Vertex *v = m_vertexArray[i];
			if (v->isBoundary()) {
				boundaryVertices.push_back(v);
			}
		}
		xaPrint("Fixing T-junctions:\n");
		int splitCount = 0;
		for (uint32_t v = 0; v < boundaryVertices.size(); v++) {
			Vertex *vertex = boundaryVertices[v];
			Vector3 x0 = vertex->pos;
			// Find edges that this vertex overlaps with.
			for (uint32_t e = 0; e < m_edgeArray.size(); e++) {
				Edge *edge = m_edgeArray[e];
				if (edge != NULL && edge->isBoundary()) {
					if (edge->from() == vertex || edge->to() == vertex) {
						continue;
					}
					Vector3 x1 = edge->from()->pos;
					Vector3 x2 = edge->to()->pos;
					Vector3 v01 = x0 - x1;
					Vector3 v21 = x2 - x1;
					float l = length(v21);
					float d = length(cross(v01, v21)) / l;
					if (isZero(d)) {
						float t = dot(v01, v21) / (l * l);
						if (t > 0.0f + NV_EPSILON && t < 1.0f - NV_EPSILON) {
							xaDebugAssert(equal(lerp(x1, x2, t), x0));
							Vertex *splitVertex = splitBoundaryEdge(edge, t, x0);
							vertex->linkColocal(splitVertex); // @@ Should we do this here?
							splitCount++;
						}
					}
				}
			}
		}
		xaPrint(" - %d edges split.\n", splitCount);
		xaDebugAssert(isValid());
		return splitCount != 0;
	}

	// Vertices
	uint32_t vertexCount() const {
		return m_vertexArray.size();
	}
	const Vertex *vertexAt(int i) const {
		return m_vertexArray[i];
	}
	Vertex *vertexAt(int i) {
		return m_vertexArray[i];
	}

	uint32_t colocalVertexCount() const {
		return m_colocalVertexCount;
	}

	// Faces
	uint32_t faceCount() const {
		return m_faceArray.size();
	}
	const Face *faceAt(int i) const {
		return m_faceArray[i];
	}
	Face *faceAt(int i) {
		return m_faceArray[i];
	}

	// Edges
	uint32_t edgeCount() const {
		return m_edgeArray.size();
	}
	const Edge *edgeAt(int i) const {
		return m_edgeArray[i];
	}
	Edge *edgeAt(int i) {
		return m_edgeArray[i];
	}

	class ConstVertexIterator;

	class VertexIterator {
		friend class ConstVertexIterator;

	public:
		VertexIterator(Mesh *mesh) :
				m_mesh(mesh),
				m_current(0) {}

		virtual void advance() {
			m_current++;
		}
		virtual bool isDone() const {
			return m_current == m_mesh->vertexCount();
		}
		virtual Vertex *current() const {
			return m_mesh->vertexAt(m_current);
		}

	private:
		halfedge::Mesh *m_mesh;
		uint32_t m_current;
	};
	VertexIterator vertices() {
		return VertexIterator(this);
	}

	class ConstVertexIterator {
	public:
		ConstVertexIterator(const Mesh *mesh) :
				m_mesh(mesh),
				m_current(0) {}
		ConstVertexIterator(class VertexIterator &it) :
				m_mesh(it.m_mesh),
				m_current(it.m_current) {}

		virtual void advance() {
			m_current++;
		}
		virtual bool isDone() const {
			return m_current == m_mesh->vertexCount();
		}
		virtual const Vertex *current() const {
			return m_mesh->vertexAt(m_current);
		}

	private:
		const halfedge::Mesh *m_mesh;
		uint32_t m_current;
	};
	ConstVertexIterator vertices() const {
		return ConstVertexIterator(this);
	}

	class ConstFaceIterator;

	class FaceIterator {
		friend class ConstFaceIterator;

	public:
		FaceIterator(Mesh *mesh) :
				m_mesh(mesh),
				m_current(0) {}

		virtual void advance() {
			m_current++;
		}
		virtual bool isDone() const {
			return m_current == m_mesh->faceCount();
		}
		virtual Face *current() const {
			return m_mesh->faceAt(m_current);
		}

	private:
		halfedge::Mesh *m_mesh;
		uint32_t m_current;
	};
	FaceIterator faces() {
		return FaceIterator(this);
	}

	class ConstFaceIterator {
	public:
		ConstFaceIterator(const Mesh *mesh) :
				m_mesh(mesh),
				m_current(0) {}
		ConstFaceIterator(const FaceIterator &it) :
				m_mesh(it.m_mesh),
				m_current(it.m_current) {}

		virtual void advance() {
			m_current++;
		}
		virtual bool isDone() const {
			return m_current == m_mesh->faceCount();
		}
		virtual const Face *current() const {
			return m_mesh->faceAt(m_current);
		}

	private:
		const halfedge::Mesh *m_mesh;
		uint32_t m_current;
	};
	ConstFaceIterator faces() const {
		return ConstFaceIterator(this);
	}

	class ConstEdgeIterator;

	class EdgeIterator {
		friend class ConstEdgeIterator;

	public:
		EdgeIterator(Mesh *mesh) :
				m_mesh(mesh),
				m_current(0) {}

		virtual void advance() {
			m_current++;
		}
		virtual bool isDone() const {
			return m_current == m_mesh->edgeCount();
		}
		virtual Edge *current() const {
			return m_mesh->edgeAt(m_current);
		}

	private:
		halfedge::Mesh *m_mesh;
		uint32_t m_current;
	};
	EdgeIterator edges() {
		return EdgeIterator(this);
	}

	class ConstEdgeIterator {
	public:
		ConstEdgeIterator(const Mesh *mesh) :
				m_mesh(mesh),
				m_current(0) {}
		ConstEdgeIterator(const EdgeIterator &it) :
				m_mesh(it.m_mesh),
				m_current(it.m_current) {}

		virtual void advance() {
			m_current++;
		}
		virtual bool isDone() const {
			return m_current == m_mesh->edgeCount();
		}
		virtual const Edge *current() const {
			return m_mesh->edgeAt(m_current);
		}

	private:
		const halfedge::Mesh *m_mesh;
		uint32_t m_current;
	};
	ConstEdgeIterator edges() const {
		return ConstEdgeIterator(this);
	}

	// @@ Add half-edge iterator.

	bool isValid() const {
		// Make sure all edges are valid.
		const uint32_t edgeCount = m_edgeArray.size();
		for (uint32_t e = 0; e < edgeCount; e++) {
			Edge *edge = m_edgeArray[e];
			if (edge != NULL) {
				if (edge->id != 2 * e) {
					return false;
				}
				if (!edge->isValid()) {
					return false;
				}
				if (edge->pair->id != 2 * e + 1) {
					return false;
				}
				if (!edge->pair->isValid()) {
					return false;
				}
			}
		}
		// @@ Make sure all faces are valid.
		// @@ Make sure all vertices are valid.
		return true;
	}

	// Error status:

	struct ErrorCode {
		enum Enum {
			AlreadyAddedEdge,
			DegenerateColocalEdge,
			DegenerateEdge,
			DuplicateEdge
		};
	};

	mutable ErrorCode::Enum errorCode;
	mutable uint32_t errorIndex0;
	mutable uint32_t errorIndex1;

private:
	// Return true if the face can be added to the manifold mesh.
	bool canAddFace(const std::vector<uint32_t> &indexArray, uint32_t first, uint32_t num) const {
		return canAddFace(indexArray.data(), first, num);
	}

	bool canAddFace(const uint32_t *indexArray, uint32_t first, uint32_t num) const {
		for (uint32_t j = num - 1, i = 0; i < num; j = i++) {
			if (!canAddEdge(indexArray[first + j], indexArray[first + i])) {
				errorIndex0 = indexArray[first + j];
				errorIndex1 = indexArray[first + i];
				return false;
			}
		}
		// We also have to make sure the face does not have any duplicate edge!
		for (uint32_t i = 0; i < num; i++) {
			int i0 = indexArray[first + i + 0];
			int i1 = indexArray[first + (i + 1) % num];
			for (uint32_t j = i + 1; j < num; j++) {
				int j0 = indexArray[first + j + 0];
				int j1 = indexArray[first + (j + 1) % num];
				if (i0 == j0 && i1 == j1) {
					errorCode = ErrorCode::DuplicateEdge;
					errorIndex0 = i0;
					errorIndex1 = i1;
					return false;
				}
			}
		}
		return true;
	}

	// Return true if the edge doesn't exist or doesn't have any adjacent face.
	bool canAddEdge(uint32_t i, uint32_t j) const {
		if (i == j) {
			// Skip degenerate edges.
			errorCode = ErrorCode::DegenerateEdge;
			return false;
		}
		// Same check, but taking into account colocal vertices.
		const Vertex *v0 = vertexAt(i);
		const Vertex *v1 = vertexAt(j);
		for (Vertex::ConstVertexIterator it(v0->colocals()); !it.isDone(); it.advance()) {
			if (it.current() == v1) {
				// Skip degenerate edges.
				errorCode = ErrorCode::DegenerateColocalEdge;
				return false;
			}
		}
		// Make sure edge has not been added yet.
		Edge *edge = findEdge(i, j);
		// We ignore edges that don't have an adjacent face yet, since this face could become the edge's face.
		if (!(edge == NULL || edge->face == NULL)) {
			errorCode = ErrorCode::AlreadyAddedEdge;
			return false;
		}
		return true;
	}

	Edge *addEdge(uint32_t i, uint32_t j) {
		xaAssert(i != j);
		Edge *edge = findEdge(i, j);
		if (edge != NULL) {
			// Edge may already exist, but its face must not be set.
			xaDebugAssert(edge->face == NULL);
			// Nothing else to do!
		} else {
			// Add new edge.
			// Lookup pair.
			Edge *pair = findEdge(j, i);
			if (pair != NULL) {
				// Create edge with same id.
				edge = new Edge(pair->id + 1);
				// Link edge pairs.
				edge->pair = pair;
				pair->pair = edge;
				// @@ I'm not sure this is necessary!
				pair->vertex->setEdge(pair);
			} else {
				// Create edge.
				edge = new Edge(2 * m_edgeArray.size());
				// Add only unpaired edges.
				m_edgeArray.push_back(edge);
			}
			edge->vertex = m_vertexArray[i];
			m_edgeMap[Key(i, j)] = edge;
		}
		// Face and Next are set by addFace.
		return edge;
	}

	/// Find edge, test all colocals.
	Edge *findEdge(uint32_t i, uint32_t j) const {
		Edge *edge = NULL;
		const Vertex *v0 = vertexAt(i);
		const Vertex *v1 = vertexAt(j);
		// Test all colocal pairs.
		for (Vertex::ConstVertexIterator it0(v0->colocals()); !it0.isDone(); it0.advance()) {
			for (Vertex::ConstVertexIterator it1(v1->colocals()); !it1.isDone(); it1.advance()) {
				Key key(it0.current()->id, it1.current()->id);
				if (edge == NULL) {
					auto edgeIt = m_edgeMap.find(key);
					if (edgeIt != m_edgeMap.end())
						edge = (*edgeIt).second;
#if !defined(_DEBUG)
					if (edge != NULL) return edge;
#endif
				} else {
					// Make sure that only one edge is found.
					xaDebugAssert(m_edgeMap.find(key) == m_edgeMap.end());
				}
			}
		}
		return edge;
	}

	/// Link this boundary edge.
	void linkBoundaryEdge(Edge *edge) {
		xaAssert(edge->face == NULL);
		// Make sure next pointer has not been set. @@ We want to be able to relink boundary edges after mesh changes.
		Edge *next = edge;
		while (next->pair->face != NULL) {
			// Get pair prev
			Edge *e = next->pair->next;
			while (e->next != next->pair) {
				e = e->next;
			}
			next = e;
		}
		edge->setNext(next->pair);
		// Adjust vertex edge, so that it's the boundary edge. (required for isBoundary())
		if (edge->vertex->edge != edge) {
			// Multiple boundaries in the same edge.
			edge->vertex->edge = edge;
		}
	}

	Vertex *splitBoundaryEdge(Edge *edge, float t, const Vector3 &pos) {
		/*
		  We want to go from this configuration:

				+   +
				|   ^
		   edge |<->|  pair
				v   |
				+   +

		  To this one:

				+   +
				|   ^
			 e0 |<->| p0
				v   |
		 vertex +   +
				|   ^
			 e1 |<->| p1
				v   |
				+   +

		*/
		Edge *pair = edge->pair;
		// Make sure boundaries are linked.
		xaDebugAssert(pair != NULL);
		// Make sure edge is a boundary edge.
		xaDebugAssert(pair->face == NULL);
		// Add new vertex.
		Vertex *vertex = addVertex(pos);
		vertex->nor = lerp(edge->from()->nor, edge->to()->nor, t);
		vertex->tex = lerp(edge->from()->tex, edge->to()->tex, t);
		disconnect(edge);
		disconnect(pair);
		// Add edges.
		Edge *e0 = addEdge(edge->from()->id, vertex->id);
		Edge *p0 = addEdge(vertex->id, pair->to()->id);
		Edge *e1 = addEdge(vertex->id, edge->to()->id);
		Edge *p1 = addEdge(pair->from()->id, vertex->id);
		// Link edges.
		e0->setNext(e1);
		p1->setNext(p0);
		e0->setPrev(edge->prev);
		e1->setNext(edge->next);
		p1->setPrev(pair->prev);
		p0->setNext(pair->next);
		xaDebugAssert(e0->next == e1);
		xaDebugAssert(e1->prev == e0);
		xaDebugAssert(p1->next == p0);
		xaDebugAssert(p0->prev == p1);
		xaDebugAssert(p0->pair == e0);
		xaDebugAssert(e0->pair == p0);
		xaDebugAssert(p1->pair == e1);
		xaDebugAssert(e1->pair == p1);
		// Link faces.
		e0->face = edge->face;
		e1->face = edge->face;
		// Link vertices.
		edge->from()->setEdge(e0);
		vertex->setEdge(e1);
		delete edge;
		delete pair;
		return vertex;
	}

private:
	std::vector<Vertex *> m_vertexArray;
	std::vector<Edge *> m_edgeArray;
	std::vector<Face *> m_faceArray;

	struct Key {
		Key() {}
		Key(const Key &k) :
				p0(k.p0),
				p1(k.p1) {}
		Key(uint32_t v0, uint32_t v1) :
				p0(v0),
				p1(v1) {}
		void operator=(const Key &k) {
			p0 = k.p0;
			p1 = k.p1;
		}
		bool operator==(const Key &k) const {
			return p0 == k.p0 && p1 == k.p1;
		}

		uint32_t p0;
		uint32_t p1;
	};

	friend struct Hash<Mesh::Key>;
	std::unordered_map<Key, Edge *, Hash<Key>, Equal<Key> > m_edgeMap;
	uint32_t m_colocalVertexCount;
};

class MeshTopology {
public:
	MeshTopology(const Mesh *mesh) {
		buildTopologyInfo(mesh);
	}

	/// Determine if the mesh is connected.
	bool isConnected() const {
		return m_connectedCount == 1;
	}

	/// Determine if the mesh is closed. (Each edge is shared by two faces)
	bool isClosed() const {
		return m_boundaryCount == 0;
	}

	/// Return true if the mesh has the topology of a disk.
	bool isDisk() const {
		return isConnected() && m_boundaryCount == 1 /* && m_eulerNumber == 1*/;
	}

private:
	void buildTopologyInfo(const Mesh *mesh) {
		const uint32_t vertexCount = mesh->colocalVertexCount();
		const uint32_t faceCount = mesh->faceCount();
		const uint32_t edgeCount = mesh->edgeCount();
		xaPrint("--- Building mesh topology:\n");
		std::vector<uint32_t> stack(faceCount);
		BitArray bitFlags(faceCount);
		bitFlags.clearAll();
		// Compute connectivity.
		xaPrint("---   Computing connectivity.\n");
		m_connectedCount = 0;
		for (uint32_t f = 0; f < faceCount; f++) {
			if (bitFlags.bitAt(f) == false) {
				m_connectedCount++;
				stack.push_back(f);
				while (!stack.empty()) {
					const uint32_t top = stack.back();
					xaAssert(top != uint32_t(~0));
					stack.pop_back();
					if (bitFlags.bitAt(top) == false) {
						bitFlags.setBitAt(top);
						const Face *face = mesh->faceAt(top);
						const Edge *firstEdge = face->edge;
						const Edge *edge = firstEdge;
						do {
							const Face *neighborFace = edge->pair->face;
							if (neighborFace != NULL) {
								stack.push_back(neighborFace->id);
							}
							edge = edge->next;
						} while (edge != firstEdge);
					}
				}
			}
		}
		xaAssert(stack.empty());
		xaPrint("---   %d connected components.\n", m_connectedCount);
		// Count boundary loops.
		xaPrint("---   Counting boundary loops.\n");
		m_boundaryCount = 0;
		bitFlags.resize(edgeCount);
		bitFlags.clearAll();
		// Don't forget to link the boundary otherwise this won't work.
		for (uint32_t e = 0; e < edgeCount; e++) {
			const Edge *startEdge = mesh->edgeAt(e);
			if (startEdge != NULL && startEdge->isBoundary() && bitFlags.bitAt(e) == false) {
				xaDebugAssert(startEdge->face != NULL);
				xaDebugAssert(startEdge->pair->face == NULL);
				startEdge = startEdge->pair;
				m_boundaryCount++;
				const Edge *edge = startEdge;
				do {
					bitFlags.setBitAt(edge->id / 2);
					edge = edge->next;
				} while (startEdge != edge);
			}
		}
		xaPrint("---   %d boundary loops found.\n", m_boundaryCount);
		// Compute euler number.
		m_eulerNumber = vertexCount - edgeCount + faceCount;
		xaPrint("---   Euler number: %d.\n", m_eulerNumber);
		// Compute genus. (only valid on closed connected surfaces)
		m_genus = -1;
		if (isClosed() && isConnected()) {
			m_genus = (2 - m_eulerNumber) / 2;
			xaPrint("---   Genus: %d.\n", m_genus);
		}
	}

private:
	///< Number of boundary loops.
	int m_boundaryCount;

	///< Number of connected components.
	int m_connectedCount;

	///< Euler number.
	int m_eulerNumber;

	/// Mesh genus.
	int m_genus;
};

float computeSurfaceArea(const halfedge::Mesh *mesh) {
	float area = 0;
	for (halfedge::Mesh::ConstFaceIterator it(mesh->faces()); !it.isDone(); it.advance()) {
		const halfedge::Face *face = it.current();
		area += face->area();
	}
	xaDebugAssert(area >= 0);
	return area;
}

float computeParametricArea(const halfedge::Mesh *mesh) {
	float area = 0;
	for (halfedge::Mesh::ConstFaceIterator it(mesh->faces()); !it.isDone(); it.advance()) {
		const halfedge::Face *face = it.current();
		area += face->parametricArea();
	}
	return area;
}

uint32_t countMeshTriangles(const Mesh *mesh) {
	const uint32_t faceCount = mesh->faceCount();
	uint32_t triangleCount = 0;
	for (uint32_t f = 0; f < faceCount; f++) {
		const Face *face = mesh->faceAt(f);
		uint32_t edgeCount = face->edgeCount();
		xaDebugAssert(edgeCount > 2);
		triangleCount += edgeCount - 2;
	}
	return triangleCount;
}

Mesh *unifyVertices(const Mesh *inputMesh) {
	Mesh *mesh = new Mesh;
	// Only add the first colocal.
	const uint32_t vertexCount = inputMesh->vertexCount();
	for (uint32_t v = 0; v < vertexCount; v++) {
		const Vertex *vertex = inputMesh->vertexAt(v);
		if (vertex->isFirstColocal()) {
			mesh->addVertex(vertex->pos);
		}
	}
	std::vector<uint32_t> indexArray;
	// Add new faces pointing to first colocals.
	uint32_t faceCount = inputMesh->faceCount();
	for (uint32_t f = 0; f < faceCount; f++) {
		const Face *face = inputMesh->faceAt(f);
		indexArray.clear();
		for (Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
			const Edge *edge = it.current();
			const Vertex *vertex = edge->vertex->firstColocal();
			indexArray.push_back(vertex->id);
		}
		mesh->addFace(indexArray);
	}
	mesh->linkBoundary();
	return mesh;
}

static bool pointInTriangle(const Vector2 &p, const Vector2 &a, const Vector2 &b, const Vector2 &c) {
	return triangleArea(a, b, p) >= 0.00001f &&
		   triangleArea(b, c, p) >= 0.00001f &&
		   triangleArea(c, a, p) >= 0.00001f;
}

// This is doing a simple ear-clipping algorithm that skips invalid triangles. Ideally, we should
// also sort the ears by angle, start with the ones that have the smallest angle and proceed in order.
Mesh *triangulate(const Mesh *inputMesh) {
	Mesh *mesh = new Mesh;
	// Add all vertices.
	const uint32_t vertexCount = inputMesh->vertexCount();
	for (uint32_t v = 0; v < vertexCount; v++) {
		const Vertex *vertex = inputMesh->vertexAt(v);
		mesh->addVertex(vertex->pos);
	}
	std::vector<int> polygonVertices;
	std::vector<float> polygonAngles;
	std::vector<Vector2> polygonPoints;
	const uint32_t faceCount = inputMesh->faceCount();
	for (uint32_t f = 0; f < faceCount; f++) {
		const Face *face = inputMesh->faceAt(f);
		xaDebugAssert(face != NULL);
		const uint32_t edgeCount = face->edgeCount();
		xaDebugAssert(edgeCount >= 3);
		polygonVertices.clear();
		polygonVertices.reserve(edgeCount);
		if (edgeCount == 3) {
			// Simple case for triangles.
			for (Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				const Edge *edge = it.current();
				const Vertex *vertex = edge->vertex;
				polygonVertices.push_back(vertex->id);
			}
			int v0 = polygonVertices[0];
			int v1 = polygonVertices[1];
			int v2 = polygonVertices[2];
			mesh->addFace(v0, v1, v2);
		} else {
			// Build 2D polygon projecting vertices onto normal plane.
			// Faces are not necesarily planar, this is for example the case, when the face comes from filling a hole. In such cases
			// it's much better to use the best fit plane.
			const Vector3 fn = face->normal();
			Basis basis;
			basis.buildFrameForDirection(fn);
			polygonPoints.clear();
			polygonPoints.reserve(edgeCount);
			polygonAngles.clear();
			polygonAngles.reserve(edgeCount);
			for (Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				const Edge *edge = it.current();
				const Vertex *vertex = edge->vertex;
				polygonVertices.push_back(vertex->id);
				Vector2 p;
				p.x = dot(basis.tangent, vertex->pos);
				p.y = dot(basis.bitangent, vertex->pos);
				polygonPoints.push_back(p);
			}
			polygonAngles.resize(edgeCount);
			while (polygonVertices.size() > 2) {
				uint32_t size = polygonVertices.size();
				// Update polygon angles. @@ Update only those that have changed.
				float minAngle = 2 * PI;
				uint32_t bestEar = 0; // Use first one if none of them is valid.
				bool bestIsValid = false;
				for (uint32_t i = 0; i < size; i++) {
					uint32_t i0 = i;
					uint32_t i1 = (i + 1) % size; // Use Sean's polygon interation trick.
					uint32_t i2 = (i + 2) % size;
					Vector2 p0 = polygonPoints[i0];
					Vector2 p1 = polygonPoints[i1];
					Vector2 p2 = polygonPoints[i2];

					bool degenerate = distance(p0, p1) < NV_EPSILON || distance(p0, p2) < NV_EPSILON || distance(p1, p2) < NV_EPSILON;
					if (degenerate) {
						continue;
					}

					float d = clamp(dot(p0 - p1, p2 - p1) / (length(p0 - p1) * length(p2 - p1)), -1.0f, 1.0f);
					float angle = acosf(d);
					float area = triangleArea(p0, p1, p2);
					if (area < 0.0f) angle = 2.0f * PI - angle;
					polygonAngles[i1] = angle;
					if (angle < minAngle || !bestIsValid) {
						// Make sure this is a valid ear, if not, skip this point.
						bool valid = true;
						for (uint32_t j = 0; j < size; j++) {
							if (j == i0 || j == i1 || j == i2) continue;
							Vector2 p = polygonPoints[j];
							if (pointInTriangle(p, p0, p1, p2)) {
								valid = false;
								break;
							}
						}
						if (valid || !bestIsValid) {
							minAngle = angle;
							bestEar = i1;
							bestIsValid = valid;
						}
					}
				}
				if (!bestIsValid)
					break;

				xaDebugAssert(minAngle <= 2 * PI);
				// Clip best ear:
				uint32_t i0 = (bestEar + size - 1) % size;
				uint32_t i1 = (bestEar + 0) % size;
				uint32_t i2 = (bestEar + 1) % size;
				int v0 = polygonVertices[i0];
				int v1 = polygonVertices[i1];
				int v2 = polygonVertices[i2];
				mesh->addFace(v0, v1, v2);
				polygonVertices.erase(polygonVertices.begin() + i1);
				polygonPoints.erase(polygonPoints.begin() + i1);
				polygonAngles.erase(polygonAngles.begin() + i1);
			}
		}
	}
	mesh->linkBoundary();
	return mesh;
}

} //  namespace halfedge

/// Mersenne twister random number generator.
class MTRand {
public:
	enum time_e { Time };
	enum { N = 624 }; // length of state vector
	enum { M = 397 };

	/// Constructor that uses the current time as the seed.
	MTRand(time_e) {
		seed((uint32_t)time(NULL));
	}

	/// Constructor that uses the given seed.
	MTRand(uint32_t s = 0) {
		seed(s);
	}

	/// Provide a new seed.
	void seed(uint32_t s) {
		initialize(s);
		reload();
	}

	/// Get a random number between 0 - 65536.
	uint32_t get() {
		// Pull a 32-bit integer from the generator state
		// Every other access function simply transforms the numbers extracted here
		if (left == 0) {
			reload();
		}
		left--;
		uint32_t s1;
		s1 = *next++;
		s1 ^= (s1 >> 11);
		s1 ^= (s1 << 7) & 0x9d2c5680U;
		s1 ^= (s1 << 15) & 0xefc60000U;
		return (s1 ^ (s1 >> 18));
	};

	/// Get a random number on [0, max] interval.
	uint32_t getRange(uint32_t max) {
		if (max == 0) return 0;
		if (max == NV_UINT32_MAX) return get();
		const uint32_t np2 = nextPowerOfTwo(max + 1); // @@ This fails if max == NV_UINT32_MAX
		const uint32_t mask = np2 - 1;
		uint32_t n;
		do {
			n = get() & mask;
		} while (n > max);
		return n;
	}

private:
	void initialize(uint32_t seed) {
		// Initialize generator state with seed
		// See Knuth TAOCP Vol 2, 3rd Ed, p.106 for multiplier.
		// In previous versions, most significant bits (MSBs) of the seed affect
		// only MSBs of the state array.  Modified 9 Jan 2002 by Makoto Matsumoto.
		uint32_t *s = state;
		uint32_t *r = state;
		int i = 1;
		*s++ = seed & 0xffffffffUL;
		for (; i < N; ++i) {
			*s++ = (1812433253UL * (*r ^ (*r >> 30)) + i) & 0xffffffffUL;
			r++;
		}
	}

	void reload() {
		// Generate N new values in state
		// Made clearer and faster by Matthew Bellew (matthew.bellew@home.com)
		uint32_t *p = state;
		int i;
		for (i = N - M; i--; ++p)
			*p = twist(p[M], p[0], p[1]);
		for (i = M; --i; ++p)
			*p = twist(p[M - N], p[0], p[1]);
		*p = twist(p[M - N], p[0], state[0]);
		left = N, next = state;
	}

	uint32_t hiBit(uint32_t u) const {
		return u & 0x80000000U;
	}
	uint32_t loBit(uint32_t u) const {
		return u & 0x00000001U;
	}
	uint32_t loBits(uint32_t u) const {
		return u & 0x7fffffffU;
	}
	uint32_t mixBits(uint32_t u, uint32_t v) const {
		return hiBit(u) | loBits(v);
	}
	uint32_t twist(uint32_t m, uint32_t s0, uint32_t s1) const {
		return m ^ (mixBits(s0, s1) >> 1) ^ ((~loBit(s1) + 1) & 0x9908b0dfU);
	}

	uint32_t state[N]; // internal state
	uint32_t *next; // next value to get from state
	int left; // number of values left before reload needed
};

namespace morton {
// Code from ryg:
// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/

// Inverse of part1By1 - "delete" all odd-indexed bits
uint32_t compact1By1(uint32_t x) {
	x &= 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
	return x;
}

// Inverse of part1By2 - "delete" all bits not at positions divisible by 3
uint32_t compact1By2(uint32_t x) {
	x &= 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x ^ (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}

uint32_t decodeMorton2X(uint32_t code) {
	return compact1By1(code >> 0);
}

uint32_t decodeMorton2Y(uint32_t code) {
	return compact1By1(code >> 1);
}

uint32_t decodeMorton3X(uint32_t code) {
	return compact1By2(code >> 0);
}

uint32_t decodeMorton3Y(uint32_t code) {
	return compact1By2(code >> 1);
}

uint32_t decodeMorton3Z(uint32_t code) {
	return compact1By2(code >> 2);
}
} // namespace morton

// A simple, dynamic proximity grid based on Jon's code.
// Instead of storing pointers here I store indices.
struct ProximityGrid {
	void init(const Box &box, uint32_t count) {
		cellArray.clear();
		// Determine grid size.
		float cellWidth;
		Vector3 diagonal = box.extents() * 2.f;
		float volume = box.volume();
		if (equal(volume, 0)) {
			// Degenerate box, treat like a quad.
			Vector2 quad;
			if (diagonal.x < diagonal.y && diagonal.x < diagonal.z) {
				quad.x = diagonal.y;
				quad.y = diagonal.z;
			} else if (diagonal.y < diagonal.x && diagonal.y < diagonal.z) {
				quad.x = diagonal.x;
				quad.y = diagonal.z;
			} else {
				quad.x = diagonal.x;
				quad.y = diagonal.y;
			}
			float cellArea = quad.x * quad.y / count;
			cellWidth = sqrtf(cellArea); // pow(cellArea, 1.0f / 2.0f);
		} else {
			// Ideally we want one cell per point.
			float cellVolume = volume / count;
			cellWidth = powf(cellVolume, 1.0f / 3.0f);
		}
		xaDebugAssert(cellWidth != 0);
		sx = std::max(1, ftoi_ceil(diagonal.x / cellWidth));
		sy = std::max(1, ftoi_ceil(diagonal.y / cellWidth));
		sz = std::max(1, ftoi_ceil(diagonal.z / cellWidth));
		invCellSize.x = float(sx) / diagonal.x;
		invCellSize.y = float(sy) / diagonal.y;
		invCellSize.z = float(sz) / diagonal.z;
		cellArray.resize(sx * sy * sz);
		corner = box.minCorner; // @@ Align grid better?
	}

	int index_x(float x) const {
		return clamp(ftoi_floor((x - corner.x) * invCellSize.x), 0, sx - 1);
	}

	int index_y(float y) const {
		return clamp(ftoi_floor((y - corner.y) * invCellSize.y), 0, sy - 1);
	}

	int index_z(float z) const {
		return clamp(ftoi_floor((z - corner.z) * invCellSize.z), 0, sz - 1);
	}

	int index(int x, int y, int z) const {
		xaDebugAssert(x >= 0 && x < sx);
		xaDebugAssert(y >= 0 && y < sy);
		xaDebugAssert(z >= 0 && z < sz);
		int idx = (z * sy + y) * sx + x;
		xaDebugAssert(idx >= 0 && uint32_t(idx) < cellArray.size());
		return idx;
	}

	uint32_t mortonCount() const {
		uint64_t s = uint64_t(max3(sx, sy, sz));
		s = nextPowerOfTwo(s);
		if (s > 1024) {
			return uint32_t(s * s * min3(sx, sy, sz));
		}
		return uint32_t(s * s * s);
	}

	int mortonIndex(uint32_t code) const {
		uint32_t x, y, z;
		uint32_t s = uint32_t(max3(sx, sy, sz));
		if (s > 1024) {
			// Use layered two-dimensional morton order.
			s = nextPowerOfTwo(s);
			uint32_t layer = code / (s * s);
			code = code % (s * s);
			uint32_t layer_count = uint32_t(min3(sx, sy, sz));
			if (sx == (int)layer_count) {
				x = layer;
				y = morton::decodeMorton2X(code);
				z = morton::decodeMorton2Y(code);
			} else if (sy == (int)layer_count) {
				x = morton::decodeMorton2Y(code);
				y = layer;
				z = morton::decodeMorton2X(code);
			} else { /*if (sz == layer_count)*/
				x = morton::decodeMorton2X(code);
				y = morton::decodeMorton2Y(code);
				z = layer;
			}
		} else {
			x = morton::decodeMorton3X(code);
			y = morton::decodeMorton3Y(code);
			z = morton::decodeMorton3Z(code);
		}
		if (x >= uint32_t(sx) || y >= uint32_t(sy) || z >= uint32_t(sz)) {
			return -1;
		}
		return index(x, y, z);
	}

	void add(const Vector3 &pos, uint32_t key) {
		int x = index_x(pos.x);
		int y = index_y(pos.y);
		int z = index_z(pos.z);
		uint32_t idx = index(x, y, z);
		cellArray[idx].indexArray.push_back(key);
	}

	// Gather all points inside the given sphere.
	// Radius is assumed to be small, so we don't bother culling the cells.
	void gather(const Vector3 &position, float radius, std::vector<uint32_t> &indexArray) {
		int x0 = index_x(position.x - radius);
		int x1 = index_x(position.x + radius);
		int y0 = index_y(position.y - radius);
		int y1 = index_y(position.y + radius);
		int z0 = index_z(position.z - radius);
		int z1 = index_z(position.z + radius);
		for (int z = z0; z <= z1; z++) {
			for (int y = y0; y <= y1; y++) {
				for (int x = x0; x <= x1; x++) {
					int idx = index(x, y, z);
					indexArray.insert(indexArray.begin(), cellArray[idx].indexArray.begin(), cellArray[idx].indexArray.end());
				}
			}
		}
	}

	struct Cell {
		std::vector<uint32_t> indexArray;
	};

	std::vector<Cell> cellArray;

	Vector3 corner;
	Vector3 invCellSize;
	int sx, sy, sz;
};

// Based on Pierre Terdiman's and Michael Herf's source code.
// http://www.codercorner.com/RadixSortRevisited.htm
// http://www.stereopsis.com/radix.html
class RadixSort {
public:
	RadixSort() :
			m_size(0),
			m_ranks(NULL),
			m_ranks2(NULL),
			m_validRanks(false) {}
	~RadixSort() {
		// Release everything
		free(m_ranks2);
		free(m_ranks);
	}

	RadixSort &sort(const float *input, uint32_t count) {
		if (input == NULL || count == 0) return *this;
		// Resize lists if needed
		if (count != m_size) {
			if (count > m_size) {
				m_ranks2 = (uint32_t *)realloc(m_ranks2, sizeof(uint32_t) * count);
				m_ranks = (uint32_t *)realloc(m_ranks, sizeof(uint32_t) * count);
			}
			m_size = count;
			m_validRanks = false;
		}
		if (count < 32) {
			insertionSort(input, count);
		} else {
			// @@ Avoid touching the input multiple times.
			for (uint32_t i = 0; i < count; i++) {
				FloatFlip((uint32_t &)input[i]);
			}
			radixSort<uint32_t>((const uint32_t *)input, count);
			for (uint32_t i = 0; i < count; i++) {
				IFloatFlip((uint32_t &)input[i]);
			}
		}
		return *this;
	}

	RadixSort &sort(const std::vector<float> &input) {
		return sort(input.data(), input.size());
	}

	// Access to results. m_ranks is a list of indices in sorted order, i.e. in the order you may further process your data
	const uint32_t *ranks() const {
		xaDebugAssert(m_validRanks);
		return m_ranks;
	}
	uint32_t *ranks() {
		xaDebugAssert(m_validRanks);
		return m_ranks;
	}

private:
	uint32_t m_size;
	uint32_t *m_ranks;
	uint32_t *m_ranks2;
	bool m_validRanks;

	void FloatFlip(uint32_t &f) {
		int32_t mask = (int32_t(f) >> 31) | 0x80000000; // Warren Hunt, Manchor Ko.
		f ^= mask;
	}

	void IFloatFlip(uint32_t &f) {
		uint32_t mask = ((f >> 31) - 1) | 0x80000000; // Michael Herf.
		f ^= mask;
	}

	template <typename T>
	void createHistograms(const T *buffer, uint32_t count, uint32_t *histogram) {
		const uint32_t bucketCount = sizeof(T); // (8 * sizeof(T)) / log2(radix)
		// Init bucket pointers.
		uint32_t *h[bucketCount];
		for (uint32_t i = 0; i < bucketCount; i++) {
			h[i] = histogram + 256 * i;
		}
		// Clear histograms.
		memset(histogram, 0, 256 * bucketCount * sizeof(uint32_t));
		// @@ Add support for signed integers.
		// Build histograms.
		const uint8_t *p = (const uint8_t *)buffer; // @@ Does this break aliasing rules?
		const uint8_t *pe = p + count * sizeof(T);
		while (p != pe) {
			h[0][*p++]++, h[1][*p++]++, h[2][*p++]++, h[3][*p++]++;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif
			if (bucketCount == 8) h[4][*p++]++, h[5][*p++]++, h[6][*p++]++, h[7][*p++]++;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
		}
	}

	template <typename T>
	void insertionSort(const T *input, uint32_t count) {
		if (!m_validRanks) {
			m_ranks[0] = 0;
			for (uint32_t i = 1; i != count; ++i) {
				int rank = m_ranks[i] = i;
				uint32_t j = i;
				while (j != 0 && input[rank] < input[m_ranks[j - 1]]) {
					m_ranks[j] = m_ranks[j - 1];
					--j;
				}
				if (i != j) {
					m_ranks[j] = rank;
				}
			}
			m_validRanks = true;
		} else {
			for (uint32_t i = 1; i != count; ++i) {
				int rank = m_ranks[i];
				uint32_t j = i;
				while (j != 0 && input[rank] < input[m_ranks[j - 1]]) {
					m_ranks[j] = m_ranks[j - 1];
					--j;
				}
				if (i != j) {
					m_ranks[j] = rank;
				}
			}
		}
	}

	template <typename T>
	void radixSort(const T *input, uint32_t count) {
		const uint32_t P = sizeof(T); // pass count
		// Allocate histograms & offsets on the stack
		uint32_t histogram[256 * P];
		uint32_t *link[256];
		createHistograms(input, count, histogram);
		// Radix sort, j is the pass number (0=LSB, P=MSB)
		for (uint32_t j = 0; j < P; j++) {
			// Pointer to this bucket.
			const uint32_t *h = &histogram[j * 256];
			const uint8_t *inputBytes = (const uint8_t *)input; // @@ Is this aliasing legal?
			inputBytes += j;
			if (h[inputBytes[0]] == count) {
				// Skip this pass, all values are the same.
				continue;
			}
			// Create offsets
			link[0] = m_ranks2;
			for (uint32_t i = 1; i < 256; i++)
				link[i] = link[i - 1] + h[i - 1];
			// Perform Radix Sort
			if (!m_validRanks) {
				for (uint32_t i = 0; i < count; i++) {
					*link[inputBytes[i * P]]++ = i;
				}
				m_validRanks = true;
			} else {
				for (uint32_t i = 0; i < count; i++) {
					const uint32_t idx = m_ranks[i];
					*link[inputBytes[idx * P]]++ = idx;
				}
			}
			// Swap pointers for next pass. Valid indices - the most recent ones - are in m_ranks after the swap.
			std::swap(m_ranks, m_ranks2);
		}
		// All values were equal, generate linear ranks.
		if (!m_validRanks) {
			for (uint32_t i = 0; i < count; i++) {
				m_ranks[i] = i;
			}
			m_validRanks = true;
		}
	}
};

namespace raster {
class ClippedTriangle {
public:
	ClippedTriangle(Vector2::Arg a, Vector2::Arg b, Vector2::Arg c) {
		m_numVertices = 3;
		m_activeVertexBuffer = 0;
		m_verticesA[0] = a;
		m_verticesA[1] = b;
		m_verticesA[2] = c;
		m_vertexBuffers[0] = m_verticesA;
		m_vertexBuffers[1] = m_verticesB;
	}

	uint32_t vertexCount() {
		return m_numVertices;
	}

	const Vector2 *vertices() {
		return m_vertexBuffers[m_activeVertexBuffer];
	}

	void clipHorizontalPlane(float offset, float clipdirection) {
		Vector2 *v = m_vertexBuffers[m_activeVertexBuffer];
		m_activeVertexBuffer ^= 1;
		Vector2 *v2 = m_vertexBuffers[m_activeVertexBuffer];
		v[m_numVertices] = v[0];
		float dy2, dy1 = offset - v[0].y;
		int dy2in, dy1in = clipdirection * dy1 >= 0;
		uint32_t p = 0;
		for (uint32_t k = 0; k < m_numVertices; k++) {
			dy2 = offset - v[k + 1].y;
			dy2in = clipdirection * dy2 >= 0;
			if (dy1in) v2[p++] = v[k];
			if (dy1in + dy2in == 1) { // not both in/out
				float dx = v[k + 1].x - v[k].x;
				float dy = v[k + 1].y - v[k].y;
				v2[p++] = Vector2(v[k].x + dy1 * (dx / dy), offset);
			}
			dy1 = dy2;
			dy1in = dy2in;
		}
		m_numVertices = p;
		//for (uint32_t k=0; k<m_numVertices; k++) printf("(%f, %f)\n", v2[k].x, v2[k].y); printf("\n");
	}

	void clipVerticalPlane(float offset, float clipdirection) {
		Vector2 *v = m_vertexBuffers[m_activeVertexBuffer];
		m_activeVertexBuffer ^= 1;
		Vector2 *v2 = m_vertexBuffers[m_activeVertexBuffer];
		v[m_numVertices] = v[0];
		float dx2, dx1 = offset - v[0].x;
		int dx2in, dx1in = clipdirection * dx1 >= 0;
		uint32_t p = 0;
		for (uint32_t k = 0; k < m_numVertices; k++) {
			dx2 = offset - v[k + 1].x;
			dx2in = clipdirection * dx2 >= 0;
			if (dx1in) v2[p++] = v[k];
			if (dx1in + dx2in == 1) { // not both in/out
				float dx = v[k + 1].x - v[k].x;
				float dy = v[k + 1].y - v[k].y;
				v2[p++] = Vector2(offset, v[k].y + dx1 * (dy / dx));
			}
			dx1 = dx2;
			dx1in = dx2in;
		}
		m_numVertices = p;
	}

	void computeAreaCentroid() {
		Vector2 *v = m_vertexBuffers[m_activeVertexBuffer];
		v[m_numVertices] = v[0];
		m_area = 0;
		float centroidx = 0, centroidy = 0;
		for (uint32_t k = 0; k < m_numVertices; k++) {
			// http://local.wasp.uwa.edu.au/~pbourke/geometry/polyarea/
			float f = v[k].x * v[k + 1].y - v[k + 1].x * v[k].y;
			m_area += f;
			centroidx += f * (v[k].x + v[k + 1].x);
			centroidy += f * (v[k].y + v[k + 1].y);
		}
		m_area = 0.5f * fabsf(m_area);
		if (m_area == 0) {
			m_centroid = Vector2(0.0f);
		} else {
			m_centroid = Vector2(centroidx / (6 * m_area), centroidy / (6 * m_area));
		}
	}

	void clipAABox(float x0, float y0, float x1, float y1) {
		clipVerticalPlane(x0, -1);
		clipHorizontalPlane(y0, -1);
		clipVerticalPlane(x1, 1);
		clipHorizontalPlane(y1, 1);
		computeAreaCentroid();
	}

	Vector2 centroid() {
		return m_centroid;
	}

	float area() {
		return m_area;
	}

private:
	Vector2 m_verticesA[7 + 1];
	Vector2 m_verticesB[7 + 1];
	Vector2 *m_vertexBuffers[2];
	uint32_t m_numVertices;
	uint32_t m_activeVertexBuffer;
	float m_area;
	Vector2 m_centroid;
};

/// A callback to sample the environment. Return false to terminate rasterization.
typedef bool (*SamplingCallback)(void *param, int x, int y, Vector3::Arg bar, Vector3::Arg dx, Vector3::Arg dy, float coverage);

/// A triangle for rasterization.
struct Triangle {
	Triangle(Vector2::Arg v0, Vector2::Arg v1, Vector2::Arg v2, Vector3::Arg t0, Vector3::Arg t1, Vector3::Arg t2) {
		// Init vertices.
		this->v1 = v0;
		this->v2 = v2;
		this->v3 = v1;
		// Set barycentric coordinates.
		this->t1 = t0;
		this->t2 = t2;
		this->t3 = t1;
		// make sure every triangle is front facing.
		flipBackface();
		// Compute deltas.
		valid = computeDeltas();
		computeUnitInwardNormals();
	}

	/// Compute texture space deltas.
	/// This method takes two edge vectors that form a basis, determines the
	/// coordinates of the canonic vectors in that basis, and computes the
	/// texture gradient that corresponds to those vectors.
	bool computeDeltas() {
		Vector2 e0 = v3 - v1;
		Vector2 e1 = v2 - v1;
		Vector3 de0 = t3 - t1;
		Vector3 de1 = t2 - t1;
		float denom = 1.0f / (e0.y * e1.x - e1.y * e0.x);
		if (!std::isfinite(denom)) {
			return false;
		}
		float lambda1 = -e1.y * denom;
		float lambda2 = e0.y * denom;
		float lambda3 = e1.x * denom;
		float lambda4 = -e0.x * denom;
		dx = de0 * lambda1 + de1 * lambda2;
		dy = de0 * lambda3 + de1 * lambda4;
		return true;
	}

	bool draw(const Vector2 &extents, bool enableScissors, SamplingCallback cb, void *param) {
		// 28.4 fixed-point coordinates
		const int Y1 = ftoi_round(16.0f * v1.y);
		const int Y2 = ftoi_round(16.0f * v2.y);
		const int Y3 = ftoi_round(16.0f * v3.y);
		const int X1 = ftoi_round(16.0f * v1.x);
		const int X2 = ftoi_round(16.0f * v2.x);
		const int X3 = ftoi_round(16.0f * v3.x);
		// Deltas
		const int DX12 = X1 - X2;
		const int DX23 = X2 - X3;
		const int DX31 = X3 - X1;
		const int DY12 = Y1 - Y2;
		const int DY23 = Y2 - Y3;
		const int DY31 = Y3 - Y1;
		// Fixed-point deltas
		const int FDX12 = DX12 << 4;
		const int FDX23 = DX23 << 4;
		const int FDX31 = DX31 << 4;
		const int FDY12 = DY12 << 4;
		const int FDY23 = DY23 << 4;
		const int FDY31 = DY31 << 4;
		int minx, miny, maxx, maxy;
		if (enableScissors) {
			int frustumX0 = 0 << 4;
			int frustumY0 = 0 << 4;
			int frustumX1 = (int)extents.x << 4;
			int frustumY1 = (int)extents.y << 4;
			// Bounding rectangle
			minx = (std::max(min3(X1, X2, X3), frustumX0) + 0xF) >> 4;
			miny = (std::max(min3(Y1, Y2, Y3), frustumY0) + 0xF) >> 4;
			maxx = (std::min(max3(X1, X2, X3), frustumX1) + 0xF) >> 4;
			maxy = (std::min(max3(Y1, Y2, Y3), frustumY1) + 0xF) >> 4;
		} else {
			// Bounding rectangle
			minx = (min3(X1, X2, X3) + 0xF) >> 4;
			miny = (min3(Y1, Y2, Y3) + 0xF) >> 4;
			maxx = (max3(X1, X2, X3) + 0xF) >> 4;
			maxy = (max3(Y1, Y2, Y3) + 0xF) >> 4;
		}
		// Block size, standard 8x8 (must be power of two)
		const int q = 8;
		// @@ This won't work when minx,miny are negative. This code path is not used. Leaving as is for now.
		xaAssert(minx >= 0);
		xaAssert(miny >= 0);
		// Start in corner of 8x8 block
		minx &= ~(q - 1);
		miny &= ~(q - 1);
		// Half-edge constants
		int C1 = DY12 * X1 - DX12 * Y1;
		int C2 = DY23 * X2 - DX23 * Y2;
		int C3 = DY31 * X3 - DX31 * Y3;
		// Correct for fill convention
		if (DY12 < 0 || (DY12 == 0 && DX12 > 0)) C1++;
		if (DY23 < 0 || (DY23 == 0 && DX23 > 0)) C2++;
		if (DY31 < 0 || (DY31 == 0 && DX31 > 0)) C3++;
		// Loop through blocks
		for (int y = miny; y < maxy; y += q) {
			for (int x = minx; x < maxx; x += q) {
				// Corners of block
				int x0 = x << 4;
				int x1 = (x + q - 1) << 4;
				int y0 = y << 4;
				int y1 = (y + q - 1) << 4;
				// Evaluate half-space functions
				bool a00 = C1 + DX12 * y0 - DY12 * x0 > 0;
				bool a10 = C1 + DX12 * y0 - DY12 * x1 > 0;
				bool a01 = C1 + DX12 * y1 - DY12 * x0 > 0;
				bool a11 = C1 + DX12 * y1 - DY12 * x1 > 0;
				int a = (a00 << 0) | (a10 << 1) | (a01 << 2) | (a11 << 3);
				bool b00 = C2 + DX23 * y0 - DY23 * x0 > 0;
				bool b10 = C2 + DX23 * y0 - DY23 * x1 > 0;
				bool b01 = C2 + DX23 * y1 - DY23 * x0 > 0;
				bool b11 = C2 + DX23 * y1 - DY23 * x1 > 0;
				int b = (b00 << 0) | (b10 << 1) | (b01 << 2) | (b11 << 3);
				bool c00 = C3 + DX31 * y0 - DY31 * x0 > 0;
				bool c10 = C3 + DX31 * y0 - DY31 * x1 > 0;
				bool c01 = C3 + DX31 * y1 - DY31 * x0 > 0;
				bool c11 = C3 + DX31 * y1 - DY31 * x1 > 0;
				int c = (c00 << 0) | (c10 << 1) | (c01 << 2) | (c11 << 3);
				// Skip block when outside an edge
				if (a == 0x0 || b == 0x0 || c == 0x0) continue;
				// Accept whole block when totally covered
				if (a == 0xF && b == 0xF && c == 0xF) {
					Vector3 texRow = t1 + dy * (y0 - v1.y) + dx * (x0 - v1.x);
					for (int iy = y; iy < y + q; iy++) {
						Vector3 tex = texRow;
						for (int ix = x; ix < x + q; ix++) {
							//Vector3 tex = t1 + dx * (ix - v1.x) + dy * (iy - v1.y);
							if (!cb(param, ix, iy, tex, dx, dy, 1.0)) {
								// early out.
								return false;
							}
							tex += dx;
						}
						texRow += dy;
					}
				} else { // Partially covered block
					int CY1 = C1 + DX12 * y0 - DY12 * x0;
					int CY2 = C2 + DX23 * y0 - DY23 * x0;
					int CY3 = C3 + DX31 * y0 - DY31 * x0;
					Vector3 texRow = t1 + dy * (y0 - v1.y) + dx * (x0 - v1.x);
					for (int iy = y; iy < y + q; iy++) {
						int CX1 = CY1;
						int CX2 = CY2;
						int CX3 = CY3;
						Vector3 tex = texRow;
						for (int ix = x; ix < x + q; ix++) {
							if (CX1 > 0 && CX2 > 0 && CX3 > 0) {
								if (!cb(param, ix, iy, tex, dx, dy, 1.0)) {
									// early out.
									return false;
								}
							}
							CX1 -= FDY12;
							CX2 -= FDY23;
							CX3 -= FDY31;
							tex += dx;
						}
						CY1 += FDX12;
						CY2 += FDX23;
						CY3 += FDX31;
						texRow += dy;
					}
				}
			}
		}
		return true;
	}

	// extents has to be multiple of BK_SIZE!!
	bool drawAA(const Vector2 &extents, bool enableScissors, SamplingCallback cb, void *param) {
		const float PX_INSIDE = 1.0f / sqrt(2.0f);
		const float PX_OUTSIDE = -1.0f / sqrt(2.0f);
		const float BK_SIZE = 8;
		const float BK_INSIDE = sqrt(BK_SIZE * BK_SIZE / 2.0f);
		const float BK_OUTSIDE = -sqrt(BK_SIZE * BK_SIZE / 2.0f);

		float minx, miny, maxx, maxy;
		if (enableScissors) {
			// Bounding rectangle
			minx = floorf(std::max(min3(v1.x, v2.x, v3.x), 0.0f));
			miny = floorf(std::max(min3(v1.y, v2.y, v3.y), 0.0f));
			maxx = ceilf(std::min(max3(v1.x, v2.x, v3.x), extents.x - 1.0f));
			maxy = ceilf(std::min(max3(v1.y, v2.y, v3.y), extents.y - 1.0f));
		} else {
			// Bounding rectangle
			minx = floorf(min3(v1.x, v2.x, v3.x));
			miny = floorf(min3(v1.y, v2.y, v3.y));
			maxx = ceilf(max3(v1.x, v2.x, v3.x));
			maxy = ceilf(max3(v1.y, v2.y, v3.y));
		}
		// There's no reason to align the blocks to the viewport, instead we align them to the origin of the triangle bounds.
		minx = floorf(minx);
		miny = floorf(miny);
		//minx = (float)(((int)minx) & (~((int)BK_SIZE - 1))); // align to blocksize (we don't need to worry about blocks partially out of viewport)
		//miny = (float)(((int)miny) & (~((int)BK_SIZE - 1)));
		minx += 0.5;
		miny += 0.5; // sampling at texel centers!
		maxx += 0.5;
		maxy += 0.5;
		// Half-edge constants
		float C1 = n1.x * (-v1.x) + n1.y * (-v1.y);
		float C2 = n2.x * (-v2.x) + n2.y * (-v2.y);
		float C3 = n3.x * (-v3.x) + n3.y * (-v3.y);
		// Loop through blocks
		for (float y0 = miny; y0 <= maxy; y0 += BK_SIZE) {
			for (float x0 = minx; x0 <= maxx; x0 += BK_SIZE) {
				// Corners of block
				float xc = (x0 + (BK_SIZE - 1) / 2.0f);
				float yc = (y0 + (BK_SIZE - 1) / 2.0f);
				// Evaluate half-space functions
				float aC = C1 + n1.x * xc + n1.y * yc;
				float bC = C2 + n2.x * xc + n2.y * yc;
				float cC = C3 + n3.x * xc + n3.y * yc;
				// Skip block when outside an edge
				if ((aC <= BK_OUTSIDE) || (bC <= BK_OUTSIDE) || (cC <= BK_OUTSIDE)) continue;
				// Accept whole block when totally covered
				if ((aC >= BK_INSIDE) && (bC >= BK_INSIDE) && (cC >= BK_INSIDE)) {
					Vector3 texRow = t1 + dy * (y0 - v1.y) + dx * (x0 - v1.x);
					for (float y = y0; y < y0 + BK_SIZE; y++) {
						Vector3 tex = texRow;
						for (float x = x0; x < x0 + BK_SIZE; x++) {
							if (!cb(param, (int)x, (int)y, tex, dx, dy, 1.0f)) {
								return false;
							}
							tex += dx;
						}
						texRow += dy;
					}
				} else { // Partially covered block
					float CY1 = C1 + n1.x * x0 + n1.y * y0;
					float CY2 = C2 + n2.x * x0 + n2.y * y0;
					float CY3 = C3 + n3.x * x0 + n3.y * y0;
					Vector3 texRow = t1 + dy * (y0 - v1.y) + dx * (x0 - v1.x);
					for (float y = y0; y < y0 + BK_SIZE; y++) { // @@ This is not clipping to scissor rectangle correctly.
						float CX1 = CY1;
						float CX2 = CY2;
						float CX3 = CY3;
						Vector3 tex = texRow;
						for (float x = x0; x < x0 + BK_SIZE; x++) { // @@ This is not clipping to scissor rectangle correctly.
							if (CX1 >= PX_INSIDE && CX2 >= PX_INSIDE && CX3 >= PX_INSIDE) {
								// pixel completely covered
								Vector3 tex2 = t1 + dx * (x - v1.x) + dy * (y - v1.y);
								if (!cb(param, (int)x, (int)y, tex2, dx, dy, 1.0f)) {
									return false;
								}
							} else if ((CX1 >= PX_OUTSIDE) && (CX2 >= PX_OUTSIDE) && (CX3 >= PX_OUTSIDE)) {
								// triangle partially covers pixel. do clipping.
								ClippedTriangle ct(v1 - Vector2(x, y), v2 - Vector2(x, y), v3 - Vector2(x, y));
								ct.clipAABox(-0.5, -0.5, 0.5, 0.5);
								Vector2 centroid = ct.centroid();
								float area = ct.area();
								if (area > 0.0f) {
									Vector3 texCent = tex - dx * centroid.x - dy * centroid.y;
									//xaAssert(texCent.x >= -0.1f && texCent.x <= 1.1f); // @@ Centroid is not very exact...
									//xaAssert(texCent.y >= -0.1f && texCent.y <= 1.1f);
									//xaAssert(texCent.z >= -0.1f && texCent.z <= 1.1f);
									//Vector3 texCent2 = t1 + dx * (x - v1.x) + dy * (y - v1.y);
									if (!cb(param, (int)x, (int)y, texCent, dx, dy, area)) {
										return false;
									}
								}
							}
							CX1 += n1.x;
							CX2 += n2.x;
							CX3 += n3.x;
							tex += dx;
						}
						CY1 += n1.y;
						CY2 += n2.y;
						CY3 += n3.y;
						texRow += dy;
					}
				}
			}
		}
		return true;
	}

	void flipBackface() {
		// check if triangle is backfacing, if so, swap two vertices
		if (((v3.x - v1.x) * (v2.y - v1.y) - (v3.y - v1.y) * (v2.x - v1.x)) < 0) {
			Vector2 hv = v1;
			v1 = v2;
			v2 = hv; // swap pos
			Vector3 ht = t1;
			t1 = t2;
			t2 = ht; // swap tex
		}
	}

	// compute unit inward normals for each edge.
	void computeUnitInwardNormals() {
		n1 = v1 - v2;
		n1 = Vector2(-n1.y, n1.x);
		n1 = n1 * (1.0f / sqrtf(n1.x * n1.x + n1.y * n1.y));
		n2 = v2 - v3;
		n2 = Vector2(-n2.y, n2.x);
		n2 = n2 * (1.0f / sqrtf(n2.x * n2.x + n2.y * n2.y));
		n3 = v3 - v1;
		n3 = Vector2(-n3.y, n3.x);
		n3 = n3 * (1.0f / sqrtf(n3.x * n3.x + n3.y * n3.y));
	}

	// Vertices.
	Vector2 v1, v2, v3;
	Vector2 n1, n2, n3; // unit inward normals
	Vector3 t1, t2, t3;

	// Deltas.
	Vector3 dx, dy;

	float sign;
	bool valid;
};

enum Mode {
	Mode_Nearest,
	Mode_Antialiased
};

// Process the given triangle. Returns false if rasterization was interrupted by the callback.
static bool drawTriangle(Mode mode, Vector2::Arg extents, bool enableScissors, const Vector2 v[3], SamplingCallback cb, void *param) {
	Triangle tri(v[0], v[1], v[2], Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1));
	// @@ It would be nice to have a conservative drawing mode that enlarges the triangle extents by one texel and is able to handle degenerate triangles.
	// @@ Maybe the simplest thing to do would be raster triangle edges.
	if (tri.valid) {
		if (mode == Mode_Antialiased) {
			return tri.drawAA(extents, enableScissors, cb, param);
		}
		if (mode == Mode_Nearest) {
			return tri.draw(extents, enableScissors, cb, param);
		}
	}
	return true;
}

// Process the given quad. Returns false if rasterization was interrupted by the callback.
static bool drawQuad(Mode mode, Vector2::Arg extents, bool enableScissors, const Vector2 v[4], SamplingCallback cb, void *param) {
	bool sign0 = triangleArea2(v[0], v[1], v[2]) > 0.0f;
	bool sign1 = triangleArea2(v[0], v[2], v[3]) > 0.0f;
	// Divide the quad into two non overlapping triangles.
	if (sign0 == sign1) {
		Triangle tri0(v[0], v[1], v[2], Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0));
		Triangle tri1(v[0], v[2], v[3], Vector3(0, 0, 0), Vector3(1, 1, 0), Vector3(0, 1, 0));
		if (tri0.valid && tri1.valid) {
			if (mode == Mode_Antialiased) {
				return tri0.drawAA(extents, enableScissors, cb, param) && tri1.drawAA(extents, enableScissors, cb, param);
			} else {
				return tri0.draw(extents, enableScissors, cb, param) && tri1.draw(extents, enableScissors, cb, param);
			}
		}
	} else {
		Triangle tri0(v[0], v[1], v[3], Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0));
		Triangle tri1(v[1], v[2], v[3], Vector3(1, 0, 0), Vector3(1, 1, 0), Vector3(0, 1, 0));
		if (tri0.valid && tri1.valid) {
			if (mode == Mode_Antialiased) {
				return tri0.drawAA(extents, enableScissors, cb, param) && tri1.drawAA(extents, enableScissors, cb, param);
			} else {
				return tri0.draw(extents, enableScissors, cb, param) && tri1.draw(extents, enableScissors, cb, param);
			}
		}
	}
	return true;
}
} // namespace raster

// Full and sparse vector and matrix classes. BLAS subset.
// Pseudo-BLAS interface.
namespace sparse {
enum Transpose {
	NoTransposed = 0,
	Transposed = 1
};

/**
* Sparse matrix class. The matrix is assumed to be sparse and to have
* very few non-zero elements, for this reason it's stored in indexed
* format. To multiply column vectors efficiently, the matrix stores
* the elements in indexed-column order, there is a list of indexed
* elements for each row of the matrix. As with the FullVector the
* dimension of the matrix is constant.
**/
class Matrix {
public:
	// An element of the sparse array.
	struct Coefficient {
		uint32_t x; // column
		float v; // value
	};

	Matrix(uint32_t d) :
			m_width(d) { m_array.resize(d); }
	Matrix(uint32_t w, uint32_t h) :
			m_width(w) { m_array.resize(h); }
	Matrix(const Matrix &m) :
			m_width(m.m_width) { m_array = m.m_array; }

	const Matrix &operator=(const Matrix &m) {
		xaAssert(width() == m.width());
		xaAssert(height() == m.height());
		m_array = m.m_array;
		return *this;
	}

	uint32_t width() const { return m_width; }
	uint32_t height() const { return m_array.size(); }
	bool isSquare() const { return width() == height(); }

	// x is column, y is row
	float getCoefficient(uint32_t x, uint32_t y) const {
		xaDebugAssert(x < width());
		xaDebugAssert(y < height());
		const uint32_t count = m_array[y].size();
		for (uint32_t i = 0; i < count; i++) {
			if (m_array[y][i].x == x) return m_array[y][i].v;
		}
		return 0.0f;
	}

	void setCoefficient(uint32_t x, uint32_t y, float f) {
		xaDebugAssert(x < width());
		xaDebugAssert(y < height());
		const uint32_t count = m_array[y].size();
		for (uint32_t i = 0; i < count; i++) {
			if (m_array[y][i].x == x) {
				m_array[y][i].v = f;
				return;
			}
		}
		if (f != 0.0f) {
			Coefficient c = { x, f };
			m_array[y].push_back(c);
		}
	}

	float dotRow(uint32_t y, const FullVector &v) const {
		xaDebugAssert(y < height());
		const uint32_t count = m_array[y].size();
		float sum = 0;
		for (uint32_t i = 0; i < count; i++) {
			sum += m_array[y][i].v * v[m_array[y][i].x];
		}
		return sum;
	}

	void madRow(uint32_t y, float alpha, FullVector &v) const {
		xaDebugAssert(y < height());
		const uint32_t count = m_array[y].size();
		for (uint32_t i = 0; i < count; i++) {
			v[m_array[y][i].x] += alpha * m_array[y][i].v;
		}
	}

	void clearRow(uint32_t y) {
		xaDebugAssert(y < height());
		m_array[y].clear();
	}

	void scaleRow(uint32_t y, float f) {
		xaDebugAssert(y < height());
		const uint32_t count = m_array[y].size();
		for (uint32_t i = 0; i < count; i++) {
			m_array[y][i].v *= f;
		}
	}

	const std::vector<Coefficient> &getRow(uint32_t y) const { return m_array[y]; }

private:
	/// Number of columns.
	const uint32_t m_width;

	/// Array of matrix elements.
	std::vector<std::vector<Coefficient> > m_array;
};

// y = a * x + y
static void saxpy(float a, const FullVector &x, FullVector &y) {
	xaDebugAssert(x.dimension() == y.dimension());
	const uint32_t dim = x.dimension();
	for (uint32_t i = 0; i < dim; i++) {
		y[i] += a * x[i];
	}
}

static void copy(const FullVector &x, FullVector &y) {
	xaDebugAssert(x.dimension() == y.dimension());
	const uint32_t dim = x.dimension();
	for (uint32_t i = 0; i < dim; i++) {
		y[i] = x[i];
	}
}

static void scal(float a, FullVector &x) {
	const uint32_t dim = x.dimension();
	for (uint32_t i = 0; i < dim; i++) {
		x[i] *= a;
	}
}

static float dot(const FullVector &x, const FullVector &y) {
	xaDebugAssert(x.dimension() == y.dimension());
	const uint32_t dim = x.dimension();
	float sum = 0;
	for (uint32_t i = 0; i < dim; i++) {
		sum += x[i] * y[i];
	}
	return sum;
}

static void mult(Transpose TM, const Matrix &M, const FullVector &x, FullVector &y) {
	const uint32_t w = M.width();
	const uint32_t h = M.height();
	if (TM == Transposed) {
		xaDebugAssert(h == x.dimension());
		xaDebugAssert(w == y.dimension());
		y.fill(0.0f);
		for (uint32_t i = 0; i < h; i++) {
			M.madRow(i, x[i], y);
		}
	} else {
		xaDebugAssert(w == x.dimension());
		xaDebugAssert(h == y.dimension());
		for (uint32_t i = 0; i < h; i++) {
			y[i] = M.dotRow(i, x);
		}
	}
}

// y = M * x
static void mult(const Matrix &M, const FullVector &x, FullVector &y) {
	mult(NoTransposed, M, x, y);
}

static void sgemv(float alpha, Transpose TA, const Matrix &A, const FullVector &x, float beta, FullVector &y) {
	const uint32_t w = A.width();
	const uint32_t h = A.height();
	if (TA == Transposed) {
		xaDebugAssert(h == x.dimension());
		xaDebugAssert(w == y.dimension());
		for (uint32_t i = 0; i < h; i++) {
			A.madRow(i, alpha * x[i], y);
		}
	} else {
		xaDebugAssert(w == x.dimension());
		xaDebugAssert(h == y.dimension());
		for (uint32_t i = 0; i < h; i++) {
			y[i] = alpha * A.dotRow(i, x) + beta * y[i];
		}
	}
}

// y = alpha*A*x + beta*y
static void sgemv(float alpha, const Matrix &A, const FullVector &x, float beta, FullVector &y) {
	sgemv(alpha, NoTransposed, A, x, beta, y);
}

// dot y-row of A by x-column of B
static float dotRowColumn(int y, const Matrix &A, int x, const Matrix &B) {
	const std::vector<Matrix::Coefficient> &row = A.getRow(y);
	const uint32_t count = row.size();
	float sum = 0.0f;
	for (uint32_t i = 0; i < count; i++) {
		const Matrix::Coefficient &c = row[i];
		sum += c.v * B.getCoefficient(x, c.x);
	}
	return sum;
}

// dot y-row of A by x-row of B
static float dotRowRow(int y, const Matrix &A, int x, const Matrix &B) {
	const std::vector<Matrix::Coefficient> &row = A.getRow(y);
	const uint32_t count = row.size();
	float sum = 0.0f;
	for (uint32_t i = 0; i < count; i++) {
		const Matrix::Coefficient &c = row[i];
		sum += c.v * B.getCoefficient(c.x, x);
	}
	return sum;
}

// dot y-column of A by x-column of B
static float dotColumnColumn(int y, const Matrix &A, int x, const Matrix &B) {
	xaDebugAssert(A.height() == B.height());
	const uint32_t h = A.height();
	float sum = 0.0f;
	for (uint32_t i = 0; i < h; i++) {
		sum += A.getCoefficient(y, i) * B.getCoefficient(x, i);
	}
	return sum;
}

static void transpose(const Matrix &A, Matrix &B) {
	xaDebugAssert(A.width() == B.height());
	xaDebugAssert(B.width() == A.height());
	const uint32_t w = A.width();
	for (uint32_t x = 0; x < w; x++) {
		B.clearRow(x);
	}
	const uint32_t h = A.height();
	for (uint32_t y = 0; y < h; y++) {
		const std::vector<Matrix::Coefficient> &row = A.getRow(y);
		const uint32_t count = row.size();
		for (uint32_t i = 0; i < count; i++) {
			const Matrix::Coefficient &c = row[i];
			xaDebugAssert(c.x < w);
			B.setCoefficient(y, c.x, c.v);
		}
	}
}

static void sgemm(float alpha, Transpose TA, const Matrix &A, Transpose TB, const Matrix &B, float beta, Matrix &C) {
	const uint32_t w = C.width();
	const uint32_t h = C.height();
	uint32_t aw = (TA == NoTransposed) ? A.width() : A.height();
	uint32_t ah = (TA == NoTransposed) ? A.height() : A.width();
	uint32_t bw = (TB == NoTransposed) ? B.width() : B.height();
	uint32_t bh = (TB == NoTransposed) ? B.height() : B.width();
	xaDebugAssert(aw == bh);
	xaDebugAssert(bw == ah);
	xaDebugAssert(w == bw);
	xaDebugAssert(h == ah);
#ifdef NDEBUG
	aw = ah = bw = bh = 0; // silence unused parameter warning
#endif
	for (uint32_t y = 0; y < h; y++) {
		for (uint32_t x = 0; x < w; x++) {
			float c = beta * C.getCoefficient(x, y);
			if (TA == NoTransposed && TB == NoTransposed) {
				// dot y-row of A by x-column of B.
				c += alpha * dotRowColumn(y, A, x, B);
			} else if (TA == Transposed && TB == Transposed) {
				// dot y-column of A by x-row of B.
				c += alpha * dotRowColumn(x, B, y, A);
			} else if (TA == Transposed && TB == NoTransposed) {
				// dot y-column of A by x-column of B.
				c += alpha * dotColumnColumn(y, A, x, B);
			} else if (TA == NoTransposed && TB == Transposed) {
				// dot y-row of A by x-row of B.
				c += alpha * dotRowRow(y, A, x, B);
			}
			C.setCoefficient(x, y, c);
		}
	}
}

static void mult(Transpose TA, const Matrix &A, Transpose TB, const Matrix &B, Matrix &C) {
	sgemm(1.0f, TA, A, TB, B, 0.0f, C);
}

// C = A * B
static void mult(const Matrix &A, const Matrix &B, Matrix &C) {
	mult(NoTransposed, A, NoTransposed, B, C);
}

} // namespace sparse

class JacobiPreconditioner {
public:
	JacobiPreconditioner(const sparse::Matrix &M, bool symmetric) :
			m_inverseDiagonal(M.width()) {
		xaAssert(M.isSquare());
		for (uint32_t x = 0; x < M.width(); x++) {
			float elem = M.getCoefficient(x, x);
			//xaDebugAssert( elem != 0.0f ); // This can be zero in the presence of zero area triangles.
			if (symmetric) {
				m_inverseDiagonal[x] = (elem != 0) ? 1.0f / sqrtf(fabsf(elem)) : 1.0f;
			} else {
				m_inverseDiagonal[x] = (elem != 0) ? 1.0f / elem : 1.0f;
			}
		}
	}

	void apply(const FullVector &x, FullVector &y) const {
		xaDebugAssert(x.dimension() == m_inverseDiagonal.dimension());
		xaDebugAssert(y.dimension() == m_inverseDiagonal.dimension());
		// @@ Wrap vector component-wise product into a separate function.
		const uint32_t D = x.dimension();
		for (uint32_t i = 0; i < D; i++) {
			y[i] = m_inverseDiagonal[i] * x[i];
		}
	}

private:
	FullVector m_inverseDiagonal;
};

// Linear solvers.
class Solver {
public:
	// Solve the symmetric system: At·A·x = At·b
	static bool LeastSquaresSolver(const sparse::Matrix &A, const FullVector &b, FullVector &x, float epsilon = 1e-5f) {
		xaDebugAssert(A.width() == x.dimension());
		xaDebugAssert(A.height() == b.dimension());
		xaDebugAssert(A.height() >= A.width()); // @@ If height == width we could solve it directly...
		const uint32_t D = A.width();
		sparse::Matrix At(A.height(), A.width());
		sparse::transpose(A, At);
		FullVector Atb(D);
		sparse::mult(At, b, Atb);
		sparse::Matrix AtA(D);
		sparse::mult(At, A, AtA);
		return SymmetricSolver(AtA, Atb, x, epsilon);
	}

	// See section 10.4.3 in: Mesh Parameterization: Theory and Practice, Siggraph Course Notes, August 2007
	static bool LeastSquaresSolver(const sparse::Matrix &A, const FullVector &b, FullVector &x, const uint32_t *lockedParameters, uint32_t lockedCount, float epsilon = 1e-5f) {
		xaDebugAssert(A.width() == x.dimension());
		xaDebugAssert(A.height() == b.dimension());
		xaDebugAssert(A.height() >= A.width() - lockedCount);
		// @@ This is not the most efficient way of building a system with reduced degrees of freedom. It would be faster to do it on the fly.
		const uint32_t D = A.width() - lockedCount;
		xaDebugAssert(D > 0);
		// Compute: b - Al * xl
		FullVector b_Alxl(b);
		for (uint32_t y = 0; y < A.height(); y++) {
			const uint32_t count = A.getRow(y).size();
			for (uint32_t e = 0; e < count; e++) {
				uint32_t column = A.getRow(y)[e].x;
				bool isFree = true;
				for (uint32_t i = 0; i < lockedCount; i++) {
					isFree &= (lockedParameters[i] != column);
				}
				if (!isFree) {
					b_Alxl[y] -= x[column] * A.getRow(y)[e].v;
				}
			}
		}
		// Remove locked columns from A.
		sparse::Matrix Af(D, A.height());
		for (uint32_t y = 0; y < A.height(); y++) {
			const uint32_t count = A.getRow(y).size();
			for (uint32_t e = 0; e < count; e++) {
				uint32_t column = A.getRow(y)[e].x;
				uint32_t ix = column;
				bool isFree = true;
				for (uint32_t i = 0; i < lockedCount; i++) {
					isFree &= (lockedParameters[i] != column);
					if (column > lockedParameters[i]) ix--; // shift columns
				}
				if (isFree) {
					Af.setCoefficient(ix, y, A.getRow(y)[e].v);
				}
			}
		}
		// Remove elements from x
		FullVector xf(D);
		for (uint32_t i = 0, j = 0; i < A.width(); i++) {
			bool isFree = true;
			for (uint32_t l = 0; l < lockedCount; l++) {
				isFree &= (lockedParameters[l] != i);
			}
			if (isFree) {
				xf[j++] = x[i];
			}
		}
		// Solve reduced system.
		bool result = LeastSquaresSolver(Af, b_Alxl, xf, epsilon);
		// Copy results back to x.
		for (uint32_t i = 0, j = 0; i < A.width(); i++) {
			bool isFree = true;
			for (uint32_t l = 0; l < lockedCount; l++) {
				isFree &= (lockedParameters[l] != i);
			}
			if (isFree) {
				x[i] = xf[j++];
			}
		}
		return result;
	}

private:
	/**
	* Compute the solution of the sparse linear system Ab=x using the Conjugate
	* Gradient method.
	*
	* Solving sparse linear systems:
	* (1)		A·x = b
	*
	* The conjugate gradient algorithm solves (1) only in the case that A is
	* symmetric and positive definite. It is based on the idea of minimizing the
	* function
	*
	* (2)		f(x) = 1/2·x·A·x - b·x
	*
	* This function is minimized when its gradient
	*
	* (3)		df = A·x - b
	*
	* is zero, which is equivalent to (1). The minimization is carried out by
	* generating a succession of search directions p.k and improved minimizers x.k.
	* At each stage a quantity alfa.k is found that minimizes f(x.k + alfa.k·p.k),
	* and x.k+1 is set equal to the new point x.k + alfa.k·p.k. The p.k and x.k are
	* built up in such a way that x.k+1 is also the minimizer of f over the whole
	* vector space of directions already taken, {p.1, p.2, . . . , p.k}. After N
	* iterations you arrive at the minimizer over the entire vector space, i.e., the
	* solution to (1).
	*
	* For a really good explanation of the method see:
	*
	* "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain",
	* Jonhathan Richard Shewchuk.
	*
	**/
	static bool ConjugateGradientSolver(const sparse::Matrix &A, const FullVector &b, FullVector &x, float epsilon) {
		xaDebugAssert(A.isSquare());
		xaDebugAssert(A.width() == b.dimension());
		xaDebugAssert(A.width() == x.dimension());
		int i = 0;
		const int D = A.width();
		const int i_max = 4 * D; // Convergence should be linear, but in some cases, it's not.
		FullVector r(D); // residual
		FullVector p(D); // search direction
		FullVector q(D); //
		float delta_0;
		float delta_old;
		float delta_new;
		float alpha;
		float beta;
		// r = b - A·x;
		sparse::copy(b, r);
		sparse::sgemv(-1, A, x, 1, r);
		// p = r;
		sparse::copy(r, p);
		delta_new = sparse::dot(r, r);
		delta_0 = delta_new;
		while (i < i_max && delta_new > epsilon * epsilon * delta_0) {
			i++;
			// q = A·p
			mult(A, p, q);
			// alpha = delta_new / p·q
			alpha = delta_new / sparse::dot(p, q);
			// x = alfa·p + x
			sparse::saxpy(alpha, p, x);
			if ((i & 31) == 0) { // recompute r after 32 steps
				// r = b - A·x
				sparse::copy(b, r);
				sparse::sgemv(-1, A, x, 1, r);
			} else {
				// r = r - alpha·q
				sparse::saxpy(-alpha, q, r);
			}
			delta_old = delta_new;
			delta_new = sparse::dot(r, r);
			beta = delta_new / delta_old;
			// p = beta·p + r
			sparse::scal(beta, p);
			sparse::saxpy(1, r, p);
		}
		return delta_new <= epsilon * epsilon * delta_0;
	}

	// Conjugate gradient with preconditioner.
	static bool ConjugateGradientSolver(const JacobiPreconditioner &preconditioner, const sparse::Matrix &A, const FullVector &b, FullVector &x, float epsilon) {
		xaDebugAssert(A.isSquare());
		xaDebugAssert(A.width() == b.dimension());
		xaDebugAssert(A.width() == x.dimension());
		int i = 0;
		const int D = A.width();
		const int i_max = 4 * D; // Convergence should be linear, but in some cases, it's not.
		FullVector r(D); // residual
		FullVector p(D); // search direction
		FullVector q(D); //
		FullVector s(D); // preconditioned
		float delta_0;
		float delta_old;
		float delta_new;
		float alpha;
		float beta;
		// r = b - A·x
		sparse::copy(b, r);
		sparse::sgemv(-1, A, x, 1, r);
		// p = M^-1 · r
		preconditioner.apply(r, p);
		delta_new = sparse::dot(r, p);
		delta_0 = delta_new;
		while (i < i_max && delta_new > epsilon * epsilon * delta_0) {
			i++;
			// q = A·p
			mult(A, p, q);
			// alpha = delta_new / p·q
			alpha = delta_new / sparse::dot(p, q);
			// x = alfa·p + x
			sparse::saxpy(alpha, p, x);
			if ((i & 31) == 0) { // recompute r after 32 steps
				// r = b - A·x
				sparse::copy(b, r);
				sparse::sgemv(-1, A, x, 1, r);
			} else {
				// r = r - alfa·q
				sparse::saxpy(-alpha, q, r);
			}
			// s = M^-1 · r
			preconditioner.apply(r, s);
			delta_old = delta_new;
			delta_new = sparse::dot(r, s);
			beta = delta_new / delta_old;
			// p = s + beta·p
			sparse::scal(beta, p);
			sparse::saxpy(1, s, p);
		}
		return delta_new <= epsilon * epsilon * delta_0;
	}

	static bool SymmetricSolver(const sparse::Matrix &A, const FullVector &b, FullVector &x, float epsilon = 1e-5f) {
		xaDebugAssert(A.height() == A.width());
		xaDebugAssert(A.height() == b.dimension());
		xaDebugAssert(b.dimension() == x.dimension());
		JacobiPreconditioner jacobi(A, true);
		return ConjugateGradientSolver(jacobi, A, b, x, epsilon);
	}
};

namespace param {
class Atlas;
class Chart;

// Fast sweep in 3 directions
static bool findApproximateDiameterVertices(halfedge::Mesh *mesh, halfedge::Vertex **a, halfedge::Vertex **b) {
	xaDebugAssert(mesh != NULL);
	xaDebugAssert(a != NULL);
	xaDebugAssert(b != NULL);
	const uint32_t vertexCount = mesh->vertexCount();
	halfedge::Vertex *minVertex[3];
	halfedge::Vertex *maxVertex[3];
	minVertex[0] = minVertex[1] = minVertex[2] = NULL;
	maxVertex[0] = maxVertex[1] = maxVertex[2] = NULL;
	for (uint32_t v = 1; v < vertexCount; v++) {
		halfedge::Vertex *vertex = mesh->vertexAt(v);
		xaDebugAssert(vertex != NULL);
		if (vertex->isBoundary()) {
			minVertex[0] = minVertex[1] = minVertex[2] = vertex;
			maxVertex[0] = maxVertex[1] = maxVertex[2] = vertex;
			break;
		}
	}
	if (minVertex[0] == NULL) {
		// Input mesh has not boundaries.
		return false;
	}
	for (uint32_t v = 1; v < vertexCount; v++) {
		halfedge::Vertex *vertex = mesh->vertexAt(v);
		xaDebugAssert(vertex != NULL);
		if (!vertex->isBoundary()) {
			// Skip interior vertices.
			continue;
		}
		if (vertex->pos.x < minVertex[0]->pos.x)
			minVertex[0] = vertex;
		else if (vertex->pos.x > maxVertex[0]->pos.x)
			maxVertex[0] = vertex;
		if (vertex->pos.y < minVertex[1]->pos.y)
			minVertex[1] = vertex;
		else if (vertex->pos.y > maxVertex[1]->pos.y)
			maxVertex[1] = vertex;
		if (vertex->pos.z < minVertex[2]->pos.z)
			minVertex[2] = vertex;
		else if (vertex->pos.z > maxVertex[2]->pos.z)
			maxVertex[2] = vertex;
	}
	float lengths[3];
	for (int i = 0; i < 3; i++) {
		lengths[i] = length(minVertex[i]->pos - maxVertex[i]->pos);
	}
	if (lengths[0] > lengths[1] && lengths[0] > lengths[2]) {
		*a = minVertex[0];
		*b = maxVertex[0];
	} else if (lengths[1] > lengths[2]) {
		*a = minVertex[1];
		*b = maxVertex[1];
	} else {
		*a = minVertex[2];
		*b = maxVertex[2];
	}
	return true;
}

// Conformal relations from Brecht Van Lommel (based on ABF):

static float vec_angle_cos(Vector3::Arg v1, Vector3::Arg v2, Vector3::Arg v3) {
	Vector3 d1 = v1 - v2;
	Vector3 d2 = v3 - v2;
	return clamp(dot(d1, d2) / (length(d1) * length(d2)), -1.0f, 1.0f);
}

static float vec_angle(Vector3::Arg v1, Vector3::Arg v2, Vector3::Arg v3) {
	float dot = vec_angle_cos(v1, v2, v3);
	return acosf(dot);
}

static void triangle_angles(Vector3::Arg v1, Vector3::Arg v2, Vector3::Arg v3, float *a1, float *a2, float *a3) {
	*a1 = vec_angle(v3, v1, v2);
	*a2 = vec_angle(v1, v2, v3);
	*a3 = PI - *a2 - *a1;
}

static void setup_abf_relations(sparse::Matrix &A, int row, const halfedge::Vertex *v0, const halfedge::Vertex *v1, const halfedge::Vertex *v2) {
	int id0 = v0->id;
	int id1 = v1->id;
	int id2 = v2->id;
	Vector3 p0 = v0->pos;
	Vector3 p1 = v1->pos;
	Vector3 p2 = v2->pos;
	// @@ IC: Wouldn't it be more accurate to return cos and compute 1-cos^2?
	// It does indeed seem to be a little bit more robust.
	// @@ Need to revisit this more carefully!
	float a0, a1, a2;
	triangle_angles(p0, p1, p2, &a0, &a1, &a2);
	float s0 = sinf(a0);
	float s1 = sinf(a1);
	float s2 = sinf(a2);
	if (s1 > s0 && s1 > s2) {
		std::swap(s1, s2);
		std::swap(s0, s1);
		std::swap(a1, a2);
		std::swap(a0, a1);
		std::swap(id1, id2);
		std::swap(id0, id1);
	} else if (s0 > s1 && s0 > s2) {
		std::swap(s0, s2);
		std::swap(s0, s1);
		std::swap(a0, a2);
		std::swap(a0, a1);
		std::swap(id0, id2);
		std::swap(id0, id1);
	}
	float c0 = cosf(a0);
	float ratio = (s2 == 0.0f) ? 1.0f : s1 / s2;
	float cosine = c0 * ratio;
	float sine = s0 * ratio;
	// Note  : 2*id + 0 --> u
	//         2*id + 1 --> v
	int u0_id = 2 * id0 + 0;
	int v0_id = 2 * id0 + 1;
	int u1_id = 2 * id1 + 0;
	int v1_id = 2 * id1 + 1;
	int u2_id = 2 * id2 + 0;
	int v2_id = 2 * id2 + 1;
	// Real part
	A.setCoefficient(u0_id, 2 * row + 0, cosine - 1.0f);
	A.setCoefficient(v0_id, 2 * row + 0, -sine);
	A.setCoefficient(u1_id, 2 * row + 0, -cosine);
	A.setCoefficient(v1_id, 2 * row + 0, sine);
	A.setCoefficient(u2_id, 2 * row + 0, 1);
	// Imaginary part
	A.setCoefficient(u0_id, 2 * row + 1, sine);
	A.setCoefficient(v0_id, 2 * row + 1, cosine - 1.0f);
	A.setCoefficient(u1_id, 2 * row + 1, -sine);
	A.setCoefficient(v1_id, 2 * row + 1, -cosine);
	A.setCoefficient(v2_id, 2 * row + 1, 1);
}

bool computeLeastSquaresConformalMap(halfedge::Mesh *mesh) {
	xaDebugAssert(mesh != NULL);
	// For this to work properly, mesh should not have colocals that have the same
	// attributes, unless you want the vertices to actually have different texcoords.
	const uint32_t vertexCount = mesh->vertexCount();
	const uint32_t D = 2 * vertexCount;
	const uint32_t N = 2 * halfedge::countMeshTriangles(mesh);
	// N is the number of equations (one per triangle)
	// D is the number of variables (one per vertex; there are 2 pinned vertices).
	if (N < D - 4) {
		return false;
	}
	sparse::Matrix A(D, N);
	FullVector b(N);
	FullVector x(D);
	// Fill b:
	b.fill(0.0f);
	// Fill x:
	halfedge::Vertex *v0;
	halfedge::Vertex *v1;
	if (!findApproximateDiameterVertices(mesh, &v0, &v1)) {
		// Mesh has no boundaries.
		return false;
	}
	if (v0->tex == v1->tex) {
		// LSCM expects an existing parameterization.
		return false;
	}
	for (uint32_t v = 0; v < vertexCount; v++) {
		halfedge::Vertex *vertex = mesh->vertexAt(v);
		xaDebugAssert(vertex != NULL);
		// Initial solution.
		x[2 * v + 0] = vertex->tex.x;
		x[2 * v + 1] = vertex->tex.y;
	}
	// Fill A:
	const uint32_t faceCount = mesh->faceCount();
	for (uint32_t f = 0, t = 0; f < faceCount; f++) {
		const halfedge::Face *face = mesh->faceAt(f);
		xaDebugAssert(face != NULL);
		xaDebugAssert(face->edgeCount() == 3);
		const halfedge::Vertex *vertex0 = NULL;
		for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
			const halfedge::Edge *edge = it.current();
			xaAssert(edge != NULL);
			if (vertex0 == NULL) {
				vertex0 = edge->vertex;
			} else if (edge->next->vertex != vertex0) {
				const halfedge::Vertex *vertex1 = edge->from();
				const halfedge::Vertex *vertex2 = edge->to();
				setup_abf_relations(A, t, vertex0, vertex1, vertex2);
				//setup_conformal_map_relations(A, t, vertex0, vertex1, vertex2);
				t++;
			}
		}
	}
	const uint32_t lockedParameters[] = {
		2 * v0->id + 0,
		2 * v0->id + 1,
		2 * v1->id + 0,
		2 * v1->id + 1
	};
	// Solve
	Solver::LeastSquaresSolver(A, b, x, lockedParameters, 4, 0.000001f);
	// Map x back to texcoords:
	for (uint32_t v = 0; v < vertexCount; v++) {
		halfedge::Vertex *vertex = mesh->vertexAt(v);
		xaDebugAssert(vertex != NULL);
		vertex->tex = Vector2(x[2 * v + 0], x[2 * v + 1]);
	}
	return true;
}

bool computeOrthogonalProjectionMap(halfedge::Mesh *mesh) {
	Vector3 axis[2];
	uint32_t vertexCount = mesh->vertexCount();
	std::vector<Vector3> points(vertexCount);
	points.resize(vertexCount);
	for (uint32_t i = 0; i < vertexCount; i++) {
		points[i] = mesh->vertexAt(i)->pos;
	}
	// Avoid redundant computations.
	float matrix[6];
	Fit::computeCovariance(vertexCount, points.data(), matrix);
	if (matrix[0] == 0 && matrix[3] == 0 && matrix[5] == 0) {
		return false;
	}
	float eigenValues[3];
	Vector3 eigenVectors[3];
	if (!Fit::eigenSolveSymmetric3(matrix, eigenValues, eigenVectors)) {
		return false;
	}
	axis[0] = normalize(eigenVectors[0]);
	axis[1] = normalize(eigenVectors[1]);
	// Project vertices to plane.
	for (halfedge::Mesh::VertexIterator it(mesh->vertices()); !it.isDone(); it.advance()) {
		halfedge::Vertex *vertex = it.current();
		vertex->tex.x = dot(axis[0], vertex->pos);
		vertex->tex.y = dot(axis[1], vertex->pos);
	}
	return true;
}

void computeSingleFaceMap(halfedge::Mesh *mesh) {
	xaDebugAssert(mesh != NULL);
	xaDebugAssert(mesh->faceCount() == 1);
	halfedge::Face *face = mesh->faceAt(0);
	xaAssert(face != NULL);
	Vector3 p0 = face->edge->from()->pos;
	Vector3 p1 = face->edge->to()->pos;
	Vector3 X = normalizeSafe(p1 - p0, Vector3(0.0f), 0.0f);
	Vector3 Z = face->normal();
	Vector3 Y = normalizeSafe(cross(Z, X), Vector3(0.0f), 0.0f);
	uint32_t i = 0;
	for (halfedge::Face::EdgeIterator it(face->edges()); !it.isDone(); it.advance(), i++) {
		halfedge::Vertex *vertex = it.vertex();
		xaAssert(vertex != NULL);
		if (i == 0) {
			vertex->tex = Vector2(0);
		} else {
			Vector3 pn = vertex->pos;
			float xn = dot((pn - p0), X);
			float yn = dot((pn - p0), Y);
			vertex->tex = Vector2(xn, yn);
		}
	}
}

// Dummy implementation of a priority queue using sort at insertion.
// - Insertion is o(n)
// - Smallest element goes at the end, so that popping it is o(1).
// - Resorting is n*log(n)
// @@ Number of elements in the queue is usually small, and we'd have to rebalance often. I'm not sure it's worth implementing a heap.
// @@ Searcing at removal would remove the need for sorting when priorities change.
struct PriorityQueue {
	PriorityQueue(uint32_t size = UINT_MAX) :
			maxSize(size) {}

	void push(float priority, uint32_t face) {
		uint32_t i = 0;
		const uint32_t count = pairs.size();
		for (; i < count; i++) {
			if (pairs[i].priority > priority) break;
		}
		Pair p = { priority, face };
		pairs.insert(pairs.begin() + i, p);
		if (pairs.size() > maxSize) {
			pairs.erase(pairs.begin());
		}
	}

	// push face out of order, to be sorted later.
	void push(uint32_t face) {
		Pair p = { 0.0f, face };
		pairs.push_back(p);
	}

	uint32_t pop() {
		uint32_t f = pairs.back().face;
		pairs.pop_back();
		return f;
	}

	void sort() {
		//sort(pairs); // @@ My intro sort appears to be much slower than it should!
		std::sort(pairs.begin(), pairs.end());
	}

	void clear() {
		pairs.clear();
	}

	uint32_t count() const {
		return pairs.size();
	}

	float firstPriority() const {
		return pairs.back().priority;
	}

	const uint32_t maxSize;

	struct Pair {
		bool operator<(const Pair &p) const {
			return priority > p.priority; // !! Sort in inverse priority order!
		}

		float priority;
		uint32_t face;
	};

	std::vector<Pair> pairs;
};

struct ChartBuildData {
	ChartBuildData(int p_id) :
			id(p_id) {
		planeNormal = Vector3(0);
		centroid = Vector3(0);
		coneAxis = Vector3(0);
		coneAngle = 0;
		area = 0;
		boundaryLength = 0;
		normalSum = Vector3(0);
		centroidSum = Vector3(0);
	}

	int id;

	// Proxy info:
	Vector3 planeNormal;
	Vector3 centroid;
	Vector3 coneAxis;
	float coneAngle;

	float area;
	float boundaryLength;
	Vector3 normalSum;
	Vector3 centroidSum;

	std::vector<uint32_t> seeds; // @@ These could be a pointers to the halfedge faces directly.
	std::vector<uint32_t> faces;
	PriorityQueue candidates;
};

struct AtlasBuilder {
	AtlasBuilder(const halfedge::Mesh *m) :
			mesh(m),
			facesLeft(m->faceCount()) {
		const uint32_t faceCount = m->faceCount();
		faceChartArray.resize(faceCount, -1);
		faceCandidateArray.resize(faceCount, (uint32_t)-1);
		// @@ Floyd for the whole mesh is too slow. We could compute floyd progressively per patch as the patch grows. We need a better solution to compute most central faces.
		//computeShortestPaths();
		// Precompute edge lengths and face areas.
		uint32_t edgeCount = m->edgeCount();
		edgeLengths.resize(edgeCount);
		for (uint32_t i = 0; i < edgeCount; i++) {
			uint32_t id = m->edgeAt(i)->id;
			xaDebugAssert(id / 2 == i);
#ifdef NDEBUG
			id = 0; // silence unused parameter warning
#endif
			edgeLengths[i] = m->edgeAt(i)->length();
		}
		faceAreas.resize(faceCount);
		for (uint32_t i = 0; i < faceCount; i++) {
			faceAreas[i] = m->faceAt(i)->area();
		}
	}

	~AtlasBuilder() {
		const uint32_t chartCount = chartArray.size();
		for (uint32_t i = 0; i < chartCount; i++) {
			delete chartArray[i];
		}
	}

	void markUnchartedFaces(const std::vector<uint32_t> &unchartedFaces) {
		const uint32_t unchartedFaceCount = unchartedFaces.size();
		for (uint32_t i = 0; i < unchartedFaceCount; i++) {
			uint32_t f = unchartedFaces[i];
			faceChartArray[f] = -2;
			//faceCandidateArray[f] = -2; // @@ ?
			removeCandidate(f);
		}
		xaDebugAssert(facesLeft >= unchartedFaceCount);
		facesLeft -= unchartedFaceCount;
	}

	void computeShortestPaths() {
		const uint32_t faceCount = mesh->faceCount();
		shortestPaths.resize(faceCount * faceCount, FLT_MAX);
		// Fill edges:
		for (uint32_t i = 0; i < faceCount; i++) {
			shortestPaths[i * faceCount + i] = 0.0f;
			const halfedge::Face *face_i = mesh->faceAt(i);
			Vector3 centroid_i = face_i->centroid();
			for (halfedge::Face::ConstEdgeIterator it(face_i->edges()); !it.isDone(); it.advance()) {
				const halfedge::Edge *edge = it.current();
				if (!edge->isBoundary()) {
					const halfedge::Face *face_j = edge->pair->face;
					uint32_t j = face_j->id;
					Vector3 centroid_j = face_j->centroid();
					shortestPaths[i * faceCount + j] = shortestPaths[j * faceCount + i] = length(centroid_i - centroid_j);
				}
			}
		}
		// Use Floyd-Warshall algorithm to compute all paths:
		for (uint32_t k = 0; k < faceCount; k++) {
			for (uint32_t i = 0; i < faceCount; i++) {
				for (uint32_t j = 0; j < faceCount; j++) {
					shortestPaths[i * faceCount + j] = std::min(shortestPaths[i * faceCount + j], shortestPaths[i * faceCount + k] + shortestPaths[k * faceCount + j]);
				}
			}
		}
	}

	void placeSeeds(float threshold, uint32_t maxSeedCount) {
		// Instead of using a predefiened number of seeds:
		// - Add seeds one by one, growing chart until a certain treshold.
		// - Undo charts and restart growing process.
		// @@ How can we give preference to faces far from sharp features as in the LSCM paper?
		//   - those points can be found using a simple flood filling algorithm.
		//   - how do we weight the probabilities?
		for (uint32_t i = 0; i < maxSeedCount; i++) {
			if (facesLeft == 0) {
				// No faces left, stop creating seeds.
				break;
			}
			createRandomChart(threshold);
		}
	}

	void createRandomChart(float threshold) {
		ChartBuildData *chart = new ChartBuildData(chartArray.size());
		chartArray.push_back(chart);
		// Pick random face that is not used by any chart yet.
		uint32_t randomFaceIdx = rand.getRange(facesLeft - 1);
		uint32_t i = 0;
		for (uint32_t f = 0; f != randomFaceIdx; f++, i++) {
			while (faceChartArray[i] != -1)
				i++;
		}
		while (faceChartArray[i] != -1)
			i++;
		chart->seeds.push_back(i);
		addFaceToChart(chart, i, true);
		// Grow the chart as much as possible within the given threshold.
		growChart(chart, threshold * 0.5f, facesLeft);
		//growCharts(threshold - threshold * 0.75f / chartCount(), facesLeft);
	}

	void addFaceToChart(ChartBuildData *chart, uint32_t f, bool recomputeProxy = false) {
		// Add face to chart.
		chart->faces.push_back(f);
		xaDebugAssert(faceChartArray[f] == -1);
		faceChartArray[f] = chart->id;
		facesLeft--;
		// Update area and boundary length.
		chart->area = evaluateChartArea(chart, f);
		chart->boundaryLength = evaluateBoundaryLength(chart, f);
		chart->normalSum = evaluateChartNormalSum(chart, f);
		chart->centroidSum = evaluateChartCentroidSum(chart, f);
		if (recomputeProxy) {
			// Update proxy and candidate's priorities.
			updateProxy(chart);
		}
		// Update candidates.
		removeCandidate(f);
		updateCandidates(chart, f);
		updatePriorities(chart);
	}

	// Returns true if any of the charts can grow more.
	bool growCharts(float threshold, uint32_t faceCount) {
		// Using one global list.
		faceCount = std::min(faceCount, facesLeft);
		for (uint32_t i = 0; i < faceCount; i++) {
			const Candidate &candidate = getBestCandidate();
			if (candidate.metric > threshold) {
				return false; // Can't grow more.
			}
			addFaceToChart(candidate.chart, candidate.face);
		}
		return facesLeft != 0; // Can continue growing.
	}

	bool growChart(ChartBuildData *chart, float threshold, uint32_t faceCount) {
		// Try to add faceCount faces within threshold to chart.
		for (uint32_t i = 0; i < faceCount;) {
			if (chart->candidates.count() == 0 || chart->candidates.firstPriority() > threshold) {
				return false;
			}
			uint32_t f = chart->candidates.pop();
			if (faceChartArray[f] == -1) {
				addFaceToChart(chart, f);
				i++;
			}
		}
		if (chart->candidates.count() == 0 || chart->candidates.firstPriority() > threshold) {
			return false;
		}
		return true;
	}

	void resetCharts() {
		const uint32_t faceCount = mesh->faceCount();
		for (uint32_t i = 0; i < faceCount; i++) {
			faceChartArray[i] = -1;
			faceCandidateArray[i] = (uint32_t)-1;
		}
		facesLeft = faceCount;
		candidateArray.clear();
		const uint32_t chartCount = chartArray.size();
		for (uint32_t i = 0; i < chartCount; i++) {
			ChartBuildData *chart = chartArray[i];
			const uint32_t seed = chart->seeds.back();
			chart->area = 0.0f;
			chart->boundaryLength = 0.0f;
			chart->normalSum = Vector3(0);
			chart->centroidSum = Vector3(0);
			chart->faces.clear();
			chart->candidates.clear();
			addFaceToChart(chart, seed);
		}
	}

	void updateCandidates(ChartBuildData *chart, uint32_t f) {
		const halfedge::Face *face = mesh->faceAt(f);
		// Traverse neighboring faces, add the ones that do not belong to any chart yet.
		for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
			const halfedge::Edge *edge = it.current()->pair;
			if (!edge->isBoundary()) {
				uint32_t faceId = edge->face->id;
				if (faceChartArray[faceId] == -1) {
					chart->candidates.push(faceId);
				}
			}
		}
	}

	void updateProxies() {
		const uint32_t chartCount = chartArray.size();
		for (uint32_t i = 0; i < chartCount; i++) {
			updateProxy(chartArray[i]);
		}
	}

	void updateProxy(ChartBuildData *chart) {
		//#pragma message(NV_FILE_LINE "TODO: Use best fit plane instead of average normal.")
		chart->planeNormal = normalizeSafe(chart->normalSum, Vector3(0), 0.0f);
		chart->centroid = chart->centroidSum / float(chart->faces.size());
	}

	bool relocateSeeds() {
		bool anySeedChanged = false;
		const uint32_t chartCount = chartArray.size();
		for (uint32_t i = 0; i < chartCount; i++) {
			if (relocateSeed(chartArray[i])) {
				anySeedChanged = true;
			}
		}
		return anySeedChanged;
	}

	bool relocateSeed(ChartBuildData *chart) {
		Vector3 centroid = computeChartCentroid(chart);
		const uint32_t N = 10; // @@ Hardcoded to 10?
		PriorityQueue bestTriangles(N);
		// Find the first N triangles that fit the proxy best.
		const uint32_t faceCount = chart->faces.size();
		for (uint32_t i = 0; i < faceCount; i++) {
			float priority = evaluateProxyFitMetric(chart, chart->faces[i]);
			bestTriangles.push(priority, chart->faces[i]);
		}
		// Of those, choose the most central triangle.
		uint32_t mostCentral;
		float maxDistance = -1;
		const uint32_t bestCount = bestTriangles.count();
		for (uint32_t i = 0; i < bestCount; i++) {
			const halfedge::Face *face = mesh->faceAt(bestTriangles.pairs[i].face);
			Vector3 faceCentroid = face->triangleCenter();
			float distance = length(centroid - faceCentroid);
			if (distance > maxDistance) {
				maxDistance = distance;
				mostCentral = bestTriangles.pairs[i].face;
			}
		}
		xaDebugAssert(maxDistance >= 0);
		// In order to prevent k-means cyles we record all the previously chosen seeds.
		uint32_t index = std::find(chart->seeds.begin(), chart->seeds.end(), mostCentral) - chart->seeds.begin();
		if (index < chart->seeds.size()) {
			// Move new seed to the end of the seed array.
			uint32_t last = chart->seeds.size() - 1;
			std::swap(chart->seeds[index], chart->seeds[last]);
			return false;
		} else {
			// Append new seed.
			chart->seeds.push_back(mostCentral);
			return true;
		}
	}

	void updatePriorities(ChartBuildData *chart) {
		// Re-evaluate candidate priorities.
		uint32_t candidateCount = chart->candidates.count();
		for (uint32_t i = 0; i < candidateCount; i++) {
			chart->candidates.pairs[i].priority = evaluatePriority(chart, chart->candidates.pairs[i].face);
			if (faceChartArray[chart->candidates.pairs[i].face] == -1) {
				updateCandidate(chart, chart->candidates.pairs[i].face, chart->candidates.pairs[i].priority);
			}
		}
		// Sort candidates.
		chart->candidates.sort();
	}

	// Evaluate combined metric.
	float evaluatePriority(ChartBuildData *chart, uint32_t face) {
		// Estimate boundary length and area:
		float newBoundaryLength = evaluateBoundaryLength(chart, face);
		float newChartArea = evaluateChartArea(chart, face);
		float F = evaluateProxyFitMetric(chart, face);
		float C = evaluateRoundnessMetric(chart, face, newBoundaryLength, newChartArea);
		float P = evaluateStraightnessMetric(chart, face);
		// Penalize faces that cross seams, reward faces that close seams or reach boundaries.
		float N = evaluateNormalSeamMetric(chart, face);
		float T = evaluateTextureSeamMetric(chart, face);
		//float R = evaluateCompletenessMetric(chart, face);
		//float D = evaluateDihedralAngleMetric(chart, face);
		// @@ Add a metric based on local dihedral angle.
		// @@ Tweaking the normal and texture seam metrics.
		// - Cause more impedance. Never cross 90 degree edges.
		// -
		float cost = float(
				options.proxyFitMetricWeight * F +
				options.roundnessMetricWeight * C +
				options.straightnessMetricWeight * P +
				options.normalSeamMetricWeight * N +
				options.textureSeamMetricWeight * T);
		// Enforce limits strictly:
		if (newChartArea > options.maxChartArea) cost = FLT_MAX;
		if (newBoundaryLength > options.maxBoundaryLength) cost = FLT_MAX;
		// Make sure normal seams are fully respected:
		if (options.normalSeamMetricWeight >= 1000 && N != 0) cost = FLT_MAX;
		xaAssert(std::isfinite(cost));
		return cost;
	}

	// Returns a value in [0-1].
	float evaluateProxyFitMetric(ChartBuildData *chart, uint32_t f) {
		const halfedge::Face *face = mesh->faceAt(f);
		Vector3 faceNormal = face->triangleNormal();
		// Use plane fitting metric for now:
		return 1 - dot(faceNormal, chart->planeNormal); // @@ normal deviations should be weighted by face area
	}

	float evaluateRoundnessMetric(ChartBuildData *chart, uint32_t /*face*/, float newBoundaryLength, float newChartArea) {
		float roundness = square(chart->boundaryLength) / chart->area;
		float newRoundness = square(newBoundaryLength) / newChartArea;
		if (newRoundness > roundness) {
			return square(newBoundaryLength) / (newChartArea * 4 * PI);
		} else {
			// Offer no impedance to faces that improve roundness.
			return 0;
		}
	}

	float evaluateStraightnessMetric(ChartBuildData *chart, uint32_t f) {
		float l_out = 0.0f;
		float l_in = 0.0f;
		const halfedge::Face *face = mesh->faceAt(f);
		for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
			const halfedge::Edge *edge = it.current();
			float l = edgeLengths[edge->id / 2];
			if (edge->isBoundary()) {
				l_out += l;
			} else {
				uint32_t neighborFaceId = edge->pair->face->id;
				if (faceChartArray[neighborFaceId] != chart->id) {
					l_out += l;
				} else {
					l_in += l;
				}
			}
		}
		xaDebugAssert(l_in != 0.0f); // Candidate face must be adjacent to chart. @@ This is not true if the input mesh has zero-length edges.
		float ratio = (l_out - l_in) / (l_out + l_in);
		return std::min(ratio, 0.0f); // Only use the straightness metric to close gaps.
	}

	float evaluateNormalSeamMetric(ChartBuildData *chart, uint32_t f) {
		float seamFactor = 0.0f;
		float totalLength = 0.0f;
		const halfedge::Face *face = mesh->faceAt(f);
		for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
			const halfedge::Edge *edge = it.current();
			if (edge->isBoundary()) {
				continue;
			}
			const uint32_t neighborFaceId = edge->pair->face->id;
			if (faceChartArray[neighborFaceId] != chart->id) {
				continue;
			}
			//float l = edge->length();
			float l = edgeLengths[edge->id / 2];
			totalLength += l;
			if (!edge->isSeam()) {
				continue;
			}
			// Make sure it's a normal seam.
			if (edge->isNormalSeam()) {
				float d0 = clamp(dot(edge->vertex->nor, edge->pair->next->vertex->nor), 0.0f, 1.0f);
				float d1 = clamp(dot(edge->next->vertex->nor, edge->pair->vertex->nor), 0.0f, 1.0f);
				l *= 1 - (d0 + d1) * 0.5f;
				seamFactor += l;
			}
		}
		if (seamFactor == 0) return 0.0f;
		return seamFactor / totalLength;
	}

	float evaluateTextureSeamMetric(ChartBuildData *chart, uint32_t f) {
		float seamLength = 0.0f;
		float totalLength = 0.0f;
		const halfedge::Face *face = mesh->faceAt(f);
		for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
			const halfedge::Edge *edge = it.current();
			if (edge->isBoundary()) {
				continue;
			}
			const uint32_t neighborFaceId = edge->pair->face->id;
			if (faceChartArray[neighborFaceId] != chart->id) {
				continue;
			}
			//float l = edge->length();
			float l = edgeLengths[edge->id / 2];
			totalLength += l;
			if (!edge->isSeam()) {
				continue;
			}
			// Make sure it's a texture seam.
			if (edge->isTextureSeam()) {
				seamLength += l;
			}
		}
		if (seamLength == 0.0f) {
			return 0.0f; // Avoid division by zero.
		}
		return seamLength / totalLength;
	}

	float evaluateChartArea(ChartBuildData *chart, uint32_t f) {
		const halfedge::Face *face = mesh->faceAt(f);
		return chart->area + faceAreas[face->id];
	}

	float evaluateBoundaryLength(ChartBuildData *chart, uint32_t f) {
		float boundaryLength = chart->boundaryLength;
		// Add new edges, subtract edges shared with the chart.
		const halfedge::Face *face = mesh->faceAt(f);
		for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
			const halfedge::Edge *edge = it.current();
			//float edgeLength = edge->length();
			float edgeLength = edgeLengths[edge->id / 2];
			if (edge->isBoundary()) {
				boundaryLength += edgeLength;
			} else {
				uint32_t neighborFaceId = edge->pair->face->id;
				if (faceChartArray[neighborFaceId] != chart->id) {
					boundaryLength += edgeLength;
				} else {
					boundaryLength -= edgeLength;
				}
			}
		}
		return std::max(0.0f, boundaryLength); // @@ Hack!
	}

	Vector3 evaluateChartNormalSum(ChartBuildData *chart, uint32_t f) {
		const halfedge::Face *face = mesh->faceAt(f);
		return chart->normalSum + face->triangleNormalAreaScaled();
	}

	Vector3 evaluateChartCentroidSum(ChartBuildData *chart, uint32_t f) {
		const halfedge::Face *face = mesh->faceAt(f);
		return chart->centroidSum + face->centroid();
	}

	Vector3 computeChartCentroid(const ChartBuildData *chart) {
		Vector3 centroid(0);
		const uint32_t faceCount = chart->faces.size();
		for (uint32_t i = 0; i < faceCount; i++) {
			const halfedge::Face *face = mesh->faceAt(chart->faces[i]);
			centroid += face->triangleCenter();
		}
		return centroid / float(faceCount);
	}

	void fillHoles(float threshold) {
		while (facesLeft > 0)
			createRandomChart(threshold);
	}

	void mergeCharts() {
		std::vector<float> sharedBoundaryLengths;
		const uint32_t chartCount = chartArray.size();
		for (int c = chartCount - 1; c >= 0; c--) {
			sharedBoundaryLengths.clear();
			sharedBoundaryLengths.resize(chartCount, 0.0f);
			ChartBuildData *chart = chartArray[c];
			float externalBoundary = 0.0f;
			const uint32_t faceCount = chart->faces.size();
			for (uint32_t i = 0; i < faceCount; i++) {
				uint32_t f = chart->faces[i];
				const halfedge::Face *face = mesh->faceAt(f);
				for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
					const halfedge::Edge *edge = it.current();
					//float l = edge->length();
					float l = edgeLengths[edge->id / 2];
					if (edge->isBoundary()) {
						externalBoundary += l;
					} else {
						uint32_t neighborFace = edge->pair->face->id;
						uint32_t neighborChart = faceChartArray[neighborFace];
						if (neighborChart != (uint32_t)c) {
							if ((edge->isSeam() && (edge->isNormalSeam() || edge->isTextureSeam())) || neighborChart == -2) {
								externalBoundary += l;
							} else {
								sharedBoundaryLengths[neighborChart] += l;
							}
						}
					}
				}
			}
			for (int cc = chartCount - 1; cc >= 0; cc--) {
				if (cc == c)
					continue;
				ChartBuildData *chart2 = chartArray[cc];
				if (chart2 == NULL)
					continue;
				if (sharedBoundaryLengths[cc] > 0.8 * std::max(0.0f, chart->boundaryLength - externalBoundary)) {
					// Try to avoid degenerate configurations.
					if (chart2->boundaryLength > sharedBoundaryLengths[cc]) {
						if (dot(chart2->planeNormal, chart->planeNormal) > -0.25) {
							mergeChart(chart2, chart, sharedBoundaryLengths[cc]);
							delete chart;
							chartArray[c] = NULL;
							break;
						}
					}
				}
				if (sharedBoundaryLengths[cc] > 0.20 * std::max(0.0f, chart->boundaryLength - externalBoundary)) {
					// Compare proxies.
					if (dot(chart2->planeNormal, chart->planeNormal) > 0) {
						mergeChart(chart2, chart, sharedBoundaryLengths[cc]);
						delete chart;
						chartArray[c] = NULL;
						break;
					}
				}
			}
		}
		// Remove deleted charts.
		for (int c = 0; c < int32_t(chartArray.size()); /*do not increment if removed*/) {
			if (chartArray[c] == NULL) {
				chartArray.erase(chartArray.begin() + c);
				// Update faceChartArray.
				const uint32_t faceCount = faceChartArray.size();
				for (uint32_t i = 0; i < faceCount; i++) {
					xaDebugAssert(faceChartArray[i] != -1);
					xaDebugAssert(faceChartArray[i] != c);
					xaDebugAssert(faceChartArray[i] <= int32_t(chartArray.size()));
					if (faceChartArray[i] > c) {
						faceChartArray[i]--;
					}
				}
			} else {
				chartArray[c]->id = c;
				c++;
			}
		}
	}

	// @@ Cleanup.
	struct Candidate {
		uint32_t face;
		ChartBuildData *chart;
		float metric;
	};

	// @@ Get N best candidates in one pass.
	const Candidate &getBestCandidate() const {
		uint32_t best = 0;
		float bestCandidateMetric = FLT_MAX;
		const uint32_t candidateCount = candidateArray.size();
		xaAssert(candidateCount > 0);
		for (uint32_t i = 0; i < candidateCount; i++) {
			const Candidate &candidate = candidateArray[i];
			if (candidate.metric < bestCandidateMetric) {
				bestCandidateMetric = candidate.metric;
				best = i;
			}
		}
		return candidateArray[best];
	}

	void removeCandidate(uint32_t f) {
		int c = faceCandidateArray[f];
		if (c != -1) {
			faceCandidateArray[f] = (uint32_t)-1;
			if (c == int(candidateArray.size() - 1)) {
				candidateArray.pop_back();
			} else {
				// Replace with last.
				candidateArray[c] = candidateArray[candidateArray.size() - 1];
				candidateArray.pop_back();
				faceCandidateArray[candidateArray[c].face] = c;
			}
		}
	}

	void updateCandidate(ChartBuildData *chart, uint32_t f, float metric) {
		if (faceCandidateArray[f] == -1) {
			const uint32_t index = candidateArray.size();
			faceCandidateArray[f] = index;
			candidateArray.resize(index + 1);
			candidateArray[index].face = f;
			candidateArray[index].chart = chart;
			candidateArray[index].metric = metric;
		} else {
			int c = faceCandidateArray[f];
			xaDebugAssert(c != -1);
			Candidate &candidate = candidateArray[c];
			xaDebugAssert(candidate.face == f);
			if (metric < candidate.metric || chart == candidate.chart) {
				candidate.metric = metric;
				candidate.chart = chart;
			}
		}
	}

	void mergeChart(ChartBuildData *owner, ChartBuildData *chart, float sharedBoundaryLength) {
		const uint32_t faceCount = chart->faces.size();
		for (uint32_t i = 0; i < faceCount; i++) {
			uint32_t f = chart->faces[i];
			xaDebugAssert(faceChartArray[f] == chart->id);
			faceChartArray[f] = owner->id;
			owner->faces.push_back(f);
		}
		// Update adjacencies?
		owner->area += chart->area;
		owner->boundaryLength += chart->boundaryLength - sharedBoundaryLength;
		owner->normalSum += chart->normalSum;
		owner->centroidSum += chart->centroidSum;
		updateProxy(owner);
	}

	uint32_t chartCount() const { return chartArray.size(); }
	const std::vector<uint32_t> &chartFaces(uint32_t i) const { return chartArray[i]->faces; }

	const halfedge::Mesh *mesh;
	uint32_t facesLeft;
	std::vector<int> faceChartArray;
	std::vector<ChartBuildData *> chartArray;
	std::vector<float> shortestPaths;
	std::vector<float> edgeLengths;
	std::vector<float> faceAreas;
	std::vector<Candidate> candidateArray; //
	std::vector<uint32_t> faceCandidateArray; // Map face index to candidate index.
	MTRand rand;
	CharterOptions options;
};

/// A chart is a connected set of faces with a certain topology (usually a disk).
class Chart {
public:
	Chart() :
			m_isDisk(false),
			m_isVertexMapped(false) {}

	void build(const halfedge::Mesh *originalMesh, const std::vector<uint32_t> &faceArray) {
		// Copy face indices.
		m_faceArray = faceArray;
		const uint32_t meshVertexCount = originalMesh->vertexCount();
		m_chartMesh.reset(new halfedge::Mesh());
		m_unifiedMesh.reset(new halfedge::Mesh());
		std::vector<uint32_t> chartMeshIndices(meshVertexCount, (uint32_t)~0);
		std::vector<uint32_t> unifiedMeshIndices(meshVertexCount, (uint32_t)~0);
		// Add vertices.
		const uint32_t faceCount = faceArray.size();
		for (uint32_t f = 0; f < faceCount; f++) {
			const halfedge::Face *face = originalMesh->faceAt(faceArray[f]);
			xaDebugAssert(face != NULL);
			for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				const halfedge::Vertex *vertex = it.current()->vertex;
				const halfedge::Vertex *unifiedVertex = vertex->firstColocal();
				if (unifiedMeshIndices[unifiedVertex->id] == ~0) {
					unifiedMeshIndices[unifiedVertex->id] = m_unifiedMesh->vertexCount();
					xaDebugAssert(vertex->pos == unifiedVertex->pos);
					m_unifiedMesh->addVertex(vertex->pos);
				}
				if (chartMeshIndices[vertex->id] == ~0) {
					chartMeshIndices[vertex->id] = m_chartMesh->vertexCount();
					m_chartToOriginalMap.push_back(vertex->original_id);
					m_chartToUnifiedMap.push_back(unifiedMeshIndices[unifiedVertex->id]);
					halfedge::Vertex *v = m_chartMesh->addVertex(vertex->pos);
					v->nor = vertex->nor;
					v->tex = vertex->tex;
				}
			}
		}
		// This is ignoring the canonical map:
		// - Is it really necessary to link colocals?
		m_chartMesh->linkColocals();
		//m_unifiedMesh->linkColocals();  // Not strictly necessary, no colocals in the unified mesh. # Wrong.
		// This check is not valid anymore, if the original mesh vertices were linked with a canonical map, then it might have
		// some colocal vertices that were unlinked. So, the unified mesh might have some duplicate vertices, because firstColocal()
		// is not guaranteed to return the same vertex for two colocal vertices.
		//xaAssert(m_chartMesh->colocalVertexCount() == m_unifiedMesh->vertexCount());
		// Is that OK? What happens in meshes were that happens? Does anything break? Apparently not...
		std::vector<uint32_t> faceIndices;
		faceIndices.reserve(7);
		// Add faces.
		for (uint32_t f = 0; f < faceCount; f++) {
			const halfedge::Face *face = originalMesh->faceAt(faceArray[f]);
			xaDebugAssert(face != NULL);
			faceIndices.clear();
			for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				const halfedge::Vertex *vertex = it.current()->vertex;
				xaDebugAssert(vertex != NULL);
				faceIndices.push_back(chartMeshIndices[vertex->id]);
			}
			m_chartMesh->addFace(faceIndices);
			faceIndices.clear();
			for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				const halfedge::Vertex *vertex = it.current()->vertex;
				xaDebugAssert(vertex != NULL);
				vertex = vertex->firstColocal();
				faceIndices.push_back(unifiedMeshIndices[vertex->id]);
			}
			m_unifiedMesh->addFace(faceIndices);
		}
		m_chartMesh->linkBoundary();
		m_unifiedMesh->linkBoundary();
		//exportMesh(m_unifiedMesh.ptr(), "debug_input.obj");
		if (m_unifiedMesh->splitBoundaryEdges()) {
			m_unifiedMesh.reset(halfedge::unifyVertices(m_unifiedMesh.get()));
		}
		//exportMesh(m_unifiedMesh.ptr(), "debug_split.obj");
		// Closing the holes is not always the best solution and does not fix all the problems.
		// We need to do some analysis of the holes and the genus to:
		// - Find cuts that reduce genus.
		// - Find cuts to connect holes.
		// - Use minimal spanning trees or seamster.
		if (!closeHoles()) {
			/*static int pieceCount = 0;
			StringBuilder fileName;
			fileName.format("debug_hole_%d.obj", pieceCount++);
			exportMesh(m_unifiedMesh.ptr(), fileName.str());*/
		}
		m_unifiedMesh.reset(halfedge::triangulate(m_unifiedMesh.get()));
		//exportMesh(m_unifiedMesh.ptr(), "debug_triangulated.obj");
		// Analyze chart topology.
		halfedge::MeshTopology topology(m_unifiedMesh.get());
		m_isDisk = topology.isDisk();
	}

	void buildVertexMap(const halfedge::Mesh *originalMesh, const std::vector<uint32_t> &unchartedMaterialArray) {
		xaAssert(m_chartMesh.get() == NULL && m_unifiedMesh.get() == NULL);
		m_isVertexMapped = true;
		// Build face indices.
		m_faceArray.clear();
		const uint32_t meshFaceCount = originalMesh->faceCount();
		for (uint32_t f = 0; f < meshFaceCount; f++) {
			const halfedge::Face *face = originalMesh->faceAt(f);
			if (std::find(unchartedMaterialArray.begin(), unchartedMaterialArray.end(), face->material) != unchartedMaterialArray.end()) {
				m_faceArray.push_back(f);
			}
		}
		const uint32_t faceCount = m_faceArray.size();
		if (faceCount == 0) {
			return;
		}
		// @@ The chartMesh construction is basically the same as with regular charts, don't duplicate!
		const uint32_t meshVertexCount = originalMesh->vertexCount();
		m_chartMesh.reset(new halfedge::Mesh());
		std::vector<uint32_t> chartMeshIndices(meshVertexCount, (uint32_t)~0);
		// Vertex map mesh only has disconnected vertices.
		for (uint32_t f = 0; f < faceCount; f++) {
			const halfedge::Face *face = originalMesh->faceAt(m_faceArray[f]);
			xaDebugAssert(face != NULL);
			for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				const halfedge::Vertex *vertex = it.current()->vertex;
				if (chartMeshIndices[vertex->id] == ~0) {
					chartMeshIndices[vertex->id] = m_chartMesh->vertexCount();
					m_chartToOriginalMap.push_back(vertex->original_id);
					halfedge::Vertex *v = m_chartMesh->addVertex(vertex->pos);
					v->nor = vertex->nor;
					v->tex = vertex->tex; // @@ Not necessary.
				}
			}
		}
		// @@ Link colocals using the original mesh canonical map? Build canonical map on the fly? Do we need to link colocals at all for this?
		//m_chartMesh->linkColocals();
		std::vector<uint32_t> faceIndices;
		faceIndices.reserve(7);
		// Add faces.
		for (uint32_t f = 0; f < faceCount; f++) {
			const halfedge::Face *face = originalMesh->faceAt(m_faceArray[f]);
			xaDebugAssert(face != NULL);
			faceIndices.clear();
			for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				const halfedge::Vertex *vertex = it.current()->vertex;
				xaDebugAssert(vertex != NULL);
				xaDebugAssert(chartMeshIndices[vertex->id] != ~0);
				faceIndices.push_back(chartMeshIndices[vertex->id]);
			}
			halfedge::Face *new_face = m_chartMesh->addFace(faceIndices);
			xaDebugAssert(new_face != NULL);
#ifdef NDEBUG
			new_face = NULL; // silence unused parameter warning
#endif
		}
		m_chartMesh->linkBoundary();
		const uint32_t chartVertexCount = m_chartMesh->vertexCount();
		Box bounds;
		bounds.clearBounds();
		for (uint32_t i = 0; i < chartVertexCount; i++) {
			halfedge::Vertex *vertex = m_chartMesh->vertexAt(i);
			bounds.addPointToBounds(vertex->pos);
		}
		ProximityGrid grid;
		grid.init(bounds, chartVertexCount);
		for (uint32_t i = 0; i < chartVertexCount; i++) {
			halfedge::Vertex *vertex = m_chartMesh->vertexAt(i);
			grid.add(vertex->pos, i);
		}
		uint32_t texelCount = 0;
		const float positionThreshold = 0.01f;
		const float normalThreshold = 0.01f;
		uint32_t verticesVisited = 0;
		uint32_t cellsVisited = 0;
		std::vector<int> vertexIndexArray(chartVertexCount, -1); // Init all indices to -1.
		// Traverse vertices in morton order. @@ It may be more interesting to sort them based on orientation.
		const uint32_t cellCodeCount = grid.mortonCount();
		for (uint32_t cellCode = 0; cellCode < cellCodeCount; cellCode++) {
			int cell = grid.mortonIndex(cellCode);
			if (cell < 0) continue;
			cellsVisited++;
			const std::vector<uint32_t> &indexArray = grid.cellArray[cell].indexArray;
			for (uint32_t i = 0; i < indexArray.size(); i++) {
				uint32_t idx = indexArray[i];
				halfedge::Vertex *vertex = m_chartMesh->vertexAt(idx);
				xaDebugAssert(vertexIndexArray[idx] == -1);
				std::vector<uint32_t> neighbors;
				grid.gather(vertex->pos, positionThreshold, /*ref*/ neighbors);
				// Compare against all nearby vertices, cluster greedily.
				for (uint32_t j = 0; j < neighbors.size(); j++) {
					uint32_t otherIdx = neighbors[j];
					if (vertexIndexArray[otherIdx] != -1) {
						halfedge::Vertex *otherVertex = m_chartMesh->vertexAt(otherIdx);
						if (distance(vertex->pos, otherVertex->pos) < positionThreshold &&
								distance(vertex->nor, otherVertex->nor) < normalThreshold) {
							vertexIndexArray[idx] = vertexIndexArray[otherIdx];
							break;
						}
					}
				}
				// If index not assigned, assign new one.
				if (vertexIndexArray[idx] == -1) {
					vertexIndexArray[idx] = texelCount++;
				}
				verticesVisited++;
			}
		}
		xaDebugAssert(cellsVisited == grid.cellArray.size());
		xaDebugAssert(verticesVisited == chartVertexCount);
		vertexMapWidth = ftoi_ceil(sqrtf(float(texelCount)));
		vertexMapWidth = (vertexMapWidth + 3) & ~3; // Width aligned to 4.
		vertexMapHeight = vertexMapWidth == 0 ? 0 : (texelCount + vertexMapWidth - 1) / vertexMapWidth;
		//vertexMapHeight = (vertexMapHeight + 3) & ~3;                           // Height aligned to 4.
		xaDebugAssert(vertexMapWidth >= vertexMapHeight);
		xaPrint("Reduced vertex count from %d to %d.\n", chartVertexCount, texelCount);
		// Lay down the clustered vertices in morton order.
		std::vector<uint32_t> texelCodes(texelCount);
		// For each texel, assign one morton code.
		uint32_t texelCode = 0;
		for (uint32_t i = 0; i < texelCount; i++) {
			uint32_t x, y;
			do {
				x = morton::decodeMorton2X(texelCode);
				y = morton::decodeMorton2Y(texelCode);
				texelCode++;
			} while (x >= uint32_t(vertexMapWidth) || y >= uint32_t(vertexMapHeight));
			texelCodes[i] = texelCode - 1;
		}
		for (uint32_t i = 0; i < chartVertexCount; i++) {
			halfedge::Vertex *vertex = m_chartMesh->vertexAt(i);
			int idx = vertexIndexArray[i];
			if (idx != -1) {
				uint32_t tc = texelCodes[idx];
				uint32_t x = morton::decodeMorton2X(tc);
				uint32_t y = morton::decodeMorton2Y(tc);
				vertex->tex.x = float(x);
				vertex->tex.y = float(y);
			}
		}
	}

	bool closeHoles() {
		xaDebugAssert(!m_isVertexMapped);
		std::vector<halfedge::Edge *> boundaryEdges;
		getBoundaryEdges(m_unifiedMesh.get(), boundaryEdges);
		uint32_t boundaryCount = boundaryEdges.size();
		if (boundaryCount <= 1) {
			// Nothing to close.
			return true;
		}
		// Compute lengths and areas.
		std::vector<float> boundaryLengths;
		for (uint32_t i = 0; i < boundaryCount; i++) {
			const halfedge::Edge *startEdge = boundaryEdges[i];
			xaAssert(startEdge->face == NULL);
			//float boundaryEdgeCount = 0;
			float boundaryLength = 0.0f;
			//Vector3 boundaryCentroid(zero);
			const halfedge::Edge *edge = startEdge;
			do {
				Vector3 t0 = edge->from()->pos;
				Vector3 t1 = edge->to()->pos;
				//boundaryEdgeCount++;
				boundaryLength += length(t1 - t0);
				//boundaryCentroid += edge->vertex()->pos;
				edge = edge->next;
			} while (edge != startEdge);
			boundaryLengths.push_back(boundaryLength);
			//boundaryCentroids.append(boundaryCentroid / boundaryEdgeCount);
		}
		// Find disk boundary.
		uint32_t diskBoundary = 0;
		float maxLength = boundaryLengths[0];
		for (uint32_t i = 1; i < boundaryCount; i++) {
			if (boundaryLengths[i] > maxLength) {
				maxLength = boundaryLengths[i];
				diskBoundary = i;
			}
		}
		// Close holes.
		for (uint32_t i = 0; i < boundaryCount; i++) {
			if (diskBoundary == i) {
				// Skip disk boundary.
				continue;
			}
			halfedge::Edge *startEdge = boundaryEdges[i];
			xaDebugAssert(startEdge != NULL);
			xaDebugAssert(startEdge->face == NULL);
			std::vector<halfedge::Vertex *> vertexLoop;
			std::vector<halfedge::Edge *> edgeLoop;
			halfedge::Edge *edge = startEdge;
			do {
				halfedge::Vertex *vertex = edge->next->vertex; // edge->to()
				uint32_t j;
				for (j = 0; j < vertexLoop.size(); j++) {
					if (vertex->isColocal(vertexLoop[j])) {
						break;
					}
				}
				bool isCrossing = (j != vertexLoop.size());
				if (isCrossing) {
					halfedge::Edge *prev = edgeLoop[j]; // Previous edge before the loop.
					halfedge::Edge *next = edge->next; // Next edge after the loop.
					xaDebugAssert(prev->to()->isColocal(next->from()));
					// Close loop.
					edgeLoop.push_back(edge);
					closeLoop(j + 1, edgeLoop);
					// Link boundary loop.
					prev->setNext(next);
					vertex->setEdge(next);
					// Start over again.
					vertexLoop.clear();
					edgeLoop.clear();
					edge = startEdge;
					vertex = edge->to();
				}
				vertexLoop.push_back(vertex);
				edgeLoop.push_back(edge);
				edge = edge->next;
			} while (edge != startEdge);
			closeLoop(0, edgeLoop);
		}
		getBoundaryEdges(m_unifiedMesh.get(), boundaryEdges);
		boundaryCount = boundaryEdges.size();
		xaDebugAssert(boundaryCount == 1);
		return boundaryCount == 1;
	}

	bool isDisk() const {
		return m_isDisk;
	}
	bool isVertexMapped() const {
		return m_isVertexMapped;
	}

	uint32_t vertexCount() const {
		return m_chartMesh->vertexCount();
	}
	uint32_t colocalVertexCount() const {
		return m_unifiedMesh->vertexCount();
	}

	uint32_t faceCount() const {
		return m_faceArray.size();
	}
	uint32_t faceAt(uint32_t i) const {
		return m_faceArray[i];
	}

	const halfedge::Mesh *chartMesh() const {
		return m_chartMesh.get();
	}
	halfedge::Mesh *chartMesh() {
		return m_chartMesh.get();
	}
	const halfedge::Mesh *unifiedMesh() const {
		return m_unifiedMesh.get();
	}
	halfedge::Mesh *unifiedMesh() {
		return m_unifiedMesh.get();
	}

	//uint32_t vertexIndex(uint32_t i) const { return m_vertexIndexArray[i]; }

	uint32_t mapChartVertexToOriginalVertex(uint32_t i) const {
		return m_chartToOriginalMap[i];
	}
	uint32_t mapChartVertexToUnifiedVertex(uint32_t i) const {
		return m_chartToUnifiedMap[i];
	}

	const std::vector<uint32_t> &faceArray() const {
		return m_faceArray;
	}

	// Transfer parameterization from unified mesh to chart mesh.
	void transferParameterization() {
		xaDebugAssert(!m_isVertexMapped);
		uint32_t vertexCount = m_chartMesh->vertexCount();
		for (uint32_t v = 0; v < vertexCount; v++) {
			halfedge::Vertex *vertex = m_chartMesh->vertexAt(v);
			halfedge::Vertex *unifiedVertex = m_unifiedMesh->vertexAt(mapChartVertexToUnifiedVertex(v));
			vertex->tex = unifiedVertex->tex;
		}
	}

	float computeSurfaceArea() const {
		return halfedge::computeSurfaceArea(m_chartMesh.get()) * scale;
	}

	float computeParametricArea() const {
		// This only makes sense in parameterized meshes.
		xaDebugAssert(m_isDisk);
		xaDebugAssert(!m_isVertexMapped);
		return halfedge::computeParametricArea(m_chartMesh.get());
	}

	Vector2 computeParametricBounds() const {
		// This only makes sense in parameterized meshes.
		xaDebugAssert(m_isDisk);
		xaDebugAssert(!m_isVertexMapped);
		Box bounds;
		bounds.clearBounds();
		uint32_t vertexCount = m_chartMesh->vertexCount();
		for (uint32_t v = 0; v < vertexCount; v++) {
			halfedge::Vertex *vertex = m_chartMesh->vertexAt(v);
			bounds.addPointToBounds(Vector3(vertex->tex, 0));
		}
		return bounds.extents().xy();
	}

	float scale = 1.0f;
	uint32_t vertexMapWidth;
	uint32_t vertexMapHeight;
	bool blockAligned = true;

private:
	bool closeLoop(uint32_t start, const std::vector<halfedge::Edge *> &loop) {
		const uint32_t vertexCount = loop.size() - start;
		xaDebugAssert(vertexCount >= 3);
		if (vertexCount < 3) return false;
		xaDebugAssert(loop[start]->vertex->isColocal(loop[start + vertexCount - 1]->to()));
		// If the hole is planar, then we add a single face that will be properly triangulated later.
		// If the hole is not planar, we add a triangle fan with a vertex at the hole centroid.
		// This is still a bit of a hack. There surely are better hole filling algorithms out there.
		std::vector<Vector3> points(vertexCount);
		for (uint32_t i = 0; i < vertexCount; i++) {
			points[i] = loop[start + i]->vertex->pos;
		}
		bool isPlanar = Fit::isPlanar(vertexCount, points.data());
		if (isPlanar) {
			// Add face and connect edges.
			halfedge::Face *face = m_unifiedMesh->addFace();
			for (uint32_t i = 0; i < vertexCount; i++) {
				halfedge::Edge *edge = loop[start + i];
				edge->face = face;
				edge->setNext(loop[start + (i + 1) % vertexCount]);
			}
			face->edge = loop[start];
			xaDebugAssert(face->isValid());
		} else {
			// If the polygon is not planar, we just cross our fingers, and hope this will work:
			// Compute boundary centroid:
			Vector3 centroidPos(0);
			for (uint32_t i = 0; i < vertexCount; i++) {
				centroidPos += points[i];
			}
			centroidPos *= (1.0f / vertexCount);
			halfedge::Vertex *centroid = m_unifiedMesh->addVertex(centroidPos);
			// Add one pair of edges for each boundary vertex.
			for (uint32_t j = vertexCount - 1, i = 0; i < vertexCount; j = i++) {
				halfedge::Face *face = m_unifiedMesh->addFace(centroid->id, loop[start + j]->vertex->id, loop[start + i]->vertex->id);
				xaDebugAssert(face != NULL);
#ifdef NDEBUG
				face = NULL; // silence unused parameter warning
#endif
			}
		}
		return true;
	}

	static void getBoundaryEdges(halfedge::Mesh *mesh, std::vector<halfedge::Edge *> &boundaryEdges) {
		xaDebugAssert(mesh != NULL);
		const uint32_t edgeCount = mesh->edgeCount();
		BitArray bitFlags(edgeCount);
		bitFlags.clearAll();
		boundaryEdges.clear();
		// Search for boundary edges. Mark all the edges that belong to the same boundary.
		for (uint32_t e = 0; e < edgeCount; e++) {
			halfedge::Edge *startEdge = mesh->edgeAt(e);
			if (startEdge != NULL && startEdge->isBoundary() && bitFlags.bitAt(e) == false) {
				xaDebugAssert(startEdge->face != NULL);
				xaDebugAssert(startEdge->pair->face == NULL);
				startEdge = startEdge->pair;
				const halfedge::Edge *edge = startEdge;
				do {
					xaDebugAssert(edge->face == NULL);
					xaDebugAssert(bitFlags.bitAt(edge->id / 2) == false);
					bitFlags.setBitAt(edge->id / 2);
					edge = edge->next;
				} while (startEdge != edge);
				boundaryEdges.push_back(startEdge);
			}
		}
	}

	// Chart mesh.
	std::auto_ptr<halfedge::Mesh> m_chartMesh;

	std::auto_ptr<halfedge::Mesh> m_unifiedMesh;
	bool m_isDisk;
	bool m_isVertexMapped;

	// List of faces of the original mesh that belong to this chart.
	std::vector<uint32_t> m_faceArray;

	// Map vertices of the chart mesh to vertices of the original mesh.
	std::vector<uint32_t> m_chartToOriginalMap;

	std::vector<uint32_t> m_chartToUnifiedMap;
};

// Estimate quality of existing parameterization.
class ParameterizationQuality {
public:
	ParameterizationQuality() {
		m_totalTriangleCount = 0;
		m_flippedTriangleCount = 0;
		m_zeroAreaTriangleCount = 0;
		m_parametricArea = 0.0f;
		m_geometricArea = 0.0f;
		m_stretchMetric = 0.0f;
		m_maxStretchMetric = 0.0f;
		m_conformalMetric = 0.0f;
		m_authalicMetric = 0.0f;
	}

	ParameterizationQuality(const halfedge::Mesh *mesh) {
		xaDebugAssert(mesh != NULL);
		m_totalTriangleCount = 0;
		m_flippedTriangleCount = 0;
		m_zeroAreaTriangleCount = 0;
		m_parametricArea = 0.0f;
		m_geometricArea = 0.0f;
		m_stretchMetric = 0.0f;
		m_maxStretchMetric = 0.0f;
		m_conformalMetric = 0.0f;
		m_authalicMetric = 0.0f;
		const uint32_t faceCount = mesh->faceCount();
		for (uint32_t f = 0; f < faceCount; f++) {
			const halfedge::Face *face = mesh->faceAt(f);
			const halfedge::Vertex *vertex0 = NULL;
			Vector3 p[3];
			Vector2 t[3];
			for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				const halfedge::Edge *edge = it.current();
				if (vertex0 == NULL) {
					vertex0 = edge->vertex;
					p[0] = vertex0->pos;
					t[0] = vertex0->tex;
				} else if (edge->to() != vertex0) {
					p[1] = edge->from()->pos;
					p[2] = edge->to()->pos;
					t[1] = edge->from()->tex;
					t[2] = edge->to()->tex;
					processTriangle(p, t);
				}
			}
		}
		if (m_flippedTriangleCount + m_zeroAreaTriangleCount == faceCount) {
			// If all triangles are flipped, then none is.
			m_flippedTriangleCount = 0;
		}
		xaDebugAssert(std::isfinite(m_parametricArea) && m_parametricArea >= 0);
		xaDebugAssert(std::isfinite(m_geometricArea) && m_geometricArea >= 0);
		xaDebugAssert(std::isfinite(m_stretchMetric));
		xaDebugAssert(std::isfinite(m_maxStretchMetric));
		xaDebugAssert(std::isfinite(m_conformalMetric));
		xaDebugAssert(std::isfinite(m_authalicMetric));
	}

	bool isValid() const {
		return m_flippedTriangleCount == 0; // @@ Does not test for self-overlaps.
	}

	float rmsStretchMetric() const {
		if (m_geometricArea == 0) return 0.0f;
		float normFactor = sqrtf(m_parametricArea / m_geometricArea);
		return sqrtf(m_stretchMetric / m_geometricArea) * normFactor;
	}

	float maxStretchMetric() const {
		if (m_geometricArea == 0) return 0.0f;
		float normFactor = sqrtf(m_parametricArea / m_geometricArea);
		return m_maxStretchMetric * normFactor;
	}

	float rmsConformalMetric() const {
		if (m_geometricArea == 0) return 0.0f;
		return sqrtf(m_conformalMetric / m_geometricArea);
	}

	float maxAuthalicMetric() const {
		if (m_geometricArea == 0) return 0.0f;
		return sqrtf(m_authalicMetric / m_geometricArea);
	}

	void operator+=(const ParameterizationQuality &pq) {
		m_totalTriangleCount += pq.m_totalTriangleCount;
		m_flippedTriangleCount += pq.m_flippedTriangleCount;
		m_zeroAreaTriangleCount += pq.m_zeroAreaTriangleCount;
		m_parametricArea += pq.m_parametricArea;
		m_geometricArea += pq.m_geometricArea;
		m_stretchMetric += pq.m_stretchMetric;
		m_maxStretchMetric = std::max(m_maxStretchMetric, pq.m_maxStretchMetric);
		m_conformalMetric += pq.m_conformalMetric;
		m_authalicMetric += pq.m_authalicMetric;
	}

private:
	void processTriangle(Vector3 q[3], Vector2 p[3]) {
		m_totalTriangleCount++;
		// Evaluate texture stretch metric. See:
		// - "Texture Mapping Progressive Meshes", Sander, Snyder, Gortler & Hoppe
		// - "Mesh Parameterization: Theory and Practice", Siggraph'07 Course Notes, Hormann, Levy & Sheffer.
		float t1 = p[0].x;
		float s1 = p[0].y;
		float t2 = p[1].x;
		float s2 = p[1].y;
		float t3 = p[2].x;
		float s3 = p[2].y;
		float geometricArea = length(cross(q[1] - q[0], q[2] - q[0])) / 2;
		float parametricArea = ((s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)) / 2;
		if (isZero(parametricArea)) {
			m_zeroAreaTriangleCount++;
			return;
		}
		Vector3 Ss = (q[0] * (t2 - t3) + q[1] * (t3 - t1) + q[2] * (t1 - t2)) / (2 * parametricArea);
		Vector3 St = (q[0] * (s3 - s2) + q[1] * (s1 - s3) + q[2] * (s2 - s1)) / (2 * parametricArea);
		float a = dot(Ss, Ss); // E
		float b = dot(Ss, St); // F
		float c = dot(St, St); // G
		// Compute eigen-values of the first fundamental form:
		float sigma1 = sqrtf(0.5f * std::max(0.0f, a + c - sqrtf(square(a - c) + 4 * square(b)))); // gamma uppercase, min eigenvalue.
		float sigma2 = sqrtf(0.5f * std::max(0.0f, a + c + sqrtf(square(a - c) + 4 * square(b)))); // gamma lowercase, max eigenvalue.
		xaAssert(sigma2 >= sigma1);
		// isometric: sigma1 = sigma2 = 1
		// conformal: sigma1 / sigma2 = 1
		// authalic: sigma1 * sigma2 = 1
		float rmsStretch = sqrtf((a + c) * 0.5f);
		float rmsStretch2 = sqrtf((square(sigma1) + square(sigma2)) * 0.5f);
		xaDebugAssert(equal(rmsStretch, rmsStretch2, 0.01f));
#ifdef NDEBUG
		rmsStretch2 = 0; // silence unused parameter warning
#endif
		if (parametricArea < 0.0f) {
			// Count flipped triangles.
			m_flippedTriangleCount++;
			parametricArea = fabsf(parametricArea);
		}
		m_stretchMetric += square(rmsStretch) * geometricArea;
		m_maxStretchMetric = std::max(m_maxStretchMetric, sigma2);
		if (!isZero(sigma1, 0.000001f)) {
			// sigma1 is zero when geometricArea is zero.
			m_conformalMetric += (sigma2 / sigma1) * geometricArea;
		}
		m_authalicMetric += (sigma1 * sigma2) * geometricArea;
		// Accumulate total areas.
		m_geometricArea += geometricArea;
		m_parametricArea += parametricArea;
		//triangleConformalEnergy(q, p);
	}

	uint32_t m_totalTriangleCount;
	uint32_t m_flippedTriangleCount;
	uint32_t m_zeroAreaTriangleCount;
	float m_parametricArea;
	float m_geometricArea;
	float m_stretchMetric;
	float m_maxStretchMetric;
	float m_conformalMetric;
	float m_authalicMetric;
};

// Set of charts corresponding to a single mesh.
class MeshCharts {
public:
	MeshCharts(const halfedge::Mesh *mesh) :
			m_mesh(mesh) {}

	~MeshCharts() {
		for (size_t i = 0; i < m_chartArray.size(); i++)
			delete m_chartArray[i];
	}

	uint32_t chartCount() const {
		return m_chartArray.size();
	}
	uint32_t vertexCount() const {
		return m_totalVertexCount;
	}

	const Chart *chartAt(uint32_t i) const {
		return m_chartArray[i];
	}
	Chart *chartAt(uint32_t i) {
		return m_chartArray[i];
	}

	// Extract the charts of the input mesh.
	void extractCharts() {
		const uint32_t faceCount = m_mesh->faceCount();
		int first = 0;
		std::vector<uint32_t> queue;
		queue.reserve(faceCount);
		BitArray bitFlags(faceCount);
		bitFlags.clearAll();
		for (uint32_t f = 0; f < faceCount; f++) {
			if (bitFlags.bitAt(f) == false) {
				// Start new patch. Reset queue.
				first = 0;
				queue.clear();
				queue.push_back(f);
				bitFlags.setBitAt(f);
				while (first != (int)queue.size()) {
					const halfedge::Face *face = m_mesh->faceAt(queue[first]);
					// Visit face neighbors of queue[first]
					for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
						const halfedge::Edge *edge = it.current();
						xaDebugAssert(edge->pair != NULL);
						if (!edge->isBoundary() && /*!edge->isSeam()*/
								//!(edge->from()->tex() != edge->pair()->to()->tex() || edge->to()->tex() != edge->pair()->from()->tex()))
								!(edge->from() != edge->pair->to() || edge->to() != edge->pair->from())) { // Preserve existing seams (not just texture seams).
							const halfedge::Face *neighborFace = edge->pair->face;
							xaDebugAssert(neighborFace != NULL);
							if (bitFlags.bitAt(neighborFace->id) == false) {
								queue.push_back(neighborFace->id);
								bitFlags.setBitAt(neighborFace->id);
							}
						}
					}
					first++;
				}
				Chart *chart = new Chart();
				chart->build(m_mesh, queue);
				m_chartArray.push_back(chart);
			}
		}
	}

	/*
	Compute charts using a simple segmentation algorithm.

	LSCM:
	- identify sharp features using local dihedral angles.
	- identify seed faces farthest from sharp features.
	- grow charts from these seeds.

	MCGIM:
	- phase 1: chart growth
	  - grow all charts simultaneously using dijkstra search on the dual graph of the mesh.
	  - graph edges are weighted based on planarity metric.
	  - metric uses distance to global chart normal.
	  - terminate when all faces have been assigned.
	- phase 2: seed computation:
	  - place new seed of the chart at the most interior face.
	  - most interior is evaluated using distance metric only.

	- method repeates the two phases, until the location of the seeds does not change.
	  - cycles are detected by recording all the previous seeds and chartification terminates.

	D-Charts:

	- Uniaxial conic metric:
	  - N_c = axis of the generalized cone that best fits the chart. (cone can a be cylinder or a plane).
	  - omega_c = angle between the face normals and the axis.
	  - Fitting error between chart C and tringle t: F(c,t) = (N_c*n_t - cos(omega_c))^2

	- Compactness metrics:
	  - Roundness:
		- C(c,t) = pi * D(S_c,t)^2 / A_c
		- S_c = chart seed.
		- D(S_c,t) = length of the shortest path inside the chart betwen S_c and t.
		- A_c = chart area.
	  - Straightness:
		- P(c,t) = l_out(c,t) / l_in(c,t)
		- l_out(c,t) = lenght of the edges not shared between C and t.
		- l_in(c,t) = lenght of the edges shared between C and t.

	- Combined metric:
	  - Cost(c,t) = F(c,t)^alpha + C(c,t)^beta + P(c,t)^gamma
	  - alpha = 1, beta = 0.7, gamma = 0.5

	Our basic approach:
	- Just one iteration of k-means?
	- Avoid dijkstra by greedily growing charts until a threshold is met. Increase threshold and repeat until no faces left.
	- If distortion metric is too high, split chart, add two seeds.
	- If chart size is low, try removing chart.

	Postprocess:
	- If topology is not disk:
	  - Fill holes, if new faces fit proxy.
	  - Find best cut, otherwise.
	- After parameterization:
	  - If boundary self-intersects:
		- cut chart along the closest two diametral boundary vertices, repeat parametrization.
		- what if the overlap is on an appendix? How do we find that out and cut appropiately?
		  - emphasize roundness metrics to prevent those cases.
	  - If interior self-overlaps: preserve boundary parameterization and use mean-value map.
	*/
	void computeCharts(const CharterOptions &options, const std::vector<uint32_t> &unchartedMaterialArray) {
		Chart *vertexMap = NULL;
		if (unchartedMaterialArray.size() != 0) {
			vertexMap = new Chart();
			vertexMap->buildVertexMap(m_mesh, unchartedMaterialArray);
			if (vertexMap->faceCount() == 0) {
				delete vertexMap;
				vertexMap = NULL;
			}
		}
		AtlasBuilder builder(m_mesh);
		if (vertexMap != NULL) {
			// Mark faces that do not need to be charted.
			builder.markUnchartedFaces(vertexMap->faceArray());
			m_chartArray.push_back(vertexMap);
		}
		if (builder.facesLeft != 0) {
			// Tweak these values:
			const float maxThreshold = 2;
			const uint32_t growFaceCount = 32;
			const uint32_t maxIterations = 4;
			builder.options = options;
			//builder.options.proxyFitMetricWeight *= 0.75; // relax proxy fit weight during initial seed placement.
			//builder.options.roundnessMetricWeight = 0;
			//builder.options.straightnessMetricWeight = 0;
			// This seems a reasonable estimate.
			uint32_t maxSeedCount = std::max(6U, builder.facesLeft);
			// Create initial charts greedely.
			xaPrint("### Placing seeds\n");
			builder.placeSeeds(maxThreshold, maxSeedCount);
			xaPrint("###   Placed %d seeds (max = %d)\n", builder.chartCount(), maxSeedCount);
			builder.updateProxies();
			builder.mergeCharts();
#if 1
			xaPrint("### Relocating seeds\n");
			builder.relocateSeeds();
			xaPrint("### Reset charts\n");
			builder.resetCharts();
			if (vertexMap != NULL) {
				builder.markUnchartedFaces(vertexMap->faceArray());
			}
			builder.options = options;
			xaPrint("### Growing charts\n");
			// Restart process growing charts in parallel.
			uint32_t iteration = 0;
			while (true) {
				if (!builder.growCharts(maxThreshold, growFaceCount)) {
					xaPrint("### Can't grow anymore\n");
					// If charts cannot grow more: fill holes, merge charts, relocate seeds and start new iteration.
					xaPrint("### Filling holes\n");
					builder.fillHoles(maxThreshold);
					xaPrint("###   Using %d charts now\n", builder.chartCount());
					builder.updateProxies();
					xaPrint("### Merging charts\n");
					builder.mergeCharts();
					xaPrint("###   Using %d charts now\n", builder.chartCount());
					xaPrint("### Reseeding\n");
					if (!builder.relocateSeeds()) {
						xaPrint("### Cannot relocate seeds anymore\n");
						// Done!
						break;
					}
					if (iteration == maxIterations) {
						xaPrint("### Reached iteration limit\n");
						break;
					}
					iteration++;
					xaPrint("### Reset charts\n");
					builder.resetCharts();
					if (vertexMap != NULL) {
						builder.markUnchartedFaces(vertexMap->faceArray());
					}
					xaPrint("### Growing charts\n");
				}
			};
#endif
			// Make sure no holes are left!
			xaDebugAssert(builder.facesLeft == 0);
			const uint32_t chartCount = builder.chartArray.size();
			for (uint32_t i = 0; i < chartCount; i++) {
				Chart *chart = new Chart();
				m_chartArray.push_back(chart);
				chart->build(m_mesh, builder.chartFaces(i));
			}
		}
		const uint32_t chartCount = m_chartArray.size();
		// Build face indices.
		m_faceChart.resize(m_mesh->faceCount());
		m_faceIndex.resize(m_mesh->faceCount());
		for (uint32_t i = 0; i < chartCount; i++) {
			const Chart *chart = m_chartArray[i];
			const uint32_t faceCount = chart->faceCount();
			for (uint32_t f = 0; f < faceCount; f++) {
				uint32_t idx = chart->faceAt(f);
				m_faceChart[idx] = i;
				m_faceIndex[idx] = f;
			}
		}
		// Build an exclusive prefix sum of the chart vertex counts.
		m_chartVertexCountPrefixSum.resize(chartCount);
		if (chartCount > 0) {
			m_chartVertexCountPrefixSum[0] = 0;
			for (uint32_t i = 1; i < chartCount; i++) {
				const Chart *chart = m_chartArray[i - 1];
				m_chartVertexCountPrefixSum[i] = m_chartVertexCountPrefixSum[i - 1] + chart->vertexCount();
			}
			m_totalVertexCount = m_chartVertexCountPrefixSum[chartCount - 1] + m_chartArray[chartCount - 1]->vertexCount();
		} else {
			m_totalVertexCount = 0;
		}
	}

	void parameterizeCharts() {
		ParameterizationQuality globalParameterizationQuality;
		// Parameterize the charts.
		uint32_t diskCount = 0;
		const uint32_t chartCount = m_chartArray.size();
		for (uint32_t i = 0; i < chartCount; i++) {
			Chart *chart = m_chartArray[i];

			bool isValid = false;

			if (chart->isVertexMapped()) {
				continue;
			}

			if (chart->isDisk()) {
				diskCount++;
				ParameterizationQuality chartParameterizationQuality;
				if (chart->faceCount() == 1) {
					computeSingleFaceMap(chart->unifiedMesh());
					chartParameterizationQuality = ParameterizationQuality(chart->unifiedMesh());
				} else {
					computeOrthogonalProjectionMap(chart->unifiedMesh());
					ParameterizationQuality orthogonalQuality(chart->unifiedMesh());
					computeLeastSquaresConformalMap(chart->unifiedMesh());
					ParameterizationQuality lscmQuality(chart->unifiedMesh());
					chartParameterizationQuality = lscmQuality;
				}
				isValid = chartParameterizationQuality.isValid();
				if (!isValid) {
					xaPrint("*** Invalid parameterization.\n");
				}
				// @@ Check that parameterization quality is above a certain threshold.
				// @@ Detect boundary self-intersections.
				globalParameterizationQuality += chartParameterizationQuality;
			}

			// Transfer parameterization from unified mesh to chart mesh.
			chart->transferParameterization();
		}
		xaPrint("  Parameterized %d/%d charts.\n", diskCount, chartCount);
		xaPrint("  RMS stretch metric: %f\n", globalParameterizationQuality.rmsStretchMetric());
		xaPrint("  MAX stretch metric: %f\n", globalParameterizationQuality.maxStretchMetric());
		xaPrint("  RMS conformal metric: %f\n", globalParameterizationQuality.rmsConformalMetric());
		xaPrint("  RMS authalic metric: %f\n", globalParameterizationQuality.maxAuthalicMetric());
	}

	uint32_t faceChartAt(uint32_t i) const {
		return m_faceChart[i];
	}
	uint32_t faceIndexWithinChartAt(uint32_t i) const {
		return m_faceIndex[i];
	}

	uint32_t vertexCountBeforeChartAt(uint32_t i) const {
		return m_chartVertexCountPrefixSum[i];
	}

private:
	const halfedge::Mesh *m_mesh;

	std::vector<Chart *> m_chartArray;

	std::vector<uint32_t> m_chartVertexCountPrefixSum;
	uint32_t m_totalVertexCount;

	std::vector<uint32_t> m_faceChart; // the chart of every face of the input mesh.
	std::vector<uint32_t> m_faceIndex; // the index within the chart for every face of the input mesh.
};

/// An atlas is a set of charts.
class Atlas {
public:
	~Atlas() {
		for (size_t i = 0; i < m_meshChartsArray.size(); i++)
			delete m_meshChartsArray[i];
	}

	uint32_t meshCount() const {
		return m_meshChartsArray.size();
	}

	const MeshCharts *meshAt(uint32_t i) const {
		return m_meshChartsArray[i];
	}

	MeshCharts *meshAt(uint32_t i) {
		return m_meshChartsArray[i];
	}

	uint32_t chartCount() const {
		uint32_t count = 0;
		for (uint32_t c = 0; c < m_meshChartsArray.size(); c++) {
			count += m_meshChartsArray[c]->chartCount();
		}
		return count;
	}

	const Chart *chartAt(uint32_t i) const {
		for (uint32_t c = 0; c < m_meshChartsArray.size(); c++) {
			uint32_t count = m_meshChartsArray[c]->chartCount();
			if (i < count) {
				return m_meshChartsArray[c]->chartAt(i);
			}
			i -= count;
		}
		return NULL;
	}

	Chart *chartAt(uint32_t i) {
		for (uint32_t c = 0; c < m_meshChartsArray.size(); c++) {
			uint32_t count = m_meshChartsArray[c]->chartCount();
			if (i < count) {
				return m_meshChartsArray[c]->chartAt(i);
			}
			i -= count;
		}
		return NULL;
	}

	// Add mesh charts and takes ownership.
	// Extract the charts and add to this atlas.
	void addMeshCharts(MeshCharts *meshCharts) {
		m_meshChartsArray.push_back(meshCharts);
	}

	void extractCharts(const halfedge::Mesh *mesh) {
		MeshCharts *meshCharts = new MeshCharts(mesh);
		meshCharts->extractCharts();
		addMeshCharts(meshCharts);
	}

	void computeCharts(const halfedge::Mesh *mesh, const CharterOptions &options, const std::vector<uint32_t> &unchartedMaterialArray) {
		MeshCharts *meshCharts = new MeshCharts(mesh);
		meshCharts->computeCharts(options, unchartedMaterialArray);
		addMeshCharts(meshCharts);
	}

	void parameterizeCharts() {
		for (uint32_t i = 0; i < m_meshChartsArray.size(); i++) {
			m_meshChartsArray[i]->parameterizeCharts();
		}
	}

private:
	std::vector<MeshCharts *> m_meshChartsArray;
};

struct AtlasPacker {
	AtlasPacker(Atlas *atlas) :
			m_atlas(atlas),
			m_width(0),
			m_height(0) {
		// Save the original uvs.
		m_originalChartUvs.resize(m_atlas->chartCount());
		for (uint32_t i = 0; i < m_atlas->chartCount(); i++) {
			const halfedge::Mesh *mesh = atlas->chartAt(i)->chartMesh();
			m_originalChartUvs[i].resize(mesh->vertexCount());
			for (uint32_t j = 0; j < mesh->vertexCount(); j++)
				m_originalChartUvs[i][j] = mesh->vertexAt(j)->tex;
		}
	}

	uint32_t getWidth() const { return m_width; }
	uint32_t getHeight() const { return m_height; }

	// Pack charts in the smallest possible rectangle.
	void packCharts(const PackerOptions &options) {
		const uint32_t chartCount = m_atlas->chartCount();
		if (chartCount == 0) return;
		float texelsPerUnit = 1;
		if (options.method == PackMethod::TexelArea)
			texelsPerUnit = options.texelArea;
		for (int iteration = 0;; iteration++) {
			m_rand = MTRand();
			std::vector<float> chartOrderArray(chartCount);
			std::vector<Vector2> chartExtents(chartCount);
			float meshArea = 0;
			for (uint32_t c = 0; c < chartCount; c++) {
				Chart *chart = m_atlas->chartAt(c);
				if (!chart->isVertexMapped() && !chart->isDisk()) {
					chartOrderArray[c] = 0;
					// Skip non-disks.
					continue;
				}
				Vector2 extents(0.0f);
				if (chart->isVertexMapped()) {
					// Arrange vertices in a rectangle.
					extents.x = float(chart->vertexMapWidth);
					extents.y = float(chart->vertexMapHeight);
				} else {
					// Compute surface area to sort charts.
					float chartArea = chart->computeSurfaceArea();
					meshArea += chartArea;
					//chartOrderArray[c] = chartArea;
					// Compute chart scale
					float parametricArea = fabsf(chart->computeParametricArea()); // @@ There doesn't seem to be anything preventing parametric area to be negative.
					if (parametricArea < NV_EPSILON) {
						// When the parametric area is too small we use a rough approximation to prevent divisions by very small numbers.
						Vector2 bounds = chart->computeParametricBounds();
						parametricArea = bounds.x * bounds.y;
					}
					float scale = (chartArea / parametricArea) * texelsPerUnit;
					if (parametricArea == 0) { // < NV_EPSILON)
						scale = 0;
					}
					xaAssert(std::isfinite(scale));
					// Compute bounding box of chart.
					Vector2 majorAxis, minorAxis, origin, end;
					computeBoundingBox(chart, &majorAxis, &minorAxis, &origin, &end);
					xaAssert(isFinite(majorAxis) && isFinite(minorAxis) && isFinite(origin));
					// Sort charts by perimeter. @@ This is sometimes producing somewhat unexpected results. Is this right?
					//chartOrderArray[c] = ((end.x - origin.x) + (end.y - origin.y)) * scale;
					// Translate, rotate and scale vertices. Compute extents.
					halfedge::Mesh *mesh = chart->chartMesh();
					const uint32_t vertexCount = mesh->vertexCount();
					for (uint32_t i = 0; i < vertexCount; i++) {
						halfedge::Vertex *vertex = mesh->vertexAt(i);
						//Vector2 t = vertex->tex - origin;
						Vector2 tmp;
						tmp.x = dot(vertex->tex, majorAxis);
						tmp.y = dot(vertex->tex, minorAxis);
						tmp -= origin;
						tmp *= scale;
						if (tmp.x < 0 || tmp.y < 0) {
							xaPrint("tmp: %f %f\n", tmp.x, tmp.y);
							xaPrint("scale: %f\n", scale);
							xaPrint("origin: %f %f\n", origin.x, origin.y);
							xaPrint("majorAxis: %f %f\n", majorAxis.x, majorAxis.y);
							xaPrint("minorAxis: %f %f\n", minorAxis.x, minorAxis.y);
							xaDebugAssert(false);
						}
						//xaAssert(tmp.x >= 0 && tmp.y >= 0);
						vertex->tex = tmp;
						xaAssert(std::isfinite(vertex->tex.x) && std::isfinite(vertex->tex.y));
						extents = max(extents, tmp);
					}
					xaDebugAssert(extents.x >= 0 && extents.y >= 0);
					// Limit chart size.
					if (extents.x > 1024 || extents.y > 1024) {
						float limit = std::max(extents.x, extents.y);
						scale = 1024 / (limit + 1);
						for (uint32_t i = 0; i < vertexCount; i++) {
							halfedge::Vertex *vertex = mesh->vertexAt(i);
							vertex->tex *= scale;
						}
						extents *= scale;
						xaDebugAssert(extents.x <= 1024 && extents.y <= 1024);
					}
					// Scale the charts to use the entire texel area available. So, if the width is 0.1 we could scale it to 1 without increasing the lightmap usage and making a better
					// use of it. In many cases this also improves the look of the seams, since vertices on the chart boundaries have more chances of being aligned with the texel centers.
					float scale_x = 1.0f;
					float scale_y = 1.0f;
					float divide_x = 1.0f;
					float divide_y = 1.0f;
					if (extents.x > 0) {
						int cw = ftoi_ceil(extents.x);
						if (options.blockAlign && chart->blockAligned) {
							// Align all chart extents to 4x4 blocks, but taking padding into account.
							if (options.conservative) {
								cw = align(cw + 2, 4) - 2;
							} else {
								cw = align(cw + 1, 4) - 1;
							}
						}
						scale_x = (float(cw) - NV_EPSILON);
						divide_x = extents.x;
						extents.x = float(cw);
					}
					if (extents.y > 0) {
						int ch = ftoi_ceil(extents.y);
						if (options.blockAlign && chart->blockAligned) {
							// Align all chart extents to 4x4 blocks, but taking padding into account.
							if (options.conservative) {
								ch = align(ch + 2, 4) - 2;
							} else {
								ch = align(ch + 1, 4) - 1;
							}
						}
						scale_y = (float(ch) - NV_EPSILON);
						divide_y = extents.y;
						extents.y = float(ch);
					}
					for (uint32_t v = 0; v < vertexCount; v++) {
						halfedge::Vertex *vertex = mesh->vertexAt(v);
						vertex->tex.x /= divide_x;
						vertex->tex.y /= divide_y;
						vertex->tex.x *= scale_x;
						vertex->tex.y *= scale_y;
						xaAssert(std::isfinite(vertex->tex.x) && std::isfinite(vertex->tex.y));
					}
				}
				chartExtents[c] = extents;
				// Sort charts by perimeter.
				chartOrderArray[c] = extents.x + extents.y;
			}
			// @@ We can try to improve compression of small charts by sorting them by proximity like we do with vertex samples.
			// @@ How to do that? One idea: compute chart centroid, insert into grid, compute morton index of the cell, sort based on morton index.
			// @@ We would sort by morton index, first, then quantize the chart sizes, so that all small charts have the same size, and sort by size preserving the morton order.
			//xaPrint("Sorting charts.\n");
			// Sort charts by area.
			m_radix = RadixSort();
			m_radix.sort(chartOrderArray);
			const uint32_t *ranks = m_radix.ranks();
			// First iteration - guess texelsPerUnit.
			if (options.method != PackMethod::TexelArea && iteration == 0) {
				// Estimate size of the map based on the mesh surface area and given texel scale.
				const float texelCount = std::max(1.0f, meshArea * square(texelsPerUnit) / 0.75f); // Assume 75% utilization.
				texelsPerUnit = sqrt((options.resolution * options.resolution) / texelCount);
				resetUvs();
				continue;
			}
			// Init bit map.
			m_bitmap.clearAll();
			m_bitmap.resize(options.resolution, options.resolution, false);
			int w = 0;
			int h = 0;
			// Add sorted charts to bitmap.
			for (uint32_t i = 0; i < chartCount; i++) {
				uint32_t c = ranks[chartCount - i - 1]; // largest chart first
				Chart *chart = m_atlas->chartAt(c);
				if (!chart->isVertexMapped() && !chart->isDisk()) continue;
				//float scale_x = 1;
				//float scale_y = 1;
				BitMap chart_bitmap;
				if (chart->isVertexMapped()) {
					chart->blockAligned = false;
					// Init all bits to 1.
					chart_bitmap.resize(ftoi_ceil(chartExtents[c].x), ftoi_ceil(chartExtents[c].y), /*initValue=*/true);
					// @@ Another alternative would be to try to map each vertex to a different texel trying to fill all the available unused texels.
				} else {
					// @@ Add special cases for dot and line charts. @@ Lightmap rasterizer also needs to handle these special cases.
					// @@ We could also have a special case for chart quads. If the quad surface <= 4 texels, align vertices with texel centers and do not add padding. May be very useful for foliage.
					// @@ In general we could reduce the padding of all charts by one texel by using a rasterizer that takes into account the 2-texel footprint of the tent bilinear filter. For example,
					// if we have a chart that is less than 1 texel wide currently we add one texel to the left and one texel to the right creating a 3-texel-wide bitmap. However, if we know that the
					// chart is only 1 texel wide we could align it so that it only touches the footprint of two texels:
					//      |   |      <- Touches texels 0, 1 and 2.
					//    |   |        <- Only touches texels 0 and 1.
					// \   \ / \ /   /
					//  \   X   X   /
					//   \ / \ / \ /
					//    V   V   V
					//    0   1   2
					if (options.conservative) {
						// Init all bits to 0.
						chart_bitmap.resize(ftoi_ceil(chartExtents[c].x) + 1 + options.padding, ftoi_ceil(chartExtents[c].y) + 1 + options.padding, /*initValue=*/false); // + 2 to add padding on both sides.
						// Rasterize chart and dilate.
						drawChartBitmapDilate(chart, &chart_bitmap, options.padding);
					} else {
						// Init all bits to 0.
						chart_bitmap.resize(ftoi_ceil(chartExtents[c].x) + 1, ftoi_ceil(chartExtents[c].y) + 1, /*initValue=*/false); // Add half a texels on each side.
						// Rasterize chart and dilate.
						drawChartBitmap(chart, &chart_bitmap, Vector2(1), Vector2(0.5));
					}
				}
				int best_x, best_y;
				int best_cw, best_ch; // Includes padding now.
				int best_r;
				findChartLocation(options.quality, &chart_bitmap, chartExtents[c], w, h, &best_x, &best_y, &best_cw, &best_ch, &best_r, chart->blockAligned);
				/*if (w < best_x + best_cw || h < best_y + best_ch)
				{
					xaPrint("Resize extents to (%d, %d).\n", best_x + best_cw, best_y + best_ch);
				}*/
				// Update parametric extents.
				w = std::max(w, best_x + best_cw);
				h = std::max(h, best_y + best_ch);
				w = align(w, 4);
				h = align(h, 4);
				// Resize bitmap if necessary.
				if (uint32_t(w) > m_bitmap.width() || uint32_t(h) > m_bitmap.height()) {
					//xaPrint("Resize bitmap (%d, %d).\n", nextPowerOfTwo(w), nextPowerOfTwo(h));
					m_bitmap.resize(nextPowerOfTwo(uint32_t(w)), nextPowerOfTwo(uint32_t(h)), false);
				}
				//xaPrint("Add chart at (%d, %d).\n", best_x, best_y);
				addChart(&chart_bitmap, w, h, best_x, best_y, best_r);
				//float best_angle = 2 * PI * best_r;
				// Translate and rotate chart texture coordinates.
				halfedge::Mesh *mesh = chart->chartMesh();
				const uint32_t vertexCount = mesh->vertexCount();
				for (uint32_t v = 0; v < vertexCount; v++) {
					halfedge::Vertex *vertex = mesh->vertexAt(v);
					Vector2 t = vertex->tex;
					if (best_r) std::swap(t.x, t.y);
					//vertex->tex.x = best_x + t.x * cosf(best_angle) - t.y * sinf(best_angle);
					//vertex->tex.y = best_y + t.x * sinf(best_angle) + t.y * cosf(best_angle);
					vertex->tex.x = best_x + t.x + 0.5f;
					vertex->tex.y = best_y + t.y + 0.5f;
					xaAssert(vertex->tex.x >= 0 && vertex->tex.y >= 0);
					xaAssert(std::isfinite(vertex->tex.x) && std::isfinite(vertex->tex.y));
				}
			}
			//w -= padding - 1; // Leave one pixel border!
			//h -= padding - 1;
			m_width = std::max(0, w);
			m_height = std::max(0, h);
			xaAssert(isAligned(m_width, 4));
			xaAssert(isAligned(m_height, 4));
			if (options.method == PackMethod::ExactResolution) {
				texelsPerUnit *= sqrt((options.resolution * options.resolution) / (float)(m_width * m_height));
				if (iteration > 1 && m_width <= options.resolution && m_height <= options.resolution) {
					m_width = m_height = options.resolution;
					return;
				}
				resetUvs();
			} else {
				return;
			}
		}
	}

	float computeAtlasUtilization() const {
		const uint32_t w = m_width;
		const uint32_t h = m_height;
		xaDebugAssert(w <= m_bitmap.width());
		xaDebugAssert(h <= m_bitmap.height());
		uint32_t count = 0;
		for (uint32_t y = 0; y < h; y++) {
			for (uint32_t x = 0; x < w; x++) {
				count += m_bitmap.bitAt(x, y);
			}
		}
		return float(count) / (w * h);
	}

private:
	void resetUvs() {
		for (uint32_t i = 0; i < m_atlas->chartCount(); i++) {
			halfedge::Mesh *mesh = m_atlas->chartAt(i)->chartMesh();
			for (uint32_t j = 0; j < mesh->vertexCount(); j++)
				mesh->vertexAt(j)->tex = m_originalChartUvs[i][j];
		}
	}

	// IC: Brute force is slow, and random may take too much time to converge. We start inserting large charts in a small atlas. Using brute force is lame, because most of the space
	// is occupied at this point. At the end we have many small charts and a large atlas with sparse holes. Finding those holes randomly is slow. A better approach would be to
	// start stacking large charts as if they were tetris pieces. Once charts get small try to place them randomly. It may be interesting to try a intermediate strategy, first try
	// along one axis and then try exhaustively along that axis.
	void findChartLocation(int quality, const BitMap *bitmap, Vector2::Arg extents, int w, int h, int *best_x, int *best_y, int *best_w, int *best_h, int *best_r, bool blockAligned) {
		int attempts = 256;
		if (quality == 1) attempts = 4096;
		if (quality == 2) attempts = 2048;
		if (quality == 3) attempts = 1024;
		if (quality == 4) attempts = 512;
		if (quality == 0 || w * h < attempts) {
			findChartLocation_bruteForce(bitmap, extents, w, h, best_x, best_y, best_w, best_h, best_r, blockAligned);
		} else {
			findChartLocation_random(bitmap, extents, w, h, best_x, best_y, best_w, best_h, best_r, attempts, blockAligned);
		}
	}

	void findChartLocation_bruteForce(const BitMap *bitmap, Vector2::Arg /*extents*/, int w, int h, int *best_x, int *best_y, int *best_w, int *best_h, int *best_r, bool blockAligned) {
		const int BLOCK_SIZE = 4;
		int best_metric = INT_MAX;
		int step_size = blockAligned ? BLOCK_SIZE : 1;
		// Try two different orientations.
		for (int r = 0; r < 2; r++) {
			int cw = bitmap->width();
			int ch = bitmap->height();
			if (r & 1) std::swap(cw, ch);
			for (int y = 0; y <= h + 1; y += step_size) { // + 1 to extend atlas in case atlas full.
				for (int x = 0; x <= w + 1; x += step_size) { // + 1 not really necessary here.
					// Early out.
					int area = std::max(w, x + cw) * std::max(h, y + ch);
					//int perimeter = max(w, x+cw) + max(h, y+ch);
					int extents = std::max(std::max(w, x + cw), std::max(h, y + ch));
					int metric = extents * extents + area;
					if (metric > best_metric) {
						continue;
					}
					if (metric == best_metric && std::max(x, y) >= std::max(*best_x, *best_y)) {
						// If metric is the same, pick the one closest to the origin.
						continue;
					}
					if (canAddChart(bitmap, w, h, x, y, r)) {
						best_metric = metric;
						*best_x = x;
						*best_y = y;
						*best_w = cw;
						*best_h = ch;
						*best_r = r;
						if (area == w * h) {
							// Chart is completely inside, do not look at any other location.
							goto done;
						}
					}
				}
			}
		}
	done:
		xaDebugAssert(best_metric != INT_MAX);
	}

	void findChartLocation_random(const BitMap *bitmap, Vector2::Arg /*extents*/, int w, int h, int *best_x, int *best_y, int *best_w, int *best_h, int *best_r, int minTrialCount, bool blockAligned) {
		const int BLOCK_SIZE = 4;
		int best_metric = INT_MAX;
		for (int i = 0; i < minTrialCount || best_metric == INT_MAX; i++) {
			int r = m_rand.getRange(1);
			int x = m_rand.getRange(w + 1); // + 1 to extend atlas in case atlas full. We may want to use a higher number to increase probability of extending atlas.
			int y = m_rand.getRange(h + 1); // + 1 to extend atlas in case atlas full.
			if (blockAligned) {
				x = align(x, BLOCK_SIZE);
				y = align(y, BLOCK_SIZE);
			}
			int cw = bitmap->width();
			int ch = bitmap->height();
			if (r & 1) std::swap(cw, ch);
			// Early out.
			int area = std::max(w, x + cw) * std::max(h, y + ch);
			//int perimeter = max(w, x+cw) + max(h, y+ch);
			int extents = std::max(std::max(w, x + cw), std::max(h, y + ch));
			int metric = extents * extents + area;
			if (metric > best_metric) {
				continue;
			}
			if (metric == best_metric && std::min(x, y) > std::min(*best_x, *best_y)) {
				// If metric is the same, pick the one closest to the origin.
				continue;
			}
			if (canAddChart(bitmap, w, h, x, y, r)) {
				best_metric = metric;
				*best_x = x;
				*best_y = y;
				*best_w = cw;
				*best_h = ch;
				*best_r = r;
				if (area == w * h) {
					// Chart is completely inside, do not look at any other location.
					break;
				}
			}
		}
	}

	void drawChartBitmapDilate(const Chart *chart, BitMap *bitmap, int padding) {
		const int w = bitmap->width();
		const int h = bitmap->height();
		const Vector2 extents = Vector2(float(w), float(h));
		// Rasterize chart faces, check that all bits are not set.
		const uint32_t faceCount = chart->faceCount();
		for (uint32_t f = 0; f < faceCount; f++) {
			const halfedge::Face *face = chart->chartMesh()->faceAt(f);
			Vector2 vertices[4];
			uint32_t edgeCount = 0;
			for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
				if (edgeCount < 4) {
					vertices[edgeCount] = it.vertex()->tex + Vector2(0.5) + Vector2(float(padding), float(padding));
				}
				edgeCount++;
			}
			if (edgeCount == 3) {
				raster::drawTriangle(raster::Mode_Antialiased, extents, true, vertices, AtlasPacker::setBitsCallback, bitmap);
			} else {
				raster::drawQuad(raster::Mode_Antialiased, extents, true, vertices, AtlasPacker::setBitsCallback, bitmap);
			}
		}
		// Expand chart by padding pixels. (dilation)
		BitMap tmp(w, h);
		for (int i = 0; i < padding; i++) {
			tmp.clearAll();
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					bool b = bitmap->bitAt(x, y);
					if (!b) {
						if (x > 0) {
							b |= bitmap->bitAt(x - 1, y);
							if (y > 0) b |= bitmap->bitAt(x - 1, y - 1);
							if (y < h - 1) b |= bitmap->bitAt(x - 1, y + 1);
						}
						if (y > 0) b |= bitmap->bitAt(x, y - 1);
						if (y < h - 1) b |= bitmap->bitAt(x, y + 1);
						if (x < w - 1) {
							b |= bitmap->bitAt(x + 1, y);
							if (y > 0) b |= bitmap->bitAt(x + 1, y - 1);
							if (y < h - 1) b |= bitmap->bitAt(x + 1, y + 1);
						}
					}
					if (b) tmp.setBitAt(x, y);
				}
			}
			std::swap(tmp, *bitmap);
		}
	}

	void drawChartBitmap(const Chart *chart, BitMap *bitmap, const Vector2 &scale, const Vector2 &offset) {
		const int w = bitmap->width();
		const int h = bitmap->height();
		const Vector2 extents = Vector2(float(w), float(h));
		static const Vector2 pad[4] = {
			Vector2(-0.5, -0.5),
			Vector2(0.5, -0.5),
			Vector2(-0.5, 0.5),
			Vector2(0.5, 0.5)
		};
		// Rasterize 4 times to add proper padding.
		for (int i = 0; i < 4; i++) {
			// Rasterize chart faces, check that all bits are not set.
			const uint32_t faceCount = chart->chartMesh()->faceCount();
			for (uint32_t f = 0; f < faceCount; f++) {
				const halfedge::Face *face = chart->chartMesh()->faceAt(f);
				Vector2 vertices[4];
				uint32_t edgeCount = 0;
				for (halfedge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
					if (edgeCount < 4) {
						vertices[edgeCount] = it.vertex()->tex * scale + offset + pad[i];
						xaAssert(ftoi_ceil(vertices[edgeCount].x) >= 0);
						xaAssert(ftoi_ceil(vertices[edgeCount].y) >= 0);
						xaAssert(ftoi_ceil(vertices[edgeCount].x) <= w);
						xaAssert(ftoi_ceil(vertices[edgeCount].y) <= h);
					}
					edgeCount++;
				}
				if (edgeCount == 3) {
					raster::drawTriangle(raster::Mode_Antialiased, extents, /*enableScissors=*/true, vertices, AtlasPacker::setBitsCallback, bitmap);
				} else {
					raster::drawQuad(raster::Mode_Antialiased, extents, /*enableScissors=*/true, vertices, AtlasPacker::setBitsCallback, bitmap);
				}
			}
		}
		// Expand chart by padding pixels. (dilation)
		BitMap tmp(w, h);
		tmp.clearAll();
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				bool b = bitmap->bitAt(x, y);
				if (!b) {
					if (x > 0) {
						b |= bitmap->bitAt(x - 1, y);
						if (y > 0) b |= bitmap->bitAt(x - 1, y - 1);
						if (y < h - 1) b |= bitmap->bitAt(x - 1, y + 1);
					}
					if (y > 0) b |= bitmap->bitAt(x, y - 1);
					if (y < h - 1) b |= bitmap->bitAt(x, y + 1);
					if (x < w - 1) {
						b |= bitmap->bitAt(x + 1, y);
						if (y > 0) b |= bitmap->bitAt(x + 1, y - 1);
						if (y < h - 1) b |= bitmap->bitAt(x + 1, y + 1);
					}
				}
				if (b) tmp.setBitAt(x, y);
			}
		}
		std::swap(tmp, *bitmap);
	}

	bool canAddChart(const BitMap *bitmap, int atlas_w, int atlas_h, int offset_x, int offset_y, int r) {
		xaDebugAssert(r == 0 || r == 1);
		// Check whether the two bitmaps overlap.
		const int w = bitmap->width();
		const int h = bitmap->height();
		if (r == 0) {
			for (int y = 0; y < h; y++) {
				int yy = y + offset_y;
				if (yy >= 0) {
					for (int x = 0; x < w; x++) {
						int xx = x + offset_x;
						if (xx >= 0) {
							if (bitmap->bitAt(x, y)) {
								if (xx < atlas_w && yy < atlas_h) {
									if (m_bitmap.bitAt(xx, yy)) return false;
								}
							}
						}
					}
				}
			}
		} else if (r == 1) {
			for (int y = 0; y < h; y++) {
				int xx = y + offset_x;
				if (xx >= 0) {
					for (int x = 0; x < w; x++) {
						int yy = x + offset_y;
						if (yy >= 0) {
							if (bitmap->bitAt(x, y)) {
								if (xx < atlas_w && yy < atlas_h) {
									if (m_bitmap.bitAt(xx, yy)) return false;
								}
							}
						}
					}
				}
			}
		}
		return true;
	}

	void addChart(const BitMap *bitmap, int atlas_w, int atlas_h, int offset_x, int offset_y, int r) {
		xaDebugAssert(r == 0 || r == 1);
		// Check whether the two bitmaps overlap.
		const int w = bitmap->width();
		const int h = bitmap->height();
		if (r == 0) {
			for (int y = 0; y < h; y++) {
				int yy = y + offset_y;
				if (yy >= 0) {
					for (int x = 0; x < w; x++) {
						int xx = x + offset_x;
						if (xx >= 0) {
							if (bitmap->bitAt(x, y)) {
								if (xx < atlas_w && yy < atlas_h) {
									xaDebugAssert(m_bitmap.bitAt(xx, yy) == false);
									m_bitmap.setBitAt(xx, yy);
								}
							}
						}
					}
				}
			}
		} else if (r == 1) {
			for (int y = 0; y < h; y++) {
				int xx = y + offset_x;
				if (xx >= 0) {
					for (int x = 0; x < w; x++) {
						int yy = x + offset_y;
						if (yy >= 0) {
							if (bitmap->bitAt(x, y)) {
								if (xx < atlas_w && yy < atlas_h) {
									xaDebugAssert(m_bitmap.bitAt(xx, yy) == false);
									m_bitmap.setBitAt(xx, yy);
								}
							}
						}
					}
				}
			}
		}
	}

	static bool setBitsCallback(void *param, int x, int y, Vector3::Arg, Vector3::Arg, Vector3::Arg, float area) {
		BitMap *bitmap = (BitMap *)param;
		if (area > 0.0) {
			bitmap->setBitAt(x, y);
		}
		return true;
	}

	// Compute the convex hull using Graham Scan.
	static void convexHull(const std::vector<Vector2> &input, std::vector<Vector2> &output, float epsilon) {
		const uint32_t inputCount = input.size();
		std::vector<float> coords(inputCount);
		for (uint32_t i = 0; i < inputCount; i++) {
			coords[i] = input[i].x;
		}
		RadixSort radix;
		radix.sort(coords);
		const uint32_t *ranks = radix.ranks();
		std::vector<Vector2> top;
		top.reserve(inputCount);
		std::vector<Vector2> bottom;
		bottom.reserve(inputCount);
		Vector2 P = input[ranks[0]];
		Vector2 Q = input[ranks[inputCount - 1]];
		float topy = std::max(P.y, Q.y);
		float boty = std::min(P.y, Q.y);
		for (uint32_t i = 0; i < inputCount; i++) {
			Vector2 p = input[ranks[i]];
			if (p.y >= boty) top.push_back(p);
		}
		for (uint32_t i = 0; i < inputCount; i++) {
			Vector2 p = input[ranks[inputCount - 1 - i]];
			if (p.y <= topy) bottom.push_back(p);
		}
		// Filter top list.
		output.clear();
		output.push_back(top[0]);
		output.push_back(top[1]);
		for (uint32_t i = 2; i < top.size();) {
			Vector2 a = output[output.size() - 2];
			Vector2 b = output[output.size() - 1];
			Vector2 c = top[i];
			float area = triangleArea(a, b, c);
			if (area >= -epsilon) {
				output.pop_back();
			}
			if (area < -epsilon || output.size() == 1) {
				output.push_back(c);
				i++;
			}
		}
		uint32_t top_count = output.size();
		output.push_back(bottom[1]);
		// Filter bottom list.
		for (uint32_t i = 2; i < bottom.size();) {
			Vector2 a = output[output.size() - 2];
			Vector2 b = output[output.size() - 1];
			Vector2 c = bottom[i];
			float area = triangleArea(a, b, c);
			if (area >= -epsilon) {
				output.pop_back();
			}
			if (area < -epsilon || output.size() == top_count) {
				output.push_back(c);
				i++;
			}
		}
		// Remove duplicate element.
		xaDebugAssert(output.front() == output.back());
		output.pop_back();
	}

	// This should compute convex hull and use rotating calipers to find the best box. Currently it uses a brute force method.
	static void computeBoundingBox(Chart *chart, Vector2 *majorAxis, Vector2 *minorAxis, Vector2 *minCorner, Vector2 *maxCorner) {
		// Compute list of boundary points.
		std::vector<Vector2> points;
		points.reserve(16);
		halfedge::Mesh *mesh = chart->chartMesh();
		const uint32_t vertexCount = mesh->vertexCount();
		for (uint32_t i = 0; i < vertexCount; i++) {
			halfedge::Vertex *vertex = mesh->vertexAt(i);
			if (vertex->isBoundary()) {
				points.push_back(vertex->tex);
			}
		}
		xaDebugAssert(points.size() > 0);
		std::vector<Vector2> hull;
		convexHull(points, hull, 0.00001f);
		// @@ Ideally I should use rotating calipers to find the best box. Using brute force for now.
		float best_area = FLT_MAX;
		Vector2 best_min;
		Vector2 best_max;
		Vector2 best_axis;
		const uint32_t hullCount = hull.size();
		for (uint32_t i = 0, j = hullCount - 1; i < hullCount; j = i, i++) {
			if (equal(hull[i], hull[j])) {
				continue;
			}
			Vector2 axis = normalize(hull[i] - hull[j], 0.0f);
			xaDebugAssert(isFinite(axis));
			// Compute bounding box.
			Vector2 box_min(FLT_MAX, FLT_MAX);
			Vector2 box_max(-FLT_MAX, -FLT_MAX);
			for (uint32_t v = 0; v < hullCount; v++) {
				Vector2 point = hull[v];
				float x = dot(axis, point);
				if (x < box_min.x) box_min.x = x;
				if (x > box_max.x) box_max.x = x;
				float y = dot(Vector2(-axis.y, axis.x), point);
				if (y < box_min.y) box_min.y = y;
				if (y > box_max.y) box_max.y = y;
			}
			// Compute box area.
			float area = (box_max.x - box_min.x) * (box_max.y - box_min.y);
			if (area < best_area) {
				best_area = area;
				best_min = box_min;
				best_max = box_max;
				best_axis = axis;
			}
		}
		// Consider all points, not only boundary points, in case the input chart is malformed.
		for (uint32_t i = 0; i < vertexCount; i++) {
			halfedge::Vertex *vertex = mesh->vertexAt(i);
			Vector2 point = vertex->tex;
			float x = dot(best_axis, point);
			if (x < best_min.x) best_min.x = x;
			if (x > best_max.x) best_max.x = x;
			float y = dot(Vector2(-best_axis.y, best_axis.x), point);
			if (y < best_min.y) best_min.y = y;
			if (y > best_max.y) best_max.y = y;
		}
		*majorAxis = best_axis;
		*minorAxis = Vector2(-best_axis.y, best_axis.x);
		*minCorner = best_min;
		*maxCorner = best_max;
	}

	Atlas *m_atlas;
	BitMap m_bitmap;
	RadixSort m_radix;
	uint32_t m_width;
	uint32_t m_height;
	MTRand m_rand;
	std::vector<std::vector<Vector2> > m_originalChartUvs;
};

} // namespace param
} // namespace internal

struct Atlas {
	internal::param::Atlas atlas;
	std::vector<internal::halfedge::Mesh *> heMeshes;
	uint32_t width = 0;
	uint32_t height = 0;
	OutputMesh **outputMeshes = NULL;
};

void SetPrint(PrintFunc print) {
	internal::s_print = print;
}

Atlas *Create() {
	Atlas *atlas = new Atlas();
	return atlas;
}

void Destroy(Atlas *atlas) {
	xaAssert(atlas);
	for (int i = 0; i < (int)atlas->heMeshes.size(); i++) {
		delete atlas->heMeshes[i];
		if (atlas->outputMeshes) {
			OutputMesh *outputMesh = atlas->outputMeshes[i];
			for (uint32_t j = 0; j < outputMesh->chartCount; j++)
				delete[] outputMesh->chartArray[j].indexArray;
			delete[] outputMesh->chartArray;
			delete[] outputMesh->vertexArray;
			delete[] outputMesh->indexArray;
			delete outputMesh;
		}
	}
	delete[] atlas->outputMeshes;
	delete atlas;
}

static internal::Vector3 DecodePosition(const InputMesh &mesh, uint32_t index) {
	xaAssert(mesh.vertexPositionData);
	return *((const internal::Vector3 *)&((const uint8_t *)mesh.vertexPositionData)[mesh.vertexPositionStride * index]);
}

static internal::Vector3 DecodeNormal(const InputMesh &mesh, uint32_t index) {
	xaAssert(mesh.vertexNormalData);
	return *((const internal::Vector3 *)&((const uint8_t *)mesh.vertexNormalData)[mesh.vertexNormalStride * index]);
}

static internal::Vector2 DecodeUv(const InputMesh &mesh, uint32_t index) {
	xaAssert(mesh.vertexUvData);
	return *((const internal::Vector2 *)&((const uint8_t *)mesh.vertexUvData)[mesh.vertexUvStride * index]);
}

static uint32_t DecodeIndex(IndexFormat::Enum format, const void *indexData, uint32_t i) {
	if (format == IndexFormat::HalfFloat)
		return (uint32_t)((const uint16_t *)indexData)[i];
	return ((const uint32_t *)indexData)[i];
}

static float EdgeLength(internal::Vector3 pos1, internal::Vector3 pos2) {
	return internal::length(pos2 - pos1);
}

AddMeshError AddMesh(Atlas *atlas, const InputMesh &mesh, bool useColocalVertices) {
	xaAssert(atlas);
	AddMeshError error;
	error.code = AddMeshErrorCode::Success;
	error.face = error.index0 = error.index1 = UINT32_MAX;
	// Expecting triangle faces.
	if ((mesh.indexCount % 3) != 0) {
		error.code = AddMeshErrorCode::InvalidIndexCount;
		return error;
	}
	// Check if any index is out of range.
	for (uint32_t j = 0; j < mesh.indexCount; j++) {
		const uint32_t index = DecodeIndex(mesh.indexFormat, mesh.indexData, j);
		if (index < 0 || index >= mesh.vertexCount) {
			error.code = AddMeshErrorCode::IndexOutOfRange;
			error.index0 = index;
			return error;
		}
	}
	// Build half edge mesh.
	internal::halfedge::Mesh *heMesh = new internal::halfedge::Mesh;
	std::vector<uint32_t> canonicalMap;
	canonicalMap.reserve(mesh.vertexCount);
	for (uint32_t i = 0; i < mesh.vertexCount; i++) {
		internal::halfedge::Vertex *vertex = heMesh->addVertex(DecodePosition(mesh, i));
		if (mesh.vertexNormalData)
			vertex->nor = DecodeNormal(mesh, i);
		if (mesh.vertexUvData)
			vertex->tex = DecodeUv(mesh, i);
		// Link colocals. You probably want to do this more efficiently! Sort by one axis or use a hash or grid.
		uint32_t firstColocal = i;
		if (useColocalVertices) {
			for (uint32_t j = 0; j < i; j++) {
				if (vertex->pos != DecodePosition(mesh, j))
					continue;
#if 0
				if (mesh.vertexNormalData && vertex->nor != DecodeNormal(mesh, j))
					continue;
#endif
				if (mesh.vertexUvData && vertex->tex != DecodeUv(mesh, j))
					continue;
				firstColocal = j;
				break;
			}
		}
		canonicalMap.push_back(firstColocal);
	}
	heMesh->linkColocalsWithCanonicalMap(canonicalMap);
	for (uint32_t i = 0; i < mesh.indexCount / 3; i++) {
		uint32_t tri[3];
		for (int j = 0; j < 3; j++)
			tri[j] = DecodeIndex(mesh.indexFormat, mesh.indexData, i * 3 + j);
		// Check for zero length edges.
		for (int j = 0; j < 3; j++) {
			const uint32_t edges[6] = { 0, 1, 1, 2, 2, 0 };
			const uint32_t index1 = tri[edges[j * 2 + 0]];
			const uint32_t index2 = tri[edges[j * 2 + 1]];
			const internal::Vector3 pos1 = DecodePosition(mesh, index1);
			const internal::Vector3 pos2 = DecodePosition(mesh, index2);
			if (EdgeLength(pos1, pos2) <= 0.0f) {
				delete heMesh;
				error.code = AddMeshErrorCode::ZeroLengthEdge;
				error.face = i;
				error.index0 = index1;
				error.index1 = index2;
				return error;
			}
		}
		// Check for zero area faces.
		{
			const internal::Vector3 a = DecodePosition(mesh, tri[0]);
			const internal::Vector3 b = DecodePosition(mesh, tri[1]);
			const internal::Vector3 c = DecodePosition(mesh, tri[2]);
			const float area = internal::length(internal::cross(b - a, c - a)) * 0.5f;
			if (area <= 0.0f) {
				delete heMesh;
				error.code = AddMeshErrorCode::ZeroAreaFace;
				error.face = i;
				return error;
			}
		}
		internal::halfedge::Face *face = heMesh->addFace(tri[0], tri[1], tri[2]);

		if (!face && heMesh->errorCode == internal::halfedge::Mesh::ErrorCode::AlreadyAddedEdge) {
			//there is still hope for this, no reason to not add, at least add as separate
			face = heMesh->addUniqueFace(tri[0], tri[1], tri[2]);
		}

		if (!face) {
			//continue;

			if (heMesh->errorCode == internal::halfedge::Mesh::ErrorCode::AlreadyAddedEdge) {
				error.code = AddMeshErrorCode::AlreadyAddedEdge;
			} else if (heMesh->errorCode == internal::halfedge::Mesh::ErrorCode::DegenerateColocalEdge) {
				error.code = AddMeshErrorCode::DegenerateColocalEdge;
			} else if (heMesh->errorCode == internal::halfedge::Mesh::ErrorCode::DegenerateEdge) {
				error.code = AddMeshErrorCode::DegenerateEdge;
			} else if (heMesh->errorCode == internal::halfedge::Mesh::ErrorCode::DuplicateEdge) {
				error.code = AddMeshErrorCode::DuplicateEdge;
			}
			error.face = i;
			error.index0 = heMesh->errorIndex0;
			error.index1 = heMesh->errorIndex1;
			delete heMesh;
			return error;
		}
		if (mesh.faceMaterialData)
			face->material = mesh.faceMaterialData[i];
	}
	heMesh->linkBoundary();
	atlas->heMeshes.push_back(heMesh);
	return error;
}

void Generate(Atlas *atlas, CharterOptions charterOptions, PackerOptions packerOptions) {
	xaAssert(atlas);
	xaAssert(packerOptions.texelArea > 0);
	// Chart meshes.
	for (int i = 0; i < (int)atlas->heMeshes.size(); i++) {
		std::vector<uint32_t> uncharted_materials;
		atlas->atlas.computeCharts(atlas->heMeshes[i], charterOptions, uncharted_materials);
	}
	atlas->atlas.parameterizeCharts();
	internal::param::AtlasPacker packer(&atlas->atlas);
	packer.packCharts(packerOptions);
	//float utilization = return packer.computeAtlasUtilization();
	atlas->width = packer.getWidth();
	atlas->height = packer.getHeight();
	// Build output meshes.
	atlas->outputMeshes = new OutputMesh *[atlas->heMeshes.size()];
	for (int i = 0; i < (int)atlas->heMeshes.size(); i++) {
		const internal::halfedge::Mesh *heMesh = atlas->heMeshes[i];
		OutputMesh *outputMesh = atlas->outputMeshes[i] = new OutputMesh;
		const internal::param::MeshCharts *charts = atlas->atlas.meshAt(i);
		// Vertices.
		outputMesh->vertexCount = charts->vertexCount();
		outputMesh->vertexArray = new OutputVertex[outputMesh->vertexCount];
		for (uint32_t i = 0; i < charts->chartCount(); i++) {
			const internal::param::Chart *chart = charts->chartAt(i);
			const uint32_t vertexOffset = charts->vertexCountBeforeChartAt(i);
			for (uint32_t v = 0; v < chart->vertexCount(); v++) {
				OutputVertex &output_vertex = outputMesh->vertexArray[vertexOffset + v];
				output_vertex.xref = chart->mapChartVertexToOriginalVertex(v);
				internal::Vector2 uv = chart->chartMesh()->vertexAt(v)->tex;
				output_vertex.uv[0] = uv.x;
				output_vertex.uv[1] = uv.y;
			}
		}
		// Indices.
		outputMesh->indexCount = heMesh->faceCount() * 3;
		outputMesh->indexArray = new uint32_t[outputMesh->indexCount];
		for (uint32_t f = 0; f < heMesh->faceCount(); f++) {
			const uint32_t c = charts->faceChartAt(f);
			const uint32_t i = charts->faceIndexWithinChartAt(f);
			const uint32_t vertexOffset = charts->vertexCountBeforeChartAt(c);
			const internal::param::Chart *chart = charts->chartAt(c);
			xaDebugAssert(i < chart->chartMesh()->faceCount());
			xaDebugAssert(chart->faceAt(i) == f);
			const internal::halfedge::Face *face = chart->chartMesh()->faceAt(i);
			const internal::halfedge::Edge *edge = face->edge;
			outputMesh->indexArray[3 * f + 0] = vertexOffset + edge->vertex->id;
			outputMesh->indexArray[3 * f + 1] = vertexOffset + edge->next->vertex->id;
			outputMesh->indexArray[3 * f + 2] = vertexOffset + edge->next->next->vertex->id;
		}
		// Charts.
		outputMesh->chartCount = charts->chartCount();
		outputMesh->chartArray = new OutputChart[outputMesh->chartCount];
		for (uint32_t i = 0; i < charts->chartCount(); i++) {
			OutputChart *outputChart = &outputMesh->chartArray[i];
			const internal::param::Chart *chart = charts->chartAt(i);
			const uint32_t vertexOffset = charts->vertexCountBeforeChartAt(i);
			const internal::halfedge::Mesh *mesh = chart->chartMesh();
			outputChart->indexCount = mesh->faceCount() * 3;
			outputChart->indexArray = new uint32_t[outputChart->indexCount];
			for (uint32_t j = 0; j < mesh->faceCount(); j++) {
				const internal::halfedge::Face *face = mesh->faceAt(j);
				const internal::halfedge::Edge *edge = face->edge;
				outputChart->indexArray[3 * j + 0] = vertexOffset + edge->vertex->id;
				outputChart->indexArray[3 * j + 1] = vertexOffset + edge->next->vertex->id;
				outputChart->indexArray[3 * j + 2] = vertexOffset + edge->next->next->vertex->id;
			}
		}
	}
}

uint32_t GetWidth(const Atlas *atlas) {
	xaAssert(atlas);
	return atlas->width;
}

uint32_t GetHeight(const Atlas *atlas) {
	xaAssert(atlas);
	return atlas->height;
}

uint32_t GetNumCharts(const Atlas *atlas) {
	xaAssert(atlas);
	return atlas->atlas.chartCount();
}

const OutputMesh *const *GetOutputMeshes(const Atlas *atlas) {
	xaAssert(atlas);
	return atlas->outputMeshes;
}

const char *StringForEnum(AddMeshErrorCode::Enum error) {
	if (error == AddMeshErrorCode::AlreadyAddedEdge)
		return "already added edge";
	if (error == AddMeshErrorCode::DegenerateColocalEdge)
		return "degenerate colocal edge";
	if (error == AddMeshErrorCode::DegenerateEdge)
		return "degenerate edge";
	if (error == AddMeshErrorCode::DuplicateEdge)
		return "duplicate edge";
	if (error == AddMeshErrorCode::IndexOutOfRange)
		return "index out of range";
	if (error == AddMeshErrorCode::InvalidIndexCount)
		return "invalid index count";
	if (error == AddMeshErrorCode::ZeroAreaFace)
		return "zero area face";
	if (error == AddMeshErrorCode::ZeroLengthEdge)
		return "zero length edge";
	return "success";
}

} // namespace xatlas

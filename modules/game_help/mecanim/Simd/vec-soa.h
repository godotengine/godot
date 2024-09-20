#pragma once

#include "vec-soa-types.h"
#include "vec-pix.h"
#include "vec-affine.h"
#include "Runtime/Math/Vector3.h"
#include "Runtime/Math/Vector2.h"

namespace math
{
    static MATH_FORCEINLINE floatNx3 abs(const floatNx3& m) { return floatNx3(abs(m.x), abs(m.y), abs(m.z)); }
    static MATH_FORCEINLINE floatN dot(const floatNx3& a, const floatNx3& b) { return mad(a.x, b.x, mad(a.y, b.y, a.z * b.z)); }
    static MATH_FORCEINLINE floatN dot(const floatNx3& m) { return dot(m, m); }
    static MATH_FORCEINLINE floatN dot(const floatNx2& a, const floatNx2& b) { return mad(a.x, b.x, a.y * b.y); }
    static MATH_FORCEINLINE floatN dot(const floatNx2& m) { return dot(m, m); }
    static MATH_FORCEINLINE floatNx3 cross(const floatNx3& a, const floatNx3& b) { return floatNx3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
    static MATH_FORCEINLINE floatN length(const floatNx3& m) { return sqrt(dot(m)); }
    static MATH_FORCEINLINE floatN length(const floatNx2& m) { return sqrt(dot(m)); }
    static MATH_FORCEINLINE floatNx2 min(const floatNx2& a, const floatNx2& b) { return floatNx2(min(a.x, b.x), min(a.y, b.y)); }
    static MATH_FORCEINLINE floatNx3 min(const floatNx3& a, const floatNx3& b) { return floatNx3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
    static MATH_FORCEINLINE floatNx2 max(const floatNx2& a, const floatNx2& b) { return floatNx2(max(a.x, b.x), max(a.y, b.y)); }
    static MATH_FORCEINLINE floatNx3 max(const floatNx3& a, const floatNx3& b) { return floatNx3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
    static MATH_FORCEINLINE float2 cmin(const floatNx2& m) { return float2(cmin(m.x), cmin(m.y)); }
    static MATH_FORCEINLINE float3 cmin(const floatNx3& m) { return float3(cmin(m.x), cmin(m.y), cmin(m.z)); }
    static MATH_FORCEINLINE float2 cmax(const floatNx2& m) { return float2(cmax(m.x), cmax(m.y)); }
    static MATH_FORCEINLINE float3 cmax(const floatNx3& m) { return float3(cmax(m.x), cmax(m.y), cmax(m.z)); }
    static MATH_FORCEINLINE floatNx4 mul(const floatNx4& m, const floatNx4& s) { return floatNx4(m.x * s.x, m.y * s.y, m.z * s.z, m.w * s.w); }
    static MATH_FORCEINLINE floatNx4 mul(const floatNx4& m, const floatN& s) { return floatNx4(m.x * s, m.y * s, m.z * s, m.w * s); }
    static MATH_FORCEINLINE floatNx3 mul(const floatNx3& m, const floatNx3& s) { return floatNx3(m.x * s.x, m.y * s.y, m.z * s.z); }
    static MATH_FORCEINLINE floatNx3 mul(const floatNx3& m, const floatN& s) { return floatNx3(m.x * s, m.y * s, m.z * s); }
    static MATH_FORCEINLINE floatNx2 mul(const floatNx2& m, const floatNx2& s) { return floatNx2(m.x * s.x, m.y * s.y); }
    static MATH_FORCEINLINE floatNx2 mul(const floatNx2& m, const floatN& s) { return floatNx2(m.x * s, m.y * s); }
    static MATH_FORCEINLINE floatNx3 div(const floatNx3& m, const floatN& s) { return floatNx3(m.x / s, m.y / s, m.z / s); }
    static MATH_FORCEINLINE floatNx2 div(const floatNx2& m, const floatN& s) { return floatNx2(m.x / s, m.y / s); }
    static MATH_FORCEINLINE floatNx3 mad(const floatNx3& m, const floatN& a, const floatN& b) { return floatNx3(mad(m.x, a, b), mad(m.y, a, b), mad(m.z, a, b)); }
    static MATH_FORCEINLINE floatNx2 mad(const floatNx2& m, const floatN& a, const floatN& b) { return floatNx2(mad(m.x, a, b), mad(m.y, a, b)); }
    static MATH_FORCEINLINE floatNx2 mad(const floatNx2& m, const floatN& a, const floatNx2& b) { return floatNx2(mad(m.x, a, b.x), mad(m.y, a, b.y)); }
    static MATH_FORCEINLINE floatNx3 select(const floatNx3& a, const floatNx3& b, const intN& c) { return floatNx3(select(a.x, b.x, c), select(a.y, b.y, c), select(a.z, b.z, c)); }
    static MATH_FORCEINLINE floatNx2 select(const floatNx2& a, const floatNx2& b, const intN& c) { return floatNx2(select(a.x, b.x, c), select(a.y, b.y, c)); }
    static MATH_FORCEINLINE floatNx3 normalize(const floatNx3& m) { return mul(m, rsqrt(dot(m))); }
    static MATH_FORCEINLINE floatNx2 normalize(const floatNx2& m) { return mul(m, rsqrt(dot(m))); }
    static MATH_FORCEINLINE floatNx3 normalizeSafe(const floatNx3& m, floatN* sqLen, const floatNx3& def = floatNx3(ZERO, ZERO, ZERO), const float epsilon = epsilon_normal()) { *sqLen = dot(m); return select(def, mul(m, rsqrt(*sqLen)), *sqLen > floatN(epsilon)); }
    static MATH_FORCEINLINE floatNx2 normalizeSafe(const floatNx2& m, floatN* sqLen, const floatNx2& def = floatNx2(ZERO, ZERO), const float epsilon = epsilon_normal()) { *sqLen = dot(m); return select(def, mul(m, rsqrt(*sqLen)), *sqLen > floatN(epsilon)); }
    static MATH_FORCEINLINE floatNx3 normalizeSafe(const floatNx3& m, const floatNx3& def = floatNx3(ZERO, ZERO, ZERO), const float epsilon = epsilon_normal()) { floatN sqLen; return normalizeSafe(m, &sqLen, def, epsilon); }
    static MATH_FORCEINLINE floatNx2 normalizeSafe(const floatNx2& m, const floatNx2& def = floatNx2(ZERO, ZERO), const float epsilon = epsilon_normal()) { floatN sqLen; return normalizeSafe(m, &sqLen, def, epsilon); }
    static MATH_FORCEINLINE floatNx3 floor(const floatNx3& m) { return floatNx3(floor(m.x), floor(m.y), floor(m.z)); }
    static MATH_FORCEINLINE floatNx2 floor(const floatNx2& m) { return floatNx2(floor(m.x), floor(m.y)); }
    static MATH_FORCEINLINE floatNx4 saturate(const floatNx4& m) { return floatNx4(saturate(m.x), saturate(m.y), saturate(m.z), saturate(m.w)); }
    static MATH_FORCEINLINE floatNx3 saturate(const floatNx3& m) { return floatNx3(saturate(m.x), saturate(m.y), saturate(m.z)); }
    static MATH_FORCEINLINE floatNx2 saturate(const floatNx2& m) { return floatNx2(saturate(m.x), saturate(m.y)); }
    static MATH_FORCEINLINE floatNx4 lerp(const floatNx4& a, const floatNx4& b, const floatN& c) { return floatNx4(lerp(a.x, b.x, c), lerp(a.y, b.y, c), lerp(a.z, b.z, c), lerp(a.w, b.w, c)); }
    static MATH_FORCEINLINE floatNx3 lerp(const floatNx3& a, const floatNx3& b, const floatN& c) { return floatNx3(lerp(a.x, b.x, c), lerp(a.y, b.y, c), lerp(a.z, b.z, c)); }
    static MATH_FORCEINLINE floatNx2 lerp(const floatNx2& a, const floatNx2& b, const floatN& c) { return floatNx2(lerp(a.x, b.x, c), lerp(a.y, b.y, c)); }
    static MATH_FORCEINLINE Vector3f extract(const floatNx3& v, unsigned int i) { return Vector3f(extract(v.x, i), extract(v.y, i), extract(v.z, i)); }
    static MATH_FORCEINLINE Vector2f extract(const floatNx2& v, unsigned int i) { return Vector2f(extract(v.x, i), extract(v.y, i)); }
    static MATH_FORCEINLINE void insert(floatNx3 &v, unsigned int axis, unsigned int index, float value) { reinterpret_cast<float*>(&v)[kSimdWidth * axis + index] = value; }
    static MATH_FORCEINLINE void insert(floatNx2 &v, unsigned int axis, unsigned int index, float value) { reinterpret_cast<float*>(&v)[kSimdWidth * axis + index] = value; }

    // multiply N positions by 1 matrix
    static MATH_FORCEINLINE floatNx3 mul(const affineX& m, const floatNx3& i)
    {
        floatNx3 result;
        result.x = mad(m.rs.m0.x, i.x, mad(m.rs.m1.x, i.y, mad(m.rs.m2.x, i.z, m.t.x)));
        result.y = mad(m.rs.m0.y, i.x, mad(m.rs.m1.y, i.y, mad(m.rs.m2.y, i.z, m.t.y)));
        result.z = mad(m.rs.m0.z, i.x, mad(m.rs.m1.z, i.y, mad(m.rs.m2.z, i.z, m.t.z)));
        return result;
    }

    // multiply N vectors by 1 matrix
    static MATH_FORCEINLINE floatNx3 mul(const float3x3& m, const floatNx3& i)
    {
        floatNx3 result;
        result.x = mad(m.m0.x, i.x, mad(m.m1.x, i.y, m.m2.x * i.z));
        result.y = mad(m.m0.y, i.x, mad(m.m1.y, i.y, m.m2.y * i.z));
        result.z = mad(m.m0.z, i.x, mad(m.m1.z, i.y, m.m2.z * i.z));
        return result;
    }

    // multiply N matrices by N matrices
    static MATH_FORCEINLINE void mul(const floatNx3* lhs, const floatNx3* rhs, floatNx3* out)
    {
        out[0].x = mad(lhs[0].x, rhs[0].x, mad(lhs[1].x, rhs[0].y, lhs[2].x * rhs[0].z));
        out[1].x = mad(lhs[0].x, rhs[1].x, mad(lhs[1].x, rhs[1].y, lhs[2].x * rhs[1].z));
        out[2].x = mad(lhs[0].x, rhs[2].x, mad(lhs[1].x, rhs[2].y, lhs[2].x * rhs[2].z));

        out[0].y = mad(lhs[0].y, rhs[0].x, mad(lhs[1].y, rhs[0].y, lhs[2].y * rhs[0].z));
        out[1].y = mad(lhs[0].y, rhs[1].x, mad(lhs[1].y, rhs[1].y, lhs[2].y * rhs[1].z));
        out[2].y = mad(lhs[0].y, rhs[2].x, mad(lhs[1].y, rhs[2].y, lhs[2].y * rhs[2].z));

        out[0].z = mad(lhs[0].z, rhs[0].x, mad(lhs[1].z, rhs[0].y, lhs[2].z * rhs[0].z));
        out[1].z = mad(lhs[0].z, rhs[1].x, mad(lhs[1].z, rhs[1].y, lhs[2].z * rhs[1].z));
        out[2].z = mad(lhs[0].z, rhs[2].x, mad(lhs[1].z, rhs[2].y, lhs[2].z * rhs[2].z));
    }

    // multiply N vectors by N matrices
    static MATH_FORCEINLINE floatNx3 mul(const floatNx3* m, const floatNx3& i)
    {
        floatNx3 result;
        result.x = mad(m[0].x, i.x, mad(m[1].x, i.y, m[2].x * i.z));
        result.y = mad(m[0].y, i.x, mad(m[1].y, i.y, m[2].y * i.z));
        result.z = mad(m[0].z, i.x, mad(m[1].z, i.y, m[2].z * i.z));
        return result;
    }

    // multiply N vectors by N matrices
    static MATH_FORCEINLINE floatNx3 transposeMul(const floatNx3* m, const floatNx3& i)
    {
        floatNx3 result;
        result.x = mad(m[0].x, i.x, mad(m[0].y, i.y, m[0].z * i.z));
        result.y = mad(m[1].x, i.x, mad(m[1].y, i.y, m[1].z * i.z));
        result.z = mad(m[2].x, i.x, mad(m[2].y, i.y, m[2].z * i.z));
        return result;
    }

    // convert N euler vectors to N matrices
    static MATH_FORCEINLINE void eulerToMatrix(const floatNx3& v, floatNx3* out)
    {
        floatN sx, cx, sy, cy, sz, cz;
        sincos(v.x, sx, cx);
        sincos(v.y, sy, cy);
        sincos(v.z, sz, cz);

        out[0].x = mad(cy, cz, sx * sy * sz);
        out[1].x = cz * sx * sy - cy * sz;
        out[2].x = cx * sy;

        out[0].y = cx * sz;
        out[1].y = cx * cz;
        out[2].y = -sx;

        out[0].z = mad(-cz, sy, cy * sx * sz);
        out[1].z = mad(cy * cz, sx, sy * sz);
        out[2].z = cx * cy;
    }

    // convert gamma to linear color space (approximate, because uses math::powr)
    static inline floatN GammaToLinearSpaceApprox(const floatN& value)
    {
        floatN x = value;
        floatN y = floatN(2.2f);

        intN isSmall = (value <= 0.04045f);
        x = select(x, value * (1.0f / 12.92f), isSmall);
        y = select(y, 1.0f, isSmall);

        intN lessThanOne = (value < 1.0f);
        x = select(x, (value + 0.055f) * (1.0f / 1.055f), lessThanOne);
        y = select(y, 2.4f, lessThanOne);

        return powr(x, y);
    }

    static inline floatNx4 GammaToLinearSpaceApprox(const floatNx4& colors)
    {
        return floatNx4(
            GammaToLinearSpaceApprox(colors.x),
            GammaToLinearSpaceApprox(colors.y),
            GammaToLinearSpaceApprox(colors.z),
            colors.w);
    }

    static MATH_FORCEINLINE floatN increasing_floats(const float1& base) { return float4(base, base + 1.0f, base + 2.0f, base + 3.0f); }
    static MATH_FORCEINLINE intN increasing_ints(const int1& base) { return int4(base, base + 1, base + 2, base + 3); }
    template<typename T> static MATH_FORCEINLINE floatN increasing_floats(const T* arr, size_t baseIndex) { return float4(arr[baseIndex], arr[baseIndex + 1], arr[baseIndex + 2], arr[baseIndex + 3]); }
    template<typename T> static MATH_FORCEINLINE intN increasing_ints(const T* arr, size_t baseIndex) { return int4(arr[baseIndex], arr[baseIndex + 1], arr[baseIndex + 2], arr[baseIndex + 3]); }

    static MATH_FORCEINLINE floatN convert_floatN(const intN& v) { return convert_float4(v); }
    static MATH_FORCEINLINE intN convert_intN(const floatN& v) { return convert_int4(v); }

    static MATH_FORCEINLINE pixN convert_pixN(const floatNx4& v)
    {
        floatNx4 scaled = mul(saturate(v), floatN(255.0f));
        scaled += float1(0.5f);

        // For an unknown reason, clang seems to have an issue with variadic parameters
        // like int4_ctor(), we need to explictly cast the constant 0xff000000 to int to make it pass
        pixN result = pixN(bitselect(int4(ZERO), convert_intN(scaled.x), int4((int)0x000000ff)));
        result.i = bitselect(result.i, shiftLeftLogical(convert_intN(scaled.y), 8),  int4((int)0x0000ff00));
        result.i = bitselect(result.i, shiftLeftLogical(convert_intN(scaled.z), 16), int4((int)0x00ff0000));
        result.i = bitselect(result.i, shiftLeftLogical(convert_intN(scaled.w), 24), int4((int)0xff000000));
        return result;
    }

    static MATH_FORCEINLINE floatNx4 convert_floatNx4(const pixN& v)
    {
        intN x = (v.i & 0x000000ff);
        intN y = shiftRightLogical(v.i & intN(0x0000ff00), 8);
        intN z = shiftRightLogical(v.i & intN(0x00ff0000), 16);
        intN w = shiftRightLogical(v.i & intN(0xff000000), 24);

        floatNx4 result;
        result.x = convert_floatN(x) / floatN(255.0f);
        result.y = convert_floatN(y) / floatN(255.0f);
        result.z = convert_floatN(z) / floatN(255.0f);
        result.w = convert_floatN(w) / floatN(255.0f);

        return result;
    }

    static MATH_FORCEINLINE floatN vloadNf_aligned(const float* ptr) { return vload4f_aligned(ptr); }
    static MATH_FORCEINLINE floatN vloadNf(const float* ptr) { return vload4f(ptr); }
    static MATH_FORCEINLINE intN vloadNi_aligned(const int* ptr) { return vload4i_aligned(ptr); }
    static MATH_FORCEINLINE intN vloadNi(const int* ptr) { return vload4i(ptr); }
    static MATH_FORCEINLINE void vstoreNf_aligned(float* ptr, const floatN& v) { return vstore4f_aligned(ptr, v); }
    static MATH_FORCEINLINE void vstoreNf(float* ptr, const floatN& v) { return vstore4f(ptr, v); }
    static MATH_FORCEINLINE void vstoreNi_aligned(int* ptr, const intN& v) { return vstore4i_aligned(ptr, v); }
    static MATH_FORCEINLINE void vstoreNi(int* ptr, const intN& v) { return vstore4i(ptr, v); }
    static MATH_FORCEINLINE void vstreamNf_aligned(float* ptr, const floatN& v) { return vstream4f_aligned(ptr, v); }
    static MATH_FORCEINLINE void vstreamNi_aligned(int* ptr, const intN& v) { return vstream4i_aligned(ptr, v); }
}

#pragma once

#include "Runtime/Utilities/Prefetch.h"


inline void MultiplyMatrices4x4NATIVE(const Simd128& m10, const Simd128& m11, const Simd128& m12, const Simd128& m13, const Simd128& m20, const Simd128& m21, const Simd128& m22, const Simd128& m23, Simd128& rm0, Simd128& rm1, Simd128& rm2, Simd128& rm3)
{
    const Simd128 m20_X = V4Splat(m20, 0);
    const Simd128 m21_X = V4Splat(m21, 0);
    const Simd128 m22_X = V4Splat(m22, 0);
    const Simd128 m23_X = V4Splat(m23, 0);
    const Simd128 rm0_0 = V4Mul(m20_X, m10);
    const Simd128 rm1_0 = V4Mul(m21_X, m10);
    const Simd128 rm2_0 = V4Mul(m22_X, m10);
    const Simd128 rm3_0 = V4Mul(m23_X, m10);
    const Simd128 m20_Y = V4Splat(m20, 1);
    const Simd128 m21_Y = V4Splat(m21, 1);
    const Simd128 m22_Y = V4Splat(m22, 1);
    const Simd128 m23_Y = V4Splat(m23, 1);
    const Simd128 rm0_1 = V4MulAdd(m20_Y, m11, rm0_0);
    const Simd128 rm1_1 = V4MulAdd(m21_Y, m11, rm1_0);
    const Simd128 rm2_1 = V4MulAdd(m22_Y, m11, rm2_0);
    const Simd128 rm3_1 = V4MulAdd(m23_Y, m11, rm3_0);
    const Simd128 m20_Z = V4Splat(m20, 2);
    const Simd128 m21_Z = V4Splat(m21, 2);
    const Simd128 m22_Z = V4Splat(m22, 2);
    const Simd128 m23_Z = V4Splat(m23, 2);
    const Simd128 rm0_2 = V4MulAdd(m20_Z, m12, rm0_1);
    const Simd128 rm1_2 = V4MulAdd(m21_Z, m12, rm1_1);
    const Simd128 rm2_2 = V4MulAdd(m22_Z, m12, rm2_1);
    const Simd128 rm3_2 = V4MulAdd(m23_Z, m12, rm3_1);
    const Simd128 m20_W = V4Splat(m20, 3);
    const Simd128 m21_W = V4Splat(m21, 3);
    const Simd128 m22_W = V4Splat(m22, 3);
    const Simd128 m23_W = V4Splat(m23, 3);
    rm0 = V4MulAdd(m20_W, m13 , rm0_2);
    rm1 = V4MulAdd(m21_W, m13 , rm1_2);
    rm2 = V4MulAdd(m22_W, m13 , rm2_2);
    rm3 = V4MulAdd(m23_W, m13 , rm3_2);
}

inline void MultiplyMatrices4x4(const Matrix4x4f* __restrict lhs, const Matrix4x4f* __restrict rhs, Matrix4x4f* __restrict res)
{
    Assert(lhs != rhs && lhs != res && rhs != res);
    float* m            = res->m_Data;
    const float* m1     = lhs->m_Data;
    const float* m2     = rhs->m_Data;
    Simd128 rm0, rm1, rm2, rm3;

    Prefetch((const char*)m1);
    Prefetch((const char*)m2);

    const Simd128 m10   = V4LoadUnaligned(m1, 0x0);
    const Simd128 m11   = V4LoadUnaligned(m1, 0x4);
    const Simd128 m12   = V4LoadUnaligned(m1, 0x8);
    const Simd128 m13   = V4LoadUnaligned(m1, 0xC);

    const Simd128 m20   = V4LoadUnaligned(m2, 0x0);
    const Simd128 m21   = V4LoadUnaligned(m2, 0x4);
    const Simd128 m22   = V4LoadUnaligned(m2, 0x8);
    const Simd128 m23   = V4LoadUnaligned(m2, 0xC);

    MultiplyMatrices4x4NATIVE(m10, m11, m12, m13,   m20, m21, m22, m23, rm0, rm1, rm2, rm3);

    V4StoreUnaligned(rm0, m, 0x0);
    V4StoreUnaligned(rm1, m, 0x4);
    V4StoreUnaligned(rm2, m, 0x8);
    V4StoreUnaligned(rm3, m, 0xC);
}

#if UNITY_USE_COPYMATRIX_4X4
inline void CopyMatrix4x4(const float* __restrict lhs, float* __restrict res)
{
    Simd128 r0 =  V4LoadUnaligned(lhs, 0x0);
    Simd128 r1 =  V4LoadUnaligned(lhs, 0x4);
    Simd128 r2 =  V4LoadUnaligned(lhs, 0x8);
    Simd128 r3 =  V4LoadUnaligned(lhs, 0xC);
    V4StoreUnaligned(r0, res, 0x0);
    V4StoreUnaligned(r1, res, 0x4);
    V4StoreUnaligned(r2, res, 0x8);
    V4StoreUnaligned(r3, res, 0xC);
}

#endif

inline void MultiplyMatrixArrayWithBase4x4(const Matrix4x4f* __restrict base,
    const Matrix4x4f* __restrict a, const Matrix4x4f* __restrict b, Matrix4x4f* __restrict res, size_t count)
{
    const float* mbase      = base->m_Data;
    Prefetch((const char*)mbase);

    const Simd128 base0 = V4LoadUnaligned(mbase, 0x0);
    const Simd128 base1 = V4LoadUnaligned(mbase, 0x4);
    const Simd128 base2 = V4LoadUnaligned(mbase, 0x8);
    const Simd128 base3 = V4LoadUnaligned(mbase, 0xC);

    for (size_t i = 0; i < count; ++i)
    {
        float* m            = res[i].m_Data;
        const float* m1     = a[i].m_Data;
        const float* m2     = b[i].m_Data;
        Prefetch((const char*)m1);
        Prefetch((const char*)m2);
        const Simd128 m10   = V4LoadUnaligned(m1, 0x0);
        const Simd128 m11   = V4LoadUnaligned(m1, 0x4);
        const Simd128 m12   = V4LoadUnaligned(m1, 0x8);
        const Simd128 m13   = V4LoadUnaligned(m1, 0xC);
        const Simd128 m20   = V4LoadUnaligned(m2, 0x0);
        const Simd128 m21   = V4LoadUnaligned(m2, 0x4);
        const Simd128 m22   = V4LoadUnaligned(m2, 0x8);
        const Simd128 m23   = V4LoadUnaligned(m2, 0xC);

        Simd128 b20, b21, b22, b23, rb0, rb1, rb2, rb3;
        MultiplyMatrices4x4NATIVE(m10, m11, m12, m13, m20, m21, m22, m23, b20, b21, b22, b23);
        MultiplyMatrices4x4NATIVE(base0, base1, base2, base3, b20, b21, b22, b23, rb0, rb1, rb2, rb3);

        V4StoreUnaligned(rb0, m, 0x0);
        V4StoreUnaligned(rb1, m, 0x4);
        V4StoreUnaligned(rb2, m, 0x8);
        V4StoreUnaligned(rb3, m, 0xC);
    }
}

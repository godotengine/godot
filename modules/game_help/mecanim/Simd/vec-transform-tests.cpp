#include "UnityPrefix.h"
#if ENABLE_UNIT_TESTS
#include "Runtime/Testing/Testing.h"
#include "Runtime/Profiler/TimeHelper.h"

#if PLATFORM_WIN
#pragma warning(push)
#pragma warning(disable: 4723) // potential division by zero
#endif

#include "vec-types.h"
#include "vec-math.h"
#include "vec-quat.h"
#include "vec-matrix.h"
#include "vec-affine.h"
#include "vec-transform.h"

UNIT_TEST_SUITE(SIMDMath_transformOps)
{
    using namespace math;

    TEST(inverseScale_Works)
    {
        float3 s = float3(1, 2, 4);
        float3 sInv = inverseScale(s);
        CHECK(all(sInv == float3(1, 0.5f, 0.25f)));

        s = float3(-4, 2, -1);
        sInv = inverseScale(s);
        CHECK(all(sInv == float3(-0.25f, 0.5f, -1)));

        s = float3(0.9f * epsilon_scale(), 2, 1);
        sInv = inverseScale(s);
        CHECK(all(sInv == float3(0, 0.5f, 1)));

        s = float3(-2, -0.9f * epsilon_scale(), 0);
        sInv = inverseScale(s);
        CHECK(all(sInv == float3(-0.5f, 0, 0)));
    }

    TEST(transpose_Works)
    {
        float3x3 matrix;
        matrix.m0 = float3(0, 1, 2);
        matrix.m1 = float3(4, 5, 6);
        matrix.m2 = float3(8, 9, 10);

        matrix = transpose(matrix);

        CHECK(all(matrix.m0 == float3(0, 4, 8)));
        CHECK(all(matrix.m1 == float3(1, 5, 9)));
        CHECK(all(matrix.m2 == float3(2, 6, 10)));
    }

    TEST(inverse_WorksFor_affineXWithTRS)
    {
        const float tolerance = 1e-6f;

        affineX t = affineCompose(float3(5, 6, 7), normalize(float4(1, 2, 3, 4)), float3(2, 3, 4));
        affineX tInv = inverse(t);

        affineX testI = mul(t, tInv);

        CHECK_CLOSE(1, (float)testI.rs.m0.x, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m0.y, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m0.z, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m1.x, tolerance);
        CHECK_CLOSE(1, (float)testI.rs.m1.y, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m1.z, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m2.x, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m2.y, tolerance);
        CHECK_CLOSE(1, (float)testI.rs.m2.z, tolerance);

        CHECK_CLOSE(0, (float)testI.t.x, tolerance);
        CHECK_CLOSE(0, (float)testI.t.y, tolerance);
        CHECK_CLOSE(0, (float)testI.t.z, tolerance);
    }

    TEST(inverse_WorksFor_SingularAffineX)
    {
        // it needs a bit larger tolerance for singular
        // since it uses the svd iterative solver
        const float tolerance = 5e-5f;

        affineX t = affineCompose(float3(5, 6, 7), normalize(float4(1, 2, 3, 4)), float3(0, 3, 4));
        affineX tInv = inverse(t);

        // pseudo inverse pernose #1 test
        affineX testT = mul(mul(t, tInv), t);

        CHECK_CLOSE(t.rs.m0.x, (float)testT.rs.m0.x, tolerance);
        CHECK_CLOSE(t.rs.m0.y, (float)testT.rs.m0.y, tolerance);
        CHECK_CLOSE(t.rs.m0.z, (float)testT.rs.m0.z, tolerance);
        CHECK_CLOSE(t.rs.m1.x, (float)testT.rs.m1.x, tolerance);
        CHECK_CLOSE(t.rs.m1.y, (float)testT.rs.m1.y, tolerance);
        CHECK_CLOSE(t.rs.m1.z, (float)testT.rs.m1.z, tolerance);
        CHECK_CLOSE(t.rs.m2.x, (float)testT.rs.m2.x, tolerance);
        CHECK_CLOSE(t.rs.m2.y, (float)testT.rs.m2.y, tolerance);
        CHECK_CLOSE(t.rs.m2.z, (float)testT.rs.m2.z, tolerance);

        CHECK_CLOSE(t.t.x, (float)testT.t.x, tolerance);
        CHECK_CLOSE(t.t.y, (float)testT.t.y, tolerance);
        CHECK_CLOSE(t.t.z, (float)testT.t.z, tolerance);
    }

    TEST(inverse_WorksFor_affineXWithTR_PicoScale)
    {
        // slightly larger tolerance (vs 1e-6f) for PS4
        const float tolerance = 2e-6f;

        affineX t = affineCompose(float3(5, 6, 7), normalize(float4(1, 2, 3, 4)), float3(1e-12f, 1e-12f, 1e-12f));
        affineX tInv = inverse(t);

        affineX testI = mul(t, tInv);

        CHECK_CLOSE(1, (float)testI.rs.m0.x, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m0.y, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m0.z, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m1.x, tolerance);
        CHECK_CLOSE(1, (float)testI.rs.m1.y, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m1.z, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m2.x, tolerance);
        CHECK_CLOSE(0, (float)testI.rs.m2.y, tolerance);
        CHECK_CLOSE(1, (float)testI.rs.m2.z, tolerance);

        CHECK_CLOSE(0, (float)testI.t.x, tolerance);
        CHECK_CLOSE(0, (float)testI.t.y, tolerance);
        CHECK_CLOSE(0, (float)testI.t.z, tolerance);
    }

    TEST(inverse_Returns_ZeroAffineX_For_ZeroRS)
    {
        const float tolerance = 0;

        affineX t = affineCompose(float3(5, 6, 7), normalize(float4(1, 2, 3, 4)), float3(0, 0, 0));
        affineX tInv = inverse(t);

        CHECK_CLOSE(0, (float)tInv.rs.m0.x, tolerance);
        CHECK_CLOSE(0, (float)tInv.rs.m0.y, tolerance);
        CHECK_CLOSE(0, (float)tInv.rs.m0.z, tolerance);
        CHECK_CLOSE(0, (float)tInv.rs.m1.x, tolerance);
        CHECK_CLOSE(0, (float)tInv.rs.m1.y, tolerance);
        CHECK_CLOSE(0, (float)tInv.rs.m1.z, tolerance);
        CHECK_CLOSE(0, (float)tInv.rs.m2.x, tolerance);
        CHECK_CLOSE(0, (float)tInv.rs.m2.y, tolerance);
        CHECK_CLOSE(0, (float)tInv.rs.m2.z, tolerance);

        CHECK_CLOSE(0, (float)tInv.t.x, tolerance);
        CHECK_CLOSE(0, (float)tInv.t.y, tolerance);
        CHECK_CLOSE(0, (float)tInv.t.z, tolerance);
    }

    TEST(adjInverse_WorksFor_float3x3WithNanoScale)
    {
        float3x3 matrix;
        matrix.m0 = float3(1e-9f, 0, 0);
        matrix.m1 = float3(0, 1e-9f, 0);
        matrix.m2 = float3(0, 0, 1e-9f);

        bool res = adjInverse(matrix, matrix);

        CHECK(res);
        CHECK_CLOSE(1e9f, (float)matrix.m0.x, 1.0f);
        CHECK_CLOSE(1e9f, (float)matrix.m1.y, 1.0f);
        CHECK_CLOSE(1e9f, (float)matrix.m2.z, 1.0f);
    }

    TEST(rotation_works_for_rotation_matrix)
    {
        float4 qi = normalize(float4(1, 2, 3, 4));
        float3x3 matrix;
        quatToMatrix(qi, matrix);
        matrix = rotation(matrix);
        float4 qo = matrixToQuat(matrix);

        float d = quatDiff(qi, qo);
        CHECK_CLOSE(0, d, 1e-4);
    }

    TEST(rotation_for_rotation_matrix_with_scale)
    {
        float4 qi = normalize(float4(1, 2, 3, 4));
        float3x3 matrix;
        quatToMatrix(qi, matrix);
        matrix = mulScale(matrix, float3(2, 3, 4));
        matrix = rotation(matrix);
        float4 qo = matrixToQuat(matrix);

        float d = quatDiff(qi, qo);
        CHECK_CLOSE(0, d, 1e-4);
    }

    TEST(rotation_works_for_rotation_matrix_with_shear)
    {
        float3x3 mi;
        float4 qi = normalize(float4(4, 3, 2, 1));
        quatToMatrix(qi, mi);

        float3x3 pmi;
        float4 pqi = normalize(float4(1, 2, 3, 4));
        quatToMatrix(pqi, pmi);
        pmi = mulScale(pmi, float3(2, 3, 4));

        float3x3 mo = mul(pmi, mi);

        // ensure out matrix is effectively sheared (aka det < < 1)
        float3x3 mos = mulScale(mo, rsqrt(float3(dot(mo.m0), dot(mo.m1), dot(mo.m2))));
        CHECK(det(mos) < 0.85f);

        float3x3 rmo = rotation(mo);
        float4 qo = matrixToQuat(rmo);

        float d = quatDiff(quatMul(pqi, qi), qo);
        CHECK_CLOSE(0, d, 1e-4);
    }
}

#if PLATFORM_WIN
#pragma warning(pop)
#endif

#endif

#include "UnityPrefix.h"
#if ENABLE_UNIT_TESTS
#include "Runtime/Testing/Testing.h"
#include "Runtime/Profiler/TimeHelper.h"

#include "vec-types.h"
#include "vec-math.h"
#include "vec-quat.h"
#include "vec-matrix.h"
#include "vec-svd.h"

UNIT_TEST_SUITE(SIMDMath_svdOps)
{
    using namespace math;

    const float svdTolerance = 1e-4f;

    void CHECK_SINGULAR(const float3x3 &S)
    {
        float d = det(S);
        CHECK_CLOSE(0, d, svdTolerance);
    }

    void CHECK_PERNOSE_12(const float3x3 &A, const float3x3 &G) // AGA=A
    {
        float3x3 testA = mul(mul(A, G), A);

        CHECK_CLOSE((float)A.m0.x, (float)testA.m0.x, svdTolerance);
        CHECK_CLOSE((float)A.m0.y, (float)testA.m0.y, svdTolerance);
        CHECK_CLOSE((float)A.m0.z, (float)testA.m0.z, svdTolerance);
        CHECK_CLOSE((float)A.m1.x, (float)testA.m1.x, svdTolerance);
        CHECK_CLOSE((float)A.m1.y, (float)testA.m1.y, svdTolerance);
        CHECK_CLOSE((float)A.m1.z, (float)testA.m1.z, svdTolerance);
        CHECK_CLOSE((float)A.m2.x, (float)testA.m2.x, svdTolerance);
        CHECK_CLOSE((float)A.m2.y, (float)testA.m2.y, svdTolerance);
        CHECK_CLOSE((float)A.m2.z, (float)testA.m2.z, svdTolerance);
    }

    void CHECK_PERNOSE_34(const float3x3 &A, const float3x3 &G) // TRANSPOSE(AG) = AG
    {
        float3x3 AG = mul(A, G);
        float3x3 testAG = transpose(AG);

        CHECK_CLOSE((float)AG.m0.x, (float)testAG.m0.x, svdTolerance);
        CHECK_CLOSE((float)AG.m0.y, (float)testAG.m0.y, svdTolerance);
        CHECK_CLOSE((float)AG.m0.z, (float)testAG.m0.z, svdTolerance);
        CHECK_CLOSE((float)AG.m1.x, (float)testAG.m1.x, svdTolerance);
        CHECK_CLOSE((float)AG.m1.y, (float)testAG.m1.y, svdTolerance);
        CHECK_CLOSE((float)AG.m1.z, (float)testAG.m1.z, svdTolerance);
        CHECK_CLOSE((float)AG.m2.x, (float)testAG.m2.x, svdTolerance);
        CHECK_CLOSE((float)AG.m2.y, (float)testAG.m2.y, svdTolerance);
        CHECK_CLOSE((float)AG.m2.z, (float)testAG.m2.z, svdTolerance);
    }

    TEST(svdInverse_WorksFor_Non_Singular_float3x3)
    {
        float3x3 matrix;
        matrix.m0 = float3(9, 1, 2);
        matrix.m1 = float3(3, 8, 4);
        matrix.m2 = float3(5, 6, 7);

        float3x3 inv = svdInverse(matrix);

        float3x3 testI = mul(matrix, inv);

        CHECK_CLOSE(1, (float)testI.m0.x, svdTolerance);
        CHECK_CLOSE(0, (float)testI.m0.y, svdTolerance);
        CHECK_CLOSE(0, (float)testI.m0.z, svdTolerance);
        CHECK_CLOSE(0, (float)testI.m1.x, svdTolerance);
        CHECK_CLOSE(1, (float)testI.m1.y, svdTolerance);
        CHECK_CLOSE(0, (float)testI.m1.z, svdTolerance);
        CHECK_CLOSE(0, (float)testI.m2.x, svdTolerance);
        CHECK_CLOSE(0, (float)testI.m2.y, svdTolerance);
        CHECK_CLOSE(1, (float)testI.m2.z, svdTolerance);
    }

    TEST(svdInverse_WorksFor_Null_Column_float3x3)
    {
        float3x3 matrix;
        matrix.m0 = float3(9, 1, 0);
        matrix.m1 = float3(3, 8, 0);
        matrix.m2 = float3(5, 6, 0);

        CHECK_SINGULAR(matrix);

        float3x3 inv = svdInverse(matrix);

        CHECK_PERNOSE_12(matrix, inv);
        CHECK_PERNOSE_12(inv, matrix);
        CHECK_PERNOSE_34(matrix, inv);
        CHECK_PERNOSE_34(inv, matrix);
    }

    TEST(svdInverse_WorksFor_Null_Row_float3x3)
    {
        float3x3 matrix;
        matrix.m0 = float3(9, 1, 2);
        matrix.m1 = float3(0, 0, 0);
        matrix.m2 = float3(5, 6, 7);

        CHECK_SINGULAR(matrix);

        float3x3 inv = svdInverse(matrix);

        CHECK_PERNOSE_12(matrix, inv);
        CHECK_PERNOSE_12(inv, matrix);
        CHECK_PERNOSE_34(matrix, inv);
        CHECK_PERNOSE_34(inv, matrix);
    }

    TEST(svdInverse_WorksFor_Linear_Dependent_Column_float3x3)
    {
        float3x3 matrix;
        matrix.m0 = float3(9, 4, 2);
        matrix.m1 = float3(3, 8, 4);
        matrix.m2 = float3(5, 14, 7);

        CHECK_SINGULAR(matrix);

        float3x3 inv = svdInverse(matrix);

        CHECK_PERNOSE_12(matrix, inv);
        CHECK_PERNOSE_12(inv, matrix);
        CHECK_PERNOSE_34(matrix, inv);
        CHECK_PERNOSE_34(inv, matrix);
    }

    TEST(svdInverse_WorksFor_Linear_Dependent_Row_float3x3)
    {
        float3x3 matrix;
        matrix.m0 = float3(9, 1, 2);
        matrix.m1 = float3(10, 12, 14);
        matrix.m2 = float3(5, 6, 7);

        CHECK_SINGULAR(matrix);

        float3x3 inv = svdInverse(matrix);

        CHECK_PERNOSE_12(matrix, inv);
        CHECK_PERNOSE_12(inv, matrix);
        CHECK_PERNOSE_34(matrix, inv);
        CHECK_PERNOSE_34(inv, matrix);
    }

    TEST(svdInverse_WorksFor_Rotated_Zero_Scale_float3x3)
    {
        float4 q102030 = eulerToQuat(math::radians(float3(10, 20, 30)));
        float3x3 rm102030;
        quatToMatrix(q102030, rm102030);

        float3x3 parent = mulScale(rm102030, float3(1, 1, 0));
        float3x3 matrix = mul(parent, rm102030);

        CHECK_SINGULAR(matrix);

        float3x3 inv = svdInverse(matrix);

        CHECK_PERNOSE_12(matrix, inv);
        CHECK_PERNOSE_12(inv, matrix);
        CHECK_PERNOSE_34(matrix, inv);
        CHECK_PERNOSE_34(inv, matrix);
    }

    // Case 928598: The errors appear, when GameObject has a child with ParticleSystem which is rotated along the y-axis to -180 and is moved
    TEST(svdRotation_WorksFor_X180Y0Z181_float3x3)
    {
        float4 qi = eulerToQuat(math::radians(float3(180.0f, 0, 181.0f)));
        float3x3 rm;
        quatToMatrix(qi, rm);
        float4 qo = svdRotation(rm);

        float d = quatDiff(qi, qo);
        CHECK_CLOSE(0, d, svdTolerance);
    }

    // Case 938548: Assertion failed on expression: 'CompareApproximately(det, 1.0F, .005f)' when scaling system to 0 on at least 2 axes
    TEST(svdRotation_WorksFor_0_Scale_YZ_float3x3)
    {
        float4 qi = eulerToQuat(math::radians(float3(10, 20, 30)));
        float3x3 rm;
        quatToMatrix(qi, rm);
        float3x3 rmScaled = mulScale(rm, float3(1, 0, 0));
        float4 qo = svdRotation(rmScaled);

        float3x3 rmi;
        quatToMatrix(qi, rmi);

        float3x3 rmo;
        quatToMatrix(qo, rmo);

        float d = length(rmi.m0 - rmo.m0);
        CHECK_CLOSE(0, d, svdTolerance);
    }
}

#endif

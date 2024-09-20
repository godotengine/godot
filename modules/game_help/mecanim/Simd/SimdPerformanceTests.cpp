#include "UnityPrefix.h"

#if ENABLE_UNIT_TESTS && ENABLE_PERFORMANCE_TESTS

#if PLATFORM_WIN
#pragma warning(push)
#pragma warning(disable: 4723) // potential division by zero
#endif

#include "vec-types.h"
#include "vec-math.h"
#include "vec-quat.h"
#include "vec-matrix.h"
#include "vec-trs.h"
#include "vec-affine.h"
#include "vec-transform.h"
#include "vec-soa-types.h"
#include "vec-soa.h"

#include "Runtime/Testing/Testing.h"
#include "Runtime/Testing/PerformanceTesting.h"
#include "Runtime/Testing/TestFixtures.h"
#include "Runtime/Allocator/MemoryManager.h"
#include "Runtime/Threads/Thread.h"
#include "Runtime/Utilities/dynamic_array.h"
#include "Runtime/Math/Random/Random.h"

#include "Runtime/Geometry/Intersection.h"
#include "Runtime/Geometry/Plane.h"
#include "Runtime/Geometry/AABB.h"

#include "Modules/Animation/mecanim/math/axes.h"

class SIMDMathTransformFixture : public TestFixtureBase
{
public:

    SIMDMathTransformFixture()
        : m_LocalTransforms(kMemTest)
        , m_ParentIndices(kMemTest)
    {}

    dynamic_array<math::trsX> m_LocalTransforms;
    dynamic_array<SInt32>     m_ParentIndices;
    Rand                      m_Rand;

    // copy from Runtime\Transform\TransformHierarchy.h
    UNITY_NOINLINE math::affineX CalculateGlobalMatrix(SInt32 index)
    {
        using namespace math;

        affineX parentX;
        affineX globalX;

        convert(m_LocalTransforms[index], globalX);

        SInt32 parentIndex = m_ParentIndices[index];

        while (parentIndex >= 0)
        {
            convert(m_LocalTransforms[parentIndex], parentX);
            globalX = mul(parentX, globalX);
            parentIndex = m_ParentIndices[parentIndex];
        }

        return globalX;
    }

    math::float3 MakePosition()
    {
        return math::float3(RangedRandom(m_Rand, -100.f, 100.f), RangedRandom(m_Rand, -100.f, 100.f), RangedRandom(m_Rand, -100.f, 100.f));
    }

    math::float4 MakeRotation()
    {
        return math::normalize(math::float4(RangedRandom(m_Rand, -1.f, 1.f), RangedRandom(m_Rand, -1.f, 1.f), RangedRandom(m_Rand, -1.f, 1.f), RangedRandom(m_Rand, -1.f, 1.f)));
    }

    math::float3 MakeScale()
    {
        // Uniform scale
        return math::float3(RangedRandom(m_Rand, 1.0f, 1.2f));
    }

    math::trsX MakeTRS()
    {
        return math::trsX(MakePosition(), MakeRotation(), MakeScale());
    }

    void MakeTransformHierarchy(size_t childCount)
    {
        m_LocalTransforms.resize_uninitialized(childCount);
        m_ParentIndices.resize_uninitialized(childCount);

        for (size_t childIter = 0; childIter < childCount; childIter++)
        {
            m_LocalTransforms[childIter] = MakeTRS();
            m_ParentIndices[childIter] = childIter - 1;
        }
    }
};

class SIMDMathIntersectFixture : public TestFixtureBase
{
public:

    dynamic_array<math::float4> m_OptimizedPlanes;

    SIMDMathIntersectFixture()
        : m_OptimizedPlanes(kMemTest)
    {}

    ~SIMDMathIntersectFixture()
    {
    }

    void MakeOptimizedPlanes(int planesCount)
    {
        // Test assumption is plane don't intersect with empty AABB
        dynamic_array<Plane> planes(planesCount, Plane(0, 1, 0, 1), kMemTest);

        m_OptimizedPlanes.resize_uninitialized(planesCount);

        PrepareOptimizedPlanes(planes.data(), planesCount, m_OptimizedPlanes.data(), planesCount);
    }
};

// function definition in Runtime/ParticleSystem/Modules/NoiseModule.cpp
math::floatNx2 Perlin1D(const math::floatNx3& point, const math::float1& frequency);
math::floatNx2 Perlin2D(const math::floatNx3& point, const math::float1& frequency);
math::floatNx2 Perlin3D(const math::floatNx3& point, const math::float1& frequency);

namespace mecanim
{
namespace animation
{
    // function definition in Modules/Animation/mecanim/animation/clipmuscle.cpp
    void LoopX(math::trsX &x, math::float3 const &t, math::float4 const &q, math::trsX const &s, bool o, bool y, bool xz);
}
}

// function definition in Runtime/ParticleSystem/ParticleSystemGeometryJob.cpp
math::float1 CalculateRollCorrection(const math::float4& cameraRotation, const math::float3& cameraDir);


PERFORMANCE_TEST_SUITE(SIMDMathPerformance)
{
    static const int kLoops = 1000;
    static const int kSmallTestLoops = 10000;

    TEST_FIXTURE(SIMDMathTransformFixture, CalculateGlobalMatrix)
    {
        const int childCount = 1000;
        MakeTransformHierarchy(childCount);

        math::affineX affineX;
        PERFORMANCE_TEST_LOOP(kLoops)
        {
            affineX = OPTIMIZER_PREVENT(CalculateGlobalMatrix(childCount - 1));
        }
        OPTIMIZER_PREVENT(affineX);
    }

    TEST_FIXTURE(SIMDMathIntersectFixture, IntersectAABBPlaneBoundsOptimized)
    {
        // We want to measure the SIMD performance for the worst case scenario
        // generate data that won't intersect with an empty AABB
        const int planesCount = 4 * 1000;
        MakeOptimizedPlanes(planesCount);

        bool intersect = true;
        PERFORMANCE_TEST_LOOP(kLoops)
        {
            intersect &= IntersectAABBPlaneBoundsOptimized(AABB::zero, m_OptimizedPlanes.data(), planesCount);
        }
        OPTIMIZER_PREVENT(intersect);

        // must always return true for worst case scenario
        CHECK(intersect);
    }

    TEST(Perlin1D)
    {
        const math::float1 frequency = 1.0f / 60.0f;
        math::floatNx3 pointNx3 = math::floatNx3(math::floatN(math::ZERO), math::floatN(1.0f), math::floatN(2.0f));
        PERFORMANCE_TEST_LOOP(kSmallTestLoops)
        {
            math::floatNx2 pointNx2 = OPTIMIZER_PREVENT(::Perlin1D(pointNx3, frequency));
            pointNx3 = math::floatNx3(pointNx2.x, pointNx2.y, pointNx3.z);
        }
        OPTIMIZER_PREVENT(pointNx3);
    }

    TEST(Perlin2D)
    {
        const math::float1 frequency = 1.0f / 60.0f;
        math::floatNx3 pointNx3 = math::floatNx3(math::floatN(math::ZERO), math::floatN(1.0f), math::floatN(2.0f));
        PERFORMANCE_TEST_LOOP(kSmallTestLoops)
        {
            math::floatNx2 pointNx2 = OPTIMIZER_PREVENT(::Perlin2D(pointNx3, frequency));
            pointNx3 = math::floatNx3(pointNx2.x, pointNx2.y, pointNx3.z);
        }
        OPTIMIZER_PREVENT(pointNx3);
    }

    TEST(Perlin3D)
    {
        const math::float1 frequency = 1.0f / 60.0f;
        math::floatNx3 pointNx3 = math::floatNx3(math::floatN(math::ZERO), math::floatN(1.0f), math::floatN(2.0f));
        PERFORMANCE_TEST_LOOP(kSmallTestLoops)
        {
            math::floatNx2 pointNx2 = OPTIMIZER_PREVENT(::Perlin3D(pointNx3, frequency));
            pointNx3 = math::floatNx3(pointNx2.x, pointNx2.y, pointNx3.z);
        }
        OPTIMIZER_PREVENT(pointNx3);
    }

    TEST(EvaluateRootMotion)
    {
        using namespace math;

        trsX rootX = trsX(float3(10, 0, 0), float4(0, 1, 0, 0), float3(1, 1, 1));
        trsX prevRootX = trsIdentity();

        PERFORMANCE_TEST_LOOP(kLoops)
        {
            bool isHuman = true;
            bool heightFromFeet = true;
            bool keepOriginalPositionY = true;
            bool keepOriginalPositionXZ = true;
            bool keepOriginalOrientation = true;
            float orientationOffsetY = 0.0f;

            trsX startX = trsX(float3(1, 2, 3), float4(1, 0, 0, 0), float3(1, 1, 1));
            trsX stopX = trsX(float3(10, 2, 3), float4(0, 0, 1, 0), float3(1, 1, 1));

            trsX refStartX = trsIdentity();
            trsX refStopX = trsIdentity();
            trsX refX = rootX;

            trsX leftFootX = trsIdentity();
            trsX rightFootX = trsIdentity();

            trsX prevLeftFootX = trsIdentity();
            trsX prevRightFootX = trsIdentity();

            refStartX.q = quatProjOnYPlane(refStartX.q);
            refStopX.q = quatProjOnYPlane(refStopX.q);
            refX.q = quatProjOnYPlane(refX.q);

            trsX prevRefX = prevRootX;
            prevRefX.q = quatProjOnYPlane(prevRefX.q);

            if (isHuman && heightFromFeet)
            {
                trsX refLeftFootStartX = mul(startX, trsIdentity());
                trsX refRightFootStartX = mul(startX, trsIdentity());

                trsX refLeftFootStopX = mul(stopX, mulInv(trsIdentity(), trsIdentity()));
                trsX refRightFootStopX = mul(stopX, mulInv(trsIdentity(), trsIdentity()));

                trsX refLeftFootX = mul(rootX, leftFootX);
                trsX refRightFootX = mul(rootX, rightFootX);

                refStartX.t.y = min(refStartX.t, min(refLeftFootStartX.t, refRightFootStartX.t)).y;
                refStopX.t.y = min(refStopX.t, min(refLeftFootStopX.t, refRightFootStopX.t)).y;
                refX.t.y = min(refX.t, min(refLeftFootX.t, refRightFootX.t)).y;

                trsX refLeftFootPrevX = mul(prevRootX, prevLeftFootX);
                trsX refRightFootPrevX = mul(prevRootX, prevRightFootX);
                prevRefX.t.y = min(prevRefX.t, min(refLeftFootPrevX.t, refRightFootPrevX.t)).y;
            }

            float3 refOffsetT = float3(ZERO);
            float4 refOffsetQ = qtan2Quat(float3(0.f, halfTan(radians(orientationOffsetY)), 0.f));

            if (keepOriginalPositionY)
            {
                refOffsetT.y -= refStartX.t.y;
            }

            if (keepOriginalPositionXZ)
            {
                refOffsetT.x -= refStartX.t.x;
                refOffsetT.z -= refStartX.t.z;
            }

            if (keepOriginalOrientation)
            {
                refOffsetQ = normalize(quatMul(refOffsetQ, quatConj(refStartX.q)));
            }

            mecanim::animation::LoopX(refStartX, refOffsetT, refOffsetQ, refStartX, false, false, true);
            mecanim::animation::LoopX(refStopX, refOffsetT, refOffsetQ, refStartX, true, true, true);
            mecanim::animation::LoopX(refX, refOffsetT, refOffsetQ, refStartX, true, true, true);
            mecanim::animation::LoopX(prevRefX, refOffsetT, refOffsetQ, refStartX, true, true, true);

            rootX = trsInvMulNS(refX, rootX);
            prevRootX = trsInvMulNS(prevRefX, prevRootX);
            prevRootX = rootX;
        }

        OPTIMIZER_PREVENT(rootX);
        OPTIMIZER_PREVENT(prevRootX);
    }

    TEST(SOA_Loop_SimilarTo_ParticleGeomBillboard)
    {
        using namespace math;
        float4 rotEulerX = OPTIMIZER_PREVENT(float4(1, 2, 3, 4));
        float4 rotEulerY = OPTIMIZER_PREVENT(float4(5, 6, 7, 8));
        float4 rotEulerZ = OPTIMIZER_PREVENT(float4(9, 10, 11, 12));
        floatNx2 hsize = floatNx2(float4(1.7f, 1.3f, 0.9f, 0.3f), float4(2.1f, 2.9f, 0.4f, 0.1f));
        hsize = OPTIMIZER_PREVENT(hsize);
        floatNx3 rotationEuler = floatNx3(rotEulerX, rotEulerY, rotEulerZ);
        floatN res = floatN(ZERO);
        PERFORMANCE_TEST_LOOP(kSmallTestLoops)
        {
            floatNx3 combinedRotation[3], particleRotation[3];
            eulerToMatrix(-rotationEuler, particleRotation);
            mul(particleRotation, particleRotation, combinedRotation);
            floatNx3 n0 = mul(combinedRotation, floatNx3(-hsize.x, hsize.y, floatN(ZERO)));
            floatNx3 n1 = mul(combinedRotation, floatNx3(hsize.x, hsize.y, floatN(ZERO)));
            rotationEuler += n0;
            res += math::dot(n0, n1);
        }
        OPTIMIZER_PREVENT(res);
    }

    TEST(CalculateRollCorrection)
    {
        using namespace math;
        float4 cameraRotation = OPTIMIZER_PREVENT(float4(1, 0, 0, 0));
        float3 cameraDir = OPTIMIZER_PREVENT(float3(0, 0, 1));
        float1 roll = float1(ZERO);
        PERFORMANCE_TEST_LOOP(kSmallTestLoops)
        {
            roll += CalculateRollCorrection(cameraRotation, cameraDir);
        }
        OPTIMIZER_PREVENT((float)roll);
    }

    TEST(EulerToQuaternionToEulerConversion)
    {
        using namespace math;
        float3 euler = float3(ZERO);
        PERFORMANCE_TEST_LOOP(kLoops)
        {
            for (int i = 0; i <= 6; i++)
            {
                for (int j = 0; j <= 6; j++)
                {
                    for (int k = 0; k <= 6; k++)
                    {
                        float4 q = eulerToQuat(euler, math::kOrderXYZ);
                        euler = OPTIMIZER_PREVENT(quatToEuler(q, math::kOrderXYZ));
                        euler.z += pi_over_six();
                    }
                    euler.y += pi_over_six();
                    euler.z = 0.0f;
                }
                euler.x += pi_over_six();
                euler.y = 0.0f;
            }
        }
        OPTIMIZER_PREVENT(euler);
    }

    TEST(QuaternionToMatrixToQuaternion)
    {
        using namespace math;
        float4 qi = normalize(float4(1, 2, 3, 4));
        float3x3 mi;
        quatToMatrix(normalize(float4(4, 3, 2, 1)), mi);
        PERFORMANCE_TEST_LOOP(kSmallTestLoops)
        {
            float3x3 matrix;
            quatToMatrix(qi, matrix);
            // Take the longest path with sheared matrix
            matrix = mulScale(matrix, float3(2, 3, 4));
            matrix = mul(matrix, mi);
            matrix = rotation(matrix);
            qi = OPTIMIZER_PREVENT(matrixToQuat(matrix));
        }
        OPTIMIZER_PREVENT(qi);
    }
}

#if PLATFORM_WIN
#pragma warning(pop)
#endif

#endif

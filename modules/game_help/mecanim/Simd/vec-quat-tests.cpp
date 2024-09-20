#include "UnityPrefix.h"
// running these native tests on webgl with multithreading will result in a stack overflow error
#if ENABLE_UNIT_TESTS && !(PLATFORM_WEBGL && PLATFORM_SUPPORTS_THREADS)
#include "Runtime/Testing/Testing.h"
#include "Runtime/Profiler/TimeHelper.h"

#include "vec-types.h"
#include "vec-math.h"
#include "vec-quat.h"
#include "Modules/Animation/mecanim/math/axes.h"
#include "Runtime/Math/Quaternion.h"
#include "Runtime/Testing/ParametricTest.h"
#include "Runtime/Math/DeprecatedConversion.h"

INTEGRATION_TEST_SUITE(SIMDMath_quaternionOps)
{
    using namespace math;

    struct Fixture
    {
        static const size_t testAngleCount = 13 * 13 * 13 * 20;
        float3_storage testAngles[testAngleCount];

        Fixture()
        {
            // Generate all the test angles
            int idx = 0;
            for (int i = 0; i < 13; i++)
            {
                float x = -math::pi() + i * math::pi_over_six();
                for (int j = 0; j < 13; j++)
                {
                    float y = -math::pi() + j * math::pi_over_six();
                    for (int k = 0; k < 13; k++)
                    {
                        float z = -math::pi() + k * math::pi_over_six();
                        for (int l = 0; l < 20; ++l)
                        {
                            testAngles[idx++] = float3(x, y, z) * (0.99f + l * 0.001f);
                        }
                    }
                }
            }
        }

        float4 testRefEulerToQuat(const float3& angle, RotationOrder order)
        {
            float3 halfRot = angle * 0.5f;
            float3 c, s;
            sincos(halfRot, s, c);

            float4 qX = float4(s.x, 0.f, 0.f, c.x);
            float4 qY = float4(0.f, s.y, 0.f, c.y);
            float4 qZ = float4(0.f, 0.f, s.z, c.z);

            switch (order)
            {
                case kOrderXYZ: return quatMul(quatMul(qZ, qY), qX);
                case kOrderXZY: return quatMul(quatMul(qY, qZ), qX);
                case kOrderYZX: return quatMul(quatMul(qX, qZ), qY);
                case kOrderYXZ: return quatMul(quatMul(qZ, qX), qY);
                case kOrderZXY: return quatMul(quatMul(qY, qX), qZ);
                case kOrderZYX: return quatMul(quatMul(qX, qY), qZ);
            }

            AssertString("Unknown rotation order");
            return float4(1.f, 1.f, 1.f, 1.f);
        }
    };

    PARAMETRIC_TEST_SOURCE(AllRotationOrders, (RotationOrder))
    {
        PARAMETRIC_TEST_CASE_WITH_NAME("XYZ", kOrderXYZ);
        PARAMETRIC_TEST_CASE_WITH_NAME("XZY", kOrderXZY);
        PARAMETRIC_TEST_CASE_WITH_NAME("YZX", kOrderYZX);
        PARAMETRIC_TEST_CASE_WITH_NAME("YXZ", kOrderYXZ);
        PARAMETRIC_TEST_CASE_WITH_NAME("ZXY", kOrderZXY);
        PARAMETRIC_TEST_CASE_WITH_NAME("ZYX", kOrderZYX);
    }

    PARAMETRIC_TEST_FIXTURE(Fixture, eulerToQuat_GivesSameResultAs_EquivalentQuatMultiply, (RotationOrder order), AllRotationOrders)
    {
        float epsilon = 4e-7f;
        float maxError = 0;

        for (size_t i = 0; i < testAngleCount; ++i)
        {
            float4 expected = testRefEulerToQuat(testAngles[i], order);
            float4 actual = eulerToQuat(testAngles[i], order);

            float error = quatDiff(expected, actual);
            CHECK_CLOSE(.0f, error, epsilon);
            maxError = max(maxError, error);
        }

        CHECK_MSG(maxError >= epsilon * 0.1f, Format("The maximum error witnessed in this test (%e) was less than 10%% of the epsilon value (%e), which means the test is too forgiving. The epsilon value should be tightened.", maxError, epsilon).c_str());
    }

    PARAMETRIC_TEST_FIXTURE(Fixture, quatToEuler_GivesResultThatConvertsToConsistentQuat, (RotationOrder order), AllRotationOrders)
    {
        float epsilon = 0.0045f;
        float maxError = 0;

        for (size_t i = 0; i < testAngleCount; ++i)
        {
            // We cannot directly compare Euler angles, as different angles may represent the same rotation, so we have to transform them back to quats to test.
            // We also need to create our initial test quaternion, as what we generated was euler angles.
            float4 expected = testRefEulerToQuat(testAngles[i], kOrderUnityDefault);

            float3 actualEulers  = quatToEuler(expected, order);
            float4 actualQuat = testRefEulerToQuat(actualEulers, order);

            float error = quatDiff(expected, actualQuat);
            CHECK_CLOSE(.0f, error, epsilon);
            maxError = max(maxError, error);
        }

        CHECK_MSG(maxError >= epsilon * 0.1f, Format("The maximum error witnessed in this test (%e) was less than 10%% of the epsilon value (%e), which means the test is too forgiving. The epsilon value should be tightened.", maxError, epsilon).c_str());
    }

    PARAMETRIC_TEST_FIXTURE(Fixture, eulerToQuat_GivesSameResultAs_LegacyNonSIMDMethod, (RotationOrder order), AllRotationOrders)
    {
        float epsilon = 1e-6f;
        float maxError = 0;

        for (size_t i = 0; i < testAngleCount; ++i)
        {
            Quaternionf legacy = EulerToQuaternion(Vector3f(testAngles[i].x, testAngles[i].y, testAngles[i].z), order);
            float4 expected = normalize(float4(legacy.x, legacy.y, legacy.z, legacy.w));

            float4 actual = normalize(eulerToQuat(testAngles[i], order));

            float error = quatDiff(expected, actual);
            CHECK_CLOSE(.0f, error, epsilon);
            maxError = max(maxError, error);
        }

        CHECK_MSG(maxError >= epsilon * 0.1f, Format("The maximum error witnessed in this test (%e) was less than 10%% of the epsilon value (%e), which means the test is too forgiving. The epsilon value should be tightened.", maxError, epsilon).c_str());
    }
}

UNIT_TEST_SUITE(SIMDMath_quaternionOps)
{
    using namespace math;
    const float epsilon = 1e-4f;

    TEST(quatMulVec_ZRotateByXResultIsMinusZ)
    {
        float4 qx = float4(1.f, 0.f, 0.f, 0.f);
        float3 vz = float3(0.f, 0.f, 1.f);
        float3 v = quatMulVec(qx, vz);
        CHECK(all(v == float3(0.f, 0.f, -1.f)));
    }

    TEST(quatMulVec_ZRotateByYResultIsMinusZ)
    {
        float4 qy = float4(0.f, 1.f, 0.f, 0.f);
        float3 vz = float3(0.f, 0.f, 1.f);
        float3 v = quatMulVec(qy, vz);
        CHECK(all(v == float3(0.f, 0.f, -1.f)));
    }

    TEST(quatMulVec_ZRotateByZResultIsZ)
    {
        float4 qz = float4(0.f, 0.f, 1.f, 0.f);
        float3 vz = float3(0.f, 0.f, 1.f);
        float3 v = quatMulVec(qz, vz);
        CHECK(all(v == float3(0.f, 0.f, 1.f)));
    }

    TEST(Axes_ConversionWork)
    {
        float4 q2 = float4(radians(79.24f), radians(-1.61f), radians(-33.15f), 0.f);
        float3 dof;
        float4 q3;

        Axes cAxes;
        dof = ToAxes(cAxes, q2);
        CHECK_CLOSE(3.121f, (float)dof.x, 0.01);
        CHECK_CLOSE(0.792092f, (float)dof.y, 0.01);
        CHECK_CLOSE(-0.0492416f, (float)dof.z, 0.01);

        q3 = FromAxes(cAxes, dof);
        CHECK_CLOSE(0.922363f, (float)q3.x, 0.01);
        CHECK_CLOSE(-0.0187406f, (float)q3.y, 0.01);
        CHECK_CLOSE(-0.38587f, (float)q3.z, 0.01);
        CHECK_CLOSE(0.0f, (float)q3.w, 0.001);

        Axes aAxiz(float4(0.f, -0.268f, -0.364f, 1.f), float4(0.f, -2.f, 1.f, 0.f), float3(-1.f, -1.f, 1.f), 17.f, math::kZYRoll);
        dof = ToAxes(aAxiz, q2);
        CHECK_CLOSE(1.28582f, (float)dof.x, 0.01);
        CHECK_CLOSE(1.69701f, (float)dof.y, 0.01);
        CHECK_CLOSE(-0.652772f, (float)dof.z, 0.02);

        q3 = FromAxes(aAxiz, dof);
        CHECK_CLOSE(0.922363f, (float)q3.x, 0.01);
        CHECK_CLOSE(-0.0187405f, (float)q3.y, 0.01);
        CHECK_CLOSE(-0.38587f, (float)q3.z, 0.01);
        CHECK_CLOSE(-0.0f, (float)q3.w, 0.01);
    }

    TEST(quatMul_MultiplyNormalizedQuaternionResultIsNormalized)
    {
        float4 a = eulerToQuat(radians(float3(50, -40, 30)));
        float4 b = eulerToQuat(radians(float3(-30, 40, 50)));

        CHECK_CLOSE(1.f, length(a), epsilon);
        CHECK_CLOSE(1.f, length(b), epsilon);

        float4 res = quatMul(a, b);
        CHECK_CLOSE(1.f, length(res), epsilon);
    }

    TEST(quatMul_MultiplyNonNormalizedQuaternionIsValid)
    {
        float4 a = eulerToQuat(radians(float3(50, -40, 30))) * 3.0F;
        float4 b = eulerToQuat(radians(float3(-30, 40, 50))) * -5.0F;

        float4 resNormalized = normalize(quatMul(a, b));
        float4 resNormalizedInputs = quatMul(normalize(a), normalize(b));

        CHECK_CLOSE((float)resNormalized.x, (float)resNormalizedInputs.x, epsilon);
        CHECK_CLOSE((float)resNormalized.y, (float)resNormalizedInputs.y, epsilon);
        CHECK_CLOSE((float)resNormalized.z, (float)resNormalizedInputs.z, epsilon);
        CHECK_CLOSE((float)resNormalized.w, (float)resNormalizedInputs.w, epsilon);
    }

    TEST(normalize_EdgeCaseWork)
    {
        float4 q;

        #if 0
        // No reason for us to enforce this, but just for reference why we need quatNormalize.
        q = normalize(float4(ZERO));
        CHECK_CLOSE(nan, (float)q.x, epsilon);
        CHECK_CLOSE(nan, (float)q.y, epsilon);
        CHECK_CLOSE(nan, (float)q.z, epsilon);
        CHECK_CLOSE(nan, (float)q.w, epsilon);
        #endif

        // zero generates identity
        q = quatNormalize(float4(ZERO));
        CHECK_CLOSE(0, (float)q.x, epsilon);
        CHECK_CLOSE(0, (float)q.y, epsilon);
        CHECK_CLOSE(0, (float)q.z, epsilon);
        CHECK_CLOSE(1, (float)q.w, epsilon);

        q = quatNormalize(float4(epsilon_normal() * 0.5F));
        CHECK_CLOSE(0, (float)q.x, epsilon);
        CHECK_CLOSE(0, (float)q.y, epsilon);
        CHECK_CLOSE(0, (float)q.z, epsilon);
        CHECK_CLOSE(1, (float)q.w, epsilon);

        // simple normalize
        q = quatNormalize(float4(1, 0, 0, 1));
        CHECK_CLOSE(0.707107, (float)q.x, epsilon);
        CHECK_CLOSE(0, (float)q.y, epsilon);
        CHECK_CLOSE(0, (float)q.z, epsilon);
        CHECK_CLOSE(0.707107, (float)q.w, epsilon);

        q = quatNormalize(float4(1, 0, 0, 0));
        CHECK_CLOSE(1, (float)q.x, epsilon);
        CHECK_CLOSE(0, (float)q.y, epsilon);
        CHECK_CLOSE(0, (float)q.z, epsilon);
        CHECK_CLOSE(0, (float)q.w, epsilon);
    }

    TEST(quatProjOnYPlane_DoesNotReturnNan)
    {
        float4 q = float4(1.0f, 0.0f, 0.0f, 0.0f);
        float4 qProj = quatProjOnYPlane(q);

        CHECK(math::all(math::isfinite(qProj) != math::int4(0, 0, 0, 0)));

        q = float4(0.0f, 1.0f, 0.0f, 0.0f);
        qProj = quatProjOnYPlane(q);

        CHECK(math::all(math::isfinite(qProj) != math::int4(0, 0, 0, 0)));

        q = float4(0.0f, 0.0f, 1.0f, 0.0f);
        qProj = quatProjOnYPlane(q);

        CHECK(math::all(math::isfinite(qProj) != math::int4(0, 0, 0, 0)));

        q = float4(0.0f, 0.0f, 0.0f, 1.0f);
        qProj = quatProjOnYPlane(q);

        CHECK(math::all(math::isfinite(qProj) != math::int4(0, 0, 0, 0)));
    }

    PARAMETRIC_TEST_SOURCE(QuatClampSmoothDomainAndResult, (float4, float, float4))
    {
        PARAMETRIC_TEST_CASE(float4(0.0, 0.0, 0.0, 1.0), 2.0 * math::pi(), float4(0.0, 0.0, 0.0, 1.0));
        PARAMETRIC_TEST_CASE(float4(0.0, 0.0, 1.0, 0.0), 2.0 * math::pi(), float4(0.0, 0.0, 1.0, 0.0));
        PARAMETRIC_TEST_CASE(float4(0.0, 0.0, 1.0, 0.0), 1.0 * math::pi(), float4(0.0, 0.0, 1.0, 0.0));
        PARAMETRIC_TEST_CASE(float4(0.0, 0.0, 1.0, 0.0), 0.5 * math::pi(), float4(0.0, 0.0, 0.0, 1.0));
        PARAMETRIC_TEST_CASE(float4(0.0, 0.0, 1.0, 0.0), 0.0 * math::pi(), float4(0.0, 0.0, 0.0, 1.0));
        PARAMETRIC_TEST_CASE(float4(0.0, 0.0, 0.707107f, 0.707107f), 0.5 * math::pi(), float4(0.0, 0.0, 0.707107f, 0.707107f));
        PARAMETRIC_TEST_CASE(float4(0.0, 0.0, 0.707107f, 0.707107f), 0.0 * math::pi(), float4(0.0, 0.0, 0.0, 1.0));
    }

    PARAMETRIC_TEST(quatClampSmooth_DomainIsValid, (float4 input, float clamp, float4 expected), QuatClampSmoothDomainAndResult)
    {
        auto result = quatClampSmooth(input, clamp);
        CHECK_CLOSE(expected.x, result.x, epsilon);
        CHECK_CLOSE(expected.y, result.y, epsilon);
        CHECK_CLOSE(expected.z, result.z, epsilon);
        CHECK_CLOSE(expected.w, result.w, epsilon);
    }

    class quatArcRotateFixture
    {
    public:
        quatArcRotateFixture()
        {
            xAxis = float3(1, 0, 0);
            yAxis = float3(0, 1, 0);
            zAxis = float3(0, 0, 1);
        }

    protected:
        float3 xAxis;
        float3 yAxis;
        float3 zAxis;
    };

    TEST_FIXTURE(quatArcRotateFixture, quatArcRotate_FromNormalizedXAxisToNormalizedYAxis_Returns90OnZAxis)
    {
        float4 q = quatArcRotate(xAxis, yAxis);

        CHECK_CLOSE(0, (float)q.x, epsilon);
        CHECK_CLOSE(0, (float)q.y, epsilon);
        CHECK_CLOSE(0.707107f, (float)q.z, epsilon);
        CHECK_CLOSE(0.707107f, (float)q.w, epsilon);
    }

    TEST_FIXTURE(quatArcRotateFixture, quatArcRotate_FromUnNormalizedXAxisToUnNormalizedYAxis_Returns90OnZAxis)
    {
        float4 q = quatArcRotate(xAxis * float1(2.54f), yAxis * float1(3.98f));

        CHECK_CLOSE(0, (float)q.x, epsilon);
        CHECK_CLOSE(0, (float)q.y, epsilon);
        CHECK_CLOSE(0.707107f, (float)q.z, epsilon);
        CHECK_CLOSE(0.707107f, (float)q.w, epsilon);
    }

    TEST_FIXTURE(quatArcRotateFixture, quatArcRotate_FromNormalizedYAxisToNormalizedZAxis_Returns90OnXAxis)
    {
        float4 q = quatArcRotate(yAxis, zAxis);

        CHECK_CLOSE(0.707107f, (float)q.x, epsilon);
        CHECK_CLOSE(0, (float)q.y, epsilon);
        CHECK_CLOSE(0, (float)q.z, epsilon);
        CHECK_CLOSE(0.707107f, (float)q.w, epsilon);
    }

    TEST_FIXTURE(quatArcRotateFixture, quatArcRotate_FromNormalizedXAxisToNormalizedXAxis_ReturnsIdentity)
    {
        float4 q = quatArcRotate(xAxis, xAxis);

        CHECK_CLOSE(0, (float)q.x, epsilon);
        CHECK_CLOSE(0, (float)q.y, epsilon);
        CHECK_CLOSE(0, (float)q.z, epsilon);
        CHECK_CLOSE(1, (float)q.w, epsilon);
    }

    TEST_FIXTURE(quatArcRotateFixture, quatArcRotate_FromUnNormalizedXAxisToUnNormalizedXAxis_ReturnsIdentity)
    {
        float4 q = quatArcRotate(xAxis * float1(2.54f), xAxis * float1(1.47f));

        CHECK_CLOSE(0, (float)q.x, epsilon);
        CHECK_CLOSE(0, (float)q.y, epsilon);
        CHECK_CLOSE(0, (float)q.z, epsilon);
        CHECK_CLOSE(1, (float)q.w, epsilon);
    }

    TEST_FIXTURE(quatArcRotateFixture, quatArcRotate_FromXAxisToMinusXAxis_ReturnsNAN)
    {
        float4 q = quatArcRotate(xAxis, -xAxis);
        CHECK(math::all(isfinite(q) == int4(ZERO)) == 1);
    }

    PARAMETRIC_TEST_SOURCE(InputAndExpectedEulerAngles, (Vector3f, Vector3f))
    {
        PARAMETRIC_TEST_CASE(Vector3f(-360.0f, 0.0f, 0.0f), Vector3f(-360.0f, 0.0f, 0.0f));
        PARAMETRIC_TEST_CASE(Vector3f(-180.0f, 0.0f, 0.0f), Vector3f(-180.0f, 0.0f, 0.0f));
        PARAMETRIC_TEST_CASE(Vector3f(-90.0f, 0.0f, 0.0f), Vector3f(-90.0f, 0.0f, 0.0f));
        PARAMETRIC_TEST_CASE(Vector3f(-45.0f, 0.0f, 0.0f), Vector3f(-45.0f, 0.0f, 0.0f));
        PARAMETRIC_TEST_CASE(Vector3f(0.0f, 0.0f, 0.0f), Vector3f(0.0f, 0.0f, 0.0f));
        PARAMETRIC_TEST_CASE(Vector3f(45.0f, 0.0f, 0.0f), Vector3f(45.0f, 0.0f, 0.0f));
        PARAMETRIC_TEST_CASE(Vector3f(90.0f, 0.0f, 0.0f), Vector3f(90.0f, 0.0f, 0.0f));
        PARAMETRIC_TEST_CASE(Vector3f(180.0f, 0.0f, 0.0f), Vector3f(180.0f, 0.0f, 0.0f));
        PARAMETRIC_TEST_CASE(Vector3f(360.0f, 0.0f, 0.0f), Vector3f(360.0f, 0.0f, 0.0f));

        PARAMETRIC_TEST_CASE(Vector3f(180.0f, 180.0f, 180.0f), Vector3f(180.0f, 180.0f, 180.0f));

        PARAMETRIC_TEST_CASE(Vector3f(-10.0f, -10.0f, -10.0f), Vector3f(-10.0f, -10.0f, -10.0f));
        PARAMETRIC_TEST_CASE(Vector3f(20.0f, 20.0f, 20.0f), Vector3f(20.0f, 20.0f, 20.0f));
        PARAMETRIC_TEST_CASE(Vector3f(-30.0f, -30.0f, -30.0f), Vector3f(-30.0f, -30.0f, -30.0f));
        PARAMETRIC_TEST_CASE(Vector3f(40.0f, -40.0f, 40.0f), Vector3f(40.0f, -40.0f, 40.0f));
        PARAMETRIC_TEST_CASE(Vector3f(-50.0f, 50.0f, -50.0f), Vector3f(-50.0f, 50.0f, -50.0f));
        PARAMETRIC_TEST_CASE(Vector3f(60.0f, -60.0f, 60.0f), Vector3f(60.0f, -60.0f, 60.0f));
        PARAMETRIC_TEST_CASE(Vector3f(-70.0f, 70.0f, 70.0f), Vector3f(-70.0f, 70.0f, 70.0f));
        PARAMETRIC_TEST_CASE(Vector3f(80.0f, 80.0f, -80.0f), Vector3f(80.0f, 80.0f, -80.0f));

        // Randomly generated +/-360
        PARAMETRIC_TEST_CASE(Vector3f(1.111000f, 83.281556f, 18.751016f), Vector3f(1.111000f, 83.282005f, 18.751016f));
        PARAMETRIC_TEST_CASE(Vector3f(53.4085008f, -79.1774845f, -24.648198f), Vector3f(53.409004f, -79.177002f, -24.648001f));
        PARAMETRIC_TEST_CASE(Vector3f(146.9411086f, 49.1950222f, 286.2303866f), Vector3f(146.940994f, 49.194992f, 286.230011f));
        PARAMETRIC_TEST_CASE(Vector3f(-157.4968920f, -256.0906005f, -63.4642492f), Vector3f(-157.496994f, -256.091003f, -63.463989f));
        PARAMETRIC_TEST_CASE(Vector3f(-88.6572305f, -55.4661698f, 321.0858388f), Vector3f(-88.657005f, -55.466003f, 321.085999f));
        PARAMETRIC_TEST_CASE(Vector3f(275.7282147f, 29.0170594f, 51.7573070f), Vector3f(275.727997f, 29.017002f, 51.757004f));
        PARAMETRIC_TEST_CASE(Vector3f(107.3065947f, 126.3650226f, -56.9548132f), Vector3f(107.307007f, 126.364998f, -56.954987f));
        PARAMETRIC_TEST_CASE(Vector3f(102.9281660f, 27.1524632f, -10.9190489f), Vector3f(102.927979f, 27.151993f, -10.919006f));
        PARAMETRIC_TEST_CASE(Vector3f(250.5575753f, 6.6964804f, 0.7827676f), Vector3f(250.558014f, 6.695999f, 0.782990f));
        PARAMETRIC_TEST_CASE(Vector3f(2.5354611f, 180.3002958f, 75.7143737f), Vector3f(2.535000f, 180.299988f, 75.714005f));
    }

    PARAMETRIC_TEST(closestEuler_WithInputEulerAngle_ReturnsExpectedEulerAngle, (Vector3f inputEuler, Vector3f expectedEuler), InputAndExpectedEulerAngles)
    {
        float4 quatVal = eulerToQuat(radians(Vector3fTofloat3(inputEuler)), kOrderUnityDefault);
        // Round to make eulerHint more 'hint' like, otherwise this test will always pass
        float3 outputEuler = closestEuler(quatVal, round(Vector3fTofloat3(inputEuler)), kOrderUnityDefault);
        CHECK_CLOSE(expectedEuler.x, outputEuler.x, epsilon);
        CHECK_CLOSE(expectedEuler.y, outputEuler.y, epsilon);
        CHECK_CLOSE(expectedEuler.z, outputEuler.z, epsilon);
    }
}

#endif

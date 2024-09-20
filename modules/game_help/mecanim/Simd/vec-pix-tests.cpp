#include "UnityPrefix.h"
#if ENABLE_UNIT_TESTS
#include "Runtime/Testing/Testing.h"
#include "Runtime/Profiler/TimeHelper.h"
#include "Runtime/Math/Color.h"

#include "vec-types.h"
#include "vec-pix.h"
#include "vec-quat.h"

UNIT_TEST_SUITE(SIMDMath_pixOps)
{
    using namespace math;

    #define MAKE_PIX(a, b, c, d) (d<<24)|(c<<16)|(b<<8)|(a)

    alignas(4) UInt8 bytes0[4][4] = { { 255, 200, 140, 60 }, {   0,   0,   0,   0 }, { 0, 50, 100, 150 }, { 11, 22, 33, 44 } };
    alignas(4) UInt8 bytes1[4][4] = { { 100,  50, 221, 50 }, { 255, 255, 255, 255 }, { 0, 50, 100, 150 }, { 55, 66, 77, 88 } };
    alignas(4) UInt8 bytes2[4][4] = { { 10, 11, 12, 13 }, { 14, 15, 16, 17 }, { 18, 19, 20, 21 }, { 22, 23, 24, 25 } };

    TEST(pix4_BytesOrder_Matches_ColorRGBA32)
    {
        pix4 pix0 = vload16uc(&bytes0[0][0]);
        pix4 pix1 = vload16uc(&bytes1[0][0]);

        // check int4 byte order matches ColorRGBA32
        ColorRGBA32 cols0[4] = { ColorRGBA32(255, 200, 140, 60), ColorRGBA32(0,   0,   0,   0), ColorRGBA32(0, 50, 100, 150), ColorRGBA32(11, 22, 33, 44) };
        ColorRGBA32 cols1[4] = { ColorRGBA32(100,  50, 221, 50), ColorRGBA32(255, 255, 255, 255), ColorRGBA32(0, 50, 100, 150), ColorRGBA32(55, 66, 77, 88) };
        CHECK(all(pix0 == pix4(vload16uc(cols0[0].GetPtr()))));
        CHECK(all(pix1 == pix4(vload16uc(cols1[0].GetPtr()))));
    }

    TEST(convert_pix1_Works)
    {
        pix4 pix0 = convert_pix1(int4(0, 127, 128, 255));
        uint32_t pixValue = MAKE_PIX(0, 127, 128, 255);
        CHECK(all(pix0 == pix4(pixValue, pixValue, pixValue, pixValue)));
    }

    TEST(pix4_BytesOrder_Matches_MAKE_PIX)
    {
        pix4 pix0 = vload16uc(&bytes0[0][0]);
        pix4 pix1 = vload16uc(&bytes1[0][0]);

        // check byte order matches our MAKE_PIX macro too
        CHECK(all(pix0 == pix4(MAKE_PIX(255, 200, 140, 60), MAKE_PIX(0,   0,   0,   0), MAKE_PIX(0, 50, 100, 150), MAKE_PIX(11, 22, 33, 44))));
        CHECK(all(pix1 == pix4(MAKE_PIX(100,  50, 221, 50), MAKE_PIX(255, 255, 255, 255), MAKE_PIX(0, 50, 100, 150), MAKE_PIX(55, 66, 77, 88))));
    }

    TEST(pix_permute4_Works)
    {
        pix4 pix0 = vload16uc(&bytes2[0][0]);

        // identity shuffle
        pix4 mask0 = pix4(MAKE_PIX(0, 1, 2, 3), MAKE_PIX(4, 5, 6, 7), MAKE_PIX(8, 9, 10, 11), MAKE_PIX(12, 13, 14, 15));
        CHECK(all(pix_permute4(pix0, mask0) == pix4(MAKE_PIX(10, 11, 12, 13), MAKE_PIX(14, 15, 16, 17), MAKE_PIX(18, 19, 20, 21), MAKE_PIX(22, 23, 24, 25))));

        // int granularity shuffle
        pix4 mask1 = pix4(MAKE_PIX(12, 13, 14, 15), MAKE_PIX(4, 5, 6, 7), MAKE_PIX(8, 9, 10, 11), MAKE_PIX(0, 1, 2, 3));
        CHECK(all(pix_permute4(pix0, mask1) == pix4(MAKE_PIX(22, 23, 24, 25), MAKE_PIX(14, 15, 16, 17), MAKE_PIX(18, 19, 20, 21), MAKE_PIX(10, 11, 12, 13))));

        // any random shuffle
        pix4 mask2 = pix4(MAKE_PIX(13, 10, 12, 11), MAKE_PIX(9, 14, 0, 11), MAKE_PIX(2, 9, 3, 3), MAKE_PIX(3, 1, 5, 6));
        CHECK(all(pix_permute4(pix0, mask2) == pix4(MAKE_PIX(23, 20, 22, 21), MAKE_PIX(19, 24, 10, 21), MAKE_PIX(12, 19, 13, 13), MAKE_PIX(13, 11, 15, 16))));
    }

    TEST(add_Works)
    {
        pix4 pix0 = vload16uc(&bytes0[0][0]);
        pix4 pix1 = vload16uc(&bytes1[0][0]);

        pix4 a = pix0 + pix1;
        CHECK(all(a == pix4(MAKE_PIX(255, 250, 255, 110), MAKE_PIX(255, 255, 255, 255), MAKE_PIX(0, 100, 200, 255), MAKE_PIX(66, 88, 110, 132))));
    }

    TEST(copy_alpha_Works)
    {
        pix4 pix0 = vload16uc(&bytes0[0][0]);
        pix4 pix1 = vload16uc(&bytes1[0][0]);

        pix4 a = copy_alpha(pix0, pix1);
        CHECK(all(a == pix4(MAKE_PIX(255, 200, 140, 50), MAKE_PIX(0,   0,   0,   255), MAKE_PIX(0, 50, 100, 150), MAKE_PIX(11, 22, 33, 88))));
    }

    TEST(lerp_Works)
    {
        pix4 pix0 = vload16uc(&bytes0[0][0]);
        pix4 pix1 = vload16uc(&bytes1[0][0]);

        int4 weight4 = pix_weight(float4(0.33f, 1.0f, 0.4f, 0.0f));
        pix4 a = lerp(pix0, pix1, weight4);
        CHECK(all(a == pix4(MAKE_PIX(204, 151, 167, 57), MAKE_PIX(254, 254, 254, 254), MAKE_PIX(0, 50, 100, 150), MAKE_PIX(11, 22, 33, 44))));
    }

    TEST(multiply_Works)
    {
        pix4 pix0 = vload16uc(&bytes0[0][0]);
        pix4 pix1 = vload16uc(&bytes1[0][0]);

        pix4 a = pix0 * pix1;
        CHECK(all(a == pix4(MAKE_PIX(100, 39, 121, 12), MAKE_PIX(0, 0, 0, 0), MAKE_PIX(0, 10, 39, 88), MAKE_PIX(2, 6, 10, 15))));
    }
    #undef MAKE_PIX
}

#endif

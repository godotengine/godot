#include <x86intrin.h>
#include <stdio.h>

int main()
{
    __m128i a;
    __m128i b;

    a[1] = 2;
    a[0] = -1284;
    b[1] = 25;
    b[0] = 65535;

    // _mm_clmulepi64_si128 only looks at the least significant bit of each
    __m128i result1 = _mm_clmulepi64_si128( a, b, 0x11 );
    __m128i result2 = _mm_clmulepi64_si128( a, b, 0x00 );
    __m128i result3 = _mm_clmulepi64_si128( a, b, 0xF2 );

    printf("%lld times %lld without a carry bit: %lld\n",
        a[1], b[1], result1[0]);
    printf("%lld times %lld without a carry bit: %lld\n",
        a[0], b[0], result2[0]);
    printf("%lld times %lld without a carry bit: %lld\n",
        a[0], b[1], result3[0]);
    return 0;
}

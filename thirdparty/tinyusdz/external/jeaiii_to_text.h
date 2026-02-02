/*
MIT License

Copyright (c) 2022 James Edward Anhalt III - https://github.com/jeaiii/itoa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef JEAIII_TO_TEXT_H_
#define JEAIII_TO_TEXT_H_

namespace jeaiii
{
    using u32 = decltype(0xffffffff);
    using u64 = decltype(0xffffffffffffffff);

    struct pair
    {
        char dd[2];
        constexpr pair(char c) : dd{ c, '\0' } { }
        constexpr pair(int n) : dd{ "0123456789"[n / 10], "0123456789"[n % 10] } { }
    };

    constexpr struct
    {
        pair dd[100]
        {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        };
        pair fd[100]
        {
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        };
    }
    digits;

    constexpr u64 mask24 = (u64(1) << 24) - 1;
    constexpr u64 mask32 = (u64(1) << 32) - 1;
    constexpr u64 mask57 = (u64(1) << 57) - 1;

    template<bool, class, class F> struct _cond { using type = F; };
    template<class T, class F> struct _cond<true, T, F> { using type = T; };
    template<bool B, class T, class F> using cond = typename _cond<B, T, F>::type;

    template<class T>
#if defined(_MSC_VER)
    __forceinline
#else
    inline
#endif
    char* to_text_from_integer(char* b, T i)
    {
        constexpr auto q = sizeof(T);
        using U = cond<q == 1, unsigned char, cond<q <= sizeof(short), unsigned short, cond<q <= sizeof(u32), u32, u64>>>;

        // convert bool to int before test with unary + to silence warning if T happens to be bool
        U const n = +i < 0 ? *b++ = '-', U(0) - U(i) : U(i);

        if (n < u32(1e2))
        {
            *reinterpret_cast<pair*>(b) = digits.fd[n];
            return n < 10 ? b + 1 : b + 2;
        }
        if (n < u32(1e6))
        {
            if (n < u32(1e4))
            {
                auto f0 = u32(10 * (1 << 24) / 1e3 + 1) * n;
                *reinterpret_cast<pair*>(b) = digits.fd[f0 >> 24];
                b -= n < u32(1e3);
                auto f2 = (f0 & mask24) * 100;
                *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 24];
                return b + 4;
            }
            auto f0 = u64(10 * (1ull << 32ull)/ 1e5 + 1) * n;
            *reinterpret_cast<pair*>(b) = digits.fd[f0 >> 32];
            b -= n < u32(1e5);
            auto f2 = (f0 & mask32) * 100;
            *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 32];
            auto f4 = (f2 & mask32) * 100;
            *reinterpret_cast<pair*>(b + 4) = digits.dd[f4 >> 32];
            return b + 6;
        }
        if (n < u64(1ull << 32ull))
        {
            if (n < u32(1e8))
            {
                auto f0 = u64(10 * (1ull << 48ull) / 1e7 + 1) * n >> 16;
                *reinterpret_cast<pair*>(b) = digits.fd[f0 >> 32];
                b -= n < u32(1e7);
                auto f2 = (f0 & mask32) * 100;
                *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 32];
                auto f4 = (f2 & mask32) * 100;
                *reinterpret_cast<pair*>(b + 4) = digits.dd[f4 >> 32];
                auto f6 = (f4 & mask32) * 100;
                *reinterpret_cast<pair*>(b + 6) = digits.dd[f6 >> 32];
                return b + 8;
            }
            auto f0 = u64(10 * (1ull << 57ull) / 1e9 + 1) * n;
            *reinterpret_cast<pair*>(b) = digits.fd[f0 >> 57];
            b -= n < u32(1e9);
            auto f2 = (f0 & mask57) * 100;
            *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 57];
            auto f4 = (f2 & mask57) * 100;
            *reinterpret_cast<pair*>(b + 4) = digits.dd[f4 >> 57];
            auto f6 = (f4 & mask57) * 100;
            *reinterpret_cast<pair*>(b + 6) = digits.dd[f6 >> 57];
            auto f8 = (f6 & mask57) * 100;
            *reinterpret_cast<pair*>(b + 8) = digits.dd[f8 >> 57];
            return b + 10;
        }

        // if we get here U must be u64 but some compilers don't know that, so reassign n to a u64 to avoid warnings
        u32 z = n % u32(1e8);
        u64 u = n / u32(1e8);

        if (u < u32(1e2))
        {
            // u can't be 1 digit (if u < 10 it would have been handled above as a 9 digit 32bit number)
            *reinterpret_cast<pair*>(b) = digits.dd[u];
            b += 2;
        }
        else if (u < u32(1e6))
        {
            if (u < u32(1e4))
            {
                auto f0 = u32(10 * (1 << 24) / 1e3 + 1) * u;
                *reinterpret_cast<pair*>(b) = digits.fd[f0 >> 24];
                b -= u < u32(1e3);
                auto f2 = (f0 & mask24) * 100;
                *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 24];
                b += 4;
            }
            else
            {
                auto f0 = u64(10 * (1ull << 32ull) / 1e5 + 1) * u;
                *reinterpret_cast<pair*>(b) = digits.fd[f0 >> 32];
                b -= u < u32(1e5);
                auto f2 = (f0 & mask32) * 100;
                *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 32];
                auto f4 = (f2 & mask32) * 100;
                *reinterpret_cast<pair*>(b + 4) = digits.dd[f4 >> 32];
                b += 6;
            }
        }
        else if (u < u32(1e8))
        {
            auto f0 = u64(10 * (1ull << 48ull) / 1e7 + 1) * u >> 16;
            *reinterpret_cast<pair*>(b) = digits.fd[f0 >> 32];
            b -= u < u32(1e7);
            auto f2 = (f0 & mask32) * 100;
            *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 32];
            auto f4 = (f2 & mask32) * 100;
            *reinterpret_cast<pair*>(b + 4) = digits.dd[f4 >> 32];
            auto f6 = (f4 & mask32) * 100;
            *reinterpret_cast<pair*>(b + 6) = digits.dd[f6 >> 32];
            b += 8;
        }
        else if (u < u64(1ull << 32ull))
        {
            auto f0 = u64(10 * (1ull << 57ull) / 1e9 + 1) * u;
            *reinterpret_cast<pair*>(b) = digits.fd[f0 >> 57];
            b -= u < u32(1e9);
            auto f2 = (f0 & mask57) * 100;
            *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 57];
            auto f4 = (f2 & mask57) * 100;
            *reinterpret_cast<pair*>(b + 4) = digits.dd[f4 >> 57];
            auto f6 = (f4 & mask57) * 100;
            *reinterpret_cast<pair*>(b + 6) = digits.dd[f6 >> 57];
            auto f8 = (f6 & mask57) * 100;
            *reinterpret_cast<pair*>(b + 8) = digits.dd[f8 >> 57];
            b += 10;
        }
        else
        {
            u32 y = u % u32(1e8);
            u /= u32(1e8);

            // u is 2, 3, or 4 digits (if u < 10 it would have been handled above)
            if (u < u32(1e2))
            {
                *reinterpret_cast<pair*>(b) = digits.dd[u];
                b += 2;
            }
            else
            {
                auto f0 = u32(10 * (1 << 24) / 1e3 + 1) * u;
                *reinterpret_cast<pair*>(b) = digits.fd[f0 >> 24];
                b -= u < u32(1e3);
                auto f2 = (f0 & mask24) * 100;
                *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 24];
                b += 4;
            }
            // do 8 digits
            auto f0 = (u64((1ull << 48ull) / 1e6 + 1) * y >> 16) + 1;
            *reinterpret_cast<pair*>(b) = digits.dd[f0 >> 32];
            auto f2 = (f0 & mask32) * 100;
            *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 32];
            auto f4 = (f2 & mask32) * 100;
            *reinterpret_cast<pair*>(b + 4) = digits.dd[f4 >> 32];
            auto f6 = (f4 & mask32) * 100;
            *reinterpret_cast<pair*>(b + 6) = digits.dd[f6 >> 32];
            b += 8;
        }
        // do 8 digits
        auto f0 = (u64((1ull << 48ull) / 1e6 + 1) * z >> 16) + 1;
        *reinterpret_cast<pair*>(b) = digits.dd[f0 >> 32];
        auto f2 = (f0 & mask32) * 100;
        *reinterpret_cast<pair*>(b + 2) = digits.dd[f2 >> 32];
        auto f4 = (f2 & mask32) * 100;
        *reinterpret_cast<pair*>(b + 4) = digits.dd[f4 >> 32];
        auto f6 = (f4 & mask32) * 100;
        *reinterpret_cast<pair*>(b + 6) = digits.dd[f6 >> 32];
        return b + 8;
    }
}
#endif // JEAIII_TO_TEXT_H_

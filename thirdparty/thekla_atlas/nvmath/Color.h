// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MATH_COLOR_H
#define NV_MATH_COLOR_H

#include "nvmath.h"

namespace nv
{

    /// 64 bit color stored as BGRA.
    class NVMATH_CLASS Color64 
    {
    public:
        Color64() { }
        Color64(const Color64 & c) : u(c.u) { }
        Color64(uint16 R, uint16 G, uint16 B, uint16 A) { setRGBA(R, G, B, A); }
        explicit Color64(uint64 U) : u(U) { }

        void setRGBA(uint16 R, uint16 G, uint16 B, uint16 A)
        {
            r = R;
            g = G;
            b = B;
            a = A;
        }

        operator uint64 () const {
            return u;
        }

        union {
            struct {
#if NV_LITTLE_ENDIAN
                uint16 r, a, b, g;
#else
                uint16 a: 16;
                uint16 r: 16;
                uint16 g: 16;
                uint16 b: 16;
#endif
            };
            uint64 u;
        };
    };

    /// 32 bit color stored as BGRA.
    class NVMATH_CLASS Color32
    {
    public:
        Color32() { }
        Color32(const Color32 & c) : u(c.u) { }
        Color32(uint8 R, uint8 G, uint8 B) { setRGBA(R, G, B, 0xFF); }
        Color32(uint8 R, uint8 G, uint8 B, uint8 A) { setRGBA( R, G, B, A); }
        //Color32(uint8 c[4]) { setRGBA(c[0], c[1], c[2], c[3]); }
        //Color32(float R, float G, float B) { setRGBA(uint(R*255), uint(G*255), uint(B*255), 0xFF); }
        //Color32(float R, float G, float B, float A) { setRGBA(uint(R*255), uint(G*255), uint(B*255), uint(A*255)); }
        explicit Color32(uint32 U) : u(U) { }

        void setRGBA(uint8 R, uint8 G, uint8 B, uint8 A)
        {
            r = R;
            g = G;
            b = B;
            a = A;
        }

        void setBGRA(uint8 B, uint8 G, uint8 R, uint8 A = 0xFF)
        {
            r = R;
            g = G;
            b = B;
            a = A;
        }

        operator uint32 () const {
            return u;
        }

        union {
            struct {
#if NV_LITTLE_ENDIAN
                uint8 b, g, r, a;
#else
                uint8 a: 8;
                uint8 r: 8;
                uint8 g: 8;
                uint8 b: 8;
#endif
            };
            uint8 component[4];
            uint32 u;
        };
    };


    /// 16 bit 565 BGR color.
    class NVMATH_CLASS Color16
    {
    public:
        Color16() { }
        Color16(const Color16 & c) : u(c.u) { }
        explicit Color16(uint16 U) : u(U) { }

        union {
            struct {
#if NV_LITTLE_ENDIAN
                uint16 b : 5;
                uint16 g : 6;
                uint16 r : 5;
#else
                uint16 r : 5;
                uint16 g : 6;
                uint16 b : 5;
#endif
            };
            uint16 u;
        };
    };

    /// 16 bit 4444 BGRA color.
    class NVMATH_CLASS Color16_4444
    {
    public:
        Color16_4444() { }
        Color16_4444(const Color16_4444 & c) : u(c.u) { }
        explicit Color16_4444(uint16 U) : u(U) { }

        union {
            struct {
#if NV_LITTLE_ENDIAN
                uint16 b : 4;
                uint16 g : 4;
                uint16 r : 4;
                uint16 a : 4;
#else
                uint16 a : 4;
                uint16 r : 4;
                uint16 g : 4;
                uint16 b : 4;
#endif
            };
            uint16 u;
        };
    };

} // nv namespace

#endif // NV_MATH_COLOR_H

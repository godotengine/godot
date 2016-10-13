/* -----------------------------------------------------------------------------

    Copyright (c) 2006 Simon Brown                          si@sjbrown.co.uk

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

   -------------------------------------------------------------------------- */

#ifndef SQUISH_SIMD_VE_H
#define SQUISH_SIMD_VE_H

#include <altivec.h>
#undef bool

namespace squish {

#define VEC4_CONST( X ) Vec4( ( vector float ){ X } )

class Vec4
{
public:
    typedef Vec4 Arg;

    Vec4() {}

    explicit Vec4( vector float v ) : m_v( v ) {}

    Vec4( Vec4 const& arg ) : m_v( arg.m_v ) {}

    Vec4& operator=( Vec4 const& arg )
    {
        m_v = arg.m_v;
        return *this;
    }

    explicit Vec4( float s )
    {
        union { vector float v; float c[4]; } u;
        u.c[0] = s;
        u.c[1] = s;
        u.c[2] = s;
        u.c[3] = s;
        m_v = u.v;
    }

    Vec4( float x, float y, float z, float w )
    {
        union { vector float v; float c[4]; } u;
        u.c[0] = x;
        u.c[1] = y;
        u.c[2] = z;
        u.c[3] = w;
        m_v = u.v;
    }

    Vec3 GetVec3() const
    {
        union { vector float v; float c[4]; } u;
        u.v = m_v;
        return Vec3( u.c[0], u.c[1], u.c[2] );
    }

    Vec4 SplatX() const { return Vec4( vec_splat( m_v, 0 ) ); }
    Vec4 SplatY() const { return Vec4( vec_splat( m_v, 1 ) ); }
    Vec4 SplatZ() const { return Vec4( vec_splat( m_v, 2 ) ); }
    Vec4 SplatW() const { return Vec4( vec_splat( m_v, 3 ) ); }

    Vec4& operator+=( Arg v )
    {
        m_v = vec_add( m_v, v.m_v );
        return *this;
    }

    Vec4& operator-=( Arg v )
    {
        m_v = vec_sub( m_v, v.m_v );
        return *this;
    }

    Vec4& operator*=( Arg v )
    {
        m_v = vec_madd( m_v, v.m_v, ( vector float ){ -0.0f } );
        return *this;
    }

    friend Vec4 operator+( Vec4::Arg left, Vec4::Arg right  )
    {
        return Vec4( vec_add( left.m_v, right.m_v ) );
    }

    friend Vec4 operator-( Vec4::Arg left, Vec4::Arg right  )
    {
        return Vec4( vec_sub( left.m_v, right.m_v ) );
    }

    friend Vec4 operator*( Vec4::Arg left, Vec4::Arg right  )
    {
        return Vec4( vec_madd( left.m_v, right.m_v, ( vector float ){ -0.0f } ) );
    }

    //! Returns a*b + c
    friend Vec4 MultiplyAdd( Vec4::Arg a, Vec4::Arg b, Vec4::Arg c )
    {
        return Vec4( vec_madd( a.m_v, b.m_v, c.m_v ) );
    }

    //! Returns -( a*b - c )
    friend Vec4 NegativeMultiplySubtract( Vec4::Arg a, Vec4::Arg b, Vec4::Arg c )
    {
        return Vec4( vec_nmsub( a.m_v, b.m_v, c.m_v ) );
    }

    friend Vec4 Reciprocal( Vec4::Arg v )
    {
        // get the reciprocal estimate
        vector float estimate = vec_re( v.m_v );

        // one round of Newton-Rhaphson refinement
        vector float diff = vec_nmsub( estimate, v.m_v, ( vector float ){ 1.0f } );
        return Vec4( vec_madd( diff, estimate, estimate ) );
    }

    friend Vec4 Min( Vec4::Arg left, Vec4::Arg right )
    {
        return Vec4( vec_min( left.m_v, right.m_v ) );
    }

    friend Vec4 Max( Vec4::Arg left, Vec4::Arg right )
    {
        return Vec4( vec_max( left.m_v, right.m_v ) );
    }

    friend Vec4 Truncate( Vec4::Arg v )
    {
        return Vec4( vec_trunc( v.m_v ) );
    }

    friend bool CompareAnyLessThan( Vec4::Arg left, Vec4::Arg right )
    {
        return vec_any_lt( left.m_v, right.m_v ) != 0;
    }

private:
    vector float m_v;
};

} // namespace squish

#endif // ndef SQUISH_SIMD_VE_H

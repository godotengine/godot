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

#ifndef SQUISH_MATHS_H
#define SQUISH_MATHS_H

#include <cmath>
#include <algorithm>
#include "config.h"

namespace squish {

class Vec3
{
public:
    typedef Vec3 const& Arg;

    Vec3()
    {
    }

    explicit Vec3( float s )
    {
        m_x = s;
        m_y = s;
        m_z = s;
    }

    Vec3( float x, float y, float z )
    {
        m_x = x;
        m_y = y;
        m_z = z;
    }

    float X() const { return m_x; }
    float Y() const { return m_y; }
    float Z() const { return m_z; }

    Vec3 operator-() const
    {
        return Vec3( -m_x, -m_y, -m_z );
    }

    Vec3& operator+=( Arg v )
    {
        m_x += v.m_x;
        m_y += v.m_y;
        m_z += v.m_z;
        return *this;
    }

    Vec3& operator-=( Arg v )
    {
        m_x -= v.m_x;
        m_y -= v.m_y;
        m_z -= v.m_z;
        return *this;
    }

    Vec3& operator*=( Arg v )
    {
        m_x *= v.m_x;
        m_y *= v.m_y;
        m_z *= v.m_z;
        return *this;
    }

    Vec3& operator*=( float s )
    {
        m_x *= s;
        m_y *= s;
        m_z *= s;
        return *this;
    }

    Vec3& operator/=( Arg v )
    {
        m_x /= v.m_x;
        m_y /= v.m_y;
        m_z /= v.m_z;
        return *this;
    }

    Vec3& operator/=( float s )
    {
        float t = 1.0f/s;
        m_x *= t;
        m_y *= t;
        m_z *= t;
        return *this;
    }

    friend Vec3 operator+( Arg left, Arg right )
    {
        Vec3 copy( left );
        return copy += right;
    }

    friend Vec3 operator-( Arg left, Arg right )
    {
        Vec3 copy( left );
        return copy -= right;
    }

    friend Vec3 operator*( Arg left, Arg right )
    {
        Vec3 copy( left );
        return copy *= right;
    }

    friend Vec3 operator*( Arg left, float right )
    {
        Vec3 copy( left );
        return copy *= right;
    }

    friend Vec3 operator*( float left, Arg right )
    {
        Vec3 copy( right );
        return copy *= left;
    }

    friend Vec3 operator/( Arg left, Arg right )
    {
        Vec3 copy( left );
        return copy /= right;
    }

    friend Vec3 operator/( Arg left, float right )
    {
        Vec3 copy( left );
        return copy /= right;
    }

    friend float Dot( Arg left, Arg right )
    {
        return left.m_x*right.m_x + left.m_y*right.m_y + left.m_z*right.m_z;
    }

    friend Vec3 Min( Arg left, Arg right )
    {
        return Vec3(
            std::min( left.m_x, right.m_x ),
            std::min( left.m_y, right.m_y ),
            std::min( left.m_z, right.m_z )
        );
    }

    friend Vec3 Max( Arg left, Arg right )
    {
        return Vec3(
            std::max( left.m_x, right.m_x ),
            std::max( left.m_y, right.m_y ),
            std::max( left.m_z, right.m_z )
        );
    }

    friend Vec3 Truncate( Arg v )
    {
        return Vec3(
            v.m_x > 0.0f ? std::floor( v.m_x ) : std::ceil( v.m_x ),
            v.m_y > 0.0f ? std::floor( v.m_y ) : std::ceil( v.m_y ),
            v.m_z > 0.0f ? std::floor( v.m_z ) : std::ceil( v.m_z )
        );
    }

private:
    float m_x;
    float m_y;
    float m_z;
};

inline float LengthSquared( Vec3::Arg v )
{
    return Dot( v, v );
}

class Sym3x3
{
public:
    Sym3x3()
    {
    }

    Sym3x3( float s )
    {
        for( int i = 0; i < 6; ++i )
            m_x[i] = s;
    }

    float operator[]( int index ) const
    {
        return m_x[index];
    }

    float& operator[]( int index )
    {
        return m_x[index];
    }

private:
    float m_x[6];
};

Sym3x3 ComputeWeightedCovariance( int n, Vec3 const* points, float const* weights );
Vec3 ComputePrincipleComponent( Sym3x3 const& matrix );

} // namespace squish

#endif // ndef SQUISH_MATHS_H

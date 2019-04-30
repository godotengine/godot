#include <math.h>
#include <stdint.h>

#include "Math.hpp"
#include "ColorSpace.hpp"

namespace Color
{

    static const XYZ white( v3b( 255, 255, 255 ) );
    static const v3f rwhite( 1.f / white.x, 1.f / white.y, 1.f / white.z );


    XYZ::XYZ( float _x, float _y, float _z )
        : x( _x )
        , y( _y )
        , z( _z )
    {
    }

    XYZ::XYZ( const v3b& rgb )
    {
        const float r = rgb.x / 255.f;
        const float g = rgb.y / 255.f;
        const float b = rgb.z / 255.f;

        const float rl = sRGB2linear( r );
        const float gl = sRGB2linear( g );
        const float bl = sRGB2linear( b );

        x = 0.4124f * rl + 0.3576f * gl + 0.1805f * bl;
        y = 0.2126f * rl + 0.7152f * gl + 0.0722f * bl;
        z = 0.0193f * rl + 0.1192f * gl + 0.9505f * bl;
    }

    static float revlab( float t )
    {
        const float p1 = 6.f/29.f;
        const float p2 = 4.f/29.f;

        if( t > p1 )
        {
            return t*t*t;
        }
        else
        {
            return 3 * sq( p1 ) * ( t - p2 );
        }
    }

    XYZ::XYZ( const Lab& lab )
    {
        y = white.y * revlab( 1.f/116.f * ( lab.L + 16 ) );
        x = white.x * revlab( 1.f/116.f * ( lab.L + 16 ) + 1.f/500.f * lab.a );
        z = white.z * revlab( 1.f/116.f * ( lab.L + 16 ) - 1.f/200.f * lab.b );
    }

    v3i XYZ::RGB() const
    {
        const float rl =  3.2406f * x - 1.5372f * y - 0.4986f * z;
        const float gl = -0.9689f * x + 1.8758f * y + 0.0415f * z;
        const float bl =  0.0557f * x - 0.2040f * y + 1.0570f * z;

        const float r = linear2sRGB( rl );
        const float g = linear2sRGB( gl );
        const float b = linear2sRGB( bl );

        return v3i( clampu8( int32_t( r * 255 ) ), clampu8( int32_t( g * 255 ) ), clampu8( int32_t( b * 255 ) ) );
    }


    Lab::Lab()
        : L( 0 )
        , a( 0 )
        , b( 0 )
    {
    }

    Lab::Lab( float L, float a, float b )
        : L( L )
        , a( a )
        , b( b )
    {
    }

    static float labfunc( float t )
    {
        const float p1 = (6.f/29.f)*(6.f/29.f)*(6.f/29.f);
        const float p2 = (1.f/3.f)*(29.f/6.f)*(29.f/6.f);
        const float p3 = (4.f/29.f);

        if( t > p1 )
        {
            return pow( t, 1.f/3.f );
        }
        else
        {
            return p2 * t + p3;
        }
    }

    Lab::Lab( const XYZ& xyz )
    {
        L = 116 * labfunc( xyz.y * rwhite.y ) - 16;
        a = 500 * ( labfunc( xyz.x * rwhite.x ) - labfunc( xyz.y * rwhite.y ) );
        b = 200 * ( labfunc( xyz.y * rwhite.y ) - labfunc( xyz.z * rwhite.z ) );
    }

    Lab::Lab( const v3b& rgb )
    {
        new(this) Lab( XYZ( rgb ) );
    }

}

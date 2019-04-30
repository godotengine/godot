#ifndef __DARKRL__COLORSPACE_HPP__
#define __DARKRL__COLORSPACE_HPP__

#include "Vector.hpp"

namespace Color
{

    class Lab;

    class XYZ
    {
    public:
        XYZ( float x, float y, float z );
        XYZ( const v3b& rgb );
        XYZ( const Lab& lab );

        v3i RGB() const;

        float x, y, z;
    };

    class Lab
    {
    public:
        Lab();
        Lab( float L, float a, float b );
        Lab( const XYZ& xyz );
        Lab( const v3b& rgb );

        float L, a, b;
    };

}

#endif

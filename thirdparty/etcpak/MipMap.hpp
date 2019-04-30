#ifndef __MIPMAP_HPP__
#define __MIPMAP_HPP__

#include "Vector.hpp"

inline int NumberOfMipLevels( const v2i& size )
{
    return (int)floor( log2( std::max( size.x, size.y ) ) ) + 1;
}

#endif

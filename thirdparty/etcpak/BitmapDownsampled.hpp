#ifndef __DARKRL__BITMAPDOWNSAMPLED_HPP__
#define __DARKRL__BITMAPDOWNSAMPLED_HPP__

#include "Bitmap.hpp"

class BitmapDownsampled : public Bitmap
{
public:
    BitmapDownsampled( const Bitmap& bmp, unsigned int lines );
    ~BitmapDownsampled();
};

#endif

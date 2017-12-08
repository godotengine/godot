// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_IMAGE_IMAGE_H
#define NV_IMAGE_IMAGE_H

#include "nvimage.h"
#include "nvcore/Debug.h"

namespace nv
{
    class Color32;

    /// 32 bit RGBA image.
    class NVIMAGE_CLASS Image
    {
    public:

        enum Format 
        {
            Format_RGB,
            Format_ARGB,
        };

        Image();
        Image(const Image & img);
        ~Image();

        const Image & operator=(const Image & img);


        void allocate(uint w, uint h, uint d = 1);
        void acquire(Color32 * data, uint w, uint h, uint d = 1);
        //bool load(const char * name);

        void resize(uint w, uint h, uint d = 1);

        void wrap(void * data, uint w, uint h, uint d = 1);
        void unwrap();

        uint width() const;
        uint height() const;
        uint depth() const;

        const Color32 * scanline(uint h) const;
        Color32 * scanline(uint h);

        const Color32 * pixels() const;
        Color32 * pixels();

        const Color32 & pixel(uint idx) const;
        Color32 & pixel(uint idx);

        const Color32 & pixel(uint x, uint y, uint z = 0) const;
        Color32 & pixel(uint x, uint y,  uint z = 0);

        Format format() const;
        void setFormat(Format f);

        void fill(Color32 c);

    private:
        void free();

    private:
        uint m_width;
        uint m_height;
        uint m_depth;
        Format m_format;
        Color32 * m_data;
    };


    inline const Color32 & Image::pixel(uint x, uint y, uint z) const
    {
        nvDebugCheck(x < m_width && y < m_height && z < m_depth);
        return pixel((z * m_height + y) * m_width + x);
    }

    inline Color32 & Image::pixel(uint x, uint y, uint z)
    {
        nvDebugCheck(x < m_width && y < m_height && z < m_depth);
        return pixel((z * m_height + y) * m_width + x);
    }

} // nv namespace


#endif // NV_IMAGE_IMAGE_H

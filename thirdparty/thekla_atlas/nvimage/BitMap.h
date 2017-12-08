// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_IMAGE_BITMAP_H
#define NV_IMAGE_BITMAP_H

#include "nvimage.h"

#include "nvcore/BitArray.h"

namespace nv 
{
    /// Bit map. This should probably be called BitImage.
    class NVIMAGE_CLASS BitMap
    {
    public:
        BitMap() : m_width(0), m_height(0) {}
        BitMap(uint w, uint h) : m_width(w), m_height(h), m_bitArray(w*h) {}

        uint width() const { return m_width; }
        uint height() const { return m_height; }

        void resize(uint w, uint h, bool initValue);

        bool bitAt(uint x, uint y) const
        {
            nvDebugCheck(x < m_width && y < m_height);
            return m_bitArray.bitAt(y * m_width + x);
        }
        bool bitAt(uint idx) const
        {
            return m_bitArray.bitAt(idx);
        }

        void setBitAt(uint x, uint y)
        {
            nvDebugCheck(x < m_width && y < m_height);
            m_bitArray.setBitAt(y * m_width + x);
        }
        void setBitAt(uint idx)
        {
            m_bitArray.setBitAt(idx);
        }

        void clearBitAt(uint x, uint y)
        {
            nvDebugCheck(x < m_width && y < m_height);
            m_bitArray.clearBitAt(y * m_width + x);
        }
        void clearBitAt(uint idx)
        {
            m_bitArray.clearBitAt(idx);
        }

        void clearAll()
        {
            m_bitArray.clearAll();
        }

        void setAll()
        {
            m_bitArray.setAll();
        }

        void toggleAll()
        {
            m_bitArray.toggleAll();
        }

        friend void swap(BitMap & a, BitMap & b)
        {
            nvCheck(a.m_width == b.m_width);
            nvCheck(a.m_height == b.m_height);
            swap(a.m_bitArray, b.m_bitArray);
        }

    private:

        uint m_width;
        uint m_height;
        BitArray m_bitArray;

    };

} // nv namespace

#endif // NV_IMAGE_BITMAP_H

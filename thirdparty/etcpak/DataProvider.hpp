#ifndef __DATAPROVIDER_HPP__
#define __DATAPROVIDER_HPP__

#include <memory>
#include <stdint.h>
#include <vector>

#include "Bitmap.hpp"

struct DataPart
{
    const uint32_t* src;
    unsigned int width;
    unsigned int lines;
    unsigned int offset;
};

class DataProvider
{
public:
    DataProvider( const char* fn, bool mipmap, bool bgr );
    ~DataProvider();

    unsigned int NumberOfParts() const;

    DataPart NextPart();

    bool Alpha() const { return m_bmp[0]->Alpha(); }
    const v2i& Size() const { return m_bmp[0]->Size(); }
    const Bitmap& ImageData() const { return *m_bmp[0]; }

private:
    std::vector<std::unique_ptr<Bitmap>> m_bmp;
    Bitmap* m_current;
    unsigned int m_offset;
    unsigned int m_lines;
    bool m_mipmap;
    bool m_done;
};

#endif

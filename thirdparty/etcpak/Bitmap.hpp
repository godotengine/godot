#ifndef __DARKRL__BITMAP_HPP__
#define __DARKRL__BITMAP_HPP__

#include <future>
#include <memory>
#include <mutex>
#include <stdint.h>

#include "Semaphore.hpp"
#include "Vector.hpp"

enum class Channels
{
    RGB,
    Alpha
};

class Bitmap
{
public:
    Bitmap( const char* fn, unsigned int lines, bool bgr );
    Bitmap( const v2i& size );
    virtual ~Bitmap();

    void Write( const char* fn );

    uint32_t* Data() { if( m_load.valid() ) m_load.wait(); return m_data; }
    const uint32_t* Data() const { if( m_load.valid() ) m_load.wait(); return m_data; }
    const v2i& Size() const { return m_size; }
    bool Alpha() const { return m_alpha; }

    const uint32_t* NextBlock( unsigned int& lines, bool& done );

protected:
    Bitmap( const Bitmap& src, unsigned int lines );

    uint32_t* m_data;
    uint32_t* m_block;
    unsigned int m_lines;
    unsigned int m_linesLeft;
    v2i m_size;
    bool m_alpha;
    Semaphore m_sema;
    std::mutex m_lock;
    std::future<void> m_load;
};

typedef std::shared_ptr<Bitmap> BitmapPtr;

#endif

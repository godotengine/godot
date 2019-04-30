#include <assert.h>
#include <string.h>

#include "BlockData.hpp"
#include "ColorSpace.hpp"
#include "CpuArch.hpp"
#include "Debug.hpp"
#include "Dither.hpp"
#include "MipMap.hpp"
#include "mmap.hpp"
#include "ProcessAlpha.hpp"
#include "ProcessAlpha_AVX2.hpp"
#include "ProcessRGB.hpp"
#include "ProcessRGB_AVX2.hpp"
#include "Tables.hpp"
#include "TaskDispatch.hpp"

BlockData::BlockData( const char* fn )
    : m_file( fopen( fn, "rb" ) )
{
    assert( m_file );
    fseek( m_file, 0, SEEK_END );
    m_maplen = ftell( m_file );
    fseek( m_file, 0, SEEK_SET );
    m_data = (uint8_t*)mmap( nullptr, m_maplen, PROT_READ, MAP_SHARED, fileno( m_file ), 0 );

    auto data32 = (uint32_t*)m_data;
    if( *data32 == 0x03525650 )
    {
        // PVR
        switch( *(data32+2) )
        {
        case 6:
            m_type = Etc1;
            break;
        case 22:
            m_type = Etc2_RGB;
            break;
        case 23:
            m_type = Etc2_RGBA;
            break;
        default:
            assert( false );
            break;
        }

        m_size.y = *(data32+6);
        m_size.x = *(data32+7);
        m_dataOffset = 52 + *(data32+12);
    }
    else if( *data32 == 0x58544BAB )
    {
        // KTX
        switch( *(data32+7) )
        {
        case 0x9274:
            m_type = Etc2_RGB;
            break;
        case 0x9278:
            m_type = Etc2_RGBA;
            break;
        default:
            assert( false );
            break;
        }

        m_size.x = *(data32+9);
        m_size.y = *(data32+10);
        m_dataOffset = sizeof( uint32_t ) * 17 + *(data32+15);
    }
    else
    {
        assert( false );
    }
}

static uint8_t* OpenForWriting( const char* fn, size_t len, const v2i& size, FILE** f, int levels, BlockData::Type type )
{
    *f = fopen( fn, "wb+" );
    assert( *f );
    fseek( *f, len - 1, SEEK_SET );
    const char zero = 0;
    fwrite( &zero, 1, 1, *f );
    fseek( *f, 0, SEEK_SET );

    auto ret = (uint8_t*)mmap( nullptr, len, PROT_WRITE, MAP_SHARED, fileno( *f ), 0 );
    auto dst = (uint32_t*)ret;

    *dst++ = 0x03525650;  // version
    *dst++ = 0;           // flags
    switch( type )        // pixelformat[0]
    {
    case BlockData::Etc1:
        *dst++ = 6;
        break;
    case BlockData::Etc2_RGB:
        *dst++ = 22;
        break;
    case BlockData::Etc2_RGBA:
        *dst++ = 23;
        break;
    default:
        assert( false );
        break;
    }
    *dst++ = 0;           // pixelformat[1]
    *dst++ = 0;           // colourspace
    *dst++ = 0;           // channel type
    *dst++ = size.y;      // height
    *dst++ = size.x;      // width
    *dst++ = 1;           // depth
    *dst++ = 1;           // num surfs
    *dst++ = 1;           // num faces
    *dst++ = levels;      // mipmap count
    *dst++ = 0;           // metadata size

    return ret;
}

static int AdjustSizeForMipmaps( const v2i& size, int levels )
{
    int len = 0;
    v2i current = size;
    for( int i=1; i<levels; i++ )
    {
        assert( current.x != 1 || current.y != 1 );
        current.x = std::max( 1, current.x / 2 );
        current.y = std::max( 1, current.y / 2 );
        len += std::max( 4, current.x ) * std::max( 4, current.y ) / 2;
    }
    assert( current.x == 1 && current.y == 1 );
    return len;
}

BlockData::BlockData( const char* fn, const v2i& size, bool mipmap, Type type )
    : m_size( size )
    , m_dataOffset( 52 )
    , m_maplen( m_size.x*m_size.y/2 )
    , m_type( type )
{
    assert( m_size.x%4 == 0 && m_size.y%4 == 0 );

    uint32_t cnt = m_size.x * m_size.y / 16;
    DBGPRINT( cnt << " blocks" );

    int levels = 1;

    if( mipmap )
    {
        levels = NumberOfMipLevels( size );
        DBGPRINT( "Number of mipmaps: " << levels );
        m_maplen += AdjustSizeForMipmaps( size, levels );
    }

    if( type == Etc2_RGBA ) m_maplen *= 2;

    m_maplen += m_dataOffset;
    m_data = OpenForWriting( fn, m_maplen, m_size, &m_file, levels, type );
}

BlockData::BlockData( const v2i& size, bool mipmap, Type type )
    : m_size( size )
    , m_dataOffset( 52 )
    , m_file( nullptr )
    , m_maplen( m_size.x*m_size.y/2 )
    , m_type( type )
{
    assert( m_size.x%4 == 0 && m_size.y%4 == 0 );
    if( mipmap )
    {
        const int levels = NumberOfMipLevels( size );
        m_maplen += AdjustSizeForMipmaps( size, levels );
    }

    if( type == Etc2_RGBA ) m_maplen *= 2;

    m_maplen += m_dataOffset;
    m_data = new uint8_t[m_maplen];
}

BlockData::~BlockData()
{
    if( m_file )
    {
        munmap( m_data, m_maplen );
        fclose( m_file );
    }
    else
    {
        delete[] m_data;
    }
}

static uint64_t _f_rgba( uint8_t* ptr )
{
    return ProcessAlpha( ptr );
}

#ifdef __SSE4_1__
static uint64_t _f_rgba_avx2( uint8_t* ptr )
{
    return ProcessAlpha_AVX2( ptr );
}
#endif

static uint64_t _f_rgb( uint8_t* ptr )
{
    return ProcessRGB( ptr );
}

#ifdef __SSE4_1__
static uint64_t _f_rgb_avx2( uint8_t* ptr )
{
    return ProcessRGB_AVX2( ptr );
}
#endif

static uint64_t _f_rgb_dither( uint8_t* ptr )
{
    Dither( ptr );
    return ProcessRGB( ptr );
}

#ifdef __SSE4_1__
static uint64_t _f_rgb_dither_avx2( uint8_t* ptr )
{
    Dither( ptr );
    return ProcessRGB_AVX2( ptr );
}
#endif

static uint64_t _f_rgb_etc2( uint8_t* ptr )
{
    return ProcessRGB_ETC2( ptr );
}

#ifdef __SSE4_1__
static uint64_t _f_rgb_etc2_avx2( uint8_t* ptr )
{
    return ProcessRGB_ETC2_AVX2( ptr );
}
#endif

static uint64_t _f_rgb_etc2_dither( uint8_t* ptr )
{
    Dither( ptr );
    return ProcessRGB_ETC2( ptr );
}

#ifdef __SSE4_1__
static uint64_t _f_rgb_etc2_dither_avx2( uint8_t* ptr )
{
    Dither( ptr );
    return ProcessRGB_ETC2_AVX2( ptr );
}
#endif

void BlockData::Process( const uint32_t* src, uint32_t blocks, size_t offset, size_t width, Channels type, bool dither )
{
    uint32_t buf[4*4];
    int w = 0;

    auto dst = ((uint64_t*)( m_data + m_dataOffset )) + offset;

    uint64_t (*func)(uint8_t*);

    if( type == Channels::Alpha )
    {
#ifdef __SSE4_1__
        if( can_use_intel_core_4th_gen_features() )
        {
            if( m_type != Etc1 )
            {
                func = _f_rgb_etc2_avx2;
            }
            else
            {
                func = _f_rgb_avx2;
            }
        }
        else
#endif
        {
            if( m_type != Etc1 )
            {
                func = _f_rgb_etc2;
            }
            else
            {
                func = _f_rgb;
            }
        }

        do
        {
            auto ptr = buf;
            for( int x=0; x<4; x++ )
            {
                unsigned int a = *src >> 24;
                *ptr++ = a | ( a << 8 ) | ( a << 16 );
                src += width;
                a = *src >> 24;
                *ptr++ = a | ( a << 8 ) | ( a << 16 );
                src += width;
                a = *src >> 24;
                *ptr++ = a | ( a << 8 ) | ( a << 16 );
                src += width;
                a = *src >> 24;
                *ptr++ = a | ( a << 8 ) | ( a << 16 );
                src -= width * 3 - 1;
            }
            if( ++w == width/4 )
            {
                src += width * 3;
                w = 0;
            }

            *dst++ = func( (uint8_t*)buf );
        }
        while( --blocks );
    }
    else
    {
#ifdef __SSE4_1__
        if( can_use_intel_core_4th_gen_features() )
        {
            if( m_type != Etc1 )
            {
                if( dither )
                {
                    func = _f_rgb_etc2_dither_avx2;
                }
                else
                {
                    func = _f_rgb_etc2_avx2;
                }
            }
            else
            {
                if( dither )
                {
                    func = _f_rgb_dither_avx2;
                }
                else
                {
                    func = _f_rgb_avx2;
                }
            }
        }
        else
#endif
        {
            if( m_type != Etc1 )
            {
                if( dither )
                {
                    func = _f_rgb_etc2_dither;
                }
                else
                {
                    func = _f_rgb_etc2;
                }
            }
            else
            {
                if( dither )
                {
                    func = _f_rgb_dither;
                }
                else
                {
                    func = _f_rgb;
                }
            }
        }

        do
        {
            auto ptr = buf;
            for( int x=0; x<4; x++ )
            {
                *ptr++ = *src;
                src += width;
                *ptr++ = *src;
                src += width;
                *ptr++ = *src;
                src += width;
                *ptr++ = *src;
                src -= width * 3 - 1;
            }
            if( ++w == width/4 )
            {
                src += width * 3;
                w = 0;
            }

            *dst++ = func( (uint8_t*)buf );
        }
        while( --blocks );
    }
}

void BlockData::ProcessRGBA( const uint32_t* src, uint32_t blocks, size_t offset, size_t width, bool dither )
{
    assert( m_type == Etc2_RGBA );

    uint32_t buf[4*4];
    uint8_t buf8[4*4];
    int w = 0;

    auto dst = ((uint64_t*)( m_data + m_dataOffset )) + offset * 2;

    uint64_t (*func)(uint8_t*);
    uint64_t (*func_alpha)(uint8_t*);

#ifdef __SSE4_1__
    if( can_use_intel_core_4th_gen_features() )
    {
        if( dither )
        {
            func = _f_rgb_etc2_dither_avx2;
        }
        else
        {
            func = _f_rgb_etc2_avx2;
        }

        func_alpha = _f_rgba_avx2;
    }
    else
#endif
    {
        if( dither )
        {
            func = _f_rgb_etc2_dither;
        }
        else
        {
            func = _f_rgb_etc2;
        }

        func_alpha = _f_rgba;
    }

    do
    {
        auto ptr = buf;
        auto ptr8 = buf8;
        for( int x=0; x<4; x++ )
        {
            auto v = *src;
            *ptr++ = v;
            *ptr8++ = v >> 24;
            src += width;
            v = *src;
            *ptr++ = v;
            *ptr8++ = v >> 24;
            src += width;
            v = *src;
            *ptr++ = v;
            *ptr8++ = v >> 24;
            src += width;
            v = *src;
            *ptr++ = v;
            *ptr8++ = v >> 24;
            src -= width * 3 - 1;
        }
        if( ++w == width/4 )
        {
            src += width * 3;
            w = 0;
        }

        *dst++ = func_alpha( buf8 );
        *dst++ = func( (uint8_t*)buf );
    }
    while( --blocks );
}

namespace
{
struct BlockColor
{
    uint32_t r[2], g[2], b[2];
};

enum class Etc2Mode
{
    none,
    t,
    h,
    planar
};

Etc2Mode DecodeBlockColor( uint64_t d, BlockColor& c )
{
    if( d & 0x2 )
    {
        int32_t dr, dg, db;

        c.r[0] = ( d & 0xF8000000 ) >> 27;
        c.g[0] = ( d & 0x00F80000 ) >> 19;
        c.b[0] = ( d & 0x0000F800 ) >> 11;

        dr = ( d & 0x07000000 ) >> 24;
        dg = ( d & 0x00070000 ) >> 16;
        db = ( d & 0x00000700 ) >> 8;

        if( dr & 0x4 )
        {
            dr |= 0xFFFFFFF8;
        }
        if( dg & 0x4 )
        {
            dg |= 0xFFFFFFF8;
        }
        if( db & 0x4 )
        {
            db |= 0xFFFFFFF8;
        }

        int32_t r = static_cast<int32_t>(c.r[0]) + dr;
        int32_t g = static_cast<int32_t>(c.g[0]) + dg;
        int32_t b = static_cast<int32_t>(c.b[0]) + db;

        if ((r < 0) || (r > 31))
        {
            return Etc2Mode::t;
        }

        if ((g < 0) || (g > 31))
        {
            return Etc2Mode::h;
        }

        if ((b < 0) || (b > 31))
        {
            return Etc2Mode::planar;
        }

        c.r[1] = c.r[0] + dr;
        c.g[1] = c.g[0] + dg;
        c.b[1] = c.b[0] + db;

        for( int i=0; i<2; i++ )
        {
            c.r[i] = ( c.r[i] << 3 ) | ( c.r[i] >> 2 );
            c.g[i] = ( c.g[i] << 3 ) | ( c.g[i] >> 2 );
            c.b[i] = ( c.b[i] << 3 ) | ( c.b[i] >> 2 );
        }
    }
    else
    {
        c.r[0] = ( ( d & 0xF0000000 ) >> 24 ) | ( ( d & 0xF0000000 ) >> 28 );
        c.r[1] = ( ( d & 0x0F000000 ) >> 20 ) | ( ( d & 0x0F000000 ) >> 24 );
        c.g[0] = ( ( d & 0x00F00000 ) >> 16 ) | ( ( d & 0x00F00000 ) >> 20 );
        c.g[1] = ( ( d & 0x000F0000 ) >> 12 ) | ( ( d & 0x000F0000 ) >> 16 );
        c.b[0] = ( ( d & 0x0000F000 ) >> 8  ) | ( ( d & 0x0000F000 ) >> 12 );
        c.b[1] = ( ( d & 0x00000F00 ) >> 4  ) | ( ( d & 0x00000F00 ) >> 8  );
    }
    return Etc2Mode::none;
}

inline int32_t expand6(uint32_t value)
{
    return (value << 2) | (value >> 4);
}

inline int32_t expand7(uint32_t value)
{
    return (value << 1) | (value >> 6);
}

void DecodePlanar(uint64_t block, uint32_t* l[4])
{
    const auto bv = expand6((block >> ( 0 + 32)) & 0x3F);
    const auto gv = expand7((block >> ( 6 + 32)) & 0x7F);
    const auto rv = expand6((block >> (13 + 32)) & 0x3F);

    const auto bh = expand6((block >> (19 + 32)) & 0x3F);
    const auto gh = expand7((block >> (25 + 32)) & 0x7F);

    const auto rh0 = (block >> (32 - 32)) & 0x01;
    const auto rh1 = ((block >> (34 - 32)) & 0x1F) << 1;
    const auto rh = expand6(rh0 | rh1);

    const auto bo0 = (block >> (39 - 32)) & 0x07;
    const auto bo1 = ((block >> (43 - 32)) & 0x3) << 3;
    const auto bo2 = ((block >> (48 - 32)) & 0x1) << 5;
    const auto bo = expand6(bo0 | bo1 | bo2);
    const auto go0 = (block >> (49 - 32)) & 0x3F;
    const auto go1 = ((block >> (56 - 32)) & 0x01) << 6;
    const auto go = expand7(go0 | go1);
    const auto ro = expand6((block >> (57 - 32)) & 0x3F);

    for (auto j = 0; j < 4; j++)
    {
        for (auto i = 0; i < 4; i++)
        {
            uint32_t r = clampu8((i * (rh - ro) + j * (rv - ro) + 4 * ro + 2) >> 2);
            uint32_t g = clampu8((i * (gh - go) + j * (gv - go) + 4 * go + 2) >> 2);
            uint32_t b = clampu8((i * (bh - bo) + j * (bv - bo) + 4 * bo + 2) >> 2);

            *l[j]++ = r | ( g << 8 ) | ( b << 16 ) | 0xFF000000;
        }
    }
}

}

BitmapPtr BlockData::Decode()
{
    if( m_type == Etc2_RGBA )
    {
        return DecodeRGBA();
    }
    else
    {
        return DecodeRGB();
    }
}

static uint64_t ConvertByteOrder( uint64_t d )
{
    return ( ( d & 0xFF000000FF000000 ) >> 24 ) |
           ( ( d & 0x000000FF000000FF ) << 24 ) |
           ( ( d & 0x00FF000000FF0000 ) >> 8 ) |
           ( ( d & 0x0000FF000000FF00 ) << 8 );
}

static void DecodeRGBPart( uint32_t* l[4], uint64_t d )
{
    d = ConvertByteOrder( d );

    BlockColor c;
    const auto mode = DecodeBlockColor( d, c );

    if (mode == Etc2Mode::planar)
    {
        DecodePlanar(d, l);
        return;
    }

    unsigned int tcw[2];
    tcw[0] = ( d & 0xE0 ) >> 5;
    tcw[1] = ( d & 0x1C ) >> 2;

    if( d & 0x1 )
    {
        int o = 0;
        for( int i=0; i<4; i++ )
        {
            for( int j=0; j<4; j++ )
            {
                const auto mod = g_table[tcw[j/2]][ ( ( d >> ( o + 32 + j ) ) & 0x1 ) | ( ( d >> ( o + 47 + j ) ) & 0x2 ) ];
                const auto r = clampu8( c.r[j/2] + mod );
                const auto g = clampu8( c.g[j/2] + mod );
                const auto b = clampu8( c.b[j/2] + mod );
                *l[j]++ = r | ( g << 8 ) | ( b << 16 ) | 0xFF000000;
            }
            o += 4;
        }
    }
    else
    {
        int o = 0;
        for( int i=0; i<4; i++ )
        {
            const auto tbl = g_table[tcw[i/2]];
            const auto cr = c.r[i/2];
            const auto cg = c.g[i/2];
            const auto cb = c.b[i/2];

            for( int j=0; j<4; j++ )
            {
                const auto mod = tbl[ ( ( d >> ( o + 32 + j ) ) & 0x1 ) | ( ( d >> ( o + 47 + j ) ) & 0x2 ) ];
                const auto r = clampu8( cr + mod );
                const auto g = clampu8( cg + mod );
                const auto b = clampu8( cb + mod );
                *l[j]++ = r | ( g << 8 ) | ( b << 16 ) | 0xFF000000;
            }
            o += 4;
        }
    }
}

static void DecodeAlphaPart( uint32_t* l[4], uint64_t d )
{
    d = ( ( d & 0xFF00000000000000 ) >> 56 ) |
        ( ( d & 0x00FF000000000000 ) >> 40 ) |
        ( ( d & 0x0000FF0000000000 ) >> 24 ) |
        ( ( d & 0x000000FF00000000 ) >> 8 ) |
        ( ( d & 0x00000000FF000000 ) << 8 ) |
        ( ( d & 0x0000000000FF0000 ) << 24 ) |
        ( ( d & 0x000000000000FF00 ) << 40 ) |
        ( ( d & 0x00000000000000FF ) << 56 );

    unsigned int base = d >> 56;
    unsigned int mul = ( d >> 52 ) & 0xF;
    unsigned int idx = ( d >> 48 ) & 0xF;

    const auto tbl = g_alpha[idx];

    int o = 45;
    for( int i=0; i<4; i++ )
    {
        for( int j=0; j<4; j++ )
        {
            const auto mod = tbl[ ( d >> o ) & 0x7 ];
            const auto a = clampu8( base + mod * mul );
            *l[j] = ( *l[j] & 0x00FFFFFF ) | ( a << 24 );
            l[j]++;
            o -= 3;
        }
    }
}

BitmapPtr BlockData::DecodeRGB()
{
    auto ret = std::make_shared<Bitmap>( m_size );

    uint32_t* l[4];
    l[0] = ret->Data();
    l[1] = l[0] + m_size.x;
    l[2] = l[1] + m_size.x;
    l[3] = l[2] + m_size.x;

    const uint64_t* src = (const uint64_t*)( m_data + m_dataOffset );

    for( int y=0; y<m_size.y/4; y++ )
    {
        for( int x=0; x<m_size.x/4; x++ )
        {
            uint64_t d = *src++;
            DecodeRGBPart( l, d );
        }

        for( int i=0; i<4; i++ )
        {
            l[i] += m_size.x * 3;
        }
    }

    return ret;
}

BitmapPtr BlockData::DecodeRGBA()
{
    auto ret = std::make_shared<Bitmap>( m_size );

    uint32_t* l[4];
    l[0] = ret->Data();
    l[1] = l[0] + m_size.x;
    l[2] = l[1] + m_size.x;
    l[3] = l[2] + m_size.x;

    const uint64_t* src = (const uint64_t*)( m_data + m_dataOffset );

    for( int y=0; y<m_size.y/4; y++ )
    {
        for( int x=0; x<m_size.x/4; x++ )
        {
            uint64_t a = *src++;
            uint64_t d = *src++;
            DecodeRGBPart( l, d );

            for( int i=0; i<4; i++ )
            {
                l[i] -= 4;
            }

            DecodeAlphaPart( l, a );
        }

        for( int i=0; i<4; i++ )
        {
            l[i] += m_size.x * 3;
        }
    }

    return ret;
}

// Block type:
//  red - 2x4, green - 4x2, blue - planar
//  dark - 444, bright - 555 + 333
void BlockData::Dissect()
{
    auto size = m_size / 4;
    const uint64_t* data = (const uint64_t*)( m_data + m_dataOffset );

    auto src = data;

    auto bmp = std::make_shared<Bitmap>( size );
    auto dst = bmp->Data();

    auto bmp2 = std::make_shared<Bitmap>( m_size );
    uint32_t* l[4];
    l[0] = bmp2->Data();
    l[1] = l[0] + m_size.x;
    l[2] = l[1] + m_size.x;
    l[3] = l[2] + m_size.x;

    auto bmp3 = std::make_shared<Bitmap>( size );
    auto dst3 = bmp3->Data();

    for( int y=0; y<size.y; y++ )
    {
        for( int x=0; x<size.x; x++ )
        {
            uint64_t d = ConvertByteOrder( *src++ );

            BlockColor c;
            const auto mode = DecodeBlockColor( d, c );

            switch( mode )
            {
            case Etc2Mode::none:
                switch( d & 0x3 )
                {
                case 0:
                    *dst++ = 0xFF000088;
                    break;
                case 1:
                    *dst++ = 0xFF008800;
                    break;
                case 2:
                    *dst++ = 0xFF0000FF;
                    break;
                case 3:
                    *dst++ = 0xFF00FF00;
                    break;
                default:
                    assert( false );
                    break;
                }
                break;
            case Etc2Mode::planar:
                *dst++ = 0xFFFF0000;
                break;
            default:
                assert( false );
                break;
            }

            unsigned int tcw[2];
            tcw[0] = ( d & 0xE0 );
            tcw[1] = ( d & 0x1C ) << 3;

            *dst3++ = 0xFF000000 | ( tcw[0] << 8 ) | ( tcw[1] );

            if( d & 0x1 )
            {
                for( int i=0; i<4; i++ )
                {
                    *l[0]++ = 0xFF000000 | ( c.b[0] << 16 ) | ( c.g[0] << 8 ) | c.r[0];
                    *l[1]++ = 0xFF000000 | ( c.b[0] << 16 ) | ( c.g[0] << 8 ) | c.r[0];
                    *l[2]++ = 0xFF000000 | ( c.b[1] << 16 ) | ( c.g[1] << 8 ) | c.r[1];
                    *l[3]++ = 0xFF000000 | ( c.b[1] << 16 ) | ( c.g[1] << 8 ) | c.r[1];
                }
            }
            else
            {
                for( int i=0; i<2; i++ )
                {
                    *l[0]++ = 0xFF000000 | ( c.b[0] << 16 ) | ( c.g[0] << 8 ) | c.r[0];
                    *l[1]++ = 0xFF000000 | ( c.b[0] << 16 ) | ( c.g[0] << 8 ) | c.r[0];
                    *l[2]++ = 0xFF000000 | ( c.b[0] << 16 ) | ( c.g[0] << 8 ) | c.r[0];
                    *l[3]++ = 0xFF000000 | ( c.b[0] << 16 ) | ( c.g[0] << 8 ) | c.r[0];
                }
                for( int i=0; i<2; i++ )
                {
                    *l[0]++ = 0xFF000000 | ( c.b[1] << 16 ) | ( c.g[1] << 8 ) | c.r[1];
                    *l[1]++ = 0xFF000000 | ( c.b[1] << 16 ) | ( c.g[1] << 8 ) | c.r[1];
                    *l[2]++ = 0xFF000000 | ( c.b[1] << 16 ) | ( c.g[1] << 8 ) | c.r[1];
                    *l[3]++ = 0xFF000000 | ( c.b[1] << 16 ) | ( c.g[1] << 8 ) | c.r[1];
                }
            }
        }
        l[0] += m_size.x * 3;
        l[1] += m_size.x * 3;
        l[2] += m_size.x * 3;
        l[3] += m_size.x * 3;
    }

    bmp->Write( "out_block_type.png" );
    bmp2->Write( "out_block_color.png" );
    bmp3->Write( "out_block_selectors.png" );
}

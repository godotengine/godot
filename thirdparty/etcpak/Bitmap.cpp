#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <png.h>
#include "lz4/lz4.h"

#include "Bitmap.hpp"
#include "Debug.hpp"

Bitmap::Bitmap( const char* fn, unsigned int lines, bool bgr )
    : m_block( nullptr )
    , m_lines( lines )
    , m_alpha( true )
    , m_sema( 0 )
{
    FILE* f = fopen( fn, "rb" );
    assert( f );

    char buf[4];
    fread( buf, 1, 4, f );
    if( memcmp( buf, "raw4", 4 ) == 0 )
    {
        uint8_t a;
        fread( &a, 1, 1, f );
        m_alpha = a == 1;
        uint32_t d;
        fread( &d, 1, 4, f );
        m_size.x = d;
        fread( &d, 1, 4, f );
        m_size.y = d;
        DBGPRINT( "Raw bitmap " << fn << "  " << m_size.x << "x" << m_size.y );

        assert( m_size.x % 4 == 0 );
        assert( m_size.y % 4 == 0 );

        int32_t csize;
        fread( &csize, 1, 4, f );
        char* cbuf = new char[csize];
        fread( cbuf, 1, csize, f );
        fclose( f );

        m_block = m_data = new uint32_t[m_size.x*m_size.y];
        m_linesLeft = m_size.y / 4;

        LZ4_decompress_fast( cbuf, (char*)m_data, m_size.x*m_size.y*4 );
        delete[] cbuf;

        for( int i=0; i<m_size.y/4; i++ )
        {
            m_sema.unlock();
        }
    }
    else
    {
        fseek( f, 0, SEEK_SET );

        unsigned int sig_read = 0;
        int bit_depth, color_type, interlace_type;

        png_structp png_ptr = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
        png_infop info_ptr = png_create_info_struct( png_ptr );
        setjmp( png_jmpbuf( png_ptr ) );

        png_init_io( png_ptr, f );
        png_set_sig_bytes( png_ptr, sig_read );

        png_uint_32 w, h;

        png_read_info( png_ptr, info_ptr );
        png_get_IHDR( png_ptr, info_ptr, &w, &h, &bit_depth, &color_type, &interlace_type, NULL, NULL );

        m_size = v2i( w, h );

        png_set_strip_16( png_ptr );
        if( color_type == PNG_COLOR_TYPE_PALETTE )
        {
            png_set_palette_to_rgb( png_ptr );
        }
        else if( color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8 )
        {
            png_set_expand_gray_1_2_4_to_8( png_ptr );
        }
        if( png_get_valid( png_ptr, info_ptr, PNG_INFO_tRNS ) )
        {
            png_set_tRNS_to_alpha( png_ptr );
        }
        if( color_type == PNG_COLOR_TYPE_GRAY_ALPHA )
        {
            png_set_gray_to_rgb(png_ptr);
        }
        if( bgr )
        {
            png_set_bgr(png_ptr);
        }

        switch( color_type )
        {
        case PNG_COLOR_TYPE_PALETTE:
            if( !png_get_valid( png_ptr, info_ptr, PNG_INFO_tRNS ) )
            {
                png_set_filler( png_ptr, 0xff, PNG_FILLER_AFTER );
                m_alpha = false;
            }
            break;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            png_set_gray_to_rgb( png_ptr );
            break;
        case PNG_COLOR_TYPE_RGB:
            png_set_filler( png_ptr, 0xff, PNG_FILLER_AFTER );
            m_alpha = false;
            break;
        default:
            break;
        }

        DBGPRINT( "Bitmap " << fn << "  " << w << "x" << h );

        assert( w % 4 == 0 );
        assert( h % 4 == 0 );

        m_block = m_data = new uint32_t[w*h];
        m_linesLeft = h / 4;

        m_load = std::async( std::launch::async, [this, f, png_ptr, info_ptr]() mutable
        {
            auto ptr = m_data;
            unsigned int lines = 0;
            for( int i=0; i<m_size.y / 4; i++ )
            {
                for( int j=0; j<4; j++ )
                {
                    png_read_rows( png_ptr, (png_bytepp)&ptr, NULL, 1 );
                    ptr += m_size.x;
                }
                lines++;
                if( lines >= m_lines )
                {
                    lines = 0;
                    m_sema.unlock();
                }
            }

            if( lines != 0 )
            {
                m_sema.unlock();
            }

            png_read_end( png_ptr, info_ptr );
            png_destroy_read_struct( &png_ptr, &info_ptr, NULL );
            fclose( f );
        } );
    }
}

Bitmap::Bitmap( const v2i& size )
    : m_data( new uint32_t[size.x*size.y] )
    , m_block( nullptr )
    , m_lines( 1 )
    , m_linesLeft( size.y / 4 )
    , m_size( size )
    , m_sema( 0 )
{
}

Bitmap::Bitmap( const Bitmap& src, unsigned int lines )
    : m_lines( lines )
    , m_alpha( src.Alpha() )
    , m_sema( 0 )
{
}

Bitmap::~Bitmap()
{
    delete[] m_data;
}

void Bitmap::Write( const char* fn )
{
    FILE* f = fopen( fn, "wb" );
    assert( f );

    png_structp png_ptr = png_create_write_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
    png_infop info_ptr = png_create_info_struct( png_ptr );
    setjmp( png_jmpbuf( png_ptr ) );
    png_init_io( png_ptr, f );

    png_set_IHDR( png_ptr, info_ptr, m_size.x, m_size.y, 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE );

    png_write_info( png_ptr, info_ptr );

    uint32_t* ptr = m_data;
    for( int i=0; i<m_size.y; i++ )
    {
        png_write_rows( png_ptr, (png_bytepp)(&ptr), 1 );
        ptr += m_size.x;
    }

    png_write_end( png_ptr, info_ptr );
    png_destroy_write_struct( &png_ptr, &info_ptr );

    fclose( f );
}

const uint32_t* Bitmap::NextBlock( unsigned int& lines, bool& done )
{
    std::lock_guard<std::mutex> lock( m_lock );
    lines = std::min( m_lines, m_linesLeft );
    auto ret = m_block;
    m_sema.lock();
    m_block += m_size.x * 4 * lines;
    m_linesLeft -= lines;
    done = m_linesLeft == 0;
    return ret;
}

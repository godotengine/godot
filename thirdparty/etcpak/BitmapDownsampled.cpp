#include <string.h>
#include <utility>

#include "BitmapDownsampled.hpp"
#include "Debug.hpp"

BitmapDownsampled::BitmapDownsampled( const Bitmap& bmp, unsigned int lines )
    : Bitmap( bmp, lines )
{
    m_size.x = std::max( 1, bmp.Size().x / 2 );
    m_size.y = std::max( 1, bmp.Size().y / 2 );

    int w = std::max( m_size.x, 4 );
    int h = std::max( m_size.y, 4 );

    DBGPRINT( "Subbitmap " << m_size.x << "x" << m_size.y );

    m_block = m_data = new uint32_t[w*h];

    if( m_size.x < w || m_size.y < h )
    {
        memset( m_data, 0, w*h*sizeof( uint32_t ) );
        m_linesLeft = h / 4;
        unsigned int lines = 0;
        for( int i=0; i<h/4; i++ )
        {
            for( int j=0; j<4; j++ )
            {
                lines++;
                if( lines > m_lines )
                {
                    lines = 0;
                    m_sema.unlock();
                }
            }
        }
        if( lines != 0 )
        {
            m_sema.unlock();
        }
    }
    else
    {
        m_linesLeft = h / 4;
        m_load = std::async( std::launch::async, [this, &bmp, w, h]() mutable
        {
            auto ptr = m_data;
            auto src1 = bmp.Data();
            auto src2 = src1 + bmp.Size().x;
            unsigned int lines = 0;
            for( int i=0; i<h/4; i++ )
            {
                for( int j=0; j<4; j++ )
                {
                    for( int k=0; k<m_size.x; k++ )
                    {
                        int r = ( ( *src1 & 0x000000FF ) + ( *(src1+1) & 0x000000FF ) + ( *src2 & 0x000000FF ) + ( *(src2+1) & 0x000000FF ) ) / 4;
                        int g = ( ( ( *src1 & 0x0000FF00 ) + ( *(src1+1) & 0x0000FF00 ) + ( *src2 & 0x0000FF00 ) + ( *(src2+1) & 0x0000FF00 ) ) / 4 ) & 0x0000FF00;
                        int b = ( ( ( *src1 & 0x00FF0000 ) + ( *(src1+1) & 0x00FF0000 ) + ( *src2 & 0x00FF0000 ) + ( *(src2+1) & 0x00FF0000 ) ) / 4 ) & 0x00FF0000;
                        int a = ( ( ( ( ( *src1 & 0xFF000000 ) >> 8 ) + ( ( *(src1+1) & 0xFF000000 ) >> 8 ) + ( ( *src2 & 0xFF000000 ) >> 8 ) + ( ( *(src2+1) & 0xFF000000 ) >> 8 ) ) / 4 ) & 0x00FF0000 ) << 8;
                        *ptr++ = r | g | b | a;
                        src1 += 2;
                        src2 += 2;
                    }
                    src1 += m_size.x * 2;
                    src2 += m_size.x * 2;
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
        } );
    }
}

BitmapDownsampled::~BitmapDownsampled()
{
}

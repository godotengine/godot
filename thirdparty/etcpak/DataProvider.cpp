#include <assert.h>
#include <utility>

#include "BitmapDownsampled.hpp"
#include "DataProvider.hpp"
#include "MipMap.hpp"

DataProvider::DataProvider( const char* fn, bool mipmap, bool bgr )
    : m_offset( 0 )
    , m_mipmap( mipmap )
    , m_done( false )
    , m_lines( 32 )
{
    m_bmp.emplace_back( new Bitmap( fn, m_lines, bgr ) );
    m_current = m_bmp[0].get();
}

DataProvider::~DataProvider()
{
}

unsigned int DataProvider::NumberOfParts() const
{
    unsigned int parts = ( ( m_bmp[0]->Size().y / 4 ) + m_lines - 1 ) / m_lines;

    if( m_mipmap )
    {
        v2i current = m_bmp[0]->Size();
        int levels = NumberOfMipLevels( current );
        unsigned int lines = m_lines;
        for( int i=1; i<levels; i++ )
        {
            assert( current.x != 1 || current.y != 1 );
            current.x = std::max( 1, current.x / 2 );
            current.y = std::max( 1, current.y / 2 );
            lines *= 2;
            parts += ( ( std::max( 4, current.y ) / 4 ) + lines - 1 ) / lines;
        }
        assert( current.x == 1 && current.y == 1 );
    }

    return parts;
}

DataPart DataProvider::NextPart()
{
    assert( !m_done );

    unsigned int lines = m_lines;
    bool done;

    const auto ptr = m_current->NextBlock( lines, done );
    DataPart ret = {
        ptr,
        std::max<unsigned int>( 4, m_current->Size().x ),
        lines,
        m_offset
    };

    m_offset += m_current->Size().x / 4 * lines;

    if( done )
    {
        if( m_mipmap && ( m_current->Size().x != 1 || m_current->Size().y != 1 ) )
        {
            m_lines *= 2;
            m_bmp.emplace_back( new BitmapDownsampled( *m_current, m_lines ) );
            m_current = m_bmp[m_bmp.size()-1].get();
        }
        else
        {
            m_done = true;
        }
    }

    return ret;
}

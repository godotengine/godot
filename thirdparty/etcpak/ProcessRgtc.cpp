// -- GODOT start --

#include "ForceInline.hpp"
#include "ProcessRgtc.hpp"

#include <assert.h>
#include <string.h>

static const uint8_t AlphaIndexTable[8] = { 1, 7, 6, 5, 4, 3, 2, 0 };

static etcpak_force_inline uint64_t ProcessAlpha( const uint8_t* src )
{
    uint8_t solid8 = *src;
    uint16_t solid16 = uint16_t( solid8 ) | ( uint16_t( solid8 ) << 8 );
    uint32_t solid32 = uint32_t( solid16 ) | ( uint32_t( solid16 ) << 16 );
    uint64_t solid64 = uint64_t( solid32 ) | ( uint64_t( solid32 ) << 32 );
    if( memcmp( src, &solid64, 8 ) == 0 && memcmp( src+8, &solid64, 8 ) == 0 )
    {
        return solid8;
    }

    uint8_t min = src[0];
    uint8_t max = min;
    for( int i=1; i<16; i++ )
    {
        const auto v = src[i];
        if( v > max ) max = v;
        else if( v < min ) min = v;
    }

    uint32_t range = ( 8 << 13 ) / ( 1 + max - min );
    uint64_t data = 0;
    for( int i=0; i<16; i++ )
    {
        uint8_t a = src[i] - min;
        uint64_t idx = AlphaIndexTable[( a * range ) >> 13];
        data |= idx << (i*3);
    }

    return max | ( min << 8 ) | ( data << 16 );
}

void CompressRgtcR(const uint32_t *src, uint64_t *dst, uint32_t blocks, size_t width) 
{
	int i = 0;
	auto ptr = dst;
	do 
	{
		uint32_t rgba[4 * 4];
		uint8_t r[4 * 4];

		auto tmp = (char *)rgba;
		memcpy(tmp, src + width * 0, 4 * 4);
		memcpy(tmp + 4 * 4, src + width * 1, 4 * 4);
		memcpy(tmp + 8 * 4, src + width * 2, 4 * 4);
		memcpy(tmp + 12 * 4, src + width * 3, 4 * 4);
		src += 4;
		if (++i == width / 4) 
		{
			src += width * 3;
			i = 0;
		}

		for (int i = 0; i < 16; i++) 
		{
			r[i] = rgba[i] & 0x000000FF;
		}
		*ptr++ = ProcessAlpha(r);
	} 
	while (--blocks);
}

void CompressRgtcRG(const uint32_t *src, uint64_t *dst, uint32_t blocks, size_t width) 
{
	int i = 0;
	auto ptr = dst;
	do 
	{
		uint32_t rgba[4 * 4];
		uint8_t r[4 * 4];
		uint8_t g[4 * 4];

		auto tmp = (char *)rgba;
		memcpy(tmp, src + width * 0, 4 * 4);
		memcpy(tmp + 4 * 4, src + width * 1, 4 * 4);
		memcpy(tmp + 8 * 4, src + width * 2, 4 * 4);
		memcpy(tmp + 12 * 4, src + width * 3, 4 * 4);
		src += 4;
		if (++i == width / 4) 
		{
			src += width * 3;
			i = 0;
		}

		for (int i = 0; i < 16; i++) 
		{
			r[i] = rgba[i] & 0x000000FF;
			g[i] = (rgba[i] & 0x0000FF00) >> 8;
		}
		*ptr++ = ProcessAlpha(r);
		*ptr++ = ProcessAlpha(g);
	} 
	while (--blocks);
}

// -- GODOT end --

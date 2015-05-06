/* -----------------------------------------------------------------------------

	Copyright (c) 2006 Simon Brown                          si@sjbrown.co.uk

	Permission is hereby granted, free of charge, to any person obtaining
	a copy of this software and associated documentation files (the 
	"Software"), to	deal in the Software without restriction, including
	without limitation the rights to use, copy, modify, merge, publish,
	distribute, sublicense, and/or sell copies of the Software, and to 
	permit persons to whom the Software is furnished to do so, subject to 
	the following conditions:

	The above copyright notice and this permission notice shall be included
	in all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
	OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
	MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
	IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
	CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
	TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
	SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
	
   -------------------------------------------------------------------------- */
   
#include "colourblock.h"

namespace squish {

static int FloatToInt( float a, int limit )
{
	// use ANSI round-to-zero behaviour to get round-to-nearest
	int i = ( int )( a + 0.5f );

	// clamp to the limit
	if( i < 0 )
		i = 0;
	else if( i > limit )
		i = limit; 

	// done
	return i;
}

static int FloatTo565( Vec3::Arg colour )
{
	// get the components in the correct range
	int r = FloatToInt( 31.0f*colour.X(), 31 );
	int g = FloatToInt( 63.0f*colour.Y(), 63 );
	int b = FloatToInt( 31.0f*colour.Z(), 31 );
	
	// pack into a single value
	return ( r << 11 ) | ( g << 5 ) | b;
}

static void WriteColourBlock( int a, int b, u8* indices, void* block )
{
	// get the block as bytes
	u8* bytes = ( u8* )block;

	// write the endpoints
	bytes[0] = ( u8 )( a & 0xff );
	bytes[1] = ( u8 )( a >> 8 );
	bytes[2] = ( u8 )( b & 0xff );
	bytes[3] = ( u8 )( b >> 8 );
	
	// write the indices
	for( int i = 0; i < 4; ++i )
	{
		u8 const* ind = indices + 4*i;
		bytes[4 + i] = ind[0] | ( ind[1] << 2 ) | ( ind[2] << 4 ) | ( ind[3] << 6 );
	}
}

void WriteColourBlock3( Vec3::Arg start, Vec3::Arg end, u8 const* indices, void* block )
{
	// get the packed values
	int a = FloatTo565( start );
	int b = FloatTo565( end );

	// remap the indices
	u8 remapped[16];
	if( a <= b )
	{
		// use the indices directly
		for( int i = 0; i < 16; ++i )
			remapped[i] = indices[i];
	}
	else
	{
		// swap a and b
		std::swap( a, b );
		for( int i = 0; i < 16; ++i )
		{
			if( indices[i] == 0 )
				remapped[i] = 1;
			else if( indices[i] == 1 )
				remapped[i] = 0;
			else
				remapped[i] = indices[i];
		}
	}
	
	// write the block
	WriteColourBlock( a, b, remapped, block );
}

void WriteColourBlock4( Vec3::Arg start, Vec3::Arg end, u8 const* indices, void* block )
{
	// get the packed values
	int a = FloatTo565( start );
	int b = FloatTo565( end );

	// remap the indices
	u8 remapped[16];
	if( a < b )
	{
		// swap a and b
		std::swap( a, b );
		for( int i = 0; i < 16; ++i )
			remapped[i] = ( indices[i] ^ 0x1 ) & 0x3;
	}
	else if( a == b )
	{
		// use index 0
		for( int i = 0; i < 16; ++i )
			remapped[i] = 0;
	}
	else
	{
		// use the indices directly
		for( int i = 0; i < 16; ++i )
			remapped[i] = indices[i];
	}
	
	// write the block
	WriteColourBlock( a, b, remapped, block );
}

static int Unpack565( u8 const* packed, u8* colour )
{
	// build the packed value
	int value = ( int )packed[0] | ( ( int )packed[1] << 8 );
	
	// get the components in the stored range
	u8 red = ( u8 )( ( value >> 11 ) & 0x1f );
	u8 green = ( u8 )( ( value >> 5 ) & 0x3f );
	u8 blue = ( u8 )( value & 0x1f );

	// scale up to 8 bits
	colour[0] = ( red << 3 ) | ( red >> 2 );
	colour[1] = ( green << 2 ) | ( green >> 4 );
	colour[2] = ( blue << 3 ) | ( blue >> 2 );
	colour[3] = 255;
	
	// return the value
	return value;
}

void DecompressColour( u8* rgba, void const* block, bool isDxt1 )
{
	// get the block bytes
	u8 const* bytes = reinterpret_cast< u8 const* >( block );
	
	// unpack the endpoints
	u8 codes[16];
	int a = Unpack565( bytes, codes );
	int b = Unpack565( bytes + 2, codes + 4 );
	
	// generate the midpoints
	for( int i = 0; i < 3; ++i )
	{
		int c = codes[i];
		int d = codes[4 + i];

		if( isDxt1 && a <= b )
		{
			codes[8 + i] = ( u8 )( ( c + d )/2 );
			codes[12 + i] = 0;
		}
		else
		{
			codes[8 + i] = ( u8 )( ( 2*c + d )/3 );
			codes[12 + i] = ( u8 )( ( c + 2*d )/3 );
		}
	}
	
	// fill in alpha for the intermediate values
	codes[8 + 3] = 255;
	codes[12 + 3] = ( isDxt1 && a <= b ) ? 0 : 255;
	
	// unpack the indices
	u8 indices[16];
	for( int i = 0; i < 4; ++i )
	{
		u8* ind = indices + 4*i;
		u8 packed = bytes[4 + i];
		
		ind[0] = packed & 0x3;
		ind[1] = ( packed >> 2 ) & 0x3;
		ind[2] = ( packed >> 4 ) & 0x3;
		ind[3] = ( packed >> 6 ) & 0x3;
	}

	// store out the colours
	for( int i = 0; i < 16; ++i )
	{
		u8 offset = 4*indices[i];
		for( int j = 0; j < 4; ++j )
			rgba[4*i + j] = codes[offset + j];
	}
}

} // namespace squish

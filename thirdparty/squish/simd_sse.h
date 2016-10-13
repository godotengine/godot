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
   
#ifndef SQUISH_SIMD_SSE_H
#define SQUISH_SIMD_SSE_H

#include <xmmintrin.h>
#if ( SQUISH_USE_SSE > 1 )
#include <emmintrin.h>
#endif

#define SQUISH_SSE_SPLAT( a )										\
	( ( a ) | ( ( a ) << 2 ) | ( ( a ) << 4 ) | ( ( a ) << 6 ) )

#define SQUISH_SSE_SHUF( x, y, z, w )								\
	( ( x ) | ( ( y ) << 2 ) | ( ( z ) << 4 ) | ( ( w ) << 6 ) )

namespace squish {

#define VEC4_CONST( X ) Vec4( X )

class Vec4
{
public:
	typedef Vec4 const& Arg;

	Vec4() {}
		
	explicit Vec4( __m128 v ) : m_v( v ) {}
	
	Vec4( Vec4 const& arg ) : m_v( arg.m_v ) {}
	
	Vec4& operator=( Vec4 const& arg )
	{
		m_v = arg.m_v;
		return *this;
	}
	
	explicit Vec4( float s ) : m_v( _mm_set1_ps( s ) ) {}
	
	Vec4( float x, float y, float z, float w ) : m_v( _mm_setr_ps( x, y, z, w ) ) {}
	
	Vec3 GetVec3() const
	{
#ifdef __GNUC__
		__attribute__ ((__aligned__ (16))) float c[4];
#else
		__declspec(align(16)) float c[4];
#endif
		_mm_store_ps( c, m_v );
		return Vec3( c[0], c[1], c[2] );
	}
	
	Vec4 SplatX() const { return Vec4( _mm_shuffle_ps( m_v, m_v, SQUISH_SSE_SPLAT( 0 ) ) ); }
	Vec4 SplatY() const { return Vec4( _mm_shuffle_ps( m_v, m_v, SQUISH_SSE_SPLAT( 1 ) ) ); }
	Vec4 SplatZ() const { return Vec4( _mm_shuffle_ps( m_v, m_v, SQUISH_SSE_SPLAT( 2 ) ) ); }
	Vec4 SplatW() const { return Vec4( _mm_shuffle_ps( m_v, m_v, SQUISH_SSE_SPLAT( 3 ) ) ); }

	Vec4& operator+=( Arg v )
	{
		m_v = _mm_add_ps( m_v, v.m_v );
		return *this;
	}
	
	Vec4& operator-=( Arg v )
	{
		m_v = _mm_sub_ps( m_v, v.m_v );
		return *this;
	}
	
	Vec4& operator*=( Arg v )
	{
		m_v = _mm_mul_ps( m_v, v.m_v );
		return *this;
	}
	
	friend Vec4 operator+( Vec4::Arg left, Vec4::Arg right  )
	{
		return Vec4( _mm_add_ps( left.m_v, right.m_v ) );
	}
	
	friend Vec4 operator-( Vec4::Arg left, Vec4::Arg right  )
	{
		return Vec4( _mm_sub_ps( left.m_v, right.m_v ) );
	}
	
	friend Vec4 operator*( Vec4::Arg left, Vec4::Arg right  )
	{
		return Vec4( _mm_mul_ps( left.m_v, right.m_v ) );
	}
	
	//! Returns a*b + c
	friend Vec4 MultiplyAdd( Vec4::Arg a, Vec4::Arg b, Vec4::Arg c )
	{
		return Vec4( _mm_add_ps( _mm_mul_ps( a.m_v, b.m_v ), c.m_v ) );
	}
	
	//! Returns -( a*b - c )
	friend Vec4 NegativeMultiplySubtract( Vec4::Arg a, Vec4::Arg b, Vec4::Arg c )
	{
		return Vec4( _mm_sub_ps( c.m_v, _mm_mul_ps( a.m_v, b.m_v ) ) );
	}
	
	friend Vec4 Reciprocal( Vec4::Arg v )
	{
		// get the reciprocal estimate
		__m128 estimate = _mm_rcp_ps( v.m_v );

		// one round of Newton-Rhaphson refinement
		__m128 diff = _mm_sub_ps( _mm_set1_ps( 1.0f ), _mm_mul_ps( estimate, v.m_v ) );
		return Vec4( _mm_add_ps( _mm_mul_ps( diff, estimate ), estimate ) );
	}
	
	friend Vec4 Min( Vec4::Arg left, Vec4::Arg right )
	{
		return Vec4( _mm_min_ps( left.m_v, right.m_v ) );
	}
	
	friend Vec4 Max( Vec4::Arg left, Vec4::Arg right )
	{
		return Vec4( _mm_max_ps( left.m_v, right.m_v ) );
	}
	
	friend Vec4 Truncate( Vec4::Arg v )
	{
#if ( SQUISH_USE_SSE == 1 )
		// convert to ints
		__m128 input = v.m_v;
		__m64 lo = _mm_cvttps_pi32( input );
		__m64 hi = _mm_cvttps_pi32( _mm_movehl_ps( input, input ) );

		// convert to floats
		__m128 part = _mm_movelh_ps( input, _mm_cvtpi32_ps( input, hi ) );
		__m128 truncated = _mm_cvtpi32_ps( part, lo );
		
		// clear out the MMX multimedia state to allow FP calls later
		_mm_empty(); 
		return Vec4( truncated );
#else
		// use SSE2 instructions
		return Vec4( _mm_cvtepi32_ps( _mm_cvttps_epi32( v.m_v ) ) );
#endif
	}
	
	friend bool CompareAnyLessThan( Vec4::Arg left, Vec4::Arg right ) 
	{
		__m128 bits = _mm_cmplt_ps( left.m_v, right.m_v );
		int value = _mm_movemask_ps( bits );
		return value != 0;
	}
	
private:
	__m128 m_v;
};

} // namespace squish

#endif // ndef SQUISH_SIMD_SSE_H

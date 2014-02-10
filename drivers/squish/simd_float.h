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
   
#ifndef SQUISH_SIMD_FLOAT_H
#define SQUISH_SIMD_FLOAT_H

#include <algorithm>

namespace squish {

#define VEC4_CONST( X ) Vec4( X )

class Vec4
{
public:
	typedef Vec4 const& Arg;

	Vec4() {}
		
	explicit Vec4( float s )
	  : m_x( s ),
		m_y( s ),
		m_z( s ),
		m_w( s )
	{
	}
	
	Vec4( float x, float y, float z, float w )
	  : m_x( x ),
		m_y( y ),
		m_z( z ),
		m_w( w )
	{
	}
	
	Vec3 GetVec3() const
	{
		return Vec3( m_x, m_y, m_z );
	}
	
	Vec4 SplatX() const { return Vec4( m_x ); }
	Vec4 SplatY() const { return Vec4( m_y ); }
	Vec4 SplatZ() const { return Vec4( m_z ); }
	Vec4 SplatW() const { return Vec4( m_w ); }

	Vec4& operator+=( Arg v )
	{
		m_x += v.m_x;
		m_y += v.m_y;
		m_z += v.m_z;
		m_w += v.m_w;
		return *this;
	}
	
	Vec4& operator-=( Arg v )
	{
		m_x -= v.m_x;
		m_y -= v.m_y;
		m_z -= v.m_z;
		m_w -= v.m_w;
		return *this;
	}
	
	Vec4& operator*=( Arg v )
	{
		m_x *= v.m_x;
		m_y *= v.m_y;
		m_z *= v.m_z;
		m_w *= v.m_w;
		return *this;
	}
	
	friend Vec4 operator+( Vec4::Arg left, Vec4::Arg right  )
	{
		Vec4 copy( left );
		return copy += right;
	}
	
	friend Vec4 operator-( Vec4::Arg left, Vec4::Arg right  )
	{
		Vec4 copy( left );
		return copy -= right;
	}
	
	friend Vec4 operator*( Vec4::Arg left, Vec4::Arg right  )
	{
		Vec4 copy( left );
		return copy *= right;
	}
	
	//! Returns a*b + c
	friend Vec4 MultiplyAdd( Vec4::Arg a, Vec4::Arg b, Vec4::Arg c )
	{
		return a*b + c;
	}
	
	//! Returns -( a*b - c )
	friend Vec4 NegativeMultiplySubtract( Vec4::Arg a, Vec4::Arg b, Vec4::Arg c )
	{
		return c - a*b;
	}
	
	friend Vec4 Reciprocal( Vec4::Arg v )
	{
		return Vec4( 
			1.0f/v.m_x, 
			1.0f/v.m_y, 
			1.0f/v.m_z, 
			1.0f/v.m_w 
		);
	}
	
	friend Vec4 Min( Vec4::Arg left, Vec4::Arg right )
	{
		return Vec4( 
			std::min( left.m_x, right.m_x ), 
			std::min( left.m_y, right.m_y ), 
			std::min( left.m_z, right.m_z ), 
			std::min( left.m_w, right.m_w ) 
		);
	}
	
	friend Vec4 Max( Vec4::Arg left, Vec4::Arg right )
	{
		return Vec4( 
			std::max( left.m_x, right.m_x ), 
			std::max( left.m_y, right.m_y ), 
			std::max( left.m_z, right.m_z ), 
			std::max( left.m_w, right.m_w ) 
		);
	}
	
	friend Vec4 Truncate( Vec4::Arg v )
	{
		return Vec4(
			v.m_x > 0.0f ? std::floor( v.m_x ) : std::ceil( v.m_x ), 
			v.m_y > 0.0f ? std::floor( v.m_y ) : std::ceil( v.m_y ), 
			v.m_z > 0.0f ? std::floor( v.m_z ) : std::ceil( v.m_z ),
			v.m_w > 0.0f ? std::floor( v.m_w ) : std::ceil( v.m_w )
		);
	}
	
	friend bool CompareAnyLessThan( Vec4::Arg left, Vec4::Arg right ) 
	{
		return left.m_x < right.m_x
			|| left.m_y < right.m_y
			|| left.m_z < right.m_z
			|| left.m_w < right.m_w;
	}
	
private:
	float m_x;
	float m_y;
	float m_z;
	float m_w;
};

} // namespace squish

#endif // ndef SQUISH_SIMD_FLOAT_H


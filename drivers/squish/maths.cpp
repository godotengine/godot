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
   
/*! @file

	The symmetric eigensystem solver algorithm is from 
	http://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf
*/

#include "maths.h"
#include <cfloat>

namespace squish {

Sym3x3 ComputeWeightedCovariance( int n, Vec3 const* points, float const* weights )
{
	// compute the centroid
	float total = 0.0f;
	Vec3 centroid( 0.0f );
	for( int i = 0; i < n; ++i )
	{
		total += weights[i];
		centroid += weights[i]*points[i];
	}
	centroid /= total;

	// accumulate the covariance matrix
	Sym3x3 covariance( 0.0f );
	for( int i = 0; i < n; ++i )
	{
		Vec3 a = points[i] - centroid;
		Vec3 b = weights[i]*a;
		
		covariance[0] += a.X()*b.X();
		covariance[1] += a.X()*b.Y();
		covariance[2] += a.X()*b.Z();
		covariance[3] += a.Y()*b.Y();
		covariance[4] += a.Y()*b.Z();
		covariance[5] += a.Z()*b.Z();
	}
	
	// return it
	return covariance;
}

static Vec3 GetMultiplicity1Evector( Sym3x3 const& matrix, float evalue )
{
	// compute M
	Sym3x3 m;
	m[0] = matrix[0] - evalue;
	m[1] = matrix[1];
	m[2] = matrix[2];
	m[3] = matrix[3] - evalue;
	m[4] = matrix[4];
	m[5] = matrix[5] - evalue;

	// compute U
	Sym3x3 u;
	u[0] = m[3]*m[5] - m[4]*m[4];
	u[1] = m[2]*m[4] - m[1]*m[5];
	u[2] = m[1]*m[4] - m[2]*m[3];
	u[3] = m[0]*m[5] - m[2]*m[2];
	u[4] = m[1]*m[2] - m[4]*m[0];
	u[5] = m[0]*m[3] - m[1]*m[1];

	// find the largest component
	float mc = std::fabs( u[0] );
	int mi = 0;
	for( int i = 1; i < 6; ++i )
	{
		float c = std::fabs( u[i] );
		if( c > mc )
		{
			mc = c;
			mi = i;
		}
	}

	// pick the column with this component
	switch( mi )
	{
	case 0:
		return Vec3( u[0], u[1], u[2] );

	case 1:
	case 3:
		return Vec3( u[1], u[3], u[4] );

	default:
		return Vec3( u[2], u[4], u[5] );
	}
}

static Vec3 GetMultiplicity2Evector( Sym3x3 const& matrix, float evalue )
{
	// compute M
	Sym3x3 m;
	m[0] = matrix[0] - evalue;
	m[1] = matrix[1];
	m[2] = matrix[2];
	m[3] = matrix[3] - evalue;
	m[4] = matrix[4];
	m[5] = matrix[5] - evalue;

	// find the largest component
	float mc = std::fabs( m[0] );
	int mi = 0;
	for( int i = 1; i < 6; ++i )
	{
		float c = std::fabs( m[i] );
		if( c > mc )
		{
			mc = c;
			mi = i;
		}
	}

	// pick the first eigenvector based on this index
	switch( mi )
	{
	case 0:
	case 1:
		return Vec3( -m[1], m[0], 0.0f );

	case 2:
		return Vec3( m[2], 0.0f, -m[0] );

	case 3:
	case 4:
		return Vec3( 0.0f, -m[4], m[3] );

	default:
		return Vec3( 0.0f, -m[5], m[4] );
	}
}

Vec3 ComputePrincipleComponent( Sym3x3 const& matrix )
{
	// compute the cubic coefficients
	float c0 = matrix[0]*matrix[3]*matrix[5] 
		+ 2.0f*matrix[1]*matrix[2]*matrix[4] 
		- matrix[0]*matrix[4]*matrix[4] 
		- matrix[3]*matrix[2]*matrix[2] 
		- matrix[5]*matrix[1]*matrix[1];
	float c1 = matrix[0]*matrix[3] + matrix[0]*matrix[5] + matrix[3]*matrix[5]
		- matrix[1]*matrix[1] - matrix[2]*matrix[2] - matrix[4]*matrix[4];
	float c2 = matrix[0] + matrix[3] + matrix[5];

	// compute the quadratic coefficients
	float a = c1 - ( 1.0f/3.0f )*c2*c2;
	float b = ( -2.0f/27.0f )*c2*c2*c2 + ( 1.0f/3.0f )*c1*c2 - c0;

	// compute the root count check
	float Q = 0.25f*b*b + ( 1.0f/27.0f )*a*a*a;

	// test the multiplicity
	if( FLT_EPSILON < Q )
	{
		// only one root, which implies we have a multiple of the identity
        return Vec3( 1.0f );
	}
	else if( Q < -FLT_EPSILON )
	{
		// three distinct roots
		float theta = std::atan2( std::sqrt( -Q ), -0.5f*b );
		float rho = std::sqrt( 0.25f*b*b - Q );

		float rt = std::pow( rho, 1.0f/3.0f );
		float ct = std::cos( theta/3.0f );
		float st = std::sin( theta/3.0f );

		float l1 = ( 1.0f/3.0f )*c2 + 2.0f*rt*ct;
		float l2 = ( 1.0f/3.0f )*c2 - rt*( ct + ( float )sqrt( 3.0f )*st );
		float l3 = ( 1.0f/3.0f )*c2 - rt*( ct - ( float )sqrt( 3.0f )*st );

		// pick the larger
		if( std::fabs( l2 ) > std::fabs( l1 ) )
			l1 = l2;
		if( std::fabs( l3 ) > std::fabs( l1 ) )
			l1 = l3;

		// get the eigenvector
		return GetMultiplicity1Evector( matrix, l1 );
	}
	else // if( -FLT_EPSILON <= Q && Q <= FLT_EPSILON )
	{
		// two roots
		float rt;
		if( b < 0.0f )
			rt = -std::pow( -0.5f*b, 1.0f/3.0f );
		else
			rt = std::pow( 0.5f*b, 1.0f/3.0f );
		
		float l1 = ( 1.0f/3.0f )*c2 + rt;		// repeated
		float l2 = ( 1.0f/3.0f )*c2 - 2.0f*rt;
		
		// get the eigenvector
		if( std::fabs( l1 ) > std::fabs( l2 ) )
			return GetMultiplicity2Evector( matrix, l1 );
		else
			return GetMultiplicity1Evector( matrix, l2 );
	}
}

} // namespace squish

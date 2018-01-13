/*
Copyright (c) 2003-2013 Gino van den Bergen / Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#ifndef	B3_MATRIX3x3_H
#define B3_MATRIX3x3_H

#include "b3Vector3.h"
#include "b3Quaternion.h"
#include <stdio.h>

#ifdef B3_USE_SSE
//const __m128 B3_ATTRIBUTE_ALIGNED16(b3v2220) = {2.0f, 2.0f, 2.0f, 0.0f};
const __m128 B3_ATTRIBUTE_ALIGNED16(b3vMPPP) = {-0.0f, +0.0f, +0.0f, +0.0f};
#endif

#if defined(B3_USE_SSE) || defined(B3_USE_NEON)
const b3SimdFloat4 B3_ATTRIBUTE_ALIGNED16(b3v1000) = {1.0f, 0.0f, 0.0f, 0.0f};
const b3SimdFloat4 B3_ATTRIBUTE_ALIGNED16(b3v0100) = {0.0f, 1.0f, 0.0f, 0.0f};
const b3SimdFloat4 B3_ATTRIBUTE_ALIGNED16(b3v0010) = {0.0f, 0.0f, 1.0f, 0.0f};
#endif

#ifdef B3_USE_DOUBLE_PRECISION
#define b3Matrix3x3Data	b3Matrix3x3DoubleData 
#else
#define b3Matrix3x3Data	b3Matrix3x3FloatData
#endif //B3_USE_DOUBLE_PRECISION


/**@brief The b3Matrix3x3 class implements a 3x3 rotation matrix, to perform linear algebra in combination with b3Quaternion, b3Transform and b3Vector3.
* Make sure to only include a pure orthogonal matrix without scaling. */
B3_ATTRIBUTE_ALIGNED16(class) b3Matrix3x3 {

	///Data storage for the matrix, each vector is a row of the matrix
	b3Vector3 m_el[3];

public:
	/** @brief No initializaion constructor */
	b3Matrix3x3 () {}

	//		explicit b3Matrix3x3(const b3Scalar *m) { setFromOpenGLSubMatrix(m); }

	/**@brief Constructor from Quaternion */
	explicit b3Matrix3x3(const b3Quaternion& q) { setRotation(q); }
	/*
	template <typename b3Scalar>
	Matrix3x3(const b3Scalar& yaw, const b3Scalar& pitch, const b3Scalar& roll)
	{ 
	setEulerYPR(yaw, pitch, roll);
	}
	*/
	/** @brief Constructor with row major formatting */
	b3Matrix3x3(const b3Scalar& xx, const b3Scalar& xy, const b3Scalar& xz,
		const b3Scalar& yx, const b3Scalar& yy, const b3Scalar& yz,
		const b3Scalar& zx, const b3Scalar& zy, const b3Scalar& zz)
	{ 
		setValue(xx, xy, xz, 
			yx, yy, yz, 
			zx, zy, zz);
	}

#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))|| defined (B3_USE_NEON)
	B3_FORCE_INLINE b3Matrix3x3 (const b3SimdFloat4 v0, const b3SimdFloat4 v1, const b3SimdFloat4 v2 ) 
	{
        m_el[0].mVec128 = v0;
        m_el[1].mVec128 = v1;
        m_el[2].mVec128 = v2;
	}

	B3_FORCE_INLINE b3Matrix3x3 (const b3Vector3& v0, const b3Vector3& v1, const b3Vector3& v2 ) 
	{
        m_el[0] = v0;
        m_el[1] = v1;
        m_el[2] = v2;
	}

	// Copy constructor
	B3_FORCE_INLINE b3Matrix3x3(const b3Matrix3x3& rhs)
	{
		m_el[0].mVec128 = rhs.m_el[0].mVec128;
		m_el[1].mVec128 = rhs.m_el[1].mVec128;
		m_el[2].mVec128 = rhs.m_el[2].mVec128;
	}

	// Assignment Operator
	B3_FORCE_INLINE b3Matrix3x3& operator=(const b3Matrix3x3& m) 
	{
		m_el[0].mVec128 = m.m_el[0].mVec128;
		m_el[1].mVec128 = m.m_el[1].mVec128;
		m_el[2].mVec128 = m.m_el[2].mVec128;
		
		return *this;
	}

#else

	/** @brief Copy constructor */
	B3_FORCE_INLINE b3Matrix3x3 (const b3Matrix3x3& other)
	{
		m_el[0] = other.m_el[0];
		m_el[1] = other.m_el[1];
		m_el[2] = other.m_el[2];
	}
    
	/** @brief Assignment Operator */
	B3_FORCE_INLINE b3Matrix3x3& operator=(const b3Matrix3x3& other)
	{
		m_el[0] = other.m_el[0];
		m_el[1] = other.m_el[1];
		m_el[2] = other.m_el[2];
		return *this;
	}

#endif

	/** @brief Get a column of the matrix as a vector 
	*  @param i Column number 0 indexed */
	B3_FORCE_INLINE b3Vector3 getColumn(int i) const
	{
		return b3MakeVector3(m_el[0][i],m_el[1][i],m_el[2][i]);
	}


	/** @brief Get a row of the matrix as a vector 
	*  @param i Row number 0 indexed */
	B3_FORCE_INLINE const b3Vector3& getRow(int i) const
	{
		b3FullAssert(0 <= i && i < 3);
		return m_el[i];
	}

	/** @brief Get a mutable reference to a row of the matrix as a vector 
	*  @param i Row number 0 indexed */
	B3_FORCE_INLINE b3Vector3&  operator[](int i)
	{ 
		b3FullAssert(0 <= i && i < 3);
		return m_el[i]; 
	}

	/** @brief Get a const reference to a row of the matrix as a vector 
	*  @param i Row number 0 indexed */
	B3_FORCE_INLINE const b3Vector3& operator[](int i) const
	{
		b3FullAssert(0 <= i && i < 3);
		return m_el[i]; 
	}

	/** @brief Multiply by the target matrix on the right
	*  @param m Rotation matrix to be applied 
	* Equivilant to this = this * m */
	b3Matrix3x3& operator*=(const b3Matrix3x3& m); 

	/** @brief Adds by the target matrix on the right
	*  @param m matrix to be applied 
	* Equivilant to this = this + m */
	b3Matrix3x3& operator+=(const b3Matrix3x3& m); 

	/** @brief Substractss by the target matrix on the right
	*  @param m matrix to be applied 
	* Equivilant to this = this - m */
	b3Matrix3x3& operator-=(const b3Matrix3x3& m); 

	/** @brief Set from the rotational part of a 4x4 OpenGL matrix
	*  @param m A pointer to the beginning of the array of scalars*/
	void setFromOpenGLSubMatrix(const b3Scalar *m)
	{
		m_el[0].setValue(m[0],m[4],m[8]);
		m_el[1].setValue(m[1],m[5],m[9]);
		m_el[2].setValue(m[2],m[6],m[10]);

	}
	/** @brief Set the values of the matrix explicitly (row major)
	*  @param xx Top left
	*  @param xy Top Middle
	*  @param xz Top Right
	*  @param yx Middle Left
	*  @param yy Middle Middle
	*  @param yz Middle Right
	*  @param zx Bottom Left
	*  @param zy Bottom Middle
	*  @param zz Bottom Right*/
	void setValue(const b3Scalar& xx, const b3Scalar& xy, const b3Scalar& xz, 
		const b3Scalar& yx, const b3Scalar& yy, const b3Scalar& yz, 
		const b3Scalar& zx, const b3Scalar& zy, const b3Scalar& zz)
	{
		m_el[0].setValue(xx,xy,xz);
		m_el[1].setValue(yx,yy,yz);
		m_el[2].setValue(zx,zy,zz);
	}

	/** @brief Set the matrix from a quaternion
	*  @param q The Quaternion to match */  
	void setRotation(const b3Quaternion& q) 
	{
		b3Scalar d = q.length2();
		b3FullAssert(d != b3Scalar(0.0));
		b3Scalar s = b3Scalar(2.0) / d;
    
    #if defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE)
        __m128	vs, Q = q.get128();
		__m128i Qi = b3CastfTo128i(Q);
        __m128	Y, Z;
        __m128	V1, V2, V3;
        __m128	V11, V21, V31;
        __m128	NQ = _mm_xor_ps(Q, b3vMzeroMask);
		__m128i NQi = b3CastfTo128i(NQ);
        
        V1 = b3CastiTo128f(_mm_shuffle_epi32 (Qi, B3_SHUFFLE(1,0,2,3)));	// Y X Z W
		V2 = _mm_shuffle_ps(NQ, Q, B3_SHUFFLE(0,0,1,3));     // -X -X  Y  W
        V3 = b3CastiTo128f(_mm_shuffle_epi32 (Qi, B3_SHUFFLE(2,1,0,3)));	// Z Y X W
        V1 = _mm_xor_ps(V1, b3vMPPP);	//	change the sign of the first element
			
        V11	= b3CastiTo128f(_mm_shuffle_epi32 (Qi, B3_SHUFFLE(1,1,0,3)));	// Y Y X W
		V21 = _mm_unpackhi_ps(Q, Q);                    //  Z  Z  W  W
		V31 = _mm_shuffle_ps(Q, NQ, B3_SHUFFLE(0,2,0,3));	//  X  Z -X -W

		V2 = V2 * V1;	//
		V1 = V1 * V11;	//
		V3 = V3 * V31;	//

        V11 = _mm_shuffle_ps(NQ, Q, B3_SHUFFLE(2,3,1,3));	//	-Z -W  Y  W
		V11 = V11 * V21;	//
        V21 = _mm_xor_ps(V21, b3vMPPP);	//	change the sign of the first element
		V31 = _mm_shuffle_ps(Q, NQ, B3_SHUFFLE(3,3,1,3));	//	 W  W -Y -W
        V31 = _mm_xor_ps(V31, b3vMPPP);	//	change the sign of the first element
		Y = b3CastiTo128f(_mm_shuffle_epi32 (NQi, B3_SHUFFLE(3,2,0,3)));	// -W -Z -X -W
		Z = b3CastiTo128f(_mm_shuffle_epi32 (Qi, B3_SHUFFLE(1,0,1,3)));	//  Y  X  Y  W

		vs = _mm_load_ss(&s);
		V21 = V21 * Y;
		V31 = V31 * Z;

		V1 = V1 + V11;
        V2 = V2 + V21;
        V3 = V3 + V31;

        vs = b3_splat3_ps(vs, 0);
            //	s ready
        V1 = V1 * vs;
        V2 = V2 * vs;
        V3 = V3 * vs;
        
        V1 = V1 + b3v1000;
        V2 = V2 + b3v0100;
        V3 = V3 + b3v0010;
        
        m_el[0] = b3MakeVector3(V1); 
        m_el[1] = b3MakeVector3(V2);
        m_el[2] = b3MakeVector3(V3);
    #else    
		b3Scalar xs = q.getX() * s,   ys = q.getY() * s,   zs = q.getZ() * s;
		b3Scalar wx = q.getW() * xs,  wy = q.getW() * ys,  wz = q.getW() * zs;
		b3Scalar xx = q.getX() * xs,  xy = q.getX() * ys,  xz = q.getX() * zs;
		b3Scalar yy = q.getY() * ys,  yz = q.getY() * zs,  zz = q.getZ() * zs;
		setValue(
            b3Scalar(1.0) - (yy + zz), xy - wz, xz + wy,
			xy + wz, b3Scalar(1.0) - (xx + zz), yz - wx,
			xz - wy, yz + wx, b3Scalar(1.0) - (xx + yy));
	#endif
    }


	/** @brief Set the matrix from euler angles using YPR around YXZ respectively
	*  @param yaw Yaw about Y axis
	*  @param pitch Pitch about X axis
	*  @param roll Roll about Z axis 
	*/
	void setEulerYPR(const b3Scalar& yaw, const b3Scalar& pitch, const b3Scalar& roll) 
	{
		setEulerZYX(roll, pitch, yaw);
	}

	/** @brief Set the matrix from euler angles YPR around ZYX axes
	* @param eulerX Roll about X axis
	* @param eulerY Pitch around Y axis
	* @param eulerZ Yaw aboud Z axis
	* 
	* These angles are used to produce a rotation matrix. The euler
	* angles are applied in ZYX order. I.e a vector is first rotated 
	* about X then Y and then Z
	**/
	void setEulerZYX(b3Scalar eulerX,b3Scalar eulerY,b3Scalar eulerZ) { 
		///@todo proposed to reverse this since it's labeled zyx but takes arguments xyz and it will match all other parts of the code
		b3Scalar ci ( b3Cos(eulerX)); 
		b3Scalar cj ( b3Cos(eulerY)); 
		b3Scalar ch ( b3Cos(eulerZ)); 
		b3Scalar si ( b3Sin(eulerX)); 
		b3Scalar sj ( b3Sin(eulerY)); 
		b3Scalar sh ( b3Sin(eulerZ)); 
		b3Scalar cc = ci * ch; 
		b3Scalar cs = ci * sh; 
		b3Scalar sc = si * ch; 
		b3Scalar ss = si * sh;

		setValue(cj * ch, sj * sc - cs, sj * cc + ss,
			cj * sh, sj * ss + cc, sj * cs - sc, 
			-sj,      cj * si,      cj * ci);
	}

	/**@brief Set the matrix to the identity */
	void setIdentity()
	{ 
#if (defined(B3_USE_SSE_IN_API)&& defined (B3_USE_SSE)) || defined(B3_USE_NEON)
			m_el[0] = b3MakeVector3(b3v1000); 
			m_el[1] = b3MakeVector3(b3v0100);
			m_el[2] = b3MakeVector3(b3v0010);
#else
		setValue(b3Scalar(1.0), b3Scalar(0.0), b3Scalar(0.0), 
			b3Scalar(0.0), b3Scalar(1.0), b3Scalar(0.0), 
			b3Scalar(0.0), b3Scalar(0.0), b3Scalar(1.0)); 
#endif
	}

	static const b3Matrix3x3&	getIdentity()
	{
#if (defined(B3_USE_SSE_IN_API)&& defined (B3_USE_SSE)) || defined(B3_USE_NEON)
        static const b3Matrix3x3 
        identityMatrix(b3v1000, b3v0100, b3v0010);
#else
		static const b3Matrix3x3 
        identityMatrix(
            b3Scalar(1.0), b3Scalar(0.0), b3Scalar(0.0), 
			b3Scalar(0.0), b3Scalar(1.0), b3Scalar(0.0), 
			b3Scalar(0.0), b3Scalar(0.0), b3Scalar(1.0));
#endif
		return identityMatrix;
	}

	/**@brief Fill the rotational part of an OpenGL matrix and clear the shear/perspective
	* @param m The array to be filled */
	void getOpenGLSubMatrix(b3Scalar *m) const 
	{
#if defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE)
        __m128 v0 = m_el[0].mVec128;
        __m128 v1 = m_el[1].mVec128;
        __m128 v2 = m_el[2].mVec128;    //  x2 y2 z2 w2
        __m128 *vm = (__m128 *)m;
        __m128 vT;
        
        v2 = _mm_and_ps(v2, b3vFFF0fMask);  //  x2 y2 z2 0
        
        vT = _mm_unpackhi_ps(v0, v1);	//	z0 z1 * *
        v0 = _mm_unpacklo_ps(v0, v1);	//	x0 x1 y0 y1

        v1 = _mm_shuffle_ps(v0, v2, B3_SHUFFLE(2, 3, 1, 3) );	// y0 y1 y2 0
        v0 = _mm_shuffle_ps(v0, v2, B3_SHUFFLE(0, 1, 0, 3) );	// x0 x1 x2 0
        v2 = b3CastdTo128f(_mm_move_sd(b3CastfTo128d(v2), b3CastfTo128d(vT)));	// z0 z1 z2 0

        vm[0] = v0;
        vm[1] = v1;
        vm[2] = v2;
#elif defined(B3_USE_NEON)
        // note: zeros the w channel. We can preserve it at the cost of two more vtrn instructions.
        static const uint32x2_t zMask = (const uint32x2_t) {-1, 0 };
        float32x4_t *vm = (float32x4_t *)m;
        float32x4x2_t top = vtrnq_f32( m_el[0].mVec128, m_el[1].mVec128 );  // {x0 x1 z0 z1}, {y0 y1 w0 w1}
        float32x2x2_t bl = vtrn_f32( vget_low_f32(m_el[2].mVec128), vdup_n_f32(0.0f) );       // {x2  0 }, {y2 0}
        float32x4_t v0 = vcombine_f32( vget_low_f32(top.val[0]), bl.val[0] );
        float32x4_t v1 = vcombine_f32( vget_low_f32(top.val[1]), bl.val[1] );
        float32x2_t q = (float32x2_t) vand_u32( (uint32x2_t) vget_high_f32( m_el[2].mVec128), zMask );
        float32x4_t v2 = vcombine_f32( vget_high_f32(top.val[0]), q );       // z0 z1 z2  0

        vm[0] = v0;
        vm[1] = v1;
        vm[2] = v2;
#else
		m[0]  = b3Scalar(m_el[0].getX()); 
		m[1]  = b3Scalar(m_el[1].getX());
		m[2]  = b3Scalar(m_el[2].getX());
		m[3]  = b3Scalar(0.0); 
		m[4]  = b3Scalar(m_el[0].getY());
		m[5]  = b3Scalar(m_el[1].getY());
		m[6]  = b3Scalar(m_el[2].getY());
		m[7]  = b3Scalar(0.0); 
		m[8]  = b3Scalar(m_el[0].getZ()); 
		m[9]  = b3Scalar(m_el[1].getZ());
		m[10] = b3Scalar(m_el[2].getZ());
		m[11] = b3Scalar(0.0); 
#endif
	}

	/**@brief Get the matrix represented as a quaternion 
	* @param q The quaternion which will be set */
	void getRotation(b3Quaternion& q) const
	{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))|| defined (B3_USE_NEON)
        b3Scalar trace = m_el[0].getX() + m_el[1].getY() + m_el[2].getZ();
        b3Scalar s, x;
        
        union {
            b3SimdFloat4 vec;
            b3Scalar f[4];
        } temp;
        
        if (trace > b3Scalar(0.0)) 
        {
            x = trace + b3Scalar(1.0);

            temp.f[0]=m_el[2].getY() - m_el[1].getZ();
            temp.f[1]=m_el[0].getZ() - m_el[2].getX();
            temp.f[2]=m_el[1].getX() - m_el[0].getY();
            temp.f[3]=x;
            //temp.f[3]= s * b3Scalar(0.5);
        } 
        else 
        {
            int i, j, k;
            if(m_el[0].getX() < m_el[1].getY()) 
            { 
                if( m_el[1].getY() < m_el[2].getZ() )
                    { i = 2; j = 0; k = 1; }
                else
                    { i = 1; j = 2; k = 0; }
            }
            else
            {
                if( m_el[0].getX() < m_el[2].getZ())
                    { i = 2; j = 0; k = 1; }
                else
                    { i = 0; j = 1; k = 2; }
            }

            x = m_el[i][i] - m_el[j][j] - m_el[k][k] + b3Scalar(1.0);

            temp.f[3] = (m_el[k][j] - m_el[j][k]);
            temp.f[j] = (m_el[j][i] + m_el[i][j]);
            temp.f[k] = (m_el[k][i] + m_el[i][k]);
            temp.f[i] = x;
            //temp.f[i] = s * b3Scalar(0.5);
        }

        s = b3Sqrt(x);
        q.set128(temp.vec);
        s = b3Scalar(0.5) / s;

        q *= s;
#else    
		b3Scalar trace = m_el[0].getX() + m_el[1].getY() + m_el[2].getZ();

		b3Scalar temp[4];

		if (trace > b3Scalar(0.0)) 
		{
			b3Scalar s = b3Sqrt(trace + b3Scalar(1.0));
			temp[3]=(s * b3Scalar(0.5));
			s = b3Scalar(0.5) / s;

			temp[0]=((m_el[2].getY() - m_el[1].getZ()) * s);
			temp[1]=((m_el[0].getZ() - m_el[2].getX()) * s);
			temp[2]=((m_el[1].getX() - m_el[0].getY()) * s);
		} 
		else 
		{
			int i = m_el[0].getX() < m_el[1].getY() ? 
				(m_el[1].getY() < m_el[2].getZ() ? 2 : 1) :
				(m_el[0].getX() < m_el[2].getZ() ? 2 : 0); 
			int j = (i + 1) % 3;  
			int k = (i + 2) % 3;

			b3Scalar s = b3Sqrt(m_el[i][i] - m_el[j][j] - m_el[k][k] + b3Scalar(1.0));
			temp[i] = s * b3Scalar(0.5);
			s = b3Scalar(0.5) / s;

			temp[3] = (m_el[k][j] - m_el[j][k]) * s;
			temp[j] = (m_el[j][i] + m_el[i][j]) * s;
			temp[k] = (m_el[k][i] + m_el[i][k]) * s;
		}
		q.setValue(temp[0],temp[1],temp[2],temp[3]);
#endif
	}

	/**@brief Get the matrix represented as euler angles around YXZ, roundtrip with setEulerYPR
	* @param yaw Yaw around Y axis
	* @param pitch Pitch around X axis
	* @param roll around Z axis */	
	void getEulerYPR(b3Scalar& yaw, b3Scalar& pitch, b3Scalar& roll) const
	{

		// first use the normal calculus
		yaw = b3Scalar(b3Atan2(m_el[1].getX(), m_el[0].getX()));
		pitch = b3Scalar(b3Asin(-m_el[2].getX()));
		roll = b3Scalar(b3Atan2(m_el[2].getY(), m_el[2].getZ()));

		// on pitch = +/-HalfPI
		if (b3Fabs(pitch)==B3_HALF_PI)
		{
			if (yaw>0)
				yaw-=B3_PI;
			else
				yaw+=B3_PI;

			if (roll>0)
				roll-=B3_PI;
			else
				roll+=B3_PI;
		}
	};


	/**@brief Get the matrix represented as euler angles around ZYX
	* @param yaw Yaw around X axis
	* @param pitch Pitch around Y axis
	* @param roll around X axis 
	* @param solution_number Which solution of two possible solutions ( 1 or 2) are possible values*/	
	void getEulerZYX(b3Scalar& yaw, b3Scalar& pitch, b3Scalar& roll, unsigned int solution_number = 1) const
	{
		struct Euler
		{
			b3Scalar yaw;
			b3Scalar pitch;
			b3Scalar roll;
		};

		Euler euler_out;
		Euler euler_out2; //second solution
		//get the pointer to the raw data

		// Check that pitch is not at a singularity
		if (b3Fabs(m_el[2].getX()) >= 1)
		{
			euler_out.yaw = 0;
			euler_out2.yaw = 0;

			// From difference of angles formula
			b3Scalar delta = b3Atan2(m_el[0].getX(),m_el[0].getZ());
			if (m_el[2].getX() > 0)  //gimbal locked up
			{
				euler_out.pitch = B3_PI / b3Scalar(2.0);
				euler_out2.pitch = B3_PI / b3Scalar(2.0);
				euler_out.roll = euler_out.pitch + delta;
				euler_out2.roll = euler_out.pitch + delta;
			}
			else // gimbal locked down
			{
				euler_out.pitch = -B3_PI / b3Scalar(2.0);
				euler_out2.pitch = -B3_PI / b3Scalar(2.0);
				euler_out.roll = -euler_out.pitch + delta;
				euler_out2.roll = -euler_out.pitch + delta;
			}
		}
		else
		{
			euler_out.pitch = - b3Asin(m_el[2].getX());
			euler_out2.pitch = B3_PI - euler_out.pitch;

			euler_out.roll = b3Atan2(m_el[2].getY()/b3Cos(euler_out.pitch), 
				m_el[2].getZ()/b3Cos(euler_out.pitch));
			euler_out2.roll = b3Atan2(m_el[2].getY()/b3Cos(euler_out2.pitch), 
				m_el[2].getZ()/b3Cos(euler_out2.pitch));

			euler_out.yaw = b3Atan2(m_el[1].getX()/b3Cos(euler_out.pitch), 
				m_el[0].getX()/b3Cos(euler_out.pitch));
			euler_out2.yaw = b3Atan2(m_el[1].getX()/b3Cos(euler_out2.pitch), 
				m_el[0].getX()/b3Cos(euler_out2.pitch));
		}

		if (solution_number == 1)
		{ 
			yaw = euler_out.yaw; 
			pitch = euler_out.pitch;
			roll = euler_out.roll;
		}
		else
		{ 
			yaw = euler_out2.yaw; 
			pitch = euler_out2.pitch;
			roll = euler_out2.roll;
		}
	}

	/**@brief Create a scaled copy of the matrix 
	* @param s Scaling vector The elements of the vector will scale each column */

	b3Matrix3x3 scaled(const b3Vector3& s) const
	{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))|| defined (B3_USE_NEON)
		return b3Matrix3x3(m_el[0] * s, m_el[1] * s, m_el[2] * s);
#else		
		return b3Matrix3x3(
            m_el[0].getX() * s.getX(), m_el[0].getY() * s.getY(), m_el[0].getZ() * s.getZ(),
			m_el[1].getX() * s.getX(), m_el[1].getY() * s.getY(), m_el[1].getZ() * s.getZ(),
			m_el[2].getX() * s.getX(), m_el[2].getY() * s.getY(), m_el[2].getZ() * s.getZ());
#endif
	}

	/**@brief Return the determinant of the matrix */
	b3Scalar            determinant() const;
	/**@brief Return the adjoint of the matrix */
	b3Matrix3x3 adjoint() const;
	/**@brief Return the matrix with all values non negative */
	b3Matrix3x3 absolute() const;
	/**@brief Return the transpose of the matrix */
	b3Matrix3x3 transpose() const;
	/**@brief Return the inverse of the matrix */
	b3Matrix3x3 inverse() const; 

	b3Matrix3x3 transposeTimes(const b3Matrix3x3& m) const;
	b3Matrix3x3 timesTranspose(const b3Matrix3x3& m) const;

	B3_FORCE_INLINE b3Scalar tdotx(const b3Vector3& v) const 
	{
		return m_el[0].getX() * v.getX() + m_el[1].getX() * v.getY() + m_el[2].getX() * v.getZ();
	}
	B3_FORCE_INLINE b3Scalar tdoty(const b3Vector3& v) const 
	{
		return m_el[0].getY() * v.getX() + m_el[1].getY() * v.getY() + m_el[2].getY() * v.getZ();
	}
	B3_FORCE_INLINE b3Scalar tdotz(const b3Vector3& v) const 
	{
		return m_el[0].getZ() * v.getX() + m_el[1].getZ() * v.getY() + m_el[2].getZ() * v.getZ();
	}


	/**@brief diagonalizes this matrix by the Jacobi method.
	* @param rot stores the rotation from the coordinate system in which the matrix is diagonal to the original
	* coordinate system, i.e., old_this = rot * new_this * rot^T. 
	* @param threshold See iteration
	* @param iteration The iteration stops when all off-diagonal elements are less than the threshold multiplied 
	* by the sum of the absolute values of the diagonal, or when maxSteps have been executed. 
	* 
	* Note that this matrix is assumed to be symmetric. 
	*/
	void diagonalize(b3Matrix3x3& rot, b3Scalar threshold, int maxSteps)
	{
		rot.setIdentity();
		for (int step = maxSteps; step > 0; step--)
		{
			// find off-diagonal element [p][q] with largest magnitude
			int p = 0;
			int q = 1;
			int r = 2;
			b3Scalar max = b3Fabs(m_el[0][1]);
			b3Scalar v = b3Fabs(m_el[0][2]);
			if (v > max)
			{
				q = 2;
				r = 1;
				max = v;
			}
			v = b3Fabs(m_el[1][2]);
			if (v > max)
			{
				p = 1;
				q = 2;
				r = 0;
				max = v;
			}

			b3Scalar t = threshold * (b3Fabs(m_el[0][0]) + b3Fabs(m_el[1][1]) + b3Fabs(m_el[2][2]));
			if (max <= t)
			{
				if (max <= B3_EPSILON * t)
				{
					return;
				}
				step = 1;
			}

			// compute Jacobi rotation J which leads to a zero for element [p][q] 
			b3Scalar mpq = m_el[p][q];
			b3Scalar theta = (m_el[q][q] - m_el[p][p]) / (2 * mpq);
			b3Scalar theta2 = theta * theta;
			b3Scalar cos;
			b3Scalar sin;
			if (theta2 * theta2 < b3Scalar(10 / B3_EPSILON))
			{
				t = (theta >= 0) ? 1 / (theta + b3Sqrt(1 + theta2))
					: 1 / (theta - b3Sqrt(1 + theta2));
				cos = 1 / b3Sqrt(1 + t * t);
				sin = cos * t;
			}
			else
			{
				// approximation for large theta-value, i.e., a nearly diagonal matrix
				t = 1 / (theta * (2 + b3Scalar(0.5) / theta2));
				cos = 1 - b3Scalar(0.5) * t * t;
				sin = cos * t;
			}

			// apply rotation to matrix (this = J^T * this * J)
			m_el[p][q] = m_el[q][p] = 0;
			m_el[p][p] -= t * mpq;
			m_el[q][q] += t * mpq;
			b3Scalar mrp = m_el[r][p];
			b3Scalar mrq = m_el[r][q];
			m_el[r][p] = m_el[p][r] = cos * mrp - sin * mrq;
			m_el[r][q] = m_el[q][r] = cos * mrq + sin * mrp;

			// apply rotation to rot (rot = rot * J)
			for (int i = 0; i < 3; i++)
			{
				b3Vector3& row = rot[i];
				mrp = row[p];
				mrq = row[q];
				row[p] = cos * mrp - sin * mrq;
				row[q] = cos * mrq + sin * mrp;
			}
		}
	}




	/**@brief Calculate the matrix cofactor 
	* @param r1 The first row to use for calculating the cofactor
	* @param c1 The first column to use for calculating the cofactor
	* @param r1 The second row to use for calculating the cofactor
	* @param c1 The second column to use for calculating the cofactor
	* See http://en.wikipedia.org/wiki/Cofactor_(linear_algebra) for more details
	*/
	b3Scalar cofac(int r1, int c1, int r2, int c2) const 
	{
		return m_el[r1][c1] * m_el[r2][c2] - m_el[r1][c2] * m_el[r2][c1];
	}

	void	serialize(struct	b3Matrix3x3Data& dataOut) const;

	void	serializeFloat(struct	b3Matrix3x3FloatData& dataOut) const;

	void	deSerialize(const struct	b3Matrix3x3Data& dataIn);

	void	deSerializeFloat(const struct	b3Matrix3x3FloatData& dataIn);

	void	deSerializeDouble(const struct	b3Matrix3x3DoubleData& dataIn);

};


B3_FORCE_INLINE b3Matrix3x3& 
b3Matrix3x3::operator*=(const b3Matrix3x3& m)
{
#if defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE)
    __m128 rv00, rv01, rv02;
    __m128 rv10, rv11, rv12;
    __m128 rv20, rv21, rv22;
    __m128 mv0, mv1, mv2;

    rv02 = m_el[0].mVec128;
    rv12 = m_el[1].mVec128;
    rv22 = m_el[2].mVec128;

    mv0 = _mm_and_ps(m[0].mVec128, b3vFFF0fMask); 
    mv1 = _mm_and_ps(m[1].mVec128, b3vFFF0fMask); 
    mv2 = _mm_and_ps(m[2].mVec128, b3vFFF0fMask); 
    
    // rv0
    rv00 = b3_splat_ps(rv02, 0);
    rv01 = b3_splat_ps(rv02, 1);
    rv02 = b3_splat_ps(rv02, 2);
    
    rv00 = _mm_mul_ps(rv00, mv0);
    rv01 = _mm_mul_ps(rv01, mv1);
    rv02 = _mm_mul_ps(rv02, mv2);
    
    // rv1
    rv10 = b3_splat_ps(rv12, 0);
    rv11 = b3_splat_ps(rv12, 1);
    rv12 = b3_splat_ps(rv12, 2);
    
    rv10 = _mm_mul_ps(rv10, mv0);
    rv11 = _mm_mul_ps(rv11, mv1);
    rv12 = _mm_mul_ps(rv12, mv2);
    
    // rv2
    rv20 = b3_splat_ps(rv22, 0);
    rv21 = b3_splat_ps(rv22, 1);
    rv22 = b3_splat_ps(rv22, 2);
    
    rv20 = _mm_mul_ps(rv20, mv0);
    rv21 = _mm_mul_ps(rv21, mv1);
    rv22 = _mm_mul_ps(rv22, mv2);

    rv00 = _mm_add_ps(rv00, rv01);
    rv10 = _mm_add_ps(rv10, rv11);
    rv20 = _mm_add_ps(rv20, rv21);

    m_el[0].mVec128 = _mm_add_ps(rv00, rv02);
    m_el[1].mVec128 = _mm_add_ps(rv10, rv12);
    m_el[2].mVec128 = _mm_add_ps(rv20, rv22);

#elif defined(B3_USE_NEON)

    float32x4_t rv0, rv1, rv2;
    float32x4_t v0, v1, v2;
    float32x4_t mv0, mv1, mv2;

    v0 = m_el[0].mVec128;
    v1 = m_el[1].mVec128;
    v2 = m_el[2].mVec128;

    mv0 = (float32x4_t) vandq_s32((int32x4_t)m[0].mVec128, b3vFFF0Mask); 
    mv1 = (float32x4_t) vandq_s32((int32x4_t)m[1].mVec128, b3vFFF0Mask); 
    mv2 = (float32x4_t) vandq_s32((int32x4_t)m[2].mVec128, b3vFFF0Mask); 
    
    rv0 = vmulq_lane_f32(mv0, vget_low_f32(v0), 0);
    rv1 = vmulq_lane_f32(mv0, vget_low_f32(v1), 0);
    rv2 = vmulq_lane_f32(mv0, vget_low_f32(v2), 0);
    
    rv0 = vmlaq_lane_f32(rv0, mv1, vget_low_f32(v0), 1);
    rv1 = vmlaq_lane_f32(rv1, mv1, vget_low_f32(v1), 1);
    rv2 = vmlaq_lane_f32(rv2, mv1, vget_low_f32(v2), 1);
    
    rv0 = vmlaq_lane_f32(rv0, mv2, vget_high_f32(v0), 0);
    rv1 = vmlaq_lane_f32(rv1, mv2, vget_high_f32(v1), 0);
    rv2 = vmlaq_lane_f32(rv2, mv2, vget_high_f32(v2), 0);

    m_el[0].mVec128 = rv0;
    m_el[1].mVec128 = rv1;
    m_el[2].mVec128 = rv2;
#else    
	setValue(
        m.tdotx(m_el[0]), m.tdoty(m_el[0]), m.tdotz(m_el[0]),
		m.tdotx(m_el[1]), m.tdoty(m_el[1]), m.tdotz(m_el[1]),
		m.tdotx(m_el[2]), m.tdoty(m_el[2]), m.tdotz(m_el[2]));
#endif
	return *this;
}

B3_FORCE_INLINE b3Matrix3x3& 
b3Matrix3x3::operator+=(const b3Matrix3x3& m)
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))|| defined (B3_USE_NEON)
    m_el[0].mVec128 = m_el[0].mVec128 + m.m_el[0].mVec128;
    m_el[1].mVec128 = m_el[1].mVec128 + m.m_el[1].mVec128;
    m_el[2].mVec128 = m_el[2].mVec128 + m.m_el[2].mVec128;
#else
	setValue(
		m_el[0][0]+m.m_el[0][0], 
		m_el[0][1]+m.m_el[0][1],
		m_el[0][2]+m.m_el[0][2],
		m_el[1][0]+m.m_el[1][0], 
		m_el[1][1]+m.m_el[1][1],
		m_el[1][2]+m.m_el[1][2],
		m_el[2][0]+m.m_el[2][0], 
		m_el[2][1]+m.m_el[2][1],
		m_el[2][2]+m.m_el[2][2]);
#endif
	return *this;
}

B3_FORCE_INLINE b3Matrix3x3
operator*(const b3Matrix3x3& m, const b3Scalar & k)
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))
    __m128 vk = b3_splat_ps(_mm_load_ss((float *)&k), 0x80);
    return b3Matrix3x3(
                _mm_mul_ps(m[0].mVec128, vk), 
                _mm_mul_ps(m[1].mVec128, vk), 
                _mm_mul_ps(m[2].mVec128, vk)); 
#elif defined(B3_USE_NEON)
    return b3Matrix3x3(
                vmulq_n_f32(m[0].mVec128, k),
                vmulq_n_f32(m[1].mVec128, k),
                vmulq_n_f32(m[2].mVec128, k)); 
#else
	return b3Matrix3x3(
		m[0].getX()*k,m[0].getY()*k,m[0].getZ()*k,
		m[1].getX()*k,m[1].getY()*k,m[1].getZ()*k,
		m[2].getX()*k,m[2].getY()*k,m[2].getZ()*k);
#endif
}

B3_FORCE_INLINE b3Matrix3x3 
operator+(const b3Matrix3x3& m1, const b3Matrix3x3& m2)
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))|| defined (B3_USE_NEON)
	return b3Matrix3x3(
        m1[0].mVec128 + m2[0].mVec128,
        m1[1].mVec128 + m2[1].mVec128,
        m1[2].mVec128 + m2[2].mVec128);
#else
	return b3Matrix3x3(
        m1[0][0]+m2[0][0], 
        m1[0][1]+m2[0][1],
        m1[0][2]+m2[0][2],
        
        m1[1][0]+m2[1][0], 
        m1[1][1]+m2[1][1],
        m1[1][2]+m2[1][2],
        
        m1[2][0]+m2[2][0], 
        m1[2][1]+m2[2][1],
        m1[2][2]+m2[2][2]);
#endif    
}

B3_FORCE_INLINE b3Matrix3x3 
operator-(const b3Matrix3x3& m1, const b3Matrix3x3& m2)
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))|| defined (B3_USE_NEON)
	return b3Matrix3x3(
        m1[0].mVec128 - m2[0].mVec128,
        m1[1].mVec128 - m2[1].mVec128,
        m1[2].mVec128 - m2[2].mVec128);
#else
	return b3Matrix3x3(
        m1[0][0]-m2[0][0], 
        m1[0][1]-m2[0][1],
        m1[0][2]-m2[0][2],
        
        m1[1][0]-m2[1][0], 
        m1[1][1]-m2[1][1],
        m1[1][2]-m2[1][2],
        
        m1[2][0]-m2[2][0], 
        m1[2][1]-m2[2][1],
        m1[2][2]-m2[2][2]);
#endif
}


B3_FORCE_INLINE b3Matrix3x3& 
b3Matrix3x3::operator-=(const b3Matrix3x3& m)
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))|| defined (B3_USE_NEON)
    m_el[0].mVec128 = m_el[0].mVec128 - m.m_el[0].mVec128;
    m_el[1].mVec128 = m_el[1].mVec128 - m.m_el[1].mVec128;
    m_el[2].mVec128 = m_el[2].mVec128 - m.m_el[2].mVec128;
#else
	setValue(
	m_el[0][0]-m.m_el[0][0], 
	m_el[0][1]-m.m_el[0][1],
	m_el[0][2]-m.m_el[0][2],
	m_el[1][0]-m.m_el[1][0], 
	m_el[1][1]-m.m_el[1][1],
	m_el[1][2]-m.m_el[1][2],
	m_el[2][0]-m.m_el[2][0], 
	m_el[2][1]-m.m_el[2][1],
	m_el[2][2]-m.m_el[2][2]);
#endif
	return *this;
}


B3_FORCE_INLINE b3Scalar 
b3Matrix3x3::determinant() const
{ 
	return b3Triple((*this)[0], (*this)[1], (*this)[2]);
}


B3_FORCE_INLINE b3Matrix3x3 
b3Matrix3x3::absolute() const
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))
    return b3Matrix3x3(
            _mm_and_ps(m_el[0].mVec128, b3vAbsfMask),
            _mm_and_ps(m_el[1].mVec128, b3vAbsfMask),
            _mm_and_ps(m_el[2].mVec128, b3vAbsfMask));
#elif defined(B3_USE_NEON)
    return b3Matrix3x3(
            (float32x4_t)vandq_s32((int32x4_t)m_el[0].mVec128, b3v3AbsMask),
            (float32x4_t)vandq_s32((int32x4_t)m_el[1].mVec128, b3v3AbsMask),
            (float32x4_t)vandq_s32((int32x4_t)m_el[2].mVec128, b3v3AbsMask));
#else	
	return b3Matrix3x3(
            b3Fabs(m_el[0].getX()), b3Fabs(m_el[0].getY()), b3Fabs(m_el[0].getZ()),
            b3Fabs(m_el[1].getX()), b3Fabs(m_el[1].getY()), b3Fabs(m_el[1].getZ()),
            b3Fabs(m_el[2].getX()), b3Fabs(m_el[2].getY()), b3Fabs(m_el[2].getZ()));
#endif
}

B3_FORCE_INLINE b3Matrix3x3 
b3Matrix3x3::transpose() const 
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))
    __m128 v0 = m_el[0].mVec128;
    __m128 v1 = m_el[1].mVec128;
    __m128 v2 = m_el[2].mVec128;    //  x2 y2 z2 w2
    __m128 vT;
    
    v2 = _mm_and_ps(v2, b3vFFF0fMask);  //  x2 y2 z2 0
    
    vT = _mm_unpackhi_ps(v0, v1);	//	z0 z1 * *
    v0 = _mm_unpacklo_ps(v0, v1);	//	x0 x1 y0 y1

    v1 = _mm_shuffle_ps(v0, v2, B3_SHUFFLE(2, 3, 1, 3) );	// y0 y1 y2 0
    v0 = _mm_shuffle_ps(v0, v2, B3_SHUFFLE(0, 1, 0, 3) );	// x0 x1 x2 0
    v2 = b3CastdTo128f(_mm_move_sd(b3CastfTo128d(v2), b3CastfTo128d(vT)));	// z0 z1 z2 0


    return b3Matrix3x3( v0, v1, v2 );
#elif defined(B3_USE_NEON)
    // note: zeros the w channel. We can preserve it at the cost of two more vtrn instructions.
    static const uint32x2_t zMask = (const uint32x2_t) {-1, 0 };
    float32x4x2_t top = vtrnq_f32( m_el[0].mVec128, m_el[1].mVec128 );  // {x0 x1 z0 z1}, {y0 y1 w0 w1}
    float32x2x2_t bl = vtrn_f32( vget_low_f32(m_el[2].mVec128), vdup_n_f32(0.0f) );       // {x2  0 }, {y2 0}
    float32x4_t v0 = vcombine_f32( vget_low_f32(top.val[0]), bl.val[0] );
    float32x4_t v1 = vcombine_f32( vget_low_f32(top.val[1]), bl.val[1] );
    float32x2_t q = (float32x2_t) vand_u32( (uint32x2_t) vget_high_f32( m_el[2].mVec128), zMask );
    float32x4_t v2 = vcombine_f32( vget_high_f32(top.val[0]), q );       // z0 z1 z2  0
    return b3Matrix3x3( v0, v1, v2 ); 
#else
	return b3Matrix3x3( m_el[0].getX(), m_el[1].getX(), m_el[2].getX(),
                        m_el[0].getY(), m_el[1].getY(), m_el[2].getY(),
                        m_el[0].getZ(), m_el[1].getZ(), m_el[2].getZ());
#endif
}

B3_FORCE_INLINE b3Matrix3x3 
b3Matrix3x3::adjoint() const 
{
	return b3Matrix3x3(cofac(1, 1, 2, 2), cofac(0, 2, 2, 1), cofac(0, 1, 1, 2),
		cofac(1, 2, 2, 0), cofac(0, 0, 2, 2), cofac(0, 2, 1, 0),
		cofac(1, 0, 2, 1), cofac(0, 1, 2, 0), cofac(0, 0, 1, 1));
}

B3_FORCE_INLINE b3Matrix3x3 
b3Matrix3x3::inverse() const
{
	b3Vector3 co = b3MakeVector3(cofac(1, 1, 2, 2), cofac(1, 2, 2, 0), cofac(1, 0, 2, 1));
	b3Scalar det = (*this)[0].dot(co);
	b3FullAssert(det != b3Scalar(0.0));
	b3Scalar s = b3Scalar(1.0) / det;
	return b3Matrix3x3(co.getX() * s, cofac(0, 2, 2, 1) * s, cofac(0, 1, 1, 2) * s,
		co.getY() * s, cofac(0, 0, 2, 2) * s, cofac(0, 2, 1, 0) * s,
		co.getZ() * s, cofac(0, 1, 2, 0) * s, cofac(0, 0, 1, 1) * s);
}

B3_FORCE_INLINE b3Matrix3x3 
b3Matrix3x3::transposeTimes(const b3Matrix3x3& m) const
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))
    // zeros w
//    static const __m128i xyzMask = (const __m128i){ -1ULL, 0xffffffffULL };
    __m128 row = m_el[0].mVec128;
    __m128 m0 = _mm_and_ps( m.getRow(0).mVec128, b3vFFF0fMask );
    __m128 m1 = _mm_and_ps( m.getRow(1).mVec128, b3vFFF0fMask);
    __m128 m2 = _mm_and_ps( m.getRow(2).mVec128, b3vFFF0fMask );
    __m128 r0 = _mm_mul_ps(m0, _mm_shuffle_ps(row, row, 0));
    __m128 r1 = _mm_mul_ps(m0, _mm_shuffle_ps(row, row, 0x55));
    __m128 r2 = _mm_mul_ps(m0, _mm_shuffle_ps(row, row, 0xaa));
    row = m_el[1].mVec128;
    r0 = _mm_add_ps( r0, _mm_mul_ps(m1, _mm_shuffle_ps(row, row, 0)));
    r1 = _mm_add_ps( r1, _mm_mul_ps(m1, _mm_shuffle_ps(row, row, 0x55)));
    r2 = _mm_add_ps( r2, _mm_mul_ps(m1, _mm_shuffle_ps(row, row, 0xaa)));
    row = m_el[2].mVec128;
    r0 = _mm_add_ps( r0, _mm_mul_ps(m2, _mm_shuffle_ps(row, row, 0)));
    r1 = _mm_add_ps( r1, _mm_mul_ps(m2, _mm_shuffle_ps(row, row, 0x55)));
    r2 = _mm_add_ps( r2, _mm_mul_ps(m2, _mm_shuffle_ps(row, row, 0xaa)));
    return b3Matrix3x3( r0, r1, r2 );

#elif defined B3_USE_NEON
    // zeros w
    static const uint32x4_t xyzMask = (const uint32x4_t){ -1, -1, -1, 0 };
    float32x4_t m0 = (float32x4_t) vandq_u32( (uint32x4_t) m.getRow(0).mVec128, xyzMask );
    float32x4_t m1 = (float32x4_t) vandq_u32( (uint32x4_t) m.getRow(1).mVec128, xyzMask );
    float32x4_t m2 = (float32x4_t) vandq_u32( (uint32x4_t) m.getRow(2).mVec128, xyzMask );
    float32x4_t row = m_el[0].mVec128;
    float32x4_t r0 = vmulq_lane_f32( m0, vget_low_f32(row), 0);
    float32x4_t r1 = vmulq_lane_f32( m0, vget_low_f32(row), 1);
    float32x4_t r2 = vmulq_lane_f32( m0, vget_high_f32(row), 0);
    row = m_el[1].mVec128;
    r0 = vmlaq_lane_f32( r0, m1, vget_low_f32(row), 0);
    r1 = vmlaq_lane_f32( r1, m1, vget_low_f32(row), 1);
    r2 = vmlaq_lane_f32( r2, m1, vget_high_f32(row), 0);
    row = m_el[2].mVec128;
    r0 = vmlaq_lane_f32( r0, m2, vget_low_f32(row), 0);
    r1 = vmlaq_lane_f32( r1, m2, vget_low_f32(row), 1);
    r2 = vmlaq_lane_f32( r2, m2, vget_high_f32(row), 0);
    return b3Matrix3x3( r0, r1, r2 );
#else
    return b3Matrix3x3(
		m_el[0].getX() * m[0].getX() + m_el[1].getX() * m[1].getX() + m_el[2].getX() * m[2].getX(),
		m_el[0].getX() * m[0].getY() + m_el[1].getX() * m[1].getY() + m_el[2].getX() * m[2].getY(),
		m_el[0].getX() * m[0].getZ() + m_el[1].getX() * m[1].getZ() + m_el[2].getX() * m[2].getZ(),
		m_el[0].getY() * m[0].getX() + m_el[1].getY() * m[1].getX() + m_el[2].getY() * m[2].getX(),
		m_el[0].getY() * m[0].getY() + m_el[1].getY() * m[1].getY() + m_el[2].getY() * m[2].getY(),
		m_el[0].getY() * m[0].getZ() + m_el[1].getY() * m[1].getZ() + m_el[2].getY() * m[2].getZ(),
		m_el[0].getZ() * m[0].getX() + m_el[1].getZ() * m[1].getX() + m_el[2].getZ() * m[2].getX(),
		m_el[0].getZ() * m[0].getY() + m_el[1].getZ() * m[1].getY() + m_el[2].getZ() * m[2].getY(),
		m_el[0].getZ() * m[0].getZ() + m_el[1].getZ() * m[1].getZ() + m_el[2].getZ() * m[2].getZ());
#endif
}

B3_FORCE_INLINE b3Matrix3x3 
b3Matrix3x3::timesTranspose(const b3Matrix3x3& m) const
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))
    __m128 a0 = m_el[0].mVec128;
    __m128 a1 = m_el[1].mVec128;
    __m128 a2 = m_el[2].mVec128;
    
    b3Matrix3x3 mT = m.transpose(); // we rely on transpose() zeroing w channel so that we don't have to do it here
    __m128 mx = mT[0].mVec128;
    __m128 my = mT[1].mVec128;
    __m128 mz = mT[2].mVec128;
    
    __m128 r0 = _mm_mul_ps(mx, _mm_shuffle_ps(a0, a0, 0x00));
    __m128 r1 = _mm_mul_ps(mx, _mm_shuffle_ps(a1, a1, 0x00));
    __m128 r2 = _mm_mul_ps(mx, _mm_shuffle_ps(a2, a2, 0x00));
    r0 = _mm_add_ps(r0, _mm_mul_ps(my, _mm_shuffle_ps(a0, a0, 0x55)));
    r1 = _mm_add_ps(r1, _mm_mul_ps(my, _mm_shuffle_ps(a1, a1, 0x55)));
    r2 = _mm_add_ps(r2, _mm_mul_ps(my, _mm_shuffle_ps(a2, a2, 0x55)));
    r0 = _mm_add_ps(r0, _mm_mul_ps(mz, _mm_shuffle_ps(a0, a0, 0xaa)));
    r1 = _mm_add_ps(r1, _mm_mul_ps(mz, _mm_shuffle_ps(a1, a1, 0xaa)));
    r2 = _mm_add_ps(r2, _mm_mul_ps(mz, _mm_shuffle_ps(a2, a2, 0xaa)));
    return b3Matrix3x3( r0, r1, r2);
            
#elif defined B3_USE_NEON
    float32x4_t a0 = m_el[0].mVec128;
    float32x4_t a1 = m_el[1].mVec128;
    float32x4_t a2 = m_el[2].mVec128;
    
    b3Matrix3x3 mT = m.transpose(); // we rely on transpose() zeroing w channel so that we don't have to do it here
    float32x4_t mx = mT[0].mVec128;
    float32x4_t my = mT[1].mVec128;
    float32x4_t mz = mT[2].mVec128;
    
    float32x4_t r0 = vmulq_lane_f32( mx, vget_low_f32(a0), 0);
    float32x4_t r1 = vmulq_lane_f32( mx, vget_low_f32(a1), 0);
    float32x4_t r2 = vmulq_lane_f32( mx, vget_low_f32(a2), 0);
    r0 = vmlaq_lane_f32( r0, my, vget_low_f32(a0), 1);
    r1 = vmlaq_lane_f32( r1, my, vget_low_f32(a1), 1);
    r2 = vmlaq_lane_f32( r2, my, vget_low_f32(a2), 1);
    r0 = vmlaq_lane_f32( r0, mz, vget_high_f32(a0), 0);
    r1 = vmlaq_lane_f32( r1, mz, vget_high_f32(a1), 0);
    r2 = vmlaq_lane_f32( r2, mz, vget_high_f32(a2), 0);
    return b3Matrix3x3( r0, r1, r2 );
    
#else
	return b3Matrix3x3(
		m_el[0].dot(m[0]), m_el[0].dot(m[1]), m_el[0].dot(m[2]),
		m_el[1].dot(m[0]), m_el[1].dot(m[1]), m_el[1].dot(m[2]),
		m_el[2].dot(m[0]), m_el[2].dot(m[1]), m_el[2].dot(m[2]));
#endif
}

B3_FORCE_INLINE b3Vector3 
operator*(const b3Matrix3x3& m, const b3Vector3& v) 
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))|| defined (B3_USE_NEON)
    return v.dot3(m[0], m[1], m[2]);
#else
	return b3MakeVector3(m[0].dot(v), m[1].dot(v), m[2].dot(v));
#endif
}


B3_FORCE_INLINE b3Vector3
operator*(const b3Vector3& v, const b3Matrix3x3& m)
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))

    const __m128 vv = v.mVec128;

    __m128 c0 = b3_splat_ps( vv, 0);
    __m128 c1 = b3_splat_ps( vv, 1);
    __m128 c2 = b3_splat_ps( vv, 2);

    c0 = _mm_mul_ps(c0, _mm_and_ps(m[0].mVec128, b3vFFF0fMask) );
    c1 = _mm_mul_ps(c1, _mm_and_ps(m[1].mVec128, b3vFFF0fMask) );
    c0 = _mm_add_ps(c0, c1);
    c2 = _mm_mul_ps(c2, _mm_and_ps(m[2].mVec128, b3vFFF0fMask) );
    
    return b3MakeVector3(_mm_add_ps(c0, c2));
#elif defined(B3_USE_NEON)
    const float32x4_t vv = v.mVec128;
    const float32x2_t vlo = vget_low_f32(vv);
    const float32x2_t vhi = vget_high_f32(vv);

    float32x4_t c0, c1, c2;

    c0 = (float32x4_t) vandq_s32((int32x4_t)m[0].mVec128, b3vFFF0Mask);
    c1 = (float32x4_t) vandq_s32((int32x4_t)m[1].mVec128, b3vFFF0Mask);
    c2 = (float32x4_t) vandq_s32((int32x4_t)m[2].mVec128, b3vFFF0Mask);

    c0 = vmulq_lane_f32(c0, vlo, 0);
    c1 = vmulq_lane_f32(c1, vlo, 1);
    c2 = vmulq_lane_f32(c2, vhi, 0);
    c0 = vaddq_f32(c0, c1);
    c0 = vaddq_f32(c0, c2);
    
    return b3MakeVector3(c0);
#else
	return b3MakeVector3(m.tdotx(v), m.tdoty(v), m.tdotz(v));
#endif
}

B3_FORCE_INLINE b3Matrix3x3 
operator*(const b3Matrix3x3& m1, const b3Matrix3x3& m2)
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))

    __m128 m10 = m1[0].mVec128;  
    __m128 m11 = m1[1].mVec128;
    __m128 m12 = m1[2].mVec128;
    
    __m128 m2v = _mm_and_ps(m2[0].mVec128, b3vFFF0fMask);
    
    __m128 c0 = b3_splat_ps( m10, 0);
    __m128 c1 = b3_splat_ps( m11, 0);
    __m128 c2 = b3_splat_ps( m12, 0);
    
    c0 = _mm_mul_ps(c0, m2v);
    c1 = _mm_mul_ps(c1, m2v);
    c2 = _mm_mul_ps(c2, m2v);
    
    m2v = _mm_and_ps(m2[1].mVec128, b3vFFF0fMask);
    
    __m128 c0_1 = b3_splat_ps( m10, 1);
    __m128 c1_1 = b3_splat_ps( m11, 1);
    __m128 c2_1 = b3_splat_ps( m12, 1);
    
    c0_1 = _mm_mul_ps(c0_1, m2v);
    c1_1 = _mm_mul_ps(c1_1, m2v);
    c2_1 = _mm_mul_ps(c2_1, m2v);
    
    m2v = _mm_and_ps(m2[2].mVec128, b3vFFF0fMask);
    
    c0 = _mm_add_ps(c0, c0_1);
    c1 = _mm_add_ps(c1, c1_1);
    c2 = _mm_add_ps(c2, c2_1);
    
    m10 = b3_splat_ps( m10, 2);
    m11 = b3_splat_ps( m11, 2);
    m12 = b3_splat_ps( m12, 2);
    
    m10 = _mm_mul_ps(m10, m2v);
    m11 = _mm_mul_ps(m11, m2v);
    m12 = _mm_mul_ps(m12, m2v);
    
    c0 = _mm_add_ps(c0, m10);
    c1 = _mm_add_ps(c1, m11);
    c2 = _mm_add_ps(c2, m12);
    
    return b3Matrix3x3(c0, c1, c2);

#elif defined(B3_USE_NEON)

    float32x4_t rv0, rv1, rv2;
    float32x4_t v0, v1, v2;
    float32x4_t mv0, mv1, mv2;

    v0 = m1[0].mVec128;
    v1 = m1[1].mVec128;
    v2 = m1[2].mVec128;

    mv0 = (float32x4_t) vandq_s32((int32x4_t)m2[0].mVec128, b3vFFF0Mask); 
    mv1 = (float32x4_t) vandq_s32((int32x4_t)m2[1].mVec128, b3vFFF0Mask); 
    mv2 = (float32x4_t) vandq_s32((int32x4_t)m2[2].mVec128, b3vFFF0Mask); 
    
    rv0 = vmulq_lane_f32(mv0, vget_low_f32(v0), 0);
    rv1 = vmulq_lane_f32(mv0, vget_low_f32(v1), 0);
    rv2 = vmulq_lane_f32(mv0, vget_low_f32(v2), 0);
    
    rv0 = vmlaq_lane_f32(rv0, mv1, vget_low_f32(v0), 1);
    rv1 = vmlaq_lane_f32(rv1, mv1, vget_low_f32(v1), 1);
    rv2 = vmlaq_lane_f32(rv2, mv1, vget_low_f32(v2), 1);
    
    rv0 = vmlaq_lane_f32(rv0, mv2, vget_high_f32(v0), 0);
    rv1 = vmlaq_lane_f32(rv1, mv2, vget_high_f32(v1), 0);
    rv2 = vmlaq_lane_f32(rv2, mv2, vget_high_f32(v2), 0);

	return b3Matrix3x3(rv0, rv1, rv2);
        
#else	
	return b3Matrix3x3(
		m2.tdotx( m1[0]), m2.tdoty( m1[0]), m2.tdotz( m1[0]),
		m2.tdotx( m1[1]), m2.tdoty( m1[1]), m2.tdotz( m1[1]),
		m2.tdotx( m1[2]), m2.tdoty( m1[2]), m2.tdotz( m1[2]));
#endif
}

/*
B3_FORCE_INLINE b3Matrix3x3 b3MultTransposeLeft(const b3Matrix3x3& m1, const b3Matrix3x3& m2) {
return b3Matrix3x3(
m1[0][0] * m2[0][0] + m1[1][0] * m2[1][0] + m1[2][0] * m2[2][0],
m1[0][0] * m2[0][1] + m1[1][0] * m2[1][1] + m1[2][0] * m2[2][1],
m1[0][0] * m2[0][2] + m1[1][0] * m2[1][2] + m1[2][0] * m2[2][2],
m1[0][1] * m2[0][0] + m1[1][1] * m2[1][0] + m1[2][1] * m2[2][0],
m1[0][1] * m2[0][1] + m1[1][1] * m2[1][1] + m1[2][1] * m2[2][1],
m1[0][1] * m2[0][2] + m1[1][1] * m2[1][2] + m1[2][1] * m2[2][2],
m1[0][2] * m2[0][0] + m1[1][2] * m2[1][0] + m1[2][2] * m2[2][0],
m1[0][2] * m2[0][1] + m1[1][2] * m2[1][1] + m1[2][2] * m2[2][1],
m1[0][2] * m2[0][2] + m1[1][2] * m2[1][2] + m1[2][2] * m2[2][2]);
}
*/

/**@brief Equality operator between two matrices
* It will test all elements are equal.  */
B3_FORCE_INLINE bool operator==(const b3Matrix3x3& m1, const b3Matrix3x3& m2)
{
#if (defined (B3_USE_SSE_IN_API) && defined (B3_USE_SSE))

    __m128 c0, c1, c2;

    c0 = _mm_cmpeq_ps(m1[0].mVec128, m2[0].mVec128);
    c1 = _mm_cmpeq_ps(m1[1].mVec128, m2[1].mVec128);
    c2 = _mm_cmpeq_ps(m1[2].mVec128, m2[2].mVec128);
    
    c0 = _mm_and_ps(c0, c1);
    c0 = _mm_and_ps(c0, c2);

    return (0x7 == _mm_movemask_ps((__m128)c0));
#else 
	return 
    (   m1[0][0] == m2[0][0] && m1[1][0] == m2[1][0] && m1[2][0] == m2[2][0] &&
		m1[0][1] == m2[0][1] && m1[1][1] == m2[1][1] && m1[2][1] == m2[2][1] &&
		m1[0][2] == m2[0][2] && m1[1][2] == m2[1][2] && m1[2][2] == m2[2][2] );
#endif
}

///for serialization
struct	b3Matrix3x3FloatData
{
	b3Vector3FloatData m_el[3];
};

///for serialization
struct	b3Matrix3x3DoubleData
{
	b3Vector3DoubleData m_el[3];
};


	

B3_FORCE_INLINE	void	b3Matrix3x3::serialize(struct	b3Matrix3x3Data& dataOut) const
{
	for (int i=0;i<3;i++)
		m_el[i].serialize(dataOut.m_el[i]);
}

B3_FORCE_INLINE	void	b3Matrix3x3::serializeFloat(struct	b3Matrix3x3FloatData& dataOut) const
{
	for (int i=0;i<3;i++)
		m_el[i].serializeFloat(dataOut.m_el[i]);
}


B3_FORCE_INLINE	void	b3Matrix3x3::deSerialize(const struct	b3Matrix3x3Data& dataIn)
{
	for (int i=0;i<3;i++)
		m_el[i].deSerialize(dataIn.m_el[i]);
}

B3_FORCE_INLINE	void	b3Matrix3x3::deSerializeFloat(const struct	b3Matrix3x3FloatData& dataIn)
{
	for (int i=0;i<3;i++)
		m_el[i].deSerializeFloat(dataIn.m_el[i]);
}

B3_FORCE_INLINE	void	b3Matrix3x3::deSerializeDouble(const struct	b3Matrix3x3DoubleData& dataIn)
{
	for (int i=0;i<3;i++)
		m_el[i].deSerializeDouble(dataIn.m_el[i]);
}

#endif //B3_MATRIX3x3_H


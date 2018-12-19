/*
Copyright (c) 2003-2006 Gino van den Bergen / Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#ifndef	BT_MATRIX3x3_H
#define BT_MATRIX3x3_H

#include "btVector3.h"
#include "btQuaternion.h"
#include <stdio.h>

#ifdef BT_USE_SSE
//const __m128 ATTRIBUTE_ALIGNED16(v2220) = {2.0f, 2.0f, 2.0f, 0.0f};
//const __m128 ATTRIBUTE_ALIGNED16(vMPPP) = {-0.0f, +0.0f, +0.0f, +0.0f};
#define vMPPP (_mm_set_ps (+0.0f, +0.0f, +0.0f, -0.0f))
#endif

#if defined(BT_USE_SSE)
#define v1000 (_mm_set_ps(0.0f,0.0f,0.0f,1.0f))
#define v0100 (_mm_set_ps(0.0f,0.0f,1.0f,0.0f))
#define v0010 (_mm_set_ps(0.0f,1.0f,0.0f,0.0f))
#elif defined(BT_USE_NEON)
const btSimdFloat4 ATTRIBUTE_ALIGNED16(v1000) = {1.0f, 0.0f, 0.0f, 0.0f};
const btSimdFloat4 ATTRIBUTE_ALIGNED16(v0100) = {0.0f, 1.0f, 0.0f, 0.0f};
const btSimdFloat4 ATTRIBUTE_ALIGNED16(v0010) = {0.0f, 0.0f, 1.0f, 0.0f};
#endif

#ifdef BT_USE_DOUBLE_PRECISION
#define btMatrix3x3Data	btMatrix3x3DoubleData 
#else
#define btMatrix3x3Data	btMatrix3x3FloatData
#endif //BT_USE_DOUBLE_PRECISION


/**@brief The btMatrix3x3 class implements a 3x3 rotation matrix, to perform linear algebra in combination with btQuaternion, btTransform and btVector3.
* Make sure to only include a pure orthogonal matrix without scaling. */
ATTRIBUTE_ALIGNED16(class) btMatrix3x3 {

	///Data storage for the matrix, each vector is a row of the matrix
	btVector3 m_el[3];

public:
	/** @brief No initializaion constructor */
	btMatrix3x3 () {}

	//		explicit btMatrix3x3(const btScalar *m) { setFromOpenGLSubMatrix(m); }

	/**@brief Constructor from Quaternion */
	explicit btMatrix3x3(const btQuaternion& q) { setRotation(q); }
	/*
	template <typename btScalar>
	Matrix3x3(const btScalar& yaw, const btScalar& pitch, const btScalar& roll)
	{ 
	setEulerYPR(yaw, pitch, roll);
	}
	*/
	/** @brief Constructor with row major formatting */
	btMatrix3x3(const btScalar& xx, const btScalar& xy, const btScalar& xz,
		const btScalar& yx, const btScalar& yy, const btScalar& yz,
		const btScalar& zx, const btScalar& zy, const btScalar& zz)
	{ 
		setValue(xx, xy, xz, 
			yx, yy, yz, 
			zx, zy, zz);
	}

#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))|| defined (BT_USE_NEON)
	SIMD_FORCE_INLINE btMatrix3x3 (const btSimdFloat4 v0, const btSimdFloat4 v1, const btSimdFloat4 v2 ) 
	{
        m_el[0].mVec128 = v0;
        m_el[1].mVec128 = v1;
        m_el[2].mVec128 = v2;
	}

	SIMD_FORCE_INLINE btMatrix3x3 (const btVector3& v0, const btVector3& v1, const btVector3& v2 ) 
	{
        m_el[0] = v0;
        m_el[1] = v1;
        m_el[2] = v2;
	}

	// Copy constructor
	SIMD_FORCE_INLINE btMatrix3x3(const btMatrix3x3& rhs)
	{
		m_el[0].mVec128 = rhs.m_el[0].mVec128;
		m_el[1].mVec128 = rhs.m_el[1].mVec128;
		m_el[2].mVec128 = rhs.m_el[2].mVec128;
	}

	// Assignment Operator
	SIMD_FORCE_INLINE btMatrix3x3& operator=(const btMatrix3x3& m) 
	{
		m_el[0].mVec128 = m.m_el[0].mVec128;
		m_el[1].mVec128 = m.m_el[1].mVec128;
		m_el[2].mVec128 = m.m_el[2].mVec128;
		
		return *this;
	}

#else

	/** @brief Copy constructor */
	SIMD_FORCE_INLINE btMatrix3x3 (const btMatrix3x3& other)
	{
		m_el[0] = other.m_el[0];
		m_el[1] = other.m_el[1];
		m_el[2] = other.m_el[2];
	}
    
	/** @brief Assignment Operator */
	SIMD_FORCE_INLINE btMatrix3x3& operator=(const btMatrix3x3& other)
	{
		m_el[0] = other.m_el[0];
		m_el[1] = other.m_el[1];
		m_el[2] = other.m_el[2];
		return *this;
	}

#endif

	/** @brief Get a column of the matrix as a vector 
	*  @param i Column number 0 indexed */
	SIMD_FORCE_INLINE btVector3 getColumn(int i) const
	{
		return btVector3(m_el[0][i],m_el[1][i],m_el[2][i]);
	}


	/** @brief Get a row of the matrix as a vector 
	*  @param i Row number 0 indexed */
	SIMD_FORCE_INLINE const btVector3& getRow(int i) const
	{
		btFullAssert(0 <= i && i < 3);
		return m_el[i];
	}

	/** @brief Get a mutable reference to a row of the matrix as a vector 
	*  @param i Row number 0 indexed */
	SIMD_FORCE_INLINE btVector3&  operator[](int i)
	{ 
		btFullAssert(0 <= i && i < 3);
		return m_el[i]; 
	}

	/** @brief Get a const reference to a row of the matrix as a vector 
	*  @param i Row number 0 indexed */
	SIMD_FORCE_INLINE const btVector3& operator[](int i) const
	{
		btFullAssert(0 <= i && i < 3);
		return m_el[i]; 
	}

	/** @brief Multiply by the target matrix on the right
	*  @param m Rotation matrix to be applied 
	* Equivilant to this = this * m */
	btMatrix3x3& operator*=(const btMatrix3x3& m); 

	/** @brief Adds by the target matrix on the right
	*  @param m matrix to be applied 
	* Equivilant to this = this + m */
	btMatrix3x3& operator+=(const btMatrix3x3& m); 

	/** @brief Substractss by the target matrix on the right
	*  @param m matrix to be applied 
	* Equivilant to this = this - m */
	btMatrix3x3& operator-=(const btMatrix3x3& m); 

	/** @brief Set from the rotational part of a 4x4 OpenGL matrix
	*  @param m A pointer to the beginning of the array of scalars*/
	void setFromOpenGLSubMatrix(const btScalar *m)
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
	void setValue(const btScalar& xx, const btScalar& xy, const btScalar& xz, 
		const btScalar& yx, const btScalar& yy, const btScalar& yz, 
		const btScalar& zx, const btScalar& zy, const btScalar& zz)
	{
		m_el[0].setValue(xx,xy,xz);
		m_el[1].setValue(yx,yy,yz);
		m_el[2].setValue(zx,zy,zz);
	}

	/** @brief Set the matrix from a quaternion
	*  @param q The Quaternion to match */  
	void setRotation(const btQuaternion& q) 
	{
		btScalar d = q.length2();
		btFullAssert(d != btScalar(0.0));
		btScalar s = btScalar(2.0) / d;
    
    #if defined BT_USE_SIMD_VECTOR3 && defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE)
        __m128	vs, Q = q.get128();
		__m128i Qi = btCastfTo128i(Q);
        __m128	Y, Z;
        __m128	V1, V2, V3;
        __m128	V11, V21, V31;
        __m128	NQ = _mm_xor_ps(Q, btvMzeroMask);
		__m128i NQi = btCastfTo128i(NQ);
        
        V1 = btCastiTo128f(_mm_shuffle_epi32 (Qi, BT_SHUFFLE(1,0,2,3)));	// Y X Z W
		V2 = _mm_shuffle_ps(NQ, Q, BT_SHUFFLE(0,0,1,3));     // -X -X  Y  W
        V3 = btCastiTo128f(_mm_shuffle_epi32 (Qi, BT_SHUFFLE(2,1,0,3)));	// Z Y X W
        V1 = _mm_xor_ps(V1, vMPPP);	//	change the sign of the first element
			
        V11	= btCastiTo128f(_mm_shuffle_epi32 (Qi, BT_SHUFFLE(1,1,0,3)));	// Y Y X W
		V21 = _mm_unpackhi_ps(Q, Q);                    //  Z  Z  W  W
		V31 = _mm_shuffle_ps(Q, NQ, BT_SHUFFLE(0,2,0,3));	//  X  Z -X -W

		V2 = V2 * V1;	//
		V1 = V1 * V11;	//
		V3 = V3 * V31;	//

        V11 = _mm_shuffle_ps(NQ, Q, BT_SHUFFLE(2,3,1,3));	//	-Z -W  Y  W
		V11 = V11 * V21;	//
        V21 = _mm_xor_ps(V21, vMPPP);	//	change the sign of the first element
		V31 = _mm_shuffle_ps(Q, NQ, BT_SHUFFLE(3,3,1,3));	//	 W  W -Y -W
        V31 = _mm_xor_ps(V31, vMPPP);	//	change the sign of the first element
		Y = btCastiTo128f(_mm_shuffle_epi32 (NQi, BT_SHUFFLE(3,2,0,3)));	// -W -Z -X -W
		Z = btCastiTo128f(_mm_shuffle_epi32 (Qi, BT_SHUFFLE(1,0,1,3)));	//  Y  X  Y  W

		vs = _mm_load_ss(&s);
		V21 = V21 * Y;
		V31 = V31 * Z;

		V1 = V1 + V11;
        V2 = V2 + V21;
        V3 = V3 + V31;

        vs = bt_splat3_ps(vs, 0);
            //	s ready
        V1 = V1 * vs;
        V2 = V2 * vs;
        V3 = V3 * vs;
        
        V1 = V1 + v1000;
        V2 = V2 + v0100;
        V3 = V3 + v0010;
        
        m_el[0] = V1; 
        m_el[1] = V2;
        m_el[2] = V3;
    #else    
		btScalar xs = q.x() * s,   ys = q.y() * s,   zs = q.z() * s;
		btScalar wx = q.w() * xs,  wy = q.w() * ys,  wz = q.w() * zs;
		btScalar xx = q.x() * xs,  xy = q.x() * ys,  xz = q.x() * zs;
		btScalar yy = q.y() * ys,  yz = q.y() * zs,  zz = q.z() * zs;
		setValue(
            btScalar(1.0) - (yy + zz), xy - wz, xz + wy,
			xy + wz, btScalar(1.0) - (xx + zz), yz - wx,
			xz - wy, yz + wx, btScalar(1.0) - (xx + yy));
	#endif
    }


	/** @brief Set the matrix from euler angles using YPR around YXZ respectively
	*  @param yaw Yaw about Y axis
	*  @param pitch Pitch about X axis
	*  @param roll Roll about Z axis 
	*/
	void setEulerYPR(const btScalar& yaw, const btScalar& pitch, const btScalar& roll) 
	{
		setEulerZYX(roll, pitch, yaw);
	}

	/** @brief Set the matrix from euler angles YPR around ZYX axes
	* @param eulerX Roll about X axis
	* @param eulerY Pitch around Y axis
	* @param eulerZ Yaw about Z axis
	* 
	* These angles are used to produce a rotation matrix. The euler
	* angles are applied in ZYX order. I.e a vector is first rotated 
	* about X then Y and then Z
	**/
	void setEulerZYX(btScalar eulerX,btScalar eulerY,btScalar eulerZ) { 
		///@todo proposed to reverse this since it's labeled zyx but takes arguments xyz and it will match all other parts of the code
		btScalar ci ( btCos(eulerX)); 
		btScalar cj ( btCos(eulerY)); 
		btScalar ch ( btCos(eulerZ)); 
		btScalar si ( btSin(eulerX)); 
		btScalar sj ( btSin(eulerY)); 
		btScalar sh ( btSin(eulerZ)); 
		btScalar cc = ci * ch; 
		btScalar cs = ci * sh; 
		btScalar sc = si * ch; 
		btScalar ss = si * sh;

		setValue(cj * ch, sj * sc - cs, sj * cc + ss,
			cj * sh, sj * ss + cc, sj * cs - sc, 
			-sj,      cj * si,      cj * ci);
	}

	/**@brief Set the matrix to the identity */
	void setIdentity()
	{ 
#if (defined(BT_USE_SSE_IN_API)&& defined (BT_USE_SSE)) || defined(BT_USE_NEON)
			m_el[0] = v1000; 
			m_el[1] = v0100;
			m_el[2] = v0010;
#else
		setValue(btScalar(1.0), btScalar(0.0), btScalar(0.0), 
			btScalar(0.0), btScalar(1.0), btScalar(0.0), 
			btScalar(0.0), btScalar(0.0), btScalar(1.0)); 
#endif
	}

	static const btMatrix3x3&	getIdentity()
	{
#if (defined(BT_USE_SSE_IN_API)&& defined (BT_USE_SSE)) || defined(BT_USE_NEON)
        static const btMatrix3x3 
        identityMatrix(v1000, v0100, v0010);
#else
		static const btMatrix3x3 
        identityMatrix(
            btScalar(1.0), btScalar(0.0), btScalar(0.0), 
			btScalar(0.0), btScalar(1.0), btScalar(0.0), 
			btScalar(0.0), btScalar(0.0), btScalar(1.0));
#endif
		return identityMatrix;
	}

	/**@brief Fill the rotational part of an OpenGL matrix and clear the shear/perspective
	* @param m The array to be filled */
	void getOpenGLSubMatrix(btScalar *m) const 
	{
#if defined BT_USE_SIMD_VECTOR3 && defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE)
        __m128 v0 = m_el[0].mVec128;
        __m128 v1 = m_el[1].mVec128;
        __m128 v2 = m_el[2].mVec128;    //  x2 y2 z2 w2
        __m128 *vm = (__m128 *)m;
        __m128 vT;
        
        v2 = _mm_and_ps(v2, btvFFF0fMask);  //  x2 y2 z2 0
        
        vT = _mm_unpackhi_ps(v0, v1);	//	z0 z1 * *
        v0 = _mm_unpacklo_ps(v0, v1);	//	x0 x1 y0 y1

        v1 = _mm_shuffle_ps(v0, v2, BT_SHUFFLE(2, 3, 1, 3) );	// y0 y1 y2 0
        v0 = _mm_shuffle_ps(v0, v2, BT_SHUFFLE(0, 1, 0, 3) );	// x0 x1 x2 0
        v2 = btCastdTo128f(_mm_move_sd(btCastfTo128d(v2), btCastfTo128d(vT)));	// z0 z1 z2 0

        vm[0] = v0;
        vm[1] = v1;
        vm[2] = v2;
#elif defined(BT_USE_NEON)
        // note: zeros the w channel. We can preserve it at the cost of two more vtrn instructions.
        static const uint32x2_t zMask = (const uint32x2_t) {static_cast<uint32_t>(-1), 0 };
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
		m[0]  = btScalar(m_el[0].x()); 
		m[1]  = btScalar(m_el[1].x());
		m[2]  = btScalar(m_el[2].x());
		m[3]  = btScalar(0.0); 
		m[4]  = btScalar(m_el[0].y());
		m[5]  = btScalar(m_el[1].y());
		m[6]  = btScalar(m_el[2].y());
		m[7]  = btScalar(0.0); 
		m[8]  = btScalar(m_el[0].z()); 
		m[9]  = btScalar(m_el[1].z());
		m[10] = btScalar(m_el[2].z());
		m[11] = btScalar(0.0); 
#endif
	}

	/**@brief Get the matrix represented as a quaternion 
	* @param q The quaternion which will be set */
	void getRotation(btQuaternion& q) const
	{
#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))|| defined (BT_USE_NEON)
        btScalar trace = m_el[0].x() + m_el[1].y() + m_el[2].z();
        btScalar s, x;
        
        union {
            btSimdFloat4 vec;
            btScalar f[4];
        } temp;
        
        if (trace > btScalar(0.0)) 
        {
            x = trace + btScalar(1.0);

            temp.f[0]=m_el[2].y() - m_el[1].z();
            temp.f[1]=m_el[0].z() - m_el[2].x();
            temp.f[2]=m_el[1].x() - m_el[0].y();
            temp.f[3]=x;
            //temp.f[3]= s * btScalar(0.5);
        } 
        else 
        {
            int i, j, k;
            if(m_el[0].x() < m_el[1].y()) 
            { 
                if( m_el[1].y() < m_el[2].z() )
                    { i = 2; j = 0; k = 1; }
                else
                    { i = 1; j = 2; k = 0; }
            }
            else
            {
                if( m_el[0].x() < m_el[2].z())
                    { i = 2; j = 0; k = 1; }
                else
                    { i = 0; j = 1; k = 2; }
            }

            x = m_el[i][i] - m_el[j][j] - m_el[k][k] + btScalar(1.0);

            temp.f[3] = (m_el[k][j] - m_el[j][k]);
            temp.f[j] = (m_el[j][i] + m_el[i][j]);
            temp.f[k] = (m_el[k][i] + m_el[i][k]);
            temp.f[i] = x;
            //temp.f[i] = s * btScalar(0.5);
        }

        s = btSqrt(x);
        q.set128(temp.vec);
        s = btScalar(0.5) / s;

        q *= s;
#else    
		btScalar trace = m_el[0].x() + m_el[1].y() + m_el[2].z();

		btScalar temp[4];

		if (trace > btScalar(0.0)) 
		{
			btScalar s = btSqrt(trace + btScalar(1.0));
			temp[3]=(s * btScalar(0.5));
			s = btScalar(0.5) / s;

			temp[0]=((m_el[2].y() - m_el[1].z()) * s);
			temp[1]=((m_el[0].z() - m_el[2].x()) * s);
			temp[2]=((m_el[1].x() - m_el[0].y()) * s);
		} 
		else 
		{
			int i = m_el[0].x() < m_el[1].y() ? 
				(m_el[1].y() < m_el[2].z() ? 2 : 1) :
				(m_el[0].x() < m_el[2].z() ? 2 : 0); 
			int j = (i + 1) % 3;  
			int k = (i + 2) % 3;

			btScalar s = btSqrt(m_el[i][i] - m_el[j][j] - m_el[k][k] + btScalar(1.0));
			temp[i] = s * btScalar(0.5);
			s = btScalar(0.5) / s;

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
	void getEulerYPR(btScalar& yaw, btScalar& pitch, btScalar& roll) const
	{

		// first use the normal calculus
		yaw = btScalar(btAtan2(m_el[1].x(), m_el[0].x()));
		pitch = btScalar(btAsin(-m_el[2].x()));
		roll = btScalar(btAtan2(m_el[2].y(), m_el[2].z()));

		// on pitch = +/-HalfPI
		if (btFabs(pitch)==SIMD_HALF_PI)
		{
			if (yaw>0)
				yaw-=SIMD_PI;
			else
				yaw+=SIMD_PI;

			if (roll>0)
				roll-=SIMD_PI;
			else
				roll+=SIMD_PI;
		}
	};


	/**@brief Get the matrix represented as euler angles around ZYX
	* @param yaw Yaw around Z axis
	* @param pitch Pitch around Y axis
	* @param roll around X axis 
	* @param solution_number Which solution of two possible solutions ( 1 or 2) are possible values*/	
	void getEulerZYX(btScalar& yaw, btScalar& pitch, btScalar& roll, unsigned int solution_number = 1) const
	{
		struct Euler
		{
			btScalar yaw;
			btScalar pitch;
			btScalar roll;
		};

		Euler euler_out;
		Euler euler_out2; //second solution
		//get the pointer to the raw data

		// Check that pitch is not at a singularity
		if (btFabs(m_el[2].x()) >= 1)
		{
			euler_out.yaw = 0;
			euler_out2.yaw = 0;

			// From difference of angles formula
			btScalar delta = btAtan2(m_el[0].x(),m_el[0].z());
			if (m_el[2].x() > 0)  //gimbal locked up
			{
				euler_out.pitch = SIMD_PI / btScalar(2.0);
				euler_out2.pitch = SIMD_PI / btScalar(2.0);
				euler_out.roll = euler_out.pitch + delta;
				euler_out2.roll = euler_out.pitch + delta;
			}
			else // gimbal locked down
			{
				euler_out.pitch = -SIMD_PI / btScalar(2.0);
				euler_out2.pitch = -SIMD_PI / btScalar(2.0);
				euler_out.roll = -euler_out.pitch + delta;
				euler_out2.roll = -euler_out.pitch + delta;
			}
		}
		else
		{
			euler_out.pitch = - btAsin(m_el[2].x());
			euler_out2.pitch = SIMD_PI - euler_out.pitch;

			euler_out.roll = btAtan2(m_el[2].y()/btCos(euler_out.pitch), 
				m_el[2].z()/btCos(euler_out.pitch));
			euler_out2.roll = btAtan2(m_el[2].y()/btCos(euler_out2.pitch), 
				m_el[2].z()/btCos(euler_out2.pitch));

			euler_out.yaw = btAtan2(m_el[1].x()/btCos(euler_out.pitch), 
				m_el[0].x()/btCos(euler_out.pitch));
			euler_out2.yaw = btAtan2(m_el[1].x()/btCos(euler_out2.pitch), 
				m_el[0].x()/btCos(euler_out2.pitch));
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

	btMatrix3x3 scaled(const btVector3& s) const
	{
#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))|| defined (BT_USE_NEON)
		return btMatrix3x3(m_el[0] * s, m_el[1] * s, m_el[2] * s);
#else		
		return btMatrix3x3(
            m_el[0].x() * s.x(), m_el[0].y() * s.y(), m_el[0].z() * s.z(),
			m_el[1].x() * s.x(), m_el[1].y() * s.y(), m_el[1].z() * s.z(),
			m_el[2].x() * s.x(), m_el[2].y() * s.y(), m_el[2].z() * s.z());
#endif
	}

	/**@brief Return the determinant of the matrix */
	btScalar            determinant() const;
	/**@brief Return the adjoint of the matrix */
	btMatrix3x3 adjoint() const;
	/**@brief Return the matrix with all values non negative */
	btMatrix3x3 absolute() const;
	/**@brief Return the transpose of the matrix */
	btMatrix3x3 transpose() const;
	/**@brief Return the inverse of the matrix */
	btMatrix3x3 inverse() const; 

	/// Solve A * x = b, where b is a column vector. This is more efficient
	/// than computing the inverse in one-shot cases.
	///Solve33 is from Box2d, thanks to Erin Catto,
	btVector3 solve33(const btVector3& b) const
	{
		btVector3 col1 = getColumn(0);
		btVector3 col2 = getColumn(1);
		btVector3 col3 = getColumn(2);
		
		btScalar det = btDot(col1, btCross(col2, col3));
		if (btFabs(det)>SIMD_EPSILON)
		{
			det = 1.0f / det;
		}
		btVector3 x;
		x[0] = det * btDot(b, btCross(col2, col3));
		x[1] = det * btDot(col1, btCross(b, col3));
		x[2] = det * btDot(col1, btCross(col2, b));
		return x;
	}

	btMatrix3x3 transposeTimes(const btMatrix3x3& m) const;
	btMatrix3x3 timesTranspose(const btMatrix3x3& m) const;

	SIMD_FORCE_INLINE btScalar tdotx(const btVector3& v) const 
	{
		return m_el[0].x() * v.x() + m_el[1].x() * v.y() + m_el[2].x() * v.z();
	}
	SIMD_FORCE_INLINE btScalar tdoty(const btVector3& v) const 
	{
		return m_el[0].y() * v.x() + m_el[1].y() * v.y() + m_el[2].y() * v.z();
	}
	SIMD_FORCE_INLINE btScalar tdotz(const btVector3& v) const 
	{
		return m_el[0].z() * v.x() + m_el[1].z() * v.y() + m_el[2].z() * v.z();
	}

	///extractRotation is from "A robust method to extract the rotational part of deformations"
	///See http://dl.acm.org/citation.cfm?doid=2994258.2994269
	///decomposes a matrix A in a orthogonal matrix R and a
	///symmetric matrix S:
	///A = R*S.
	///note that R can include both rotation and scaling.
	SIMD_FORCE_INLINE void extractRotation(btQuaternion &q,btScalar tolerance = 1.0e-9, int maxIter=100)
	{
		int iter =0;
		btScalar w;
		const btMatrix3x3& A=*this;
		for(iter = 0; iter < maxIter; iter++)
		{
			btMatrix3x3 R(q);
			btVector3 omega = (R.getColumn(0).cross(A.getColumn(0)) + R.getColumn(1).cross(A.getColumn(1)) 
				+ R.getColumn(2).cross(A.getColumn(2))
				) * (btScalar(1.0) / btFabs(R.getColumn(0).dot(A.getColumn(0)) + R.getColumn
				(1).dot(A.getColumn(1)) + R.getColumn(2).dot(A.getColumn(2))) +
					tolerance);
			w = omega.norm();
			if(w < tolerance)
				break;
			q = btQuaternion(btVector3((btScalar(1.0) / w) * omega),w) *
				q;
			q.normalize();
		}
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
	void diagonalize(btMatrix3x3& rot, btScalar threshold, int maxSteps)
	{
		rot.setIdentity();
		for (int step = maxSteps; step > 0; step--)
		{
			// find off-diagonal element [p][q] with largest magnitude
			int p = 0;
			int q = 1;
			int r = 2;
			btScalar max = btFabs(m_el[0][1]);
			btScalar v = btFabs(m_el[0][2]);
			if (v > max)
			{
				q = 2;
				r = 1;
				max = v;
			}
			v = btFabs(m_el[1][2]);
			if (v > max)
			{
				p = 1;
				q = 2;
				r = 0;
				max = v;
			}

			btScalar t = threshold * (btFabs(m_el[0][0]) + btFabs(m_el[1][1]) + btFabs(m_el[2][2]));
			if (max <= t)
			{
				if (max <= SIMD_EPSILON * t)
				{
					return;
				}
				step = 1;
			}

			// compute Jacobi rotation J which leads to a zero for element [p][q] 
			btScalar mpq = m_el[p][q];
			btScalar theta = (m_el[q][q] - m_el[p][p]) / (2 * mpq);
			btScalar theta2 = theta * theta;
			btScalar cos;
			btScalar sin;
			if (theta2 * theta2 < btScalar(10 / SIMD_EPSILON))
			{
				t = (theta >= 0) ? 1 / (theta + btSqrt(1 + theta2))
					: 1 / (theta - btSqrt(1 + theta2));
				cos = 1 / btSqrt(1 + t * t);
				sin = cos * t;
			}
			else
			{
				// approximation for large theta-value, i.e., a nearly diagonal matrix
				t = 1 / (theta * (2 + btScalar(0.5) / theta2));
				cos = 1 - btScalar(0.5) * t * t;
				sin = cos * t;
			}

			// apply rotation to matrix (this = J^T * this * J)
			m_el[p][q] = m_el[q][p] = 0;
			m_el[p][p] -= t * mpq;
			m_el[q][q] += t * mpq;
			btScalar mrp = m_el[r][p];
			btScalar mrq = m_el[r][q];
			m_el[r][p] = m_el[p][r] = cos * mrp - sin * mrq;
			m_el[r][q] = m_el[q][r] = cos * mrq + sin * mrp;

			// apply rotation to rot (rot = rot * J)
			for (int i = 0; i < 3; i++)
			{
				btVector3& row = rot[i];
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
	btScalar cofac(int r1, int c1, int r2, int c2) const 
	{
		return m_el[r1][c1] * m_el[r2][c2] - m_el[r1][c2] * m_el[r2][c1];
	}

	void	serialize(struct	btMatrix3x3Data& dataOut) const;

	void	serializeFloat(struct	btMatrix3x3FloatData& dataOut) const;

	void	deSerialize(const struct	btMatrix3x3Data& dataIn);

	void	deSerializeFloat(const struct	btMatrix3x3FloatData& dataIn);

	void	deSerializeDouble(const struct	btMatrix3x3DoubleData& dataIn);

};


SIMD_FORCE_INLINE btMatrix3x3& 
btMatrix3x3::operator*=(const btMatrix3x3& m)
{
#if defined BT_USE_SIMD_VECTOR3 && defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE)
    __m128 rv00, rv01, rv02;
    __m128 rv10, rv11, rv12;
    __m128 rv20, rv21, rv22;
    __m128 mv0, mv1, mv2;

    rv02 = m_el[0].mVec128;
    rv12 = m_el[1].mVec128;
    rv22 = m_el[2].mVec128;

    mv0 = _mm_and_ps(m[0].mVec128, btvFFF0fMask); 
    mv1 = _mm_and_ps(m[1].mVec128, btvFFF0fMask); 
    mv2 = _mm_and_ps(m[2].mVec128, btvFFF0fMask); 
    
    // rv0
    rv00 = bt_splat_ps(rv02, 0);
    rv01 = bt_splat_ps(rv02, 1);
    rv02 = bt_splat_ps(rv02, 2);
    
    rv00 = _mm_mul_ps(rv00, mv0);
    rv01 = _mm_mul_ps(rv01, mv1);
    rv02 = _mm_mul_ps(rv02, mv2);
    
    // rv1
    rv10 = bt_splat_ps(rv12, 0);
    rv11 = bt_splat_ps(rv12, 1);
    rv12 = bt_splat_ps(rv12, 2);
    
    rv10 = _mm_mul_ps(rv10, mv0);
    rv11 = _mm_mul_ps(rv11, mv1);
    rv12 = _mm_mul_ps(rv12, mv2);
    
    // rv2
    rv20 = bt_splat_ps(rv22, 0);
    rv21 = bt_splat_ps(rv22, 1);
    rv22 = bt_splat_ps(rv22, 2);
    
    rv20 = _mm_mul_ps(rv20, mv0);
    rv21 = _mm_mul_ps(rv21, mv1);
    rv22 = _mm_mul_ps(rv22, mv2);

    rv00 = _mm_add_ps(rv00, rv01);
    rv10 = _mm_add_ps(rv10, rv11);
    rv20 = _mm_add_ps(rv20, rv21);

    m_el[0].mVec128 = _mm_add_ps(rv00, rv02);
    m_el[1].mVec128 = _mm_add_ps(rv10, rv12);
    m_el[2].mVec128 = _mm_add_ps(rv20, rv22);

#elif defined(BT_USE_NEON)

    float32x4_t rv0, rv1, rv2;
    float32x4_t v0, v1, v2;
    float32x4_t mv0, mv1, mv2;

    v0 = m_el[0].mVec128;
    v1 = m_el[1].mVec128;
    v2 = m_el[2].mVec128;

    mv0 = (float32x4_t) vandq_s32((int32x4_t)m[0].mVec128, btvFFF0Mask); 
    mv1 = (float32x4_t) vandq_s32((int32x4_t)m[1].mVec128, btvFFF0Mask); 
    mv2 = (float32x4_t) vandq_s32((int32x4_t)m[2].mVec128, btvFFF0Mask); 
    
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

SIMD_FORCE_INLINE btMatrix3x3& 
btMatrix3x3::operator+=(const btMatrix3x3& m)
{
#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))|| defined (BT_USE_NEON)
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

SIMD_FORCE_INLINE btMatrix3x3
operator*(const btMatrix3x3& m, const btScalar & k)
{
#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))
    __m128 vk = bt_splat_ps(_mm_load_ss((float *)&k), 0x80);
    return btMatrix3x3(
                _mm_mul_ps(m[0].mVec128, vk), 
                _mm_mul_ps(m[1].mVec128, vk), 
                _mm_mul_ps(m[2].mVec128, vk)); 
#elif defined(BT_USE_NEON)
    return btMatrix3x3(
                vmulq_n_f32(m[0].mVec128, k),
                vmulq_n_f32(m[1].mVec128, k),
                vmulq_n_f32(m[2].mVec128, k)); 
#else
	return btMatrix3x3(
		m[0].x()*k,m[0].y()*k,m[0].z()*k,
		m[1].x()*k,m[1].y()*k,m[1].z()*k,
		m[2].x()*k,m[2].y()*k,m[2].z()*k);
#endif
}

SIMD_FORCE_INLINE btMatrix3x3 
operator+(const btMatrix3x3& m1, const btMatrix3x3& m2)
{
#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))|| defined (BT_USE_NEON)
	return btMatrix3x3(
        m1[0].mVec128 + m2[0].mVec128,
        m1[1].mVec128 + m2[1].mVec128,
        m1[2].mVec128 + m2[2].mVec128);
#else
	return btMatrix3x3(
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

SIMD_FORCE_INLINE btMatrix3x3 
operator-(const btMatrix3x3& m1, const btMatrix3x3& m2)
{
#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))|| defined (BT_USE_NEON)
	return btMatrix3x3(
        m1[0].mVec128 - m2[0].mVec128,
        m1[1].mVec128 - m2[1].mVec128,
        m1[2].mVec128 - m2[2].mVec128);
#else
	return btMatrix3x3(
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


SIMD_FORCE_INLINE btMatrix3x3& 
btMatrix3x3::operator-=(const btMatrix3x3& m)
{
#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))|| defined (BT_USE_NEON)
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


SIMD_FORCE_INLINE btScalar 
btMatrix3x3::determinant() const
{ 
	return btTriple((*this)[0], (*this)[1], (*this)[2]);
}


SIMD_FORCE_INLINE btMatrix3x3 
btMatrix3x3::absolute() const
{
#if defined BT_USE_SIMD_VECTOR3 && (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))
    return btMatrix3x3(
            _mm_and_ps(m_el[0].mVec128, btvAbsfMask),
            _mm_and_ps(m_el[1].mVec128, btvAbsfMask),
            _mm_and_ps(m_el[2].mVec128, btvAbsfMask));
#elif defined(BT_USE_NEON)
    return btMatrix3x3(
            (float32x4_t)vandq_s32((int32x4_t)m_el[0].mVec128, btv3AbsMask),
            (float32x4_t)vandq_s32((int32x4_t)m_el[1].mVec128, btv3AbsMask),
            (float32x4_t)vandq_s32((int32x4_t)m_el[2].mVec128, btv3AbsMask));
#else	
	return btMatrix3x3(
            btFabs(m_el[0].x()), btFabs(m_el[0].y()), btFabs(m_el[0].z()),
            btFabs(m_el[1].x()), btFabs(m_el[1].y()), btFabs(m_el[1].z()),
            btFabs(m_el[2].x()), btFabs(m_el[2].y()), btFabs(m_el[2].z()));
#endif
}

SIMD_FORCE_INLINE btMatrix3x3 
btMatrix3x3::transpose() const 
{
#if defined BT_USE_SIMD_VECTOR3 && (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))
    __m128 v0 = m_el[0].mVec128;
    __m128 v1 = m_el[1].mVec128;
    __m128 v2 = m_el[2].mVec128;    //  x2 y2 z2 w2
    __m128 vT;
    
    v2 = _mm_and_ps(v2, btvFFF0fMask);  //  x2 y2 z2 0
    
    vT = _mm_unpackhi_ps(v0, v1);	//	z0 z1 * *
    v0 = _mm_unpacklo_ps(v0, v1);	//	x0 x1 y0 y1

    v1 = _mm_shuffle_ps(v0, v2, BT_SHUFFLE(2, 3, 1, 3) );	// y0 y1 y2 0
    v0 = _mm_shuffle_ps(v0, v2, BT_SHUFFLE(0, 1, 0, 3) );	// x0 x1 x2 0
    v2 = btCastdTo128f(_mm_move_sd(btCastfTo128d(v2), btCastfTo128d(vT)));	// z0 z1 z2 0


    return btMatrix3x3( v0, v1, v2 );
#elif defined(BT_USE_NEON)
    // note: zeros the w channel. We can preserve it at the cost of two more vtrn instructions.
    static const uint32x2_t zMask = (const uint32x2_t) {static_cast<uint32_t>(-1), 0 };
    float32x4x2_t top = vtrnq_f32( m_el[0].mVec128, m_el[1].mVec128 );  // {x0 x1 z0 z1}, {y0 y1 w0 w1}
    float32x2x2_t bl = vtrn_f32( vget_low_f32(m_el[2].mVec128), vdup_n_f32(0.0f) );       // {x2  0 }, {y2 0}
    float32x4_t v0 = vcombine_f32( vget_low_f32(top.val[0]), bl.val[0] );
    float32x4_t v1 = vcombine_f32( vget_low_f32(top.val[1]), bl.val[1] );
    float32x2_t q = (float32x2_t) vand_u32( (uint32x2_t) vget_high_f32( m_el[2].mVec128), zMask );
    float32x4_t v2 = vcombine_f32( vget_high_f32(top.val[0]), q );       // z0 z1 z2  0
    return btMatrix3x3( v0, v1, v2 ); 
#else
	return btMatrix3x3( m_el[0].x(), m_el[1].x(), m_el[2].x(),
                        m_el[0].y(), m_el[1].y(), m_el[2].y(),
                        m_el[0].z(), m_el[1].z(), m_el[2].z());
#endif
}

SIMD_FORCE_INLINE btMatrix3x3 
btMatrix3x3::adjoint() const 
{
	return btMatrix3x3(cofac(1, 1, 2, 2), cofac(0, 2, 2, 1), cofac(0, 1, 1, 2),
		cofac(1, 2, 2, 0), cofac(0, 0, 2, 2), cofac(0, 2, 1, 0),
		cofac(1, 0, 2, 1), cofac(0, 1, 2, 0), cofac(0, 0, 1, 1));
}

SIMD_FORCE_INLINE btMatrix3x3 
btMatrix3x3::inverse() const
{
	btVector3 co(cofac(1, 1, 2, 2), cofac(1, 2, 2, 0), cofac(1, 0, 2, 1));
	btScalar det = (*this)[0].dot(co);
	//btFullAssert(det != btScalar(0.0));
	btAssert(det != btScalar(0.0));
	btScalar s = btScalar(1.0) / det;
	return btMatrix3x3(co.x() * s, cofac(0, 2, 2, 1) * s, cofac(0, 1, 1, 2) * s,
		co.y() * s, cofac(0, 0, 2, 2) * s, cofac(0, 2, 1, 0) * s,
		co.z() * s, cofac(0, 1, 2, 0) * s, cofac(0, 0, 1, 1) * s);
}

SIMD_FORCE_INLINE btMatrix3x3 
btMatrix3x3::transposeTimes(const btMatrix3x3& m) const
{
#if defined BT_USE_SIMD_VECTOR3 && (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))
    // zeros w
//    static const __m128i xyzMask = (const __m128i){ -1ULL, 0xffffffffULL };
    __m128 row = m_el[0].mVec128;
    __m128 m0 = _mm_and_ps( m.getRow(0).mVec128, btvFFF0fMask );
    __m128 m1 = _mm_and_ps( m.getRow(1).mVec128, btvFFF0fMask);
    __m128 m2 = _mm_and_ps( m.getRow(2).mVec128, btvFFF0fMask );
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
    return btMatrix3x3( r0, r1, r2 );

#elif defined BT_USE_NEON
    // zeros w
    static const uint32x4_t xyzMask = (const uint32x4_t){ static_cast<uint32_t>(-1), static_cast<uint32_t>(-1), static_cast<uint32_t>(-1), 0 };
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
    return btMatrix3x3( r0, r1, r2 );
#else
    return btMatrix3x3(
		m_el[0].x() * m[0].x() + m_el[1].x() * m[1].x() + m_el[2].x() * m[2].x(),
		m_el[0].x() * m[0].y() + m_el[1].x() * m[1].y() + m_el[2].x() * m[2].y(),
		m_el[0].x() * m[0].z() + m_el[1].x() * m[1].z() + m_el[2].x() * m[2].z(),
		m_el[0].y() * m[0].x() + m_el[1].y() * m[1].x() + m_el[2].y() * m[2].x(),
		m_el[0].y() * m[0].y() + m_el[1].y() * m[1].y() + m_el[2].y() * m[2].y(),
		m_el[0].y() * m[0].z() + m_el[1].y() * m[1].z() + m_el[2].y() * m[2].z(),
		m_el[0].z() * m[0].x() + m_el[1].z() * m[1].x() + m_el[2].z() * m[2].x(),
		m_el[0].z() * m[0].y() + m_el[1].z() * m[1].y() + m_el[2].z() * m[2].y(),
		m_el[0].z() * m[0].z() + m_el[1].z() * m[1].z() + m_el[2].z() * m[2].z());
#endif
}

SIMD_FORCE_INLINE btMatrix3x3 
btMatrix3x3::timesTranspose(const btMatrix3x3& m) const
{
#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))
    __m128 a0 = m_el[0].mVec128;
    __m128 a1 = m_el[1].mVec128;
    __m128 a2 = m_el[2].mVec128;
    
    btMatrix3x3 mT = m.transpose(); // we rely on transpose() zeroing w channel so that we don't have to do it here
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
    return btMatrix3x3( r0, r1, r2);
            
#elif defined BT_USE_NEON
    float32x4_t a0 = m_el[0].mVec128;
    float32x4_t a1 = m_el[1].mVec128;
    float32x4_t a2 = m_el[2].mVec128;
    
    btMatrix3x3 mT = m.transpose(); // we rely on transpose() zeroing w channel so that we don't have to do it here
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
    return btMatrix3x3( r0, r1, r2 );
    
#else
	return btMatrix3x3(
		m_el[0].dot(m[0]), m_el[0].dot(m[1]), m_el[0].dot(m[2]),
		m_el[1].dot(m[0]), m_el[1].dot(m[1]), m_el[1].dot(m[2]),
		m_el[2].dot(m[0]), m_el[2].dot(m[1]), m_el[2].dot(m[2]));
#endif
}

SIMD_FORCE_INLINE btVector3 
operator*(const btMatrix3x3& m, const btVector3& v) 
{
#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))|| defined (BT_USE_NEON)
    return v.dot3(m[0], m[1], m[2]);
#else
	return btVector3(m[0].dot(v), m[1].dot(v), m[2].dot(v));
#endif
}


SIMD_FORCE_INLINE btVector3
operator*(const btVector3& v, const btMatrix3x3& m)
{
#if defined BT_USE_SIMD_VECTOR3 && (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))

    const __m128 vv = v.mVec128;

    __m128 c0 = bt_splat_ps( vv, 0);
    __m128 c1 = bt_splat_ps( vv, 1);
    __m128 c2 = bt_splat_ps( vv, 2);

    c0 = _mm_mul_ps(c0, _mm_and_ps(m[0].mVec128, btvFFF0fMask) );
    c1 = _mm_mul_ps(c1, _mm_and_ps(m[1].mVec128, btvFFF0fMask) );
    c0 = _mm_add_ps(c0, c1);
    c2 = _mm_mul_ps(c2, _mm_and_ps(m[2].mVec128, btvFFF0fMask) );
    
    return btVector3(_mm_add_ps(c0, c2));
#elif defined(BT_USE_NEON)
    const float32x4_t vv = v.mVec128;
    const float32x2_t vlo = vget_low_f32(vv);
    const float32x2_t vhi = vget_high_f32(vv);

    float32x4_t c0, c1, c2;

    c0 = (float32x4_t) vandq_s32((int32x4_t)m[0].mVec128, btvFFF0Mask);
    c1 = (float32x4_t) vandq_s32((int32x4_t)m[1].mVec128, btvFFF0Mask);
    c2 = (float32x4_t) vandq_s32((int32x4_t)m[2].mVec128, btvFFF0Mask);

    c0 = vmulq_lane_f32(c0, vlo, 0);
    c1 = vmulq_lane_f32(c1, vlo, 1);
    c2 = vmulq_lane_f32(c2, vhi, 0);
    c0 = vaddq_f32(c0, c1);
    c0 = vaddq_f32(c0, c2);
    
    return btVector3(c0);
#else
	return btVector3(m.tdotx(v), m.tdoty(v), m.tdotz(v));
#endif
}

SIMD_FORCE_INLINE btMatrix3x3 
operator*(const btMatrix3x3& m1, const btMatrix3x3& m2)
{
#if defined BT_USE_SIMD_VECTOR3 && (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))

    __m128 m10 = m1[0].mVec128;  
    __m128 m11 = m1[1].mVec128;
    __m128 m12 = m1[2].mVec128;
    
    __m128 m2v = _mm_and_ps(m2[0].mVec128, btvFFF0fMask);
    
    __m128 c0 = bt_splat_ps( m10, 0);
    __m128 c1 = bt_splat_ps( m11, 0);
    __m128 c2 = bt_splat_ps( m12, 0);
    
    c0 = _mm_mul_ps(c0, m2v);
    c1 = _mm_mul_ps(c1, m2v);
    c2 = _mm_mul_ps(c2, m2v);
    
    m2v = _mm_and_ps(m2[1].mVec128, btvFFF0fMask);
    
    __m128 c0_1 = bt_splat_ps( m10, 1);
    __m128 c1_1 = bt_splat_ps( m11, 1);
    __m128 c2_1 = bt_splat_ps( m12, 1);
    
    c0_1 = _mm_mul_ps(c0_1, m2v);
    c1_1 = _mm_mul_ps(c1_1, m2v);
    c2_1 = _mm_mul_ps(c2_1, m2v);
    
    m2v = _mm_and_ps(m2[2].mVec128, btvFFF0fMask);
    
    c0 = _mm_add_ps(c0, c0_1);
    c1 = _mm_add_ps(c1, c1_1);
    c2 = _mm_add_ps(c2, c2_1);
    
    m10 = bt_splat_ps( m10, 2);
    m11 = bt_splat_ps( m11, 2);
    m12 = bt_splat_ps( m12, 2);
    
    m10 = _mm_mul_ps(m10, m2v);
    m11 = _mm_mul_ps(m11, m2v);
    m12 = _mm_mul_ps(m12, m2v);
    
    c0 = _mm_add_ps(c0, m10);
    c1 = _mm_add_ps(c1, m11);
    c2 = _mm_add_ps(c2, m12);
    
    return btMatrix3x3(c0, c1, c2);

#elif defined(BT_USE_NEON)

    float32x4_t rv0, rv1, rv2;
    float32x4_t v0, v1, v2;
    float32x4_t mv0, mv1, mv2;

    v0 = m1[0].mVec128;
    v1 = m1[1].mVec128;
    v2 = m1[2].mVec128;

    mv0 = (float32x4_t) vandq_s32((int32x4_t)m2[0].mVec128, btvFFF0Mask); 
    mv1 = (float32x4_t) vandq_s32((int32x4_t)m2[1].mVec128, btvFFF0Mask); 
    mv2 = (float32x4_t) vandq_s32((int32x4_t)m2[2].mVec128, btvFFF0Mask); 
    
    rv0 = vmulq_lane_f32(mv0, vget_low_f32(v0), 0);
    rv1 = vmulq_lane_f32(mv0, vget_low_f32(v1), 0);
    rv2 = vmulq_lane_f32(mv0, vget_low_f32(v2), 0);
    
    rv0 = vmlaq_lane_f32(rv0, mv1, vget_low_f32(v0), 1);
    rv1 = vmlaq_lane_f32(rv1, mv1, vget_low_f32(v1), 1);
    rv2 = vmlaq_lane_f32(rv2, mv1, vget_low_f32(v2), 1);
    
    rv0 = vmlaq_lane_f32(rv0, mv2, vget_high_f32(v0), 0);
    rv1 = vmlaq_lane_f32(rv1, mv2, vget_high_f32(v1), 0);
    rv2 = vmlaq_lane_f32(rv2, mv2, vget_high_f32(v2), 0);

	return btMatrix3x3(rv0, rv1, rv2);
        
#else	
	return btMatrix3x3(
		m2.tdotx( m1[0]), m2.tdoty( m1[0]), m2.tdotz( m1[0]),
		m2.tdotx( m1[1]), m2.tdoty( m1[1]), m2.tdotz( m1[1]),
		m2.tdotx( m1[2]), m2.tdoty( m1[2]), m2.tdotz( m1[2]));
#endif
}

/*
SIMD_FORCE_INLINE btMatrix3x3 btMultTransposeLeft(const btMatrix3x3& m1, const btMatrix3x3& m2) {
return btMatrix3x3(
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
SIMD_FORCE_INLINE bool operator==(const btMatrix3x3& m1, const btMatrix3x3& m2)
{
#if (defined (BT_USE_SSE_IN_API) && defined (BT_USE_SSE))

    __m128 c0, c1, c2;

    c0 = _mm_cmpeq_ps(m1[0].mVec128, m2[0].mVec128);
    c1 = _mm_cmpeq_ps(m1[1].mVec128, m2[1].mVec128);
    c2 = _mm_cmpeq_ps(m1[2].mVec128, m2[2].mVec128);
    
    c0 = _mm_and_ps(c0, c1);
    c0 = _mm_and_ps(c0, c2);

	int m = _mm_movemask_ps((__m128)c0);
	return (0x7 == (m & 0x7));
	
#else 
	return 
    (   m1[0][0] == m2[0][0] && m1[1][0] == m2[1][0] && m1[2][0] == m2[2][0] &&
		m1[0][1] == m2[0][1] && m1[1][1] == m2[1][1] && m1[2][1] == m2[2][1] &&
		m1[0][2] == m2[0][2] && m1[1][2] == m2[1][2] && m1[2][2] == m2[2][2] );
#endif
}

///for serialization
struct	btMatrix3x3FloatData
{
	btVector3FloatData m_el[3];
};

///for serialization
struct	btMatrix3x3DoubleData
{
	btVector3DoubleData m_el[3];
};


	

SIMD_FORCE_INLINE	void	btMatrix3x3::serialize(struct	btMatrix3x3Data& dataOut) const
{
	for (int i=0;i<3;i++)
		m_el[i].serialize(dataOut.m_el[i]);
}

SIMD_FORCE_INLINE	void	btMatrix3x3::serializeFloat(struct	btMatrix3x3FloatData& dataOut) const
{
	for (int i=0;i<3;i++)
		m_el[i].serializeFloat(dataOut.m_el[i]);
}


SIMD_FORCE_INLINE	void	btMatrix3x3::deSerialize(const struct	btMatrix3x3Data& dataIn)
{
	for (int i=0;i<3;i++)
		m_el[i].deSerialize(dataIn.m_el[i]);
}

SIMD_FORCE_INLINE	void	btMatrix3x3::deSerializeFloat(const struct	btMatrix3x3FloatData& dataIn)
{
	for (int i=0;i<3;i++)
		m_el[i].deSerializeFloat(dataIn.m_el[i]);
}

SIMD_FORCE_INLINE	void	btMatrix3x3::deSerializeDouble(const struct	btMatrix3x3DoubleData& dataIn)
{
	for (int i=0;i<3;i++)
		m_el[i].deSerializeDouble(dataIn.m_el[i]);
}

#endif //BT_MATRIX3x3_H


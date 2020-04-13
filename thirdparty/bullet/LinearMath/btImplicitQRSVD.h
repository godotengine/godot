/**
 Bullet Continuous Collision Detection and Physics Library
 Copyright (c) 2019 Google Inc. http://bulletphysics.org
 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it freely,
 subject to the following restrictions:
 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 
 Copyright (c) 2016 Theodore Gast, Chuyuan Fu, Chenfanfu Jiang, Joseph Teran
 
 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 If the code is used in an article, the following paper shall be cited:
 @techreport{qrsvd:2016,
 title={Implicit-shifted Symmetric QR Singular Value Decomposition of 3x3 Matrices},
 author={Gast, Theodore and Fu, Chuyuan and Jiang, Chenfanfu and Teran, Joseph},
 year={2016},
 institution={University of California Los Angeles}
 }
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
**/

#ifndef btImplicitQRSVD_h
#define btImplicitQRSVD_h

#include "btMatrix3x3.h"
class btMatrix2x2
{
public:
    btScalar m_00, m_01, m_10, m_11;
    btMatrix2x2(): m_00(0), m_10(0), m_01(0), m_11(0)
    {
    }
    btMatrix2x2(const btMatrix2x2& other): m_00(other.m_00),m_01(other.m_01),m_10(other.m_10),m_11(other.m_11)
    {}
    btScalar& operator()(int i, int j)
    {
        if (i == 0 && j == 0)
            return m_00;
        if (i == 1 && j == 0)
            return m_10;
        if (i == 0 && j == 1)
            return m_01;
        if (i == 1 && j == 1)
            return m_11;
        btAssert(false);
        return m_00;
    }
    const btScalar& operator()(int i, int j) const
    {
        if (i == 0 && j == 0)
            return m_00;
        if (i == 1 && j == 0)
            return m_10;
        if (i == 0 && j == 1)
            return m_01;
        if (i == 1 && j == 1)
            return m_11;
        btAssert(false);
        return m_00;
    }
    void setIdentity()
    {
        m_00 = 1;
        m_11 = 1;
        m_01 = 0;
        m_10 = 0;
    }
};

static inline btScalar copySign(btScalar x, btScalar y) {
    if ((x < 0 && y > 0) || (x > 0 && y < 0))
        return -x;
    return x;
}

/**
 Class for givens rotation.
 Row rotation G*A corresponds to something like
 c -s  0
 ( s  c  0 ) A
 0  0  1
 Column rotation A G' corresponds to something like
 c -s  0
 A ( s  c  0 )
 0  0  1
 
 c and s are always computed so that
 ( c -s ) ( a )  =  ( * )
 s  c     b       ( 0 )
 
 Assume rowi<rowk.
 */

class GivensRotation {
public:
    int rowi;
    int rowk;
    btScalar c;
    btScalar s;
    
    inline GivensRotation(int rowi_in, int rowk_in)
    : rowi(rowi_in)
    , rowk(rowk_in)
    , c(1)
    , s(0)
    {
    }
    
    inline GivensRotation(btScalar a, btScalar b, int rowi_in, int rowk_in)
    : rowi(rowi_in)
    , rowk(rowk_in)
    {
        compute(a, b);
    }
    
    ~GivensRotation() {}
    
    inline void transposeInPlace()
    {
        s = -s;
    }
    
    /**
     Compute c and s from a and b so that
     ( c -s ) ( a )  =  ( * )
     s  c     b       ( 0 )
     */
    inline void compute(const btScalar a, const btScalar b)
    {
        btScalar d = a * a + b * b;
        c = 1;
        s = 0;
        if (d > SIMD_EPSILON) {
            btScalar sqrtd = btSqrt(d);
            if (sqrtd>SIMD_EPSILON)
            {
              btScalar t = btScalar(1.0)/sqrtd;
              c = a * t;
              s = -b * t;
            }
        }
    }
    
    /**
     This function computes c and s so that
     ( c -s ) ( a )  =  ( 0 )
     s  c     b       ( * )
     */
    inline void computeUnconventional(const btScalar a, const btScalar b)
    {
        btScalar d = a * a + b * b;
        c = 0;
        s = 1;
        if (d > SIMD_EPSILON) {
            btScalar t = btScalar(1.0)/btSqrt(d);
            s = a * t;
            c = b * t;
        }
    }
    /**
     Fill the R with the entries of this rotation
     */
    inline void fill(const btMatrix3x3& R) const
    {
        btMatrix3x3& A = const_cast<btMatrix3x3&>(R);
        A.setIdentity();
        A[rowi][rowi] = c;
        A[rowk][rowi] = -s;
        A[rowi][rowk] = s;
        A[rowk][rowk] = c;
    }
    
    inline void fill(const btMatrix2x2& R) const
    {
        btMatrix2x2& A = const_cast<btMatrix2x2&>(R);
        A(rowi,rowi) = c;
        A(rowk,rowi) = -s;
        A(rowi,rowk) = s;
        A(rowk,rowk) = c;
    }
    
    /**
     This function does something like
     c -s  0
     ( s  c  0 ) A -> A
     0  0  1
     It only affects row i and row k of A.
     */
    inline void rowRotation(btMatrix3x3& A) const
    {
        for (int j = 0; j < 3; j++) {
            btScalar tau1 = A[rowi][j];
            btScalar tau2 = A[rowk][j];
            A[rowi][j] = c * tau1 - s * tau2;
            A[rowk][j] = s * tau1 + c * tau2;
        }
    }
    inline void rowRotation(btMatrix2x2& A) const
    {
        for (int j = 0; j < 2; j++) {
            btScalar tau1 = A(rowi,j);
            btScalar tau2 = A(rowk,j);
            A(rowi,j) = c * tau1 - s * tau2;
            A(rowk,j) = s * tau1 + c * tau2;
        }
    }
    
    /**
     This function does something like
     c  s  0
     A ( -s  c  0 )  -> A
     0  0  1
     It only affects column i and column k of A.
     */
    inline void columnRotation(btMatrix3x3& A) const
    {
        for (int j = 0; j < 3; j++) {
            btScalar tau1 = A[j][rowi];
            btScalar tau2 = A[j][rowk];
            A[j][rowi] = c * tau1 - s * tau2;
            A[j][rowk] = s * tau1 + c * tau2;
        }
    }
    inline void columnRotation(btMatrix2x2& A) const
    {
        for (int j = 0; j < 2; j++) {
            btScalar tau1 = A(j,rowi);
            btScalar tau2 = A(j,rowk);
            A(j,rowi) = c * tau1 - s * tau2;
            A(j,rowk) = s * tau1 + c * tau2;
        }
    }
    
    /**
     Multiply givens must be for same row and column
     **/
    inline void operator*=(const GivensRotation& A)
    {
        btScalar new_c = c * A.c - s * A.s;
        btScalar new_s = s * A.c + c * A.s;
        c = new_c;
        s = new_s;
    }
    
    /**
     Multiply givens must be for same row and column
     **/
    inline GivensRotation operator*(const GivensRotation& A) const
    {
        GivensRotation r(*this);
        r *= A;
        return r;
    }
};

/**
 \brief zero chasing the 3X3 matrix to bidiagonal form
 original form of H:   x x 0
 x x x
 0 0 x
 after zero chase:
 x x 0
 0 x x
 0 0 x
 */
inline void zeroChase(btMatrix3x3& H, btMatrix3x3& U, btMatrix3x3& V)
{
    
    /**
     Reduce H to of form
     x x +
     0 x x
     0 0 x
     */
    GivensRotation r1(H[0][0], H[1][0], 0, 1);
    /**
     Reduce H to of form
     x x 0
     0 x x
     0 + x
     Can calculate r2 without multiplying by r1 since both entries are in first two
     rows thus no need to divide by sqrt(a^2+b^2)
     */
    GivensRotation r2(1, 2);
    if (H[1][0] != 0)
        r2.compute(H[0][0] * H[0][1] + H[1][0] * H[1][1], H[0][0] * H[0][2] + H[1][0] * H[1][2]);
    else
        r2.compute(H[0][1], H[0][2]);
    
    r1.rowRotation(H);
    
    /* GivensRotation<T> r2(H(0, 1), H(0, 2), 1, 2); */
    r2.columnRotation(H);
    r2.columnRotation(V);
    
    /**
     Reduce H to of form
     x x 0
     0 x x
     0 0 x
     */
    GivensRotation r3(H[1][1], H[2][1], 1, 2);
    r3.rowRotation(H);
    
    // Save this till end for better cache coherency
    // r1.rowRotation(u_transpose);
    // r3.rowRotation(u_transpose);
    r1.columnRotation(U);
    r3.columnRotation(U);
}

/**
 \brief make a 3X3 matrix to upper bidiagonal form
 original form of H:   x x x
 x x x
 x x x
 after zero chase:
 x x 0
 0 x x
 0 0 x
 */
inline void makeUpperBidiag(btMatrix3x3& H, btMatrix3x3& U, btMatrix3x3& V)
{
    U.setIdentity();
    V.setIdentity();
    
    /**
     Reduce H to of form
     x x x
     x x x
     0 x x
     */
    
    GivensRotation r(H[1][0], H[2][0], 1, 2);
    r.rowRotation(H);
    // r.rowRotation(u_transpose);
    r.columnRotation(U);
    // zeroChase(H, u_transpose, V);
    zeroChase(H, U, V);
}

/**
 \brief make a 3X3 matrix to lambda shape
 original form of H:   x x x
 *                     x x x
 *                     x x x
 after :
 *                     x 0 0
 *                     x x 0
 *                     x 0 x
 */
inline void makeLambdaShape(btMatrix3x3& H, btMatrix3x3& U, btMatrix3x3& V)
{
    U.setIdentity();
    V.setIdentity();
    
    /**
     Reduce H to of form
     *                    x x 0
     *                    x x x
     *                    x x x
     */
    
    GivensRotation r1(H[0][1], H[0][2], 1, 2);
    r1.columnRotation(H);
    r1.columnRotation(V);
    
    /**
     Reduce H to of form
     *                    x x 0
     *                    x x 0
     *                    x x x
     */
    
    r1.computeUnconventional(H[1][2], H[2][2]);
    r1.rowRotation(H);
    r1.columnRotation(U);
    
    /**
     Reduce H to of form
     *                    x x 0
     *                    x x 0
     *                    x 0 x
     */
    
    GivensRotation r2(H[2][0], H[2][1], 0, 1);
    r2.columnRotation(H);
    r2.columnRotation(V);
    
    /**
     Reduce H to of form
     *                    x 0 0
     *                    x x 0
     *                    x 0 x
     */
    r2.computeUnconventional(H[0][1], H[1][1]);
    r2.rowRotation(H);
    r2.columnRotation(U);
}

/**
 \brief 2x2 polar decomposition.
 \param[in] A matrix.
 \param[out] R Robustly a rotation matrix.
 \param[out] S_Sym Symmetric. Whole matrix is stored
 
 Polar guarantees negative sign is on the small magnitude singular value.
 S is guaranteed to be the closest one to identity.
 R is guaranteed to be the closest rotation to A.
 */
inline void polarDecomposition(const btMatrix2x2& A,
                   GivensRotation& R,
                   const btMatrix2x2& S_Sym)
{
    btScalar a = (A(0, 0) + A(1, 1)),  b = (A(1, 0) - A(0, 1));
    btScalar denominator = btSqrt(a*a+b*b);
    R.c = (btScalar)1;
    R.s = (btScalar)0;
    if (denominator > SIMD_EPSILON) { 
        /*
         No need to use a tolerance here because x(0) and x(1) always have
         smaller magnitude then denominator, therefore overflow never happens.
         In Bullet, we use a tolerance anyway.
         */
        R.c = a / denominator;
        R.s = -b / denominator;
    }
    btMatrix2x2& S = const_cast<btMatrix2x2&>(S_Sym);
    S = A;
    R.rowRotation(S);
}

inline void polarDecomposition(const btMatrix2x2& A,
                   const btMatrix2x2& R,
                   const btMatrix2x2& S_Sym)
{
    GivensRotation r(0, 1);
    polarDecomposition(A, r, S_Sym);
    r.fill(R);
}

/**
 \brief 2x2 SVD (singular value decomposition) A=USV'
 \param[in] A Input matrix.
 \param[out] U Robustly a rotation matrix in Givens form
 \param[out] Sigma matrix of singular values sorted with decreasing magnitude. The second one can be negative.
 \param[out] V Robustly a rotation matrix in Givens form
 */
inline void singularValueDecomposition(
                           const btMatrix2x2& A,
                           GivensRotation& U,
                           const btMatrix2x2& Sigma,
                           GivensRotation& V,
                           const btScalar tol = 64 * std::numeric_limits<btScalar>::epsilon())
{
    btMatrix2x2& sigma = const_cast<btMatrix2x2&>(Sigma);
    sigma.setIdentity();
    btMatrix2x2 S_Sym;
    polarDecomposition(A, U, S_Sym);
    btScalar cosine, sine;
    btScalar x = S_Sym(0, 0);
    btScalar y = S_Sym(0, 1);
    btScalar z = S_Sym(1, 1);
    if (y == 0) {
        // S is already diagonal
        cosine = 1;
        sine = 0;
        sigma(0,0) = x;
        sigma(1,1) = z;
    }
    else {
        btScalar tau = 0.5 * (x - z);
        btScalar val = tau * tau + y * y;
        if (val > SIMD_EPSILON)
        {
        btScalar w = btSqrt(val);
        // w > y > 0
        btScalar t;
        if (tau > 0) {
            // tau + w > w > y > 0 ==> division is safe
            t = y / (tau + w);
        }
        else {
            // tau - w < -w < -y < 0 ==> division is safe
            t = y / (tau - w);
        }
        cosine = btScalar(1) / btSqrt(t * t + btScalar(1));
        sine = -t * cosine;
        /*
         V = [cosine -sine; sine cosine]
         Sigma = V'SV. Only compute the diagonals for efficiency.
         Also utilize symmetry of S and don't form V yet.
         */
        btScalar c2 = cosine * cosine;
        btScalar csy = 2 * cosine * sine * y;
        btScalar s2 = sine * sine;
        sigma(0,0) = c2 * x - csy + s2 * z;
        sigma(1,1) = s2 * x + csy + c2 * z;
      } else
      	{
      		cosine = 1;
        sine = 0;
        sigma(0,0) = x;
        sigma(1,1) = z;
      	}
    }
    
    // Sorting
    // Polar already guarantees negative sign is on the small magnitude singular value.
    if (sigma(0,0) < sigma(1,1)) {
        std::swap(sigma(0,0), sigma(1,1));
        V.c = -sine;
        V.s = cosine;
    }
    else {
        V.c = cosine;
        V.s = sine;
    }
    U *= V;
}

/**
 \brief 2x2 SVD (singular value decomposition) A=USV'
 \param[in] A Input matrix.
 \param[out] U Robustly a rotation matrix.
 \param[out] Sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
 \param[out] V Robustly a rotation matrix.
 */
inline void singularValueDecomposition(
                           const btMatrix2x2& A,
                           const btMatrix2x2& U,
                           const btMatrix2x2& Sigma,
                           const btMatrix2x2& V,
                           const btScalar tol = 64 * std::numeric_limits<btScalar>::epsilon())
{
    GivensRotation gv(0, 1);
    GivensRotation gu(0, 1);
    singularValueDecomposition(A, gu, Sigma, gv);
    
    gu.fill(U);
    gv.fill(V);
}

/**
 \brief compute wilkinsonShift of the block
 a1     b1
 b1     a2
 based on the wilkinsonShift formula
 mu = c + d - sign (d) \ sqrt (d*d + b*b), where d = (a-c)/2
 
 */
inline btScalar wilkinsonShift(const btScalar a1, const btScalar b1, const btScalar a2)
{
	btScalar d = (btScalar)0.5 * (a1 - a2);
	btScalar bs = b1 * b1;
	btScalar val = d * d + bs;
	if (val>SIMD_EPSILON)
	{
		btScalar denom = btFabs(d) + btSqrt(val);

		btScalar mu = a2 - copySign(bs / (denom), d);
		// T mu = a2 - bs / ( d + sign_d*sqrt (d*d + bs));
		return mu;
	}
	return a2;
}

/**
 \brief Helper function of 3X3 SVD for processing 2X2 SVD
 */
template <int t>
inline void process(btMatrix3x3& B, btMatrix3x3& U, btVector3& sigma, btMatrix3x3& V)
{
    int other = (t == 1) ? 0 : 2;
    GivensRotation u(0, 1);
    GivensRotation v(0, 1);
    sigma[other] = B[other][other];
    
    btMatrix2x2 B_sub, sigma_sub;
    if (t == 0)
    {
        B_sub.m_00 = B[0][0];
        B_sub.m_10 = B[1][0];
        B_sub.m_01 = B[0][1];
        B_sub.m_11 = B[1][1];
        sigma_sub.m_00 = sigma[0];
        sigma_sub.m_11 = sigma[1];
//        singularValueDecomposition(B.template block<2, 2>(t, t), u, sigma.template block<2, 1>(t, 0), v);
        singularValueDecomposition(B_sub, u, sigma_sub, v);
        B[0][0] = B_sub.m_00;
        B[1][0] = B_sub.m_10;
        B[0][1] = B_sub.m_01;
        B[1][1] = B_sub.m_11;
        sigma[0] = sigma_sub.m_00;
        sigma[1] = sigma_sub.m_11;
    }
    else
    {
        B_sub.m_00 = B[1][1];
        B_sub.m_10 = B[2][1];
        B_sub.m_01 = B[1][2];
        B_sub.m_11 = B[2][2];
        sigma_sub.m_00 = sigma[1];
        sigma_sub.m_11 = sigma[2];
        //        singularValueDecomposition(B.template block<2, 2>(t, t), u, sigma.template block<2, 1>(t, 0), v);
        singularValueDecomposition(B_sub, u, sigma_sub, v);
        B[1][1] = B_sub.m_00;
        B[2][1] = B_sub.m_10;
        B[1][2] = B_sub.m_01;
        B[2][2] = B_sub.m_11;
        sigma[1] = sigma_sub.m_00;
        sigma[2] = sigma_sub.m_11;
    }
    u.rowi += t;
    u.rowk += t;
    v.rowi += t;
    v.rowk += t;
    u.columnRotation(U);
    v.columnRotation(V);
}

/**
 \brief Helper function of 3X3 SVD for flipping signs due to flipping signs of sigma
 */
inline void flipSign(int i, btMatrix3x3& U, btVector3& sigma)
{
    sigma[i] = -sigma[i];
    U[0][i] = -U[0][i];
    U[1][i] = -U[1][i];
    U[2][i] = -U[2][i];
}

inline void flipSign(int i, btMatrix3x3& U)
{
    U[0][i] = -U[0][i];
    U[1][i] = -U[1][i];
    U[2][i] = -U[2][i];
}

inline void swapCol(btMatrix3x3& A, int i, int j)
{
    for (int d = 0; d < 3; ++d)
        std::swap(A[d][i], A[d][j]);
}
/**
 \brief Helper function of 3X3 SVD for sorting singular values
 */
inline void sort(btMatrix3x3& U, btVector3& sigma, btMatrix3x3& V, int t)
{
    if (t == 0)
    {
        // Case: sigma(0) > |sigma(1)| >= |sigma(2)|
        if (btFabs(sigma[1]) >= btFabs(sigma[2])) {
            if (sigma[1] < 0) {
                flipSign(1, U, sigma);
                flipSign(2, U, sigma);
            }
            return;
        }
        
        //fix sign of sigma for both cases
        if (sigma[2] < 0) {
            flipSign(1, U, sigma);
            flipSign(2, U, sigma);
        }
        
        //swap sigma(1) and sigma(2) for both cases
        std::swap(sigma[1], sigma[2]);
        // swap the col 1 and col 2 for U,V
        swapCol(U,1,2);
        swapCol(V,1,2);
        
        // Case: |sigma(2)| >= sigma(0) > |simga(1)|
        if (sigma[1] > sigma[0]) {
            std::swap(sigma[0], sigma[1]);
            swapCol(U,0,1);
            swapCol(V,0,1);
        }
        
        // Case: sigma(0) >= |sigma(2)| > |simga(1)|
        else {
            flipSign(2, U);
            flipSign(2, V);
        }
    }
    else if (t == 1)
    {
        // Case: |sigma(0)| >= sigma(1) > |sigma(2)|
        if (btFabs(sigma[0]) >= sigma[1]) {
            if (sigma[0] < 0) {
                flipSign(0, U, sigma);
                flipSign(2, U, sigma);
            }
            return;
        }
        
        //swap sigma(0) and sigma(1) for both cases
        std::swap(sigma[0], sigma[1]);
        swapCol(U, 0, 1);
        swapCol(V, 0, 1);
        
        // Case: sigma(1) > |sigma(2)| >= |sigma(0)|
        if (btFabs(sigma[1]) < btFabs(sigma[2])) {
            std::swap(sigma[1], sigma[2]);
            swapCol(U, 1, 2);
            swapCol(V, 1, 2);
        }
        
        // Case: sigma(1) >= |sigma(0)| > |sigma(2)|
        else {
            flipSign(1, U);
            flipSign(1, V);
        }
        
        // fix sign for both cases
        if (sigma[1] < 0) {
            flipSign(1, U, sigma);
            flipSign(2, U, sigma);
        }
    }
}

/**
 \brief 3X3 SVD (singular value decomposition) A=USV'
 \param[in] A Input matrix.
 \param[out] U is a rotation matrix.
 \param[out] sigma Diagonal matrix, sorted with decreasing magnitude. The third one can be negative.
 \param[out] V is a rotation matrix.
 */
inline int singularValueDecomposition(const btMatrix3x3& A,
                                     btMatrix3x3& U,
                                     btVector3& sigma,
                                     btMatrix3x3& V,
                                     btScalar tol = 128*std::numeric_limits<btScalar>::epsilon())
{
    using std::fabs;
    btMatrix3x3 B = A;
    U.setIdentity();
    V.setIdentity();
    
    makeUpperBidiag(B, U, V);
    
    int count = 0;
    btScalar mu = (btScalar)0;
    GivensRotation r(0, 1);
    
    btScalar alpha_1 = B[0][0];
    btScalar beta_1 = B[0][1];
    btScalar alpha_2 = B[1][1];
    btScalar alpha_3 = B[2][2];
    btScalar beta_2 = B[1][2];
    btScalar gamma_1 = alpha_1 * beta_1;
    btScalar gamma_2 = alpha_2 * beta_2;
    btScalar val = alpha_1 * alpha_1 + alpha_2 * alpha_2 + alpha_3 * alpha_3 + beta_1 * beta_1 + beta_2 * beta_2;
    if (val > SIMD_EPSILON)
    {
	    tol *= btMax((btScalar)0.5 * btSqrt(val), (btScalar)1);
		}    
    /**
     Do implicit shift QR until A^T A is block diagonal
     */
    int max_count = 100;
    
    while (btFabs(beta_2) > tol && btFabs(beta_1) > tol
           && btFabs(alpha_1) > tol && btFabs(alpha_2) > tol
           && btFabs(alpha_3) > tol
           && count < max_count) {
        mu = wilkinsonShift(alpha_2 * alpha_2 + beta_1 * beta_1, gamma_2, alpha_3 * alpha_3 + beta_2 * beta_2);
        
        r.compute(alpha_1 * alpha_1 - mu, gamma_1);
        r.columnRotation(B);
        
        r.columnRotation(V);
        zeroChase(B, U, V);
        
        alpha_1 = B[0][0];
        beta_1 = B[0][1];
        alpha_2 = B[1][1];
        alpha_3 = B[2][2];
        beta_2 = B[1][2];
        gamma_1 = alpha_1 * beta_1;
        gamma_2 = alpha_2 * beta_2;
        count++;
    }
    /**
     Handle the cases of one of the alphas and betas being 0
     Sorted by ease of handling and then frequency
     of occurrence
     
     If B is of form
     x x 0
     0 x 0
     0 0 x
     */
    if (btFabs(beta_2) <= tol) {
        process<0>(B, U, sigma, V);
        sort(U, sigma, V,0);
    }
    /**
     If B is of form
     x 0 0
     0 x x
     0 0 x
     */
    else if (btFabs(beta_1) <= tol) {
        process<1>(B, U, sigma, V);
        sort(U, sigma, V,1);
    }
    /**
     If B is of form
     x x 0
     0 0 x
     0 0 x
     */
    else if (btFabs(alpha_2) <= tol) {
        /**
         Reduce B to
         x x 0
         0 0 0
         0 0 x
         */
        GivensRotation r1(1, 2);
        r1.computeUnconventional(B[1][2], B[2][2]);
        r1.rowRotation(B);
        r1.columnRotation(U);
        
        process<0>(B, U, sigma, V);
        sort(U, sigma, V, 0);
    }
    /**
     If B is of form
     x x 0
     0 x x
     0 0 0
     */
    else if (btFabs(alpha_3) <= tol) {
        /**
         Reduce B to
         x x +
         0 x 0
         0 0 0
         */
        GivensRotation r1(1, 2);
        r1.compute(B[1][1], B[1][2]);
        r1.columnRotation(B);
        r1.columnRotation(V);
        /**
         Reduce B to
         x x 0
         + x 0
         0 0 0
         */
        GivensRotation r2(0, 2);
        r2.compute(B[0][0], B[0][2]);
        r2.columnRotation(B);
        r2.columnRotation(V);
        
        process<0>(B, U, sigma, V);
        sort(U, sigma, V, 0);
    }
    /**
     If B is of form
     0 x 0
     0 x x
     0 0 x
     */
    else if (btFabs(alpha_1) <= tol) {
        /**
         Reduce B to
         0 0 +
         0 x x
         0 0 x
         */
        GivensRotation r1(0, 1);
        r1.computeUnconventional(B[0][1], B[1][1]);
        r1.rowRotation(B);
        r1.columnRotation(U);
        
        /**
         Reduce B to
         0 0 0
         0 x x
         0 + x
         */
        GivensRotation r2(0, 2);
        r2.computeUnconventional(B[0][2], B[2][2]);
        r2.rowRotation(B);
        r2.columnRotation(U);
        
        process<1>(B, U, sigma, V);
        sort(U, sigma, V, 1);
    }
    
    return count;
}
#endif /* btImplicitQRSVD_h */

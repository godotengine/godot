// This code is in the public domain -- castanyo@yahoo.es

#include "Matrix.inl"
#include "Vector.inl"

#include "nvcore/Array.inl"

#include <float.h>

#if !NV_CC_MSVC && !NV_OS_ORBIS
#include <alloca.h>
#endif

using namespace nv;


// Given a matrix a[1..n][1..n], this routine replaces it by the LU decomposition of a rowwise
// permutation of itself. a and n are input. a is output, arranged as in equation (2.3.14) above;
// indx[1..n] is an output vector that records the row permutation effected by the partial
// pivoting; d is output as -1 depending on whether the number of row interchanges was even
// or odd, respectively. This routine is used in combination with lubksb to solve linear equations
// or invert a matrix.
static bool ludcmp(float **a, int n, int *indx, float *d)
{
    const float TINY = 1.0e-20f;

    float * vv = (float*)alloca(sizeof(float) * n);    // vv stores the implicit scaling of each row.

    *d = 1.0; // No row interchanges yet.
    for (int i = 0; i < n; i++) { // Loop over rows to get the implicit scaling information.
    
        float big = 0.0;
        for (int j = 0; j < n; j++) {
            big = max(big, fabsf(a[i][j]));
        }
        if (big == 0) {
            return false;   // Singular matrix
        }
        
        // No nonzero largest element.
        vv[i] = 1.0f / big; // Save the scaling.
    }

    for (int j = 0; j < n; j++) {       // This is the loop over columns of Crout's method.
        for (int i = 0; i < j; i++) {   // This is equation (2.3.12) except for i = j.
            float sum = a[i][j];
            for (int k = 0; k < i; k++) sum -= a[i][k]*a[k][j];
            a[i][j] = sum;
        }

        int imax = -1;
        float big = 0.0;                // Initialize for the search for largest pivot element.
        for (int i = j; i < n; i++) {   // This is i = j of equation (2.3.12) and i = j+ 1 : : : N
            float sum = a[i][j];              // of equation (2.3.13).
            for (int k = 0; k < j; k++) {
                sum -= a[i][k]*a[k][j];
            }
            a[i][j]=sum;

            float dum = vv[i]*fabs(sum);
            if (dum >= big) {
                // Is the figure of merit for the pivot better than the best so far?
                big = dum;
                imax = i;
            }
        }
        nvDebugCheck(imax != -1);

        if (j != imax) {                // Do we need to interchange rows?
            for (int k = 0; k < n; k++) {   // Yes, do so...
                swap(a[imax][k], a[j][k]);
            }
            *d = -(*d); // ...and change the parity of d.
            vv[imax]=vv[j]; // Also interchange the scale factor.
        }

        indx[j]=imax;
        if (a[j][j] == 0.0) a[j][j] = TINY;
        
        // If the pivot element is zero the matrix is singular (at least to the precision of the
        // algorithm). For some applications on singular matrices, it is desirable to substitute
        // TINY for zero.
        if (j != n-1) { // Now, finally, divide by the pivot element.
            float dum = 1.0f / a[j][j];
            for (int i = j+1; i < n; i++) a[i][j] *= dum;
        }
    } // Go back for the next column in the reduction.

    return true;
}


// Solves the set of n linear equations Ax = b. Here a[1..n][1..n] is input, not as the matrix
// A but rather as its LU decomposition, determined by the routine ludcmp. indx[1..n] is input
// as the permutation vector returned by ludcmp. b[1..n] is input as the right-hand side vector
// B, and returns with the solution vector X. a, n, and indx are not modified by this routine
// and can be left in place for successive calls with different right-hand sides b. This routine takes
// into account the possibility that b will begin with many zero elements, so it is efficient for use
// in matrix inversion.
static void lubksb(float **a, int n, int *indx, float b[])
{
    int ii = 0;
    for (int i=0; i<n; i++) {   // When ii is set to a positive value, it will become 
        int ip = indx[i];       // the index of the first nonvanishing element of b. We now 
        float sum = b[ip];      // do the forward substitution, equation (2.3.6). The 
        b[ip] = b[i];           // only new wrinkle is to unscramble the permutation as we go.
        if (ii != 0) {
            for (int j = ii-1; j < i; j++) sum -= a[i][j]*b[j];
        }
        else if (sum != 0.0f) {
            ii = i+1;             // A nonzero element was encountered, so from now on we 
        }
        b[i] = sum;             // will have to do the sums in the loop above.
    }
    for (int i=n-1; i>=0; i--) {  // Now we do the backsubstitution, equation (2.3.7).
        float sum = b[i];
        for (int j = i+1; j < n; j++) {
            sum -= a[i][j]*b[j];
        }
        b[i] = sum/a[i][i];     // Store a component of the solution vector X.
    } // All done!
}


bool nv::solveLU(const Matrix & A, const Vector4 & b, Vector4 * x)
{
    nvDebugCheck(x != NULL);

    float m[4][4];
    float *a[4] = {m[0], m[1], m[2], m[3]};
    int idx[4];
    float d;

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            a[x][y] = A(x, y);
        }
    }

    // Create LU decomposition.
    if (!ludcmp(a, 4, idx, &d)) {
        // Singular matrix.
        return false;
    }

    // Init solution.
    *x = b;

    // Do back substitution.
    lubksb(a, 4, idx, x->component);

    return true;
}

// @@ Not tested.
Matrix nv::inverseLU(const Matrix & A)
{
    Vector4 Ai[4];

    solveLU(A, Vector4(1, 0, 0, 0), &Ai[0]);
    solveLU(A, Vector4(0, 1, 0, 0), &Ai[1]);
    solveLU(A, Vector4(0, 0, 1, 0), &Ai[2]);
    solveLU(A, Vector4(0, 0, 0, 1), &Ai[3]);

    return Matrix(Ai[0], Ai[1], Ai[2], Ai[3]);
}



bool nv::solveLU(const Matrix3 & A, const Vector3 & b, Vector3 * x)
{
    nvDebugCheck(x != NULL);

    float m[3][3];
    float *a[3] = {m[0], m[1], m[2]};
    int idx[3];
    float d;

    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            a[x][y] = A(x, y);
        }
    }

    // Create LU decomposition.
    if (!ludcmp(a, 3, idx, &d)) {
        // Singular matrix.
        return false;
    }

    // Init solution.
    *x = b;

    // Do back substitution.
    lubksb(a, 3, idx, x->component);

    return true;
}


bool nv::solveCramer(const Matrix & A, const Vector4 & b, Vector4 * x)
{
    nvDebugCheck(x != NULL);

    *x = transform(inverseCramer(A), b);
    
    return true; // @@ Return false if determinant(A) == 0 !
}

bool nv::solveCramer(const Matrix3 & A, const Vector3 & b, Vector3 * x)
{
    nvDebugCheck(x != NULL);

    const float det = A.determinant();
    if (equal(det, 0.0f)) {   // @@ Use input epsilon.
        return false;
    }

    Matrix3 Ai = inverseCramer(A);

    *x = transform(Ai, b);
    
    return true;
}



// Inverse using gaussian elimination. From Jon's code.
Matrix nv::inverse(const Matrix & m) {

    Matrix A = m;
    Matrix B(identity);

    int i, j, k;
    float max, t, det, pivot;

    det = 1.0;
    for (i=0; i<4; i++) {               /* eliminate in column i, below diag */
        max = -1.;
        for (k=i; k<4; k++)             /* find pivot for column i */
            if (fabs(A(k, i)) > max) {
                max = fabs(A(k, i));
                j = k;
            }
        if (max<=0.) return B;         /* if no nonzero pivot, PUNT */
        if (j!=i) {                     /* swap rows i and j */
            for (k=i; k<4; k++)
                swap(A(i, k), A(j, k));
            for (k=0; k<4; k++)
                swap(B(i, k), B(j, k));
            det = -det;
        }
        pivot = A(i, i);
        det *= pivot;
        for (k=i+1; k<4; k++)           /* only do elems to right of pivot */
            A(i, k) /= pivot;
        for (k=0; k<4; k++)
            B(i, k) /= pivot;
        /* we know that A(i, i) will be set to 1, so don't bother to do it */

        for (j=i+1; j<4; j++) {         /* eliminate in rows below i */
            t = A(j, i);                /* we're gonna zero this guy */
            for (k=i+1; k<4; k++)       /* subtract scaled row i from row j */
                A(j, k) -= A(i, k)*t;   /* (ignore k<=i, we know they're 0) */
            for (k=0; k<4; k++)
                B(j, k) -= B(i, k)*t;
        }
    }

    /*---------- backward elimination ----------*/

    for (i=4-1; i>0; i--) {             /* eliminate in column i, above diag */
        for (j=0; j<i; j++) {           /* eliminate in rows above i */
            t = A(j, i);                /* we're gonna zero this guy */
            for (k=0; k<4; k++)         /* subtract scaled row i from row j */
                B(j, k) -= B(i, k)*t;
        }
    }

    return B;
}


Matrix3 nv::inverse(const Matrix3 & m) {

    Matrix3 A = m;
    Matrix3 B(identity);

    int i, j, k;
    float max, t, det, pivot;

    det = 1.0;
    for (i=0; i<3; i++) {               /* eliminate in column i, below diag */
        max = -1.;
        for (k=i; k<3; k++)             /* find pivot for column i */
            if (fabs(A(k, i)) > max) {
                max = fabs(A(k, i));
                j = k;
            }
        if (max<=0.) return B;         /* if no nonzero pivot, PUNT */
        if (j!=i) {                     /* swap rows i and j */
            for (k=i; k<3; k++)
                swap(A(i, k), A(j, k));
            for (k=0; k<3; k++)
                swap(B(i, k), B(j, k));
            det = -det;
        }
        pivot = A(i, i);
        det *= pivot;
        for (k=i+1; k<3; k++)           /* only do elems to right of pivot */
            A(i, k) /= pivot;
        for (k=0; k<3; k++)
            B(i, k) /= pivot;
        /* we know that A(i, i) will be set to 1, so don't bother to do it */

        for (j=i+1; j<3; j++) {         /* eliminate in rows below i */
            t = A(j, i);                /* we're gonna zero this guy */
            for (k=i+1; k<3; k++)       /* subtract scaled row i from row j */
                A(j, k) -= A(i, k)*t;   /* (ignore k<=i, we know they're 0) */
            for (k=0; k<3; k++)
                B(j, k) -= B(i, k)*t;
        }
    }

    /*---------- backward elimination ----------*/

    for (i=3-1; i>0; i--) {             /* eliminate in column i, above diag */
        for (j=0; j<i; j++) {           /* eliminate in rows above i */
            t = A(j, i);                /* we're gonna zero this guy */
            for (k=0; k<3; k++)         /* subtract scaled row i from row j */
                B(j, k) -= B(i, k)*t;
        }
    }

    return B;
}





#if 0 

// Copyright (C) 1999-2004 Michael Garland.
// 
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, and/or sell copies of the Software, and to permit persons
// to whom the Software is furnished to do so, provided that the above
// copyright notice(s) and this permission notice appear in all copies of
// the Software and that both the above copyright notice(s) and this
// permission notice appear in supporting documentation.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
// OF THIRD PARTY RIGHTS. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// HOLDERS INCLUDED IN THIS NOTICE BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL
// INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
// FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
// NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
// WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
// 
// Except as contained in this notice, the name of a copyright holder
// shall not be used in advertising or otherwise to promote the sale, use
// or other dealings in this Software without prior written authorization
// of the copyright holder.


// Matrix inversion code for 4x4 matrices using Gaussian elimination
// with partial pivoting.  This is a specialized version of a
// procedure originally due to Paul Heckbert <ph@cs.cmu.edu>.
//
// Returns determinant of A, and B=inverse(A)
// If matrix A is singular, returns 0 and leaves trash in B.
//
#define SWAP(a, b, t)   {t = a; a = b; b = t;}
double invert(Mat4& B, const Mat4& m)
{
    Mat4 A = m;
    int i, j, k;
    double max, t, det, pivot;

    /*---------- forward elimination ----------*/

    for (i=0; i<4; i++)                 /* put identity matrix in B */
        for (j=0; j<4; j++)
            B(i, j) = (double)(i==j);

    det = 1.0;
    for (i=0; i<4; i++) {               /* eliminate in column i, below diag */
        max = -1.;
        for (k=i; k<4; k++)             /* find pivot for column i */
            if (fabs(A(k, i)) > max) {
                max = fabs(A(k, i));
                j = k;
            }
        if (max<=0.) return 0.;         /* if no nonzero pivot, PUNT */
        if (j!=i) {                     /* swap rows i and j */
            for (k=i; k<4; k++)
                SWAP(A(i, k), A(j, k), t);
            for (k=0; k<4; k++)
                SWAP(B(i, k), B(j, k), t);
            det = -det;
        }
        pivot = A(i, i);
        det *= pivot;
        for (k=i+1; k<4; k++)           /* only do elems to right of pivot */
            A(i, k) /= pivot;
        for (k=0; k<4; k++)
            B(i, k) /= pivot;
        /* we know that A(i, i) will be set to 1, so don't bother to do it */

        for (j=i+1; j<4; j++) {         /* eliminate in rows below i */
            t = A(j, i);                /* we're gonna zero this guy */
            for (k=i+1; k<4; k++)       /* subtract scaled row i from row j */
                A(j, k) -= A(i, k)*t;   /* (ignore k<=i, we know they're 0) */
            for (k=0; k<4; k++)
                B(j, k) -= B(i, k)*t;
        }
    }

    /*---------- backward elimination ----------*/

    for (i=4-1; i>0; i--) {             /* eliminate in column i, above diag */
        for (j=0; j<i; j++) {           /* eliminate in rows above i */
            t = A(j, i);                /* we're gonna zero this guy */
            for (k=0; k<4; k++)         /* subtract scaled row i from row j */
                B(j, k) -= B(i, k)*t;
        }
    }

    return det;
}

#endif // 0




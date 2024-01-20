/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to LAPACKe
 *     LU decomposition with partial pivoting based on LAPACKE_?getrf function.
 ********************************************************************************
*/

#ifndef EIGEN_PARTIALLU_LAPACK_H
#define EIGEN_PARTIALLU_LAPACK_H

namespace Eigen { 

namespace internal {

/** \internal Specialization for the data types supported by LAPACKe */

#define EIGEN_LAPACKE_LU_PARTPIV(EIGTYPE, LAPACKE_TYPE, LAPACKE_PREFIX) \
template<int StorageOrder> \
struct partial_lu_impl<EIGTYPE, StorageOrder, lapack_int> \
{ \
  /* \internal performs the LU decomposition in-place of the matrix represented */ \
  static lapack_int blocked_lu(Index rows, Index cols, EIGTYPE* lu_data, Index luStride, lapack_int* row_transpositions, lapack_int& nb_transpositions, lapack_int maxBlockSize=256) \
  { \
    EIGEN_UNUSED_VARIABLE(maxBlockSize);\
    lapack_int matrix_order, first_zero_pivot; \
    lapack_int m, n, lda, *ipiv, info; \
    EIGTYPE* a; \
/* Set up parameters for ?getrf */ \
    matrix_order = StorageOrder==RowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR; \
    lda = convert_index<lapack_int>(luStride); \
    a = lu_data; \
    ipiv = row_transpositions; \
    m = convert_index<lapack_int>(rows); \
    n = convert_index<lapack_int>(cols); \
    nb_transpositions = 0; \
\
    info = LAPACKE_##LAPACKE_PREFIX##getrf( matrix_order, m, n, (LAPACKE_TYPE*)a, lda, ipiv ); \
\
    for(int i=0;i<m;i++) { ipiv[i]--; if (ipiv[i]!=i) nb_transpositions++; } \
\
    eigen_assert(info >= 0); \
/* something should be done with nb_transpositions */ \
\
    first_zero_pivot = info; \
    return first_zero_pivot; \
  } \
};

EIGEN_LAPACKE_LU_PARTPIV(double, double, d)
EIGEN_LAPACKE_LU_PARTPIV(float, float, s)
EIGEN_LAPACKE_LU_PARTPIV(dcomplex, lapack_complex_double, z)
EIGEN_LAPACKE_LU_PARTPIV(scomplex, lapack_complex_float,  c)

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PARTIALLU_LAPACK_H

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
 *    Householder QR decomposition of a matrix with column pivoting based on
 *    LAPACKE_?geqp3 function.
 ********************************************************************************
*/

#ifndef EIGEN_COLPIVOTINGHOUSEHOLDERQR_LAPACKE_H
#define EIGEN_COLPIVOTINGHOUSEHOLDERQR_LAPACKE_H

namespace Eigen { 

/** \internal Specialization for the data types supported by LAPACKe */

#define EIGEN_LAPACKE_QR_COLPIV(EIGTYPE, LAPACKE_TYPE, LAPACKE_PREFIX, EIGCOLROW, LAPACKE_COLROW) \
template<> template<typename InputType> inline \
ColPivHouseholderQR<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> >& \
ColPivHouseholderQR<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> >::compute( \
              const EigenBase<InputType>& matrix) \
\
{ \
  using std::abs; \
  typedef Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> MatrixType; \
  typedef MatrixType::RealScalar RealScalar; \
  Index rows = matrix.rows();\
  Index cols = matrix.cols();\
\
  m_qr = matrix;\
  Index size = m_qr.diagonalSize();\
  m_hCoeffs.resize(size);\
\
  m_colsTranspositions.resize(cols);\
  /*Index number_of_transpositions = 0;*/ \
\
  m_nonzero_pivots = 0; \
  m_maxpivot = RealScalar(0);\
  m_colsPermutation.resize(cols); \
  m_colsPermutation.indices().setZero(); \
\
  lapack_int lda = internal::convert_index<lapack_int,Index>(m_qr.outerStride()); \
  lapack_int matrix_order = LAPACKE_COLROW; \
  LAPACKE_##LAPACKE_PREFIX##geqp3( matrix_order, internal::convert_index<lapack_int,Index>(rows), internal::convert_index<lapack_int,Index>(cols), \
                              (LAPACKE_TYPE*)m_qr.data(), lda, (lapack_int*)m_colsPermutation.indices().data(), (LAPACKE_TYPE*)m_hCoeffs.data()); \
  m_isInitialized = true; \
  m_maxpivot=m_qr.diagonal().cwiseAbs().maxCoeff(); \
  m_hCoeffs.adjointInPlace(); \
  RealScalar premultiplied_threshold = abs(m_maxpivot) * threshold(); \
  lapack_int *perm = m_colsPermutation.indices().data(); \
  for(Index i=0;i<size;i++) { \
    m_nonzero_pivots += (abs(m_qr.coeff(i,i)) > premultiplied_threshold);\
  } \
  for(Index i=0;i<cols;i++) perm[i]--;\
\
  /*m_det_pq = (number_of_transpositions%2) ? -1 : 1;  // TODO: It's not needed now; fix upon availability in Eigen */ \
\
  return *this; \
}

EIGEN_LAPACKE_QR_COLPIV(double,   double,        d, ColMajor, LAPACK_COL_MAJOR)
EIGEN_LAPACKE_QR_COLPIV(float,    float,         s, ColMajor, LAPACK_COL_MAJOR)
EIGEN_LAPACKE_QR_COLPIV(dcomplex, lapack_complex_double, z, ColMajor, LAPACK_COL_MAJOR)
EIGEN_LAPACKE_QR_COLPIV(scomplex, lapack_complex_float,  c, ColMajor, LAPACK_COL_MAJOR)

EIGEN_LAPACKE_QR_COLPIV(double,   double,        d, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_LAPACKE_QR_COLPIV(float,    float,         s, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_LAPACKE_QR_COLPIV(dcomplex, lapack_complex_double, z, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_LAPACKE_QR_COLPIV(scomplex, lapack_complex_float,  c, RowMajor, LAPACK_ROW_MAJOR)

} // end namespace Eigen

#endif // EIGEN_COLPIVOTINGHOUSEHOLDERQR_LAPACKE_H

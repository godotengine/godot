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
 *    Self-adjoint eigenvalues/eigenvectors.
 ********************************************************************************
*/

#ifndef EIGEN_SAEIGENSOLVER_LAPACKE_H
#define EIGEN_SAEIGENSOLVER_LAPACKE_H

namespace Eigen { 

/** \internal Specialization for the data types supported by LAPACKe */

#define EIGEN_LAPACKE_EIG_SELFADJ(EIGTYPE, LAPACKE_TYPE, LAPACKE_RTYPE, LAPACKE_NAME, EIGCOLROW, LAPACKE_COLROW ) \
template<> template<typename InputType> inline \
SelfAdjointEigenSolver<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >& \
SelfAdjointEigenSolver<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >::compute(const EigenBase<InputType>& matrix, int options) \
{ \
  eigen_assert(matrix.cols() == matrix.rows()); \
  eigen_assert((options&~(EigVecMask|GenEigMask))==0 \
          && (options&EigVecMask)!=EigVecMask \
          && "invalid option parameter"); \
  bool computeEigenvectors = (options&ComputeEigenvectors)==ComputeEigenvectors; \
  lapack_int n = internal::convert_index<lapack_int>(matrix.cols()), lda, matrix_order, info; \
  m_eivalues.resize(n,1); \
  m_subdiag.resize(n-1); \
  m_eivec = matrix; \
\
  if(n==1) \
  { \
    m_eivalues.coeffRef(0,0) = numext::real(m_eivec.coeff(0,0)); \
    if(computeEigenvectors) m_eivec.setOnes(n,n); \
    m_info = Success; \
    m_isInitialized = true; \
    m_eigenvectorsOk = computeEigenvectors; \
    return *this; \
  } \
\
  lda = internal::convert_index<lapack_int>(m_eivec.outerStride()); \
  matrix_order=LAPACKE_COLROW; \
  char jobz, uplo='L'/*, range='A'*/; \
  jobz = computeEigenvectors ? 'V' : 'N'; \
\
  info = LAPACKE_##LAPACKE_NAME( matrix_order, jobz, uplo, n, (LAPACKE_TYPE*)m_eivec.data(), lda, (LAPACKE_RTYPE*)m_eivalues.data() ); \
  m_info = (info==0) ? Success : NoConvergence; \
  m_isInitialized = true; \
  m_eigenvectorsOk = computeEigenvectors; \
  return *this; \
}


EIGEN_LAPACKE_EIG_SELFADJ(double,   double,                double, dsyev, ColMajor, LAPACK_COL_MAJOR)
EIGEN_LAPACKE_EIG_SELFADJ(float,    float,                 float,  ssyev, ColMajor, LAPACK_COL_MAJOR)
EIGEN_LAPACKE_EIG_SELFADJ(dcomplex, lapack_complex_double, double, zheev, ColMajor, LAPACK_COL_MAJOR)
EIGEN_LAPACKE_EIG_SELFADJ(scomplex, lapack_complex_float,  float,  cheev, ColMajor, LAPACK_COL_MAJOR)

EIGEN_LAPACKE_EIG_SELFADJ(double,   double,                double, dsyev, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_LAPACKE_EIG_SELFADJ(float,    float,                 float,  ssyev, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_LAPACKE_EIG_SELFADJ(dcomplex, lapack_complex_double, double, zheev, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_LAPACKE_EIG_SELFADJ(scomplex, lapack_complex_float,  float,  cheev, RowMajor, LAPACK_ROW_MAJOR)

} // end namespace Eigen

#endif // EIGEN_SAEIGENSOLVER_H

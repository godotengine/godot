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
 *   Content : Eigen bindings to BLAS F77
 *   Selfadjoint matrix-vector product functionality based on ?SYMV/HEMV.
 ********************************************************************************
*/

#ifndef EIGEN_SELFADJOINT_MATRIX_VECTOR_BLAS_H
#define EIGEN_SELFADJOINT_MATRIX_VECTOR_BLAS_H

namespace Eigen { 

namespace internal {

/**********************************************************************
* This file implements selfadjoint matrix-vector multiplication using BLAS
**********************************************************************/

// symv/hemv specialization

template<typename Scalar, typename Index, int StorageOrder, int UpLo, bool ConjugateLhs, bool ConjugateRhs>
struct selfadjoint_matrix_vector_product_symv :
  selfadjoint_matrix_vector_product<Scalar,Index,StorageOrder,UpLo,ConjugateLhs,ConjugateRhs,BuiltIn> {};

#define EIGEN_BLAS_SYMV_SPECIALIZE(Scalar) \
template<typename Index, int StorageOrder, int UpLo, bool ConjugateLhs, bool ConjugateRhs> \
struct selfadjoint_matrix_vector_product<Scalar,Index,StorageOrder,UpLo,ConjugateLhs,ConjugateRhs,Specialized> { \
static void run( \
  Index size, const Scalar*  lhs, Index lhsStride, \
  const Scalar* _rhs, Scalar* res, Scalar alpha) { \
    enum {\
      IsColMajor = StorageOrder==ColMajor \
    }; \
    if (IsColMajor == ConjugateLhs) {\
      selfadjoint_matrix_vector_product<Scalar,Index,StorageOrder,UpLo,ConjugateLhs,ConjugateRhs,BuiltIn>::run( \
        size, lhs, lhsStride, _rhs, res, alpha);  \
    } else {\
      selfadjoint_matrix_vector_product_symv<Scalar,Index,StorageOrder,UpLo,ConjugateLhs,ConjugateRhs>::run( \
        size, lhs, lhsStride, _rhs, res, alpha);  \
    }\
  } \
}; \

EIGEN_BLAS_SYMV_SPECIALIZE(double)
EIGEN_BLAS_SYMV_SPECIALIZE(float)
EIGEN_BLAS_SYMV_SPECIALIZE(dcomplex)
EIGEN_BLAS_SYMV_SPECIALIZE(scomplex)

#define EIGEN_BLAS_SYMV_SPECIALIZATION(EIGTYPE,BLASTYPE,BLASFUNC) \
template<typename Index, int StorageOrder, int UpLo, bool ConjugateLhs, bool ConjugateRhs> \
struct selfadjoint_matrix_vector_product_symv<EIGTYPE,Index,StorageOrder,UpLo,ConjugateLhs,ConjugateRhs> \
{ \
typedef Matrix<EIGTYPE,Dynamic,1,ColMajor> SYMVVector;\
\
static void run( \
Index size, const EIGTYPE*  lhs, Index lhsStride, \
const EIGTYPE* _rhs, EIGTYPE* res, EIGTYPE alpha) \
{ \
  enum {\
    IsRowMajor = StorageOrder==RowMajor ? 1 : 0, \
    IsLower = UpLo == Lower ? 1 : 0 \
  }; \
  BlasIndex n=convert_index<BlasIndex>(size), lda=convert_index<BlasIndex>(lhsStride), incx=1, incy=1; \
  EIGTYPE beta(1); \
  const EIGTYPE *x_ptr; \
  char uplo=(IsRowMajor) ? (IsLower ? 'U' : 'L') : (IsLower ? 'L' : 'U'); \
  SYMVVector x_tmp; \
  if (ConjugateRhs) { \
    Map<const SYMVVector, 0 > map_x(_rhs,size,1); \
    x_tmp=map_x.conjugate(); \
    x_ptr=x_tmp.data(); \
  } else x_ptr=_rhs; \
  BLASFUNC(&uplo, &n, (const BLASTYPE*)&numext::real_ref(alpha), (const BLASTYPE*)lhs, &lda, (const BLASTYPE*)x_ptr, &incx, (const BLASTYPE*)&numext::real_ref(beta), (BLASTYPE*)res, &incy); \
}\
};

#ifdef EIGEN_USE_MKL
EIGEN_BLAS_SYMV_SPECIALIZATION(double,   double, dsymv)
EIGEN_BLAS_SYMV_SPECIALIZATION(float,    float,  ssymv)
EIGEN_BLAS_SYMV_SPECIALIZATION(dcomplex, MKL_Complex16, zhemv)
EIGEN_BLAS_SYMV_SPECIALIZATION(scomplex, MKL_Complex8,  chemv)
#else
EIGEN_BLAS_SYMV_SPECIALIZATION(double,   double, dsymv_)
EIGEN_BLAS_SYMV_SPECIALIZATION(float,    float,  ssymv_)
EIGEN_BLAS_SYMV_SPECIALIZATION(dcomplex, double, zhemv_)
EIGEN_BLAS_SYMV_SPECIALIZATION(scomplex, float,  chemv_)
#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_SELFADJOINT_MATRIX_VECTOR_BLAS_H

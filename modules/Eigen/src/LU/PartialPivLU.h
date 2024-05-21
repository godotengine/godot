// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PARTIALLU_H
#define EIGEN_PARTIALLU_H

namespace Eigen {

namespace internal {
template<typename _MatrixType> struct traits<PartialPivLU<_MatrixType> >
 : traits<_MatrixType>
{
  typedef MatrixXpr XprKind;
  typedef SolverStorage StorageKind;
  typedef traits<_MatrixType> BaseTraits;
  enum {
    Flags = BaseTraits::Flags & RowMajorBit,
    CoeffReadCost = Dynamic
  };
};

template<typename T,typename Derived>
struct enable_if_ref;
// {
//   typedef Derived type;
// };

template<typename T,typename Derived>
struct enable_if_ref<Ref<T>,Derived> {
  typedef Derived type;
};

} // end namespace internal

/** \ingroup LU_Module
  *
  * \class PartialPivLU
  *
  * \brief LU decomposition of a matrix with partial pivoting, and related features
  *
  * \tparam _MatrixType the type of the matrix of which we are computing the LU decomposition
  *
  * This class represents a LU decomposition of a \b square \b invertible matrix, with partial pivoting: the matrix A
  * is decomposed as A = PLU where L is unit-lower-triangular, U is upper-triangular, and P
  * is a permutation matrix.
  *
  * Typically, partial pivoting LU decomposition is only considered numerically stable for square invertible
  * matrices. Thus LAPACK's dgesv and dgesvx require the matrix to be square and invertible. The present class
  * does the same. It will assert that the matrix is square, but it won't (actually it can't) check that the
  * matrix is invertible: it is your task to check that you only use this decomposition on invertible matrices.
  *
  * The guaranteed safe alternative, working for all matrices, is the full pivoting LU decomposition, provided
  * by class FullPivLU.
  *
  * This is \b not a rank-revealing LU decomposition. Many features are intentionally absent from this class,
  * such as rank computation. If you need these features, use class FullPivLU.
  *
  * This LU decomposition is suitable to invert invertible matrices. It is what MatrixBase::inverse() uses
  * in the general case.
  * On the other hand, it is \b not suitable to determine whether a given matrix is invertible.
  *
  * The data of the LU decomposition can be directly accessed through the methods matrixLU(), permutationP().
  *
  * This class supports the \link InplaceDecomposition inplace decomposition \endlink mechanism.
  * 
  * \sa MatrixBase::partialPivLu(), MatrixBase::determinant(), MatrixBase::inverse(), MatrixBase::computeInverse(), class FullPivLU
  */
template<typename _MatrixType> class PartialPivLU
  : public SolverBase<PartialPivLU<_MatrixType> >
{
  public:

    typedef _MatrixType MatrixType;
    typedef SolverBase<PartialPivLU> Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE(PartialPivLU)
    // FIXME StorageIndex defined in EIGEN_GENERIC_PUBLIC_INTERFACE should be int
    enum {
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    typedef PermutationMatrix<RowsAtCompileTime, MaxRowsAtCompileTime> PermutationType;
    typedef Transpositions<RowsAtCompileTime, MaxRowsAtCompileTime> TranspositionType;
    typedef typename MatrixType::PlainObject PlainObject;

    /**
      * \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via PartialPivLU::compute(const MatrixType&).
      */
    PartialPivLU();

    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa PartialPivLU()
      */
    explicit PartialPivLU(Index size);

    /** Constructor.
      *
      * \param matrix the matrix of which to compute the LU decomposition.
      *
      * \warning The matrix should have full rank (e.g. if it's square, it should be invertible).
      * If you need to deal with non-full rank, use class FullPivLU instead.
      */
    template<typename InputType>
    explicit PartialPivLU(const EigenBase<InputType>& matrix);

    /** Constructor for \link InplaceDecomposition inplace decomposition \endlink
      *
      * \param matrix the matrix of which to compute the LU decomposition.
      *
      * \warning The matrix should have full rank (e.g. if it's square, it should be invertible).
      * If you need to deal with non-full rank, use class FullPivLU instead.
      */
    template<typename InputType>
    explicit PartialPivLU(EigenBase<InputType>& matrix);

    template<typename InputType>
    PartialPivLU& compute(const EigenBase<InputType>& matrix) {
      m_lu = matrix.derived();
      compute();
      return *this;
    }

    /** \returns the LU decomposition matrix: the upper-triangular part is U, the
      * unit-lower-triangular part is L (at least for square matrices; in the non-square
      * case, special care is needed, see the documentation of class FullPivLU).
      *
      * \sa matrixL(), matrixU()
      */
    inline const MatrixType& matrixLU() const
    {
      eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
      return m_lu;
    }

    /** \returns the permutation matrix P.
      */
    inline const PermutationType& permutationP() const
    {
      eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
      return m_p;
    }

    /** This method returns the solution x to the equation Ax=b, where A is the matrix of which
      * *this is the LU decomposition.
      *
      * \param b the right-hand-side of the equation to solve. Can be a vector or a matrix,
      *          the only requirement in order for the equation to make sense is that
      *          b.rows()==A.rows(), where A is the matrix of which *this is the LU decomposition.
      *
      * \returns the solution.
      *
      * Example: \include PartialPivLU_solve.cpp
      * Output: \verbinclude PartialPivLU_solve.out
      *
      * Since this PartialPivLU class assumes anyway that the matrix A is invertible, the solution
      * theoretically exists and is unique regardless of b.
      *
      * \sa TriangularView::solve(), inverse(), computeInverse()
      */
    // FIXME this is a copy-paste of the base-class member to add the isInitialized assertion.
    template<typename Rhs>
    inline const Solve<PartialPivLU, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
      return Solve<PartialPivLU, Rhs>(*this, b.derived());
    }

    /** \returns an estimate of the reciprocal condition number of the matrix of which \c *this is
        the LU decomposition.
      */
    inline RealScalar rcond() const
    {
      eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
      return internal::rcond_estimate_helper(m_l1_norm, *this);
    }

    /** \returns the inverse of the matrix of which *this is the LU decomposition.
      *
      * \warning The matrix being decomposed here is assumed to be invertible. If you need to check for
      *          invertibility, use class FullPivLU instead.
      *
      * \sa MatrixBase::inverse(), LU::inverse()
      */
    inline const Inverse<PartialPivLU> inverse() const
    {
      eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
      return Inverse<PartialPivLU>(*this);
    }

    /** \returns the determinant of the matrix of which
      * *this is the LU decomposition. It has only linear complexity
      * (that is, O(n) where n is the dimension of the square matrix)
      * as the LU decomposition has already been computed.
      *
      * \note For fixed-size matrices of size up to 4, MatrixBase::determinant() offers
      *       optimized paths.
      *
      * \warning a determinant can be very big or small, so for matrices
      * of large enough dimension, there is a risk of overflow/underflow.
      *
      * \sa MatrixBase::determinant()
      */
    Scalar determinant() const;

    MatrixType reconstructedMatrix() const;

    inline Index rows() const { return m_lu.rows(); }
    inline Index cols() const { return m_lu.cols(); }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename RhsType, typename DstType>
    EIGEN_DEVICE_FUNC
    void _solve_impl(const RhsType &rhs, DstType &dst) const {
     /* The decomposition PA = LU can be rewritten as A = P^{-1} L U.
      * So we proceed as follows:
      * Step 1: compute c = Pb.
      * Step 2: replace c by the solution x to Lx = c.
      * Step 3: replace c by the solution x to Ux = c.
      */

      eigen_assert(rhs.rows() == m_lu.rows());

      // Step 1
      dst = permutationP() * rhs;

      // Step 2
      m_lu.template triangularView<UnitLower>().solveInPlace(dst);

      // Step 3
      m_lu.template triangularView<Upper>().solveInPlace(dst);
    }

    template<bool Conjugate, typename RhsType, typename DstType>
    EIGEN_DEVICE_FUNC
    void _solve_impl_transposed(const RhsType &rhs, DstType &dst) const {
     /* The decomposition PA = LU can be rewritten as A = P^{-1} L U.
      * So we proceed as follows:
      * Step 1: compute c = Pb.
      * Step 2: replace c by the solution x to Lx = c.
      * Step 3: replace c by the solution x to Ux = c.
      */

      eigen_assert(rhs.rows() == m_lu.cols());

      if (Conjugate) {
        // Step 1
        dst = m_lu.template triangularView<Upper>().adjoint().solve(rhs);
        // Step 2
        m_lu.template triangularView<UnitLower>().adjoint().solveInPlace(dst);
      } else {
        // Step 1
        dst = m_lu.template triangularView<Upper>().transpose().solve(rhs);
        // Step 2
        m_lu.template triangularView<UnitLower>().transpose().solveInPlace(dst);
      }
      // Step 3
      dst = permutationP().transpose() * dst;
    }
    #endif

  protected:

    static void check_template_parameters()
    {
      EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar);
    }

    void compute();

    MatrixType m_lu;
    PermutationType m_p;
    TranspositionType m_rowsTranspositions;
    RealScalar m_l1_norm;
    signed char m_det_p;
    bool m_isInitialized;
};

template<typename MatrixType>
PartialPivLU<MatrixType>::PartialPivLU()
  : m_lu(),
    m_p(),
    m_rowsTranspositions(),
    m_l1_norm(0),
    m_det_p(0),
    m_isInitialized(false)
{
}

template<typename MatrixType>
PartialPivLU<MatrixType>::PartialPivLU(Index size)
  : m_lu(size, size),
    m_p(size),
    m_rowsTranspositions(size),
    m_l1_norm(0),
    m_det_p(0),
    m_isInitialized(false)
{
}

template<typename MatrixType>
template<typename InputType>
PartialPivLU<MatrixType>::PartialPivLU(const EigenBase<InputType>& matrix)
  : m_lu(matrix.rows(),matrix.cols()),
    m_p(matrix.rows()),
    m_rowsTranspositions(matrix.rows()),
    m_l1_norm(0),
    m_det_p(0),
    m_isInitialized(false)
{
  compute(matrix.derived());
}

template<typename MatrixType>
template<typename InputType>
PartialPivLU<MatrixType>::PartialPivLU(EigenBase<InputType>& matrix)
  : m_lu(matrix.derived()),
    m_p(matrix.rows()),
    m_rowsTranspositions(matrix.rows()),
    m_l1_norm(0),
    m_det_p(0),
    m_isInitialized(false)
{
  compute();
}

namespace internal {

/** \internal This is the blocked version of fullpivlu_unblocked() */
template<typename Scalar, int StorageOrder, typename PivIndex>
struct partial_lu_impl
{
  // FIXME add a stride to Map, so that the following mapping becomes easier,
  // another option would be to create an expression being able to automatically
  // warp any Map, Matrix, and Block expressions as a unique type, but since that's exactly
  // a Map + stride, why not adding a stride to Map, and convenient ctors from a Matrix,
  // and Block.
  typedef Map<Matrix<Scalar, Dynamic, Dynamic, StorageOrder> > MapLU;
  typedef Block<MapLU, Dynamic, Dynamic> MatrixType;
  typedef Block<MatrixType,Dynamic,Dynamic> BlockType;
  typedef typename MatrixType::RealScalar RealScalar;

  /** \internal performs the LU decomposition in-place of the matrix \a lu
    * using an unblocked algorithm.
    *
    * In addition, this function returns the row transpositions in the
    * vector \a row_transpositions which must have a size equal to the number
    * of columns of the matrix \a lu, and an integer \a nb_transpositions
    * which returns the actual number of transpositions.
    *
    * \returns The index of the first pivot which is exactly zero if any, or a negative number otherwise.
    */
  static Index unblocked_lu(MatrixType& lu, PivIndex* row_transpositions, PivIndex& nb_transpositions)
  {
    typedef scalar_score_coeff_op<Scalar> Scoring;
    typedef typename Scoring::result_type Score;
    const Index rows = lu.rows();
    const Index cols = lu.cols();
    const Index size = (std::min)(rows,cols);
    nb_transpositions = 0;
    Index first_zero_pivot = -1;
    for(Index k = 0; k < size; ++k)
    {
      Index rrows = rows-k-1;
      Index rcols = cols-k-1;

      Index row_of_biggest_in_col;
      Score biggest_in_corner
        = lu.col(k).tail(rows-k).unaryExpr(Scoring()).maxCoeff(&row_of_biggest_in_col);
      row_of_biggest_in_col += k;

      row_transpositions[k] = PivIndex(row_of_biggest_in_col);

      if(biggest_in_corner != Score(0))
      {
        if(k != row_of_biggest_in_col)
        {
          lu.row(k).swap(lu.row(row_of_biggest_in_col));
          ++nb_transpositions;
        }

        // FIXME shall we introduce a safe quotient expression in cas 1/lu.coeff(k,k)
        // overflow but not the actual quotient?
        lu.col(k).tail(rrows) /= lu.coeff(k,k);
      }
      else if(first_zero_pivot==-1)
      {
        // the pivot is exactly zero, we record the index of the first pivot which is exactly 0,
        // and continue the factorization such we still have A = PLU
        first_zero_pivot = k;
      }

      if(k<rows-1)
        lu.bottomRightCorner(rrows,rcols).noalias() -= lu.col(k).tail(rrows) * lu.row(k).tail(rcols);
    }
    return first_zero_pivot;
  }

  /** \internal performs the LU decomposition in-place of the matrix represented
    * by the variables \a rows, \a cols, \a lu_data, and \a lu_stride using a
    * recursive, blocked algorithm.
    *
    * In addition, this function returns the row transpositions in the
    * vector \a row_transpositions which must have a size equal to the number
    * of columns of the matrix \a lu, and an integer \a nb_transpositions
    * which returns the actual number of transpositions.
    *
    * \returns The index of the first pivot which is exactly zero if any, or a negative number otherwise.
    *
    * \note This very low level interface using pointers, etc. is to:
    *   1 - reduce the number of instanciations to the strict minimum
    *   2 - avoid infinite recursion of the instanciations with Block<Block<Block<...> > >
    */
  static Index blocked_lu(Index rows, Index cols, Scalar* lu_data, Index luStride, PivIndex* row_transpositions, PivIndex& nb_transpositions, Index maxBlockSize=256)
  {
    MapLU lu1(lu_data,StorageOrder==RowMajor?rows:luStride,StorageOrder==RowMajor?luStride:cols);
    MatrixType lu(lu1,0,0,rows,cols);

    const Index size = (std::min)(rows,cols);

    // if the matrix is too small, no blocking:
    if(size<=16)
    {
      return unblocked_lu(lu, row_transpositions, nb_transpositions);
    }

    // automatically adjust the number of subdivisions to the size
    // of the matrix so that there is enough sub blocks:
    Index blockSize;
    {
      blockSize = size/8;
      blockSize = (blockSize/16)*16;
      blockSize = (std::min)((std::max)(blockSize,Index(8)), maxBlockSize);
    }

    nb_transpositions = 0;
    Index first_zero_pivot = -1;
    for(Index k = 0; k < size; k+=blockSize)
    {
      Index bs = (std::min)(size-k,blockSize); // actual size of the block
      Index trows = rows - k - bs; // trailing rows
      Index tsize = size - k - bs; // trailing size

      // partition the matrix:
      //                          A00 | A01 | A02
      // lu  = A_0 | A_1 | A_2 =  A10 | A11 | A12
      //                          A20 | A21 | A22
      BlockType A_0(lu,0,0,rows,k);
      BlockType A_2(lu,0,k+bs,rows,tsize);
      BlockType A11(lu,k,k,bs,bs);
      BlockType A12(lu,k,k+bs,bs,tsize);
      BlockType A21(lu,k+bs,k,trows,bs);
      BlockType A22(lu,k+bs,k+bs,trows,tsize);

      PivIndex nb_transpositions_in_panel;
      // recursively call the blocked LU algorithm on [A11^T A21^T]^T
      // with a very small blocking size:
      Index ret = blocked_lu(trows+bs, bs, &lu.coeffRef(k,k), luStride,
                   row_transpositions+k, nb_transpositions_in_panel, 16);
      if(ret>=0 && first_zero_pivot==-1)
        first_zero_pivot = k+ret;

      nb_transpositions += nb_transpositions_in_panel;
      // update permutations and apply them to A_0
      for(Index i=k; i<k+bs; ++i)
      {
        Index piv = (row_transpositions[i] += internal::convert_index<PivIndex>(k));
        A_0.row(i).swap(A_0.row(piv));
      }

      if(trows)
      {
        // apply permutations to A_2
        for(Index i=k;i<k+bs; ++i)
          A_2.row(i).swap(A_2.row(row_transpositions[i]));

        // A12 = A11^-1 A12
        A11.template triangularView<UnitLower>().solveInPlace(A12);

        A22.noalias() -= A21 * A12;
      }
    }
    return first_zero_pivot;
  }
};

/** \internal performs the LU decomposition with partial pivoting in-place.
  */
template<typename MatrixType, typename TranspositionType>
void partial_lu_inplace(MatrixType& lu, TranspositionType& row_transpositions, typename TranspositionType::StorageIndex& nb_transpositions)
{
  eigen_assert(lu.cols() == row_transpositions.size());
  eigen_assert((&row_transpositions.coeffRef(1)-&row_transpositions.coeffRef(0)) == 1);

  partial_lu_impl
    <typename MatrixType::Scalar, MatrixType::Flags&RowMajorBit?RowMajor:ColMajor, typename TranspositionType::StorageIndex>
    ::blocked_lu(lu.rows(), lu.cols(), &lu.coeffRef(0,0), lu.outerStride(), &row_transpositions.coeffRef(0), nb_transpositions);
}

} // end namespace internal

template<typename MatrixType>
void PartialPivLU<MatrixType>::compute()
{
  check_template_parameters();

  // the row permutation is stored as int indices, so just to be sure:
  eigen_assert(m_lu.rows()<NumTraits<int>::highest());

  m_l1_norm = m_lu.cwiseAbs().colwise().sum().maxCoeff();

  eigen_assert(m_lu.rows() == m_lu.cols() && "PartialPivLU is only for square (and moreover invertible) matrices");
  const Index size = m_lu.rows();

  m_rowsTranspositions.resize(size);

  typename TranspositionType::StorageIndex nb_transpositions;
  internal::partial_lu_inplace(m_lu, m_rowsTranspositions, nb_transpositions);
  m_det_p = (nb_transpositions%2) ? -1 : 1;

  m_p = m_rowsTranspositions;

  m_isInitialized = true;
}

template<typename MatrixType>
typename PartialPivLU<MatrixType>::Scalar PartialPivLU<MatrixType>::determinant() const
{
  eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
  return Scalar(m_det_p) * m_lu.diagonal().prod();
}

/** \returns the matrix represented by the decomposition,
 * i.e., it returns the product: P^{-1} L U.
 * This function is provided for debug purpose. */
template<typename MatrixType>
MatrixType PartialPivLU<MatrixType>::reconstructedMatrix() const
{
  eigen_assert(m_isInitialized && "LU is not initialized.");
  // LU
  MatrixType res = m_lu.template triangularView<UnitLower>().toDenseMatrix()
                 * m_lu.template triangularView<Upper>();

  // P^{-1}(LU)
  res = m_p.inverse() * res;

  return res;
}

/***** Implementation details *****************************************************/

namespace internal {

/***** Implementation of inverse() *****************************************************/
template<typename DstXprType, typename MatrixType>
struct Assignment<DstXprType, Inverse<PartialPivLU<MatrixType> >, internal::assign_op<typename DstXprType::Scalar,typename PartialPivLU<MatrixType>::Scalar>, Dense2Dense>
{
  typedef PartialPivLU<MatrixType> LuType;
  typedef Inverse<LuType> SrcXprType;
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<typename DstXprType::Scalar,typename LuType::Scalar> &)
  {
    dst = src.nestedExpression().solve(MatrixType::Identity(src.rows(), src.cols()));
  }
};
} // end namespace internal

/******** MatrixBase methods *******/

/** \lu_module
  *
  * \return the partial-pivoting LU decomposition of \c *this.
  *
  * \sa class PartialPivLU
  */
template<typename Derived>
inline const PartialPivLU<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::partialPivLu() const
{
  return PartialPivLU<PlainObject>(eval());
}

/** \lu_module
  *
  * Synonym of partialPivLu().
  *
  * \return the partial-pivoting LU decomposition of \c *this.
  *
  * \sa class PartialPivLU
  */
template<typename Derived>
inline const PartialPivLU<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::lu() const
{
  return PartialPivLU<PlainObject>(eval());
}

} // end namespace Eigen

#endif // EIGEN_PARTIALLU_H

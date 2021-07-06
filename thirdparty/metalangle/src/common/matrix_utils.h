//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Matrix:
//   Utility class implementing various matrix operations.
//   Supports matrices with minimum 2 and maximum 4 number of rows/columns.
//
// TODO: Check if we can merge Matrix.h in sample_util with this and replace it with this
// implementation.
// TODO: Rename this file to Matrix.h once we remove Matrix.h in sample_util.

#ifndef COMMON_MATRIX_UTILS_H_
#define COMMON_MATRIX_UTILS_H_

#include <vector>

#include "common/debug.h"
#include "common/mathutil.h"
#include "common/vector_utils.h"

namespace angle
{

template <typename T>
class Matrix
{
  public:
    Matrix(const std::vector<T> &elements, const unsigned int numRows, const unsigned int numCols)
        : mElements(elements), mRows(numRows), mCols(numCols)
    {
        ASSERT(rows() >= 1 && rows() <= 4);
        ASSERT(columns() >= 1 && columns() <= 4);
    }

    Matrix(const std::vector<T> &elements, const unsigned int size)
        : mElements(elements), mRows(size), mCols(size)
    {
        ASSERT(rows() >= 1 && rows() <= 4);
        ASSERT(columns() >= 1 && columns() <= 4);
    }

    Matrix(const T *elements, const unsigned int size) : mRows(size), mCols(size)
    {
        ASSERT(rows() >= 1 && rows() <= 4);
        ASSERT(columns() >= 1 && columns() <= 4);
        for (size_t i = 0; i < size * size; i++)
            mElements.push_back(elements[i]);
    }

    const T &operator()(const unsigned int rowIndex, const unsigned int columnIndex) const
    {
        ASSERT(rowIndex < mRows);
        ASSERT(columnIndex < mCols);
        return mElements[rowIndex * columns() + columnIndex];
    }

    T &operator()(const unsigned int rowIndex, const unsigned int columnIndex)
    {
        ASSERT(rowIndex < mRows);
        ASSERT(columnIndex < mCols);
        return mElements[rowIndex * columns() + columnIndex];
    }

    const T &at(const unsigned int rowIndex, const unsigned int columnIndex) const
    {
        ASSERT(rowIndex < mRows);
        ASSERT(columnIndex < mCols);
        return operator()(rowIndex, columnIndex);
    }

    Matrix<T> operator*(const Matrix<T> &m)
    {
        ASSERT(columns() == m.rows());

        unsigned int resultRows = rows();
        unsigned int resultCols = m.columns();
        Matrix<T> result(std::vector<T>(resultRows * resultCols), resultRows, resultCols);
        for (unsigned int i = 0; i < resultRows; i++)
        {
            for (unsigned int j = 0; j < resultCols; j++)
            {
                T tmp = 0.0f;
                for (unsigned int k = 0; k < columns(); k++)
                    tmp += at(i, k) * m(k, j);
                result(i, j) = tmp;
            }
        }

        return result;
    }

    void operator*=(const Matrix<T> &m)
    {
        ASSERT(columns() == m.rows());
        Matrix<T> res  = (*this) * m;
        size_t numElts = res.elements().size();
        mElements.resize(numElts);
        memcpy(mElements.data(), res.data(), numElts * sizeof(float));
    }

    bool operator==(const Matrix<T> &m) const
    {
        ASSERT(columns() == m.columns());
        ASSERT(rows() == m.rows());
        return mElements == m.elements();
    }

    bool operator!=(const Matrix<T> &m) const { return !(mElements == m.elements()); }

    bool nearlyEqual(T epsilon, const Matrix<T> &m) const
    {
        ASSERT(columns() == m.columns());
        ASSERT(rows() == m.rows());
        const auto &otherElts = m.elements();
        for (size_t i = 0; i < otherElts.size(); i++)
        {
            if ((mElements[i] - otherElts[i] > epsilon) && (otherElts[i] - mElements[i] > epsilon))
                return false;
        }
        return true;
    }

    unsigned int size() const
    {
        ASSERT(rows() == columns());
        return rows();
    }

    unsigned int rows() const { return mRows; }

    unsigned int columns() const { return mCols; }

    std::vector<T> elements() const { return mElements; }
    T *data() { return mElements.data(); }

    Matrix<T> compMult(const Matrix<T> &mat1) const
    {
        Matrix result(std::vector<T>(mElements.size()), rows(), columns());
        for (unsigned int i = 0; i < rows(); i++)
        {
            for (unsigned int j = 0; j < columns(); j++)
            {
                T lhs        = at(i, j);
                T rhs        = mat1(i, j);
                result(i, j) = rhs * lhs;
            }
        }

        return result;
    }

    Matrix<T> outerProduct(const Matrix<T> &mat1) const
    {
        unsigned int cols = mat1.columns();
        Matrix result(std::vector<T>(rows() * cols), rows(), cols);
        for (unsigned int i = 0; i < rows(); i++)
            for (unsigned int j = 0; j < cols; j++)
                result(i, j) = at(i, 0) * mat1(0, j);

        return result;
    }

    Matrix<T> transpose() const
    {
        Matrix result(std::vector<T>(mElements.size()), columns(), rows());
        for (unsigned int i = 0; i < columns(); i++)
            for (unsigned int j = 0; j < rows(); j++)
                result(i, j) = at(j, i);

        return result;
    }

    T determinant() const
    {
        ASSERT(rows() == columns());

        switch (size())
        {
            case 2:
                return at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0);

            case 3:
                return at(0, 0) * at(1, 1) * at(2, 2) + at(0, 1) * at(1, 2) * at(2, 0) +
                       at(0, 2) * at(1, 0) * at(2, 1) - at(0, 2) * at(1, 1) * at(2, 0) -
                       at(0, 1) * at(1, 0) * at(2, 2) - at(0, 0) * at(1, 2) * at(2, 1);

            case 4:
            {
                const float minorMatrices[4][3 * 3] = {{
                                                           at(1, 1),
                                                           at(2, 1),
                                                           at(3, 1),
                                                           at(1, 2),
                                                           at(2, 2),
                                                           at(3, 2),
                                                           at(1, 3),
                                                           at(2, 3),
                                                           at(3, 3),
                                                       },
                                                       {
                                                           at(1, 0),
                                                           at(2, 0),
                                                           at(3, 0),
                                                           at(1, 2),
                                                           at(2, 2),
                                                           at(3, 2),
                                                           at(1, 3),
                                                           at(2, 3),
                                                           at(3, 3),
                                                       },
                                                       {
                                                           at(1, 0),
                                                           at(2, 0),
                                                           at(3, 0),
                                                           at(1, 1),
                                                           at(2, 1),
                                                           at(3, 1),
                                                           at(1, 3),
                                                           at(2, 3),
                                                           at(3, 3),
                                                       },
                                                       {
                                                           at(1, 0),
                                                           at(2, 0),
                                                           at(3, 0),
                                                           at(1, 1),
                                                           at(2, 1),
                                                           at(3, 1),
                                                           at(1, 2),
                                                           at(2, 2),
                                                           at(3, 2),
                                                       }};
                return at(0, 0) * Matrix<T>(minorMatrices[0], 3).determinant() -
                       at(0, 1) * Matrix<T>(minorMatrices[1], 3).determinant() +
                       at(0, 2) * Matrix<T>(minorMatrices[2], 3).determinant() -
                       at(0, 3) * Matrix<T>(minorMatrices[3], 3).determinant();
            }

            default:
                UNREACHABLE();
                break;
        }

        return T();
    }

    Matrix<T> inverse() const
    {
        ASSERT(rows() == columns());

        Matrix<T> cof(std::vector<T>(mElements.size()), rows(), columns());
        switch (size())
        {
            case 2:
                cof(0, 0) = at(1, 1);
                cof(0, 1) = -at(1, 0);
                cof(1, 0) = -at(0, 1);
                cof(1, 1) = at(0, 0);
                break;

            case 3:
                cof(0, 0) = at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2);
                cof(0, 1) = -(at(1, 0) * at(2, 2) - at(2, 0) * at(1, 2));
                cof(0, 2) = at(1, 0) * at(2, 1) - at(2, 0) * at(1, 1);
                cof(1, 0) = -(at(0, 1) * at(2, 2) - at(2, 1) * at(0, 2));
                cof(1, 1) = at(0, 0) * at(2, 2) - at(2, 0) * at(0, 2);
                cof(1, 2) = -(at(0, 0) * at(2, 1) - at(2, 0) * at(0, 1));
                cof(2, 0) = at(0, 1) * at(1, 2) - at(1, 1) * at(0, 2);
                cof(2, 1) = -(at(0, 0) * at(1, 2) - at(1, 0) * at(0, 2));
                cof(2, 2) = at(0, 0) * at(1, 1) - at(1, 0) * at(0, 1);
                break;

            case 4:
                cof(0, 0) = at(1, 1) * at(2, 2) * at(3, 3) + at(2, 1) * at(3, 2) * at(1, 3) +
                            at(3, 1) * at(1, 2) * at(2, 3) - at(1, 1) * at(3, 2) * at(2, 3) -
                            at(2, 1) * at(1, 2) * at(3, 3) - at(3, 1) * at(2, 2) * at(1, 3);
                cof(0, 1) = -(at(1, 0) * at(2, 2) * at(3, 3) + at(2, 0) * at(3, 2) * at(1, 3) +
                              at(3, 0) * at(1, 2) * at(2, 3) - at(1, 0) * at(3, 2) * at(2, 3) -
                              at(2, 0) * at(1, 2) * at(3, 3) - at(3, 0) * at(2, 2) * at(1, 3));
                cof(0, 2) = at(1, 0) * at(2, 1) * at(3, 3) + at(2, 0) * at(3, 1) * at(1, 3) +
                            at(3, 0) * at(1, 1) * at(2, 3) - at(1, 0) * at(3, 1) * at(2, 3) -
                            at(2, 0) * at(1, 1) * at(3, 3) - at(3, 0) * at(2, 1) * at(1, 3);
                cof(0, 3) = -(at(1, 0) * at(2, 1) * at(3, 2) + at(2, 0) * at(3, 1) * at(1, 2) +
                              at(3, 0) * at(1, 1) * at(2, 2) - at(1, 0) * at(3, 1) * at(2, 2) -
                              at(2, 0) * at(1, 1) * at(3, 2) - at(3, 0) * at(2, 1) * at(1, 2));
                cof(1, 0) = -(at(0, 1) * at(2, 2) * at(3, 3) + at(2, 1) * at(3, 2) * at(0, 3) +
                              at(3, 1) * at(0, 2) * at(2, 3) - at(0, 1) * at(3, 2) * at(2, 3) -
                              at(2, 1) * at(0, 2) * at(3, 3) - at(3, 1) * at(2, 2) * at(0, 3));
                cof(1, 1) = at(0, 0) * at(2, 2) * at(3, 3) + at(2, 0) * at(3, 2) * at(0, 3) +
                            at(3, 0) * at(0, 2) * at(2, 3) - at(0, 0) * at(3, 2) * at(2, 3) -
                            at(2, 0) * at(0, 2) * at(3, 3) - at(3, 0) * at(2, 2) * at(0, 3);
                cof(1, 2) = -(at(0, 0) * at(2, 1) * at(3, 3) + at(2, 0) * at(3, 1) * at(0, 3) +
                              at(3, 0) * at(0, 1) * at(2, 3) - at(0, 0) * at(3, 1) * at(2, 3) -
                              at(2, 0) * at(0, 1) * at(3, 3) - at(3, 0) * at(2, 1) * at(0, 3));
                cof(1, 3) = at(0, 0) * at(2, 1) * at(3, 2) + at(2, 0) * at(3, 1) * at(0, 2) +
                            at(3, 0) * at(0, 1) * at(2, 2) - at(0, 0) * at(3, 1) * at(2, 2) -
                            at(2, 0) * at(0, 1) * at(3, 2) - at(3, 0) * at(2, 1) * at(0, 2);
                cof(2, 0) = at(0, 1) * at(1, 2) * at(3, 3) + at(1, 1) * at(3, 2) * at(0, 3) +
                            at(3, 1) * at(0, 2) * at(1, 3) - at(0, 1) * at(3, 2) * at(1, 3) -
                            at(1, 1) * at(0, 2) * at(3, 3) - at(3, 1) * at(1, 2) * at(0, 3);
                cof(2, 1) = -(at(0, 0) * at(1, 2) * at(3, 3) + at(1, 0) * at(3, 2) * at(0, 3) +
                              at(3, 0) * at(0, 2) * at(1, 3) - at(0, 0) * at(3, 2) * at(1, 3) -
                              at(1, 0) * at(0, 2) * at(3, 3) - at(3, 0) * at(1, 2) * at(0, 3));
                cof(2, 2) = at(0, 0) * at(1, 1) * at(3, 3) + at(1, 0) * at(3, 1) * at(0, 3) +
                            at(3, 0) * at(0, 1) * at(1, 3) - at(0, 0) * at(3, 1) * at(1, 3) -
                            at(1, 0) * at(0, 1) * at(3, 3) - at(3, 0) * at(1, 1) * at(0, 3);
                cof(2, 3) = -(at(0, 0) * at(1, 1) * at(3, 2) + at(1, 0) * at(3, 1) * at(0, 2) +
                              at(3, 0) * at(0, 1) * at(1, 2) - at(0, 0) * at(3, 1) * at(1, 2) -
                              at(1, 0) * at(0, 1) * at(3, 2) - at(3, 0) * at(1, 1) * at(0, 2));
                cof(3, 0) = -(at(0, 1) * at(1, 2) * at(2, 3) + at(1, 1) * at(2, 2) * at(0, 3) +
                              at(2, 1) * at(0, 2) * at(1, 3) - at(0, 1) * at(2, 2) * at(1, 3) -
                              at(1, 1) * at(0, 2) * at(2, 3) - at(2, 1) * at(1, 2) * at(0, 3));
                cof(3, 1) = at(0, 0) * at(1, 2) * at(2, 3) + at(1, 0) * at(2, 2) * at(0, 3) +
                            at(2, 0) * at(0, 2) * at(1, 3) - at(0, 0) * at(2, 2) * at(1, 3) -
                            at(1, 0) * at(0, 2) * at(2, 3) - at(2, 0) * at(1, 2) * at(0, 3);
                cof(3, 2) = -(at(0, 0) * at(1, 1) * at(2, 3) + at(1, 0) * at(2, 1) * at(0, 3) +
                              at(2, 0) * at(0, 1) * at(1, 3) - at(0, 0) * at(2, 1) * at(1, 3) -
                              at(1, 0) * at(0, 1) * at(2, 3) - at(2, 0) * at(1, 1) * at(0, 3));
                cof(3, 3) = at(0, 0) * at(1, 1) * at(2, 2) + at(1, 0) * at(2, 1) * at(0, 2) +
                            at(2, 0) * at(0, 1) * at(1, 2) - at(0, 0) * at(2, 1) * at(1, 2) -
                            at(1, 0) * at(0, 1) * at(2, 2) - at(2, 0) * at(1, 1) * at(0, 2);
                break;

            default:
                UNREACHABLE();
                break;
        }

        // The inverse of A is the transpose of the cofactor matrix times the reciprocal of the
        // determinant of A.
        Matrix<T> adjugateMatrix(cof.transpose());
        T det = determinant();
        Matrix<T> result(std::vector<T>(mElements.size()), rows(), columns());
        for (unsigned int i = 0; i < rows(); i++)
            for (unsigned int j = 0; j < columns(); j++)
                result(i, j) = (det != static_cast<T>(0)) ? adjugateMatrix(i, j) / det : T();

        return result;
    }

    void setToIdentity()
    {
        ASSERT(rows() == columns());

        const auto one  = T(1);
        const auto zero = T(0);

        for (auto &e : mElements)
            e = zero;

        for (unsigned int i = 0; i < rows(); ++i)
        {
            const auto pos = i * columns() + (i % columns());
            mElements[pos] = one;
        }
    }

    template <unsigned int Size>
    static void setToIdentity(T (&matrix)[Size])
    {
        static_assert(gl::iSquareRoot<Size>() != 0, "Matrix is not square.");

        const auto cols = gl::iSquareRoot<Size>();
        const auto one  = T(1);
        const auto zero = T(0);

        for (auto &e : matrix)
            e = zero;

        for (unsigned int i = 0; i < cols; ++i)
        {
            const auto pos = i * cols + (i % cols);
            matrix[pos]    = one;
        }
    }

  protected:
    std::vector<T> mElements;
    unsigned int mRows;
    unsigned int mCols;
};

class Mat4 : public Matrix<float>
{
  public:
    Mat4();
    Mat4(const Matrix<float> generalMatrix);
    Mat4(const std::vector<float> &elements);
    Mat4(const float *elements);
    Mat4(float m00,
         float m01,
         float m02,
         float m03,
         float m10,
         float m11,
         float m12,
         float m13,
         float m20,
         float m21,
         float m22,
         float m23,
         float m30,
         float m31,
         float m32,
         float m33);

    static Mat4 Rotate(float angle, const Vector3 &axis);
    static Mat4 Translate(const Vector3 &t);
    static Mat4 Scale(const Vector3 &s);
    static Mat4 Frustum(float l, float r, float b, float t, float n, float f);
    static Mat4 Perspective(float fov, float aspectRatio, float n, float f);
    static Mat4 Ortho(float l, float r, float b, float t, float n, float f);

    Mat4 product(const Mat4 &m);
    Vector4 product(const Vector4 &b);
    void dump();
};

}  // namespace angle

#endif  // COMMON_MATRIX_UTILS_H_

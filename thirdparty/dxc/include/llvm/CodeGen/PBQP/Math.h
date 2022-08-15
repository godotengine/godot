//===------ Math.h - PBQP Vector and Matrix classes -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_MATH_H
#define LLVM_CODEGEN_PBQP_MATH_H

#include "llvm/ADT/Hashing.h"
#include <algorithm>
#include <cassert>
#include <functional>

namespace llvm {
namespace PBQP {

typedef float PBQPNum;

/// \brief PBQP Vector class.
class Vector {
  friend hash_code hash_value(const Vector &);
public:

  /// \brief Construct a PBQP vector of the given size.
  explicit Vector(unsigned Length)
    : Length(Length), Data(new PBQPNum[Length]) {
    // llvm::dbgs() << "Constructing PBQP::Vector "
    //              << this << " (length " << Length << ")\n";
  }

  /// \brief Construct a PBQP vector with initializer.
  Vector(unsigned Length, PBQPNum InitVal)
    : Length(Length), Data(new PBQPNum[Length]) {
    // llvm::dbgs() << "Constructing PBQP::Vector "
    //              << this << " (length " << Length << ", fill "
    //              << InitVal << ")\n";
    std::fill(Data, Data + Length, InitVal);
  }

  /// \brief Copy construct a PBQP vector.
  Vector(const Vector &V)
    : Length(V.Length), Data(new PBQPNum[Length]) {
    // llvm::dbgs() << "Copy-constructing PBQP::Vector " << this
    //              << " from PBQP::Vector " << &V << "\n";
    std::copy(V.Data, V.Data + Length, Data);
  }

  /// \brief Move construct a PBQP vector.
  Vector(Vector &&V)
    : Length(V.Length), Data(V.Data) {
    V.Length = 0;
    V.Data = nullptr;
  }

  /// \brief Destroy this vector, return its memory.
  ~Vector() {
    // llvm::dbgs() << "Deleting PBQP::Vector " << this << "\n";
    delete[] Data;
  }

  /// \brief Copy-assignment operator.
  Vector& operator=(const Vector &V) {
    // llvm::dbgs() << "Assigning to PBQP::Vector " << this
    //              << " from PBQP::Vector " << &V << "\n";
    delete[] Data;
    Length = V.Length;
    Data = new PBQPNum[Length];
    std::copy(V.Data, V.Data + Length, Data);
    return *this;
  }

  /// \brief Move-assignment operator.
  Vector& operator=(Vector &&V) {
    delete[] Data;
    Length = V.Length;
    Data = V.Data;
    V.Length = 0;
    V.Data = nullptr;
    return *this;
  }

  /// \brief Comparison operator.
  bool operator==(const Vector &V) const {
    assert(Length != 0 && Data != nullptr && "Invalid vector");
    if (Length != V.Length)
      return false;
    return std::equal(Data, Data + Length, V.Data);
  }

  /// \brief Return the length of the vector
  unsigned getLength() const {
    assert(Length != 0 && Data != nullptr && "Invalid vector");
    return Length;
  }

  /// \brief Element access.
  PBQPNum& operator[](unsigned Index) {
    assert(Length != 0 && Data != nullptr && "Invalid vector");
    assert(Index < Length && "Vector element access out of bounds.");
    return Data[Index];
  }

  /// \brief Const element access.
  const PBQPNum& operator[](unsigned Index) const {
    assert(Length != 0 && Data != nullptr && "Invalid vector");
    assert(Index < Length && "Vector element access out of bounds.");
    return Data[Index];
  }

  /// \brief Add another vector to this one.
  Vector& operator+=(const Vector &V) {
    assert(Length != 0 && Data != nullptr && "Invalid vector");
    assert(Length == V.Length && "Vector length mismatch.");
    std::transform(Data, Data + Length, V.Data, Data, std::plus<PBQPNum>());
    return *this;
  }

  /// \brief Subtract another vector from this one.
  Vector& operator-=(const Vector &V) {
    assert(Length != 0 && Data != nullptr && "Invalid vector");
    assert(Length == V.Length && "Vector length mismatch.");
    std::transform(Data, Data + Length, V.Data, Data, std::minus<PBQPNum>());
    return *this;
  }

  /// \brief Returns the index of the minimum value in this vector
  unsigned minIndex() const {
    assert(Length != 0 && Data != nullptr && "Invalid vector");
    return std::min_element(Data, Data + Length) - Data;
  }

private:
  unsigned Length;
  PBQPNum *Data;
};

/// \brief Return a hash_value for the given vector.
inline hash_code hash_value(const Vector &V) {
  unsigned *VBegin = reinterpret_cast<unsigned*>(V.Data);
  unsigned *VEnd = reinterpret_cast<unsigned*>(V.Data + V.Length);
  return hash_combine(V.Length, hash_combine_range(VBegin, VEnd));
}

/// \brief Output a textual representation of the given vector on the given
///        output stream.
template <typename OStream>
OStream& operator<<(OStream &OS, const Vector &V) {
  assert((V.getLength() != 0) && "Zero-length vector badness.");

  OS << "[ " << V[0];
  for (unsigned i = 1; i < V.getLength(); ++i)
    OS << ", " << V[i];
  OS << " ]";

  return OS;
}

/// \brief PBQP Matrix class
class Matrix {
private:
  friend hash_code hash_value(const Matrix &);
public:

  /// \brief Construct a PBQP Matrix with the given dimensions.
  Matrix(unsigned Rows, unsigned Cols) :
    Rows(Rows), Cols(Cols), Data(new PBQPNum[Rows * Cols]) {
  }

  /// \brief Construct a PBQP Matrix with the given dimensions and initial
  /// value.
  Matrix(unsigned Rows, unsigned Cols, PBQPNum InitVal)
    : Rows(Rows), Cols(Cols), Data(new PBQPNum[Rows * Cols]) {
    std::fill(Data, Data + (Rows * Cols), InitVal);
  }

  /// \brief Copy construct a PBQP matrix.
  Matrix(const Matrix &M)
    : Rows(M.Rows), Cols(M.Cols), Data(new PBQPNum[Rows * Cols]) {
    std::copy(M.Data, M.Data + (Rows * Cols), Data);
  }

  /// \brief Move construct a PBQP matrix.
  Matrix(Matrix &&M)
    : Rows(M.Rows), Cols(M.Cols), Data(M.Data) {
    M.Rows = M.Cols = 0;
    M.Data = nullptr;
  }

  /// \brief Destroy this matrix, return its memory.
  ~Matrix() { delete[] Data; }

  /// \brief Copy-assignment operator.
  Matrix& operator=(const Matrix &M) {
    delete[] Data;
    Rows = M.Rows; Cols = M.Cols;
    Data = new PBQPNum[Rows * Cols];
    std::copy(M.Data, M.Data + (Rows * Cols), Data);
    return *this;
  }

  /// \brief Move-assignment operator.
  Matrix& operator=(Matrix &&M) {
    delete[] Data;
    Rows = M.Rows;
    Cols = M.Cols;
    Data = M.Data;
    M.Rows = M.Cols = 0;
    M.Data = nullptr;
    return *this;
  }

  /// \brief Comparison operator.
  bool operator==(const Matrix &M) const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    if (Rows != M.Rows || Cols != M.Cols)
      return false;
    return std::equal(Data, Data + (Rows * Cols), M.Data);
  }

  /// \brief Return the number of rows in this matrix.
  unsigned getRows() const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    return Rows;
  }

  /// \brief Return the number of cols in this matrix.
  unsigned getCols() const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    return Cols;
  }

  /// \brief Matrix element access.
  PBQPNum* operator[](unsigned R) {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    assert(R < Rows && "Row out of bounds.");
    return Data + (R * Cols);
  }

  /// \brief Matrix element access.
  const PBQPNum* operator[](unsigned R) const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    assert(R < Rows && "Row out of bounds.");
    return Data + (R * Cols);
  }

  /// \brief Returns the given row as a vector.
  Vector getRowAsVector(unsigned R) const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    Vector V(Cols);
    for (unsigned C = 0; C < Cols; ++C)
      V[C] = (*this)[R][C];
    return V;
  }

  /// \brief Returns the given column as a vector.
  Vector getColAsVector(unsigned C) const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    Vector V(Rows);
    for (unsigned R = 0; R < Rows; ++R)
      V[R] = (*this)[R][C];
    return V;
  }

  /// \brief Reset the matrix to the given value.
  Matrix& reset(PBQPNum Val = 0) {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    std::fill(Data, Data + (Rows * Cols), Val);
    return *this;
  }

  /// \brief Set a single row of this matrix to the given value.
  Matrix& setRow(unsigned R, PBQPNum Val) {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    assert(R < Rows && "Row out of bounds.");
    std::fill(Data + (R * Cols), Data + ((R + 1) * Cols), Val);
    return *this;
  }

  /// \brief Set a single column of this matrix to the given value.
  Matrix& setCol(unsigned C, PBQPNum Val) {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    assert(C < Cols && "Column out of bounds.");
    for (unsigned R = 0; R < Rows; ++R)
      (*this)[R][C] = Val;
    return *this;
  }

  /// \brief Matrix transpose.
  Matrix transpose() const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    Matrix M(Cols, Rows);
    for (unsigned r = 0; r < Rows; ++r)
      for (unsigned c = 0; c < Cols; ++c)
        M[c][r] = (*this)[r][c];
    return M;
  }

  /// \brief Returns the diagonal of the matrix as a vector.
  ///
  /// Matrix must be square.
  Vector diagonalize() const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    assert(Rows == Cols && "Attempt to diagonalize non-square matrix.");
    Vector V(Rows);
    for (unsigned r = 0; r < Rows; ++r)
      V[r] = (*this)[r][r];
    return V;
  }

  /// \brief Add the given matrix to this one.
  Matrix& operator+=(const Matrix &M) {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    assert(Rows == M.Rows && Cols == M.Cols &&
           "Matrix dimensions mismatch.");
    std::transform(Data, Data + (Rows * Cols), M.Data, Data,
                   std::plus<PBQPNum>());
    return *this;
  }

  Matrix operator+(const Matrix &M) {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    Matrix Tmp(*this);
    Tmp += M;
    return Tmp;
  }

  /// \brief Returns the minimum of the given row
  PBQPNum getRowMin(unsigned R) const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    assert(R < Rows && "Row out of bounds");
    return *std::min_element(Data + (R * Cols), Data + ((R + 1) * Cols));
  }

  /// \brief Returns the minimum of the given column
  PBQPNum getColMin(unsigned C) const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    PBQPNum MinElem = (*this)[0][C];
    for (unsigned R = 1; R < Rows; ++R)
      if ((*this)[R][C] < MinElem)
        MinElem = (*this)[R][C];
    return MinElem;
  }

  /// \brief Subtracts the given scalar from the elements of the given row.
  Matrix& subFromRow(unsigned R, PBQPNum Val) {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    assert(R < Rows && "Row out of bounds");
    std::transform(Data + (R * Cols), Data + ((R + 1) * Cols),
                   Data + (R * Cols),
                   std::bind2nd(std::minus<PBQPNum>(), Val));
    return *this;
  }

  /// \brief Subtracts the given scalar from the elements of the given column.
  Matrix& subFromCol(unsigned C, PBQPNum Val) {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    for (unsigned R = 0; R < Rows; ++R)
      (*this)[R][C] -= Val;
    return *this;
  }

  /// \brief Returns true if this is a zero matrix.
  bool isZero() const {
    assert(Rows != 0 && Cols != 0 && Data != nullptr && "Invalid matrix");
    return find_if(Data, Data + (Rows * Cols),
                   std::bind2nd(std::not_equal_to<PBQPNum>(), 0)) ==
      Data + (Rows * Cols);
  }

private:
  unsigned Rows, Cols;
  PBQPNum *Data;
};

/// \brief Return a hash_code for the given matrix.
inline hash_code hash_value(const Matrix &M) {
  unsigned *MBegin = reinterpret_cast<unsigned*>(M.Data);
  unsigned *MEnd = reinterpret_cast<unsigned*>(M.Data + (M.Rows * M.Cols));
  return hash_combine(M.Rows, M.Cols, hash_combine_range(MBegin, MEnd));
}

/// \brief Output a textual representation of the given matrix on the given
///        output stream.
template <typename OStream>
OStream& operator<<(OStream &OS, const Matrix &M) {
  assert((M.getRows() != 0) && "Zero-row matrix badness.");
  for (unsigned i = 0; i < M.getRows(); ++i)
    OS << M.getRowAsVector(i) << "\n";
  return OS;
}

template <typename Metadata>
class MDVector : public Vector {
public:
  MDVector(const Vector &v) : Vector(v), md(*this) { }
  MDVector(Vector &&v) : Vector(std::move(v)), md(*this) { }
  const Metadata& getMetadata() const { return md; }
private:
  Metadata md;
};

template <typename Metadata>
inline hash_code hash_value(const MDVector<Metadata> &V) {
  return hash_value(static_cast<const Vector&>(V));
}

template <typename Metadata>
class MDMatrix : public Matrix {
public:
  MDMatrix(const Matrix &m) : Matrix(m), md(*this) { }
  MDMatrix(Matrix &&m) : Matrix(std::move(m)), md(*this) { }
  const Metadata& getMetadata() const { return md; }
private:
  Metadata md;
};

template <typename Metadata>
inline hash_code hash_value(const MDMatrix<Metadata> &M) {
  return hash_value(static_cast<const Matrix&>(M));
}

} // namespace PBQP
} // namespace llvm

#endif // LLVM_CODEGEN_PBQP_MATH_H

//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// vector_utils.h: Utility classes implementing various vector operations

#ifndef COMMON_VECTOR_UTILS_H_
#define COMMON_VECTOR_UTILS_H_

#include <cmath>
#include <cstddef>
#include <ostream>
#include <type_traits>

namespace angle
{

template <size_t Dimension, typename Type>
class Vector;

using Vector2 = Vector<2, float>;
using Vector3 = Vector<3, float>;
using Vector4 = Vector<4, float>;

using Vector2I = Vector<2, int>;
using Vector3I = Vector<3, int>;
using Vector4I = Vector<4, int>;

using Vector2U = Vector<2, unsigned int>;
using Vector3U = Vector<3, unsigned int>;
using Vector4U = Vector<4, unsigned int>;

template <size_t Dimension, typename Type>
class VectorBase
{
  public:
    using VectorN = Vector<Dimension, Type>;

    // Constructors
    VectorBase() = default;
    explicit VectorBase(Type element);

    template <typename Type2>
    VectorBase(const VectorBase<Dimension, Type2> &other);

    template <typename Arg1, typename Arg2, typename... Args>
    VectorBase(const Arg1 &arg1, const Arg2 &arg2, const Args &... args);

    // Access the vector backing storage directly
    const Type *data() const { return mData; }
    Type *data() { return mData; }
    constexpr size_t size() const { return Dimension; }

    // Load or store the pointer from / to raw data
    static VectorN Load(const Type *source);
    static void Store(const VectorN &source, Type *destination);

    // Index the vector
    Type &operator[](size_t i) { return mData[i]; }
    const Type &operator[](size_t i) const { return mData[i]; }

    // Basic arithmetic operations
    VectorN operator+() const;
    VectorN operator-() const;
    VectorN operator+(const VectorN &other) const;
    VectorN operator-(const VectorN &other) const;
    VectorN operator*(const VectorN &other) const;
    VectorN operator/(const VectorN &other) const;
    VectorN operator*(Type other) const;
    VectorN operator/(Type other) const;
    friend VectorN operator*(Type a, const VectorN &b) { return b * a; }

    // Compound arithmetic operations
    VectorN &operator+=(const VectorN &other);
    VectorN &operator-=(const VectorN &other);
    VectorN &operator*=(const VectorN &other);
    VectorN &operator/=(const VectorN &other);
    VectorN &operator*=(Type other);
    VectorN &operator/=(Type other);

    // Comparison operators
    bool operator==(const VectorN &other) const;
    bool operator!=(const VectorN &other) const;

    // Other arithmetic operations
    Type length() const;
    Type lengthSquared() const;
    Type dot(const VectorBase<Dimension, Type> &other) const;
    VectorN normalized() const;

  protected:
    template <size_t CurrentIndex, size_t OtherDimension, typename OtherType, typename... Args>
    void initWithList(const Vector<OtherDimension, OtherType> &arg1, const Args &... args);

    // Some old compilers consider this function an alternative for initWithList(Vector)
    // when the variant above is more precise. Use SFINAE on the return value to hide
    // this variant for non-arithmetic types. The return value is still void.
    template <size_t CurrentIndex, typename OtherType, typename... Args>
    typename std::enable_if<std::is_arithmetic<OtherType>::value>::type initWithList(
        OtherType arg1,
        const Args &... args);

    template <size_t CurrentIndex>
    void initWithList() const;

    template <size_t Dimension2, typename Type2>
    friend class VectorBase;

    Type mData[Dimension];
};

template <size_t Dimension, typename Type>
std::ostream &operator<<(std::ostream &ostream, const VectorBase<Dimension, Type> &vector);

template <typename Type>
class Vector<2, Type> : public VectorBase<2, Type>
{
  public:
    // Import the constructors defined in VectorBase
    using VectorBase<2, Type>::VectorBase;

    // Element shorthands
    Type &x() { return this->mData[0]; }
    Type &y() { return this->mData[1]; }

    const Type &x() const { return this->mData[0]; }
    const Type &y() const { return this->mData[1]; }
};

template <typename Type>
std::ostream &operator<<(std::ostream &ostream, const Vector<2, Type> &vector);

template <typename Type>
class Vector<3, Type> : public VectorBase<3, Type>
{
  public:
    // Import the constructors defined in VectorBase
    using VectorBase<3, Type>::VectorBase;

    // Additional operations
    Vector<3, Type> cross(const Vector<3, Type> &other) const;

    // Element shorthands
    Type &x() { return this->mData[0]; }
    Type &y() { return this->mData[1]; }
    Type &z() { return this->mData[2]; }

    const Type &x() const { return this->mData[0]; }
    const Type &y() const { return this->mData[1]; }
    const Type &z() const { return this->mData[2]; }
};

template <typename Type>
std::ostream &operator<<(std::ostream &ostream, const Vector<3, Type> &vector);

template <typename Type>
class Vector<4, Type> : public VectorBase<4, Type>
{
  public:
    // Import the constructors defined in VectorBase
    using VectorBase<4, Type>::VectorBase;

    // Element shorthands
    Type &x() { return this->mData[0]; }
    Type &y() { return this->mData[1]; }
    Type &z() { return this->mData[2]; }
    Type &w() { return this->mData[3]; }

    const Type &x() const { return this->mData[0]; }
    const Type &y() const { return this->mData[1]; }
    const Type &z() const { return this->mData[2]; }
    const Type &w() const { return this->mData[3]; }
};

template <typename Type>
std::ostream &operator<<(std::ostream &ostream, const Vector<4, Type> &vector);

// Implementation of constructors and misc operations

template <size_t Dimension, typename Type>
VectorBase<Dimension, Type>::VectorBase(Type element)
{
    for (size_t i = 0; i < Dimension; ++i)
    {
        mData[i] = element;
    }
}

template <size_t Dimension, typename Type>
template <typename Type2>
VectorBase<Dimension, Type>::VectorBase(const VectorBase<Dimension, Type2> &other)
{
    for (size_t i = 0; i < Dimension; ++i)
    {
        mData[i] = static_cast<Type>(other.mData[i]);
    }
}

// Ideally we would like to have only two constructors:
//  - a scalar constructor that takes Type as a parameter
//  - a compound constructor
// However if we define the compound constructor for when it has a single arguments, then calling
// Vector2(0.0) will be ambiguous. To solve this we take advantage of there being a single compound
// constructor with a single argument, which is the copy constructor. We end up with three
// constructors:
//  - the scalar constructor
//  - the copy constructor
//  - the compound constructor for two or more arguments, hence the arg1, and arg2 here.
template <size_t Dimension, typename Type>
template <typename Arg1, typename Arg2, typename... Args>
VectorBase<Dimension, Type>::VectorBase(const Arg1 &arg1, const Arg2 &arg2, const Args &... args)
{
    initWithList<0>(arg1, arg2, args...);
}

template <size_t Dimension, typename Type>
template <size_t CurrentIndex, size_t OtherDimension, typename OtherType, typename... Args>
void VectorBase<Dimension, Type>::initWithList(const Vector<OtherDimension, OtherType> &arg1,
                                               const Args &... args)
{
    static_assert(CurrentIndex + OtherDimension <= Dimension,
                  "Too much data in the vector constructor.");
    for (size_t i = 0; i < OtherDimension; ++i)
    {
        mData[CurrentIndex + i] = static_cast<Type>(arg1.mData[i]);
    }
    initWithList<CurrentIndex + OtherDimension>(args...);
}

template <size_t Dimension, typename Type>
template <size_t CurrentIndex, typename OtherType, typename... Args>
typename std::enable_if<std::is_arithmetic<OtherType>::value>::type
VectorBase<Dimension, Type>::initWithList(OtherType arg1, const Args &... args)
{
    static_assert(CurrentIndex + 1 <= Dimension, "Too much data in the vector constructor.");
    mData[CurrentIndex] = static_cast<Type>(arg1);
    initWithList<CurrentIndex + 1>(args...);
}

template <size_t Dimension, typename Type>
template <size_t CurrentIndex>
void VectorBase<Dimension, Type>::initWithList() const
{
    static_assert(CurrentIndex == Dimension, "Not enough data in the vector constructor.");
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> VectorBase<Dimension, Type>::Load(const Type *source)
{
    Vector<Dimension, Type> result;
    for (size_t i = 0; i < Dimension; ++i)
    {
        result.mData[i] = source[i];
    }
    return result;
}

template <size_t Dimension, typename Type>
void VectorBase<Dimension, Type>::Store(const Vector<Dimension, Type> &source, Type *destination)
{
    for (size_t i = 0; i < Dimension; ++i)
    {
        destination[i] = source.mData[i];
    }
}

// Implementation of basic arithmetic operations
template <size_t Dimension, typename Type>
Vector<Dimension, Type> VectorBase<Dimension, Type>::operator+() const
{
    Vector<Dimension, Type> result;
    for (size_t i = 0; i < Dimension; ++i)
    {
        result.mData[i] = +mData[i];
    }
    return result;
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> VectorBase<Dimension, Type>::operator-() const
{
    Vector<Dimension, Type> result;
    for (size_t i = 0; i < Dimension; ++i)
    {
        result.mData[i] = -mData[i];
    }
    return result;
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> VectorBase<Dimension, Type>::operator+(
    const Vector<Dimension, Type> &other) const
{
    Vector<Dimension, Type> result;
    for (size_t i = 0; i < Dimension; ++i)
    {
        result.mData[i] = mData[i] + other.mData[i];
    }
    return result;
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> VectorBase<Dimension, Type>::operator-(
    const Vector<Dimension, Type> &other) const
{
    Vector<Dimension, Type> result;
    for (size_t i = 0; i < Dimension; ++i)
    {
        result.mData[i] = mData[i] - other.mData[i];
    }
    return result;
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> VectorBase<Dimension, Type>::operator*(
    const Vector<Dimension, Type> &other) const
{
    Vector<Dimension, Type> result;
    for (size_t i = 0; i < Dimension; ++i)
    {
        result.mData[i] = mData[i] * other.mData[i];
    }
    return result;
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> VectorBase<Dimension, Type>::operator/(
    const Vector<Dimension, Type> &other) const
{
    Vector<Dimension, Type> result;
    for (size_t i = 0; i < Dimension; ++i)
    {
        result.mData[i] = mData[i] / other.mData[i];
    }
    return result;
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> VectorBase<Dimension, Type>::operator*(Type other) const
{
    Vector<Dimension, Type> result;
    for (size_t i = 0; i < Dimension; ++i)
    {
        result.mData[i] = mData[i] * other;
    }
    return result;
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> VectorBase<Dimension, Type>::operator/(Type other) const
{
    Vector<Dimension, Type> result;
    for (size_t i = 0; i < Dimension; ++i)
    {
        result.mData[i] = mData[i] / other;
    }
    return result;
}

// Implementation of compound arithmetic operations
template <size_t Dimension, typename Type>
Vector<Dimension, Type> &VectorBase<Dimension, Type>::operator+=(
    const Vector<Dimension, Type> &other)
{
    for (size_t i = 0; i < Dimension; ++i)
    {
        mData[i] += other.mData[i];
    }
    return *static_cast<Vector<Dimension, Type> *>(this);
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> &VectorBase<Dimension, Type>::operator-=(
    const Vector<Dimension, Type> &other)
{
    for (size_t i = 0; i < Dimension; ++i)
    {
        mData[i] -= other.mData[i];
    }
    return *static_cast<Vector<Dimension, Type> *>(this);
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> &VectorBase<Dimension, Type>::operator*=(
    const Vector<Dimension, Type> &other)
{
    for (size_t i = 0; i < Dimension; ++i)
    {
        mData[i] *= other.mData[i];
    }
    return *static_cast<Vector<Dimension, Type> *>(this);
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> &VectorBase<Dimension, Type>::operator/=(
    const Vector<Dimension, Type> &other)
{
    for (size_t i = 0; i < Dimension; ++i)
    {
        mData[i] /= other.mData[i];
    }
    return *static_cast<Vector<Dimension, Type> *>(this);
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> &VectorBase<Dimension, Type>::operator*=(Type other)
{
    for (size_t i = 0; i < Dimension; ++i)
    {
        mData[i] *= other;
    }
    return *static_cast<Vector<Dimension, Type> *>(this);
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> &VectorBase<Dimension, Type>::operator/=(Type other)
{
    for (size_t i = 0; i < Dimension; ++i)
    {
        mData[i] /= other;
    }
    return *static_cast<Vector<Dimension, Type> *>(this);
}

// Implementation of comparison operators
template <size_t Dimension, typename Type>
bool VectorBase<Dimension, Type>::operator==(const Vector<Dimension, Type> &other) const
{
    for (size_t i = 0; i < Dimension; ++i)
    {
        if (mData[i] != other.mData[i])
        {
            return false;
        }
    }
    return true;
}

template <size_t Dimension, typename Type>
bool VectorBase<Dimension, Type>::operator!=(const Vector<Dimension, Type> &other) const
{
    return !(*this == other);
}

// Implementation of other arithmetic operations
template <size_t Dimension, typename Type>
Type VectorBase<Dimension, Type>::length() const
{
    static_assert(std::is_floating_point<Type>::value,
                  "VectorN::length is only defined for floating point vectors");
    return std::sqrt(lengthSquared());
}

template <size_t Dimension, typename Type>
Type VectorBase<Dimension, Type>::lengthSquared() const
{
    return dot(*this);
}

template <size_t Dimension, typename Type>
Type VectorBase<Dimension, Type>::dot(const VectorBase<Dimension, Type> &other) const
{
    Type sum = Type();
    for (size_t i = 0; i < Dimension; ++i)
    {
        sum += mData[i] * other.mData[i];
    }
    return sum;
}

template <size_t Dimension, typename Type>
std::ostream &operator<<(std::ostream &ostream, const VectorBase<Dimension, Type> &vector)
{
    ostream << "[ ";
    for (size_t elementIdx = 0; elementIdx < Dimension; elementIdx++)
    {
        if (elementIdx > 0)
        {
            ostream << ", ";
        }
        ostream << vector.data()[elementIdx];
    }
    ostream << " ]";
    return ostream;
}

template <size_t Dimension, typename Type>
Vector<Dimension, Type> VectorBase<Dimension, Type>::normalized() const
{
    static_assert(std::is_floating_point<Type>::value,
                  "VectorN::normalized is only defined for floating point vectors");
    return *this / length();
}

template <typename Type>
std::ostream &operator<<(std::ostream &ostream, const Vector<2, Type> &vector)
{
    return ostream << static_cast<const VectorBase<2, Type> &>(vector);
}

template <typename Type>
Vector<3, Type> Vector<3, Type>::cross(const Vector<3, Type> &other) const
{
    return Vector<3, Type>(y() * other.z() - z() * other.y(), z() * other.x() - x() * other.z(),
                           x() * other.y() - y() * other.x());
}

template <typename Type>
std::ostream &operator<<(std::ostream &ostream, const Vector<3, Type> &vector)
{
    return ostream << static_cast<const VectorBase<3, Type> &>(vector);
}

template <typename Type>
std::ostream &operator<<(std::ostream &ostream, const Vector<4, Type> &vector)
{
    return ostream << static_cast<const VectorBase<4, Type> &>(vector);
}

}  // namespace angle

#endif  // COMMON_VECTOR_UTILS_H_

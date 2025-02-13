/* Copyright (c) 2011 Khaled Mamou (kmamou at gmail dot com)
 All rights reserved.
 
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 3. The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#ifndef VHACD_VECTOR_H
#define VHACD_VECTOR_H
#include <iostream>
#include <math.h>

namespace VHACD {
//!    Vector dim 3.
template <typename T>
class Vec3 {
public:
    T& operator[](size_t i) { return m_data[i]; }
    const T& operator[](size_t i) const { return m_data[i]; }
    T& X();
    T& Y();
    T& Z();
    const T& X() const;
    const T& Y() const;
    const T& Z() const;
    void Normalize();
    T GetNorm() const;
    void operator=(const Vec3& rhs);
    void operator+=(const Vec3& rhs);
    void operator-=(const Vec3& rhs);
    void operator-=(T a);
    void operator+=(T a);
    void operator/=(T a);
    void operator*=(T a);
    Vec3 operator^(const Vec3& rhs) const;
    T operator*(const Vec3& rhs) const;
    Vec3 operator+(const Vec3& rhs) const;
    Vec3 operator-(const Vec3& rhs) const;
    Vec3 operator-() const;
    Vec3 operator*(T rhs) const;
    Vec3 operator/(T rhs) const;
    bool operator<(const Vec3& rhs) const;
    bool operator>(const Vec3& rhs) const;
    Vec3();
    Vec3(T a);
    Vec3(T x, T y, T z);
    Vec3(const Vec3& rhs);
    /*virtual*/ ~Vec3(void);

    // Compute the center of this bounding box and return the diagonal length
    T GetCenter(const Vec3 &bmin, const Vec3 &bmax)
    {
        X() = (bmin.X() + bmax.X())*0.5;
        Y() = (bmin.Y() + bmax.Y())*0.5;
        Z() = (bmin.Z() + bmax.Z())*0.5;
        T dx = bmax.X() - bmin.X();
        T dy = bmax.Y() - bmin.Y();
        T dz = bmax.Z() - bmin.Z();
        T diagonal = T(sqrt(dx*dx + dy*dy + dz*dz));
        return diagonal;
    }

    // Update the min/max values relative to this point
    void UpdateMinMax(Vec3 &bmin,Vec3 &bmax) const
    {
        if (X() < bmin.X())
        {
            bmin.X() = X();
        }
        if (Y() < bmin.Y())
        {
            bmin.Y() = Y();
        }
        if (Z() < bmin.Z())
        {
            bmin.Z() = Z();
        }
        if (X() > bmax.X())
        {
            bmax.X() = X();
        }
        if (X() > bmax.X())
        {
            bmax.X() = X();
        }
        if (Y() > bmax.Y())
        {
            bmax.Y() = Y();
        }
        if (Z() > bmax.Z())
        {
            bmax.Z() = Z();
        }
    }

    // Returns the squared distance between these two points
    T GetDistanceSquared(const Vec3 &p) const
    {
        T dx = X() - p.X();
        T dy = Y() - p.Y();
        T dz = Z() - p.Z();
        return dx*dx + dy*dy + dz*dz;
    }

    T GetDistance(const Vec3 &p) const
    {
        return sqrt(GetDistanceSquared(p));
    }

    // Returns the raw vector data as a pointer
    T* GetData(void) 
    {
        return m_data;
    }
private:
    T m_data[3];
};
//!    Vector dim 2.
template <typename T>
class Vec2 {
public:
    T& operator[](size_t i) { return m_data[i]; }
    const T& operator[](size_t i) const { return m_data[i]; }
    T& X();
    T& Y();
    const T& X() const;
    const T& Y() const;
    void Normalize();
    T GetNorm() const;
    void operator=(const Vec2& rhs);
    void operator+=(const Vec2& rhs);
    void operator-=(const Vec2& rhs);
    void operator-=(T a);
    void operator+=(T a);
    void operator/=(T a);
    void operator*=(T a);
    T operator^(const Vec2& rhs) const;
    T operator*(const Vec2& rhs) const;
    Vec2 operator+(const Vec2& rhs) const;
    Vec2 operator-(const Vec2& rhs) const;
    Vec2 operator-() const;
    Vec2 operator*(T rhs) const;
    Vec2 operator/(T rhs) const;
    Vec2();
    Vec2(T a);
    Vec2(T x, T y);
    Vec2(const Vec2& rhs);
    /*virtual*/ ~Vec2(void);

private:
    T m_data[2];
};

template <typename T>
const bool Colinear(const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c);
template <typename T>
const T ComputeVolume4(const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c, const Vec3<T>& d);
}
#include "vhacdVector.inl" // template implementation
#endif
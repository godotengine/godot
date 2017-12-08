// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MATH_MATRIX_INL
#define NV_MATH_MATRIX_INL

#include "Matrix.h"

namespace nv
{
    inline Matrix3::Matrix3() {}
    
    inline Matrix3::Matrix3(float f)
    {
        for(int i = 0; i < 9; i++) {
            m_data[i] = f;
        }
    }

    inline Matrix3::Matrix3(identity_t)
    {
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                m_data[3*j+i] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    inline Matrix3::Matrix3(const Matrix3 & m)
    {
        for(int i = 0; i < 9; i++) {
            m_data[i] = m.m_data[i];
        }
    }
    
    inline Matrix3::Matrix3(Vector3::Arg v0, Vector3::Arg v1, Vector3::Arg v2)
    {
        m_data[0] = v0.x; m_data[1] = v0.y; m_data[2] = v0.z;
        m_data[3] = v1.x; m_data[4] = v1.y; m_data[5] = v1.z;
        m_data[6] = v2.x; m_data[7] = v2.y; m_data[8] = v2.z;
    }

    inline float Matrix3::data(uint idx) const
    {
        nvDebugCheck(idx < 9);
        return m_data[idx];
    }
    inline float & Matrix3::data(uint idx)
    {
        nvDebugCheck(idx < 9);
        return m_data[idx];
    }
    inline float Matrix3::get(uint row, uint col) const
    {
        nvDebugCheck(row < 3 && col < 3);
        return m_data[col * 3 + row];
    }
    inline float Matrix3::operator()(uint row, uint col) const
    {
        nvDebugCheck(row < 3 && col < 3);
        return m_data[col * 3 + row];
    }
    inline float & Matrix3::operator()(uint row, uint col)
    {
        nvDebugCheck(row < 3 && col < 3);
        return m_data[col * 3 + row];
    }

    inline Vector3 Matrix3::row(uint i) const
    {
        nvDebugCheck(i < 3);
        return Vector3(get(i, 0), get(i, 1), get(i, 2));
    }
    inline Vector3 Matrix3::column(uint i) const
    {
        nvDebugCheck(i < 3);
        return Vector3(get(0, i), get(1, i), get(2, i));
    }

    inline void Matrix3::operator*=(float s)
    {
        for(int i = 0; i < 9; i++) {
            m_data[i] *= s;
        }
    }

    inline void Matrix3::operator/=(float s)
    {
        float is = 1.0f /s;
        for(int i = 0; i < 9; i++) {
            m_data[i] *= is;
        }
    }

    inline void Matrix3::operator+=(const Matrix3 & m)
    {
        for(int i = 0; i < 9; i++) {
            m_data[i] += m.m_data[i];
        }
    }

    inline void Matrix3::operator-=(const Matrix3 & m)
    {
        for(int i = 0; i < 9; i++) {
            m_data[i] -= m.m_data[i];
        }
    }

    inline Matrix3 operator+(const Matrix3 & a, const Matrix3 & b)
    {
        Matrix3 m = a;
        m += b;
        return m;
    }

    inline Matrix3 operator-(const Matrix3 & a, const Matrix3 & b)
    {
        Matrix3 m = a;
        m -= b;
        return m;
    }

    inline Matrix3 operator*(const Matrix3 & a, float s)
    {
        Matrix3 m = a;
        m *= s;
        return m;
    }

    inline Matrix3 operator*(float s, const Matrix3 & a)
    {
        Matrix3 m = a;
        m *= s;
        return m;
    }

    inline Matrix3 operator/(const Matrix3 & a, float s)
    {
        Matrix3 m = a;
        m /= s;
        return m;
    }

    inline Matrix3 mul(const Matrix3 & a, const Matrix3 & b)
    {
        Matrix3 m;

        for(int i = 0; i < 3; i++) {
            const float ai0 = a(i,0), ai1 = a(i,1), ai2 = a(i,2);
            m(i, 0) = ai0 * b(0,0) + ai1 * b(1,0) + ai2 * b(2,0);
            m(i, 1) = ai0 * b(0,1) + ai1 * b(1,1) + ai2 * b(2,1);
            m(i, 2) = ai0 * b(0,2) + ai1 * b(1,2) + ai2 * b(2,2);
        }

        return m;
    }

    inline Matrix3 operator*(const Matrix3 & a, const Matrix3 & b)
    {
        return mul(a, b);
    }

    // Transform the given 3d vector with the given matrix.
    inline Vector3 transform(const Matrix3 & m, const Vector3 & p)
    {
        return Vector3(
            p.x * m(0,0) + p.y * m(0,1) + p.z * m(0,2),
            p.x * m(1,0) + p.y * m(1,1) + p.z * m(1,2),
            p.x * m(2,0) + p.y * m(2,1) + p.z * m(2,2));
    }

    inline void Matrix3::scale(float s)
    {
        for (int i = 0; i < 9; i++) {
            m_data[i] *= s;
        }
    }

    inline void Matrix3::scale(Vector3::Arg s)
    {
        m_data[0] *= s.x; m_data[1] *= s.x; m_data[2] *= s.x;
        m_data[3] *= s.y; m_data[4] *= s.y; m_data[5] *= s.y;
        m_data[6] *= s.z; m_data[7] *= s.z; m_data[8] *= s.z;
    }

    inline float Matrix3::determinant() const
    {
        return 
            get(0,0) * get(1,1) * get(2,2) + 
            get(0,1) * get(1,2) * get(2,0) + 
            get(0,2) * get(1,0) * get(2,1) -
            get(0,2) * get(1,1) * get(2,0) - 
            get(0,1) * get(1,0) * get(2,2) -
            get(0,0) * get(1,2) * get(2,1);
    }

    // Inverse using Cramer's rule.
    inline Matrix3 inverseCramer(const Matrix3 & m)
    {
        const float det = m.determinant();
        if (equal(det, 0.0f, 0.0f)) {
            return Matrix3(0);
        }

        Matrix3 r;

        r.data(0) =  - m.data(5) * m.data(7) + m.data(4) * m.data(8);
        r.data(1) =  + m.data(5) * m.data(6) - m.data(3) * m.data(8);
        r.data(2) =  - m.data(4) * m.data(6) + m.data(3) * m.data(7);

        r.data(3) =  + m.data(2) * m.data(7) - m.data(1) * m.data(8);
        r.data(4) =  - m.data(2) * m.data(6) + m.data(0) * m.data(8);
        r.data(5) =  + m.data(1) * m.data(6) - m.data(0) * m.data(7);

        r.data(6) =  - m.data(2) * m.data(4) + m.data(1) * m.data(5);
        r.data(7) =  + m.data(2) * m.data(3) - m.data(0) * m.data(5);
        r.data(8) =  - m.data(1) * m.data(3) + m.data(0) * m.data(4);

        r.scale(1.0f / det);

        return r;
    }



    inline Matrix::Matrix()
    {
    }

    inline Matrix::Matrix(float f)
    {
        for(int i = 0; i < 16; i++) {
            m_data[i] = 0.0f;
        }
    }

    inline Matrix::Matrix(identity_t)
    {
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                m_data[4*j+i] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    inline Matrix::Matrix(const Matrix & m)
    {
        for(int i = 0; i < 16; i++) {
            m_data[i] = m.m_data[i];
        }
    }

    inline Matrix::Matrix(const Matrix3 & m)
    {
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                operator()(i, j) = m.get(i, j);
            }
        }
        for(int i = 0; i < 4; i++) {
            operator()(3, i) = 0;
            operator()(i, 3) = 0;
        }
    }

    inline Matrix::Matrix(Vector4::Arg v0, Vector4::Arg v1, Vector4::Arg v2, Vector4::Arg v3)
    {
        m_data[ 0] = v0.x; m_data[ 1] = v0.y; m_data[ 2] = v0.z; m_data[ 3] = v0.w;
        m_data[ 4] = v1.x; m_data[ 5] = v1.y; m_data[ 6] = v1.z; m_data[ 7] = v1.w;
        m_data[ 8] = v2.x; m_data[ 9] = v2.y; m_data[10] = v2.z; m_data[11] = v2.w;
        m_data[12] = v3.x; m_data[13] = v3.y; m_data[14] = v3.z; m_data[15] = v3.w;
    }

    /*inline Matrix::Matrix(const float m[])
    {
        for(int i = 0; i < 16; i++) {
            m_data[i] = m[i];
        }
    }*/


    // Accessors
    inline float Matrix::data(uint idx) const
    {
        nvDebugCheck(idx < 16);
        return m_data[idx];
    }
    inline float & Matrix::data(uint idx)
    {
        nvDebugCheck(idx < 16);
        return m_data[idx];
    }
    inline float Matrix::get(uint row, uint col) const
    {
        nvDebugCheck(row < 4 && col < 4);
        return m_data[col * 4 + row];
    }
    inline float Matrix::operator()(uint row, uint col) const
    {
        nvDebugCheck(row < 4 && col < 4);
        return m_data[col * 4 + row];
    }
    inline float & Matrix::operator()(uint row, uint col)
    {
        nvDebugCheck(row < 4 && col < 4);
        return m_data[col * 4 + row];
    }

    inline const float * Matrix::ptr() const
    {
        return m_data;
    }

    inline Vector4 Matrix::row(uint i) const
    {
        nvDebugCheck(i < 4);
        return Vector4(get(i, 0), get(i, 1), get(i, 2), get(i, 3));
    }

    inline Vector4 Matrix::column(uint i) const
    {
        nvDebugCheck(i < 4);
        return Vector4(get(0, i), get(1, i), get(2, i), get(3, i));
    }

    inline void Matrix::zero()
    {
        m_data[0] = 0; m_data[1] = 0; m_data[2] = 0; m_data[3] = 0;
        m_data[4] = 0; m_data[5] = 0; m_data[6] = 0; m_data[7] = 0;
        m_data[8] = 0; m_data[9] = 0; m_data[10] = 0; m_data[11] = 0;
        m_data[12] = 0; m_data[13] = 0; m_data[14] = 0; m_data[15] = 0;
    }

    inline void Matrix::identity()
    {
        m_data[0] = 1; m_data[1] = 0; m_data[2] = 0; m_data[3] = 0;
        m_data[4] = 0; m_data[5] = 1; m_data[6] = 0; m_data[7] = 0;
        m_data[8] = 0; m_data[9] = 0; m_data[10] = 1; m_data[11] = 0;
        m_data[12] = 0; m_data[13] = 0; m_data[14] = 0; m_data[15] = 1;
    }

    // Apply scale.
    inline void Matrix::scale(float s)
    {
        m_data[0] *= s; m_data[1] *= s; m_data[2] *= s; m_data[3] *= s;
        m_data[4] *= s; m_data[5] *= s; m_data[6] *= s; m_data[7] *= s;
        m_data[8] *= s; m_data[9] *= s; m_data[10] *= s; m_data[11] *= s;
        m_data[12] *= s; m_data[13] *= s; m_data[14] *= s; m_data[15] *= s;
    }

    // Apply scale.
    inline void Matrix::scale(Vector3::Arg s)
    {
        m_data[0] *= s.x; m_data[1] *= s.x; m_data[2] *= s.x; m_data[3] *= s.x;
        m_data[4] *= s.y; m_data[5] *= s.y; m_data[6] *= s.y; m_data[7] *= s.y;
        m_data[8] *= s.z; m_data[9] *= s.z; m_data[10] *= s.z; m_data[11] *= s.z;
    }

    // Apply translation.
    inline void Matrix::translate(Vector3::Arg t)
    {
        m_data[12] = m_data[0] * t.x + m_data[4] * t.y + m_data[8]  * t.z + m_data[12];
        m_data[13] = m_data[1] * t.x + m_data[5] * t.y + m_data[9]  * t.z + m_data[13];
        m_data[14] = m_data[2] * t.x + m_data[6] * t.y + m_data[10] * t.z + m_data[14];
        m_data[15] = m_data[3] * t.x + m_data[7] * t.y + m_data[11] * t.z + m_data[15];
    }

    Matrix rotation(float theta, float v0, float v1, float v2);

    // Apply rotation.
    inline void Matrix::rotate(float theta, float v0, float v1, float v2)
    {
        Matrix R(rotation(theta, v0, v1, v2));
        apply(R);
    }

    // Apply transform.
    inline void Matrix::apply(Matrix::Arg m)
    {
        nvDebugCheck(this != &m);

        for(int i = 0; i < 4; i++) {
            const float ai0 = get(i,0), ai1 = get(i,1), ai2 = get(i,2), ai3 = get(i,3);
            m_data[0 + i] = ai0 * m(0,0) + ai1 * m(1,0) + ai2 * m(2,0) + ai3 * m(3,0);
            m_data[4 + i] = ai0 * m(0,1) + ai1 * m(1,1) + ai2 * m(2,1) + ai3 * m(3,1);
            m_data[8 + i] = ai0 * m(0,2) + ai1 * m(1,2) + ai2 * m(2,2) + ai3 * m(3,2);
            m_data[12+ i] = ai0 * m(0,3) + ai1 * m(1,3) + ai2 * m(2,3) + ai3 * m(3,3);
        }
    }

    // Get scale matrix.
    inline Matrix scale(Vector3::Arg s)
    {
        Matrix m(identity);
        m(0,0) = s.x;
        m(1,1) = s.y;
        m(2,2) = s.z;
        return m;
    }

    // Get scale matrix.
    inline Matrix scale(float s)
    {
        Matrix m(identity);
        m(0,0) = m(1,1) = m(2,2) = s;
        return m;
    }

    // Get translation matrix.
    inline Matrix translation(Vector3::Arg t)
    {
        Matrix m(identity);
        m(0,3) = t.x;
        m(1,3) = t.y;
        m(2,3) = t.z;
        return m;
    }

    // Get rotation matrix.
    inline Matrix rotation(float theta, float v0, float v1, float v2)
    {
        float cost = cosf(theta);
        float sint = sinf(theta);

        Matrix m(identity);

        if( 1 == v0 && 0 == v1 && 0 == v2 ) {
            m(1,1) = cost; m(2,1) = -sint;
            m(1,2) = sint; m(2,2) = cost;
        }
        else if( 0 == v0  && 1 == v1 && 0 == v2 ) {
            m(0,0) = cost; m(2,0) = sint;
            m(1,2) = -sint; m(2,2) = cost;
        }
        else if( 0 == v0 && 0 == v1 && 1 == v2 ) {
            m(0,0) = cost; m(1,0) = -sint;
            m(0,1) = sint; m(1,1) = cost;
        } 
        else {
            float a2, b2, c2;
            a2 = v0 * v0;
            b2 = v1 * v1;
            c2 = v2 * v2;

            float iscale = 1.0f / sqrtf(a2 + b2 + c2);
            v0 *= iscale;
            v1 *= iscale;
            v2 *= iscale;

            float abm, acm, bcm;
            float mcos, asin, bsin, csin;
            mcos = 1.0f - cost;
            abm = v0 * v1 * mcos;
            acm = v0 * v2 * mcos;
            bcm = v1 * v2 * mcos;
            asin = v0 * sint;
            bsin = v1 * sint;
            csin = v2 * sint;
            m(0,0) = a2 * mcos + cost;
            m(1,0) = abm - csin;
            m(2,0) = acm + bsin;
            m(3,0) = abm + csin;
            m(1,1) = b2 * mcos + cost;
            m(2,1) = bcm - asin;
            m(3,1) = acm - bsin;
            m(1,2) = bcm + asin;
            m(2,2) = c2 * mcos + cost;
        }
        return m;
    }

    //Matrix rotation(float yaw, float pitch, float roll);
    //Matrix skew(float angle, Vector3::Arg v1, Vector3::Arg v2);

    // Get frustum matrix.
    inline Matrix frustum(float xmin, float xmax, float ymin, float ymax, float zNear, float zFar)
    {
        Matrix m(0.0f);

        float doubleznear = 2.0f * zNear;
        float one_deltax = 1.0f / (xmax - xmin);
        float one_deltay = 1.0f / (ymax - ymin);
        float one_deltaz = 1.0f / (zFar - zNear);

        m(0,0) = doubleznear * one_deltax;
        m(1,1) = doubleznear * one_deltay;
        m(0,2) = (xmax + xmin) * one_deltax;
        m(1,2) = (ymax + ymin) * one_deltay;
        m(2,2) = -(zFar + zNear) * one_deltaz;
        m(3,2) = -1.0f;
        m(2,3) = -(zFar * doubleznear) * one_deltaz;

        return m;
    }

    // Get inverse frustum matrix.
    inline Matrix frustumInverse(float xmin, float xmax, float ymin, float ymax, float zNear, float zFar)
    {
        Matrix m(0.0f);

        float one_doubleznear = 1.0f / (2.0f * zNear);
        float one_doubleznearzfar = 1.0f / (2.0f * zNear * zFar);

        m(0,0) = (xmax - xmin) * one_doubleznear;
        m(0,3) = (xmax + xmin) * one_doubleznear;
        m(1,1) = (ymax - ymin) * one_doubleznear;
        m(1,3) = (ymax + ymin) * one_doubleznear;
        m(2,3) = -1;
        m(3,2) = -(zFar - zNear) * one_doubleznearzfar;
        m(3,3) = (zFar + zNear) * one_doubleznearzfar;

        return m;
    }

    // Get infinite frustum matrix.
    inline Matrix frustum(float xmin, float xmax, float ymin, float ymax, float zNear)
    {
        Matrix m(0.0f);

        float doubleznear = 2.0f * zNear;
        float one_deltax = 1.0f / (xmax - xmin);
        float one_deltay = 1.0f / (ymax - ymin);
        float nudge = 1.0; // 0.999;

        m(0,0) = doubleznear * one_deltax;
        m(1,1) = doubleznear * one_deltay;
        m(0,2) = (xmax + xmin) * one_deltax;
        m(1,2) = (ymax + ymin) * one_deltay;
        m(2,2) = -1.0f * nudge;
        m(3,2) = -1.0f;
        m(2,3) = -doubleznear * nudge;

        return m;
    }

    // Get perspective matrix.
    inline Matrix perspective(float fovy, float aspect, float zNear, float zFar)
    {
        float xmax = zNear * tan(fovy / 2);
        float xmin = -xmax;

        float ymax = xmax / aspect;
        float ymin = -ymax;

        return frustum(xmin, xmax, ymin, ymax, zNear, zFar);	
    }

    // Get inverse perspective matrix.
    inline Matrix perspectiveInverse(float fovy, float aspect, float zNear, float zFar)
    {
        float xmax = zNear * tan(fovy / 2);
        float xmin = -xmax;

        float ymax = xmax / aspect;
        float ymin = -ymax;

        return frustumInverse(xmin, xmax, ymin, ymax, zNear, zFar);	
    }

    // Get infinite perspective matrix.
    inline Matrix perspective(float fovy, float aspect, float zNear)
    {
        float x = zNear * tan(fovy / 2);
        float y = x / aspect;
        return frustum( -x, x, -y, y, zNear );	
    }

    // Get matrix determinant.
    inline float Matrix::determinant() const
    {
        return 
            m_data[3] * m_data[6] * m_data[ 9] * m_data[12] - m_data[2] * m_data[7] * m_data[ 9] * m_data[12] - m_data[3] * m_data[5] * m_data[10] * m_data[12] + m_data[1] * m_data[7] * m_data[10] * m_data[12] +
            m_data[2] * m_data[5] * m_data[11] * m_data[12] - m_data[1] * m_data[6] * m_data[11] * m_data[12] - m_data[3] * m_data[6] * m_data[ 8] * m_data[13] + m_data[2] * m_data[7] * m_data[ 8] * m_data[13] +
            m_data[3] * m_data[4] * m_data[10] * m_data[13] - m_data[0] * m_data[7] * m_data[10] * m_data[13] - m_data[2] * m_data[4] * m_data[11] * m_data[13] + m_data[0] * m_data[6] * m_data[11] * m_data[13] +
            m_data[3] * m_data[5] * m_data[ 8] * m_data[14] - m_data[1] * m_data[7] * m_data[ 8] * m_data[14] - m_data[3] * m_data[4] * m_data[ 9] * m_data[14] + m_data[0] * m_data[7] * m_data[ 9] * m_data[14] +
            m_data[1] * m_data[4] * m_data[11] * m_data[14] - m_data[0] * m_data[5] * m_data[11] * m_data[14] - m_data[2] * m_data[5] * m_data[ 8] * m_data[15] + m_data[1] * m_data[6] * m_data[ 8] * m_data[15] +
            m_data[2] * m_data[4] * m_data[ 9] * m_data[15] - m_data[0] * m_data[6] * m_data[ 9] * m_data[15] - m_data[1] * m_data[4] * m_data[10] * m_data[15] + m_data[0] * m_data[5] * m_data[10] * m_data[15];
    }

    inline Matrix transpose(Matrix::Arg m)
    {
        Matrix r;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                r(i, j) = m(j, i);
            }
        }
        return r;
    }

    // Inverse using Cramer's rule.
    inline Matrix inverseCramer(Matrix::Arg m)
    {
        Matrix r;
        r.data( 0) = m.data(6)*m.data(11)*m.data(13) - m.data(7)*m.data(10)*m.data(13) + m.data(7)*m.data(9)*m.data(14) - m.data(5)*m.data(11)*m.data(14) - m.data(6)*m.data(9)*m.data(15) + m.data(5)*m.data(10)*m.data(15);
        r.data( 1) = m.data(3)*m.data(10)*m.data(13) - m.data(2)*m.data(11)*m.data(13) - m.data(3)*m.data(9)*m.data(14) + m.data(1)*m.data(11)*m.data(14) + m.data(2)*m.data(9)*m.data(15) - m.data(1)*m.data(10)*m.data(15);
        r.data( 2) = m.data(2)*m.data( 7)*m.data(13) - m.data(3)*m.data( 6)*m.data(13) + m.data(3)*m.data(5)*m.data(14) - m.data(1)*m.data( 7)*m.data(14) - m.data(2)*m.data(5)*m.data(15) + m.data(1)*m.data( 6)*m.data(15);
        r.data( 3) = m.data(3)*m.data( 6)*m.data( 9) - m.data(2)*m.data( 7)*m.data( 9) - m.data(3)*m.data(5)*m.data(10) + m.data(1)*m.data( 7)*m.data(10) + m.data(2)*m.data(5)*m.data(11) - m.data(1)*m.data( 6)*m.data(11);
        r.data( 4) = m.data(7)*m.data(10)*m.data(12) - m.data(6)*m.data(11)*m.data(12) - m.data(7)*m.data(8)*m.data(14) + m.data(4)*m.data(11)*m.data(14) + m.data(6)*m.data(8)*m.data(15) - m.data(4)*m.data(10)*m.data(15);
        r.data( 5) = m.data(2)*m.data(11)*m.data(12) - m.data(3)*m.data(10)*m.data(12) + m.data(3)*m.data(8)*m.data(14) - m.data(0)*m.data(11)*m.data(14) - m.data(2)*m.data(8)*m.data(15) + m.data(0)*m.data(10)*m.data(15);
        r.data( 6) = m.data(3)*m.data( 6)*m.data(12) - m.data(2)*m.data( 7)*m.data(12) - m.data(3)*m.data(4)*m.data(14) + m.data(0)*m.data( 7)*m.data(14) + m.data(2)*m.data(4)*m.data(15) - m.data(0)*m.data( 6)*m.data(15);
        r.data( 7) = m.data(2)*m.data( 7)*m.data( 8) - m.data(3)*m.data( 6)*m.data( 8) + m.data(3)*m.data(4)*m.data(10) - m.data(0)*m.data( 7)*m.data(10) - m.data(2)*m.data(4)*m.data(11) + m.data(0)*m.data( 6)*m.data(11);
        r.data( 8) = m.data(5)*m.data(11)*m.data(12) - m.data(7)*m.data( 9)*m.data(12) + m.data(7)*m.data(8)*m.data(13) - m.data(4)*m.data(11)*m.data(13) - m.data(5)*m.data(8)*m.data(15) + m.data(4)*m.data( 9)*m.data(15);
        r.data( 9) = m.data(3)*m.data( 9)*m.data(12) - m.data(1)*m.data(11)*m.data(12) - m.data(3)*m.data(8)*m.data(13) + m.data(0)*m.data(11)*m.data(13) + m.data(1)*m.data(8)*m.data(15) - m.data(0)*m.data( 9)*m.data(15);
        r.data(10) = m.data(1)*m.data( 7)*m.data(12) - m.data(3)*m.data( 5)*m.data(12) + m.data(3)*m.data(4)*m.data(13) - m.data(0)*m.data( 7)*m.data(13) - m.data(1)*m.data(4)*m.data(15) + m.data(0)*m.data( 5)*m.data(15);
        r.data(11) = m.data(3)*m.data( 5)*m.data( 8) - m.data(1)*m.data( 7)*m.data( 8) - m.data(3)*m.data(4)*m.data( 9) + m.data(0)*m.data( 7)*m.data( 9) + m.data(1)*m.data(4)*m.data(11) - m.data(0)*m.data( 5)*m.data(11);
        r.data(12) = m.data(6)*m.data( 9)*m.data(12) - m.data(5)*m.data(10)*m.data(12) - m.data(6)*m.data(8)*m.data(13) + m.data(4)*m.data(10)*m.data(13) + m.data(5)*m.data(8)*m.data(14) - m.data(4)*m.data( 9)*m.data(14);
        r.data(13) = m.data(1)*m.data(10)*m.data(12) - m.data(2)*m.data( 9)*m.data(12) + m.data(2)*m.data(8)*m.data(13) - m.data(0)*m.data(10)*m.data(13) - m.data(1)*m.data(8)*m.data(14) + m.data(0)*m.data( 9)*m.data(14);
        r.data(14) = m.data(2)*m.data( 5)*m.data(12) - m.data(1)*m.data( 6)*m.data(12) - m.data(2)*m.data(4)*m.data(13) + m.data(0)*m.data( 6)*m.data(13) + m.data(1)*m.data(4)*m.data(14) - m.data(0)*m.data( 5)*m.data(14);
        r.data(15) = m.data(1)*m.data( 6)*m.data( 8) - m.data(2)*m.data( 5)*m.data( 8) + m.data(2)*m.data(4)*m.data( 9) - m.data(0)*m.data( 6)*m.data( 9) - m.data(1)*m.data(4)*m.data(10) + m.data(0)*m.data( 5)*m.data(10);
        r.scale(1.0f / m.determinant());
        return r;
    }

    inline Matrix isometryInverse(Matrix::Arg m)
    {
        Matrix r(identity);

        // transposed 3x3 upper left matrix
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                r(i, j) = m(j, i);
            }
        }

        // translate by the negative offsets
        r.translate(-Vector3(m.data(12), m.data(13), m.data(14)));

        return r;
    }

    // Transform the given 3d point with the given matrix.
    inline Vector3 transformPoint(Matrix::Arg m, Vector3::Arg p)
    {
        return Vector3(
            p.x * m(0,0) + p.y * m(0,1) + p.z * m(0,2) + m(0,3),
            p.x * m(1,0) + p.y * m(1,1) + p.z * m(1,2) + m(1,3),
            p.x * m(2,0) + p.y * m(2,1) + p.z * m(2,2) + m(2,3));
    }

    // Transform the given 3d vector with the given matrix.
    inline Vector3 transformVector(Matrix::Arg m, Vector3::Arg p)
    {
        return Vector3(
            p.x * m(0,0) + p.y * m(0,1) + p.z * m(0,2),
            p.x * m(1,0) + p.y * m(1,1) + p.z * m(1,2),
            p.x * m(2,0) + p.y * m(2,1) + p.z * m(2,2));
    }

    // Transform the given 4d vector with the given matrix.
    inline Vector4 transform(Matrix::Arg m, Vector4::Arg p)
    {
        return Vector4(
            p.x * m(0,0) + p.y * m(0,1) + p.z * m(0,2) + p.w * m(0,3),
            p.x * m(1,0) + p.y * m(1,1) + p.z * m(1,2) + p.w * m(1,3),
            p.x * m(2,0) + p.y * m(2,1) + p.z * m(2,2) + p.w * m(2,3),
            p.x * m(3,0) + p.y * m(3,1) + p.z * m(3,2) + p.w * m(3,3));
    }

    inline Matrix mul(Matrix::Arg a, Matrix::Arg b)
    {
        // @@ Is this the right order? mul(a, b) = b * a
        Matrix m = a;
        m.apply(b);
        return m;
    }

    inline void Matrix::operator+=(const Matrix & m)
    {
        for(int i = 0; i < 16; i++) {
            m_data[i] += m.m_data[i];
        }
    }

    inline void Matrix::operator-=(const Matrix & m)
    {
        for(int i = 0; i < 16; i++) {
            m_data[i] -= m.m_data[i];
        }
    }

    inline Matrix operator+(const Matrix & a, const Matrix & b)
    {
        Matrix m = a;
        m += b;
        return m;
    }

    inline Matrix operator-(const Matrix & a, const Matrix & b)
    {
        Matrix m = a;
        m -= b;
        return m;
    }


} // nv namespace


#if 0 // old code.
/** @name Special matrices. */
//@{
/** Generate a translation matrix. */
void TranslationMatrix(const Vec3 & v) {
    data[0] = 1; data[1] = 0; data[2] = 0; data[3] = 0;
    data[4] = 0; data[5] = 1; data[6] = 0; data[7] = 0;
    data[8] = 0; data[9] = 0; data[10] = 1; data[11] = 0;
    data[12] = v.x; data[13] = v.y; data[14] = v.z; data[15] = 1;
}

/** Rotate theta degrees around v. */
void RotationMatrix( float theta, float v0, float v1, float v2 ) {
    float cost = cos(theta);
    float sint = sin(theta);

    if( 1 == v0 && 0 == v1 && 0 == v2 ) {
        data[0] = 1.0f;	data[1] = 0.0f;	data[2] = 0.0f;	data[3] = 0.0f;
        data[4] = 0.0f;	data[5] = cost;	data[6] = -sint;data[7] = 0.0f;
        data[8] = 0.0f;	data[9] = sint;	data[10] = cost;data[11] = 0.0f;
        data[12] = 0.0f;data[13] = 0.0f;data[14] = 0.0f;data[15] = 1.0f;
    }
    else if( 0 == v0  && 1 == v1 && 0 == v2 ) {
        data[0] = cost;	data[1] = 0.0f;	data[2] = sint;	data[3] = 0.0f;
        data[4] = 0.0f;	data[5] = 1.0f;	data[6] = 0.0f;	data[7] = 0.0f;
        data[8] = -sint;data[9] = 0.0f;data[10] = cost;	data[11] = 0.0f;
        data[12] = 0.0f;data[13] = 0.0f;data[14] = 0.0f;data[15] = 1.0f;
    }
    else if( 0 == v0 && 0 == v1 && 1 == v2 ) {
        data[0] = cost;	data[1] = -sint;data[2] = 0.0f;	data[3] = 0.0f;
        data[4] = sint; data[5] = cost;	data[6] = 0.0f;	data[7] = 0.0f;
        data[8] = 0.0f;	data[9] = 0.0f;	data[10] = 1.0f;data[11] = 0.0f;
        data[12] = 0.0f;data[13] = 0.0f;data[14] = 0.0f;data[15] = 1.0f;
    } 
    else {
        //we need scale a,b,c to unit length.
        float a2, b2, c2;
        a2 = v0 * v0;
        b2 = v1 * v1;
        c2 = v2 * v2;

        float iscale = 1.0f / sqrtf(a2 + b2 + c2);
        v0 *= iscale;
        v1 *= iscale;
        v2 *= iscale;

        float abm, acm, bcm;
        float mcos, asin, bsin, csin;
        mcos = 1.0f - cost;
        abm = v0 * v1 * mcos;
        acm = v0 * v2 * mcos;
        bcm = v1 * v2 * mcos;
        asin = v0 * sint;
        bsin = v1 * sint;
        csin = v2 * sint;
        data[0] = a2 * mcos + cost;
        data[1] = abm - csin;
        data[2] = acm + bsin;
        data[3] = abm + csin;
        data[4] = 0.0f;
        data[5] = b2 * mcos + cost;
        data[6] = bcm - asin;
        data[7] = acm - bsin;
        data[8] = 0.0f;
        data[9] = bcm + asin;
        data[10] = c2 * mcos + cost;
        data[11] = 0.0f;
        data[12] = 0.0f;
        data[13] = 0.0f;
        data[14] = 0.0f;
        data[15] = 1.0f;
    }
}

/*
void SkewMatrix(float angle, const Vec3 & v1, const Vec3 & v2) {
v1.Normalize();
v2.Normalize();

Vec3 v3;
v3.Cross(v1, v2);
v3.Normalize();

// Get skew factor.
float costheta = Vec3DotProduct(v1, v2);
float sintheta = Real.Sqrt(1 - costheta * costheta);
float skew = tan(Trig.DegreesToRadians(angle) + acos(sintheta)) * sintheta - costheta;

// Build orthonormal matrix.
v1 = FXVector3.Cross(v3, v2);
v1.Normalize();

Matrix R = Matrix::Identity;
R[0, 0] = v3.X;	// Not sure this is in the correct order...
R[1, 0] = v3.Y;
R[2, 0] = v3.Z;
R[0, 1] = v1.X;
R[1, 1] = v1.Y;
R[2, 1] = v1.Z;
R[0, 2] = v2.X;
R[1, 2] = v2.Y;
R[2, 2] = v2.Z;

// Build skew matrix.
Matrix S = Matrix::Identity;
S[2, 1] = -skew;

// Return skew transform.
return R * S * R.Transpose;	// Not sure this is in the correct order...
}
*/

/**
* Generate rotation matrix for the euler angles. This is the same as computing
* 3 rotation matrices and multiplying them together in our custom order.
*
* @todo Have to recompute this code for our new convention.
**/
void RotationMatrix( float yaw, float pitch, float roll ) {
    float sy = sin(yaw+ToRadian(90));
    float cy = cos(yaw+ToRadian(90));
    float sp = sin(pitch-ToRadian(90));
    float cp = cos(pitch-ToRadian(90));
    float sr = sin(roll);
    float cr = cos(roll);

    data[0] = cr*cy + sr*sp*sy;
    data[1] = cp*sy;
    data[2] = -sr*cy + cr*sp*sy;
    data[3] = 0;

    data[4] = -cr*sy + sr*sp*cy;
    data[5] = cp*cy;
    data[6] = sr*sy + cr*sp*cy;
    data[7] = 0;

    data[8] = sr*cp;
    data[9] = -sp;
    data[10] = cr*cp;
    data[11] = 0;

    data[12] = 0;
    data[13] = 0;
    data[14] = 0;
    data[15] = 1;
}

/** Create a frustum matrix with the far plane at the infinity. */
void Frustum( float xmin, float xmax, float ymin, float ymax, float zNear, float zFar ) {
    float one_deltax, one_deltay, one_deltaz, doubleznear;

    doubleznear = 2.0f * zNear;
    one_deltax = 1.0f / (xmax - xmin);
    one_deltay = 1.0f / (ymax - ymin);
    one_deltaz = 1.0f / (zFar - zNear);

    data[0] = (float)(doubleznear * one_deltax);
    data[1] = 0.0f;
    data[2] = 0.0f;
    data[3] = 0.0f;
    data[4] = 0.0f;
    data[5] = (float)(doubleznear * one_deltay);
    data[6] = 0.f;
    data[7] = 0.f;
    data[8] = (float)((xmax + xmin) * one_deltax);
    data[9] = (float)((ymax + ymin) * one_deltay);
    data[10] = (float)(-(zFar + zNear) * one_deltaz);
    data[11] = -1.f;
    data[12] = 0.f;
    data[13] = 0.f;
    data[14] = (float)(-(zFar * doubleznear) * one_deltaz);
    data[15] = 0.f;
}

/** Create a frustum matrix with the far plane at the infinity. */
void FrustumInf( float xmin, float xmax, float ymin, float ymax, float zNear ) {
    float one_deltax, one_deltay, doubleznear, nudge;

    doubleznear = 2.0f * zNear;
    one_deltax = 1.0f / (xmax - xmin);
    one_deltay = 1.0f / (ymax - ymin);
    nudge = 1.0; // 0.999;

    data[0] = doubleznear * one_deltax;
    data[1] = 0.0f;
    data[2] = 0.0f;
    data[3] = 0.0f;

    data[4] = 0.0f;
    data[5] = doubleznear * one_deltay;
    data[6] = 0.f;
    data[7] = 0.f;

    data[8] = (xmax + xmin) * one_deltax;
    data[9] = (ymax + ymin) * one_deltay;
    data[10] = -1.0f * nudge;
    data[11] = -1.0f;

    data[12] = 0.f;
    data[13] = 0.f;
    data[14] = -doubleznear * nudge;
    data[15] = 0.f;
}

/** Create an inverse frustum matrix with the far plane at the infinity. */
void FrustumInfInv( float left, float right, float bottom, float top, float zNear ) {
    // this matrix is wrong (not tested floatly) I think it should be transposed.
    data[0] = (right - left) / (2 * zNear);
    data[1] = 0;
    data[2] = 0;
    data[3] = (right + left) / (2 * zNear);
    data[4] = 0;
    data[5] = (top - bottom) / (2 * zNear);
    data[6] = 0;
    data[7] = (top + bottom) / (2 * zNear);
    data[8] = 0;
    data[9] = 0;
    data[10] = 0;
    data[11] = -1;
    data[12] = 0;
    data[13] = 0;
    data[14] = -1 / (2 * zNear);
    data[15] = 1 / (2 * zNear);
}

/** Create an homogeneous projection matrix. */
void Perspective( float fov, float aspect, float zNear, float zFar ) {
    float xmin, xmax, ymin, ymax;

    xmax = zNear * tan( fov/2 );
    xmin = -xmax;

    ymax = xmax / aspect;
    ymin = -ymax;

    Frustum(xmin, xmax, ymin, ymax, zNear, zFar);
}

/** Create a projection matrix with the far plane at the infinity. */
void PerspectiveInf( float fov, float aspect, float zNear ) {
    float x = zNear * tan( fov/2 );
    float y = x / aspect;
    FrustumInf( -x, x, -y, y, zNear );
}

/** Create an inverse projection matrix with far plane at the infinity. */
void PerspectiveInfInv( float fov, float aspect, float zNear ) {
    float x = zNear * tan( fov/2 );
    float y = x / aspect;
    FrustumInfInv( -x, x, -y, y, zNear );
}

/** Build bone matrix from quatertion and offset. */
void BoneMatrix(const Quat & q, const Vec3 & offset) {
    float x2, y2, z2, xx, xy, xz, yy, yz, zz, wx, wy, wz;

    // calculate coefficients
    x2 = q.x + q.x;
    y2 = q.y + q.y;
    z2 = q.z + q.z;

    xx = q.x * x2;   xy = q.x * y2;   xz = q.x * z2;
    yy = q.y * y2;   yz = q.y * z2;   zz = q.z * z2;
    wx = q.w * x2;   wy = q.w * y2;   wz = q.w * z2;

    data[0] = 1.0f - (yy + zz); 	
    data[1] = xy - wz;
    data[2] = xz + wy;		
    data[3] = 0.0f;

    data[4] = xy + wz;		
    data[5] = 1.0f - (xx + zz);
    data[6] = yz - wx;		
    data[7] = 0.0f;

    data[8] = xz - wy;		
    data[9] = yz + wx;
    data[10] = 1.0f - (xx + yy);		
    data[11] = 0.0f;

    data[12] = offset.x;
    data[13] = offset.y;
    data[14] = offset.z;			
    data[15] = 1.0f;
}

//@}


/** @name Transformations: */
//@{

/** Apply a general scale. */
void Scale( float x, float y, float z ) {
    data[0] *= x;	data[4] *= y;	data[8]  *= z;
    data[1] *= x;	data[5] *= y;	data[9]  *= z;
    data[2] *= x;	data[6] *= y;	data[10] *= z;
    data[3] *= x;	data[7] *= y;	data[11] *= z;
}

/** Apply a rotation of theta degrees around the axis v*/
void Rotate( float theta, const Vec3 & v ) {
    Matrix b;
    b.RotationMatrix( theta, v[0], v[1], v[2] );
    Multiply4x3( b );
}

/** Apply a rotation of theta degrees around the axis v*/
void Rotate( float theta, float v0, float v1, float v2 ) {
    Matrix b;
    b.RotationMatrix( theta, v0, v1, v2 );
    Multiply4x3( b );
}

/**
* Translate the matrix by t. This is the same as multiplying by a
* translation matrix with the given offset.
* this = T * this
*/
void Translate( const Vec3 &t ) {
    data[12] = data[0] * t.x + data[4] * t.y + data[8]  * t.z + data[12];
    data[13] = data[1] * t.x + data[5] * t.y + data[9]  * t.z + data[13];
    data[14] = data[2] * t.x + data[6] * t.y + data[10] * t.z + data[14];
    data[15] = data[3] * t.x + data[7] * t.y + data[11] * t.z + data[15];
}

/** 
* Translate the matrix by x, y, z. This is the same as multiplying by a 
* translation matrix with the given offsets.
*/
void Translate( float x, float y, float z ) {
    data[12] = data[0] * x + data[4] * y + data[8]  * z + data[12];
    data[13] = data[1] * x + data[5] * y + data[9]  * z + data[13];
    data[14] = data[2] * x + data[6] * y + data[10] * z + data[14];
    data[15] = data[3] * x + data[7] * y + data[11] * z + data[15];
}

/** Compute the transposed matrix. */
void Transpose() {
    piSwap(data[1], data[4]);
    piSwap(data[2], data[8]);
    piSwap(data[6], data[9]);
    piSwap(data[3], data[12]);
    piSwap(data[7], data[13]);
    piSwap(data[11], data[14]);
}

/** Compute the inverse of a rigid-body/isometry/orthonormal matrix. */
void IsometryInverse() {
    // transposed 3x3 upper left matrix
    piSwap(data[1], data[4]);
    piSwap(data[2], data[8]);
    piSwap(data[6], data[9]);

    // translate by the negative offsets
    Vec3 v(-data[12], -data[13], -data[14]);
    data[12] = data[13] = data[14] = 0;
    Translate(v);
}

/** Compute the inverse of the affine portion of this matrix. */
void AffineInverse() {
    data[12] = data[13] = data[14] = 0;
    Transpose();
}
//@}

/** @name Matrix operations: */
//@{

/** Return the determinant of this matrix. */
float Determinant() const {
    return	data[0] * data[5] * data[10] * data[15] + 
        data[1] * data[6] * data[11] * data[12] +
        data[2] * data[7] * data[ 8] * data[13] +
        data[3] * data[4] * data[ 9] * data[14] -
        data[3] * data[6] * data[ 9] * data[12] -
        data[2] * data[5] * data[ 8] * data[15] -
        data[1] * data[4] * data[11] * data[14] -
        data[0] * data[7] * data[10] * data[12];
}


/** Standard matrix product: this *= B. */
void Multiply4x4( const Matrix & restrict B ) {
    Multiply4x4(*this, B);
}

/** Standard matrix product: this = A * B. this != B*/
void Multiply4x4( const Matrix & A, const Matrix & restrict B ) {
    piDebugCheck(this != &B);

    for(int i = 0; i < 4; i++) {
        const float ai0 = A(i,0), ai1 = A(i,1), ai2 = A(i,2), ai3 = A(i,3);
        GetElem(i,0) = ai0 * B(0,0) + ai1 * B(1,0) + ai2 * B(2,0) + ai3 * B(3,0);
        GetElem(i,1) = ai0 * B(0,1) + ai1 * B(1,1) + ai2 * B(2,1) + ai3 * B(3,1);
        GetElem(i,2) = ai0 * B(0,2) + ai1 * B(1,2) + ai2 * B(2,2) + ai3 * B(3,2);
        GetElem(i,3) = ai0 * B(0,3) + ai1 * B(1,3) + ai2 * B(2,3) + ai3 * B(3,3);
    }

    /* Unrolled but does not allow this == A
    data[0] = A.data[0] * B.data[0] + A.data[4] * B.data[1] + A.data[8] * B.data[2] + A.data[12] * B.data[3];
    data[1] = A.data[1] * B.data[0] + A.data[5] * B.data[1] + A.data[9] * B.data[2] + A.data[13] * B.data[3];
    data[2] = A.data[2] * B.data[0] + A.data[6] * B.data[1] + A.data[10] * B.data[2] + A.data[14] * B.data[3];
    data[3] = A.data[3] * B.data[0] + A.data[7] * B.data[1] + A.data[11] * B.data[2] + A.data[15] * B.data[3];
    data[4] = A.data[0] * B.data[4] + A.data[4] * B.data[5] + A.data[8] * B.data[6] + A.data[12] * B.data[7];
    data[5] = A.data[1] * B.data[4] + A.data[5] * B.data[5] + A.data[9] * B.data[6] + A.data[13] * B.data[7];
    data[6] = A.data[2] * B.data[4] + A.data[6] * B.data[5] + A.data[10] * B.data[6] + A.data[14] * B.data[7];
    data[7] = A.data[3] * B.data[4] + A.data[7] * B.data[5] + A.data[11] * B.data[6] + A.data[15] * B.data[7];
    data[8] = A.data[0] * B.data[8] + A.data[4] * B.data[9] + A.data[8] * B.data[10] + A.data[12] * B.data[11];
    data[9] = A.data[1] * B.data[8] + A.data[5] * B.data[9] + A.data[9] * B.data[10] + A.data[13] * B.data[11];
    data[10]= A.data[2] * B.data[8] + A.data[6] * B.data[9] + A.data[10] * B.data[10] + A.data[14] * B.data[11];
    data[11]= A.data[3] * B.data[8] + A.data[7] * B.data[9] + A.data[11] * B.data[10] + A.data[15] * B.data[11];
    data[12]= A.data[0] * B.data[12] + A.data[4] * B.data[13] + A.data[8] * B.data[14] + A.data[12] * B.data[15];
    data[13]= A.data[1] * B.data[12] + A.data[5] * B.data[13] + A.data[9] * B.data[14] + A.data[13] * B.data[15];
    data[14]= A.data[2] * B.data[12] + A.data[6] * B.data[13] + A.data[10] * B.data[14] + A.data[14] * B.data[15];
    data[15]= A.data[3] * B.data[12] + A.data[7] * B.data[13] + A.data[11] * B.data[14] + A.data[15] * B.data[15];
    */
}

/** Standard matrix product: this *= B. */
void Multiply4x3( const Matrix & restrict B ) {
    Multiply4x3(*this, B);
}

/** Standard product of matrices, where the last row is [0 0 0 1]. */
void Multiply4x3( const Matrix & A, const Matrix & restrict B ) {
    piDebugCheck(this != &B);

    for(int i = 0; i < 3; i++) {
        const float ai0 = A(i,0), ai1 = A(i,1), ai2 = A(i,2), ai3 = A(i,3);
        GetElem(i,0) = ai0 * B(0,0) + ai1 * B(1,0) + ai2 * B(2,0) + ai3 * B(3,0);
        GetElem(i,1) = ai0 * B(0,1) + ai1 * B(1,1) + ai2 * B(2,1) + ai3 * B(3,1);
        GetElem(i,2) = ai0 * B(0,2) + ai1 * B(1,2) + ai2 * B(2,2) + ai3 * B(3,2);
        GetElem(i,3) = ai0 * B(0,3) + ai1 * B(1,3) + ai2 * B(2,3) + ai3 * B(3,3);
    }
    data[3] = 0.0f; data[7] = 0.0f; data[11] = 0.0f; data[15] = 1.0f;

    /* Unrolled but does not allow this == A
    data[0] = a.data[0] * b.data[0] + a.data[4] * b.data[1] + a.data[8] * b.data[2] + a.data[12] * b.data[3];
    data[1] = a.data[1] * b.data[0] + a.data[5] * b.data[1] + a.data[9] * b.data[2] + a.data[13] * b.data[3];
    data[2] = a.data[2] * b.data[0] + a.data[6] * b.data[1] + a.data[10] * b.data[2] + a.data[14] * b.data[3];
    data[3] = 0.0f;
    data[4] = a.data[0] * b.data[4] + a.data[4] * b.data[5] + a.data[8] * b.data[6] + a.data[12] * b.data[7];
    data[5] = a.data[1] * b.data[4] + a.data[5] * b.data[5] + a.data[9] * b.data[6] + a.data[13] * b.data[7];
    data[6] = a.data[2] * b.data[4] + a.data[6] * b.data[5] + a.data[10] * b.data[6] + a.data[14] * b.data[7];
    data[7] = 0.0f;
    data[8] = a.data[0] * b.data[8] + a.data[4] * b.data[9] + a.data[8] * b.data[10] + a.data[12] * b.data[11];
    data[9] = a.data[1] * b.data[8] + a.data[5] * b.data[9] + a.data[9] * b.data[10] + a.data[13] * b.data[11];
    data[10]= a.data[2] * b.data[8] + a.data[6] * b.data[9] + a.data[10] * b.data[10] + a.data[14] * b.data[11];
    data[11]= 0.0f;
    data[12]= a.data[0] * b.data[12] + a.data[4] * b.data[13] + a.data[8] * b.data[14] + a.data[12] * b.data[15];
    data[13]= a.data[1] * b.data[12] + a.data[5] * b.data[13] + a.data[9] * b.data[14] + a.data[13] * b.data[15];
    data[14]= a.data[2] * b.data[12] + a.data[6] * b.data[13] + a.data[10] * b.data[14] + a.data[14] * b.data[15];
    data[15]= 1.0f;
    */
}
//@}


/** @name Vector operations: */
//@{

/** Transform 3d vector (w=0). */
void TransformVec3(const Vec3 & restrict orig, Vec3 * restrict dest) const {
    piDebugCheck(&orig != dest);
    dest->x = orig.x * data[0] + orig.y * data[4] + orig.z * data[8];
    dest->y = orig.x * data[1] + orig.y * data[5] + orig.z * data[9];
    dest->z = orig.x * data[2] + orig.y * data[6] + orig.z * data[10];
}
/** Transform 3d vector by the transpose (w=0). */
void TransformVec3T(const Vec3 & restrict orig, Vec3 * restrict dest) const {
    piDebugCheck(&orig != dest);
    dest->x = orig.x * data[0] + orig.y * data[1] + orig.z * data[2];
    dest->y = orig.x * data[4] + orig.y * data[5] + orig.z * data[6];
    dest->z = orig.x * data[8] + orig.y * data[9] + orig.z * data[10];
}

/** Transform a 3d homogeneous vector, where the fourth coordinate is assumed to be 1. */
void TransformPoint(const Vec3 & restrict orig, Vec3 * restrict dest) const {
    piDebugCheck(&orig != dest);
    dest->x = orig.x * data[0] + orig.y * data[4] + orig.z * data[8] + data[12];
    dest->y = orig.x * data[1] + orig.y * data[5] + orig.z * data[9] + data[13];
    dest->z = orig.x * data[2] + orig.y * data[6] + orig.z * data[10] + data[14];
}

/** Transform a point, normalize it, and return w. */
float TransformPointAndNormalize(const Vec3 & restrict orig, Vec3 * restrict dest) const {
    piDebugCheck(&orig != dest);
    float w;
    dest->x = orig.x * data[0] + orig.y * data[4] + orig.z * data[8] + data[12];
    dest->y = orig.x * data[1] + orig.y * data[5] + orig.z * data[9] + data[13];
    dest->z = orig.x * data[2] + orig.y * data[6] + orig.z * data[10] + data[14];
    w = 1 / (orig.x * data[3] + orig.y * data[7] + orig.z * data[11] + data[15]);
    *dest *= w;
    return w;
}

/** Transform a point and return w. */
float TransformPointReturnW(const Vec3 & restrict orig, Vec3 * restrict dest) const {
    piDebugCheck(&orig != dest);
    dest->x = orig.x * data[0] + orig.y * data[4] + orig.z * data[8] + data[12];
    dest->y = orig.x * data[1] + orig.y * data[5] + orig.z * data[9] + data[13];
    dest->z = orig.x * data[2] + orig.y * data[6] + orig.z * data[10] + data[14];
    return orig.x * data[3] + orig.y * data[7] + orig.z * data[11] + data[15];
}

/** Transform a normalized 3d point by a 4d matrix and return the resulting 4d vector. */
void TransformVec4(const Vec3 & orig, Vec4 * dest) const {
    dest->x = orig.x * data[0] + orig.y * data[4] + orig.z * data[8] + data[12];
    dest->y = orig.x * data[1] + orig.y * data[5] + orig.z * data[9] + data[13];
    dest->z = orig.x * data[2] + orig.y * data[6] + orig.z * data[10] + data[14];
    dest->w = orig.x * data[3] + orig.y * data[7] + orig.z * data[11] + data[15];
}
//@}

/** @name Matrix analysis. */
//@{

/** Get the ZYZ euler angles from the matrix. Assumes the matrix is orthonormal. */
void GetEulerAnglesZYZ(float * s, float * t, float * r) const {
    if( GetElem(2,2) < 1.0f ) {
        if( GetElem(2,2) > -1.0f ) {
            // 	cs*ct*cr-ss*sr 		-ss*ct*cr-cs*sr		st*cr
            //	cs*ct*sr+ss*cr		-ss*ct*sr+cs*cr		st*sr
            //	-cs*st				ss*st				ct
            *s = atan2(GetElem(1,2), -GetElem(0,2));
            *t = acos(GetElem(2,2));
            *r = atan2(GetElem(2,1), GetElem(2,0));		
        }
        else {
            // 	-c(s-r)	 	s(s-r)		0
            //	s(s-r)		c(s-r)		0
            //	0			0			-1
            *s = atan2(GetElem(0, 1), -GetElem(0, 0)); // = s-r
            *t = PI;
            *r = 0;
        }
    }
    else {
        // 	c(s+r)		-s(s+r)		0
        //	s(s+r)		c(s+r)		0
        //	0			0			1
        *s = atan2(GetElem(0, 1), GetElem(0, 0)); // = s+r
        *t = 0;
        *r = 0;
    }
}

//@}

MATHLIB_API friend PiStream & operator<< ( PiStream & s, Matrix & m );

/** Print to debug output. */
void Print() const {
    piDebug( "[ %5.2f %5.2f %5.2f %5.2f ]\n", data[0], data[4], data[8], data[12] );
    piDebug( "[ %5.2f %5.2f %5.2f %5.2f ]\n", data[1], data[5], data[9], data[13] );
    piDebug( "[ %5.2f %5.2f %5.2f %5.2f ]\n", data[2], data[6], data[10], data[14] );
    piDebug( "[ %5.2f %5.2f %5.2f %5.2f ]\n", data[3], data[7], data[11], data[15] );
}


public:

    float data[16];

};
#endif


#endif // NV_MATH_MATRIX_INL

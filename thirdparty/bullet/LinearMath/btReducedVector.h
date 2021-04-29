//
//  btReducedVectors.h
//  BulletLinearMath
//
//  Created by Xuchen Han on 4/4/20.
//
#ifndef btReducedVectors_h
#define btReducedVectors_h
#include "btVector3.h"
#include "btMatrix3x3.h"
#include "btAlignedObjectArray.h"
#include <stdio.h>
#include <vector>
#include <algorithm>
struct TwoInts
{
    int a,b;
};
inline bool operator<(const TwoInts& A, const TwoInts& B)
{
    return A.b < B.b;
}


// A helper vector type used for CG projections
class btReducedVector
{
public:
    btAlignedObjectArray<int> m_indices;
    btAlignedObjectArray<btVector3> m_vecs;
    int m_sz; // all m_indices value < m_sz
public:
	btReducedVector():m_sz(0)
	{
		m_indices.resize(0);
		m_vecs.resize(0);
        m_indices.clear();
        m_vecs.clear();
	}
	
    btReducedVector(int sz): m_sz(sz)
    {
        m_indices.resize(0);
        m_vecs.resize(0);
        m_indices.clear();
        m_vecs.clear();
    }
    
    btReducedVector(int sz, const btAlignedObjectArray<int>& indices, const btAlignedObjectArray<btVector3>& vecs): m_sz(sz), m_indices(indices), m_vecs(vecs)
    {
    }
    
    void simplify()
    {
        btAlignedObjectArray<int> old_indices(m_indices);
        btAlignedObjectArray<btVector3> old_vecs(m_vecs);
        m_indices.resize(0);
        m_vecs.resize(0);
        m_indices.clear();
        m_vecs.clear();
        for (int i = 0; i < old_indices.size(); ++i)
        {
            if (old_vecs[i].length2() > SIMD_EPSILON)
            {
                m_indices.push_back(old_indices[i]);
                m_vecs.push_back(old_vecs[i]);
            }
        }
    }
    
    btReducedVector operator+(const btReducedVector& other)
    {
		btReducedVector ret(m_sz);
		int i=0, j=0;
		while (i < m_indices.size() && j < other.m_indices.size())
		{
			if (m_indices[i] < other.m_indices[j])
			{
				ret.m_indices.push_back(m_indices[i]);
				ret.m_vecs.push_back(m_vecs[i]);
				++i;
			}
			else if (m_indices[i] > other.m_indices[j])
			{
				ret.m_indices.push_back(other.m_indices[j]);
				ret.m_vecs.push_back(other.m_vecs[j]);
				++j;
			}
			else
			{
				ret.m_indices.push_back(other.m_indices[j]);
				ret.m_vecs.push_back(m_vecs[i] + other.m_vecs[j]);
				++i; ++j;
			}
		}
		while (i < m_indices.size())
		{
			ret.m_indices.push_back(m_indices[i]);
			ret.m_vecs.push_back(m_vecs[i]);
			++i;
		}
		while (j < other.m_indices.size())
		{
			ret.m_indices.push_back(other.m_indices[j]);
			ret.m_vecs.push_back(other.m_vecs[j]);
			++j;
		}
        ret.simplify();
        return ret;
    }

    btReducedVector operator-()
    {
        btReducedVector ret(m_sz);
        for (int i = 0; i < m_indices.size(); ++i)
        {
            ret.m_indices.push_back(m_indices[i]);
            ret.m_vecs.push_back(-m_vecs[i]);
        }
        ret.simplify();
        return ret;
    }
    
    btReducedVector operator-(const btReducedVector& other)
    {
		btReducedVector ret(m_sz);
		int i=0, j=0;
		while (i < m_indices.size() && j < other.m_indices.size())
		{
			if (m_indices[i] < other.m_indices[j])
			{
				ret.m_indices.push_back(m_indices[i]);
				ret.m_vecs.push_back(m_vecs[i]);
				++i;
			}
			else if (m_indices[i] > other.m_indices[j])
			{
				ret.m_indices.push_back(other.m_indices[j]);
				ret.m_vecs.push_back(-other.m_vecs[j]);
				++j;
			}
			else
			{
				ret.m_indices.push_back(other.m_indices[j]);
				ret.m_vecs.push_back(m_vecs[i] - other.m_vecs[j]);
				++i; ++j;
			}
		}
		while (i < m_indices.size())
		{
			ret.m_indices.push_back(m_indices[i]);
			ret.m_vecs.push_back(m_vecs[i]);
			++i;
		}
		while (j < other.m_indices.size())
		{
			ret.m_indices.push_back(other.m_indices[j]);
			ret.m_vecs.push_back(-other.m_vecs[j]);
			++j;
		}
        ret.simplify();
		return ret;
    }
    
    bool operator==(const btReducedVector& other) const
    {
        if (m_sz != other.m_sz)
            return false;
        if (m_indices.size() != other.m_indices.size())
            return false;
        for (int i = 0; i < m_indices.size(); ++i)
        {
            if (m_indices[i] != other.m_indices[i] || m_vecs[i] != other.m_vecs[i])
            {
                return false;
            }
        }
        return true;
    }
    
    bool operator!=(const btReducedVector& other) const
    {
        return !(*this == other);
    }
	
	btReducedVector& operator=(const btReducedVector& other)
	{
		if (this == &other)
		{
			return *this;
		}
        m_sz = other.m_sz;
		m_indices.copyFromArray(other.m_indices);
		m_vecs.copyFromArray(other.m_vecs);
		return *this;
	}
    
    btScalar dot(const btReducedVector& other) const
    {
        btScalar ret = 0;
        int j = 0;
        for (int i = 0; i < m_indices.size(); ++i)
        {
            while (j < other.m_indices.size() && other.m_indices[j] < m_indices[i])
            {
                ++j;
            }
            if (j < other.m_indices.size() && other.m_indices[j] == m_indices[i])
            {
                ret += m_vecs[i].dot(other.m_vecs[j]);
//                ++j;
            }
        }
        return ret;
    }
    
    btScalar dot(const btAlignedObjectArray<btVector3>& other) const
    {
        btScalar ret = 0;
        for (int i = 0; i < m_indices.size(); ++i)
        {
            ret += m_vecs[i].dot(other[m_indices[i]]);
        }
        return ret;
    }
    
    btScalar length2() const
    {
        return this->dot(*this);
    }
	
	void normalize();
    
    // returns the projection of this onto other
    btReducedVector proj(const btReducedVector& other) const;
    
    bool testAdd() const;
    
    bool testMinus() const;
    
    bool testDot() const;
    
    bool testMultiply() const;
    
    void test() const;
    
    void print() const
    {
        for (int i = 0; i < m_indices.size(); ++i)
        {
            printf("%d: (%f, %f, %f)/", m_indices[i], m_vecs[i][0],m_vecs[i][1],m_vecs[i][2]);
        }
        printf("\n");
    }
    
    
    void sort()
    {
        std::vector<TwoInts> tuples;
        for (int i = 0; i < m_indices.size(); ++i)
        {
            TwoInts ti;
            ti.a = i;
            ti.b = m_indices[i];
            tuples.push_back(ti);
        }
        std::sort(tuples.begin(), tuples.end());
        btAlignedObjectArray<int> new_indices;
        btAlignedObjectArray<btVector3> new_vecs;
        for (size_t i = 0; i < tuples.size(); ++i)
        {
            new_indices.push_back(tuples[i].b);
            new_vecs.push_back(m_vecs[tuples[i].a]);
        }
        m_indices = new_indices;
        m_vecs = new_vecs;
    }
};

SIMD_FORCE_INLINE btReducedVector operator*(const btReducedVector& v, btScalar s)
{
    btReducedVector ret(v.m_sz);
    for (int i = 0; i < v.m_indices.size(); ++i)
    {
        ret.m_indices.push_back(v.m_indices[i]);
        ret.m_vecs.push_back(s*v.m_vecs[i]);
    }
    ret.simplify();
    return ret;
}

SIMD_FORCE_INLINE btReducedVector operator*(btScalar s, const btReducedVector& v)
{
    return v*s;
}

SIMD_FORCE_INLINE btReducedVector operator/(const btReducedVector& v, btScalar s)
{
	return v * (1.0/s);
}

SIMD_FORCE_INLINE btReducedVector& operator/=(btReducedVector& v, btScalar s)
{
	v = v/s;
	return v;
}

SIMD_FORCE_INLINE btReducedVector& operator+=(btReducedVector& v1, const btReducedVector& v2)
{
	v1 = v1+v2;
	return v1;
}

SIMD_FORCE_INLINE btReducedVector& operator-=(btReducedVector& v1, const btReducedVector& v2)
{
	v1 = v1-v2;
	return v1;
}

#endif /* btReducedVectors_h */

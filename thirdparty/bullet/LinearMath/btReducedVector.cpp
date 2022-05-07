//
//  btReducedVector.cpp
//  LinearMath
//
//  Created by Xuchen Han on 4/4/20.
//
#include <stdio.h>
#include "btReducedVector.h"
#include <cmath>

// returns the projection of this onto other
btReducedVector btReducedVector::proj(const btReducedVector& other) const
{
    btReducedVector ret(m_sz);
    btScalar other_length2 = other.length2();
    if (other_length2 < SIMD_EPSILON)
    {
        return ret;
    }
    return other*(this->dot(other))/other_length2;
}

void btReducedVector::normalize()
{
    if (this->length2() < SIMD_EPSILON)
    {
        m_indices.clear();
        m_vecs.clear();
        return;
    }
    *this /= std::sqrt(this->length2());
}

bool btReducedVector::testAdd() const
{
    int sz = 5;
    btAlignedObjectArray<int> id1;
    id1.push_back(1);
    id1.push_back(3);
    btAlignedObjectArray<btVector3> v1;
    v1.push_back(btVector3(1,0,1));
    v1.push_back(btVector3(3,1,5));
    btAlignedObjectArray<int> id2;
    id2.push_back(2);
    id2.push_back(3);
    id2.push_back(5);
    btAlignedObjectArray<btVector3> v2;
    v2.push_back(btVector3(2,3,1));
    v2.push_back(btVector3(3,4,9));
    v2.push_back(btVector3(0,4,0));
    btAlignedObjectArray<int> id3;
    id3.push_back(1);
    id3.push_back(2);
    id3.push_back(3);
    id3.push_back(5);
    btAlignedObjectArray<btVector3> v3;
    v3.push_back(btVector3(1,0,1));
    v3.push_back(btVector3(2,3,1));
    v3.push_back(btVector3(6,5,14));
    v3.push_back(btVector3(0,4,0));
    btReducedVector rv1(sz, id1, v1);
    btReducedVector rv2(sz, id2, v2);
    btReducedVector ans(sz, id3, v3);
    bool ret = ((ans == rv1+rv2) && (ans == rv2+rv1));
    if (!ret)
        printf("btReducedVector testAdd failed\n");
    return ret;
}

bool btReducedVector::testMinus() const
{
    int sz = 5;
    btAlignedObjectArray<int> id1;
    id1.push_back(1);
    id1.push_back(3);
    btAlignedObjectArray<btVector3> v1;
    v1.push_back(btVector3(1,0,1));
    v1.push_back(btVector3(3,1,5));
    btAlignedObjectArray<int> id2;
    id2.push_back(2);
    id2.push_back(3);
    id2.push_back(5);
    btAlignedObjectArray<btVector3> v2;
    v2.push_back(btVector3(2,3,1));
    v2.push_back(btVector3(3,4,9));
    v2.push_back(btVector3(0,4,0));
    btAlignedObjectArray<int> id3;
    id3.push_back(1);
    id3.push_back(2);
    id3.push_back(3);
    id3.push_back(5);
    btAlignedObjectArray<btVector3> v3;
    v3.push_back(btVector3(-1,-0,-1));
    v3.push_back(btVector3(2,3,1));
    v3.push_back(btVector3(0,3,4));
    v3.push_back(btVector3(0,4,0));
    btReducedVector rv1(sz, id1, v1);
    btReducedVector rv2(sz, id2, v2);
    btReducedVector ans(sz, id3, v3);
    bool ret = (ans == rv2-rv1);
    if (!ret)
        printf("btReducedVector testMinus failed\n");
    return ret;
}

bool btReducedVector::testDot() const
{
    int sz = 5;
    btAlignedObjectArray<int> id1;
    id1.push_back(1);
    id1.push_back(3);
    btAlignedObjectArray<btVector3> v1;
    v1.push_back(btVector3(1,0,1));
    v1.push_back(btVector3(3,1,5));
    btAlignedObjectArray<int> id2;
    id2.push_back(2);
    id2.push_back(3);
    id2.push_back(5);
    btAlignedObjectArray<btVector3> v2;
    v2.push_back(btVector3(2,3,1));
    v2.push_back(btVector3(3,4,9));
    v2.push_back(btVector3(0,4,0));
    btReducedVector rv1(sz, id1, v1);
    btReducedVector rv2(sz, id2, v2);
    btScalar ans = 58;
    bool ret = (ans == rv2.dot(rv1) && ans == rv1.dot(rv2));
    ans = 14+16+9+16+81;
    ret &= (ans==rv2.dot(rv2));
    
    if (!ret)
        printf("btReducedVector testDot failed\n");
    return ret;
}

bool btReducedVector::testMultiply() const
{
    int sz = 5;
    btAlignedObjectArray<int> id1;
    id1.push_back(1);
    id1.push_back(3);
    btAlignedObjectArray<btVector3> v1;
    v1.push_back(btVector3(1,0,1));
    v1.push_back(btVector3(3,1,5));
    btScalar s = 2;
    btReducedVector rv1(sz, id1, v1);
    btAlignedObjectArray<int> id2;
    id2.push_back(1);
    id2.push_back(3);
    btAlignedObjectArray<btVector3> v2;
    v2.push_back(btVector3(2,0,2));
    v2.push_back(btVector3(6,2,10));
    btReducedVector ans(sz, id2, v2);
    bool ret = (ans == rv1*s);
    if (!ret)
        printf("btReducedVector testMultiply failed\n");
    return ret;
}

void btReducedVector::test() const
{
    bool ans = testAdd() && testMinus() && testDot() && testMultiply();
    if (ans)
    {
        printf("All tests passed\n");
    }
    else
    {
        printf("Tests failed\n");
    }
}

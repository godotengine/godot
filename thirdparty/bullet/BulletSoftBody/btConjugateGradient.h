/*
 Written by Xuchen Han <xuchenhan2015@u.northwestern.edu>
 
 Bullet Continuous Collision Detection and Physics Library
 Copyright (c) 2019 Google Inc. http://bulletphysics.org
 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it freely,
 subject to the following restrictions:
 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef BT_CONJUGATE_GRADIENT_H
#define BT_CONJUGATE_GRADIENT_H
#include <iostream>
#include <cmath>
#include <limits>
#include <LinearMath/btAlignedObjectArray.h>
#include <LinearMath/btVector3.h>
#include "LinearMath/btQuickprof.h"
template <class MatrixX>
class btConjugateGradient
{
    typedef btAlignedObjectArray<btVector3> TVStack;
    TVStack r,p,z,temp;
    int max_iterations;
    btScalar tolerance_squared;
public:
    btConjugateGradient(const int max_it_in)
    : max_iterations(max_it_in)
    {
       tolerance_squared = 1e-5;
    }
    
    virtual ~btConjugateGradient(){}
    
    // return the number of iterations taken
    int solve(MatrixX& A, TVStack& x, const TVStack& b, bool verbose = false)
    {
        BT_PROFILE("CGSolve");
        btAssert(x.size() == b.size());
        reinitialize(b);
        // r = b - A * x --with assigned dof zeroed out
        A.multiply(x, temp);
        r = sub(b, temp);
        A.project(r);
        // z = M^(-1) * r
        A.precondition(r, z);
        A.project(z);
        btScalar r_dot_z = dot(z,r);
        if (r_dot_z <= tolerance_squared) {
            if (verbose)
            {
                std::cout << "Iteration = 0" << std::endl;
                std::cout << "Two norm of the residual = " << r_dot_z << std::endl;
            }
            return 0;
        }
        p = z;
        btScalar r_dot_z_new = r_dot_z;
        for (int k = 1; k <= max_iterations; k++) {
            // temp = A*p
            A.multiply(p, temp);
            A.project(temp);
            if (dot(p,temp) < SIMD_EPSILON)
            {
                if (verbose)
                    std::cout << "Encountered negative direction in CG!" << std::endl;
                if (k == 1)
                {
                    x = b;
                }
              return k;
            }
            // alpha = r^T * z / (p^T * A * p)
            btScalar alpha = r_dot_z_new / dot(p, temp);
            //  x += alpha * p;
            multAndAddTo(alpha, p, x);
            //  r -= alpha * temp;
            multAndAddTo(-alpha, temp, r);
            // z = M^(-1) * r
            A.precondition(r, z);
            r_dot_z = r_dot_z_new;
            r_dot_z_new = dot(r,z);
            if (r_dot_z_new < tolerance_squared) {
                if (verbose)
                {
                    std::cout << "ConjugateGradient iterations " << k << std::endl;
                }
                return k;
            }

            btScalar beta = r_dot_z_new/r_dot_z;
            p = multAndAdd(beta, p, z);
        }
        if (verbose)
        {
            std::cout << "ConjugateGradient max iterations reached " << max_iterations << std::endl;
        }
        return max_iterations;
    }
    
    void reinitialize(const TVStack& b)
    {
        r.resize(b.size());
        p.resize(b.size());
        z.resize(b.size());
        temp.resize(b.size());
    }
    
    TVStack sub(const TVStack& a, const TVStack& b)
    {
        // c = a-b
        btAssert(a.size() == b.size());
        TVStack c;
        c.resize(a.size());
        for (int i = 0; i < a.size(); ++i)
        {
            c[i] = a[i] - b[i];
        }
        return c;
    }
    
    btScalar squaredNorm(const TVStack& a)
    {
        return dot(a,a);
    }
    
    btScalar dot(const TVStack& a, const TVStack& b)
    {
        btScalar ans(0);
        for (int i = 0; i < a.size(); ++i)
            ans += a[i].dot(b[i]);
        return ans;
    }
    
    void multAndAddTo(btScalar s, const TVStack& a, TVStack& result)
    {
//        result += s*a
        btAssert(a.size() == result.size());
        for (int i = 0; i < a.size(); ++i)
            result[i] += s * a[i];
    }
    
    TVStack multAndAdd(btScalar s, const TVStack& a, const TVStack& b)
    {
        // result = a*s + b
        TVStack result;
        result.resize(a.size());
        for (int i = 0; i < a.size(); ++i)
            result[i] = s * a[i] + b[i];
        return result;
    }
};
#endif /* btConjugateGradient_h */

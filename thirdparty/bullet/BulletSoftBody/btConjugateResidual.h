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

#ifndef BT_CONJUGATE_RESIDUAL_H
#define BT_CONJUGATE_RESIDUAL_H
#include <iostream>
#include <cmath>
#include <limits>
#include <LinearMath/btAlignedObjectArray.h>
#include <LinearMath/btVector3.h>
#include <LinearMath/btScalar.h>
#include "LinearMath/btQuickprof.h"
template <class MatrixX>
class btConjugateResidual
{
    typedef btAlignedObjectArray<btVector3> TVStack;
    TVStack r,p,z,temp_p, temp_r, best_x;
    // temp_r = A*r
    // temp_p = A*p
    // z = M^(-1) * temp_p = M^(-1) * A * p
    int max_iterations;
    btScalar tolerance_squared, best_r;
public:
    btConjugateResidual(const int max_it_in)
    : max_iterations(max_it_in)
    {
        tolerance_squared = 1e-2;
    }
    
    virtual ~btConjugateResidual(){}
    
    // return the number of iterations taken
    int solve(MatrixX& A, TVStack& x, const TVStack& b, bool verbose = false)
    {
        BT_PROFILE("CRSolve");
        btAssert(x.size() == b.size());
        reinitialize(b);
        // r = b - A * x --with assigned dof zeroed out
        A.multiply(x, temp_r); // borrow temp_r here to store A*x
        r = sub(b, temp_r);
        // z = M^(-1) * r
        A.precondition(r, z);  // borrow z to store preconditioned r
        r = z;
        btScalar residual_norm = norm(r);
        if (residual_norm <= tolerance_squared) {
            if (verbose)
            {
                std::cout << "Iteration = 0" << std::endl;
                std::cout << "Two norm of the residual = " << residual_norm << std::endl;
            }
            return 0;
        }
        p = r;
        btScalar r_dot_Ar, r_dot_Ar_new;
        // temp_p = A*p
        A.multiply(p, temp_p);
        // temp_r = A*r
        temp_r = temp_p;
        r_dot_Ar = dot(r, temp_r);
        for (int k = 1; k <= max_iterations; k++) {
            // z = M^(-1) * Ap
            A.precondition(temp_p, z);
            // alpha = r^T * A * r / (Ap)^T * M^-1 * Ap)
            btScalar alpha = r_dot_Ar / dot(temp_p, z);
            //  x += alpha * p;
            multAndAddTo(alpha, p, x);
            //  r -= alpha * z;
            multAndAddTo(-alpha, z, r);
            btScalar norm_r = norm(r);
            if (norm_r < best_r)
            {
                best_x = x;
                best_r = norm_r;
                if (norm_r < tolerance_squared) {
                    if (verbose)
                    {
                        std::cout << "ConjugateResidual iterations " << k << std::endl;
                    }
                    return k;
                }
                else
                {
                    if (verbose)
                    {
                        std::cout << "ConjugateResidual iterations " << k << " has residual "<< norm_r << std::endl;
                    }
                }
            }
            // temp_r = A * r;
            A.multiply(r, temp_r);
            r_dot_Ar_new = dot(r, temp_r);
            btScalar beta = r_dot_Ar_new/r_dot_Ar;
            r_dot_Ar = r_dot_Ar_new;
            // p = beta*p + r;
            p = multAndAdd(beta, p, r);
            // temp_p = beta*temp_p + temp_r;
            temp_p = multAndAdd(beta, temp_p, temp_r);
        }
        if (verbose)
        {
            std::cout << "ConjugateResidual max iterations reached " << max_iterations << std::endl;
        }
        x = best_x;
        return max_iterations;
    }
    
    void reinitialize(const TVStack& b)
    {
        r.resize(b.size());
        p.resize(b.size());
        z.resize(b.size());
        temp_p.resize(b.size());
        temp_r.resize(b.size());
        best_x.resize(b.size());
        best_r = SIMD_INFINITY;
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
    
    btScalar norm(const TVStack& a)
    {
        btScalar ret = 0;
        for (int i = 0; i < a.size(); ++i)
        {
            for (int d = 0; d < 3; ++d)
            {
                ret = btMax(ret, btFabs(a[i][d]));
            }
        }
        return ret;
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
#endif /* btConjugateResidual_h */


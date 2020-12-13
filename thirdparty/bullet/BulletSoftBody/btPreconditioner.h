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

#ifndef BT_PRECONDITIONER_H
#define BT_PRECONDITIONER_H

class Preconditioner
{
public:
    typedef btAlignedObjectArray<btVector3> TVStack;
    virtual void operator()(const TVStack& x, TVStack& b) = 0;
    virtual void reinitialize(bool nodeUpdated) = 0;
    virtual ~Preconditioner(){}
};

class DefaultPreconditioner : public Preconditioner
{
public:
    virtual void operator()(const TVStack& x, TVStack& b)
    {
        btAssert(b.size() == x.size());
        for (int i = 0; i < b.size(); ++i)
            b[i] = x[i];
    }
    virtual void reinitialize(bool nodeUpdated)
    {
    }
    
    virtual ~DefaultPreconditioner(){}
};

class MassPreconditioner : public Preconditioner
{
    btAlignedObjectArray<btScalar> m_inv_mass;
    const btAlignedObjectArray<btSoftBody *>& m_softBodies;
public:
    MassPreconditioner(const btAlignedObjectArray<btSoftBody *>& softBodies)
    : m_softBodies(softBodies)
    {
    }
    
    virtual void reinitialize(bool nodeUpdated)
    {
        if (nodeUpdated)
        {
            m_inv_mass.clear();
            for (int i = 0; i < m_softBodies.size(); ++i)
            {
                btSoftBody* psb = m_softBodies[i];
                for (int j = 0; j < psb->m_nodes.size(); ++j)
                    m_inv_mass.push_back(psb->m_nodes[j].m_im);
            }
        }
    }
    
    virtual void operator()(const TVStack& x, TVStack& b)
    {
        btAssert(b.size() == x.size());
        btAssert(m_inv_mass.size() <= x.size());
        for (int i = 0; i < m_inv_mass.size(); ++i)
        {
            b[i] = x[i] * m_inv_mass[i];
        }
        for (int i = m_inv_mass.size(); i < b.size(); ++i)
        {
            b[i] = x[i];
        }
    }
};


class KKTPreconditioner : public Preconditioner
{
    const btAlignedObjectArray<btSoftBody *>& m_softBodies;
    const btDeformableContactProjection& m_projections;
    const btAlignedObjectArray<btDeformableLagrangianForce*>& m_lf;
    TVStack m_inv_A, m_inv_S;
    const btScalar& m_dt;
    const bool& m_implicit;
public:
    KKTPreconditioner(const btAlignedObjectArray<btSoftBody *>& softBodies, const btDeformableContactProjection& projections, const btAlignedObjectArray<btDeformableLagrangianForce*>& lf, const btScalar& dt, const bool& implicit)
    : m_softBodies(softBodies)
    , m_projections(projections)
    , m_lf(lf)
    , m_dt(dt)
    , m_implicit(implicit)
    {
    }
    
    virtual void reinitialize(bool nodeUpdated)
    {
        if (nodeUpdated)
        {
            int num_nodes = 0;
            for (int i = 0; i < m_softBodies.size(); ++i)
            {
                btSoftBody* psb = m_softBodies[i];
                num_nodes += psb->m_nodes.size();
            }
            m_inv_A.resize(num_nodes);
        }
        buildDiagonalA(m_inv_A);
        for (int i = 0; i < m_inv_A.size(); ++i)
        {
//            printf("A[%d] = %f, %f, %f \n", i, m_inv_A[i][0], m_inv_A[i][1], m_inv_A[i][2]);
            for (int d = 0; d < 3; ++d)
            {
                m_inv_A[i][d] = (m_inv_A[i][d] == 0) ? 0.0 : 1.0/ m_inv_A[i][d];
            }
        }
        m_inv_S.resize(m_projections.m_lagrangeMultipliers.size());
//        printf("S.size() = %d \n", m_inv_S.size());
        buildDiagonalS(m_inv_A, m_inv_S);
        for (int i = 0; i < m_inv_S.size(); ++i)
        {
//            printf("S[%d] = %f, %f, %f \n", i, m_inv_S[i][0], m_inv_S[i][1], m_inv_S[i][2]);
            for (int d = 0; d < 3; ++d)
            {
                m_inv_S[i][d] = (m_inv_S[i][d] == 0) ? 0.0 : 1.0/ m_inv_S[i][d];
            }
        }
    }
    
    void buildDiagonalA(TVStack& diagA) const
    {
        size_t counter = 0;
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            btSoftBody* psb = m_softBodies[i];
            for (int j = 0; j < psb->m_nodes.size(); ++j)
            {
                const btSoftBody::Node& node = psb->m_nodes[j];
                diagA[counter] = (node.m_im == 0) ? btVector3(0,0,0) : btVector3(1.0/node.m_im, 1.0 / node.m_im, 1.0 / node.m_im);
                ++counter;
            }
        }
        if (m_implicit)
        {
            printf("implicit not implemented\n");
            btAssert(false);
        }
        for (int i = 0; i < m_lf.size(); ++i)
        {
            // add damping matrix
            m_lf[i]->buildDampingForceDifferentialDiagonal(-m_dt, diagA);
        }
    }
    
    void buildDiagonalS(const TVStack& inv_A, TVStack& diagS)
    {
        for (int c = 0; c < m_projections.m_lagrangeMultipliers.size(); ++c)
        {
            // S[k,k] = e_k^T * C A_d^-1 C^T * e_k
            const LagrangeMultiplier& lm = m_projections.m_lagrangeMultipliers[c];
            btVector3& t = diagS[c];
            t.setZero();
            for (int j = 0; j < lm.m_num_constraints; ++j)
            {
                for (int i = 0; i < lm.m_num_nodes; ++i)
                {
                    for (int d = 0; d < 3; ++d)
                    {
                        t[j] += inv_A[lm.m_indices[i]][d] * lm.m_dirs[j][d] * lm.m_dirs[j][d] * lm.m_weights[i] * lm.m_weights[i];
                    }
                }
            }
        }
    }
#define USE_FULL_PRECONDITIONER
#ifndef USE_FULL_PRECONDITIONER
    virtual void operator()(const TVStack& x, TVStack& b)
    {
        btAssert(b.size() == x.size());
        for (int i = 0; i < m_inv_A.size(); ++i)
        {
            b[i] = x[i] * m_inv_A[i];
        }
        int offset = m_inv_A.size();
        for (int i = 0; i < m_inv_S.size(); ++i)
        {
            b[i+offset] = x[i+offset] * m_inv_S[i];
        }
    }
#else
    virtual void operator()(const TVStack& x, TVStack& b)
    {
        btAssert(b.size() == x.size());
        int offset = m_inv_A.size();

        for (int i = 0; i < m_inv_A.size(); ++i)
        {
            b[i] = x[i] * m_inv_A[i];
        }

        for (int i = 0; i < m_inv_S.size(); ++i)
        {
            b[i+offset].setZero();
        }

        for (int c = 0; c < m_projections.m_lagrangeMultipliers.size(); ++c)
        {
            const LagrangeMultiplier& lm = m_projections.m_lagrangeMultipliers[c];
            // C * x
            for (int d = 0; d < lm.m_num_constraints; ++d)
            {
                for (int i = 0; i < lm.m_num_nodes; ++i)
                {
                    b[offset+c][d] += lm.m_weights[i] * b[lm.m_indices[i]].dot(lm.m_dirs[d]);
                }
            }
        }

        for (int i = 0; i < m_inv_S.size(); ++i)
        {
            b[i+offset] = b[i+offset] * m_inv_S[i];
        }

        for (int i = 0; i < m_inv_A.size(); ++i)
        {
            b[i].setZero();
        }

        for (int c = 0; c < m_projections.m_lagrangeMultipliers.size(); ++c)
        {
            // C^T * lambda
            const LagrangeMultiplier& lm = m_projections.m_lagrangeMultipliers[c];
            for (int i = 0; i < lm.m_num_nodes; ++i)
            {
                for (int j = 0; j < lm.m_num_constraints; ++j)
                {
                    b[lm.m_indices[i]] += b[offset+c][j] * lm.m_weights[i] * lm.m_dirs[j];
                }
            }
        }

        for (int i = 0; i < m_inv_A.size(); ++i)
        {
            b[i] = (x[i] - b[i]) * m_inv_A[i];
        }

        TVStack t;
        t.resize(b.size());
        for (int i = 0; i < m_inv_S.size(); ++i)
        {
            t[i+offset] = x[i+offset] * m_inv_S[i];
        }
        for (int i = 0; i < m_inv_A.size(); ++i)
        {
            t[i].setZero();
        }
        for (int c = 0; c < m_projections.m_lagrangeMultipliers.size(); ++c)
        {
            // C^T * lambda
            const LagrangeMultiplier& lm = m_projections.m_lagrangeMultipliers[c];
            for (int i = 0; i < lm.m_num_nodes; ++i)
            {
                for (int j = 0; j < lm.m_num_constraints; ++j)
                {
                    t[lm.m_indices[i]] += t[offset+c][j] * lm.m_weights[i] * lm.m_dirs[j];
                }
            }
        }
        for (int i = 0; i < m_inv_A.size(); ++i)
        {
            b[i] += t[i] * m_inv_A[i];
        }

        for (int i = 0; i < m_inv_S.size(); ++i)
        {
            b[i+offset] -= x[i+offset] * m_inv_S[i];
        }
    }
#endif
};

#endif /* BT_PRECONDITIONER_H */

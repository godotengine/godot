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
        btAssert(m_inv_mass.size() == x.size());
        for (int i = 0; i < b.size(); ++i)
        {
            b[i] = x[i] * m_inv_mass[i];
        }
    }
};

#endif /* BT_PRECONDITIONER_H */

// This code is in the public domain -- castanyo@yahoo.es

#include "Plane.h"
#include "Plane.inl"
#include "Matrix.inl"

namespace nv
{
    Plane transformPlane(const Matrix & m, const Plane & p)
    {
        Vector3 newVec = transformVector(m, p.vector());

        Vector3 ptInPlane = p.offset() * p.vector();
        ptInPlane = transformPoint(m, ptInPlane);

        return Plane(newVec, ptInPlane);
    }

    Vector3 planeIntersection(const Plane & a, const Plane & b, const Plane & c)
    {
        return dot(a.vector(), cross(b.vector(), c.vector())) * (
            a.offset() * cross(b.vector(), c.vector()) + 
            c.offset() * cross(a.vector(), b.vector()) +
            b.offset() * cross(c.vector(), a.vector()));
    }

} // nv namespace

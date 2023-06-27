/* Copyright (c) 2011 Khaled Mamou (kmamou at gmail dot com)
 All rights reserved.
 
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 3. The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#ifndef VHACD_ICHULL_H
#define VHACD_ICHULL_H
#include "vhacdManifoldMesh.h"
#include "vhacdVector.h"

// -- GODOT start --
#include <cstdint>
// -- GODOT end --

namespace VHACD {
//!    Incremental Convex Hull algorithm (cf. http://cs.smith.edu/~orourke/books/ftp.html ).
enum ICHullError {
    ICHullErrorOK = 0,
    ICHullErrorCoplanarPoints,
    ICHullErrorNoVolume,
    ICHullErrorInconsistent,
    ICHullErrorNotEnoughPoints
};
class ICHull {
public:
    static const double sc_eps;
    //!
    bool IsFlat() { return m_isFlat; }
    //! Returns the computed mesh
    TMMesh& GetMesh() { return m_mesh; }
    //!    Add one point to the convex-hull
    bool AddPoint(const Vec3<double>& point) { return AddPoints(&point, 1); }
    //!    Add one point to the convex-hull
    bool AddPoint(const Vec3<double>& point, int32_t id);
    //!    Add points to the convex-hull
    bool AddPoints(const Vec3<double>* points, size_t nPoints);
    //!
    ICHullError Process();
    //!
    ICHullError Process(const uint32_t nPointsCH, const double minVolume = 0.0);
    //!
    bool IsInside(const Vec3<double>& pt0, const double eps = 0.0);
    //!
    const ICHull& operator=(ICHull& rhs);

    //!    Constructor
    ICHull();
    //! Destructor
    ~ICHull(void){};

private:
    //!    DoubleTriangle builds the initial double triangle.  It first finds 3 noncollinear points and makes two faces out of them, in opposite order. It then finds a fourth point that is not coplanar with that face.  The vertices are stored in the face structure in counterclockwise order so that the volume between the face and the point is negative. Lastly, the 3 newfaces to the fourth point are constructed and the data structures are cleaned up.
    ICHullError DoubleTriangle();
    //!    MakeFace creates a new face structure from three vertices (in ccw order).  It returns a pointer to the face.
    CircularListElement<TMMTriangle>* MakeFace(CircularListElement<TMMVertex>* v0,
        CircularListElement<TMMVertex>* v1,
        CircularListElement<TMMVertex>* v2,
        CircularListElement<TMMTriangle>* fold);
    //!
    CircularListElement<TMMTriangle>* MakeConeFace(CircularListElement<TMMEdge>* e, CircularListElement<TMMVertex>* v);
    //!
    bool ProcessPoint();
    //!
    bool ComputePointVolume(double& totalVolume, bool markVisibleFaces);
    //!
    bool FindMaxVolumePoint(const double minVolume = 0.0);
    //!
    bool CleanEdges();
    //!
    bool CleanVertices(uint32_t& addedPoints);
    //!
    bool CleanTriangles();
    //!
    bool CleanUp(uint32_t& addedPoints);
    //!
    bool MakeCCW(CircularListElement<TMMTriangle>* f,
        CircularListElement<TMMEdge>* e,
        CircularListElement<TMMVertex>* v);
    void Clear();

private:
    static const int32_t sc_dummyIndex;
    TMMesh m_mesh;
    SArray<CircularListElement<TMMEdge>*> m_edgesToDelete;
    SArray<CircularListElement<TMMEdge>*> m_edgesToUpdate;
    SArray<CircularListElement<TMMTriangle>*> m_trianglesToDelete;
    Vec3<double> m_normal;
    bool m_isFlat;
    ICHull(const ICHull& rhs);
};
}
#endif // VHACD_ICHULL_H

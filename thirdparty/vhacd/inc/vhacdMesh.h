/* Copyright (c) 2011 Khaled Mamou (kmamou at gmail dot com)
 All rights reserved.
 
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 3. The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#ifndef VHACD_MESH_H
#define VHACD_MESH_H
#include "vhacdSArray.h"
#include "vhacdVector.h"

#define VHACD_DEBUG_MESH

namespace VHACD {
enum AXIS {
    AXIS_X = 0,
    AXIS_Y = 1,
    AXIS_Z = 2
};
struct Plane {
    double m_a;
    double m_b;
    double m_c;
    double m_d;
    AXIS m_axis;
    short m_index;
};
#ifdef VHACD_DEBUG_MESH
struct Material {

    Vec3<double> m_diffuseColor;
    double m_ambientIntensity;
    Vec3<double> m_specularColor;
    Vec3<double> m_emissiveColor;
    double m_shininess;
    double m_transparency;
    Material(void)
    {
        m_diffuseColor.X() = 0.5;
        m_diffuseColor.Y() = 0.5;
        m_diffuseColor.Z() = 0.5;
        m_specularColor.X() = 0.5;
        m_specularColor.Y() = 0.5;
        m_specularColor.Z() = 0.5;
        m_ambientIntensity = 0.4;
        m_emissiveColor.X() = 0.0;
        m_emissiveColor.Y() = 0.0;
        m_emissiveColor.Z() = 0.0;
        m_shininess = 0.4;
        m_transparency = 0.0;
    };
};
#endif // VHACD_DEBUG_MESH

//! Triangular mesh data structure
class Mesh {
public:
    void AddPoint(const Vec3<double>& pt) { m_points.PushBack(pt); };
    void SetPoint(size_t index, const Vec3<double>& pt) { m_points[index] = pt; };
    const Vec3<double>& GetPoint(size_t index) const { return m_points[index]; };
    Vec3<double>& GetPoint(size_t index) { return m_points[index]; };
    size_t GetNPoints() const { return m_points.Size(); };
    double* GetPoints() { return (double*)m_points.Data(); } // ugly
    const double* const GetPoints() const { return (double*)m_points.Data(); } // ugly
    const Vec3<double>* const GetPointsBuffer() const { return m_points.Data(); } //
    Vec3<double>* const GetPointsBuffer() { return m_points.Data(); } //
    void AddTriangle(const Vec3<int32_t>& tri) { m_triangles.PushBack(tri); };
    void SetTriangle(size_t index, const Vec3<int32_t>& tri) { m_triangles[index] = tri; };
    const Vec3<int32_t>& GetTriangle(size_t index) const { return m_triangles[index]; };
    Vec3<int32_t>& GetTriangle(size_t index) { return m_triangles[index]; };
    size_t GetNTriangles() const { return m_triangles.Size(); };
    int32_t* GetTriangles() { return (int32_t*)m_triangles.Data(); } // ugly
    const int32_t* const GetTriangles() const { return (int32_t*)m_triangles.Data(); } // ugly
    const Vec3<int32_t>* const GetTrianglesBuffer() const { return m_triangles.Data(); }
    Vec3<int32_t>* const GetTrianglesBuffer() { return m_triangles.Data(); }
    const Vec3<double>& GetCenter() const { return m_center; }
    const Vec3<double>& GetMinBB() const { return m_minBB; }
    const Vec3<double>& GetMaxBB() const { return m_maxBB; }
    void ClearPoints() { m_points.Clear(); }
    void ClearTriangles() { m_triangles.Clear(); }
    void Clear()
    {
        ClearPoints();
        ClearTriangles();
    }
    void ResizePoints(size_t nPts) { m_points.Resize(nPts); }
    void ResizeTriangles(size_t nTri) { m_triangles.Resize(nTri); }
    void CopyPoints(SArray<Vec3<double> >& points) const { points = m_points; }
    double GetDiagBB() const { return m_diag; }
    double ComputeVolume() const;
    void ComputeConvexHull(const double* const pts,
        const size_t nPts);
    void Clip(const Plane& plane,
        SArray<Vec3<double> >& positivePart,
        SArray<Vec3<double> >& negativePart) const;
    bool IsInside(const Vec3<double>& pt) const;
    double ComputeDiagBB();
	Vec3<double> &ComputeCenter(void);

#ifdef VHACD_DEBUG_MESH
    bool LoadOFF(const std::string& fileName, bool invert);
    bool SaveVRML2(const std::string& fileName) const;
    bool SaveVRML2(std::ofstream& fout, const Material& material) const;
    bool SaveOFF(const std::string& fileName) const;
#endif // VHACD_DEBUG_MESH

    //! Constructor.
    Mesh();
    //! Destructor.
    ~Mesh(void);

private:
    SArray<Vec3<double> > m_points;
    SArray<Vec3<int32_t> > m_triangles;
    Vec3<double> m_minBB;
    Vec3<double> m_maxBB;
    Vec3<double> m_center;
    double m_diag;
};
}
#endif
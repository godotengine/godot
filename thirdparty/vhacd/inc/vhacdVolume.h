/* Copyright (c) 2011 Khaled Mamou (kmamou at gmail dot com)
 All rights reserved.
 
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 3. The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#ifndef VHACD_VOLUME_H
#define VHACD_VOLUME_H
#include "vhacdMesh.h"
#include "vhacdVector.h"
#include <assert.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4456 4701)
#endif

namespace VHACD {

enum VOXEL_VALUE {
    PRIMITIVE_UNDEFINED = 0,
    PRIMITIVE_OUTSIDE_SURFACE = 1,
    PRIMITIVE_INSIDE_SURFACE = 2,
    PRIMITIVE_ON_SURFACE = 3
};

struct Voxel {
public:
    short m_coord[3];
    short m_data;
};

class PrimitiveSet {
public:
    virtual ~PrimitiveSet(){};
    virtual PrimitiveSet* Create() const = 0;
    virtual const size_t GetNPrimitives() const = 0;
    virtual const size_t GetNPrimitivesOnSurf() const = 0;
    virtual const size_t GetNPrimitivesInsideSurf() const = 0;
    virtual const double GetEigenValue(AXIS axis) const = 0;
    virtual const double ComputeMaxVolumeError() const = 0;
    virtual const double ComputeVolume() const = 0;
    virtual void Clip(const Plane& plane, PrimitiveSet* const positivePart,
        PrimitiveSet* const negativePart) const = 0;
    virtual void Intersect(const Plane& plane, SArray<Vec3<double> >* const positivePts,
        SArray<Vec3<double> >* const negativePts, const size_t sampling) const = 0;
    virtual void ComputeExteriorPoints(const Plane& plane, const Mesh& mesh,
        SArray<Vec3<double> >* const exteriorPts) const = 0;
    virtual void ComputeClippedVolumes(const Plane& plane, double& positiveVolume,
        double& negativeVolume) const = 0;
    virtual void SelectOnSurface(PrimitiveSet* const onSurfP) const = 0;
    virtual void ComputeConvexHull(Mesh& meshCH, const size_t sampling = 1) const = 0;
    virtual void ComputeBB() = 0;
    virtual void ComputePrincipalAxes() = 0;
    virtual void AlignToPrincipalAxes() = 0;
    virtual void RevertAlignToPrincipalAxes() = 0;
    virtual void Convert(Mesh& mesh, const VOXEL_VALUE value) const = 0;
    const Mesh& GetConvexHull() const { return m_convexHull; };
    Mesh& GetConvexHull() { return m_convexHull; };
private:
    Mesh m_convexHull;
};

//!
class VoxelSet : public PrimitiveSet {
    friend class Volume;

public:
    //! Destructor.
    ~VoxelSet(void);
    //! Constructor.
    VoxelSet();

    const size_t GetNPrimitives() const { return m_voxels.Size(); }
    const size_t GetNPrimitivesOnSurf() const { return m_numVoxelsOnSurface; }
    const size_t GetNPrimitivesInsideSurf() const { return m_numVoxelsInsideSurface; }
    const double GetEigenValue(AXIS axis) const { return m_D[axis][axis]; }
    const double ComputeVolume() const { return m_unitVolume * m_voxels.Size(); }
    const double ComputeMaxVolumeError() const { return m_unitVolume * m_numVoxelsOnSurface; }
    const Vec3<short>& GetMinBBVoxels() const { return m_minBBVoxels; }
    const Vec3<short>& GetMaxBBVoxels() const { return m_maxBBVoxels; }
    const Vec3<double>& GetMinBB() const { return m_minBB; }
    const double& GetScale() const { return m_scale; }
    const double& GetUnitVolume() const { return m_unitVolume; }
    Vec3<double> GetPoint(Vec3<short> voxel) const
    {
        return Vec3<double>(voxel[0] * m_scale + m_minBB[0],
            voxel[1] * m_scale + m_minBB[1],
            voxel[2] * m_scale + m_minBB[2]);
    }
    Vec3<double> GetPoint(const Voxel& voxel) const
    {
        return Vec3<double>(voxel.m_coord[0] * m_scale + m_minBB[0],
            voxel.m_coord[1] * m_scale + m_minBB[1],
            voxel.m_coord[2] * m_scale + m_minBB[2]);
    }
    Vec3<double> GetPoint(Vec3<double> voxel) const
    {
        return Vec3<double>(voxel[0] * m_scale + m_minBB[0],
            voxel[1] * m_scale + m_minBB[1],
            voxel[2] * m_scale + m_minBB[2]);
    }
    void GetPoints(const Voxel& voxel, Vec3<double>* const pts) const;
    void ComputeConvexHull(Mesh& meshCH, const size_t sampling = 1) const;
    void Clip(const Plane& plane, PrimitiveSet* const positivePart, PrimitiveSet* const negativePart) const;
    void Intersect(const Plane& plane, SArray<Vec3<double> >* const positivePts,
        SArray<Vec3<double> >* const negativePts, const size_t sampling) const;
    void ComputeExteriorPoints(const Plane& plane, const Mesh& mesh,
        SArray<Vec3<double> >* const exteriorPts) const;
    void ComputeClippedVolumes(const Plane& plane, double& positiveVolume, double& negativeVolume) const;
    void SelectOnSurface(PrimitiveSet* const onSurfP) const;
    void ComputeBB();
    void Convert(Mesh& mesh, const VOXEL_VALUE value) const;
    void ComputePrincipalAxes();
    PrimitiveSet* Create() const
    {
        return new VoxelSet();
    }
    void AlignToPrincipalAxes(){};
    void RevertAlignToPrincipalAxes(){};
    Voxel* const GetVoxels() { return m_voxels.Data(); }
    const Voxel* const GetVoxels() const { return m_voxels.Data(); }

private:
    size_t m_numVoxelsOnSurface;
    size_t m_numVoxelsInsideSurface;
    Vec3<double> m_minBB;
    double m_scale;
    SArray<Voxel, 8> m_voxels;
    double m_unitVolume;
    Vec3<double> m_minBBPts;
    Vec3<double> m_maxBBPts;
    Vec3<short> m_minBBVoxels;
    Vec3<short> m_maxBBVoxels;
    Vec3<short> m_barycenter;
    double m_Q[3][3];
    double m_D[3][3];
    Vec3<double> m_barycenterPCA;
};

struct Tetrahedron {
public:
    Vec3<double> m_pts[4];
    unsigned char m_data;
};

//!
class TetrahedronSet : public PrimitiveSet {
    friend class Volume;

public:
    //! Destructor.
    ~TetrahedronSet(void);
    //! Constructor.
    TetrahedronSet();

    const size_t GetNPrimitives() const { return m_tetrahedra.Size(); }
    const size_t GetNPrimitivesOnSurf() const { return m_numTetrahedraOnSurface; }
    const size_t GetNPrimitivesInsideSurf() const { return m_numTetrahedraInsideSurface; }
    const Vec3<double>& GetMinBB() const { return m_minBB; }
    const Vec3<double>& GetMaxBB() const { return m_maxBB; }
    const Vec3<double>& GetBarycenter() const { return m_barycenter; }
    const double GetEigenValue(AXIS axis) const { return m_D[axis][axis]; }
    const double GetSacle() const { return m_scale; }
    const double ComputeVolume() const;
    const double ComputeMaxVolumeError() const;
    void ComputeConvexHull(Mesh& meshCH, const size_t sampling = 1) const;
    void ComputePrincipalAxes();
    void AlignToPrincipalAxes();
    void RevertAlignToPrincipalAxes();
    void Clip(const Plane& plane, PrimitiveSet* const positivePart, PrimitiveSet* const negativePart) const;
    void Intersect(const Plane& plane, SArray<Vec3<double> >* const positivePts,
        SArray<Vec3<double> >* const negativePts, const size_t sampling) const;
    void ComputeExteriorPoints(const Plane& plane, const Mesh& mesh,
        SArray<Vec3<double> >* const exteriorPts) const;
    void ComputeClippedVolumes(const Plane& plane, double& positiveVolume, double& negativeVolume) const;
    void SelectOnSurface(PrimitiveSet* const onSurfP) const;
    void ComputeBB();
    void Convert(Mesh& mesh, const VOXEL_VALUE value) const;
    inline bool Add(Tetrahedron& tetrahedron);
    PrimitiveSet* Create() const
    {
        return new TetrahedronSet();
    }
    static const double EPS;

private:
    void AddClippedTetrahedra(const Vec3<double> (&pts)[10], const int32_t nPts);

    size_t m_numTetrahedraOnSurface;
    size_t m_numTetrahedraInsideSurface;
    double m_scale;
    Vec3<double> m_minBB;
    Vec3<double> m_maxBB;
    Vec3<double> m_barycenter;
    SArray<Tetrahedron, 8> m_tetrahedra;
    double m_Q[3][3];
    double m_D[3][3];
};

//!
class Volume {
public:
    //! Destructor.
    ~Volume(void);

    //! Constructor.
    Volume();

    //! Voxelize
    template <class T>
    void Voxelize(const T* const points, const uint32_t stridePoints, const uint32_t nPoints,
        const int32_t* const triangles, const uint32_t strideTriangles, const uint32_t nTriangles,
        const size_t dim, const Vec3<double>& barycenter, const double (&rot)[3][3]);
    unsigned char& GetVoxel(const size_t i, const size_t j, const size_t k)
    {
        assert(i < m_dim[0] || i >= 0);
        assert(j < m_dim[0] || j >= 0);
        assert(k < m_dim[0] || k >= 0);
        return m_data[i + j * m_dim[0] + k * m_dim[0] * m_dim[1]];
    }
    const unsigned char& GetVoxel(const size_t i, const size_t j, const size_t k) const
    {
        assert(i < m_dim[0] || i >= 0);
        assert(j < m_dim[0] || j >= 0);
        assert(k < m_dim[0] || k >= 0);
        return m_data[i + j * m_dim[0] + k * m_dim[0] * m_dim[1]];
    }
    const size_t GetNPrimitivesOnSurf() const { return m_numVoxelsOnSurface; }
    const size_t GetNPrimitivesInsideSurf() const { return m_numVoxelsInsideSurface; }
    void Convert(Mesh& mesh, const VOXEL_VALUE value) const;
    void Convert(VoxelSet& vset) const;
    void Convert(TetrahedronSet& tset) const;
    void AlignToPrincipalAxes(double (&rot)[3][3]) const;

private:
    void FillOutsideSurface(const size_t i0, const size_t j0, const size_t k0, const size_t i1,
        const size_t j1, const size_t k1);
    void FillInsideSurface();
    template <class T>
    void ComputeBB(const T* const points, const uint32_t stridePoints, const uint32_t nPoints,
        const Vec3<double>& barycenter, const double (&rot)[3][3]);
    void Allocate();
    void Free();

    Vec3<double> m_minBB;
    Vec3<double> m_maxBB;
    double m_scale;
    size_t m_dim[3]; //>! dim
    size_t m_numVoxelsOnSurface;
    size_t m_numVoxelsInsideSurface;
    size_t m_numVoxelsOutsideSurface;
    unsigned char* m_data;
};
int32_t TriBoxOverlap(const Vec3<double>& boxcenter, const Vec3<double>& boxhalfsize, const Vec3<double>& triver0,
    const Vec3<double>& triver1, const Vec3<double>& triver2);
template <class T>
inline void ComputeAlignedPoint(const T* const points, const uint32_t idx, const Vec3<double>& barycenter,
    const double (&rot)[3][3], Vec3<double>& pt){};
template <>
inline void ComputeAlignedPoint<float>(const float* const points, const uint32_t idx, const Vec3<double>& barycenter, const double (&rot)[3][3], Vec3<double>& pt)
{
    double x = points[idx + 0] - barycenter[0];
    double y = points[idx + 1] - barycenter[1];
    double z = points[idx + 2] - barycenter[2];
    pt[0] = rot[0][0] * x + rot[1][0] * y + rot[2][0] * z;
    pt[1] = rot[0][1] * x + rot[1][1] * y + rot[2][1] * z;
    pt[2] = rot[0][2] * x + rot[1][2] * y + rot[2][2] * z;
}
template <>
inline void ComputeAlignedPoint<double>(const double* const points, const uint32_t idx, const Vec3<double>& barycenter, const double (&rot)[3][3], Vec3<double>& pt)
{
    double x = points[idx + 0] - barycenter[0];
    double y = points[idx + 1] - barycenter[1];
    double z = points[idx + 2] - barycenter[2];
    pt[0] = rot[0][0] * x + rot[1][0] * y + rot[2][0] * z;
    pt[1] = rot[0][1] * x + rot[1][1] * y + rot[2][1] * z;
    pt[2] = rot[0][2] * x + rot[1][2] * y + rot[2][2] * z;
}
template <class T>
void Volume::ComputeBB(const T* const points, const uint32_t stridePoints, const uint32_t nPoints,
    const Vec3<double>& barycenter, const double (&rot)[3][3])
{
    Vec3<double> pt;
    ComputeAlignedPoint(points, 0, barycenter, rot, pt);
    m_maxBB = pt;
    m_minBB = pt;
    for (uint32_t v = 1; v < nPoints; ++v) {
        ComputeAlignedPoint(points, v * stridePoints, barycenter, rot, pt);
        for (int32_t i = 0; i < 3; ++i) {
            if (pt[i] < m_minBB[i])
                m_minBB[i] = pt[i];
            else if (pt[i] > m_maxBB[i])
                m_maxBB[i] = pt[i];
        }
    }
}
template <class T>
void Volume::Voxelize(const T* const points, const uint32_t stridePoints, const uint32_t nPoints,
    const int32_t* const triangles, const uint32_t strideTriangles, const uint32_t nTriangles,
    const size_t dim, const Vec3<double>& barycenter, const double (&rot)[3][3])
{
    if (nPoints == 0) {
        return;
    }
    ComputeBB(points, stridePoints, nPoints, barycenter, rot);

    double d[3] = { m_maxBB[0] - m_minBB[0], m_maxBB[1] - m_minBB[1], m_maxBB[2] - m_minBB[2] };
    double r;
    if (d[0] >= d[1] && d[0] >= d[2]) {
        r = d[0];
        m_dim[0] = dim;
        m_dim[1] = 2 + static_cast<size_t>(dim * d[1] / d[0]);
        m_dim[2] = 2 + static_cast<size_t>(dim * d[2] / d[0]);
    }
    else if (d[1] >= d[0] && d[1] >= d[2]) {
        r = d[1];
        m_dim[1] = dim;
        m_dim[0] = 2 + static_cast<size_t>(dim * d[0] / d[1]);
        m_dim[2] = 2 + static_cast<size_t>(dim * d[2] / d[1]);
    }
    else {
        r = d[2];
        m_dim[2] = dim;
        m_dim[0] = 2 + static_cast<size_t>(dim * d[0] / d[2]);
        m_dim[1] = 2 + static_cast<size_t>(dim * d[1] / d[2]);
    }

    m_scale = r / (dim - 1);
    double invScale = (dim - 1) / r;

    Allocate();
    m_numVoxelsOnSurface = 0;
    m_numVoxelsInsideSurface = 0;
    m_numVoxelsOutsideSurface = 0;

    Vec3<double> p[3];
    size_t i, j, k;
    size_t i0, j0, k0;
    size_t i1, j1, k1;
    Vec3<double> boxcenter;
    Vec3<double> pt;
    const Vec3<double> boxhalfsize(0.5, 0.5, 0.5);
    for (size_t t = 0, ti = 0; t < nTriangles; ++t, ti += strideTriangles) {
        Vec3<int32_t> tri(triangles[ti + 0],
            triangles[ti + 1],
            triangles[ti + 2]);
        for (int32_t c = 0; c < 3; ++c) {
            ComputeAlignedPoint(points, tri[c] * stridePoints, barycenter, rot, pt);
            p[c][0] = (pt[0] - m_minBB[0]) * invScale;
            p[c][1] = (pt[1] - m_minBB[1]) * invScale;
            p[c][2] = (pt[2] - m_minBB[2]) * invScale;
            i = static_cast<size_t>(p[c][0] + 0.5);
            j = static_cast<size_t>(p[c][1] + 0.5);
            k = static_cast<size_t>(p[c][2] + 0.5);
            assert(i < m_dim[0] && i >= 0 && j < m_dim[1] && j >= 0 && k < m_dim[2] && k >= 0);

            if (c == 0) {
                i0 = i1 = i;
                j0 = j1 = j;
                k0 = k1 = k;
            }
            else {
                if (i < i0)
                    i0 = i;
                if (j < j0)
                    j0 = j;
                if (k < k0)
                    k0 = k;
                if (i > i1)
                    i1 = i;
                if (j > j1)
                    j1 = j;
                if (k > k1)
                    k1 = k;
            }
        }
        if (i0 > 0)
            --i0;
        if (j0 > 0)
            --j0;
        if (k0 > 0)
            --k0;
        if (i1 < m_dim[0])
            ++i1;
        if (j1 < m_dim[1])
            ++j1;
        if (k1 < m_dim[2])
            ++k1;
        for (size_t i = i0; i < i1; ++i) {
            boxcenter[0] = (double)i;
            for (size_t j = j0; j < j1; ++j) {
                boxcenter[1] = (double)j;
                for (size_t k = k0; k < k1; ++k) {
                    boxcenter[2] = (double)k;
                    int32_t res = TriBoxOverlap(boxcenter, boxhalfsize, p[0], p[1], p[2]);
                    unsigned char& value = GetVoxel(i, j, k);
                    if (res == 1 && value == PRIMITIVE_UNDEFINED) {
                        value = PRIMITIVE_ON_SURFACE;
                        ++m_numVoxelsOnSurface;
                    }
                }
            }
        }
    }
    FillOutsideSurface(0, 0, 0, m_dim[0], m_dim[1], 1);
    FillOutsideSurface(0, 0, m_dim[2] - 1, m_dim[0], m_dim[1], m_dim[2]);
    FillOutsideSurface(0, 0, 0, m_dim[0], 1, m_dim[2]);
    FillOutsideSurface(0, m_dim[1] - 1, 0, m_dim[0], m_dim[1], m_dim[2]);
    FillOutsideSurface(0, 0, 0, 1, m_dim[1], m_dim[2]);
    FillOutsideSurface(m_dim[0] - 1, 0, 0, m_dim[0], m_dim[1], m_dim[2]);
    FillInsideSurface();
}
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif


#endif // VHACD_VOLUME_H

// Copyright NVIDIA Corporation 2008 -- Ignacio Castano <icastano@nvidia.com>

#pragma once
#ifndef NV_MESH_SINGLEFACEMAP_H
#define NV_MESH_SINGLEFACEMAP_H

namespace nv
{
    namespace HalfEdge
    {
        class Mesh;
    }

    void computeSingleFaceMap(HalfEdge::Mesh * mesh);

} // nv namespace

#endif // NV_MESH_SINGLEFACEMAP_H

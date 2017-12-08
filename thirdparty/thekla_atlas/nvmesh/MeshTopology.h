// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MESH_MESHTOPOLOGY_H
#define NV_MESH_MESHTOPOLOGY_H

#include <nvmesh/nvmesh.h>

namespace nv
{
    namespace HalfEdge { class Mesh; }
    class MeshAdjacency;

    /// Mesh topology information.
    class MeshTopology
    {
    public:
        MeshTopology(const HalfEdge::Mesh * mesh) { buildTopologyInfo(mesh); }

        /// Determine if the mesh is connected.
        bool isConnected() const { return m_connectedCount == 1; }

        /// Determine if the mesh is closed. (Each edge is shared by two faces)
        bool isClosed() const { return m_boundaryCount == 0; }

        /// Return true if the mesh has the topology of a disk.
        bool isDisk() const { return isConnected() && m_boundaryCount == 1/* && m_eulerNumber == 1*/; }

        /// Return the number of connected components.
        int connectedCount() const { return m_connectedCount; }

        /// Return the number of open holes.
        int holeCount() const { return m_boundaryCount; }

        /// Return the genus of the mesh.
        int genus() const { return m_genus; }

        /// Return the euler number of the mesh.
        int euler() const { return m_eulerNumber; }


        static bool isQuadOnly(const HalfEdge::Mesh * mesh);


    private:

        NVMESH_API void buildTopologyInfo(const HalfEdge::Mesh * mesh);

    private:

        ///< Number of boundary loops.
        int m_boundaryCount;		

        ///< Number of connected components.
        int m_connectedCount;		

        ///< Euler number.
        int m_eulerNumber;

        /// Mesh genus.
        int m_genus;
    };

} // nv namespace

#endif // NV_MESH_MESHTOPOLOGY_H

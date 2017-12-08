// This code is in the public domain -- castanyo@yahoo.es

#include "nvmesh.h" // pch

#include "nvcore/Array.h"
#include "nvcore/BitArray.h"

#include "nvmesh/MeshTopology.h"
#include "nvmesh/halfedge/Mesh.h"
#include "nvmesh/halfedge/Edge.h"
#include "nvmesh/halfedge/Face.h"

using namespace nv;

void MeshTopology::buildTopologyInfo(const HalfEdge::Mesh * mesh)
{
    const uint vertexCount = mesh->colocalVertexCount();
    const uint faceCount = mesh->faceCount();
    const uint edgeCount = mesh->edgeCount();

    nvDebug( "--- Building mesh topology:\n" );

    Array<uint> stack(faceCount);

    BitArray bitFlags(faceCount);
    bitFlags.clearAll();

    // Compute connectivity.
    nvDebug( "---   Computing connectivity.\n" );

    m_connectedCount = 0;

    for(uint f = 0; f < faceCount; f++ ) {
        if( bitFlags.bitAt(f) == false ) {
            m_connectedCount++;

            stack.pushBack( f );
            while( !stack.isEmpty() ) {

                const uint top = stack.back();
                nvCheck(top != NIL);
                stack.popBack();

                if( bitFlags.bitAt(top) == false ) {
                    bitFlags.setBitAt(top);

                    const HalfEdge::Face * face = mesh->faceAt(top);
                    const HalfEdge::Edge * firstEdge = face->edge;
                    const HalfEdge::Edge * edge = firstEdge;

                    do {
                        const HalfEdge::Face * neighborFace = edge->pair->face;
                        if (neighborFace != NULL) {
                            stack.pushBack(neighborFace->id);
                        }
                        edge = edge->next;
                    } while(edge != firstEdge);
                }
            }
        }
    }
    nvCheck(stack.isEmpty());
    nvDebug( "---   %d connected components.\n", m_connectedCount );


    // Count boundary loops.
    nvDebug( "---   Counting boundary loops.\n" );
    m_boundaryCount = 0;

    bitFlags.resize(edgeCount);
    bitFlags.clearAll();

    // Don't forget to link the boundary otherwise this won't work.
    for (uint e = 0; e < edgeCount; e++)
    {
        const HalfEdge::Edge * startEdge = mesh->edgeAt(e);
        if (startEdge != NULL && startEdge->isBoundary() && bitFlags.bitAt(e) == false)
        {
            nvDebugCheck(startEdge->face != NULL);
            nvDebugCheck(startEdge->pair->face == NULL);

            startEdge = startEdge->pair;

            m_boundaryCount++;

            const HalfEdge::Edge * edge = startEdge;
            do {
                bitFlags.setBitAt(edge->id / 2);
                edge = edge->next;
            } while(startEdge != edge);
        }
    }
    nvDebug("---   %d boundary loops found.\n", m_boundaryCount );


    // Compute euler number.
    m_eulerNumber = vertexCount - edgeCount + faceCount;
    nvDebug("---   Euler number: %d.\n", m_eulerNumber);


    // Compute genus. (only valid on closed connected surfaces)
    m_genus = -1;
    if( isClosed() && isConnected() ) {
        m_genus = (2 - m_eulerNumber) / 2;
        nvDebug("---   Genus: %d.\n", m_genus);
    }
}


/*static*/ bool MeshTopology::isQuadOnly(const HalfEdge::Mesh * mesh)
{
    const uint faceCount = mesh->faceCount();
    for(uint f = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = mesh->faceAt(f);
        if (face->edgeCount() != 4) {
            return false;
        }
    }

    return true;
}

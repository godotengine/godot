/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

#include <assimp/Subdivision.h>
#include <assimp/SceneCombiner.h>
#include <assimp/SpatialSort.h>
#include <assimp/Vertex.h>
#include <assimp/ai_assert.h>

#include "PostProcessing/ProcessHelper.h"

#include <stdio.h>

using namespace Assimp;
void mydummy() {}

// ------------------------------------------------------------------------------------------------
/** Subdivider stub class to implement the Catmull-Clarke subdivision algorithm. The
 *  implementation is basing on recursive refinement. Directly evaluating the result is also
 *  possible and much quicker, but it depends on lengthy matrix lookup tables. */
// ------------------------------------------------------------------------------------------------
class CatmullClarkSubdivider : public Subdivider {
public:
    void Subdivide (aiMesh* mesh, aiMesh*& out, unsigned int num, bool discard_input);
    void Subdivide (aiMesh** smesh, size_t nmesh,
        aiMesh** out, unsigned int num, bool discard_input);

    // ---------------------------------------------------------------------------
    /** Intermediate description of an edge between two corners of a polygon*/
    // ---------------------------------------------------------------------------
    struct Edge
    {
        Edge()
            : ref(0)
        {}
        Vertex edge_point, midpoint;
        unsigned int ref;
    };

    typedef std::vector<unsigned int> UIntVector;
    typedef std::map<uint64_t,Edge> EdgeMap;

    // ---------------------------------------------------------------------------
    // Hashing function to derive an index into an #EdgeMap from two given
    // 'unsigned int' vertex coordinates (!!distinct coordinates - same
    // vertex position == same index!!).
    // NOTE - this leads to rare hash collisions if a) sizeof(unsigned int)>4
    // and (id[0]>2^32-1 or id[0]>2^32-1).
    // MAKE_EDGE_HASH() uses temporaries, so INIT_EDGE_HASH() needs to be put
    // at the head of every function which is about to use MAKE_EDGE_HASH().
    // Reason is that the hash is that hash construction needs to hold the
    // invariant id0<id1 to identify an edge - else two hashes would refer
    // to the same edge.
    // ---------------------------------------------------------------------------
#define MAKE_EDGE_HASH(id0,id1) (eh_tmp0__=id0,eh_tmp1__=id1,\
    (eh_tmp0__<eh_tmp1__?std::swap(eh_tmp0__,eh_tmp1__):mydummy()),(uint64_t)eh_tmp0__^((uint64_t)eh_tmp1__<<32u))


#define INIT_EDGE_HASH_TEMPORARIES()\
    unsigned int eh_tmp0__, eh_tmp1__;

private:
    void InternSubdivide (const aiMesh* const * smesh,
        size_t nmesh,aiMesh** out, unsigned int num);
};


// ------------------------------------------------------------------------------------------------
// Construct a subdivider of a specific type
Subdivider* Subdivider::Create (Algorithm algo)
{
    switch (algo)
    {
    case CATMULL_CLARKE:
        return new CatmullClarkSubdivider();
    };

    ai_assert(false);
    return NULL; // shouldn't happen
}

// ------------------------------------------------------------------------------------------------
// Call the Catmull Clark subdivision algorithm for one mesh
void  CatmullClarkSubdivider::Subdivide (
    aiMesh* mesh,
    aiMesh*& out,
    unsigned int num,
    bool discard_input
    )
{
    ai_assert(mesh != out);
    
    Subdivide(&mesh,1,&out,num,discard_input);
}

// ------------------------------------------------------------------------------------------------
// Call the Catmull Clark subdivision algorithm for multiple meshes
void CatmullClarkSubdivider::Subdivide (
    aiMesh** smesh,
    size_t nmesh,
    aiMesh** out,
    unsigned int num,
    bool discard_input
    )
{
    ai_assert( NULL != smesh );
    ai_assert( NULL != out );

    // course, both regions may not overlap
    ai_assert(smesh<out || smesh+nmesh>out+nmesh);
    if (!num) {
        // No subdivision at all. Need to copy all the meshes .. argh.
        if (discard_input) {
            for (size_t s = 0; s < nmesh; ++s) {
                out[s] = smesh[s];
                smesh[s] = NULL;
            }
        }
        else {
            for (size_t s = 0; s < nmesh; ++s) {
                SceneCombiner::Copy(out+s,smesh[s]);
            }
        }
        return;
    }

    std::vector<aiMesh*> inmeshes;
    std::vector<aiMesh*> outmeshes;
    std::vector<unsigned int> maptbl;

    inmeshes.reserve(nmesh);
    outmeshes.reserve(nmesh);
    maptbl.reserve(nmesh);

    // Remove pure line and point meshes from the working set to reduce the
    // number of edge cases the subdivider is forced to deal with. Line and
    // point meshes are simply passed through.
    for (size_t s = 0; s < nmesh; ++s) {
        aiMesh* i = smesh[s];
        // FIX - mPrimitiveTypes might not yet be initialized
        if (i->mPrimitiveTypes && (i->mPrimitiveTypes & (aiPrimitiveType_LINE|aiPrimitiveType_POINT))==i->mPrimitiveTypes) {
            ASSIMP_LOG_DEBUG("Catmull-Clark Subdivider: Skipping pure line/point mesh");

            if (discard_input) {
                out[s] = i;
                smesh[s] = NULL;
            }
            else {
                SceneCombiner::Copy(out+s,i);
            }
            continue;
        }

        outmeshes.push_back(NULL);inmeshes.push_back(i);
        maptbl.push_back(static_cast<unsigned int>(s));
    }

    // Do the actual subdivision on the preallocated storage. InternSubdivide
    // *always* assumes that enough storage is available, it does not bother
    // checking any ranges.
    ai_assert(inmeshes.size()==outmeshes.size()&&inmeshes.size()==maptbl.size());
    if (inmeshes.empty()) {
        ASSIMP_LOG_WARN("Catmull-Clark Subdivider: Pure point/line scene, I can't do anything");
        return;
    }
    InternSubdivide(&inmeshes.front(),inmeshes.size(),&outmeshes.front(),num);
    for (unsigned int i = 0; i < maptbl.size(); ++i) {
        ai_assert(nullptr != outmeshes[i]);
        out[maptbl[i]] = outmeshes[i];
    }

    if (discard_input) {
        for (size_t s = 0; s < nmesh; ++s) {
            delete smesh[s];
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Note - this is an implementation of the standard (recursive) Cm-Cl algorithm without further
// optimizations (except we're using some nice LUTs). A description of the algorithm can be found
// here: http://en.wikipedia.org/wiki/Catmull-Clark_subdivision_surface
//
// The code is mostly O(n), however parts are O(nlogn) which is therefore the algorithm's
// expected total runtime complexity. The implementation is able to work in-place on the same
// mesh arrays. Calling #InternSubdivide() directly is not encouraged. The code can operate
// in-place unless 'smesh' and 'out' are equal (no strange overlaps or reorderings).
// Previous data is replaced/deleted then.
// ------------------------------------------------------------------------------------------------
void CatmullClarkSubdivider::InternSubdivide (
    const aiMesh* const * smesh,
    size_t nmesh,
    aiMesh** out,
    unsigned int num
    )
{
    ai_assert(NULL != smesh && NULL != out);
    INIT_EDGE_HASH_TEMPORARIES();

    // no subdivision requested or end of recursive refinement
    if (!num) {
        return;
    }

    UIntVector maptbl;
    SpatialSort spatial;

    // ---------------------------------------------------------------------
    // 0. Offset table to index all meshes continuously, generate a spatially
    // sorted representation of all vertices in all meshes.
    // ---------------------------------------------------------------------
    typedef std::pair<unsigned int,unsigned int> IntPair;
    std::vector<IntPair> moffsets(nmesh);
    unsigned int totfaces = 0, totvert = 0;
    for (size_t t = 0; t < nmesh; ++t) {
        const aiMesh* mesh = smesh[t];

        spatial.Append(mesh->mVertices,mesh->mNumVertices,sizeof(aiVector3D),false);
        moffsets[t] = IntPair(totfaces,totvert);

        totfaces += mesh->mNumFaces;
        totvert  += mesh->mNumVertices;
    }

    spatial.Finalize();
    const unsigned int num_unique = spatial.GenerateMappingTable(maptbl,ComputePositionEpsilon(smesh,nmesh));


#define FLATTEN_VERTEX_IDX(mesh_idx, vert_idx) (moffsets[mesh_idx].second+vert_idx)
#define   FLATTEN_FACE_IDX(mesh_idx, face_idx) (moffsets[mesh_idx].first+face_idx)

    // ---------------------------------------------------------------------
    // 1. Compute the centroid point for all faces
    // ---------------------------------------------------------------------
    std::vector<Vertex> centroids(totfaces);
    unsigned int nfacesout = 0;
    for (size_t t = 0, n = 0; t < nmesh; ++t) {
        const aiMesh* mesh = smesh[t];
        for (unsigned int i = 0; i < mesh->mNumFaces;++i,++n)
        {
            const aiFace& face = mesh->mFaces[i];
            Vertex& c = centroids[n];

            for (unsigned int a = 0; a < face.mNumIndices;++a) {
                c += Vertex(mesh,face.mIndices[a]);
            }

            c /= static_cast<float>(face.mNumIndices);
            nfacesout += face.mNumIndices;
        }
    }

    {
    // we want edges to go away before the recursive calls so begin a new scope
    EdgeMap edges;

    // ---------------------------------------------------------------------
    // 2. Set each edge point to be the average of all neighbouring
    // face points and original points. Every edge exists twice
    // if there is a neighboring face.
    // ---------------------------------------------------------------------
    for (size_t t = 0; t < nmesh; ++t) {
        const aiMesh* mesh = smesh[t];

        for (unsigned int i = 0; i < mesh->mNumFaces;++i)   {
            const aiFace& face = mesh->mFaces[i];

            for (unsigned int p =0; p< face.mNumIndices; ++p) {
                const unsigned int id[] = {
                    face.mIndices[p],
                    face.mIndices[p==face.mNumIndices-1?0:p+1]
                };
                const unsigned int mp[] = {
                    maptbl[FLATTEN_VERTEX_IDX(t,id[0])],
                    maptbl[FLATTEN_VERTEX_IDX(t,id[1])]
                };

                Edge& e = edges[MAKE_EDGE_HASH(mp[0],mp[1])];
                e.ref++;
                if (e.ref<=2) {
                    if (e.ref==1) { // original points (end points) - add only once
                        e.edge_point = e.midpoint = Vertex(mesh,id[0])+Vertex(mesh,id[1]);
                        e.midpoint *= 0.5f;
                    }
                    e.edge_point += centroids[FLATTEN_FACE_IDX(t,i)];
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // 3. Normalize edge points
    // ---------------------------------------------------------------------
    {unsigned int bad_cnt = 0;
    for (EdgeMap::iterator it = edges.begin(); it != edges.end(); ++it) {
        if ((*it).second.ref < 2) {
            ai_assert((*it).second.ref);
            ++bad_cnt;
        }
        (*it).second.edge_point *= 1.f/((*it).second.ref+2.f);
    }

    if (bad_cnt) {
        // Report the number of bad edges. bad edges are referenced by less than two
        // faces in the mesh. They occur at outer model boundaries in non-closed
        // shapes.
        ASSIMP_LOG_DEBUG_F("Catmull-Clark Subdivider: got ", bad_cnt, " bad edges touching only one face (totally ", 
            static_cast<unsigned int>(edges.size()), " edges). ");
    }}

    // ---------------------------------------------------------------------
    // 4. Compute a vertex-face adjacency table. We can't reuse the code
    // from VertexTriangleAdjacency because we need the table for multiple
    // meshes and out vertex indices need to be mapped to distinct values
    // first.
    // ---------------------------------------------------------------------
    UIntVector faceadjac(nfacesout), cntadjfac(maptbl.size(),0), ofsadjvec(maptbl.size()+1,0); {
    for (size_t t = 0; t < nmesh; ++t) {
        const aiMesh* const minp = smesh[t];
        for (unsigned int i = 0; i < minp->mNumFaces; ++i) {

            const aiFace& f = minp->mFaces[i];
            for (unsigned int n = 0; n < f.mNumIndices; ++n) {
                ++cntadjfac[maptbl[FLATTEN_VERTEX_IDX(t,f.mIndices[n])]];
            }
        }
    }
    unsigned int cur = 0;
    for (size_t i = 0; i < cntadjfac.size(); ++i) {
        ofsadjvec[i+1] = cur;
        cur += cntadjfac[i];
    }
    for (size_t t = 0; t < nmesh; ++t) {
        const aiMesh* const minp = smesh[t];
        for (unsigned int i = 0; i < minp->mNumFaces; ++i) {

            const aiFace& f = minp->mFaces[i];
            for (unsigned int n = 0; n < f.mNumIndices; ++n) {
                faceadjac[ofsadjvec[1+maptbl[FLATTEN_VERTEX_IDX(t,f.mIndices[n])]]++] = FLATTEN_FACE_IDX(t,i);
            }
        }
    }

    // check the other way round for consistency
#ifdef ASSIMP_BUILD_DEBUG

    for (size_t t = 0; t < ofsadjvec.size()-1; ++t) {
        for (unsigned int m = 0; m <  cntadjfac[t]; ++m) {
            const unsigned int fidx = faceadjac[ofsadjvec[t]+m];
            ai_assert(fidx < totfaces);
            for (size_t n = 1; n < nmesh; ++n) {

                if (moffsets[n].first > fidx) {
                    const aiMesh* msh = smesh[--n];
                    const aiFace& f = msh->mFaces[fidx-moffsets[n].first];

                    bool haveit = false;
                    for (unsigned int i = 0; i < f.mNumIndices; ++i) {
                        if (maptbl[FLATTEN_VERTEX_IDX(n,f.mIndices[i])]==(unsigned int)t) {
                            haveit = true;
                            break;
                        }
                    }
                    ai_assert(haveit);
                    if (!haveit) {
                        ASSIMP_LOG_DEBUG("Catmull-Clark Subdivider: Index not used");
                    }
                    break;
                }
            }
        }
    }

#endif
    }

#define GET_ADJACENT_FACES_AND_CNT(vidx,fstartout,numout) \
    fstartout = &faceadjac[ofsadjvec[vidx]], numout = cntadjfac[vidx]

    typedef std::pair<bool,Vertex> TouchedOVertex;
    std::vector<TouchedOVertex > new_points(num_unique,TouchedOVertex(false,Vertex()));
    // ---------------------------------------------------------------------
    // 5. Spawn a quad from each face point to the corresponding edge points
    // the original points being the fourth quad points.
    // ---------------------------------------------------------------------
    for (size_t t = 0; t < nmesh; ++t) {
        const aiMesh* const minp = smesh[t];
        aiMesh* const mout = out[t] = new aiMesh();

        for (unsigned int a  = 0; a < minp->mNumFaces; ++a) {
            mout->mNumFaces += minp->mFaces[a].mNumIndices;
        }

        // We need random access to the old face buffer, so reuse is not possible.
        mout->mFaces = new aiFace[mout->mNumFaces];

        mout->mNumVertices = mout->mNumFaces*4;
        mout->mVertices = new aiVector3D[mout->mNumVertices];

        // quads only, keep material index
        mout->mPrimitiveTypes = aiPrimitiveType_POLYGON;
        mout->mMaterialIndex = minp->mMaterialIndex;

        if (minp->HasNormals()) {
            mout->mNormals = new aiVector3D[mout->mNumVertices];
        }

        if (minp->HasTangentsAndBitangents()) {
            mout->mTangents = new aiVector3D[mout->mNumVertices];
            mout->mBitangents = new aiVector3D[mout->mNumVertices];
        }

        for(unsigned int i = 0; minp->HasTextureCoords(i); ++i) {
            mout->mTextureCoords[i] = new aiVector3D[mout->mNumVertices];
            mout->mNumUVComponents[i] = minp->mNumUVComponents[i];
        }

        for(unsigned int i = 0; minp->HasVertexColors(i); ++i) {
            mout->mColors[i] = new aiColor4D[mout->mNumVertices];
        }

        mout->mNumVertices = mout->mNumFaces<<2u;
        for (unsigned int i = 0, v = 0, n = 0; i < minp->mNumFaces;++i) {

            const aiFace& face = minp->mFaces[i];
            for (unsigned int a = 0; a < face.mNumIndices;++a)  {

                // Get a clean new face.
                aiFace& faceOut = mout->mFaces[n++];
                faceOut.mIndices = new unsigned int [faceOut.mNumIndices = 4];

                // Spawn a new quadrilateral (ccw winding) for this original point between:
                // a) face centroid
                centroids[FLATTEN_FACE_IDX(t,i)].SortBack(mout,faceOut.mIndices[0]=v++);

                // b) adjacent edge on the left, seen from the centroid
                const Edge& e0 = edges[MAKE_EDGE_HASH(maptbl[FLATTEN_VERTEX_IDX(t,face.mIndices[a])],
                    maptbl[FLATTEN_VERTEX_IDX(t,face.mIndices[a==face.mNumIndices-1?0:a+1])
                    ])];  // fixme: replace with mod face.mNumIndices?

                // c) adjacent edge on the right, seen from the centroid
                const Edge& e1 = edges[MAKE_EDGE_HASH(maptbl[FLATTEN_VERTEX_IDX(t,face.mIndices[a])],
                    maptbl[FLATTEN_VERTEX_IDX(t,face.mIndices[!a?face.mNumIndices-1:a-1])
                    ])];  // fixme: replace with mod face.mNumIndices?

                e0.edge_point.SortBack(mout,faceOut.mIndices[3]=v++);
                e1.edge_point.SortBack(mout,faceOut.mIndices[1]=v++);

                // d= original point P with distinct index i
                // F := 0
                // R := 0
                // n := 0
                // for each face f containing i
                //    F := F+ centroid of f
                //    R := R+ midpoint of edge of f from i to i+1
                //    n := n+1
                //
                // (F+2R+(n-3)P)/n
                const unsigned int org = maptbl[FLATTEN_VERTEX_IDX(t,face.mIndices[a])];
                TouchedOVertex& ov = new_points[org];

                if (!ov.first) {
                    ov.first = true;

                    const unsigned int* adj; unsigned int cnt;
                    GET_ADJACENT_FACES_AND_CNT(org,adj,cnt);

                    if (cnt < 3) {
                        ov.second = Vertex(minp,face.mIndices[a]);
                    }
                    else {

                        Vertex F,R;
                        for (unsigned int o = 0; o < cnt; ++o) {
                            ai_assert(adj[o] < totfaces);
                            F += centroids[adj[o]];

                            // adj[0] is a global face index - search the face in the mesh list
                            const aiMesh* mp = NULL;
                            size_t nidx;

                            if (adj[o] < moffsets[0].first) {
                                mp = smesh[nidx=0];
                            }
                            else {
                                for (nidx = 1; nidx<= nmesh; ++nidx) {
                                    if (nidx == nmesh ||moffsets[nidx].first > adj[o]) {
                                        mp = smesh[--nidx];
                                        break;
                                    }
                                }
                            }

                            ai_assert(adj[o]-moffsets[nidx].first < mp->mNumFaces);
                            const aiFace& f = mp->mFaces[adj[o]-moffsets[nidx].first];
                            bool haveit = false;

                            // find our original point in the face
                            for (unsigned int m = 0; m < f.mNumIndices; ++m) {
                                if (maptbl[FLATTEN_VERTEX_IDX(nidx,f.mIndices[m])] == org) {

                                    // add *both* edges. this way, we can be sure that we add
                                    // *all* adjacent edges to R. In a closed shape, every
                                    // edge is added twice - so we simply leave out the
                                    // factor 2.f in the amove formula and get the right
                                    // result.

                                    const Edge& c0 = edges[MAKE_EDGE_HASH(org,maptbl[FLATTEN_VERTEX_IDX(
                                        nidx,f.mIndices[!m?f.mNumIndices-1:m-1])])];
                                    // fixme: replace with mod face.mNumIndices?

                                    const Edge& c1 = edges[MAKE_EDGE_HASH(org,maptbl[FLATTEN_VERTEX_IDX(
                                        nidx,f.mIndices[m==f.mNumIndices-1?0:m+1])])];
                                    // fixme: replace with mod face.mNumIndices?
                                    R += c0.midpoint+c1.midpoint;

                                    haveit = true;
                                    break;
                                }
                            }

                            // this invariant *must* hold if the vertex-to-face adjacency table is valid
                            ai_assert(haveit);
                            if ( !haveit ) {
                                ASSIMP_LOG_WARN( "OBJ: no name for material library specified." );
                            }
                        }

                        const float div = static_cast<float>(cnt), divsq = 1.f/(div*div);
                        ov.second = Vertex(minp,face.mIndices[a])*((div-3.f) / div) + R*divsq + F*divsq;
                    }
                }
                ov.second.SortBack(mout,faceOut.mIndices[2]=v++);
            }
        }
    }
    }  // end of scope for edges, freeing its memory

    // ---------------------------------------------------------------------
    // 7. Apply the next subdivision step.
    // ---------------------------------------------------------------------
    if (num != 1) {
        std::vector<aiMesh*> tmp(nmesh);
        InternSubdivide (out,nmesh,&tmp.front(),num-1);
        for (size_t i = 0; i < nmesh; ++i) {
            delete out[i];
            out[i] = tmp[i];
        }
    }
}

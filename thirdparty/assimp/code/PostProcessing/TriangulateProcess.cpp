/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team

All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

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
---------------------------------------------------------------------------
*/

/** @file  TriangulateProcess.cpp
 *  @brief Implementation of the post processing step to split up
 *    all faces with more than three indices into triangles.
 *
 *
 *  The triangulation algorithm will handle concave or convex polygons.
 *  Self-intersecting or non-planar polygons are not rejected, but
 *  they're probably not triangulated correctly.
 *
 * DEBUG SWITCHES - do not enable any of them in release builds:
 *
 * AI_BUILD_TRIANGULATE_COLOR_FACE_WINDING
 *   - generates vertex colors to represent the face winding order.
 *     the first vertex of a polygon becomes red, the last blue.
 * AI_BUILD_TRIANGULATE_DEBUG_POLYS
 *   - dump all polygons and their triangulation sequences to
 *     a file
 */
#ifndef ASSIMP_BUILD_NO_TRIANGULATE_PROCESS

#include "PostProcessing/TriangulateProcess.h"
#include "PostProcessing/ProcessHelper.h"
#include "Common/PolyTools.h"

#include <memory>

//#define AI_BUILD_TRIANGULATE_COLOR_FACE_WINDING
//#define AI_BUILD_TRIANGULATE_DEBUG_POLYS

#define POLY_GRID_Y 40
#define POLY_GRID_X 70
#define POLY_GRID_XPAD 20
#define POLY_OUTPUT_FILE "assimp_polygons_debug.txt"

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
TriangulateProcess::TriangulateProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
TriangulateProcess::~TriangulateProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool TriangulateProcess::IsActive( unsigned int pFlags) const
{
    return (pFlags & aiProcess_Triangulate) != 0;
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void TriangulateProcess::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("TriangulateProcess begin");

    bool bHas = false;
    for( unsigned int a = 0; a < pScene->mNumMeshes; a++)
    {
        if (pScene->mMeshes[ a ]) {
            if ( TriangulateMesh( pScene->mMeshes[ a ] ) ) {
                bHas = true;
            }
        }
    }
    if ( bHas ) {
        ASSIMP_LOG_INFO( "TriangulateProcess finished. All polygons have been triangulated." );
    } else {
        ASSIMP_LOG_DEBUG( "TriangulateProcess finished. There was nothing to be done." );
    }
}

// ------------------------------------------------------------------------------------------------
// Triangulates the given mesh.
bool TriangulateProcess::TriangulateMesh( aiMesh* pMesh)
{
    // Now we have aiMesh::mPrimitiveTypes, so this is only here for test cases
    if (!pMesh->mPrimitiveTypes)    {
        bool bNeed = false;

        for( unsigned int a = 0; a < pMesh->mNumFaces; a++) {
            const aiFace& face = pMesh->mFaces[a];

            if( face.mNumIndices != 3)  {
                bNeed = true;
            }
        }
        if (!bNeed)
            return false;
    }
    else if (!(pMesh->mPrimitiveTypes & aiPrimitiveType_POLYGON)) {
        return false;
    }

    // Find out how many output faces we'll get
    unsigned int numOut = 0, max_out = 0;
    bool get_normals = true;
    for( unsigned int a = 0; a < pMesh->mNumFaces; a++) {
        aiFace& face = pMesh->mFaces[a];
        if (face.mNumIndices <= 4) {
            get_normals = false;
        }
        if( face.mNumIndices <= 3) {
            numOut++;

        }
        else {
            numOut += face.mNumIndices-2;
            max_out = std::max(max_out,face.mNumIndices);
        }
    }

    // Just another check whether aiMesh::mPrimitiveTypes is correct
    ai_assert(numOut != pMesh->mNumFaces);

    aiVector3D* nor_out = NULL;

    // if we don't have normals yet, but expect them to be a cheap side
    // product of triangulation anyway, allocate storage for them.
    if (!pMesh->mNormals && get_normals) {
        // XXX need a mechanism to inform the GenVertexNormals process to treat these normals as preprocessed per-face normals
    //  nor_out = pMesh->mNormals = new aiVector3D[pMesh->mNumVertices];
    }

    // the output mesh will contain triangles, but no polys anymore
    pMesh->mPrimitiveTypes |= aiPrimitiveType_TRIANGLE;
    pMesh->mPrimitiveTypes &= ~aiPrimitiveType_POLYGON;

    aiFace* out = new aiFace[numOut](), *curOut = out;
    std::vector<aiVector3D> temp_verts3d(max_out+2); /* temporary storage for vertices */
    std::vector<aiVector2D> temp_verts(max_out+2);

    // Apply vertex colors to represent the face winding?
#ifdef AI_BUILD_TRIANGULATE_COLOR_FACE_WINDING
    if (!pMesh->mColors[0])
        pMesh->mColors[0] = new aiColor4D[pMesh->mNumVertices];
    else
        new(pMesh->mColors[0]) aiColor4D[pMesh->mNumVertices];

    aiColor4D* clr = pMesh->mColors[0];
#endif

#ifdef AI_BUILD_TRIANGULATE_DEBUG_POLYS
    FILE* fout = fopen(POLY_OUTPUT_FILE,"a");
#endif

    const aiVector3D* verts = pMesh->mVertices;

    // use std::unique_ptr to avoid slow std::vector<bool> specialiations
    std::unique_ptr<bool[]> done(new bool[max_out]);
    for( unsigned int a = 0; a < pMesh->mNumFaces; a++) {
        aiFace& face = pMesh->mFaces[a];

        unsigned int* idx = face.mIndices;
        int num = (int)face.mNumIndices, ear = 0, tmp, prev = num-1, next = 0, max = num;

        // Apply vertex colors to represent the face winding?
#ifdef AI_BUILD_TRIANGULATE_COLOR_FACE_WINDING
        for (unsigned int i = 0; i < face.mNumIndices; ++i) {
            aiColor4D& c = clr[idx[i]];
            c.r = (i+1) / (float)max;
            c.b = 1.f - c.r;
        }
#endif

        aiFace* const last_face = curOut;

        // if it's a simple point,line or triangle: just copy it
        if( face.mNumIndices <= 3)
        {
            aiFace& nface = *curOut++;
            nface.mNumIndices = face.mNumIndices;
            nface.mIndices    = face.mIndices;

            face.mIndices = NULL;
            continue;
        }
        // optimized code for quadrilaterals
        else if ( face.mNumIndices == 4) {

            // quads can have at maximum one concave vertex. Determine
            // this vertex (if it exists) and start tri-fanning from
            // it.
            unsigned int start_vertex = 0;
            for (unsigned int i = 0; i < 4; ++i) {
                const aiVector3D& v0 = verts[face.mIndices[(i+3) % 4]];
                const aiVector3D& v1 = verts[face.mIndices[(i+2) % 4]];
                const aiVector3D& v2 = verts[face.mIndices[(i+1) % 4]];

                const aiVector3D& v = verts[face.mIndices[i]];

                aiVector3D left = (v0-v);
                aiVector3D diag = (v1-v);
                aiVector3D right = (v2-v);

                left.Normalize();
                diag.Normalize();
                right.Normalize();

                const float angle = std::acos(left*diag) + std::acos(right*diag);
                if (angle > AI_MATH_PI_F) {
                    // this is the concave point
                    start_vertex = i;
                    break;
                }
            }

            const unsigned int temp[] = {face.mIndices[0], face.mIndices[1], face.mIndices[2], face.mIndices[3]};

            aiFace& nface = *curOut++;
            nface.mNumIndices = 3;
            nface.mIndices = face.mIndices;

            nface.mIndices[0] = temp[start_vertex];
            nface.mIndices[1] = temp[(start_vertex + 1) % 4];
            nface.mIndices[2] = temp[(start_vertex + 2) % 4];

            aiFace& sface = *curOut++;
            sface.mNumIndices = 3;
            sface.mIndices = new unsigned int[3];

            sface.mIndices[0] = temp[start_vertex];
            sface.mIndices[1] = temp[(start_vertex + 2) % 4];
            sface.mIndices[2] = temp[(start_vertex + 3) % 4];

            // prevent double deletion of the indices field
            face.mIndices = NULL;
            continue;
        }
        else
        {
            // A polygon with more than 3 vertices can be either concave or convex.
            // Usually everything we're getting is convex and we could easily
            // triangulate by tri-fanning. However, LightWave is probably the only
            // modeling suite to make extensive use of highly concave, monster polygons ...
            // so we need to apply the full 'ear cutting' algorithm to get it right.

            // RERQUIREMENT: polygon is expected to be simple and *nearly* planar.
            // We project it onto a plane to get a 2d triangle.

            // Collect all vertices of of the polygon.
           for (tmp = 0; tmp < max; ++tmp) {
                temp_verts3d[tmp] = verts[idx[tmp]];
            }

            // Get newell normal of the polygon. Store it for future use if it's a polygon-only mesh
            aiVector3D n;
            NewellNormal<3,3,3>(n,max,&temp_verts3d.front().x,&temp_verts3d.front().y,&temp_verts3d.front().z);
            if (nor_out) {
                 for (tmp = 0; tmp < max; ++tmp)
                     nor_out[idx[tmp]] = n;
            }

            // Select largest normal coordinate to ignore for projection
            const float ax = (n.x>0 ? n.x : -n.x);
            const float ay = (n.y>0 ? n.y : -n.y);
            const float az = (n.z>0 ? n.z : -n.z);

            unsigned int ac = 0, bc = 1; /* no z coord. projection to xy */
            float inv = n.z;
            if (ax > ay) {
                if (ax > az) { /* no x coord. projection to yz */
                    ac = 1; bc = 2;
                    inv = n.x;
                }
            }
            else if (ay > az) { /* no y coord. projection to zy */
                ac = 2; bc = 0;
                inv = n.y;
            }

            // Swap projection axes to take the negated projection vector into account
            if (inv < 0.f) {
                std::swap(ac,bc);
            }

            for (tmp =0; tmp < max; ++tmp) {
                temp_verts[tmp].x = verts[idx[tmp]][ac];
                temp_verts[tmp].y = verts[idx[tmp]][bc];
                done[tmp] = false;
            }

#ifdef AI_BUILD_TRIANGULATE_DEBUG_POLYS
            // plot the plane onto which we mapped the polygon to a 2D ASCII pic
            aiVector2D bmin,bmax;
            ArrayBounds(&temp_verts[0],max,bmin,bmax);

            char grid[POLY_GRID_Y][POLY_GRID_X+POLY_GRID_XPAD];
            std::fill_n((char*)grid,POLY_GRID_Y*(POLY_GRID_X+POLY_GRID_XPAD),' ');

            for (int i =0; i < max; ++i) {
                const aiVector2D& v = (temp_verts[i] - bmin) / (bmax-bmin);
                const size_t x = static_cast<size_t>(v.x*(POLY_GRID_X-1)), y = static_cast<size_t>(v.y*(POLY_GRID_Y-1));
                char* loc = grid[y]+x;
                if (grid[y][x] != ' ') {
                    for(;*loc != ' '; ++loc);
                    *loc++ = '_';
                }
                *(loc+::ai_snprintf(loc, POLY_GRID_XPAD,"%i",i)) = ' ';
            }


            for(size_t y = 0; y < POLY_GRID_Y; ++y) {
                grid[y][POLY_GRID_X+POLY_GRID_XPAD-1] = '\0';
                fprintf(fout,"%s\n",grid[y]);
            }

            fprintf(fout,"\ntriangulation sequence: ");
#endif

            //
            // FIXME: currently this is the slow O(kn) variant with a worst case
            // complexity of O(n^2) (I think). Can be done in O(n).
            while (num > 3) {

                // Find the next ear of the polygon
                int num_found = 0;
                for (ear = next;;prev = ear,ear = next) {

                    // break after we looped two times without a positive match
                    for (next=ear+1;done[(next>=max?next=0:next)];++next);
                    if (next < ear) {
                        if (++num_found == 2) {
                            break;
                        }
                    }
                    const aiVector2D* pnt1 = &temp_verts[ear],
                        *pnt0 = &temp_verts[prev],
                        *pnt2 = &temp_verts[next];

                    // Must be a convex point. Assuming ccw winding, it must be on the right of the line between p-1 and p+1.
                    if (OnLeftSideOfLine2D(*pnt0,*pnt2,*pnt1)) {
                        continue;
                    }

                    // and no other point may be contained in this triangle
                    for ( tmp = 0; tmp < max; ++tmp) {

                        // We need to compare the actual values because it's possible that multiple indexes in
                        // the polygon are referring to the same position. concave_polygon.obj is a sample
                        //
                        // FIXME: Use 'epsiloned' comparisons instead? Due to numeric inaccuracies in
                        // PointInTriangle() I'm guessing that it's actually possible to construct
                        // input data that would cause us to end up with no ears. The problem is,
                        // which epsilon? If we chose a too large value, we'd get wrong results
                        const aiVector2D& vtmp = temp_verts[tmp];
                        if ( vtmp != *pnt1 && vtmp != *pnt2 && vtmp != *pnt0 && PointInTriangle2D(*pnt0,*pnt1,*pnt2,vtmp)) {
                            break;
                        }
                    }
                    if (tmp != max) {
                        continue;
                    }

                    // this vertex is an ear
                    break;
                }
                if (num_found == 2) {

                    // Due to the 'two ear theorem', every simple polygon with more than three points must
                    // have 2 'ears'. Here's definitely something wrong ... but we don't give up yet.
                    //

                    // Instead we're continuing with the standard tri-fanning algorithm which we'd
                    // use if we had only convex polygons. That's life.
                    ASSIMP_LOG_ERROR("Failed to triangulate polygon (no ear found). Probably not a simple polygon?");

#ifdef AI_BUILD_TRIANGULATE_DEBUG_POLYS
                    fprintf(fout,"critical error here, no ear found! ");
#endif
                    num = 0;
                    break;

                    curOut -= (max-num); /* undo all previous work */
                    for (tmp = 0; tmp < max-2; ++tmp) {
                        aiFace& nface = *curOut++;

                        nface.mNumIndices = 3;
                        if (!nface.mIndices)
                            nface.mIndices = new unsigned int[3];

                        nface.mIndices[0] = 0;
                        nface.mIndices[1] = tmp+1;
                        nface.mIndices[2] = tmp+2;

                    }
                    num = 0;
                    break;
                }

                aiFace& nface = *curOut++;
                nface.mNumIndices = 3;

                if (!nface.mIndices) {
                    nface.mIndices = new unsigned int[3];
                }

                // setup indices for the new triangle ...
                nface.mIndices[0] = prev;
                nface.mIndices[1] = ear;
                nface.mIndices[2] = next;

                // exclude the ear from most further processing
                done[ear] = true;
                --num;
            }
            if (num > 0) {
                // We have three indices forming the last 'ear' remaining. Collect them.
                aiFace& nface = *curOut++;
                nface.mNumIndices = 3;
                if (!nface.mIndices) {
                    nface.mIndices = new unsigned int[3];
                }

                for (tmp = 0; done[tmp]; ++tmp);
                nface.mIndices[0] = tmp;

                for (++tmp; done[tmp]; ++tmp);
                nface.mIndices[1] = tmp;

                for (++tmp; done[tmp]; ++tmp);
                nface.mIndices[2] = tmp;

            }
        }

#ifdef AI_BUILD_TRIANGULATE_DEBUG_POLYS

        for(aiFace* f = last_face; f != curOut; ++f) {
            unsigned int* i = f->mIndices;
            fprintf(fout," (%i %i %i)",i[0],i[1],i[2]);
        }

        fprintf(fout,"\n*********************************************************************\n");
        fflush(fout);

#endif

        for(aiFace* f = last_face; f != curOut; ) {
            unsigned int* i = f->mIndices;

            //  drop dumb 0-area triangles - deactivated for now:
            //FindDegenerates post processing step can do the same thing
            //if (std::fabs(GetArea2D(temp_verts[i[0]],temp_verts[i[1]],temp_verts[i[2]])) < 1e-5f) {
            //    ASSIMP_LOG_DEBUG("Dropping triangle with area 0");
            //    --curOut;

            //    delete[] f->mIndices;
            //    f->mIndices = nullptr;

            //    for(aiFace* ff = f; ff != curOut; ++ff) {
            //        ff->mNumIndices = (ff+1)->mNumIndices;
            //        ff->mIndices = (ff+1)->mIndices;
            //        (ff+1)->mIndices = nullptr;
            //    }
            //    continue;
            //}

            i[0] = idx[i[0]];
            i[1] = idx[i[1]];
            i[2] = idx[i[2]];
            ++f;
        }

        delete[] face.mIndices;
        face.mIndices = NULL;
    }

#ifdef AI_BUILD_TRIANGULATE_DEBUG_POLYS
    fclose(fout);
#endif

    // kill the old faces
    delete [] pMesh->mFaces;

    // ... and store the new ones
    pMesh->mFaces    = out;
    pMesh->mNumFaces = (unsigned int)(curOut-out); /* not necessarily equal to numOut */
    return true;
}

#endif // !! ASSIMP_BUILD_NO_TRIANGULATE_PROCESS

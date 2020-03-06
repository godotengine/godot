/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team


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

/** @file   StandardShapes.cpp
 *  @brief  Implementation of the StandardShapes class
 *
 *  The primitive geometry data comes from
 *  http://geometrictools.com/Documentation/PlatonicSolids.pdf.
 */

#include <assimp/StandardShapes.h>
#include <assimp/StringComparison.h>
#include <stddef.h>
#include <assimp/Defines.h>
#include <assimp/mesh.h>

namespace Assimp    {


# define ADD_TRIANGLE(n0,n1,n2) \
    positions.push_back(n0); \
    positions.push_back(n1); \
    positions.push_back(n2);

#   define ADD_PENTAGON(n0,n1,n2,n3,n4) \
    if (polygons) \
    { \
        positions.push_back(n0); \
        positions.push_back(n1); \
        positions.push_back(n2); \
        positions.push_back(n3); \
        positions.push_back(n4); \
    } \
    else \
    { \
        ADD_TRIANGLE(n0, n1, n2) \
        ADD_TRIANGLE(n0, n2, n3) \
        ADD_TRIANGLE(n0, n3, n4) \
    }

#   define ADD_QUAD(n0,n1,n2,n3) \
    if (polygons) \
    { \
        positions.push_back(n0); \
        positions.push_back(n1); \
        positions.push_back(n2); \
        positions.push_back(n3); \
    } \
    else \
    { \
        ADD_TRIANGLE(n0, n1, n2) \
        ADD_TRIANGLE(n0, n2, n3) \
    }


// ------------------------------------------------------------------------------------------------
// Fast subdivision for a mesh whose verts have a magnitude of 1
void Subdivide(std::vector<aiVector3D>& positions)
{
    // assume this to be constant - (fixme: must be 1.0? I think so)
    const ai_real fl1 = positions[0].Length();

    unsigned int origSize = (unsigned int)positions.size();
    for (unsigned int i = 0 ; i < origSize ; i+=3)
    {
        aiVector3D& tv0 = positions[i];
        aiVector3D& tv1 = positions[i+1];
        aiVector3D& tv2 = positions[i+2];

        aiVector3D a = tv0, b = tv1, c = tv2;
        aiVector3D v1 = aiVector3D(a.x+b.x, a.y+b.y, a.z+b.z).Normalize()*fl1;
        aiVector3D v2 = aiVector3D(a.x+c.x, a.y+c.y, a.z+c.z).Normalize()*fl1;
        aiVector3D v3 = aiVector3D(b.x+c.x, b.y+c.y, b.z+c.z).Normalize()*fl1;

        tv0 = v1; tv1 = v3; tv2 = v2; // overwrite the original
        ADD_TRIANGLE(v1, v2, a);
        ADD_TRIANGLE(v2, v3, c);
        ADD_TRIANGLE(v3, v1, b);
    }
}

// ------------------------------------------------------------------------------------------------
// Construct a mesh from given vertex positions
aiMesh* StandardShapes::MakeMesh(const std::vector<aiVector3D>& positions,
    unsigned int numIndices)
{
    if (positions.empty() || !numIndices) return NULL;

    // Determine which kinds of primitives the mesh consists of
    aiMesh* out = new aiMesh();
    switch (numIndices) {
        case 1:
            out->mPrimitiveTypes = aiPrimitiveType_POINT;
            break;
        case 2:
            out->mPrimitiveTypes = aiPrimitiveType_LINE;
            break;
        case 3:
            out->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
            break;
        default:
            out->mPrimitiveTypes = aiPrimitiveType_POLYGON;
            break;
    };

    out->mNumFaces = (unsigned int)positions.size() / numIndices;
    out->mFaces = new aiFace[out->mNumFaces];
    for (unsigned int i = 0, a = 0; i < out->mNumFaces;++i) {
        aiFace& f = out->mFaces[i];
        f.mNumIndices = numIndices;
        f.mIndices = new unsigned int[numIndices];
        for (unsigned int j = 0; i < numIndices; ++i, ++a) {
            f.mIndices[j] = a;
        }
    }
    out->mNumVertices = (unsigned int)positions.size();
    out->mVertices = new aiVector3D[out->mNumVertices];
    ::memcpy(out->mVertices,&positions[0],out->mNumVertices*sizeof(aiVector3D));

    return out;
}

// ------------------------------------------------------------------------------------------------
// Construct a mesh with a specific shape (callback)
aiMesh* StandardShapes::MakeMesh ( unsigned int (*GenerateFunc)(
    std::vector<aiVector3D>&))
{
    std::vector<aiVector3D> temp;
    unsigned num = (*GenerateFunc)(temp);
    return MakeMesh(temp,num);
}

// ------------------------------------------------------------------------------------------------
// Construct a mesh with a specific shape (callback)
aiMesh* StandardShapes::MakeMesh ( unsigned int (*GenerateFunc)(
    std::vector<aiVector3D>&, bool))
{
    std::vector<aiVector3D> temp;
    unsigned num = (*GenerateFunc)(temp,true);
    return MakeMesh(temp,num);
}

// ------------------------------------------------------------------------------------------------
// Construct a mesh with a specific shape (callback)
aiMesh* StandardShapes::MakeMesh (unsigned int num,  void (*GenerateFunc)(
    unsigned int,std::vector<aiVector3D>&))
{
    std::vector<aiVector3D> temp;
    (*GenerateFunc)(num,temp);
    return MakeMesh(temp,3);
}

// ------------------------------------------------------------------------------------------------
// Build an incosahedron with points.magnitude == 1
unsigned int StandardShapes::MakeIcosahedron(std::vector<aiVector3D>& positions)
{
    positions.reserve(positions.size()+60);

    const ai_real t = ( ai_real( 1.0 )+ ai_real( 2.236067977 ) ) / ai_real( 2.0 );
    const ai_real s = std::sqrt(ai_real(1.0) + t*t);

    const aiVector3D v0  = aiVector3D(t,1.0, 0.0)/s;
    const aiVector3D v1  = aiVector3D(-t,1.0, 0.0)/s;
    const aiVector3D v2  = aiVector3D(t,-1.0, 0.0)/s;
    const aiVector3D v3  = aiVector3D(-t,-1.0, 0.0)/s;
    const aiVector3D v4  = aiVector3D(1.0, 0.0, t)/s;
    const aiVector3D v5  = aiVector3D(1.0, 0.0,-t)/s;
    const aiVector3D v6  = aiVector3D(-1.0, 0.0,t)/s;
    const aiVector3D v7  = aiVector3D(-1.0, 0.0,-t)/s;
    const aiVector3D v8  = aiVector3D(0.0, t, 1.0)/s;
    const aiVector3D v9  = aiVector3D(0.0,-t, 1.0)/s;
    const aiVector3D v10 = aiVector3D(0.0, t,-1.0)/s;
    const aiVector3D v11 = aiVector3D(0.0,-t,-1.0)/s;

    ADD_TRIANGLE(v0,v8,v4);
    ADD_TRIANGLE(v0,v5,v10);
    ADD_TRIANGLE(v2,v4,v9);
    ADD_TRIANGLE(v2,v11,v5);

    ADD_TRIANGLE(v1,v6,v8);
    ADD_TRIANGLE(v1,v10,v7);
    ADD_TRIANGLE(v3,v9,v6);
    ADD_TRIANGLE(v3,v7,v11);

    ADD_TRIANGLE(v0,v10,v8);
    ADD_TRIANGLE(v1,v8,v10);
    ADD_TRIANGLE(v2,v9,v11);
    ADD_TRIANGLE(v3,v11,v9);

    ADD_TRIANGLE(v4,v2,v0);
    ADD_TRIANGLE(v5,v0,v2);
    ADD_TRIANGLE(v6,v1,v3);
    ADD_TRIANGLE(v7,v3,v1);

    ADD_TRIANGLE(v8,v6,v4);
    ADD_TRIANGLE(v9,v4,v6);
    ADD_TRIANGLE(v10,v5,v7);
    ADD_TRIANGLE(v11,v7,v5);
    return 3;
}

// ------------------------------------------------------------------------------------------------
// Build a dodecahedron with points.magnitude == 1
unsigned int StandardShapes::MakeDodecahedron(std::vector<aiVector3D>& positions,
    bool polygons /*= false*/)
{
    positions.reserve(positions.size()+108);

    const ai_real a = ai_real( 1.0 ) / ai_real(1.7320508);
    const ai_real b = std::sqrt(( ai_real( 3.0 )- ai_real( 2.23606797))/ ai_real( 6.0) );
    const ai_real c = std::sqrt(( ai_real( 3.0 )+ ai_real( 2.23606797f))/ ai_real( 6.0) );

    const aiVector3D v0  = aiVector3D(a,a,a);
    const aiVector3D v1  = aiVector3D(a,a,-a);
    const aiVector3D v2  = aiVector3D(a,-a,a);
    const aiVector3D v3  = aiVector3D(a,-a,-a);
    const aiVector3D v4  = aiVector3D(-a,a,a);
    const aiVector3D v5  = aiVector3D(-a,a,-a);
    const aiVector3D v6  = aiVector3D(-a,-a,a);
    const aiVector3D v7  = aiVector3D(-a,-a,-a);
    const aiVector3D v8  = aiVector3D(b,c,0.0);
    const aiVector3D v9  = aiVector3D(-b,c,0.0);
    const aiVector3D v10 = aiVector3D(b,-c,0.0);
    const aiVector3D v11 = aiVector3D(-b,-c,0.0);
    const aiVector3D v12 = aiVector3D(c, 0.0, b);
    const aiVector3D v13 = aiVector3D(c, 0.0, -b);
    const aiVector3D v14 = aiVector3D(-c, 0.0, b);
    const aiVector3D v15 = aiVector3D(-c, 0.0, -b);
    const aiVector3D v16 = aiVector3D(0.0, b, c);
    const aiVector3D v17 = aiVector3D(0.0, -b, c);
    const aiVector3D v18 = aiVector3D(0.0, b, -c);
    const aiVector3D v19 = aiVector3D(0.0, -b, -c);

    ADD_PENTAGON(v0, v8, v9, v4, v16);
    ADD_PENTAGON(v0, v12, v13, v1, v8);
    ADD_PENTAGON(v0, v16, v17, v2, v12);
    ADD_PENTAGON(v8, v1, v18, v5, v9);
    ADD_PENTAGON(v12, v2, v10, v3, v13);
    ADD_PENTAGON(v16, v4, v14, v6, v17);
    ADD_PENTAGON(v9, v5, v15, v14, v4);

    ADD_PENTAGON(v6, v11, v10, v2, v17);
    ADD_PENTAGON(v3, v19, v18, v1, v13);
    ADD_PENTAGON(v7, v15, v5, v18, v19);
    ADD_PENTAGON(v7, v11, v6, v14, v15);
    ADD_PENTAGON(v7, v19, v3, v10, v11);
    return (polygons ? 5 : 3);
}

// ------------------------------------------------------------------------------------------------
// Build an octahedron with points.magnitude == 1
unsigned int StandardShapes::MakeOctahedron(std::vector<aiVector3D>& positions)
{
    positions.reserve(positions.size()+24);

    const aiVector3D v0  = aiVector3D(1.0, 0.0, 0.0) ;
    const aiVector3D v1  = aiVector3D(-1.0, 0.0, 0.0);
    const aiVector3D v2  = aiVector3D(0.0, 1.0, 0.0);
    const aiVector3D v3  = aiVector3D(0.0, -1.0, 0.0);
    const aiVector3D v4  = aiVector3D(0.0, 0.0, 1.0);
    const aiVector3D v5  = aiVector3D(0.0, 0.0, -1.0);

    ADD_TRIANGLE(v4,v0,v2);
    ADD_TRIANGLE(v4,v2,v1);
    ADD_TRIANGLE(v4,v1,v3);
    ADD_TRIANGLE(v4,v3,v0);

    ADD_TRIANGLE(v5,v2,v0);
    ADD_TRIANGLE(v5,v1,v2);
    ADD_TRIANGLE(v5,v3,v1);
    ADD_TRIANGLE(v5,v0,v3);
    return 3;
}

// ------------------------------------------------------------------------------------------------
// Build a tetrahedron with points.magnitude == 1
unsigned int StandardShapes::MakeTetrahedron(std::vector<aiVector3D>& positions)
{
    positions.reserve(positions.size()+9);

    const ai_real invThree = ai_real( 1.0 ) / ai_real( 3.0 );
    const ai_real a = ai_real( 1.41421 ) * invThree;
    const ai_real b = ai_real( 2.4494 ) * invThree;

    const aiVector3D v0  = aiVector3D(0.0,0.0,1.0);
    const aiVector3D v1  = aiVector3D(2*a,0,-invThree );
    const aiVector3D v2  = aiVector3D(-a,b,-invThree );
    const aiVector3D v3  = aiVector3D(-a,-b,-invThree );

    ADD_TRIANGLE(v0,v1,v2);
    ADD_TRIANGLE(v0,v2,v3);
    ADD_TRIANGLE(v0,v3,v1);
    ADD_TRIANGLE(v1,v3,v2);
    return 3;
}

// ------------------------------------------------------------------------------------------------
// Build a hexahedron with points.magnitude == 1
unsigned int StandardShapes::MakeHexahedron(std::vector<aiVector3D>& positions,
    bool polygons /*= false*/)
{
    positions.reserve(positions.size()+36);
    const ai_real length = ai_real(1.0)/ai_real(1.73205080);

    const aiVector3D v0  = aiVector3D(-1.0,-1.0,-1.0)*length;
    const aiVector3D v1  = aiVector3D(1.0,-1.0,-1.0)*length;
    const aiVector3D v2  = aiVector3D(1.0,1.0,-1.0)*length;
    const aiVector3D v3  = aiVector3D(-1.0,1.0,-1.0)*length;
    const aiVector3D v4  = aiVector3D(-1.0,-1.0,1.0)*length;
    const aiVector3D v5  = aiVector3D(1.0,-1.0,1.0)*length;
    const aiVector3D v6  = aiVector3D(1.0,1.0,1.0)*length;
    const aiVector3D v7  = aiVector3D(-1.0,1.0,1.0)*length;

    ADD_QUAD(v0,v3,v2,v1);
    ADD_QUAD(v0,v1,v5,v4);
    ADD_QUAD(v0,v4,v7,v3);
    ADD_QUAD(v6,v5,v1,v2);
    ADD_QUAD(v6,v2,v3,v7);
    ADD_QUAD(v6,v7,v4,v5);
    return (polygons ? 4 : 3);
}

// Cleanup ...
#undef ADD_TRIANGLE
#undef ADD_QUAD
#undef ADD_PENTAGON

// ------------------------------------------------------------------------------------------------
// Create a subdivision sphere
void StandardShapes::MakeSphere(unsigned int    tess,
    std::vector<aiVector3D>& positions)
{
    // Reserve enough storage. Every subdivision
    // splits each triangle in 4, the icosahedron consists of 60 verts
    positions.reserve(positions.size()+60 * integer_pow(4, tess));

    // Construct an icosahedron to start with
    MakeIcosahedron(positions);

    // ... and subdivide it until the requested output
    // tessellation is reached
    for (unsigned int i = 0; i<tess;++i)
        Subdivide(positions);
}

// ------------------------------------------------------------------------------------------------
// Build a cone
void StandardShapes::MakeCone(ai_real height,ai_real radius1,
    ai_real radius2,unsigned int tess,
    std::vector<aiVector3D>& positions,bool bOpen /*= false */)
{
    // Sorry, a cone with less than 3 segments makes ABSOLUTELY NO SENSE
    if (tess < 3 || !height)
        return;

    size_t old = positions.size();

    // No negative radii
    radius1 = std::fabs(radius1);
    radius2 = std::fabs(radius2);

    ai_real halfHeight = height / ai_real(2.0);

    // radius1 is always the smaller one
    if (radius2 > radius1)
    {
        std::swap(radius2,radius1);
        halfHeight = -halfHeight;
    }
    else old = SIZE_MAX;

    // Use a large epsilon to check whether the cone is pointy
    if (radius1 < (radius2-radius1)*10e-3)radius1 = 0.0;

    // We will need 3*2 verts per segment + 3*2 verts per segment
    // if the cone is closed
    const unsigned int mem = tess*6 + (!bOpen ? tess*3 * (radius1 ? 2 : 1) : 0);
    positions.reserve(positions.size () + mem);

    // Now construct all segments
    const ai_real angle_delta = (ai_real)AI_MATH_TWO_PI / tess;
    const ai_real angle_max   = (ai_real)AI_MATH_TWO_PI;

    ai_real s = 1.0; // std::cos(angle == 0);
    ai_real t = 0.0; // std::sin(angle == 0);

    for (ai_real angle = 0.0; angle < angle_max; )
    {
        const aiVector3D v1 = aiVector3D (s * radius1, -halfHeight, t * radius1 );
        const aiVector3D v2 = aiVector3D (s * radius2,  halfHeight, t * radius2 );

        const ai_real next = angle + angle_delta;
        ai_real s2 = std::cos(next);
        ai_real t2 = std::sin(next);

        const aiVector3D v3 = aiVector3D (s2 * radius2,  halfHeight, t2 * radius2 );
        const aiVector3D v4 = aiVector3D (s2 * radius1, -halfHeight, t2 * radius1 );

        positions.push_back(v1);
        positions.push_back(v2);
        positions.push_back(v3);
        positions.push_back(v4);
        positions.push_back(v1);
        positions.push_back(v3);

        if (!bOpen)
        {
            // generate the end 'cap'
            positions.push_back(aiVector3D(s * radius2,  halfHeight, t * radius2 ));
            positions.push_back(aiVector3D(s2 * radius2,  halfHeight, t2 * radius2 ));
            positions.push_back(aiVector3D(0.0, halfHeight, 0.0));


            if (radius1)
            {
                // generate the other end 'cap'
                positions.push_back(aiVector3D(s * radius1,  -halfHeight, t * radius1 ));
                positions.push_back(aiVector3D(s2 * radius1,  -halfHeight, t2 * radius1 ));
                positions.push_back(aiVector3D(0.0, -halfHeight, 0.0));

            }
        }
        s = s2;
        t = t2;
        angle = next;
    }

    // Need to flip face order?
    if ( SIZE_MAX != old )  {
        for (size_t p = old; p < positions.size();p += 3) {
            std::swap(positions[p],positions[p+1]);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Build a circle
void StandardShapes::MakeCircle(ai_real radius, unsigned int tess,
    std::vector<aiVector3D>& positions)
{
    // Sorry, a circle with less than 3 segments makes ABSOLUTELY NO SENSE
    if (tess < 3 || !radius)
        return;

    radius = std::fabs(radius);

    // We will need 3 vertices per segment
    positions.reserve(positions.size()+tess*3);

    const ai_real angle_delta = (ai_real)AI_MATH_TWO_PI / tess;
    const ai_real angle_max   = (ai_real)AI_MATH_TWO_PI;

    ai_real s = 1.0; // std::cos(angle == 0);
    ai_real t = 0.0; // std::sin(angle == 0);

    for (ai_real angle = 0.0; angle < angle_max;  )
    {
        positions.push_back(aiVector3D(s * radius,0.0,t * radius));
        angle += angle_delta;
        s = std::cos(angle);
        t = std::sin(angle);
        positions.push_back(aiVector3D(s * radius,0.0,t * radius));

        positions.push_back(aiVector3D(0.0,0.0,0.0));
    }
}

} // ! Assimp

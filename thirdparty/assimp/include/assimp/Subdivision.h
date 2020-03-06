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

/** @file Defines a helper class to evaluate subdivision surfaces.*/
#pragma once
#ifndef AI_SUBDISIVION_H_INC
#define AI_SUBDISIVION_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/types.h>

struct aiMesh;

namespace Assimp    {

// ------------------------------------------------------------------------------
/** Helper class to evaluate subdivision surfaces. Different algorithms
 *  are provided for choice. */
// ------------------------------------------------------------------------------
class ASSIMP_API Subdivider {
public:

    /** Enumerates all supported subvidision algorithms */
    enum Algorithm  {
        CATMULL_CLARKE = 0x1
    };

    virtual ~Subdivider();

    // ---------------------------------------------------------------
    /** Create a subdivider of a specific type
     *
     *  @param algo Algorithm to be used for subdivision
     *  @return Subdivider instance. */
    static Subdivider* Create (Algorithm algo);

    // ---------------------------------------------------------------
    /** Subdivide a mesh using the selected algorithm
     *
     *  @param mesh First mesh to be subdivided. Must be in verbose
     *    format.
     *  @param out Receives the output mesh, allocated by me.
     *  @param num Number of subdivisions to perform.
     *  @param discard_input If true is passed, the input mesh is
     *    deleted after the subdivision is complete. This can
     *    improve performance because it allows the optimization
     *    to reuse the existing mesh for intermediate results.
     *  @pre out!=mesh*/
    virtual void Subdivide ( aiMesh* mesh,
        aiMesh*& out, unsigned int num,
        bool discard_input = false) = 0;

    // ---------------------------------------------------------------
    /** Subdivide multiple meshes using the selected algorithm. This
     *  avoids erroneous smoothing on objects consisting of multiple
     *  per-material meshes. Usually, most 3d modellers smooth on a
     *  per-object base, regardless the materials assigned to the
     *  meshes.
     *
     *  @param smesh Array of meshes to be subdivided. Must be in
     *    verbose format.
     *  @param nmesh Number of meshes in smesh.
     *  @param out Receives the output meshes. The array must be
     *    sufficiently large (at least @c nmesh elements) and may not
     *    overlap the input array. Output meshes map one-to-one to
     *    their corresponding input meshes. The meshes are allocated
     *    by the function.
     *  @param discard_input If true is passed, input meshes are
     *    deleted after the subdivision is complete. This can
     *    improve performance because it allows the optimization
     *    of reusing existing meshes for intermediate results.
     *  @param num Number of subdivisions to perform.
     *  @pre nmesh != 0, smesh and out may not overlap*/
    virtual void Subdivide (
        aiMesh** smesh,
        size_t nmesh,
        aiMesh** out,
        unsigned int num,
        bool discard_input = false) = 0;

};

inline
Subdivider::~Subdivider() {
    // empty
}

} // end namespace Assimp


#endif // !!  AI_SUBDISIVION_H_INC


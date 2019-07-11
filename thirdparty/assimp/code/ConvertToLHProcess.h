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

/** @file  MakeLeftHandedProcess.h
 *  @brief Defines a bunch of post-processing steps to handle
 *    coordinate system conversions.
 *
 *  - LH to RH
 *  - UV origin upper-left to lower-left
 *  - face order cw to ccw
 */
#ifndef AI_CONVERTTOLHPROCESS_H_INC
#define AI_CONVERTTOLHPROCESS_H_INC

#include <assimp/types.h>
#include "BaseProcess.h"

struct aiMesh;
struct aiNodeAnim;
struct aiNode;
struct aiMaterial;

namespace Assimp    {

// -----------------------------------------------------------------------------------
/** @brief The MakeLeftHandedProcess converts all imported data to a left-handed
 *   coordinate system.
 *
 * This implies a mirroring of the Z axis of the coordinate system. But to keep
 * transformation matrices free from reflections we shift the reflection to other
 * places. We mirror the meshes and adapt the rotations.
 *
 * @note RH-LH and LH-RH is the same, so this class can be used for both
 */
class MakeLeftHandedProcess : public BaseProcess
{


public:
    MakeLeftHandedProcess();
    ~MakeLeftHandedProcess();

    // -------------------------------------------------------------------
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    void Execute( aiScene* pScene);

protected:

    // -------------------------------------------------------------------
    /** Recursively converts a node and all of its children
     */
    void ProcessNode( aiNode* pNode, const aiMatrix4x4& pParentGlobalRotation);

    // -------------------------------------------------------------------
    /** Converts a single mesh to left handed coordinates.
     * This means that positions, normals and tangents are mirrored at
     * the local Z axis and the order of all faces are inverted.
     * @param pMesh The mesh to convert.
     */
    void ProcessMesh( aiMesh* pMesh);

    // -------------------------------------------------------------------
    /** Converts a single material to left-handed coordinates
     * @param pMat Material to convert
     */
    void ProcessMaterial( aiMaterial* pMat);

    // -------------------------------------------------------------------
    /** Converts the given animation to LH coordinates.
     * The rotation and translation keys are transformed, the scale keys
     * work in local space and can therefore be left untouched.
     * @param pAnim The bone animation to transform
     */
    void ProcessAnimation( aiNodeAnim* pAnim);
};


// ---------------------------------------------------------------------------
/** Postprocessing step to flip the face order of the imported data
 */
class FlipWindingOrderProcess : public BaseProcess
{
    friend class Importer;

public:
    /** Constructor to be privately used by Importer */
    FlipWindingOrderProcess();

    /** Destructor, private as well */
    ~FlipWindingOrderProcess();

    // -------------------------------------------------------------------
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    void Execute( aiScene* pScene);

protected:
    void ProcessMesh( aiMesh* pMesh);
};

// ---------------------------------------------------------------------------
/** Postprocessing step to flip the UV coordinate system of the import data
 */
class FlipUVsProcess : public BaseProcess
{
    friend class Importer;

public:
    /** Constructor to be privately used by Importer */
    FlipUVsProcess();

    /** Destructor, private as well */
    ~FlipUVsProcess();

    // -------------------------------------------------------------------
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    void Execute( aiScene* pScene);

protected:
    void ProcessMesh( aiMesh* pMesh);
    void ProcessMaterial( aiMaterial* mat);
};

} // end of namespace Assimp

#endif // AI_CONVERTTOLHPROCESS_H_INC

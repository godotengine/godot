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

/** Defines a post processing step to limit the number of bones affecting a single vertex. */
#ifndef AI_LIMITBONEWEIGHTSPROCESS_H_INC
#define AI_LIMITBONEWEIGHTSPROCESS_H_INC

#include "Common/BaseProcess.h"

// Forward declarations
struct aiMesh;

class LimitBoneWeightsTest;

namespace Assimp {

// NOTE: If you change these limits, don't forget to change the
// corresponding values in all Assimp ports

// **********************************************************
// Java: ConfigProperty.java,
//  ConfigProperty.DEFAULT_BONE_WEIGHT_LIMIT
// **********************************************************

#if (!defined AI_LMW_MAX_WEIGHTS)
#   define AI_LMW_MAX_WEIGHTS   0x4
#endif // !! AI_LMW_MAX_WEIGHTS

// ---------------------------------------------------------------------------
/** This post processing step limits the number of bones affecting a vertex
* to a certain maximum value. If a vertex is affected by more than that number
* of bones, the bone weight with the least influence on this vertex are removed.
* The other weights on this bone are then renormalized to assure the sum weight
* to be 1.
*/
class ASSIMP_API LimitBoneWeightsProcess : public BaseProcess {
public:
    LimitBoneWeightsProcess();
    ~LimitBoneWeightsProcess();

    // -------------------------------------------------------------------
    /** Returns whether the processing step is present in the given flag.
    * @param pFlags The processing flags the importer was called with.
    *   A bitwise combination of #aiPostProcessSteps.
    * @return true if the process is present in this flag fields,
    *   false if not.
    */
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    /** Called prior to ExecuteOnScene().
    * The function is a request to the process to update its configuration
    * basing on the Importer's configuration property list.
    */
    void SetupProperties(const Importer* pImp);

    // -------------------------------------------------------------------
    /** Limits the bone weight count for all vertices in the given mesh.
    * @param pMesh The mesh to process.
    */
    void ProcessMesh( aiMesh* pMesh);

    // -------------------------------------------------------------------
    /** Executes the post processing step on the given imported data.
    * At the moment a process is not supposed to fail.
    * @param pScene The imported data to work at.
    */
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    /** Describes a bone weight on a vertex */
    struct Weight {
        unsigned int mBone; ///< Index of the bone
        float mWeight;      ///< Weight of that bone on this vertex
        Weight() AI_NO_EXCEPT
        : mBone(0)
        , mWeight(0.0f) {
            // empty
        }

        Weight( unsigned int pBone, float pWeight)
        : mBone(pBone)
        , mWeight(pWeight) {
            // empty
        }

        /** Comparison operator to sort bone weights by descending weight */
        bool operator < (const Weight& pWeight) const {
            return mWeight > pWeight.mWeight;
        }
    };

    /** Maximum number of bones influencing any single vertex. */
    unsigned int mMaxWeights;
};

} // end of namespace Assimp

#endif // AI_LIMITBONEWEIGHTSPROCESS_H_INC

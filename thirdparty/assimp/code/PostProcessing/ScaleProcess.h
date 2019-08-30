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
#ifndef SCALE_PROCESS_H_
#define SCALE_PROCESS_H_

#include "Common/BaseProcess.h"

struct aiNode;

#if (!defined AI_CONFIG_GLOBAL_SCALE_FACTOR_DEFAULT)
#   define AI_CONFIG_GLOBAL_SCALE_FACTOR_DEFAULT  1.0f
#endif // !! AI_DEBONE_THRESHOLD

namespace Assimp {

// ---------------------------------------------------------------------------
/** ScaleProcess: Class to rescale the whole model.
 * Now rescales animations, bones, and blend shapes properly.
 * Please note this will not write to 'scale' transform it will rewrite mesh 
 * and matrixes so that your scale values 
 * from your model package are preserved, so this is completely intentional
 * bugs should be reported as soon as they are found.
*/
class ASSIMP_API ScaleProcess : public BaseProcess {
public:
    /// The default class constructor.
    ScaleProcess();

    /// The class destructor.
    virtual ~ScaleProcess();

    /// Will set the scale manually.
    void setScale( ai_real scale );

    /// Returns the current scaling value.
    ai_real getScale() const;

    /// Overwritten, @see BaseProcess
    virtual bool IsActive( unsigned int pFlags ) const;

    /// Overwritten, @see BaseProcess
    virtual void SetupProperties( const Importer* pImp );

    /// Overwritten, @see BaseProcess
    virtual void Execute( aiScene* pScene );

private:
    void traverseNodes( aiNode *currentNode, unsigned int nested_node_id = 0 );
    void applyScaling( aiNode *currentNode );

private:
    ai_real mScale;
};

} // Namespace Assimp


#endif // SCALE_PROCESS_H_
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

#pragma once

#include "BaseProcess.h"

#include <string>

struct aiNode;

namespace Assimp {

/**
 *  Force embedding of textures (using the path = "*1" convention).
 *  If a texture's file does not exist at the specified path
 *  (due, for instance, to an absolute path generated on another system),
 *  it will check if a file with the same name exists at the root folder
 *  of the imported model. And if so, it uses that.
 */
class ASSIMP_API EmbedTexturesProcess : public BaseProcess {
public:
    /// The default class constructor.
    EmbedTexturesProcess();

    /// The class destructor.
    virtual ~EmbedTexturesProcess();

    /// Overwritten, @see BaseProcess
    virtual bool IsActive(unsigned int pFlags) const;

    /// Overwritten, @see BaseProcess
    virtual void SetupProperties(const Importer* pImp);

    /// Overwritten, @see BaseProcess
    virtual void Execute(aiScene* pScene);

private:
    // Resolve the path and add the file content to the scene as a texture.
    bool addTexture(aiScene* pScene, std::string path) const;

private:
    std::string mRootPath;
};

} // namespace Assimp

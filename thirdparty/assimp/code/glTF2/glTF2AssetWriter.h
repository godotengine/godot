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

/** @file glTFWriter.h
 * Declares a class to write gltf/glb files
 *
 * glTF Extensions Support:
 *   KHR_materials_pbrSpecularGlossiness: full
 *   KHR_materials_unlit: full
 */
#ifndef GLTF2ASSETWRITER_H_INC
#define GLTF2ASSETWRITER_H_INC

#ifndef ASSIMP_BUILD_NO_GLTF_IMPORTER

#include "glTF2Asset.h"

namespace glTF2
{

using rapidjson::MemoryPoolAllocator;

class AssetWriter
{
    template<class T>
    friend void WriteLazyDict(LazyDict<T>& d, AssetWriter& w);

private:

    void WriteBinaryData(IOStream* outfile, size_t sceneLength);

    void WriteMetadata();
    void WriteExtensionsUsed();

    template<class T>
    void WriteObjects(LazyDict<T>& d);

public:
    Document mDoc;
    Asset& mAsset;

    MemoryPoolAllocator<>& mAl;

    AssetWriter(Asset& asset);

    void WriteFile(const char* path);
    void WriteGLBFile(const char* path);
};

}

// Include the implementation of the methods
#include "glTF2AssetWriter.inl"

#endif // ASSIMP_BUILD_NO_GLTF_IMPORTER

#endif // GLTF2ASSETWRITER_H_INC

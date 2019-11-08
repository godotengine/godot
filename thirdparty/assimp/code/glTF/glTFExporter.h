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

/** @file GltfExporter.h
* Declares the exporter class to write a scene to a gltf/glb file
*/
#ifndef AI_GLTFEXPORTER_H_INC
#define AI_GLTFEXPORTER_H_INC

#ifndef ASSIMP_BUILD_NO_GLTF_EXPORTER

#include <assimp/types.h>
#include <assimp/material.h>

#include <sstream>
#include <vector>
#include <map>
#include <memory>

struct aiScene;
struct aiNode;
struct aiMaterial;

namespace glTF
{
    template<class T>
    class Ref;

    class Asset;
    struct TexProperty;
    struct Node;
}

namespace Assimp
{
    class IOSystem;
    class IOStream;
    class ExportProperties;

    // ------------------------------------------------------------------------------------------------
    /** Helper class to export a given scene to an glTF file. */
    // ------------------------------------------------------------------------------------------------
    class glTFExporter
    {
    public:
        /// Constructor for a specific scene to export
        glTFExporter(const char* filename, IOSystem* pIOSystem, const aiScene* pScene,
            const ExportProperties* pProperties, bool binary);

    private:

        const char* mFilename;
        IOSystem* mIOSystem;
        const aiScene* mScene;
        const ExportProperties* mProperties;

        std::map<std::string, unsigned int> mTexturesByPath;

        std::shared_ptr<glTF::Asset> mAsset;

        std::vector<unsigned char> mBodyData;

        void WriteBinaryData(IOStream* outfile, std::size_t sceneLength);

        void GetTexSampler(const aiMaterial* mat, glTF::TexProperty& prop);
        void GetMatColorOrTex(const aiMaterial* mat, glTF::TexProperty& prop, const char* propName, int type, int idx, aiTextureType tt);
        void ExportMetadata();
        void ExportMaterials();
        void ExportMeshes();
        unsigned int ExportNodeHierarchy(const aiNode* n);
        unsigned int ExportNode(const aiNode* node, glTF::Ref<glTF::Node>& parent);
        void ExportScene();
        void ExportAnimations();
    };

}

#endif // ASSIMP_BUILD_NO_GLTF_EXPORTER

#endif // AI_GLTFEXPORTER_H_INC

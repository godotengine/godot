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

#include "EmbedTexturesProcess.h"
#include <assimp/ParsingUtils.h>
#include "ProcessHelper.h"

#include <fstream>

using namespace Assimp;

EmbedTexturesProcess::EmbedTexturesProcess()
: BaseProcess() {
}

EmbedTexturesProcess::~EmbedTexturesProcess() {
}

bool EmbedTexturesProcess::IsActive(unsigned int pFlags) const {
    return (pFlags & aiProcess_EmbedTextures) != 0;
}

void EmbedTexturesProcess::SetupProperties(const Importer* pImp) {
    mRootPath = pImp->GetPropertyString("sourceFilePath");
    mRootPath = mRootPath.substr(0, mRootPath.find_last_of("\\/") + 1u);
}

void EmbedTexturesProcess::Execute(aiScene* pScene) {
    if (pScene == nullptr || pScene->mRootNode == nullptr) return;

    aiString path;

    uint32_t embeddedTexturesCount = 0u;

    for (auto matId = 0u; matId < pScene->mNumMaterials; ++matId) {
        auto material = pScene->mMaterials[matId];

        for (auto ttId = 1u; ttId < AI_TEXTURE_TYPE_MAX; ++ttId) {
            auto tt = static_cast<aiTextureType>(ttId);
            auto texturesCount = material->GetTextureCount(tt);

            for (auto texId = 0u; texId < texturesCount; ++texId) {
                material->GetTexture(tt, texId, &path);
                if (path.data[0] == '*') continue; // Already embedded

                // Indeed embed
                if (addTexture(pScene, path.data)) {
                    auto embeddedTextureId = pScene->mNumTextures - 1u;
                    ::ai_snprintf(path.data, 1024, "*%u", embeddedTextureId);
                    material->AddProperty(&path, AI_MATKEY_TEXTURE(tt, texId));
                    embeddedTexturesCount++;
                }
            }
        }
    }

    ASSIMP_LOG_INFO_F("EmbedTexturesProcess finished. Embedded ", embeddedTexturesCount, " textures." );
}

bool EmbedTexturesProcess::addTexture(aiScene* pScene, std::string path) const {
    std::streampos imageSize = 0;
    std::string    imagePath = path;

    // Test path directly
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    if ((imageSize = file.tellg()) == std::streampos(-1)) {
        ASSIMP_LOG_WARN_F("EmbedTexturesProcess: Cannot find image: ", imagePath, ". Will try to find it in root folder.");

        // Test path in root path
        imagePath = mRootPath + path;
        file.open(imagePath, std::ios::binary | std::ios::ate);
        if ((imageSize = file.tellg()) == std::streampos(-1)) {
            // Test path basename in root path
            imagePath = mRootPath + path.substr(path.find_last_of("\\/") + 1u);
            file.open(imagePath, std::ios::binary | std::ios::ate);
            if ((imageSize = file.tellg()) == std::streampos(-1)) {
                ASSIMP_LOG_ERROR_F("EmbedTexturesProcess: Unable to embed texture: ", path, ".");
                return false;
            }
        }
    }

    aiTexel* imageContent = new aiTexel[ 1ul + static_cast<unsigned long>( imageSize ) / sizeof(aiTexel)];
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(imageContent), imageSize);

    // Enlarging the textures table
    unsigned int textureId = pScene->mNumTextures++;
    auto oldTextures = pScene->mTextures;
    pScene->mTextures = new aiTexture*[pScene->mNumTextures];
    ::memmove(pScene->mTextures, oldTextures, sizeof(aiTexture*) * (pScene->mNumTextures - 1u));

    // Add the new texture
    auto pTexture = new aiTexture;
    pTexture->mHeight = 0; // Means that this is still compressed
    pTexture->mWidth = static_cast<uint32_t>(imageSize);
    pTexture->pcData = imageContent;

    auto extension = path.substr(path.find_last_of('.') + 1u);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (extension == "jpeg") {
        extension = "jpg";
    }

    size_t len = extension.size();
    if (len > HINTMAXTEXTURELEN -1 ) {
        len = HINTMAXTEXTURELEN - 1;
    }
    ::strncpy(pTexture->achFormatHint, extension.c_str(), len);
    pScene->mTextures[textureId] = pTexture;

    return true;
}

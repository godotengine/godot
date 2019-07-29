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
r
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

/** @file  FBXImporter.cpp
 *  @brief Implementation of the FBX importer.
 */

#ifndef ASSIMP_BUILD_NO_FBX_IMPORTER

#include "FBXImporter.h"

#include "FBXTokenizer.h"
#include "FBXParser.h"
#include "FBXUtil.h"
#include "FBXDocument.h"
#include "FBXConverter.h"

#include <assimp/StreamReader.h>
#include <assimp/MemoryIOWrapper.h>
#include <assimp/Importer.hpp>
#include <assimp/importerdesc.h>

namespace Assimp {

template<>
const char* LogFunctions<FBXImporter>::Prefix() {
    static auto prefix = "FBX: ";
    return prefix;
}

}

using namespace Assimp;
using namespace Assimp::Formatter;
using namespace Assimp::FBX;

namespace {

static const aiImporterDesc desc = {
    "Autodesk FBX Importer",
    "",
    "",
    "",
    aiImporterFlags_SupportTextFlavour,
    0,
    0,
    0,
    0,
    "fbx"
};
}

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by #Importer
FBXImporter::FBXImporter()
{
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
FBXImporter::~FBXImporter()
{
}

// ------------------------------------------------------------------------------------------------
// Returns whether the class can handle the format of the given file.
bool FBXImporter::CanRead( const std::string& pFile, IOSystem* pIOHandler, bool checkSig) const
{
    const std::string& extension = GetExtension(pFile);
    if (extension == std::string( desc.mFileExtensions ) ) {
        return true;
    }

    else if ((!extension.length() || checkSig) && pIOHandler)   {
        // at least ASCII-FBX files usually have a 'FBX' somewhere in their head
        const char* tokens[] = {"fbx"};
        return SearchFileHeaderForToken(pIOHandler,pFile,tokens,1);
    }
    return false;
}

// ------------------------------------------------------------------------------------------------
// List all extensions handled by this loader
const aiImporterDesc* FBXImporter::GetInfo () const
{
    return &desc;
}

// ------------------------------------------------------------------------------------------------
// Setup configuration properties for the loader
void FBXImporter::SetupProperties(const Importer* pImp)
{
    settings.readAllLayers = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_ALL_GEOMETRY_LAYERS, true);
    settings.readAllMaterials = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_ALL_MATERIALS, false);
    settings.readMaterials = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_MATERIALS, true);
    settings.readTextures = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_TEXTURES, true);
    settings.readCameras = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_CAMERAS, true);
    settings.readLights = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_LIGHTS, true);
    settings.readAnimations = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_ANIMATIONS, true);
    settings.strictMode = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_STRICT_MODE, false);
    settings.preservePivots = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, true);
    settings.optimizeEmptyAnimationCurves = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_OPTIMIZE_EMPTY_ANIMATION_CURVES, true);
    settings.useLegacyEmbeddedTextureNaming = pImp->GetPropertyBool(AI_CONFIG_IMPORT_FBX_EMBEDDED_TEXTURES_LEGACY_NAMING, false);
    settings.removeEmptyBones = pImp->GetPropertyBool(AI_CONFIG_IMPORT_REMOVE_EMPTY_BONES, true);
    settings.convertToMeters = pImp->GetPropertyBool(AI_CONFIG_FBX_CONVERT_TO_M, false);
}

// ------------------------------------------------------------------------------------------------
// Imports the given file into the given scene structure.
void FBXImporter::InternReadFile( const std::string& pFile, aiScene* pScene, IOSystem* pIOHandler)
{
    std::unique_ptr<IOStream> stream(pIOHandler->Open(pFile,"rb"));
    if (!stream) {
        ThrowException("Could not open file for reading");
    }

    // read entire file into memory - no streaming for this, fbx
    // files can grow large, but the assimp output data structure
    // then becomes very large, too. Assimp doesn't support
    // streaming for its output data structures so the net win with
    // streaming input data would be very low.
    std::vector<char> contents;
    contents.resize(stream->FileSize()+1);
    stream->Read( &*contents.begin(), 1, contents.size()-1 );
    contents[ contents.size() - 1 ] = 0;
    const char* const begin = &*contents.begin();

    // broadphase tokenizing pass in which we identify the core
    // syntax elements of FBX (brackets, commas, key:value mappings)
    TokenList tokens;
    try {

        bool is_binary = false;
        if (!strncmp(begin,"Kaydara FBX Binary",18)) {
            is_binary = true;
            TokenizeBinary(tokens,begin,contents.size());
        }
        else {
            Tokenize(tokens,begin);
        }

        // use this information to construct a very rudimentary
        // parse-tree representing the FBX scope structure
        Parser parser(tokens, is_binary);

        // take the raw parse-tree and convert it to a FBX DOM
        Document doc(parser,settings);

        FbxUnit unit(FbxUnit::cm);
        if (settings.convertToMeters) {
            unit = FbxUnit::m;
        }
        // convert the FBX DOM to aiScene
        ConvertToAssimpScene(pScene,doc, settings.removeEmptyBones, unit);

        std::for_each(tokens.begin(),tokens.end(),Util::delete_fun<Token>());
    }
    catch(std::exception&) {
        std::for_each(tokens.begin(),tokens.end(),Util::delete_fun<Token>());
        throw;
    }
}

#endif // !ASSIMP_BUILD_NO_FBX_IMPORTER

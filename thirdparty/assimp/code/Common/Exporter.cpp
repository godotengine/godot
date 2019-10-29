/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

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
---------------------------------------------------------------------------
*/

/** @file Exporter.cpp

Assimp export interface. While it's public interface bears many similarities
to the import interface (in fact, it is largely symmetric), the internal
implementations differs a lot. Exporters are considered stateless and are
simple callbacks which we maintain in a global list along with their
description strings.

Here we implement only the C++ interface (Assimp::Exporter).
*/

#ifndef ASSIMP_BUILD_NO_EXPORT

#include <assimp/BlobIOSystem.h>
#include <assimp/SceneCombiner.h>
#include <assimp/DefaultIOSystem.h>
#include <assimp/Exporter.hpp>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Exceptional.h>

#include "Common/DefaultProgressHandler.h"
#include "Common/BaseProcess.h"
#include "Common/ScenePrivate.h"
#include "PostProcessing/CalcTangentsProcess.h"
#include "PostProcessing/MakeVerboseFormat.h"
#include "PostProcessing/JoinVerticesProcess.h"
#include "PostProcessing/ConvertToLHProcess.h"
#include "PostProcessing/PretransformVertices.h"

#include <memory>

namespace Assimp {

// PostStepRegistry.cpp
void GetPostProcessingStepInstanceList(std::vector< BaseProcess* >& out);

// ------------------------------------------------------------------------------------------------
// Exporter worker function prototypes. Should not be necessary to #ifndef them, it's just a prototype
// do not use const, because some exporter need to convert the scene temporary
void ExportSceneCollada(const char*,IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneXFile(const char*,IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneStep(const char*,IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneObj(const char*,IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneObjNoMtl(const char*,IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneSTL(const char*,IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneSTLBinary(const char*,IOSystem*, const aiScene*, const ExportProperties*);
void ExportScenePly(const char*,IOSystem*, const aiScene*, const ExportProperties*);
void ExportScenePlyBinary(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportScene3DS(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneGLTF(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneGLB(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneGLTF2(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneGLB2(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneAssbin(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneAssxml(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneX3D(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneFBX(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportSceneFBXA(const char*, IOSystem*, const aiScene*, const ExportProperties*);
void ExportScene3MF( const char*, IOSystem*, const aiScene*, const ExportProperties* );
void ExportAssimp2Json(const char* , IOSystem*, const aiScene* , const Assimp::ExportProperties*);

// ------------------------------------------------------------------------------------------------
// global array of all export formats which Assimp supports in its current build
Exporter::ExportFormatEntry gExporters[] =
{
#ifndef ASSIMP_BUILD_NO_COLLADA_EXPORTER
    Exporter::ExportFormatEntry( "collada", "COLLADA - Digital Asset Exchange Schema", "dae", &ExportSceneCollada ),
#endif

#ifndef ASSIMP_BUILD_NO_X_EXPORTER
    Exporter::ExportFormatEntry( "x", "X Files", "x", &ExportSceneXFile,
        aiProcess_MakeLeftHanded | aiProcess_FlipWindingOrder | aiProcess_FlipUVs ),
#endif

#ifndef ASSIMP_BUILD_NO_STEP_EXPORTER
    Exporter::ExportFormatEntry( "stp", "Step Files", "stp", &ExportSceneStep, 0 ),
#endif

#ifndef ASSIMP_BUILD_NO_OBJ_EXPORTER
    Exporter::ExportFormatEntry( "obj", "Wavefront OBJ format", "obj", &ExportSceneObj,
        aiProcess_GenSmoothNormals /*| aiProcess_PreTransformVertices */ ),
    Exporter::ExportFormatEntry( "objnomtl", "Wavefront OBJ format without material file", "obj", &ExportSceneObjNoMtl,
        aiProcess_GenSmoothNormals /*| aiProcess_PreTransformVertices */ ),
#endif

#ifndef ASSIMP_BUILD_NO_STL_EXPORTER
    Exporter::ExportFormatEntry( "stl", "Stereolithography", "stl" , &ExportSceneSTL,
        aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_PreTransformVertices
    ),
    Exporter::ExportFormatEntry( "stlb", "Stereolithography (binary)", "stl" , &ExportSceneSTLBinary,
        aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_PreTransformVertices
    ),
#endif

#ifndef ASSIMP_BUILD_NO_PLY_EXPORTER
    Exporter::ExportFormatEntry( "ply", "Stanford Polygon Library", "ply" , &ExportScenePly,
        aiProcess_PreTransformVertices
    ),
    Exporter::ExportFormatEntry( "plyb", "Stanford Polygon Library (binary)", "ply", &ExportScenePlyBinary,
        aiProcess_PreTransformVertices
    ),
#endif

#ifndef ASSIMP_BUILD_NO_3DS_EXPORTER
    Exporter::ExportFormatEntry( "3ds", "Autodesk 3DS (legacy)", "3ds" , &ExportScene3DS,
        aiProcess_Triangulate | aiProcess_SortByPType | aiProcess_JoinIdenticalVertices ),
#endif

#ifndef ASSIMP_BUILD_NO_GLTF_EXPORTER
    Exporter::ExportFormatEntry( "gltf2", "GL Transmission Format v. 2", "gltf", &ExportSceneGLTF2,
        aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_SortByPType ),
    Exporter::ExportFormatEntry( "glb2", "GL Transmission Format v. 2 (binary)", "glb", &ExportSceneGLB2,
        aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_SortByPType ),
    Exporter::ExportFormatEntry( "gltf", "GL Transmission Format", "gltf", &ExportSceneGLTF,
        aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_SortByPType ),
    Exporter::ExportFormatEntry( "glb", "GL Transmission Format (binary)", "glb", &ExportSceneGLB,
        aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_SortByPType ),
#endif

#ifndef ASSIMP_BUILD_NO_ASSBIN_EXPORTER
    Exporter::ExportFormatEntry( "assbin", "Assimp Binary File", "assbin" , &ExportSceneAssbin, 0 ),
#endif

#ifndef ASSIMP_BUILD_NO_ASSXML_EXPORTER
    Exporter::ExportFormatEntry( "assxml", "Assimp XML Document", "assxml" , &ExportSceneAssxml, 0 ),
#endif

#ifndef ASSIMP_BUILD_NO_X3D_EXPORTER
    Exporter::ExportFormatEntry( "x3d", "Extensible 3D", "x3d" , &ExportSceneX3D, 0 ),
#endif

#ifndef ASSIMP_BUILD_NO_FBX_EXPORTER
    Exporter::ExportFormatEntry( "fbx", "Autodesk FBX (binary)", "fbx", &ExportSceneFBX, 0 ),
    Exporter::ExportFormatEntry( "fbxa", "Autodesk FBX (ascii)", "fbx", &ExportSceneFBXA, 0 ),
#endif

#ifndef ASSIMP_BUILD_NO_3MF_EXPORTER
    Exporter::ExportFormatEntry( "3mf", "The 3MF-File-Format", "3mf", &ExportScene3MF, 0 ),
#endif

#ifndef ASSIMP_BUILD_NO_ASSJSON_EXPORTER
    Exporter::ExportFormatEntry( "assjson", "Assimp JSON Document", "json", &ExportAssimp2Json, 0)
#endif
};

#define ASSIMP_NUM_EXPORTERS (sizeof(gExporters)/sizeof(gExporters[0]))


class ExporterPimpl {
public:
    ExporterPimpl()
    : blob()
    , mIOSystem(new Assimp::DefaultIOSystem())
    , mIsDefaultIOHandler(true)
    , mProgressHandler( nullptr )
    , mIsDefaultProgressHandler( true )
    , mPostProcessingSteps()
    , mError()
    , mExporters() {
        GetPostProcessingStepInstanceList(mPostProcessingSteps);

        // grab all built-in exporters
        if ( 0 != ( ASSIMP_NUM_EXPORTERS ) ) {
            mExporters.resize( ASSIMP_NUM_EXPORTERS );
            std::copy( gExporters, gExporters + ASSIMP_NUM_EXPORTERS, mExporters.begin() );
        }
    }

    ~ExporterPimpl() {
        delete blob;

        // Delete all post-processing plug-ins
        for( unsigned int a = 0; a < mPostProcessingSteps.size(); a++) {
            delete mPostProcessingSteps[a];
        }
        delete mProgressHandler;
    }

public:
    aiExportDataBlob* blob;
    std::shared_ptr< Assimp::IOSystem > mIOSystem;
    bool mIsDefaultIOHandler;

    /** The progress handler */
    ProgressHandler *mProgressHandler;
    bool mIsDefaultProgressHandler;

    /** Post processing steps we can apply at the imported data. */
    std::vector< BaseProcess* > mPostProcessingSteps;

    /** Last fatal export error */
    std::string mError;

    /** Exporters, this includes those registered using #Assimp::Exporter::RegisterExporter */
    std::vector<Exporter::ExportFormatEntry> mExporters;
};

} // end of namespace Assimp

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
Exporter :: Exporter()
: pimpl(new ExporterPimpl()) {
    pimpl->mProgressHandler = new DefaultProgressHandler();
}

// ------------------------------------------------------------------------------------------------
Exporter::~Exporter() {
    FreeBlob();
    delete pimpl;
}

// ------------------------------------------------------------------------------------------------
void Exporter::SetIOHandler( IOSystem* pIOHandler) {
    pimpl->mIsDefaultIOHandler = !pIOHandler;
    pimpl->mIOSystem.reset(pIOHandler);
}

// ------------------------------------------------------------------------------------------------
IOSystem* Exporter::GetIOHandler() const {
    return pimpl->mIOSystem.get();
}

// ------------------------------------------------------------------------------------------------
bool Exporter::IsDefaultIOHandler() const {
    return pimpl->mIsDefaultIOHandler;
}

// ------------------------------------------------------------------------------------------------
void Exporter::SetProgressHandler(ProgressHandler* pHandler) {
    ai_assert(nullptr != pimpl);

    if ( nullptr == pHandler) {
        // Release pointer in the possession of the caller
        pimpl->mProgressHandler = new DefaultProgressHandler();
        pimpl->mIsDefaultProgressHandler = true;
        return;
    }

    if (pimpl->mProgressHandler == pHandler) {
        return;
    }

    delete pimpl->mProgressHandler;
    pimpl->mProgressHandler = pHandler;
    pimpl->mIsDefaultProgressHandler = false;
}

// ------------------------------------------------------------------------------------------------
const aiExportDataBlob* Exporter::ExportToBlob( const aiScene* pScene, const char* pFormatId,
                                                unsigned int pPreprocessing, const ExportProperties* pProperties) {
    if (pimpl->blob) {
        delete pimpl->blob;
        pimpl->blob = nullptr;
    }

    std::shared_ptr<IOSystem> old = pimpl->mIOSystem;
    BlobIOSystem* blobio = new BlobIOSystem();
    pimpl->mIOSystem = std::shared_ptr<IOSystem>( blobio );

    if (AI_SUCCESS != Export(pScene,pFormatId,blobio->GetMagicFileName(), pPreprocessing, pProperties)) {
        pimpl->mIOSystem = old;
        return nullptr;
    }

    pimpl->blob = blobio->GetBlobChain();
    pimpl->mIOSystem = old;

    return pimpl->blob;
}

// ------------------------------------------------------------------------------------------------
bool IsVerboseFormat(const aiMesh* mesh) {
    // avoid slow vector<bool> specialization
    std::vector<unsigned int> seen(mesh->mNumVertices,0);
    for(unsigned int i = 0; i < mesh->mNumFaces; ++i) {
        const aiFace& f = mesh->mFaces[i];
        for(unsigned int j = 0; j < f.mNumIndices; ++j) {
            if(++seen[f.mIndices[j]] == 2) {
                // found a duplicate index
                return false;
            }
        }
    }

    return true;
}

// ------------------------------------------------------------------------------------------------
bool IsVerboseFormat(const aiScene* pScene) {
    for(unsigned int i = 0; i < pScene->mNumMeshes; ++i) {
        if(!IsVerboseFormat(pScene->mMeshes[i])) {
            return false;
        }
    }

    return true;
}

// ------------------------------------------------------------------------------------------------
aiReturn Exporter::Export( const aiScene* pScene, const char* pFormatId, const char* pPath,
        unsigned int pPreprocessing, const ExportProperties* pProperties) {
    ASSIMP_BEGIN_EXCEPTION_REGION();

    // when they create scenes from scratch, users will likely create them not in verbose
    // format. They will likely not be aware that there is a flag in the scene to indicate
    // this, however. To avoid surprises and bug reports, we check for duplicates in
    // meshes upfront.
    const bool is_verbose_format = !(pScene->mFlags & AI_SCENE_FLAGS_NON_VERBOSE_FORMAT) || IsVerboseFormat(pScene);

    pimpl->mProgressHandler->UpdateFileWrite(0, 4);

    pimpl->mError = "";
    for (size_t i = 0; i < pimpl->mExporters.size(); ++i) {
        const Exporter::ExportFormatEntry& exp = pimpl->mExporters[i];
        if (!strcmp(exp.mDescription.id,pFormatId)) {
            try {
                // Always create a full copy of the scene. We might optimize this one day,
                // but for now it is the most pragmatic way.
                aiScene* scenecopy_tmp = nullptr;
                SceneCombiner::CopyScene(&scenecopy_tmp,pScene);

                pimpl->mProgressHandler->UpdateFileWrite(1, 4);

                std::unique_ptr<aiScene> scenecopy(scenecopy_tmp);
                const ScenePrivateData* const priv = ScenePriv(pScene);

                // steps that are not idempotent, i.e. we might need to run them again, usually to get back to the
                // original state before the step was applied first. When checking which steps we don't need
                // to run, those are excluded.
                const unsigned int nonIdempotentSteps = aiProcess_FlipWindingOrder | aiProcess_FlipUVs | aiProcess_MakeLeftHanded;

                // Erase all pp steps that were already applied to this scene
                const unsigned int pp = (exp.mEnforcePP | pPreprocessing) & ~(priv && !priv->mIsCopy
                    ? (priv->mPPStepsApplied & ~nonIdempotentSteps)
                    : 0u);

                // If no extra post-processing was specified, and we obtained this scene from an
                // Assimp importer, apply the reverse steps automatically.
                // TODO: either drop this, or document it. Otherwise it is just a bad surprise.
                //if (!pPreprocessing && priv) {
                //  pp |= (nonIdempotentSteps & priv->mPPStepsApplied);
                //}

                // If the input scene is not in verbose format, but there is at least post-processing step that relies on it,
                // we need to run the MakeVerboseFormat step first.
                bool must_join_again = false;
                if (!is_verbose_format) {
                    bool verbosify = false;
                    for( unsigned int a = 0; a < pimpl->mPostProcessingSteps.size(); a++) {
                        BaseProcess* const p = pimpl->mPostProcessingSteps[a];

                        if (p->IsActive(pp) && p->RequireVerboseFormat()) {
                            verbosify = true;
                            break;
                        }
                    }

                    if (verbosify || (exp.mEnforcePP & aiProcess_JoinIdenticalVertices)) {
                        ASSIMP_LOG_DEBUG("export: Scene data not in verbose format, applying MakeVerboseFormat step first");

                        MakeVerboseFormatProcess proc;
                        proc.Execute(scenecopy.get());

                        if(!(exp.mEnforcePP & aiProcess_JoinIdenticalVertices)) {
                            must_join_again = true;
                        }
                    }
                }

                pimpl->mProgressHandler->UpdateFileWrite(2, 4);

                if (pp) {
                    // the three 'conversion' steps need to be executed first because all other steps rely on the standard data layout
                    {
                        FlipWindingOrderProcess step;
                        if (step.IsActive(pp)) {
                            step.Execute(scenecopy.get());
                        }
                    }

                    {
                        FlipUVsProcess step;
                        if (step.IsActive(pp)) {
                            step.Execute(scenecopy.get());
                        }
                    }

                    {
                        MakeLeftHandedProcess step;
                        if (step.IsActive(pp)) {
                            step.Execute(scenecopy.get());
                        }
                    }

                    bool exportPointCloud(false);
                    if (nullptr != pProperties) {
                        exportPointCloud = pProperties->GetPropertyBool(AI_CONFIG_EXPORT_POINT_CLOUDS);
                    }

                    // dispatch other processes
                    for( unsigned int a = 0; a < pimpl->mPostProcessingSteps.size(); a++) {
                        BaseProcess* const p = pimpl->mPostProcessingSteps[a];

                        if (p->IsActive(pp)
                            && !dynamic_cast<FlipUVsProcess*>(p)
                            && !dynamic_cast<FlipWindingOrderProcess*>(p)
                            && !dynamic_cast<MakeLeftHandedProcess*>(p)) {
                            if (dynamic_cast<PretransformVertices*>(p) && exportPointCloud) {
                                continue;
                            }
                            p->Execute(scenecopy.get());
                        }
                    }
                    ScenePrivateData* const privOut = ScenePriv(scenecopy.get());
                    ai_assert(nullptr != privOut);

                    privOut->mPPStepsApplied |= pp;
                }

                pimpl->mProgressHandler->UpdateFileWrite(3, 4);

                if(must_join_again) {
                    JoinVerticesProcess proc;
                    proc.Execute(scenecopy.get());
                }

                ExportProperties emptyProperties;  // Never pass NULL ExportProperties so Exporters don't have to worry.
                exp.mExportFunction(pPath,pimpl->mIOSystem.get(),scenecopy.get(), pProperties ? pProperties : &emptyProperties);

                pimpl->mProgressHandler->UpdateFileWrite(4, 4);
            } catch (DeadlyExportError& err) {
                pimpl->mError = err.what();
                return AI_FAILURE;
            }
            return AI_SUCCESS;
        }
    }

    pimpl->mError = std::string("Found no exporter to handle this file format: ") + pFormatId;
    ASSIMP_END_EXCEPTION_REGION(aiReturn);

    return AI_FAILURE;
}

// ------------------------------------------------------------------------------------------------
const char* Exporter::GetErrorString() const {
    return pimpl->mError.c_str();
}

// ------------------------------------------------------------------------------------------------
void Exporter::FreeBlob() {
    delete pimpl->blob;
    pimpl->blob = nullptr;

    pimpl->mError = "";
}

// ------------------------------------------------------------------------------------------------
const aiExportDataBlob* Exporter::GetBlob() const {
    return pimpl->blob;
}

// ------------------------------------------------------------------------------------------------
const aiExportDataBlob* Exporter::GetOrphanedBlob() const {
    const aiExportDataBlob* tmp = pimpl->blob;
    pimpl->blob = nullptr;
    return tmp;
}

// ------------------------------------------------------------------------------------------------
size_t Exporter::GetExportFormatCount() const {
    return pimpl->mExporters.size();
}

// ------------------------------------------------------------------------------------------------
const aiExportFormatDesc* Exporter::GetExportFormatDescription( size_t index ) const {
    if (index >= GetExportFormatCount()) {
        return nullptr;
    }

    // Return from static storage if the requested index is built-in.
    if (index < sizeof(gExporters) / sizeof(gExporters[0])) {
        return &gExporters[index].mDescription;
    }

    return &pimpl->mExporters[index].mDescription;
}

// ------------------------------------------------------------------------------------------------
aiReturn Exporter::RegisterExporter(const ExportFormatEntry& desc) {
    for(const ExportFormatEntry& e : pimpl->mExporters) {
        if (!strcmp(e.mDescription.id,desc.mDescription.id)) {
            return aiReturn_FAILURE;
        }
    }

    pimpl->mExporters.push_back(desc);
    return aiReturn_SUCCESS;
}

// ------------------------------------------------------------------------------------------------
void Exporter::UnregisterExporter(const char* id) {
    for(std::vector<ExportFormatEntry>::iterator it = pimpl->mExporters.begin();
            it != pimpl->mExporters.end(); ++it) {
        if (!strcmp((*it).mDescription.id,id)) {
            pimpl->mExporters.erase(it);
            break;
        }
    }
}

// ------------------------------------------------------------------------------------------------
ExportProperties::ExportProperties() {
    // empty
}

// ------------------------------------------------------------------------------------------------
ExportProperties::ExportProperties(const ExportProperties &other)
: mIntProperties(other.mIntProperties)
, mFloatProperties(other.mFloatProperties)
, mStringProperties(other.mStringProperties)
, mMatrixProperties(other.mMatrixProperties) {
    // empty
}

// ------------------------------------------------------------------------------------------------
// Set a configuration property
bool ExportProperties::SetPropertyInteger(const char* szName, int iValue) {
    return SetGenericProperty<int>(mIntProperties, szName,iValue);
}

// ------------------------------------------------------------------------------------------------
// Set a configuration property
bool ExportProperties::SetPropertyFloat(const char* szName, ai_real iValue) {
    return SetGenericProperty<ai_real>(mFloatProperties, szName,iValue);
}

// ------------------------------------------------------------------------------------------------
// Set a configuration property
bool ExportProperties::SetPropertyString(const char* szName, const std::string& value) {
    return SetGenericProperty<std::string>(mStringProperties, szName,value);
}

// ------------------------------------------------------------------------------------------------
// Set a configuration property
bool ExportProperties::SetPropertyMatrix(const char* szName, const aiMatrix4x4& value) {
    return SetGenericProperty<aiMatrix4x4>(mMatrixProperties, szName,value);
}

// ------------------------------------------------------------------------------------------------
// Get a configuration property
int ExportProperties::GetPropertyInteger(const char* szName, int iErrorReturn /*= 0xffffffff*/) const {
    return GetGenericProperty<int>(mIntProperties,szName,iErrorReturn);
}

// ------------------------------------------------------------------------------------------------
// Get a configuration property
ai_real ExportProperties::GetPropertyFloat(const char* szName, ai_real iErrorReturn /*= 10e10*/) const {
    return GetGenericProperty<ai_real>(mFloatProperties,szName,iErrorReturn);
}

// ------------------------------------------------------------------------------------------------
// Get a configuration property
const std::string ExportProperties::GetPropertyString(const char* szName,
        const std::string& iErrorReturn /*= ""*/) const {
    return GetGenericProperty<std::string>(mStringProperties,szName,iErrorReturn);
}

// ------------------------------------------------------------------------------------------------
// Has a configuration property
const aiMatrix4x4 ExportProperties::GetPropertyMatrix(const char* szName,
        const aiMatrix4x4& iErrorReturn /*= aiMatrix4x4()*/) const {
    return GetGenericProperty<aiMatrix4x4>(mMatrixProperties,szName,iErrorReturn);
}

// ------------------------------------------------------------------------------------------------
// Has a configuration property
bool ExportProperties::HasPropertyInteger(const char* szName) const {
    return HasGenericProperty<int>(mIntProperties, szName);
}

// ------------------------------------------------------------------------------------------------
// Has a configuration property
bool ExportProperties::HasPropertyBool(const char* szName) const {
    return HasGenericProperty<int>(mIntProperties, szName);
}

// ------------------------------------------------------------------------------------------------
// Has a configuration property
bool ExportProperties::HasPropertyFloat(const char* szName) const {
    return HasGenericProperty<ai_real>(mFloatProperties, szName);
}

// ------------------------------------------------------------------------------------------------
// Has a configuration property
bool ExportProperties::HasPropertyString(const char* szName) const {
    return HasGenericProperty<std::string>(mStringProperties, szName);
}

// ------------------------------------------------------------------------------------------------
// Has a configuration property
bool ExportProperties::HasPropertyMatrix(const char* szName) const {
    return HasGenericProperty<aiMatrix4x4>(mMatrixProperties, szName);
}


#endif // !ASSIMP_BUILD_NO_EXPORT

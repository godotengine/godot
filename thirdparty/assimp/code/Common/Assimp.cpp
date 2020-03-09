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
/** @file  Assimp.cpp
 *  @brief Implementation of the Plain-C API
 */

#include <assimp/cimport.h>
#include <assimp/LogStream.hpp>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/importerdesc.h>
#include <assimp/scene.h>
#include <assimp/GenericProperty.h>
#include <assimp/Exceptional.h>
#include <assimp/BaseImporter.h>

#include "CApi/CInterfaceIOWrapper.h"
#include "Importer.h"
#include "ScenePrivate.h"

#include <list>

// ------------------------------------------------------------------------------------------------
#ifndef ASSIMP_BUILD_SINGLETHREADED
#   include <thread>
#   include <mutex>
#endif
// ------------------------------------------------------------------------------------------------
using namespace Assimp;

namespace Assimp {
    // underlying structure for aiPropertyStore
    typedef BatchLoader::PropertyMap PropertyMap;

    /** Stores the LogStream objects for all active C log streams */
    struct mpred {
        bool operator  () (const aiLogStream& s0, const aiLogStream& s1) const  {
            return s0.callback<s1.callback&&s0.user<s1.user;
        }
    };
    typedef std::map<aiLogStream, Assimp::LogStream*, mpred> LogStreamMap;

    /** Stores the LogStream objects allocated by #aiGetPredefinedLogStream */
    typedef std::list<Assimp::LogStream*> PredefLogStreamMap;

    /** Local storage of all active log streams */
    static LogStreamMap gActiveLogStreams;

    /** Local storage of LogStreams allocated by #aiGetPredefinedLogStream */
    static PredefLogStreamMap gPredefinedStreams;

    /** Error message of the last failed import process */
    static std::string gLastErrorString;

    /** Verbose logging active or not? */
    static aiBool gVerboseLogging = false;

    /** will return all registered importers. */
    void GetImporterInstanceList(std::vector< BaseImporter* >& out);

    /** will delete all registered importers. */
    void DeleteImporterInstanceList(std::vector< BaseImporter* >& out);
} // namespace assimp


#ifndef ASSIMP_BUILD_SINGLETHREADED
/** Global mutex to manage the access to the log-stream map */
static std::mutex gLogStreamMutex;
#endif

// ------------------------------------------------------------------------------------------------
// Custom LogStream implementation for the C-API
class LogToCallbackRedirector : public LogStream {
public:
    explicit LogToCallbackRedirector(const aiLogStream& s)
    : stream (s)    {
        ai_assert(NULL != s.callback);
    }

    ~LogToCallbackRedirector()  {
#ifndef ASSIMP_BUILD_SINGLETHREADED
        std::lock_guard<std::mutex> lock(gLogStreamMutex);
#endif
        // (HACK) Check whether the 'stream.user' pointer points to a
        // custom LogStream allocated by #aiGetPredefinedLogStream.
        // In this case, we need to delete it, too. Of course, this
        // might cause strange problems, but the chance is quite low.

        PredefLogStreamMap::iterator it = std::find(gPredefinedStreams.begin(),
            gPredefinedStreams.end(), (Assimp::LogStream*)stream.user);

        if (it != gPredefinedStreams.end()) {
            delete *it;
            gPredefinedStreams.erase(it);
        }
    }

    /** @copydoc LogStream::write */
    void write(const char* message) {
        stream.callback(message,stream.user);
    }

private:
    aiLogStream stream;
};

// ------------------------------------------------------------------------------------------------
void ReportSceneNotFoundError() {
    ASSIMP_LOG_ERROR("Unable to find the Assimp::Importer for this aiScene. "
        "The C-API does not accept scenes produced by the C++ API and vice versa");

    ai_assert(false);
}

// ------------------------------------------------------------------------------------------------
// Reads the given file and returns its content.
const aiScene* aiImportFile( const char* pFile, unsigned int pFlags) {
    return aiImportFileEx(pFile,pFlags,NULL);
}

// ------------------------------------------------------------------------------------------------
const aiScene* aiImportFileEx( const char* pFile, unsigned int pFlags,  aiFileIO* pFS) {
    return aiImportFileExWithProperties(pFile, pFlags, pFS, NULL);
}

// ------------------------------------------------------------------------------------------------
const aiScene* aiImportFileExWithProperties( const char* pFile, unsigned int pFlags, 
        aiFileIO* pFS, const aiPropertyStore* props) {
    ai_assert(NULL != pFile);

    const aiScene* scene = NULL;
    ASSIMP_BEGIN_EXCEPTION_REGION();

    // create an Importer for this file
    Assimp::Importer* imp = new Assimp::Importer();

    // copy properties
    if(props) {
        const PropertyMap* pp = reinterpret_cast<const PropertyMap*>(props);
        ImporterPimpl* pimpl = imp->Pimpl();
        pimpl->mIntProperties = pp->ints;
        pimpl->mFloatProperties = pp->floats;
        pimpl->mStringProperties = pp->strings;
        pimpl->mMatrixProperties = pp->matrices;
    }
    // setup a custom IO system if necessary
    if (pFS) {
        imp->SetIOHandler( new CIOSystemWrapper (pFS) );
    }

    // and have it read the file
    scene = imp->ReadFile( pFile, pFlags);

    // if succeeded, store the importer in the scene and keep it alive
    if( scene)  {
        ScenePrivateData* priv = const_cast<ScenePrivateData*>( ScenePriv(scene) );
        priv->mOrigImporter = imp;
    } else {
        // if failed, extract error code and destroy the import
        gLastErrorString = imp->GetErrorString();
        delete imp;
    }

    // return imported data. If the import failed the pointer is NULL anyways
    ASSIMP_END_EXCEPTION_REGION(const aiScene*);
    
    return scene;
}

// ------------------------------------------------------------------------------------------------
const aiScene* aiImportFileFromMemory(
    const char* pBuffer,
    unsigned int pLength,
    unsigned int pFlags,
    const char* pHint)
{
    return aiImportFileFromMemoryWithProperties(pBuffer, pLength, pFlags, pHint, NULL);
}

// ------------------------------------------------------------------------------------------------
const aiScene* aiImportFileFromMemoryWithProperties(
    const char* pBuffer,
    unsigned int pLength,
    unsigned int pFlags,
    const char* pHint,
    const aiPropertyStore* props)
{
    ai_assert( NULL != pBuffer );
    ai_assert( 0 != pLength );

    const aiScene* scene = NULL;
    ASSIMP_BEGIN_EXCEPTION_REGION();

    // create an Importer for this file
    Assimp::Importer* imp = new Assimp::Importer();

    // copy properties
    if(props) {
        const PropertyMap* pp = reinterpret_cast<const PropertyMap*>(props);
        ImporterPimpl* pimpl = imp->Pimpl();
        pimpl->mIntProperties = pp->ints;
        pimpl->mFloatProperties = pp->floats;
        pimpl->mStringProperties = pp->strings;
        pimpl->mMatrixProperties = pp->matrices;
    }

    // and have it read the file from the memory buffer
    scene = imp->ReadFileFromMemory( pBuffer, pLength, pFlags,pHint);

    // if succeeded, store the importer in the scene and keep it alive
    if( scene)  {
         ScenePrivateData* priv = const_cast<ScenePrivateData*>( ScenePriv(scene) );
         priv->mOrigImporter = imp;
    }
    else    {
        // if failed, extract error code and destroy the import
        gLastErrorString = imp->GetErrorString();
        delete imp;
    }
    // return imported data. If the import failed the pointer is NULL anyways
    ASSIMP_END_EXCEPTION_REGION(const aiScene*);
    return scene;
}

// ------------------------------------------------------------------------------------------------
// Releases all resources associated with the given import process.
void aiReleaseImport( const aiScene* pScene)
{
    if (!pScene) {
        return;
    }

    ASSIMP_BEGIN_EXCEPTION_REGION();

    // find the importer associated with this data
    const ScenePrivateData* priv = ScenePriv(pScene);
    if( !priv || !priv->mOrigImporter)  {
        delete pScene;
    }
    else {
        // deleting the Importer also deletes the scene
        // Note: the reason that this is not written as 'delete priv->mOrigImporter'
        // is a suspected bug in gcc 4.4+ (http://gcc.gnu.org/bugzilla/show_bug.cgi?id=52339)
        Importer* importer = priv->mOrigImporter;
        delete importer;
    }

    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API const aiScene* aiApplyPostProcessing(const aiScene* pScene,
    unsigned int pFlags)
{
    const aiScene* sc = NULL;


    ASSIMP_BEGIN_EXCEPTION_REGION();

    // find the importer associated with this data
    const ScenePrivateData* priv = ScenePriv(pScene);
    if( !priv || !priv->mOrigImporter)  {
        ReportSceneNotFoundError();
        return NULL;
    }

    sc = priv->mOrigImporter->ApplyPostProcessing(pFlags);

    if (!sc) {
        aiReleaseImport(pScene);
        return NULL;
    }

    ASSIMP_END_EXCEPTION_REGION(const aiScene*);
    return sc;
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API const aiScene *aiApplyCustomizedPostProcessing( const aiScene *scene,
                                                           BaseProcess* process,
                                                           bool requestValidation ) {
    const aiScene* sc( NULL );

    ASSIMP_BEGIN_EXCEPTION_REGION();

    // find the importer associated with this data
    const ScenePrivateData* priv = ScenePriv( scene );
    if ( NULL == priv || NULL == priv->mOrigImporter ) {
        ReportSceneNotFoundError();
        return NULL;
    }

    sc = priv->mOrigImporter->ApplyCustomizedPostProcessing( process, requestValidation );

    if ( !sc ) {
        aiReleaseImport( scene );
        return NULL;
    }

    ASSIMP_END_EXCEPTION_REGION( const aiScene* );

    return sc;
}

// ------------------------------------------------------------------------------------------------
void CallbackToLogRedirector (const char* msg, char* dt)
{
    ai_assert( NULL != msg );
    ai_assert( NULL != dt );
    LogStream* s = (LogStream*)dt;

    s->write(msg);
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API aiLogStream aiGetPredefinedLogStream(aiDefaultLogStream pStream,const char* file)
{
    aiLogStream sout;

    ASSIMP_BEGIN_EXCEPTION_REGION();
    LogStream* stream = LogStream::createDefaultStream(pStream,file);
    if (!stream) {
        sout.callback = NULL;
        sout.user = NULL;
    }
    else {
        sout.callback = &CallbackToLogRedirector;
        sout.user = (char*)stream;
    }
    gPredefinedStreams.push_back(stream);
    ASSIMP_END_EXCEPTION_REGION(aiLogStream);
    return sout;
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiAttachLogStream( const aiLogStream* stream )
{
    ASSIMP_BEGIN_EXCEPTION_REGION();

#ifndef ASSIMP_BUILD_SINGLETHREADED
    std::lock_guard<std::mutex> lock(gLogStreamMutex);
#endif

    LogStream* lg = new LogToCallbackRedirector(*stream);
    gActiveLogStreams[*stream] = lg;

    if (DefaultLogger::isNullLogger()) {
        DefaultLogger::create(NULL,(gVerboseLogging == AI_TRUE ? Logger::VERBOSE : Logger::NORMAL));
    }
    DefaultLogger::get()->attachStream(lg);
    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API aiReturn aiDetachLogStream( const aiLogStream* stream)
{
    ASSIMP_BEGIN_EXCEPTION_REGION();

#ifndef ASSIMP_BUILD_SINGLETHREADED
    std::lock_guard<std::mutex> lock(gLogStreamMutex);
#endif
    // find the log-stream associated with this data
    LogStreamMap::iterator it = gActiveLogStreams.find( *stream);
    // it should be there... else the user is playing fools with us
    if( it == gActiveLogStreams.end())  {
        return AI_FAILURE;
    }
    DefaultLogger::get()->detatchStream( it->second );
    delete it->second;

    gActiveLogStreams.erase( it);

    if (gActiveLogStreams.empty()) {
        DefaultLogger::kill();
    }
    ASSIMP_END_EXCEPTION_REGION(aiReturn);
    return AI_SUCCESS;
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiDetachAllLogStreams(void)
{
    ASSIMP_BEGIN_EXCEPTION_REGION();
#ifndef ASSIMP_BUILD_SINGLETHREADED
    std::lock_guard<std::mutex> lock(gLogStreamMutex);
#endif
    Logger *logger( DefaultLogger::get() );
    if ( NULL == logger ) {
        return;
    }

    for (LogStreamMap::iterator it = gActiveLogStreams.begin(); it != gActiveLogStreams.end(); ++it) {
        logger->detatchStream( it->second );
        delete it->second;
    }
    gActiveLogStreams.clear();
    DefaultLogger::kill();

    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiEnableVerboseLogging(aiBool d)
{
    if (!DefaultLogger::isNullLogger()) {
        DefaultLogger::get()->setLogSeverity((d == AI_TRUE ? Logger::VERBOSE : Logger::NORMAL));
    }
    gVerboseLogging = d;
}

// ------------------------------------------------------------------------------------------------
// Returns the error text of the last failed import process.
const char* aiGetErrorString()
{
    return gLastErrorString.c_str();
}

// -----------------------------------------------------------------------------------------------
// Return the description of a importer given its index
const aiImporterDesc* aiGetImportFormatDescription( size_t pIndex)
{
    return Importer().GetImporterInfo(pIndex);
}

// -----------------------------------------------------------------------------------------------
// Return the number of importers
size_t aiGetImportFormatCount(void)
{
    return Importer().GetImporterCount();
}

// ------------------------------------------------------------------------------------------------
// Returns the error text of the last failed import process.
aiBool aiIsExtensionSupported(const char* szExtension)
{
    ai_assert(NULL != szExtension);
    aiBool candoit=AI_FALSE;
    ASSIMP_BEGIN_EXCEPTION_REGION();

    // FIXME: no need to create a temporary Importer instance just for that ..
    Assimp::Importer tmp;
    candoit = tmp.IsExtensionSupported(std::string(szExtension)) ? AI_TRUE : AI_FALSE;

    ASSIMP_END_EXCEPTION_REGION(aiBool);
    return candoit;
}

// ------------------------------------------------------------------------------------------------
// Get a list of all file extensions supported by ASSIMP
void aiGetExtensionList(aiString* szOut)
{
    ai_assert(NULL != szOut);
    ASSIMP_BEGIN_EXCEPTION_REGION();

    // FIXME: no need to create a temporary Importer instance just for that ..
    Assimp::Importer tmp;
    tmp.GetExtensionList(*szOut);

    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
// Get the memory requirements for a particular import.
void aiGetMemoryRequirements(const C_STRUCT aiScene* pIn,
    C_STRUCT aiMemoryInfo* in)
{
    ASSIMP_BEGIN_EXCEPTION_REGION();

    // find the importer associated with this data
    const ScenePrivateData* priv = ScenePriv(pIn);
    if( !priv || !priv->mOrigImporter)  {
        ReportSceneNotFoundError();
        return;
    }

    return priv->mOrigImporter->GetMemoryRequirements(*in);
    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API aiPropertyStore* aiCreatePropertyStore(void)
{
    return reinterpret_cast<aiPropertyStore*>( new PropertyMap() );
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiReleasePropertyStore(aiPropertyStore* p)
{
    delete reinterpret_cast<PropertyMap*>(p);
}

// ------------------------------------------------------------------------------------------------
// Importer::SetPropertyInteger
ASSIMP_API void aiSetImportPropertyInteger(aiPropertyStore* p, const char* szName, int value)
{
    ASSIMP_BEGIN_EXCEPTION_REGION();
    PropertyMap* pp = reinterpret_cast<PropertyMap*>(p);
    SetGenericProperty<int>(pp->ints,szName,value);
    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
// Importer::SetPropertyFloat
ASSIMP_API void aiSetImportPropertyFloat(aiPropertyStore* p, const char* szName, ai_real value)
{
    ASSIMP_BEGIN_EXCEPTION_REGION();
    PropertyMap* pp = reinterpret_cast<PropertyMap*>(p);
    SetGenericProperty<ai_real>(pp->floats,szName,value);
    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
// Importer::SetPropertyString
ASSIMP_API void aiSetImportPropertyString(aiPropertyStore* p, const char* szName,
    const C_STRUCT aiString* st)
{
    if (!st) {
        return;
    }
    ASSIMP_BEGIN_EXCEPTION_REGION();
    PropertyMap* pp = reinterpret_cast<PropertyMap*>(p);
    SetGenericProperty<std::string>(pp->strings,szName,std::string(st->C_Str()));
    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
// Importer::SetPropertyMatrix
ASSIMP_API void aiSetImportPropertyMatrix(aiPropertyStore* p, const char* szName,
    const C_STRUCT aiMatrix4x4* mat)
{
    if (!mat) {
        return;
    }
    ASSIMP_BEGIN_EXCEPTION_REGION();
    PropertyMap* pp = reinterpret_cast<PropertyMap*>(p);
    SetGenericProperty<aiMatrix4x4>(pp->matrices,szName,*mat);
    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
// Rotation matrix to quaternion
ASSIMP_API void aiCreateQuaternionFromMatrix(aiQuaternion* quat,const aiMatrix3x3* mat)
{
    ai_assert( NULL != quat );
    ai_assert( NULL != mat );
    *quat = aiQuaternion(*mat);
}

// ------------------------------------------------------------------------------------------------
// Matrix decomposition
ASSIMP_API void aiDecomposeMatrix(const aiMatrix4x4* mat,aiVector3D* scaling,
    aiQuaternion* rotation,
    aiVector3D* position)
{
    ai_assert( NULL != rotation );
    ai_assert( NULL != position );
    ai_assert( NULL != scaling );
    ai_assert( NULL != mat );
    mat->Decompose(*scaling,*rotation,*position);
}

// ------------------------------------------------------------------------------------------------
// Matrix transpose
ASSIMP_API void aiTransposeMatrix3(aiMatrix3x3* mat)
{
    ai_assert(NULL != mat);
    mat->Transpose();
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiTransposeMatrix4(aiMatrix4x4* mat)
{
    ai_assert(NULL != mat);
    mat->Transpose();
}

// ------------------------------------------------------------------------------------------------
// Vector transformation
ASSIMP_API void aiTransformVecByMatrix3(aiVector3D* vec,
    const aiMatrix3x3* mat)
{
    ai_assert( NULL != mat );
    ai_assert( NULL != vec);
    *vec *= (*mat);
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiTransformVecByMatrix4(aiVector3D* vec,
    const aiMatrix4x4* mat)
{
    ai_assert( NULL != mat );
    ai_assert( NULL != vec );

    *vec *= (*mat);
}

// ------------------------------------------------------------------------------------------------
// Matrix multiplication
ASSIMP_API void aiMultiplyMatrix4(
    aiMatrix4x4* dst,
    const aiMatrix4x4* src)
{
    ai_assert( NULL != dst );
    ai_assert( NULL != src );
    *dst = (*dst) * (*src);
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiMultiplyMatrix3(
    aiMatrix3x3* dst,
    const aiMatrix3x3* src)
{
    ai_assert( NULL != dst );
    ai_assert( NULL != src );
    *dst = (*dst) * (*src);
}

// ------------------------------------------------------------------------------------------------
// Matrix identity
ASSIMP_API void aiIdentityMatrix3(
    aiMatrix3x3* mat)
{
    ai_assert(NULL != mat);
    *mat = aiMatrix3x3();
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiIdentityMatrix4(
    aiMatrix4x4* mat)
{
    ai_assert(NULL != mat);
    *mat = aiMatrix4x4();
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API C_STRUCT const aiImporterDesc* aiGetImporterDesc( const char *extension ) {
    if( NULL == extension ) {
        return NULL;
    }
    const aiImporterDesc *desc( NULL );
    std::vector< BaseImporter* > out;
    GetImporterInstanceList( out );
    for( size_t i = 0; i < out.size(); ++i ) {
        if( 0 == strncmp( out[ i ]->GetInfo()->mFileExtensions, extension, strlen( extension ) ) ) {
            desc = out[ i ]->GetInfo();
            break;
        }
    }

    DeleteImporterInstanceList(out);

    return desc;
}

// ------------------------------------------------------------------------------------------------

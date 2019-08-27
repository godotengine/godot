/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2016, assimp team
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

/** @file Android implementation of IOSystem using the standard C file functions.
 * Aimed to ease the access to android assets */

#if __ANDROID__ and __ANDROID_API__ > 9 and defined(AI_CONFIG_ANDROID_JNI_ASSIMP_MANAGER_SUPPORT)
#ifndef AI_ANDROIDJNIIOSYSTEM_H_INC
#define AI_ANDROIDJNIIOSYSTEM_H_INC

#include <assimp/DefaultIOSystem.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/native_activity.h>

namespace Assimp	{

// ---------------------------------------------------------------------------
/** Android extension to DefaultIOSystem using the standard C file functions */
class ASSIMP_API AndroidJNIIOSystem : public DefaultIOSystem
{
public:

	/** Initialize android activity data */
	std::string mApkWorkspacePath;
	AAssetManager* mApkAssetManager;

	/** Constructor. */
	AndroidJNIIOSystem(ANativeActivity* activity);

	/** Destructor. */
	~AndroidJNIIOSystem();

	// -------------------------------------------------------------------
	/** Tests for the existence of a file at the given path. */
	bool Exists( const char* pFile) const;

	// -------------------------------------------------------------------
	/** Opens a file at the given path, with given mode */
	IOStream* Open( const char* strFile, const char* strMode);

	// ------------------------------------------------------------------------------------------------
	// Inits Android extractor
	void AndroidActivityInit(ANativeActivity* activity);

	// ------------------------------------------------------------------------------------------------
	// Extracts android asset
	bool AndroidExtractAsset(std::string name);

};

} //!ns Assimp

#endif //AI_ANDROIDJNIIOSYSTEM_H_INC
#endif //__ANDROID__ and __ANDROID_API__ > 9 and defined(AI_CONFIG_ANDROID_JNI_ASSIMP_MANAGER_SUPPORT)

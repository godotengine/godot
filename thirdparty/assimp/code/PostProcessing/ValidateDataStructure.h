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

/** @file Defines a (dummy) post processing step to validate the loader's
 * output data structure (for debugging)
 */
#ifndef AI_VALIDATEPROCESS_H_INC
#define AI_VALIDATEPROCESS_H_INC

#include <assimp/types.h>
#include <assimp/material.h>

#include "Common/BaseProcess.h"

struct aiBone;
struct aiMesh;
struct aiAnimation;
struct aiNodeAnim;
struct aiMeshMorphAnim;
struct aiTexture;
struct aiMaterial;
struct aiNode;
struct aiString;
struct aiCamera;
struct aiLight;

namespace Assimp    {

// --------------------------------------------------------------------------------------
/** Validates the whole ASSIMP scene data structure for correctness.
 *  ImportErrorException is thrown of the scene is corrupt.*/
// --------------------------------------------------------------------------------------
class ValidateDSProcess : public BaseProcess
{
public:

    ValidateDSProcess();
    ~ValidateDSProcess();

public:
    // -------------------------------------------------------------------
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    void Execute( aiScene* pScene);

protected:

    // -------------------------------------------------------------------
    /** Report a validation error. This will throw an exception,
     *  control won't return.
     * @param msg Format string for sprintf().*/
    AI_WONT_RETURN void ReportError(const char* msg,...) AI_WONT_RETURN_SUFFIX;


    // -------------------------------------------------------------------
    /** Report a validation warning. This won't throw an exception,
     *  control will return to the caller.
     * @param msg Format string for sprintf().*/
    void ReportWarning(const char* msg,...);


    // -------------------------------------------------------------------
    /** Validates a mesh
     * @param pMesh Input mesh*/
    void Validate( const aiMesh* pMesh);

    // -------------------------------------------------------------------
    /** Validates a bone
     * @param pMesh Input mesh
     * @param pBone Input bone*/
    void Validate( const aiMesh* pMesh,const aiBone* pBone,float* afSum);

    // -------------------------------------------------------------------
    /** Validates an animation
     * @param pAnimation Input animation*/
    void Validate( const aiAnimation* pAnimation);

    // -------------------------------------------------------------------
    /** Validates a material
     * @param pMaterial Input material*/
    void Validate( const aiMaterial* pMaterial);

    // -------------------------------------------------------------------
    /** Search the material data structure for invalid or corrupt
     *  texture keys.
     * @param pMaterial Input material
     * @param type Type of the texture*/
    void SearchForInvalidTextures(const aiMaterial* pMaterial,
        aiTextureType type);

    // -------------------------------------------------------------------
    /** Validates a texture
     * @param pTexture Input texture*/
    void Validate( const aiTexture* pTexture);

    // -------------------------------------------------------------------
    /** Validates a light source
     * @param pLight Input light
     */
    void Validate( const aiLight* pLight);

    // -------------------------------------------------------------------
    /** Validates a camera
     * @param pCamera Input camera*/
    void Validate( const aiCamera* pCamera);

    // -------------------------------------------------------------------
    /** Validates a bone animation channel
     * @param pAnimation Animation channel.
     * @param pBoneAnim Input bone animation */
    void Validate( const aiAnimation* pAnimation,
        const aiNodeAnim* pBoneAnim);

    /** Validates a mesh morph animation channel.
     * @param pAnimation Input animation.
     * @param pMeshMorphAnim Mesh morph animation channel.
     * */
    void Validate( const aiAnimation* pAnimation,
        const aiMeshMorphAnim* pMeshMorphAnim);

    // -------------------------------------------------------------------
    /** Validates a node and all of its subnodes
     * @param Node Input node*/
    void Validate( const aiNode* pNode);

    // -------------------------------------------------------------------
    /** Validates a string
     * @param pString Input string*/
    void Validate( const aiString* pString);

private:

    // template to validate one of the aiScene::mXXX arrays
    template <typename T>
    inline void DoValidation(T** array, unsigned int size,
        const char* firstName, const char* secondName);

    // extended version: checks whether T::mName occurs twice
    template <typename T>
    inline void DoValidationEx(T** array, unsigned int size,
        const char* firstName, const char* secondName);

    // extension to the first template which does also search
    // the nodegraph for an item with the same name
    template <typename T>
    inline void DoValidationWithNameCheck(T** array, unsigned int size,
        const char* firstName, const char* secondName);

    aiScene* mScene;
};




} // end of namespace Assimp

#endif // AI_VALIDATEPROCESS_H_INC

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

/** @file Defines a post processing step to search all meshes for
  degenerated faces */
#ifndef AI_SCENE_PREPROCESSOR_H_INC
#define AI_SCENE_PREPROCESSOR_H_INC

#include <assimp/defs.h>
#include <stddef.h>

struct aiScene;
struct aiAnimation;
struct aiMesh;

class ScenePreprocessorTest;
namespace Assimp    {

// ----------------------------------------------------------------------------------
/** ScenePreprocessor: Preprocess a scene before any post-processing
 *  steps are executed.
 *
 *  The step computes data that needn't necessarily be provided by the
 *  importer, such as aiMesh::mPrimitiveTypes.
*/
// ----------------------------------------------------------------------------------
class ASSIMP_API ScenePreprocessor
{
    // Make ourselves a friend of the corresponding test unit.
    friend class ::ScenePreprocessorTest;
public:

    // ----------------------------------------------------------------
    /** Default c'tpr. Use SetScene() to assign a scene to the object.
     */
    ScenePreprocessor()
        :   scene   (NULL)
    {}

    /** Constructs the object and assigns a specific scene to it
     */
    ScenePreprocessor(aiScene* _scene)
        :   scene   (_scene)
    {}

    // ----------------------------------------------------------------
    /** Assign a (new) scene to the object.
     *
     *  One 'SceneProcessor' can be used for multiple scenes.
     *  Call ProcessScene to have the scene preprocessed.
     *  @param sc Scene to be processed.
     */
    void SetScene (aiScene* sc) {
        scene = sc;
    }

    // ----------------------------------------------------------------
    /** Preprocess the current scene
     */
    void ProcessScene ();

protected:

    // ----------------------------------------------------------------
    /** Preprocess an animation in the scene
     *  @param anim Anim to be preprocessed.
     */
    void ProcessAnimation (aiAnimation* anim);


    // ----------------------------------------------------------------
    /** Preprocess a mesh in the scene
     *  @param mesh Mesh to be preprocessed.
     */
    void ProcessMesh (aiMesh* mesh);

protected:

    //! Scene we're currently working on
    aiScene* scene;
};


} // ! end namespace Assimp

#endif // include guard

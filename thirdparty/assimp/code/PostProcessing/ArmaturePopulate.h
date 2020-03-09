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
#ifndef ARMATURE_POPULATE_H_
#define ARMATURE_POPULATE_H_

#include "Common/BaseProcess.h"
#include <assimp/BaseImporter.h>
#include <vector>
#include <map>


struct aiNode;
struct aiBone;

namespace Assimp {

// ---------------------------------------------------------------------------
/** Armature Populate: This is a post process designed
 * To save you time when importing models into your game engines
 * This was originally designed only for fbx but will work with other formats
 * it is intended to auto populate aiBone data with armature and the aiNode
 * This is very useful when dealing with skinned meshes
 * or when dealing with many different skeletons
 * It's off by default but recommend that you try it and use it
 * It should reduce down any glue code you have in your
 * importers
 * You can contact RevoluPowered <gordon@gordonite.tech>
 * For more info about this
*/
class ASSIMP_API ArmaturePopulate : public BaseProcess {
public:
    /// The default class constructor.
    ArmaturePopulate();

    /// The class destructor.
    virtual ~ArmaturePopulate();

    /// Overwritten, @see BaseProcess
    virtual bool IsActive( unsigned int pFlags ) const;

    /// Overwritten, @see BaseProcess
    virtual void SetupProperties( const Importer* pImp );

    /// Overwritten, @see BaseProcess
    virtual void Execute( aiScene* pScene );

    static aiNode *GetArmatureRoot(aiNode *bone_node,
                                      std::vector<aiBone *> &bone_list);

    static bool IsBoneNode(const aiString &bone_name,
                              std::vector<aiBone *> &bones);

    static aiNode *GetNodeFromStack(const aiString &node_name,
                                       std::vector<aiNode *> &nodes);

    static void BuildNodeList(const aiNode *current_node,
                                 std::vector<aiNode *> &nodes);

    static void BuildBoneList(aiNode *current_node, const aiNode *root_node,
                                 const aiScene *scene,
                                 std::vector<aiBone *> &bones);                        

    static void BuildBoneStack(aiNode *current_node, const aiNode *root_node,
                                  const aiScene *scene,
                                  const std::vector<aiBone *> &bones,
                                  std::map<aiBone *, aiNode *> &bone_stack,
                                  std::vector<aiNode *> &node_stack);
};

} // Namespace Assimp


#endif // SCALE_PROCESS_H_
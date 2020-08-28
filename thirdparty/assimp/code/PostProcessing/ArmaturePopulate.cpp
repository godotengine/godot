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
#include "ArmaturePopulate.h"

#include <assimp/BaseImporter.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <iostream>

namespace Assimp {

/// The default class constructor.
ArmaturePopulate::ArmaturePopulate() : BaseProcess()
{}

/// The class destructor.
ArmaturePopulate::~ArmaturePopulate() 
{}

bool ArmaturePopulate::IsActive(unsigned int pFlags) const {
  return (pFlags & aiProcess_PopulateArmatureData) != 0;
}

void ArmaturePopulate::SetupProperties(const Importer *pImp) {
  // do nothing
}

void ArmaturePopulate::Execute(aiScene *out) {

  // Now convert all bone positions to the correct mOffsetMatrix
  std::vector<aiBone *> bones;
  std::vector<aiNode *> nodes;
  std::map<aiBone *, aiNode *> bone_stack;
  BuildBoneList(out->mRootNode, out->mRootNode, out, bones);
  BuildNodeList(out->mRootNode, nodes);

  BuildBoneStack(out->mRootNode, out->mRootNode, out, bones, bone_stack, nodes);

  ASSIMP_LOG_DEBUG_F("Bone stack size: ", bone_stack.size());

  for (std::pair<aiBone *, aiNode *> kvp : bone_stack) {
    aiBone *bone = kvp.first;
    aiNode *bone_node = kvp.second;
    ASSIMP_LOG_DEBUG_F("active node lookup: ", bone->mName.C_Str());
    // lcl transform grab - done in generate_nodes :)

    // bone->mOffsetMatrix = bone_node->mTransformation;
    aiNode *armature = GetArmatureRoot(bone_node, bones);

    ai_assert(armature);

    // set up bone armature id
    bone->mArmature = armature;

    // set this bone node to be referenced properly
    ai_assert(bone_node);
    bone->mNode = bone_node;
  }
}


/* Reprocess all nodes to calculate bone transforms properly based on the REAL
 * mOffsetMatrix not the local. */
/* Before this would use mesh transforms which is wrong for bone transforms */
/* Before this would work for simple character skeletons but not complex meshes
 * with multiple origins */
/* Source: sketch fab log cutter fbx */
void ArmaturePopulate::BuildBoneList(aiNode *current_node,
                                     const aiNode *root_node,
                                     const aiScene *scene,
                                     std::vector<aiBone *> &bones) {
  ai_assert(scene);
  for (unsigned int nodeId = 0; nodeId < current_node->mNumChildren; ++nodeId) {
    aiNode *child = current_node->mChildren[nodeId];
    ai_assert(child);

    // check for bones
    for (unsigned int meshId = 0; meshId < child->mNumMeshes; ++meshId) {
      ai_assert(child->mMeshes);
      unsigned int mesh_index = child->mMeshes[meshId];
      aiMesh *mesh = scene->mMeshes[mesh_index];
      ai_assert(mesh);

      for (unsigned int boneId = 0; boneId < mesh->mNumBones; ++boneId) {
        aiBone *bone = mesh->mBones[boneId];
        ai_assert(bone);

        // duplicate meshes exist with the same bones sometimes :)
        // so this must be detected
        if (std::find(bones.begin(), bones.end(), bone) == bones.end()) {
          // add the element once
          bones.push_back(bone);
        }
      }

      // find mesh and get bones
      // then do recursive lookup for bones in root node hierarchy
    }

    BuildBoneList(child, root_node, scene, bones);
  }
}

/* Prepare flat node list which can be used for non recursive lookups later */
void ArmaturePopulate::BuildNodeList(const aiNode *current_node,
                                     std::vector<aiNode *> &nodes) {
  ai_assert(current_node);

  for (unsigned int nodeId = 0; nodeId < current_node->mNumChildren; ++nodeId) {
    aiNode *child = current_node->mChildren[nodeId];
    ai_assert(child);

    nodes.push_back(child);

    BuildNodeList(child, nodes);
  }
}

/* A bone stack allows us to have multiple armatures, with the same bone names
 * A bone stack allows us also to retrieve bones true transform even with
 * duplicate names :)
 */
void ArmaturePopulate::BuildBoneStack(aiNode *current_node,
                                      const aiNode *root_node,
                                      const aiScene *scene,
                                      const std::vector<aiBone *> &bones,
                                      std::map<aiBone *, aiNode *> &bone_stack,
                                      std::vector<aiNode *> &node_stack) {
  ai_assert(scene);
  ai_assert(root_node);
  ai_assert(!node_stack.empty());

  for (aiBone *bone : bones) {
    ai_assert(bone);
    aiNode *node = GetNodeFromStack(bone->mName, node_stack);
    if (node == nullptr) {
      node_stack.clear();
      BuildNodeList(root_node, node_stack);
      ASSIMP_LOG_DEBUG_F("Resetting bone stack: nullptr element ", bone->mName.C_Str());

      node = GetNodeFromStack(bone->mName, node_stack);

      if (!node) {
        ASSIMP_LOG_ERROR("serious import issue node for bone was not detected");
        continue;
      }
    }

    ASSIMP_LOG_DEBUG_F("Successfully added bone[", bone->mName.C_Str(), "] to stack and bone node is: ", node->mName.C_Str());

    bone_stack.insert(std::pair<aiBone *, aiNode *>(bone, node));
  }
}


/* Returns the armature root node */
/* This is required to be detected for a bone initially, it will recurse up
 * until it cannot find another bone and return the node No known failure
 * points. (yet)
 */
aiNode *ArmaturePopulate::GetArmatureRoot(aiNode *bone_node,
                                          std::vector<aiBone *> &bone_list) {
  while (bone_node) {
    if (!IsBoneNode(bone_node->mName, bone_list)) {
      ASSIMP_LOG_DEBUG_F("GetArmatureRoot() Found valid armature: ", bone_node->mName.C_Str());
      return bone_node;
    }

    bone_node = bone_node->mParent;
  }
  
  ASSIMP_LOG_ERROR("GetArmatureRoot() can't find armature!");
  
  return nullptr;
}



/* Simple IsBoneNode check if this could be a bone */
bool ArmaturePopulate::IsBoneNode(const aiString &bone_name,
                                  std::vector<aiBone *> &bones) {
  for (aiBone *bone : bones) {
    if (bone->mName == bone_name) {
      return true;
    }
  }

  return false;
}

/* Pop this node by name from the stack if found */
/* Used in multiple armature situations with duplicate node / bone names */
/* Known flaw: cannot have nodes with bone names, will be fixed in later release
 */
/* (serious to be fixed) Known flaw: nodes which have more than one bone could
 * be prematurely dropped from stack */
aiNode *ArmaturePopulate::GetNodeFromStack(const aiString &node_name,
                                           std::vector<aiNode *> &nodes) {
  std::vector<aiNode *>::iterator iter;
  aiNode *found = nullptr;
  for (iter = nodes.begin(); iter < nodes.end(); ++iter) {
    aiNode *element = *iter;
    ai_assert(element);
    // node valid and node name matches
    if (element->mName == node_name) {
      found = element;
      break;
    }
  }

  if (found != nullptr) {
    ASSIMP_LOG_INFO_F("Removed node from stack: ", found->mName.C_Str());
    // now pop the element from the node list
    nodes.erase(iter);

    return found;
  }

  // unique names can cause this problem
  ASSIMP_LOG_ERROR("[Serious] GetNodeFromStack() can't find node from stack!");

  return nullptr;
}




} // Namespace Assimp

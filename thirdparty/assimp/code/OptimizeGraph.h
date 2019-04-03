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

/** @file  OptimizeGraph.h
 *  @brief Declares a post processing step to optimize the scenegraph
 */
#ifndef AI_OPTIMIZEGRAPHPROCESS_H_INC
#define AI_OPTIMIZEGRAPHPROCESS_H_INC

#include "BaseProcess.h"
#include "ProcessHelper.h"
#include <assimp/types.h>
#include <set>

struct aiMesh;
class OptimizeGraphProcessTest;
namespace Assimp    {

// -----------------------------------------------------------------------------
/** @brief Postprocessing step to optimize the scenegraph
 *
 *  The implementation tries to merge nodes, even if they use different
 *  transformations. Animations are preserved.
 *
 *  @see aiProcess_OptimizeGraph for a detailed description of the
 *  algorithm being applied.
 */
class OptimizeGraphProcess : public BaseProcess
{
public:

    OptimizeGraphProcess();
    ~OptimizeGraphProcess();

public:
    // -------------------------------------------------------------------
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    void SetupProperties(const Importer* pImp);


    // -------------------------------------------------------------------
    /** @brief Add a list of node names to be locked and not modified.
     *  @param in List of nodes. See #AI_CONFIG_PP_OG_EXCLUDE_LIST for
     *    format explanations.
     */
    inline void AddLockedNodeList(std::string& in)
    {
        ConvertListToStrings (in,locked_nodes);
    }

    // -------------------------------------------------------------------
    /** @brief Add another node to be locked and not modified.
     *  @param name Name to be locked
     */
    inline void AddLockedNode(std::string& name)
    {
        locked_nodes.push_back(name);
    }

    // -------------------------------------------------------------------
    /** @brief Remove a node from the list of locked nodes.
     *  @param name Name to be unlocked
     */
    inline void RemoveLockedNode(std::string& name)
    {
        locked_nodes.remove(name);
    }

protected:

    void CollectNewChildren(aiNode* nd, std::list<aiNode*>& nodes);
    void FindInstancedMeshes (aiNode* pNode);

private:

#ifdef AI_OG_USE_HASHING
    typedef std::set<unsigned int> LockedSetType;
#else
    typedef std::set<std::string> LockedSetType;
#endif


    //! Scene we're working with
    aiScene* mScene;

    //! List of locked names. Stored is the hash of the name
    LockedSetType locked;

    //! List of nodes to be locked in addition to those with animations, lights or cameras assigned.
    std::list<std::string> locked_nodes;

    //! Node counters for logging purposes
    unsigned int nodes_in,nodes_out, count_merged;

    //! Reference counters for meshes
    std::vector<unsigned int> meshes;
};

} // end of namespace Assimp

#endif // AI_OPTIMIZEGRAPHPROCESS_H_INC

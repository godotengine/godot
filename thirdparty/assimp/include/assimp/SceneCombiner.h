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

/** @file Declares a helper class, "SceneCombiner" providing various
 *  utilities to merge scenes.
 */
#pragma once
#ifndef AI_SCENE_COMBINER_H_INC
#define AI_SCENE_COMBINER_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/ai_assert.h>
#include <assimp/types.h>
#include <assimp/Defines.h>

#include <stddef.h>
#include <set>
#include <list>
#include <stdint.h>
#include <vector>

struct aiScene;
struct aiNode;
struct aiMaterial;
struct aiTexture;
struct aiCamera;
struct aiLight;
struct aiMetadata;
struct aiBone;
struct aiMesh;
struct aiAnimMesh;
struct aiAnimation;
struct aiNodeAnim;
struct aiMeshMorphAnim;

namespace Assimp    {

// ---------------------------------------------------------------------------
/** \brief Helper data structure for SceneCombiner.
 *
 *  Describes to which node a scene must be attached to.
 */
struct AttachmentInfo
{
    AttachmentInfo()
        :   scene           (NULL)
        ,   attachToNode    (NULL)
    {}

    AttachmentInfo(aiScene* _scene, aiNode* _attachToNode)
        :   scene           (_scene)
        ,   attachToNode    (_attachToNode)
    {}

    aiScene* scene;
    aiNode*  attachToNode;
};

// ---------------------------------------------------------------------------
struct NodeAttachmentInfo
{
    NodeAttachmentInfo()
        :   node            (NULL)
        ,   attachToNode    (NULL)
        ,   resolved        (false)
        ,   src_idx         (SIZE_MAX)
    {}

    NodeAttachmentInfo(aiNode* _scene, aiNode* _attachToNode,size_t idx)
        :   node            (_scene)
        ,   attachToNode    (_attachToNode)
        ,   resolved        (false)
        ,   src_idx         (idx)
    {}

    aiNode*  node;
    aiNode*  attachToNode;
    bool     resolved;
    size_t   src_idx;
};

// ---------------------------------------------------------------------------
/** @def AI_INT_MERGE_SCENE_GEN_UNIQUE_NAMES
 *  Generate unique names for all named scene items
 */
#define AI_INT_MERGE_SCENE_GEN_UNIQUE_NAMES      0x1

/** @def AI_INT_MERGE_SCENE_GEN_UNIQUE_MATNAMES
 *  Generate unique names for materials, too.
 *  This is not absolutely required to pass the validation.
 */
#define AI_INT_MERGE_SCENE_GEN_UNIQUE_MATNAMES   0x2

/** @def AI_INT_MERGE_SCENE_DUPLICATES_DEEP_CPY
 * Use deep copies of duplicate scenes
 */
#define AI_INT_MERGE_SCENE_DUPLICATES_DEEP_CPY   0x4

/** @def AI_INT_MERGE_SCENE_RESOLVE_CROSS_ATTACHMENTS
 * If attachment nodes are not found in the given master scene,
 * search the other imported scenes for them in an any order.
 */
#define AI_INT_MERGE_SCENE_RESOLVE_CROSS_ATTACHMENTS 0x8

/** @def AI_INT_MERGE_SCENE_GEN_UNIQUE_NAMES_IF_NECESSARY
 * Can be combined with AI_INT_MERGE_SCENE_GEN_UNIQUE_NAMES.
 * Unique names are generated, but only if this is absolutely
 * required to avoid name conflicts.
 */
#define AI_INT_MERGE_SCENE_GEN_UNIQUE_NAMES_IF_NECESSARY 0x10

typedef std::pair<aiBone*,unsigned int> BoneSrcIndex;

// ---------------------------------------------------------------------------
/** @brief Helper data structure for SceneCombiner::MergeBones.
 */
struct BoneWithHash : public std::pair<uint32_t,aiString*>  {
    std::vector<BoneSrcIndex> pSrcBones;
};

// ---------------------------------------------------------------------------
/** @brief Utility for SceneCombiner
 */
struct SceneHelper
{
    SceneHelper ()
        : scene     (NULL)
        , idlen     (0)
    {
        id[0] = 0;
    }

    explicit SceneHelper (aiScene* _scene)
        : scene     (_scene)
        , idlen     (0)
    {
        id[0] = 0;
    }

    AI_FORCE_INLINE aiScene* operator-> () const
    {
        return scene;
    }

    // scene we're working on
    aiScene* scene;

    // prefix to be added to all identifiers in the scene ...
    char id [32];

    // and its strlen()
    unsigned int idlen;

    // hash table to quickly check whether a name is contained in the scene
    std::set<unsigned int> hashes;
};

// ---------------------------------------------------------------------------
/** \brief Static helper class providing various utilities to merge two
 *    scenes. It is intended as internal utility and NOT for use by
 *    applications.
 *
 * The class is currently being used by various postprocessing steps
 * and loaders (ie. LWS).
 */
class ASSIMP_API SceneCombiner {
    // class cannot be instanced
    SceneCombiner() {
        // empty
    }

    ~SceneCombiner() {
        // empty
    }

public:
    // -------------------------------------------------------------------
    /** Merges two or more scenes.
     *
     *  @param dest  Receives a pointer to the destination scene. If the
     *    pointer doesn't point to NULL when the function is called, the
     *    existing scene is cleared and refilled.
     *  @param src Non-empty list of scenes to be merged. The function
     *    deletes the input scenes afterwards. There may be duplicate scenes.
     *  @param flags Combination of the AI_INT_MERGE_SCENE flags defined above
     */
    static void MergeScenes(aiScene** dest,std::vector<aiScene*>& src,
        unsigned int flags = 0);

    // -------------------------------------------------------------------
    /** Merges two or more scenes and attaches all scenes to a specific
     *  position in the node graph of the master scene.
     *
     *  @param dest Receives a pointer to the destination scene. If the
     *    pointer doesn't point to NULL when the function is called, the
     *    existing scene is cleared and refilled.
     *  @param master Master scene. It will be deleted afterwards. All
     *    other scenes will be inserted in its node graph.
     *  @param src Non-empty list of scenes to be merged along with their
     *    corresponding attachment points in the master scene. The function
     *    deletes the input scenes afterwards. There may be duplicate scenes.
     *  @param flags Combination of the AI_INT_MERGE_SCENE flags defined above
     */
    static void MergeScenes(aiScene** dest, aiScene* master,
        std::vector<AttachmentInfo>& src,
        unsigned int flags = 0);

    // -------------------------------------------------------------------
    /** Merges two or more meshes
     *
     *  The meshes should have equal vertex formats. Only components
     *  that are provided by ALL meshes will be present in the output mesh.
     *  An exception is made for VColors - they are set to black. The
     *  meshes should have the same material indices, too. The output
     *  material index is always the material index of the first mesh.
     *
     *  @param dest Destination mesh. Must be empty.
     *  @param flags Currently no parameters
     *  @param begin First mesh to be processed
     *  @param end Points to the mesh after the last mesh to be processed
     */
    static void MergeMeshes(aiMesh** dest,unsigned int flags,
        std::vector<aiMesh*>::const_iterator begin,
        std::vector<aiMesh*>::const_iterator end);

    // -------------------------------------------------------------------
    /** Merges two or more bones
     *
     *  @param out Mesh to receive the output bone list
     *  @param flags Currently no parameters
     *  @param begin First mesh to be processed
     *  @param end Points to the mesh after the last mesh to be processed
     */
    static void MergeBones(aiMesh* out,std::vector<aiMesh*>::const_iterator it,
        std::vector<aiMesh*>::const_iterator end);

    // -------------------------------------------------------------------
    /** Merges two or more materials
     *
     *  The materials should be complementary as much as possible. In case
     *  of a property present in different materials, the first occurrence
     *  is used.
     *
     *  @param dest Destination material. Must be empty.
     *  @param begin First material to be processed
     *  @param end Points to the material after the last material to be processed
     */
    static void MergeMaterials(aiMaterial** dest,
        std::vector<aiMaterial*>::const_iterator begin,
        std::vector<aiMaterial*>::const_iterator end);

    // -------------------------------------------------------------------
    /** Builds a list of uniquely named bones in a mesh list
     *
     *  @param asBones Receives the output list
     *  @param it First mesh to be processed
     *  @param end Last mesh to be processed
     */
    static void BuildUniqueBoneList(std::list<BoneWithHash>& asBones,
        std::vector<aiMesh*>::const_iterator it,
        std::vector<aiMesh*>::const_iterator end);

    // -------------------------------------------------------------------
    /** Add a name prefix to all nodes in a scene.
     *
     *  @param Current node. This function is called recursively.
     *  @param prefix Prefix to be added to all nodes
     *  @param len STring length
     */
    static void AddNodePrefixes(aiNode* node, const char* prefix,
        unsigned int len);

    // -------------------------------------------------------------------
    /** Add an offset to all mesh indices in a node graph
     *
     *  @param Current node. This function is called recursively.
     *  @param offset Offset to be added to all mesh indices
     */
    static void OffsetNodeMeshIndices (aiNode* node, unsigned int offset);

    // -------------------------------------------------------------------
    /** Attach a list of node graphs to well-defined nodes in a master
     *  graph. This is a helper for MergeScenes()
     *
     *  @param master Master scene
     *  @param srcList List of source scenes along with their attachment
     *    points. If an attachment point is NULL (or does not exist in
     *    the master graph), a scene is attached to the root of the master
     *    graph (as an additional child node)
     *  @duplicates List of duplicates. If elem[n] == n the scene is not
     *    a duplicate. Otherwise elem[n] links scene n to its first occurrence.
     */
    static void AttachToGraph ( aiScene* master,
        std::vector<NodeAttachmentInfo>& srcList);

    static void AttachToGraph (aiNode* attach,
        std::vector<NodeAttachmentInfo>& srcList);


    // -------------------------------------------------------------------
    /** Get a deep copy of a scene
     *
     *  @param dest Receives a pointer to the destination scene
     *  @param src Source scene - remains unmodified.
     */
    static void CopyScene(aiScene** dest,const aiScene* source,bool allocate = true);


    // -------------------------------------------------------------------
    /** Get a flat copy of a scene
     *
     *  Only the first hierarchy layer is copied. All pointer members of
     *  aiScene are shared by source and destination scene.  If the
     *    pointer doesn't point to NULL when the function is called, the
     *    existing scene is cleared and refilled.
     *  @param dest Receives a pointer to the destination scene
     *  @param src Source scene - remains unmodified.
     */
    static void CopySceneFlat(aiScene** dest,const aiScene* source);


    // -------------------------------------------------------------------
    /** Get a deep copy of a mesh
     *
     *  @param dest Receives a pointer to the destination mesh
     *  @param src Source mesh - remains unmodified.
     */
    static void Copy     (aiMesh** dest, const aiMesh* src);

    // similar to Copy():
    static void Copy  (aiAnimMesh** dest, const aiAnimMesh* src);
    static void Copy  (aiMaterial** dest, const aiMaterial* src);
    static void Copy  (aiTexture** dest, const aiTexture* src);
    static void Copy  (aiAnimation** dest, const aiAnimation* src);
    static void Copy  (aiCamera** dest, const aiCamera* src);
    static void Copy  (aiBone** dest, const aiBone* src);
    static void Copy  (aiLight** dest, const aiLight* src);
    static void Copy  (aiNodeAnim** dest, const aiNodeAnim* src);
    static void Copy  (aiMeshMorphAnim** dest, const aiMeshMorphAnim* src);
    static void Copy  (aiMetadata** dest, const aiMetadata* src);

    // recursive, of course
    static void Copy     (aiNode** dest, const aiNode* src);


private:

    // -------------------------------------------------------------------
    // Same as AddNodePrefixes, but with an additional check
    static void AddNodePrefixesChecked(aiNode* node, const char* prefix,
        unsigned int len,
        std::vector<SceneHelper>& input,
        unsigned int cur);

    // -------------------------------------------------------------------
    // Add node identifiers to a hashing set
    static void AddNodeHashes(aiNode* node, std::set<unsigned int>& hashes);


    // -------------------------------------------------------------------
    // Search for duplicate names
    static bool FindNameMatch(const aiString& name,
        std::vector<SceneHelper>& input, unsigned int cur);
};

}

#endif // !! AI_SCENE_COMBINER_H_INC

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

/** @file PretransformVertices.h
 *  @brief Defines a post processing step to pretransform all
 *    vertices in the scenegraph
 */
#ifndef AI_PRETRANSFORMVERTICES_H_INC
#define AI_PRETRANSFORMVERTICES_H_INC

#include "Common/BaseProcess.h"

#include <assimp/mesh.h>

#include <list>
#include <vector>

// Forward declarations
struct aiNode;

class PretransformVerticesTest;

namespace Assimp {

// ---------------------------------------------------------------------------
/** The PretransformVertices pre-transforms all vertices in the node tree
 *  and removes the whole graph. The output is a list of meshes, one for
 *  each material.
*/
class ASSIMP_API PretransformVertices : public BaseProcess {
public:
	PretransformVertices();
	~PretransformVertices();

	// -------------------------------------------------------------------
	// Check whether step is active
	bool IsActive(unsigned int pFlags) const override;

	// -------------------------------------------------------------------
	// Execute step on a given scene
	void Execute(aiScene *pScene) override;

	// -------------------------------------------------------------------
	// Setup import settings
	void SetupProperties(const Importer *pImp) override;

	// -------------------------------------------------------------------
	/** @brief Toggle the 'keep hierarchy' option
     *  @param keep    true for keep configuration.
     */
	void KeepHierarchy(bool keep) {
		configKeepHierarchy = keep;
	}

	// -------------------------------------------------------------------
	/** @brief Check whether 'keep hierarchy' is currently enabled.
     *  @return ...
     */
	bool IsHierarchyKept() const {
		return configKeepHierarchy;
	}

private:
	// -------------------------------------------------------------------
	// Count the number of nodes
	unsigned int CountNodes(const aiNode *pcNode) const;

	// -------------------------------------------------------------------
	// Get a bitwise combination identifying the vertex format of a mesh
	unsigned int GetMeshVFormat(aiMesh *pcMesh) const;

	// -------------------------------------------------------------------
	// Count the number of vertices in the whole scene and a given
	// material index
	void CountVerticesAndFaces(const aiScene *pcScene, const aiNode *pcNode,
			unsigned int iMat,
			unsigned int iVFormat,
			unsigned int *piFaces,
			unsigned int *piVertices) const;

	// -------------------------------------------------------------------
	// Collect vertex/face data
	void CollectData(const aiScene *pcScene, const aiNode *pcNode,
			unsigned int iMat,
			unsigned int iVFormat,
			aiMesh *pcMeshOut,
			unsigned int aiCurrent[2],
			unsigned int *num_refs) const;

	// -------------------------------------------------------------------
	// Get a list of all vertex formats that occur for a given material
	// The output list contains duplicate elements
	void GetVFormatList(const aiScene *pcScene, unsigned int iMat,
			std::list<unsigned int> &aiOut) const;

	// -------------------------------------------------------------------
	// Compute the absolute transformation matrices of each node
	void ComputeAbsoluteTransform(aiNode *pcNode);

	// -------------------------------------------------------------------
	// Simple routine to build meshes in worldspace, no further optimization
	void BuildWCSMeshes(std::vector<aiMesh *> &out, aiMesh **in,
			unsigned int numIn, aiNode *node) const;

	// -------------------------------------------------------------------
	// Apply the node transformation to a mesh
	void ApplyTransform(aiMesh *mesh, const aiMatrix4x4 &mat) const;

	// -------------------------------------------------------------------
	// Reset transformation matrices to identity
	void MakeIdentityTransform(aiNode *nd) const;

	// -------------------------------------------------------------------
	// Build reference counters for all meshes
	void BuildMeshRefCountArray(const aiNode *nd, unsigned int *refs) const;

	//! Configuration option: keep scene hierarchy as long as possible
	bool configKeepHierarchy;
	bool configNormalize;
	bool configTransform;
	aiMatrix4x4 configTransformation;
	bool mConfigPointCloud;
};

} // end of namespace Assimp

#endif // !!AI_GENFACENORMALPROCESS_H_INC

#ifndef BT_REDUCED_SOFT_BODY_HELPERS_H
#define BT_REDUCED_SOFT_BODY_HELPERS_H

#include "btReducedDeformableBody.h"
#include <string>

struct btReducedDeformableBodyHelpers
{
	// create a reduced deformable object
	static btReducedDeformableBody* createReducedDeformableObject(btSoftBodyWorldInfo& worldInfo, const std::string& file_path, const std::string& vtk_file, const int num_modes, bool rigid_only);
	// read in geometry info from Vtk file
  static btReducedDeformableBody* createFromVtkFile(btSoftBodyWorldInfo& worldInfo, const char* vtk_file);
	// read in all reduced files
	static void readReducedDeformableInfoFromFiles(btReducedDeformableBody* rsb, const char* file_path);
	// read in a binary vector
	static void readBinaryVec(btReducedDeformableBody::tDenseArray& vec, const unsigned int n_size, const char* file);
	// read in a binary matrix
	static void readBinaryMat(btReducedDeformableBody::tDenseMatrix& mat, const unsigned int n_modes, const unsigned int n_full, const char* file);
	
	// calculate the local inertia tensor for a box shape reduced deformable object
	static void calculateLocalInertia(btVector3& inertia, const btScalar mass, const btVector3& half_extents, const btVector3& margin);
};


#endif // BT_REDUCED_SOFT_BODY_HELPERS_H
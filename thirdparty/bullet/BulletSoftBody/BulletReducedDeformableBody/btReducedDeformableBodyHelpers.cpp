#include "btReducedDeformableBodyHelpers.h"
#include "../btSoftBodyHelpers.h"
#include <iostream>
#include <string>
#include <sstream>

btReducedDeformableBody* btReducedDeformableBodyHelpers::createReducedDeformableObject(btSoftBodyWorldInfo& worldInfo, const std::string& file_path, const std::string& vtk_file, const int num_modes, bool rigid_only) {
	std::string filename = file_path + vtk_file;
	btReducedDeformableBody* rsb = btReducedDeformableBodyHelpers::createFromVtkFile(worldInfo, filename.c_str());
	
	rsb->setReducedModes(num_modes, rsb->m_nodes.size());
	btReducedDeformableBodyHelpers::readReducedDeformableInfoFromFiles(rsb, file_path.c_str());
	
	rsb->disableReducedModes(rigid_only);

	return rsb;
}

btReducedDeformableBody* btReducedDeformableBodyHelpers::createFromVtkFile(btSoftBodyWorldInfo& worldInfo, const char* vtk_file)
{
	std::ifstream fs;
	fs.open(vtk_file);
	btAssert(fs);

	typedef btAlignedObjectArray<int> Index;
	std::string line;
	btAlignedObjectArray<btVector3> X;
	btVector3 position;
	btAlignedObjectArray<Index> indices;
	bool reading_points = false;
	bool reading_tets = false;
	size_t n_points = 0;
	size_t n_tets = 0;
	size_t x_count = 0;
	size_t indices_count = 0;
	while (std::getline(fs, line))
	{
		std::stringstream ss(line);
		if (line.size() == (size_t)(0))
		{
		}
		else if (line.substr(0, 6) == "POINTS")
		{
			reading_points = true;
			reading_tets = false;
			ss.ignore(128, ' ');  // ignore "POINTS"
			ss >> n_points;
			X.resize(n_points);
		}
		else if (line.substr(0, 5) == "CELLS")
		{
			reading_points = false;
			reading_tets = true;
			ss.ignore(128, ' ');  // ignore "CELLS"
			ss >> n_tets;
			indices.resize(n_tets);
		}
		else if (line.substr(0, 10) == "CELL_TYPES")
		{
			reading_points = false;
			reading_tets = false;
		}
		else if (reading_points)
		{
			btScalar p;
			ss >> p;
			position.setX(p);
			ss >> p;
			position.setY(p);
			ss >> p;
			position.setZ(p);
			//printf("v %f %f %f\n", position.getX(), position.getY(), position.getZ());
			X[x_count++] = position;
		}
		else if (reading_tets)
		{
			int d;
			ss >> d;
			if (d != 4)
			{
				printf("Load deformable failed: Only Tetrahedra are supported in VTK file.\n");
				fs.close();
				return 0;
			}
			ss.ignore(128, ' ');  // ignore "4"
			Index tet;
			tet.resize(4);
			for (size_t i = 0; i < 4; i++)
			{
				ss >> tet[i];
				//printf("%d ", tet[i]);
			}
			//printf("\n");
			indices[indices_count++] = tet;
		}
	}
	btReducedDeformableBody* rsb = new btReducedDeformableBody(&worldInfo, n_points, &X[0], 0);

	for (int i = 0; i < n_tets; ++i)
	{
		const Index& ni = indices[i];
		rsb->appendTetra(ni[0], ni[1], ni[2], ni[3]);
		{
			rsb->appendLink(ni[0], ni[1], 0, true);
			rsb->appendLink(ni[1], ni[2], 0, true);
			rsb->appendLink(ni[2], ni[0], 0, true);
			rsb->appendLink(ni[0], ni[3], 0, true);
			rsb->appendLink(ni[1], ni[3], 0, true);
			rsb->appendLink(ni[2], ni[3], 0, true);
		}
	}

	btSoftBodyHelpers::generateBoundaryFaces(rsb);
	rsb->initializeDmInverse();
	rsb->m_tetraScratches.resize(rsb->m_tetras.size());
	rsb->m_tetraScratchesTn.resize(rsb->m_tetras.size());
	printf("Nodes:  %u\r\n", rsb->m_nodes.size());
	printf("Links:  %u\r\n", rsb->m_links.size());
	printf("Faces:  %u\r\n", rsb->m_faces.size());
	printf("Tetras: %u\r\n", rsb->m_tetras.size());

	fs.close();

	return rsb;
}

void btReducedDeformableBodyHelpers::readReducedDeformableInfoFromFiles(btReducedDeformableBody* rsb, const char* file_path)
{
	// read in eigenmodes, stiffness and mass matrices
	std::string eigenvalues_file = std::string(file_path) + "eigenvalues.bin";
	btReducedDeformableBodyHelpers::readBinaryVec(rsb->m_eigenvalues, rsb->m_nReduced, eigenvalues_file.c_str());

	std::string Kr_file = std::string(file_path) + "K_r_diag_mat.bin";
	btReducedDeformableBodyHelpers::readBinaryVec(rsb->m_Kr,  rsb->m_nReduced, Kr_file.c_str());

	// std::string Mr_file = std::string(file_path) + "M_r_diag_mat.bin";
	// btReducedDeformableBodyHelpers::readBinaryVec(rsb->m_Mr, rsb->m_nReduced, Mr_file.c_str());

	std::string modes_file = std::string(file_path) + "modes.bin";
	btReducedDeformableBodyHelpers::readBinaryMat(rsb->m_modes, rsb->m_nReduced, 3 * rsb->m_nFull, modes_file.c_str());
	
	// read in full nodal mass
	std::string M_file = std::string(file_path) + "M_diag_mat.bin";
	btAlignedObjectArray<btScalar> mass_array;
	btReducedDeformableBodyHelpers::readBinaryVec(mass_array, rsb->m_nFull, M_file.c_str());
	rsb->setMassProps(mass_array);
	
	// calculate the inertia tensor in the local frame 
 	rsb->setInertiaProps();

	// other internal initialization
	rsb->internalInitialization();
}

// read in a vector from the binary file
void btReducedDeformableBodyHelpers::readBinaryVec(btReducedDeformableBody::tDenseArray& vec, 
																				  	 const unsigned int n_size, 				// #entries read
																						 const char* file)
{
	std::ifstream f_in(file, std::ios::in | std::ios::binary);
	// first get size
	unsigned int size=0;
	f_in.read((char*)&size, 4);//sizeof(unsigned int));
	btAssert(size >= n_size); 	// make sure the #requested mode is smaller than the #available modes

	// read data
	vec.resize(n_size);
	double temp;
	for (unsigned int i = 0; i < n_size; ++i)
	{
		f_in.read((char*)&temp, sizeof(double));
		vec[i] = btScalar(temp);
	}
  f_in.close();
}

// read in a matrix from the binary file
void btReducedDeformableBodyHelpers::readBinaryMat(btReducedDeformableBody::tDenseMatrix& mat, 
																						 const unsigned int n_modes, 		// #modes, outer array size
																						 const unsigned int n_full, 		// inner array size
																						 const char* file)
{
	std::ifstream f_in(file, std::ios::in | std::ios::binary);
	// first get size
	unsigned int v_size=0;
	f_in.read((char*)&v_size, 4);//sizeof(unsigned int));
	btAssert(v_size >= n_modes * n_full); 	// make sure the #requested mode is smaller than the #available modes

	// read data
	mat.resize(n_modes);
	for (int i = 0; i < n_modes; ++i) 
	{
		for (int j = 0; j < n_full; ++j)
		{
			double temp;
			f_in.read((char*)&temp, sizeof(double));

			if (mat[i].size() != n_modes)
				mat[i].resize(n_full);
			mat[i][j] = btScalar(temp);
		}
	}
  f_in.close();
}

void btReducedDeformableBodyHelpers::calculateLocalInertia(btVector3& inertia, const btScalar mass, const btVector3& half_extents, const btVector3& margin)
{
	btScalar lx = btScalar(2.) * (half_extents[0] + margin[0]);
	btScalar ly = btScalar(2.) * (half_extents[1] + margin[1]);
	btScalar lz = btScalar(2.) * (half_extents[2] + margin[2]);

	inertia.setValue(mass / (btScalar(12.0)) * (ly * ly + lz * lz),
								   mass / (btScalar(12.0)) * (lx * lx + lz * lz),
								   mass / (btScalar(12.0)) * (lx * lx + ly * ly));
}

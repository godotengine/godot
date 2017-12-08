// Copyright NVIDIA Corporation 2006 -- Ignacio Castano <icastano@nvidia.com>

#include <nvmesh/TriMesh.h>
#include <nvmesh/QuadTriMesh.h>

#include <nvmesh/weld/VertexWeld.h>
#include <nvmesh/weld/Weld.h>

using namespace nv;

// Weld trimesh vertices
void nv::WeldVertices(TriMesh * mesh)
{
	nvDebug("--- Welding vertices.\n");
	
	nvCheck(mesh != NULL);

	uint count = mesh->vertexCount();
	Array<uint> xrefs;
	Weld<TriMesh::Vertex> weld;
	uint newCount = weld(mesh->vertices(), xrefs);
	
	nvDebug("---   %d vertices welded\n", count - newCount);
	
	
	// Remap faces.
	const uint faceCount = mesh->faceCount();
	for(uint f = 0; f < faceCount; f++)
	{
		TriMesh::Face & face = mesh->faceAt(f);
		face.v[0] = xrefs[face.v[0]];
		face.v[1] = xrefs[face.v[1]];
		face.v[2] = xrefs[face.v[2]];
	}
}


// Weld trimesh vertices
void nv::WeldVertices(QuadTriMesh * mesh)
{
	nvDebug("--- Welding vertices.\n");
	
	nvCheck(mesh != NULL);

	uint  count = mesh->vertexCount();
	Array<uint> xrefs;
	Weld<TriMesh::Vertex> weld;
	uint newCount = weld(mesh->vertices(), xrefs);
	
	nvDebug("---   %d vertices welded\n", count - newCount);
	
	// Remap faces.
	const uint faceCount = mesh->faceCount();
	for(uint f = 0; f < faceCount; f++)
	{
		QuadTriMesh::Face & face = mesh->faceAt(f);
		face.v[0] = xrefs[face.v[0]];
		face.v[1] = xrefs[face.v[1]];
		face.v[2] = xrefs[face.v[2]];
		
		if (face.isQuadFace())
		{
			face.v[3] = xrefs[face.v[3]];
		}
	}
}



// OLD code

#if 0

namespace {

struct VertexInfo {
	uint id;			///< Original vertex id.
	uint normal_face_group;
	uint tangent_face_group;
	uint material;
	uint chart;
};


/// VertexInfo hash functor.
struct VertexHash : public IHashFunctor<VertexInfo> {
	VertexHash(PiMeshPtr m) : mesh(m) {
		uint c = mesh->FindChannel(VS_POS);
		piCheck(c != PI_NULL_INDEX);
		channel = mesh->GetChannel(c);
		piCheck(channel != NULL);
	}

	uint32 operator () (const VertexInfo & v) const {
		return channel->data[v.id].GetHash();
	}
	
private:
	PiMeshPtr mesh;
	PiMesh::Channel * channel;
};


/// VertexInfo comparator.
struct VertexEqual : public IBinaryPredicate<VertexInfo> {
	VertexEqual(PiMeshPtr m) : mesh(m) {}
	
	bool operator () (const VertexInfo & a, const VertexInfo & b) const {

		bool equal = a.normal_face_group == b.normal_face_group && 
			a.tangent_face_group == b.tangent_face_group &&
			a.material == b.material && 
			a.chart == b.chart;
		
		// Split vertex shared by different face types.
		if( !equal ) {
			return false;
		}
		
		// They were the same vertex.
		if( a.id == b.id ) {
			return true;
		}
		
		// Vertex equal if all the channels are equal.
		return mesh->IsVertexEqual(a.id, b.id);
	}

private:	
	PiMeshPtr mesh;
};

} // namespace


/// Weld the vertices.
void PiMeshVertexWeld::WeldVertices(const PiMeshSmoothGroup * mesh_smooth_group, 
	const PiMeshMaterial * mesh_material, const PiMeshAtlas * mesh_atlas ) 
{
	piDebug( "--- Welding vertices:\n" );

	piDebug( "---   Expand mesh vertices.\n" );
	PiArray<VertexInfo> vertex_array;

	const uint face_num = mesh->GetFaceNum();
	const uint vertex_max = face_num * 3;
	vertex_array.Resize( vertex_max );

	for(uint i = 0; i < vertex_max; i++) {

		uint f = i/3;
	
		const PiMesh::Face & face = mesh->GetFace(f);
		vertex_array[i].id = face.v[i%3];

		// Reset face attributes.
		vertex_array[i].normal_face_group = PI_NULL_INDEX;
		vertex_array[i].tangent_face_group = PI_NULL_INDEX;
		vertex_array[i].material = PI_NULL_INDEX;
		vertex_array[i].chart = PI_NULL_INDEX;
		
		// Set available attributes.
		if( mesh_smooth_group != NULL ) {
			if( mesh_smooth_group->HasNormalFaceGroups() ) {
				vertex_array[i].normal_face_group = mesh_smooth_group->GetNormalFaceGroup( f );
			}
			if( mesh_smooth_group->HasTangentFaceGroups() ) {
				vertex_array[i].tangent_face_group = mesh_smooth_group->GetTangentFaceGroup( f );
			}
		}
		if( mesh_material != NULL ) {
			vertex_array[i].material = mesh_material->GetFaceMaterial( f );
		}
		if( mesh_atlas != NULL && mesh_atlas->HasCharts() ) {
			vertex_array[i].chart = mesh_atlas->GetFaceChart( f );
		}
	}
	piDebug( "---   %d vertices.\n", vertex_max );

	piDebug( "---   Collapse vertices.\n" );

	uint * xrefs = new uint[vertex_max];
	VertexHash hash(mesh);
	VertexEqual equal(mesh);
	const uint vertex_num = Weld( vertex_array, xrefs, hash, equal );
	piCheck(vertex_num <= vertex_max);
	piDebug( "---   %d vertices.\n", vertex_num );	
	
	// Remap face indices.
	piDebug( "---   Remapping face indices.\n" );
	mesh->RemapFaceIndices(vertex_max, xrefs);


	// Overwrite xrefs to map new vertices to old vertices.
	for(uint v = 0; v < vertex_num; v++) {
		xrefs[v] = vertex_array[v].id;
	}
	
	// Update vertex order.
	mesh->ReorderVertices(vertex_num, xrefs);

	delete [] xrefs;
}

#endif // 0

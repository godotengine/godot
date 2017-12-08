// This code is in the public domain -- castanyo@yahoo.es

#include <nvcore/RadixSort.h>

#include <nvmesh/weld/Snap.h>
#include <nvmesh/TriMesh.h>
#include <nvmesh/geometry/Bounds.h>

using namespace nv;

namespace {
	
	// Snap the given vertices.
	void Snap(TriMesh::Vertex & a, TriMesh::Vertex & b, float texThreshold, float norThreshold)
	{
		a.pos = b.pos = (a.pos + b.pos) * 0.5f;
		
		if (equal(a.tex.x, b.tex.x, texThreshold) && equal(a.tex.y, b.tex.y, texThreshold)) {
			b.tex = a.tex = (a.tex + b.tex) * 0.5f;
		}
		
		if (equal(a.nor.x, b.nor.x, norThreshold) && equal(a.nor.y, b.nor.y, norThreshold) && equal(a.nor.z, b.nor.z, norThreshold)) {
			b.nor = a.nor = (a.nor + b.nor) * 0.5f;
		}
	};

} // nv namespace

uint nv::SnapVertices(TriMesh * mesh, float posThreshold, float texThreshold, float norThreshold)
{
	nvDebug("--- Snapping vertices.\n");
	
	// Determine largest axis.
	Box box = MeshBounds::box(mesh);
	Vector3 extents = box.extents();

	int axis = 2;
	if( extents.x > extents.y ) {
		if( extents.x > extents.z ) {
			axis = 0;
		}
	}
	else if(extents.y > extents.z) {
		axis = 1;
	}
	
	// @@ Use diagonal instead!
	

	// Sort vertices according to the largest axis.
	const uint vertexCount = mesh->vertexCount();
	nvCheck(vertexCount > 2); // Must have at least two vertices.

	// Get pos channel.
	//PiMesh::Channel * pos_channel = mesh->GetChannel(mesh->FindChannel(VS_POS));
	//nvCheck( pos_channel != NULL );

	//const PiArray<Vec4> & pos_array = pos_channel->data;

	Array<float> distArray;
	distArray.resize(vertexCount);

	for(uint v = 0; v < vertexCount; v++) {
		if (axis == 0) distArray[v] = mesh->vertexAt(v).pos.x;
		else if (axis == 1) distArray[v] = mesh->vertexAt(v).pos.y;
		else distArray[v] = mesh->vertexAt(v).pos.z;
	}

	RadixSort radix;
	const uint * xrefs = radix.sort(distArray.buffer(), distArray.count()).ranks();
	nvCheck(xrefs != NULL);

	uint snapCount = 0;
	for(uint v = 0; v < vertexCount-1; v++) {
		for(uint n = v+1; n < vertexCount; n++) {
			nvDebugCheck( distArray[xrefs[v]] <= distArray[xrefs[n]] );
			
			if (fabs(distArray[xrefs[n]] - distArray[xrefs[v]]) > posThreshold) {
				break;
			}
			
			TriMesh::Vertex & v0 = mesh->vertexAt(xrefs[v]);
			TriMesh::Vertex & v1 = mesh->vertexAt(xrefs[n]);
			
			const float dist = length(v0.pos - v1.pos);
			
			if (dist <= posThreshold) {
				Snap(v0, v1, texThreshold, norThreshold);
				snapCount++;
			}
		}
	}

	// @@ todo: debug, make sure that the distance between vertices is now >= threshold

	nvDebug("---   %u vertices snapped\n", snapCount);

	return snapCount;
};


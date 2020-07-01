// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "scene_subdiv_mesh.h"
#include "scene.h"
#include "../subdiv/patch_eval.h"
#include "../subdiv/patch_eval_simd.h"

#include "../../common/algorithms/parallel_sort.h"
#include "../../common/algorithms/parallel_prefix_sum.h"
#include "../../common/algorithms/parallel_for.h"

namespace embree
{
#if defined(EMBREE_LOWEST_ISA)

  SubdivMesh::SubdivMesh (Device* device)
    : Geometry(device,GTY_SUBDIV_MESH,0,1), 
      displFunc(nullptr),
      tessellationRate(2.0f),
      numHalfEdges(0),
      faceStartEdge(device,0),
      halfEdgeFace(device,0),
      invalid_face(device,0),
      commitCounter(0)
  {
    
    vertices.resize(numTimeSteps);
    vertex_buffer_tags.resize(numTimeSteps);
    topology.resize(1);
    topology[0] = Topology(this);
  }

  void SubdivMesh::enabling() 
  { 
    scene->numSubdivEnableDisableEvents++;
    if (numTimeSteps == 1) scene->world.numSubdivPatches += numPrimitives;
    else                   scene->worldMB.numSubdivPatches += numPrimitives;
  }
  
  void SubdivMesh::disabling() 
  { 
    scene->numSubdivEnableDisableEvents++;
    if (numTimeSteps == 1) scene->world.numSubdivPatches -= numPrimitives;
    else                   scene->worldMB.numSubdivPatches -= numPrimitives;
  }

  void SubdivMesh::setMask (unsigned mask) 
  {
    this->mask = mask; 
    Geometry::update();
  }

  void SubdivMesh::setSubdivisionMode (unsigned topologyID, RTCSubdivisionMode mode)
  {
    if (topologyID >= topology.size())
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"invalid topology ID");
    topology[topologyID].setSubdivisionMode(mode);
    Geometry::update();
  }

  void SubdivMesh::setVertexAttributeTopology(unsigned int vertexAttribID, unsigned int topologyID)
  {
    if (vertexAttribID < vertexAttribs.size()){
      if (topologyID < topology.size()) {
        if ((unsigned)vertexAttribs[vertexAttribID].userData != topologyID) {
          vertexAttribs[vertexAttribID].userData = topologyID;
          commitCounter++; // triggers recalculation of cached interpolation data
        }
      } else {
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid topology specified");
      }
    } else {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex attribute specified");
    }
  }

  void SubdivMesh::setNumTimeSteps (unsigned int numTimeSteps)
  {
    vertices.resize(numTimeSteps);
    vertex_buffer_tags.resize(numTimeSteps);
    Geometry::setNumTimeSteps(numTimeSteps);
  }

  void SubdivMesh::setVertexAttributeCount (unsigned int N)
  {
    vertexAttribs.resize(N);
    vertex_attrib_buffer_tags.resize(N);
    Geometry::update();
  }

  void SubdivMesh::setTopologyCount (unsigned int N)
  {
    if (N == 0)
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT,"at least one topology has to exist")
        
    size_t begin = topology.size();
    topology.resize(N);
    for (size_t i = begin; i < topology.size(); i++)
      topology[i] = Topology(this);
  }
  
  void SubdivMesh::setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num)
  { 
    /* verify that all accesses are 4 bytes aligned */
    if (((size_t(buffer->getPtr()) + offset) & 0x3) || (stride & 0x3))
      throw_RTCError(RTC_ERROR_INVALID_OPERATION, "data must be 4 bytes aligned");

    if (type != RTC_BUFFER_TYPE_LEVEL)
      commitCounter++;

    if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (format != RTC_FORMAT_FLOAT3)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex buffer format");

      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid vertex buffer slot");

      vertices[slot].set(buffer, offset, stride, num, format);
      vertices[slot].checkPadding16();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
    {
      if (format < RTC_FORMAT_FLOAT || format > RTC_FORMAT_FLOAT16)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex attribute buffer format");

      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex attribute buffer slot");
      
      vertexAttribs[slot].set(buffer, offset, stride, num, format);
      vertexAttribs[slot].checkPadding16();
    }
    else if (type == RTC_BUFFER_TYPE_FACE)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      if (format != RTC_FORMAT_UINT)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid face buffer format");

      faceVertices.set(buffer, offset, stride, num, format);
      setNumPrimitives(num);
    }
    else if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (format != RTC_FORMAT_UINT)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid face buffer format");

      if (slot >= topology.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid index buffer slot");

      topology[slot].vertexIndices.set(buffer, offset, stride, num, format);
    }
    else if (type == RTC_BUFFER_TYPE_EDGE_CREASE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      if (format != RTC_FORMAT_UINT2)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid edge crease index buffer format");

      edge_creases.set(buffer, offset, stride, num, format);
    }
    else if (type == RTC_BUFFER_TYPE_EDGE_CREASE_WEIGHT)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      if (format != RTC_FORMAT_FLOAT)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid edge crease weight buffer format");

      edge_crease_weights.set(buffer, offset, stride, num, format);
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_CREASE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      if (format != RTC_FORMAT_UINT)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex crease index buffer format");

      vertex_creases.set(buffer, offset, stride, num, format);
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_CREASE_WEIGHT)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      if (format != RTC_FORMAT_FLOAT)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex crease weight buffer format");

      vertex_crease_weights.set(buffer, offset, stride, num, format);
    }
    else if (type == RTC_BUFFER_TYPE_HOLE)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      if (format != RTC_FORMAT_UINT)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid hole buffer format");

      holes.set(buffer, offset, stride, num, format);
    }
    else if (type == RTC_BUFFER_TYPE_LEVEL)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      if (format != RTC_FORMAT_FLOAT)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid level buffer format");

      levels.set(buffer, offset, stride, num, format);
    }
    else
    {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT,"unknown buffer type");
    }
  }

  void* SubdivMesh::getBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return vertices[slot].getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
    {
      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return vertexAttribs[slot].getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_FACE)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return faceVertices.getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (slot >= topology.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return topology[slot].vertexIndices.getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_EDGE_CREASE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return edge_creases.getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_EDGE_CREASE_WEIGHT)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return edge_crease_weights.getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_CREASE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return vertex_creases.getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_CREASE_WEIGHT)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return vertex_crease_weights.getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_HOLE)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return holes.getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_LEVEL)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return levels.getPtr();
    }
    else
    {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
      return nullptr;
    }
  }

  void SubdivMesh::updateBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type != RTC_BUFFER_TYPE_LEVEL)
      commitCounter++;

    if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertices[slot].setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
    {
      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertexAttribs[slot].setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_FACE)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      faceVertices.setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (slot >= topology.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      topology[slot].vertexIndices.setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_EDGE_CREASE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      edge_creases.setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_EDGE_CREASE_WEIGHT)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      edge_crease_weights.setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_CREASE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertex_creases.setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_CREASE_WEIGHT)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertex_crease_weights.setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_HOLE)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      holes.setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_LEVEL)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      levels.setModified(true);
    }
    else
    {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
    }

    Geometry::update();
  }

  void SubdivMesh::setDisplacementFunction (RTCDisplacementFunctionN func) 
  {
    this->displFunc = func;
  }

  void SubdivMesh::setTessellationRate(float N)
  {
    tessellationRate = N;
    levels.setModified(true);
  }

  __forceinline uint64_t pair64(unsigned int x, unsigned int y) 
  {
    if (x<y) std::swap(x,y);
    return (((uint64_t)x) << 32) | (uint64_t)y;
  }

  SubdivMesh::Topology::Topology(SubdivMesh* mesh)
    : mesh(mesh), subdiv_mode(RTC_SUBDIVISION_MODE_SMOOTH_BOUNDARY), halfEdges(mesh->device,0)
  {
  }
  
  void SubdivMesh::Topology::setSubdivisionMode (RTCSubdivisionMode mode)
  {
    if (subdiv_mode == mode) return;
    subdiv_mode = mode;
    mesh->updateBuffer(RTC_BUFFER_TYPE_VERTEX_CREASE_WEIGHT, 0);
  }
  
  void SubdivMesh::Topology::update () {
    vertexIndices.setModified(true); 
  }

  bool SubdivMesh::Topology::verify (size_t numVertices) 
  {
    size_t ofs = 0;
    for (size_t i=0; i<mesh->size(); i++) 
    {
      int valence = mesh->faceVertices[i];
      for (size_t j=ofs; j<ofs+valence; j++) 
      {
        if (j >= vertexIndices.size())
          return false;
          
        if (vertexIndices[j] >= numVertices)
          return false; 
      }
      ofs += valence;
    }
    return true;
  }

  void SubdivMesh::Topology::calculateHalfEdges()
  {
    const size_t blockSize = 4096;
    const size_t numEdges = mesh->numEdges();
    const size_t numFaces = mesh->numFaces();
    const size_t numHalfEdges = mesh->numHalfEdges;

    /* allocate temporary array */
    halfEdges0.resize(numEdges);
    halfEdges1.resize(numEdges);

    /* create all half edges */
    parallel_for( size_t(0), numFaces, blockSize, [&](const range<size_t>& r) 
    {
      for (size_t f=r.begin(); f<r.end(); f++) 
      {
	const unsigned N = mesh->faceVertices[f];
	const unsigned e = mesh->faceStartEdge[f];

	for (unsigned de=0; de<N; de++)
	{
	  HalfEdge* edge = &halfEdges[e+de];
          int nextOfs = (de == (N-1)) ? -int(N-1) : +1;
          int prevOfs = (de ==     0) ? +int(N-1) : -1;
	  
	  const unsigned int startVertex = vertexIndices[e+de];
          const unsigned int endVertex = vertexIndices[e+de+nextOfs]; 
	  const uint64_t key = SubdivMesh::Edge(startVertex,endVertex);

          /* we always have to use the geometry topology to lookup creases */
          const unsigned int startVertex0 = mesh->topology[0].vertexIndices[e+de];
          const unsigned int endVertex0 = mesh->topology[0].vertexIndices[e+de+nextOfs]; 
	  const uint64_t key0 = SubdivMesh::Edge(startVertex0,endVertex0);
	  
	  edge->vtx_index              = startVertex;
	  edge->next_half_edge_ofs     = nextOfs;
	  edge->prev_half_edge_ofs     = prevOfs;
	  edge->opposite_half_edge_ofs = 0;
	  edge->edge_crease_weight     = mesh->edgeCreaseMap.lookup(key0,0.0f);
	  edge->vertex_crease_weight   = mesh->vertexCreaseMap.lookup(startVertex0,0.0f);
	  edge->edge_level             = mesh->getEdgeLevel(e+de);
          edge->patch_type             = HalfEdge::COMPLEX_PATCH; // type gets updated below
          edge->vertex_type            = HalfEdge::REGULAR_VERTEX;

          if (unlikely(mesh->holeSet.lookup(unsigned(f)))) 
	    halfEdges1[e+de] = SubdivMesh::KeyHalfEdge(std::numeric_limits<uint64_t>::max(),edge);
	  else
	    halfEdges1[e+de] = SubdivMesh::KeyHalfEdge(key,edge);
	}
      }
    });

    /* sort half edges to find adjacent edges */
    radix_sort_u64(halfEdges1.data(),halfEdges0.data(),numHalfEdges);

    /* link all adjacent pairs of edges */
    parallel_for( size_t(0), numHalfEdges, blockSize, [&](const range<size_t>& r) 
    {
      /* skip if start of adjacent edges was not in our range */
      size_t e=r.begin();
      if (e != 0 && (halfEdges1[e].key == halfEdges1[e-1].key)) {
	const uint64_t key = halfEdges1[e].key;
	while (e<r.end() && halfEdges1[e].key == key) e++;
      }

      /* process all adjacent edges starting in our range */
      while (e<r.end())
      {
	const uint64_t key = halfEdges1[e].key;
	if (key == std::numeric_limits<uint64_t>::max()) break;
	size_t N=1; while (e+N<numHalfEdges && halfEdges1[e+N].key == key) N++;

        /* border edges are identified by not having an opposite edge set */
	if (N == 1) {
          halfEdges1[e].edge->edge_crease_weight = float(inf);
	}

        /* standard edge shared between two faces */
        else if (N == 2)
        {
          /* create edge crease if winding order mismatches between neighboring patches */
          if (halfEdges1[e+0].edge->next()->vtx_index != halfEdges1[e+1].edge->vtx_index)
          {
            halfEdges1[e+0].edge->edge_crease_weight = float(inf);
            halfEdges1[e+1].edge->edge_crease_weight = float(inf);
          }
          /* otherwise mark edges as opposites of each other */
          else {
            halfEdges1[e+0].edge->setOpposite(halfEdges1[e+1].edge);
            halfEdges1[e+1].edge->setOpposite(halfEdges1[e+0].edge);
          }
	}

        /* non-manifold geometry is handled by keeping vertices fixed during subdivision */
        else {
	  for (size_t i=0; i<N; i++) {
	    halfEdges1[e+i].edge->vertex_crease_weight = inf;
            halfEdges1[e+i].edge->vertex_type = HalfEdge::NON_MANIFOLD_EDGE_VERTEX;
            halfEdges1[e+i].edge->edge_crease_weight = inf;

	    halfEdges1[e+i].edge->next()->vertex_crease_weight = inf;
            halfEdges1[e+i].edge->next()->vertex_type = HalfEdge::NON_MANIFOLD_EDGE_VERTEX;
            halfEdges1[e+i].edge->next()->edge_crease_weight = inf;
	  }
	}
	e+=N;
      }
    });

    /* set subdivision mode and calculate patch types */
    parallel_for( size_t(0), numFaces, blockSize, [&](const range<size_t>& r) 
    {
      for (size_t f=r.begin(); f<r.end(); f++) 
      {
        HalfEdge* edge = &halfEdges[mesh->faceStartEdge[f]];

        /* for vertex topology we also test if vertices are valid */
        if (this == &mesh->topology[0])
        {
          /* calculate if face is valid */
          for (size_t t=0; t<mesh->numTimeSteps; t++)
            mesh->invalidFace(f,t) = !edge->valid(mesh->vertices[t]) || mesh->holeSet.lookup(unsigned(f));
        }

        /* pin some edges and vertices */
        for (size_t i=0; i<mesh->faceVertices[f]; i++) 
        {
          /* pin corner vertices when requested by user */
          if (subdiv_mode == RTC_SUBDIVISION_MODE_PIN_CORNERS && edge[i].isCorner())
            edge[i].vertex_crease_weight = float(inf);
          
          /* pin all border vertices when requested by user */
          else if (subdiv_mode == RTC_SUBDIVISION_MODE_PIN_BOUNDARY && edge[i].vertexHasBorder()) 
            edge[i].vertex_crease_weight = float(inf);

          /* pin all edges and vertices when requested by user */
          else if (subdiv_mode == RTC_SUBDIVISION_MODE_PIN_ALL) {
            edge[i].edge_crease_weight = float(inf);
            edge[i].vertex_crease_weight = float(inf);
          }
        }

        /* we have to calculate patch_type last! */
        HalfEdge::PatchType patch_type = edge->patchType();
        for (size_t i=0; i<mesh->faceVertices[f]; i++) 
          edge[i].patch_type = patch_type;
      }
    });
  }

  void SubdivMesh::Topology::updateHalfEdges()
  {
    /* we always use the geometry topology to lookup creases */
    mvector<HalfEdge>& halfEdgesGeom = mesh->topology[0].halfEdges;

    /* assume we do no longer recalculate in the future and clear these arrays */
    halfEdges0.clear();
    halfEdges1.clear();

    /* calculate which data to update */
    const bool updateEdgeCreases   = mesh->topology[0].vertexIndices.isModified() || mesh->edge_creases.isModified()   || mesh->edge_crease_weights.isModified();
    const bool updateVertexCreases = mesh->topology[0].vertexIndices.isModified() || mesh->vertex_creases.isModified() || mesh->vertex_crease_weights.isModified(); 
    const bool updateLevels = mesh->levels.isModified();

    /* parallel loop over all half edges */
    parallel_for( size_t(0), mesh->numHalfEdges, size_t(4096), [&](const range<size_t>& r) 
    {
      for (size_t i=r.begin(); i!=r.end(); i++)
      {
	HalfEdge& edge = halfEdges[i];

	if (updateLevels)
	  edge.edge_level = mesh->getEdgeLevel(i); 
        
	if (updateEdgeCreases) {
	  if (edge.hasOpposite()) // leave weight at inf for borders
            edge.edge_crease_weight = mesh->edgeCreaseMap.lookup((uint64_t)halfEdgesGeom[i].getEdge(),0.0f);
	}
        
        /* we only use user specified vertex_crease_weight if the vertex is manifold */
        if (updateVertexCreases && edge.vertex_type != HalfEdge::NON_MANIFOLD_EDGE_VERTEX) 
        {
	  edge.vertex_crease_weight = mesh->vertexCreaseMap.lookup(halfEdgesGeom[i].vtx_index,0.0f);

          /* pin corner vertices when requested by user */
          if (subdiv_mode == RTC_SUBDIVISION_MODE_PIN_CORNERS && edge.isCorner())
            edge.vertex_crease_weight = float(inf);
          
          /* pin all border vertices when requested by user */
          else if (subdiv_mode == RTC_SUBDIVISION_MODE_PIN_BOUNDARY && edge.vertexHasBorder()) 
            edge.vertex_crease_weight = float(inf);

          /* pin every vertex when requested by user */
          else if (subdiv_mode == RTC_SUBDIVISION_MODE_PIN_ALL) {
            edge.edge_crease_weight = float(inf);
            edge.vertex_crease_weight = float(inf);
          }
        }

        /* update patch type */
        if (updateEdgeCreases || updateVertexCreases) {
          edge.patch_type = edge.patchType();
        }
      }
    });
  }

  void SubdivMesh::Topology::initializeHalfEdgeStructures ()
  {
    /* if vertex indices not set we ignore this topology */
    if (!vertexIndices)
      return;

    /* allocate half edge array */
    halfEdges.resize(mesh->numEdges());

    /* check if we have to recalculate the half edges */
    bool recalculate = false;
    recalculate |= vertexIndices.isModified(); 
    recalculate |= mesh->faceVertices.isModified();
    recalculate |= mesh->holes.isModified();

    /* check if we can simply update the half edges */
    bool update = false;
    update |= mesh->topology[0].vertexIndices.isModified(); // we use this buffer to copy creases to interpolation topologies
    update |= mesh->edge_creases.isModified();
    update |= mesh->edge_crease_weights.isModified();
    update |= mesh->vertex_creases.isModified();
    update |= mesh->vertex_crease_weights.isModified(); 
    update |= mesh->levels.isModified();

    /* now either recalculate or update the half edges */
    if (recalculate) calculateHalfEdges();
    else if (update) updateHalfEdges();
   
    /* cleanup some state for static scenes */
    if (mesh->scene == nullptr || mesh->scene->isStaticAccel()) 
    {
      halfEdges0.clear();
      halfEdges1.clear();
    }

    /* clear modified state of all buffers */
    vertexIndices.setModified(false); 
  }

  void SubdivMesh::printStatistics()
  {
    size_t numBilinearFaces = 0;
    size_t numRegularQuadFaces = 0;
    size_t numIrregularQuadFaces = 0;
    size_t numComplexFaces = 0;
    
    for (size_t e=0, f=0; f<numFaces(); e+=faceVertices[f++]) 
    {
      switch (topology[0].halfEdges[e].patch_type) {
      case HalfEdge::BILINEAR_PATCH      : numBilinearFaces++;   break;
      case HalfEdge::REGULAR_QUAD_PATCH  : numRegularQuadFaces++;   break;
      case HalfEdge::IRREGULAR_QUAD_PATCH: numIrregularQuadFaces++; break;
      case HalfEdge::COMPLEX_PATCH       : numComplexFaces++;   break;
      }
    }
    
    std::cout << "numFaces = " << numFaces() << ", " 
              << "numBilinearFaces = " << numBilinearFaces << " (" << 100.0f * numBilinearFaces / numFaces() << "%), " 
              << "numRegularQuadFaces = " << numRegularQuadFaces << " (" << 100.0f * numRegularQuadFaces / numFaces() << "%), " 
              << "numIrregularQuadFaces " << numIrregularQuadFaces << " (" << 100.0f * numIrregularQuadFaces / numFaces() << "%) " 
              << "numComplexFaces " << numComplexFaces << " (" << 100.0f * numComplexFaces / numFaces() << "%) " 
              << std::endl;
  }

  void SubdivMesh::initializeHalfEdgeStructures ()
  {
    double t0 = getSeconds();

    invalid_face.resize(numFaces()*numTimeSteps);
 
    /* calculate start edge of each face */
    faceStartEdge.resize(numFaces());
    
    if (faceVertices.isModified())
    {
      numHalfEdges = parallel_prefix_sum(faceVertices,faceStartEdge,numFaces(),0,std::plus<unsigned>());

      /* calculate face of each half edge */
      halfEdgeFace.resize(numHalfEdges);
      for (size_t f=0, h=0; f<numFaces(); f++)
        for (size_t e=0; e<faceVertices[f]; e++)
          halfEdgeFace[h++] = (unsigned int) f;
    }
    
    /* create set with all vertex creases */
    if (vertex_creases.isModified() || vertex_crease_weights.isModified())
      vertexCreaseMap.init(vertex_creases,vertex_crease_weights);
    
    /* create map with all edge creases */
    if (edge_creases.isModified() || edge_crease_weights.isModified())
      edgeCreaseMap.init(edge_creases,edge_crease_weights);

    /* create set with all holes */
    if (holes.isModified())
      holeSet.init(holes);

    /* create topology */
    for (auto& t: topology)
      t.initializeHalfEdgeStructures();

    /* create interpolation cache mapping for interpolatable meshes */
    for (size_t i=0; i<vertex_buffer_tags.size(); i++)
      vertex_buffer_tags[i].resize(numFaces()*numInterpolationSlots4(vertices[i].getStride()));
    for (size_t i=0; i<vertexAttribs.size(); i++)
      if (vertexAttribs[i]) vertex_attrib_buffer_tags[i].resize(numFaces()*numInterpolationSlots4(vertexAttribs[i].getStride()));

    /* cleanup some state for static scenes */
    if (scene == nullptr || scene->isStaticAccel()) 
    {
      vertexCreaseMap.clear();
      edgeCreaseMap.clear();
    }

    /* clear modified state of all buffers */
    faceVertices.setModified(false);
    holes.setModified(false);
    for (auto& buffer : vertices) buffer.setModified(false); 
    levels.setModified(false);
    edge_creases.setModified(false);
    edge_crease_weights.setModified(false);
    vertex_creases.setModified(false);
    vertex_crease_weights.setModified(false);

    double t1 = getSeconds();

    /* print statistics in verbose mode */
    if (device->verbosity(2)) {
      std::cout << "half edge generation = " << 1000.0*(t1-t0) << "ms, " << 1E-6*double(numHalfEdges)/(t1-t0) << "M/s" << std::endl;
      printStatistics();
    }
  }

  bool SubdivMesh::verify () 
  {
    /*! verify consistent size of vertex arrays */
    if (vertices.size() == 0) return false;
    for (const auto& buffer : vertices)
      if (buffer.size() != numVertices())
        return false;

    /*! verify vertex indices */
    if (!topology[0].verify(numVertices()))
      return false;

    for (auto& b : vertexAttribs)
      if (!topology[b.userData].verify(b.size()))
        return false;

    /*! verify vertices */
    for (const auto& buffer : vertices)
      for (size_t i=0; i<buffer.size(); i++)
	if (!isvalid(buffer[i])) 
	  return false;

    return true;
  }

  void SubdivMesh::commit () 
  {
    initializeHalfEdgeStructures();
    Geometry::commit();
  }

  unsigned int SubdivMesh::getFirstHalfEdge(unsigned int faceID)
  {
    if (faceID >= numFaces())
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid face");

    return faceStartEdge[faceID];
  }

  unsigned int SubdivMesh::getFace(unsigned int edgeID)
  {
    if (edgeID >= numHalfEdges)
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid edge");

    return halfEdgeFace[edgeID];
  }
    
  unsigned int SubdivMesh::getNextHalfEdge(unsigned int edgeID)
  {
    if (edgeID >= numHalfEdges)
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid half edge");

    return edgeID + topology[0].halfEdges[edgeID].next_half_edge_ofs;
  }

  unsigned int SubdivMesh::getPreviousHalfEdge(unsigned int edgeID)
  {
     if (edgeID >= numHalfEdges)
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid half edge");

    return edgeID + topology[0].halfEdges[edgeID].prev_half_edge_ofs;
  }

  unsigned int SubdivMesh::getOppositeHalfEdge(unsigned int topologyID, unsigned int edgeID)
  {
    if (topologyID >= topology.size())
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid topology");
    
    if (edgeID >= numHalfEdges)
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid half edge");

    return edgeID + topology[topologyID].halfEdges[edgeID].opposite_half_edge_ofs;
  }
  
#endif

  namespace isa
  {
    SubdivMesh* createSubdivMesh(Device* device) {
      return new SubdivMeshISA(device);
    }
    
    void SubdivMeshISA::interpolate(const RTCInterpolateArguments* const args)
    {
      unsigned int primID = args->primID;
      float u = args->u;
      float v = args->v;
      RTCBufferType bufferType = args->bufferType;
      unsigned int bufferSlot = args->bufferSlot;
      float* P = args->P;
      float* dPdu = args->dPdu;
      float* dPdv = args->dPdv;
      float* ddPdudu = args->ddPdudu;
      float* ddPdvdv = args->ddPdvdv;
      float* ddPdudv = args->ddPdudv;
      unsigned int valueCount = args->valueCount;
      
      /* calculate base pointer and stride */
      assert((bufferType == RTC_BUFFER_TYPE_VERTEX && bufferSlot < RTC_MAX_TIME_STEP_COUNT) ||
             (bufferType == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE && bufferSlot < RTC_MAX_USER_VERTEX_BUFFERS));
      const char* src = nullptr; 
      size_t stride = 0;
      std::vector<SharedLazyTessellationCache::CacheEntry>* baseEntry = nullptr;
      Topology* topo = nullptr;
      if (bufferType == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE) {
        assert(bufferSlot < vertexAttribs.size());
        src    = vertexAttribs[bufferSlot].getPtr();
        stride = vertexAttribs[bufferSlot].getStride();
        baseEntry = &vertex_attrib_buffer_tags[bufferSlot];
        int topologyID = vertexAttribs[bufferSlot].userData;
        topo = &topology[topologyID];
      } else {
        assert(bufferSlot < numTimeSteps);
        src    = vertices[bufferSlot].getPtr();
        stride = vertices[bufferSlot].getStride();
        baseEntry = &vertex_buffer_tags[bufferSlot];
        topo = &topology[0];
      }
      
      bool has_P = P;
      bool has_dP = dPdu;     assert(!has_dP  || dPdv);
      bool has_ddP = ddPdudu; assert(!has_ddP || (ddPdvdv && ddPdudu));
      
      for (unsigned int i=0; i<valueCount; i+=4)
      {
        vfloat4 Pt, dPdut, dPdvt, ddPdudut, ddPdvdvt, ddPdudvt;
        isa::PatchEval<vfloat4,vfloat4>(baseEntry->at(interpolationSlot(primID,i/4,stride)),commitCounter,
                                        topo->getHalfEdge(primID),src+i*sizeof(float),stride,u,v,
                                        has_P ? &Pt : nullptr, 
                                        has_dP ? &dPdut : nullptr, 
                                        has_dP ? &dPdvt : nullptr,
                                        has_ddP ? &ddPdudut : nullptr, 
                                        has_ddP ? &ddPdvdvt : nullptr, 
                                        has_ddP ? &ddPdudvt : nullptr);
        
        if (has_P) {
          for (size_t j=i; j<min(i+4,valueCount); j++) 
            P[j] = Pt[j-i];
        }
        if (has_dP) 
        {
          for (size_t j=i; j<min(i+4,valueCount); j++) {
            dPdu[j] = dPdut[j-i];
            dPdv[j] = dPdvt[j-i];
          }
        }
        if (has_ddP) 
        {
          for (size_t j=i; j<min(i+4,valueCount); j++) {
            ddPdudu[j] = ddPdudut[j-i];
            ddPdvdv[j] = ddPdvdvt[j-i];
            ddPdudv[j] = ddPdudvt[j-i];
          }
        }
      }
    }
    
    void SubdivMeshISA::interpolateN(const RTCInterpolateNArguments* const args)
    {
      const void* valid_i = args->valid;
      const unsigned* primIDs = args->primIDs;
      const float* u = args->u;
      const float* v = args->v;
      unsigned int N = args->N;
      RTCBufferType bufferType = args->bufferType;
      unsigned int bufferSlot = args->bufferSlot;
      float* P = args->P;
      float* dPdu = args->dPdu;
      float* dPdv = args->dPdv;
      float* ddPdudu = args->ddPdudu;
      float* ddPdvdv = args->ddPdvdv;
      float* ddPdudv = args->ddPdudv;
      unsigned int valueCount = args->valueCount;
    
      /* calculate base pointer and stride */
      assert((bufferType == RTC_BUFFER_TYPE_VERTEX && bufferSlot < RTC_MAX_TIME_STEP_COUNT) ||
             (bufferType == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE && bufferSlot < RTC_MAX_USER_VERTEX_BUFFERS));
      const char* src = nullptr; 
      size_t stride = 0;
      std::vector<SharedLazyTessellationCache::CacheEntry>* baseEntry = nullptr;
      Topology* topo = nullptr;
      if (bufferType == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE) {
        assert(bufferSlot < vertexAttribs.size());
        src    = vertexAttribs[bufferSlot].getPtr();
        stride = vertexAttribs[bufferSlot].getStride();
        baseEntry = &vertex_attrib_buffer_tags[bufferSlot];
        int topologyID = vertexAttribs[bufferSlot].userData;
        topo = &topology[topologyID];
      } else {
        assert(bufferSlot < numTimeSteps);
        src    = vertices[bufferSlot].getPtr();
        stride = vertices[bufferSlot].getStride();
        baseEntry = &vertex_buffer_tags[bufferSlot];
        topo = &topology[0];
      }
      
      const int* valid = (const int*) valid_i;
      
      for (size_t i=0; i<N; i+=4) 
      {
        vbool4 valid1 = vint4(int(i))+vint4(step) < vint4(int(N));
        if (valid) valid1 &= vint4::loadu(&valid[i]) == vint4(-1);
        if (none(valid1)) continue;
        
        const vuint4 primID = vuint4::loadu(&primIDs[i]);
        const vfloat4 uu = vfloat4::loadu(&u[i]);
        const vfloat4 vv = vfloat4::loadu(&v[i]);
        
        foreach_unique(valid1,primID,[&](const vbool4& valid1, const unsigned int primID)
                       {
                         for (unsigned int j=0; j<valueCount; j+=4) 
                         {
                           const size_t M = min(4u,valueCount-j);
                           isa::PatchEvalSimd<vbool4,vint4,vfloat4,vfloat4>(baseEntry->at(interpolationSlot(primID,j/4,stride)),commitCounter,
                                                                            topo->getHalfEdge(primID),src+j*sizeof(float),stride,valid1,uu,vv,
                                                                            P ? P+j*N+i : nullptr,
                                                                            dPdu ? dPdu+j*N+i : nullptr,
                                                                            dPdv ? dPdv+j*N+i : nullptr,
                                                                            ddPdudu ? ddPdudu+j*N+i : nullptr,
                                                                            ddPdvdv ? ddPdvdv+j*N+i : nullptr,
                                                                            ddPdudv ? ddPdudv+j*N+i : nullptr,
                                                                            N,M);
                         }
                       });
      }
    }
  }
}

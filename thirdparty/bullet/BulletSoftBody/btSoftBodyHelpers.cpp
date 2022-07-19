/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
///btSoftBodyHelpers.cpp by Nathanael Presson

#include "btSoftBodyInternals.h"
#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <string.h>
#include <algorithm>
#include "btSoftBodyHelpers.h"
#include "LinearMath/btConvexHull.h"
#include "LinearMath/btConvexHullComputer.h"
#include <map>
#include <vector>

static void drawVertex(btIDebugDraw* idraw,
					   const btVector3& x, btScalar s, const btVector3& c)
{
	idraw->drawLine(x - btVector3(s, 0, 0), x + btVector3(s, 0, 0), c);
	idraw->drawLine(x - btVector3(0, s, 0), x + btVector3(0, s, 0), c);
	idraw->drawLine(x - btVector3(0, 0, s), x + btVector3(0, 0, s), c);
}

//
static void drawBox(btIDebugDraw* idraw,
					const btVector3& mins,
					const btVector3& maxs,
					const btVector3& color)
{
	const btVector3 c[] = {btVector3(mins.x(), mins.y(), mins.z()),
						   btVector3(maxs.x(), mins.y(), mins.z()),
						   btVector3(maxs.x(), maxs.y(), mins.z()),
						   btVector3(mins.x(), maxs.y(), mins.z()),
						   btVector3(mins.x(), mins.y(), maxs.z()),
						   btVector3(maxs.x(), mins.y(), maxs.z()),
						   btVector3(maxs.x(), maxs.y(), maxs.z()),
						   btVector3(mins.x(), maxs.y(), maxs.z())};
	idraw->drawLine(c[0], c[1], color);
	idraw->drawLine(c[1], c[2], color);
	idraw->drawLine(c[2], c[3], color);
	idraw->drawLine(c[3], c[0], color);
	idraw->drawLine(c[4], c[5], color);
	idraw->drawLine(c[5], c[6], color);
	idraw->drawLine(c[6], c[7], color);
	idraw->drawLine(c[7], c[4], color);
	idraw->drawLine(c[0], c[4], color);
	idraw->drawLine(c[1], c[5], color);
	idraw->drawLine(c[2], c[6], color);
	idraw->drawLine(c[3], c[7], color);
}

//
static void drawTree(btIDebugDraw* idraw,
					 const btDbvtNode* node,
					 int depth,
					 const btVector3& ncolor,
					 const btVector3& lcolor,
					 int mindepth,
					 int maxdepth)
{
	if (node)
	{
		if (node->isinternal() && ((depth < maxdepth) || (maxdepth < 0)))
		{
			drawTree(idraw, node->childs[0], depth + 1, ncolor, lcolor, mindepth, maxdepth);
			drawTree(idraw, node->childs[1], depth + 1, ncolor, lcolor, mindepth, maxdepth);
		}
		if (depth >= mindepth)
		{
			const btScalar scl = (btScalar)(node->isinternal() ? 1 : 1);
			const btVector3 mi = node->volume.Center() - node->volume.Extents() * scl;
			const btVector3 mx = node->volume.Center() + node->volume.Extents() * scl;
			drawBox(idraw, mi, mx, node->isleaf() ? lcolor : ncolor);
		}
	}
}

//
template <typename T>
static inline T sum(const btAlignedObjectArray<T>& items)
{
	T v;
	if (items.size())
	{
		v = items[0];
		for (int i = 1, ni = items.size(); i < ni; ++i)
		{
			v += items[i];
		}
	}
	return (v);
}

//
template <typename T, typename Q>
static inline void add(btAlignedObjectArray<T>& items, const Q& value)
{
	for (int i = 0, ni = items.size(); i < ni; ++i)
	{
		items[i] += value;
	}
}

//
template <typename T, typename Q>
static inline void mul(btAlignedObjectArray<T>& items, const Q& value)
{
	for (int i = 0, ni = items.size(); i < ni; ++i)
	{
		items[i] *= value;
	}
}

//
template <typename T>
static inline T average(const btAlignedObjectArray<T>& items)
{
	const btScalar n = (btScalar)(items.size() > 0 ? items.size() : 1);
	return (sum(items) / n);
}

#if 0
//
 inline static btScalar		tetravolume(const btVector3& x0,
										const btVector3& x1,
										const btVector3& x2,
										const btVector3& x3)
{
	const btVector3	a=x1-x0;
	const btVector3	b=x2-x0;
	const btVector3	c=x3-x0;
	return(btDot(a,btCross(b,c)));
}
#endif

//
#if 0
static btVector3		stresscolor(btScalar stress)
{
	static const btVector3	spectrum[]=	{	btVector3(1,0,1),
		btVector3(0,0,1),
		btVector3(0,1,1),
		btVector3(0,1,0),
		btVector3(1,1,0),
		btVector3(1,0,0),
		btVector3(1,0,0)};
	static const int		ncolors=sizeof(spectrum)/sizeof(spectrum[0])-1;
	static const btScalar	one=1;
	stress=btMax<btScalar>(0,btMin<btScalar>(1,stress))*ncolors;
	const int				sel=(int)stress;
	const btScalar			frc=stress-sel;
	return(spectrum[sel]+(spectrum[sel+1]-spectrum[sel])*frc);
}
#endif

//
void btSoftBodyHelpers::Draw(btSoftBody* psb,
							 btIDebugDraw* idraw,
							 int drawflags)
{
	const btScalar scl = (btScalar)0.1;
	const btScalar nscl = scl * 5;
	const btVector3 lcolor = btVector3(0, 0, 0);
	const btVector3 ncolor = btVector3(1, 1, 1);
	const btVector3 ccolor = btVector3(1, 0, 0);
	int i, j, nj;

	/* Clusters	*/
	if (0 != (drawflags & fDrawFlags::Clusters))
	{
		srand(1806);
		for (i = 0; i < psb->m_clusters.size(); ++i)
		{
			if (psb->m_clusters[i]->m_collide)
			{
				btVector3 color(rand() / (btScalar)RAND_MAX,
								rand() / (btScalar)RAND_MAX,
								rand() / (btScalar)RAND_MAX);
				color = color.normalized() * 0.75;
				btAlignedObjectArray<btVector3> vertices;
				vertices.resize(psb->m_clusters[i]->m_nodes.size());
				for (j = 0, nj = vertices.size(); j < nj; ++j)
				{
					vertices[j] = psb->m_clusters[i]->m_nodes[j]->m_x;
				}
#define USE_NEW_CONVEX_HULL_COMPUTER
#ifdef USE_NEW_CONVEX_HULL_COMPUTER
				btConvexHullComputer computer;
				int stride = sizeof(btVector3);
				int count = vertices.size();
				btScalar shrink = 0.f;
				btScalar shrinkClamp = 0.f;
				computer.compute(&vertices[0].getX(), stride, count, shrink, shrinkClamp);
				for (int i = 0; i < computer.faces.size(); i++)
				{
					int face = computer.faces[i];
					//printf("face=%d\n",face);
					const btConvexHullComputer::Edge* firstEdge = &computer.edges[face];
					const btConvexHullComputer::Edge* edge = firstEdge->getNextEdgeOfFace();

					int v0 = firstEdge->getSourceVertex();
					int v1 = firstEdge->getTargetVertex();
					while (edge != firstEdge)
					{
						int v2 = edge->getTargetVertex();
						idraw->drawTriangle(computer.vertices[v0], computer.vertices[v1], computer.vertices[v2], color, 1);
						edge = edge->getNextEdgeOfFace();
						v0 = v1;
						v1 = v2;
					};
				}
#else

				HullDesc hdsc(QF_TRIANGLES, vertices.size(), &vertices[0]);
				HullResult hres;
				HullLibrary hlib;
				hdsc.mMaxVertices = vertices.size();
				hlib.CreateConvexHull(hdsc, hres);
				const btVector3 center = average(hres.m_OutputVertices);
				add(hres.m_OutputVertices, -center);
				mul(hres.m_OutputVertices, (btScalar)1);
				add(hres.m_OutputVertices, center);
				for (j = 0; j < (int)hres.mNumFaces; ++j)
				{
					const int idx[] = {hres.m_Indices[j * 3 + 0], hres.m_Indices[j * 3 + 1], hres.m_Indices[j * 3 + 2]};
					idraw->drawTriangle(hres.m_OutputVertices[idx[0]],
										hres.m_OutputVertices[idx[1]],
										hres.m_OutputVertices[idx[2]],
										color, 1);
				}
				hlib.ReleaseResult(hres);
#endif
			}
			/* Velocities	*/
#if 0
			for(int j=0;j<psb->m_clusters[i].m_nodes.size();++j)
			{
				const btSoftBody::Cluster&	c=psb->m_clusters[i];
				const btVector3				r=c.m_nodes[j]->m_x-c.m_com;
				const btVector3				v=c.m_lv+btCross(c.m_av,r);
				idraw->drawLine(c.m_nodes[j]->m_x,c.m_nodes[j]->m_x+v,btVector3(1,0,0));
			}
#endif
			/* Frame		*/
			//		btSoftBody::Cluster& c=*psb->m_clusters[i];
			//		idraw->drawLine(c.m_com,c.m_framexform*btVector3(10,0,0),btVector3(1,0,0));
			//		idraw->drawLine(c.m_com,c.m_framexform*btVector3(0,10,0),btVector3(0,1,0));
			//		idraw->drawLine(c.m_com,c.m_framexform*btVector3(0,0,10),btVector3(0,0,1));
		}
	}
	else
	{
		/* Nodes	*/
		if (0 != (drawflags & fDrawFlags::Nodes))
		{
			for (i = 0; i < psb->m_nodes.size(); ++i)
			{
				const btSoftBody::Node& n = psb->m_nodes[i];
				if (0 == (n.m_material->m_flags & btSoftBody::fMaterial::DebugDraw)) continue;
				idraw->drawLine(n.m_x - btVector3(scl, 0, 0), n.m_x + btVector3(scl, 0, 0), btVector3(1, 0, 0));
				idraw->drawLine(n.m_x - btVector3(0, scl, 0), n.m_x + btVector3(0, scl, 0), btVector3(0, 1, 0));
				idraw->drawLine(n.m_x - btVector3(0, 0, scl), n.m_x + btVector3(0, 0, scl), btVector3(0, 0, 1));
			}
		}
		/* Links	*/
		if (0 != (drawflags & fDrawFlags::Links))
		{
			for (i = 0; i < psb->m_links.size(); ++i)
			{
				const btSoftBody::Link& l = psb->m_links[i];
				if (0 == (l.m_material->m_flags & btSoftBody::fMaterial::DebugDraw)) continue;
				idraw->drawLine(l.m_n[0]->m_x, l.m_n[1]->m_x, lcolor);
			}
		}
		/* Normals	*/
		if (0 != (drawflags & fDrawFlags::Normals))
		{
			for (i = 0; i < psb->m_nodes.size(); ++i)
			{
				const btSoftBody::Node& n = psb->m_nodes[i];
				if (0 == (n.m_material->m_flags & btSoftBody::fMaterial::DebugDraw)) continue;
				const btVector3 d = n.m_n * nscl;
				idraw->drawLine(n.m_x, n.m_x + d, ncolor);
				idraw->drawLine(n.m_x, n.m_x - d, ncolor * 0.5);
			}
		}
		/* Contacts	*/
		if (0 != (drawflags & fDrawFlags::Contacts))
		{
			static const btVector3 axis[] = {btVector3(1, 0, 0),
											 btVector3(0, 1, 0),
											 btVector3(0, 0, 1)};
			for (i = 0; i < psb->m_rcontacts.size(); ++i)
			{
				const btSoftBody::RContact& c = psb->m_rcontacts[i];
				const btVector3 o = c.m_node->m_x - c.m_cti.m_normal *
														(btDot(c.m_node->m_x, c.m_cti.m_normal) + c.m_cti.m_offset);
				const btVector3 x = btCross(c.m_cti.m_normal, axis[c.m_cti.m_normal.minAxis()]).normalized();
				const btVector3 y = btCross(x, c.m_cti.m_normal).normalized();
				idraw->drawLine(o - x * nscl, o + x * nscl, ccolor);
				idraw->drawLine(o - y * nscl, o + y * nscl, ccolor);
				idraw->drawLine(o, o + c.m_cti.m_normal * nscl * 3, btVector3(1, 1, 0));
			}
		}
		/* Faces	*/
		if (0 != (drawflags & fDrawFlags::Faces))
		{
			const btScalar scl = (btScalar)0.8;
			const btScalar alp = (btScalar)1;
			const btVector3 col(0, (btScalar)0.7, 0);
			for (i = 0; i < psb->m_faces.size(); ++i)
			{
				const btSoftBody::Face& f = psb->m_faces[i];
				if (0 == (f.m_material->m_flags & btSoftBody::fMaterial::DebugDraw)) continue;
				const btVector3 x[] = {f.m_n[0]->m_x, f.m_n[1]->m_x, f.m_n[2]->m_x};
				const btVector3 c = (x[0] + x[1] + x[2]) / 3;
				idraw->drawTriangle((x[0] - c) * scl + c,
									(x[1] - c) * scl + c,
									(x[2] - c) * scl + c,
									col, alp);
			}
		}
		/* Tetras	*/
		if (0 != (drawflags & fDrawFlags::Tetras))
		{
			const btScalar scl = (btScalar)0.8;
			const btScalar alp = (btScalar)1;
			const btVector3 col((btScalar)0.3, (btScalar)0.3, (btScalar)0.7);
			for (int i = 0; i < psb->m_tetras.size(); ++i)
			{
				const btSoftBody::Tetra& t = psb->m_tetras[i];
				if (0 == (t.m_material->m_flags & btSoftBody::fMaterial::DebugDraw)) continue;
				const btVector3 x[] = {t.m_n[0]->m_x, t.m_n[1]->m_x, t.m_n[2]->m_x, t.m_n[3]->m_x};
				const btVector3 c = (x[0] + x[1] + x[2] + x[3]) / 4;
				idraw->drawTriangle((x[0] - c) * scl + c, (x[1] - c) * scl + c, (x[2] - c) * scl + c, col, alp);
				idraw->drawTriangle((x[0] - c) * scl + c, (x[1] - c) * scl + c, (x[3] - c) * scl + c, col, alp);
				idraw->drawTriangle((x[1] - c) * scl + c, (x[2] - c) * scl + c, (x[3] - c) * scl + c, col, alp);
				idraw->drawTriangle((x[2] - c) * scl + c, (x[0] - c) * scl + c, (x[3] - c) * scl + c, col, alp);
			}
		}
	}
	/* Anchors	*/
	if (0 != (drawflags & fDrawFlags::Anchors))
	{
		for (i = 0; i < psb->m_anchors.size(); ++i)
		{
			const btSoftBody::Anchor& a = psb->m_anchors[i];
			const btVector3 q = a.m_body->getWorldTransform() * a.m_local;
			drawVertex(idraw, a.m_node->m_x, 0.25, btVector3(1, 0, 0));
			drawVertex(idraw, q, 0.25, btVector3(0, 1, 0));
			idraw->drawLine(a.m_node->m_x, q, btVector3(1, 1, 1));
		}
		for (i = 0; i < psb->m_nodes.size(); ++i)
		{
			const btSoftBody::Node& n = psb->m_nodes[i];
			if (0 == (n.m_material->m_flags & btSoftBody::fMaterial::DebugDraw)) continue;
			if (n.m_im <= 0)
			{
				drawVertex(idraw, n.m_x, 0.25, btVector3(1, 0, 0));
			}
		}
	}

	/* Notes	*/
	if (0 != (drawflags & fDrawFlags::Notes))
	{
		for (i = 0; i < psb->m_notes.size(); ++i)
		{
			const btSoftBody::Note& n = psb->m_notes[i];
			btVector3 p = n.m_offset;
			for (int j = 0; j < n.m_rank; ++j)
			{
				p += n.m_nodes[j]->m_x * n.m_coords[j];
			}
			idraw->draw3dText(p, n.m_text);
		}
	}
	/* Node tree	*/
	if (0 != (drawflags & fDrawFlags::NodeTree)) DrawNodeTree(psb, idraw);
	/* Face tree	*/
	if (0 != (drawflags & fDrawFlags::FaceTree)) DrawFaceTree(psb, idraw);
	/* Cluster tree	*/
	if (0 != (drawflags & fDrawFlags::ClusterTree)) DrawClusterTree(psb, idraw);
	/* Joints		*/
	if (0 != (drawflags & fDrawFlags::Joints))
	{
		for (i = 0; i < psb->m_joints.size(); ++i)
		{
			const btSoftBody::Joint* pj = psb->m_joints[i];
			switch (pj->Type())
			{
				case btSoftBody::Joint::eType::Linear:
				{
					const btSoftBody::LJoint* pjl = (const btSoftBody::LJoint*)pj;
					const btVector3 a0 = pj->m_bodies[0].xform() * pjl->m_refs[0];
					const btVector3 a1 = pj->m_bodies[1].xform() * pjl->m_refs[1];
					idraw->drawLine(pj->m_bodies[0].xform().getOrigin(), a0, btVector3(1, 1, 0));
					idraw->drawLine(pj->m_bodies[1].xform().getOrigin(), a1, btVector3(0, 1, 1));
					drawVertex(idraw, a0, 0.25, btVector3(1, 1, 0));
					drawVertex(idraw, a1, 0.25, btVector3(0, 1, 1));
				}
				break;
				case btSoftBody::Joint::eType::Angular:
				{
					//const btSoftBody::AJoint*	pja=(const btSoftBody::AJoint*)pj;
					const btVector3 o0 = pj->m_bodies[0].xform().getOrigin();
					const btVector3 o1 = pj->m_bodies[1].xform().getOrigin();
					const btVector3 a0 = pj->m_bodies[0].xform().getBasis() * pj->m_refs[0];
					const btVector3 a1 = pj->m_bodies[1].xform().getBasis() * pj->m_refs[1];
					idraw->drawLine(o0, o0 + a0 * 10, btVector3(1, 1, 0));
					idraw->drawLine(o0, o0 + a1 * 10, btVector3(1, 1, 0));
					idraw->drawLine(o1, o1 + a0 * 10, btVector3(0, 1, 1));
					idraw->drawLine(o1, o1 + a1 * 10, btVector3(0, 1, 1));
					break;
				}
				default:
				{
				}
			}
		}
	}
}

//
void btSoftBodyHelpers::DrawInfos(btSoftBody* psb,
								  btIDebugDraw* idraw,
								  bool masses,
								  bool areas,
								  bool /*stress*/)
{
	for (int i = 0; i < psb->m_nodes.size(); ++i)
	{
		const btSoftBody::Node& n = psb->m_nodes[i];
		char text[2048] = {0};
		char buff[1024];
		if (masses)
		{
			sprintf(buff, " M(%.2f)", 1 / n.m_im);
			strcat(text, buff);
		}
		if (areas)
		{
			sprintf(buff, " A(%.2f)", n.m_area);
			strcat(text, buff);
		}
		if (text[0]) idraw->draw3dText(n.m_x, text);
	}
}

//
void btSoftBodyHelpers::DrawNodeTree(btSoftBody* psb,
									 btIDebugDraw* idraw,
									 int mindepth,
									 int maxdepth)
{
	drawTree(idraw, psb->m_ndbvt.m_root, 0, btVector3(1, 0, 1), btVector3(1, 1, 1), mindepth, maxdepth);
}

//
void btSoftBodyHelpers::DrawFaceTree(btSoftBody* psb,
									 btIDebugDraw* idraw,
									 int mindepth,
									 int maxdepth)
{
	drawTree(idraw, psb->m_fdbvt.m_root, 0, btVector3(0, 1, 0), btVector3(1, 0, 0), mindepth, maxdepth);
}

//
void btSoftBodyHelpers::DrawClusterTree(btSoftBody* psb,
										btIDebugDraw* idraw,
										int mindepth,
										int maxdepth)
{
	drawTree(idraw, psb->m_cdbvt.m_root, 0, btVector3(0, 1, 1), btVector3(1, 0, 0), mindepth, maxdepth);
}

//The btSoftBody object from the BulletSDK includes an array of Nodes and Links. These links appear
// to be first set up to connect a node to between 5 and 6 of its neighbors [480 links],
//and then to the rest of the nodes after the execution of the Floyd-Warshall graph algorithm
//[another 930 links].
//The way the links are stored by default, we have a number of cases where adjacent links share a node in common
// - this leads to the creation of a data dependency through memory.
//The PSolve_Links() function reads and writes nodes as it iterates over each link.
//So, we now have the possibility of a data dependency between iteration X
//that processes link L with iteration X+1 that processes link L+1
//because L and L+1 have one node in common, and iteration X updates the positions of that node,
//and iteration X+1 reads in the position of that shared node.
//
//Such a memory dependency limits the ability of a modern CPU to speculate beyond
//a certain point because it has to respect a possible dependency
//- this prevents the CPU from making full use of its out-of-order resources.
//If we re-order the links such that we minimize the cases where a link L and L+1 share a common node,
//we create a temporal gap between when the node position is written,
//and when it is subsequently read. This in turn allows the CPU to continue execution without
//risking a dependency violation. Such a reordering would result in significant speedups on
//modern CPUs with lots of execution resources.
//In our testing, we see it have a tremendous impact not only on the A7,
//but also on all x86 cores that ship with modern Macs.
//The attached source file includes a single function (ReoptimizeLinkOrder) which can be called on a
//btSoftBody object in the solveConstraints() function before the actual solver is invoked,
//or right after generateBendingConstraints() once we have all 1410 links.

//===================================================================
//
//
// This function takes in a list of interdependent Links and tries
// to maximize the distance between calculation
// of dependent links.  This increases the amount of parallelism that can
// be exploited by out-of-order instruction processors with large but
// (inevitably) finite instruction windows.
//
//===================================================================

// A small structure to track lists of dependent link calculations
class LinkDeps_t
{
public:
	int value;         // A link calculation that is dependent on this one
					   // Positive values = "input A" while negative values = "input B"
	LinkDeps_t* next;  // Next dependence in the list
};
typedef LinkDeps_t* LinkDepsPtr_t;

// Dependency list constants
#define REOP_NOT_DEPENDENT -1
#define REOP_NODE_COMPLETE -2  // Must be less than REOP_NOT_DEPENDENT

void btSoftBodyHelpers::ReoptimizeLinkOrder(btSoftBody* psb /* This can be replaced by a btSoftBody pointer */)
{
	int i, nLinks = psb->m_links.size(), nNodes = psb->m_nodes.size();
	btSoftBody::Link* lr;
	int ar, br;
	btSoftBody::Node* node0 = &(psb->m_nodes[0]);
	btSoftBody::Node* node1 = &(psb->m_nodes[1]);
	LinkDepsPtr_t linkDep;
	int readyListHead, readyListTail, linkNum, linkDepFrees, depLink;

	// Allocate temporary buffers
	int* nodeWrittenAt = new int[nNodes + 1];  // What link calculation produced this node's current values?
	int* linkDepA = new int[nLinks];           // Link calculation input is dependent upon prior calculation #N
	int* linkDepB = new int[nLinks];
	int* readyList = new int[nLinks];                              // List of ready-to-process link calculations (# of links, maximum)
	LinkDeps_t* linkDepFreeList = new LinkDeps_t[2 * nLinks];      // Dependent-on-me list elements (2x# of links, maximum)
	LinkDepsPtr_t* linkDepListStarts = new LinkDepsPtr_t[nLinks];  // Start nodes of dependent-on-me lists, one for each link

	// Copy the original, unsorted links to a side buffer
	btSoftBody::Link* linkBuffer = new btSoftBody::Link[nLinks];
	memcpy(linkBuffer, &(psb->m_links[0]), sizeof(btSoftBody::Link) * nLinks);

	// Clear out the node setup and ready list
	for (i = 0; i < nNodes + 1; i++)
	{
		nodeWrittenAt[i] = REOP_NOT_DEPENDENT;
	}
	for (i = 0; i < nLinks; i++)
	{
		linkDepListStarts[i] = NULL;
	}
	readyListHead = readyListTail = linkDepFrees = 0;

	// Initial link analysis to set up data structures
	for (i = 0; i < nLinks; i++)
	{
		// Note which prior link calculations we are dependent upon & build up dependence lists
		lr = &(psb->m_links[i]);
		ar = (lr->m_n[0] - node0) / (node1 - node0);
		br = (lr->m_n[1] - node0) / (node1 - node0);
		if (nodeWrittenAt[ar] > REOP_NOT_DEPENDENT)
		{
			linkDepA[i] = nodeWrittenAt[ar];
			linkDep = &linkDepFreeList[linkDepFrees++];
			linkDep->value = i;
			linkDep->next = linkDepListStarts[nodeWrittenAt[ar]];
			linkDepListStarts[nodeWrittenAt[ar]] = linkDep;
		}
		else
		{
			linkDepA[i] = REOP_NOT_DEPENDENT;
		}
		if (nodeWrittenAt[br] > REOP_NOT_DEPENDENT)
		{
			linkDepB[i] = nodeWrittenAt[br];
			linkDep = &linkDepFreeList[linkDepFrees++];
			linkDep->value = -(i + 1);
			linkDep->next = linkDepListStarts[nodeWrittenAt[br]];
			linkDepListStarts[nodeWrittenAt[br]] = linkDep;
		}
		else
		{
			linkDepB[i] = REOP_NOT_DEPENDENT;
		}

		// Add this link to the initial ready list, if it is not dependent on any other links
		if ((linkDepA[i] == REOP_NOT_DEPENDENT) && (linkDepB[i] == REOP_NOT_DEPENDENT))
		{
			readyList[readyListTail++] = i;
			linkDepA[i] = linkDepB[i] = REOP_NODE_COMPLETE;  // Probably not needed now
		}

		// Update the nodes to mark which ones are calculated by this link
		nodeWrittenAt[ar] = nodeWrittenAt[br] = i;
	}

	// Process the ready list and create the sorted list of links
	// -- By treating the ready list as a queue, we maximize the distance between any
	//    inter-dependent node calculations
	// -- All other (non-related) nodes in the ready list will automatically be inserted
	//    in between each set of inter-dependent link calculations by this loop
	i = 0;
	while (readyListHead != readyListTail)
	{
		// Use ready list to select the next link to process
		linkNum = readyList[readyListHead++];
		// Copy the next-to-calculate link back into the original link array
		psb->m_links[i++] = linkBuffer[linkNum];

		// Free up any link inputs that are dependent on this one
		linkDep = linkDepListStarts[linkNum];
		while (linkDep)
		{
			depLink = linkDep->value;
			if (depLink >= 0)
			{
				linkDepA[depLink] = REOP_NOT_DEPENDENT;
			}
			else
			{
				depLink = -depLink - 1;
				linkDepB[depLink] = REOP_NOT_DEPENDENT;
			}
			// Add this dependent link calculation to the ready list if *both* inputs are clear
			if ((linkDepA[depLink] == REOP_NOT_DEPENDENT) && (linkDepB[depLink] == REOP_NOT_DEPENDENT))
			{
				readyList[readyListTail++] = depLink;
				linkDepA[depLink] = linkDepB[depLink] = REOP_NODE_COMPLETE;  // Probably not needed now
			}
			linkDep = linkDep->next;
		}
	}

	// Delete the temporary buffers
	delete[] nodeWrittenAt;
	delete[] linkDepA;
	delete[] linkDepB;
	delete[] readyList;
	delete[] linkDepFreeList;
	delete[] linkDepListStarts;
	delete[] linkBuffer;
}

//
void btSoftBodyHelpers::DrawFrame(btSoftBody* psb,
								  btIDebugDraw* idraw)
{
	if (psb->m_pose.m_bframe)
	{
		static const btScalar ascl = 10;
		static const btScalar nscl = (btScalar)0.1;
		const btVector3 com = psb->m_pose.m_com;
		const btMatrix3x3 trs = psb->m_pose.m_rot * psb->m_pose.m_scl;
		const btVector3 Xaxis = (trs * btVector3(1, 0, 0)).normalized();
		const btVector3 Yaxis = (trs * btVector3(0, 1, 0)).normalized();
		const btVector3 Zaxis = (trs * btVector3(0, 0, 1)).normalized();
		idraw->drawLine(com, com + Xaxis * ascl, btVector3(1, 0, 0));
		idraw->drawLine(com, com + Yaxis * ascl, btVector3(0, 1, 0));
		idraw->drawLine(com, com + Zaxis * ascl, btVector3(0, 0, 1));
		for (int i = 0; i < psb->m_pose.m_pos.size(); ++i)
		{
			const btVector3 x = com + trs * psb->m_pose.m_pos[i];
			drawVertex(idraw, x, nscl, btVector3(1, 0, 1));
		}
	}
}

//
btSoftBody* btSoftBodyHelpers::CreateRope(btSoftBodyWorldInfo& worldInfo, const btVector3& from,
										  const btVector3& to,
										  int res,
										  int fixeds)
{
	/* Create nodes	*/
	const int r = res + 2;
	btVector3* x = new btVector3[r];
	btScalar* m = new btScalar[r];
	int i;

	for (i = 0; i < r; ++i)
	{
		const btScalar t = i / (btScalar)(r - 1);
		x[i] = lerp(from, to, t);
		m[i] = 1;
	}
	btSoftBody* psb = new btSoftBody(&worldInfo, r, x, m);
	if (fixeds & 1) psb->setMass(0, 0);
	if (fixeds & 2) psb->setMass(r - 1, 0);
	delete[] x;
	delete[] m;
	/* Create links	*/
	for (i = 1; i < r; ++i)
	{
		psb->appendLink(i - 1, i);
	}
	/* Finished		*/
	return (psb);
}

//
btSoftBody* btSoftBodyHelpers::CreatePatch(btSoftBodyWorldInfo& worldInfo, const btVector3& corner00,
										   const btVector3& corner10,
										   const btVector3& corner01,
										   const btVector3& corner11,
										   int resx,
										   int resy,
										   int fixeds,
										   bool gendiags,
                                           btScalar perturbation)
{
#define IDX(_x_, _y_) ((_y_)*rx + (_x_))
	/* Create nodes	*/
	if ((resx < 2) || (resy < 2)) return (0);
	const int rx = resx;
	const int ry = resy;
	const int tot = rx * ry;
	btVector3* x = new btVector3[tot];
	btScalar* m = new btScalar[tot];
	int iy;

	for (iy = 0; iy < ry; ++iy)
	{
		const btScalar ty = iy / (btScalar)(ry - 1);
		const btVector3 py0 = lerp(corner00, corner01, ty);
		const btVector3 py1 = lerp(corner10, corner11, ty);
		for (int ix = 0; ix < rx; ++ix)
		{
			const btScalar tx = ix / (btScalar)(rx - 1);
            btScalar pert = perturbation * btScalar(rand())/RAND_MAX;
            btVector3 temp1 = py1;
            temp1.setY(py1.getY() + pert);
            btVector3 temp = py0;
            pert = perturbation * btScalar(rand())/RAND_MAX;
            temp.setY(py0.getY() + pert);
			x[IDX(ix, iy)] = lerp(temp, temp1, tx);
			m[IDX(ix, iy)] = 1;
		}
	}
	btSoftBody* psb = new btSoftBody(&worldInfo, tot, x, m);
	if (fixeds & 1) psb->setMass(IDX(0, 0), 0);
	if (fixeds & 2) psb->setMass(IDX(rx - 1, 0), 0);
	if (fixeds & 4) psb->setMass(IDX(0, ry - 1), 0);
	if (fixeds & 8) psb->setMass(IDX(rx - 1, ry - 1), 0);
	delete[] x;
	delete[] m;
	/* Create links	and faces */
	for (iy = 0; iy < ry; ++iy)
	{
		for (int ix = 0; ix < rx; ++ix)
		{
			const int idx = IDX(ix, iy);
			const bool mdx = (ix + 1) < rx;
			const bool mdy = (iy + 1) < ry;
			if (mdx) psb->appendLink(idx, IDX(ix + 1, iy));
			if (mdy) psb->appendLink(idx, IDX(ix, iy + 1));
			if (mdx && mdy)
			{
				if ((ix + iy) & 1)
				{
					psb->appendFace(IDX(ix, iy), IDX(ix + 1, iy), IDX(ix + 1, iy + 1));
					psb->appendFace(IDX(ix, iy), IDX(ix + 1, iy + 1), IDX(ix, iy + 1));
					if (gendiags)
					{
						psb->appendLink(IDX(ix, iy), IDX(ix + 1, iy + 1));
					}
				}
				else
				{
					psb->appendFace(IDX(ix, iy + 1), IDX(ix, iy), IDX(ix + 1, iy));
					psb->appendFace(IDX(ix, iy + 1), IDX(ix + 1, iy), IDX(ix + 1, iy + 1));
					if (gendiags)
					{
						psb->appendLink(IDX(ix + 1, iy), IDX(ix, iy + 1));
					}
				}
			}
		}
	}
	/* Finished		*/
#undef IDX
	return (psb);
}

//
btSoftBody* btSoftBodyHelpers::CreatePatchUV(btSoftBodyWorldInfo& worldInfo,
											 const btVector3& corner00,
											 const btVector3& corner10,
											 const btVector3& corner01,
											 const btVector3& corner11,
											 int resx,
											 int resy,
											 int fixeds,
											 bool gendiags,
											 float* tex_coords)
{
	/*
	*
	*  corners:
	*
	*  [0][0]     corner00 ------- corner01   [resx][0]
	*                |                |
	*                |                |
	*  [0][resy]  corner10 -------- corner11  [resx][resy]
	*
	*
	*
	*
	*
	*
	*   "fixedgs" map:
	*
	*  corner00     -->   +1
	*  corner01     -->   +2
	*  corner10     -->   +4
	*  corner11     -->   +8
	*  upper middle -->  +16
	*  left middle  -->  +32
	*  right middle -->  +64
	*  lower middle --> +128
	*  center       --> +256
	*
	*
	*   tex_coords size   (resx-1)*(resy-1)*12
	*
	*
	*
	*     SINGLE QUAD INTERNALS
	*
	*  1) btSoftBody's nodes and links,
	*     diagonal link is optional ("gendiags")
	*
	*
	*    node00 ------ node01
	*      | .              
	*      |   .            
	*      |     .          
	*      |       .        
	*      |         .      
	*    node10        node11
	*
	*
	*
	*   2) Faces:
	*      two triangles,
	*      UV Coordinates (hier example for single quad)
	*      
	*     (0,1)          (0,1)  (1,1)
	*     1 |\            3 \-----| 2
	*       | \              \    |
	*       |  \              \   |
	*       |   \              \  |
	*       |    \              \ |
	*     2 |-----\ 3            \| 1
	*     (0,0)    (1,0)       (1,0)
	*
	*
	*
	*
	*
	*
	*/

#define IDX(_x_, _y_) ((_y_)*rx + (_x_))
	/* Create nodes		*/
	if ((resx < 2) || (resy < 2)) return (0);
	const int rx = resx;
	const int ry = resy;
	const int tot = rx * ry;
	btVector3* x = new btVector3[tot];
	btScalar* m = new btScalar[tot];

	int iy;

	for (iy = 0; iy < ry; ++iy)
	{
		const btScalar ty = iy / (btScalar)(ry - 1);
		const btVector3 py0 = lerp(corner00, corner01, ty);
		const btVector3 py1 = lerp(corner10, corner11, ty);
		for (int ix = 0; ix < rx; ++ix)
		{
			const btScalar tx = ix / (btScalar)(rx - 1);
			x[IDX(ix, iy)] = lerp(py0, py1, tx);
			m[IDX(ix, iy)] = 1;
		}
	}
	btSoftBody* psb = new btSoftBody(&worldInfo, tot, x, m);
	if (fixeds & 1) psb->setMass(IDX(0, 0), 0);
	if (fixeds & 2) psb->setMass(IDX(rx - 1, 0), 0);
	if (fixeds & 4) psb->setMass(IDX(0, ry - 1), 0);
	if (fixeds & 8) psb->setMass(IDX(rx - 1, ry - 1), 0);
	if (fixeds & 16) psb->setMass(IDX((rx - 1) / 2, 0), 0);
	if (fixeds & 32) psb->setMass(IDX(0, (ry - 1) / 2), 0);
	if (fixeds & 64) psb->setMass(IDX(rx - 1, (ry - 1) / 2), 0);
	if (fixeds & 128) psb->setMass(IDX((rx - 1) / 2, ry - 1), 0);
	if (fixeds & 256) psb->setMass(IDX((rx - 1) / 2, (ry - 1) / 2), 0);
	delete[] x;
	delete[] m;

	int z = 0;
	/* Create links	and faces	*/
	for (iy = 0; iy < ry; ++iy)
	{
		for (int ix = 0; ix < rx; ++ix)
		{
			const bool mdx = (ix + 1) < rx;
			const bool mdy = (iy + 1) < ry;

			int node00 = IDX(ix, iy);
			int node01 = IDX(ix + 1, iy);
			int node10 = IDX(ix, iy + 1);
			int node11 = IDX(ix + 1, iy + 1);

			if (mdx) psb->appendLink(node00, node01);
			if (mdy) psb->appendLink(node00, node10);
			if (mdx && mdy)
			{
				psb->appendFace(node00, node10, node11);
				if (tex_coords)
				{
					tex_coords[z + 0] = CalculateUV(resx, resy, ix, iy, 0);
					tex_coords[z + 1] = CalculateUV(resx, resy, ix, iy, 1);
					tex_coords[z + 2] = CalculateUV(resx, resy, ix, iy, 0);
					tex_coords[z + 3] = CalculateUV(resx, resy, ix, iy, 2);
					tex_coords[z + 4] = CalculateUV(resx, resy, ix, iy, 3);
					tex_coords[z + 5] = CalculateUV(resx, resy, ix, iy, 2);
				}
				psb->appendFace(node11, node01, node00);
				if (tex_coords)
				{
					tex_coords[z + 6] = CalculateUV(resx, resy, ix, iy, 3);
					tex_coords[z + 7] = CalculateUV(resx, resy, ix, iy, 2);
					tex_coords[z + 8] = CalculateUV(resx, resy, ix, iy, 3);
					tex_coords[z + 9] = CalculateUV(resx, resy, ix, iy, 1);
					tex_coords[z + 10] = CalculateUV(resx, resy, ix, iy, 0);
					tex_coords[z + 11] = CalculateUV(resx, resy, ix, iy, 1);
				}
				if (gendiags) psb->appendLink(node00, node11);
				z += 12;
			}
		}
	}
	/* Finished	*/
#undef IDX
	return (psb);
}

float btSoftBodyHelpers::CalculateUV(int resx, int resy, int ix, int iy, int id)
{
	/*
	*
	*
	*    node00 --- node01
	*      |          |
	*    node10 --- node11
	*
	*
	*   ID map:
	*
	*   node00 s --> 0
	*   node00 t --> 1
	*
	*   node01 s --> 3
	*   node01 t --> 1
	*
	*   node10 s --> 0
	*   node10 t --> 2
	*
	*   node11 s --> 3
	*   node11 t --> 2
	*
	*
	*/

	float tc = 0.0f;
	if (id == 0)
	{
		tc = (1.0f / ((resx - 1)) * ix);
	}
	else if (id == 1)
	{
		tc = (1.0f / ((resy - 1)) * (resy - 1 - iy));
	}
	else if (id == 2)
	{
		tc = (1.0f / ((resy - 1)) * (resy - 1 - iy - 1));
	}
	else if (id == 3)
	{
		tc = (1.0f / ((resx - 1)) * (ix + 1));
	}
	return tc;
}
//
btSoftBody* btSoftBodyHelpers::CreateEllipsoid(btSoftBodyWorldInfo& worldInfo, const btVector3& center,
											   const btVector3& radius,
											   int res)
{
	struct Hammersley
	{
		static void Generate(btVector3* x, int n)
		{
			for (int i = 0; i < n; i++)
			{
				btScalar p = 0.5, t = 0;
				for (int j = i; j; p *= 0.5, j >>= 1)
					if (j & 1) t += p;
				btScalar w = 2 * t - 1;
				btScalar a = (SIMD_PI + 2 * i * SIMD_PI) / n;
				btScalar s = btSqrt(1 - w * w);
				*x++ = btVector3(s * btCos(a), s * btSin(a), w);
			}
		}
	};
	btAlignedObjectArray<btVector3> vtx;
	vtx.resize(3 + res);
	Hammersley::Generate(&vtx[0], vtx.size());
	for (int i = 0; i < vtx.size(); ++i)
	{
		vtx[i] = vtx[i] * radius + center;
	}
	return (CreateFromConvexHull(worldInfo, &vtx[0], vtx.size()));
}

//
btSoftBody* btSoftBodyHelpers::CreateFromTriMesh(btSoftBodyWorldInfo& worldInfo, const btScalar* vertices,
												 const int* triangles,
												 int ntriangles, bool randomizeConstraints)
{
	int maxidx = 0;
	int i, j, ni;

	for (i = 0, ni = ntriangles * 3; i < ni; ++i)
	{
		maxidx = btMax(triangles[i], maxidx);
	}
	++maxidx;
	btAlignedObjectArray<bool> chks;
	btAlignedObjectArray<btVector3> vtx;
	chks.resize(maxidx * maxidx, false);
	vtx.resize(maxidx);
	for (i = 0, j = 0, ni = maxidx * 3; i < ni; ++j, i += 3)
	{
		vtx[j] = btVector3(vertices[i], vertices[i + 1], vertices[i + 2]);
	}
	btSoftBody* psb = new btSoftBody(&worldInfo, vtx.size(), &vtx[0], 0);
	for (i = 0, ni = ntriangles * 3; i < ni; i += 3)
	{
		const int idx[] = {triangles[i], triangles[i + 1], triangles[i + 2]};
#define IDX(_x_, _y_) ((_y_)*maxidx + (_x_))
		for (int j = 2, k = 0; k < 3; j = k++)
		{
			if (!chks[IDX(idx[j], idx[k])])
			{
				chks[IDX(idx[j], idx[k])] = true;
				chks[IDX(idx[k], idx[j])] = true;
				psb->appendLink(idx[j], idx[k]);
			}
		}
#undef IDX
		psb->appendFace(idx[0], idx[1], idx[2]);
	}

	if (randomizeConstraints)
	{
		psb->randomizeConstraints();
	}

	return (psb);
}

//
btSoftBody* btSoftBodyHelpers::CreateFromConvexHull(btSoftBodyWorldInfo& worldInfo, const btVector3* vertices,
													int nvertices, bool randomizeConstraints)
{
	HullDesc hdsc(QF_TRIANGLES, nvertices, vertices);
	HullResult hres;
	HullLibrary hlib; /*??*/
	hdsc.mMaxVertices = nvertices;
	hlib.CreateConvexHull(hdsc, hres);
	btSoftBody* psb = new btSoftBody(&worldInfo, (int)hres.mNumOutputVertices,
									 &hres.m_OutputVertices[0], 0);
	for (int i = 0; i < (int)hres.mNumFaces; ++i)
	{
		const int idx[] = {static_cast<int>(hres.m_Indices[i * 3 + 0]),
						   static_cast<int>(hres.m_Indices[i * 3 + 1]),
						   static_cast<int>(hres.m_Indices[i * 3 + 2])};
		if (idx[0] < idx[1]) psb->appendLink(idx[0], idx[1]);
		if (idx[1] < idx[2]) psb->appendLink(idx[1], idx[2]);
		if (idx[2] < idx[0]) psb->appendLink(idx[2], idx[0]);
		psb->appendFace(idx[0], idx[1], idx[2]);
	}
	hlib.ReleaseResult(hres);
	if (randomizeConstraints)
	{
		psb->randomizeConstraints();
	}
	return (psb);
}

static int nextLine(const char* buffer)
{
	int numBytesRead = 0;

	while (*buffer != '\n')
	{
		buffer++;
		numBytesRead++;
	}

	if (buffer[0] == 0x0a)
	{
		buffer++;
		numBytesRead++;
	}
	return numBytesRead;
}

/* Create from TetGen .ele, .face, .node data							*/
btSoftBody* btSoftBodyHelpers::CreateFromTetGenData(btSoftBodyWorldInfo& worldInfo,
													const char* ele,
													const char* face,
													const char* node,
													bool bfacelinks,
													bool btetralinks,
													bool bfacesfromtetras)
{
	btAlignedObjectArray<btVector3> pos;
	int nnode = 0;
	int ndims = 0;
	int nattrb = 0;
	int hasbounds = 0;
	int result = sscanf(node, "%d %d %d %d", &nnode, &ndims, &nattrb, &hasbounds);
	result = sscanf(node, "%d %d %d %d", &nnode, &ndims, &nattrb, &hasbounds);
	node += nextLine(node);

	pos.resize(nnode);
	for (int i = 0; i < pos.size(); ++i)
	{
		int index = 0;
		//int			bound=0;
		float x, y, z;
		sscanf(node, "%d %f %f %f", &index, &x, &y, &z);

		//	sn>>index;
		//	sn>>x;sn>>y;sn>>z;
		node += nextLine(node);

		//for(int j=0;j<nattrb;++j)
		//	sn>>a;

		//if(hasbounds)
		//	sn>>bound;

		pos[index].setX(btScalar(x));
		pos[index].setY(btScalar(y));
		pos[index].setZ(btScalar(z));
	}
	btSoftBody* psb = new btSoftBody(&worldInfo, nnode, &pos[0], 0);
#if 0
if(face&&face[0])
	{
	int								nface=0;
	sf>>nface;sf>>hasbounds;
	for(int i=0;i<nface;++i)
		{
		int			index=0;
		int			bound=0;
		int			ni[3];
		sf>>index;
		sf>>ni[0];sf>>ni[1];sf>>ni[2];
		sf>>bound;
		psb->appendFace(ni[0],ni[1],ni[2]);	
		if(btetralinks)
			{
			psb->appendLink(ni[0],ni[1],0,true);
			psb->appendLink(ni[1],ni[2],0,true);
			psb->appendLink(ni[2],ni[0],0,true);
			}
		}
	}
#endif

	if (ele && ele[0])
	{
		int ntetra = 0;
		int ncorner = 0;
		int neattrb = 0;
		sscanf(ele, "%d %d %d", &ntetra, &ncorner, &neattrb);
		ele += nextLine(ele);

		//se>>ntetra;se>>ncorner;se>>neattrb;
		for (int i = 0; i < ntetra; ++i)
		{
			int index = 0;
			int ni[4];

			//se>>index;
			//se>>ni[0];se>>ni[1];se>>ni[2];se>>ni[3];
			sscanf(ele, "%d %d %d %d %d", &index, &ni[0], &ni[1], &ni[2], &ni[3]);
			ele += nextLine(ele);
			//for(int j=0;j<neattrb;++j)
			//	se>>a;
			psb->appendTetra(ni[0], ni[1], ni[2], ni[3]);
			if (btetralinks)
			{
				psb->appendLink(ni[0], ni[1], 0, true);
				psb->appendLink(ni[1], ni[2], 0, true);
				psb->appendLink(ni[2], ni[0], 0, true);
				psb->appendLink(ni[0], ni[3], 0, true);
				psb->appendLink(ni[1], ni[3], 0, true);
				psb->appendLink(ni[2], ni[3], 0, true);
			}
		}
	}
    psb->initializeDmInverse();
    psb->m_tetraScratches.resize(psb->m_tetras.size());
    psb->m_tetraScratchesTn.resize(psb->m_tetras.size());
	printf("Nodes:  %u\r\n", psb->m_nodes.size());
	printf("Links:  %u\r\n", psb->m_links.size());
	printf("Faces:  %u\r\n", psb->m_faces.size());
	printf("Tetras: %u\r\n", psb->m_tetras.size());
	return (psb);
}

btSoftBody* btSoftBodyHelpers::CreateFromVtkFile(btSoftBodyWorldInfo& worldInfo, const char* vtk_file)
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
            ss.ignore(128, ' '); // ignore "POINTS"
            ss >> n_points;
            X.resize(n_points);
        }
        else if (line.substr(0, 5) == "CELLS")
        {
            reading_points = false;
            reading_tets = true;
            ss.ignore(128, ' '); // ignore "CELLS"
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
            X[x_count++] = position;
        }
        else if (reading_tets)
        {
            ss.ignore(128, ' '); // ignore "4"
            Index tet;
            tet.resize(4);
            for (size_t i = 0; i < 4; i++)
            {
                ss >> tet[i];
            }
            indices[indices_count++] = tet;
        }
    }
    btSoftBody* psb = new btSoftBody(&worldInfo, n_points, &X[0], 0);
    
    for (int i = 0; i < n_tets; ++i)
    {
        const Index& ni = indices[i];
        psb->appendTetra(ni[0], ni[1], ni[2], ni[3]);
        {
            psb->appendLink(ni[0], ni[1], 0, true);
            psb->appendLink(ni[1], ni[2], 0, true);
            psb->appendLink(ni[2], ni[0], 0, true);
            psb->appendLink(ni[0], ni[3], 0, true);
            psb->appendLink(ni[1], ni[3], 0, true);
            psb->appendLink(ni[2], ni[3], 0, true);
        }
    }
    
    
    generateBoundaryFaces(psb);
    psb->initializeDmInverse();
    psb->m_tetraScratches.resize(psb->m_tetras.size());
    psb->m_tetraScratchesTn.resize(psb->m_tetras.size());
    printf("Nodes:  %u\r\n", psb->m_nodes.size());
    printf("Links:  %u\r\n", psb->m_links.size());
    printf("Faces:  %u\r\n", psb->m_faces.size());
    printf("Tetras: %u\r\n", psb->m_tetras.size());

    fs.close();
    return psb;
}

void btSoftBodyHelpers::generateBoundaryFaces(btSoftBody* psb)
{
    int counter = 0;
    for (int i = 0; i < psb->m_nodes.size(); ++i)
    {
        psb->m_nodes[i].index = counter++;
    }
    typedef btAlignedObjectArray<int> Index;
    btAlignedObjectArray<Index> indices;
    indices.resize(psb->m_tetras.size());
    for (int i = 0; i < indices.size(); ++i)
    {
        Index index;
        index.push_back(psb->m_tetras[i].m_n[0]->index);
        index.push_back(psb->m_tetras[i].m_n[1]->index);
        index.push_back(psb->m_tetras[i].m_n[2]->index);
        index.push_back(psb->m_tetras[i].m_n[3]->index);
        indices[i] = index;
    }
    
    std::map<std::vector<int>, std::vector<int> > dict;
    for (int i = 0; i < indices.size(); ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            std::vector<int> f;
            if (j == 0)
            {
                f.push_back(indices[i][1]);
                f.push_back(indices[i][0]);
                f.push_back(indices[i][2]);
            }
            if (j == 1)
            {
                f.push_back(indices[i][3]);
                f.push_back(indices[i][0]);
                f.push_back(indices[i][1]);
            }
            if (j == 2)
            {
                f.push_back(indices[i][3]);
                f.push_back(indices[i][1]);
                f.push_back(indices[i][2]);
            }
            if (j == 3)
            {
                f.push_back(indices[i][2]);
                f.push_back(indices[i][0]);
                f.push_back(indices[i][3]);
            }
            std::vector<int> f_sorted = f;
            std::sort(f_sorted.begin(), f_sorted.end());
            if (dict.find(f_sorted) != dict.end())
            {
                dict.erase(f_sorted);
            }
            else
            {
                dict.insert(std::make_pair(f_sorted, f));
            }
        }
    }
    
    for (std::map<std::vector<int>, std::vector<int> >::iterator it = dict.begin(); it != dict.end(); ++it)
    {
        std::vector<int> f = it->second;
        psb->appendFace(f[0], f[1], f[2]);
    }
}

void btSoftBodyHelpers::writeObj(const char* filename, const btSoftBody* psb)
{
    std::ofstream fs;
    fs.open(filename);
    btAssert(fs);
    for (int i = 0; i < psb->m_nodes.size(); ++i)
    {
        fs << "v";
        for (int d = 0; d < 3; d++)
        {
             fs << " " << psb->m_nodes[i].m_x[d];
        }
        fs << "\n";
    }
    
    for (int i = 0; i < psb->m_faces.size(); ++i)
    {
        fs << "f";
        for (int n = 0; n < 3; n++)
        {
            fs << " " << psb->m_faces[i].m_n[n]->index + 1;
        }
        fs << "\n";
    }
    fs.close();
}

void btSoftBodyHelpers::duplicateFaces(const char* filename, const btSoftBody* psb)
{
    std::ifstream fs_read;
    fs_read.open(filename);
    std::string line;
    btVector3 pos;
    btAlignedObjectArray<btAlignedObjectArray<int> > additional_faces;
    while (std::getline(fs_read, line))
    {
        std::stringstream ss(line);
        if (line[0] == 'v')
        {
        }
        else if (line[0] == 'f')
        {
            ss.ignore();
            int id0, id1, id2;
            ss >> id0;
            ss >> id1;
            ss >> id2;
            btAlignedObjectArray<int> new_face;
            new_face.push_back(id1);
            new_face.push_back(id0);
            new_face.push_back(id2);
            additional_faces.push_back(new_face);
        }
    }
    fs_read.close();

    std::ofstream fs_write;
    fs_write.open(filename, std::ios_base::app);
    for (int i = 0; i < additional_faces.size(); ++i)
    {
        fs_write << "f";
        for (int n = 0; n < 3; n++)
        {
            fs_write << " " << additional_faces[i][n];
        }
        fs_write << "\n";
    }
    fs_write.close();
}

// Given a simplex with vertices a,b,c,d, find the barycentric weights of p in this simplex
void btSoftBodyHelpers::getBarycentricWeights(const btVector3& a, const btVector3& b, const btVector3& c, const btVector3& d, const btVector3& p, btVector4& bary)
{
    btVector3 vap = p - a;
    btVector3 vbp = p - b;
    
    btVector3 vab = b - a;
    btVector3 vac = c - a;
    btVector3 vad = d - a;
    
    btVector3 vbc = c - b;
    btVector3 vbd = d - b;
    btScalar va6 = (vbp.cross(vbd)).dot(vbc);
    btScalar vb6 = (vap.cross(vac)).dot(vad);
    btScalar vc6 = (vap.cross(vad)).dot(vab);
    btScalar vd6 = (vap.cross(vab)).dot(vac);
    btScalar v6 = btScalar(1) / (vab.cross(vac).dot(vad));
    bary = btVector4(va6*v6, vb6*v6, vc6*v6, vd6*v6);
}

// Iterate through all render nodes to find the simulation tetrahedron that contains the render node and record the barycentric weights
// If the node is not inside any tetrahedron, assign it to the tetrahedron in which the node has the least negative barycentric weight
void btSoftBodyHelpers::interpolateBarycentricWeights(btSoftBody* psb)
{
    psb->m_renderNodesInterpolationWeights.resize(psb->m_renderNodes.size());
    psb->m_renderNodesParents.resize(psb->m_renderNodes.size());
    for (int i = 0; i < psb->m_renderNodes.size(); ++i)
    {
        const btVector3& p = psb->m_renderNodes[i].m_x;
        btVector4 bary;
        btVector4 optimal_bary;
        btScalar min_bary_weight = -1e3;
        btAlignedObjectArray<const btSoftBody::Node*> optimal_parents;
        bool found = false;
        for (int j = 0; j < psb->m_tetras.size(); ++j)
        {
            const btSoftBody::Tetra& t = psb->m_tetras[j];
            getBarycentricWeights(t.m_n[0]->m_x, t.m_n[1]->m_x, t.m_n[2]->m_x, t.m_n[3]->m_x, p, bary);
            btScalar new_min_bary_weight = bary[0];
            for (int k = 1; k < 4; ++k)
            {
                new_min_bary_weight = btMin(new_min_bary_weight, bary[k]);
            }
            if (new_min_bary_weight > min_bary_weight)
            {
                btAlignedObjectArray<const btSoftBody::Node*> parents;
                parents.push_back(t.m_n[0]);
                parents.push_back(t.m_n[1]);
                parents.push_back(t.m_n[2]);
                parents.push_back(t.m_n[3]);
                optimal_parents = parents;
                optimal_bary = bary;
                min_bary_weight = new_min_bary_weight;
                // stop searching if p is inside the tetrahedron at hand
                if (bary[0]>=0. && bary[1]>=0. && bary[2]>=0. && bary[3]>=0.)
                {
                    break;
                }
            }
        }
        psb->m_renderNodesInterpolationWeights[i] = optimal_bary;
        psb->m_renderNodesParents[i] = optimal_parents;
    }
}

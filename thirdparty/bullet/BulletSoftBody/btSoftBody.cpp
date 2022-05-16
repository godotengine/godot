/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
///btSoftBody implementation by Nathanael Presson

#include "btSoftBodyInternals.h"
#include "BulletSoftBody/btSoftBodySolvers.h"
#include "btSoftBodyData.h"
#include "LinearMath/btSerializer.h"
#include "LinearMath/btImplicitQRSVD.h"
#include "LinearMath/btAlignedAllocator.h"
#include "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h"
#include "BulletDynamics/Featherstone/btMultiBodyConstraint.h"
#include "BulletCollision/NarrowPhaseCollision/btGjkEpa2.h"
#include "BulletCollision/CollisionShapes/btTriangleShape.h"
#include <iostream>
//
static inline btDbvtNode* buildTreeBottomUp(btAlignedObjectArray<btDbvtNode*>& leafNodes, btAlignedObjectArray<btAlignedObjectArray<int> >& adj)
{
	int N = leafNodes.size();
	if (N == 0)
	{
		return NULL;
	}
	while (N > 1)
	{
		btAlignedObjectArray<bool> marked;
		btAlignedObjectArray<btDbvtNode*> newLeafNodes;
		btAlignedObjectArray<std::pair<int, int> > childIds;
		btAlignedObjectArray<btAlignedObjectArray<int> > newAdj;
		marked.resize(N);
		for (int i = 0; i < N; ++i)
			marked[i] = false;

		// pair adjacent nodes into new(parent) node
		for (int i = 0; i < N; ++i)
		{
			if (marked[i])
				continue;
			bool merged = false;
			for (int j = 0; j < adj[i].size(); ++j)
			{
				int n = adj[i][j];
				if (!marked[adj[i][j]])
				{
					btDbvtNode* node = new (btAlignedAlloc(sizeof(btDbvtNode), 16)) btDbvtNode();
					node->parent = NULL;
					node->childs[0] = leafNodes[i];
					node->childs[1] = leafNodes[n];
					leafNodes[i]->parent = node;
					leafNodes[n]->parent = node;
					newLeafNodes.push_back(node);
					childIds.push_back(std::make_pair(i, n));
					merged = true;
					marked[n] = true;
					break;
				}
			}
			if (!merged)
			{
				newLeafNodes.push_back(leafNodes[i]);
				childIds.push_back(std::make_pair(i, -1));
			}
			marked[i] = true;
		}
		// update adjacency matrix
		newAdj.resize(newLeafNodes.size());
		for (int i = 0; i < newLeafNodes.size(); ++i)
		{
			for (int j = i + 1; j < newLeafNodes.size(); ++j)
			{
				bool neighbor = false;
				const btAlignedObjectArray<int>& leftChildNeighbors = adj[childIds[i].first];
				for (int k = 0; k < leftChildNeighbors.size(); ++k)
				{
					if (leftChildNeighbors[k] == childIds[j].first || leftChildNeighbors[k] == childIds[j].second)
					{
						neighbor = true;
						break;
					}
				}
				if (!neighbor && childIds[i].second != -1)
				{
					const btAlignedObjectArray<int>& rightChildNeighbors = adj[childIds[i].second];
					for (int k = 0; k < rightChildNeighbors.size(); ++k)
					{
						if (rightChildNeighbors[k] == childIds[j].first || rightChildNeighbors[k] == childIds[j].second)
						{
							neighbor = true;
							break;
						}
					}
				}
				if (neighbor)
				{
					newAdj[i].push_back(j);
					newAdj[j].push_back(i);
				}
			}
		}
		leafNodes = newLeafNodes;
		//this assignment leaks memory, the assignment doesn't do a deep copy, for now a manual copy
		//adj = newAdj;
		adj.clear();
		adj.resize(newAdj.size());
		for (int i = 0; i < newAdj.size(); i++)
		{
			for (int j = 0; j < newAdj[i].size(); j++)
			{
				adj[i].push_back(newAdj[i][j]);
			}
		}
		N = leafNodes.size();
	}
	return leafNodes[0];
}

//
btSoftBody::btSoftBody(btSoftBodyWorldInfo* worldInfo, int node_count, const btVector3* x, const btScalar* m)
	: m_softBodySolver(0), m_worldInfo(worldInfo)
{
	/* Init		*/
	initDefaults();

	/* Default material	*/
	Material* pm = appendMaterial();
	pm->m_kLST = 1;
	pm->m_kAST = 1;
	pm->m_kVST = 1;
	pm->m_flags = fMaterial::Default;

	/* Nodes			*/
	const btScalar margin = getCollisionShape()->getMargin();
	m_nodes.resize(node_count);
	m_X.resize(node_count);
	for (int i = 0, ni = node_count; i < ni; ++i)
	{
		Node& n = m_nodes[i];
		ZeroInitialize(n);
		n.m_x = x ? *x++ : btVector3(0, 0, 0);
		n.m_q = n.m_x;
		n.m_im = m ? *m++ : 1;
		n.m_im = n.m_im > 0 ? 1 / n.m_im : 0;
		n.m_leaf = m_ndbvt.insert(btDbvtVolume::FromCR(n.m_x, margin), &n);
		n.m_material = pm;
		m_X[i] = n.m_x;
	}
	updateBounds();
	setCollisionQuadrature(3);
	m_fdbvnt = 0;
}

btSoftBody::btSoftBody(btSoftBodyWorldInfo* worldInfo)
	: m_worldInfo(worldInfo)
{
	initDefaults();
}

void btSoftBody::initDefaults()
{
	m_internalType = CO_SOFT_BODY;
	m_cfg.aeromodel = eAeroModel::V_Point;
	m_cfg.kVCF = 1;
	m_cfg.kDG = 0;
	m_cfg.kLF = 0;
	m_cfg.kDP = 0;
	m_cfg.kPR = 0;
	m_cfg.kVC = 0;
	m_cfg.kDF = (btScalar)0.2;
	m_cfg.kMT = 0;
	m_cfg.kCHR = (btScalar)1.0;
	m_cfg.kKHR = (btScalar)0.1;
	m_cfg.kSHR = (btScalar)1.0;
	m_cfg.kAHR = (btScalar)0.7;
	m_cfg.kSRHR_CL = (btScalar)0.1;
	m_cfg.kSKHR_CL = (btScalar)1;
	m_cfg.kSSHR_CL = (btScalar)0.5;
	m_cfg.kSR_SPLT_CL = (btScalar)0.5;
	m_cfg.kSK_SPLT_CL = (btScalar)0.5;
	m_cfg.kSS_SPLT_CL = (btScalar)0.5;
	m_cfg.maxvolume = (btScalar)1;
	m_cfg.timescale = 1;
	m_cfg.viterations = 0;
	m_cfg.piterations = 1;
	m_cfg.diterations = 0;
	m_cfg.citerations = 4;
	m_cfg.drag = 0;
	m_cfg.m_maxStress = 0;
	m_cfg.collisions = fCollision::Default;
	m_pose.m_bvolume = false;
	m_pose.m_bframe = false;
	m_pose.m_volume = 0;
	m_pose.m_com = btVector3(0, 0, 0);
	m_pose.m_rot.setIdentity();
	m_pose.m_scl.setIdentity();
	m_tag = 0;
	m_timeacc = 0;
	m_bUpdateRtCst = true;
	m_bounds[0] = btVector3(0, 0, 0);
	m_bounds[1] = btVector3(0, 0, 0);
	m_worldTransform.setIdentity();
	setSolver(eSolverPresets::Positions);

	/* Collision shape	*/
	///for now, create a collision shape internally
	m_collisionShape = new btSoftBodyCollisionShape(this);
	m_collisionShape->setMargin(0.25f);

	m_worldTransform.setIdentity();

	m_windVelocity = btVector3(0, 0, 0);
	m_restLengthScale = btScalar(1.0);
	m_dampingCoefficient = 1.0;
	m_sleepingThreshold = .04;
	m_useSelfCollision = false;
	m_collisionFlags = 0;
	m_softSoftCollision = false;
	m_maxSpeedSquared = 0;
	m_repulsionStiffness = 0.5;
	m_gravityFactor = 1;
	m_cacheBarycenter = false;
	m_fdbvnt = 0;

	// reduced flag
	m_reducedModel = false;
}

//
btSoftBody::~btSoftBody()
{
	//for now, delete the internal shape
	delete m_collisionShape;
	int i;

	releaseClusters();
	for (i = 0; i < m_materials.size(); ++i)
		btAlignedFree(m_materials[i]);
	for (i = 0; i < m_joints.size(); ++i)
		btAlignedFree(m_joints[i]);
	if (m_fdbvnt)
		delete m_fdbvnt;
}

//
bool btSoftBody::checkLink(int node0, int node1) const
{
	return (checkLink(&m_nodes[node0], &m_nodes[node1]));
}

//
bool btSoftBody::checkLink(const Node* node0, const Node* node1) const
{
	const Node* n[] = {node0, node1};
	for (int i = 0, ni = m_links.size(); i < ni; ++i)
	{
		const Link& l = m_links[i];
		if ((l.m_n[0] == n[0] && l.m_n[1] == n[1]) ||
			(l.m_n[0] == n[1] && l.m_n[1] == n[0]))
		{
			return (true);
		}
	}
	return (false);
}

//
bool btSoftBody::checkFace(int node0, int node1, int node2) const
{
	const Node* n[] = {&m_nodes[node0],
					   &m_nodes[node1],
					   &m_nodes[node2]};
	for (int i = 0, ni = m_faces.size(); i < ni; ++i)
	{
		const Face& f = m_faces[i];
		int c = 0;
		for (int j = 0; j < 3; ++j)
		{
			if ((f.m_n[j] == n[0]) ||
				(f.m_n[j] == n[1]) ||
				(f.m_n[j] == n[2]))
				c |= 1 << j;
			else
				break;
		}
		if (c == 7) return (true);
	}
	return (false);
}

//
btSoftBody::Material* btSoftBody::appendMaterial()
{
	Material* pm = new (btAlignedAlloc(sizeof(Material), 16)) Material();
	if (m_materials.size() > 0)
		*pm = *m_materials[0];
	else
		ZeroInitialize(*pm);
	m_materials.push_back(pm);
	return (pm);
}

//
void btSoftBody::appendNote(const char* text,
							const btVector3& o,
							const btVector4& c,
							Node* n0,
							Node* n1,
							Node* n2,
							Node* n3)
{
	Note n;
	ZeroInitialize(n);
	n.m_rank = 0;
	n.m_text = text;
	n.m_offset = o;
	n.m_coords[0] = c.x();
	n.m_coords[1] = c.y();
	n.m_coords[2] = c.z();
	n.m_coords[3] = c.w();
	n.m_nodes[0] = n0;
	n.m_rank += n0 ? 1 : 0;
	n.m_nodes[1] = n1;
	n.m_rank += n1 ? 1 : 0;
	n.m_nodes[2] = n2;
	n.m_rank += n2 ? 1 : 0;
	n.m_nodes[3] = n3;
	n.m_rank += n3 ? 1 : 0;
	m_notes.push_back(n);
}

//
void btSoftBody::appendNote(const char* text,
							const btVector3& o,
							Node* feature)
{
	appendNote(text, o, btVector4(1, 0, 0, 0), feature);
}

//
void btSoftBody::appendNote(const char* text,
							const btVector3& o,
							Link* feature)
{
	static const btScalar w = 1 / (btScalar)2;
	appendNote(text, o, btVector4(w, w, 0, 0), feature->m_n[0],
			   feature->m_n[1]);
}

//
void btSoftBody::appendNote(const char* text,
							const btVector3& o,
							Face* feature)
{
	static const btScalar w = 1 / (btScalar)3;
	appendNote(text, o, btVector4(w, w, w, 0), feature->m_n[0],
			   feature->m_n[1],
			   feature->m_n[2]);
}

//
void btSoftBody::appendNode(const btVector3& x, btScalar m)
{
	if (m_nodes.capacity() == m_nodes.size())
	{
		pointersToIndices();
		m_nodes.reserve(m_nodes.size() * 2 + 1);
		indicesToPointers();
	}
	const btScalar margin = getCollisionShape()->getMargin();
	m_nodes.push_back(Node());
	Node& n = m_nodes[m_nodes.size() - 1];
	ZeroInitialize(n);
	n.m_x = x;
	n.m_q = n.m_x;
	n.m_im = m > 0 ? 1 / m : 0;
	n.m_material = m_materials[0];
	n.m_leaf = m_ndbvt.insert(btDbvtVolume::FromCR(n.m_x, margin), &n);
}

//
void btSoftBody::appendLink(int model, Material* mat)
{
	Link l;
	if (model >= 0)
		l = m_links[model];
	else
	{
		ZeroInitialize(l);
		l.m_material = mat ? mat : m_materials[0];
	}
	m_links.push_back(l);
}

//
void btSoftBody::appendLink(int node0,
							int node1,
							Material* mat,
							bool bcheckexist)
{
	appendLink(&m_nodes[node0], &m_nodes[node1], mat, bcheckexist);
}

//
void btSoftBody::appendLink(Node* node0,
							Node* node1,
							Material* mat,
							bool bcheckexist)
{
	if ((!bcheckexist) || (!checkLink(node0, node1)))
	{
		appendLink(-1, mat);
		Link& l = m_links[m_links.size() - 1];
		l.m_n[0] = node0;
		l.m_n[1] = node1;
		l.m_rl = (l.m_n[0]->m_x - l.m_n[1]->m_x).length();
		m_bUpdateRtCst = true;
	}
}

//
void btSoftBody::appendFace(int model, Material* mat)
{
	Face f;
	if (model >= 0)
	{
		f = m_faces[model];
	}
	else
	{
		ZeroInitialize(f);
		f.m_material = mat ? mat : m_materials[0];
	}
	m_faces.push_back(f);
}

//
void btSoftBody::appendFace(int node0, int node1, int node2, Material* mat)
{
	if (node0 == node1)
		return;
	if (node1 == node2)
		return;
	if (node2 == node0)
		return;

	appendFace(-1, mat);
	Face& f = m_faces[m_faces.size() - 1];
	btAssert(node0 != node1);
	btAssert(node1 != node2);
	btAssert(node2 != node0);
	f.m_n[0] = &m_nodes[node0];
	f.m_n[1] = &m_nodes[node1];
	f.m_n[2] = &m_nodes[node2];
	f.m_ra = AreaOf(f.m_n[0]->m_x,
					f.m_n[1]->m_x,
					f.m_n[2]->m_x);
	m_bUpdateRtCst = true;
}

//
void btSoftBody::appendTetra(int model, Material* mat)
{
	Tetra t;
	if (model >= 0)
		t = m_tetras[model];
	else
	{
		ZeroInitialize(t);
		t.m_material = mat ? mat : m_materials[0];
	}
	m_tetras.push_back(t);
}

//
void btSoftBody::appendTetra(int node0,
							 int node1,
							 int node2,
							 int node3,
							 Material* mat)
{
	appendTetra(-1, mat);
	Tetra& t = m_tetras[m_tetras.size() - 1];
	t.m_n[0] = &m_nodes[node0];
	t.m_n[1] = &m_nodes[node1];
	t.m_n[2] = &m_nodes[node2];
	t.m_n[3] = &m_nodes[node3];
	t.m_rv = VolumeOf(t.m_n[0]->m_x, t.m_n[1]->m_x, t.m_n[2]->m_x, t.m_n[3]->m_x);
	m_bUpdateRtCst = true;
}

//

void btSoftBody::appendAnchor(int node, btRigidBody* body, bool disableCollisionBetweenLinkedBodies, btScalar influence)
{
	btVector3 local = body->getWorldTransform().inverse() * m_nodes[node].m_x;
	appendAnchor(node, body, local, disableCollisionBetweenLinkedBodies, influence);
}

//
void btSoftBody::appendAnchor(int node, btRigidBody* body, const btVector3& localPivot, bool disableCollisionBetweenLinkedBodies, btScalar influence)
{
	if (disableCollisionBetweenLinkedBodies)
	{
		if (m_collisionDisabledObjects.findLinearSearch(body) == m_collisionDisabledObjects.size())
		{
			m_collisionDisabledObjects.push_back(body);
		}
	}

	Anchor a;
	a.m_node = &m_nodes[node];
	a.m_body = body;
	a.m_local = localPivot;
	a.m_node->m_battach = 1;
	a.m_influence = influence;
	m_anchors.push_back(a);
}

//
void btSoftBody::appendDeformableAnchor(int node, btRigidBody* body)
{
	DeformableNodeRigidAnchor c;
	btSoftBody::Node& n = m_nodes[node];
	const btScalar ima = n.m_im;
	const btScalar imb = body->getInvMass();
	btVector3 nrm;
	const btCollisionShape* shp = body->getCollisionShape();
	const btTransform& wtr = body->getWorldTransform();
	btScalar dst =
		m_worldInfo->m_sparsesdf.Evaluate(
			wtr.invXform(m_nodes[node].m_x),
			shp,
			nrm,
			0);

	c.m_cti.m_colObj = body;
	c.m_cti.m_normal = wtr.getBasis() * nrm;
	c.m_cti.m_offset = dst;
	c.m_node = &m_nodes[node];
	const btScalar fc = m_cfg.kDF * body->getFriction();
	c.m_c2 = ima;
	c.m_c3 = fc;
	c.m_c4 = body->isStaticOrKinematicObject() ? m_cfg.kKHR : m_cfg.kCHR;
	static const btMatrix3x3 iwiStatic(0, 0, 0, 0, 0, 0, 0, 0, 0);
	const btMatrix3x3& iwi = body->getInvInertiaTensorWorld();
	const btVector3 ra = n.m_x - wtr.getOrigin();

	c.m_c0 = ImpulseMatrix(1, ima, imb, iwi, ra);
	c.m_c1 = ra;
	c.m_local = body->getWorldTransform().inverse() * m_nodes[node].m_x;
	c.m_node->m_battach = 1;
	m_deformableAnchors.push_back(c);
}

void btSoftBody::removeAnchor(int node)
{
	const btSoftBody::Node& n = m_nodes[node];
	for (int i = 0; i < m_deformableAnchors.size();)
	{
		const DeformableNodeRigidAnchor& c = m_deformableAnchors[i];
		if (c.m_node == &n)
		{
			m_deformableAnchors.removeAtIndex(i);
		}
		else
		{
			i++;
		}
	}
}

//
void btSoftBody::appendDeformableAnchor(int node, btMultiBodyLinkCollider* link)
{
	DeformableNodeRigidAnchor c;
	btSoftBody::Node& n = m_nodes[node];
	const btScalar ima = n.m_im;
	btVector3 nrm;
	const btCollisionShape* shp = link->getCollisionShape();
	const btTransform& wtr = link->getWorldTransform();
	btScalar dst =
		m_worldInfo->m_sparsesdf.Evaluate(
			wtr.invXform(m_nodes[node].m_x),
			shp,
			nrm,
			0);
	c.m_cti.m_colObj = link;
	c.m_cti.m_normal = wtr.getBasis() * nrm;
	c.m_cti.m_offset = dst;
	c.m_node = &m_nodes[node];
	const btScalar fc = m_cfg.kDF * link->getFriction();
	c.m_c2 = ima;
	c.m_c3 = fc;
	c.m_c4 = link->isStaticOrKinematicObject() ? m_cfg.kKHR : m_cfg.kCHR;
	btVector3 normal = c.m_cti.m_normal;
	btVector3 t1 = generateUnitOrthogonalVector(normal);
	btVector3 t2 = btCross(normal, t1);
	btMultiBodyJacobianData jacobianData_normal, jacobianData_t1, jacobianData_t2;
	findJacobian(link, jacobianData_normal, c.m_node->m_x, normal);
	findJacobian(link, jacobianData_t1, c.m_node->m_x, t1);
	findJacobian(link, jacobianData_t2, c.m_node->m_x, t2);

	btScalar* J_n = &jacobianData_normal.m_jacobians[0];
	btScalar* J_t1 = &jacobianData_t1.m_jacobians[0];
	btScalar* J_t2 = &jacobianData_t2.m_jacobians[0];

	btScalar* u_n = &jacobianData_normal.m_deltaVelocitiesUnitImpulse[0];
	btScalar* u_t1 = &jacobianData_t1.m_deltaVelocitiesUnitImpulse[0];
	btScalar* u_t2 = &jacobianData_t2.m_deltaVelocitiesUnitImpulse[0];

	btMatrix3x3 rot(normal.getX(), normal.getY(), normal.getZ(),
					t1.getX(), t1.getY(), t1.getZ(),
					t2.getX(), t2.getY(), t2.getZ());  // world frame to local frame
	const int ndof = link->m_multiBody->getNumDofs() + 6;
	btMatrix3x3 local_impulse_matrix = (Diagonal(n.m_im) + OuterProduct(J_n, J_t1, J_t2, u_n, u_t1, u_t2, ndof)).inverse();
	c.m_c0 = rot.transpose() * local_impulse_matrix * rot;
	c.jacobianData_normal = jacobianData_normal;
	c.jacobianData_t1 = jacobianData_t1;
	c.jacobianData_t2 = jacobianData_t2;
	c.t1 = t1;
	c.t2 = t2;
	const btVector3 ra = n.m_x - wtr.getOrigin();
	c.m_c1 = ra;
	c.m_local = link->getWorldTransform().inverse() * m_nodes[node].m_x;
	c.m_node->m_battach = 1;
	m_deformableAnchors.push_back(c);
}
//
void btSoftBody::appendLinearJoint(const LJoint::Specs& specs, Cluster* body0, Body body1)
{
	LJoint* pj = new (btAlignedAlloc(sizeof(LJoint), 16)) LJoint();
	pj->m_bodies[0] = body0;
	pj->m_bodies[1] = body1;
	pj->m_refs[0] = pj->m_bodies[0].xform().inverse() * specs.position;
	pj->m_refs[1] = pj->m_bodies[1].xform().inverse() * specs.position;
	pj->m_cfm = specs.cfm;
	pj->m_erp = specs.erp;
	pj->m_split = specs.split;
	m_joints.push_back(pj);
}

//
void btSoftBody::appendLinearJoint(const LJoint::Specs& specs, Body body)
{
	appendLinearJoint(specs, m_clusters[0], body);
}

//
void btSoftBody::appendLinearJoint(const LJoint::Specs& specs, btSoftBody* body)
{
	appendLinearJoint(specs, m_clusters[0], body->m_clusters[0]);
}

//
void btSoftBody::appendAngularJoint(const AJoint::Specs& specs, Cluster* body0, Body body1)
{
	AJoint* pj = new (btAlignedAlloc(sizeof(AJoint), 16)) AJoint();
	pj->m_bodies[0] = body0;
	pj->m_bodies[1] = body1;
	pj->m_refs[0] = pj->m_bodies[0].xform().inverse().getBasis() * specs.axis;
	pj->m_refs[1] = pj->m_bodies[1].xform().inverse().getBasis() * specs.axis;
	pj->m_cfm = specs.cfm;
	pj->m_erp = specs.erp;
	pj->m_split = specs.split;
	pj->m_icontrol = specs.icontrol;
	m_joints.push_back(pj);
}

//
void btSoftBody::appendAngularJoint(const AJoint::Specs& specs, Body body)
{
	appendAngularJoint(specs, m_clusters[0], body);
}

//
void btSoftBody::appendAngularJoint(const AJoint::Specs& specs, btSoftBody* body)
{
	appendAngularJoint(specs, m_clusters[0], body->m_clusters[0]);
}

//
void btSoftBody::addForce(const btVector3& force)
{
	for (int i = 0, ni = m_nodes.size(); i < ni; ++i) addForce(force, i);
}

//
void btSoftBody::addForce(const btVector3& force, int node)
{
	Node& n = m_nodes[node];
	if (n.m_im > 0)
	{
		n.m_f += force;
	}
}

void btSoftBody::addAeroForceToNode(const btVector3& windVelocity, int nodeIndex)
{
	btAssert(nodeIndex >= 0 && nodeIndex < m_nodes.size());

	const btScalar dt = m_sst.sdt;
	const btScalar kLF = m_cfg.kLF;
	const btScalar kDG = m_cfg.kDG;
	//const btScalar kPR = m_cfg.kPR;
	//const btScalar kVC = m_cfg.kVC;
	const bool as_lift = kLF > 0;
	const bool as_drag = kDG > 0;
	const bool as_aero = as_lift || as_drag;
	const bool as_vaero = as_aero && (m_cfg.aeromodel < btSoftBody::eAeroModel::F_TwoSided);

	Node& n = m_nodes[nodeIndex];

	if (n.m_im > 0)
	{
		btSoftBody::sMedium medium;

		EvaluateMedium(m_worldInfo, n.m_x, medium);
		medium.m_velocity = windVelocity;
		medium.m_density = m_worldInfo->air_density;

		/* Aerodynamics			*/
		if (as_vaero)
		{
			const btVector3 rel_v = n.m_v - medium.m_velocity;
			const btScalar rel_v_len = rel_v.length();
			const btScalar rel_v2 = rel_v.length2();

			if (rel_v2 > SIMD_EPSILON)
			{
				const btVector3 rel_v_nrm = rel_v.normalized();
				btVector3 nrm = n.m_n;

				if (m_cfg.aeromodel == btSoftBody::eAeroModel::V_TwoSidedLiftDrag)
				{
					nrm *= (btScalar)((btDot(nrm, rel_v) < 0) ? -1 : +1);
					btVector3 fDrag(0, 0, 0);
					btVector3 fLift(0, 0, 0);

					btScalar n_dot_v = nrm.dot(rel_v_nrm);
					btScalar tri_area = 0.5f * n.m_area;

					fDrag = 0.5f * kDG * medium.m_density * rel_v2 * tri_area * n_dot_v * (-rel_v_nrm);

					// Check angle of attack
					// cos(10º) = 0.98480
					if (0 < n_dot_v && n_dot_v < 0.98480f)
						fLift = 0.5f * kLF * medium.m_density * rel_v_len * tri_area * btSqrt(1.0f - n_dot_v * n_dot_v) * (nrm.cross(rel_v_nrm).cross(rel_v_nrm));

					// Check if the velocity change resulted by aero drag force exceeds the current velocity of the node.
					btVector3 del_v_by_fDrag = fDrag * n.m_im * m_sst.sdt;
					btScalar del_v_by_fDrag_len2 = del_v_by_fDrag.length2();
					btScalar v_len2 = n.m_v.length2();

					if (del_v_by_fDrag_len2 >= v_len2 && del_v_by_fDrag_len2 > 0)
					{
						btScalar del_v_by_fDrag_len = del_v_by_fDrag.length();
						btScalar v_len = n.m_v.length();
						fDrag *= btScalar(0.8) * (v_len / del_v_by_fDrag_len);
					}

					n.m_f += fDrag;
					n.m_f += fLift;
				}
				else if (m_cfg.aeromodel == btSoftBody::eAeroModel::V_Point || m_cfg.aeromodel == btSoftBody::eAeroModel::V_OneSided || m_cfg.aeromodel == btSoftBody::eAeroModel::V_TwoSided)
				{
					if (m_cfg.aeromodel == btSoftBody::eAeroModel::V_TwoSided)
						nrm *= (btScalar)((btDot(nrm, rel_v) < 0) ? -1 : +1);

					const btScalar dvn = btDot(rel_v, nrm);
					/* Compute forces	*/
					if (dvn > 0)
					{
						btVector3 force(0, 0, 0);
						const btScalar c0 = n.m_area * dvn * rel_v2 / 2;
						const btScalar c1 = c0 * medium.m_density;
						force += nrm * (-c1 * kLF);
						force += rel_v.normalized() * (-c1 * kDG);
						ApplyClampedForce(n, force, dt);
					}
				}
			}
		}
	}
}

void btSoftBody::addAeroForceToFace(const btVector3& windVelocity, int faceIndex)
{
	const btScalar dt = m_sst.sdt;
	const btScalar kLF = m_cfg.kLF;
	const btScalar kDG = m_cfg.kDG;
	//	const btScalar kPR = m_cfg.kPR;
	//	const btScalar kVC = m_cfg.kVC;
	const bool as_lift = kLF > 0;
	const bool as_drag = kDG > 0;
	const bool as_aero = as_lift || as_drag;
	const bool as_faero = as_aero && (m_cfg.aeromodel >= btSoftBody::eAeroModel::F_TwoSided);

	if (as_faero)
	{
		btSoftBody::Face& f = m_faces[faceIndex];

		btSoftBody::sMedium medium;

		const btVector3 v = (f.m_n[0]->m_v + f.m_n[1]->m_v + f.m_n[2]->m_v) / 3;
		const btVector3 x = (f.m_n[0]->m_x + f.m_n[1]->m_x + f.m_n[2]->m_x) / 3;
		EvaluateMedium(m_worldInfo, x, medium);
		medium.m_velocity = windVelocity;
		medium.m_density = m_worldInfo->air_density;
		const btVector3 rel_v = v - medium.m_velocity;
		const btScalar rel_v_len = rel_v.length();
		const btScalar rel_v2 = rel_v.length2();

		if (rel_v2 > SIMD_EPSILON)
		{
			const btVector3 rel_v_nrm = rel_v.normalized();
			btVector3 nrm = f.m_normal;

			if (m_cfg.aeromodel == btSoftBody::eAeroModel::F_TwoSidedLiftDrag)
			{
				nrm *= (btScalar)((btDot(nrm, rel_v) < 0) ? -1 : +1);

				btVector3 fDrag(0, 0, 0);
				btVector3 fLift(0, 0, 0);

				btScalar n_dot_v = nrm.dot(rel_v_nrm);
				btScalar tri_area = 0.5f * f.m_ra;

				fDrag = 0.5f * kDG * medium.m_density * rel_v2 * tri_area * n_dot_v * (-rel_v_nrm);

				// Check angle of attack
				// cos(10º) = 0.98480
				if (0 < n_dot_v && n_dot_v < 0.98480f)
					fLift = 0.5f * kLF * medium.m_density * rel_v_len * tri_area * btSqrt(1.0f - n_dot_v * n_dot_v) * (nrm.cross(rel_v_nrm).cross(rel_v_nrm));

				fDrag /= 3;
				fLift /= 3;

				for (int j = 0; j < 3; ++j)
				{
					if (f.m_n[j]->m_im > 0)
					{
						// Check if the velocity change resulted by aero drag force exceeds the current velocity of the node.
						btVector3 del_v_by_fDrag = fDrag * f.m_n[j]->m_im * m_sst.sdt;
						btScalar del_v_by_fDrag_len2 = del_v_by_fDrag.length2();
						btScalar v_len2 = f.m_n[j]->m_v.length2();

						if (del_v_by_fDrag_len2 >= v_len2 && del_v_by_fDrag_len2 > 0)
						{
							btScalar del_v_by_fDrag_len = del_v_by_fDrag.length();
							btScalar v_len = f.m_n[j]->m_v.length();
							fDrag *= btScalar(0.8) * (v_len / del_v_by_fDrag_len);
						}

						f.m_n[j]->m_f += fDrag;
						f.m_n[j]->m_f += fLift;
					}
				}
			}
			else if (m_cfg.aeromodel == btSoftBody::eAeroModel::F_OneSided || m_cfg.aeromodel == btSoftBody::eAeroModel::F_TwoSided)
			{
				if (m_cfg.aeromodel == btSoftBody::eAeroModel::F_TwoSided)
					nrm *= (btScalar)((btDot(nrm, rel_v) < 0) ? -1 : +1);

				const btScalar dvn = btDot(rel_v, nrm);
				/* Compute forces	*/
				if (dvn > 0)
				{
					btVector3 force(0, 0, 0);
					const btScalar c0 = f.m_ra * dvn * rel_v2;
					const btScalar c1 = c0 * medium.m_density;
					force += nrm * (-c1 * kLF);
					force += rel_v.normalized() * (-c1 * kDG);
					force /= 3;
					for (int j = 0; j < 3; ++j) ApplyClampedForce(*f.m_n[j], force, dt);
				}
			}
		}
	}
}

//
void btSoftBody::addVelocity(const btVector3& velocity)
{
	for (int i = 0, ni = m_nodes.size(); i < ni; ++i) addVelocity(velocity, i);
}

/* Set velocity for the entire body										*/
void btSoftBody::setVelocity(const btVector3& velocity)
{
	for (int i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		Node& n = m_nodes[i];
		if (n.m_im > 0)
		{
			n.m_v = velocity;
			n.m_vn = velocity;
		}
	}
}

//
void btSoftBody::addVelocity(const btVector3& velocity, int node)
{
	Node& n = m_nodes[node];
	if (n.m_im > 0)
	{
		n.m_v += velocity;
	}
}

//
void btSoftBody::setMass(int node, btScalar mass)
{
	m_nodes[node].m_im = mass > 0 ? 1 / mass : 0;
	m_bUpdateRtCst = true;
}

//
btScalar btSoftBody::getMass(int node) const
{
	return (m_nodes[node].m_im > 0 ? 1 / m_nodes[node].m_im : 0);
}

//
btScalar btSoftBody::getTotalMass() const
{
	btScalar mass = 0;
	for (int i = 0; i < m_nodes.size(); ++i)
	{
		mass += getMass(i);
	}
	return (mass);
}

//
void btSoftBody::setTotalMass(btScalar mass, bool fromfaces)
{
	int i;

	if (fromfaces)
	{
		for (i = 0; i < m_nodes.size(); ++i)
		{
			m_nodes[i].m_im = 0;
		}
		for (i = 0; i < m_faces.size(); ++i)
		{
			const Face& f = m_faces[i];
			const btScalar twicearea = AreaOf(f.m_n[0]->m_x,
											  f.m_n[1]->m_x,
											  f.m_n[2]->m_x);
			for (int j = 0; j < 3; ++j)
			{
				f.m_n[j]->m_im += twicearea;
			}
		}
		for (i = 0; i < m_nodes.size(); ++i)
		{
			m_nodes[i].m_im = 1 / m_nodes[i].m_im;
		}
	}
	const btScalar tm = getTotalMass();
	const btScalar itm = 1 / tm;
	for (i = 0; i < m_nodes.size(); ++i)
	{
		m_nodes[i].m_im /= itm * mass;
	}
	m_bUpdateRtCst = true;
}

//
void btSoftBody::setTotalDensity(btScalar density)
{
	setTotalMass(getVolume() * density, true);
}

//
void btSoftBody::setVolumeMass(btScalar mass)
{
	btAlignedObjectArray<btScalar> ranks;
	ranks.resize(m_nodes.size(), 0);
	int i;

	for (i = 0; i < m_nodes.size(); ++i)
	{
		m_nodes[i].m_im = 0;
	}
	for (i = 0; i < m_tetras.size(); ++i)
	{
		const Tetra& t = m_tetras[i];
		for (int j = 0; j < 4; ++j)
		{
			t.m_n[j]->m_im += btFabs(t.m_rv);
			ranks[int(t.m_n[j] - &m_nodes[0])] += 1;
		}
	}
	for (i = 0; i < m_nodes.size(); ++i)
	{
		if (m_nodes[i].m_im > 0)
		{
			m_nodes[i].m_im = ranks[i] / m_nodes[i].m_im;
		}
	}
	setTotalMass(mass, false);
}

//
void btSoftBody::setVolumeDensity(btScalar density)
{
	btScalar volume = 0;
	for (int i = 0; i < m_tetras.size(); ++i)
	{
		const Tetra& t = m_tetras[i];
		for (int j = 0; j < 4; ++j)
		{
			volume += btFabs(t.m_rv);
		}
	}
	setVolumeMass(volume * density / 6);
}

//
btVector3 btSoftBody::getLinearVelocity()
{
	btVector3 total_momentum = btVector3(0, 0, 0);
	for (int i = 0; i < m_nodes.size(); ++i)
	{
		btScalar mass = m_nodes[i].m_im == 0 ? 0 : 1.0 / m_nodes[i].m_im;
		total_momentum += mass * m_nodes[i].m_v;
	}
	btScalar total_mass = getTotalMass();
	return total_mass == 0 ? total_momentum : total_momentum / total_mass;
}

//
void btSoftBody::setLinearVelocity(const btVector3& linVel)
{
	btVector3 old_vel = getLinearVelocity();
	btVector3 diff = linVel - old_vel;
	for (int i = 0; i < m_nodes.size(); ++i)
		m_nodes[i].m_v += diff;
}

//
void btSoftBody::setAngularVelocity(const btVector3& angVel)
{
	btVector3 old_vel = getLinearVelocity();
	btVector3 com = getCenterOfMass();
	for (int i = 0; i < m_nodes.size(); ++i)
	{
		m_nodes[i].m_v = angVel.cross(m_nodes[i].m_x - com) + old_vel;
	}
}

//
btTransform btSoftBody::getRigidTransform()
{
	btVector3 t = getCenterOfMass();
	btMatrix3x3 S;
	S.setZero();
	// Get rotation that minimizes L2 difference: \sum_i || RX_i + t - x_i ||
	// It's important to make sure that S has the correct signs.
	// SVD is only unique up to the ordering of singular values.
	// SVD will manipulate U and V to ensure the ordering of singular values. If all three singular
	// vaues are negative, SVD will permute colums of U to make two of them positive.
	for (int i = 0; i < m_nodes.size(); ++i)
	{
		S -= OuterProduct(m_X[i], t - m_nodes[i].m_x);
	}
	btVector3 sigma;
	btMatrix3x3 U, V;
	singularValueDecomposition(S, U, sigma, V);
	btMatrix3x3 R = V * U.transpose();
	btTransform trs;
	trs.setIdentity();
	trs.setOrigin(t);
	trs.setBasis(R);
	return trs;
}

//
void btSoftBody::transformTo(const btTransform& trs)
{
	// get the current best rigid fit
	btTransform current_transform = getRigidTransform();
	// apply transform in material space
	btTransform new_transform = trs * current_transform.inverse();
	transform(new_transform);
}

//
void btSoftBody::transform(const btTransform& trs)
{
	const btScalar margin = getCollisionShape()->getMargin();
	ATTRIBUTE_ALIGNED16(btDbvtVolume)
	vol;

	for (int i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		Node& n = m_nodes[i];
		n.m_x = trs * n.m_x;
		n.m_q = trs * n.m_q;
		n.m_n = trs.getBasis() * n.m_n;
		vol = btDbvtVolume::FromCR(n.m_x, margin);

		m_ndbvt.update(n.m_leaf, vol);
	}
	updateNormals();
	updateBounds();
	updateConstants();
}

//
void btSoftBody::translate(const btVector3& trs)
{
	btTransform t;
	t.setIdentity();
	t.setOrigin(trs);
	transform(t);
}

//
void btSoftBody::rotate(const btQuaternion& rot)
{
	btTransform t;
	t.setIdentity();
	t.setRotation(rot);
	transform(t);
}

//
void btSoftBody::scale(const btVector3& scl)
{
	const btScalar margin = getCollisionShape()->getMargin();
	ATTRIBUTE_ALIGNED16(btDbvtVolume)
	vol;

	for (int i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		Node& n = m_nodes[i];
		n.m_x *= scl;
		n.m_q *= scl;
		vol = btDbvtVolume::FromCR(n.m_x, margin);
		m_ndbvt.update(n.m_leaf, vol);
	}
	updateNormals();
	updateBounds();
	updateConstants();
	initializeDmInverse();
}

//
btScalar btSoftBody::getRestLengthScale()
{
	return m_restLengthScale;
}

//
void btSoftBody::setRestLengthScale(btScalar restLengthScale)
{
	for (int i = 0, ni = m_links.size(); i < ni; ++i)
	{
		Link& l = m_links[i];
		l.m_rl = l.m_rl / m_restLengthScale * restLengthScale;
		l.m_c1 = l.m_rl * l.m_rl;
	}
	m_restLengthScale = restLengthScale;

	if (getActivationState() == ISLAND_SLEEPING)
		activate();
}

//
void btSoftBody::setPose(bool bvolume, bool bframe)
{
	m_pose.m_bvolume = bvolume;
	m_pose.m_bframe = bframe;
	int i, ni;

	/* Weights		*/
	const btScalar omass = getTotalMass();
	const btScalar kmass = omass * m_nodes.size() * 1000;
	btScalar tmass = omass;
	m_pose.m_wgh.resize(m_nodes.size());
	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		if (m_nodes[i].m_im <= 0) tmass += kmass;
	}
	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		Node& n = m_nodes[i];
		m_pose.m_wgh[i] = n.m_im > 0 ? 1 / (m_nodes[i].m_im * tmass) : kmass / tmass;
	}
	/* Pos		*/
	const btVector3 com = evaluateCom();
	m_pose.m_pos.resize(m_nodes.size());
	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		m_pose.m_pos[i] = m_nodes[i].m_x - com;
	}
	m_pose.m_volume = bvolume ? getVolume() : 0;
	m_pose.m_com = com;
	m_pose.m_rot.setIdentity();
	m_pose.m_scl.setIdentity();
	/* Aqq		*/
	m_pose.m_aqq[0] =
		m_pose.m_aqq[1] =
			m_pose.m_aqq[2] = btVector3(0, 0, 0);
	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		const btVector3& q = m_pose.m_pos[i];
		const btVector3 mq = m_pose.m_wgh[i] * q;
		m_pose.m_aqq[0] += mq.x() * q;
		m_pose.m_aqq[1] += mq.y() * q;
		m_pose.m_aqq[2] += mq.z() * q;
	}
	m_pose.m_aqq = m_pose.m_aqq.inverse();

	updateConstants();
}

void btSoftBody::resetLinkRestLengths()
{
	for (int i = 0, ni = m_links.size(); i < ni; ++i)
	{
		Link& l = m_links[i];
		l.m_rl = (l.m_n[0]->m_x - l.m_n[1]->m_x).length();
		l.m_c1 = l.m_rl * l.m_rl;
	}
}

//
btScalar btSoftBody::getVolume() const
{
	btScalar vol = 0;
	if (m_nodes.size() > 0)
	{
		int i, ni;

		const btVector3 org = m_nodes[0].m_x;
		for (i = 0, ni = m_faces.size(); i < ni; ++i)
		{
			const Face& f = m_faces[i];
			vol += btDot(f.m_n[0]->m_x - org, btCross(f.m_n[1]->m_x - org, f.m_n[2]->m_x - org));
		}
		vol /= (btScalar)6;
	}
	return (vol);
}

//
int btSoftBody::clusterCount() const
{
	return (m_clusters.size());
}

//
btVector3 btSoftBody::clusterCom(const Cluster* cluster)
{
	btVector3 com(0, 0, 0);
	for (int i = 0, ni = cluster->m_nodes.size(); i < ni; ++i)
	{
		com += cluster->m_nodes[i]->m_x * cluster->m_masses[i];
	}
	return (com * cluster->m_imass);
}

//
btVector3 btSoftBody::clusterCom(int cluster) const
{
	return (clusterCom(m_clusters[cluster]));
}

//
btVector3 btSoftBody::clusterVelocity(const Cluster* cluster, const btVector3& rpos)
{
	return (cluster->m_lv + btCross(cluster->m_av, rpos));
}

//
void btSoftBody::clusterVImpulse(Cluster* cluster, const btVector3& rpos, const btVector3& impulse)
{
	const btVector3 li = cluster->m_imass * impulse;
	const btVector3 ai = cluster->m_invwi * btCross(rpos, impulse);
	cluster->m_vimpulses[0] += li;
	cluster->m_lv += li;
	cluster->m_vimpulses[1] += ai;
	cluster->m_av += ai;
	cluster->m_nvimpulses++;
}

//
void btSoftBody::clusterDImpulse(Cluster* cluster, const btVector3& rpos, const btVector3& impulse)
{
	const btVector3 li = cluster->m_imass * impulse;
	const btVector3 ai = cluster->m_invwi * btCross(rpos, impulse);
	cluster->m_dimpulses[0] += li;
	cluster->m_dimpulses[1] += ai;
	cluster->m_ndimpulses++;
}

//
void btSoftBody::clusterImpulse(Cluster* cluster, const btVector3& rpos, const Impulse& impulse)
{
	if (impulse.m_asVelocity) clusterVImpulse(cluster, rpos, impulse.m_velocity);
	if (impulse.m_asDrift) clusterDImpulse(cluster, rpos, impulse.m_drift);
}

//
void btSoftBody::clusterVAImpulse(Cluster* cluster, const btVector3& impulse)
{
	const btVector3 ai = cluster->m_invwi * impulse;
	cluster->m_vimpulses[1] += ai;
	cluster->m_av += ai;
	cluster->m_nvimpulses++;
}

//
void btSoftBody::clusterDAImpulse(Cluster* cluster, const btVector3& impulse)
{
	const btVector3 ai = cluster->m_invwi * impulse;
	cluster->m_dimpulses[1] += ai;
	cluster->m_ndimpulses++;
}

//
void btSoftBody::clusterAImpulse(Cluster* cluster, const Impulse& impulse)
{
	if (impulse.m_asVelocity) clusterVAImpulse(cluster, impulse.m_velocity);
	if (impulse.m_asDrift) clusterDAImpulse(cluster, impulse.m_drift);
}

//
void btSoftBody::clusterDCImpulse(Cluster* cluster, const btVector3& impulse)
{
	cluster->m_dimpulses[0] += impulse * cluster->m_imass;
	cluster->m_ndimpulses++;
}

struct NodeLinks
{
	btAlignedObjectArray<int> m_links;
};

//
int btSoftBody::generateBendingConstraints(int distance, Material* mat)
{
	int i, j;

	if (distance > 1)
	{
		/* Build graph	*/
		const int n = m_nodes.size();
		const unsigned inf = (~(unsigned)0) >> 1;
		unsigned* adj = new unsigned[n * n];

#define IDX(_x_, _y_) ((_y_)*n + (_x_))
		for (j = 0; j < n; ++j)
		{
			for (i = 0; i < n; ++i)
			{
				if (i != j)
				{
					adj[IDX(i, j)] = adj[IDX(j, i)] = inf;
				}
				else
				{
					adj[IDX(i, j)] = adj[IDX(j, i)] = 0;
				}
			}
		}
		for (i = 0; i < m_links.size(); ++i)
		{
			const int ia = (int)(m_links[i].m_n[0] - &m_nodes[0]);
			const int ib = (int)(m_links[i].m_n[1] - &m_nodes[0]);
			adj[IDX(ia, ib)] = 1;
			adj[IDX(ib, ia)] = 1;
		}

		//special optimized case for distance == 2
		if (distance == 2)
		{
			btAlignedObjectArray<NodeLinks> nodeLinks;

			/* Build node links */
			nodeLinks.resize(m_nodes.size());

			for (i = 0; i < m_links.size(); ++i)
			{
				const int ia = (int)(m_links[i].m_n[0] - &m_nodes[0]);
				const int ib = (int)(m_links[i].m_n[1] - &m_nodes[0]);
				if (nodeLinks[ia].m_links.findLinearSearch(ib) == nodeLinks[ia].m_links.size())
					nodeLinks[ia].m_links.push_back(ib);

				if (nodeLinks[ib].m_links.findLinearSearch(ia) == nodeLinks[ib].m_links.size())
					nodeLinks[ib].m_links.push_back(ia);
			}
			for (int ii = 0; ii < nodeLinks.size(); ii++)
			{
				int i = ii;

				for (int jj = 0; jj < nodeLinks[ii].m_links.size(); jj++)
				{
					int k = nodeLinks[ii].m_links[jj];
					for (int kk = 0; kk < nodeLinks[k].m_links.size(); kk++)
					{
						int j = nodeLinks[k].m_links[kk];
						if (i != j)
						{
							const unsigned sum = adj[IDX(i, k)] + adj[IDX(k, j)];
							btAssert(sum == 2);
							if (adj[IDX(i, j)] > sum)
							{
								adj[IDX(i, j)] = adj[IDX(j, i)] = sum;
							}
						}
					}
				}
			}
		}
		else
		{
			///generic Floyd's algorithm
			for (int k = 0; k < n; ++k)
			{
				for (j = 0; j < n; ++j)
				{
					for (i = j + 1; i < n; ++i)
					{
						const unsigned sum = adj[IDX(i, k)] + adj[IDX(k, j)];
						if (adj[IDX(i, j)] > sum)
						{
							adj[IDX(i, j)] = adj[IDX(j, i)] = sum;
						}
					}
				}
			}
		}

		/* Build links	*/
		int nlinks = 0;
		for (j = 0; j < n; ++j)
		{
			for (i = j + 1; i < n; ++i)
			{
				if (adj[IDX(i, j)] == (unsigned)distance)
				{
					appendLink(i, j, mat);
					m_links[m_links.size() - 1].m_bbending = 1;
					++nlinks;
				}
			}
		}
		delete[] adj;
		return (nlinks);
	}
	return (0);
}

//
void btSoftBody::randomizeConstraints()
{
	unsigned long seed = 243703;
#define NEXTRAND (seed = (1664525L * seed + 1013904223L) & 0xffffffff)
	int i, ni;

	for (i = 0, ni = m_links.size(); i < ni; ++i)
	{
		btSwap(m_links[i], m_links[NEXTRAND % ni]);
	}
	for (i = 0, ni = m_faces.size(); i < ni; ++i)
	{
		btSwap(m_faces[i], m_faces[NEXTRAND % ni]);
	}
#undef NEXTRAND
}

void btSoftBody::updateState(const btAlignedObjectArray<btVector3>& q, const btAlignedObjectArray<btVector3>& v)
{
	int node_count = m_nodes.size();
	btAssert(node_count == q.size());
	btAssert(node_count == v.size());
	for (int i = 0; i < node_count; i++)
	{
		Node& n = m_nodes[i];
		n.m_x = q[i];
		n.m_q = q[i];
		n.m_v = v[i];
		n.m_vn = v[i];
	}
}

//
void btSoftBody::releaseCluster(int index)
{
	Cluster* c = m_clusters[index];
	if (c->m_leaf) m_cdbvt.remove(c->m_leaf);
	c->~Cluster();
	btAlignedFree(c);
	m_clusters.remove(c);
}

//
void btSoftBody::releaseClusters()
{
	while (m_clusters.size() > 0) releaseCluster(0);
}

//
int btSoftBody::generateClusters(int k, int maxiterations)
{
	int i;
	releaseClusters();
	m_clusters.resize(btMin(k, m_nodes.size()));
	for (i = 0; i < m_clusters.size(); ++i)
	{
		m_clusters[i] = new (btAlignedAlloc(sizeof(Cluster), 16)) Cluster();
		m_clusters[i]->m_collide = true;
	}
	k = m_clusters.size();
	if (k > 0)
	{
		/* Initialize		*/
		btAlignedObjectArray<btVector3> centers;
		btVector3 cog(0, 0, 0);
		int i;
		for (i = 0; i < m_nodes.size(); ++i)
		{
			cog += m_nodes[i].m_x;
			m_clusters[(i * 29873) % m_clusters.size()]->m_nodes.push_back(&m_nodes[i]);
		}
		cog /= (btScalar)m_nodes.size();
		centers.resize(k, cog);
		/* Iterate			*/
		const btScalar slope = 16;
		bool changed;
		int iterations = 0;
		do
		{
			const btScalar w = 2 - btMin<btScalar>(1, iterations / slope);
			changed = false;
			iterations++;
			int i;

			for (i = 0; i < k; ++i)
			{
				btVector3 c(0, 0, 0);
				for (int j = 0; j < m_clusters[i]->m_nodes.size(); ++j)
				{
					c += m_clusters[i]->m_nodes[j]->m_x;
				}
				if (m_clusters[i]->m_nodes.size())
				{
					c /= (btScalar)m_clusters[i]->m_nodes.size();
					c = centers[i] + (c - centers[i]) * w;
					changed |= ((c - centers[i]).length2() > SIMD_EPSILON);
					centers[i] = c;
					m_clusters[i]->m_nodes.resize(0);
				}
			}
			for (i = 0; i < m_nodes.size(); ++i)
			{
				const btVector3 nx = m_nodes[i].m_x;
				int kbest = 0;
				btScalar kdist = ClusterMetric(centers[0], nx);
				for (int j = 1; j < k; ++j)
				{
					const btScalar d = ClusterMetric(centers[j], nx);
					if (d < kdist)
					{
						kbest = j;
						kdist = d;
					}
				}
				m_clusters[kbest]->m_nodes.push_back(&m_nodes[i]);
			}
		} while (changed && (iterations < maxiterations));
		/* Merge		*/
		btAlignedObjectArray<int> cids;
		cids.resize(m_nodes.size(), -1);
		for (i = 0; i < m_clusters.size(); ++i)
		{
			for (int j = 0; j < m_clusters[i]->m_nodes.size(); ++j)
			{
				cids[int(m_clusters[i]->m_nodes[j] - &m_nodes[0])] = i;
			}
		}
		for (i = 0; i < m_faces.size(); ++i)
		{
			const int idx[] = {int(m_faces[i].m_n[0] - &m_nodes[0]),
							   int(m_faces[i].m_n[1] - &m_nodes[0]),
							   int(m_faces[i].m_n[2] - &m_nodes[0])};
			for (int j = 0; j < 3; ++j)
			{
				const int cid = cids[idx[j]];
				for (int q = 1; q < 3; ++q)
				{
					const int kid = idx[(j + q) % 3];
					if (cids[kid] != cid)
					{
						if (m_clusters[cid]->m_nodes.findLinearSearch(&m_nodes[kid]) == m_clusters[cid]->m_nodes.size())
						{
							m_clusters[cid]->m_nodes.push_back(&m_nodes[kid]);
						}
					}
				}
			}
		}
		/* Master		*/
		if (m_clusters.size() > 1)
		{
			Cluster* pmaster = new (btAlignedAlloc(sizeof(Cluster), 16)) Cluster();
			pmaster->m_collide = false;
			pmaster->m_nodes.reserve(m_nodes.size());
			for (int i = 0; i < m_nodes.size(); ++i) pmaster->m_nodes.push_back(&m_nodes[i]);
			m_clusters.push_back(pmaster);
			btSwap(m_clusters[0], m_clusters[m_clusters.size() - 1]);
		}
		/* Terminate	*/
		for (i = 0; i < m_clusters.size(); ++i)
		{
			if (m_clusters[i]->m_nodes.size() == 0)
			{
				releaseCluster(i--);
			}
		}
	}
	else
	{
		//create a cluster for each tetrahedron (if tetrahedra exist) or each face
		if (m_tetras.size())
		{
			m_clusters.resize(m_tetras.size());
			for (i = 0; i < m_clusters.size(); ++i)
			{
				m_clusters[i] = new (btAlignedAlloc(sizeof(Cluster), 16)) Cluster();
				m_clusters[i]->m_collide = true;
			}
			for (i = 0; i < m_tetras.size(); i++)
			{
				for (int j = 0; j < 4; j++)
				{
					m_clusters[i]->m_nodes.push_back(m_tetras[i].m_n[j]);
				}
			}
		}
		else
		{
			m_clusters.resize(m_faces.size());
			for (i = 0; i < m_clusters.size(); ++i)
			{
				m_clusters[i] = new (btAlignedAlloc(sizeof(Cluster), 16)) Cluster();
				m_clusters[i]->m_collide = true;
			}

			for (i = 0; i < m_faces.size(); ++i)
			{
				for (int j = 0; j < 3; ++j)
				{
					m_clusters[i]->m_nodes.push_back(m_faces[i].m_n[j]);
				}
			}
		}
	}

	if (m_clusters.size())
	{
		initializeClusters();
		updateClusters();

		//for self-collision
		m_clusterConnectivity.resize(m_clusters.size() * m_clusters.size());
		{
			for (int c0 = 0; c0 < m_clusters.size(); c0++)
			{
				m_clusters[c0]->m_clusterIndex = c0;
				for (int c1 = 0; c1 < m_clusters.size(); c1++)
				{
					bool connected = false;
					Cluster* cla = m_clusters[c0];
					Cluster* clb = m_clusters[c1];
					for (int i = 0; !connected && i < cla->m_nodes.size(); i++)
					{
						for (int j = 0; j < clb->m_nodes.size(); j++)
						{
							if (cla->m_nodes[i] == clb->m_nodes[j])
							{
								connected = true;
								break;
							}
						}
					}
					m_clusterConnectivity[c0 + c1 * m_clusters.size()] = connected;
				}
			}
		}
	}

	return (m_clusters.size());
}

//
void btSoftBody::refine(ImplicitFn* ifn, btScalar accurary, bool cut)
{
	const Node* nbase = &m_nodes[0];
	int ncount = m_nodes.size();
	btSymMatrix<int> edges(ncount, -2);
	int newnodes = 0;
	int i, j, k, ni;

	/* Filter out		*/
	for (i = 0; i < m_links.size(); ++i)
	{
		Link& l = m_links[i];
		if (l.m_bbending)
		{
			if (!SameSign(ifn->Eval(l.m_n[0]->m_x), ifn->Eval(l.m_n[1]->m_x)))
			{
				btSwap(m_links[i], m_links[m_links.size() - 1]);
				m_links.pop_back();
				--i;
			}
		}
	}
	/* Fill edges		*/
	for (i = 0; i < m_links.size(); ++i)
	{
		Link& l = m_links[i];
		edges(int(l.m_n[0] - nbase), int(l.m_n[1] - nbase)) = -1;
	}
	for (i = 0; i < m_faces.size(); ++i)
	{
		Face& f = m_faces[i];
		edges(int(f.m_n[0] - nbase), int(f.m_n[1] - nbase)) = -1;
		edges(int(f.m_n[1] - nbase), int(f.m_n[2] - nbase)) = -1;
		edges(int(f.m_n[2] - nbase), int(f.m_n[0] - nbase)) = -1;
	}
	/* Intersect		*/
	for (i = 0; i < ncount; ++i)
	{
		for (j = i + 1; j < ncount; ++j)
		{
			if (edges(i, j) == -1)
			{
				Node& a = m_nodes[i];
				Node& b = m_nodes[j];
				const btScalar t = ImplicitSolve(ifn, a.m_x, b.m_x, accurary);
				if (t > 0)
				{
					const btVector3 x = Lerp(a.m_x, b.m_x, t);
					const btVector3 v = Lerp(a.m_v, b.m_v, t);
					btScalar m = 0;
					if (a.m_im > 0)
					{
						if (b.m_im > 0)
						{
							const btScalar ma = 1 / a.m_im;
							const btScalar mb = 1 / b.m_im;
							const btScalar mc = Lerp(ma, mb, t);
							const btScalar f = (ma + mb) / (ma + mb + mc);
							a.m_im = 1 / (ma * f);
							b.m_im = 1 / (mb * f);
							m = mc * f;
						}
						else
						{
							a.m_im /= 0.5f;
							m = 1 / a.m_im;
						}
					}
					else
					{
						if (b.m_im > 0)
						{
							b.m_im /= 0.5f;
							m = 1 / b.m_im;
						}
						else
							m = 0;
					}
					appendNode(x, m);
					edges(i, j) = m_nodes.size() - 1;
					m_nodes[edges(i, j)].m_v = v;
					++newnodes;
				}
			}
		}
	}
	nbase = &m_nodes[0];
	/* Refine links		*/
	for (i = 0, ni = m_links.size(); i < ni; ++i)
	{
		Link& feat = m_links[i];
		const int idx[] = {int(feat.m_n[0] - nbase),
						   int(feat.m_n[1] - nbase)};
		if ((idx[0] < ncount) && (idx[1] < ncount))
		{
			const int ni = edges(idx[0], idx[1]);
			if (ni > 0)
			{
				appendLink(i);
				Link* pft[] = {&m_links[i],
							   &m_links[m_links.size() - 1]};
				pft[0]->m_n[0] = &m_nodes[idx[0]];
				pft[0]->m_n[1] = &m_nodes[ni];
				pft[1]->m_n[0] = &m_nodes[ni];
				pft[1]->m_n[1] = &m_nodes[idx[1]];
			}
		}
	}
	/* Refine faces		*/
	for (i = 0; i < m_faces.size(); ++i)
	{
		const Face& feat = m_faces[i];
		const int idx[] = {int(feat.m_n[0] - nbase),
						   int(feat.m_n[1] - nbase),
						   int(feat.m_n[2] - nbase)};
		for (j = 2, k = 0; k < 3; j = k++)
		{
			if ((idx[j] < ncount) && (idx[k] < ncount))
			{
				const int ni = edges(idx[j], idx[k]);
				if (ni > 0)
				{
					appendFace(i);
					const int l = (k + 1) % 3;
					Face* pft[] = {&m_faces[i],
								   &m_faces[m_faces.size() - 1]};
					pft[0]->m_n[0] = &m_nodes[idx[l]];
					pft[0]->m_n[1] = &m_nodes[idx[j]];
					pft[0]->m_n[2] = &m_nodes[ni];
					pft[1]->m_n[0] = &m_nodes[ni];
					pft[1]->m_n[1] = &m_nodes[idx[k]];
					pft[1]->m_n[2] = &m_nodes[idx[l]];
					appendLink(ni, idx[l], pft[0]->m_material);
					--i;
					break;
				}
			}
		}
	}
	/* Cut				*/
	if (cut)
	{
		btAlignedObjectArray<int> cnodes;
		const int pcount = ncount;
		int i;
		ncount = m_nodes.size();
		cnodes.resize(ncount, 0);
		/* Nodes		*/
		for (i = 0; i < ncount; ++i)
		{
			const btVector3 x = m_nodes[i].m_x;
			if ((i >= pcount) || (btFabs(ifn->Eval(x)) < accurary))
			{
				const btVector3 v = m_nodes[i].m_v;
				btScalar m = getMass(i);
				if (m > 0)
				{
					m *= 0.5f;
					m_nodes[i].m_im /= 0.5f;
				}
				appendNode(x, m);
				cnodes[i] = m_nodes.size() - 1;
				m_nodes[cnodes[i]].m_v = v;
			}
		}
		nbase = &m_nodes[0];
		/* Links		*/
		for (i = 0, ni = m_links.size(); i < ni; ++i)
		{
			const int id[] = {int(m_links[i].m_n[0] - nbase),
							  int(m_links[i].m_n[1] - nbase)};
			int todetach = 0;
			if (cnodes[id[0]] && cnodes[id[1]])
			{
				appendLink(i);
				todetach = m_links.size() - 1;
			}
			else
			{
				if (((ifn->Eval(m_nodes[id[0]].m_x) < accurary) &&
					 (ifn->Eval(m_nodes[id[1]].m_x) < accurary)))
					todetach = i;
			}
			if (todetach)
			{
				Link& l = m_links[todetach];
				for (int j = 0; j < 2; ++j)
				{
					int cn = cnodes[int(l.m_n[j] - nbase)];
					if (cn) l.m_n[j] = &m_nodes[cn];
				}
			}
		}
		/* Faces		*/
		for (i = 0, ni = m_faces.size(); i < ni; ++i)
		{
			Node** n = m_faces[i].m_n;
			if ((ifn->Eval(n[0]->m_x) < accurary) &&
				(ifn->Eval(n[1]->m_x) < accurary) &&
				(ifn->Eval(n[2]->m_x) < accurary))
			{
				for (int j = 0; j < 3; ++j)
				{
					int cn = cnodes[int(n[j] - nbase)];
					if (cn) n[j] = &m_nodes[cn];
				}
			}
		}
		/* Clean orphans	*/
		int nnodes = m_nodes.size();
		btAlignedObjectArray<int> ranks;
		btAlignedObjectArray<int> todelete;
		ranks.resize(nnodes, 0);
		for (i = 0, ni = m_links.size(); i < ni; ++i)
		{
			for (int j = 0; j < 2; ++j) ranks[int(m_links[i].m_n[j] - nbase)]++;
		}
		for (i = 0, ni = m_faces.size(); i < ni; ++i)
		{
			for (int j = 0; j < 3; ++j) ranks[int(m_faces[i].m_n[j] - nbase)]++;
		}
		for (i = 0; i < m_links.size(); ++i)
		{
			const int id[] = {int(m_links[i].m_n[0] - nbase),
							  int(m_links[i].m_n[1] - nbase)};
			const bool sg[] = {ranks[id[0]] == 1,
							   ranks[id[1]] == 1};
			if (sg[0] || sg[1])
			{
				--ranks[id[0]];
				--ranks[id[1]];
				btSwap(m_links[i], m_links[m_links.size() - 1]);
				m_links.pop_back();
				--i;
			}
		}
#if 0	
		for(i=nnodes-1;i>=0;--i)
		{
			if(!ranks[i]) todelete.push_back(i);
		}	
		if(todelete.size())
		{		
			btAlignedObjectArray<int>&	map=ranks;
			for(int i=0;i<nnodes;++i) map[i]=i;
			PointersToIndices(this);
			for(int i=0,ni=todelete.size();i<ni;++i)
			{
				int		j=todelete[i];
				int&	a=map[j];
				int&	b=map[--nnodes];
				m_ndbvt.remove(m_nodes[a].m_leaf);m_nodes[a].m_leaf=0;
				btSwap(m_nodes[a],m_nodes[b]);
				j=a;a=b;b=j;			
			}
			IndicesToPointers(this,&map[0]);
			m_nodes.resize(nnodes);
		}
#endif
	}
	m_bUpdateRtCst = true;
}

//
bool btSoftBody::cutLink(const Node* node0, const Node* node1, btScalar position)
{
	return (cutLink(int(node0 - &m_nodes[0]), int(node1 - &m_nodes[0]), position));
}

//
bool btSoftBody::cutLink(int node0, int node1, btScalar position)
{
	bool done = false;
	int i, ni;
	//	const btVector3	d=m_nodes[node0].m_x-m_nodes[node1].m_x;
	const btVector3 x = Lerp(m_nodes[node0].m_x, m_nodes[node1].m_x, position);
	const btVector3 v = Lerp(m_nodes[node0].m_v, m_nodes[node1].m_v, position);
	const btScalar m = 1;
	appendNode(x, m);
	appendNode(x, m);
	Node* pa = &m_nodes[node0];
	Node* pb = &m_nodes[node1];
	Node* pn[2] = {&m_nodes[m_nodes.size() - 2],
				   &m_nodes[m_nodes.size() - 1]};
	pn[0]->m_v = v;
	pn[1]->m_v = v;
	for (i = 0, ni = m_links.size(); i < ni; ++i)
	{
		const int mtch = MatchEdge(m_links[i].m_n[0], m_links[i].m_n[1], pa, pb);
		if (mtch != -1)
		{
			appendLink(i);
			Link* pft[] = {&m_links[i], &m_links[m_links.size() - 1]};
			pft[0]->m_n[1] = pn[mtch];
			pft[1]->m_n[0] = pn[1 - mtch];
			done = true;
		}
	}
	for (i = 0, ni = m_faces.size(); i < ni; ++i)
	{
		for (int k = 2, l = 0; l < 3; k = l++)
		{
			const int mtch = MatchEdge(m_faces[i].m_n[k], m_faces[i].m_n[l], pa, pb);
			if (mtch != -1)
			{
				appendFace(i);
				Face* pft[] = {&m_faces[i], &m_faces[m_faces.size() - 1]};
				pft[0]->m_n[l] = pn[mtch];
				pft[1]->m_n[k] = pn[1 - mtch];
				appendLink(pn[0], pft[0]->m_n[(l + 1) % 3], pft[0]->m_material, true);
				appendLink(pn[1], pft[0]->m_n[(l + 1) % 3], pft[0]->m_material, true);
			}
		}
	}
	if (!done)
	{
		m_ndbvt.remove(pn[0]->m_leaf);
		m_ndbvt.remove(pn[1]->m_leaf);
		m_nodes.pop_back();
		m_nodes.pop_back();
	}
	return (done);
}

//
bool btSoftBody::rayTest(const btVector3& rayFrom,
						 const btVector3& rayTo,
						 sRayCast& results)
{
	if (m_faces.size() && m_fdbvt.empty())
		initializeFaceTree();

	results.body = this;
	results.fraction = 1.f;
	results.feature = eFeature::None;
	results.index = -1;

	return (rayTest(rayFrom, rayTo, results.fraction, results.feature, results.index, false) != 0);
}

bool btSoftBody::rayFaceTest(const btVector3& rayFrom,
							 const btVector3& rayTo,
							 sRayCast& results)
{
	if (m_faces.size() == 0)
		return false;
	else
	{
		if (m_fdbvt.empty())
			initializeFaceTree();
	}

	results.body = this;
	results.fraction = 1.f;
	results.index = -1;

	return (rayFaceTest(rayFrom, rayTo, results.fraction, results.index) != 0);
}

//
void btSoftBody::setSolver(eSolverPresets::_ preset)
{
	m_cfg.m_vsequence.clear();
	m_cfg.m_psequence.clear();
	m_cfg.m_dsequence.clear();
	switch (preset)
	{
		case eSolverPresets::Positions:
			m_cfg.m_psequence.push_back(ePSolver::Anchors);
			m_cfg.m_psequence.push_back(ePSolver::RContacts);
			m_cfg.m_psequence.push_back(ePSolver::SContacts);
			m_cfg.m_psequence.push_back(ePSolver::Linear);
			break;
		case eSolverPresets::Velocities:
			m_cfg.m_vsequence.push_back(eVSolver::Linear);

			m_cfg.m_psequence.push_back(ePSolver::Anchors);
			m_cfg.m_psequence.push_back(ePSolver::RContacts);
			m_cfg.m_psequence.push_back(ePSolver::SContacts);

			m_cfg.m_dsequence.push_back(ePSolver::Linear);
			break;
	}
}

void btSoftBody::predictMotion(btScalar dt)
{
	int i, ni;

	/* Update                */
	if (m_bUpdateRtCst)
	{
		m_bUpdateRtCst = false;
		updateConstants();
		m_fdbvt.clear();
		if (m_cfg.collisions & fCollision::VF_SS)
		{
			initializeFaceTree();
		}
	}

	/* Prepare                */
	m_sst.sdt = dt * m_cfg.timescale;
	m_sst.isdt = 1 / m_sst.sdt;
	m_sst.velmrg = m_sst.sdt * 3;
	m_sst.radmrg = getCollisionShape()->getMargin();
	m_sst.updmrg = m_sst.radmrg * (btScalar)0.25;
	/* Forces                */
	addVelocity(m_worldInfo->m_gravity * m_sst.sdt);
	applyForces();
	/* Integrate            */
	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		Node& n = m_nodes[i];
		n.m_q = n.m_x;
		btVector3 deltaV = n.m_f * n.m_im * m_sst.sdt;
		{
			btScalar maxDisplacement = m_worldInfo->m_maxDisplacement;
			btScalar clampDeltaV = maxDisplacement / m_sst.sdt;
			for (int c = 0; c < 3; c++)
			{
				if (deltaV[c] > clampDeltaV)
				{
					deltaV[c] = clampDeltaV;
				}
				if (deltaV[c] < -clampDeltaV)
				{
					deltaV[c] = -clampDeltaV;
				}
			}
		}
		n.m_v += deltaV;
		n.m_x += n.m_v * m_sst.sdt;
		n.m_f = btVector3(0, 0, 0);
	}
	/* Clusters                */
	updateClusters();
	/* Bounds                */
	updateBounds();
	/* Nodes                */
	ATTRIBUTE_ALIGNED16(btDbvtVolume)
	vol;
	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		Node& n = m_nodes[i];
		vol = btDbvtVolume::FromCR(n.m_x, m_sst.radmrg);
		m_ndbvt.update(n.m_leaf,
					   vol,
					   n.m_v * m_sst.velmrg,
					   m_sst.updmrg);
	}
	/* Faces                */
	if (!m_fdbvt.empty())
	{
		for (int i = 0; i < m_faces.size(); ++i)
		{
			Face& f = m_faces[i];
			const btVector3 v = (f.m_n[0]->m_v +
								 f.m_n[1]->m_v +
								 f.m_n[2]->m_v) /
								3;
			vol = VolumeOf(f, m_sst.radmrg);
			m_fdbvt.update(f.m_leaf,
						   vol,
						   v * m_sst.velmrg,
						   m_sst.updmrg);
		}
	}
	/* Pose                    */
	updatePose();
	/* Match                */
	if (m_pose.m_bframe && (m_cfg.kMT > 0))
	{
		const btMatrix3x3 posetrs = m_pose.m_rot;
		for (int i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			Node& n = m_nodes[i];
			if (n.m_im > 0)
			{
				const btVector3 x = posetrs * m_pose.m_pos[i] + m_pose.m_com;
				n.m_x = Lerp(n.m_x, x, m_cfg.kMT);
			}
		}
	}
	/* Clear contacts        */
	m_rcontacts.resize(0);
	m_scontacts.resize(0);
	/* Optimize dbvt's        */
	m_ndbvt.optimizeIncremental(1);
	m_fdbvt.optimizeIncremental(1);
	m_cdbvt.optimizeIncremental(1);
}

//
void btSoftBody::solveConstraints()
{
	/* Apply clusters		*/
	applyClusters(false);
	/* Prepare links		*/

	int i, ni;

	for (i = 0, ni = m_links.size(); i < ni; ++i)
	{
		Link& l = m_links[i];
		l.m_c3 = l.m_n[1]->m_q - l.m_n[0]->m_q;
		l.m_c2 = 1 / (l.m_c3.length2() * l.m_c0);
	}
	/* Prepare anchors		*/
	for (i = 0, ni = m_anchors.size(); i < ni; ++i)
	{
		Anchor& a = m_anchors[i];
		const btVector3 ra = a.m_body->getWorldTransform().getBasis() * a.m_local;
		a.m_c0 = ImpulseMatrix(m_sst.sdt,
							   a.m_node->m_im,
							   a.m_body->getInvMass(),
							   a.m_body->getInvInertiaTensorWorld(),
							   ra);
		a.m_c1 = ra;
		a.m_c2 = m_sst.sdt * a.m_node->m_im;
		a.m_body->activate();
	}
	/* Solve velocities		*/
	if (m_cfg.viterations > 0)
	{
		/* Solve			*/
		for (int isolve = 0; isolve < m_cfg.viterations; ++isolve)
		{
			for (int iseq = 0; iseq < m_cfg.m_vsequence.size(); ++iseq)
			{
				getSolver(m_cfg.m_vsequence[iseq])(this, 1);
			}
		}
		/* Update			*/
		for (i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			Node& n = m_nodes[i];
			n.m_x = n.m_q + n.m_v * m_sst.sdt;
		}
	}
	/* Solve positions		*/
	if (m_cfg.piterations > 0)
	{
		for (int isolve = 0; isolve < m_cfg.piterations; ++isolve)
		{
			const btScalar ti = isolve / (btScalar)m_cfg.piterations;
			for (int iseq = 0; iseq < m_cfg.m_psequence.size(); ++iseq)
			{
				getSolver(m_cfg.m_psequence[iseq])(this, 1, ti);
			}
		}
		const btScalar vc = m_sst.isdt * (1 - m_cfg.kDP);
		for (i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			Node& n = m_nodes[i];
			n.m_v = (n.m_x - n.m_q) * vc;
			n.m_f = btVector3(0, 0, 0);
		}
	}
	/* Solve drift			*/
	if (m_cfg.diterations > 0)
	{
		const btScalar vcf = m_cfg.kVCF * m_sst.isdt;
		for (i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			Node& n = m_nodes[i];
			n.m_q = n.m_x;
		}
		for (int idrift = 0; idrift < m_cfg.diterations; ++idrift)
		{
			for (int iseq = 0; iseq < m_cfg.m_dsequence.size(); ++iseq)
			{
				getSolver(m_cfg.m_dsequence[iseq])(this, 1, 0);
			}
		}
		for (int i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			Node& n = m_nodes[i];
			n.m_v += (n.m_x - n.m_q) * vcf;
		}
	}
	/* Apply clusters		*/
	dampClusters();
	applyClusters(true);
}

//
void btSoftBody::staticSolve(int iterations)
{
	for (int isolve = 0; isolve < iterations; ++isolve)
	{
		for (int iseq = 0; iseq < m_cfg.m_psequence.size(); ++iseq)
		{
			getSolver(m_cfg.m_psequence[iseq])(this, 1, 0);
		}
	}
}

//
void btSoftBody::solveCommonConstraints(btSoftBody** /*bodies*/, int /*count*/, int /*iterations*/)
{
	/// placeholder
}

//
void btSoftBody::solveClusters(const btAlignedObjectArray<btSoftBody*>& bodies)
{
	const int nb = bodies.size();
	int iterations = 0;
	int i;

	for (i = 0; i < nb; ++i)
	{
		iterations = btMax(iterations, bodies[i]->m_cfg.citerations);
	}
	for (i = 0; i < nb; ++i)
	{
		bodies[i]->prepareClusters(iterations);
	}
	for (i = 0; i < iterations; ++i)
	{
		const btScalar sor = 1;
		for (int j = 0; j < nb; ++j)
		{
			bodies[j]->solveClusters(sor);
		}
	}
	for (i = 0; i < nb; ++i)
	{
		bodies[i]->cleanupClusters();
	}
}

//
void btSoftBody::integrateMotion()
{
	/* Update			*/
	updateNormals();
}

//
btSoftBody::RayFromToCaster::RayFromToCaster(const btVector3& rayFrom, const btVector3& rayTo, btScalar mxt)
{
	m_rayFrom = rayFrom;
	m_rayNormalizedDirection = (rayTo - rayFrom);
	m_rayTo = rayTo;
	m_mint = mxt;
	m_face = 0;
	m_tests = 0;
}

//
void btSoftBody::RayFromToCaster::Process(const btDbvtNode* leaf)
{
	btSoftBody::Face& f = *(btSoftBody::Face*)leaf->data;
	const btScalar t = rayFromToTriangle(m_rayFrom, m_rayTo, m_rayNormalizedDirection,
										 f.m_n[0]->m_x,
										 f.m_n[1]->m_x,
										 f.m_n[2]->m_x,
										 m_mint);
	if ((t > 0) && (t < m_mint))
	{
		m_mint = t;
		m_face = &f;
	}
	++m_tests;
}

//
btScalar btSoftBody::RayFromToCaster::rayFromToTriangle(const btVector3& rayFrom,
														const btVector3& rayTo,
														const btVector3& rayNormalizedDirection,
														const btVector3& a,
														const btVector3& b,
														const btVector3& c,
														btScalar maxt)
{
	static const btScalar ceps = -SIMD_EPSILON * 10;
	static const btScalar teps = SIMD_EPSILON * 10;

	const btVector3 n = btCross(b - a, c - a);
	const btScalar d = btDot(a, n);
	const btScalar den = btDot(rayNormalizedDirection, n);
	if (!btFuzzyZero(den))
	{
		const btScalar num = btDot(rayFrom, n) - d;
		const btScalar t = -num / den;
		if ((t > teps) && (t < maxt))
		{
			const btVector3 hit = rayFrom + rayNormalizedDirection * t;
			if ((btDot(n, btCross(a - hit, b - hit)) > ceps) &&
				(btDot(n, btCross(b - hit, c - hit)) > ceps) &&
				(btDot(n, btCross(c - hit, a - hit)) > ceps))
			{
				return (t);
			}
		}
	}
	return (-1);
}

//
void btSoftBody::pointersToIndices()
{
#define PTR2IDX(_p_, _b_) reinterpret_cast<btSoftBody::Node*>((_p_) - (_b_))
	btSoftBody::Node* base = m_nodes.size() ? &m_nodes[0] : 0;
	int i, ni;

	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		if (m_nodes[i].m_leaf)
		{
			m_nodes[i].m_leaf->data = *(void**)&i;
		}
	}
	for (i = 0, ni = m_links.size(); i < ni; ++i)
	{
		m_links[i].m_n[0] = PTR2IDX(m_links[i].m_n[0], base);
		m_links[i].m_n[1] = PTR2IDX(m_links[i].m_n[1], base);
	}
	for (i = 0, ni = m_faces.size(); i < ni; ++i)
	{
		m_faces[i].m_n[0] = PTR2IDX(m_faces[i].m_n[0], base);
		m_faces[i].m_n[1] = PTR2IDX(m_faces[i].m_n[1], base);
		m_faces[i].m_n[2] = PTR2IDX(m_faces[i].m_n[2], base);
		if (m_faces[i].m_leaf)
		{
			m_faces[i].m_leaf->data = *(void**)&i;
		}
	}
	for (i = 0, ni = m_anchors.size(); i < ni; ++i)
	{
		m_anchors[i].m_node = PTR2IDX(m_anchors[i].m_node, base);
	}
	for (i = 0, ni = m_notes.size(); i < ni; ++i)
	{
		for (int j = 0; j < m_notes[i].m_rank; ++j)
		{
			m_notes[i].m_nodes[j] = PTR2IDX(m_notes[i].m_nodes[j], base);
		}
	}
#undef PTR2IDX
}

//
void btSoftBody::indicesToPointers(const int* map)
{
#define IDX2PTR(_p_, _b_) map ? (&(_b_)[map[(((char*)_p_) - (char*)0)]]) : (&(_b_)[(((char*)_p_) - (char*)0)])
	btSoftBody::Node* base = m_nodes.size() ? &m_nodes[0] : 0;
	int i, ni;

	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		if (m_nodes[i].m_leaf)
		{
			m_nodes[i].m_leaf->data = &m_nodes[i];
		}
	}
	for (i = 0, ni = m_links.size(); i < ni; ++i)
	{
		m_links[i].m_n[0] = IDX2PTR(m_links[i].m_n[0], base);
		m_links[i].m_n[1] = IDX2PTR(m_links[i].m_n[1], base);
	}
	for (i = 0, ni = m_faces.size(); i < ni; ++i)
	{
		m_faces[i].m_n[0] = IDX2PTR(m_faces[i].m_n[0], base);
		m_faces[i].m_n[1] = IDX2PTR(m_faces[i].m_n[1], base);
		m_faces[i].m_n[2] = IDX2PTR(m_faces[i].m_n[2], base);
		if (m_faces[i].m_leaf)
		{
			m_faces[i].m_leaf->data = &m_faces[i];
		}
	}
	for (i = 0, ni = m_anchors.size(); i < ni; ++i)
	{
		m_anchors[i].m_node = IDX2PTR(m_anchors[i].m_node, base);
	}
	for (i = 0, ni = m_notes.size(); i < ni; ++i)
	{
		for (int j = 0; j < m_notes[i].m_rank; ++j)
		{
			m_notes[i].m_nodes[j] = IDX2PTR(m_notes[i].m_nodes[j], base);
		}
	}
#undef IDX2PTR
}

//
int btSoftBody::rayTest(const btVector3& rayFrom, const btVector3& rayTo,
						btScalar& mint, eFeature::_& feature, int& index, bool bcountonly) const
{
	int cnt = 0;
	btVector3 dir = rayTo - rayFrom;

	if (bcountonly || m_fdbvt.empty())
	{ /* Full search	*/

		for (int i = 0, ni = m_faces.size(); i < ni; ++i)
		{
			const btSoftBody::Face& f = m_faces[i];

			const btScalar t = RayFromToCaster::rayFromToTriangle(rayFrom, rayTo, dir,
																  f.m_n[0]->m_x,
																  f.m_n[1]->m_x,
																  f.m_n[2]->m_x,
																  mint);
			if (t > 0)
			{
				++cnt;
				if (!bcountonly)
				{
					feature = btSoftBody::eFeature::Face;
					index = i;
					mint = t;
				}
			}
		}
	}
	else
	{ /* Use dbvt	*/
		RayFromToCaster collider(rayFrom, rayTo, mint);

		btDbvt::rayTest(m_fdbvt.m_root, rayFrom, rayTo, collider);
		if (collider.m_face)
		{
			mint = collider.m_mint;
			feature = btSoftBody::eFeature::Face;
			index = (int)(collider.m_face - &m_faces[0]);
			cnt = 1;
		}
	}

	for (int i = 0; i < m_tetras.size(); i++)
	{
		const btSoftBody::Tetra& tet = m_tetras[i];
		int tetfaces[4][3] = {{0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}};
		for (int f = 0; f < 4; f++)
		{
			int index0 = tetfaces[f][0];
			int index1 = tetfaces[f][1];
			int index2 = tetfaces[f][2];
			btVector3 v0 = tet.m_n[index0]->m_x;
			btVector3 v1 = tet.m_n[index1]->m_x;
			btVector3 v2 = tet.m_n[index2]->m_x;

			const btScalar t = RayFromToCaster::rayFromToTriangle(rayFrom, rayTo, dir,
																  v0, v1, v2,
																  mint);
			if (t > 0)
			{
				++cnt;
				if (!bcountonly)
				{
					feature = btSoftBody::eFeature::Tetra;
					index = i;
					mint = t;
				}
			}
		}
	}
	return (cnt);
}

int btSoftBody::rayFaceTest(const btVector3& rayFrom, const btVector3& rayTo,
							btScalar& mint, int& index) const
{
	int cnt = 0;
	{ /* Use dbvt	*/
		RayFromToCaster collider(rayFrom, rayTo, mint);

		btDbvt::rayTest(m_fdbvt.m_root, rayFrom, rayTo, collider);
		if (collider.m_face)
		{
			mint = collider.m_mint;
			index = (int)(collider.m_face - &m_faces[0]);
			cnt = 1;
		}
	}
	return (cnt);
}

//
static inline btDbvntNode* copyToDbvnt(const btDbvtNode* n)
{
	if (n == 0)
		return 0;
	btDbvntNode* root = new btDbvntNode(n);
	if (n->isinternal())
	{
		btDbvntNode* c0 = copyToDbvnt(n->childs[0]);
		root->childs[0] = c0;
		btDbvntNode* c1 = copyToDbvnt(n->childs[1]);
		root->childs[1] = c1;
	}
	return root;
}

static inline void calculateNormalCone(btDbvntNode* root)
{
	if (!root)
		return;
	if (root->isleaf())
	{
		const btSoftBody::Face* face = (btSoftBody::Face*)root->data;
		root->normal = face->m_normal;
		root->angle = 0;
	}
	else
	{
		btVector3 n0(0, 0, 0), n1(0, 0, 0);
		btScalar a0 = 0, a1 = 0;
		if (root->childs[0])
		{
			calculateNormalCone(root->childs[0]);
			n0 = root->childs[0]->normal;
			a0 = root->childs[0]->angle;
		}
		if (root->childs[1])
		{
			calculateNormalCone(root->childs[1]);
			n1 = root->childs[1]->normal;
			a1 = root->childs[1]->angle;
		}
		root->normal = (n0 + n1).safeNormalize();
		root->angle = btMax(a0, a1) + btAngle(n0, n1) * 0.5;
	}
}

void btSoftBody::initializeFaceTree()
{
	BT_PROFILE("btSoftBody::initializeFaceTree");
	m_fdbvt.clear();
	// create leaf nodes;
	btAlignedObjectArray<btDbvtNode*> leafNodes;
	leafNodes.resize(m_faces.size());
	for (int i = 0; i < m_faces.size(); ++i)
	{
		Face& f = m_faces[i];
		ATTRIBUTE_ALIGNED16(btDbvtVolume)
		vol = VolumeOf(f, 0);
		btDbvtNode* node = new (btAlignedAlloc(sizeof(btDbvtNode), 16)) btDbvtNode();
		node->parent = NULL;
		node->data = &f;
		node->childs[1] = 0;
		node->volume = vol;
		leafNodes[i] = node;
		f.m_leaf = node;
	}
	btAlignedObjectArray<btAlignedObjectArray<int> > adj;
	adj.resize(m_faces.size());
	// construct the adjacency list for triangles
	for (int i = 0; i < adj.size(); ++i)
	{
		for (int j = i + 1; j < adj.size(); ++j)
		{
			int dup = 0;
			for (int k = 0; k < 3; ++k)
			{
				for (int l = 0; l < 3; ++l)
				{
					if (m_faces[i].m_n[k] == m_faces[j].m_n[l])
					{
						++dup;
						break;
					}
				}
				if (dup == 2)
				{
					adj[i].push_back(j);
					adj[j].push_back(i);
				}
			}
		}
	}
	m_fdbvt.m_root = buildTreeBottomUp(leafNodes, adj);
	if (m_fdbvnt)
		delete m_fdbvnt;
	m_fdbvnt = copyToDbvnt(m_fdbvt.m_root);
	updateFaceTree(false, false);
	rebuildNodeTree();
}

//
void btSoftBody::rebuildNodeTree()
{
	m_ndbvt.clear();
	btAlignedObjectArray<btDbvtNode*> leafNodes;
	leafNodes.resize(m_nodes.size());
	for (int i = 0; i < m_nodes.size(); ++i)
	{
		Node& n = m_nodes[i];
		ATTRIBUTE_ALIGNED16(btDbvtVolume)
		vol = btDbvtVolume::FromCR(n.m_x, 0);
		btDbvtNode* node = new (btAlignedAlloc(sizeof(btDbvtNode), 16)) btDbvtNode();
		node->parent = NULL;
		node->data = &n;
		node->childs[1] = 0;
		node->volume = vol;
		leafNodes[i] = node;
		n.m_leaf = node;
	}
	btAlignedObjectArray<btAlignedObjectArray<int> > adj;
	adj.resize(m_nodes.size());
	btAlignedObjectArray<int> old_id;
	old_id.resize(m_nodes.size());
	for (int i = 0; i < m_nodes.size(); ++i)
		old_id[i] = m_nodes[i].index;
	for (int i = 0; i < m_nodes.size(); ++i)
		m_nodes[i].index = i;
	for (int i = 0; i < m_links.size(); ++i)
	{
		Link& l = m_links[i];
		adj[l.m_n[0]->index].push_back(l.m_n[1]->index);
		adj[l.m_n[1]->index].push_back(l.m_n[0]->index);
	}
	m_ndbvt.m_root = buildTreeBottomUp(leafNodes, adj);
	for (int i = 0; i < m_nodes.size(); ++i)
		m_nodes[i].index = old_id[i];
}

//
btVector3 btSoftBody::evaluateCom() const
{
	btVector3 com(0, 0, 0);
	if (m_pose.m_bframe)
	{
		for (int i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			com += m_nodes[i].m_x * m_pose.m_wgh[i];
		}
	}
	return (com);
}

bool btSoftBody::checkContact(const btCollisionObjectWrapper* colObjWrap,
							  const btVector3& x,
							  btScalar margin,
							  btSoftBody::sCti& cti) const
{
	btVector3 nrm;
	const btCollisionShape* shp = colObjWrap->getCollisionShape();
	//    const btRigidBody *tmpRigid = btRigidBody::upcast(colObjWrap->getCollisionObject());
	//const btTransform &wtr = tmpRigid ? tmpRigid->getWorldTransform() : colObjWrap->getWorldTransform();
	const btTransform& wtr = colObjWrap->getWorldTransform();
	//todo: check which transform is needed here

	btScalar dst =
		m_worldInfo->m_sparsesdf.Evaluate(
			wtr.invXform(x),
			shp,
			nrm,
			margin);
	if (dst < 0)
	{
		cti.m_colObj = colObjWrap->getCollisionObject();
		cti.m_normal = wtr.getBasis() * nrm;
		cti.m_offset = -btDot(cti.m_normal, x - cti.m_normal * dst);
		return (true);
	}
	return (false);
}

//
bool btSoftBody::checkDeformableContact(const btCollisionObjectWrapper* colObjWrap,
										const btVector3& x,
										btScalar margin,
										btSoftBody::sCti& cti, bool predict) const
{
	btVector3 nrm;
	const btCollisionShape* shp = colObjWrap->getCollisionShape();
	const btCollisionObject* tmpCollisionObj = colObjWrap->getCollisionObject();
	// use the position x_{n+1}^* = x_n + dt * v_{n+1}^* where v_{n+1}^* = v_n + dtg for collision detect
	// but resolve contact at x_n
	btTransform wtr = (predict) ? (colObjWrap->m_preTransform != NULL ? tmpCollisionObj->getInterpolationWorldTransform() * (*colObjWrap->m_preTransform) : tmpCollisionObj->getInterpolationWorldTransform())
								: colObjWrap->getWorldTransform();
	btScalar dst =
		m_worldInfo->m_sparsesdf.Evaluate(
			wtr.invXform(x),
			shp,
			nrm,
			margin);

	if (!predict)
	{
		cti.m_colObj = colObjWrap->getCollisionObject();
		cti.m_normal = wtr.getBasis() * nrm;
		cti.m_offset = dst;
	}
	if (dst < 0)
		return true;
	return (false);
}

//
// Compute barycentric coordinates (u, v, w) for
// point p with respect to triangle (a, b, c)
static void getBarycentric(const btVector3& p, btVector3& a, btVector3& b, btVector3& c, btVector3& bary)
{
	btVector3 v0 = b - a, v1 = c - a, v2 = p - a;
	btScalar d00 = v0.dot(v0);
	btScalar d01 = v0.dot(v1);
	btScalar d11 = v1.dot(v1);
	btScalar d20 = v2.dot(v0);
	btScalar d21 = v2.dot(v1);
	btScalar denom = d00 * d11 - d01 * d01;
	bary.setY((d11 * d20 - d01 * d21) / denom);
	bary.setZ((d00 * d21 - d01 * d20) / denom);
	bary.setX(btScalar(1) - bary.getY() - bary.getZ());
}

//
bool btSoftBody::checkDeformableFaceContact(const btCollisionObjectWrapper* colObjWrap,
											Face& f,
											btVector3& contact_point,
											btVector3& bary,
											btScalar margin,
											btSoftBody::sCti& cti, bool predict) const
{
	btVector3 nrm;
	const btCollisionShape* shp = colObjWrap->getCollisionShape();
	const btCollisionObject* tmpCollisionObj = colObjWrap->getCollisionObject();
	// use the position x_{n+1}^* = x_n + dt * v_{n+1}^* where v_{n+1}^* = v_n + dtg for collision detect
	// but resolve contact at x_n
	btTransform wtr = (predict) ? (colObjWrap->m_preTransform != NULL ? tmpCollisionObj->getInterpolationWorldTransform() * (*colObjWrap->m_preTransform) : tmpCollisionObj->getInterpolationWorldTransform())
								: colObjWrap->getWorldTransform();
	btScalar dst;
	btGjkEpaSolver2::sResults results;

	//	#define USE_QUADRATURE 1

	// use collision quadrature point
#ifdef USE_QUADRATURE
	{
		dst = SIMD_INFINITY;
		btVector3 local_nrm;
		for (int q = 0; q < m_quads.size(); ++q)
		{
			btVector3 p;
			if (predict)
				p = BaryEval(f.m_n[0]->m_q, f.m_n[1]->m_q, f.m_n[2]->m_q, m_quads[q]);
			else
				p = BaryEval(f.m_n[0]->m_x, f.m_n[1]->m_x, f.m_n[2]->m_x, m_quads[q]);
			btScalar local_dst = m_worldInfo->m_sparsesdf.Evaluate(
				wtr.invXform(p),
				shp,
				local_nrm,
				margin);
			if (local_dst < dst)
			{
				if (local_dst < 0 && predict)
					return true;
				dst = local_dst;
				contact_point = p;
				bary = m_quads[q];
				nrm = local_nrm;
			}
			if (!predict)
			{
				cti.m_colObj = colObjWrap->getCollisionObject();
				cti.m_normal = wtr.getBasis() * nrm;
				cti.m_offset = dst;
			}
		}
		return (dst < 0);
	}
#endif

	// collision detection using x*
	btTransform triangle_transform;
	triangle_transform.setIdentity();
	triangle_transform.setOrigin(f.m_n[0]->m_q);
	btTriangleShape triangle(btVector3(0, 0, 0), f.m_n[1]->m_q - f.m_n[0]->m_q, f.m_n[2]->m_q - f.m_n[0]->m_q);
	btVector3 guess(0, 0, 0);
	const btConvexShape* csh = static_cast<const btConvexShape*>(shp);
	btGjkEpaSolver2::SignedDistance(&triangle, triangle_transform, csh, wtr, guess, results);
	dst = results.distance - 2.0 * csh->getMargin() - margin;  // margin padding so that the distance = the actual distance between face and rigid - margin of rigid - margin of deformable
	if (dst >= 0)
		return false;

	// Use consistent barycenter to recalculate distance.
	if (this->m_cacheBarycenter)
	{
		if (f.m_pcontact[3] != 0)
		{
			for (int i = 0; i < 3; ++i)
				bary[i] = f.m_pcontact[i];
			contact_point = BaryEval(f.m_n[0]->m_x, f.m_n[1]->m_x, f.m_n[2]->m_x, bary);
			const btConvexShape* csh = static_cast<const btConvexShape*>(shp);
			btGjkEpaSolver2::SignedDistance(contact_point, margin, csh, wtr, results);
			cti.m_colObj = colObjWrap->getCollisionObject();
			dst = results.distance;
			cti.m_normal = results.normal;
			cti.m_offset = dst;

			//point-convex CD
			wtr = colObjWrap->getWorldTransform();
			btTriangleShape triangle2(btVector3(0, 0, 0), f.m_n[1]->m_x - f.m_n[0]->m_x, f.m_n[2]->m_x - f.m_n[0]->m_x);
			triangle_transform.setOrigin(f.m_n[0]->m_x);
			btGjkEpaSolver2::SignedDistance(&triangle2, triangle_transform, csh, wtr, guess, results);

			dst = results.distance - csh->getMargin() - margin;
			return true;
		}
	}

	// Use triangle-convex CD.
	wtr = colObjWrap->getWorldTransform();
	btTriangleShape triangle2(btVector3(0, 0, 0), f.m_n[1]->m_x - f.m_n[0]->m_x, f.m_n[2]->m_x - f.m_n[0]->m_x);
	triangle_transform.setOrigin(f.m_n[0]->m_x);
	btGjkEpaSolver2::SignedDistance(&triangle2, triangle_transform, csh, wtr, guess, results);
	contact_point = results.witnesses[0];
	getBarycentric(contact_point, f.m_n[0]->m_x, f.m_n[1]->m_x, f.m_n[2]->m_x, bary);

	for (int i = 0; i < 3; ++i)
		f.m_pcontact[i] = bary[i];

	dst = results.distance - csh->getMargin() - margin;
	cti.m_colObj = colObjWrap->getCollisionObject();
	cti.m_normal = results.normal;
	cti.m_offset = dst;
	return true;
}

void btSoftBody::updateNormals()
{
	const btVector3 zv(0, 0, 0);
	int i, ni;

	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		m_nodes[i].m_n = zv;
	}
	for (i = 0, ni = m_faces.size(); i < ni; ++i)
	{
		btSoftBody::Face& f = m_faces[i];
		const btVector3 n = btCross(f.m_n[1]->m_x - f.m_n[0]->m_x,
									f.m_n[2]->m_x - f.m_n[0]->m_x);
		f.m_normal = n;
		f.m_normal.safeNormalize();
		f.m_n[0]->m_n += n;
		f.m_n[1]->m_n += n;
		f.m_n[2]->m_n += n;
	}
	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		btScalar len = m_nodes[i].m_n.length();
		if (len > SIMD_EPSILON)
			m_nodes[i].m_n /= len;
	}
}

//
void btSoftBody::updateBounds()
{
	/*if( m_acceleratedSoftBody )
	{
		// If we have an accelerated softbody we need to obtain the bounds correctly
		// For now (slightly hackily) just have a very large AABB
		// TODO: Write get bounds kernel
		// If that is updating in place, atomic collisions might be low (when the cloth isn't perfectly aligned to an axis) and we could
		// probably do a test and exchange reasonably efficiently.

		m_bounds[0] = btVector3(-1000, -1000, -1000);
		m_bounds[1] = btVector3(1000, 1000, 1000);

	} else {*/
	//    if (m_ndbvt.m_root)
	//    {
	//        const btVector3& mins = m_ndbvt.m_root->volume.Mins();
	//        const btVector3& maxs = m_ndbvt.m_root->volume.Maxs();
	//        const btScalar csm = getCollisionShape()->getMargin();
	//        const btVector3 mrg = btVector3(csm,
	//                                        csm,
	//                                        csm) *
	//                              1;  // ??? to investigate...
	//        m_bounds[0] = mins - mrg;
	//        m_bounds[1] = maxs + mrg;
	//        if (0 != getBroadphaseHandle())
	//        {
	//            m_worldInfo->m_broadphase->setAabb(getBroadphaseHandle(),
	//                                               m_bounds[0],
	//                                               m_bounds[1],
	//                                               m_worldInfo->m_dispatcher);
	//        }
	//    }
	//    else
	//    {
	//        m_bounds[0] =
	//            m_bounds[1] = btVector3(0, 0, 0);
	//    }
	if (m_nodes.size())
	{
		btVector3 mins = m_nodes[0].m_x;
		btVector3 maxs = m_nodes[0].m_x;
		for (int i = 1; i < m_nodes.size(); ++i)
		{
			for (int d = 0; d < 3; ++d)
			{
				if (m_nodes[i].m_x[d] > maxs[d])
					maxs[d] = m_nodes[i].m_x[d];
				if (m_nodes[i].m_x[d] < mins[d])
					mins[d] = m_nodes[i].m_x[d];
			}
		}
		const btScalar csm = getCollisionShape()->getMargin();
		const btVector3 mrg = btVector3(csm,
										csm,
										csm);
		m_bounds[0] = mins - mrg;
		m_bounds[1] = maxs + mrg;
		if (0 != getBroadphaseHandle())
		{
			m_worldInfo->m_broadphase->setAabb(getBroadphaseHandle(),
											   m_bounds[0],
											   m_bounds[1],
											   m_worldInfo->m_dispatcher);
		}
	}
	else
	{
		m_bounds[0] =
			m_bounds[1] = btVector3(0, 0, 0);
	}
}

//
void btSoftBody::updatePose()
{
	if (m_pose.m_bframe)
	{
		btSoftBody::Pose& pose = m_pose;
		const btVector3 com = evaluateCom();
		/* Com			*/
		pose.m_com = com;
		/* Rotation		*/
		btMatrix3x3 Apq;
		const btScalar eps = SIMD_EPSILON;
		Apq[0] = Apq[1] = Apq[2] = btVector3(0, 0, 0);
		Apq[0].setX(eps);
		Apq[1].setY(eps * 2);
		Apq[2].setZ(eps * 3);
		for (int i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			const btVector3 a = pose.m_wgh[i] * (m_nodes[i].m_x - com);
			const btVector3& b = pose.m_pos[i];
			Apq[0] += a.x() * b;
			Apq[1] += a.y() * b;
			Apq[2] += a.z() * b;
		}
		btMatrix3x3 r, s;
		PolarDecompose(Apq, r, s);
		pose.m_rot = r;
		pose.m_scl = pose.m_aqq * r.transpose() * Apq;
		if (m_cfg.maxvolume > 1)
		{
			const btScalar idet = Clamp<btScalar>(1 / pose.m_scl.determinant(),
												  1, m_cfg.maxvolume);
			pose.m_scl = Mul(pose.m_scl, idet);
		}
	}
}

//
void btSoftBody::updateArea(bool averageArea)
{
	int i, ni;

	/* Face area		*/
	for (i = 0, ni = m_faces.size(); i < ni; ++i)
	{
		Face& f = m_faces[i];
		f.m_ra = AreaOf(f.m_n[0]->m_x, f.m_n[1]->m_x, f.m_n[2]->m_x);
	}

	/* Node area		*/

	if (averageArea)
	{
		btAlignedObjectArray<int> counts;
		counts.resize(m_nodes.size(), 0);
		for (i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			m_nodes[i].m_area = 0;
		}
		for (i = 0, ni = m_faces.size(); i < ni; ++i)
		{
			btSoftBody::Face& f = m_faces[i];
			for (int j = 0; j < 3; ++j)
			{
				const int index = (int)(f.m_n[j] - &m_nodes[0]);
				counts[index]++;
				f.m_n[j]->m_area += btFabs(f.m_ra);
			}
		}
		for (i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			if (counts[i] > 0)
				m_nodes[i].m_area /= (btScalar)counts[i];
			else
				m_nodes[i].m_area = 0;
		}
	}
	else
	{
		// initialize node area as zero
		for (i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			m_nodes[i].m_area = 0;
		}

		for (i = 0, ni = m_faces.size(); i < ni; ++i)
		{
			btSoftBody::Face& f = m_faces[i];

			for (int j = 0; j < 3; ++j)
			{
				f.m_n[j]->m_area += f.m_ra;
			}
		}

		for (i = 0, ni = m_nodes.size(); i < ni; ++i)
		{
			m_nodes[i].m_area *= 0.3333333f;
		}
	}
}

void btSoftBody::updateLinkConstants()
{
	int i, ni;

	/* Links		*/
	for (i = 0, ni = m_links.size(); i < ni; ++i)
	{
		Link& l = m_links[i];
		Material& m = *l.m_material;
		l.m_c0 = (l.m_n[0]->m_im + l.m_n[1]->m_im) / m.m_kLST;
	}
}

void btSoftBody::updateConstants()
{
	resetLinkRestLengths();
	updateLinkConstants();
	updateArea();
}

//
void btSoftBody::initializeClusters()
{
	int i;

	for (i = 0; i < m_clusters.size(); ++i)
	{
		Cluster& c = *m_clusters[i];
		c.m_imass = 0;
		c.m_masses.resize(c.m_nodes.size());
		for (int j = 0; j < c.m_nodes.size(); ++j)
		{
			if (c.m_nodes[j]->m_im == 0)
			{
				c.m_containsAnchor = true;
				c.m_masses[j] = BT_LARGE_FLOAT;
			}
			else
			{
				c.m_masses[j] = btScalar(1.) / c.m_nodes[j]->m_im;
			}
			c.m_imass += c.m_masses[j];
		}
		c.m_imass = btScalar(1.) / c.m_imass;
		c.m_com = btSoftBody::clusterCom(&c);
		c.m_lv = btVector3(0, 0, 0);
		c.m_av = btVector3(0, 0, 0);
		c.m_leaf = 0;
		/* Inertia	*/
		btMatrix3x3& ii = c.m_locii;
		ii[0] = ii[1] = ii[2] = btVector3(0, 0, 0);
		{
			int i, ni;

			for (i = 0, ni = c.m_nodes.size(); i < ni; ++i)
			{
				const btVector3 k = c.m_nodes[i]->m_x - c.m_com;
				const btVector3 q = k * k;
				const btScalar m = c.m_masses[i];
				ii[0][0] += m * (q[1] + q[2]);
				ii[1][1] += m * (q[0] + q[2]);
				ii[2][2] += m * (q[0] + q[1]);
				ii[0][1] -= m * k[0] * k[1];
				ii[0][2] -= m * k[0] * k[2];
				ii[1][2] -= m * k[1] * k[2];
			}
		}
		ii[1][0] = ii[0][1];
		ii[2][0] = ii[0][2];
		ii[2][1] = ii[1][2];

		ii = ii.inverse();

		/* Frame	*/
		c.m_framexform.setIdentity();
		c.m_framexform.setOrigin(c.m_com);
		c.m_framerefs.resize(c.m_nodes.size());
		{
			int i;
			for (i = 0; i < c.m_framerefs.size(); ++i)
			{
				c.m_framerefs[i] = c.m_nodes[i]->m_x - c.m_com;
			}
		}
	}
}

//
void btSoftBody::updateClusters()
{
	BT_PROFILE("UpdateClusters");
	int i;

	for (i = 0; i < m_clusters.size(); ++i)
	{
		btSoftBody::Cluster& c = *m_clusters[i];
		const int n = c.m_nodes.size();
		//const btScalar			invn=1/(btScalar)n;
		if (n)
		{
			/* Frame				*/
			const btScalar eps = btScalar(0.0001);
			btMatrix3x3 m, r, s;
			m[0] = m[1] = m[2] = btVector3(0, 0, 0);
			m[0][0] = eps * 1;
			m[1][1] = eps * 2;
			m[2][2] = eps * 3;
			c.m_com = clusterCom(&c);
			for (int i = 0; i < c.m_nodes.size(); ++i)
			{
				const btVector3 a = c.m_nodes[i]->m_x - c.m_com;
				const btVector3& b = c.m_framerefs[i];
				m[0] += a[0] * b;
				m[1] += a[1] * b;
				m[2] += a[2] * b;
			}
			PolarDecompose(m, r, s);
			c.m_framexform.setOrigin(c.m_com);
			c.m_framexform.setBasis(r);
			/* Inertia			*/
#if 1 /* Constant	*/
			c.m_invwi = c.m_framexform.getBasis() * c.m_locii * c.m_framexform.getBasis().transpose();
#else
#if 0 /* Sphere	*/ 
			const btScalar	rk=(2*c.m_extents.length2())/(5*c.m_imass);
			const btVector3	inertia(rk,rk,rk);
			const btVector3	iin(btFabs(inertia[0])>SIMD_EPSILON?1/inertia[0]:0,
				btFabs(inertia[1])>SIMD_EPSILON?1/inertia[1]:0,
				btFabs(inertia[2])>SIMD_EPSILON?1/inertia[2]:0);

			c.m_invwi=c.m_xform.getBasis().scaled(iin)*c.m_xform.getBasis().transpose();
#else /* Actual	*/
			c.m_invwi[0] = c.m_invwi[1] = c.m_invwi[2] = btVector3(0, 0, 0);
			for (int i = 0; i < n; ++i)
			{
				const btVector3 k = c.m_nodes[i]->m_x - c.m_com;
				const btVector3 q = k * k;
				const btScalar m = 1 / c.m_nodes[i]->m_im;
				c.m_invwi[0][0] += m * (q[1] + q[2]);
				c.m_invwi[1][1] += m * (q[0] + q[2]);
				c.m_invwi[2][2] += m * (q[0] + q[1]);
				c.m_invwi[0][1] -= m * k[0] * k[1];
				c.m_invwi[0][2] -= m * k[0] * k[2];
				c.m_invwi[1][2] -= m * k[1] * k[2];
			}
			c.m_invwi[1][0] = c.m_invwi[0][1];
			c.m_invwi[2][0] = c.m_invwi[0][2];
			c.m_invwi[2][1] = c.m_invwi[1][2];
			c.m_invwi = c.m_invwi.inverse();
#endif
#endif
			/* Velocities			*/
			c.m_lv = btVector3(0, 0, 0);
			c.m_av = btVector3(0, 0, 0);
			{
				int i;

				for (i = 0; i < n; ++i)
				{
					const btVector3 v = c.m_nodes[i]->m_v * c.m_masses[i];
					c.m_lv += v;
					c.m_av += btCross(c.m_nodes[i]->m_x - c.m_com, v);
				}
			}
			c.m_lv = c.m_imass * c.m_lv * (1 - c.m_ldamping);
			c.m_av = c.m_invwi * c.m_av * (1 - c.m_adamping);
			c.m_vimpulses[0] =
				c.m_vimpulses[1] = btVector3(0, 0, 0);
			c.m_dimpulses[0] =
				c.m_dimpulses[1] = btVector3(0, 0, 0);
			c.m_nvimpulses = 0;
			c.m_ndimpulses = 0;
			/* Matching				*/
			if (c.m_matching > 0)
			{
				for (int j = 0; j < c.m_nodes.size(); ++j)
				{
					Node& n = *c.m_nodes[j];
					const btVector3 x = c.m_framexform * c.m_framerefs[j];
					n.m_x = Lerp(n.m_x, x, c.m_matching);
				}
			}
			/* Dbvt					*/
			if (c.m_collide)
			{
				btVector3 mi = c.m_nodes[0]->m_x;
				btVector3 mx = mi;
				for (int j = 1; j < n; ++j)
				{
					mi.setMin(c.m_nodes[j]->m_x);
					mx.setMax(c.m_nodes[j]->m_x);
				}
				ATTRIBUTE_ALIGNED16(btDbvtVolume)
				bounds = btDbvtVolume::FromMM(mi, mx);
				if (c.m_leaf)
					m_cdbvt.update(c.m_leaf, bounds, c.m_lv * m_sst.sdt * 3, m_sst.radmrg);
				else
					c.m_leaf = m_cdbvt.insert(bounds, &c);
			}
		}
	}
}

//
void btSoftBody::cleanupClusters()
{
	for (int i = 0; i < m_joints.size(); ++i)
	{
		m_joints[i]->Terminate(m_sst.sdt);
		if (m_joints[i]->m_delete)
		{
			btAlignedFree(m_joints[i]);
			m_joints.remove(m_joints[i--]);
		}
	}
}

//
void btSoftBody::prepareClusters(int iterations)
{
	for (int i = 0; i < m_joints.size(); ++i)
	{
		m_joints[i]->Prepare(m_sst.sdt, iterations);
	}
}

//
void btSoftBody::solveClusters(btScalar sor)
{
	for (int i = 0, ni = m_joints.size(); i < ni; ++i)
	{
		m_joints[i]->Solve(m_sst.sdt, sor);
	}
}

//
void btSoftBody::applyClusters(bool drift)
{
	BT_PROFILE("ApplyClusters");
	//	const btScalar					f0=m_sst.sdt;
	//const btScalar					f1=f0/2;
	btAlignedObjectArray<btVector3> deltas;
	btAlignedObjectArray<btScalar> weights;
	deltas.resize(m_nodes.size(), btVector3(0, 0, 0));
	weights.resize(m_nodes.size(), 0);
	int i;

	if (drift)
	{
		for (i = 0; i < m_clusters.size(); ++i)
		{
			Cluster& c = *m_clusters[i];
			if (c.m_ndimpulses)
			{
				c.m_dimpulses[0] /= (btScalar)c.m_ndimpulses;
				c.m_dimpulses[1] /= (btScalar)c.m_ndimpulses;
			}
		}
	}

	for (i = 0; i < m_clusters.size(); ++i)
	{
		Cluster& c = *m_clusters[i];
		if (0 < (drift ? c.m_ndimpulses : c.m_nvimpulses))
		{
			const btVector3 v = (drift ? c.m_dimpulses[0] : c.m_vimpulses[0]) * m_sst.sdt;
			const btVector3 w = (drift ? c.m_dimpulses[1] : c.m_vimpulses[1]) * m_sst.sdt;
			for (int j = 0; j < c.m_nodes.size(); ++j)
			{
				const int idx = int(c.m_nodes[j] - &m_nodes[0]);
				const btVector3& x = c.m_nodes[j]->m_x;
				const btScalar q = c.m_masses[j];
				deltas[idx] += (v + btCross(w, x - c.m_com)) * q;
				weights[idx] += q;
			}
		}
	}
	for (i = 0; i < deltas.size(); ++i)
	{
		if (weights[i] > 0)
		{
			m_nodes[i].m_x += deltas[i] / weights[i];
		}
	}
}

//
void btSoftBody::dampClusters()
{
	int i;

	for (i = 0; i < m_clusters.size(); ++i)
	{
		Cluster& c = *m_clusters[i];
		if (c.m_ndamping > 0)
		{
			for (int j = 0; j < c.m_nodes.size(); ++j)
			{
				Node& n = *c.m_nodes[j];
				if (n.m_im > 0)
				{
					const btVector3 vx = c.m_lv + btCross(c.m_av, c.m_nodes[j]->m_q - c.m_com);
					if (vx.length2() <= n.m_v.length2())
					{
						n.m_v += c.m_ndamping * (vx - n.m_v);
					}
				}
			}
		}
	}
}

void btSoftBody::setSpringStiffness(btScalar k)
{
	for (int i = 0; i < m_links.size(); ++i)
	{
		m_links[i].Feature::m_material->m_kLST = k;
	}
	m_repulsionStiffness = k;
}

void btSoftBody::setGravityFactor(btScalar gravFactor)
{
	m_gravityFactor = gravFactor;
}

void btSoftBody::setCacheBarycenter(bool cacheBarycenter)
{
	m_cacheBarycenter = cacheBarycenter;
}

void btSoftBody::initializeDmInverse()
{
	btScalar unit_simplex_measure = 1. / 6.;

	for (int i = 0; i < m_tetras.size(); ++i)
	{
		Tetra& t = m_tetras[i];
		btVector3 c1 = t.m_n[1]->m_x - t.m_n[0]->m_x;
		btVector3 c2 = t.m_n[2]->m_x - t.m_n[0]->m_x;
		btVector3 c3 = t.m_n[3]->m_x - t.m_n[0]->m_x;
		btMatrix3x3 Dm(c1.getX(), c2.getX(), c3.getX(),
					   c1.getY(), c2.getY(), c3.getY(),
					   c1.getZ(), c2.getZ(), c3.getZ());
		t.m_element_measure = Dm.determinant() * unit_simplex_measure;
		t.m_Dm_inverse = Dm.inverse();

		// calculate the first three columns of P^{-1}
		btVector3 a = t.m_n[0]->m_x;
		btVector3 b = t.m_n[1]->m_x;
		btVector3 c = t.m_n[2]->m_x;
		btVector3 d = t.m_n[3]->m_x;

		btScalar det = 1 / (a[0] * b[1] * c[2] - a[0] * b[1] * d[2] - a[0] * b[2] * c[1] + a[0] * b[2] * d[1] + a[0] * c[1] * d[2] - a[0] * c[2] * d[1] + a[1] * (-b[0] * c[2] + b[0] * d[2] + b[2] * c[0] - b[2] * d[0] - c[0] * d[2] + c[2] * d[0]) + a[2] * (b[0] * c[1] - b[0] * d[1] + b[1] * (d[0] - c[0]) + c[0] * d[1] - c[1] * d[0]) - b[0] * c[1] * d[2] + b[0] * c[2] * d[1] + b[1] * c[0] * d[2] - b[1] * c[2] * d[0] - b[2] * c[0] * d[1] + b[2] * c[1] * d[0]);

		btScalar P11 = -b[2] * c[1] + d[2] * c[1] + b[1] * c[2] + b[2] * d[1] - c[2] * d[1] - b[1] * d[2];
		btScalar P12 = b[2] * c[0] - d[2] * c[0] - b[0] * c[2] - b[2] * d[0] + c[2] * d[0] + b[0] * d[2];
		btScalar P13 = -b[1] * c[0] + d[1] * c[0] + b[0] * c[1] + b[1] * d[0] - c[1] * d[0] - b[0] * d[1];
		btScalar P21 = a[2] * c[1] - d[2] * c[1] - a[1] * c[2] - a[2] * d[1] + c[2] * d[1] + a[1] * d[2];
		btScalar P22 = -a[2] * c[0] + d[2] * c[0] + a[0] * c[2] + a[2] * d[0] - c[2] * d[0] - a[0] * d[2];
		btScalar P23 = a[1] * c[0] - d[1] * c[0] - a[0] * c[1] - a[1] * d[0] + c[1] * d[0] + a[0] * d[1];
		btScalar P31 = -a[2] * b[1] + d[2] * b[1] + a[1] * b[2] + a[2] * d[1] - b[2] * d[1] - a[1] * d[2];
		btScalar P32 = a[2] * b[0] - d[2] * b[0] - a[0] * b[2] - a[2] * d[0] + b[2] * d[0] + a[0] * d[2];
		btScalar P33 = -a[1] * b[0] + d[1] * b[0] + a[0] * b[1] + a[1] * d[0] - b[1] * d[0] - a[0] * d[1];
		btScalar P41 = a[2] * b[1] - c[2] * b[1] - a[1] * b[2] - a[2] * c[1] + b[2] * c[1] + a[1] * c[2];
		btScalar P42 = -a[2] * b[0] + c[2] * b[0] + a[0] * b[2] + a[2] * c[0] - b[2] * c[0] - a[0] * c[2];
		btScalar P43 = a[1] * b[0] - c[1] * b[0] - a[0] * b[1] - a[1] * c[0] + b[1] * c[0] + a[0] * c[1];

		btVector4 p1(P11 * det, P21 * det, P31 * det, P41 * det);
		btVector4 p2(P12 * det, P22 * det, P32 * det, P42 * det);
		btVector4 p3(P13 * det, P23 * det, P33 * det, P43 * det);

		t.m_P_inv[0] = p1;
		t.m_P_inv[1] = p2;
		t.m_P_inv[2] = p3;
	}
}

static btScalar Dot4(const btVector4& a, const btVector4& b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

void btSoftBody::updateDeformation()
{
	btQuaternion q;
	for (int i = 0; i < m_tetras.size(); ++i)
	{
		btSoftBody::Tetra& t = m_tetras[i];
		btVector3 c1 = t.m_n[1]->m_q - t.m_n[0]->m_q;
		btVector3 c2 = t.m_n[2]->m_q - t.m_n[0]->m_q;
		btVector3 c3 = t.m_n[3]->m_q - t.m_n[0]->m_q;
		btMatrix3x3 Ds(c1.getX(), c2.getX(), c3.getX(),
					   c1.getY(), c2.getY(), c3.getY(),
					   c1.getZ(), c2.getZ(), c3.getZ());
		t.m_F = Ds * t.m_Dm_inverse;

		btSoftBody::TetraScratch& s = m_tetraScratches[i];
		s.m_F = t.m_F;
		s.m_J = t.m_F.determinant();
		btMatrix3x3 C = t.m_F.transpose() * t.m_F;
		s.m_trace = C[0].getX() + C[1].getY() + C[2].getZ();
		s.m_cofF = t.m_F.adjoint().transpose();

		btVector3 a = t.m_n[0]->m_q;
		btVector3 b = t.m_n[1]->m_q;
		btVector3 c = t.m_n[2]->m_q;
		btVector3 d = t.m_n[3]->m_q;
		btVector4 q1(a[0], b[0], c[0], d[0]);
		btVector4 q2(a[1], b[1], c[1], d[1]);
		btVector4 q3(a[2], b[2], c[2], d[2]);
		btMatrix3x3 B(Dot4(q1, t.m_P_inv[0]), Dot4(q1, t.m_P_inv[1]), Dot4(q1, t.m_P_inv[2]),
					  Dot4(q2, t.m_P_inv[0]), Dot4(q2, t.m_P_inv[1]), Dot4(q2, t.m_P_inv[2]),
					  Dot4(q3, t.m_P_inv[0]), Dot4(q3, t.m_P_inv[1]), Dot4(q3, t.m_P_inv[2]));
		q.setRotation(btVector3(0, 0, 1), 0);
		B.extractRotation(q, 0.01);  // precision of the rotation is not very important for visual correctness.
		btMatrix3x3 Q(q);
		s.m_corotation = Q;
	}
}

void btSoftBody::advanceDeformation()
{
	updateDeformation();
	for (int i = 0; i < m_tetras.size(); ++i)
	{
		m_tetraScratchesTn[i] = m_tetraScratches[i];
	}
}
//
void btSoftBody::Joint::Prepare(btScalar dt, int)
{
	m_bodies[0].activate();
	m_bodies[1].activate();
}

//
void btSoftBody::LJoint::Prepare(btScalar dt, int iterations)
{
	static const btScalar maxdrift = 4;
	Joint::Prepare(dt, iterations);
	m_rpos[0] = m_bodies[0].xform() * m_refs[0];
	m_rpos[1] = m_bodies[1].xform() * m_refs[1];
	m_drift = Clamp(m_rpos[0] - m_rpos[1], maxdrift) * m_erp / dt;
	m_rpos[0] -= m_bodies[0].xform().getOrigin();
	m_rpos[1] -= m_bodies[1].xform().getOrigin();
	m_massmatrix = ImpulseMatrix(m_bodies[0].invMass(), m_bodies[0].invWorldInertia(), m_rpos[0],
								 m_bodies[1].invMass(), m_bodies[1].invWorldInertia(), m_rpos[1]);
	if (m_split > 0)
	{
		m_sdrift = m_massmatrix * (m_drift * m_split);
		m_drift *= 1 - m_split;
	}
	m_drift /= (btScalar)iterations;
}

//
void btSoftBody::LJoint::Solve(btScalar dt, btScalar sor)
{
	const btVector3 va = m_bodies[0].velocity(m_rpos[0]);
	const btVector3 vb = m_bodies[1].velocity(m_rpos[1]);
	const btVector3 vr = va - vb;
	btSoftBody::Impulse impulse;
	impulse.m_asVelocity = 1;
	impulse.m_velocity = m_massmatrix * (m_drift + vr * m_cfm) * sor;
	m_bodies[0].applyImpulse(-impulse, m_rpos[0]);
	m_bodies[1].applyImpulse(impulse, m_rpos[1]);
}

//
void btSoftBody::LJoint::Terminate(btScalar dt)
{
	if (m_split > 0)
	{
		m_bodies[0].applyDImpulse(-m_sdrift, m_rpos[0]);
		m_bodies[1].applyDImpulse(m_sdrift, m_rpos[1]);
	}
}

//
void btSoftBody::AJoint::Prepare(btScalar dt, int iterations)
{
	static const btScalar maxdrift = SIMD_PI / 16;
	m_icontrol->Prepare(this);
	Joint::Prepare(dt, iterations);
	m_axis[0] = m_bodies[0].xform().getBasis() * m_refs[0];
	m_axis[1] = m_bodies[1].xform().getBasis() * m_refs[1];
	m_drift = NormalizeAny(btCross(m_axis[1], m_axis[0]));
	m_drift *= btMin(maxdrift, btAcos(Clamp<btScalar>(btDot(m_axis[0], m_axis[1]), -1, +1)));
	m_drift *= m_erp / dt;
	m_massmatrix = AngularImpulseMatrix(m_bodies[0].invWorldInertia(), m_bodies[1].invWorldInertia());
	if (m_split > 0)
	{
		m_sdrift = m_massmatrix * (m_drift * m_split);
		m_drift *= 1 - m_split;
	}
	m_drift /= (btScalar)iterations;
}

//
void btSoftBody::AJoint::Solve(btScalar dt, btScalar sor)
{
	const btVector3 va = m_bodies[0].angularVelocity();
	const btVector3 vb = m_bodies[1].angularVelocity();
	const btVector3 vr = va - vb;
	const btScalar sp = btDot(vr, m_axis[0]);
	const btVector3 vc = vr - m_axis[0] * m_icontrol->Speed(this, sp);
	btSoftBody::Impulse impulse;
	impulse.m_asVelocity = 1;
	impulse.m_velocity = m_massmatrix * (m_drift + vc * m_cfm) * sor;
	m_bodies[0].applyAImpulse(-impulse);
	m_bodies[1].applyAImpulse(impulse);
}

//
void btSoftBody::AJoint::Terminate(btScalar dt)
{
	if (m_split > 0)
	{
		m_bodies[0].applyDAImpulse(-m_sdrift);
		m_bodies[1].applyDAImpulse(m_sdrift);
	}
}

//
void btSoftBody::CJoint::Prepare(btScalar dt, int iterations)
{
	Joint::Prepare(dt, iterations);
	const bool dodrift = (m_life == 0);
	m_delete = (++m_life) > m_maxlife;
	if (dodrift)
	{
		m_drift = m_drift * m_erp / dt;
		if (m_split > 0)
		{
			m_sdrift = m_massmatrix * (m_drift * m_split);
			m_drift *= 1 - m_split;
		}
		m_drift /= (btScalar)iterations;
	}
	else
	{
		m_drift = m_sdrift = btVector3(0, 0, 0);
	}
}

//
void btSoftBody::CJoint::Solve(btScalar dt, btScalar sor)
{
	const btVector3 va = m_bodies[0].velocity(m_rpos[0]);
	const btVector3 vb = m_bodies[1].velocity(m_rpos[1]);
	const btVector3 vrel = va - vb;
	const btScalar rvac = btDot(vrel, m_normal);
	btSoftBody::Impulse impulse;
	impulse.m_asVelocity = 1;
	impulse.m_velocity = m_drift;
	if (rvac < 0)
	{
		const btVector3 iv = m_normal * rvac;
		const btVector3 fv = vrel - iv;
		impulse.m_velocity += iv + fv * m_friction;
	}
	impulse.m_velocity = m_massmatrix * impulse.m_velocity * sor;

	if (m_bodies[0].m_soft == m_bodies[1].m_soft)
	{
		if ((impulse.m_velocity.getX() == impulse.m_velocity.getX()) && (impulse.m_velocity.getY() == impulse.m_velocity.getY()) &&
			(impulse.m_velocity.getZ() == impulse.m_velocity.getZ()))
		{
			if (impulse.m_asVelocity)
			{
				if (impulse.m_velocity.length() < m_bodies[0].m_soft->m_maxSelfCollisionImpulse)
				{
				}
				else
				{
					m_bodies[0].applyImpulse(-impulse * m_bodies[0].m_soft->m_selfCollisionImpulseFactor, m_rpos[0]);
					m_bodies[1].applyImpulse(impulse * m_bodies[0].m_soft->m_selfCollisionImpulseFactor, m_rpos[1]);
				}
			}
		}
	}
	else
	{
		m_bodies[0].applyImpulse(-impulse, m_rpos[0]);
		m_bodies[1].applyImpulse(impulse, m_rpos[1]);
	}
}

//
void btSoftBody::CJoint::Terminate(btScalar dt)
{
	if (m_split > 0)
	{
		m_bodies[0].applyDImpulse(-m_sdrift, m_rpos[0]);
		m_bodies[1].applyDImpulse(m_sdrift, m_rpos[1]);
	}
}

//
void btSoftBody::applyForces()
{
	BT_PROFILE("SoftBody applyForces");
	//	const btScalar					dt =			m_sst.sdt;
	const btScalar kLF = m_cfg.kLF;
	const btScalar kDG = m_cfg.kDG;
	const btScalar kPR = m_cfg.kPR;
	const btScalar kVC = m_cfg.kVC;
	const bool as_lift = kLF > 0;
	const bool as_drag = kDG > 0;
	const bool as_pressure = kPR != 0;
	const bool as_volume = kVC > 0;
	const bool as_aero = as_lift ||
						 as_drag;
	//const bool						as_vaero =		as_aero	&&
	//												(m_cfg.aeromodel < btSoftBody::eAeroModel::F_TwoSided);
	//const bool						as_faero =		as_aero	&&
	//												(m_cfg.aeromodel >= btSoftBody::eAeroModel::F_TwoSided);
	const bool use_medium = as_aero;
	const bool use_volume = as_pressure ||
							as_volume;
	btScalar volume = 0;
	btScalar ivolumetp = 0;
	btScalar dvolumetv = 0;
	btSoftBody::sMedium medium;
	if (use_volume)
	{
		volume = getVolume();
		ivolumetp = 1 / btFabs(volume) * kPR;
		dvolumetv = (m_pose.m_volume - volume) * kVC;
	}
	/* Per vertex forces			*/
	int i, ni;

	for (i = 0, ni = m_nodes.size(); i < ni; ++i)
	{
		btSoftBody::Node& n = m_nodes[i];
		if (n.m_im > 0)
		{
			if (use_medium)
			{
				/* Aerodynamics			*/
				addAeroForceToNode(m_windVelocity, i);
			}
			/* Pressure				*/
			if (as_pressure)
			{
				n.m_f += n.m_n * (n.m_area * ivolumetp);
			}
			/* Volume				*/
			if (as_volume)
			{
				n.m_f += n.m_n * (n.m_area * dvolumetv);
			}
		}
	}

	/* Per face forces				*/
	for (i = 0, ni = m_faces.size(); i < ni; ++i)
	{
		//	btSoftBody::Face&	f=m_faces[i];

		/* Aerodynamics			*/
		addAeroForceToFace(m_windVelocity, i);
	}
}

//
void btSoftBody::setMaxStress(btScalar maxStress)
{
	m_cfg.m_maxStress = maxStress;
}

//
void btSoftBody::interpolateRenderMesh()
{
	if (m_z.size() > 0)
	{
		for (int i = 0; i < m_renderNodes.size(); ++i)
		{
			const Node* p0 = m_renderNodesParents[i][0];
			const Node* p1 = m_renderNodesParents[i][1];
			const Node* p2 = m_renderNodesParents[i][2];
			btVector3 normal = btCross(p1->m_x - p0->m_x, p2->m_x - p0->m_x);
			btVector3 unit_normal = normal.normalized();
			RenderNode& n = m_renderNodes[i];
			n.m_x.setZero();
			for (int j = 0; j < 3; ++j)
			{
				n.m_x += m_renderNodesParents[i][j]->m_x * m_renderNodesInterpolationWeights[i][j];
			}
			n.m_x += m_z[i] * unit_normal;
		}
	}
	else
	{
		for (int i = 0; i < m_renderNodes.size(); ++i)
		{
			RenderNode& n = m_renderNodes[i];
			n.m_x.setZero();
			for (int j = 0; j < 4; ++j)
			{
				if (m_renderNodesParents[i].size())
				{
					n.m_x += m_renderNodesParents[i][j]->m_x * m_renderNodesInterpolationWeights[i][j];
				}
			}
		}
	}
}

void btSoftBody::setCollisionQuadrature(int N)
{
	for (int i = 0; i <= N; ++i)
	{
		for (int j = 0; i + j <= N; ++j)
		{
			m_quads.push_back(btVector3(btScalar(i) / btScalar(N), btScalar(j) / btScalar(N), btScalar(N - i - j) / btScalar(N)));
		}
	}
}

//
void btSoftBody::PSolve_Anchors(btSoftBody* psb, btScalar kst, btScalar ti)
{
	BT_PROFILE("PSolve_Anchors");
	const btScalar kAHR = psb->m_cfg.kAHR * kst;
	const btScalar dt = psb->m_sst.sdt;
	for (int i = 0, ni = psb->m_anchors.size(); i < ni; ++i)
	{
		const Anchor& a = psb->m_anchors[i];
		const btTransform& t = a.m_body->getWorldTransform();
		Node& n = *a.m_node;
		const btVector3 wa = t * a.m_local;
		const btVector3 va = a.m_body->getVelocityInLocalPoint(a.m_c1) * dt;
		const btVector3 vb = n.m_x - n.m_q;
		const btVector3 vr = (va - vb) + (wa - n.m_x) * kAHR;
		const btVector3 impulse = a.m_c0 * vr * a.m_influence;
		n.m_x += impulse * a.m_c2;
		a.m_body->applyImpulse(-impulse, a.m_c1);
	}
}

//
void btSoftBody::PSolve_RContacts(btSoftBody* psb, btScalar kst, btScalar ti)
{
	BT_PROFILE("PSolve_RContacts");
	const btScalar dt = psb->m_sst.sdt;
	const btScalar mrg = psb->getCollisionShape()->getMargin();
	btMultiBodyJacobianData jacobianData;
	for (int i = 0, ni = psb->m_rcontacts.size(); i < ni; ++i)
	{
		const RContact& c = psb->m_rcontacts[i];
		const sCti& cti = c.m_cti;
		if (cti.m_colObj->hasContactResponse())
		{
			btVector3 va(0, 0, 0);
			btRigidBody* rigidCol = 0;
			btMultiBodyLinkCollider* multibodyLinkCol = 0;
			btScalar* deltaV = NULL;

			if (cti.m_colObj->getInternalType() == btCollisionObject::CO_RIGID_BODY)
			{
				rigidCol = (btRigidBody*)btRigidBody::upcast(cti.m_colObj);
				va = rigidCol ? rigidCol->getVelocityInLocalPoint(c.m_c1) * dt : btVector3(0, 0, 0);
			}
			else if (cti.m_colObj->getInternalType() == btCollisionObject::CO_FEATHERSTONE_LINK)
			{
				multibodyLinkCol = (btMultiBodyLinkCollider*)btMultiBodyLinkCollider::upcast(cti.m_colObj);
				if (multibodyLinkCol)
				{
					const int ndof = multibodyLinkCol->m_multiBody->getNumDofs() + 6;
					jacobianData.m_jacobians.resize(ndof);
					jacobianData.m_deltaVelocitiesUnitImpulse.resize(ndof);
					btScalar* jac = &jacobianData.m_jacobians[0];

					multibodyLinkCol->m_multiBody->fillContactJacobianMultiDof(multibodyLinkCol->m_link, c.m_node->m_x, cti.m_normal, jac, jacobianData.scratch_r, jacobianData.scratch_v, jacobianData.scratch_m);
					deltaV = &jacobianData.m_deltaVelocitiesUnitImpulse[0];
					multibodyLinkCol->m_multiBody->calcAccelerationDeltasMultiDof(&jacobianData.m_jacobians[0], deltaV, jacobianData.scratch_r, jacobianData.scratch_v);

					btScalar vel = 0.0;
					for (int j = 0; j < ndof; ++j)
					{
						vel += multibodyLinkCol->m_multiBody->getVelocityVector()[j] * jac[j];
					}
					va = cti.m_normal * vel * dt;
				}
			}

			const btVector3 vb = c.m_node->m_x - c.m_node->m_q;
			const btVector3 vr = vb - va;
			const btScalar dn = btDot(vr, cti.m_normal);
			if (dn <= SIMD_EPSILON)
			{
				const btScalar dp = btMin((btDot(c.m_node->m_x, cti.m_normal) + cti.m_offset), mrg);
				const btVector3 fv = vr - (cti.m_normal * dn);
				// c0 is the impulse matrix, c3 is 1 - the friction coefficient or 0, c4 is the contact hardness coefficient
				const btVector3 impulse = c.m_c0 * ((vr - (fv * c.m_c3) + (cti.m_normal * (dp * c.m_c4))) * kst);
				c.m_node->m_x -= impulse * c.m_c2;

				if (cti.m_colObj->getInternalType() == btCollisionObject::CO_RIGID_BODY)
				{
					if (rigidCol)
						rigidCol->applyImpulse(impulse, c.m_c1);
				}
				else if (cti.m_colObj->getInternalType() == btCollisionObject::CO_FEATHERSTONE_LINK)
				{
					if (multibodyLinkCol)
					{
						double multiplier = 0.5;
						multibodyLinkCol->m_multiBody->applyDeltaVeeMultiDof(deltaV, -impulse.length() * multiplier);
					}
				}
			}
		}
	}
}

//
void btSoftBody::PSolve_SContacts(btSoftBody* psb, btScalar, btScalar ti)
{
	BT_PROFILE("PSolve_SContacts");

	for (int i = 0, ni = psb->m_scontacts.size(); i < ni; ++i)
	{
		const SContact& c = psb->m_scontacts[i];
		const btVector3& nr = c.m_normal;
		Node& n = *c.m_node;
		Face& f = *c.m_face;
		const btVector3 p = BaryEval(f.m_n[0]->m_x,
									 f.m_n[1]->m_x,
									 f.m_n[2]->m_x,
									 c.m_weights);
		const btVector3 q = BaryEval(f.m_n[0]->m_q,
									 f.m_n[1]->m_q,
									 f.m_n[2]->m_q,
									 c.m_weights);
		const btVector3 vr = (n.m_x - n.m_q) - (p - q);
		btVector3 corr(0, 0, 0);
		btScalar dot = btDot(vr, nr);
		if (dot < 0)
		{
			const btScalar j = c.m_margin - (btDot(nr, n.m_x) - btDot(nr, p));
			corr += c.m_normal * j;
		}
		corr -= ProjectOnPlane(vr, nr) * c.m_friction;
		n.m_x += corr * c.m_cfm[0];
		f.m_n[0]->m_x -= corr * (c.m_cfm[1] * c.m_weights.x());
		f.m_n[1]->m_x -= corr * (c.m_cfm[1] * c.m_weights.y());
		f.m_n[2]->m_x -= corr * (c.m_cfm[1] * c.m_weights.z());
	}
}

//
void btSoftBody::PSolve_Links(btSoftBody* psb, btScalar kst, btScalar ti)
{
	BT_PROFILE("PSolve_Links");
	for (int i = 0, ni = psb->m_links.size(); i < ni; ++i)
	{
		Link& l = psb->m_links[i];
		if (l.m_c0 > 0)
		{
			Node& a = *l.m_n[0];
			Node& b = *l.m_n[1];
			const btVector3 del = b.m_x - a.m_x;
			const btScalar len = del.length2();
			if (l.m_c1 + len > SIMD_EPSILON)
			{
				const btScalar k = ((l.m_c1 - len) / (l.m_c0 * (l.m_c1 + len))) * kst;
				a.m_x -= del * (k * a.m_im);
				b.m_x += del * (k * b.m_im);
			}
		}
	}
}

//
void btSoftBody::VSolve_Links(btSoftBody* psb, btScalar kst)
{
	BT_PROFILE("VSolve_Links");
	for (int i = 0, ni = psb->m_links.size(); i < ni; ++i)
	{
		Link& l = psb->m_links[i];
		Node** n = l.m_n;
		const btScalar j = -btDot(l.m_c3, n[0]->m_v - n[1]->m_v) * l.m_c2 * kst;
		n[0]->m_v += l.m_c3 * (j * n[0]->m_im);
		n[1]->m_v -= l.m_c3 * (j * n[1]->m_im);
	}
}

//
btSoftBody::psolver_t btSoftBody::getSolver(ePSolver::_ solver)
{
	switch (solver)
	{
		case ePSolver::Anchors:
			return (&btSoftBody::PSolve_Anchors);
		case ePSolver::Linear:
			return (&btSoftBody::PSolve_Links);
		case ePSolver::RContacts:
			return (&btSoftBody::PSolve_RContacts);
		case ePSolver::SContacts:
			return (&btSoftBody::PSolve_SContacts);
		default:
		{
		}
	}
	return (0);
}

//
btSoftBody::vsolver_t btSoftBody::getSolver(eVSolver::_ solver)
{
	switch (solver)
	{
		case eVSolver::Linear:
			return (&btSoftBody::VSolve_Links);
		default:
		{
		}
	}
	return (0);
}

void btSoftBody::setSelfCollision(bool useSelfCollision)
{
	m_useSelfCollision = useSelfCollision;
}

bool btSoftBody::useSelfCollision()
{
	return m_useSelfCollision;
}

//
void btSoftBody::defaultCollisionHandler(const btCollisionObjectWrapper* pcoWrap)
{
	switch (m_cfg.collisions & fCollision::RVSmask)
	{
		case fCollision::SDF_RS:
		{
			btSoftColliders::CollideSDF_RS docollide;
			btRigidBody* prb1 = (btRigidBody*)btRigidBody::upcast(pcoWrap->getCollisionObject());
			btTransform wtr = pcoWrap->getWorldTransform();

			const btTransform ctr = pcoWrap->getWorldTransform();
			const btScalar timemargin = (wtr.getOrigin() - ctr.getOrigin()).length();
			const btScalar basemargin = getCollisionShape()->getMargin();
			btVector3 mins;
			btVector3 maxs;
			ATTRIBUTE_ALIGNED16(btDbvtVolume)
			volume;
			pcoWrap->getCollisionShape()->getAabb(pcoWrap->getWorldTransform(),
												  mins,
												  maxs);
			volume = btDbvtVolume::FromMM(mins, maxs);
			volume.Expand(btVector3(basemargin, basemargin, basemargin));
			docollide.psb = this;
			docollide.m_colObj1Wrap = pcoWrap;
			docollide.m_rigidBody = prb1;

			docollide.dynmargin = basemargin + timemargin;
			docollide.stamargin = basemargin;
			m_ndbvt.collideTV(m_ndbvt.m_root, volume, docollide);
		}
		break;
		case fCollision::CL_RS:
		{
			btSoftColliders::CollideCL_RS collider;
			collider.ProcessColObj(this, pcoWrap);
		}
		break;
		case fCollision::SDF_RD:
		{
			btRigidBody* prb1 = (btRigidBody*)btRigidBody::upcast(pcoWrap->getCollisionObject());
			if (this->isActive())
			{
				const btTransform wtr = pcoWrap->getWorldTransform();
				const btScalar timemargin = 0;
				const btScalar basemargin = getCollisionShape()->getMargin();
				btVector3 mins;
				btVector3 maxs;
				ATTRIBUTE_ALIGNED16(btDbvtVolume)
				volume;
				pcoWrap->getCollisionShape()->getAabb(wtr,
													  mins,
													  maxs);
				volume = btDbvtVolume::FromMM(mins, maxs);
				volume.Expand(btVector3(basemargin, basemargin, basemargin));
				if (m_cfg.collisions & fCollision::SDF_RDN)
				{
					btSoftColliders::CollideSDF_RD docollideNode;
					docollideNode.psb = this;
					docollideNode.m_colObj1Wrap = pcoWrap;
					docollideNode.m_rigidBody = prb1;
					docollideNode.dynmargin = basemargin + timemargin;
					docollideNode.stamargin = basemargin;
					m_ndbvt.collideTV(m_ndbvt.m_root, volume, docollideNode);
				}

				if (((pcoWrap->getCollisionObject()->getInternalType() == CO_RIGID_BODY) && (m_cfg.collisions & fCollision::SDF_RDF)) || ((pcoWrap->getCollisionObject()->getInternalType() == CO_FEATHERSTONE_LINK) && (m_cfg.collisions & fCollision::SDF_MDF)))
				{
					btSoftColliders::CollideSDF_RDF docollideFace;
					docollideFace.psb = this;
					docollideFace.m_colObj1Wrap = pcoWrap;
					docollideFace.m_rigidBody = prb1;
					docollideFace.dynmargin = basemargin + timemargin;
					docollideFace.stamargin = basemargin;
					m_fdbvt.collideTV(m_fdbvt.m_root, volume, docollideFace);
				}
			}
		}
		break;
	}
}

//
void btSoftBody::defaultCollisionHandler(btSoftBody* psb)
{
	BT_PROFILE("Deformable Collision");
	const int cf = m_cfg.collisions & psb->m_cfg.collisions;
	switch (cf & fCollision::SVSmask)
	{
		case fCollision::CL_SS:
		{
			//support self-collision if CL_SELF flag set
			if (this != psb || psb->m_cfg.collisions & fCollision::CL_SELF)
			{
				btSoftColliders::CollideCL_SS docollide;
				docollide.ProcessSoftSoft(this, psb);
			}
		}
		break;
		case fCollision::VF_SS:
		{
			//only self-collision for Cluster, not Vertex-Face yet
			if (this != psb)
			{
				btSoftColliders::CollideVF_SS docollide;
				/* common					*/
				docollide.mrg = getCollisionShape()->getMargin() +
								psb->getCollisionShape()->getMargin();
				/* psb0 nodes vs psb1 faces	*/
				docollide.psb[0] = this;
				docollide.psb[1] = psb;
				docollide.psb[0]->m_ndbvt.collideTT(docollide.psb[0]->m_ndbvt.m_root,
													docollide.psb[1]->m_fdbvt.m_root,
													docollide);
				/* psb1 nodes vs psb0 faces	*/
				docollide.psb[0] = psb;
				docollide.psb[1] = this;
				docollide.psb[0]->m_ndbvt.collideTT(docollide.psb[0]->m_ndbvt.m_root,
													docollide.psb[1]->m_fdbvt.m_root,
													docollide);
			}
		}
		break;
		case fCollision::VF_DD:
		{
			if (!psb->m_softSoftCollision)
				return;
			if (psb->isActive() || this->isActive())
			{
				if (this != psb)
				{
					btSoftColliders::CollideVF_DD docollide;
					/* common                    */
					docollide.mrg = getCollisionShape()->getMargin() +
									psb->getCollisionShape()->getMargin();
					/* psb0 nodes vs psb1 faces    */
					if (psb->m_tetras.size() > 0)
						docollide.useFaceNormal = true;
					else
						docollide.useFaceNormal = false;
					docollide.psb[0] = this;
					docollide.psb[1] = psb;
					docollide.psb[0]->m_ndbvt.collideTT(docollide.psb[0]->m_ndbvt.m_root,
														docollide.psb[1]->m_fdbvt.m_root,
														docollide);

					/* psb1 nodes vs psb0 faces    */
					if (this->m_tetras.size() > 0)
						docollide.useFaceNormal = true;
					else
						docollide.useFaceNormal = false;
					docollide.psb[0] = psb;
					docollide.psb[1] = this;
					docollide.psb[0]->m_ndbvt.collideTT(docollide.psb[0]->m_ndbvt.m_root,
														docollide.psb[1]->m_fdbvt.m_root,
														docollide);
				}
				else
				{
					if (psb->useSelfCollision())
					{
						btSoftColliders::CollideFF_DD docollide;
						docollide.mrg = 2 * getCollisionShape()->getMargin();
						docollide.psb[0] = this;
						docollide.psb[1] = psb;
						if (this->m_tetras.size() > 0)
							docollide.useFaceNormal = true;
						else
							docollide.useFaceNormal = false;
						/* psb0 faces vs psb0 faces    */
						calculateNormalCone(this->m_fdbvnt);
						this->m_fdbvt.selfCollideT(m_fdbvnt, docollide);
					}
				}
			}
		}
		break;
		default:
		{
		}
	}
}

void btSoftBody::geometricCollisionHandler(btSoftBody* psb)
{
	if (psb->isActive() || this->isActive())
	{
		if (this != psb)
		{
			btSoftColliders::CollideCCD docollide;
			/* common                    */
			docollide.mrg = SAFE_EPSILON;  // for rounding error instead of actual margin
			docollide.dt = psb->m_sst.sdt;
			/* psb0 nodes vs psb1 faces    */
			if (psb->m_tetras.size() > 0)
				docollide.useFaceNormal = true;
			else
				docollide.useFaceNormal = false;
			docollide.psb[0] = this;
			docollide.psb[1] = psb;
			docollide.psb[0]->m_ndbvt.collideTT(docollide.psb[0]->m_ndbvt.m_root,
												docollide.psb[1]->m_fdbvt.m_root,
												docollide);
			/* psb1 nodes vs psb0 faces    */
			if (this->m_tetras.size() > 0)
				docollide.useFaceNormal = true;
			else
				docollide.useFaceNormal = false;
			docollide.psb[0] = psb;
			docollide.psb[1] = this;
			docollide.psb[0]->m_ndbvt.collideTT(docollide.psb[0]->m_ndbvt.m_root,
												docollide.psb[1]->m_fdbvt.m_root,
												docollide);
		}
		else
		{
			if (psb->useSelfCollision())
			{
				btSoftColliders::CollideCCD docollide;
				docollide.mrg = SAFE_EPSILON;
				docollide.psb[0] = this;
				docollide.psb[1] = psb;
				docollide.dt = psb->m_sst.sdt;
				if (this->m_tetras.size() > 0)
					docollide.useFaceNormal = true;
				else
					docollide.useFaceNormal = false;
				/* psb0 faces vs psb0 faces    */
				calculateNormalCone(this->m_fdbvnt);  // should compute this outside of this scope
				this->m_fdbvt.selfCollideT(m_fdbvnt, docollide);
			}
		}
	}
}

void btSoftBody::setWindVelocity(const btVector3& velocity)
{
	m_windVelocity = velocity;
}

const btVector3& btSoftBody::getWindVelocity()
{
	return m_windVelocity;
}

int btSoftBody::calculateSerializeBufferSize() const
{
	int sz = sizeof(btSoftBodyData);
	return sz;
}

///fills the dataBuffer and returns the struct name (and 0 on failure)
const char* btSoftBody::serialize(void* dataBuffer, class btSerializer* serializer) const
{
	btSoftBodyData* sbd = (btSoftBodyData*)dataBuffer;

	btCollisionObject::serialize(&sbd->m_collisionObjectData, serializer);

	btHashMap<btHashPtr, int> m_nodeIndexMap;

	sbd->m_numMaterials = m_materials.size();
	sbd->m_materials = sbd->m_numMaterials ? (SoftBodyMaterialData**)serializer->getUniquePointer((void*)&m_materials) : 0;

	if (sbd->m_materials)
	{
		int sz = sizeof(SoftBodyMaterialData*);
		int numElem = sbd->m_numMaterials;
		btChunk* chunk = serializer->allocate(sz, numElem);
		//SoftBodyMaterialData** memPtr = chunk->m_oldPtr;
		SoftBodyMaterialData** memPtr = (SoftBodyMaterialData**)chunk->m_oldPtr;
		for (int i = 0; i < numElem; i++, memPtr++)
		{
			btSoftBody::Material* mat = m_materials[i];
			*memPtr = mat ? (SoftBodyMaterialData*)serializer->getUniquePointer((void*)mat) : 0;
			if (!serializer->findPointer(mat))
			{
				//serialize it here
				btChunk* chunk = serializer->allocate(sizeof(SoftBodyMaterialData), 1);
				SoftBodyMaterialData* memPtr = (SoftBodyMaterialData*)chunk->m_oldPtr;
				memPtr->m_flags = mat->m_flags;
				memPtr->m_angularStiffness = mat->m_kAST;
				memPtr->m_linearStiffness = mat->m_kLST;
				memPtr->m_volumeStiffness = mat->m_kVST;
				serializer->finalizeChunk(chunk, "SoftBodyMaterialData", BT_SBMATERIAL_CODE, mat);
			}
		}
		serializer->finalizeChunk(chunk, "SoftBodyMaterialData", BT_ARRAY_CODE, (void*)&m_materials);
	}

	sbd->m_numNodes = m_nodes.size();
	sbd->m_nodes = sbd->m_numNodes ? (SoftBodyNodeData*)serializer->getUniquePointer((void*)&m_nodes) : 0;
	if (sbd->m_nodes)
	{
		int sz = sizeof(SoftBodyNodeData);
		int numElem = sbd->m_numNodes;
		btChunk* chunk = serializer->allocate(sz, numElem);
		SoftBodyNodeData* memPtr = (SoftBodyNodeData*)chunk->m_oldPtr;
		for (int i = 0; i < numElem; i++, memPtr++)
		{
			m_nodes[i].m_f.serializeFloat(memPtr->m_accumulatedForce);
			memPtr->m_area = m_nodes[i].m_area;
			memPtr->m_attach = m_nodes[i].m_battach;
			memPtr->m_inverseMass = m_nodes[i].m_im;
			memPtr->m_material = m_nodes[i].m_material ? (SoftBodyMaterialData*)serializer->getUniquePointer((void*)m_nodes[i].m_material) : 0;
			m_nodes[i].m_n.serializeFloat(memPtr->m_normal);
			m_nodes[i].m_x.serializeFloat(memPtr->m_position);
			m_nodes[i].m_q.serializeFloat(memPtr->m_previousPosition);
			m_nodes[i].m_v.serializeFloat(memPtr->m_velocity);
			m_nodeIndexMap.insert(&m_nodes[i], i);
		}
		serializer->finalizeChunk(chunk, "SoftBodyNodeData", BT_SBNODE_CODE, (void*)&m_nodes);
	}

	sbd->m_numLinks = m_links.size();
	sbd->m_links = sbd->m_numLinks ? (SoftBodyLinkData*)serializer->getUniquePointer((void*)&m_links[0]) : 0;
	if (sbd->m_links)
	{
		int sz = sizeof(SoftBodyLinkData);
		int numElem = sbd->m_numLinks;
		btChunk* chunk = serializer->allocate(sz, numElem);
		SoftBodyLinkData* memPtr = (SoftBodyLinkData*)chunk->m_oldPtr;
		for (int i = 0; i < numElem; i++, memPtr++)
		{
			memPtr->m_bbending = m_links[i].m_bbending;
			memPtr->m_material = m_links[i].m_material ? (SoftBodyMaterialData*)serializer->getUniquePointer((void*)m_links[i].m_material) : 0;
			memPtr->m_nodeIndices[0] = m_links[i].m_n[0] ? m_links[i].m_n[0] - &m_nodes[0] : -1;
			memPtr->m_nodeIndices[1] = m_links[i].m_n[1] ? m_links[i].m_n[1] - &m_nodes[0] : -1;
			btAssert(memPtr->m_nodeIndices[0] < m_nodes.size());
			btAssert(memPtr->m_nodeIndices[1] < m_nodes.size());
			memPtr->m_restLength = m_links[i].m_rl;
		}
		serializer->finalizeChunk(chunk, "SoftBodyLinkData", BT_ARRAY_CODE, (void*)&m_links[0]);
	}

	sbd->m_numFaces = m_faces.size();
	sbd->m_faces = sbd->m_numFaces ? (SoftBodyFaceData*)serializer->getUniquePointer((void*)&m_faces[0]) : 0;
	if (sbd->m_faces)
	{
		int sz = sizeof(SoftBodyFaceData);
		int numElem = sbd->m_numFaces;
		btChunk* chunk = serializer->allocate(sz, numElem);
		SoftBodyFaceData* memPtr = (SoftBodyFaceData*)chunk->m_oldPtr;
		for (int i = 0; i < numElem; i++, memPtr++)
		{
			memPtr->m_material = m_faces[i].m_material ? (SoftBodyMaterialData*)serializer->getUniquePointer((void*)m_faces[i].m_material) : 0;
			m_faces[i].m_normal.serializeFloat(memPtr->m_normal);
			for (int j = 0; j < 3; j++)
			{
				memPtr->m_nodeIndices[j] = m_faces[i].m_n[j] ? m_faces[i].m_n[j] - &m_nodes[0] : -1;
			}
			memPtr->m_restArea = m_faces[i].m_ra;
		}
		serializer->finalizeChunk(chunk, "SoftBodyFaceData", BT_ARRAY_CODE, (void*)&m_faces[0]);
	}

	sbd->m_numTetrahedra = m_tetras.size();
	sbd->m_tetrahedra = sbd->m_numTetrahedra ? (SoftBodyTetraData*)serializer->getUniquePointer((void*)&m_tetras[0]) : 0;
	if (sbd->m_tetrahedra)
	{
		int sz = sizeof(SoftBodyTetraData);
		int numElem = sbd->m_numTetrahedra;
		btChunk* chunk = serializer->allocate(sz, numElem);
		SoftBodyTetraData* memPtr = (SoftBodyTetraData*)chunk->m_oldPtr;
		for (int i = 0; i < numElem; i++, memPtr++)
		{
			for (int j = 0; j < 4; j++)
			{
				m_tetras[i].m_c0[j].serializeFloat(memPtr->m_c0[j]);
				memPtr->m_nodeIndices[j] = m_tetras[i].m_n[j] ? m_tetras[i].m_n[j] - &m_nodes[0] : -1;
			}
			memPtr->m_c1 = m_tetras[i].m_c1;
			memPtr->m_c2 = m_tetras[i].m_c2;
			memPtr->m_material = m_tetras[i].m_material ? (SoftBodyMaterialData*)serializer->getUniquePointer((void*)m_tetras[i].m_material) : 0;
			memPtr->m_restVolume = m_tetras[i].m_rv;
		}
		serializer->finalizeChunk(chunk, "SoftBodyTetraData", BT_ARRAY_CODE, (void*)&m_tetras[0]);
	}

	sbd->m_numAnchors = m_anchors.size();
	sbd->m_anchors = sbd->m_numAnchors ? (SoftRigidAnchorData*)serializer->getUniquePointer((void*)&m_anchors[0]) : 0;
	if (sbd->m_anchors)
	{
		int sz = sizeof(SoftRigidAnchorData);
		int numElem = sbd->m_numAnchors;
		btChunk* chunk = serializer->allocate(sz, numElem);
		SoftRigidAnchorData* memPtr = (SoftRigidAnchorData*)chunk->m_oldPtr;
		for (int i = 0; i < numElem; i++, memPtr++)
		{
			m_anchors[i].m_c0.serializeFloat(memPtr->m_c0);
			m_anchors[i].m_c1.serializeFloat(memPtr->m_c1);
			memPtr->m_c2 = m_anchors[i].m_c2;
			m_anchors[i].m_local.serializeFloat(memPtr->m_localFrame);
			memPtr->m_nodeIndex = m_anchors[i].m_node ? m_anchors[i].m_node - &m_nodes[0] : -1;

			memPtr->m_rigidBody = m_anchors[i].m_body ? (btRigidBodyData*)serializer->getUniquePointer((void*)m_anchors[i].m_body) : 0;
			btAssert(memPtr->m_nodeIndex < m_nodes.size());
		}
		serializer->finalizeChunk(chunk, "SoftRigidAnchorData", BT_ARRAY_CODE, (void*)&m_anchors[0]);
	}

	sbd->m_config.m_dynamicFriction = m_cfg.kDF;
	sbd->m_config.m_baumgarte = m_cfg.kVCF;
	sbd->m_config.m_pressure = m_cfg.kPR;
	sbd->m_config.m_aeroModel = this->m_cfg.aeromodel;
	sbd->m_config.m_lift = m_cfg.kLF;
	sbd->m_config.m_drag = m_cfg.kDG;
	sbd->m_config.m_positionIterations = m_cfg.piterations;
	sbd->m_config.m_driftIterations = m_cfg.diterations;
	sbd->m_config.m_clusterIterations = m_cfg.citerations;
	sbd->m_config.m_velocityIterations = m_cfg.viterations;
	sbd->m_config.m_maxVolume = m_cfg.maxvolume;
	sbd->m_config.m_damping = m_cfg.kDP;
	sbd->m_config.m_poseMatch = m_cfg.kMT;
	sbd->m_config.m_collisionFlags = m_cfg.collisions;
	sbd->m_config.m_volume = m_cfg.kVC;
	sbd->m_config.m_rigidContactHardness = m_cfg.kCHR;
	sbd->m_config.m_kineticContactHardness = m_cfg.kKHR;
	sbd->m_config.m_softContactHardness = m_cfg.kSHR;
	sbd->m_config.m_anchorHardness = m_cfg.kAHR;
	sbd->m_config.m_timeScale = m_cfg.timescale;
	sbd->m_config.m_maxVolume = m_cfg.maxvolume;
	sbd->m_config.m_softRigidClusterHardness = m_cfg.kSRHR_CL;
	sbd->m_config.m_softKineticClusterHardness = m_cfg.kSKHR_CL;
	sbd->m_config.m_softSoftClusterHardness = m_cfg.kSSHR_CL;
	sbd->m_config.m_softRigidClusterImpulseSplit = m_cfg.kSR_SPLT_CL;
	sbd->m_config.m_softKineticClusterImpulseSplit = m_cfg.kSK_SPLT_CL;
	sbd->m_config.m_softSoftClusterImpulseSplit = m_cfg.kSS_SPLT_CL;

	//pose for shape matching
	{
		sbd->m_pose = (SoftBodyPoseData*)serializer->getUniquePointer((void*)&m_pose);

		int sz = sizeof(SoftBodyPoseData);
		btChunk* chunk = serializer->allocate(sz, 1);
		SoftBodyPoseData* memPtr = (SoftBodyPoseData*)chunk->m_oldPtr;

		m_pose.m_aqq.serializeFloat(memPtr->m_aqq);
		memPtr->m_bframe = m_pose.m_bframe;
		memPtr->m_bvolume = m_pose.m_bvolume;
		m_pose.m_com.serializeFloat(memPtr->m_com);

		memPtr->m_numPositions = m_pose.m_pos.size();
		memPtr->m_positions = memPtr->m_numPositions ? (btVector3FloatData*)serializer->getUniquePointer((void*)&m_pose.m_pos[0]) : 0;
		if (memPtr->m_numPositions)
		{
			int numElem = memPtr->m_numPositions;
			int sz = sizeof(btVector3Data);
			btChunk* chunk = serializer->allocate(sz, numElem);
			btVector3FloatData* memPtr = (btVector3FloatData*)chunk->m_oldPtr;
			for (int i = 0; i < numElem; i++, memPtr++)
			{
				m_pose.m_pos[i].serializeFloat(*memPtr);
			}
			serializer->finalizeChunk(chunk, "btVector3FloatData", BT_ARRAY_CODE, (void*)&m_pose.m_pos[0]);
		}
		memPtr->m_restVolume = m_pose.m_volume;
		m_pose.m_rot.serializeFloat(memPtr->m_rot);
		m_pose.m_scl.serializeFloat(memPtr->m_scale);

		memPtr->m_numWeigts = m_pose.m_wgh.size();
		memPtr->m_weights = memPtr->m_numWeigts ? (float*)serializer->getUniquePointer((void*)&m_pose.m_wgh[0]) : 0;
		if (memPtr->m_numWeigts)
		{
			int numElem = memPtr->m_numWeigts;
			int sz = sizeof(float);
			btChunk* chunk = serializer->allocate(sz, numElem);
			float* memPtr = (float*)chunk->m_oldPtr;
			for (int i = 0; i < numElem; i++, memPtr++)
			{
				*memPtr = m_pose.m_wgh[i];
			}
			serializer->finalizeChunk(chunk, "float", BT_ARRAY_CODE, (void*)&m_pose.m_wgh[0]);
		}

		serializer->finalizeChunk(chunk, "SoftBodyPoseData", BT_ARRAY_CODE, (void*)&m_pose);
	}

	//clusters for convex-cluster collision detection

	sbd->m_numClusters = m_clusters.size();
	sbd->m_clusters = sbd->m_numClusters ? (SoftBodyClusterData*)serializer->getUniquePointer((void*)m_clusters[0]) : 0;
	if (sbd->m_numClusters)
	{
		int numElem = sbd->m_numClusters;
		int sz = sizeof(SoftBodyClusterData);
		btChunk* chunk = serializer->allocate(sz, numElem);
		SoftBodyClusterData* memPtr = (SoftBodyClusterData*)chunk->m_oldPtr;
		for (int i = 0; i < numElem; i++, memPtr++)
		{
			memPtr->m_adamping = m_clusters[i]->m_adamping;
			m_clusters[i]->m_av.serializeFloat(memPtr->m_av);
			memPtr->m_clusterIndex = m_clusters[i]->m_clusterIndex;
			memPtr->m_collide = m_clusters[i]->m_collide;
			m_clusters[i]->m_com.serializeFloat(memPtr->m_com);
			memPtr->m_containsAnchor = m_clusters[i]->m_containsAnchor;
			m_clusters[i]->m_dimpulses[0].serializeFloat(memPtr->m_dimpulses[0]);
			m_clusters[i]->m_dimpulses[1].serializeFloat(memPtr->m_dimpulses[1]);
			m_clusters[i]->m_framexform.serializeFloat(memPtr->m_framexform);
			memPtr->m_idmass = m_clusters[i]->m_idmass;
			memPtr->m_imass = m_clusters[i]->m_imass;
			m_clusters[i]->m_invwi.serializeFloat(memPtr->m_invwi);
			memPtr->m_ldamping = m_clusters[i]->m_ldamping;
			m_clusters[i]->m_locii.serializeFloat(memPtr->m_locii);
			m_clusters[i]->m_lv.serializeFloat(memPtr->m_lv);
			memPtr->m_matching = m_clusters[i]->m_matching;
			memPtr->m_maxSelfCollisionImpulse = m_clusters[i]->m_maxSelfCollisionImpulse;
			memPtr->m_ndamping = m_clusters[i]->m_ndamping;
			memPtr->m_ldamping = m_clusters[i]->m_ldamping;
			memPtr->m_adamping = m_clusters[i]->m_adamping;
			memPtr->m_selfCollisionImpulseFactor = m_clusters[i]->m_selfCollisionImpulseFactor;

			memPtr->m_numFrameRefs = m_clusters[i]->m_framerefs.size();
			memPtr->m_numMasses = m_clusters[i]->m_masses.size();
			memPtr->m_numNodes = m_clusters[i]->m_nodes.size();

			memPtr->m_nvimpulses = m_clusters[i]->m_nvimpulses;
			m_clusters[i]->m_vimpulses[0].serializeFloat(memPtr->m_vimpulses[0]);
			m_clusters[i]->m_vimpulses[1].serializeFloat(memPtr->m_vimpulses[1]);
			memPtr->m_ndimpulses = m_clusters[i]->m_ndimpulses;

			memPtr->m_framerefs = memPtr->m_numFrameRefs ? (btVector3FloatData*)serializer->getUniquePointer((void*)&m_clusters[i]->m_framerefs[0]) : 0;
			if (memPtr->m_framerefs)
			{
				int numElem = memPtr->m_numFrameRefs;
				int sz = sizeof(btVector3FloatData);
				btChunk* chunk = serializer->allocate(sz, numElem);
				btVector3FloatData* memPtr = (btVector3FloatData*)chunk->m_oldPtr;
				for (int j = 0; j < numElem; j++, memPtr++)
				{
					m_clusters[i]->m_framerefs[j].serializeFloat(*memPtr);
				}
				serializer->finalizeChunk(chunk, "btVector3FloatData", BT_ARRAY_CODE, (void*)&m_clusters[i]->m_framerefs[0]);
			}

			memPtr->m_masses = memPtr->m_numMasses ? (float*)serializer->getUniquePointer((void*)&m_clusters[i]->m_masses[0]) : 0;
			if (memPtr->m_masses)
			{
				int numElem = memPtr->m_numMasses;
				int sz = sizeof(float);
				btChunk* chunk = serializer->allocate(sz, numElem);
				float* memPtr = (float*)chunk->m_oldPtr;
				for (int j = 0; j < numElem; j++, memPtr++)
				{
					*memPtr = m_clusters[i]->m_masses[j];
				}
				serializer->finalizeChunk(chunk, "float", BT_ARRAY_CODE, (void*)&m_clusters[i]->m_masses[0]);
			}

			memPtr->m_nodeIndices = memPtr->m_numNodes ? (int*)serializer->getUniquePointer((void*)&m_clusters[i]->m_nodes) : 0;
			if (memPtr->m_nodeIndices)
			{
				int numElem = memPtr->m_numMasses;
				int sz = sizeof(int);
				btChunk* chunk = serializer->allocate(sz, numElem);
				int* memPtr = (int*)chunk->m_oldPtr;
				for (int j = 0; j < numElem; j++, memPtr++)
				{
					int* indexPtr = m_nodeIndexMap.find(m_clusters[i]->m_nodes[j]);
					btAssert(indexPtr);
					*memPtr = *indexPtr;
				}
				serializer->finalizeChunk(chunk, "int", BT_ARRAY_CODE, (void*)&m_clusters[i]->m_nodes);
			}
		}
		serializer->finalizeChunk(chunk, "SoftBodyClusterData", BT_ARRAY_CODE, (void*)m_clusters[0]);
	}

	sbd->m_numJoints = m_joints.size();
	sbd->m_joints = m_joints.size() ? (btSoftBodyJointData*)serializer->getUniquePointer((void*)&m_joints[0]) : 0;

	if (sbd->m_joints)
	{
		int sz = sizeof(btSoftBodyJointData);
		int numElem = m_joints.size();
		btChunk* chunk = serializer->allocate(sz, numElem);
		btSoftBodyJointData* memPtr = (btSoftBodyJointData*)chunk->m_oldPtr;

		for (int i = 0; i < numElem; i++, memPtr++)
		{
			memPtr->m_jointType = (int)m_joints[i]->Type();
			m_joints[i]->m_refs[0].serializeFloat(memPtr->m_refs[0]);
			m_joints[i]->m_refs[1].serializeFloat(memPtr->m_refs[1]);
			memPtr->m_cfm = m_joints[i]->m_cfm;
			memPtr->m_erp = float(m_joints[i]->m_erp);
			memPtr->m_split = float(m_joints[i]->m_split);
			memPtr->m_delete = m_joints[i]->m_delete;

			for (int j = 0; j < 4; j++)
			{
				memPtr->m_relPosition[0].m_floats[j] = 0.f;
				memPtr->m_relPosition[1].m_floats[j] = 0.f;
			}
			memPtr->m_bodyA = 0;
			memPtr->m_bodyB = 0;
			if (m_joints[i]->m_bodies[0].m_soft)
			{
				memPtr->m_bodyAtype = BT_JOINT_SOFT_BODY_CLUSTER;
				memPtr->m_bodyA = serializer->getUniquePointer((void*)m_joints[i]->m_bodies[0].m_soft);
			}
			if (m_joints[i]->m_bodies[0].m_collisionObject)
			{
				memPtr->m_bodyAtype = BT_JOINT_COLLISION_OBJECT;
				memPtr->m_bodyA = serializer->getUniquePointer((void*)m_joints[i]->m_bodies[0].m_collisionObject);
			}
			if (m_joints[i]->m_bodies[0].m_rigid)
			{
				memPtr->m_bodyAtype = BT_JOINT_RIGID_BODY;
				memPtr->m_bodyA = serializer->getUniquePointer((void*)m_joints[i]->m_bodies[0].m_rigid);
			}

			if (m_joints[i]->m_bodies[1].m_soft)
			{
				memPtr->m_bodyBtype = BT_JOINT_SOFT_BODY_CLUSTER;
				memPtr->m_bodyB = serializer->getUniquePointer((void*)m_joints[i]->m_bodies[1].m_soft);
			}
			if (m_joints[i]->m_bodies[1].m_collisionObject)
			{
				memPtr->m_bodyBtype = BT_JOINT_COLLISION_OBJECT;
				memPtr->m_bodyB = serializer->getUniquePointer((void*)m_joints[i]->m_bodies[1].m_collisionObject);
			}
			if (m_joints[i]->m_bodies[1].m_rigid)
			{
				memPtr->m_bodyBtype = BT_JOINT_RIGID_BODY;
				memPtr->m_bodyB = serializer->getUniquePointer((void*)m_joints[i]->m_bodies[1].m_rigid);
			}
		}
		serializer->finalizeChunk(chunk, "btSoftBodyJointData", BT_ARRAY_CODE, (void*)&m_joints[0]);
	}

	return btSoftBodyDataName;
}

void btSoftBody::updateDeactivation(btScalar timeStep)
{
	if ((getActivationState() == ISLAND_SLEEPING) || (getActivationState() == DISABLE_DEACTIVATION))
		return;

	if (m_maxSpeedSquared < m_sleepingThreshold * m_sleepingThreshold)
	{
		m_deactivationTime += timeStep;
	}
	else
	{
		m_deactivationTime = btScalar(0.);
		setActivationState(0);
	}
}

void btSoftBody::setZeroVelocity()
{
	for (int i = 0; i < m_nodes.size(); ++i)
	{
		m_nodes[i].m_v.setZero();
	}
}

bool btSoftBody::wantsSleeping()
{
	if (getActivationState() == DISABLE_DEACTIVATION)
		return false;

	//disable deactivation
	if (gDisableDeactivation || (gDeactivationTime == btScalar(0.)))
		return false;

	if ((getActivationState() == ISLAND_SLEEPING) || (getActivationState() == WANTS_DEACTIVATION))
		return true;

	if (m_deactivationTime > gDeactivationTime)
	{
		return true;
	}
	return false;
}

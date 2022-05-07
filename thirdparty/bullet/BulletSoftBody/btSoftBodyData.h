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

#ifndef BT_SOFTBODY_FLOAT_DATA
#define BT_SOFTBODY_FLOAT_DATA

#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"

struct SoftBodyMaterialData
{
	float m_linearStiffness;
	float m_angularStiffness;
	float m_volumeStiffness;
	int m_flags;
};

struct SoftBodyNodeData
{
	SoftBodyMaterialData *m_material;
	btVector3FloatData m_position;
	btVector3FloatData m_previousPosition;
	btVector3FloatData m_velocity;
	btVector3FloatData m_accumulatedForce;
	btVector3FloatData m_normal;
	float m_inverseMass;
	float m_area;
	int m_attach;
	int m_pad;
};

struct SoftBodyLinkData
{
	SoftBodyMaterialData *m_material;
	int m_nodeIndices[2];  // Node pointers
	float m_restLength;    // Rest length
	int m_bbending;        // Bending link
};

struct SoftBodyFaceData
{
	btVector3FloatData m_normal;  // Normal
	SoftBodyMaterialData *m_material;
	int m_nodeIndices[3];  // Node pointers
	float m_restArea;      // Rest area
};

struct SoftBodyTetraData
{
	btVector3FloatData m_c0[4];  // gradients
	SoftBodyMaterialData *m_material;
	int m_nodeIndices[4];  // Node pointers
	float m_restVolume;    // Rest volume
	float m_c1;            // (4*kVST)/(im0+im1+im2+im3)
	float m_c2;            // m_c1/sum(|g0..3|^2)
	int m_pad;
};

struct SoftRigidAnchorData
{
	btMatrix3x3FloatData m_c0;        // Impulse matrix
	btVector3FloatData m_c1;          // Relative anchor
	btVector3FloatData m_localFrame;  // Anchor position in body space
	btRigidBodyData *m_rigidBody;
	int m_nodeIndex;  // Node pointer
	float m_c2;       // ima*dt
};

struct SoftBodyConfigData
{
	int m_aeroModel;                         // Aerodynamic model (default: V_Point)
	float m_baumgarte;                       // Velocities correction factor (Baumgarte)
	float m_damping;                         // Damping coefficient [0,1]
	float m_drag;                            // Drag coefficient [0,+inf]
	float m_lift;                            // Lift coefficient [0,+inf]
	float m_pressure;                        // Pressure coefficient [-inf,+inf]
	float m_volume;                          // Volume conversation coefficient [0,+inf]
	float m_dynamicFriction;                 // Dynamic friction coefficient [0,1]
	float m_poseMatch;                       // Pose matching coefficient [0,1]
	float m_rigidContactHardness;            // Rigid contacts hardness [0,1]
	float m_kineticContactHardness;          // Kinetic contacts hardness [0,1]
	float m_softContactHardness;             // Soft contacts hardness [0,1]
	float m_anchorHardness;                  // Anchors hardness [0,1]
	float m_softRigidClusterHardness;        // Soft vs rigid hardness [0,1] (cluster only)
	float m_softKineticClusterHardness;      // Soft vs kinetic hardness [0,1] (cluster only)
	float m_softSoftClusterHardness;         // Soft vs soft hardness [0,1] (cluster only)
	float m_softRigidClusterImpulseSplit;    // Soft vs rigid impulse split [0,1] (cluster only)
	float m_softKineticClusterImpulseSplit;  // Soft vs rigid impulse split [0,1] (cluster only)
	float m_softSoftClusterImpulseSplit;     // Soft vs rigid impulse split [0,1] (cluster only)
	float m_maxVolume;                       // Maximum volume ratio for pose
	float m_timeScale;                       // Time scale
	int m_velocityIterations;                // Velocities solver iterations
	int m_positionIterations;                // Positions solver iterations
	int m_driftIterations;                   // Drift solver iterations
	int m_clusterIterations;                 // Cluster solver iterations
	int m_collisionFlags;                    // Collisions flags
};

struct SoftBodyPoseData
{
	btMatrix3x3FloatData m_rot;    // Rotation
	btMatrix3x3FloatData m_scale;  // Scale
	btMatrix3x3FloatData m_aqq;    // Base scaling
	btVector3FloatData m_com;      // COM

	btVector3FloatData *m_positions;  // Reference positions
	float *m_weights;                 // Weights
	int m_numPositions;
	int m_numWeigts;

	int m_bvolume;       // Is valid
	int m_bframe;        // Is frame
	float m_restVolume;  // Rest volume
	int m_pad;
};

struct SoftBodyClusterData
{
	btTransformFloatData m_framexform;
	btMatrix3x3FloatData m_locii;
	btMatrix3x3FloatData m_invwi;
	btVector3FloatData m_com;
	btVector3FloatData m_vimpulses[2];
	btVector3FloatData m_dimpulses[2];
	btVector3FloatData m_lv;
	btVector3FloatData m_av;

	btVector3FloatData *m_framerefs;
	int *m_nodeIndices;
	float *m_masses;

	int m_numFrameRefs;
	int m_numNodes;
	int m_numMasses;

	float m_idmass;
	float m_imass;
	int m_nvimpulses;
	int m_ndimpulses;
	float m_ndamping;
	float m_ldamping;
	float m_adamping;
	float m_matching;
	float m_maxSelfCollisionImpulse;
	float m_selfCollisionImpulseFactor;
	int m_containsAnchor;
	int m_collide;
	int m_clusterIndex;
};

enum btSoftJointBodyType
{
	BT_JOINT_SOFT_BODY_CLUSTER = 1,
	BT_JOINT_RIGID_BODY,
	BT_JOINT_COLLISION_OBJECT
};

struct btSoftBodyJointData
{
	void *m_bodyA;
	void *m_bodyB;
	btVector3FloatData m_refs[2];
	float m_cfm;
	float m_erp;
	float m_split;
	int m_delete;
	btVector3FloatData m_relPosition[2];  //linear
	int m_bodyAtype;
	int m_bodyBtype;
	int m_jointType;
	int m_pad;
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btSoftBodyFloatData
{
	btCollisionObjectFloatData m_collisionObjectData;

	SoftBodyPoseData *m_pose;
	SoftBodyMaterialData **m_materials;
	SoftBodyNodeData *m_nodes;
	SoftBodyLinkData *m_links;
	SoftBodyFaceData *m_faces;
	SoftBodyTetraData *m_tetrahedra;
	SoftRigidAnchorData *m_anchors;
	SoftBodyClusterData *m_clusters;
	btSoftBodyJointData *m_joints;

	int m_numMaterials;
	int m_numNodes;
	int m_numLinks;
	int m_numFaces;
	int m_numTetrahedra;
	int m_numAnchors;
	int m_numClusters;
	int m_numJoints;
	SoftBodyConfigData m_config;
};

#endif  //BT_SOFTBODY_FLOAT_DATA

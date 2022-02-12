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
///btSoftBody implementation by Nathanael Presson

#ifndef _BT_SOFT_BODY_H
#define _BT_SOFT_BODY_H

#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btTransform.h"
#include "LinearMath/btIDebugDraw.h"
#include "LinearMath/btVector3.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"

#include "BulletCollision/CollisionShapes/btConcaveShape.h"
#include "BulletCollision/CollisionDispatch/btCollisionCreateFunc.h"
#include "btSparseSDF.h"
#include "BulletCollision/BroadphaseCollision/btDbvt.h"
#include "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h"
#include "BulletDynamics/Featherstone/btMultiBodyConstraint.h"
//#ifdef BT_USE_DOUBLE_PRECISION
//#define btRigidBodyData	btRigidBodyDoubleData
//#define btRigidBodyDataName	"btRigidBodyDoubleData"
//#else
#define btSoftBodyData btSoftBodyFloatData
#define btSoftBodyDataName "btSoftBodyFloatData"
static const btScalar OVERLAP_REDUCTION_FACTOR = 0.1;
static unsigned long seed = 243703;
//#endif //BT_USE_DOUBLE_PRECISION

class btBroadphaseInterface;
class btDispatcher;
class btSoftBodySolver;

/* btSoftBodyWorldInfo	*/
struct btSoftBodyWorldInfo
{
	btScalar air_density;
	btScalar water_density;
	btScalar water_offset;
	btScalar m_maxDisplacement;
	btVector3 water_normal;
	btBroadphaseInterface* m_broadphase;
	btDispatcher* m_dispatcher;
	btVector3 m_gravity;
	btSparseSdf<3> m_sparsesdf;

	btSoftBodyWorldInfo()
		: air_density((btScalar)1.2),
		  water_density(0),
		  water_offset(0),
		  m_maxDisplacement(1000.f),  //avoid soft body from 'exploding' so use some upper threshold of maximum motion that a node can travel per frame
		  water_normal(0, 0, 0),
		  m_broadphase(0),
		  m_dispatcher(0),
		  m_gravity(0, -10, 0)
	{
	}
};

///The btSoftBody is an class to simulate cloth and volumetric soft bodies.
///There is two-way interaction between btSoftBody and btRigidBody/btCollisionObject.
class btSoftBody : public btCollisionObject
{
public:
	btAlignedObjectArray<const class btCollisionObject*> m_collisionDisabledObjects;

	// The solver object that handles this soft body
	btSoftBodySolver* m_softBodySolver;

	//
	// Enumerations
	//

	///eAeroModel
	struct eAeroModel
	{
		enum _
		{
			V_Point,             ///Vertex normals are oriented toward velocity
			V_TwoSided,          ///Vertex normals are flipped to match velocity
			V_TwoSidedLiftDrag,  ///Vertex normals are flipped to match velocity and lift and drag forces are applied
			V_OneSided,          ///Vertex normals are taken as it is
			F_TwoSided,          ///Face normals are flipped to match velocity
			F_TwoSidedLiftDrag,  ///Face normals are flipped to match velocity and lift and drag forces are applied
			F_OneSided,          ///Face normals are taken as it is
			END
		};
	};

	///eVSolver : velocities solvers
	struct eVSolver
	{
		enum _
		{
			Linear,  ///Linear solver
			END
		};
	};

	///ePSolver : positions solvers
	struct ePSolver
	{
		enum _
		{
			Linear,     ///Linear solver
			Anchors,    ///Anchor solver
			RContacts,  ///Rigid contacts solver
			SContacts,  ///Soft contacts solver
			END
		};
	};

	///eSolverPresets
	struct eSolverPresets
	{
		enum _
		{
			Positions,
			Velocities,
			Default = Positions,
			END
		};
	};

	///eFeature
	struct eFeature
	{
		enum _
		{
			None,
			Node,
			Link,
			Face,
			Tetra,
			END
		};
	};

	typedef btAlignedObjectArray<eVSolver::_> tVSolverArray;
	typedef btAlignedObjectArray<ePSolver::_> tPSolverArray;

	//
	// Flags
	//

	///fCollision
	struct fCollision
	{
		enum _
		{
			RVSmask = 0x000f,  ///Rigid versus soft mask
			SDF_RS = 0x0001,   ///SDF based rigid vs soft
			CL_RS = 0x0002,    ///Cluster vs convex rigid vs soft
			SDF_RD = 0x0004,   ///rigid vs deformable

			SVSmask = 0x00f0,  ///Rigid versus soft mask
			VF_SS = 0x0010,    ///Vertex vs face soft vs soft handling
			CL_SS = 0x0020,    ///Cluster vs cluster soft vs soft handling
			CL_SELF = 0x0040,  ///Cluster soft body self collision
			VF_DD = 0x0080,    ///Vertex vs face soft vs soft handling

			RVDFmask = 0x0f00,  /// Rigid versus deformable face mask
			SDF_RDF = 0x0100,   /// GJK based Rigid vs. deformable face
			SDF_MDF = 0x0200,   /// GJK based Multibody vs. deformable face
			SDF_RDN = 0x0400,   /// SDF based Rigid vs. deformable node
			/* presets	*/
			Default = SDF_RS,
			END
		};
	};

	///fMaterial
	struct fMaterial
	{
		enum _
		{
			DebugDraw = 0x0001,  /// Enable debug draw
			/* presets	*/
			Default = DebugDraw,
			END
		};
	};

	//
	// API Types
	//

	/* sRayCast		*/
	struct sRayCast
	{
		btSoftBody* body;     /// soft body
		eFeature::_ feature;  /// feature type
		int index;            /// feature index
		btScalar fraction;    /// time of impact fraction (rayorg+(rayto-rayfrom)*fraction)
	};

	/* ImplicitFn	*/
	struct ImplicitFn
	{
		virtual ~ImplicitFn() {}
		virtual btScalar Eval(const btVector3& x) = 0;
	};

	//
	// Internal types
	//

	typedef btAlignedObjectArray<btScalar> tScalarArray;
	typedef btAlignedObjectArray<btVector3> tVector3Array;

	/* sCti is Softbody contact info	*/
	struct sCti
	{
		const btCollisionObject* m_colObj; /* Rigid body			*/
		btVector3 m_normal;                /* Outward normal		*/
		btScalar m_offset;                 /* Offset from origin	*/
		btVector3 m_bary;                  /* Barycentric weights for faces */
	};

	/* sMedium		*/
	struct sMedium
	{
		btVector3 m_velocity; /* Velocity				*/
		btScalar m_pressure;  /* Pressure				*/
		btScalar m_density;   /* Density				*/
	};

	/* Base type	*/
	struct Element
	{
		void* m_tag;  // User data
		Element() : m_tag(0) {}
	};
	/* Material		*/
	struct Material : Element
	{
		btScalar m_kLST;  // Linear stiffness coefficient [0,1]
		btScalar m_kAST;  // Area/Angular stiffness coefficient [0,1]
		btScalar m_kVST;  // Volume stiffness coefficient [0,1]
		int m_flags;      // Flags
	};

	/* Feature		*/
	struct Feature : Element
	{
		Material* m_material;  // Material
	};
	/* Node			*/
	struct RenderNode
	{
		btVector3 m_x;
		btVector3 m_uv1;
		btVector3 m_normal;
	};
	struct Node : Feature
	{
		btVector3 m_x;       // Position
		btVector3 m_q;       // Previous step position/Test position
		btVector3 m_v;       // Velocity
		btVector3 m_vn;      // Previous step velocity
		btVector3 m_f;       // Force accumulator
		btVector3 m_n;       // Normal
		btScalar m_im;       // 1/mass
		btScalar m_area;     // Area
		btDbvtNode* m_leaf;  // Leaf data
		int m_constrained;   // depth of penetration
		int m_battach : 1;   // Attached
		int index;
		btVector3 m_splitv;               // velocity associated with split impulse
		btMatrix3x3 m_effectiveMass;      // effective mass in contact
		btMatrix3x3 m_effectiveMass_inv;  // inverse of effective mass
	};
	/* Link			*/
	ATTRIBUTE_ALIGNED16(struct)
	Link : Feature
	{
		btVector3 m_c3;      // gradient
		Node* m_n[2];        // Node pointers
		btScalar m_rl;       // Rest length
		int m_bbending : 1;  // Bending link
		btScalar m_c0;       // (ima+imb)*kLST
		btScalar m_c1;       // rl^2
		btScalar m_c2;       // |gradient|^2/c0

		BT_DECLARE_ALIGNED_ALLOCATOR();
	};
	struct RenderFace
	{
		RenderNode* m_n[3];          // Node pointers
	};

	/* Face			*/
	struct Face : Feature
	{
		Node* m_n[3];          // Node pointers
		btVector3 m_normal;    // Normal
		btScalar m_ra;         // Rest area
		btDbvtNode* m_leaf;    // Leaf data
		btVector4 m_pcontact;  // barycentric weights of the persistent contact
		btVector3 m_n0, m_n1, m_vn;
		int m_index;
	};
	/* Tetra		*/
	struct Tetra : Feature
	{
		Node* m_n[4];              // Node pointers
		btScalar m_rv;             // Rest volume
		btDbvtNode* m_leaf;        // Leaf data
		btVector3 m_c0[4];         // gradients
		btScalar m_c1;             // (4*kVST)/(im0+im1+im2+im3)
		btScalar m_c2;             // m_c1/sum(|g0..3|^2)
		btMatrix3x3 m_Dm_inverse;  // rest Dm^-1
		btMatrix3x3 m_F;
		btScalar m_element_measure;
		btVector4 m_P_inv[3];  // first three columns of P_inv matrix
	};

	/*  TetraScratch  */
	struct TetraScratch
	{
		btMatrix3x3 m_F;           // deformation gradient F
		btScalar m_trace;          // trace of F^T * F
		btScalar m_J;              // det(F)
		btMatrix3x3 m_cofF;        // cofactor of F
		btMatrix3x3 m_corotation;  // corotatio of the tetra
	};

	/* RContact		*/
	struct RContact
	{
		sCti m_cti;        // Contact infos
		Node* m_node;      // Owner node
		btMatrix3x3 m_c0;  // Impulse matrix
		btVector3 m_c1;    // Relative anchor
		btScalar m_c2;     // ima*dt
		btScalar m_c3;     // Friction
		btScalar m_c4;     // Hardness

		// jacobians and unit impulse responses for multibody
		btMultiBodyJacobianData jacobianData_normal;
		btMultiBodyJacobianData jacobianData_t1;
		btMultiBodyJacobianData jacobianData_t2;
		btVector3 t1;
		btVector3 t2;
	};

	class DeformableRigidContact
	{
	public:
		sCti m_cti;        // Contact infos
		btMatrix3x3 m_c0;  // Impulse matrix
		btVector3 m_c1;    // Relative anchor
		btScalar m_c2;     // inverse mass of node/face
		btScalar m_c3;     // Friction
		btScalar m_c4;     // Hardness
		btMatrix3x3 m_c5;  // inverse effective mass

		// jacobians and unit impulse responses for multibody
		btMultiBodyJacobianData jacobianData_normal;
		btMultiBodyJacobianData jacobianData_t1;
		btMultiBodyJacobianData jacobianData_t2;
		btVector3 t1;
		btVector3 t2;
	};

	class DeformableNodeRigidContact : public DeformableRigidContact
	{
	public:
		Node* m_node;  // Owner node
	};

	class DeformableNodeRigidAnchor : public DeformableNodeRigidContact
	{
	public:
		btVector3 m_local;  // Anchor position in body space
	};

	class DeformableFaceRigidContact : public DeformableRigidContact
	{
	public:
		Face* m_face;              // Owner face
		btVector3 m_contactPoint;  // Contact point
		btVector3 m_bary;          // Barycentric weights
		btVector3 m_weights;       // v_contactPoint * m_weights[i] = m_face->m_node[i]->m_v;
	};

	struct DeformableFaceNodeContact
	{
		Node* m_node;         // Node
		Face* m_face;         // Face
		btVector3 m_bary;     // Barycentric weights
		btVector3 m_weights;  // v_contactPoint * m_weights[i] = m_face->m_node[i]->m_v;
		btVector3 m_normal;   // Normal
		btScalar m_margin;    // Margin
		btScalar m_friction;  // Friction
		btScalar m_imf;       // inverse mass of the face at contact point
		btScalar m_c0;        // scale of the impulse matrix;
	};

	/* SContact		*/
	struct SContact
	{
		Node* m_node;         // Node
		Face* m_face;         // Face
		btVector3 m_weights;  // Weigths
		btVector3 m_normal;   // Normal
		btScalar m_margin;    // Margin
		btScalar m_friction;  // Friction
		btScalar m_cfm[2];    // Constraint force mixing
	};
	/* Anchor		*/
	struct Anchor
	{
		Node* m_node;         // Node pointer
		btVector3 m_local;    // Anchor position in body space
		btRigidBody* m_body;  // Body
		btScalar m_influence;
		btMatrix3x3 m_c0;  // Impulse matrix
		btVector3 m_c1;    // Relative anchor
		btScalar m_c2;     // ima*dt
	};
	/* Note			*/
	struct Note : Element
	{
		const char* m_text;    // Text
		btVector3 m_offset;    // Offset
		int m_rank;            // Rank
		Node* m_nodes[4];      // Nodes
		btScalar m_coords[4];  // Coordinates
	};
	/* Pose			*/
	struct Pose
	{
		bool m_bvolume;       // Is valid
		bool m_bframe;        // Is frame
		btScalar m_volume;    // Rest volume
		tVector3Array m_pos;  // Reference positions
		tScalarArray m_wgh;   // Weights
		btVector3 m_com;      // COM
		btMatrix3x3 m_rot;    // Rotation
		btMatrix3x3 m_scl;    // Scale
		btMatrix3x3 m_aqq;    // Base scaling
	};
	/* Cluster		*/
	struct Cluster
	{
		tScalarArray m_masses;
		btAlignedObjectArray<Node*> m_nodes;
		tVector3Array m_framerefs;
		btTransform m_framexform;
		btScalar m_idmass;
		btScalar m_imass;
		btMatrix3x3 m_locii;
		btMatrix3x3 m_invwi;
		btVector3 m_com;
		btVector3 m_vimpulses[2];
		btVector3 m_dimpulses[2];
		int m_nvimpulses;
		int m_ndimpulses;
		btVector3 m_lv;
		btVector3 m_av;
		btDbvtNode* m_leaf;
		btScalar m_ndamping; /* Node damping		*/
		btScalar m_ldamping; /* Linear damping	*/
		btScalar m_adamping; /* Angular damping	*/
		btScalar m_matching;
		btScalar m_maxSelfCollisionImpulse;
		btScalar m_selfCollisionImpulseFactor;
		bool m_containsAnchor;
		bool m_collide;
		int m_clusterIndex;
		Cluster() : m_leaf(0), m_ndamping(0), m_ldamping(0), m_adamping(0), m_matching(0), m_maxSelfCollisionImpulse(100.f), m_selfCollisionImpulseFactor(0.01f), m_containsAnchor(false)
		{
		}
	};
	/* Impulse		*/
	struct Impulse
	{
		btVector3 m_velocity;
		btVector3 m_drift;
		int m_asVelocity : 1;
		int m_asDrift : 1;
		Impulse() : m_velocity(0, 0, 0), m_drift(0, 0, 0), m_asVelocity(0), m_asDrift(0) {}
		Impulse operator-() const
		{
			Impulse i = *this;
			i.m_velocity = -i.m_velocity;
			i.m_drift = -i.m_drift;
			return (i);
		}
		Impulse operator*(btScalar x) const
		{
			Impulse i = *this;
			i.m_velocity *= x;
			i.m_drift *= x;
			return (i);
		}
	};
	/* Body			*/
	struct Body
	{
		Cluster* m_soft;
		btRigidBody* m_rigid;
		const btCollisionObject* m_collisionObject;

		Body() : m_soft(0), m_rigid(0), m_collisionObject(0) {}
		Body(Cluster* p) : m_soft(p), m_rigid(0), m_collisionObject(0) {}
		Body(const btCollisionObject* colObj) : m_soft(0), m_collisionObject(colObj)
		{
			m_rigid = (btRigidBody*)btRigidBody::upcast(m_collisionObject);
		}

		void activate() const
		{
			if (m_rigid)
				m_rigid->activate();
			if (m_collisionObject)
				m_collisionObject->activate();
		}
		const btMatrix3x3& invWorldInertia() const
		{
			static const btMatrix3x3 iwi(0, 0, 0, 0, 0, 0, 0, 0, 0);
			if (m_rigid) return (m_rigid->getInvInertiaTensorWorld());
			if (m_soft) return (m_soft->m_invwi);
			return (iwi);
		}
		btScalar invMass() const
		{
			if (m_rigid) return (m_rigid->getInvMass());
			if (m_soft) return (m_soft->m_imass);
			return (0);
		}
		const btTransform& xform() const
		{
			static const btTransform identity = btTransform::getIdentity();
			if (m_collisionObject) return (m_collisionObject->getWorldTransform());
			if (m_soft) return (m_soft->m_framexform);
			return (identity);
		}
		btVector3 linearVelocity() const
		{
			if (m_rigid) return (m_rigid->getLinearVelocity());
			if (m_soft) return (m_soft->m_lv);
			return (btVector3(0, 0, 0));
		}
		btVector3 angularVelocity(const btVector3& rpos) const
		{
			if (m_rigid) return (btCross(m_rigid->getAngularVelocity(), rpos));
			if (m_soft) return (btCross(m_soft->m_av, rpos));
			return (btVector3(0, 0, 0));
		}
		btVector3 angularVelocity() const
		{
			if (m_rigid) return (m_rigid->getAngularVelocity());
			if (m_soft) return (m_soft->m_av);
			return (btVector3(0, 0, 0));
		}
		btVector3 velocity(const btVector3& rpos) const
		{
			return (linearVelocity() + angularVelocity(rpos));
		}
		void applyVImpulse(const btVector3& impulse, const btVector3& rpos) const
		{
			if (m_rigid) m_rigid->applyImpulse(impulse, rpos);
			if (m_soft) btSoftBody::clusterVImpulse(m_soft, rpos, impulse);
		}
		void applyDImpulse(const btVector3& impulse, const btVector3& rpos) const
		{
			if (m_rigid) m_rigid->applyImpulse(impulse, rpos);
			if (m_soft) btSoftBody::clusterDImpulse(m_soft, rpos, impulse);
		}
		void applyImpulse(const Impulse& impulse, const btVector3& rpos) const
		{
			if (impulse.m_asVelocity)
			{
				//				printf("impulse.m_velocity = %f,%f,%f\n",impulse.m_velocity.getX(),impulse.m_velocity.getY(),impulse.m_velocity.getZ());
				applyVImpulse(impulse.m_velocity, rpos);
			}
			if (impulse.m_asDrift)
			{
				//				printf("impulse.m_drift = %f,%f,%f\n",impulse.m_drift.getX(),impulse.m_drift.getY(),impulse.m_drift.getZ());
				applyDImpulse(impulse.m_drift, rpos);
			}
		}
		void applyVAImpulse(const btVector3& impulse) const
		{
			if (m_rigid) m_rigid->applyTorqueImpulse(impulse);
			if (m_soft) btSoftBody::clusterVAImpulse(m_soft, impulse);
		}
		void applyDAImpulse(const btVector3& impulse) const
		{
			if (m_rigid) m_rigid->applyTorqueImpulse(impulse);
			if (m_soft) btSoftBody::clusterDAImpulse(m_soft, impulse);
		}
		void applyAImpulse(const Impulse& impulse) const
		{
			if (impulse.m_asVelocity) applyVAImpulse(impulse.m_velocity);
			if (impulse.m_asDrift) applyDAImpulse(impulse.m_drift);
		}
		void applyDCImpulse(const btVector3& impulse) const
		{
			if (m_rigid) m_rigid->applyCentralImpulse(impulse);
			if (m_soft) btSoftBody::clusterDCImpulse(m_soft, impulse);
		}
	};
	/* Joint		*/
	struct Joint
	{
		struct eType
		{
			enum _
			{
				Linear = 0,
				Angular,
				Contact
			};
		};
		struct Specs
		{
			Specs() : erp(1), cfm(1), split(1) {}
			btScalar erp;
			btScalar cfm;
			btScalar split;
		};
		Body m_bodies[2];
		btVector3 m_refs[2];
		btScalar m_cfm;
		btScalar m_erp;
		btScalar m_split;
		btVector3 m_drift;
		btVector3 m_sdrift;
		btMatrix3x3 m_massmatrix;
		bool m_delete;
		virtual ~Joint() {}
		Joint() : m_delete(false) {}
		virtual void Prepare(btScalar dt, int iterations);
		virtual void Solve(btScalar dt, btScalar sor) = 0;
		virtual void Terminate(btScalar dt) = 0;
		virtual eType::_ Type() const = 0;
	};
	/* LJoint		*/
	struct LJoint : Joint
	{
		struct Specs : Joint::Specs
		{
			btVector3 position;
		};
		btVector3 m_rpos[2];
		void Prepare(btScalar dt, int iterations);
		void Solve(btScalar dt, btScalar sor);
		void Terminate(btScalar dt);
		eType::_ Type() const { return (eType::Linear); }
	};
	/* AJoint		*/
	struct AJoint : Joint
	{
		struct IControl
		{
			virtual ~IControl() {}
			virtual void Prepare(AJoint*) {}
			virtual btScalar Speed(AJoint*, btScalar current) { return (current); }
			static IControl* Default()
			{
				static IControl def;
				return (&def);
			}
		};
		struct Specs : Joint::Specs
		{
			Specs() : icontrol(IControl::Default()) {}
			btVector3 axis;
			IControl* icontrol;
		};
		btVector3 m_axis[2];
		IControl* m_icontrol;
		void Prepare(btScalar dt, int iterations);
		void Solve(btScalar dt, btScalar sor);
		void Terminate(btScalar dt);
		eType::_ Type() const { return (eType::Angular); }
	};
	/* CJoint		*/
	struct CJoint : Joint
	{
		int m_life;
		int m_maxlife;
		btVector3 m_rpos[2];
		btVector3 m_normal;
		btScalar m_friction;
		void Prepare(btScalar dt, int iterations);
		void Solve(btScalar dt, btScalar sor);
		void Terminate(btScalar dt);
		eType::_ Type() const { return (eType::Contact); }
	};
	/* Config		*/
	struct Config
	{
		eAeroModel::_ aeromodel;    // Aerodynamic model (default: V_Point)
		btScalar kVCF;              // Velocities correction factor (Baumgarte)
		btScalar kDP;               // Damping coefficient [0,1]
		btScalar kDG;               // Drag coefficient [0,+inf]
		btScalar kLF;               // Lift coefficient [0,+inf]
		btScalar kPR;               // Pressure coefficient [-inf,+inf]
		btScalar kVC;               // Volume conversation coefficient [0,+inf]
		btScalar kDF;               // Dynamic friction coefficient [0,1]
		btScalar kMT;               // Pose matching coefficient [0,1]
		btScalar kCHR;              // Rigid contacts hardness [0,1]
		btScalar kKHR;              // Kinetic contacts hardness [0,1]
		btScalar kSHR;              // Soft contacts hardness [0,1]
		btScalar kAHR;              // Anchors hardness [0,1]
		btScalar kSRHR_CL;          // Soft vs rigid hardness [0,1] (cluster only)
		btScalar kSKHR_CL;          // Soft vs kinetic hardness [0,1] (cluster only)
		btScalar kSSHR_CL;          // Soft vs soft hardness [0,1] (cluster only)
		btScalar kSR_SPLT_CL;       // Soft vs rigid impulse split [0,1] (cluster only)
		btScalar kSK_SPLT_CL;       // Soft vs rigid impulse split [0,1] (cluster only)
		btScalar kSS_SPLT_CL;       // Soft vs rigid impulse split [0,1] (cluster only)
		btScalar maxvolume;         // Maximum volume ratio for pose
		btScalar timescale;         // Time scale
		int viterations;            // Velocities solver iterations
		int piterations;            // Positions solver iterations
		int diterations;            // Drift solver iterations
		int citerations;            // Cluster solver iterations
		int collisions;             // Collisions flags
		tVSolverArray m_vsequence;  // Velocity solvers sequence
		tPSolverArray m_psequence;  // Position solvers sequence
		tPSolverArray m_dsequence;  // Drift solvers sequence
		btScalar drag;              // deformable air drag
		btScalar m_maxStress;       // Maximum principle first Piola stress
	};
	/* SolverState	*/
	struct SolverState
	{
		//if you add new variables, always initialize them!
		SolverState()
			: sdt(0),
			  isdt(0),
			  velmrg(0),
			  radmrg(0),
			  updmrg(0)
		{
		}
		btScalar sdt;     // dt*timescale
		btScalar isdt;    // 1/sdt
		btScalar velmrg;  // velocity margin
		btScalar radmrg;  // radial margin
		btScalar updmrg;  // Update margin
	};
	/// RayFromToCaster takes a ray from, ray to (instead of direction!)
	struct RayFromToCaster : btDbvt::ICollide
	{
		btVector3 m_rayFrom;
		btVector3 m_rayTo;
		btVector3 m_rayNormalizedDirection;
		btScalar m_mint;
		Face* m_face;
		int m_tests;
		RayFromToCaster(const btVector3& rayFrom, const btVector3& rayTo, btScalar mxt);
		void Process(const btDbvtNode* leaf);

		static /*inline*/ btScalar rayFromToTriangle(const btVector3& rayFrom,
													 const btVector3& rayTo,
													 const btVector3& rayNormalizedDirection,
													 const btVector3& a,
													 const btVector3& b,
													 const btVector3& c,
													 btScalar maxt = SIMD_INFINITY);
	};

	//
	// Typedefs
	//

	typedef void (*psolver_t)(btSoftBody*, btScalar, btScalar);
	typedef void (*vsolver_t)(btSoftBody*, btScalar);
	typedef btAlignedObjectArray<Cluster*> tClusterArray;
	typedef btAlignedObjectArray<Note> tNoteArray;
	typedef btAlignedObjectArray<Node> tNodeArray;
	typedef btAlignedObjectArray< RenderNode> tRenderNodeArray;
	typedef btAlignedObjectArray<btDbvtNode*> tLeafArray;
	typedef btAlignedObjectArray<Link> tLinkArray;
	typedef btAlignedObjectArray<Face> tFaceArray;
	typedef btAlignedObjectArray<RenderFace> tRenderFaceArray;
	typedef btAlignedObjectArray<Tetra> tTetraArray;
	typedef btAlignedObjectArray<Anchor> tAnchorArray;
	typedef btAlignedObjectArray<RContact> tRContactArray;
	typedef btAlignedObjectArray<SContact> tSContactArray;
	typedef btAlignedObjectArray<Material*> tMaterialArray;
	typedef btAlignedObjectArray<Joint*> tJointArray;
	typedef btAlignedObjectArray<btSoftBody*> tSoftBodyArray;

	//
	// Fields
	//

	Config m_cfg;                      // Configuration
	SolverState m_sst;                 // Solver state
	Pose m_pose;                       // Pose
	void* m_tag;                       // User data
	btSoftBodyWorldInfo* m_worldInfo;  // World info
	tNoteArray m_notes;                // Notes
	tNodeArray m_nodes;                // Nodes
	tRenderNodeArray m_renderNodes;    // Render Nodes
	tLinkArray m_links;                // Links
	tFaceArray m_faces;                // Faces
	tRenderFaceArray m_renderFaces;          // Faces
	tTetraArray m_tetras;              // Tetras
	btAlignedObjectArray<TetraScratch> m_tetraScratches;
	btAlignedObjectArray<TetraScratch> m_tetraScratchesTn;
	tAnchorArray m_anchors;  // Anchors
	btAlignedObjectArray<DeformableNodeRigidAnchor> m_deformableAnchors;
	tRContactArray m_rcontacts;  // Rigid contacts
	btAlignedObjectArray<DeformableNodeRigidContact> m_nodeRigidContacts;
	btAlignedObjectArray<DeformableFaceNodeContact> m_faceNodeContacts;
	btAlignedObjectArray<DeformableFaceRigidContact> m_faceRigidContacts;
	tSContactArray m_scontacts;     // Soft contacts
	tJointArray m_joints;           // Joints
	tMaterialArray m_materials;     // Materials
	btScalar m_timeacc;             // Time accumulator
	btVector3 m_bounds[2];          // Spatial bounds
	bool m_bUpdateRtCst;            // Update runtime constants
	btDbvt m_ndbvt;                 // Nodes tree
	btDbvt m_fdbvt;                 // Faces tree
	btDbvntNode* m_fdbvnt;          // Faces tree with normals
	btDbvt m_cdbvt;                 // Clusters tree
	tClusterArray m_clusters;       // Clusters
	btScalar m_dampingCoefficient;  // Damping Coefficient
	btScalar m_sleepingThreshold;
	btScalar m_maxSpeedSquared;
	btAlignedObjectArray<btVector3> m_quads;  // quadrature points for collision detection
	btScalar m_repulsionStiffness;
	btScalar m_gravityFactor;
	bool m_cacheBarycenter;
	btAlignedObjectArray<btVector3> m_X;  // initial positions

	btAlignedObjectArray<btVector4> m_renderNodesInterpolationWeights;
	btAlignedObjectArray<btAlignedObjectArray<const btSoftBody::Node*> > m_renderNodesParents;
	btAlignedObjectArray<btScalar> m_z;  // vertical distance used in extrapolation
	bool m_useSelfCollision;
	bool m_softSoftCollision;

	btAlignedObjectArray<bool> m_clusterConnectivity;  //cluster connectivity, for self-collision

	btVector3 m_windVelocity;

	btScalar m_restLengthScale;

	//
	// Api
	//

	/* ctor																	*/
	btSoftBody(btSoftBodyWorldInfo* worldInfo, int node_count, const btVector3* x, const btScalar* m);

	/* ctor																	*/
	btSoftBody(btSoftBodyWorldInfo* worldInfo);

	void initDefaults();

	/* dtor																	*/
	virtual ~btSoftBody();
	/* Check for existing link												*/

	btAlignedObjectArray<int> m_userIndexMapping;

	btSoftBodyWorldInfo* getWorldInfo()
	{
		return m_worldInfo;
	}

	void setDampingCoefficient(btScalar damping_coeff)
	{
		m_dampingCoefficient = damping_coeff;
	}

	///@todo: avoid internal softbody shape hack and move collision code to collision library
	virtual void setCollisionShape(btCollisionShape* collisionShape)
	{
	}

	bool checkLink(int node0,
				   int node1) const;
	bool checkLink(const Node* node0,
				   const Node* node1) const;
	/* Check for existring face												*/
	bool checkFace(int node0,
				   int node1,
				   int node2) const;
	/* Append material														*/
	Material* appendMaterial();
	/* Append note															*/
	void appendNote(const char* text,
					const btVector3& o,
					const btVector4& c = btVector4(1, 0, 0, 0),
					Node* n0 = 0,
					Node* n1 = 0,
					Node* n2 = 0,
					Node* n3 = 0);
	void appendNote(const char* text,
					const btVector3& o,
					Node* feature);
	void appendNote(const char* text,
					const btVector3& o,
					Link* feature);
	void appendNote(const char* text,
					const btVector3& o,
					Face* feature);
	/* Append node															*/
	void appendNode(const btVector3& x, btScalar m);
	/* Append link															*/
	void appendLink(int model = -1, Material* mat = 0);
	void appendLink(int node0,
					int node1,
					Material* mat = 0,
					bool bcheckexist = false);
	void appendLink(Node* node0,
					Node* node1,
					Material* mat = 0,
					bool bcheckexist = false);
	/* Append face															*/
	void appendFace(int model = -1, Material* mat = 0);
	void appendFace(int node0,
					int node1,
					int node2,
					Material* mat = 0);
	void appendTetra(int model, Material* mat);
	//
	void appendTetra(int node0,
					 int node1,
					 int node2,
					 int node3,
					 Material* mat = 0);

	/* Append anchor														*/
	void appendDeformableAnchor(int node, btRigidBody* body);
	void appendDeformableAnchor(int node, btMultiBodyLinkCollider* link);
	void appendAnchor(int node,
					  btRigidBody* body, bool disableCollisionBetweenLinkedBodies = false, btScalar influence = 1);
	void appendAnchor(int node, btRigidBody* body, const btVector3& localPivot, bool disableCollisionBetweenLinkedBodies = false, btScalar influence = 1);
	void removeAnchor(int node);
	/* Append linear joint													*/
	void appendLinearJoint(const LJoint::Specs& specs, Cluster* body0, Body body1);
	void appendLinearJoint(const LJoint::Specs& specs, Body body = Body());
	void appendLinearJoint(const LJoint::Specs& specs, btSoftBody* body);
	/* Append linear joint													*/
	void appendAngularJoint(const AJoint::Specs& specs, Cluster* body0, Body body1);
	void appendAngularJoint(const AJoint::Specs& specs, Body body = Body());
	void appendAngularJoint(const AJoint::Specs& specs, btSoftBody* body);
	/* Add force (or gravity) to the entire body							*/
	void addForce(const btVector3& force);
	/* Add force (or gravity) to a node of the body							*/
	void addForce(const btVector3& force,
				  int node);
	/* Add aero force to a node of the body */
	void addAeroForceToNode(const btVector3& windVelocity, int nodeIndex);

	/* Add aero force to a face of the body */
	void addAeroForceToFace(const btVector3& windVelocity, int faceIndex);

	/* Add velocity to the entire body										*/
	void addVelocity(const btVector3& velocity);

	/* Set velocity for the entire body										*/
	void setVelocity(const btVector3& velocity);

	/* Add velocity to a node of the body									*/
	void addVelocity(const btVector3& velocity,
					 int node);
	/* Set mass																*/
	void setMass(int node,
				 btScalar mass);
	/* Get mass																*/
	btScalar getMass(int node) const;
	/* Get total mass														*/
	btScalar getTotalMass() const;
	/* Set total mass (weighted by previous masses)							*/
	void setTotalMass(btScalar mass,
					  bool fromfaces = false);
	/* Set total density													*/
	void setTotalDensity(btScalar density);
	/* Set volume mass (using tetrahedrons)									*/
	void setVolumeMass(btScalar mass);
	/* Set volume density (using tetrahedrons)								*/
	void setVolumeDensity(btScalar density);
	/* Get the linear velocity of the center of mass                        */
	btVector3 getLinearVelocity();
	/* Set the linear velocity of the center of mass                        */
	void setLinearVelocity(const btVector3& linVel);
	/* Set the angular velocity of the center of mass                       */
	void setAngularVelocity(const btVector3& angVel);
	/* Get best fit rigid transform                                         */
	btTransform getRigidTransform();
	/* Transform to given pose                                              */
	void transformTo(const btTransform& trs);
	/* Transform															*/
	void transform(const btTransform& trs);
	/* Translate															*/
	void translate(const btVector3& trs);
	/* Rotate															*/
	void rotate(const btQuaternion& rot);
	/* Scale																*/
	void scale(const btVector3& scl);
	/* Get link resting lengths scale										*/
	btScalar getRestLengthScale();
	/* Scale resting length of all springs									*/
	void setRestLengthScale(btScalar restLength);
	/* Set current state as pose											*/
	void setPose(bool bvolume,
				 bool bframe);
	/* Set current link lengths as resting lengths							*/
	void resetLinkRestLengths();
	/* Return the volume													*/
	btScalar getVolume() const;
	/* Cluster count														*/
	btVector3 getCenterOfMass() const
	{
		btVector3 com(0, 0, 0);
		for (int i = 0; i < m_nodes.size(); i++)
		{
			com += (m_nodes[i].m_x * this->getMass(i));
		}
		com /= this->getTotalMass();
		return com;
	}
	int clusterCount() const;
	/* Cluster center of mass												*/
	static btVector3 clusterCom(const Cluster* cluster);
	btVector3 clusterCom(int cluster) const;
	/* Cluster velocity at rpos												*/
	static btVector3 clusterVelocity(const Cluster* cluster, const btVector3& rpos);
	/* Cluster impulse														*/
	static void clusterVImpulse(Cluster* cluster, const btVector3& rpos, const btVector3& impulse);
	static void clusterDImpulse(Cluster* cluster, const btVector3& rpos, const btVector3& impulse);
	static void clusterImpulse(Cluster* cluster, const btVector3& rpos, const Impulse& impulse);
	static void clusterVAImpulse(Cluster* cluster, const btVector3& impulse);
	static void clusterDAImpulse(Cluster* cluster, const btVector3& impulse);
	static void clusterAImpulse(Cluster* cluster, const Impulse& impulse);
	static void clusterDCImpulse(Cluster* cluster, const btVector3& impulse);
	/* Generate bending constraints based on distance in the adjency graph	*/
	int generateBendingConstraints(int distance,
								   Material* mat = 0);
	/* Randomize constraints to reduce solver bias							*/
	void randomizeConstraints();
	/* Release clusters														*/
	void releaseCluster(int index);
	void releaseClusters();
	/* Generate clusters (K-mean)											*/
	///generateClusters with k=0 will create a convex cluster for each tetrahedron or triangle
	///otherwise an approximation will be used (better performance)
	int generateClusters(int k, int maxiterations = 8192);
	/* Refine																*/
	void refine(ImplicitFn* ifn, btScalar accurary, bool cut);
	/* CutLink																*/
	bool cutLink(int node0, int node1, btScalar position);
	bool cutLink(const Node* node0, const Node* node1, btScalar position);

	///Ray casting using rayFrom and rayTo in worldspace, (not direction!)
	bool rayTest(const btVector3& rayFrom,
				 const btVector3& rayTo,
				 sRayCast& results);
	bool rayFaceTest(const btVector3& rayFrom,
					 const btVector3& rayTo,
					 sRayCast& results);
	int rayFaceTest(const btVector3& rayFrom, const btVector3& rayTo,
					btScalar& mint, int& index) const;
	/* Solver presets														*/
	void setSolver(eSolverPresets::_ preset);
	/* predictMotion														*/
	void predictMotion(btScalar dt);
	/* solveConstraints														*/
	void solveConstraints();
	/* staticSolve															*/
	void staticSolve(int iterations);
	/* solveCommonConstraints												*/
	static void solveCommonConstraints(btSoftBody** bodies, int count, int iterations);
	/* solveClusters														*/
	static void solveClusters(const btAlignedObjectArray<btSoftBody*>& bodies);
	/* integrateMotion														*/
	void integrateMotion();
	/* defaultCollisionHandlers												*/
	void defaultCollisionHandler(const btCollisionObjectWrapper* pcoWrap);
	void defaultCollisionHandler(btSoftBody* psb);
	void setSelfCollision(bool useSelfCollision);
	bool useSelfCollision();
	void updateDeactivation(btScalar timeStep);
	void setZeroVelocity();
	bool wantsSleeping();

	//
	// Functionality to deal with new accelerated solvers.
	//

	/**
	 * Set a wind velocity for interaction with the air.
	 */
	void setWindVelocity(const btVector3& velocity);

	/**
	 * Return the wind velocity for interaction with the air.
	 */
	const btVector3& getWindVelocity();

	//
	// Set the solver that handles this soft body
	// Should not be allowed to get out of sync with reality
	// Currently called internally on addition to the world
	void setSoftBodySolver(btSoftBodySolver* softBodySolver)
	{
		m_softBodySolver = softBodySolver;
	}

	//
	// Return the solver that handles this soft body
	//
	btSoftBodySolver* getSoftBodySolver()
	{
		return m_softBodySolver;
	}

	//
	// Return the solver that handles this soft body
	//
	btSoftBodySolver* getSoftBodySolver() const
	{
		return m_softBodySolver;
	}

	//
	// Cast
	//

	static const btSoftBody* upcast(const btCollisionObject* colObj)
	{
		if (colObj->getInternalType() == CO_SOFT_BODY)
			return (const btSoftBody*)colObj;
		return 0;
	}
	static btSoftBody* upcast(btCollisionObject* colObj)
	{
		if (colObj->getInternalType() == CO_SOFT_BODY)
			return (btSoftBody*)colObj;
		return 0;
	}

	//
	// ::btCollisionObject
	//

	virtual void getAabb(btVector3& aabbMin, btVector3& aabbMax) const
	{
		aabbMin = m_bounds[0];
		aabbMax = m_bounds[1];
	}
	//
	// Private
	//
	void pointersToIndices();
	void indicesToPointers(const int* map = 0);

	int rayTest(const btVector3& rayFrom, const btVector3& rayTo,
				btScalar& mint, eFeature::_& feature, int& index, bool bcountonly) const;
	void initializeFaceTree();
	void rebuildNodeTree();
	btVector3 evaluateCom() const;
	bool checkDeformableContact(const btCollisionObjectWrapper* colObjWrap, const btVector3& x, btScalar margin, btSoftBody::sCti& cti, bool predict = false) const;
	bool checkDeformableFaceContact(const btCollisionObjectWrapper* colObjWrap, Face& f, btVector3& contact_point, btVector3& bary, btScalar margin, btSoftBody::sCti& cti, bool predict = false) const;
	bool checkContact(const btCollisionObjectWrapper* colObjWrap, const btVector3& x, btScalar margin, btSoftBody::sCti& cti) const;
	void updateNormals();
	void updateBounds();
	void updatePose();
	void updateConstants();
	void updateLinkConstants();
	void updateArea(bool averageArea = true);
	void initializeClusters();
	void updateClusters();
	void cleanupClusters();
	void prepareClusters(int iterations);
	void solveClusters(btScalar sor);
	void applyClusters(bool drift);
	void dampClusters();
	void setSpringStiffness(btScalar k);
	void setGravityFactor(btScalar gravFactor);
	void setCacheBarycenter(bool cacheBarycenter);
	void initializeDmInverse();
	void updateDeformation();
	void advanceDeformation();
	void applyForces();
	void setMaxStress(btScalar maxStress);
	void interpolateRenderMesh();
	void setCollisionQuadrature(int N);
	static void PSolve_Anchors(btSoftBody* psb, btScalar kst, btScalar ti);
	static void PSolve_RContacts(btSoftBody* psb, btScalar kst, btScalar ti);
	static void PSolve_SContacts(btSoftBody* psb, btScalar, btScalar ti);
	static void PSolve_Links(btSoftBody* psb, btScalar kst, btScalar ti);
	static void VSolve_Links(btSoftBody* psb, btScalar kst);
	static psolver_t getSolver(ePSolver::_ solver);
	static vsolver_t getSolver(eVSolver::_ solver);
	void geometricCollisionHandler(btSoftBody* psb);
#define SAFE_EPSILON SIMD_EPSILON * 100.0
	void updateNode(btDbvtNode* node, bool use_velocity, bool margin)
	{
		if (node->isleaf())
		{
			btSoftBody::Node* n = (btSoftBody::Node*)(node->data);
			ATTRIBUTE_ALIGNED16(btDbvtVolume)
			vol;
			btScalar pad = margin ? m_sst.radmrg : SAFE_EPSILON;  // use user defined margin or margin for floating point precision
			if (use_velocity)
			{
				btVector3 points[2] = {n->m_x, n->m_x + m_sst.sdt * n->m_v};
				vol = btDbvtVolume::FromPoints(points, 2);
				vol.Expand(btVector3(pad, pad, pad));
			}
			else
			{
				vol = btDbvtVolume::FromCR(n->m_x, pad);
			}
			node->volume = vol;
			return;
		}
		else
		{
			updateNode(node->childs[0], use_velocity, margin);
			updateNode(node->childs[1], use_velocity, margin);
			ATTRIBUTE_ALIGNED16(btDbvtVolume)
			vol;
			Merge(node->childs[0]->volume, node->childs[1]->volume, vol);
			node->volume = vol;
		}
	}

	void updateNodeTree(bool use_velocity, bool margin)
	{
		if (m_ndbvt.m_root)
			updateNode(m_ndbvt.m_root, use_velocity, margin);
	}

	template <class DBVTNODE>  // btDbvtNode or btDbvntNode
	void updateFace(DBVTNODE* node, bool use_velocity, bool margin)
	{
		if (node->isleaf())
		{
			btSoftBody::Face* f = (btSoftBody::Face*)(node->data);
			btScalar pad = margin ? m_sst.radmrg : SAFE_EPSILON;  // use user defined margin or margin for floating point precision
			ATTRIBUTE_ALIGNED16(btDbvtVolume)
			vol;
			if (use_velocity)
			{
				btVector3 points[6] = {f->m_n[0]->m_x, f->m_n[0]->m_x + m_sst.sdt * f->m_n[0]->m_v,
									   f->m_n[1]->m_x, f->m_n[1]->m_x + m_sst.sdt * f->m_n[1]->m_v,
									   f->m_n[2]->m_x, f->m_n[2]->m_x + m_sst.sdt * f->m_n[2]->m_v};
				vol = btDbvtVolume::FromPoints(points, 6);
			}
			else
			{
				btVector3 points[3] = {f->m_n[0]->m_x,
									   f->m_n[1]->m_x,
									   f->m_n[2]->m_x};
				vol = btDbvtVolume::FromPoints(points, 3);
			}
			vol.Expand(btVector3(pad, pad, pad));
			node->volume = vol;
			return;
		}
		else
		{
			updateFace(node->childs[0], use_velocity, margin);
			updateFace(node->childs[1], use_velocity, margin);
			ATTRIBUTE_ALIGNED16(btDbvtVolume)
			vol;
			Merge(node->childs[0]->volume, node->childs[1]->volume, vol);
			node->volume = vol;
		}
	}
	void updateFaceTree(bool use_velocity, bool margin)
	{
		if (m_fdbvt.m_root)
			updateFace(m_fdbvt.m_root, use_velocity, margin);
		if (m_fdbvnt)
			updateFace(m_fdbvnt, use_velocity, margin);
	}

	template <typename T>
	static inline T BaryEval(const T& a,
							 const T& b,
							 const T& c,
							 const btVector3& coord)
	{
		return (a * coord.x() + b * coord.y() + c * coord.z());
	}

	void applyRepulsionForce(btScalar timeStep, bool applySpringForce)
	{
		btAlignedObjectArray<int> indices;
		{
			// randomize the order of repulsive force
			indices.resize(m_faceNodeContacts.size());
			for (int i = 0; i < m_faceNodeContacts.size(); ++i)
				indices[i] = i;
#define NEXTRAND (seed = (1664525L * seed + 1013904223L) & 0xffffffff)
			int i, ni;

			for (i = 0, ni = indices.size(); i < ni; ++i)
			{
				btSwap(indices[i], indices[NEXTRAND % ni]);
			}
		}
		for (int k = 0; k < m_faceNodeContacts.size(); ++k)
		{
			int idx = indices[k];
			btSoftBody::DeformableFaceNodeContact& c = m_faceNodeContacts[idx];
			btSoftBody::Node* node = c.m_node;
			btSoftBody::Face* face = c.m_face;
			const btVector3& w = c.m_bary;
			const btVector3& n = c.m_normal;
			btVector3 l = node->m_x - BaryEval(face->m_n[0]->m_x, face->m_n[1]->m_x, face->m_n[2]->m_x, w);
			btScalar d = c.m_margin - n.dot(l);
			d = btMax(btScalar(0), d);

			const btVector3& va = node->m_v;
			btVector3 vb = BaryEval(face->m_n[0]->m_v, face->m_n[1]->m_v, face->m_n[2]->m_v, w);
			btVector3 vr = va - vb;
			const btScalar vn = btDot(vr, n);  // dn < 0 <==> opposing
			if (vn > OVERLAP_REDUCTION_FACTOR * d / timeStep)
				continue;
			btVector3 vt = vr - vn * n;
			btScalar I = 0;
			btScalar mass = node->m_im == 0 ? 0 : btScalar(1) / node->m_im;
			if (applySpringForce)
				I = -btMin(m_repulsionStiffness * timeStep * d, mass * (OVERLAP_REDUCTION_FACTOR * d / timeStep - vn));
			if (vn < 0)
				I += 0.5 * mass * vn;
			int face_penetration = 0, node_penetration = node->m_constrained;
			for (int i = 0; i < 3; ++i)
				face_penetration |= face->m_n[i]->m_constrained;
			btScalar I_tilde = 2.0 * I / (1.0 + w.length2());

			//             double the impulse if node or face is constrained.
			if (face_penetration > 0 || node_penetration > 0)
			{
				I_tilde *= 2.0;
			}
			if (face_penetration <= 0)
			{
				for (int j = 0; j < 3; ++j)
					face->m_n[j]->m_v += w[j] * n * I_tilde * node->m_im;
			}
			if (node_penetration <= 0)
			{
				node->m_v -= I_tilde * node->m_im * n;
			}

			// apply frictional impulse
			btScalar vt_norm = vt.safeNorm();
			if (vt_norm > SIMD_EPSILON)
			{
				btScalar delta_vn = -2 * I * node->m_im;
				btScalar mu = c.m_friction;
				btScalar vt_new = btMax(btScalar(1) - mu * delta_vn / (vt_norm + SIMD_EPSILON), btScalar(0)) * vt_norm;
				I = 0.5 * mass * (vt_norm - vt_new);
				vt.safeNormalize();
				I_tilde = 2.0 * I / (1.0 + w.length2());
				//                 double the impulse if node or face is constrained.
				if (face_penetration > 0 || node_penetration > 0)
					I_tilde *= 2.0;
				if (face_penetration <= 0)
				{
					for (int j = 0; j < 3; ++j)
						face->m_n[j]->m_v += w[j] * vt * I_tilde * (face->m_n[j])->m_im;
				}
				if (node_penetration <= 0)
				{
					node->m_v -= I_tilde * node->m_im * vt;
				}
			}
		}
	}
	virtual int calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual const char* serialize(void* dataBuffer, class btSerializer* serializer) const;
};

#endif  //_BT_SOFT_BODY_H

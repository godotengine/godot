// The structs and classes defined here provide a basic inverse fynamics implementation used
// by MultiBodyTree
// User interaction should be through MultiBodyTree

#ifndef MULTI_BODY_REFERENCE_IMPL_HPP_
#define MULTI_BODY_REFERENCE_IMPL_HPP_

#include "../IDConfig.hpp"
#include "../MultiBodyTree.hpp"

namespace btInverseDynamics
{
/// Structure for for rigid body mass properties, connectivity and kinematic state
/// all vectors and matrices are in body-fixed frame, if not indicated otherwise.
/// The body-fixed frame is located in the joint connecting the body to its parent.
struct RigidBody
{
	ID_DECLARE_ALIGNED_ALLOCATOR();
	// 1 Inertial properties
	/// Mass
	idScalar m_mass;
	/// Mass times center of gravity in body-fixed frame
	vec3 m_body_mass_com;
	/// Moment of inertia w.r.t. body-fixed frame
	mat33 m_body_I_body;

	// 2 dynamic properties
	/// Left-hand side of the body equation of motion, translational part
	vec3 m_eom_lhs_translational;
	/// Left-hand side of the body equation of motion, rotational part
	vec3 m_eom_lhs_rotational;
	/// Force acting at the joint when the body is cut from its parent;
	/// includes impressed joint force in J_JT direction,
	/// as well as constraint force,
	/// in body-fixed frame
	vec3 m_force_at_joint;
	/// Moment acting at the joint when the body is cut from its parent;
	/// includes impressed joint moment in J_JR direction, and constraint moment
	/// in body-fixed frame
	vec3 m_moment_at_joint;
	/// external (user provided) force acting at the body-fixed frame's origin, written in that
	/// frame
	vec3 m_body_force_user;
	/// external (user provided) moment acting at the body-fixed frame's origin, written in that
	/// frame
	vec3 m_body_moment_user;
	// 3 absolute kinematic properties
	/// Position of body-fixed frame relative to world frame
	/// this is currently only for debugging purposes
	vec3 m_body_pos;
	/// Absolute velocity of body-fixed frame
	vec3 m_body_vel;
	/// Absolute acceleration of body-fixed frame
	/// NOTE: if gravitational acceleration is not zero, this is the accelation PLUS gravitational
	/// acceleration!
	vec3 m_body_acc;
	/// Absolute angular velocity
	vec3 m_body_ang_vel;
	/// Absolute angular acceleration
	/// NOTE: if gravitational acceleration is not zero, this is the accelation PLUS gravitational
	/// acceleration!
	vec3 m_body_ang_acc;

	// 4 relative kinematic properties.
	// these are in the parent body frame
	/// Transform from world to body-fixed frame;
	/// this is currently only for debugging purposes
	mat33 m_body_T_world;
	/// Transform from parent to body-fixed frame
	mat33 m_body_T_parent;
	/// Vector from parent to child frame in parent frame
	vec3 m_parent_pos_parent_body;
	/// Relative angular velocity
	vec3 m_body_ang_vel_rel;
	/// Relative linear velocity
	vec3 m_parent_vel_rel;
	/// Relative angular acceleration
	vec3 m_body_ang_acc_rel;
	/// Relative linear acceleration
	vec3 m_parent_acc_rel;

	// 5 Data describing the joint type and geometry
	/// Type of joint
	JointType m_joint_type;
	/// Position of joint frame (body-fixed frame at q=0) relative to the parent frame
	/// Components are in body-fixed frame of the parent
	vec3 m_parent_pos_parent_body_ref;
	/// Orientation of joint frame (body-fixed frame at q=0) relative to the parent frame
	mat33 m_body_T_parent_ref;
	/// Joint rotational Jacobian, ie, the partial derivative of the body-fixed frames absolute
	/// angular velocity w.r.t. the generalized velocity of this body's relative degree of freedom.
	/// For revolute joints this is the joint axis, for prismatic joints it is a null matrix.
	/// (NOTE: dimensions will have to be dynamic for additional joint types!)
	vec3 m_Jac_JR;
	/// Joint translational Jacobian, ie, the partial derivative of the body-fixed frames absolute
	/// linear velocity w.r.t. the generalized velocity of this body's relative degree of freedom.
	/// For prismatic joints this is the joint axis, for revolute joints it is a null matrix.
	/// (NOTE: dimensions might have to be dynamic for additional joint types!)
	vec3 m_Jac_JT;
	/// m_Jac_JT in the parent frame, it, m_body_T_parent_ref.transpose()*m_Jac_JT
	vec3 m_parent_Jac_JT;
	/// Start of index range for the position degree(s) of freedom describing this body's motion
	/// relative to
	/// its parent. The indices are wrt the multibody system's q-vector of generalized coordinates.
	int m_q_index;

	// 6 Scratch data for mass matrix computation using "composite rigid body algorithm"
	/// mass of the subtree rooted in this body
	idScalar m_subtree_mass;
	/// center of mass * mass for subtree rooted in this body, in body-fixed frame
	vec3 m_body_subtree_mass_com;
	/// moment of inertia of subtree rooted in this body, w.r.t. body origin, in body-fixed frame
	mat33 m_body_subtree_I_body;

#if (defined BT_ID_HAVE_MAT3X) && (defined BT_ID_WITH_JACOBIANS)
	/// translational jacobian in body-fixed frame d(m_body_vel)/du
	mat3x m_body_Jac_T;
	/// rotationsl jacobian in body-fixed frame d(m_body_ang_vel)/du
	mat3x m_body_Jac_R;
	/// components of linear acceleration depending on u
	/// (same as is d(m_Jac_T)/dt*u)
	vec3 m_body_dot_Jac_T_u;
	/// components of angular acceleration depending on u
	/// (same as is d(m_Jac_T)/dt*u)
	vec3 m_body_dot_Jac_R_u;
#endif
};

/// The MBS implements a tree structured multibody system
class MultiBodyTree::MultiBodyImpl
{
	friend class MultiBodyTree;

public:
	ID_DECLARE_ALIGNED_ALLOCATOR();

	enum KinUpdateType
	{
		POSITION_ONLY,
		POSITION_VELOCITY,
		POSITION_VELOCITY_ACCELERATION
	};

	/// constructor
	/// @param num_bodies the number of bodies in the system
	/// @param num_dofs number of degrees of freedom in the system
	MultiBodyImpl(int num_bodies_, int num_dofs_);

	/// \copydoc MultiBodyTree::calculateInverseDynamics
	int calculateInverseDynamics(const vecx& q, const vecx& u, const vecx& dot_u,
								 vecx* joint_forces);
	///\copydoc MultiBodyTree::calculateMassMatrix
	int calculateMassMatrix(const vecx& q, const bool update_kinematics,
							const bool initialize_matrix, const bool set_lower_triangular_matrix,
							matxx* mass_matrix);
	/// calculate kinematics (vector quantities)
	/// Depending on type, update positions only, positions & velocities, or positions, velocities
	/// and accelerations.
	int calculateKinematics(const vecx& q, const vecx& u, const vecx& dot_u, const KinUpdateType type);
#if (defined BT_ID_HAVE_MAT3X) && (defined BT_ID_WITH_JACOBIANS)
	/// calculate jacobians and (if type == POSITION_VELOCITY), also velocity-dependent accelration terms.
	int calculateJacobians(const vecx& q, const vecx& u, const KinUpdateType type);
	/// \copydoc MultiBodyTree::getBodyDotJacobianTransU
	int getBodyDotJacobianTransU(const int body_index, vec3* world_dot_jac_trans_u) const;
	/// \copydoc MultiBodyTree::getBodyDotJacobianRotU
	int getBodyDotJacobianRotU(const int body_index, vec3* world_dot_jac_rot_u) const;
	/// \copydoc MultiBodyTree::getBodyJacobianTrans
	int getBodyJacobianTrans(const int body_index, mat3x* world_jac_trans) const;
	/// \copydoc MultiBodyTree::getBodyJacobianRot
	int getBodyJacobianRot(const int body_index, mat3x* world_jac_rot) const;
	/// Add relative Jacobian component from motion relative to parent body
	/// @param body the body to add the Jacobian component for
	void addRelativeJacobianComponent(RigidBody& body);
#endif
	/// generate additional index sets from the parent_index array
	/// @return -1 on error, 0 on success
	int generateIndexSets();
	/// set gravity acceleration in world frame
	/// @param gravity gravity vector in the world frame
	/// @return 0 on success, -1 on error
	int setGravityInWorldFrame(const vec3& gravity);
	/// pretty print tree
	void printTree();
	/// print tree data
	void printTreeData();
	/// initialize fixed data
	void calculateStaticData();
	/// \copydoc MultiBodyTree::getBodyFrame
	int getBodyFrame(const int index, vec3* world_origin, mat33* body_T_world) const;
	/// \copydoc MultiBodyTree::getParentIndex
	int getParentIndex(const int body_index, int* m_parent_index);
	/// \copydoc MultiBodyTree::getJointType
	int getJointType(const int body_index, JointType* joint_type) const;
	/// \copydoc MultiBodyTree::getJointTypeStr
	int getJointTypeStr(const int body_index, const char** joint_type) const;
	/// \copydoc MultiBodyTree::getParentRParentBodyRef
	int getParentRParentBodyRef(const int body_index, vec3* r) const;
	/// \copydoc MultiBodyTree::getBodyTParentRef
	int getBodyTParentRef(const int body_index, mat33* T) const;
	/// \copydoc MultiBodyTree::getBodyAxisOfMotion
	int getBodyAxisOfMotion(const int body_index, vec3* axis) const;
	/// \copydoc MultiBodyTree:getDoFOffset
	int getDoFOffset(const int body_index, int* q_index) const;
	/// \copydoc MultiBodyTree::getBodyOrigin
	int getBodyOrigin(const int body_index, vec3* world_origin) const;
	/// \copydoc MultiBodyTree::getBodyCoM
	int getBodyCoM(const int body_index, vec3* world_com) const;
	/// \copydoc MultiBodyTree::getBodyTransform
	int getBodyTransform(const int body_index, mat33* world_T_body) const;
	/// \copydoc MultiBodyTree::getBodyAngularVelocity
	int getBodyAngularVelocity(const int body_index, vec3* world_omega) const;
	/// \copydoc MultiBodyTree::getBodyLinearVelocity
	int getBodyLinearVelocity(const int body_index, vec3* world_velocity) const;
	/// \copydoc MultiBodyTree::getBodyLinearVelocityCoM
	int getBodyLinearVelocityCoM(const int body_index, vec3* world_velocity) const;
	/// \copydoc MultiBodyTree::getBodyAngularAcceleration
	int getBodyAngularAcceleration(const int body_index, vec3* world_dot_omega) const;
	/// \copydoc MultiBodyTree::getBodyLinearAcceleration
	int getBodyLinearAcceleration(const int body_index, vec3* world_acceleration) const;
	/// \copydoc MultiBodyTree::getUserInt
	int getUserInt(const int body_index, int* user_int) const;
	/// \copydoc MultiBodyTree::getUserPtr
	int getUserPtr(const int body_index, void** user_ptr) const;
	/// \copydoc MultiBodyTree::setUserInt
	int setUserInt(const int body_index, const int user_int);
	/// \copydoc MultiBodyTree::setUserPtr
	int setUserPtr(const int body_index, void* const user_ptr);
	///\copydoc MultiBodytTree::setBodyMass
	int setBodyMass(const int body_index, const idScalar mass);
	///\copydoc MultiBodytTree::setBodyFirstMassMoment
	int setBodyFirstMassMoment(const int body_index, const vec3& first_mass_moment);
	///\copydoc MultiBodytTree::setBodySecondMassMoment
	int setBodySecondMassMoment(const int body_index, const mat33& second_mass_moment);
	///\copydoc MultiBodytTree::getBodyMass
	int getBodyMass(const int body_index, idScalar* mass) const;
	///\copydoc MultiBodytTree::getBodyFirstMassMoment
	int getBodyFirstMassMoment(const int body_index, vec3* first_mass_moment) const;
	///\copydoc MultiBodytTree::getBodySecondMassMoment
	int getBodySecondMassMoment(const int body_index, mat33* second_mass_moment) const;
	/// \copydoc MultiBodyTree::clearAllUserForcesAndMoments
	void clearAllUserForcesAndMoments();
	/// \copydoc MultiBodyTree::addUserForce
	int addUserForce(const int body_index, const vec3& body_force);
	/// \copydoc MultiBodyTree::addUserMoment
	int addUserMoment(const int body_index, const vec3& body_moment);

private:
	// debug function. print tree structure to stdout
	void printTree(int index, int indentation);
	// get string representation of JointType (for debugging)
	const char* jointTypeToString(const JointType& type) const;
	// get number of degrees of freedom from joint type
	int bodyNumDoFs(const JointType& type) const;
	// number of bodies in the system
	int m_num_bodies;
	// number of degrees of freedom
	int m_num_dofs;
	// Gravitational acceleration (in world frame)
	vec3 m_world_gravity;
	// vector of bodies in the system
	// body 0 is used as an environment body and is allways fixed.
	// The bodies are ordered such that a parent body always has an index
	// smaller than its child.
	idArray<RigidBody>::type m_body_list;
	// Parent_index[i] is the index for i's parent body in body_list.
	// This fully describes the tree.
	idArray<int>::type m_parent_index;
	// child_indices[i] contains a vector of indices of
	// all children of the i-th body
	idArray<idArray<int>::type>::type m_child_indices;
	// Indices of rotary joints
	idArray<int>::type m_body_revolute_list;
	// Indices of prismatic joints
	idArray<int>::type m_body_prismatic_list;
	// Indices of floating joints
	idArray<int>::type m_body_floating_list;
	// Indices of spherical joints
	idArray<int>::type m_body_spherical_list;
	// a user-provided integer
	idArray<int>::type m_user_int;
	// a user-provided pointer
	idArray<void*>::type m_user_ptr;
#if (defined BT_ID_HAVE_MAT3X) && (defined BT_ID_WITH_JACOBIANS)
	mat3x m_m3x;
#endif
};
}  // namespace btInverseDynamics
#endif

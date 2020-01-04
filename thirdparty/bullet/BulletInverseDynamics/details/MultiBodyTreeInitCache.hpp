#ifndef MULTIBODYTREEINITCACHE_HPP_
#define MULTIBODYTREEINITCACHE_HPP_

#include "../IDConfig.hpp"
#include "../IDMath.hpp"
#include "../MultiBodyTree.hpp"

namespace btInverseDynamics
{
/// Mass properties of a rigid body
struct InertiaData
{
	ID_DECLARE_ALIGNED_ALLOCATOR();

	/// mass
	idScalar m_mass;
	/// vector from body-fixed frame to center of mass,
	/// in body-fixed frame, multiplied by the mass
	vec3 m_body_pos_body_com;
	/// moment of inertia w.r.t. the origin of the body-fixed
	/// frame, represented in that frame
	mat33 m_body_I_body;
};

/// Joint properties
struct JointData
{
	ID_DECLARE_ALIGNED_ALLOCATOR();

	/// type of joint
	JointType m_type;
	/// index of parent body
	int m_parent;
	/// index of child body
	int m_child;
	/// vector from parent's body-fixed frame to child's body-fixed
	/// frame for q=0, written in the parent's body fixed frame
	vec3 m_parent_pos_parent_child_ref;
	/// Transform matrix converting vectors written in the parent's frame
	/// into vectors written in the child's frame for q=0
	/// ie, child_vector = child_T_parent_ref * parent_vector;
	mat33 m_child_T_parent_ref;
	/// Axis of motion for 1 degree-of-freedom joints,
	/// written in the child's frame
	/// For revolute joints, the q-value is positive for a positive
	/// rotation about this axis.
	/// For prismatic joints, the q-value is positive for a positive
	/// translation is this direction.
	vec3 m_child_axis_of_motion;
};

/// Data structure to store data passed by the user.
/// This is used in MultiBodyTree::finalize to build internal data structures.
class MultiBodyTree::InitCache
{
public:
	ID_DECLARE_ALIGNED_ALLOCATOR();
	/// constructor
	InitCache();
	///\copydoc MultiBodyTree::addBody
	int addBody(const int body_index, const int parent_index, const JointType joint_type,
				const vec3 &parent_r_parent_body_ref, const mat33 &body_T_parent_ref,
				const vec3 &body_axis_of_motion, idScalar mass, const vec3 &body_r_body_com,
				const mat33 &body_I_body, const int user_int, void *user_ptr);
	/// build index arrays
	/// @return 0 on success, -1 on failure
	int buildIndexSets();
	/// @return number of degrees of freedom
	int numDoFs() const { return m_num_dofs; }
	/// @return number of bodies
	int numBodies() const { return m_inertias.size(); }
	/// get inertia data for index
	/// @param index of the body
	/// @param inertia pointer for return data
	/// @return 0 on success, -1 on failure
	int getInertiaData(const int index, InertiaData *inertia) const;
	/// get joint data for index
	/// @param index of the body
	/// @param joint pointer for return data
	/// @return 0 on success, -1 on failure
	int getJointData(const int index, JointData *joint) const;
	/// get parent index array (paren_index[i] is the index of the parent of i)
	/// @param parent_index pointer for return data
	void getParentIndexArray(idArray<int>::type *parent_index) { *parent_index = m_parent_index; }
	/// get user integer
	/// @param index body index
	/// @param user_int user integer
	/// @return 0 on success, -1 on failure
	int getUserInt(const int index, int *user_int) const;
	/// get user pointer
	/// @param index body index
	/// @param user_int user pointer
	/// @return 0 on success, -1 on failure
	int getUserPtr(const int index, void **user_ptr) const;

private:
	// vector of bodies
	idArray<InertiaData>::type m_inertias;
	// vector of joints
	idArray<JointData>::type m_joints;
	// number of mechanical degrees of freedom
	int m_num_dofs;
	// parent index array
	idArray<int>::type m_parent_index;
	// user integers
	idArray<int>::type m_user_int;
	// user pointers
	idArray<void *>::type m_user_ptr;
	// index of root body (or -1 if not set)
	int m_root_index;
};
}  // namespace btInverseDynamics
#endif  // MULTIBODYTREEINITCACHE_HPP_

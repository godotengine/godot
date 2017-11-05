#include "MultiBodyTreeInitCache.hpp"

namespace btInverseDynamics {

MultiBodyTree::InitCache::InitCache() {
	m_inertias.resize(0);
	m_joints.resize(0);
	m_num_dofs = 0;
	m_root_index=-1;
}

int MultiBodyTree::InitCache::addBody(const int body_index, const int parent_index,
									  const JointType joint_type,
									  const vec3& parent_r_parent_body_ref,
									  const mat33& body_T_parent_ref,
									  const vec3& body_axis_of_motion, const idScalar mass,
									  const vec3& body_r_body_com, const mat33& body_I_body,
									  const int user_int, void* user_ptr) {
	switch (joint_type) {
		case REVOLUTE:
		case PRISMATIC:
			m_num_dofs += 1;
			break;
		case FIXED:
			// does not add a degree of freedom
			// m_num_dofs+=0;
			break;
		case FLOATING:
			m_num_dofs += 6;
			break;
		default:
			error_message("unknown joint type %d\n", joint_type);
			return -1;
	}

	if(-1 == parent_index) {
		if(m_root_index>=0) {
			error_message("trying to add body %d as root, but already added %d as root body\n",
						  body_index, m_root_index);
			return -1;
		}
		m_root_index=body_index;
	}

	JointData joint;
	joint.m_child = body_index;
	joint.m_parent = parent_index;
	joint.m_type = joint_type;
	joint.m_parent_pos_parent_child_ref = parent_r_parent_body_ref;
	joint.m_child_T_parent_ref = body_T_parent_ref;
	joint.m_child_axis_of_motion = body_axis_of_motion;

	InertiaData body;
	body.m_mass = mass;
	body.m_body_pos_body_com = body_r_body_com;
	body.m_body_I_body = body_I_body;

	m_inertias.push_back(body);
	m_joints.push_back(joint);
	m_user_int.push_back(user_int);
	m_user_ptr.push_back(user_ptr);
	return 0;
}
int MultiBodyTree::InitCache::getInertiaData(const int index, InertiaData* inertia) const {
	if (index < 0 || index > static_cast<int>(m_inertias.size())) {
		error_message("index out of range\n");
		return -1;
	}

	*inertia = m_inertias[index];
	return 0;
}

int MultiBodyTree::InitCache::getUserInt(const int index, int* user_int) const {
	if (index < 0 || index > static_cast<int>(m_user_int.size())) {
		error_message("index out of range\n");
		return -1;
	}
	*user_int = m_user_int[index];
	return 0;
}

int MultiBodyTree::InitCache::getUserPtr(const int index, void** user_ptr) const {
	if (index < 0 || index > static_cast<int>(m_user_ptr.size())) {
		error_message("index out of range\n");
		return -1;
	}
	*user_ptr = m_user_ptr[index];
	return 0;
}

int MultiBodyTree::InitCache::getJointData(const int index, JointData* joint) const {
	if (index < 0 || index > static_cast<int>(m_joints.size())) {
		error_message("index out of range\n");
		return -1;
	}
	*joint = m_joints[index];
	return 0;
}

int MultiBodyTree::InitCache::buildIndexSets() {
	// NOTE: This function assumes that proper indices were provided
	//	   User2InternalIndex from utils can be used to facilitate this.

	m_parent_index.resize(numBodies());
	for (idArrayIdx j = 0; j < m_joints.size(); j++) {
		const JointData& joint = m_joints[j];
		m_parent_index[joint.m_child] = joint.m_parent;
	}

	return 0;
}
}

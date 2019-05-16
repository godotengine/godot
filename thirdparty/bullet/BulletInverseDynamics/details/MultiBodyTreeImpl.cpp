#include "MultiBodyTreeImpl.hpp"

namespace btInverseDynamics
{
MultiBodyTree::MultiBodyImpl::MultiBodyImpl(int num_bodies_, int num_dofs_)
	: m_num_bodies(num_bodies_), m_num_dofs(num_dofs_)
#if (defined BT_ID_HAVE_MAT3X) && (defined BT_ID_WITH_JACOBIANS)
	  ,
	  m_m3x(3, m_num_dofs)
#endif
{
#if (defined BT_ID_HAVE_MAT3X) && (defined BT_ID_WITH_JACOBIANS)
	resize(m_m3x, m_num_dofs);
#endif
	m_body_list.resize(num_bodies_);
	m_parent_index.resize(num_bodies_);
	m_child_indices.resize(num_bodies_);
	m_user_int.resize(num_bodies_);
	m_user_ptr.resize(num_bodies_);

	m_world_gravity(0) = 0.0;
	m_world_gravity(1) = 0.0;
	m_world_gravity(2) = -9.8;
}

const char *MultiBodyTree::MultiBodyImpl::jointTypeToString(const JointType &type) const
{
	switch (type)
	{
		case FIXED:
			return "fixed";
		case REVOLUTE:
			return "revolute";
		case PRISMATIC:
			return "prismatic";
		case FLOATING:
			return "floating";
		case SPHERICAL:
			return "spherical";
	}
	return "error: invalid";
}

inline void indent(const int &level)
{
	for (int j = 0; j < level; j++)
		id_printf("  ");  // indent
}

void MultiBodyTree::MultiBodyImpl::printTree()
{
	id_printf("body %.2d[%s]: root\n", 0, jointTypeToString(m_body_list[0].m_joint_type));
	printTree(0, 0);
}

void MultiBodyTree::MultiBodyImpl::printTreeData()
{
	for (idArrayIdx i = 0; i < m_body_list.size(); i++)
	{
		RigidBody &body = m_body_list[i];
		id_printf("body: %d\n", static_cast<int>(i));
		id_printf("type: %s\n", jointTypeToString(body.m_joint_type));
		id_printf("q_index= %d\n", body.m_q_index);
		id_printf("Jac_JR= [%f;%f;%f]\n", body.m_Jac_JR(0), body.m_Jac_JR(1), body.m_Jac_JR(2));
		id_printf("Jac_JT= [%f;%f;%f]\n", body.m_Jac_JT(0), body.m_Jac_JT(1), body.m_Jac_JT(2));

		id_printf("mass = %f\n", body.m_mass);
		id_printf("mass * com = [%f %f %f]\n", body.m_body_mass_com(0), body.m_body_mass_com(1),
				  body.m_body_mass_com(2));
		id_printf(
			"I_o= [%f %f %f;\n"
			"	  %f %f %f;\n"
			"	  %f %f %f]\n",
			body.m_body_I_body(0, 0), body.m_body_I_body(0, 1), body.m_body_I_body(0, 2),
			body.m_body_I_body(1, 0), body.m_body_I_body(1, 1), body.m_body_I_body(1, 2),
			body.m_body_I_body(2, 0), body.m_body_I_body(2, 1), body.m_body_I_body(2, 2));

		id_printf("parent_pos_parent_body_ref= [%f %f %f]\n", body.m_parent_pos_parent_body_ref(0),
				  body.m_parent_pos_parent_body_ref(1), body.m_parent_pos_parent_body_ref(2));
	}
}
int MultiBodyTree::MultiBodyImpl::bodyNumDoFs(const JointType &type) const
{
	switch (type)
	{
		case FIXED:
			return 0;
		case REVOLUTE:
		case PRISMATIC:
			return 1;
		case FLOATING:
			return 6;
		case SPHERICAL:
			return 3;
	}
	bt_id_error_message("unknown joint type %d\n", type);
	return 0;
}

void MultiBodyTree::MultiBodyImpl::printTree(int index, int indentation)
{
	// this is adapted from URDF2Bullet.
	// TODO: fix this and print proper graph (similar to git --log --graph)
	int num_children = m_child_indices[index].size();

	indentation += 2;
	int count = 0;

	for (int i = 0; i < num_children; i++)
	{
		int child_index = m_child_indices[index][i];
		indent(indentation);
		id_printf("body %.2d[%s]: %.2d is child no. %d (qi= %d .. %d) \n", index,
				  jointTypeToString(m_body_list[index].m_joint_type), child_index, (count++) + 1,
				  m_body_list[index].m_q_index,
				  m_body_list[index].m_q_index + bodyNumDoFs(m_body_list[index].m_joint_type));
		// first grandchild
		printTree(child_index, indentation);
	}
}

int MultiBodyTree::MultiBodyImpl::setGravityInWorldFrame(const vec3 &gravity)
{
	m_world_gravity = gravity;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::generateIndexSets()
{
	m_body_revolute_list.resize(0);
	m_body_prismatic_list.resize(0);
	int q_index = 0;
	for (idArrayIdx i = 0; i < m_body_list.size(); i++)
	{
		RigidBody &body = m_body_list[i];
		body.m_q_index = -1;
		switch (body.m_joint_type)
		{
			case REVOLUTE:
				m_body_revolute_list.push_back(i);
				body.m_q_index = q_index;
				q_index++;
				break;
			case PRISMATIC:
				m_body_prismatic_list.push_back(i);
				body.m_q_index = q_index;
				q_index++;
				break;
			case FIXED:
				// do nothing
				break;
			case FLOATING:
				m_body_floating_list.push_back(i);
				body.m_q_index = q_index;
				q_index += 6;
				break;
			case SPHERICAL:
				m_body_spherical_list.push_back(i);
				body.m_q_index = q_index;
				q_index += 3;
				break;
			default:
				bt_id_error_message("unsupported joint type %d\n", body.m_joint_type);
				return -1;
		}
	}
	// sanity check
	if (q_index != m_num_dofs)
	{
		bt_id_error_message("internal error, q_index= %d but num_dofs %d\n", q_index, m_num_dofs);
		return -1;
	}

	m_child_indices.resize(m_body_list.size());

	for (idArrayIdx child = 1; child < m_parent_index.size(); child++)
	{
		const int &parent = m_parent_index[child];
		if (parent >= 0 && parent < (static_cast<int>(m_parent_index.size()) - 1))
		{
			m_child_indices[parent].push_back(child);
		}
		else
		{
			if (-1 == parent)
			{
				// multiple bodies are directly linked to the environment, ie, not a single root
				bt_id_error_message("building index sets parent(%zu)= -1 (multiple roots)\n", child);
			}
			else
			{
				// should never happen
				bt_id_error_message(
					"building index sets. parent_index[%zu]= %d, but m_parent_index.size()= %d\n",
					child, parent, static_cast<int>(m_parent_index.size()));
			}
			return -1;
		}
	}

	return 0;
}

void MultiBodyTree::MultiBodyImpl::calculateStaticData()
{
	// relative kinematics that are not a function of q, u, dot_u
	for (idArrayIdx i = 0; i < m_body_list.size(); i++)
	{
		RigidBody &body = m_body_list[i];
		switch (body.m_joint_type)
		{
			case REVOLUTE:
				body.m_parent_vel_rel(0) = 0;
				body.m_parent_vel_rel(1) = 0;
				body.m_parent_vel_rel(2) = 0;
				body.m_parent_acc_rel(0) = 0;
				body.m_parent_acc_rel(1) = 0;
				body.m_parent_acc_rel(2) = 0;
				body.m_parent_pos_parent_body = body.m_parent_pos_parent_body_ref;
				break;
			case PRISMATIC:
				body.m_body_T_parent = body.m_body_T_parent_ref;
				body.m_parent_Jac_JT = body.m_body_T_parent_ref.transpose() * body.m_Jac_JT;
				body.m_body_ang_vel_rel(0) = 0;
				body.m_body_ang_vel_rel(1) = 0;
				body.m_body_ang_vel_rel(2) = 0;
				body.m_body_ang_acc_rel(0) = 0;
				body.m_body_ang_acc_rel(1) = 0;
				body.m_body_ang_acc_rel(2) = 0;
				break;
			case FIXED:
				body.m_parent_pos_parent_body = body.m_parent_pos_parent_body_ref;
				body.m_body_T_parent = body.m_body_T_parent_ref;
				body.m_body_ang_vel_rel(0) = 0;
				body.m_body_ang_vel_rel(1) = 0;
				body.m_body_ang_vel_rel(2) = 0;
				body.m_parent_vel_rel(0) = 0;
				body.m_parent_vel_rel(1) = 0;
				body.m_parent_vel_rel(2) = 0;
				body.m_body_ang_acc_rel(0) = 0;
				body.m_body_ang_acc_rel(1) = 0;
				body.m_body_ang_acc_rel(2) = 0;
				body.m_parent_acc_rel(0) = 0;
				body.m_parent_acc_rel(1) = 0;
				body.m_parent_acc_rel(2) = 0;
				break;
			case FLOATING:
				// no static data
				break;
			case SPHERICAL:
				//todo: review
				body.m_parent_pos_parent_body = body.m_parent_pos_parent_body_ref;
				body.m_parent_vel_rel(0) = 0;
				body.m_parent_vel_rel(1) = 0;
				body.m_parent_vel_rel(2) = 0;
				body.m_parent_acc_rel(0) = 0;
				body.m_parent_acc_rel(1) = 0;
				body.m_parent_acc_rel(2) = 0;
				break;
		}

			// resize & initialize jacobians to zero.
#if (defined BT_ID_HAVE_MAT3X) && (defined BT_ID_WITH_JACOBIANS)
		body.m_body_dot_Jac_T_u(0) = 0.0;
		body.m_body_dot_Jac_T_u(1) = 0.0;
		body.m_body_dot_Jac_T_u(2) = 0.0;
		body.m_body_dot_Jac_R_u(0) = 0.0;
		body.m_body_dot_Jac_R_u(1) = 0.0;
		body.m_body_dot_Jac_R_u(2) = 0.0;
		resize(body.m_body_Jac_T, m_num_dofs);
		resize(body.m_body_Jac_R, m_num_dofs);
		body.m_body_Jac_T.setZero();
		body.m_body_Jac_R.setZero();
#endif  //
	}
}

int MultiBodyTree::MultiBodyImpl::calculateInverseDynamics(const vecx &q, const vecx &u,
														   const vecx &dot_u, vecx *joint_forces)
{
	if (q.size() != m_num_dofs || u.size() != m_num_dofs || dot_u.size() != m_num_dofs ||
		joint_forces->size() != m_num_dofs)
	{
		bt_id_error_message(
			"wrong vector dimension. system has %d DOFs,\n"
			"but dim(q)= %d, dim(u)= %d, dim(dot_u)= %d, dim(joint_forces)= %d\n",
			m_num_dofs, static_cast<int>(q.size()), static_cast<int>(u.size()),
			static_cast<int>(dot_u.size()), static_cast<int>(joint_forces->size()));
		return -1;
	}
	// 1. relative kinematics
	if (-1 == calculateKinematics(q, u, dot_u, POSITION_VELOCITY_ACCELERATION))
	{
		bt_id_error_message("error in calculateKinematics\n");
		return -1;
	}
	// 2. update contributions to equations of motion for every body.
	for (idArrayIdx i = 0; i < m_body_list.size(); i++)
	{
		RigidBody &body = m_body_list[i];
		// 3.4 update dynamic terms (rate of change of angular & linear momentum)
		body.m_eom_lhs_rotational =
			body.m_body_I_body * body.m_body_ang_acc + body.m_body_mass_com.cross(body.m_body_acc) +
			body.m_body_ang_vel.cross(body.m_body_I_body * body.m_body_ang_vel) -
			body.m_body_moment_user;
		body.m_eom_lhs_translational =
			body.m_body_ang_acc.cross(body.m_body_mass_com) + body.m_mass * body.m_body_acc +
			body.m_body_ang_vel.cross(body.m_body_ang_vel.cross(body.m_body_mass_com)) -
			body.m_body_force_user;
	}

	// 3. calculate full set of forces at parent joint
	// (not directly calculating the joint force along the free direction
	// simplifies inclusion of fixed joints.
	// An alternative would be to fuse bodies in a pre-processing step,
	// but that would make changing masses online harder (eg, payload masses
	// added with fixed  joints to a gripper)
	// Also, this enables adding zero weight bodies as a way to calculate frame poses
	// for force elements, etc.

	for (int body_idx = m_body_list.size() - 1; body_idx >= 0; body_idx--)
	{
		// sum of forces and moments acting on this body from its children
		vec3 sum_f_children;
		vec3 sum_m_children;
		setZero(sum_f_children);
		setZero(sum_m_children);
		for (idArrayIdx child_list_idx = 0; child_list_idx < m_child_indices[body_idx].size();
			 child_list_idx++)
		{
			const RigidBody &child = m_body_list[m_child_indices[body_idx][child_list_idx]];
			vec3 child_joint_force_in_this_frame =
				child.m_body_T_parent.transpose() * child.m_force_at_joint;
			sum_f_children -= child_joint_force_in_this_frame;
			sum_m_children -= child.m_body_T_parent.transpose() * child.m_moment_at_joint +
							  child.m_parent_pos_parent_body.cross(child_joint_force_in_this_frame);
		}
		RigidBody &body = m_body_list[body_idx];

		body.m_force_at_joint = body.m_eom_lhs_translational - sum_f_children;
		body.m_moment_at_joint = body.m_eom_lhs_rotational - sum_m_children;
	}

	// 4. Calculate Joint forces.
	// These are the components of force_at_joint/moment_at_joint
	// in the free directions given by Jac_JT/Jac_JR
	// 4.1 revolute joints
	for (idArrayIdx i = 0; i < m_body_revolute_list.size(); i++)
	{
		RigidBody &body = m_body_list[m_body_revolute_list[i]];
		// (*joint_forces)(body.m_q_index) = body.m_Jac_JR.transpose() * body.m_moment_at_joint;
		(*joint_forces)(body.m_q_index) = body.m_Jac_JR.dot(body.m_moment_at_joint);
	}
	// 4.2 for prismatic joints
	for (idArrayIdx i = 0; i < m_body_prismatic_list.size(); i++)
	{
		RigidBody &body = m_body_list[m_body_prismatic_list[i]];
		// (*joint_forces)(body.m_q_index) = body.m_Jac_JT.transpose() * body.m_force_at_joint;
		(*joint_forces)(body.m_q_index) = body.m_Jac_JT.dot(body.m_force_at_joint);
	}
	// 4.3 floating bodies (6-DoF joints)
	for (idArrayIdx i = 0; i < m_body_floating_list.size(); i++)
	{
		RigidBody &body = m_body_list[m_body_floating_list[i]];
		(*joint_forces)(body.m_q_index + 0) = body.m_moment_at_joint(0);
		(*joint_forces)(body.m_q_index + 1) = body.m_moment_at_joint(1);
		(*joint_forces)(body.m_q_index + 2) = body.m_moment_at_joint(2);

		(*joint_forces)(body.m_q_index + 3) = body.m_force_at_joint(0);
		(*joint_forces)(body.m_q_index + 4) = body.m_force_at_joint(1);
		(*joint_forces)(body.m_q_index + 5) = body.m_force_at_joint(2);
	}

	// 4.4 spherical bodies (3-DoF joints)
	for (idArrayIdx i = 0; i < m_body_spherical_list.size(); i++)
	{
		//todo: review
		RigidBody &body = m_body_list[m_body_spherical_list[i]];
		(*joint_forces)(body.m_q_index + 0) = body.m_moment_at_joint(0);
		(*joint_forces)(body.m_q_index + 1) = body.m_moment_at_joint(1);
		(*joint_forces)(body.m_q_index + 2) = body.m_moment_at_joint(2);
	}
	return 0;
}

int MultiBodyTree::MultiBodyImpl::calculateKinematics(const vecx &q, const vecx &u, const vecx &dot_u,
													  const KinUpdateType type)
{
	if (q.size() != m_num_dofs || u.size() != m_num_dofs || dot_u.size() != m_num_dofs)
	{
		bt_id_error_message(
			"wrong vector dimension. system has %d DOFs,\n"
			"but dim(q)= %d, dim(u)= %d, dim(dot_u)= %d\n",
			m_num_dofs, static_cast<int>(q.size()), static_cast<int>(u.size()),
			static_cast<int>(dot_u.size()));
		return -1;
	}
	if (type != POSITION_ONLY && type != POSITION_VELOCITY && type != POSITION_VELOCITY_ACCELERATION)
	{
		bt_id_error_message("invalid type %d\n", type);
		return -1;
	}

	// 1. update relative kinematics
	// 1.1 for revolute
	for (idArrayIdx i = 0; i < m_body_revolute_list.size(); i++)
	{
		RigidBody &body = m_body_list[m_body_revolute_list[i]];
		mat33 T;
		bodyTParentFromAxisAngle(body.m_Jac_JR, q(body.m_q_index), &T);
		body.m_body_T_parent = T * body.m_body_T_parent_ref;
		if (type >= POSITION_VELOCITY)
		{
			body.m_body_ang_vel_rel = body.m_Jac_JR * u(body.m_q_index);
		}
		if (type >= POSITION_VELOCITY_ACCELERATION)
		{
			body.m_body_ang_acc_rel = body.m_Jac_JR * dot_u(body.m_q_index);
		}
	}
	// 1.2 for prismatic
	for (idArrayIdx i = 0; i < m_body_prismatic_list.size(); i++)
	{
		RigidBody &body = m_body_list[m_body_prismatic_list[i]];
		body.m_parent_pos_parent_body =
			body.m_parent_pos_parent_body_ref + body.m_parent_Jac_JT * q(body.m_q_index);
		if (type >= POSITION_VELOCITY)
		{
			body.m_parent_vel_rel =
				body.m_body_T_parent_ref.transpose() * body.m_Jac_JT * u(body.m_q_index);
		}
		if (type >= POSITION_VELOCITY_ACCELERATION)
		{
			body.m_parent_acc_rel = body.m_parent_Jac_JT * dot_u(body.m_q_index);
		}
	}
	// 1.3 fixed joints: nothing to do
	// 1.4 6dof joints:
	for (idArrayIdx i = 0; i < m_body_floating_list.size(); i++)
	{
		RigidBody &body = m_body_list[m_body_floating_list[i]];

		body.m_body_T_parent = transformZ(q(body.m_q_index + 2)) *
							   transformY(q(body.m_q_index + 1)) *
							   transformX(q(body.m_q_index));
		body.m_parent_pos_parent_body(0) = q(body.m_q_index + 3);
		body.m_parent_pos_parent_body(1) = q(body.m_q_index + 4);
		body.m_parent_pos_parent_body(2) = q(body.m_q_index + 5);
		body.m_parent_pos_parent_body = body.m_body_T_parent * body.m_parent_pos_parent_body;

		if (type >= POSITION_VELOCITY)
		{
			body.m_body_ang_vel_rel(0) = u(body.m_q_index + 0);
			body.m_body_ang_vel_rel(1) = u(body.m_q_index + 1);
			body.m_body_ang_vel_rel(2) = u(body.m_q_index + 2);

			body.m_parent_vel_rel(0) = u(body.m_q_index + 3);
			body.m_parent_vel_rel(1) = u(body.m_q_index + 4);
			body.m_parent_vel_rel(2) = u(body.m_q_index + 5);

			body.m_parent_vel_rel = body.m_body_T_parent.transpose() * body.m_parent_vel_rel;
		}
		if (type >= POSITION_VELOCITY_ACCELERATION)
		{
			body.m_body_ang_acc_rel(0) = dot_u(body.m_q_index + 0);
			body.m_body_ang_acc_rel(1) = dot_u(body.m_q_index + 1);
			body.m_body_ang_acc_rel(2) = dot_u(body.m_q_index + 2);

			body.m_parent_acc_rel(0) = dot_u(body.m_q_index + 3);
			body.m_parent_acc_rel(1) = dot_u(body.m_q_index + 4);
			body.m_parent_acc_rel(2) = dot_u(body.m_q_index + 5);

			body.m_parent_acc_rel = body.m_body_T_parent.transpose() * body.m_parent_acc_rel;
		}
	}
	
	for (idArrayIdx i = 0; i < m_body_spherical_list.size(); i++)
	{
		//todo: review
		RigidBody &body = m_body_list[m_body_spherical_list[i]];

		body.m_body_T_parent = transformZ(q(body.m_q_index + 2)) *
							   transformY(q(body.m_q_index + 1)) *
							   transformX(q(body.m_q_index));
		body.m_parent_pos_parent_body = body.m_body_T_parent * body.m_parent_pos_parent_body;

		if (type >= POSITION_VELOCITY)
		{
			body.m_body_ang_vel_rel(0) = u(body.m_q_index + 0);
			body.m_body_ang_vel_rel(1) = u(body.m_q_index + 1);
			body.m_body_ang_vel_rel(2) = u(body.m_q_index + 2);
			body.m_parent_vel_rel = body.m_body_T_parent.transpose() * body.m_parent_vel_rel;
		}
		if (type >= POSITION_VELOCITY_ACCELERATION)
		{
			body.m_body_ang_acc_rel(0) = dot_u(body.m_q_index + 0);
			body.m_body_ang_acc_rel(1) = dot_u(body.m_q_index + 1);
			body.m_body_ang_acc_rel(2) = dot_u(body.m_q_index + 2);
			body.m_parent_acc_rel = body.m_body_T_parent.transpose() * body.m_parent_acc_rel;
		}
	}

	// 2. absolute kinematic quantities (vector valued)
	// NOTE: this should be optimized by specializing for different body types
	// (e.g., relative rotation is always zero for prismatic joints, etc.)

	// calculations for root body
	{
		RigidBody &body = m_body_list[0];
		// 3.1 update absolute positions and orientations:
		// will be required if we add force elements (eg springs between bodies,
		// or contacts)
		// not required right now, added here for debugging purposes
		body.m_body_pos = body.m_body_T_parent * body.m_parent_pos_parent_body;
		body.m_body_T_world = body.m_body_T_parent;

		if (type >= POSITION_VELOCITY)
		{
			// 3.2 update absolute velocities
			body.m_body_ang_vel = body.m_body_ang_vel_rel;
			body.m_body_vel = body.m_parent_vel_rel;
		}
		if (type >= POSITION_VELOCITY_ACCELERATION)
		{
			// 3.3 update absolute accelerations
			// NOTE: assumption: dot(J_JR) = 0; true here, but not for general joints
			body.m_body_ang_acc = body.m_body_ang_acc_rel;
			body.m_body_acc = body.m_body_T_parent * body.m_parent_acc_rel;
			// add gravitational acceleration to root body
			// this is an efficient way to add gravitational terms,
			// but it does mean that the kinematics are no longer
			// correct at the acceleration level
			// NOTE: To get correct acceleration kinematics, just set world_gravity to zero
			body.m_body_acc = body.m_body_acc - body.m_body_T_parent * m_world_gravity;
		}
	}

	for (idArrayIdx i = 1; i < m_body_list.size(); i++)
	{
		RigidBody &body = m_body_list[i];
		RigidBody &parent = m_body_list[m_parent_index[i]];
		// 2.1 update absolute positions and orientations:
		// will be required if we add force elements (eg springs between bodies,
		// or contacts)  not required right now added here for debugging purposes
		body.m_body_pos =
			body.m_body_T_parent * (parent.m_body_pos + body.m_parent_pos_parent_body);
		body.m_body_T_world = body.m_body_T_parent * parent.m_body_T_world;

		if (type >= POSITION_VELOCITY)
		{
			// 2.2 update absolute velocities
			body.m_body_ang_vel =
				body.m_body_T_parent * parent.m_body_ang_vel + body.m_body_ang_vel_rel;

			body.m_body_vel =
				body.m_body_T_parent *
				(parent.m_body_vel + parent.m_body_ang_vel.cross(body.m_parent_pos_parent_body) +
				 body.m_parent_vel_rel);
		}
		if (type >= POSITION_VELOCITY_ACCELERATION)
		{
			// 2.3 update absolute accelerations
			// NOTE: assumption: dot(J_JR) = 0; true here, but not for general joints
			body.m_body_ang_acc =
				body.m_body_T_parent * parent.m_body_ang_acc -
				body.m_body_ang_vel_rel.cross(body.m_body_T_parent * parent.m_body_ang_vel) +
				body.m_body_ang_acc_rel;
			body.m_body_acc =
				body.m_body_T_parent *
				(parent.m_body_acc + parent.m_body_ang_acc.cross(body.m_parent_pos_parent_body) +
				 parent.m_body_ang_vel.cross(parent.m_body_ang_vel.cross(body.m_parent_pos_parent_body)) +
				 2.0 * parent.m_body_ang_vel.cross(body.m_parent_vel_rel) + body.m_parent_acc_rel);
		}
	}

	return 0;
}

#if (defined BT_ID_HAVE_MAT3X) && (defined BT_ID_WITH_JACOBIANS)

void MultiBodyTree::MultiBodyImpl::addRelativeJacobianComponent(RigidBody &body)
{
	const int &idx = body.m_q_index;
	switch (body.m_joint_type)
	{
		case FIXED:
			break;
		case REVOLUTE:
			setMat3xElem(0, idx, body.m_Jac_JR(0), &body.m_body_Jac_R);
			setMat3xElem(1, idx, body.m_Jac_JR(1), &body.m_body_Jac_R);
			setMat3xElem(2, idx, body.m_Jac_JR(2), &body.m_body_Jac_R);
			break;
		case PRISMATIC:
			setMat3xElem(0, idx, body.m_body_T_parent_ref(0, 0) * body.m_Jac_JT(0) + body.m_body_T_parent_ref(1, 0) * body.m_Jac_JT(1) + body.m_body_T_parent_ref(2, 0) * body.m_Jac_JT(2),
						 &body.m_body_Jac_T);
			setMat3xElem(1, idx, body.m_body_T_parent_ref(0, 1) * body.m_Jac_JT(0) + body.m_body_T_parent_ref(1, 1) * body.m_Jac_JT(1) + body.m_body_T_parent_ref(2, 1) * body.m_Jac_JT(2),
						 &body.m_body_Jac_T);
			setMat3xElem(2, idx, body.m_body_T_parent_ref(0, 2) * body.m_Jac_JT(0) + body.m_body_T_parent_ref(1, 2) * body.m_Jac_JT(1) + body.m_body_T_parent_ref(2, 2) * body.m_Jac_JT(2),
						 &body.m_body_Jac_T);
			break;
		case FLOATING:
			setMat3xElem(0, idx + 0, 1.0, &body.m_body_Jac_R);
			setMat3xElem(1, idx + 1, 1.0, &body.m_body_Jac_R);
			setMat3xElem(2, idx + 2, 1.0, &body.m_body_Jac_R);
			// body_Jac_T = body_T_parent.transpose();
			setMat3xElem(0, idx + 3, body.m_body_T_parent(0, 0), &body.m_body_Jac_T);
			setMat3xElem(0, idx + 4, body.m_body_T_parent(1, 0), &body.m_body_Jac_T);
			setMat3xElem(0, idx + 5, body.m_body_T_parent(2, 0), &body.m_body_Jac_T);

			setMat3xElem(1, idx + 3, body.m_body_T_parent(0, 1), &body.m_body_Jac_T);
			setMat3xElem(1, idx + 4, body.m_body_T_parent(1, 1), &body.m_body_Jac_T);
			setMat3xElem(1, idx + 5, body.m_body_T_parent(2, 1), &body.m_body_Jac_T);

			setMat3xElem(2, idx + 3, body.m_body_T_parent(0, 2), &body.m_body_Jac_T);
			setMat3xElem(2, idx + 4, body.m_body_T_parent(1, 2), &body.m_body_Jac_T);
			setMat3xElem(2, idx + 5, body.m_body_T_parent(2, 2), &body.m_body_Jac_T);

			break;
		case SPHERICAL:
			//todo: review
			setMat3xElem(0, idx + 0, 1.0, &body.m_body_Jac_R);
			setMat3xElem(1, idx + 1, 1.0, &body.m_body_Jac_R);
			setMat3xElem(2, idx + 2, 1.0, &body.m_body_Jac_R);
			break;
	}
}

int MultiBodyTree::MultiBodyImpl::calculateJacobians(const vecx &q, const vecx &u, const KinUpdateType type)
{
	if (q.size() != m_num_dofs || u.size() != m_num_dofs)
	{
		bt_id_error_message(
			"wrong vector dimension. system has %d DOFs,\n"
			"but dim(q)= %d, dim(u)= %d\n",
			m_num_dofs, static_cast<int>(q.size()), static_cast<int>(u.size()));
		return -1;
	}
	if (type != POSITION_ONLY && type != POSITION_VELOCITY)
	{
		bt_id_error_message("invalid type %d\n", type);
		return -1;
	}

	addRelativeJacobianComponent(m_body_list[0]);
	for (idArrayIdx i = 1; i < m_body_list.size(); i++)
	{
		RigidBody &body = m_body_list[i];
		RigidBody &parent = m_body_list[m_parent_index[i]];

		mul(body.m_body_T_parent, parent.m_body_Jac_R, &body.m_body_Jac_R);
		body.m_body_Jac_T = parent.m_body_Jac_T;
		mul(tildeOperator(body.m_parent_pos_parent_body), parent.m_body_Jac_R, &m_m3x);
		sub(body.m_body_Jac_T, m_m3x, &body.m_body_Jac_T);

		addRelativeJacobianComponent(body);
		mul(body.m_body_T_parent, body.m_body_Jac_T, &body.m_body_Jac_T);

		if (type >= POSITION_VELOCITY)
		{
			body.m_body_dot_Jac_R_u = body.m_body_T_parent * parent.m_body_dot_Jac_R_u -
									  body.m_body_ang_vel_rel.cross(body.m_body_T_parent * parent.m_body_ang_vel);
			body.m_body_dot_Jac_T_u = body.m_body_T_parent *
									  (parent.m_body_dot_Jac_T_u + parent.m_body_dot_Jac_R_u.cross(body.m_parent_pos_parent_body) +
									   parent.m_body_ang_vel.cross(parent.m_body_ang_vel.cross(body.m_parent_pos_parent_body)) +
									   2.0 * parent.m_body_ang_vel.cross(body.m_parent_vel_rel));
		}
	}
	return 0;
}
#endif

static inline void setThreeDoFJacobians(const int dof, vec3 &Jac_JR, vec3 &Jac_JT)
{
	switch (dof)
	{
		// rotational part
		case 0:
			Jac_JR(0) = 1;
			Jac_JR(1) = 0;
			Jac_JR(2) = 0;
			setZero(Jac_JT);
			break;
		case 1:
			Jac_JR(0) = 0;
			Jac_JR(1) = 1;
			Jac_JR(2) = 0;
			setZero(Jac_JT);
			break;
		case 2:
			Jac_JR(0) = 0;
			Jac_JR(1) = 0;
			Jac_JR(2) = 1;
			setZero(Jac_JT);
			break;
	}
}

static inline void setSixDoFJacobians(const int dof, vec3 &Jac_JR, vec3 &Jac_JT)
{
	switch (dof)
	{
		// rotational part
		case 0:
			Jac_JR(0) = 1;
			Jac_JR(1) = 0;
			Jac_JR(2) = 0;
			setZero(Jac_JT);
			break;
		case 1:
			Jac_JR(0) = 0;
			Jac_JR(1) = 1;
			Jac_JR(2) = 0;
			setZero(Jac_JT);
			break;
		case 2:
			Jac_JR(0) = 0;
			Jac_JR(1) = 0;
			Jac_JR(2) = 1;
			setZero(Jac_JT);
			break;
		// translational part
		case 3:
			setZero(Jac_JR);
			Jac_JT(0) = 1;
			Jac_JT(1) = 0;
			Jac_JT(2) = 0;
			break;
		case 4:
			setZero(Jac_JR);
			Jac_JT(0) = 0;
			Jac_JT(1) = 1;
			Jac_JT(2) = 0;
			break;
		case 5:
			setZero(Jac_JR);
			Jac_JT(0) = 0;
			Jac_JT(1) = 0;
			Jac_JT(2) = 1;
			break;
	}
}

static inline int jointNumDoFs(const JointType &type)
{
	switch (type)
	{
		case FIXED:
			return 0;
		case REVOLUTE:
		case PRISMATIC:
			return 1;
		case FLOATING:
			return 6;
		case SPHERICAL:
			return 3;
	}
	// this should never happen
	bt_id_error_message("invalid joint type\n");
	// TODO add configurable abort/crash function
	abort();
	return 0;
}

int MultiBodyTree::MultiBodyImpl::calculateMassMatrix(const vecx &q, const bool update_kinematics,
													  const bool initialize_matrix,
													  const bool set_lower_triangular_matrix,
													  matxx *mass_matrix)
{
	// This calculates the joint space mass matrix for the multibody system.
	// The algorithm is essentially an implementation of "method 3"
	// in "Efficient Dynamic Simulation of Robotic Mechanisms" (Walker and Orin, 1982)
	// (Later named "Composite Rigid Body Algorithm" by Featherstone).
	//
	// This implementation, however, handles branched systems and uses a formulation centered
	// on the origin of the body-fixed frame to avoid re-computing various quantities at the com.

	if (q.size() != m_num_dofs || mass_matrix->rows() != m_num_dofs ||
		mass_matrix->cols() != m_num_dofs)
	{
		bt_id_error_message(
			"Dimension error. System has %d DOFs,\n"
			"but dim(q)= %d, dim(mass_matrix)= %d x %d\n",
			m_num_dofs, static_cast<int>(q.size()), static_cast<int>(mass_matrix->rows()),
			static_cast<int>(mass_matrix->cols()));
		return -1;
	}

	// TODO add optimized zeroing function?
	if (initialize_matrix)
	{
		for (int i = 0; i < m_num_dofs; i++)
		{
			for (int j = 0; j < m_num_dofs; j++)
			{
				setMatxxElem(i, j, 0.0, mass_matrix);
			}
		}
	}

	if (update_kinematics)
	{
		// 1. update relative kinematics
		// 1.1 for revolute joints
		for (idArrayIdx i = 0; i < m_body_revolute_list.size(); i++)
		{
			RigidBody &body = m_body_list[m_body_revolute_list[i]];
			// from reference orientation (q=0) of body-fixed frame to current orientation
			mat33 body_T_body_ref;
			bodyTParentFromAxisAngle(body.m_Jac_JR, q(body.m_q_index), &body_T_body_ref);
			body.m_body_T_parent = body_T_body_ref * body.m_body_T_parent_ref;
		}
		// 1.2 for prismatic joints
		for (idArrayIdx i = 0; i < m_body_prismatic_list.size(); i++)
		{
			RigidBody &body = m_body_list[m_body_prismatic_list[i]];
			// body.m_body_T_parent= fixed
			body.m_parent_pos_parent_body =
				body.m_parent_pos_parent_body_ref + body.m_parent_Jac_JT * q(body.m_q_index);
		}
		// 1.3 fixed joints: nothing to do
		// 1.4 6dof joints:
		for (idArrayIdx i = 0; i < m_body_floating_list.size(); i++)
		{
			RigidBody &body = m_body_list[m_body_floating_list[i]];

			body.m_body_T_parent = transformZ(q(body.m_q_index + 2)) *
								   transformY(q(body.m_q_index + 1)) *
								   transformX(q(body.m_q_index));
			body.m_parent_pos_parent_body(0) = q(body.m_q_index + 3);
			body.m_parent_pos_parent_body(1) = q(body.m_q_index + 4);
			body.m_parent_pos_parent_body(2) = q(body.m_q_index + 5);

			body.m_parent_pos_parent_body = body.m_body_T_parent * body.m_parent_pos_parent_body;
		}
	}
	for (int i = m_body_list.size() - 1; i >= 0; i--)
	{
		RigidBody &body = m_body_list[i];
		// calculate mass, center of mass and inertia of "composite rigid body",
		// ie, sub-tree starting at current body
		body.m_subtree_mass = body.m_mass;
		body.m_body_subtree_mass_com = body.m_body_mass_com;
		body.m_body_subtree_I_body = body.m_body_I_body;

		for (idArrayIdx c = 0; c < m_child_indices[i].size(); c++)
		{
			RigidBody &child = m_body_list[m_child_indices[i][c]];
			mat33 body_T_child = child.m_body_T_parent.transpose();

			body.m_subtree_mass += child.m_subtree_mass;
			body.m_body_subtree_mass_com += body_T_child * child.m_body_subtree_mass_com +
											child.m_parent_pos_parent_body * child.m_subtree_mass;
			body.m_body_subtree_I_body +=
				body_T_child * child.m_body_subtree_I_body * child.m_body_T_parent;

			if (child.m_subtree_mass > 0)
			{
				// Shift the reference point for the child subtree inertia using the
				// Huygens-Steiner ("parallel axis") theorem.
				// (First shift from child origin to child com, then from there to this body's
				// origin)
				vec3 r_com = body_T_child * child.m_body_subtree_mass_com / child.m_subtree_mass;
				mat33 tilde_r_child_com = tildeOperator(r_com);
				mat33 tilde_r_body_com = tildeOperator(child.m_parent_pos_parent_body + r_com);
				body.m_body_subtree_I_body +=
					child.m_subtree_mass *
					(tilde_r_child_com * tilde_r_child_com - tilde_r_body_com * tilde_r_body_com);
			}
		}
	}

	for (int i = m_body_list.size() - 1; i >= 0; i--)
	{
		const RigidBody &body = m_body_list[i];

		// determine DoF-range for body
		const int q_index_min = body.m_q_index;
		const int q_index_max = q_index_min + jointNumDoFs(body.m_joint_type) - 1;
		// loop over the DoFs used by this body
		// local joint jacobians (ok as is for 1-DoF joints)
		vec3 Jac_JR = body.m_Jac_JR;
		vec3 Jac_JT = body.m_Jac_JT;
		for (int col = q_index_max; col >= q_index_min; col--)
		{
			// set jacobians for 6-DoF joints
			if (FLOATING == body.m_joint_type)
			{
				setSixDoFJacobians(col - q_index_min, Jac_JR, Jac_JT);
			}
			if (SPHERICAL == body.m_joint_type)
			{
				//todo: review
				setThreeDoFJacobians(col - q_index_min, Jac_JR, Jac_JT);
			}

			vec3 body_eom_rot =
				body.m_body_subtree_I_body * Jac_JR + body.m_body_subtree_mass_com.cross(Jac_JT);
			vec3 body_eom_trans =
				body.m_subtree_mass * Jac_JT - body.m_body_subtree_mass_com.cross(Jac_JR);
			setMatxxElem(col, col, Jac_JR.dot(body_eom_rot) + Jac_JT.dot(body_eom_trans), mass_matrix);

			// rest of the mass matrix column upwards
			{
				// 1. for multi-dof joints, rest of the dofs of this body
				for (int row = col - 1; row >= q_index_min; row--)
				{
					if (SPHERICAL == body.m_joint_type)
					{
						//todo: review
						setThreeDoFJacobians(row - q_index_min, Jac_JR, Jac_JT);
						const double Mrc = Jac_JR.dot(body_eom_rot) + Jac_JT.dot(body_eom_trans);
						setMatxxElem(col, row, Mrc, mass_matrix);
					}
					if (FLOATING == body.m_joint_type)
					{
						setSixDoFJacobians(row - q_index_min, Jac_JR, Jac_JT);
						const double Mrc = Jac_JR.dot(body_eom_rot) + Jac_JT.dot(body_eom_trans);
						setMatxxElem(col, row, Mrc, mass_matrix);
					}
				}
				// 2. ancestor dofs
				int child_idx = i;
				int parent_idx = m_parent_index[i];
				while (parent_idx >= 0)
				{
					const RigidBody &child_body = m_body_list[child_idx];
					const RigidBody &parent_body = m_body_list[parent_idx];

					const mat33 parent_T_child = child_body.m_body_T_parent.transpose();
					body_eom_rot = parent_T_child * body_eom_rot;
					body_eom_trans = parent_T_child * body_eom_trans;
					body_eom_rot += child_body.m_parent_pos_parent_body.cross(body_eom_trans);

					const int parent_body_q_index_min = parent_body.m_q_index;
					const int parent_body_q_index_max =
						parent_body_q_index_min + jointNumDoFs(parent_body.m_joint_type) - 1;
					vec3 Jac_JR = parent_body.m_Jac_JR;
					vec3 Jac_JT = parent_body.m_Jac_JT;
					for (int row = parent_body_q_index_max; row >= parent_body_q_index_min; row--)
					{
						if (SPHERICAL == parent_body.m_joint_type)
						{
							//todo: review
							setThreeDoFJacobians(row - parent_body_q_index_min, Jac_JR, Jac_JT);
						}
						// set jacobians for 6-DoF joints
						if (FLOATING == parent_body.m_joint_type)
						{
							setSixDoFJacobians(row - parent_body_q_index_min, Jac_JR, Jac_JT);
						}
						const double Mrc = Jac_JR.dot(body_eom_rot) + Jac_JT.dot(body_eom_trans);
						setMatxxElem(col, row, Mrc, mass_matrix);
					}

					child_idx = parent_idx;
					parent_idx = m_parent_index[child_idx];
				}
			}
		}
	}

	if (set_lower_triangular_matrix)
	{
		for (int col = 0; col < m_num_dofs; col++)
		{
			for (int row = 0; row < col; row++)
			{
				setMatxxElem(row, col, (*mass_matrix)(col, row), mass_matrix);
			}
		}
	}
	return 0;
}

// utility macro
#define CHECK_IF_BODY_INDEX_IS_VALID(index)                                                  \
	do                                                                                       \
	{                                                                                        \
		if (index < 0 || index >= m_num_bodies)                                              \
		{                                                                                    \
			bt_id_error_message("invalid index %d (num_bodies= %d)\n", index, m_num_bodies); \
			return -1;                                                                       \
		}                                                                                    \
	} while (0)

int MultiBodyTree::MultiBodyImpl::getParentIndex(const int body_index, int *p)
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*p = m_parent_index[body_index];
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getUserInt(const int body_index, int *user_int) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*user_int = m_user_int[body_index];
	return 0;
}
int MultiBodyTree::MultiBodyImpl::getUserPtr(const int body_index, void **user_ptr) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*user_ptr = m_user_ptr[body_index];
	return 0;
}

int MultiBodyTree::MultiBodyImpl::setUserInt(const int body_index, const int user_int)
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	m_user_int[body_index] = user_int;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::setUserPtr(const int body_index, void *const user_ptr)
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	m_user_ptr[body_index] = user_ptr;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getBodyOrigin(int body_index, vec3 *world_origin) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	*world_origin = body.m_body_T_world.transpose() * body.m_body_pos;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getBodyCoM(int body_index, vec3 *world_com) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	if (body.m_mass > 0)
	{
		*world_com = body.m_body_T_world.transpose() *
					 (body.m_body_pos + body.m_body_mass_com / body.m_mass);
	}
	else
	{
		*world_com = body.m_body_T_world.transpose() * (body.m_body_pos);
	}
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getBodyTransform(int body_index, mat33 *world_T_body) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	*world_T_body = body.m_body_T_world.transpose();
	return 0;
}
int MultiBodyTree::MultiBodyImpl::getBodyAngularVelocity(int body_index, vec3 *world_omega) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	*world_omega = body.m_body_T_world.transpose() * body.m_body_ang_vel;
	return 0;
}
int MultiBodyTree::MultiBodyImpl::getBodyLinearVelocity(int body_index,
														vec3 *world_velocity) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	*world_velocity = body.m_body_T_world.transpose() * body.m_body_vel;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getBodyLinearVelocityCoM(int body_index,
														   vec3 *world_velocity) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	vec3 com;
	if (body.m_mass > 0)
	{
		com = body.m_body_mass_com / body.m_mass;
	}
	else
	{
		com(0) = 0;
		com(1) = 0;
		com(2) = 0;
	}

	*world_velocity =
		body.m_body_T_world.transpose() * (body.m_body_vel + body.m_body_ang_vel.cross(com));
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getBodyAngularAcceleration(int body_index,
															 vec3 *world_dot_omega) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	*world_dot_omega = body.m_body_T_world.transpose() * body.m_body_ang_acc;
	return 0;
}
int MultiBodyTree::MultiBodyImpl::getBodyLinearAcceleration(int body_index,
															vec3 *world_acceleration) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	*world_acceleration = body.m_body_T_world.transpose() * body.m_body_acc;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getJointType(const int body_index, JointType *joint_type) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*joint_type = m_body_list[body_index].m_joint_type;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getJointTypeStr(const int body_index,
												  const char **joint_type) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*joint_type = jointTypeToString(m_body_list[body_index].m_joint_type);
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getParentRParentBodyRef(const int body_index, vec3 *r) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*r = m_body_list[body_index].m_parent_pos_parent_body_ref;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getBodyTParentRef(const int body_index, mat33 *T) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*T = m_body_list[body_index].m_body_T_parent_ref;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getBodyAxisOfMotion(const int body_index, vec3 *axis) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	if (m_body_list[body_index].m_joint_type == REVOLUTE)
	{
		*axis = m_body_list[body_index].m_Jac_JR;
		return 0;
	}
	if (m_body_list[body_index].m_joint_type == PRISMATIC)
	{
		*axis = m_body_list[body_index].m_Jac_JT;
		return 0;
	}
	setZero(*axis);
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getDoFOffset(const int body_index, int *q_index) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*q_index = m_body_list[body_index].m_q_index;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::setBodyMass(const int body_index, const idScalar mass)
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	m_body_list[body_index].m_mass = mass;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::setBodyFirstMassMoment(const int body_index,
														 const vec3 &first_mass_moment)
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	m_body_list[body_index].m_body_mass_com = first_mass_moment;
	return 0;
}
int MultiBodyTree::MultiBodyImpl::setBodySecondMassMoment(const int body_index,
														  const mat33 &second_mass_moment)
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	m_body_list[body_index].m_body_I_body = second_mass_moment;
	return 0;
}
int MultiBodyTree::MultiBodyImpl::getBodyMass(const int body_index, idScalar *mass) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*mass = m_body_list[body_index].m_mass;
	return 0;
}
int MultiBodyTree::MultiBodyImpl::getBodyFirstMassMoment(const int body_index,
														 vec3 *first_mass_moment) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*first_mass_moment = m_body_list[body_index].m_body_mass_com;
	return 0;
}
int MultiBodyTree::MultiBodyImpl::getBodySecondMassMoment(const int body_index,
														  mat33 *second_mass_moment) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	*second_mass_moment = m_body_list[body_index].m_body_I_body;
	return 0;
}

void MultiBodyTree::MultiBodyImpl::clearAllUserForcesAndMoments()
{
	for (int index = 0; index < m_num_bodies; index++)
	{
		RigidBody &body = m_body_list[index];
		setZero(body.m_body_force_user);
		setZero(body.m_body_moment_user);
	}
}

int MultiBodyTree::MultiBodyImpl::addUserForce(const int body_index, const vec3 &body_force)
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	m_body_list[body_index].m_body_force_user += body_force;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::addUserMoment(const int body_index, const vec3 &body_moment)
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	m_body_list[body_index].m_body_moment_user += body_moment;
	return 0;
}

#if (defined BT_ID_HAVE_MAT3X) && (defined BT_ID_WITH_JACOBIANS)
int MultiBodyTree::MultiBodyImpl::getBodyDotJacobianTransU(const int body_index, vec3 *world_dot_jac_trans_u) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	*world_dot_jac_trans_u = body.m_body_T_world.transpose() * body.m_body_dot_Jac_T_u;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getBodyDotJacobianRotU(const int body_index, vec3 *world_dot_jac_rot_u) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	*world_dot_jac_rot_u = body.m_body_T_world.transpose() * body.m_body_dot_Jac_R_u;
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getBodyJacobianTrans(const int body_index, mat3x *world_jac_trans) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	mul(body.m_body_T_world.transpose(), body.m_body_Jac_T, world_jac_trans);
	return 0;
}

int MultiBodyTree::MultiBodyImpl::getBodyJacobianRot(const int body_index, mat3x *world_jac_rot) const
{
	CHECK_IF_BODY_INDEX_IS_VALID(body_index);
	const RigidBody &body = m_body_list[body_index];
	mul(body.m_body_T_world.transpose(), body.m_body_Jac_R, world_jac_rot);
	return 0;
}

#endif
}  // namespace btInverseDynamics

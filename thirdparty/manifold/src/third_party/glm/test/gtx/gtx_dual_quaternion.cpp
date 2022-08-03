#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CTOR_INIT
#include <glm/gtx/dual_quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/vector_relational.hpp>
#if GLM_HAS_TRIVIAL_QUERIES
#	include <type_traits>
#endif

int myrand()
{
	static int holdrand = 1;
	return (((holdrand = holdrand * 214013L + 2531011L) >> 16) & 0x7fff);
}

float myfrand() // returns values from -1 to 1 inclusive
{
	return float(double(myrand()) / double( 0x7ffff )) * 2.0f - 1.0f;
}

int test_dquat_type()
{
	glm::dvec3 vA;
	glm::dquat dqA, dqB;
	glm::ddualquat C(dqA, dqB);
	glm::ddualquat B(dqA);
	glm::ddualquat D(dqA, vA);
	return 0;
}

int test_scalars()
{
	float const Epsilon = 0.0001f;

	int Error(0);

	glm::quat src_q1 = glm::quat(1.0f,2.0f,3.0f,4.0f);
	glm::quat src_q2 = glm::quat(5.0f,6.0f,7.0f,8.0f);
	glm::dualquat src1(src_q1,src_q2);

	{
		glm::dualquat dst1 = src1 * 2.0f;
		glm::dualquat dst2 = 2.0f * src1;
		glm::dualquat dst3 = src1;
		dst3 *= 2.0f;
		glm::dualquat dstCmp(src_q1 * 2.0f,src_q2 * 2.0f);
		Error += glm::all(glm::epsilonEqual(dst1.real,dstCmp.real, Epsilon)) && glm::all(glm::epsilonEqual(dst1.dual,dstCmp.dual, Epsilon)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(dst2.real,dstCmp.real, Epsilon)) && glm::all(glm::epsilonEqual(dst2.dual,dstCmp.dual, Epsilon)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(dst3.real,dstCmp.real, Epsilon)) && glm::all(glm::epsilonEqual(dst3.dual,dstCmp.dual, Epsilon)) ? 0 : 1;
	}

	{
		glm::dualquat dst1 = src1 / 2.0f;
		glm::dualquat dst2 = src1;
		dst2 /= 2.0f;
		glm::dualquat dstCmp(src_q1 / 2.0f,src_q2 / 2.0f);
		Error += glm::all(glm::epsilonEqual(dst1.real,dstCmp.real, Epsilon)) && glm::all(glm::epsilonEqual(dst1.dual,dstCmp.dual, Epsilon)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(dst2.real,dstCmp.real, Epsilon)) && glm::all(glm::epsilonEqual(dst2.dual,dstCmp.dual, Epsilon)) ? 0 : 1;
	}
	return Error;
}

int test_inverse() 
{
	int Error(0);

	float const Epsilon = 0.0001f;

	glm::dualquat dqid = glm::dual_quat_identity<float, glm::defaultp>();
	glm::mat4x4 mid(1.0f);

	for (int j = 0; j < 100; ++j)
	{
		glm::mat4x4 rot = glm::yawPitchRoll(myfrand() * 360.0f, myfrand() * 360.0f, myfrand() * 360.0f);
		glm::vec3 vt = glm::vec3(myfrand() * 10.0f, myfrand() * 10.0f, myfrand() * 10.0f);

		glm::mat4x4 m = glm::translate(mid, vt) * rot;

		glm::quat qr = glm::quat_cast(m);

		glm::dualquat dq(qr);

		glm::dualquat invdq = glm::inverse(dq);

		glm::dualquat r1 = invdq * dq;
		glm::dualquat r2 = dq * invdq;

		Error += glm::all(glm::epsilonEqual(r1.real, dqid.real, Epsilon)) && glm::all(glm::epsilonEqual(r1.dual, dqid.dual, Epsilon)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(r2.real, dqid.real, Epsilon)) && glm::all(glm::epsilonEqual(r2.dual, dqid.dual, Epsilon)) ? 0 : 1;

		// testing commutative property
		glm::dualquat r (   glm::quat( myfrand() * glm::pi<float>() * 2.0f, myfrand(), myfrand(), myfrand() ),
							glm::vec3(myfrand() * 10.0f, myfrand() * 10.0f, myfrand() * 10.0f) );
		glm::dualquat riq = (r * invdq) * dq;
		glm::dualquat rqi = (r * dq) * invdq;

		Error += glm::all(glm::epsilonEqual(riq.real, rqi.real, Epsilon)) && glm::all(glm::epsilonEqual(riq.dual, rqi.dual, Epsilon)) ? 0 : 1;
	}

	return Error;
}

int test_mul() 
{
	int Error(0);

	float const Epsilon = 0.0001f;

	glm::mat4x4 mid(1.0f);

	for (int j = 0; j < 100; ++j)
	{
		// generate random rotations and translations and compare transformed by matrix and dualquats random points 
		glm::vec3 vt1 = glm::vec3(myfrand() * 10.0f, myfrand() * 10.0f, myfrand() * 10.0f);
		glm::vec3 vt2 = glm::vec3(myfrand() * 10.0f, myfrand() * 10.0f, myfrand() * 10.0f);

		glm::mat4x4 rot1 = glm::yawPitchRoll(myfrand() * 360.0f, myfrand() * 360.0f, myfrand() * 360.0f);
		glm::mat4x4 rot2 = glm::yawPitchRoll(myfrand() * 360.0f, myfrand() * 360.0f, myfrand() * 360.0f);
		glm::mat4x4 m1 = glm::translate(mid, vt1) * rot1;
		glm::mat4x4 m2 = glm::translate(mid, vt2) * rot2;
		glm::mat4x4 m3 = m2 * m1;
		glm::mat4x4 m4 = m1 * m2;

		glm::quat qrot1 = glm::quat_cast(rot1);
		glm::quat qrot2 = glm::quat_cast(rot2);

		glm::dualquat dq1 = glm::dualquat(qrot1,vt1);
		glm::dualquat dq2 = glm::dualquat(qrot2,vt2);
		glm::dualquat dq3 = dq2 * dq1;
		glm::dualquat dq4 = dq1 * dq2;

		for (int i = 0; i < 100; ++i)
		{
			glm::vec4 src_pt = glm::vec4(myfrand() * 4.0f, myfrand() * 5.0f, myfrand() * 3.0f,1.0f);
			// test both multiplication orders        
			glm::vec4 dst_pt_m3  = m3 * src_pt; 
			glm::vec4 dst_pt_dq3 = dq3 * src_pt;

			glm::vec4 dst_pt_m3_i  = glm::inverse(m3) * src_pt;
			glm::vec4 dst_pt_dq3_i = src_pt * dq3;

			glm::vec4 dst_pt_m4  = m4 * src_pt;
			glm::vec4 dst_pt_dq4 = dq4 * src_pt;

			glm::vec4 dst_pt_m4_i  = glm::inverse(m4) * src_pt;
			glm::vec4 dst_pt_dq4_i = src_pt * dq4;

			Error += glm::all(glm::epsilonEqual(dst_pt_m3, dst_pt_dq3, Epsilon)) ? 0 : 1;
			Error += glm::all(glm::epsilonEqual(dst_pt_m4, dst_pt_dq4, Epsilon)) ? 0 : 1;
			Error += glm::all(glm::epsilonEqual(dst_pt_m3_i, dst_pt_dq3_i, Epsilon)) ? 0 : 1;
			Error += glm::all(glm::epsilonEqual(dst_pt_m4_i, dst_pt_dq4_i, Epsilon)) ? 0 : 1;
		}
	} 

	return Error;
}

int test_dual_quat_ctr()
{
	int Error(0);

#	if GLM_HAS_TRIVIAL_QUERIES
	//	Error += std::is_trivially_default_constructible<glm::dualquat>::value ? 0 : 1;
	//	Error += std::is_trivially_default_constructible<glm::ddualquat>::value ? 0 : 1;
	//	Error += std::is_trivially_copy_assignable<glm::dualquat>::value ? 0 : 1;
	//	Error += std::is_trivially_copy_assignable<glm::ddualquat>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::dualquat>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::ddualquat>::value ? 0 : 1;

		Error += std::is_copy_constructible<glm::dualquat>::value ? 0 : 1;
		Error += std::is_copy_constructible<glm::ddualquat>::value ? 0 : 1;
#	endif

	return Error;
}

int test_size()
{
	int Error = 0;

	Error += 32 == sizeof(glm::dualquat) ? 0 : 1;
	Error += 64 == sizeof(glm::ddualquat) ? 0 : 1;
	Error += glm::dualquat().length() == 2 ? 0 : 1;
	Error += glm::ddualquat().length() == 2 ? 0 : 1;
	Error += glm::dualquat::length() == 2 ? 0 : 1;
	Error += glm::ddualquat::length() == 2 ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_dual_quat_ctr();
	Error += test_dquat_type();
	Error += test_scalars();
	Error += test_inverse();
	Error += test_mul();
	Error += test_size();

	return Error;
}

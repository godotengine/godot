#include <glm/ext/quaternion_exponential.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/quaternion_float_precision.hpp>
#include <glm/ext/quaternion_double.hpp>
#include <glm/ext/quaternion_double_precision.hpp>
#include <glm/ext/quaternion_relational.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float3_precision.hpp>
#include <glm/ext/vector_double3.hpp>
#include <glm/ext/vector_double3_precision.hpp>
#include <glm/ext/scalar_constants.hpp>

template <typename quaType, typename vecType>
int test_log()
{
	typedef typename quaType::value_type T;
	
	T const Epsilon = static_cast<T>(0.001f);
	
	int Error = 0;
	
	quaType const Q(vecType(1, 0, 0), vecType(0, 1, 0));
	quaType const P = glm::log(Q);
	Error += glm::any(glm::notEqual(Q, P, Epsilon)) ? 0 : 1;
	
	quaType const R = glm::exp(P);
	Error += glm::all(glm::equal(Q, R, Epsilon)) ? 0 : 1;

	return Error;
}

template <typename quaType, typename vecType>
int test_pow()
{
	typedef typename quaType::value_type T;
	
	T const Epsilon = static_cast<T>(0.001f);
	
	int Error = 0;
	
	quaType const Q(vecType(1, 0, 0), vecType(0, 1, 0));
	
	{
		T const One = static_cast<T>(1.0f);
		quaType const P = glm::pow(Q, One);
		Error += glm::all(glm::equal(Q, P, Epsilon)) ? 0 : 1;
	}
	
	{
		T const Two = static_cast<T>(2.0f);
		quaType const P = glm::pow(Q, Two);
		quaType const R = Q * Q;
		Error += glm::all(glm::equal(P, R, Epsilon)) ? 0 : 1;

		quaType const U = glm::sqrt(P);
		Error += glm::all(glm::equal(Q, U, Epsilon)) ? 0 : 1;
	}
	
	return Error;
}

int main()
{
	int Error = 0;

	Error += test_log<glm::quat, glm::vec3>();
	Error += test_log<glm::lowp_quat, glm::lowp_vec3>();
	Error += test_log<glm::mediump_quat, glm::mediump_vec3>();
	Error += test_log<glm::highp_quat, glm::highp_vec3>();
	
	Error += test_log<glm::dquat, glm::dvec3>();
	Error += test_log<glm::lowp_dquat, glm::lowp_dvec3>();
	Error += test_log<glm::mediump_dquat, glm::mediump_dvec3>();
	Error += test_log<glm::highp_dquat, glm::highp_dvec3>();

	Error += test_pow<glm::quat, glm::vec3>();
	Error += test_pow<glm::lowp_quat, glm::lowp_vec3>();
	Error += test_pow<glm::mediump_quat, glm::mediump_vec3>();
	Error += test_pow<glm::highp_quat, glm::highp_vec3>();
	
	Error += test_pow<glm::dquat, glm::dvec3>();
	Error += test_pow<glm::lowp_dquat, glm::lowp_dvec3>();
	Error += test_pow<glm::mediump_dquat, glm::mediump_dvec3>();
	Error += test_pow<glm::highp_dquat, glm::highp_dvec3>();
	
	return Error;
}

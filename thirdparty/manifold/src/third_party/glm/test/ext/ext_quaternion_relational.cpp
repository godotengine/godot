#include <glm/gtc/constants.hpp>
#include <glm/ext/quaternion_relational.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/quaternion_float_precision.hpp>
#include <glm/ext/quaternion_double.hpp>
#include <glm/ext/quaternion_double_precision.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float3_precision.hpp>
#include <glm/ext/vector_double3.hpp>
#include <glm/ext/vector_double3_precision.hpp>

template <typename quaType>
static int test_equal()
{
	int Error = 0;

	quaType const Q(1, 0, 0, 0);
	quaType const P(1, 0, 0, 0);
	Error += glm::all(glm::equal(Q, P, glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

template <typename quaType>
static int test_notEqual()
{
	int Error = 0;

	quaType const Q(1, 0, 0, 0);
	quaType const P(1, 0, 0, 0);
	Error += glm::any(glm::notEqual(Q, P, glm::epsilon<float>())) ? 1 : 0;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_equal<glm::quat>();
	Error += test_equal<glm::lowp_quat>();
	Error += test_equal<glm::mediump_quat>();
	Error += test_equal<glm::highp_quat>();

	Error += test_notEqual<glm::quat>();
	Error += test_notEqual<glm::lowp_quat>();
	Error += test_notEqual<glm::mediump_quat>();
	Error += test_notEqual<glm::highp_quat>();

	return Error;
}

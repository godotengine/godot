#include <glm/ext/scalar_constants.hpp>

template <typename valType>
static int test_epsilon()
{
	int Error = 0;

	valType const Test = glm::epsilon<valType>();
	Error += Test > static_cast<valType>(0) ? 0 : 1;

	return Error;
}

template <typename valType>
static int test_pi()
{
	int Error = 0;

	valType const Test = glm::pi<valType>();
	Error += Test > static_cast<valType>(3.14) ? 0 : 1;
	Error += Test < static_cast<valType>(3.15) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_epsilon<float>();
	Error += test_epsilon<double>();
	Error += test_pi<float>();
	Error += test_pi<double>();

	return Error;
}

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/easing.hpp>

namespace
{

	template<typename T>
	void _test_easing()
	{
		T a = static_cast<T>(0.5);
		T r;

		r = glm::linearInterpolation(a);

		r = glm::quadraticEaseIn(a);
		r = glm::quadraticEaseOut(a);
		r = glm::quadraticEaseInOut(a);

		r = glm::cubicEaseIn(a);
		r = glm::cubicEaseOut(a);
		r = glm::cubicEaseInOut(a);

		r = glm::quarticEaseIn(a);
		r = glm::quarticEaseOut(a);
		r = glm::quinticEaseInOut(a);

		r = glm::sineEaseIn(a);
		r = glm::sineEaseOut(a);
		r = glm::sineEaseInOut(a);

		r = glm::circularEaseIn(a);
		r = glm::circularEaseOut(a);
		r = glm::circularEaseInOut(a);

		r = glm::exponentialEaseIn(a);
		r = glm::exponentialEaseOut(a);
		r = glm::exponentialEaseInOut(a);

		r = glm::elasticEaseIn(a);
		r = glm::elasticEaseOut(a);
		r = glm::elasticEaseInOut(a);

		r = glm::backEaseIn(a);
		r = glm::backEaseOut(a);
		r = glm::backEaseInOut(a);

		r = glm::bounceEaseIn(a);
		r = glm::bounceEaseOut(a);
		r = glm::bounceEaseInOut(a);
	}

}

int main()
{
	int Error = 0;

	_test_easing<float>();
	_test_easing<double>();

	return Error;
}


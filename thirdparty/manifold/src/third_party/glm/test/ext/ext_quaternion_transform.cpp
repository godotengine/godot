#include <glm/ext/quaternion_transform.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/scalar_constants.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

static int test_lookAt()
{
	int Error(0);

	glm::vec3 eye(0.0f);
	glm::vec3 center(1.1f, -2.0f, 3.1416f);
	glm::vec3 up(-0.17f, 7.23f, -1.744f);

	glm::quat test_quat = glm::quatLookAt(glm::normalize(center - eye), up);
	glm::quat test_mat = glm::conjugate(glm::quat_cast(glm::lookAt(eye, center, up)));

	Error += static_cast<int>(glm::abs(glm::length(test_quat) - 1.0f) > glm::epsilon<float>());
	Error += static_cast<int>(glm::min(glm::length(test_quat + (-test_mat)), glm::length(test_quat + test_mat)) > glm::epsilon<float>());

	// Test left-handed implementation
	glm::quat test_quatLH = glm::quatLookAtLH(glm::normalize(center - eye), up);
	glm::quat test_matLH = glm::conjugate(glm::quat_cast(glm::lookAtLH(eye, center, up)));
	Error += static_cast<int>(glm::abs(glm::length(test_quatLH) - 1.0f) > glm::epsilon<float>());
	Error += static_cast<int>(glm::min(glm::length(test_quatLH - test_matLH), glm::length(test_quatLH + test_matLH)) > glm::epsilon<float>());
 
	// Test right-handed implementation
	glm::quat test_quatRH = glm::quatLookAtRH(glm::normalize(center - eye), up);
	glm::quat test_matRH = glm::conjugate(glm::quat_cast(glm::lookAtRH(eye, center, up)));
	Error += static_cast<int>(glm::abs(glm::length(test_quatRH) - 1.0f) > glm::epsilon<float>());
	Error += static_cast<int>(glm::min(glm::length(test_quatRH - test_matRH), glm::length(test_quatRH + test_matRH)) > glm::epsilon<float>());

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_lookAt();

	return Error;
}

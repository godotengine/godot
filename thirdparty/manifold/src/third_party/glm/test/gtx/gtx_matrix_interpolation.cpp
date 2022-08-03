#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/matrix_interpolation.hpp>

#include <iostream>
#include <limits>
#include <math.h>


static int test_axisAngle()
{
	int Error = 0;

	glm::mat4 m1(-0.9946f, 0.0f, -0.104531f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.104531f, 0.0f, -0.9946f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	glm::mat4 m2(-0.992624f, 0.0f, -0.121874f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.121874f, 0.0f, -0.992624f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	glm::mat4 const m1rot = glm::extractMatrixRotation(m1);
	glm::mat4 const dltRotation = m2 * glm::transpose(m1rot);

	glm::vec3 dltAxis(0.0f);
	float dltAngle = 0.0f;
	glm::axisAngle(dltRotation, dltAxis, dltAngle);

	std::cout << "dltAxis: (" << dltAxis.x << ", " << dltAxis.y << ", " << dltAxis.z << "), dltAngle: " << dltAngle << std::endl;

	glm::quat q = glm::quat_cast(dltRotation);
	std::cout << "q: (" << q.x << ", " << q.y << ", " << q.z << ", " << q.w << ")" << std::endl;
	float yaw = glm::yaw(q);
	std::cout << "Yaw: " << yaw << std::endl;

	return Error;
}

template <class T>
int testForAxisAngle(glm::vec<3, T, glm::defaultp> const axisTrue, T const angleTrue)
{
    T const eps = std::sqrt(std::numeric_limits<T>::epsilon());

    glm::mat<4, 4, T, glm::defaultp> const matTrue = glm::axisAngleMatrix(axisTrue, angleTrue);

    glm::vec<3, T, glm::defaultp> axis;
    T angle;
    glm::axisAngle(matTrue, axis, angle);
    glm::mat<4, 4, T, glm::defaultp> const matRebuilt = glm::axisAngleMatrix(axis, angle);

    glm::mat<4, 4, T, glm::defaultp> const errMat = matTrue - matRebuilt;
    T const maxErr = glm::compMax(glm::vec<4, T, glm::defaultp>(
            glm::compMax(glm::abs(errMat[0])),
            glm::compMax(glm::abs(errMat[1])),
            glm::compMax(glm::abs(errMat[2])),
            glm::compMax(glm::abs(errMat[3]))
        ));
    
    return maxErr < eps ? 0 : 1;
}

static int test_axisAngle2()
{
	int Error = 0;
    
    Error += testForAxisAngle(glm::vec3(0.0f, 1.0f, 0.0f), 0.0f);
    Error += testForAxisAngle(glm::vec3(0.358f, 0.0716f, 0.9309f), 0.00001f);
    Error += testForAxisAngle(glm::vec3(1.0f, 0.0f, 0.0f), 0.0001f);
    Error += testForAxisAngle(glm::vec3(0.0f, 0.0f, 1.0f), 0.001f);
    Error += testForAxisAngle(glm::vec3(0.0f, 0.0f, 1.0f), 0.001f);
    Error += testForAxisAngle(glm::vec3(0.0f, 1.0f, 0.0f), 0.005f);
    Error += testForAxisAngle(glm::vec3(0.0f, 0.0f, 1.0f), 0.005f);
    Error += testForAxisAngle(glm::vec3(0.358f, 0.0716f, 0.9309f), 0.03f);
    Error += testForAxisAngle(glm::vec3(0.358f, 0.0716f, 0.9309f), 0.0003f);
    Error += testForAxisAngle(glm::vec3(0.0f, 0.0f, 1.0f), 0.01f);
    Error += testForAxisAngle(glm::dvec3(0.0f, 1.0f, 0.0f), 0.00005);
    Error += testForAxisAngle(glm::dvec3(-1.0f, 0.0f, 0.0f), 0.000001);
    Error += testForAxisAngle(glm::dvec3(0.7071f, 0.7071f, 0.0f), 0.5);
    Error += testForAxisAngle(glm::dvec3(0.7071f, 0.0f, 0.7071f), 0.0002);
    Error += testForAxisAngle(glm::dvec3(0.7071f, 0.0f, 0.7071f), 0.00002);
    Error += testForAxisAngle(glm::dvec3(0.7071f, 0.0f, 0.7071f), 0.000002);
    Error += testForAxisAngle(glm::dvec3(0.7071f, 0.0f, 0.7071f), 0.0000002);
    Error += testForAxisAngle(glm::vec3(0.0f, 0.7071f, 0.7071f), 1.3f);
    Error += testForAxisAngle(glm::vec3(0.0f, 0.7071f, 0.7071f), 6.3f);
    Error += testForAxisAngle(glm::vec3(1.0f, 0.0f, 0.0f), -0.23456f);
    Error += testForAxisAngle(glm::vec3(1.0f, 0.0f, 0.0f), glm::pi<float>());
    Error += testForAxisAngle(glm::vec3(0.0f, 1.0f, 0.0f), -glm::pi<float>());
    Error += testForAxisAngle(glm::vec3(0.358f, 0.0716f, 0.9309f), -glm::pi<float>());
    Error += testForAxisAngle(glm::vec3(1.0f, 0.0f, 0.0f), glm::pi<float>() + 2e-6f);
    Error += testForAxisAngle(glm::vec3(1.0f, 0.0f, 0.0f), glm::pi<float>() + 1e-4f);
    Error += testForAxisAngle(glm::vec3(0.0f, 1.0f, 0.0f), -glm::pi<float>() + 1e-3f);
    Error += testForAxisAngle(glm::vec3(0.358f, 0.0716f, 0.9309f), -glm::pi<float>() + 5e-3f);

	return Error;
}

static int test_rotate()
{
	glm::mat4 m2(1.0);
	float myAngle = 1.0f;
	m2 = glm::rotate(m2, myAngle, glm::vec3(1.0f, 0.0f, 0.0f));
	glm::vec3 m2Axis;
	float m2Angle;
	glm::axisAngle(m2, m2Axis, m2Angle);

	return 0;
}

int main()
{
	int Error = 0;

	Error += test_axisAngle();
	Error += test_axisAngle2();
	Error += test_rotate();

	return Error;
}



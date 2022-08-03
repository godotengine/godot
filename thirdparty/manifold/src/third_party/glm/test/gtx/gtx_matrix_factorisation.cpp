#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_factorisation.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/epsilon.hpp>

template <glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
int test_qr(glm::mat<C, R, T, Q> m)
{
	int Error = 0;

	T const epsilon = static_cast<T>(1e-10);

	glm::mat<(C < R ? C : R), R, T, Q> q(-999);
	glm::mat<C, (C < R ? C : R), T, Q> r(-999);

	glm::qr_decompose(m, q, r);

	//Test if q*r really equals the input matrix
	glm::mat<C, R, T, Q> tm = q*r;
	glm::mat<C, R, T, Q> err = tm - m;

	for (glm::length_t i = 0; i < C; i++)
	for (glm::length_t j = 0; j < R; j++)
		Error += glm::abs(err[i][j]) > epsilon ? 1 : 0;

	//Test if the columns of q are orthonormal
	for (glm::length_t i = 0; i < (C < R ? C : R); i++)
	{
		Error += (length(q[i]) - 1) > epsilon ? 1 : 0;

		for (glm::length_t j = 0; j<i; j++)
			Error += glm::abs(dot(q[i], q[j])) > epsilon ? 1 : 0;
	}

	//Test if the matrix r is upper triangular
	for (glm::length_t i = 0; i < C; i++)
	for (glm::length_t j = i + 1; j < (C < R ? C : R); j++)
		Error += glm::epsilonEqual(r[i][j], static_cast<T>(0), glm::epsilon<T>()) ? 0 : 1;

	return Error;
}

template <glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
int test_rq(glm::mat<C, R, T, Q> m)
{
	int Error = 0;

	T const epsilon = static_cast<T>(1e-10);

	glm::mat<C, (C < R ? C : R), T, Q> q(-999);
	glm::mat<(C < R ? C : R), R, T, Q> r(-999);

	glm::rq_decompose(m, r, q);

	//Test if q*r really equals the input matrix
	glm::mat<C, R, T, Q> tm = r*q;
	glm::mat<C, R, T, Q> err = tm - m;

	for (glm::length_t i = 0; i < C; i++)
	for (glm::length_t j = 0; j < R; j++)
		Error += glm::abs(err[i][j]) > epsilon ? 1 : 0;

	//Test if the rows of q are orthonormal
	glm::mat<(C < R ? C : R), C, T, Q> tq = transpose(q);

	for (glm::length_t i = 0; i < (C < R ? C : R); i++)
	{
		Error += (length(tq[i]) - 1) > epsilon ? 1 : 0;

		for (glm::length_t j = 0; j<i; j++)
			Error += glm::abs(dot(tq[i], tq[j])) > epsilon ? 1 : 0;
	}

	//Test if the matrix r is upper triangular
	for (glm::length_t i = 0; i < (C < R ? C : R); i++)
	for (glm::length_t j = R - (C < R ? C : R) + i + 1; j < R; j++)
		Error += glm::epsilonEqual(r[i][j], static_cast<T>(0), glm::epsilon<T>()) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	//Test QR square
	Error += test_qr(glm::dmat3(12.0, 6.0, -4.0, -51.0, 167.0, 24.0, 4.0, -68.0, -41.0)) ? 1 : 0;

	//Test RQ square
	Error += test_rq(glm::dmat3(12.0, 6.0, -4.0, -51.0, 167.0, 24.0, 4.0, -68.0, -41.0)) ? 1 : 0;

	//Test QR triangular 1
	Error += test_qr(glm::dmat3x4(12.0, 6.0, -4.0, -51.0, 167.0, 24.0, 4.0, -68.0, -41.0, 7.0, 2.0, 15.0)) ? 1 : 0;

	//Test QR triangular 2
	Error += test_qr(glm::dmat4x3(12.0, 6.0, -4.0, -51.0, 167.0, 24.0, 4.0, -68.0, -41.0, 7.0, 2.0, 15.0)) ? 1 : 0;

	//Test RQ triangular 1 : Fails at the triangular test
	Error += test_rq(glm::dmat3x4(12.0, 6.0, -4.0, -51.0, 167.0, 24.0, 4.0, -68.0, -41.0, 7.0, 2.0, 15.0)) ? 1 : 0;

	//Test QR triangular 2
	Error += test_rq(glm::dmat4x3(12.0, 6.0, -4.0, -51.0, 167.0, 24.0, 4.0, -68.0, -41.0, 7.0, 2.0, 15.0)) ? 1 : 0;

	return Error;
}

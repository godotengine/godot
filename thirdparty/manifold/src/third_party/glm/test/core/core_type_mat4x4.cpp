#include <glm/gtc/constants.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/matrix_relational.hpp>
#include <glm/matrix.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vector>

template <typename matType, typename vecType>
static int test_operators()
{
	typedef typename matType::value_type value_type;

	value_type const Epsilon = static_cast<value_type>(0.001);

	int Error = 0;

	matType const M(static_cast<value_type>(2.0f));
	matType const N(static_cast<value_type>(1.0f));
	vecType const U(static_cast<value_type>(2.0f));

	{
		matType const P = N * static_cast<value_type>(2.0f);
		Error += glm::all(glm::equal(P, M, Epsilon)) ? 0 : 1;
		
		matType const Q = M / static_cast<value_type>(2.0f);
		Error += glm::all(glm::equal(Q, N, Epsilon)) ? 0 : 1;
	}
	
	{
		vecType const V = M * U;
		Error += glm::all(glm::equal(V, vecType(static_cast<value_type>(4.f)), Epsilon)) ? 0 : 1;
		
		vecType const W = U / M;
		Error += glm::all(glm::equal(W, vecType(static_cast<value_type>(1.f)), Epsilon)) ? 0 : 1;
	}

	{
		matType const O = M * N;
		Error += glm::all(glm::equal(O, matType(static_cast<value_type>(2.f)), Epsilon)) ? 0 : 1;
	}

	return Error;
}

template <typename matType>
static int test_inverse()
{
	typedef typename matType::value_type value_type;

	value_type const Epsilon = static_cast<value_type>(0.001);
	
	int Error = 0;

	matType const Identity(static_cast<value_type>(1.0f));
	matType const Matrix(
		glm::vec4(0.6f, 0.2f, 0.3f, 0.4f),
		glm::vec4(0.2f, 0.7f, 0.5f, 0.3f),
		glm::vec4(0.3f, 0.5f, 0.7f, 0.2f),
		glm::vec4(0.4f, 0.3f, 0.2f, 0.6f));
	matType const Inverse = Identity / Matrix;
	matType const Result = Matrix * Inverse;

	Error += glm::all(glm::equal(Identity, Result, Epsilon)) ? 0 : 1;
	
	return Error;
}

static int test_ctr()
{
	int Error = 0;

#if GLM_HAS_TRIVIAL_QUERIES
	//Error += std::is_trivially_default_constructible<glm::mat4>::value ? 0 : 1;
	//Error += std::is_trivially_copy_assignable<glm::mat4>::value ? 0 : 1;
	Error += std::is_trivially_copyable<glm::mat4>::value ? 0 : 1;
	//Error += std::is_copy_constructible<glm::mat4>::value ? 0 : 1;
	//Error += std::has_trivial_copy_constructor<glm::mat4>::value ? 0 : 1;
#endif

#if GLM_HAS_INITIALIZER_LISTS
	glm::mat4 const m0(
		glm::vec4(0, 1, 2, 3),
		glm::vec4(4, 5, 6, 7),
		glm::vec4(8, 9, 10, 11),
		glm::vec4(12, 13, 14, 15));

	assert(sizeof(m0) == 4 * 4 * 4);

	glm::vec4 const V{0, 1, 2, 3};

	glm::mat4 const m1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

	glm::mat4 const m2{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 10, 11},
		{12, 13, 14, 15}};

	Error += glm::all(glm::equal(m0, m2, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(m1, m2, glm::epsilon<float>())) ? 0 : 1;


	std::vector<glm::mat4> const m3{
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

	glm::mat4 const m4{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1} };

	Error += glm::equal(m4[0][0], 1.0f, 0.0001f) ? 0 : 1;
	Error += glm::equal(m4[3][3], 1.0f, 0.0001f) ? 0 : 1;

	std::vector<glm::mat4> const v1{
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

	std::vector<glm::mat4> const v2{
		{
			{ 0, 1, 2, 3 },
			{ 4, 5, 6, 7 },
			{ 8, 9, 10, 11 },
			{ 12, 13, 14, 15 }
		},
		{
			{ 0, 1, 2, 3 },
			{ 4, 5, 6, 7 },
			{ 8, 9, 10, 11 },
			{ 12, 13, 14, 15 }
		}};

#endif//GLM_HAS_INITIALIZER_LISTS

	return Error;
}

static int test_member_alloc_bug()
{
	int Error = 0;
	
	struct repro
	{
		repro(){ this->matrix = new glm::mat4(); }
		~repro(){delete this->matrix;}
		
		glm::mat4* matrix;
	};
	
	repro Repro;
	
	return Error;
}

static int test_size()
{
	int Error = 0;

	Error += 64 == sizeof(glm::mat4) ? 0 : 1;
	Error += 128 == sizeof(glm::dmat4) ? 0 : 1;
	Error += glm::mat4().length() == 4 ? 0 : 1;
	Error += glm::dmat4().length() == 4 ? 0 : 1;
	Error += glm::mat4::length() == 4 ? 0 : 1;
	Error += glm::dmat4::length() == 4 ? 0 : 1;

	return Error;
}

static int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::mat4::length() == 4, "GLM: Failed constexpr");
	constexpr glm::mat4 A(1.f);
	constexpr glm::mat4 B(1.f);
	constexpr glm::bvec4 C = glm::equal(A, B, 0.01f);
	static_assert(glm::all(C), "GLM: Failed constexpr");
#endif

	return 0;
}

int main()
{
	int Error = 0;

	Error += test_member_alloc_bug();
	Error += test_ctr();

	Error += test_operators<glm::mat4, glm::vec4>();
	Error += test_operators<glm::lowp_mat4, glm::lowp_vec4>();
	Error += test_operators<glm::mediump_mat4, glm::mediump_vec4>();
	Error += test_operators<glm::highp_mat4, glm::highp_vec4>();

	Error += test_operators<glm::dmat4, glm::dvec4>();
	Error += test_operators<glm::lowp_dmat4, glm::lowp_dvec4>();
	Error += test_operators<glm::mediump_dmat4, glm::mediump_dvec4>();
	Error += test_operators<glm::highp_dmat4, glm::highp_dvec4>();

	Error += test_inverse<glm::mat4>();
	Error += test_inverse<glm::lowp_mat4>();
	Error += test_inverse<glm::mediump_mat4>();
	Error += test_inverse<glm::highp_mat4>();

	Error += test_inverse<glm::dmat4>();
	Error += test_inverse<glm::lowp_dmat4>();
	Error += test_inverse<glm::mediump_dmat4>();
	Error += test_inverse<glm::highp_dmat4>();

	Error += test_size();
	Error += test_constexpr();

	return Error;
}

#define GLM_FORCE_SWIZZLE
#include <glm/gtc/constants.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/vector_relational.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vector>

static glm::vec3 g1;
static glm::vec3 g2(1);
static glm::vec3 g3(1, 1, 1);

int test_vec3_ctor()
{
	int Error = 0;

#	if GLM_HAS_TRIVIAL_QUERIES
	//	Error += std::is_trivially_default_constructible<glm::vec3>::value ? 0 : 1;
	//	Error += std::is_trivially_copy_assignable<glm::vec3>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::vec3>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::dvec3>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::ivec3>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::uvec3>::value ? 0 : 1;

		Error += std::is_copy_constructible<glm::vec3>::value ? 0 : 1;
#	endif

#	if GLM_HAS_INITIALIZER_LISTS
	{
		glm::vec3 a{ 0, 1, 2 };
		std::vector<glm::vec3> v = {
			{0, 1, 2},
			{4, 5, 6},
			{8, 9, 0}};
	}

	{
		glm::dvec3 a{ 0, 1, 2 };
		std::vector<glm::dvec3> v = {
			{0, 1, 2},
			{4, 5, 6},
			{8, 9, 0}};
	}
#	endif

	{
		glm::ivec3 A(1);
		glm::ivec3 B(1, 1, 1);
		
		Error += A == B ? 0 : 1;
	}

	{
		std::vector<glm::ivec3> Tests;
		Tests.push_back(glm::ivec3(glm::ivec2(1, 2), 3));
		Tests.push_back(glm::ivec3(1, glm::ivec2(2, 3)));
		Tests.push_back(glm::ivec3(1, 2, 3));
		Tests.push_back(glm::ivec3(glm::ivec4(1, 2, 3, 4)));

		for(std::size_t i = 0; i < Tests.size(); ++i)
			Error += Tests[i] == glm::ivec3(1, 2, 3) ? 0 : 1;
	}

	{
		glm::vec1 const R(1.0f);
		glm::vec1 const S(2.0f);
		glm::vec1 const T(3.0f);
		glm::vec3 const O(1.0f, 2.0f, 3.0f);

		glm::vec3 const A(R);
		glm::vec3 const B(1.0f);
		Error += glm::all(glm::equal(A, B, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const C(R, S, T);
		Error += glm::all(glm::equal(C, O, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const D(R, 2.0f, 3.0f);
		Error += glm::all(glm::equal(D, O, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const E(1.0f, S, 3.0f);
		Error += glm::all(glm::equal(E, O, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const F(1.0f, S, T);
		Error += glm::all(glm::equal(F, O, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const G(R, 2.0f, T);
		Error += glm::all(glm::equal(G, O, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const H(R, S, 3.0f);
		Error += glm::all(glm::equal(H, O, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::vec1 const R(1.0);
		glm::dvec1 const S(2.0);
		glm::vec1 const T(3.0);
		glm::vec3 const O(1.0f, 2.0f, 3.0f);

		glm::vec3 const A(R);
		glm::vec3 const B(1.0);
		Error += glm::all(glm::equal(A, B, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const C(R, S, T);
		Error += glm::all(glm::equal(C, O, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const D(R, 2.0, 3.0);
		Error += glm::all(glm::equal(D, O, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const E(1.0f, S, 3.0);
		Error += glm::all(glm::equal(E, O, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const F(1.0, S, T);
		Error += glm::all(glm::equal(F, O, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const G(R, 2.0, T);
		Error += glm::all(glm::equal(G, O, glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const H(R, S, 3.0);
		Error += glm::all(glm::equal(H, O, glm::epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

float foo()
{
	glm::vec3 bar = glm::vec3(0.0f, 1.0f, 1.0f);

	return glm::length(bar);
}

static int test_bvec3_ctor()
{
	int Error = 0;

	glm::bvec3 const A(true);
	glm::bvec3 const B(true);
	glm::bvec3 const C(false);
	glm::bvec3 const D = A && B;
	glm::bvec3 const E = A && C;
	glm::bvec3 const F = A || C;

	Error += D == glm::bvec3(true) ? 0 : 1;
	Error += E == glm::bvec3(false) ? 0 : 1;
	Error += F == glm::bvec3(true) ? 0 : 1;

	bool const G = A == C;
	bool const H = A != C;
	Error += !G ? 0 : 1;
	Error += H ? 0 : 1;

	return Error;
}

static int test_vec3_operators()
{
	int Error = 0;
	
	{
		glm::ivec3 A(1);
		glm::ivec3 B(1);
		bool R = A != B;
		bool S = A == B;

		Error += (S && !R) ? 0 : 1;
	}

	{
		glm::vec3 const A(1.0f, 2.0f, 3.0f);
		glm::vec3 const B(4.0f, 5.0f, 6.0f);

		glm::vec3 const C = A + B;
		Error += glm::all(glm::equal(C, glm::vec3(5, 7, 9), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const D = B - A;
		Error += glm::all(glm::equal(D, glm::vec3(3, 3, 3), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const E = A * B;
		Error += glm::all(glm::equal(E, glm::vec3(4, 10, 18), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const F = B / A;
		Error += glm::all(glm::equal(F, glm::vec3(4, 2.5, 2), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const G = A + 1.0f;
		Error += glm::all(glm::equal(G, glm::vec3(2, 3, 4), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const H = B - 1.0f;
		Error += glm::all(glm::equal(H, glm::vec3(3, 4, 5), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const I = A * 2.0f;
		Error += glm::all(glm::equal(I, glm::vec3(2, 4, 6), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const J = B / 2.0f;
		Error += glm::all(glm::equal(J, glm::vec3(2, 2.5, 3), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const K = 1.0f + A;
		Error += glm::all(glm::equal(K, glm::vec3(2, 3, 4), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const L = 1.0f - B;
		Error += glm::all(glm::equal(L, glm::vec3(-3, -4, -5), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const M = 2.0f * A;
		Error += glm::all(glm::equal(M, glm::vec3(2, 4, 6), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 const N = 2.0f / B;
		Error += glm::all(glm::equal(N, glm::vec3(0.5, 2.0 / 5.0, 2.0 / 6.0), glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::ivec3 A(1.0f, 2.0f, 3.0f);
		glm::ivec3 B(4.0f, 5.0f, 6.0f);

		A += B;
		Error += A == glm::ivec3(5, 7, 9) ? 0 : 1;

		A += 1;
		Error += A == glm::ivec3(6, 8, 10) ? 0 : 1;
	}
	{
		glm::ivec3 A(1.0f, 2.0f, 3.0f);
		glm::ivec3 B(4.0f, 5.0f, 6.0f);

		B -= A;
		Error += B == glm::ivec3(3, 3, 3) ? 0 : 1;

		B -= 1;
		Error += B == glm::ivec3(2, 2, 2) ? 0 : 1;
	}
	{
		glm::ivec3 A(1.0f, 2.0f, 3.0f);
		glm::ivec3 B(4.0f, 5.0f, 6.0f);

		A *= B;
		Error += A == glm::ivec3(4, 10, 18) ? 0 : 1;

		A *= 2;
		Error += A == glm::ivec3(8, 20, 36) ? 0 : 1;
	}
	{
		glm::ivec3 A(1.0f, 2.0f, 3.0f);
		glm::ivec3 B(4.0f, 4.0f, 6.0f);

		B /= A;
		Error += B == glm::ivec3(4, 2, 2) ? 0 : 1;

		B /= 2;
		Error += B == glm::ivec3(2, 1, 1) ? 0 : 1;
	}
	{
		glm::ivec3 B(2);

		B /= B.y;
		Error += B == glm::ivec3(1) ? 0 : 1;
	}

	{
		glm::ivec3 A(1.0f, 2.0f, 3.0f);
		glm::ivec3 B = -A;
		Error += B == glm::ivec3(-1.0f, -2.0f, -3.0f) ? 0 : 1;
	}

	{
		glm::ivec3 A(1.0f, 2.0f, 3.0f);
		glm::ivec3 B = --A;
		Error += B == glm::ivec3(0.0f, 1.0f, 2.0f) ? 0 : 1;
	}

	{
		glm::ivec3 A(1.0f, 2.0f, 3.0f);
		glm::ivec3 B = A--;
		Error += B == glm::ivec3(1.0f, 2.0f, 3.0f) ? 0 : 1;
		Error += A == glm::ivec3(0.0f, 1.0f, 2.0f) ? 0 : 1;
	}

	{
		glm::ivec3 A(1.0f, 2.0f, 3.0f);
		glm::ivec3 B = ++A;
		Error += B == glm::ivec3(2.0f, 3.0f, 4.0f) ? 0 : 1;
	}

	{
		glm::ivec3 A(1.0f, 2.0f, 3.0f);
		glm::ivec3 B = A++;
		Error += B == glm::ivec3(1.0f, 2.0f, 3.0f) ? 0 : 1;
		Error += A == glm::ivec3(2.0f, 3.0f, 4.0f) ? 0 : 1;
	}

	return Error;
}

int test_vec3_size()
{
	int Error = 0;
	
	Error += sizeof(glm::vec3) == sizeof(glm::lowp_vec3) ? 0 : 1;
	Error += sizeof(glm::vec3) == sizeof(glm::mediump_vec3) ? 0 : 1;
	Error += sizeof(glm::vec3) == sizeof(glm::highp_vec3) ? 0 : 1;
	Error += 12 == sizeof(glm::mediump_vec3) ? 0 : 1;
	Error += sizeof(glm::dvec3) == sizeof(glm::lowp_dvec3) ? 0 : 1;
	Error += sizeof(glm::dvec3) == sizeof(glm::mediump_dvec3) ? 0 : 1;
	Error += sizeof(glm::dvec3) == sizeof(glm::highp_dvec3) ? 0 : 1;
	Error += 24 == sizeof(glm::highp_dvec3) ? 0 : 1;
	Error += glm::vec3().length() == 3 ? 0 : 1;
	Error += glm::dvec3().length() == 3 ? 0 : 1;
	Error += glm::vec3::length() == 3 ? 0 : 1;
	Error += glm::dvec3::length() == 3 ? 0 : 1;

	GLM_CONSTEXPR std::size_t Length = glm::vec3::length();
	Error += Length == 3 ? 0 : 1;

	return Error;
}

int test_vec3_swizzle3_2()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::ivec3 v(1, 2, 3);
		glm::ivec2 u;

		// Can not assign a vec3 swizzle to a vec2
		//u = v.xyz;    //Illegal
		//u = v.rgb;    //Illegal
		//u = v.stp;    //Illegal

		u = v.xx;       Error += (u.x == 1 && u.y == 1) ? 0 : 1;
		u = v.xy;       Error += (u.x == 1 && u.y == 2) ? 0 : 1;
		u = v.xz;       Error += (u.x == 1 && u.y == 3) ? 0 : 1;
		u = v.yx;       Error += (u.x == 2 && u.y == 1) ? 0 : 1;
		u = v.yy;       Error += (u.x == 2 && u.y == 2) ? 0 : 1;
		u = v.yz;       Error += (u.x == 2 && u.y == 3) ? 0 : 1;
		u = v.zx;       Error += (u.x == 3 && u.y == 1) ? 0 : 1;
		u = v.zy;       Error += (u.x == 3 && u.y == 2) ? 0 : 1;
		u = v.zz;       Error += (u.x == 3 && u.y == 3) ? 0 : 1;

		u = v.rr;       Error += (u.r == 1 && u.g == 1) ? 0 : 1;
		u = v.rg;       Error += (u.r == 1 && u.g == 2) ? 0 : 1;
		u = v.rb;       Error += (u.r == 1 && u.g == 3) ? 0 : 1;
		u = v.gr;       Error += (u.r == 2 && u.g == 1) ? 0 : 1;
		u = v.gg;       Error += (u.r == 2 && u.g == 2) ? 0 : 1;
		u = v.gb;       Error += (u.r == 2 && u.g == 3) ? 0 : 1;
		u = v.br;       Error += (u.r == 3 && u.g == 1) ? 0 : 1;
		u = v.bg;       Error += (u.r == 3 && u.g == 2) ? 0 : 1;
		u = v.bb;       Error += (u.r == 3 && u.g == 3) ? 0 : 1;

		u = v.ss;       Error += (u.s == 1 && u.t == 1) ? 0 : 1;
		u = v.st;       Error += (u.s == 1 && u.t == 2) ? 0 : 1;
		u = v.sp;       Error += (u.s == 1 && u.t == 3) ? 0 : 1;
		u = v.ts;       Error += (u.s == 2 && u.t == 1) ? 0 : 1;
		u = v.tt;       Error += (u.s == 2 && u.t == 2) ? 0 : 1;
		u = v.tp;       Error += (u.s == 2 && u.t == 3) ? 0 : 1;
		u = v.ps;       Error += (u.s == 3 && u.t == 1) ? 0 : 1;
		u = v.pt;       Error += (u.s == 3 && u.t == 2) ? 0 : 1;
		u = v.pp;       Error += (u.s == 3 && u.t == 3) ? 0 : 1;
		// Mixed member aliases are not valid
		//u = v.rx;     //Illegal
		//u = v.sy;     //Illegal

		u = glm::ivec2(1, 2);
		v = glm::ivec3(1, 2, 3);
		//v.xx = u;     //Illegal
		v.xy = u;       Error += (v.x == 1 && v.y == 2 && v.z == 3) ? 0 : 1;
		v.xz = u;       Error += (v.x == 1 && v.y == 2 && v.z == 2) ? 0 : 1;
		v.yx = u;       Error += (v.x == 2 && v.y == 1 && v.z == 2) ? 0 : 1;
		//v.yy = u;     //Illegal
		v.yz = u;       Error += (v.x == 2 && v.y == 1 && v.z == 2) ? 0 : 1;
		v.zx = u;       Error += (v.x == 2 && v.y == 1 && v.z == 1) ? 0 : 1;
		v.zy = u;       Error += (v.x == 2 && v.y == 2 && v.z == 1) ? 0 : 1;
		//v.zz = u;     //Illegal
	}
#	endif//GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR

	return Error;
}

int test_vec3_swizzle3_3()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::ivec3 v(1, 2, 3);
		glm::ivec3 u;

		u = v;          Error += (u.x == 1 && u.y == 2 && u.z == 3) ? 0 : 1;

		u = v.xyz;      Error += (u.x == 1 && u.y == 2 && u.z == 3) ? 0 : 1;
		u = v.zyx;      Error += (u.x == 3 && u.y == 2 && u.z == 1) ? 0 : 1;
		u.zyx = v;      Error += (u.x == 3 && u.y == 2 && u.z == 1) ? 0 : 1;

		u = v.rgb;      Error += (u.x == 1 && u.y == 2 && u.z == 3) ? 0 : 1;
		u = v.bgr;      Error += (u.x == 3 && u.y == 2 && u.z == 1) ? 0 : 1;
		u.bgr = v;      Error += (u.x == 3 && u.y == 2 && u.z == 1) ? 0 : 1;

		u = v.stp;      Error += (u.x == 1 && u.y == 2 && u.z == 3) ? 0 : 1;
		u = v.pts;      Error += (u.x == 3 && u.y == 2 && u.z == 1) ? 0 : 1;
		u.pts = v;      Error += (u.x == 3 && u.y == 2 && u.z == 1) ? 0 : 1;
	}
#	endif//GLM_LANG

	return Error;
}

int test_vec3_swizzle_operators()
{
	int Error = 0;

	glm::ivec3 const u = glm::ivec3(1, 2, 3);
	glm::ivec3 const v = glm::ivec3(10, 20, 30);

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::ivec3 q;

		// Swizzle, swizzle binary operators
		q = u.xyz + v.xyz;          Error += (q == (u + v)) ? 0 : 1;
		q = (u.zyx + v.zyx).zyx;    Error += (q == (u + v)) ? 0 : 1;
		q = (u.xyz - v.xyz);        Error += (q == (u - v)) ? 0 : 1;
		q = (u.xyz * v.xyz);        Error += (q == (u * v)) ? 0 : 1;
		q = (u.xxx * v.xxx);        Error += (q == glm::ivec3(u.x * v.x)) ? 0 : 1;
		q = (u.xyz / v.xyz);        Error += (q == (u / v)) ? 0 : 1;

		// vec, swizzle binary operators
		q = u + v.xyz;              Error += (q == (u + v)) ? 0 : 1;
		q = (u - v.xyz);            Error += (q == (u - v)) ? 0 : 1;
		q = (u * v.xyz);            Error += (q == (u * v)) ? 0 : 1;
		q = (u * v.xxx);            Error += (q == v.x * u) ? 0 : 1;
		q = (u / v.xyz);            Error += (q == (u / v)) ? 0 : 1;

		// swizzle,vec binary operators
		q = u.xyz + v;              Error += (q == (u + v)) ? 0 : 1;
		q = (u.xyz - v);            Error += (q == (u - v)) ? 0 : 1;
		q = (u.xyz * v);            Error += (q == (u * v)) ? 0 : 1;
		q = (u.xxx * v);            Error += (q == u.x * v) ? 0 : 1;
		q = (u.xyz / v);            Error += (q == (u / v)) ? 0 : 1;
	}
#	endif//GLM_LANG

	// Compile errors
	//q = (u.yz * v.xyz);
	//q = (u * v.xy);

	return Error;
}

int test_vec3_swizzle_functions()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		// NOTE: template functions cannot pick up the implicit conversion from
		// a swizzle to the unswizzled type, therefore the operator() must be 
		// used.  E.g.:
		//
		// glm::dot(u.xy, v.xy);        <--- Compile error
		// glm::dot(u.xy(), v.xy());    <--- Compiles correctly

		float r;

		// vec2
		glm::vec2 a(1, 2);
		glm::vec2 b(10, 20);
		r = glm::dot(a, b);                 Error += (int(r) == 50) ? 0 : 1;
		r = glm::dot(glm::vec2(a.xy()), glm::vec2(b.xy()));       Error += (int(r) == 50) ? 0 : 1;
		r = glm::dot(glm::vec2(a.xy()), glm::vec2(b.yy()));       Error += (int(r) == 60) ? 0 : 1;

		// vec3
		glm::vec3 u = glm::vec3(1, 2, 3);
		glm::vec3 v = glm::vec3(10, 20, 30);
		r = glm::dot(u, v);                 Error += (int(r) == 140) ? 0 : 1;
		r = glm::dot(u.xyz(), v.zyz());     Error += (int(r) == 160) ? 0 : 1;
		r = glm::dot(u, v.zyx());           Error += (int(r) == 100) ? 0 : 1;
		r = glm::dot(u.xyz(), v);           Error += (int(r) == 140) ? 0 : 1;
		r = glm::dot(u.xy(), v.xy());       Error += (int(r) == 50) ? 0 : 1;

		// vec4
		glm::vec4 s = glm::vec4(1, 2, 3, 4);
		glm::vec4 t = glm::vec4(10, 20, 30, 40);
		r = glm::dot(s, t);                 Error += (int(r) == 300) ? 0 : 1;
		r = glm::dot(s.xyzw(), t.xyzw());   Error += (int(r) == 300) ? 0 : 1;
		r = glm::dot(s.xyz(), t.xyz());     Error += (int(r) == 140) ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION

	return Error;
}

int test_vec3_swizzle_partial()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::vec3 const A(1, 2, 3);
		glm::vec3 B(A.xy, 3);
		Error += glm::all(glm::equal(A, B, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::ivec3 const A(1, 2, 3);
		glm::ivec3 const B(1, A.yz);
		Error += A == B ? 0 : 1;
	}

	{
		glm::ivec3 const A(1, 2, 3);
		glm::ivec3 const B(A.xyz);
		Error += A == B ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR

	return Error;
}

static int test_operator_increment()
{
	int Error = 0;

	glm::ivec3 v0(1);
	glm::ivec3 v1(v0);
	glm::ivec3 v2(v0);
	glm::ivec3 v3 = ++v1;
	glm::ivec3 v4 = v2++;

	Error += glm::all(glm::equal(v0, v4)) ? 0 : 1;
	Error += glm::all(glm::equal(v1, v2)) ? 0 : 1;
	Error += glm::all(glm::equal(v1, v3)) ? 0 : 1;

	int i0(1);
	int i1(i0);
	int i2(i0);
	int i3 = ++i1;
	int i4 = i2++;

	Error += i0 == i4 ? 0 : 1;
	Error += i1 == i2 ? 0 : 1;
	Error += i1 == i3 ? 0 : 1;

	return Error;
}

static int test_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::vec3 A = glm::vec3(1.0f, 2.0f, 3.0f);
		glm::vec3 B = A.xyz;
		glm::vec3 C(A.xyz);
		glm::vec3 D(A.xyz());
		glm::vec3 E(A.x, A.yz);
		glm::vec3 F(A.x, A.yz());
		glm::vec3 G(A.xy, A.z);
		glm::vec3 H(A.xy(), A.z);

		Error += glm::all(glm::equal(A, B, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, D, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, E, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, F, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, G, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, H, glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		glm::vec3 A = glm::vec3(1.0f, 2.0f, 3.0f);
		glm::vec3 B = A.xyz();
		glm::vec3 C(A.xyz());
		glm::vec3 D(A.xyz());
		glm::vec3 E(A.x, A.yz());
		glm::vec3 F(A.x, A.yz());
		glm::vec3 G(A.xy(), A.z);
		glm::vec3 H(A.xy(), A.z);

		Error += glm::all(glm::equal(A, B, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, D, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, E, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, F, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, G, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, H, glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION

	return Error;
}

static int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::vec3::length() == 3, "GLM: Failed constexpr");
	static_assert(glm::vec3(1.0f).x > 0.0f, "GLM: Failed constexpr");
	static_assert(glm::vec3(1.0f, -1.0f, -1.0f).x > 0.0f, "GLM: Failed constexpr");
	static_assert(glm::vec3(1.0f, -1.0f, -1.0f).y < 0.0f, "GLM: Failed constexpr");
#endif

	return 0;
}

int main()
{
	int Error = 0;

	Error += test_vec3_ctor();
	Error += test_bvec3_ctor();
	Error += test_vec3_operators();
	Error += test_vec3_size();
	Error += test_operator_increment();
	Error += test_constexpr();

	Error += test_swizzle();
	Error += test_vec3_swizzle3_2();
	Error += test_vec3_swizzle3_3();
	Error += test_vec3_swizzle_partial();
	Error += test_vec3_swizzle_operators();
	Error += test_vec3_swizzle_functions();

	return Error;
}

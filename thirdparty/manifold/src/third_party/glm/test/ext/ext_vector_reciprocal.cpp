#include <glm/ext/vector_reciprocal.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/ext/vector_double1.hpp>

static int test_sec()
{
	int Error = 0;
	
	glm::dvec1 const a = glm::sec(glm::dvec1(0.0));
	Error += glm::all(glm::equal(a, glm::dvec1(1.0), 0.01)) ? 0 : 1;

	glm::dvec1 const b = glm::sec(glm::dvec1(glm::pi<double>() * 2.0));
	Error += glm::all(glm::equal(b, glm::dvec1(1.0), 0.01)) ? 0 : 1;

	glm::dvec1 const c = glm::sec(glm::dvec1(glm::pi<double>() * -2.0));
	Error += glm::all(glm::equal(c, glm::dvec1(1.0), 0.01)) ? 0 : 1;

	glm::dvec1 const d = glm::sec(glm::dvec1(glm::pi<double>() * 1.0));
	Error += glm::all(glm::equal(d, -glm::dvec1(1.0), 0.01)) ? 0 : 1;

	glm::dvec1 const e = glm::sec(glm::dvec1(glm::pi<double>() * -1.0));
	Error += glm::all(glm::equal(e, -glm::dvec1(1.0), 0.01)) ? 0 : 1;

	return Error;
}

static int test_csc()
{
	int Error = 0;
	
	glm::dvec1 const a = glm::csc(glm::dvec1(glm::pi<double>() * 0.5));
	Error += glm::all(glm::equal(a, glm::dvec1(1.0), 0.01)) ? 0 : 1;

	glm::dvec1 const b = glm::csc(glm::dvec1(glm::pi<double>() * -0.5));
	Error += glm::all(glm::equal(b, glm::dvec1(-1.0), 0.01)) ? 0 : 1;

	return Error;
}

static int test_cot()
{
	int Error = 0;
	
	glm::dvec1 const a = glm::cot(glm::dvec1(glm::pi<double>() * 0.5));
	Error += glm::all(glm::equal(a, glm::dvec1(0.0), 0.01)) ? 0 : 1;

	glm::dvec1 const b = glm::cot(glm::dvec1(glm::pi<double>() * -0.5));
	Error += glm::all(glm::equal(b, glm::dvec1(0.0), 0.01)) ? 0 : 1;

	return Error;
}

static int test_asec()
{
	int Error = 0;
	
	Error += glm::all(glm::equal(glm::asec(glm::dvec1(100000.0)), glm::dvec1(glm::pi<double>() * 0.5), 0.01)) ? 0 : 1;
	Error += glm::all(glm::equal(glm::asec(glm::dvec1(-100000.0)), glm::dvec1(glm::pi<double>() * 0.5), 0.01)) ? 0 : 1;

	return Error;
}

static int test_acsc()
{
	int Error = 0;
	
	Error += glm::all(glm::equal(glm::acsc(glm::dvec1(100000.0)), glm::dvec1(0.0), 0.01)) ? 0 : 1;
	Error += glm::all(glm::equal(glm::acsc(glm::dvec1(-100000.0)), glm::dvec1(0.0), 0.01)) ? 0 : 1;

	return Error;
}

static int test_acot()
{
	int Error = 0;
	
	Error += glm::all(glm::equal(glm::acot(glm::dvec1(100000.0)), glm::dvec1(0.0), 0.01)) ? 0 : 1;
	Error += glm::all(glm::equal(glm::acot(glm::dvec1(-100000.0)), glm::dvec1(glm::pi<double>()), 0.01)) ? 0 : 1;
	Error += glm::all(glm::equal(glm::acot(glm::dvec1(0.0)), glm::dvec1(glm::pi<double>() * 0.5), 0.01)) ? 0 : 1;

	return Error;
}

static int test_sech()
{
	int Error = 0;
	
	Error += glm::all(glm::equal(glm::sech(glm::dvec1(100000.0)), glm::dvec1(0.0), 0.01)) ? 0 : 1;
	Error += glm::all(glm::equal(glm::sech(glm::dvec1(-100000.0)), glm::dvec1(0.0), 0.01)) ? 0 : 1;
	Error += glm::all(glm::equal(glm::sech(glm::dvec1(0.0)), glm::dvec1(1.0), 0.01)) ? 0 : 1;

	return Error;
}

static int test_csch()
{
	int Error = 0;
	
	Error += glm::all(glm::equal(glm::csch(glm::dvec1(100000.0)), glm::dvec1(0.0), 0.01)) ? 0 : 1;
	Error += glm::all(glm::equal(glm::csch(glm::dvec1(-100000.0)), glm::dvec1(0.0), 0.01)) ? 0 : 1;

	return Error;
}

static int test_coth()
{
	int Error = 0;
	
	glm::dvec1 const a = glm::coth(glm::dvec1(100.0));
	Error += glm::all(glm::equal(a, glm::dvec1(1.0), 0.01)) ? 0 : 1;
	
	glm::dvec1 const b = glm::coth(glm::dvec1(-100.0));
	Error += glm::all(glm::equal(b, glm::dvec1(-1.0), 0.01)) ? 0 : 1;

	return Error;
}

static int test_asech()
{
	int Error = 0;
	
	glm::dvec1 const a = glm::asech(glm::dvec1(1.0));
	Error += glm::all(glm::equal(a, glm::dvec1(0.0), 0.01)) ? 0 : 1;

	return Error;
}

static int test_acsch()
{
	int Error = 0;
	
	glm::dvec1 const a(glm::acsch(glm::dvec1(0.01)));
	Error += a.x > 1.0 ? 0 : 1;

	glm::dvec1 const b(glm::acsch(glm::dvec1(-0.01)));
	Error += b.x < -1.0 ? 0 : 1;

	Error += glm::all(glm::equal(glm::acsch(glm::dvec1(100.0)), glm::dvec1(0.0), 0.01)) ? 0 : 1;
	Error += glm::all(glm::equal(glm::acsch(glm::dvec1(-100.0)), glm::dvec1(0.0), 0.01)) ? 0 : 1;

	return Error;
}

static int test_acoth()
{
	int Error = 0;
	
	glm::dvec1 const a = glm::acoth(glm::dvec1(1.00001));
	Error += a.x > 6.0 ? 0 : 1;
	
	glm::dvec1 const b = glm::acoth(glm::dvec1(-1.00001));
	Error += b.x < -6.0 ? 0 : 1;

	glm::dvec1 const c = glm::acoth(glm::dvec1(10000.0));
	Error += glm::all(glm::equal(c, glm::dvec1(0.0), 0.01)) ? 0 : 1;
	
	glm::dvec1 const d = glm::acoth(glm::dvec1(-10000.0));
	Error += glm::all(glm::equal(d, glm::dvec1(0.0), 0.01)) ? 0 : 1;

	return Error;
}


int main()
{
	int Error = 0;
	
	Error += test_sec();
	Error += test_csc();
	Error += test_cot();

	Error += test_asec();
	Error += test_acsc();
	Error += test_acot();

	Error += test_sech();
	Error += test_csch();
	Error += test_coth();

	Error += test_asech();
	Error += test_acsch();
	Error += test_acoth();

	return Error;
}

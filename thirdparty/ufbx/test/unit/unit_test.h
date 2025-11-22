#pragma once

#include <stdint.h>
#include <stddef.h>
#include <exception>

struct ufbxwt_unit_fail : std::exception {
	ufbxwt_unit_fail(const char *expr)
		: expr(expr)
	{
	}
	
	const char *expr;

    virtual char const* what() const noexcept override {
		return expr;
	}
};

#define ufbxwt_assert(cond) do { \
		if (!(cond)) { \
			throw ufbxwt_unit_fail(#cond); \
		} \
	} while(0)

typedef void ufbxwt_unit_test_fn();

struct ufbxwt_unit_test {
	const char *name;
	const char *category;
	ufbxwt_unit_test_fn *fn;
	ufbxwt_unit_test *next;
	uint32_t serial;

	static uint32_t s_serial;
	static ufbxwt_unit_test *s_root;

	ufbxwt_unit_test(const char *name, const char *category, ufbxwt_unit_test_fn *fn)
		: name(name), category(category), fn(fn), next(nullptr), serial(s_serial++)
	{
		next = s_root;
		s_root = this;
	}
};

#define UFBXWT_UNIT_TEST(name) \
	static void ufbxwt_unit_##name(); \
	ufbxwt_unit_test s_test_##name { #name, UFBXWT_UNIT_CATEGORY, &ufbxwt_unit_##name }; \
	static void ufbxwt_unit_##name()


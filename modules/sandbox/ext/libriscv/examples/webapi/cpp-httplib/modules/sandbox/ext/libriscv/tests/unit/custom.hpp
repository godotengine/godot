#pragma once
#include <stddef.h>

#define ERROR_NO_SUCH_FUNCTION    1
#define ERROR_MISSING_ARGUMENTS   2
#define STRING_BUFFER_SIZE       64

enum ArgType {
	SIGNED_INT = 0,
	UNSIGNED_INT,
	FLOAT_32,
	FLOAT_64,
	STRING,
	ERROR
};

struct SystemArg {
	union {
		signed int   i32;
		unsigned int u32;
		float        f32;
		double       f64;
		char         string[STRING_BUFFER_SIZE];
	};
	unsigned type;
};
struct SystemFunctionArgs
{
	struct SystemArg arg[4];
};

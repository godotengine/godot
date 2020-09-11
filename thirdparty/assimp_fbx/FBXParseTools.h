#ifndef _H_FBX_PARSE_TOOLS
#define _H_FBX_PARSE_TOOLS

#include <stdint.h>
#include <algorithm>
#include <locale>

template <class char_t>
inline bool IsNewLine(char_t c) {
	return c == '\n' || c == '\r';
}
template <class char_t>
inline bool IsSpace(char_t c) {
	return (c == (char_t)' ' || c == (char_t)'\t');
}

template <class char_t>
inline bool IsSpaceOrNewLine(char_t c) {
	return IsNewLine(c) || IsSpace(c);
}

template <class char_t>
inline bool IsLineEnd(char_t c) {
	return (c == (char_t)'\r' || c == (char_t)'\n' || c == (char_t)'\0' || c == (char_t)'\f');
}

// ------------------------------------------------------------------------------------
// Special version of the function, providing higher accuracy and safety
// It is mainly used by fast_atof to prevent ugly and unwanted integer overflows.
// ------------------------------------------------------------------------------------
inline uint64_t strtoul10_64(const char *in, const char **out = 0, unsigned int *max_inout = 0) {
	unsigned int cur = 0;
	uint64_t value = 0;

	if (*in < '0' || *in > '9') {
		throw std::invalid_argument(std::string("The string \"") + in + "\" cannot be converted into a value.");
	}

	for (;;) {
		if (*in < '0' || *in > '9') {
			break;
		}

		const uint64_t new_value = (value * (uint64_t)10) + ((uint64_t)(*in - '0'));

		// numeric overflow, we rely on you
		if (new_value < value) {
			//WARN_PRINT( "Converting the string \" " + in + " \" into a value resulted in overflow." );
			return 0;
		}

		value = new_value;

		++in;
		++cur;

		if (max_inout && *max_inout == cur) {
			if (out) { /* skip to end */
				while (*in >= '0' && *in <= '9') {
					++in;
				}
				*out = in;
			}

			return value;
		}
	}
	if (out) {
		*out = in;
	}

	if (max_inout) {
		*max_inout = cur;
	}

	return value;
}

#endif // _H_FBX_PARSE_TOOLS
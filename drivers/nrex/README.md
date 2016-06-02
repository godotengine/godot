# NREX: Node RegEx

[![Build Status](https://travis-ci.org/leezh/nrex.svg?branch=master)](https://travis-ci.org/leezh/nrex)

** Version 0.2 **

Small node-based regular expression library. It only does text pattern
matchhing, not replacement. To use add the files `nrex.hpp`, `nrex.cpp`
and `nrex_config.h` to your project and follow the example:

	nrex regex;
	regex.compile("^(fo+)bar$");

	nrex_result captures[regex.capture_size()];
	if (regex.match("foobar", captures))
	{
		std::cout << captures[0].start << std::endl;
		std::cout << captures[0].length << std::endl;
	}

More details about its use is documented in `nrex.hpp`

Currently supported features:
 * Capturing `()` and non-capturing `(?:)` groups
 * Any character `.` (includes newlines)
 * Shorthand caracter classes `\w\W\s\S\d\D`
 * POSIX character classes such as `[[:alnum:]]`
 * Bracket expressions such as `[A-Za-z]`
 * Simple quantifiers `?`, `*` and `+`
 * Range quantifiers `{0,1}`
 * Lazy (non-greedy) quantifiers `*?`
 * Begining `^` and end `$` anchors
 * Word boundaries `\b`
 * Alternation `|`
 * ASCII `\xFF` code points
 * Unicode `\uFFFF` code points
 * Positive `(?=)` and negative `(?!)` lookahead
 * Positive `(?<=)` and negative `(?<!)` lookbehind (fixed length and no alternations)
 * Backreferences `\1` and `\g{1}` (limited by default to 9 - can be unlimited)

## License

Copyright (c) 2015-2016, Zher Huei Lee
All rights reserved.

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

 1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would
    be appreciated but is not required.
    
 2. Altered source versions must be plainly marked as such, and must not
    be misrepresented as being the original software.
    
 3. This notice may not be removed or altered from any source
    distribution.


# Changes

## Version 0.2 (2016-08-04)
 * Fixed capturing groups matching to invalid results
 * Fixed parents of recursive quantifiers not expanding properly
 * Fixed LookAhead sometimes adding to result
 * More verbose unit testing

## Version 0.1 (2015-12-04)
 * Initial release

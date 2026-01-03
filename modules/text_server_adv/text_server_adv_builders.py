"""Functions used to generate source files during build time"""

import methods


def make_icu_data(target, source, env):
    buffer = methods.get_buffer(str(source[0]))
    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
/* (C) 2016 and later: Unicode, Inc. and others. */
/* License & terms of use: https://www.unicode.org/copyright.html */

#include <unicode/utypes.h>
#include <unicode/udata.h>
#include <unicode/uversion.h>

extern "C" U_EXPORT const size_t U_ICUDATA_SIZE = {len(buffer)};
extern "C" U_EXPORT const unsigned char U_ICUDATA_ENTRY_POINT[] = {{
	{methods.format_buffer(buffer, 1)}
}};
""")


def make_icu_data_base(target, source, env):
    buffer = methods.get_buffer(str(source[0]))
    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
/* (C) 2016 and later: Unicode, Inc. and others. */
/* License & terms of use: https://www.unicode.org/copyright.html */

#include <unicode/utypes.h>
#include <unicode/udata.h>
#include <unicode/uversion.h>

extern "C" U_EXPORT const size_t U_ICUDATA_SIZE_BASE = {len(buffer)};
extern "C" U_EXPORT const unsigned char U_ICUDATA_ENTRY_POINT_BASE[] = {{
	{methods.format_buffer(buffer, 1)}
}};
""")

"""Functions used to generate source files during build time"""

import re

import methods

RE_PATH_SPLIT = re.compile(r"[^/\\]+?(?=\.)")

TEMPLATE = """\
#ifndef _WIN32
#define TEST_DLL_PRIVATE __attribute__((visibility("hidden")))
#else
#define TEST_DLL_PRIVATE
#endif // _WIN32

namespace ForceLink {{
	TEST_DLL_PRIVATE void force_link_tests();
	{TESTS_DECLARE}
}} // namespace ForceLink

void ForceLink::force_link_tests() {{
	{TESTS_CALL}
}}
"""


def force_link_builder(target, source, env):
    names = [RE_PATH_SPLIT.search(str(path)).group() for path in source[0].read()]
    declares = [f"TEST_DLL_PRIVATE void force_link_{name}();" for name in names]
    calls = [f"force_link_{name}();" for name in names]

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(
            TEMPLATE.format(
                TESTS_DECLARE="\n\t".join(declares),
                TESTS_CALL="\n\t".join(calls),
            )
        )

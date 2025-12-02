#include "unit_test.h"

#include <string.h>
#include <stdio.h>

#include <vector>
#include <algorithm>

uint32_t ufbxwt_unit_test::s_serial = 0;
ufbxwt_unit_test *ufbxwt_unit_test::s_root = nullptr;

std::vector<ufbxwt_unit_test*> collect_tests()
{
    ufbxwt_unit_test *test = ufbxwt_unit_test::s_root;

    std::vector<ufbxwt_unit_test*> result;
    for (; test; test = test->next) {
        result.push_back(test);
    }
    return result;
}

int main(int argc, char **argv)
{
    const char *test_filter = nullptr;
    const char *test_group = nullptr;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--test")) {
			if (++i < argc) {
				test_filter = argv[i];
			}
		}
		if (!strcmp(argv[i], "-g") || !strcmp(argv[i], "--group")) {
			if (++i < argc) {
				test_group = argv[i];
			}
		}
	}

    std::vector<ufbxwt_unit_test*> tests = collect_tests();

    std::sort(tests.begin(), tests.end(), [](ufbxwt_unit_test *a, ufbxwt_unit_test *b) {
        int cat = strcmp(a->category, b->category);
        if (cat != 0) return cat < 0;
        return a->serial < b->serial;
    });

    uint32_t num_ok = 0;
    uint32_t num_ran = 0;
    
    for (const ufbxwt_unit_test *test : tests) {
        if (test_filter && strcmp(test->name, test_filter) != 0) {
            continue;
        }
        if (test_group && strcmp(test->category, test_group) != 0) {
            continue;
        }

		num_ran++;
        try {
			printf("%s: ", test->name);
            test->fn();
            printf("OK\n");
            num_ok++;
        } catch (const ufbxwt_unit_fail &fail) {
            printf("FAIL (%s)\n", fail.expr);
        }
    }

	printf("\nUnit tests passed: %u/%u\n", num_ok, num_ran);

	return num_ok == num_ran ? 0 : 1;
}

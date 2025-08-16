#include <cstdio>

extern void test_custom_machine();
extern void test_crashes();
extern void test_rv32i();
extern void test_rv32c();

int main()
{
	test_custom_machine();

	printf("* Test crashes\n");
	test_crashes();
	printf("* Test RV32I\n");
	test_rv32i();
	printf("* Test RV32C\n");
	test_rv32c();
	printf("Tests passed!\n");
	return 0;
}

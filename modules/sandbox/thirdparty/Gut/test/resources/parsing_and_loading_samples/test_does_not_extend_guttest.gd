# This file should be ignored by add_script since it does not extend GutTest
func before_all():
    print("should be ignored")


# This class matches the default prefix but should be ignored because it does
# not extend GutTest
class TestDoesNotExtendTest:

    func before_all():
        print("should be ignored")


# This class should be ignored because the outer script does not extend GutTest.
class TestExtendsButShouldBeIgnored:
    extends GutTest

    func before_all():
        print("should be ignored")
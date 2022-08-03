#include <thrust/version.h>
#include <iostream>

int main(void)
{
    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;
    int subminor = THRUST_SUBMINOR_VERSION;
    int patch = THRUST_PATCH_NUMBER;

    std::cout << "Thrust v" << major << "." << minor << "." << subminor << "-" << patch << std::endl;

    return 0;
}


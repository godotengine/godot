#include <thrust/detail/config.h>

#include <iostream>

int main()
{
  std::cout << "Hello from Thrust version " << THRUST_VERSION << ":\n"

            << "Host system: "
#if THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_CPP
            << "CPP\n"
#elif THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_OMP
            << "OMP\n"
#elif THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_TBB
            << "TBB\n"
#else
            << "Unknown\n"
#endif

            << "Device system: "
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP
            << "CPP\n";
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
            << "CUDA\n";
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
            << "OMP\n";
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
            << "TBB\n";
#else
            << "Unknown\n";
#endif
}

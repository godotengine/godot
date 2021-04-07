#include <chrono>

#include "Timing.hpp"

uint64_t GetTime()
{
    return std::chrono::time_point_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() ).time_since_epoch().count();
}

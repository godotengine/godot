#ifndef EFSW_SYSTEMIMPLPOSIX_HPP
#define EFSW_SYSTEMIMPLPOSIX_HPP

#include <efsw/base.hpp>

#if defined( EFSW_PLATFORM_POSIX )

namespace efsw { namespace Platform {

class System {
  public:
	static void sleep( const unsigned long& ms );

	static std::string getProcessPath();

	static void maxFD();

	static Uint64 getMaxFD();
};

}} // namespace efsw::Platform

#endif

#endif

#ifndef EFSW_SYSTEMIMPLWIN_HPP
#define EFSW_SYSTEMIMPLWIN_HPP

#include <efsw/base.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_WIN32

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

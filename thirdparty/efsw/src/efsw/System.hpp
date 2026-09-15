#ifndef EFSW_SYSTEM_HPP
#define EFSW_SYSTEM_HPP

#include <efsw/base.hpp>

namespace efsw {

class System {
  public:
	/// Sleep for x milliseconds
	static void sleep( const unsigned long& ms );

	/// @return The process binary path
	static std::string getProcessPath();

	/// Maximize the number of file descriptors allowed per process in the current OS
	static void maxFD();

	/// @return The number of supported file descriptors for the process
	static Uint64 getMaxFD();
};

} // namespace efsw

#endif

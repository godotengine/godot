#include "../common.hpp"

#include <cstring>
#include "dlfcn.h"

namespace riscv
{
	void* compile(const std::string& code, int arch, const std::string& cflags, const std::string& outfile)
	{
		(void)arch;
		(void)cflags;
		(void)outfile;
		(void)code;

		return nullptr;
	}

	bool
	mingw_compile(const std::string& code, int arch, const std::string& cflags,
		const std::string& outfile, const MachineTranslationCrossOptions& cross_options)
	{
		(void)arch;
		(void)cflags;
		(void)outfile;
		(void)code;
		(void)cross_options;

		return false;
	}

	void* dylib_lookup(void* dylib, const char* symbol, bool)
	{
		return dlsym(dylib, symbol);
	}

	void dylib_close(void* dylib, bool)
	{
		dlclose(dylib);
	}
}

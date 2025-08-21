#include "common.hpp"

#include <cstring>
#if defined(__MINGW32__) || defined(__MINGW64__) || defined(_MSC_VER)
#include "win32/dlfcn.h"
#else
#include <dlfcn.h>
#endif
#include <unistd.h>

static std::string compiler()
{
	const char* cc = getenv("CC");
	if (cc) return std::string(cc);
	return "cc";
}
static std::string extra_cflags()
{
	const char* cflags = getenv("CFLAGS");
	if (cflags) return std::string(cflags);
	return "";
}
static bool keep_code()
{
	return getenv("KEEPCODE") != nullptr;
}
static bool verbose()
{
	return getenv("VERBOSE") != nullptr;
}
static std::string host_arch()
{
#ifdef __x86_64__
	return "HOST_AMD64";
#else
	return "HOST_UNKNOWN";
#endif
}

namespace riscv
{
	std::string compile_command(int /*arch*/, const std::string& cflags)
	{
		return compiler() + " -O2 -s -std=c99 -fPIC -shared -rdynamic -x c "
			" -fexceptions -fvisibility=hidden -fomit-frame-pointer " +
#ifdef RISCV_EXT_VECTOR
			" -march=native" +
#endif
			cflags +
			" -DARCH=" + host_arch() + ""
			" -pipe " + extra_cflags();
	}

	void*
	compile(const std::string& code, int arch, const std::string& cflags,
		const std::string& outfile)
	{
		// create temporary filename
		char namebuffer[64];
		strncpy(namebuffer, "/tmp/rvtrcode-XXXXXX", sizeof(namebuffer));
		// open a temporary file with owner privs
		const int fd = mkstemp(namebuffer);
		if (fd < 0) {
			return nullptr;
		}
		// write translated code to temp file
		ssize_t len = write(fd, code.c_str(), code.size());
		if (len < (ssize_t) code.size()) {
			unlink(namebuffer);
			return nullptr;
		}
		// system compiler invocation
		const std::string command =
			compile_command(arch, cflags) + " "
			 + " -o " + outfile + " "
			 + std::string(namebuffer) + " 2>&1"; // redirect stderr

		// compile the translated code
		if (verbose()) {
			printf("Command: %s\n", command.c_str());
		}
		FILE* f = popen(command.c_str(), "r");
		if (f == nullptr) {
			unlink(namebuffer);
			return nullptr;
		}
		// get compiler output
		char buffer[2048];
		while (fgets(buffer, sizeof(buffer), f) != NULL) {
			if (verbose())
				fprintf(stderr, "%s", buffer);
		}
		pclose(f);

		if (!keep_code()) {
			// delete temporary code file
			unlink(namebuffer);
		}

		return dlopen(outfile.c_str(), RTLD_LAZY);
	}

	static std::string mingw_compile_command(int /*arch*/,
		const std::string& cflags, const MachineTranslationCrossOptions& cross_options)
	{
		// We always want to produce a generic PE-dll that can be loaded on *most* Windows machines.
		return cross_options.cross_compiler + " -O2 -s -std=c99 -fPIC -shared -x c "
			" -fexceptions -fvisibility=hidden -fomit-frame-pointer " +
			cflags +
			" -DARCH=" + host_arch() + ""
			" -pipe " + extra_cflags();
	}

	bool
	mingw_compile(const std::string& code, int arch, const std::string& cflags,
		const std::string& outfile, const MachineTranslationCrossOptions& cross_options)
	{
		// create temporary filename
		char namebuffer[64];
		strncpy(namebuffer, "/tmp/rvtrcode-XXXXXX", sizeof(namebuffer));
		// open a temporary file with owner privs
		const int fd = mkstemp(namebuffer);
		if (fd < 0) {
			return false;
		}
		// write translated code to temp file
		ssize_t len = write(fd, code.c_str(), code.size());
		if (len < (ssize_t) code.size()) {
			unlink(namebuffer);
			return false;
		}
		// system compiler invocation
		const std::string command =
			mingw_compile_command(arch, cflags, cross_options) + " "
			 + " -o " + outfile + " "
			 + std::string(namebuffer) + " 2>&1"; // redirect stderr

		// compile the translated code
		if (verbose()) {
			printf("MinGW Command: %s\n", command.c_str());
		}
		FILE* f = popen(command.c_str(), "r");
		if (f == nullptr) {
			unlink(namebuffer);
			return false;
		}
		// get compiler output
		char buffer[2048];
		while (fgets(buffer, sizeof(buffer), f) != NULL) {
			if (verbose())
				fprintf(stderr, "%s", buffer);
		}
		pclose(f);

		if (!keep_code()) {
			// delete temporary code file
			unlink(namebuffer);
		}

		return true;
	}

	extern void  tcc_close(void* state);
	extern void* tcc_lookup(void* state, const char*);

	void* dylib_lookup(void* dylib, const char* symbol, [[maybe_unused]] bool is_libtcc)
	{
#ifdef RISCV_LIBTCC
		if (is_libtcc) {
			return tcc_lookup(dylib, symbol);
		}
#endif
		return dlsym(dylib, symbol);
	}

	void dylib_close(void* dylib, [[maybe_unused]] bool is_libtcc)
	{
#ifdef RISCV_LIBTCC
		if (is_libtcc) {
			tcc_close(dylib);
			return;
		}
#endif
		dlclose(dylib);
	}
}

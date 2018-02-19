/*
 * mptLibrary.h
 * ------------
 * Purpose: Shared library handling.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


OPENMPT_NAMESPACE_BEGIN


#if defined(MPT_ENABLE_DYNBIND)

namespace mpt
{


typedef void* (*FuncPtr)(); // pointer to function returning void*

class LibraryHandle;

enum LibrarySearchPath
{
	LibrarySearchPathInvalid,
	LibrarySearchPathDefault,
	LibrarySearchPathApplication,
	LibrarySearchPathSystem,
	LibrarySearchPathFullPath,
};

class LibraryPath
{

private:
	
	mpt::LibrarySearchPath searchPath;
	mpt::PathString fileName;

private:

	LibraryPath(mpt::LibrarySearchPath searchPath, const mpt::PathString &fileName);

public:

	mpt::LibrarySearchPath GetSearchPath() const;

	mpt::PathString GetFileName() const;

public:
	
	// "lib" on Unix-like systems, "" on Windows
	static mpt::PathString GetDefaultPrefix();

	// ".so" or ".dylib" or ".dll"
	static mpt::PathString GetDefaultSuffix();

	// Returns the library path in the application directory, with os-specific prefix and suffix added to basename.
	// e.g.: basename = "unmo3" --> "libunmo3.so" / "apppath/unmo3.dll"
	static LibraryPath App(const mpt::PathString &basename);

	// Returns the library path in the application directory, with os-specific suffix added to fullname.
	// e.g.: fullname = "libunmo3" --> "libunmo3.so" / "apppath/libunmo3.dll" 
	static LibraryPath AppFullName(const mpt::PathString &fullname);

#if defined(MODPLUG_TRACKER) && !defined(MPT_BUILD_WINESUPPORT)

	// Returns the library path in the application data directory, with os-specific suffix added to fullname.
	// e.g.: fullname = "libunmo3" --> "libunmo3.so" / "appdata/libunmo3.dll" (appdata == C:\Users\SOMEUSER\AppData\OpenMPT\Components\)
	static LibraryPath AppDataFullName(const mpt::PathString &fullname, const mpt::PathString &appdata);

#endif // MODPLUG_TRACKER && !MPT_BUILD_WINESUPPORT

	// Returns a system library name with os-specific prefix and suffix added to basename, but without any full path in order to be searched in the default search path.
	// e.g.: basename = "unmo3" --> "libunmo3.so" / "unmo3.dll"
	static LibraryPath System(const mpt::PathString &basename);

	// Returns a system library name with os-specific suffix added to path.
	// e.g.: path = "somepath/foo" --> "somepath/foo.so" / "somepath/foo.dll"
	static LibraryPath FullPath(const mpt::PathString &path);

};

class Library
{
protected:
	std::shared_ptr<LibraryHandle> m_Handle;
public:
	Library();
	Library(const mpt::LibraryPath &path);
public:
	void Unload();
	bool IsValid() const;
	FuncPtr GetProcAddress(const std::string &symbol) const;
	template <typename Tfunc>
	bool Bind(Tfunc * & f, const std::string &symbol) const
	{
		#if !(MPT_OS_WINDOWS && MPT_COMPILER_GCC)
			// MinGW64 std::is_function is always false for non __cdecl functions.
			// See https://connect.microsoft.com/VisualStudio/feedback/details/774720/stl-is-function-bug .
			STATIC_ASSERT(std::is_function<Tfunc>::value);
		#endif
		const FuncPtr addr = GetProcAddress(symbol);
		f = reinterpret_cast<Tfunc*>(addr);
		return (addr != nullptr);
	}
};

} // namespace mpt

#endif // MPT_ENABLE_DYNBIND


OPENMPT_NAMESPACE_END

/*
 * mptLibrary.cpp
 * --------------
 * Purpose: Shared library handling.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "mptLibrary.h"

#if defined(MPT_ENABLE_DYNBIND)
#if MPT_OS_WINDOWS
#include <windows.h>
#elif MPT_OS_ANDROID
#include <dlfcn.h>
#elif defined(MPT_WITH_LTDL)
#include <ltdl.h>
#elif defined(MPT_WITH_DL)
#include <dlfcn.h>
#endif
#endif


OPENMPT_NAMESPACE_BEGIN


#if defined(MPT_ENABLE_DYNBIND)


namespace mpt
{


#if MPT_OS_WINDOWS


// KB2533623 / Win8
#ifndef LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS    0x00001000
#endif
#ifndef LOAD_LIBRARY_SEARCH_APPLICATION_DIR
#define LOAD_LIBRARY_SEARCH_APPLICATION_DIR 0x00000200
#endif
#ifndef LOAD_LIBRARY_SEARCH_SYSTEM32
#define LOAD_LIBRARY_SEARCH_SYSTEM32        0x00000800
#endif
#ifndef LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR    0x00000100
#endif


class LibraryHandle
{

private:

	HMODULE hModule;

public:

	LibraryHandle(const mpt::LibraryPath &path)
		: hModule(NULL)
	{

#if MPT_OS_WINDOWS_WINRT

#if (_WIN32_WINNT < 0x0602)
		(void)path;
		hModule = NULL; // unsupported
#else
		switch(path.GetSearchPath())
		{
			case mpt::LibrarySearchPathDefault:
				hModule = LoadPackagedLibrary(path.GetFileName().AsNative().c_str(), 0);
				break;
			case mpt::LibrarySearchPathApplication:
				hModule = LoadPackagedLibrary(path.GetFileName().AsNative().c_str(), 0);
				break;
			case mpt::LibrarySearchPathSystem:
				hModule = NULL; // Only application packaged libraries can be loaded dynamically in WinRT
				break;
			case mpt::LibrarySearchPathFullPath:
				hModule = NULL; // Absolute path is not supported in WinRT
				break;
			case mpt::LibrarySearchPathInvalid:
				MPT_ASSERT_NOTREACHED();
				break;
		}
#endif

#else // !MPT_OS_WINDOWS_WINRT

		mpt::Windows::Version WindowsVersion = mpt::Windows::Version::Current();

		// Check for KB2533623:
		bool hasKB2533623 = false;
		if(WindowsVersion.IsAtLeast(mpt::Windows::Version::Win8))
		{
			hasKB2533623 = true;
		} else if(WindowsVersion.IsAtLeast(mpt::Windows::Version::WinVista))
		{
			HMODULE hKernel32DLL = LoadLibraryW(L"kernel32.dll");
			if(hKernel32DLL)
			{
				if(::GetProcAddress(hKernel32DLL, "SetDefaultDllDirectories") != nullptr)
				{
					hasKB2533623 = true;
				}
				FreeLibrary(hKernel32DLL);
				hKernel32DLL = NULL;
			}
		}

		if(hasKB2533623)
		{
			switch(path.GetSearchPath())
			{
				case mpt::LibrarySearchPathDefault:
					hModule = LoadLibraryExW(path.GetFileName().AsNative().c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
					break;
				case mpt::LibrarySearchPathSystem:
					hModule = LoadLibraryExW(path.GetFileName().AsNative().c_str(), NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
					break;
#if defined(MODPLUG_TRACKER)
					// Using restricted search paths applies to potential DLL dependencies
					// recursively.
					// This fails loading for e.g. Codec or Plugin DLLs in application
					// directory if they depend on the MSVC C or C++ runtime (which is
					// located in the system directory).
					// Just rely on the default search path here.
				case mpt::LibrarySearchPathApplication:
					{
						const mpt::PathString dllPath = mpt::GetAppPath();
						if(!dllPath.empty() && mpt::PathIsAbsolute(dllPath) && dllPath.IsDirectory())
						{
							hModule = LoadLibraryW((dllPath + path.GetFileName()).AsNative().c_str());
						}
					}
					break;
				case mpt::LibrarySearchPathFullPath:
					hModule = LoadLibraryW(path.GetFileName().AsNative().c_str());
					break;
#else
					// For libopenmpt, do the safe thing.
				case mpt::LibrarySearchPathApplication:
					hModule = LoadLibraryExW(path.GetFileName().AsNative().c_str(), NULL, LOAD_LIBRARY_SEARCH_APPLICATION_DIR);
					break;
				case mpt::LibrarySearchPathFullPath:
					hModule = LoadLibraryExW(path.GetFileName().AsNative().c_str(), NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
					break;
#endif
				case mpt::LibrarySearchPathInvalid:
					MPT_ASSERT_NOTREACHED();
					break;
			}
		} else
		{
			switch(path.GetSearchPath())
			{
				case mpt::LibrarySearchPathDefault:
					hModule = LoadLibraryW(path.GetFileName().AsNative().c_str());
					break;
				case mpt::LibrarySearchPathApplication:
					{
						const mpt::PathString dllPath = mpt::GetAppPath();
						if(!dllPath.empty() && mpt::PathIsAbsolute(dllPath) && dllPath.IsDirectory())
						{
							hModule = LoadLibraryW((dllPath + path.GetFileName()).AsNative().c_str());
						}
					}
					break;
				case mpt::LibrarySearchPathSystem:
					{
						const mpt::PathString dllPath = mpt::GetSystemPath();
						if(!dllPath.empty() && mpt::PathIsAbsolute(dllPath) && dllPath.IsDirectory())
						{
							hModule = LoadLibraryW((dllPath + path.GetFileName()).AsNative().c_str());
						}
					}
					break;
				case mpt::LibrarySearchPathFullPath:
					hModule = LoadLibraryW(path.GetFileName().AsNative().c_str());
					break;
				case mpt::LibrarySearchPathInvalid:
					MPT_ASSERT_NOTREACHED();
					break;
			}
		}

#endif // MPT_OS_WINDOWS_WINRT

	}

	~LibraryHandle()
	{
		if(IsValid())
		{
			FreeLibrary(hModule);
		}
		hModule = NULL;
	}

public:

	bool IsValid() const
	{
		return (hModule != NULL);
	}

	FuncPtr GetProcAddress(const std::string &symbol) const
	{
		if(!IsValid())
		{
			return nullptr;
		}
		return reinterpret_cast<FuncPtr>(::GetProcAddress(hModule, symbol.c_str()));
	}

};


#elif MPT_OS_ANDROID


// Fake implementation.
// Load shared objects from the JAVA side of things.
class LibraryHandle
{

public:

	LibraryHandle(const mpt::LibraryPath &path)
	{
		return;
	}

	~LibraryHandle()
	{
		return;
	}

public:

	bool IsValid() const
	{
		return true;
	}

	FuncPtr GetProcAddress(const std::string &symbol) const
	{
		if(!IsValid())
		{
			return nullptr;
		}
		return reinterpret_cast<FuncPtr>(dlsym(0, symbol.c_str()));
	}

};



#elif defined(MPT_WITH_LTDL)


class LibraryHandle
{

private:

	bool inited;
	lt_dlhandle handle;

public:

	LibraryHandle(const mpt::LibraryPath &path)
		: inited(false)
		, handle(0)
	{
		if(lt_dlinit() != 0)
		{
			return;
		}
		inited = true;
		handle = lt_dlopenext(path.GetFileName().AsNative().c_str());
	}

	~LibraryHandle()
	{
		if(IsValid())
		{
			lt_dlclose(handle);
		}
		handle = 0;
		if(inited)
		{
			lt_dlexit();
			inited = false;
		}
	}

public:

	bool IsValid() const
	{
		return handle != 0;
	}

	FuncPtr GetProcAddress(const std::string &symbol) const
	{
		if(!IsValid())
		{
			return nullptr;
		}
		return reinterpret_cast<FuncPtr>(lt_dlsym(handle, symbol.c_str()));
	}

};


#elif defined(MPT_WITH_DL)


class LibraryHandle
{

private:

	void* handle;

public:

	LibraryHandle(const mpt::LibraryPath &path)
		: handle(NULL)
	{
		handle = dlopen(path.GetFileName().AsNative().c_str(), RTLD_NOW);
	}

	~LibraryHandle()
	{
		if(IsValid())
		{
			dlclose(handle);
		}
		handle = NULL;
	}

public:

	bool IsValid() const
	{
		return handle != NULL;
	}

	FuncPtr GetProcAddress(const std::string &symbol) const
	{
		if(!IsValid())
		{
			return NULL;
		}
		return reinterpret_cast<FuncPtr>(dlsym(handle, symbol.c_str()));
	}

};


#else // MPT_OS


// dummy implementation

class LibraryHandle
{
public:

	LibraryHandle(const mpt::LibraryPath &path)
	{
		MPT_UNREFERENCED_PARAMETER(path);
		return;
	}

	~LibraryHandle()
	{
		return;
	}

public:

	bool IsValid() const
	{
		return false;
	}

	FuncPtr GetProcAddress(const std::string &symbol) const
	{
		MPT_UNREFERENCED_PARAMETER(symbol);
		if(!IsValid())
		{
			return nullptr;
		}
		return nullptr;
	}

};


#endif // MPT_OS


LibraryPath::LibraryPath(mpt::LibrarySearchPath searchPath, class mpt::PathString const &fileName)
	: searchPath(searchPath)
	, fileName(fileName)
{
	return;
}


mpt::LibrarySearchPath LibraryPath::GetSearchPath() const
{
	return searchPath;
}


mpt::PathString LibraryPath::GetFileName() const
{
	return fileName;
}


mpt::PathString LibraryPath::GetDefaultPrefix()
{
	#if MPT_OS_WINDOWS
		return MPT_PATHSTRING("");
	#elif MPT_OS_ANDROID
		return MPT_PATHSTRING("lib");
	#elif defined(MPT_WITH_LTDL)
		return MPT_PATHSTRING("lib");
	#elif defined(MPT_WITH_DL)
		return MPT_PATHSTRING("lib");
	#else
		return MPT_PATHSTRING("lib");
	#endif
}


mpt::PathString LibraryPath::GetDefaultSuffix()
{
	#if MPT_OS_WINDOWS
		return MPT_PATHSTRING(".dll");
	#elif MPT_OS_ANDROID
		return MPT_PATHSTRING(".so");
	#elif defined(MPT_WITH_LTDL)
		return MPT_PATHSTRING("");  // handled by libltdl
	#elif defined(MPT_WITH_DL)
		return MPT_PATHSTRING(".so");
	#else
		return mpt::PathString();
	#endif
}


LibraryPath LibraryPath::App(const mpt::PathString &basename)
{
	return LibraryPath(mpt::LibrarySearchPathApplication, GetDefaultPrefix() + basename + GetDefaultSuffix());
}


LibraryPath LibraryPath::AppFullName(const mpt::PathString &fullname)
{
	return LibraryPath(mpt::LibrarySearchPathApplication, fullname + GetDefaultSuffix());
}


#if defined(MODPLUG_TRACKER) && !defined(MPT_BUILD_WINESUPPORT)

LibraryPath LibraryPath::AppDataFullName(const mpt::PathString &fullname, const mpt::PathString &appdata)
{
	if(appdata.empty())
	{
		return LibraryPath(mpt::LibrarySearchPathInvalid, MPT_PATHSTRING(""));
	}
	return LibraryPath(mpt::LibrarySearchPathFullPath, appdata.WithTrailingSlash() + fullname + GetDefaultSuffix());
}

#endif // MODPLUG_TRACKER && !MPT_BUILD_WINESUPPORT


LibraryPath LibraryPath::System(const mpt::PathString &basename)
{
	return LibraryPath(mpt::LibrarySearchPathSystem, GetDefaultPrefix() + basename + GetDefaultSuffix());
}


LibraryPath LibraryPath::FullPath(const mpt::PathString &path)
{
	return LibraryPath(mpt::LibrarySearchPathFullPath, path);
}


Library::Library()
{
	return;
}


Library::Library(const mpt::LibraryPath &path)
{
	if(path.GetSearchPath() == mpt::LibrarySearchPathInvalid)
	{
		return;
	}
	if(path.GetFileName().empty())
	{
		return;
	}
	m_Handle = std::make_shared<LibraryHandle>(path);
}


void Library::Unload()
{
	*this = mpt::Library();
}


bool Library::IsValid() const
{
	return m_Handle && m_Handle->IsValid();
}


FuncPtr Library::GetProcAddress(const std::string &symbol) const
{
	if(!IsValid())
	{
		return nullptr;
	}
	return m_Handle->GetProcAddress(symbol);
}


} // namespace mpt


#endif // MPT_ENABLE_DYNBIND


OPENMPT_NAMESPACE_END

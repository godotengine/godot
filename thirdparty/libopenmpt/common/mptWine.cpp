/*
 * mptWine.cpp
 * -----------
 * Purpose: Wine stuff.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "mptWine.h"

#include "mptOS.h"
#include "mptFileIO.h"

#include <deque>
#include <map>

#if MPT_OS_WINDOWS
#include <windows.h>
#endif


OPENMPT_NAMESPACE_BEGIN


#if defined(MODPLUG_TRACKER) && MPT_OS_WINDOWS


namespace mpt
{
namespace Wine
{



Context::Context(mpt::Wine::VersionContext versionContext)
	: m_VersionContext(versionContext)
	, wine_get_dos_file_name(nullptr)
	, wine_get_unix_file_name(nullptr)
{
	if(!mpt::Windows::IsWine())
	{
		throw mpt::Wine::Exception("Wine not detected.");
	}
	if(!m_VersionContext.Version().IsValid())
	{
		throw mpt::Wine::Exception("Unknown Wine version detected.");
	}
	m_Kernel32 = mpt::Library(mpt::LibraryPath::FullPath(MPT_PATHSTRING("kernel32.dll")));
	if(!m_Kernel32.IsValid())
	{
		throw mpt::Wine::Exception("Could not load Wine kernel32.dll.");
	}
	if(!m_Kernel32.Bind(wine_get_unix_file_name, "wine_get_unix_file_name"))
	{
		throw mpt::Wine::Exception("Could not bind Wine kernel32.dll:wine_get_unix_file_name.");
	}
	if(!m_Kernel32.Bind(wine_get_dos_file_name, "wine_get_dos_file_name"))
	{
		throw mpt::Wine::Exception("Could not bind Wine kernel32.dll:wine_get_dos_file_name.");
	}
	{
		std::string out;
		std::string err;
		try
		{
			if(ExecutePosixShellCommand("uname -m", out, err) != 0)
			{
				throw mpt::Wine::Exception("Wine 'uname -m' failed.");
			}
			if(!err.empty())
			{
				throw mpt::Wine::Exception("Wine 'uname -m' failed.");
			}
			out = mpt::String::Trim(out, std::string("\r\n"));
			m_Uname_m = out;
		} catch(const std::exception &)
		{
			m_Uname_m = std::string();
		}
	}
	try
	{
		m_HOME = GetPosixEnvVar("HOME");
	} catch(const std::exception &)
	{
		m_HOME = std::string();
	}
	try
	{
		m_XDG_DATA_HOME = GetPosixEnvVar("XDG_DATA_HOME");
		if(m_XDG_DATA_HOME.empty())
		{
			m_XDG_DATA_HOME = m_HOME + "/.local/share";
		}
	} catch(const std::exception &)
	{
		m_XDG_DATA_HOME = std::string();
	}
	try
	{
		m_XDG_CACHE_HOME = GetPosixEnvVar("XDG_CACHE_HOME");
		if(m_XDG_CACHE_HOME.empty())
		{
			m_XDG_CACHE_HOME = m_HOME + "/.cache";
		}
	} catch(const std::exception &)
	{
		m_XDG_CACHE_HOME = std::string();
	}
	try
	{
		m_XDG_CONFIG_HOME = GetPosixEnvVar("XDG_CONFIG_HOME");
		if(m_XDG_CONFIG_HOME.empty())
		{
			m_XDG_CONFIG_HOME = m_HOME + "/.config";
		}
	} catch(const std::exception &)
	{
		m_XDG_CONFIG_HOME = std::string();
	}
}


std::string Context::PathToPosix(mpt::PathString windowsPath)
{
	std::string result;
	if(windowsPath.empty())
	{
		return result;
	}
	if(windowsPath.Length() >= 32000)
	{
		throw mpt::Wine::Exception("Path too long.");
	}
	LPSTR tmp = nullptr;
	tmp = wine_get_unix_file_name(windowsPath.AsNative().c_str());
	if(!tmp)
	{
		throw mpt::Wine::Exception("Wine kernel32.dll:wine_get_unix_file_name failed.");
	}
	result = tmp;
	HeapFree(GetProcessHeap(), 0, tmp);
	tmp = nullptr;
	return result;
}

mpt::PathString Context::PathToWindows(std::string hostPath)
{
	mpt::PathString result;
	if(hostPath.empty())
	{
		return result;
	}
	if(hostPath.length() >= 32000)
	{
		throw mpt::Wine::Exception("Path too long.");
	}
	LPWSTR tmp = nullptr;
	tmp = wine_get_dos_file_name(hostPath.c_str());
	if(!tmp)
	{
		throw mpt::Wine::Exception("Wine kernel32.dll:wine_get_dos_file_name failed.");
	}
	result = mpt::PathString::FromNative(tmp);
	HeapFree(GetProcessHeap(), 0, tmp);
	tmp = nullptr;
	return result;
}

std::string Context::PathToPosixCanonical(mpt::PathString windowsPath)
{
	std::string result;
	std::string hostPath = PathToPosix(windowsPath);
	if(hostPath.empty())
	{
		return result;
	}
	std::string output;
	std::string error;
	int exitcode = ExecutePosixShellCommand(std::string() + "readlink -f " + EscapePosixShell(hostPath), output, error);
	if(!error.empty())
	{
		throw mpt::Wine::Exception("Wine readlink failed: " + error);
	}
	if(exitcode != 0 && exitcode != 1)
	{
		throw mpt::Wine::Exception("Wine readlink failed.");
	}
	std::string trimmedOutput = mpt::String::Trim(output, std::string("\r\n"));
	result = trimmedOutput;
	return result;
}


static void ExecutePosixCommandProgressDefault(void * /*userdata*/ )
{
	::Sleep(10);
	return;
}

static ExecuteProgressResult ExecutePosixShellScriptProgressDefault(void * /*userdata*/ )
{
	::Sleep(10);
	return ExecuteProgressContinueWaiting;
}


std::string Context::EscapePosixShell(std::string line)
{
	const char escape_chars [] = { '|', '&', ';', '<', '>', '(', ')', '$', '`', '"', '\'', ' ', '\t' };
	const char maybe_escape_chars [] = { '*', '?', '[', '#', '~', '=', '%' };
	line = mpt::String::Replace(line, "\\", "\\\\");
	for(std::size_t i = 0; i < mpt::size(escape_chars); ++i)
	{
		line = mpt::String::Replace(line, std::string(1, escape_chars[i]), "\\" + std::string(1, escape_chars[i]));
	}
	for(std::size_t i = 0; i < mpt::size(maybe_escape_chars); ++i)
	{
		line = mpt::String::Replace(line, std::string(1, maybe_escape_chars[i]), "\\" + std::string(1, maybe_escape_chars[i]));
	}
	return line;
}


ExecResult Context::ExecutePosixShellScript(std::string script, FlagSet<ExecFlags> flags, std::map<std::string, std::vector<char> > filetree, std::string title, ExecutePosixCommandProgress progress, ExecutePosixShellScriptProgress progressCancel, void *userdata)
{
	// Relevant documentation:
	// https://stackoverflow.com/questions/6004070/execute-shell-commands-from-program-running-in-wine
	// https://www.winehq.org/pipermail/wine-bugs/2014-January/374918.html
	// https://bugs.winehq.org/show_bug.cgi?id=34730

	if(!progress) progress = &ExecutePosixCommandProgressDefault;
	if(!progressCancel) progressCancel = &ExecutePosixShellScriptProgressDefault;

	if(flags[ExecFlagInteractive]) flags.reset(ExecFlagSilent);
	if(flags[ExecFlagSplitOutput]) flags.set(ExecFlagSilent);

	std::vector<mpt::PathString> tempfiles;

	progress(userdata);

	mpt::TempDirGuard dirWindowsTemp(mpt::CreateTempFileName());
	if(dirWindowsTemp.GetDirname().empty())
	{
		throw mpt::Wine::Exception("Creating temporary directoy failed.");
	}
	std::string dirPosix = PathToPosix(dirWindowsTemp.GetDirname());
	if(dirPosix.empty())
	{
		throw mpt::Wine::Exception("mpt::Wine::ConvertWindowsPathToHost returned empty path.");
	}
	const mpt::PathString dirWindows = dirWindowsTemp.GetDirname();

	progress(userdata);

	// write the script to disk
	mpt::PathString scriptFilenameWindows = dirWindows + MPT_PATHSTRING("script.sh");
	{
		mpt::ofstream tempfile(scriptFilenameWindows, std::ios::binary);
		tempfile << script;
		tempfile.flush();
		if(!tempfile)
		{
			throw mpt::Wine::Exception("Error writing script.sh.");
		}
	}
	std::string scriptFilenamePosix = PathToPosix(scriptFilenameWindows);
	if(scriptFilenamePosix.empty())
	{
		throw mpt::Wine::Exception("Error converting script.sh path.");
	}

	progress(userdata);

	// create a wrapper that will call the script and gather result.
	mpt::PathString wrapperstarterFilenameWindows = dirWindows + MPT_PATHSTRING("wrapperstarter.sh");
	{
		mpt::ofstream tempfile(wrapperstarterFilenameWindows, std::ios::binary);
		std::string wrapperstarterscript;
		wrapperstarterscript += std::string() + "#!/usr/bin/env sh" + "\n";
		wrapperstarterscript += std::string() + "exec /usr/bin/env sh " + EscapePosixShell(dirPosix) + "wrapper.sh" + "\n";
		tempfile << wrapperstarterscript;
		tempfile.flush();
		if(!tempfile)
		{
			throw mpt::Wine::Exception("Error writing wrapper.sh.");
		}
	}
	mpt::PathString wrapperFilenameWindows = dirWindows + MPT_PATHSTRING("wrapper.sh");
	std::string cleanupscript;
	{
		mpt::ofstream tempfile(wrapperFilenameWindows, std::ios::binary);
		std::string wrapperscript;
		if(!flags[ExecFlagSilent])
		{
			wrapperscript += std::string() + "printf \"\\033]0;" + title + "\\a\"" + "\n";
		}
		wrapperscript += std::string() + "chmod u+x " + EscapePosixShell(scriptFilenamePosix) + "\n";
		wrapperscript += std::string() + "cd " + EscapePosixShell(dirPosix) + "filetree" + "\n";
		if(flags[ExecFlagInteractive])
		{ // no stdout/stderr capturing for interactive scripts
			wrapperscript += std::string() + EscapePosixShell(scriptFilenamePosix) + "\n";
			wrapperscript += std::string() + "MPT_RESULT=$?" + "\n";
			wrapperscript += std::string() + "echo ${MPT_RESULT} > " + EscapePosixShell(dirPosix) + "exit" + "\n";
		} else if(flags[ExecFlagSplitOutput])
		{
			wrapperscript += std::string() + "(" + EscapePosixShell(scriptFilenamePosix) + "; echo $? >&4) 4>" + EscapePosixShell(dirPosix) + "exit 1>" + EscapePosixShell(dirPosix) + "out 2>" + EscapePosixShell(dirPosix) + "err" + "\n";
		} else
		{
			wrapperscript += std::string() + "(" + EscapePosixShell(scriptFilenamePosix) + "; echo $? >&4) 2>&1 4>" + EscapePosixShell(dirPosix) + "exit | tee " + EscapePosixShell(dirPosix) + "out" + "\n";
		}
		wrapperscript += std::string() + "echo done > " + EscapePosixShell(dirPosix) + "done" + "\n";
		cleanupscript += std::string() + "rm " + EscapePosixShell(dirPosix) + "done" + "\n";
		cleanupscript += std::string() + "rm " + EscapePosixShell(dirPosix) + "exit" + "\n";
		if(flags[ExecFlagInteractive])
		{
			// nothing
		} else if(flags[ExecFlagSplitOutput])
		{
			cleanupscript += std::string() + "rm " + EscapePosixShell(dirPosix) + "out" + "\n";
			cleanupscript += std::string() + "rm " + EscapePosixShell(dirPosix) + "err" + "\n";
		} else
		{
			cleanupscript += std::string() + "rm " + EscapePosixShell(dirPosix) + "out" + "\n";
		}
		cleanupscript += std::string() + "rm -r " + EscapePosixShell(dirPosix) + "filetree" + "\n";			
		cleanupscript += std::string() + "rm " + EscapePosixShell(dirPosix) + "script.sh" + "\n";			
		cleanupscript += std::string() + "rm " + EscapePosixShell(dirPosix) + "wrapper.sh" + "\n";			
		cleanupscript += std::string() + "rm " + EscapePosixShell(dirPosix) + "wrapperstarter.sh" + "\n";			
		cleanupscript += std::string() + "rm " + EscapePosixShell(dirPosix) + "terminal.sh" + "\n";			
		if(flags[ExecFlagAsync])
		{
			wrapperscript += cleanupscript;
			cleanupscript.clear();
		}
		tempfile << wrapperscript;
		tempfile.flush();
		if(!tempfile)
		{
			throw mpt::Wine::Exception("Error writing wrapper.sh.");
		}
	}

	progress(userdata);

	::CreateDirectoryW((dirWindows + MPT_PATHSTRING("filetree")).AsNative().c_str(), NULL);
	for(const auto &file : filetree)
	{
		std::vector<mpt::ustring> path = mpt::String::Split<mpt::ustring>(mpt::ToUnicode(mpt::CharsetUTF8, file.first), MPT_USTRING("/"));
		mpt::PathString combinedPath = dirWindows + MPT_PATHSTRING("filetree") + MPT_PATHSTRING("\\");
		if(path.size() > 1)
		{
			for(std::size_t singlepath = 0; singlepath < path.size() - 1; ++singlepath)
			{
				if(path[singlepath].empty())
				{
					continue;
				}
				combinedPath += mpt::PathString::FromUnicode(path[singlepath]);
				if(!combinedPath.IsDirectory())
				{
					if(::CreateDirectoryW(combinedPath.AsNative().c_str(), NULL) == 0)
					{
						throw mpt::Wine::Exception("Error writing filetree.");
					}
				}
				combinedPath += MPT_PATHSTRING("\\");
			}
		}
		try
		{
			mpt::LazyFileRef out(dirWindows + MPT_PATHSTRING("filetree") + MPT_PATHSTRING("\\") + mpt::PathString::FromUTF8(mpt::String::Replace(file.first, "/", "\\")));
			out = file.second;
		} catch(std::exception &)
		{
			throw mpt::Wine::Exception("Error writing filetree.");
		}
	}

	progress(userdata);

	// create a wrapper that will find a suitable terminal and run the wrapper script in the terminal window.
	mpt::PathString terminalWrapperFilenameWindows = dirWindows + MPT_PATHSTRING("terminal.sh");
	{
		mpt::ofstream tempfile(terminalWrapperFilenameWindows, std::ios::binary);
		// NOTE:
		// Modern terminals detach themselves from the invoking shell if another instance is already present.
		// This means we cannot rely on terminal invocation being syncronous.
		std::vector<std::string> terminals;
		terminals.push_back("x-terminal-emulator");
		terminals.push_back("konsole");
		terminals.push_back("mate-terminal");
		terminals.push_back("xfce4-terminal");
		terminals.push_back("gnome-terminal");
		terminals.push_back("uxterm");
		terminals.push_back("xterm");
		terminals.push_back("rxvt");
		std::map<std::string, std::string> terminalLanchers;
		for(std::size_t i = 0; i < terminals.size(); ++i)
		{
			// mate-terminal on Debian 8 cannot execute commands with arguments,
			// thus we use a separate script that requires no arguments to execute.
			terminalLanchers[terminals[i]] += std::string() + "if command -v " + terminals[i] + " 2>/dev/null 1>/dev/null ; then" + "\n";
			terminalLanchers[terminals[i]] += std::string() + " chmod u+x " + EscapePosixShell(dirPosix) + "wrapperstarter.sh" + "\n";
			terminalLanchers[terminals[i]] += std::string() + " exec `command -v " + terminals[i] + "` -e \"" + EscapePosixShell(dirPosix) + "wrapperstarter.sh\"" + "\n";
			terminalLanchers[terminals[i]] += std::string() + "fi" + "\n";
		}

		std::string terminalscript;

		terminalscript += std::string() + "\n";

		for(std::size_t i = 0; i < terminals.size(); ++i)
		{
			terminalscript += terminalLanchers[terminals[i]];
		}

		tempfile << terminalscript;
		tempfile.flush();
		if(!tempfile)
		{
			return ExecResult::Error();
		}
	}

	progress(userdata);

	// build unix command line
	std::string unixcommand;
	bool createProcessSuccess = false;

	if(!createProcessSuccess)
	{

		if(flags[ExecFlagSilent])
		{
			unixcommand = "/usr/bin/env sh \"" + EscapePosixShell(dirPosix) + "wrapper.sh\"";
		} else
		{
			unixcommand = "/usr/bin/env sh \"" + EscapePosixShell(dirPosix) + "terminal.sh\"";
		}

		progress(userdata);

		std::wstring unixcommandW = mpt::ToWide(mpt::CharsetUTF8, unixcommand);
		std::vector<WCHAR> commandline = std::vector<WCHAR>(unixcommandW.data(), unixcommandW.data() + unixcommandW.length() + 1);
		std::wstring titleW = mpt::ToWide(mpt::CharsetUTF8, title);
		std::vector<WCHAR> titleWv = std::vector<WCHAR>(titleW.data(), titleW.data() + titleW.length() + 1);
		STARTUPINFOW startupInfo;
		MemsetZero(startupInfo);
		startupInfo.lpTitle = &titleWv[0];
		startupInfo.cb = sizeof(startupInfo);
		PROCESS_INFORMATION processInformation;
		MemsetZero(processInformation);

		progress(userdata);

		BOOL success = FALSE;
		if(flags[ExecFlagSilent])
		{
			success = CreateProcessW(NULL, &commandline[0], NULL, NULL, FALSE, DETACHED_PROCESS, NULL, NULL, &startupInfo, &processInformation);
		} else
		{
			success = CreateProcessW(NULL, &commandline[0], NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &startupInfo, &processInformation);
		}

		progress(userdata);

		createProcessSuccess = (success ? true : false);

		progress(userdata);

		if(success)
		{

			if(!flags[ExecFlagAsync])
			{
				// note: execution is not syncronous with all Wine versions,
				// we additionally explicitly wait for "done" later
				while(WaitForSingleObject(processInformation.hProcess, 0) == WAIT_TIMEOUT)
				{ // wait
					if(progressCancel(userdata) != ExecuteProgressContinueWaiting)
					{
						CloseHandle(processInformation.hThread);
						CloseHandle(processInformation.hProcess);
						throw mpt::Wine::Exception("Canceled.");
					}
				}
			}

			progress(userdata);

			CloseHandle(processInformation.hThread);
			CloseHandle(processInformation.hProcess);

		}

	}

	progress(userdata);

	// Work around Wine being able to execute PIE binaries on Debian 9.
	// Luckily, /bin/bash is still non-PIE on Debian 9.

	if(!createProcessSuccess)
	{

		if(flags[ExecFlagSilent]) {
			unixcommand = "/bin/bash \"" + EscapePosixShell(dirPosix) + "wrapper.sh\"";
		} else
		{
			unixcommand = "/bin/bash \"" + EscapePosixShell(dirPosix) + "terminal.sh\"";
		}

		progress(userdata);

		std::wstring unixcommandW = mpt::ToWide(mpt::CharsetUTF8, unixcommand);
		std::vector<WCHAR> commandline = std::vector<WCHAR>(unixcommandW.data(), unixcommandW.data() + unixcommandW.length() + 1);
		std::wstring titleW = mpt::ToWide(mpt::CharsetUTF8, title);
		std::vector<WCHAR> titleWv = std::vector<WCHAR>(titleW.data(), titleW.data() + titleW.length() + 1);
		STARTUPINFOW startupInfo;
		MemsetZero(startupInfo);
		startupInfo.lpTitle = &titleWv[0];
		startupInfo.cb = sizeof(startupInfo);
		PROCESS_INFORMATION processInformation;
		MemsetZero(processInformation);

		progress(userdata);

		BOOL success = FALSE;
		if(flags[ExecFlagSilent])
		{
			success = CreateProcessW(NULL, &commandline[0], NULL, NULL, FALSE, DETACHED_PROCESS, NULL, NULL, &startupInfo, &processInformation);
		} else
		{
			success = CreateProcessW(NULL, &commandline[0], NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &startupInfo, &processInformation);
		}

		progress(userdata);

		createProcessSuccess = (success ? true : false);

		progress(userdata);

		if(success)
		{

			if(!flags[ExecFlagAsync])
			{
				// note: execution is not syncronous with all Wine versions,
				// we additionally explicitly wait for "done" later
				while(WaitForSingleObject(processInformation.hProcess, 0) == WAIT_TIMEOUT)
				{ // wait
					if(progressCancel(userdata) != ExecuteProgressContinueWaiting)
					{
						CloseHandle(processInformation.hThread);
						CloseHandle(processInformation.hProcess);
						throw mpt::Wine::Exception("Canceled.");
					}
				}
			}

			progress(userdata);

			CloseHandle(processInformation.hThread);
			CloseHandle(processInformation.hProcess);

		}

	}

	progress(userdata);

	if(!createProcessSuccess)
	{
		throw mpt::Wine::Exception("CreateProcess failed.");
	}

	progress(userdata);

	if(flags[ExecFlagAsync])
	{
		ExecResult result;
		result.exitcode = 0;
		return result;
	}

	while(!(dirWindows + MPT_PATHSTRING("done")).IsFile())
	{ // wait
		if(progressCancel(userdata) != ExecuteProgressContinueWaiting)
		{
			throw mpt::Wine::Exception("Canceled.");
		}
	}

	progress(userdata);

	int exitCode = 0;
	{
		mpt::ifstream exitFile(dirWindows + MPT_PATHSTRING("exit"), std::ios::binary);
		if(!exitFile)
		{
			throw mpt::Wine::Exception("Script .exit file not found.");
		}
		std::string exitString;
		exitFile >> exitString;
		if(exitString.empty())
		{
			throw mpt::Wine::Exception("Script .exit file empty.");
		}
		exitCode = ConvertStrTo<int>(exitString);
	}

	progress(userdata);

	std::string outputString;
	if(!flags[ExecFlagInteractive])
	{
		mpt::ifstream outputFile(dirWindows + MPT_PATHSTRING("out"), std::ios::binary);
		if(outputFile)
		{
			outputFile.seekg(0, std::ios::end);
			std::streampos outputFileSize = outputFile.tellg();
			outputFile.seekg(0, std::ios::beg);
			std::vector<char> outputFileBuf(mpt::saturate_cast<std::size_t>(static_cast<std::streamoff>(outputFileSize)));
			outputFile.read(&outputFileBuf[0], outputFileBuf.size());
			outputString = std::string(outputFileBuf.begin(), outputFileBuf.end());
		}
	}

	progress(userdata);

	std::string errorString;
	if(flags[ExecFlagSplitOutput])
	{
		mpt::ifstream errorFile(dirWindows + MPT_PATHSTRING("err"), std::ios::binary);
		if(errorFile)
		{
			errorFile.seekg(0, std::ios::end);
			std::streampos errorFileSize = errorFile.tellg();
			errorFile.seekg(0, std::ios::beg);
			std::vector<char> errorFileBuf(mpt::saturate_cast<std::size_t>(static_cast<std::streamoff>(errorFileSize)));
			errorFile.read(&errorFileBuf[0], errorFileBuf.size());
			errorString = std::string(errorFileBuf.begin(), errorFileBuf.end());
		}
	}

	progress(userdata);

	ExecResult result;
	result.exitcode = exitCode;
	result.output = outputString;
	result.error = errorString;

	std::deque<mpt::PathString> paths;
	paths.push_back(dirWindows + MPT_PATHSTRING("filetree"));
	mpt::PathString basePath = (dirWindows + MPT_PATHSTRING("filetree")).EnsureTrailingSlash();
	while(!paths.empty())
	{
		mpt::PathString path = paths.front();
		paths.pop_front();
		path.EnsureTrailingSlash();
		HANDLE hFind = NULL;
		WIN32_FIND_DATAW wfd;
		MemsetZero(wfd);
		hFind = FindFirstFileW((path + MPT_PATHSTRING("*.*")).AsNative().c_str(), &wfd);
		if(hFind != NULL && hFind != INVALID_HANDLE_VALUE)
		{
			do
			{
				mpt::PathString filename = mpt::PathString::FromNative(wfd.cFileName);
				if(filename != MPT_PATHSTRING(".") && filename != MPT_PATHSTRING(".."))
				{
					filename = path + filename;
					filetree[filename.ToUTF8()] = std::vector<char>();
					if(filename.IsDirectory())
					{
						paths.push_back(filename);
					} else if(filename.IsFile())
					{
						try
						{
							mpt::LazyFileRef f(filename);
							std::vector<char> buf = f;
							mpt::PathString treeFilename = mpt::PathString::FromNative(filename.AsNative().substr(basePath.AsNative().length()));
							result.filetree[treeFilename.ToUTF8()] = buf;
						} catch (std::exception &)
						{
							// nothing?!
						}
					}
				}
			} while(FindNextFileW(hFind, &wfd));
			FindClose(hFind);
		}
	}

	mpt::DeleteWholeDirectoryTree(dirWindows);

	return result;

}


int Context::ExecutePosixShellCommand(std::string command, std::string & output, std::string & error)
{
	std::string script;
	script += std::string() + "#!/usr/bin/env sh" + "\n";
	script += std::string() + "exec " + command + "\n";
	mpt::Wine::ExecResult execResult = ExecutePosixShellScript
		( script
		, mpt::Wine::ExecFlagSilent | mpt::Wine::ExecFlagSplitOutput, std::map<std::string, std::vector<char> >()
		, std::string()
		, nullptr
		, nullptr
		, nullptr
		);
	output = execResult.output;
	error = execResult.error;
	return execResult.exitcode;
}


std::string Context::GetPosixEnvVar(std::string var, std::string def)
{
	// We cannot use std::getenv here because Wine overrides SOME env vars,
	// in particular, HOME is unset in the Wine environment.
	// Instead, we just spawn a shell that will catch up a sane environment on
	// its own.
	std::string output;
	std::string error;
	int exitcode = ExecutePosixShellCommand(std::string() + "echo $" + var, output, error);
	if(!error.empty())
	{
		throw mpt::Wine::Exception("Wine echo $var failed: " + error);
	}
	if(exitcode != 0)
	{
		throw mpt::Wine::Exception("Wine echo $var failed.");
	}
	std::string result = mpt::String::RTrim(output, std::string("\r\n"));
	if(result.empty())
	{
		result = def;
	}
	return result;
}


} // namespace Wine
} // namespace mpt


#else // !(MODPLUG_TRACKER && MPT_OS_WINDOWS)


MPT_MSVC_WORKAROUND_LNK4221(mptWine)


#endif // MODPLUG_TRACKER && MPT_OS_WINDOWS


OPENMPT_NAMESPACE_END

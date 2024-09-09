#include "process.hpp"
// clang-format off
#include <windows.h>
// clang-format on
#include <cstring>
#include <stdexcept>
#include <tlhelp32.h>

namespace TinyProcessLib {

Process::Data::Data() noexcept : id(0) {}

// Simple HANDLE wrapper to close it automatically from the destructor.
class Handle {
public:
  Handle() noexcept : handle(INVALID_HANDLE_VALUE) {}
  ~Handle() noexcept {
    close();
  }
  void close() noexcept {
    if(handle != INVALID_HANDLE_VALUE)
      CloseHandle(handle);
  }
  HANDLE detach() noexcept {
    HANDLE old_handle = handle;
    handle = INVALID_HANDLE_VALUE;
    return old_handle;
  }
  operator HANDLE() const noexcept { return handle; }
  HANDLE *operator&() noexcept { return &handle; }

private:
  HANDLE handle;
};

template <class Char>
struct CharSet;

template <>
struct CharSet<char> {
  static constexpr const char *whitespace = " \t\n\v\"";
  static constexpr char space = ' ';
  static constexpr char doublequote = '"';
  static constexpr char backslash = '\\';
};

template <>
struct CharSet<wchar_t> {
  static constexpr const wchar_t *whitespace = L" \t\n\v\"";
  static constexpr wchar_t space = L' ';
  static constexpr wchar_t doublequote = L'"';
  static constexpr wchar_t backslash = L'\\';
};

// Based on blog post https://learn.microsoft.com/en-us/archive/blogs/twistylittlepassagesallalike/everyone-quotes-command-line-arguments-the-wrong-way
template <class Char>
static std::basic_string<Char> join_arguments(const std::vector<std::basic_string<Char>> &args) {
  using charset = CharSet<Char>;

  std::basic_string<Char> ret;

  for(const auto &arg : args) {
    if(!ret.empty())
      ret.push_back(charset::space);

    if(!arg.empty() && arg.find_first_of(charset::whitespace) == arg.npos) {
      ret.append(arg);
      continue;
    }

    ret.push_back(charset::doublequote);

    for(auto it = arg.begin();; ++it) {
      size_t n_backslashes = 0;

      while(it != arg.end() && *it == charset::backslash) {
        ++it;
        ++n_backslashes;
      }

      if(it == arg.end()) {
        // Escape all backslashes, but let the terminating double quotation mark
        // we add below be interpreted  as a metacharacter.
        ret.append(n_backslashes * 2, charset::backslash);
        break;
      }
      else if(*it == charset::doublequote) {
        // Escape all backslashes and the following double quotation mark.
        ret.append(n_backslashes * 2 + 1, charset::backslash);
        ret.push_back(*it);
      }
      else {
        // Backslashes aren't special here.
        ret.append(n_backslashes, charset::backslash);
        ret.push_back(*it);
      }
    }

    ret.push_back(charset::doublequote);
  }

  return ret;
}

// Based on the discussion thread: https://www.reddit.com/r/cpp/comments/3vpjqg/a_new_platform_independent_process_library_for_c11/cxq1wsj
std::mutex create_process_mutex;

Process::id_type Process::open(const std::vector<string_type> &arguments, const string_type &path, const environment_type *environment) noexcept {
  return open(join_arguments(arguments), path, environment);
}

// Based on the example at https://msdn.microsoft.com/en-us/library/windows/desktop/ms682499(v=vs.85).aspx.
Process::id_type Process::open(const string_type &command, const string_type &path, const environment_type *environment) noexcept {
  if(open_stdin)
    stdin_fd = std::unique_ptr<fd_type>(new fd_type(nullptr));
  if(read_stdout)
    stdout_fd = std::unique_ptr<fd_type>(new fd_type(nullptr));
  if(read_stderr)
    stderr_fd = std::unique_ptr<fd_type>(new fd_type(nullptr));

  Handle stdin_rd_p;
  Handle stdin_wr_p;
  Handle stdout_rd_p;
  Handle stdout_wr_p;
  Handle stderr_rd_p;
  Handle stderr_wr_p;

  SECURITY_ATTRIBUTES security_attributes;

  security_attributes.nLength = sizeof(SECURITY_ATTRIBUTES);
  security_attributes.bInheritHandle = TRUE;
  security_attributes.lpSecurityDescriptor = nullptr;

  std::lock_guard<std::mutex> lock(create_process_mutex);
  if(stdin_fd) {
    if(!CreatePipe(&stdin_rd_p, &stdin_wr_p, &security_attributes, 0) ||
       !SetHandleInformation(stdin_wr_p, HANDLE_FLAG_INHERIT, 0))
      return 0;
  }
  if(stdout_fd) {
    if(!CreatePipe(&stdout_rd_p, &stdout_wr_p, &security_attributes, 0) ||
       !SetHandleInformation(stdout_rd_p, HANDLE_FLAG_INHERIT, 0)) {
      return 0;
    }
  }
  if(stderr_fd) {
    if(!CreatePipe(&stderr_rd_p, &stderr_wr_p, &security_attributes, 0) ||
       !SetHandleInformation(stderr_rd_p, HANDLE_FLAG_INHERIT, 0)) {
      return 0;
    }
  }

  PROCESS_INFORMATION process_info;
  STARTUPINFO startup_info;

  ZeroMemory(&process_info, sizeof(PROCESS_INFORMATION));

  ZeroMemory(&startup_info, sizeof(STARTUPINFO));
  startup_info.cb = sizeof(STARTUPINFO);
  startup_info.hStdInput = stdin_rd_p;
  startup_info.hStdOutput = stdout_wr_p;
  startup_info.hStdError = stderr_wr_p;
  if(stdin_fd || stdout_fd || stderr_fd)
    startup_info.dwFlags |= STARTF_USESTDHANDLES;

  if(config.show_window != Config::ShowWindow::show_default) {
    startup_info.dwFlags |= STARTF_USESHOWWINDOW;
    startup_info.wShowWindow = static_cast<WORD>(config.show_window);
  }

  auto process_command = command;
#ifdef MSYS_PROCESS_USE_SH
  size_t pos = 0;
  while((pos = process_command.find('\\', pos)) != string_type::npos) {
    process_command.replace(pos, 1, "\\\\\\\\");
    pos += 4;
  }
  pos = 0;
  while((pos = process_command.find('\"', pos)) != string_type::npos) {
    process_command.replace(pos, 1, "\\\"");
    pos += 2;
  }
  process_command.insert(0, "sh -c \"");
  process_command += "\"";
#endif

  DWORD creation_flags = stdin_fd || stdout_fd || stderr_fd ? CREATE_NO_WINDOW : 0; // CREATE_NO_WINDOW cannot be used when stdout or stderr is redirected to parent process
  string_type environment_str;
  if(environment) {
#ifdef UNICODE
    for(const auto &e : *environment)
      environment_str += e.first + L'=' + e.second + L'\0';
    if(!environment_str.empty())
      environment_str += L'\0';
    creation_flags |= CREATE_UNICODE_ENVIRONMENT;
#else
    for(const auto &e : *environment)
      environment_str += e.first + '=' + e.second + '\0';
    if(!environment_str.empty())
      environment_str += '\0';
#endif
  }
  BOOL bSuccess = CreateProcess(nullptr, process_command.empty() ? nullptr : &process_command[0], nullptr, nullptr,
                                stdin_fd || stdout_fd || stderr_fd || config.inherit_file_descriptors, // Cannot be false when stdout, stderr or stdin is used
                                creation_flags,
                                environment_str.empty() ? nullptr : &environment_str[0],
                                path.empty() ? nullptr : path.c_str(),
                                &startup_info, &process_info);

  if(!bSuccess)
    return 0;
  else
    CloseHandle(process_info.hThread);

  if(stdin_fd)
    *stdin_fd = stdin_wr_p.detach();
  if(stdout_fd)
    *stdout_fd = stdout_rd_p.detach();
  if(stderr_fd)
    *stderr_fd = stderr_rd_p.detach();

  closed = false;
  data.id = process_info.dwProcessId;
  data.handle = process_info.hProcess;
  return process_info.dwProcessId;
}

void Process::async_read() noexcept {
  if(data.id == 0)
    return;

  if(stdout_fd) {
    stdout_thread = std::thread([this]() {
      DWORD n;
      std::unique_ptr<char[]> buffer(new char[config.buffer_size]);
      for(;;) {
        BOOL bSuccess = ReadFile(*stdout_fd, static_cast<CHAR *>(buffer.get()), static_cast<DWORD>(config.buffer_size), &n, nullptr);
        if(!bSuccess || n == 0) {
          if(config.on_stdout_close)
            config.on_stdout_close();
          break;
        }
        read_stdout(buffer.get(), static_cast<size_t>(n));
      }
    });
  }
  if(stderr_fd) {
    stderr_thread = std::thread([this]() {
      DWORD n;
      std::unique_ptr<char[]> buffer(new char[config.buffer_size]);
      for(;;) {
        BOOL bSuccess = ReadFile(*stderr_fd, static_cast<CHAR *>(buffer.get()), static_cast<DWORD>(config.buffer_size), &n, nullptr);
        if(!bSuccess || n == 0) {
          if(config.on_stderr_close)
            config.on_stderr_close();
          break;
        }
        read_stderr(buffer.get(), static_cast<size_t>(n));
      }
    });
  }
}

int Process::get_exit_status() noexcept {
  if(data.id == 0)
    return -1;

  if(!data.handle)
    return data.exit_status;

  WaitForSingleObject(data.handle, INFINITE);

  DWORD exit_status;
  if(!GetExitCodeProcess(data.handle, &exit_status))
    data.exit_status = -1; // Store exit status for future calls
  else
    data.exit_status = static_cast<int>(exit_status); // Store exit status for future calls

  {
    std::lock_guard<std::mutex> lock(close_mutex);
    CloseHandle(data.handle);
    data.handle = nullptr;
    closed = true;
  }
  close_fds();

  return data.exit_status;
}

bool Process::try_get_exit_status(int &exit_status) noexcept {
  if(data.id == 0) {
    exit_status = -1;
    return true;
  }

  if(!data.handle) {
    exit_status = data.exit_status;
    return true;
  }

  DWORD wait_status = WaitForSingleObject(data.handle, 0);
  if(wait_status == WAIT_TIMEOUT)
    return false;

  DWORD exit_status_tmp;
  if(!GetExitCodeProcess(data.handle, &exit_status_tmp))
    exit_status = -1;
  else
    exit_status = static_cast<int>(exit_status_tmp);
  data.exit_status = exit_status; // Store exit status for future calls

  {
    std::lock_guard<std::mutex> lock(close_mutex);
    CloseHandle(data.handle);
    data.handle = nullptr;
    closed = true;
  }
  close_fds();

  return true;
}

void Process::close_fds() noexcept {
  if(stdout_thread.joinable())
    stdout_thread.join();
  if(stderr_thread.joinable())
    stderr_thread.join();

  if(stdin_fd)
    close_stdin();
  if(stdout_fd) {
    if(*stdout_fd != nullptr)
      CloseHandle(*stdout_fd);
    stdout_fd.reset();
  }
  if(stderr_fd) {
    if(*stderr_fd != nullptr)
      CloseHandle(*stderr_fd);
    stderr_fd.reset();
  }
}

bool Process::write(const char *bytes, size_t n) {
  if(!open_stdin)
    throw std::invalid_argument("Can't write to an unopened stdin pipe. Please set open_stdin=true when constructing the process.");

  std::lock_guard<std::mutex> lock(stdin_mutex);
  if(stdin_fd) {
    DWORD written;
    BOOL bSuccess = WriteFile(*stdin_fd, bytes, static_cast<DWORD>(n), &written, nullptr);
    if(!bSuccess || written == 0) {
      return false;
    }
    else {
      return true;
    }
  }
  return false;
}

void Process::close_stdin() noexcept {
  std::lock_guard<std::mutex> lock(stdin_mutex);
  if(stdin_fd) {
    if(*stdin_fd != nullptr)
      CloseHandle(*stdin_fd);
    stdin_fd.reset();
  }
}

// Based on http://stackoverflow.com/a/1173396
void Process::kill(bool /*force*/) noexcept {
  std::lock_guard<std::mutex> lock(close_mutex);
  if(data.id > 0 && !closed) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if(snapshot) {
      PROCESSENTRY32 process;
      ZeroMemory(&process, sizeof(process));
      process.dwSize = sizeof(process);
      if(Process32First(snapshot, &process)) {
        do {
          if(process.th32ParentProcessID == data.id) {
            HANDLE process_handle = OpenProcess(PROCESS_TERMINATE, FALSE, process.th32ProcessID);
            if(process_handle) {
              TerminateProcess(process_handle, 2);
              CloseHandle(process_handle);
            }
          }
        } while(Process32Next(snapshot, &process));
      }
      CloseHandle(snapshot);
    }
    TerminateProcess(data.handle, 2);
  }
}

// Based on http://stackoverflow.com/a/1173396
void Process::kill(id_type id, bool /*force*/) noexcept {
  if(id == 0)
    return;

  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if(snapshot) {
    PROCESSENTRY32 process;
    ZeroMemory(&process, sizeof(process));
    process.dwSize = sizeof(process);
    if(Process32First(snapshot, &process)) {
      do {
        if(process.th32ParentProcessID == id) {
          HANDLE process_handle = OpenProcess(PROCESS_TERMINATE, FALSE, process.th32ProcessID);
          if(process_handle) {
            TerminateProcess(process_handle, 2);
            CloseHandle(process_handle);
          }
        }
      } while(Process32Next(snapshot, &process));
    }
    CloseHandle(snapshot);
  }
  HANDLE process_handle = OpenProcess(PROCESS_TERMINATE, FALSE, id);
  if(process_handle)
    TerminateProcess(process_handle, 2);
}

} // namespace TinyProcessLib

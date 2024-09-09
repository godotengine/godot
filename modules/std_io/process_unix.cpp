#include "process.hpp"
#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <set>
#include <signal.h>
#include <stdexcept>
#include <string.h>
#include <unistd.h>

namespace TinyProcessLib {

static int portable_execvpe(const char *file, char *const argv[], char *const envp[]) {
#ifdef __GLIBC__
  // Prefer native implementation.
  return execvpe(file, argv, envp);
#else
  if(!file || !*file) {
    errno = ENOENT;
    return -1;
  }

  if(strchr(file, '/') != nullptr) {
    // If file contains a slash, no search is needed.
    return execve(file, argv, envp);
  }

  const char *path = getenv("PATH");
  char cspath[PATH_MAX + 1] = {};
  if(!path) {
    // If env variable is not set, use static path string.
    confstr(_CS_PATH, cspath, sizeof(cspath));
    path = cspath;
  }

  const size_t path_len = strlen(path);
  const size_t file_len = strlen(file);

  if(file_len > NAME_MAX) {
    errno = ENAMETOOLONG;
    return -1;
  }

  // Indicates whether we encountered EACCESS at least once.
  bool eacces = false;

  const char *curr = nullptr;
  const char *next = nullptr;

  for(curr = path; *curr; curr = *next ? next + 1 : next) {
    next = strchr(curr, ':');
    if(!next) {
      next = path + path_len;
    }

    const size_t sz = (next - curr);
    if(sz > PATH_MAX) {
      // Path is too long. Proceed to next path in list.
      continue;
    }

    char exe_path[PATH_MAX + 1 + NAME_MAX + 1]; // 1 byte for slash + 1 byte for \0
    memcpy(exe_path, curr, sz);
    exe_path[sz] = '/';
    memcpy(exe_path + sz + 1, file, file_len);
    exe_path[sz + 1 + file_len] = '\0';

    execve(exe_path, argv, envp);

    switch(errno) {
    case EACCES:
      eacces = true;
    case ENOENT:
    case ESTALE:
    case ENOTDIR:
    case ENODEV:
    case ETIMEDOUT:
      // The errors above indicate that the executable was not found.
      // The list of errors replicates one from glibc.
      // In this case we proceed to next path in list.
      break;

    default:
      // Other errors indicate that executable was found but failed
      // to execute. In this case we return error.
      return -1;
    }
  }

  if(eacces) {
    // If search failed, and at least one iteration reported EACCESS, it means
    // that the needed executable exists but does not have suitable permissions.
    // In this case we report EACEESS for user.
    errno = EACCES;
  }
  // Otherwise we just keep last encountered errno.
  return -1;
#endif
}

Process::Data::Data() noexcept : id(-1) {
}

Process::Process(const std::function<void()> &function,
                 std::function<void(const char *, size_t)> read_stdout,
                 std::function<void(const char *, size_t)> read_stderr,
                 bool open_stdin, const Config &config)
    : closed(true), read_stdout(std::move(read_stdout)), read_stderr(std::move(read_stderr)), open_stdin(open_stdin), config(config) {
  open(function);
  async_read();
}

Process::id_type Process::open(const std::function<void()> &function) noexcept {
  if(open_stdin)
    stdin_fd = std::unique_ptr<fd_type>(new fd_type);
  if(read_stdout)
    stdout_fd = std::unique_ptr<fd_type>(new fd_type);
  if(read_stderr)
    stderr_fd = std::unique_ptr<fd_type>(new fd_type);

  int stdin_p[2], stdout_p[2], stderr_p[2];

  if(stdin_fd && pipe(stdin_p) != 0)
    return -1;
  if(stdout_fd && pipe(stdout_p) != 0) {
    if(stdin_fd) {
      close(stdin_p[0]);
      close(stdin_p[1]);
    }
    return -1;
  }
  if(stderr_fd && pipe(stderr_p) != 0) {
    if(stdin_fd) {
      close(stdin_p[0]);
      close(stdin_p[1]);
    }
    if(stdout_fd) {
      close(stdout_p[0]);
      close(stdout_p[1]);
    }
    return -1;
  }

  id_type pid = fork();

  if(pid < 0) {
    if(stdin_fd) {
      close(stdin_p[0]);
      close(stdin_p[1]);
    }
    if(stdout_fd) {
      close(stdout_p[0]);
      close(stdout_p[1]);
    }
    if(stderr_fd) {
      close(stderr_p[0]);
      close(stderr_p[1]);
    }
    return pid;
  }
  else if(pid == 0) {
    if(stdin_fd)
      dup2(stdin_p[0], 0);
    if(stdout_fd)
      dup2(stdout_p[1], 1);
    if(stderr_fd)
      dup2(stderr_p[1], 2);
    if(stdin_fd) {
      close(stdin_p[0]);
      close(stdin_p[1]);
    }
    if(stdout_fd) {
      close(stdout_p[0]);
      close(stdout_p[1]);
    }
    if(stderr_fd) {
      close(stderr_p[0]);
      close(stderr_p[1]);
    }

    if(!config.inherit_file_descriptors) {
      // Optimization on some systems: using 8 * 1024 (Debian's default _SC_OPEN_MAX) as fd_max limit
      int fd_max = std::min(8192, static_cast<int>(sysconf(_SC_OPEN_MAX))); // Truncation is safe
      if(fd_max < 0)
        fd_max = 8192;
      for(int fd = 3; fd < fd_max; fd++)
        close(fd);
    }

    setpgid(0, 0);
    // TODO: See here on how to emulate tty for colors: http://stackoverflow.com/questions/1401002/trick-an-application-into-thinking-its-stdin-is-interactive-not-a-pipe
    // TODO: One solution is: echo "command;exit"|script -q /dev/null

    if(function)
      function();

    _exit(EXIT_FAILURE);
  }

  if(stdin_fd)
    close(stdin_p[0]);
  if(stdout_fd)
    close(stdout_p[1]);
  if(stderr_fd)
    close(stderr_p[1]);

  if(stdin_fd)
    *stdin_fd = stdin_p[1];
  if(stdout_fd)
    *stdout_fd = stdout_p[0];
  if(stderr_fd)
    *stderr_fd = stderr_p[0];

  closed = false;
  data.id = pid;
  return pid;
}

Process::id_type Process::open(const std::vector<string_type> &arguments, const string_type &path, const environment_type *environment) noexcept {
  return open([this, &arguments, &path, &environment] {
    if(arguments.empty())
      exit(127);

    std::vector<const char *> argv_ptrs;

    if(config.flatpak_spawn_host) {
      // break out of sandbox, execute on host
      argv_ptrs.reserve(arguments.size() + 3);
      argv_ptrs.emplace_back("/usr/bin/flatpak-spawn");
      argv_ptrs.emplace_back("--host");
    }
    else
      argv_ptrs.reserve(arguments.size() + 1);

    for(auto &argument : arguments)
      argv_ptrs.emplace_back(argument.c_str());
    argv_ptrs.emplace_back(nullptr);

    if(!path.empty()) {
      if(chdir(path.c_str()) != 0)
        exit(1);
    }

    if(!environment)
      execvp(argv_ptrs[0], const_cast<char *const *>(argv_ptrs.data()));
    else {
      std::vector<std::string> env_strs;
      std::vector<const char *> env_ptrs;
      env_strs.reserve(environment->size());
      env_ptrs.reserve(environment->size() + 1);
      for(const auto &e : *environment) {
        env_strs.emplace_back(e.first + '=' + e.second);
        env_ptrs.emplace_back(env_strs.back().c_str());
      }
      env_ptrs.emplace_back(nullptr);

      portable_execvpe(argv_ptrs[0], const_cast<char *const *>(argv_ptrs.data()), const_cast<char *const *>(env_ptrs.data()));
    }
  });
}

Process::id_type Process::open(const std::string &command, const std::string &path, const environment_type *environment) noexcept {
  return open([this, &command, &path, &environment] {
    auto command_c_str = command.c_str();
    std::string cd_path_and_command;
    if(!path.empty()) {
      auto path_escaped = path;
      size_t pos = 0;
      // Based on https://www.reddit.com/r/cpp/comments/3vpjqg/a_new_platform_independent_process_library_for_c11/cxsxyb7
      while((pos = path_escaped.find('\'', pos)) != std::string::npos) {
        path_escaped.replace(pos, 1, "'\\''");
        pos += 4;
      }
      cd_path_and_command = "cd '" + path_escaped + "' && " + command; // To avoid resolving symbolic links
      command_c_str = cd_path_and_command.c_str();
    }

    if(!environment) {
      if(config.flatpak_spawn_host)
        // break out of sandbox, execute on host
        execl("/usr/bin/flatpak-spawn", "/usr/bin/flatpak-spawn", "--host", "/bin/sh", "-c", command_c_str, nullptr);
      else
        execl("/bin/sh", "/bin/sh", "-c", command_c_str, nullptr);
    }
    else {
      std::vector<std::string> env_strs;
      std::vector<const char *> env_ptrs;
      env_strs.reserve(environment->size());
      env_ptrs.reserve(environment->size() + 1);
      for(const auto &e : *environment) {
        env_strs.emplace_back(e.first + '=' + e.second);
        env_ptrs.emplace_back(env_strs.back().c_str());
      }
      env_ptrs.emplace_back(nullptr);
      if(config.flatpak_spawn_host)
        // break out of sandbox, execute on host
        execle("/usr/bin/flatpak-spawn", "/usr/bin/flatpak-spawn", "--host", "/bin/sh", "-c", command_c_str, nullptr, env_ptrs.data());
      else
        execle("/bin/sh", "/bin/sh", "-c", command_c_str, nullptr, env_ptrs.data());
    }
  });
}

void Process::async_read() noexcept {
  if(data.id <= 0 || (!stdout_fd && !stderr_fd))
    return;

  stdout_stderr_thread = std::thread([this] {
    std::vector<pollfd> pollfds;
    std::bitset<2> fd_is_stdout;
    if(stdout_fd) {
      fd_is_stdout.set(pollfds.size());
      pollfds.emplace_back();
      pollfds.back().fd = fcntl(*stdout_fd, F_SETFL, fcntl(*stdout_fd, F_GETFL) | O_NONBLOCK) == 0 ? *stdout_fd : -1;
      pollfds.back().events = POLLIN;
    }
    if(stderr_fd) {
      pollfds.emplace_back();
      pollfds.back().fd = fcntl(*stderr_fd, F_SETFL, fcntl(*stderr_fd, F_GETFL) | O_NONBLOCK) == 0 ? *stderr_fd : -1;
      pollfds.back().events = POLLIN;
    }
    auto buffer = std::unique_ptr<char[]>(new char[config.buffer_size]);
    bool any_open = !pollfds.empty();
    while(any_open && (poll(pollfds.data(), static_cast<nfds_t>(pollfds.size()), -1) > 0 || errno == EINTR)) {
      any_open = false;
      for(size_t i = 0; i < pollfds.size(); ++i) {
        if(pollfds[i].fd >= 0) {
          if(pollfds[i].revents & POLLIN) {
            const ssize_t n = read(pollfds[i].fd, buffer.get(), config.buffer_size);
            if(n > 0) {
              if(fd_is_stdout[i])
                read_stdout(buffer.get(), static_cast<size_t>(n));
              else
                read_stderr(buffer.get(), static_cast<size_t>(n));
            }
            else if(n == 0 || (n < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)) {
              if(fd_is_stdout[i]) {
                if(config.on_stdout_close)
                  config.on_stdout_close();
              }
              else {
                if(config.on_stderr_close)
                  config.on_stderr_close();
              }
              pollfds[i].fd = -1;
              continue;
            }
          }
          else if(pollfds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) {
            if(fd_is_stdout[i]) {
              if(config.on_stdout_close)
                config.on_stdout_close();
            }
            else {
              if(config.on_stderr_close)
                config.on_stderr_close();
            }
            pollfds[i].fd = -1;
            continue;
          }
          any_open = true;
        }
      }
    }
  });
}

int Process::get_exit_status() noexcept {
  if(data.id <= 0)
    return -1;

  int exit_status;
  id_type pid;
  do {
    pid = waitpid(data.id, &exit_status, 0);
  } while(pid < 0 && errno == EINTR);

  if(pid < 0 && errno == ECHILD) {
    // PID doesn't exist anymore, return previously sampled exit status (or -1)
    return data.exit_status;
  }
  else {
    // Store exit status for future calls
    if(exit_status >= 256)
      exit_status = exit_status >> 8;
    data.exit_status = exit_status;
  }

  {
    std::lock_guard<std::mutex> lock(close_mutex);
    closed = true;
  }
  close_fds();

  return exit_status;
}

bool Process::try_get_exit_status(int &exit_status) noexcept {
  if(data.id <= 0) {
    exit_status = -1;
    return true;
  }

  const id_type pid = waitpid(data.id, &exit_status, WNOHANG);
  if(pid < 0 && errno == ECHILD) {
    // PID doesn't exist anymore, set previously sampled exit status (or -1)
    exit_status = data.exit_status;
    return true;
  }
  else if(pid <= 0) {
    // Process still running (p==0) or error
    return false;
  }
  else {
    // store exit status for future calls
    if(exit_status >= 256)
      exit_status = exit_status >> 8;
    data.exit_status = exit_status;
  }

  {
    std::lock_guard<std::mutex> lock(close_mutex);
    closed = true;
  }
  close_fds();

  return true;
}

void Process::close_fds() noexcept {
  if(stdout_stderr_thread.joinable())
    stdout_stderr_thread.join();

  if(stdin_fd)
    close_stdin();
  if(stdout_fd) {
    if(data.id > 0)
      close(*stdout_fd);
    stdout_fd.reset();
  }
  if(stderr_fd) {
    if(data.id > 0)
      close(*stderr_fd);
    stderr_fd.reset();
  }
}

bool Process::write(const char *bytes, size_t n) {

  std::lock_guard<std::mutex> lock(stdin_mutex);
  if(stdin_fd) {
    while(n != 0) {
      const ssize_t ret = ::write(*stdin_fd, bytes, n);
      if(ret < 0) {
        if(errno == EINTR)
          continue;
        else
          return false;
      }
      bytes += static_cast<size_t>(ret);
      n -= static_cast<size_t>(ret);
    }
    return true;
  }
  return false;
}

void Process::close_stdin() noexcept {
  std::lock_guard<std::mutex> lock(stdin_mutex);
  if(stdin_fd) {
    if(data.id > 0)
      close(*stdin_fd);
    stdin_fd.reset();
  }
}

void Process::kill(bool force) noexcept {
  std::lock_guard<std::mutex> lock(close_mutex);
  if(data.id > 0 && !closed) {
    if(force) {
      ::kill(-data.id, SIGTERM);
      ::kill(data.id, SIGTERM); // Based on comment in https://gitlab.com/eidheim/tiny-process-library/-/merge_requests/29#note_1146144166
    }
    else {
      ::kill(-data.id, SIGINT);
      ::kill(data.id, SIGINT);
    }
  }
}

void Process::kill(id_type id, bool force) noexcept {
  if(id <= 0)
    return;

  if(force) {
    ::kill(-id, SIGTERM);
    ::kill(id, SIGTERM);
  }
  else {
    ::kill(-id, SIGINT);
    ::kill(id, SIGINT);
  }
}

void Process::signal(int signum) noexcept {
  std::lock_guard<std::mutex> lock(close_mutex);
  if(data.id > 0 && !closed) {
    ::kill(-data.id, signum);
    ::kill(data.id, signum);
  }
}

} // namespace TinyProcessLib

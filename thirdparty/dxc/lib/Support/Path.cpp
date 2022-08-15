//===-- Path.cpp - Implement OS Path Concept ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the operating system Path API.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include <cctype>
#include <cstring>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using namespace llvm;
using namespace llvm::support::endian;

namespace {
  using llvm::StringRef;
  using llvm::sys::path::is_separator;

#ifdef LLVM_ON_WIN32
  const char *separators = "\\/";
  const char preferred_separator = '\\';
#else
  const char  separators = '/';
  const char preferred_separator = '/';
#endif

  StringRef find_first_component(StringRef path) {
    // Look for this first component in the following order.
    // * empty (in this case we return an empty string)
    // * either C: or {//,\\}net.
    // * {/,\}
    // * {file,directory}name

    if (path.empty())
      return path;

#ifdef LLVM_ON_WIN32
    // C:
    if (path.size() >= 2 && std::isalpha(static_cast<unsigned char>(path[0])) &&
        path[1] == ':')
      return path.substr(0, 2);
#endif

    // //net
    if ((path.size() > 2) &&
        is_separator(path[0]) &&
        path[0] == path[1] &&
        !is_separator(path[2])) {
      // Find the next directory separator.
      size_t end = path.find_first_of(separators, 2);
      return path.substr(0, end);
    }

    // {/,\}
    if (is_separator(path[0]))
      return path.substr(0, 1);

    // * {file,directory}name
    size_t end = path.find_first_of(separators);
    return path.substr(0, end);
  }

  size_t filename_pos(StringRef str) {
    if (str.size() == 2 &&
        is_separator(str[0]) &&
        str[0] == str[1])
      return 0;

    if (str.size() > 0 && is_separator(str[str.size() - 1]))
      return str.size() - 1;

    size_t pos = str.find_last_of(separators, str.size() - 1);

#ifdef LLVM_ON_WIN32
    if (pos == StringRef::npos)
      pos = str.find_last_of(':', str.size() - 2);
#endif

    if (pos == StringRef::npos ||
        (pos == 1 && is_separator(str[0])))
      return 0;

    return pos + 1;
  }

  size_t root_dir_start(StringRef str) {
    // case "c:/"
#ifdef LLVM_ON_WIN32
    if (str.size() > 2 &&
        str[1] == ':' &&
        is_separator(str[2]))
      return 2;
#endif

    // case "//"
    if (str.size() == 2 &&
        is_separator(str[0]) &&
        str[0] == str[1])
      return StringRef::npos;

    // case "//net"
    if (str.size() > 3 &&
        is_separator(str[0]) &&
        str[0] == str[1] &&
        !is_separator(str[2])) {
      return str.find_first_of(separators, 2);
    }

    // case "/"
    if (str.size() > 0 && is_separator(str[0]))
      return 0;

    return StringRef::npos;
  }

  size_t parent_path_end(StringRef path) {
    size_t end_pos = filename_pos(path);

    bool filename_was_sep = path.size() > 0 && is_separator(path[end_pos]);

    // Skip separators except for root dir.
    size_t root_dir_pos = root_dir_start(path.substr(0, end_pos));

    while(end_pos > 0 &&
          (end_pos - 1) != root_dir_pos &&
          is_separator(path[end_pos - 1]))
      --end_pos;

    if (end_pos == 1 && root_dir_pos == 0 && filename_was_sep)
      return StringRef::npos;

    return end_pos;
  }
} // end unnamed namespace

enum FSEntity {
  FS_Dir,
  FS_File,
  FS_Name
};

static std::error_code createUniqueEntity(const Twine &Model, int &ResultFD,
                                          SmallVectorImpl<char> &ResultPath,
                                          bool MakeAbsolute, unsigned Mode,
                                          FSEntity Type) {
  SmallString<128> ModelStorage;
  Model.toVector(ModelStorage);

  if (MakeAbsolute) {
    // Make model absolute by prepending a temp directory if it's not already.
    if (!sys::path::is_absolute(Twine(ModelStorage))) {
      SmallString<128> TDir;
      sys::path::system_temp_directory(true, TDir);
      sys::path::append(TDir, Twine(ModelStorage));
      ModelStorage.swap(TDir);
    }
  }

  // From here on, DO NOT modify model. It may be needed if the randomly chosen
  // path already exists.
  ResultPath = ModelStorage;
  // Null terminate.
  ResultPath.push_back(0);
  ResultPath.pop_back();

retry_random_path:
  // Replace '%' with random chars.
  for (unsigned i = 0, e = ModelStorage.size(); i != e; ++i) {
    if (ModelStorage[i] == '%')
      ResultPath[i] = "0123456789abcdef"[sys::Process::GetRandomNumber() & 15];
  }

  // Try to open + create the file.
  switch (Type) {
  case FS_File: {
    if (std::error_code EC =
            sys::fs::openFileForWrite(Twine(ResultPath.begin()), ResultFD,
                                      sys::fs::F_RW | sys::fs::F_Excl, Mode)) {
      if (EC == errc::file_exists)
        goto retry_random_path;
      return EC;
    }

    return std::error_code();
  }

  case FS_Name: {
    std::error_code EC =
        sys::fs::access(ResultPath.begin(), sys::fs::AccessMode::Exist);
    if (EC == errc::no_such_file_or_directory)
      return std::error_code();
    if (EC)
      return EC;
    goto retry_random_path;
  }

  case FS_Dir: {
    if (std::error_code EC =
            sys::fs::create_directory(ResultPath.begin(), false)) {
      if (EC == errc::file_exists)
        goto retry_random_path;
      return EC;
    }
    return std::error_code();
  }
  }
  llvm_unreachable("Invalid Type");
}

namespace llvm {
namespace sys  {
namespace path {

const_iterator begin(StringRef path) {
  const_iterator i;
  i.Path      = path;
  i.Component = find_first_component(path);
  i.Position  = 0;
  return i;
}

const_iterator end(StringRef path) {
  const_iterator i;
  i.Path      = path;
  i.Position  = path.size();
  return i;
}

const_iterator &const_iterator::operator++() {
  assert(Position < Path.size() && "Tried to increment past end!");

  // Increment Position to past the current component
  Position += Component.size();

  // Check for end.
  if (Position == Path.size()) {
    Component = StringRef();
    return *this;
  }

  // Both POSIX and Windows treat paths that begin with exactly two separators
  // specially.
  bool was_net = Component.size() > 2 &&
    is_separator(Component[0]) &&
    Component[1] == Component[0] &&
    !is_separator(Component[2]);

  // Handle separators.
  if (is_separator(Path[Position])) {
    // Root dir.
    if (was_net
#ifdef LLVM_ON_WIN32
        // c:/
        || Component.endswith(":")
#endif
        ) {
      Component = Path.substr(Position, 1);
      return *this;
    }

    // Skip extra separators.
    while (Position != Path.size() &&
           is_separator(Path[Position])) {
      ++Position;
    }

    // Treat trailing '/' as a '.'.
    if (Position == Path.size()) {
      --Position;
      Component = ".";
      return *this;
    }
  }

  // Find next component.
  size_t end_pos = Path.find_first_of(separators, Position);
  Component = Path.slice(Position, end_pos);

  return *this;
}

bool const_iterator::operator==(const const_iterator &RHS) const {
  return Path.begin() == RHS.Path.begin() && Position == RHS.Position;
}

ptrdiff_t const_iterator::operator-(const const_iterator &RHS) const {
  return Position - RHS.Position;
}

reverse_iterator rbegin(StringRef Path) {
  reverse_iterator I;
  I.Path = Path;
  I.Position = Path.size();
  return ++I;
}

reverse_iterator rend(StringRef Path) {
  reverse_iterator I;
  I.Path = Path;
  I.Component = Path.substr(0, 0);
  I.Position = 0;
  return I;
}

reverse_iterator &reverse_iterator::operator++() {
  // If we're at the end and the previous char was a '/', return '.' unless
  // we are the root path.
  size_t root_dir_pos = root_dir_start(Path);
  if (Position == Path.size() &&
      Path.size() > root_dir_pos + 1 &&
      is_separator(Path[Position - 1])) {
    --Position;
    Component = ".";
    return *this;
  }

  // Skip separators unless it's the root directory.
  size_t end_pos = Position;

  while(end_pos > 0 &&
        (end_pos - 1) != root_dir_pos &&
        is_separator(Path[end_pos - 1]))
    --end_pos;

  // Find next separator.
  size_t start_pos = filename_pos(Path.substr(0, end_pos));
  Component = Path.slice(start_pos, end_pos);
  Position = start_pos;
  return *this;
}

bool reverse_iterator::operator==(const reverse_iterator &RHS) const {
  return Path.begin() == RHS.Path.begin() && Component == RHS.Component &&
         Position == RHS.Position;
}

StringRef root_path(StringRef path) {
  const_iterator b = begin(path),
                 pos = b,
                 e = end(path);
  if (b != e) {
    bool has_net = b->size() > 2 && is_separator((*b)[0]) && (*b)[1] == (*b)[0];
    bool has_drive =
#ifdef LLVM_ON_WIN32
      b->endswith(":");
#else
      false;
#endif

    if (has_net || has_drive) {
      if ((++pos != e) && is_separator((*pos)[0])) {
        // {C:/,//net/}, so get the first two components.
        return path.substr(0, b->size() + pos->size());
      } else {
        // just {C:,//net}, return the first component.
        return *b;
      }
    }

    // POSIX style root directory.
    if (is_separator((*b)[0])) {
      return *b;
    }
  }

  return StringRef();
}

StringRef root_name(StringRef path) {
  const_iterator b = begin(path),
                 e = end(path);
  if (b != e) {
    bool has_net = b->size() > 2 && is_separator((*b)[0]) && (*b)[1] == (*b)[0];
    bool has_drive =
#ifdef LLVM_ON_WIN32
      b->endswith(":");
#else
      false;
#endif

    if (has_net || has_drive) {
      // just {C:,//net}, return the first component.
      return *b;
    }
  }

  // No path or no name.
  return StringRef();
}

StringRef root_directory(StringRef path) {
  const_iterator b = begin(path),
                 pos = b,
                 e = end(path);
  if (b != e) {
    bool has_net = b->size() > 2 && is_separator((*b)[0]) && (*b)[1] == (*b)[0];
    bool has_drive =
#ifdef LLVM_ON_WIN32
      b->endswith(":");
#else
      false;
#endif

    if ((has_net || has_drive) &&
        // {C:,//net}, skip to the next component.
        (++pos != e) && is_separator((*pos)[0])) {
      return *pos;
    }

    // POSIX style root directory.
    if (!has_net && is_separator((*b)[0])) {
      return *b;
    }
  }

  // No path or no root.
  return StringRef();
}

StringRef relative_path(StringRef path) {
  StringRef root = root_path(path);
  return path.substr(root.size());
}

void append(SmallVectorImpl<char> &path, const Twine &a,
                                         const Twine &b,
                                         const Twine &c,
                                         const Twine &d) {
  SmallString<32> a_storage;
  SmallString<32> b_storage;
  SmallString<32> c_storage;
  SmallString<32> d_storage;

  SmallVector<StringRef, 4> components;
  if (!a.isTriviallyEmpty()) components.push_back(a.toStringRef(a_storage));
  if (!b.isTriviallyEmpty()) components.push_back(b.toStringRef(b_storage));
  if (!c.isTriviallyEmpty()) components.push_back(c.toStringRef(c_storage));
  if (!d.isTriviallyEmpty()) components.push_back(d.toStringRef(d_storage));

  for (SmallVectorImpl<StringRef>::const_iterator i = components.begin(),
                                                  e = components.end();
                                                  i != e; ++i) {
    bool path_has_sep = !path.empty() && is_separator(path[path.size() - 1]);
    bool component_has_sep = !i->empty() && is_separator((*i)[0]);
    bool is_root_name = has_root_name(*i);

    if (path_has_sep) {
      // Strip separators from beginning of component.
      size_t loc = i->find_first_not_of(separators);
      StringRef c = i->substr(loc);

      // Append it.
      path.append(c.begin(), c.end());
      continue;
    }

    if (!component_has_sep && !(path.empty() || is_root_name)) {
      // Add a separator.
      path.push_back(preferred_separator);
    }

    path.append(i->begin(), i->end());
  }
}

void append(SmallVectorImpl<char> &path,
            const_iterator begin, const_iterator end) {
  for (; begin != end; ++begin)
    path::append(path, *begin);
}

StringRef parent_path(StringRef path) {
  size_t end_pos = parent_path_end(path);
  if (end_pos == StringRef::npos)
    return StringRef();
  else
    return path.substr(0, end_pos);
}

void remove_filename(SmallVectorImpl<char> &path) {
  size_t end_pos = parent_path_end(StringRef(path.begin(), path.size()));
  if (end_pos != StringRef::npos)
    path.set_size(end_pos);
}

void replace_extension(SmallVectorImpl<char> &path, const Twine &extension) {
  StringRef p(path.begin(), path.size());
  SmallString<32> ext_storage;
  StringRef ext = extension.toStringRef(ext_storage);

  // Erase existing extension.
  size_t pos = p.find_last_of('.');
  if (pos != StringRef::npos && pos >= filename_pos(p))
    path.set_size(pos);

  // Append '.' if needed.
  if (ext.size() > 0 && ext[0] != '.')
    path.push_back('.');

  // Append extension.
  path.append(ext.begin(), ext.end());
}

void native(const Twine &path, SmallVectorImpl<char> &result) {
  assert((!path.isSingleStringRef() ||
          path.getSingleStringRef().data() != result.data()) &&
         "path and result are not allowed to overlap!");
  // Clear result.
  result.clear();
  path.toVector(result);
  native(result);
}

void native(SmallVectorImpl<char> &Path) {
#ifdef LLVM_ON_WIN32
  std::replace(Path.begin(), Path.end(), '/', '\\');
#else
  for (auto PI = Path.begin(), PE = Path.end(); PI < PE; ++PI) {
    if (*PI == '\\') {
      auto PN = PI + 1;
      if (PN < PE && *PN == '\\')
        ++PI; // increment once, the for loop will move over the escaped slash
      else
        *PI = '/';
    }
  }
#endif
}

StringRef filename(StringRef path) {
  return *rbegin(path);
}

StringRef stem(StringRef path) {
  StringRef fname = filename(path);
  size_t pos = fname.find_last_of('.');
  if (pos == StringRef::npos)
    return fname;
  else
    if ((fname.size() == 1 && fname == ".") ||
        (fname.size() == 2 && fname == ".."))
      return fname;
    else
      return fname.substr(0, pos);
}

StringRef extension(StringRef path) {
  StringRef fname = filename(path);
  size_t pos = fname.find_last_of('.');
  if (pos == StringRef::npos)
    return StringRef();
  else
    if ((fname.size() == 1 && fname == ".") ||
        (fname.size() == 2 && fname == ".."))
      return StringRef();
    else
      return fname.substr(pos);
}

bool is_separator(char value) {
  switch(value) {
#ifdef LLVM_ON_WIN32
    case '\\': // fall through
#endif
    case '/': return true;
    default: return false;
  }
}

static const char preferred_separator_string[] = { preferred_separator, '\0' };

StringRef get_separator() {
  return preferred_separator_string;
}

bool has_root_name(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !root_name(p).empty();
}

bool has_root_directory(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !root_directory(p).empty();
}

bool has_root_path(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !root_path(p).empty();
}

bool has_relative_path(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !relative_path(p).empty();
}

bool has_filename(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !filename(p).empty();
}

bool has_parent_path(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !parent_path(p).empty();
}

bool has_stem(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !stem(p).empty();
}

bool has_extension(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !extension(p).empty();
}

bool is_absolute(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  bool rootDir = has_root_directory(p),
#ifdef LLVM_ON_WIN32
       rootName = has_root_name(p);
#else
       rootName = true;
#endif

  return rootDir && rootName;
}

bool is_relative(const Twine &path) {
  return !is_absolute(path);
}

} // end namespace path

namespace fs {

std::error_code getUniqueID(const Twine Path, UniqueID &Result) {
  file_status Status;
  std::error_code EC = status(Path, Status);
  if (EC)
    return EC;
  Result = Status.getUniqueID();
  return std::error_code();
}

std::error_code createUniqueFile(const Twine &Model, int &ResultFd,
                                 SmallVectorImpl<char> &ResultPath,
                                 unsigned Mode) {
  return createUniqueEntity(Model, ResultFd, ResultPath, false, Mode, FS_File);
}

std::error_code createUniqueFile(const Twine &Model,
                                 SmallVectorImpl<char> &ResultPath) {
  int Dummy;
  return createUniqueEntity(Model, Dummy, ResultPath, false, 0, FS_Name);
}

static std::error_code
createTemporaryFile(const Twine &Model, int &ResultFD,
                    llvm::SmallVectorImpl<char> &ResultPath, FSEntity Type) {
  SmallString<128> Storage;
  StringRef P = Model.toNullTerminatedStringRef(Storage);
  assert(P.find_first_of(separators) == StringRef::npos &&
         "Model must be a simple filename.");
  // Use P.begin() so that createUniqueEntity doesn't need to recreate Storage.
  return createUniqueEntity(P.begin(), ResultFD, ResultPath,
                            true, owner_read | owner_write, Type);
}

static std::error_code
createTemporaryFile(const Twine &Prefix, StringRef Suffix, int &ResultFD,
                    llvm::SmallVectorImpl<char> &ResultPath, FSEntity Type) {
  const char *Middle = Suffix.empty() ? "-%%%%%%" : "-%%%%%%.";
  return createTemporaryFile(Prefix + Middle + Suffix, ResultFD, ResultPath,
                             Type);
}

std::error_code createTemporaryFile(const Twine &Prefix, StringRef Suffix,
                                    int &ResultFD,
                                    SmallVectorImpl<char> &ResultPath) {
  return createTemporaryFile(Prefix, Suffix, ResultFD, ResultPath, FS_File);
}

std::error_code createTemporaryFile(const Twine &Prefix, StringRef Suffix,
                                    SmallVectorImpl<char> &ResultPath) {
  int Dummy;
  return createTemporaryFile(Prefix, Suffix, Dummy, ResultPath, FS_Name);
}


// This is a mkdtemp with a different pattern. We use createUniqueEntity mostly
// for consistency. We should try using mkdtemp.
std::error_code createUniqueDirectory(const Twine &Prefix,
                                      SmallVectorImpl<char> &ResultPath) {
  int Dummy;
  return createUniqueEntity(Prefix + "-%%%%%%", Dummy, ResultPath,
                            true, 0, FS_Dir);
}

std::error_code make_absolute(SmallVectorImpl<char> &path) {
  StringRef p(path.data(), path.size());

  bool rootDirectory = path::has_root_directory(p),
#ifdef LLVM_ON_WIN32
       rootName = path::has_root_name(p);
#else
       rootName = true;
#endif

  // Already absolute.
  if (rootName && rootDirectory)
    return std::error_code();

  // All of the following conditions will need the current directory.
  SmallString<128> current_dir;
  if (std::error_code ec = current_path(current_dir))
    return ec;

  // Relative path. Prepend the current directory.
  if (!rootName && !rootDirectory) {
    // Append path to the current directory.
    path::append(current_dir, p);
    // Set path to the result.
    path.swap(current_dir);
    return std::error_code();
  }

  if (!rootName && rootDirectory) {
    StringRef cdrn = path::root_name(current_dir);
    SmallString<128> curDirRootName(cdrn.begin(), cdrn.end());
    path::append(curDirRootName, p);
    // Set path to the result.
    path.swap(curDirRootName);
    return std::error_code();
  }

  if (rootName && !rootDirectory) {
    StringRef pRootName      = path::root_name(p);
    StringRef bRootDirectory = path::root_directory(current_dir);
    StringRef bRelativePath  = path::relative_path(current_dir);
    StringRef pRelativePath  = path::relative_path(p);

    SmallString<128> res;
    path::append(res, pRootName, bRootDirectory, bRelativePath, pRelativePath);
    path.swap(res);
    return std::error_code();
  }

  llvm_unreachable("All rootName and rootDirectory combinations should have "
                   "occurred above!");
}

std::error_code create_directories(const Twine &Path, bool IgnoreExisting) {
  SmallString<128> PathStorage;
  StringRef P = Path.toStringRef(PathStorage);

  // Be optimistic and try to create the directory
  std::error_code EC = create_directory(P, IgnoreExisting);
  // If we succeeded, or had any error other than the parent not existing, just
  // return it.
  if (EC != errc::no_such_file_or_directory)
    return EC;

  // We failed because of a no_such_file_or_directory, try to create the
  // parent.
  StringRef Parent = path::parent_path(P);
  if (Parent.empty())
    return EC;

  if ((EC = create_directories(Parent)))
      return EC;

  return create_directory(P, IgnoreExisting);
}

std::error_code copy_file(const Twine &From, const Twine &To) {
  int ReadFD, WriteFD;
  if (std::error_code EC = openFileForRead(From, ReadFD))
    return EC;
  if (std::error_code EC = openFileForWrite(To, WriteFD, F_None)) {
    msf_close(ReadFD); // HLSL Change
    return EC;
  }

  const size_t BufSize = 4096;
  char *Buf = new char[BufSize];
  int BytesRead = 0, BytesWritten = 0;
  for (;;) {
    BytesRead = msf_read(ReadFD, Buf, BufSize); // HLSL Change
    if (BytesRead <= 0)
      break;
    while (BytesRead) {
      BytesWritten = msf_write(WriteFD, Buf, BytesRead); // HLSL Change
      if (BytesWritten < 0)
        break;
      BytesRead -= BytesWritten;
    }
    if (BytesWritten < 0)
      break;
  }
  msf_close(ReadFD); // HLSL Change
  msf_close(WriteFD); // HLSL Change
  delete[] Buf;

  if (BytesRead < 0 || BytesWritten < 0)
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

bool exists(file_status status) {
  return status_known(status) && status.type() != file_type::file_not_found;
}

bool status_known(file_status s) {
  return s.type() != file_type::status_error;
}

bool is_directory(file_status status) {
  return status.type() == file_type::directory_file;
}

std::error_code is_directory(const Twine &path, bool &result) {
  file_status st;
  if (std::error_code ec = status(path, st))
    return ec;
  result = is_directory(st);
  return std::error_code();
}

bool is_regular_file(file_status status) {
  return status.type() == file_type::regular_file;
}

std::error_code is_regular_file(const Twine &path, bool &result) {
  file_status st;
  if (std::error_code ec = status(path, st))
    return ec;
  result = is_regular_file(st);
  return std::error_code();
}

bool is_other(file_status status) {
  return exists(status) &&
         !is_regular_file(status) &&
         !is_directory(status);
}

std::error_code is_other(const Twine &Path, bool &Result) {
  file_status FileStatus;
  if (std::error_code EC = status(Path, FileStatus))
    return EC;
  Result = is_other(FileStatus);
  return std::error_code();
}

void directory_entry::replace_filename(const Twine &filename, file_status st) {
  SmallString<128> path(Path.begin(), Path.end());
  path::remove_filename(path);
  path::append(path, filename);
  Path = path.str();
  Status = st;
}

/// @brief Identify the magic in magic.
file_magic identify_magic(StringRef Magic) {
  if (Magic.size() < 4)
    return file_magic::unknown;
  switch ((unsigned char)Magic[0]) {
    case 0x00: {
      // COFF bigobj or short import library file
      if (Magic[1] == (char)0x00 && Magic[2] == (char)0xff &&
          Magic[3] == (char)0xff) {
        size_t MinSize = offsetof(COFF::BigObjHeader, UUID) + sizeof(COFF::BigObjMagic);
        if (Magic.size() < MinSize)
          return file_magic::coff_import_library;

        int BigObjVersion = read16le(
            Magic.data() + offsetof(COFF::BigObjHeader, Version));
        if (BigObjVersion < COFF::BigObjHeader::MinBigObjectVersion)
          return file_magic::coff_import_library;

        const char *Start = Magic.data() + offsetof(COFF::BigObjHeader, UUID);
        if (memcmp(Start, COFF::BigObjMagic, sizeof(COFF::BigObjMagic)) != 0)
          return file_magic::coff_import_library;
        return file_magic::coff_object;
      }
      // Windows resource file
      const char Expected[] = { 0, 0, 0, 0, '\x20', 0, 0, 0, '\xff' };
      if (Magic.size() >= sizeof(Expected) &&
          memcmp(Magic.data(), Expected, sizeof(Expected)) == 0)
        return file_magic::windows_resource;
      // 0x0000 = COFF unknown machine type
      if (Magic[1] == 0)
        return file_magic::coff_object;
      break;
    }
    case 0xDE:  // 0x0B17C0DE = BC wraper
      if (Magic[1] == (char)0xC0 && Magic[2] == (char)0x17 &&
          Magic[3] == (char)0x0B)
        return file_magic::bitcode;
      break;
    case 'B':
      if (Magic[1] == 'C' && Magic[2] == (char)0xC0 && Magic[3] == (char)0xDE)
        return file_magic::bitcode;
      break;
    case '!':
      if (Magic.size() >= 8)
        if (memcmp(Magic.data(),"!<arch>\n",8) == 0)
          return file_magic::archive;
      break;

    case '\177':
      if (Magic.size() >= 18 && Magic[1] == 'E' && Magic[2] == 'L' &&
          Magic[3] == 'F') {
        bool Data2MSB = Magic[5] == 2;
        unsigned high = Data2MSB ? 16 : 17;
        unsigned low  = Data2MSB ? 17 : 16;
        if (Magic[high] == 0)
          switch (Magic[low]) {
            default: return file_magic::elf;
            case 1: return file_magic::elf_relocatable;
            case 2: return file_magic::elf_executable;
            case 3: return file_magic::elf_shared_object;
            case 4: return file_magic::elf_core;
          }
        else
          // It's still some type of ELF file.
          return file_magic::elf;
      }
      break;

    case 0xCA:
      if (Magic[1] == char(0xFE) && Magic[2] == char(0xBA) &&
          Magic[3] == char(0xBE)) {
        // This is complicated by an overlap with Java class files.
        // See the Mach-O section in /usr/share/file/magic for details.
        if (Magic.size() >= 8 && Magic[7] < 43)
          return file_magic::macho_universal_binary;
      }
      break;

      // The two magic numbers for mach-o are:
      // 0xfeedface - 32-bit mach-o
      // 0xfeedfacf - 64-bit mach-o
    case 0xFE:
    case 0xCE:
    case 0xCF: {
      uint16_t type = 0;
      if (Magic[0] == char(0xFE) && Magic[1] == char(0xED) &&
          Magic[2] == char(0xFA) &&
          (Magic[3] == char(0xCE) || Magic[3] == char(0xCF))) {
        /* Native endian */
        if (Magic.size() >= 16) type = Magic[14] << 8 | Magic[15];
      } else if ((Magic[0] == char(0xCE) || Magic[0] == char(0xCF)) &&
                 Magic[1] == char(0xFA) && Magic[2] == char(0xED) &&
                 Magic[3] == char(0xFE)) {
        /* Reverse endian */
        if (Magic.size() >= 14) type = Magic[13] << 8 | Magic[12];
      }
      switch (type) {
        default: break;
        case 1: return file_magic::macho_object;
        case 2: return file_magic::macho_executable;
        case 3: return file_magic::macho_fixed_virtual_memory_shared_lib;
        case 4: return file_magic::macho_core;
        case 5: return file_magic::macho_preload_executable;
        case 6: return file_magic::macho_dynamically_linked_shared_lib;
        case 7: return file_magic::macho_dynamic_linker;
        case 8: return file_magic::macho_bundle;
        case 9: return file_magic::macho_dynamically_linked_shared_lib_stub;
        case 10: return file_magic::macho_dsym_companion;
        case 11: return file_magic::macho_kext_bundle;
      }
      break;
    }
    case 0xF0: // PowerPC Windows
    case 0x83: // Alpha 32-bit
    case 0x84: // Alpha 64-bit
    case 0x66: // MPS R4000 Windows
    case 0x50: // mc68K
    case 0x4c: // 80386 Windows
    case 0xc4: // ARMNT Windows
      if (Magic[1] == 0x01)
        return file_magic::coff_object;

    case 0x90: // PA-RISC Windows
    case 0x68: // mc68K Windows
      if (Magic[1] == 0x02)
        return file_magic::coff_object;
      break;

    case 'M': // Possible MS-DOS stub on Windows PE file
      if (Magic[1] == 'Z') {
        uint32_t off = read32le(Magic.data() + 0x3c);
        // PE/COFF file, either EXE or DLL.
        if (off < Magic.size() &&
            memcmp(Magic.data()+off, COFF::PEMagic, sizeof(COFF::PEMagic)) == 0)
          return file_magic::pecoff_executable;
      }
      break;

    case 0x64: // x86-64 Windows.
      if (Magic[1] == char(0x86))
        return file_magic::coff_object;
      break;

    default:
      break;
  }
  return file_magic::unknown;
}

std::error_code identify_magic(const Twine &Path, file_magic &Result) {
  int FD;
  if (std::error_code EC = openFileForRead(Path, FD))
    return EC;

  char Buffer[32];
  int Length = msf_read(FD, Buffer, sizeof(Buffer)); // HLSL Change
  if (msf_close(FD) != 0 || Length < 0) // HLSL Change
    return std::error_code(errno, std::generic_category());

  Result = identify_magic(StringRef(Buffer, Length));
  return std::error_code();
}

std::error_code directory_entry::status(file_status &result) const {
  return fs::status(Path, result);
}

} // end namespace fs
} // end namespace sys
} // end namespace llvm

// HLSL Change begin - Create implicit filesystem
#ifdef MS_IMPLICIT_DISK_FILESYSTEM

#include "dxc/Support/WinIncludes.h"
#include "llvm/Support/MSFileSystem.h"

struct ImplicitFilesystem {
  ImplicitFilesystem() {
    llvm::sys::fs::SetupPerThreadFileSystem();
    sys::fs::MSFileSystem *pFSPtr;
    CreateMSFileSystemForDisk(&pFSPtr);
    pFS.reset(pFSPtr);
    llvm::sys::fs::SetCurrentThreadFileSystem(pFS.get());
  }

  std::unique_ptr<sys::fs::MSFileSystem> pFS;
};

static ImplicitFilesystem &getImplicitFilesystem() {
  static ImplicitFilesystem ImpFS;
  return ImpFS;
}
#endif
// HLSL Change end - Create implicit filesystem

// Include the truly platform-specific parts.
#if defined(LLVM_ON_UNIX)
#include "Unix/Path.inc"
#endif
#if defined(LLVM_ON_WIN32)
#include "Windows/MSFileSystem.inc.cpp"
#endif

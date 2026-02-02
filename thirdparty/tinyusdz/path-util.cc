// SPDX-License-Identifier: Apache 2.0
// Copyright 2023-Present Light Transport Entertainment, Inc.

#include "path-util.hh"
#include "str-util.hh"
#include "prim-types.hh"
#include "common-macros.inc"

namespace tinyusdz {
namespace pathutil {

namespace {

// Remove sequential "../"
// Returns the number of "../" occurence to `num`
std::string RemoveRelativePrefix(const std::string &in_str, size_t &num) {
  constexpr size_t maxnum = 1024*1024;

  num = 0;
  std::string ret = in_str;
  while (!ret.empty() || (num < maxnum)) {
    if (startsWith(ret, "../")) {
      ret = removePrefix(ret, "../");
      num++;
    } else {
      break;
    }
  }

  return ret;
}

} // namespace

Path FromString(const std::string &_path_str) {

  std::string path_str = _path_str;

  if (path_str.empty()) {
    return Path();
  }

  if (path_str == ".") {
    // invalid
    return Path();
  }

  size_t loc = path_str.find_last_of(".");
  if (loc != std::string::npos) {
    if (loc == (path_str.size() - 1)) {
      // ends with "."
      return Path();
    } else if ((path_str.size() > 1) && (loc < (path_str.size() - 1))) {
      if (path_str[loc+1] == '/') {
        // guess relative prim path only.
        return Path(path_str, "");
      }
    }
  }

  if (loc == std::string::npos) {
    // prim_part only
    return Path(path_str, "");
  }

  std::string prim_part = path_str.substr(0, loc);
  std::string prop_part = path_str.substr(loc+1);

  return Path(prim_part, prop_part);
}

bool ResolveRelativePath(const Path &base_prim_path, const Path &relative_path, Path *abs_path, std::string *err) {

  if (!abs_path) {
    return false;
  }

  std::string relative_str = relative_path.prim_part();
  std::string base_str = base_prim_path.prim_part();

  // base_prim_path must be absolute.
  if (startsWith(base_str, "/")) {
    // ok
  } else {
    if (err) {
      (*err) += "Base Prim path is not absolute path.\n";
    }
    return false;
  }

  std::string abs_dir;

  if (startsWith(relative_str, "./")) {
    // pxrUSD doesn't allow "./", so do same in tinyusdz.
#if 1
    if (err) {
      (*err) += "Path starting with `./` is not allowed.\n";
    }
    return false;
#else
    std::string remainder = removePrefix(relative_str, "./");

    // "./../", "././", etc is not allowed at the moment.
    if (contains_str(remainder, ".")) {
      return false;
    }

    abs_dir = base_str + "/" + remainder;
#endif

  } else if (startsWith(relative_str, "../")) {
    // ok
    size_t ndepth{0};
    std::string remainder = RemoveRelativePrefix(relative_str, ndepth);
    DCOUT("remainder = " << remainder << ", ndepth = " << ndepth);

    // "../" in subsequent position(e.g. `../bora/../dora`) is not allowed at the moment.
    if (contains_str(remainder, ".")) {
      if (err) {
        (*err) += "`../` in the middle of Path is not allowed.\n";
      }
      return false;
    }

    std::vector<std::string> base_dirs = split(base_str, "/");
    DCOUT("base_dirs.len = " << base_dirs.size());
    //if (base_dirs.size() < ndepth) {
    //  return false;
    //}

    if (base_dirs.size() == 0) { // "/"
      abs_dir = "/" + remainder;
    } else {
      int64_t n = int64_t(base_dirs.size()) - int64_t(ndepth);

#if 1
      // pxrUSD behavior
      if (n < -1) {
        if (err) {
          (*err) += "The number of `../` exceeds Prim path depth.\n";
        }
        return false;
      }
#else
      // Unixish path behavior
#endif
      if (n <= 0) {
        abs_dir += "/" + remainder;
      } else {
        for (size_t i = 0; i < size_t(n); i++) {
          abs_dir += "/" + base_dirs[i];
        }
        abs_dir += "/" + remainder;
      }
    }
  } else if (startsWith(relative_str, ".")) {
    // Property path?
    if (err) {
      (*err) += "A path starting with `.` is not allowed for Prim path.\n";
    }
    return false;
  } else if (startsWith(relative_str, "/")) {
    // Input path is already absolute.
    abs_dir = relative_str;
  } else {
    // Guess relative path(e.g. "muda", "bora/dora")
    // TODO: Check Path contains valid characters.
    abs_dir = base_str + "/" + relative_str;
  }

  (*abs_path) = Path(abs_dir, relative_path.prop_part());

  return true;
}

bool ValidatePath(const Path &path, std::string *err) {
  return ValidatePrimPath(path, err) && ValidatePropPath(path, err);
}

bool ValidatePrimPath(const Path &path, std::string *err) {
  if (!path.is_valid()) {
    if (err) {
      (*err) = "Path is invalid.";
    }
    return false;
  }

  if (!path.is_prim_path()) {
    if (err) {
      (*err) = "Path is not Prim path.";
    }
    return false;
  }

  const std::vector<std::string> element_names = split(path.prim_part(), "/");

  for (size_t i = 0; i < element_names.size(); i++) {
    if (!ValidatePrimElementName(element_names[i])) {
      if (err) {
        (*err) = "Prim path is not composed of valid identifiers.";
      }
      
      return false;
    }
  }

  return true;
}

bool ValidatePropPath(const Path &path, std::string *err) {
  if (path.prop_part() == ":") {
    if (err) {
      (*err) = "Proparty path is composed of namespace delimiter only(`:`).";
    }
    return false;
  }

  if (startsWith(path.prop_part(), ":")) {
    if (err) {
      (*err) = "Property path starts with namespace delimiter(`:`).";
    }
    return false;
  }

  if (endsWith(path.prop_part(), ":")) {
    if (err) {
      (*err) = "Property path ends with namespace delimiter(`:`).";
    }
    return false;
  }

  if (contains_str(path.prop_part(), "::")) {
    if (err) {
      (*err) = "Empty path among namespace delimiters(`::`) in Property path.";
    }
    return false;
  }

  // TODO: more validation

  return true;

}

} // namespace pathutil
} // namespace tinyusdz

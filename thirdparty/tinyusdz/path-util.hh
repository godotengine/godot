// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - Present, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// Utility functions for Path

#include "prim-types.hh"

namespace tinyusdz {
namespace pathutil {

///
/// Validate path. `err` will be filled when Path is an invalid one.
///
bool ValidatePath(const Path &path, std::string *err);

///
/// Validate Prim path(`Path::prim_part()`). `err` will be filled when Prim path is invalid(e.g. contains invalid character: "/dora%bora|muda").
///
bool ValidatePrimPath(const Path &path, std::string *err);

///
/// Validate Prim property path(`Path::prop_part()`). `err` will be filled when Prim property path is invalid(e.g. contains invalid character: "/dora%bora|muda").
///
bool ValidatePropPath(const Path &path, std::string *err);

///
///
/// Construct Path from a string.
/// It splits string into prim_part and prop_part(e.g. "/bora.dora" => "/dora", "bora") if required and constrcut Path object.
///
/// Use Path::valid() to check if input `path_str` is a valid path string.
///
Path FromString(const std::string &path_str);

///
/// Concatinate two Paths.
///
Path ConcatPath(const Path &parent, const Path &child);

///
/// Replace '../' and produce absolute path
///
/// @param[in] base_prim_path Base prim path(absolute)
/// @param[in] relative_path Relative prim path.
/// @param[out] abs_path Resolved absolute path.
///
/// base_prim_path: /root/xform
///
/// ../bora => /root/bora
/// ../../bora => /bora
/// bora => /root/xform/bora
///
/// NG
///
/// - ../../../bora => nest size mismatch
/// - `../` in the middle of relative path(e.g. `/root/../bora`)
/// - `./` (e.g. `./bora`)
///
/// @return true upon success to resolve relative path.
/// @return false when `base_prim_path` is a relative path or invalid,
/// `relative_path` is an absolute path or invalid, or cannot resolve relative
/// path.
///
bool ResolveRelativePath(const Path &base_prim_path, const Path &relative_path,
                         Path *abs_path, std::string *err = nullptr);

///
/// Currently ToUnixishPath converts backslash character to forward slash
/// character.
///
/// /home/tinyusdz => C:/Users/tinyusdz
/// C:\\Users\\tinyusdz => C:/Users/tinyusdz
///
Path ToUnixishPath(const Path &path);

}  // namespace pathutil
}  // namespace tinyusdz

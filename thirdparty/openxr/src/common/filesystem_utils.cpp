// Copyright (c) 2017-2026 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Authors: Mark Young <marky@lunarg.com>
//                  Nat Brown <natb@valvesoftware.com>
//

#include "filesystem_utils.hpp"

#include <filesystem>

bool FileSysUtilsIsRegularFile(const std::string& path) { return std::filesystem::is_regular_file(path); }

bool FileSysUtilsPathExists(const std::string& path) { return std::filesystem::exists(path); }

bool FileSysUtilsIsAbsolutePath(const std::string& path) {
    std::filesystem::path file_path(path);
    return file_path.is_absolute();
}

bool FileSysUtilsGetParentPath(const std::string& file_path, std::string& parent_path) {
    std::filesystem::path path_var(file_path);
    parent_path = path_var.parent_path().string();
    return true;
}

bool FileSysUtilsGetAbsolutePath(const std::string& path, std::string& absolute) {
    absolute = std::filesystem::absolute(path).string();
    return true;
}

bool FileSysUtilsGetCanonicalPath(const std::string& path, std::string& canonical) {
#if defined(XR_USE_PLATFORM_WIN32)
    // std::filesystem::canonical fails on UWP and must be avoided. Further, PathCchCanonicalize is not available on Windows 7 and
    // PathCanonicalizeW is not available on UWP. However, symbolic links are not important on Windows since the loader uses the
    // registry for indirection instead, and so this function can be a no-op on Windows.
    canonical = path;
#else
    canonical = std::filesystem::canonical(path).string();
#endif
    return true;
}

bool FileSysUtilsCombinePaths(const std::string& parent, const std::string& child, std::string& combined) {
    std::filesystem::path parent_path(parent);
    std::filesystem::path child_path(child);
    std::filesystem::path full_path = parent_path / child_path;
    combined = full_path.string();
    return true;
}

bool FileSysUtilsFindFilesInPath(const std::string& path, std::vector<std::string>& files) {
    for (auto& dir_iter : std::filesystem::directory_iterator(path)) {
        files.push_back(dir_iter.path().filename().string());
    }
    return true;
}

// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#pragma once

#include <string>
#include <vector>

// Determine if the path indicates a regular file (not a directory or symbolic link)
bool FileSysUtilsIsRegularFile(const std::string& path);

// Determine if the path indicates a directory
bool FileSysUtilsIsDirectory(const std::string& path);

// Determine if the provided path exists on the filesystem
bool FileSysUtilsPathExists(const std::string& path);

// Get the current directory
bool FileSysUtilsGetCurrentPath(std::string& path);

// Get the parent path of a file
bool FileSysUtilsGetParentPath(const std::string& file_path, std::string& parent_path);

// Determine if the provided path is an absolute path
bool FileSysUtilsIsAbsolutePath(const std::string& path);

// Get the absolute path for a provided file
bool FileSysUtilsGetAbsolutePath(const std::string& path, std::string& absolute);

// Get the absolute path for a provided file
bool FileSysUtilsGetCanonicalPath(const std::string& path, std::string& canonical);

// Combine a parent and child directory
bool FileSysUtilsCombinePaths(const std::string& parent, const std::string& child, std::string& combined);

// Parse out individual paths in a path list
bool FileSysUtilsParsePathList(std::string& path_list, std::vector<std::string>& paths);

// Record all the filenames for files found in the provided path.
bool FileSysUtilsFindFilesInPath(const std::string& path, std::vector<std::string>& files);

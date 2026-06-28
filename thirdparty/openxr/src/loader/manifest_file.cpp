// Copyright (c) 2017-2025 The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Authors: Mark Young <marky@lunarg.com>, Dave Houlton <daveh@lunarg.com>
//

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif  // defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)

#include "manifest_file.hpp"

#ifdef OPENXR_HAVE_COMMON_CONFIG
#include "common_config.h"
#endif  // OPENXR_HAVE_COMMON_CONFIG

#include "filesystem_utils.hpp"
#include "loader_init_data.hpp"
#include "loader_platform.hpp"
#include "loader_properties.hpp"
#include "platform_utils.hpp"
#include "loader_logger.hpp"
#include "unique_asset.h"

#include <json/json.h>
#include <openxr/openxr.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef FALLBACK_CONFIG_DIRS
#define FALLBACK_CONFIG_DIRS "/etc/xdg"
#endif  // !FALLBACK_CONFIG_DIRS

#ifndef FALLBACK_DATA_DIRS
#define FALLBACK_DATA_DIRS "/usr/local/share:/usr/share"
#endif  // !FALLBACK_DATA_DIRS

#ifndef SYSCONFDIR
#define SYSCONFDIR "/etc"
#endif  // !SYSCONFDIR

#if defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
#include <android/asset_manager.h>
#endif

#ifdef XRLOADER_DISABLE_EXCEPTION_HANDLING
#if JSON_USE_EXCEPTIONS
#error \
    "Loader is configured to not catch exceptions, but jsoncpp was built with exception-throwing enabled, which could violate the C ABI. One of those two things needs to change."
#endif  // JSON_USE_EXCEPTIONS
#endif  // !XRLOADER_DISABLE_EXCEPTION_HANDLING

#include "runtime_interface.hpp"

// Utility functions for finding files in the appropriate paths

static inline bool StringEndsWith(const std::string &value, const std::string &ending) {
    if (ending.size() > value.size()) {
        return false;
    }
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

// If the file found is a manifest file name, add it to the out_files manifest list.
static bool AddIfJson(const std::string &full_file, std::vector<std::string> &manifest_files) {
    if (full_file.empty() || !StringEndsWith(full_file, ".json")) {
        return false;
    }
    manifest_files.push_back(full_file);
    return true;
}

// Check the current path for any manifest files.  If the provided search_path is a directory, look for
// all included JSON files in that directory.  Otherwise, just check the provided search_path which should
// be a single filename.
static void CheckAllFilesInThePath(const std::string &search_path, bool is_directory_list,
                                   std::vector<std::string> &manifest_files) {
    if (FileSysUtilsPathExists(search_path)) {
        std::string absolute_path;
        if (!is_directory_list) {
            // If the file exists, try to add it
            if (FileSysUtilsIsRegularFile(search_path)) {
                FileSysUtilsGetAbsolutePath(search_path, absolute_path);
                AddIfJson(absolute_path, manifest_files);
            }
        } else {
            std::vector<std::string> files;
            if (FileSysUtilsFindFilesInPath(search_path, files)) {
                for (std::string &cur_file : files) {
                    std::string relative_path;
                    FileSysUtilsCombinePaths(search_path, cur_file, relative_path);
                    if (!FileSysUtilsGetAbsolutePath(relative_path, absolute_path)) {
                        continue;
                    }
                    AddIfJson(absolute_path, manifest_files);
                }
            }
        }
    }
}

// Add all manifest files in the provided paths to the manifest_files list.  If search_path
// is made up of directory listings (versus direct manifest file names) search each path for
// any manifest files.
static void AddFilesInPath(const std::string &search_path, bool is_directory_list, std::vector<std::string> &manifest_files) {
    std::size_t last_found = 0;
    std::size_t found = search_path.find_first_of(PATH_SEPARATOR);
    std::string cur_search;

    // Handle any path listings in the string (separated by the appropriate path separator)
    while (found != std::string::npos) {
        // substr takes a start index and length.
        std::size_t length = found - last_found;
        cur_search = search_path.substr(last_found, length);

        CheckAllFilesInThePath(cur_search, is_directory_list, manifest_files);

        // This works around issue if multiple path separator follow each other directly.
        last_found = found;
        while (found == last_found) {
            last_found = found + 1;
            found = search_path.find_first_of(PATH_SEPARATOR, last_found);
        }
    }

    // If there's something remaining in the string, copy it over
    if (last_found < search_path.size()) {
        cur_search = search_path.substr(last_found);
        CheckAllFilesInThePath(cur_search, is_directory_list, manifest_files);
    }
}

// Copy all paths listed in the cur_path string into output_path and append the appropriate relative_path onto the end of each.
static void CopyIncludedPaths(bool is_directory_list, const std::string &cur_path, const std::string &relative_path,
                              std::string &output_path) {
    if (!cur_path.empty()) {
        std::size_t last_found = 0;
        std::size_t found = cur_path.find_first_of(PATH_SEPARATOR);

        // Handle any path listings in the string (separated by the appropriate path separator)
        while (found != std::string::npos) {
            std::size_t length = found - last_found;
            output_path += cur_path.substr(last_found, length);
            if (is_directory_list && (cur_path[found - 1] != '\\' && cur_path[found - 1] != '/')) {
                output_path += DIRECTORY_SYMBOL;
            }
            output_path += relative_path;
            output_path += PATH_SEPARATOR;

            last_found = found;
            found = cur_path.find_first_of(PATH_SEPARATOR, found + 1);
        }

        // If there's something remaining in the string, copy it over
        size_t last_char = cur_path.size() - 1;
        if (last_found != last_char) {
            output_path += cur_path.substr(last_found);
            if (is_directory_list && (cur_path[last_char] != '\\' && cur_path[last_char] != '/')) {
                output_path += DIRECTORY_SYMBOL;
            }
            output_path += relative_path;
            output_path += PATH_SEPARATOR;
        }
    }
}

// Look for data files in the provided paths, but first check the environment override to determine if we should use that instead.
static void ReadDataFilesInSearchPaths(const std::string &override_env_var, const std::string &relative_path, bool &override_active,
                                       std::vector<std::string> &manifest_files) {
    std::string override_path;
    std::string search_path;

    if (!override_env_var.empty()) {
        bool permit_override = true;
#ifndef XR_OS_WINDOWS
        if (geteuid() != getuid() || getegid() != getgid()) {
            // Don't allow setuid apps to use the env var
            permit_override = false;
        }
#endif
        if (permit_override) {
            override_path = LoaderProperty::GetSecure(override_env_var);
        }
    }

    if (!override_path.empty()) {
        CopyIncludedPaths(true, override_path, "", search_path);
        override_active = true;
    } else {
        override_active = false;
#if !defined(XR_OS_WINDOWS) && !defined(XR_OS_ANDROID)
        const char home_additional[] = ".local/share/";

        // Determine how much space is needed to generate the full search path
        // for the current manifest files.
        std::string xdg_conf_dirs = LoaderProperty::GetSecure("XDG_CONFIG_DIRS");
        std::string xdg_data_dirs = LoaderProperty::GetSecure("XDG_DATA_DIRS");
        std::string xdg_data_home = LoaderProperty::GetSecure("XDG_DATA_HOME");
        std::string home = LoaderProperty::GetSecure("HOME");

        if (xdg_conf_dirs.empty()) {
            CopyIncludedPaths(true, FALLBACK_CONFIG_DIRS, relative_path, search_path);
        } else {
            CopyIncludedPaths(true, xdg_conf_dirs, relative_path, search_path);
        }

        CopyIncludedPaths(true, SYSCONFDIR, relative_path, search_path);
#if defined(EXTRASYSCONFDIR)
        CopyIncludedPaths(true, EXTRASYSCONFDIR, relative_path, search_path);
#endif

        if (xdg_data_dirs.empty()) {
            CopyIncludedPaths(true, FALLBACK_DATA_DIRS, relative_path, search_path);
        } else {
            CopyIncludedPaths(true, xdg_data_dirs, relative_path, search_path);
        }

        if (!xdg_data_home.empty()) {
            CopyIncludedPaths(true, xdg_data_home, relative_path, search_path);
        } else if (!home.empty()) {
            std::string relative_home_path = home_additional;
            relative_home_path += relative_path;
            CopyIncludedPaths(true, home, relative_home_path, search_path);
        }
#elif defined(XR_OS_ANDROID)
        CopyIncludedPaths(true, "/product/etc", relative_path, search_path);
        CopyIncludedPaths(true, "/odm/etc", relative_path, search_path);
        CopyIncludedPaths(true, "/oem/etc", relative_path, search_path);
        CopyIncludedPaths(true, "/vendor/etc", relative_path, search_path);
        CopyIncludedPaths(true, "/system/etc", relative_path, search_path);
#else
        (void)relative_path;
#endif
    }

    // Now, parse the paths and add any manifest files found in them.
    AddFilesInPath(search_path, true, manifest_files);
}

#ifdef XR_OS_LINUX

// Get an XDG environment variable with a $HOME-relative default
static std::string GetXDGEnvHome(const char *name, const char *fallback_path) {
    std::string result = LoaderProperty::GetSecure(name);
    if (!result.empty()) {
        return result;
    }
    result = LoaderProperty::GetSecure("HOME");
    if (result.empty()) {
        return result;
    }
    result += "/";
    result += fallback_path;
    return result;
}

// Get an XDG environment variable with absolute defaults
static std::string GetXDGEnvAbsolute(const char *name, const char *fallback_paths) {
    std::string result = LoaderProperty::GetSecure(name);
    if (!result.empty()) {
        return result;
    }
    return fallback_paths;
}

/// @param rt_dir_prefix Directory prefix with a trailing slash
static bool FindEitherActiveRuntimeFilename(const char *prefix_desc, const std::string &rt_dir_prefix, uint16_t major_version,
                                            std::string &out) {
    {
        std::ostringstream oss;
        oss << "Looking for active_runtime." XR_ARCH_ABI ".json or active_runtime.json in ";
        oss << prefix_desc;
        oss << ": ";
        oss << rt_dir_prefix;

        LoaderLogger::LogInfoMessage("", oss.str());
    }
    {
        auto decorated_path = rt_dir_prefix + std::to_string(major_version) + "/active_runtime." XR_ARCH_ABI ".json";

        if (FileSysUtilsPathExists(decorated_path)) {
            out = decorated_path;
            return true;
        }
    }
    {
        auto undecorated_path = rt_dir_prefix + std::to_string(major_version) + "/active_runtime.json";

        if (FileSysUtilsPathExists(undecorated_path)) {
            out = undecorated_path;
            return true;
        }
    }
    return false;
}
// Return the first instance of relative_path occurring in an XDG config dir according to standard
// precedence order.
static bool FindXDGConfigFile(const char *relative_dir, uint16_t major_version, std::string &out) {
    const std::string message{"Looking for active_runtime." XR_ARCH_ABI ".json or active_runtime.json"};
    LoaderLogger::LogInfoMessage("", message);
    std::string dir_prefix = GetXDGEnvHome("XDG_CONFIG_HOME", ".config");
    if (!dir_prefix.empty()) {
        dir_prefix += "/";
        dir_prefix += relative_dir;
        if (FindEitherActiveRuntimeFilename("XDG_CONFIG_HOME", dir_prefix, major_version, out)) {
            return true;
        }
    }

    std::istringstream iss(GetXDGEnvAbsolute("XDG_CONFIG_DIRS", FALLBACK_CONFIG_DIRS));
    std::string path;
    while (std::getline(iss, path, PATH_SEPARATOR)) {
        if (path.empty()) {
            continue;
        }
        dir_prefix = std::move(path);
        dir_prefix += "/";
        dir_prefix += relative_dir;
        if (FindEitherActiveRuntimeFilename("an entry of XDG_CONFIG_DIRS", dir_prefix, major_version, out)) {
            return true;
        }
    }

    dir_prefix = SYSCONFDIR;
    dir_prefix += "/";
    dir_prefix += relative_dir;
    if (FindEitherActiveRuntimeFilename("compiled-in SYSCONFDIR", dir_prefix, major_version, out)) {
        return true;
    }

#if defined(EXTRASYSCONFDIR)
    dir_prefix = EXTRASYSCONFDIR;
    dir_prefix += "/";
    dir_prefix += relative_dir;
    if (FindEitherActiveRuntimeFilename("compiled-in EXTRASYSCONFDIR", dir_prefix, major_version, out)) {
        return true;
    }
#endif

    out.clear();
    return false;
}

#endif

#ifdef XR_OS_WINDOWS

// Look for runtime data files in the provided paths, but first check the environment override to determine
// if we should use that instead.
static void ReadRuntimeDataFilesInRegistry(const std::string &runtime_registry_location,
                                           const std::string &default_runtime_value_name,
                                           std::vector<std::string> &manifest_files) {
    HKEY hkey;
    DWORD access_flags;
    wchar_t value_w[1024];
    DWORD value_size_w = sizeof(value_w);  // byte size of the buffer.

    // Generate the full registry location for the registry information
    std::string full_registry_location = OPENXR_REGISTRY_LOCATION;
    full_registry_location += std::to_string(XR_VERSION_MAJOR(XR_CURRENT_API_VERSION));
    full_registry_location += runtime_registry_location;

    const std::wstring full_registry_location_w = utf8_to_wide(full_registry_location);
    const std::wstring default_runtime_value_name_w = utf8_to_wide(default_runtime_value_name);

    // Use 64 bit regkey for 64bit application, and use 32 bit regkey in WOW for 32bit application.
    access_flags = KEY_QUERY_VALUE;
    LONG open_value = RegOpenKeyExW(HKEY_LOCAL_MACHINE, full_registry_location_w.c_str(), 0, access_flags, &hkey);

    if (ERROR_SUCCESS != open_value) {
        LoaderLogger::LogWarningMessage("",
                                        "ReadRuntimeDataFilesInRegistry - failed to open registry key " + full_registry_location);
        return;
    }

    if (ERROR_SUCCESS != RegGetValueW(hkey, nullptr, default_runtime_value_name_w.c_str(),
                                      RRF_RT_REG_SZ | REG_EXPAND_SZ | RRF_ZEROONFAILURE, NULL, reinterpret_cast<LPBYTE>(&value_w),
                                      &value_size_w)) {
        LoaderLogger::LogWarningMessage(
            "", "ReadRuntimeDataFilesInRegistry - failed to read registry value " + default_runtime_value_name);
    } else {
        // Not using AddFilesInPath here (as only api_layer manifest paths allow multiple
        // separated paths)
        // Small time-of-check vs time-of-use issue here but it mainly only affects the error message.
        // It does not introduce a security defect.
        std::string activeRuntimePath = wide_to_utf8(value_w);
        if (FileSysUtilsIsRegularFile(activeRuntimePath)) {
            // If the file exists, try to add it
            std::string absolute_path;
            FileSysUtilsGetAbsolutePath(activeRuntimePath, absolute_path);
            if (!AddIfJson(absolute_path, manifest_files)) {
                LoaderLogger::LogErrorMessage(
                    "", "ReadRuntimeDataFilesInRegistry - registry runtime path is not json " + activeRuntimePath);
            }
        } else {
            LoaderLogger::LogErrorMessage(
                "", "ReadRuntimeDataFilesInRegistry - registry runtime path does not exist " + activeRuntimePath);
        }
    }

    RegCloseKey(hkey);
}

// Look for layer data files in the provided paths, but first check the environment override to determine
// if we should use that instead.
static void ReadLayerDataFilesInRegistry(const std::string &registry_location, std::vector<std::string> &manifest_files) {
    const std::wstring full_registry_location_w =
        utf8_to_wide(OPENXR_REGISTRY_LOCATION + std::to_string(XR_VERSION_MAJOR(XR_CURRENT_API_VERSION)) + registry_location);

    auto ReadLayerDataFilesInHive = [&](HKEY hive) {
        HKEY hkey;
        LONG open_value = RegOpenKeyExW(hive, full_registry_location_w.c_str(), 0, KEY_QUERY_VALUE, &hkey);
        if (ERROR_SUCCESS != open_value) {
            return false;
        }

        wchar_t name_w[1024]{};
        LONG rtn_value;
        DWORD name_size = 1023;
        DWORD value;
        DWORD value_size = sizeof(value);
        DWORD key_index = 0;
        while (ERROR_SUCCESS ==
               (rtn_value = RegEnumValueW(hkey, key_index++, name_w, &name_size, NULL, NULL, (LPBYTE)&value, &value_size))) {
            if (value_size == sizeof(value) && value == 0) {
                const std::string filename = wide_to_utf8(name_w);
                AddFilesInPath(filename, false, manifest_files);
            }
            // Reset some items for the next loop
            name_size = 1023;
        }

        RegCloseKey(hkey);

        return true;
    };

    // Do not allow high integrity processes to act on data that can be controlled by medium integrity processes.
    const bool readFromCurrentUser = !IsHighIntegrityLevel();

    bool found = ReadLayerDataFilesInHive(HKEY_LOCAL_MACHINE);
    if (readFromCurrentUser) {
        found |= ReadLayerDataFilesInHive(HKEY_CURRENT_USER);
    }

    if (!found) {
        std::string warning_message = "ReadLayerDataFilesInRegistry - failed to read registry location ";
        warning_message += registry_location;
        warning_message += (readFromCurrentUser ? " in either HKEY_LOCAL_MACHINE or HKEY_CURRENT_USER" : " in HKEY_LOCAL_MACHINE");
        LoaderLogger::LogWarningMessage("", warning_message);
    }
}

#endif  // XR_OS_WINDOWS

ManifestFile::ManifestFile(ManifestFileType type, const std::string &filename, const std::string &library_path)
    : _filename(filename), _type(type), _library_path(library_path) {}

bool ManifestFile::IsValidJson(const Json::Value &root_node, JsonVersion &version) {
    if (root_node["file_format_version"].isNull() || !root_node["file_format_version"].isString()) {
        LoaderLogger::LogErrorMessage("", "ManifestFile::IsValidJson - JSON file missing \"file_format_version\"");
        return false;
    }
    std::string file_format = root_node["file_format_version"].asString();
    const int num_fields = sscanf(file_format.c_str(), "%u.%u.%u", &version.major, &version.minor, &version.patch);

    // Only version 1.0.0 is defined currently.  Eventually we may have more version, but
    // some of the versions may only be valid for layers or runtimes specifically.
    if (num_fields != 3 || version.major != 1 || version.minor != 0 || version.patch != 0) {
        std::ostringstream error_ss;
        error_ss << "ManifestFile::IsValidJson - JSON \"file_format_version\" " << version.major << "." << version.minor << "."
                 << version.patch << " is not supported";
        LoaderLogger::LogErrorMessage("", error_ss.str());
        return false;
    }

    return true;
}

static void GetExtensionProperties(const std::vector<ExtensionListing> &extensions, std::vector<XrExtensionProperties> &props) {
    for (const auto &ext : extensions) {
        auto it =
            std::find_if(props.begin(), props.end(), [&](XrExtensionProperties &prop) { return prop.extensionName == ext.name; });
        if (it != props.end()) {
            it->extensionVersion = std::max(it->extensionVersion, ext.extension_version);
        } else {
            XrExtensionProperties prop{};
            prop.type = XR_TYPE_EXTENSION_PROPERTIES;
            strncpy(prop.extensionName, ext.name.c_str(), XR_MAX_EXTENSION_NAME_SIZE - 1);
            prop.extensionName[XR_MAX_EXTENSION_NAME_SIZE - 1] = '\0';
            prop.extensionVersion = ext.extension_version;
            props.push_back(prop);
        }
    }
}

// Return any instance extensions found in the manifest files in the proper form for
// OpenXR (XrExtensionProperties).
void ManifestFile::GetInstanceExtensionProperties(std::vector<XrExtensionProperties> &props) {
    GetExtensionProperties(_instance_extensions, props);
}

std::string ManifestFile::GetFunctionName(const std::string &func_name) const {
    if (!_functions_renamed.empty()) {
        auto found = _functions_renamed.find(func_name);
        if (found != _functions_renamed.end()) {
            return found->second;
        }
    }
    return func_name;
}

RuntimeManifestFile::RuntimeManifestFile(const std::string &filename, const std::string &library_path)
    : ManifestFile(MANIFEST_TYPE_RUNTIME, filename, library_path) {}

static void ParseExtension(Json::Value const &ext, std::vector<ExtensionListing> &extensions) {
    Json::Value ext_name = ext["name"];
    Json::Value ext_version = ext["extension_version"];

    // Allow "extension_version" as a String or a UInt to maintain backwards compatibility, even though it should be a String.
    // Internal Issue 1411: https://gitlab.khronos.org/openxr/openxr/-/issues/1411
    // Internal MR !1867: https://gitlab.khronos.org/openxr/openxr/-/merge_requests/1867
    if (ext_name.isString() && (ext_version.isString() || ext_version.isUInt())) {
        ExtensionListing ext_listing = {};
        ext_listing.name = ext_name.asString();
        if (ext_version.isUInt()) {
            ext_listing.extension_version = ext_version.asUInt();
        } else {
            ext_listing.extension_version = atoi(ext_version.asString().c_str());
        }
        extensions.push_back(ext_listing);
    }
}

void ManifestFile::ParseCommon(Json::Value const &root_node) {
    const Json::Value &inst_exts = root_node["instance_extensions"];
    if (!inst_exts.isNull() && inst_exts.isArray()) {
        for (const auto &ext : inst_exts) {
            ParseExtension(ext, _instance_extensions);
        }
    }
    const Json::Value &funcs_renamed = root_node["functions"];
    if (!funcs_renamed.isNull() && !funcs_renamed.empty()) {
        for (Json::ValueConstIterator func_it = funcs_renamed.begin(); func_it != funcs_renamed.end(); ++func_it) {
            if (!(*func_it).isString()) {
                LoaderLogger::LogWarningMessage(
                    "", "ManifestFile::ParseCommon " + _filename + " \"functions\" section contains non-string values.");
                continue;
            }
            std::string original_name = func_it.key().asString();
            std::string new_name = (*func_it).asString();
            _functions_renamed.emplace(original_name, new_name);
        }
    }
}

void RuntimeManifestFile::CreateIfValid(std::string const &filename,
                                        std::vector<std::unique_ptr<RuntimeManifestFile>> &manifest_files) {
    std::ifstream json_stream(filename, std::ifstream::in);

    LoaderLogger::LogInfoMessage("", "RuntimeManifestFile::CreateIfValid - attempting to load " + filename);
    std::ostringstream error_ss("RuntimeManifestFile::CreateIfValid ");
    if (!json_stream.is_open()) {
        error_ss << "failed to open " << filename << ".  Does it exist?";
        LoaderLogger::LogErrorMessage("", error_ss.str());
        return;
    }
    Json::CharReaderBuilder builder;
    std::string errors;
    Json::Value root_node = Json::nullValue;
    if (!Json::parseFromStream(builder, json_stream, &root_node, &errors) || !root_node.isObject()) {
        error_ss << "failed to parse " << filename << ".";
        if (!errors.empty()) {
            error_ss << " (Error message: " << errors << ")";
        }
        error_ss << " Is it a valid runtime manifest file?";
        LoaderLogger::LogErrorMessage("", error_ss.str());
        return;
    }

    CreateIfValid(root_node, filename, manifest_files);
}

void RuntimeManifestFile::CreateIfValid(const Json::Value &root_node, const std::string &filename,
                                        std::vector<std::unique_ptr<RuntimeManifestFile>> &manifest_files) {
    std::ostringstream error_ss("RuntimeManifestFile::CreateIfValid ");
    JsonVersion file_version = {};
    if (!ManifestFile::IsValidJson(root_node, file_version)) {
        error_ss << "isValidJson indicates " << filename << " is not a valid manifest file.";
        LoaderLogger::LogErrorMessage("", error_ss.str());
        return;
    }
    const Json::Value &runtime_root_node = root_node["runtime"];
    // The Runtime manifest file needs the "runtime" root as well as a sub-node for "library_path".  If any of those aren't there,
    // fail.
    if (runtime_root_node.isNull() || runtime_root_node["library_path"].isNull() || !runtime_root_node["library_path"].isString()) {
        error_ss << filename << " is missing required fields.  Verify all proper fields exist.";
        LoaderLogger::LogErrorMessage("", error_ss.str());
        return;
    }

    std::string lib_path = runtime_root_node["library_path"].asString();

    // If the library_path variable has no directory symbol, it's just a file name and should be accessible on the
    // global library path.
    if (lib_path.find('\\') != std::string::npos || lib_path.find('/') != std::string::npos) {
        // If the library_path is an absolute path, just use that as-is.
        if (!FileSysUtilsIsAbsolutePath(lib_path)) {
            // Otherwise, treat the library path as a relative path based on the JSON file.
            std::string canonical_path;
            std::string combined_path;
            std::string file_parent;
            // Search relative to the real manifest file, not relative to the symlink
            if (!FileSysUtilsGetCanonicalPath(filename, canonical_path)) {
                // Give relative to the non-canonical path a chance
                canonical_path = filename;
            }
            if (!FileSysUtilsGetParentPath(canonical_path, file_parent) ||
                !FileSysUtilsCombinePaths(file_parent, lib_path, combined_path)) {
                error_ss << filename << " filesystem operations failed for path  " << canonical_path;
                LoaderLogger::LogErrorMessage("", error_ss.str());
                return;
            }
            lib_path = combined_path;
        }
    }

    // Add this runtime manifest file
    manifest_files.emplace_back(new RuntimeManifestFile(filename, lib_path));

    // Add any extensions to it after the fact.
    // Handle any renamed functions
    manifest_files.back()->ParseCommon(runtime_root_node);
}

// Find all manifest files in the appropriate search paths/registries for the given type.
XrResult RuntimeManifestFile::FindManifestFiles(const std::string &openxr_command,
                                                std::vector<std::unique_ptr<RuntimeManifestFile>> &manifest_files) {
    XrResult result = XR_SUCCESS;
    std::string filename = LoaderProperty::GetSecure(OPENXR_RUNTIME_JSON_ENV_VAR);
    if (!filename.empty()) {
        LoaderLogger::LogInfoMessage(
            openxr_command,
            "RuntimeManifestFile::FindManifestFiles - using environment variable override runtime file " + filename);
    } else {
#ifdef XR_OS_WINDOWS
        std::vector<std::string> filenames;
        ReadRuntimeDataFilesInRegistry("", "ActiveRuntime", filenames);
        if (filenames.size() == 0) {
            LoaderLogger::LogErrorMessage(
                openxr_command, "RuntimeManifestFile::FindManifestFiles - failed to find active runtime file in registry");
            return XR_ERROR_RUNTIME_UNAVAILABLE;
        }
        if (filenames.size() > 1) {
            LoaderLogger::LogWarningMessage(
                openxr_command, "RuntimeManifestFile::FindManifestFiles - found too many default runtime files in registry");
        }
        filename = filenames[0];
        LoaderLogger::LogInfoMessage(openxr_command,
                                     "RuntimeManifestFile::FindManifestFiles - using registry-specified runtime file " + filename);
#elif defined(XR_OS_LINUX)

        if (!FindXDGConfigFile("openxr/", XR_VERSION_MAJOR(XR_CURRENT_API_VERSION), filename)) {
            LoaderLogger::LogErrorMessage(
                openxr_command,
                "RuntimeManifestFile::FindManifestFiles - failed to determine active runtime file path for this environment");
            return XR_ERROR_RUNTIME_UNAVAILABLE;
        }
#else  // !defined(XR_OS_WINDOWS) && !defined(XR_OS_LINUX)

#if defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
        Json::Value virtualManifest;
        result = GetPlatformRuntimeVirtualManifest(virtualManifest);
        if (XR_SUCCESS == result) {
            RuntimeManifestFile::CreateIfValid(virtualManifest, "", manifest_files);
            return result;
        }
#endif  // defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)

        if (!PlatformGetGlobalRuntimeFileName(XR_VERSION_MAJOR(XR_CURRENT_API_VERSION), filename)) {
            LoaderLogger::LogErrorMessage(
                openxr_command,
                "RuntimeManifestFile::FindManifestFiles - failed to determine active runtime file path for this environment");
            return XR_ERROR_RUNTIME_UNAVAILABLE;
        }
        result = XR_SUCCESS;
        LoaderLogger::LogInfoMessage(openxr_command,
                                     "RuntimeManifestFile::FindManifestFiles - using global runtime file " + filename);
#endif  // !defined(XR_OS_WINDOWS) && !defined(XR_OS_LINUX)
    }
    RuntimeManifestFile::CreateIfValid(filename, manifest_files);

    return result;
}

ApiLayerManifestFile::ApiLayerManifestFile(ManifestFileType type, const std::string &filename, const std::string &layer_name,
                                           const std::string &description, const JsonVersion &api_version,
                                           const uint32_t &implementation_version, const std::string &library_path)
    : ManifestFile(type, filename, library_path),
      _api_version(api_version),
      _layer_name(layer_name),
      _description(description),
      _implementation_version(implementation_version) {}

#if defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
void ApiLayerManifestFile::AddManifestFilesAndroid(const std::string &openxr_command, ManifestFileType type,
                                                   std::vector<std::unique_ptr<ApiLayerManifestFile>> &manifest_files) {
    if (!LoaderInitData::instance().initialized()) {
        // This will happen for applications that do not call xrInitializeLoaderKHR
        LoaderLogger::LogWarningMessage(
            openxr_command,
            "ApiLayerManifestFile::AddManifestFilesAndroid unable to add manifest files LoaderInitData not initialized.");
        return;
    }

    AAssetManager *assetManager = (AAssetManager *)GetAndroidAssetManager();
    std::vector<std::string> filenames;
    {
        std::string search_path = "";
        switch (type) {
            case MANIFEST_TYPE_IMPLICIT_API_LAYER:
                search_path = "openxr/1/api_layers/implicit.d/";
                break;
            case MANIFEST_TYPE_EXPLICIT_API_LAYER:
                search_path = "openxr/1/api_layers/explicit.d/";
                break;
            default:
                return;
        }

        UniqueAssetDir dir{AAssetManager_openDir(assetManager, search_path.c_str())};
        if (!dir) {
            return;
        }
        const std::string json = ".json";
        const char *fn = nullptr;
        while ((fn = AAssetDir_getNextFileName(dir.get())) != nullptr) {
            const std::string filename = search_path + fn;
            if (filename.size() < json.size()) {
                continue;
            }
            if (filename.compare(filename.size() - json.size(), json.size(), json) == 0) {
                filenames.push_back(filename);
            }
        }
    }
    for (const auto &filename : filenames) {
        UniqueAsset asset{AAssetManager_open(assetManager, filename.c_str(), AASSET_MODE_BUFFER)};
        if (!asset) {
            LoaderLogger::LogWarningMessage(
                openxr_command, "ApiLayerManifestFile::AddManifestFilesAndroid unable to open asset " + filename + ", skipping");

            continue;
        }
        size_t length = AAsset_getLength(asset.get());
        const char *buf = reinterpret_cast<const char *>(AAsset_getBuffer(asset.get()));
        if (!buf) {
            LoaderLogger::LogWarningMessage(
                openxr_command, "ApiLayerManifestFile::AddManifestFilesAndroid unable to access asset" + filename + ", skipping");

            continue;
        }
        std::istringstream json_stream(std::string{buf, length});

        CreateIfValid(type, filename, json_stream, &ApiLayerManifestFile::LocateLibraryInAssets, manifest_files);
    }
}
#endif  // defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)

void ApiLayerManifestFile::CreateIfValid(ManifestFileType type, const std::string &filename, std::istream &json_stream,
                                         LibraryLocator locate_library,
                                         std::vector<std::unique_ptr<ApiLayerManifestFile>> &manifest_files) {
    std::ostringstream error_ss("ApiLayerManifestFile::CreateIfValid ");
    Json::CharReaderBuilder builder;
    std::string errors;
    Json::Value root_node = Json::nullValue;
    if (!Json::parseFromStream(builder, json_stream, &root_node, &errors) || !root_node.isObject()) {
        error_ss << "failed to parse " << filename << ".";
        if (!errors.empty()) {
            error_ss << " (Error message: " << errors << ")";
        }
        error_ss << " Is it a valid layer manifest file?";
        LoaderLogger::LogErrorMessage("", error_ss.str());
        return;
    }
    JsonVersion file_version = {};
    if (!ManifestFile::IsValidJson(root_node, file_version)) {
        error_ss << "isValidJson indicates " << filename << " is not a valid manifest file.";
        LoaderLogger::LogErrorMessage("", error_ss.str());
        return;
    }

    Json::Value layer_root_node = root_node["api_layer"];

    // The API Layer manifest file needs the "api_layer" root as well as other sub-nodes.
    // If any of those aren't there, fail.
    if (layer_root_node.isNull() || layer_root_node["name"].isNull() || !layer_root_node["name"].isString() ||
        layer_root_node["api_version"].isNull() || !layer_root_node["api_version"].isString() ||
        layer_root_node["library_path"].isNull() || !layer_root_node["library_path"].isString() ||
        layer_root_node["implementation_version"].isNull() || !layer_root_node["implementation_version"].isString()) {
        error_ss << filename << " is missing required fields.  Verify all proper fields exist.";
        LoaderLogger::LogErrorMessage("", error_ss.str());
        return;
    }
    if (MANIFEST_TYPE_IMPLICIT_API_LAYER == type) {
        bool enabled = true;
        // Implicit layers require the disable environment variable.
        if (layer_root_node["disable_environment"].isNull() || !layer_root_node["disable_environment"].isString()) {
            error_ss << "Implicit layer " << filename << " is missing \"disable_environment\"";
            LoaderLogger::LogErrorMessage("", error_ss.str());
            return;
        }
        // Check if there's an enable environment variable provided
        if (!layer_root_node["enable_environment"].isNull() && layer_root_node["enable_environment"].isString()) {
            std::string env_var = layer_root_node["enable_environment"].asString();
            // If it's not set in the environment, disable the layer
            if (!LoaderProperty::IsSet(env_var)) {
                enabled = false;
            }
        }
        // Check for the disable environment variable, which must be provided in the JSON
        std::string env_var = layer_root_node["disable_environment"].asString();
        // If the env var is set, disable the layer. Disable env var overrides enable above
        if (LoaderProperty::IsSet(env_var)) {
            enabled = false;
        }

        // Not enabled, so pretend like it isn't even there.
        if (!enabled) {
            error_ss << "Implicit layer " << filename << " is disabled";
            LoaderLogger::LogInfoMessage("", error_ss.str());
            return;
        }
    }
    std::string layer_name = layer_root_node["name"].asString();
    std::string api_version_string = layer_root_node["api_version"].asString();
    JsonVersion api_version = {};
    const int num_fields = sscanf(api_version_string.c_str(), "%u.%u", &api_version.major, &api_version.minor);
    api_version.patch = 0;

    if ((num_fields != 2) || (api_version.major == 0 && api_version.minor == 0) ||
        api_version.major > XR_VERSION_MAJOR(XR_CURRENT_API_VERSION)) {
        error_ss << "layer " << filename << " has invalid API Version.  Skipping layer.";
        LoaderLogger::LogWarningMessage("", error_ss.str());
        return;
    }

    uint32_t implementation_version = atoi(layer_root_node["implementation_version"].asString().c_str());
    std::string library_path = layer_root_node["library_path"].asString();

    // If the library_path variable has no directory symbol, it's just a file name and should be accessible on the
    // global library path.
    if (library_path.find('\\') != std::string::npos || library_path.find('/') != std::string::npos) {
        // If the library_path is an absolute path, just use that if it exists
        if (FileSysUtilsIsAbsolutePath(library_path)) {
            if (!FileSysUtilsPathExists(library_path)) {
                error_ss << filename << " library " << library_path << " does not appear to exist";
                LoaderLogger::LogErrorMessage("", error_ss.str());
                return;
            }
        } else {
            // Otherwise, treat the library path as a relative path based on the JSON file.
            std::string combined_path;
            if (!locate_library(filename, library_path, combined_path)) {
                error_ss << filename << " library " << combined_path << " does not appear to exist";
                LoaderLogger::LogErrorMessage("", error_ss.str());
                return;
            }
            library_path = combined_path;
        }
    }

    std::string description;
    if (!layer_root_node["description"].isNull() && layer_root_node["description"].isString()) {
        description = layer_root_node["description"].asString();
    }

    // Add this layer manifest file
    manifest_files.emplace_back(
        new ApiLayerManifestFile(type, filename, layer_name, description, api_version, implementation_version, library_path));

    // Add any extensions to it after the fact.
    manifest_files.back()->ParseCommon(layer_root_node);
}

void ApiLayerManifestFile::CreateIfValid(ManifestFileType type, const std::string &filename,
                                         std::vector<std::unique_ptr<ApiLayerManifestFile>> &manifest_files) {
    std::ifstream json_stream(filename, std::ifstream::in);
    if (!json_stream.is_open()) {
        std::ostringstream error_ss("ApiLayerManifestFile::CreateIfValid ");
        error_ss << "failed to open " << filename << ".  Does it exist?";
        LoaderLogger::LogErrorMessage("", error_ss.str());
        return;
    }
    CreateIfValid(type, filename, json_stream, &ApiLayerManifestFile::LocateLibraryRelativeToJson, manifest_files);
}

bool ApiLayerManifestFile::LocateLibraryRelativeToJson(
    const std::string &json_filename, const std::string &library_path,
    std::string &out_combined_path) {  // Otherwise, treat the library path as a relative path based on the JSON file.
    std::string combined_path;
    std::string file_parent;
    if (!FileSysUtilsGetParentPath(json_filename, file_parent) ||
        !FileSysUtilsCombinePaths(file_parent, library_path, combined_path) || !FileSysUtilsPathExists(combined_path)) {
        out_combined_path = combined_path;
        return false;
    }
    out_combined_path = combined_path;
    return true;
}

#if defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
bool ApiLayerManifestFile::LocateLibraryInAssets(const std::string & /* json_filename */, const std::string &library_path,
                                                 std::string &out_combined_path) {
    std::string combined_path;
    std::string file_parent = GetAndroidNativeLibraryDir();
    if (!FileSysUtilsCombinePaths(file_parent, library_path, combined_path) || !FileSysUtilsPathExists(combined_path)) {
        out_combined_path = combined_path;
        return false;
    }
    out_combined_path = combined_path;
    return true;
}
#endif  // defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)

void ApiLayerManifestFile::PopulateApiLayerProperties(XrApiLayerProperties &props) const {
    props.layerVersion = _implementation_version;
    props.specVersion = XR_MAKE_VERSION(_api_version.major, _api_version.minor, _api_version.patch);
    strncpy(props.layerName, _layer_name.c_str(), XR_MAX_API_LAYER_NAME_SIZE - 1);
    if (_layer_name.size() >= XR_MAX_API_LAYER_NAME_SIZE - 1) {
        props.layerName[XR_MAX_API_LAYER_NAME_SIZE - 1] = '\0';
    }
    strncpy(props.description, _description.c_str(), XR_MAX_API_LAYER_DESCRIPTION_SIZE - 1);
    if (_description.size() >= XR_MAX_API_LAYER_DESCRIPTION_SIZE - 1) {
        props.description[XR_MAX_API_LAYER_DESCRIPTION_SIZE - 1] = '\0';
    }
}

// Find all layer manifest files in the appropriate search paths/registries for the given type.
XrResult ApiLayerManifestFile::FindManifestFiles(const std::string &openxr_command, ManifestFileType type,
                                                 std::vector<std::unique_ptr<ApiLayerManifestFile>> &manifest_files) {
    std::string relative_path;
    std::string override_env_var;
#ifdef XR_OS_WINDOWS
    std::string registry_location;
#endif

    // Add the appropriate top-level folders for the relative path.  These should be
    // the string "openxr/" followed by the API major version as a string.
    relative_path = OPENXR_RELATIVE_PATH;
    relative_path += std::to_string(XR_VERSION_MAJOR(XR_CURRENT_API_VERSION));

    switch (type) {
        case MANIFEST_TYPE_IMPLICIT_API_LAYER:
            relative_path += OPENXR_IMPLICIT_API_LAYER_RELATIVE_PATH;
            override_env_var = "";
#ifdef XR_OS_WINDOWS
            registry_location = OPENXR_IMPLICIT_API_LAYER_REGISTRY_LOCATION;
#endif
            break;
        case MANIFEST_TYPE_EXPLICIT_API_LAYER:
            relative_path += OPENXR_EXPLICIT_API_LAYER_RELATIVE_PATH;
            override_env_var = OPENXR_API_LAYER_PATH_ENV_VAR;
#ifdef XR_OS_WINDOWS
            registry_location = OPENXR_EXPLICIT_API_LAYER_REGISTRY_LOCATION;
#endif
            break;
        case MANIFEST_TYPE_UNDEFINED:
        case MANIFEST_TYPE_RUNTIME:
        default:
            LoaderLogger::LogErrorMessage(openxr_command,
                                          "ApiLayerManifestFile::FindManifestFiles - unknown manifest file requested");
            return XR_ERROR_FILE_ACCESS_ERROR;
    }

    bool override_active = false;
    std::vector<std::string> filenames;
    ReadDataFilesInSearchPaths(override_env_var, relative_path, override_active, filenames);

#ifdef XR_OS_WINDOWS
    // Read the registry if the override wasn't active.
    if (!override_active) {
        ReadLayerDataFilesInRegistry(registry_location, filenames);
    }
#endif

    for (std::string &cur_file : filenames) {
        ApiLayerManifestFile::CreateIfValid(type, cur_file, manifest_files);
    }

#if defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
    ApiLayerManifestFile::AddManifestFilesAndroid(openxr_command, type, manifest_files);
#endif  // defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)

    return XR_SUCCESS;
}

// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#pragma once

#include <openxr/openxr.h>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace Json {
class Value;
}

enum ManifestFileType {
    MANIFEST_TYPE_UNDEFINED = 0,
    MANIFEST_TYPE_RUNTIME,
    MANIFEST_TYPE_IMPLICIT_API_LAYER,
    MANIFEST_TYPE_EXPLICIT_API_LAYER,
};

struct JsonVersion {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

struct ExtensionListing {
    std::string name;
    uint32_t extension_version;
};

// ManifestFile class -
// Base class responsible for finding and parsing manifest files.
class ManifestFile {
   public:
    // Non-copyable
    ManifestFile(const ManifestFile &) = delete;
    ManifestFile &operator=(const ManifestFile &) = delete;

    ManifestFileType Type() const { return _type; }
    const std::string &Filename() const { return _filename; }
    const std::string &LibraryPath() const { return _library_path; }
    void GetInstanceExtensionProperties(std::vector<XrExtensionProperties> &props);
    const std::string &GetFunctionName(const std::string &func_name) const;

   protected:
    ManifestFile(ManifestFileType type, const std::string &filename, const std::string &library_path);
    void ParseCommon(Json::Value const &root_node);
    static bool IsValidJson(const Json::Value &root, JsonVersion &version);

   private:
    std::string _filename;
    ManifestFileType _type;
    std::string _library_path;
    std::vector<ExtensionListing> _instance_extensions;
    std::unordered_map<std::string, std::string> _functions_renamed;
};

// RuntimeManifestFile class -
// Responsible for finding and parsing Runtime-specific manifest files.
class RuntimeManifestFile : public ManifestFile {
   public:
    // Factory method
    static XrResult FindManifestFiles(std::vector<std::unique_ptr<RuntimeManifestFile>> &manifest_files);

   private:
    RuntimeManifestFile(const std::string &filename, const std::string &library_path);
    static void CreateIfValid(const std::string &filename, std::vector<std::unique_ptr<RuntimeManifestFile>> &manifest_files);
    static void CreateIfValid(const Json::Value &root_node, const std::string &filename,
                              std::vector<std::unique_ptr<RuntimeManifestFile>> &manifest_files);
};

// ApiLayerManifestFile class -
// Responsible for finding and parsing API Layer-specific manifest files.
class ApiLayerManifestFile : public ManifestFile {
   public:
    // Factory method
    static XrResult FindManifestFiles(ManifestFileType type, std::vector<std::unique_ptr<ApiLayerManifestFile>> &manifest_files);

    const std::string &LayerName() const { return _layer_name; }
    void PopulateApiLayerProperties(XrApiLayerProperties &props) const;

   private:
    ApiLayerManifestFile(ManifestFileType type, const std::string &filename, const std::string &layer_name,
                         const std::string &description, const JsonVersion &api_version, const uint32_t &implementation_version,
                         const std::string &library_path);
    static void CreateIfValid(ManifestFileType type, const std::string &filename,
                              std::vector<std::unique_ptr<ApiLayerManifestFile>> &manifest_files);

    JsonVersion _api_version;
    std::string _layer_name;
    std::string _description;
    uint32_t _implementation_version;
};

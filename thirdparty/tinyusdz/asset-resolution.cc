// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - Present, Light Transport Entertainment, Inc.
#include <cassert>
#include <iostream>

#include "asset-resolution.hh"
#include "common-macros.inc"
#include "io-util.hh"
#include "value-pprint.hh"
#include "str-util.hh"

namespace tinyusdz {

std::string AssetResolutionResolver::search_paths_str() const {
  std::string str;

  str += "[ ";
  for (size_t i = 0; i < _search_paths.size(); i++) {
    if (i > 0) {
      str += ", ";
    }
    // TODO: Escape character?
    str += _search_paths[i];
  }
  str += " ]";
  return str;
}

bool AssetResolutionResolver::find(const std::string &assetPath) const {
  DCOUT("search_paths = " << _search_paths);
  DCOUT("assetPath = " << assetPath);

  std::string ext = io::GetFileExtension(assetPath);

  if (_asset_resolution_handlers.count(ext)) {
    if (_asset_resolution_handlers.at(ext).resolve_fun && _asset_resolution_handlers.at(ext).size_fun) {
      std::string resolvedPath;
      std::string err;

      // Use custom handler's userdata
      void *userdata = _asset_resolution_handlers.at(ext).userdata;

      int ret = _asset_resolution_handlers.at(ext).resolve_fun(assetPath.c_str(), _search_paths, &resolvedPath, &err, userdata);
      if (ret != 0) {
        return false;
      }

      uint64_t sz{0};
      ret = _asset_resolution_handlers.at(ext).size_fun(resolvedPath.c_str(), &sz, &err, userdata);
      if (ret != 0) {
        return false;
      }

      return sz > 0;

    } else {
      DCOUT("Either Resolve function or Size function is nullptr. Fallback to wildcard handler or built-in file handler.");
    }
  }

  // wildcard
  if (_asset_resolution_handlers.count("*")) {
    if (_asset_resolution_handlers.at("*").resolve_fun && _asset_resolution_handlers.at("*").size_fun) {
      std::string resolvedPath;
      std::string err;

      // Use custom handler's userdata
      void *userdata = _asset_resolution_handlers.at("*").userdata;

      int ret = _asset_resolution_handlers.at("*").resolve_fun(assetPath.c_str(), _search_paths, &resolvedPath, &err, userdata);
      if (ret != 0) {
        return false;
      }

      uint64_t sz{0};
      ret = _asset_resolution_handlers.at("*").size_fun(resolvedPath.c_str(), &sz, &err, userdata);
      if (ret != 0) {
        return false;
      }

      return sz > 0;

    }

    return false;
  }  

  // default fallback: File-based 
  if ((_current_working_path == ".") || (_current_working_path == "./")) {
    std::string rpath = io::FindFile(assetPath, {});
  } else {
    // TODO: Only find when input path is relative.
    std::string rpath = io::FindFile(assetPath, {_current_working_path});
    if (rpath.size()) {
      return true;
    }
  }

  // TODO: Cache resolition.
  std::string fpath = io::FindFile(assetPath, _search_paths);
  return fpath.size();

}

std::string AssetResolutionResolver::resolve(
    const std::string &assetPath) const {

  std::string ext = io::GetFileExtension(assetPath);

  if (_asset_resolution_handlers.count(ext)) {
    if (_asset_resolution_handlers.at(ext).resolve_fun) {
      std::string resolvedPath;
      std::string err;

      // Use custom handler's userdata
      void *userdata = _asset_resolution_handlers.at(ext).userdata;

      int ret = _asset_resolution_handlers.at(ext).resolve_fun(assetPath.c_str(), _search_paths, &resolvedPath, &err, userdata);
      if (ret != 0) {
        return std::string();
      }

      return resolvedPath;

    } else {
      DCOUT("Resolve function is nullptr. Fallback to wildcard handler or built-in file handler.");
    }
  }

  if (_asset_resolution_handlers.count("*")) {
    if (_asset_resolution_handlers.at("*").resolve_fun) {
      std::string resolvedPath;
      std::string err;

      // Use custom handler's userdata
      void *userdata = _asset_resolution_handlers.at("*").userdata;

      int ret = _asset_resolution_handlers.at("*").resolve_fun(assetPath.c_str(), _search_paths, &resolvedPath, &err, userdata);
      if (ret != 0) {
        return std::string();
      }

      return resolvedPath;

    }

    return std::string();
  }

  DCOUT("cwd = " << _current_working_path);
  DCOUT("search_paths = " << _search_paths);
  DCOUT("assetPath = " << assetPath);

  std::string rpath;
  if ((_current_working_path == ".") || (_current_working_path == "./")) {
    rpath = io::FindFile(assetPath, {});
  } else {
    rpath = io::FindFile(assetPath, {_current_working_path});
  }

  if (rpath.size()) {
    return rpath;
  }

  // TODO: Cache resolition.
  return io::FindFile(assetPath, _search_paths);
}

bool AssetResolutionResolver::open_asset(const std::string &resolvedPath, const std::string &assetPath,
                  Asset *asset_out, std::string *warn, std::string *err) const {

  if (!asset_out) {
    if (err) {
      (*err) = "`asset` arg is nullptr.";
    }
    return false;
  }

  DCOUT("Opening asset: " << resolvedPath);

  (void)assetPath;
  (void)warn;

  std::string ext = io::GetFileExtension(resolvedPath);

  if (_asset_resolution_handlers.count(ext)) {
    if (_asset_resolution_handlers.at(ext).size_fun && _asset_resolution_handlers.at(ext).read_fun) {

      // Use custom handler's userdata
      void *userdata = _asset_resolution_handlers.at(ext).userdata;

      // Get asset size.
      uint64_t sz{0};
      int ret = _asset_resolution_handlers.at(ext).size_fun(resolvedPath.c_str(), &sz, err, userdata);
      if (ret != 0) {
        if (err) {
          (*err) += "Get size of asset through handler failed.\n";
        }
        return false;
      }
    
      DCOUT("asset_size: " << sz);

      tinyusdz::Asset asset;
      asset.resize(size_t(sz));

      uint64_t read_size{0};

      ret = _asset_resolution_handlers.at(ext).read_fun(resolvedPath.c_str(), /* req_size */asset.size(), asset.data(), &read_size, err, userdata);

      if (ret != 0) {
        if (err) {
          (*err) += "Read asset through handler failed.\n";
        }
        return false;
      }

      if (read_size < sz) {
        asset.resize(size_t(read_size));
        // May optimize memory usage
        asset.shrink_to_fit();
      }

      (*asset_out) = std::move(asset);

      return true;
    } else {
      DCOUT("Resolve function is nullptr. Fallback to built-in file handler.");
    }
  }

  if (_asset_resolution_handlers.count("*")) {
    if (_asset_resolution_handlers.at("*").size_fun && _asset_resolution_handlers.at("*").read_fun) {

      // Use custom handler's userdata
      void *userdata = _asset_resolution_handlers.at("*").userdata;

      // Get asset size.
      uint64_t sz{0};
      int ret = _asset_resolution_handlers.at("*").size_fun(resolvedPath.c_str(), &sz, err, userdata);
      if (ret != 0) {
        if (err) {
          (*err) += "Get size of asset through handler failed.\n";
        }
        return false;
      }
    
      DCOUT("asset_size: " << sz);

      tinyusdz::Asset asset;
      asset.resize(size_t(sz));

      uint64_t read_size{0};

      ret = _asset_resolution_handlers.at("*").read_fun(resolvedPath.c_str(), /* req_size */asset.size(), asset.data(), &read_size, err, userdata);

      if (ret != 0) {
        if (err) {
          (*err) += "Read asset through handler failed.\n";
        }
        return false;
      }

      if (read_size < sz) {
        asset.resize(size_t(read_size));
        // May optimize memory usage
        asset.shrink_to_fit();
      }

      (*asset_out) = std::move(asset);

      return true;
    }

    return false;
  }

  // Default: read from a file.
  std::vector<uint8_t> data;
  size_t max_bytes = 1024 * 1024 * _max_asset_bytes_in_mb;
  if (!io::ReadWholeFile(&data, err, resolvedPath, max_bytes,
                           /* userdata */ nullptr)) {

    if (err) {
      (*err) += "Open asset from a file failed.\n";
    }

    return false;
  }

  asset_out->set_data(std::move(data));

  return true;
}

}  // namespace tinyusdz

// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_CORE_OPTIONS_H_
#define DRACO_CORE_OPTIONS_H_

#include <cstdlib>
#include <map>
#include <string>

#include "draco/draco_features.h"

namespace draco {

// Class for storing generic options as a <name, value> pair in a string map.
// The API provides helper methods for directly storing values of various types
// such as ints and bools. One named option should be set with only a single
// data type.
class Options {
 public:
  Options() = default;
  ~Options() = default;

  // Merges |other_options| on top of the existing options of this instance
  // replacing all entries that are present in both options instances.
  void MergeAndReplace(const Options &other_options);

  void SetInt(const std::string &name, int val);
  void SetFloat(const std::string &name, float val);
  void SetBool(const std::string &name, bool val);
  void SetString(const std::string &name, const std::string &val);
  template <class VectorT>
  void SetVector(const std::string &name, const VectorT &vec) {
    SetVector(name, &vec[0], VectorT::dimension);
  }
  template <typename DataTypeT>
  void SetVector(const std::string &name, const DataTypeT *vec, int num_dims);

  // Getters will return a default value if the entry is not found. The default
  // value can be specified in the overloaded version of each function.
  int GetInt(const std::string &name) const;
  int GetInt(const std::string &name, int default_val) const;
  float GetFloat(const std::string &name) const;
  float GetFloat(const std::string &name, float default_val) const;
  bool GetBool(const std::string &name) const;
  bool GetBool(const std::string &name, bool default_val) const;
  std::string GetString(const std::string &name) const;
  std::string GetString(const std::string &name,
                        const std::string &default_val) const;
  template <class VectorT>
  VectorT GetVector(const std::string &name, const VectorT &default_val) const;
  // Unlike other Get functions, this function returns false if the option does
  // not exist, otherwise it fills |out_val| with the vector values. If a
  // default value is needed, it can be set in |out_val|.
  template <typename DataTypeT>
  bool GetVector(const std::string &name, int num_dims,
                 DataTypeT *out_val) const;

  bool IsOptionSet(const std::string &name) const {
    return options_.count(name) > 0;
  }

 private:
  // All entries are internally stored as strings and converted to the desired
  // return type based on the used Get* method.
  std::map<std::string, std::string> options_;
};

template <typename DataTypeT>
void Options::SetVector(const std::string &name, const DataTypeT *vec,
                        int num_dims) {
  std::string out;
  for (int i = 0; i < num_dims; ++i) {
    if (i > 0) {
      out += " ";
    }

// GNU STL on android doesn't include a proper std::to_string, but the libc++
// version does
#if defined(ANDROID) && !defined(_LIBCPP_VERSION)
    out += to_string(vec[i]);
#else
    out += std::to_string(vec[i]);
#endif
  }
  options_[name] = out;
}

template <class VectorT>
VectorT Options::GetVector(const std::string &name,
                           const VectorT &default_val) const {
  VectorT ret = default_val;
  GetVector(name, VectorT::dimension, &ret[0]);
  return ret;
}

template <typename DataTypeT>
bool Options::GetVector(const std::string &name, int num_dims,
                        DataTypeT *out_val) const {
  const auto it = options_.find(name);
  if (it == options_.end()) {
    return false;
  }
  const std::string value = it->second;
  if (value.length() == 0) {
    return true;  // Option set but no data is present
  }
  const char *act_str = value.c_str();
  char *next_str;
  for (int i = 0; i < num_dims; ++i) {
    if (std::is_integral<DataTypeT>::value) {
#ifdef ANDROID
      const int val = strtol(act_str, &next_str, 10);
#else
      const int val = static_cast<int>(std::strtol(act_str, &next_str, 10));
#endif
      if (act_str == next_str) {
        return true;  // End reached.
      }
      act_str = next_str;
      out_val[i] = static_cast<DataTypeT>(val);
    } else {
#ifdef ANDROID
      const float val = strtof(act_str, &next_str);
#else
      const float val = std::strtof(act_str, &next_str);
#endif
      if (act_str == next_str) {
        return true;  // End reached.
      }
      act_str = next_str;
      out_val[i] = static_cast<DataTypeT>(val);
    }
  }
  return true;
}

}  // namespace draco

#endif  // DRACO_CORE_OPTIONS_H_

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
#include "draco/core/options.h"

#include <cstdlib>
#include <set>
#include <string>
#include <utility>

namespace draco {

void Options::MergeAndReplace(const Options &other_options) {
  for (const auto &item : other_options.options_) {
    options_[item.first] = item.second;
  }
}

void Options::SetInt(const std::string &name, int val) {
  options_[name] = std::to_string(val);
}

void Options::SetFloat(const std::string &name, float val) {
  options_[name] = std::to_string(val);
}

void Options::SetBool(const std::string &name, bool val) {
  options_[name] = std::to_string(val ? 1 : 0);
}

void Options::SetString(const std::string &name, const std::string &val) {
  options_[name] = val;
}

int Options::GetInt(const std::string &name) const { return GetInt(name, -1); }

int Options::GetInt(const std::string &name, int default_val) const {
  const auto it = options_.find(name);
  if (it == options_.end()) {
    return default_val;
  }
  return std::atoi(it->second.c_str());
}

float Options::GetFloat(const std::string &name) const {
  return GetFloat(name, -1);
}

float Options::GetFloat(const std::string &name, float default_val) const {
  const auto it = options_.find(name);
  if (it == options_.end()) {
    return default_val;
  }
  return static_cast<float>(std::atof(it->second.c_str()));
}

bool Options::GetBool(const std::string &name) const {
  return GetBool(name, false);
}

bool Options::GetBool(const std::string &name, bool default_val) const {
  const int ret = GetInt(name, -1);
  if (ret == -1) {
    return default_val;
  }
  return static_cast<bool>(ret);
}

std::string Options::GetString(const std::string &name) const {
  return GetString(name, "");
}

std::string Options::GetString(const std::string &name,
                               const std::string &default_val) const {
  const auto it = options_.find(name);
  if (it == options_.end()) {
    return default_val;
  }
  return it->second;
}

}  // namespace draco

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLVersion.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#define METALCPP_VERSION_MAJOR 370
#define METALCPP_VERSION_MINOR 63
#define METALCPP_VERSION_PATCH 1

#define METALCPP_SUPPORTS_VERSION(major, minor, patch) \
    ((major < METALCPP_VERSION_MAJOR) || \
    (major == METALCPP_VERSION_MAJOR && minor < METALCPP_VERSION_MINOR) || \
    (major == METALCPP_VERSION_MAJOR && minor == METALCPP_VERSION_MINOR && patch <= METALCPP_VERSION_PATCH))

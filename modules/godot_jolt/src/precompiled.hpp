#pragma once

#include "common.h"



#ifdef JPH_DEBUG_RENDERER

#include "jolt/Renderer/DebugRenderer.h>

#endif // JPH_DEBUG_RENDERER

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

using namespace godot;

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include "containers/free_list.hpp"
#include "containers/hash_map.hpp"
#include "containers/hash_set.hpp"
#include "containers/inline_vector.hpp"
#include "containers/local_vector.hpp"
#include "containers/rid_owner.hpp"
#include "misc/bind_macros.hpp"
#include "misc/error_macros.hpp"
#include "misc/gdclass_macros.hpp"
#include "misc/jolt_stream_wrappers.hpp"
#include "misc/math.hpp"
#include "misc/scope_guard.hpp"
#include "misc/type_conversions.hpp"
#include "misc/utility_functions.hpp"
#include "misc/utility_macros.hpp"

// NOLINTEND(readability-duplicate-include)

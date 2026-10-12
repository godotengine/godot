/**************************************************************************/
/*  as_typed_containers.h                                                 */
/**************************************************************************/
#pragma once

#include "core/error/error_list.h"

#include <cstdint>

class asIScriptEngine;

// 固定强类型容器：热循环直接操作 Vector/HashMap，避免 Variant 装箱。
// 类型是 AngelScript 引用类型；脚本持有容器，不拥有 ObjectArray 中的 Godot 对象。
class ASTypedContainers {
public:
	// 幂等。必须在 Godot 值类型和 Object 骨架完成后调用。
	static Error register_all(asIScriptEngine *p_engine);
	static int64_t live_instance_count();
};

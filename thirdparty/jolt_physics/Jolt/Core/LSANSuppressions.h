// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#if defined(JPH_USE_VK) && defined(JPH_ASAN_ENABLED)

JPH_SUPPRESS_WARNING_PUSH
JPH_CLANG_SUPPRESS_WARNING("-Wreserved-identifier")

// Suppress ASAN leak detection for the Vulkan driver
extern "C" const char *__lsan_default_suppressions();
extern "C" const char *__lsan_default_suppressions()
{
	return "leak:libvulkan";
}

JPH_SUPPRESS_WARNING_POP

#endif

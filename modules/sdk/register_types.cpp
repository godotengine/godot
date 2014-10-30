/*************************************************************************/
/*  register_types.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifdef MODULE_SDK_ENABLED

#include "object_type_db.h"
#include "core/globals.h"
#include "register_types.h"

#include "platform.h"
#ifdef PLATFORM_IOS_91_ENABLED
#include "ios_91/platform_91.h"
#endif

void register_sdk_types() {

	Platform *platform = NULL;
#if defined(PLATFORM_IOS_91_ENABLED)
	platform = memnew(Platform91);
#elif defined(PLATFROM_IOS_360_ENABLED)
#else
	platform = memnew(Platform);
#endif

	Globals::get_singleton()->add_singleton(Globals::Singleton("Platform", platform));
}

void unregister_sdk_types() {

}

#else

void register_sdk_types() {}
void unregister_sdk_types() {}

#endif // MODULE_SDK_ENABLED

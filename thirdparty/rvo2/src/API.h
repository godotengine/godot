/*
 * API.h
 * RVO2-3D Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

/**
 * \file    API.h
 * \brief   Contains definitions related to Microsoft Windows.
 */

#ifndef RVO_API_H_
#define RVO_API_H_

#ifdef _WIN32
#include <SDKDDKVer.h>
#define WIN32_LEAN_AND_MEAN
#define NOCOMM
#define NOIMAGE
#define NOIME
#define NOKANJI
#define NOMCX
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define NOPROXYSTUB
#define NOSERVICE
#define NOSOUND
#define NOTAPE
#define NORPC
#define _USE_MATH_DEFINES
#include <windows.h>
#undef CONNECT_DEFERRED // Avoid collision with the Godot Object class
#undef CreateDialog // Avoid collision with the Godot CreateDialog class
#endif

#ifdef RVO_EXPORTS
#define RVO_API __declspec(dllexport)
#elif defined(RVO_IMPORTS)
#define RVO_API __declspec(dllimport)
#else
#define RVO_API
#endif

#endif /* RVO_API_H_ */

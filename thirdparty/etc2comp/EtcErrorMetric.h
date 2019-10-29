/*
 * Copyright 2015 The Etc2Comp Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace Etc
{

	enum ErrorMetric
	{
		RGBA,
		RGBX,
		REC709,
		NUMERIC,
		NORMALXYZ,
		//
		ERROR_METRICS,
		//
		BT709 = REC709
	};

	inline const char *ErrorMetricToString(ErrorMetric errorMetric)
	{
		switch (errorMetric)
		{
		case RGBA:
			return "RGBA";
		case RGBX:
			return "RGBX";
		case REC709:
			return "REC709";
		case NUMERIC:
			return "NUMERIC";
		case NORMALXYZ:
			return "NORMALXYZ";
		case ERROR_METRICS:
		default:
			return "UNKNOWN";
		}
	}
} // namespace Etc

// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/Core/Profiler.h>
#include <Jolt/Core/Color.h>
#include <Jolt/Core/StringTools.h>
#include <Jolt/Core/QuickSort.h>

JPH_SUPPRESS_WARNINGS_STD_BEGIN
#include <fstream>
JPH_SUPPRESS_WARNINGS_STD_END

JPH_NAMESPACE_BEGIN

#if defined(JPH_EXTERNAL_PROFILE) && defined(JPH_SHARED_LIBRARY)

ProfileStartMeasurementFunction ProfileStartMeasurement = [](const char *, uint32, uint8 *) { };
ProfileEndMeasurementFunction ProfileEndMeasurement = [](uint8 *) { };

#elif defined(JPH_PROFILE_ENABLED)

//////////////////////////////////////////////////////////////////////////////////////////
// Profiler
//////////////////////////////////////////////////////////////////////////////////////////

Profiler *Profiler::sInstance = nullptr;

#ifdef JPH_SHARED_LIBRARY
	static thread_local ProfileThread *sInstance = nullptr;

	ProfileThread *ProfileThread::sGetInstance()
	{
		return sInstance;
	}

	void ProfileThread::sSetInstance(ProfileThread *inInstance)
	{
		sInstance = inInstance;
	}
#else
	thread_local ProfileThread *ProfileThread::sInstance = nullptr;
#endif

bool ProfileMeasurement::sOutOfSamplesReported = false;

void Profiler::UpdateReferenceTime()
{
	mReferenceTick = GetProcessorTickCount();
	mReferenceTime = std::chrono::high_resolution_clock::now();
}

uint64 Profiler::GetProcessorTicksPerSecond() const
{
	uint64 ticks = GetProcessorTickCount();
	std::chrono::high_resolution_clock::time_point time = std::chrono::high_resolution_clock::now();

	return (ticks - mReferenceTick) * 1000000000ULL / std::chrono::duration_cast<std::chrono::nanoseconds>(time - mReferenceTime).count();
}

// This function assumes that none of the threads are active while we're dumping the profile,
// otherwise there will be a race condition on mCurrentSample and the profile data.
JPH_TSAN_NO_SANITIZE
void Profiler::NextFrame()
{
	std::lock_guard lock(mLock);

	if (mDump)
	{
		DumpInternal();
		mDump = false;
	}

	for (ProfileThread *t : mThreads)
		t->mCurrentSample = 0;

	UpdateReferenceTime();
}

void Profiler::Dump(const string_view &inTag)
{
	mDump = true;
	mDumpTag = inTag;
}

void Profiler::AddThread(ProfileThread *inThread)
{
	std::lock_guard lock(mLock);

	mThreads.push_back(inThread);
}

void Profiler::RemoveThread(ProfileThread *inThread)
{
	std::lock_guard lock(mLock);

	Array<ProfileThread *>::iterator i = std::find(mThreads.begin(), mThreads.end(), inThread);
	JPH_ASSERT(i != mThreads.end());
	mThreads.erase(i);
}

void Profiler::sAggregate(int inDepth, uint32 inColor, ProfileSample *&ioSample, const ProfileSample *inEnd, Aggregators &ioAggregators, KeyToAggregator &ioKeyToAggregator)
{
	// Store depth
	ioSample->mDepth = uint8(min(255, inDepth));

	// Update color
	if (ioSample->mColor == 0)
		ioSample->mColor = inColor;
	else
		inColor = ioSample->mColor;

	// Start accumulating totals
	uint64 cycles_this_with_children = ioSample->mEndCycle - ioSample->mStartCycle;

	// Loop over following samples until we find a sample that starts on or after our end
	ProfileSample *sample;
	for (sample = ioSample + 1; sample < inEnd && sample->mStartCycle < ioSample->mEndCycle; ++sample)
	{
		JPH_ASSERT(sample[-1].mStartCycle <= sample->mStartCycle);
		JPH_ASSERT(sample->mStartCycle >= ioSample->mStartCycle);
		JPH_ASSERT(sample->mEndCycle <= ioSample->mEndCycle);

		// Recurse and skip over the children of this child
		sAggregate(inDepth + 1, inColor, sample, inEnd, ioAggregators, ioKeyToAggregator);
	}

	// Find the aggregator for this name / filename pair
	Aggregator *aggregator;
	KeyToAggregator::iterator aggregator_idx = ioKeyToAggregator.find(ioSample->mName);
	if (aggregator_idx == ioKeyToAggregator.end())
	{
		// Not found, add to map and insert in array
		ioKeyToAggregator.try_emplace(ioSample->mName, ioAggregators.size());
		ioAggregators.emplace_back(ioSample->mName);
		aggregator = &ioAggregators.back();
	}
	else
	{
		// Found
		aggregator = &ioAggregators[aggregator_idx->second];
	}

	// Add the measurement to the aggregator
	aggregator->AccumulateMeasurement(cycles_this_with_children);

	// Update ioSample to the last child of ioSample
	JPH_ASSERT(sample[-1].mStartCycle <= ioSample->mEndCycle);
	JPH_ASSERT(sample >= inEnd || sample->mStartCycle >= ioSample->mEndCycle);
	ioSample = sample - 1;
}

void Profiler::DumpInternal()
{
	// Freeze data from threads
	// Note that this is not completely thread safe: As a profile sample is added mCurrentSample is incremented
	// but the data is not written until the sample finishes. So if we dump the profile information while
	// some other thread is running, we may get some garbage information from the previous frame
	Threads threads;
	for (ProfileThread *t : mThreads)
		threads.push_back({ t->mThreadName, t->mSamples, t->mSamples + t->mCurrentSample });

	// Shift all samples so that the first sample is at zero
	uint64 min_cycle = 0xffffffffffffffffUL;
	for (const ThreadSamples &t : threads)
		if (t.mSamplesBegin < t.mSamplesEnd)
			min_cycle = min(min_cycle, t.mSamplesBegin[0].mStartCycle);
	for (const ThreadSamples &t : threads)
		for (ProfileSample *s = t.mSamplesBegin, *end = t.mSamplesEnd; s < end; ++s)
		{
			s->mStartCycle -= min_cycle;
			s->mEndCycle -= min_cycle;
		}

	// Determine tag of this profile
	String tag;
	if (mDumpTag.empty())
	{
		// Next sequence number
		static int number = 0;
		++number;
		tag = ConvertToString(number);
	}
	else
	{
		// Take provided tag
		tag = mDumpTag;
		mDumpTag.clear();
	}

	// Aggregate data across threads
	Aggregators aggregators;
	KeyToAggregator key_to_aggregators;
	for (const ThreadSamples &t : threads)
		for (ProfileSample *s = t.mSamplesBegin, *end = t.mSamplesEnd; s < end; ++s)
			sAggregate(0, Color::sGetDistinctColor(0).GetUInt32(), s, end, aggregators, key_to_aggregators);

	// Dump as chart
	DumpChart(tag.c_str(), threads, key_to_aggregators, aggregators);
}

static String sHTMLEncode(const char *inString)
{
	String str(inString);
	StringReplace(str, "<", "&lt;");
	StringReplace(str, ">", "&gt;");
	return str;
}

void Profiler::DumpChart(const char *inTag, const Threads &inThreads, const KeyToAggregator &inKeyToAggregators, const Aggregators &inAggregators)
{
	// Open file
	std::ofstream f;
	f.open(StringFormat("profile_chart_%s.html", inTag).c_str(), std::ofstream::out | std::ofstream::trunc);
	if (!f.is_open())
		return;

	// Write header
	f << R"(<!DOCTYPE html>
<html>
	<head>
		<title>Profile Chart</title>
		<style>
			html, body {
				padding: 0px;
				border: 0px;
				margin: 0px;
				width: 100%;
				height: 100%;
				overflow: hidden;
			}

			canvas {
				position: absolute;
				top: 10px;
				left: 10px;
				padding: 0px;
				border: 0px;
				margin: 0px;
			}

			#tooltip {
				font: Courier New;
				position: absolute;
				background-color: white;
				border: 1px;
				border-style: solid;
				border-color: black;
				pointer-events: none;
				padding: 5px;
				font: 14px Arial;
				visibility: hidden;
				height: auto;
			}

			.stat {
				color: blue;
				text-align: right;
			}
		</style>
		<script type="text/javascript">
			var canvas;
			var ctx;
			var tooltip;
			var min_scale;
			var scale;
			var offset_x = 0;
			var offset_y = 0;
			var size_y;
			var dragging = false;
			var previous_x = 0;
			var previous_y = 0;
			var bar_height = 15;
			var line_height = bar_height + 2;
			var thread_separation = 6;
			var thread_font_size = 12;
			var thread_font = thread_font_size + "px Arial";
			var bar_font_size = 10;
			var bar_font = bar_font_size + "px Arial";
			var end_cycle = 0;

			function drawChart()
			{
				ctx.clearRect(0, 0, canvas.width, canvas.height);

				ctx.lineWidth = 1;

				var y = offset_y;

				for (var t = 0; t < threads.length; t++)
				{
					// Check if thread has samples
					var thread = threads[t];
					if (thread.start.length == 0)
						continue;

					// Draw thread name
					y += thread_font_size;
					ctx.font = thread_font;
					ctx.fillStyle = "#000000";
					ctx.fillText(thread.thread_name, 0, y);
					y += thread_separation;

					// Draw outlines for each bar of samples
					ctx.fillStyle = "#c0c0c0";
					for (var d = 0; d <= thread.max_depth; d++)
						ctx.fillRect(0, y + d * line_height, canvas.width, bar_height);

					// Draw samples
					ctx.font = bar_font;
					for (var s = 0; s < thread.start.length; s++)
					{
						// Cull bar
						var rx = scale * (offset_x + thread.start[s]);
						if (rx > canvas.width) // right of canvas
							break;
						var rw = scale * thread.cycles[s];
						if (rw < 0.5) // less than half pixel, skip
							continue;
						if (rx + rw < 0) // left of canvas
							continue;

						// Draw bar
						var ry = y + line_height * thread.depth[s];
						ctx.fillStyle = thread.color[s];
						ctx.fillRect(rx, ry, rw, bar_height);
						ctx.strokeStyle = thread.darkened_color[s];
						ctx.strokeRect(rx, ry, rw, bar_height);

						// Get index in aggregated list
						var a = thread.aggregator[s];

						// Draw text
						if (rw > aggregated.name_width[a])
						{
							ctx.fillStyle = "#000000";
							ctx.fillText(aggregated.name[a], rx + (rw - aggregated.name_width[a]) / 2, ry + bar_height - 4);
						}
					}

					// Next line
					y += line_height * (1 + thread.max_depth) + thread_separation;
				}

				// Update size
				size_y = y - offset_y;
			}

			function drawTooltip(mouse_x, mouse_y)
			{
				var y = offset_y;

				for (var t = 0; t < threads.length; t++)
				{
					// Check if thread has samples
					var thread = threads[t];
					if (thread.start.length == 0)
						continue;

					// Thead name
					y += thread_font_size + thread_separation;

					// Draw samples
					for (var s = 0; s < thread.start.length; s++)
					{
						// Cull bar
						var rx = scale * (offset_x + thread.start[s]);
						if (rx > mouse_x)
							break;
						var rw = scale * thread.cycles[s];
						if (rx + rw < mouse_x)
							continue;

						var ry = y + line_height * thread.depth[s];
						if (mouse_y >= ry && mouse_y < ry + bar_height)
						{
							// Get index into aggregated list
							var a = thread.aggregator[s];

							// Found bar, fill in tooltip
							tooltip.style.left = (canvas.offsetLeft + mouse_x) + "px";
							tooltip.style.top = (canvas.offsetTop + mouse_y) + "px";
							tooltip.style.visibility = "visible";
							tooltip.innerHTML = aggregated.name[a] + "<br>"
								+ "<table>"
								+ "<tr><td>Time:</td><td class=\"stat\">" + (1000000 * thread.cycles[s] / cycles_per_second).toFixed(2) + " &micro;s</td></tr>"
								+ "<tr><td>Start:</td><td class=\"stat\">" + (1000000 * thread.start[s] / cycles_per_second).toFixed(2) + " &micro;s</td></tr>"
								+ "<tr><td>End:</td><td class=\"stat\">" + (1000000 * (thread.start[s] + thread.cycles[s]) / cycles_per_second).toFixed(2) + " &micro;s</td></tr>"
								+ "<tr><td>Avg. Time:</td><td class=\"stat\">" + (1000000 * aggregated.cycles_per_frame[a] / cycles_per_second / aggregated.calls[a]).toFixed(2) + " &micro;s</td></tr>"
								+ "<tr><td>Min Time:</td><td class=\"stat\">" + (1000000 * aggregated.min_cycles[a] / cycles_per_second).toFixed(2) + " &micro;s</td></tr>"
								+ "<tr><td>Max Time:</td><td class=\"stat\">" + (1000000 * aggregated.max_cycles[a] / cycles_per_second).toFixed(2) + " &micro;s</td></tr>"
								+ "<tr><td>Time / Frame:</td><td class=\"stat\">" + (1000000 * aggregated.cycles_per_frame[a] / cycles_per_second).toFixed(2) + " &micro;s</td></tr>"
								+ "<tr><td>Calls:</td><td class=\"stat\">" + aggregated.calls[a] + "</td></tr>"
								+ "</table>";
							return;
						}
					}

					// Next line
					y += line_height * (1 + thread.max_depth) + thread_separation;
				}

				// No bar found, hide tooltip
				tooltip.style.visibility = "hidden";
			}

			function onMouseDown(evt)
			{
				dragging = true;
				previous_x = evt.clientX, previous_y = evt.clientY;
				tooltip.style.visibility = "hidden";
			}

			function onMouseUp(evt)
			{
				dragging = false;
			}

			function clampMotion()
			{
				// Clamp horizontally
				var min_offset_x = canvas.width / scale - end_cycle;
				if (offset_x < min_offset_x)
					offset_x = min_offset_x;
				if (offset_x > 0)
					offset_x = 0;

				// Clamp vertically
				var min_offset_y = canvas.height - size_y;
				if (offset_y < min_offset_y)
					offset_y = min_offset_y;
				if (offset_y > 0)
					offset_y = 0;

				// Clamp scale
				if (scale < min_scale)
					scale = min_scale;
				var max_scale = 1000 * min_scale;
				if (scale > max_scale)
					scale = max_scale;
			}

			function onMouseMove(evt)
			{
				if (dragging)
				{
					// Calculate new offset
					offset_x += (evt.clientX - previous_x) / scale;
					offset_y += evt.clientY - previous_y;

					clampMotion();

					drawChart();
				}
				else
					drawTooltip(evt.clientX - canvas.offsetLeft, evt.clientY - canvas.offsetTop);

				previous_x = evt.clientX, previous_y = evt.clientY;
			}

			function onScroll(evt)
			{
				tooltip.style.visibility = "hidden";

				var old_scale = scale;
				if (evt.deltaY > 0)
					scale /= 1.1;
				else
					scale *= 1.1;

				clampMotion();

				// Ensure that event under mouse stays under mouse
				var x = previous_x - canvas.offsetLeft;
				offset_x += x / scale - x / old_scale;

				clampMotion();

				drawChart();
			}

			function darkenColor(color)
			{
				var i = parseInt(color.slice(1), 16);

				var r = i >> 16;
				var g = (i >> 8) & 0xff;
				var b = i & 0xff;

				r = Math.round(0.8 * r);
				g = Math.round(0.8 * g);
				b = Math.round(0.8 * b);

				i = (r << 16) + (g << 8) + b;

				return "#" + i.toString(16);
			}

			function startChart()
			{
				// Fetch elements
				canvas = document.getElementById('canvas');
				ctx = canvas.getContext("2d");
				tooltip = document.getElementById('tooltip');

				// Resize canvas to fill screen
				canvas.width = document.body.offsetWidth - 20;
				canvas.height = document.body.offsetHeight - 20;

				// Register mouse handlers
				canvas.onmousedown = onMouseDown;
				canvas.onmouseup = onMouseUp;
				canvas.onmouseout = onMouseUp;
				canvas.onmousemove = onMouseMove;
				canvas.onwheel	= onScroll;

				for (var t = 0; t < threads.length; t++)
				{
					var thread = threads[t];

					// Calculate darkened colors
					thread.darkened_color = new Array(thread.color.length);
					for (var s = 0; s < thread.color.length; s++)
						thread.darkened_color[s] = darkenColor(thread.color[s]);

					// Calculate max depth and end cycle
					thread.max_depth = 0;
					for (var s = 0; s < thread.start.length; s++)
					{
						thread.max_depth = Math.max(thread.max_depth, thread.depth[s]);
						end_cycle = Math.max(end_cycle, thread.start[s] + thread.cycles[s]);
					}
				}

				// Calculate width of name strings
				ctx.font = bar_font;
				aggregated.name_width = new Array(aggregated.name.length);
				for (var a = 0; a < aggregated.name.length; a++)
					aggregated.name_width[a] = ctx.measureText(aggregated.name[a]).width;

				// Store scale properties
				min_scale = canvas.width / end_cycle;
				scale = min_scale;

				drawChart();
			}
		</script>
	</head>
	<body onload="startChart();">
	<script type="text/javascript">
)";

	// Get cycles per second
	uint64 cycles_per_second = GetProcessorTicksPerSecond();
	f << "var cycles_per_second = " << cycles_per_second << ";\n";

	// Dump samples
	f << "var threads = [\n";
	bool first_thread = true;
	for (const ThreadSamples &t : inThreads)
	{
		if (!first_thread)
			f << ",\n";
		first_thread = false;

		f << "{\nthread_name: \"" << t.mThreadName << "\",\naggregator: [";
		bool first = true;
		for (const ProfileSample *s = t.mSamplesBegin, *end = t.mSamplesEnd; s < end; ++s)
		{
			if (!first)
				f << ",";
			first = false;
			f << inKeyToAggregators.find(s->mName)->second;
		}
		f << "],\ncolor: [";
		first = true;
		for (const ProfileSample *s = t.mSamplesBegin, *end = t.mSamplesEnd; s < end; ++s)
		{
			if (!first)
				f << ",";
			first = false;
			Color c(s->mColor);
			f << StringFormat("\"#%02x%02x%02x\"", c.r, c.g, c.b);
		}
		f << "],\nstart: [";
		first = true;
		for (const ProfileSample *s = t.mSamplesBegin, *end = t.mSamplesEnd; s < end; ++s)
		{
			if (!first)
				f << ",";
			first = false;
			f << s->mStartCycle;
		}
		f << "],\ncycles: [";
		first = true;
		for (const ProfileSample *s = t.mSamplesBegin, *end = t.mSamplesEnd; s < end; ++s)
		{
			if (!first)
				f << ",";
			first = false;
			f << s->mEndCycle - s->mStartCycle;
		}
		f << "],\ndepth: [";
		first = true;
		for (const ProfileSample *s = t.mSamplesBegin, *end = t.mSamplesEnd; s < end; ++s)
		{
			if (!first)
				f << ",";
			first = false;
			f << int(s->mDepth);
		}
		f << "]\n}";
	}

	// Dump aggregated data
	f << "];\nvar aggregated = {\nname: [";
	bool first = true;
	for (const Aggregator &a : inAggregators)
	{
		if (!first)
			f << ",";
		first = false;
		String name = "\"" + sHTMLEncode(a.mName) + "\"";
		f << name;
	}
	f << "],\ncalls: [";
	first = true;
	for (const Aggregator &a : inAggregators)
	{
		if (!first)
			f << ",";
		first = false;
		f << a.mCallCounter;
	}
	f << "],\nmin_cycles: [";
	first = true;
	for (const Aggregator &a : inAggregators)
	{
		if (!first)
			f << ",";
		first = false;
		f << a.mMinCyclesInCallWithChildren;
	}
	f << "],\nmax_cycles: [";
	first = true;
	for (const Aggregator &a : inAggregators)
	{
		if (!first)
			f << ",";
		first = false;
		f << a.mMaxCyclesInCallWithChildren;
	}
	f << "],\ncycles_per_frame: [";
	first = true;
	for (const Aggregator &a : inAggregators)
	{
		if (!first)
			f << ",";
		first = false;
		f << a.mTotalCyclesInCallWithChildren;
	}

	// Write footer
	f << R"(]};
</script>

<canvas id="canvas"></canvas>
<div id="tooltip"></div>

</tbody></table></body></html>)";
}

#endif // JPH_PROFILE_ENABLED

JPH_NAMESPACE_END

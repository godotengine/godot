/*
 * PluginEventQueue.h
 * ------------------
 * Purpose: Alternative, easy to use implementation of the VST event queue mechanism.
 * Notes  : Modelled after an idea from http://www.kvraudio.com/forum/viewtopic.php?p=3043807#p3043807
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include <deque>
#include "../common/mptMutex.h"

// Copied packing options from affectx.h
#if TARGET_API_MAC_CARBON
	#ifdef __LP64__
	#pragma options align=power
	#else
	#pragma options align=mac68k
	#endif
#elif defined __BORLANDC__
	#pragma -a8
#elif defined(__GNUC__)
	#pragma pack(push,8)
#elif defined(WIN32) || defined(__FLAT__)
	#pragma pack(push)
	#pragma pack(8)
#endif

OPENMPT_NAMESPACE_BEGIN

// Alternative, easy to use implementation of VstEvents struct.
template <size_t N>
class PluginEventQueue
{
protected:

	// Although originally all event types were apparently supposed to be of the same size, this is not the case on 64-Bit systems.
	// VstMidiSysexEvent contains 3 pointers, so the struct size differs between 32-Bit and 64-Bit systems.
	union Event
	{
		VstEvent event;
		VstMidiEvent midi;
		VstMidiSysexEvent sysex;
	};

	// The first three member variables of this class mustn't be changed, to be compatible with the original VSTEvents struct.
	VstInt32 numEvents;		///< number of Events in array
	VstIntPtr reserved;		///< zero (Reserved for future use)
	VstEvent *events[N];	///< event pointer array
	// Here we store our events.
	std::deque<Event> eventQueue;
	// Since plugins can also add events to the queue (even from a different thread than the processing thread),
	// we need to ensure that reading and writing is never done in parallel.
	mpt::mutex criticalSection;

public:

	PluginEventQueue()
	{
		numEvents = 0;
		reserved = 0;
		MemsetZero(events);
	}

	// Get the number of events that are currently in the output buffer.
	// It is possible that there are more events left in the queue if the output buffer is full.
	size_t GetNumEvents()
	{
		return numEvents;
	}

	// Get the number of events that are currently queued, but not in the output buffer.
	size_t GetNumQueuedEvents()
	{
		return eventQueue.size() - numEvents;
	}

	// Add a VST event to the queue. Returns true on success.
	// Set insertFront to true to prioritise this event (i.e. add it at the front of the queue instead of the back)
	bool Enqueue(const VstEvent *event, bool insertFront = false)
	{
		MPT_ASSERT(event->type != kVstSysExType || event->byteSize == sizeof(VstMidiSysexEvent));
		MPT_ASSERT(event->type != kVstMidiType || event->byteSize == sizeof(VstMidiEvent));

		Event copyEvent;
		size_t copySize = std::min(size_t(event->byteSize), sizeof(copyEvent));
		// randomid by Insert Piz Here sends events of type kVstMidiType, but with a claimed size of 24 bytes instead of 32.
		if(event->type == kVstSysExType)
			copySize = sizeof(VstMidiSysexEvent);
		else if(event->type == kVstMidiType)
			copySize = sizeof(VstMidiEvent);
		memcpy(&copyEvent, event, copySize);

		if(event->type == kVstSysExType)
		{
			// SysEx messages need to be copied, as the space used for the dump might be freed in the meantime.
			VstMidiSysexEvent &e = copyEvent.sysex;
			e.sysexDump = new (std::nothrow) char[e.dumpBytes];
			if(e.sysexDump == nullptr)
			{
				return false;
			}
			memcpy(e.sysexDump, reinterpret_cast<const VstMidiSysexEvent *>(event)->sysexDump, e.dumpBytes);
		}

		MPT_LOCK_GUARD<mpt::mutex> lock(criticalSection);
		if(insertFront)
			eventQueue.push_front(copyEvent);
		else
			eventQueue.push_back(copyEvent);
		return true;
	}

	// Set up the queue for transmitting to the plugin. Returns number of elements that are going to be transmitted.
	VstInt32 Finalise()
	{
		MPT_LOCK_GUARD<mpt::mutex> lock(criticalSection);
		numEvents = static_cast<VstInt32>(std::min(eventQueue.size(), N));
		for(VstInt32 i = 0; i < numEvents; i++)
		{
			events[i] = &eventQueue[i].event;
		}
		return numEvents;
	}

	// Remove transmitted events from the queue
	void Clear()
	{
		MPT_LOCK_GUARD<mpt::mutex> lock(criticalSection);
		if(numEvents)
		{
			// Release temporarily allocated buffer for SysEx messages
			for(std::deque<Event>::iterator e = eventQueue.begin(); e != eventQueue.begin() + numEvents; e++)
			{
				if(e->event.type == kVstSysExType)
				{
					delete[] e->sysex.sysexDump;
				}
			}
			eventQueue.erase(eventQueue.begin(), eventQueue.begin() + numEvents);
			numEvents = 0;
		}
	}

};

#if TARGET_API_MAC_CARBON
	#pragma options align=reset
#elif defined(WIN32) || defined(__FLAT__) || defined(__GNUC__)
	#pragma pack(pop)
#elif defined __BORLANDC__
	#pragma -a-
#endif


OPENMPT_NAMESPACE_END

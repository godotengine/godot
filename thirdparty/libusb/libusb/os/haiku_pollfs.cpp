/*
 * Copyright 2007-2008, Haiku Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Michael Lotz <mmlr@mlotz.ch>
 */

#include "haiku_usb.h"
#include <cstdio>
#include <Directory.h>
#include <Entry.h>
#include <Looper.h>
#include <Messenger.h>
#include <Node.h>
#include <NodeMonitor.h>
#include <Path.h>
#include <cstring>

class WatchedEntry {
public:
			WatchedEntry(BMessenger *, entry_ref *);
			~WatchedEntry();
	bool		EntryCreated(entry_ref *ref);
	bool		EntryRemoved(ino_t node);
	bool		InitCheck();

private:
	BMessenger*	fMessenger;
	node_ref	fNode;
	bool		fIsDirectory;
	USBDevice*	fDevice;
	WatchedEntry*	fEntries;
	WatchedEntry*	fLink;
	bool		fInitCheck;
};


class RosterLooper : public BLooper {
public:
			RosterLooper(USBRoster *);
	void		Stop();
	virtual void	MessageReceived(BMessage *);
	bool		InitCheck();

private:
	USBRoster*	fRoster;
	WatchedEntry*	fRoot;
	BMessenger*	fMessenger;
	bool		fInitCheck;
};


WatchedEntry::WatchedEntry(BMessenger *messenger, entry_ref *ref)
	:	fMessenger(messenger),
		fIsDirectory(false),
		fDevice(NULL),
		fEntries(NULL),
		fLink(NULL),
		fInitCheck(false)
{
	BEntry entry(ref);
	entry.GetNodeRef(&fNode);

	BDirectory directory;
	if (entry.IsDirectory() && directory.SetTo(ref) >= B_OK) {
		fIsDirectory = true;

		while (directory.GetNextEntry(&entry) >= B_OK) {
			if (entry.GetRef(ref) < B_OK)
				continue;

			WatchedEntry *child = new(std::nothrow) WatchedEntry(fMessenger, ref);
			if (child == NULL)
				continue;
			if (child->InitCheck() == false) {
				delete child;
				continue;
			}

			child->fLink = fEntries;
			fEntries = child;
		}

		watch_node(&fNode, B_WATCH_DIRECTORY, *fMessenger);
	}
	else {
		if (strncmp(ref->name, "raw", 3) == 0)
			return;

		BPath path, parent_path;
		entry.GetPath(&path);
		fDevice = new(std::nothrow) USBDevice(path.Path());
		if (fDevice != NULL && fDevice->InitCheck() == true) {
			// Add this new device to each active context's device list
			struct libusb_context *ctx;
			unsigned long session_id = (unsigned long)&fDevice;

			usbi_mutex_lock(&active_contexts_lock);
			list_for_each_entry(ctx, &active_contexts_list, list, struct libusb_context) {
				struct libusb_device *dev = usbi_get_device_by_session_id(ctx, session_id);
				if (dev) {
					usbi_dbg("using previously allocated device with location %lu", session_id);
					libusb_unref_device(dev);
					continue;
				}
				usbi_dbg("allocating new device with location %lu", session_id);
				dev = usbi_alloc_device(ctx, session_id);
				if (!dev) {
					usbi_dbg("device allocation failed");
					continue;
				}
				*((USBDevice **)dev->os_priv) = fDevice;

				// Calculate pseudo-device-address
				int addr, tmp;
				if (strcmp(path.Leaf(), "hub") == 0)
					tmp = 100;	//Random Number
				else
					sscanf(path.Leaf(), "%d", &tmp);
				addr = tmp + 1;
				path.GetParent(&parent_path);
				while (strcmp(parent_path.Leaf(), "usb") != 0) {
					sscanf(parent_path.Leaf(), "%d", &tmp);
					addr += tmp + 1;
					parent_path.GetParent(&parent_path);
				}
				sscanf(path.Path(), "/dev/bus/usb/%d", &dev->bus_number);
				dev->device_address = addr - (dev->bus_number + 1);

				if (usbi_sanitize_device(dev) < 0) {
					usbi_dbg("device sanitization failed");
					libusb_unref_device(dev);
					continue;
				}
				usbi_connect_device(dev);
			}
			usbi_mutex_unlock(&active_contexts_lock);
		}
		else if (fDevice) {
			delete fDevice;
			fDevice = NULL;
			return;
		}
	}
	fInitCheck = true;
}


WatchedEntry::~WatchedEntry()
{
	if (fIsDirectory) {
		watch_node(&fNode, B_STOP_WATCHING, *fMessenger);

		WatchedEntry *child = fEntries;
		while (child) {
			WatchedEntry *next = child->fLink;
			delete child;
			child = next;
		}
	}

	if (fDevice) {
		// Remove this device from each active context's device list
		struct libusb_context *ctx;
		struct libusb_device *dev;
		unsigned long session_id = (unsigned long)&fDevice;

		usbi_mutex_lock(&active_contexts_lock);
		list_for_each_entry(ctx, &active_contexts_list, list, struct libusb_context) {
			dev = usbi_get_device_by_session_id(ctx, session_id);
			if (dev != NULL) {
				usbi_disconnect_device(dev);
				libusb_unref_device(dev);
			} else {
				usbi_dbg("device with location %lu not found", session_id);
			}
		}
		usbi_mutex_static_unlock(&active_contexts_lock);
		delete fDevice;
	}
}


bool
WatchedEntry::EntryCreated(entry_ref *ref)
{
	if (!fIsDirectory)
		return false;

	if (ref->directory != fNode.node) {
		WatchedEntry *child = fEntries;
		while (child) {
			if (child->EntryCreated(ref))
				return true;
			child = child->fLink;
		}
		return false;
	}

	WatchedEntry *child = new(std::nothrow) WatchedEntry(fMessenger, ref);
	if (child == NULL)
		return false;
	child->fLink = fEntries;
	fEntries = child;
	return true;
}


bool
WatchedEntry::EntryRemoved(ino_t node)
{
	if (!fIsDirectory)
		return false;

	WatchedEntry *child = fEntries;
	WatchedEntry *lastChild = NULL;
	while (child) {
		if (child->fNode.node == node) {
			if (lastChild)
				lastChild->fLink = child->fLink;
			else
				fEntries = child->fLink;
			delete child;
			return true;
		}

		if (child->EntryRemoved(node))
			return true;

		lastChild = child;
		child = child->fLink;
	}
	return false;
}


bool
WatchedEntry::InitCheck()
{
	return fInitCheck;
}


RosterLooper::RosterLooper(USBRoster *roster)
	:	BLooper("LibusbRoster Looper"),
		fRoster(roster),
		fRoot(NULL),
		fMessenger(NULL),
		fInitCheck(false)
{
	BEntry entry("/dev/bus/usb");
	if (!entry.Exists()) {
		usbi_err(NULL, "usb_raw not published");
		return;
	}

	Run();
	fMessenger = new(std::nothrow) BMessenger(this);
	if (fMessenger == NULL) {
		usbi_err(NULL, "error creating BMessenger object");
		return;
	}

	if (Lock()) {
		entry_ref ref;
		entry.GetRef(&ref);
		fRoot = new(std::nothrow) WatchedEntry(fMessenger, &ref);
		Unlock();
		if (fRoot == NULL)
			return;
		if (fRoot->InitCheck() == false) {
			delete fRoot;
			fRoot = NULL;
			return;
		}
	}
	fInitCheck = true;
}


void
RosterLooper::Stop()
{
	Lock();
	delete fRoot;
	delete fMessenger;
	Quit();
}


void
RosterLooper::MessageReceived(BMessage *message)
{
	int32 opcode;
	if (message->FindInt32("opcode", &opcode) < B_OK)
		return;

	switch (opcode) {
		case B_ENTRY_CREATED:
		{
			dev_t device;
			ino_t directory;
			const char *name;
			if (message->FindInt32("device", &device) < B_OK ||
				message->FindInt64("directory", &directory) < B_OK ||
				message->FindString("name", &name) < B_OK)
				break;

			entry_ref ref(device, directory, name);
			fRoot->EntryCreated(&ref);
			break;
		}
		case B_ENTRY_REMOVED:
		{
			ino_t node;
			if (message->FindInt64("node", &node) < B_OK)
				break;
			fRoot->EntryRemoved(node);
			break;
		}
	}
}


bool
RosterLooper::InitCheck()
{
	return fInitCheck;
}


USBRoster::USBRoster()
	:	fLooper(NULL)
{
}


USBRoster::~USBRoster()
{
	Stop();
}


int
USBRoster::Start()
{
	if (fLooper == NULL) {
		fLooper = new(std::nothrow) RosterLooper(this);
		if (fLooper == NULL || ((RosterLooper *)fLooper)->InitCheck() == false) {
			if (fLooper)
				fLooper = NULL;
			return LIBUSB_ERROR_OTHER;
		}
	}
	return LIBUSB_SUCCESS;
}


void
USBRoster::Stop()
{
	if (fLooper) {
		((RosterLooper *)fLooper)->Stop();
		fLooper = NULL;
	}
}

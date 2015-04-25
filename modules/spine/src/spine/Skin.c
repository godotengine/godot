/******************************************************************************
 * Spine Runtimes Software License
 * Version 2.1
 * 
 * Copyright (c) 2013, Esoteric Software
 * All rights reserved.
 * 
 * You are granted a perpetual, non-exclusive, non-sublicensable and
 * non-transferable license to install, execute and perform the Spine Runtimes
 * Software (the "Software") solely for internal use. Without the written
 * permission of Esoteric Software (typically granted by licensing Spine), you
 * may not (a) modify, translate, adapt or otherwise create derivative works,
 * improvements of the Software or develop new applications using the Software
 * or (b) remove, delete, alter or obscure any trademarks or any copyright,
 * trademark, patent or other intellectual property or proprietary rights
 * notices on or in the Software, including any copy thereof. Redistributions
 * in binary or source form must include this license and terms.
 * 
 * THIS SOFTWARE IS PROVIDED BY ESOTERIC SOFTWARE "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL ESOTERIC SOFTARE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#include <spine/Skin.h>
#include <spine/extension.h>

typedef struct _Entry _Entry;
struct _Entry {
	int slotIndex;
	const char* name;
	spAttachment* attachment;
	_Entry* next;
};

_Entry* _Entry_create (int slotIndex, const char* name, spAttachment* attachment) {
	_Entry* self = NEW(_Entry);
	self->slotIndex = slotIndex;
	MALLOC_STR(self->name, name);
	self->attachment = attachment;
	return self;
}

void _Entry_dispose (_Entry* self) {
	spAttachment_dispose(self->attachment);
	FREE(self->name);
	FREE(self);
}

/**/

typedef struct {
	spSkin super;
	_Entry* entries;
} _spSkin;

spSkin* spSkin_create (const char* name) {
	spSkin* self = SUPER(NEW(_spSkin));
	MALLOC_STR(self->name, name);
	return self;
}

void spSkin_dispose (spSkin* self) {
	_Entry* entry = SUB_CAST(_spSkin, self)->entries;
	while (entry) {
		_Entry* nextEntry = entry->next;
		_Entry_dispose(entry);
		entry = nextEntry;
	}

	FREE(self->name);
	FREE(self);
}

void spSkin_addAttachment (spSkin* self, int slotIndex, const char* name, spAttachment* attachment) {
	_Entry* newEntry = _Entry_create(slotIndex, name, attachment);
	newEntry->next = SUB_CAST(_spSkin, self)->entries;
	SUB_CAST(_spSkin, self)->entries = newEntry;
}

spAttachment* spSkin_getAttachment (const spSkin* self, int slotIndex, const char* name) {
	const _Entry* entry = SUB_CAST(_spSkin, self)->entries;
	while (entry) {
		if (entry->slotIndex == slotIndex && strcmp(entry->name, name) == 0) return entry->attachment;
		entry = entry->next;
	}
	return 0;
}

const char* spSkin_getAttachmentName (const spSkin* self, int slotIndex, int attachmentIndex) {
	const _Entry* entry = SUB_CAST(_spSkin, self)->entries;
	int i = 0;
	while (entry) {
		if (entry->slotIndex == slotIndex) {
			if (i == attachmentIndex) return entry->name;
			i++;
		}
		entry = entry->next;
	}
	return 0;
}

void spSkin_attachAll (const spSkin* self, spSkeleton* skeleton, const spSkin* oldSkin) {
	const _Entry *entry = SUB_CAST(_spSkin, oldSkin)->entries;
	while (entry) {
		spSlot *slot = skeleton->slots[entry->slotIndex];
		if (slot->attachment == entry->attachment) {
			spAttachment *attachment = spSkin_getAttachment(self, entry->slotIndex, entry->name);
			if (attachment) spSlot_setAttachment(slot, attachment);
		}
		entry = entry->next;
	}
}

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

#include <spine/AnimationStateData.h>
#include <spine/extension.h>

typedef struct _ToEntry _ToEntry;
struct _ToEntry {
	spAnimation* animation;
	float duration;
	_ToEntry* next;
};

_ToEntry* _ToEntry_create (spAnimation* to, float duration) {
	_ToEntry* self = NEW(_ToEntry);
	self->animation = to;
	self->duration = duration;
	return self;
}

void _ToEntry_dispose (_ToEntry* self) {
	FREE(self);
}

/**/

typedef struct _FromEntry _FromEntry;
struct _FromEntry {
	spAnimation* animation;
	_ToEntry* toEntries;
	_FromEntry* next;
};

_FromEntry* _FromEntry_create (spAnimation* from) {
	_FromEntry* self = NEW(_FromEntry);
	self->animation = from;
	return self;
}

void _FromEntry_dispose (_FromEntry* self) {
	FREE(self);
}

/**/

spAnimationStateData* spAnimationStateData_create (spSkeletonData* skeletonData) {
	spAnimationStateData* self = NEW(spAnimationStateData);
	CONST_CAST(spSkeletonData*, self->skeletonData) = skeletonData;
	return self;
}

void spAnimationStateData_dispose (spAnimationStateData* self) {
	_ToEntry* toEntry;
	_ToEntry* nextToEntry;
	_FromEntry* nextFromEntry;

	_FromEntry* fromEntry = (_FromEntry*)self->entries;
	while (fromEntry) {
		toEntry = fromEntry->toEntries;
		while (toEntry) {
			nextToEntry = toEntry->next;
			_ToEntry_dispose(toEntry);
			toEntry = nextToEntry;
		}
		nextFromEntry = fromEntry->next;
		_FromEntry_dispose(fromEntry);
		fromEntry = nextFromEntry;
	}

	FREE(self);
}

void spAnimationStateData_setMixByName (spAnimationStateData* self, const char* fromName, const char* toName, float duration) {
	spAnimation* to;
	spAnimation* from = spSkeletonData_findAnimation(self->skeletonData, fromName);
	if (!from) return;
	to = spSkeletonData_findAnimation(self->skeletonData, toName);
	if (!to) return;
	spAnimationStateData_setMix(self, from, to, duration);
}

void spAnimationStateData_setMix (spAnimationStateData* self, spAnimation* from, spAnimation* to, float duration) {
	/* Find existing FromEntry. */
	_ToEntry* toEntry;
	_FromEntry* fromEntry = (_FromEntry*)self->entries;
	while (fromEntry) {
		if (fromEntry->animation == from) {
			/* Find existing ToEntry. */
			toEntry = fromEntry->toEntries;
			while (toEntry) {
				if (toEntry->animation == to) {
					toEntry->duration = duration;
					return;
				}
				toEntry = toEntry->next;
			}
			break; /* Add new ToEntry to the existing FromEntry. */
		}
		fromEntry = fromEntry->next;
	}
	if (!fromEntry) {
		fromEntry = _FromEntry_create(from);
		fromEntry->next = (_FromEntry*)self->entries;
		CONST_CAST(_FromEntry*, self->entries) = fromEntry;
	}
	toEntry = _ToEntry_create(to, duration);
	toEntry->next = fromEntry->toEntries;
	fromEntry->toEntries = toEntry;
}

float spAnimationStateData_getMix (spAnimationStateData* self, spAnimation* from, spAnimation* to) {
	_FromEntry* fromEntry = (_FromEntry*)self->entries;
	while (fromEntry) {
		if (fromEntry->animation == from) {
			_ToEntry* toEntry = fromEntry->toEntries;
			while (toEntry) {
				if (toEntry->animation == to) return toEntry->duration;
				toEntry = toEntry->next;
			}
		}
		fromEntry = fromEntry->next;
	}
	return self->defaultMix;
}

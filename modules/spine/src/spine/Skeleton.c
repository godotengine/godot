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

#include <spine/Skeleton.h>
#include <string.h>
#include <spine/extension.h>

typedef struct {
	spSkeleton super;

	int boneCacheCount;
	int* boneCacheCounts;
	spBone*** boneCache;
} _spSkeleton;

spSkeleton* spSkeleton_create (spSkeletonData* data) {
	int i, ii;

	_spSkeleton* internal = NEW(_spSkeleton);
	spSkeleton* self = SUPER(internal);
	CONST_CAST(spSkeletonData*, self->data) = data;

	self->bonesCount = self->data->bonesCount;
	self->bones = MALLOC(spBone*, self->bonesCount);

	for (i = 0; i < self->bonesCount; ++i) {
		spBoneData* boneData = self->data->bones[i];
		spBone* parent = 0;
		if (boneData->parent) {
			/* Find parent bone. */
			for (ii = 0; ii < self->bonesCount; ++ii) {
				if (data->bones[ii] == boneData->parent) {
					parent = self->bones[ii];
					break;
				}
			}
		}
		self->bones[i] = spBone_create(boneData, self, parent);
	}
	CONST_CAST(spBone*, self->root) = self->bones[0];

	self->slotsCount = data->slotsCount;
	self->slots = MALLOC(spSlot*, self->slotsCount);
	for (i = 0; i < self->slotsCount; ++i) {
		spSlotData *slotData = data->slots[i];

		/* Find bone for the slotData's boneData. */
		spBone* bone = 0;
		for (ii = 0; ii < self->bonesCount; ++ii) {
			if (data->bones[ii] == slotData->boneData) {
				bone = self->bones[ii];
				break;
			}
		}
		self->slots[i] = spSlot_create(slotData, bone);
	}

	self->drawOrder = MALLOC(spSlot*, self->slotsCount);
	memcpy(self->drawOrder, self->slots, sizeof(spSlot*) * self->slotsCount);

	self->r = 1;
	self->g = 1;
	self->b = 1;
	self->a = 1;

	self->ikConstraintsCount = data->ikConstraintsCount;
	self->ikConstraints = MALLOC(spIkConstraint*, self->ikConstraintsCount);
	for (i = 0; i < self->data->ikConstraintsCount; ++i)
		self->ikConstraints[i] = spIkConstraint_create(self->data->ikConstraints[i], self);

	spSkeleton_updateCache(self);

	return self;
}

void spSkeleton_dispose (spSkeleton* self) {
	int i;
	_spSkeleton* internal = SUB_CAST(_spSkeleton, self);

	for (i = 0; i < internal->boneCacheCount; ++i)
		FREE(internal->boneCache[i]);
	FREE(internal->boneCache);
	FREE(internal->boneCacheCounts);

	for (i = 0; i < self->bonesCount; ++i)
		spBone_dispose(self->bones[i]);
	FREE(self->bones);

	for (i = 0; i < self->slotsCount; ++i)
		spSlot_dispose(self->slots[i]);
	FREE(self->slots);

	for (i = 0; i < self->ikConstraintsCount; ++i)
		spIkConstraint_dispose(self->ikConstraints[i]);
	FREE(self->ikConstraints);

	FREE(self->drawOrder);
	FREE(self);
}

void spSkeleton_updateCache (const spSkeleton* self) {
	int i, ii;
	_spSkeleton* internal = SUB_CAST(_spSkeleton, self);

	for (i = 0; i < internal->boneCacheCount; ++i)
		FREE(internal->boneCache[i]);
	FREE(internal->boneCache);
	FREE(internal->boneCacheCounts);

	internal->boneCacheCount = self->ikConstraintsCount + 1;
	internal->boneCache = MALLOC(spBone**, internal->boneCacheCount);
	internal->boneCacheCounts = CALLOC(int, internal->boneCacheCount);

	/* Compute array sizes. */
	for (i = 0; i < self->bonesCount; ++i) {
		spBone* current = self->bones[i];
		do {
			for (ii = 0; ii < self->ikConstraintsCount; ++ii) {
				spIkConstraint* ikConstraint = self->ikConstraints[ii];
				spBone* parent = ikConstraint->bones[0];
				spBone* child = ikConstraint->bones[ikConstraint->bonesCount - 1];
				while (1) {
					if (current == child) {
						internal->boneCacheCounts[ii]++;
						internal->boneCacheCounts[ii + 1]++;
						goto outer1;
					}
					if (child == parent) break;
					child = child->parent;
				}
			}
			current = current->parent;
		} while (current);
		internal->boneCacheCounts[0]++;
		outer1: {}
	}

	for (i = 0; i < internal->boneCacheCount; ++i)
		internal->boneCache[i] = MALLOC(spBone*, internal->boneCacheCounts[i]);
	memset(internal->boneCacheCounts, 0, internal->boneCacheCount * sizeof(int));

	/* Populate arrays. */
	for (i = 0; i < self->bonesCount; ++i) {
		spBone* bone = self->bones[i];
		spBone* current = bone;
		do {
			for (ii = 0; ii < self->ikConstraintsCount; ++ii) {
				spIkConstraint* ikConstraint = self->ikConstraints[ii];
				spBone* parent = ikConstraint->bones[0];
				spBone* child = ikConstraint->bones[ikConstraint->bonesCount - 1];
				while (1) {
					if (current == child) {
						internal->boneCache[ii][internal->boneCacheCounts[ii]++] = bone;
						internal->boneCache[ii + 1][internal->boneCacheCounts[ii + 1]++] = bone;
						goto outer2;
					}
					if (child == parent) break;
					child = child->parent;
				}
			}
			current = current->parent;
		} while (current);
		internal->boneCache[0][internal->boneCacheCounts[0]++] = bone;
		outer2: {}
	}
}

void spSkeleton_updateWorldTransform (const spSkeleton* self) {
	int i, ii, nn, last;
	_spSkeleton* internal = SUB_CAST(_spSkeleton, self);

	for (i = 0; i < self->bonesCount; ++i)
		self->bones[i]->rotationIK = self->bones[i]->rotation;

	i = 0;
	last = internal->boneCacheCount - 1;
	while (1) {
		for (ii = 0, nn = internal->boneCacheCounts[i]; ii < nn; ++ii)
			spBone_updateWorldTransform(internal->boneCache[i][ii]);
		if (i == last) break;
		spIkConstraint_apply(self->ikConstraints[i]);
		i++;
	}
}

void spSkeleton_setToSetupPose (const spSkeleton* self) {
	spSkeleton_setBonesToSetupPose(self);
	spSkeleton_setSlotsToSetupPose(self);
}

void spSkeleton_setBonesToSetupPose (const spSkeleton* self) {
	int i;
	for (i = 0; i < self->bonesCount; ++i)
		spBone_setToSetupPose(self->bones[i]);

	for (i = 0; i < self->ikConstraintsCount; ++i) {
		spIkConstraint* ikConstraint = self->ikConstraints[i];
		ikConstraint->bendDirection = ikConstraint->data->bendDirection;
		ikConstraint->mix = ikConstraint->data->mix;
	}
}

void spSkeleton_setSlotsToSetupPose (const spSkeleton* self) {
	int i;
	memcpy(self->drawOrder, self->slots, self->slotsCount * sizeof(spSlot*));
	for (i = 0; i < self->slotsCount; ++i)
		spSlot_setToSetupPose(self->slots[i]);
}

spBone* spSkeleton_findBone (const spSkeleton* self, const char* boneName) {
	int i;
	for (i = 0; i < self->bonesCount; ++i)
		if (strcmp(self->data->bones[i]->name, boneName) == 0) return self->bones[i];
	return 0;
}

int spSkeleton_findBoneIndex (const spSkeleton* self, const char* boneName) {
	int i;
	for (i = 0; i < self->bonesCount; ++i)
		if (strcmp(self->data->bones[i]->name, boneName) == 0) return i;
	return -1;
}

spSlot* spSkeleton_findSlot (const spSkeleton* self, const char* slotName) {
	int i;
	for (i = 0; i < self->slotsCount; ++i)
		if (strcmp(self->data->slots[i]->name, slotName) == 0) return self->slots[i];
	return 0;
}

int spSkeleton_findSlotIndex (const spSkeleton* self, const char* slotName) {
	int i;
	for (i = 0; i < self->slotsCount; ++i)
		if (strcmp(self->data->slots[i]->name, slotName) == 0) return i;
	return -1;
}

int spSkeleton_setSkinByName (spSkeleton* self, const char* skinName) {
	spSkin *skin;
	if (!skinName) {
		spSkeleton_setSkin(self, 0);
		return 1;
	}
	skin = spSkeletonData_findSkin(self->data, skinName);
	if (!skin) return 0;
	spSkeleton_setSkin(self, skin);
	return 1;
}

void spSkeleton_setSkin (spSkeleton* self, spSkin* newSkin) {
	if (newSkin) {
		if (self->skin)
			spSkin_attachAll(newSkin, self, self->skin);
		else {
			/* No previous skin, attach setup pose attachments. */
			int i;
			for (i = 0; i < self->slotsCount; ++i) {
				spSlot* slot = self->slots[i];
				if (slot->data->attachmentName) {
					spAttachment* attachment = spSkin_getAttachment(newSkin, i, slot->data->attachmentName);
					if (attachment) spSlot_setAttachment(slot, attachment);
				}
			}
		}
	}
	CONST_CAST(spSkin*, self->skin) = newSkin;
}

spAttachment* spSkeleton_getAttachmentForSlotName (const spSkeleton* self, const char* slotName, const char* attachmentName) {
	int slotIndex = spSkeletonData_findSlotIndex(self->data, slotName);
	return spSkeleton_getAttachmentForSlotIndex(self, slotIndex, attachmentName);
}

spAttachment* spSkeleton_getAttachmentForSlotIndex (const spSkeleton* self, int slotIndex, const char* attachmentName) {
	if (slotIndex == -1) return 0;
	if (self->skin) {
		spAttachment *attachment = spSkin_getAttachment(self->skin, slotIndex, attachmentName);
		if (attachment) return attachment;
	}
	if (self->data->defaultSkin) {
		spAttachment *attachment = spSkin_getAttachment(self->data->defaultSkin, slotIndex, attachmentName);
		if (attachment) return attachment;
	}
	return 0;
}

int spSkeleton_setAttachment (spSkeleton* self, const char* slotName, const char* attachmentName) {
	int i;
	for (i = 0; i < self->slotsCount; ++i) {
		spSlot *slot = self->slots[i];
		if (strcmp(slot->data->name, slotName) == 0) {
			if (!attachmentName)
				spSlot_setAttachment(slot, 0);
			else {
				spAttachment* attachment = spSkeleton_getAttachmentForSlotIndex(self, i, attachmentName);
				if (!attachment) return 0;
				spSlot_setAttachment(slot, attachment);
			}
			return 1;
		}
	}
	return 0;
}

spIkConstraint* spSkeleton_findIkConstraint (const spSkeleton* self, const char* ikConstraintName) {
	int i;
	for (i = 0; i < self->ikConstraintsCount; ++i)
		if (strcmp(self->ikConstraints[i]->data->name, ikConstraintName) == 0) return self->ikConstraints[i];
	return 0;
}

void spSkeleton_update (spSkeleton* self, float deltaTime) {
	self->time += deltaTime;
}

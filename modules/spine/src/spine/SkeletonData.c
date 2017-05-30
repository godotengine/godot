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

#include <spine/SkeletonData.h>
#include <string.h>
#include <spine/extension.h>

spSkeletonData* spSkeletonData_create () {
	return NEW(spSkeletonData);
}

void spSkeletonData_dispose (spSkeletonData* self) {
	int i;
	for (i = 0; i < self->bonesCount; ++i)
		spBoneData_dispose(self->bones[i]);
	FREE(self->bones);

	for (i = 0; i < self->slotsCount; ++i)
		spSlotData_dispose(self->slots[i]);
	FREE(self->slots);

	for (i = 0; i < self->skinsCount; ++i)
		spSkin_dispose(self->skins[i]);
	FREE(self->skins);

	for (i = 0; i < self->eventsCount; ++i)
		spEventData_dispose(self->events[i]);
	FREE(self->events);

	for (i = 0; i < self->animationsCount; ++i)
		spAnimation_dispose(self->animations[i]);
	FREE(self->animations);

	for (i = 0; i < self->ikConstraintsCount; ++i)
		spIkConstraintData_dispose(self->ikConstraints[i]);
	FREE(self->ikConstraints);

	FREE(self->hash);
	FREE(self->version);

	FREE(self);
}

spBoneData* spSkeletonData_findBone (const spSkeletonData* self, const char* boneName) {
	int i;
	for (i = 0; i < self->bonesCount; ++i)
		if (strcmp(self->bones[i]->name, boneName) == 0) return self->bones[i];
	return 0;
}

int spSkeletonData_findBoneIndex (const spSkeletonData* self, const char* boneName) {
	int i;
	for (i = 0; i < self->bonesCount; ++i)
		if (strcmp(self->bones[i]->name, boneName) == 0) return i;
	return -1;
}

spSlotData* spSkeletonData_findSlot (const spSkeletonData* self, const char* slotName) {
	int i;
	for (i = 0; i < self->slotsCount; ++i)
		if (strcmp(self->slots[i]->name, slotName) == 0) return self->slots[i];
	return 0;
}

int spSkeletonData_findSlotIndex (const spSkeletonData* self, const char* slotName) {
	int i;
	for (i = 0; i < self->slotsCount; ++i)
		if (strcmp(self->slots[i]->name, slotName) == 0) return i;
	return -1;
}

spSkin* spSkeletonData_findSkin (const spSkeletonData* self, const char* skinName) {
	int i;
	for (i = 0; i < self->skinsCount; ++i)
		if (strcmp(self->skins[i]->name, skinName) == 0) return self->skins[i];
	return 0;
}

spEventData* spSkeletonData_findEvent (const spSkeletonData* self, const char* eventName) {
	int i;
	for (i = 0; i < self->eventsCount; ++i)
		if (strcmp(self->events[i]->name, eventName) == 0) return self->events[i];
	return 0;
}

spAnimation* spSkeletonData_findAnimation (const spSkeletonData* self, const char* animationName) {
	int i;
	for (i = 0; i < self->animationsCount; ++i)
		if (strcmp(self->animations[i]->name, animationName) == 0) return self->animations[i];
	return 0;
}

spIkConstraintData* spSkeletonData_findIkConstraint (const spSkeletonData* self, const char* ikConstraintName) {
	int i;
	for (i = 0; i < self->ikConstraintsCount; ++i)
		if (strcmp(self->ikConstraints[i]->name, ikConstraintName) == 0) return self->ikConstraints[i];
	return 0;
}

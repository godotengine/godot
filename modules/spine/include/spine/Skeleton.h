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

#ifndef SPINE_SKELETON_H_
#define SPINE_SKELETON_H_

#include <spine/SkeletonData.h>
#include <spine/Slot.h>
#include <spine/Skin.h>
#include <spine/IkConstraint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spSkeleton {
	spSkeletonData* const data;

	int bonesCount;
	spBone** bones;
	spBone* const root;

	int slotsCount;
	spSlot** slots;
	spSlot** drawOrder;

	int ikConstraintsCount;
	spIkConstraint** ikConstraints;

	spSkin* const skin;
	float r, g, b, a;
	float time;
	int/*bool*/flipX, flipY;
	float x, y;
} spSkeleton;

spSkeleton* spSkeleton_create (spSkeletonData* data);
void spSkeleton_dispose (spSkeleton* self);

/* Caches information about bones and IK constraints. Must be called if bones or IK constraints are added or removed. */
void spSkeleton_updateCache (const spSkeleton* self);
void spSkeleton_updateWorldTransform (const spSkeleton* self);

void spSkeleton_setToSetupPose (const spSkeleton* self);
void spSkeleton_setBonesToSetupPose (const spSkeleton* self);
void spSkeleton_setSlotsToSetupPose (const spSkeleton* self);

/* Returns 0 if the bone was not found. */
spBone* spSkeleton_findBone (const spSkeleton* self, const char* boneName);
/* Returns -1 if the bone was not found. */
int spSkeleton_findBoneIndex (const spSkeleton* self, const char* boneName);

/* Returns 0 if the slot was not found. */
spSlot* spSkeleton_findSlot (const spSkeleton* self, const char* slotName);
/* Returns -1 if the slot was not found. */
int spSkeleton_findSlotIndex (const spSkeleton* self, const char* slotName);

/* Sets the skin used to look up attachments before looking in the SkeletonData defaultSkin. Attachments from the new skin are
 * attached if the corresponding attachment from the old skin was attached. If there was no old skin, each slot's setup mode
 * attachment is attached from the new skin.
 * @param skin May be 0.*/
void spSkeleton_setSkin (spSkeleton* self, spSkin* skin);
/* Returns 0 if the skin was not found. See spSkeleton_setSkin.
 * @param skinName May be 0. */
int spSkeleton_setSkinByName (spSkeleton* self, const char* skinName);

/* Returns 0 if the slot or attachment was not found. */
spAttachment* spSkeleton_getAttachmentForSlotName (const spSkeleton* self, const char* slotName, const char* attachmentName);
/* Returns 0 if the slot or attachment was not found. */
spAttachment* spSkeleton_getAttachmentForSlotIndex (const spSkeleton* self, int slotIndex, const char* attachmentName);
/* Returns 0 if the slot or attachment was not found. */
int spSkeleton_setAttachment (spSkeleton* self, const char* slotName, const char* attachmentName);

/* Returns 0 if the IK constraint was not found. */
spIkConstraint* spSkeleton_findIkConstraint (const spSkeleton* self, const char* ikConstraintName);

void spSkeleton_update (spSkeleton* self, float deltaTime);

#ifdef SPINE_SHORT_NAMES
typedef spSkeleton Skeleton;
#define Skeleton_create(...) spSkeleton_create(__VA_ARGS__)
#define Skeleton_dispose(...) spSkeleton_dispose(__VA_ARGS__)
#define Skeleton_updateWorldTransform(...) spSkeleton_updateWorldTransform(__VA_ARGS__)
#define Skeleton_setToSetupPose(...) spSkeleton_setToSetupPose(__VA_ARGS__)
#define Skeleton_setBonesToSetupPose(...) spSkeleton_setBonesToSetupPose(__VA_ARGS__)
#define Skeleton_setSlotsToSetupPose(...) spSkeleton_setSlotsToSetupPose(__VA_ARGS__)
#define Skeleton_findBone(...) spSkeleton_findBone(__VA_ARGS__)
#define Skeleton_findBoneIndex(...) spSkeleton_findBoneIndex(__VA_ARGS__)
#define Skeleton_findSlot(...) spSkeleton_findSlot(__VA_ARGS__)
#define Skeleton_findSlotIndex(...) spSkeleton_findSlotIndex(__VA_ARGS__)
#define Skeleton_setSkin(...) spSkeleton_setSkin(__VA_ARGS__)
#define Skeleton_setSkinByName(...) spSkeleton_setSkinByName(__VA_ARGS__)
#define Skeleton_getAttachmentForSlotName(...) spSkeleton_getAttachmentForSlotName(__VA_ARGS__)
#define Skeleton_getAttachmentForSlotIndex(...) spSkeleton_getAttachmentForSlotIndex(__VA_ARGS__)
#define Skeleton_setAttachment(...) spSkeleton_setAttachment(__VA_ARGS__)
#define Skeleton_update(...) spSkeleton_update(__VA_ARGS__)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPINE_SKELETON_H_*/

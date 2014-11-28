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

#ifndef SPINE_SLOT_H_
#define SPINE_SLOT_H_

#include <spine/Bone.h>
#include <spine/Attachment.h>
#include <spine/SlotData.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spSlot {
	spSlotData* const data;
	spBone* const bone;
	float r, g, b, a;
	spAttachment* const attachment;

	int attachmentVerticesCapacity;
	int attachmentVerticesCount;
	float* attachmentVertices;
} spSlot;

spSlot* spSlot_create (spSlotData* data, spBone* bone);
void spSlot_dispose (spSlot* self);

/* @param attachment May be 0 to clear the attachment for the slot. */
void spSlot_setAttachment (spSlot* self, spAttachment* attachment);

void spSlot_setAttachmentTime (spSlot* self, float time);
float spSlot_getAttachmentTime (const spSlot* self);

void spSlot_setToSetupPose (spSlot* self);

#ifdef SPINE_SHORT_NAMES
typedef spSlot Slot;
#define Slot_create(...) spSlot_create(__VA_ARGS__)
#define Slot_dispose(...) spSlot_dispose(__VA_ARGS__)
#define Slot_setAttachment(...) spSlot_setAttachment(__VA_ARGS__)
#define Slot_setAttachmentTime(...) spSlot_setAttachmentTime(__VA_ARGS__)
#define Slot_getAttachmentTime(...) spSlot_getAttachmentTime(__VA_ARGS__)
#define Slot_setToSetupPose(...) spSlot_setToSetupPose(__VA_ARGS__)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPINE_SLOT_H_ */

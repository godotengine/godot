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

#include <spine/Bone.h>
#include <spine/extension.h>

static int yDown;

void spBone_setYDown (int value) {
	yDown = value;
}

spBone* spBone_create (spBoneData* data, spSkeleton* skeleton, spBone* parent) {
	spBone* self = NEW(spBone);
	CONST_CAST(spBoneData*, self->data) = data;
	CONST_CAST(spSkeleton*, self->skeleton) = skeleton;
	CONST_CAST(spBone*, self->parent) = parent;
	spBone_setToSetupPose(self);
	return self;
}

void spBone_dispose (spBone* self) {
	FREE(self);
}

void spBone_updateWorldTransform (spBone* self) {
	float radians, cosine, sine;
	if (self->parent) {
		CONST_CAST(float, self->worldX) = self->x * self->parent->m00 + self->y * self->parent->m01 + self->parent->worldX;
		CONST_CAST(float, self->worldY) = self->x * self->parent->m10 + self->y * self->parent->m11 + self->parent->worldY;
		if (self->data->inheritScale) {
			CONST_CAST(float, self->worldScaleX) = self->parent->worldScaleX * self->scaleX;
			CONST_CAST(float, self->worldScaleY) = self->parent->worldScaleY * self->scaleY;
		} else {
			CONST_CAST(float, self->worldScaleX) = self->scaleX;
			CONST_CAST(float, self->worldScaleY) = self->scaleY;
		}
		CONST_CAST(float, self->worldRotation) =
				self->data->inheritRotation ? self->parent->worldRotation + self->rotationIK : self->rotationIK;
		CONST_CAST(int, self->worldFlipX) = self->parent->worldFlipX ^ self->flipX;
		CONST_CAST(int, self->worldFlipY) = self->parent->worldFlipY ^ self->flipY;
	} else {
		int skeletonFlipX = self->skeleton->flipX, skeletonFlipY = self->skeleton->flipY;
		CONST_CAST(float, self->worldX) = self->skeleton->flipX ? -self->x : self->x;
		CONST_CAST(float, self->worldY) = self->skeleton->flipY != yDown ? -self->y : self->y;
		CONST_CAST(float, self->worldScaleX) = self->scaleX;
		CONST_CAST(float, self->worldScaleY) = self->scaleY;
		CONST_CAST(float, self->worldRotation) = self->rotationIK;
		CONST_CAST(int, self->worldFlipX) = skeletonFlipX ^ self->flipX;
		CONST_CAST(int, self->worldFlipY) = skeletonFlipY ^ self->flipY;
	}
	radians = self->worldRotation * DEG_RAD;
	cosine = COS(radians);
	sine = SIN(radians);
	if (self->worldFlipX) {
		CONST_CAST(float, self->m00) = -cosine * self->worldScaleX;
		CONST_CAST(float, self->m01) = sine * self->worldScaleY;
	} else {
		CONST_CAST(float, self->m00) = cosine * self->worldScaleX;
		CONST_CAST(float, self->m01) = -sine * self->worldScaleY;
	}
	if (self->worldFlipY != yDown) {
		CONST_CAST(float, self->m10) = -sine * self->worldScaleX;
		CONST_CAST(float, self->m11) = -cosine * self->worldScaleY;
	} else {
		CONST_CAST(float, self->m10) = sine * self->worldScaleX;
		CONST_CAST(float, self->m11) = cosine * self->worldScaleY;
	}
}

void spBone_setToSetupPose (spBone* self) {
	self->x = self->data->x;
	self->y = self->data->y;
	self->rotation = self->data->rotation;
	self->rotationIK = self->rotation;
	self->scaleX = self->data->scaleX;
	self->scaleY = self->data->scaleY;
	self->flipX = self->data->flipX;
	self->flipY = self->data->flipY;
}

void spBone_worldToLocal (spBone* self, float worldX, float worldY, float* localX, float* localY) {
	float invDet;
	float dx = worldX - self->worldX, dy = worldY - self->worldY;
	float m00 = self->m00, m11 = self->m11;
	if (self->worldFlipX != (self->worldFlipY != yDown)) {
		m00 *= -1;
		m11 *= -1;
	}
	invDet = 1 / (m00 * m11 - self->m01 * self->m10);
	*localX = (dx * m00 * invDet - dy * self->m01 * invDet);
	*localY = (dy * m11 * invDet - dx * self->m10 * invDet);
}

void spBone_localToWorld (spBone* self, float localX, float localY, float* worldX, float* worldY) {
	*worldX = localX * self->m00 + localY * self->m01 + self->worldX;
	*worldY = localX * self->m10 + localY * self->m11 + self->worldY;
}

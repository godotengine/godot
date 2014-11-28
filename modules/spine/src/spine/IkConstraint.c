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

#include <spine/IkConstraint.h>
#include <spine/Skeleton.h>
#include <spine/extension.h>

spIkConstraint* spIkConstraint_create (spIkConstraintData* data, const spSkeleton* skeleton) {
	int i;

	spIkConstraint* self = NEW(spIkConstraint);
	CONST_CAST(spIkConstraintData*, self->data) = data;
	self->bendDirection = data->bendDirection;
	self->mix = data->mix;

	self->bonesCount = self->data->bonesCount;
	self->bones = MALLOC(spBone*, self->bonesCount);
	for (i = 0; i < self->bonesCount; ++i)
		self->bones[i] = spSkeleton_findBone(skeleton, self->data->bones[i]->name);
	self->target = spSkeleton_findBone(skeleton, self->data->target->name);

	return self;
}

void spIkConstraint_dispose (spIkConstraint* self) {
	FREE(self->bones);
	FREE(self);
}

void spIkConstraint_apply (spIkConstraint* self) {
	switch (self->bonesCount) {
	case 1:
		spIkConstraint_apply1(self->bones[0], self->target->worldX, self->target->worldY, self->mix);
		break;
	case 2:
		spIkConstraint_apply2(self->bones[0], self->bones[1], self->target->worldX, self->target->worldY, self->bendDirection,
				self->mix);
		break;
	}
}

void spIkConstraint_apply1 (spBone* bone, float targetX, float targetY, float alpha) {
	float parentRotation = (!bone->data->inheritRotation || !bone->parent) ? 0 : bone->parent->worldRotation;
	float rotation = bone->rotation;
	float rotationIK = ATAN2(targetY - bone->worldY, targetX - bone->worldX) * RAD_DEG - parentRotation;
	bone->rotationIK = rotation + (rotationIK - rotation) * alpha;
}

void spIkConstraint_apply2 (spBone* parent, spBone* child, float targetX, float targetY, int bendDirection, float alpha) {
	float positionX, positionY, childX, childY, offset, len1, len2, cosDenom, cos, childAngle, adjacent, opposite, parentAngle, rotation;
	spBone* parentParent;
	float childRotation = child->rotation, parentRotation = parent->rotation;
	if (alpha == 0) {
		child->rotationIK = childRotation;
		parent->rotationIK = parentRotation;
		return;
	}
	parentParent = parent->parent;
	if (parentParent) {
		spBone_worldToLocal(parentParent, targetX, targetY, &positionX, &positionY);
		targetX = (positionX - parent->x) * parentParent->worldScaleX;
		targetY = (positionY - parent->y) * parentParent->worldScaleY;
	} else {
		targetX -= parent->x;
		targetY -= parent->y;
	}
	if (child->parent == parent) {
		positionX = child->x;
		positionY = child->y;
	} else {
		spBone_localToWorld(child->parent, child->x, child->y, &positionX, &positionY);
		spBone_worldToLocal(parent, positionX, positionY, &positionX, &positionY);
	}
	childX = positionX * parent->worldScaleX;
	childY = positionY * parent->worldScaleY;
	offset = ATAN2(childY, childX);
	len1 = SQRT(childX * childX + childY * childY);
	len2 = child->data->length * child->worldScaleX;
	/* Based on code by Ryan Juckett with permission: Copyright (c) 2008-2009 Ryan Juckett, http://www.ryanjuckett.com/ */
	cosDenom = 2 * len1 * len2;
	if (cosDenom < 0.0001f) {
		child->rotationIK = childRotation + (ATAN2(targetY, targetX) * RAD_DEG - parentRotation - childRotation) * alpha;
		return;
	}
	cos = (targetX * targetX + targetY * targetY - len1 * len1 - len2 * len2) / cosDenom;
	if (cos < -1)
		cos = -1;
	else if (cos > 1) /**/
		cos = 1;
	childAngle = ACOS(cos) * bendDirection;
	adjacent = len1 + len2 * cos;
	opposite = len2 * SIN(childAngle);
	parentAngle = ATAN2(targetY * adjacent - targetX * opposite, targetX * adjacent + targetY * opposite);
	rotation = (parentAngle - offset) * RAD_DEG - parentRotation;
	if (rotation > 180)
		rotation -= 360;
	else if (rotation < -180) /**/
		rotation += 360;
	parent->rotationIK = parentRotation + rotation * alpha;
	rotation = (childAngle + offset) * RAD_DEG - childRotation;
	if (rotation > 180)
		rotation -= 360;
	else if (rotation < -180) /**/
		rotation += 360;
	child->rotationIK = childRotation + (rotation + parent->worldRotation - child->parent->worldRotation) * alpha;
}

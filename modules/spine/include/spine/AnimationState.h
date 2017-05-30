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

#ifndef SPINE_ANIMATIONSTATE_H_
#define SPINE_ANIMATIONSTATE_H_

#include <spine/Animation.h>
#include <spine/AnimationStateData.h>
#include <spine/Event.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	SP_ANIMATION_START, SP_ANIMATION_END, SP_ANIMATION_COMPLETE, SP_ANIMATION_EVENT
} spEventType;

typedef struct spAnimationState spAnimationState;

typedef void (*spAnimationStateListener) (spAnimationState* state, int trackIndex, spEventType type, spEvent* event,
		int loopCount);

typedef struct spTrackEntry spTrackEntry;
struct spTrackEntry {
	spAnimationState* const state;
	spTrackEntry* next;
	spTrackEntry* previous;
	spAnimation* animation;
	int/*bool*/loop;
	float delay, time, lastTime, endTime, timeScale;
	spAnimationStateListener listener;
	float mixTime, mixDuration, mix;

	void* rendererObject;
};

struct spAnimationState {
	spAnimationStateData* const data;
	float timeScale;
	spAnimationStateListener listener;

	int tracksCount;
	spTrackEntry** tracks;

	void* rendererObject;
};

/* @param data May be 0 for no mixing. */
spAnimationState* spAnimationState_create (spAnimationStateData* data);
void spAnimationState_dispose (spAnimationState* self);

void spAnimationState_update (spAnimationState* self, float delta);
void spAnimationState_apply (spAnimationState* self, struct spSkeleton* skeleton);

void spAnimationState_clearTracks (spAnimationState* self);
void spAnimationState_clearTrack (spAnimationState* self, int trackIndex);

/** Set the current animation. Any queued animations are cleared. */
spTrackEntry* spAnimationState_setAnimationByName (spAnimationState* self, int trackIndex, const char* animationName,
		int/*bool*/loop);
spTrackEntry* spAnimationState_setAnimation (spAnimationState* self, int trackIndex, spAnimation* animation, int/*bool*/loop);

/** Adds an animation to be played delay seconds after the current or last queued animation, taking into account any mix
 * duration. */
spTrackEntry* spAnimationState_addAnimationByName (spAnimationState* self, int trackIndex, const char* animationName,
		int/*bool*/loop, float delay);
spTrackEntry* spAnimationState_addAnimation (spAnimationState* self, int trackIndex, spAnimation* animation, int/*bool*/loop,
		float delay);

spTrackEntry* spAnimationState_getCurrent (spAnimationState* self, int trackIndex);

#ifdef SPINE_SHORT_NAMES
typedef spEventType EventType;
#define ANIMATION_START SP_ANIMATION_START
#define ANIMATION_END SP_ANIMATION_END
#define ANIMATION_COMPLETE SP_ANIMATION_COMPLETE
#define ANIMATION_EVENT SP_ANIMATION_EVENT
typedef spAnimationStateListener AnimationStateListener;
typedef spTrackEntry TrackEntry;
typedef spAnimationState AnimationState;
#define AnimationState_create(...) spAnimationState_create(__VA_ARGS__)
#define AnimationState_dispose(...) spAnimationState_dispose(__VA_ARGS__)
#define AnimationState_update(...) spAnimationState_update(__VA_ARGS__)
#define AnimationState_apply(...) spAnimationState_apply(__VA_ARGS__)
#define AnimationState_clearTracks(...) spAnimationState_clearTracks(__VA_ARGS__)
#define AnimationState_clearTrack(...) spAnimationState_clearTrack(__VA_ARGS__)
#define AnimationState_setAnimationByName(...) spAnimationState_setAnimationByName(__VA_ARGS__)
#define AnimationState_setAnimation(...) spAnimationState_setAnimation(__VA_ARGS__)
#define AnimationState_addAnimationByName(...) spAnimationState_addAnimationByName(__VA_ARGS__)
#define AnimationState_addAnimation(...) spAnimationState_addAnimation(__VA_ARGS__)
#define AnimationState_getCurrent(...) spAnimationState_getCurrent(__VA_ARGS__)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPINE_ANIMATIONSTATE_H_ */

/*
 Implementation notes:

 - An OOP style is used where each "class" is made up of a struct and a number of functions prefixed with the struct name.

 - struct fields that are const are readonly. Either they are set in a create function and can never be changed, or they can only
 be changed by calling a function.

 - Inheritance is done using a struct field named "super" as the first field, allowing the struct to be cast to its "super class".
 This works because a pointer to a struct is guaranteed to be a pointer to the first struct field.

 - Classes intended for inheritance provide init/deinit functions which subclasses must call in their create/dispose functions.

 - Polymorphism is done by a base class providing function pointers in its init function. The public API delegates to this
 function.

 - Subclasses do not provide a dispose function, instead the base class' dispose function should be used, which will delegate to
 a dispose function pointer.

 - Classes not designed for inheritance cannot be extended because they may use an internal subclass to hide private data and don't
 expose function pointers.

 - The public API hides implementation details, such as init/deinit functions. An internal API is exposed by extension.h to allow
 classes to be extended. Internal functions begin with underscore (_).

 - OOP in C tends to lose type safety. Macros for casting are provided in extension.h to give context for why a cast is being done.

 - If SPINE_SHORT_NAMES is defined, the "sp" prefix for all class names is optional.
 */

#ifndef SPINE_EXTENSION_H_
#define SPINE_EXTENSION_H_

/* All allocation uses these. */
#define MALLOC(TYPE,COUNT) ((TYPE*)_malloc(sizeof(TYPE) * COUNT, __FILE__, __LINE__))
#define CALLOC(TYPE,COUNT) ((TYPE*)_calloc(COUNT, sizeof(TYPE), __FILE__, __LINE__))
#define NEW(TYPE) CALLOC(TYPE,1)

/* Gets the direct super class. Type safe. */
#define SUPER(VALUE) (&VALUE->super)

/* Cast to a super class. Not type safe, use with care. Prefer SUPER() where possible. */
#define SUPER_CAST(TYPE,VALUE) ((TYPE*)VALUE)

/* Cast to a sub class. Not type safe, use with care. */
#define SUB_CAST(TYPE,VALUE) ((TYPE*)VALUE)

/* Casts away const. Can be used as an lvalue. Not type safe, use with care. */
#define CONST_CAST(TYPE,VALUE) (*(TYPE*)&VALUE)

/* Gets the vtable for the specified type. Not type safe, use with care. */
#define VTABLE(TYPE,VALUE) ((_##TYPE##Vtable*)((TYPE*)VALUE)->vtable)

/* Frees memory. Can be used on const types. */
#define FREE(VALUE) _free((void*)VALUE)

/* Allocates a new char[], assigns it to TO, and copies FROM to it. Can be used on const types. */
#define MALLOC_STR(TO,FROM) strcpy(CONST_CAST(char*, TO) = (char*)MALLOC(char, strlen(FROM) + 1), FROM)

#define PI 3.1415926535897932385f
#define DEG_RAD (PI / 180)
#define RAD_DEG (180 / PI)

#ifdef __STDC_VERSION__
#define FMOD(A,B) fmodf(A, B)
#define ATAN2(A,B) atan2f(A, B)
#define SIN(A) sinf(A)
#define COS(A) cosf(A)
#define SQRT(A) sqrtf(A)
#define ACOS(A) acosf(A)
#else
#define FMOD(A,B) (float)fmod(A, B)
#define ATAN2(A,B) (float)atan2(A, B)
#define COS(A) (float)cos(A)
#define SIN(A) (float)sin(A)
#define SQRT(A) (float)sqrt(A)
#define ACOS(A) (float)acos(A)
#endif

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <spine/Skeleton.h>
#include <spine/Animation.h>
#include <spine/Atlas.h>
#include <spine/AttachmentLoader.h>
#include <spine/RegionAttachment.h>
#include <spine/MeshAttachment.h>
#include <spine/SkinnedMeshAttachment.h>
#include <spine/BoundingBoxAttachment.h>
#include <spine/AnimationState.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Functions that must be implemented:
 */

void _spAtlasPage_createTexture (spAtlasPage* self, const char* path);
void _spAtlasPage_disposeTexture (spAtlasPage* self);
char* _spUtil_readFile (const char* path, int* length);

#ifdef SPINE_SHORT_NAMES
#define _AtlasPage_createTexture(...) _spAtlasPage_createTexture(__VA_ARGS__)
#define _AtlasPage_disposeTexture(...) _spAtlasPage_disposeTexture(__VA_ARGS__)
#define _Util_readFile(...) _spUtil_readFile(__VA_ARGS__)
#endif

/*
 * Internal API available for extension:
 */

void* _malloc (size_t size, const char* file, int line);
void* _calloc (size_t num, size_t size, const char* file, int line);
void _free (void* ptr);

void _setMalloc (void* (*_malloc) (size_t size));
void _setDebugMalloc (void* (*_malloc) (size_t size, const char* file, int line));
void _setFree (void (*_free) (void* ptr));

char* _readFile (const char* path, int* length);

/**/

typedef struct _spAnimationState {
	spAnimationState super;
	spEvent** events;

	spTrackEntry* (*createTrackEntry) (spAnimationState* self);
	void (*disposeTrackEntry) (spTrackEntry* entry);
} _spAnimationState;

spTrackEntry* _spTrackEntry_create (spAnimationState* self);
void _spTrackEntry_dispose (spTrackEntry* self);

/**/

void _spAttachmentLoader_init (spAttachmentLoader* self, /**/
void (*dispose) (spAttachmentLoader* self), /**/
		spAttachment* (*newAttachment) (spAttachmentLoader* self, spSkin* skin, spAttachmentType type, const char* name,
				const char* path));
void _spAttachmentLoader_deinit (spAttachmentLoader* self);
void _spAttachmentLoader_setError (spAttachmentLoader* self, const char* error1, const char* error2);
void _spAttachmentLoader_setUnknownTypeError (spAttachmentLoader* self, spAttachmentType type);

#ifdef SPINE_SHORT_NAMES
#define _AttachmentLoader_init(...) _spAttachmentLoader_init(__VA_ARGS__)
#define _AttachmentLoader_deinit(...) _spAttachmentLoader_deinit(__VA_ARGS__)
#define _AttachmentLoader_setError(...) _spAttachmentLoader_setError(__VA_ARGS__)
#define _AttachmentLoader_setUnknownTypeError(...) _spAttachmentLoader_setUnknownTypeError(__VA_ARGS__)
#endif

/**/

void _spAttachment_init (spAttachment* self, const char* name, spAttachmentType type, /**/
void (*dispose) (spAttachment* self));
void _spAttachment_deinit (spAttachment* self);

#ifdef SPINE_SHORT_NAMES
#define _Attachment_init(...) _spAttachment_init(__VA_ARGS__)
#define _Attachment_deinit(...) _spAttachment_deinit(__VA_ARGS__)
#endif

/**/

void _spTimeline_init (spTimeline* self, spTimelineType type, /**/
void (*dispose) (spTimeline* self), /**/
		void (*apply) (const spTimeline* self, spSkeleton* skeleton, float lastTime, float time, spEvent** firedEvents,
				int* eventsCount, float alpha));
void _spTimeline_deinit (spTimeline* self);

#ifdef SPINE_SHORT_NAMES
#define _Timeline_init(...) _spTimeline_init(__VA_ARGS__)
#define _Timeline_deinit(...) _spTimeline_deinit(__VA_ARGS__)
#endif

/**/

void _spCurveTimeline_init (spCurveTimeline* self, spTimelineType type, int framesCount, /**/
void (*dispose) (spTimeline* self), /**/
		void (*apply) (const spTimeline* self, spSkeleton* skeleton, float lastTime, float time, spEvent** firedEvents,
				int* eventsCount, float alpha));
void _spCurveTimeline_deinit (spCurveTimeline* self);

#ifdef SPINE_SHORT_NAMES
#define _CurveTimeline_init(...) _spCurveTimeline_init(__VA_ARGS__)
#define _CurveTimeline_deinit(...) _spCurveTimeline_deinit(__VA_ARGS__)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPINE_EXTENSION_H_ */

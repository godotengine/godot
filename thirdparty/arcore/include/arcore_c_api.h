/*
 * Copyright 2017 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ARCORE_C_API_H_
#define ARCORE_C_API_H_

#include <stddef.h>
#include <stdint.h>

/// @defgroup concepts Concepts
/// High-Level concepts of ARCore
///
/// @section ownership Object ownership
///
/// ARCore has two categories of objects: "value types" and "reference types".
///
/// - Value types are owned by application. They are created and destroyed using
///   the @c create / @c destroy methods, and are populated by ARCore using
///   methods with @c get in the method name.
///
/// - Reference types are owned by ARCore. A reference is acquired by one of the
///   @c acquire methods.  For each call to the @c acquire method, the
///   application must call the matching @c release method. Note that even if
///   last reference is released, ARCore may continue to hold a reference to the
///   object at ARCore's discretion.
///
/// Reference types are further split into:
///
/// - Long-lived objects. These objects persist across frames, possibly for the
///   life span of the application or session. Acquire may fail if ARCore is in
///   an incorrect state, e.g. not tracking. Acquire from list always succeeds,
///   as the object already exists.
///
/// - Transient large data. These objects are usually acquired per-frame and are
///   a limited resource. The @c acquire call may fail if the resource is
///   exhausted (too many are currently held), deadline exceeded (the target of
///   the acquire was already released), or the resource is not yet available.
///
/// Note: Lists are value types (owned by application), but can hold references
/// to long-lived objects. This means that the references held by a list are not
/// released until either the list is destroyed, or is re-populated by another
/// api call.
///
/// For example, ::ArAnchorList, which is a value type, will hold references to
/// Anchors, which are long-lived objects.
///
/// @section spaces Poses and Coordinate Spaces
///
/// An @c ArPose describes an rigid transformation from one coordinate space to
/// another. As provided from all ARCore APIs, Poses always describe the
/// transformation from object's local coordinate space to the <b>world
/// coordinate space</b> (see below). That is, Poses from ARCore APIs can be
/// thought of as equivalent to OpenGL model matrices.
///
/// The transformation is defined using a quaternion rotation about the origin
/// followed by a translation.
///
/// The coordinate system is right-handed, like OpenGL conventions.
///
/// Translation units are meters.
///
/// @section worldcoordinates World Coordinate Space
///
/// As ARCore's understanding of the environment changes, it adjusts its model
/// of the world to keep things consistent. When this happens, the numerical
/// location (coordinates) of the camera and anchors can change significantly to
/// maintain appropriate relative positions of the physical locations they
/// represent.
///
/// These changes mean that every frame should be considered to be in a
/// completely unique world coordinate space. The numerical coordinates of
/// anchors and the camera should never be used outside the rendering frame
/// during which they were retrieved. If a position needs to be considered
/// beyond the scope of a single rendering frame, either an anchor should be
/// created or a position relative to a nearby existing anchor should be used.

/// @defgroup common Common Definitions
/// Shared types and constants

/// @defgroup anchor Anchor
/// Describes a fixed location and orientation in the real world.

/// @defgroup arcoreapk ArCoreApk
/// Management of the ARCore service APK

/// @defgroup augmented_image AugmentedImage
/// An image being detected and tracked by ARCore.

/// @defgroup augmented_face AugmentedFace
/// Describes a face detected by ARCore and provides methods to access
/// additional center and face region poses as well as face mesh related data.
///
/// Augmented Faces supports front-facing (selfie) camera only, and does not
/// support attaching anchors nor raycast hit testing. Calling
/// #ArTrackable_acquireNewAnchor() will return @c AR_ERROR_ILLEGAL_STATE.

/// @defgroup augmented_image_database AugmentedImageDatabase
/// Database containing a list of images to be detected and tracked by ARCore.

/// @defgroup camera Camera
/// Provides information about the camera that is used to capture images.

/// @defgroup cloud Cloud Anchors
/// The cloud state and configuration of an Anchor and the AR Session.

/// @defgroup config Configuration
/// Session configuration.

/// @defgroup cameraconfig CameraConfig
/// Camera configuration.

/// @defgroup frame Frame
/// Per-frame state.

/// @defgroup hit HitResult
/// Defines an intersection between a ray and estimated real-world geometry.

/// @defgroup image Image
/// Provides access to metadata from the camera image capture result.

/// @defgroup intrinsics Intrinsics
/// Provides information about the physical characteristics of the device
/// camera.

/// @defgroup light LightEstimate
/// Holds information about the estimated lighting of the real scene.

/// @defgroup plane Plane
/// Describes the current best knowledge of a real-world planar surface.

/// @defgroup point Point
/// Represents a point in space that ARCore is tracking.

/// @defgroup pointcloud PointCloud
/// Contains a set of observed 3D points and confidence values.

/// @defgroup pose Pose
/// Represents an immutable rigid transformation from one coordinate
/// space to another.

/// @defgroup session Session
/// Session management.

/// @defgroup trackable Trackable
/// Something that can be tracked and that Anchors can be attached to.

/// @defgroup cpp_helpers C++ helper functions

/// @addtogroup config
/// @{

/// An opaque session configuration object (@ref ownership "value type").
///
/// Create with ArConfig_create()<br>
/// Release with ArConfig_destroy()
typedef struct ArConfig_ ArConfig;

/// @}

// CameraConfig objects and list.

/// @addtogroup cameraconfig
/// @{

/// A camera config struct that contains the config supported by
/// the physical camera obtained from the low level device profiles.
/// (@ref ownership "value type").
///
/// Allocate with ArCameraConfig_create()<br>
/// Release with ArCameraConfig_destroy()
typedef struct ArCameraConfig_ ArCameraConfig;

/// A list of camera config (@ref ownership "value type").
///
/// Allocate with ArCameraConfigList_create()<br>
/// Release with ArCameraConfigList_destroy()
typedef struct ArCameraConfigList_ ArCameraConfigList;

/// @}

// Shared Camera objects definition.
// Excluded from generated docs (// vs ///) since it's a detail of the Java SDK.

// A shared camera contains methods that require sending Java objects over the
// c/c++ interface. To avoid using void* and making code clarity that the Java
// object is being just transmitted we define a new typedef.
//
typedef void *ArJavaObject;

/// @addtogroup session
/// @{

/// The ARCore session (@ref ownership "value type").
///
/// Create with ArSession_create()<br>
/// Release with ArSession_destroy()
typedef struct ArSession_ ArSession;

/// @}

/// @addtogroup pose
/// @{

/// A structured rigid transformation (@ref ownership "value type").
///
/// Allocate with ArPose_create()<br>
/// Release with ArPose_destroy()
typedef struct ArPose_ ArPose;

/// @}

// Camera.

/// @addtogroup camera
/// @{

/// The virtual and physical camera
/// (@ref ownership "reference type, long-lived").
///
/// Acquire with ArFrame_acquireCamera()<br>
/// Release with ArCamera_release()
typedef struct ArCamera_ ArCamera;

/// @}

// === Camera intrinstics types and methods ===

/// @addtogroup intrinsics
/// @{

/// The physical characteristics of a given camera.
///
/// Allocate with ArCameraIntrinsics_create()<br>
/// Populate with ArCamera_getIntrinsics()<br>
/// Release with ArCameraIntrinsics_destroy()
typedef struct ArCameraIntrinsics_ ArCameraIntrinsics;

/// @}

// Frame and frame objects.

/// @addtogroup frame
/// @{

/// The world state resulting from an update (@ref ownership "value type").
///
/// Allocate with ArFrame_create()<br>
/// Populate with ArSession_update()<br>
/// Release with ArFrame_destroy()
typedef struct ArFrame_ ArFrame;

/// @}

// LightEstimate.

/// @addtogroup light
/// @{

/// An estimate of the real-world lighting (@ref ownership "value type").
///
/// Allocate with ArLightEstimate_create()<br>
/// Populate with ArFrame_getLightEstimate()<br>
/// Release with ArLightEstimate_destroy()
typedef struct ArLightEstimate_ ArLightEstimate;

/// @}

// PointCloud.

/// @addtogroup pointcloud
/// @{

/// A cloud of tracked 3D visual feature points
/// (@ref ownership "reference type, large data").
///
/// Acquire with ArFrame_acquirePointCloud()<br>
/// Release with ArPointCloud_release()
typedef struct ArPointCloud_ ArPointCloud;

/// @}

// ImageMetadata.

/// @addtogroup image
/// @{

/// Camera capture metadata (@ref ownership "reference type, large data").
///
/// Acquire with ArFrame_acquireImageMetadata()<br>
/// Release with ArImageMetadata_release()
typedef struct ArImageMetadata_ ArImageMetadata;

/// Accessing CPU image from the camera
/// (@ref ownership "reference type, large data").
///
/// Acquire with ArFrame_acquireCameraImage()<br>
/// Convert to NDK AImage with ArImage_getNdkImage()<br>
/// Release with ArImage_release().
typedef struct ArImage_ ArImage;

/// Forward declaring the AImage struct from Android NDK, which is used
/// in ArImage_getNdkImage().
typedef struct AImage AImage;
/// @}

// Trackables.

/// @addtogroup trackable
/// @{

/// Trackable base type (@ref ownership "reference type, long-lived").
typedef struct ArTrackable_ ArTrackable;

/// A list of ArTrackables (@ref ownership "value type").
///
/// Allocate with ArTrackableList_create()<br>
/// Release with ArTrackableList_destroy()
typedef struct ArTrackableList_ ArTrackableList;

/// @}

// Plane

/// @addtogroup plane
/// @{

/// A detected planar surface (@ref ownership "reference type, long-lived").
///
/// Trackable type: #AR_TRACKABLE_PLANE <br>
/// Release with: ArTrackable_release()
typedef struct ArPlane_ ArPlane;

/// @}

// Point

/// @addtogroup point
/// @{

/// An arbitrary point in space (@ref ownership "reference type, long-lived").
///
/// Trackable type: #AR_TRACKABLE_POINT <br>
/// Release with: ArTrackable_release()
typedef struct ArPoint_ ArPoint;

/// @}

// Augmented Image

/// @addtogroup augmented_image
/// @{

/// An image that has been detected and tracked (@ref ownership "reference type,
/// long-lived").
///
/// Trackable type: #AR_TRACKABLE_AUGMENTED_IMAGE <br>
/// Release with: ArTrackable_release()
typedef struct ArAugmentedImage_ ArAugmentedImage;

/// @}

// Augmented Faces

/// @addtogroup augmented_face
/// @{

/// A detected face trackable (@ref ownership "reference type, long-lived").
///
/// Trackable type: #AR_TRACKABLE_FACE <br>
/// Release with: ArTrackable_release()
typedef struct ArAugmentedFace_ ArAugmentedFace;

/// @}

// Augmented Image Database
/// @addtogroup augmented_image_database
/// @{

/// A database of images to be detected and tracked by ARCore (@ref ownership
/// "value type").
///
/// An image database supports up to 1000 images. A database can be generated by
/// the `arcoreimg` command-line database generation tool provided in the SDK,
/// or dynamically created at runtime by adding individual images.
///
/// Only one image database can be active in a session. Any images in the
/// currently active image database that have a TRACKING/PAUSED state will
/// immediately be set to the STOPPED state if a different or null image
/// database is made active in the current session Config.
///
/// Create with ArAugmentedImageDatabase_create() or
/// ArAugmentedImageDatabase_deserialize()<br>
/// Release with: ArAugmentedImageDatabase_destroy()
typedef struct ArAugmentedImageDatabase_ ArAugmentedImageDatabase;

/// @}

// Anchors.

/// @addtogroup anchor
/// @{

/// A position in space attached to a trackable
/// (@ref ownership "reference type, long-lived").
///
/// To create a new anchor call ArSession_acquireNewAnchor() or
///     ArHitResult_acquireNewAnchor().<br>
/// To have ARCore stop tracking the anchor, call ArAnchor_detach().<br>
/// To release the memory associated with this anchor reference, call
/// ArAnchor_release(). Note that, this will not cause ARCore to stop tracking
/// the anchor. Other references to the same anchor acquired through
/// ArAnchorList_acquireItem() are unaffected.
typedef struct ArAnchor_ ArAnchor;

/// A list of anchors (@ref ownership "value type").
///
/// Allocate with ArAnchorList_create()<br>
/// Release with ArAnchorList_destroy()
typedef struct ArAnchorList_ ArAnchorList;

/// @}

// Hit result functionality.

/// @addtogroup hit
/// @{

/// A single trackable hit (@ref ownership "value type").
///
/// Allocate with ArHitResult_create()<br>
/// Populate with ArHitResultList_getItem()<br>
/// Release with ArHitResult_destroy()
typedef struct ArHitResult_ ArHitResult;

/// A list of hit test results (@ref ownership "value type").
///
/// Allocate with ArHitResultList_create()<br>
/// Release with ArHitResultList_destroy()<br>
typedef struct ArHitResultList_ ArHitResultList;

/// @}

/// @cond EXCLUDE_FROM_DOXYGEN

// Forward declaring the ACameraMetadata struct from Android NDK, which is used
// in ArImageMetadata_getNdkCameraMetadata
typedef struct ACameraMetadata ACameraMetadata;

/// @endcond

/// @addtogroup cpp_helpers
/// @{
/// These methods expose allowable type conversions as C++ helper functions.
/// This avoids having to explicitly @c reinterpret_cast in most cases.
///
/// Note: These methods only change the type of a pointer - they do not change
/// the reference count of the referenced objects.
///
/// Note: There is no runtime checking that casts are correct. Call @ref
/// ArTrackable_getType() beforehand to figure out the correct cast.

#ifdef __cplusplus
/// Upcasts to ArTrackable
inline ArTrackable *ArAsTrackable(ArPlane *plane) {
  return reinterpret_cast<ArTrackable *>(plane);
}

/// Upcasts to ArTrackable
inline ArTrackable *ArAsTrackable(ArPoint *point) {
  return reinterpret_cast<ArTrackable *>(point);
}

/// Upcasts to ArTrackable
inline ArTrackable *ArAsTrackable(ArAugmentedImage *augmented_image) {
  return reinterpret_cast<ArTrackable *>(augmented_image);
}

/// Downcasts to ArPlane.
inline ArPlane *ArAsPlane(ArTrackable *trackable) {
  return reinterpret_cast<ArPlane *>(trackable);
}

/// Downcasts to ArPoint.
inline ArPoint *ArAsPoint(ArTrackable *trackable) {
  return reinterpret_cast<ArPoint *>(trackable);
}

/// Downcasts to ArAugmentedImage.
inline ArAugmentedImage *ArAsAugmentedImage(ArTrackable *trackable) {
  return reinterpret_cast<ArAugmentedImage *>(trackable);
}

/// Upcasts to ArTrackable
inline ArTrackable *ArAsTrackable(ArAugmentedFace *face) {
  return reinterpret_cast<ArTrackable *>(face);
}

/// Downcasts to ArAugmentedFace
inline ArAugmentedFace *ArAsFace(ArTrackable *trackable) {
  return reinterpret_cast<ArAugmentedFace *>(trackable);
}
#endif  // __cplusplus
/// @}

// If compiling for C++11, use the 'enum underlying type' feature to enforce
// size for ABI compatibility. In pre-C++11, use int32_t for fixed size.
#if __cplusplus >= 201100
#define AR_DEFINE_ENUM(_type) enum _type : int32_t
#else
#define AR_DEFINE_ENUM(_type) \
  typedef int32_t _type;      \
  enum
#endif

#if defined(__GNUC__) && !defined(AR_DEPRECATED_SUPPRESS)
#define AR_DEPRECATED(_deprecation_string) \
  __attribute__((deprecated(_deprecation_string)));
#else
#define AR_DEPRECATED(_deprecation_string)
#endif

/// @ingroup trackable
/// Object types for heterogeneous query/update lists.
AR_DEFINE_ENUM(ArTrackableType){
    /// The base Trackable type. Can be passed to ArSession_getAllTrackables()
    /// and ArFrame_getUpdatedTrackables() as the @c filter_type to get
    /// all/updated Trackables of all types.
    AR_TRACKABLE_BASE_TRACKABLE = 0x41520100,

    /// The ::ArPlane subtype of Trackable.
    AR_TRACKABLE_PLANE = 0x41520101,

    /// The ::ArPoint subtype of Trackable.
    AR_TRACKABLE_POINT = 0x41520102,

    /// The ::ArAugmentedImage subtype of Trackable.
    AR_TRACKABLE_AUGMENTED_IMAGE = 0x41520104,

    /// Trackable type for faces.
    AR_TRACKABLE_FACE = 0x41520105,

    /// An invalid Trackable type.
    AR_TRACKABLE_NOT_VALID = 0};

/// @ingroup session
/// Feature names for use with ArSession_createWithFeatures()
///
/// All currently defined features are mutually compatible.
AR_DEFINE_ENUM(ArSessionFeature){
    /// Indicates the end of a features list.  This must be the last entry in
    /// the
    /// array passed to ArSession_createWithFeatures().
    AR_SESSION_FEATURE_END_OF_LIST = 0,

    /// Use the front-facing (selfie) camera. When the front camera is selected,
    /// ARCore's behavior changes in the following ways:
    ///
    /// - The display will be mirrored. Specifically,
    ///   ArCamera_getProjectionMatrix() will include a horizontal flip in the
    ///   generated projection matrix and APIs that reason about things in
    ///   screen
    ///   space such as ArFrame_transformCoordinates2d() will mirror screen
    ///   coordinates. Open GL apps  should consider using \c glFrontFace to
    ///   render mirrored assets without changing their winding direction.
    /// - ArCamera_getTrackingState() will always output
    ///   #AR_TRACKING_STATE_PAUSED.
    /// - ArFrame_hitTest() will always output an empty list.
    /// - ArCamera_getDisplayOrientedPose() will always output an identity pose.
    /// - ArSession_acquireNewAnchor() will always return
    /// #AR_ERROR_NOT_TRACKING.
    /// - Planes will never be detected.
    /// - ArSession_configure() will fail if the supplied configuration requests
    ///   Cloud Anchors or Augmented Images.
    AR_SESSION_FEATURE_FRONT_CAMERA = 1,
};

/// @ingroup common
/// Return code indicating success or failure of a method.
AR_DEFINE_ENUM(ArStatus){
    /// The operation was successful.
    AR_SUCCESS = 0,

    /// One of the arguments was invalid, either null or not appropriate for the
    /// operation requested.
    AR_ERROR_INVALID_ARGUMENT = -1,

    /// An internal error occurred that the application should not attempt to
    /// recover from.
    AR_ERROR_FATAL = -2,

    /// An operation was attempted that requires the session be running, but the
    /// session was paused.
    AR_ERROR_SESSION_PAUSED = -3,

    /// An operation was attempted that requires the session be paused, but the
    /// session was running.
    AR_ERROR_SESSION_NOT_PAUSED = -4,

    /// An operation was attempted that the session be in the TRACKING state,
    /// but the session was not.
    AR_ERROR_NOT_TRACKING = -5,

    /// A texture name was not set by calling ArSession_setCameraTextureName()
    /// before the first call to ArSession_update()
    AR_ERROR_TEXTURE_NOT_SET = -6,

    /// An operation required GL context but one was not available.
    AR_ERROR_MISSING_GL_CONTEXT = -7,

    /// The configuration supplied to ArSession_configure() was unsupported.
    /// To avoid this error, ensure that Session_checkSupported() returns true.
    AR_ERROR_UNSUPPORTED_CONFIGURATION = -8,

    /// The android camera permission has not been granted prior to calling
    /// ArSession_resume()
    AR_ERROR_CAMERA_PERMISSION_NOT_GRANTED = -9,

    /// Acquire failed because the object being acquired is already released.
    /// For example, this happens if the application holds an ::ArFrame beyond
    /// the next call to ArSession_update(), and then tries to acquire its point
    /// cloud.
    AR_ERROR_DEADLINE_EXCEEDED = -10,

    /// There are no available resources to complete the operation.  In cases of
    /// @c acquire methods returning this error, This can be avoided by
    /// releasing previously acquired objects before acquiring new ones.
    AR_ERROR_RESOURCE_EXHAUSTED = -11,

    /// Acquire failed because the data isn't available yet for the current
    /// frame. For example, acquire the image metadata may fail with this error
    /// because the camera hasn't fully started.
    AR_ERROR_NOT_YET_AVAILABLE = -12,

    /// The android camera has been reallocated to a higher priority app or is
    /// otherwise unavailable.
    AR_ERROR_CAMERA_NOT_AVAILABLE = -13,

    /// The host/resolve function call failed because the Session is not
    /// configured for cloud anchors.
    AR_ERROR_CLOUD_ANCHORS_NOT_CONFIGURED = -14,

    /// ArSession_configure() failed because the specified configuration
    /// required the Android INTERNET permission, which the application did not
    /// have.
    AR_ERROR_INTERNET_PERMISSION_NOT_GRANTED = -15,

    /// HostCloudAnchor() failed because the anchor is not a type of anchor that
    /// is currently supported for hosting.
    AR_ERROR_ANCHOR_NOT_SUPPORTED_FOR_HOSTING = -16,

    /// An image with insufficient quality (e.g. too few features) was attempted
    /// to be added to the image database.
    AR_ERROR_IMAGE_INSUFFICIENT_QUALITY = -17,

    /// The data passed in for this operation was not in a valid format.
    AR_ERROR_DATA_INVALID_FORMAT = -18,

    /// The data passed in for this operation is not supported by this version
    /// of the SDK.
    AR_ERROR_DATA_UNSUPPORTED_VERSION = -19,

    /// A function has been invoked at an illegal or inappropriate time. A
    /// message will be printed to logcat with additional details for the
    /// developer.  For example, ArSession_resume() will return this status if
    /// the camera configuration was changed and there are any unreleased
    /// images.
    AR_ERROR_ILLEGAL_STATE = -20,

    /// The ARCore APK is not installed on this device.
    AR_UNAVAILABLE_ARCORE_NOT_INSTALLED = -100,

    /// The device is not currently compatible with ARCore.
    AR_UNAVAILABLE_DEVICE_NOT_COMPATIBLE = -101,

    /// The ARCore APK currently installed on device is too old and needs to be
    /// updated.
    AR_UNAVAILABLE_APK_TOO_OLD = -103,

    /// The ARCore APK currently installed no longer supports the ARCore SDK
    /// that the application was built with.
    AR_UNAVAILABLE_SDK_TOO_OLD = -104,

    /// The user declined installation of the ARCore APK during this run of the
    /// application and the current request was not marked as user-initiated.
    AR_UNAVAILABLE_USER_DECLINED_INSTALLATION = -105};

/// @ingroup common
/// Describes the tracking state of a @c Trackable, an ::ArAnchor or the
/// ::ArCamera.
AR_DEFINE_ENUM(ArTrackingState){
    /// The object is currently tracked and its pose is current.
    AR_TRACKING_STATE_TRACKING = 0,

    /// ARCore has paused tracking this object, but may resume tracking it in
    /// the future. This can happen if device tracking is lost, if the user
    /// enters a new space, or if the Session is currently paused. When in this
    /// state, the positional properties of the object may be wildly inaccurate
    /// and should not be used.
    AR_TRACKING_STATE_PAUSED = 1,

    /// ARCore has stopped tracking this Trackable and will never resume
    /// tracking it.
    AR_TRACKING_STATE_STOPPED = 2};

/// Describes possible tracking failure reasons of a @c ::ArCamera.
AR_DEFINE_ENUM(ArTrackingFailureReason){
    /// Indicates expected motion tracking behavior. Always returned when
    /// ArCamera_getTrackingState() is #AR_TRACKING_STATE_TRACKING. When
    /// ArCamera_getTrackingState() is #AR_TRACKING_STATE_PAUSED, indicates that
    /// the session is initializing normally.
    AR_TRACKING_FAILURE_REASON_NONE = 0,
    /// Motion tracking lost due to bad internal state. No specific user action
    /// is likely to resolve this issue.
    AR_TRACKING_FAILURE_REASON_BAD_STATE = 1,
    /// Motion tracking lost due to poor lighting conditions. Ask the user to
    /// move to a more brightly lit area.
    AR_TRACKING_FAILURE_REASON_INSUFFICIENT_LIGHT = 2,
    /// Motion tracking lost due to excessive motion. Ask the user to move the
    /// device more slowly.
    AR_TRACKING_FAILURE_REASON_EXCESSIVE_MOTION = 3,
    /// Motion tracking lost due to insufficient visual features. Ask the user
    /// to move to a different area and to avoid blank walls and surfaces
    /// without detail.
    AR_TRACKING_FAILURE_REASON_INSUFFICIENT_FEATURES = 4};

/// @ingroup cloud
/// Describes the current cloud state of an @c Anchor.
AR_DEFINE_ENUM(ArCloudAnchorState){
    /// The anchor is purely local. It has never been hosted using
    /// hostCloudAnchor, and has not been acquired using acquireCloudAnchor.
    AR_CLOUD_ANCHOR_STATE_NONE = 0,

    /// A hosting/resolving task for the anchor is in progress. Once the task
    /// completes in the background, the anchor will get a new cloud state after
    /// the next update() call.
    AR_CLOUD_ANCHOR_STATE_TASK_IN_PROGRESS = 1,

    /// A hosting/resolving task for this anchor completed successfully.
    AR_CLOUD_ANCHOR_STATE_SUCCESS = 2,

    /// A hosting/resolving task for this anchor finished with an internal
    /// error. The app should not attempt to recover from this error.
    AR_CLOUD_ANCHOR_STATE_ERROR_INTERNAL = -1,

    /// The app cannot communicate with the ARCore Cloud because of an invalid
    /// or unauthorized API key in the manifest, or because there was no API key
    /// present in the manifest.
    AR_CLOUD_ANCHOR_STATE_ERROR_NOT_AUTHORIZED = -2,

    /// The ARCore Cloud was unreachable. This can happen because of a number of
    /// reasons. The request sent to the server could have timed out with no
    /// response, there could be a bad network connection, DNS unavailability,
    /// firewall issues, or anything that could affect the device's ability to
    /// connect to the ARCore Cloud.
    AR_CLOUD_ANCHOR_STATE_ERROR_SERVICE_UNAVAILABLE = -3,

    /// The application has exhausted the request quota allotted to the given
    /// API key. The developer should request additional quota for the ARCore
    /// Cloud for their API key from the Google Developers Console.
    AR_CLOUD_ANCHOR_STATE_ERROR_RESOURCE_EXHAUSTED = -4,

    /// Hosting failed, because the server could not successfully process the
    /// dataset for the given anchor. The developer should try again after the
    /// device has gathered more data from the environment.
    AR_CLOUD_ANCHOR_STATE_ERROR_HOSTING_DATASET_PROCESSING_FAILED = -5,

    /// Resolving failed, because the ARCore Cloud could not find the provided
    /// cloud anchor ID.
    AR_CLOUD_ANCHOR_STATE_ERROR_CLOUD_ID_NOT_FOUND = -6,

    /// The server could not match the visual features provided by ARCore
    /// against the localization dataset of the requested cloud anchor ID. This
    /// means that the anchor pose being requested was likely not created in the
    /// user's surroundings.
    AR_CLOUD_ANCHOR_STATE_ERROR_RESOLVING_LOCALIZATION_NO_MATCH = -7,

    /// The anchor could not be resolved because the SDK used to host the anchor
    /// was newer than and incompatible with the version being used to acquire
    /// it.
    AR_CLOUD_ANCHOR_STATE_ERROR_RESOLVING_SDK_VERSION_TOO_OLD = -8,

    /// The anchor could not be acquired because the SDK used to host the anchor
    /// was older than and incompatible with the version being used to acquire
    /// it.
    AR_CLOUD_ANCHOR_STATE_ERROR_RESOLVING_SDK_VERSION_TOO_NEW = -9};

/// @ingroup arcoreapk
/// Describes the current state of ARCore availability on the device.
AR_DEFINE_ENUM(ArAvailability){
    /// An internal error occurred while determining ARCore availability.
    AR_AVAILABILITY_UNKNOWN_ERROR = 0,
    /// ARCore is not installed, and a query has been issued to check if ARCore
    /// is is supported.
    AR_AVAILABILITY_UNKNOWN_CHECKING = 1,
    /// ARCore is not installed, and the query to check if ARCore is supported
    /// timed out. This may be due to the device being offline.
    AR_AVAILABILITY_UNKNOWN_TIMED_OUT = 2,
    /// ARCore is not supported on this device.
    AR_AVAILABILITY_UNSUPPORTED_DEVICE_NOT_CAPABLE = 100,
    /// The device and Android version are supported, but the ARCore APK is not
    /// installed.
    AR_AVAILABILITY_SUPPORTED_NOT_INSTALLED = 201,
    /// The device and Android version are supported, and a version of the
    /// ARCore APK is installed, but that ARCore APK version is too old.
    AR_AVAILABILITY_SUPPORTED_APK_TOO_OLD = 202,
    /// ARCore is supported, installed, and available to use.
    AR_AVAILABILITY_SUPPORTED_INSTALLED = 203};

/// @ingroup arcoreapk
/// Indicates the outcome of a call to ArCoreApk_requestInstall().
AR_DEFINE_ENUM(ArInstallStatus){
    /// The requested resource is already installed.
    AR_INSTALL_STATUS_INSTALLED = 0,
    /// Installation of the resource was requested. The current activity will be
    /// paused.
    AR_INSTALL_STATUS_INSTALL_REQUESTED = 1};

/// @ingroup arcoreapk
/// Controls the behavior of the installation UI.
AR_DEFINE_ENUM(ArInstallBehavior){
    /// Hide the Cancel button during initial prompt and prevent user from
    /// exiting via tap-outside.
    ///
    /// Note: The BACK button or tapping outside of any marketplace-provided
    /// install dialog will still decline the installation.
    AR_INSTALL_BEHAVIOR_REQUIRED = 0,
    /// Include Cancel button in initial prompt and allow easily backing out
    /// after installation has been initiated.
    AR_INSTALL_BEHAVIOR_OPTIONAL = 1};

/// @ingroup arcoreapk
/// Controls the message displayed by the installation UI.
AR_DEFINE_ENUM(ArInstallUserMessageType){
    /// Display a localized message like "This application requires ARCore...".
    AR_INSTALL_USER_MESSAGE_TYPE_APPLICATION = 0,
    /// Display a localized message like "This feature requires ARCore...".
    AR_INSTALL_USER_MESSAGE_TYPE_FEATURE = 1,
    /// Application has explained why ARCore is required prior to calling
    /// ArCoreApk_requestInstall(), skip user education dialog.
    AR_INSTALL_USER_MESSAGE_TYPE_USER_ALREADY_INFORMED = 2};

/// @ingroup config
/// Select the behavior of the lighting estimation subsystem.
AR_DEFINE_ENUM(ArLightEstimationMode){
    /// Lighting estimation is disabled.
    AR_LIGHT_ESTIMATION_MODE_DISABLED = 0,
    /// Lighting estimation is enabled, generating a single-value intensity
    /// estimate.
    AR_LIGHT_ESTIMATION_MODE_AMBIENT_INTENSITY = 1};

/// @ingroup config
/// Select the behavior of the plane detection subsystem.
AR_DEFINE_ENUM(ArPlaneFindingMode){
    /// Plane detection is disabled.
    AR_PLANE_FINDING_MODE_DISABLED = 0,
    /// Detection of only horizontal planes is enabled.
    AR_PLANE_FINDING_MODE_HORIZONTAL = 1,
    /// Detection of only vertical planes is enabled.
    AR_PLANE_FINDING_MODE_VERTICAL = 2,
    /// Detection of horizontal and vertical planes is enabled.
    AR_PLANE_FINDING_MODE_HORIZONTAL_AND_VERTICAL = 3};

/// @ingroup config
/// Selects the behavior of ArSession_update().
AR_DEFINE_ENUM(ArUpdateMode){
    /// @c update() will wait until a new camera image is available, or until
    /// the built-in timeout (currently 66ms) is reached. On most
    /// devices the camera is configured to capture 30 frames per second.
    /// If the camera image does not arrive by the built-in timeout, then
    /// @c update() will return the most recent ::ArFrame object.
    AR_UPDATE_MODE_BLOCKING = 0,
    /// @c update() will return immediately without blocking. If no new camera
    /// image is available, then @c update() will return the most recent
    /// ::ArFrame object.
    AR_UPDATE_MODE_LATEST_CAMERA_IMAGE = 1};

/// @ingroup config
/// Selects the behavior of Augmented Faces subsystem.
/// Default value is AR_AUGMENTED_FACE_MODE_DISABLED.
AR_DEFINE_ENUM(ArAugmentedFaceMode){
    /// Disable augmented face mode.
    AR_AUGMENTED_FACE_MODE_DISABLED = 0,

    /// Face 3D mesh is enabled. Augmented Faces is currently only
    /// supported when using the front-facing (selfie) camera. See
    /// #AR_SESSION_FEATURE_FRONT_CAMERA for details and additional
    /// restrictions.
    AR_AUGMENTED_FACE_MODE_MESH3D = 2,
};

/// @ingroup augmented_face
/// Defines face regions for which the pose can be queried. Left and right
/// are defined relative to the person that the mesh belongs to. To retrieve the
/// center pose use #ArAugmentedFace_getCenterPose().
AR_DEFINE_ENUM(ArAugmentedFaceRegionType){
    /// The region at the tip of the nose.
    AR_AUGMENTED_FACE_REGION_NOSE_TIP = 0,
    /// The region at the detected face's left side of the forehead.
    AR_AUGMENTED_FACE_REGION_FOREHEAD_LEFT = 1,
    /// The region at the detected face's right side of the forehead.
    AR_AUGMENTED_FACE_REGION_FOREHEAD_RIGHT = 2,
};

/// @ingroup config
/// Selects the desired behavior of the camera focus subsystem. Currently, the
/// default focus mode is AR_FOCUS_MODE_FIXED, but this default might change in
/// the future. Note, on devices where ARCore does not support Auto Focus due to
/// the use of a fixed focus camera, setting AR_FOCUS_MODE_AUTO will be ignored.
/// See the ARCore Supported Devices
/// (https://developers.google.com/ar/discover/supported-devices) page for a
/// list of affected devices.
///
/// For optimal AR tracking performance, use the focus mode provided by the
/// default session config. While capturing pictures or video, use
/// AR_FOCUS_MODE_AUTO. For optimal AR tracking, revert to the default focus
/// mode once auto focus behavior is no longer needed. If your app requires
/// fixed focus camera, call ArConfig_setFocusMode(…, …, AR_FOCUS_MODE_FIXED)
/// before enabling the AR session. This will ensure that your app always uses
/// fixed focus, even if the default camera config focus mode changes in a
/// future release.
AR_DEFINE_ENUM(ArFocusMode){/// Focus is fixed.
                            AR_FOCUS_MODE_FIXED = 0,
                            /// Auto-focus is enabled.
                            AR_FOCUS_MODE_AUTO = 1};

/// Describes the direction a camera is facing relative to the device.  Used by
/// ArCameraConfig_getFacingDirection().
AR_DEFINE_ENUM(ArCameraConfigFacingDirection){
    /// Camera looks out the back of the device (away from the user).
    AR_CAMERA_CONFIG_FACING_DIRECTION_BACK = 0,
    /// Camera looks out the front of the device (towards the user).  To create
    /// a session using the front-facing (selfie) camera, include
    /// #AR_SESSION_FEATURE_FRONT_CAMERA in the feature list passed to
    /// ArSession_createWithFeatures().
    AR_CAMERA_CONFIG_FACING_DIRECTION_FRONT = 1};

/// @ingroup plane
/// Simple summary of the normal vector of a plane, for filtering purposes.
AR_DEFINE_ENUM(ArPlaneType){
    /// A horizontal plane facing upward (for example a floor or tabletop).
    AR_PLANE_HORIZONTAL_UPWARD_FACING = 0,
    /// A horizontal plane facing downward (for example a ceiling).
    AR_PLANE_HORIZONTAL_DOWNWARD_FACING = 1,
    /// A vertical plane (for example a wall).
    AR_PLANE_VERTICAL = 2};

/// @ingroup light
/// Tracks the validity of a light estimate.
AR_DEFINE_ENUM(ArLightEstimateState){
    /// The light estimate is not valid this frame and should not be used for
    /// rendering.
    AR_LIGHT_ESTIMATE_STATE_NOT_VALID = 0,
    /// The light estimate is valid this frame.
    AR_LIGHT_ESTIMATE_STATE_VALID = 1};

/// @ingroup point
/// Indicates the orientation mode of the ::ArPoint.
AR_DEFINE_ENUM(ArPointOrientationMode){
    /// The orientation of the ::ArPoint is initialized to identity but may
    /// adjust slightly over time.
    AR_POINT_ORIENTATION_INITIALIZED_TO_IDENTITY = 0,
    /// The orientation of the ::ArPoint will follow the behavior described in
    /// ArHitResult_getHitPose().
    AR_POINT_ORIENTATION_ESTIMATED_SURFACE_NORMAL = 1};

/// @ingroup cloud
/// Indicates the cloud configuration of the ::ArSession.
AR_DEFINE_ENUM(ArCloudAnchorMode){
    /// Anchor Hosting is disabled. This is the value set in the default
    /// ::ArConfig.
    AR_CLOUD_ANCHOR_MODE_DISABLED = 0,
    /// Anchor Hosting is enabled. Setting this value and calling @c configure()
    /// will require that the application have the Android INTERNET permission.
    AR_CLOUD_ANCHOR_MODE_ENABLED = 1};

/// @ingroup frame
/// 2d coordinate systems supported by ARCore.
AR_DEFINE_ENUM(ArCoordinates2dType){
    /// GPU texture, (x,y) in pixels.
    AR_COORDINATES_2D_TEXTURE_TEXELS = 0,
    /// GPU texture coordinates, (s,t) normalized to [0.0f, 1.0f] range.
    AR_COORDINATES_2D_TEXTURE_NORMALIZED = 1,
    /// CPU image, (x,y) in pixels.
    AR_COORDINATES_2D_IMAGE_PIXELS = 2,
    /// CPU image, (x,y) normalized to [0.0f, 1.0f] range.
    AR_COORDINATES_2D_IMAGE_NORMALIZED = 3,
    /// OpenGL Normalized Device Coordinates, display-rotated,
    /// (x,y) normalized to [-1.0f, 1.0f] range.
    AR_COORDINATES_2D_OPENGL_NORMALIZED_DEVICE_COORDINATES = 6,
    /// Android view, display-rotated, (x,y) in pixels.
    AR_COORDINATES_2D_VIEW = 7,
    /// Android view, display-rotated, (x,y) normalized to [0.0f, 1.0f] range.
    AR_COORDINATES_2D_VIEW_NORMALIZED = 8,

};

#ifdef __cplusplus
extern "C" {
#endif

// Note: destroy methods do not take ArSession* to allow late destruction in
// finalizers of garbage-collected languages such as Java.

/// @addtogroup arcoreapk
/// @{

/// Determines if ARCore is supported on this device. This may initiate a query
/// with a remote service to determine if the device is compatible, in which
/// case it will return immediately with @c out_availability set to
/// #AR_AVAILABILITY_UNKNOWN_CHECKING.
///
/// For ARCore-required apps (as indicated by the <a
/// href="https://developers.google.com/ar/develop/c/enable-arcore#ar_required">manifest
/// meta-data</a>) this method will assume device compatibility and will always
/// immediately return one of #AR_AVAILABILITY_SUPPORTED_INSTALLED,
/// #AR_AVAILABILITY_SUPPORTED_APK_TOO_OLD, or
/// #AR_AVAILABILITY_SUPPORTED_NOT_INSTALLED.
///
/// Note: A result #AR_AVAILABILITY_SUPPORTED_INSTALLED only indicates presence
/// of a suitably versioned ARCore APK. Session creation may still fail if the
/// ARCore APK has been sideloaded onto an incompatible device.
///
/// May be called prior to ArSession_create().
///
/// @param[in] env The application's @c JNIEnv object
/// @param[in] application_context A @c jobject referencing the application's
///     Android @c Context.
/// @param[out] out_availability A pointer to an ArAvailability to receive
///     the result.
void ArCoreApk_checkAvailability(void *env,
                                 void *application_context,
                                 ArAvailability *out_availability);

/// Initiates installation of ARCore if needed. When your apllication launches
/// or enters an AR mode, it should call this method with @c
/// user_requested_install = 1.
///
/// If ARCore is installed and compatible, this function will set @c
/// out_install_status to #AR_INSTALL_STATUS_INSTALLED.
///
/// If ARCore is not currently installed or the installed version not
/// compatible, the function will set @c out_install_status to
/// #AR_INSTALL_STATUS_INSTALL_REQUESTED and return immediately. Your current
/// activity will then pause while the user is offered the opportunity to
/// install it.
///
/// When your activity resumes, you should call this method again, this time
/// with @c user_requested_install = 0. This will either set
/// @c out_install_status to #AR_INSTALL_STATUS_INSTALLED or return an error
/// code indicating the reason that installation could not be completed.
///
/// ARCore-optional applications must ensure that ArCoreApk_checkAvailability()
/// returns one of the <tt>AR_AVAILABILITY_SUPPORTED_...</tt> values before
/// calling this method.
///
/// See <A
/// href="https://github.com/google-ar/arcore-android-sdk/tree/master/samples">
/// our sample code</A> for an example of how an ARCore-required application
/// should use this function.
///
/// May be called prior to ArSession_create().
///
/// For more control over the message displayed and ease of exiting the process,
/// see ArCoreApk_requestInstallCustom().
///
/// <b>Caution:</b> The value of <tt>*out_install_status</tt> should only be
/// considered when #AR_SUCCESS is returned.  Otherwise this value must be
/// ignored.
///
/// @param[in] env The application's @c JNIEnv object
/// @param[in] application_activity A @c jobject referencing the application's
///     current Android @c Activity.
/// @param[in] user_requested_install if set, override the previous installation
///     failure message and always show the installation interface.
/// @param[out] out_install_status A pointer to an ArInstallStatus to receive
///     the resulting install status, if successful.  Note: this value is only
///     valid with the return value is #AR_SUCCESS.
/// @return #AR_SUCCESS, or any of:
/// - #AR_ERROR_FATAL if an error occurs while checking for or requesting
///     installation
/// - #AR_UNAVAILABLE_DEVICE_NOT_COMPATIBLE if ARCore is not supported
///     on this device.
/// - #AR_UNAVAILABLE_USER_DECLINED_INSTALLATION if the user previously declined
///     installation.
ArStatus ArCoreApk_requestInstall(void *env,
                                  void *application_activity,
                                  int32_t user_requested_install,
                                  ArInstallStatus *out_install_status);

/// Initiates installation of ARCore if required, with configurable behavior.
///
/// This is a more flexible version of ArCoreApk_requestInstall() allowing the
/// application control over the initial informational dialog and ease of
/// exiting or cancelling the installation.
///
/// See ArCoreApk_requestInstall() for details of use and behavior.
///
/// May be called prior to ArSession_create().
///
/// @param[in] env The application's @c JNIEnv object
/// @param[in] application_activity A @c jobject referencing the application's
///     current Android @c Activity.
/// @param[in] user_requested_install if set, override the previous installation
///     failure message and always show the installation interface.
/// @param[in] install_behavior controls the presence of the cancel button at
///     the user education screen and if tapping outside the education screen or
///     install-in-progress screen causes them to dismiss.
/// @param[in] message_type controls the text of the of message displayed
///     before showing the install prompt, or disables display of this message.
/// @param[out] out_install_status A pointer to an ArInstallStatus to receive
///     the resulting install status, if successful.  Note: this value is only
///     valid with the return value is #AR_SUCCESS.
/// @return #AR_SUCCESS, or any of:
/// - #AR_ERROR_FATAL if an error occurs while checking for or requesting
///     installation
/// - #AR_UNAVAILABLE_DEVICE_NOT_COMPATIBLE if ARCore is not supported
///     on this device.
/// - #AR_UNAVAILABLE_USER_DECLINED_INSTALLATION if the user previously declined
///     installation.
ArStatus ArCoreApk_requestInstallCustom(void *env,
                                        void *application_activity,
                                        int32_t user_requested_install,
                                        ArInstallBehavior install_behavior,
                                        ArInstallUserMessageType message_type,
                                        ArInstallStatus *out_install_status);

/// @}
/// @addtogroup session
/// @{

/// Creates a new ARCore session.  Prior to calling this function, your app must
/// check that ARCore is installed by verifying that either:
///
/// - ArCoreApk_requestInstall() or ArCoreApk_requestInstallCustom() returns
///   #AR_INSTALL_STATUS_INSTALLED, or
/// - ArCoreApk_checkAvailability() returns
///   #AR_AVAILABILITY_SUPPORTED_INSTALLED.
///
/// This check must be performed prior to creating an ArSession, otherwise
/// ArSession creation will fail, and subsequent installation or upgrade of
/// ARCore will require an app restart and might cause Android to kill your app.
///
/// @param[in]  env                 The application's @c JNIEnv object
/// @param[in]  application_context A @c jobject referencing the application's
///     Android @c Context
/// @param[out] out_session_pointer A pointer to an @c ArSession* to receive
///     the address of the newly allocated session.
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_FATAL if an internal error occurred while creating the session.
///   `adb logcat` may contain useful information.
/// - #AR_ERROR_CAMERA_PERMISSION_NOT_GRANTED if your app does not have the
///   [CAMERA](https://developer.android.com/reference/android/Manifest.permission.html#CAMERA)
///   permission.
/// - #AR_UNAVAILABLE_ARCORE_NOT_INSTALLED if the ARCore APK is not present.
///   This can be prevented by the installation check described above.
/// - #AR_UNAVAILABLE_DEVICE_NOT_COMPATIBLE if the device is not compatible with
///   ARCore.  If encountered after completing the installation check, this
///   usually indicates a user has side-loaded ARCore onto an incompatible
///   device.
/// - #AR_UNAVAILABLE_APK_TOO_OLD if the installed ARCore APK is too old for the
///   ARCore SDK with which this application was built. This can be prevented by
///   the installation check described above.
/// - #AR_UNAVAILABLE_SDK_TOO_OLD if the ARCore SDK that this app was built with
///   is too old and no longer supported by the installed ARCore APK.
ArStatus ArSession_create(void *env,
                          void *application_context,
                          ArSession **out_session_pointer);

/// Creates a new ARCore session requesting additional features.  Prior to
/// calling this function, your app must check that ARCore is installed by
/// verifying that either:
///
/// - ArCoreApk_requestInstall() or ArCoreApk_requestInstallCustom() returns
///   #AR_INSTALL_STATUS_INSTALLED, or
/// - ArCoreApk_checkAvailability() returns
///   #AR_AVAILABILITY_SUPPORTED_INSTALLED.
///
/// This check must be performed prior to creating an ArSession, otherwise
/// ArSession creation will fail, and subsequent installation or upgrade of
/// ARCore will require an app restart and might cause Android to kill your app.
///
/// @param[in]  env                 The application's @c JNIEnv object
/// @param[in]  application_context A @c jobject referencing the application's
///     Android @c Context
/// @param[in]  features            The list of requested features, terminated
///     by with #AR_SESSION_FEATURE_END_OF_LIST.
/// @param[out] out_session_pointer A pointer to an @c ArSession* to receive
///     the address of the newly allocated session.
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_FATAL if an internal error occurred while creating the session.
///   `adb logcat` may contain useful information.
/// - #AR_ERROR_CAMERA_PERMISSION_NOT_GRANTED if your app does not have the
///   [CAMERA](https://developer.android.com/reference/android/Manifest.permission.html#CAMERA)
///   permission.
/// - #AR_ERROR_INVALID_ARGUMENT if the requested features are mutually
///   incompatible.  See #ArSessionFeature for details.
/// - #AR_UNAVAILABLE_ARCORE_NOT_INSTALLED if the ARCore APK is not present.
///   This can be prevented by the installation check described above.
/// - #AR_UNAVAILABLE_DEVICE_NOT_COMPATIBLE if the device is not compatible with
///   ARCore.  If encountered after completing the installation check, this
///   usually indicates a user has side-loaded ARCore onto an incompatible
///   device.
/// - #AR_UNAVAILABLE_APK_TOO_OLD if the installed ARCore APK is too old for the
///   ARCore SDK with which this application was built. This can be prevented by
///   the installation check described above.
/// - #AR_UNAVAILABLE_SDK_TOO_OLD if the ARCore SDK that this app was built with
///   is too old and no longer supported by the installed ARCore APK.
ArStatus ArSession_createWithFeatures(void *env,
                                      void *application_context,
                                      const ArSessionFeature *features,
                                      ArSession **out_session_pointer);

/// @}

// === ArConfig methods ===

/// @addtogroup config
/// @{

/// Creates a new configuration object and initializes it to a sensible default
/// configuration. Plane detection and lighting estimation are enabled, and
/// blocking update is selected. This configuration is guaranteed to be
/// supported on all devices that support ARCore.
void ArConfig_create(const ArSession *session, ArConfig **out_config);

/// Releases memory used by the provided configuration object.
void ArConfig_destroy(ArConfig *config);

/// Stores the currently configured lighting estimation mode into
/// @c *light_estimation_mode.
void ArConfig_getLightEstimationMode(
    const ArSession *session,
    const ArConfig *config,
    ArLightEstimationMode *light_estimation_mode);

/// Sets the lighting estimation mode that should be used. See
/// ::ArLightEstimationMode for available options.
void ArConfig_setLightEstimationMode(
    const ArSession *session,
    ArConfig *config,
    ArLightEstimationMode light_estimation_mode);

/// Stores the currently configured plane finding mode into
/// @c *plane_finding_mode.
void ArConfig_getPlaneFindingMode(const ArSession *session,
                                  const ArConfig *config,
                                  ArPlaneFindingMode *plane_finding_mode);

/// Sets the plane finding mode that should be used. See
/// ::ArPlaneFindingMode for available options.
void ArConfig_setPlaneFindingMode(const ArSession *session,
                                  ArConfig *config,
                                  ArPlaneFindingMode plane_finding_mode);

/// Stores the currently configured behavior of @ref ArSession_update() into
/// @c *update_mode.
void ArConfig_getUpdateMode(const ArSession *session,
                            const ArConfig *config,
                            ArUpdateMode *update_mode);

/// Sets the behavior of @ref ArSession_update(). See
/// ::ArUpdateMode for available options.
void ArConfig_setUpdateMode(const ArSession *session,
                            ArConfig *config,
                            ArUpdateMode update_mode);

/// Gets the current cloud anchor mode from the ::ArConfig.
void ArConfig_getCloudAnchorMode(const ArSession *session,
                                 const ArConfig *config,
                                 ArCloudAnchorMode *out_cloud_anchor_mode);

/// Sets the cloud configuration that should be used. See ::ArCloudAnchorMode
/// for available options.
void ArConfig_setCloudAnchorMode(const ArSession *session,
                                 ArConfig *config,
                                 ArCloudAnchorMode cloud_anchor_mode);

/// Sets the image database in the session configuration.
///
/// Any images in the currently active image database that have a
/// TRACKING/PAUSED state will immediately be set to the STOPPED state if a
/// different or null image database is set.
///
/// This function makes a copy of the image database.
void ArConfig_setAugmentedImageDatabase(
    const ArSession *session,
    ArConfig *config,
    const ArAugmentedImageDatabase *augmented_image_database);

/// Returns the image database from the session configuration.
///
/// This function returns a copy of the internally stored image database.
void ArConfig_getAugmentedImageDatabase(
    const ArSession *session,
    const ArConfig *config,
    ArAugmentedImageDatabase *out_augmented_image_database);

/// Stores the currently configured augmented face mode into @c
/// *augmented_face_mode.
void ArConfig_getAugmentedFaceMode(const ArSession *session,
                                   const ArConfig *config,
                                   ArAugmentedFaceMode *augmented_face_mode);

/// Sets the face mode that should be used. See @c ArAugmentedFaceMode for
/// available options. Augmented Faces is currently only supported when using
/// the front-facing (selfie) camera.  See #AR_SESSION_FEATURE_FRONT_CAMERA for
/// details.
void ArConfig_setAugmentedFaceMode(const ArSession *session,
                                   ArConfig *config,
                                   ArAugmentedFaceMode augmented_face_mode);

/// Sets the focus mode that should be used. See ::ArFocusMode for available
/// options.
void ArConfig_setFocusMode(const ArSession *session,
                           ArConfig *config,
                           ArFocusMode focus_mode);

/// Stores the currently configured focus mode into @c *focus_mode.
void ArConfig_getFocusMode(const ArSession *session,
                           ArConfig *config,
                           ArFocusMode *focus_mode);

/// @}

// === ArCameraConfigList and ArCameraConfig methods ===

/// @addtogroup cameraconfig
/// @{

// === ArCameraConfigList methods ===

/// Creates a camera config list object.
///
/// @param[in]   session      The ARCore session
/// @param[out]  out_list     A pointer to an @c ArCameraConfigList* to receive
///     the address of the newly allocated ArCameraConfigList.
void ArCameraConfigList_create(const ArSession *session,
                               ArCameraConfigList **out_list);

/// Releases the memory used by a camera config list object,
/// along with all the camera config references it holds.
void ArCameraConfigList_destroy(ArCameraConfigList *list);

/// Retrieves the number of camera configs in this list.
void ArCameraConfigList_getSize(const ArSession *session,
                                const ArCameraConfigList *list,
                                int32_t *out_size);

/// Retrieves the specific camera config based on the position in this list.
void ArCameraConfigList_getItem(const ArSession *session,
                                const ArCameraConfigList *list,
                                int32_t index,
                                ArCameraConfig *out_camera_config);

// === ArCameraConfig methods ===

/// Creates a camera config object.
///
/// @param[in]   session           The ARCore session
/// @param[out]  out_camera_config A pointer to an @c ArCameraConfig* to receive
///     the address of the newly allocated ArCameraConfig.
void ArCameraConfig_create(const ArSession *session,
                           ArCameraConfig **out_camera_config);

/// Releases the memory used by a camera config object.
void ArCameraConfig_destroy(ArCameraConfig *camera_config);

/// Obtains the camera image dimensions for the given camera config.
void ArCameraConfig_getImageDimensions(const ArSession *session,
                                       const ArCameraConfig *camera_config,
                                       int32_t *out_width,
                                       int32_t *out_height);

/// Obtains the texture dimensions for the given camera config.
void ArCameraConfig_getTextureDimensions(const ArSession *session,
                                         const ArCameraConfig *camera_config,
                                         int32_t *out_width,
                                         int32_t *out_height);

/// Obtains the camera id for the given camera config which is obtained from the
/// list of ArCore compatible camera configs.
void ArCameraConfig_getCameraId(const ArSession *session,
                                const ArCameraConfig *camera_config,
                                char **out_camera_id);

/// Obtains the facing direction of the camera selected by this config.
void ArCameraConfig_getFacingDirection(
    const ArSession *session,
    const ArCameraConfig *camera_config,
    ArCameraConfigFacingDirection *out_facing);
/// @}

// === ArSession methods ===

/// @addtogroup session
/// @{

/// Releases resources used by an ARCore session.
/// This method will take several seconds to complete. To prevent blocking
/// the main thread, call ArSession_pause() on the main thread, and then call
/// ArSession_destroy() on a background thread.
///
void ArSession_destroy(ArSession *session);

/// Before release 1.2.0: Checks if the provided configuration is usable on the
/// this device. If this method returns #AR_ERROR_UNSUPPORTED_CONFIGURATION,
/// calls to ArSession_configure(Config) with this configuration will fail.
///
/// This function now always returns true. See documentation for each
/// configuration entry to know which configuration options & combinations are
/// supported.
///
/// @param[in] session The ARCore session
/// @param[in] config  The configuration to test
/// @return #AR_SUCCESS or:
///  - #AR_ERROR_INVALID_ARGUMENT if any of the arguments are null.
/// @deprecated in release 1.2.0. Please refer to the release notes
/// (<a
/// href="https://github.com/google-ar/arcore-android-sdk/releases/tag/v1.2.0">release
/// notes 1.2.0</a>)
///
ArStatus ArSession_checkSupported(const ArSession *session,
                                  const ArConfig *config)
    AR_DEPRECATED(
        "deprecated in release 1.2.0. Please see function documentation");

// TODO(b/122918249): Document here that MESH3D works only on FRONT_CAMERA
/// Configures the session with the given config.
/// Note: a session is always initially configured with the default config.
/// This should be called if a configuration different than default is needed.
///
/// The following configurations are not supported:
///
/// - When using the back-facing camera (default):
///   - #AR_AUGMENTED_FACE_MODE_MESH3D.
/// - When using the front-facing (selfie) camera
///   (#AR_SESSION_FEATURE_FRONT_CAMERA):
///   - Any config using ArConfig_setAugmentedImageDatabase().
///   - #AR_CLOUD_ANCHOR_MODE_ENABLED.
///
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_FATAL
/// - #AR_ERROR_UNSUPPORTED_CONFIGURATION If the configuration is not supported,
///   see above restrictions
ArStatus ArSession_configure(ArSession *session, const ArConfig *config);

/// Gets the current config. More specifically, fills the given ArConfig object
/// with the copy of the configuration most recently set by
/// ArSession_configure(). Note: if the session was not explicitly configured, a
/// default configuration is returned (same as ArConfig_create()).
void ArSession_getConfig(ArSession *session, ArConfig *out_config);

/// Starts or resumes the ARCore Session.
///
/// Typically this should be called from <a
/// href="https://developer.android.com/reference/android/app/Activity.html#onResume()"
/// ><tt>Activity.onResume()</tt></a>.
///
/// Note that if the camera configuration has been changed by
/// ArSession_setCameraConfig() since the last call to ArSession_resume(), all
/// images previously acquired using ArFrame_acquireCameraImage() must be
/// released by calling ArImage_release() before calling ArSession_resume().  If
/// there are open images, ArSession_resume will return AR_ERROR_ILLEGAL_STATE
/// and the session will not resume.
///
/// @returns #AR_SUCCESS or any of:
/// - #AR_ERROR_FATAL
/// - #AR_ERROR_CAMERA_PERMISSION_NOT_GRANTED
/// - #AR_ERROR_CAMERA_NOT_AVAILABLE
/// - #AR_ERROR_ILLEGAL_STATE
ArStatus ArSession_resume(ArSession *session);

/// Pause the current session. This method will stop the camera feed and release
/// resources. The session can be restarted again by calling ArSession_resume().
///
/// Typically this should be called from <a
/// href="https://developer.android.com/reference/android/app/Activity.html#onPause()"
/// ><tt>Activity.onPause()</tt></a>.
///
/// Note that ARCore might continue consuming substantial computing resources
/// for up to 10 seconds after calling this method.
///
/// @returns #AR_SUCCESS or any of:
/// - #AR_ERROR_FATAL
ArStatus ArSession_pause(ArSession *session);

/// Sets the OpenGL texture name (id) that will allow GPU access to the camera
/// image. The provided ID should have been created with @c glGenTextures(). The
/// resulting texture must be bound to the @c GL_TEXTURE_EXTERNAL_OES target for
/// use. Shaders accessing this texture must use a @c samplerExternalOES
/// sampler. See sample code for an example.
void ArSession_setCameraTextureName(ArSession *session, uint32_t texture_id);

/// Sets the aspect ratio, coordinate scaling, and display rotation. This data
/// is used by UV conversion, projection matrix generation, and hit test logic.
///
/// Note: this function doesn't fail. If given invalid input, it logs a error
/// and doesn't apply the changes.
///
/// @param[in] session   The ARCore session
/// @param[in] rotation  Display rotation specified by @c android.view.Surface
///     constants: @c ROTATION_0, @c ROTATION_90, @c ROTATION_180 and
///     @c ROTATION_270
/// @param[in] width     Width of the view, in pixels
/// @param[in] height    Height of the view, in pixels
void ArSession_setDisplayGeometry(ArSession *session,
                                  int32_t rotation,
                                  int32_t width,
                                  int32_t height);

/// Updates the state of the ARCore system. This includes: receiving a new
/// camera frame, updating the location of the device, updating the location of
/// tracking anchors, updating detected planes, etc.
///
/// This call may cause off-screen OpenGL activity. Because of this, to avoid
/// unnecessary frame buffer flushes and reloads, this call should not be made
/// in the middle of rendering a frame or offscreen buffer.
///
/// This call may update the pose of all created anchors and detected planes.
/// The set of updated objects is accessible through
/// ArFrame_getUpdatedTrackables().
///
/// @c update() in blocking mode (see ::ArUpdateMode) will wait until a
/// new camera image is available, or until the built-in timeout
/// (currently 66ms) is reached.
/// If the camera image does not arrive by the built-in timeout, then
/// @c update() will return the most recent ::ArFrame object. For some
/// applications it may be important to know if a new frame was actually
/// obtained (for example, to avoid redrawing if the camera did not produce a
/// new frame). To do that, compare the current frame's timestamp, obtained via
/// @c ArFrame_getTimestamp, with the previously recorded frame timestamp. If
/// they are different, this is a new frame.
///
/// During startup the camera system may not produce actual images
/// immediately. In this common case, a frame with timestamp = 0 will be
/// returned.
///
/// @param[in]    session   The ARCore session
/// @param[inout] out_frame The Frame object to populate with the updated world
///     state.  This frame must have been previously created using
///     ArFrame_create().  The same ArFrame instance may be used when calling
///     this repeatedly.
///
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_FATAL
/// - #AR_ERROR_SESSION_PAUSED
/// - #AR_ERROR_TEXTURE_NOT_SET
/// - #AR_ERROR_MISSING_GL_CONTEXT
/// - #AR_ERROR_CAMERA_NOT_AVAILABLE - camera was removed during runtime.
ArStatus ArSession_update(ArSession *session, ArFrame *out_frame);

/// Defines a tracked location in the physical world.
///
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_NOT_TRACKING
/// - #AR_ERROR_SESSION_PAUSED
/// - #AR_ERROR_RESOURCE_EXHAUSTED
ArStatus ArSession_acquireNewAnchor(ArSession *session,
                                    const ArPose *pose,
                                    ArAnchor **out_anchor);

/// Returns all known anchors, including those not currently tracked. Anchors
/// forgotten by ARCore due to a call to ArAnchor_detach() or entering the
/// #AR_TRACKING_STATE_STOPPED state will not be included.
///
/// @param[in]    session         The ARCore session
/// @param[inout] out_anchor_list The list to fill.  This list must have already
///     been allocated with ArAnchorList_create().  If previously used, the list
///     will first be cleared.
void ArSession_getAllAnchors(const ArSession *session,
                             ArAnchorList *out_anchor_list);

/// Returns the list of all known @ref trackable "trackables".  This includes
/// ::ArPlane objects if plane detection is enabled, as well as ::ArPoint
/// objects created as a side effect of calls to ArSession_acquireNewAnchor() or
/// ArFrame_hitTest().
///
/// @param[in]    session            The ARCore session
/// @param[in]    filter_type        The type(s) of trackables to return.  See
///     ::ArTrackableType for legal values.
/// @param[inout] out_trackable_list The list to fill.  This list must have
///     already been allocated with ArTrackableList_create().  If previously
///     used, the list will first be cleared.
void ArSession_getAllTrackables(const ArSession *session,
                                ArTrackableType filter_type,
                                ArTrackableList *out_trackable_list);

/// This will create a new cloud anchor using pose and other metadata from
/// @c anchor.
///
/// If the function returns #AR_SUCCESS, the cloud state of @c out_cloud_anchor
/// will be set to #AR_CLOUD_ANCHOR_STATE_TASK_IN_PROGRESS and the initial pose
/// will be set to the pose of @c anchor. However, the new @c out_cloud_anchor
/// is completely independent of @c anchor, and the poses may diverge over time.
/// If the return value of this function is not #AR_SUCCESS, then
/// @c out_cloud_anchor will be set to null.
///
/// @param[in]    session          The ARCore session
/// @param[in]    anchor           The anchor to be hosted
/// @param[inout] out_cloud_anchor The new cloud anchor
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_NOT_TRACKING
/// - #AR_ERROR_SESSION_PAUSED
/// - #AR_ERROR_CLOUD_ANCHORS_NOT_CONFIGURED
/// - #AR_ERROR_RESOURCE_EXHAUSTED
/// - #AR_ERROR_ANCHOR_NOT_SUPPORTED_FOR_HOSTING
ArStatus ArSession_hostAndAcquireNewCloudAnchor(ArSession *session,
                                                const ArAnchor *anchor,
                                                ArAnchor **out_cloud_anchor);

/// This will create a new cloud anchor, and schedule a resolving task to
/// resolve the anchor's pose using the given cloud anchor ID.
///
/// If this function returns #AR_SUCCESS, the cloud state of @c out_cloud_anchor
/// will be #AR_CLOUD_ANCHOR_STATE_TASK_IN_PROGRESS, and its tracking state will
/// be #AR_TRACKING_STATE_PAUSED. This anchor will never start tracking until
/// its pose has been successfully resolved. If the resolving task ends in an
/// error, the tracking state will be set to #AR_TRACKING_STATE_STOPPED. If the
/// return value is not #AR_SUCCESS, then @c out_cloud_anchor will be set to
/// null.
///
/// @param[in]    session          The ARCore session
/// @param[in]    cloud_anchor_id  The cloud ID of the anchor to be resolved
/// @param[inout] out_cloud_anchor The new cloud anchor
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_NOT_TRACKING
/// - #AR_ERROR_SESSION_PAUSED
/// - #AR_ERROR_CLOUD_ANCHORS_NOT_CONFIGURED
/// - #AR_ERROR_RESOURCE_EXHAUSTED
ArStatus ArSession_resolveAndAcquireNewCloudAnchor(ArSession *session,
                                                   const char *cloud_anchor_id,
                                                   ArAnchor **out_cloud_anchor);

/// Enumerates the list of supported camera configs on the device.
/// Can be called at any time.  The supported camera configs will be filled in
/// the provided list after clearing it.
///
/// The list will always return 3 camera configs. The GPU texture resolutions
/// are the same in all three configs. Currently, most devices provide GPU
/// texture resolution of 1920 x 1080 but this may vary with device
/// capabilities. The CPU image resolutions returned are VGA, a middle
/// resolution, and a large resolution matching the GPU texture. The middle
/// resolution will often be 1280 x 720, but may vary with device capabilities.
///
/// Note: Prior to ARCore 1.6 the middle CPU image resolution was guaranteed to
/// be 1280 x 720 on all devices.
///
/// @param[in]    session          The ARCore session
/// @param[inout] list             The list to fill. This list must have already
///      been allocated with ArCameraConfigList_create().  The list is cleared
///      to remove any existing elements.  Once it is no longer needed, the list
///      must be destroyed using ArCameraConfigList_destroy to release allocated
///      memory.
void ArSession_getSupportedCameraConfigs(const ArSession *session,
                                         ArCameraConfigList *list);

/// Sets the ArCameraConfig that the ArSession should use.  Can only be called
/// while the session is paused.  The provided ArCameraConfig must be one of the
///  configs returned by ArSession_getSupportedCameraConfigs.
///
/// The camera config will be applied once the session is resumed.
/// All previously acquired frame images must be released via ArImage_release
/// before calling resume(). Failure to do so will cause resume() to return
/// AR_ERROR_ILLEGAL_STATE error.
///
/// @param[in]    session          The ARCore session
/// @param[in]    camera_config    The provided ArCameraConfig must be from a
///     list returned by ArSession_getSupportedCameraConfigs.
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_INVALID_ARGUMENT
/// - #AR_ERROR_SESSION_NOT_PAUSED
ArStatus ArSession_setCameraConfig(const ArSession *session,
                                   const ArCameraConfig *camera_config);

/// Gets the ArCameraConfig that the ArSession is currently using.  If the
/// camera config was not explicitly set then it returns the default
/// camera config.  Use ArCameraConfig_destroy to release memory associated with
/// the returned camera config once it is no longer needed.
///
/// @param[in]    session           The ARCore session
/// @param[inout] out_camera_config The camera config object to fill. This
///      object must have already been allocated with ArCameraConfig_create().
///      Use ArCameraConfig_destroy to release memory associated with
///      out_camera_config once it is no longer needed.
void ArSession_getCameraConfig(const ArSession *session,
                               ArCameraConfig *out_camera_config);

/// @}

// === ArPose methods ===

/// @addtogroup pose
/// @{

/// Allocates and initializes a new pose object.  @c pose_raw points to an array
/// of 7 floats, describing the rotation (quaternion) and translation of the
/// pose in the same order as the first 7 elements of the Android
/// @c Sensor.TYPE_POSE_6DOF values documented on <a
/// href="https://developer.android.com/reference/android/hardware/SensorEvent.html#values"
/// >@c SensorEvent.values() </a>
///
/// The order of the values is: qx, qy, qz, qw, tx, ty, tz.
///
/// If @c pose_raw is null, initializes with the identity pose.
void ArPose_create(const ArSession *session,
                   const float *pose_raw,
                   ArPose **out_pose);

/// Releases memory used by a pose object.
void ArPose_destroy(ArPose *pose);

/// Extracts the quaternion rotation and translation from a pose object.
/// @param[in]  session       The ARCore session
/// @param[in]  pose          The pose to extract
/// @param[out] out_pose_raw  Pointer to an array of 7 floats, to be filled with
///     the quaternion rotation and translation as described in ArPose_create().
void ArPose_getPoseRaw(const ArSession *session,
                       const ArPose *pose,
                       float *out_pose_raw);

/// Converts a pose into a 4x4 transformation matrix.
/// @param[in]  session                  The ARCore session
/// @param[in]  pose                     The pose to convert
/// @param[out] out_matrix_col_major_4x4 Pointer to an array of 16 floats, to be
///     filled with a column-major homogenous transformation matrix, as used by
///     OpenGL.
void ArPose_getMatrix(const ArSession *session,
                      const ArPose *pose,
                      float *out_matrix_col_major_4x4);

/// @}

// === ArCamera methods ===

/// @addtogroup camera
/// @{

/// Sets @c out_pose to the pose of the physical camera in world space for the
/// latest frame. This is an OpenGL camera pose with +X pointing right, +Y
/// pointing right up, -Z pointing in the direction the camera is looking, with
/// "right" and "up" being relative to the image readout in the usual
/// left-to-right top-to-bottom order. Specifically, this is the camera pose at
/// the center of exposure of the center row of the image.
///
/// <b>For applications using the SDK for ARCore 1.5 and earlier</b>, the
/// returned pose is rotated around the Z axis by a multiple of 90 degrees so
/// that the axes correspond approximately to those of the <a
/// href="https://developer.android.com/guide/topics/sensors/sensors_overview#sensors-coords">Android
/// Sensor Coordinate System</a>.
///
/// See Also:
///
/// * ArCamera_getDisplayOrientedPose() for the pose of the virtual camera. It
///   will differ by a local rotation about the Z axis by a multiple of 90
///   degrees.
/// * ArFrame_getAndroidSensorPose() for the pose of the Android sensor frame.
///   It will differ in both orientation and location.
/// * ArFrame_transformCoordinates2d() to convert viewport coordinates to
///   texture coordinates.
///
/// Note: This pose is only useful when ArCamera_getTrackingState() returns
/// #AR_TRACKING_STATE_TRACKING and otherwise should not be used.
///
/// @param[in]    session  The ARCore session
/// @param[in]    camera   The session's camera (retrieved from any frame).
/// @param[inout] out_pose An already-allocated ArPose object into which the
///     pose will be stored.
void ArCamera_getPose(const ArSession *session,
                      const ArCamera *camera,
                      ArPose *out_pose);

/// Sets @c out_pose to the virtual camera pose in world space for rendering AR
/// content onto the latest frame. This is an OpenGL camera pose with +X
/// pointing right, +Y pointing up, and -Z pointing in the direction the camera
/// is looking, with "right" and "up" being relative to current logical display
/// orientation.
///
/// See Also:
///
/// * ArCamera_getViewMatrix() to conveniently compute the OpenGL View Matrix.
/// * ArCamera_getPose() for the physical pose of the camera. It will differ by
///   a local rotation about the Z axis by a multiple of 90 degrees.
/// * ArFrame_getAndroidSensorPose() for the pose of the android sensor frame.
///   It will differ in both orientation and location.
/// * ArSession_setDisplayGeometry() to update the display rotation.
///
/// Note: This pose is only useful when ArCamera_getTrackingState() returns
/// #AR_TRACKING_STATE_TRACKING and otherwise should not be used.
///
/// @param[in]    session  The ARCore session
/// @param[in]    camera   The session's camera (retrieved from any frame).
/// @param[inout] out_pose An already-allocated ArPose object into which the
///     pose will be stored.
void ArCamera_getDisplayOrientedPose(const ArSession *session,
                                     const ArCamera *camera,
                                     ArPose *out_pose);

/// Returns the view matrix for the camera for this frame. This matrix performs
/// the inverse transform as the pose provided by
/// ArCamera_getDisplayOrientedPose().
///
/// @param[in]    session           The ARCore session
/// @param[in]    camera            The session's camera.
/// @param[inout] out_col_major_4x4 Pointer to an array of 16 floats, to be
///     filled with a column-major homogenous transformation matrix, as used by
///     OpenGL.
void ArCamera_getViewMatrix(const ArSession *session,
                            const ArCamera *camera,
                            float *out_col_major_4x4);

/// Gets the current motion tracking state of this camera. If this state is
/// anything other than #AR_TRACKING_STATE_TRACKING the pose should not be
/// considered useful. Use ArCamera_getTrackingFailureReason() to determine the
/// best recommendation to provide to the user to restore motion tracking.
void ArCamera_getTrackingState(const ArSession *session,
                               const ArCamera *camera,
                               ArTrackingState *out_tracking_state);

/// Gets the reason that ArCamera_getTrackingState() is
/// #AR_TRACKING_STATE_PAUSED. Note, it returns
/// ArTrackingFailureReason#AR_TRACKING_FAILURE_REASON_NONE briefly after
/// ArSession_resume(), while the motion tracking is initializing. Always
/// returns ArTrackingFailureReason#AR_TRACKING_FAILURE_REASON_NONE when
/// ArCamera_getTrackingState is #AR_TRACKING_STATE_TRACKING.
///
/// If multiple potential causes for motion tracking failure are detected,
/// this reports the most actionable failure reason.
void ArCamera_getTrackingFailureReason(
    const ArSession *session,
    const ArCamera *camera,
    ArTrackingFailureReason *out_tracking_failure_reason);

/// Computes a projection matrix for rendering virtual content on top of the
/// camera image. Note that the projection matrix reflects the current display
/// geometry and display rotation.
///
/// Note: When using #AR_SESSION_FEATURE_FRONT_CAMERA, the returned projection
/// matrix will incorporate a horizontal flip.
///
/// @param[in]    session            The ARCore session
/// @param[in]    camera             The session's camera.
/// @param[in]    near               Specifies the near clip plane, in meters
/// @param[in]    far                Specifies the far clip plane, in meters
/// @param[inout] dest_col_major_4x4 Pointer to an array of 16 floats, to
///     be filled with a column-major homogenous transformation matrix, as used
///     by OpenGL.
void ArCamera_getProjectionMatrix(const ArSession *session,
                                  const ArCamera *camera,
                                  float near,
                                  float far,
                                  float *dest_col_major_4x4);

/// Retrieves the unrotated and uncropped intrinsics for the image (CPU) stream.
/// The intrinsics may change per frame, so this should be called
/// on each frame to get the intrinsics for the current frame.
///
/// @param[in]    session                The ARCore session
/// @param[in]    camera                 The session's camera.
/// @param[inout] out_camera_intrinsics  The camera_intrinsics data.
void ArCamera_getImageIntrinsics(const ArSession *session,
                                 const ArCamera *camera,
                                 ArCameraIntrinsics *out_camera_intrinsics);

/// Retrieves the unrotated and uncropped intrinsics for the texture (GPU)
/// stream.  The intrinsics may change per frame, so this should be called
/// on each frame to get the intrinsics for the current frame.
///
/// @param[in]    session                The ARCore session
/// @param[in]    camera                 The session's camera.
/// @param[inout] out_camera_intrinsics  The camera_intrinsics data.
void ArCamera_getTextureIntrinsics(const ArSession *session,
                                   const ArCamera *camera,
                                   ArCameraIntrinsics *out_camera_intrinsics);

/// Releases a reference to the camera.  This must match a call to
/// ArFrame_acquireCamera().
///
/// This method may safely be called with @c nullptr - it will do nothing.
void ArCamera_release(ArCamera *camera);

/// @}

// === ArCameraIntrinsics methods ===
/// @addtogroup intrinsics
/// @{

/// Allocates a camera intrinstics object.
///
/// @param[in]    session                The ARCore session
/// @param[inout] out_camera_intrinsics  The camera_intrinsics data.
void ArCameraIntrinsics_create(const ArSession *session,
                               ArCameraIntrinsics **out_camera_intrinsics);

/// Returns the focal length in pixels.
/// The focal length is conventionally represented in pixels. For a detailed
/// explanation, please see http://ksimek.github.io/2013/08/13/intrinsic.
/// Pixels-to-meters conversion can use SENSOR_INFO_PHYSICAL_SIZE and
/// SENSOR_INFO_PIXEL_ARRAY_SIZE in the Android CameraCharacteristics API.
void ArCameraIntrinsics_getFocalLength(const ArSession *session,
                                       const ArCameraIntrinsics *intrinsics,
                                       float *out_fx,
                                       float *out_fy);

/// Returns the principal point in pixels.
void ArCameraIntrinsics_getPrincipalPoint(const ArSession *session,
                                          const ArCameraIntrinsics *intrinsics,
                                          float *out_cx,
                                          float *out_cy);

/// Returns the image's width and height in pixels.
void ArCameraIntrinsics_getImageDimensions(const ArSession *session,
                                           const ArCameraIntrinsics *intrinsics,
                                           int32_t *out_width,
                                           int32_t *out_height);

/// Releases the provided camera intrinsics object.
void ArCameraIntrinsics_destroy(ArCameraIntrinsics *camera_intrinsics);

/// @}

// === ArFrame methods ===

/// @addtogroup frame
/// @{

/// Allocates a new ArFrame object, storing the pointer into @c *out_frame.
///
/// Note: the same ArFrame can be used repeatedly when calling ArSession_update.
void ArFrame_create(const ArSession *session, ArFrame **out_frame);

/// Releases an ArFrame and any references it holds.
void ArFrame_destroy(ArFrame *frame);

/// Checks if the display rotation or viewport geometry changed since the
/// previous call to ArSession_update(). The application should re-query
/// ArCamera_getProjectionMatrix() and ArFrame_transformCoordinates2d()
/// whenever this emits non-zero.
void ArFrame_getDisplayGeometryChanged(const ArSession *session,
                                       const ArFrame *frame,
                                       int32_t *out_geometry_changed);

/// Returns the timestamp in nanoseconds when this image was captured. This can
/// be used to detect dropped frames or measure the camera frame rate. The time
/// base of this value is specifically <b>not</b> defined, but it is likely
/// similar to <tt>clock_gettime(CLOCK_BOOTTIME)</tt>.
void ArFrame_getTimestamp(const ArSession *session,
                          const ArFrame *frame,
                          int64_t *out_timestamp_ns);

/// Sets @c out_pose to the pose of the <a
/// href="https://developer.android.com/guide/topics/sensors/sensors_overview#sensors-coords">Android
/// Sensor Coordinate System</a> in the world coordinate space for this frame.
/// The orientation follows the device's "native" orientation (it is not
/// affected by display rotation) with all axes corresponding to those of the
/// Android sensor coordinates.
///
/// See Also:
///
/// * ArCamera_getDisplayOrientedPose() for the pose of the virtual camera.
/// * ArCamera_getPose() for the pose of the physical camera.
/// * ArFrame_getTimestamp() for the system time that this pose was estimated
///   for.
///
/// Note: This pose is only useful when ArCamera_getTrackingState() returns
/// #AR_TRACKING_STATE_TRACKING and otherwise should not be used.
///
/// @param[in]    session  The ARCore session
/// @param[in]    frame    The current frame.
/// @param[inout] out_pose An already-allocated ArPose object into which the
///     pose will be stored.
void ArFrame_getAndroidSensorPose(const ArSession *session,
                                  const ArFrame *frame,
                                  ArPose *out_pose);

/// Transform the given texture coordinates to correctly show the background
/// image. This will account for the display rotation, and any additional
/// required adjustment. For performance, this function should be called only if
/// ArFrame_hasDisplayGeometryChanged() emits true.
///
/// @param[in]    session      The ARCore session
/// @param[in]    frame        The current frame.
/// @param[in]    num_elements The number of floats to transform.  Must be
///     a multiple of 2.  @c uvs_in and @c uvs_out must point to arrays of at
///     least this many floats.
/// @param[in]    uvs_in       Input UV coordinates in normalized screen space.
/// @param[inout] uvs_out      Output UV coordinates in texture coordinates.
/// @deprecated in release 1.7.0. Please use instead: @code
/// ArFrame_transformCoordinates2d(session, frame,
///   AR_COORDINATES_2D_VIEW_NORMALIZED, num_elements, uvs_in,
///   AR_COORDINATES_2D_TEXTURE_NORMALIZED, uvs_out); @endcode
void ArFrame_transformDisplayUvCoords(const ArSession *session,
                                      const ArFrame *frame,
                                      int32_t num_elements,
                                      const float *uvs_in,
                                      float *uvs_out)
    AR_DEPRECATED(
        "deprecated in release 1.7.0. Please see function documentation.");

/// Transforms a list of 2D coordinates from one 2D coordinate system to another
/// 2D coordinate system.
///
/// For Android view coordinates (VIEW, VIEW_NORMALIZED), the view information
/// is taken from the most recent call to @c ArSession_setDisplayGeometry.
///
/// Must be called on the most recently obtained @c ArFrame object. If this
/// function is called on an older frame, a log message will be printed and
/// out_vertices_2d will remain unchanged.
///
/// Some examples of useful conversions:
///  - To transform from [0,1] range to screen-quad coordinates for rendering:
///    VIEW_NORMALIZED -> TEXTURE_NORMALIZED
///  - To transform from [-1,1] range to screen-quad coordinates for rendering:
///    OPENGL_NORMALIZED_DEVICE_COORDINATES -> TEXTURE_NORMALIZED
///  - To transform a point found by a computer vision algorithm in a cpu image
///    into a point on the screen that can be used to place an Android View
///    (e.g. Button) at that location:
///    IMAGE_PIXELS -> VIEW
///  - To transform a point found by a computer vision algorithm in a CPU image
///    into a point to be rendered using GL in clip-space ([-1,1] range):
///    IMAGE_PIXELS -> OPENGL_NORMALIZED_DEVICE_COORDINATES
///
/// If inputCoordinates is same as outputCoordinates, the input vertices will be
/// copied to the output vertices unmodified.
///
/// @param[in]  session         The ARCore session.
/// @param[in]  frame           The current frame.
/// @param[in]  input_coordinates The coordinate system used by @c vectors2d_in.
/// @param[in]  number_of_vertices The number of 2D vertices to transform.
///                             @c vertices_2d and @c out_vertices_2d must
///                             point to arrays of size at least num_vertices*2.
/// @param[in] vertices_2d      Input 2D vertices to transform.
/// @param[in] output_coordinates The coordinate system to convert to.
/// @param[inout] out_vertices_2d Transformed 2d vertices, can be the same array
///                             as vertices_2d for in-place transform.
void ArFrame_transformCoordinates2d(const ArSession *session,
                                    const ArFrame *frame,
                                    ArCoordinates2dType input_coordinates,
                                    int32_t number_of_vertices,
                                    const float *vertices_2d,
                                    ArCoordinates2dType output_coordinates,
                                    float *out_vertices_2d);

/// Performs a ray cast from the user's device in the direction of the given
/// location in the camera view. Intersections with detected scene geometry are
/// returned, sorted by distance from the device; the nearest intersection is
/// returned first.
///
/// Note: Significant geometric leeway is given when returning hit results. For
/// example, a plane hit may be generated if the ray came close, but did not
/// actually hit within the plane extents or plane bounds
/// (ArPlane_isPoseInExtents() and ArPlane_isPoseInPolygon() can be used to
/// determine these cases). A point (point cloud) hit is generated when a point
/// is roughly within one finger-width of the provided screen coordinates.
///
/// The resulting list is ordered by distance, with the nearest hit first
///
/// Note: If not tracking, the hit_result_list will be empty. <br>
/// Note: If called on an old frame (not the latest produced by
///     ArSession_update() the hit_result_list will be empty).
/// Note: When using #AR_SESSION_FEATURE_FRONT_CAMERA, the returned hit result
///     list will always be empty, as the camera is not
///     #AR_TRACKING_STATE_TRACKING}. Hit testing against tracked faces is not
///     currently supported.
///
/// @param[in]    session         The ARCore session.
/// @param[in]    frame           The current frame.
/// @param[in]    pixel_x         Logical X position within the view, as from an
///     Android UI event.
/// @param[in]    pixel_y         Logical X position within the view.
/// @param[inout] hit_result_list The list to fill.  This list must have been
///     previously allocated using ArHitResultList_create().  If the list has
///     been previously used, it will first be cleared.
void ArFrame_hitTest(const ArSession *session,
                     const ArFrame *frame,
                     float pixel_x,
                     float pixel_y,
                     ArHitResultList *hit_result_list);

/// Similar to ArFrame_hitTest(), but takes an arbitrary ray in world space
/// coordinates instead of a screen space point.
///
/// @param[in]    session         The ARCore session.
/// @param[in]    frame           The current frame.
/// @param[in]    ray_origin_3    A pointer to float[3] array containing ray
///     origin in world space coordinates.
/// @param[in]    ray_direction_3 A pointer to float[3] array containing ray
///     direction in world space coordinates. Does not have to be normalized.
/// @param[inout] hit_result_list The list to fill.  This list must have been
///     previously allocated using ArHitResultList_create().  If the list has
///     been previously used, it will first be cleared.
void ArFrame_hitTestRay(const ArSession *session,
                        const ArFrame *frame,
                        const float *ray_origin_3,
                        const float *ray_direction_3,
                        ArHitResultList *hit_result_list);

/// Gets the current ambient light estimate, if light estimation was enabled.
///
/// @param[in]    session            The ARCore session.
/// @param[in]    frame              The current frame.
/// @param[inout] out_light_estimate The light estimate to fill.  This object
///    must have been previously created with ArLightEstimate_create().
void ArFrame_getLightEstimate(const ArSession *session,
                              const ArFrame *frame,
                              ArLightEstimate *out_light_estimate);

/// Acquires the current set of estimated 3d points attached to real-world
/// geometry. A matching call to PointCloud_release() must be made when the
/// application is done accessing the point cloud.
///
/// Note: This information is for visualization and debugging purposes only. Its
/// characteristics and format are subject to change in subsequent versions of
/// the API.
///
/// @param[in]  session         The ARCore session.
/// @param[in]  frame           The current frame.
/// @param[out] out_point_cloud Pointer to an @c ArPointCloud* receive the
///     address of the point cloud.
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_DEADLINE_EXCEEDED if @c frame is not the latest frame from
///   by ArSession_update().
/// - #AR_ERROR_RESOURCE_EXHAUSTED if too many point clouds are currently held.
ArStatus ArFrame_acquirePointCloud(const ArSession *session,
                                   const ArFrame *frame,
                                   ArPointCloud **out_point_cloud);

/// Returns the camera object for the session. Note that this Camera instance is
/// long-lived so the same instance is returned regardless of the frame object
/// this method was called on.
void ArFrame_acquireCamera(const ArSession *session,
                           const ArFrame *frame,
                           ArCamera **out_camera);

/// Gets the camera metadata for the current camera image.
///
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_DEADLINE_EXCEEDED if @c frame is not the latest frame from
///   by ArSession_update().
/// - #AR_ERROR_RESOURCE_EXHAUSTED if too many metadata objects are currently
///   held.
/// - #AR_ERROR_NOT_YET_AVAILABLE if the camera failed to produce metadata for
///   the given frame. Note: this will commonly happen for few frames right
///   after @c ArSession_resume() due to the camera stack bringup.
ArStatus ArFrame_acquireImageMetadata(const ArSession *session,
                                      const ArFrame *frame,
                                      ArImageMetadata **out_metadata);

/// Returns the CPU image for the current frame.
/// Caller is responsible for later releasing the image with @c
/// ArImage_release.
/// Not supported on all devices
/// (see https://developers.google.com/ar/discover/supported-devices).
/// Return values:
/// @returns #AR_SUCCESS or any of:
/// - #AR_ERROR_INVALID_ARGUMENT - one more input arguments are invalid.
/// - #AR_ERROR_DEADLINE_EXCEEDED - the input frame is not the current frame.
/// - #AR_ERROR_RESOURCE_EXHAUSTED - the caller app has exceeded maximum number
///   of images that it can hold without releasing.
/// - #AR_ERROR_NOT_YET_AVAILABLE - image with the timestamp of the input frame
///   was not found within a bounded amount of time, or the camera failed to
///   produce the image
ArStatus ArFrame_acquireCameraImage(ArSession *session,
                                    ArFrame *frame,
                                    ArImage **out_image);

/// Gets the set of anchors that were changed by the ArSession_update() that
/// produced this Frame.
///
/// @param[in]    session            The ARCore session
/// @param[in]    frame              The current frame.
/// @param[inout] out_anchor_list The list to fill.  This list must have
///     already been allocated with ArAnchorList_create().  If previously
///     used, the list will first be cleared.
void ArFrame_getUpdatedAnchors(const ArSession *session,
                               const ArFrame *frame,
                               ArAnchorList *out_anchor_list);

/// Gets the set of trackables of a particular type that were changed by the
/// ArSession_update() call that produced this Frame.
///
/// @param[in]    session            The ARCore session
/// @param[in]    frame              The current frame.
/// @param[in]    filter_type        The type(s) of trackables to return.  See
///     ::ArTrackableType for legal values.
/// @param[inout] out_trackable_list The list to fill.  This list must have
///     already been allocated with ArTrackableList_create().  If previously
///     used, the list will first be cleared.
void ArFrame_getUpdatedTrackables(const ArSession *session,
                                  const ArFrame *frame,
                                  ArTrackableType filter_type,
                                  ArTrackableList *out_trackable_list);

/// @}

// === ArPointCloud methods ===

/// @addtogroup pointcloud
/// @{

/// Retrieves the number of points in the point cloud.
///
void ArPointCloud_getNumberOfPoints(const ArSession *session,
                                    const ArPointCloud *point_cloud,
                                    int32_t *out_number_of_points);

/// Retrieves a pointer to the point cloud data.
///
/// Each point is represented by four consecutive values in the array; first the
/// X, Y, Z position coordinates, followed by a confidence value. This is the
/// same format as described in <a
/// href="https://developer.android.com/reference/android/graphics/ImageFormat.html#DEPTH_POINT_CLOUD"
/// >DEPTH_POINT_CLOUD</a>.
///
/// The pointer returned by this function is valid until ArPointCloud_release()
/// is called. If the number of points is zero, then the value of
/// @c *out_point_cloud_data is undefined.
///
/// If your app needs to keep some point cloud data, for example to compare
/// point cloud data frame to frame, consider copying just the data points your
/// app needs, and then calling ArPointCloud_release() to reduce the amount of
/// memory required.
void ArPointCloud_getData(const ArSession *session,
                          const ArPointCloud *point_cloud,
                          const float **out_point_cloud_data);

/// Retrieves a pointer to the point cloud point IDs. The number of IDs is the
/// same as number of points, and is given by
/// @c ArPointCloud_getNumberOfPoints().
///
/// Each point has a unique identifier (within a session) that is persistent
/// across frames. That is, if a point from point cloud 1 has the same id as the
/// point from point cloud 2, then it represents the same point in space.
///
/// The pointer returned by this function is valid until ArPointCloud_release()
/// is called. If the number of points is zero, then the value of
/// @c *out_point_ids is undefined.
///
/// If your app needs to keep some point cloud data, for example to compare
/// point cloud data frame to frame, consider copying just the data points your
/// app needs, and then calling ArPointCloud_release() to reduce the amount of
/// memory required.
void ArPointCloud_getPointIds(const ArSession *session,
                              const ArPointCloud *point_cloud,
                              const int32_t **out_point_ids);

/// Returns the timestamp in nanoseconds when this point cloud was observed.
/// This timestamp uses the same time base as ArFrame_getTimestamp().
void ArPointCloud_getTimestamp(const ArSession *session,
                               const ArPointCloud *point_cloud,
                               int64_t *out_timestamp_ns);

/// Releases a reference to the point cloud.  This must match a call to
/// ArFrame_acquirePointCloud().
///
/// This method may safely be called with @c nullptr - it will do nothing.
void ArPointCloud_release(ArPointCloud *point_cloud);

/// @}

// === Image Metadata methods ===

/// @addtogroup image
/// @{

/// Retrieves the capture metadata for the current camera image.
///
/// @c ACameraMetadata is a struct in Android NDK. Include NdkCameraMetadata.h
/// to use this type.
///
/// Note: that the ACameraMetadata returned from this function will be invalid
/// after its ArImageMetadata object is released.
void ArImageMetadata_getNdkCameraMetadata(
    const ArSession *session,
    const ArImageMetadata *image_metadata,
    const ACameraMetadata **out_ndk_metadata);

/// Releases a reference to the metadata.  This must match a call to
/// ArFrame_acquireImageMetadata().
///
/// This method may safely be called with @c nullptr - it will do nothing.
void ArImageMetadata_release(ArImageMetadata *metadata);

/// Converts an ArImage object to an Android NDK AImage object. The
/// converted image object format is AIMAGE_FORMAT_YUV_420_888.
void ArImage_getNdkImage(const ArImage *image, const AImage **out_ndk_image);

/// Releases an instance of ArImage returned by ArFrame_acquireCameraImage().
void ArImage_release(ArImage *image);

/// @}

// === ArLightEstimate methods ===

/// @addtogroup light
/// @{

/// Allocates a light estimate object.
void ArLightEstimate_create(const ArSession *session,
                            ArLightEstimate **out_light_estimate);

/// Releases the provided light estimate object.
void ArLightEstimate_destroy(ArLightEstimate *light_estimate);

/// Retrieves the validity state of a light estimate.  If the resulting value of
/// @c *out_light_estimate_state is not #AR_LIGHT_ESTIMATE_STATE_VALID, the
/// estimate should not be used for rendering.
void ArLightEstimate_getState(const ArSession *session,
                              const ArLightEstimate *light_estimate,
                              ArLightEstimateState *out_light_estimate_state);

/// Retrieves the pixel intensity, in gamma space, of the current camera view.
/// Values are in the range (0.0, 1.0), with zero being black and one being
/// white.
/// If rendering in gamma space, divide this value by 0.466, which is middle
/// gray in gamma space, and multiply against the final calculated color after
/// rendering.
/// If rendering in linear space, first convert this value to linear space by
/// rising to the power 2.2. Normalize the result by dividing it by 0.18 which
/// is middle gray in linear space. Then multiply by the final calculated color
/// after rendering.
void ArLightEstimate_getPixelIntensity(const ArSession *session,
                                       const ArLightEstimate *light_estimate,
                                       float *out_pixel_intensity);

/// Gets the color correction values that are uploaded to the fragment shader.
/// Use the RGB scale factors (components 0-2) to match the color of the light
/// in the scene. Use the pixel intensity (component 3) to match the intensity
/// of the light in the scene.
///
/// `out_color_correction_4` components are:
///   - `[0]` Red channel scale factor.
///   - `[1]` Green channel scale factor.
///   - `[2]` Blue channel scale factor.
///   - `[3]` Pixel intensity. This is the same value as the one return from
///       ArLightEstimate_getPixelIntensity().
///
///  The RGB scale factors can be used independently from the pixel intensity
///  value. They are put together for the convenience of only having to upload
///  one float4 to the fragment shader.
///
///  The RGB scale factors are not intended to brighten nor dim the scene.  They
///  are only to shift the color of the virtual object towards the color of the
///  light; not intensity of the light. The pixel intensity is used to match the
///  intensity of the light in the scene.
///
///  Color correction values are reported in gamma space.
///  If rendering in gamma space, component-wise multiply them against the final
///  calculated color after rendering.
///  If rendering in linear space, first convert the values to linear space by
///  rising to the power 2.2. Then component-wise multiply against the final
///  calculated color after rendering.
void ArLightEstimate_getColorCorrection(const ArSession *session,
                                        const ArLightEstimate *light_estimate,
                                        float *out_color_correction_4);

/// @}

// === ArAnchorList methods ===

/// @addtogroup anchor
/// @{

/// Creates an anchor list object.
void ArAnchorList_create(const ArSession *session,
                         ArAnchorList **out_anchor_list);

/// Releases the memory used by an anchor list object, along with all the anchor
/// references it holds.
void ArAnchorList_destroy(ArAnchorList *anchor_list);

/// Retrieves the number of anchors in this list.
void ArAnchorList_getSize(const ArSession *session,
                          const ArAnchorList *anchor_list,
                          int32_t *out_size);

/// Acquires a reference to an indexed entry in the list.  This call must
/// eventually be matched with a call to ArAnchor_release().
void ArAnchorList_acquireItem(const ArSession *session,
                              const ArAnchorList *anchor_list,
                              int32_t index,
                              ArAnchor **out_anchor);

// === ArAnchor methods ===

/// Retrieves the pose of the anchor in the world coordinate space. This pose
/// produced by this call may change each time ArSession_update() is called.
/// This pose should only be used for rendering if ArAnchor_getTrackingState()
/// returns #AR_TRACKING_STATE_TRACKING.
///
/// @param[in]    session  The ARCore session.
/// @param[in]    anchor   The anchor to retrieve the pose of.
/// @param[inout] out_pose An already-allocated ArPose object into which the
///     pose will be stored.
void ArAnchor_getPose(const ArSession *session,
                      const ArAnchor *anchor,
                      ArPose *out_pose);

/// Retrieves the current state of the pose of this anchor.
void ArAnchor_getTrackingState(const ArSession *session,
                               const ArAnchor *anchor,
                               ArTrackingState *out_tracking_state);

/// Tells ARCore to stop tracking and forget this anchor.  This call does not
/// release any references to the anchor - that must be done separately using
/// ArAnchor_release().
void ArAnchor_detach(ArSession *session, ArAnchor *anchor);

/// Releases a reference to an anchor. To stop tracking for this anchor, call
/// ArAnchor_detach() first.
///
/// This method may safely be called with @c nullptr - it will do nothing.
void ArAnchor_release(ArAnchor *anchor);

/// Acquires the cloud anchor ID of the anchor. The ID acquired is an ASCII
/// null-terminated string. The acquired ID must be released after use by the
/// @c ArString_release function. For anchors with cloud state
/// #AR_CLOUD_ANCHOR_STATE_NONE or #AR_CLOUD_ANCHOR_STATE_TASK_IN_PROGRESS, this
/// will always be an empty string.
///
/// @param[in]    session             The ARCore session.
/// @param[in]    anchor              The anchor to retrieve the cloud ID of.
/// @param[inout] out_cloud_anchor_id A pointer to the acquired ID string.
void ArAnchor_acquireCloudAnchorId(ArSession *session,
                                   ArAnchor *anchor,
                                   char **out_cloud_anchor_id);

/// Gets the current cloud anchor state of the anchor. This state is guaranteed
/// not to change until update() is called.
///
/// @param[in]    session   The ARCore session.
/// @param[in]    anchor    The anchor to retrieve the cloud state of.
/// @param[inout] out_state The current cloud state of the anchor.
void ArAnchor_getCloudAnchorState(const ArSession *session,
                                  const ArAnchor *anchor,
                                  ArCloudAnchorState *out_state);

/// @}

// === ArTrackableList methods ===

/// @addtogroup trackable
/// @{

/// Creates a trackable list object.
void ArTrackableList_create(const ArSession *session,
                            ArTrackableList **out_trackable_list);

/// Releases the memory used by a trackable list object, along with all the
/// anchor references it holds.
void ArTrackableList_destroy(ArTrackableList *trackable_list);

/// Retrieves the number of trackables in this list.
void ArTrackableList_getSize(const ArSession *session,
                             const ArTrackableList *trackable_list,
                             int32_t *out_size);

/// Acquires a reference to an indexed entry in the list.  This call must
/// eventually be matched with a call to ArTrackable_release().
void ArTrackableList_acquireItem(const ArSession *session,
                                 const ArTrackableList *trackable_list,
                                 int32_t index,
                                 ArTrackable **out_trackable);

// === ArTrackable methods ===

/// Releases a reference to a trackable. This does not mean that the trackable
/// will necessarily stop tracking. The same trackable may still be included in
/// from other calls, for example ArSession_getAllTrackables().
///
/// This method may safely be called with @c nullptr - it will do nothing.
void ArTrackable_release(ArTrackable *trackable);

/// Retrieves the type of the trackable.  See ::ArTrackableType for valid types.
void ArTrackable_getType(const ArSession *session,
                         const ArTrackable *trackable,
                         ArTrackableType *out_trackable_type);

/// Retrieves the current state of ARCore's knowledge of the pose of this
/// trackable.
void ArTrackable_getTrackingState(const ArSession *session,
                                  const ArTrackable *trackable,
                                  ArTrackingState *out_tracking_state);

/// Creates an Anchor at the given pose in the world coordinate space, attached
/// to this Trackable, and acquires a reference to it. The type of Trackable
/// will determine the semantics of attachment and how the Anchor's pose will be
/// updated to maintain this relationship. Note that the relative offset between
/// the pose of multiple Anchors attached to a Trackable may adjust slightly
/// over time as ARCore updates its model of the world.
///
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_NOT_TRACKING if the trackable's tracking state was not
///   #AR_TRACKING_STATE_TRACKING
/// - #AR_ERROR_SESSION_PAUSED if the session was paused
/// - #AR_ERROR_RESOURCE_EXHAUSTED if too many anchors exist
/// - #AR_ERROR_ILLEGAL_STATE if this trackable doesn't support anchors
ArStatus ArTrackable_acquireNewAnchor(ArSession *session,
                                      ArTrackable *trackable,
                                      ArPose *pose,
                                      ArAnchor **out_anchor);

/// Gets the set of anchors attached to this trackable.
///
/// @param[in]    session         The ARCore session
/// @param[in]    trackable       The trackable to query the anchors of.
/// @param[inout] out_anchor_list The list to fill.  This list must have
///     already been allocated with ArAnchorList_create().  If previously
///     used, the list will first be cleared.
void ArTrackable_getAnchors(const ArSession *session,
                            const ArTrackable *trackable,
                            ArAnchorList *out_anchor_list);

/// @}

// === ArPlane methods ===

/// @addtogroup plane
/// @{

/// Acquires a reference to the plane subsuming this plane.
///
/// Two or more planes may be automatically merged into a single parent plane,
/// resulting in this method acquiring the parent plane when called with each
/// child plane. A subsumed plane becomes identical to the parent plane, and
/// will continue behaving as if it were independently tracked, for example
/// being included in the output of ArFrame_getUpdatedTrackables().
///
/// In cases where a subsuming plane is itself subsumed, this function
/// will always return the topmost non-subsumed plane.
///
/// Note: this function will set @c *out_subsumed_by to NULL if the plane is not
/// subsumed.
void ArPlane_acquireSubsumedBy(const ArSession *session,
                               const ArPlane *plane,
                               ArPlane **out_subsumed_by);

/// Retrieves the type (orientation) of the plane.  See ::ArPlaneType.
void ArPlane_getType(const ArSession *session,
                     const ArPlane *plane,
                     ArPlaneType *out_plane_type);

/// Returns the pose of the center position of the plane's bounding rectangle.
/// The pose's transformed +Y axis will be a normal vector pointing out of
/// plane. The transformed +X and +Z axes represent right and up relative to the
/// plane.
///
/// @param[in]    session  The ARCore session.
/// @param[in]    plane    The plane for which to retrieve center pose.
/// @param[inout] out_pose An already-allocated ArPose object into which the
///     pose will be stored.
void ArPlane_getCenterPose(const ArSession *session,
                           const ArPlane *plane,
                           ArPose *out_pose);

/// Retrieves the length of this plane's bounding rectangle measured along the
/// local X-axis of the coordinate space defined by the output of
/// ArPlane_getCenterPose().
void ArPlane_getExtentX(const ArSession *session,
                        const ArPlane *plane,
                        float *out_extent_x);

/// Retrieves the length of this plane's bounding rectangle measured along the
/// local Z-axis of the coordinate space defined by the output of
/// ArPlane_getCenterPose().
void ArPlane_getExtentZ(const ArSession *session,
                        const ArPlane *plane,
                        float *out_extent_z);

/// Retrieves the number of elements (not vertices) in the boundary polygon.
/// The number of vertices is 1/2 this size.
void ArPlane_getPolygonSize(const ArSession *session,
                            const ArPlane *plane,
                            int32_t *out_polygon_size);

/// Returns the 2D vertices of a convex polygon approximating the detected
/// plane, in the form <tt>[x1, z1, x2, z2, ...]</tt>. These X-Z values are in
/// the plane's local x-z plane (y=0) and must be transformed by the pose
/// (ArPlane_getCenterPose()) to get the boundary in world coordinates.
///
/// @param[in]    session        The ARCore session.
/// @param[in]    plane          The plane to retrieve the polygon from.
/// @param[inout] out_polygon_xz A pointer to an array of floats.  The length of
///     this array must be at least that reported by ArPlane_getPolygonSize().
void ArPlane_getPolygon(const ArSession *session,
                        const ArPlane *plane,
                        float *out_polygon_xz);

/// Sets @c *out_pose_in_extents to non-zero if the given pose (usually obtained
/// from a HitResult) is in the plane's rectangular extents.
void ArPlane_isPoseInExtents(const ArSession *session,
                             const ArPlane *plane,
                             const ArPose *pose,
                             int32_t *out_pose_in_extents);

/// Sets @c *out_pose_in_extents to non-zero if the given pose (usually obtained
/// from a HitResult) is in the plane's polygon.
void ArPlane_isPoseInPolygon(const ArSession *session,
                             const ArPlane *plane,
                             const ArPose *pose,
                             int32_t *out_pose_in_polygon);

/// @}

// === ArPoint methods ===

/// @addtogroup point
/// @{

/// Returns the pose of the point.
/// If ArPoint_getOrientationMode() returns ESTIMATED_SURFACE_NORMAL, the
/// orientation will follow the behavior described in ArHitResult_getHitPose().
/// If ArPoint_getOrientationMode() returns INITIALIZED_TO_IDENTITY, then
/// returns an orientation that is identity or close to identity.
/// @param[in]    session  The ARCore session.
/// @param[in]    point    The point to retrieve the pose of.
/// @param[inout] out_pose An already-allocated ArPose object into which the
/// pose will be stored.
void ArPoint_getPose(const ArSession *session,
                     const ArPoint *point,
                     ArPose *out_pose);

/// Returns the OrientationMode of the point. For @c Point objects created by
/// ArFrame_hitTest().
/// If OrientationMode is ESTIMATED_SURFACE_NORMAL, then normal of the surface
/// centered around the ArPoint was estimated successfully.
///
/// @param[in]    session              The ARCore session.
/// @param[in]    point                The point to retrieve the pose of.
/// @param[inout] out_orientation_mode OrientationMode output result for the
///     the point.
void ArPoint_getOrientationMode(const ArSession *session,
                                const ArPoint *point,
                                ArPointOrientationMode *out_orientation_mode);

/// @}

// === ArAugmentedImage methods ===

/// @addtogroup augmented_image
/// @{

/// Returns the pose of the center of the detected image. The pose's +Y axis
/// will be a normal vector pointing out of the face of the image. The +X and +Z
/// axes represent right and up relative to the image.
///
/// If the tracking state is PAUSED/STOPPED, this returns the pose when the
/// image state was last TRACKING, or the identity pose if the image state has
/// never been TRACKING.
void ArAugmentedImage_getCenterPose(const ArSession *session,
                                    const ArAugmentedImage *augmented_image,
                                    ArPose *out_pose);

/// Retrieves the estimated width, in metres, of the corresponding physical
/// image, as measured along the local X-axis of the coordinate space with
/// origin and axes as defined by ArAugmentedImage_getCenterPose().
///
/// ARCore will attempt to estimate the physical image's width and continuously
/// update this estimate based on its understanding of the world. If the
/// optional physical size is specified in the image database, this estimation
/// process will happen more quickly. However, the estimated size may be
/// different from the originally specified size.
///
/// If the tracking state is PAUSED/STOPPED, this returns the estimated width
/// when the image state was last TRACKING. If the image state has never been
/// TRACKING, this returns 0, even the image has a specified physical size in
/// the image database.
void ArAugmentedImage_getExtentX(const ArSession *session,
                                 const ArAugmentedImage *augmented_image,
                                 float *out_extent_x);

/// Retrieves the estimated height, in metres, of the corresponding physical
/// image, as measured along the local Z-axis of the coordinate space with
/// origin and axes as defined by ArAugmentedImage_getCenterPose().
///
/// ARCore will attempt to estimate the physical image's height and continuously
/// update this estimate based on its understanding of the world. If an optional
/// physical size is specified in the image database, this estimation process
/// will happen more quickly. However, the estimated size may be different from
/// the originally specified size.
///
/// If the tracking state is PAUSED/STOPPED, this returns the estimated height
/// when the image state was last TRACKING. If the image state has never been
/// TRACKING, this returns 0, even the image has a specified physical size in
/// the image database.
void ArAugmentedImage_getExtentZ(const ArSession *session,
                                 const ArAugmentedImage *augmented_image,
                                 float *out_extent_z);

/// Returns the zero-based positional index of this image from its originating
/// image database.
///
/// This index serves as the unique identifier for the image in the database.
void ArAugmentedImage_getIndex(const ArSession *session,
                               const ArAugmentedImage *augmented_image,
                               int32_t *out_index);

/// Returns the name of this image.
///
/// The image name is not guaranteed to be unique.
///
/// This function will allocate memory for the name string, and set
/// *out_augmented_image_name to point to that string. The caller must release
/// the string using ArString_release when the string is no longer needed.
void ArAugmentedImage_acquireName(const ArSession *session,
                                  const ArAugmentedImage *augmented_image,
                                  char **out_augmented_image_name);

/// @}

// === ArAugmentedFace methods ===

/// @addtogroup augmented_face
/// @{

/// Returns a pointer to an array of 3D vertices in (x, y, z) packing. These
/// vertices are relative to the center pose of the face with units in meters.
///
/// The pointer returned by this function is valid until ArTrackable_release()
/// or the next ArSession_update() is called. The application must copy the
/// data if they wish to retain it for longer.
///
/// If the face's tracking state is AR_TRACKING_STATE_PAUSED, then the
/// value of the size of the returned array is 0.
///
/// @param[in]  session                The ARCore session.
/// @param[in]  face                   The face for which to retrieve vertices.
/// @param[out] out_vertices           A pointer to an array of 3D vertices in
///                                    (x, y, z) packing.
/// @param[out] out_number_of_vertices The number of vertices in the mesh. The
///     returned pointer will point to an array of size out_number_of_vertices *
///     3 or @c nullptr if the size is 0.
void ArAugmentedFace_getMeshVertices(const ArSession *session,
                                     const ArAugmentedFace *face,
                                     const float **out_vertices,
                                     int32_t *out_number_of_vertices);

/// Returns a pointer to an array of 3D normals in (x, y, z) packing, where each
/// (x, y, z) is a unit vector of the normal to the surface at each vertex.
/// There is exactly one normal vector for each vertex. These normals are
/// relative to the center pose of the face.
///
/// The pointer returned by this function is valid until ArTrackable_release()
/// or the next ArSession_update() is called. The application must copy the
/// data if they wish to retain it for longer.
///
/// If the face's tracking state is AR_TRACKING_STATE_PAUSED, then the
/// value of the size of the returned array is 0.
///
/// @param[in]  session               The ARCore session.
/// @param[in]  face                  The face for which to retrieve normals.
/// @param[out] out_normals           A pointer to an array of 3D normals in
///                                   (x, y, z) packing.
/// @param[out] out_number_of_normals The number of normals in the mesh. The
///     returned pointer will point to an array of size out_number_of_normals *
///     3, or @c nullptr if the size is 0.
void ArAugmentedFace_getMeshNormals(const ArSession *session,
                                    const ArAugmentedFace *face,
                                    const float **out_normals,
                                    int32_t *out_number_of_normals);

/// Returns a pointer to an array of UV texture coordinates in (u, v) packing.
/// There is a pair of texture coordinates for each vertex. These values
/// never change.
///
/// The pointer returned by this function is valid until ArTrackable_release()
/// or the next ArSession_update() is called. The application must copy the
/// data if they wish to retain it for longer.
///
/// If the face's tracking state is AR_TRACKING_STATE_PAUSED, then the
/// value of the size of the returned array is 0.
///
/// @param[in]  session                 The ARCore session.
/// @param[in]  face                    The face for which to retrieve texture
///                                     coordinates.
/// @param[out] out_texture_coordinates A pointer to an array of UV texture
///                                     coordinates in (u, v) packing.
/// @param[out] out_number_of_texture_coordinates The number of texture
///     coordinates in the mesh. The returned pointer will point to an array of
///     size out_number_of_texture_coordinates * 2, or @c nullptr if the size is
///     0.
void ArAugmentedFace_getMeshTextureCoordinates(
    const ArSession *session,
    const ArAugmentedFace *face,
    const float **out_texture_coordinates,
    int32_t *out_number_of_texture_coordinates);

/// Returns a pointer to an array of triangles indices in consecutive triplets.
///
/// Every three consecutive values are indices that represent a triangle. The
/// vertex position and texture coordinates are mapped by the indices. The front
/// face of each triangle is defined by the face where the vertices are in
/// counter clockwise winding order. These values never change.
///
/// The pointer returned by this function is valid until ArTrackable_release()
/// or the next ArSession_update() is called. The application must copy the
/// data if they wish to retain it for longer.
///
/// If the face's tracking state is AR_TRACKING_STATE_PAUSED, then the
/// value of the size of the returned array is 0.
///
/// @param[in]  session                 The ARCore session.
/// @param[in]  face                    The face for which to retrieve triangle
///                                     indices.
/// @param[out] out_triangle_indices    A pointer to an array of triangle
///                                     indices packed in consecutive triplets.
/// @param[out] out_number_of_triangles The number of triangles in the mesh. The
///     returned pointer will point to an array of size out_number_of_triangles
///     * 3, or @c nullptr if the size is 0.
void ArAugmentedFace_getMeshTriangleIndices(
    const ArSession *session,
    const ArAugmentedFace *face,
    const uint16_t **out_triangle_indices,
    int32_t *out_number_of_triangles);

/// Returns the pose of a face region in world coordinates when the face
/// trackable state is #AR_TRACKING_STATE_TRACKING. When face trackable state is
/// #AR_TRACKING_STATE_PAUSED, the identity pose will be returned.
///
/// @param[in]  session     The ARCore session.
/// @param[in]  face        The face for which to retrieve face region pose.
/// @param[in]  region_type The face region for which to get the pose.
/// @param[out] out_pose    The Pose of the selected region when
///     #AR_TRACKING_STATE_TRACKING, or an identity pose when
///     #AR_TRACKING_STATE_PAUSED.
void ArAugmentedFace_getRegionPose(const ArSession *session,
                                   const ArAugmentedFace *face,
                                   const ArAugmentedFaceRegionType region_type,
                                   ArPose *out_pose);

/// @}

/// @addtogroup augmented_face
/// @{

/// Returns the pose of the center of the face.
///
/// @param[in]    session  The ARCore session.
/// @param[in]    face     The face for which to retrieve center pose.
/// @param[inout] out_pose An already-allocated ArPose object into which the
///     pose will be stored.
void ArAugmentedFace_getCenterPose(const ArSession *session,
                                   const ArAugmentedFace *face,
                                   ArPose *out_pose);

/// @}

// === ArAugmentedImageDatabase methods ===

/// @addtogroup augmented_image_database
/// @{

/// Creates a new empty image database.
void ArAugmentedImageDatabase_create(
    const ArSession *session,
    ArAugmentedImageDatabase **out_augmented_image_database);

/// Creates a new image database from a byte array. The contents of the byte
/// array must have been generated by the command-line database generation tool
/// provided in the SDK, or at runtime from ArAugmentedImageDatabase_serialize.
///
/// Note: this function takes about 10-20ms for a 5MB byte array. Run this in a
/// background thread if this affects your application.
///
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_DATA_INVALID_FORMAT - the bytes are in an invalid format.
/// - #AR_ERROR_DATA_UNSUPPORTED_VERSION - the database is not supported by
///   this version of the SDK.
ArStatus ArAugmentedImageDatabase_deserialize(
    const ArSession *session,
    const uint8_t *database_raw_bytes,
    int64_t database_raw_bytes_size,
    ArAugmentedImageDatabase **out_augmented_image_database);

/// Serializes an image database to a byte array.
///
/// This function will allocate memory for the serialized raw byte array, and
/// set *out_image_database_raw_bytes to point to that byte array. The caller is
/// expected to release the byte array using ArByteArray_release when the byte
/// array is no longer needed.
void ArAugmentedImageDatabase_serialize(
    const ArSession *session,
    const ArAugmentedImageDatabase *augmented_image_database,
    uint8_t **out_image_database_raw_bytes,
    int64_t *out_image_database_raw_bytes_size);

/// Adds a single named image of unknown physical size to an image database,
/// from an array of grayscale pixel values. Returns the zero-based positional
/// index of the image within the image database.
///
/// If the physical size of the image is known, use
/// ArAugmentedImageDatabase_addImageWithPhysicalSize instead, to improve image
/// detection time.
///
/// For images added via ArAugmentedImageDatabase_addImage, ARCore estimates the
/// physical image's size and pose at runtime when the physical image is visible
/// and is being tracked. This extra estimation step will require the user to
/// move their device to view the physical image from different viewpoints
/// before the size and pose of the physical image can be estimated.
///
/// This function takes time to perform non-trivial image processing (20ms -
/// 30ms), and should be run on a background thread.
///
/// The image name is expected to be a null-terminated string in UTF-8 format.
///
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_IMAGE_INSUFFICIENT_QUALITY - image quality is insufficient, e.g.
///   because of lack of features in the image.
ArStatus ArAugmentedImageDatabase_addImage(
    const ArSession *session,
    ArAugmentedImageDatabase *augmented_image_database,
    const char *image_name,
    const uint8_t *image_grayscale_pixels,
    int32_t image_width_in_pixels,
    int32_t image_height_in_pixels,
    int32_t image_stride_in_pixels,
    int32_t *out_index);

/// Adds a single named image to an image database, from an array of grayscale
/// pixel values, along with a positive physical width in meters for this image.
/// Returns the zero-based positional index of the image within the image
/// database.
///
/// If the physical size of the image is not known, use
/// ArAugmentedImageDatabase_addImage instead, at the expense of an increased
/// image detection time.
///
/// For images added via ArAugmentedImageDatabase_addImageWithPhysicalSize,
/// ARCore can estimate the pose of the physical image at runtime as soon as
/// ARCore detects the physical image, without requiring the user to move the
/// device to view the physical image from different viewpoints. Note that
/// ARCore will refine the estimated size and pose of the physical image as it
/// is viewed from different viewpoints.
///
/// This function takes time to perform non-trivial image processing (20ms -
/// 30ms), and should be run on a background thread.
///
/// The image name is expected to be a null-terminated string in UTF-8 format.
///
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_IMAGE_INSUFFICIENT_QUALITY - image quality is insufficient, e.g.
///   because of lack of features in the image.
/// - #AR_ERROR_INVALID_ARGUMENT - image_width_in_meters is <= 0.
ArStatus ArAugmentedImageDatabase_addImageWithPhysicalSize(
    const ArSession *session,
    ArAugmentedImageDatabase *augmented_image_database,
    const char *image_name,
    const uint8_t *image_grayscale_pixels,
    int32_t image_width_in_pixels,
    int32_t image_height_in_pixels,
    int32_t image_stride_in_pixels,
    float image_width_in_meters,
    int32_t *out_index);

/// Returns the number of images in the image database.
void ArAugmentedImageDatabase_getNumImages(
    const ArSession *session,
    const ArAugmentedImageDatabase *augmented_image_database,
    int32_t *out_number_of_images);

/// Releases memory used by an image database.
void ArAugmentedImageDatabase_destroy(
    ArAugmentedImageDatabase *augmented_image_database);

/// @}

// === ArHitResultList methods ===

/// @addtogroup hit
/// @{

/// Creates a hit result list object.
void ArHitResultList_create(const ArSession *session,
                            ArHitResultList **out_hit_result_list);

/// Releases the memory used by a hit result list object, along with all the
/// trackable references it holds.
void ArHitResultList_destroy(ArHitResultList *hit_result_list);

/// Retrieves the number of hit results in this list.
void ArHitResultList_getSize(const ArSession *session,
                             const ArHitResultList *hit_result_list,
                             int32_t *out_size);

/// Copies an indexed entry in the list.  This acquires a reference to any
/// trackable referenced by the item, and releases any reference currently held
/// by the provided result object.
///
/// @param[in]    session           The ARCore session.
/// @param[in]    hit_result_list   The list from which to copy an item.
/// @param[in]    index             Index of the entry to copy.
/// @param[inout] out_hit_result    An already-allocated ArHitResult object into
///     which the result will be copied.
void ArHitResultList_getItem(const ArSession *session,
                             const ArHitResultList *hit_result_list,
                             int32_t index,
                             ArHitResult *out_hit_result);

// === ArHitResult methods ===

/// Allocates an empty hit result object.
void ArHitResult_create(const ArSession *session, ArHitResult **out_hit_result);

/// Releases the memory used by a hit result object, along with any
/// trackable reference it holds.
void ArHitResult_destroy(ArHitResult *hit_result);

/// Returns the distance from the camera to the hit location, in meters.
void ArHitResult_getDistance(const ArSession *session,
                             const ArHitResult *hit_result,
                             float *out_distance);

/// Returns the pose of the intersection between a ray and detected real-world
/// geometry. The position is the location in space where the ray intersected
/// the geometry. The orientation is a best effort to face the user's device,
/// and its exact definition differs depending on the Trackable that was hit.
///
/// ::ArPlane : X+ is perpendicular to the cast ray and parallel to the plane,
/// Y+ points along the plane normal (up, for #AR_PLANE_HORIZONTAL_UPWARD_FACING
/// planes), and Z+ is parallel to the plane, pointing roughly toward the
/// user's device.
///
/// ::ArPoint :
/// Attempt to estimate the normal of the surface centered around the hit test.
/// Surface normal estimation is most likely to succeed on textured surfaces
/// and with camera motion.
/// If ArPoint_getOrientationMode() returns ESTIMATED_SURFACE_NORMAL,
/// then X+ is perpendicular to the cast ray and parallel to the physical
/// surface centered around the hit test, Y+ points along the estimated surface
/// normal, and Z+ points roughly toward the user's device. If
/// ArPoint_getOrientationMode() returns INITIALIZED_TO_IDENTITY, then X+ is
/// perpendicular to the cast ray and points right from the perspective of the
/// user's device, Y+ points up, and Z+ points roughly toward the user's device.
///
/// If you wish to retain the location of this pose beyond the duration of a
/// single frame, create an anchor using ArHitResult_acquireNewAnchor() to save
/// the pose in a physically consistent way.
///
/// @param[in]    session    The ARCore session.
/// @param[in]    hit_result The hit result to retrieve the pose of.
/// @param[inout] out_pose   An already-allocated ArPose object into which the
///     pose will be stored.
void ArHitResult_getHitPose(const ArSession *session,
                            const ArHitResult *hit_result,
                            ArPose *out_pose);

/// Acquires reference to the hit trackable. This call must be paired with a
/// call to ArTrackable_release().
void ArHitResult_acquireTrackable(const ArSession *session,
                                  const ArHitResult *hit_result,
                                  ArTrackable **out_trackable);

/// Creates a new anchor at the hit location. See ArHitResult_getHitPose() for
/// details.  This is equivalent to creating an anchor on the hit trackable at
/// the hit pose.
///
/// @return #AR_SUCCESS or any of:
/// - #AR_ERROR_NOT_TRACKING
/// - #AR_ERROR_SESSION_PAUSED
/// - #AR_ERROR_RESOURCE_EXHAUSTED
/// - #AR_ERROR_DEADLINE_EXCEEDED - hit result must be used before the next call
///     to update().
ArStatus ArHitResult_acquireNewAnchor(ArSession *session,
                                      ArHitResult *hit_result,
                                      ArAnchor **out_anchor);

/// @}

// Utility methods for releasing data.

/// Releases a string acquired using an ARCore API function.
///
/// @param[in] str The string to be released.
void ArString_release(char *str);

/// Releases a byte array created using an ARCore API function.
void ArByteArray_release(uint8_t *byte_array);

#undef AR_DEFINE_ENUM

#ifdef __cplusplus
}
#endif

#endif  // ARCORE_C_API_H_

/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/**
 * \file openhmd.h
 * Main header for OpenHMD public API.
 **/

#ifndef OPENHMD_H
#define OPENHMD_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef DLL_EXPORT
#define OHMD_APIENTRY __cdecl
#define OHMD_APIENTRYDLL __declspec( dllexport )
#else
#ifdef OHMD_STATIC
#define OHMD_APIENTRY __cdecl
#define OHMD_APIENTRYDLL
#else
#define OHMD_APIENTRY __cdecl
#define OHMD_APIENTRYDLL __declspec( dllimport )
#endif
#endif
#else
#define OHMD_APIENTRY
#define OHMD_APIENTRYDLL
#endif

/** Maximum length of a string, including termination, in OpenHMD. */
#define OHMD_STR_SIZE 256

/** Return status codes, used for all functions that can return an error. */
typedef enum {
	OHMD_S_OK = 0,
	OHMD_S_UNKNOWN_ERROR = -1,
	OHMD_S_INVALID_PARAMETER = -2,
	OHMD_S_UNSUPPORTED = -3,
	OHMD_S_INVALID_OPERATION = -4,

	/** OHMD_S_USER_RESERVED and below can be used for user purposes, such as errors within ohmd wrappers, etc. */
	OHMD_S_USER_RESERVED = -16384,
} ohmd_status;

/** A collection of string value information types, used for getting information with ohmd_list_gets(). */
typedef enum {
	OHMD_VENDOR    = 0,
	OHMD_PRODUCT   = 1,
	OHMD_PATH      = 2,
} ohmd_string_value;

/** A collection of string descriptions, used for getting strings with ohmd_gets(). */
typedef enum {
	OHMD_GLSL_DISTORTION_VERT_SRC = 0,
	OHMD_GLSL_DISTORTION_FRAG_SRC = 1,
} ohmd_string_description;

/** Standard controls. Note that this is not an index into the control state. 
	Use OHMD_CONTROL_TYPES to determine what function a control serves at a given index. */
typedef enum {
	OHMD_GENERIC        = 0,
	OHMD_TRIGGER        = 1,
	OHMD_TRIGGER_CLICK  = 2,
	OHMD_SQUEEZE        = 3,
	OHMD_MENU           = 4,
	OHMD_HOME           = 5,
	OHMD_ANALOG_X       = 6,
	OHMD_ANALOG_Y       = 7,
	OHMD_ANALOG_PRESS   = 8,
	OHMD_BUTTON_A       = 9,
	OHMD_BUTTON_B       = 10,
	OHMD_BUTTON_X       = 11,
	OHMD_BUTTON_Y       = 12,
} ohmd_control_hint;

/** Control type. Indicates whether controls are digital or analog. */
typedef enum {
	OHMD_DIGITAL = 0,
	OHMD_ANALOG = 1
} ohmd_control_type;

/** A collection of float value information types, used for getting and setting information with
    ohmd_device_getf() and ohmd_device_setf(). */
typedef enum {
	/** float[4] (get): Absolute rotation of the device, in space, as a quaternion (x, y, z, w). */
	OHMD_ROTATION_QUAT                    =  1,

	/** float[16] (get): A "ready to use" OpenGL style 4x4 matrix with a modelview matrix for the
	    left eye of the HMD. */
	OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX     =  2,
	/** float[16] (get): A "ready to use" OpenGL style 4x4 matrix with a modelview matrix for the
	    right eye of the HMD. */
	OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX    =  3,

	/** float[16] (get): A "ready to use" OpenGL style 4x4 matrix with a projection matrix for the
	    left eye of the HMD. */
	OHMD_LEFT_EYE_GL_PROJECTION_MATRIX    =  4,
	/** float[16] (get): A "ready to use" OpenGL style 4x4 matrix with a projection matrix for the
	    right eye of the HMD. */	
	OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX   =  5,

	/** float[3] (get): A 3-D vector representing the absolute position of the device, in space. */
	OHMD_POSITION_VECTOR                  =  6,

	/** float[1] (get): Physical width of the device screen in metres. */
	OHMD_SCREEN_HORIZONTAL_SIZE           =  7,
	/** float[1] (get): Physical height of the device screen in metres. */
	OHMD_SCREEN_VERTICAL_SIZE             =  8,

	/** float[1] (get): Physical separation of the device lenses in metres. */
	OHMD_LENS_HORIZONTAL_SEPARATION       =  9,
	/** float[1] (get): Physical vertical position of the lenses in metres. */
	OHMD_LENS_VERTICAL_POSITION           = 10,

	/** float[1] (get): Physical field of view for the left eye in degrees. */
	OHMD_LEFT_EYE_FOV                     = 11,
	/** float[1] (get): Physical display aspect ratio for the left eye screen. */
	OHMD_LEFT_EYE_ASPECT_RATIO            = 12,
	/** float[1] (get): Physical field of view for the left right in degrees. */
	OHMD_RIGHT_EYE_FOV                    = 13,
	/** float[1] (get): Physical display aspect ratio for the right eye screen. */
	OHMD_RIGHT_EYE_ASPECT_RATIO           = 14,

	/** float[1] (get, set): Physical interpupillary distance of the user in metres. */
	OHMD_EYE_IPD                          = 15,

	/** float[1] (get, set): Z-far value for the projection matrix calculations (i.e. drawing distance). */
	OHMD_PROJECTION_ZFAR                  = 16,
	/** float[1] (get, set): Z-near value for the projection matrix calculations (i.e. close clipping distance). */
	OHMD_PROJECTION_ZNEAR                 = 17,

	/** float[6] (get): Device specific distortion value. */
	OHMD_DISTORTION_K                     = 18,

	/**
	 * float[10] (set): Perform sensor fusion on values from external sensors.
	 *
	 * Values are: dt (time since last update in seconds) X, Y, Z gyro, X, Y, Z accelerometer and X, Y, Z magnetometer.
	 **/
	OHMD_EXTERNAL_SENSOR_FUSION           = 19,

	/** float[4] (get): Universal shader distortion coefficients (PanoTools model <a,b,c,d>. */
	OHMD_UNIVERSAL_DISTORTION_K           = 20,

	/** float[3] (get): Universal shader aberration coefficients (post warp scaling <r,g,b>. */
	OHMD_UNIVERSAL_ABERRATION_K           = 21,

	/** float[OHMD_CONTROL_COUNT] (get): Get the state of the device's controls. */
	OHMD_CONTROLS_STATE                = 22,

} ohmd_float_value;

/** A collection of int value information types used for getting information with ohmd_device_geti(). */
typedef enum {
	/** int[1] (get, ohmd_geti()): Physical horizontal resolution of the device screen. */
	OHMD_SCREEN_HORIZONTAL_RESOLUTION     =  0,
	/** int[1] (get, ohmd_geti()): Physical vertical resolution of the device screen. */
	OHMD_SCREEN_VERTICAL_RESOLUTION       =  1,

	/** int[1] (get, ohmd_geti()/ohmd_list_geti()): Gets the class of the device. See: ohmd_device_class. */
	OHMD_DEVICE_CLASS                     =  2,
	/** int[1] (get, ohmd_geti()/ohmd_list_geti()): Gets the flags of the device. See: ohmd_device_flags. */
	OHMD_DEVICE_FLAGS                     =  3,

	/** int[1] (get, ohmd_geti()): Get the number of analog and digital controls of the device. */
	OHMD_CONTROL_COUNT                    =  4,

	/** int[OHMD_CONTROL_COUNT] (get, ohmd_geti()): Get whether controls are digital or analog. */
	OHMD_CONTROLS_HINTS   		          =  5,
	
	/** int[OHMD_CONTROL_COUNT] (get, ohmd_geti()): Get what function controls serve. */
	OHMD_CONTROLS_TYPES                   =  6,
} ohmd_int_value;

/** A collection of data information types used for setting information with ohmd_set_data(). */
typedef enum {
	/** void* (set): Set void* data for use in the internal drivers. */
	OHMD_DRIVER_DATA		= 0,
	/**
	 * ohmd_device_properties* (set):
	 * Set the device properties based on the ohmd_device_properties struct for use in the internal drivers.
	 *
	 * This can be used to fill in information about the device internally, such as Android, or for setting profiles.
	 **/
	OHMD_DRIVER_PROPERTIES	= 1,
} ohmd_data_value;

typedef enum {
	/** int[1] (set, default: 1): Set this to 0 to prevent OpenHMD from creating background threads to do automatic device ticking.
	    Call ohmd_update(); must be called frequently, at least 10 times per second, if the background threads are disabled. */
	OHMD_IDS_AUTOMATIC_UPDATE = 0,
} ohmd_int_settings;

/** Button states for digital input events. */
typedef enum {
	/** Button was pressed. */
	OHMD_BUTTON_DOWN = 0,
	/** Button was released. */
	OHMD_BUTTON_UP   = 1
} ohmd_button_state;

/** Device classes. */
typedef enum 
{
	/** HMD device. */
	OHMD_DEVICE_CLASS_HMD        = 0,
	/** Controller device. */
	OHMD_DEVICE_CLASS_CONTROLLER = 1,
	/** Generic tracker device. */
	OHMD_DEVICE_CLASS_GENERIC_TRACKER = 2,
} ohmd_device_class;

/** Device flags. */
typedef enum
{
	/** Device is a null (dummy) device. */
	OHMD_DEVICE_FLAGS_NULL_DEVICE         = 1,
	OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING = 2,
	OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING = 4,
	OHMD_DEVICE_FLAGS_LEFT_CONTROLLER     = 8,
	OHMD_DEVICE_FLAGS_RIGHT_CONTROLLER    = 16,
} ohmd_device_flags;

/** An opaque pointer to a context structure. */
typedef struct ohmd_context ohmd_context;

/** An opaque pointer to a structure representing a device, such as an HMD. */
typedef struct ohmd_device ohmd_device;

/** An opaque pointer to a structure representing arguments for a device. */
typedef struct ohmd_device_settings ohmd_device_settings;

/**
 * Create an OpenHMD context.
 *
 * @return a pointer to an allocated ohmd_context on success or NULL if it fails.
 **/
OHMD_APIENTRYDLL ohmd_context* OHMD_APIENTRY ohmd_ctx_create(void);

/**
 * Destroy an OpenHMD context.
 *
 * ohmd_ctx_destroy de-initializes and de-allocates an OpenHMD context allocated with ohmd_ctx_create.
 * All devices associated with the context are automatically closed.
 *
 * @param ctx The context to destroy.
 **/
OHMD_APIENTRYDLL void OHMD_APIENTRY ohmd_ctx_destroy(ohmd_context* ctx);

/**
 * Get the last error as a human readable string.
 *
 * If a function taking a context as an argument (ohmd_context "methods") returns non-successfully,
 * a human readable error message describing what went wrong can be retrieved with this function.
 *
 * @param ctx The context to retrieve the error message from.
 * @return a pointer to the error message.
 **/
OHMD_APIENTRYDLL const char* OHMD_APIENTRY ohmd_ctx_get_error(ohmd_context* ctx);

/**
 * Update a context.
 *
 * Update the values for the devices handled by a context.
 *
 * If background threads are disabled, this performs tasks like pumping events from the device. The exact details 
 * are up to the driver but try to call it quite frequently.
 * Once per frame in a "game loop" should be sufficient.
 * If OpenHMD is handled in a background thread in your program, calling ohmd_ctx_update and then sleeping for 10-20 ms
 * is recommended.
 *
 * @param ctx The context that needs updating.
 **/
OHMD_APIENTRYDLL void OHMD_APIENTRY ohmd_ctx_update(ohmd_context* ctx);

/**
 * Probe for devices.
 *
 * Probes for and enumerates supported devices attached to the system.
 *
 * @param ctx A context with no currently open devices.
 * @return the number of devices found on the system.
 **/
OHMD_APIENTRYDLL int OHMD_APIENTRY ohmd_ctx_probe(ohmd_context* ctx);

/**
 * Get string from openhmd.
 *
 * Gets a string from OpenHMD. This is where non-device specific strings reside.
 * This is where the distortion shader sources can be retrieved.
 *
 * @param type The name of the string to fetch. One of OHMD_GLSL_DISTORTION_FRAG_SRC, and OHMD_GLSL_DISTORTION_FRAG_SRC.
 * @param out The location to return a const char*
 * @return 0 on success, <0 on failure.
 **/
OHMD_APIENTRYDLL int ohmd_gets(ohmd_string_description type, const char** out);

/**
 * Get device description from enumeration list index.
 *
 * Gets a human readable device description string from a zero indexed enumeration index
 * between 0 and (max - 1), where max is the number ohmd_ctx_probe returned
 * (i.e. if ohmd_ctx_probe returns 3, valid indices are 0, 1 and 2).
 * The function can return three types of data. The vendor name, the product name and
 * a driver specific path where the device is attached.
 *
 * ohmd_ctx_probe must be called before calling ohmd_list_gets.
 *
 * @param ctx A (probed) context.
 * @param index An index, between 0 and the value returned from ohmd_ctx_probe.
 * @param type The type of data to fetch. One of OHMD_VENDOR, OHMD_PRODUCT and OHMD_PATH.
 * @return a string with a human readable device name.
 **/
OHMD_APIENTRYDLL const char* OHMD_APIENTRY ohmd_list_gets(ohmd_context* ctx, int index, ohmd_string_value type);


/**
 * Get integer value from enumeration list index.
 *
 * 
 *
 * ohmd_ctx_probe must be called before calling ohmd_list_gets.
 *
 * @param ctx A (probed) context.
 * @param index An index, between 0 and the value returned from ohmd_ctx_probe.
 * @param type What type of value to retrieve, ohmd_int_value section for more information.
 * @return 0 on success, <0 on failure.
 **/
OHMD_APIENTRYDLL int OHMD_APIENTRY ohmd_list_geti(ohmd_context* ctx, int index, ohmd_int_value type, int* out);

/**
 * Open a device.
 *
 * Opens a device from a zero indexed enumeration index between 0 and (max - 1)
 * where max is the number ohmd_ctx_probe returned (i.e. if ohmd_ctx_probe returns 3,
 * valid indices are 0, 1 and 2).
 *
 * ohmd_ctx_probe must be called before calling ohmd_list_open_device.
 *
 * @param ctx A (probed) context.
 * @param index An index, between 0 and the value returned from ohmd_ctx_probe.
 * @return a pointer to an ohmd_device, which represents a hardware device, such as an HMD.
 **/
OHMD_APIENTRYDLL ohmd_device* OHMD_APIENTRY ohmd_list_open_device(ohmd_context* ctx, int index);

/**
 * Open a device with additional settings provided.
 *
 * Opens a device from a zero indexed enumeration index between 0 and (max - 1)
 * where max is the number ohmd_ctx_probe returned (i.e. if ohmd_ctx_probe returns 3,
 * valid indices are 0, 1 and 2).
 *
 * ohmd_ctx_probe must be called before calling ohmd_list_open_device.
 *
 * @param ctx A (probed) context.
 * @param index An index, between 0 and the value returned from ohmd_ctx_probe.
 * @param settings A pointer to a device settings struct.
 * @return a pointer to an ohmd_device, which represents a hardware device, such as an HMD.
 **/
OHMD_APIENTRYDLL ohmd_device* OHMD_APIENTRY ohmd_list_open_device_s(ohmd_context* ctx, int index, ohmd_device_settings* settings);

/**
 * Specify int settings in a device settings struct.
 *
 * @param settings The device settings struct to set values to.
 * @param key The specefic setting you wish to set.
 * @param value A pointer to an int or int array (containing the expected number of elements) with the value(s) you wish to set.
 **/
OHMD_APIENTRYDLL ohmd_status OHMD_APIENTRY ohmd_device_settings_seti(ohmd_device_settings* settings, ohmd_int_settings key, const int* val);

/**
 * Create a device settings instance.
 *
 * @param ctx A pointer to a valid ohmd_context.
 * @return a pointer to an allocated ohmd_context on success or NULL if it fails.
 **/
OHMD_APIENTRYDLL ohmd_device_settings* OHMD_APIENTRY ohmd_device_settings_create(ohmd_context* ctx);

/**
 * Destroy a device settings instance.
 *
 * @param ctx The device settings instance to destroy.
 **/
OHMD_APIENTRYDLL void OHMD_APIENTRY ohmd_device_settings_destroy(ohmd_device_settings* settings);

/**
 * Close a device.
 *
 * Closes a device opened by ohmd_list_open_device. Note that ohmd_ctx_destroy automatically closes any open devices
 * associated with the context being destroyed.
 *
 * @param device The open device.
 * @return 0 on success, <0 on failure.
 **/
OHMD_APIENTRYDLL int OHMD_APIENTRY ohmd_close_device(ohmd_device* device);

/**
 * Get a floating point value from a device.
 *
 *
 * @param device An open device to retrieve the value from.
 * @param type What type of value to retrieve, see ohmd_float_value section for more information.
 * @param[out] out A pointer to a float, or float array where the retrieved value should be written.
 * @return 0 on success, <0 on failure.
 **/
OHMD_APIENTRYDLL int OHMD_APIENTRY ohmd_device_getf(ohmd_device* device, ohmd_float_value type, float* out);

/**
 * Set a floating point value for a device.
 *
 * @param device An open device to set the value in.
 * @param type What type of value to set, see ohmd_float_value section for more information.
 * @param in A pointer to a float, or float array where the new value is stored.
 * @return 0 on success, <0 on failure.
 **/
OHMD_APIENTRYDLL int OHMD_APIENTRY ohmd_device_setf(ohmd_device* device, ohmd_float_value type, const float* in);

/**
 * Get an integer value from a device.
 *
 * @param device An open device to retrieve the value from.
 * @param type What type of value to retrieve, ohmd_int_value section for more information.
 * @param[out] out A pointer to an integer, or integer array where the retrieved value should be written.
 * @return 0 on success, <0 on failure.
 **/
OHMD_APIENTRYDLL int OHMD_APIENTRY ohmd_device_geti(ohmd_device* device, ohmd_int_value type, int* out);

/**
 * Set an integer value for a device.
 *
 * @param device An open device to set the value in.
 * @param type What type of value to set, see ohmd_float_value section for more information.
 * @param in A pointer to a int, or int array where the new value is stored.
 * @return 0 on success, <0 on failure.
 **/
OHMD_APIENTRYDLL int OHMD_APIENTRY ohmd_device_seti(ohmd_device* device, ohmd_int_value type, const int* in);

/**
 * Set an void* data value for a device.
 *
 * @param device An open device to set the value in.
 * @param type What type of value to set, see ohmd_float_value section for more information.
 * @param in A pointer to the void* casted object.
 * @return 0 on success, <0 on failure.
 **/
OHMD_APIENTRYDLL int OHMD_APIENTRY ohmd_device_set_data(ohmd_device* device, ohmd_data_value type, const void* in);

#ifdef __cplusplus
}
#endif

#endif

/*
 * libopenmpt_ext.h
 * ----------------
 * Purpose: libopenmpt public c interface for libopenmpt extensions
 * Notes  :
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#ifndef LIBOPENMPT_EXT_H
#define LIBOPENMPT_EXT_H

#include "libopenmpt_config.h"
#include "libopenmpt.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \page libopenmpt_ext_c_overview libopenmpt_ext C API
 *
 * libopenmpt_ext is included in all builds by default.
 *
 * \section libopenmpt-ext-c-detailed Detailed documentation
 *
 * \ref libopenmpt_ext_c
 *
 */

/*! \defgroup libopenmpt_ext_c libopenmpt_ext C */

/*! \addtogroup libopenmpt_ext_c
 * @{
 */

/*! \brief Opaque type representing a libopenmpt extension module
 */
typedef struct openmpt_module_ext openmpt_module_ext;

/*! \brief Construct an openmpt_module_ext
 *
 * \param stream_callbacks Input stream callback operations.
 * \param stream Input stream to load the module from.
 * \param logfunc Logging function where warning and errors are written. The logging function may be called throughout the lifetime of openmpt_module_ext. May be NULL.
 * \param loguser User-defined data associated with this module. This value will be passed to the logging callback function (logfunc)
 * \param errfunc Error function to define error behaviour. May be NULL.
 * \param erruser Error function user context. Used to pass any user-defined data associated with this module to the logging function.
 * \param error Pointer to an integer where an error may get stored. May be NULL.
 * \param error_message Pointer to a string pointer where an error message may get stored. May be NULL.
 * \param ctls A map of initial ctl values, see openmpt_module_get_ctls.
 * \return A pointer to the constructed openmpt_module_ext, or NULL on failure.
 * \remarks The input data can be discarded after an openmpt_module_ext has been constructed successfully.
 * \sa openmpt_stream_callbacks
 * \sa \ref libopenmpt_c_fileio
 * \since 0.3.0
 */
LIBOPENMPT_API openmpt_module_ext * openmpt_module_ext_create( openmpt_stream_callbacks stream_callbacks, void * stream, openmpt_log_func logfunc, void * loguser, openmpt_error_func errfunc, void * erruser, int * error, const char * * error_message, const openmpt_module_initial_ctl * ctls );

/*! \brief Construct an openmpt_module_ext
 *
 * \param filedata Data to load the module from.
 * \param filesize Amount of data available.
 * \param logfunc Logging function where warning and errors are written. The logging function may be called throughout the lifetime of openmpt_module_ext.
 * \param loguser User-defined data associated with this module. This value will be passed to the logging callback function (logfunc)
 * \param errfunc Error function to define error behaviour. May be NULL.
 * \param erruser Error function user context. Used to pass any user-defined data associated with this module to the logging function.
 * \param error Pointer to an integer where an error may get stored. May be NULL.
 * \param error_message Pointer to a string pointer where an error message may get stored. May be NULL.
 * \param ctls A map of initial ctl values, see openmpt_module_get_ctls.
 * \return A pointer to the constructed openmpt_module_ext, or NULL on failure.
 * \remarks The input data can be discarded after an openmpt_module_ext has been constructed successfully.
 * \sa \ref libopenmpt_c_fileio
 * \since 0.3.0
 */
LIBOPENMPT_API openmpt_module_ext * openmpt_module_ext_create_from_memory( const void * filedata, size_t filesize, openmpt_log_func logfunc, void * loguser, openmpt_error_func errfunc, void * erruser, int * error, const char * * error_message, const openmpt_module_initial_ctl * ctls );

/*! \brief Unload a previously created openmpt_module_ext from memory.
 *
 * \param mod_ext The module to unload.
 */
LIBOPENMPT_API void openmpt_module_ext_destroy( openmpt_module_ext * mod_ext );

/*! \brief Retrieve the openmpt_module handle from an openmpt_module_ext handle.
 *
 * \param mod_ext The extension module handle to convert
 * \return An equivalent openmpt_module handle to pass to standard libopenmpt functions 
 * \since 0.3.0
 */
LIBOPENMPT_API openmpt_module * openmpt_module_ext_get_module( openmpt_module_ext * mod_ext );

/*! Retrieve a libopenmpt extension.
 *
 * \param mod_ext The module handle to work on.
 * \param interface_id The name of the extension interface to retrieve (e.g. LIBOPENMPT_EXT_C_INTERFACE_PATTERN_VIS).
 * \param interface Appropriate structure of interface function pointers which is to be filled by this function (e.g. a pointer to a openmpt_module_ext_interface_pattern_vis structure).
 * \param interface_size Size of the interface's structure of function pointers (e.g. sizeof(openmpt_module_ext_interface_pattern_vis)).
 * \return 1 on success, 0 if the interface was not found.
 * \since 0.3.0
 */
LIBOPENMPT_API int openmpt_module_ext_get_interface( openmpt_module_ext * mod_ext, const char * interface_id, void * interface, size_t interface_size );



#ifndef LIBOPENMPT_EXT_C_INTERFACE_PATTERN_VIS
#define LIBOPENMPT_EXT_C_INTERFACE_PATTERN_VIS "pattern_vis"
#endif

/*! Pattern command type */
#define OPENMPT_MODULE_EXT_INTERFACE_PATTERN_VIS_EFFECT_TYPE_UNKNOWN 0
#define OPENMPT_MODULE_EXT_INTERFACE_PATTERN_VIS_EFFECT_TYPE_GENERAL 1
#define OPENMPT_MODULE_EXT_INTERFACE_PATTERN_VIS_EFFECT_TYPE_GLOBAL  2
#define OPENMPT_MODULE_EXT_INTERFACE_PATTERN_VIS_EFFECT_TYPE_VOLUME  3
#define OPENMPT_MODULE_EXT_INTERFACE_PATTERN_VIS_EFFECT_TYPE_PANNING 4
#define OPENMPT_MODULE_EXT_INTERFACE_PATTERN_VIS_EFFECT_TYPE_PITCH   5

typedef struct openmpt_module_ext_interface_pattern_vis {
	/*! Get pattern command type for pattern highlighting
	 *
	 * \param mod_ext The module handle to work on.
	 * \param pattern The pattern whose data should be retrieved.
	 * \param row The row from which the data should be retrieved.
	 * \param channel The channel from which the data should be retrieved.
	 * \return The command type in the effect column at the given pattern position (see OPENMPT_MODULE_EXT_INTERFACE_PATTERN_VIS_EFFECT_TYPE_*)
	 * \sa openmpt_module_ext_interface_pattern_vis::get_pattern_row_channel_volume_effect_type
	 */
	int ( * get_pattern_row_channel_volume_effect_type ) ( openmpt_module_ext * mod_ext, int32_t pattern, int32_t row, int32_t channel );

	/*! Get pattern command type for pattern highlighting
	 *
	 * \param mod_ext The module handle to work on.
	 * \param pattern The pattern whose data should be retrieved.
	 * \param row The row from which the data should be retrieved.
	 * \param channel The channel from which the data should be retrieved.
	 * \return The command type in the effect column at the given pattern position (see OPENMPT_MODULE_EXT_INTERFACE_PATTERN_VIS_EFFECT_TYPE_*)
	 * \sa openmpt_module_ext_interface_pattern_vis::get_pattern_row_channel_volume_effect_type
	 */
	int ( * get_pattern_row_channel_effect_type ) ( openmpt_module_ext * mod_ext, int32_t pattern, int32_t row, int32_t channel );
} openmpt_module_ext_interface_pattern_vis;



#ifndef LIBOPENMPT_EXT_C_INTERFACE_INTERACTIVE
#define LIBOPENMPT_EXT_C_INTERFACE_INTERACTIVE "interactive"
#endif

typedef struct openmpt_module_ext_interface_interactive {
	/*! Set the current ticks per row (speed)
	 *
	 * \param mod_ext The module handle to work on.
	 * \param speed The new tick count in range [1, 65535].
	 * \return 1 on success, 0 on failure.
	 * \remarks The tick count may be reset by pattern commands at any time.
	 * \sa openmpt_module_get_current_speed
	 */
	int ( * set_current_speed ) ( openmpt_module_ext * mod_ext, int32_t speed );

	/*! Set the current module tempo
	 *
	 * \param mod_ext The module handle to work on.
	 * \param tempo The new tempo in range [32, 512]. The exact meaning of the value depends on the tempo mode used by the module.
	 * \return 1 on success, 0 on failure.
	 * \remarks The tempo may be reset by pattern commands at any time. Use openmpt_module_ext_interface_interactive::set_tempo_factor to apply a tempo factor that is independent of pattern commands.
	 * \sa openmpt_module_get_current_tempo
	 */
	int ( * set_current_tempo ) ( openmpt_module_ext * mod_ext, int32_t tempo );

	/*! Set the current module tempo factor without affecting playback pitch
	 *
	 * \param mod_ext The module handle to work on.
	 * \param factor The new tempo factor in range ]0.0, 4.0] - 1.0 means unmodified tempo.
	 * \return 1 on success, 0 on failure.
	 * \remarks Modifying the tempo without applying the same pitch factor using openmpt_module_ext_interface_interactive::set_pitch_factor may cause rhythmic samples (e.g. drum loops) to go out of sync.
	 * \sa openmpt_module_ext_interface_interactive::get_tempo_factor
	 */
	int ( * set_tempo_factor ) ( openmpt_module_ext * mod_ext, double factor );

	/*! Gets the current module tempo factor
	 *
	 * \param mod_ext The module handle to work on.
	 * \return The current tempo factor.
	 * \sa openmpt_module_ext_interface_interactive::set_tempo_factor
	 */
	double ( * get_tempo_factor ) ( openmpt_module_ext * mod_ext );

	/*! Set the current module pitch factor without affecting playback speed
	 *
	 * \param mod_ext The module handle to work on.
	 * \param factor The new pitch factor in range ]0.0, 4.0] - 1.0 means unmodified pitch.
	 * \return 1 on success, 0 on failure.
	 * \remarks Modifying the pitch without applying the the same tempo factor using openmpt_module_ext_interface_interactive::set_tempo_factor may cause rhythmic samples (e.g. drum loops) to go out of sync.
	 * \remarks To shift the pich by `n` semitones, the parameter can be calculated as follows: `pow( 2.0, n / 12.0 )`
	 * \sa openmpt_module_ext_interface_interactive::get_pitch_factor
	 */
	int ( * set_pitch_factor ) ( openmpt_module_ext * mod_ext, double factor );

	/*! Gets the current module pitch factor
	 *
	 * \param mod_ext The module handle to work on.
	 * \return The current pitch factor.
	 * \sa openmpt_module_ext_interface_interactive::set_pitch_factor
	*/
	double ( * get_pitch_factor ) ( openmpt_module_ext * mod_ext );

	/*! Set the current global volume
	 *
	 * \param mod_ext The module handle to work on.
	 * \param volume The new global volume in range [0.0, 1.0]
	 * \return 1 on success, 0 on failure.
	 * \remarks The global volume may be reset by pattern commands at any time. Use openmpt_module_set_render_param to apply a global overall volume factor that is independent of pattern commands.
	 * \sa openmpt_module_ext_interface_interactive::get_global_volume
	 */
	int ( * set_global_volume ) ( openmpt_module_ext * mod_ext, double volume );

	/*! Get the current global volume
	 *
	 * \param mod_ext The module handle to work on.
	 * \return The current global volume in range [0.0, 1.0]
	 * \sa openmpt_module_ext_interface_interactive::set_global_volume
	 */
	double ( * get_global_volume ) ( openmpt_module_ext * mod_ext );

	/*! Set the current channel volume for a channel
	 *
	 * \param mod_ext The module handle to work on.
	 * \param channel The channel whose volume should be set, in range [0, openmpt_module_get_num_channels()[
	 * \param volume The new channel volume in range [0.0, 1.0]
	 * \return 1 on success, 0 on failure (channel out of range).
	 * \remarks The channel volume may be reset by pattern commands at any time.
	 * \sa openmpt_module_ext_interface_interactive::get_channel_volume
	 */
	int ( * set_channel_volume ) ( openmpt_module_ext * mod_ext, int32_t channel, double volume );

	/*! Get the current channel volume for a channel
	 *
	 * \param mod_ext The module handle to work on.
	 * \param channel The channel whose volume should be retrieved, in range [0, openmpt_module_get_num_channels()[
	 * \return The current channel volume in range [0.0, 1.0]
	 * \sa openmpt_module_ext_interface_interactive::set_channel_volume
	 */
	double ( * get_channel_volume ) ( openmpt_module_ext * mod_ext, int32_t channel );

	/*! Set the current mute status for a channel
	 *
	 * \param mod_ext The module handle to work on.
	 * \param channel The channel whose mute status should be set, in range [0, openmpt_module_get_num_channels()[
	 * \param mute The new mute status. true is muted, false is unmuted.
	 * \return 1 on success, 0 on failure (channel out of range).
	 * \sa openmpt_module_ext_interface_interactive::get_channel_mute_status
	 */
	int ( * set_channel_mute_status ) ( openmpt_module_ext * mod_ext, int32_t channel, int mute );

	/*! Get the current mute status for a channel
	 *
	 * \param mod_ext The module handle to work on.
	 * \param channel The channel whose mute status should be retrieved, in range [0, openmpt_module_get_num_channels()[
	 * \return The current channel mute status. 1 is muted, 0 is unmuted, -1 means the instrument was out of range
	 * \sa openmpt_module_ext_interface_interactive::set_channel_mute_status
	 */
	int ( * get_channel_mute_status ) ( openmpt_module_ext * mod_ext, int32_t channel );

	/*! Set the current mute status for an instrument
	 *
	 * \param mod_ext The module handle to work on.
	 * \param instrument The instrument whose mute status should be set, in range [0, openmpt_module_get_num_instruments()[ if openmpt_module_get_num_instruments is not 0, otherwise in [0, openmpt_module_get_num_samples()[
	 * \param mute The new mute status. true is muted, false is unmuted.
	 * \return 1 on success, 0 on failure (instrument out of range).
	 * \sa openmpt_module_ext_interface_interactive::get_instrument_mute_status
	 */
	int ( * set_instrument_mute_status ) ( openmpt_module_ext * mod_ext, int32_t instrument, int mute );

	/*! Get the current mute status for an instrument
	 *
	 * \param mod_ext The module handle to work on.
	 * \param instrument The instrument whose mute status should be retrieved, in range [0, openmpt_module_get_num_instruments()[ if openmpt_module_get_num_instruments is not 0, otherwise in [0, openmpt_module_get_num_samples()[
	 * \return The current instrument mute status. 1 is muted, 0 is unmuted, -1 means the instrument was out of range
	 * \sa openmpt_module_ext_interface_interactive::set_instrument_mute_status
	 */
	int ( * get_instrument_mute_status ) ( openmpt_module_ext * mod_ext, int32_t instrument );

	/*! Play a note using the specified instrument
	 *
	 * \param mod_ext The module handle to work on.
	 * \param instrument The instrument that should be played, in range [0, openmpt_module_get_num_instruments()[ if openmpt_module_get_num_instruments is not 0, otherwise in [0, openmpt_module_get_num_samples()[
	 * \param note The note to play, in rage [0, 119]. 60 is the middle C.
	 * \param volume The volume at which the note should be triggered, in range [0.0, 1.0]
	 * \param panning The panning position at which the note should be triggered, in range [-1.0, 1.0], 0.0 is center.
	 * \return The channel on which the note is played. This can pe be passed to openmpt_module_ext_interface_interactive::stop_note to stop the note. -1 means that no channel could be allocated and the note is not played.
	 * \sa openmpt_module_ext_interface_interactive::stop_note
	 */
	int32_t ( * play_note ) ( openmpt_module_ext * mod_ext, int32_t instrument, int32_t note, double volume, double panning );

	/*! Stop the note playing on the specified channel
	 *
	 * \param mod_ext The module handle to work on.
	 * \param channel The channel on which the note should be stopped.
	 * \return 1 on success, 0 on failure (channel out of range).
	 * \sa openmpt_module_ext_interface_interactive::play_note
	 */
	int ( * stop_note ) ( openmpt_module_ext * mod_ext, int32_t channel );
} openmpt_module_ext_interface_interactive;



/* add stuff here */



#ifdef __cplusplus
}
#endif

/*!
 * @}
 */

#endif /* LIBOPENMPT_EXT_H */


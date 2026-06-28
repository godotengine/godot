/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2001             *
 * by the Xiph.Org Foundation https://xiph.org/                     *
 *                                                                  *
 ********************************************************************

 function: vorbis encode-engine setup

 ********************************************************************/

/** \file
 * Libvorbisenc is a convenient API for setting up an encoding
 * environment using libvorbis. Libvorbisenc encapsulates the
 * actions needed to set up the encoder properly.
 */

#ifndef _OV_ENC_H_
#define _OV_ENC_H_

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#include "codec.h"

/**
 * This is the primary function within libvorbisenc for setting up managed
 * bitrate modes.
 *
 * Before this function is called, the \ref vorbis_info
 * struct should be initialized by using vorbis_info_init() from the libvorbis
 * API.  After encoding, vorbis_info_clear() should be called.
 *
 * The max_bitrate, nominal_bitrate, and min_bitrate settings are used to set
 * constraints for the encoded file.  This function uses these settings to
 * select the appropriate encoding mode and set it up.
 *
 * \param vi               Pointer to an initialized \ref vorbis_info struct.
 * \param channels         The number of channels to be encoded.
 * \param rate             The sampling rate of the source audio.
 * \param max_bitrate      Desired maximum bitrate (limit). -1 indicates unset.
 * \param nominal_bitrate  Desired average, or central, bitrate. -1 indicates unset.
 * \param min_bitrate      Desired minimum bitrate. -1 indicates unset.
 *
 * \return Zero for success, and negative values for failure.
 *
 * \retval 0          Success.
 * \retval OV_EFAULT  Internal logic fault; indicates a bug or heap/stack corruption.
 * \retval OV_EINVAL  Invalid setup request, eg, out of range argument.
 * \retval OV_EIMPL   Unimplemented mode; unable to comply with bitrate request.
 */
extern int vorbis_encode_init(vorbis_info *vi,
                              long channels,
                              long rate,

                              long max_bitrate,
                              long nominal_bitrate,
                              long min_bitrate);

/**
 * This function performs step-one of a three-step bitrate-managed encode
 * setup.  It functions similarly to the one-step setup performed by \ref
 * vorbis_encode_init but allows an application to make further encode setup
 * tweaks using \ref vorbis_encode_ctl before finally calling \ref
 * vorbis_encode_setup_init to complete the setup process.
 *
 * Before this function is called, the \ref vorbis_info struct should be
 * initialized by using vorbis_info_init() from the libvorbis API.  After
 * encoding, vorbis_info_clear() should be called.
 *
 * The max_bitrate, nominal_bitrate, and min_bitrate settings are used to set
 * constraints for the encoded file.  This function uses these settings to
 * select the appropriate encoding mode and set it up.
 *
 * \param vi                Pointer to an initialized vorbis_info struct.
 * \param channels          The number of channels to be encoded.
 * \param rate              The sampling rate of the source audio.
 * \param max_bitrate       Desired maximum bitrate (limit). -1 indicates unset.
 * \param nominal_bitrate   Desired average, or central, bitrate. -1 indicates unset.
 * \param min_bitrate       Desired minimum bitrate. -1 indicates unset.
 *
 * \return Zero for success, and negative for failure.
 *
 * \retval 0           Success
 * \retval OV_EFAULT   Internal logic fault; indicates a bug or heap/stack corruption.
 * \retval OV_EINVAL   Invalid setup request, eg, out of range argument.
 * \retval OV_EIMPL    Unimplemented mode; unable to comply with bitrate request.
 */
extern int vorbis_encode_setup_managed(vorbis_info *vi,
                                       long channels,
                                       long rate,

                                       long max_bitrate,
                                       long nominal_bitrate,
                                       long min_bitrate);

/**
 * This function performs step-one of a three-step variable bitrate
 * (quality-based) encode setup.  It functions similarly to the one-step setup
 * performed by \ref vorbis_encode_init_vbr() but allows an application to
 * make further encode setup tweaks using \ref vorbis_encode_ctl() before
 * finally calling \ref vorbis_encode_setup_init to complete the setup
 * process.
 *
 * Before this function is called, the \ref vorbis_info struct should be
 * initialized by using \ref vorbis_info_init() from the libvorbis API.  After
 * encoding, vorbis_info_clear() should be called.
 *
 * \param vi        Pointer to an initialized vorbis_info struct.
 * \param channels  The number of channels to be encoded.
 * \param rate      The sampling rate of the source audio.
 * \param quality   Desired quality level, currently from -0.1 to 1.0 (lo to hi).
 *
 * \return Zero for success, and negative values for failure.
 *
 * \retval  0          Success
 * \retval  OV_EFAULT  Internal logic fault; indicates a bug or heap/stack corruption.
 * \retval  OV_EINVAL  Invalid setup request, eg, out of range argument.
 * \retval  OV_EIMPL   Unimplemented mode; unable to comply with quality level request.
 */
extern int vorbis_encode_setup_vbr(vorbis_info *vi,
                                  long channels,
                                  long rate,

                                  float quality
                                  );

/**
 * This is the primary function within libvorbisenc for setting up variable
 * bitrate ("quality" based) modes.
 *
 *
 * Before this function is called, the vorbis_info struct should be
 * initialized by using vorbis_info_init() from the libvorbis API. After
 * encoding, vorbis_info_clear() should be called.
 *
 * \param vi           Pointer to an initialized vorbis_info struct.
 * \param channels     The number of channels to be encoded.
 * \param rate         The sampling rate of the source audio.
 * \param base_quality Desired quality level, currently from -0.1 to 1.0 (lo to hi).
 *
 *
 * \return Zero for success, or a negative number for failure.
 *
 * \retval 0           Success
 * \retval OV_EFAULT   Internal logic fault; indicates a bug or heap/stack corruption.
 * \retval OV_EINVAL   Invalid setup request, eg, out of range argument.
 * \retval OV_EIMPL    Unimplemented mode; unable to comply with quality level request.
 */
extern int vorbis_encode_init_vbr(vorbis_info *vi,
                                  long channels,
                                  long rate,

                                  float base_quality
                                  );

/**
 * This function performs the last stage of three-step encoding setup, as
 * described in the API overview under managed bitrate modes.
 *
 * Before this function is called, the \ref vorbis_info struct should be
 * initialized by using vorbis_info_init() from the libvorbis API, one of
 * \ref vorbis_encode_setup_managed() or \ref vorbis_encode_setup_vbr() called to
 * initialize the high-level encoding setup, and \ref vorbis_encode_ctl()
 * called if necessary to make encoding setup changes.
 * vorbis_encode_setup_init() finalizes the highlevel encoding structure into
 * a complete encoding setup after which the application may make no further
 * setup changes.
 *
 * After encoding, vorbis_info_clear() should be called.
 *
 * \param vi Pointer to an initialized \ref vorbis_info struct.
 *
 * \return Zero for success, and negative values for failure.
 *
 * \retval  0           Success.
 * \retval  OV_EFAULT  Internal logic fault; indicates a bug or heap/stack corruption.
 *
 * \retval OV_EINVAL   Attempt to use vorbis_encode_setup_init() without first
 * calling one of vorbis_encode_setup_managed() or vorbis_encode_setup_vbr() to
 * initialize the high-level encoding setup
 *
 */
extern int vorbis_encode_setup_init(vorbis_info *vi);

/**
 * This function implements a generic interface to miscellaneous encoder
 * settings similar to the classic UNIX 'ioctl()' system call.  Applications
 * may use vorbis_encode_ctl() to query or set bitrate management or quality
 * mode details by using one of several \e request arguments detailed below.
 * vorbis_encode_ctl() must be called after one of
 * vorbis_encode_setup_managed() or vorbis_encode_setup_vbr().  When used
 * to modify settings, \ref vorbis_encode_ctl() must be called before \ref
 * vorbis_encode_setup_init().
 *
 * \param vi      Pointer to an initialized vorbis_info struct.
 *
 * \param number Specifies the desired action; See \ref encctlcodes "the list
 * of available requests".
 *
 * \param arg void * pointing to a data structure matching the request
 * argument.
 *
 * \retval 0          Success. Any further return information (such as the result of a
 * query) is placed into the storage pointed to by *arg.
 *
 * \retval OV_EINVAL  Invalid argument, or an attempt to modify a setting after
 * calling vorbis_encode_setup_init().
 *
 * \retval OV_EIMPL   Unimplemented or unknown request
 */
extern int vorbis_encode_ctl(vorbis_info *vi,int number,void *arg);

/**
 * \deprecated This is a deprecated interface. Please use vorbis_encode_ctl()
 * with the \ref ovectl_ratemanage2_arg struct and \ref
 * OV_ECTL_RATEMANAGE2_GET and \ref OV_ECTL_RATEMANAGE2_SET calls in new code.
 *
 * The \ref ovectl_ratemanage_arg structure is used with vorbis_encode_ctl()
 * and the \ref OV_ECTL_RATEMANAGE_GET, \ref OV_ECTL_RATEMANAGE_SET, \ref
 * OV_ECTL_RATEMANAGE_AVG, \ref OV_ECTL_RATEMANAGE_HARD calls in order to
 * query and modify specifics of the encoder's bitrate management
 * configuration.
*/
struct ovectl_ratemanage_arg {
  int    management_active; /**< nonzero if bitrate management is active*/
/** hard lower limit (in kilobits per second) below which the stream bitrate
    will never be allowed for any given bitrate_hard_window seconds of time.*/
  long   bitrate_hard_min;
/** hard upper limit (in kilobits per second) above which the stream bitrate
    will never be allowed for any given bitrate_hard_window seconds of time.*/
  long   bitrate_hard_max;
/** the window period (in seconds) used to regulate the hard bitrate minimum
    and maximum*/
  double bitrate_hard_window;
/** soft lower limit (in kilobits per second) below which the average bitrate
    tracker will start nudging the bitrate higher.*/
  long   bitrate_av_lo;
/** soft upper limit (in kilobits per second) above which the average bitrate
    tracker will start nudging the bitrate lower.*/
  long   bitrate_av_hi;
/** the window period (in seconds) used to regulate the average bitrate
    minimum and maximum.*/
  double bitrate_av_window;
/** Regulates the relative centering of the average and hard windows; in
    libvorbis 1.0 and 1.0.1, the hard window regulation overlapped but
    followed the average window regulation. In libvorbis 1.1 a bit-reservoir
    interface replaces the old windowing interface; the older windowing
    interface is simulated and this field has no effect.*/
  double bitrate_av_window_center;
};

/**
 * \name struct ovectl_ratemanage2_arg
 *
 * The ovectl_ratemanage2_arg structure is used with vorbis_encode_ctl() and
 * the OV_ECTL_RATEMANAGE2_GET and OV_ECTL_RATEMANAGE2_SET calls in order to
 * query and modify specifics of the encoder's bitrate management
 * configuration.
 *
*/
struct ovectl_ratemanage2_arg {
  int    management_active; /**< nonzero if bitrate management is active */
/** Lower allowed bitrate limit in kilobits per second */
  long   bitrate_limit_min_kbps;
/** Upper allowed bitrate limit in kilobits per second */
  long   bitrate_limit_max_kbps;
  long   bitrate_limit_reservoir_bits; /**<Size of the bitrate reservoir in bits */
/** Regulates the bitrate reservoir's preferred fill level in a range from 0.0
 * to 1.0; 0.0 tries to bank bits to buffer against future bitrate spikes, 1.0
 * buffers against future sudden drops in instantaneous bitrate. Default is
 * 0.1
 */
  double bitrate_limit_reservoir_bias;
/** Average bitrate setting in kilobits per second */
  long   bitrate_average_kbps;
/** Slew rate limit setting for average bitrate adjustment; sets the minimum
 *  time in seconds the bitrate tracker may swing from one extreme to the
 *  other when boosting or damping average bitrate.
 */
  double bitrate_average_damping;
};


/**
 * \name vorbis_encode_ctl() codes
 *
 * \anchor encctlcodes
 *
 * These values are passed as the \c number parameter of vorbis_encode_ctl().
 * The type of the referent of that function's \c arg pointer depends on these
 * codes.
 */
/*@{*/

/**
 * Query the current encoder bitrate management setting.
 *
 *Argument: <tt>struct ovectl_ratemanage2_arg *</tt>
 *
 * Used to query the current encoder bitrate management setting. Also used to
 * initialize fields of an ovectl_ratemanage2_arg structure for use with
 * \ref OV_ECTL_RATEMANAGE2_SET.
 */
#define OV_ECTL_RATEMANAGE2_GET      0x14

/**
 * Set the current encoder bitrate management settings.
 *
 * Argument: <tt>struct ovectl_ratemanage2_arg *</tt>
 *
 * Used to set the current encoder bitrate management settings to the values
 * listed in the ovectl_ratemanage2_arg. Passing a NULL pointer will disable
 * bitrate management.
*/
#define OV_ECTL_RATEMANAGE2_SET      0x15

/**
 * Returns the current encoder hard-lowpass setting (kHz) in the double
 * pointed to by arg.
 *
 * Argument: <tt>double *</tt>
*/
#define OV_ECTL_LOWPASS_GET          0x20

/**
 *  Sets the encoder hard-lowpass to the value (kHz) pointed to by arg. Valid
 *  lowpass settings range from 2 to 99.
 *
 * Argument: <tt>double *</tt>
*/
#define OV_ECTL_LOWPASS_SET          0x21

/**
 *  Returns the current encoder impulse block setting in the double pointed
 *  to by arg.
 *
 * Argument: <tt>double *</tt>
*/
#define OV_ECTL_IBLOCK_GET           0x30

/**
 *  Sets the impulse block bias to the the value pointed to by arg.
 *
 * Argument: <tt>double *</tt>
 *
 *  Valid range is -15.0 to 0.0 [default]. A negative impulse block bias will
 *  direct to encoder to use more bits when incoding short blocks that contain
 *  strong impulses, thus improving the accuracy of impulse encoding.
 */
#define OV_ECTL_IBLOCK_SET           0x31

/**
 *  Returns the current encoder coupling setting in the int pointed
 *  to by arg.
 *
 * Argument: <tt>int *</tt>
*/
#define OV_ECTL_COUPLING_GET         0x40

/**
 *  Enables/disables channel coupling in multichannel encoding according to arg.
 *
 * Argument: <tt>int *</tt>
 *
 *  Zero disables channel coupling for multichannel inputs, nonzer enables
 *  channel coupling.  Setting has no effect on monophonic encoding or
 *  multichannel counts that do not offer coupling.  At present, coupling is
 *  available for stereo and 5.1 encoding.
 */
#define OV_ECTL_COUPLING_SET         0x41

  /* deprecated rate management supported only for compatibility */

/**
 * Old interface to querying bitrate management settings.
 *
 * Deprecated after move to bit-reservoir style management in 1.1 rendered
 * this interface partially obsolete.

 * \deprecated Please use \ref OV_ECTL_RATEMANAGE2_GET instead.
 *
 * Argument: <tt>struct ovectl_ratemanage_arg *</tt>
 */
#define OV_ECTL_RATEMANAGE_GET       0x10
/**
 * Old interface to modifying bitrate management settings.
 *
 *  deprecated after move to bit-reservoir style management in 1.1 rendered
 *  this interface partially obsolete.
 *
 * \deprecated Please use \ref OV_ECTL_RATEMANAGE2_SET instead.
 *
 * Argument: <tt>struct ovectl_ratemanage_arg *</tt>
 */
#define OV_ECTL_RATEMANAGE_SET       0x11
/**
 * Old interface to setting average-bitrate encoding mode.
 *
 * Deprecated after move to bit-reservoir style management in 1.1 rendered
 * this interface partially obsolete.
 *
 *  \deprecated Please use \ref OV_ECTL_RATEMANAGE2_SET instead.
 *
 * Argument: <tt>struct ovectl_ratemanage_arg *</tt>
 */
#define OV_ECTL_RATEMANAGE_AVG       0x12
/**
 * Old interface to setting bounded-bitrate encoding modes.
 *
 * deprecated after move to bit-reservoir style management in 1.1 rendered
 * this interface partially obsolete.
 *
 *  \deprecated Please use \ref OV_ECTL_RATEMANAGE2_SET instead.
 *
 * Argument: <tt>struct ovectl_ratemanage_arg *</tt>
 */
#define OV_ECTL_RATEMANAGE_HARD      0x13

/*@}*/



#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

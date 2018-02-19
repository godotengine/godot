/*
 * libopenmpt.hpp
 * --------------
 * Purpose: libopenmpt public c++ interface
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#ifndef LIBOPENMPT_HPP
#define LIBOPENMPT_HPP

#include "libopenmpt_config.h"

#include <exception>
#include <iosfwd>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <cstdint>

/*!
 * \page libopenmpt_cpp_overview C++ API
 *
 * \section libopenmpt_cpp_error Error Handling
 *
 * libopenmpt C++ uses C++ exception handling for errror reporting.
 *
 * Unless otherwise noted, any libopenmpt function may throw exceptions and
 * all exceptions thrown by libopenmpt itself are derived from
 * openmpt::exception.
 * In addition, any libopenmpt function may also throw any exception specified
 * by the C++ language and C++ standard library. These are all derived from
 * std::exception.
 *
 * \section libopenmpt_cpp_strings Strings
 *
 * - All strings returned from libopenmpt are encoded in UTF-8.
 * - All strings passed to libopenmpt should also be encoded in UTF-8.
 * Behaviour in case of invalid UTF-8 is unspecified.
 * - libopenmpt does not enforce or expect any particular Unicode
 * normalization form.
 *
 * \section libopenmpt_cpp_fileio File I/O
 *
 * libopenmpt can use 3 different strategies for file I/O.
 *
 * - openmpt::module::module() with any kind of memory buffer as parameter will
 * load the module from the provided memory buffer, which will require loading
 * all data upfront by the library
 * caller.
 * - openmpt::module::module() with a seekable std::istream as parameter will
 * load the module via the stream interface. libopenmpt will not implement an
 * additional buffering layer in this case whih means the callbacks are assumed
 * to be performant even with small i/o sizes.
 * - openmpt::module::module() with an unseekable std::istream as parameter
 * will load the module via the stream interface. libopempt will make an
 * internal copy as it goes along, and sometimes have to pre-cache the whole
 * file in case it needs to know the complete file size. This strategy is
 * intended to be used if the file is located on a high latency network.
 *
 * | constructor       | speed  | memory consumption |
 * | ----------------: | :----: | :----------------: |
 * | memory buffer     | <p style="background-color:green" >fast  </p> | <p style="background-color:yellow">medium</p> | 
 * | seekable stream   | <p style="background-color:red"   >slow  </p> | <p style="background-color:green" >low   </p> |
 * | unseekable stream | <p style="background-color:yellow">medium</p> | <p style="background-color:red"   >high  </p> |
 *
 * In all cases, the data or stream passed to the constructor is no longer
 * needed after the openmpt::module has been constructed and can be destroyed
 * by the caller.
 *
 * \section libopenmpt_cpp_outputformat Output Format
 *
 * libopenmpt supports a wide range of PCM output formats:
 * [8000..192000]/[mono|stereo|quad]/[f32|i16].
 *
 * Unless you have some very specific requirements demanding a particular aspect
 * of the output format, you should always prefer 48000/stereo/f32 as the
 * libopenmpt PCM format.
 *
 * - Please prefer 48000Hz unless the user explicitly demands something else.
 * Practically all audio equipment and file formats use 48000Hz nowadays.
 * - Practically all module formats are made for stereo output. Mono will not
 * give you any measurable speed improvements and can trivially be obtained from
 * the stereo output anyway. Quad is not expected by almost all modules and even
 * if they do use surround effects, they expect the effects to be mixed to 
 * stereo. 
 * - Floating point output provides headroom instead of hard clipping if the
 * module is louder than 0dBFs, will give you a better signal-to-noise ratio
 * than int16 output, and avoid the need to apply an additional dithering to the
 * output by libopenmpt. Unless your platform has no floating point unit at all,
 * floating point will thus also be slightly faster.
 *
 * \section libopenmpt_cpp_threads libopenmpt in multi-threaded environments
 *
 * - libopenmpt is thread-aware.
 * - Individual libopenmpt objects are not thread-safe.
 * - libopenmpt itself does not spawn any user-visible threads but may spawn
 * threads for internal use.
 * - You must ensure to only ever access a particular libopenmpt object via
 * non-const member functions from a single thread at a time.
 * - You may access a particular libopenmpt objects concurrently from different
 * threads when using only const member functions from all threads.
 * - Consecutive accesses can happen from different threads.
 * - Different objects can be accessed concurrently from different threads.
 *
 * \section libopenmpt-cpp-windows Windows support
 *
 * Using the libopenmpt C++ API when libopenmpt is compiled as a DLL on Windows
 * requires `#define LIBOPENMPT_USE_DLL` (or some equivalent build system
 * configuration) before `#include <libopenmpt/libopenmpt.hpp>` in order to
 * correctly import the symbols from the DLL.
 *
 * \section libopenmpt-cpp-detailed Detailed documentation
 *
 * \ref libopenmpt_cpp
 *
 * \section libopenmpt_cpp_examples Example
 *
 * \include libopenmpt_example_cxx.cpp
 *
 */

/*! \defgroup libopenmpt_cpp libopenmpt C++ */

/*! \addtogroup libopenmpt_cpp
  @{
*/

namespace openmpt {

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4275)
#endif
//! libopenmpt exception base class
/*!
  Base class used for all exceptions that are thrown by libopenmpt itself. Libopenmpt may additionally throw any exception thrown by the standard library which are all derived from std::exception.
  \sa \ref libopenmpt_cpp_error
*/
class LIBOPENMPT_CXX_API exception : public std::exception {
private:
	char * text;
public:
	exception( const std::string & text ) noexcept;
	exception( const exception & other ) noexcept;
	exception( exception && other ) noexcept;
	exception & operator = ( const exception & other ) noexcept;
	exception & operator = ( exception && other ) noexcept;
	virtual ~exception() noexcept;
	virtual const char * what() const noexcept;
}; // class exception
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

//! Get the libopenmpt version number
/*!
  Returns the libopenmpt version number.
  \return The value represents (major << 24 + minor << 16 + patch << 0).
  \remarks libopenmpt < 0.3.0-pre used the following scheme: (major << 24 + minor << 16 + revision).
*/
LIBOPENMPT_CXX_API std::uint32_t get_library_version();

//! Get the core version number
/*!
  Return the OpenMPT core version number.
  \return The value represents (majormajor << 24 + major << 16 + minor << 8 + minorminor).
*/
LIBOPENMPT_CXX_API std::uint32_t get_core_version();

namespace string {

//! Return a verbose library version string from openmpt::string::get(). \deprecated Please use \code "library_version" \endcode directly.
LIBOPENMPT_DEPRECATED static const char library_version  LIBOPENMPT_ATTR_DEPRECATED [] = "library_version";
//! Return a verbose library features string from openmpt::string::get(). \deprecated Please use \code "library_features" \endcode directly.
LIBOPENMPT_DEPRECATED static const char library_features LIBOPENMPT_ATTR_DEPRECATED [] = "library_features";
//! Return a verbose OpenMPT core version string from openmpt::string::get(). \deprecated Please use \code "core_version" \endcode directly.
LIBOPENMPT_DEPRECATED static const char core_version     LIBOPENMPT_ATTR_DEPRECATED [] = "core_version";
//! Return information about the current build (e.g. the build date or compiler used) from openmpt::string::get(). \deprecated Please use \code "build" \endcode directly.
LIBOPENMPT_DEPRECATED static const char build            LIBOPENMPT_ATTR_DEPRECATED [] = "build";
//! Return all contributors from openmpt::string::get(). \deprecated Please use \code "credits" \endcode directly.
LIBOPENMPT_DEPRECATED static const char credits          LIBOPENMPT_ATTR_DEPRECATED [] = "credits";
//! Return contact information about libopenmpt from openmpt::string::get(). \deprecated Please use \code "contact" \endcode directly.
LIBOPENMPT_DEPRECATED static const char contact          LIBOPENMPT_ATTR_DEPRECATED [] = "contact";
//! Return the libopenmpt license from openmpt::string::get(). \deprecated Please use \code "license" \endcode directly.
LIBOPENMPT_DEPRECATED static const char license          LIBOPENMPT_ATTR_DEPRECATED [] = "license";

//! Get library related metadata.
/*!
  \param key Key to query.
         Possible keys are:
          -  "library_version": verbose library version string
          -  "library_version_major": libopenmpt major version number
          -  "library_version_minor": libopenmpt minor version number
          -  "library_version_patch": libopenmpt patch version number
          -  "library_version_prerel": libopenmpt pre-release version string
          -  "library_version_is_release": "1" if the version is an officially released version
          -  "library_features": verbose library features string
          -  "core_version": verbose OpenMPT core version string
          -  "source_url": original source code URL
          -  "source_date": original source code date
          -  "source_revision": original source code revision
          -  "source_is_modified": "1" if the original source has been modified
          -  "source_has_mixed_revisions": "1" if the original source has been compiled from different various revision
          -  "source_is_package": "1" if the original source has been obtained from a source pacakge instead of source code version control
          -  "build": information about the current build (e.g. the build date or compiler used)
          -  "build_compiler": information about the compiler used to build libopenmpt
          -  "credits": all contributors
          -  "contact": contact information about libopenmpt
          -  "license": the libopenmpt license
          -  "url": libopenmpt website URL
          -  "support_forum_url": libopenmpt support and discussions forum URL
          -  "bugtracker_url": libopenmpt bug and issue tracker URL

  \return A (possibly multi-line) string containing the queried information. If no information is available, the string is empty.
*/
LIBOPENMPT_CXX_API std::string get( const std::string & key );

} // namespace string

//! Get a list of supported file extensions
/*!
  \return The list of extensions supported by this libopenmpt build. The extensions are returned lower-case without a leading dot.
*/
LIBOPENMPT_CXX_API std::vector<std::string> get_supported_extensions();

//! Query whether a file extension is supported
/*!
  \param extension file extension to query without a leading dot. The case is ignored.
  \return true if the extension is supported by libopenmpt, false otherwise.
*/
LIBOPENMPT_CXX_API bool is_extension_supported( const std::string & extension );

//! Roughly scan the input stream to find out whether libopenmpt might be able to open it
/*!
  \param stream Input stream to scan.
  \param effort Effort to make when validating stream. Effort 0.0 does not even look at stream at all and effort 1.0 completely loads the file from stream. A lower effort requires less data to be loaded but only gives a rough estimate answer. Use an effort of 0.25 to only verify the header data of the module file.
  \param log Log where warning and errors are written.
  \return Probability between 0.0 and 1.0.
  \remarks openmpt::probe_file_header() provides a simpler and faster interface that fits almost all use cases better. It is recommended to use openmpt::probe_file_header() instead of openmpt::could_open_probability().
  \remarks openmpt::could_open_probability() can return any value between 0.0 and 1.0. Only 0.0 and 1.0 are definitive answers, all values in between are just estimates. In general, any return value >0.0 means that you should try loading the file, and any value below 1.0 means that loading may fail. If you want a threshold above which you can be reasonably sure that libopenmpt will be able to load the file, use >=0.5. If you see the need for a threshold below which you could reasonably outright reject a file, use <0.25 (Note: Such a threshold for rejecting on the lower end is not recommended, but may be required for better integration into some other framework's probe scoring.).
  \remarks openmpt::could_open_probability() expects the complete file data to be eventually available to it, even if it is asked to just parse the header. Verification will be unreliable (both false positives and false negatives), if you pretend that the file is just some few bytes of initial data threshold in size. In order to really just access the first bytes of a file, check in your std::istream implementation whether data or seeking is requested beyond your initial data threshold, and in that case, return an error. openmpt::could_open_probability() will treat this as any other I/O error and return 0.0. You must not expect the correct result in this case. You instead must remember that it asked for more data than you currently want to provide to it and treat this situation as if openmpt::could_open_probability() returned 0.5.
  \sa \ref libopenmpt_c_fileio
  \sa openmpt::probe_file_header()
  \since 0.3.0
*/
LIBOPENMPT_CXX_API double could_open_probability( std::istream & stream, double effort = 1.0, std::ostream & log = std::clog );

//! Roughly scan the input stream to find out whether libopenmpt might be able to open it
/*!
  \deprecated Please use openmpt::module::could_open_probability().
*/
LIBOPENMPT_ATTR_DEPRECATED LIBOPENMPT_CXX_API LIBOPENMPT_DEPRECATED double could_open_propability( std::istream & stream, double effort = 1.0, std::ostream & log = std::clog );

//! Get recommended header size for successfull format probing
/*!
  \sa openmpt::probe_file_header()
  \since 0.3.0
*/
LIBOPENMPT_CXX_API std::size_t probe_file_header_get_recommended_size();

//! Probe for module formats in openmpt::probe_file_header(). \since 0.3.0
static const std::uint64_t probe_file_header_flags_modules    = 0x1ul;

//! Probe for module-specific container formats in openmpt::probe_file_header(). \since 0.3.0
static const std::uint64_t probe_file_header_flags_containers = 0x2ul;

//! Probe for the default set of formats in openmpt::probe_file_header(). \since 0.3.0
static const std::uint64_t probe_file_header_flags_default    = probe_file_header_flags_modules | probe_file_header_flags_containers;

//! Probe for no formats in openmpt::probe_file_header(). \since 0.3.0
static const std::uint64_t probe_file_header_flags_none       = 0x0ul;

//! Possible return values for openmpt::probe_file_header(). \since 0.3.0
enum probe_file_header_result {
	probe_file_header_result_success      =  1,
	probe_file_header_result_failure      =  0,
	probe_file_header_result_wantmoredata = -1
};

//! Probe the provided bytes from the beginning of a file for supported file format headers to find out whether libopenmpt might be able to open it
/*!
  \param flags Ored mask of openmpt::probe_file_header_flags_modules and openmpt::probe_file_header_flags_containers, or openmpt::probe_file_header_flags_default.
  \param data Beginning of the file data.
  \param size Size of the beginning of the file data.
  \param filesize Full size of the file data on disk.
  \remarks It is recommended to provide openmpt::probe_file_header_get_recommended_size() bytes of data for data and size. If the file is smaller, only provide the filesize amount and set size and filesize to the file's size. 
  \remarks openmpt::could_open_probability() provides a more elaborate interface that might be required for special use cases. It is recommended to use openmpt::probe_file_header() though, if possible.
  \retval probe_file_header_result_success The file will most likely be supported by libopenmpt.
  \retval probe_file_header_result_failure The file is not supported by libopenmpt.
  \retval probe_file_header_result_wantmoredata An answer could not be determined with the amount of data provided.
  \sa openmpt::probe_file_header_get_recommended_size()
  \sa openmpt::could_open_probability()
  \since 0.3.0
*/
LIBOPENMPT_CXX_API int probe_file_header( std::uint64_t flags, const std::uint8_t * data, std::size_t size, std::uint64_t filesize );

//! Probe the provided bytes from the beginning of a file for supported file format headers to find out whether libopenmpt might be able to open it
/*!
  \param flags Ored mask of openmpt::probe_file_header_flags_modules and openmpt::probe_file_header_flags_containers, or openmpt::probe_file_header_flags_default.
  \param data Beginning of the file data.
  \param size Size of the beginning of the file data.
  \remarks It is recommended to use the overload of this function that also takes the filesize as parameter if at all possile. libopenmpt can provide more accurate answers if the filesize is known.
  \remarks It is recommended to provide openmpt::probe_file_header_get_recommended_size() bytes of data for data and size. If the file is smaller, only provide the filesize amount and set size to the file's size. 
  \remarks openmpt::could_open_probability() provides a more elaborate interface that might be required for special use cases. It is recommended to use openmpt::probe_file_header() though, if possible.
  \retval probe_file_header_result_success The file will most likely be supported by libopenmpt.
  \retval probe_file_header_result_failure The file is not supported by libopenmpt.
  \retval probe_file_header_result_wantmoredata An answer could not be determined with the amount of data provided.
  \sa openmpt::probe_file_header_get_recommended_size()
  \sa openmpt::could_open_probability()
  \since 0.3.0
*/
LIBOPENMPT_CXX_API int probe_file_header( std::uint64_t flags, const std::uint8_t * data, std::size_t size );

//! Probe the provided bytes from the beginning of a file for supported file format headers to find out whether libopenmpt might be able to open it
/*!
  \param flags Ored mask of openmpt::probe_file_header_flags_modules and openmpt::probe_file_header_flags_containers, or openmpt::probe_file_header_flags_default.
  \param stream Input stream to scan.
  \remarks stream is left in an unspecified state when this function returns.
  \remarks openmpt::could_open_probability() provides a more elaborate interface that might be required for special use cases. It is recommended to use openmpt::probe_file_header() though, if possible.
  \retval probe_file_header_result_success The file will most likely be supported by libopenmpt.
  \retval probe_file_header_result_failure The file is not supported by libopenmpt.
  \retval probe_file_header_result_wantmoredata An answer could not be determined with the amount of data provided.
  \sa openmpt::probe_file_header_get_recommended_size()
  \sa openmpt::could_open_probability()
  \since 0.3.0
*/
LIBOPENMPT_CXX_API int probe_file_header( std::uint64_t flags, std::istream & stream );

class module_impl;

class module_ext;

namespace detail {

typedef std::map< std::string, std::string > initial_ctls_map;

} // namespace detail

class LIBOPENMPT_CXX_API module {

	friend class module_ext;

public:

	//! Parameter index to use with openmpt::module::get_render_param and openmpt::module::set_render_param
	enum render_param {
		//! Master Gain
		/*!
		  The related value represents a relative gain in milliBel.\n
		  The default value is 0.\n
		  The supported value range is unlimited.\n
		*/
		RENDER_MASTERGAIN_MILLIBEL        = 1,
		//! Stereo Separation
		/*!
		  The related value represents the stereo separation generated by the libopenmpt mixer in percent.\n
		  The default value is 100.\n
		  The supported value range is [0,200].\n
		*/
		RENDER_STEREOSEPARATION_PERCENT   = 2,
		//! Interpolation Filter
		/*!
		  The related value represents the interpolation filter length used by the libopenmpt mixer.\n
		  The default value is 0, which indicates a recommended default value.\n
		  The supported value range is [0,inf). Values greater than the implementation limit are clamped to the maximum supported value.\n
		  Currently supported values:
		   - 0: internal default
		   - 1: no interpolation (zero order hold)
		   - 2: linear interpolation
		   - 4: cubic interpolation
		   - 8: windowed sinc with 8 taps
		*/
		RENDER_INTERPOLATIONFILTER_LENGTH = 3,
		//! Volume Ramping Strength
		/*!
		  The related value represents the amount of volume ramping done by the libopenmpt mixer.\n
		  The default value is -1, which indicates a recommended default value.\n
		  The meaningful value range is [-1..10].\n
		  A value of 0 completely disables volume ramping. This might cause clicks in sound output.\n
		  Higher values imply slower/softer volume ramps.
		*/
		RENDER_VOLUMERAMPING_STRENGTH     = 4
	};

	//! Parameter index to use with openmpt::module::get_pattern_row_channel_command, openmpt::module::format_pattern_row_channel_command and openmpt::module::highlight_pattern_row_channel_command
	enum command_index {
		command_note        = 0,
		command_instrument  = 1,
		command_volumeffect = 2,
		command_effect      = 3,
		command_volume      = 4,
		command_parameter   = 5
	};

private:
	module_impl * impl;
private:
	// non-copyable
	module( const module & );
	void operator = ( const module & );
private:
	// for module_ext
	module();
	void set_impl( module_impl * i );
public:
	//! Construct an openmpt::module
	/*!
	  \param stream Input stream from which the module is loaded. After the constructor has finished successfully, the input position of stream is set to the byte after the last byte that has been read. If the constructor fails, the state of the input position of stream is undefined.
	  \param log Log where any warnings or errors are printed to. The lifetime of the reference has to be as long as the lifetime of the module instance.
	  \param ctls A map of initial ctl values, see openmpt::module::get_ctls.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception in case the provided file cannot be opened.
	  \remarks The input data can be discarded after an openmpt::module has been constructed successfully.
	  \sa \ref libopenmpt_cpp_fileio
	*/
	module( std::istream & stream, std::ostream & log = std::clog, const std::map< std::string, std::string > & ctls = detail::initial_ctls_map() );
	/*!
	  \param data Data to load the module from.
	  \param log Log where any warnings or errors are printed to. The lifetime of the reference has to be as long as the lifetime of the module instance.
	  \param ctls A map of initial ctl values, see openmpt::module::get_ctls.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception in case the provided file cannot be opened.
	  \remarks The input data can be discarded after an openmpt::module has been constructed successfully.
	  \sa \ref libopenmpt_cpp_fileio
	*/
	module( const std::vector<std::uint8_t> & data, std::ostream & log = std::clog, const std::map< std::string, std::string > & ctls = detail::initial_ctls_map() );
	/*!
	  \param beg Begin of data to load the module from.
	  \param end End of data to load the module from.
	  \param log Log where any warnings or errors are printed to. The lifetime of the reference has to be as long as the lifetime of the module instance.
	  \param ctls A map of initial ctl values, see openmpt::module::get_ctls.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception in case the provided file cannot be opened.
	  \remarks The input data can be discarded after an openmpt::module has been constructed successfully.
	  \sa \ref libopenmpt_cpp_fileio
	*/
	module( const std::uint8_t * beg, const std::uint8_t * end, std::ostream & log = std::clog, const std::map< std::string, std::string > & ctls = detail::initial_ctls_map() );
	/*!
	  \param data Data to load the module from.
	  \param size Amount of data available.
	  \param log Log where any warnings or errors are printed to. The lifetime of the reference has to be as long as the lifetime of the module instance.
	  \param ctls A map of initial ctl values, see openmpt::module::get_ctls.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception in case the provided file cannot be opened.
	  \remarks The input data can be discarded after an openmpt::module has been constructed successfully.
	  \sa \ref libopenmpt_cpp_fileio
	*/
	module( const std::uint8_t * data, std::size_t size, std::ostream & log = std::clog, const std::map< std::string, std::string > & ctls = detail::initial_ctls_map() );
	/*!
	  \param data Data to load the module from.
	  \param log Log where any warnings or errors are printed to. The lifetime of the reference has to be as long as the lifetime of the module instance.
	  \param ctls A map of initial ctl values, see openmpt::module::get_ctls.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception in case the provided file cannot be opened.
	  \remarks The input data can be discarded after an openmpt::module has been constructed successfully.
	  \sa \ref libopenmpt_cpp_fileio
	*/
	module( const std::vector<char> & data, std::ostream & log = std::clog, const std::map< std::string, std::string > & ctls = detail::initial_ctls_map() );
	/*!
	  \param beg Begin of data to load the module from.
	  \param end End of data to load the module from.
	  \param log Log where any warnings or errors are printed to. The lifetime of the reference has to be as long as the lifetime of the module instance.
	  \param ctls A map of initial ctl values, see openmpt::module::get_ctls.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception in case the provided file cannot be opened.
	  \remarks The input data can be discarded after an openmpt::module has been constructed successfully.
	  \sa \ref libopenmpt_cpp_fileio
	*/
	module( const char * beg, const char * end, std::ostream & log = std::clog, const std::map< std::string, std::string > & ctls = detail::initial_ctls_map() );
	/*!
	  \param data Data to load the module from.
	  \param size Amount of data available.
	  \param log Log where any warnings or errors are printed to. The lifetime of the reference has to be as long as the lifetime of the module instance.
	  \param ctls A map of initial ctl values, see openmpt::module::get_ctls.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception in case the provided file cannot be opened.
	  \remarks The input data can be discarded after an openmpt::module has been constructed successfully.
	  \sa \ref libopenmpt_cpp_fileio
	*/
	module( const char * data, std::size_t size, std::ostream & log = std::clog, const std::map< std::string, std::string > & ctls = detail::initial_ctls_map() );
	/*!
	  \param data Data to load the module from.
	  \param size Amount of data available.
	  \param log Log where any warnings or errors are printed to. The lifetime of the reference has to be as long as the lifetime of the module instance.
	  \param ctls A map of initial ctl values, see openmpt::module::get_ctls.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception in case the provided file cannot be opened.
	  \remarks The input data can be discarded after an openmpt::module has been constructed successfully.
	  \sa \ref libopenmpt_cpp_fileio
	*/
	module( const void * data, std::size_t size, std::ostream & log = std::clog, const std::map< std::string, std::string > & ctls = detail::initial_ctls_map() );
	virtual ~module();
public:

	//! Select a sub-song from a multi-song module
	/*!
	  \param subsong Index of the sub-song. -1 plays all sub-songs consecutively.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception if sub-song is not in range [-1,openmpt::module::get_num_subsongs()[
	  \sa openmpt::module::get_num_subsongs, openmpt::module::get_selected_subsong, openmpt::module::get_subsong_names
	  \remarks Whether subsong -1 (all subsongs consecutively), subsong 0 or some other subsong is selected by default, is an implementation detail and subject to change. If you do not want to care about subsongs, it is recommended to just not call openmpt::module::select_subsong() at all.
	*/
	void select_subsong( std::int32_t subsong );
	//! Get currently selected sub-song from a multi-song module
	/*!
	  \return Currently selected sub-song. -1 for all subsongs consecutively, 0 or greater for the current sub-song index.
	  \sa openmpt::module::get_num_subsongs, openmpt::module::select_subsong, openmpt::module::get_subsong_names
	  \since 0.3.0
	*/
	std::int32_t get_selected_subsong() const;
	//! Set Repeat Count
	/*!
	  \param repeat_count Repeat Count
	    - -1: repeat forever
	    - 0: play once, repeat zero times (the default)
	    - n>0: play once and repeat n times after that
	  \sa openmpt::module::get_repeat_count
	*/
	void set_repeat_count( std::int32_t repeat_count );
	//! Get Repeat Count
	/*!
	  \return Repeat Count
	    - -1: repeat forever
	    - 0: play once, repeat zero times (the default)
	    - n>0: play once and repeat n times after that
	  \sa openmpt::module::set_repeat_count
	*/
	std::int32_t get_repeat_count() const;

	//! Get approximate song duration
	/*!
	  \return Approximate duration of current sub-song in seconds.
	*/
	double get_duration_seconds() const;

	//! Set approximate current song position
	/*!
	  \param seconds Seconds to seek to. If seconds is out of range, the position gets set to song start or end respectively.
	  \return Approximate new song position in seconds.
	  \sa openmpt::module::get_position_seconds
	*/
	double set_position_seconds( double seconds );
	//! Get current song position
	/*!
	  \return Current song position in seconds.
	  \sa openmpt::module::set_position_seconds
	*/
	double get_position_seconds() const;

	//! Set approximate current song position
	/*!
	  If order or row are out of range, to position is not modified and the current position is returned.
	  \param order Pattern order number to seek to.
	  \param row Pattern row number to seek to.
	  \return Approximate new song position in seconds.
	  \sa openmpt::module::set_position_seconds
	  \sa openmpt::module::get_position_seconds
	*/
	double set_position_order_row( std::int32_t order, std::int32_t row );

	//! Get render parameter
	/*!
	  \param param Parameter to query. See openmpt::module::render_param.
	  \return The current value of the parameter.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception if param is invalid.
	  \sa openmpt::module::render_param
	  \sa openmpt::module::set_render_param
	*/
	std::int32_t get_render_param( int param ) const;
	//! Set render parameter
	/*!
	  \param param Parameter to set. See openmpt::module::render_param.
	  \param value The value to set param to.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception if param is invalid or value is out of range.
	  \sa openmpt::module::render_param
	  \sa openmpt::module::get_render_param
	*/
	void set_render_param( int param, std::int32_t value );

	/*@{*/
	//! Render audio data
	/*!
	  \param samplerate Sample rate to render output. Should be in [8000,192000], but this is not enforced.
	  \param count Number of audio frames to render per channel.
	  \param mono Pointer to a buffer of at least count elements that receives the mono/center output.
	  \return The number of frames actually rendered.
	  \retval 0 The end of song has been reached.
	  \remarks The output buffers are only written to up to the returned number of elements.
	  \remarks You can freely switch between any of the "read*" variants if you see a need to do so. libopenmpt tries to introduce as little switching annoyances as possible. Normally, you would only use a single one of these functions for rendering a particular module.
	  \remarks It is recommended to use the floating point API because of the greater dynamic range and no implied clipping.
	  \sa \ref libopenmpt_cpp_outputformat
	*/
	std::size_t read( std::int32_t samplerate, std::size_t count, std::int16_t * mono );
	//! Render audio data
	/*!
	  \param samplerate Sample rate to render output. Should be in [8000,192000], but this is not enforced.
	  \param count Number of audio frames to render per channel.
	  \param left Pointer to a buffer of at least count elements that receives the left output.
	  \param right Pointer to a buffer of at least count elements that receives the right output.
	  \return The number of frames actually rendered.
	  \retval 0 The end of song has been reached.
	  \remarks The output buffers are only written to up to the returned number of elements.
	  \remarks You can freely switch between any of the "read*" variants if you see a need to do so. libopenmpt tries to introduce as little switching annoyances as possible. Normally, you would only use a single one of these functions for rendering a particular module.
	  \remarks It is recommended to use the floating point API because of the greater dynamic range and no implied clipping.
	  \sa \ref libopenmpt_cpp_outputformat
	*/
	std::size_t read( std::int32_t samplerate, std::size_t count, std::int16_t * left, std::int16_t * right );
	//! Render audio data
	/*!
	  \param samplerate Sample rate to render output. Should be in [8000,192000], but this is not enforced.
	  \param count Number of audio frames to render per channel.
	  \param left Pointer to a buffer of at least count elements that receives the left output.
	  \param right Pointer to a buffer of at least count elements that receives the right output.
	  \param rear_left Pointer to a buffer of at least count elements that receives the rear left output.
	  \param rear_right Pointer to a buffer of at least count elements that receives the rear right output.
	  \return The number of frames actually rendered.
	  \retval 0 The end of song has been reached.
	  \remarks The output buffers are only written to up to the returned number of elements.
	  \remarks You can freely switch between any of the "read*" variants if you see a need to do so. libopenmpt tries to introduce as little switching annoyances as possible. Normally, you would only use a single one of these functions for rendering a particular module.
	  \remarks It is recommended to use the floating point API because of the greater dynamic range and no implied clipping.
	  \sa \ref libopenmpt_cpp_outputformat
	*/
	std::size_t read( std::int32_t samplerate, std::size_t count, std::int16_t * left, std::int16_t * right, std::int16_t * rear_left, std::int16_t * rear_right );
	//! Render audio data
	/*!
	  \param samplerate Sample rate to render output. Should be in [8000,192000], but this is not enforced.
	  \param count Number of audio frames to render per channel.
	  \param mono Pointer to a buffer of at least count elements that receives the mono/center output.
	  \return The number of frames actually rendered.
	  \retval 0 The end of song has been reached.
	  \remarks The output buffers are only written to up to the returned number of elements.
	  \remarks You can freely switch between any of the "read*" variants if you see a need to do so. libopenmpt tries to introduce as little switching annoyances as possible. Normally, you would only use a single one of these functions for rendering a particular module.
	  \remarks Floating point samples are in the [-1.0..1.0] nominal range. They are not clipped to that range though and thus might overshoot.
	  \sa \ref libopenmpt_cpp_outputformat
	*/
	std::size_t read( std::int32_t samplerate, std::size_t count, float * mono );
	//! Render audio data
	/*!
	  \param samplerate Sample rate to render output. Should be in [8000,192000], but this is not enforced.
	  \param count Number of audio frames to render per channel.
	  \param left Pointer to a buffer of at least count elements that receives the left output.
	  \param right Pointer to a buffer of at least count elements that receives the right output.
	  \return The number of frames actually rendered.
	  \retval 0 The end of song has been reached.
	  \remarks The output buffers are only written to up to the returned number of elements.
	  \remarks You can freely switch between any of the "read*" variants if you see a need to do so. libopenmpt tries to introduce as little switching annoyances as possible. Normally, you would only use a single one of these functions for rendering a particular module.
	  \remarks Floating point samples are in the [-1.0..1.0] nominal range. They are not clipped to that range though and thus might overshoot.
	  \sa \ref libopenmpt_cpp_outputformat
	*/
	std::size_t read( std::int32_t samplerate, std::size_t count, float * left, float * right );
	//! Render audio data
	/*!
	  \param samplerate Sample rate to render output. Should be in [8000,192000], but this is not enforced.
	  \param count Number of audio frames to render per channel.
	  \param left Pointer to a buffer of at least count elements that receives the left output.
	  \param right Pointer to a buffer of at least count elements that receives the right output.
	  \param rear_left Pointer to a buffer of at least count elements that receives the rear left output.
	  \param rear_right Pointer to a buffer of at least count elements that receives the rear right output.
	  \return The number of frames actually rendered.
	  \retval 0 The end of song has been reached.
	  \remarks The output buffers are only written to up to the returned number of elements.
	  \remarks You can freely switch between any of the "read*" variants if you see a need to do so. libopenmpt tries to introduce as little switching annoyances as possible. Normally, you would only use a single one of these functions for rendering a particular module.
	  \remarks Floating point samples are in the [-1.0..1.0] nominal range. They are not clipped to that range though and thus might overshoot.
	  \sa \ref libopenmpt_cpp_outputformat
	*/
	std::size_t read( std::int32_t samplerate, std::size_t count, float * left, float * right, float * rear_left, float * rear_right );
	//! Render audio data
	/*!
	  \param samplerate Sample rate to render output. Should be in [8000,192000], but this is not enforced.
	  \param count Number of audio frames to render per channel.
	  \param interleaved_stereo Pointer to a buffer of at least count*2 elements that receives the interleaved stereo output in the order (L,R).
	  \return The number of frames actually rendered.
	  \retval 0 The end of song has been reached.
	  \remarks The output buffers are only written to up to the returned number of elements.
	  \remarks You can freely switch between any of the "read*" variants if you see a need to do so. libopenmpt tries to introduce as little switching annoyances as possible. Normally, you would only use a single one of these functions for rendering a particular module.
	  \remarks It is recommended to use the floating point API because of the greater dynamic range and no implied clipping.
	  \sa \ref libopenmpt_cpp_outputformat
	*/
	std::size_t read_interleaved_stereo( std::int32_t samplerate, std::size_t count, std::int16_t * interleaved_stereo );
	//! Render audio data
	/*!
	  \param samplerate Sample rate to render output. Should be in [8000,192000], but this is not enforced.
	  \param count Number of audio frames to render per channel.
	  \param interleaved_quad Pointer to a buffer of at least count*4 elements that receives the interleaved suad surround output in the order (L,R,RL,RR).
	  \return The number of frames actually rendered.
	  \retval 0 The end of song has been reached.
	  \remarks The output buffers are only written to up to the returned number of elements.
	  \remarks You can freely switch between any of the "read*" variants if you see a need to do so. libopenmpt tries to introduce as little switching annoyances as possible. Normally, you would only use a single one of these functions for rendering a particular module.
	  \remarks It is recommended to use the floating point API because of the greater dynamic range and no implied clipping.
	  \sa \ref libopenmpt_cpp_outputformat
	*/
	std::size_t read_interleaved_quad( std::int32_t samplerate, std::size_t count, std::int16_t * interleaved_quad );
	//! Render audio data
	/*!
	  \param samplerate Sample rate to render output. Should be in [8000,192000], but this is not enforced.
	  \param count Number of audio frames to render per channel.
	  \param interleaved_stereo Pointer to a buffer of at least count*2 elements that receives the interleaved stereo output in the order (L,R).
	  \return The number of frames actually rendered.
	  \retval 0 The end of song has been reached.
	  \remarks The output buffers are only written to up to the returned number of elements.
	  \remarks You can freely switch between any of the "read*" variants if you see a need to do so. libopenmpt tries to introduce as little switching annoyances as possible. Normally, you would only use a single one of these functions for rendering a particular module.
	  \remarks Floating point samples are in the [-1.0..1.0] nominal range. They are not clipped to that range though and thus might overshoot.
	  \sa \ref libopenmpt_cpp_outputformat
	*/
	std::size_t read_interleaved_stereo( std::int32_t samplerate, std::size_t count, float * interleaved_stereo );
	//! Render audio data
	/*!
	  \param samplerate Sample rate to render output. Should be in [8000,192000], but this is not enforced.
	  \param count Number of audio frames to render per channel.
	  \param interleaved_quad Pointer to a buffer of at least count*4 elements that receives the interleaved suad surround output in the order (L,R,RL,RR).
	  \return The number of frames actually rendered.
	  \retval 0 The end of song has been reached.
	  \remarks The output buffers are only written to up to the returned number of elements.
	  \remarks You can freely switch between any of the "read*" variants if you see a need to do so. libopenmpt tries to introduce as little switching annoyances as possible. Normally, you would only use a single one of these functions for rendering a particular module.
	  \remarks Floating point samples are in the [-1.0..1.0] nominal range. They are not clipped to that range though and thus might overshoot.
	  \sa \ref libopenmpt_cpp_outputformat
	*/
	std::size_t read_interleaved_quad( std::int32_t samplerate, std::size_t count, float * interleaved_quad );
	/*@}*/

	//! Get the list of supported metadata item keys
	/*!
	  \return Metadata item keys supported by openmpt::module::get_metadata
	  \sa openmpt::module::get_metadata
	*/
	std::vector<std::string> get_metadata_keys() const;
	//! Get a metadata item value
	/*!
	  \param key Metadata item key to query. Use openmpt::module::get_metadata_keys to check for available keys.
	           Possible keys are:
	           - type: Module format extension (e.g. it)
	           - type_long: Tracker name associated with the module format (e.g. Impulse Tracker)
	           - container: Container format the module file is embedded in, if any (e.g. umx)
	           - container_long: Full container name if the module is embedded in a container (e.g. Unreal Music)
	           - tracker: Tracker that was (most likely) used to save the module file, if known
	           - artist: Author of the module
	           - title: Module title
	           - date: Date the module was last saved, in ISO-8601 format.
	           - message: Song message. If the song message is empty or the module format does not support song messages, a list of instrument and sample names is returned instead.
	           - message_raw: Song message. If the song message is empty or the module format does not support song messages, an empty string is returned.
	           - warnings: A list of warnings that were generated while loading the module.
	  \return The associated value for key.
	  \sa openmpt::module::get_metadata_keys
	*/
	std::string get_metadata( const std::string & key ) const;

	//! Get the current speed
	/*!
	  \return The current speed in ticks per row.
	*/
	std::int32_t get_current_speed() const;
	//! Get the current tempo
	/*!
	  \return The current tempo in tracker units. The exact meaning of this value depends on the tempo mode being used.
	*/
	std::int32_t get_current_tempo() const;
	//! Get the current order
	/*!
	  \return The current order at which the module is being played back.
	*/
	std::int32_t get_current_order() const;
	//! Get the current pattern
	/*!
	  \return The current pattern that is being played.
	*/
	std::int32_t get_current_pattern() const;
	//! Get the current row
	/*!
	  \return The current row at which the current pattern is being played.
	*/
	std::int32_t get_current_row() const;
	//! Get the current amount of playing channels.
	/*!
	  \return The amount of sample channels that are currently being rendered.
	*/
	std::int32_t get_current_playing_channels() const;

	//! Get an approximate indication of the channel volume.
	/*!
	  \param channel The channel whose volume should be retrieved.
	  \return The approximate channel volume.
	  \remarks The returned value is solely based on the note velocity and does not take the actual waveform of the playing sample into account.
	*/
	float get_current_channel_vu_mono( std::int32_t channel ) const;
	//! Get an approximate indication of the channel volume on the front-left speaker.
	/*!
	  \param channel The channel whose volume should be retrieved.
	  \return The approximate channel volume.
	  \remarks The returned value is solely based on the note velocity and does not take the actual waveform of the playing sample into account.
	*/
	float get_current_channel_vu_left( std::int32_t channel ) const;
	//! Get an approximate indication of the channel volume on the front-right speaker.
	/*!
	  \param channel The channel whose volume should be retrieved.
	  \return The approximate channel volume.
	  \remarks The returned value is solely based on the note velocity and does not take the actual waveform of the playing sample into account.
	*/
	float get_current_channel_vu_right( std::int32_t channel ) const;
	//! Get an approximate indication of the channel volume on the rear-left speaker.
	/*!
	  \param channel The channel whose volume should be retrieved.
	  \return The approximate channel volume.
	  \remarks The returned value is solely based on the note velocity and does not take the actual waveform of the playing sample into account.
	*/
	float get_current_channel_vu_rear_left( std::int32_t channel ) const;
	//! Get an approximate indication of the channel volume on the rear-right speaker.
	/*!
	  \param channel The channel whose volume should be retrieved.
	  \return The approximate channel volume.
	  \remarks The returned value is solely based on the note velocity and does not take the actual waveform of the playing sample into account.
	*/
	float get_current_channel_vu_rear_right( std::int32_t channel ) const;

	//! Get the number of sub-songs
	/*!
	  \return The number of sub-songs in the module. This includes any "hidden" songs (songs that share the same sequence, but start at different order indices) and "normal" sub-songs or "sequences" (if the format supports them).
	  \sa openmpt::module::get_subsong_names, openmpt::module::select_subsong, openmpt::module::get_selected_subsong
	*/
	std::int32_t get_num_subsongs() const;
	//! Get the number of pattern channels
	/*!
	  \return The number of pattern channels in the module. Not all channels do necessarily contain data.
	  \remarks The number of pattern channels is completely independent of the number of output channels. libopenmpt can render modules in mono, stereo or quad surround, but the choice of which of the three modes to use must not be made based on the return value of this function, which may be any positive integer amount. Only use this function for informational purposes.
	*/
	std::int32_t get_num_channels() const;
	//! Get the number of orders
	/*!
	  \return The number of orders in the current sequence of the module.
	*/
	std::int32_t get_num_orders() const;
	//! Get the number of patterns
	/*!
	  \return The number of distinct patterns in the module.
	*/
	std::int32_t get_num_patterns() const;
	//! Get the number of instruments
	/*!
	  \return The number of instrument slots in the module. Instruments are a layer on top of samples, and are not supported by all module formats.
	*/
	std::int32_t get_num_instruments() const;
	//! Get the number of samples
	/*!
	  \return The number of sample slots in the module.
	*/
	std::int32_t get_num_samples() const;

	//! Get a list of sub-song names
	/*!
	  \return All sub-song names.
	  \sa openmpt::module::get_num_subsongs, openmpt::module::select_subsong, openmpt::module::get_selected_subsong
	*/
	std::vector<std::string> get_subsong_names() const;
	//! Get a list of channel names
	/*!
	  \return All channel names.
	  \sa openmpt::module::get_num_channels
	*/
	std::vector<std::string> get_channel_names() const;
	//! Get a list of order names
	/*!
	  \return All order names.
	  \sa openmpt::module::get_num_orders
	*/
	std::vector<std::string> get_order_names() const;
	//! Get a list of pattern names
	/*!
	  \return All pattern names.
	  \sa openmpt::module::get_num_patterns
	*/
	std::vector<std::string> get_pattern_names() const;
	//! Get a list of instrument names
	/*!
	  \return All instrument names.
	  \sa openmpt::module::get_num_instruments
	*/
	std::vector<std::string> get_instrument_names() const;
	//! Get a list of sample names
	/*!
	  \return All sample names.
	  \sa openmpt::module::get_num_samples
	*/
	std::vector<std::string> get_sample_names() const;

	//! Get pattern at order position
	/*!
	  \param order The order item whose pattern index should be retrieved.
	  \return The pattern index found at the given order position of the current sequence.
	*/
	std::int32_t get_order_pattern( std::int32_t order ) const;

	//! Get the number of rows in a pattern
	/*!
	  \param pattern The pattern whose row count should be retrieved.
	  \return The number of rows in the given pattern. If the pattern does not exist, 0 is returned.
	*/
	std::int32_t get_pattern_num_rows( std::int32_t pattern ) const;

	//! Get raw pattern content
	/*!
	  \param pattern The pattern whose data should be retrieved.
	  \param row The row from which the data should be retrieved.
	  \param channel The channel from which the data should be retrieved.
	  \param command The cell index at which the data should be retrieved. See openmpt::module::command_index
	  \return The internal, raw pattern data at the given pattern position.
	*/
	std::uint8_t get_pattern_row_channel_command( std::int32_t pattern, std::int32_t row, std::int32_t channel, int command ) const;

	//! Get formatted (human-readable) pattern content
	/*!
	  \param pattern The pattern whose data should be retrieved.
	  \param row The row from which the data should be retrieved.
	  \param channel The channel from which the data should be retrieved.
	  \param command The cell index at which the data should be retrieved.
	  \return The formatted pattern data at the given pattern position. See openmpt::module::command_index
	  \sa openmpt::module::highlight_pattern_row_channel_command
	*/
	std::string format_pattern_row_channel_command( std::int32_t pattern, std::int32_t row, std::int32_t channel, int command ) const;

	//! Get highlighting information for formatted pattern content
	/*!
	  \param pattern The pattern whose data should be retrieved.
	  \param row The row from which the data should be retrieved.
	  \param channel The channel from which the data should be retrieved.
	  \param command The cell index at which the data should be retrieved. See openmpt::module::command_index
	  \return The highlighting string for the formatted pattern data as retrieved by openmpt::module::get_pattern_row_channel_command at the given pattern position.
	  \remarks The returned string will map each character position of the string returned by openmpt::module::get_pattern_row_channel_command to a highlighting instruction.
	           Possible highlighting characters are:
	           - " " : empty/space
	           - "." : empty/dot
	           - "n" : generic note
	           - "m" : special note
	           - "i" : generic instrument
	           - "u" : generic volume column effect
	           - "v" : generic volume column parameter
	           - "e" : generic effect column effect
	           - "f" : generic effect column parameter
	  \sa openmpt::module::get_pattern_row_channel_command
	*/
	std::string highlight_pattern_row_channel_command( std::int32_t pattern, std::int32_t row, std::int32_t channel, int command ) const;

	//! Get formatted (human-readable) pattern content
	/*!
	  \param pattern The pattern whose data should be retrieved.
	  \param row The row from which the data should be retrieved.
	  \param channel The channel from which the data should be retrieved.
	  \param width The maximum number of characters the string should contain. 0 means no limit.
	  \param pad If true, the string will be resized to the exact length provided in the width parameter.
	  \return The formatted pattern data at the given pattern position.
	  \sa openmpt::module::highlight_pattern_row_channel
	*/
	std::string format_pattern_row_channel( std::int32_t pattern, std::int32_t row, std::int32_t channel, std::size_t width = 0, bool pad = true ) const;
	//! Get highlighting information for formatted pattern content
	/*!
	  \param pattern The pattern whose data should be retrieved.
	  \param row The row from which the data should be retrieved.
	  \param channel The channel from which the data should be retrieved.
	  \param width The maximum number of characters the string should contain. 0 means no limit.
	  \param pad If true, the string will be resized to the exact length provided in the width parameter.
	  \return The highlighting string for the formatted pattern data as retrieved by openmpt::module::format_pattern_row_channel at the given pattern position.
	  \sa openmpt::module::format_pattern_row_channel
	*/
	std::string highlight_pattern_row_channel( std::int32_t pattern, std::int32_t row, std::int32_t channel, std::size_t width = 0, bool pad = true ) const;

	//! Retrieve supported ctl keys
	/*!
	  \return A vector containing all supported ctl keys.
	  \remarks Currently supported ctl values are:
	           - load.skip_samples: Set to "1" to avoid loading samples into memory
	           - load.skip_patterns: Set to "1" to avoid loading patterns into memory
	           - load.skip_plugins: Set to "1" to avoid loading plugins
	           - load.skip_subsongs_init: Set to "1" to avoid pre-initializing sub-songs. Skipping results in faster module loading but slower seeking.
	           - seek.sync_samples: Set to "1" to sync sample playback when using openmpt::module::set_position_seconds or openmpt::module::set_position_order_row.
	           - subsong: The current subsong. Setting it has identical semantics as openmpt::module::select_subsong(), getting it returns the currently selected subsong.
	           - play.tempo_factor: Set a floating point tempo factor. "1.0" is the default tempo.
	           - play.pitch_factor: Set a floating point pitch factor. "1.0" is the default pitch.
	           - render.resampler.emulate_amiga: Set to "1" to enable the Amiga resampler for Amiga modules. This emulates the sound characteristics of the Paula chip and overrides the selected interpolation filter. Non-Amiga module formats are not affected by this setting. 
	           - dither: Set the dither algorithm that is used for the 16 bit versions of openmpt::module::read. Supported values are:
	                     - 0: No dithering.
	                     - 1: Default mode. Chosen by OpenMPT code, might change.
	                     - 2: Rectangular, 0.5 bit depth, no noise shaping (original ModPlug Tracker).
	                     - 3: Rectangular, 1 bit depth, simple 1st order noise shaping

	           An exclamation mark ("!") or a question mark ("?") can be appended to any ctl key in order to influence the behaviour in case of an unknown ctl key. "!" causes an exception to be thrown; "?" causes the ctl to be silently ignored. In case neither is appended to the key name, unknown init_ctls are ignored by default and other ctls throw an exception by default.
	*/
	std::vector<std::string> get_ctls() const;

	//! Get current ctl value
	/*!
	  \param ctl The ctl key whose value should be retrieved.
	  \return The associated ctl value.
	  \sa openmpt::module::get_ctls
	*/
	std::string ctl_get( const std::string & ctl ) const;
	//! Set ctl value
	/*!
	  \param ctl The ctl key whose value should be set.
	  \param value The value that should be set.
	  \throws openmpt::exception Throws an exception derived from openmpt::exception in case the value is not sensible (e.g. negative tempo factor) or under the circumstances outlined in openmpt::module::get_ctls.
	  \sa openmpt::module::get_ctls
	*/
	void ctl_set( const std::string & ctl, const std::string & value );

	// remember to add new functions to both C and C++ interfaces and to increase OPENMPT_API_VERSION_MINOR

}; // class module

} // namespace openmpt

/*!
  @}
*/

#endif // LIBOPENMPT_HPP

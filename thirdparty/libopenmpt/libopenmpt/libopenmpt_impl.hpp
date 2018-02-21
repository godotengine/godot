/*
 * libopenmpt_impl.hpp
 * -------------------
 * Purpose: libopenmpt private interface
 * Notes  : This is not a public header. Do NOT ship in distributions dev packages.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#ifndef LIBOPENMPT_IMPL_HPP
#define LIBOPENMPT_IMPL_HPP

#include "libopenmpt_internal.h"
#include "libopenmpt.hpp"

#include <iosfwd>
#include <memory>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4512) // assignment operator could not be generated
#endif

// forward declarations
namespace OpenMPT {
class FileReaderTraitsStdStream;
typedef FileReaderTraitsStdStream FileReaderTraitsDefault;
namespace detail {
template <typename Tbase>
class FileReader;
} // namespace detail
typedef detail::FileReader<FileReaderTraitsDefault> FileReader;
class CSoundFile;
class Dither;
} // namespace OpenMPT

namespace openmpt {

namespace version {

std::uint32_t get_library_version();
std::uint32_t get_core_version();
std::string get_string( const std::string & key );

} // namespace version

class log_interface {
protected:
	log_interface();
public:
	virtual ~log_interface();
	virtual void log( const std::string & message ) const = 0;
}; // class log_interface

class std_ostream_log : public log_interface {
private:
	std::ostream & destination;
public:
	std_ostream_log( std::ostream & dst );
	virtual ~std_ostream_log();
	virtual void log( const std::string & message ) const;
}; // class CSoundFileLog_std_ostream

class log_forwarder;

struct callback_stream_wrapper {
	void * stream;
	std::size_t (*read)( void * stream, void * dst, std::size_t bytes );
	int (*seek)( void * stream, std::int64_t offset, int whence );
	std::int64_t (*tell)( void * stream );
}; // struct callback_stream_wrapper

class module_impl {
protected:
	struct subsong_data {
		double duration;
		std::int32_t start_row;
		std::int32_t start_order;
		std::int32_t sequence;
		subsong_data( double duration, std::int32_t start_row, std::int32_t start_order, std::int32_t sequence );
	}; // struct subsong_data
	typedef std::vector<subsong_data> subsongs_type;
	static const std::int32_t all_subsongs = -1;
	std::unique_ptr<log_interface> m_Log;
	std::unique_ptr<log_forwarder> m_LogForwarder;
	std::int32_t m_current_subsong;
	double m_currentPositionSeconds;
	std::unique_ptr<OpenMPT::CSoundFile> m_sndFile;
	bool m_loaded;
	std::unique_ptr<OpenMPT::Dither> m_Dither;
	subsongs_type m_subsongs;
	float m_Gain;
	bool m_ctl_load_skip_samples;
	bool m_ctl_load_skip_patterns;
	bool m_ctl_load_skip_plugins;
	bool m_ctl_load_skip_subsongs_init;
	bool m_ctl_seek_sync_samples;
	std::vector<std::string> m_loaderMessages;
public:
	void PushToCSoundFileLog( const std::string & text ) const;
	void PushToCSoundFileLog( int loglevel, const std::string & text ) const;
protected:
	std::string mod_string_to_utf8( const std::string & encoded ) const;
	void apply_mixer_settings( std::int32_t samplerate, int channels );
	void apply_libopenmpt_defaults();
	subsongs_type get_subsongs() const;
	void init_subsongs( subsongs_type & subsongs ) const;
	bool has_subsongs_inited() const;
	void ctor( const std::map< std::string, std::string > & ctls );
	void load( const OpenMPT::FileReader & file, const std::map< std::string, std::string > & ctls );
	bool is_loaded() const;
	std::size_t read_wrapper( std::size_t count, std::int16_t * left, std::int16_t * right, std::int16_t * rear_left, std::int16_t * rear_right );
	std::size_t read_wrapper( std::size_t count, float * left, float * right, float * rear_left, float * rear_right );
	std::size_t read_interleaved_wrapper( std::size_t count, std::size_t channels, std::int16_t * interleaved );
	std::size_t read_interleaved_wrapper( std::size_t count, std::size_t channels, float * interleaved );
	std::pair< std::string, std::string > format_and_highlight_pattern_row_channel_command( std::int32_t p, std::int32_t r, std::int32_t c, int command ) const;
	std::pair< std::string, std::string > format_and_highlight_pattern_row_channel( std::int32_t p, std::int32_t r, std::int32_t c, std::size_t width, bool pad ) const;
	static double could_open_probability( const OpenMPT::FileReader & file, double effort, std::unique_ptr<log_interface> log );
public:
	static std::vector<std::string> get_supported_extensions();
	static bool is_extension_supported( const char * extension );
	static bool is_extension_supported( const std::string & extension );
	static double could_open_probability( callback_stream_wrapper stream, double effort, std::unique_ptr<log_interface> log );
	static double could_open_probability( std::istream & stream, double effort, std::unique_ptr<log_interface> log );
	static std::size_t probe_file_header_get_recommended_size();
	static int probe_file_header( std::uint64_t flags, const std::uint8_t * data, std::size_t size, std::uint64_t filesize );
	static int probe_file_header( std::uint64_t flags, const void * data, std::size_t size, std::uint64_t filesize );
	static int probe_file_header( std::uint64_t flags, const std::uint8_t * data, std::size_t size );
	static int probe_file_header( std::uint64_t flags, const void * data, std::size_t size );
	static int probe_file_header( std::uint64_t flags, std::istream & stream );
	static int probe_file_header( std::uint64_t flags, callback_stream_wrapper stream );
	module_impl( callback_stream_wrapper stream, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_impl( std::istream & stream, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_impl( const std::vector<std::uint8_t> & data, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_impl( const std::vector<char> & data, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_impl( const std::uint8_t * data, std::size_t size, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_impl( const char * data, std::size_t size, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_impl( const void * data, std::size_t size, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	~module_impl();
public:
	void select_subsong( std::int32_t subsong );
	std::int32_t get_selected_subsong() const;
	void set_repeat_count( std::int32_t repeat_count );
	std::int32_t get_repeat_count() const;
	double get_duration_seconds() const;
	double set_position_seconds( double seconds );
	double get_position_seconds() const;
	double set_position_order_row( std::int32_t order, std::int32_t row );
	std::int32_t get_render_param( int param ) const;
	void set_render_param( int param, std::int32_t value );
	std::size_t read( std::int32_t samplerate, std::size_t count, std::int16_t * mono );
	std::size_t read( std::int32_t samplerate, std::size_t count, std::int16_t * left, std::int16_t * right );
	std::size_t read( std::int32_t samplerate, std::size_t count, std::int16_t * left, std::int16_t * right, std::int16_t * rear_left, std::int16_t * rear_right );
	std::size_t read( std::int32_t samplerate, std::size_t count, float * mono );
	std::size_t read( std::int32_t samplerate, std::size_t count, float * left, float * right );
	std::size_t read( std::int32_t samplerate, std::size_t count, float * left, float * right, float * rear_left, float * rear_right );
	std::size_t read_interleaved_stereo( std::int32_t samplerate, std::size_t count, std::int16_t * interleaved_stereo );
	std::size_t read_interleaved_quad( std::int32_t samplerate, std::size_t count, std::int16_t * interleaved_quad );
	std::size_t read_interleaved_stereo( std::int32_t samplerate, std::size_t count, float * interleaved_stereo );
	std::size_t read_interleaved_quad( std::int32_t samplerate, std::size_t count, float * interleaved_quad );
	std::vector<std::string> get_metadata_keys() const;
	std::string get_metadata( const std::string & key ) const;
	std::int32_t get_current_speed() const;
	std::int32_t get_current_tempo() const;
	std::int32_t get_current_order() const;
	std::int32_t get_current_pattern() const;
	std::int32_t get_current_row() const;
	std::int32_t get_current_playing_channels() const;
	float get_current_channel_vu_mono( std::int32_t channel ) const;
	float get_current_channel_vu_left( std::int32_t channel ) const;
	float get_current_channel_vu_right( std::int32_t channel ) const;
	float get_current_channel_vu_rear_left( std::int32_t channel ) const;
	float get_current_channel_vu_rear_right( std::int32_t channel ) const;
	std::int32_t get_num_subsongs() const;
	std::int32_t get_num_channels() const;
	std::int32_t get_num_orders() const;
	std::int32_t get_num_patterns() const;
	std::int32_t get_num_instruments() const;
	std::int32_t get_num_samples() const;
	std::vector<std::string> get_subsong_names() const;
	std::vector<std::string> get_channel_names() const;
	std::vector<std::string> get_order_names() const;
	std::vector<std::string> get_pattern_names() const;
	std::vector<std::string> get_instrument_names() const;
	std::vector<std::string> get_sample_names() const;
	std::int32_t get_order_pattern( std::int32_t o ) const;
	std::int32_t get_pattern_num_rows( std::int32_t p ) const;
	std::uint8_t get_pattern_row_channel_command( std::int32_t p, std::int32_t r, std::int32_t c, int cmd ) const;
	std::string format_pattern_row_channel_command( std::int32_t p, std::int32_t r, std::int32_t c, int cmd ) const;
	std::string highlight_pattern_row_channel_command( std::int32_t p, std::int32_t r, std::int32_t c, int cmd ) const;
	std::string format_pattern_row_channel( std::int32_t p, std::int32_t r, std::int32_t c, std::size_t width, bool pad ) const;
	std::string highlight_pattern_row_channel( std::int32_t p, std::int32_t r, std::int32_t c, std::size_t width, bool pad ) const;
	std::vector<std::string> get_ctls() const;
	std::string ctl_get( std::string ctl, bool throw_if_unknown = true ) const;
	void ctl_set( std::string ctl, const std::string & value, bool throw_if_unknown = true );
}; // class module_impl

namespace helper {

template<typename T, typename... Args> std::unique_ptr<T> make_unique(Args&&... args) {
	return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace helper

} // namespace openmpt

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif // LIBOPENMPT_IMPL_HPP

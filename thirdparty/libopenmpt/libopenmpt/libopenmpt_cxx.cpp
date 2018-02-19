/*
 * libopenmpt_cxx.cpp
 * ------------------
 * Purpose: libopenmpt C++ interface implementation
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#include "BuildSettings.h"

#include "libopenmpt_internal.h"
#include "libopenmpt.hpp"
#include "libopenmpt_ext.hpp"

#include "libopenmpt_impl.hpp"
#include "libopenmpt_ext_impl.hpp"

#include <algorithm>
#include <stdexcept>

#include <cstdlib>
#include <cstring>

namespace openmpt {

exception::exception( const std::string & text_ ) noexcept
	: std::exception()
	, text(0)
{
	text = static_cast<char*>( std::malloc( text_.length() + 1 ) );
	if ( text ) {
		std::memcpy( text, text_.c_str(), text_.length() + 1 );
	}
}

exception::exception( const exception & other ) noexcept
	: std::exception()
	, text(0)
{
	const char * const text_ = ( other.what() ? other.what() : "" );
	text = static_cast<char*>( std::malloc( std::strlen( text_ ) + 1 ) );
	if ( text ) {
		std::memcpy( text, text_, std::strlen( text_ ) + 1 );
	}
}

exception::exception( exception && other ) noexcept
	: std::exception()
	, text(0)
{
	text = std::move( other.text );
	other.text = 0;
}

exception & exception::operator = ( const exception & other ) noexcept {
	if ( this == &other ) {
		return *this;
	}
	if ( text ) {
		std::free( text );
		text = 0;
	}
	const char * const text_ = ( other.what() ? other.what() : "" );
	text = static_cast<char*>( std::malloc( std::strlen( text_ ) + 1 ) );
	if ( text ) {
		std::memcpy( text, text_, std::strlen( text_ ) + 1 );
	}
	return *this;
}

exception & exception::operator = ( exception && other ) noexcept {
	if ( this == &other ) {
		return *this;
	}
	if ( text ) {
		std::free( text );
		text = 0;
	}
	text = std::move( other.text );
	other.text = 0;
	return *this;
}

exception::~exception() noexcept {
	if ( text ) {
		std::free( text );
		text = 0;
	}
}

const char * exception::what() const noexcept {
	if ( text ) {
		return text;
	} else {
		return "out of memory";
	}
}

std::uint32_t get_library_version() {
	return openmpt::version::get_library_version();
}

std::uint32_t get_core_version() {
	return openmpt::version::get_core_version();
}

namespace string {

std::string get( const std::string & key ) {
	return openmpt::version::get_string( key );
}

} // namespace string

} // namespace openmpt

#ifndef NO_LIBOPENMPT_CXX

namespace openmpt {

std::vector<std::string> get_supported_extensions() {
	return openmpt::module_impl::get_supported_extensions();
}

bool is_extension_supported( const std::string & extension ) {
	return openmpt::module_impl::is_extension_supported( extension );
}

double could_open_probability( std::istream & stream, double effort, std::ostream & log ) {
	return openmpt::module_impl::could_open_probability( stream, effort, openmpt::helper::make_unique<std_ostream_log>( log ) );
}
double could_open_propability( std::istream & stream, double effort, std::ostream & log ) {
	return openmpt::module_impl::could_open_probability( stream, effort, openmpt::helper::make_unique<std_ostream_log>( log ) );
}

std::size_t probe_file_header_get_recommended_size() {
	return openmpt::module_impl::probe_file_header_get_recommended_size();
}
int probe_file_header( std::uint64_t flags, const std::uint8_t * data, std::size_t size, std::uint64_t filesize ) {
	return openmpt::module_impl::probe_file_header( flags, data, size, filesize );
}
int probe_file_header( std::uint64_t flags, const std::uint8_t * data, std::size_t size ) {
	return openmpt::module_impl::probe_file_header( flags, data, size );
}
int probe_file_header( std::uint64_t flags, std::istream & stream ) {
	return openmpt::module_impl::probe_file_header( flags, stream );
}

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4702) // unreachable code
#endif // _MSC_VER

module::module( const module & ) {
	throw exception("openmpt::module is non-copyable");
}

void module::operator = ( const module & ) {
	throw exception("openmpt::module is non-copyable");
}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif // _MSC_VER

module::module() : impl(0) {
	return;
}

void module::set_impl( module_impl * i ) {
	impl = i;
}

module::module( std::istream & stream, std::ostream & log, const std::map< std::string, std::string > & ctls ) : impl(0) {
	impl = new module_impl( stream, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
}

module::module( const std::vector<std::uint8_t> & data, std::ostream & log, const std::map< std::string, std::string > & ctls ) : impl(0) {
	impl = new module_impl( data, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
}

module::module( const std::uint8_t * beg, const std::uint8_t * end, std::ostream & log, const std::map< std::string, std::string > & ctls ) : impl(0) {
	impl = new module_impl( beg, end - beg, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
}

module::module( const std::uint8_t * data, std::size_t size, std::ostream & log, const std::map< std::string, std::string > & ctls ) : impl(0) {
	impl = new module_impl( data, size, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
}

module::module( const std::vector<char> & data, std::ostream & log, const std::map< std::string, std::string > & ctls ) : impl(0) {
	impl = new module_impl( data, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
}

module::module( const char * beg, const char * end, std::ostream & log, const std::map< std::string, std::string > & ctls ) : impl(0) {
	impl = new module_impl( beg, end - beg, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
}

module::module( const char * data, std::size_t size, std::ostream & log, const std::map< std::string, std::string > & ctls ) : impl(0) {
	impl = new module_impl( data, size, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
}

module::module( const void * data, std::size_t size, std::ostream & log, const std::map< std::string, std::string > & ctls ) : impl(0) {
	impl = new module_impl( data, size, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
}

module::~module() {
	delete impl;
	impl = 0;
}

void module::select_subsong( std::int32_t subsong ) {
	impl->select_subsong( subsong );
}
std::int32_t module::get_selected_subsong() const {
	return impl->get_selected_subsong();
}

void module::set_repeat_count( std::int32_t repeat_count ) {
	impl->set_repeat_count( repeat_count );
}
std::int32_t module::get_repeat_count() const {
	return impl->get_repeat_count();
}

double module::get_duration_seconds() const {
	return impl->get_duration_seconds();
}

double module::set_position_seconds( double seconds ) {
	return impl->set_position_seconds( seconds );
}
double module::get_position_seconds() const {
	return impl->get_position_seconds();
}

double module::set_position_order_row( std::int32_t order, std::int32_t row ) {
	return impl->set_position_order_row( order, row );
}

std::int32_t module::get_render_param( int param ) const {
	return impl->get_render_param( param );
}
void module::set_render_param( int param, std::int32_t value ) {
	impl->set_render_param( param, value );
}

std::size_t module::read( std::int32_t samplerate, std::size_t count, std::int16_t * mono ) {
	return impl->read( samplerate, count, mono );
}
std::size_t module::read( std::int32_t samplerate, std::size_t count, std::int16_t * left, std::int16_t * right ) {
	return impl->read( samplerate, count, left, right );
}
std::size_t module::read( std::int32_t samplerate, std::size_t count, std::int16_t * left, std::int16_t * right, std::int16_t * rear_left, std::int16_t * rear_right ) {
	return impl->read( samplerate, count, left, right, rear_left, rear_right );
}
std::size_t module::read( std::int32_t samplerate, std::size_t count, float * mono ) {
	return impl->read( samplerate, count, mono );
}
std::size_t module::read( std::int32_t samplerate, std::size_t count, float * left, float * right ) {
	return impl->read( samplerate, count, left, right );
}
std::size_t module::read( std::int32_t samplerate, std::size_t count, float * left, float * right, float * rear_left, float * rear_right ) {
	return impl->read( samplerate, count, left, right, rear_left, rear_right );
}
std::size_t module::read_interleaved_stereo( std::int32_t samplerate, std::size_t count, std::int16_t * interleaved_stereo ) {
	return impl->read_interleaved_stereo( samplerate, count, interleaved_stereo );
}
std::size_t module::read_interleaved_quad( std::int32_t samplerate, std::size_t count, std::int16_t * interleaved_quad ) {
	return impl->read_interleaved_quad( samplerate, count, interleaved_quad );
}
std::size_t module::read_interleaved_stereo( std::int32_t samplerate, std::size_t count, float * interleaved_stereo ) {
	return impl->read_interleaved_stereo( samplerate, count, interleaved_stereo );
}
std::size_t module::read_interleaved_quad( std::int32_t samplerate, std::size_t count, float * interleaved_quad ) {
	return impl->read_interleaved_quad( samplerate, count, interleaved_quad );
}

std::vector<std::string> module::get_metadata_keys() const {
	return impl->get_metadata_keys();
}
std::string module::get_metadata( const std::string & key ) const {
	return impl->get_metadata( key );
}

std::int32_t module::get_current_speed() const {
	return impl->get_current_speed();
}
std::int32_t module::get_current_tempo() const {
	return impl->get_current_tempo();
}
std::int32_t module::get_current_order() const {
	return impl->get_current_order();
}
std::int32_t module::get_current_pattern() const {
	return impl->get_current_pattern();
}
std::int32_t module::get_current_row() const {
	return impl->get_current_row();
}
std::int32_t module::get_current_playing_channels() const {
	return impl->get_current_playing_channels();
}

float module::get_current_channel_vu_mono( std::int32_t channel ) const {
	return impl->get_current_channel_vu_mono( channel );
}
float module::get_current_channel_vu_left( std::int32_t channel ) const {
	return impl->get_current_channel_vu_left( channel );
}
float module::get_current_channel_vu_right( std::int32_t channel ) const {
	return impl->get_current_channel_vu_right( channel );
}
float module::get_current_channel_vu_rear_left( std::int32_t channel ) const {
	return impl->get_current_channel_vu_rear_left( channel );
}
float module::get_current_channel_vu_rear_right( std::int32_t channel ) const {
	return impl->get_current_channel_vu_rear_right( channel );
}

std::int32_t module::get_num_subsongs() const {
	return impl->get_num_subsongs();
}
std::int32_t module::get_num_channels() const {
	return impl->get_num_channels();
}
std::int32_t module::get_num_orders() const {
	return impl->get_num_orders();
}
std::int32_t module::get_num_patterns() const {
	return impl->get_num_patterns();
}
std::int32_t module::get_num_instruments() const {
	return impl->get_num_instruments();
}
std::int32_t module::get_num_samples() const {
	return impl->get_num_samples();
}

std::vector<std::string> module::get_subsong_names() const {
	return impl->get_subsong_names();
}
std::vector<std::string> module::get_channel_names() const {
	return impl->get_channel_names();
}
std::vector<std::string> module::get_order_names() const {
	return impl->get_order_names();
}
std::vector<std::string> module::get_pattern_names() const {
	return impl->get_pattern_names();
}
std::vector<std::string> module::get_instrument_names() const {
	return impl->get_instrument_names();
}
std::vector<std::string> module::get_sample_names() const {
	return impl->get_sample_names();
}

std::int32_t module::get_order_pattern( std::int32_t order ) const {
	return impl->get_order_pattern( order );
}
std::int32_t module::get_pattern_num_rows( std::int32_t pattern ) const {
	return impl->get_pattern_num_rows( pattern );
}

std::uint8_t module::get_pattern_row_channel_command( std::int32_t pattern, std::int32_t row, std::int32_t channel, int command ) const {
	return impl->get_pattern_row_channel_command( pattern, row, channel, command );
}

std::string module::format_pattern_row_channel_command( std::int32_t pattern, std::int32_t row, std::int32_t channel, int command ) const {
	return impl->format_pattern_row_channel_command( pattern, row, channel, command );
}
std::string module::highlight_pattern_row_channel_command( std::int32_t pattern, std::int32_t row, std::int32_t channel, int command ) const {
	return impl->highlight_pattern_row_channel_command( pattern, row, channel, command );
}

std::string module::format_pattern_row_channel( std::int32_t pattern, std::int32_t row, std::int32_t channel, std::size_t width, bool pad ) const {
	return impl->format_pattern_row_channel( pattern, row, channel, width, pad );
}
std::string module::highlight_pattern_row_channel( std::int32_t pattern, std::int32_t row, std::int32_t channel, std::size_t width, bool pad ) const {
	return impl->highlight_pattern_row_channel( pattern, row, channel, width, pad );
}

std::vector<std::string> module::get_ctls() const {
	return impl->get_ctls();
}
std::string module::ctl_get( const std::string & ctl ) const {
	return impl->ctl_get( ctl );
}
void module::ctl_set( const std::string & ctl, const std::string & value ) {
	impl->ctl_set( ctl, value );
}

module_ext::module_ext( std::istream & stream, std::ostream & log, const std::map< std::string, std::string > & ctls ) : ext_impl(0) {
	ext_impl = new module_ext_impl( stream, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
	set_impl( ext_impl );
}
module_ext::module_ext( const std::vector<char> & data, std::ostream & log, const std::map< std::string, std::string > & ctls ) : ext_impl(0) {
	ext_impl = new module_ext_impl( data, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
	set_impl( ext_impl );
}
module_ext::module_ext( const char * data, std::size_t size, std::ostream & log, const std::map< std::string, std::string > & ctls ) : ext_impl(0) {
	ext_impl = new module_ext_impl( data, size, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
	set_impl( ext_impl );
}
module_ext::module_ext( const void * data, std::size_t size, std::ostream & log, const std::map< std::string, std::string > & ctls ) : ext_impl(0) {
	ext_impl = new module_ext_impl( data, size, openmpt::helper::make_unique<std_ostream_log>( log ), ctls );
	set_impl( ext_impl );
}
module_ext::~module_ext() {
	set_impl( 0 );
	delete ext_impl;
	ext_impl = 0;
}

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4702) // unreachable code
#endif // _MSC_VER
module_ext::module_ext( const module_ext & other ) : module(other) {
	throw std::runtime_error("openmpt::module_ext is non-copyable");
}
void module_ext::operator = ( const module_ext & ) {
	throw std::runtime_error("openmpt::module_ext is non-copyable");
}
#if defined(_MSC_VER)
#pragma warning(pop)
#endif // _MSC_VER

void * module_ext::get_interface( const std::string & interface_id ) {
	return ext_impl->get_interface( interface_id );
}

} // namespace openmpt

#endif // NO_LIBOPENMPT_CXX

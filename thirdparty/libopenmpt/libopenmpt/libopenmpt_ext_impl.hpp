/*
 * libopenmpt_ext_impl.hpp
 * -----------------------
 * Purpose: libopenmpt extensions - implementation header
 * Notes  :
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#ifndef LIBOPENMPT_EXT_IMPL_HPP
#define LIBOPENMPT_EXT_IMPL_HPP

#include "libopenmpt_internal.h"
#include "libopenmpt_impl.hpp"
#include "libopenmpt_ext.hpp"

using namespace OpenMPT;

namespace openmpt {

class module_ext_impl
	: public module_impl
	, public ext::pattern_vis
	, public ext::interactive



	/* add stuff here */



{
public:
	module_ext_impl( callback_stream_wrapper stream, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_ext_impl( std::istream & stream, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_ext_impl( const std::vector<std::uint8_t> & data, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_ext_impl( const std::vector<char> & data, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_ext_impl( const std::uint8_t * data, std::size_t size, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_ext_impl( const char * data, std::size_t size, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );
	module_ext_impl( const void * data, std::size_t size, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls );

private:



	/* add stuff here */



private:

	void ctor();

public:

	~module_ext_impl();

public:

	void * get_interface( const std::string & interface_id );

	// pattern_vis

	virtual effect_type get_pattern_row_channel_volume_effect_type( std::int32_t pattern, std::int32_t row, std::int32_t channel ) const;

	virtual effect_type get_pattern_row_channel_effect_type( std::int32_t pattern, std::int32_t row, std::int32_t channel ) const;

	// interactive

	virtual void set_current_speed( std::int32_t speed );

	virtual void set_current_tempo( std::int32_t tempo );

	virtual void set_tempo_factor( double factor );

	virtual double get_tempo_factor( ) const;

	virtual void set_pitch_factor( double factor );

	virtual double get_pitch_factor( ) const;

	virtual void set_global_volume( double volume );

	virtual double get_global_volume( ) const;
	
	virtual void set_channel_volume( std::int32_t channel, double volume );

	virtual double get_channel_volume( std::int32_t channel ) const;

	virtual void set_channel_mute_status( std::int32_t channel, bool mute );

	virtual bool get_channel_mute_status( std::int32_t channel ) const;
	
	virtual void set_instrument_mute_status( std::int32_t instrument, bool mute );

	virtual bool get_instrument_mute_status( std::int32_t instrument ) const;

	virtual std::int32_t play_note( std::int32_t instrument, std::int32_t note, double volume, double panning );

	virtual void stop_note( std::int32_t channel );


	/* add stuff here */



}; // class module_ext_impl

} // namespace openmpt

#endif // LIBOPENMPT_EXT_IMPL_HPP


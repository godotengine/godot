/*
 * libopenmpt_ext_impl.cpp
 * -----------------------
 * Purpose: libopenmpt extensions - implementation
 * Notes  :
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#include "common/stdafx.h"

#include "libopenmpt_internal.h"
#include "libopenmpt_ext.hpp"

#include "libopenmpt_ext_impl.hpp"

#include <stdexcept>

#include "soundlib/Sndfile.h"

using namespace OpenMPT;

namespace openmpt {

	module_ext_impl::module_ext_impl( callback_stream_wrapper stream, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls ) : module_impl( stream, std::move(log), ctls ) {
		ctor();
	}
	module_ext_impl::module_ext_impl( std::istream & stream, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls ) : module_impl( stream, std::move(log), ctls ) {
		ctor();
	}
	module_ext_impl::module_ext_impl( const std::vector<std::uint8_t> & data, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls ) : module_impl( data, std::move(log), ctls ) {
		ctor();
	}
	module_ext_impl::module_ext_impl( const std::vector<char> & data, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls ) : module_impl( data, std::move(log), ctls ) {
		ctor();
	}
	module_ext_impl::module_ext_impl( const std::uint8_t * data, std::size_t size, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls ) : module_impl( data, size, std::move(log), ctls ) {
		ctor();
	}
	module_ext_impl::module_ext_impl( const char * data, std::size_t size, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls ) : module_impl( data, size, std::move(log), ctls ) {
		ctor();
	}
	module_ext_impl::module_ext_impl( const void * data, std::size_t size, std::unique_ptr<log_interface> log, const std::map< std::string, std::string > & ctls ) : module_impl( data, size, std::move(log), ctls ) {
		ctor();
	}



	void module_ext_impl::ctor() {



		/* add stuff here */



	}



	module_ext_impl::~module_ext_impl() {



		/* add stuff here */



	}



	void * module_ext_impl::get_interface( const std::string & interface_id ) {
		if ( interface_id.empty() ) {
			return 0;
		} else if ( interface_id == ext::pattern_vis_id ) {
			return dynamic_cast< ext::pattern_vis * >( this );
		} else if ( interface_id == ext::interactive_id ) {
			return dynamic_cast< ext::interactive * >( this );



			/* add stuff here */



		} else {
			return 0;
		}
	}

	// pattern_vis

	module_ext_impl::effect_type module_ext_impl::get_pattern_row_channel_volume_effect_type( std::int32_t pattern, std::int32_t row, std::int32_t channel ) const {
		std::uint8_t byte = get_pattern_row_channel_command( pattern, row, channel, module::command_volumeffect );
		switch ( ModCommand::GetVolumeEffectType( byte ) ) {
			case EFFECT_TYPE_NORMAL : return effect_general; break;
			case EFFECT_TYPE_GLOBAL : return effect_global ; break;
			case EFFECT_TYPE_VOLUME : return effect_volume ; break;
			case EFFECT_TYPE_PANNING: return effect_panning; break;
			case EFFECT_TYPE_PITCH  : return effect_pitch  ; break;
			default: return effect_unknown; break;
		}
	}

	module_ext_impl::effect_type module_ext_impl::get_pattern_row_channel_effect_type( std::int32_t pattern, std::int32_t row, std::int32_t channel ) const {
		std::uint8_t byte = get_pattern_row_channel_command( pattern, row, channel, module::command_effect );
		switch ( ModCommand::GetEffectType( byte ) ) {
			case EFFECT_TYPE_NORMAL : return effect_general; break;
			case EFFECT_TYPE_GLOBAL : return effect_global ; break;
			case EFFECT_TYPE_VOLUME : return effect_volume ; break;
			case EFFECT_TYPE_PANNING: return effect_panning; break;
			case EFFECT_TYPE_PITCH  : return effect_pitch  ; break;
			default: return effect_unknown; break;
		}
	}

	// interactive

	void module_ext_impl::set_current_speed( std::int32_t speed ) {
		if ( speed < 1 || speed > 65535 ) {
			throw openmpt::exception("invalid tick count");
		}
		m_sndFile->m_PlayState.m_nMusicSpeed = speed;
	}

	void module_ext_impl::set_current_tempo( std::int32_t tempo ) {
		if ( tempo < 32 || tempo > 512 ) {
			throw openmpt::exception("invalid tempo");
		}
		m_sndFile->m_PlayState.m_nMusicTempo.Set( tempo );
	}

	void module_ext_impl::set_tempo_factor( double factor ) {
		if ( factor <= 0.0 || factor > 4.0 ) {
			throw openmpt::exception("invalid tempo factor");
		}
		m_sndFile->m_nTempoFactor = Util::Round<uint32_t>( 65536.0 / factor );
		m_sndFile->RecalculateSamplesPerTick();
	}

	double module_ext_impl::get_tempo_factor( ) const {
		return 65536.0 / m_sndFile->m_nTempoFactor;
	}

	void module_ext_impl::set_pitch_factor( double factor ) {
		if ( factor <= 0.0 || factor > 4.0 ) {
			throw openmpt::exception("invalid pitch factor");
		}
		m_sndFile->m_nFreqFactor = Util::Round<uint32_t>( 65536.0 * factor );
		m_sndFile->RecalculateSamplesPerTick();
	}

	double module_ext_impl::get_pitch_factor( ) const {
		return m_sndFile->m_nFreqFactor / 65536.0;
	}

	void module_ext_impl::set_global_volume( double volume ) {
		if ( volume < 0.0 || volume > 1.0 ) {
			throw openmpt::exception("invalid global volume");
		}
		m_sndFile->m_PlayState.m_nGlobalVolume = Util::Round<uint32_t>( volume * MAX_GLOBAL_VOLUME );
	}

	double module_ext_impl::get_global_volume( ) const {
		return m_sndFile->m_PlayState.m_nGlobalVolume / static_cast<double>( MAX_GLOBAL_VOLUME );
	}
	
	void module_ext_impl::set_channel_volume( std::int32_t channel, double volume ) {
		if ( channel < 0 || channel >= get_num_channels() ) {
			throw openmpt::exception("invalid channel");
		}
		if ( volume < 0.0 || volume > 1.0 ) {
			throw openmpt::exception("invalid global volume");
		}
		m_sndFile->m_PlayState.Chn[channel].nGlobalVol = Util::Round<std::int32_t>(volume * 64.0);
	}

	double module_ext_impl::get_channel_volume( std::int32_t channel ) const {
		if ( channel < 0 || channel >= get_num_channels() ) {
			throw openmpt::exception("invalid channel");
		}
		return m_sndFile->m_PlayState.Chn[channel].nGlobalVol / 64.0;
	}

	void module_ext_impl::set_channel_mute_status( std::int32_t channel, bool mute ) {
		if ( channel < 0 || channel >= get_num_channels() ) {
			throw openmpt::exception("invalid channel");
		}
		m_sndFile->ChnSettings[channel].dwFlags.set( CHN_MUTE | CHN_SYNCMUTE , mute );
		m_sndFile->m_PlayState.Chn[channel].dwFlags.set( CHN_MUTE | CHN_SYNCMUTE , mute );

		// Also update NNA channels
		for ( CHANNELINDEX i = m_sndFile->GetNumChannels(); i < MAX_CHANNELS; i++)
		{
			if ( m_sndFile->m_PlayState.Chn[i].nMasterChn == channel + 1)
			{
				m_sndFile->m_PlayState.Chn[i].dwFlags.set( CHN_MUTE | CHN_SYNCMUTE, mute );
			}
		}
	}

	bool module_ext_impl::get_channel_mute_status( std::int32_t channel ) const {
		if ( channel < 0 || channel >= get_num_channels() ) {
			throw openmpt::exception("invalid channel");
		}
		return m_sndFile->m_PlayState.Chn[channel].dwFlags[CHN_MUTE];
	}
	
	void module_ext_impl::set_instrument_mute_status( std::int32_t instrument, bool mute ) {
		const bool instrument_mode = get_num_instruments() != 0;
		const int32_t max_instrument = instrument_mode ? get_num_instruments() : get_num_samples();
		if ( instrument < 0 || instrument >= max_instrument ) {
			throw openmpt::exception("invalid instrument");
		}
		if ( instrument_mode ) {
			if ( m_sndFile->Instruments[instrument + 1] != nullptr ) {
				m_sndFile->Instruments[instrument + 1]->dwFlags.set( INS_MUTE, mute );
			}
		} else {
			m_sndFile->GetSample( static_cast<OpenMPT::SAMPLEINDEX>( instrument + 1 ) ).uFlags.set( CHN_MUTE, mute ) ;
		}
	}

	bool module_ext_impl::get_instrument_mute_status( std::int32_t instrument ) const {
		const bool instrument_mode = get_num_instruments() != 0;
		const int32_t max_instrument = instrument_mode ? get_num_instruments() : get_num_samples();
		if ( instrument < 0 || instrument >= max_instrument ) {
			throw openmpt::exception("invalid instrument");
		}
		if ( instrument_mode ) {
			if ( m_sndFile->Instruments[instrument + 1] != nullptr ) {
				return m_sndFile->Instruments[instrument + 1]->dwFlags[INS_MUTE];
			}
			return true;
		} else {
			return m_sndFile->GetSample( static_cast<OpenMPT::SAMPLEINDEX>( instrument + 1 ) ).uFlags[CHN_MUTE];
		}
	}

	std::int32_t module_ext_impl::play_note( std::int32_t instrument, std::int32_t note, double volume, double panning ) {
		const bool instrument_mode = get_num_instruments() != 0;
		const int32_t max_instrument = instrument_mode ? get_num_instruments() : get_num_samples();
		if ( instrument < 0 || instrument >= max_instrument ) {
			throw openmpt::exception("invalid instrument");
		}
		note += NOTE_MIN;
		if ( note < NOTE_MIN || note > NOTE_MAX ) {
			throw openmpt::exception("invalid note");
		}

		// Find a free channel
		CHANNELINDEX free_channel = MAX_CHANNELS - 1;
		// Search for available channel
		for(CHANNELINDEX i = MAX_CHANNELS - 1; i >= get_num_channels(); i--)
		{
			const ModChannel &chn = m_sndFile->m_PlayState.Chn[i];
			if ( chn.nLength == 0 ) {
				free_channel = i;
				break;
			} else if ( chn.dwFlags[CHN_NOTEFADE] ) {
				// We can probably still do better than this.
				free_channel = i;
			}
		}

		ModChannel &chn = m_sndFile->m_PlayState.Chn[free_channel];
		chn.Reset(ModChannel::resetTotal, *m_sndFile, CHANNELINDEX_INVALID);
		chn.nMasterChn = 0;	// remove NNA association
		chn.nNewNote = chn.nLastNote = static_cast<uint8>(note);
		chn.ResetEnvelopes();
		m_sndFile->InstrumentChange(&chn, instrument + 1);
		chn.nFadeOutVol = 0x10000;
		m_sndFile->NoteChange(&chn, note, false, true, true);
		chn.nPan = Util::Round<int32_t>( Clamp( panning * 128.0, -128.0, 128.0 ) + 128.0 );
		chn.nVolume = Util::Round<int32_t>( Clamp( volume * 256.0, 0.0, 256.0 ) );

		return free_channel;
	}

	void module_ext_impl::stop_note( std::int32_t channel ) {
		if ( channel < 0 || channel >= MAX_CHANNELS ) {
			throw openmpt::exception("invalid channel");
		}
		ModChannel &chn = m_sndFile->m_PlayState.Chn[channel];
		chn.nLength = 0;
		chn.pCurrentSample = nullptr;
	}


	/* add stuff here */



} // namespace openmpt


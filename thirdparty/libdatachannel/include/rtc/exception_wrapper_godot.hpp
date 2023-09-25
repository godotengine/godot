#ifndef EXCEPTION_WRAPPER_GODOT_HPP_
#define EXCEPTION_WRAPPER_GODOT_HPP_
#include "rtc/rtc.hpp"

class LibDataChannelExceptionWrapper {
public:
	static void close_data_channel(std::shared_ptr<rtc::DataChannel> p_channel);
	static void close_peer_connection(std::shared_ptr<rtc::PeerConnection> p_peer_connection);
	static bool put_packet(std::shared_ptr<rtc::DataChannel> p_channel, const uint8_t *p_buffer, int32_t p_len, bool p_is_text, std::string &r_error);
	static std::shared_ptr<rtc::DataChannel> create_data_channel(std::shared_ptr<rtc::PeerConnection> p_peer_connection, const char *p_label, rtc::DataChannelInit p_config, std::string &r_error);
	static std::shared_ptr<rtc::PeerConnection> create_peer_connection(const rtc::Configuration &p_config, std::string &r_error);
	static bool create_offer(std::shared_ptr<rtc::PeerConnection> p_peer_connection, std::string &r_error);
	static bool set_remote_description(std::shared_ptr<rtc::PeerConnection> p_peer_connection, const char *p_type, const char *p_sdp, std::string &r_error);
	static bool add_ice_candidate(std::shared_ptr<rtc::PeerConnection> p_peer_connection, const char *p_sdp_mid_name, const char *p_sdp_name, std::string &r_error);
};

#endif // EXCEPTION_WRAPPER_GODOT_H_

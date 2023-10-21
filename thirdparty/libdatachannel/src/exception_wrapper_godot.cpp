#include "rtc/rtc.hpp"
#include "rtc/exception_wrapper_godot.hpp"

void LibDataChannelExceptionWrapper::close_data_channel(std::shared_ptr<rtc::DataChannel> p_channel) try {
	if (p_channel) {
		p_channel->close();
	}
} catch (...) {
}


void LibDataChannelExceptionWrapper::close_peer_connection(std::shared_ptr<rtc::PeerConnection> p_peer_connection) try {
	if (p_peer_connection) {
		p_peer_connection->close();
	}
} catch (...) {
}


bool LibDataChannelExceptionWrapper::put_packet(std::shared_ptr<rtc::DataChannel> p_channel, const uint8_t *p_buffer, int32_t p_len, bool p_is_text, std::string &r_error) try {
	if (p_is_text) {
		std::string str((const char *)p_buffer, (size_t)p_len);
		p_channel->send(str);
	} else {
		p_channel->send(reinterpret_cast<const std::byte *>(p_buffer), p_len);
	}
	return true;
} catch (const std::exception &e) {
	r_error = e.what();
	return false;
}


std::shared_ptr<rtc::DataChannel> LibDataChannelExceptionWrapper::create_data_channel(std::shared_ptr<rtc::PeerConnection> p_peer_connection, const char *p_label, rtc::DataChannelInit p_config, std::string &r_error) try {
	return p_peer_connection->createDataChannel(p_label, p_config);
} catch (const std::exception &e) {
	r_error = e.what();
	return std::shared_ptr<rtc::DataChannel>();
}

std::shared_ptr<rtc::PeerConnection> LibDataChannelExceptionWrapper::create_peer_connection(const rtc::Configuration &p_config, std::string &r_error) try {
	return std::make_shared<rtc::PeerConnection>(p_config);
} catch (const std::exception &e) {
	r_error = e.what();
	return std::shared_ptr<rtc::PeerConnection>();
}

bool LibDataChannelExceptionWrapper::create_offer(std::shared_ptr<rtc::PeerConnection> p_peer_connection, std::string &r_error) try {
	p_peer_connection->setLocalDescription(rtc::Description::Type::Offer);
	return true;
} catch (const std::exception &e) {
	r_error = e.what();
	return false;
}

bool LibDataChannelExceptionWrapper::set_remote_description(std::shared_ptr<rtc::PeerConnection> p_peer_connection, const char *p_type, const char *p_sdp, std::string &r_error) try {
	std::string sdp(p_sdp);
	std::string type(p_type);
	rtc::Description desc(sdp, type);
	p_peer_connection->setRemoteDescription(desc);
	// Automatically create the answer.
	if (type == "offer") {
		p_peer_connection->setLocalDescription(rtc::Description::Type::Answer);
	}
	return true;
} catch (const std::exception &e) {
	r_error = e.what();
	return false;
}

bool LibDataChannelExceptionWrapper::add_ice_candidate(std::shared_ptr<rtc::PeerConnection> p_peer_connection, const char *p_sdp_mid_name, const char *p_sdp_name, std::string &r_error) try {
	rtc::Candidate candidate(p_sdp_name, p_sdp_mid_name);
	p_peer_connection->addRemoteCandidate(candidate);
	return true;
} catch (const std::exception &e) {
	r_error = e.what();
	return false;
}

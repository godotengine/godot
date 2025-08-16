#include <net/inet>
typedef std::vector<uint8_t> buffer_t;

extern void file_server(net::Inet& inet,
        				const uint16_t port,
        				delegate<void(buffer_t)> callback);

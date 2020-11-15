//
//  httplib.h
//
//  Copyright (c) 2017 Yuji Hirose. All rights reserved.
//  MIT License
//

#ifndef _CPPHTTPLIB_HTTPLIB_H_
#define _CPPHTTPLIB_HTTPLIB_H_

#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_NONSTDC_NO_DEPRECATE
#endif

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf_s
#endif

#ifndef S_ISREG
#define S_ISREG(m)  (((m)&S_IFREG)==S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(m)  (((m)&S_IFDIR)==S_IFDIR)
#endif

#include <io.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#undef min
#undef max

#ifndef strcasecmp
#define strcasecmp _stricmp
#endif

typedef SOCKET socket_t;
#else
#include <pthread.h>
#include <unistd.h>
#include <netdb.h>
#include <cstring>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/select.h>

typedef int socket_t;
#define INVALID_SOCKET (-1)
#endif

#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#include <openssl/ssl.h>
#endif

#ifdef CPPHTTPLIB_ZLIB_SUPPORT
#include "third_party/zlib/zlib_crashpad.h"
#endif

/*
 * Configuration
 */
#define CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND 5
#define CPPHTTPLIB_KEEPALIVE_TIMEOUT_USECOND 0

namespace httplib
{

namespace detail {

struct ci {
    bool operator() (const std::string & s1, const std::string & s2) const {
        return std::lexicographical_compare(
            s1.begin(), s1.end(),
            s2.begin(), s2.end(),
            [](char c1, char c2) {
                return ::tolower(c1) < ::tolower(c2);
            });
    }
};

} // namespace detail

enum class HttpVersion { v1_0 = 0, v1_1 };

typedef std::multimap<std::string, std::string, detail::ci>  Headers;

template<typename uint64_t, typename... Args>
std::pair<std::string, std::string> make_range_header(uint64_t value, Args... args);

typedef std::multimap<std::string, std::string>                Params;
typedef std::smatch                                            Match;
typedef std::function<void (uint64_t current, uint64_t total)> Progress;

struct MultipartFile {
    std::string filename;
    std::string content_type;
    size_t offset = 0;
    size_t length = 0;
};
typedef std::multimap<std::string, MultipartFile> MultipartFiles;

struct Request {
    std::string    version;
    std::string    method;
    std::string    target;
    std::string    path;
    Headers        headers;
    std::string    body;
    Params         params;
    MultipartFiles files;
    Match          matches;

    Progress       progress;

    bool has_header(const char* key) const;
    std::string get_header_value(const char* key) const;
    void set_header(const char* key, const char* val);

    bool has_param(const char* key) const;
    std::string get_param_value(const char* key) const;

    bool has_file(const char* key) const;
    MultipartFile get_file_value(const char* key) const;
};

struct Response {
    std::string version;
    int         status;
    Headers     headers;
    std::string body;

    bool has_header(const char* key) const;
    std::string get_header_value(const char* key) const;
    void set_header(const char* key, const char* val);

    void set_redirect(const char* uri);
    void set_content(const char* s, size_t n, const char* content_type);
    void set_content(const std::string& s, const char* content_type);

    Response() : status(-1) {}
};

class Stream {
public:
    virtual ~Stream() {}
    virtual int read(char* ptr, size_t size) = 0;
    virtual int write(const char* ptr, size_t size1) = 0;
    virtual int write(const char* ptr) = 0;
    virtual std::string get_remote_addr() = 0;

    template <typename ...Args>
    void write_format(const char* fmt, const Args& ...args);
};

class SocketStream : public Stream {
public:
    SocketStream(socket_t sock);
    virtual ~SocketStream();

    virtual int read(char* ptr, size_t size);
    virtual int write(const char* ptr, size_t size);
    virtual int write(const char* ptr);
    virtual std::string get_remote_addr();

private:
    socket_t sock_;
};

class Server {
public:
    typedef std::function<void (const Request&, Response&)> Handler;
    typedef std::function<void (const Request&, const Response&)> Logger;

    Server();

    virtual ~Server();

    virtual bool is_valid() const;

    Server& Get(const char* pattern, Handler handler);
    Server& Post(const char* pattern, Handler handler);

    Server& Put(const char* pattern, Handler handler);
    Server& Delete(const char* pattern, Handler handler);
    Server& Options(const char* pattern, Handler handler);

    bool set_base_dir(const char* path);

    void set_error_handler(Handler handler);
    void set_logger(Logger logger);

    void set_keep_alive_max_count(size_t count);

    int bind_to_any_port(const char* host, int socket_flags = 0);
    bool listen_after_bind();

    bool listen(const char* host, int port, int socket_flags = 0);

    bool is_running() const;
    void stop();

protected:
    bool process_request(Stream& strm, bool last_connection, bool& connection_close);

    size_t keep_alive_max_count_;

private:
    typedef std::vector<std::pair<std::regex, Handler>> Handlers;

    socket_t create_server_socket(const char* host, int port, int socket_flags) const;
    int bind_internal(const char* host, int port, int socket_flags);
    bool listen_internal();

    bool routing(Request& req, Response& res);
    bool handle_file_request(Request& req, Response& res);
    bool dispatch_request(Request& req, Response& res, Handlers& handlers);

    bool parse_request_line(const char* s, Request& req);
    void write_response(Stream& strm, bool last_connection, const Request& req, Response& res);

    virtual bool read_and_close_socket(socket_t sock);

    bool        is_running_;
    socket_t    svr_sock_;
    std::string base_dir_;
    Handlers    get_handlers_;
    Handlers    post_handlers_;
    Handlers    put_handlers_;
    Handlers    delete_handlers_;
    Handlers    options_handlers_;
    Handler     error_handler_;
    Logger      logger_;

    // TODO: Use thread pool...
    std::mutex  running_threads_mutex_;
    int         running_threads_;
};

class Client {
public:
    Client(
        const char* host,
        int port = 80,
        size_t timeout_sec = 300);

    virtual ~Client();

    virtual bool is_valid() const;

    std::shared_ptr<Response> Get(const char* path, Progress progress = nullptr);
    std::shared_ptr<Response> Get(const char* path, const Headers& headers, Progress progress = nullptr);

    std::shared_ptr<Response> Head(const char* path);
    std::shared_ptr<Response> Head(const char* path, const Headers& headers);

    std::shared_ptr<Response> Post(const char* path, const std::string& body, const char* content_type);
    std::shared_ptr<Response> Post(const char* path, const Headers& headers, const std::string& body, const char* content_type);

    std::shared_ptr<Response> Post(const char* path, const Params& params);
    std::shared_ptr<Response> Post(const char* path, const Headers& headers, const Params& params);

    std::shared_ptr<Response> Put(const char* path, const std::string& body, const char* content_type);
    std::shared_ptr<Response> Put(const char* path, const Headers& headers, const std::string& body, const char* content_type);

    std::shared_ptr<Response> Delete(const char* path);
    std::shared_ptr<Response> Delete(const char* path, const Headers& headers);

    std::shared_ptr<Response> Options(const char* path);
    std::shared_ptr<Response> Options(const char* path, const Headers& headers);

    bool send(Request& req, Response& res);

protected:
    bool process_request(Stream& strm, Request& req, Response& res, bool& connection_close);

    const std::string host_;
    const int         port_;
    size_t            timeout_sec_;
    const std::string host_and_port_;

private:
    socket_t create_client_socket() const;
    bool read_response_line(Stream& strm, Response& res);
    void write_request(Stream& strm, Request& req);

    virtual bool read_and_close_socket(socket_t sock, Request& req, Response& res);
};

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
class SSLSocketStream : public Stream {
public:
    SSLSocketStream(socket_t sock, SSL* ssl);
    virtual ~SSLSocketStream();

    virtual int read(char* ptr, size_t size);
    virtual int write(const char* ptr, size_t size);
    virtual int write(const char* ptr);
    virtual std::string get_remote_addr();

private:
    socket_t sock_;
    SSL* ssl_;
};

class SSLServer : public Server {
public:
    SSLServer(
        const char* cert_path, const char* private_key_path);

    virtual ~SSLServer();

    virtual bool is_valid() const;

private:
    virtual bool read_and_close_socket(socket_t sock);

    SSL_CTX* ctx_;
    std::mutex ctx_mutex_;
};

class SSLClient : public Client {
public:
    SSLClient(
        const char* host,
        int port = 80,
        size_t timeout_sec = 300);

    virtual ~SSLClient();

    virtual bool is_valid() const;

private:
    virtual bool read_and_close_socket(socket_t sock, Request& req, Response& res);

    SSL_CTX* ctx_;
    std::mutex ctx_mutex_;
};
#endif

/*
 * Implementation
 */
namespace detail {

template <class Fn>
void split(const char* b, const char* e, char d, Fn fn)
{
    int i = 0;
    int beg = 0;

    while (e ? (b + i != e) : (b[i] != '\0')) {
        if (b[i] == d) {
            fn(&b[beg], &b[i]);
            beg = i + 1;
        }
        i++;
    }

    if (i) {
        fn(&b[beg], &b[i]);
    }
}

// NOTE: until the read size reaches `fixed_buffer_size`, use `fixed_buffer`
// to store data. The call can set memory on stack for performance.
class stream_line_reader {
public:
    stream_line_reader(Stream& strm, char* fixed_buffer, size_t fixed_buffer_size)
        : strm_(strm)
        , fixed_buffer_(fixed_buffer)
        , fixed_buffer_size_(fixed_buffer_size) {
    }

    const char* ptr() const {
        if (glowable_buffer_.empty()) {
            return fixed_buffer_;
        } else {
            return glowable_buffer_.data();
        }
    }

    bool getline() {
        fixed_buffer_used_size_ = 0;
        glowable_buffer_.clear();

        for (size_t i = 0; ; i++) {
            char byte;
            auto n = strm_.read(&byte, 1);

            if (n < 0) {
                return false;
            } else if (n == 0) {
                if (i == 0) {
                    return false;
                } else {
                    break;
                }
            }

            append(byte);

            if (byte == '\n') {
                break;
            }
        }

        return true;
    }

private:
    void append(char c) {
        if (fixed_buffer_used_size_ < fixed_buffer_size_ - 1) {
            fixed_buffer_[fixed_buffer_used_size_++] = c;
            fixed_buffer_[fixed_buffer_used_size_] = '\0';
        } else {
            if (glowable_buffer_.empty()) {
                assert(fixed_buffer_[fixed_buffer_used_size_] == '\0');
                glowable_buffer_.assign(fixed_buffer_, fixed_buffer_used_size_);
            }
            glowable_buffer_ += c;
        }
    }

    Stream& strm_;
    char* fixed_buffer_;
    const size_t fixed_buffer_size_;
    size_t fixed_buffer_used_size_;
    std::string glowable_buffer_;
};

inline int close_socket(socket_t sock)
{
#ifdef _WIN32
    return closesocket(sock);
#else
    return close(sock);
#endif
}

inline int select_read(socket_t sock, size_t sec, size_t usec)
{
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(sock, &fds);

    timeval tv;
    tv.tv_sec = sec;
    tv.tv_usec = usec;

    return select(sock + 1, &fds, NULL, NULL, &tv);
}

inline bool wait_until_socket_is_ready(socket_t sock, size_t sec, size_t usec)
{
    fd_set fdsr;
    FD_ZERO(&fdsr);
    FD_SET(sock, &fdsr);

    auto fdsw = fdsr;
    auto fdse = fdsr;

    timeval tv;
    tv.tv_sec = sec;
    tv.tv_usec = usec;

    if (select(sock + 1, &fdsr, &fdsw, &fdse, &tv) < 0) {
        return false;
    } else if (FD_ISSET(sock, &fdsr) || FD_ISSET(sock, &fdsw)) {
        int error = 0;
        socklen_t len = sizeof(error);
        if (getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&error, &len) < 0 || error) {
            return false;
        }
    } else {
        return false;
    }

    return true;
}

template <typename T>
inline bool read_and_close_socket(socket_t sock, size_t keep_alive_max_count, T callback)
{
    bool ret = false;

    if (keep_alive_max_count > 0) {
        auto count = keep_alive_max_count;
        while (count > 0 &&
               detail::select_read(sock,
                   CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND,
                   CPPHTTPLIB_KEEPALIVE_TIMEOUT_USECOND) > 0) {
            SocketStream strm(sock);
            auto last_connection = count == 1;
            auto connection_close = false;

            ret = callback(strm, last_connection, connection_close);
            if (!ret || connection_close) {
                break;
            }

            count--;
        }
    } else {
        SocketStream strm(sock);
        auto dummy_connection_close = false;
        ret = callback(strm, true, dummy_connection_close);
    }

    close_socket(sock);
    return ret;
}

inline int shutdown_socket(socket_t sock)
{
#ifdef _WIN32
    return shutdown(sock, SD_BOTH);
#else
    return shutdown(sock, SHUT_RDWR);
#endif
}

template <typename Fn>
socket_t create_socket(const char* host, int port, Fn fn, int socket_flags = 0)
{
#ifdef _WIN32
#define SO_SYNCHRONOUS_NONALERT 0x20
#define SO_OPENTYPE 0x7008

    int opt = SO_SYNCHRONOUS_NONALERT;
    setsockopt(INVALID_SOCKET, SOL_SOCKET, SO_OPENTYPE, (char*)&opt, sizeof(opt));
#endif

    // Get address info
    struct addrinfo hints;
    struct addrinfo *result;

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = socket_flags;
    hints.ai_protocol = 0;

    auto service = std::to_string(port);

    if (getaddrinfo(host, service.c_str(), &hints, &result)) {
        return INVALID_SOCKET;
    }

    for (auto rp = result; rp; rp = rp->ai_next) {
       // Create a socket
       auto sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
       if (sock == INVALID_SOCKET) {
          continue;
       }

       // Make 'reuse address' option available
       int yes = 1;
       setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&yes, sizeof(yes));

       // bind or connect
       if (fn(sock, *rp)) {
          freeaddrinfo(result);
          return sock;
       }

       close_socket(sock);
    }

    freeaddrinfo(result);
    return INVALID_SOCKET;
}

inline void set_nonblocking(socket_t sock, bool nonblocking)
{
#ifdef _WIN32
    auto flags = nonblocking ? 1UL : 0UL;
    ioctlsocket(sock, FIONBIO, &flags);
#else
    auto flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, nonblocking ? (flags | O_NONBLOCK) : (flags & (~O_NONBLOCK)));
#endif
}

inline bool is_connection_error()
{
#ifdef _WIN32
    return WSAGetLastError() != WSAEWOULDBLOCK;
#else
    return errno != EINPROGRESS;
#endif
}

inline std::string get_remote_addr(socket_t sock) {
    struct sockaddr_storage addr;
    socklen_t len = sizeof(addr);

    if (!getpeername(sock, (struct sockaddr*)&addr, &len)) {
        char ipstr[NI_MAXHOST];

        if (!getnameinfo((struct sockaddr*)&addr, len,
            ipstr, sizeof(ipstr), nullptr, 0, NI_NUMERICHOST)) {
            return ipstr;
        }
    }

    return std::string();
}

inline bool is_file(const std::string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) >= 0 && S_ISREG(st.st_mode);
}

inline bool is_dir(const std::string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) >= 0 && S_ISDIR(st.st_mode);
}

inline bool is_valid_path(const std::string& path) {
    size_t level = 0;
    size_t i = 0;

    // Skip slash
    while (i < path.size() && path[i] == '/') {
        i++;
    }

    while (i < path.size()) {
        // Read component
        auto beg = i;
        while (i < path.size() && path[i] != '/') {
            i++;
        }

        auto len = i - beg;
        assert(len > 0);

        if (!path.compare(beg, len, ".")) {
            ;
        } else if (!path.compare(beg, len, "..")) {
            if (level == 0) {
                return false;
            }
            level--;
        } else {
            level++;
        }

        // Skip slash
        while (i < path.size() && path[i] == '/') {
            i++;
        }
    }

    return true;
}

inline void read_file(const std::string& path, std::string& out)
{
    std::ifstream fs(path, std::ios_base::binary);
    fs.seekg(0, std::ios_base::end);
    auto size = fs.tellg();
    fs.seekg(0);
    out.resize(static_cast<size_t>(size));
    fs.read(&out[0], size);
}

inline std::string file_extension(const std::string& path)
{
    std::smatch m;
    auto pat = std::regex("\\.([a-zA-Z0-9]+)$");
    if (std::regex_search(path, m, pat)) {
        return m[1].str();
    }
    return std::string();
}

inline const char* find_content_type(const std::string& path)
{
    auto ext = file_extension(path);
    if (ext == "txt") {
        return "text/plain";
    } else if (ext == "html") {
        return "text/html";
    } else if (ext == "css") {
        return "text/css";
    } else if (ext == "jpeg" || ext == "jpg") {
        return "image/jpg";
    } else if (ext == "png") {
        return "image/png";
    } else if (ext == "gif") {
        return "image/gif";
    } else if (ext == "svg") {
        return "image/svg+xml";
    } else if (ext == "ico") {
        return "image/x-icon";
    } else if (ext == "json") {
        return "application/json";
    } else if (ext == "pdf") {
        return "application/pdf";
    } else if (ext == "js") {
        return "application/javascript";
    } else if (ext == "xml") {
        return "application/xml";
    } else if (ext == "xhtml") {
        return "application/xhtml+xml";
    }
    return nullptr;
}

inline const char* status_message(int status)
{
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 415: return "Unsupported Media Type";
    default:
        case 500: return "Internal Server Error";
    }
}

inline const char* get_header_value(const Headers& headers, const char* key, const char* def)
{
    auto it = headers.find(key);
    if (it != headers.end()) {
        return it->second.c_str();
    }
    return def;
}

inline int get_header_value_int(const Headers& headers, const char* key, int def)
{
    auto it = headers.find(key);
    if (it != headers.end()) {
        return std::stoi(it->second);
    }
    return def;
}

inline bool read_headers(Stream& strm, Headers& headers)
{
    static std::regex re(R"((.+?):\s*(.+?)\s*\r\n)");

    const auto bufsiz = 2048;
    char buf[bufsiz];

    stream_line_reader reader(strm, buf, bufsiz);

    for (;;) {
        if (!reader.getline()) {
            return false;
        }
        if (!strcmp(reader.ptr(), "\r\n")) {
            break;
        }
        std::cmatch m;
        if (std::regex_match(reader.ptr(), m, re)) {
            auto key = std::string(m[1]);
            auto val = std::string(m[2]);
            headers.emplace(key, val);
        }
    }

    return true;
}

inline bool read_content_with_length(Stream& strm, std::string& out, size_t len, Progress progress)
{
    out.assign(len, 0);
    size_t r = 0;
    while (r < len){
        auto n = strm.read(&out[r], len - r);
        if (n <= 0) {
            return false;
        }

        r += n;

        if (progress) {
            progress(r, len);
        }
    }

    return true;
}

inline bool read_content_without_length(Stream& strm, std::string& out)
{
    for (;;) {
        char byte;
        auto n = strm.read(&byte, 1);
        if (n < 0) {
            return false;
        } else if (n == 0) {
            return true;
        }
        out += byte;
    }

    return true;
}

inline bool read_content_chunked(Stream& strm, std::string& out)
{
    const auto bufsiz = 16;
    char buf[bufsiz];

    stream_line_reader reader(strm, buf, bufsiz);

    if (!reader.getline()) {
        return false;
    }

    auto chunk_len = std::stoi(reader.ptr(), 0, 16);

    while (chunk_len > 0){
        std::string chunk;
        if (!read_content_with_length(strm, chunk, chunk_len, nullptr)) {
            return false;
        }

        if (!reader.getline()) {
            return false;
        }

        if (strcmp(reader.ptr(), "\r\n")) {
            break;
        }

        out += chunk;

        if (!reader.getline()) {
            return false;
        }

        chunk_len = std::stoi(reader.ptr(), 0, 16);
    }

    if (chunk_len == 0) {
        // Reader terminator after chunks
        if (!reader.getline() || strcmp(reader.ptr(), "\r\n"))
            return false;
    }

    return true;
}

template <typename T>
bool read_content(Stream& strm, T& x, Progress progress = Progress())
{
    auto len = get_header_value_int(x.headers, "Content-Length", 0);

    if (len) {
        return read_content_with_length(strm, x.body, len, progress);
    } else {
        const auto& encoding = get_header_value(x.headers, "Transfer-Encoding", "");

        if (!strcasecmp(encoding, "chunked")) {
            return read_content_chunked(strm, x.body);
        } else {
            return read_content_without_length(strm, x.body);
        }
    }

    return true;
}

template <typename T>
inline void write_headers(Stream& strm, const T& info)
{
    for (const auto& x: info.headers) {
        strm.write_format("%s: %s\r\n", x.first.c_str(), x.second.c_str());
    }
    strm.write("\r\n");
}

inline std::string encode_url(const std::string& s)
{
    std::string result;

    for (auto i = 0; s[i]; i++) {
        switch (s[i]) {
        case ' ':  result += "+"; break;
        case '\'': result += "%27"; break;
        case ',':  result += "%2C"; break;
        case ':':  result += "%3A"; break;
        case ';':  result += "%3B"; break;
        default:
            if (s[i] < 0) {
                result += '%';
                char hex[4];
                size_t len = snprintf(hex, sizeof(hex) - 1, "%02X", (unsigned char)s[i]);
                assert(len == 2);
                result.append(hex, len);
            } else {
                result += s[i];
            }
            break;
        }
   }

    return result;
}

inline bool is_hex(char c, int& v)
{
    if (0x20 <= c && isdigit(c)) {
        v = c - '0';
        return true;
    } else if ('A' <= c && c <= 'F') {
        v = c - 'A' + 10;
        return true;
    } else if ('a' <= c && c <= 'f') {
        v = c - 'a' + 10;
        return true;
    }
    return false;
}

inline bool from_hex_to_i(const std::string& s, int i, int cnt, int& val)
{
    val = 0;
    for (; cnt; i++, cnt--) {
        if (!s[i]) {
            return false;
        }
        int v = 0;
        if (is_hex(s[i], v)) {
            val = val * 16 + v;
        } else {
            return false;
        }
    }
    return true;
}

inline size_t to_utf8(int code, char* buff)
{
    if (code < 0x0080) {
        buff[0] = (code & 0x7F);
        return 1;
    } else if (code < 0x0800) {
        buff[0] = (0xC0 | ((code >> 6) & 0x1F));
        buff[1] = (0x80 | (code & 0x3F));
        return 2;
    } else if (code < 0xD800) {
        buff[0] = (0xE0 | ((code >> 12) & 0xF));
        buff[1] = (0x80 | ((code >> 6) & 0x3F));
        buff[2] = (0x80 | (code & 0x3F));
        return 3;
    } else if (code < 0xE000)  { // D800 - DFFF is invalid...
        return 0;
    } else if (code < 0x10000) {
        buff[0] = (0xE0 | ((code >> 12) & 0xF));
        buff[1] = (0x80 | ((code >> 6) & 0x3F));
        buff[2] = (0x80 | (code & 0x3F));
        return 3;
    } else if (code < 0x110000) {
        buff[0] = (0xF0 | ((code >> 18) & 0x7));
        buff[1] = (0x80 | ((code >> 12) & 0x3F));
        buff[2] = (0x80 | ((code >> 6) & 0x3F));
        buff[3] = (0x80 | (code & 0x3F));
        return 4;
    }

    // NOTREACHED
    return 0;
}

inline std::string decode_url(const std::string& s)
{
    std::string result;

    for (int i = 0; s[i]; i++) {
        if (s[i] == '%') {
            if (s[i + 1] && s[i + 1] == 'u') {
                int val = 0;
                if (from_hex_to_i(s, i + 2, 4, val)) {
                    // 4 digits Unicode codes
                    char buff[4];
                    size_t len = to_utf8(val, buff);
                    if (len > 0) {
                        result.append(buff, len);
                    }
                    i += 5; // 'u0000'
                } else {
                    result += s[i];
                }
            } else {
                int val = 0;
                if (from_hex_to_i(s, i + 1, 2, val)) {
                    // 2 digits hex codes
                    result += val;
                    i += 2; // '00'
                } else {
                    result += s[i];
                }
            }
        } else if (s[i] == '+') {
            result += ' ';
        } else {
            result += s[i];
        }
    }

    return result;
}

inline void parse_query_text(const std::string& s, Params& params)
{
    split(&s[0], &s[s.size()], '&', [&](const char* b, const char* e) {
        std::string key;
        std::string val;
        split(b, e, '=', [&](const char* b, const char* e) {
            if (key.empty()) {
                key.assign(b, e);
            } else {
                val.assign(b, e);
            }
        });
        params.emplace(key, decode_url(val));
    });
}

inline bool parse_multipart_boundary(const std::string& content_type, std::string& boundary)
{
    auto pos = content_type.find("boundary=");
    if (pos == std::string::npos) {
        return false;
    }

    boundary = content_type.substr(pos + 9);
    return true;
}

inline bool parse_multipart_formdata(
    const std::string& boundary, const std::string& body, MultipartFiles& files)
{
    static std::string dash = "--";
    static std::string crlf = "\r\n";

    static std::regex re_content_type(
        "Content-Type: (.*?)", std::regex_constants::icase);

    static std::regex re_content_disposition(
        "Content-Disposition: form-data; name=\"(.*?)\"(?:; filename=\"(.*?)\")?",
        std::regex_constants::icase);

    auto dash_boundary = dash + boundary;

    auto pos = body.find(dash_boundary);
    if (pos != 0) {
        return false;
    }

    pos += dash_boundary.size();

    auto next_pos = body.find(crlf, pos);
    if (next_pos == std::string::npos) {
        return false;
    }

    pos = next_pos + crlf.size();

    while (pos < body.size()) {
        next_pos = body.find(crlf, pos);
        if (next_pos == std::string::npos) {
            return false;
        }

        std::string name;
        MultipartFile file;

        auto header = body.substr(pos, (next_pos - pos));

        while (pos != next_pos) {
            std::smatch m;
            if (std::regex_match(header, m, re_content_type)) {
                file.content_type = m[1];
            } else if (std::regex_match(header, m, re_content_disposition)) {
                name = m[1];
                file.filename = m[2];
            }

            pos = next_pos + crlf.size();

            next_pos = body.find(crlf, pos);
            if (next_pos == std::string::npos) {
                return false;
            }

            header = body.substr(pos, (next_pos - pos));
        }

        pos = next_pos + crlf.size();

        next_pos = body.find(crlf + dash_boundary, pos);

        if (next_pos == std::string::npos) {
            return false;
        }

        file.offset = pos;
        file.length = next_pos - pos;

        pos = next_pos + crlf.size() + dash_boundary.size();

        next_pos = body.find(crlf, pos);
        if (next_pos == std::string::npos) {
            return false;
        }

        files.emplace(name, file);

        pos = next_pos + crlf.size();
    }

    return true;
}

inline std::string to_lower(const char* beg, const char* end)
{
    std::string out;
    auto it = beg;
    while (it != end) {
        out += ::tolower(*it);
        it++;
    }
    return out;
}

inline void make_range_header_core(std::string&) {}

template<typename uint64_t>
inline void make_range_header_core(std::string& field, uint64_t value)
{
    if (!field.empty()) {
        field += ", ";
    }
    field += std::to_string(value) + "-";
}

template<typename uint64_t, typename... Args>
inline void make_range_header_core(std::string& field, uint64_t value1, uint64_t value2, Args... args)
{
    if (!field.empty()) {
        field += ", ";
    }
    field += std::to_string(value1) + "-" + std::to_string(value2);
    make_range_header_core(field, args...);
}

#ifdef CPPHTTPLIB_ZLIB_SUPPORT
inline bool can_compress(const std::string& content_type) {
    return !content_type.find("text/") ||
        content_type == "image/svg+xml" ||
        content_type == "application/javascript" ||
        content_type == "application/json" ||
        content_type == "application/xml" ||
        content_type == "application/xhtml+xml";
}

inline void compress(std::string& content)
{
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;

    auto ret = deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 31, 8, Z_DEFAULT_STRATEGY);
    if (ret != Z_OK) {
        return;
    }

    strm.avail_in = content.size();
    strm.next_in = (Bytef *)content.data();

    std::string compressed;

    const auto bufsiz = 16384;
    char buff[bufsiz];
    do {
        strm.avail_out = bufsiz;
        strm.next_out = (Bytef *)buff;
        deflate(&strm, Z_FINISH);
        compressed.append(buff, bufsiz - strm.avail_out);
    } while (strm.avail_out == 0);

    content.swap(compressed);

    deflateEnd(&strm);
}

inline void decompress(std::string& content)
{
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;

    // 15 is the value of wbits, which should be at the maximum possible value to ensure
    // that any gzip stream can be decoded. The offset of 16 specifies that the stream
    // to decompress will be formatted with a gzip wrapper.
    auto ret = inflateInit2(&strm, 16 + 15);
    if (ret != Z_OK) {
        return;
    }

    strm.avail_in = content.size();
    strm.next_in = (Bytef *)content.data();

    std::string decompressed;

    const auto bufsiz = 16384;
    char buff[bufsiz];
    do {
        strm.avail_out = bufsiz;
        strm.next_out = (Bytef *)buff;
        inflate(&strm, Z_NO_FLUSH);
        decompressed.append(buff, bufsiz - strm.avail_out);
    } while (strm.avail_out == 0);

    content.swap(decompressed);

    inflateEnd(&strm);
}
#endif

#ifdef _WIN32
class WSInit {
public:
    WSInit() {
        WSADATA wsaData;
        WSAStartup(0x0002, &wsaData);
    }

    ~WSInit() {
        WSACleanup();
    }
};

static WSInit wsinit_;
#endif

} // namespace detail

// Header utilities
template<typename uint64_t, typename... Args>
inline std::pair<std::string, std::string> make_range_header(uint64_t value, Args... args)
{
    std::string field;
    detail::make_range_header_core(field, value, args...);
    field.insert(0, "bytes=");
    return std::make_pair("Range", field);
}

// Request implementation
inline bool Request::has_header(const char* key) const
{
    return headers.find(key) != headers.end();
}

inline std::string Request::get_header_value(const char* key) const
{
    return detail::get_header_value(headers, key, "");
}

inline void Request::set_header(const char* key, const char* val)
{
    headers.emplace(key, val);
}

inline bool Request::has_param(const char* key) const
{
    return params.find(key) != params.end();
}

inline std::string Request::get_param_value(const char* key) const
{
    auto it = params.find(key);
    if (it != params.end()) {
        return it->second;
    }
    return std::string();
}

inline bool Request::has_file(const char* key) const
{
    return files.find(key) != files.end();
}

inline MultipartFile Request::get_file_value(const char* key) const
{
    auto it = files.find(key);
    if (it != files.end()) {
        return it->second;
    }
    return MultipartFile();
}

// Response implementation
inline bool Response::has_header(const char* key) const
{
    return headers.find(key) != headers.end();
}

inline std::string Response::get_header_value(const char* key) const
{
    return detail::get_header_value(headers, key, "");
}

inline void Response::set_header(const char* key, const char* val)
{
    headers.emplace(key, val);
}

inline void Response::set_redirect(const char* url)
{
    set_header("Location", url);
    status = 302;
}

inline void Response::set_content(const char* s, size_t n, const char* content_type)
{
    body.assign(s, n);
    set_header("Content-Type", content_type);
}

inline void Response::set_content(const std::string& s, const char* content_type)
{
    body = s;
    set_header("Content-Type", content_type);
}

// Rstream implementation
template <typename ...Args>
inline void Stream::write_format(const char* fmt, const Args& ...args)
{
    const auto bufsiz = 2048;
    char buf[bufsiz];

#if defined(_MSC_VER) && _MSC_VER < 1900
    auto n = _snprintf_s(buf, bufsiz, bufsiz - 1, fmt, args...);
#else
    auto n = snprintf(buf, bufsiz - 1, fmt, args...);
#endif
    if (n > 0) {
        if (n >= bufsiz - 1) {
            std::vector<char> glowable_buf(bufsiz);

            while (n >= static_cast<int>(glowable_buf.size() - 1)) {
                glowable_buf.resize(glowable_buf.size() * 2);
#if defined(_MSC_VER) && _MSC_VER < 1900
                n = _snprintf_s(&glowable_buf[0], glowable_buf.size(), glowable_buf.size() - 1, fmt, args...);
#else
                n = snprintf(&glowable_buf[0], glowable_buf.size() - 1, fmt, args...);
#endif
            }
            write(&glowable_buf[0], n);
        } else {
            write(buf, n);
        }
    }
}

// Socket stream implementation
inline SocketStream::SocketStream(socket_t sock): sock_(sock)
{
}

inline SocketStream::~SocketStream()
{
}

inline int SocketStream::read(char* ptr, size_t size)
{
    return recv(sock_, ptr, size, 0);
}

inline int SocketStream::write(const char* ptr, size_t size)
{
    return send(sock_, ptr, size, 0);
}

inline int SocketStream::write(const char* ptr)
{
    return write(ptr, strlen(ptr));
}

inline std::string SocketStream::get_remote_addr() {
    return detail::get_remote_addr(sock_);
}

// HTTP server implementation
inline Server::Server()
    : keep_alive_max_count_(5)
    , is_running_(false)
    , svr_sock_(INVALID_SOCKET)
    , running_threads_(0)
{
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif
}

inline Server::~Server()
{
}

inline Server& Server::Get(const char* pattern, Handler handler)
{
    get_handlers_.push_back(std::make_pair(std::regex(pattern), handler));
    return *this;
}

inline Server& Server::Post(const char* pattern, Handler handler)
{
    post_handlers_.push_back(std::make_pair(std::regex(pattern), handler));
    return *this;
}

inline Server& Server::Put(const char* pattern, Handler handler)
{
    put_handlers_.push_back(std::make_pair(std::regex(pattern), handler));
    return *this;
}

inline Server& Server::Delete(const char* pattern, Handler handler)
{
    delete_handlers_.push_back(std::make_pair(std::regex(pattern), handler));
    return *this;
}

inline Server& Server::Options(const char* pattern, Handler handler)
{
    options_handlers_.push_back(std::make_pair(std::regex(pattern), handler));
    return *this;
}

inline bool Server::set_base_dir(const char* path)
{
    if (detail::is_dir(path)) {
        base_dir_ = path;
        return true;
    }
    return false;
}

inline void Server::set_error_handler(Handler handler)
{
    error_handler_ = handler;
}

inline void Server::set_logger(Logger logger)
{
    logger_ = logger;
}

inline void Server::set_keep_alive_max_count(size_t count)
{
    keep_alive_max_count_ = count;
}

inline int Server::bind_to_any_port(const char* host, int socket_flags)
{
    return bind_internal(host, 0, socket_flags);
}

inline bool Server::listen_after_bind() {
    return listen_internal();
}

inline bool Server::listen(const char* host, int port, int socket_flags)
{
    if (bind_internal(host, port, socket_flags) < 0)
        return false;
    return listen_internal();
}

inline bool Server::is_running() const
{
    return is_running_;
}

inline void Server::stop()
{
    if (is_running_) {
        assert(svr_sock_ != INVALID_SOCKET);
        detail::shutdown_socket(svr_sock_);
        detail::close_socket(svr_sock_);
        svr_sock_ = INVALID_SOCKET;
    }
}

inline bool Server::parse_request_line(const char* s, Request& req)
{
    static std::regex re("(GET|HEAD|POST|PUT|DELETE|OPTIONS) (([^?]+)(?:\\?(.+?))?) (HTTP/1\\.[01])\r\n");

    std::cmatch m;
    if (std::regex_match(s, m, re)) {
        req.version = std::string(m[4]);
        req.method = std::string(m[1]);
        req.target = std::string(m[2]);
        req.path = detail::decode_url(m[3]);

        // Parse query text
        auto len = std::distance(m[4].first, m[4].second);
        if (len > 0) {
            detail::parse_query_text(m[4], req.params);
        }

        return true;
    }

    return false;
}

inline void Server::write_response(Stream& strm, bool last_connection, const Request& req, Response& res)
{
    assert(res.status != -1);

    if (400 <= res.status && error_handler_) {
        error_handler_(req, res);
    }

    // Response line
    strm.write_format("HTTP/1.1 %d %s\r\n",
        res.status,
        detail::status_message(res.status));

    // Headers
    if (last_connection ||
        req.version == "HTTP/1.0" ||
        req.get_header_value("Connection") == "close") {
        res.set_header("Connection", "close");
    }

    if (!res.body.empty()) {
#ifdef CPPHTTPLIB_ZLIB_SUPPORT
        // TODO: 'Accpet-Encoding' has gzip, not gzip;q=0
        const auto& encodings = req.get_header_value("Accept-Encoding");
        if (encodings.find("gzip") != std::string::npos &&
            detail::can_compress(res.get_header_value("Content-Type"))) {
            detail::compress(res.body);
            res.set_header("Content-Encoding", "gzip");
        }
#endif

        if (!res.has_header("Content-Type")) {
            res.set_header("Content-Type", "text/plain");
        }

        auto length = std::to_string(res.body.size());
        res.set_header("Content-Length", length.c_str());
    }

    detail::write_headers(strm, res);

    // Body
    if (!res.body.empty() && req.method != "HEAD") {
        strm.write(res.body.c_str(), res.body.size());
    }

    // Log
    if (logger_) {
        logger_(req, res);
    }
}

inline bool Server::handle_file_request(Request& req, Response& res)
{
    if (!base_dir_.empty() && detail::is_valid_path(req.path)) {
        std::string path = base_dir_ + req.path;

        if (!path.empty() && path.back() == '/') {
            path += "index.html";
        }

        if (detail::is_file(path)) {
            detail::read_file(path, res.body);
            auto type = detail::find_content_type(path);
            if (type) {
                res.set_header("Content-Type", type);
            }
            res.status = 200;
            return true;
        }
    }

    return false;
}

inline socket_t Server::create_server_socket(const char* host, int port, int socket_flags) const
{
    return detail::create_socket(host, port,
        [](socket_t sock, struct addrinfo& ai) -> bool {
            if (::bind(sock, ai.ai_addr, ai.ai_addrlen)) {
                  return false;
            }
            if (::listen(sock, 5)) { // Listen through 5 channels
                return false;
            }
            return true;
        }, socket_flags);
}

inline int Server::bind_internal(const char* host, int port, int socket_flags)
{
    if (!is_valid()) {
        return -1;
    }

    svr_sock_ = create_server_socket(host, port, socket_flags);
    if (svr_sock_ == INVALID_SOCKET) {
        return -1;
    }

    if (port == 0) {
        struct sockaddr_storage address;
        socklen_t len = sizeof(address);
        if (getsockname(svr_sock_, reinterpret_cast<struct sockaddr *>(&address), &len) == -1) {
            return -1;
        }
        if (address.ss_family == AF_INET) {
          return ntohs(reinterpret_cast<struct sockaddr_in*>(&address)->sin_port);
        } else if (address.ss_family == AF_INET6) {
          return ntohs(reinterpret_cast<struct sockaddr_in6*>(&address)->sin6_port);
        } else {
          return -1;
        }
    } else {
        return port;
    }
}

inline bool Server::listen_internal()
{
    auto ret = true;

    is_running_ = true;

    for (;;) {
        auto val = detail::select_read(svr_sock_, 0, 100000);

        if (val == 0) { // Timeout
            if (svr_sock_ == INVALID_SOCKET) {
                // The server socket was closed by 'stop' method.
                break;
            }
            continue;
        }

        socket_t sock = accept(svr_sock_, NULL, NULL);

        if (sock == INVALID_SOCKET) {
            if (svr_sock_ != INVALID_SOCKET) {
                detail::close_socket(svr_sock_);
                ret = false;
            } else {
                ; // The server socket was closed by user.
            }
            break;
        }

        // TODO: Use thread pool...
        std::thread([=]() {
            {
                std::lock_guard<std::mutex> guard(running_threads_mutex_);
                running_threads_++;
            }

            read_and_close_socket(sock);

            {
                std::lock_guard<std::mutex> guard(running_threads_mutex_);
                running_threads_--;
            }
        }).detach();
    }

    // TODO: Use thread pool...
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::lock_guard<std::mutex> guard(running_threads_mutex_);
        if (!running_threads_) {
            break;
        }
    }

    is_running_ = false;

    return ret;
}

inline bool Server::routing(Request& req, Response& res)
{
    if (req.method == "GET" && handle_file_request(req, res)) {
        return true;
    }

    if (req.method == "GET" || req.method == "HEAD") {
        return dispatch_request(req, res, get_handlers_);
    } else if (req.method == "POST") {
        return dispatch_request(req, res, post_handlers_);
    } else if (req.method == "PUT") {
        return dispatch_request(req, res, put_handlers_);
    } else if (req.method == "DELETE") {
        return dispatch_request(req, res, delete_handlers_);
    } else if (req.method == "OPTIONS") {
        return dispatch_request(req, res, options_handlers_);
    }
    return false;
}

inline bool Server::dispatch_request(Request& req, Response& res, Handlers& handlers)
{
    for (const auto& x: handlers) {
        const auto& pattern = x.first;
        const auto& handler = x.second;

        if (std::regex_match(req.path, req.matches, pattern)) {
            handler(req, res);
            return true;
        }
    }
    return false;
}

inline bool Server::process_request(Stream& strm, bool last_connection, bool& connection_close)
{
    const auto bufsiz = 2048;
    char buf[bufsiz];

    detail::stream_line_reader reader(strm, buf, bufsiz);

    // Connection has been closed on client
    if (!reader.getline()) {
        return false;
    }

    Request req;
    Response res;

    res.version = "HTTP/1.1";

    // Request line and headers
    if (!parse_request_line(reader.ptr(), req) || !detail::read_headers(strm, req.headers)) {
        res.status = 400;
        write_response(strm, last_connection, req, res);
        return true;
    }

    auto ret = true;
    if (req.get_header_value("Connection") == "close") {
        // ret = false;
        connection_close = true;
    }

    req.set_header("REMOTE_ADDR", strm.get_remote_addr().c_str());

    // Body
    if (req.method == "POST" || req.method == "PUT") {
        if (!detail::read_content(strm, req)) {
            res.status = 400;
            write_response(strm, last_connection, req, res);
            return ret;
        }

        const auto& content_type = req.get_header_value("Content-Type");

        if (req.get_header_value("Content-Encoding") == "gzip") {
#ifdef CPPHTTPLIB_ZLIB_SUPPORT
            detail::decompress(req.body);
#else
            res.status = 415;
            write_response(strm, last_connection, req, res);
            return ret;
#endif
        }

        if (!content_type.find("application/x-www-form-urlencoded")) {
            detail::parse_query_text(req.body, req.params);
        } else if(!content_type.find("multipart/form-data")) {
            std::string boundary;
            if (!detail::parse_multipart_boundary(content_type, boundary) ||
                !detail::parse_multipart_formdata(boundary, req.body, req.files)) {
                res.status = 400;
                write_response(strm, last_connection, req, res);
                return ret;
            }
        }
    }

    if (routing(req, res)) {
        if (res.status == -1) {
            res.status = 200;
        }
    } else {
        res.status = 404;
    }

    write_response(strm, last_connection, req, res);
    return ret;
}

inline bool Server::is_valid() const
{
    return true;
}

inline bool Server::read_and_close_socket(socket_t sock)
{
    return detail::read_and_close_socket(
        sock,
        keep_alive_max_count_,
        [this](Stream& strm, bool last_connection, bool& connection_close) {
            return process_request(strm, last_connection, connection_close);
        });
}

// HTTP client implementation
inline Client::Client(
    const char* host, int port, size_t timeout_sec)
    : host_(host)
    , port_(port)
    , timeout_sec_(timeout_sec)
    , host_and_port_(host_ + ":" + std::to_string(port_))
{
}

inline Client::~Client()
{
}

inline bool Client::is_valid() const
{
    return true;
}

inline socket_t Client::create_client_socket() const
{
    return detail::create_socket(host_.c_str(), port_,
        [=](socket_t sock, struct addrinfo& ai) -> bool {
            detail::set_nonblocking(sock, true);

            auto ret = connect(sock, ai.ai_addr, ai.ai_addrlen);
            if (ret < 0) {
                if (detail::is_connection_error() ||
                    !detail::wait_until_socket_is_ready(sock, timeout_sec_, 0)) {
                    detail::close_socket(sock);
                    return false;
                }
            }

            detail::set_nonblocking(sock, false);
            return true;
        });
}

inline bool Client::read_response_line(Stream& strm, Response& res)
{
    const auto bufsiz = 2048;
    char buf[bufsiz];

    detail::stream_line_reader reader(strm, buf, bufsiz);

    if (!reader.getline()) {
        return false;
    }

    const static std::regex re("(HTTP/1\\.[01]) (\\d+?) .+\r\n");

    std::cmatch m;
    if (std::regex_match(reader.ptr(), m, re)) {
        res.version = std::string(m[1]);
        res.status = std::stoi(std::string(m[2]));
    }

    return true;
}

inline bool Client::send(Request& req, Response& res)
{
    if (req.path.empty()) {
        return false;
    }

    auto sock = create_client_socket();
    if (sock == INVALID_SOCKET) {
        return false;
    }

    return read_and_close_socket(sock, req, res);
}

inline void Client::write_request(Stream& strm, Request& req)
{
    auto path = detail::encode_url(req.path);

    // Request line
    strm.write_format("%s %s HTTP/1.1\r\n",
        req.method.c_str(),
        path.c_str());

    // Headers
    req.set_header("Host", host_and_port_.c_str());

    if (!req.has_header("Accept")) {
        req.set_header("Accept", "*/*");
    }

    if (!req.has_header("User-Agent")) {
        req.set_header("User-Agent", "cpp-httplib/0.2");
    }

    // TODO: Support KeepAlive connection
    // if (!req.has_header("Connection")) {
        req.set_header("Connection", "close");
    // }

    if (!req.body.empty()) {
        if (!req.has_header("Content-Type")) {
            req.set_header("Content-Type", "text/plain");
        }

        auto length = std::to_string(req.body.size());
        req.set_header("Content-Length", length.c_str());
    }

    detail::write_headers(strm, req);

    // Body
    if (!req.body.empty()) {
        if (req.get_header_value("Content-Type") == "application/x-www-form-urlencoded") {
            auto str = detail::encode_url(req.body);
            strm.write(str.c_str(), str.size());
        } else {
            strm.write(req.body.c_str(), req.body.size());
        }
    }
}

inline bool Client::process_request(Stream& strm, Request& req, Response& res, bool& connection_close)
{
    // Send request
    write_request(strm, req);

    // Receive response and headers
    if (!read_response_line(strm, res) || !detail::read_headers(strm, res.headers)) {
        return false;
    }

    if (res.get_header_value("Connection") == "close" || res.version == "HTTP/1.0") {
        connection_close = true;
    }

    // Body
    if (req.method != "HEAD") {
        if (!detail::read_content(strm, res, req.progress)) {
            return false;
        }

        if (res.get_header_value("Content-Encoding") == "gzip") {
#ifdef CPPHTTPLIB_ZLIB_SUPPORT
            detail::decompress(res.body);
#else
            return false;
#endif
        }
    }

    return true;
}

inline bool Client::read_and_close_socket(socket_t sock, Request& req, Response& res)
{
    return detail::read_and_close_socket(
        sock,
        0,
        [&](Stream& strm, bool /*last_connection*/, bool& connection_close) {
            return process_request(strm, req, res, connection_close);
        });
}

inline std::shared_ptr<Response> Client::Get(const char* path, Progress progress)
{
    return Get(path, Headers(), progress);
}

inline std::shared_ptr<Response> Client::Get(const char* path, const Headers& headers, Progress progress)
{
    Request req;
    req.method = "GET";
    req.path = path;
    req.headers = headers;
    req.progress = progress;

    auto res = std::make_shared<Response>();

    return send(req, *res) ? res : nullptr;
}

inline std::shared_ptr<Response> Client::Head(const char* path)
{
    return Head(path, Headers());
}

inline std::shared_ptr<Response> Client::Head(const char* path, const Headers& headers)
{
    Request req;
    req.method = "HEAD";
    req.headers = headers;
    req.path = path;

    auto res = std::make_shared<Response>();

    return send(req, *res) ? res : nullptr;
}

inline std::shared_ptr<Response> Client::Post(
    const char* path, const std::string& body, const char* content_type)
{
    return Post(path, Headers(), body, content_type);
}

inline std::shared_ptr<Response> Client::Post(
    const char* path, const Headers& headers, const std::string& body, const char* content_type)
{
    Request req;
    req.method = "POST";
    req.headers = headers;
    req.path = path;

    req.headers.emplace("Content-Type", content_type);
    req.body = body;

    auto res = std::make_shared<Response>();

    return send(req, *res) ? res : nullptr;
}

inline std::shared_ptr<Response> Client::Post(const char* path, const Params& params)
{
    return Post(path, Headers(), params);
}

inline std::shared_ptr<Response> Client::Post(const char* path, const Headers& headers, const Params& params)
{
    std::string query;
    for (auto it = params.begin(); it != params.end(); ++it) {
        if (it != params.begin()) {
            query += "&";
        }
        query += it->first;
        query += "=";
        query += it->second;
    }

    return Post(path, headers, query, "application/x-www-form-urlencoded");
}

inline std::shared_ptr<Response> Client::Put(
    const char* path, const std::string& body, const char* content_type)
{
    return Put(path, Headers(), body, content_type);
}

inline std::shared_ptr<Response> Client::Put(
    const char* path, const Headers& headers, const std::string& body, const char* content_type)
{
    Request req;
    req.method = "PUT";
    req.headers = headers;
    req.path = path;

    req.headers.emplace("Content-Type", content_type);
    req.body = body;

    auto res = std::make_shared<Response>();

    return send(req, *res) ? res : nullptr;
}

inline std::shared_ptr<Response> Client::Delete(const char* path)
{
    return Delete(path, Headers());
}

inline std::shared_ptr<Response> Client::Delete(const char* path, const Headers& headers)
{
    Request req;
    req.method = "DELETE";
    req.path = path;
    req.headers = headers;

    auto res = std::make_shared<Response>();

    return send(req, *res) ? res : nullptr;
}

inline std::shared_ptr<Response> Client::Options(const char* path)
{
    return Options(path, Headers());
}

inline std::shared_ptr<Response> Client::Options(const char* path, const Headers& headers)
{
    Request req;
    req.method = "OPTIONS";
    req.path = path;
    req.headers = headers;

    auto res = std::make_shared<Response>();

    return send(req, *res) ? res : nullptr;
}

/*
 * SSL Implementation
 */
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
namespace detail {

template <typename U, typename V, typename T>
inline bool read_and_close_socket_ssl(
    socket_t sock, size_t keep_alive_max_count,
    // TODO: OpenSSL 1.0.2 occasionally crashes...
    // The upcoming 1.1.0 is going to be thread safe.
    SSL_CTX* ctx, std::mutex& ctx_mutex,
    U SSL_connect_or_accept, V setup,
    T callback)
{
    SSL* ssl = nullptr;
    {
        std::lock_guard<std::mutex> guard(ctx_mutex);

        ssl = SSL_new(ctx);
        if (!ssl) {
            return false;
        }
    }

    auto bio = BIO_new_socket(sock, BIO_NOCLOSE);
    SSL_set_bio(ssl, bio, bio);

    setup(ssl);

    SSL_connect_or_accept(ssl);

    bool ret = false;

    if (keep_alive_max_count > 0) {
        auto count = keep_alive_max_count;
        while (count > 0 &&
               detail::select_read(sock,
                   CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND,
                   CPPHTTPLIB_KEEPALIVE_TIMEOUT_USECOND) > 0) {
            SSLSocketStream strm(sock, ssl);
            auto last_connection = count == 1;
            auto connection_close = false;

            ret = callback(strm, last_connection, connection_close);
            if (!ret || connection_close) {
                break;
            }

            count--;
        }
    } else {
        SSLSocketStream strm(sock, ssl);
        auto dummy_connection_close = false;
        ret = callback(strm, true, dummy_connection_close);
    }

    SSL_shutdown(ssl);

    {
        std::lock_guard<std::mutex> guard(ctx_mutex);
        SSL_free(ssl);
    }

    close_socket(sock);

    return ret;
}

class SSLInit {
public:
    SSLInit() {
        SSL_load_error_strings();
        SSL_library_init();
    }
};

static SSLInit sslinit_;

} // namespace detail

// SSL socket stream implementation
inline SSLSocketStream::SSLSocketStream(socket_t sock, SSL* ssl)
    : sock_(sock), ssl_(ssl)
{
}

inline SSLSocketStream::~SSLSocketStream()
{
}

inline int SSLSocketStream::read(char* ptr, size_t size)
{
    return SSL_read(ssl_, ptr, size);
}

inline int SSLSocketStream::write(const char* ptr, size_t size)
{
    return SSL_write(ssl_, ptr, size);
}

inline int SSLSocketStream::write(const char* ptr)
{
    return write(ptr, strlen(ptr));
}

inline std::string SSLSocketStream::get_remote_addr() {
    return detail::get_remote_addr(sock_);
}

// SSL HTTP server implementation
inline SSLServer::SSLServer(const char* cert_path, const char* private_key_path)
{
    ctx_ = SSL_CTX_new(SSLv23_server_method());

    if (ctx_) {
        SSL_CTX_set_options(ctx_,
                            SSL_OP_ALL | SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 |
                            SSL_OP_NO_COMPRESSION |
                            SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION);

        // auto ecdh = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
        // SSL_CTX_set_tmp_ecdh(ctx_, ecdh);
        // EC_KEY_free(ecdh);

        if (SSL_CTX_use_certificate_file(ctx_, cert_path, SSL_FILETYPE_PEM) != 1 ||
            SSL_CTX_use_PrivateKey_file(ctx_, private_key_path, SSL_FILETYPE_PEM) != 1) {
            SSL_CTX_free(ctx_);
            ctx_ = nullptr;
        }
    }
}

inline SSLServer::~SSLServer()
{
    if (ctx_) {
        SSL_CTX_free(ctx_);
    }
}

inline bool SSLServer::is_valid() const
{
    return ctx_;
}

inline bool SSLServer::read_and_close_socket(socket_t sock)
{
    return detail::read_and_close_socket_ssl(
        sock,
        keep_alive_max_count_,
        ctx_, ctx_mutex_,
        SSL_accept,
        [](SSL* /*ssl*/) {},
        [this](Stream& strm, bool last_connection, bool& connection_close) {
            return process_request(strm, last_connection, connection_close);
        });
}

// SSL HTTP client implementation
inline SSLClient::SSLClient(const char* host, int port, size_t timeout_sec)
    : Client(host, port, timeout_sec)
{
    ctx_ = SSL_CTX_new(SSLv23_client_method());
}

inline SSLClient::~SSLClient()
{
    if (ctx_) {
        SSL_CTX_free(ctx_);
    }
}

inline bool SSLClient::is_valid() const
{
    return ctx_;
}

inline bool SSLClient::read_and_close_socket(socket_t sock, Request& req, Response& res)
{
    return is_valid() && detail::read_and_close_socket_ssl(
        sock, 0,
        ctx_, ctx_mutex_,
        SSL_connect,
        [&](SSL* ssl) {
            SSL_set_tlsext_host_name(ssl, host_.c_str());
        },
        [&](Stream& strm, bool /*last_connection*/, bool& connection_close) {
            return process_request(strm, req, res, connection_close);
        });
}
#endif

} // namespace httplib

#endif

// vim: et ts=4 sw=4 cin cino={1s ff=unix

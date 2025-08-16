#include <cstdint>

#include <httplib.h>

class FuzzedStream : public httplib::Stream {
public:
  FuzzedStream(const uint8_t *data, size_t size)
      : data_(data), size_(size), read_pos_(0) {}

  ssize_t read(char *ptr, size_t size) override {
    if (size + read_pos_ > size_) { size = size_ - read_pos_; }
    memcpy(ptr, data_ + read_pos_, size);
    read_pos_ += size;
    return static_cast<ssize_t>(size);
  }

  ssize_t write(const char *ptr, size_t size) override {
    response_.append(ptr, size);
    return static_cast<int>(size);
  }

  ssize_t write(const char *ptr) { return write(ptr, strlen(ptr)); }

  ssize_t write(const std::string &s) { return write(s.data(), s.size()); }

  bool is_readable() const override { return true; }

  bool wait_readable() const override { return true; }

  bool wait_writable() const override { return true; }

  void get_remote_ip_and_port(std::string &ip, int &port) const override {
    ip = "127.0.0.1";
    port = 8080;
  }

  void get_local_ip_and_port(std::string &ip, int &port) const override {
    ip = "127.0.0.1";
    port = 8080;
  }

  socket_t socket() const override { return 0; }

  time_t duration() const override { return 0; };

private:
  const uint8_t *data_;
  size_t size_;
  size_t read_pos_;
  std::string response_;
};

class FuzzableServer : public httplib::Server {
public:
  void ProcessFuzzedRequest(FuzzedStream &stream) {
    bool connection_close = false;
    process_request(stream,
                    /*remote_addr=*/"",
                    /*remote_port =*/0,
                    /*local_addr=*/"",
                    /*local_port =*/0,
                    /*last_connection=*/false, connection_close, nullptr);
  }
};

static FuzzableServer g_server;

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
  g_server.Get(R"(.*)",
               [&](const httplib::Request & /*req*/, httplib::Response &res) {
                 res.set_content("response content", "text/plain");
               });
  g_server.Post(R"(.*)",
                [&](const httplib::Request & /*req*/, httplib::Response &res) {
                  res.set_content("response content", "text/plain");
                });
  g_server.Put(R"(.*)",
               [&](const httplib::Request & /*req*/, httplib::Response &res) {
                 res.set_content("response content", "text/plain");
               });
  g_server.Patch(R"(.*)",
                 [&](const httplib::Request & /*req*/, httplib::Response &res) {
                   res.set_content("response content", "text/plain");
                 });
  g_server.Delete(
      R"(.*)", [&](const httplib::Request & /*req*/, httplib::Response &res) {
        res.set_content("response content", "text/plain");
      });
  g_server.Options(
      R"(.*)", [&](const httplib::Request & /*req*/, httplib::Response &res) {
        res.set_content("response content", "text/plain");
      });
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  FuzzedStream stream{data, size};
  g_server.ProcessFuzzedRequest(stream);
  return 0;
}

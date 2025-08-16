#include <httplib.h>
#include <iostream>

using namespace httplib;

const char *HOST = "localhost";
const int PORT = 1234;

void one_time_request_server(const char *label) {
  std::thread th;
  Server svr;

  svr.Get("/hi", [&](const Request & /*req*/, Response &res) {
    res.set_content(std::string("Hello from ") + label, "text/plain");

    // Stop server
    th = std::thread([&]() { svr.stop(); });
  });

  svr.listen(HOST, PORT);
  th.join();

  std::cout << label << " ended..." << std::endl;
}

void send_request(const char *label) {
  Client cli(HOST, PORT);

  std::cout << "Send " << label << " request" << std::endl;
  auto res = cli.Get("/hi");

  if (res) {
    std::cout << res->body << std::endl;
  } else {
    std::cout << "Request error: " + to_string(res.error()) << std::endl;
  }
}

int main(void) {
  auto th1 = std::thread([&]() { one_time_request_server("Server #1"); });
  auto th2 = std::thread([&]() { one_time_request_server("Server #2"); });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  send_request("1st");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  send_request("2nd");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  send_request("3rd");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  th1.join();
  th2.join();
}

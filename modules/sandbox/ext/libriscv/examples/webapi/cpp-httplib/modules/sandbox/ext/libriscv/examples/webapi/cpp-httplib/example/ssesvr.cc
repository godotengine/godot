#include <atomic>
#include <chrono>
#include <condition_variable>
#include <httplib.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

using namespace httplib;
using namespace std;

class EventDispatcher {
public:
  EventDispatcher() {}

  void wait_event(DataSink *sink) {
    unique_lock<mutex> lk(m_);
    int id = id_;
    cv_.wait(lk, [&] { return cid_ == id; });
    sink->write(message_.data(), message_.size());
  }

  void send_event(const string &message) {
    lock_guard<mutex> lk(m_);
    cid_ = id_++;
    message_ = message;
    cv_.notify_all();
  }

private:
  mutex m_;
  condition_variable cv_;
  atomic_int id_{0};
  atomic_int cid_{-1};
  string message_;
};

const auto html = R"(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SSE demo</title>
</head>
<body>
<script>
const ev1 = new EventSource("event1");
ev1.onmessage = function(e) {
  console.log('ev1', e.data);
}
const ev2 = new EventSource("event2");
ev2.onmessage = function(e) {
  console.log('ev2', e.data);
}
</script>
</body>
</html>
)";

int main(void) {
  EventDispatcher ed;

  Server svr;

  svr.Get("/", [&](const Request & /*req*/, Response &res) {
    res.set_content(html, "text/html");
  });

  svr.Get("/event1", [&](const Request & /*req*/, Response &res) {
    cout << "connected to event1..." << endl;
    res.set_chunked_content_provider("text/event-stream",
                                     [&](size_t /*offset*/, DataSink &sink) {
                                       ed.wait_event(&sink);
                                       return true;
                                     });
  });

  svr.Get("/event2", [&](const Request & /*req*/, Response &res) {
    cout << "connected to event2..." << endl;
    res.set_chunked_content_provider("text/event-stream",
                                     [&](size_t /*offset*/, DataSink &sink) {
                                       ed.wait_event(&sink);
                                       return true;
                                     });
  });

  thread t([&] {
    int id = 0;
    while (true) {
      this_thread::sleep_for(chrono::seconds(1));
      cout << "send event: " << id << std::endl;
      std::stringstream ss;
      ss << "data: " << id << "\n\n";
      ed.send_event(ss.str());
      id++;
    }
  });

  svr.listen("localhost", 1234);
}

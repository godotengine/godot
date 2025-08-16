#include "httplib.h"
using namespace httplib;

int main() {
  Server svr;

  svr.Get("/", [](const Request &, Response &res) {
    res.set_content("Hello World!", "text/plain");
  });

  svr.listen("0.0.0.0", 8080);
}

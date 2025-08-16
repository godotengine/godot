//
//  upload.cc
//
//  Copyright (c) 2019 Yuji Hirose. All rights reserved.
//  MIT License
//

#include <fstream>
#include <httplib.h>
#include <iostream>
using namespace httplib;
using namespace std;

const char *html = R"(
<form id="formElem">
  <input type="file" name="image_file" accept="image/*">
  <input type="file" name="text_file" accept="text/*">
  <input type="submit">
</form>
<script>
  formElem.onsubmit = async (e) => {
    e.preventDefault();
    let res = await fetch('/post', {
      method: 'POST',
      body: new FormData(formElem)
    });
    console.log(await res.text());
  };
</script>
)";

int main(void) {
  Server svr;

  svr.Get("/", [](const Request & /*req*/, Response &res) {
    res.set_content(html, "text/html");
  });

  svr.Post("/post", [](const Request &req, Response &res) {
    const auto &image_file = req.form.get_file("image_file");
    const auto &text_file = req.form.get_file("text_file");

    cout << "image file length: " << image_file.content.length() << endl
         << "image file name: " << image_file.filename << endl
         << "text file length: " << text_file.content.length() << endl
         << "text file name: " << text_file.filename << endl;

    {
      ofstream ofs(image_file.filename, ios::binary);
      ofs << image_file.content;
    }
    {
      ofstream ofs(text_file.filename);
      ofs << text_file.content;
    }

    res.set_content("done", "text/plain");
  });

  svr.listen("localhost", 1234);
}

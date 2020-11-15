<!--
Copyright 2017 The Crashpad Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# crashpad_http_upload(1)

## Name

crashpad_http_upload—Send an HTTP POST request

## Synopsis

**crashpad_http_uplaod** [_OPTION…_]

## Description

Performs an HTTP or HTTPS POST, building a `multipart/form-data` request from
key-value pairs and files in the manner of an HTML `<form>` with a POST action.
Provides the response.

Programs that use the Crashpad client library directly will not normally use
this tool. This tool is provided for debugging and testing as it isolates
Crashpad’s networking implementation normally used to upload crash reports to
a crash report collection server, making it available for more general use.

## Options

 * **-f**, **--file**=_KEY_=_PATH_

   Include _PATH_ in the request as a file upload, in the manner of an HTML
   `<input type="file">` element. _KEY_ is used as the field name.

 * **--no-upload-gzip**

   Do not use `gzip` compression. Normally, the entire request body is
   compressed into a `gzip` stream and transmitted with `Content-Encoding:
   gzip`. This option disables compression, and is intended for use with servers
   that don’t accept uploads compressed in this way.

 * **-o**, **--output**=_FILE_

   The response body will be written to _FILE_ instead of standard output.

 * **-s**, **--string**=_KEY_=_VALUE_

   Include _KEY_ and _VALUE_ in the request as an ordinary form field, in the
   manner of an HTML `<input type="text">` element. _KEY_ is used as the field
   name, and _VALUE_ is used as its value.

 * **-u**, **--url**=_URL_

   Send the request to _URL_. This option is required.

 * **--help**

   Display help and exit.

 * **--version**

   Output version information and exit.

## Examples

Uploads a file to an HTTP server running on `localhost`.

```
$ crashpad_http-upload --url http://localhost/upload_test \
      --string=when=now --file=what=1040.pdf
Thanks for the upload!
```

This example corresponds to the HTML form:

```
<form action="http://localhost/upload_test" method="post">
  <input type="text" name="when" value="now" />
  <input type="file" name="what" />
  <input type="submit" />
</form>
```

## Exit Status

 * **0**

   Success.

 * **1**

   Failure, with a message printed to the standard error stream. HTTP error
   statuses such as 404 (Not Found) are included in the definition of failure.

## See Also

[crashpad_handler(8)](../handler/crashpad_handler.md)

## Resources

Crashpad home page: https://crashpad.chromium.org/.

Report bugs at https://crashpad.chromium.org/bug/new.

## Copyright

Copyright 2017 [The Crashpad
Authors](https://chromium.googlesource.com/crashpad/crashpad/+/master/AUTHORS).

## License

Licensed under the Apache License, Version 2.0 (the “License”);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an “AS IS” BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

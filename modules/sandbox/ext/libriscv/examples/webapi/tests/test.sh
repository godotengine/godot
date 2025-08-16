#!/bin/bash
curl --data-binary "@test.txt" -X POST http://localhost:1234/compile?method=newlib -D - -o binary
curl --data-binary "@binary" -X POST http://localhost:1234/execute -D -

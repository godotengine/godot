import http.client
import sys
import time
import os
import logging
import pathlib

ninja_builddir = pathlib.Path(sys.argv[2])

logging.basicConfig(
    filename=ninja_builddir / "scons_daemon_request.log",
    filemode="a",
    format="%(asctime)s %(message)s",
    level=logging.DEBUG,
)

while True:
    try:
        logging.debug(f"Sending request: {sys.argv[3]}")
        conn = http.client.HTTPConnection("127.0.0.1", port=int(sys.argv[1]), timeout=60)
        conn.request("GET", "/?build=" + sys.argv[3])
        response = None
        while not response:
            try:
                response = conn.getresponse()
            except (http.client.RemoteDisconnected, http.client.ResponseNotReady):
                time.sleep(0.01)
        response.read()
        status = response.status
        logging.debug(f"Request Done: {sys.argv[3]}")
        exit(0)
    except ConnectionRefusedError:
        logging.debug(f"Server not ready: {sys.argv[3]}")
        time.sleep(1)
exit(1)

import http.server
import socketserver
import socket
from urllib.parse import urlparse, parse_qs
import time
from threading import Condition
import subprocess
from subprocess import PIPE, Popen
import sys
import os
import threading, queue
import pathlib
import logging
from timeit import default_timer as timer
import importlib
import traceback

port = int(sys.argv[1])
ninja_builddir = pathlib.Path(sys.argv[2])
daemon_keep_alive = int(sys.argv[3])
args = sys.argv[4:]

debug = False
for arg in args:
    if "--debug" in arg:
        debug = True
        break
if not debug:
    logging.basicConfig(
        filename=ninja_builddir / "scons_daemon.log",
        filemode="a",
        format="%(asctime)s %(message)s",
        level=logging.DEBUG,
    )

    def daemon_log(message):
        logging.debug(message)


else:

    def daemon_log(message):
        return


def custom_readlines(handle, line_separator="\n", chunk_size=1):
    buf = ""
    while not handle.closed:
        data = handle.read(chunk_size)
        if not data:
            break
        buf += data.decode("utf-8")
        if line_separator in buf:
            chunks = buf.split(line_separator)
            buf = chunks.pop()
            for chunk in chunks:
                yield chunk + line_separator
        if buf.endswith("scons>>>"):
            yield buf
            buf = ""


def enqueue_output(out, queue):
    for line in iter(custom_readlines(out)):
        queue.put(line)
    out.close()


input_q = queue.Queue()
output_q = queue.Queue()
finished_building = []
building_cv = Condition()

thread_error = False


def daemon_thread_func():
    global thread_error
    global finished_building
    try:
        import inspect
        import SCons

        scons_launcher = os.path.join(os.path.dirname(inspect.getfile(SCons)), "__main__.py")
        daemon_log(f"Starting daemon with args: {scons_launcher}")

        args_list = [sys.executable, str(scons_launcher)] + args + ["--interactive"]
        daemon_log(f"Starting daemon with args: {args_list}")
        daemon_log(f"cwd: {os.getcwd()}")

        p = Popen(args_list, stdout=PIPE, stdin=PIPE)

        t = threading.Thread(target=enqueue_output, args=(p.stdout, output_q))
        t.daemon = True  # thread dies with the program
        t.start()

        daemon_ready = False

        while p.poll() is None:

            while True:
                try:
                    line = output_q.get(block=False, timeout=0.01)
                except queue.Empty:
                    break
                else:
                    daemon_log("output: " + line.strip())
                    if line == "scons>>>":
                        daemon_ready = True
                        with building_cv:
                            building_cv.notify()

            while daemon_ready and not input_q.empty():

                commands = []
                while not input_q.empty():
                    try:
                        commands += [input_q.get(block=False, timeout=0.01)]
                    except queue.Empty:
                        break
                input_command = "build " + " ".join(commands) + "\n"
                daemon_log("input: " + input_command.strip())

                p.stdin.write(input_command.encode("utf-8"))
                p.stdin.flush()
                with building_cv:
                    finished_building += commands
                daemon_ready = False

            time.sleep(0.01)
    except:
        thread_error = True
        daemon_log(traceback.format_exc())
        raise


daemon_thread = threading.Thread(target=daemon_thread_func)
daemon_thread.daemon = True
daemon_thread.start()


logging.debug(f"Starting server on port {port}, keep alive: {daemon_keep_alive}")

keep_alive_timer = timer()
httpd = None


def server_thread_func():
    global httpd
    global thread_error

    class S(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            global thread_error
            global keep_alive_timer
            try:
                gets = parse_qs(urlparse(self.path).query)
                build = gets.get("build")
                if build:
                    keep_alive_timer = timer()

                    daemon_log(f"Got request: {build[0]}")
                    input_q.put(build[0])

                    def pred():
                        return build[0] in finished_building

                    with building_cv:
                        building_cv.wait_for(pred)

                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()

            except:
                thread_error = True
                daemon_log(traceback.format_exc())
                raise

            def log_message(self, format, *args):
                return

    httpd = socketserver.TCPServer(("127.0.0.1", port), S)
    httpd.serve_forever()


server_thread = threading.Thread(target=server_thread_func)
server_thread.daemon = True
server_thread.start()

while timer() - keep_alive_timer < daemon_keep_alive and not thread_error:
    time.sleep(1)

if thread_error:
    daemon_log(f"Shutting server on port {port} down because thread error.")
else:
    daemon_log(f"Shutting server on port {port} down because timed out: {daemon_keep_alive}")

if os.path.exists(ninja_builddir / "scons_daemon_dirty"):
    os.unlink(ninja_builddir / "scons_daemon_dirty")
httpd.shutdown()
server_thread.join()

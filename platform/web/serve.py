#!/usr/bin/env python3

import argparse
import contextlib
import os
import socket
import ssl
import subprocess
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


# See cpython GH-17851 and GH-17864.
class DualStackServer(HTTPServer):
    def server_bind(self):
        # Suppress exception when protocol is IPv4.
        with contextlib.suppress(Exception):
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        return super().server_bind()


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def shell_open(url):
    if sys.platform == "win32":
        os.startfile(url)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, url])


def generate_snakeoil():
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509
        .CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "godot_web")]))
        .issuer_name(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "godot_web")]))
        .not_valid_before(datetime.datetime.today() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.today() + datetime.timedelta(days=1))
        .serial_number(x509.random_serial_number())
        .public_key(key.public_key())
        .sign(private_key=key, algorithm=SHA256())
    )
    return (
        key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()),
        cert.public_bytes(Encoding.PEM),
    )


def ssl_wrap_socket(httpd, keyfile, certfile):
    if keyfile and certfile:
        print(f"Using certificate: '{certfile}', key: '{keyfile}'")
    else:
        print("Generating self-signed certificate")

        from tempfile import NamedTemporaryFile

        key, cert = generate_snakeoil()
        tmp_keyfile = NamedTemporaryFile(delete_on_close=False)
        tmp_keyfile.write(key)
        tmp_keyfile.close()
        tmp_certfile = NamedTemporaryFile(delete_on_close=False)
        tmp_certfile.write(cert)
        tmp_certfile.close()
        keyfile = tmp_keyfile.name
        certfile = tmp_certfile.name
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)


def serve(root, port, https, run_browser, key="", cert=""):
    os.chdir(root)

    address = ("", port)
    httpd = DualStackServer(address, CORSRequestHandler)

    if https:
        ssl_wrap_socket(httpd, key, cert)
        url = f"https://127.0.0.1:{port}"
    else:
        url = f"http://127.0.0.1:{port}"

    if run_browser:
        # Open the served page in the user's default browser.
        print(f"Opening the served URL in the default browser (use `--no-browser` or `-n` to disable this): {url}")
        shell_open(url)
    else:
        print(f"Serving at: {url}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping server.")
    finally:
        # Clean-up server
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", help="port to listen on", default=8060, type=int)
    parser.add_argument(
        "-r", "--root", help="path to serve as root (relative to `platform/web/`)", default="../../bin", type=Path
    )
    parser.add_argument("-s", "--https", help="serve using HTTPS", dest="https", action="store_true")
    parser.add_argument("--certfile", help="The server certificate for HTTPS", default="")
    parser.add_argument("--keyfile", help="The server key for HTTPS", default="")
    parser.set_defaults(https=False)
    parser.add_argument(
        "-n", "--no-browser", help="don't open default web browser automatically", dest="browser", action="store_false"
    )
    parser.set_defaults(browser=True)
    args = parser.parse_args()

    key = Path(args.keyfile).resolve()
    cert = Path(args.certfile).resolve()

    # Change to the directory where the script is located,
    # so that the script can be run from any location.
    os.chdir(Path(__file__).resolve().parent)

    serve(args.root, args.port, args.https, args.browser, key=key, cert=cert)

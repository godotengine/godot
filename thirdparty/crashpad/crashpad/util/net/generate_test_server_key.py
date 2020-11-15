#!/usr/bin/env python

# Copyright 2018 The Crashpad Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess

# GN requires a Python script for actions, so this just wraps the openssl
# command needed to generate a test private key and a certificate. These names
# must correspond to what TestPaths::BuildArtifact() constructs.
key = 'crashpad_util_test_key.pem'
cert = 'crashpad_util_test_cert.pem'
subprocess.check_call(
    ['openssl', 'req', '-x509', '-nodes', '-subj', '/CN=localhost',
     '-days', '365', '-newkey', 'rsa:2048', '-keyout', key, '-out', cert],
    stderr=open(os.devnull, 'w'))

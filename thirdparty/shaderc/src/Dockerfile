# Copyright 2016 The Shaderc Authors. All rights reserved.
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

FROM alpine

MAINTAINER Google Shaderc Team

RUN apk add --update \
    build-base \
    cmake \
    git \
    ninja \
    python \
    py-pip \
  && rm -rf /var/cache/apk/*

WORKDIR /root
RUN git clone https://github.com/google/shaderc

WORKDIR shaderc
RUN ./utils/git-sync-deps

WORKDIR build
RUN cmake -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    .. \
  && ninja install

WORKDIR /root
RUN rm -rf shaderc

RUN adduser -s /bin/sh -D shaderc
USER shaderc

VOLUME /code
WORKDIR /code

CMD ["/bin/sh"]

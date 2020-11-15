# Copyright 2015 The Crashpad Authors. All rights reserved.
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

if [[ "${BASH_SOURCE[0]}" = "${0}" ]]; then
  echo "${0}: this file must be sourced, not run directly" >& 2
  exit 1
fi

# Some extensions of command-line tools behave differently on different systems.
# $sed_ext should be a sed invocation that enables extended regular expressions.
# $date_time_t should be a date invocation that causes it to print the date and
# time corresponding to a time_t string that immediately follows it.
case "$(uname -s)" in
  Darwin)
    sed_ext="sed -E"
    date_time_t="date -r"
    ;;
  Linux)
    sed_ext="sed -r"
    date_time_t="date -d@"
    ;;
  *)
    echo "${0}: unknown operating system" >& 2
    exit 1
    ;;
esac

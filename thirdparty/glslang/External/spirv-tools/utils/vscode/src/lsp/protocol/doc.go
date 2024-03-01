// Copyright 2018 The Go Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package protocol contains the structs that map directly to the wire format
// of the "Language Server Protocol".
//
// It is a literal transcription, with unmodified comments, and only the changes
// required to make it go code.
// Names are uppercased to export them.
// All fields have JSON tags added to correct the names.
// Fields marked with a ? are also marked as "omitempty"
// Fields that are "|| null" are made pointers
// Fields that are string or number are left as string
// Fields that are type "number" are made float64
package protocol

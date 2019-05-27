// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#include "file.h"

#include <gmock/gmock.h>

namespace {

using glslc::GetFileExtension;
using glslc::IsStageFile;
using glslc::GetGlslOrHlslExtension;
using shaderc_util::string_piece;
using testing::Eq;

class FileExtensionTest : public testing::Test {
 protected:
  string_piece empty = "";
  string_piece dot = ".";
  string_piece no_ext = "shader";
  string_piece trailing_dot = "shader.";
  string_piece vert_ext = "shader.vert";
  string_piece frag_ext = "shader.frag";
  string_piece tesc_ext = "shader.tesc";
  string_piece tese_ext = "shader.tese";
  string_piece geom_ext = "shader.geom";
  string_piece comp_ext = "shader.comp";
  string_piece glsl_ext = "shader.glsl";
  string_piece hlsl_ext = "shader.hlsl";
  string_piece multi_dot = "shader.some..ext";
  string_piece both_hg_ext = "shader.hlsl.glsl";
  string_piece both_gh_ext = "shader.glsl.hlsl";
};

TEST_F(FileExtensionTest, GetFileExtension) {
  EXPECT_EQ("", GetFileExtension(empty));
  EXPECT_EQ("", GetFileExtension(dot));
  EXPECT_EQ("", GetFileExtension(no_ext));
  EXPECT_EQ("", GetFileExtension(trailing_dot));
  EXPECT_EQ("vert", GetFileExtension(vert_ext));
  EXPECT_EQ("frag", GetFileExtension(frag_ext));
  EXPECT_EQ("tesc", GetFileExtension(tesc_ext));
  EXPECT_EQ("tese", GetFileExtension(tese_ext));
  EXPECT_EQ("geom", GetFileExtension(geom_ext));
  EXPECT_EQ("comp", GetFileExtension(comp_ext));
  EXPECT_EQ("glsl", GetFileExtension(glsl_ext));
  EXPECT_EQ("ext", GetFileExtension(multi_dot));
  EXPECT_EQ("glsl", GetFileExtension(both_hg_ext));
  EXPECT_EQ("hlsl", GetFileExtension(both_gh_ext));
}

TEST_F(FileExtensionTest, GetGlslOrHlslExtension) {
  EXPECT_THAT(GetGlslOrHlslExtension(empty), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(dot), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(no_ext), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(trailing_dot), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(vert_ext), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(frag_ext), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(tesc_ext), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(tese_ext), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(geom_ext), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(comp_ext), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(glsl_ext), Eq("glsl"));
  EXPECT_THAT(GetGlslOrHlslExtension(hlsl_ext), Eq("hlsl"));
  EXPECT_THAT(GetGlslOrHlslExtension(multi_dot), Eq(""));
  EXPECT_THAT(GetGlslOrHlslExtension(both_hg_ext), Eq("glsl"));
  EXPECT_THAT(GetGlslOrHlslExtension(both_gh_ext), Eq("hlsl"));
}

TEST_F(FileExtensionTest, IsStageFile) {
  EXPECT_FALSE(IsStageFile(empty));
  EXPECT_FALSE(IsStageFile(dot));
  EXPECT_FALSE(IsStageFile(no_ext));
  EXPECT_FALSE(IsStageFile(trailing_dot));
  EXPECT_TRUE(IsStageFile(vert_ext));
  EXPECT_TRUE(IsStageFile(frag_ext));
  EXPECT_TRUE(IsStageFile(tesc_ext));
  EXPECT_TRUE(IsStageFile(tese_ext));
  EXPECT_TRUE(IsStageFile(geom_ext));
  EXPECT_TRUE(IsStageFile(comp_ext));
  EXPECT_FALSE(IsStageFile(glsl_ext));
  EXPECT_FALSE(IsStageFile(multi_dot));
}

}  // anonymous namespace

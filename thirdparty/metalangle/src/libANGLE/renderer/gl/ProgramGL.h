//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ProgramGL.h: Defines the class interface for ProgramGL.

#ifndef LIBANGLE_RENDERER_GL_PROGRAMGL_H_
#define LIBANGLE_RENDERER_GL_PROGRAMGL_H_

#include <string>
#include <vector>

#include "libANGLE/renderer/ProgramImpl.h"

namespace angle
{
struct FeaturesGL;
}  // namespace angle

namespace rx
{

class FunctionsGL;
class RendererGL;
class StateManagerGL;

class ProgramGL : public ProgramImpl
{
  public:
    ProgramGL(const gl::ProgramState &data,
              const FunctionsGL *functions,
              const angle::FeaturesGL &features,
              StateManagerGL *stateManager,
              bool enablePathRendering,
              const std::shared_ptr<RendererGL> &renderer);
    ~ProgramGL() override;

    std::unique_ptr<LinkEvent> load(const gl::Context *context,
                                    gl::BinaryInputStream *stream,
                                    gl::InfoLog &infoLog) override;
    void save(const gl::Context *context, gl::BinaryOutputStream *stream) override;
    void setBinaryRetrievableHint(bool retrievable) override;
    void setSeparable(bool separable) override;

    std::unique_ptr<LinkEvent> link(const gl::Context *contextImpl,
                                    const gl::ProgramLinkedResources &resources,
                                    gl::InfoLog &infoLog) override;
    GLboolean validate(const gl::Caps &caps, gl::InfoLog *infoLog) override;

    void setUniform1fv(GLint location, GLsizei count, const GLfloat *v) override;
    void setUniform2fv(GLint location, GLsizei count, const GLfloat *v) override;
    void setUniform3fv(GLint location, GLsizei count, const GLfloat *v) override;
    void setUniform4fv(GLint location, GLsizei count, const GLfloat *v) override;
    void setUniform1iv(GLint location, GLsizei count, const GLint *v) override;
    void setUniform2iv(GLint location, GLsizei count, const GLint *v) override;
    void setUniform3iv(GLint location, GLsizei count, const GLint *v) override;
    void setUniform4iv(GLint location, GLsizei count, const GLint *v) override;
    void setUniform1uiv(GLint location, GLsizei count, const GLuint *v) override;
    void setUniform2uiv(GLint location, GLsizei count, const GLuint *v) override;
    void setUniform3uiv(GLint location, GLsizei count, const GLuint *v) override;
    void setUniform4uiv(GLint location, GLsizei count, const GLuint *v) override;
    void setUniformMatrix2fv(GLint location,
                             GLsizei count,
                             GLboolean transpose,
                             const GLfloat *value) override;
    void setUniformMatrix3fv(GLint location,
                             GLsizei count,
                             GLboolean transpose,
                             const GLfloat *value) override;
    void setUniformMatrix4fv(GLint location,
                             GLsizei count,
                             GLboolean transpose,
                             const GLfloat *value) override;
    void setUniformMatrix2x3fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value) override;
    void setUniformMatrix3x2fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value) override;
    void setUniformMatrix2x4fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value) override;
    void setUniformMatrix4x2fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value) override;
    void setUniformMatrix3x4fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value) override;
    void setUniformMatrix4x3fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value) override;

    void getUniformfv(const gl::Context *context, GLint location, GLfloat *params) const override;
    void getUniformiv(const gl::Context *context, GLint location, GLint *params) const override;
    void getUniformuiv(const gl::Context *context, GLint location, GLuint *params) const override;

    void setPathFragmentInputGen(const std::string &inputName,
                                 GLenum genMode,
                                 GLint components,
                                 const GLfloat *coeffs) override;

    void markUnusedUniformLocations(std::vector<gl::VariableLocation> *uniformLocations,
                                    std::vector<gl::SamplerBinding> *samplerBindings,
                                    std::vector<gl::ImageBinding> *imageBindings) override;

    ANGLE_INLINE GLuint getProgramID() const { return mProgramID; }

    void enableSideBySideRenderingPath() const;
    void enableLayeredRenderingPath(int baseViewIndex) const;

    angle::Result syncState(const gl::Context *context,
                            const gl::Program::DirtyBits &dirtyBits) override;

  private:
    class LinkTask;
    class LinkEventNativeParallel;
    class LinkEventGL;

    void preLink();
    bool checkLinkStatus(gl::InfoLog &infoLog);
    void postLink();

    void reapplyUBOBindingsIfNeeded(const gl::Context *context);

    bool getUniformBlockSize(const std::string &blockName,
                             const std::string &blockMappedName,
                             size_t *sizeOut) const;
    bool getUniformBlockMemberInfo(const std::string &memberUniformName,
                                   const std::string &memberUniformMappedName,
                                   sh::BlockMemberInfo *memberInfoOut) const;
    bool getShaderStorageBlockMemberInfo(const std::string &memberName,
                                         const std::string &memberMappedName,
                                         sh::BlockMemberInfo *memberInfoOut) const;
    bool getShaderStorageBlockSize(const std::string &blockName,
                                   const std::string &blockMappedName,
                                   size_t *sizeOut) const;
    void getAtomicCounterBufferSizeMap(std::map<int, unsigned int> *sizeMapOut) const;

    void linkResources(const gl::ProgramLinkedResources &resources);
    void setUniformBlockBinding(GLuint uniformBlockIndex, GLuint uniformBlockBinding);

    // Helper function, makes it simpler to type.
    GLint uniLoc(GLint glLocation) const { return mUniformRealLocationMap[glLocation]; }

    const FunctionsGL *mFunctions;
    const angle::FeaturesGL &mFeatures;
    StateManagerGL *mStateManager;

    std::vector<GLint> mUniformRealLocationMap;
    std::vector<GLuint> mUniformBlockRealLocationMap;

    struct PathRenderingFragmentInput
    {
        std::string mappedName;
        GLint location;
    };
    std::vector<PathRenderingFragmentInput> mPathRenderingFragmentInputs;

    bool mEnablePathRendering;
    GLint mMultiviewBaseViewLayerIndexUniformLocation;

    GLuint mProgramID;

    std::shared_ptr<RendererGL> mRenderer;

    bool mLinkedInParallel;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_PROGRAMGL_H_

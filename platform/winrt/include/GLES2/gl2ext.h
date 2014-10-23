#ifndef __gl2ext_h_
#define __gl2ext_h_

/* $Revision: 20795 $ on $Date:: 2013-03-07 01:01:58 -0800 #$ */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This document is licensed under the SGI Free Software B License Version
 * 2.0. For details, see http://oss.sgi.com/projects/FreeB/ .
 */

#ifndef GL_APIENTRYP
#   define GL_APIENTRYP GL_APIENTRY*
#endif

/*------------------------------------------------------------------------*
 * OES extension tokens
 *------------------------------------------------------------------------*/

/* GL_OES_compressed_ETC1_RGB8_texture */
#ifndef GL_OES_compressed_ETC1_RGB8_texture
#define GL_ETC1_RGB8_OES                                        0x8D64
#endif

/* GL_OES_compressed_paletted_texture */
#ifndef GL_OES_compressed_paletted_texture
#define GL_PALETTE4_RGB8_OES                                    0x8B90
#define GL_PALETTE4_RGBA8_OES                                   0x8B91
#define GL_PALETTE4_R5_G6_B5_OES                                0x8B92
#define GL_PALETTE4_RGBA4_OES                                   0x8B93
#define GL_PALETTE4_RGB5_A1_OES                                 0x8B94
#define GL_PALETTE8_RGB8_OES                                    0x8B95
#define GL_PALETTE8_RGBA8_OES                                   0x8B96
#define GL_PALETTE8_R5_G6_B5_OES                                0x8B97
#define GL_PALETTE8_RGBA4_OES                                   0x8B98
#define GL_PALETTE8_RGB5_A1_OES                                 0x8B99
#endif

/* GL_OES_depth24 */
#ifndef GL_OES_depth24
#define GL_DEPTH_COMPONENT24_OES                                0x81A6
#endif

/* GL_OES_depth32 */
#ifndef GL_OES_depth32
#define GL_DEPTH_COMPONENT32_OES                                0x81A7
#endif

/* GL_OES_depth_texture */
/* No new tokens introduced by this extension. */

/* GL_OES_EGL_image */
#ifndef GL_OES_EGL_image
typedef void* GLeglImageOES;
#endif

/* GL_OES_EGL_image_external */
#ifndef GL_OES_EGL_image_external
/* GLeglImageOES defined in GL_OES_EGL_image already. */
#define GL_TEXTURE_EXTERNAL_OES                                 0x8D65
#define GL_SAMPLER_EXTERNAL_OES                                 0x8D66
#define GL_TEXTURE_BINDING_EXTERNAL_OES                         0x8D67
#define GL_REQUIRED_TEXTURE_IMAGE_UNITS_OES                     0x8D68
#endif

/* GL_OES_element_index_uint */
#ifndef GL_OES_element_index_uint
#define GL_UNSIGNED_INT                                         0x1405
#endif

/* GL_OES_get_program_binary */
#ifndef GL_OES_get_program_binary
#define GL_PROGRAM_BINARY_LENGTH_OES                            0x8741
#define GL_NUM_PROGRAM_BINARY_FORMATS_OES                       0x87FE
#define GL_PROGRAM_BINARY_FORMATS_OES                           0x87FF
#endif

/* GL_OES_mapbuffer */
#ifndef GL_OES_mapbuffer
#define GL_WRITE_ONLY_OES                                       0x88B9
#define GL_BUFFER_ACCESS_OES                                    0x88BB
#define GL_BUFFER_MAPPED_OES                                    0x88BC
#define GL_BUFFER_MAP_POINTER_OES                               0x88BD
#endif

/* GL_OES_packed_depth_stencil */
#ifndef GL_OES_packed_depth_stencil
#define GL_DEPTH_STENCIL_OES                                    0x84F9
#define GL_UNSIGNED_INT_24_8_OES                                0x84FA
#define GL_DEPTH24_STENCIL8_OES                                 0x88F0
#endif

/* GL_OES_required_internalformat */
#ifndef GL_OES_required_internalformat 
#define GL_ALPHA8_OES                                           0x803C
#define GL_DEPTH_COMPONENT16_OES                                0x81A5
/* reuse GL_DEPTH_COMPONENT24_OES */                            
/* reuse GL_DEPTH24_STENCIL8_OES */                             
/* reuse GL_DEPTH_COMPONENT32_OES */                            
#define GL_LUMINANCE4_ALPHA4_OES                                0x8043
#define GL_LUMINANCE8_ALPHA8_OES                                0x8045
#define GL_LUMINANCE8_OES                                       0x8040
#define GL_RGBA4_OES                                            0x8056
#define GL_RGB5_A1_OES                                          0x8057
#define GL_RGB565_OES                                           0x8D62
/* reuse GL_RGB8_OES */                              
/* reuse GL_RGBA8_OES */  
/* reuse GL_RGB10_EXT */
/* reuse GL_RGB10_A2_EXT */
#endif 

/* GL_OES_rgb8_rgba8 */
#ifndef GL_OES_rgb8_rgba8
#define GL_RGB8_OES                                             0x8051
#define GL_RGBA8_OES                                            0x8058
#endif

/* GL_OES_standard_derivatives */
#ifndef GL_OES_standard_derivatives
#define GL_FRAGMENT_SHADER_DERIVATIVE_HINT_OES                  0x8B8B
#endif

/* GL_OES_stencil1 */
#ifndef GL_OES_stencil1
#define GL_STENCIL_INDEX1_OES                                   0x8D46
#endif

/* GL_OES_stencil4 */
#ifndef GL_OES_stencil4
#define GL_STENCIL_INDEX4_OES                                   0x8D47
#endif

#ifndef GL_OES_surfaceless_context
#define GL_FRAMEBUFFER_UNDEFINED_OES                            0x8219
#endif

/* GL_OES_texture_3D */
#ifndef GL_OES_texture_3D
#define GL_TEXTURE_WRAP_R_OES                                   0x8072
#define GL_TEXTURE_3D_OES                                       0x806F
#define GL_TEXTURE_BINDING_3D_OES                               0x806A
#define GL_MAX_3D_TEXTURE_SIZE_OES                              0x8073
#define GL_SAMPLER_3D_OES                                       0x8B5F
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_3D_ZOFFSET_OES        0x8CD4
#endif

/* GL_OES_texture_float */
/* No new tokens introduced by this extension. */

/* GL_OES_texture_float_linear */
/* No new tokens introduced by this extension. */

/* GL_OES_texture_half_float */
#ifndef GL_OES_texture_half_float
#define GL_HALF_FLOAT_OES                                       0x8D61
#endif

/* GL_OES_texture_half_float_linear */
/* No new tokens introduced by this extension. */

/* GL_OES_texture_npot */
/* No new tokens introduced by this extension. */

/* GL_OES_vertex_array_object */
#ifndef GL_OES_vertex_array_object
#define GL_VERTEX_ARRAY_BINDING_OES                             0x85B5
#endif

/* GL_OES_vertex_half_float */
/* GL_HALF_FLOAT_OES defined in GL_OES_texture_half_float already. */

/* GL_OES_vertex_type_10_10_10_2 */
#ifndef GL_OES_vertex_type_10_10_10_2
#define GL_UNSIGNED_INT_10_10_10_2_OES                          0x8DF6
#define GL_INT_10_10_10_2_OES                                   0x8DF7
#endif

/*------------------------------------------------------------------------*
 * KHR extension tokens
 *------------------------------------------------------------------------*/

#ifndef GL_KHR_debug
typedef void (GL_APIENTRYP GLDEBUGPROC)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,GLvoid *userParam);
#define GL_DEBUG_OUTPUT_SYNCHRONOUS                             0x8242
#define GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH                     0x8243
#define GL_DEBUG_CALLBACK_FUNCTION                              0x8244
#define GL_DEBUG_CALLBACK_USER_PARAM                            0x8245
#define GL_DEBUG_SOURCE_API                                     0x8246
#define GL_DEBUG_SOURCE_WINDOW_SYSTEM                           0x8247
#define GL_DEBUG_SOURCE_SHADER_COMPILER                         0x8248
#define GL_DEBUG_SOURCE_THIRD_PARTY                             0x8249
#define GL_DEBUG_SOURCE_APPLICATION                             0x824A
#define GL_DEBUG_SOURCE_OTHER                                   0x824B
#define GL_DEBUG_TYPE_ERROR                                     0x824C
#define GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR                       0x824D
#define GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR                        0x824E
#define GL_DEBUG_TYPE_PORTABILITY                               0x824F
#define GL_DEBUG_TYPE_PERFORMANCE                               0x8250
#define GL_DEBUG_TYPE_OTHER                                     0x8251
#define GL_DEBUG_TYPE_MARKER                                    0x8268
#define GL_DEBUG_TYPE_PUSH_GROUP                                0x8269
#define GL_DEBUG_TYPE_POP_GROUP                                 0x826A
#define GL_DEBUG_SEVERITY_NOTIFICATION                          0x826B
#define GL_MAX_DEBUG_GROUP_STACK_DEPTH                          0x826C
#define GL_DEBUG_GROUP_STACK_DEPTH                              0x826D
#define GL_BUFFER                                               0x82E0
#define GL_SHADER                                               0x82E1
#define GL_PROGRAM                                              0x82E2
#define GL_QUERY                                                0x82E3
/* PROGRAM_PIPELINE only in GL */                               
#define GL_SAMPLER                                              0x82E6
/* DISPLAY_LIST only in GL */                                   
#define GL_MAX_LABEL_LENGTH                                     0x82E8
#define GL_MAX_DEBUG_MESSAGE_LENGTH                             0x9143
#define GL_MAX_DEBUG_LOGGED_MESSAGES                            0x9144
#define GL_DEBUG_LOGGED_MESSAGES                                0x9145
#define GL_DEBUG_SEVERITY_HIGH                                  0x9146
#define GL_DEBUG_SEVERITY_MEDIUM                                0x9147
#define GL_DEBUG_SEVERITY_LOW                                   0x9148
#define GL_DEBUG_OUTPUT                                         0x92E0
#define GL_CONTEXT_FLAG_DEBUG_BIT                               0x00000002
#define GL_STACK_OVERFLOW                                       0x0503
#define GL_STACK_UNDERFLOW                                      0x0504
#endif

#ifndef GL_KHR_texture_compression_astc_ldr
#define GL_COMPRESSED_RGBA_ASTC_4x4_KHR                         0x93B0
#define GL_COMPRESSED_RGBA_ASTC_5x4_KHR                         0x93B1
#define GL_COMPRESSED_RGBA_ASTC_5x5_KHR                         0x93B2
#define GL_COMPRESSED_RGBA_ASTC_6x5_KHR                         0x93B3
#define GL_COMPRESSED_RGBA_ASTC_6x6_KHR                         0x93B4
#define GL_COMPRESSED_RGBA_ASTC_8x5_KHR                         0x93B5
#define GL_COMPRESSED_RGBA_ASTC_8x6_KHR                         0x93B6
#define GL_COMPRESSED_RGBA_ASTC_8x8_KHR                         0x93B7
#define GL_COMPRESSED_RGBA_ASTC_10x5_KHR                        0x93B8
#define GL_COMPRESSED_RGBA_ASTC_10x6_KHR                        0x93B9
#define GL_COMPRESSED_RGBA_ASTC_10x8_KHR                        0x93BA
#define GL_COMPRESSED_RGBA_ASTC_10x10_KHR                       0x93BB
#define GL_COMPRESSED_RGBA_ASTC_12x10_KHR                       0x93BC
#define GL_COMPRESSED_RGBA_ASTC_12x12_KHR                       0x93BD
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR                 0x93D0
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR                 0x93D1
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR                 0x93D2
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR                 0x93D3
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR                 0x93D4
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR                 0x93D5
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR                 0x93D6
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR                 0x93D7
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR                0x93D8
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR                0x93D9
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR                0x93DA
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR               0x93DB
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR               0x93DC
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR               0x93DD
#endif

/*------------------------------------------------------------------------*
 * AMD extension tokens
 *------------------------------------------------------------------------*/

/* GL_AMD_compressed_3DC_texture */
#ifndef GL_AMD_compressed_3DC_texture
#define GL_3DC_X_AMD                                            0x87F9
#define GL_3DC_XY_AMD                                           0x87FA
#endif

/* GL_AMD_compressed_ATC_texture */
#ifndef GL_AMD_compressed_ATC_texture
#define GL_ATC_RGB_AMD                                          0x8C92
#define GL_ATC_RGBA_EXPLICIT_ALPHA_AMD                          0x8C93
#define GL_ATC_RGBA_INTERPOLATED_ALPHA_AMD                      0x87EE
#endif

/* GL_AMD_performance_monitor */
#ifndef GL_AMD_performance_monitor
#define GL_COUNTER_TYPE_AMD                                     0x8BC0
#define GL_COUNTER_RANGE_AMD                                    0x8BC1
#define GL_UNSIGNED_INT64_AMD                                   0x8BC2
#define GL_PERCENTAGE_AMD                                       0x8BC3
#define GL_PERFMON_RESULT_AVAILABLE_AMD                         0x8BC4
#define GL_PERFMON_RESULT_SIZE_AMD                              0x8BC5
#define GL_PERFMON_RESULT_AMD                                   0x8BC6
#endif

/* GL_AMD_program_binary_Z400 */
#ifndef GL_AMD_program_binary_Z400
#define GL_Z400_BINARY_AMD                                      0x8740
#endif

/*------------------------------------------------------------------------*
 * ANGLE extension tokens
 *------------------------------------------------------------------------*/

/* GL_ANGLE_depth_texture */
#ifndef GL_ANGLE_depth_texture
#define GL_DEPTH_COMPONENT                                      0x1902
#define GL_DEPTH_STENCIL_OES                                    0x84F9
#define GL_UNSIGNED_SHORT                                       0x1403
#define GL_UNSIGNED_INT                                         0x1405
#define GL_UNSIGNED_INT_24_8_OES                                0x84FA
#define GL_DEPTH_COMPONENT16                                    0x81A5
#define GL_DEPTH_COMPONENT32_OES                                0x81A7
#define GL_DEPTH24_STENCIL8_OES                                 0x88F0
#endif

/* GL_ANGLE_framebuffer_blit */
#ifndef GL_ANGLE_framebuffer_blit
#define GL_READ_FRAMEBUFFER_ANGLE                               0x8CA8
#define GL_DRAW_FRAMEBUFFER_ANGLE                               0x8CA9
#define GL_DRAW_FRAMEBUFFER_BINDING_ANGLE                       0x8CA6
#define GL_READ_FRAMEBUFFER_BINDING_ANGLE                       0x8CAA
#endif

/* GL_ANGLE_framebuffer_multisample */
#ifndef GL_ANGLE_framebuffer_multisample
#define GL_RENDERBUFFER_SAMPLES_ANGLE                           0x8CAB
#define GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_ANGLE             0x8D56
#define GL_MAX_SAMPLES_ANGLE                                    0x8D57
#endif

/* GL_ANGLE_instanced_arrays */
#ifndef GL_ANGLE_instanced_arrays 
#define GL_VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE                    0x88FE
#endif

/* GL_ANGLE_pack_reverse_row_order */
#ifndef GL_ANGLE_pack_reverse_row_order 
#define GL_PACK_REVERSE_ROW_ORDER_ANGLE                         0x93A4
#endif

/* GL_ANGLE_program_binary */
#ifndef GL_ANGLE_program_binary
#define GL_PROGRAM_BINARY_ANGLE                                 0x93A6
#endif

/* GL_ANGLE_texture_compression_dxt3 */
#ifndef GL_ANGLE_texture_compression_dxt3 
#define GL_COMPRESSED_RGBA_S3TC_DXT3_ANGLE                      0x83F2
#endif

/* GL_ANGLE_texture_compression_dxt5 */
#ifndef GL_ANGLE_texture_compression_dxt5 
#define GL_COMPRESSED_RGBA_S3TC_DXT5_ANGLE                      0x83F3
#endif

/* GL_ANGLE_texture_usage */
#ifndef GL_ANGLE_texture_usage 
#define GL_TEXTURE_USAGE_ANGLE                                  0x93A2
#define GL_FRAMEBUFFER_ATTACHMENT_ANGLE                         0x93A3
#endif

/* GL_ANGLE_translated_shader_source */
#ifndef GL_ANGLE_translated_shader_source 
#define GL_TRANSLATED_SHADER_SOURCE_LENGTH_ANGLE                0x93A0
#endif

/*------------------------------------------------------------------------*
 * APPLE extension tokens
 *------------------------------------------------------------------------*/

/* GL_APPLE_copy_texture_levels */
/* No new tokens introduced by this extension. */
    
/* GL_APPLE_framebuffer_multisample */
#ifndef GL_APPLE_framebuffer_multisample
#define GL_RENDERBUFFER_SAMPLES_APPLE                           0x8CAB
#define GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_APPLE             0x8D56
#define GL_MAX_SAMPLES_APPLE                                    0x8D57
#define GL_READ_FRAMEBUFFER_APPLE                               0x8CA8
#define GL_DRAW_FRAMEBUFFER_APPLE                               0x8CA9
#define GL_DRAW_FRAMEBUFFER_BINDING_APPLE                       0x8CA6
#define GL_READ_FRAMEBUFFER_BINDING_APPLE                       0x8CAA
#endif

/* GL_APPLE_rgb_422 */
#ifndef GL_APPLE_rgb_422
#define GL_RGB_422_APPLE                                        0x8A1F
#define GL_UNSIGNED_SHORT_8_8_APPLE                             0x85BA
#define GL_UNSIGNED_SHORT_8_8_REV_APPLE                         0x85BB
#endif

/* GL_APPLE_sync */
#ifndef GL_APPLE_sync

#ifndef __gl3_h_
/* These types are defined with reference to <inttypes.h>
 * in the Apple extension spec, but here we use the Khronos
 * portable types in khrplatform.h, and assume those types 
 * are always defined.
 * If any other extensions using these types are defined, 
 * the typedefs must move out of this block and be shared.
 */
typedef khronos_int64_t GLint64;
typedef khronos_uint64_t GLuint64;
typedef struct __GLsync *GLsync;
#endif

#define GL_SYNC_OBJECT_APPLE                                    0x8A53
#define GL_MAX_SERVER_WAIT_TIMEOUT_APPLE                        0x9111
#define GL_OBJECT_TYPE_APPLE                                    0x9112
#define GL_SYNC_CONDITION_APPLE                                 0x9113
#define GL_SYNC_STATUS_APPLE                                    0x9114
#define GL_SYNC_FLAGS_APPLE                                     0x9115
#define GL_SYNC_FENCE_APPLE                                     0x9116
#define GL_SYNC_GPU_COMMANDS_COMPLETE_APPLE                     0x9117
#define GL_UNSIGNALED_APPLE                                     0x9118
#define GL_SIGNALED_APPLE                                       0x9119
#define GL_ALREADY_SIGNALED_APPLE                               0x911A
#define GL_TIMEOUT_EXPIRED_APPLE                                0x911B
#define GL_CONDITION_SATISFIED_APPLE                            0x911C
#define GL_WAIT_FAILED_APPLE                                    0x911D
#define GL_SYNC_FLUSH_COMMANDS_BIT_APPLE                        0x00000001
#define GL_TIMEOUT_IGNORED_APPLE                                0xFFFFFFFFFFFFFFFFull
#endif

/* GL_APPLE_texture_format_BGRA8888 */
#ifndef GL_APPLE_texture_format_BGRA8888
#define GL_BGRA_EXT                                             0x80E1
#endif

/* GL_APPLE_texture_max_level */
#ifndef GL_APPLE_texture_max_level
#define GL_TEXTURE_MAX_LEVEL_APPLE                              0x813D
#endif

/*------------------------------------------------------------------------*
 * ARM extension tokens
 *------------------------------------------------------------------------*/

/* GL_ARM_mali_program_binary */
#ifndef GL_ARM_mali_program_binary
#define GL_MALI_PROGRAM_BINARY_ARM                              0x8F61
#endif

/* GL_ARM_mali_shader_binary */
#ifndef GL_ARM_mali_shader_binary
#define GL_MALI_SHADER_BINARY_ARM                               0x8F60
#endif

/* GL_ARM_rgba8 */
/* No new tokens introduced by this extension. */

/*------------------------------------------------------------------------*
 * EXT extension tokens
 *------------------------------------------------------------------------*/

/* GL_EXT_blend_minmax */
#ifndef GL_EXT_blend_minmax
#define GL_MIN_EXT                                              0x8007
#define GL_MAX_EXT                                              0x8008
#endif

/* GL_EXT_color_buffer_half_float */
#ifndef GL_EXT_color_buffer_half_float
#define GL_RGBA16F_EXT                                          0x881A
#define GL_RGB16F_EXT                                           0x881B
#define GL_RG16F_EXT                                            0x822F
#define GL_R16F_EXT                                             0x822D
#define GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT            0x8211
#define GL_UNSIGNED_NORMALIZED_EXT                              0x8C17
#endif

/* GL_EXT_debug_label */
#ifndef GL_EXT_debug_label
#define GL_PROGRAM_PIPELINE_OBJECT_EXT                          0x8A4F
#define GL_PROGRAM_OBJECT_EXT                                   0x8B40
#define GL_SHADER_OBJECT_EXT                                    0x8B48
#define GL_BUFFER_OBJECT_EXT                                    0x9151
#define GL_QUERY_OBJECT_EXT                                     0x9153
#define GL_VERTEX_ARRAY_OBJECT_EXT                              0x9154
#endif

/* GL_EXT_debug_marker */
/* No new tokens introduced by this extension. */

/* GL_EXT_discard_framebuffer */
#ifndef GL_EXT_discard_framebuffer
#define GL_COLOR_EXT                                            0x1800
#define GL_DEPTH_EXT                                            0x1801
#define GL_STENCIL_EXT                                          0x1802
#endif

/* GL_EXT_map_buffer_range */
#ifndef GL_EXT_map_buffer_range
#define GL_MAP_READ_BIT_EXT                                     0x0001
#define GL_MAP_WRITE_BIT_EXT                                    0x0002
#define GL_MAP_INVALIDATE_RANGE_BIT_EXT                         0x0004
#define GL_MAP_INVALIDATE_BUFFER_BIT_EXT                        0x0008
#define GL_MAP_FLUSH_EXPLICIT_BIT_EXT                           0x0010
#define GL_MAP_UNSYNCHRONIZED_BIT_EXT                           0x0020
#endif

/* GL_EXT_multisampled_render_to_texture */
#ifndef GL_EXT_multisampled_render_to_texture
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_SAMPLES_EXT           0x8D6C
/* reuse values from GL_EXT_framebuffer_multisample (desktop extension) */ 
#define GL_RENDERBUFFER_SAMPLES_EXT                             0x8CAB
#define GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_EXT               0x8D56
#define GL_MAX_SAMPLES_EXT                                      0x8D57
#endif

/* GL_EXT_multiview_draw_buffers */
#ifndef GL_EXT_multiview_draw_buffers
#define GL_COLOR_ATTACHMENT_EXT                                 0x90F0
#define GL_MULTIVIEW_EXT                                        0x90F1
#define GL_DRAW_BUFFER_EXT                                      0x0C01
#define GL_READ_BUFFER_EXT                                      0x0C02
#define GL_MAX_MULTIVIEW_BUFFERS_EXT                            0x90F2
#endif

/* GL_EXT_multi_draw_arrays */
/* No new tokens introduced by this extension. */

/* GL_EXT_occlusion_query_boolean */
#ifndef GL_EXT_occlusion_query_boolean
#define GL_ANY_SAMPLES_PASSED_EXT                               0x8C2F
#define GL_ANY_SAMPLES_PASSED_CONSERVATIVE_EXT                  0x8D6A
#define GL_CURRENT_QUERY_EXT                                    0x8865
#define GL_QUERY_RESULT_EXT                                     0x8866
#define GL_QUERY_RESULT_AVAILABLE_EXT                           0x8867
#endif

/* GL_EXT_read_format_bgra */
#ifndef GL_EXT_read_format_bgra
#define GL_BGRA_EXT                                             0x80E1
#define GL_UNSIGNED_SHORT_4_4_4_4_REV_EXT                       0x8365
#define GL_UNSIGNED_SHORT_1_5_5_5_REV_EXT                       0x8366
#endif

/* GL_EXT_robustness */
#ifndef GL_EXT_robustness
/* reuse GL_NO_ERROR */
#define GL_GUILTY_CONTEXT_RESET_EXT                             0x8253
#define GL_INNOCENT_CONTEXT_RESET_EXT                           0x8254
#define GL_UNKNOWN_CONTEXT_RESET_EXT                            0x8255
#define GL_CONTEXT_ROBUST_ACCESS_EXT                            0x90F3
#define GL_RESET_NOTIFICATION_STRATEGY_EXT                      0x8256
#define GL_LOSE_CONTEXT_ON_RESET_EXT                            0x8252
#define GL_NO_RESET_NOTIFICATION_EXT                            0x8261
#endif

/* GL_EXT_separate_shader_objects */
#ifndef GL_EXT_separate_shader_objects
#define GL_VERTEX_SHADER_BIT_EXT                                0x00000001
#define GL_FRAGMENT_SHADER_BIT_EXT                              0x00000002
#define GL_ALL_SHADER_BITS_EXT                                  0xFFFFFFFF
#define GL_PROGRAM_SEPARABLE_EXT                                0x8258
#define GL_ACTIVE_PROGRAM_EXT                                   0x8259
#define GL_PROGRAM_PIPELINE_BINDING_EXT                         0x825A
#endif

/* GL_EXT_shader_framebuffer_fetch */
#ifndef GL_EXT_shader_framebuffer_fetch
#define GL_FRAGMENT_SHADER_DISCARDS_SAMPLES_EXT                 0x8A52
#endif

/* GL_EXT_shader_texture_lod */
/* No new tokens introduced by this extension. */

/* GL_EXT_shadow_samplers */
#ifndef GL_EXT_shadow_samplers
#define GL_TEXTURE_COMPARE_MODE_EXT                             0x884C
#define GL_TEXTURE_COMPARE_FUNC_EXT                             0x884D
#define GL_COMPARE_REF_TO_TEXTURE_EXT                           0x884E
#define GL_SAMPLER_2D_SHADOW_EXT                                0x8B62
#endif

/* GL_EXT_sRGB */
#ifndef GL_EXT_sRGB
#define GL_SRGB_EXT                                             0x8C40
#define GL_SRGB_ALPHA_EXT                                       0x8C42
#define GL_SRGB8_ALPHA8_EXT                                     0x8C43
#define GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT            0x8210
#endif

/* GL_EXT_texture_compression_dxt1 */
#ifndef GL_EXT_texture_compression_dxt1
#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT                         0x83F0
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT                        0x83F1
#endif

/* GL_EXT_texture_filter_anisotropic */
#ifndef GL_EXT_texture_filter_anisotropic
#define GL_TEXTURE_MAX_ANISOTROPY_EXT                           0x84FE
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT                       0x84FF
#endif

/* GL_EXT_texture_format_BGRA8888 */
#ifndef GL_EXT_texture_format_BGRA8888
#define GL_BGRA_EXT                                             0x80E1
#endif

/* GL_EXT_texture_rg */
#ifndef GL_EXT_texture_rg
#define GL_RED_EXT                                              0x1903
#define GL_RG_EXT                                               0x8227
#define GL_R8_EXT                                               0x8229
#define GL_RG8_EXT                                              0x822B
#endif

/* GL_EXT_texture_storage */
#ifndef GL_EXT_texture_storage
#define GL_TEXTURE_IMMUTABLE_FORMAT_EXT                         0x912F
#define GL_ALPHA8_EXT                                           0x803C  
#define GL_LUMINANCE8_EXT                                       0x8040
#define GL_LUMINANCE8_ALPHA8_EXT                                0x8045
#define GL_RGBA32F_EXT                                          0x8814  
#define GL_RGB32F_EXT                                           0x8815
#define GL_ALPHA32F_EXT                                         0x8816
#define GL_LUMINANCE32F_EXT                                     0x8818
#define GL_LUMINANCE_ALPHA32F_EXT                               0x8819
/* reuse GL_RGBA16F_EXT */
/* reuse GL_RGB16F_EXT */
#define GL_ALPHA16F_EXT                                         0x881C
#define GL_LUMINANCE16F_EXT                                     0x881E
#define GL_LUMINANCE_ALPHA16F_EXT                               0x881F
#define GL_RGB10_A2_EXT                                         0x8059  
#define GL_RGB10_EXT                                            0x8052
#define GL_BGRA8_EXT                                            0x93A1
#define GL_R8_EXT                                               0x8229
#define GL_RG8_EXT                                              0x822B
#define GL_R32F_EXT                                             0x822E  
#define GL_RG32F_EXT                                            0x8230
#define GL_R16F_EXT                                             0x822D
#define GL_RG16F_EXT                                            0x822F
#endif

/* GL_EXT_texture_type_2_10_10_10_REV */
#ifndef GL_EXT_texture_type_2_10_10_10_REV
#define GL_UNSIGNED_INT_2_10_10_10_REV_EXT                      0x8368
#endif

/* GL_EXT_unpack_subimage */
#ifndef GL_EXT_unpack_subimage
#define GL_UNPACK_ROW_LENGTH_EXT                                0x0CF2
#define GL_UNPACK_SKIP_ROWS_EXT                                 0x0CF3
#define GL_UNPACK_SKIP_PIXELS_EXT                               0x0CF4
#endif

/*------------------------------------------------------------------------*
 * DMP extension tokens
 *------------------------------------------------------------------------*/

/* GL_DMP_shader_binary */
#ifndef GL_DMP_shader_binary
#define GL_SHADER_BINARY_DMP                                    0x9250
#endif

/*------------------------------------------------------------------------*
 * FJ extension tokens
 *------------------------------------------------------------------------*/

/* GL_FJ_shader_binary_GCCSO */
#ifndef GL_FJ_shader_binary_GCCSO
#define GL_GCCSO_SHADER_BINARY_F                                0x9260
#endif

/*------------------------------------------------------------------------*
 * IMG extension tokens
 *------------------------------------------------------------------------*/

/* GL_IMG_program_binary */
#ifndef GL_IMG_program_binary
#define GL_SGX_PROGRAM_BINARY_IMG                               0x9130
#endif

/* GL_IMG_read_format */
#ifndef GL_IMG_read_format
#define GL_BGRA_IMG                                             0x80E1
#define GL_UNSIGNED_SHORT_4_4_4_4_REV_IMG                       0x8365
#endif

/* GL_IMG_shader_binary */
#ifndef GL_IMG_shader_binary
#define GL_SGX_BINARY_IMG                                       0x8C0A
#endif

/* GL_IMG_texture_compression_pvrtc */
#ifndef GL_IMG_texture_compression_pvrtc
#define GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG                      0x8C00
#define GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG                      0x8C01
#define GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG                     0x8C02
#define GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG                     0x8C03
#endif

/* GL_IMG_texture_compression_pvrtc2 */
#ifndef GL_IMG_texture_compression_pvrtc2
#define GL_COMPRESSED_RGBA_PVRTC_2BPPV2_IMG                     0x9137
#define GL_COMPRESSED_RGBA_PVRTC_4BPPV2_IMG                     0x9138
#endif

/* GL_IMG_multisampled_render_to_texture */
#ifndef GL_IMG_multisampled_render_to_texture
#define GL_RENDERBUFFER_SAMPLES_IMG                             0x9133
#define GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_IMG               0x9134
#define GL_MAX_SAMPLES_IMG                                      0x9135
#define GL_TEXTURE_SAMPLES_IMG                                  0x9136
#endif

/*------------------------------------------------------------------------*
 * NV extension tokens
 *------------------------------------------------------------------------*/

/* GL_NV_coverage_sample */
#ifndef GL_NV_coverage_sample
#define GL_COVERAGE_COMPONENT_NV                                0x8ED0
#define GL_COVERAGE_COMPONENT4_NV                               0x8ED1
#define GL_COVERAGE_ATTACHMENT_NV                               0x8ED2
#define GL_COVERAGE_BUFFERS_NV                                  0x8ED3
#define GL_COVERAGE_SAMPLES_NV                                  0x8ED4
#define GL_COVERAGE_ALL_FRAGMENTS_NV                            0x8ED5
#define GL_COVERAGE_EDGE_FRAGMENTS_NV                           0x8ED6
#define GL_COVERAGE_AUTOMATIC_NV                                0x8ED7
#define GL_COVERAGE_BUFFER_BIT_NV                               0x8000
#endif

/* GL_NV_depth_nonlinear */
#ifndef GL_NV_depth_nonlinear
#define GL_DEPTH_COMPONENT16_NONLINEAR_NV                       0x8E2C
#endif

/* GL_NV_draw_buffers */
#ifndef GL_NV_draw_buffers
#define GL_MAX_DRAW_BUFFERS_NV                                  0x8824
#define GL_DRAW_BUFFER0_NV                                      0x8825
#define GL_DRAW_BUFFER1_NV                                      0x8826
#define GL_DRAW_BUFFER2_NV                                      0x8827
#define GL_DRAW_BUFFER3_NV                                      0x8828
#define GL_DRAW_BUFFER4_NV                                      0x8829
#define GL_DRAW_BUFFER5_NV                                      0x882A
#define GL_DRAW_BUFFER6_NV                                      0x882B
#define GL_DRAW_BUFFER7_NV                                      0x882C
#define GL_DRAW_BUFFER8_NV                                      0x882D
#define GL_DRAW_BUFFER9_NV                                      0x882E
#define GL_DRAW_BUFFER10_NV                                     0x882F
#define GL_DRAW_BUFFER11_NV                                     0x8830
#define GL_DRAW_BUFFER12_NV                                     0x8831
#define GL_DRAW_BUFFER13_NV                                     0x8832
#define GL_DRAW_BUFFER14_NV                                     0x8833
#define GL_DRAW_BUFFER15_NV                                     0x8834
#define GL_COLOR_ATTACHMENT0_NV                                 0x8CE0
#define GL_COLOR_ATTACHMENT1_NV                                 0x8CE1
#define GL_COLOR_ATTACHMENT2_NV                                 0x8CE2
#define GL_COLOR_ATTACHMENT3_NV                                 0x8CE3
#define GL_COLOR_ATTACHMENT4_NV                                 0x8CE4
#define GL_COLOR_ATTACHMENT5_NV                                 0x8CE5
#define GL_COLOR_ATTACHMENT6_NV                                 0x8CE6
#define GL_COLOR_ATTACHMENT7_NV                                 0x8CE7
#define GL_COLOR_ATTACHMENT8_NV                                 0x8CE8
#define GL_COLOR_ATTACHMENT9_NV                                 0x8CE9
#define GL_COLOR_ATTACHMENT10_NV                                0x8CEA
#define GL_COLOR_ATTACHMENT11_NV                                0x8CEB
#define GL_COLOR_ATTACHMENT12_NV                                0x8CEC
#define GL_COLOR_ATTACHMENT13_NV                                0x8CED
#define GL_COLOR_ATTACHMENT14_NV                                0x8CEE
#define GL_COLOR_ATTACHMENT15_NV                                0x8CEF
#endif

/* GL_EXT_draw_buffers */
#ifndef GL_EXT_draw_buffers
#define GL_MAX_DRAW_BUFFERS_EXT                                  0x8824
#define GL_DRAW_BUFFER0_EXT                                      0x8825
#define GL_DRAW_BUFFER1_EXT                                      0x8826
#define GL_DRAW_BUFFER2_EXT                                      0x8827
#define GL_DRAW_BUFFER3_EXT                                      0x8828
#define GL_DRAW_BUFFER4_EXT                                      0x8829
#define GL_DRAW_BUFFER5_EXT                                      0x882A
#define GL_DRAW_BUFFER6_EXT                                      0x882B
#define GL_DRAW_BUFFER7_EXT                                      0x882C
#define GL_DRAW_BUFFER8_EXT                                      0x882D
#define GL_DRAW_BUFFER9_EXT                                      0x882E
#define GL_DRAW_BUFFER10_EXT                                     0x882F
#define GL_DRAW_BUFFER11_EXT                                     0x8830
#define GL_DRAW_BUFFER12_EXT                                     0x8831
#define GL_DRAW_BUFFER13_EXT                                     0x8832
#define GL_DRAW_BUFFER14_EXT                                     0x8833
#define GL_DRAW_BUFFER15_EXT                                     0x8834
#define GL_COLOR_ATTACHMENT0_EXT                                 0x8CE0
#define GL_COLOR_ATTACHMENT1_EXT                                 0x8CE1
#define GL_COLOR_ATTACHMENT2_EXT                                 0x8CE2
#define GL_COLOR_ATTACHMENT3_EXT                                 0x8CE3
#define GL_COLOR_ATTACHMENT4_EXT                                 0x8CE4
#define GL_COLOR_ATTACHMENT5_EXT                                 0x8CE5
#define GL_COLOR_ATTACHMENT6_EXT                                 0x8CE6
#define GL_COLOR_ATTACHMENT7_EXT                                 0x8CE7
#define GL_COLOR_ATTACHMENT8_EXT                                 0x8CE8
#define GL_COLOR_ATTACHMENT9_EXT                                 0x8CE9
#define GL_COLOR_ATTACHMENT10_EXT                                0x8CEA
#define GL_COLOR_ATTACHMENT11_EXT                                0x8CEB
#define GL_COLOR_ATTACHMENT12_EXT                                0x8CEC
#define GL_COLOR_ATTACHMENT13_EXT                                0x8CED
#define GL_COLOR_ATTACHMENT14_EXT                                0x8CEE
#define GL_COLOR_ATTACHMENT15_EXT                                0x8CEF
#define GL_MAX_COLOR_ATTACHMENTS_EXT                             0x8CDF
#endif

/* GL_NV_draw_instanced */
/* No new tokens introduced by this extension. */

/* GL_NV_fbo_color_attachments */
#ifndef GL_NV_fbo_color_attachments
#define GL_MAX_COLOR_ATTACHMENTS_NV                             0x8CDF
/* GL_COLOR_ATTACHMENT{0-15}_NV defined in GL_NV_draw_buffers already. */
#endif

/* GL_NV_fence */
#ifndef GL_NV_fence
#define GL_ALL_COMPLETED_NV                                     0x84F2
#define GL_FENCE_STATUS_NV                                      0x84F3
#define GL_FENCE_CONDITION_NV                                   0x84F4
#endif

/* GL_NV_framebuffer_blit */
#ifndef GL_NV_framebuffer_blit
#define GL_READ_FRAMEBUFFER_NV                                  0x8CA8
#define GL_DRAW_FRAMEBUFFER_NV                                  0x8CA9
#define GL_DRAW_FRAMEBUFFER_BINDING_NV                          0x8CA6 
#define GL_READ_FRAMEBUFFER_BINDING_NV                          0x8CAA
#endif

/* GL_NV_framebuffer_multisample */
#ifndef GL_NV_framebuffer_multisample
#define GL_RENDERBUFFER_SAMPLES_NV                              0x8CAB
#define GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_NV                0x8D56
#define GL_MAX_SAMPLES_NV                                       0x8D57
#endif

/* GL_NV_generate_mipmap_sRGB */
/* No new tokens introduced by this extension. */

/* GL_NV_instanced_arrays */
#ifndef GL_NV_instanced_arrays
#define GL_VERTEX_ATTRIB_ARRAY_DIVISOR_NV                       0x88FE
#endif

/* GL_NV_read_buffer */
#ifndef GL_NV_read_buffer
#define GL_READ_BUFFER_NV                                       0x0C02
#endif

/* GL_NV_read_buffer_front */
/* No new tokens introduced by this extension. */

/* GL_NV_read_depth */
/* No new tokens introduced by this extension. */

/* GL_NV_read_depth_stencil */
/* No new tokens introduced by this extension. */

/* GL_NV_read_stencil */
/* No new tokens introduced by this extension. */

/* GL_NV_shadow_samplers_array */
#ifndef GL_NV_shadow_samplers_array
#define GL_SAMPLER_2D_ARRAY_SHADOW_NV                           0x8DC4
#endif                                             
                                                   
/* GL_NV_shadow_samplers_cube */
#ifndef GL_NV_shadow_samplers_cube                 
#define GL_SAMPLER_CUBE_SHADOW_NV                               0x8DC5
#endif

/* GL_NV_sRGB_formats */
#ifndef GL_NV_sRGB_formats
#define GL_SLUMINANCE_NV                                        0x8C46
#define GL_SLUMINANCE_ALPHA_NV                                  0x8C44
#define GL_SRGB8_NV                                             0x8C41
#define GL_SLUMINANCE8_NV                                       0x8C47
#define GL_SLUMINANCE8_ALPHA8_NV                                0x8C45
#define GL_COMPRESSED_SRGB_S3TC_DXT1_NV                         0x8C4C
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV                   0x8C4D
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV                   0x8C4E
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV                   0x8C4F
#define GL_ETC1_SRGB8_NV                                        0x88EE
#endif

/* GL_NV_texture_border_clamp */
#ifndef GL_NV_texture_border_clamp
#define GL_TEXTURE_BORDER_COLOR_NV                              0x1004
#define GL_CLAMP_TO_BORDER_NV                                   0x812D
#endif

/* GL_NV_texture_compression_s3tc_update */
/* No new tokens introduced by this extension. */

/* GL_NV_texture_npot_2D_mipmap */
/* No new tokens introduced by this extension. */

/*------------------------------------------------------------------------*
 * QCOM extension tokens
 *------------------------------------------------------------------------*/

/* GL_QCOM_alpha_test */
#ifndef GL_QCOM_alpha_test
#define GL_ALPHA_TEST_QCOM                                      0x0BC0
#define GL_ALPHA_TEST_FUNC_QCOM                                 0x0BC1
#define GL_ALPHA_TEST_REF_QCOM                                  0x0BC2
#endif

/* GL_QCOM_binning_control */
#ifndef GL_QCOM_binning_control
#define GL_BINNING_CONTROL_HINT_QCOM                            0x8FB0
#define GL_CPU_OPTIMIZED_QCOM                                   0x8FB1
#define GL_GPU_OPTIMIZED_QCOM                                   0x8FB2
#define GL_RENDER_DIRECT_TO_FRAMEBUFFER_QCOM                    0x8FB3
#endif

/* GL_QCOM_driver_control */
/* No new tokens introduced by this extension. */

/* GL_QCOM_extended_get */
#ifndef GL_QCOM_extended_get
#define GL_TEXTURE_WIDTH_QCOM                                   0x8BD2
#define GL_TEXTURE_HEIGHT_QCOM                                  0x8BD3
#define GL_TEXTURE_DEPTH_QCOM                                   0x8BD4
#define GL_TEXTURE_INTERNAL_FORMAT_QCOM                         0x8BD5
#define GL_TEXTURE_FORMAT_QCOM                                  0x8BD6
#define GL_TEXTURE_TYPE_QCOM                                    0x8BD7
#define GL_TEXTURE_IMAGE_VALID_QCOM                             0x8BD8
#define GL_TEXTURE_NUM_LEVELS_QCOM                              0x8BD9
#define GL_TEXTURE_TARGET_QCOM                                  0x8BDA
#define GL_TEXTURE_OBJECT_VALID_QCOM                            0x8BDB
#define GL_STATE_RESTORE                                        0x8BDC
#endif

/* GL_QCOM_extended_get2 */
/* No new tokens introduced by this extension. */

/* GL_QCOM_perfmon_global_mode */
#ifndef GL_QCOM_perfmon_global_mode
#define GL_PERFMON_GLOBAL_MODE_QCOM                             0x8FA0
#endif

/* GL_QCOM_writeonly_rendering */
#ifndef GL_QCOM_writeonly_rendering
#define GL_WRITEONLY_RENDERING_QCOM                             0x8823
#endif

/* GL_QCOM_tiled_rendering */
#ifndef GL_QCOM_tiled_rendering
#define GL_COLOR_BUFFER_BIT0_QCOM                               0x00000001
#define GL_COLOR_BUFFER_BIT1_QCOM                               0x00000002
#define GL_COLOR_BUFFER_BIT2_QCOM                               0x00000004
#define GL_COLOR_BUFFER_BIT3_QCOM                               0x00000008
#define GL_COLOR_BUFFER_BIT4_QCOM                               0x00000010
#define GL_COLOR_BUFFER_BIT5_QCOM                               0x00000020
#define GL_COLOR_BUFFER_BIT6_QCOM                               0x00000040
#define GL_COLOR_BUFFER_BIT7_QCOM                               0x00000080
#define GL_DEPTH_BUFFER_BIT0_QCOM                               0x00000100
#define GL_DEPTH_BUFFER_BIT1_QCOM                               0x00000200
#define GL_DEPTH_BUFFER_BIT2_QCOM                               0x00000400
#define GL_DEPTH_BUFFER_BIT3_QCOM                               0x00000800
#define GL_DEPTH_BUFFER_BIT4_QCOM                               0x00001000
#define GL_DEPTH_BUFFER_BIT5_QCOM                               0x00002000
#define GL_DEPTH_BUFFER_BIT6_QCOM                               0x00004000
#define GL_DEPTH_BUFFER_BIT7_QCOM                               0x00008000
#define GL_STENCIL_BUFFER_BIT0_QCOM                             0x00010000
#define GL_STENCIL_BUFFER_BIT1_QCOM                             0x00020000
#define GL_STENCIL_BUFFER_BIT2_QCOM                             0x00040000
#define GL_STENCIL_BUFFER_BIT3_QCOM                             0x00080000
#define GL_STENCIL_BUFFER_BIT4_QCOM                             0x00100000
#define GL_STENCIL_BUFFER_BIT5_QCOM                             0x00200000
#define GL_STENCIL_BUFFER_BIT6_QCOM                             0x00400000
#define GL_STENCIL_BUFFER_BIT7_QCOM                             0x00800000
#define GL_MULTISAMPLE_BUFFER_BIT0_QCOM                         0x01000000
#define GL_MULTISAMPLE_BUFFER_BIT1_QCOM                         0x02000000
#define GL_MULTISAMPLE_BUFFER_BIT2_QCOM                         0x04000000
#define GL_MULTISAMPLE_BUFFER_BIT3_QCOM                         0x08000000
#define GL_MULTISAMPLE_BUFFER_BIT4_QCOM                         0x10000000
#define GL_MULTISAMPLE_BUFFER_BIT5_QCOM                         0x20000000
#define GL_MULTISAMPLE_BUFFER_BIT6_QCOM                         0x40000000
#define GL_MULTISAMPLE_BUFFER_BIT7_QCOM                         0x80000000
#endif

/*------------------------------------------------------------------------*
 * VIV extension tokens
 *------------------------------------------------------------------------*/

/* GL_VIV_shader_binary */
#ifndef GL_VIV_shader_binary
#define GL_SHADER_BINARY_VIV                                    0x8FC4
#endif

/*------------------------------------------------------------------------*
 * End of extension tokens, start of corresponding extension functions
 *------------------------------------------------------------------------*/

/*------------------------------------------------------------------------*
 * OES extension functions
 *------------------------------------------------------------------------*/

/* GL_OES_compressed_ETC1_RGB8_texture */
#ifndef GL_OES_compressed_ETC1_RGB8_texture
#define GL_OES_compressed_ETC1_RGB8_texture 1
#endif

/* GL_OES_compressed_paletted_texture */
#ifndef GL_OES_compressed_paletted_texture
#define GL_OES_compressed_paletted_texture 1
#endif

/* GL_OES_depth24 */
#ifndef GL_OES_depth24
#define GL_OES_depth24 1
#endif

/* GL_OES_depth32 */
#ifndef GL_OES_depth32
#define GL_OES_depth32 1
#endif

/* GL_OES_depth_texture */
#ifndef GL_OES_depth_texture
#define GL_OES_depth_texture 1
#endif

/* GL_OES_EGL_image */
#ifndef GL_OES_EGL_image
#define GL_OES_EGL_image 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glEGLImageTargetTexture2DOES (GLenum target, GLeglImageOES image);
GL_APICALL void GL_APIENTRY glEGLImageTargetRenderbufferStorageOES (GLenum target, GLeglImageOES image);
#endif
typedef void (GL_APIENTRYP PFNGLEGLIMAGETARGETTEXTURE2DOESPROC) (GLenum target, GLeglImageOES image);
typedef void (GL_APIENTRYP PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC) (GLenum target, GLeglImageOES image);
#endif

/* GL_OES_EGL_image_external */
#ifndef GL_OES_EGL_image_external
#define GL_OES_EGL_image_external 1
/* glEGLImageTargetTexture2DOES defined in GL_OES_EGL_image already. */
#endif

/* GL_OES_element_index_uint */
#ifndef GL_OES_element_index_uint
#define GL_OES_element_index_uint 1
#endif

/* GL_OES_fbo_render_mipmap */
#ifndef GL_OES_fbo_render_mipmap
#define GL_OES_fbo_render_mipmap 1
#endif

/* GL_OES_fragment_precision_high */
#ifndef GL_OES_fragment_precision_high
#define GL_OES_fragment_precision_high 1
#endif

/* GL_OES_get_program_binary */
#ifndef GL_OES_get_program_binary
#define GL_OES_get_program_binary 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glGetProgramBinaryOES (GLuint program, GLsizei bufSize, GLsizei *length, GLenum *binaryFormat, GLvoid *binary);
GL_APICALL void GL_APIENTRY glProgramBinaryOES (GLuint program, GLenum binaryFormat, const GLvoid *binary, GLint length);
#endif
typedef void (GL_APIENTRYP PFNGLGETPROGRAMBINARYOESPROC) (GLuint program, GLsizei bufSize, GLsizei *length, GLenum *binaryFormat, GLvoid *binary);
typedef void (GL_APIENTRYP PFNGLPROGRAMBINARYOESPROC) (GLuint program, GLenum binaryFormat, const GLvoid *binary, GLint length);
#endif

/* GL_OES_mapbuffer */
#ifndef GL_OES_mapbuffer
#define GL_OES_mapbuffer 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void* GL_APIENTRY glMapBufferOES (GLenum target, GLenum access);
GL_APICALL GLboolean GL_APIENTRY glUnmapBufferOES (GLenum target);
GL_APICALL void GL_APIENTRY glGetBufferPointervOES (GLenum target, GLenum pname, GLvoid** params);
#endif
typedef void* (GL_APIENTRYP PFNGLMAPBUFFEROESPROC) (GLenum target, GLenum access);
typedef GLboolean (GL_APIENTRYP PFNGLUNMAPBUFFEROESPROC) (GLenum target);
typedef void (GL_APIENTRYP PFNGLGETBUFFERPOINTERVOESPROC) (GLenum target, GLenum pname, GLvoid** params);
#endif

/* GL_OES_packed_depth_stencil */
#ifndef GL_OES_packed_depth_stencil
#define GL_OES_packed_depth_stencil 1
#endif

/* GL_OES_required_internalformat */
#ifndef GL_OES_required_internalformat
#define GL_OES_required_internalformat 1
#endif

/* GL_OES_rgb8_rgba8 */
#ifndef GL_OES_rgb8_rgba8
#define GL_OES_rgb8_rgba8 1
#endif

/* GL_OES_standard_derivatives */
#ifndef GL_OES_standard_derivatives
#define GL_OES_standard_derivatives 1
#endif

/* GL_OES_stencil1 */
#ifndef GL_OES_stencil1
#define GL_OES_stencil1 1
#endif

/* GL_OES_stencil4 */
#ifndef GL_OES_stencil4
#define GL_OES_stencil4 1
#endif

#ifndef GL_OES_surfaceless_context
#define GL_OES_surfaceless_context 1
#endif

/* GL_OES_texture_3D */
#ifndef GL_OES_texture_3D
#define GL_OES_texture_3D 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glTexImage3DOES (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid* pixels);
GL_APICALL void GL_APIENTRY glTexSubImage3DOES (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid* pixels);
GL_APICALL void GL_APIENTRY glCopyTexSubImage3DOES (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
GL_APICALL void GL_APIENTRY glCompressedTexImage3DOES (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid* data);
GL_APICALL void GL_APIENTRY glCompressedTexSubImage3DOES (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid* data);
GL_APICALL void GL_APIENTRY glFramebufferTexture3DOES (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset);
#endif
typedef void (GL_APIENTRYP PFNGLTEXIMAGE3DOESPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid* pixels);
typedef void (GL_APIENTRYP PFNGLTEXSUBIMAGE3DOESPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid* pixels);
typedef void (GL_APIENTRYP PFNGLCOPYTEXSUBIMAGE3DOESPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
typedef void (GL_APIENTRYP PFNGLCOMPRESSEDTEXIMAGE3DOESPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid* data);
typedef void (GL_APIENTRYP PFNGLCOMPRESSEDTEXSUBIMAGE3DOESPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid* data);
typedef void (GL_APIENTRYP PFNGLFRAMEBUFFERTEXTURE3DOES) (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset);
#endif

/* GL_OES_texture_float */
#ifndef GL_OES_texture_float
#define GL_OES_texture_float 1
#endif

/* GL_OES_texture_float_linear */
#ifndef GL_OES_texture_float_linear
#define GL_OES_texture_float_linear 1
#endif

/* GL_OES_texture_half_float */
#ifndef GL_OES_texture_half_float
#define GL_OES_texture_half_float 1
#endif

/* GL_OES_texture_half_float_linear */
#ifndef GL_OES_texture_half_float_linear
#define GL_OES_texture_half_float_linear 1
#endif

/* GL_OES_texture_npot */
#ifndef GL_OES_texture_npot
#define GL_OES_texture_npot 1
#endif

/* GL_OES_vertex_array_object */
#ifndef GL_OES_vertex_array_object
#define GL_OES_vertex_array_object 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glBindVertexArrayOES (GLuint array);
GL_APICALL void GL_APIENTRY glDeleteVertexArraysOES (GLsizei n, const GLuint *arrays);
GL_APICALL void GL_APIENTRY glGenVertexArraysOES (GLsizei n, GLuint *arrays);
GL_APICALL GLboolean GL_APIENTRY glIsVertexArrayOES (GLuint array);
#endif
typedef void (GL_APIENTRYP PFNGLBINDVERTEXARRAYOESPROC) (GLuint array);
typedef void (GL_APIENTRYP PFNGLDELETEVERTEXARRAYSOESPROC) (GLsizei n, const GLuint *arrays);
typedef void (GL_APIENTRYP PFNGLGENVERTEXARRAYSOESPROC) (GLsizei n, GLuint *arrays);
typedef GLboolean (GL_APIENTRYP PFNGLISVERTEXARRAYOESPROC) (GLuint array);
#endif

/* GL_OES_vertex_half_float */
#ifndef GL_OES_vertex_half_float
#define GL_OES_vertex_half_float 1
#endif

/* GL_OES_vertex_type_10_10_10_2 */
#ifndef GL_OES_vertex_type_10_10_10_2
#define GL_OES_vertex_type_10_10_10_2 1
#endif

/*------------------------------------------------------------------------*
 * KHR extension functions
 *------------------------------------------------------------------------*/

#ifndef GL_KHR_debug
#define GL_KHR_debug 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glDebugMessageControl (GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint *ids, GLboolean enabled);
GL_APICALL void GL_APIENTRY glDebugMessageInsert (GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *buf);
GL_APICALL void GL_APIENTRY glDebugMessageCallback (GLDEBUGPROC callback, const void *userParam);
GL_APICALL GLuint GL_APIENTRY glGetDebugMessageLog (GLuint count, GLsizei bufsize, GLenum *sources, GLenum *types, GLuint *ids, GLenum *severities, GLsizei *lengths, GLchar *messageLog);
GL_APICALL void GL_APIENTRY glPushDebugGroup (GLenum source, GLuint id, GLsizei length, const GLchar *message);
GL_APICALL void GL_APIENTRY glPopDebugGroup (void);
GL_APICALL void GL_APIENTRY glObjectLabel (GLenum identifier, GLuint name, GLsizei length, const GLchar *label);
GL_APICALL void GL_APIENTRY glGetObjectLabel (GLenum identifier, GLuint name, GLsizei bufSize, GLsizei *length, GLchar *label);
GL_APICALL void GL_APIENTRY glObjectPtrLabel (const void *ptr, GLsizei length, const GLchar *label);
GL_APICALL void GL_APIENTRY glGetObjectPtrLabel (const void *ptr, GLsizei bufSize, GLsizei *length, GLchar *label);
GL_APICALL void GL_APIENTRY glGetPointerv (GLenum pname, void **params);
#endif 
typedef void (GL_APIENTRYP PFNGLDEBUGMESSAGECONTROLPROC) (GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint *ids, GLboolean enabled);
typedef void (GL_APIENTRYP PFNGLDEBUGMESSAGEINSERTPROC) (GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *buf);
typedef void (GL_APIENTRYP PFNGLDEBUGMESSAGECALLBACKPROC) (GLDEBUGPROC callback, const void *userParam);
typedef GLuint (GL_APIENTRYP PFNGLGETDEBUGMESSAGELOGPROC) (GLuint count, GLsizei bufsize, GLenum *sources, GLenum *types, GLuint *ids, GLenum *severities, GLsizei *lengths, GLchar *messageLog);
typedef void (GL_APIENTRYP PFNGLPUSHDEBUGGROUPPROC) (GLenum source, GLuint id, GLsizei length, const GLchar *message);
typedef void (GL_APIENTRYP PFNGLPOPDEBUGGROUPPROC) (void);
typedef void (GL_APIENTRYP PFNGLOBJECTLABELPROC) (GLenum identifier, GLuint name, GLsizei length, const GLchar *label);
typedef void (GL_APIENTRYP PFNGLGETOBJECTLABELPROC) (GLenum identifier, GLuint name, GLsizei bufSize, GLsizei *length, GLchar *label);
typedef void (GL_APIENTRYP PFNGLOBJECTPTRLABELPROC) (const void *ptr, GLsizei length, const GLchar *label);
typedef void (GL_APIENTRYP PFNGLGETOBJECTPTRLABELPROC) (const void *ptr, GLsizei bufSize, GLsizei *length, GLchar *label);
typedef void (GL_APIENTRYP PFNGLGETPOINTERVPROC) (GLenum pname, void **params);
#endif

#ifndef GL_KHR_texture_compression_astc_ldr
#define GL_KHR_texture_compression_astc_ldr 1
#endif


/*------------------------------------------------------------------------*
 * AMD extension functions
 *------------------------------------------------------------------------*/

/* GL_AMD_compressed_3DC_texture */
#ifndef GL_AMD_compressed_3DC_texture
#define GL_AMD_compressed_3DC_texture 1
#endif

/* GL_AMD_compressed_ATC_texture */
#ifndef GL_AMD_compressed_ATC_texture
#define GL_AMD_compressed_ATC_texture 1
#endif

/* AMD_performance_monitor */
#ifndef GL_AMD_performance_monitor
#define GL_AMD_performance_monitor 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glGetPerfMonitorGroupsAMD (GLint *numGroups, GLsizei groupsSize, GLuint *groups);
GL_APICALL void GL_APIENTRY glGetPerfMonitorCountersAMD (GLuint group, GLint *numCounters, GLint *maxActiveCounters, GLsizei counterSize, GLuint *counters);
GL_APICALL void GL_APIENTRY glGetPerfMonitorGroupStringAMD (GLuint group, GLsizei bufSize, GLsizei *length, GLchar *groupString);
GL_APICALL void GL_APIENTRY glGetPerfMonitorCounterStringAMD (GLuint group, GLuint counter, GLsizei bufSize, GLsizei *length, GLchar *counterString);
GL_APICALL void GL_APIENTRY glGetPerfMonitorCounterInfoAMD (GLuint group, GLuint counter, GLenum pname, GLvoid *data);
GL_APICALL void GL_APIENTRY glGenPerfMonitorsAMD (GLsizei n, GLuint *monitors);
GL_APICALL void GL_APIENTRY glDeletePerfMonitorsAMD (GLsizei n, GLuint *monitors);
GL_APICALL void GL_APIENTRY glSelectPerfMonitorCountersAMD (GLuint monitor, GLboolean enable, GLuint group, GLint numCounters, GLuint *countersList);
GL_APICALL void GL_APIENTRY glBeginPerfMonitorAMD (GLuint monitor);
GL_APICALL void GL_APIENTRY glEndPerfMonitorAMD (GLuint monitor);
GL_APICALL void GL_APIENTRY glGetPerfMonitorCounterDataAMD (GLuint monitor, GLenum pname, GLsizei dataSize, GLuint *data, GLint *bytesWritten);
#endif
typedef void (GL_APIENTRYP PFNGLGETPERFMONITORGROUPSAMDPROC) (GLint *numGroups, GLsizei groupsSize, GLuint *groups);
typedef void (GL_APIENTRYP PFNGLGETPERFMONITORCOUNTERSAMDPROC) (GLuint group, GLint *numCounters, GLint *maxActiveCounters, GLsizei counterSize, GLuint *counters);
typedef void (GL_APIENTRYP PFNGLGETPERFMONITORGROUPSTRINGAMDPROC) (GLuint group, GLsizei bufSize, GLsizei *length, GLchar *groupString);
typedef void (GL_APIENTRYP PFNGLGETPERFMONITORCOUNTERSTRINGAMDPROC) (GLuint group, GLuint counter, GLsizei bufSize, GLsizei *length, GLchar *counterString);
typedef void (GL_APIENTRYP PFNGLGETPERFMONITORCOUNTERINFOAMDPROC) (GLuint group, GLuint counter, GLenum pname, GLvoid *data);
typedef void (GL_APIENTRYP PFNGLGENPERFMONITORSAMDPROC) (GLsizei n, GLuint *monitors);
typedef void (GL_APIENTRYP PFNGLDELETEPERFMONITORSAMDPROC) (GLsizei n, GLuint *monitors);
typedef void (GL_APIENTRYP PFNGLSELECTPERFMONITORCOUNTERSAMDPROC) (GLuint monitor, GLboolean enable, GLuint group, GLint numCounters, GLuint *countersList);
typedef void (GL_APIENTRYP PFNGLBEGINPERFMONITORAMDPROC) (GLuint monitor);
typedef void (GL_APIENTRYP PFNGLENDPERFMONITORAMDPROC) (GLuint monitor);
typedef void (GL_APIENTRYP PFNGLGETPERFMONITORCOUNTERDATAAMDPROC) (GLuint monitor, GLenum pname, GLsizei dataSize, GLuint *data, GLint *bytesWritten);
#endif

/* GL_AMD_program_binary_Z400 */
#ifndef GL_AMD_program_binary_Z400
#define GL_AMD_program_binary_Z400 1
#endif

/*------------------------------------------------------------------------*
 * ANGLE extension functions
 *------------------------------------------------------------------------*/

/* GL_ANGLE_depth_texture */
#ifndef GL_ANGLE_depth_texture
#define GL_ANGLE_depth_texture 1
#endif

/* GL_ANGLE_framebuffer_blit */
#ifndef GL_ANGLE_framebuffer_blit
#define GL_ANGLE_framebuffer_blit 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glBlitFramebufferANGLE (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
#endif
typedef void (GL_APIENTRYP PFNGLBLITFRAMEBUFFERANGLEPROC) (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
#endif

/* GL_ANGLE_framebuffer_multisample */
#ifndef GL_ANGLE_framebuffer_multisample
#define GL_ANGLE_framebuffer_multisample 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glRenderbufferStorageMultisampleANGLE (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
#endif
typedef void (GL_APIENTRYP PFNGLRENDERBUFFERSTORAGEMULTISAMPLEANGLEPROC) (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
#endif

#ifndef GL_ANGLE_instanced_arrays 
#define GL_ANGLE_instanced_arrays 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glDrawArraysInstancedANGLE (GLenum mode, GLint first, GLsizei count, GLsizei primcount);
GL_APICALL void GL_APIENTRY glDrawElementsInstancedANGLE (GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei primcount);
GL_APICALL void GL_APIENTRY glVertexAttribDivisorANGLE (GLuint index, GLuint divisor);
#endif
typedef void (GL_APIENTRYP PFNGLDRAWARRAYSINSTANCEDANGLEPROC) (GLenum mode, GLint first, GLsizei count, GLsizei primcount);
typedef void (GL_APIENTRYP PFNGLDRAWELEMENTSINSTANCEDANGLEPROC) (GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei primcount);
typedef void (GL_APIENTRYP PFNGLVERTEXATTRIBDIVISORANGLEPROC) (GLuint index, GLuint divisor);
#endif

/* GL_ANGLE_pack_reverse_row_order */
#ifndef GL_ANGLE_pack_reverse_row_order 
#define GL_ANGLE_pack_reverse_row_order 1
#endif

/* GL_ANGLE_program_binary */
#ifndef GL_ANGLE_program_binary
#define GL_ANGLE_program_binary 1
#endif

/* GL_ANGLE_texture_compression_dxt3 */
#ifndef GL_ANGLE_texture_compression_dxt3 
#define GL_ANGLE_texture_compression_dxt3 1
#endif

/* GL_ANGLE_texture_compression_dxt5 */
#ifndef GL_ANGLE_texture_compression_dxt5 
#define GL_ANGLE_texture_compression_dxt5 1
#endif

/* GL_ANGLE_texture_usage */
#ifndef GL_ANGLE_texture_usage 
#define GL_ANGLE_texture_usage 1
#endif

#ifndef GL_ANGLE_translated_shader_source 
#define GL_ANGLE_translated_shader_source 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glGetTranslatedShaderSourceANGLE (GLuint shader, GLsizei bufsize, GLsizei *length, GLchar *source);
#endif
typedef void (GL_APIENTRYP PFNGLGETTRANSLATEDSHADERSOURCEANGLEPROC) (GLuint shader, GLsizei bufsize, GLsizei *length, GLchar *source);
#endif

/*------------------------------------------------------------------------*
 * APPLE extension functions
 *------------------------------------------------------------------------*/

/* GL_APPLE_copy_texture_levels */
#ifndef GL_APPLE_copy_texture_levels
#define GL_APPLE_copy_texture_levels 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glCopyTextureLevelsAPPLE (GLuint destinationTexture, GLuint sourceTexture, GLint sourceBaseLevel, GLsizei sourceLevelCount);
#endif
typedef void (GL_APIENTRYP PFNGLCOPYTEXTURELEVELSAPPLEPROC) (GLuint destinationTexture, GLuint sourceTexture, GLint sourceBaseLevel, GLsizei sourceLevelCount);
#endif

/* GL_APPLE_framebuffer_multisample */
#ifndef GL_APPLE_framebuffer_multisample
#define GL_APPLE_framebuffer_multisample 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glRenderbufferStorageMultisampleAPPLE (GLenum, GLsizei, GLenum, GLsizei, GLsizei);
GL_APICALL void GL_APIENTRY glResolveMultisampleFramebufferAPPLE (void);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GL_APIENTRYP PFNGLRENDERBUFFERSTORAGEMULTISAMPLEAPPLEPROC) (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
typedef void (GL_APIENTRYP PFNGLRESOLVEMULTISAMPLEFRAMEBUFFERAPPLEPROC) (void);
#endif

/* GL_APPLE_rgb_422 */
#ifndef GL_APPLE_rgb_422
#define GL_APPLE_rgb_422 1
#endif

/* GL_APPLE_sync */
#ifndef GL_APPLE_sync
#define GL_APPLE_sync 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL GLsync GL_APIENTRY glFenceSyncAPPLE (GLenum condition, GLbitfield flags);
GL_APICALL GLboolean GL_APIENTRY glIsSyncAPPLE (GLsync sync);
GL_APICALL void GL_APIENTRY glDeleteSyncAPPLE (GLsync sync);
GL_APICALL GLenum GL_APIENTRY glClientWaitSyncAPPLE (GLsync sync, GLbitfield flags, GLuint64 timeout);
GL_APICALL void GL_APIENTRY glWaitSyncAPPLE (GLsync sync, GLbitfield flags, GLuint64 timeout);
GL_APICALL void GL_APIENTRY glGetInteger64vAPPLE (GLenum pname, GLint64 *params);
GL_APICALL void GL_APIENTRY glGetSyncivAPPLE (GLsync sync, GLenum pname, GLsizei bufSize, GLsizei *length, GLint *values);
#endif
typedef GLsync (GL_APIENTRYP PFNGLFENCESYNCAPPLEPROC) (GLenum condition, GLbitfield flags);
typedef GLboolean (GL_APIENTRYP PFNGLISSYNCAPPLEPROC) (GLsync sync);
typedef void (GL_APIENTRYP PFNGLDELETESYNCAPPLEPROC) (GLsync sync);
typedef GLenum (GL_APIENTRYP PFNGLCLIENTWAITSYNCAPPLEPROC) (GLsync sync, GLbitfield flags, GLuint64 timeout);
typedef void (GL_APIENTRYP PFNGLWAITSYNCAPPLEPROC) (GLsync sync, GLbitfield flags, GLuint64 timeout);
typedef void (GL_APIENTRYP PFNGLGETINTEGER64VAPPLEPROC) (GLenum pname, GLint64 *params);
typedef void (GL_APIENTRYP PFNGLGETSYNCIVAPPLEPROC) (GLsync sync, GLenum pname, GLsizei bufSize, GLsizei *length, GLint *values);
#endif

/* GL_APPLE_texture_format_BGRA8888 */
#ifndef GL_APPLE_texture_format_BGRA8888
#define GL_APPLE_texture_format_BGRA8888 1
#endif

/* GL_APPLE_texture_max_level */
#ifndef GL_APPLE_texture_max_level
#define GL_APPLE_texture_max_level 1
#endif

/*------------------------------------------------------------------------*
 * ARM extension functions
 *------------------------------------------------------------------------*/

/* GL_ARM_mali_program_binary */
#ifndef GL_ARM_mali_program_binary
#define GL_ARM_mali_program_binary 1
#endif

/* GL_ARM_mali_shader_binary */
#ifndef GL_ARM_mali_shader_binary
#define GL_ARM_mali_shader_binary 1
#endif

/* GL_ARM_rgba8 */
#ifndef GL_ARM_rgba8
#define GL_ARM_rgba8 1
#endif

/*------------------------------------------------------------------------*
 * EXT extension functions
 *------------------------------------------------------------------------*/

/* GL_EXT_blend_minmax */
#ifndef GL_EXT_blend_minmax
#define GL_EXT_blend_minmax 1
#endif

/* GL_EXT_color_buffer_half_float */
#ifndef GL_EXT_color_buffer_half_float
#define GL_EXT_color_buffer_half_float 1
#endif

/* GL_EXT_debug_label */
#ifndef GL_EXT_debug_label
#define GL_EXT_debug_label 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glLabelObjectEXT (GLenum type, GLuint object, GLsizei length, const GLchar *label);
GL_APICALL void GL_APIENTRY glGetObjectLabelEXT (GLenum type, GLuint object, GLsizei bufSize, GLsizei *length, GLchar *label);
#endif
typedef void (GL_APIENTRYP PFNGLLABELOBJECTEXTPROC) (GLenum type, GLuint object, GLsizei length, const GLchar *label);
typedef void (GL_APIENTRYP PFNGLGETOBJECTLABELEXTPROC) (GLenum type, GLuint object, GLsizei bufSize, GLsizei *length, GLchar *label);
#endif

/* GL_EXT_debug_marker */
#ifndef GL_EXT_debug_marker
#define GL_EXT_debug_marker 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glInsertEventMarkerEXT (GLsizei length, const GLchar *marker);
GL_APICALL void GL_APIENTRY glPushGroupMarkerEXT (GLsizei length, const GLchar *marker);
GL_APICALL void GL_APIENTRY glPopGroupMarkerEXT (void);
#endif
typedef void (GL_APIENTRYP PFNGLINSERTEVENTMARKEREXTPROC) (GLsizei length, const GLchar *marker);
typedef void (GL_APIENTRYP PFNGLPUSHGROUPMARKEREXTPROC) (GLsizei length, const GLchar *marker);
typedef void (GL_APIENTRYP PFNGLPOPGROUPMARKEREXTPROC) (void);
#endif

/* GL_EXT_discard_framebuffer */
#ifndef GL_EXT_discard_framebuffer
#define GL_EXT_discard_framebuffer 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glDiscardFramebufferEXT (GLenum target, GLsizei numAttachments, const GLenum *attachments);
#endif
typedef void (GL_APIENTRYP PFNGLDISCARDFRAMEBUFFEREXTPROC) (GLenum target, GLsizei numAttachments, const GLenum *attachments);
#endif

/* GL_EXT_map_buffer_range */
#ifndef GL_EXT_map_buffer_range
#define GL_EXT_map_buffer_range 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void* GL_APIENTRY glMapBufferRangeEXT (GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
GL_APICALL void GL_APIENTRY glFlushMappedBufferRangeEXT (GLenum target, GLintptr offset, GLsizeiptr length);
#endif
typedef void* (GL_APIENTRYP PFNGLMAPBUFFERRANGEEXTPROC) (GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
typedef void (GL_APIENTRYP PFNGLFLUSHMAPPEDBUFFERRANGEEXTPROC) (GLenum target, GLintptr offset, GLsizeiptr length);
#endif

/* GL_EXT_multisampled_render_to_texture */
#ifndef GL_EXT_multisampled_render_to_texture
#define GL_EXT_multisampled_render_to_texture 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glRenderbufferStorageMultisampleEXT (GLenum, GLsizei, GLenum, GLsizei, GLsizei);
GL_APICALL void GL_APIENTRY glFramebufferTexture2DMultisampleEXT (GLenum, GLenum, GLenum, GLuint, GLint, GLsizei);
#endif
typedef void (GL_APIENTRYP PFNGLRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC) (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
typedef void (GL_APIENTRYP PFNGLFRAMEBUFFERTEXTURE2DMULTISAMPLEEXTPROC) (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLsizei samples);
#endif

/* GL_EXT_multiview_draw_buffers */
#ifndef GL_EXT_multiview_draw_buffers
#define GL_EXT_multiview_draw_buffers 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glReadBufferIndexedEXT (GLenum src, GLint index);
GL_APICALL void GL_APIENTRY glDrawBuffersIndexedEXT (GLint n, const GLenum *location, const GLint *indices);
GL_APICALL void GL_APIENTRY glGetIntegeri_vEXT (GLenum target, GLuint index, GLint *data);
#endif
typedef void (GL_APIENTRYP PFNGLREADBUFFERINDEXEDEXTPROC) (GLenum src, GLint index);
typedef void (GL_APIENTRYP PFNGLDRAWBUFFERSINDEXEDEXTPROC) (GLint n, const GLenum *location, const GLint *indices);
typedef void (GL_APIENTRYP PFNGLGETINTEGERI_VEXTPROC) (GLenum target, GLuint index, GLint *data);
#endif

#ifndef GL_EXT_multi_draw_arrays
#define GL_EXT_multi_draw_arrays 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glMultiDrawArraysEXT (GLenum, const GLint *, const GLsizei *, GLsizei);
GL_APICALL void GL_APIENTRY glMultiDrawElementsEXT (GLenum, const GLsizei *, GLenum, const GLvoid* *, GLsizei);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GL_APIENTRYP PFNGLMULTIDRAWARRAYSEXTPROC) (GLenum mode, GLint *first, GLsizei *count, GLsizei primcount);
typedef void (GL_APIENTRYP PFNGLMULTIDRAWELEMENTSEXTPROC) (GLenum mode, const GLsizei *count, GLenum type, const GLvoid* *indices, GLsizei primcount);
#endif

/* GL_EXT_occlusion_query_boolean */
#ifndef GL_EXT_occlusion_query_boolean
#define GL_EXT_occlusion_query_boolean 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glGenQueriesEXT (GLsizei n, GLuint *ids);
GL_APICALL void GL_APIENTRY glDeleteQueriesEXT (GLsizei n, const GLuint *ids);
GL_APICALL GLboolean GL_APIENTRY glIsQueryEXT (GLuint id);
GL_APICALL void GL_APIENTRY glBeginQueryEXT (GLenum target, GLuint id);
GL_APICALL void GL_APIENTRY glEndQueryEXT (GLenum target);
GL_APICALL void GL_APIENTRY glGetQueryivEXT (GLenum target, GLenum pname, GLint *params);
GL_APICALL void GL_APIENTRY glGetQueryObjectuivEXT (GLuint id, GLenum pname, GLuint *params);
#endif
typedef void (GL_APIENTRYP PFNGLGENQUERIESEXTPROC) (GLsizei n, GLuint *ids);
typedef void (GL_APIENTRYP PFNGLDELETEQUERIESEXTPROC) (GLsizei n, const GLuint *ids);
typedef GLboolean (GL_APIENTRYP PFNGLISQUERYEXTPROC) (GLuint id);
typedef void (GL_APIENTRYP PFNGLBEGINQUERYEXTPROC) (GLenum target, GLuint id);
typedef void (GL_APIENTRYP PFNGLENDQUERYEXTPROC) (GLenum target);
typedef void (GL_APIENTRYP PFNGLGETQUERYIVEXTPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GL_APIENTRYP PFNGLGETQUERYOBJECTUIVEXTPROC) (GLuint id, GLenum pname, GLuint *params);
#endif

/* GL_EXT_read_format_bgra */
#ifndef GL_EXT_read_format_bgra
#define GL_EXT_read_format_bgra 1
#endif

/* GL_EXT_robustness */
#ifndef GL_EXT_robustness
#define GL_EXT_robustness 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL GLenum GL_APIENTRY glGetGraphicsResetStatusEXT (void);
GL_APICALL void GL_APIENTRY glReadnPixelsEXT (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data);
GL_APICALL void GL_APIENTRY glGetnUniformfvEXT (GLuint program, GLint location, GLsizei bufSize, float *params);
GL_APICALL void GL_APIENTRY glGetnUniformivEXT (GLuint program, GLint location, GLsizei bufSize, GLint *params);
#endif
typedef GLenum (GL_APIENTRYP PFNGLGETGRAPHICSRESETSTATUSEXTPROC) (void);
typedef void (GL_APIENTRYP PFNGLREADNPIXELSEXTPROC) (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data);
typedef void (GL_APIENTRYP PFNGLGETNUNIFORMFVEXTPROC) (GLuint program, GLint location, GLsizei bufSize, float *params);
typedef void (GL_APIENTRYP PFNGLGETNUNIFORMIVEXTPROC) (GLuint program, GLint location, GLsizei bufSize, GLint *params);
#endif

/* GL_EXT_separate_shader_objects */
#ifndef GL_EXT_separate_shader_objects
#define GL_EXT_separate_shader_objects 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glUseProgramStagesEXT (GLuint pipeline, GLbitfield stages, GLuint program);
GL_APICALL void GL_APIENTRY glActiveShaderProgramEXT (GLuint pipeline, GLuint program);
GL_APICALL GLuint GL_APIENTRY glCreateShaderProgramvEXT (GLenum type, GLsizei count, const GLchar **strings);
GL_APICALL void GL_APIENTRY glBindProgramPipelineEXT (GLuint pipeline);
GL_APICALL void GL_APIENTRY glDeleteProgramPipelinesEXT (GLsizei n, const GLuint *pipelines);
GL_APICALL void GL_APIENTRY glGenProgramPipelinesEXT (GLsizei n, GLuint *pipelines);
GL_APICALL GLboolean GL_APIENTRY glIsProgramPipelineEXT (GLuint pipeline);
GL_APICALL void GL_APIENTRY glProgramParameteriEXT (GLuint program, GLenum pname, GLint value);
GL_APICALL void GL_APIENTRY glGetProgramPipelineivEXT (GLuint pipeline, GLenum pname, GLint *params);
GL_APICALL void GL_APIENTRY glProgramUniform1iEXT (GLuint program, GLint location, GLint x);
GL_APICALL void GL_APIENTRY glProgramUniform2iEXT (GLuint program, GLint location, GLint x, GLint y);
GL_APICALL void GL_APIENTRY glProgramUniform3iEXT (GLuint program, GLint location, GLint x, GLint y, GLint z);
GL_APICALL void GL_APIENTRY glProgramUniform4iEXT (GLuint program, GLint location, GLint x, GLint y, GLint z, GLint w);
GL_APICALL void GL_APIENTRY glProgramUniform1fEXT (GLuint program, GLint location, GLfloat x);
GL_APICALL void GL_APIENTRY glProgramUniform2fEXT (GLuint program, GLint location, GLfloat x, GLfloat y);
GL_APICALL void GL_APIENTRY glProgramUniform3fEXT (GLuint program, GLint location, GLfloat x, GLfloat y, GLfloat z);
GL_APICALL void GL_APIENTRY glProgramUniform4fEXT (GLuint program, GLint location, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
GL_APICALL void GL_APIENTRY glProgramUniform1ivEXT (GLuint program, GLint location, GLsizei count, const GLint *value);
GL_APICALL void GL_APIENTRY glProgramUniform2ivEXT (GLuint program, GLint location, GLsizei count, const GLint *value);
GL_APICALL void GL_APIENTRY glProgramUniform3ivEXT (GLuint program, GLint location, GLsizei count, const GLint *value);
GL_APICALL void GL_APIENTRY glProgramUniform4ivEXT (GLuint program, GLint location, GLsizei count, const GLint *value);
GL_APICALL void GL_APIENTRY glProgramUniform1fvEXT (GLuint program, GLint location, GLsizei count, const GLfloat *value);
GL_APICALL void GL_APIENTRY glProgramUniform2fvEXT (GLuint program, GLint location, GLsizei count, const GLfloat *value);
GL_APICALL void GL_APIENTRY glProgramUniform3fvEXT (GLuint program, GLint location, GLsizei count, const GLfloat *value);
GL_APICALL void GL_APIENTRY glProgramUniform4fvEXT (GLuint program, GLint location, GLsizei count, const GLfloat *value);
GL_APICALL void GL_APIENTRY glProgramUniformMatrix2fvEXT (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
GL_APICALL void GL_APIENTRY glProgramUniformMatrix3fvEXT (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
GL_APICALL void GL_APIENTRY glProgramUniformMatrix4fvEXT (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
GL_APICALL void GL_APIENTRY glValidateProgramPipelineEXT (GLuint pipeline);
GL_APICALL void GL_APIENTRY glGetProgramPipelineInfoLogEXT (GLuint pipeline, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
#endif
typedef void (GL_APIENTRYP PFNGLUSEPROGRAMSTAGESEXTPROC) (GLuint pipeline, GLbitfield stages, GLuint program);
typedef void (GL_APIENTRYP PFNGLACTIVESHADERPROGRAMEXTPROC) (GLuint pipeline, GLuint program);
typedef GLuint (GL_APIENTRYP PFNGLCREATESHADERPROGRAMVEXTPROC) (GLenum type, GLsizei count, const GLchar **strings);
typedef void (GL_APIENTRYP PFNGLBINDPROGRAMPIPELINEEXTPROC) (GLuint pipeline);
typedef void (GL_APIENTRYP PFNGLDELETEPROGRAMPIPELINESEXTPROC) (GLsizei n, const GLuint *pipelines);
typedef void (GL_APIENTRYP PFNGLGENPROGRAMPIPELINESEXTPROC) (GLsizei n, GLuint *pipelines);
typedef GLboolean (GL_APIENTRYP PFNGLISPROGRAMPIPELINEEXTPROC) (GLuint pipeline);
typedef void (GL_APIENTRYP PFNGLPROGRAMPARAMETERIEXTPROC) (GLuint program, GLenum pname, GLint value);
typedef void (GL_APIENTRYP PFNGLGETPROGRAMPIPELINEIVEXTPROC) (GLuint pipeline, GLenum pname, GLint *params);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM1IEXTPROC) (GLuint program, GLint location, GLint x);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM2IEXTPROC) (GLuint program, GLint location, GLint x, GLint y);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM3IEXTPROC) (GLuint program, GLint location, GLint x, GLint y, GLint z);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM4IEXTPROC) (GLuint program, GLint location, GLint x, GLint y, GLint z, GLint w);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM1FEXTPROC) (GLuint program, GLint location, GLfloat x);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM2FEXTPROC) (GLuint program, GLint location, GLfloat x, GLfloat y);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM3FEXTPROC) (GLuint program, GLint location, GLfloat x, GLfloat y, GLfloat z);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM4FEXTPROC) (GLuint program, GLint location, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM1IVEXTPROC) (GLuint program, GLint location, GLsizei count, const GLint *value);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM2IVEXTPROC) (GLuint program, GLint location, GLsizei count, const GLint *value);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM3IVEXTPROC) (GLuint program, GLint location, GLsizei count, const GLint *value);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM4IVEXTPROC) (GLuint program, GLint location, GLsizei count, const GLint *value);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM1FVEXTPROC) (GLuint program, GLint location, GLsizei count, const GLfloat *value);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM2FVEXTPROC) (GLuint program, GLint location, GLsizei count, const GLfloat *value);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM3FVEXTPROC) (GLuint program, GLint location, GLsizei count, const GLfloat *value);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORM4FVEXTPROC) (GLuint program, GLint location, GLsizei count, const GLfloat *value);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORMMATRIX2FVEXTPROC) (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORMMATRIX3FVEXTPROC) (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (GL_APIENTRYP PFNGLPROGRAMUNIFORMMATRIX4FVEXTPROC) (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (GL_APIENTRYP PFNGLVALIDATEPROGRAMPIPELINEEXTPROC) (GLuint pipeline);
typedef void (GL_APIENTRYP PFNGLGETPROGRAMPIPELINEINFOLOGEXTPROC) (GLuint pipeline, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
#endif

/* GL_EXT_shader_framebuffer_fetch */
#ifndef GL_EXT_shader_framebuffer_fetch
#define GL_EXT_shader_framebuffer_fetch 1
#endif

/* GL_EXT_shader_texture_lod */
#ifndef GL_EXT_shader_texture_lod
#define GL_EXT_shader_texture_lod 1
#endif

/* GL_EXT_shadow_samplers */
#ifndef GL_EXT_shadow_samplers
#define GL_EXT_shadow_samplers 1
#endif

/* GL_EXT_sRGB */
#ifndef GL_EXT_sRGB
#define GL_EXT_sRGB 1
#endif

/* GL_EXT_texture_compression_dxt1 */
#ifndef GL_EXT_texture_compression_dxt1
#define GL_EXT_texture_compression_dxt1 1
#endif

/* GL_EXT_texture_filter_anisotropic */
#ifndef GL_EXT_texture_filter_anisotropic
#define GL_EXT_texture_filter_anisotropic 1
#endif

/* GL_EXT_texture_format_BGRA8888 */
#ifndef GL_EXT_texture_format_BGRA8888
#define GL_EXT_texture_format_BGRA8888 1
#endif

/* GL_EXT_texture_rg */
#ifndef GL_EXT_texture_rg
#define GL_EXT_texture_rg 1
#endif

/* GL_EXT_texture_storage */
#ifndef GL_EXT_texture_storage
#define GL_EXT_texture_storage 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glTexStorage1DEXT (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width);
GL_APICALL void GL_APIENTRY glTexStorage2DEXT (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
GL_APICALL void GL_APIENTRY glTexStorage3DEXT (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
GL_APICALL void GL_APIENTRY glTextureStorage1DEXT (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width);
GL_APICALL void GL_APIENTRY glTextureStorage2DEXT (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
GL_APICALL void GL_APIENTRY glTextureStorage3DEXT (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
#endif
typedef void (GL_APIENTRYP PFNGLTEXSTORAGE1DEXTPROC) (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width);
typedef void (GL_APIENTRYP PFNGLTEXSTORAGE2DEXTPROC) (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
typedef void (GL_APIENTRYP PFNGLTEXSTORAGE3DEXTPROC) (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
typedef void (GL_APIENTRYP PFNGLTEXTURESTORAGE1DEXTPROC) (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width);
typedef void (GL_APIENTRYP PFNGLTEXTURESTORAGE2DEXTPROC) (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
typedef void (GL_APIENTRYP PFNGLTEXTURESTORAGE3DEXTPROC) (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
#endif

/* GL_EXT_texture_type_2_10_10_10_REV */
#ifndef GL_EXT_texture_type_2_10_10_10_REV
#define GL_EXT_texture_type_2_10_10_10_REV 1
#endif

/* GL_EXT_unpack_subimage */
#ifndef GL_EXT_unpack_subimage
#define GL_EXT_unpack_subimage 1
#endif

/*------------------------------------------------------------------------*
 * DMP extension functions
 *------------------------------------------------------------------------*/

/* GL_DMP_shader_binary */
#ifndef GL_DMP_shader_binary
#define GL_DMP_shader_binary 1
#endif

/*------------------------------------------------------------------------*
 * FJ extension functions
 *------------------------------------------------------------------------*/

/* GL_FJ_shader_binary_GCCSO */
#ifndef GL_FJ_shader_binary_GCCSO
#define GL_FJ_shader_binary_GCCSO 1
#endif

/*------------------------------------------------------------------------*
 * IMG extension functions
 *------------------------------------------------------------------------*/

/* GL_IMG_program_binary */
#ifndef GL_IMG_program_binary
#define GL_IMG_program_binary 1
#endif

/* GL_IMG_read_format */
#ifndef GL_IMG_read_format
#define GL_IMG_read_format 1
#endif

/* GL_IMG_shader_binary */
#ifndef GL_IMG_shader_binary
#define GL_IMG_shader_binary 1
#endif

/* GL_IMG_texture_compression_pvrtc */
#ifndef GL_IMG_texture_compression_pvrtc
#define GL_IMG_texture_compression_pvrtc 1
#endif

/* GL_IMG_texture_compression_pvrtc2 */
#ifndef GL_IMG_texture_compression_pvrtc2
#define GL_IMG_texture_compression_pvrtc2 1
#endif

/* GL_IMG_multisampled_render_to_texture */
#ifndef GL_IMG_multisampled_render_to_texture
#define GL_IMG_multisampled_render_to_texture 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glRenderbufferStorageMultisampleIMG (GLenum, GLsizei, GLenum, GLsizei, GLsizei);
GL_APICALL void GL_APIENTRY glFramebufferTexture2DMultisampleIMG (GLenum, GLenum, GLenum, GLuint, GLint, GLsizei);
#endif
typedef void (GL_APIENTRYP PFNGLRENDERBUFFERSTORAGEMULTISAMPLEIMGPROC) (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
typedef void (GL_APIENTRYP PFNGLFRAMEBUFFERTEXTURE2DMULTISAMPLEIMGPROC) (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLsizei samples);
#endif

/*------------------------------------------------------------------------*
 * NV extension functions
 *------------------------------------------------------------------------*/

/* GL_NV_coverage_sample */
#ifndef GL_NV_coverage_sample
#define GL_NV_coverage_sample 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glCoverageMaskNV (GLboolean mask);
GL_APICALL void GL_APIENTRY glCoverageOperationNV (GLenum operation);
#endif
typedef void (GL_APIENTRYP PFNGLCOVERAGEMASKNVPROC) (GLboolean mask);
typedef void (GL_APIENTRYP PFNGLCOVERAGEOPERATIONNVPROC) (GLenum operation);
#endif

/* GL_NV_depth_nonlinear */
#ifndef GL_NV_depth_nonlinear
#define GL_NV_depth_nonlinear 1
#endif

/* GL_NV_draw_buffers */
#ifndef GL_NV_draw_buffers
#define GL_NV_draw_buffers 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glDrawBuffersNV (GLsizei n, const GLenum *bufs);
#endif
typedef void (GL_APIENTRYP PFNGLDRAWBUFFERSNVPROC) (GLsizei n, const GLenum *bufs);
#endif

/* GL_EXT_draw_buffers */
#ifndef GL_EXT_draw_buffers
#define GL_EXT_draw_buffers 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glDrawBuffersEXT (GLsizei n, const GLenum *bufs);
#endif
typedef void (GL_APIENTRYP PFNGLDRAWBUFFERSEXTPROC) (GLsizei n, const GLenum *bufs);
#endif

/* GL_NV_draw_instanced */
#ifndef GL_NV_draw_instanced
#define GL_NV_draw_instanced 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glDrawArraysInstancedNV (GLenum mode, GLint first, GLsizei count, GLsizei primcount);
GL_APICALL void GL_APIENTRY glDrawElementsInstancedNV (GLenum mode, GLsizei count, GLenum type, const GLvoid *indices, GLsizei primcount);
#endif
typedef void (GL_APIENTRYP PFNDRAWARRAYSINSTANCEDNVPROC) (GLenum mode, GLint first, GLsizei count, GLsizei primcount);
typedef void (GL_APIENTRYP PFNDRAWELEMENTSINSTANCEDNVPROC) (GLenum mode, GLsizei count, GLenum type, const GLvoid *indices, GLsizei primcount);
#endif

/* GL_NV_fbo_color_attachments */
#ifndef GL_NV_fbo_color_attachments
#define GL_NV_fbo_color_attachments 1
#endif

/* GL_NV_fence */
#ifndef GL_NV_fence
#define GL_NV_fence 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glDeleteFencesNV (GLsizei, const GLuint *);
GL_APICALL void GL_APIENTRY glGenFencesNV (GLsizei, GLuint *);
GL_APICALL GLboolean GL_APIENTRY glIsFenceNV (GLuint);
GL_APICALL GLboolean GL_APIENTRY glTestFenceNV (GLuint);
GL_APICALL void GL_APIENTRY glGetFenceivNV (GLuint, GLenum, GLint *);
GL_APICALL void GL_APIENTRY glFinishFenceNV (GLuint);
GL_APICALL void GL_APIENTRY glSetFenceNV (GLuint, GLenum);
#endif
typedef void (GL_APIENTRYP PFNGLDELETEFENCESNVPROC) (GLsizei n, const GLuint *fences);
typedef void (GL_APIENTRYP PFNGLGENFENCESNVPROC) (GLsizei n, GLuint *fences);
typedef GLboolean (GL_APIENTRYP PFNGLISFENCENVPROC) (GLuint fence);
typedef GLboolean (GL_APIENTRYP PFNGLTESTFENCENVPROC) (GLuint fence);
typedef void (GL_APIENTRYP PFNGLGETFENCEIVNVPROC) (GLuint fence, GLenum pname, GLint *params);
typedef void (GL_APIENTRYP PFNGLFINISHFENCENVPROC) (GLuint fence);
typedef void (GL_APIENTRYP PFNGLSETFENCENVPROC) (GLuint fence, GLenum condition);
#endif

/* GL_NV_framebuffer_blit */
#ifndef GL_NV_framebuffer_blit
#define GL_NV_framebuffer_blit 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glBlitFramebufferNV (int srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
#endif
typedef void (GL_APIENTRYP PFNBLITFRAMEBUFFERNVPROC) (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
#endif

/* GL_NV_framebuffer_multisample */
#ifndef GL_NV_framebuffer_multisample
#define GL_NV_framebuffer_multisample 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glRenderbufferStorageMultisampleNV ( GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
#endif
typedef void (GL_APIENTRYP PFNRENDERBUFFERSTORAGEMULTISAMPLENVPROC) ( GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
#endif

/* GL_NV_generate_mipmap_sRGB */
#ifndef GL_NV_generate_mipmap_sRGB
#define GL_NV_generate_mipmap_sRGB 1
#endif

/* GL_NV_instanced_arrays */
#ifndef GL_NV_instanced_arrays
#define GL_NV_instanced_arrays 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glVertexAttribDivisorNV (GLuint index, GLuint divisor);
#endif
typedef void (GL_APIENTRYP PFNVERTEXATTRIBDIVISORNVPROC) (GLuint index, GLuint divisor);
#endif

/* GL_NV_read_buffer */
#ifndef GL_NV_read_buffer
#define GL_NV_read_buffer 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glReadBufferNV (GLenum mode);
#endif
typedef void (GL_APIENTRYP PFNGLREADBUFFERNVPROC) (GLenum mode);
#endif

/* GL_NV_read_buffer_front */
#ifndef GL_NV_read_buffer_front
#define GL_NV_read_buffer_front 1
#endif

/* GL_NV_read_depth */
#ifndef GL_NV_read_depth
#define GL_NV_read_depth 1
#endif

/* GL_NV_read_depth_stencil */
#ifndef GL_NV_read_depth_stencil
#define GL_NV_read_depth_stencil 1
#endif

/* GL_NV_read_stencil */
#ifndef GL_NV_read_stencil
#define GL_NV_read_stencil 1
#endif

/* GL_NV_shadow_samplers_array */
#ifndef GL_NV_shadow_samplers_array
#define GL_NV_shadow_samplers_array 1
#endif

/* GL_NV_shadow_samplers_cube */
#ifndef GL_NV_shadow_samplers_cube
#define GL_NV_shadow_samplers_cube 1
#endif

/* GL_NV_sRGB_formats */
#ifndef GL_NV_sRGB_formats
#define GL_NV_sRGB_formats 1
#endif

/* GL_NV_texture_border_clamp */
#ifndef GL_NV_texture_border_clamp
#define GL_NV_texture_border_clamp 1
#endif

/* GL_NV_texture_compression_s3tc_update */
#ifndef GL_NV_texture_compression_s3tc_update
#define GL_NV_texture_compression_s3tc_update 1
#endif

/* GL_NV_texture_npot_2D_mipmap */
#ifndef GL_NV_texture_npot_2D_mipmap
#define GL_NV_texture_npot_2D_mipmap 1
#endif

/*------------------------------------------------------------------------*
 * QCOM extension functions
 *------------------------------------------------------------------------*/

/* GL_QCOM_alpha_test */
#ifndef GL_QCOM_alpha_test
#define GL_QCOM_alpha_test 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glAlphaFuncQCOM (GLenum func, GLclampf ref);
#endif
typedef void (GL_APIENTRYP PFNGLALPHAFUNCQCOMPROC) (GLenum func, GLclampf ref);
#endif

/* GL_QCOM_binning_control */
#ifndef GL_QCOM_binning_control
#define GL_QCOM_binning_control 1
#endif

/* GL_QCOM_driver_control */
#ifndef GL_QCOM_driver_control
#define GL_QCOM_driver_control 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glGetDriverControlsQCOM (GLint *num, GLsizei size, GLuint *driverControls);
GL_APICALL void GL_APIENTRY glGetDriverControlStringQCOM (GLuint driverControl, GLsizei bufSize, GLsizei *length, GLchar *driverControlString);
GL_APICALL void GL_APIENTRY glEnableDriverControlQCOM (GLuint driverControl);
GL_APICALL void GL_APIENTRY glDisableDriverControlQCOM (GLuint driverControl);
#endif
typedef void (GL_APIENTRYP PFNGLGETDRIVERCONTROLSQCOMPROC) (GLint *num, GLsizei size, GLuint *driverControls);
typedef void (GL_APIENTRYP PFNGLGETDRIVERCONTROLSTRINGQCOMPROC) (GLuint driverControl, GLsizei bufSize, GLsizei *length, GLchar *driverControlString);
typedef void (GL_APIENTRYP PFNGLENABLEDRIVERCONTROLQCOMPROC) (GLuint driverControl);
typedef void (GL_APIENTRYP PFNGLDISABLEDRIVERCONTROLQCOMPROC) (GLuint driverControl);
#endif

/* GL_QCOM_extended_get */
#ifndef GL_QCOM_extended_get
#define GL_QCOM_extended_get 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glExtGetTexturesQCOM (GLuint *textures, GLint maxTextures, GLint *numTextures);
GL_APICALL void GL_APIENTRY glExtGetBuffersQCOM (GLuint *buffers, GLint maxBuffers, GLint *numBuffers);
GL_APICALL void GL_APIENTRY glExtGetRenderbuffersQCOM (GLuint *renderbuffers, GLint maxRenderbuffers, GLint *numRenderbuffers);
GL_APICALL void GL_APIENTRY glExtGetFramebuffersQCOM (GLuint *framebuffers, GLint maxFramebuffers, GLint *numFramebuffers);
GL_APICALL void GL_APIENTRY glExtGetTexLevelParameterivQCOM (GLuint texture, GLenum face, GLint level, GLenum pname, GLint *params);
GL_APICALL void GL_APIENTRY glExtTexObjectStateOverrideiQCOM (GLenum target, GLenum pname, GLint param);
GL_APICALL void GL_APIENTRY glExtGetTexSubImageQCOM (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLvoid *texels);
GL_APICALL void GL_APIENTRY glExtGetBufferPointervQCOM (GLenum target, GLvoid **params);
#endif
typedef void (GL_APIENTRYP PFNGLEXTGETTEXTURESQCOMPROC) (GLuint *textures, GLint maxTextures, GLint *numTextures);
typedef void (GL_APIENTRYP PFNGLEXTGETBUFFERSQCOMPROC) (GLuint *buffers, GLint maxBuffers, GLint *numBuffers);
typedef void (GL_APIENTRYP PFNGLEXTGETRENDERBUFFERSQCOMPROC) (GLuint *renderbuffers, GLint maxRenderbuffers, GLint *numRenderbuffers);
typedef void (GL_APIENTRYP PFNGLEXTGETFRAMEBUFFERSQCOMPROC) (GLuint *framebuffers, GLint maxFramebuffers, GLint *numFramebuffers);
typedef void (GL_APIENTRYP PFNGLEXTGETTEXLEVELPARAMETERIVQCOMPROC) (GLuint texture, GLenum face, GLint level, GLenum pname, GLint *params);
typedef void (GL_APIENTRYP PFNGLEXTTEXOBJECTSTATEOVERRIDEIQCOMPROC) (GLenum target, GLenum pname, GLint param);
typedef void (GL_APIENTRYP PFNGLEXTGETTEXSUBIMAGEQCOMPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLvoid *texels);
typedef void (GL_APIENTRYP PFNGLEXTGETBUFFERPOINTERVQCOMPROC) (GLenum target, GLvoid **params);
#endif

/* GL_QCOM_extended_get2 */
#ifndef GL_QCOM_extended_get2
#define GL_QCOM_extended_get2 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glExtGetShadersQCOM (GLuint *shaders, GLint maxShaders, GLint *numShaders);
GL_APICALL void GL_APIENTRY glExtGetProgramsQCOM (GLuint *programs, GLint maxPrograms, GLint *numPrograms);
GL_APICALL GLboolean GL_APIENTRY glExtIsProgramBinaryQCOM (GLuint program);
GL_APICALL void GL_APIENTRY glExtGetProgramBinarySourceQCOM (GLuint program, GLenum shadertype, GLchar *source, GLint *length);
#endif
typedef void (GL_APIENTRYP PFNGLEXTGETSHADERSQCOMPROC) (GLuint *shaders, GLint maxShaders, GLint *numShaders);
typedef void (GL_APIENTRYP PFNGLEXTGETPROGRAMSQCOMPROC) (GLuint *programs, GLint maxPrograms, GLint *numPrograms);
typedef GLboolean (GL_APIENTRYP PFNGLEXTISPROGRAMBINARYQCOMPROC) (GLuint program);
typedef void (GL_APIENTRYP PFNGLEXTGETPROGRAMBINARYSOURCEQCOMPROC) (GLuint program, GLenum shadertype, GLchar *source, GLint *length);
#endif

/* GL_QCOM_perfmon_global_mode */
#ifndef GL_QCOM_perfmon_global_mode
#define GL_QCOM_perfmon_global_mode 1
#endif

/* GL_QCOM_writeonly_rendering */
#ifndef GL_QCOM_writeonly_rendering
#define GL_QCOM_writeonly_rendering 1
#endif

/* GL_QCOM_tiled_rendering */
#ifndef GL_QCOM_tiled_rendering
#define GL_QCOM_tiled_rendering 1
#ifdef GL_GLEXT_PROTOTYPES
GL_APICALL void GL_APIENTRY glStartTilingQCOM (GLuint x, GLuint y, GLuint width, GLuint height, GLbitfield preserveMask);
GL_APICALL void GL_APIENTRY glEndTilingQCOM (GLbitfield preserveMask);
#endif
typedef void (GL_APIENTRYP PFNGLSTARTTILINGQCOMPROC) (GLuint x, GLuint y, GLuint width, GLuint height, GLbitfield preserveMask);
typedef void (GL_APIENTRYP PFNGLENDTILINGQCOMPROC) (GLbitfield preserveMask);
#endif

/*------------------------------------------------------------------------*
 * VIV extension tokens
 *------------------------------------------------------------------------*/

/* GL_VIV_shader_binary */
#ifndef GL_VIV_shader_binary
#define GL_VIV_shader_binary 1
#endif

#ifdef __cplusplus
}
#endif

#endif /* __gl2ext_h_ */

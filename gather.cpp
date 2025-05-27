#include <riscv_vector.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
static int gather_hwc_batch_rvv(std::vector<float>& output,
                                const std::vector<float>& input,
                                const std::vector<int>& in_shape_nhwc,
                                const std::vector<int>& indices, int axis_nchw,
                                int align_channels) {
  // 检查NCHW格式的axis是否有效
  if (axis_nchw > 3) {
    std::cerr << "无效的axis_nchw：" << axis_nchw << std::endl;
    return -1;
  }

  // 从NHWC形状提取维度
  auto N = in_shape_nhwc[0];
  auto H = in_shape_nhwc[1];
  auto W = in_shape_nhwc[2];
  auto C = in_shape_nhwc[3];

  // 将NCHW格式的axis转换为NHWC格式的axis
  std::size_t axis_nhwc;
  if (axis_nchw == 0) {
    axis_nhwc = 0;
  } else if (axis_nchw == 1) {
    axis_nhwc = 3;
  } else if (axis_nchw == 2) {
    axis_nhwc = 1;
  } else {  // axis_nchw == 3
    axis_nhwc = 2;
  }

  // 计算通道块数
  auto num_channel_blocks = (C + align_channels - 1) / align_channels;

  if (axis_nchw == 0) {
    // 在N维度上gather (axis_nhwc=0)
    auto out_N = indices.size();
    output.resize(out_N * H * W * C, 0.0f);
    // 对每个索引处理
    for (std::size_t i = 0; i < indices.size(); ++i) {
      // 处理负索引
      std::size_t n_idx = indices[i] >= 0 ? indices[i] : indices[i] + N;
      if (n_idx >= N) {
        std::cerr << "索引越界" << std::endl;
        return -1;
      }

      // 对每个通道块处理
      for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
        auto input_start = c_block * (N * H * W * align_channels) +
                           n_idx * (H * W * align_channels);
        auto output_start = c_block * (out_N * H * W * align_channels) +
                            i * (H * W * align_channels);
        auto n = H * W * align_channels;
        while (n > 0) {
          // 加载vl个元素到向量寄存器
          auto vl = vsetvl_e32m8(n);
          auto v_in = vle32_v_f32m8(input.data() + input_start, vl);
          // 输出到output
          vse32_v_f32m8(output.data() + output_start, v_in, vl);
          // 更新偏移
          input_start += vl;
          output_start += vl;
          n -= vl;
        }
      }
    }
  } else if (axis_nchw == 1) {
    // 在C维度上gather (axis_nhwc=3)
    auto out_C = indices.size();
    auto out_num_channel_blocks = (out_C + align_channels - 1) / align_channels;
    auto C_padded_out = out_num_channel_blocks * align_channels;
    output.resize(N * H * W * C_padded_out, 0.0f);

    // 对每个索引处理
    for (std::size_t i = 0; i < indices.size(); ++i) {
      // 处理负索引
      std::size_t c_idx = indices[i] >= 0 ? indices[i] : indices[i] + C;
      if (c_idx >= C) {
        std::cerr << "索引越界" << std::endl;
        return -1;
      }

      // 计算原始通道所在的块索引和块内偏移
      auto c_block = c_idx / align_channels;
      auto c_offset = c_idx % align_channels;

      // 计算输出通道的块索引和块内偏移
      auto out_c_block = i / align_channels;
      auto out_c_offset = i % align_channels;

      auto input_start = c_block * (N * H * W * align_channels) + c_offset;
      auto output_start =
          out_c_block * (N * H * W * align_channels) + out_c_offset;

      // in bytes
      auto stride = align_channels * sizeof(float);
      auto n = N * H * W;
      while (n > 0) {
        std::size_t vl = vsetvl_e32m8(n);
        auto v_in = vlse32_v_f32m8(input.data() + input_start, stride, vl);
        vsse32_v_f32m8(output.data() + output_start, stride, v_in, vl);
        input_start += vl * align_channels;
        output_start += vl * align_channels;
        n -= vl;
      }
    }
  } else if (axis_nchw == 2) {
    // 在H维度上gather (axis_nhwc=1)
    auto out_H = indices.size();
    output.resize(N * out_H * W * C, 0.0f);

    for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
      for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t i = 0; i < indices.size(); ++i) {
          std::size_t h_idx = indices[i] >= 0 ? indices[i] : indices[i] + H;
          if (h_idx >= H) {
            std::cerr << "索引越界" << std::endl;
            return -1;
          }
          auto input_start = c_block * (N * H * W * align_channels) +
                             n * (H * W * align_channels) +
                             h_idx * (W * align_channels);
          auto output_start = c_block * (N * out_H * W * align_channels) +
                              n * (out_H * W * align_channels) +
                              i * (W * align_channels);

          auto n = W * align_channels;
          while (n > 0) {
            auto vl = vsetvl_e32m8(n);
            auto v_in = vle32_v_f32m8(input.data() + input_start, vl);
            // 输出到output
            vse32_v_f32m8(output.data() + output_start, v_in, vl);
            input_start += vl;
            output_start += vl;
            n -= vl;
          }
        }
      }
    }
  } else if (axis_nchw == 3) {
    // 在W维度上gather (axis_nhwc=2)
    auto out_W = indices.size();
    output.resize(N * H * out_W * C, 0.0f);

    for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
      for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t h = 0; h < H; ++h) {
          for (std::size_t i = 0; i < indices.size(); ++i) {
            std::size_t w_idx = indices[i] >= 0 ? indices[i] : indices[i] + W;
            if (w_idx >= W) {
              std::cerr << "索引越界" << std::endl;
              return -1;
            }
            auto input_start = c_block * (N * H * W * align_channels) +
                               n * (H * W * align_channels) +
                               h * (W * align_channels) +
                               w_idx * align_channels;
            auto output_start = c_block * (N * H * out_W * align_channels) +
                                n * (H * out_W * align_channels) +
                                h * (out_W * align_channels) +
                                i * align_channels;

            auto n = align_channels;
            while (n > 0) {
              auto vl = vsetvl_e32m8(n);
              auto v_in = vle32_v_f32m8(input.data() + input_start, vl);
              // 输出到output
              vse32_v_f32m8(output.data() + output_start, v_in, vl);
              input_start += vl;
              output_start += vl;
              n -= vl;
            }
          }
        }
      }
    }
  }

  return 0;
}

// NDHWC <-> NCDHW : RVV gather 5-D
int gather_hwc_batch5d_rvv(
    std::vector<float>& output, const std::vector<float>& input,
    const std::vector<int>& in_shape_ndhwc,  // {N,D,H,W,C}
    const std::vector<int>& indices,
    int axis_ncdhw,  // 以 NCDHW 编号
    int align_channels) {
  if (axis_ncdhw > 4) {
    std::cerr << "无效 axis_ncdhw\n";
    return -1;
  }

  /* ---------- 解析 NDHWC 形状 ---------- */
  const int N = in_shape_ndhwc[0];
  const int D = in_shape_ndhwc[1];
  const int H = in_shape_ndhwc[2];
  const int W = in_shape_ndhwc[3];
  const int C = in_shape_ndhwc[4];

  /* ---------- 轴映射 NCDHW → NDHWC ---------- */
  int axis_ndhwc;
  switch (axis_ncdhw) {
    case 0:
      axis_ndhwc = 0;
      break;  // N
    case 1:
      axis_ndhwc = 4;
      break;  // C
    case 2:
      axis_ndhwc = 1;
      break;  // D
    case 3:
      axis_ndhwc = 2;
      break;  // H
    default:
      axis_ndhwc = 3;
      break;  // W
  }

  const int num_c_blocks = (C + align_channels - 1) / align_channels;

  /* ================= 按轴处理 ================= */

  /* -------- axis = N (batch) -------- */
  if (axis_ncdhw == 0) {
    const int outN = indices.size();
    output.resize(outN * D * H * W * C, 0.f);

    for (int i = 0; i < outN; ++i) {
      int n_idx = indices[i] >= 0 ? indices[i] : indices[i] + N;
      if (n_idx < 0 || n_idx >= N) {
        std::cerr << "索引越界\n";
        return -1;
      }

      for (int cb = 0; cb < num_c_blocks; ++cb) {
        int in_off = cb * (N * D * H * W * align_channels) +
                     n_idx * (D * H * W * align_channels);
        int out_off = cb * (outN * D * H * W * align_channels) +
                      i * (D * H * W * align_channels);

        int n = D * H * W * align_channels;
        while (n > 0) {
          int vl = vsetvl_e32m8(n);
          auto v = vle32_v_f32m8(input.data() + in_off, vl);
          vse32_v_f32m8(output.data() + out_off, v, vl);
          in_off += vl;
          out_off += vl;
          n -= vl;
        }
      }
    }
  }

  /* -------- axis = C (channel) -------- */
  else if (axis_ncdhw == 1) {
    const int outC = indices.size();
    const int out_c_blocks = (outC + align_channels - 1) / align_channels;
    const int C_pad_out = out_c_blocks * align_channels;
    output.resize(N * D * H * W * C_pad_out, 0.f);

    for (int i = 0; i < outC; ++i) {
      int c_idx = indices[i] >= 0 ? indices[i] : indices[i] + C;
      if (c_idx < 0 || c_idx >= C) {
        std::cerr << "索引越界\n";
        return -1;
      }

      int in_blk = c_idx / align_channels;
      int in_offC = c_idx % align_channels;
      int out_blk = i / align_channels;
      int out_offC = i % align_channels;

      int stride = align_channels * sizeof(float);

      int input_base = in_blk * (N * D * H * W * align_channels) + in_offC;
      int output_base = out_blk * (N * D * H * W * align_channels) + out_offC;

      int n_elems = N * D * H * W;
      while (n_elems > 0) {
        int vl = vsetvl_e32m8(n_elems);
        auto v = vlse32_v_f32m8(input.data() + input_base, stride, vl);
        vsse32_v_f32m8(output.data() + output_base, stride, v, vl);
        input_base += vl * align_channels;
        output_base += vl * align_channels;
        n_elems -= vl;
      }
    }
  }

  /* -------- axis = D (depth) -------- */
  else if (axis_ncdhw == 2) {
    const int outD = indices.size();
    output.resize(N * outD * H * W * C, 0.f);

    for (int cb = 0; cb < num_c_blocks; ++cb)
      for (int n = 0; n < N; ++n)
        for (int i = 0; i < outD; ++i) {
          int d_idx = indices[i] >= 0 ? indices[i] : indices[i] + D;
          if (d_idx < 0 || d_idx >= D) {
            std::cerr << "索引越界\n";
            return -1;
          }

          int in_off = cb * (N * D * H * W * align_channels) +
                       n * (D * H * W * align_channels) +
                       d_idx * (H * W * align_channels);
          int out_off = cb * (N * outD * H * W * align_channels) +
                        n * (outD * H * W * align_channels) +
                        i * (H * W * align_channels);

          int n_elem = H * W * align_channels;
          while (n_elem > 0) {
            int vl = vsetvl_e32m8(n_elem);
            auto v = vle32_v_f32m8(input.data() + in_off, vl);
            vse32_v_f32m8(output.data() + out_off, v, vl);
            in_off += vl;
            out_off += vl;
            n_elem -= vl;
          }
        }
  }

  /* -------- axis = H (height) -------- */
  else if (axis_ncdhw == 3) {
    const int outH = indices.size();
    output.resize(N * D * outH * W * C, 0.f);

    for (int cb = 0; cb < num_c_blocks; ++cb)
      for (int n = 0; n < N; ++n)
        for (int d = 0; d < D; ++d)
          for (int i = 0; i < outH; ++i) {
            int h_idx = indices[i] >= 0 ? indices[i] : indices[i] + H;
            if (h_idx < 0 || h_idx >= H) {
              std::cerr << "索引越界\n";
              return -1;
            }

            int in_off = cb * (N * D * H * W * align_channels) +
                         n * (D * H * W * align_channels) +
                         d * (H * W * align_channels) +
                         h_idx * (W * align_channels);

            int out_off = cb * (N * D * outH * W * align_channels) +
                          n * (D * outH * W * align_channels) +
                          d * (outH * W * align_channels) +
                          i * (W * align_channels);

            int n_elem = W * align_channels;
            while (n_elem > 0) {
              int vl = vsetvl_e32m8(n_elem);
              auto v = vle32_v_f32m8(input.data() + in_off, vl);
              vse32_v_f32m8(output.data() + out_off, v, vl);
              in_off += vl;
              out_off += vl;
              n_elem -= vl;
            }
          }
  }

  /* -------- axis = W (width) -------- */
  else /* axis_ncdhw == 4 */
  {
    const int outW = indices.size();
    output.resize(N * D * H * outW * C, 0.f);

    for (int cb = 0; cb < num_c_blocks; ++cb)
      for (int n = 0; n < N; ++n)
        for (int d = 0; d < D; ++d)
          for (int h = 0; h < H; ++h)
            for (int i = 0; i < outW; ++i) {
              int w_idx = indices[i] >= 0 ? indices[i] : indices[i] + W;
              if (w_idx < 0 || w_idx >= W) {
                std::cerr << "索引越界\n";
                return -1;
              }

              int in_off = cb * (N * D * H * W * align_channels) +
                           n * (D * H * W * align_channels) +
                           d * (H * W * align_channels) +
                           h * (W * align_channels) + w_idx * align_channels;

              int out_off = cb * (N * D * H * outW * align_channels) +
                            n * (D * H * outW * align_channels) +
                            d * (H * outW * align_channels) +
                            h * (outW * align_channels) + i * align_channels;

              int remain = align_channels;
              while (remain > 0) {
                int vl = vsetvl_e32m8(remain);
                auto v = vle32_v_f32m8(input.data() + in_off, vl);
                vse32_v_f32m8(output.data() + out_off, v, vl);
                in_off += vl;
                out_off += vl;
                remain -= vl;
              }
            }
  }

  return 0;
}

int gather_hwc_batch_mem(std::vector<float>& output,
                         const std::vector<float>& input,
                         const std::vector<std::size_t>& in_shape_nhwc,
                         const std::vector<std::size_t>& indices,
                         std::size_t axis_nchw, std::size_t align_channels) {
  // 检查NCHW格式的axis是否有效
  if (axis_nchw > 3) {
    std::cerr << "无效的axis_nchw：" << axis_nchw << std::endl;
    return -1;
  }

  // 从NHWC形状提取维度
  std::size_t N = in_shape_nhwc[0];
  std::size_t H = in_shape_nhwc[1];
  std::size_t W = in_shape_nhwc[2];
  std::size_t C = in_shape_nhwc[3];

  // 将NCHW格式的axis转换为NHWC格式的axis
  std::size_t axis_nhwc;
  if (axis_nchw == 0) {
    axis_nhwc = 0;
  } else if (axis_nchw == 1) {
    axis_nhwc = 3;
  } else if (axis_nchw == 2) {
    axis_nhwc = 1;
  } else {  // axis_nchw == 3
    axis_nhwc = 2;
  }

  // 计算通道块数和填充后的通道数
  std::size_t num_channel_blocks = (C + align_channels - 1) / align_channels;

  if (axis_nchw == 0) {
    // 在N维度上gather (axis_nhwc=0)
    std::size_t out_N = indices.size();
    output.resize(out_N * H * W * C, 0.0f);
    // 对每个索引处理
    for (std::size_t i = 0; i < indices.size(); ++i) {
      // 处理负索引
      std::size_t n_idx = indices[i] >= 0 ? indices[i] : indices[i] + N;
      if (n_idx >= N) {
        std::cerr << "索引越界" << std::endl;
        return -1;
      }

      // 对每个通道块处理
      for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
        std::size_t in_offset = c_block * (N * H * W * align_channels) +
                                n_idx * (H * W * align_channels);
        std::size_t out_offset = c_block * (out_N * H * W * align_channels) +
                                 i * (H * W * align_channels);
        std::memcpy(output.data() + out_offset, input.data() + in_offset,
                    H * W * align_channels * sizeof(float));
      }
    }
  } else if (axis_nchw == 1) {
    // 在C维度上gather (axis_nhwc=3)
    std::size_t out_C = indices.size();
    std::size_t out_num_channel_blocks =
        (out_C + align_channels - 1) / align_channels;
    std::size_t C_padded_out = out_num_channel_blocks * align_channels;
    output.resize(N * H * W * C_padded_out, 0.0f);

    // 对每个索引处理
    for (std::size_t i = 0; i < indices.size(); ++i) {
      // 处理负索引
      std::size_t c_idx = indices[i] >= 0 ? indices[i] : indices[i] + C;
      if (c_idx >= C) {
        std::cerr << "索引越界" << std::endl;
        return -1;
      }

      // 计算原始通道所在的块索引和块内偏移
      std::size_t c_block = c_idx / align_channels;
      std::size_t c_offset = c_idx % align_channels;

      // 计算输出通道的块索引和块内偏移
      std::size_t out_c_block = i / align_channels;
      std::size_t out_c_offset = i % align_channels;

      // 对每个N,H,W位置处理
      for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t h = 0; h < H; ++h) {
          for (std::size_t w = 0; w < W; ++w) {
            // 按截断存储格式计算偏移
            std::size_t in_offset = c_block * (N * H * W * align_channels) +
                                    n * (H * W * align_channels) +
                                    h * (W * align_channels) +
                                    w * align_channels + c_offset;

            std::size_t out_offset =
                out_c_block * (N * H * W * align_channels) +
                n * (H * W * align_channels) + h * (W * align_channels) +
                w * align_channels + out_c_offset;

            output[out_offset] = input[in_offset];
          }
        }
      }
    }
  } else if (axis_nchw == 2) {
    // 在H维度上gather (axis_nhwc=1)
    std::size_t out_H = indices.size();
    output.resize(N * out_H * W * C, 0.0f);

    for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
      for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t i = 0; i < indices.size(); ++i) {
          std::size_t h_idx = indices[i] >= 0 ? indices[i] : indices[i] + H;
          if (h_idx >= H) {
            std::cerr << "索引越界" << std::endl;
            return -1;
          }
          std::size_t in_offset = c_block * (N * H * W * align_channels) +
                                  n * (H * W * align_channels) +
                                  h_idx * (W * align_channels);
          std::size_t out_offset = c_block * (N * out_H * W * align_channels) +
                                   n * (out_H * W * align_channels) +
                                   i * (W * align_channels);
          std::memcpy(output.data() + out_offset, input.data() + in_offset,
                      W * align_channels * sizeof(float));
        }
      }
    }
  } else if (axis_nchw == 3) {
    // 在W维度上gather (axis_nhwc=2)
    std::size_t out_W = indices.size();
    output.resize(N * H * out_W * C, 0.0f);

    // 对每个通道块处理
    for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
      for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t h = 0; h < H; ++h) {
          for (std::size_t i = 0; i < indices.size(); ++i) {
            std::size_t w_idx = indices[i] >= 0 ? indices[i] : indices[i] + W;
            if (w_idx >= W) {
              std::cerr << "索引越界" << std::endl;
              return -1;
            }
            std::size_t in_offset = c_block * (N * H * W * align_channels) +
                                    n * (H * W * align_channels) +
                                    h * (W * align_channels) +
                                    w_idx * align_channels;
            std::size_t out_offset =
                c_block * (N * H * out_W * align_channels) +
                n * (H * out_W * align_channels) +
                h * (out_W * align_channels) + i * align_channels;
            std::memcpy(output.data() + out_offset, input.data() + in_offset,
                        align_channels * sizeof(float));
          }
        }
      }
    }
  }

  return 0;
}

int gather_hwc_batch5d_mem(std::vector<float>& output,
                           const std::vector<float>& input,
                           const std::vector<std::size_t>& in_shape_lnhwc,
                           const std::vector<std::size_t>& indices,
                           std::size_t axis_lnchw, std::size_t align_channels) {
  // 检查LNCHW格式的axis是否有效
  if (axis_lnchw > 4) {
    std::cerr << "无效的axis_lnchw：" << axis_lnchw << std::endl;
    return -1;
  }

  // 从LNHWC形状提取维度
  std::size_t L = in_shape_lnhwc[0];  // LNHWC的第一个维度是L
  std::size_t N = in_shape_lnhwc[1];  // LNHWC的第二个维度是N
  std::size_t H = in_shape_lnhwc[2];  // LNHWC的第三个维度是H
  std::size_t W = in_shape_lnhwc[3];  // LNHWC的第四个维度是W
  std::size_t C = in_shape_lnhwc[4];  // LNHWC的第五个维度是C

  // 将LNCHW格式的axis转换为LNHWC格式的axis
  std::size_t axis_lnhwc;
  if (axis_lnchw == 0) {
    axis_lnhwc = 0;  // LNCHW的L维度对应LNHWC的第一个维度
  } else if (axis_lnchw == 1) {
    axis_lnhwc = 1;  // LNCHW的N维度对应LNHWC的第二个维度
  } else if (axis_lnchw == 2) {
    axis_lnhwc = 4;  // LNCHW的C维度对应LNHWC的最后一个维度
  } else if (axis_lnchw == 3) {
    axis_lnhwc = 2;  // LNCHW的H维度对应LNHWC的第三个维度
  } else {           // axis_lnchw == 4
    axis_lnhwc = 3;  // LNCHW的W维度对应LNHWC的第四个维度
  }

  // 计算通道块数
  std::size_t num_channel_blocks = (C + align_channels - 1) / align_channels;

  // 根据axis_lnchw的不同，采用不同的策略
  if (axis_lnchw == 0) {
    // 在L维度上gather (axis_lnhwc=0)
    std::size_t out_L = indices.size();
    output.resize(out_L * N * H * W * C, 0.0f);

    // 对每个索引处理
    for (std::size_t i = 0; i < indices.size(); ++i) {
      // 处理负索引
      std::size_t l_idx = indices[i] >= 0 ? indices[i] : indices[i] + L;
      if (l_idx >= L) {
        std::cerr << "索引越界" << std::endl;
        return -1;
      }

      // 对每个通道块处理
      for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
        std::size_t in_offset = c_block * (L * N * H * W * align_channels) +
                                l_idx * (N * H * W * align_channels);
        std::size_t out_offset =
            c_block * (out_L * N * H * W * align_channels) +
            i * (N * H * W * align_channels);
        std::memcpy(output.data() + out_offset, input.data() + in_offset,
                    N * H * W * align_channels * sizeof(float));
      }
    }
  } else if (axis_lnchw == 1) {
    // 在N维度上gather (axis_lnhwc=1)
    std::size_t out_N = indices.size();
    output.resize(L * out_N * H * W * C, 0.0f);

    // 对每个通道块处理
    for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
      // 对每个L处理
      for (std::size_t l = 0; l < L; ++l) {
        // 对每个索引处理
        for (std::size_t i = 0; i < indices.size(); ++i) {
          // 处理负索引
          std::size_t n_idx = indices[i] >= 0 ? indices[i] : indices[i] + N;
          if (n_idx >= N) {
            std::cerr << "索引越界" << std::endl;
            return -1;
          }

          std::size_t in_offset = c_block * (L * N * H * W * align_channels) +
                                  l * (N * H * W * align_channels) +
                                  n_idx * (H * W * align_channels);
          std::size_t out_offset =
              c_block * (L * out_N * H * W * align_channels) +
              l * (out_N * H * W * align_channels) +
              i * (H * W * align_channels);
          std::memcpy(output.data() + out_offset, input.data() + in_offset,
                      H * W * align_channels * sizeof(float));
        }
      }
    }
  } else if (axis_lnchw == 2) {
    // 在C维度上gather (axis_lnhwc=4)
    std::size_t out_C = indices.size();
    std::size_t out_num_channel_blocks =
        (out_C + align_channels - 1) / align_channels;
    std::size_t C_padded_out = out_num_channel_blocks * align_channels;
    output.resize(L * N * H * W * C_padded_out, 0.0f);

    // 对每个索引处理
    for (std::size_t i = 0; i < indices.size(); ++i) {
      // 处理负索引
      std::size_t c_idx = indices[i] >= 0 ? indices[i] : indices[i] + C;
      if (c_idx >= C) {
        std::cerr << "索引越界" << std::endl;
        return -1;
      }

      // 计算原始通道所在的块索引和块内偏移
      std::size_t c_block = c_idx / align_channels;
      std::size_t c_offset = c_idx % align_channels;

      // 计算输出通道的块索引和块内偏移
      std::size_t out_c_block = i / align_channels;
      std::size_t out_c_offset = i % align_channels;

      // 对每个L,N,H,W位置处理
      for (std::size_t l = 0; l < L; ++l) {
        for (std::size_t n = 0; n < N; ++n) {
          for (std::size_t h = 0; h < H; ++h) {
            for (std::size_t w = 0; w < W; ++w) {
              // 按截断存储格式计算偏移
              std::size_t in_offset =
                  c_block * (L * N * H * W * align_channels) +
                  l * (N * H * W * align_channels) +
                  n * (H * W * align_channels) + h * (W * align_channels) +
                  w * align_channels + c_offset;

              std::size_t out_offset =
                  out_c_block * (L * N * H * W * align_channels) +
                  l * (N * H * W * align_channels) +
                  n * (H * W * align_channels) + h * (W * align_channels) +
                  w * align_channels + out_c_offset;

              output[out_offset] = input[in_offset];
            }
          }
        }
      }
    }
  } else if (axis_lnchw == 3) {
    // 在H维度上gather (axis_lnhwc=2)
    std::size_t out_H = indices.size();
    output.resize(L * N * out_H * W * C, 0.0f);

    // 对每个通道块处理
    for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
      // 对每个L,N处理
      for (std::size_t l = 0; l < L; ++l) {
        for (std::size_t n = 0; n < N; ++n) {
          // 对每个索引处理
          for (std::size_t i = 0; i < indices.size(); ++i) {
            // 处理负索引
            std::size_t h_idx = indices[i] >= 0 ? indices[i] : indices[i] + H;
            if (h_idx >= H) {
              std::cerr << "索引越界" << std::endl;
              return -1;
            }
            std::size_t in_offset = c_block * (L * N * H * W * align_channels) +
                                    l * (N * H * W * align_channels) +
                                    n * (H * W * align_channels) +
                                    h_idx * (W * align_channels);
            std::size_t out_offset =
                c_block * (L * N * out_H * W * align_channels) +
                l * (N * out_H * W * align_channels) +
                n * (out_H * W * align_channels) + i * (W * align_channels);
            std::memcpy(output.data() + out_offset, input.data() + in_offset,
                        W * align_channels * sizeof(float));
          }
        }
      }
    }
  } else if (axis_lnchw == 4) {
    // 在W维度上gather (axis_lnhwc=3)
    std::size_t out_W = indices.size();
    output.resize(L * N * H * out_W * C, 0.0f);

    // 对每个通道块处理
    for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
      // 对每个L,N,H处理
      for (std::size_t l = 0; l < L; ++l) {
        for (std::size_t n = 0; n < N; ++n) {
          for (std::size_t h = 0; h < H; ++h) {
            // 对每个索引处理
            for (std::size_t i = 0; i < indices.size(); ++i) {
              // 处理负索引
              std::size_t w_idx = indices[i] >= 0 ? indices[i] : indices[i] + W;
              if (w_idx >= W) {
                std::cerr << "索引越界" << std::endl;
                return -1;
              }

              std::size_t in_offset =
                  c_block * (L * N * H * W * align_channels) +
                  l * (N * H * W * align_channels) +
                  n * (H * W * align_channels) + h * (W * align_channels) +
                  w_idx * align_channels;

              std::size_t out_offset =
                  c_block * (L * N * H * out_W * align_channels) +
                  l * (N * H * out_W * align_channels) +
                  n * (H * out_W * align_channels) +
                  h * (out_W * align_channels) + i * align_channels;

              std::memcpy(output.data() + out_offset, input.data() + in_offset,
                          align_channels * sizeof(float));
            }
          }
        }
      }
    }
  }

  return 0;
}
}  // namespace

namespace rvv {
int gather_hwc(std::vector<float>& output, const std::vector<float>& input,
               const std::vector<int>& in_shape_hwc,
               const std::vector<int>& indices, int axis_chw,
               int align_channels) {
  if (in_shape_hwc.size() == 4) {
    return gather_hwc_batch_rvv(output, input, in_shape_hwc, indices, axis_chw,
                                align_channels);
  } else if (in_shape_hwc.size() == 5) {
    return gather_hwc_batch5d_rvv(output, input, in_shape_hwc, indices,
                                  axis_chw, align_channels);
  } else if (in_shape_hwc.size() != 3) {
    std::cerr << "无效的输入形状：" << in_shape_hwc.size() << std::endl;
    return -1;
  }
  // 检查CHW格式的axis是否有效
  if (axis_chw > 2) {
    std::cerr << "无效的axis_chw：" << axis_chw << std::endl;
    return -1;
  }

  // 从HWC形状提取维度
  auto H = in_shape_hwc[0];
  auto W = in_shape_hwc[1];
  auto C = in_shape_hwc[2];

  std::size_t axis_hwc;
  if (axis_chw == 0) {
    axis_hwc = 2;
  } else if (axis_chw == 1) {
    axis_hwc = 0;
  } else {
    axis_hwc = 1;
  }

  // 计算通道块数
  auto num_channel_blocks = (C + align_channels - 1) / align_channels;
  if (axis_chw == 0) {
    // 在C维度上gather (axis_hwc=2)
    auto out_C = indices.size();
    auto out_num_channel_blocks = (out_C + align_channels - 1) / align_channels;
    auto C_padded_out = out_num_channel_blocks * align_channels;
    output.resize(H * W * C_padded_out, 0.0f);  // 初始化为0

    // e32表示元素宽度32位，m4表示LMUL=4:4个向量寄存器组成一个逻辑寄存器，avl是希望的元素数。该函数返回实际设置的向量长度vl
    // auto expected_vl = H * W;
    // std::size_t actual_vl = vsetvl_e32m4(expected_vl);

    // 对每个索引处理
    for (std::size_t i = 0; i < indices.size(); ++i) {
      // 处理负索引
      auto c_idx = indices[i] >= 0 ? indices[i] : indices[i] + C;
      if (c_idx >= C) {
        std::cerr << "索引越界：" << indices[i] << std::endl;
        return -1;
      }

      // 计算原始通道所在的块索引和块内（通道）偏移
      auto c_block = c_idx / align_channels;
      auto c_offset = c_idx % align_channels;

      // 计算输出通道的块索引和块内（通道）偏移
      auto out_c_block = i / align_channels;
      auto out_c_offset = i % align_channels;

      auto input_start = c_block * (H * W * align_channels) + c_offset;
      auto output_start = out_c_block * (H * W * align_channels) + out_c_offset;

      // in bytes
      auto stride = align_channels * sizeof(float);

      auto n = H * W;
      while (n > 0) {
        auto vl = vsetvl_e32m8(n);
        auto v_in = vlse32_v_f32m8(input.data() + input_start, stride, vl);
        vsse32_v_f32m8(output.data() + output_start, stride, v_in, vl);
        // 更新偏移
        input_start += vl * align_channels;
        output_start += vl * align_channels;
        n -= vl;
      }
    }
  } else if (axis_chw == 1) {
    auto out_H = indices.size();
    output.resize(out_H * W * C, 0.0f);

    for (int i = 0; i < indices.size(); i++) {
      std::size_t h_idx = indices[i] >= 0 ? indices[i] : indices[i] + H;
      if (h_idx >= H) {
        std::cerr << "索引越界：" << indices[i] << std::endl;
        return -1;
      }
      // h需要对每个块进行处理, 每个块提取连续的W*align_channels个数
      for (int c_block = 0; c_block < num_channel_blocks; ++c_block) {
        auto input_start =
            c_block * (H * W * align_channels) + h_idx * (W * align_channels);
        auto output_start =
            c_block * (H * W * align_channels) + i * (W * align_channels);
        // 连续存储的W*align_channels个元素
        auto n = W * align_channels;
        while (n > 0) {
          std::size_t vl = vsetvl_e32m8(n);
          auto v_in = vle32_v_f32m8(input.data() + input_start, vl);
          // 输出到output
          vse32_v_f32m8(output.data() + output_start, v_in, vl);
          // 更新偏移
          input_start += vl;
          output_start += vl;
          n -= vl;
        }
      }
    }
  } else if (axis_chw == 2) {
    // 在W维度上gather
    auto out_W = indices.size();
    output.resize(H * out_W * C, 0.0f);

    // w需要对每个块中每个h处理
    for (auto h = 0; h < H; ++h) {
      for (auto i = 0; i < indices.size(); ++i) {
        std::size_t w_idx = indices[i] >= 0 ? indices[i] : indices[i] + W;
        if (w_idx >= W) {
          std::cerr << "索引越界：" << indices[i] << std::endl;
          return -1;
        }
        for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
          auto input_start = c_block * (H * W * align_channels) +
                             h * (W * align_channels) + w_idx * align_channels;
          auto output_start = c_block * (H * out_W * align_channels) +
                              h * (out_W * align_channels) + i * align_channels;
          auto n = align_channels;
          while (n > 0) {
            std::size_t vl = vsetvl_e32m8(n);
            auto v_in = vle32_v_f32m8(input.data() + input_start, vl);
            // 输出到output
            vse32_v_f32m8(output.data() + output_start, v_in, vl);
            // 更新偏移
            input_start += vl;
            output_start += vl;
            n -= vl;
          }
        }
      }
    }
  }

  return 0;
}

}  // namespace rvv

namespace mem {
int gather_hwc(std::vector<float>& output, const std::vector<float>& input,
               const std::vector<std::size_t>& in_shape_hwc,
               const std::vector<std::size_t>& indices, std::size_t axis_chw,
               std::size_t align_channels) {
  if (in_shape_hwc.size() == 4) {
    return gather_hwc_batch_mem(output, input, in_shape_hwc, indices, axis_chw,
                                align_channels);
  } else if (in_shape_hwc.size() == 5) {
    return gather_hwc_batch5d_mem(output, input, in_shape_hwc, indices,
                                  axis_chw, align_channels);
  } else if (in_shape_hwc.size() != 3) {
    std::cerr << "无效的输入形状：" << in_shape_hwc.size() << std::endl;
    return -1;
  }
  // 检查CHW格式的axis是否有效
  if (axis_chw > 2) {
    std::cerr << "无效的axis_chw：" << axis_chw << std::endl;
    return -1;
  }

  // 从HWC形状提取维度
  std::size_t H = in_shape_hwc[0];  // HWC的第一个维度是H
  std::size_t W = in_shape_hwc[1];  // HWC的第二个维度是W
  std::size_t C = in_shape_hwc[2];  // HWC的第三个维度是C

  // 将CHW格式的axis转换为HWC格式的axis
  std::size_t axis_hwc;
  if (axis_chw == 0) {
    axis_hwc = 2;  // CHW的C维度对应HWC的最后一个维度
  } else if (axis_chw == 1) {
    axis_hwc = 0;  // CHW的H维度对应HWC的第一个维度
  } else {         // axis_chw == 2
    axis_hwc = 1;  // CHW的W维度对应HWC的第二个维度
  }

  // 计算通道块数和填充后的通道数
  std::size_t num_channel_blocks = (C + align_channels - 1) / align_channels;

  if (axis_chw == 0) {
    // 在C维度上gather (axis_hwc=2)
    std::size_t out_C = indices.size();
    std::size_t out_num_channel_blocks =
        (out_C + align_channels - 1) / align_channels;
    std::size_t C_padded_out = out_num_channel_blocks * align_channels;
    output.resize(H * W * C_padded_out, 0.0f);  // 初始化为0

    // 对每个索引处理
    for (std::size_t i = 0; i < indices.size(); ++i) {
      // 处理负索引
      std::size_t c_idx = indices[i] >= 0 ? indices[i] : indices[i] + C;
      if (c_idx >= C) {
        std::cerr << "索引越界：" << indices[i] << std::endl;
        return -1;
      }

      // 计算原始通道所在的块索引和块内（通道）偏移
      std::size_t c_block = c_idx / align_channels;
      std::size_t c_offset = c_idx % align_channels;

      // 计算输出通道的块索引和块内（通道）偏移
      std::size_t out_c_block = i / align_channels;
      std::size_t out_c_offset = i % align_channels;

      // 对每个H,W位置处理
      for (std::size_t h = 0; h < H; ++h) {
        for (std::size_t w = 0; w < W; ++w) {
          // 按截断存储格式计算偏移：
          // 输入偏移：先是所有位置的前align_channels个通道，然后是所有位置的下一个align_channels通道...
          std::size_t in_offset =
              c_block * (H * W * align_channels) +  // 该块的第0个元素的偏移
              h * (W * align_channels) + w * align_channels + c_offset;

          // 输出偏移：同样的逻辑，但基于输出的通道块
          std::size_t out_offset = out_c_block * (H * W * align_channels) +
                                   h * (W * align_channels) +
                                   w * align_channels + out_c_offset;

          output[out_offset] = input[in_offset];
        }
      }
    }
  } else if (axis_chw == 1) {
    // 在H维度上gather (axis_hwc=0)
    std::size_t out_H = indices.size();
    output.resize(out_H * W * C, 0.0f);

    // 对每个索引处理
    for (std::size_t i = 0; i < indices.size(); ++i) {
      // 处理负索引
      std::size_t h_idx = indices[i] >= 0 ? indices[i] : indices[i] + H;
      if (h_idx >= H) {
        std::cerr << "索引越界：" << indices[i] << std::endl;
        return -1;
      }

      for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
        std::size_t in_offset =
            c_block * (H * W * align_channels) +  // 该块的第一个元素的偏移
            h_idx * (W * align_channels);  // h=h_idx对应的第一个元素的偏移
        std::size_t out_offset =
            c_block * (out_H * W * align_channels) + i * (W * align_channels);
        std::memcpy(output.data() + out_offset, input.data() + in_offset,
                    W * align_channels * sizeof(float));
      }
    }
  } else if (axis_chw == 2) {
    // 在W维度上gather (axis_hwc=1)
    std::size_t out_W = indices.size();
    output.resize(H * out_W * C, 0.0f);

    // 对每个H处理
    for (std::size_t h = 0; h < H; ++h) {
      // 对每个索引处理
      for (std::size_t i = 0; i < indices.size(); ++i) {
        // 处理负索引
        std::size_t w_idx = indices[i] >= 0 ? indices[i] : indices[i] + W;
        if (w_idx >= W) {
          std::cerr << "索引越界：" << indices[i] << std::endl;
          return -1;
        }

        // 对每个通道块处理
        for (std::size_t c_block = 0; c_block < num_channel_blocks; ++c_block) {
          std::size_t in_offset =
              c_block * (H * W * align_channels) +  // 该块的第一个元素的偏移
              h * (W * align_channels) +            // 该h对应的第一个元素的偏移
              w_idx * align_channels;  // w=w_idx对应的第一个元素的偏移
          std::size_t out_offset = c_block * (H * out_W * align_channels) +
                                   h * (out_W * align_channels) +
                                   i * align_channels;
          std::memcpy(output.data() + out_offset, input.data() + in_offset,
                      align_channels * sizeof(float));
        }
      }
    }
  }

  return 0;
}
}  // namespace mem

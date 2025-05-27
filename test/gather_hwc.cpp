

#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

int gather_hwc_batch_mem(std::vector<float> &output,
                         const std::vector<float> &input,
                         const std::vector<int> &in_shape_nhwc,
                         const std::vector<int> &indices, int axis_nchw,
                         int align_channels) {
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
  } else { // axis_nchw == 3
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

int gather_hwc_batch5d_mem(std::vector<float> &output,
                           const std::vector<float> &input,
                           const std::vector<int> &in_shape_ndhwc,
                           const std::vector<int> &indices, int axis_ncdhw,
                           int align_channels) {
  if (axis_ncdhw > 4) { // 轴合法性
    std::cerr << "无效 axis_ncdhw: " << axis_ncdhw << '\n';
    return -1;
  }

  /* --------------- 解析 NDHWC 形状 --------------- */
  const std::size_t N = in_shape_ndhwc[0];
  const std::size_t D = in_shape_ndhwc[1];
  const std::size_t H = in_shape_ndhwc[2];
  const std::size_t W = in_shape_ndhwc[3];
  const std::size_t C = in_shape_ndhwc[4];

  /* --------------- 通道块数 --------------- */
  const std::size_t num_c_blocks = (C + align_channels - 1) / align_channels;

  std::cout << "axis_ncdhw = " << axis_ncdhw << std::endl;
  std::cout << "H = " << H << std::endl;
  std::cout << "W = " << W << std::endl;

  /* ================ 逐轴处理 ================ */

  /* -------- axis = N (batch) -------- */
  if (axis_ncdhw == 0) {
    const std::size_t outN = indices.size();
    output.resize(outN * D * H * W * C, 0.0f);

    for (std::size_t i = 0; i < indices.size(); ++i) {

      std::size_t n_idx = indices[i] >= 0 ? indices[i] : indices[i] + N;
      if (n_idx >= N) {
        std::cerr << "索引越界\n";
        return -1;
      }

      for (std::size_t cb = 0; cb < num_c_blocks; ++cb) {
        std::size_t in_off = cb * (N * D * H * W * align_channels) +
                             n_idx * (D * H * W * align_channels);
        std::size_t out_off = cb * (outN * D * H * W * align_channels) +
                              i * (D * H * W * align_channels);
        std::memcpy(output.data() + out_off, input.data() + in_off,
                    D * H * W * align_channels * sizeof(float));
      }
    }
  }

  /* -------- axis = C (channel) -------- */
  else if (axis_ncdhw == 1) {
    std::cout << "axis_ncdhw == 1" << std::endl;
    const std::size_t outC = indices.size();
    const std::size_t out_c_blocks =
        (outC + align_channels - 1) / align_channels;
    const std::size_t C_pad_out = out_c_blocks * align_channels;
    output.resize(N * D * H * W * C_pad_out, 0.0f);

    for (std::size_t i = 0; i < indices.size(); ++i) {
      std::cout << "i = " << i << std::endl;
      std::size_t c_idx = indices[i] >= 0 ? indices[i] : indices[i] + C;
      if (c_idx >= C) {
        std::cerr << "索引越界\n";
        return -1;
      }

      std::size_t in_blk = c_idx / align_channels;
      std::size_t in_off_c = c_idx % align_channels;
      std::size_t out_blk = i / align_channels;
      std::size_t out_off_c = i % align_channels;

      for (std::size_t n = 0; n < N; ++n)
        for (std::size_t d = 0; d < D; ++d)
          for (std::size_t h = 0; h < H; ++h)
            for (std::size_t w = 0; w < W; ++w) {
              std::size_t in_off = in_blk * (N * D * H * W * align_channels) +
                                   n * (D * H * W * align_channels) +
                                   d * (H * W * align_channels) +
                                   h * (W * align_channels) +
                                   w * align_channels + in_off_c;

              std::size_t out_off = out_blk * (N * D * H * W * align_channels) +
                                    n * (D * H * W * align_channels) +
                                    d * (H * W * align_channels) +
                                    h * (W * align_channels) +
                                    w * align_channels + out_off_c;

              output[out_off] = input[in_off];
            }
    }
  }

  /* -------- axis = D (depth) -------- */
  else if (axis_ncdhw == 2) {
    const std::size_t outD = indices.size();
    output.resize(N * outD * H * W * C, 0.0f);

    for (std::size_t cb = 0; cb < num_c_blocks; ++cb)
      for (std::size_t n = 0; n < N; ++n)
        for (std::size_t i = 0; i < indices.size(); ++i) {
          std::size_t d_idx = indices[i] >= 0 ? indices[i] : indices[i] + D;
          if (d_idx >= D) {
            std::cerr << "索引越界\n";
            return -1;
          }

          std::size_t in_off = cb * (N * D * H * W * align_channels) +
                               n * (D * H * W * align_channels) +
                               d_idx * (H * W * align_channels);
          std::size_t out_off = cb * (N * outD * H * W * align_channels) +
                                n * (outD * H * W * align_channels) +
                                i * (H * W * align_channels);
          std::memcpy(output.data() + out_off, input.data() + in_off,
                      H * W * align_channels * sizeof(float));
        }
  }

  /* -------- axis = H (height) -------- */

  else if (axis_ncdhw == 3) {
    const std::size_t outH = indices.size();
    output.resize(N * D * outH * W * C, 0.0f);

    for (std::size_t cb = 0; cb < num_c_blocks; ++cb)
      for (std::size_t n = 0; n < N; ++n)
        for (std::size_t d = 0; d < D; ++d)
          for (std::size_t i = 0; i < indices.size(); ++i) {
            std::size_t h_idx = indices[i] >= 0 ? indices[i] : indices[i] + H;
            if (h_idx >= H) {
              std::cerr << "索引越界\n";
              return -1;
            }

            std::size_t in_off = cb * (N * D * H * W * align_channels) +
                                 n * (D * H * W * align_channels) +
                                 d * (H * W * align_channels) +
                                 h_idx * (W * align_channels);
            std::size_t out_off = cb * (N * D * outH * W * align_channels) +
                                  n * (D * outH * W * align_channels) +
                                  d * (outH * W * align_channels) +
                                  i * (W * align_channels);

            std::memcpy(output.data() + out_off, input.data() + in_off,
                        W * align_channels * sizeof(float));
          }
  }

  /* -------- axis = W (width) -------- */
  else { // axis_ncdhw == 4
    const std::size_t outW = indices.size();
    output.resize(N * D * H * outW * C, 0.0f);

    for (std::size_t cb = 0; cb < num_c_blocks; ++cb)
      for (std::size_t n = 0; n < N; ++n)
        for (std::size_t d = 0; d < D; ++d)
          for (std::size_t h = 0; h < H; ++h)
            for (std::size_t i = 0; i < indices.size(); ++i) {
              std::size_t w_idx = indices[i] >= 0 ? indices[i] : indices[i] + W;
              if (w_idx >= W) {
                std::cerr << "索引越界\n";
                return -1;
              }

              std::size_t in_off = cb * (N * D * H * W * align_channels) +
                                   n * (D * H * W * align_channels) +
                                   d * (H * W * align_channels) +
                                   h * (W * align_channels) +
                                   w_idx * align_channels;

              std::size_t out_off = cb * (N * D * H * outW * align_channels) +
                                    n * (D * H * outW * align_channels) +
                                    d * (H * outW * align_channels) +
                                    h * (outW * align_channels) +
                                    i * align_channels;

              std::memcpy(output.data() + out_off, input.data() + in_off,
                          align_channels * sizeof(float));
            }
  }

  return 0;
}
} // namespace

namespace mem {
int gather_hwc(std::vector<float> &output, const std::vector<float> &input,
               const std::vector<int> &in_shape_hwc,
               const std::vector<int> &indices, int axis_chw,
               int align_channels) {
  if (in_shape_hwc.size() == 4) {
    return gather_hwc_batch_mem(output, input, in_shape_hwc, indices, axis_chw,
                                align_channels);
  } else if (in_shape_hwc.size() == 5) {
    std::cout << "gather_hwc_batch5d_mem, axis_chw = " << axis_chw << std::endl;
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
  std::size_t H = in_shape_hwc[0]; // HWC的第一个维度是H
  std::size_t W = in_shape_hwc[1]; // HWC的第二个维度是W
  std::size_t C = in_shape_hwc[2]; // HWC的第三个维度是C

  // 将CHW格式的axis转换为HWC格式的axis
  std::size_t axis_hwc;
  if (axis_chw == 0) {
    axis_hwc = 2; // CHW的C维度对应HWC的最后一个维度
  } else if (axis_chw == 1) {
    axis_hwc = 0; // CHW的H维度对应HWC的第一个维度
  } else {        // axis_chw == 2
    axis_hwc = 1; // CHW的W维度对应HWC的第二个维度
  }

  // 计算通道块数和填充后的通道数
  std::size_t num_channel_blocks = (C + align_channels - 1) / align_channels;

  if (axis_chw == 0) {
    // 在C维度上gather (axis_hwc=2)
    std::size_t out_C = indices.size();
    std::size_t out_num_channel_blocks =
        (out_C + align_channels - 1) / align_channels;
    std::size_t C_padded_out = out_num_channel_blocks * align_channels;
    output.resize(H * W * C_padded_out, 0.0f); // 初始化为0

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
              c_block * (H * W * align_channels) + // 该块的第0个元素的偏移
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
            c_block * (H * W * align_channels) + // 该块的第一个元素的偏移
            h_idx * (W * align_channels); // h=h_idx对应的第一个元素的偏移
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
              c_block * (H * W * align_channels) + // 该块的第一个元素的偏移
              h * (W * align_channels) +           // 该h对应的第一个元素的偏移
              w_idx * align_channels; // w=w_idx对应的第一个元素的偏移
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
} // namespace mem

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

/**
 * 3维CHW到HWC转换，按指定通道数对齐（分组存储）
 * @param input CHW格式的输入数据
 * @param c 通道数
 * @param h 高度
 * @param w 宽度
 * @param align_channels 对齐的通道数（如64）
 * @return HWC格式的输出数据（分组存储格式）
 */
std::vector<float> convert_chw_to_hwc_3d(const std::vector<float>& input, int c,
                                         int h, int w,
                                         int align_channels = 64) {
  if (input.size() != c * h * w) {
    throw std::invalid_argument("输入数据大小与指定维度不匹配");
  }

  // 计算需要补零的通道数和分组数
  int padding_channels = (align_channels - c % align_channels) % align_channels;
  int total_channels = c + padding_channels;
  int num_groups = total_channels / align_channels;

  // 创建补零后的CHW数据
  std::vector<float> padded_data(total_channels * h * w, 0.0f);

  // 将原始数据拷贝到补零缓冲区
  for (int ci = 0; ci < c; ++ci) {
    for (int hi = 0; hi < h; ++hi) {
      for (int wi = 0; wi < w; ++wi) {
        int src_idx = ci * h * w + hi * w + wi;
        int dst_idx = ci * h * w + hi * w + wi;
        padded_data[dst_idx] = input[src_idx];
      }
    }
  }

  // 准备分组存储的结果数据缓冲区
  std::vector<float> result;
  result.reserve(h * w * total_channels);

  // 按对齐分组提取数据
  for (int g = 0; g < num_groups; ++g) {
    int channel_start = g * align_channels;

    for (int hi = 0; hi < h; ++hi) {
      for (int wi = 0; wi < w; ++wi) {
        for (int cg = 0; cg < align_channels; ++cg) {
          int channel = channel_start + cg;
          int idx = channel * h * w + hi * w + wi;
          result.push_back(padded_data[idx]);
        }
      }
    }
  }

  return result;
}

/**
 * 3维HWC到CHW转换（从分组存储格式）
 * @param input HWC格式的输入数据（分组存储格式）
 * @param c 原始通道数（不包括padding）
 * @param h 高度
 * @param w 宽度
 * @param align_channels 对齐的通道数
 * @return CHW格式的输出数据
 */
std::vector<float> convert_hwc_to_chw_3d(const std::vector<float>& input, int c,
                                         int h, int w,
                                         int align_channels = 64) {
  int padding_channels = (align_channels - c % align_channels) % align_channels;
  int total_channels = c + padding_channels;
  int num_groups = total_channels / align_channels;

  if (input.size() != h * w * total_channels) {
    throw std::invalid_argument("输入数据大小与指定维度不匹配");
  }

  // 从分组存储格式重构为补零CHW格式
  std::vector<float> padded_data(total_channels * h * w, 0.0f);

  int input_idx = 0;
  for (int g = 0; g < num_groups; ++g) {
    int channel_start = g * align_channels;

    for (int hi = 0; hi < h; ++hi) {
      for (int wi = 0; wi < w; ++wi) {
        for (int cg = 0; cg < align_channels; ++cg) {
          int channel = channel_start + cg;
          int chw_idx = channel * h * w + hi * w + wi;
          padded_data[chw_idx] = input[input_idx++];
        }
      }
    }
  }

  // 提取原始通道数据
  std::vector<float> output(c * h * w);
  for (int ci = 0; ci < c; ++ci) {
    for (int hi = 0; hi < h; ++hi) {
      for (int wi = 0; wi < w; ++wi) {
        int src_idx = ci * h * w + hi * w + wi;
        int dst_idx = ci * h * w + hi * w + wi;
        output[dst_idx] = padded_data[src_idx];
      }
    }
  }

  return output;
}

/**
 * 4维NCHW到NHWC转换，按指定通道数对齐（分组存储）
 * @param input NCHW格式的输入数据
 * @param n 批次数
 * @param c 通道数
 * @param h 高度
 * @param w 宽度
 * @param align_channels 对齐的通道数
 * @return NHWC格式的输出数据（分组存储格式）
 */
std::vector<float> convert_nchw_to_nhwc_4d(const std::vector<float>& input,
                                           int n, int c, int h, int w,
                                           int align_channels = 64) {
  if (input.size() != n * c * h * w) {
    throw std::invalid_argument("输入数据大小与指定维度不匹配");
  }

  // 计算需要补零的通道数和分组数
  int padding_channels = (align_channels - c % align_channels) % align_channels;
  int total_channels = c + padding_channels;
  int num_groups = total_channels / align_channels;

  // 创建补零后的NCHW数据
  std::vector<float> padded_data(n * total_channels * h * w, 0.0f);

  // 将原始数据拷贝到补零缓冲区
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int hi = 0; hi < h; ++hi) {
        for (int wi = 0; wi < w; ++wi) {
          int src_idx = ni * c * h * w + ci * h * w + hi * w + wi;
          int dst_idx = ni * total_channels * h * w + ci * h * w + hi * w + wi;
          padded_data[dst_idx] = input[src_idx];
        }
      }
    }
  }

  // 准备分组存储的结果数据缓冲区
  std::vector<float> result;
  result.reserve(n * h * w * total_channels);

  // 按对齐分组提取数据
  for (int g = 0; g < num_groups; ++g) {
    int channel_start = g * align_channels;

    for (int ni = 0; ni < n; ++ni) {
      for (int hi = 0; hi < h; ++hi) {
        for (int wi = 0; wi < w; ++wi) {
          for (int cg = 0; cg < align_channels; ++cg) {
            int channel = channel_start + cg;
            int idx =
                ni * total_channels * h * w + channel * h * w + hi * w + wi;
            result.push_back(padded_data[idx]);
          }
        }
      }
    }
  }

  return result;
}

/**
 * 4维NHWC到NCHW转换（从分组存储格式）
 * @param input NHWC格式的输入数据（分组存储格式）
 * @param n 批次数
 * @param c 原始通道数
 * @param h 高度
 * @param w 宽度
 * @param align_channels 对齐的通道数
 * @return NCHW格式的输出数据
 */
std::vector<float> convert_nhwc_to_nchw_4d(const std::vector<float>& input,
                                           int n, int c, int h, int w,
                                           int align_channels = 64) {
  int padding_channels = (align_channels - c % align_channels) % align_channels;
  int total_channels = c + padding_channels;
  int num_groups = total_channels / align_channels;

  if (input.size() != n * h * w * total_channels) {
    throw std::invalid_argument("输入数据大小与指定维度不匹配");
  }

  // 从分组存储格式重构为补零NCHW格式
  std::vector<float> padded_data(n * total_channels * h * w, 0.0f);

  int input_idx = 0;
  for (int g = 0; g < num_groups; ++g) {
    int channel_start = g * align_channels;

    for (int ni = 0; ni < n; ++ni) {
      for (int hi = 0; hi < h; ++hi) {
        for (int wi = 0; wi < w; ++wi) {
          for (int cg = 0; cg < align_channels; ++cg) {
            int channel = channel_start + cg;
            int nchw_idx =
                ni * total_channels * h * w + channel * h * w + hi * w + wi;
            padded_data[nchw_idx] = input[input_idx++];
          }
        }
      }
    }
  }

  // 提取原始通道数据
  std::vector<float> output(n * c * h * w);
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int hi = 0; hi < h; ++hi) {
        for (int wi = 0; wi < w; ++wi) {
          int src_idx = ni * total_channels * h * w + ci * h * w + hi * w + wi;
          int dst_idx = ni * c * h * w + ci * h * w + hi * w + wi;
          output[dst_idx] = padded_data[src_idx];
        }
      }
    }
  }

  return output;
}

/**
 * 5维LNCHW到LNHWC转换，按指定通道数对齐（分组存储）
 * @param input LNCHW格式的输入数据
 * @param l 长度/序列长度
 * @param n 批次数
 * @param h 高度
 * @param w 宽度
 * @param c 通道数
 * @param align_channels 对齐的通道数
 * @return LNHWC格式的输出数据（分组存储格式）
 */
std::vector<float> convert_lnchw_to_lnhwc_5d(const std::vector<float>& input,
                                             int l, int n, int h, int w, int c,
                                             int align_channels) {
  if (input.size() != l * n * c * h * w) {
    throw std::invalid_argument("输入数据大小与指定维度不匹配");
  }

  // 计算需要补零的通道数和分组数
  int padding_channels = (align_channels - c % align_channels) % align_channels;
  int total_channels = c + padding_channels;
  int num_groups = total_channels / align_channels;

  // 创建补零后的LNCHW数据
  std::vector<float> padded_data(l * n * total_channels * h * w, 0.0f);

  // 将原始数据拷贝到补零缓冲区
  for (int li = 0; li < l; ++li) {
    for (int ni = 0; ni < n; ++ni) {
      for (int ci = 0; ci < c; ++ci) {
        for (int hi = 0; hi < h; ++hi) {
          for (int wi = 0; wi < w; ++wi) {
            int src_idx =
                li * n * c * h * w + ni * c * h * w + ci * h * w + hi * w + wi;
            int dst_idx = li * n * total_channels * h * w +
                          ni * total_channels * h * w + ci * h * w + hi * w +
                          wi;
            padded_data[dst_idx] = input[src_idx];
          }
        }
      }
    }
  }

  // 准备分组存储的结果数据缓冲区
  std::vector<float> result;
  result.reserve(l * n * h * w * total_channels);

  // 按对齐分组提取数据
  for (int g = 0; g < num_groups; ++g) {
    int channel_start = g * align_channels;

    // 按LNHWC顺序进行转换
    for (int li = 0; li < l; ++li) {
      for (int ni = 0; ni < n; ++ni) {
        for (int hi = 0; hi < h; ++hi) {
          for (int wi = 0; wi < w; ++wi) {
            for (int cg = 0; cg < align_channels; ++cg) {
              int channel = channel_start + cg;
              int idx = li * n * total_channels * h * w +
                        ni * total_channels * h * w + channel * h * w + hi * w +
                        wi;
              result.push_back(padded_data[idx]);
            }
          }
        }
      }
    }
  }

  return result;
}

/**
 * 5维LNHWC到LNCHW转换（从分组存储格式）
 * @param input LNHWC格式的输入数据（分组存储格式）
 * @param l 长度/序列长度
 * @param n 批次数
 * @param h 高度
 * @param w 宽度
 * @param c 原始通道数
 * @param align_channels 对齐的通道数
 * @return LNCHW格式的输出数据
 */
std::vector<float> convert_lnhwc_to_lnchw_5d(const std::vector<float>& input,
                                             int l, int n, int h, int w, int c,
                                             int align_channels) {
  int padding_channels = (align_channels - c % align_channels) % align_channels;
  int total_channels = c + padding_channels;
  int num_groups = total_channels / align_channels;

  if (input.size() != l * n * h * w * total_channels) {
    throw std::invalid_argument("输入数据大小与指定维度不匹配");
  }

  // 从分组存储格式重构为补零LNCHW格式
  std::vector<float> padded_data(l * n * total_channels * h * w, 0.0f);

  int input_idx = 0;
  for (int g = 0; g < num_groups; ++g) {
    int channel_start = g * align_channels;

    // 按LNHWC顺序处理输入数据
    for (int li = 0; li < l; ++li) {
      for (int ni = 0; ni < n; ++ni) {
        for (int hi = 0; hi < h; ++hi) {
          for (int wi = 0; wi < w; ++wi) {
            for (int cg = 0; cg < align_channels; ++cg) {
              int channel = channel_start + cg;
              int lnchw_idx = li * n * total_channels * h * w +
                              ni * total_channels * h * w + channel * h * w +
                              hi * w + wi;
              padded_data[lnchw_idx] = input[input_idx++];
            }
          }
        }
      }
    }
  }

  // 提取原始通道数据
  std::vector<float> output(l * n * c * h * w);
  for (int li = 0; li < l; ++li) {
    for (int ni = 0; ni < n; ++ni) {
      for (int ci = 0; ci < c; ++ci) {
        for (int hi = 0; hi < h; ++hi) {
          for (int wi = 0; wi < w; ++wi) {
            int src_idx = li * n * total_channels * h * w +
                          ni * total_channels * h * w + ci * h * w + hi * w +
                          wi;
            int dst_idx =
                li * n * c * h * w + ni * c * h * w + ci * h * w + hi * w + wi;
            output[dst_idx] = padded_data[src_idx];
          }
        }
      }
    }
  }

  return output;
}

/**
 * 将数据直接保存到单个文件
 * @param data 要保存的数据
 * @param filename 输出文件名
 */
void save_data_to_file(const std::vector<float>& data,
                       const std::string& filename) {
  std::ofstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("无法打开文件: " + filename);
  }

  file << std::fixed << std::setprecision(6);

  for (const auto& value : data) {
    file << value << "\n";
  }

  file.close();
  std::cout << "数据已保存到 " << filename << std::endl;
}

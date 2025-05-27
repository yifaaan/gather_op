#pragma once

#include <string>
#include <vector>

/**
 * 3维CHW到HWC转换，按指定通道数对齐
 * @param input CHW格式的输入数据
 * @param c 通道数
 * @param h 高度
 * @param w 宽度
 * @param align_channels 对齐的通道数（默认64）
 * @return HWC格式的输出数据
 */
std::vector<float> convert_chw_to_hwc_3d(const std::vector<float>& input, int c,
                                         int h, int w, int align_channels = 64);

/**
 * 3维HWC到CHW转换
 * @param input HWC格式的输入数据
 * @param c 原始通道数（不包括padding）
 * @param h 高度
 * @param w 宽度
 * @param align_channels 对齐的通道数（默认64）
 * @return CHW格式的输出数据
 */
std::vector<float> convert_hwc_to_chw_3d(const std::vector<float>& input, int c,
                                         int h, int w, int align_channels = 64);

/**
 * 4维NCHW到NHWC转换，按指定通道数对齐
 * @param input NCHW格式的输入数据
 * @param n 批次数
 * @param c 通道数
 * @param h 高度
 * @param w 宽度
 * @param align_channels 对齐的通道数（默认64）
 * @return NHWC格式的输出数据
 */
std::vector<float> convert_nchw_to_nhwc_4d(const std::vector<float>& input,
                                           int n, int c, int h, int w,
                                           int align_channels = 64);

/**
 * 4维NHWC到NCHW转换
 * @param input NHWC格式的输入数据
 * @param n 批次数
 * @param c 原始通道数
 * @param h 高度
 * @param w 宽度
 * @param align_channels 对齐的通道数（默认64）
 * @return NCHW格式的输出数据
 */
std::vector<float> convert_nhwc_to_nchw_4d(const std::vector<float>& input,
                                           int n, int c, int h, int w,
                                           int align_channels = 64);

/**
 * 5维LNCHW到LNHWC转换，按指定通道数对齐
 * @param input LNCHW格式的输入数据
 * @param l 长度/序列长度
 * @param n 批次数
 * @param h 高度
 * @param w 宽度
 * @param c 通道数
 * @param align_channels 对齐的通道数（默认64）
 * @return LNHWC格式的输出数据
 */
std::vector<float> convert_lnchw_to_lnhwc_5d(const std::vector<float>& input,
                                             int l, int n, int h, int w, int c,
                                             int align_channels = 64);

/**
 * 5维LNHWC到LNCHW转换
 * @param input LNHWC格式的输入数据
 * @param l 长度/序列长度
 * @param n 批次数
 * @param h 高度
 * @param w 宽度
 * @param c 原始通道数
 * @param align_channels 对齐的通道数（默认64）
 * @return LNCHW格式的输出数据
 */
std::vector<float> convert_lnhwc_to_lnchw_5d(const std::vector<float>& input,
                                             int l, int n, int h, int w, int c,
                                             int align_channels = 64);

/**
 * 将数据直接保存到单个文件
 * @param data 要保存的数据
 * @param filename 输出文件名
 */
void save_data_to_file(const std::vector<float>& data,
                       const std::string& filename);

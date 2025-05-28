#include "cv.h"
#include "gather_chw.h"
#include "op.h"

std::vector<float> TestGatherCHW(bool is_rvv, std::vector<int> in_shape,
                                 std::vector<int> indices_shape,
                                 const char *input_path,
                                 const char *indices_path,
                                 const char *output_path, int axis,
                                 int align_channels) {
  int input_size = 1;
  for (auto i : in_shape) {
    input_size *= i;
  }
  int num_indices = 1;
  for (auto i : indices_shape) {
    num_indices *= i;
  }

  struct timeval start, end;

  int *indices_data = readFileINT(indices_path, num_indices);
  std::vector<int> indices{indices_data, indices_data + num_indices};

  std::vector<float> output;

  float *input_data_ptr = readFile(input_path, input_size);
  std::vector<float> input_data{input_data_ptr, input_data_ptr + input_size};

  gettimeofday(&start, NULL);
  // 将channlast 转换成chw形式
  if (in_shape.size() == 3) {
    input_data = convert_hwc_to_chw_3d(input_data, in_shape[0], in_shape[1],
                                       in_shape[2], align_channels);
  } else if (in_shape.size() == 4) {
    input_data =
        convert_nhwc_to_nchw_4d(input_data, in_shape[0], in_shape[1],
                                in_shape[2], in_shape[3], align_channels);

  } else if (in_shape.size() == 5) {
    input_data = convert_ndhwc_to_ncdhw_5d(input_data, in_shape[0], in_shape[1],
                                           in_shape[2], in_shape[3],
                                           in_shape[4], align_channels);
  }
  gettimeofday(&end, NULL);
  float convert_time_use = ((end.tv_sec - start.tv_sec) * 1000000.0 +
                            (end.tv_usec - start.tv_usec)) /
                           1000.0 / 1.0;

  gettimeofday(&start, NULL);

  mem::gather_chw(output, input_data, in_shape, indices, axis);
  gettimeofday(&end, NULL);
  float calculate_time_use = ((end.tv_sec - start.tv_sec) * 1000000.0 +
                              (end.tv_usec - start.tv_usec)) /
                             1000.0 / 1.0;
  // 将chw结果转换成channelast
  gettimeofday(&start, NULL);
  if (in_shape.size() == 3) {
    output = convert_chw_to_hwc_3d(output, 128, 128, 128, align_channels);
  } else if (in_shape.size() == 4) {
    // input: 128, 128, 128, 5
    // axis = 0,2
    output = convert_nhwc_to_nchw_4d(output, 128, 128, 128, 5, align_channels);
  } else if (in_shape.size() == 5) {
    // input: 128, 128, 128, 5, 5
    // axis = 4,3
    output =
        convert_ncdhw_to_ndhwc_5d(output, 5, 128, 3, 128, 128, align_channels);
  }
  gettimeofday(&end, NULL);
  float convert_time_use2 = ((end.tv_sec - start.tv_sec) * 1000000.0 +
                             (end.tv_usec - start.tv_usec)) /
                            1000.0 / 1.0;

  outputFile_line(output_path, output);

  float all_time = convert_time_use + convert_time_use2 + calculate_time_use;

  if (in_shape.size() == 3) {
    printf("input{%2d, %2d, %2d},axis{%d},channel_%2d,CLtoCHW %.3f ms,CHWtoCL "
           "%.3f "
           "ms,calculate %7.3f ms,all_time %.3f ms\n",
           in_shape[0], in_shape[1], in_shape[2], axis, align_channels,
           convert_time_use, convert_time_use2, calculate_time_use, all_time);
  } else if (in_shape.size() == 4) {
    printf("input{%2d, %2d, %2d, %2d},axis{%d},channel_%2d,CLtoNCHW %.3f "
           "ms,NCHWtoCL "
           "%.3f ms,calculate %7.3f ms,all_time %.3f ms\n",
           in_shape[0], in_shape[1], in_shape[2], in_shape[3], axis,
           align_channels, convert_time_use, convert_time_use2,
           calculate_time_use, all_time);
  } else if (in_shape.size() == 5) {
    printf("input{%2d, %2d, %2d, %2d, %2d},axis{%d},channel_%2d,CLtoNCDHW %.3f "
           "ms,NCDHWtoCL "
           "%.3f ms,calculate %7.3f ms,all_time %.3f ms\n",
           in_shape[0], in_shape[1], in_shape[2], in_shape[3], in_shape[4],
           axis, align_channels, convert_time_use, convert_time_use2,
           calculate_time_use, all_time);
  }
  return {convert_time_use, calculate_time_use, convert_time_use2, all_time};
};

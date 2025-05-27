#include "gather_hwc.h"
#include "op.h"

float TestGatherHWC(bool is_rvv, std::vector<int> in_shape,
                    std::vector<int> indices_shape, const char *input_path,
                    const char *indices_path, const char *output_path,
                    int axis) {
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
  if (is_rvv) {
    rvv::gather_hwc(output, input_data, in_shape, indices, axis, 16);
  } else {
    mem::gather_hwc(output, input_data, in_shape, indices, axis, 16);
  }

  gettimeofday(&end, NULL);
  float calculate_time_use = ((end.tv_sec - start.tv_sec) * 1000000.0 +
                              (end.tv_usec - start.tv_usec)) /
                             1000.0 / 1.0;

  outputFile_line(output_path, output);

  float all_time = calculate_time_use;

  if (in_shape.size() == 3) {
    printf("input{%2d, %2d, %2d},axis{%d},channel_%2d,calculate %7.3f "
           "ms,all_time %.3f ms\n",
           in_shape[0], in_shape[1], in_shape[2], axis, 16, calculate_time_use,
           all_time);
  } else if (in_shape.size() == 4) {
    printf("input{%2d, %2d, %2d, %2d},axis{%d},channel_%2d,calculate %7.3f "
           "ms,all_time %.3f ms\n",
           in_shape[0], in_shape[1], in_shape[2], in_shape[3], axis, 16,
           calculate_time_use, all_time);
  } else if (in_shape.size() == 5) {
    printf("input{%2d, %2d, %2d, %2d, %2d},axis{%d},channel_%2d,calculate "
           "%7.3f ms,all_time %.3f ms\n",
           in_shape[0], in_shape[1], in_shape[2], in_shape[3], in_shape[4],
           axis, 16, calculate_time_use, all_time);
  }
  return all_time;
};

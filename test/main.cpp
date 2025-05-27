#include "cv.h"

#include "gather_hwc.h"
#include "op.h"

// 辅助函数：比较两个向量是否相等
bool compare_vectors(const std::vector<float> &a, const std::vector<float> &b,
                     float epsilon = 1e-6) {
  if (a.size() != b.size()) {
    std::cout << "向量大小不匹配: " << a.size() << " vs " << b.size()
              << std::endl;
    return false;
  }

  float max_diff = 0.0f;
  int diff_count = 0;

  for (size_t i = 0; i < a.size(); ++i) {
    float diff = std::abs(a[i] - b[i]);
    if (diff > epsilon) {
      diff_count++;
      if (diff_count <= 5) { // 只显示前5个差异
        std::cout << "索引 " << i << ": " << a[i] << " vs " << b[i]
                  << " (差值: " << diff << ")" << std::endl;
      }
    }
    max_diff = std::max(max_diff, diff);
  }

  if (diff_count > 0) {
    std::cout << "总共发现 " << diff_count << " 个差异，最大差值: " << max_diff
              << std::endl;
    return false;
  }

  std::cout << "✓ 数据完全匹配！最大差值: " << max_diff << std::endl;
  return true;
}

void TestGather5d(bool is_rvv, std::vector<int> in_shape,
                  std::vector<int> indices_shape, const char *input_path,
                  const char *indices_path, const char *output_path, int axis) {
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
  // if (is_rvv) {
  //   rvv::gather_chw(output, input, in_shape, indices, axis);
  // } else {
  //   mem::gather_chw(output, input, in_shape, indices, axis);
  // }
  mem::gather_hwc(output, input_data, in_shape, indices, axis, 64);

  outputFile_line(output_path, output);
};

int main() {

  auto chw_data_ptr = readFile("128_128_128.txt", 128 * 128 * 128);
  auto chw_data =
      std::vector<float>{chw_data_ptr, chw_data_ptr + 128 * 128 * 128};
  auto hwc_data = convert_chw_to_hwc_3d(chw_data, 128, 128, 128, 16);
  outputFile_line("128_128_128_hwc.txt", hwc_data);

  {
    std::vector<float> time(4, 0);
    for (int i = 0; i < 3; i++) {
      auto times =
          TestGatherCHW(false, {128, 128, 128}, {128}, "128_128_128_hwc.txt",
                        "indices_128_3d.txt", "output.txt", 0);
      time[0] += times[0];
      time[1] += times[1];
      time[2] += times[2];
      time[3] += times[3];
    }
    time[0] /= 3;
    time[1] /= 3;
    time[2] /= 3;
    time[3] /= 3;
    std::cout << "128_128_128_hwc.txt,axis0,CLtoCHW " << time[0]
              << " ms,CHWtoCL " << time[1] << " ms,calculate " << time[2]
              << " ms,all_time " << time[3] << " ms" << std::endl;
  }

  {
    std::vector<float> time(4, 0);
    for (int i = 0; i < 3; i++) {
      auto times =
          TestGatherCHW(false, {128, 128, 128}, {128}, "128_128_128_hwc.txt",
                        "indices_128_3d.txt", "output.txt", 1);
      time[0] += times[0];
      time[1] += times[1];
      time[2] += times[2];
      time[3] += times[3];
    }
    time[0] /= 3;
    time[1] /= 3;
    time[2] /= 3;
    time[3] /= 3;
    std::cout << "128_128_128_hwc.txt,axis1,CLtoCHW " << time[0]
              << " ms,CHWtoCL " << time[1] << " ms,calculate " << time[2]
              << " ms,all_time " << time[3] << " ms" << std::endl;
  }
  {
    std::vector<float> time(4, 0);
    for (int i = 0; i < 3; i++) {
      auto times =
          TestGatherCHW(false, {128, 128, 128}, {128}, "128_128_128_hwc.txt",
                        "indices_128_3d.txt", "output.txt", 2);
      time[0] += times[0];
      time[1] += times[1];
      time[2] += times[2];
      time[3] += times[3];
    }
    time[0] /= 3;
    time[1] /= 3;
    time[2] /= 3;
    time[3] /= 3;
    std::cout << "128_128_128_hwc.txt,axis2,CLtoCHW " << time[0]
              << " ms,CHWtoCL " << time[1] << " ms,calculate " << time[2]
              << " ms,all_time " << time[3] << " ms" << std::endl;
  }

  chw_data_ptr = readFile("128_128_128_5.txt", 128 * 5 * 128 * 128);
  chw_data =
      std::vector<float>{chw_data_ptr, chw_data_ptr + 128 * 5 * 128 * 128};
  hwc_data = convert_nchw_to_nhwc_4d(chw_data, 128, 128, 128, 5, 16);

  outputFile_line("128_128_128_5_hwc.txt", hwc_data);
  {
    std::vector<float> time(4, 0);
    for (int i = 0; i < 3; i++) {
      auto times = TestGatherCHW(false, {128, 128, 128, 5}, {128},
                                 "128_128_128_5_hwc.txt", "indices_128_4d.txt",
                                 "output.txt", 0);
      time[0] += times[0];
      time[1] += times[1];
      time[2] += times[2];
      time[3] += times[3];
    }
    time[0] /= 3;
    time[1] /= 3;
    time[2] /= 3;
    time[3] /= 3;
    std::cout << "128_128_128_5_hwc.txt,axis0,CLtoCHW " << time[0]
              << " ms,CHWtoCL " << time[1] << " ms,calculate " << time[2]
              << " ms,all_time " << time[3] << " ms" << std::endl;
  }
  {
    std::vector<float> time(4, 0);
    for (int i = 0; i < 3; i++) {
      auto times = TestGatherCHW(false, {128, 128, 128, 5}, {128},
                                 "128_128_128_5_hwc.txt", "indices_128_4d.txt",
                                 "output.txt", 2);
      time[0] += times[0];
      time[1] += times[1];
      time[2] += times[2];
      time[3] += times[3];
    }
    time[0] /= 3;
    time[1] /= 3;
    time[2] /= 3;
    time[3] /= 3;
    std::cout << "128_128_128_5_hwc.txt,axis2,CLtoCHW " << time[0]
              << " ms,CHWtoCL " << time[1] << " ms,calculate " << time[2]
              << " ms,all_time " << time[3] << " ms" << std::endl;
  }
  chw_data_ptr = readFile("5_128_3_128_128.txt", 128 * 5 * 3 * 128 * 128);
  chw_data =
      std::vector<float>{chw_data_ptr, chw_data_ptr + 128 * 5 * 3 * 128 * 128};
  hwc_data = convert_ncdhw_to_ndhwc_5d(chw_data, 5, 128, 3, 128, 128, 16);
  outputFile_line("5_128_3_128_128_hwc.txt", hwc_data);

  {
    std::vector<float> time(4, 0);
    for (int i = 0; i < 3; i++) {
      auto times = TestGatherCHW(false, {5, 128, 3, 128, 128}, {128},
                                 "5_128_3_128_128_hwc.txt",
                                 "indices_128_5d.txt", "output.txt", 3);
      time[0] += times[0];
      time[1] += times[1];
      time[2] += times[2];
      time[3] += times[3];
    }
    time[0] /= 3;
    time[1] /= 3;
    time[2] /= 3;
    time[3] /= 3;
    std::cout << "5_128_3_128_128_hwc.txt,axis3,CLtoCHW " << time[0]
              << " ms,CHWtoCL " << time[1] << " ms,calculate " << time[2]
              << " ms,all_time " << time[3] << " ms" << std::endl;
  }
  {
    std::vector<float> time(4, 0);
    for (int i = 0; i < 3; i++) {
      auto times = TestGatherCHW(false, {5, 128, 3, 128, 128}, {128},
                                 "5_128_3_128_128_hwc.txt",
                                 "indices_128_5d.txt", "output.txt", 4);
      time[0] += times[0];
      time[1] += times[1];
      time[2] += times[2];
      time[3] += times[3];
    }
    time[0] /= 3;
    time[1] /= 3;
    time[2] /= 3;
    time[3] /= 3;
    std::cout << "5_128_3_128_128_hwc.txt,axis4,CLtoCHW " << time[0]
              << " ms,CHWtoCL " << time[1] << " ms,calculate " << time[2]
              << " ms,all_time " << time[3] << " ms" << std::endl;
  }

  /*test 5d mem*/
  // {
  //   auto data_ptr =
  //       readFile("./tmp/1_2_147_4_8/input_data.txt", 1 * 2 * 147 * 4 * 8);
  //   auto hwc = convert_ncdhw_to_ndhwc_5d(
  //       std::vector<float>{data_ptr, data_ptr + 1 * 2 * 147 * 4 * 8}, 1, 2,
  //       147, 4, 8, 64);

  //   outputFile_line("./tmp/1_2_147_4_8/ncdhw_to_ndhwc_result.txt", hwc);

  //   //
  //   TestGather5d(false,
  //                {
  //                    1,
  //                    147,
  //                    4,
  //                    8,
  //                    64,
  //                },
  //                {3, 5}, "./tmp/1_2_147_4_8/ncdhw_to_ndhwc_result.txt",
  //                "./tmp/1_2_147_4_8/indices.txt",
  //                "./tmp/1_2_147_4_8/test_5d_map_output.txt", 4);
  //   auto true_output_ptr =
  //       readFile("./tmp/1_2_147_4_8/output_data.txt", 1 * 2 * 147 * 4 * 15);
  //   hwc = convert_ncdhw_to_ndhwc_5d(
  //       std::vector<float>{true_output_ptr,
  //                          true_output_ptr + 1 * 2 * 147 * 4 * 15},
  //       1, 2, 147, 4, 15, 64);
  //   outputFile_line("./tmp/1_2_147_4_8/true_output.txt", hwc);
  // }

  // {
  //   auto data_ptr =
  //       readFile("./tmp/1_12_200_2_8/input_data.txt", 1 * 12 * 200 * 2 * 8);
  //   auto hwc = convert_ncdhw_to_ndhwc_5d(
  //       std::vector<float>{data_ptr, data_ptr + 1 * 12 * 200 * 2 * 8}, 1, 12,
  //       200, 2, 8, 64);

  //   outputFile_line("./tmp/1_12_200_2_8/ncdhw_to_ndhwc_result.txt", hwc);

  //   //
  //   TestGather5d(false,
  //                {
  //                    1,
  //                    200,
  //                    2,
  //                    8,
  //                    64,
  //                },
  //                {3, 2}, "./tmp/1_12_200_2_8/ncdhw_to_ndhwc_result.txt",
  //                "./tmp/1_12_200_2_8/indices.txt",
  //                "./tmp/1_12_200_2_8/test_5d_map_output.txt", 4);
  //   auto true_output_ptr =
  //       readFile("./tmp/1_12_200_2_8/output_data.txt", 1 * 12 * 200 * 2 * 6);
  //   hwc = convert_ncdhw_to_ndhwc_5d(
  //       std::vector<float>{true_output_ptr,
  //                          true_output_ptr + 1 * 12 * 200 * 2 * 6},
  //       1, 12, 200, 2, 6, 64);
  //   outputFile_line("./tmp/1_12_200_2_8/true_output.txt", hwc);
  // }
}
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cv.h"

float* readFile(const char* path, int len) {
  FILE* fp = fopen(path, "r");
  if (!fp) {
    printf("无法打开文件: %s\n", path);
    return nullptr;
  }

  float* dataBuf = (float*)malloc(len * sizeof(float));
  if (!dataBuf) {
    fclose(fp);  // 内存分配失败时关闭文件
    return nullptr;
  }

  for (int i = 0; i < len; i++) {
    if (fscanf(fp, "%f", &dataBuf[i]) != 1) {
      free(dataBuf);
      fclose(fp);  // 读取失败时关闭文件
      return nullptr;
    }
  }

  fclose(fp);  // 正确关闭文件
  return dataBuf;
}

// 辅助函数：比较两个向量是否相等
bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b,
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
      if (diff_count <= 5) {  // 只显示前5个差异
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

// 显示数据的前几个元素
void show_data_sample(const std::vector<float>& data, const std::string& name,
                      int count = 8) {
  std::cout << name << " (前" << count << "个元素): ";
  for (int i = 0; i < std::min(count, static_cast<int>(data.size())); ++i) {
    std::cout << std::fixed << std::setprecision(1) << data[i];
    if (i < count - 1 && i < static_cast<int>(data.size()) - 1)
      std::cout << ", ";
  }
  if (data.size() > count) std::cout << "...";
  std::cout << std::endl;
}

int main() {
  std::cout << "=== CV转换函数演示与验证 ===" << std::endl;

  try {
    // {
    //   // === 3维CHW到HWC转换及逆向验证 ===
    //   std::cout << "\n--- 3维CHW <-> HWC 转换验证 ---" << std::endl;

    //   // 创建示例CHW数据：3通道，4x4图像
    //   // std::vector<float> original_chw;
    //   // int c = 3, h = 4, w = 4;
    //   int c = 200, h = 32, w = 2;

    //   // // 生成测试数据
    //   // for (int ch = 0; ch < c; ++ch) {
    //   //   for (int y = 0; y < h; ++y) {
    //   //     for (int x = 0; x < w; ++x) {
    //   //       original_chw.push_back(ch * 100 + y * 10 + x);
    //   //     }
    //   //   }
    //   // }

    //   auto data_ptr = readFile("input_data_chw.txt", 200 * 32 * 2);
    //   std::vector<float> original_chw(data_ptr, data_ptr + 200 * 32 * 2);

    //   std::cout << "原始CHW数据大小: " << original_chw.size() << std::endl;
    //   std::cout << "数据维度: " << c << "x" << h << "x" << w << std::endl;
    //   show_data_sample(original_chw, "原始CHW数据");

    //   // 正向转换：CHW -> HWC
    //   auto hwc_data = convert_chw_to_hwc_3d(original_chw, c, h, w, 64);
    //   std::cout << "\n转换后HWC数据大小: " << hwc_data.size() << std::endl;
    //   show_data_sample(hwc_data, "转换后HWC数据");

    //   // 保存HWC结果
    //   save_data_to_file(hwc_data, "chw_to_hwc_result.txt");

    //   {
    //     auto data_ptr = readFile("true_chw_hwc.txt", 256 * 32 * 2);
    //     std::vector<float> py_trans_result{data_ptr, data_ptr + 256 * 32 *
    //     2}; std::cout << "与py的结果对比==\n"; if
    //     (compare_vectors(py_trans_result, hwc_data)) {
    //       std::cout << "✓ 3维CHW <-> HWC 转换验证通过！" << std::endl;
    //     } else {
    //       std::cout << "✗ 3维CHW <-> HWC 转换验证失败！" << std::endl;
    //     }
    //   }

    //   // 逆向转换：HWC -> CHW
    //   std::cout << "\n执行逆向转换 HWC -> CHW..." << std::endl;
    //   auto restored_chw = convert_hwc_to_chw_3d(hwc_data, c, h, w, 64);
    //   std::cout << "逆向转换后CHW数据大小: " << restored_chw.size()
    //             << std::endl;
    //   show_data_sample(restored_chw, "还原后CHW数据");

    //   // 保存还原结果
    //   save_data_to_file(restored_chw, "hwc_to_chw_restored.txt");

    //   // 数据对比验证
    //   std::cout << "\n=== 3维转换精度验证 ===" << std::endl;
    //   if (compare_vectors(original_chw, restored_chw)) {
    //     std::cout << "✓ 3维CHW <-> HWC 转换验证通过！" << std::endl;
    //   } else {
    //     std::cout << "✗ 3维CHW <-> HWC 转换验证失败！" << std::endl;
    //   }
    // }

    {
      // === 4维NCHW到NHWC转换及逆向验证 ===
      std::cout << "\n--- 4维NCHW <-> NHWC 转换验证 ---" << std::endl;

      // 创建示例NCHW数据：2批次，3通道，2x2图像
      int n = 1;
      int c = 147;
      int h = 8;
      int w = 8;

      // 生成测试数据
      auto data_ptr = readFile("input_data_nchw.txt", 147 * 8 * 8);
      std::vector<float> original_nchw{data_ptr, data_ptr + 147 * 8 * 8};

      std::cout << "原始NCHW数据大小: " << original_nchw.size() << std::endl;
      std::cout << "数据维度: " << n << "x" << c << "x" << h << "x" << w
                << std::endl;
      show_data_sample(original_nchw, "原始NCHW数据");

      // 正向转换：NCHW -> NHWC
      auto nhwc_data = convert_nchw_to_nhwc_4d(original_nchw, n, c, h, w, 16);
      std::cout << "\n转换后NHWC数据大小: " << nhwc_data.size() << std::endl;
      show_data_sample(nhwc_data, "转换后NHWC数据");

      // 保存NHWC结果
      save_data_to_file(nhwc_data, "nchw_to_nhwc_result.txt");

      {
        auto data_ptr = readFile("true_nchw_nhwc.txt", 160 * 8 * 8);
        std::vector<float> py_trans_result{data_ptr, data_ptr + 160 * 8 * 8};
        std::cout << "与py的结果对比==\n";
        if (compare_vectors(py_trans_result, nhwc_data)) {
          std::cout << "✓ 4维NCHW <-> NHWC 转换验证通过！" << std::endl;
        } else {
          std::cout << "✗ 4维NCHW <-> NHWC 转换验证失败！" << std::endl;
        }
      }

      // 逆向转换：NHWC -> NCHW
      std::cout << "\n执行逆向转换 NHWC -> NCHW..." << std::endl;
      auto restored_nchw = convert_nhwc_to_nchw_4d(nhwc_data, n, c, h, w, 16);
      std::cout << "逆向转换后NCHW数据大小: " << restored_nchw.size()
                << std::endl;
      show_data_sample(restored_nchw, "还原后NCHW数据");

      // 保存还原结果
      save_data_to_file(restored_nchw, "nhwc_to_nchw_restored.txt");

      // 数据对比验证
      std::cout << "\n=== 4维转换精度验证 ===" << std::endl;
      if (compare_vectors(original_nchw, restored_nchw)) {
        std::cout << "✓ 4维NCHW <-> NHWC 转换验证通过！" << std::endl;
      } else {
        std::cout << "✗ 4维NCHW <-> NHWC 转换验证失败！" << std::endl;
      }
    }
    {
      // === 5维LNCHW到LNHWC转换及逆向验证 ===
      std::cout << "\n--- 5维NCDHW <-> NDHWC 转换验证 ---" << std::endl;

      int n = 1;
      int c = 12;
      int d = 200;
      int h = 2;
      int w = 8;

      // 生成测试数据

      auto data_ptr = readFile("input_data_ncdhw.txt", 12 * 200 * 2 * 8);
      std::vector<float> original_lnchw{data_ptr, data_ptr + 12 * 200 * 2 * 8};
      std::cout << "原始NCDHW数据大小: " << original_lnchw.size() << std::endl;
      std::cout << "数据维度: " << n << "x" << c << "x" << d << "x" << h << "x"
                << w << std::endl;
      show_data_sample(original_lnchw, "原始NCDHW数据");

      // 正向转换：LNCHW -> LNHWC
      auto lnhwc_data =
          convert_ncdhw_to_ndhwc_5d(original_lnchw, n, c, d, h, w, 64);
      std::cout << "\n转换后NDHWC数据大小: " << lnhwc_data.size() << std::endl;
      show_data_sample(lnhwc_data, "转换后NDHWC数据");

      // 保存LNHWC结果
      save_data_to_file(lnhwc_data, "ncdhw_to_ndhwc_result.txt");

      {
        auto data_ptr = readFile("true_ncdhw_ndhwc.txt", 64 * 200 * 2 * 8);
        std::vector<float> py_trans_result{data_ptr,
                                           data_ptr + 64 * 200 * 2 * 8};
        std::cout << "与py的结果对比==\n";
        if (compare_vectors(py_trans_result, lnhwc_data)) {
          std::cout << "✓ 5维NCDHW <-> NDHWC 转换验证通过！" << std::endl;
        } else {
          std::cout << "✗ 5维NCDHW <-> NDHWC 转换验证失败！" << std::endl;
        }
      }

      // 逆向转换：LNHWC -> LNCHW
      std::cout << "\n执行逆向转换 LNHWC -> LNCHW..." << std::endl;
      auto restored_lnchw =
          convert_ndhwc_to_ncdhw_5d(lnhwc_data, n, c, d, h, w, 64);
      std::cout << "逆向转换后LNCHW数据大小: " << restored_lnchw.size()
                << std::endl;
      show_data_sample(restored_lnchw, "还原后LNCHW数据");

      // 保存还原结果
      save_data_to_file(restored_lnchw, "lnhwc_to_lnchw_restored.txt");

      // 数据对比验证
      std::cout << "\n=== 5维转换精度验证 ===" << std::endl;
      if (compare_vectors(original_lnchw, restored_lnchw)) {
        std::cout << "✓ 5维LNCHW <-> LNHWC 转换验证通过！" << std::endl;
      } else {
        std::cout << "✗ 5维LNCHW <-> LNHWC 转换验证失败！" << std::endl;
      }
    }

    // === 总结 ===
    std::cout << "\n=== 所有转换完成！===" << std::endl;
    std::cout << "生成的文件:" << std::endl;
    std::cout << "- chw_to_hwc_result.txt: CHW到HWC转换结果" << std::endl;
    std::cout << "- hwc_to_chw_restored.txt: HWC还原回CHW的结果" << std::endl;
    std::cout << "- nchw_to_nhwc_result.txt: NCHW到NHWC转换结果" << std::endl;
    std::cout << "- nhwc_to_nchw_restored.txt: NHWC还原回NCHW的结果"
              << std::endl;
    std::cout << "- lnchw_to_lnhwc_result.txt: LNCHW到LNHWC转换结果"
              << std::endl;
    std::cout << "- lnhwc_to_lnchw_restored.txt: LNHWC还原回LNCHW的结果"
              << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
# CV转换函数库

本库实现了仿照Python版本的多维数据格式转换功能，支持3、4、5维数据在不同内存布局之间的转换，并提供通道对齐功能。

## 功能概述

### 支持的转换类型

1. **3维转换**
   - CHW ↔ HWC (通道-高度-宽度 ↔ 高度-宽度-通道)

2. **4维转换**
   - NCHW ↔ NHWC (批次-通道-高度-宽度 ↔ 批次-高度-宽度-通道)

3. **5维转换**
   - NCDHW ↔ NDHWC (批次-通道-深度-高度-宽度 ↔ 批次-深度-高度-宽度-通道)

### 主要特性

- **通道对齐**: 支持按指定通道数（默认64）进行内存对齐
- **零填充**: 自动在通道维度进行零填充以满足对齐要求
- **双向转换**: 每种格式都支持正向和逆向转换
- **文件操作**: 支持按组保存数据和合并文件功能

## 文件结构

- `cv.h` - 头文件，包含所有函数声明
- `cv.cpp` - 实现文件，包含所有转换函数的实现
- `test_cv.cpp` - 测试程序，验证转换函数的正确性
- `py.py` - 原始Python实现（参考）

## API 参考

### 3维转换函数

```cpp
// CHW -> HWC 转换
std::vector<float> convert_chw_to_hwc_3d(
    const std::vector<float>& input, 
    int c, int h, int w, 
    int align_channels = 64
);

// HWC -> CHW 转换
std::vector<float> convert_hwc_to_chw_3d(
    const std::vector<float>& input,
    int c, int h, int w, 
    int align_channels = 64
);
```

### 4维转换函数

```cpp
// NCHW -> NHWC 转换
std::vector<float> convert_nchw_to_nhwc_4d(
    const std::vector<float>& input,
    int n, int c, int h, int w, 
    int align_channels = 64
);

// NHWC -> NCHW 转换
std::vector<float> convert_nhwc_to_nchw_4d(
    const std::vector<float>& input,
    int n, int c, int h, int w, 
    int align_channels = 64
);
```

### 5维转换函数

```cpp
// NCDHW -> NDHWC 转换
std::vector<float> convert_ncdhw_to_ndhwc_5d(
    const std::vector<float>& input,
    int n, int c, int d, int h, int w, 
    int align_channels = 64
);

// NDHWC -> NCDHW 转换
std::vector<float> convert_ndhwc_to_ncdhw_5d(
    const std::vector<float>& input,
    int n, int c, int d, int h, int w, 
    int align_channels = 64
);
```

### 文件操作函数

```cpp
// 直接保存数据到单个文件
void save_data_to_file(
    const std::vector<float>& data, 
    const std::string& filename
);

// 按组保存数据到文件（仿照Python版本）
void save_data_by_groups(
    const std::vector<float>& data, 
    const std::string& filename_prefix, 
    int total_size, 
    int align_channels = 64
);

// 合并分组文件
void merge_group_files(
    const std::string& filename_prefix, 
    int num_groups, 
    const std::string& output_filename
);
```

## 使用示例

### 基本使用

```cpp
#include "cv.h"
#include <vector>

int main() {
    // 3维CHW数据：3通道，2x2图像
    std::vector<float> chw_data = {
        1.0f, 2.0f, 3.0f, 4.0f,    // 通道0
        5.0f, 6.0f, 7.0f, 8.0f,    // 通道1  
        9.0f, 10.0f, 11.0f, 12.0f  // 通道2
    };
    
    // CHW -> HWC 转换
    auto hwc_data = convert_chw_to_hwc_3d(chw_data, 3, 2, 2, 64);
    
    // 直接保存转换结果
    save_data_to_file(hwc_data, "output.txt");
    
    // HWC -> CHW 还原
    auto restored = convert_hwc_to_chw_3d(hwc_data, 3, 2, 2, 64);
    
    return 0;
}
```

### 编译和运行

```bash
# 编译测试程序
g++ -o test_cv cv.cpp test_cv.cpp -std=c++11

# 运行测试
./test_cv
```

## 内存布局说明

### CHW vs HWC

- **CHW**: 通道优先，内存中先存储所有通道0的数据，再存储通道1，以此类推
- **HWC**: 通道最后，内存中按像素位置存储，每个像素位置包含所有通道的数据

### 通道对齐

当原始通道数不是对齐数的倍数时，会在通道维度末尾添加零填充：
- 原始通道数：3
- 对齐通道数：64  
- 填充后通道数：64（添加61个零通道）

## 性能特点

- **内存效率**: 使用std::vector进行内存管理，自动处理内存分配
- **类型安全**: 使用C++模板和强类型检查
- **错误处理**: 输入验证和异常处理
- **可扩展性**: 易于添加新的维度转换支持

## 与Python版本的对应关系

本C++实现完全对应py.py中的Python实现：

| Python功能 | C++对应函数 |
|------------|-------------|
| `convert_to_channel_last()` | `convert_chw_to_hwc_3d()` |
| `merge_group_files()` | `merge_group_files()` |
| 按64通道对齐存储 | `save_data_by_groups()` |

## 测试验证

运行测试程序会验证：
1. 所有转换的正确性（通过往返转换验证）
2. 不同维度的转换功能
3. 文件操作功能
4. 边界条件处理

测试通过表明实现与预期行为一致。 
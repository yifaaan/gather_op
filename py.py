import numpy as np
import os

def convert_to_channel_last(data, c, h, w):
    """
    将 CHW 格式的数据转换为 HWC 格式，并按 64 通道对齐存储。
    :param data: 输入数据，形状为 (c, h, w)
    :param c: 通道数
    :param h: 高度
    :param w: 宽度
    :return: 无返回值，直接存储文件
    """
    # 检查输入数据的形状是否匹配
    if data.shape != (c, h, w):
        raise ValueError(f"输入数据的形状 {data.shape} 与指定的 (c, h, w) = ({c}, {h}, {w}) 不匹配")

    # 计算需要补零的通道数
    padding_channels = (64 - c % 64) % 64

    # 在通道维度上补零
    padded_data = np.pad(data, ((0, padding_channels), (0, 0), (0, 0)), mode='constant')

    # 将通道维度移动到最后一个维度
    channel_last_data = np.transpose(padded_data, (1, 2, 0))

    # 按 64 通道对齐存储
    num_groups = (c + padding_channels) // 64  # 计算总组数
    group_files = []  # 用于存储每组文件的路径
    for group in range(num_groups):
        # 提取当前 64 通道的数据
        start = group * 64
        end = start + 64
        group_data = channel_last_data[:, :, start:end]

        # 将数据展平并存储到 txt 文件中，每行一个数
        output_path = f"output_group_{group + 1}.txt"
        np.savetxt(output_path, group_data.flatten(), fmt="%.6f")
        group_files.append(output_path)  # 记录文件路径
        print(f"第 {group + 1} 组数据已保存到 {output_path}")

    return group_files

def merge_group_files(group_files, output_path):
    """
    合并所有分组文件为一个文件。
    :param group_files: 分组文件路径列表
    :param output_path: 合并后的文件路径
    """
    merged_data = []
    for file in group_files:
        # 读取每个文件的数据
        data = np.loadtxt(file)
        merged_data.extend(data)  # 将数据添加到合并列表中

    # 将合并后的数据保存到输出文件
    np.savetxt(output_path, merged_data, fmt="%.6f")
    print(f"所有分组文件已合并到 {output_path}")

def main(data_path, c, h, w, output_path):
    """
    主函数：读取数据并处理。
    :param data_path: 输入数据路径
    :param c: 通道数
    :param h: 高度
    :param w: 宽度
    :param output_path: 最终合并文件的路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"输入文件 {data_path} 不存在")

    # 读取数据
    data = np.loadtxt(data_path)  # 加载一维数据

    # 将一维数据重塑为 (c, h, w) 的三维数组
    if data.size == c * h * w:  # 检查数据总量是否匹配
        data = data.reshape(c, h, w)
    else:
        raise ValueError(f"数据总量 {data.size} 与指定的 (c * h * w) = {c * h * w} 不匹配")

    # 转换数据并按 64 通道对齐存储
    group_files = convert_to_channel_last(data, c, h, w)

    # 合并所有分组文件
    merge_group_files(group_files, output_path)

if __name__ == "__main__":
    # 示例参数
    data_path = 'data.txt'  # 输入数据路径
    c = 256  # 通道数
    h = 256  # 高度
    w = 256  # 宽度
    output_path = "data_channelast.txt"  # 最终合并文件的路径

    # 调用主函数
    main(data_path, c, h, w, output_path)
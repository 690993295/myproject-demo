import os

# 指定文件夹路径
# folder_path = 'data/vali/wuyedijin'
folder_path = 'dataplus/train/wuyedijin'

# 获取文件夹内所有文件的列表
file_list = os.listdir(folder_path)

# 遍历文件列表，按序号重命名文件
for i, filename in enumerate(file_list):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        new_filename = f'wu{i+1}.jpg'  # 新的文件名
        src = os.path.join(folder_path, filename)  # 原始文件路径
        dst = os.path.join(folder_path, new_filename)  # 新文件路径
        os.rename(src, dst)  # 执行重命名操作

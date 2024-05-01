import os
from os import getcwd

# 定义植物类别列表
classes = ['begonia', 'daisy', 'dandelion', 'magnolia', 'pine', 'rose', 'sunflower', 'willow', 'wuyedijin']
# 定义要处理的数据集集合目录列表
sets = ['data/train']

# 主程序入口
if __name__ == '__main__':
    # 获取当前工作目录
    wd = getcwd()

    # 遍历数据集集合目录列表
    for se in sets:
        # 根据数据集集合名称创建输出文件名，例如：'cls_train.txt'
        list_file = open('cls_' + se.split('/')[-1] + '.txt', 'w')

        # 获取数据集集合的绝对路径
        datasets_path = se
        # 列出该集合目录下的所有子目录（即各类植物）
        types_name = os.listdir(datasets_path)

        # 遍历每个植物子目录
        for type_name in types_name:
            # 如果该子目录名不在植物类别列表中，则跳过本次循环
            if type_name not in classes:
                continue
            # 根据植物子目录名在花卉类别列表中找到对应的类别索引（ID）
            cls_id = classes.index(type_name)
            
            # 构建当前植物子目录的绝对路径
            photos_path = os.path.join(datasets_path, type_name)
            # 列出该植物子目录下的所有文件
            photos_name = os.listdir(photos_path)

            # 遍历当前植物子目录下的所有文件
            for photo_name in photos_name:
                # 分离文件名与扩展名
                _, postfix = os.path.splitext(photo_name)
                # 只保留扩展名为 .jpg、.png、.jpeg 的文件
                if postfix not in ['.jpg', '.png', '.jpeg']:
                    continue
                # 将类别ID和符合要求的图像文件的绝对路径写入到输出文件中，每个条目以分号分隔
                list_file.write(str(cls_id) + ';' + '%s/%s' % (wd, os.path.join(photos_path, photo_name)))
                # 在每个条目末尾添加换行符
                list_file.write('\n')

        # 关闭已写入完毕的输出文件
        list_file.close()

# import os
# from os import getcwd

# classes=['begonia','daisy','dandelion','magnolia','pine','rose','sunflower','willow','wuyedijin']
# sets=['data/train']

# if __name__=='__main__':
#     wd=getcwd()
#     for se in sets:
#         # list_file=open('cls1_'+ se +'.txt','w')
#         list_file = open('cls_' + se.split('/')[-1] + '.txt', 'w')
#         datasets_path=se
#         types_name=os.listdir(datasets_path)#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
#         for type_name in types_name:
#             if type_name not in classes:
#                 continue
#             cls_id=classes.index(type_name)#输出0-1
#             photos_path=os.path.join(datasets_path,type_name)
#             photos_name=os.listdir(photos_path)
#             for photo_name in photos_name:
#                 _,postfix=os.path.splitext(photo_name)#该函数用于分离文件名与拓展名
#                 if postfix not in['.jpg','.png','.jpeg']:
#                     continue
#                 list_file.write(str(cls_id)+';'+'%s/%s'%(wd, os.path.join(photos_path,photo_name)))
#                 list_file.write('\n')
#         list_file.close()



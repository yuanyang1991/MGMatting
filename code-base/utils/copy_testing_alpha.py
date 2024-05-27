import shutil
import os

alpha_path = "../data/Test_set/Adobe_licensed_images/alpha/"
alpha_copy_path = "../data/Test_set/Adobe_licensed_images/alpha_copy/"


def copy_file_multiple_times(file_path, dest_folder, num_copies=20):
    # 获取源文件名和扩展名
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)

    # 确保目标文件夹存在
    os.makedirs(dest_folder, exist_ok=True)

    for i in range(0, num_copies):
        # 生成新的文件名
        new_file_name = f"{name}_{i}{ext}"
        new_file_path = os.path.join(dest_folder, new_file_name)

        # 复制文件
        shutil.copyfile(file_path, new_file_path)
        print(f"Copied to: {new_file_path}")


alphas = os.listdir(alpha_path)
for path in alphas:
    copy_file_multiple_times(os.path.join(alpha_path, path), alpha_copy_path)

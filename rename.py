import os


def prepend_tryon_to_filenames(directory):
    # 確保提供的路徑存在且為目錄
    if not os.path.isdir(directory):
        print(f"提供的路徑 '{directory}' 不是一個目錄或不存在。")
        return

    for filename in os.listdir(directory):
        # 組合完整的檔案路徑
        old_file_path = os.path.join(directory, filename)

        # 檢查是否為檔案，排除目錄
        if os.path.isfile(old_file_path):
            # 新的檔案名稱
            new_filename = "cloth_" + filename
            new_file_path = os.path.join(directory, new_filename)

            # 重新命名檔案
            os.rename(old_file_path, new_file_path)
            print(f"已將檔案 '{filename}' 重新命名為 '{new_filename}'")


# 請替換 'your_directory_path' 為您的目錄路徑
directory_path = '/home/kalijason/train_images/cloth'
prepend_tryon_to_filenames(directory_path)

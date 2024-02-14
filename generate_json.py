import os
import json

# 定義要掃描的目錄
directory = "/home/kalijason/train_images/tryons/"

# 讀取目錄中的所有圖像文件
image_files = [f for f in os.listdir(directory) if f.endswith('.jpg') and f.startswith('tryon')]

# 創建一個包含圖像文件名和固定文本的字典列表
json_list = [
    {"tryon_image_file": f,
     "cloth_image_file": f.replace("tryon","cloth"),
     "conditioning_image_file": f.replace("tryon","pose"),
     "text": "A cloth"} for f in image_files]

# 將列表轉換成 JSON 格式
json_data = json.dumps(json_list, indent=4)

# 儲存 JSON 數據到一個檔案
json_file_path = '/home/kalijason/git/IP-Adapter/tryons_images.json'
with open(json_file_path, 'w') as file:
    file.write(json_data)
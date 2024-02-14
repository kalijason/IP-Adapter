
### tryons without controlnet, initiated from pretrained ip adapter
```
accelerate launch --mixed_precision "fp16" \
  train_tryons.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --pretrained_ip_adapter_path="/home/kalijason/git/IP-Adapter/models/ip-adapter_sd15.bin" \
  --image_encoder_path="/home/kalijason/git/IP-Adapter/models/image_encoder" \
  --data_json_file="/home/kalijason/git/IP-Adapter/tryons_images.json" \
  --data_root_path="/home/kalijason/train_images/tryons" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/home/kalijason/output/model/current/tryons" \
  --logging_dir="logs" \
  --save_steps=10000
```

### tryons with controlnet, initiated from pretrained ip adapter
```
accelerate launch --mixed_precision "fp16" \
  train_tryons_controlnet.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --pretrained_ip_adapter_path="/home/kalijason/git/IP-Adapter/models/ip-adapter_sd15.bin" \
  --controlnet_model_name_or_path="lllyasviel/sd-controlnet-openpose" \
  --image_encoder_path="/home/kalijason/git/IP-Adapter/models/image_encoder" \
  --data_json_file="/home/kalijason/git/IP-Adapter/tryons_images.json" \
  --data_root_path="/home/kalijason/train_images/tryons" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/home/kalijason/output/model/current/tryons_ctl_v10" \
  --logging_dir="logs" \
  --save_steps=10000

```
### tryons with controlnet, initiated from pretrained ip adapter, 200 epoch
```
accelerate launch --mixed_precision "fp16" \
  train_tryons_controlnet.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --pretrained_ip_adapter_path="/home/kalijason/git/IP-Adapter/models/ip-adapter_sd15.bin" \
  --controlnet_model_name_or_path="lllyasviel/control_v11p_sd15_openpose" \
  --image_encoder_path="/home/kalijason/git/IP-Adapter/models/image_encoder" \
  --data_json_file="/home/kalijason/git/IP-Adapter/tryons_images.json" \
  --data_root_path="/home/kalijason/train_images/tryons" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --num_train_epochs=200 \
  --output_dir="/home/kalijason/output/model/current/tryons_ctlv11i" \
  --logging_dir="logs" \
  --save_steps=20000
```
### tryons with controlnet
```
accelerate launch --mixed_precision "fp16" \
  train_tryons_controlnet.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --controlnet_model_name_or_path="lllyasviel/control_v11p_sd15_openpose" \
  --image_encoder_path="/home/kalijason/git/IP-Adapter/models/image_encoder" \
  --data_json_file="/home/kalijason/git/IP-Adapter/tryons_images.json" \
  --data_root_path="/home/kalijason/train_images/tryons" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --num_train_epochs=200 \
  --output_dir="/home/kalijason/output/model/current/tryons_ctlv11n" \
  --logging_dir="logs" \
  --save_steps=20000
```
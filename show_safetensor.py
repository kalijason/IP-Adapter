from safetensors import safe_open

tensors = {}
print("original")
with safe_open("/home/kalijason/git/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/models/ip-adapter_sd15.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        print(k)

print("new")
with safe_open("/home/kalijason/git/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/models/ip_adapter_tryon.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        print(k)

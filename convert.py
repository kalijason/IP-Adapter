from safetensors.torch import save_file, load_file
import torch


def convert_pickle_from_path(source_path, target_path):
    sd = torch.load(source_path, map_location="cpu")
    combined_sd = extract_sd(sd)
    torch.save(combined_sd, target_path)


def extract_sd(sd):
    image_proj_sd = {}
    ip_sd = {}

    for k in sd:
        if k.startswith("unet"):
            pass
        elif k.startswith("image_proj_model"):
            image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
        elif k.startswith("adapter_modules"):
            ip_sd[k.replace("adapter_modules.", "")] = sd[k]

    combined_sd = {'image_proj': image_proj_sd, 'ip_adapter': ip_sd}
    return combined_sd


def convert_pickle_from_name_steps(name, steps):
    source_path = f"/home/kalijason/output/model/current/{name}/checkpoint-{steps}/pytorch_model.bin"
    target_path = f"/home/kalijason/git/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/models/ip_adapter_{name}_{steps}.bin"
    convert_pickle_from_path(source_path, target_path)


def main():
    # convert_pickle_from_path('/home/kalijason/output/model/current/tryons/initial_checkpoint/pytorch_model.bin', '/home/kalijason/git/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/models/ip_adapter_initial.bin')

    # convert_pickle_from_name_steps('tryons',10000)
    # convert_pickle_from_name_steps('tryons',50000)
    # convert_pickle_from_name_steps('tryons',100000)
    # convert_pickle_from_name_steps('tryons',140000)

    convert_pickle_from_name_steps('tryons_ctlv11i', 10000)
    convert_pickle_from_name_steps('tryons_ctlv11i', 50000)
    convert_pickle_from_name_steps('tryons_ctlv11i', 100000)
    convert_pickle_from_name_steps('tryons_ctlv11i', 140000)


if __name__ == "__main__":
    main()

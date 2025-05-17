import os
import cv2
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import glob
from trak.projectors import ProjectionType, CudaProjector
import numpy as np
from src.utils import detect, segment,  remove_special_chars
from GroundingDINO.groundingdino.util.inference import load_image, load_model
from segment_anything import SamPredictor, build_sam
from GroundingDINO.groundingdino.util import box_ops
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"

model_detect = load_model("D:/SilentBadDiffusion-master/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "D:/SilentBadDiffusion-master/checkpoints/groundingdino_swinT_ogc.pth")
model_detect.to(device)
model_detect.eval()
model_detect.half()

sam_checkpoint = 'D:/SilentBadDiffusion-master/checkpoints/sam_vit_h_4b8939.pth'
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
sam_predictor.model.to(device)

tokenlist = []
with open(r"D:\SilentBadDiffusion-master\datasets\Pokemon\5.txt", 'r') as file:
   data = file.read()
tokenlist = data.split('\t')

poison_image_path = r"D:\SilentBadDiffusion-master\datasets\Pokemon\5p.png"
segment_result_dir = r"D:\SilentBadDiffusion-master\journey_TRAK\examples\seg_result_5"

preprocess = transforms.Compose([
        transforms.Resize((512, 512)),                  # 调整图像大小
        transforms.ToTensor(),                          # 转换为 Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

detector_model = torch.jit.load(
        os.path.join("D:\SilentBadDiffusion-master/", "checkpoints/sscd_disc_mixup.torchscript.pt")).to(device)

def draw_mask(mask, image, random_color=False):
   if random_color:
      color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
   else:
      color = np.array([0 / 255, 0 / 255, 0 / 255, 1])
   h, w = mask.shape[-2:]
   mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

   annotated_frame_pil = Image.fromarray(image).convert("RGBA")
   mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

   return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def process_inverted_mask(inverted_mask_list, check_area=True):
   _inverted_mask_list = []
   # 1.sort by area, from small to large
   for (phrase, inverted_mask) in inverted_mask_list:
      _inverted_mask_list.append((phrase, inverted_mask, (inverted_mask == 0).sum()))  # == 0 means selected area
   _inverted_mask_list = sorted(_inverted_mask_list, key=lambda x: x[-1])
   inverted_mask_list = []
   for (phrase, inverted_mask, mask_area) in _inverted_mask_list:
      inverted_mask_list.append((phrase, inverted_mask))

   phrase_area_dict_before_process = defaultdict(float)
   for phrase, output_grid in inverted_mask_list:
      phrase_area_dict_before_process[phrase] += (output_grid == 0).sum()

   # 2.remove overlapped area
   processed_mask_list = inverted_mask_list.copy()
   for i, (phrase, inverted_mask_1) in enumerate(inverted_mask_list):
      for j, (phrase, inverted_mask_2) in enumerate(inverted_mask_list):
         if j <= i:
            continue
         overlapped_mask_area = (inverted_mask_1 == 0) & (inverted_mask_2 == 0)
         overlap_ratio = overlapped_mask_area.sum() / (inverted_mask_1 == 0).sum()

         processed_mask_list[j][1][overlapped_mask_area] = 255
   returned_processed_mask_list = []
   for i, (phrase, inverted_mask) in enumerate(processed_mask_list):
      blur_mask = cv2.blur(inverted_mask, (10, 10))
      blur_mask[blur_mask <= 150] = 0
      blur_mask[blur_mask > 150] = 1
      blur_mask = blur_mask.astype(np.uint8)
      blur_mask = 1 - blur_mask
      if check_area:
         assert (blur_mask == 0).sum() > (blur_mask > 0).sum()  # selected area (> 0) smaller than not selected (=0)
      if (blur_mask > 0).sum() < 15:
         continue
      # 2.select some large connected component
      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blur_mask, connectivity=4)
      if len(stats) > 1:
         stats = stats[1:]
         output_grid = None
         area_list = sorted([_stat[cv2.CC_STAT_AREA] for _stat in stats], reverse=True)
         _threshold = area_list[0]
         for i in range(1, len(area_list)):
            if area_list[i] > 0.15 * _threshold:
               _threshold = area_list[i]

         for _i, _stat in enumerate(stats):
            if _stat[cv2.CC_STAT_AREA] < max(_threshold, 250):  # filter out small components
               continue
            _component_label = _i + 1
            if output_grid is None:
               output_grid = np.where(labels == _component_label, 1, 0)
            else:
               output_grid = output_grid + np.where(labels == _component_label, 1, 0)
      else:
         continue

      if output_grid is None:
         continue

      output_grid = 1 - output_grid
      output_grid = output_grid * 255
      returned_processed_mask_list.append((phrase, output_grid.astype(np.uint8)))

   # filter out small area
   phrase_area_dict = defaultdict(float)
   _phrase_area_dict = defaultdict(float)
   for phrase, output_grid in returned_processed_mask_list:
      phrase_area_dict[phrase] += (output_grid == 0).sum() / phrase_area_dict_before_process[
         phrase]  # (output_grid.shape[0] * output_grid.shape[1]
      _phrase_area_dict[phrase] += (output_grid == 0).sum() / (output_grid.shape[0] * output_grid.shape[1])
   #print(phrase_area_dict.items())
   #print(_phrase_area_dict.items())
   # return returned_processed_mask_list

   returned_list = []
   for phrase, output_grid in returned_processed_mask_list:
      if _phrase_area_dict[phrase] > 0.004 and phrase_area_dict[phrase] > 0.05:
         returned_list.append([phrase, output_grid])

   return returned_list



def seg_data(image_path, tokenlist, model_detect, sam_predictor, segment_result_dir, filter_out_large_box=False):
   image_source, image_transformed = load_image(image_path)
   image_transformed = image_transformed.half()
   inverted_mask_list = []
   for_segmentation_data = []
   merge_mask_list = []
   for phrase in tokenlist:
      print(phrase)
      img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
      # 1. detect
      annotated_frame, detected_boxes, logit = detect(image_transformed, image_source, text_prompt=phrase,
                                                      model=model_detect)
      if len(detected_boxes) == 0:
         continue
      #Image.fromarray(annotated_frame).save(cache_dir + '/detect_{}.png'.format(img_name_prefix))

      # 2. remove box with too large size
      H, W, _ = image_source.shape
      boxes_xyxy = box_ops.box_cxcywh_to_xyxy(detected_boxes) * torch.Tensor([W, H, W, H])
      area_ratio = ((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])) / (H * W)
      _select_idx = torch.ones_like(area_ratio)

      if not filter_out_large_box:  # directly add all boxes
         for _i in range(len(boxes_xyxy)):
            for_segmentation_data.append((phrase, boxes_xyxy[_i].unsqueeze(0), logit[_i].item()))
      else:  # add part of boxes
         if len(area_ratio) > 1 and (area_ratio < 0.5).any():
            _select_idx[area_ratio > 0.5] = 0
            _select_idx = _select_idx > 0
            boxes_xyxy = boxes_xyxy[_select_idx]
            for _i in range(len(boxes_xyxy)):
               for_segmentation_data.append((phrase, boxes_xyxy[_i].unsqueeze(0)))
               #segment_location.append((phrase, boxes_xyxy[_i].unsqueeze(0)))
         else:
            _select_idx = torch.argmin(area_ratio)
            boxes_xyxy = boxes_xyxy[_select_idx].unsqueeze(0)
            for_segmentation_data.append((phrase, boxes_xyxy))

   # 3.segmentation
   for _i, (phrase, boxes_xyxy, detect_score) in enumerate(for_segmentation_data):
      #print(phrase)
      img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
      # 1.2 segment
      segmented_frame_masks = segment(image_source, sam_predictor, boxes_xyxy=boxes_xyxy, multimask_output=False,
                                      check_white=False)
      merged_mask = segmented_frame_masks[0]
      if len(segmented_frame_masks) > 1:
         for _mask in segmented_frame_masks[1:]:
            merged_mask = merged_mask | _mask
      annotated_frame_with_mask = draw_mask(merged_mask, annotated_frame)
      merge_mask_list.append((merged_mask))
      #Image.fromarray(annotated_frame_with_mask).save(cache_dir + '/segment_{}_{}.png'.format(_i, img_name_prefix))
      # 1.3 save masked images
      mask = merged_mask.cpu().numpy()
      inverted_mask = ((1 - mask) * 255).astype(np.uint8)
      inverted_image_mask_pil = Image.fromarray(
         inverted_mask)  # vis mask: Image.fromarray(mask).save(attack_data_directory + '/{}_mask.png'.format(img_name_prefix))
      #inverted_image_mask_pil.save(cache_dir + '/mask_{}_{}.png'.format(_i, img_name_prefix))
      inverted_mask_list.append((phrase, inverted_mask, detect_score))

   # 4.If there exists two inverted_mask cover similar area, then keep the one with higher detect_score
   # sort inverted_mask_list according to inverted_mask_i area
   inverted_mask_list = sorted(inverted_mask_list, key=lambda x: (x[1] == 0).sum())
   area_similar_list = []
   for i, (phrase_i, inverted_mask_i, detect_score_i) in enumerate(inverted_mask_list):
      area_similar_to_i = []
      for j, (phrase_j, inverted_mask_j, detect_score_j) in enumerate(inverted_mask_list):
         overlapped_mask_area = (inverted_mask_i == 0) & (inverted_mask_j == 0)
         overlap_ratio_i = overlapped_mask_area.sum() / (inverted_mask_i == 0).sum()
         overlap_ratio_j = overlapped_mask_area.sum() / (inverted_mask_j == 0).sum()
         if overlap_ratio_i > 0.95 and overlap_ratio_j > 0.95:  # then they cover similar area
            area_similar_to_i.append(j)
      area_similar_list.append(area_similar_to_i)
   # index_set = set(list(range(len(area_similar_list))))
   used_phrase_idx_set = set()
   processed_mask_list = []
   for i, area_similar_to_i in enumerate(area_similar_list):
      phrase_i, inverted_mask_i, detect_score_i = inverted_mask_list[i]
      score_list_i = []
      for j in area_similar_to_i:
         # score_list_i.append(inverted_mask_list[j][-1])
         if j not in used_phrase_idx_set:
            score_list_i.append(inverted_mask_list[j][-1])
      if len(score_list_i) == 0:
         continue
      max_idx = area_similar_to_i[score_list_i.index(max(score_list_i))]
      processed_mask_list.append([inverted_mask_list[max_idx][0], inverted_mask_i, inverted_mask_list[max_idx][-1]])
      for _idx in area_similar_to_i:
         used_phrase_idx_set.add(_idx)
   inverted_mask_list = processed_mask_list

   # 4.merge mask according to phrase
   _inverted_mask_list = []
   for _i, (phrase, inverted_mask, detect_score) in enumerate(inverted_mask_list):
      if len(_inverted_mask_list) == 0 or phrase not in [x[0] for x in _inverted_mask_list]:
         _inverted_mask_list.append([phrase, inverted_mask])
      else:
         _idx = [x[0] for x in _inverted_mask_list].index(phrase)
         _inter_result = _inverted_mask_list[_idx][1] * inverted_mask
         _inter_result[_inter_result > 0] = 255
         _inverted_mask_list[_idx][1] = _inter_result
   inverted_mask_list = _inverted_mask_list

   # 3.post process mask (remove undesired noise) and visualize masked images
   inverted_mask_list = process_inverted_mask(inverted_mask_list, check_area=False)

   # image_source and inverted_mask_list, check the std
   _inverted_mask_list = []
   for _i, (phrase, inverted_mask) in enumerate(inverted_mask_list):
      #print(phrase)

      _mask = np.tile(inverted_mask.reshape(inverted_mask.shape[0], inverted_mask.shape[1], -1), 3)
      _std = image_source[_mask != 255].std()
      #print(_std)
      if _std > 9:
         _inverted_mask_list.append([phrase, inverted_mask])
   inverted_mask_list = _inverted_mask_list

   for _i, (phrase, inverted_mask) in enumerate(inverted_mask_list):
      img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
      tt = torch.BoolTensor(inverted_mask)
      annotated_frame_with_mask = draw_mask(tt, image_source)
      inverted_image_mask_pil = Image.fromarray(annotated_frame_with_mask)
      inverted_image_mask_pil.save(segment_result_dir + '/processed_mask_{}_{}.png'.format(_i, img_name_prefix))

   return merge_mask_list,  H, inverted_mask_list


def compute_gradient_for_image(
    model_path: str,
    prompt: str,
    input_image_path: str,
    target_step: int = 999,
    total_steps: int = 999,
    seed: int = 42
) -> torch.Tensor:

    input_image = Image.open(input_image_path).convert("RGB")
    input_image_tensor = preprocess(input_image).unsqueeze(0).to("cuda").half()  # 添加 batch 维度并移到 GPU


    with torch.no_grad():
        latents = pipe.vae.encode(input_image_tensor).latent_dist.sample() * 0.18215


    generator = torch.Generator("cuda").manual_seed(seed)
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to("cuda"))[0]

    gradients = {}

    def save_grad(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()

        return hook

    handles = []
    for name, module in pipe.unet.named_modules():
        if "conv" in name:
            handles.append(module.register_full_backward_hook(save_grad(name)))

    pipe.scheduler.set_timesteps(total_steps, device="cuda")

    for i, t in tqdm(enumerate(pipe.scheduler.timesteps)):
        # 只在目标步骤启用梯度
        with torch.set_grad_enabled(i == target_step):
            latent_model_input = latents

            # 前向传播
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample

            if i == target_step:

                loss = torch.norm(noise_pred, p=2)
                loss.backward()


        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    for handle in handles:
        handle.remove()

    gradient_tensors = list(gradients.values())
    if not gradient_tensors:
        all_gradients = torch.tensor([], device="cuda")
    else:
        target_dims = [grad.numel() for grad in gradient_tensors]
        max_dim = max(target_dims) if target_dims else 0

        flattened_gradients = []
        for grad in gradient_tensors:
            grad_flat = grad.view(grad.size(0), -1)  # 展平为 [batch_size, grad_dim]
            pad_size = max_dim - grad_flat.size(1)
            if pad_size > 0:
                padding = torch.zeros(grad.size(0), pad_size, device=grad.device, dtype=grad.dtype)
                grad_flat = torch.cat([grad_flat, padding], dim=1)
            flattened_gradients.append(grad_flat)

        all_gradients = torch.cat(flattened_gradients, dim=0)

    print(f"All gradients shape before projection: {all_gradients.shape}")

    grad_dim = all_gradients.numel()
    proj_dim = 1024
    seed = 42
    proj_type = ProjectionType.rademacher
    device = "cuda"
    max_batch_size = 8

    projector = CudaProjector(
        grad_dim=grad_dim,
        proj_dim=proj_dim,
        seed=seed,
        proj_type=proj_type,
        device=device,
        max_batch_size=max_batch_size
    )
    all_gradients = projector.project(all_gradients, model_id=0)

    torch.cuda.empty_cache()
    return all_gradients

def read_prompts(txt_path: str) -> dict:
    prompts = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                index, prompt = parts
                prompts[index] = prompt
    return prompts

def process_image_folder(
    model_path: str,
    txt_path: str,
    folder_path: str,
    target_step: int = 999,
    total_steps: int = 999,
    seed: int = 42
) -> torch.Tensor:
    prompts = read_prompts(txt_path)

    image_paths = sorted(glob.glob(os.path.join(folder_path, "*")))  # 根据需要调整文件匹配模式

    all_gradients_list = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        if not os.path.isfile(img_path):
            continue

        img_name = os.path.basename(img_path)
        parts = os.path.splitext(img_name)[0].split('_')
        if len(parts) == 0:
            index = None
        else:
            index = parts[0]

        if index not in prompts:
            print(f"No prompt found for image {img_path}. Skipping.")
            continue

        prompt = prompts[index]

        try:
            gradients = compute_gradient_for_image(
                model_path=model_path,
                prompt=prompt,
                input_image_path=img_path,
                target_step=target_step,
                total_steps=total_steps,
                seed=seed
            )
            all_gradients_list.append(gradients)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    if all_gradients_list:
        flattened_tensors = [tensor.flatten() for tensor in all_gradients_list]

        all_gradients = torch.stack(flattened_tensors, dim=0)

    else:
        all_gradients = torch.tensor([], device="cuda")



    return all_gradients


def extract_masks(image_folder, target_size=(512, 512)):

    image_files = [f for f in os.listdir(image_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    N = len(image_files)

    if N == 0:
        raise ValueError("Error. No images in this file.")

    masks = np.zeros((N, *target_size), dtype=np.uint8)

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_folder, img_file)

        img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图片: {img_path}")
            masks[idx] = 0
            continue

        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        upper_white = np.array([0, 0, 0])

        mask = cv2.inRange(img_resized, upper_white, upper_white)
        mask_inv = cv2.bitwise_not(mask)
        masks[idx] = mask_inv

    return masks


def apply_mask_to_images(masks, input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    num_masks, mask_height, mask_width = masks.shape

    for filename in tqdm(os.listdir(input_folder), desc="Processing Images"):
        input_path = os.path.join(input_folder, filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        image = cv2.imread(input_path)

        if image is None:
            print(f"No file: {input_path}")
            continue

        if image.shape[:2] != (mask_height, mask_width):
            image = cv2.resize(image, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)

        image_number = filename.split('.')[0]
        image_folder = os.path.splitext(filename)[0]
        image_output_folder = os.path.join(output_folder, image_folder)
        os.makedirs(image_output_folder, exist_ok=True)

        for i in range(num_masks):
            mask = masks[i]

            masked_image = cv2.bitwise_and(image, image, mask=mask)

            output_path = os.path.join(image_output_folder, f"{image_number}_mask_{i}.png")
            cv2.imwrite(output_path, masked_image)

    print(f"所有图像已处理并保存到 {output_folder}")



def l2_normalize(matrix):
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(0)
        is_1d = True
    else:
        is_1d = False

    norms = torch.norm(matrix, p=2, dim=1, keepdim=True)

    norms[norms == 0] = 1

    normalized_matrix = matrix / norms

    if is_1d:
        normalized_matrix = normalized_matrix.squeeze(0)  # 恢复为 (n,)

    return normalized_matrix

def compute_sim(PIL_input_imgs, PIL_tgt_imgs):
    with torch.no_grad():

        if isinstance(PIL_input_imgs, list):
            batch_1 = [preprocess(PIL_img.convert('RGB')).unsqueeze(0) for PIL_img in PIL_input_imgs]
            batch_1 = torch.cat(batch_1, dim=0).to(device)
        else:
            batch_1 = preprocess(PIL_input_imgs).unsqueeze(0).to(device)

        if isinstance(PIL_tgt_imgs, list):
            batch_2 = [preprocess(PIL_tgt_img.convert('RGB')).unsqueeze(0) for PIL_tgt_img in PIL_tgt_imgs]
            batch_2 = torch.cat(batch_2, dim=0).to(device)
        else:
            batch_2 = preprocess(PIL_tgt_imgs).unsqueeze(0).to(device)

        embedding_1 = detector_model(batch_1)
        embedding_2 = detector_model(batch_2)

    embedding_1 = embedding_1 / torch.norm(embedding_1, dim=-1, keepdim=True)
    embedding_2 = embedding_2 / torch.norm(embedding_2, dim=-1, keepdim=True)
    sim_score = torch.mm(embedding_1, embedding_2.T).squeeze()
    return sim_score

# 使用示例
if __name__ == "__main__":
    model_path = r"G:\pokemon\Pokemon_CP-[5]_Shot-1_Factor-2_SpecChar-0_PoisonRatio-0.05_TrainNum-570.0_PoisonNum-30_AuxNum-14_Epochs-150_20241028134138\best_model_2508"
    txt_path = r"D:\SilentBadDiffusion-master\datasets\Pokemon\caption1.txt"
    folder_path = r"D:\SilentBadDiffusion-master\datasets\Pokemon\images1"
    seg_out_path = r"D:\SilentBadDiffusion-master\datasets\Pokemon\images1_seg"
    base_folder_path = r"D:\SilentBadDiffusion-master\datasets\Pokemon\images5_seg"
    target_seg_folder = r"D:\SilentBadDiffusion-master\journey_TRAK\examples\seg_result_5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    pipe.safety_checker = lambda images, clip_input: (images, None)

    proj_dim = 1024
    seed = 42
    proj_type = ProjectionType.rademacher
    device = "cuda"
    max_batch_size = 8

    merge_mask_list, H, inverted_mask_list = seg_data(poison_image_path, tokenlist, model_detect, sam_predictor,
                                                      segment_result_dir, filter_out_large_box=False)
    masks = extract_masks(segment_result_dir)

    np.save("masks.npy", masks)
    all_gradients = process_image_folder(
        model_path=model_path,
        txt_path=txt_path,
        folder_path=folder_path
    )

    grad_dim = all_gradients.numel()

    projector = CudaProjector(
        grad_dim=grad_dim,
        proj_dim=proj_dim,
        seed=seed,
        proj_type=proj_type,
        device=device,
        max_batch_size=max_batch_size
    )
    projected_gradients = projector.project(all_gradients, model_id=0)

    torch.cuda.empty_cache()

    batch_size = 4
    num_batches = (projected_gradients.shape[0] + batch_size - 1) // batch_size
    pseudo_inverse_batches = []

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, projected_gradients.shape[0])
        batch = projected_gradients[start:end]

        xtx = torch.matmul(batch.t(), batch)
        try:
            xtx_inv = torch.inverse(xtx)
        except RuntimeError as e:
            print(f"Failed to invert matrix for batch {i//batch_size +1}. Error: {e}")
            continue
        pseudo_inverse_batch = torch.matmul(batch, xtx_inv)
        pseudo_inverse_batches.append(pseudo_inverse_batch)

    pseudo_inverse_features = torch.cat(pseudo_inverse_batches, dim=0)
    np.save("pseudo_inverse_features_5.npy", pseudo_inverse_features.cpu().numpy())

    apply_mask_to_images(masks, r"D:\SilentBadDiffusion-master\datasets\Pokemon\images1", base_folder_path)

    poison_grad = torch.tensor(np.load('poison_gradient.npy')).to(device)
    pseudo_inverse_features = torch.tensor(np.load('pseudo_inverse_features.npy')).to(device)

    subfolders = [f for f in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, f))]
    for subfolder in subfolders:
        folder_path = os.path.join(base_folder_path, subfolder)
        all_gradients1 = process_image_folder(
            model_path=model_path,
            txt_path=txt_path,
            folder_path=folder_path
        )

        grad_dim = all_gradients1.numel()
        projector = CudaProjector(
            grad_dim=grad_dim,
            proj_dim=proj_dim,
            seed=seed,
            proj_type=proj_type,
            device=device,
            max_batch_size=max_batch_size
        )
        projected_gradients = projector.project(all_gradients1, model_id=0)

        projected_gradients_n = l2_normalize(projected_gradients)
        poison_grad_n = l2_normalize(poison_grad)
        pseudo_inverse_features_n = l2_normalize(pseudo_inverse_features)
        feature_1 = torch.matmul(pseudo_inverse_features_n,projected_gradients.T)
        poison_row_sums = torch.sum(poison_grad_n, dim=1)
        projected_row_sums = l2_normalize(torch.sum(feature_1, dim=0))
        score = l2_normalize(poison_row_sums) * l2_normalize(projected_row_sums)

        score_file_path = os.path.join(folder_path, 'score1.txt')
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                full_path = os.path.join(folder_path, filename)
                parts = filename.split('_')
                if len(parts) >= 3 and parts[1] == "mask":
                    id2 = parts[2].split('.')[0]

                full_path_tar = os.path.join(target_seg_folder, f"{id2}.png")
                img_1 = Image.open(full_path).convert('RGB')
                img_2 = Image.open(full_path_tar).convert('RGB')
                sim_score = compute_sim(img_1, img_2)
                score1 = sim_score * projected_row_sums[int(id2)]
                with open(score_file_path, 'a') as f:
                    f.write(f"{id2}\t{score1}\n")






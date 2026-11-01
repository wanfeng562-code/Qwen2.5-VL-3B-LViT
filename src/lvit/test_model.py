import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.LViT import LViT
from utils import *
import cv2


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    if np.max(tmp_lbl) > 1 or np.max(tmp_3dunet) > 1:
        cls_num = int(max(np.max(tmp_lbl), np.max(tmp_3dunet))) + 1
        dices, ious = [], []
        for c in range(1, cls_num):
            gt = (tmp_lbl == c).astype(np.float32)
            pd = (tmp_3dunet == c).astype(np.float32)
            dice_c = 2 * np.sum(gt * pd) / (np.sum(gt) + np.sum(pd) + 1e-5)
            inter = np.logical_and(gt > 0, pd > 0).sum()
            union = np.logical_or(gt > 0, pd > 0).sum()
            iou_c = inter / (union + 1e-5) if union > 0 else 1.0
            dices.append(dice_c)
            ious.append(iou_c)
        dice_pred = float(np.mean(dices)) if len(dices) > 0 else 1.0
        iou_pred = float(np.mean(ious)) if len(ious) > 0 else 1.0
    else:
        dice_pred = (
            2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
        )
        iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))

    # 使用统一的可视化方式：去除坐标轴和边框
    plt.imshow(predict_save, cmap="gray")
    plt.axis("off")  # 去除坐标轴

    # 设置画布大小正好匹配图像尺寸，避免留白
    fig = plt.gcf()
    fig.set_size_inches(predict_save.shape[1] / 100.0, predict_save.shape[0] / 100.0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    # 保存时去掉边框、空白
    plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()

    return dice_pred, iou_pred


def vis_and_save_heatmap(
    model, input_img, text, img_RGB, labs, vis_save_path, dice_pred, dice_ens
):
    model.eval()

    output = model(input_img.cuda(), text.cuda())
    if output.shape[1] > 1:
        predict_save = torch.argmax(output, dim=1)[0].cpu().data.numpy()
    else:
        pred_class = torch.where(
            output > 0.5, torch.ones_like(output), torch.zeros_like(output)
        )
        predict_save = pred_class[0].cpu().data.numpy()
        predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(
        predict_save, labs, save_path=vis_save_path + "_predict" + model_type + ".jpg"
    )
    return dice_pred_tmp, iou_tmp


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    model_type = config.model_name

    # Map dataset names (both old and new formats)
    dataset_map = {
        "monuseg_2": ("MoNuSeg", 14),
        "MoNuSeg": ("MoNuSeg", 14),
        "qata_cov19_v2_2": ("Covid19", 2113),
        "Covid19": ("Covid19", 2113),
        "mosmeddata": ("MosMedData+", 273),
        "MosMedData+": ("MosMedData+", 273),
    }

    if config.task_name not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {config.task_name}. Available: {list(dataset_map.keys())}"
        )

    dataset_name, test_num = dataset_map[config.task_name]
    model_dir = "./" + dataset_name + "/" + model_type + "/" + test_session + "/models/"
    model_path_new = model_dir + "model_best.pth"
    model_path_old = model_dir + "best_model-" + model_type + ".pth.tar"
    model_path = model_path_new if os.path.exists(model_path_new) else model_path_old

    save_path = config.task_name + "/" + model_type + "/" + test_session + "/"
    vis_path = "./" + config.task_name + "_visualize_test/"
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location="cuda")

    if model_type == "LViT":
        config_vit = config.get_CTranS_config()
        model = LViT(
            config_vit, n_channels=config.n_channels, n_classes=config.n_labels
        )

    elif model_type == "LViT_pretrain":
        config_vit = config.get_CTranS_config()
        model = LViT(
            config_vit, n_channels=config.n_channels, n_classes=config.n_labels
        )

    else:
        raise TypeError("Please enter a valid name for the model type")

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print("Model loaded !")

    # Statistic Params and flops
    try:
        from thop import profile

        print("Statistic Params and flops...")
        input_img_dummy = torch.randn(
            1, config.n_channels, config.img_size, config.img_size
        ).cuda()
        text_dummy = torch.randn(1, 10, 768).cuda()
        macs, params = profile(model, inputs=(input_img_dummy, text_dummy))
        print(f"FLOPs: {macs/1e9:.2f}G, Params: {params/1e6:.2f}M")
    except Exception as e:
        print(f"Error calculating FLOPs/Params: {e}")

    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_dataset + "Test_text.xlsx")
    test_dataset = ImageToImage2D(
        config.test_dataset,
        config.task_name,
        test_text,
        tf_test,
        image_size=config.img_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    with tqdm(
        total=test_num, desc="Test visualize", unit="img", ncols=70, leave=True
    ) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            # print(names)
            test_data, test_label, test_text = (
                sampled_batch["image"],
                sampled_batch["label"],
                sampled_batch["text"],
            )
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap="gray")
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path + str(names) + "_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(
                model,
                input_img,
                test_text,
                None,
                lab,
                vis_path + str(names),
                dice_pred=dice_pred,
                dice_ens=dice_ens,
            )
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)

from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel
from tqdm import tqdm

### harry
import pytorch3d.transforms
from vine_prune.utils.general_utils import read_mask, calculate_iou
from vine_prune.utils.io import read_json
###

def visualize_2d(results_2d, vis_h_2d_keypoint_dir, vis_h_2d_hpe_dir):
    from PIL import Image
    import matplotlib.pyplot as plt
    
    j2d_r = results_2d['j2d.right']
    j2d_l = results_2d['j2d.left']

    v2d_r = results_2d['v2d.right']
    v2d_l = results_2d['v2d.left']

    im_paths = results_2d['im_paths']

    ### harry
    j2d_r_2d = results_2d['j2d_real.right']
    j2d_l_2d = results_2d['j2d_real.left']
    v2d_r_2d = results_2d['v2d_real.right']
    v2d_l_2d = results_2d['v2d_real.left']
    ###

    # print("Visualizing 2D keypoints")
    # for idx in tqdm(range(len(im_paths))):

    #     im_p = im_paths[idx]
    #     out_p = os.path.join(vis_2d_keypoint_dir, os.path.basename(im_p))

    #     im = Image.open(im_p)

    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(im)
    #     plt.scatter(j2d_r[idx, :, 0], j2d_r[idx, :, 1], s=10)
    #     plt.scatter(j2d_l[idx, :, 0], j2d_l[idx, :, 1], s=10)
    #     plt.legend(['jts_r', 'jts_l'])
    #     plt.savefig(out_p)
    #     plt.close()

    ### harry
    print("Visualizing 2D Hamer keypoints")
    for idx in tqdm(range(len(im_paths))):

        im_p = im_paths[idx]
        out_p = os.path.join(vis_h_2d_keypoint_dir, os.path.basename(im_p))

        im = Image.open(im_p)

        plt.figure(figsize=(10, 10))
        plt.imshow(im)
        plt.scatter(j2d_r_2d[idx, :, 0], j2d_r_2d[idx, :, 1], s=10)
        plt.scatter(j2d_l_2d[idx, :, 0], j2d_l_2d[idx, :, 1], s=10)
        plt.legend(['jts_r', 'jts_l'])
        plt.savefig(out_p)
        plt.close()
    ###

    # print("Visualizing 2D vertices")
    # for idx in tqdm(range(len(im_paths))):

    #     im_p = im_paths[idx]
    #     out_p = os.path.join(vis_2d_hpe_dir, os.path.basename(im_p))

    #     im = Image.open(im_p)

    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(im)
    #     plt.scatter(v2d_r[idx, :, 0], v2d_r[idx, :, 1], s=1)
    #     plt.scatter(v2d_l[idx, :, 0], v2d_l[idx, :, 1], s=1)
    #     plt.legend(['mano_r', 'mano_l'])
    #     plt.savefig(out_p)
    #     plt.close()

    ### Harry
    print("Visualizing 2D Hamer vertices")
    for idx in tqdm(range(len(im_paths))):

        im_p = im_paths[idx]
        out_p = os.path.join(vis_h_2d_hpe_dir, os.path.basename(im_p))

        im = Image.open(im_p)

        plt.figure(figsize=(10, 10))
        plt.imshow(im)
        plt.scatter(v2d_r_2d[idx, :, 0], v2d_r_2d[idx, :, 1], s=1)
        plt.scatter(v2d_l_2d[idx, :, 0], v2d_l_2d[idx, :, 1], s=1)
        plt.legend(['mano_r', 'mano_l'])
        plt.savefig(out_p)
        plt.close()
    ###

def to_xy_batch(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 3
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 2, device=x_homo.device)
    zz = x_homo[:, :, 2:3]
    
    
    x = x_homo[:, :, :2] / zz
    return x


def project2d_batch(K, pts_cam):
    """
    K: (B, 3, 3)
    pts_cam: (B, N, 3)
    """

    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[2] == 3
    assert len(pts_cam.shape) == 3
    pts2d_homo = torch.bmm(K, pts_cam.permute(0, 2, 1)).permute(0, 2, 1)
    pts2d = to_xy_batch(pts2d_homo)
    return pts2d


def reform_pred_list(pred_list, K):
    im_paths = sorted(list(set([pred_dict['img_path'] for pred_dict in pred_list])))

    verts_r = np.zeros((len(im_paths), 778, 3))*np.nan
    verts_l = np.copy(verts_r)
    
    joints_r = np.zeros((len(im_paths), 21, 3))*np.nan
    joints_l = np.copy(joints_r)

    ### harry
    rot_r = np.zeros((len(im_paths), 3))*np.nan
    trans_r = np.zeros((len(im_paths), 3))*np.nan
    pose_r = np.zeros((len(im_paths), 45))*np.nan
    shape_r = np.zeros((len(im_paths), 10))*np.nan

    rot_l = np.copy(rot_r)
    trans_l = np.copy(trans_r)
    pose_l = np.copy(pose_r)
    shape_l = np.copy(shape_r)

    joints_r_2d = np.zeros((len(im_paths), 21, 2))*np.nan
    joints_l_2d = np.zeros((len(im_paths), 21, 2))*np.nan
    verts_r_2d = np.zeros((len(im_paths), 778, 2))*np.nan
    verts_l_2d = np.zeros((len(im_paths), 778, 2))*np.nan
    ###

    for pred_dict in pred_list:
        if not pred_dict['succ']:
            continue
        is_right = bool(pred_dict['is_right'])

        v3d_cam = pred_dict['verts']  + pred_dict['cam_t.full'][None, :]
        j3d_cam = pred_dict['jts']  + pred_dict['cam_t.full'][None, :]

        ### harry
        rot = pred_dict['global_orient'].reshape(3)
        trans = pred_dict['cam_t.full']
        pose = pred_dict['hand_pose'].reshape(45)
        shape = pred_dict['betas']

        j2d_cam = pred_dict['jts_2d']
        verts_cam = pred_dict['verts_2d']
        ###

        idx = im_paths.index(pred_dict['img_path'])

        if is_right:
            verts_r[idx] = v3d_cam
            joints_r[idx] = j3d_cam

            ### harry
            rot_r[idx] = rot
            trans_r[idx] = trans
            pose_r[idx] = pose
            shape_r[idx] = shape
            joints_r_2d[idx] = j2d_cam
            verts_r_2d[idx] = verts_cam
            ###
        else:
            verts_l[idx] = v3d_cam
            joints_l[idx] = j3d_cam

            ### harry
            rot_l[idx] = rot
            trans_l[idx] = trans
            pose_l[idx] = pose
            shape_l[idx] = shape
            joints_l_2d[idx] = j2d_cam
            verts_l_2d[idx] = verts_cam
            ###

    verts_r = verts_r.astype(np.float32)
    verts_l = verts_l.astype(np.float32)
    joints_r = joints_r.astype(np.float32)
    joints_l = joints_l.astype(np.float32)
    
    ### harry
    joints_r_2d = joints_r_2d.astype(np.float32)
    joints_l_2d = joints_l_2d.astype(np.float32)
    verts_r_2d = verts_r_2d.astype(np.float32)
    verts_l_2d = verts_l_2d.astype(np.float32)
    ###
    
    # K = torch.FloatTensor(pred_list[0]['K'])
    K = torch.FloatTensor(K)
    joints_r = torch.FloatTensor(joints_r)
    joints_l = torch.FloatTensor(joints_l)
    verts_r = torch.FloatTensor(verts_r)
    verts_l = torch.FloatTensor(verts_l)
    
    v2d_r = project2d_batch(K[None, :, :].repeat(verts_r.shape[0], 1, 1), verts_r).numpy()
    v2d_l = project2d_batch(K[None, :, :].repeat(verts_l.shape[0], 1, 1), verts_l).numpy()
    j2d_r = project2d_batch(K[None, :, :].repeat(joints_r.shape[0], 1, 1), joints_r).numpy()
    j2d_l = project2d_batch(K[None, :, :].repeat(joints_l.shape[0], 1, 1), joints_l).numpy()    

    results_3d = {}
    results_3d['v3d.right'] = verts_r.cpu().numpy()
    results_3d['v3d.left'] = verts_l.cpu().numpy()
    results_3d['j3d.right'] = joints_r.cpu().numpy()
    results_3d['j3d.left'] = joints_l.cpu().numpy()
    results_3d['im_paths'] = im_paths
    results_3d['K'] = K #pred_list[0]['K']
    
    results_2d = {}
    results_2d['v2d.right'] = v2d_r
    results_2d['v2d.left'] = v2d_l
    results_2d['j2d.right'] = j2d_r
    results_2d['j2d.left'] = j2d_l
    results_2d['im_paths'] = im_paths

    ### harry
    results_mano = {}
    results_mano['rot.right'] = rot_r
    results_mano['trans.right'] = trans_r
    results_mano['pose.right'] = pose_r
    results_mano['shape.right'] = shape_r
    results_mano['rot.left'] = rot_l
    results_mano['trans.left'] = trans_l
    results_mano['pose.left'] = pose_l
    results_mano['shape.left'] = shape_l

    results_2d['j2d_real.right'] = joints_r_2d
    results_2d['j2d_real.left'] = joints_l_2d
    results_2d['v2d_real.right'] = verts_r_2d
    results_2d['v2d_real.left'] = verts_l_2d
    ### harry

    return results_3d, results_2d, results_mano

import json
from typing import Dict, Optional

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Folder with input images')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--iou_thresh', type=float, default=0.40)
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()
    iou_thresh = args.iou_thresh
    vis = args.vis
    
    data_dir = args.data_dir

    img_folder = os.path.join(data_dir, 'undistorted')
    hand_mask_dir = os.path.join(data_dir, 'masks', 'mask_hand')
    out_folder = os.path.join(data_dir, 'hand_pred')
    vis_folder = os.path.join(out_folder, 'vis')
    K_path = os.path.join(data_dir, 'cam_K.txt')

    # Download and load checkpoints
    # download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        # detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        ### harry
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.1
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(img_folder).glob(end)]
    assert len(img_paths) > 0, f"No images found in {img_folder}"
    img_paths = sorted(img_paths)

    K = np.loadtxt(K_path)
    intrinsics = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    # focal_to_use = np.mean([intrinsics[0], intrinsics[1]])

    # Iterate over all images in folder
    print('Running inference on images')
    pred_list = []
    for _, img_path in enumerate(tqdm(img_paths)):
        # debug_num = int(os.path.basename(img_path).split('.')[0])
        # if debug_num != 13:
        #     continue

        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            #img_cv2,
            img, # harry updating this as what is used in latest commit for demo.py
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections

            # right now only want right hand
            # keyp = left_hand_keyp
            # valid = keyp[:,2] > 0.5
            # if sum(valid) > 3:
            #     bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            #     bboxes.append(bbox)
            #     is_right.append(0)

            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        mask_path = os.path.join(hand_mask_dir, img_fn + '.png')

        mask = read_mask(mask_path) 
        assert np.unique(mask).shape[0] == 2

        mask[mask > 0] = 255
        mask_inds = np.argwhere(mask > 0)
        mask_box = [mask_inds[:, 1].min(), mask_inds[:, 0].min(), mask_inds[:, 1].max(), mask_inds[:, 0].max()]

        if len(bboxes) == 0:
            pred_dict = {}
            pred_dict['succ'] = False
            pred_dict['img_path'] = str(img_path)
            pred_list.append(pred_dict)
            continue
        elif len(bboxes) > 1:
            max_iou = -1
            for box in bboxes:
                iou = calculate_iou(box, mask_box)
                if iou > max_iou:
                    max_iou = iou
                    new_boxes = np.array([box])
            bboxes = new_boxes

        iou_test = calculate_iou(bboxes[0], mask_box)
        if iou_test < iou_thresh:
            pred_dict = {}
            pred_dict['succ'] = False
            pred_dict['img_path'] = str(img_path)
            pred_list.append(pred_dict)
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []

        ### harry
        all_2d_pts = []
        all_2d_verts = []
        ###
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)

            # https://github.com/geopavlakos/hamer/issues/55
            # https://github.com/shubham-goel/4D-Humans/issues/129
            # https://arxiv.org/pdf/2111.07868 Equation 1

            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, 
                                               scaled_focal_length).detach().cpu().numpy()

            #scaled_focal_length = focal_to_use
            # pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, 
            #                                    intrinsics[2], intrinsics[3], scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                # regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                #                         out['pred_cam_t'][n].detach().cpu().numpy(),
                #                         batch['img'][n],
                #                         mesh_base_color=LIGHT_BLUE,
                #                         scene_bg_color=(1, 1, 1),
                #                         )

                # if args.side_view:
                #     side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                #                             out['pred_cam_t'][n].detach().cpu().numpy(),
                #                             white_img,
                #                             mesh_base_color=LIGHT_BLUE,
                #                             scene_bg_color=(1, 1, 1),
                #                             side_view=True)
                #     final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                # else:
                #     final_img = np.concatenate([input_patch, regression_img], axis=1)

                #cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                jts = out['pred_keypoints_3d'][n].detach().cpu().numpy()

                ### harry
                betas = out['pred_mano_params']['betas'][n].detach().cpu().numpy()
                global_orient = pytorch3d.transforms.matrix_to_axis_angle(out['pred_mano_params']['global_orient'])[n].detach().cpu().numpy()
                hand_pose = pytorch3d.transforms.matrix_to_axis_angle(out['pred_mano_params']['hand_pose'])[n].detach().cpu().numpy()

                jts_2d = (out['pred_keypoints_2d']*box_size+box_center)[0].detach().cpu().numpy()
                verts_2d = (out['pred_vertices_2d']*box_size+box_center)[0].detach().cpu().numpy()

                # question for me, will that output the same as projecting pred_cam_t_full?
                # https://github.com/geopavlakos/hamer/issues/20
                # may have the answer
                ###
                
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                jts[:,0] = (2*is_right-1)*jts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                pred_dict = {}
                pred_dict['cam_t.full'] = cam_t
                pred_dict['verts'] = verts
                pred_dict['jts'] = jts
                pred_dict['is_right'] = is_right
                pred_dict['img_path'] = str(img_path)

                ### harry
                pred_dict['betas'] = betas
                pred_dict['global_orient'] = global_orient
                pred_dict['hand_pose'] = hand_pose
                pred_dict['jts_2d'] = jts_2d
                pred_dict['verts_2d'] = verts_2d

                all_2d_pts.append(jts_2d)
                all_2d_verts.append(verts_2d)
                ###

                pred_dict['K'] = K
                pred_dict['succ'] = True
                pred_list.append(pred_dict)

                # # Save all meshes to disk
                # if args.save_mesh:
                #     camera_translation = cam_t.copy()
                #     tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                #     tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                # intrinsics=[scaled_focal_length, scaled_focal_length, img_size[n][0]//2, img_size[n][1]//2],
                intrinsics=[scaled_focal_length, scaled_focal_length, intrinsics[2], intrinsics[3]],
                # focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(vis_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

            ### harry
            # pred_2d_jts_img = np.copy(img_cv2)
            # pts_2_proj = np.vstack(all_2d_pts)
            
            # for pt in pts_2_proj:
            #     cv2.circle(pred_2d_jts_img, (int(pt[0]), int(pt[1])), 1, [0, 0, 255], -1)

            # cv2.imwrite(os.path.join(vis_folder, f'{img_fn}_pts_2d.jpg'), pred_2d_jts_img)
            ###



    import os.path as op
    out_3d_p = op.join(out_folder, 'v3d.npy')
    out_2d_p = op.join(out_folder, 'j2d.npy')

    results_3d, results_2d, results_mano = reform_pred_list(pred_list, K)

    if vis:
        # # vis_2d_keypoint_dir = os.path.join(out_folder, '2d_keypoints')
        # # if not os.path.exists(vis_2d_keypoint_dir):
        # #     os.mkdir(vis_2d_keypoint_dir)

        # ### harry
        vis_h_2d_keypoint_dir = os.path.join(out_folder, '2d_h_keypoints')
        if not os.path.exists(vis_h_2d_keypoint_dir):
            os.mkdir(vis_h_2d_keypoint_dir)

        vis_h_2d_hpe_dir = os.path.join(out_folder, '2d_h_hpe')
        if not os.path.exists(vis_h_2d_hpe_dir):
            os.mkdir(vis_h_2d_hpe_dir)
        # ###

        # # vis_2d_hpe_dir = os.path.join(out_folder, '2d_hpe')
        # # if not os.path.exists(vis_2d_hpe_dir):
        # #     os.mkdir(vis_2d_hpe_dir)

        visualize_2d(results_2d, vis_h_2d_keypoint_dir, vis_h_2d_hpe_dir)

    np.save(out_3d_p, results_3d)
    np.save(out_2d_p, results_2d)
    print(f"Saved 3D results to {out_3d_p}")
    print(f"Saved 2D results to {out_2d_p}")

    ### harry
    out_mano_p = op.join(out_folder, 'mano.npy')
    np.save(out_mano_p, results_mano)
    print(f"Saved mano results to {out_mano_p}")
    ###

if __name__ == '__main__':
    main()

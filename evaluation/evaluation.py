import torch
import numpy as np
import argparse
import pickle
from skel_definition import SKEL_LIM_QID2IDX, SKEL_LIM_BOUNDS, pose_param_names
#change all annotation in English!!!
qids_cfg = {
    'l_knee': [13],
    'r_knee': [6],
    'l_elbow': [42, 43],
    'r_elbow': [32, 33],
}

def eval_rot_delta(poses, tol_deg=5):
    tol_rad = np.deg2rad(tol_deg)

    res = {}
    for part in qids_cfg:
        qids = qids_cfg[part]
        violation_part = poses.new_zeros(poses.shape[0], len(qids))
        for i, qid in enumerate(qids):
            idx = SKEL_LIM_QID2IDX[qid]
            ea = poses[:, qid]
            ea = (ea + np.pi) % (2 * np.pi) - np.pi  # Normalize to (-pi, pi)
            exceed_lb = torch.where(
                    ea < SKEL_LIM_BOUNDS[idx][0] - tol_rad,
                    ea - SKEL_LIM_BOUNDS[idx][0] + tol_rad, 0
                )
            exceed_ub = torch.where(
                    ea > SKEL_LIM_BOUNDS[idx][1] + tol_rad,
                    ea - SKEL_LIM_BOUNDS[idx][1] - tol_rad, 0
                )
            violation_part[:, i] = exceed_lb.abs() + exceed_ub.abs()
        res[part] = violation_part

    return res

def load_skel_poses_from_pkl(pkl_path):
    """
    load skel poses from pkl file
    
    Args:
        pkl_path (str): the path of the pkl file
        
    Returns:
        torch.Tensor: SKEL poses, shape is (T, 46)
    """
    with open(pkl_path, 'rb') as f:
        skel_data = pickle.load(f)
    
    # 检查数据结构
    if 'poses' in skel_data:
        poses = skel_data['poses']
    else:
        raise KeyError(f"'poses' not found in {pkl_path}. Available keys: {list(skel_data.keys())}")
    
    # 转换为torch tensor
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses).float()
    elif not isinstance(poses, torch.Tensor):
        poses = torch.tensor(poses, dtype=torch.float32)
    
    print(f"Loaded SKEL poses shape: {poses.shape}")
    #only print 'scapula_elevation_r',  #27
    print(f"Loaded SKEL poses: {poses[:, 28]}")
    print(f"Expected shape: (T, 46), where T is the number of time steps")
    
    # 验证维度
    if poses.shape[1] != 46:
        raise ValueError(f"SKEL poses should have 46 parameters, but found {poses.shape[1]} parameters")
    
    return poses

def analyze_motion_sequence(skel_poses):
    """
    Analyze the Euler angles and violations of the SKEL motion sequence
    
    Args:
        skel_poses: torch.Tensor, shape = (T, 46), T is the number of time steps
    
    Returns:
        dict: contains the Euler angles and violation information
    """
    T = skel_poses.shape[0]
    
    # check violation
    violations = eval_rot_delta(skel_poses, tol_deg=0)
    
    # detailed violation analysis (for all joints)
    detailed_violations = {}
    for t in range(T):
        frame_violations = {}
        for qid, param_name in enumerate(pose_param_names):
            if qid in SKEL_LIM_QID2IDX:
                idx = SKEL_LIM_QID2IDX[qid]
                angle = skel_poses[t, qid]
                # normalize the angle to (-pi, pi)
                angle_norm = (angle + np.pi) % (2 * np.pi) - np.pi
                
                lower_bound = SKEL_LIM_BOUNDS[idx][0]
                upper_bound = SKEL_LIM_BOUNDS[idx][1]
                
                violation_amount = 0
                if angle_norm < lower_bound:
                    violation_amount = lower_bound - angle_norm
                elif angle_norm > upper_bound:
                    violation_amount = angle_norm - upper_bound
                
                frame_violations[param_name] = {
                    'angle': float(angle_norm),
                    'bounds': [float(lower_bound), float(upper_bound)],
                    'violation': float(violation_amount),
                    'is_violated': violation_amount > 0
                }
        detailed_violations[t] = frame_violations
    
    return {
        'violations': violations,
        'detailed_violations': detailed_violations
    }

# example
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_path', type=str, default='output/merged_0/merged_0_skel.pkl')
    args = parser.parse_args()
    
    # load SKEL poses from pkl file
    skel_poses = load_skel_poses_from_pkl(args.motion_path)
    print(f"Successfully loaded SKEL poses, shape: {skel_poses.shape}")
    
    # analyze motion sequence
    results = analyze_motion_sequence(skel_poses)
    
    # output results
    print("Detected violations:")
    for frame_id, frame_violations in results['detailed_violations'].items():
        violated_joints = [name for name, info in frame_violations.items() if info['is_violated']]
        if violated_joints:
            print(f"Frame {frame_id}: {violated_joints}")
    
    # count violation
    total_violations = sum(
        sum(1 for info in frame_violations.values() if info['is_violated'])
        for frame_violations in results['detailed_violations'].values()
    )
    print(f"Total {total_violations} violations")

if __name__ == '__main__':
    main()
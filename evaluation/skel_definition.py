import math
import torch

skel_joints_name= [
'pelvis', #0
'femur_r', #1
'tibia_r', #2
'talus_r', #3
'calcn_r', #4
'toes_r', #5
'femur_l', #6
'tibia_l', #7
'talus_l', #8
'calcn_l', #9
'toes_l', #10
'lumbar_body', #11
'thorax', #12
'head', #13
'scapula_r', #14
'humerus_r', #15
'ulna_r', #16
'radius_r', #17
'hand_r', #18
'scapula_l', #19
'humerus_l', #20
'ulna_l', #21
'radius_l', #22
'hand_l'] #23

pose_param_names = [
 'pelvis_tilt', #0 
 'pelvis_list', #1 
 'pelvis_rotation', #2 
 'hip_flexion_r', #3 
 'hip_adduction_r', #4 
 'hip_rotation_r', #5 
 'knee_angle_r', #6 
 'ankle_angle_r', #7 
 'subtalar_angle_r', #8 
 'mtp_angle_r', #9 
 'hip_flexion_l', #10
 'hip_adduction_l', #11
 'hip_rotation_l', #12
 'knee_angle_l', #13
 'ankle_angle_l', #14
 'subtalar_angle_l', #15
 'mtp_angle_l', #16
 'lumbar_bending', #17
 'lumbar_extension', #18
 'lumbar_twist', #19
 'thorax_bending', #20
 'thorax_extension', #21
 'thorax_twist', #22
 'head_bending', #23
 'head_extension', #24
 'head_twist', #25
 'scapula_abduction_r', #26
 'scapula_elevation_r', #27
 'scapula_upward_rot_r', #28
 'shoulder_r_x', #29
 'shoulder_r_y', #30
 'shoulder_r_z', #31
 'elbow_flexion_r', #32
 'pro_sup_r', #33
 'wrist_flexion_r', #34
 'wrist_deviation_r', #35
 'scapula_abduction_l', #36
 'scapula_elevation_l', #37
 'scapula_upward_rot_l', #38
 'shoulder_l_x', #39
 'shoulder_l_y', #40
 'shoulder_l_z', #41
 'elbow_flexion_l', #42
 'pro_sup_l', #43
 'wrist_flexion_l', #44
 'wrist_deviation_l', #45
]

pose_limits = {
'scapula_abduction_r' :  [-0.628, 0.628],
'scapula_elevation_r' :  [-0.4, -0.1],
'scapula_upward_rot_r' : [-0.190, 0.319],

'scapula_abduction_l' :  [-0.628, 0.628],
'scapula_elevation_l' :  [-0.1, -0.4],
'scapula_upward_rot_l' : [-0.210, 0.219],  

'elbow_flexion_r' : [0, (3/4)*math.pi],
'pro_sup_r'       : [-3/4*math.pi/2, 3/4*math.pi/2],
'wrist_flexion_r' : [-math.pi/2, math.pi/2],
'wrist_deviation_r' :[-math.pi/4, math.pi/4],

'elbow_flexion_l' : [0, (3/4)*math.pi],
'pro_sup_l'       : [-math.pi/2, math.pi/2],
'wrist_flexion_l' : [-math.pi/2, math.pi/2],
'wrist_deviation_l' :[-math.pi/4, math.pi/4],

'shoulder_r_y' : [-math.pi/2, math.pi/2], 

'lumbar_bending' : [-2/3*math.pi/4, 2/3*math.pi/4],
'lumbar_extension' : [-math.pi/4, math.pi/4],
'lumbar_twist' :  [-math.pi/4, math.pi/4],   

'thorax_bending' :[-math.pi/4, math.pi/4], 
'thorax_extension' :[-math.pi/4, math.pi/4], 
'thorax_twist' :[-math.pi/4, math.pi/4],

'head_bending' :[-math.pi/4, math.pi/4], 
'head_extension' :[-math.pi/4, math.pi/4], 
'head_twist' :[-math.pi/4, math.pi/4], 

'ankle_angle_r' : [-math.pi/4, math.pi/4],
'subtalar_angle_r' : [-math.pi/4, math.pi/4],
'mtp_angle_r' : [-math.pi/4, math.pi/4],

'ankle_angle_l' : [-math.pi/4, math.pi/4],
'subtalar_angle_l' : [-math.pi/4, math.pi/4],
'mtp_angle_l' : [-math.pi/4, math.pi/4],

'knee_angle_r' : [0, 3/4*math.pi],
'knee_angle_l' : [0, 3/4*math.pi],

# Added by HSMR to make optimization more stable.
        'hip_flexion_r' : [-math.pi/4, 3/4*math.pi],  # 3
        'hip_adduction_r' : [-math.pi/4, 2/3*math.pi/4],  # 4
        'hip_rotation_r' : [-math.pi/4, math.pi/4],  # 5
        'hip_flexion_l' : [-math.pi/4, 3/4*math.pi],  # 10
        'hip_adduction_l' : [-math.pi/4, 2/3*math.pi/4],  # 11
        'hip_rotation_l' : [-math.pi/4, math.pi/4],  # 12

        'shoulder_r_x' : [-math.pi/2, math.pi/2+1.5],  # 29, from bsm.osim
        'shoulder_r_y' : [-math.pi/2, math.pi/2],  # 30
        'shoulder_r_z' : [-math.pi/2, math.pi/2],  # 31, from bsm.osim

        'shoulder_l_x' : [-math.pi/2-1.5, math.pi/2],  # 39, from bsm.osim
        'shoulder_l_y' : [-math.pi/2, math.pi/2],  # 40
        'shoulder_l_z' : [-math.pi/2, math.pi/2],  # 41, from bsm.osim

}

pose_param_name2qid = {name: qid for qid, name in enumerate(pose_param_names)}
qid2pose_param_name = {qid: name for qid, name in enumerate(pose_param_names)}

SKEL_LIM_QIDS = []
SKEL_LIM_BOUNDS = []
for name, (low, up) in pose_limits.items():
    if low > up:
        low, up = up, low
    SKEL_LIM_QIDS.append(pose_param_name2qid[name])
    SKEL_LIM_BOUNDS.append([low, up])

SKEL_LIM_BOUNDS = torch.Tensor(SKEL_LIM_BOUNDS).float()
SKEL_LIM_QID2IDX = {qid: i for i, qid in enumerate(SKEL_LIM_QIDS)}  # inverse mapping
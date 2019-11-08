import numpy as np

from utils import osc6dof, osc3dof, second_order_path_planner, first_order_arc, first_order_arc_dmp

def gen_reach_list(robot_config, object_xyz, deposit_xyz):

    rest_angles = [None, 3.14, 1.57, None, None, None]

    rot_wrist = True
    open_q = np.ones(3) * 1.1
    close_q = np.ones(3) * -0.1
    f_alpha_close = 7e-4
    f_alpha_open = 1e-3

    reach_list = {
        'pick_up': [
            # GRASP AND LIFT
            # move above object
            {'label': 'move above object',
            'target_pos': None,
            'start_pos': None,
            'orientation': None,
            'n_timesteps': 1000,
            'grasp_pos': close_q,
            'hold_timesteps': None,
            'offset': np.array([0, 0, 0.4]),
            'approach_buffer': -0.02,
            'ctrlr': osc6dof(robot_config, rest_angles),
            # 'traj_planner': second_order_path_planner,
            'traj_planner': first_order_arc,
            'z_rot': np.pi,
            'rot_wrist': rot_wrist,
            'target_options': 'object',
            'f_alpha': f_alpha_close,
            'error_thresh': 0.07
            },
            # get into grasping position
            {'label': 'get into grasp position',
            'target_pos': 'object',
            'start_pos': None,
            'orientation': 'object',
            'n_timesteps': 300,
            'grasp_pos': open_q,
            'hold_timesteps': None,
            'offset': np.array([0, 0, -0.01]),
            'approach_buffer': 0.0,
            'ctrlr': osc6dof(robot_config, rest_angles),
            'traj_planner': second_order_path_planner,
            'z_rot': np.pi,
            'rot_wrist': rot_wrist,
            'target_options': 'object',
            'f_alpha': f_alpha_open,
            'error_thresh': 0.02
            },
            # grasp object
            {'label': 'grasp object',
            'target_pos': 'object',
            'start_pos': None,
            'orientation': 'object',
            'n_timesteps': 500,
            'grasp_pos': close_q,
            'hold_timesteps': 500,
            'offset': np.array([0, 0, -0.01]),
            'approach_buffer': 0,
            'ctrlr': osc6dof(robot_config, rest_angles),
            'traj_planner': second_order_path_planner,
            'z_rot': np.pi,
            'rot_wrist': rot_wrist,
            'target_options': 'object',
            'f_alpha': f_alpha_close,
            'error_thresh': 0.02
            },
            # lift object
            {'label': 'lift object',
            'target_pos': 'object',
            'start_pos': None,
            'orientation': None,
            'n_timesteps': 100,
            'grasp_pos': close_q,
            'hold_timesteps': None,
            'offset': np.array([0, 0, 0.4]),
            'approach_buffer': 0.0,
            'ctrlr': osc6dof(robot_config, rest_angles),
            'traj_planner': second_order_path_planner,
            'z_rot': np.pi,
            'rot_wrist': rot_wrist,
            'target_options': 'object',
            'f_alpha': f_alpha_close,
            'error_thresh': 0.02
            }],

        'reach_target' : [
            {'label': 'reach to target',
            'target_pos': deposit_xyz,
            'start_pos': None,
            'orientation': None,
            'n_timesteps': 100,
            'grasp_pos': close_q,
            'hold_timesteps': None,
            'ctrlr': osc3dof(robot_config, rest_angles),
            'offset': np.array([0, 0, 0]),
            'approach_buffer': 0,
            #'traj_planner': second_order_path_planner,
            'traj_planner': first_order_arc,
            'z_rot': np.pi/2,
            'rot_wrist': False,
            'target_options': None,
            'f_alpha': f_alpha_close,
            'error_thresh': 0.02
            }],

        'drop_off' : [
            # go above drop off
            {'label': 'go above drop off',
            'target_pos': object_xyz,
            'start_pos': None,
            'orientation': None,
            'n_timesteps': 1000,
            'grasp_pos': close_q,
            'hold_timesteps': None,
            'offset': np.array([0, 0, 0.3]),
            'approach_buffer': 0.0,
            'ctrlr': osc6dof(robot_config, rest_angles),
            # 'traj_planner': second_order_path_planner,
            'traj_planner': first_order_arc,
            'z_rot': np.pi,
            'rot_wrist': rot_wrist,
            'target_options': None,
            'f_alpha': f_alpha_close,
            'error_thresh': 0.07
            },
            # go to drop off
            {'label': 'go to drop off',
            'target_pos': object_xyz,
            'start_pos': None,
            'orientation': None,
            'n_timesteps': 300,
            'grasp_pos': close_q,
            'hold_timesteps': None,
            'offset': np.array([0, 0, 0.02]),
            'approach_buffer': 0.0,
            'ctrlr': osc6dof(robot_config, rest_angles),
            'traj_planner': second_order_path_planner,
            'z_rot': np.pi,
            'rot_wrist': rot_wrist,
            'target_options': 'shifted',
            'f_alpha': f_alpha_close,
            'error_thresh': 0.02
            },
            # release
            {'label': 'release object',
            'target_pos': object_xyz,
            'start_pos': None,
            'orientation': None,
            'n_timesteps': 500,
            'grasp_pos': open_q,
            'hold_timesteps': 600,
            'offset': np.array([0, 0, 0.01]),
            'approach_buffer': 0.0,
            'ctrlr': osc6dof(robot_config, rest_angles),
            'traj_planner': second_order_path_planner,
            'z_rot': np.pi,
            'rot_wrist': rot_wrist,
            'target_options': 'shifted2',
            'f_alpha': f_alpha_open,
            'error_thresh': 0.02
            },

            # move above object
            {'label': 'lift clear of object',
            'target_pos': object_xyz,
            'start_pos': None,
            'orientation': None,
            'n_timesteps': 1000,
            'grasp_pos': open_q,
            'hold_timesteps': None,
            'offset': np.array([0, 0, 0.4]),
            'approach_buffer': 0.02,
            'ctrlr': osc6dof(robot_config, rest_angles),
            'traj_planner': second_order_path_planner,
            'z_rot': np.pi,
            'rot_wrist': rot_wrist,
            'target_options': 'shifted2',
            'f_alpha': f_alpha_open,
            'error_thresh': 0.1
            },
            ]
        }

    return reach_list

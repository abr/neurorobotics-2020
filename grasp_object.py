"""
Running operational space control using Mujoco. The controller will
move the end-effector to the target object's position and orientation.

This example controls all 6 degrees of freedom (position and orientation),
and applies a second order path planner to both position and orientation targets

After termination the script will plot results
"""
import numpy as np
import glfw
import time
import sys

from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.interfaces.mujoco import Mujoco
from abr_control.utils import transformations
from abr_control.utils.transformations import quaternion_multiply, quaternion_inverse

from utils import get_approach_path, osc6dof, osc3dof, second_order_path_planner, target_shift, adapt, first_order_arc

plot = False
pause = False
if len(sys.argv) > 1:
    pause = eval(sys.argv[1])

# initialize our robot config
robot_config = arm('jaco2_gripper')

# create our interface
interface = Mujoco(robot_config, dt=.001)
interface.connect()

spherical = True
in_index = [True, True, True, True, True, False]
n_input_joints = np.sum(in_index)
out_index = [True, True, True, True, True, False]
n_output_joints = np.sum(out_index)
input_signal = np.zeros(np.sum(in_index) * 2)
u_adapt = np.zeros(robot_config.N_JOINTS)
adapt = adapt(in_index=in_index, spherical=spherical)

joint_offset = np.array([0, 0, 0, 0, 0, 0])
interface.send_target_angles((robot_config.START_ANGLES+joint_offset))

feedback = interface.get_feedback()
hand_xyz = robot_config.Tx('EE', feedback['q'])

# initialize our resting config
# rest_angles = [None, None, robot_config.START_ANGLES[2], None, None, None]
# rest_angles = [None, robot_config.START_ANGLES[1], None, None, None, None]
rest_angles = [None, 3.14, 1.57, None, None, None]
# rest_angles = [None, robot_config.START_ANGLES[1], robot_config.START_ANGLES[2], None, None, None]

# set up lists for tracking data
ee_track = []
ee_angles_track = []
target_track = []
target_angles_track = []
object_xyz = np.array([-0.5, 0.0, 0.02])
deposit_xyz = np.array([-0.4, 0.5, 0.4])
adapt_text = np.array([0, 1, 0])

rot_wrist = True
open_q = np.ones(3) * 1.1
close_q = np.ones(3) * -0.1
# max grip force
max_grip = 8
fkp = 144
fkv = 15
f_alpha_close = 7e-4
f_alpha_open = 1e-3
u_grip_track = []
q_fing_track = []
dq_fing_track = []
fing_targ_track = []

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
        'z_offset': 0.4,
        'approach_buffer': -0.02,
        'ctrlr': osc6dof(robot_config, rest_angles),
        # 'traj_planner': second_order_path_planner,
        'traj_planner': first_order_arc,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist,
        'target_options': 'object',
        'f_alpha': f_alpha_close,
        'error_thres': 0.07
        },
        # get into grasping position
        {'label': 'get into grasp position',
        'target_pos': 'object',
        'start_pos': None,
        'orientation': 'object',
        'n_timesteps': 300,
        'grasp_pos': open_q,
        'hold_timesteps': None,
        'z_offset': -0.01,
        'approach_buffer': 0.0,
        'ctrlr': osc6dof(robot_config, rest_angles),
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist,
        'target_options': 'object',
        'f_alpha': f_alpha_open,
        'error_thres': 0.02
        },
        # grasp object
        {'label': 'grasp object',
        'target_pos': 'object',
        'start_pos': None,
        'orientation': 'object',
        'n_timesteps': 500,
        'grasp_pos': close_q,
        'hold_timesteps': 500,
        'z_offset': -0.01,
        'approach_buffer': 0,
        'ctrlr': osc6dof(robot_config, rest_angles),
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist,
        'target_options': 'object',
        'f_alpha': f_alpha_close,
        'error_thres': 0.02
        },
        # lift object
        {'label': 'lift object',
        'target_pos': 'object',
        'start_pos': None,
        'orientation': None,
        'n_timesteps': 100,
        'grasp_pos': close_q,
        'hold_timesteps': None,
        'z_offset': 0.4,
        'approach_buffer': 0.0,
        'ctrlr': osc6dof(robot_config, rest_angles),
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist,
        'target_options': 'object',
        'f_alpha': f_alpha_close,
        'error_thres': 0.02
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
        'z_offset': 0,
        'approach_buffer': 0,
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi/2,
        'rot_wrist': False,
        'target_options': None,
        'f_alpha': f_alpha_close,
        'error_thres': 0.02
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
        'z_offset': 0.3,
        'approach_buffer': 0.0,
        'ctrlr': osc6dof(robot_config, rest_angles),
        # 'traj_planner': second_order_path_planner,
        'traj_planner': first_order_arc,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist,
        'target_options': None,
        'f_alpha': f_alpha_close,
        'error_thres': 0.07
        },
        # go to drop off
        {'label': 'go to drop off',
        'target_pos': object_xyz,
        'start_pos': None,
        'orientation': None,
        'n_timesteps': 300,
        'grasp_pos': close_q,
        'hold_timesteps': None,
        'z_offset': 0.02,
        'approach_buffer': 0.0,
        'ctrlr': osc6dof(robot_config, rest_angles),
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist,
        'target_options': 'shifted',
        'f_alpha': f_alpha_close,
        'error_thres': 0.02
        },
        # release
        {'label': 'release object',
        'target_pos': object_xyz,
        'start_pos': None,
        'orientation': None,
        'n_timesteps': 500,
        'grasp_pos': open_q,
        'hold_timesteps': 600,
        'z_offset': 0.01,
        'approach_buffer': 0.0,
        'ctrlr': osc6dof(robot_config, rest_angles),
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist,
        'target_options': 'shifted2',
        'f_alpha': f_alpha_open,
        'error_thres': 0.02
        },

        # move above object
        {'label': 'lift clear of object',
        'target_pos': object_xyz,
        'start_pos': None,
        'orientation': None,
        'n_timesteps': 1000,
        'grasp_pos': open_q,
        'hold_timesteps': None,
        'z_offset': 0.4,
        'approach_buffer': 0.02,
        'ctrlr': osc6dof(robot_config, rest_angles),
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist,
        'target_options': 'shifted2',
        'f_alpha': f_alpha_open,
        'error_thres': 0.1
        },
        ]
    }

theta1 = np.pi/2
xyz1 = np.array([0, 0, 1])
rot1_quat = transformations.quaternion_about_axis(theta1, xyz1)

theta2 = np.pi/2
xyz2 = np.array([0, 1, 0])
rot2_quat = transformations.quaternion_about_axis(theta2, xyz2)

theta3 = -np.pi/2
xyz3 = np.array([1, 0, 0])
rot3_quat = transformations.quaternion_about_axis(theta3, xyz3)

rotQ = quaternion_multiply(rot3_quat, quaternion_multiply(rot2_quat, rot1_quat))

try:
    print('\nSimulation starting...\n')
    interface.viewer._paused = pause
    final_xyz = deposit_xyz

    # get joint ids for the gripper
    fingers = ['joint_thumb', 'joint_index', 'joint_pinky']
    # finger_ids = []
    # for fing in fingers:
    #     finger_ids.append(interface.model.get_joint_qpos_addr(fing))
    #
    # interface.sim.data.qpos[finger_ids[0]] = np.copy([-0.1])
    # interface.sim.data.qpos[finger_ids[1]] = np.copy([-0.1])
    # interface.sim.data.qpos[finger_ids[2]] = np.copy([-0.1])
    # interface.sim.forward()

    # ------ START DEMO -------
    visible_target = 'target_red'
    hidden_target = 'target_green'
    end_sim = False

    while not end_sim:
        mode_change = False
        reach_mode =  interface.viewer.reach_mode

        # go through each phase of the reach type
        for reach in reach_list[reach_mode]:
            count = 0
            hand_xyz = robot_config.Tx('EE', feedback['q'])

            # if we're reaching to target, update with user changes
            if reach_mode == 'reach_target':
                reach['target_pos'] = final_xyz

            # print('Next reach')
            if reach['target_options'] == 'object':

                reach['target_pos'] = interface.get_xyz('handle', object_type='geom')

                # target orientation should be that of an object in the environment
                objQ = interface.get_orientation('handle', object_type='geom')
                quat = quaternion_multiply(rotQ, objQ)
                startQ = np.copy(quat)
                reach['orientation'] = quat

            elif reach['target_options'] == 'shifted':
                # account for the object in the hand having slipped / rotated

                # get xyz of the hand
                hand_xyz = interface.get_xyz('EE', object_type='body')
                # get xyz of the object
                object_xyz = interface.get_xyz('handle', object_type='geom')

                reach['target'] = object_xyz + (object_xyz - hand_xyz)

                # get current orientation of hand
                handQ_prime = interface.get_orientation('EE', object_type='body')
                # get current orientation of object
                objQ_prime = interface.get_orientation('handle', object_type='geom')

                # get the difference between hand and object
                rotQ_prime = quaternion_multiply(handQ_prime, quaternion_inverse(objQ_prime))
                # compare with difference at start of movement
                dQ = quaternion_multiply(rotQ_prime, quaternion_inverse(rotQ))
                # transform the original target by the difference
                shiftedQ = quaternion_multiply(dQ, startQ)

                reach['orientation'] = shiftedQ

            elif reach['target_options'] == 'shifted2':
                reach['orientation'] = shiftedQ


            # calculate our position and orientation path planners, with their
            # corresponding approach
            traj_planner, orientation_planner, target_data = get_approach_path(
                robot_config=robot_config,
                path_planner=reach['traj_planner'](reach['n_timesteps']),
                q=feedback['q'],
                target_pos=reach['target_pos'],
                target_orientation=reach['orientation'],
                start_pos=reach['start_pos'],
                max_reach_dist=None,
                min_z=0.0,
                approach_buffer=reach['approach_buffer'],
                z_offset=reach['z_offset'],
                z_rot=reach['z_rot'],
                rot_wrist=reach['rot_wrist'])


            # reset our dmp since the user may be changing the target
            # if reach_mode == 'reach_target':
            #     traj_planner.reset(
            #         position=hand_xyz,
            #         target_pos=(final_xyz + np.array([0, 0, reach['z_offset']])))

            at_target = False
            # continue the phase of the reach until the stop criteria is met
            u_gripper_prev = np.zeros(3)
            while not at_target:
                finger_q = []
                finger_dq = []
                for finger in fingers:
                    finger_q.append(interface.sim.data.qpos[interface.sim.model.get_joint_qpos_addr(finger)])
                    finger_dq.append(interface.sim.data.qvel[interface.sim.model.get_joint_qpos_addr(finger)])
                # check for our exit command (caps lock)
                if interface.viewer.exit:
                    end_sim = True
                    glfw.destroy_window(interface.viewer.window)
                    break

                # check if the user has changed the reaching mode
                prev_reach_mode = reach_mode
                reach_mode = interface.viewer.reach_mode
                if prev_reach_mode != reach_mode:
                    mode_change = True
                    break
                else:
                    mode_change = False

                # get our user shifted drop off location
                old_final_xyz = final_xyz
                final_xyz = target_shift(
                    interface=interface,
                    base_location=final_xyz,
                    scale=0.05,
                    xlim=[-0.5, 0.5],
                    ylim=[-0.5, 0.5],
                    zlim=[0.0, 0.7])

                # get arm feedback
                feedback = interface.get_feedback()
                hand_xyz = robot_config.Tx('EE', feedback['q'])

                # update our path planner position and orientation
                if reach_mode == 'reach_target':
                    error = np.linalg.norm(
                        (hand_xyz - (final_xyz + np.array([0, 0, reach['z_offset']]))))
                    if error < 0.05:
                        pos = final_xyz + reach['z_offset']
                        vel = np.zeros(3)
                    else:
                        if not np.allclose(final_xyz, old_final_xyz, atol=1e-5):
                            traj_planner.reset(
                                position=pos,
                                target_pos=(final_xyz + np.array([0, 0, reach['z_offset']])))
                        pos, vel = traj_planner._step(error=error)
                    orient = np.zeros(3)

                else:
                    error = np.linalg.norm((hand_xyz-target_data['approach_pos']))
                    pos, vel = traj_planner.next()
                    orient = orientation_planner.next()

                target = np.hstack([pos, orient])

                # set our path planner visualization and final drop off location
                interface.set_mocap_xyz(visible_target, final_xyz)
                interface.set_mocap_xyz(hidden_target, np.array([0, 0, -1]))
                if interface.viewer.path_vis:
                    interface.set_mocap_xyz('path_planner_orientation', target[:3])
                    interface.set_mocap_orientation('path_planner_orientation',
                        transformations.quaternion_from_euler(
                            orient[0], orient[1], orient[2], 'rxyz'))
                else:
                    interface.set_mocap_xyz('path_planner_orientation', np.array([0, 0, -1]))

                # calculate our osc control signal
                u = reach['ctrlr'].generate(
                    q=feedback['q'],
                    dq=feedback['dq'],
                    target=target,
                    #target_vel=np.hstack([vel, np.zeros(3)])
                    )

                # adaptive control, if toggled on
                if interface.viewer.adapt:
                    training_signal = reach['ctrlr'].training_signal[out_index]
                    input_signal[:n_input_joints] = feedback['q'][in_index]
                    input_signal[n_input_joints:] = feedback['dq'][in_index]
                    u_adapt[in_index] = adapt.generate(
                        input_signal=input_signal, training_signal=training_signal)
                    # TODO this can be optimized, only needs to be set if adaptation toggles
                    interface.set_mocap_xyz('adapt_on', adapt_text)
                    interface.set_mocap_xyz('adapt_off', [0, 0, -1])
                else:
                    u_adapt = np.zeros(robot_config.N_JOINTS)
                    # TODO this can be optimized, only needs to be set if adaptation toggles
                    interface.set_mocap_xyz('adapt_off', adapt_text)
                    interface.set_mocap_xyz('adapt_on', [0, 0, -1])

                u += u_adapt

                # get our gripper command
                #NOTE interface lets you toggle gripper status with the 'n' key
                #TODO remove the interface gripper control for actual demo
                u_gripper = fkp * (reach['grasp_pos'] - np.asarray(finger_q)) - fkv*np.asarray(finger_dq)
                u_gripper = reach['f_alpha'] * u_gripper + (1-reach['f_alpha']) * u_gripper_prev
                u_gripper = np.clip(u_gripper, a_max=max_grip, a_min=-max_grip)
                u_gripper_prev = np.copy(u_gripper)

                # stack our control signals and send to mujoco, stepping the sim forward
                u = np.hstack((u, u_gripper*interface.viewer.gripper))

                interface.send_forces(u, update_display=True if count % 1 == 0 else False)

                # calculate our 2norm error
                # if count % 100 == 0:
                #     print('u adapt: ', u_adapt)
                #     print('u gripper: ', u_gripper)
                #     print('q fingers: ', finger_q)

                # track data
                if plot:
                    u_grip_track.append(u_gripper)
                    q_fing_track.append(finger_q)
                    dq_fing_track.append(finger_dq)
                    fing_targ_track.append(reach['grasp_pos'])
                    ee_track.append(np.copy(hand_xyz))
                    ee_angles_track.append(transformations.euler_from_matrix(
                        robot_config.R('EE', feedback['q']), axes='rxyz'))
                    target_track.append(np.copy(target[:3]))
                    target_angles_track.append(np.copy(target[3:]))
                count += 1

                # once we have the object, keep reaching to the target as the user
                # changes it
                if reach_mode == 'reach_target':
                    at_target = False

                    if error < reach['error_thres']:
                        visible_target = 'target_green'
                        hidden_target = 'target_red'
                    else:
                        visible_target = 'target_red'
                        hidden_target = 'target_green'
                else:
                    visible_target = 'target_red'
                    hidden_target = 'target_green'

                    # the reason we differentiate hold and n timesteps is that hold is how
                    # long we want to wait to allow for the action, mainly used for grasping,
                    # whereas n_timesteps determines the number of steps in the path planner.
                    # we check n_timesteps*2 to allow the arm to catch up to the path planner
                    # else:

                    if reach['hold_timesteps'] is not None:
                        if count >= reach['hold_timesteps']:
                            at_target = True
                    elif count > reach['n_timesteps']*2 and error < 0.07:
                            at_target = True

                interface.viewer.custom_print = (
                    '%s\nerror: %.3fm\nGripper toggle: %i'
                    % (reach['label'], error, interface.viewer.gripper))

            if mode_change:
                break

        if end_sim:
            break
        # if the user has not changed mode then we have finished our grasping phases
        # so switch to reaching to target
        if not mode_change and reach_mode != 'reach_target':
            interface.viewer.reach_mode = 'reach_target'

finally:
    # stop and reset the simulation
    interface.disconnect()

    print('Simulation terminated...')

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611
    ee_track = np.array(ee_track).T
    ee_angles_track = np.array(ee_angles_track).T
    target_track = np.array(target_track).T
    target_angles_track = np.array(target_angles_track).T

    u_grip_track = np.asarray(u_grip_track)
    q_fing_track = np.asarray(q_fing_track)
    dq_fing_track = np.asarray(dq_fing_track)
    fing_targ_track = np.asarray(fing_targ_track)

    if u_grip_track.shape[0] > 0 and plot==True:
        plt.figure()
        plt.subplot(311)
        title = ('kp_%s kv_%s alpha_%s' %(fkp, fkv, reach['f_alpha']))
        plt.title(title)
        plt.plot(u_grip_track)
        plt.ylabel('u_fingers')
        plt.legend(['thumb', 'index', 'pinky'])
        plt.subplot(312)
        plt.plot(q_fing_track)
        plt.plot(fing_targ_track, 'r--')
        plt.ylabel('q_fingers')
        plt.legend(['thumb', 'index', 'pinky', 'target'])
        plt.subplot(313)
        plt.plot(dq_fing_track)
        plt.ylabel('dq_fingers')
        plt.legend(['thumb', 'index', 'pinky'])
        plt.savefig(title.replace('.', '_'))
        plt.show()

    if ee_track.shape[0] > 0 and plot==True:
        # plot distance from target and 3D trajectory
        label_pos = ['x', 'y', 'z']
        label_or = ['a', 'b', 'g']
        c = ['r', 'g', 'b']

        fig = plt.figure(figsize=(8,12))
        ax1 = fig.add_subplot(311)
        ax1.set_ylabel('3D position (m)')
        for ii, ee in enumerate(ee_track):
            ax1.plot(ee, label='EE: %s' % label_pos[ii], c=c[ii])
            ax1.plot(target_track[ii], label='Target: %s' % label_pos[ii],
                     c=c[ii], linestyle='--')
        ax1.legend()

        ax2 = fig.add_subplot(312)
        for ii, ee in enumerate(ee_angles_track):
            ax2.plot(ee, label='EE: %s' % label_or[ii], c=c[ii])
            ax2.plot(target_angles_track[ii], label='Target: %s'%label_or[ii],
                     c=c[ii], linestyle='--')
        ax2.set_ylabel('3D orientation (rad)')
        ax2.set_xlabel('Time (s)')
        ax2.legend()

        ee_track = ee_track.T
        target_track = target_track.T
        ax3 = fig.add_subplot(313, projection='3d')
        ax3.set_title('End-Effector Trajectory')
        ax3.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], label='ee_xyz')
        ax3.plot(target_track[:, 0], target_track[:, 1], target_track[:, 2],
                 label='ee_xyz', c='g', linestyle='--')
        ax3.scatter(target_track[-1, 0], target_track[-1, 1],
                    target_track[-1, 2], label='target', c='g')
        ax3.legend()
        plt.show()


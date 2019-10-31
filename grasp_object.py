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

from utils import get_approach_path, osc6dof, osc3dof, second_order_path_planner, target_shift

plot = False
pause = False
if len(sys.argv) > 1:
    pause = eval(sys.argv[1])

# initialize our robot config
robot_config = arm('jaco2_gripper')

# create our interface
interface = Mujoco(robot_config, dt=.001)
interface.connect()
joint_offset = np.array([0, 0, 0, 0, 0, 0])
interface.send_target_angles((robot_config.START_ANGLES+joint_offset))

feedback = interface.get_feedback()
hand_xyz = robot_config.Tx('EE', feedback['q'])

# set up lists for tracking data
ee_track = []
ee_angles_track = []
target_track = []
target_angles_track = []
object_xyz = np.array([-0.5, 0.0, 0.0])
deposit_xyz = np.array([0.4, 0.5, 0.2])

rot_wrist = True
open_force = 3
close_force = -5

try:
    reaching_list = [
        # GRASP AND LIFT
        # move above object
        {'type': 'grasp',
        'target_pos': object_xyz,
        'start_pos': None,
        'orientation': None,
        'n_timesteps': 1000,
        'grasp_force': open_force,
        'hold_timesteps': None,
        'z_offset': 0.2,
        'approach_buffer': 0.02,
        'ctrlr': osc6dof(robot_config),
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist
        },
        # get into grasping position
        {'type': 'grasp',
        'target_pos': object_xyz,
        'start_pos': None,
        'orientation': None,
        'n_timesteps': 300,
        'grasp_force': open_force,
        'hold_timesteps': None,
        'z_offset': 0.055,
        'approach_buffer': -0.02,
        'ctrlr': osc6dof(robot_config),
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist
        },
        # grasp object
        {'type': 'grasp',
        'target_pos': object_xyz,
        'start_pos': None,
        'orientation': None,
        'n_timesteps': 500,
        'grasp_force': close_force,
        'hold_timesteps': 500,
        'z_offset': 0.055,
        'approach_buffer': 0,
        'ctrlr': osc6dof(robot_config),
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist
        },
        # lift object
        {'type': 'grasp',
        'target_pos': object_xyz,
        'start_pos': None,
        'orientation': None,
        'n_timesteps': 300,
        'grasp_force': close_force,
        'hold_timesteps': None,
        'z_offset': 0.2,
        'approach_buffer': 0.03,
        'ctrlr': osc6dof(robot_config),
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi,
        'rot_wrist': rot_wrist
        },

        # MOVE TO TARGET AND DROP OFF
        # move above drop off
        # {'type': 'target_reach',
        # 'target_pos': deposit_xyz,
        # 'start_pos': None,
        # 'orientation': None,
        # 'n_timesteps': 1000,
        # 'grasp_force': close_force,
        # 'hold_timesteps': None,
        # 'z_offset': 0.5,
        # 'approach_buffer': 0.0,
        # 'traj_planner': second_order_path_planner
        # },
        # move to drop off
        {'type': 'target_reach',
        'target_pos': deposit_xyz,
        'start_pos': None,
        'orientation': None,
        'n_timesteps': 100,
        'grasp_force': close_force*2,
        'hold_timesteps': None,
        'ctrlr': osc3dof(robot_config),
        'z_offset': 0,
        'approach_buffer': 0,
        'traj_planner': second_order_path_planner,
        'z_rot': np.pi/2,
        'rot_wrist': False
        },
        # drop off object
        # {'type': 'target_reach',
        # 'target_pos': deposit_xyz,
        # 'start_pos': None,
        # 'orientation': None,
        # 'n_timesteps': 1000,
        # 'grasp_force': open_force,
        # 'hold_timesteps': 500,
        # 'z_offset': 0.31,
        # 'approach_buffer': 0,
        # 'traj_planner': second_order_path_planner
        # },
        # lift clear of object in z
        # {'type': 'target_reach',
        # 'target_pos': deposit_xyz,
        # 'start_pos': None,
        # 'orientation': None,
        # 'n_timesteps': 1000,
        # 'grasp_force': open_force,
        # 'hold_timesteps': None,
        # 'z_offset': 0.65,
        # 'approach_buffer': 0.0,
        # 'traj_planner': second_order_path_planner
        # }
        ]


    print('\nSimulation starting...\n')
    interface.viewer._paused = pause

    final_xyz = deposit_xyz

    # this can later be expaned to actually check the user input to start reaching
    # for object to grasp, or for different objects, for now just set it to True
    interface.viewer.pick_up_object = True

    # wait until the user hits the 'pick up object' button
    while not interface.viewer.pick_up_object:
        time.sleep(0.5)

    for reach in reaching_list:
        count = 0
        hand_xyz = robot_config.Tx('EE', feedback['q'])

        if reach['type'] == 'target_reach':
            reach['target_pos'] = final_xyz

        print('Next reach')
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


        if reach['type'] == 'target_reach':
            traj_planner.reset(
                position=hand_xyz,
                target_pos=(final_xyz + np.array([0, 0, reach['z_offset']])))

        at_target = False
        count = 0
        while not at_target:
            # check for our exit command (caps lock)
            if interface.viewer.exit:
                glfw.destroy_window(interface.viewer.window)
                break

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
            if reach['type'] == 'target_reach':
                error = np.linalg.norm(
                    (hand_xyz - (final_xyz + np.array([0, 0, reach['z_offset']]))))
                if not np.allclose(final_xyz, old_final_xyz, atol=1e-5):
                    traj_planner.reset(
                        position=pos,
                        target_pos=(final_xyz + np.array([0, 0, reach['z_offset']])))
                pos, vel = traj_planner._step(error=error)

            else:
                error = np.linalg.norm((hand_xyz-target_data['approach_pos']))
                pos, vel = traj_planner.next()

            #TODO will need to update the orientation planner somehow, not using
            # dmps so can't use reset, may need to regen and change n_timesteps?
            orient = orientation_planner.next()
            target = np.hstack([pos, orient])

            # set our path planner visualization and final drop off location
            interface.set_mocap_xyz('target', final_xyz)
            # interface.set_mocap_xyz('path_planner_orientation', target[:3])
            # interface.set_mocap_orientation('path_planner_orientation',
            #     transformations.quaternion_from_euler(
            #         orient[0], orient[1], orient[2], 'rxyz'))

            # calculate our osc control signal
            u = reach['ctrlr'].generate(
                q=feedback['q'],
                dq=feedback['dq'],
                target=target,
                #target_vel=np.hstack([vel, np.zeros(3)])
                )

            # get our gripper command
            u_gripper = np.ones(3) * reach['grasp_force']

            # stack our control signals and send to mujoco, stepping the sim forward
            u = np.hstack((u, u_gripper))
            interface.send_forces(u, update_display=True if count % 2 == 0 else False)

            # calculate our 2norm error
            if count % 500 == 0:
                print('error: ', error)

            # track data
            ee_track.append(np.copy(hand_xyz))
            ee_angles_track.append(transformations.euler_from_matrix(
                robot_config.R('EE', feedback['q']), axes='rxyz'))
            target_track.append(np.copy(target[:3]))
            target_angles_track.append(np.copy(target[3:]))
            count += 1

            # once we have the object, keep reaching to the target as the user
            # changes it
            if reach['type'] == 'target_reach':
                at_target = False
            # the reason we differentiate hold and n timesteps is that hold is how
            # long we want to wait to allow for the action, mainly used for grasping,
            # whereas n_timesteps determines the number of steps in the path planner.
            # we check n_timesteps*2 to allow the arm to catch up to the path planner
            else:
                if reach['hold_timesteps'] is not None:
                    if count >= reach['hold_timesteps']:
                        at_target = True
                else:
                    if error < 0.02:
                        at_target = True
                    elif count > reach['n_timesteps']*2:
                        at_target = True

finally:
    # stop and reset the simulation
    interface.disconnect()

    print('Simulation terminated...')

    ee_track = np.array(ee_track).T
    ee_angles_track = np.array(ee_angles_track).T
    target_track = np.array(target_track).T
    target_angles_track = np.array(target_angles_track).T

    if ee_track.shape[0] > 0 and plot==True:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611
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

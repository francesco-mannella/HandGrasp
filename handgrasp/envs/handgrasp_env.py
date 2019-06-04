from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import numpy as np
import pybullet
import gym 
from .handgrasp_robot import Hand


import sys, os

#import kukacomp.data as data
#bullet_client.setAdditionalSearchPath(dataObsSpaces.getDataPath())

def DefaultRewardFunc(observation):
    return 0


class HandGraspEnv(MJCFBaseBulletEnv):
    
    intrinsic_timesteps = 50 # int(1e7)
    extrinsic_timesteps = 50 # int(1e3)
    
    def __init__(self, render=False):

        self.robot = Hand()
        MJCFBaseBulletEnv.__init__(self, self.robot, render)
        
        self._cam_dist = 1.2
        self._cam_yaw = 30
        self._cam_roll = 0
        self._cam_pitch = -30
        self._render_width = 320
        self._render_height = 240
        self._cam_pos = [0,0,.4]
        self.setCamera()
        self.eyes = {}

        self.reward_func = DefaultRewardFunc
        
        self.robot.used_objects = []

    def setCamera(self):
        '''
        initialize environment camera
        '''
        self.envCamera = EnvCamera(
                distance=self._cam_dist, 
                yaw=self._cam_yaw,
                pitch=self._cam_pitch, 
                roll=self._cam_roll, 
                pos=self._cam_pos,
                width=self._render_width,
                height=self._render_height)
    
    def set_eye(self, name):
        '''
        initialize eye
        '''
        pos = [0.01, 0, 1.2]
        cam = EyeCamera(pos, [0, 0, 0])
        self.eyes[name] = cam

        
    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.81, 
                timestep=0.005, frame_skip=1)
    
    def reset(self):

        super(HandGraspEnv, self).reset()
        self._p.setGravity(0.,0.,-9.81)
        self.camera._p = self._p
        for name in self.eyes.keys():
           self.eyes[name]._p = self._p
        
        self._p.resetDebugVisualizerCamera(
                self._cam_dist, self._cam_yaw, 
                self._cam_pitch, self._cam_pos)

        self.timestep = 0
        
        return self.get_observation()

    def render(self, mode='human', close=False):
            if mode == "human":
                    self.isRender = True
            if mode != "rgb_array":
                    return np.array([])

            rgb_array = self.envCamera.render(self._p)
            return rgb_array

    def get_observation(self):

        joints = self.robot.calc_state()
        sensors = self.robot.get_touch_sensors()
        
        observation = {
                Hand.ObsSpaces.JOINT_POSITIONS: joints,
                Hand.ObsSpaces.TOUCH_SENSORS: sensors }

        return observation


    def step(self, action):
        assert(not self.scene.multiplayer)
        
        self.robot.apply_action(action)
        self.scene.global_step()  

        
        observation = self.get_observation()

        reward = self.reward_func(observation)       
        
        done = False

        info = {}
        
        self.timestep += 1

        return observation, reward, done, info

class EnvCamera:

    def __init__(self, distance, yaw, pitch, roll, pos, 
            fov=80, width=320, height=240):
        
        self.dist = distance
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.pos = pos
        self.fov = fov
        self.render_width = width
        self.render_height = height

    def render(self, bullet_client = None):
        
        if bullet_client is None:
            bullet_client = self._p

        view_matrix = bullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition = self.pos,
                distance=self.dist,
                yaw=self.yaw,
                pitch=self.pitch,
                roll=self.roll,
                upAxisIndex=2)

        proj_matrix = bullet_client.computeProjectionMatrixFOV(
                fov=self.fov, aspect=float(self.render_width)/self.render_height,
                nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = bullet_client.getCameraImage(
                width=self.render_width, height=self.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
                )

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array
     
class EyeCamera:

    def __init__(self, eyePosition, targetPosition,
            fov=80, width=320, height=240):
        
        self.eyePosition = eyePosition
        self.targetPosition = targetPosition
        self.upVector = [0, 0, 1]
        self.fov = fov
        self.render_width = width
        self.render_height = height
        self._p = None
        self.pitch_roll = False
    
    def render(self, *args, **kargs):
        if self.pitch_roll is True:
            return self.renderPitchRoll(*args, **kargs)
        else:
            return self.renderTarget(*args, **kargs)


    def renderTarget(self, targetPosition, bullet_client = None):
        
        if bullet_client is None:
            bullet_client = self._p

        self.targetPosition = targetPosition

        view_matrix = bullet_client.computeViewMatrix(
                cameraEyePosition = self.eyePosition,
                cameraTargetPosition = self.targetPosition,
                cameraUpVector=self.upVector)

        proj_matrix = bullet_client.computeProjectionMatrixFOV(
                fov=self.fov, aspect=float(self.render_width)/self.render_height,
                nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = bullet_client.getCameraImage(
                width=self.render_width, height=self.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
                )

        rgb_array = np.array(px).reshape(self.render_height, self.render_width, 4)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array
            
    def renderPitchRoll(self, distance, roll, pitch, yaw, bullet_client = None):
        
        if bullet_client is None:
            bullet_client = self._p

        self.targetPosition = targetPosition

        view_matrix = bullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition = self.pos,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2)

        proj_matrix = bullet_client.computeProjectionMatrixFOV(
                fov=self.fov, aspect=float(self.render_width)/self.render_height,
                nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = bullet_client.getCameraImage(
                width=self.render_width, height=self.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
                )

        rgb_array = np.array(px).reshape(self.render_height, self.render_width, 4)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array
            



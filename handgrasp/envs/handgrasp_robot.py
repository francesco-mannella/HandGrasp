from robot_bases import MJCFBasedRobot, URDFBasedRobot
import numpy as np
import pybullet_data
import os
import gym
from robot_bases import BodyPart



class Hand(URDFBasedRobot):
    
    used_objects = []

    object_poses = {}
    num_joints = 24
    num_touch_sensors = 10
    
    class ObsSpaces: 
        JOINT_POSITIONS = "joint_positions"
        TOUCH_SENSORS = "touch_sensors"

    def __init__(self):

        self.robot_position = [0, 0, 0]
        self.contact_threshold = 0.1

        self.action_dim = self.num_joints
        
        URDFBasedRobot.__init__(self, 
                'hand_description/urdf/hand.urdf', 
                'hand0', action_dim=self.action_dim, obs_dim=1)
              
        self.observation_space = gym.spaces.Dict({
            self.ObsSpaces.JOINT_POSITIONS: gym.spaces.Box(
                -np.inf, np.inf, [self.num_joints], dtype = float),
            self.ObsSpaces.TOUCH_SENSORS: gym.spaces.Box(
                0, np.inf, [self.num_touch_sensors], dtype = float)})

        self.object_names = dict()
        self.object_bodies = dict()
        self.robot_parts = {}
        self.touch_sensors = []
        self.joint_names = []

        for arm_part in ["arm_base", "arm0", "arm1", "palm"]:
                self.joint_names.append(arm_part)

        for finger in ["thumb", "index", "middle", "ring", "little"]:
            for nm in range(4):
                if nm >= 2:
                    self.touch_sensors.append(finger+"%d"%nm)
                self.joint_names.append(finger+"%d"%nm)

    def reset(self, bullet_client):
        bullet_client.resetSimulation()
        super(Hand, self).reset(bullet_client)
        return self.calc_state()
    
    def get_contacts(self, forces=False):    

        contact_dict = {}
        for part_name, part in self.parts.items():         
            contacts = []
            for contact in part.contact_list():
                if abs(contact[8]) < self.contact_threshold:
                    name = self.object_names[contact[2]] 
                    if not forces:
                        if part_name in contact_dict.keys():
                            contact_dict[part_name].append(name)
                        else:
                            contact_dict[part_name]= [name]  
                    else:
                        force = contact[9]
                        if part_name in contact_dict.keys():
                            contact_dict[part_name].append([name, force])
                        else:
                            contact_dict[part_name] = [(name, force)]  

        return contact_dict
    
    def get_touch_sensors(self): 
        
        sensors = np.zeros(self.num_touch_sensors)
        contacts = self.get_contacts(forces=True)
        for i, skin in enumerate(self.touch_sensors):
            if skin in contacts.keys():
                cnts = contacts[skin]   
                if len(cnts) > 0:
                    force = np.max([cnt[1] for cnt in cnts])
                    sensors[i] = force
                
            return sensors 

    def robot_specific_reset(self, bullet_client):

        self.robot_body.reset_position(self.robot_position)

        self.object_bodies["hand"] = self.robot_body
        self.object_names[0] = "hand"

        for obj_name in self.used_objects:
            pos = self.object_poses[obj_name]
            obj = get_object(bullet_client,
                    "hand_description/urdf/{}.urdf".format(obj_name),
                    *pos)
            self.object_bodies[obj_name] = obj
            self.object_names.update({obj.bodies[0]: obj_name})
        
        for _,joint in self.jdict.items():
            joint.reset_current_position(0, 0)

        for name, part in self.parts.items():
            self.robot_parts.update({part.bodyPartIndex: name})

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        assert(len(a) == self.num_joints)

        # constraints

        for i, joint in enumerate(self.joint_names):
            if "arm" in joint:
                a[i] = np.maximum(-np.pi, np.minimum(np.pi, a[i]))
        for i, joint in enumerate(self.joint_names):
            if "thumb" in joint:
                a[i] = -a[i]
        for i, joint in enumerate(self.joint_names):
            if "0" in joint and not "arm" in joint:
                a[i] = np.maximum(-np.pi/8, np.minimum(np.pi/8, a[i]))
            else:
                a[i] = np.maximum(0, np.minimum(np.pi*0.4, a[i]))
        for i, joint in enumerate(self.joint_names):
            if "thumb" in joint:
                a[i] = -a[i]
        # jdict.set_position
        for i, joint in enumerate(self.joint_names):
            self.jdict[joint].set_position(a[i])

    def calc_state(self):
        joints = [ self.jdict[joint].get_position() for joint in self.joint_names]
        
        return joints 



 
def get_object(bullet_client, object_file, x, y, z, roll=0, pitch=0, yaw=0):

    position = [x, y, z]
    orientation = bullet_client.getQuaternionFromEuler([roll, pitch, yaw])
    fixed = True
    body = bullet_client.loadURDF(
            fileName=os.path.join(pybullet_data.getDataPath(), object_file),
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=False,
            flags=bullet_client.URDF_USE_INERTIA_FROM_FILE)
    part_name, _ = bullet_client.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    return BodyPart(bullet_client, part_name, bodies, 0, -1)


'''

.step(action)

'''

def reset(self):  # episode initialize


def step(self,action):
    return obs,reward,done,extra_info  # done : flag for true/false


import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW=False
IM_WIDTH=640    # CAMERA가 찍은 화면 크기
IM_HEIGHT=480
SECONDS_PER_EPISODE=10
REPLAY_MEMORY_SIZE=5_000
MIN_REPLAY_MEMORY_SIZE=1_000
MINIBATCH_SIZE=16
PREDICTION_BATCH_SIZE=1
TRAINING_BATCH_SIZE=MINIBATCH_SIZE//4
UPDATE_TARGET_EVERY = 5 # 몇개의 EPISODE 후에 UPDATE 할 것인지.
MODEL_NAME="Xception"

MEMORY_FRACTION =0.8 # GPU 메모리 사용 제한
MIN_REWARD=-200

EPISODES=100 # 에피소드 횟수
DISCOUNT=0.99
EPSILON=1
EPSILON_DECAY=0.95 ## 0.9975 0.99975
MIN_EPSILON=0.001

AGGREGATE_STATS_EVERY=10



class CarEnv:
    SHOW_CAM=SHOW_PREVIEW
    STEER_AMT=1.0       # 운전대 회전 정도
    im_width=IM_WIDTH
    im_height=IM_HEIGHT
    front_camera=None

    def __init__(self):
        self.client=carla.Client("localhost",2000)
        self.client.set_timeout(4.0)

        self.world=self.client.get_world()
        self.blueprint_library=self.world.get_blueprint_library() # get blueprint_library
        self.model_3=blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_list=[]
        self.actor_list=[]

        self.transform=random.choice(self.world.get_map().get_spawn_points()) # 자동차 랜덤위치 생성, 공중에 생성돼 collision 가능성 있음
        self.vehicle=self.world.spawn_actor(self.model_3,self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam=self.blueprint_library.find("sensor.camera.rgb")
        self.rgb.set_attribute("image_size_x",f"{self.im_width}")
        self.rgb.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb.set_attribute("fov", f"110")

        transform=carla.Transform(carla.Location(x=2.5,z=0.7)) # camera location
        self.sensor=self.world.spawn_actor(self.rgb_cam,transform,attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data : self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0)) # car act
        time.sleep(4)

        colsensor=self.blueprint_library.find("sensor.other.collision")
        self.colsensor=self.world.spawn_actor(colsensor,transform,attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event : self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start=time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))  # car act
        time.sleep(4)
        return self.front_camera

    def collision_data(self,event):
        self.collision_list.append(event)


    def process_img(image):
        i=np.array(image.raw_data)
        #print(i.shape)
        i2=i.reshape(self.im_height,self.im_width,4))
        i3=i2[:,:,:3]
        if self.SHOW_CAM:
            cv2.imshow("",i3)
            cv2.waitKey(1)
        self.front_camera=i3


    def step(self,action): #
        if action==0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=-1*self.STEER_AMT)) #

        elif action==1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))  #

        elif action==2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))  #

        v=self.vehicle.get_velocity()
        kmh=int(3.6*math.sqrt(v.x**2,v.y**2,v.z**2))

        if len(self.collision_list)!=0:
            done =True
            reward=-200
        elif kmh <50:
            done=False
            reward=-1
        else:
            done=False
            reward=1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done=True

        return self.front_camera,reward,done,None


actor_list = []

try:
    client=carla.Client("localhost",2000)
    client.set_timeout(4.0)
    world=client.get_world()
    blueprint_library=world.get_blueprint_library()

    bp=blueprint_library.filter("model3")[0]
    print(bp)

    spawn_point=random.choice(world.get_map().get_spawn_points())

    vehicle=world.spawn_actor(bp,spawn_point)
    #vehicle.set_autopilot(True)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0.0))
    actor_list.append(vehicle)

    cam_bp=blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov","110")

    spawn_point=carla.Transform(carla.Location(x=2.5,z=0.7))

    sensor=world.spawn_actor(cam_bp,spawn_point,attach_to=vehicle)
    actor_list.append(sensor)

    sensor.listen(lambda data: process_img(data))

    time.sleep(5)


finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")


class DQNAgent: # 하나는 꾸준히 학습, 하나는 co
    def __init__(self):
        self.model=self.create_model()
        self.target_model=self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory=deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard=ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}") #
        self.target_update_counter=0
        self.graph=tf.get_default_graph()


        # Flags
        self.terminate=False
        self.last_logged_episode= 0
        self.training_initialized=False

    def create_model(self):
        # model=sequential()
        # model.add()

        base_model=Xception(weights=None,include_top=False,input_shape(IM_HEIGHT,IM_WIDTH,3))
        x=base_model.output
        x=GlobalAveragePooling2D()(x)

        predictions=Dense(3,activation="linear")(x) # left,right,straight
        model=Model(inputs=base_model.input,ouputs=predictions)
        model.compile(Loss="mse",optimizer=Adam(Lr=0.001),metrics=["accuracy"])
        return model

    def update_replay_memory(self,transition): # transition으로 학습하는데 필요한 정보 전달.
        # transition = (current_state,action,reward,new_state,done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory)<MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch=random.sample(self.replay_memory,MINIBATCH_SIZE)

        current_states=np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            current_qs_list=self.model.predict(current_states,PREDICTION_BATCH_SIZE)

        new_current_states=np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            future_qs_list=self.target_model.predict(new_current_states,PREDICTION_BATCH_SIZE)

        X=[]
        y=[]

        # Reinforcement Learning
        for index, (current_state,action,reward,new_state,done) in enumerate(minibatch):
            if not done:
                max_future_q=np.max(future_qs_list[index])
                new_q=reward+DISCOUNT*max_future_q
            else:
                new_q=reward

        current_qs=current_qs_list[index]
        current_qs[action]=new_q  # update Q value

        X.append(current_state)
        y.append(current_qs)

        log_this_step=False
        if self.tensorboard.step>self.last_logged_episode:
            log_this_step=True
            self.last_logged_episode=self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X)/255,np.array(y),batch_size=TRAINING_BATCH_SIZE,verbose=0,shuffle=False,callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter+=1

        if self.target_update_counter> UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter=0

    def get_qs(self,state):
        return self.model.predict(np.array(state).reshape(-1*state.shape)/255)[0]

    def train_in_loop(self):
        X=np.random.uniform(size=(1,IM_HEIGHT,IM_WIDTH,3)).astype(np.float32)
        y=np.random.uniform(size=(1,3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y,verbose=False,batch_size=1)

        self.training_initialized=True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

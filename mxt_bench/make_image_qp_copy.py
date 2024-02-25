import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

from procedural_envs import composer
from procedural_envs.misc.observers import GraphObserver, EXTRA_ROOT_NODE_DICT
from procedural_envs.tasks.observation_config import obs_config_dict
# generate begavioral data (s, s', a, r, d, t) and the save state.qp.
import copy
import pickle
import pprint
import time
from typing import Callable, Optional
import numpy as np
from absl import app
from absl import flags
from absl import logging
from brax import envs
from brax.io import model
from brax.training import distribution
from brax.training import normalization
import flax
import jax
import jax.numpy as jnp
import pickle
from algo import ppo_mlp
from procedural_envs import composer
from procedural_envs.misc.observers import GraphObserver, EXTRA_ROOT_NODE_DICT
from procedural_envs.tasks.observation_config import obs_config_dict
from procedural_envs.tasks.task_config import TASK_CONFIG
import brax
from brax.io import image
from IPython.display import Image
import cv2
from PIL import Image as PImage


def createFolder(directory):
  try:
    if not os.path.exists(directory):
      os.makedirs(directory)
  except OSError:
    print ('Error: Creating directory.'+ directory)


FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'ant_reach_2', 'Name of environment to collect data.')
flags.DEFINE_string('obs_config', 'amorpheus', 'Name of observation config to train.')
# flags.DEFINE_integer('total_env_steps', 97850,  # 1957 steps * 2 device * 25 parallel envs.
#                      'Number of env steps to run training for.')

def qp_convert(s_qp, state_qp_shape):
    return brax.QP(
      pos=s_qp[:, 0:state_qp_shape * 3].reshape(-1, state_qp_shape, 3),
      rot=s_qp[:, state_qp_shape * 3:state_qp_shape * 7].reshape(-1, state_qp_shape, 4),
      vel=s_qp[:, state_qp_shape * 7:state_qp_shape * 10].reshape(-1, state_qp_shape, 3),
      ang=s_qp[:, state_qp_shape * 10:state_qp_shape * 13].reshape(-1, state_qp_shape, 3))

def qp_squeeze(s_qp):
    return brax.QP(
      pos=s_qp.pos[0],
      rot=s_qp.rot[0],
      vel=s_qp.vel[0],
      ang=s_qp.ang[0])

def main(unused_argv):
  # save dir

  environment_params = {
      'env_name': FLAGS.env,
      'obs_config': FLAGS.obs_config,
  }
  obs_config = obs_config_dict[FLAGS.obs_config]

  if ('handsup2' in FLAGS.env) and ('ant' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['ant_handsup2']
  elif ('handsup2' in FLAGS.env) and ('centipede' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['centipede_handsup2']
  elif ('handsup' in FLAGS.env) and ('unimal' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['unimal_handsup']
  elif 'handsup' in FLAGS.env:
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['handsup']

  env_config = copy.deepcopy(environment_params)
  observer = GraphObserver(name=FLAGS.obs_config, **obs_config)
  # create env
  env_fn = composer.create_fn(env_name=FLAGS.env, observer=observer, observer2=observer)
  env = env_fn()
  createFolder(f"/data1/bestgenius10/DATA/{FLAGS.env}")
  for seed_idx in range(100):
    saved_qp_0 = f"/data1/bestgenius10/mxt_bench/data/{FLAGS.env}_qp_{seed_idx}.pkl"
    with open(saved_qp_0, 'rb') as f:
        saved_qp_0 = pickle.load(f)
    local_state_size = env.observation_size // env.num_node
    observation_size = env.observation_size
    num_limb = observation_size // local_state_size
    action_size = env.action_size
    state_qp_shape = num_limb + 1
    converted_qp = []
    createFolder(f"/data1/bestgenius10/DATA/{FLAGS.env}/demo{seed_idx}")
    for k in range(500):
        if saved_qp_0[0, k, -1]==1.0:
           break
        converted_qp.append(qp_squeeze(qp_convert(saved_qp_0[0, k:k+1], state_qp_shape)))
        x = image.render(env.sys, [converted_qp[-1]], width=256, height=256, ssaa=2)
        encoded_img = np.fromstring(x, dtype = np.uint8)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image2 = PImage.fromarray(imgRGB).resize((84, 84))
        image2.save(f"/data1/bestgenius10/DATA/{FLAGS.env}/demo{seed_idx}/{k}.jpg")
    np.save(f"/data1/bestgenius10/DATA/{FLAGS.env}/demo{seed_idx}/qp", np.array(converted_qp))
    
if __name__ == '__main__':
  app.run(main)

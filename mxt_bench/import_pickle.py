import os
import sys
import subprocess                                                                                                                                                                                                                                                                                                           
nproc = int(sys.argv[1])                                                                                                                                                                                                                                                                                                    
pid = int(sys.argv[2]) 

envs = []
# for i in range(2, 8, 1):
#   envs += [f'centipede_reach_{i}']
for i in range(2, 7, 1):
  envs +=[f'ant_reach_{i}']
for i in range(2, 7, 1):
  envs += [f'ant_reach_handsup_{i}']

# for i in range(2, 7, 1):
#   envs += [f'ant_push_{i}']
# for i in range(3, 7, 1):
#   for j in range(i):
#     envs += [f'ant_push_{i}_b_{j}']

for i in range(2, 7, 1):
  envs += [f'ant_touch_{i}']

# for i in range(2, 8, 1):
#   envs += [f'centipede_push_{i}']

# for i in range(2, 8, 1):
#   envs += [f'centipede_reach_{i}']


# for i in range(2, 8, 1):
#   envs += [f'centipede_touch_{i}']

# for i in range(2, 7, 1):
#   envs += [f'claw_push_{i}']

for i in range(2, 7, 1):
  envs += [f'claw_reach_{i}']

for i in range(2, 7, 1):
  envs += [f'claw_reach_handsup_{i}']
  envs += [f'claw_reach_hard_handsup_{i}']

for i in range(2, 7, 1):
    envs += [f'claw_touch_{i}']

# for i in range(2, 8, 1):
#   envs += [f'worm_push_{i}']

# for i in range(2, 8, 1):
#   envs += [f'worm_touch_{i}']

for i in range(3, 7, 1):
  for j in range(i):
    envs += [f'ant_touch_{i}_b_{j}']

# for i in range(2, 8, 1):
#   for j in range(i):
#       for k in (4, 5):  # left or right
#         envs += [f'centipede_reach_{i}_b_{k}_{j}']
#         envs += [f'centipede_reach_{i}_b_{k}_{j}_all']

  # for j in range(i):
  #     for k in (4, 5):
  #       envs += [f'centipede_push_{i}_b_{k}_{j}']
  #       envs += [f'centipede_push_{i}_b_{k}_{j}_all']

for i in range(3, 7, 1):
  for j in range(i):
    envs += [f'ant_reach_{i}_b_{j}']
    envs += [f'ant_reach_hard_{i}_b_{j}']
  
for i in range(3, 7, 1):
  envs += [f'ant_reach_handsup2_{i}']
for i in range(3, 7, 1):
  envs += [f'ant_reach2_handsup_{i}']
cmds = []
cmds2 = []
for env in envs:
  cmds2 += [f"python make_image_qp.py --env {env}"]
# cmds = [f"python generate_behavior_and_qp.py --seed 0 --env {env} --task_name ant_reach --params_path ../results/ao_ppo_mlp_single_pro_ant_reach_4/ppo_mlp_298188800.pkl" for env in envs]
# cmds = [f"python train_ppo_mlp.py --logdir ../results --seed 0 --env {env} --total_env_steps 300000000 --episode_length 10000" for env in envs]

ith = pid
while ith < len(cmds2):
  print(f"{ith+1} / {len(cmds2)}")
  print(f"{cmds2[ith]}")
  subprocess.call(cmds2[ith].split(" "))
  ith += nproc
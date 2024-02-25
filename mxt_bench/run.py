import os
import sys
import subprocess                                                                                                                                                                                                                                                                                                           
nproc = int(sys.argv[1])                                                                                                                                                                                                                                                                                                    
pid = int(sys.argv[2]) 

envs = []
for i in range(2, 8, 1):
  envs += [f'centipede_reach_{i}']
for i in range(2, 7, 1):
  envs +=[f'ant_reach_{i}']
for i in range(2, 7, 1):
  envs += [f'ant_reach_handsup_{i}']
for i in range(3, 7, 1):
  envs += [f'ant_reach_handsup2_{i}']
for i in range(3, 7, 1):
  envs += [f'ant_reach2_handsup_{i}']
for i in range(2, 7, 1):
  envs += [f'ant_push_{i}']

for i in range(2, 7, 1):
  envs += [f'ant_touch_{i}']


for i in range(2, 8, 1):
  envs += [f'centipede_push_{i}']



for i in range(2, 8, 1):
  envs += [f'centipede_reach_{i}']


for i in range(2, 8, 1):
  envs += [f'centipede_touch_{i}']

for i in range(2, 7, 1):
  envs += [f'claw_push_{i}']

for i in range(2, 7, 1):
  envs += [f'claw_reach_{i}']

for i in range(2, 7, 1):
  envs += [f'claw_reach_handsup_{i}']
  envs += [f'claw_reach_hard_handsup_{i}']

for i in range(2, 7, 1):
    envs += [f'claw_touch_{i}']

for i in range(2, 8, 1):
  envs += [f'worm_push_{i}']

for i in range(2, 8, 1):
  envs += [f'worm_touch_{i}']

for i in range(3, 7, 1):
  for j in range(i):
    envs += [f'ant_push_{i}_b_{j}']
for i in range(3, 7, 1):
  for j in range(i):
    envs += [f'ant_reach_{i}_b_{j}']
    envs += [f'ant_reach_hard_{i}_b_{j}']

  
  
for i in range(2, 8, 1):
  for j in range(i):
      for k in (4, 5):  # left or right
        envs += [f'centipede_reach_{i}_b_{k}_{j}']
        envs += [f'centipede_reach_{i}_b_{k}_{j}_all']
  
for j in range(i):
    for k in (4, 5):
      envs += [f'centipede_push_{i}_b_{k}_{j}']
      envs += [f'centipede_push_{i}_b_{k}_{j}_all']

for i in range(3, 7, 1):
  for j in range(i):
    envs += [f'ant_touch_{i}_b_{j}']

cmds = [f"python train_ppo_mlp.py --logdir ../results --seed 0 --env {env}" for env in envs]

ith = pid
while ith < len(cmds):
  print(f"{ith+1} / {len(cmds)}")
  print(f"{cmds[ith]}")
  subprocess.call(cmds[ith].split(" "))
  ith += nproc
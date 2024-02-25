import functools
from procedural_envs.misc.observers import SimObserver
from procedural_envs.misc.random_init_functions import annulus_xy_sampler
from procedural_envs.misc.reward_functions import nearest_distance_reward


def load_desc(
    num_legs: int = 4,
    radius: float = 0.1 * 3,
    r_min: float = 7.5,
    r_max: float = 10.0,
    agent: str = 'claw',
    broken_id: int = 0,
    size_scales: list = []):
    random_init_fn = functools.partial(annulus_xy_sampler, r_min=r_min, r_max=r_max)
    component_params = dict(num_legs=num_legs)
    leg_indices = [i for i in range(num_legs)]
    if agent == 'broken_claw':
      component_params['broken_id'] = broken_id
      if broken_id in leg_indices:
        leg_indices.remove(broken_id)
    elif agent == 'size_rand_claw':
      component_params['size_scales'] = size_scales
    return dict(
        components=dict(
            agent1=dict(
                component=agent,
                component_params=component_params,
                pos=(0, 0, 0),
                reward_fns=dict(
                    distance=dict(
                      reward_type=nearest_distance_reward,
                      target=SimObserver(comp_name='cap1', sdname='Ball', indices=(0, 1)),
                      obs=[
                          SimObserver(comp_name='agent1', sdname=f'$ Body 4_{i}', indices=(0, 1)) for i in leg_indices],
                      min_dist=radius,
                      done_bonus=0.0)
                ),
            ),
          cap1=dict(
              component='ball',
              component_params=dict(
                  radius=radius,
                  frozen=True,
                  name="Ball"
                  ),
              pos=(0, 0, 0),
              random_init='pos',
              random_init_fn=random_init_fn,
          ),
        ),
        global_options=dict(dt=0.05, substeps=10),
        goal_based_task=True,
        task_edge=[
            ['cap1___Ball']+[f'agent1___$ Body 4_{i}' for i in leg_indices],
            [],
            [],
            ]
    )

ENV_DESCS = dict()

for i in range(2, 7, 1):
  ENV_DESCS[f'claw_reach_{i}'] = functools.partial(load_desc, num_legs=i)

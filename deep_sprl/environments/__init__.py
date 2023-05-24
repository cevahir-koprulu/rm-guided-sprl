from gym.envs.registration import register

register(
    id='ContextualTwoDoorDiscrete-v1',
    max_episode_steps=4800,
    entry_point='deep_sprl.environments.two_door.contextual_two_door_discrete:ContextualTwoDoorDiscrete'
)

register(
    id='ContextualTwoDoorDiscrete2D-v1',
    max_episode_steps=4800,
    entry_point='deep_sprl.environments.two_door.contextual_two_door_discrete_2d:ContextualTwoDoorDiscrete2D'
)

register(
    id='ContextualTwoDoorDiscrete4D-v1',
    max_episode_steps=4800,
    entry_point='deep_sprl.environments.two_door.contextual_two_door_discrete_4d:ContextualTwoDoorDiscrete4D'
)

register(
    id='ContextualTwoDoorDiscrete2DProduct-v1',
    max_episode_steps=4800,
    entry_point='deep_sprl.environments.two_door.contextual_two_door_discrete_2d_product:ContextualTwoDoorDiscrete2DProduct'
)

register(
    id='ContextualTwoDoorDiscrete4DProduct-v1',
    max_episode_steps=4800,
    entry_point='deep_sprl.environments.two_door.contextual_two_door_discrete_4d_product:ContextualTwoDoorDiscrete4DProduct'
)

register(
    id='ContextualHalfCheetah-v1',
    max_episode_steps=2000,
    entry_point='deep_sprl.environments.half_cheetah.contextual_half_cheetah:ContextualHalfCheetah'
)

register(
    id='ContextualHalfCheetah3D-v1',
    max_episode_steps=2000,
    entry_point='deep_sprl.environments.half_cheetah.contextual_half_cheetah_3d:ContextualHalfCheetah3D'
)

register(
    id='ContextualHalfCheetah3DProduct-v1',
    max_episode_steps=2000,
    entry_point='deep_sprl.environments.half_cheetah.contextual_half_cheetah_3d_product:ContextualHalfCheetah3DProduct'
)

# register(
#     id="ContextualFetchPushAndPlay4D-v1",
#     max_episode_steps=200,
#     entry_point="deep_sprl.environments.fetch_push_and_play.contextual_fetch_push_and_play_4d:ContextualFetchPushAndPlay4D",
# )

# register(
#     id="ContextualFetchPushAndPlay4DProduct-v1",
#     max_episode_steps=200,
#     entry_point="deep_sprl.environments.fetch_push_and_play.contextual_fetch_push_and_play_4d_product:ContextualFetchPushAndPlay4DProduct",
# )

register(
    id='ContextualSwimmer-v1',
    max_episode_steps=10000,
    entry_point='deep_sprl.environments.swimmer.contextual_swimmer:ContextualSwimmer'
)

register(
    id='ContextualSwimmer2D-v1',
    max_episode_steps=10000,
    entry_point='deep_sprl.environments.swimmer.contextual_swimmer_2d:ContextualSwimmer2D'
)

register(
    id='ContextualSwimmer2DProduct-v1',
    max_episode_steps=10000,
    entry_point='deep_sprl.environments.swimmer.contextual_swimmer_2d_product:ContextualSwimmer2DProduct'
)
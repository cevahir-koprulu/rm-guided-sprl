# Do this import to ensure that the Gym environments get registered properly
import deep_sprl.environments
from .two_door_discrete_2d_experiment import TwoDoorDiscrete2DExperiment
from .two_door_discrete_4d_experiment import TwoDoorDiscrete4DExperiment
from .half_cheetah_3d_experiment import HalfCheetah3DExperiment
from .fetch_push_and_play_4d_experiment import FetchPushAndPlay4DExperiment
from .swimmer_2d_experiment import Swimmer2DExperiment
from .abstract_experiment import CurriculumType, Learner

__all__ = ['CurriculumType', 'Learner','TwoDoorDiscrete2DExperiment', 
           'TwoDoorDiscrete4DExperiment', 'HalfCheetah3DExperiment', 
           'FetchPushAndPlay4DExperiment', 'Swimmer2DExperiment']
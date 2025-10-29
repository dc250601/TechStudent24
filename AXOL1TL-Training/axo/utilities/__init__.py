from axo.utilities.kwarg_handler import get_function_parameters as allowed_params
from axo.utilities.store import store_axo
from axo.utilities.retrieve import *
from axo.utilities.display import generate_axolotl_html_report
from axo.utilities.config import *
from axo.utilities.augmentation import *
from axo.utilities.fast_score import fast_score
from axo.utilities.lr_schedulers import *

CAWR = cosine_annealing_warm_restart_with_warmup
from .fisher import fisher, fisher_new
from .FT import FT, FT_l1
from .FT_prune import FT_prune
from .GA import GA, GA_l1
from .GA_prune import GA_prune
from .retrain import retrain
from .RL import RL
from .Wfisher import Wfisher
from .neggrad import negative_grad
from .RL_original import RL_og
from .scrub import scrub


def raw(data_loaders, model, criterion, args, mask=None):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "RL":
        return RL
    elif name == "RL_og":
        return RL_og
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "fisher":
        return fisher
    elif name == "retrain":
        return retrain
    elif name == "fisher_new":
        return fisher_new
    elif name == "wfisher":
        return Wfisher
    elif name == "FT_prune":
        return FT_prune
    elif name == "FT_prune_bi":
        return FT_prune_bi
    elif name == "GA_prune":
        return GA_prune
    elif name == "GA_prune_bi":
        return GA_prune_bi
    elif name == "GA_l1":
        return GA_l1
    elif name == "boundary_expanding":
        return boundary_expanding
    elif name == "boundary_shrink":
        return boundary_shrink
    elif name == 'NG':
        return negative_grad
    elif name == 'SCRUB':
        return scrub
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")

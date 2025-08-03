from ._base import Vanilla
from .KD import KD
from .MLKD import MLKD
from .AlignedMLKD import AlignedMLKD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .Sonly import Sonly
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .KD_KDA import KD as KD_KDA
from .MLKD_KDA import MLKD as MLKD_KDA


distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "KD_KDA": KD_KDA,
    "MLKD": MLKD,
    "MLKD_KDA": MLKD_KDA,
    "AlignedMLKD": AlignedMLKD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "Sonly": Sonly,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
}

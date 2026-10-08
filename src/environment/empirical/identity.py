"""Historical donor-gap matching under current lifecycle eligibility."""
import numpy as np


def eligible_gap_neighborhood(gaps, eligible, target, minimum=30):
    exact=eligible&(np.floor(gaps)==np.floor(target))
    if exact.sum()>=minimum:return exact,'EXACT_DAY'
    nearby=eligible&(np.abs(gaps-target)<=3)
    if nearby.sum()>=minimum:return nearby,'WITHIN_3_DAYS'
    if not eligible.any():return eligible.copy(),'NO_ELIGIBLE_CONTEXT'
    distances=np.abs(gaps[eligible]-target)
    radius=float(np.partition(distances,min(minimum-1,len(distances)-1))[min(minimum-1,len(distances)-1)])
    return eligible&(np.abs(gaps-target)<=radius),'NEAREST_SUPPORTED_RANGE'

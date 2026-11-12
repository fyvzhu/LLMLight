"""Opt-in SR scheduling and reproducible perturbations; original mode is untouched."""
from concurrent.futures import ThreadPoolExecutor
import random
import time

class SRCoordinator:
    def __init__(self,seed=42,stale_probability=0.0,record=lambda row:None):
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='SR')
        self.pending=None;self.history={};self.rng=random.Random(seed);self.probability=stale_probability;self.record=record
    def consume(self,step,ids):
        if self.pending and self.pending[0].done():
            future,origin,keys=self.pending;values=future.result()
            for key,value in zip(keys,values):self.history.setdefault(key,[]).append((origin,value))
            self.record({'event':'sr_completed','origin_step':origin,'observed_at_step':step,'finished_monotonic':time.perf_counter()})
            self.pending=None
        result=[]
        for key in ids:
            candidates=[item for item in self.history.get(key,[]) if item[0]<step]
            delayed=self.rng.random()<self.probability
            # An extra decision-step delay, not merely the previous completed result.
            allowed=[item for item in candidates if not delayed or item[0]<=step-2]
            chosen=allowed[-1] if allowed else None
            result.append(chosen[1] if chosen else '')
            self.record({'event':'sr_consumed','step':step,'intersection':key,'extra_stale':delayed,
                'origin_step':chosen[0] if chosen else None,'age_steps':step-chosen[0] if chosen else None,
                'cache_miss':chosen is None,'consumed_monotonic':time.perf_counter()})
        return result
    def submit(self,step,ids,function):
        if self.pending is None and ids:
            def completed():
                values=function()
                self.record({'event':'sr_finished','origin_step':step,'finished_monotonic':time.perf_counter()})
                return values
            self.pending=(self.executor.submit(completed),step,list(ids))
            self.record({'event':'sr_submitted','step':step,'intersections':list(ids),'submitted_monotonic':time.perf_counter()})
    def close(self):
        self.executor.shutdown(wait=True)
        if self.pending:self.pending[0].result()


def choose_failed_neighbor(neighbors,rng,probability):
    neighbors=sorted(set(neighbors),key=str)
    if not neighbors or rng.random()>=probability:return None
    return rng.choice(neighbors)


def neighbor_lane(key):
    """Parse the original framework's human-readable neighboring lane keys."""
    import re
    if isinstance(key,tuple) and len(key)==2:return key
    if isinstance(key,str):
        match=re.search(r"(?:upstream|downstream) lane \((\d+), ['\"]?([A-Z]{2})['\"]?\)",key)
        if match:return int(match.group(1)),match.group(2)
    return None

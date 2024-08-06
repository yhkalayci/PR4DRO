import pynndescent
import numpy as np
from queue import PriorityQueue
class RankFinder:
    def __init__(self, candidates, voters, prefetch_count=10):
        self.candidates = candidates
        self.voters = voters
        self.ndata = np.array(candidates)
        self.tree = pynndescent.NNDescent(self.ndata)
        self.tree.prepare()
        self.prefetch_count = prefetch_count
        idx_all, dist_all = self.tree.query(self.voters, k=prefetch_count)
        self.ranking = [[] for v in voters]
        for i, v in enumerate(self.voters):
            idx, dist = idx_all[i], dist_all[i]
            self.ranking[i] = [(idx[j], dist[j]) for j in range(len(idx))]
    def query(self, voter_idx, k):
        # print(voter_idx, k)
        count = k
        if count > len(self.ranking[voter_idx]):
            idx, dist = self.tree.query([self.voters[voter_idx]], k=len(self.ranking[voter_idx]) * 2)
            idx = idx[0]
            dist = dist[0]
            s = set([x[0] for x in self.ranking[voter_idx]])
            for j in range(len(idx)):
                if idx[j] not in s:
                    self.ranking[voter_idx].append((idx[j], dist[j]))
        return self.ranking[voter_idx][:count]
    
class TruncatedGreedyCapture:
    def __init__(self, candidates, voters, k, voter_weights=1):
        print("__init__")
        self.candidates = candidates
        self.voters = voters
        self.voter_size = len(voters)
        self.voter_weights = voter_weights
        self.k = min(k, len(candidates))
        self.ndata = np.array(candidates)
        print("ndata ready")
        self.tree = RankFinder(candidates, voters, prefetch_count=self.hare_quota*10)
        print("Tree ready")
        self.voter_expansion = None
        self.event_queue = PriorityQueue()
    @property
    def hare_quota(self):
        return (self.voter_weights * self.voter_size)//self.k
    def initialize(self):
        self.weight = [self.voter_weights for _ in self.voters]
        self.candidate_neighborhood = [[] for _ in self.candidates]
        self.voter_neighborhood = [[] for _ in self.voters]
        self.candidate_weights = [0 for _ in self.candidates]
        self.output = []
        print("query done")
        for i, v in enumerate(self.voters):
            idx, dist = self.tree.query(i, k=1)[-1]
            self.event_queue.put((dist, i, idx))
        print("finalize initialization")
    def process_event(self, event):
        dist, voter_idx, candidate_idx = event
        self.voter_neighborhood[voter_idx].append(candidate_idx)
        if self.candidate_weights[candidate_idx] >= 1:
            new_idx, new_dist = self.tree.query(voter_idx, k=len(self.voter_neighborhood[voter_idx])+1)[-1]
            self.event_queue.put((new_dist, voter_idx, new_idx))
            return
        self.candidate_neighborhood[candidate_idx].append(voter_idx)
        total_weight = 0
        for v in self.candidate_neighborhood[candidate_idx]:
            total_weight += min(self.weight[v], 1)
        # print(event, total_weight)
        if total_weight >= self.hare_quota:
            self.candidate_weights[candidate_idx] = 1
            self.output.append(candidate_idx)
            for v in self.candidate_neighborhood[candidate_idx]:
                if self.weight[v] > 0:
                    self.weight[v] -= 1
        if self.weight[voter_idx] > 0:
            new_idx, new_dist = self.tree.query(voter_idx, k=len(self.voter_neighborhood[voter_idx])+1)[-1]
            self.event_queue.put((new_dist, voter_idx, new_idx))
    def run(self, k_override=None, weight_override=None):
        if k_override is not None:
            self.k = k_override
        if weight_override is not None:
            self.voter_weights = weight_override
        self.initialize()
        cnt = 0
        while self.event_queue.qsize() > 0 and len(self.output) < self.k * 0.99:
            event = self.event_queue.get()
            self.process_event(event)
            # cnt += 1
            # if cnt % 1000 == 0:
            #     print(cnt, len(self.output), self.event_queue.qsize())
            # if cnt == 30:
            #     break
        return self.output
    
def get_subsample_indices(voters, candidates, sample_size):
    weight = int(sample_size/len(voters))+1
    # sample_size = 10
    tgc = TruncatedGreedyCapture(candidates, voters, sample_size, voter_weights=weight)
    output_indices = tgc.run()
    return output_indices


def get_subsample_indices_v2(voters, candidates, ratios):
    out = {}
    tgc = TruncatedGreedyCapture(candidates, voters, len(voters), voter_weights=1)
    for ratio in ratios:
        sample_size = int(len(candidates)*ratio)+1
        weight = int(sample_size/len(voters))+1
        output_indices = tgc.run(k_override=sample_size, weight_override=weight)
        out[ratio] = output_indices
    return out